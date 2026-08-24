"""588080 V3G daily rule-signal notifier.

This program downloads public adjusted daily prices, evaluates the frozen V3G
rule and optionally sends the result to a Feishu custom bot. It never connects
to a broker and never places an order.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests


SYMBOL = "588080"
SYMBOL_NAME = "科创50ETF易方达"
RULE_VERSION = "V3G-20260824-ensemble-v1"
MA_PERIODS = (40, 50, 60, 80, 100)
CONFIRMATION_DAYS = (1, 2, 3)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ENTRY_VOTES = 10
EXIT_VOTES = 6
TOTAL_VOTES = len(MA_PERIODS) * len(CONFIRMATION_DAYS)
TIMEZONE = ZoneInfo("Asia/Shanghai")
TENCENT_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

# Signal-only configuration: deliberately excludes account capital, holdings,
# costs and other personal/account-specific values.
RULE_CONFIG = {
    "rule_version": RULE_VERSION,
    "symbol": SYMBOL,
    "ma_periods": list(MA_PERIODS),
    "macd": [MACD_FAST, MACD_SLOW, MACD_SIGNAL],
    "confirmation_days": list(CONFIRMATION_DAYS),
    "entry_votes": ENTRY_VOTES,
    "exit_votes": EXIT_VOTES,
    "execution": "close signal, next trading day open rule reminder",
}
RULE_FINGERPRINT = hashlib.sha256(
    json.dumps(RULE_CONFIG, ensure_ascii=False, sort_keys=True).encode("utf-8")
).hexdigest()
EXPECTED_RULE_FINGERPRINT = (
    "daf0be01dad72917ecb41fbd94f7d9775570b0bdaa4c4d7ddc5d1dbd38ee9292"
)


@dataclass(frozen=True)
class SignalReport:
    generated_at: str
    market_date: str
    close: float
    votes: int
    total_votes: int
    model_state: str
    previous_model_state: str
    state_changed: bool
    data_fresh: bool
    rule_action: str
    rule_version: str
    rule_fingerprint: str
    data_source: str
    boundary: str


def validate_frozen_rule() -> None:
    if RULE_FINGERPRINT != EXPECTED_RULE_FINGERPRINT:
        raise RuntimeError(
            "冻结规则指纹不匹配，程序已停止。请不要在同一版本中修改参数。"
        )


def normalize_rows(rows: list[list[object]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("行情接口返回空数据")
    frame = pd.DataFrame(rows).iloc[:, :6].copy()
    frame.columns = ["date", "open", "close", "high", "low", "volume"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "close", "high", "low", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().drop_duplicates("date", keep="last").sort_values("date")
    if len(frame) < max(MA_PERIODS) + MACD_SLOW:
        raise ValueError(f"有效行情仅 {len(frame)} 条，不足以计算冻结规则")
    bad_price = (frame[["open", "close", "high", "low"]] <= 0).any(axis=1)
    if bad_price.any():
        raise ValueError("行情中出现非正价格，程序已停止")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError("行情 OHLC 关系异常，程序已停止")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError("行情 OHLC 关系异常，程序已停止")
    return frame.reset_index(drop=True)


def fetch_history(session: requests.Session | None = None) -> pd.DataFrame:
    client = session or requests.Session()
    client.headers.update({"User-Agent": "588080-V3G-Signal/1.0"})
    response = client.get(
        TENCENT_URL,
        params={"param": f"sh{SYMBOL},day,1990-01-01,2050-01-01,640,qfq"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    node = (payload.get("data") or {}).get(f"sh{SYMBOL}") or {}
    rows = node.get("qfqday")
    if not rows:
        raise ValueError("未返回前复权 qfqday 数据，已拒绝改用其他口径")
    return normalize_rows(rows)


def compute_v3g(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().set_index("date")
    close = data["close"]
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_positive = dif > dea

    vote_columns: dict[str, pd.Series] = {}
    for ma_period in MA_PERIODS:
        moving_average = close.rolling(ma_period).mean()
        for days in CONFIRMATION_DAYS:
            confirmed = macd_positive.rolling(days).sum().eq(days)
            vote_columns[f"ma{ma_period}_confirm{days}"] = (
                confirmed & close.gt(moving_average)
            ).astype(int)

    result = pd.DataFrame(index=data.index)
    result["close"] = close
    result["dif"] = dif
    result["dea"] = dea
    result["votes"] = pd.DataFrame(vote_columns, index=data.index).sum(axis=1)

    state = "CASH"
    states: list[str] = []
    for votes in result["votes"].astype(int):
        if state == "CASH" and votes >= ENTRY_VOTES:
            state = SYMBOL
        elif state == SYMBOL and votes <= EXIT_VOTES:
            state = "CASH"
        states.append(state)
    result["state"] = states
    return result


def build_action(
    previous_state: str, current_state: str, data_fresh: bool
) -> str:
    if not data_fresh:
        return "数据尚未更新：规则不动作，等待下一个有效交易日收盘"
    if previous_state == current_state == "CASH":
        return "下一交易日继续空仓等待"
    if previous_state == current_state == SYMBOL:
        return "下一交易日继续保持模型持仓"
    if previous_state == "CASH" and current_state == SYMBOL:
        return "状态切换：下一交易日开盘按规则转为模型持仓"
    if previous_state == SYMBOL and current_state == "CASH":
        return "状态切换：下一交易日开盘按规则转为空仓"
    raise ValueError(f"未知模型状态：{previous_state} -> {current_state}")


def build_report(
    result: pd.DataFrame, now: datetime | None = None
) -> SignalReport:
    if len(result) < 2:
        raise ValueError("至少需要两个交易日才能判断状态是否切换")
    current_time = now.astimezone(TIMEZONE) if now else datetime.now(TIMEZONE)
    latest = result.iloc[-1]
    previous = result.iloc[-2]
    market_date = pd.Timestamp(result.index[-1]).date()
    if market_date > current_time.date():
        raise ValueError("行情日期晚于当前日期，程序已停止")
    data_fresh = market_date == current_time.date()
    previous_state = str(previous["state"])
    current_state = str(latest["state"])
    return SignalReport(
        generated_at=current_time.isoformat(timespec="seconds"),
        market_date=market_date.isoformat(),
        close=round(float(latest["close"]), 4),
        votes=int(latest["votes"]),
        total_votes=TOTAL_VOTES,
        model_state=current_state,
        previous_model_state=previous_state,
        state_changed=previous_state != current_state,
        data_fresh=data_fresh,
        rule_action=build_action(previous_state, current_state, data_fresh),
        rule_version=RULE_VERSION,
        rule_fingerprint=RULE_FINGERPRINT,
        data_source="腾讯证券前复权日线 qfqday",
        boundary="规则信号提醒；不读取真实账户，不连接券商，不自动下单",
    )


def state_label(state: str) -> str:
    return "模型持仓 588080" if state == SYMBOL else "模型空仓 CASH"


def format_message(report: SignalReport) -> str:
    freshness = "已更新" if report.data_fresh else "未更新"
    changed = "是" if report.state_changed else "否"
    return "\n".join(
        [
            f"【V3G规则信号｜{report.market_date}】",
            f"{SYMBOL_NAME}（{SYMBOL}）收盘：{report.close:.3f}",
            f"投票：{report.votes}/{report.total_votes}（10票入场，6票离场）",
            f"模型状态：{state_label(report.model_state)}",
            f"状态切换：{changed}",
            f"行情日期：{report.market_date}（{freshness}）",
            f"规则提示：{report.rule_action}",
            "说明：这是冻结规则的机械提醒，不读取你的实际持仓；不构成收益保证或个性化投资建议。",
            f"规则版本：{report.rule_version}",
        ]
    )


def validate_feishu_webhook(webhook: str) -> None:
    parsed = urlparse(webhook)
    allowed_hosts = {"open.feishu.cn", "open.larksuite.com"}
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError("飞书 Webhook 必须使用官方 HTTPS 地址")
    if "/open-apis/bot/v2/hook/" not in parsed.path:
        raise ValueError("飞书 Webhook 路径格式不正确")


def feishu_payload(text: str, secret: str | None, timestamp: int | None = None) -> dict:
    payload: dict[str, object] = {
        "msg_type": "text",
        "content": {"text": text},
    }
    if secret:
        current_timestamp = int(time.time()) if timestamp is None else timestamp
        string_to_sign = f"{current_timestamp}\n{secret}"
        digest = hmac.new(
            string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
        payload["timestamp"] = str(current_timestamp)
        payload["sign"] = base64.b64encode(digest).decode("utf-8")
    return payload


def send_feishu(webhook: str, secret: str | None, text: str) -> None:
    validate_feishu_webhook(webhook)
    response = requests.post(
        webhook,
        json=feishu_payload(text, secret),
        timeout=20,
    )
    response.raise_for_status()
    body = response.json()
    code = body.get("code", body.get("StatusCode", 0))
    if code not in (0, "0", None):
        message = body.get("msg", body.get("StatusMessage", "unknown error"))
        raise RuntimeError(f"飞书机器人拒绝消息：{message}")


def write_report(report: SignalReport, text: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = report.generated_at[:10]
    json_path = output_dir / f"{filename}.json"
    text_path = output_dir / f"{filename}.txt"
    json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    text_path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="588080 V3G 每日规则信号提醒")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只抓取、计算和生成报告，不发送飞书消息",
    )
    parser.add_argument(
        "--output-dir", default="reports", help="报告保存目录（默认 reports）"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_frozen_rule()
        history = fetch_history()
        result = compute_v3g(history)
        report = build_report(result)
        message = format_message(report)
        write_report(report, message, Path(args.output_dir))
        print(message)

        webhook = os.environ.get("FEISHU_WEBHOOK", "").strip()
        secret = os.environ.get("FEISHU_SECRET", "").strip() or None
        if args.dry_run:
            print("\n[完成] dry-run：已生成报告，未发送消息。")
        elif not webhook:
            print("\n[完成] 未配置 FEISHU_WEBHOOK；已生成报告，未发送消息。")
        else:
            send_feishu(webhook, secret, message)
            print("\n[完成] 飞书消息发送成功。")
        return 0
    except Exception as exc:
        print(f"[失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
