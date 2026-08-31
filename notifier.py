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
from datetime import datetime, time as wall_time
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
BASE_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = BASE_DIR / "data" / f"{SYMBOL}_qfq_snapshot.csv"
CLOSE_DATA_CHECK_TIME = wall_time(15, 5)
SCHEDULED_WINDOW_START_MINUTE = 15 * 60
SCHEDULED_WINDOW_END_MINUTE = 22 * 60 + 30

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
class PlainAction:
    code: str
    label: str
    explanation: str


@dataclass(frozen=True)
class HistoryResult:
    frame: pd.DataFrame
    source: str
    warning: str


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
    action_code: str
    action_label: str
    action_explanation: str
    rule_action: str
    rule_version: str
    rule_fingerprint: str
    data_source: str
    data_warning: str
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


def fetch_tencent_block(client: requests.Session, end_date: str) -> pd.DataFrame:
    response = client.get(
        TENCENT_URL,
        params={"param": f"sh{SYMBOL},day,1990-01-01,{end_date},640,qfq"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    node = (payload.get("data") or {}).get(f"sh{SYMBOL}") or {}
    rows = node.get("qfqday")
    if not rows:
        raise ValueError("未返回前复权 qfqday 数据，已拒绝改用其他口径")
    return normalize_rows(rows)


def fetch_tencent_history(session: requests.Session | None = None) -> pd.DataFrame:
    client = session or requests.Session()
    client.trust_env = False
    client.headers.update({"User-Agent": "588080-V3G-Signal/1.0"})

    parts: list[pd.DataFrame] = []
    end_date = "2050-01-01"
    previous_earliest: pd.Timestamp | None = None
    listing_floor = pd.Timestamp("2020-11-16")
    for _ in range(6):
        block = fetch_tencent_block(client, end_date)
        earliest = pd.Timestamp(block["date"].min())
        if previous_earliest is not None and earliest >= previous_earliest:
            break
        parts.append(block)
        if earliest <= listing_floor:
            break
        previous_earliest = earliest
        end_date = (earliest + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    combined = pd.concat(parts, ignore_index=True)
    duplicates = combined[combined.duplicated("date", keep=False)]
    if not duplicates.empty:
        spread = duplicates.groupby("date")["close"].agg(["min", "max"])
        relative_difference = ((spread["max"] / spread["min"]) - 1).abs()
        if float(relative_difference.max()) > 1e-8:
            raise ValueError("腾讯分页行情的复权口径不一致")
    columns = ["date", "open", "close", "high", "low", "volume"]
    return normalize_rows(combined[columns].values.tolist())


def fetch_baostock_history() -> pd.DataFrame:
    import baostock as bs

    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"Baostock 登录失败：{login_result.error_msg}")
    try:
        query = bs.query_history_k_data_plus(
            f"sh.{SYMBOL}",
            "date,open,high,low,close,volume",
            start_date="2020-11-16",
            end_date=datetime.now(TIMEZONE).date().isoformat(),
            frequency="d",
            adjustflag="2",
        )
        if query.error_code != "0":
            raise RuntimeError(f"Baostock 查询失败：{query.error_msg}")
        rows: list[list[str]] = []
        while query.next():
            row = query.get_row_data()
            # Baostock reports ETF volume in shares; Tencent uses 100-share lots.
            # Volume is not part of V3G, but normalizing keeps the snapshot coherent.
            volume_lots = str(float(row[5]) / 100)
            rows.append([row[0], row[1], row[4], row[2], row[3], volume_lots])
        return normalize_rows(rows)
    finally:
        bs.logout()


def load_snapshot(path: Path = SNAPSHOT_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = ["date", "open", "close", "high", "low", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"历史快照缺少字段：{','.join(missing)}")
    return normalize_rows(frame[required].values.tolist())


def merge_with_snapshot(snapshot: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    old = snapshot.set_index("date")
    new = recent.set_index("date")
    overlap = old.index.intersection(new.index)
    if len(overlap) < 20:
        raise ValueError("备用行情与历史快照重叠不足20个交易日")
    old_close = old.loc[overlap, "close"].astype(float)
    new_close = new.loc[overlap, "close"].astype(float)
    relative_difference = ((new_close / old_close) - 1).abs()
    if float(relative_difference.max()) > 0.002:
        raise ValueError("备用行情与历史快照复权口径不一致")
    combined = pd.concat([snapshot, recent], ignore_index=True)
    combined = combined.drop_duplicates("date", keep="last").sort_values("date")
    return combined.reset_index(drop=True)


def save_snapshot(frame: pd.DataFrame, path: Path = SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame[["date", "open", "close", "high", "low", "volume"]].copy()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output.to_csv(path, index=False, encoding="utf-8")


def latest_market_date(frame: pd.DataFrame):
    return pd.Timestamp(frame["date"].max()).date()


def fetch_history(
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> HistoryResult:
    current_time = now.astimezone(TIMEZONE) if now else datetime.now(TIMEZONE)
    errors: list[str] = []
    tencent_result: HistoryResult | None = None
    try:
        tencent_result = HistoryResult(
            frame=fetch_tencent_history(session),
            source="腾讯证券前复权日线 qfqday",
            warning="",
        )
    except Exception as exc:
        errors.append(f"腾讯：{type(exc).__name__}")

    if tencent_result is not None:
        tencent_date = latest_market_date(tencent_result.frame)
        should_check_backup = (
            current_time.time() >= CLOSE_DATA_CHECK_TIME
            and tencent_date < current_time.date()
        )
        if not should_check_backup:
            return tencent_result

    snapshot = load_snapshot()
    try:
        recent = fetch_baostock_history()
        if snapshot is None:
            raise RuntimeError("缺少历史快照，不能安全拼接备用行情")
        backup_frame = merge_with_snapshot(snapshot, recent)
        if (
            tencent_result is not None
            and latest_market_date(backup_frame)
            <= latest_market_date(tencent_result.frame)
        ):
            return HistoryResult(
                frame=tencent_result.frame,
                source=tencent_result.source,
                warning=(
                    f"截至{current_time.strftime('%H:%M')}，主备行情均尚未发布"
                    f"{current_time.date().isoformat()}收盘数据。"
                ),
            )
        warning = "腾讯源暂时不可用，已切换备用行情并核对复权口径。"
        if tencent_result is not None:
            warning = (
                f"腾讯日线停留在{latest_market_date(tencent_result.frame).isoformat()}，"
                "已自动采用日期更新的备用行情并核对复权口径。"
            )
        return HistoryResult(
            frame=backup_frame,
            source="Baostock前复权日线 + 已核对历史快照",
            warning=warning,
        )
    except Exception as exc:
        errors.append(f"Baostock：{type(exc).__name__}")

    if tencent_result is not None:
        return HistoryResult(
            frame=tencent_result.frame,
            source=tencent_result.source,
            warning=(
                f"腾讯日线停留在{latest_market_date(tencent_result.frame).isoformat()}，"
                "备用行情暂时不可用；系统只会显示观望。"
            ),
        )

    if snapshot is not None:
        return HistoryResult(
            frame=snapshot,
            source="仓库内已核对历史快照",
            warning=(
                "实时行情源暂时不可用，仅使用历史快照；如果日期未更新，"
                "系统只会显示观望。"
            ),
        )
    raise RuntimeError("所有行情来源均不可用：" + "；".join(errors))


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
) -> PlainAction:
    if not data_fresh:
        return PlainAction(
            code="WAIT",
            label="⚪ 数据延迟",
            explanation="今天的正式收盘数据尚未发布，暂不操作，等待自动补发。",
        )
    if previous_state == current_state == "CASH":
        return PlainAction(
            code="WAIT",
            label="🟡 观望",
            explanation="规则目前没有新的买入信号，暂不买入。",
        )
    if previous_state == current_state == SYMBOL:
        return PlainAction(
            code="HOLD",
            label="🔵 继续持有",
            explanation="规则仍支持持有588080，今天没有新的买卖变化。",
        )
    if previous_state == "CASH" and current_state == SYMBOL:
        return PlainAction(
            code="BUY",
            label="🟢 买入提醒",
            explanation=(
                "规则刚从观望转为持有；如你决定跟随，下一交易日开盘后人工确认，"
                "不要追盘中价格。"
            ),
        )
    if previous_state == SYMBOL and current_state == "CASH":
        return PlainAction(
            code="SELL",
            label="🔴 卖出提醒",
            explanation=(
                "规则刚从持有转为观望；如你决定跟随，下一交易日开盘后人工确认。"
            ),
        )
    raise ValueError(f"未知模型状态：{previous_state} -> {current_state}")


def build_report(
    result: pd.DataFrame,
    now: datetime | None = None,
    data_source: str = "测试行情",
    data_warning: str = "",
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
    action = build_action(previous_state, current_state, data_fresh)
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
        action_code=action.code,
        action_label=action.label,
        action_explanation=action.explanation,
        rule_action=action.explanation,
        rule_version=RULE_VERSION,
        rule_fingerprint=RULE_FINGERPRINT,
        data_source=data_source,
        data_warning=data_warning,
        boundary="规则信号提醒；不读取真实账户，不连接券商，不自动下单",
    )


def state_label(state: str) -> str:
    return "持有588080" if state == SYMBOL else "观望（规则未持有588080）"


def format_message(report: SignalReport) -> str:
    generated_date = report.generated_at[:10]
    if not report.data_fresh:
        lines = [
            f"【588080数据检查｜{generated_date}】",
            "当前状态：⚪ 收盘数据延迟",
            f"怎么做：{report.action_explanation}",
            "",
            (
                f"最后可用行情日期：{report.market_date}"
                f"（不是{generated_date}的正式信号）"
            ),
            f"数据来源：{report.data_source}",
        ]
        if report.data_warning:
            lines.append(f"数据说明：{report.data_warning}")
        lines.extend(
            [
                "",
                "系统取得今日收盘数据后会自动补发正式结果。",
                "数据延迟期间不会显示旧收盘价或旧投票，也不会发出买卖提醒。",
                "",
                "本项目只发通知，不读取真实持仓、不连接券商、不会自动下单。",
            ]
        )
        return "\n".join(lines)

    lines = [
        f"【588080每日提醒｜{report.market_date}】",
        f"今日结论：{report.action_label}",
        f"怎么做：{report.action_explanation}",
        "",
        f"{SYMBOL_NAME}（{SYMBOL}）收盘：{report.close:.3f}",
        f"投票：{report.votes}/{report.total_votes}（10票买入，6票卖出）",
        f"规则状态：{state_label(report.model_state)}",
        f"行情日期：{report.market_date}（已更新）",
        f"数据来源：{report.data_source}",
    ]
    if report.data_warning:
        lines.append(f"数据说明：{report.data_warning}")
    lines.extend(
        [
            "",
            "简单说明：",
            "• 买入提醒＝规则今天首次转为持有。",
            "• 卖出提醒＝规则今天首次转为观望。",
            "• 继续持有＝没有新的卖出变化。",
            "• 观望＝没有新的买入变化，不等于今天新发出卖出提醒。",
            "",
            "本项目只发通知，不读取真实持仓、不连接券商、不会自动下单。",
            "已有旧持仓时，请勿仅凭一条普通“观望”消息临时处理。",
        ]
    )
    return "\n".join(lines)


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


def load_existing_report(output_dir: Path, generated_date: str) -> dict | None:
    path = output_dir / f"{generated_date}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def should_send_notification(
    existing: dict | None, report: SignalReport
) -> bool:
    if existing is None:
        return True
    fields = (
        "market_date",
        "close",
        "votes",
        "model_state",
        "action_code",
        "data_fresh",
    )
    current = asdict(report)
    return any(existing.get(field) != current.get(field) for field in fields)


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


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def should_skip_delayed_schedule(
    now: datetime | None = None,
    event_name: str | None = None,
) -> bool:
    current_time = now.astimezone(TIMEZONE) if now else datetime.now(TIMEZONE)
    current_event = (
        event_name if event_name is not None else os.environ.get("GITHUB_EVENT_NAME", "")
    )
    if current_event != "schedule":
        return False
    minute_of_day = current_time.hour * 60 + current_time.minute
    return (
        current_time.weekday() >= 5
        or minute_of_day < SCHEDULED_WINDOW_START_MINUTE
        or minute_of_day > SCHEDULED_WINDOW_END_MINUTE
    )


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    try:
        if should_skip_delayed_schedule():
            print(
                "[完成] GitHub定时任务抵达时间已超出工作日收盘检查窗口；"
                "已跳过，避免把延迟任务误报为新交易日信号。"
            )
            return 0
        validate_frozen_rule()
        history = fetch_history()
        save_snapshot(history.frame)
        result = compute_v3g(history.frame)
        report = build_report(
            result,
            data_source=history.source,
            data_warning=history.warning,
        )
        message = format_message(report)
        output_dir = Path(args.output_dir)
        existing_report = load_existing_report(
            output_dir, report.generated_at[:10]
        )
        notification_needed = should_send_notification(existing_report, report)
        write_report(report, message, output_dir)
        print(message)

        webhook = (
            os.environ.get("FEISHU_WEBHOOK", "").strip()
            or os.environ.get("FEISHU_WEBHOOK_URL", "").strip()
        )
        secret = (
            os.environ.get("FEISHU_SECRET", "").strip()
            or os.environ.get("FEISHU_WEBHOOK_SECRET", "").strip()
            or None
        )
        if args.dry_run:
            print("\n[完成] dry-run：已生成报告，未发送消息。")
        elif not webhook:
            print("\n[完成] 未配置飞书 Webhook；已生成报告，未发送消息。")
        elif not notification_needed:
            print("\n[完成] 今日同一收盘结果已通知，本次不重复发送。")
        else:
            send_feishu(webhook, secret, message)
            print("\n[完成] 飞书消息发送成功。")
        return 0
    except Exception as exc:
        print(f"[失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
