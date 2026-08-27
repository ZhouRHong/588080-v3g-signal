from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

import notifier


class V3GNotifierTests(unittest.TestCase):
    def test_frozen_rule_fingerprint(self) -> None:
        notifier.validate_frozen_rule()
        self.assertEqual(
            notifier.RULE_FINGERPRINT, notifier.EXPECTED_RULE_FINGERPRINT
        )

    def test_hysteresis_retains_state_between_thresholds(self) -> None:
        previous = "588080"
        current = previous
        for votes in [9, 8, 7]:
            if current == "CASH" and votes >= notifier.ENTRY_VOTES:
                current = "588080"
            elif current == "588080" and votes <= notifier.EXIT_VOTES:
                current = "CASH"
        self.assertEqual(current, "588080")

    def test_stale_data_never_proposes_rule_trade(self) -> None:
        action = notifier.build_action("CASH", "588080", data_fresh=False)
        self.assertEqual(action.code, "WAIT")
        self.assertIn("数据延迟", action.label)

    def test_stale_message_does_not_show_old_vote_as_today_signal(self) -> None:
        index = pd.to_datetime(["2026-08-24", "2026-08-25"])
        frame = pd.DataFrame(
            {
                "close": [1.641, 1.643],
                "votes": [0, 0],
                "state": ["CASH", "CASH"],
            },
            index=index,
        )
        now = datetime(2026, 8, 26, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        message = notifier.format_message(notifier.build_report(frame, now=now))
        self.assertIn("【588080数据检查｜2026-08-26】", message)
        self.assertIn("收盘数据延迟", message)
        self.assertIn("不是2026-08-26的正式信号", message)
        self.assertNotIn("投票：", message)
        self.assertNotIn("收盘：1.643", message)

    def test_four_plain_language_actions(self) -> None:
        cases = [
            ("CASH", "CASH", "WAIT", "观望"),
            ("588080", "588080", "HOLD", "继续持有"),
            ("CASH", "588080", "BUY", "买入提醒"),
            ("588080", "CASH", "SELL", "卖出提醒"),
        ]
        for previous, current, code, label in cases:
            with self.subTest(previous=previous, current=current):
                action = notifier.build_action(previous, current, data_fresh=True)
                self.assertEqual(action.code, code)
                self.assertIn(label, action.label)

    def test_report_marks_state_change(self) -> None:
        index = pd.to_datetime(["2026-08-21", "2026-08-24"])
        frame = pd.DataFrame(
            {
                "close": [1.6, 1.7],
                "votes": [9, 10],
                "state": ["CASH", "588080"],
            },
            index=index,
        )
        now = datetime(2026, 8, 24, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        report = notifier.build_report(frame, now=now)
        self.assertTrue(report.data_fresh)
        self.assertTrue(report.state_changed)
        self.assertEqual(report.action_code, "BUY")
        self.assertIn("买入提醒", report.action_label)

    def test_message_hides_cash_jargon(self) -> None:
        index = pd.to_datetime(["2026-08-21", "2026-08-24"])
        frame = pd.DataFrame(
            {
                "close": [1.7, 1.6],
                "votes": [0, 0],
                "state": ["CASH", "CASH"],
            },
            index=index,
        )
        now = datetime(2026, 8, 24, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        message = notifier.format_message(notifier.build_report(frame, now=now))
        self.assertNotIn("CASH", message)
        self.assertIn("今日结论：🟡 观望", message)
        self.assertIn("不等于今天新发出卖出提醒", message)

    def test_backup_history_merges_only_after_overlap_check(self) -> None:
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        snapshot = pd.DataFrame(
            {
                "date": dates,
                "open": 1.0,
                "close": 1.0,
                "high": 1.1,
                "low": 0.9,
                "volume": 100.0,
            }
        )
        recent = snapshot.iloc[-20:].copy()
        recent.loc[recent.index[-1], "close"] = 1.001
        merged = notifier.merge_with_snapshot(snapshot, recent)
        self.assertEqual(len(merged), 30)
        self.assertAlmostEqual(float(merged.iloc[-1]["close"]), 1.001)

        inconsistent = recent.copy()
        inconsistent["close"] = 1.01
        with self.assertRaises(ValueError):
            notifier.merge_with_snapshot(snapshot, inconsistent)

    def test_stale_tencent_uses_fresher_backup_after_close(self) -> None:
        columns = ["date", "open", "close", "high", "low", "volume"]
        tencent = pd.DataFrame(
            [["2026-08-24", 1.69, 1.641, 1.70, 1.61, 100]],
            columns=columns,
        )
        backup = pd.DataFrame(
            [["2026-08-25", 1.614, 1.643, 1.661, 1.602, 200]],
            columns=columns,
        )
        now = datetime(2026, 8, 25, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        with (
            patch.object(notifier, "fetch_tencent_history", return_value=tencent),
            patch.object(notifier, "load_snapshot", return_value=tencent),
            patch.object(notifier, "fetch_baostock_history", return_value=backup),
            patch.object(notifier, "merge_with_snapshot", return_value=backup),
        ):
            history = notifier.fetch_history(now=now)
        self.assertEqual(
            notifier.latest_market_date(history.frame).isoformat(), "2026-08-25"
        )
        self.assertIn("Baostock", history.source)
        self.assertIn("停留在2026-08-24", history.warning)

    def test_same_daily_result_is_not_notified_twice(self) -> None:
        index = pd.to_datetime(["2026-08-24", "2026-08-25"])
        result = pd.DataFrame(
            {
                "close": [1.641, 1.643],
                "votes": [0, 0],
                "state": ["CASH", "CASH"],
            },
            index=index,
        )
        now = datetime(2026, 8, 25, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        report = notifier.build_report(result, now=now)
        self.assertFalse(notifier.should_send_notification(vars(report), report))
        stale = dict(vars(report), market_date="2026-08-24", data_fresh=False)
        self.assertTrue(notifier.should_send_notification(stale, report))

    def test_feishu_signature_is_deterministic(self) -> None:
        first = notifier.feishu_payload("hello", "secret", timestamp=1234567890)
        second = notifier.feishu_payload("hello", "secret", timestamp=1234567890)
        self.assertEqual(first, second)
        self.assertIn("sign", first)

    def test_rejects_non_official_webhook(self) -> None:
        with self.assertRaises(ValueError):
            notifier.validate_feishu_webhook("https://example.com/hook/test")


if __name__ == "__main__":
    unittest.main()
