from __future__ import annotations

import unittest
from datetime import datetime
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
        self.assertIn(
            "不动作", notifier.build_action("CASH", "588080", data_fresh=False)
        )

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
        self.assertIn("状态切换", report.rule_action)

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
