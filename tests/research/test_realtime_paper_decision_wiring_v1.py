from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "research" / "realtime_paper_decision_wiring_v1.py"
CONFIG_PATH = ROOT / "config" / "realtime_paper_decision_wiring_v1.json"
FIXTURE_PATH = (
    ROOT
    / "research"
    / "drafts"
    / "EXP-20260801-002-realtime-paper-decision-wiring-v1.fold_manifest.json"
)

SPEC = importlib.util.spec_from_file_location("realtime_paper_decision_wiring_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RealtimePaperDecisionWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MODULE.load_config(CONFIG_PATH)
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.zone = ZoneInfo("Asia/Tokyo")

    def inputs(self, case_id: str, robust_er: float = 1.7):
        case = next(row for row in self.fixture["cases"] if row["case_id"] == case_id)
        decision_time = datetime.fromisoformat(case["decision_time"])
        candidate = {
            "race_id": "2026080201010101",
            "ticket_type": "wide",
            "candidate_pair_key": "01-02",
        }
        candidate["candidate_freeze_record_hash"] = MODULE.candidate_digest(candidate)
        quote_received = case.get("quote_received_at")
        quote = {
            "race_id": candidate["race_id"],
            "ticket_type": candidate["ticket_type"],
            "candidate_pair_key": candidate["candidate_pair_key"],
            "quote_source_event_time": quote_received,
            "quote_received_at": quote_received,
            "quote_contract_ok": True,
            "market_status": "OPEN",
        }
        value_record = {
            "candidate_freeze_record_hash": candidate["candidate_freeze_record_hash"],
            "decision_generated_at": decision_time.isoformat(),
            "robust_expected_return": robust_er,
            "starter_universe_unchanged": True,
        }
        schedule = {"post_time": case["post_time"]}
        return candidate, quote, value_record, schedule, decision_time

    def evaluate(self, case_id: str, robust_er: float = 1.7):
        return MODULE.evaluate_paper_decision(
            *self.inputs(case_id, robust_er), self.config
        )

    def test_registered_fresh_case_is_paper_ready(self) -> None:
        result = self.evaluate("fresh_quote_normal_window", 1.7)
        self.assertEqual(result.status, "PAPER_READY")
        self.assertEqual(result.trigger, "NORMAL_WINDOW")
        self.assertFalse(result.formal_buy)
        self.assertFalse(result.send_order)
        self.assertEqual(result.paper_stake_yen, 0)

    def test_registered_stale_quote_fails_closed(self) -> None:
        result = self.evaluate("stale_quote_fail_close", 1.7)
        self.assertEqual(result.status, "NO_BET")
        self.assertEqual(result.reason, "QUOTE_STALE")

    def test_registered_missing_quote_fails_closed(self) -> None:
        result = self.evaluate("missing_quote_fail_close", 1.7)
        self.assertEqual(result.status, "NO_BET")
        self.assertEqual(result.reason, "QUOTE_TIME_MISSING")

    def test_registered_late_decision_fails_closed(self) -> None:
        result = self.evaluate("late_decision_fail_close", 1.7)
        self.assertEqual(result.status, "NO_BET")
        self.assertEqual(result.reason, "PAPER_DECISION_DEADLINE_MISSED")

    def test_registered_candidate_hash_change_fails_closed(self) -> None:
        candidate, quote, value, schedule, now = self.inputs(
            "candidate_hash_change_fail_close"
        )
        candidate["candidate_freeze_record_hash"] = "0" * 64
        result = MODULE.evaluate_paper_decision(
            candidate, quote, value, schedule, now, self.config
        )
        self.assertEqual(result.status, "NO_BET")
        self.assertEqual(result.reason, "CANDIDATE_HASH_MISMATCH")

    def test_value_must_be_generated_after_quote(self) -> None:
        candidate, quote, value, schedule, now = self.inputs(
            "fresh_quote_normal_window"
        )
        value["decision_generated_at"] = (
            datetime.fromisoformat(quote["quote_received_at"]) - timedelta(seconds=1)
        ).isoformat()
        result = MODULE.evaluate_paper_decision(
            candidate, quote, value, schedule, now, self.config
        )
        self.assertEqual(result.status, "NO_BET")
        self.assertEqual(result.reason, "SOURCE_TIME_ORDER_VIOLATION")

    def test_wrong_pair_fails_closed(self) -> None:
        candidate, quote, value, schedule, now = self.inputs(
            "fresh_quote_normal_window"
        )
        quote["candidate_pair_key"] = "01-03"
        result = MODULE.evaluate_paper_decision(
            candidate, quote, value, schedule, now, self.config
        )
        self.assertEqual(result.status, "NO_BET")
        self.assertEqual(result.reason, "QUOTE_CANDIDATE_PAIR_MISMATCH")

    def test_low_value_waits_without_candidate_change(self) -> None:
        candidate, quote, value, schedule, now = self.inputs(
            "fresh_quote_normal_window", robust_er=1.49
        )
        original_hash = candidate["candidate_freeze_record_hash"]
        result = MODULE.evaluate_paper_decision(
            candidate, quote, value, schedule, now, self.config
        )
        self.assertEqual(result.status, "WAIT")
        self.assertEqual(result.reason, "ROBUST_EXPECTED_RETURN_BELOW_THRESHOLD")
        self.assertEqual(result.candidate_freeze_record_hash, original_hash)

    def test_duplicate_cycle_appends_one_paper_event(self) -> None:
        decision = self.evaluate("duplicate_cycle_idempotent", 1.7)
        candidate, quote, value, schedule, now = self.inputs(
            "duplicate_cycle_idempotent", 1.7
        )
        later = now + timedelta(seconds=10)
        quote["quote_source_event_time"] = later.isoformat()
        quote["quote_received_at"] = later.isoformat()
        value["decision_generated_at"] = later.isoformat()
        repeated_decision = MODULE.evaluate_paper_decision(
            candidate, quote, value, schedule, later, self.config
        )
        records: list[dict] = []
        self.assertTrue(MODULE.append_idempotent(records, decision))
        self.assertFalse(MODULE.append_idempotent(records, repeated_decision))
        self.assertEqual(len(records), 1)
        MODULE.verify_output_safety(records)

    def test_candidate_payload_rejects_market_fields(self) -> None:
        candidate, _, _, _, _ = self.inputs("fresh_quote_normal_window")
        candidate["live_odds"] = 4.2
        with self.assertRaisesRegex(ValueError, "market or result"):
            MODULE.candidate_digest(candidate)

    def test_unsafe_config_is_rejected(self) -> None:
        unsafe = deepcopy(self.config)
        unsafe["safety"]["send_order"] = True
        with self.assertRaisesRegex(ValueError, "send_order"):
            temporary = ROOT / "research" / "drafts" / ".unsafe_config_fixture.json"
            try:
                temporary.write_text(json.dumps(unsafe), encoding="utf-8")
                MODULE.load_config(temporary)
            finally:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
