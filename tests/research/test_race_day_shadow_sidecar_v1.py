from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SIDECAR_PATH = ROOT / "scripts" / "research" / "race_day_shadow_sidecar_v1.py"
RENDERER_PATH = (
    ROOT / "scripts" / "research" / "render_race_day_shadow_sidecar_v1.py"
)
CONFIG_PATH = ROOT / "config" / "race_day_shadow_sidecar_v1.json"
FIXTURE_PATH = (
    ROOT
    / "research"
    / "drafts"
    / "EXP-20260801-003-race-day-shadow-sidecar-v1.fold_manifest.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SIDECAR = load_module("race_day_shadow_sidecar_v1", SIDECAR_PATH)
RENDERER = load_module("render_race_day_shadow_sidecar_v1", RENDERER_PATH)


class RaceDayShadowSidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sidecar_config = SIDECAR.load_sidecar_config(CONFIG_PATH)
        cls.coordinator_config = SIDECAR.coordinator_config_for(cls.sidecar_config)
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def case(self, case_id: str) -> dict:
        return next(row for row in self.fixture["cases"] if row["case_id"] == case_id)

    @staticmethod
    def odds_html(source_time: str | None, pair_key: str = "1-2") -> str:
        source_label = ""
        if source_time:
            parsed = datetime.fromisoformat(source_time)
            source_label = f"{parsed.hour}時{parsed.minute:02d}分現在オッズ"
        first, second = pair_key.split("-")
        return f"""<!doctype html>
<html lang="ja"><body>
  <div class="time">{source_label}</div>
  <table class="odds wide">
    <caption>{first}</caption>
    <tbody><tr><th>{second}</th><td><span class="min">4.2</span> - <span class="max">4.8</span></td></tr></tbody>
  </table>
</body></html>"""

    def inputs(self, case_id: str) -> dict:
        case = self.case(case_id)
        pair_key = "1-2"
        candidate_identity = {
            "race_id": "2026080201010101",
            "ticket_type": "wide",
            "candidate_pair_key": pair_key,
        }
        candidate_hash = SIDECAR.candidate_digest(candidate_identity)
        candidate = {
            **candidate_identity,
            "candidate_freeze_record_hash": candidate_hash,
            "candidate_freeze_contract_ok": case["candidate_freeze_contract_ok"],
            "candidate_uses_odds": False,
            "candidate_freeze_persist_ack_at": case[
                "candidate_freeze_persist_ack_at"
            ],
            "starter_universe_hash_at_freeze": "universe-hash-001",
        }
        schedule = {
            "scheduled_post_time": case["scheduled_post_time"],
            "schedule_record_hash": (
                "schedule-hash-001" if case["schedule_contract_ok"] else ""
            ),
            "schedule_contract_ok": case["schedule_contract_ok"],
        }
        universe = {
            "race_id": candidate_identity["race_id"],
            "ticket_type": "wide",
            "quote_pair_key": pair_key,
            "starter_universe_hash_at_quote": (
                "universe-hash-001"
                if case["starter_universe_unchanged"]
                else "universe-hash-changed"
            ),
            "starter_universe_unchanged": case["starter_universe_unchanged"],
            "market_status": "OPEN",
        }
        received = datetime.fromisoformat(case["quote_received_at"])
        source_time = case.get("quote_source_event_time")
        html_pair = "1-2" if case["quote_pair_matches"] else "1-3"
        return {
            "candidate_row": candidate,
            "schedule_record": schedule,
            "universe_observation": universe,
            "odds_html": self.odds_html(source_time, html_pair),
            "odds_join_started_at": case["odds_join_started_at"],
            "quote_request_started_at": (
                datetime.fromisoformat(case["odds_join_started_at"])
                + timedelta(seconds=2)
            ).isoformat(),
            "quote_received_at": case["quote_received_at"],
            "decision_time": (received + timedelta(seconds=5)).isoformat(),
            "robust_expected_return": 1.7,
            "sidecar_config": self.sidecar_config,
            "coordinator_config": self.coordinator_config,
        }

    def evaluate(self, case_id: str) -> dict:
        return SIDECAR.evaluate_sidecar_decision(**self.inputs(case_id))

    def test_all_registered_contract_cases(self) -> None:
        for case in self.fixture["cases"]:
            with self.subTest(case_id=case["case_id"]):
                event = self.evaluate(case["case_id"])
                expected = case["expected_status"]
                if expected == "ONE_PAPER_EVENT":
                    expected = "PAPER_READY"
                self.assertEqual(event["status"], expected)
                self.assertFalse(event["formal_buy"])
                self.assertFalse(event["send_order"])
                self.assertEqual(event["paper_stake_yen"], 0)

    def test_registered_fail_close_reasons(self) -> None:
        expected = {
            "source_timestamp_missing_fail_close": "QUOTE_SOURCE_TIME_MISSING_OR_AMBIGUOUS",
            "quote_received_after_cutoff_fail_close": "STRICT_QUOTE_CUTOFF_FAILED",
            "candidate_freeze_contract_false_fail_close": "CANDIDATE_FREEZE_CONTRACT_FAILED",
            "schedule_version_missing_fail_close": "SCHEDULE_CONTRACT_FAILED",
            "starter_universe_changed_fail_close": "STARTER_UNIVERSE_CHANGED",
            "wrong_pair_fail_close": "EXACT_CANDIDATE_QUOTE_MISSING_OR_AMBIGUOUS",
        }
        for case_id, reason in expected.items():
            with self.subTest(case_id=case_id):
                self.assertEqual(self.evaluate(case_id)["reason"], reason)

    def test_valid_quote_uses_declared_minute_resolution(self) -> None:
        event = self.evaluate("exact_pair_source_time_before_cutoff")
        self.assertEqual(event["status"], "PAPER_READY")
        self.assertEqual(event["trigger"], "EARLY_VALUE")
        self.assertEqual(event["quote_source_event_time"], "2026-08-02T18:26:00+09:00")

    def test_exact_wide_parser_canonicalizes_pair(self) -> None:
        parsed = SIDECAR.parse_exact_wide_quote(
            self.odds_html("2026-08-02T18:26:00+09:00", "2-1"), "01-02"
        )
        self.assertEqual(parsed, {"odds_low": 4.2, "odds_high": 4.8})

    def test_duplicate_capture_appends_once(self) -> None:
        event = self.evaluate("duplicate_capture_is_idempotent")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shadow.jsonl"
            self.assertTrue(SIDECAR.append_event_jsonl(path, event))
            self.assertFalse(SIDECAR.append_event_jsonl(path, event))
            records = SIDECAR.read_event_jsonl(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["idempotency_key"], event["idempotency_key"])

    def test_failure_target_is_retained_in_ledger(self) -> None:
        event = self.evaluate("starter_universe_changed_fail_close")
        self.assertEqual(event["status"], "NO_BET")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shadow.jsonl"
            self.assertTrue(SIDECAR.append_event_jsonl(path, event))
            self.assertEqual(SIDECAR.read_event_jsonl(path)[0]["reason"], "STARTER_UNIVERSE_CHANGED")

    def test_renderer_is_separate_and_read_only(self) -> None:
        records = [
            self.evaluate("exact_pair_source_time_before_cutoff"),
            self.evaluate("wrong_pair_fail_close"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            json_path = Path(temporary) / "sidecar.json"
            html_path = Path(temporary) / "sidecar.html"
            payload = RENDERER.write_artifacts(
                records,
                json_path=json_path,
                html_path=html_path,
                generated_at="2026-08-01T00:00:00Z",
            )
            html = html_path.read_text(encoding="utf-8").lower()
            self.assertEqual(payload["record_count"], 2)
            self.assertFalse(payload["formal_buy"])
            self.assertFalse(payload["send_order"])
            self.assertEqual(payload["stake"], 0)
            self.assertNotIn("<form", html)
            self.assertNotIn("<button", html)
            self.assertNotIn("<input", html)
            self.assertIn("閲覧専用", html)

    def test_forbidden_post_race_input_fails_closed(self) -> None:
        inputs = self.inputs("exact_pair_source_time_before_cutoff")
        inputs["candidate_row"]["payout"] = 420
        event = SIDECAR.evaluate_sidecar_decision(**inputs)
        self.assertEqual(event["status"], "NO_BET")
        self.assertEqual(event["reason"], "FORBIDDEN_INPUT_FIELD")

    def test_quote_request_identity_mismatch_fails_closed(self) -> None:
        inputs = self.inputs("exact_pair_source_time_before_cutoff")
        inputs["universe_observation"]["race_id"] = "2026080201010102"
        event = SIDECAR.evaluate_sidecar_decision(**inputs)
        self.assertEqual(event["status"], "NO_BET")
        self.assertEqual(event["reason"], "QUOTE_REQUEST_IDENTITY_MISMATCH")

    def test_missing_decision_time_fails_closed(self) -> None:
        inputs = self.inputs("exact_pair_source_time_before_cutoff")
        inputs["decision_time"] = None
        event = SIDECAR.evaluate_sidecar_decision(**inputs)
        self.assertEqual(event["status"], "NO_BET")
        self.assertEqual(event["reason"], "DECISION_TIME_MISSING")

    def test_unsafe_sidecar_config_is_rejected(self) -> None:
        unsafe = deepcopy(self.sidecar_config)
        unsafe["safety"]["send_order"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe.json"
            path.write_text(json.dumps(unsafe), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "send_order"):
                SIDECAR.load_sidecar_config(path)


if __name__ == "__main__":
    unittest.main()
