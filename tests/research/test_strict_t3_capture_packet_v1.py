from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "research" / "build_strict_t3_capture_packet_v1.py"
CONFIG_PATH = ROOT / "config" / "strict_t3_shadow_decision_exp011.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module("build_strict_t3_capture_packet_v1", MODULE_PATH)


class StrictT3CapturePacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MODULE.load_config(CONFIG_PATH)

    def candidate(self, *, ready: bool = True) -> dict:
        candidate = {
            "schema_version": 1,
            "experiment_id": "EXP-20260802-010",
            "cohort_id": "synthetic-source",
            "card_id": "synthetic-card",
            "race_id": "202601010401",
            "race_no": 1,
            "scheduled_post_time_asof": "2026-08-02T15:00:00+09:00",
            "record_status": "CANDIDATE_READY" if ready else "FAILED",
            "candidate_freeze_contract_ok": ready,
            "failure_reason_codes": [] if ready else ["CANDIDATE_SOURCE_NOT_READY"],
            "candidate_horse_id_1": "1" if ready else "",
            "candidate_horse_id_2": "2" if ready else "",
            "candidate_pair_key": "1-2" if ready else "",
            "p_wide_coherent_raw": 0.39 if ready else None,
            "p_action_calibrated": 0.40 if ready else None,
            "confidence_gate_pass": ready,
            "starter_universe_hash_at_freeze": "universe-001",
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        candidate["candidate_freeze_record_hash"] = MODULE.candidate_record_digest(
            candidate
        )
        return candidate

    def acknowledgement(self, candidate: dict, packet_sha: str) -> dict:
        return {
            "experiment_id": "EXP-20260802-010",
            "race_id": candidate["race_id"],
            "record_status": candidate["record_status"],
            "candidate_freeze_record_hash": candidate["candidate_freeze_record_hash"],
            "candidate_freeze_persist_ack_at": "2026-08-02T14:55:00+09:00",
            "packet_file_sha256": packet_sha,
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }

    @staticmethod
    def schedule() -> dict:
        record = {
            "race_id": "202601010401",
            "scheduled_post_time_used": "2026-08-02T15:00:00+09:00",
            "schedule_version": "schedule-v1",
            "schedule_source_event_time": "2026-08-02T14:54:00+09:00",
            "schedule_received_at": "2026-08-02T14:54:01+09:00",
            "schedule_contract_ok": True,
        }
        record["schedule_record_hash"] = MODULE.schedule_record_digest(record)
        return record

    @staticmethod
    def universe() -> dict:
        return {
            "race_id": "202601010401",
            "starter_universe_hash_at_t3": "universe-001",
            "starter_universe_unchanged": True,
            "scratch_known_by_t3": False,
            "universe_observed_at": "2026-08-02T14:56:59+09:00",
        }

    @staticmethod
    def quote() -> dict:
        return {
            "race_id": "202601010401",
            "ticket_type": "wide",
            "quote_pair_key": "1-2",
            "quote_unique": True,
            "odds_join_started_at": "2026-08-02T14:56:30+09:00",
            "quote_request_started_at": "2026-08-02T14:56:31+09:00",
            "t3_quote_source_event_time": "2026-08-02T14:56:45+09:00",
            "t3_quote_received_at": "2026-08-02T14:56:50+09:00",
            "t3_quote_selected_asof_time": "2026-08-02T14:56:55+09:00",
            "feed_heartbeat_source_event_time": "2026-08-02T14:56:48+09:00",
            "feed_heartbeat_received_at": "2026-08-02T14:56:52+09:00",
            "feed_sequence_id": "feed-100",
            "market_status": "OPEN",
            "t3_wide_odds_low": 4.0,
            "t3_wide_odds_high": 4.6,
        }

    def write_candidate(self, root: Path, candidate: dict) -> tuple[Path, str]:
        path = root / "candidate.json"
        path.write_text(MODULE.canonical_json(candidate) + "\n", encoding="utf-8")
        return path, MODULE.file_sha256(path)

    def test_ready_candidate_is_bound_to_exact_quote_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.candidate()
            _path, packet_sha = self.write_candidate(root, candidate)
            packet = MODULE.build_capture_packet(
                candidate=candidate,
                acknowledgement=self.acknowledgement(candidate, packet_sha),
                candidate_packet_sha256=packet_sha,
                schedule_record=self.schedule(),
                universe_observation=self.universe(),
                quote_observation=self.quote(),
                captured_at="2026-08-02T14:57:00+09:00",
                data_class="synthetic",
                config=self.config,
            )
            MODULE.verify_capture_packet(packet, self.config)
            self.assertEqual(packet["candidate_pair_key"], "1-2")
            self.assertTrue(packet["quote_evaluation_allowed"])
            self.assertFalse(packet["candidate_uses_odds"])
            self.assertFalse(packet["formal_buy"])
            self.assertFalse(packet["send_order"])
            self.assertEqual(packet["stake"], 0)

    def test_source_failure_cannot_enter_quote_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.candidate(ready=False)
            _path, packet_sha = self.write_candidate(root, candidate)
            acknowledgement = self.acknowledgement(candidate, packet_sha)
            packet = MODULE.build_capture_packet(
                candidate=candidate,
                acknowledgement=acknowledgement,
                candidate_packet_sha256=packet_sha,
                schedule_record=None,
                universe_observation=None,
                quote_observation=None,
                captured_at="2026-08-02T14:57:00+09:00",
                data_class="synthetic",
                config=self.config,
            )
            self.assertEqual(packet["packet_status"], "SOURCE_NOT_READY")
            self.assertFalse(packet["quote_evaluation_allowed"])
            self.assertEqual(packet["candidate_pair_key"], "")
            with self.assertRaisesRegex(ValueError, "must not enter quote"):
                MODULE.build_capture_packet(
                    candidate=candidate,
                    acknowledgement=acknowledgement,
                    candidate_packet_sha256=packet_sha,
                    schedule_record=self.schedule(),
                    universe_observation=self.universe(),
                    quote_observation=self.quote(),
                    captured_at="2026-08-02T14:57:00+09:00",
                    data_class="synthetic",
                    config=self.config,
                )

    def test_candidate_record_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.candidate()
            candidate["p_action_calibrated"] = 0.99
            _path, packet_sha = self.write_candidate(root, candidate)
            with self.assertRaisesRegex(ValueError, "candidate record hash"):
                MODULE.build_capture_packet(
                    candidate=candidate,
                    acknowledgement=self.acknowledgement(candidate, packet_sha),
                    candidate_packet_sha256=packet_sha,
                    schedule_record=self.schedule(),
                    universe_observation=self.universe(),
                    quote_observation=self.quote(),
                    captured_at="2026-08-02T14:57:00+09:00",
                    data_class="synthetic",
                    config=self.config,
                )

    def test_forbidden_post_race_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.candidate()
            _path, packet_sha = self.write_candidate(root, candidate)
            quote = self.quote()
            quote["payout"] = 420
            with self.assertRaisesRegex(ValueError, "forbidden input"):
                MODULE.build_capture_packet(
                    candidate=candidate,
                    acknowledgement=self.acknowledgement(candidate, packet_sha),
                    candidate_packet_sha256=packet_sha,
                    schedule_record=self.schedule(),
                    universe_observation=self.universe(),
                    quote_observation=quote,
                    captured_at="2026-08-02T14:57:00+09:00",
                    data_class="synthetic",
                    config=self.config,
                )

    def test_immutable_packet_cannot_be_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            payload = {"capture_packet_hash": "one"}
            MODULE.write_json_atomic_immutable(path, payload)
            MODULE.write_json_atomic_immutable(path, payload)
            with self.assertRaisesRegex(ValueError, "immutable"):
                MODULE.write_json_atomic_immutable(
                    path, {"capture_packet_hash": "two"}
                )

    def test_real_data_requires_running_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "research").mkdir()
            (root / "research" / "REGISTRY.jsonl").write_text(
                json.dumps(
                    {
                        "experiment_id": self.config["experiment_id"],
                        "status": "preparing",
                        "real_data_execution_allowed": False,
                        "formal_buy": False,
                        "send_order": False,
                        "stake": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "RUNNING"):
                MODULE.assert_real_data_authorized(root, self.config["experiment_id"])

    @staticmethod
    def jra_pages() -> dict[str, bytes]:
        top_cname = "pw15oli00/6D"
        venue_cname = "pw15orl012026010420260802/ABCDEF"
        detail_cname = "pw155abcS301202601040120260802Z/ABCDEF"
        top = (
            "<html><body><a onclick=\"doAction('/JRADB/accessO.html', "
            f"'{venue_cname}')\">venue</a></body></html>"
        )
        venue = (
            "<html><body><a onclick=\"doAction('/JRADB/accessO.html', "
            f"'{detail_cname}')\">wide</a></body></html>"
        )
        detail = """
        <html><body>
          <p>発走時刻 15:00</p><p>オッズ 14:56 現在</p>
          <table class="wide"><caption>1</caption><tbody>
            <tr><th>2</th><td><span class="min">4.0</span><span class="max">4.6</span></td></tr>
            <tr><th>3</th><td><span class="min">5.0</span><span class="max">5.8</span></td></tr>
          </tbody></table>
          <table class="wide"><caption>2</caption><tbody>
            <tr><th>3</th><td><span class="min">6.0</span><span class="max">6.8</span></td></tr>
          </tbody></table>
        </body></html>
        """
        return {
            top_cname: top.encode("cp932"),
            venue_cname: venue.encode("cp932"),
            detail_cname: detail.encode("cp932"),
        }

    def test_jra_public_capture_uses_only_exact_frozen_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.candidate()
            candidate["starter_universe_hash_at_freeze"] = MODULE.canonical_digest(
                {"race_id": candidate["race_id"], "runners": ["1", "2", "3"]}
            )
            candidate["candidate_freeze_record_hash"] = MODULE.candidate_record_digest(
                candidate
            )
            _path, packet_sha = self.write_candidate(root, candidate)
            pages = self.jra_pages()
            moments = iter(
                datetime.fromisoformat(value)
                for value in (
                    "2026-08-02T14:56:20+09:00",
                    "2026-08-02T14:56:21+09:00",
                    "2026-08-02T14:56:22+09:00",
                    "2026-08-02T14:56:23+09:00",
                    "2026-08-02T14:56:30+09:00",
                    "2026-08-02T14:56:31+09:00",
                    "2026-08-02T14:56:32+09:00",
                    "2026-08-02T14:56:33+09:00",
                    "2026-08-02T14:56:34+09:00",
                )
            )
            packet = MODULE.build_jra_official_capture_packet(
                candidate=candidate,
                acknowledgement=self.acknowledgement(candidate, packet_sha),
                candidate_packet_sha256=packet_sha,
                output_raw_html=root / "source.html",
                fetch_cname=lambda cname: pages[cname],
                clock=lambda: next(moments),
                data_class="synthetic",
                config=self.config,
            )
            MODULE.verify_capture_packet(packet, self.config)
            quote = packet["quote_observation"]
            self.assertEqual(quote["quote_pair_key"], "1-2")
            self.assertEqual(quote["t3_wide_odds_low"], 4.0)
            self.assertEqual(quote["t3_wide_odds_high"], 4.6)
            self.assertEqual(packet["public_source"]["request_count"], 3)
            self.assertTrue(packet["universe_observation"]["starter_universe_unchanged"])
            self.assertEqual(
                MODULE.extract_jra_quote_source_time(
                    pages[next(key for key in pages if key.startswith("pw155"))].decode("cp932"),
                    source_race_key="2026080201202601",
                    timezone_name="Asia/Tokyo",
                ),
                datetime(2026, 8, 2, 14, 56, tzinfo=ZoneInfo("Asia/Tokyo")),
            )

    def test_jra_source_time_missing_and_access_challenge_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            MODULE.extract_jra_quote_source_time(
                "<html>時刻なし</html>",
                source_race_key="2026080201010401",
                timezone_name="Asia/Tokyo",
            )
        with self.assertRaises(MODULE.JraAccessRestrictionError):
            MODULE.assert_public_jra_page_available("<html>CAPTCHA challenge</html>")

    def test_unsafe_config_is_rejected(self) -> None:
        unsafe = deepcopy(self.config)
        unsafe["quote_contract"]["alternative_pair_search_allowed"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe.json"
            path.write_text(json.dumps(unsafe), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "alternative pair"):
                MODULE.load_config(path)


if __name__ == "__main__":
    unittest.main()
