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
MODULE_PATH = ROOT / "scripts" / "research" / "run_grade_r_one_card_pilot_v1.py"
CONFIG_PATH = ROOT / "config" / "grade_r_one_card_pilot_v1.json"
FIXTURE_PATH = (
    ROOT
    / "research"
    / "drafts"
    / "EXP-20260801-004-grade-r-one-card-pilot-v1.fold_manifest.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module("run_grade_r_one_card_pilot_v1", MODULE_PATH)


class GradeROneCardPilotTests(unittest.TestCase):
    POST_TIMES = {
        1: "09:50",
        2: "10:20",
        3: "10:50",
        4: "11:20",
        5: "11:50",
        6: "15:10",
        7: "15:45",
        8: "16:20",
        9: "16:55",
        10: "17:30",
        11: "18:00",
        12: "18:30",
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MODULE.load_pilot_config(CONFIG_PATH)
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    @staticmethod
    def odds_html(pair_key: str, source_hour: int, source_minute: int) -> str:
        first, second = pair_key.split("-")
        return f"""<!doctype html>
<html lang="ja"><body>
  <div class="time">{source_hour}時{source_minute:02d}分現在オッズ</div>
  <table class="odds wide">
    <caption>{first}</caption>
    <tbody><tr><th>{second}</th><td><span class="min">4.2</span> - <span class="max">4.8</span></td></tr></tbody>
  </table>
</body></html>"""

    def write_packet(
        self,
        root: Path,
        *,
        race_id: str,
        race_no: int,
        robust_er: float = 1.7,
        source_seconds_before_post: int = 240,
        received_seconds_after_source: int = 20,
        candidate_hash_override: str | None = None,
        universe_changed: bool = False,
        schedule_ok: bool = True,
    ) -> Path:
        pair_key = "1-2"
        candidate_identity = {
            "race_id": race_id,
            "ticket_type": "wide",
            "candidate_pair_key": pair_key,
        }
        candidate_hash = MODULE.candidate_digest(candidate_identity)
        post = datetime.fromisoformat(
            f"2026-08-02T{self.POST_TIMES[race_no]}:00+09:00"
        )
        source = post - timedelta(seconds=source_seconds_before_post)
        freeze_ack = source - timedelta(minutes=6)
        odds_join = source - timedelta(seconds=30)
        request = source - timedelta(seconds=25)
        received = source + timedelta(seconds=received_seconds_after_source)
        decision = received + timedelta(seconds=5)
        html_path = root / f"{race_id}.html"
        html_path.write_text(
            self.odds_html(pair_key, source.hour, source.minute),
            encoding="utf-8",
        )
        packet = {
            "race_id": race_id,
            "candidate_row": {
                **candidate_identity,
                "candidate_freeze_record_hash": candidate_hash_override
                if candidate_hash_override is not None
                else candidate_hash,
                "candidate_freeze_contract_ok": True,
                "candidate_uses_odds": False,
                "candidate_freeze_persist_ack_at": freeze_ack.isoformat(),
                "starter_universe_hash_at_freeze": "universe-001",
            },
            "schedule_record": {
                "scheduled_post_time": post.isoformat(),
                "schedule_record_hash": "schedule-001" if schedule_ok else "",
                "schedule_contract_ok": schedule_ok,
            },
            "universe_observation": {
                "race_id": race_id,
                "ticket_type": "wide",
                "quote_pair_key": pair_key,
                "starter_universe_hash_at_quote": "universe-changed"
                if universe_changed
                else "universe-001",
                "starter_universe_unchanged": not universe_changed,
                "market_status": "OPEN",
            },
            "odds_html_path": html_path.name,
            "odds_join_started_at": odds_join.isoformat(),
            "quote_request_started_at": request.isoformat(),
            "quote_received_at": received.isoformat(),
            "decision_time": decision.isoformat(),
            "robust_expected_return": robust_er,
        }
        path = root / f"{race_id}.packet.json"
        path.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
        return path

    def build_manifest(
        self,
        root: Path,
        *,
        packet_races: set[int],
        packet_options: dict[int, dict] | None = None,
    ) -> Path:
        packet_options = packet_options or {}
        records = []
        for race_no in range(1, 13):
            race_id = f"2026040204{race_no:02d}"
            record = {
                "card_id": "2026040204",
                "race_id": race_id,
                "race_no": race_no,
                "scheduled_post_time": f"2026-08-02T{self.POST_TIMES[race_no]}:00+09:00",
                "target_registered": True,
                "capture_packet_path": "",
            }
            if race_no in packet_races:
                packet = self.write_packet(
                    root,
                    race_id=race_id,
                    race_no=race_no,
                    **packet_options.get(race_no, {}),
                )
                record["capture_packet_path"] = packet.name
            records.append(record)
        manifest = {
            "schema_version": 1,
            "experiment_id": self.config["experiment_id"],
            "cohort_id": self.config["cohort_id"],
            "data_class": "synthetic",
            "target_card": {
                "race_date": "2026-08-02",
                "venue_code": "04",
                "meeting_no": 2,
                "day_no": 4,
            },
            "records": records,
        }
        path = root / "target_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def run_manifest(self, root: Path, manifest: Path, now: str) -> dict:
        return MODULE.run_one_card(
            manifest_path=manifest,
            ledger_path=root / "ledger.jsonl",
            summary_path=root / "summary.json",
            config=self.config,
            now=MODULE._parse_time(now, self.config["timezone"]),
        )

    def test_full_card_all_valid_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_manifest(root, packet_races=set(range(1, 13)))
            summary = self.run_manifest(root, manifest, "2026-08-02T18:30:00+09:00")
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["recorded_target_rows"], 12)
            self.assertEqual(summary["status_counts"], {"PAPER_READY": 12})
            self.assertEqual(summary["candidate_changes_after_odds"], 0)

    def test_low_value_becomes_no_bet_without_pair_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_manifest(
                root,
                packet_races=set(range(1, 13)),
                packet_options={4: {"robust_er": 1.49}},
            )
            summary = self.run_manifest(root, manifest, "2026-08-02T18:30:00+09:00")
            records = MODULE.read_event_jsonl(root / "ledger.jsonl")
            row = next(record for record in records if record["race_no"] == 4)
            self.assertEqual(row["status"], "NO_BET")
            self.assertEqual(row["reason"], "NO_BET_VALUE_BELOW_THRESHOLD")
            self.assertEqual(row["candidate_pair_key"], "1-2")
            self.assertFalse(row["candidate_changed_after_odds"])
            self.assertEqual(summary["status"], "PASS")

    def test_missing_packet_before_cutoff_stays_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_manifest(root, packet_races=set())
            summary = self.run_manifest(root, manifest, "2026-08-02T09:00:00+09:00")
            self.assertEqual(summary["status"], "IN_PROGRESS")
            self.assertEqual(summary["recorded_target_rows"], 0)
            self.assertEqual(len(summary["pending_target_race_ids"]), 12)

    def test_missing_packet_after_cutoff_is_retained_as_no_bet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_manifest(root, packet_races=set(range(2, 13)))
            summary = self.run_manifest(root, manifest, "2026-08-02T18:30:00+09:00")
            records = MODULE.read_event_jsonl(root / "ledger.jsonl")
            first = next(record for record in records if record["race_no"] == 1)
            self.assertEqual(first["status"], "NO_BET")
            self.assertEqual(first["reason"], "NO_BET_T3_CAPTURE_PACKET_MISSING")
            self.assertEqual(summary["recorded_target_rows"], 12)

    def test_restart_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_manifest(root, packet_races=set(range(1, 13)))
            first = self.run_manifest(root, manifest, "2026-08-02T18:30:00+09:00")
            second = self.run_manifest(root, manifest, "2026-08-02T18:31:00+09:00")
            records = MODULE.read_event_jsonl(root / "ledger.jsonl")
            self.assertEqual(first["recorded_target_rows"], 12)
            self.assertEqual(second["recorded_target_rows"], 12)
            self.assertEqual(len(records), 12)
            self.assertEqual(second["duplicate_decision_rows"], 0)

    def test_target_manifest_must_include_all_twelve_races(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self.build_manifest(root, packet_races=set())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["records"].pop()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "race numbers mismatch"):
                MODULE.validate_target_manifest(manifest, self.config)

    def test_candidate_hash_failure_is_no_bet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_manifest(
                root,
                packet_races=set(range(1, 13)),
                packet_options={3: {"candidate_hash_override": "0" * 64}},
            )
            summary = self.run_manifest(root, manifest, "2026-08-02T18:30:00+09:00")
            records = MODULE.read_event_jsonl(root / "ledger.jsonl")
            row = next(record for record in records if record["race_no"] == 3)
            self.assertEqual(row["reason"], "NO_BET_CANDIDATE_HASH_MISMATCH")
            self.assertEqual(summary["candidate_changes_after_odds"], 0)

    def test_universe_change_is_no_bet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_manifest(
                root,
                packet_races=set(range(1, 13)),
                packet_options={5: {"universe_changed": True}},
            )
            self.run_manifest(root, manifest, "2026-08-02T18:30:00+09:00")
            records = MODULE.read_event_jsonl(root / "ledger.jsonl")
            row = next(record for record in records if record["race_no"] == 5)
            self.assertEqual(row["reason"], "NO_BET_STARTER_UNIVERSE_CHANGED")

    def test_schedule_failure_is_no_bet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_manifest(
                root,
                packet_races=set(range(1, 13)),
                packet_options={6: {"schedule_ok": False}},
            )
            self.run_manifest(root, manifest, "2026-08-02T18:30:00+09:00")
            records = MODULE.read_event_jsonl(root / "ledger.jsonl")
            row = next(record for record in records if record["race_no"] == 6)
            self.assertEqual(row["reason"], "NO_BET_SCHEDULE_CONTRACT_FAILURE")

    def test_quote_after_strict_cutoff_is_no_bet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_manifest(
                root,
                packet_races=set(range(1, 13)),
                packet_options={7: {"source_seconds_before_post": 150}},
            )
            self.run_manifest(root, manifest, "2026-08-02T18:30:00+09:00")
            records = MODULE.read_event_jsonl(root / "ledger.jsonl")
            row = next(record for record in records if record["race_no"] == 7)
            self.assertEqual(row["reason"], "NO_BET_T3_QUOTE_ASOF_VIOLATION")

    def test_real_data_requires_running_registry_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "research").mkdir()
            event = {
                "experiment_id": self.config["experiment_id"],
                "status": "preparing",
                "real_data_execution_allowed": False,
                "run_scope_digest": None,
                "formal_buy": False,
                "send_order": False,
                "stake": 0,
            }
            (root / "research" / "REGISTRY.jsonl").write_text(
                json.dumps(event) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "RUNNING"):
                MODULE.assert_real_data_authorized(root, self.config["experiment_id"])

    def test_unsafe_config_is_rejected(self) -> None:
        unsafe = deepcopy(self.config)
        unsafe["policy"]["alternative_pair_search_allowed"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe.json"
            path.write_text(json.dumps(unsafe), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "alternative pair"):
                MODULE.load_pilot_config(path)


if __name__ == "__main__":
    unittest.main()
