from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "race_day_contract_hardening_exp012.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CANDIDATE = load_module(
    "exp012_candidate",
    ROOT / "scripts" / "research" / "build_grade_r_candidate_freeze_packets_v1.py",
)
CAPTURE = load_module(
    "exp012_capture",
    ROOT / "scripts" / "research" / "build_strict_t3_capture_packet_v1.py",
)
RUNNER = load_module(
    "exp012_runner",
    ROOT / "scripts" / "research" / "run_strict_t3_shadow_decision_v1.py",
)


class RaceDayContractHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = CAPTURE.load_config(CONFIG_PATH)
        CANDIDATE.load_adapter_config(CONFIG_PATH)
        self.bundle = {
            "artifact_id": "exp012-synthetic-bundle",
            "model_kind": "linear_top3_set_softmax",
            "candidate_policy": "non_odds_top1_wide_pair_from_coherent_top3_softmax",
            "feature_cols": ["f_primary"],
            "mean": [0.0],
            "std": [1.0],
            "weights": [1.0],
            "temperature": 1.0,
        }
        self.bundle_path = self.root / "bundle.json"
        self.bundle_path.write_text(
            CANDIDATE.canonical_json(self.bundle) + "\n", encoding="utf-8"
        )
        self.targets = self._targets()
        self.target_path = self.root / "targets.json"
        self.source_path = self.root / "sources.json"
        self.features_path = self.root / "features.csv"
        self.output_dir = self.root / "candidate_output"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _targets(self) -> dict:
        records = []
        for race_no in range(1, 13):
            records.append(
                {
                    "race_id": f"2026040204{race_no:02d}",
                    "race_no": race_no,
                    "target_registered": True,
                    "scheduled_post_time": "2026-08-02T15:00:00+09:00",
                    "candidate_feature_cutoff_time": "2026-08-02T08:00:00+09:00",
                }
            )
        return {
            "experiment_id": self.config["experiment_id"],
            "cohort_id": self.config["cohort_id"],
            "data_class": "synthetic",
            "target_card": self.config["target_card"],
            "records": records,
        }

    def _write_candidate_inputs(self, domains: dict[int, str]) -> None:
        bundle_hash = CANDIDATE.file_sha256(self.bundle_path)
        schema_hash = CANDIDATE.canonical_digest(self.bundle["feature_cols"])
        feature_rows = []
        for target in self.targets["records"]:
            race_id = target["race_id"]
            for triplet, value in (
                (("1", "2", "3"), 3.0),
                (("1", "2", "4"), 2.5),
                (("1", "3", "4"), 1.5),
                (("2", "3", "4"), 1.0),
            ):
                feature_rows.append(
                    {
                        "race_id": race_id,
                        "horse_id_1": triplet[0],
                        "horse_id_2": triplet[1],
                        "horse_id_3": triplet[2],
                        "f_primary": str(value),
                    }
                )
        with self.features_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
            writer.writeheader()
            writer.writerows(feature_rows)
        snapshot_hash = CANDIDATE.file_sha256(self.features_path)
        records = []
        for target in self.targets["records"]:
            race_no = int(target["race_no"])
            race_id = str(target["race_id"])
            domain = domains.get(race_no, "flat_turf")
            record = {
                "race_id": race_id,
                "source_contract_ok": True,
                "input_snapshot_hash": snapshot_hash,
                "feature_input_max_source_event_time": "2026-08-02T07:00:00+09:00",
                "source_received_at": "2026-08-02T07:30:00+09:00",
                "candidate_feature_cutoff_time": "2026-08-02T08:00:00+09:00",
                "starter_universe_hash_at_freeze": CANDIDATE.canonical_digest(
                    {"race_id": race_id, "runners": ["1", "2", "3", "4"]}
                ),
                "runner_ids": ["1", "2", "3", "4"],
                "inference_bundle_hash": bundle_hash,
                "feature_schema_hash": schema_hash,
                "race_domain": domain,
                "race_domain_source_hash": CANDIDATE.canonical_digest(
                    {"race_id": race_id, "race_domain": domain}
                ),
            }
            record["source_record_hash"] = CANDIDATE.canonical_digest(record)
            records.append(record)
        self.target_path.write_text(
            CANDIDATE.canonical_json(self.targets) + "\n", encoding="utf-8"
        )
        self.source_path.write_text(
            CANDIDATE.canonical_json(
                {"schema_version": 1, "data_class": "synthetic", "records": records}
            )
            + "\n",
            encoding="utf-8",
        )

    def _run_candidates(self, now: str) -> dict:
        return CANDIDATE.run_adapter(
            target_manifest_path=self.target_path,
            feature_source_manifest_path=self.source_path,
            top3_feature_csv_path=self.features_path,
            runner_feature_csv_path=None,
            inference_bundle_path=self.bundle_path,
            output_dir=self.output_dir,
            config=self.config,
            now=datetime.fromisoformat(now),
            execution_mode="synthetic",
        )

    def _candidate_packet(self, race_no: int) -> dict:
        race_id = f"2026040204{race_no:02d}"
        return CANDIDATE.load_json_object(
            self.output_dir / "packets" / f"{race_id}.candidate_freeze.json"
        )

    def test_supported_turf_before_first_cutoff(self) -> None:
        self._write_candidate_inputs({1: "flat_turf"})
        self._run_candidates("2026-08-02T07:59:59.700000+09:00")
        packet = self._candidate_packet(1)
        self.assertEqual(packet["record_status"], "CANDIDATE_READY")
        self.assertEqual(packet["race_domain"], "flat_turf")

    def test_supported_dirt_before_first_cutoff(self) -> None:
        self._write_candidate_inputs({2: "flat_dirt"})
        self._run_candidates("2026-08-02T07:59:59.700000+09:00")
        packet = self._candidate_packet(2)
        self.assertEqual(packet["record_status"], "CANDIDATE_READY")
        self.assertEqual(packet["race_domain"], "flat_dirt")

    def test_obstacle_race_domain(self) -> None:
        self._write_candidate_inputs({3: "obstacle"})
        self._run_candidates("2026-08-02T07:59:59.700000+09:00")
        packet = self._candidate_packet(3)
        self.assertEqual(packet["record_status"], "FAILED")
        self.assertEqual(packet["failure_reason_codes"], ["UNSUPPORTED_RACE_TYPE"])
        self.assertEqual(packet["candidate_pair_key"], "")

    def test_unknown_race_domain(self) -> None:
        self._write_candidate_inputs({4: "unknown"})
        self._run_candidates("2026-08-02T07:59:59.700000+09:00")
        packet = self._candidate_packet(4)
        self.assertEqual(packet["record_status"], "FAILED")
        self.assertEqual(packet["failure_reason_codes"], ["UNSUPPORTED_RACE_TYPE"])

    def test_batch_committed_after_earliest_cutoff(self) -> None:
        self._write_candidate_inputs({})
        summary = self._run_candidates("2026-08-02T08:00:00+09:00")
        self.assertEqual(summary["recorded_target_rows"], 12)
        self.assertEqual(summary["failed_rows"], 12)
        self.assertEqual(
            self._candidate_packet(12)["failure_reason_codes"],
            ["SOURCE_READINESS_DEADLINE_MISSED"],
        )

    def _decision_fixture(self) -> tuple[dict, dict, str, dict]:
        candidate = {
            "schema_version": 1,
            "experiment_id": self.config["source_experiment_id"],
            "cohort_id": "synthetic-source",
            "card_id": "synthetic-card",
            "race_id": "202601010401",
            "race_no": 1,
            "scheduled_post_time_asof": "2026-08-02T15:00:00+09:00",
            "record_status": "CANDIDATE_READY",
            "candidate_freeze_contract_ok": True,
            "failure_reason_codes": [],
            "candidate_horse_id_1": "1",
            "candidate_horse_id_2": "2",
            "candidate_pair_key": "1-2",
            "p_wide_coherent_raw": 0.39,
            "p_action_calibrated": 0.40,
            "confidence_gate_pass": True,
            "starter_universe_hash_at_freeze": CAPTURE.canonical_digest(
                {"race_id": "202601010401", "runners": ["1", "2", "3"]}
            ),
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        candidate["candidate_freeze_record_hash"] = CAPTURE.candidate_record_digest(
            candidate
        )
        candidate_path = self.root / "candidate.json"
        candidate_path.write_text(CAPTURE.canonical_json(candidate) + "\n", encoding="utf-8")
        packet_sha = CAPTURE.file_sha256(candidate_path)
        acknowledgement = {
            "experiment_id": self.config["source_experiment_id"],
            "race_id": candidate["race_id"],
            "record_status": "CANDIDATE_READY",
            "candidate_freeze_record_hash": candidate["candidate_freeze_record_hash"],
            "candidate_freeze_persist_ack_at": "2026-08-02T14:55:00+09:00",
            "packet_file_sha256": packet_sha,
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        schedule = CAPTURE.build_schedule_lock_record(
            candidate=candidate,
            scheduled_post_time="2026-08-02T15:01:00+09:00",
            schedule_source_event_time="2026-08-02T14:56:20+09:00",
            schedule_received_at="2026-08-02T14:56:21+09:00",
            schedule_lock_time="2026-08-02T14:56:22+09:00",
            source_reference="synthetic-schedule",
            source_payload_sha256="a" * 64,
            config=self.config,
        )
        return candidate, acknowledgement, packet_sha, schedule

    def test_odd_minute_post_unchanged(self) -> None:
        _candidate, _ack, _sha, schedule = self._decision_fixture()
        attempts = CAPTURE.quote_attempt_times(schedule, self.config)
        self.assertEqual(attempts[0].isoformat(), "2026-08-02T14:57:30+09:00")

    def test_schedule_moves_one_minute_later_before_odds(self) -> None:
        candidate, acknowledgement, packet_sha, _schedule = self._decision_fixture()
        document = "<html><p>発走時刻 15:01</p></html>"
        schedule = CAPTURE.build_schedule_lock_from_document(
            candidate=candidate,
            document=document,
            schedule_source_event_time="2026-08-02T14:56:20+09:00",
            schedule_received_at="2026-08-02T14:56:21+09:00",
            schedule_lock_time="2026-08-02T14:56:22+09:00",
            source_reference="synthetic-schedule",
            config=self.config,
        )
        self.assertEqual(
            CAPTURE.parse_time(schedule["scheduled_post_time_used"], "Asia/Tokyo"),
            datetime.fromisoformat("2026-08-02T15:01:00+09:00"),
        )
        self.assertEqual(
            CAPTURE.parse_time(schedule["schedule_poll_open_time"], "Asia/Tokyo"),
            datetime.fromisoformat("2026-08-02T14:57:30+09:00"),
        )
        self.assertTrue(schedule["schedule_only"])

        fetch_calls = 0

        def fetch_cname(_cname: str) -> bytes:
            nonlocal fetch_calls
            fetch_calls += 1
            return b""

        with self.assertRaisesRegex(
            CAPTURE.ScheduleContractError,
            "outside the locked strict polling window",
        ):
            CAPTURE.build_jra_official_capture_packet(
                candidate=candidate,
                acknowledgement=acknowledgement,
                candidate_packet_sha256=packet_sha,
                output_raw_html=self.root / "must-not-exist.html",
                fetch_cname=fetch_cname,
                clock=lambda: datetime.fromisoformat("2026-08-02T14:57:29+09:00"),
                data_class="synthetic",
                config=self.config,
                schedule_record=schedule,
            )

        self.assertEqual(fetch_calls, 0)
        self.assertFalse((self.root / "must-not-exist.html").exists())

    def test_schedule_update_arrives_after_new_cutoff(self) -> None:
        candidate, _ack, _sha, _schedule = self._decision_fixture()
        late = CAPTURE.build_schedule_lock_record(
            candidate=candidate,
            scheduled_post_time="2026-08-02T14:59:00+09:00",
            schedule_source_event_time="2026-08-02T14:56:20+09:00",
            schedule_received_at="2026-08-02T14:56:30+09:00",
            schedule_lock_time="2026-08-02T14:56:31+09:00",
            source_reference="synthetic-late-schedule",
            source_payload_sha256="b" * 64,
            config=self.config,
        )
        self.assertFalse(late["schedule_contract_ok"])
        self.assertEqual(late["schedule_failure_reason"], "NO_BET_SCHEDULE_CONTRACT_FAILURE")

    def _evaluate_quote(self, source_time: str) -> dict:
        candidate, acknowledgement, packet_sha, schedule = self._decision_fixture()
        universe = {
            "race_id": candidate["race_id"],
            "starter_universe_hash_at_t3": candidate[
                "starter_universe_hash_at_freeze"
            ],
            "starter_universe_unchanged": True,
            "scratch_known_by_t3": False,
            "universe_observed_at": "2026-08-02T14:57:50+09:00",
        }
        quote = {
            "race_id": candidate["race_id"],
            "ticket_type": "wide",
            "quote_pair_key": "1-2",
            "quote_unique": True,
            "odds_join_started_at": "2026-08-02T14:57:30+09:00",
            "quote_request_started_at": "2026-08-02T14:57:31+09:00",
            "t3_quote_source_event_time": source_time,
            "t3_quote_received_at": "2026-08-02T14:57:40+09:00",
            "t3_quote_selected_asof_time": "2026-08-02T14:57:45+09:00",
            "feed_heartbeat_source_event_time": "2026-08-02T14:57:35+09:00",
            "feed_heartbeat_received_at": "2026-08-02T14:57:40+09:00",
            "feed_sequence_id": "synthetic-feed",
            "market_status": "OPEN",
            "t3_wide_odds_low": 4.0,
            "t3_wide_odds_high": 4.6,
        }
        packet = CAPTURE.build_capture_packet(
            candidate=candidate,
            acknowledgement=acknowledgement,
            candidate_packet_sha256=packet_sha,
            schedule_record=schedule,
            universe_observation=universe,
            quote_observation=quote,
            captured_at="2026-08-02T14:57:50+09:00",
            data_class="synthetic",
            config=self.config,
        )
        return RUNNER.evaluate_ready_capture(
            candidate=candidate,
            acknowledgement=acknowledgement,
            candidate_packet_sha256=packet_sha,
            capture_packet=packet,
            card_id="synthetic-card",
            config=self.config,
            committed_at=datetime.fromisoformat("2026-08-02T14:58:05+09:00"),
        )

    def test_provider_quote_source_age_90_seconds(self) -> None:
        decision = self._evaluate_quote("2026-08-02T14:56:00+09:00")
        self.assertEqual(decision["decision_reason"], "NO_BET_T3_QUOTE_STALE")
        self.assertGreater(decision["quote_age_seconds"], 60)

    def test_supported_exact_schedule_and_fresh_quote(self) -> None:
        decision = self._evaluate_quote("2026-08-02T14:57:30+09:00")
        self.assertEqual(decision["decision_reason"], "STRICT_T3_CONTRACTS_PASSED")
        self.assertEqual(decision["shadow_action"], "PAPER_READY")
        self.assertFalse(decision["formal_buy"])
        self.assertFalse(decision["send_order"])
        self.assertEqual(decision["stake"], 0)


if __name__ == "__main__":
    unittest.main()
