from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "config" / "schedule_terminal_ledger_e2e_exp014.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCHEDULE = load_module(
    "schedule_only_provider_v1",
    ROOT / "scripts" / "research" / "schedule_only_provider_v1.py",
)
CAPTURE = load_module(
    "exp014_capture",
    ROOT / "scripts" / "research" / "build_strict_t3_capture_packet_v1.py",
)
RUNNER = load_module(
    "exp014_runner",
    ROOT / "scripts" / "research" / "run_strict_t3_shadow_decision_v1.py",
)


class ScheduleTerminalLedgerE2ETests(unittest.TestCase):
    NOW = datetime.fromisoformat("2026-08-02T14:56:31+09:00")

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        manifest_ref = cls.spec["case_manifest"]
        manifest_path = ROOT / manifest_ref["path"]
        observed_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if observed_sha != manifest_ref["sha256"]:
            raise AssertionError("EXP014 fixed case manifest hash mismatch")
        cls.case_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_case_ids = [row["case_id"] for row in cls.case_manifest["cases"]]
        if observed_case_ids != cls.spec["expected_case_ids"]:
            raise AssertionError("EXP014 fixed case identifiers changed")
        if len(observed_case_ids) != 12:
            raise AssertionError("EXP014 must retain exactly twelve fixed cases")
        safety = cls.spec["safety"]
        if safety != {
            "data_class": "synthetic",
            "real_data_access": False,
            "odds_access": False,
            "result_access": False,
            "roi_calculation": False,
            "production_write": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }:
            raise AssertionError("EXP014 safety contract changed")

        base_config = CAPTURE.load_config(ROOT / cls.spec["base_runner_config"])
        cls.config = copy.deepcopy(base_config)
        cls.config["experiment_id"] = cls.spec["experiment_id"]
        cls.config["source_experiment_id"] = cls.spec["source_experiment_id"]
        cls.config["cohort_id"] = cls.spec["cohort_id"]
        cls.config["population"] = {
            "expected_target_rows": 12,
            "expected_candidate_ready_rows": 12,
            "expected_source_failure_rows": 0,
            "source_failure_reason": "CANDIDATE_SOURCE_NOT_READY",
            "allowed_source_failure_reasons": ["CANDIDATE_SOURCE_NOT_READY"],
        }

    @staticmethod
    def race_id(race_no: int) -> str:
        return f"2026040204{race_no:02d}"

    def candidate(self, race_no: int) -> dict:
        race_id = self.race_id(race_no)
        candidate = {
            "schema_version": 1,
            "experiment_id": self.config["source_experiment_id"],
            "cohort_id": "synthetic-exp014-source",
            "card_id": "synthetic-exp014-card",
            "race_id": race_id,
            "race_no": race_no,
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
                {"race_id": race_id, "runners": ["1", "2", "3"]}
            ),
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        candidate["candidate_freeze_record_hash"] = CAPTURE.candidate_record_digest(
            candidate
        )
        return candidate

    def acknowledgement(
        self, candidate: dict, packet_sha: str, packet_path: str
    ) -> dict:
        return {
            "schema_version": 1,
            "experiment_id": self.config["source_experiment_id"],
            "cohort_id": "synthetic-exp014-source",
            "race_id": candidate["race_id"],
            "race_no": candidate["race_no"],
            "record_status": "CANDIDATE_READY",
            "candidate_freeze_record_hash": candidate[
                "candidate_freeze_record_hash"
            ],
            "candidate_freeze_persist_ack_at": "2026-08-02T14:55:00+09:00",
            "packet_path": packet_path,
            "packet_file_sha256": packet_sha,
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }

    def write_population(
        self, root: Path, *, count: int = 1
    ) -> tuple[Path, list[dict]]:
        packet_dir = root / "packets"
        packet_dir.mkdir(parents=True)
        ledger_path = root / "candidate_freeze_ledger.jsonl"
        acknowledgements = []
        rows = []
        for race_no in range(1, count + 1):
            candidate = self.candidate(race_no)
            relative_packet = Path("packets") / f"{candidate['race_id']}.json"
            packet_path = root / relative_packet
            packet_path.write_text(
                CAPTURE.canonical_json(candidate) + "\n", encoding="utf-8"
            )
            packet_sha = CAPTURE.file_sha256(packet_path)
            acknowledgement = self.acknowledgement(
                candidate, packet_sha, relative_packet.as_posix()
            )
            acknowledgements.append(acknowledgement)
            rows.append(
                {
                    "card_id": "synthetic-exp014-card",
                    "candidate": candidate,
                    "acknowledgement": acknowledgement,
                    "candidate_packet_sha256": packet_sha,
                }
            )
        ledger_path.write_text(
            "".join(
                CAPTURE.canonical_json(acknowledgement) + "\n"
                for acknowledgement in acknowledgements
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "experiment_id": self.config["experiment_id"],
            "source_experiment_id": self.config["source_experiment_id"],
            "data_class": "synthetic",
            "sources": [
                {
                    "card_id": "synthetic-exp014-card",
                    "candidate_ledger_jsonl": str(ledger_path),
                    "candidate_ledger_sha256": CAPTURE.file_sha256(ledger_path),
                }
            ],
        }
        manifest_path = root / "source_manifest.json"
        manifest_path.write_text(
            CAPTURE.canonical_json(manifest) + "\n", encoding="utf-8"
        )
        return manifest_path, rows

    def schedule_observation(
        self,
        race_no: int = 1,
        *,
        observation_id: str = "obs-1",
        version: str = "v1",
        post: str = "2026-08-02T15:00:00+09:00",
        source: str = "2026-08-02T14:56:20+09:00",
        received: str = "2026-08-02T14:56:29+09:00",
    ) -> dict:
        race_id = self.race_id(race_no)
        return {
            "schema_version": 1,
            "schedule_provider_id": "file_backed_schedule_v1",
            "schedule_observation_id": observation_id,
            "race_id": race_id,
            "scheduled_post_time": post,
            "source_event_time": source,
            "received_at": received,
            "source_reference": f"synthetic:{observation_id}:{version}",
            "source_payload_sha256": hashlib.sha256(
                f"{race_id}:{observation_id}:{version}:{post}".encode("ascii")
            ).hexdigest(),
            "schedule_version": version,
        }

    @staticmethod
    def write_schedule(path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
        )

    def run_worker(
        self,
        root: Path,
        manifest_path: Path,
        provider,
        *,
        enforce_expected_counts: bool = False,
    ) -> tuple[dict, list[str]]:
        quote_calls: list[str] = []

        def forbidden_quote_fetch(*_args, **_kwargs):
            quote_calls.append("called")
            raise AssertionError("EXP014 synthetic schedule failure entered odds path")

        def forbidden_sleep(_seconds: float) -> None:
            raise AssertionError("EXP014 deterministic worker unexpectedly waited")

        summary = RUNNER.run_live_jra_card(
            source_manifest_path=manifest_path,
            capture_packet_dir=root / "captures",
            raw_html_dir=root / "raw",
            decision_ledger_path=root / "decisions.jsonl",
            summary_path=root / "summary.json",
            config=self.config,
            fetch_cname=forbidden_quote_fetch,
            fetch_schedule_document=provider,
            clock=lambda: self.NOW,
            sleeper=forbidden_sleep,
            enforce_expected_counts=enforce_expected_counts,
        )
        return summary, quote_calls

    def run_file_provider_case(
        self, records: list[dict] | None, expected_reason: str
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _rows = self.write_population(root)
            schedule_path = root / "schedule.jsonl"
            if records is not None:
                self.write_schedule(schedule_path, records)
            provider = RUNNER.build_file_schedule_provider(
                schedule_path, config=self.config, clock=lambda: self.NOW
            )
            summary, quote_calls = self.run_worker(root, manifest_path, provider)
            decisions = RUNNER.read_decision_jsonl(root / "decisions.jsonl")
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0]["decision_reason"], expected_reason)
            self.assertEqual(quote_calls, [])
            self.assertFalse(decisions[0]["formal_buy"])
            self.assertFalse(decisions[0]["send_order"])
            self.assertEqual(decisions[0]["stake"], 0)

    def write_failure_marker(
        self, root: Path, row: dict, reason: str
    ) -> Path:
        path = root / "captures" / f"{row['candidate']['race_id']}.capture_failure.json"
        RUNNER._capture_failure_marker(
            path=path,
            candidate=row["candidate"],
            reason=reason,
            observed_at=self.NOW,
            config=self.config,
        )
        return path

    def run_decisions(
        self, root: Path, manifest_path: Path, *, enforce_expected_counts: bool = False
    ) -> dict:
        return RUNNER.run_shadow_decisions(
            source_manifest_path=manifest_path,
            capture_packet_dir=root / "captures",
            decision_ledger_path=root / "decisions.jsonl",
            summary_path=root / "summary.json",
            config=self.config,
            now=self.NOW,
            enforce_expected_counts=enforce_expected_counts,
        )

    def test_provider_unavailable_to_terminal_decision(self) -> None:
        self.run_file_provider_case(
            None, "NO_BET_SCHEDULE_SOURCE_UNAVAILABLE"
        )

    def test_conflicting_version_to_terminal_decision(self) -> None:
        first = self.schedule_observation(observation_id="obs-1", version="v1")
        conflict = self.schedule_observation(
            observation_id="obs-2",
            version="v1",
            post="2026-08-02T15:02:00+09:00",
        )
        self.run_file_provider_case(
            [first, conflict], "NO_BET_SCHEDULE_CONTRACT_FAILURE"
        )

    def test_missing_source_time_to_terminal_decision(self) -> None:
        record = self.schedule_observation()
        del record["source_event_time"]
        self.run_file_provider_case(
            [record], "NO_BET_SCHEDULE_CONTRACT_FAILURE"
        )

    def test_reversed_source_receive_time_to_terminal_decision(self) -> None:
        record = self.schedule_observation(
            source="2026-08-02T14:56:29+09:00",
            received="2026-08-02T14:56:20+09:00",
        )
        self.run_file_provider_case(
            [record], "NO_BET_SCHEDULE_CONTRACT_FAILURE"
        )

    def test_forbidden_odds_field_to_terminal_decision(self) -> None:
        record = self.schedule_observation()
        record["odds"] = 3.5
        self.run_file_provider_case(
            [record], "NO_BET_SCHEDULE_CONTRACT_FAILURE"
        )

    def test_late_schedule_update_to_terminal_decision(self) -> None:
        record = self.schedule_observation(post="2026-08-02T14:59:00+09:00")
        self.run_file_provider_case(
            [record], "NO_BET_SCHEDULE_CONTRACT_FAILURE"
        )

    def test_corrupt_persisted_schedule_lock_to_terminal_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, rows = self.write_population(root)
            candidate = rows[0]["candidate"]
            schedule = CAPTURE.build_schedule_lock_record(
                candidate=candidate,
                scheduled_post_time="2026-08-02T15:00:00+09:00",
                schedule_source_event_time="2026-08-02T14:56:20+09:00",
                schedule_received_at="2026-08-02T14:56:29+09:00",
                schedule_lock_time=self.NOW,
                source_reference="synthetic:corrupt-lock",
                source_payload_sha256="a" * 64,
                config=self.config,
            )
            schedule["schedule_record_hash"] = "0" * 64
            schedule_path = (
                root / "captures" / f"{candidate['race_id']}.schedule_lock.json"
            )
            schedule_path.parent.mkdir(parents=True)
            schedule_path.write_text(
                CAPTURE.canonical_json(schedule) + "\n", encoding="utf-8"
            )
            summary = self.run_decisions(root, manifest_path)
            decisions = RUNNER.read_decision_jsonl(root / "decisions.jsonl")
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(decisions), 1)
            self.assertEqual(
                decisions[0]["decision_reason"],
                "NO_BET_SCHEDULE_CONTRACT_FAILURE",
            )

    def test_duplicate_worker_invocation_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _rows = self.write_population(root)
            schedule_path = root / "unavailable.jsonl"
            provider = RUNNER.build_file_schedule_provider(
                schedule_path, config=self.config, clock=lambda: self.NOW
            )
            first, first_quote_calls = self.run_worker(root, manifest_path, provider)
            second, second_quote_calls = self.run_worker(root, manifest_path, provider)
            decisions = RUNNER.read_decision_jsonl(root / "decisions.jsonl")
            self.assertEqual(first, second)
            self.assertEqual(len(decisions), 1)
            self.assertEqual(first_quote_calls + second_quote_calls, [])
            concurrent_equivalent = copy.deepcopy(decisions[0])
            concurrent_equivalent["t3_decision_committed_at"] = (
                "2026-08-02T14:56:31.999+09:00"
            )
            concurrent_equivalent["t3_decision_record_hash"] = (
                RUNNER._decision_digest(concurrent_equivalent)
            )
            self.assertFalse(
                RUNNER.append_decision_jsonl(
                    root / "decisions.jsonl", concurrent_equivalent
                )
            )
            self.assertEqual(
                len(RUNNER.read_decision_jsonl(root / "decisions.jsonl")), 1
            )

    def test_restart_after_failure_marker_before_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, rows = self.write_population(root)
            marker_path = self.write_failure_marker(
                root, rows[0], "NO_BET_SCHEDULE_SOURCE_UNAVAILABLE"
            )
            marker = CAPTURE.load_json_object(marker_path)
            RUNNER.verify_capture_failure_marker(
                marker, candidate=rows[0]["candidate"], config=self.config
            )
            self.assertFalse((root / "decisions.jsonl").exists())
            summary = self.run_decisions(root, manifest_path)
            decisions = RUNNER.read_decision_jsonl(root / "decisions.jsonl")
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(decisions), 1)
            self.assertEqual(
                decisions[0]["decision_reason"],
                "NO_BET_SCHEDULE_SOURCE_UNAVAILABLE",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, rows = self.write_population(root)
            marker_path = self.write_failure_marker(
                root, rows[0], "NO_BET_SCHEDULE_SOURCE_UNAVAILABLE"
            )
            marker = CAPTURE.load_json_object(marker_path)
            marker["capture_failure_record_hash"] = "0" * 64
            marker_path.write_text(
                CAPTURE.canonical_json(marker) + "\n", encoding="utf-8"
            )
            summary = self.run_decisions(root, manifest_path)
            decisions = RUNNER.read_decision_jsonl(root / "decisions.jsonl")
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(decisions), 1)
            self.assertEqual(
                decisions[0]["decision_reason"], "NO_BET_CAPTURE_PACKET_INVALID"
            )

    def test_orphan_idempotency_marker_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, rows = self.write_population(root)
            row = rows[0]
            self.write_failure_marker(
                root, row, "NO_BET_SCHEDULE_SOURCE_UNAVAILABLE"
            )
            decision = RUNNER.no_capture_decision(
                candidate=row["candidate"],
                acknowledgement=row["acknowledgement"],
                card_id=row["card_id"],
                config=self.config,
                committed_at=self.NOW,
                reason="NO_BET_SCHEDULE_SOURCE_UNAVAILABLE",
            )
            ledger_path = root / "decisions.jsonl"
            marker = RUNNER._decision_marker_path(
                ledger_path, decision["idempotency_key"]
            )
            marker.parent.mkdir(parents=True)
            marker.write_text(decision["idempotency_key"], encoding="ascii")
            summary = self.run_decisions(root, manifest_path)
            decisions = RUNNER.read_decision_jsonl(ledger_path)
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(decisions), 1)
            self.assertEqual(
                decisions[0]["t3_decision_record_hash"],
                decision["t3_decision_record_hash"],
            )
            self.assertEqual(
                marker.read_text(encoding="ascii"), decision["idempotency_key"]
            )

    def test_restart_after_ledger_append_rebuilds_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, rows = self.write_population(root)
            row = rows[0]
            decision = RUNNER.no_capture_decision(
                candidate=row["candidate"],
                acknowledgement=row["acknowledgement"],
                card_id=row["card_id"],
                config=self.config,
                committed_at=self.NOW,
                reason="NO_BET_SCHEDULE_SOURCE_UNAVAILABLE",
            )
            ledger_path = root / "decisions.jsonl"
            self.assertTrue(RUNNER.append_decision_jsonl(ledger_path, decision))
            self.assertFalse((root / "summary.json").exists())
            summary = self.run_decisions(root, manifest_path)
            persisted = CAPTURE.load_json_object(root / "summary.json")
            self.assertEqual(summary, persisted)
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(RUNNER.read_decision_jsonl(ledger_path)), 1)

    def test_mixed_failure_card_preserves_complete_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, rows = self.write_population(root, count=12)
            for row in rows[8:]:
                reason = (
                    "NO_BET_SCHEDULE_SOURCE_UNAVAILABLE"
                    if row["candidate"]["race_no"] % 2
                    else "NO_BET_SCHEDULE_CONTRACT_FAILURE"
                )
                self.write_failure_marker(root, row, reason)

            def mixed_provider(candidate: dict) -> dict:
                if int(candidate["race_no"]) <= 4:
                    raise SCHEDULE.ScheduleProviderUnavailable(
                        "synthetic provider unavailable"
                    )
                raise SCHEDULE.ScheduleProviderContractError(
                    "synthetic schedule contract failure"
                )

            summary, quote_calls = self.run_worker(
                root,
                manifest_path,
                mixed_provider,
                enforce_expected_counts=True,
            )
            decisions = RUNNER.read_decision_jsonl(root / "decisions.jsonl")
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["expected_target_rows"], 12)
            self.assertEqual(summary["recorded_target_rows"], 12)
            self.assertEqual(summary["missing_target_race_ids"], [])
            self.assertEqual(summary["duplicate_decision_rows"], 0)
            self.assertEqual(len(decisions), 12)
            self.assertEqual(len({row["race_id"] for row in decisions}), 12)
            self.assertEqual(quote_calls, [])
            self.assertEqual(
                summary["reason_counts"],
                {
                    "NO_BET_SCHEDULE_CONTRACT_FAILURE": 6,
                    "NO_BET_SCHEDULE_SOURCE_UNAVAILABLE": 6,
                },
            )
            self.assertTrue(all(row["shadow_action"] == "NO_BET" for row in decisions))
            self.assertTrue(all(row["formal_buy"] is False for row in decisions))
            self.assertTrue(all(row["send_order"] is False for row in decisions))
            self.assertTrue(all(row["stake"] == 0 for row in decisions))


if __name__ == "__main__":
    unittest.main()
