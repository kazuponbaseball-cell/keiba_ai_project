from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "research" / "build_grade_r_candidate_freeze_packets_v1.py"
SPEC = importlib.util.spec_from_file_location("candidate_freeze_adapter_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CandidateFreezeAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = MODULE.load_adapter_config(
            ROOT / "config" / "grade_r_candidate_freeze_adapter_v1.json"
        )
        self.bundle = {
            "artifact_id": "synthetic_linear_top3_bundle",
            "model_kind": "linear_top3_set_softmax",
            "candidate_policy": "non_odds_top1_wide_pair_from_coherent_top3_softmax",
            "feature_cols": ["f_primary", "f_balance"],
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "weights": [1.0, 0.2],
            "temperature": 1.0,
        }
        self.bundle_path = self.root / "bundle.json"
        self.bundle_path.write_text(
            MODULE.canonical_json(self.bundle) + "\n", encoding="utf-8"
        )
        self.bundle_hash = MODULE.file_sha256(self.bundle_path)
        self.feature_schema_hash = MODULE.canonical_digest(self.bundle["feature_cols"])
        self.target_path = self.root / "targets.json"
        self.source_path = self.root / "sources.json"
        self.features_path = self.root / "features.csv"
        self.output_dir = self.root / "output"
        self.targets = self._make_targets()
        self.sources = self._make_sources()
        self.feature_rows = self._make_feature_rows()
        self._write_inputs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_targets(self) -> dict:
        records = []
        for race_no in range(1, 13):
            records.append(
                {
                    "race_id": f"2026040204{race_no:02d}",
                    "race_no": race_no,
                    "target_registered": True,
                    "scheduled_post_time": f"2026-08-02T{9 + (race_no - 1) // 2:02d}:{20 if race_no % 2 == 0 else 50:02d}:00+09:00",
                    "candidate_feature_cutoff_time": "2026-08-02T08:00:00+09:00",
                }
            )
        return {
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

    def _source_record(self, race_id: str) -> dict:
        record = {
            "race_id": race_id,
            "source_contract_ok": True,
            "input_snapshot_hash": MODULE.canonical_digest({"snapshot": race_id}),
            "feature_input_max_source_event_time": "2026-08-02T07:00:00+09:00",
            "source_received_at": "2026-08-02T07:30:00+09:00",
            "candidate_feature_cutoff_time": "2026-08-02T08:00:00+09:00",
            "starter_universe_hash_at_freeze": MODULE.canonical_digest(
                {"race_id": race_id, "runners": ["1", "2", "3", "4"]}
            ),
            "runner_ids": ["1", "2", "3", "4"],
            "inference_bundle_hash": self.bundle_hash,
            "feature_schema_hash": self.feature_schema_hash,
        }
        record["source_record_hash"] = MODULE.canonical_digest(record)
        return record

    def _make_sources(self) -> dict:
        return {
            "schema_version": 1,
            "data_class": "synthetic",
            "records": [
                self._source_record(record["race_id"])
                for record in self.targets["records"]
            ],
        }

    def _make_feature_rows(self) -> list[dict[str, str]]:
        rows = []
        runner_scores = {"1": 1.2, "2": 0.9, "3": 0.3, "4": -0.4}
        for target in self.targets["records"]:
            for horse_1, horse_2, horse_3 in (
                ("1", "2", "3"),
                ("1", "2", "4"),
                ("1", "3", "4"),
                ("2", "3", "4"),
            ):
                values = [runner_scores[horse_1], runner_scores[horse_2], runner_scores[horse_3]]
                rows.append(
                    {
                        "race_id": target["race_id"],
                        "horse_id_1": horse_1,
                        "horse_id_2": horse_2,
                        "horse_id_3": horse_3,
                        "f_primary": str(sum(values)),
                        "f_balance": str(min(values)),
                        "current_odds": "",
                    }
                )
        return rows

    def _write_inputs(self) -> None:
        self.target_path.write_text(
            MODULE.canonical_json(self.targets) + "\n", encoding="utf-8"
        )
        self.source_path.write_text(
            MODULE.canonical_json(self.sources) + "\n", encoding="utf-8"
        )
        headers = [
            "race_id",
            "horse_id_1",
            "horse_id_2",
            "horse_id_3",
            "f_primary",
            "f_balance",
            "current_odds",
        ]
        with self.features_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.feature_rows)

    def _rehash_source(self, record: dict) -> None:
        record.pop("source_record_hash", None)
        record["source_record_hash"] = MODULE.canonical_digest(record)

    def _run(self) -> dict:
        return MODULE.run_adapter(
            target_manifest_path=self.target_path,
            feature_source_manifest_path=self.source_path,
            top3_feature_csv_path=self.features_path,
            inference_bundle_path=self.bundle_path,
            output_dir=self.output_dir,
            config=self.config,
            now=datetime(2026, 8, 2, 8, 5, tzinfo=ZoneInfo("Asia/Tokyo")),
            execution_mode="synthetic",
        )

    def _failure_packet(self, race_id: str) -> dict:
        return MODULE.load_json_object(
            self.output_dir / "packets" / f"{race_id}.candidate_freeze.json"
        )

    def test_complete_card_produces_twelve_safe_candidates(self) -> None:
        summary = self._run()
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["recorded_target_rows"], 12)
        self.assertEqual(summary["candidate_ready_rows"], 12)
        self.assertEqual(summary["failed_rows"], 0)
        self.assertEqual(summary["candidate_freeze_packet_ledger_completeness"], 1.0)
        packet = self._failure_packet("202604020401")
        self.assertEqual(packet["candidate_pair_key"], "1-2")
        self.assertLessEqual(packet["set_probability_mass_error"], 1e-10)
        self.assertLessEqual(packet["wide_probability_mass_error"], 1e-10)
        self.assertFalse(packet["candidate_uses_odds"])
        self.assertFalse(packet["formal_buy"])
        self.assertFalse(packet["send_order"])
        self.assertEqual(packet["stake"], 0)

    def test_missing_feature_rows_are_retained_as_failure(self) -> None:
        race_id = "202604020412"
        self.feature_rows = [row for row in self.feature_rows if row["race_id"] != race_id]
        self._write_inputs()
        summary = self._run()
        self.assertEqual(summary["status"], "IN_PROGRESS")
        self.assertEqual(summary["recorded_target_rows"], 12)
        self.assertEqual(summary["failed_rows"], 1)
        self.assertEqual(
            self._failure_packet(race_id)["failure_reason_codes"],
            ["CANDIDATE_SOURCE_NOT_READY"],
        )

    def test_forbidden_market_value_fails_only_affected_race(self) -> None:
        race_id = "202604020405"
        next(row for row in self.feature_rows if row["race_id"] == race_id)["current_odds"] = "4.2"
        self._write_inputs()
        summary = self._run()
        self.assertEqual(summary["recorded_target_rows"], 12)
        self.assertEqual(
            self._failure_packet(race_id)["failure_reason_codes"],
            ["FORBIDDEN_CANDIDATE_INPUT_COLUMN"],
        )

    def test_late_feature_source_fails_closed(self) -> None:
        race_id = "202604020406"
        record = next(row for row in self.sources["records"] if row["race_id"] == race_id)
        record["source_received_at"] = "2026-08-02T08:00:01+09:00"
        self._rehash_source(record)
        self._write_inputs()
        self._run()
        self.assertEqual(
            self._failure_packet(race_id)["failure_reason_codes"],
            ["FEATURE_SOURCE_TIME_VIOLATION"],
        )

    def test_runner_universe_mismatch_fails_closed(self) -> None:
        race_id = "202604020407"
        record = next(row for row in self.sources["records"] if row["race_id"] == race_id)
        record["runner_ids"] = ["1", "2", "3"]
        record["starter_universe_hash_at_freeze"] = MODULE.canonical_digest(
            {"race_id": race_id, "runners": record["runner_ids"]}
        )
        self._rehash_source(record)
        self._write_inputs()
        self._run()
        self.assertEqual(
            self._failure_packet(race_id)["failure_reason_codes"],
            ["STARTER_UNIVERSE_MISMATCH"],
        )

    def test_incomplete_triplet_universe_fails_probability_contract(self) -> None:
        race_id = "202604020408"
        removed = False
        kept = []
        for row in self.feature_rows:
            if row["race_id"] == race_id and not removed:
                removed = True
                continue
            kept.append(row)
        self.feature_rows = kept
        self._write_inputs()
        self._run()
        self.assertEqual(
            self._failure_packet(race_id)["failure_reason_codes"],
            ["PROBABILITY_CONTRACT_VIOLATION"],
        )

    def test_bundle_hash_mismatch_fails_closed(self) -> None:
        race_id = "202604020409"
        record = next(row for row in self.sources["records"] if row["race_id"] == race_id)
        record["inference_bundle_hash"] = "0" * 64
        self._rehash_source(record)
        self._write_inputs()
        self._run()
        self.assertEqual(
            self._failure_packet(race_id)["failure_reason_codes"],
            ["INFERENCE_BUNDLE_HASH_MISMATCH"],
        )

    def test_second_run_is_idempotent(self) -> None:
        first = self._run()
        second = self._run()
        self.assertEqual(first, second)
        ledger = MODULE.read_jsonl(self.output_dir / "candidate_freeze_ledger.jsonl")
        self.assertEqual(len(ledger), 12)
        self.assertEqual(len({record["idempotency_key"] for record in ledger}), 12)

    def test_missing_target_invalidates_before_writing(self) -> None:
        self.targets["records"].pop()
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "exactly the registered 12 races"):
            self._run()
        self.assertFalse((self.output_dir / "candidate_freeze_ledger.jsonl").exists())

    def test_real_data_requires_running_registry(self) -> None:
        registry_root = self.root / "repo"
        (registry_root / "research").mkdir(parents=True)
        (registry_root / "research" / "REGISTRY.jsonl").write_text(
            MODULE.canonical_json(
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
            MODULE.assert_real_data_authorized(registry_root, self.config["experiment_id"])


if __name__ == "__main__":
    unittest.main()
