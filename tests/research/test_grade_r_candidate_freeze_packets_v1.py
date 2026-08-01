from __future__ import annotations

import csv
import importlib
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
        snapshot_hash = MODULE.file_sha256(self.features_path)
        for record in self.sources["records"]:
            record["input_snapshot_hash"] = snapshot_hash
            self._rehash_source(record)
        self.source_path.write_text(
            MODULE.canonical_json(self.sources) + "\n", encoding="utf-8"
        )

    def _rehash_source(self, record: dict) -> None:
        record.pop("source_record_hash", None)
        record["source_record_hash"] = MODULE.canonical_digest(record)

    def _run(self) -> dict:
        return MODULE.run_adapter(
            target_manifest_path=self.target_path,
            feature_source_manifest_path=self.source_path,
            top3_feature_csv_path=self.features_path,
            runner_feature_csv_path=None,
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

    def test_runner_rows_build_the_frozen_m1c_feature_schema(self) -> None:
        feature_cols = [
            "sum_primary_strength",
            "triplet_min_ability_floor",
            "triplet_second_min_ability_floor",
            "triplet_mean_ability_floor",
            "triplet_min_recent_stability",
            "triplet_second_min_recent_stability",
            "triplet_mean_recent_weighted",
            "triplet_mean_condition_recent",
            "triplet_experience_risk_count",
            "triplet_growth_zone_count",
            "m1c_any_core_missing",
            "triplet_min_pair_joint_fit",
            "triplet_max_pair_clash",
            "triplet_max_shared_failure",
            "triplet_max_pair_scenario_variance",
        ]
        bundle = {"feature_cols": feature_cols}
        runner_rows = []
        for horse_no, ai_score, front, closer in (
            ("1", "82", "0.8", "0.1"),
            ("2", "75", "0.6", "0.2"),
            ("3", "61", "0.2", "0.7"),
            ("4", "52", "0.1", "0.8"),
        ):
            runner_rows.append(
                {
                    "race_id": "202604020401",
                    "horse_no": horse_no,
                    "ai_score": ai_score,
                    "ai_rank": horse_no,
                    "front_running_tendency": front,
                    "closing_tendency": closer,
                    "race_front_runner_count": "2",
                    "race_early_pressure_score": "0.62",
                    "race_pace_collapse_risk": "0.56",
                    "race_slow_pace_risk": "0.10",
                    "ability_floor_score_5": "0.55",
                    "ability_stability_score_3": "0.60",
                    "recent_weighted_score_3": "0.58",
                    "condition_adjusted_recent_ability_score": "0.57",
                    "career_shallow_flag": "0",
                    "career_growth_zone_flag": "1",
                }
            )
        generated = MODULE.build_top3_features_from_runner_rows(runner_rows, bundle)
        self.assertEqual(len(generated["202604020401"]), 4)
        self.assertEqual(
            set(feature_cols).difference(generated["202604020401"][0]), set()
        )
        self.assertEqual(generated["202604020401"][0]["m1c_any_core_missing"], "0.0")

    def test_sanitize_entry_snapshot_blanks_current_market_fields(self) -> None:
        import pandas as pd

        raw_path = self.root / "raw_entry.csv"
        sanitized_path = self.root / "sanitized_entry.csv"
        rows = []
        for race_no in range(1, 13):
            for horse_no in range(1, 4):
                rows.append(
                    {
                        "Ｒ": race_no,
                        "馬番": horse_no,
                        "血統登録番号": f"{race_no:02d}{horse_no:02d}",
                        "レースID(新/馬番無)": f"old-{race_no}",
                        "単勝オッズ": "4.2",
                        "人気": "1",
                    }
                )
        pd.DataFrame(rows).to_csv(raw_path, index=False, encoding="utf-8-sig")
        summary = MODULE._sanitize_entry_snapshot(
            raw_entry_path=raw_path,
            output_path=sanitized_path,
            targets=self.targets["records"],
            baseline_config={"data": {"race_id_column": "レースID(新/馬番無)"}},
        )
        sanitized = pd.read_csv(sanitized_path, encoding="utf-8-sig", dtype=str)
        self.assertEqual(summary["race_count"], 12)
        self.assertTrue(sanitized["単勝オッズ"].fillna("").eq("").all())
        self.assertTrue(sanitized["人気"].fillna("").eq("").all())
        self.assertEqual(
            set(sanitized["race_id"]),
            {record["race_id"] for record in self.targets["records"]},
        )

    def test_sanitize_entry_snapshot_retains_missing_races_in_summary(self) -> None:
        import pandas as pd

        raw_path = self.root / "partial_entry.csv"
        sanitized_path = self.root / "partial_entry.sanitized.csv"
        pd.DataFrame(
            [
                {
                    "Ｒ": 1,
                    "馬番": horse_no,
                    "血統登録番号": f"01{horse_no:02d}",
                    "レースID(新/馬番無)": "old-1",
                }
                for horse_no in range(1, 4)
            ]
        ).to_csv(raw_path, index=False, encoding="utf-8-sig")
        summary = MODULE._sanitize_entry_snapshot(
            raw_entry_path=raw_path,
            output_path=sanitized_path,
            targets=self.targets["records"],
            baseline_config={"data": {"race_id_column": "レースID(新/馬番無)"}},
        )
        self.assertEqual(summary["race_count"], 1)
        self.assertEqual(len(summary["missing_race_ids"]), 11)
        self.assertNotIn("202604020401", summary["missing_race_ids"])

    def test_public_entry_metadata_parsers_are_deterministic(self) -> None:
        self.assertEqual(MODULE._parse_race_class("3勝クラス"), "3勝")
        self.assertEqual(MODULE._parse_race_class("リステッド (L)"), "OP(L)")
        self.assertEqual(MODULE._parse_going("芝1600m / 馬場:稍重"), "稍")
        self.assertEqual(MODULE._parse_track_code("芝1600m (外)", "芝"), "8")
        self.assertEqual(MODULE._parse_track_code("ダ1800m", "ダ"), "1")

    def test_clean_worktree_data_loader_shim_supplies_inference_contract(self) -> None:
        MODULE._install_data_loader_shim()
        loaders = importlib.import_module("src.data.loaders")
        baseline = MODULE.load_json_object(ROOT / "config" / "baseline_features.json")
        required = loaders.inference_required_columns(baseline)
        optional = loaders.inference_optional_columns(baseline)
        self.assertIn(baseline["data"]["race_id_column"], required)
        self.assertIn(baseline["data"]["abnormal_column"], optional)

    def test_public_entry_capture_uses_fixed_card_and_blanks_market(self) -> None:
        import pandas as pd

        MODULE._install_data_loader_shim()
        fetcher = importlib.import_module("scripts.fetch_netkeiba_entries_snapshot")
        runner_rows = "".join(
            f"""
            <tr class="HorseList">
              <td class="Waku{horse_no}">{horse_no}</td>
              <td class="Umaban{horse_no}">{horse_no}</td>
              <td class="HorseInfo"><span class="HorseName"><a href="/horse/202010000{horse_no}/">horse-{horse_no}</a></span></td>
              <td class="Barei">牡4</td>
              <td class="Jockey"><a href="/jockey/result/recent/0100{horse_no}/">jockey</a></td>
              <td>57.0</td>
              <td class="Trainer"><a href="/trainer/result/recent/0100{horse_no}/">trainer</a></td>
              <td><span id="odds-{horse_no}">4.2</span><span id="ninki-{horse_no}">1</span></td>
            </tr>
            """
            for horse_no in range(1, 4)
        )
        fixture = f"""
        <html><body>
          <div class="RaceName">Synthetic Race 出馬表</div>
          <div class="RaceData01">09:50発走 / 芝1600m (外) 馬場:良</div>
          <div class="RaceData02">3頭 / 1勝クラス</div>
          <table class="ShutubaTable">{runner_rows}</table>
        </body></html>
        """
        raw_path = self.root / "captured_entry.csv"
        capture_manifest_path = self.root / "capture_manifest.json"
        original = fetcher._read_url
        fetcher._read_url = lambda *args, **kwargs: fixture
        try:
            summary = MODULE.capture_public_entry_snapshot(
                target_manifest_path=self.target_path,
                baseline_config_path=ROOT / "config" / "baseline_features.json",
                output_csv_path=raw_path,
                capture_manifest_path=capture_manifest_path,
                cache_dir=self.root / "cache",
                config=self.config,
                refresh=False,
                sleep_seconds=0.0,
            )
        finally:
            fetcher._read_url = original
        captured = pd.read_csv(raw_path, encoding="utf-8-sig", dtype=str)
        manifest = MODULE.load_json_object(capture_manifest_path)
        self.assertEqual(summary["captured_races"], 12)
        self.assertEqual(summary["runner_rows"], 36)
        self.assertEqual(len(manifest["records"]), 12)
        self.assertTrue(captured["単勝オッズ"].fillna("").eq("").all())
        self.assertTrue(captured["人気"].fillna("").eq("").all())
        self.assertEqual(set(captured["場所"]), {"新潟"})

    def test_precomputed_runner_snapshot_builds_hash_bound_source_manifest(self) -> None:
        import pandas as pd

        raw_path = self.root / "raw_entry.csv"
        enriched_path = self.root / "enriched_runner.csv"
        runner_path = self.root / "runner_snapshot.csv"
        manifest_path = self.root / "feature_source_manifest.json"
        raw_path.write_text("synthetic entry snapshot\n", encoding="utf-8")
        rows = []
        for race_no in range(1, 13):
            race_id = f"2026040204{race_no:02d}"
            for horse_no in range(1, 5):
                rows.append(
                    {
                        "race_id": race_id,
                        "horse_no": horse_no,
                        "horse_id": f"{race_no:02d}{horse_no:02d}",
                        "horse_name": f"horse-{race_no}-{horse_no}",
                        "ai_score": 100 - horse_no,
                        "ai_rank": horse_no,
                        "expected_pace": "middle",
                        "front_running_tendency": 0.1 * horse_no,
                        "closing_tendency": 0.1 * (5 - horse_no),
                        "race_front_runner_count": 2,
                        "race_early_pressure_score": 0.5,
                        "ability_floor_score_5": 0.55,
                        "ability_stability_score_3": 0.60,
                        "recent_weighted_score_3": 0.58,
                        "current_odds": "",
                    }
                )
        pd.DataFrame(rows).to_csv(enriched_path, index=False, encoding="utf-8-sig")
        summary = MODULE.prepare_runner_snapshot(
            target_manifest_path=self.target_path,
            raw_entry_path=raw_path,
            inference_bundle_path=self.bundle_path,
            runner_output_path=runner_path,
            source_manifest_path=manifest_path,
            work_dir=self.root / "work",
            config=self.config,
            precomputed_enriched_runner_path=enriched_path,
            source_observed_at=datetime(
                2026, 8, 2, 7, 30, tzinfo=ZoneInfo("Asia/Tokyo")
            ),
        )
        manifest = MODULE.load_json_object(manifest_path)
        runner = pd.read_csv(runner_path, encoding="utf-8-sig", dtype=str)
        self.assertEqual(summary["target_races"], 12)
        self.assertEqual(summary["source_contract_ok_races"], 12)
        self.assertEqual(len(manifest["records"]), 12)
        self.assertNotIn("current_odds", runner.columns)
        self.assertEqual(manifest["runner_snapshot_sha256"], MODULE.file_sha256(runner_path))
        for record in manifest["records"]:
            claimed = record["source_record_hash"]
            payload = {key: value for key, value in record.items() if key != "source_record_hash"}
            self.assertEqual(claimed, MODULE.canonical_digest(payload))


if __name__ == "__main__":
    unittest.main()
