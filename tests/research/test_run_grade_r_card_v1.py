from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.research import run_grade_r_card_v1 as module


ROOT = Path(__file__).resolve().parents[2]
DIRECT_SHA256 = "b008392d3ca8af8703ad5b8308f8a90157511a0434066a6eb7b774ada656da7c"


class GradeRCardRunnerTests(unittest.TestCase):
    def _config(
        self,
        root: Path,
        *,
        history_mode: str = "target_direct",
        race_date: str = "2026-08-09",
    ) -> tuple[dict[str, object], Path, Path]:
        direct_manifest = root / "direct-history.json"
        target_manifest = root / "targets.json"
        target_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_id": "SYNTHETIC-EXP028",
                    "cohort_id": "synthetic-card",
                    "target_card": {
                        "race_date": race_date,
                        "venue_code": "99",
                        "meeting_no": 1,
                        "day_no": 1,
                    },
                    "records": [
                        {"race_id": "race-1", "race_no": 1},
                        {"race_id": "race-2", "race_no": 2},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        sources: dict[str, object] = {
            "dr": {"observed_at": "2026-08-08T19:11:06+09:00"},
            "du": {"observed_at": "2026-08-08T19:11:21+09:00"},
        }
        if history_mode == "target_direct":
            sources["direct_history_manifest"] = {
                "observed_at": "2026-08-08T21:24:02+09:00",
                "path": str(direct_manifest),
                "sha256": DIRECT_SHA256,
            }
        else:
            sources["html"] = {"observed_at": "2026-08-08T20:00:00+09:00"}
        config: dict[str, object] = {
            "schema_version": 1,
            "experiment_id": "SYNTHETIC-EXP028",
            "race_date": race_date,
            "timezone": "Asia/Tokyo",
            "bundle": {
                "sha256": "a" * 64,
                "candidate_policy_sha256": "b" * 64,
                "inference_bundle": str(root / "bundle.json"),
            },
            "candidate_policy": {
                "name": "SYNTHETIC_NON_ODDS",
                "primary_confidence_threshold": 0.2,
            },
            "cards": [
                {
                    "slug": "synthetic",
                    "cohort_id": "synthetic-card",
                    "target_manifest": str(target_manifest),
                }
            ],
            "history": {
                "baseline_config": "config/baseline_features.json",
                "baseline_model": str(root / "baseline.pkl"),
                "historical_csv": str(root / "history.csv"),
                "ability_history_dir": str(root / "ability"),
                "minimum_history_date": "20260802",
                "maximum_history_date": "20260808",
                "recent_result_globs": [],
                "entry_globs": [],
            },
            "input_contract": {
                "history_mode": history_mode,
                "expected_races": 2,
            },
            "input_sources": sources,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        return config, config_path, direct_manifest

    def _add_history_bridge(
        self,
        root: Path,
        config: dict[str, object],
    ) -> tuple[Path, Path, Path]:
        recent = root / "recent-results.csv"
        entry = root / "entry-snapshot.csv"
        pd.DataFrame(
            [
                {
                    "race_id": "2026080201010101",
                    "horse_no": "1",
                    "finish": "1",
                },
                {
                    "race_id": "2026080201010101",
                    "horse_no": "2",
                    "finish": "2",
                },
            ]
        ).to_csv(recent, index=False, encoding="utf-8-sig")
        pd.DataFrame(
            [
                {
                    "race_id": "2026080201010101",
                    "馬番": "1",
                    "horse_id": "horse-1",
                    "日付S": "20260802",
                },
                {
                    "race_id": "2026080201010101",
                    "馬番": "2",
                    "horse_id": "horse-2",
                    "日付S": "20260802",
                },
            ]
        ).to_csv(entry, index=False, encoding="utf-8-sig")
        manifest = root / "history-bridge.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_id": config["experiment_id"],
                    "artifacts": [
                        {
                            "role": "recent_results",
                            "config_history_key": "recent_result_globs",
                            "path": str(recent),
                            "sha256": module.candidate.file_sha256(recent),
                            "byte_count": recent.stat().st_size,
                            "row_count": 2,
                            "race_count": 1,
                            "race_id_column": "race_id",
                            "horse_key_column": "horse_no",
                            "date_column": None,
                            "date_from_race_id_prefix": 8,
                            "minimum_date": "20260802",
                            "maximum_date": "20260802",
                            "duplicate_race_horse_rows": 0,
                        },
                        {
                            "role": "entry_snapshot",
                            "config_history_key": "entry_globs",
                            "path": str(entry),
                            "sha256": module.candidate.file_sha256(entry),
                            "byte_count": entry.stat().st_size,
                            "row_count": 2,
                            "race_count": 1,
                            "race_id_column": "race_id",
                            "horse_key_column": "馬番",
                            "date_column": "日付S",
                            "date_from_race_id_prefix": 0,
                            "minimum_date": "20260802",
                            "maximum_date": "20260802",
                            "duplicate_race_horse_rows": 0,
                        },
                    ],
                    "formal_buy": False,
                    "send_order": False,
                    "stake": 0,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        input_contract = config["input_contract"]
        input_sources = config["input_sources"]
        history = config["history"]
        assert isinstance(input_contract, dict)
        assert isinstance(input_sources, dict)
        assert isinstance(history, dict)
        input_contract["require_history_bridge_manifest"] = True
        input_sources["history_bridge_manifest"] = {
            "path": str(manifest),
            "sha256": module.candidate.file_sha256(manifest),
        }
        history["recent_result_globs"] = [str(recent)]
        history["entry_globs"] = [str(entry)]
        return manifest, recent, entry

    @staticmethod
    def _terminal_table() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "venue": "synthetic",
                    "race_id": "race-1",
                    "race_no": 1,
                    "record_status": "CANDIDATE_READY",
                    "shadow_action": "PENDING_STRICT_T3",
                    "candidate_uses_odds": False,
                    "formal_buy": False,
                    "send_order": False,
                    "stake": 0,
                },
                {
                    "venue": "synthetic",
                    "race_id": "race-2",
                    "race_no": 2,
                    "record_status": "FAILED",
                    "shadow_action": "NO_BET_CONTRACT",
                    "candidate_uses_odds": False,
                    "formal_buy": False,
                    "send_order": False,
                    "stake": 0,
                },
            ]
        )

    def test_target_date_token_accepts_iso_and_compact_dates(self) -> None:
        self.assertEqual("20260809", module._target_date_token({"race_date": "2026-08-09"}))
        self.assertEqual("20260809", module._target_date_token({"race_date": "20260809"}))
        with self.assertRaisesRegex(ValueError, "invalid race_date"):
            module._target_date_token({"race_date": "2026/08/09"})

    def test_target_direct_observation_clock_does_not_require_html(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _, _ = self._config(Path(temporary))
        observed = module._source_observed_at(config)
        self.assertEqual(
            datetime.fromisoformat("2026-08-08T21:24:02+09:00"), observed
        )
        self.assertNotIn("html", config["input_sources"])

    def test_html_observation_clock_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _, _ = self._config(Path(temporary), history_mode="html")
        observed = module._source_observed_at(config)
        self.assertEqual(
            datetime.fromisoformat("2026-08-08T20:00:00+09:00"), observed
        )
        path, sha256 = module._validated_direct_history_override(config, None, None)
        self.assertIsNone(path)
        self.assertIsNone(sha256)

    def test_target_direct_override_requires_frozen_path_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _, direct_manifest = self._config(Path(temporary))
            with self.assertRaisesRegex(ValueError, "explicit manifest path"):
                module._validated_direct_history_override(config, None, None)
            with self.assertRaisesRegex(ValueError, "requires both path and sha256"):
                module._validated_direct_history_override(config, direct_manifest, None)
            with self.assertRaisesRegex(ValueError, "differs from frozen config"):
                module._validated_direct_history_override(
                    config, Path(temporary) / "other.json", DIRECT_SHA256
                )
            with self.assertRaisesRegex(ValueError, "differs from frozen config"):
                module._validated_direct_history_override(
                    config, direct_manifest, "1" * 64
                )
            path, sha256 = module._validated_direct_history_override(
                config, direct_manifest, DIRECT_SHA256
            )
            self.assertEqual(direct_manifest.resolve(), path)
            self.assertEqual(DIRECT_SHA256, sha256)

    def test_history_bridge_preflight_verifies_hash_shape_and_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _ = self._config(root)
            manifest, _, _ = self._add_history_bridge(root, config)
            summary = module._validate_history_bridge_manifest(config)
        self.assertTrue(summary["required"])
        self.assertTrue(summary["contract_ok"])
        self.assertEqual(manifest.resolve(), Path(summary["manifest_path"]))
        self.assertEqual(2, len(summary["artifacts"]))
        self.assertEqual(
            {"entry_snapshot", "recent_results"},
            {artifact["role"] for artifact in summary["artifacts"]},
        )

    def test_history_bridge_preflight_fails_before_import_on_artifact_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, config_path, direct_manifest = self._config(root)
            _, recent, _ = self._add_history_bridge(root, config)
            config_path.write_text(
                json.dumps(config, ensure_ascii=False), encoding="utf-8"
            )
            recent.write_text("tampered\n", encoding="utf-8")
            with (
                patch.object(module.candidate, "assert_real_data_authorized"),
                patch.object(module, "import_multicard") as importer,
                self.assertRaisesRegex(ValueError, "recent_results hash mismatch"),
            ):
                module.run(
                    config_path,
                    root / "output",
                    direct_history_manifest_path=direct_manifest,
                    direct_history_manifest_sha256=DIRECT_SHA256,
                )
            importer.assert_not_called()

    def test_candidate_adapter_source_priority_matches_history_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct, _, _ = self._config(root)
            html, _, _ = self._config(root, history_mode="html")
            manifest = json.loads((root / "targets.json").read_text(encoding="utf-8"))
        direct_adapter = module._candidate_config(direct, manifest)
        html_adapter = module._candidate_config(html, manifest)
        self.assertEqual(
            ["fixed_target_direct_dr_du_and_authority_manifest"],
            direct_adapter["runner_snapshot_contract"]["source_priority"],
        )
        self.assertEqual(
            ["fixed_target_multicard_html_and_du"],
            html_adapter["runner_snapshot_contract"]["source_priority"],
        )
        self.assertEqual([1, 2], direct_adapter["target_card"]["expected_race_numbers"])

    def test_run_propagates_manifest_and_uses_dynamic_snapshot_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, config_path, direct_manifest = self._config(root)
            output_root = root / "output"
            table = self._terminal_table()
            with (
                patch.object(module.candidate, "assert_real_data_authorized") as authorize,
                patch.object(
                    module, "import_multicard", return_value={"runner_rows": 2}
                ) as importer,
                patch.object(
                    module.candidate,
                    "load_adapter_config",
                    side_effect=lambda path: module.load_json(path),
                ),
                patch.object(
                    module.candidate,
                    "prepare_runner_snapshot",
                    return_value={"target_races": 2},
                ) as prepare,
                patch.object(
                    module,
                    "freeze_card_per_race",
                    return_value={"terminal_records": 2},
                ),
                patch.object(module, "_candidate_table", return_value=table) as candidate_table,
            ):
                summary = module.run(
                    config_path,
                    output_root,
                    direct_history_manifest_path=direct_manifest,
                    direct_history_manifest_sha256=DIRECT_SHA256,
                )

            authorize.assert_called_once_with(module.ROOT, config["experiment_id"])
            importer.assert_called_once_with(
                config_path,
                output_root,
                direct_history_manifest_path=direct_manifest.resolve(),
                direct_history_manifest_sha256=DIRECT_SHA256,
            )
            prepare_call = prepare.call_args.kwargs
            self.assertEqual("entry_snapshot_20260809.csv", prepare_call["raw_entry_path"].name)
            self.assertEqual(
                "runner_snapshot_20260809.csv",
                prepare_call["runner_output_path"].name,
            )
            self.assertEqual(
                "feature_source_manifest_20260809.json",
                prepare_call["source_manifest_path"].name,
            )
            self.assertEqual(
                datetime.fromisoformat("2026-08-08T21:24:02+09:00"),
                prepare_call["source_observed_at"],
            )
            candidate_table.assert_called_once_with(output_root, config["cards"], "20260809")
            self.assertEqual(2, summary["registered_target_rows"])
            self.assertEqual(2, summary["terminal_record_rows"])
            self.assertEqual(0, summary["missing_target_rows"])
            self.assertFalse(summary["formal_buy"])
            self.assertFalse(summary["send_order"])
            self.assertEqual(0, summary["stake"])

    def test_candidate_table_reads_only_configured_target_date(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "synthetic" / "raw_entry"
            freeze_dir = root / "synthetic" / "candidate_freeze"
            packet_dir = freeze_dir / "packets"
            packet_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"horse_id": "1", "horse_name": "Alpha"},
                    {"horse_id": "2", "horse_name": "Beta"},
                ]
            ).to_csv(raw_dir / "entry_snapshot_20260809.csv", index=False, encoding="utf-8-sig")
            packet = {
                "race_id": "race-1",
                "race_no": 1,
                "candidate_horse_id_1": "1",
                "candidate_horse_id_2": "2",
                "candidate_pair_key": "1-2",
                "p_wide_coherent_raw": 0.3,
                "p_action_calibrated": 0.31,
                "top1_top2_margin": 0.1,
                "confidence_gate_pass": True,
                "record_status": "CANDIDATE_READY",
                "failure_reason_codes": [],
            }
            (packet_dir / "race-1.json").write_text(json.dumps(packet), encoding="utf-8")
            (freeze_dir / "candidate_freeze_ledger.jsonl").write_text(
                json.dumps({"packet_path": "packets/race-1.json"}) + "\n",
                encoding="utf-8",
            )
            table = module._candidate_table(root, [{"slug": "synthetic"}], "20260809")
            self.assertEqual("Alpha", table.iloc[0]["horse_name_1"])
            self.assertEqual("Beta", table.iloc[0]["horse_name_2"])
            self.assertFalse(bool(table.iloc[0]["candidate_uses_odds"]))

    def test_candidate_denominator_fails_closed(self) -> None:
        table = self._terminal_table()
        summary = module._validate_candidate_table(table, {"race-1", "race-2"})
        self.assertEqual(2, summary["terminal_record_rows"])
        with self.assertRaisesRegex(ValueError, "terminal denominator mismatch"):
            module._validate_candidate_table(table.iloc[:1], {"race-1", "race-2"})
        unsafe = table.copy()
        unsafe.loc[0, "formal_buy"] = True
        with self.assertRaisesRegex(ValueError, "BUY/order safety"):
            module._validate_candidate_table(unsafe, {"race-1", "race-2"})

    def test_exp028_config_preserves_exp026_prediction_contract(self) -> None:
        exp026 = json.loads(
            (ROOT / "config" / "grade_r_card_20260809_exp026.json").read_text(
                encoding="utf-8"
            )
        )
        exp028 = json.loads(
            (ROOT / "config" / "grade_r_card_20260809_exp028.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("EXP-20260808-028", exp028["experiment_id"])
        self.assertEqual(exp026["bundle"], exp028["bundle"])
        self.assertEqual(exp026["candidate_policy"], exp028["candidate_policy"])
        self.assertEqual(exp026["history"], exp028["history"])
        self.assertEqual(exp026["race_date"], exp028["race_date"])
        self.assertEqual(exp026["input_sources"]["dr"], exp028["input_sources"]["dr"])
        self.assertEqual(exp026["input_sources"]["du"], exp028["input_sources"]["du"])
        self.assertNotIn("html", exp028["input_sources"])
        self.assertEqual(
            "outputs/research/EXP-20260808-027/"
            "target_direct_history_source_manifest_20260809.json",
            exp028["input_sources"]["direct_history_manifest"]["path"],
        )
        self.assertEqual(
            DIRECT_SHA256,
            exp028["input_sources"]["direct_history_manifest"]["sha256"],
        )
        self.assertFalse(exp028["safety"]["formal_buy"])
        self.assertFalse(exp028["safety"]["send_order"])
        self.assertEqual(0, exp028["safety"]["stake"])

        all_records: list[dict[str, object]] = []
        for card026, card028 in zip(exp026["cards"], exp028["cards"], strict=True):
            old_manifest = json.loads(
                (ROOT / card026["target_manifest"]).read_text(encoding="utf-8")
            )
            new_manifest = json.loads(
                (ROOT / card028["target_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual("EXP-20260808-028", new_manifest["experiment_id"])
            self.assertEqual(old_manifest["records"], new_manifest["records"])
            self.assertEqual(old_manifest["target_card"], new_manifest["target_card"])
            self.assertEqual(
                old_manifest["source_fold_manifest"],
                new_manifest["source_fold_manifest"],
            )
            self.assertFalse(new_manifest["formal_buy"])
            self.assertFalse(new_manifest["send_order"])
            self.assertEqual(0, new_manifest["stake"])
            all_records.extend(new_manifest["records"])

        self.assertEqual(36, len(all_records))
        self.assertEqual(495, sum(int(record["runner_count"]) for record in all_records))
        self.assertEqual(1, sum(record["race_domain"] == "obstacle" for record in all_records))

    def test_exp029_config_preserves_exp028_prediction_contract(self) -> None:
        exp028 = json.loads(
            (ROOT / "config" / "grade_r_card_20260809_exp028.json").read_text(
                encoding="utf-8"
            )
        )
        exp029 = json.loads(
            (ROOT / "config" / "grade_r_card_20260809_exp029.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("EXP-20260808-029", exp029["experiment_id"])
        self.assertEqual(exp028["bundle"], exp029["bundle"])
        self.assertEqual(exp028["candidate_policy"], exp029["candidate_policy"])
        self.assertEqual(exp028["race_date"], exp029["race_date"])
        self.assertEqual(exp028["safety"], exp029["safety"])
        self.assertEqual(
            exp028["input_sources"]["direct_history_manifest"],
            exp029["input_sources"]["direct_history_manifest"],
        )
        self.assertTrue(exp029["input_contract"]["require_history_bridge_manifest"])
        self.assertEqual(
            "00a77ce8e52820f8009d5d182c68241f346e5f39854e22a98df1c95d77e5c5c4",
            exp029["input_sources"]["history_bridge_manifest"]["sha256"],
        )
        bridge_path = ROOT / exp029["input_sources"]["history_bridge_manifest"]["path"]
        self.assertEqual(
            exp029["input_sources"]["history_bridge_manifest"]["sha256"],
            module.candidate.file_sha256(bridge_path),
        )
        for key in (
            "ability_history_dir",
            "baseline_config",
            "baseline_model",
            "historical_csv",
            "maximum_history_date",
            "minimum_history_date",
        ):
            self.assertEqual(exp028["history"][key], exp029["history"][key])

        all_records: list[dict[str, object]] = []
        for card028, card029 in zip(exp028["cards"], exp029["cards"], strict=True):
            old_manifest = json.loads(
                (ROOT / card028["target_manifest"]).read_text(encoding="utf-8")
            )
            new_manifest = json.loads(
                (ROOT / card029["target_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual("EXP-20260808-029", new_manifest["experiment_id"])
            self.assertEqual(old_manifest["records"], new_manifest["records"])
            self.assertEqual(old_manifest["target_card"], new_manifest["target_card"])
            self.assertEqual(
                old_manifest["source_fold_manifest"],
                new_manifest["source_fold_manifest"],
            )
            self.assertFalse(new_manifest["formal_buy"])
            self.assertFalse(new_manifest["send_order"])
            self.assertEqual(0, new_manifest["stake"])
            all_records.extend(new_manifest["records"])
        self.assertEqual(36, len(all_records))
        self.assertEqual(495, sum(int(record["runner_count"]) for record in all_records))


if __name__ == "__main__":
    unittest.main()
