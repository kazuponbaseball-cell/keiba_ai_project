from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "research" / "build_grade_r_candidate_freeze_packets_v1.py"
SPEC = importlib.util.spec_from_file_location("target_manifest_projection_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TargetManifestProjectionContractTest(unittest.TestCase):
    PROJECTIONS = {
        "niigata": {
            "source": "research/drafts/EXP-20260802-007-real-card-target-manifest.json",
            "projected": "research/drafts/EXP-20260802-010-niigata-target-manifest.json",
            "config": "config/grade_r_candidate_freeze_niigata_20260802_v5.json",
        },
        "sapporo": {
            "source": "research/drafts/EXP-20260802-008-sapporo-target-manifest.json",
            "projected": "research/drafts/EXP-20260802-010-sapporo-target-manifest.json",
            "config": "config/grade_r_candidate_freeze_sapporo_20260802_v3.json",
        },
        "chukyo": {
            "source": "research/drafts/EXP-20260802-008-chukyo-target-manifest.json",
            "projected": "research/drafts/EXP-20260802-010-chukyo-target-manifest.json",
            "config": "config/grade_r_candidate_freeze_chukyo_20260802_v3.json",
        },
    }

    @staticmethod
    def _load(path: str) -> dict:
        return json.loads((ROOT / path).read_text(encoding="utf-8"))

    def test_projection_changes_only_experiment_and_cohort(self) -> None:
        all_race_ids: list[str] = []
        for projection in self.PROJECTIONS.values():
            source = self._load(projection["source"])
            projected = self._load(projection["projected"])

            self.assertEqual(projected["experiment_id"], "EXP-20260802-010")
            self.assertNotEqual(projected["cohort_id"], source["cohort_id"])

            source_immutable = copy.deepcopy(source)
            projected_immutable = copy.deepcopy(projected)
            for payload in (source_immutable, projected_immutable):
                payload.pop("experiment_id")
                payload.pop("cohort_id")
            self.assertEqual(projected_immutable, source_immutable)

            records = projected["records"]
            self.assertEqual(len(records), 12)
            self.assertEqual(len({record["race_id"] for record in records}), 12)
            all_race_ids.extend(record["race_id"] for record in records)

        self.assertEqual(len(all_race_ids), 36)
        self.assertEqual(len(set(all_race_ids)), 36)

    def test_projected_manifests_match_successor_configs(self) -> None:
        for projection in self.PROJECTIONS.values():
            config = MODULE.load_adapter_config(ROOT / projection["config"])
            projected = self._load(projection["projected"])
            records = MODULE.validate_target_manifest(projected, config)

            self.assertEqual(len(records), 12)
            self.assertEqual([record["race_no"] for record in records], list(range(1, 13)))
            self.assertFalse(projected["target_selection_uses_odds"])
            self.assertFalse(projected["formal_buy"])
            self.assertFalse(projected["send_order"])
            self.assertEqual(projected["stake"], 0)


if __name__ == "__main__":
    unittest.main()
