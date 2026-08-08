from __future__ import annotations

import csv
import importlib.util
import json
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "research" / "build_grade_r_candidate_freeze_packets_v1.py"
SPEC = importlib.util.spec_from_file_location("multi_card_candidate_freeze_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MultiCardCandidateFreezeContractTest(unittest.TestCase):
    CARDS = {
        "sapporo": {
            "config": "config/grade_r_candidate_freeze_sapporo_20260802_v1.json",
            "target": "research/drafts/EXP-20260802-008-sapporo-target-manifest.json",
            "netkeiba": "research/drafts/EXP-20260802-008-sapporo-netkeiba-target-card.csv",
            "venue_code": "01",
        },
        "chukyo": {
            "config": "config/grade_r_candidate_freeze_chukyo_20260802_v1.json",
            "target": "research/drafts/EXP-20260802-008-chukyo-target-manifest.json",
            "netkeiba": "research/drafts/EXP-20260802-008-chukyo-netkeiba-target-card.csv",
            "venue_code": "07",
        },
    }

    def test_both_cards_are_fixed_and_valid(self) -> None:
        all_race_ids: list[str] = []
        for card in self.CARDS.values():
            config = MODULE.load_adapter_config(ROOT / card["config"])
            manifest = json.loads((ROOT / card["target"]).read_text(encoding="utf-8"))
            records = MODULE.validate_target_manifest(manifest, config)

            self.assertEqual(len(records), 12)
            self.assertEqual([record["race_no"] for record in records], list(range(1, 13)))
            self.assertTrue(all(record["race_id"][4:6] == card["venue_code"] for record in records))
            self.assertFalse(manifest["target_selection_uses_odds"])
            self.assertFalse(manifest["formal_buy"])
            self.assertFalse(manifest["send_order"])
            self.assertEqual(manifest["stake"], 0)

            for record in records:
                post = datetime.fromisoformat(record["scheduled_post_time"])
                cutoff = datetime.fromisoformat(record["candidate_feature_cutoff_time"])
                self.assertEqual((post - cutoff).total_seconds(), 15 * 60)
            all_race_ids.extend(record["race_id"] for record in records)

        self.assertEqual(len(all_race_ids), 24)
        self.assertEqual(len(set(all_race_ids)), 24)

    def test_public_entry_targets_are_fixed_before_market_access(self) -> None:
        netkeiba_ids: list[str] = []
        for card in self.CARDS.values():
            with (ROOT / card["netkeiba"]).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual(len({row["race_id"] for row in rows}), 12)
            netkeiba_ids.extend(row["race_id"] for row in rows)

        self.assertEqual(len(netkeiba_ids), 24)
        self.assertEqual(len(set(netkeiba_ids)), 24)

        fold = json.loads(
            (
                ROOT
                / "research/drafts/EXP-20260802-008-grade-r-sapporo-chukyo-candidate-freeze-v1.fold_manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(fold["selection_as_of"], "2026-08-02T08:48:52+09:00")
        self.assertFalse(fold["selection_source"]["market_fields_used"])
        self.assertEqual(sum(card["expected_race_count"] for card in fold["target_cards"]), 24)

    def test_runner_universe_and_acquisition_contracts_fail_closed(self) -> None:
        universe = json.loads(
            (
                ROOT / "research/drafts/EXP-20260802-008-runner-universe-manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(universe["target_race_ids"]), 24)
        self.assertEqual(universe["late_or_unavailable_target_action"], "FAIL_CLOSE_AND_RETAIN")
        self.assertFalse(universe["candidate_substitution_allowed"])
        self.assertFalse(universe["formal_buy"])
        self.assertFalse(universe["send_order"])
        self.assertEqual(universe["stake"], 0)

        for name in self.CARDS:
            path = (
                ROOT
                / f"research/drafts/EXP-20260802-008-{name}-runner-source-acquisition-manifest.json"
            )
            contract = json.loads(path.read_text(encoding="utf-8"))
            entry = contract["entry_source"]
            self.assertFalse(entry["target_selection_uses_odds"])
            self.assertTrue(entry["unavailable_races_remain_in_candidate_denominator"])
            self.assertTrue(entry["current_odds_and_popularity_are_blank_before_inference"])
            self.assertFalse(contract["formal_buy"])
            self.assertFalse(contract["send_order"])
            self.assertEqual(contract["stake"], 0)


if __name__ == "__main__":
    unittest.main()
