from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ABILITY = load_module(
    "enrich_prediction_basic_ability_features_freshness_test",
    ROOT / "scripts" / "enrich_prediction_basic_ability_features.py",
)
ADAPTER = load_module(
    "build_grade_r_candidate_freeze_packets_freshness_test",
    ROOT / "scripts" / "research" / "build_grade_r_candidate_freeze_packets_v1.py",
)


class RecentResultGlobTests(unittest.TestCase):
    def test_absolute_glob_expands_wildcard_directory_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = (
                root
                / "outputs"
                / "analysis"
                / "race_day_review_20260801"
                / "parsed_results_all_horses.csv"
            )
            expected.parent.mkdir(parents=True)
            expected.write_text("race_id,horse_no,finish\n", encoding="utf-8")
            pattern = str(
                root
                / "outputs"
                / "analysis"
                / "race_day_review_*"
                / "parsed_results_all_horses.csv"
            )

            self.assertEqual(ABILITY.expand_globs([pattern]), [expected.resolve()])

    def test_relative_glob_uses_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = root / "data" / "run_20260801" / "result.csv"
            expected.parent.mkdir(parents=True)
            expected.write_text("race_id,horse_no,finish\n", encoding="utf-8")
            original_root = ABILITY.ROOT
            ABILITY.ROOT = root
            try:
                actual = ABILITY.expand_globs(["data/run_*/result.csv"])
            finally:
                ABILITY.ROOT = original_root

            self.assertEqual(actual, [expected.resolve()])


class RecentResultFreshnessContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "runner_snapshot_contract": {
                "recent_result_freshness": {
                    "required": True,
                    "minimum_matched_file_count": 1,
                    "minimum_joined_rows": 1,
                    "minimum_history_date": "20260801",
                    "maximum_history_date": "20260801",
                }
            }
        }

    def test_fresh_summary_passes(self) -> None:
        ADAPTER._validate_recent_result_freshness(
            {
                "matched_recent_result_file_count": 9,
                "recent_result_history_rows": 1234,
                "history_date_max": "20260801",
            },
            self.config,
        )

    def test_zero_matches_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ADAPTER.CandidateContractError, "matched_recent_result_file_count"
        ):
            ADAPTER._validate_recent_result_freshness(
                {
                    "matched_recent_result_file_count": 0,
                    "recent_result_history_rows": 0,
                    "history_date_max": "20260215",
                },
                self.config,
            )

    def test_stale_history_fails_closed(self) -> None:
        with self.assertRaisesRegex(ADAPTER.CandidateContractError, "20260215"):
            ADAPTER._validate_recent_result_freshness(
                {
                    "matched_recent_result_file_count": 9,
                    "recent_result_history_rows": 1234,
                    "history_date_max": "20260215",
                },
                self.config,
            )

    def test_current_day_history_fails_closed(self) -> None:
        with self.assertRaisesRegex(ADAPTER.CandidateContractError, "20260802"):
            ADAPTER._validate_recent_result_freshness(
                {
                    "matched_recent_result_file_count": 9,
                    "recent_result_history_rows": 1234,
                    "history_date_max": "20260802",
                },
                self.config,
            )


if __name__ == "__main__":
    unittest.main()
