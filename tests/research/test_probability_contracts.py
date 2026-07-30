from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "check_probability_contracts.py"
)
SPEC = importlib.util.spec_from_file_location("check_probability_contracts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
contracts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contracts)


def row(race_id: str, horses: tuple[str, str, str], probability: object) -> dict[str, object]:
    return {
        "race_id": race_id,
        "horse_1": horses[0],
        "horse_2": horses[1],
        "horse_3": horses[2],
        "top3_probability": probability,
    }


class ProbabilityContractTests(unittest.TestCase):
    def test_valid_rows_preserve_both_probability_masses(self) -> None:
        rows = [
            row("R1", ("A", "B", "C"), 0.4),
            row("R1", ("A", "B", "D"), 0.3),
            row("R1", ("A", "C", "D"), 0.2),
            row("R1", ("B", "C", "D"), 0.1),
            row("R2", ("1", "2", "3"), 1.0),
        ]

        summary = contracts.validate_probability_rows(rows)

        self.assertTrue(summary["contract_ok"])
        self.assertEqual(summary["race_count"], 2)
        self.assertAlmostEqual(summary["races"]["R1"]["top3_probability_sum"], 1.0)
        self.assertAlmostEqual(summary["races"]["R1"]["wide_pair_probability_sum"], 3.0)
        self.assertAlmostEqual(summary["races"]["R2"]["top3_probability_sum"], 1.0)
        self.assertAlmostEqual(summary["races"]["R2"]["wide_pair_probability_sum"], 3.0)
        self.assertNotIn("wide_pair_probabilities", summary["races"]["R1"])
        self.assertEqual(summary["runner_universe_source"], "observed_union")
        self.assertIn("cannot detect", summary["runner_universe_limitation"])

    def test_derived_unordered_pair_marginals_are_reported(self) -> None:
        rows = [
            row("R1", ("A", "B", "C"), 0.4),
            row("R1", ("D", "B", "A"), 0.3),
            row("R1", ("A", "D", "C"), 0.2),
            row("R1", ("D", "C", "B"), 0.1),
        ]

        summary = contracts.validate_probability_rows(rows, include_pair_details=True)
        pair_rows = summary["races"]["R1"]["wide_pair_probabilities"]
        marginals = {
            (pair["horse_1"], pair["horse_2"]): pair["probability"] for pair in pair_rows
        }

        expected = {
            ("A", "B"): 0.7,
            ("A", "C"): 0.6,
            ("A", "D"): 0.5,
            ("B", "C"): 0.5,
            ("B", "D"): 0.4,
            ("C", "D"): 0.3,
        }
        self.assertEqual(set(marginals), set(expected))
        for pair, probability in expected.items():
            self.assertAlmostEqual(marginals[pair], probability)
        self.assertAlmostEqual(math.fsum(marginals.values()), 3.0)

    def test_top3_mass_failure_also_exposes_derived_wide_mass_failure(self) -> None:
        rows = [
            row("R1", ("A", "B", "C"), 0.6),
            row("R1", ("A", "B", "D"), 0.3),
        ]

        with self.assertRaises(contracts.ProbabilityContractError) as caught:
            contracts.validate_probability_rows(rows)

        codes = {item["code"] for item in caught.exception.violations}
        self.assertIn("top3_probability_mass", codes)
        self.assertIn("wide_pair_probability_mass", codes)
        self.assertAlmostEqual(
            caught.exception.summary["races"]["R1"]["top3_probability_sum"], 0.9
        )
        self.assertAlmostEqual(
            caught.exception.summary["races"]["R1"]["wide_pair_probability_sum"], 2.7
        )

    def test_negative_and_non_finite_probabilities_fail(self) -> None:
        cases = (
            (-0.01, "negative_probability"),
            (float("nan"), "non_finite_probability"),
            (float("inf"), "non_finite_probability"),
            (float("-inf"), "non_finite_probability"),
        )
        for probability, expected_code in cases:
            with self.subTest(probability=probability):
                with self.assertRaises(contracts.ProbabilityContractError) as caught:
                    contracts.validate_probability_rows(
                        [row("R1", ("A", "B", "C"), probability)]
                    )
                codes = {item["code"] for item in caught.exception.violations}
                self.assertIn(expected_code, codes)

    def test_duplicate_set_is_order_invariant(self) -> None:
        rows = [
            row("R1", ("A", "B", "C"), 0.5),
            row("R1", ("C", "A", "B"), 0.5),
        ]

        with self.assertRaises(contracts.ProbabilityContractError) as caught:
            contracts.validate_probability_rows(rows)

        codes = {item["code"] for item in caught.exception.violations}
        self.assertIn("duplicate_top3_set", codes)
        self.assertEqual(caught.exception.summary["valid_row_count"], 1)

    def test_tolerance_cannot_be_widened_to_bypass_contract(self) -> None:
        rows = [row("R1", ("A", "B", "C"), 0.5)]

        with self.assertRaisesRegex(ValueError, "between zero"):
            contracts.validate_probability_rows(rows, tolerance=3.0)

    def test_incomplete_observed_top3_universe_fails(self) -> None:
        rows = [
            row("R1", ("A", "B", "C"), 0.4),
            row("R1", ("A", "B", "D"), 0.3),
            row("R1", ("A", "C", "D"), 0.3),
            # B/C/D is deliberately absent.  The observed union still has all
            # four runners, and the probability masses themselves are valid.
        ]

        with self.assertRaises(contracts.ProbabilityContractError) as caught:
            contracts.validate_probability_rows(rows)

        codes = {item["code"] for item in caught.exception.violations}
        self.assertIn("incomplete_top3_universe", codes)
        self.assertNotIn("top3_probability_mass", codes)
        self.assertNotIn("wide_pair_probability_mass", codes)
        race_summary = caught.exception.summary["races"]["R1"]
        self.assertEqual(race_summary["observed_runner_count"], 4)
        self.assertEqual(race_summary["unique_top3_set_count"], 3)
        self.assertEqual(race_summary["expected_top3_set_count"], 4)

    def test_external_runner_count_detects_fully_omitted_runner(self) -> None:
        only_set = row("R1", ("A", "B", "C"), 1.0)
        only_set["runner_count"] = 4

        with self.assertRaises(contracts.ProbabilityContractError) as caught:
            contracts.validate_probability_rows(
                [only_set], runner_count_column="runner_count"
            )

        codes = {item["code"] for item in caught.exception.violations}
        self.assertIn("runner_universe_size_mismatch", codes)
        self.assertIn("incomplete_top3_universe", codes)
        self.assertIsNone(caught.exception.summary["runner_universe_limitation"])

    def test_json_identifiers_reject_boolean_and_float_values(self) -> None:
        bad_race = row("R1", ("A", "B", "C"), 1.0)
        bad_race["race_id"] = True
        bad_horse = row("R2", ("A", "B", "C"), 1.0)
        bad_horse["horse_1"] = 1.5

        with self.assertRaises(contracts.ProbabilityContractError) as caught:
            contracts.validate_probability_rows([bad_race, bad_horse])

        codes = {item["code"] for item in caught.exception.violations}
        self.assertIn("invalid_race_id", codes)
        self.assertIn("invalid_horse_id", codes)

    def test_csv_cli_returns_zero_and_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "probabilities.csv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "race_id",
                        "horse_1",
                        "horse_2",
                        "horse_3",
                        "top3_probability",
                    ),
                )
                writer.writeheader()
                writer.writerow(row("R1", ("A", "B", "C"), 1.0))

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = contracts.main([str(input_path)])

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["contract_ok"])
        self.assertEqual(payload["input_row_count"], 1)
        self.assertFalse(payload["pair_details_included"])
        self.assertNotIn("wide_pair_probabilities", payload["races"]["R1"])

    def test_jsonl_cli_returns_one_for_contract_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "probabilities.jsonl"
            with input_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(row("R1", ("A", "B", "C"), 0.75)) + "\n")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = contracts.main([str(input_path)])

        self.assertEqual(exit_code, 1)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["contract_ok"])
        codes = {item["code"] for item in payload["violations"]}
        self.assertIn("top3_probability_mass", codes)
        self.assertIn("wide_pair_probability_mass", codes)

    def test_cli_rejects_huge_finite_probability_with_structured_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "probabilities.jsonl"
            with input_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(row("R1", ("A", "B", "C"), 1e308)) + "\n")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = contracts.main([str(input_path)])

        self.assertEqual(exit_code, 1)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["contract_ok"], False)
        codes = {item["code"] for item in payload["violations"]}
        self.assertIn("probability_above_one", codes)

    def test_cli_rejects_overbroad_tolerance_as_input_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "probabilities.jsonl"
            with input_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(row("R1", ("A", "B", "C"), 0.5)) + "\n")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = contracts.main([str(input_path), "--tolerance", "3.0"])

        self.assertEqual(exit_code, 2)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["error_type"], "input_error")
        self.assertIn("between zero", payload["message"])

    def test_csv_reader_streams_and_csv_error_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "broken.csv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(
                    "race_id,horse_1,horse_2,horse_3,top3_probability\n"
                    'R1,"unterminated,B,C,1.0\n'
                )

            rows = contracts._load_rows(input_path, None)
            self.assertNotIsInstance(rows, list)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = contracts.main([str(input_path)])

        self.assertEqual(exit_code, 2)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["contract_ok"], False)
        self.assertEqual(payload["error_type"], "input_error")


if __name__ == "__main__":
    unittest.main()
