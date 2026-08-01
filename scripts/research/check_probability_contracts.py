"""Validate the Top3-set softmax and derived wide-pair probability contract.

The validator intentionally knows nothing about odds, candidate selection, value
judgement, or BUY logic.  It only checks probabilities supplied by an upstream
research artifact.  If no external runner-count column is supplied, universe
completeness is measured against the union of runners observed in the input; that
cannot detect a runner omitted from every Top3 row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO


DEFAULT_TOLERANCE = 1e-10
MAX_TOLERANCE = 1e-10
DEFAULT_RACE_COLUMN = "race_id"
DEFAULT_HORSE_COLUMNS = ("horse_1", "horse_2", "horse_3")
DEFAULT_PROBABILITY_COLUMN = "top3_probability"
CONTRACT_NAME = "top3_set_softmax_to_wide_pairs_v1"


class ProbabilityContractError(ValueError):
    """Raised when one or more probability-contract checks fail.

    ``summary`` is the same JSON-serialisable report that the CLI emits.  This
    lets callers inspect every violation without parsing an exception message.
    """

    def __init__(self, message: str, *, summary: dict[str, Any]) -> None:
        super().__init__(message)
        self.summary = summary
        self.violations = summary.get("violations", [])


def _violation(code: str, message: str, **details: Any) -> dict[str, Any]:
    violation: dict[str, Any] = {"code": code, "message": message}
    violation.update(details)
    return violation


def _canonical_identifier(value: Any, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{label} must be a string or integer")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is empty")
    return text


def _parse_probability(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean values are not probabilities")
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("probability is empty")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability is not numeric") from exc


def _parse_runner_count(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("runner count must be an integer")
    if isinstance(value, int):
        runner_count = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if not text.isascii() or not text.isdecimal():
            raise ValueError("runner count must be an integer")
        try:
            runner_count = int(text)
        except (ValueError, OverflowError) as exc:
            raise ValueError("runner count must be an integer") from exc
    else:
        raise ValueError("runner count must be an integer")
    if runner_count < 3:
        raise ValueError("runner count must be at least three")
    return runner_count


def _safe_fsum(values: Iterable[float]) -> float:
    try:
        return math.fsum(values)
    except OverflowError:
        return math.inf


def validate_probability_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    race_column: str = DEFAULT_RACE_COLUMN,
    horse_columns: Sequence[str] = DEFAULT_HORSE_COLUMNS,
    probability_column: str = DEFAULT_PROBABILITY_COLUMN,
    runner_count_column: str | None = None,
    include_pair_details: bool = False,
) -> dict[str, Any]:
    """Validate Top3-set probabilities and the wide pairs derived from them.

    Each row represents one unordered Top3 set.  Within every race, the Top3
    set probabilities must sum to one.  A set with probability ``p`` contributes
    ``p`` to each of its three unordered wide pairs, so the derived wide-pair
    probabilities must sum to three.

    The function returns a detailed summary on success.  Pair-level marginals
    are omitted unless ``include_pair_details`` is true.  With no
    ``runner_count_column``, completeness uses only the observed runner union and
    therefore cannot detect a runner omitted from every row.

    It raises
    :class:`ProbabilityContractError` on failure; the exception's ``summary``
    attribute contains the detailed failure report.
    """

    try:
        checked_tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"tolerance must be finite and between zero and {MAX_TOLERANCE:g}"
        ) from exc
    if (
        not math.isfinite(checked_tolerance)
        or checked_tolerance < 0.0
        or checked_tolerance > MAX_TOLERANCE
    ):
        raise ValueError(
            f"tolerance must be finite and between zero and {MAX_TOLERANCE:g}"
        )
    if len(horse_columns) != 3 or len(set(horse_columns)) != 3:
        raise ValueError("horse_columns must contain exactly three unique names")

    violations: list[dict[str, Any]] = []
    input_row_count = 0
    valid_row_count = 0
    probabilities_by_race: dict[str, list[float]] = defaultdict(list)
    pair_probabilities_by_race: dict[str, dict[tuple[str, str], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    seen_sets_by_race: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    observed_runners_by_race: dict[str, set[str]] = defaultdict(set)
    declared_runner_counts_by_race: dict[str, set[int]] = defaultdict(set)

    for row_number, row in enumerate(rows, start=1):
        input_row_count += 1
        if not isinstance(row, Mapping):
            violations.append(
                _violation(
                    "invalid_row",
                    "row must be a mapping",
                    row_number=row_number,
                    value_type=type(row).__name__,
                )
            )
            continue

        try:
            race_id = _canonical_identifier(row[race_column], label=race_column)
        except KeyError:
            violations.append(
                _violation(
                    "missing_column",
                    f"missing required column: {race_column}",
                    row_number=row_number,
                    column=race_column,
                )
            )
            continue
        except ValueError as exc:
            violations.append(
                _violation(
                    "invalid_race_id",
                    str(exc),
                    row_number=row_number,
                    column=race_column,
                )
            )
            continue

        # Initialise the race even if another field is invalid.  Its zero mass
        # makes the incomplete contract explicit in the final report.
        probabilities_by_race[race_id]
        pair_probabilities_by_race[race_id]
        seen_sets_by_race[race_id]
        observed_runners_by_race[race_id]
        declared_runner_counts_by_race[race_id]

        if runner_count_column is not None:
            try:
                raw_runner_count = row[runner_count_column]
            except KeyError:
                violations.append(
                    _violation(
                        "missing_column",
                        f"missing required column: {runner_count_column}",
                        row_number=row_number,
                        race_id=race_id,
                        column=runner_count_column,
                    )
                )
            else:
                try:
                    declared_runner_count = _parse_runner_count(raw_runner_count)
                except ValueError as exc:
                    violations.append(
                        _violation(
                            "invalid_runner_count",
                            str(exc),
                            row_number=row_number,
                            race_id=race_id,
                            column=runner_count_column,
                            value=repr(raw_runner_count),
                        )
                    )
                else:
                    declared_runner_counts_by_race[race_id].add(declared_runner_count)

        horses: list[str] = []
        horse_error = False
        for column in horse_columns:
            try:
                horses.append(_canonical_identifier(row[column], label=column))
            except KeyError:
                violations.append(
                    _violation(
                        "missing_column",
                        f"missing required column: {column}",
                        row_number=row_number,
                        race_id=race_id,
                        column=column,
                    )
                )
                horse_error = True
                break
            except ValueError as exc:
                violations.append(
                    _violation(
                        "invalid_horse_id",
                        str(exc),
                        row_number=row_number,
                        race_id=race_id,
                        column=column,
                    )
                )
                horse_error = True
                break
        if horse_error:
            continue
        if len(set(horses)) != 3:
            violations.append(
                _violation(
                    "invalid_top3_set",
                    "a Top3 set must contain three distinct horses",
                    row_number=row_number,
                    race_id=race_id,
                    horses=horses,
                )
            )
            continue

        top3_set = tuple(sorted(horses))
        observed_runners_by_race[race_id].update(top3_set)
        if top3_set in seen_sets_by_race[race_id]:
            violations.append(
                _violation(
                    "duplicate_top3_set",
                    "duplicate unordered Top3 set within a race",
                    row_number=row_number,
                    race_id=race_id,
                    horses=list(top3_set),
                )
            )
            continue
        # Universe completeness is structural, so record the unique set even if
        # its probability later proves invalid.
        seen_sets_by_race[race_id].add(top3_set)

        try:
            raw_probability = row[probability_column]
        except KeyError:
            violations.append(
                _violation(
                    "missing_column",
                    f"missing required column: {probability_column}",
                    row_number=row_number,
                    race_id=race_id,
                    column=probability_column,
                )
            )
            continue
        try:
            probability = _parse_probability(raw_probability)
        except ValueError as exc:
            violations.append(
                _violation(
                    "invalid_probability",
                    str(exc),
                    row_number=row_number,
                    race_id=race_id,
                    column=probability_column,
                    value=repr(raw_probability),
                )
            )
            continue
        if not math.isfinite(probability):
            violations.append(
                _violation(
                    "non_finite_probability",
                    "probability must be finite",
                    row_number=row_number,
                    race_id=race_id,
                    value=repr(raw_probability),
                )
            )
            continue
        if probability < 0.0:
            violations.append(
                _violation(
                    "negative_probability",
                    "probability must be non-negative",
                    row_number=row_number,
                    race_id=race_id,
                    value=probability,
                )
            )
            continue
        if probability > 1.0:
            violations.append(
                _violation(
                    "probability_above_one",
                    "probability must not exceed one",
                    row_number=row_number,
                    race_id=race_id,
                    value=probability,
                )
            )
            continue

        probabilities_by_race[race_id].append(probability)
        horse_a, horse_b, horse_c = top3_set
        for pair in ((horse_a, horse_b), (horse_a, horse_c), (horse_b, horse_c)):
            pair_probabilities_by_race[race_id][pair] += probability
        valid_row_count += 1

    if input_row_count == 0:
        violations.append(_violation("empty_input", "no probability rows were supplied"))

    race_summaries: dict[str, dict[str, Any]] = {}
    top3_errors: list[float] = []
    wide_errors: list[float] = []
    for race_id in sorted(probabilities_by_race):
        top3_mass = _safe_fsum(probabilities_by_race[race_id])
        pair_probabilities = pair_probabilities_by_race[race_id]
        wide_mass = _safe_fsum(pair_probabilities.values())
        top3_error = abs(top3_mass - 1.0)
        wide_error = abs(wide_mass - 3.0)
        top3_errors.append(top3_error)
        wide_errors.append(wide_error)

        observed_runner_count = len(observed_runners_by_race[race_id])
        declared_counts = declared_runner_counts_by_race[race_id]
        contract_runner_count: int | None
        if runner_count_column is None:
            contract_runner_count = observed_runner_count
            runner_universe_source = "observed_union"
        elif len(declared_counts) == 1:
            contract_runner_count = next(iter(declared_counts))
            runner_universe_source = f"column:{runner_count_column}"
        else:
            contract_runner_count = None
            runner_universe_source = f"column:{runner_count_column}"
            if len(declared_counts) > 1:
                violations.append(
                    _violation(
                        "inconsistent_runner_count",
                        "runner count is inconsistent within the race",
                        race_id=race_id,
                        values=sorted(declared_counts),
                        column=runner_count_column,
                    )
                )

        if (
            runner_count_column is not None
            and contract_runner_count is not None
            and observed_runner_count != contract_runner_count
        ):
            violations.append(
                _violation(
                    "runner_universe_size_mismatch",
                    "observed runner union does not match the declared runner count",
                    race_id=race_id,
                    observed_runner_count=observed_runner_count,
                    declared_runner_count=contract_runner_count,
                    column=runner_count_column,
                )
            )

        expected_set_count: int | None = None
        unique_set_count = len(seen_sets_by_race[race_id])
        if contract_runner_count is not None:
            expected_set_count = math.comb(contract_runner_count, 3)
            if unique_set_count != expected_set_count:
                violations.append(
                    _violation(
                        "incomplete_top3_universe",
                        "unique Top3-set count does not equal C(runner_count, 3)",
                        race_id=race_id,
                        runner_count=contract_runner_count,
                        observed_unique_set_count=unique_set_count,
                        expected_unique_set_count=expected_set_count,
                        runner_universe_source=runner_universe_source,
                    )
                )

        if top3_error > checked_tolerance:
            violations.append(
                _violation(
                    "top3_probability_mass",
                    "Top3-set probabilities do not sum to one",
                    race_id=race_id,
                    observed=top3_mass,
                    expected=1.0,
                    absolute_error=top3_error,
                    tolerance=checked_tolerance,
                )
            )
        if wide_error > checked_tolerance:
            violations.append(
                _violation(
                    "wide_pair_probability_mass",
                    "derived wide-pair probabilities do not sum to three",
                    race_id=race_id,
                    observed=wide_mass,
                    expected=3.0,
                    absolute_error=wide_error,
                    tolerance=checked_tolerance,
                )
            )

        race_summary: dict[str, Any] = {
            "top3_set_count": len(probabilities_by_race[race_id]),
            "unique_top3_set_count": unique_set_count,
            "expected_top3_set_count": expected_set_count,
            "observed_runner_count": observed_runner_count,
            "contract_runner_count": contract_runner_count,
            "runner_universe_source": runner_universe_source,
            "top3_probability_sum": top3_mass,
            "top3_mass_absolute_error": top3_error,
            "wide_pair_count": len(pair_probabilities),
            "wide_pair_probability_sum": wide_mass,
            "wide_pair_mass_absolute_error": wide_error,
        }
        if include_pair_details:
            race_summary["wide_pair_probabilities"] = [
                {
                    "horse_1": pair[0],
                    "horse_2": pair[1],
                    "probability": pair_probabilities[pair],
                }
                for pair in sorted(pair_probabilities)
            ]
        race_summaries[race_id] = race_summary

    summary: dict[str, Any] = {
        "contract": CONTRACT_NAME,
        "contract_ok": not violations,
        "tolerance": checked_tolerance,
        "input_row_count": input_row_count,
        "valid_row_count": valid_row_count,
        "race_count": len(race_summaries),
        "runner_universe_source": (
            "observed_union"
            if runner_count_column is None
            else f"column:{runner_count_column}"
        ),
        "runner_universe_limitation": (
            "Observed-union completeness cannot detect a runner omitted from every Top3 row. "
            "Supply runner_count_column for an external count check."
            if runner_count_column is None
            else None
        ),
        "pair_details_included": include_pair_details,
        "max_top3_mass_absolute_error": max(top3_errors, default=None),
        "max_wide_pair_mass_absolute_error": max(wide_errors, default=None),
        "races": race_summaries,
        "violations": violations,
    }
    if violations:
        raise ProbabilityContractError(
            f"probability contract failed with {len(violations)} violation(s)",
            summary=summary,
        )
    return summary


def _read_csv(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, strict=True)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            yield row


def _load_rows(path: Path, input_format: str | None) -> Iterator[dict[str, Any]]:
    selected_format = input_format
    if selected_format is None:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            selected_format = "csv"
        elif suffix in {".jsonl", ".ndjson"}:
            selected_format = "jsonl"
        else:
            raise ValueError("cannot infer input format; use --input-format csv or jsonl")
    if selected_format == "csv":
        return _read_csv(path)
    return _read_jsonl(path)


def _write_json(payload: Mapping[str, Any], stream: TextIO) -> None:
    json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Top3-set softmax mass and derived wide-pair mass."
    )
    parser.add_argument("input", type=Path, help="CSV or JSONL probability artifact")
    parser.add_argument("--input-format", choices=("csv", "jsonl"))
    parser.add_argument("--race-column", default=DEFAULT_RACE_COLUMN)
    parser.add_argument(
        "--horse-columns",
        nargs=3,
        metavar=("HORSE_1", "HORSE_2", "HORSE_3"),
        default=DEFAULT_HORSE_COLUMNS,
    )
    parser.add_argument("--probability-column", default=DEFAULT_PROBABILITY_COLUMN)
    parser.add_argument(
        "--runner-count-column",
        help=(
            "optional external runner-count column; without it, completeness uses the "
            "observed runner union and cannot detect runners absent from every row"
        ),
    )
    parser.add_argument(
        "--include-pair-details",
        action="store_true",
        help="include every derived wide-pair marginal in the JSON summary",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"mass tolerance in [0, {MAX_TOLERANCE:g}] (default: {DEFAULT_TOLERANCE:g})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        rows = _load_rows(args.input, args.input_format)
        summary = validate_probability_rows(
            rows,
            tolerance=args.tolerance,
            race_column=args.race_column,
            horse_columns=args.horse_columns,
            probability_column=args.probability_column,
            runner_count_column=args.runner_count_column,
            include_pair_details=args.include_pair_details,
        )
    except ProbabilityContractError as exc:
        _write_json(exc.summary, sys.stdout)
        return 1
    except (OSError, ValueError, csv.Error) as exc:
        _write_json(
            {
                "contract": CONTRACT_NAME,
                "contract_ok": False,
                "error_type": "input_error",
                "message": str(exc),
            },
            sys.stdout,
        )
        return 2
    _write_json(summary, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
