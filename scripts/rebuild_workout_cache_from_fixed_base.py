from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def date_mask(frame: pd.DataFrame, date_col: str, min_date: str | None, max_date: str | None) -> pd.Series:
    dates = pd.to_numeric(frame[date_col], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if min_date:
        mask &= dates >= normalize_date(min_date)
    if max_date:
        mask &= dates <= normalize_date(max_date)
    return mask


def normalize_date(value: str) -> int:
    text = str(value).strip()
    if len(text) == 8 and text.startswith("20"):
        text = text[2:]
    return int(text)


def merge_one(
    *,
    fixed_base_csv: Path,
    old_workout_csv: Path,
    output_csv: Path,
    race_col: str,
    horse_col: str,
    date_col: str,
    min_date: str | None,
    max_date: str | None,
) -> dict[str, Any]:
    fixed = pd.read_csv(fixed_base_csv, encoding="utf-8-sig", low_memory=False)
    fixed = fixed[date_mask(fixed, date_col, min_date, max_date)].copy()
    old_header = pd.read_csv(old_workout_csv, encoding="utf-8-sig", nrows=0)
    workout_cols = [c for c in old_header.columns if c.startswith("workout_")]
    keep_cols = [race_col, horse_col, *workout_cols]
    old = pd.read_csv(old_workout_csv, encoding="utf-8-sig", usecols=keep_cols, low_memory=False)
    old = old.drop_duplicates([race_col, horse_col], keep="last")

    overlap_workout_cols = [c for c in workout_cols if c in fixed.columns]
    if overlap_workout_cols:
        fixed = fixed.drop(columns=overlap_workout_cols)
    merged = fixed.merge(old, on=[race_col, horse_col], how="left", validate="m:1")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")

    matched = merged[workout_cols].notna().any(axis=1) if workout_cols else pd.Series(False, index=merged.index)
    return {
        "fixed_base_csv": str(fixed_base_csv),
        "old_workout_csv": str(old_workout_csv),
        "output_csv": str(output_csv),
        "rows": int(len(merged)),
        "workout_columns": int(len(workout_cols)),
        "workout_matched_rows": int(matched.sum()),
        "workout_matched_rate": float(matched.mean()) if len(matched) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild workout-enriched cache by applying existing workout columns to a "
            "pedigree-leakage-fixed base cache. This avoids re-scanning all workout files; "
            "workout columns are independent of the pedigree same-race history fix."
        )
    )
    parser.add_argument("--fixed-base-dir", required=True)
    parser.add_argument("--old-workout-dir", default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/baseline_features.json")
    parser.add_argument("--race-col", default=None)
    parser.add_argument("--horse-col", default=None)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--min-race-date", default="230101")
    parser.add_argument("--max-race-date", default="260613")
    args = parser.parse_args()

    fixed_base_dir = project_path(args.fixed_base_dir)
    old_workout_dir = project_path(args.old_workout_dir)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    race_col = args.race_col or config["data"]["race_id_column"]
    horse_col = args.horse_col or config["data"]["horse_id_column"]
    date_col = args.date_col or config["data"]["date_column"]

    train = merge_one(
        fixed_base_csv=fixed_base_dir / "train_features.csv",
        old_workout_csv=old_workout_dir / "train_features.csv",
        output_csv=output_dir / "train_features.csv",
        race_col=race_col,
        horse_col=horse_col,
        date_col=date_col,
        min_date=args.min_race_date,
        max_date=args.max_race_date,
    )
    test = merge_one(
        fixed_base_csv=fixed_base_dir / "test_features.csv",
        old_workout_csv=old_workout_dir / "test_features.csv",
        output_csv=output_dir / "test_features.csv",
        race_col=race_col,
        horse_col=horse_col,
        date_col=date_col,
        min_date=args.min_race_date,
        max_date=args.max_race_date,
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "fixed_base_plus_existing_workout_columns",
        "pedigree_fix_scope": "base historical/pedigree features regenerated after src/features/baseline.py same-race history fix",
        "workout_scope": "reused existing workout_* columns; workout extraction is independent of pedigree history aggregation",
        "train": train,
        "test": test,
    }
    (output_dir / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
