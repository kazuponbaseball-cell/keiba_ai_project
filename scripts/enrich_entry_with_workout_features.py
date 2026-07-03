from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.workouts import build_workout_features_from_file, merge_workout_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge workout features into an entry CSV.")
    parser.add_argument("--entry-csv", required=True)
    parser.add_argument("--workout-csv", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    entry_path = Path(args.entry_csv)
    workout_path = Path(args.workout_csv)
    output_path = Path(args.output_csv) if args.output_csv else entry_path.with_name(f"{entry_path.stem}_with_workout.csv")

    entry = pd.read_csv(entry_path, encoding="utf-8-sig", low_memory=False)
    workout_features = build_workout_features_from_file(workout_path)
    race_col = _first_existing(entry, ["レースID(新/馬番無)", "race_id", "レースID"])
    horse_id_col = _first_existing(entry, ["血統登録番号", "horse_id"], required=False)
    horse_number_col = _first_existing(entry, ["馬番", "馬 番", "horse_number"], required=False)

    merged = merge_workout_features(
        entry,
        workout_features,
        race_col=race_col,
        horse_id_col=horse_id_col,
        horse_number_col=horse_number_col,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    workout_cols = [col for col in merged.columns if col.startswith("workout_")]
    matched = int(merged[workout_cols].notna().any(axis=1).sum()) if workout_cols else 0
    print(
        json.dumps(
            {
                "input_rows": int(len(entry)),
                "matched_rows": matched,
                "workout_feature_columns": workout_cols,
                "output_csv": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _first_existing(frame: pd.DataFrame, candidates: list[str], *, required: bool = True) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    if required:
        raise ValueError(f"None of these columns exist: {candidates}")
    return None


if __name__ == "__main__":
    main()
