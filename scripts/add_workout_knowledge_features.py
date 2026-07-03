from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.workout_knowledge_features import (
    KNOWLEDGE_CATEGORICAL_FEATURES,
    KNOWLEDGE_NUMERIC_FEATURES,
    add_workout_knowledge_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add trainer-specific workout knowledge features to a race-entry CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--workouts-csv", default="data/processed/target/workouts_20230101_20260613.csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--lookback-days", type=int, default=21)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv, encoding="utf-8-sig", low_memory=False)
    workouts = pd.read_csv(args.workouts_csv, encoding="utf-8-sig", low_memory=False)
    enriched = add_workout_knowledge_features(frame, workouts, lookback_days=args.lookback_days)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False, encoding="utf-8-sig")

    added = [col for col in KNOWLEDGE_NUMERIC_FEATURES + KNOWLEDGE_CATEGORICAL_FEATURES if col in enriched.columns]
    print(
        json.dumps(
            {
                "rows": int(len(enriched)),
                "added_columns": added,
                "output_csv": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
