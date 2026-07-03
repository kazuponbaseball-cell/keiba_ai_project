from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.workouts import build_workout_features_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build workout/training features from a workout CSV.")
    parser.add_argument("--workout-csv", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    workout_path = Path(args.workout_csv)
    output_path = Path(args.output_csv) if args.output_csv else workout_path.with_name(f"{workout_path.stem}_features.csv")
    features = build_workout_features_from_file(workout_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": int(len(features)), "output_csv": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
