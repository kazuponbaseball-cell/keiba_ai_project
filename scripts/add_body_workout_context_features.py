from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.body_workout_context import add_body_workout_context_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Add body-weight context and past good-run workout match features.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--race-col", default="レースID(新/馬番無)")
    parser.add_argument("--date-col", default="日付")
    parser.add_argument("--horse-col", default="血統登録番号")
    parser.add_argument("--rank-col", default="確定着順")
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv, encoding="utf-8-sig", low_memory=False)
    enriched = add_body_workout_context_features(
        frame,
        race_col=args.race_col,
        date_col=args.date_col,
        horse_col=args.horse_col,
        rank_col=args.rank_col,
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False, encoding="utf-8-sig")
    added = [c for c in enriched.columns if c.startswith("body_") or c.startswith("horse_goodrun_") or c.startswith("horse_last_goodrun_") or c == "horse_workout_past_goodrun_match_score"]
    print(json.dumps({"rows": int(len(enriched)), "added_columns": added, "output_csv": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
