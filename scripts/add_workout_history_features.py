from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.workout_history import add_workout_pattern_history_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Add prior workout pattern history features.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--race-col", default=None)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--rank-col", default=None)
    parser.add_argument("--odds-col", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv) if args.output_csv else input_path.with_name(f"{input_path.stem}_with_workout_history.csv")
    frame = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)

    race_col = args.race_col or _first_existing(frame, ["race_id", "レースID(新/馬番無)", "繝ｬ繝ｼ繧ｹID(譁ｰ/鬥ｬ逡ｪ辟｡)"])
    date_col = args.date_col or _first_existing(frame, ["date", "日付", "譌･莉・"])
    rank_col = args.rank_col or _first_existing(frame, ["finish_rank", "rank", "確定着順", "遒ｺ螳夂捩鬆・"])
    odds_col = args.odds_col or _first_existing(frame, ["win_odds", "odds", "単勝オッズ"], required=False)

    enriched = add_workout_pattern_history_features(
        frame,
        race_col=race_col,
        date_col=date_col,
        rank_col=rank_col,
        odds_col=odds_col,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False, encoding="utf-8-sig")
    feature_cols = [col for col in enriched.columns if col.startswith("workout_") and col.endswith(("_rate", "_score", "_starts", "_roi"))]
    print(
        json.dumps(
            {
                "input_rows": int(len(frame)),
                "output_rows": int(len(enriched)),
                "added_history_feature_count": len(feature_cols),
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
