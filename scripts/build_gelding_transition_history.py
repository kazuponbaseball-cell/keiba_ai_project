from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.gelding_transition import build_gelding_history  # noqa: E402


DEFAULT_FEATURES = [
    Path(
        "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
        "body_weight_backfilled/owner_breeder_enriched/"
        "train_features_with_same_day_bias_v3_retro_body_breeder.csv"
    ),
    Path(
        "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
        "body_weight_backfilled/owner_breeder_enriched/"
        "test_features_with_same_day_bias_v3_retro_body_breeder.csv"
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact gelding-transition history table for runtime use.")
    parser.add_argument("--feature-csv", action="append", default=[], help="Historical feature CSV. Can be specified multiple times.")
    parser.add_argument("--output-csv", default="data/processed/gelding_transition/gelding_transition_history.csv")
    parser.add_argument("--summary-json", default="outputs/analysis/gelding_transition_runtime_history/summary.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    paths = [Path(p) for p in args.feature_csv] if args.feature_csv else DEFAULT_FEATURES
    paths = [p if p.is_absolute() else root / p for p in paths]
    history = build_gelding_history(paths)

    output = Path(args.output_csv)
    output = output if output.is_absolute() else root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_csv": str(output),
        "source_csvs": [str(p) for p in paths],
        "rows": int(len(history)),
        "horses": int(history["horse_id"].nunique()) if not history.empty else 0,
        "known_gelding_transition_rows": int((history["gelding_start_no_since_transition"] > 0).sum()) if not history.empty else 0,
    }
    summary_path = Path(args.summary_json)
    summary_path = summary_path if summary_path.is_absolute() else root / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
