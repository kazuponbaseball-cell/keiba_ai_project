from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config
from src.features.odds_timeline import build_odds_timeline_features_from_file, merge_odds_timeline_features
from src.utils.paths import ensure_dir, project_path


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_path(path)


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge odds timeline features into an entry snapshot CSV.")
    parser.add_argument("--config", default="config/baseline_features.json")
    parser.add_argument("--entry-csv", default="data/datasets/inference/weekly/entry_snapshot.csv")
    parser.add_argument("--timeline-csv", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    horse_number_col = "馬番"
    entry_path = _resolve(args.entry_csv)
    timeline_path = _resolve(args.timeline_csv)
    output_path = _resolve(args.output_csv) if args.output_csv else entry_path.with_name(entry_path.stem + "_with_odds_timeline.csv")

    try:
        entry = pd.read_csv(entry_path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        entry = pd.read_csv(entry_path, encoding=config["data"].get("encoding", "cp932"), low_memory=False)
    race_col = _pick_column(list(entry.columns), [race_col, "レースID(新/馬番無)", "race_id"]) or race_col
    horse_number_col = _pick_column(list(entry.columns), [horse_number_col, "馬 番", "horse_number"]) or horse_number_col
    if race_col not in entry.columns:
        raise ValueError(f"Entry CSV is missing race column. Tried config column and common aliases.")
    if horse_number_col not in entry.columns:
        raise ValueError("Entry CSV is missing horse number column. Tried 馬番, 馬 番, horse_number.")

    features = build_odds_timeline_features_from_file(timeline_path)
    merged = merge_odds_timeline_features(entry, features, race_col=race_col, horse_number_col=horse_number_col)

    ensure_dir(output_path.parent)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "entry_csv": str(entry_path),
                "timeline_csv": str(timeline_path),
                "output_csv": str(output_path),
                "rows": int(len(merged)),
                "matched_rows": int(merged["odds_latest_win"].notna().sum()) if "odds_latest_win" in merged.columns else 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
