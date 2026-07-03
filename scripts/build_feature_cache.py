from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_historical_csv, load_json_config, required_columns
from src.evaluate.evaluate_baseline import EXTRA_COLUMNS
from src.features.baseline import prepare_training_frame, split_by_recent_dates
from src.utils.paths import ensure_dir, project_path


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_path(path)


def _fill_from_pedigree_master(raw: pd.DataFrame, pedigree_master_path: str) -> pd.DataFrame:
    master_path = _resolve_project_path(pedigree_master_path)
    master = pd.read_csv(master_path, dtype=str, encoding="utf-8-sig")
    horse_id_column = "血統登録番号"
    if horse_id_column not in raw.columns:
        raise ValueError(f"Cannot merge pedigree master because {horse_id_column} is missing from raw data.")
    if horse_id_column not in master.columns:
        raise ValueError(f"Cannot merge pedigree master because {horse_id_column} is missing from {master_path}.")

    fill_columns = [column for column in ["種牡馬", "母馬", "母父馬", "母母馬", "父父馬", "父母馬"] if column in master.columns]
    master = master[[horse_id_column, *fill_columns]].drop_duplicates(horse_id_column, keep="last")
    master = master.rename(columns={column: f"{column}__pedigree" for column in fill_columns})

    raw = raw.copy()
    raw[horse_id_column] = raw[horse_id_column].astype(str)
    master[horse_id_column] = master[horse_id_column].astype(str)
    merged = raw.merge(master, on=horse_id_column, how="left")
    for column in fill_columns:
        pedigree_column = f"{column}__pedigree"
        if column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna() & (merged[column].astype(str).str.strip() != ""), merged[pedigree_column])
        else:
            merged[column] = merged[pedigree_column]
        merged = merged.drop(columns=[pedigree_column])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached train/test feature CSVs for faster model experiments.")
    parser.add_argument("--config", default="config/baseline_features.json")
    parser.add_argument("--output-dir", default="data/datasets/cache/pace_style_time")
    parser.add_argument("--pedigree-master", default=None, help="Optional TARGET pedigree master CSV to fill bloodline fields.")
    args = parser.parse_args()

    config = load_json_config(args.config)
    columns = set(required_columns(config, for_prediction=True))
    columns.update(EXTRA_COLUMNS)

    raw = load_historical_csv(config, columns=sorted(columns))
    if args.pedigree_master:
        raw = _fill_from_pedigree_master(raw, args.pedigree_master)
    frame = prepare_training_frame(raw, config)
    train_df, test_df, cutoff = split_by_recent_dates(frame, config)

    output_dir = ensure_dir(project_path(args.output_dir))
    train_path = output_dir / "train_features.csv"
    test_path = output_dir / "test_features.csv"
    metadata_path = output_dir / "metadata.json"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    metadata = {
        "config": args.config,
        "temporal_test_cutoff_date": int(cutoff),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
