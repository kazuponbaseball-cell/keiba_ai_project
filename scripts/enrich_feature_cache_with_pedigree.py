from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config
from src.features.baseline import add_bloodline_features, add_lap_aptitude_features
from src.utils.paths import ensure_dir, project_path


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_path(path)


def _fill_from_pedigree_master(frame: pd.DataFrame, pedigree_master_path: Path, horse_id_column: str) -> pd.DataFrame:
    master = pd.read_csv(pedigree_master_path, dtype=str, encoding="utf-8-sig")
    if horse_id_column not in frame.columns:
        raise ValueError(f"{horse_id_column} is missing from feature cache.")
    if horse_id_column not in master.columns:
        raise ValueError(f"{horse_id_column} is missing from {pedigree_master_path}.")

    fill_columns = [column for column in ["種牡馬", "母馬", "母父馬", "母母馬", "父父馬", "父母馬"] if column in master.columns]
    master = master[[horse_id_column, *fill_columns]].drop_duplicates(horse_id_column, keep="last")
    master = master.rename(columns={column: f"{column}__pedigree" for column in fill_columns})

    out = frame.copy()
    out[horse_id_column] = out[horse_id_column].astype(str)
    master[horse_id_column] = master[horse_id_column].astype(str)
    out = out.merge(master, on=horse_id_column, how="left")
    for column in fill_columns:
        pedigree_column = f"{column}__pedigree"
        if column in out.columns:
            out[column] = out[column].where(out[column].notna() & (out[column].astype(str).str.strip() != ""), out[pedigree_column])
        else:
            out[column] = out[pedigree_column]
        out = out.drop(columns=[pedigree_column])
    return out


def _fill_from_historical_csv(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    historical_path = _resolve_project_path(config["data"]["historical_csv"])
    race_col = config["data"]["race_id_column"]
    horse_col = config["data"]["horse_id_column"]
    encoding = config["data"].get("encoding", "cp932")
    fill_columns = ["PCI", "PCI3", "RPCI", "Ave-3F"]

    needed = [race_col, horse_col, *fill_columns]
    raw = pd.read_csv(historical_path, encoding=encoding, usecols=needed, low_memory=False)
    raw = raw.drop_duplicates([race_col, horse_col], keep="last")
    raw = raw.rename(columns={column: f"{column}__historical" for column in fill_columns})

    out = frame.copy()
    out[race_col] = out[race_col].astype(str)
    out[horse_col] = out[horse_col].astype(str)
    raw[race_col] = raw[race_col].astype(str)
    raw[horse_col] = raw[horse_col].astype(str)
    out = out.merge(raw, on=[race_col, horse_col], how="left")
    for column in fill_columns:
        historical_column = f"{column}__historical"
        if column in out.columns:
            out[column] = out[column].where(out[column].notna(), out[historical_column])
        else:
            out[column] = out[historical_column]
        out = out.drop(columns=[historical_column])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Add TARGET pedigree-derived bloodline features to an existing cache.")
    parser.add_argument("--config", default="config/baseline_features.json")
    parser.add_argument("--input-dir", default="data/datasets/cache/confirmed_opponent_form")
    parser.add_argument("--output-dir", default="data/datasets/cache/target_pedigree_confirmed_opponent")
    parser.add_argument("--pedigree-master", default="data/processed/target/pedigree_master.csv")
    args = parser.parse_args()

    config = load_json_config(args.config)
    horse_id_column = config["data"]["horse_id_column"]
    input_dir = _resolve_project_path(args.input_dir)
    output_dir = ensure_dir(_resolve_project_path(args.output_dir))
    pedigree_master_path = _resolve_project_path(args.pedigree_master)

    train_df = pd.read_csv(input_dir / "train_features.csv", encoding="utf-8-sig", low_memory=False)
    test_df = pd.read_csv(input_dir / "test_features.csv", encoding="utf-8-sig", low_memory=False)
    train_df["_cache_split"] = "train"
    test_df["_cache_split"] = "test"
    frame = pd.concat([train_df, test_df], ignore_index=True)

    frame = _fill_from_pedigree_master(frame, pedigree_master_path, horse_id_column)
    frame = _fill_from_historical_csv(frame, config)
    frame = add_lap_aptitude_features(frame, config)
    frame = add_bloodline_features(frame, config)

    train_out = frame[frame["_cache_split"] == "train"].drop(columns=["_cache_split"])
    test_out = frame[frame["_cache_split"] == "test"].drop(columns=["_cache_split"])
    train_path = output_dir / "train_features.csv"
    test_path = output_dir / "test_features.csv"
    metadata_path = output_dir / "metadata.json"
    train_out.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_out.to_csv(test_path, index=False, encoding="utf-8-sig")

    metadata = {
        "config": args.config,
        "source_cache": str(input_dir),
        "pedigree_master": str(pedigree_master_path),
        "train_rows": int(len(train_out)),
        "test_rows": int(len(test_out)),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "sire_non_null_rate": float(frame["種牡馬"].notna().mean()) if "種牡馬" in frame.columns else 0.0,
        "bms_non_null_rate": float(frame["母父馬"].notna().mean()) if "母父馬" in frame.columns else 0.0,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
