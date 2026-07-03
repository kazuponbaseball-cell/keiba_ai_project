from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config  # noqa: E402
from src.utils.paths import ensure_dir, project_path  # noqa: E402


DEFAULT_TRAIN = "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features_with_same_day_bias_v3_retro.csv"
DEFAULT_TEST = "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv"


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def build_previous_body_weight(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    raw_path = project_path(data_cfg["historical_csv"])
    race_col = data_cfg["race_id_column"]
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]
    usecols = [race_col, horse_col, date_col, "馬体重", "馬体重増減"]
    raw = pd.read_csv(raw_path, encoding=data_cfg.get("encoding", "cp932"), usecols=usecols, low_memory=False)
    raw["_date_num"] = _num(raw[date_col])
    raw["_race_num"] = _num(raw[race_col])
    raw["_body_weight_current"] = _num(raw["馬体重"])
    raw["_body_weight_delta_current"] = _num(raw["馬体重増減"])
    raw = raw.sort_values([horse_col, "_date_num", "_race_num"], kind="mergesort")
    raw["prev_body_weight_backfilled"] = raw.groupby(horse_col, sort=False)["_body_weight_current"].shift()
    raw["prev_body_weight_delta_backfilled"] = raw.groupby(horse_col, sort=False)["_body_weight_delta_current"].shift()
    return raw[[race_col, horse_col, date_col, "prev_body_weight_backfilled", "prev_body_weight_delta_backfilled"]]


def backfill_one(path: Path, previous: pd.DataFrame, config: dict, output_dir: Path) -> dict:
    data_cfg = config["data"]
    race_col = data_cfg["race_id_column"]
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]
    frame = pd.read_csv(path, low_memory=False)
    before_weight = _num(frame.get("前走馬体重", pd.Series(index=frame.index, dtype=float))).notna().sum()
    before_delta = _num(frame.get("前走馬体重増減", pd.Series(index=frame.index, dtype=float))).notna().sum()
    merged = frame.merge(previous, on=[race_col, horse_col, date_col], how="left", validate="m:1")
    if "前走馬体重" not in merged.columns:
        merged["前走馬体重"] = pd.NA
    if "前走馬体重増減" not in merged.columns:
        merged["前走馬体重増減"] = pd.NA
    current_weight = _num(merged["前走馬体重"])
    current_delta = _num(merged["前走馬体重増減"])
    merged["前走馬体重"] = current_weight.fillna(merged["prev_body_weight_backfilled"])
    merged["前走馬体重増減"] = current_delta.fillna(merged["prev_body_weight_delta_backfilled"])
    after_weight = _num(merged["前走馬体重"]).notna().sum()
    after_delta = _num(merged["前走馬体重増減"]).notna().sum()
    merged = merged.drop(columns=["prev_body_weight_backfilled", "prev_body_weight_delta_backfilled"])
    out_path = output_dir / f"{path.stem}_body_weight_backfilled{path.suffix}"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    return {
        "input": str(path),
        "output": str(out_path),
        "rows": int(len(frame)),
        "before_prev_body_weight_nonnull": int(before_weight),
        "after_prev_body_weight_nonnull": int(after_weight),
        "before_prev_body_weight_delta_nonnull": int(before_delta),
        "after_prev_body_weight_delta_nonnull": int(after_delta),
        "added_prev_body_weight": int(after_weight - before_weight),
        "added_prev_body_weight_delta": int(after_delta - before_delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill previous body-weight columns from raw current body weight.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN)
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--output-dir", default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled")
    args = parser.parse_args()

    config = load_json_config(args.config)
    output_dir = ensure_dir(project_path(args.output_dir))
    previous = build_previous_body_weight(config)
    results = [
        backfill_one(project_path(args.train_csv), previous, config, output_dir),
        backfill_one(project_path(args.test_csv), previous, config, output_dir),
    ]
    summary = {"output_dir": str(output_dir), "files": results}
    with (output_dir / "backfill_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
