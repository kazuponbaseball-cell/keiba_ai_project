from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.loaders import load_historical_csv, load_json_config, required_columns
from src.features.baseline import prepare_training_frame
from src.utils.paths import ensure_dir, project_path
from src.utils.runtime_config import load_runtime_config


def _existing(cols: list[str], available: list[str]) -> list[str]:
    return [col for col in cols if col in available]


def build_normalized_tables(
    runtime_config_path: str = "config/data_pipeline.json",
    feature_config_path: str | None = None,
    frame: pd.DataFrame | None = None,
) -> dict[str, object]:
    runtime = load_runtime_config(runtime_config_path)
    feature_config = load_json_config(
        feature_config_path or runtime["pipeline"]["baseline_feature_config"]
    )
    if frame is None:
        raw = load_historical_csv(feature_config, columns=required_columns(feature_config, for_prediction=True))
        frame = prepare_training_frame(raw, feature_config)

    data_cfg = feature_config["data"]
    race_col = data_cfg["race_id_column"]
    horse_col = data_cfg["horse_id_column"]
    horse_name_col = data_cfg["horse_name_column"]
    date_col = data_cfg["date_column"]
    rank_col = data_cfg["rank_column"]

    out_root = ensure_dir(project_path(runtime["normalized"]["root_dir"]))

    race_cols = _existing(
        [
            race_col,
            date_col,
            "日付S",
            "場所",
            "Ｒ",
            "レース名",
            "発走時刻",
            "芝・ダ",
            "距離",
            "クラス名",
            "頭数",
            "出走頭数",
            "馬場状態",
            "競走種別",
        ],
        list(frame.columns),
    )
    races = frame[race_cols].drop_duplicates(subset=[race_col]).sort_values([date_col, race_col])

    runner_cols = _existing(
        [
            race_col,
            horse_col,
            horse_name_col,
            date_col,
            "場所",
            "レース名",
            "芝・ダ",
            "距離",
            "馬番",
            "枠番",
            "年齢",
            "性別",
            "斤量",
            "騎手コード",
            "調教師コード",
            "単勝オッズ",
            "人気",
        ],
        list(frame.columns),
    )
    runners = frame[runner_cols].copy()

    result_cols = _existing(
        [
            race_col,
            horse_col,
            horse_name_col,
            date_col,
            rank_col,
            "target_win",
            "target_top3",
            "target_score",
            "着差タイム",
            "上り3F",
            "上り3F順",
            "前4角.1",
            "人気",
            "単勝オッズ",
        ],
        list(frame.columns),
    )
    results = frame[result_cols].copy()

    latest_horses = (
        frame.sort_values([horse_col, date_col, race_col], kind="mergesort")
        .groupby(horse_col, as_index=False)
        .tail(1)
    )
    horse_cols = _existing(
        [
            horse_col,
            horse_name_col,
            date_col,
            "年齢",
            "性別",
            "キャリア",
            "場所",
            "芝・ダ",
            "距離",
            "騎手コード",
            "調教師コード",
            "horse_turf_starts",
            "horse_turf_top3_rate",
            "horse_dirt_starts",
            "horse_dirt_top3_rate",
        ],
        list(latest_horses.columns),
    )
    horses = latest_horses[horse_cols].copy()

    paths = {
        "races": out_root / runtime["normalized"]["races_file"],
        "runners": out_root / runtime["normalized"]["runners_file"],
        "results": out_root / runtime["normalized"]["results_file"],
        "horses": out_root / runtime["normalized"]["horses_file"],
    }
    races.to_csv(paths["races"], index=False, encoding="utf-8-sig")
    runners.to_csv(paths["runners"], index=False, encoding="utf-8-sig")
    results.to_csv(paths["results"], index=False, encoding="utf-8-sig")
    horses.to_csv(paths["horses"], index=False, encoding="utf-8-sig")

    summary = {
        "normalized_root": str(out_root),
        "rows": {
            "races": int(len(races)),
            "runners": int(len(runners)),
            "results": int(len(results)),
            "horses": int(len(horses)),
        },
        "files": {name: str(path) for name, path in paths.items()},
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized race/runner/result tables from the historical CSV.")
    parser.add_argument("--runtime-config", default="config/data_pipeline.json")
    parser.add_argument("--feature-config", default=None)
    args = parser.parse_args()
    summary = build_normalized_tables(args.runtime_config, args.feature_config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
