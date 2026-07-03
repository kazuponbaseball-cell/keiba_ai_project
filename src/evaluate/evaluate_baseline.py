from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loaders import load_historical_csv, load_json_config, required_columns
from src.features.baseline import add_pace_scenario_scores, prepare_training_frame, split_by_recent_dates
from src.utils.paths import ensure_dir, project_path


EXTRA_COLUMNS = [
    "人気",
    "単勝オッズ",
    "単勝配当",
    "複勝配当",
    "芝・ダ",
    "距離",
    "場所",
    "クラス名",
    "頭数",
    "出走頭数",
]


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def _distance_category(values: pd.Series) -> pd.Series:
    numeric = _num(values)
    bins = [-np.inf, 1300, 1600, 2000, 2400, np.inf]
    labels = ["sprint", "mile", "middle", "classic", "long"]
    return pd.cut(numeric, bins=bins, labels=labels).astype("string").fillna("unknown")


def _return_metrics(df: pd.DataFrame, race_col: str, *, label: str) -> dict[str, Any]:
    rows = len(df)
    races = df[race_col].nunique() if race_col in df else 0
    if rows == 0:
        return {
            "segment": label,
            "rows": 0,
            "races": 0,
            "win_rate": 0.0,
            "top3_rate": 0.0,
            "avg_popularity": None,
            "avg_ai_rank": None,
            "win_roi": 0.0,
            "place_roi": 0.0,
        }

    win_pay = _num(df.get("単勝配当", pd.Series(0, index=df.index))).fillna(0.0).where(df["target_win"] == 1, 0.0)
    place_pay = _num(df.get("複勝配当", pd.Series(0, index=df.index))).fillna(0.0).where(df["target_top3"] == 1, 0.0)
    return {
        "segment": label,
        "rows": int(rows),
        "races": int(races),
        "win_rate": float(df["target_win"].mean()),
        "top3_rate": float(df["target_top3"].mean()),
        "avg_popularity": float(_num(df["人気"]).mean()) if "人気" in df else None,
        "avg_ai_rank": float(df["ai_rank"].mean()) if "ai_rank" in df else None,
        "win_roi": float(win_pay.sum() / (rows * 100.0)),
        "place_roi": float(place_pay.sum() / (rows * 100.0)),
    }


def _summarize_by_rank(df: pd.DataFrame, rank_col: str, race_col: str, *, max_rank: int = 10) -> pd.DataFrame:
    rows = []
    for rank in range(1, max_rank + 1):
        part = df[df[rank_col] == rank]
        rows.append(_return_metrics(part, race_col, label=f"{rank_col}_{rank}"))
    return pd.DataFrame(rows)


def _summarize_segments(df: pd.DataFrame, race_col: str) -> pd.DataFrame:
    segments = {
        "all_test_rows": df,
        "ai_top1": df[df["ai_rank"] == 1],
        "ai_top3": df[df["ai_rank"] <= 3],
        "favorite_top1": df[df["pop_rank"] == 1],
        "favorite_top3": df[df["pop_rank"] <= 3],
        "ai_top3_pop5plus": df[(df["ai_rank"] <= 3) & (_num(df["人気"]) >= 5)],
        "ai_top3_pop7plus": df[(df["ai_rank"] <= 3) & (_num(df["人気"]) >= 7)],
        "ai_over_market_3plus": df[df["ai_pop_gap"] <= -3],
        "ai_over_market_5plus": df[df["ai_pop_gap"] <= -5],
        "danger_favorite_ai6plus": df[(_num(df["人気"]) <= 3) & (df["ai_rank"] >= 6)],
        "confident_ai_top1_gap_005": df[(df["ai_rank"] == 1) & (df["ai_score_gap_to_second"] >= 0.05)],
        "confident_ai_top1_gap_010": df[(df["ai_rank"] == 1) & (df["ai_score_gap_to_second"] >= 0.10)],
    }
    return pd.DataFrame([_return_metrics(part, race_col, label=name) for name, part in segments.items()])


def _summarize_group(df: pd.DataFrame, group_col: str, race_col: str) -> pd.DataFrame:
    rows = []
    for value, part in df.groupby(group_col, dropna=False):
        rows.append(_return_metrics(part, race_col, label=str(value)))
    out = pd.DataFrame(rows)
    out.insert(0, "group", group_col)
    return out.sort_values(["group", "segment"])


def _race_level_metrics(df: pd.DataFrame, race_col: str) -> dict[str, float]:
    ai_top3 = df[df["ai_rank"] <= 3].groupby(race_col)["target_win"].max()
    pop_top3 = df[df["pop_rank"] <= 3].groupby(race_col)["target_win"].max()
    winners = df[df["target_win"] == 1]
    return {
        "races": float(df[race_col].nunique()),
        "rows": float(len(df)),
        "ai_top3_contains_winner_rate": float(ai_top3.mean()) if len(ai_top3) else 0.0,
        "pop_top3_contains_winner_rate": float(pop_top3.mean()) if len(pop_top3) else 0.0,
        "winner_mean_ai_rank": float(winners["ai_rank"].mean()) if len(winners) else 0.0,
        "winner_mean_pop_rank": float(winners["pop_rank"].mean()) if len(winners) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline AI rank versus popularity and returns.")
    parser.add_argument("--config", default="config/baseline_features.json")
    parser.add_argument("--model", default="models/baseline/baseline_ranker.pkl")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--test-csv", default=None, help="Use a prepared temporal test feature CSV instead of rebuilding features.")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    date_col = config["data"]["date_column"]

    if args.test_csv:
        test_df = pd.read_csv(project_path(args.test_csv), low_memory=False)
        cutoff = int(pd.to_numeric(test_df[date_col], errors="coerce").min())
    else:
        columns = set(required_columns(config, for_prediction=True))
        columns.update(EXTRA_COLUMNS)
        raw = load_historical_csv(config, columns=sorted(columns))
        frame = prepare_training_frame(raw, config)
        _train_df, test_df, cutoff = split_by_recent_dates(frame, config)

    with project_path(args.model).open("rb") as f:
        model = pickle.load(f)

    test = test_df.copy()
    test["ai_score"] = model.predict(test)
    test["ai_rank"] = test.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    test = add_pace_scenario_scores(test)
    test["pop_rank"] = test.groupby(race_col)["人気"].rank(ascending=True, method="first").astype(int)
    test["ai_pop_gap"] = test["ai_rank"] - test["pop_rank"]
    test["distance_category_eval"] = _distance_category(test["距離"])

    top_scores = test.groupby(race_col)["ai_score"].transform("max")
    second_scores = test[test["ai_rank"] == 2].set_index(race_col)["ai_score"]
    test["_second_score"] = test[race_col].map(second_scores)
    test["ai_score_gap_to_second"] = (top_scores - test["_second_score"]).fillna(0.0)
    test = test.drop(columns=["_second_score"])

    output_dir = ensure_dir(project_path(args.output_dir))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(output_dir / stamp)

    segment_summary = _summarize_segments(test, race_col)
    ai_rank_summary = _summarize_by_rank(test, "ai_rank", race_col, max_rank=10)
    pop_rank_summary = _summarize_by_rank(test, "pop_rank", race_col, max_rank=10)
    surface_summary = _summarize_group(test, "芝・ダ", race_col)
    distance_summary = _summarize_group(test, "distance_category_eval", race_col)
    pace_summary = _summarize_group(test, "expected_pace", race_col)
    class_summary = _summarize_group(test, "クラス名", race_col)

    detail_cols = [
        race_col,
        "日付S",
        "場所",
        "Ｒ",
        "レース名",
        "馬番",
        "馬名",
        "ai_rank",
        "ai_score",
        "expected_pace",
        "slow_ai_score",
        "middle_ai_score",
        "fast_ai_score",
        "front_running_tendency",
        "closing_tendency",
        "race_front_runner_count",
        "race_early_pressure_score",
        "pop_rank",
        "人気",
        "単勝オッズ",
        "確定着順",
        "単勝配当",
        "複勝配当",
        "ai_pop_gap",
        "芝・ダ",
        "距離",
        "distance_category_eval",
    ]
    detail = test[[col for col in detail_cols if col in test.columns]].sort_values([race_col, "ai_rank"])

    segment_summary.to_csv(run_dir / "segment_summary.csv", index=False, encoding="utf-8-sig")
    ai_rank_summary.to_csv(run_dir / "ai_rank_summary.csv", index=False, encoding="utf-8-sig")
    pop_rank_summary.to_csv(run_dir / "popularity_rank_summary.csv", index=False, encoding="utf-8-sig")
    surface_summary.to_csv(run_dir / "surface_summary.csv", index=False, encoding="utf-8-sig")
    distance_summary.to_csv(run_dir / "distance_summary.csv", index=False, encoding="utf-8-sig")
    pace_summary.to_csv(run_dir / "pace_summary.csv", index=False, encoding="utf-8-sig")
    class_summary.to_csv(run_dir / "class_summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(run_dir / "prediction_detail.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "run_dir": str(run_dir),
        "config": args.config,
        "model": args.model,
        "temporal_test_cutoff_date": int(cutoff),
        "race_level_metrics": _race_level_metrics(test, race_col),
        "top_segments": segment_summary.to_dict(orient="records"),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
