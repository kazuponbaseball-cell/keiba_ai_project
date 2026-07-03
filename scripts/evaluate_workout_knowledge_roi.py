from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config
from src.features.workout_knowledge import evaluate_workout_knowledge, prepare_workouts_for_knowledge
from src.utils.paths import ensure_dir, project_path


def _num(values: pd.Series | Any) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if series.dtype == object:
        series = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def _to_datetime(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    text = text.where(~text.str.fullmatch(r"\d{6}", na=False), "20" + text)
    return pd.to_datetime(text, errors="coerce", format="mixed")


def _select_workouts_for_row(row: pd.Series, workouts_by_horse: dict[str, pd.DataFrame], date_col: str, lookback_days: int) -> pd.DataFrame:
    horse_id = str(row.get("血統登録番号", row.get("horse_id", ""))).replace(".0", "")
    race_date = _to_datetime(pd.Series([row.get(date_col, row.get("日付"))])).iloc[0]
    w = workouts_by_horse.get(horse_id)
    if w is None or pd.isna(race_date):
        return pd.DataFrame(columns=["workout_date_dt"])
    days = (race_date - w["workout_date_dt"]).dt.days
    return w[(days >= 0) & (days <= lookback_days)].sort_values("workout_date_dt", kind="mergesort")


def _score_knowledge(test: pd.DataFrame, workouts: pd.DataFrame, *, date_col: str, lookback_days: int) -> pd.DataFrame:
    prepared = prepare_workouts_for_knowledge(workouts)
    prepared["horse_id"] = prepared["horse_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    workouts_by_horse = {str(horse_id): part.copy() for horse_id, part in prepared.groupby("horse_id", sort=False)}

    rows: list[dict[str, Any]] = []
    for _, row in test.iterrows():
        selected = _select_workouts_for_row(row, workouts_by_horse, date_col, lookback_days)
        result = evaluate_workout_knowledge(row, selected)
        rows.append(result)
    return pd.DataFrame(rows)


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
            "avg_popularity": np.nan,
            "avg_ai_rank": np.nan,
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
        "avg_popularity": float(_num(df["人気"]).mean()) if "人気" in df else np.nan,
        "avg_ai_rank": float(df["ai_rank"].mean()) if "ai_rank" in df else np.nan,
        "win_roi": float(win_pay.sum() / (rows * 100.0)),
        "place_roi": float(place_pay.sum() / (rows * 100.0)),
    }


def _summarize(scored: pd.DataFrame, race_col: str) -> pd.DataFrame:
    grade_high = scored["grade"].isin(["S", "A"])
    grade_mid = scored["grade"].isin(["S", "A", "B"])
    grade_low = scored["grade"].isin(["C", "D"])
    segments = {
        "all_test_rows": scored,
        "grade_S": scored[scored["grade"] == "S"],
        "grade_A": scored[scored["grade"] == "A"],
        "grade_B": scored[scored["grade"] == "B"],
        "grade_C": scored[scored["grade"] == "C"],
        "grade_D": scored[scored["grade"] == "D"],
        "grade_A_or_S": scored[grade_high],
        "grade_B_or_higher": scored[grade_mid],
        "AI1_all": scored[scored["ai_rank"] == 1],
        "AI1_grade_A_or_S": scored[(scored["ai_rank"] == 1) & grade_high],
        "AI1_grade_B_or_higher": scored[(scored["ai_rank"] == 1) & grade_mid],
        "AI1_grade_C_or_lower": scored[(scored["ai_rank"] == 1) & grade_low],
        "AI1_grade_D": scored[(scored["ai_rank"] == 1) & (scored["grade"] == "D")],
        "AI3_all": scored[scored["ai_rank"] <= 3],
        "AI3_grade_A_or_S": scored[(scored["ai_rank"] <= 3) & grade_high],
        "AI3_grade_B_or_higher": scored[(scored["ai_rank"] <= 3) & grade_mid],
        "AI3_pop5plus_grade_A_or_S": scored[(scored["ai_rank"] <= 3) & grade_high & (_num(scored["人気"]) >= 5)],
        "AI3_pop7plus_grade_A_or_S": scored[(scored["ai_rank"] <= 3) & grade_high & (_num(scored["人気"]) >= 7)],
    }
    return pd.DataFrame([_return_metrics(part, race_col, label=name) for name, part in segments.items()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate workout knowledge grades against win/place ROI.")
    parser.add_argument("--config", default="config/baseline_features_workout.json")
    parser.add_argument("--model", default="models/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/baseline_ranker.pkl")
    parser.add_argument("--test-csv", default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features.csv")
    parser.add_argument("--workouts-csv", default="data/processed/target/workouts_20230101_20260613.csv")
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--summary-name", default="workout_knowledge_roi_summary_2023plus.csv")
    parser.add_argument("--detail-name", default="workout_knowledge_scored_test_2023plus.csv")
    parser.add_argument("--lookback-days", type=int, default=21)
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    date_col = config["data"]["date_column"]

    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    workouts = pd.read_csv(project_path(args.workouts_csv), low_memory=False)
    with project_path(args.model).open("rb") as f:
        model = pickle.load(f)

    scored = test.copy()
    if "target_win" not in scored.columns:
        rank = _num(scored[config["data"]["rank_column"]])
        scored["target_win"] = (rank == 1).astype(float)
        scored["target_top3"] = (rank <= 3).astype(float)

    scored["ai_score"] = model.predict(scored)
    scored["ai_rank"] = scored.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)

    knowledge = _score_knowledge(scored, workouts, date_col=date_col, lookback_days=args.lookback_days)
    for col in ["trainer", "trainer_code", "matched_pattern", "plus_factors", "minus_factors", "grade", "grade_score", "comment"]:
        scored[col] = knowledge[col].values

    summary = _summarize(scored, race_col)

    output_dir = ensure_dir(project_path(args.output_dir))
    summary_path = output_dir / args.summary_name
    detail_path = output_dir / args.detail_name
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    detail_cols = [
        race_col,
        date_col,
        "場所",
        "Ｒ",
        "レース名",
        "馬番",
        "馬名",
        "人気",
        "単勝オッズ",
        "確定着順",
        "単勝配当",
        "複勝配当",
        "ai_rank",
        "ai_score",
        "trainer",
        "trainer_code",
        "grade",
        "grade_score",
        "matched_pattern",
        "plus_factors",
        "minus_factors",
        "comment",
    ]
    scored[[col for col in detail_cols if col in scored.columns]].to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(json.dumps({"summary": str(summary_path), "detail": str(detail_path), "rows": len(scored)}, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
