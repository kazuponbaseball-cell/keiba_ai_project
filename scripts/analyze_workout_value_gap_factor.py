from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _num(values: pd.Series) -> pd.Series:
    if values.dtype == object or str(values.dtype).startswith("string"):
        values = (
            values.astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(values, errors="coerce")


def _metrics(df: pd.DataFrame, *, label: str, race_col: str) -> dict[str, Any]:
    rows = len(df)
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
    rank = _num(df["確定着順"])
    win = rank == 1
    top3 = rank <= 3
    win_pay = _num(df["単勝配当"]).fillna(0.0).where(win, 0.0)
    place_pay = _num(df["複勝配当"]).fillna(0.0).where(top3, 0.0)
    return {
        "segment": label,
        "rows": int(rows),
        "races": int(df[race_col].nunique()),
        "win_rate": float(win.mean()),
        "top3_rate": float(top3.mean()),
        "avg_popularity": float(_num(df["人気"]).mean()),
        "avg_ai_rank": float(_num(df["ai_rank"]).mean()),
        "win_roi": float(win_pay.sum() / (rows * 100.0)),
        "place_roi": float(place_pay.sum() / (rows * 100.0)),
    }


def add_value_gap_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    popularity = _num(out["人気"])
    ai_rank = _num(out["ai_rank"])
    grade_score = _num(out["grade_score"])
    high_grade = out["grade"].isin(["S", "A"])
    mid_grade = out["grade"].isin(["S", "A", "B"])
    out["workout_value_gap"] = popularity - ai_rank
    out["workout_high_value_gap"] = out["workout_value_gap"].where(high_grade, 0.0)
    out["workout_value_score"] = grade_score + out["workout_value_gap"].clip(lower=0) * 0.5
    out["workout_value_gap_3plus_flag"] = ((out["workout_value_gap"] >= 3) & high_grade).astype(float)
    out["workout_value_gap_5plus_flag"] = ((out["workout_value_gap"] >= 5) & high_grade).astype(float)
    out["workout_ai_top3_value_flag"] = ((ai_rank <= 3) & high_grade & (out["workout_value_gap"] >= 2)).astype(float)
    out["workout_ai_top3_big_value_flag"] = ((ai_rank <= 3) & high_grade & (out["workout_value_gap"] >= 4)).astype(float)
    out["workout_mid_grade_value_flag"] = ((ai_rank <= 3) & mid_grade & (out["workout_value_gap"] >= 3)).astype(float)
    return out


def summarize(df: pd.DataFrame, *, race_col: str) -> pd.DataFrame:
    high_grade = df["grade"].isin(["S", "A"])
    segments = {
        "AI1_all": df[df["ai_rank"] == 1],
        "AI1_high_grade": df[(df["ai_rank"] == 1) & high_grade],
        "AI1_high_grade_value_gap_1plus": df[(df["ai_rank"] == 1) & high_grade & (df["workout_value_gap"] >= 1)],
        "AI1_high_grade_value_gap_3plus": df[(df["ai_rank"] == 1) & high_grade & (df["workout_value_gap"] >= 3)],
        "AI3_high_grade": df[(df["ai_rank"] <= 3) & high_grade],
        "AI3_high_grade_value_gap_2plus": df[(df["ai_rank"] <= 3) & high_grade & (df["workout_value_gap"] >= 2)],
        "AI3_high_grade_value_gap_3plus": df[(df["ai_rank"] <= 3) & high_grade & (df["workout_value_gap"] >= 3)],
        "AI3_high_grade_value_gap_4plus": df[(df["ai_rank"] <= 3) & high_grade & (df["workout_value_gap"] >= 4)],
        "AI3_high_grade_value_gap_5plus": df[(df["ai_rank"] <= 3) & high_grade & (df["workout_value_gap"] >= 5)],
        "AI3_high_grade_pop5plus": df[(df["ai_rank"] <= 3) & high_grade & (_num(df["人気"]) >= 5)],
        "AI3_high_grade_pop7plus": df[(df["ai_rank"] <= 3) & high_grade & (_num(df["人気"]) >= 7)],
        "value_score_7plus": df[df["workout_value_score"] >= 7],
        "value_score_8plus": df[df["workout_value_score"] >= 8],
    }
    return pd.DataFrame([_metrics(part, label=name, race_col=race_col) for name, part in segments.items()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a workout value-gap factor using AI rank, market popularity, and workout knowledge.")
    parser.add_argument("--input-csv", default="outputs/analysis/workout_knowledge_scored_test_2023plus.csv")
    parser.add_argument("--output-detail", default="outputs/analysis/workout_value_gap_scored_test_2023plus.csv")
    parser.add_argument("--output-summary", default="outputs/analysis/workout_value_gap_summary_2023plus.csv")
    parser.add_argument("--race-col", default="レースID(新/馬番無)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig", low_memory=False)
    scored = add_value_gap_factors(df)
    summary = summarize(scored, race_col=args.race_col)

    detail_path = Path(args.output_detail)
    summary_path = Path(args.output_summary)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(json.dumps({"detail": str(detail_path), "summary": str(summary_path), "rows": len(scored)}, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
