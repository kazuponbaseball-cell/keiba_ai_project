from __future__ import annotations

import argparse
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


Condition = tuple[str, Callable[[pd.DataFrame], pd.Series]]


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def _metrics(df: pd.DataFrame, race_col: str, label: str) -> dict[str, float | int | str | None]:
    rows = len(df)
    if rows == 0:
        return {
            "segment": label,
            "bets": 0,
            "races": 0,
            "win_rate": 0.0,
            "top3_rate": 0.0,
            "avg_popularity": None,
            "avg_ai_rank": None,
            "win_roi": 0.0,
            "place_roi": 0.0,
        }
    win_pay = _num(df["単勝配当"]).fillna(0.0).where(df["target_win"] == 1, 0.0)
    place_pay = _num(df["複勝配当"]).fillna(0.0).where(df["target_top3"] == 1, 0.0)
    return {
        "segment": label,
        "bets": int(rows),
        "races": int(df[race_col].nunique()),
        "win_rate": float(df["target_win"].mean()),
        "top3_rate": float(df["target_top3"].mean()),
        "avg_popularity": float(_num(df["人気"]).mean()),
        "avg_ai_rank": float(df["ai_rank"].mean()),
        "win_roi": float(win_pay.sum() / (rows * 100.0)),
        "place_roi": float(place_pay.sum() / (rows * 100.0)),
    }


def _add_ranks(df: pd.DataFrame, model_path: Path, race_col: str) -> pd.DataFrame:
    with model_path.open("rb") as f:
        model = pickle.load(f)
    out = df.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    out["pop_rank"] = out.groupby(race_col)["人気"].rank(ascending=True, method="first").astype(int)
    out["ai_pop_gap"] = out["ai_rank"] - out["pop_rank"]
    return out


def _conditions(df: pd.DataFrame) -> list[Condition]:
    q = {
        "blood_hi": float(_num(df["bloodline_high_confidence_fit_score"]).quantile(0.75)),
        "blood_top": float(_num(df["bloodline_high_confidence_fit_score"]).quantile(0.90)),
        "lift_hi": float(_num(df["bloodline_lift_fit_score"]).quantile(0.75)),
        "pair_hi": float(_num(df["bloodline_pair_fit_score"]).quantile(0.75)),
        "pace_hi": float(_num(df["pace_fit_score"]).quantile(0.75)),
        "draw_hi": float(_num(df["draw_pace_fit_score"]).quantile(0.75)),
        "draw_low": float(_num(df["current_draw_advantage_score"]).quantile(0.25)),
        "collapse_hi": float(_num(df["race_pace_collapse_risk"]).quantile(0.75)),
        "slow_hi": float(_num(df["race_slow_pace_risk"]).quantile(0.75)),
        "lap_fit_hi": float(_num(df["lap_aptitude_fit_score"]).quantile(0.75)),
        "lap_fast_hi": float(_num(df["horse_fast_lap_score_past5"]).quantile(0.75)),
        "lap_slow_hi": float(_num(df["horse_slow_lap_score_past5"]).quantile(0.75)),
        "lap_instant_hi": float(_num(df["horse_instant_lap_score_past5"]).quantile(0.75)),
        "lap_sustain_hi": float(_num(df["horse_sustain_lap_score_past5"]).quantile(0.75)),
    }
    return [
        ("ai_top3", lambda x: x["ai_rank"] <= 3),
        ("ai_top5", lambda x: x["ai_rank"] <= 5),
        ("pop5plus", lambda x: _num(x["人気"]) >= 5),
        ("pop7plus", lambda x: _num(x["人気"]) >= 7),
        ("blood_hi", lambda x: _num(x["bloodline_high_confidence_fit_score"]) >= q["blood_hi"]),
        ("blood_top10", lambda x: _num(x["bloodline_high_confidence_fit_score"]) >= q["blood_top"]),
        ("blood_lift_hi", lambda x: _num(x["bloodline_lift_fit_score"]) >= q["lift_hi"]),
        ("blood_pair_hi", lambda x: _num(x["bloodline_pair_fit_score"]) >= q["pair_hi"]),
        ("blood_reliable", lambda x: _num(x["bloodline_reliability_score"]) >= 0.8),
        ("pace_fit_hi", lambda x: _num(x["pace_fit_score"]) >= q["pace_hi"]),
        ("draw_pace_hi", lambda x: _num(x["draw_pace_fit_score"]) >= q["draw_hi"]),
        ("draw_disadvantaged", lambda x: _num(x["current_draw_advantage_score"]) <= q["draw_low"]),
        ("pace_collapse_hi", lambda x: _num(x["race_pace_collapse_risk"]) >= q["collapse_hi"]),
        ("slow_pace_hi", lambda x: _num(x["race_slow_pace_risk"]) >= q["slow_hi"]),
        ("lap_fit_hi", lambda x: _num(x["lap_aptitude_fit_score"]) >= q["lap_fit_hi"]),
        ("lap_fast_hi", lambda x: _num(x["horse_fast_lap_score_past5"]) >= q["lap_fast_hi"]),
        ("lap_slow_hi", lambda x: _num(x["horse_slow_lap_score_past5"]) >= q["lap_slow_hi"]),
        ("lap_instant_hi", lambda x: _num(x["horse_instant_lap_score_past5"]) >= q["lap_instant_hi"]),
        ("lap_sustain_hi", lambda x: _num(x["horse_sustain_lap_score_past5"]) >= q["lap_sustain_hi"]),
        ("bias_recent_hi", lambda x: _num(x["bias_adjusted_recent_score"]) >= 0.70),
        ("market_gap3", lambda x: x["ai_pop_gap"] <= -3),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search combo betting segments across pedigree, bias, and pace features.")
    parser.add_argument("--config", default="config/baseline_features.json")
    parser.add_argument("--model", default="models/target_pedigree_interactions_confirmed_opponent/baseline_ranker.pkl")
    parser.add_argument("--test-csv", default="data/datasets/cache/target_pedigree_interactions_confirmed_opponent/test_features.csv")
    parser.add_argument("--output-dir", default="outputs/segment_analysis_target_pedigree_interactions")
    parser.add_argument("--min-bets", type=int, default=200)
    parser.add_argument("--max-combo-size", type=int, default=4)
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    test = _add_ranks(test, project_path(args.model), race_col)

    conditions = _conditions(test)
    rows: list[dict[str, float | int | str | None]] = []
    base_mask = pd.Series(True, index=test.index)
    rows.append(_metrics(test, race_col, "all"))
    for size in range(1, args.max_combo_size + 1):
        for combo in combinations(conditions, size):
            label = "&".join(name for name, _func in combo)
            mask = base_mask.copy()
            for _name, func in combo:
                mask &= func(test)
            part = test[mask]
            if len(part) < args.min_bets:
                continue
            rows.append(_metrics(part, race_col, label))

    out = pd.DataFrame(rows)
    out = out.sort_values(["place_roi", "win_roi", "bets"], ascending=[False, False, False])
    output_dir = ensure_dir(project_path(args.output_dir))
    summary_path = output_dir / "combo_segments.csv"
    json_path = output_dir / "top_combo_segments.json"
    out.to_csv(summary_path, index=False, encoding="utf-8-sig")
    top = out.head(30).to_dict(orient="records")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary_csv": str(summary_path), "top_segments": top}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary_csv": str(summary_path), "top_segments": top[:12]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
