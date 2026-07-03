from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


MaskFunc = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Condition:
    name: str
    func: MaskFunc


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _col(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Missing required column. Tried: {candidates}")


def _maybe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _add_scores(df: pd.DataFrame, model_path: Path, race_col: str) -> pd.DataFrame:
    with model_path.open("rb") as f:
        model = pickle.load(f)
    out = df.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)

    popularity_col = _col(out, ["人気", "莠ｺ豌・"])
    odds_col = _col(out, ["単勝オッズ", "蜊伜享繧ｪ繝・ぜ"])
    out["popularity_num"] = _num(out[popularity_col])
    out["odds_decimal"] = _num(out[odds_col])
    out["pop_rank"] = out.groupby(race_col)["popularity_num"].rank(ascending=True, method="first").astype(int)
    out["ai_pop_gap"] = out["ai_rank"] - out["pop_rank"]

    race_second = out.groupby(race_col)["ai_score"].transform(lambda s: s.sort_values(ascending=False).iloc[1] if len(s) > 1 else np.nan)
    race_median = out.groupby(race_col)["ai_score"].transform("median")
    out["ai_score_gap_to_second"] = (out["ai_score"] - race_second).where(out["ai_rank"] == 1, 0.0).fillna(0.0)
    out["ai_top_score_vs_median"] = (out["ai_score"] - race_median).where(out["ai_rank"] == 1, 0.0).fillna(0.0)
    out["ai_confidence_score"] = (
        out["ai_score_gap_to_second"].clip(lower=0.0) * 0.7
        + out["ai_top_score_vs_median"].clip(lower=0.0) * 0.3
    )
    return out


def _payoff_col(df: pd.DataFrame, kind: str) -> str:
    if kind == "win":
        return _col(df, ["単勝配当", "蜊伜享驟榊ｽ・"])
    if kind == "place":
        return _col(df, ["複勝配当", "隍・享驟榊ｽ・"])
    raise ValueError(kind)


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
            "avg_odds": None,
            "avg_ai_rank": None,
            "win_roi": 0.0,
            "place_roi": 0.0,
            "profit_win_flat100": 0.0,
            "profit_place_flat100": 0.0,
        }
    win_pay = _num(df[_payoff_col(df, "win")]).fillna(0.0).where(df["target_win"] == 1, 0.0)
    place_pay = _num(df[_payoff_col(df, "place")]).fillna(0.0).where(df["target_top3"] == 1, 0.0)
    stake = rows * 100.0
    return {
        "segment": label,
        "bets": int(rows),
        "races": int(df[race_col].nunique()),
        "win_rate": float(df["target_win"].mean()),
        "top3_rate": float(df["target_top3"].mean()),
        "avg_popularity": float(df["popularity_num"].mean()),
        "avg_odds": float(df["odds_decimal"].mean()),
        "avg_ai_rank": float(df["ai_rank"].mean()),
        "win_roi": float(win_pay.sum() / stake),
        "place_roi": float(place_pay.sum() / stake),
        "profit_win_flat100": float(win_pay.sum() - stake),
        "profit_place_flat100": float(place_pay.sum() - stake),
    }


def _q(df: pd.DataFrame, column: str, quantile: float, default: float = 0.0) -> float:
    if column not in df.columns:
        return default
    values = _num(df[column]).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return default
    return float(values.quantile(quantile))


def _add_if_present(conditions: list[Condition], df: pd.DataFrame, name: str, column: str, op: str, threshold: float) -> None:
    if column not in df.columns:
        return
    if op == ">=":
        conditions.append(Condition(name, lambda x, c=column, t=threshold: _num(x[c]).fillna(-np.inf) >= t))
    elif op == "<=":
        conditions.append(Condition(name, lambda x, c=column, t=threshold: _num(x[c]).fillna(np.inf) <= t))
    elif op == ">":
        conditions.append(Condition(name, lambda x, c=column, t=threshold: _num(x[c]).fillna(-np.inf) > t))
    else:
        raise ValueError(op)


def _build_conditions(discovery: pd.DataFrame) -> list[Condition]:
    conditions: list[Condition] = [
        Condition("ai_top1", lambda x: x["ai_rank"] == 1),
        Condition("ai_top3", lambda x: x["ai_rank"] <= 3),
        Condition("ai_top5", lambda x: x["ai_rank"] <= 5),
        Condition("pop5plus", lambda x: x["popularity_num"] >= 5),
        Condition("pop7plus", lambda x: x["popularity_num"] >= 7),
        Condition("pop10plus", lambda x: x["popularity_num"] >= 10),
        Condition("odds10plus", lambda x: x["odds_decimal"] >= 10),
        Condition("odds20plus", lambda x: x["odds_decimal"] >= 20),
        Condition("ai_market_gap3plus", lambda x: x["ai_pop_gap"] <= -3),
        Condition("ai_market_gap5plus", lambda x: x["ai_pop_gap"] <= -5),
        Condition("top1_gap005", lambda x: x["ai_score_gap_to_second"] >= 0.05),
        Condition("top1_gap010", lambda x: x["ai_score_gap_to_second"] >= 0.10),
    ]

    high_columns = [
        ("pace_fit_hi", "pace_fit_score", 0.75),
        ("draw_pace_hi", "draw_pace_fit_score", 0.75),
        ("bias_recent_hi", "bias_adjusted_recent_score", 0.75),
        ("lap_fit_hi", "lap_aptitude_fit_score", 0.75),
        ("lap_reliable_hi", "lap_aptitude_reliability_score", 0.75),
        ("member_level_hi", "race_member_level_rank_score", 0.75),
        ("confirmed_member_hi", "confirmed_member_level_adjusted_score", 0.75),
        ("blood_hi", "bloodline_high_confidence_fit_score", 0.75),
        ("blood_lift_hi", "bloodline_lift_fit_score", 0.75),
        ("workout_knowledge_hi", "workout_knowledge_grade_score", 0.75),
        ("workout_load_hi", "workout_load_density_score", 0.75),
        ("trainer_workout_roi_hi", "workout_trainer_pattern_win_roi", 0.75),
        ("horse_workout_roi_hi", "workout_horse_pattern_win_roi", 0.75),
        ("same_day_bias_fit_pos", "same_day_bias_fit_score", 0.60),
        ("same_day_pop_fit_pos", "same_day_pop_adjusted_pace_fit_score", 0.60),
        ("same_day_adversity_fit_pos", "same_day_adversity_fit_score", 0.60),
        ("prev_retro_resistant_hi", "prev_retro_bias_resistant_score", 0.75),
        ("prev_retro_adversity_hi", "prev_retro_bias_adversity_score", 0.75),
        ("prev_retro_excuse_hi", "prev_retro_bias_excuse_score", 0.75),
        ("past3_retro_resistant_hi", "past3_retro_bias_resistant_score", 0.75),
        ("past3_retro_adversity_hi", "past3_retro_bias_adversity_score", 0.75),
        ("past3_retro_excuse_hi", "past3_retro_bias_excuse_score", 0.75),
    ]
    for name, column, quantile in high_columns:
        _add_if_present(conditions, discovery, name, column, ">=", _q(discovery, column, quantile))

    low_columns = [
        ("retro_overhelped_low", "prev_retro_bias_overhelped_score", 0.50),
        ("draw_disadvantaged", "current_draw_advantage_score", 0.25),
    ]
    for name, column, quantile in low_columns:
        _add_if_present(conditions, discovery, name, column, "<=", _q(discovery, column, quantile))

    if "same_day_bias_ready" in discovery.columns:
        conditions.append(Condition("same_day_ready", lambda x: _num(x["same_day_bias_ready"]).fillna(0) >= 1))
    if "workout_knowledge_registered_flag" in discovery.columns:
        conditions.append(Condition("workout_rule_registered", lambda x: _num(x["workout_knowledge_registered_flag"]).fillna(0) >= 1))

    return conditions


def _split_temporal(df: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_values = pd.to_numeric(df[date_col], errors="coerce")
    unique_dates = np.array(sorted(date_values.dropna().unique()))
    if len(unique_dates) < 4:
        half = len(df) // 2
        return df.iloc[:half].copy(), df.iloc[half:].copy()
    cutoff = unique_dates[len(unique_dates) // 2]
    return df[date_values <= cutoff].copy(), df[date_values > cutoff].copy()


def _segment_mask(mask_by_name: dict[str, pd.Series], index: pd.Index, combo: tuple[Condition, ...]) -> pd.Series:
    mask = pd.Series(True, index=index)
    for condition in combo:
        mask &= mask_by_name[condition.name]
    return mask


def _precompute_masks(df: pd.DataFrame, conditions: list[Condition]) -> dict[str, pd.Series]:
    masks = {}
    for condition in conditions:
        masks[condition.name] = condition.func(df).reindex(df.index).fillna(False).astype(bool)
    return masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Search validated betting policy segments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/betting_policy_segments")
    parser.add_argument("--min-discovery-bets", type=int, default=200)
    parser.add_argument("--min-validation-bets", type=int, default=100)
    parser.add_argument("--max-combo-size", type=int, default=4)
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    date_col = config["data"]["date_column"]
    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = _add_scores(df, project_path(args.model), race_col)
    discovery, validation = _split_temporal(scored, date_col)
    conditions = _build_conditions(discovery)
    discovery_masks = _precompute_masks(discovery, conditions)
    validation_masks = _precompute_masks(validation, conditions)

    rows = []
    all_metrics = _metrics(scored, race_col, "all")
    rows.append({"segment": "all", **{f"all_{k}": v for k, v in all_metrics.items() if k != "segment"}})

    for size in range(1, args.max_combo_size + 1):
        for combo in combinations(conditions, size):
            label = "&".join(c.name for c in combo)
            discovery_part = discovery[_segment_mask(discovery_masks, discovery.index, combo)]
            if len(discovery_part) < args.min_discovery_bets:
                continue
            validation_part = validation[_segment_mask(validation_masks, validation.index, combo)]
            if len(validation_part) < args.min_validation_bets:
                continue
            d = _metrics(discovery_part, race_col, label)
            v = _metrics(validation_part, race_col, label)
            rows.append(
                {
                    "segment": label,
                    **{f"discovery_{k}": value for k, value in d.items() if k != "segment"},
                    **{f"validation_{k}": value for k, value in v.items() if k != "segment"},
                    "roi_stability_win": float(v["win_roi"] - d["win_roi"]),
                    "roi_stability_place": float(v["place_roi"] - d["place_roi"]),
                }
            )

    out = pd.DataFrame(rows)
    segment_rows = out[out["segment"] != "all"].copy()
    if not segment_rows.empty:
        segment_rows = segment_rows.sort_values(
            ["validation_win_roi", "validation_place_roi", "validation_bets"],
            ascending=[False, False, False],
        )
    output_dir = ensure_dir(project_path(args.output_dir))
    all_path = output_dir / "all_segments.csv"
    top_path = output_dir / "validated_top_segments.csv"
    json_path = output_dir / "summary.json"
    out.to_csv(all_path, index=False, encoding="utf-8-sig")
    segment_rows.to_csv(top_path, index=False, encoding="utf-8-sig")
    summary = {
        "rows": int(len(out)),
        "conditions": [c.name for c in conditions],
        "all_segments_csv": str(all_path),
        "validated_top_segments_csv": str(top_path),
        "top_validation_win_roi": segment_rows.head(20).to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
