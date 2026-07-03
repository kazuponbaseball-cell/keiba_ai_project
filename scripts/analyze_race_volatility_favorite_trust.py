from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_positive_overlay_policy import (  # noqa: E402
    CLASS_COL,
    ODDS_COL,
    POPULARITY_COL,
    RACE_COL,
    _metrics,
    _num,
    _score,
)


def _q(frame: pd.DataFrame, column: str, q: float, default: float = 0.0) -> float:
    if column not in frame.columns:
        return default
    values = _num(frame[column]).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return default
    return float(values.quantile(q))


def _race_feature_frame(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    optional_cols = [
        "workout_knowledge_grade_score",
        "bloodline_high_confidence_fit_score",
        "bloodline_lift_fit_score",
        "body_young_maturity_score",
        "same_day_bias_volatility",
        "same_day_pop_adjusted_bias_volatility",
    ]
    for col in optional_cols:
        if col not in scored.columns:
            scored[col] = np.nan
    odds = _num(scored[ODDS_COL], scored.index).replace(0, np.nan)
    implied = 1.0 / odds
    scored["_implied"] = implied
    scored["_implied_share"] = implied / implied.groupby(scored[RACE_COL]).transform("sum")

    favorite = (
        scored.sort_values([RACE_COL, "popularity_num", "ai_rank"], ascending=[True, True, True])
        .groupby(RACE_COL, as_index=False)
        .head(1)
        .copy()
    )
    favorite = favorite[
        [
            RACE_COL,
            POPULARITY_COL,
            ODDS_COL,
            "ai_rank",
            "ai_score",
            "target_win",
            "target_top3",
            "pace_fit_score",
            "lap_aptitude_fit_score",
            "lap_aptitude_reliability_score",
            "bias_adjusted_recent_score",
            "draw_pace_fit_score",
            "workout_knowledge_grade_score",
            "bloodline_high_confidence_fit_score",
            "bloodline_lift_fit_score",
            "body_young_maturity_score",
        ]
    ].rename(
        columns={
            ODDS_COL: "favorite_odds",
            "ai_rank": "favorite_ai_rank",
            "ai_score": "favorite_ai_score",
            "target_win": "favorite_win",
            "target_top3": "favorite_top3",
            "pace_fit_score": "favorite_pace_fit",
            "lap_aptitude_fit_score": "favorite_lap_fit",
            "lap_aptitude_reliability_score": "favorite_lap_reliability",
            "bias_adjusted_recent_score": "favorite_bias_adjusted",
            "draw_pace_fit_score": "favorite_draw_pace",
            "workout_knowledge_grade_score": "favorite_workout_knowledge",
            "bloodline_high_confidence_fit_score": "favorite_blood_high_conf",
            "bloodline_lift_fit_score": "favorite_blood_lift",
            "body_young_maturity_score": "favorite_body_young_maturity",
        }
    )

    ai_top = (
        scored.sort_values([RACE_COL, "ai_rank"], ascending=[True, True])
        .groupby(RACE_COL, as_index=False)
        .head(1)
        .copy()
    )
    ai_top = ai_top[
        [
            RACE_COL,
            "ai_score",
            "ai_score_gap_to_second",
            "popularity_num",
            ODDS_COL,
            "target_win",
            "target_top3",
        ]
    ].rename(
        columns={
            "ai_score": "ai_top_score",
            "ai_score_gap_to_second": "ai_top_gap",
            "popularity_num": "ai_top_popularity",
            ODDS_COL: "ai_top_odds",
            "target_win": "ai_top_win",
            "target_top3": "ai_top_top3",
        }
    )

    winners = scored[scored["target_win"].eq(1)][[RACE_COL, "popularity_num", ODDS_COL]].rename(
        columns={"popularity_num": "winner_popularity", ODDS_COL: "winner_odds"}
    )
    top3 = scored[scored["target_top3"].eq(1)].groupby(RACE_COL).agg(
        top3_pop_sum=("popularity_num", "sum"),
        top3_max_pop=("popularity_num", "max"),
        top3_avg_pop=("popularity_num", "mean"),
    )

    agg_spec = {
        "field_size": ("popularity_num", "size"),
        "odds_entropy": ("_implied_share", lambda s: float(-(s.dropna() * np.log(s.dropna())).sum())),
        "favorite_implied_share": ("_implied_share", "max"),
        "market_top3_share": ("_implied_share", lambda s: float(s.sort_values(ascending=False).head(3).sum())),
        "ai_score_spread": ("ai_score", lambda s: float(s.max() - s.median())),
        "ai_top3_score_std": ("ai_score", lambda s: float(s.sort_values(ascending=False).head(3).std(ddof=0))),
    }
    for col in [
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "race_early_pressure_score",
        "race_member_depth_score",
        "race_strong_opponent_ratio",
        "race_member_level_rank_score",
        "same_day_bias_volatility",
        "same_day_pop_adjusted_bias_volatility",
    ]:
        if col in scored.columns:
            agg_spec[col] = (col, "mean")
    race = scored.groupby(RACE_COL).agg(**agg_spec)
    race = race.reset_index().merge(favorite, on=RACE_COL, how="left").merge(ai_top, on=RACE_COL, how="left")
    race = race.merge(winners, on=RACE_COL, how="left").merge(top3.reset_index(), on=RACE_COL, how="left")
    race["favorite_win"] = _num(race["favorite_win"], race.index, 0).fillna(0)
    race["favorite_top3"] = _num(race["favorite_top3"], race.index, 0).fillna(0)
    race["favorite_odds_num"] = _num(race["favorite_odds"], race.index)
    race["ai_top_odds_num"] = _num(race["ai_top_odds"], race.index)
    race["winner_popularity"] = _num(race["winner_popularity"], race.index)
    race["is_rough_win_pop5"] = race["winner_popularity"].ge(5)
    race["is_rough_fav_out"] = race["favorite_top3"].eq(0)
    race["is_rough_top3_pop_sum_ge12"] = _num(race["top3_pop_sum"], race.index).ge(12)
    race["ai_agrees_favorite"] = _num(race["favorite_ai_rank"], race.index).eq(1)
    return race


def _bin_summary(race: pd.DataFrame, column: str, q: int = 4) -> pd.DataFrame:
    values = _num(race[column], race.index)
    if values.nunique(dropna=True) < 3:
        race["_bin"] = values
    else:
        race["_bin"] = pd.qcut(values.rank(method="first"), q=q, labels=False, duplicates="drop")
    return (
        race.groupby("_bin", dropna=False)
        .agg(
            races=(RACE_COL, "nunique"),
            favorite_win_rate=("favorite_win", "mean"),
            favorite_top3_rate=("favorite_top3", "mean"),
            rough_win_pop5_rate=("is_rough_win_pop5", "mean"),
            fav_out_rate=("is_rough_fav_out", "mean"),
            rough_top3_sum_rate=("is_rough_top3_pop_sum_ge12", "mean"),
            avg_value=(column, "mean"),
        )
        .reset_index()
        .assign(feature=column)
    )


def _condition_summaries(scored: pd.DataFrame, race: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = lambda col, quant: _q(scored, col, quant)
    fav = scored[scored["popularity_num"].eq(1)].copy()
    fav["favorite_trust_score"] = 0.0
    checks = {
        "ai_rank1": fav["ai_rank"].eq(1),
        "top_gap": fav["ai_score_gap_to_second"].ge(0.05),
        "pace_fit": _num(fav.get("pace_fit_score"), fav.index).ge(q("pace_fit_score", 0.65)),
        "lap_reliable": _num(fav.get("lap_aptitude_reliability_score"), fav.index).ge(q("lap_aptitude_reliability_score", 0.65)),
        "bias_fit": _num(fav.get("bias_adjusted_recent_score"), fav.index).ge(q("bias_adjusted_recent_score", 0.65)),
        "draw_fit": _num(fav.get("draw_pace_fit_score"), fav.index).ge(q("draw_pace_fit_score", 0.65)),
        "blood_fit": _num(fav.get("bloodline_high_confidence_fit_score"), fav.index).ge(q("bloodline_high_confidence_fit_score", 0.65)),
        "workout_fit": _num(fav.get("workout_knowledge_grade_score"), fav.index).ge(q("workout_knowledge_grade_score", 0.65)),
    }
    for mask in checks.values():
        fav["favorite_trust_score"] += mask.fillna(False).astype(float)

    fav_rows = []
    for label, mask in {
        "favorite_all": pd.Series(True, index=fav.index),
        "favorite_ai_rank1": checks["ai_rank1"],
        "favorite_ai_rank1_gap005": checks["ai_rank1"] & checks["top_gap"],
        "favorite_trust_score_ge3": fav["favorite_trust_score"].ge(3),
        "favorite_trust_score_ge4": fav["favorite_trust_score"].ge(4),
        "favorite_ai1_trust_ge3": checks["ai_rank1"] & fav["favorite_trust_score"].ge(3),
        "favorite_ai1_trust_ge4": checks["ai_rank1"] & fav["favorite_trust_score"].ge(4),
        "favorite_ai1_low_odds_le25": checks["ai_rank1"] & _num(fav[ODDS_COL], fav.index).le(2.5),
        "favorite_ai1_mid_odds_25_45": checks["ai_rank1"] & _num(fav[ODDS_COL], fav.index).between(2.5, 4.5),
    }.items():
        fav_rows.append(_metrics(fav[mask.fillna(False)], label))

    race_rows = []
    race_conditions = {
        "race_all": pd.Series(True, index=race.index),
        "low_vol_fav_share_hi": _num(race["favorite_implied_share"], race.index).ge(race["favorite_implied_share"].quantile(0.70)),
        "low_vol_market_top3_share_hi": _num(race["market_top3_share"], race.index).ge(race["market_top3_share"].quantile(0.70)),
        "low_vol_ai_agrees_fav": race["ai_agrees_favorite"],
        "low_vol_ai_agrees_fav_market_top3_hi": race["ai_agrees_favorite"]
        & _num(race["market_top3_share"], race.index).ge(race["market_top3_share"].quantile(0.60)),
        "rough_entropy_hi": _num(race["odds_entropy"], race.index).ge(race["odds_entropy"].quantile(0.70)),
        "rough_pace_collapse_hi": _num(race["race_pace_collapse_risk"], race.index).ge(race["race_pace_collapse_risk"].quantile(0.70)),
        "rough_depth_hi": _num(race["race_member_depth_score"], race.index).ge(race["race_member_depth_score"].quantile(0.70)),
        "rough_fav_ai_not1": ~race["ai_agrees_favorite"],
    }
    for label, mask in race_conditions.items():
        part = race[mask.fillna(False)]
        if len(part) == 0:
            continue
        race_rows.append(
            {
                "condition": label,
                "races": int(part[RACE_COL].nunique()),
                "favorite_win_rate": float(part["favorite_win"].mean()),
                "favorite_top3_rate": float(part["favorite_top3"].mean()),
                "rough_win_pop5_rate": float(part["is_rough_win_pop5"].mean()),
                "fav_out_rate": float(part["is_rough_fav_out"].mean()),
                "rough_top3_sum_ge12_rate": float(part["is_rough_top3_pop_sum_ge12"].mean()),
                "avg_fav_odds": float(_num(part["favorite_odds_num"], part.index).mean()),
                "avg_winner_popularity": float(_num(part["winner_popularity"], part.index).mean()),
            }
        )

    return pd.DataFrame(fav_rows), pd.DataFrame(race_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/test_features_with_same_day_bias_v3_retro_body_context.csv",
    )
    parser.add_argument("--model", default="models/body_context_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--output-dir", default="outputs/analysis/race_volatility_favorite_trust")
    args = parser.parse_args()

    scored = _score(pd.read_csv(args.test_csv, low_memory=False), Path(args.model))
    race = _race_feature_frame(scored)
    fav_summary, race_summary = _condition_summaries(scored, race)
    bin_features = [
        "favorite_odds_num",
        "favorite_implied_share",
        "market_top3_share",
        "odds_entropy",
        "ai_score_spread",
        "race_pace_collapse_risk",
        "race_early_pressure_score",
        "race_member_depth_score",
        "same_day_bias_volatility",
        "favorite_ai_rank",
        "favorite_lap_reliability",
        "favorite_bias_adjusted",
        "favorite_draw_pace",
    ]
    bins = pd.concat([_bin_summary(race, col) for col in bin_features if col in race.columns], ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    race.to_csv(output_dir / "race_level_volatility.csv", index=False, encoding="utf-8-sig")
    fav_summary.to_csv(output_dir / "favorite_trust_summary.csv", index=False, encoding="utf-8-sig")
    race_summary.to_csv(output_dir / "race_volatility_condition_summary.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(output_dir / "race_volatility_feature_bins.csv", index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(output_dir),
        "races": int(race[RACE_COL].nunique()),
        "favorite_summary": fav_summary.to_dict(orient="records"),
        "race_condition_summary": race_summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Favorite trust")
    print(fav_summary.to_string(index=False))
    print("\nRace volatility")
    print(race_summary.to_string(index=False))
    print(json.dumps({"output_dir": str(output_dir), "races": payload["races"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
