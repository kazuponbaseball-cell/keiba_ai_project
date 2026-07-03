from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.simple_ranker import SimpleRaceRanker
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _norm01(series: pd.Series) -> pd.Series:
    s = _num(series).replace([np.inf, -np.inf], np.nan)
    lo = s.quantile(0.05)
    hi = s.quantile(0.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.5)


HORSE_NUMERIC_CANDIDATES = [
    "ai_rank_num",
    "pop_rank_num",
    "market_odds_live_or_final",
    "market_win_prob_norm",
    "ai_win_prob_proxy",
    "ai_market_prob_diff",
    "ai_market_prob_ratio",
    "win_ev_proxy",
    "market_overlay_score",
    "late_odds_drop_rate",
    "session_odds_drop_rate",
    "late_odds_drift_rate",
    "late_value_survives_score",
    "projected_front5_prob",
    "win_suitability_score",
    "place_suitability_score",
    "wide_axis_score",
    "wide_partner_score",
    "skip_risk_score",
    "quinella_model_score",
    "quinella_model_prob",
    "quinella_model_rank",
    "quinella_model_score_norm",
    "race_front_runner_ratio",
    "race_pace_collapse_risk",
    "race_slow_pace_risk",
    "pace_fit_score",
    "front_advantage_score",
    "positioning_advantage_score",
    "draw_pace_fit_score",
    "horse_front_run_rate_past5",
    "horse_closer_rate_past5",
    "same_day_bias_ready",
    "same_day_bias_volatility",
    "same_day_bias_fit_score",
    "same_day_pop_adjusted_pace_fit_score",
    "same_day_projected_front_load_score",
    "same_day_projected_closer_load_score",
    "same_day_front_collapse_index",
    "same_day_closer_blocked_index",
    "prev_corner4_position_rate",
    "front_running_tendency",
    "horse_stalker_rate_past5",
    "front_pressure_rank_score",
    "solo_lead_potential",
    "PCI",
    "PCI3",
    "RPCI",
    "distance",
    "頭数",
    "出走頭数",
    "枠番",
]

HORSE_CATEGORICAL_CANDIDATES = [
    "venue",
    "surface",
    "distance_category_eval",
    "expected_pace",
    "class_group",
    "field_bin",
    "rpci_bin",
    "pci_bin",
    "style_bucket",
    "bias_volatility_bin",
    "馬場状態",
    "クラス名",
]


def _existing_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _prepare_horse_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    finish = _num(out.get("finish_num"), out.index)
    pop = _num(out.get("pop_rank_num"), out.index)
    ai_rank = _num(out.get("ai_rank_num"), out.index)
    odds = _num(out.get("market_odds_live_or_final"), out.index)
    popular = pop.between(1, 3, inclusive="both") | odds.le(5.0)
    model_focus = popular | ai_rank.le(3)
    out["target_danger_popular_miss_place"] = (popular & finish.gt(3)).astype(float)
    out["danger_model_sample_weight"] = np.where(model_focus, 1.0, 0.35)
    return out


def _fit_weighted_ranker(df: pd.DataFrame, target: str, numeric: list[str], categorical: list[str], weight_col: str | None = None) -> SimpleRaceRanker:
    if weight_col and weight_col in df.columns:
        weights = _num(df[weight_col], df.index, 1.0).clip(lower=0.05, upper=5.0)
        train = df.loc[df.index.repeat(np.ceil(weights * 3).astype(int).clip(1, 15))].copy()
    else:
        train = df
    return SimpleRaceRanker(numeric_features=numeric, categorical_features=categorical, categorical_top_k=80, ridge_alpha=12.0).fit(train, target)


def _race_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["race_id"] = work["race_id"].astype(str)
    ai_rank = _num(work.get("ai_rank_num"), work.index)
    pop = _num(work.get("pop_rank_num"), work.index)
    finish = _num(work.get("finish_num"), work.index)
    work["_is_ai_top1"] = ai_rank.eq(1)
    work["_is_ai_top3"] = ai_rank.le(3)
    work["_is_pop_top3"] = pop.le(3)
    work["_is_place"] = finish.le(3)
    work["_is_win"] = finish.eq(1)

    top1 = work[work["_is_ai_top1"]].copy()
    top1_cols = [
        "race_id",
        "finish_num",
        "market_odds_live_or_final",
        "danger_favorite_score",
        "skip_risk_score",
        "win_suitability_score",
        "place_suitability_score",
        "quinella_model_score_norm",
    ]
    top1_cols = [c for c in top1_cols if c in top1.columns]
    top1 = top1[top1_cols].rename(
        columns={
            "finish_num": "ai1_finish",
            "market_odds_live_or_final": "ai1_odds",
            "danger_favorite_score": "ai1_rule_danger",
            "skip_risk_score": "ai1_skip_risk",
            "win_suitability_score": "ai1_win_score",
            "place_suitability_score": "ai1_place_score",
            "quinella_model_score_norm": "ai1_quinella_norm",
        }
    )

    by = work.groupby("race_id", as_index=False).agg(
        race_field_size=("horse_no", "count"),
        race_front_pressure=("race_early_pressure_score", "max"),
        race_front_count=("race_front_runner_count_y", "max"),
        race_pace_collapse=("race_pace_collapse_risk", "max"),
        race_slow_risk=("race_slow_pace_risk", "max"),
        race_bias_volatility=("same_day_bias_volatility", "max"),
        race_same_day_ready=("same_day_bias_ready", "max"),
        race_avg_late_drop=("late_odds_drop_rate", "mean"),
        race_avg_late_drift=("late_odds_drift_rate", "mean"),
        race_max_rule_danger=("danger_favorite_score", "max"),
        race_avg_skip_risk=("skip_risk_score", "mean"),
        race_avg_projected_front5=("projected_front5_prob", "mean"),
        race_market_top3_prob_sum=("market_win_prob_norm", lambda s: float(_num(s).nlargest(3).sum())),
        race_ai_top3_place_count=("_is_place", lambda s: 0.0),
    )
    top3_place = work[work["_is_ai_top3"]].groupby("race_id")["_is_place"].sum().rename("race_ai_top3_place_count").reset_index()
    winner_ai_rank = work[work["_is_win"]].groupby("race_id")["ai_rank_num"].min().rename("winner_ai_rank").reset_index()
    by = by.drop(columns=["race_ai_top3_place_count"], errors="ignore").merge(top3_place, on="race_id", how="left")
    by = by.merge(winner_ai_rank, on="race_id", how="left")
    by = by.merge(top1, on="race_id", how="left")

    firsts = work.groupby("race_id").first(numeric_only=False).reset_index()
    for col in HORSE_CATEGORICAL_CANDIDATES:
        if col in firsts.columns:
            by[col] = firsts.set_index("race_id").loc[by["race_id"], col].to_numpy()

    by["target_race_difficulty"] = (
        0.60 * _num(by["ai1_finish"], by.index, 99.0).gt(3).astype(float)
        + 0.25 * _num(by["race_ai_top3_place_count"], by.index, 0.0).le(1).astype(float)
        + 0.15 * _num(by["winner_ai_rank"], by.index, 99.0).gt(5).astype(float)
    )
    return by


RACE_NUMERIC_CANDIDATES = [
    "race_field_size",
    "race_front_pressure",
    "race_front_count",
    "race_pace_collapse",
    "race_slow_risk",
    "race_bias_volatility",
    "race_same_day_ready",
    "race_avg_late_drop",
    "race_avg_late_drift",
    "race_max_rule_danger",
    "race_avg_skip_risk",
    "race_avg_projected_front5",
    "race_market_top3_prob_sum",
    "ai1_odds",
    "ai1_rule_danger",
    "ai1_skip_risk",
    "ai1_win_score",
    "ai1_place_score",
    "ai1_quinella_norm",
]


def _score_walkforward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out = _prepare_horse_targets(df)
    out["year"] = out["race_id"].astype(str).str[:4].astype(int)
    years = sorted(out["year"].dropna().astype(int).unique())
    numeric = _existing_columns(out, HORSE_NUMERIC_CANDIDATES)
    categorical = _existing_columns(out, HORSE_CATEGORICAL_CANDIDATES)
    out["danger_popular_model_score"] = np.nan
    models_meta: dict[str, object] = {"danger_models": {}, "race_difficulty_models": {}}

    race_frame = _race_rows(out)
    race_frame["year"] = race_frame["race_id"].astype(str).str[:4].astype(int)
    race_numeric = _existing_columns(race_frame, RACE_NUMERIC_CANDIDATES)
    race_categorical = _existing_columns(race_frame, HORSE_CATEGORICAL_CANDIDATES)
    race_frame["race_difficulty_model_score"] = np.nan

    for year in years[1:]:
        train = out[out["year"] < year].copy()
        test_idx = out.index[out["year"].eq(year)]
        if train.empty or len(test_idx) == 0:
            continue
        danger_model = _fit_weighted_ranker(train, "target_danger_popular_miss_place", numeric, categorical, "danger_model_sample_weight")
        pred = pd.Series(danger_model.predict(out.loc[test_idx]), index=test_idx)
        out.loc[test_idx, "danger_popular_model_score"] = pred.clip(0.0, 1.0)
        models_meta["danger_models"][str(year)] = danger_model.to_metadata()

        race_train = race_frame[race_frame["year"] < year].copy()
        race_test_idx = race_frame.index[race_frame["year"].eq(year)]
        if not race_train.empty and len(race_test_idx):
            race_model = SimpleRaceRanker(
                numeric_features=race_numeric,
                categorical_features=race_categorical,
                categorical_top_k=60,
                ridge_alpha=10.0,
            ).fit(race_train, "target_race_difficulty")
            race_pred = pd.Series(race_model.predict(race_frame.loc[race_test_idx]), index=race_test_idx)
            race_frame.loc[race_test_idx, "race_difficulty_model_score"] = race_pred.clip(0.0, 1.0)
            models_meta["race_difficulty_models"][str(year)] = race_model.to_metadata()

    # Fallback for the first year so downstream scripts can still run; this year
    # is not used as a walk-forward test year in the current betting policy.
    out["danger_popular_model_score"] = out["danger_popular_model_score"].fillna(out.get("danger_favorite_score", 0.0))
    race_frame["race_difficulty_model_score"] = race_frame["race_difficulty_model_score"].fillna(_norm01(race_frame["target_race_difficulty"]))

    race_scores = race_frame[["race_id", "race_difficulty_model_score", "target_race_difficulty"]].copy()
    out = out.merge(race_scores, on="race_id", how="left")
    out["danger_popular_model_rank"] = out.groupby("race_id")["danger_popular_model_score"].rank(ascending=False, method="first")
    out["danger_popular_hybrid_score"] = (
        0.55 * _num(out["danger_popular_model_score"], out.index, 0.0)
        + 0.30 * _num(out.get("danger_favorite_score"), out.index, 0.0)
        + 0.15 * _num(out.get("skip_risk_score"), out.index, 0.0)
    ).clip(0.0, 1.0)
    return out, race_frame, models_meta


def _diagnostics(scored: pd.DataFrame, race_frame: pd.DataFrame) -> dict:
    pop = _num(scored.get("pop_rank_num"), scored.index)
    focus = scored[pop.le(3)].copy()
    if focus.empty:
        danger_bins = []
    else:
        focus["danger_bin"] = pd.qcut(focus["danger_popular_hybrid_score"], q=5, labels=False, duplicates="drop")
        danger_bins = (
            focus.groupby("danger_bin", dropna=False)
            .agg(
                rows=("race_id", "size"),
                avg_score=("danger_popular_hybrid_score", "mean"),
                miss_place_rate=("target_danger_popular_miss_place", "mean"),
            )
            .reset_index()
            .to_dict(orient="records")
        )
    race_eval = race_frame.copy()
    race_eval["difficulty_bin"] = pd.qcut(race_eval["race_difficulty_model_score"], q=5, labels=False, duplicates="drop")
    race_bins = (
        race_eval.groupby("difficulty_bin", dropna=False)
        .agg(
            races=("race_id", "size"),
            avg_score=("race_difficulty_model_score", "mean"),
            actual_bad_race=("target_race_difficulty", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    return {"danger_popular_bins": danger_bins, "race_difficulty_bins": race_bins}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train walk-forward dangerous-popular and race-difficulty models.")
    parser.add_argument("--scored-csv", default="outputs/analysis/quinella_top2_model_v1/investment_features_with_quinella_score.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/risk_models_v1")
    parser.add_argument("--model-dir", default="models/risk_models_v1")
    args = parser.parse_args()

    df = pd.read_csv(project_path(args.scored_csv), dtype={"race_id": str}, low_memory=False)
    scored, race_frame, model_meta = _score_walkforward(df)
    out_dir = ensure_dir(project_path(args.output_dir))
    model_dir = ensure_dir(project_path(args.model_dir))
    scored_path = out_dir / "investment_features_with_risk_models.csv"
    race_path = out_dir / "race_difficulty_model_scores.csv"
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    race_frame.to_csv(race_path, index=False, encoding="utf-8-sig")
    diagnostics = _diagnostics(scored, race_frame)
    payload = {
        "output_dir": str(out_dir),
        "model_dir": str(model_dir),
        "scored_csv": str(scored_path),
        "race_scores_csv": str(race_path),
        "rows": int(len(scored)),
        "races": int(scored["race_id"].nunique()),
        "diagnostics": diagnostics,
        "models": model_meta,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(model_meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
