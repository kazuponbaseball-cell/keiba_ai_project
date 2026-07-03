from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_priority_s_betting_policy import (
    _attach_race_features,
    _existing_raw_csv,
    _load_pairs,
    _load_wide,
    _metrics,
    _pair_universe,
    _walkforward,
)
from src.utils.paths import ensure_dir, project_path


VERTICAL_COLS = [
    "vertical_condition_fit_score",
    "vertical_condition_mismatch_score",
    "vertical_condition_reliability_score",
    "vertical_condition_positive_flag",
    "vertical_condition_negative_flag",
    "workout_vs_horse_best_total_gap_sec",
    "workout_vs_horse_best_final1_gap_sec",
    "workout_horse_regression_flag",
    "layoff_vertical_fit_score",
    "direction_vertical_fit_score",
    "course_vertical_fit_score",
    "surface_distance_vertical_fit_score",
    "workout_vertical_fit_score",
]


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


def _load_vertical(path: Path) -> pd.DataFrame:
    usecols = ["レースID(新/馬番無)", "馬番", *VERTICAL_COLS]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    df = df.rename(columns={"レースID(新/馬番無)": "race_id", "馬番": "horse_no"})
    df["race_id"] = df["race_id"].astype(str)
    df["horse_no"] = _num(df["horse_no"]).astype("Int64")
    for col in VERTICAL_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = _num(df[col]).fillna(0.0)
    return df


def _merge_vertical(scored: pd.DataFrame, vertical: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["horse_no"] = _num(out["horse_no"]).astype("Int64")
    out = out.merge(vertical, on=["race_id", "horse_no"], how="left")
    for col in VERTICAL_COLS:
        out[col] = _num(out.get(col), out.index, 0.0).fillna(0.0)
    out["vertical_overpopular_risk_score"] = (
        0.45 * _norm01(out["vertical_condition_mismatch_score"])
        + 0.25 * _norm01(out["workout_vs_horse_best_total_gap_sec"])
        + 0.20 * _num(out["vertical_condition_negative_flag"]).clip(0.0, 1.0)
        + 0.10 * _num(out["workout_horse_regression_flag"]).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["vertical_underpopular_value_score"] = (
        0.35 * _norm01(out["vertical_condition_fit_score"])
        + 0.25 * _norm01(out["surface_distance_vertical_fit_score"])
        + 0.20 * _norm01(out["course_vertical_fit_score"])
        + 0.10 * _norm01(out["workout_vertical_fit_score"])
        + 0.10 * _num(out["vertical_condition_positive_flag"]).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    return out


def _apply_variant(scored: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = scored.copy()
    pop = _num(out.get("pop_rank_num"), out.index, 99).fillna(99)
    odds = _num(out.get("market_odds_live_or_final"), out.index, np.nan)
    overpopular = pop.le(3) & _num(out["vertical_overpopular_risk_score"]).ge(0.66)
    underpopular = pop.ge(4) & odds.ge(6.0) & _num(out["vertical_underpopular_value_score"]).ge(0.62)

    if variant in {"anchor_risk_downweight", "combined_overlay"}:
        danger_add = np.where(overpopular, 0.18, 0.0)
        out["danger_favorite_score"] = (_num(out.get("danger_favorite_score"), out.index, 0.0).fillna(0.0) + danger_add).clip(0.0, 1.0)
        if "danger_popular_hybrid_score" in out.columns:
            out["danger_popular_hybrid_score"] = (
                _num(out["danger_popular_hybrid_score"]).fillna(out["danger_favorite_score"]) + danger_add
            ).clip(0.0, 1.0)
        out["skip_risk_score"] = (_num(out.get("skip_risk_score"), out.index, 0.0).fillna(0.0) + np.where(overpopular, 0.12, 0.0)).clip(0.0, 1.0)
        for col in ["win_suitability_score", "place_suitability_score", "wide_axis_score", "quinella_model_score_norm"]:
            if col in out.columns:
                out[col] = (_num(out[col]).fillna(0.0) * np.where(overpopular, 0.88, 1.0)).clip(0.0, 1.0)

    if variant == "combined_overlay":
        boost = np.where(underpopular, 0.10, 0.0)
        for col in ["market_overlay_score", "wide_partner_score", "place_suitability_score", "quinella_model_score_norm"]:
            if col in out.columns:
                out[col] = (_num(out[col]).fillna(0.0) + boost).clip(0.0, 1.0)
        if "late_value_survives_score" in out.columns:
            out["late_value_survives_score"] = (_num(out["late_value_survives_score"]).fillna(0.0) + np.where(underpopular, 0.06, 0.0)).clip(0.0, 1.0)

    out["vertical_overpopular_flag"] = overpopular.astype(float)
    out["vertical_underpopular_flag"] = underpopular.astype(float)
    return out


def _run_policy(scored: pd.DataFrame, wide: pd.DataFrame, pairs: pd.DataFrame, min_train_races: int, min_race_hit: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    universe = _attach_race_features(_pair_universe(scored, wide, pairs), scored)
    train_grid, wf_summary, wf_tickets = _walkforward(universe, min_train_races, min_race_hit)
    return universe, train_grid, wf_summary, wf_tickets, _metrics(wf_tickets, "walkforward_total")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate vertical horse-context signals as over/under-popularity ROI gates.")
    parser.add_argument("--scored-csv", default="outputs/analysis/risk_models_v1/investment_features_with_risk_models.csv")
    parser.add_argument("--vertical-csv", default="outputs/analysis/vertical_horse_context_v1/test_features_with_vertical_context.csv")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--output-dir", default="outputs/analysis/vertical_context_roi_v1")
    parser.add_argument("--min-train-races", type=int, default=250)
    parser.add_argument("--min-race-hit", type=float, default=0.15)
    args = parser.parse_args()

    scored = pd.read_csv(project_path(args.scored_csv), low_memory=False)
    vertical = _load_vertical(project_path(args.vertical_csv))
    merged = _merge_vertical(scored, vertical)
    wide = _load_wide(project_path(args.wide_payoff_csv))
    pairs = _load_pairs(_existing_raw_csv(args.raw_csv), project_path(args.config))
    out_dir = ensure_dir(project_path(args.output_dir))

    variants = ["baseline", "anchor_risk_downweight", "combined_overlay"]
    summary_rows = []
    for variant in variants:
        scored_variant = merged if variant == "baseline" else _apply_variant(merged, variant)
        universe, train_grid, wf_summary, wf_tickets, total = _run_policy(
            scored_variant,
            wide,
            pairs,
            args.min_train_races,
            args.min_race_hit,
        )
        variant_dir = ensure_dir(out_dir / variant)
        scored_variant.to_csv(variant_dir / "scored_with_vertical_adjustments.csv", index=False, encoding="utf-8-sig")
        universe.to_csv(variant_dir / "priority_s_pair_universe.csv", index=False, encoding="utf-8-sig")
        train_grid.to_csv(variant_dir / "walkforward_train_grid_top40.csv", index=False, encoding="utf-8-sig")
        wf_summary.to_csv(variant_dir / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
        wf_tickets.to_csv(variant_dir / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
        total["variant"] = variant
        summary_rows.append(total)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "vertical_context_roi_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(out_dir),
        "variants": summary.to_dict(orient="records"),
        "note": "Vertical context used only as over-popular risk downweight and under-popular value boost; no current-race result columns are used.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
