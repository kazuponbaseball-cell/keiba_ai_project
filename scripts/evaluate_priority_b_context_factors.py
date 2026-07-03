from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_priority_a_non_day_factors import _metric, _num
from scripts.optimize_operational_win_addon import _json_default
from src.utils.paths import ensure_dir, project_path


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _norm01(series: pd.Series, default: float = 0.5, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if lo is None:
        lo = float(x.quantile(0.05)) if x.notna().any() else 0.0
    if hi is None:
        hi = float(x.quantile(0.95)) if x.notna().any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(default, index=series.index, dtype=float)
    return ((x - lo) / (hi - lo)).fillna(default).clip(0.0, 1.0)


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    return _num(df.get(name), df.index, default).fillna(default)


def enrich_b_factors(tickets: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    if "year" not in df.columns:
        df["year"] = df["race_id"].astype(str).str[:4].astype(int)

    field_size = _col(df, "race_field_size", _col(df, "出走頭数", 14.0)).clip(5, 18)
    field_compact = 1.0 - ((field_size - 5.0) / 13.0).clip(0.0, 1.0)
    chaos = _clip01(_col(df, "race_chaos_score", _col(df, "race_difficulty_score", 0.5)))
    difficulty = _clip01(_col(df, "race_difficulty_score", _col(df, "anchor_race_difficulty_model", 0.5)))
    solid = _clip01(_col(df, "race_solidness_score", 1.0 - chaos))
    front_collapse = _clip01(_col(df, "race_pace_collapse", _col(df, "race_pace_collapse_risk", 0.0)))
    volatility = _clip01(_col(df, "race_bias_volatility", _col(df, "same_day_bias_volatility", 0.0)))
    favorite_danger = _clip01(_col(df, "race_favorite_danger", 0.0))
    df["b_race_readability_score"] = _clip01(
        0.24 * solid
        + 0.20 * (1.0 - chaos)
        + 0.18 * (1.0 - difficulty)
        + 0.14 * field_compact
        + 0.10 * (1.0 - front_collapse)
        + 0.08 * (1.0 - volatility)
        + 0.06 * (1.0 - favorite_danger)
    )

    anchor_front = _clip01(_col(df, "projected_front5_prob", _col(df, "ticket_front_position_reliability_score", 0.5)))
    pace_fit = _clip01(_col(df, "pace_fit_score", _col(df, "pace_fit_score_feature", 0.5)))
    front_adv = _clip01(_col(df, "front_advantage_score", _col(df, "front_advantage_score_feature", 0.5)))
    pos_adv = _clip01(_col(df, "positioning_advantage_score", _col(df, "positioning_advantage_score_feature", 0.5)))
    draw_fit = _clip01(_col(df, "draw_pace_fit_score", _col(df, "draw_pace_fit_score_feature", 0.5)))
    corner_fit = _clip01(
        0.60 * _col(df, "anchor_ctx_corner_rpci_score", 0.5)
        + 0.40 * _col(df, "partner_ctx_corner_rpci_score", _col(df, "anchor_ctx_corner_rpci_score", 0.5))
    )
    df["b_course_pace_shape_fit_score"] = _clip01(
        0.22 * pace_fit
        + 0.18 * front_adv
        + 0.18 * pos_adv
        + 0.16 * draw_fit
        + 0.14 * anchor_front
        + 0.12 * corner_fit
    )

    overlay = _norm01(_col(df, "market_overlay_score", _col(df, "overlay", 0.0)), 0.5)
    survives = _clip01(_col(df, "late_value_survives_score", 0.5))
    margin = _norm01(_col(df, "runtime_odds_margin_ratio", _col(df, "min_odds_margin_ratio", 1.0)), 0.5, lo=0.4, hi=2.0)
    expected_roi = _norm01(_col(df, "runtime_expected_roi", _col(df, "expected_roi_after_slippage", 1.0)), 0.5, lo=0.5, hi=3.0)
    late_risk = _clip01(_col(df, "live_odds_movement_risk", 0.0) + 0.5 * _col(df, "late_drift_flag", 0.0))
    df["b_market_stability_value_score"] = _clip01(
        0.30 * overlay
        + 0.25 * survives
        + 0.22 * margin
        + 0.18 * expected_roi
        - 0.15 * late_risk
    )

    hit_prob = _norm01(_col(df, "ticket_hit_prob", 0.0), 0.5)
    stake_quality = _clip01(_col(df, "stake_quality_score", 0.5))
    sizing = _clip01(_col(df, "ticket_sizing_score", 0.5))
    danger = _clip01(_col(df, "ticket_danger_popular_score", _col(df, "skip_risk_score", 0.0)))
    df["b_ticket_quality_score"] = _clip01(
        0.34 * hit_prob
        + 0.24 * stake_quality
        + 0.20 * sizing
        + 0.12 * df["b_market_stability_value_score"]
        + 0.10 * (1.0 - danger)
    )

    df["b_priority_net_score"] = _clip01(
        0.26 * df["b_race_readability_score"]
        + 0.27 * df["b_course_pace_shape_fit_score"]
        + 0.25 * df["b_market_stability_value_score"]
        + 0.22 * df["b_ticket_quality_score"]
    )
    return df


def _round_stake(stake: pd.Series) -> pd.Series:
    return (np.floor(stake.clip(lower=0.0) / 100.0) * 100.0).clip(lower=100.0)


def _reprice(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pay = _num(
        out.get("runtime_backtest_pay_per100"),
        out.index,
        _num(out.get("quote_pay_proxy_per100"), out.index, 0.0),
    ).fillna(0.0)
    out["runtime_return_yen"] = np.where(out.get("hit", False).astype(bool), pay * out["runtime_stake_yen"] / 100.0, 0.0)
    return out


def apply_overlay(df: pd.DataFrame, mode: str, train_year: int, max_stake: float) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    train = out[out["year"].eq(train_year)].copy()
    score_train = _num(train.get("b_priority_net_score"), train.index, np.nan)
    high = float(score_train.quantile(0.75))
    low = float(score_train.quantile(0.25))

    stake = _num(out.get("runtime_stake_yen"), out.index, 0.0).fillna(0.0)
    score = _num(out.get("b_priority_net_score"), out.index, 0.0).fillna(0.0)
    high_mask = score.ge(high) & stake.gt(0)
    low_mask = score.lt(low) & stake.gt(0)

    out["pre_priority_b_stake_yen"] = stake
    out["priority_b_context_action"] = "KEEP"
    new_stake = stake.copy()

    if mode in {"boost_high_b_110", "both"}:
        new_stake.loc[high_mask] = (new_stake.loc[high_mask] * 1.10).clip(upper=max_stake)
        out.loc[high_mask, "priority_b_context_action"] = "BOOST_HIGH_B"
    if mode in {"reduce_low_b_50", "both"}:
        new_stake.loc[low_mask] = new_stake.loc[low_mask] * 0.50
        out.loc[low_mask, "priority_b_context_action"] = np.where(
            out.loc[low_mask, "priority_b_context_action"].eq("KEEP"),
            "REDUCE_LOW_B",
            out.loc[low_mask, "priority_b_context_action"] + "+REDUCE_LOW_B",
        )
    if mode == "skip_low_b":
        new_stake.loc[low_mask] = 0.0
        out.loc[low_mask, "priority_b_context_action"] = "SKIP_LOW_B"

    out["runtime_stake_yen"] = np.where(new_stake.gt(0), _round_stake(new_stake), 0.0)
    out["runtime_reason"] = out.get("runtime_reason", "").astype(str) + "|priority_b_context_overlay:" + mode
    out = _reprice(out)
    meta = {
        "mode": mode,
        "train_year": train_year,
        "max_stake": max_stake,
        "b_high_threshold": high,
        "b_low_threshold": low,
        "high_count": int(high_mask.sum()),
        "low_count": int(low_mask.sum()),
    }
    return out, meta


def _segments(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = df[_num(df.get("runtime_stake_yen"), df.index, 0.0).fillna(0.0).gt(0)].copy()
    features = [
        "b_race_readability_score",
        "b_course_pace_shape_fit_score",
        "b_market_stability_value_score",
        "b_ticket_quality_score",
        "b_priority_net_score",
    ]
    for feature in features:
        values = _num(base.get(feature), base.index, np.nan)
        try:
            bins = pd.qcut(values.rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])
        except Exception:
            continue
        tmp = base.assign(segment=bins)
        for (ticket_type, segment), g in tmp.groupby(["ticket_type", "segment"], observed=True):
            m = _metric(g, f"{feature}_{ticket_type}_{segment}")
            m.update({"feature": feature, "ticket_type": ticket_type, "segment": str(segment), "avg_feature": float(_num(g.get(feature), g.index, np.nan).mean())})
            rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate priority-B context factors from existing ticket/race columns.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/live_runtime_safety_overlay_v1/live_safety_overlaid_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/priority_b_context_factors_v1")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--max-stake", type=float, default=3000.0)
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    enriched = enrich_b_factors(tickets)

    out_dir = ensure_dir(project_path(args.output_dir))
    enriched.to_csv(out_dir / "b_priority_enriched_tickets.csv", index=False, encoding="utf-8-sig")
    _segments(enriched).to_csv(out_dir / "feature_segments.csv", index=False, encoding="utf-8-sig")

    metrics = [_metric(enriched, "base_all")]
    overlays = {}
    for mode in ["boost_high_b_110", "reduce_low_b_50", "both", "skip_low_b"]:
        overlaid, meta = apply_overlay(enriched, mode, args.train_year, args.max_stake)
        overlaid.to_csv(out_dir / f"{mode}_tickets.csv", index=False, encoding="utf-8-sig")
        overlays[mode] = meta
        metrics.append(_metric(overlaid, f"{mode}_all"))
        for year, g in overlaid.groupby("year"):
            metrics.append(_metric(g, f"{mode}_{int(year)}"))
    for year, g in enriched.groupby("year"):
        metrics.append(_metric(g, f"base_{int(year)}"))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    payload = {
        "tickets_csv": args.tickets_csv,
        "output_dir": str(out_dir),
        "overlays": overlays,
        "base_all": metrics[0],
        "best_by_profit": metrics_df.sort_values("profit_yen", ascending=False).head(1).to_dict("records")[0],
        "best_by_roi": metrics_df[metrics_df["races"].ge(150)].sort_values("roi", ascending=False).head(1).to_dict("records")[0],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
