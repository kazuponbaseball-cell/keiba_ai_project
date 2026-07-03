from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAX_CHAMPION_UMAREN_ODDS = 120.0
CHAMPION_SCORE_THRESHOLD = 0.86
CHALLENGER_SCORE_GAP_Z_MAX = 0.25
SAFETY_MARGIN = 0.07


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={"race_id": str, "raceId": str}, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype={"race_id": str, "raceId": str}, low_memory=False)


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def text(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=str)
    return frame[col].astype("string").fillna(default).astype(str)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def simple_pair_key(pair_id: pd.Series) -> pd.Series:
    return pair_id.astype(str).str.replace(":umaren:", ":", regex=False)


def parse_pair_key(pair_key: pd.Series) -> tuple[pd.Series, pd.Series]:
    pair = pair_key.astype(str).str.extract(r":(?P<a>\d+)-(?P<b>\d+)$")
    return pd.to_numeric(pair["a"], errors="coerce"), pd.to_numeric(pair["b"], errors="coerce")


def finish_top2_map(pnl: pd.DataFrame) -> pd.DataFrame:
    if pnl.empty or "raceId" not in pnl.columns or "finishTop3" not in pnl.columns:
        return pd.DataFrame(columns=["race_id", "top1", "top2"])
    out = pnl[["raceId", "finishTop3"]].dropna().drop_duplicates("raceId").copy()
    parts = out["finishTop3"].astype(str).str.extract(r"^\s*(\d+)[^\d]+(\d+)")
    out["top1"] = pd.to_numeric(parts[0], errors="coerce")
    out["top2"] = pd.to_numeric(parts[1], errors="coerce")
    out = out.rename(columns={"raceId": "race_id"})
    out["race_id"] = out["race_id"].astype(str)
    return out[["race_id", "top1", "top2"]].dropna()


def bounded(series: pd.Series, lo: float = 0.0, hi: float = 1.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(lo).clip(lo, hi)


def estimate_survival_probability(row: pd.Series) -> float:
    required = pd.to_numeric(row.get("required_min_odds"), errors="coerce")
    if not np.isfinite(required) or required <= 0:
        return np.nan
    points: list[tuple[float, float]] = []
    for q, col in [
        (0.10, "final_odds_pred_p10"),
        (0.20, "final_odds_pred_p20"),
        (0.50, "final_odds_pred_p50"),
        (0.80, "final_odds_pred_p80"),
        (0.90, "final_odds_pred_p90"),
    ]:
        value = pd.to_numeric(row.get(col), errors="coerce")
        if np.isfinite(value) and value > 0:
            points.append((q, float(value)))
    if not points:
        return np.nan
    points = sorted(points, key=lambda item: item[1])
    odds = [p[1] for p in points]
    qs = [p[0] for p in points]
    if required <= odds[0]:
        percentile = qs[0]
    elif required >= odds[-1]:
        percentile = qs[-1]
    else:
        percentile = float(np.interp(required, odds, qs))
    return max(0.0, min(1.0, 1.0 - percentile))


def add_score_gap_metrics(out: pd.DataFrame) -> pd.DataFrame:
    score = num(out, "strongest_current_score", 0.0)
    race_std = score.groupby(text(out, "race_id")).transform("std")
    global_std = float(score.std()) if np.isfinite(score.std()) and score.std() > 0 else 0.05
    race_std = pd.to_numeric(race_std, errors="coerce").fillna(global_std)
    race_std = race_std.mask(race_std.lt(0.005), global_std).clip(lower=0.005)
    out["score_gap"] = (CHAMPION_SCORE_THRESHOLD - score).clip(lower=0.0)
    out["score_context_std"] = race_std
    out["score_gap_z"] = out["score_gap"] / out["score_context_std"]
    return out


def add_challenger_value_metrics(out: pd.DataFrame) -> pd.DataFrame:
    hit_prob = bounded(num(out, "ticket_hit_prob", 0.0))
    danger = bounded(num(out, "pair_danger_score_external_ai", 0.0))
    pred_unc = bounded(num(out, "prediction_uncertainty_score", 0.0))
    first_unc = bounded(num(out, "first_condition_pair_uncertainty_score", 0.0))
    front5 = bounded(num(out, "projected_front5_prob", 0.0))
    margin = num(out, "min_odds_margin_ratio", 0.0).clip(lower=0.0)

    probability_haircut = (1.0 - 0.25 * pred_unc - 0.15 * danger - 0.10 * first_unc).clip(0.60, 1.0)
    out["pair_probability_low"] = (hit_prob * probability_haircut).clip(0.001, 1.0)
    out["required_min_odds"] = (1.0 + SAFETY_MARGIN) / out["pair_probability_low"]
    q20 = num(out, "final_odds_pred_p20")
    q50 = num(out, "final_odds_pred_p50")
    out["conservative_expected_roi_q20"] = out["pair_probability_low"] * q20
    out["median_expected_roi_q50"] = out["pair_probability_low"] * q50
    out["value_survival_probability"] = out.apply(estimate_survival_probability, axis=1)
    out["value_survival_zone"] = np.select(
        [
            out["value_survival_probability"].ge(0.80)
            & q20.ge(out["required_min_odds"])
            & out["conservative_expected_roi_q20"].ge(1.10),
            out["value_survival_probability"].between(0.65, 0.80, inclusive="left")
            | (q20.lt(out["required_min_odds"]) & q50.ge(out["required_min_odds"])),
        ],
        ["GREEN", "AMBER"],
        default="RED",
    )
    out["pair_fragility_proxy"] = (
        0.35 * pred_unc
        + 0.25 * danger
        + 0.20 * first_unc
        + 0.10 * (1.0 - front5)
        + 0.10 * (1.0 - (margin / 4.0).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    out["model_dispersion_proxy"] = pred_unc
    return out


def add_shadow_actions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_score_gap_metrics(out)
    out = add_challenger_value_metrics(out)

    reasons = text(out, "rejection_reasons")
    single_reason = num(out, "rejection_reason_count", 99).eq(1)
    live_odds = num(out, "live_odds")
    score = num(out, "strongest_current_score", 0.0)
    danger = num(out, "pair_danger_score_external_ai", 1.0)
    pred_unc = num(out, "prediction_uncertainty_score", 1.0)
    market_eff = num(out, "market_efficiency_score", 1.0)
    front5 = num(out, "projected_front5_prob", 0.0)
    margin = num(out, "min_odds_margin_ratio", 0.0)
    t5_odds = num(out, "t5_odds")
    p20_odds = num(out, "final_odds_pred_p20")

    champion_pass = reasons.eq("PASS")
    champion_pass_group = out[champion_pass].copy()
    if not champion_pass_group.empty:
        fragility_p75 = float(champion_pass_group["pair_fragility_proxy"].quantile(0.75))
        dispersion_p90 = float(champion_pass_group["model_dispersion_proxy"].quantile(0.90))
    else:
        fragility_p75 = float(out["pair_fragility_proxy"].quantile(0.75))
        dispersion_p90 = float(out["model_dispersion_proxy"].quantile(0.90))
    if not np.isfinite(fragility_p75):
        fragility_p75 = 0.50
    if not np.isfinite(dispersion_p90):
        dispersion_p90 = 0.60
    out["champion_pass_fragility_p75"] = fragility_p75
    out["champion_pass_dispersion_p90"] = dispersion_p90

    out["odds_survival_ready"] = t5_odds.gt(0) & p20_odds.gt(0)
    out["conservative_ev_margin_q20"] = out["conservative_expected_roi_q20"] - 1.0
    out["t5_to_final_p20_odds_ratio"] = np.where(t5_odds.gt(0), p20_odds / t5_odds, np.nan)

    hard_stop = (
        reasons.str.contains("LIVE_ODDS_MISSING", regex=False)
        | reasons.str.contains("ODDS_TOO_HIGH", regex=False)
        | live_odds.gt(MAX_CHAMPION_UMAREN_ODDS)
    )
    multi_fail = num(out, "rejection_reason_count", 99).ge(2)

    score_only_challenger_s = (
        single_reason
        & reasons.eq("SCORE_FAIL")
        & score.lt(CHAMPION_SCORE_THRESHOLD)
        & out["score_gap_z"].gt(0)
        & out["score_gap_z"].le(CHALLENGER_SCORE_GAP_Z_MAX)
        & live_odds.gt(0)
        & live_odds.le(MAX_CHAMPION_UMAREN_ODDS)
        & out["odds_survival_ready"]
        & out["value_survival_probability"].ge(0.80)
        & out["conservative_expected_roi_q20"].ge(1.10)
        & out["pair_fragility_proxy"].le(fragility_p75)
        & out["model_dispersion_proxy"].le(dispersion_p90)
        & out["value_survival_zone"].eq("GREEN")
    )
    score_only_watch = (
        single_reason
        & reasons.eq("SCORE_FAIL")
        & live_odds.gt(0)
        & live_odds.le(MAX_CHAMPION_UMAREN_ODDS)
        & out["score_gap_z"].gt(0)
        & out["score_gap_z"].le(0.50)
    )
    margin_near_shadow = (
        single_reason
        & reasons.eq("MARGIN_FAIL")
        & margin.ge(2.20)
        & live_odds.le(MAX_CHAMPION_UMAREN_ODDS)
        & out["conservative_expected_roi_q20"].ge(1.20)
        & danger.lt(0.35)
        & pred_unc.lt(0.55)
    )
    front_near_shadow = (
        single_reason
        & reasons.eq("FRONT_PROBABILITY_FAIL")
        & front5.ge(0.52)
        & live_odds.le(100.0)
        & out["conservative_expected_roi_q20"].ge(1.35)
        & danger.lt(0.30)
        & pred_unc.lt(0.55)
    )
    difficulty_near_shadow = (
        single_reason
        & reasons.eq("RACE_DIFFICULTY_FAIL")
        & pred_unc.lt(0.62)
        & market_eff.lt(0.35)
        & live_odds.le(100.0)
        & out["conservative_expected_roi_q20"].ge(1.50)
    )

    out["shadow_action"] = np.select(
        [
            champion_pass,
            score_only_challenger_s,
            score_only_watch,
            margin_near_shadow,
            front_near_shadow,
            difficulty_near_shadow,
            hard_stop,
            multi_fail,
        ],
        [
            "CHAMPION_PREPOST_ELIGIBLE_SHADOW",
            "CHALLENGER_S_SCORE_ONLY_GREEN_SHADOW",
            "WATCH_SCORE_ONLY_NEAR_FAIL_NOT_GREEN",
            "CHALLENGER_MARGIN_NEAR_FAIL_SHADOW",
            "CHALLENGER_FRONT_NEAR_FAIL_SHADOW",
            "CHALLENGER_RACE_DIFFICULTY_NEAR_FAIL_SHADOW",
            "DO_NOT_PROMOTE_ABSOLUTE_GATE",
            "DO_NOT_PROMOTE_MULTIPLE_FAIL",
        ],
        default="WATCH_ONLY_SINGLE_FAIL",
    )
    out["shadow_buy_permission"] = "SHADOW_ONLY_NOT_LIVE_BUY"
    out["shadow_reason"] = np.select(
        [
            out["shadow_action"].eq("CHAMPION_PREPOST_ELIGIBLE_SHADOW"),
            out["shadow_action"].eq("CHALLENGER_S_SCORE_ONLY_GREEN_SHADOW"),
            out["shadow_action"].eq("WATCH_SCORE_ONLY_NEAR_FAIL_NOT_GREEN"),
            out["shadow_action"].eq("DO_NOT_PROMOTE_ABSOLUTE_GATE"),
            out["shadow_action"].eq("DO_NOT_PROMOTE_MULTIPLE_FAIL"),
        ],
        [
            "Champion gates pass in pre-post audit; keep as shadow/OOS fixed evidence only.",
            "Only SCORE misses, and value-survival, q20 EV, odds cap, fragility, and dispersion gates pass.",
            "Only SCORE misses, but value-survival or fragility gates are not strong enough for Challenger-S.",
            "Absolute gate failed such as missing odds or odds cap.",
            "Multiple gates failed; do not promote by threshold relaxation.",
        ],
        default="Single-gate or watch-only candidate; collect evidence without live promotion.",
    )
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("shadow_action", dropna=False)
        .agg(
            candidates=("pair_key", "count"),
            races=("race_id", "nunique"),
            avg_live_odds=("live_odds", "mean"),
            avg_t5_odds=("t5_odds", "mean"),
            avg_final_odds=("final_odds", "mean"),
            avg_final_odds_p20=("final_odds_pred_p20", "mean"),
            avg_value_survival_probability=("value_survival_probability", "mean"),
            avg_required_min_odds=("required_min_odds", "mean"),
            avg_conservative_ev_q20=("conservative_expected_roi_q20", "mean"),
            avg_score=("strongest_current_score", "mean"),
            avg_score_gap_z=("score_gap_z", "mean"),
            avg_front5=("projected_front5_prob", "mean"),
            avg_pair_danger=("pair_danger_score_external_ai", "mean"),
            avg_pair_fragility=("pair_fragility_proxy", "mean"),
            avg_prediction_uncertainty=("prediction_uncertainty_score", "mean"),
        )
        .reset_index()
        .sort_values(["candidates"], ascending=False)
    )


def add_proxy_returns(df: pd.DataFrame, pnl: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    top2 = finish_top2_map(pnl)
    if top2.empty:
        out["proxy_hit_umaren_top2"] = False
        out["proxy_stake_yen"] = 100.0
        out["proxy_return_yen"] = 0.0
        out["proxy_profit_yen"] = -100.0
        return out
    a, b = parse_pair_key(text(out, "pair_key"))
    out["race_id"] = out["race_id"].astype(str)
    out["pair_a_no"] = a
    out["pair_b_no"] = b
    out = out.merge(top2, on="race_id", how="left")
    top_lo = np.minimum(out["top1"], out["top2"])
    top_hi = np.maximum(out["top1"], out["top2"])
    pair_lo = np.minimum(out["pair_a_no"], out["pair_b_no"])
    pair_hi = np.maximum(out["pair_a_no"], out["pair_b_no"])
    hit = pair_lo.eq(top_lo) & pair_hi.eq(top_hi)
    out["proxy_hit_umaren_top2"] = hit.fillna(False)
    out["proxy_stake_yen"] = 100.0
    out["proxy_return_yen"] = np.where(out["proxy_hit_umaren_top2"], num(out, "live_odds", 0.0) * 100.0, 0.0)
    out["proxy_profit_yen"] = out["proxy_return_yen"] - out["proxy_stake_yen"]
    return out


def summarize_returns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "proxy_return_yen" not in df.columns:
        return pd.DataFrame()
    rows = []
    for action, part in df.groupby("shadow_action", dropna=False):
        stake = float(num(part, "proxy_stake_yen", 100.0).sum())
        ret = float(num(part, "proxy_return_yen", 0.0).sum())
        rows.append(
            {
                "shadow_action": action,
                "candidates": int(len(part)),
                "races": int(part["race_id"].nunique()),
                "hits": int(part["proxy_hit_umaren_top2"].sum()),
                "stake_yen": stake,
                "return_yen": ret,
                "profit_yen": ret - stake,
                "proxy_roi_pct": (ret / stake * 100.0) if stake > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["candidates"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shadow-only Challenger candidates from rejection ledger and odds survival model.")
    parser.add_argument("--ledger-csv", default="outputs/analysis/candidate_rejection_ledger_prepost_sim_v1/candidate_rejection_ledger.csv")
    parser.add_argument("--survival-csv", default="outputs/analysis/final_odds_survival_model_v1/final_odds_survival_dataset.csv")
    parser.add_argument("--pnl-detail-csv", default="outputs/analysis/current_live_pnl/current_live_pnl_detail.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/shadow_challenger_candidates_v1")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = read_csv(project_path(args.ledger_csv))
    survival = read_csv(project_path(args.survival_csv))
    pnl = read_csv(project_path(args.pnl_detail_csv))
    if ledger.empty or survival.empty:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "rows": 0,
            "reason": "missing ledger or survival dataset",
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    survival = survival.copy()
    survival["pair_key"] = simple_pair_key(text(survival, "pair_id"))
    survival_cols = [
        "pair_key",
        "t5_odds",
        "t5_snapshot_at",
        "t3_odds",
        "t3_snapshot_at",
        "final_odds",
        "final_snapshot_at",
        "final_odds_pred_p10",
        "final_odds_pred_p20",
        "final_odds_pred_p50",
        "final_odds_pred_p80",
        "final_odds_pred_p90",
        "conservative_expected_roi",
        "conservative_expected_roi_p20",
        "median_expected_roi",
        "log_final_over_t5",
        "log_final_over_t3",
    ]
    survival_cols = [c for c in survival_cols if c in survival.columns]
    merged = ledger.merge(survival[survival_cols].drop_duplicates("pair_key"), on="pair_key", how="left")
    merged = add_shadow_actions(merged)
    merged = add_proxy_returns(merged, pnl)

    sort_cols = ["shadow_action", "race_id", "conservative_expected_roi_q20", "strongest_current_score"]
    merged = merged.sort_values(sort_cols, ascending=[True, True, False, False])
    action_summary = summarize(merged)
    return_summary = summarize_returns(merged)
    candidate_focus = merged[
        merged["shadow_action"].str.startswith("CHAMPION_", na=False)
        | merged["shadow_action"].str.startswith("CHALLENGER_", na=False)
    ].copy()

    merged.to_csv(out_dir / "shadow_challenger_candidates.csv", index=False, encoding="utf-8-sig")
    candidate_focus.to_csv(out_dir / "shadow_challenger_focus.csv", index=False, encoding="utf-8-sig")
    action_summary.to_csv(out_dir / "shadow_challenger_action_summary.csv", index=False, encoding="utf-8-sig")
    return_summary.to_csv(out_dir / "shadow_challenger_return_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(merged)),
        "races": int(merged["race_id"].nunique()),
        "focus_rows": int(len(candidate_focus)),
        "focus_races": int(candidate_focus["race_id"].nunique()) if not candidate_focus.empty else 0,
        "challenger_s_rows": int(merged["shadow_action"].eq("CHALLENGER_S_SCORE_ONLY_GREEN_SHADOW").sum()),
        "challenger_s_races": int(merged.loc[merged["shadow_action"].eq("CHALLENGER_S_SCORE_ONLY_GREEN_SHADOW"), "race_id"].nunique()),
        "action_summary": action_summary.to_dict(orient="records"),
        "return_summary": return_summary.to_dict(orient="records"),
        "policy": {
            "status": "shadow_only",
            "live_buy_change": False,
            "champion_changed": False,
            "score_only_rule": {
                "single_rejection_reason": "SCORE_FAIL",
                "score_gap_z_max": CHALLENGER_SCORE_GAP_Z_MAX,
                "max_umaren_odds": MAX_CHAMPION_UMAREN_ODDS,
                "min_value_survival_probability": 0.80,
                "min_conservative_expected_roi_q20": 1.10,
                "safety_margin": SAFETY_MARGIN,
                "fragility_rule": "not worse than Champion-pass p75",
                "dispersion_rule": "not worse than Champion-pass p90",
            },
            "note": "External-AI Challenger candidates are observed only; Champion BUY gates are unchanged.",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
