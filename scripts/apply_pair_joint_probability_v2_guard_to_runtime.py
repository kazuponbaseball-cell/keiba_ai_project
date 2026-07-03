from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from refine_pair_joint_probability_v2 import (
    DEFAULT_FRONT5,
    DEFAULT_PAIR_UNIVERSE,
    DEFAULT_RUNNER_CONTEXT,
    FEATURES,
    add_context_side,
    add_pair_features,
    fit_calibrator,
    fit_logistic,
    load_runner_context,
    num,
    project_path,
)


DEFAULT_CANDIDATES = "outputs/analysis/current_strongest_runtime_v1/current_strongest_all_candidates.csv"
DEFAULT_TICKETS = "outputs/analysis/current_strongest_runtime_v1/selected_after_live_safety.csv"
DEFAULT_PAIR_ODDS = "data/processed/live_odds/realtime_pair_odds_latest.csv"
DEFAULT_OUT = "outputs/analysis/current_strongest_runtime_v1/pair_joint_v2_runtime_guard"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def first_existing(frame: pd.DataFrame, names: list[str], default: float = np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return num(frame, name, default)
    return pd.Series(default, index=frame.index, dtype=float)


def pair_key(frame: pd.DataFrame) -> pd.Series:
    a = first_existing(frame, ["anchor_no", "anchor_horse_no", "a_no"], np.nan)
    b = first_existing(frame, ["partner_no", "partner_horse_no", "b_no"], np.nan)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return frame["race_id"].astype(str) + ":" + lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)


def load_historical_training(pair_universe_path: Path, runner_context_path: Path, front5_path: Path) -> pd.DataFrame:
    universe = pd.read_csv(pair_universe_path, encoding="utf-8-sig", low_memory=False)
    universe["race_id"] = universe["race_id"].astype(str)
    context = load_runner_context(runner_context_path, front5_path if front5_path.exists() else None)
    universe = add_context_side(universe, context, "anchor", "anchor_no")
    universe = add_context_side(universe, context, "partner", "partner_no")
    return add_pair_features(universe)


def fit_full_history_models(history: pd.DataFrame) -> dict[str, Any]:
    wide_model = fit_logistic(history, "wide_label", FEATURES)
    umaren_model = fit_logistic(history, "umaren_label", FEATURES)
    train = history.copy()
    train["joint_v2_wide_raw"] = wide_model.predict(train)
    train["joint_v2_umaren_raw"] = umaren_model.predict(train)
    return {
        "wide_model": wide_model,
        "umaren_model": umaren_model,
        "wide_calibrator": fit_calibrator(train, "joint_v2_wide_raw", "wide_label"),
        "umaren_calibrator": fit_calibrator(train, "joint_v2_umaren_raw", "umaren_label"),
        "train_rows": int(len(train)),
        "train_min_year": int(train["year"].min()),
        "train_max_year": int(train["year"].max()),
    }


def runtime_features(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["year"] = pd.to_numeric(out["race_id"].str[:4], errors="coerce").fillna(0).astype(int)
    out["anchor_no"] = first_existing(out, ["anchor_no", "anchor_horse_no", "a_no"], np.nan)
    out["partner_no"] = first_existing(out, ["partner_no", "partner_horse_no", "b_no"], np.nan)
    out["pair_key"] = pair_key(out)

    out["anchor_odds"] = first_existing(out, ["anchor_live_win_odds", "anchor_odds"], 1.0).clip(lower=1.0)
    out["partner_odds"] = first_existing(out, ["partner_live_win_odds", "partner_odds"], 1.0).clip(lower=1.0)
    out["anchor_ai_win_prob"] = first_existing(out, ["anchor_ai_prob", "anchor_ai_win_prob"], 0.0).clip(0, 1)
    out["partner_ai_win_prob"] = first_existing(out, ["partner_ai_prob", "partner_ai_win_prob"], 0.0).clip(0, 1)
    out["anchor_pop"] = first_existing(out, ["anchor_pop", "anchor_ai_rank_num"], 99.0)
    out["partner_pop"] = first_existing(out, ["partner_pop", "partner_ai_rank_num", "popularity"], 99.0)
    out["anchor_market_win_prob"] = (1.0 / out["anchor_odds"]).clip(0, 1)
    out["partner_market_win_prob"] = (1.0 / out["partner_odds"]).clip(0, 1)
    out["anchor_win_score"] = out["anchor_ai_win_prob"]
    out["partner_win_score"] = out["partner_ai_win_prob"]
    out["anchor_place_score"] = first_existing(out, ["anchor_place_score"], 0.0).clip(0, 1)
    out["partner_place_score"] = first_existing(out, ["partner_place_score"], 0.0).clip(0, 1)
    out["anchor_quinella_score"] = first_existing(out, ["anchor_quinella_score"], 0.0).clip(0, 1)
    out["partner_quinella_score"] = first_existing(out, ["partner_quinella_score"], 0.0).clip(0, 1)
    out["wide_axis_score"] = out["anchor_quinella_score"]
    out["wide_partner_score"] = out["partner_quinella_score"]
    out["market_overlay_score"] = first_existing(out, ["market_overlay_score"], 0.0).clip(0, 1)
    out["late_value_survives_score"] = first_existing(out, ["late_value_survives_score"], 0.0).clip(0, 1)
    out["projected_front5_prob"] = first_existing(
        out,
        ["projected_front5_prob", "anchor_projected_front5_prob", "anchor_front5_model_prob"],
        0.5,
    ).clip(0.02, 0.98)
    out["pair_score"] = first_existing(out, ["pair_score"], 0.0)
    out["pair_quinella_score"] = first_existing(out, ["pair_quinella_score"], 0.0)

    a_front = first_existing(out, ["anchor_front5_model_prob", "anchor_projected_front5_prob"], 0.5).clip(0.02, 0.98)
    p_front = first_existing(out, ["partner_front5_model_prob", "partner_projected_front5_prob"], 0.5).clip(0.02, 0.98)
    a_closer = (1.0 - a_front).clip(0.02, 0.98)
    p_closer = (1.0 - p_front).clip(0.02, 0.98)
    pressure = first_existing(out, ["anchor_race_early_pressure_score", "partner_race_early_pressure_score"], 0.0).clip(0, 1)
    collapse = first_existing(out, ["anchor_race_pace_collapse_risk", "partner_race_pace_collapse_risk"], 0.0).clip(0, 1)
    slow = first_existing(out, ["anchor_race_slow_pace_risk", "partner_race_slow_pace_risk"], 0.0).clip(0, 1)
    front_adv = first_existing(out, ["anchor_front_advantage_score", "partner_front_advantage_score"], 0.0).clip(0, 1)

    out["joint_place_product"] = np.sqrt((out["anchor_place_score"] * out["partner_place_score"]).clip(0, 1))
    out["joint_win_product"] = np.sqrt((out["anchor_win_score"] * out["partner_win_score"]).clip(0, 1))
    out["joint_q_product"] = np.sqrt((out["anchor_quinella_score"] * out["partner_quinella_score"]).clip(0, 1))
    out["joint_market_product"] = np.sqrt((out["anchor_market_win_prob"] * out["partner_market_win_prob"]).clip(0, 1))
    out["front_pair_min"] = np.minimum(a_front, p_front)
    out["front_pair_max"] = np.maximum(a_front, p_front)
    out["closer_pair_max"] = np.maximum(a_closer, p_closer)
    out["front_closer_complement"] = np.maximum(a_front * p_closer, p_front * a_closer)
    out["front_front_clash"] = a_front * p_front * pressure
    out["front_front_slow_fit"] = a_front * p_front * (0.50 * slow + 0.50 * front_adv)
    out["collapse_fit"] = collapse * (0.55 * out["closer_pair_max"] + 0.45 * out["front_closer_complement"])
    out["style_diversity"] = (a_front - p_front).abs() + (a_closer - p_closer).abs()
    out["danger_sum"] = (
        first_existing(out, ["anchor_danger_popular_score", "ticket_danger_popular_score"], 0.0).clip(0, 1)
        + first_existing(out, ["partner_danger_popular_score", "ticket_danger_popular_in_pair_score"], 0.0).clip(0, 1)
    )
    out["danger_max"] = np.maximum(
        first_existing(out, ["anchor_danger_popular_score", "ticket_danger_popular_score"], 0.0).clip(0, 1),
        first_existing(out, ["partner_danger_popular_score", "ticket_danger_popular_in_pair_score"], 0.0).clip(0, 1),
    )
    out["skip_risk_score"] = first_existing(out, ["skip_risk_score"], 0.0).clip(0, 1)
    out["odds_geom"] = np.sqrt(out["anchor_odds"] * out["partner_odds"])
    out["partner_value_flag"] = ((out["partner_pop"] >= 4) | (out["partner_odds"] >= 8.0)).astype(float)
    out["wide_quote_proxy"] = 100.0 * out["odds_geom"] * 0.45
    out["umaren_quote_proxy"] = 100.0 * (out["anchor_odds"] * out["partner_odds"] * 0.32).clip(1.3, 260.0)
    return out


def score_runtime_candidates(candidates: pd.DataFrame, models: dict[str, Any], prob_threshold: float, ev_threshold: float) -> pd.DataFrame:
    out = runtime_features(candidates)
    out["joint_v2_wide_raw"] = models["wide_model"].predict(out)
    out["joint_v2_umaren_raw"] = models["umaren_model"].predict(out)
    out["joint_v2_wide_prob"] = models["wide_calibrator"].apply(out["joint_v2_wide_raw"])
    out["joint_v2_umaren_prob"] = models["umaren_calibrator"].apply(out["joint_v2_umaren_raw"])
    out["joint_v2_wide_ev_proxy"] = out["joint_v2_wide_prob"] * out["wide_quote_proxy"] / 100.0
    out["joint_v2_umaren_ev_proxy"] = out["joint_v2_umaren_prob"] * out["umaren_quote_proxy"] / 100.0
    live_odds = first_existing(out, ["live_odds"], np.nan)
    out["joint_v2_live_expected_roi"] = np.where(
        out.get("ticket_type", "").astype(str).eq("wide"),
        out["joint_v2_wide_prob"] * live_odds,
        out["joint_v2_umaren_prob"] * live_odds,
    )
    wide_mask = out.get("ticket_type", "").astype(str).eq("wide")
    out["joint_v2_runtime_guard_ok"] = (
        wide_mask
        & out["joint_v2_wide_prob"].ge(prob_threshold)
        & out["joint_v2_wide_ev_proxy"].ge(ev_threshold)
    )
    out["joint_v2_runtime_guard_status"] = np.select(
        [
            out["joint_v2_runtime_guard_ok"],
            wide_mask,
            out.get("ticket_type", "").astype(str).eq("umaren"),
        ],
        ["OK", "WEAK_JOINT", "UMAREN_SHADOW_ONLY"],
        default="NOT_PAIR",
    )
    out["joint_v2_runtime_guard_reason"] = np.select(
        [
            out["joint_v2_runtime_guard_ok"],
            wide_mask,
            out.get("ticket_type", "").astype(str).eq("umaren"),
        ],
        [
            "existing value candidate is confirmed by pair joint probability",
            "existing candidate lacks enough pair joint confirmation",
            "umaren V2 is informative but too payoff-concentrated for final BUY gating",
        ],
        default="joint guard is not applied",
    )
    return out


def normalized_pair_columns(frame: pd.DataFrame, a_col: str = "a_no", b_col: str = "b_no") -> pd.DataFrame:
    out = frame.copy()
    a = pd.to_numeric(out[a_col], errors="coerce")
    b = pd.to_numeric(out[b_col], errors="coerce")
    out["_a_lo"] = np.minimum(a, b).astype("Int64")
    out["_b_hi"] = np.maximum(a, b).astype("Int64")
    return out


def build_wide_shadow_candidates(candidates: pd.DataFrame, pair_odds_path: Path) -> pd.DataFrame:
    if candidates.empty or not pair_odds_path.exists():
        return pd.DataFrame()
    odds = pd.read_csv(pair_odds_path, dtype={"race_id": str}, encoding="utf-8-sig", low_memory=False)
    if odds.empty or "ticket_type" not in odds.columns:
        return pd.DataFrame()
    wide = odds[odds["ticket_type"].astype(str).eq("wide")].copy()
    if wide.empty:
        return pd.DataFrame()
    wide = normalized_pair_columns(wide, "a_no", "b_no")
    wide = wide.drop_duplicates(["race_id", "_a_lo", "_b_hi"], keep="last")
    keep = [
        "race_id",
        "_a_lo",
        "_b_hi",
        "live_pay_per100",
        "live_odds",
        "popularity",
        "snapshot_at",
        "parser_mode",
        "source",
        "live_odds_min",
        "live_odds_max",
    ]
    base = normalized_pair_columns(candidates, "a_no", "b_no")
    shadow = base.merge(wide[[c for c in keep if c in wide.columns]], on=["race_id", "_a_lo", "_b_hi"], how="inner", suffixes=("", "_wide"))
    if shadow.empty:
        return shadow
    for col in ["live_pay_per100", "live_odds", "popularity", "snapshot_at", "parser_mode", "source", "live_odds_min", "live_odds_max"]:
        wide_col = f"{col}_wide"
        if wide_col in shadow.columns:
            shadow[col] = shadow[wide_col]
    shadow["ticket_type"] = "wide"
    shadow["wide_shadow_from_umaren_candidate"] = True
    shadow["runtime_expected_roi_original_umaren"] = pd.to_numeric(
        candidates.get("runtime_expected_roi", pd.Series(np.nan, index=candidates.index)), errors="coerce"
    ).reindex(shadow.index)
    return shadow.drop(columns=[c for c in shadow.columns if c.endswith("_wide") or c in {"_a_lo", "_b_hi"}], errors="ignore")


def merge_ticket_annotations(tickets: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty:
        return tickets.copy()
    out = tickets.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["pair_key"] = pair_key(out)
    ann = scored.drop_duplicates(["race_id", "pair_key"], keep="last")
    keep = [
        "race_id",
        "pair_key",
        "joint_v2_wide_prob",
        "joint_v2_umaren_prob",
        "joint_v2_wide_ev_proxy",
        "joint_v2_umaren_ev_proxy",
        "joint_v2_live_expected_roi",
        "joint_v2_runtime_guard_ok",
        "joint_v2_runtime_guard_status",
        "joint_v2_runtime_guard_reason",
    ]
    out = out.merge(ann[[c for c in keep if c in ann.columns]], on=["race_id", "pair_key"], how="left")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-universe", default=DEFAULT_PAIR_UNIVERSE)
    parser.add_argument("--runner-context", default=DEFAULT_RUNNER_CONTEXT)
    parser.add_argument("--front5", default=DEFAULT_FRONT5)
    parser.add_argument("--candidates-csv", default=DEFAULT_CANDIDATES)
    parser.add_argument("--tickets-csv", default=DEFAULT_TICKETS)
    parser.add_argument("--pair-odds-csv", default=DEFAULT_PAIR_ODDS)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--prob-threshold", type=float, default=0.045)
    parser.add_argument("--ev-threshold", type=float, default=0.34)
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = load_historical_training(project_path(args.pair_universe), project_path(args.runner_context), project_path(args.front5))
    models = fit_full_history_models(history)
    candidates_path = project_path(args.candidates_csv)
    candidates = pd.read_csv(candidates_path, dtype={"race_id": str}, encoding="utf-8-sig", low_memory=False)
    scored = score_runtime_candidates(candidates, models, args.prob_threshold, args.ev_threshold)
    scored.to_csv(out_dir / "current_candidates_with_joint_v2_guard.csv", index=False, encoding="utf-8-sig")

    wide_shadow = build_wide_shadow_candidates(candidates, project_path(args.pair_odds_csv))
    if wide_shadow.empty:
        wide_shadow_scored = pd.DataFrame()
        wide_shadow_ok = pd.DataFrame()
    else:
        wide_shadow_scored = score_runtime_candidates(wide_shadow, models, args.prob_threshold, args.ev_threshold)
        wide_shadow_scored.to_csv(out_dir / "wide_shadow_candidates_with_joint_v2_guard.csv", index=False, encoding="utf-8-sig")
        wide_shadow_ok = wide_shadow_scored[wide_shadow_scored["joint_v2_runtime_guard_ok"]].copy()
        if not wide_shadow_ok.empty:
            wide_shadow_ok = wide_shadow_ok.sort_values(
                ["race_id", "joint_v2_wide_ev_proxy", "joint_v2_wide_prob", "live_odds"],
                ascending=[True, False, False, False],
            )
            wide_shadow_ok = wide_shadow_ok.groupby("race_id", as_index=False).head(1).sort_values(
                ["joint_v2_wide_ev_proxy", "joint_v2_wide_prob"],
                ascending=[False, False],
            )
        wide_shadow_ok.to_csv(out_dir / "wide_shadow_guard_ok_candidates.csv", index=False, encoding="utf-8-sig")

    tickets_path = project_path(args.tickets_csv)
    ticket_rows = pd.read_csv(tickets_path, dtype={"race_id": str}, encoding="utf-8-sig", low_memory=False) if tickets_path.exists() else pd.DataFrame()
    annotated_tickets = merge_ticket_annotations(ticket_rows, scored)
    annotated_tickets.to_csv(out_dir / "selected_tickets_with_joint_v2_guard.csv", index=False, encoding="utf-8-sig")

    summary = {
        "candidates_csv": str(candidates_path),
        "tickets_csv": str(tickets_path),
        "pair_odds_csv": str(project_path(args.pair_odds_csv)),
        "train_rows": models["train_rows"],
        "train_min_year": models["train_min_year"],
        "train_max_year": models["train_max_year"],
        "prob_threshold": args.prob_threshold,
        "ev_threshold": args.ev_threshold,
        "candidate_rows": int(len(scored)),
        "wide_candidate_rows": int(scored.get("ticket_type", "").astype(str).eq("wide").sum()),
        "joint_guard_ok_rows": int(scored["joint_v2_runtime_guard_ok"].sum()),
        "wide_shadow_rows": int(len(wide_shadow_scored)),
        "wide_shadow_guard_ok_rows": int(len(wide_shadow_ok)),
        "wide_shadow_guard_ok_races": int(wide_shadow_ok["race_id"].nunique()) if not wide_shadow_ok.empty else 0,
        "ticket_rows": int(len(annotated_tickets)),
        "ticket_guard_ok_rows": int(annotated_tickets.get("joint_v2_runtime_guard_ok", pd.Series(False, index=annotated_tickets.index)).fillna(False).astype(bool).sum()) if not annotated_tickets.empty else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready({"ok": True, "out_dir": str(out_dir), **summary}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
