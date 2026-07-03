from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.apply_runtime_odds_decision_rules import _decision_summary, _runtime_metrics, apply_decisions
from scripts.optimize_operational_win_addon import _json_default
from scripts.strict_pair_probability_roi_protocol import (
    add_calibrated_probs,
    build_raw_probability_features,
    load_universe,
    max_drawdown,
    num,
)
from src.utils.paths import ensure_dir, project_path


PAIR_TYPES = {"wide", "umaren"}


def _race_year(df: pd.DataFrame) -> pd.Series:
    if "year" in df.columns:
        year = pd.to_numeric(df["year"], errors="coerce")
    else:
        year = pd.to_numeric(df["race_id"].astype(str).str[:4], errors="coerce")
    return year.astype("Int64")


def _pair_key(df: pd.DataFrame) -> pd.Series:
    a = pd.to_numeric(df.get("anchor_no"), errors="coerce")
    b = pd.to_numeric(df.get("partner_no"), errors="coerce")
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return (
        df["race_id"].astype(str)
        + ":"
        + lo.astype("Int64").astype(str)
        + "-"
        + hi.astype("Int64").astype(str)
    )


def _metrics_from_stake(df: pd.DataFrame, label: str, stake_col: str, return_col: str) -> dict:
    selected = df[pd.to_numeric(df.get(stake_col), errors="coerce").fillna(0.0).gt(0)].copy()
    if selected.empty:
        return {
            "label": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
        }
    stake = float(pd.to_numeric(selected[stake_col], errors="coerce").fillna(0.0).sum())
    ret = float(pd.to_numeric(selected[return_col], errors="coerce").fillna(0.0).sum())
    race = selected.groupby("race_id", sort=False).agg(
        stake=(stake_col, "sum"),
        ret=(return_col, "sum"),
        hit=("hit", "max"),
    )
    profit = race["ret"] - race["stake"]
    return {
        "label": label,
        "tickets": int(len(selected)),
        "races": int(selected["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(selected["hit"].astype(bool).mean()) if "hit" in selected.columns else 0.0,
        "race_hit_rate": float(race["hit"].astype(bool).mean()) if "hit" in race.columns else 0.0,
        "max_drawdown_yen": max_drawdown(profit),
    }


def _score_universe_no_leak(universe: pd.DataFrame, ticket_years: list[int], min_train_rows: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in ticket_years:
        train = universe[universe["year"].lt(year)].copy()
        apply_to = universe[universe["year"].eq(year)].copy()
        if len(train) < min_train_rows or apply_to.empty:
            continue
        scored, meta = add_calibrated_probs(train, apply_to)
        scored["pair_calibration_year"] = year
        scored["pair_calibration_train_rows"] = int(len(train))
        scored["pair_calibration_train_min_year"] = int(train["year"].min())
        scored["pair_calibration_train_max_year"] = int(train["year"].max())
        for key, value in meta.items():
            scored[f"pair_calibration_{key}"] = value
        frames.append(scored)
    if not frames:
        return pd.DataFrame()
    scored_all = pd.concat(frames, ignore_index=True, sort=False)
    scored_all["pair_calibration_pair_key"] = _pair_key(scored_all)
    keep = [
        "race_id",
        "pair_calibration_pair_key",
        "front_raw",
        "front5_prob_cal",
        "wide_joint_raw",
        "wide_joint_model_raw",
        "wide_hit_prob_cal",
        "wide_ev_proxy",
        "umaren_joint_raw",
        "umaren_joint_model_raw",
        "umaren_hit_prob_cal",
        "umaren_ev_proxy",
        "strict_rank_score",
        "pair_calibration_year",
        "pair_calibration_train_rows",
        "pair_calibration_train_min_year",
        "pair_calibration_train_max_year",
        "pair_calibration_front_fallback",
        "pair_calibration_wide_fallback",
        "pair_calibration_umaren_fallback",
    ]
    return scored_all[[c for c in keep if c in scored_all.columns]].drop_duplicates(
        ["race_id", "pair_calibration_pair_key"], keep="last"
    )


def apply_pair_calibration(
    tickets: pd.DataFrame,
    scored_pairs: pd.DataFrame,
    *,
    blend_weight: float,
    default_target_roi_wide: float,
    default_target_roi_umaren: float,
) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["year"] = _race_year(out)
    out["ticket_type"] = out.get("ticket_type", "").astype(str)
    out["pair_calibration_pair_key"] = _pair_key(out)

    out = out.merge(scored_pairs, on=["race_id", "pair_calibration_pair_key"], how="left")

    idx = out.index
    old_prob = num(out.get("ticket_hit_prob"), idx, np.nan)
    old_required = num(out.get("required_pay_per100"), idx, np.nan)
    old_quote = num(out.get("quote_pay_proxy_per100"), idx, np.nan)
    out["ticket_hit_prob_before_pair_calibration"] = old_prob
    out["required_pay_per100_before_pair_calibration"] = old_required

    wide_prob = num(out.get("wide_hit_prob_cal"), idx, np.nan)
    umaren_prob = num(out.get("umaren_hit_prob_cal"), idx, np.nan)
    pair_prob = pd.Series(np.nan, index=idx, dtype=float)
    pair_prob.loc[out["ticket_type"].eq("wide")] = wide_prob.loc[out["ticket_type"].eq("wide")]
    pair_prob.loc[out["ticket_type"].eq("umaren")] = umaren_prob.loc[out["ticket_type"].eq("umaren")]
    out["pair_calibrated_hit_prob"] = pair_prob
    out["pair_calibration_available"] = out["pair_calibrated_hit_prob"].notna() & out["ticket_type"].isin(PAIR_TYPES)

    blend = float(np.clip(blend_weight, 0.0, 1.0))
    new_prob = old_prob.copy()
    pair_mask = out["pair_calibration_available"]
    has_old = old_prob.gt(0)
    new_prob.loc[pair_mask & has_old] = (
        (1.0 - blend) * old_prob.loc[pair_mask & has_old]
        + blend * out.loc[pair_mask & has_old, "pair_calibrated_hit_prob"]
    )
    new_prob.loc[pair_mask & ~has_old] = out.loc[pair_mask & ~has_old, "pair_calibrated_hit_prob"]
    new_prob = new_prob.clip(0.001, 0.95)
    out["ticket_hit_prob"] = new_prob
    out["pair_calibration_blend_weight"] = np.where(pair_mask, blend, 0.0)

    implied_target_roi = (old_prob * old_required / 100.0).replace([np.inf, -np.inf], np.nan)
    fallback_target = pd.Series(np.nan, index=idx, dtype=float)
    fallback_target.loc[out["ticket_type"].eq("wide")] = default_target_roi_wide
    fallback_target.loc[out["ticket_type"].eq("umaren")] = default_target_roi_umaren
    target_roi = implied_target_roi.where(implied_target_roi.gt(0), fallback_target)
    recomputed_required = (100.0 * target_roi / new_prob.replace(0, np.nan)).clip(100.0, 20000.0)
    out.loc[pair_mask, "required_pay_per100"] = recomputed_required.loc[pair_mask]
    out.loc[pair_mask, "min_acceptable_odds"] = out.loc[pair_mask, "required_pay_per100"] / 100.0
    out.loc[pair_mask, "runtime_expected_roi"] = (
        out.loc[pair_mask, "ticket_hit_prob"]
        * old_quote.loc[pair_mask].fillna(num(out.get("quote_pay_proxy_per100"), idx, 0.0).loc[pair_mask])
        / 100.0
    )
    out.loc[pair_mask, "pair_probability_runtime_note"] = "walkforward_pair_probability_calibrated"
    out.loc[~pair_mask, "pair_probability_runtime_note"] = np.where(
        out.loc[~pair_mask, "ticket_type"].isin(PAIR_TYPES),
        "pair_calibration_unavailable_kept_existing",
        "not_pair_ticket_kept_existing",
    )
    return out


def _probability_shift_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for ticket_type, g in df.groupby("ticket_type", dropna=False):
        old = num(g.get("ticket_hit_prob_before_pair_calibration"), g.index, np.nan)
        new = num(g.get("ticket_hit_prob"), g.index, np.nan)
        rows.append(
            {
                "ticket_type": ticket_type,
                "tickets": int(len(g)),
                "calibrated_tickets": int(g.get("pair_calibration_available", pd.Series(False, index=g.index)).astype(bool).sum()),
                "avg_prob_before": float(old.mean()),
                "avg_prob_after": float(new.mean()),
                "avg_prob_delta": float((new - old).mean()),
                "median_prob_delta": float((new - old).median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject walk-forward calibrated pair probabilities into runtime tickets without changing the original strongest files."
    )
    parser.add_argument("--tickets-csv", default="outputs/analysis/robust_expansion_runtime_ready_v1/standard_plus_robust_runtime_ready_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/pair_probability_runtime_v1")
    parser.add_argument("--blend-weight", type=float, default=0.50)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--default-target-roi-wide", type=float, default=1.25)
    parser.add_argument("--default-target-roi-umaren", type=float, default=1.35)
    parser.add_argument("--evaluate-proxy", action="store_true", help="Also run runtime odds decisions using proxy odds for historical comparison.")
    args = parser.parse_args()

    out_dir = ensure_dir(project_path(args.output_dir))
    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    ticket_years = sorted(int(y) for y in _race_year(tickets).dropna().unique())

    universe = build_raw_probability_features(load_universe())
    scored_pairs = _score_universe_no_leak(universe, ticket_years, args.min_train_rows)
    scored_pairs.to_csv(out_dir / "walkforward_pair_probability_scores.csv", index=False, encoding="utf-8-sig")

    calibrated = apply_pair_calibration(
        tickets,
        scored_pairs,
        blend_weight=args.blend_weight,
        default_target_roi_wide=args.default_target_roi_wide,
        default_target_roi_umaren=args.default_target_roi_umaren,
    )
    calibrated.to_csv(out_dir / "pair_calibrated_runtime_tickets.csv", index=False, encoding="utf-8-sig")

    coverage = (
        calibrated.groupby(["year", "ticket_type"], dropna=False)
        .agg(
            tickets=("race_id", "size"),
            races=("race_id", "nunique"),
            calibrated_tickets=("pair_calibration_available", "sum"),
            avg_pair_prob=("pair_calibrated_hit_prob", "mean"),
        )
        .reset_index()
    )
    coverage.to_csv(out_dir / "pair_calibration_coverage.csv", index=False, encoding="utf-8-sig")
    shift = _probability_shift_summary(calibrated)
    shift.to_csv(out_dir / "probability_shift_summary.csv", index=False, encoding="utf-8-sig")

    metrics_rows = []
    if "runtime_stake_yen" in tickets.columns and "runtime_return_yen" in tickets.columns:
        metrics_rows.append(_metrics_from_stake(tickets, "input_runtime_ready", "runtime_stake_yen", "runtime_return_yen"))
    if args.evaluate_proxy:
        decisions = apply_decisions(calibrated, pair_live=None, single_live=None, use_proxy_when_missing=True)
        decisions.to_csv(out_dir / "proxy_runtime_ticket_decisions.csv", index=False, encoding="utf-8-sig")
        selected = decisions[decisions["runtime_stake_yen"].gt(0)].copy()
        selected.to_csv(out_dir / "proxy_runtime_selected_tickets.csv", index=False, encoding="utf-8-sig")
        _decision_summary(decisions).to_csv(out_dir / "proxy_runtime_decision_summary.csv", index=False, encoding="utf-8-sig")
        metrics_rows.append(_runtime_metrics(decisions, "pair_calibrated_proxy_runtime"))
        if "year" in decisions.columns:
            for year, g in decisions.groupby("year"):
                metrics_rows.append(_runtime_metrics(g, f"pair_calibrated_proxy_year_{int(year)}"))
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "tickets_csv": args.tickets_csv,
        "ticket_years": ticket_years,
        "blend_weight": args.blend_weight,
        "min_train_rows": args.min_train_rows,
        "scored_pair_rows": int(len(scored_pairs)),
        "calibrated_tickets": int(calibrated["pair_calibration_available"].sum()),
        "calibrated_pair_types": calibrated.loc[calibrated["pair_calibration_available"], "ticket_type"].value_counts().to_dict(),
        "coverage_csv": str(out_dir / "pair_calibration_coverage.csv"),
        "tickets_out": str(out_dir / "pair_calibrated_runtime_tickets.csv"),
        "metrics": metrics.to_dict(orient="records"),
        "note": "No-leak calibration: each ticket year only uses earlier universe years. 2024 tickets keep existing probabilities because no earlier pair universe is available.",
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
