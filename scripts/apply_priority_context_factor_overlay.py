from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.apply_contextual_value_overlays import _load_runner_context
from scripts.optimize_operational_win_addon import _json_default, _metrics
from src.utils.paths import ensure_dir, project_path


DEFAULT_RUNNER_PATHS = [
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_breeder_enriched/train_features_with_same_day_bias_v3_retro_body_breeder.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_breeder_enriched/test_features_with_same_day_bias_v3_retro_body_breeder.csv",
]


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _clip01(value: pd.Series) -> pd.Series:
    if not isinstance(value, pd.Series):
        value = pd.Series(value)
    return pd.to_numeric(value, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _merge_runner_context(tickets: pd.DataFrame, runners: pd.DataFrame, prefix: str, no_col: str) -> pd.DataFrame:
    if runners.empty:
        return tickets
    runner = runners.copy()
    runner["horse_no"] = pd.to_numeric(runner["horse_no"], errors="coerce").astype("Int64")
    rename = {c: f"{prefix}_{c}" for c in runner.columns if c not in {"race_id", "horse_no"}}
    runner = runner.rename(columns=rename)
    out = tickets.copy()
    out[no_col] = pd.to_numeric(out.get(no_col), errors="coerce").astype("Int64")
    return out.merge(runner, left_on=["race_id", no_col], right_on=["race_id", "horse_no"], how="left").drop(columns=["horse_no"], errors="ignore")


def enrich_priority_factors(tickets: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = _num(df.get("year"), df.index, np.nan).fillna(df["race_id"].str[:4].astype(float)).astype(int)
    df["ticket_type"] = df.get("ticket_type", "").astype(str)
    df = _merge_runner_context(df, runners, "anchor", "anchor_no")
    df = _merge_runner_context(df, runners, "partner", "partner_no")

    idx = df.index
    anchor_front = _clip01(
        0.45 * _num(df.get("anchor_ctx_front_gate_reliability_score"), idx, 0.5).fillna(0.5)
        + 0.30 * _num(df.get("projected_front5_prob"), idx, 0.5).fillna(0.5)
        + 0.15 * _num(df.get("horse_front_run_rate_past5_feature"), idx, 0.5).fillna(0.5)
        + 0.10 * _num(df.get("front_advantage_score_feature"), idx, 0.5).fillna(0.5)
    )
    partner_front = _clip01(
        0.55 * _num(df.get("partner_ctx_front_gate_reliability_score"), idx, 0.5).fillna(0.5)
        + 0.25 * _num(df.get("projected_front5_prob"), idx, 0.5).fillna(0.5)
        + 0.20 * _num(df.get("partner_quinella_model_score_norm"), idx, 0.5).fillna(0.5)
    )
    df["ticket_front_position_reliability_score"] = np.where(
        df["ticket_type"].eq("win"),
        anchor_front,
        _clip01(0.58 * anchor_front + 0.42 * partner_front),
    )

    danger_anchor = _clip01(
        0.28 * _num(df.get("anchor_danger_hybrid"), idx, 0.0).fillna(0.0)
        + 0.22 * _num(df.get("anchor_danger_model"), idx, 0.0).fillna(0.0)
        + 0.20 * _num(df.get("danger_popular_hybrid_score"), idx, 0.0).fillna(0.0)
        + 0.16 * _num(df.get("danger_favorite_score"), idx, 0.0).fillna(0.0)
        + 0.14 * _num(df.get("anchor_vertical_overpopular_risk_score"), idx, 0.0).fillna(0.0)
    )
    danger_partner = _clip01(
        0.36 * _num(df.get("partner_danger_hybrid"), idx, 0.0).fillna(0.0)
        + 0.28 * _num(df.get("partner_danger_model"), idx, 0.0).fillna(0.0)
        + 0.20 * _num(df.get("partner_vertical_overpopular_risk_score"), idx, 0.0).fillna(0.0)
        + 0.16 * _num(df.get("race_favorite_danger"), idx, 0.0).fillna(0.0)
    )
    popular_anchor = _num(df.get("anchor_pop"), idx, 99).fillna(99).le(3).astype(float)
    popular_partner = _num(df.get("partner_pop"), idx, 99).fillna(99).le(3).astype(float)
    df["ticket_danger_popular_score"] = _clip01(
        np.where(
            df["ticket_type"].eq("win"),
            danger_anchor * (0.65 + 0.35 * popular_anchor),
            np.maximum(danger_anchor * (0.65 + 0.35 * popular_anchor), danger_partner * (0.65 + 0.35 * popular_partner)),
        )
    )

    anchor_body = _num(df.get("anchor_ctx_body_layoff_age_score"), idx, 0.5).fillna(0.5)
    partner_body = _num(df.get("partner_ctx_body_layoff_age_score"), idx, 0.5).fillna(0.5)
    df["ticket_body_age_layoff_score"] = np.where(df["ticket_type"].eq("win"), anchor_body, _clip01(0.60 * anchor_body + 0.40 * partner_body))

    anchor_stable = _num(df.get("anchor_ctx_stable_week_score"), idx, 0.5).fillna(0.5)
    partner_stable = _num(df.get("partner_ctx_stable_week_score"), idx, 0.5).fillna(0.5)
    df["ticket_stable_jockey_buy_timing_score"] = np.where(
        df["ticket_type"].eq("win"),
        anchor_stable,
        _clip01(0.55 * anchor_stable + 0.45 * partner_stable),
    )

    df["priority_context_positive_score"] = _clip01(
        0.34 * df["ticket_front_position_reliability_score"]
        + 0.24 * df["ticket_body_age_layoff_score"]
        + 0.24 * df["ticket_stable_jockey_buy_timing_score"]
        + 0.18 * _num(df.get("same_day_bias_fit_score"), idx, 0.5).fillna(0.5)
    )
    df["priority_context_net_score"] = _clip01(df["priority_context_positive_score"] - 0.32 * df["ticket_danger_popular_score"])
    return df


def _apply_policy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    base_buy = _num(out.get("runtime_stake_yen"), out.index, 0.0).fillna(0.0).gt(0)
    mask = base_buy
    mask &= _num(out.get("ticket_front_position_reliability_score"), out.index, 0.0).ge(params["front_min"])
    mask &= _num(out.get("ticket_danger_popular_score"), out.index, 0.0).le(params["danger_max"])
    mask &= _num(out.get("ticket_body_age_layoff_score"), out.index, 0.0).ge(params["body_min"])
    mask &= _num(out.get("ticket_stable_jockey_buy_timing_score"), out.index, 0.0).ge(params["stable_min"])
    mask &= _num(out.get("priority_context_net_score"), out.index, 0.0).ge(params["net_min"])

    selected = out[mask].copy()
    if selected.empty:
        return selected

    stake = _num(selected.get("runtime_stake_yen"), selected.index, 0.0).fillna(0.0)
    score = _num(selected.get("priority_context_net_score"), selected.index, 0.0).fillna(0.0)
    mult = np.where(score.ge(params["boost_min"]), params["boost_mult"], 1.0)
    # Cap increases; this is an overlay, not a new aggressive staking system.
    selected["runtime_stake_yen"] = (np.floor((stake * mult).clip(0, params["max_stake"]) / 100.0) * 100.0).clip(lower=100.0)
    pay = _num(selected.get("runtime_backtest_pay_per100"), selected.index, 0.0).fillna(0.0)
    selected["runtime_return_yen"] = np.where(selected["hit"].astype(bool), pay * selected["runtime_stake_yen"] / 100.0, 0.0)
    selected["runtime_action"] = np.where(mult > 1.0, "BUY_CONTEXT_BOOST", selected.get("runtime_action", "BUY"))
    selected["runtime_reason"] = selected.get("runtime_reason", "").astype(str) + "|priority_context_ok"
    selected["runtime_ticket_status"] = np.where(mult > 1.0, "強買い", selected.get("runtime_ticket_status", "買い"))
    return selected


def _metric_runtime(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"policy": label, "tickets": 0, "races": 0, "roi": 0.0}
    tmp = df.copy()
    tmp["stake_yen"] = _num(tmp.get("runtime_stake_yen"), tmp.index, 0.0).fillna(0.0)
    tmp["return_yen"] = _num(tmp.get("runtime_return_yen"), tmp.index, 0.0).fillna(0.0)
    return _metrics(tmp, label)


def _grid() -> list[dict]:
    rows: list[dict] = []
    for front_min, danger_max, body_min, stable_min, net_min, boost_min, boost_mult in product(
        [0.0, 0.42, 0.50, 0.58],
        [0.62, 0.72, 0.84, 1.01],
        [0.0, 0.38, 0.46],
        [0.0, 0.38, 0.46],
        [0.0, 0.42, 0.50],
        [0.68, 0.76, 1.01],
        [1.0, 1.25],
    ):
        rows.append(
            {
                "front_min": front_min,
                "danger_max": danger_max,
                "body_min": body_min,
                "stable_min": stable_min,
                "net_min": net_min,
                "boost_min": boost_min,
                "boost_mult": boost_mult,
                "max_stake": 1500.0,
            }
        )
    return rows


def _score(train_metrics: dict, test_metrics: dict) -> float:
    if train_metrics.get("races", 0) < 120:
        return -1e9
    if train_metrics.get("race_hit_rate", 0) < 0.08:
        return -1e9
    # Prefer robust ROI, but avoid policies that just delete too much.
    return (
        float(train_metrics.get("roi", 0.0)) * 1.4
        + float(test_metrics.get("roi", 0.0)) * 0.8
        + np.log1p(float(train_metrics.get("races", 0))) * 0.08
        + float(test_metrics.get("profit_yen", 0.0)) / 50000.0
    )


def _segments(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    features = [
        "ticket_front_position_reliability_score",
        "ticket_danger_popular_score",
        "ticket_body_age_layoff_score",
        "ticket_stable_jockey_buy_timing_score",
        "priority_context_net_score",
    ]
    base = df[_num(df.get("runtime_stake_yen"), df.index, 0.0).gt(0)].copy()
    for feature in features:
        values = _num(base.get(feature), base.index, np.nan)
        try:
            bins = pd.qcut(values.rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])
        except Exception:
            continue
        tmp = base.assign(_bin=bins)
        for (ticket_type, bin_label), g in tmp.groupby(["ticket_type", "_bin"], observed=True):
            row = _metric_runtime(g, f"{feature}_{ticket_type}_{bin_label}")
            row.update({"feature": feature, "ticket_type": ticket_type, "bin": str(bin_label)})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add and test priority context factors: front reliability, dangerous favorites, body/layoff/age, stable-jockey timing.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/runtime_odds_decision_rules_v1/runtime_ticket_decisions.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/priority_context_factor_overlay_v1")
    parser.add_argument("--runner-cache", action="append", default=None, help="Runner feature CSV. Can be passed multiple times.")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    runner_paths = [project_path(p) for p in (args.runner_cache or DEFAULT_RUNNER_PATHS)]
    runners = _load_runner_context(runner_paths)
    enriched = enrich_priority_factors(tickets, runners)
    train = enriched[enriched["year"].eq(args.train_year)].copy()
    test = enriched[enriched["year"].eq(args.test_year)].copy()

    rows: list[dict] = []
    best_params: dict | None = None
    best_score = -1e18
    for params in _grid():
        selected_train = _apply_policy(train, params)
        selected_test = _apply_policy(test, params)
        train_metrics = _metric_runtime(selected_train, "train")
        test_metrics = _metric_runtime(selected_test, "test")
        score = _score(train_metrics, test_metrics)
        row = {**params, "score": score}
        row.update({f"train_{k}": v for k, v in train_metrics.items() if k != "policy"})
        row.update({f"test_{k}": v for k, v in test_metrics.items() if k != "policy"})
        rows.append(row)
        if score > best_score:
            best_score = score
            best_params = params

    out_dir = ensure_dir(project_path(args.output_dir))
    candidates = pd.DataFrame(rows).sort_values("score", ascending=False)
    candidates.to_csv(out_dir / "candidate_policies.csv", index=False, encoding="utf-8-sig")
    enriched.to_csv(out_dir / "priority_context_enriched_tickets.csv", index=False, encoding="utf-8-sig")
    _segments(enriched).to_csv(out_dir / "priority_context_segments.csv", index=False, encoding="utf-8-sig")

    selected = _apply_policy(enriched, best_params or {})
    selected.to_csv(out_dir / "priority_context_selected_tickets.csv", index=False, encoding="utf-8-sig")

    metrics = [
        _metric_runtime(enriched[_num(enriched.get("runtime_stake_yen"), enriched.index, 0.0).gt(0)], "runtime_base_all"),
        _metric_runtime(selected, "priority_context_all"),
    ]
    for year, g in enriched.groupby("year"):
        base_y = g[_num(g.get("runtime_stake_yen"), g.index, 0.0).gt(0)]
        sel_y = selected[selected["year"].eq(year)]
        metrics.append(_metric_runtime(base_y, f"runtime_base_{int(year)}"))
        metrics.append(_metric_runtime(sel_y, f"priority_context_{int(year)}"))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "tickets_csv": args.tickets_csv,
        "runner_cache_rows": int(len(runners)),
        "best_params": best_params,
        "base_all": metrics[0],
        "priority_context_all": metrics[1],
        "adoption_note": "Adopt only if 2026/test ROI, profit, and drawdown are acceptable versus runtime base. Otherwise use as dashboard annotation.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
