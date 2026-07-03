from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_priority_context_factor_overlay import DEFAULT_RUNNER_PATHS, _load_runner_context, enrich_priority_factors
from evaluate_priority_a_non_day_factors import _load_runners, enrich_a_factors
from evaluate_priority_b_context_factors import enrich_b_factors
from strict_pair_probability_roi_protocol import VENUE_CODE, load_universe, max_drawdown, num, norm01
from validate_final_kernel_race_level import add_kernel_features
from src.utils.paths import project_path


OUT = Path("outputs/analysis/strongest_final_strength_model_v1")


def clip01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def make_ticket_like_pairs(base: pd.DataFrame, ticket_type: str) -> pd.DataFrame:
    out = base.copy()
    out["ticket_type"] = ticket_type
    out["runtime_stake_yen"] = 100.0
    if ticket_type == "wide":
        hit_proxy = (
            0.38 * clip01(out["pair_score"])
            + 0.26 * clip01(out["wide_partner_score"])
            + 0.18 * clip01(out["projected_front5_prob"])
            + 0.18 * clip01(out["market_overlay_score"])
        ).clip(0.0, 1.0)
        out["quote_pay_proxy_per100"] = out["wide_quote_proxy"]
        out["runtime_backtest_pay_per100"] = out["wide_pay"]
        out["hit"] = out["wide_hit"].astype(bool)
    else:
        hit_proxy = (
            0.38 * clip01(out["pair_quinella_score"])
            + 0.22 * clip01(out["anchor_quinella_score"])
            + 0.22 * clip01(out["partner_quinella_score"])
            + 0.18 * clip01(out["market_overlay_score"])
        ).clip(0.0, 1.0)
        out["quote_pay_proxy_per100"] = out["umaren_quote_proxy"]
        out["runtime_backtest_pay_per100"] = out["umaren_pay"]
        out["hit"] = out["umaren_hit"].astype(bool)

    out["ticket_hit_prob"] = hit_proxy
    out["runtime_expected_roi"] = hit_proxy * num(out["quote_pay_proxy_per100"]).fillna(0.0) / 100.0
    out["expected_roi_after_slippage"] = out["runtime_expected_roi"]
    out["runtime_odds_margin_ratio"] = (out["runtime_expected_roi"] / 1.15).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["min_odds_margin_ratio"] = out["runtime_odds_margin_ratio"]
    out["stake_quality_score"] = (
        0.42 * clip01(out["market_overlay_score"])
        + 0.30 * clip01(out["late_value_survives_score"])
        + 0.28 * hit_proxy
    ).clip(0.0, 1.0)
    out["ticket_sizing_score"] = (
        0.52 * out["stake_quality_score"]
        + 0.28 * clip01(out["pair_score"])
        + 0.20 * clip01(out["projected_front5_prob"])
    ).clip(0.0, 1.0)
    out["race_difficulty_score"] = num(out.get("race_difficulty_score"), out.index, 0.50).fillna(
        num(out.get("anchor_race_difficulty_model"), out.index, 0.50)
    )
    out["race_chaos_score"] = num(out.get("race_chaos_score"), out.index, out["race_difficulty_score"]).fillna(out["race_difficulty_score"])
    out["race_solidness_score"] = num(out.get("race_solidness_score"), out.index, 1.0 - out["race_chaos_score"]).fillna(1.0 - out["race_chaos_score"])
    return out


def enrich_strength_features(ticket_like: pd.DataFrame) -> pd.DataFrame:
    context_runners = _load_runner_context([project_path(p) for p in DEFAULT_RUNNER_PATHS])
    a_runners = _load_runners([project_path(p) for p in DEFAULT_RUNNER_PATHS])

    enriched = enrich_priority_factors(ticket_like, context_runners)
    enriched = enrich_a_factors(enriched, a_runners)
    enriched = enrich_b_factors(enriched)
    return add_strength_scores(enriched)


def add_strength_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["strength_market_late_score"] = (
        0.55 * norm01(out["market_overlay_score"], lo=0.30, hi=0.92)
        + 0.45 * norm01(out["late_value_survives_score"], lo=0.30, hi=0.92)
    ).clip(0.0, 1.0)
    out["strength_front_underdog_score"] = (
        0.42 * norm01(out["projected_front5_prob"], lo=0.42, hi=0.92)
        + 0.28 * norm01(out["partner_odds"], lo=5.0, hi=40.0)
        + 0.18 * norm01(out["market_overlay_score"], lo=0.30, hi=0.92)
        + 0.12 * (1.0 - norm01(out["partner_danger"], lo=0.0, hi=0.35))
    ).clip(0.0, 1.0)
    out["strength_joint_score"] = (
        0.40 * norm01(out["pair_score"], lo=0.52, hi=0.86)
        + 0.36 * norm01(out["pair_quinella_score"], lo=0.46, hi=0.80)
        + 0.24 * norm01(out["partner_quinella_score"], lo=0.30, hi=0.80)
    ).clip(0.0, 1.0)
    out["strength_safety_score"] = (
        0.34 * (1.0 - norm01(out["ticket_danger_popular_score"], lo=0.0, hi=0.70))
        + 0.26 * (1.0 - norm01(out["anchor_danger"], lo=0.0, hi=0.55))
        + 0.22 * (1.0 - norm01(out["partner_danger"], lo=0.0, hi=0.35))
        + 0.18 * (1.0 - norm01(out["race_difficulty_score"], lo=0.25, hi=0.80))
    ).clip(0.0, 1.0)
    out["strength_context_score"] = (
        0.32 * norm01(out["priority_context_net_score"], lo=0.28, hi=0.50)
        + 0.28 * norm01(out["a_priority_net_score"], lo=0.20, hi=0.55)
        + 0.30 * norm01(out["b_priority_net_score"], lo=0.42, hi=0.72)
        + 0.10 * norm01(out["ticket_front_position_reliability_score"], lo=0.35, hi=0.70)
    ).clip(0.0, 1.0)

    is_umaren = out["ticket_type"].astype(str).eq("umaren")
    wide_score = (
        0.24 * out["strength_market_late_score"]
        + 0.22 * out["strength_front_underdog_score"]
        + 0.19 * out["strength_joint_score"]
        + 0.18 * out["strength_safety_score"]
        + 0.17 * out["strength_context_score"]
    )
    umaren_score = (
        0.26 * out["strength_joint_score"]
        + 0.23 * out["strength_market_late_score"]
        + 0.20 * out["strength_context_score"]
        + 0.17 * out["strength_safety_score"]
        + 0.14 * out["strength_front_underdog_score"]
    )
    out["strongest_ticket_score"] = np.where(is_umaren, umaren_score, wide_score).clip(0.0, 1.0)
    return out


def build_pair_strength_universe() -> pd.DataFrame:
    base = add_kernel_features(load_universe())
    wide = enrich_strength_features(make_ticket_like_pairs(base, "wide"))
    umaren = enrich_strength_features(make_ticket_like_pairs(base, "umaren"))

    keep_cols = [
        "race_id",
        "anchor_no",
        "partner_no",
        "strongest_ticket_score",
        "strength_market_late_score",
        "strength_front_underdog_score",
        "strength_joint_score",
        "strength_safety_score",
        "strength_context_score",
        "ticket_danger_popular_score",
        "race_difficulty_score",
        "a_priority_net_score",
        "b_priority_net_score",
        "priority_context_net_score",
        "ticket_front_position_reliability_score",
        "runtime_expected_roi",
        "ticket_hit_prob",
    ]
    wide_small = wide[keep_cols].rename(columns={c: f"wide_{c}" for c in keep_cols if c not in {"race_id", "anchor_no", "partner_no"}})
    umaren_small = umaren[keep_cols].rename(columns={c: f"umaren_{c}" for c in keep_cols if c not in {"race_id", "anchor_no", "partner_no"}})

    pair = base.merge(wide_small, on=["race_id", "anchor_no", "partner_no"], how="left")
    pair = pair.merge(umaren_small, on=["race_id", "anchor_no", "partner_no"], how="left")
    pair["strongest_pair_score"] = np.maximum(
        num(pair.get("wide_strongest_ticket_score"), pair.index, 0.0).fillna(0.0),
        num(pair.get("umaren_strongest_ticket_score"), pair.index, 0.0).fillna(0.0),
    )
    pair["strongest_pair_score"] = (
        0.76 * pair["strongest_pair_score"]
        + 0.14 * norm01(pair["final_kernel_score"], lo=0.40, hi=0.90)
        + 0.10 * norm01(pair["market_overlay_score"], lo=0.30, hi=0.92)
    ).clip(0.0, 1.0)
    return pair


def policy_grid() -> list[dict]:
    allowed_all = set(VENUE_CODE.values()) | {"Unknown"}
    rows: list[dict] = []
    for coverage, venue_policy, going_policy, min_market, min_late, min_safety, min_context, umaren_min, umaren_quote_min in product(
        [0.03, 0.05, 0.08, 0.10],
        ["all", "skip_hakodate"],
        ["all", "skip_soft_heavy"],
        [0.60, 0.72],
        [0.60, 0.72],
        [0.42, 0.52],
        [0.42, 0.50],
        [0.58, 0.64],
        [1000.0, 1400.0],
    ):
        rows.append(
            {
                "coverage": coverage,
                "venue_policy": venue_policy,
                "venue_allowed": allowed_all if venue_policy == "all" else allowed_all - {"Hakodate"},
                "going_policy": going_policy,
                "going_allowed": {"Good", "Yielding", "Soft", "Heavy", "Unknown"}
                if going_policy == "all"
                else {"Good", "Yielding", "Unknown"},
                "min_market": min_market,
                "min_late": min_late,
                "min_safety": min_safety,
                "min_context": min_context,
                "umaren_ticket_min": umaren_min,
                "umaren_quote_min": umaren_quote_min,
            }
        )
    return rows


def prefilter(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    mask = (
        df["venue"].isin(params["venue_allowed"])
        & df["going"].isin(params["going_allowed"])
        & num(df["market_overlay_score"]).ge(params["min_market"])
        & num(df["late_value_survives_score"]).ge(params["min_late"])
        & num(df["wide_strength_safety_score"]).ge(params["min_safety"])
        & num(df["wide_strength_context_score"]).ge(params["min_context"])
        & num(df["partner_odds"]).between(5.0, 45.0)
        & num(df["anchor_danger"]).le(0.55)
        & num(df["partner_danger"]).le(0.35)
    )
    return df[mask].copy()


def race_representatives(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    work = prefilter(df, params)
    if work.empty:
        return work
    return (
        work.sort_values(
            ["race_id", "strongest_pair_score", "market_overlay_score", "pair_score"],
            ascending=[True, False, False, False],
        )
        .groupby("race_id", as_index=False)
        .head(1)
        .copy()
    )


def threshold_from_coverage(train: pd.DataFrame, params: dict) -> float:
    reps = race_representatives(train, params)
    if reps.empty:
        return float("inf")
    return float(reps["strongest_pair_score"].quantile(1.0 - params["coverage"]))


def select_pairs(df: pd.DataFrame, params: dict, threshold: float) -> pd.DataFrame:
    reps = race_representatives(df, params)
    if reps.empty:
        return reps
    return reps[reps["strongest_pair_score"].ge(threshold)].copy()


def stake_from_score(score: pd.Series, lo: float, hi: float) -> pd.Series:
    raw = lo + (hi - lo) * norm01(score, lo=0.48, hi=0.86)
    return (np.floor(raw / 100.0) * 100.0).clip(lower=100.0)


def tickets_from_pairs(pairs: pd.DataFrame, params: dict) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    base = pairs.copy()
    base["pair_key"] = base["race_id"] + ":" + base["anchor_no"].astype(str) + "-" + base["partner_no"].astype(str)
    frames: list[pd.DataFrame] = []

    wide = base.copy()
    wide["ticket_type"] = "wide"
    wide["stake_yen"] = stake_from_score(wide["wide_strongest_ticket_score"], 400.0, 1200.0)
    wide["hit"] = wide["wide_hit"].astype(bool)
    wide["return_yen"] = np.where(wide["hit"], wide["wide_pay"] * wide["stake_yen"] / 100.0, 0.0)
    frames.append(wide)

    umaren_mask = (
        num(base["umaren_strongest_ticket_score"]).ge(params["umaren_ticket_min"])
        & num(base["umaren_quote_proxy"]).ge(params["umaren_quote_min"])
        & num(base["partner_odds"]).le(25.0)
        & num(base["pair_score"]).ge(0.70)
    )
    umaren = base[umaren_mask].copy()
    if not umaren.empty:
        umaren["ticket_type"] = "umaren"
        umaren["stake_yen"] = stake_from_score(umaren["umaren_strongest_ticket_score"], 600.0, 1800.0)
        umaren["hit"] = umaren["umaren_hit"].astype(bool)
        umaren["return_yen"] = np.where(umaren["hit"], umaren["umaren_pay"] * umaren["stake_yen"] / 100.0, 0.0)
        frames.append(umaren)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["ticket_key"] = out["ticket_type"] + ":" + out["pair_key"]
    return out


def metrics(tickets: pd.DataFrame, label: str) -> dict:
    if tickets.empty:
        return {"label": label, "tickets": 0, "races": 0, "stake_yen": 0.0, "return_yen": 0.0, "profit_yen": 0.0, "roi": 0.0}
    stake = float(tickets["stake_yen"].sum())
    ret = float(tickets["return_yen"].sum())
    race = tickets.groupby("race_id", sort=False).agg(stake=("stake_yen", "sum"), ret=("return_yen", "sum"), hit=("hit", "max"))
    profit = race["ret"] - race["stake"]
    return {
        "label": label,
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(tickets["hit"].mean()),
        "race_hit_rate": float(race["hit"].mean()),
        "max_drawdown_yen": max_drawdown(profit),
        "wide_tickets": int((tickets["ticket_type"] == "wide").sum()),
        "umaren_tickets": int((tickets["ticket_type"] == "umaren").sum()),
    }


def evaluate(df: pd.DataFrame, params: dict, threshold: float, label: str) -> tuple[dict, pd.DataFrame]:
    pairs = select_pairs(df, params, threshold)
    tickets = tickets_from_pairs(pairs, params)
    m = metrics(tickets, label)
    total = int(df["race_id"].nunique())
    m["candidate_races"] = total
    m["race_selection_rate"] = float(m["races"] / total) if total else 0.0
    return m, tickets


def policy_score(m: dict) -> float:
    if m.get("races", 0) < 40:
        return -1e9
    if m.get("race_selection_rate", 1.0) > 0.12:
        return -1e9
    if m.get("race_hit_rate", 0.0) < 0.06:
        return -1e9
    return (
        float(m["roi"]) * np.sqrt(max(float(m["race_hit_rate"]), 0.001)) * np.log1p(float(m["races"]))
        + float(m["profit_yen"]) / 100000.0
    )


def walkforward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grids = policy_grid()
    train_rows: list[pd.DataFrame] = []
    wf_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    for test_year in [2025, 2026]:
        train = df[df["year"] < test_year].copy()
        test = df[df["year"] == test_year].copy()
        rows: list[dict] = []
        for i, params in enumerate(grids):
            threshold = threshold_from_coverage(train, params)
            m, _ = evaluate(train, params, threshold, f"train_{test_year}_{i}")
            row = m | {k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}}
            row["grid_id"] = i
            row["score_threshold"] = threshold
            row["test_year"] = test_year
            row["selection_score"] = policy_score(m)
            rows.append(row)
        grid = pd.DataFrame(rows).sort_values(["selection_score", "roi"], ascending=[False, False])
        train_rows.append(grid.head(100))
        best = grid.iloc[0]
        params = grids[int(best["grid_id"])]
        m, tickets = evaluate(test, params, float(best["score_threshold"]), f"wf_test_{test_year}")
        m.update({k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}})
        m["test_year"] = test_year
        m["train_roi"] = float(best["roi"])
        m["train_races"] = int(best["races"])
        m["train_profit_yen"] = float(best["profit_yen"])
        m["train_race_selection_rate"] = float(best["race_selection_rate"])
        m["score_threshold"] = float(best["score_threshold"])
        wf_rows.append(m)
        if not tickets.empty:
            tmp = tickets.copy()
            tmp["test_year"] = test_year
            tmp["selected_grid_id"] = int(best["grid_id"])
            ticket_frames.append(tmp)
    return (
        pd.concat(train_rows, ignore_index=True, sort=False),
        pd.DataFrame(wf_rows),
        pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame(),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    universe = build_pair_strength_universe()
    universe.to_csv(OUT / "pair_strength_universe.csv", index=False, encoding="utf-8-sig")
    train_grid, summary, tickets = walkforward(universe)
    train_grid.to_csv(OUT / "walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    tickets.to_csv(OUT / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")

    cols = [
        "label",
        "test_year",
        "candidate_races",
        "races",
        "race_selection_rate",
        "tickets",
        "stake_yen",
        "return_yen",
        "profit_yen",
        "roi",
        "race_hit_rate",
        "max_drawdown_yen",
        "wide_tickets",
        "umaren_tickets",
        "coverage",
        "venue_policy",
        "going_policy",
        "min_market",
        "min_late",
        "min_safety",
        "min_context",
        "umaren_ticket_min",
        "umaren_quote_min",
        "train_roi",
        "train_races",
    ]
    print("STRONGEST FINAL STRENGTH WALKFORWARD")
    print(summary[cols].to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
