from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from validate_final_kernel_race_level import (
    add_kernel_features,
    load_universe,
    max_drawdown,
    metrics,
    num,
    norm01,
    tickets_from_pairs,
)


OUT = Path("outputs/analysis/final_kernel_ai_factor_rescue_v1")


def add_ai_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ai_market_late_score"] = (
        0.58 * norm01(out["market_overlay_score"], lo=0.35, hi=0.90)
        + 0.42 * norm01(out["late_value_survives_score"], lo=0.35, hi=0.90)
    ).clip(0.0, 1.0)
    out["ai_front_underdog_score"] = (
        0.42 * norm01(out["projected_front5_prob"], lo=0.45, hi=0.90)
        + 0.33 * norm01(out["partner_odds"], lo=6.0, hi=40.0)
        + 0.17 * norm01(out["market_overlay_score"], lo=0.35, hi=0.90)
        + 0.08 * (1.0 - norm01(out["partner_danger"], lo=0.0, hi=0.35))
    ).clip(0.0, 1.0)
    out["ai_joint_pair_score"] = (
        0.44 * norm01(out["pair_score"], lo=0.55, hi=0.85)
        + 0.34 * norm01(out["pair_quinella_score"], lo=0.48, hi=0.78)
        + 0.22 * norm01(out["partner_quinella_score"], lo=0.35, hi=0.78)
    ).clip(0.0, 1.0)
    out["ai_danger_safety_score"] = (
        0.55 * (1.0 - norm01(out["partner_danger"], lo=0.0, hi=0.35))
        + 0.45 * (1.0 - norm01(out["anchor_danger"], lo=0.0, hi=0.55))
    ).clip(0.0, 1.0)
    out["ai_rescue_score"] = (
        0.32 * out["ai_market_late_score"]
        + 0.28 * out["ai_front_underdog_score"]
        + 0.25 * out["ai_joint_pair_score"]
        + 0.15 * out["ai_danger_safety_score"]
    ).clip(0.0, 1.0)
    return out


def policy_grid() -> list[dict]:
    rows: list[dict] = []
    for kernel_cov, rescue_cov, partner_odds_min, umaren_quote_min, market_min, late_min, rescue_pd_max in product(
        [0.05, 0.08, 0.10],
        [0.01, 0.02, 0.03],
        [6.0, 10.0],
        [1000.0, 1200.0, 1800.0],
        [0.75, 0.80],
        [0.75],
        [0.20, 0.35],
    ):
        rows.append(
            {
                "kernel_coverage": kernel_cov,
                "rescue_coverage": rescue_cov,
                "venue_policy": "skip_hakodate",
                "going_policy": "skip_soft_heavy",
                "partner_odds_min": partner_odds_min,
                "partner_odds_max": 40.0,
                "umaren_quote_min": umaren_quote_min,
                "market_min": market_min,
                "late_min": late_min,
                "rescue_partner_danger_max": rescue_pd_max,
                "stake_profile": "finalish",
            }
        )
    return rows


def base_mask(df: pd.DataFrame, params: dict) -> pd.Series:
    return (
        df["venue"].ne("Hakodate")
        & df["going"].isin({"Good", "Yielding", "Unknown"})
        & df["wide_axis_score"].ge(0.62)
        & df["partner_odds"].between(params["partner_odds_min"], params["partner_odds_max"])
        & df["anchor_danger"].le(0.55)
    )


def kernel_candidates(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    mask = (
        base_mask(df, params)
        & df["wide_partner_score"].ge(0.66)
        & df["projected_front5_prob"].ge(0.60)
        & df["partner_danger"].le(0.35)
    )
    out = df[mask].copy()
    if out.empty:
        return out
    out["selection_branch"] = "kernel"
    out["selection_score"] = out["final_kernel_score"]
    return (
        out.sort_values(["race_id", "selection_score", "pair_score"], ascending=[True, False, False])
        .groupby("race_id", as_index=False)
        .head(1)
        .copy()
    )


def rescue_candidates(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    mask = (
        base_mask(df, params)
        & df["wide_partner_score"].ge(0.60)
        & df["projected_front5_prob"].ge(0.60)
        & df["partner_danger"].le(params["rescue_partner_danger_max"])
        & df["market_overlay_score"].ge(params["market_min"])
        & df["late_value_survives_score"].ge(params["late_min"])
        & df["ai_front_underdog_score"].ge(0.55)
        & df["ai_joint_pair_score"].ge(0.48)
    )
    out = df[mask].copy()
    if out.empty:
        return out
    out["selection_branch"] = "ai_factor_rescue"
    out["selection_score"] = out["ai_rescue_score"]
    return (
        out.sort_values(["race_id", "selection_score", "market_overlay_score"], ascending=[True, False, False])
        .groupby("race_id", as_index=False)
        .head(1)
        .copy()
    )


def thresholds(train: pd.DataFrame, params: dict) -> tuple[float, float]:
    k = kernel_candidates(train, params)
    r = rescue_candidates(train, params)
    kt = float(k["selection_score"].quantile(1.0 - params["kernel_coverage"])) if not k.empty else float("inf")
    rt = float(r["selection_score"].quantile(1.0 - params["rescue_coverage"])) if not r.empty else float("inf")
    return kt, rt


def select_pairs(df: pd.DataFrame, params: dict, kernel_threshold: float, rescue_threshold: float) -> pd.DataFrame:
    k = kernel_candidates(df, params)
    r = rescue_candidates(df, params)
    if not k.empty:
        k = k[k["selection_score"].ge(kernel_threshold)].copy()
    if not r.empty:
        r = r[r["selection_score"].ge(rescue_threshold)].copy()
    both = pd.concat([k, r], ignore_index=True, sort=False)
    if both.empty:
        return both
    return (
        both.sort_values(["race_id", "selection_score", "market_overlay_score"], ascending=[True, False, False])
        .groupby("race_id", as_index=False)
        .head(1)
        .copy()
    )


def tickets_from_ai_pairs(pairs: pd.DataFrame, params: dict) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    # Use the existing final-ish staking/ticket conversion, but relax umaren quote to the policy value.
    converted_params = {
        "stake_profile": "finalish",
        "umaren_pair_score_min": 0.72,
        "partner_quinella_min": 0.54,
        "umaren_partner_odds_max": 25.0,
        "umaren_quote_min": params["umaren_quote_min"],
    }
    return tickets_from_pairs(pairs, converted_params)


def evaluate(df: pd.DataFrame, params: dict, kt: float, rt: float, label: str) -> tuple[dict, pd.DataFrame]:
    pairs = select_pairs(df, params, kt, rt)
    tickets = tickets_from_ai_pairs(pairs, params)
    m = metrics(tickets, label)
    total_races = int(df["race_id"].nunique())
    m["candidate_races"] = total_races
    m["race_selection_rate"] = float(m["races"] / total_races) if total_races else 0.0
    if not tickets.empty and "selection_branch" in tickets.columns:
        m["kernel_tickets"] = int(tickets["selection_branch"].eq("kernel").sum())
        m["rescue_tickets"] = int(tickets["selection_branch"].eq("ai_factor_rescue").sum())
    else:
        m["kernel_tickets"] = 0
        m["rescue_tickets"] = 0
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
        test = df[df["year"].eq(test_year)].copy()
        rows: list[dict] = []
        for i, params in enumerate(grids):
            kt, rt = thresholds(train, params)
            m, _ = evaluate(train, params, kt, rt, f"train_{test_year}_{i}")
            row = m | params
            row["grid_id"] = i
            row["kernel_threshold"] = kt
            row["rescue_threshold"] = rt
            row["test_year"] = test_year
            row["selection_score"] = policy_score(m)
            rows.append(row)
        grid = pd.DataFrame(rows).sort_values(["selection_score", "roi"], ascending=[False, False])
        train_rows.append(grid.head(100))
        best = grid.iloc[0]
        params = grids[int(best["grid_id"])]
        m, tickets = evaluate(test, params, float(best["kernel_threshold"]), float(best["rescue_threshold"]), f"wf_test_{test_year}")
        m.update(params)
        m["test_year"] = test_year
        m["train_roi"] = float(best["roi"])
        m["train_races"] = int(best["races"])
        m["train_race_selection_rate"] = float(best["race_selection_rate"])
        m["train_profit_yen"] = float(best["profit_yen"])
        m["kernel_threshold"] = float(best["kernel_threshold"])
        m["rescue_threshold"] = float(best["rescue_threshold"])
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


def max_drawdown_from_tickets(tickets: pd.DataFrame) -> float:
    if tickets.empty:
        return 0.0
    race = tickets.groupby("race_id", sort=False).agg(stake=("stake_yen", "sum"), ret=("return_yen", "sum"))
    return max_drawdown(race["ret"] - race["stake"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_ai_factor_scores(add_kernel_features(load_universe()))
    train_grid, summary, tickets = walkforward(df)
    train_grid.to_csv(OUT / "walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    tickets.to_csv(OUT / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
    print("AI FACTOR RESCUE WALKFORWARD")
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
        "kernel_tickets",
        "rescue_tickets",
        "kernel_coverage",
        "rescue_coverage",
        "partner_odds_min",
        "umaren_quote_min",
        "market_min",
        "rescue_partner_danger_max",
        "train_roi",
        "train_races",
    ]
    print(summary[cols].to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
