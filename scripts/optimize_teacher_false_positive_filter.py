from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from strict_pair_probability_roi_protocol import max_drawdown, num


SCORED = Path("outputs/analysis/strongest_teacher_distillation_v1/teacher_scored_ticket_candidates.csv")
OUT = Path("outputs/analysis/teacher_false_positive_filter_v1")


def norm01(s: pd.Series, lo: float, hi: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(lo)
    if hi <= lo:
        return pd.Series(0.5, index=x.index)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0)


def base_select(df: pd.DataFrame, coverage: float, max_tickets_per_race: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    race_score = df.groupby("race_id")["teacher_edge_score"].max()
    threshold = float(race_score.quantile(1.0 - coverage))
    selected = df[df["teacher_edge_score"].ge(threshold)].copy()
    return (
        selected.sort_values(["race_id", "teacher_edge_score", "runtime_expected_roi"], ascending=[True, False, False])
        .groupby("race_id", as_index=False)
        .head(max_tickets_per_race)
        .copy()
    )


def add_false_positive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["overheated_value_score"] = (
        0.46 * norm01(out["runtime_expected_roi"], 3.0, 25.0)
        + 0.34 * norm01(out["quote_proxy"], 600.0, 4000.0)
        + 0.20 * norm01(out["partner_odds"], 20.0, 90.0)
    ).clip(0.0, 1.0)
    out["quality_support_score"] = (
        0.28 * num(out["strength_joint_score"]).fillna(0.0)
        + 0.24 * num(out["strength_context_score"]).fillna(0.0)
        + 0.20 * num(out["teacher_similarity_score"]).fillna(0.0)
        + 0.16 * num(out["ticket_front_position_reliability_score"]).fillna(0.0)
        + 0.12 * num(out["strength_safety_score"]).fillna(0.0)
    ).clip(0.0, 1.0)
    out["fragile_overlay_score"] = (
        out["overheated_value_score"] * (1.0 - out["quality_support_score"]).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    return out


def apply_filter(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    common = (
        out["teacher_similarity_score"].ge(params["similarity_min"])
        & out["quality_support_score"].ge(params["quality_min"])
        & out["fragile_overlay_score"].le(params["fragile_max"])
    )

    wide = out["ticket_type"].eq("wide")
    wide_keep = (
        wide
        & out["partner_odds"].le(params["wide_partner_odds_max"])
        & out["runtime_expected_roi"].between(params["wide_roi_min"], params["wide_roi_max"])
        & out["quote_proxy"].between(params["wide_quote_min"], params["wide_quote_max"])
        & out["strength_joint_score"].ge(params["wide_joint_min"])
    )

    umaren = out["ticket_type"].eq("umaren")
    umaren_keep = (
        umaren
        & out["partner_odds"].between(params["umaren_partner_odds_min"], params["umaren_partner_odds_max"])
        & out["runtime_expected_roi"].between(params["umaren_roi_min"], params["umaren_roi_max"])
        & out["quote_proxy"].between(params["umaren_quote_min"], params["umaren_quote_max"])
        & out["strength_joint_score"].ge(params["umaren_joint_min"])
    )
    return out[common & (wide_keep | umaren_keep)].copy()


def stake_selected(tickets: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    base = np.where(out["ticket_type"].eq("wide"), 400.0, 600.0)
    high = np.where(out["ticket_type"].eq("wide"), 1100.0, 1600.0)
    raw = base + (high - base) * norm01(out["quality_support_score"], 0.42, 0.70)
    out["stake_yen"] = (np.floor(raw / 100.0) * 100.0).clip(lower=100.0)
    out["return_yen"] = np.where(out["hit"].astype(bool), out["pay_per100"] * out["stake_yen"] / 100.0, 0.0)
    return out


def metrics(tickets: pd.DataFrame, label: str) -> dict:
    if tickets.empty:
        return {
            "label": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
            "wide_tickets": 0,
            "umaren_tickets": 0,
        }
    stake = float(tickets["stake_yen"].sum())
    ret = float(tickets["return_yen"].sum())
    race = tickets.groupby("race_id", sort=False).agg(stake=("stake_yen", "sum"), ret=("return_yen", "sum"), hit=("hit", "max"))
    pnl = race["ret"] - race["stake"]
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
        "max_drawdown_yen": max_drawdown(pnl),
        "wide_tickets": int(tickets["ticket_type"].eq("wide").sum()),
        "umaren_tickets": int(tickets["ticket_type"].eq("umaren").sum()),
    }


def policy_grid() -> list[dict]:
    rows: list[dict] = []
    for (
        coverage,
        max_tickets_per_race,
        similarity_min,
        quality_min,
        fragile_max,
        wide_partner_odds_max,
        wide_roi_max,
        wide_quote_max,
        wide_joint_min,
        umaren_partner_odds_max,
        umaren_roi_max,
        umaren_quote_max,
        umaren_joint_min,
    ) in product(
        [0.05, 0.08],
        [1],
        [0.46, 0.52],
        [0.42, 0.48],
        [0.42],
        [45.0, 60.0],
        [6.0, 9.0],
        [650.0, 900.0],
        [0.50, 0.58],
        [18.0, 25.0],
        [12.0, 18.0],
        [1600.0, 2400.0],
        [0.52, 0.60],
    ):
        rows.append(
            {
                "coverage": coverage,
                "max_tickets_per_race": max_tickets_per_race,
                "similarity_min": similarity_min,
                "quality_min": quality_min,
                "fragile_max": fragile_max,
                "wide_partner_odds_max": wide_partner_odds_max,
                "wide_roi_min": 0.8,
                "wide_roi_max": wide_roi_max,
                "wide_quote_min": 250.0,
                "wide_quote_max": wide_quote_max,
                "wide_joint_min": wide_joint_min,
                "umaren_partner_odds_min": 5.0,
                "umaren_partner_odds_max": umaren_partner_odds_max,
                "umaren_roi_min": 1.0,
                "umaren_roi_max": umaren_roi_max,
                "umaren_quote_min": 800.0,
                "umaren_quote_max": umaren_quote_max,
                "umaren_joint_min": umaren_joint_min,
            }
        )
    return rows


def select_policy(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    selected_tests: list[pd.DataFrame] = []
    for i, params in enumerate(policy_grid()):
        base_train = base_select(train, params["coverage"], params["max_tickets_per_race"])
        base_test = base_select(test, params["coverage"], params["max_tickets_per_race"])
        filt_train = stake_selected(apply_filter(base_train, params))
        filt_test = stake_selected(apply_filter(base_test, params))
        m_train = metrics(filt_train, f"train_{i}")
        m_test = metrics(filt_test, f"test_{i}")
        if m_train["races"] < 25 or m_train["race_hit_rate"] < 0.06:
            score = -1e9
        else:
            score = (
                m_train["roi"] * np.sqrt(max(m_train["race_hit_rate"], 0.001)) * np.log1p(m_train["races"])
                + m_train["profit_yen"] / 80000.0
                - abs(m_train["max_drawdown_yen"]) / 120000.0
            )
        row = params | {"grid_id": i, "selection_score": score}
        row.update({f"train_{k}": v for k, v in m_train.items()})
        row.update({f"test_{k}": v for k, v in m_test.items()})
        rows.append(row)
        if not filt_test.empty:
            tmp = filt_test.copy()
            tmp["grid_id"] = i
            selected_tests.append(tmp)
    grid = pd.DataFrame(rows).sort_values(["selection_score", "train_roi"], ascending=[False, False])
    best = grid.iloc[0]
    best_params = {k: best[k] for k in policy_grid()[0].keys()}
    best_train = stake_selected(apply_filter(base_select(train, best_params["coverage"], int(best_params["max_tickets_per_race"])), best_params))
    best_test = stake_selected(apply_filter(base_select(test, best_params["coverage"], int(best_params["max_tickets_per_race"])), best_params))
    return grid, best_train, best_test


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(SCORED, dtype={"race_id": str}, low_memory=False)
    scored = add_false_positive_features(scored)
    train = scored[scored["year"].eq(2025)].copy()
    test = scored[scored["year"].eq(2026)].copy()
    grid, best_train, best_test = select_policy(train, test)

    grid.to_csv(OUT / "false_positive_filter_grid.csv", index=False, encoding="utf-8-sig")
    best_train.to_csv(OUT / "best_filtered_train_2025_tickets.csv", index=False, encoding="utf-8-sig")
    best_test.to_csv(OUT / "best_filtered_test_2026_tickets.csv", index=False, encoding="utf-8-sig")

    target_races = {"2026012410010112", "2026013105010111"}
    target_check = add_false_positive_features(scored[scored["race_id"].isin(target_races)].copy())
    target_check.to_csv(OUT / "target_2026_filter_check.csv", index=False, encoding="utf-8-sig")

    print("FALSE POSITIVE FILTER")
    cols = [
        "grid_id",
        "coverage",
        "max_tickets_per_race",
        "similarity_min",
        "quality_min",
        "fragile_max",
        "wide_partner_odds_max",
        "wide_roi_max",
        "wide_quote_max",
        "wide_joint_min",
        "umaren_partner_odds_max",
        "umaren_roi_max",
        "umaren_quote_max",
        "umaren_joint_min",
        "train_races",
        "train_roi",
        "train_profit_yen",
        "train_race_hit_rate",
        "test_races",
        "test_roi",
        "test_profit_yen",
        "test_race_hit_rate",
        "selection_score",
    ]
    print(grid[cols].head(30).to_string(index=False))
    print("\nBEST TRAIN")
    print(metrics(best_train, "best_train"))
    print("\nBEST TEST")
    print(metrics(best_test, "best_test"))
    print("\nBEST TEST TICKETS")
    show_cols = [
        "race_id",
        "ticket_type",
        "anchor_name",
        "partner_name",
        "teacher_edge_score",
        "quality_support_score",
        "fragile_overlay_score",
        "runtime_expected_roi",
        "partner_odds",
        "quote_proxy",
        "strength_joint_score",
        "hit",
        "pay_per100",
        "stake_yen",
        "return_yen",
    ]
    print(best_test[show_cols].to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
