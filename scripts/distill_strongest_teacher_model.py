from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strict_pair_probability_roi_protocol import max_drawdown, num, norm01


UNIVERSE = Path("outputs/analysis/strongest_final_strength_model_v1/pair_strength_universe.csv")
FINAL = Path("outputs/analysis/final_operational_quality_v1/standard_explained_tickets.csv")
OUT = Path("outputs/analysis/strongest_teacher_distillation_v1")


FEATURES = [
    "ticket_score",
    "pair_score",
    "pair_quinella_score",
    "partner_quinella_score",
    "final_kernel_score",
    "market_overlay_score",
    "late_value_survives_score",
    "projected_front5_prob",
    "partner_odds",
    "quote_proxy",
    "runtime_expected_roi",
    "ticket_hit_prob",
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
]


def load_final_teacher() -> set[str]:
    final = pd.read_csv(FINAL, dtype={"race_id": str}, low_memory=False)
    final = final[num(final.get("runtime_stake_yen"), final.index, 0.0).fillna(0.0).gt(0)].copy()
    final = final[final["ticket_type"].isin(["wide", "umaren"])].copy()
    final["anchor_no"] = num(final["anchor_no"]).astype("Int64")
    final["partner_no"] = num(final["partner_no"]).astype("Int64")
    final["teacher_key"] = (
        final["ticket_type"].astype(str)
        + ":"
        + final["race_id"].astype(str)
        + ":"
        + final["anchor_no"].astype(str)
        + "-"
        + final["partner_no"].astype(str)
    )
    return set(final.loc[final["race_id"].str[:4].eq("2025"), "teacher_key"])


def ticket_candidates() -> pd.DataFrame:
    u = pd.read_csv(UNIVERSE, dtype={"race_id": str}, low_memory=False)
    frames: list[pd.DataFrame] = []
    for ticket_type, prefix, quote_col, hit_col, pay_col in [
        ("wide", "wide", "wide_quote_proxy", "wide_hit", "wide_pay"),
        ("umaren", "umaren", "umaren_quote_proxy", "umaren_hit", "umaren_pay"),
    ]:
        t = u.copy()
        t["ticket_type"] = ticket_type
        t["ticket_score"] = num(t.get(f"{prefix}_strongest_ticket_score"), t.index, 0.0).fillna(0.0)
        t["quote_proxy"] = num(t.get(quote_col), t.index, 0.0).fillna(0.0)
        t["runtime_expected_roi"] = num(t.get(f"{prefix}_runtime_expected_roi"), t.index, np.nan).fillna(
            t["ticket_score"] * t["quote_proxy"] / 100.0
        )
        t["ticket_hit_prob"] = num(t.get(f"{prefix}_ticket_hit_prob"), t.index, np.nan).fillna(t["ticket_score"])
        for col in [
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
        ]:
            t[col] = num(t.get(f"{prefix}_{col}"), t.index, np.nan).fillna(num(t.get(col), t.index, 0.0))
        t["hit"] = t[hit_col].astype(bool)
        t["pay_per100"] = num(t[pay_col]).fillna(0.0)
        t["year"] = t["race_id"].str[:4].astype(int)
        t["anchor_no"] = num(t["anchor_no"]).astype("Int64")
        t["partner_no"] = num(t["partner_no"]).astype("Int64")
        t["teacher_key"] = (
            t["ticket_type"].astype(str)
            + ":"
            + t["race_id"].astype(str)
            + ":"
            + t["anchor_no"].astype(str)
            + "-"
            + t["partner_no"].astype(str)
        )
        frames.append(t)
    out = pd.concat(frames, ignore_index=True, sort=False)
    teacher_keys = load_final_teacher()
    out["teacher_selected_2025"] = out["teacher_key"].isin(teacher_keys)
    return out


def fit_profile(train: pd.DataFrame, ticket_type: str) -> dict:
    pos = train[train["ticket_type"].eq(ticket_type) & train["teacher_selected_2025"]].copy()
    if pos.empty:
        pos = train[train["ticket_type"].eq(ticket_type)].nlargest(100, "ticket_score").copy()
    profile: dict[str, tuple[float, float]] = {}
    for feature in FEATURES:
        x = num(pos.get(feature), pos.index, np.nan).replace([np.inf, -np.inf], np.nan).dropna()
        med = float(x.median()) if len(x) else 0.0
        iqr = float(x.quantile(0.75) - x.quantile(0.25)) if len(x) else 0.15
        profile[feature] = (med, max(iqr, 0.08))
    return profile


def score_with_profile(df: pd.DataFrame, profiles: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()
    scores = pd.Series(0.0, index=out.index)
    weights = {
        "ticket_score": 1.25,
        "market_overlay_score": 1.15,
        "late_value_survives_score": 1.15,
        "runtime_expected_roi": 1.10,
        "strength_market_late_score": 1.05,
        "strength_front_underdog_score": 1.00,
        "strength_joint_score": 1.05,
        "strength_safety_score": 0.95,
        "strength_context_score": 1.00,
        "b_priority_net_score": 1.00,
        "a_priority_net_score": 0.85,
    }
    total_weight = sum(weights.get(f, 0.65) for f in FEATURES)
    for ticket_type, profile in profiles.items():
        mask = out["ticket_type"].eq(ticket_type)
        if not mask.any():
            continue
        subtotal = pd.Series(0.0, index=out.index[mask])
        for feature in FEATURES:
            med, scale = profile[feature]
            x = num(out.loc[mask, feature]).fillna(med)
            similarity = np.exp(-np.abs(x - med) / scale)
            directional = norm01(x, lo=max(0.0, med - scale), hi=min(max(float(x.max()), med + scale), med + 2 * scale))
            subtotal += weights.get(feature, 0.65) * (0.72 * similarity + 0.28 * directional)
        scores.loc[mask] = subtotal / total_weight
    out["teacher_similarity_score"] = scores.clip(0.0, 1.0)
    out["teacher_edge_score"] = (
        0.54 * out["teacher_similarity_score"]
        + 0.22 * norm01(out["runtime_expected_roi"], lo=0.6, hi=3.5)
        + 0.14 * norm01(out["market_overlay_score"], lo=0.3, hi=0.9)
        + 0.10 * norm01(out["late_value_survives_score"], lo=0.3, hi=0.9)
    ).clip(0.0, 1.0)
    return out


def select_tickets(df: pd.DataFrame, coverage: float, min_edge: float, max_tickets_per_race: int) -> pd.DataFrame:
    race_score = df.groupby("race_id")["teacher_edge_score"].max()
    threshold = float(race_score.quantile(1.0 - coverage)) if len(race_score) else float("inf")
    selected = df[df["teacher_edge_score"].ge(max(threshold, min_edge))].copy()
    if selected.empty:
        return selected
    # Keep the strongest ticket structure per race, allowing a wide+umaren pair when both are high.
    selected = (
        selected.sort_values(["race_id", "teacher_edge_score", "runtime_expected_roi"], ascending=[True, False, False])
        .groupby("race_id", as_index=False)
        .head(max_tickets_per_race)
        .copy()
    )
    return selected


def stake_selected(tickets: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    base = np.where(out["ticket_type"].eq("wide"), 400.0, 600.0)
    high = np.where(out["ticket_type"].eq("wide"), 1200.0, 1800.0)
    raw = base + (high - base) * norm01(out["teacher_edge_score"], lo=0.50, hi=0.90)
    out["stake_yen"] = (np.floor(raw / 100.0) * 100.0).clip(lower=100.0)
    out["return_yen"] = np.where(out["hit"], out["pay_per100"] * out["stake_yen"] / 100.0, 0.0)
    return out


def metrics(tickets: pd.DataFrame, label: str) -> dict:
    if tickets.empty:
        return {"label": label, "tickets": 0, "races": 0, "stake_yen": 0.0, "return_yen": 0.0, "profit_yen": 0.0, "roi": 0.0}
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tickets = ticket_candidates()
    train = tickets[tickets["year"].eq(2025)].copy()
    test = tickets[tickets["year"].eq(2026)].copy()
    profiles = {ticket_type: fit_profile(train, ticket_type) for ticket_type in ["wide", "umaren"]}
    scored = score_with_profile(tickets, profiles)
    scored.to_csv(OUT / "teacher_scored_ticket_candidates.csv", index=False, encoding="utf-8-sig")

    rows: list[dict] = []
    selected_frames: list[pd.DataFrame] = []
    for coverage in [0.01, 0.02, 0.03, 0.05, 0.08]:
        for min_edge in [0.58, 0.62, 0.66, 0.70]:
            for max_tickets in [1, 2]:
                train_sel = stake_selected(select_tickets(scored[scored["year"].eq(2025)], coverage, min_edge, max_tickets))
                test_sel = stake_selected(select_tickets(scored[scored["year"].eq(2026)], coverage, min_edge, max_tickets))
                m_train = metrics(train_sel, "train_2025")
                m_test = metrics(test_sel, "test_2026")
                score = (
                    m_train["roi"] * np.sqrt(max(m_train.get("race_hit_rate", 0.0), 0.001)) * np.log1p(max(m_train["races"], 1))
                    + m_train["profit_yen"] / 100000.0
                )
                row = {
                    "coverage": coverage,
                    "min_edge": min_edge,
                    "max_tickets_per_race": max_tickets,
                    "selection_score": score,
                    **{f"train_{k}": v for k, v in m_train.items()},
                    **{f"test_{k}": v for k, v in m_test.items()},
                }
                rows.append(row)
                test_tmp = test_sel.copy()
                test_tmp["coverage"] = coverage
                test_tmp["min_edge"] = min_edge
                test_tmp["max_tickets_per_race"] = max_tickets
                selected_frames.append(test_tmp)
    grid = pd.DataFrame(rows).sort_values(["selection_score", "train_roi"], ascending=[False, False])
    grid.to_csv(OUT / "teacher_policy_grid.csv", index=False, encoding="utf-8-sig")
    best = grid.iloc[0]
    best_test = stake_selected(
        select_tickets(
            scored[scored["year"].eq(2026)],
            float(best["coverage"]),
            float(best["min_edge"]),
            int(best["max_tickets_per_race"]),
        )
    )
    best_test.to_csv(OUT / "teacher_selected_2026_tickets.csv", index=False, encoding="utf-8-sig")

    target_races = {"2026012410010112", "2026013105010111"}
    target_check = scored[scored["race_id"].isin(target_races)].copy()
    target_check.to_csv(OUT / "teacher_target_2026_check.csv", index=False, encoding="utf-8-sig")

    print("TEACHER DISTILLATION")
    print(
        grid[
            [
                "coverage",
                "min_edge",
                "max_tickets_per_race",
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
        ]
        .head(20)
        .to_string(index=False)
    )
    print("\nBEST 2026")
    print(metrics(best_test, "best_test_2026"))
    print("\nTARGET 2026 TOP")
    print(
        target_check[
            [
                "race_id",
                "ticket_type",
                "anchor_name",
                "partner_name",
                "teacher_edge_score",
                "teacher_similarity_score",
                "runtime_expected_roi",
                "ticket_score",
                "market_overlay_score",
                "late_value_survives_score",
                "projected_front5_prob",
                "partner_odds",
                "hit",
                "pay_per100",
            ]
        ]
        .sort_values(["race_id", "teacher_edge_score"], ascending=[True, False])
        .head(20)
        .to_string(index=False)
    )
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
