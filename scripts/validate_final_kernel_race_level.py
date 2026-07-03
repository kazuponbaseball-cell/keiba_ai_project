from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from strict_pair_probability_roi_protocol import VENUE_CODE, load_universe, max_drawdown, num, norm01


FINAL = Path("outputs/analysis/final_operational_quality_v1/standard_explained_tickets.csv")
OUT = Path("outputs/analysis/final_kernel_race_level_v1")


def add_kernel_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    for col in [
        "wide_axis_score",
        "wide_partner_score",
        "market_overlay_score",
        "late_value_survives_score",
        "projected_front5_prob",
        "pair_score",
        "pair_quinella_score",
        "anchor_quinella_score",
        "partner_quinella_score",
        "anchor_danger",
        "partner_danger",
        "anchor_odds",
        "partner_odds",
        "wide_pay",
        "umaren_pay",
    ]:
        out[col] = num(out.get(col), idx, 0.0).fillna(0.0)

    out["wide_quote_proxy"] = 100.0 * (
        np.sqrt(out["anchor_odds"].clip(lower=1.0) * out["partner_odds"].clip(lower=1.0)) * 0.45
    ).clip(1.1, 120.0)
    out["umaren_quote_proxy"] = 100.0 * (
        out["anchor_odds"].clip(lower=1.0) * out["partner_odds"].clip(lower=1.0) * 0.32
    ).clip(1.3, 260.0)
    out["final_kernel_score"] = (
        0.22 * norm01(out["wide_axis_score"])
        + 0.22 * norm01(out["wide_partner_score"])
        + 0.18 * norm01(out["pair_score"])
        + 0.13 * norm01(out["market_overlay_score"])
        + 0.12 * norm01(out["projected_front5_prob"])
        + 0.08 * norm01(out["pair_quinella_score"])
        + 0.05 * norm01(out["umaren_quote_proxy"], lo=500.0, hi=6000.0)
        - 0.08 * norm01(out["anchor_danger"], lo=0.0, hi=0.7)
        - 0.10 * norm01(out["partner_danger"], lo=0.0, hi=0.5)
    )
    out["final_kernel_score"] = out["final_kernel_score"].fillna(0.0)
    return out


def policy_grid() -> list[dict]:
    venue_policies = {
        "all": set(VENUE_CODE.values()) | {"Unknown"},
        "skip_hakodate": (set(VENUE_CODE.values()) | {"Unknown"}) - {"Hakodate"},
        "positive_venues": {"Fukushima", "Niigata", "Chukyo", "Hanshin", "Kokura", "Tokyo"},
    }
    going_policies = {
        "all": {"Good", "Yielding", "Soft", "Heavy", "Unknown"},
        "skip_heavy": {"Good", "Yielding", "Soft", "Unknown"},
        "skip_soft_heavy": {"Good", "Yielding", "Unknown"},
    }
    rows: list[dict] = []
    for (
        coverage,
        venue_policy,
        going_policy,
        axis_min,
        partner_min,
        odds_min,
        odds_max,
        front_min,
        partner_q_min,
        pair_min,
        umaren_quote_min,
        stake_profile,
    ) in product(
        [0.05, 0.063, 0.08, 0.10],
        ["all", "skip_hakodate"],
        ["all", "skip_soft_heavy"],
        [0.62],
        [0.60, 0.66],
        [6.0, 10.0],
        [40.0],
        [0.45, 0.60],
        [0.54],
        [0.74, 0.78],
        [1200.0, 1800.0],
        ["flat", "finalish"],
    ):
        rows.append(
            {
                "coverage": coverage,
                "venue_policy": venue_policy,
                "venue_allowed": venue_policies[venue_policy],
                "going_policy": going_policy,
                "going_allowed": going_policies[going_policy],
                "axis_min": axis_min,
                "partner_min": partner_min,
                "partner_odds_min": odds_min,
                "partner_odds_max": odds_max,
                "front_min": front_min,
                "anchor_danger_max": 0.55,
                "partner_danger_max": 0.35,
                "partner_quinella_min": partner_q_min,
                "umaren_pair_score_min": pair_min,
                "umaren_partner_odds_max": 25.0,
                "umaren_quote_min": umaren_quote_min,
                "stake_profile": stake_profile,
            }
        )
    return rows


def prefilter(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    mask = (
        df["venue"].isin(params["venue_allowed"])
        & df["going"].isin(params["going_allowed"])
        & df["wide_axis_score"].ge(params["axis_min"])
        & df["wide_partner_score"].ge(params["partner_min"])
        & df["partner_odds"].between(params["partner_odds_min"], params["partner_odds_max"])
        & df["projected_front5_prob"].ge(params["front_min"])
        & df["anchor_danger"].le(params["anchor_danger_max"])
        & df["partner_danger"].le(params["partner_danger_max"])
    )
    return df[mask].copy()


def race_representatives(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    work = prefilter(df, params)
    if work.empty:
        return work
    return (
        work.sort_values(
            ["race_id", "final_kernel_score", "pair_score", "market_overlay_score"],
            ascending=[True, False, False, False],
        )
        .groupby("race_id", as_index=False)
        .head(1)
        .copy()
    )


def threshold_from_coverage(train_df: pd.DataFrame, params: dict) -> float:
    reps = race_representatives(train_df, params)
    if reps.empty:
        return float("inf")
    return float(reps["final_kernel_score"].quantile(1.0 - params["coverage"]))


def select_pairs(df: pd.DataFrame, params: dict, threshold: float) -> pd.DataFrame:
    reps = race_representatives(df, params)
    if reps.empty:
        return reps
    return reps[reps["final_kernel_score"].ge(threshold)].copy()


def stake_values(base: pd.DataFrame, params: dict, ticket_type: str) -> pd.Series:
    if params["stake_profile"] == "flat":
        return pd.Series(200.0 if ticket_type == "wide" else 100.0, index=base.index)
    value = (
        0.45 * norm01(base["market_overlay_score"], lo=0.4, hi=0.9)
        + 0.35 * norm01(base["pair_score"], lo=0.65, hi=0.85)
        + 0.20 * norm01(base["final_kernel_score"], lo=0.55, hi=0.85)
    ).clip(0.0, 1.0)
    if ticket_type == "wide":
        raw = 400.0 + 700.0 * value
    else:
        raw = 600.0 + 1100.0 * value
    return (np.floor(raw / 100.0) * 100.0).clip(lower=100.0)


def tickets_from_pairs(pairs: pd.DataFrame, params: dict) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    base = pairs.copy()
    base["pair_key"] = base["race_id"] + ":" + base["anchor_no"].astype(str) + "-" + base["partner_no"].astype(str)
    frames: list[pd.DataFrame] = []

    wide = base.copy()
    wide["ticket_type"] = "wide"
    wide["stake_yen"] = stake_values(wide, params, "wide")
    wide["hit"] = wide["wide_hit"].astype(bool)
    wide["return_yen"] = np.where(wide["hit"], wide["wide_pay"] * wide["stake_yen"] / 100.0, 0.0)
    frames.append(wide)

    umaren_mask = (
        base["pair_score"].ge(params["umaren_pair_score_min"])
        & base["partner_quinella_score"].ge(params["partner_quinella_min"])
        & base["partner_odds"].le(params["umaren_partner_odds_max"])
        & base["umaren_quote_proxy"].ge(params["umaren_quote_min"])
    )
    umaren = base[umaren_mask].copy()
    if not umaren.empty:
        umaren["ticket_type"] = "umaren"
        umaren["stake_yen"] = stake_values(umaren, params, "umaren")
        umaren["hit"] = umaren["umaren_hit"].astype(bool)
        umaren["return_yen"] = np.where(umaren["hit"], umaren["umaren_pay"] * umaren["stake_yen"] / 100.0, 0.0)
        frames.append(umaren)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["year"] = out["race_id"].str[:4].astype(int)
    out["ticket_key"] = out["ticket_type"] + ":" + out["pair_key"]
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
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
        }
    stake = float(tickets["stake_yen"].sum())
    ret = float(tickets["return_yen"].sum())
    by_race = tickets.groupby("race_id", sort=False).agg(
        stake=("stake_yen", "sum"),
        ret=("return_yen", "sum"),
        hit=("hit", "max"),
    )
    profit = by_race["ret"] - by_race["stake"]
    return {
        "label": label,
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(tickets["hit"].mean()),
        "race_hit_rate": float(by_race["hit"].mean()),
        "max_drawdown_yen": max_drawdown(profit),
        "wide_tickets": int((tickets["ticket_type"] == "wide").sum()),
        "umaren_tickets": int((tickets["ticket_type"] == "umaren").sum()),
    }


def evaluate(df: pd.DataFrame, params: dict, threshold: float, label: str) -> tuple[dict, pd.DataFrame]:
    pairs = select_pairs(df, params, threshold)
    tickets = tickets_from_pairs(pairs, params)
    m = metrics(tickets, label)
    total_races = int(df["race_id"].nunique())
    m["candidate_races"] = total_races
    m["race_selection_rate"] = float(m["races"] / total_races) if total_races else 0.0
    return m, tickets


def policy_score(m: dict) -> float:
    if m.get("races", 0) < 60:
        return -1e9
    if m.get("race_selection_rate", 1.0) > 0.18:
        return -1e9
    if m.get("race_hit_rate", 0.0) < 0.04:
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
        train_grid = pd.DataFrame(rows).sort_values(["selection_score", "roi"], ascending=[False, False])
        train_rows.append(train_grid.head(100))
        best = train_grid.iloc[0]
        params = grids[int(best["grid_id"])]
        m, tickets = evaluate(test, params, float(best["score_threshold"]), f"wf_test_{test_year}")
        m.update({k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}})
        m["test_year"] = test_year
        m["train_roi"] = float(best["roi"])
        m["train_races"] = int(best["races"])
        m["train_race_selection_rate"] = float(best["race_selection_rate"])
        m["train_profit_yen"] = float(best["profit_yen"])
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


def final_exact_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(FINAL, dtype={"race_id": str}, low_memory=False)
    final = final[num(final.get("runtime_stake_yen"), final.index, 0.0).fillna(0.0).gt(0)].copy()
    final["stake_yen"] = num(final["runtime_stake_yen"]).fillna(0.0)
    final["return_yen"] = num(final["runtime_return_yen"]).fillna(0.0)
    final["hit"] = final["hit"].astype(bool)
    final["year"] = final["race_id"].str[:4].astype(int)

    rows = [metrics(final, "final_exact_all")]
    for y, g in final.groupby("year"):
        rows.append(metrics(g, f"final_exact_{int(y)}"))

    race = final.groupby("race_id", sort=False).agg(
        year=("year", "first"),
        stake_yen=("stake_yen", "sum"),
        return_yen=("return_yen", "sum"),
        hit=("hit", "max"),
        tickets=("race_id", "size"),
    ).reset_index()
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    sensitivity: list[dict] = []
    for drop_n in [0, 1, 3, 5, 10, 15, 20]:
        keep = race.sort_values("profit_yen", ascending=False).iloc[drop_n:].copy()
        stake = float(keep["stake_yen"].sum())
        ret = float(keep["return_yen"].sum())
        sensitivity.append(
            {
                "drop_top_profit_races": drop_n,
                "races": int(len(keep)),
                "stake_yen": stake,
                "return_yen": ret,
                "profit_yen": ret - stake,
                "roi": ret / stake if stake else 0.0,
                "race_hit_rate": float(keep["hit"].mean()) if len(keep) else 0.0,
            }
        )
    return pd.DataFrame(rows), race.sort_values("profit_yen", ascending=False), pd.DataFrame(sensitivity)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_kernel_features(load_universe())
    train_grid, wf_summary, wf_tickets = walkforward(df)
    exact_summary, exact_by_race, exact_sensitivity = final_exact_audit()

    train_grid.to_csv(OUT / "walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(OUT / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    wf_tickets.to_csv(OUT / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
    exact_summary.to_csv(OUT / "final_exact_summary.csv", index=False, encoding="utf-8-sig")
    exact_by_race.to_csv(OUT / "final_exact_by_race.csv", index=False, encoding="utf-8-sig")
    exact_sensitivity.to_csv(OUT / "final_exact_top_hit_sensitivity.csv", index=False, encoding="utf-8-sig")

    print("FINAL KERNEL RACE-LEVEL WALKFORWARD")
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
        "venue_policy",
        "going_policy",
        "coverage",
        "stake_profile",
        "train_roi",
        "train_races",
        "train_race_selection_rate",
    ]
    print(wf_summary[cols].to_string(index=False))
    print("\nFINAL EXACT")
    print(exact_summary[["label", "tickets", "races", "stake_yen", "return_yen", "profit_yen", "roi", "race_hit_rate", "max_drawdown_yen"]].to_string(index=False))
    print("\nFINAL EXACT TOP-HIT SENSITIVITY")
    print(exact_sensitivity.to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
