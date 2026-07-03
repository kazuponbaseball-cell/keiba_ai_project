from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from strict_pair_probability_roi_protocol import (
    VENUE_CODE,
    add_calibrated_probs,
    build_raw_probability_features,
    load_universe,
    metrics,
    num,
    policy_score,
    tickets_from_pairs,
)


OUT = Path("outputs/analysis/strict_race_level_roi_v1")


def policy_grid() -> list[dict]:
    venue_policies = {
        "skip_hakodate": set(VENUE_CODE.values()) - {"Hakodate"},
        "positive_venues": {"Fukushima", "Niigata", "Chukyo", "Hanshin", "Kokura", "Tokyo"},
    }
    going_policies = {
        "skip_heavy": {"Good", "Yielding", "Soft", "Unknown"},
        "skip_soft_heavy": {"Good", "Yielding", "Unknown"},
    }
    rows: list[dict] = []
    for (
        coverage,
        venue_policy,
        going_policy,
        wide_ev_min,
        umaren_ev_min,
        front_min,
        market_min,
        partner_odds_min,
    ) in product(
        [0.05, 0.10, 0.15],
        venue_policies.keys(),
        going_policies.keys(),
        [1.25, 1.50],
        [1.50],
        [0.25, 0.40],
        [0.35],
        [10.0, 12.0],
    ):
        rows.append(
            {
                "coverage": coverage,
                "venue_policy": venue_policy,
                "venue_allowed": venue_policies[venue_policy],
                "going_policy": going_policy,
                "going_allowed": going_policies[going_policy],
                "wide_ev_min": wide_ev_min,
                "umaren_ev_min": umaren_ev_min,
                "front_min": front_min,
                "market_min": market_min,
                "partner_odds_min": partner_odds_min,
                "ticket_mode": "wide_umaren",
                "wide_stake": 200.0,
                "umaren_stake": 100.0,
            }
        )
    return rows


def race_representatives(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["policy_rank_score"] = num(work.get("strict_rank_score"), work.index, 0.0).fillna(0.0)
    sort_cols = ["race_id", "policy_rank_score", "wide_ev_proxy", "umaren_ev_proxy", "front5_prob_cal"]
    return (
        work.sort_values(sort_cols, ascending=[True, False, False, False, False])
        .groupby("race_id", as_index=False)
        .head(1)
        .copy()
    )


def threshold_from_race_coverage(df: pd.DataFrame, coverage: float) -> float:
    reps = race_representatives(df)
    if reps.empty:
        return float("inf")
    return float(num(reps.get("policy_rank_score"), reps.index, 0.0).fillna(0.0).quantile(1.0 - coverage))


def select_race_level_pairs(df: pd.DataFrame, params: dict, threshold: float) -> pd.DataFrame:
    reps = race_representatives(df)
    if reps.empty:
        return reps
    mask = (
        reps["venue"].isin(params["venue_allowed"])
        & reps["going"].isin(params["going_allowed"])
        & reps["policy_rank_score"].ge(threshold)
        & reps["wide_ev_proxy"].ge(params["wide_ev_min"])
        & reps["front5_prob_cal"].ge(params["front_min"])
        & reps["market_overlay_score"].ge(params["market_min"])
        & reps["partner_odds"].ge(params["partner_odds_min"])
        & reps["partner_danger"].le(0.35)
        & reps["anchor_danger"].le(0.55)
    )
    return reps[mask].copy()


def evaluate(apply_df: pd.DataFrame, params: dict, threshold: float, label: str) -> tuple[dict, pd.DataFrame]:
    pairs = select_race_level_pairs(apply_df, params, threshold)
    tickets = tickets_from_pairs(pairs, params)
    return metrics(tickets, label), tickets


def add_selection_rate(m: dict, scored: pd.DataFrame) -> dict:
    total_races = int(scored["race_id"].nunique())
    out = dict(m)
    out["candidate_races"] = total_races
    out["race_selection_rate"] = (float(out["races"]) / total_races) if total_races else 0.0
    return out


def walkforward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid_rows = policy_grid()
    train_rows: list[pd.DataFrame] = []
    wf_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    for test_year in [2025, 2026]:
        train_raw = df[df["year"] < test_year].copy()
        test_raw = df[df["year"] == test_year].copy()
        train_scored, meta = add_calibrated_probs(train_raw, train_raw)
        test_scored, _ = add_calibrated_probs(train_raw, test_raw)
        rows: list[dict] = []
        for i, params in enumerate(grid_rows):
            threshold = threshold_from_race_coverage(train_scored, params["coverage"])
            m, _ = evaluate(train_scored, params, threshold, f"train_{test_year}_{i}")
            m = add_selection_rate(m, train_scored)
            row = m | {k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}}
            row["grid_id"] = i
            row["score_threshold"] = threshold
            row["test_year"] = test_year
            row["selection_score"] = policy_score(m)
            row.update(meta)
            rows.append(row)
        train_grid = pd.DataFrame(rows).sort_values(["selection_score", "roi"], ascending=[False, False])
        train_rows.append(train_grid.head(100))

        best = train_grid.iloc[0]
        params = grid_rows[int(best["grid_id"])]
        m, tickets = evaluate(test_scored, params, float(best["score_threshold"]), f"wf_test_{test_year}")
        m = add_selection_rate(m, test_scored)
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
    tickets_out = pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame()
    return pd.concat(train_rows, ignore_index=True), pd.DataFrame(wf_rows), tickets_out


def train_val_holdout(df: pd.DataFrame) -> pd.DataFrame:
    grid_rows = policy_grid()
    train_raw = df[df["year"] == 2024].copy()
    val_raw = df[df["year"] == 2025].copy()
    hold_raw = df[df["year"] == 2026].copy()
    train_scored, meta = add_calibrated_probs(train_raw, train_raw)
    val_scored, _ = add_calibrated_probs(train_raw, val_raw)
    hold_scored, _ = add_calibrated_probs(pd.concat([train_raw, val_raw], ignore_index=True, sort=False), hold_raw)
    rows: list[dict] = []
    for i, params in enumerate(grid_rows):
        threshold = threshold_from_race_coverage(train_scored, params["coverage"])
        m_train, _ = evaluate(train_scored, params, threshold, f"train_2024_{i}")
        m_val, _ = evaluate(val_scored, params, threshold, f"val_2025_{i}")
        m_hold, _ = evaluate(hold_scored, params, threshold, f"hold_2026_{i}")
        m_train = add_selection_rate(m_train, train_scored)
        m_val = add_selection_rate(m_val, val_scored)
        m_hold = add_selection_rate(m_hold, hold_scored)
        row = {k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}}
        row["grid_id"] = i
        row["score_threshold"] = threshold
        row.update({f"train_{k}": v for k, v in m_train.items()})
        row.update({f"val_{k}": v for k, v in m_val.items()})
        row.update({f"hold_{k}": v for k, v in m_hold.items()})
        row.update(meta)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["dev_score"] = (
        out["val_roi"] * np.sqrt(out["val_race_hit_rate"].clip(lower=0.001)) * np.log1p(out["val_races"])
        + 0.25 * out["train_roi"]
        + out["val_profit_yen"] / 100000.0
    )
    return out.sort_values(["dev_score", "val_profit_yen"], ascending=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_raw_probability_features(load_universe())
    train_grid, wf_summary, wf_tickets = walkforward(df)
    holdout = train_val_holdout(df)
    train_grid.to_csv(OUT / "walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(OUT / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    wf_tickets.to_csv(OUT / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(OUT / "train2024_val2025_hold2026_grid.csv", index=False, encoding="utf-8-sig")

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
        "train_roi",
        "train_races",
        "train_race_selection_rate",
    ]
    print("RACE-LEVEL WALKFORWARD")
    print(wf_summary[cols].to_string(index=False))
    print("\nTRAIN2024 -> VAL2025 -> HOLD2026 TOP")
    hcols = [
        "grid_id",
        "train_races",
        "train_race_selection_rate",
        "train_roi",
        "val_races",
        "val_race_selection_rate",
        "val_roi",
        "val_profit_yen",
        "hold_races",
        "hold_race_selection_rate",
        "hold_roi",
        "hold_profit_yen",
        "venue_policy",
        "going_policy",
        "coverage",
        "wide_ev_min",
        "front_min",
        "partner_odds_min",
    ]
    print(holdout[hcols].head(25).to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
