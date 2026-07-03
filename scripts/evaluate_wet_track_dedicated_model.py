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


OUT = Path("outputs/analysis/wet_track_dedicated_model_v1")

GOING_GROUPS = {
    "yielding_plus": {"Yielding", "Soft", "Heavy"},
    "soft_heavy": {"Soft", "Heavy"},
    "heavy_only": {"Heavy"},
}

VENUE_POLICIES = {
    "all": set(VENUE_CODE.values()) | {"Unknown"},
    "skip_hakodate": (set(VENUE_CODE.values()) | {"Unknown"}) - {"Hakodate"},
    "positive_venues": {"Fukushima", "Niigata", "Chukyo", "Hanshin", "Kokura", "Tokyo"},
}


def race_representatives(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["policy_rank_score"] = num(work.get("strict_rank_score"), work.index, 0.0).fillna(0.0)
    sort_cols = ["race_id", "policy_rank_score", "wide_ev_proxy", "umaren_ev_proxy", "front5_prob_cal"]
    return (
        work.sort_values(sort_cols, ascending=[True, False, False, False, False])
        .groupby("race_id", as_index=False)
        .head(1)
        .copy()
    )


def threshold_from_coverage(df: pd.DataFrame, params: dict) -> float:
    reps = race_representatives(df)
    reps = reps[
        reps["going"].isin(params["going_allowed"])
        & reps["venue"].isin(params["venue_allowed"])
    ].copy()
    if reps.empty:
        return float("inf")
    return float(num(reps["policy_rank_score"], reps.index, 0.0).fillna(0.0).quantile(1.0 - params["coverage"]))


def policy_grid() -> list[dict]:
    rows: list[dict] = []
    for (
        going_group,
        calibration_scope,
        venue_policy,
        coverage,
        wide_ev_min,
        umaren_ev_min,
        front_min,
        market_min,
        partner_odds_min,
        partner_danger_max,
        anchor_danger_max,
        umaren_stake,
    ) in product(
        GOING_GROUPS.keys(),
        ["all", "wet_only"],
        ["all", "positive_venues"],
        [0.05, 0.10, 0.15],
        [1.25, 1.50],
        [1.50],
        [0.25, 0.40, 0.55],
        [0.35, 0.45],
        [5.0, 10.0],
        [0.35],
        [0.55],
        [0.0, 100.0],
    ):
        rows.append(
            {
                "going_group": going_group,
                "going_allowed": GOING_GROUPS[going_group],
                "calibration_scope": calibration_scope,
                "venue_policy": venue_policy,
                "venue_allowed": VENUE_POLICIES[venue_policy],
                "coverage": coverage,
                "wide_ev_min": wide_ev_min,
                "umaren_ev_min": umaren_ev_min,
                "front_min": front_min,
                "market_min": market_min,
                "partner_odds_min": partner_odds_min,
                "partner_danger_max": partner_danger_max,
                "anchor_danger_max": anchor_danger_max,
                "ticket_mode": "wide_umaren" if umaren_stake > 0 else "wide_only",
                "wide_stake": 200.0,
                "umaren_stake": umaren_stake,
            }
        )
    return rows


def select_race_level_pairs(df: pd.DataFrame, params: dict, threshold: float) -> pd.DataFrame:
    reps = race_representatives(df)
    if reps.empty:
        return reps
    mask = (
        reps["going"].isin(params["going_allowed"])
        & reps["venue"].isin(params["venue_allowed"])
        & reps["policy_rank_score"].ge(threshold)
        & reps["wide_ev_proxy"].ge(params["wide_ev_min"])
        & reps["front5_prob_cal"].ge(params["front_min"])
        & reps["market_overlay_score"].ge(params["market_min"])
        & reps["partner_odds"].ge(params["partner_odds_min"])
        & reps["partner_danger"].le(params["partner_danger_max"])
        & reps["anchor_danger"].le(params["anchor_danger_max"])
    )
    return reps[mask].copy()


def evaluate(df: pd.DataFrame, params: dict, threshold: float, label: str) -> tuple[dict, pd.DataFrame]:
    pairs = select_race_level_pairs(df, params, threshold)
    tickets = tickets_from_pairs(pairs, params)
    return metrics(tickets, label), tickets


def add_selection_rate(m: dict, scored: pd.DataFrame, params: dict) -> dict:
    eligible = scored[
        scored["going"].isin(params["going_allowed"])
        & scored["venue"].isin(params["venue_allowed"])
    ]
    total_races = int(eligible["race_id"].nunique())
    out = dict(m)
    out["eligible_races"] = total_races
    out["race_selection_rate"] = float(out["races"]) / total_races if total_races else 0.0
    return out


def calibrate_for_params(train_raw: pd.DataFrame, apply_raw: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict]:
    if params["calibration_scope"] == "wet_only":
        train_cal = train_raw[train_raw["going"].isin(params["going_allowed"])].copy()
        if train_cal["race_id"].nunique() < 80 or len(train_cal) < 800:
            train_cal = train_raw
    else:
        train_cal = train_raw
    return add_calibrated_probs(train_cal, apply_raw)


def run_walkforward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grids = policy_grid()
    train_top_frames: list[pd.DataFrame] = []
    wf_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    for test_year in [2025, 2026]:
        train_raw = df[df["year"] < test_year].copy()
        test_raw = df[df["year"] == test_year].copy()
        rows: list[dict] = []
        scored_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame, dict]] = {}
        for i, params in enumerate(grids):
            cache_key = (params["calibration_scope"], params["going_group"])
            if cache_key not in scored_cache:
                train_scored, meta = calibrate_for_params(train_raw, train_raw, params)
                test_scored, _ = calibrate_for_params(train_raw, test_raw, params)
                scored_cache[cache_key] = (train_scored, test_scored, meta)
            train_scored, _, meta = scored_cache[cache_key]
            threshold = threshold_from_coverage(train_scored, params)
            m, _ = evaluate(train_scored, params, threshold, f"train_{test_year}_{i}")
            m = add_selection_rate(m, train_scored, params)
            row = m | {k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}}
            row["grid_id"] = i
            row["test_year"] = test_year
            row["score_threshold"] = threshold
            row["selection_score"] = policy_score(m)
            row.update(meta)
            rows.append(row)
        train_grid = pd.DataFrame(rows).sort_values(["selection_score", "roi"], ascending=[False, False])
        train_top_frames.append(train_grid.head(200))

        best = train_grid.iloc[0]
        params = grids[int(best["grid_id"])]
        _, test_scored, _ = scored_cache[(params["calibration_scope"], params["going_group"])]
        m, tickets = evaluate(test_scored, params, float(best["score_threshold"]), f"wf_test_{test_year}")
        m = add_selection_rate(m, test_scored, params)
        m.update({k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}})
        m["test_year"] = test_year
        m["train_roi"] = float(best["roi"])
        m["train_races"] = int(best["races"])
        m["train_profit_yen"] = float(best["profit_yen"])
        m["train_race_hit_rate"] = float(best["race_hit_rate"])
        m["score_threshold"] = float(best["score_threshold"])
        wf_rows.append(m)
        if not tickets.empty:
            tmp = tickets.copy()
            tmp["test_year"] = test_year
            tmp["selected_grid_id"] = int(best["grid_id"])
            ticket_frames.append(tmp)
    tickets_out = pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame()
    return pd.concat(train_top_frames, ignore_index=True), pd.DataFrame(wf_rows), tickets_out


def train_val_holdout(df: pd.DataFrame) -> pd.DataFrame:
    grids = policy_grid()
    train_raw = df[df["year"] == 2024].copy()
    val_raw = df[df["year"] == 2025].copy()
    hold_raw = df[df["year"] == 2026].copy()
    rows: list[dict] = []
    cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]] = {}
    for i, params in enumerate(grids):
        cache_key = (params["calibration_scope"], params["going_group"])
        if cache_key not in cache:
            train_scored, meta = calibrate_for_params(train_raw, train_raw, params)
            val_scored, _ = calibrate_for_params(train_raw, val_raw, params)
            train_val_raw = pd.concat([train_raw, val_raw], ignore_index=True, sort=False)
            hold_scored, _ = calibrate_for_params(train_val_raw, hold_raw, params)
            cache[cache_key] = (train_scored, val_scored, hold_scored, meta)
        train_scored, val_scored, hold_scored, meta = cache[cache_key]
        threshold = threshold_from_coverage(train_scored, params)
        m_train, _ = evaluate(train_scored, params, threshold, f"train_2024_{i}")
        m_val, _ = evaluate(val_scored, params, threshold, f"val_2025_{i}")
        m_hold, _ = evaluate(hold_scored, params, threshold, f"hold_2026_{i}")
        m_train = add_selection_rate(m_train, train_scored, params)
        m_val = add_selection_rate(m_val, val_scored, params)
        m_hold = add_selection_rate(m_hold, hold_scored, params)
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
    train_top, wf_summary, wf_tickets = run_walkforward(df)
    holdout = train_val_holdout(df)
    train_top.to_csv(OUT / "walkforward_train_top200.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(OUT / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    wf_tickets.to_csv(OUT / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(OUT / "train2024_val2025_hold2026_grid.csv", index=False, encoding="utf-8-sig")
    print("WET TRACK DEDICATED WALKFORWARD")
    cols = [
        "label",
        "test_year",
        "eligible_races",
        "races",
        "race_selection_rate",
        "tickets",
        "stake_yen",
        "return_yen",
        "profit_yen",
        "roi",
        "race_hit_rate",
        "max_drawdown_yen",
        "going_group",
        "calibration_scope",
        "venue_policy",
        "coverage",
        "wide_ev_min",
        "front_min",
        "market_min",
        "partner_odds_min",
        "ticket_mode",
        "train_roi",
        "train_races",
    ]
    print(wf_summary[cols].to_string(index=False))
    print("\nTRAIN2024 -> VAL2025 -> HOLD2026 TOP")
    hcols = [
        "grid_id",
        "train_races",
        "train_roi",
        "val_races",
        "val_roi",
        "hold_races",
        "hold_roi",
        "going_group",
        "calibration_scope",
        "venue_policy",
        "coverage",
        "wide_ev_min",
        "front_min",
        "market_min",
        "partner_odds_min",
        "ticket_mode",
    ]
    print(holdout[hcols].head(25).to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
