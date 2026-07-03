from __future__ import annotations

from pathlib import Path

import pandas as pd

from strict_pair_probability_roi_protocol import (
    add_calibrated_probs,
    build_raw_probability_features,
    load_universe,
    metrics,
    num,
    tickets_from_pairs,
)


OUT = Path("outputs/analysis/wet_track_quick_gate_v1")


PARAMS = [
    {
        "name": "wet_balanced",
        "going_allowed": {"Yielding", "Soft", "Heavy"},
        "venue_allowed": {"Sapporo", "Fukushima", "Niigata", "Tokyo", "Nakayama", "Chukyo", "Kyoto", "Hanshin", "Kokura"},
        "coverage": 0.10,
        "wide_ev_min": 1.25,
        "umaren_ev_min": 1.50,
        "front_min": 0.40,
        "market_min": 0.35,
        "partner_odds_min": 5.0,
        "partner_danger_max": 0.35,
        "anchor_danger_max": 0.55,
        "wide_stake": 200.0,
        "umaren_stake": 100.0,
    },
    {
        "name": "wet_strict_hit",
        "going_allowed": {"Yielding", "Soft", "Heavy"},
        "venue_allowed": {"Fukushima", "Niigata", "Tokyo", "Chukyo", "Hanshin", "Kokura"},
        "coverage": 0.05,
        "wide_ev_min": 1.50,
        "umaren_ev_min": 1.75,
        "front_min": 0.55,
        "market_min": 0.45,
        "partner_odds_min": 5.0,
        "partner_danger_max": 0.35,
        "anchor_danger_max": 0.45,
        "wide_stake": 200.0,
        "umaren_stake": 0.0,
    },
    {
        "name": "soft_heavy_value",
        "going_allowed": {"Soft", "Heavy"},
        "venue_allowed": {"Sapporo", "Fukushima", "Niigata", "Tokyo", "Nakayama", "Chukyo", "Kyoto", "Hanshin", "Kokura"},
        "coverage": 0.10,
        "wide_ev_min": 1.25,
        "umaren_ev_min": 1.50,
        "front_min": 0.40,
        "market_min": 0.35,
        "partner_odds_min": 5.0,
        "partner_danger_max": 0.35,
        "anchor_danger_max": 0.55,
        "wide_stake": 200.0,
        "umaren_stake": 100.0,
    },
    {
        "name": "soft_heavy_strict",
        "going_allowed": {"Soft", "Heavy"},
        "venue_allowed": {"Fukushima", "Niigata", "Tokyo", "Chukyo", "Hanshin", "Kokura"},
        "coverage": 0.05,
        "wide_ev_min": 1.50,
        "umaren_ev_min": 1.75,
        "front_min": 0.55,
        "market_min": 0.45,
        "partner_odds_min": 5.0,
        "partner_danger_max": 0.35,
        "anchor_danger_max": 0.45,
        "wide_stake": 200.0,
        "umaren_stake": 0.0,
    },
]


def race_representatives(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["policy_rank_score"] = num(work.get("strict_rank_score"), work.index, 0.0).fillna(0.0)
    return (
        work.sort_values(
            ["race_id", "policy_rank_score", "wide_ev_proxy", "umaren_ev_proxy", "front5_prob_cal"],
            ascending=[True, False, False, False, False],
        )
        .groupby("race_id", as_index=False)
        .head(1)
        .copy()
    )


def threshold(train_scored: pd.DataFrame, params: dict) -> float:
    reps = race_representatives(train_scored)
    eligible = reps[
        reps["going"].isin(params["going_allowed"])
        & reps["venue"].isin(params["venue_allowed"])
    ]
    if eligible.empty:
        return float("inf")
    return float(num(eligible["policy_rank_score"], eligible.index, 0.0).fillna(0.0).quantile(1.0 - params["coverage"]))


def select_pairs(scored: pd.DataFrame, params: dict, score_threshold: float) -> pd.DataFrame:
    reps = race_representatives(scored)
    if reps.empty:
        return reps
    mask = (
        reps["going"].isin(params["going_allowed"])
        & reps["venue"].isin(params["venue_allowed"])
        & reps["policy_rank_score"].ge(score_threshold)
        & reps["wide_ev_proxy"].ge(params["wide_ev_min"])
        & reps["front5_prob_cal"].ge(params["front_min"])
        & reps["market_overlay_score"].ge(params["market_min"])
        & reps["partner_odds"].ge(params["partner_odds_min"])
        & reps["partner_danger"].le(params["partner_danger_max"])
        & reps["anchor_danger"].le(params["anchor_danger_max"])
    )
    return reps[mask].copy()


def evaluate(scored: pd.DataFrame, params: dict, score_threshold: float, label: str) -> tuple[dict, pd.DataFrame]:
    pairs = select_pairs(scored, params, score_threshold)
    tickets = tickets_from_pairs(pairs, params)
    m = metrics(tickets, label)
    eligible = scored[
        scored["going"].isin(params["going_allowed"])
        & scored["venue"].isin(params["venue_allowed"])
    ]
    m["eligible_races"] = int(eligible["race_id"].nunique())
    m["race_selection_rate"] = float(m["races"]) / m["eligible_races"] if m["eligible_races"] else 0.0
    return m, tickets


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = build_raw_probability_features(load_universe())
    rows = []
    ticket_frames = []
    for test_year in [2025, 2026]:
        train_raw = raw[raw["year"] < test_year].copy()
        test_raw = raw[raw["year"] == test_year].copy()
        train_scored, _ = add_calibrated_probs(train_raw, train_raw)
        test_scored, _ = add_calibrated_probs(train_raw, test_raw)
        for params in PARAMS:
            th = threshold(train_scored, params)
            train_m, _ = evaluate(train_scored, params, th, f"train_{test_year}_{params['name']}")
            test_m, tickets = evaluate(test_scored, params, th, f"test_{test_year}_{params['name']}")
            row = {
                "test_year": test_year,
                "model": params["name"],
                "score_threshold": th,
                "going_allowed": ",".join(sorted(params["going_allowed"])),
                "coverage": params["coverage"],
                "wide_ev_min": params["wide_ev_min"],
                "front_min": params["front_min"],
                "market_min": params["market_min"],
                "partner_odds_min": params["partner_odds_min"],
            }
            row.update({f"train_{k}": v for k, v in train_m.items()})
            row.update({f"test_{k}": v for k, v in test_m.items()})
            rows.append(row)
            if not tickets.empty:
                t = tickets.copy()
                t["test_year"] = test_year
                t["model"] = params["name"]
                ticket_frames.append(t)
    summary = pd.DataFrame(rows)
    tickets_out = pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame()
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    tickets_out.to_csv(OUT / "selected_tickets.csv", index=False, encoding="utf-8-sig")
    print(summary[
        [
            "test_year",
            "model",
            "test_eligible_races",
            "test_races",
            "test_race_selection_rate",
            "test_tickets",
            "test_stake_yen",
            "test_return_yen",
            "test_profit_yen",
            "test_roi",
            "test_race_hit_rate",
            "test_max_drawdown_yen",
            "train_races",
            "train_roi",
            "train_race_hit_rate",
        ]
    ].to_string(index=False))
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
