from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_strongest_final_strength_model import (
    metrics,
    policy_grid,
    policy_score,
    prefilter,
    race_representatives,
    stake_from_score,
    threshold_from_coverage,
)


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "outputs" / "analysis" / "strongest_final_strength_model_v1" / "pair_strength_universe.csv"
OUT = ROOT / "outputs" / "analysis" / "ticket_conversion_fixes_v1"

FAST_PARAMS_BY_TEST_YEAR = {
    2025: {
        "coverage": 0.05,
        "venue_policy": "all",
        "going_policy": "all",
        "min_market": 0.60,
        "min_late": 0.72,
        "min_safety": 0.42,
        "min_context": 0.42,
        "umaren_ticket_min": 0.58,
        "umaren_quote_min": 1000.0,
    },
    2026: {
        "coverage": 0.08,
        "venue_policy": "all",
        "going_policy": "all",
        "min_market": 0.60,
        "min_late": 0.72,
        "min_safety": 0.52,
        "min_context": 0.50,
        "umaren_ticket_min": 0.58,
        "umaren_quote_min": 1000.0,
    },
}


def num(s: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if s is None:
        if index is None:
            raise ValueError("index is required")
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(s, errors="coerce")


def clip01(s: pd.Series | None, index: pd.Index) -> pd.Series:
    return num(s, index, 0.0).fillna(0.0).clip(0.0, 1.0)


def add_conversion_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    anchor_rank_proxy = num(out.get("anchor_quinella_model_rank"), idx, 99).fillna(99)
    partner_rank = num(out.get("partner_ai_rank"), idx, 99).fillna(99)
    anchor_pop = num(out.get("anchor_pop"), idx, 99).fillna(99)
    partner_pop = num(out.get("partner_pop"), idx, 99).fillna(99)
    partner_place = clip01(out.get("partner_place_score"), idx)
    partner_q = clip01(out.get("partner_quinella_score"), idx)
    pair_q = clip01(out.get("pair_quinella_score"), idx)
    pair_score = clip01(out.get("pair_score"), idx)
    front = clip01(out.get("projected_front5_prob"), idx)

    out["partner_ability_floor_score"] = (
        0.34 * partner_place
        + 0.30 * partner_q
        + 0.22 * pair_q
        + 0.14 * pair_score
    ).clip(0.0, 1.0)
    out["partner_ability_floor_fail_flag"] = (
        (partner_place.lt(0.42) & partner_q.lt(0.46))
        | partner_rank.gt(6)
        | (out["partner_ability_floor_score"].lt(0.50) & front.lt(0.72))
    ).astype(float)
    out["top_eval_in_pair_flag"] = (
        anchor_rank_proxy.le(3)
        | partner_rank.le(3)
        | anchor_pop.le(2)
        | partner_pop.le(2)
    ).astype(float)
    out["top_eval_missing_risk_flag"] = (
        out["top_eval_in_pair_flag"].lt(1)
        & num(out.get("market_overlay_score"), idx, 0.0).lt(0.78)
    ).astype(float)
    out["umaren_suitable_score"] = (
        0.36 * pair_q
        + 0.28 * partner_q
        + 0.20 * partner_place
        + 0.10 * clip01(out.get("umaren_strongest_ticket_score"), idx)
        + 0.06 * front
    ).clip(0.0, 1.0)
    out["wide_suitable_score"] = (
        0.34 * clip01(out.get("wide_strongest_ticket_score"), idx)
        + 0.24 * partner_place
        + 0.18 * pair_score
        + 0.14 * front
        + 0.10 * clip01(out.get("market_overlay_score"), idx)
    ).clip(0.0, 1.0)
    out["umaren_over_wide_edge_score"] = (
        out["umaren_suitable_score"] - out["wide_suitable_score"]
    ).clip(-1.0, 1.0)
    out["front_but_partner_weak_flag"] = (
        front.ge(0.70)
        & out["partner_ability_floor_score"].lt(0.55)
    ).astype(float)
    anchor_horses = out[
        ["race_id", "anchor_no", "anchor_name", "anchor_pop", "anchor_odds", "anchor_quinella_model_rank"]
    ].rename(
        columns={
            "anchor_no": "horse_no",
            "anchor_name": "horse_name",
            "anchor_pop": "pop",
            "anchor_odds": "odds",
            "anchor_quinella_model_rank": "rank_proxy",
        }
    )
    partner_horses = out[
        ["race_id", "partner_no", "partner_name", "partner_pop", "partner_odds", "partner_ai_rank"]
    ].rename(
        columns={
            "partner_no": "horse_no",
            "partner_name": "horse_name",
            "partner_pop": "pop",
            "partner_odds": "odds",
            "partner_ai_rank": "rank_proxy",
        }
    )
    horses = pd.concat([anchor_horses, partner_horses], ignore_index=True, sort=False)
    horses["pop_num"] = num(horses["pop"]).fillna(99)
    horses["odds_num"] = num(horses["odds"]).fillna(999)
    horses["rank_num"] = num(horses["rank_proxy"]).fillna(99)
    top_fav = (
        horses.sort_values(["race_id", "pop_num", "odds_num", "rank_num"])
        .drop_duplicates("race_id", keep="first")
        [["race_id", "horse_no", "horse_name", "pop_num", "odds_num"]]
        .rename(
            columns={
                "horse_no": "race_top_favorite_no",
                "horse_name": "race_top_favorite_name",
                "pop_num": "race_top_favorite_pop",
                "odds_num": "race_top_favorite_odds",
            }
        )
    )
    out = out.merge(top_fav, on="race_id", how="left")
    out["top_favorite_in_pair_flag"] = (
        num(out["anchor_no"]).eq(num(out["race_top_favorite_no"]))
        | num(out["partner_no"]).eq(num(out["race_top_favorite_no"]))
    ).astype(float)
    out["dominant_favorite_missing_flag"] = (
        out["top_favorite_in_pair_flag"].lt(1.0)
        & (
            num(out["race_top_favorite_pop"]).le(2)
            | num(out["race_top_favorite_odds"]).le(3.0)
        )
        & num(out.get("market_overlay_score"), out.index, 0.0).lt(0.86)
    ).astype(float)
    out["dominant_favorite_missing_strict_flag"] = (
        out["top_favorite_in_pair_flag"].lt(1.0)
        & (
            num(out["race_top_favorite_pop"]).le(2)
            | num(out["race_top_favorite_odds"]).le(3.0)
        )
    ).astype(float)
    return out


def with_allowed_sets(params: dict) -> dict:
    out = dict(params)
    allowed_all = {
        "Sapporo",
        "Hakodate",
        "Fukushima",
        "Niigata",
        "Tokyo",
        "Nakayama",
        "Chukyo",
        "Kyoto",
        "Hanshin",
        "Kokura",
        "Unknown",
    }
    out["venue_allowed"] = allowed_all if out["venue_policy"] == "all" else allowed_all - {"Hakodate"}
    out["going_allowed"] = (
        {"Good", "Yielding", "Soft", "Heavy", "Unknown"}
        if out["going_policy"] == "all"
        else {"Good", "Yielding", "Unknown"}
    )
    return out


def variant_prefilter(df: pd.DataFrame, params: dict, variant: str) -> pd.DataFrame:
    work = prefilter(df, params)
    if work.empty:
        return work
    if variant in {"partner_floor", "partner_floor_top_eval", "dynamic_ticket_strict"}:
        work = work[work["partner_ability_floor_fail_flag"].lt(1)].copy()
    if variant in {"top_eval_guard", "partner_floor_top_eval"}:
        work = work[work["top_eval_missing_risk_flag"].lt(1)].copy()
    if variant == "front_partner_floor":
        work = work[work["front_but_partner_weak_flag"].lt(1)].copy()
    if variant in {"favorite_guard", "partner_floor_favorite_guard"}:
        work = work[work["dominant_favorite_missing_flag"].lt(1)].copy()
    if variant in {"favorite_strict_guard", "partner_floor_favorite_strict_guard"}:
        work = work[work["dominant_favorite_missing_strict_flag"].lt(1)].copy()
    return work


def variant_representatives(df: pd.DataFrame, params: dict, variant: str) -> pd.DataFrame:
    work = variant_prefilter(df, params, variant)
    if work.empty:
        return work
    score = "strongest_pair_score"
    if variant == "dynamic_ticket_strict":
        work = work.copy()
        work["conversion_adjusted_score"] = (
            0.90 * num(work["strongest_pair_score"]).fillna(0.0)
            + 0.06 * work["partner_ability_floor_score"]
            + 0.04 * work["top_eval_in_pair_flag"]
        ).clip(0.0, 1.0)
        score = "conversion_adjusted_score"
    return (
        work.sort_values(
            ["race_id", score, "market_overlay_score", "pair_score"],
            ascending=[True, False, False, False],
        )
        .groupby("race_id", as_index=False)
        .head(1)
        .copy()
    )


def variant_threshold(train: pd.DataFrame, params: dict, variant: str) -> float:
    reps = variant_representatives(train, params, variant)
    if reps.empty:
        return float("inf")
    score_col = "conversion_adjusted_score" if variant == "dynamic_ticket_strict" else "strongest_pair_score"
    return float(reps[score_col].quantile(1.0 - params["coverage"]))


def variant_select_pairs(df: pd.DataFrame, params: dict, threshold: float, variant: str) -> pd.DataFrame:
    reps = variant_representatives(df, params, variant)
    if reps.empty:
        return reps
    score_col = "conversion_adjusted_score" if variant == "dynamic_ticket_strict" else "strongest_pair_score"
    return reps[reps[score_col].ge(threshold)].copy()


def tickets_from_pairs_variant(pairs: pd.DataFrame, params: dict, variant: str) -> pd.DataFrame:
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
    if variant in {"umaren_partner_floor", "dynamic_ticket_strict"}:
        umaren_mask &= (
            base["partner_ability_floor_score"].ge(0.58)
            & num(base["partner_place_score"]).ge(0.42)
            & num(base["partner_quinella_score"]).ge(0.46)
            & base["umaren_suitable_score"].ge(0.58)
        )
    if variant == "umaren_top_eval_guard":
        umaren_mask &= base["top_eval_in_pair_flag"].ge(1)
    if variant == "umaren_favorite_guard":
        umaren_mask &= base["dominant_favorite_missing_flag"].lt(1)
    if variant == "umaren_favorite_strict_guard":
        umaren_mask &= base["dominant_favorite_missing_strict_flag"].lt(1)
    if variant == "wide_only":
        umaren_mask &= False

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


def evaluate_variant(df: pd.DataFrame, params: dict, threshold: float, variant: str, label: str) -> tuple[dict, pd.DataFrame]:
    pairs = variant_select_pairs(df, params, threshold, variant)
    tickets = tickets_from_pairs_variant(pairs, params, variant)
    m = metrics(tickets, label)
    total = int(df["race_id"].nunique())
    m["candidate_races"] = total
    m["race_selection_rate"] = float(m["races"] / total) if total else 0.0
    return m, tickets


def walkforward_variant(df: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grids = policy_grid()
    train_rows: list[pd.DataFrame] = []
    wf_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    for test_year in [2025, 2026]:
        train = df[df["year"] < test_year].copy()
        test = df[df["year"] == test_year].copy()
        rows: list[dict] = []
        for i, params in enumerate(grids):
            threshold = variant_threshold(train, params, variant)
            m, _ = evaluate_variant(train, params, threshold, variant, f"{variant}_train_{test_year}_{i}")
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
        m, tickets = evaluate_variant(test, params, float(best["score_threshold"]), variant, f"{variant}_wf_test_{test_year}")
        m.update({k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}})
        m["test_year"] = test_year
        m["train_roi"] = float(best["roi"])
        m["train_races"] = int(best["races"])
        m["train_profit_yen"] = float(best["profit_yen"])
        m["train_race_selection_rate"] = float(best["race_selection_rate"])
        m["score_threshold"] = float(best["score_threshold"])
        m["variant"] = variant
        wf_rows.append(m)
        if not tickets.empty:
            tmp = tickets.copy()
            tmp["test_year"] = test_year
            tmp["selected_grid_id"] = int(best["grid_id"])
            tmp["variant"] = variant
            ticket_frames.append(tmp)
    return (
        pd.concat(train_rows, ignore_index=True, sort=False),
        pd.DataFrame(wf_rows),
        pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame(),
    )


def walkforward_variant_fast(df: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows: list[pd.DataFrame] = []
    wf_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    for test_year in [2025, 2026]:
        params = with_allowed_sets(FAST_PARAMS_BY_TEST_YEAR[test_year])
        train = df[df["year"] < test_year].copy()
        test = df[df["year"] == test_year].copy()
        threshold = variant_threshold(train, params, variant)
        train_m, _ = evaluate_variant(train, params, threshold, variant, f"{variant}_train_{test_year}_fixed")
        train_row = train_m | {k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}}
        train_row["test_year"] = test_year
        train_row["score_threshold"] = threshold
        train_row["selection_score"] = policy_score(train_m)
        train_rows.append(pd.DataFrame([train_row]))
        m, tickets = evaluate_variant(test, params, threshold, variant, f"{variant}_wf_test_{test_year}_fixed")
        m.update({k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}})
        m["test_year"] = test_year
        m["train_roi"] = float(train_m["roi"])
        m["train_races"] = int(train_m["races"])
        m["train_profit_yen"] = float(train_m["profit_yen"])
        m["train_race_selection_rate"] = float(train_m["race_selection_rate"])
        m["score_threshold"] = threshold
        m["variant"] = variant
        wf_rows.append(m)
        if not tickets.empty:
            tmp = tickets.copy()
            tmp["test_year"] = test_year
            tmp["selected_grid_id"] = -1
            tmp["variant"] = variant
            ticket_frames.append(tmp)
    return (
        pd.concat(train_rows, ignore_index=True, sort=False),
        pd.DataFrame(wf_rows),
        pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame(),
    )


def summarize_tickets(tickets: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame()
    out = (
        tickets.groupby(["variant", "ticket_type"], dropna=False)
        .agg(
            tickets=("race_id", "size"),
            races=("race_id", "nunique"),
            stake_yen=("stake_yen", "sum"),
            return_yen=("return_yen", "sum"),
            hits=("hit", "sum"),
        )
        .reset_index()
    )
    out["profit_yen"] = out["return_yen"] - out["stake_yen"]
    out["roi"] = np.where(out["stake_yen"] > 0, out["return_yen"] / out["stake_yen"], np.nan)
    out["hit_rate"] = out["hits"] / out["tickets"]
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full-grid", action="store_true", help="Run the original full grid search for every variant.")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    feature_cache = OUT / "pair_strength_universe_with_conversion_features.csv"
    if feature_cache.exists():
        universe = pd.read_csv(feature_cache, dtype={"race_id": str}, low_memory=False)
    else:
        universe = pd.read_csv(UNIVERSE, dtype={"race_id": str}, low_memory=False)
        universe = add_conversion_features(universe)
        universe.to_csv(feature_cache, index=False, encoding="utf-8-sig")

    variants = [
        "baseline",
        "partner_floor",
        "top_eval_guard",
        "partner_floor_top_eval",
        "front_partner_floor",
        "favorite_guard",
        "partner_floor_favorite_guard",
        "favorite_strict_guard",
        "partner_floor_favorite_strict_guard",
        "umaren_partner_floor",
        "umaren_top_eval_guard",
        "umaren_favorite_guard",
        "umaren_favorite_strict_guard",
        "wide_only",
        "dynamic_ticket_strict",
    ]
    train_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    ticket_frames: list[pd.DataFrame] = []
    for variant in variants:
        if args.full_grid:
            train, summary, tickets = walkforward_variant(universe, variant)
        else:
            train, summary, tickets = walkforward_variant_fast(universe, variant)
        train["variant"] = variant
        summary["variant"] = variant
        train_frames.append(train)
        summary_frames.append(summary)
        if not tickets.empty:
            ticket_frames.append(tickets)

    train_all = pd.concat(train_frames, ignore_index=True, sort=False)
    summary_all = pd.concat(summary_frames, ignore_index=True, sort=False)
    tickets_all = pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame()

    train_all.to_csv(OUT / "walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
    summary_all.to_csv(OUT / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    tickets_all.to_csv(OUT / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
    ticket_type_summary = summarize_tickets(tickets_all)
    ticket_type_summary.to_csv(OUT / "ticket_type_summary.csv", index=False, encoding="utf-8-sig")

    totals = (
        summary_all.groupby("variant", as_index=False)
        .agg(
            races=("races", "sum"),
            tickets=("tickets", "sum"),
            stake_yen=("stake_yen", "sum"),
            return_yen=("return_yen", "sum"),
            profit_yen=("profit_yen", "sum"),
            avg_race_hit_rate=("race_hit_rate", "mean"),
            max_drawdown_yen=("max_drawdown_yen", "min"),
            test_years=("test_year", "nunique"),
        )
    )
    totals["roi"] = np.where(totals["stake_yen"] > 0, totals["return_yen"] / totals["stake_yen"], np.nan)
    totals = totals.sort_values(["roi", "profit_yen"], ascending=[False, False])
    totals.to_csv(OUT / "summary_totals.csv", index=False, encoding="utf-8-sig")

    compact_cols = [
        "variant",
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
        "train_roi",
        "train_races",
    ]
    compact = summary_all[compact_cols].copy()
    compact.to_csv(OUT / "summary_compact.csv", index=False, encoding="utf-8-sig")
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_dir": str(OUT),
                "variants": variants,
                "totals": totals.to_dict(orient="records"),
                "compact": compact.to_dict(orient="records"),
                "ticket_type_summary": ticket_type_summary.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("TICKET CONVERSION FIX WALKFORWARD")
    print(compact.to_string(index=False))
    print("\nTOTALS")
    print(totals.to_string(index=False))
    print("\nTICKET TYPES")
    print(ticket_type_summary.to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
