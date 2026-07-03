from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_fixed_budget_ticket_portfolio import _existing_raw_csv, _load_payoffs
from scripts.evaluate_market_edge_pair_strategy import _attach_wide_payoffs, _load_wide_payoffs
from scripts.evaluate_rollover_strategy import _date_key, _race_sort_key
from scripts.evaluate_solid_axis_ticket_model import _build_pair_tickets, _pick_horse_no_col
from scripts.evaluate_ticket_strategies import _add_model_columns, _col
from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


def _base_anchor_cols(scored: pd.DataFrame, race_col: str) -> pd.DataFrame:
    horse_no_col = _pick_horse_no_col(scored)
    cols = [
        race_col,
        "horse_name_for_ticket",
        horse_no_col,
        "ai_rank",
        "pop_rank",
        "popularity_num",
        "odds_decimal",
        "rank_num",
        "target_top3",
        "ai_score",
        "ai_score_gap_to_second",
    ]
    out = scored[cols].rename(
        columns={
            horse_no_col: "horse_no",
            "horse_name_for_ticket": "horse",
            "odds_decimal": "odds",
            "rank_num": "finish",
        }
    )
    out[race_col] = out[race_col].astype(str)
    return out


def _single_candidate_tickets(scored: pd.DataFrame, horse_pay: pd.DataFrame, race_col: str) -> pd.DataFrame:
    base = _base_anchor_cols(scored, race_col)
    pay = horse_pay.copy()
    pay[race_col] = pay[race_col].astype(str)
    pay = pay.rename(columns={"rank_num": "finish"})
    base = base.merge(pay, on=[race_col, "finish"], how="left")

    frames = []
    odds_values = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    gap_values = [0.05, 0.10, 0.15, 0.20, 0.30]
    for max_odds in odds_values:
        for min_gap in gap_values:
            mask = (
                base["ai_rank"].eq(1)
                & pd.to_numeric(base["odds"], errors="coerce").le(max_odds)
                & pd.to_numeric(base["ai_score_gap_to_second"], errors="coerce").ge(min_gap)
            )
            anchor = base[mask].copy()
            if anchor.empty:
                continue
            anchor = (
                anchor.sort_values([race_col, "ai_score_gap_to_second", "ai_score"], ascending=[True, False, False])
                .groupby(race_col, as_index=False)
                .head(1)
            )
            for ticket_type, hit_expr, pay_col in [
                ("win", anchor["finish"].eq(1), "単勝配当"),
                ("place", anchor["finish"].le(3), "複勝配当"),
            ]:
                part = anchor.copy()
                part["strategy_name"] = f"{ticket_type}_axis_o{str(max_odds).replace('.', '_')}_g{str(min_gap).replace('.', '_')}"
                part["ticket_type"] = ticket_type
                part["hit"] = hit_expr.astype(bool)
                part["return_per100"] = pd.to_numeric(part[pay_col], errors="coerce").fillna(0.0).where(part["hit"], 0.0)
                frames.append(part)
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if out.empty:
        return out
    out["race_id"] = out[race_col].astype(str)
    out["sort_key"] = out["race_id"].map(_race_sort_key)
    out["date_key"] = out["race_id"].map(_date_key)
    return out


def _pair_candidate_tickets(scored: pd.DataFrame, payoffs: pd.DataFrame, wide_payoffs: pd.DataFrame, race_col: str) -> pd.DataFrame:
    pairs = _build_pair_tickets(scored, race_col)
    if pairs.empty:
        return pairs
    pairs = _attach_wide_payoffs(pairs, wide_payoffs, race_col)
    pay = payoffs[[race_col, "umaren_pay"]].copy()
    pay[race_col] = pay[race_col].astype(str)
    pairs = pairs.merge(pay, on=race_col, how="left")
    frames = []
    for ticket_type, hit_col, ret_col in [
        ("wide", "wide_hit", "wide_pay"),
        ("umaren", "umaren_hit", "umaren_pay"),
    ]:
        part = pairs.copy()
        part["strategy_name"] = part["strategy"] + "_" + ticket_type
        part["ticket_type"] = ticket_type
        part["hit"] = part[hit_col].astype(bool)
        part["return_per100"] = pd.to_numeric(part[ret_col], errors="coerce").fillna(0.0).where(part["hit"], 0.0)
        part["race_id"] = part[race_col].astype(str)
        part["sort_key"] = part["race_id"].map(_race_sort_key)
        part["date_key"] = part["race_id"].map(_date_key)
        frames.append(part)
    return pd.concat(frames, ignore_index=True, sort=False)


def _strategy_stats(tickets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in tickets.groupby("strategy_name"):
        hits = g[g["hit"]]
        rows.append(
            {
                "strategy_name": name,
                "ticket_type": g["ticket_type"].iloc[0],
                "tickets": int(len(g)),
                "races": int(g["race_id"].nunique()),
                "hit_rate": float(g["hit"].mean()),
                "avg_hit_return": float(hits["return_per100"].mean()) if len(hits) else 0.0,
                "median_hit_return": float(hits["return_per100"].median()) if len(hits) else 0.0,
                "roi_reference": float(g["return_per100"].sum() / (len(g) * 100.0)) if len(g) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _select_hit_first_policy(
    calib_stats: pd.DataFrame,
    *,
    min_tickets: int,
    min_hit_return: float,
    max_hit_return: float,
    target_hit_return: float,
) -> pd.DataFrame:
    stats = calib_stats[
        (calib_stats["tickets"] >= min_tickets)
        & (calib_stats["avg_hit_return"].between(min_hit_return, max_hit_return))
    ].copy()
    if stats.empty:
        stats = calib_stats[
            (calib_stats["tickets"] >= max(30, min_tickets // 2))
            & (calib_stats["avg_hit_return"].between(min_hit_return - 20, max_hit_return + 20))
        ].copy()
    stats["hit_first_score"] = (
        stats["hit_rate"] * 100.0
        - (stats["avg_hit_return"].sub(target_hit_return).abs() / 2.0)
        - (stats["median_hit_return"].sub(target_hit_return).abs() / 4.0)
    )
    return stats.sort_values(["hit_first_score", "hit_rate"], ascending=[False, False])


def _apply_policy(valid: pd.DataFrame, policy: pd.DataFrame, max_per_day: int | None) -> pd.DataFrame:
    selected = valid[valid["strategy_name"].isin(policy["strategy_name"])].copy()
    if selected.empty:
        return selected
    score = policy.set_index("strategy_name")["hit_first_score"].to_dict()
    hit_rate = policy.set_index("strategy_name")["hit_rate"].to_dict()
    avg_return = policy.set_index("strategy_name")["avg_hit_return"].to_dict()
    selected["policy_score"] = selected["strategy_name"].map(score)
    selected["calib_hit_rate"] = selected["strategy_name"].map(hit_rate)
    selected["calib_avg_hit_return"] = selected["strategy_name"].map(avg_return)
    selected = (
        selected.sort_values(
            ["race_id", "policy_score", "calib_hit_rate", "calib_avg_hit_return"],
            ascending=[True, False, False, True],
        )
        .groupby("race_id", as_index=False)
        .head(1)
        .sort_values("sort_key")
        .reset_index(drop=True)
    )
    if max_per_day is not None:
        selected = (
            selected.sort_values(["date_key", "policy_score", "calib_hit_rate"], ascending=[True, False, False])
            .groupby("date_key", as_index=False)
            .head(max_per_day)
            .sort_values("sort_key")
            .reset_index(drop=True)
        )
    return selected


def _flat_metrics(tickets: pd.DataFrame, label: str) -> dict[str, object]:
    if tickets.empty:
        return {"policy": label, "tickets": 0, "days": 0}
    hits = tickets[tickets["hit"]]
    return {
        "policy": label,
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "days": int(tickets["date_key"].nunique()),
        "races_per_day_avg": float(tickets.groupby("date_key")["race_id"].nunique().mean()),
        "races_per_day_median": float(tickets.groupby("date_key")["race_id"].nunique().median()),
        "races_per_day_max": int(tickets.groupby("date_key")["race_id"].nunique().max()),
        "hit_rate": float(tickets["hit"].mean()),
        "avg_hit_return": float(hits["return_per100"].mean()) if len(hits) else 0.0,
        "median_hit_return": float(hits["return_per100"].median()) if len(hits) else 0.0,
        "roi_reference": float(tickets["return_per100"].sum() / (len(tickets) * 100.0)),
    }


def _consecutive_metrics(tickets: pd.DataFrame, n_values: list[int]) -> pd.DataFrame:
    rows = []
    ordered = tickets.sort_values("sort_key").reset_index(drop=True)
    for n in n_values:
        sessions = []
        for start in range(len(ordered)):
            part = ordered.iloc[start : start + n]
            if len(part) < n:
                continue
            bankroll = 10000.0
            all_hit = True
            for _, row in part.iterrows():
                if not bool(row["hit"]):
                    all_hit = False
                    bankroll = 0.0
                    break
                bankroll = bankroll * float(row["return_per100"]) / 100.0
            sessions.append({"all_hit": all_hit, "final_bankroll": bankroll})
        if not sessions:
            continue
        df = pd.DataFrame(sessions)
        rows.append(
            {
                "legs": n,
                "sessions": int(len(df)),
                "all_hit_rate": float(df["all_hit"].mean()),
                "median_final_bankroll": float(df["final_bankroll"].median()),
                "mean_final_bankroll": float(df["final_bankroll"].mean()),
                "p90_final_bankroll": float(df["final_bankroll"].quantile(0.90)),
                "max_final_bankroll": float(df["final_bankroll"].max()),
            }
        )
    return pd.DataFrame(rows)


def _daily_distribution(tickets: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame()
    daily = (
        tickets.groupby("date_key", as_index=False)
        .agg(
            qualifying_races=("race_id", "nunique"),
            hit_rate=("hit", "mean"),
            avg_hit_return=("return_per100", lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0),
        )
        .sort_values("date_key")
    )
    return daily


def main() -> None:
    parser = argparse.ArgumentParser(description="Hit-first rollover model: target 1.5x-ish payouts, maximize hit rate and race selection.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default="models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--test-csv", default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv")
    parser.add_argument("--raw-csv", default="date/raw/蜈ｨ遶ｶ襍ｰ鬥ｬ謌千ｸｾ.csv")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/hit_first_rollover_model_v1")
    parser.add_argument("--min-hit-return", type=float, default=135.0)
    parser.add_argument("--max-hit-return", type=float, default=170.0)
    parser.add_argument("--target-hit-return", type=float, default=150.0)
    parser.add_argument("--min-tickets", type=int, default=80)
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    encoding = config["data"].get("encoding", "cp932")
    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = _add_model_columns(df, project_path(args.model), race_col)
    scored[race_col] = scored[race_col].astype(str)

    raw_csv = _existing_raw_csv(args.raw_csv)
    payoffs, horse_pay = _load_payoffs(raw_csv, race_col, encoding)
    single = _single_candidate_tickets(scored, horse_pay, race_col)
    pair = _pair_candidate_tickets(scored, payoffs, _load_wide_payoffs(args.wide_payoff_csv, race_col), race_col)
    tickets = pd.concat([single, pair], ignore_index=True, sort=False).sort_values("sort_key").reset_index(drop=True)

    split_key = tickets["date_key"].quantile(0.5)
    calib = tickets[tickets["date_key"].le(split_key)].copy()
    valid = tickets[tickets["date_key"].gt(split_key)].copy()
    calib_stats = _strategy_stats(calib)
    valid_stats = _strategy_stats(valid)
    policy = _select_hit_first_policy(
        calib_stats,
        min_tickets=args.min_tickets,
        min_hit_return=args.min_hit_return,
        max_hit_return=args.max_hit_return,
        target_hit_return=args.target_hit_return,
    )

    selected_all = _apply_policy(valid, policy, max_per_day=None)
    selected_max1 = _apply_policy(valid, policy, max_per_day=1)
    selected_max2 = _apply_policy(valid, policy, max_per_day=2)

    flat = pd.DataFrame(
        [
            _flat_metrics(selected_all, "hit_first_all_qualifying"),
            _flat_metrics(selected_max1, "hit_first_max1_per_day"),
            _flat_metrics(selected_max2, "hit_first_max2_per_day"),
        ]
    )
    consecutive = []
    for label, selected in [
        ("all_qualifying", selected_all),
        ("max1_per_day", selected_max1),
        ("max2_per_day", selected_max2),
    ]:
        cm = _consecutive_metrics(selected, [2, 3, 4, 5, 6])
        if not cm.empty:
            cm["policy"] = label
            consecutive.append(cm)
    consecutive_df = pd.concat(consecutive, ignore_index=True, sort=False) if consecutive else pd.DataFrame()
    daily = _daily_distribution(selected_all)

    out_dir = ensure_dir(project_path(args.output_dir))
    calib_stats.to_csv(out_dir / "calibration_strategy_stats.csv", index=False, encoding="utf-8-sig")
    valid_stats.to_csv(out_dir / "validation_strategy_stats.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(out_dir / "selected_hit_first_policy.csv", index=False, encoding="utf-8-sig")
    selected_all.to_csv(out_dir / "hit_first_selected_tickets_all.csv", index=False, encoding="utf-8-sig")
    selected_max1.to_csv(out_dir / "hit_first_selected_tickets_max1_per_day.csv", index=False, encoding="utf-8-sig")
    flat.to_csv(out_dir / "hit_first_flat_summary.csv", index=False, encoding="utf-8-sig")
    consecutive_df.to_csv(out_dir / "hit_first_consecutive_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out_dir / "hit_first_daily_distribution.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "selection_target": {
            "min_hit_return": args.min_hit_return,
            "max_hit_return": args.max_hit_return,
            "target_hit_return": args.target_hit_return,
            "primary_metric": "hit_rate",
            "roi_role": "reference_only",
        },
        "selected_policy_top": policy.head(20).to_dict(orient="records"),
        "validation_flat_summary": flat.to_dict(orient="records"),
        "consecutive_summary": consecutive_df.to_dict(orient="records"),
        "daily_occurrence_summary": {
            "days": int(daily.shape[0]) if not daily.empty else 0,
            "avg_per_day": float(daily["qualifying_races"].mean()) if not daily.empty else 0.0,
            "median_per_day": float(daily["qualifying_races"].median()) if not daily.empty else 0.0,
            "max_per_day": int(daily["qualifying_races"].max()) if not daily.empty else 0,
            "zero_day_rate_in_observed_days": 0.0,
        },
        "notes": [
            "ROI is included only as a reference, not as the selection objective.",
            "Historical wide pre-race odds are approximated via realized hit payoff; live use should use real-time wide odds.",
            "Each selected race keeps only the highest hit-first score ticket.",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
