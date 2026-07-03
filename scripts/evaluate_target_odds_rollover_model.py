from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_rollover_strategy import _date_key, _race_sort_key
from scripts.evaluate_solid_axis_ticket_model import (
    _build_pair_tickets,
    _metrics_pair,
    _metrics_single,
    _single_from_candidates,
)
from scripts.evaluate_fixed_budget_ticket_portfolio import _existing_raw_csv, _load_payoffs
from scripts.evaluate_market_edge_pair_strategy import _attach_wide_payoffs, _load_wide_payoffs
from scripts.evaluate_ticket_strategies import _add_model_columns
from scripts.optimize_rollover_hit_model import _prepare_fast_sources, _simulate_plan_fast
from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


def _race_id_col(df: pd.DataFrame) -> str:
    for col in ["レースID(新/馬番無)", "race_id"]:
        if col in df.columns:
            return col
    return df.columns[0]


def _single_tickets(singles: pd.DataFrame) -> pd.DataFrame:
    race_col = _race_id_col(singles)
    out = singles.copy()
    out["race_id"] = out[race_col].astype(str)
    out["strategy_name"] = out["solid_strategy"]
    out["ticket_name"] = out["solid_strategy"]
    out["hit"] = out["hit"].astype(bool)
    out["return_per100"] = pd.to_numeric(out["return_per100"], errors="coerce").fillna(0.0)
    out["sort_key"] = out["race_id"].map(_race_sort_key)
    out["date_key"] = out["race_id"].map(_date_key)
    return out[
        [
            "race_id",
            "sort_key",
            "date_key",
            "ticket_name",
            "strategy_name",
            "ticket_type",
            "hit",
            "return_per100",
            "a_horse",
            "a_odds",
        ]
    ]


def _pair_tickets_for_type(pair_tickets: pd.DataFrame, race_col: str) -> pd.DataFrame:
    frames = []
    for ticket_type, hit_col, ret_col in [
        ("wide", "wide_hit", "wide_pay"),
        ("umaren", "umaren_hit", "umaren_pay"),
    ]:
        frame = pair_tickets.copy()
        frame["race_id"] = frame[race_col].astype(str)
        frame["ticket_type"] = ticket_type
        frame["strategy_name"] = frame["strategy"] + "_" + ticket_type
        frame["ticket_name"] = frame["strategy_name"]
        frame["hit"] = frame[hit_col].astype(bool)
        frame["return_per100"] = pd.to_numeric(frame[ret_col], errors="coerce").fillna(0.0).where(frame["hit"], 0.0)
        frame["sort_key"] = frame["race_id"].map(_race_sort_key)
        frame["date_key"] = frame["race_id"].map(_date_key)
        frames.append(
            frame[
                [
                    "race_id",
                    "sort_key",
                    "date_key",
                    "ticket_name",
                    "strategy_name",
                    "ticket_type",
                    "hit",
                    "return_per100",
                    "a_horse",
                    "b_horse",
                    "a_odds",
                    "b_odds",
                ]
            ]
        )
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
                "roi": float(g["return_per100"].sum() / (len(g) * 100.0)),
                "avg_hit_return": float(hits["return_per100"].mean()) if len(hits) else 0.0,
                "median_hit_return": float(hits["return_per100"].median()) if len(hits) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _select_policy_stats(
    calib_stats: pd.DataFrame,
    *,
    min_tickets: int,
    min_hit_rate: float,
    min_avg_return: float,
    max_avg_return: float,
    target_return: float,
) -> pd.DataFrame:
    stats = calib_stats[
        (calib_stats["tickets"] >= min_tickets)
        & (calib_stats["hit_rate"] >= min_hit_rate)
        & (calib_stats["avg_hit_return"].between(min_avg_return, max_avg_return))
    ].copy()
    if stats.empty:
        stats = calib_stats[
            (calib_stats["tickets"] >= min_tickets)
            & (calib_stats["hit_rate"] >= min_hit_rate)
            & (calib_stats["avg_hit_return"].between(110, 220))
        ].copy()
    stats["target_odds_score"] = (
        stats["hit_rate"] * 3.0
        + stats["roi"].clip(upper=1.3)
        - (stats["avg_hit_return"].sub(target_return).abs() / 100.0)
    )
    return stats.sort_values(["target_odds_score", "hit_rate", "roi"], ascending=[False, False, False])


def _build_policy_tickets(tickets: pd.DataFrame, policy_stats: pd.DataFrame, max_per_race: int) -> pd.DataFrame:
    selected = tickets[tickets["strategy_name"].isin(policy_stats["strategy_name"])].copy()
    if selected.empty:
        return selected
    score_map = policy_stats.set_index("strategy_name")["target_odds_score"].to_dict()
    hit_map = policy_stats.set_index("strategy_name")["hit_rate"].to_dict()
    return_map = policy_stats.set_index("strategy_name")["avg_hit_return"].to_dict()
    selected["policy_score"] = selected["strategy_name"].map(score_map)
    selected["calib_hit_rate"] = selected["strategy_name"].map(hit_map)
    selected["calib_avg_hit_return"] = selected["strategy_name"].map(return_map)
    selected = selected.sort_values(
        ["race_id", "policy_score", "calib_hit_rate", "calib_avg_hit_return"],
        ascending=[True, False, False, False],
    )
    return selected.groupby("race_id", as_index=False).head(max_per_race).reset_index(drop=True)


def _flat_metrics(tickets: pd.DataFrame, label: str) -> dict[str, object]:
    if tickets.empty:
        return {"policy": label, "tickets": 0, "races": 0}
    stake = len(tickets) * 100.0
    ret = tickets["return_per100"].sum()
    profit = tickets["return_per100"].fillna(0.0) - 100.0
    curve = profit.cumsum()
    dd = curve.cummax() - curve
    return {
        "policy": label,
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "hit_rate": float(tickets["hit"].mean()),
        "roi": float(ret / stake),
        "avg_hit_return": float(tickets.loc[tickets["hit"], "return_per100"].mean()) if int(tickets["hit"].sum()) else 0.0,
        "median_hit_return": float(tickets.loc[tickets["hit"], "return_per100"].median()) if int(tickets["hit"].sum()) else 0.0,
        "profit_flat100": float(ret - stake),
        "max_drawdown_flat100": float(dd.max()) if not dd.empty else 0.0,
    }


def _rollover_sources(policy_tickets: pd.DataFrame, min_tickets: int) -> dict[str, pd.DataFrame]:
    sources = {}
    for name, g in policy_tickets.groupby("strategy_name"):
        if len(g) >= min_tickets:
            frame = g.copy()
            frame["ticket_name"] = name
            sources[name] = frame.sort_values("sort_key")
    if len(policy_tickets) >= min_tickets:
        pooled = policy_tickets.sort_values(["race_id", "policy_score"], ascending=[True, False]).groupby("race_id", as_index=False).head(1)
        pooled = pooled.copy()
        pooled["ticket_name"] = "target_odds_best_per_race"
        sources["target_odds_best_per_race"] = pooled.sort_values("sort_key")
    return sources


def _rollover_eval(sources: dict[str, pd.DataFrame], targets: list[int]) -> pd.DataFrame:
    fast = _prepare_fast_sources(sources)
    plans = {}
    for name in sources:
        plans[f"{name}_x2"] = [name, name]
        plans[f"{name}_x3"] = [name, name, name]
    rows = []
    for target in targets:
        for plan_name, plan in plans.items():
            for fraction in [1.0, 0.75, 0.5, 0.33]:
                summary, _ = _simulate_plan_fast(
                    fast,
                    plan_name,
                    plan,
                    initial_bankroll=10000,
                    stake_fraction=fraction,
                    target_bankroll=target,
                )
                if int(summary["sessions"]) == 0:
                    continue
                summary["target_bankroll"] = target
                summary["legs"] = " > ".join(plan)
                rows.append(summary)
    return pd.DataFrame(rows).sort_values(
        ["target_bankroll", "target_rate", "complete_rate", "mean_final_bankroll"],
        ascending=[True, False, False, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate target-odds rollover model using win/place/wide/umaren tickets.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default="models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--test-csv", default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv")
    parser.add_argument("--candidate-csv", default="outputs/analysis/fixed_budget_ticket_portfolio_10000/candidate_tickets.csv")
    parser.add_argument("--raw-csv", default="date/raw/蜈ｨ遶ｶ襍ｰ鬥ｬ謌千ｸｾ.csv")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/target_odds_rollover_model_v1")
    parser.add_argument("--target-return", type=float, default=150.0)
    parser.add_argument("--min-avg-return", type=float, default=130.0)
    parser.add_argument("--max-avg-return", type=float, default=180.0)
    parser.add_argument("--min-hit-rate", type=float, default=0.55)
    parser.add_argument("--min-tickets", type=int, default=80)
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    encoding = config["data"].get("encoding", "cp932")
    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = _add_model_columns(df, project_path(args.model), race_col)
    scored[race_col] = scored[race_col].astype(str)
    raw_csv = _existing_raw_csv(args.raw_csv)
    payoffs, _ = _load_payoffs(raw_csv, race_col, encoding)

    singles = _single_from_candidates(project_path(args.candidate_csv))
    single_tickets = _single_tickets(singles)
    pair_tickets = _build_pair_tickets(scored, race_col)
    pair_tickets = _attach_wide_payoffs(pair_tickets, _load_wide_payoffs(args.wide_payoff_csv, race_col), race_col)
    pay = payoffs[[race_col, "umaren_pay"]].copy()
    pay[race_col] = pay[race_col].astype(str)
    pair_tickets = pair_tickets.merge(pay, on=race_col, how="left")
    pair_type_tickets = _pair_tickets_for_type(pair_tickets, race_col)

    all_tickets = pd.concat([single_tickets, pair_type_tickets], ignore_index=True, sort=False)
    all_tickets = all_tickets.sort_values("sort_key").reset_index(drop=True)

    split_key = all_tickets["date_key"].quantile(0.5)
    calib = all_tickets[all_tickets["date_key"].le(split_key)].copy()
    valid = all_tickets[all_tickets["date_key"].gt(split_key)].copy()
    calib_stats = _strategy_stats(calib)
    valid_stats = _strategy_stats(valid)

    policy_stats = _select_policy_stats(
        calib_stats,
        min_tickets=args.min_tickets,
        min_hit_rate=args.min_hit_rate,
        min_avg_return=args.min_avg_return,
        max_avg_return=args.max_avg_return,
        target_return=args.target_return,
    )
    valid_policy_tickets = _build_policy_tickets(valid, policy_stats, max_per_race=1)
    flat_summary = pd.DataFrame(
        [
            _flat_metrics(valid_policy_tickets, "target_odds_best_per_race"),
            *[
                _flat_metrics(valid_policy_tickets[valid_policy_tickets["strategy_name"].eq(name)], name)
                for name in policy_stats["strategy_name"].head(12)
            ],
        ]
    ).sort_values(["hit_rate", "roi"], ascending=[False, False])

    sources = _rollover_sources(valid_policy_tickets, min_tickets=max(30, args.min_tickets // 2))
    rollover = _rollover_eval(sources, [15000, 20000, 50000, 100000]) if sources else pd.DataFrame()

    out_dir = ensure_dir(project_path(args.output_dir))
    calib_stats.to_csv(out_dir / "calibration_strategy_stats.csv", index=False, encoding="utf-8-sig")
    valid_stats.to_csv(out_dir / "validation_strategy_stats.csv", index=False, encoding="utf-8-sig")
    policy_stats.to_csv(out_dir / "selected_target_odds_policy_stats.csv", index=False, encoding="utf-8-sig")
    valid_policy_tickets.to_csv(out_dir / "target_odds_policy_tickets.csv", index=False, encoding="utf-8-sig")
    flat_summary.to_csv(out_dir / "target_odds_flat_summary.csv", index=False, encoding="utf-8-sig")
    if not rollover.empty:
        rollover.to_csv(out_dir / "target_odds_rollover_summary.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "target_return_per100": args.target_return,
        "calibration_selected_strategies": policy_stats.head(20).to_dict(orient="records"),
        "validation_flat_summary": flat_summary.head(20).to_dict(orient="records"),
        "rollover_top_by_target": {
            str(target): rollover[rollover["target_bankroll"].eq(target)].head(8).to_dict(orient="records")
            for target in [15000, 20000, 50000, 100000]
        }
        if not rollover.empty
        else {},
        "interpretation": {
            "model": "Select solid win/place/wide tickets whose calibrated hit return is near 1.5x.",
            "live_use": "Replace calibrated average return with real-time win/place/wide odds at decision time.",
            "warning": "Historical wide odds before the race are approximated by strategy-level hit payoff here.",
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
