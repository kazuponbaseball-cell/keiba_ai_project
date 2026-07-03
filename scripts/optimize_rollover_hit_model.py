from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_rollover_strategy import _load_ticket_universe
from src.utils.paths import ensure_dir, project_path


def _stats(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"tickets": 0, "hit_rate": 0.0, "roi": 0.0, "avg_hit_return": 0.0}
    hits = df[df["hit"]]
    return {
        "tickets": int(len(df)),
        "hit_rate": float(df["hit"].mean()),
        "roi": float(df["return_per100"].sum() / (len(df) * 100.0)),
        "avg_hit_return": float(hits["return_per100"].mean()) if len(hits) else 0.0,
    }


def _copy_variant(df: pd.DataFrame, name: str, mask: pd.Series) -> pd.DataFrame:
    out = df[mask].copy()
    if not out.empty:
        out["ticket_name"] = name
    return out


def _build_variants(base: dict[str, pd.DataFrame], min_tickets: int) -> dict[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {}
    variants.update(base)

    for name in ["place_clear_head", "place_core_anchor", "win_clear_head"]:
        if name not in base:
            continue
        df = base[name]
        odds = pd.to_numeric(df.get("a_odds"), errors="coerce")
        pop = pd.to_numeric(df.get("a_popularity"), errors="coerce")
        specs = [
            (f"{name}_odds_le_1_8", odds.le(1.8)),
            (f"{name}_odds_le_2_2", odds.le(2.2)),
            (f"{name}_odds_le_3_0", odds.le(3.0)),
            (f"{name}_pop_le_2", pop.le(2)),
            (f"{name}_pop_le_3", pop.le(3)),
        ]
        for variant_name, mask in specs:
            part = _copy_variant(df, variant_name, mask.fillna(False))
            if len(part) >= min_tickets:
                variants[variant_name] = part

    for name in ["wide_strong_value", "wide_value_top2", "wide_late_value_top2"]:
        if name not in base:
            continue
        df = base[name]
        b_odds = pd.to_numeric(df.get("b_odds"), errors="coerce")
        b_pop = pd.to_numeric(df.get("b_popularity"), errors="coerce")
        specs = [
            (f"{name}_b_odds_10_50", b_odds.ge(10) & b_odds.lt(50)),
            (f"{name}_b_odds_20_100", b_odds.ge(20) & b_odds.lt(100)),
            (f"{name}_b_pop_5_9", b_pop.ge(5) & b_pop.le(9)),
            (f"{name}_b_pop_7plus", b_pop.ge(7)),
            (f"{name}_b_pop_10plus", b_pop.ge(10)),
        ]
        for variant_name, mask in specs:
            part = _copy_variant(df, variant_name, mask.fillna(False))
            if len(part) >= min_tickets:
                variants[variant_name] = part

    return variants


def _source_summary(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in sources.items():
        row = {"ticket_name": name}
        row.update(_stats(df))
        row["rollover_leg_score"] = (
            row["hit_rate"] * 2.0
            + max(row["roi"] - 1.0, 0.0) * 1.5
            + min(row["avg_hit_return"] / 1000.0, 2.5) * 0.4
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rollover_leg_score", "hit_rate"], ascending=[False, False])


def _candidate_plans(source_stats: pd.DataFrame) -> dict[str, list[str]]:
    stats = source_stats.set_index("ticket_name")
    survival = source_stats[(source_stats["hit_rate"] >= 0.68) & (source_stats["tickets"] >= 100)]["ticket_name"].tolist()
    bridge = source_stats[(source_stats["hit_rate"] >= 0.35) & (source_stats["tickets"] >= 100)]["ticket_name"].tolist()
    growth = source_stats[(source_stats["roi"] >= 1.05) & (source_stats["tickets"] >= 80)]["ticket_name"].tolist()

    plans: dict[str, list[str]] = {}
    for a, b in product(survival[:4], growth[:5]):
        plans[f"survival_then_growth__{a}__{b}"] = [a, b]
    for a, b, c in product(survival[:3], bridge[:4], growth[:5]):
        if a == b and b == c:
            continue
        plans[f"survival_bridge_growth__{a}__{b}__{c}"] = [a, b, c]
    for a, b, c in product(survival[:3], survival[:3], growth[:5]):
        plans[f"survival_x2_growth__{a}__{b}__{c}"] = [a, b, c]

    # Add a few conservative baselines so the optimizer is not forced into longshot-only plans.
    for a in survival[:4]:
        plans[f"survival_x3__{a}"] = [a, a, a]

    # Drop plans whose first leg is not actually a high survival leg.
    return {name: plan for name, plan in plans.items() if stats.loc[plan[0], "hit_rate"] >= 0.68}


def _model_score(row: dict[str, object], *, target_bankroll: int) -> float:
    target_rate = float(row.get("target_rate", 0.0))
    complete_rate = float(row.get("complete_rate", 0.0))
    median_bankroll = float(row.get("median_final_bankroll", 0.0))
    mean_bankroll = float(row.get("mean_final_bankroll", 0.0))
    max_dd = float(row.get("max_drawdown_if_sequential", 0.0))
    worst = abs(float(row.get("worst_profit", 0.0)))
    return (
        target_rate * 100.0
        + complete_rate * 20.0
        + min(median_bankroll / 10000.0, 5.0) * 3.0
        + min(mean_bankroll / target_bankroll, 2.0) * 5.0
        - min(max_dd / 1000000.0, 5.0) * 2.5
        - min(worst / 10000.0, 1.0) * 1.0
    )


def _stake_amount(bankroll: float, fraction: float, min_unit: int = 100) -> int:
    return int((bankroll * fraction) // min_unit) * min_unit


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    curve = values.cumsum()
    return float((curve.cummax() - curve).max())


def _prepare_fast_sources(sources: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    fast: dict[str, dict[str, object]] = {}
    for name, df in sources.items():
        ordered = df.sort_values("sort_key").reset_index(drop=True)
        fast[name] = {
            "df": ordered,
            "keys": ordered["sort_key"].to_numpy(dtype=np.int64),
        }
    return fast


def _simulate_plan_fast(
    fast_sources: dict[str, dict[str, object]],
    plan_name: str,
    plan: list[str],
    *,
    initial_bankroll: int,
    stake_fraction: float,
    target_bankroll: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    first_df = fast_sources[plan[0]]["df"]
    results: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []

    for start_idx in range(len(first_df)):
        bankroll = float(initial_bankroll)
        last_sort = -1
        legs: list[dict[str, object]] = []
        for leg_no, source_name in enumerate(plan, start=1):
            bundle = fast_sources[source_name]
            df = bundle["df"]
            keys = bundle["keys"]
            if leg_no == 1:
                if start_idx >= len(df):
                    break
                row = df.iloc[start_idx]
                if int(row["sort_key"]) <= last_sort:
                    break
            else:
                pos = int(np.searchsorted(keys, last_sort, side="right"))
                if pos >= len(df):
                    break
                row = df.iloc[pos]
            stake = _stake_amount(bankroll, stake_fraction)
            if stake < 100:
                break
            bankroll -= stake
            ret = float(row["return_per100"]) * stake / 100.0 if bool(row["hit"]) else 0.0
            bankroll += ret
            legs.append(
                {
                    "leg": leg_no,
                    "ticket_name": source_name,
                    "race_id": row["race_id"],
                    "date_key": int(row["date_key"]),
                    "stake": stake,
                    "hit": bool(row["hit"]),
                    "return_yen": ret,
                    "bankroll_after": bankroll,
                    "return_per100": float(row["return_per100"]),
                }
            )
            last_sort = int(row["sort_key"])
            if not bool(row["hit"]) or bankroll >= target_bankroll:
                break
        if not legs:
            continue
        sim = {
            "plan": plan_name,
            "stake_fraction": stake_fraction,
            "start_race_id": legs[0]["race_id"],
            "start_date_key": legs[0]["date_key"],
            "legs_run": len(legs),
            "all_legs_hit": len(legs) == len(plan) and all(leg["hit"] for leg in legs),
            "reached_target": bankroll >= target_bankroll,
            "final_bankroll": bankroll,
            "profit": bankroll - initial_bankroll,
        }
        results.append(sim)
        for leg in legs:
            leg.update({"plan": plan_name, "start_race_id": legs[0]["race_id"], "stake_fraction": stake_fraction})
            leg_rows.append(leg)

    result_df = pd.DataFrame(results)
    leg_df = pd.DataFrame(leg_rows)
    if result_df.empty:
        return {
            "plan": plan_name,
            "stake_fraction": stake_fraction,
            "sessions": 0,
            "target_rate": 0.0,
            "complete_rate": 0.0,
            "median_final_bankroll": 0.0,
            "mean_final_bankroll": 0.0,
            "p90_final_bankroll": 0.0,
            "p99_final_bankroll": 0.0,
            "max_final_bankroll": 0.0,
            "avg_profit_per_session": 0.0,
            "worst_profit": 0.0,
            "max_drawdown_if_sequential": 0.0,
        }, leg_df

    return {
        "plan": plan_name,
        "stake_fraction": stake_fraction,
        "sessions": int(len(result_df)),
        "avg_legs_run": float(result_df["legs_run"].mean()),
        "target_rate": float(result_df["reached_target"].mean()),
        "complete_rate": float(result_df["all_legs_hit"].mean()),
        "median_final_bankroll": float(result_df["final_bankroll"].median()),
        "mean_final_bankroll": float(result_df["final_bankroll"].mean()),
        "p90_final_bankroll": float(result_df["final_bankroll"].quantile(0.90)),
        "p99_final_bankroll": float(result_df["final_bankroll"].quantile(0.99)),
        "max_final_bankroll": float(result_df["final_bankroll"].max()),
        "avg_profit_per_session": float(result_df["profit"].mean()),
        "worst_profit": float(result_df["profit"].min()),
        "max_drawdown_if_sequential": _max_drawdown(result_df.sort_values("start_date_key")["profit"]),
    }, leg_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a hit-rate-first rollover model.")
    parser.add_argument("--portfolio-csv", default="outputs/analysis/fixed_budget_ticket_portfolio_10000/candidate_tickets.csv")
    parser.add_argument("--wide-csv", default="outputs/analysis/roi_segments_walkforward_v1/wide_pair_tickets_enriched.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/rollover_hit_model_v1")
    parser.add_argument("--initial-bankroll", type=int, default=10000)
    parser.add_argument("--target-bankroll", type=int, default=100000)
    parser.add_argument("--min-tickets", type=int, default=80)
    args = parser.parse_args()

    base = _load_ticket_universe(project_path(args.portfolio_csv), project_path(args.wide_csv))
    sources = _build_variants(base, args.min_tickets)
    source_stats = _source_summary(sources)
    plans = _candidate_plans(source_stats)
    fast_sources = _prepare_fast_sources(sources)

    summaries = []
    leg_frames = []
    for plan_name, plan in plans.items():
        for stake_fraction in [1.0, 0.75, 0.5, 0.33]:
            summary, legs = _simulate_plan_fast(
                fast_sources,
                plan_name,
                plan,
                initial_bankroll=args.initial_bankroll,
                stake_fraction=stake_fraction,
                target_bankroll=args.target_bankroll,
            )
            if int(summary["sessions"]) == 0:
                continue
            summary["model_score"] = _model_score(summary, target_bankroll=args.target_bankroll)
            summary["legs"] = " > ".join(plan)
            summaries.append(summary)
            if not legs.empty:
                legs["model_plan"] = plan_name
                legs["model_legs"] = " > ".join(plan)
                leg_frames.append(legs)

    out_dir = ensure_dir(project_path(args.output_dir))
    source_stats.to_csv(out_dir / "rollover_source_variants.csv", index=False, encoding="utf-8-sig")
    plan_df = pd.DataFrame(summaries).sort_values(["model_score", "target_rate", "median_final_bankroll"], ascending=[False, False, False])
    plan_df.to_csv(out_dir / "rollover_hit_model_plans.csv", index=False, encoding="utf-8-sig")
    if leg_frames:
        pd.concat(leg_frames, ignore_index=True, sort=False).to_csv(out_dir / "rollover_hit_model_legs.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "initial_bankroll": args.initial_bankroll,
        "target_bankroll": args.target_bankroll,
        "selected_model": plan_df.head(1).to_dict(orient="records"),
        "top10_models": plan_df.head(10).to_dict(orient="records"),
        "top_source_variants": source_stats.head(20).to_dict(orient="records"),
        "interpretation": {
            "primary_objective": "Hit-rate-first rollover with enough payout power to reach the target.",
            "recommended_live_shape": "Use a survival first leg, then a balanced/growth leg. Avoid pure low-hit growth as the first leg.",
            "caution": "This is optimized on historical candidates. Live use needs odds-at-decision-time and pre-race-only filters.",
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
