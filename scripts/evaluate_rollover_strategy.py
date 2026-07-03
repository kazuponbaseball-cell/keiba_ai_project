from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


RACE_COL_CANDIDATES = [
    "race_id",
    "race_id_for_sort",
    "レースID(新/馬番無)",
]


def _race_col(df: pd.DataFrame) -> str:
    for col in RACE_COL_CANDIDATES:
        if col in df.columns:
            return col
    return df.columns[0]


def _race_sort_key(race_id: object) -> int:
    text = str(race_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return 0
    return int(digits[:16])


def _date_key(race_id: object) -> int:
    text = str(race_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits[:8]) if len(digits) >= 8 else 0


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    curve = values.cumsum()
    return float((curve.cummax() - curve).max())


def _normalize_ticket_frame(
    df: pd.DataFrame,
    *,
    race_col: str,
    ticket_name: str,
    ticket_type: str,
    horse_cols: list[str],
    hit_col: str,
    return_col: str,
    filter_mask: pd.Series | None = None,
    sort_cols: list[str] | None = None,
    ascending: list[bool] | None = None,
) -> pd.DataFrame:
    work = df.copy()
    if filter_mask is not None:
        work = work[filter_mask].copy()
    if work.empty:
        return pd.DataFrame()
    work["race_id"] = work[race_col].astype(str)
    work["sort_key"] = work["race_id"].map(_race_sort_key)
    work["date_key"] = work["race_id"].map(_date_key)
    work["ticket_name"] = ticket_name
    work["ticket_type"] = ticket_type
    work["hit"] = work[hit_col].fillna(False).astype(bool)
    work["return_per100"] = pd.to_numeric(work[return_col], errors="coerce").fillna(0.0)
    for col in horse_cols:
        if col not in work.columns:
            work[col] = ""
    keep = [
        "race_id",
        "sort_key",
        "date_key",
        "ticket_name",
        "ticket_type",
        "hit",
        "return_per100",
    ] + horse_cols
    if sort_cols:
        work = work.sort_values(["sort_key"] + sort_cols, ascending=[True] + (ascending or [True] * len(sort_cols)))
    else:
        work = work.sort_values(["sort_key"])
    return work[keep].groupby("race_id", as_index=False).head(1).sort_values("sort_key").reset_index(drop=True)


def _load_ticket_universe(portfolio_csv: Path, wide_csv: Path) -> dict[str, pd.DataFrame]:
    ticket_sets: dict[str, pd.DataFrame] = {}

    portfolio = pd.read_csv(portfolio_csv, low_memory=False)
    pc = _race_col(portfolio)
    for strategy in ["place_clear_head", "place_core_anchor", "win_clear_head", "win_value_top3_gap"]:
        if "strategy" not in portfolio.columns:
            continue
        part = _normalize_ticket_frame(
            portfolio,
            race_col=pc,
            ticket_name=strategy,
            ticket_type=str(portfolio.loc[portfolio["strategy"].eq(strategy), "ticket_type"].dropna().head(1).iloc[0])
            if portfolio["strategy"].eq(strategy).any()
            else strategy,
            horse_cols=["a_horse", "a_ai_rank", "a_popularity", "a_odds"],
            hit_col="hit",
            return_col="return_per100",
            filter_mask=portfolio["strategy"].eq(strategy),
            sort_cols=["a_ai_rank", "a_popularity", "a_odds"],
            ascending=[True, True, False],
        )
        if not part.empty:
            ticket_sets[strategy] = part

    wide = pd.read_csv(wide_csv, low_memory=False)
    wc = _race_col(wide)
    wide_specs = {
        "wide_strong_value": wide["strategy"].eq("anchor_x_strong_value_top1"),
        "wide_value_top2": wide["strategy"].eq("anchor_x_value_top2"),
        "wide_late_value_top2": wide["strategy"].eq("anchor_x_late_value_top2") if "strategy" in wide.columns else pd.Series(False, index=wide.index),
    }
    if {"partner_style", "partner_odds_band"}.issubset(wide.columns):
        wide_specs["wide_front_odds20_100_oracle"] = wide["partner_style"].eq("front") & wide["partner_odds_band"].isin(["20_50", "50_100"])
    if {"partner_style", "partner_pop_band"}.issubset(wide.columns):
        wide_specs["wide_front_pop10plus_oracle"] = wide["partner_style"].eq("front") & wide["partner_pop_band"].eq("pop10plus")
    for name, mask in wide_specs.items():
        part = _normalize_ticket_frame(
            wide,
            race_col=wc,
            ticket_name=name,
            ticket_type="wide",
            horse_cols=["a_horse", "b_horse", "a_ai_rank", "b_ai_rank", "a_popularity", "b_popularity", "a_odds", "b_odds"],
            hit_col="wide_hit",
            return_col="wide_return",
            filter_mask=mask,
            sort_cols=["b_popularity", "b_odds"],
            ascending=[False, False],
        )
        if not part.empty:
            ticket_sets[name] = part

    return ticket_sets


def _stake_amount(bankroll: float, fraction: float, min_unit: int) -> int:
    stake = int((bankroll * fraction) // min_unit) * min_unit
    return max(0, stake)


def _simulate_one_roll(
    sources: dict[str, pd.DataFrame],
    plan: list[str],
    start_idx: int,
    *,
    initial_bankroll: int,
    stake_fraction: float,
    min_unit: int,
    target_bankroll: int,
) -> dict[str, object]:
    bankroll = float(initial_bankroll)
    last_sort = -1
    legs: list[dict[str, object]] = []
    for leg_no, source_name in enumerate(plan, start=1):
        source = sources[source_name]
        candidates = source[source["sort_key"].gt(last_sort)]
        if leg_no == 1:
            candidates = candidates.iloc[start_idx : start_idx + 1]
        else:
            candidates = candidates.head(1)
        if candidates.empty:
            break
        row = candidates.iloc[0]
        stake = _stake_amount(bankroll, stake_fraction, min_unit)
        if stake < min_unit:
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
        if not bool(row["hit"]):
            break
        if bankroll >= target_bankroll:
            break
    all_legs_hit = len(legs) == len(plan) and all(leg["hit"] for leg in legs)
    reached_target = bankroll >= target_bankroll
    return {
        "legs_run": len(legs),
        "all_legs_hit": all_legs_hit,
        "reached_target": reached_target,
        "final_bankroll": bankroll,
        "profit": bankroll - initial_bankroll,
        "legs": legs,
    }


def _simulate_plan(
    sources: dict[str, pd.DataFrame],
    plan_name: str,
    plan: list[str],
    *,
    initial_bankroll: int,
    stake_fraction: float,
    min_unit: int,
    target_bankroll: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    first = sources[plan[0]]
    results = []
    leg_rows = []
    for i in range(len(first)):
        sim = _simulate_one_roll(
            sources,
            plan,
            i,
            initial_bankroll=initial_bankroll,
            stake_fraction=stake_fraction,
            min_unit=min_unit,
            target_bankroll=target_bankroll,
        )
        rows = sim.pop("legs")
        if not rows:
            continue
        start_race_id = rows[0]["race_id"]
        sim.update(
            {
                "plan": plan_name,
                "stake_fraction": stake_fraction,
                "start_race_id": start_race_id,
                "start_date_key": rows[0]["date_key"],
            }
        )
        results.append(sim)
        for row in rows:
            row.update({"plan": plan_name, "start_race_id": start_race_id, "stake_fraction": stake_fraction})
            leg_rows.append(row)
    result_df = pd.DataFrame(results)
    leg_df = pd.DataFrame(leg_rows)
    if result_df.empty:
        summary = {
            "plan": plan_name,
            "stake_fraction": stake_fraction,
            "sessions": 0,
            "target_rate": 0.0,
            "complete_rate": 0.0,
            "median_final_bankroll": 0.0,
            "mean_final_bankroll": 0.0,
            "p90_final_bankroll": 0.0,
            "max_final_bankroll": 0.0,
            "avg_profit_per_session": 0.0,
            "worst_profit": 0.0,
            "max_drawdown_if_sequential": 0.0,
        }
        return summary, leg_df
    summary = {
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
    }
    return summary, leg_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate bankroll rollover strategies from existing ticket candidates.")
    parser.add_argument("--portfolio-csv", default="outputs/analysis/fixed_budget_ticket_portfolio_10000/candidate_tickets.csv")
    parser.add_argument("--wide-csv", default="outputs/analysis/roi_segments_walkforward_v1/wide_pair_tickets_enriched.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/rollover_strategy_v1")
    parser.add_argument("--initial-bankroll", type=int, default=10000)
    parser.add_argument("--target-bankroll", type=int, default=1000000)
    args = parser.parse_args()

    sources = _load_ticket_universe(project_path(args.portfolio_csv), project_path(args.wide_csv))
    plans = {
        "place_x3_survival": ["place_clear_head", "place_clear_head", "place_clear_head"],
        "place_x2_then_wide_strong": ["place_clear_head", "place_clear_head", "wide_strong_value"],
        "place_then_wide_strong": ["place_clear_head", "wide_strong_value"],
        "wide_strong_x2": ["wide_strong_value", "wide_strong_value"],
        "wide_strong_x3": ["wide_strong_value", "wide_strong_value", "wide_strong_value"],
        "win_value_then_wide_strong": ["win_value_top3_gap", "wide_strong_value"],
    }
    if "wide_front_odds20_100_oracle" in sources:
        plans.update(
            {
                "oracle_front_wide_x2": ["wide_front_odds20_100_oracle", "wide_front_odds20_100_oracle"],
                "oracle_place_then_front_wide": ["place_clear_head", "wide_front_odds20_100_oracle"],
                "oracle_front_wide_x3": ["wide_front_odds20_100_oracle", "wide_front_odds20_100_oracle", "wide_front_odds20_100_oracle"],
            }
        )

    summaries = []
    leg_frames = []
    for plan_name, plan in plans.items():
        if any(name not in sources for name in plan):
            continue
        for fraction in [1.0, 0.5]:
            summary, legs = _simulate_plan(
                sources,
                plan_name,
                plan,
                initial_bankroll=args.initial_bankroll,
                stake_fraction=fraction,
                min_unit=100,
                target_bankroll=args.target_bankroll,
            )
            summaries.append(summary)
            if not legs.empty:
                leg_frames.append(legs)

    out_dir = ensure_dir(project_path(args.output_dir))
    summary_df = pd.DataFrame(summaries).sort_values(["target_rate", "mean_final_bankroll"], ascending=[False, False])
    summary_df.to_csv(out_dir / "rollover_summary.csv", index=False, encoding="utf-8-sig")
    if leg_frames:
        pd.concat(leg_frames, ignore_index=True, sort=False).to_csv(out_dir / "rollover_legs.csv", index=False, encoding="utf-8-sig")

    source_summary = []
    for name, df in sources.items():
        source_summary.append(
            {
                "ticket_name": name,
                "tickets": int(len(df)),
                "hit_rate": float(df["hit"].mean()) if len(df) else 0.0,
                "roi_flat100": float(df["return_per100"].sum() / (len(df) * 100.0)) if len(df) else 0.0,
                "avg_return_hit": float(df.loc[df["hit"], "return_per100"].mean()) if int(df["hit"].sum()) else 0.0,
            }
        )
    source_df = pd.DataFrame(source_summary).sort_values("roi_flat100", ascending=False)
    source_df.to_csv(out_dir / "rollover_ticket_source_summary.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "initial_bankroll": args.initial_bankroll,
        "target_bankroll": args.target_bankroll,
        "notes": [
            "This is a rollover/session simulation, not a flat-stake ROI test.",
            "Plans containing 'oracle' use post-race 4-corner style diagnostics and must be replaced by pre-race projected-position features before live use.",
            "Wide returns use actual historical wide payoffs already imported from PC-KEIBA/TARGET-compatible data.",
        ],
        "top_plans": summary_df.head(12).to_dict(orient="records"),
        "ticket_sources": source_df.to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
