from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_rollover_strategy import _load_ticket_universe
from scripts.optimize_rollover_hit_model import _prepare_fast_sources, _simulate_plan_fast
from src.utils.paths import ensure_dir, project_path


def _variant(df: pd.DataFrame, name: str, mask: pd.Series) -> pd.DataFrame:
    out = df[mask.fillna(False)].copy()
    if not out.empty:
        out["ticket_name"] = name
    return out


def _solid_sources(base: dict[str, pd.DataFrame], min_tickets: int) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    for base_name in ["place_clear_head", "place_core_anchor", "win_clear_head", "win_core_anchor"]:
        if base_name not in base:
            continue
        df = base[base_name]
        odds = pd.to_numeric(df.get("a_odds"), errors="coerce")
        pop = pd.to_numeric(df.get("a_popularity"), errors="coerce")
        specs = [
            (base_name, pd.Series(True, index=df.index)),
            (f"{base_name}_odds_le_1_4", odds.le(1.4)),
            (f"{base_name}_odds_le_1_6", odds.le(1.6)),
            (f"{base_name}_odds_le_1_8", odds.le(1.8)),
            (f"{base_name}_odds_le_2_0", odds.le(2.0)),
            (f"{base_name}_odds_le_2_2", odds.le(2.2)),
            (f"{base_name}_pop1", pop.eq(1)),
            (f"{base_name}_pop_le_2", pop.le(2)),
        ]
        for name, mask in specs:
            part = _variant(df, name, mask)
            if len(part) >= min_tickets:
                sources[name] = part
    return sources


def _source_stats(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in sources.items():
        hits = df[df["hit"]]
        rows.append(
            {
                "ticket_name": name,
                "tickets": int(len(df)),
                "hit_rate": float(df["hit"].mean()),
                "roi": float(df["return_per100"].sum() / (len(df) * 100.0)) if len(df) else 0.0,
                "avg_hit_return": float(hits["return_per100"].mean()) if len(hits) else 0.0,
                "median_hit_return": float(hits["return_per100"].median()) if len(hits) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["hit_rate", "roi"], ascending=[False, False])


def _plans(stats: pd.DataFrame) -> dict[str, list[str]]:
    place = stats[stats["ticket_name"].str.startswith("place_")].copy()
    win = stats[stats["ticket_name"].str.startswith("win_")].copy()
    place = place[(place["hit_rate"] >= 0.75) & (place["tickets"] >= 80)].head(4)
    win = win[(win["hit_rate"] >= 0.45) & (win["tickets"] >= 80)].head(3)

    plans: dict[str, list[str]] = {}
    for name in place["ticket_name"]:
        for n in [2, 3, 5, 8]:
            plans[f"{name}_x{n}"] = [name] * n
    for p in place["ticket_name"].head(3):
        for w in win["ticket_name"].head(3):
            plans[f"{p}_x2_then_{w}"] = [p, p, w]
            plans[f"{p}_then_{w}_x2"] = [p, w, w]
            plans[f"{p}_then_{w}"] = [p, w]
    for w in win["ticket_name"].head(5):
        for n in [2, 3]:
            plans[f"{w}_x{n}"] = [w] * n
    return plans


def _score(row: dict[str, object]) -> float:
    target_rate = float(row.get("target_rate", 0.0))
    complete_rate = float(row.get("complete_rate", 0.0))
    median_final = float(row.get("median_final_bankroll", 0.0))
    mean_final = float(row.get("mean_final_bankroll", 0.0))
    worst = abs(float(row.get("worst_profit", 0.0)))
    dd = float(row.get("max_drawdown_if_sequential", 0.0))
    return (
        complete_rate * 40.0
        + target_rate * 80.0
        + min(median_final / 10000.0, 2.0) * 8.0
        + min(mean_final / 10000.0, 2.0) * 4.0
        - min(worst / 10000.0, 1.0) * 4.0
        - min(dd / 1000000.0, 3.0) * 2.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate solid hit-rate-first rollover plans without longshot legs.")
    parser.add_argument("--portfolio-csv", default="outputs/analysis/fixed_budget_ticket_portfolio_10000/candidate_tickets.csv")
    parser.add_argument("--wide-csv", default="outputs/analysis/roi_segments_walkforward_v1/wide_pair_tickets_enriched.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/solid_rollover_model_v1")
    parser.add_argument("--initial-bankroll", type=int, default=10000)
    parser.add_argument("--min-tickets", type=int, default=80)
    args = parser.parse_args()

    base = _load_ticket_universe(project_path(args.portfolio_csv), project_path(args.wide_csv))
    sources = _solid_sources(base, min_tickets=args.min_tickets)
    stats = _source_stats(sources)
    plans = _plans(stats)
    fast = _prepare_fast_sources(sources)

    rows = []
    leg_frames = []
    for target in [20000, 50000, 100000]:
        for plan_name, plan in plans.items():
            for fraction in [1.0, 0.75, 0.5, 0.33]:
                summary, legs = _simulate_plan_fast(
                    fast,
                    plan_name,
                    plan,
                    initial_bankroll=args.initial_bankroll,
                    stake_fraction=fraction,
                    target_bankroll=target,
                )
                if int(summary["sessions"]) == 0:
                    continue
                summary["target_bankroll"] = target
                summary["legs"] = " > ".join(plan)
                summary["solid_score"] = _score(summary)
                rows.append(summary)
                if not legs.empty:
                    legs["target_bankroll"] = target
                    legs["solid_plan"] = plan_name
                    legs["solid_legs"] = " > ".join(plan)
                    leg_frames.append(legs)

    out_dir = ensure_dir(project_path(args.output_dir))
    result = pd.DataFrame(rows).sort_values(["target_bankroll", "solid_score"], ascending=[True, False])
    stats.to_csv(out_dir / "solid_ticket_sources.csv", index=False, encoding="utf-8-sig")
    result.to_csv(out_dir / "solid_rollover_plans.csv", index=False, encoding="utf-8-sig")
    if leg_frames:
        pd.concat(leg_frames, ignore_index=True, sort=False).to_csv(out_dir / "solid_rollover_legs.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "initial_bankroll": args.initial_bankroll,
        "top_by_target": {
            str(target): result[result["target_bankroll"].eq(target)].head(10).to_dict(orient="records")
            for target in [20000, 50000, 100000]
        },
        "ticket_sources": stats.head(30).to_dict(orient="records"),
        "interpretation": {
            "model_type": "Solid rollover: no longshot growth leg, race selection first.",
            "rule": "Only use high-confidence win/place candidates with strict odds/popularity filters.",
            "caution": "Higher hit rate lowers growth. This model is better for small-step rollover than one-shot large bankroll jumps.",
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
