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
from scripts.evaluate_ticket_strategies import _add_model_columns, _col
from scripts.optimize_rollover_hit_model import _prepare_fast_sources, _simulate_plan_fast
from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


def _pick_horse_no_col(scored: pd.DataFrame) -> str:
    col = _col(scored, ["馬番", "umaban", "horse_number"])
    if col:
        return col
    candidates = [c for c in scored.columns if "馬番" in c or "umaban" in c.lower()]
    if candidates:
        return candidates[0]
    raise KeyError("Horse number column was not found.")


def _base_cols(scored: pd.DataFrame, race_col: str, prefix: str) -> pd.DataFrame:
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
        "ai_score_gap_to_second",
        "ai_score",
    ]
    return scored[cols].rename(
        columns={
            "horse_name_for_ticket": f"{prefix}_horse",
            horse_no_col: f"{prefix}_horse_no",
            "ai_rank": f"{prefix}_ai_rank",
            "pop_rank": f"{prefix}_pop_rank",
            "popularity_num": f"{prefix}_popularity",
            "odds_decimal": f"{prefix}_odds",
            "rank_num": f"{prefix}_finish",
            "target_top3": f"{prefix}_top3",
            "ai_score_gap_to_second": f"{prefix}_gap",
            "ai_score": f"{prefix}_score",
        }
    )


def _select_anchor(scored: pd.DataFrame, race_col: str, *, max_odds: float, min_gap: float) -> pd.DataFrame:
    mask = (
        scored["ai_rank"].eq(1)
        & scored["ai_score_gap_to_second"].ge(min_gap)
        & pd.to_numeric(scored["odds_decimal"], errors="coerce").le(max_odds)
    )
    anchor = (
        scored[mask]
        .sort_values([race_col, "ai_score_gap_to_second", "ai_score"], ascending=[True, False, False])
        .groupby(race_col, as_index=False)
        .head(1)
    )
    return _base_cols(anchor, race_col, "a")


def _select_partners(scored: pd.DataFrame, race_col: str, *, partner_mode: str, topn: int) -> pd.DataFrame:
    odds = pd.to_numeric(scored["odds_decimal"], errors="coerce")
    if partner_mode == "ai2":
        mask = scored["ai_rank"].between(2, 2)
        sort_cols = [race_col, "ai_rank", "pop_rank", "odds_decimal"]
        ascending = [True, True, True, True]
    elif partner_mode == "ai23":
        mask = scored["ai_rank"].between(2, 3)
        sort_cols = [race_col, "ai_rank", "pop_rank", "odds_decimal"]
        ascending = [True, True, True, True]
    elif partner_mode == "pop23":
        mask = scored["pop_rank"].between(2, 3)
        sort_cols = [race_col, "pop_rank", "ai_rank", "odds_decimal"]
        ascending = [True, True, True, True]
    elif partner_mode == "solid":
        mask = (scored["ai_rank"].between(2, 5)) & (scored["pop_rank"].le(5)) & odds.le(8.0)
        sort_cols = [race_col, "ai_rank", "pop_rank", "odds_decimal"]
        ascending = [True, True, True, True]
    else:
        raise ValueError(partner_mode)
    partners = scored[mask].sort_values(sort_cols, ascending=ascending).groupby(race_col, as_index=False).head(topn)
    return _base_cols(partners, race_col, "b")


def _build_pair_tickets(scored: pd.DataFrame, race_col: str) -> pd.DataFrame:
    frames = []
    for max_odds, min_gap in [(1.4, 0.10), (1.6, 0.10), (1.8, 0.10), (2.2, 0.10), (1.8, 0.15)]:
        anchors = _select_anchor(scored, race_col, max_odds=max_odds, min_gap=min_gap)
        for partner_mode, topn in [("ai2", 1), ("ai23", 2), ("pop23", 2), ("solid", 2)]:
            partners = _select_partners(scored, race_col, partner_mode=partner_mode, topn=topn)
            tickets = anchors.merge(partners, on=race_col, how="inner")
            tickets = tickets[tickets["a_horse"] != tickets["b_horse"]].copy()
            if tickets.empty:
                continue
            tickets["strategy"] = f"axis_o{str(max_odds).replace('.', '_')}_g{str(min_gap).replace('.', '_')}_{partner_mode}_top{topn}"
            tickets["ticket_type"] = "pair"
            tickets["wide_hit"] = tickets["a_top3"].eq(1) & tickets["b_top3"].eq(1)
            tickets["umaren_hit"] = tickets["a_finish"].le(2) & tickets["b_finish"].le(2)
            frames.append(tickets)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _single_from_candidates(candidate_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(candidate_csv, low_memory=False)
    keep = df["strategy"].isin(["place_clear_head", "win_clear_head", "place_core_anchor", "win_core_anchor"])
    df = df[keep].copy()
    odds = pd.to_numeric(df["a_odds"], errors="coerce")
    out_frames = []
    for base_strategy in ["place_clear_head", "win_clear_head", "place_core_anchor", "win_core_anchor"]:
        base = df[df["strategy"].eq(base_strategy)].copy()
        for max_odds in [1.4, 1.6, 1.8, 2.2]:
            part = base[pd.to_numeric(base["a_odds"], errors="coerce").le(max_odds)].copy()
            if part.empty:
                continue
            part["solid_strategy"] = f"{base_strategy}_odds_le_{str(max_odds).replace('.', '_')}"
            out_frames.append(part)
    return pd.concat(out_frames, ignore_index=True, sort=False) if out_frames else pd.DataFrame()


def _metrics_single(singles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in singles.groupby("solid_strategy"):
        stake = len(g) * 100.0
        ret = g["return_per100"].fillna(0.0).sum()
        hit_rate = float(g["hit"].mean())
        rows.append(
            {
                "strategy": strategy,
                "ticket_type": g["ticket_type"].iloc[0],
                "tickets": int(len(g)),
                "races": int(g["レースID(新/馬番無)"].nunique()) if "レースID(新/馬番無)" in g.columns else int(g.iloc[:, 0].nunique()),
                "hit_rate": hit_rate,
                "roi": float(ret / stake) if stake else 0.0,
                "avg_hit_return": float(g.loc[g["hit"], "return_per100"].mean()) if int(g["hit"].sum()) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _metrics_pair(tickets: pd.DataFrame, payoffs: pd.DataFrame, race_col: str) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame()
    work = tickets.copy()
    work[race_col] = work[race_col].astype(str)
    if "umaren_pay" not in work.columns:
        pay = payoffs[[race_col, "umaren_pay"]].copy()
        pay[race_col] = pay[race_col].astype(str)
        work = work.merge(pay, on=race_col, how="left")
    rows = []
    for strategy, g in work.groupby("strategy"):
        stake = len(g) * 100.0
        wide_return = g["wide_pay"].fillna(0.0).where(g["wide_hit"], 0.0).sum()
        umaren_return = g["umaren_pay"].fillna(0.0).where(g["umaren_hit"], 0.0).sum()
        rows.append(
            {
                "strategy": strategy,
                "tickets": int(len(g)),
                "races": int(g[race_col].nunique()),
                "avg_anchor_odds": float(g["a_odds"].mean()),
                "avg_partner_pop": float(g["b_popularity"].mean()),
                "avg_partner_odds": float(g["b_odds"].mean()),
                "wide_hit_rate": float(g["wide_hit"].mean()),
                "wide_roi": float(wide_return / stake) if stake else 0.0,
                "wide_avg_hit_return": float(g.loc[g["wide_hit"], "wide_pay"].mean()) if int(g["wide_hit"].sum()) else 0.0,
                "umaren_hit_rate": float(g["umaren_hit"].mean()),
                "umaren_roi": float(umaren_return / stake) if stake else 0.0,
                "umaren_avg_hit_return": float(g.loc[g["umaren_hit"], "umaren_pay"].mean()) if int(g["umaren_hit"].sum()) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["wide_hit_rate", "wide_roi"], ascending=[False, False])


def _ticket_sources_for_rollover(singles: pd.DataFrame, pairs: pd.DataFrame, race_col: str) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    race_id_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in singles.columns else singles.columns[0]
    for strategy, g in singles.groupby("solid_strategy"):
        frame = pd.DataFrame(
            {
                "race_id": g[race_id_col].astype(str),
                "sort_key": g[race_id_col].map(_race_sort_key),
                "date_key": g[race_id_col].map(_date_key),
                "ticket_name": strategy,
                "ticket_type": g["ticket_type"],
                "hit": g["hit"].astype(bool),
                "return_per100": pd.to_numeric(g["return_per100"], errors="coerce").fillna(0.0),
                "a_horse": g["a_horse"],
                "a_odds": g["a_odds"],
            }
        ).sort_values("sort_key")
        if len(frame) >= 80:
            sources[strategy] = frame
    for strategy, g in pairs.groupby("strategy"):
        for ticket_type, hit_col, ret_col in [
            ("wide", "wide_hit", "wide_pay"),
            ("umaren", "umaren_hit", "umaren_pay"),
        ]:
            frame = pd.DataFrame(
                {
                    "race_id": g[race_col].astype(str),
                    "sort_key": g[race_col].map(_race_sort_key),
                    "date_key": g[race_col].map(_date_key),
                    "ticket_name": f"{strategy}_{ticket_type}",
                    "ticket_type": ticket_type,
                    "hit": g[hit_col].astype(bool),
                    "return_per100": pd.to_numeric(g[ret_col], errors="coerce").fillna(0.0).where(g[hit_col].astype(bool), 0.0),
                    "a_horse": g["a_horse"],
                    "b_horse": g["b_horse"],
                    "a_odds": g["a_odds"],
                    "b_odds": g["b_odds"],
                }
            ).sort_values("sort_key")
            if len(frame) >= 80:
                sources[f"{strategy}_{ticket_type}"] = frame
    return sources


def _run_rollover(sources: dict[str, pd.DataFrame], targets: list[int]) -> pd.DataFrame:
    source_stats = []
    for name, df in sources.items():
        source_stats.append(
            {
                "name": name,
                "hit": float(df["hit"].mean()),
                "roi": float(df["return_per100"].sum() / (len(df) * 100.0)),
                "tickets": int(len(df)),
            }
        )
    stats = pd.DataFrame(source_stats)
    survival = stats[stats["hit"].ge(0.85)].sort_values(["hit", "roi"], ascending=[False, False]).head(4)["name"].tolist()
    growth = stats[stats["hit"].ge(0.35)].sort_values(["roi", "hit"], ascending=[False, False]).head(8)["name"].tolist()
    plans: dict[str, list[str]] = {}
    for a in survival:
        plans[f"{a}_x2"] = [a, a]
        plans[f"{a}_x3"] = [a, a, a]
        for b in growth:
            if a != b:
                plans[f"{a}_then_{b}"] = [a, b]
                plans[f"{a}_x2_then_{b}"] = [a, a, b]
    fast = _prepare_fast_sources(sources)
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
    return pd.DataFrame(rows).sort_values(["target_bankroll", "target_rate", "complete_rate", "mean_final_bankroll"], ascending=[True, False, False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate solid-axis ticket choices for rollover betting.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default="models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--test-csv", default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv")
    parser.add_argument("--candidate-csv", default="outputs/analysis/fixed_budget_ticket_portfolio_10000/candidate_tickets.csv")
    parser.add_argument("--raw-csv", default="date/raw/蜈ｨ遶ｶ襍ｰ鬥ｬ謌千ｸｾ.csv")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/solid_axis_ticket_model_v1")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    encoding = config["data"].get("encoding", "cp932")
    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = _add_model_columns(df, project_path(args.model), race_col)
    scored[race_col] = scored[race_col].astype(str)

    raw_csv = _existing_raw_csv(args.raw_csv)
    payoffs, _ = _load_payoffs(raw_csv, race_col, encoding)
    pair_tickets = _build_pair_tickets(scored, race_col)
    pair_tickets = _attach_wide_payoffs(pair_tickets, _load_wide_payoffs(args.wide_payoff_csv, race_col), race_col)
    pay_for_pairs = payoffs[[race_col, "umaren_pay"]].copy()
    pay_for_pairs[race_col] = pay_for_pairs[race_col].astype(str)
    pair_tickets = pair_tickets.merge(pay_for_pairs, on=race_col, how="left")

    singles = _single_from_candidates(project_path(args.candidate_csv))
    single_summary = _metrics_single(singles)
    pair_summary = _metrics_pair(pair_tickets, payoffs, race_col)
    sources = _ticket_sources_for_rollover(singles, pair_tickets, race_col)
    rollover = _run_rollover(sources, [20000, 50000, 100000])

    out_dir = ensure_dir(project_path(args.output_dir))
    single_summary.to_csv(out_dir / "solid_axis_single_summary.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(out_dir / "solid_axis_pair_summary.csv", index=False, encoding="utf-8-sig")
    pair_tickets.to_csv(out_dir / "solid_axis_pair_tickets.csv", index=False, encoding="utf-8-sig")
    rollover.to_csv(out_dir / "solid_axis_rollover_plans.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "single_top": single_summary.sort_values(["hit_rate", "roi"], ascending=[False, False]).head(12).to_dict(orient="records"),
        "pair_top_by_hit": pair_summary.head(12).to_dict(orient="records"),
        "rollover_top_by_target": {
            str(target): rollover[rollover["target_bankroll"].eq(target)].head(10).to_dict(orient="records")
            for target in [20000, 50000, 100000]
        },
        "interpretation": {
            "point": "This model avoids longshot-first logic. It only evaluates races with a strong low-odds AI axis, then chooses fixed ticket types.",
            "live_rule_shape": "First select a solid-axis race; then choose place, win, wide-to-AI2, or umaren-to-AI2 depending on the strategy table.",
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
