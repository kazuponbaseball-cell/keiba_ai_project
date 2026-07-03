from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_dynamic_pair_ticket_allocation import _load_pairs, _load_wide, _pair_universe
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _norm01(series: pd.Series) -> pd.Series:
    s = _num(series).replace([np.inf, -np.inf], np.nan)
    lo = s.quantile(0.05)
    hi = s.quantile(0.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.5)


def _existing_raw_csv(path_text: str) -> Path:
    path = project_path(path_text)
    if path.exists():
        return path
    files = list(project_path("date/raw").glob("*.csv"))
    if not files:
        raise FileNotFoundError(path_text)
    return files[0]


def _race_level_features(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    race = "race_id"
    df[race] = df[race].astype(str)
    ai_rank = _num(df.get("ai_rank_num"), df.index)
    pop = _num(df.get("pop_rank_num"), df.index)
    odds = _num(df.get("market_odds_live_or_final"), df.index)

    top1 = df[ai_rank.eq(1)].copy()
    top1 = top1[[race]].assign(
        race_anchor_danger=_num(top1.get("danger_popular_hybrid_score"), top1.index, np.nan).fillna(
            _num(top1.get("danger_favorite_score"), top1.index, 0.0)
        ),
        race_anchor_skip=_num(top1.get("skip_risk_score"), top1.index, 0.0),
        race_anchor_odds=_num(top1.get("market_odds_live_or_final"), top1.index, np.nan),
        race_anchor_win_score=_num(top1.get("win_suitability_score"), top1.index, 0.0),
        race_anchor_model_difficulty=_num(top1.get("race_difficulty_model_score"), top1.index, np.nan),
    )

    danger_source = "danger_popular_hybrid_score" if "danger_popular_hybrid_score" in df.columns else "danger_favorite_score"
    by = df.groupby(race, as_index=False).agg(
        race_field_size=("horse_no", "count"),
        race_favorite_danger=(danger_source, "max"),
        race_avg_late_drop=("late_odds_drop_rate", "mean"),
        race_avg_late_drift=("late_odds_drift_rate", "mean"),
        race_front_pressure=("race_early_pressure_score", "max"),
        race_front_count=("race_front_runner_count_y", "max"),
        race_pace_collapse=("race_pace_collapse_risk", "max"),
        race_slow_risk=("race_slow_pace_risk", "max"),
        race_bias_volatility=("same_day_bias_volatility", "max"),
        race_same_day_ready=("same_day_bias_ready", "max"),
    )
    by = by.merge(top1, on=race, how="left")

    market_top3 = (
        df[pop.between(1, 3, inclusive="both")]
        .groupby(race)["market_win_prob_norm"]
        .sum()
        .rename("race_market_top3_prob_sum")
        .reset_index()
    )
    by = by.merge(market_top3, on=race, how="left")
    by["race_market_top3_prob_sum"] = by["race_market_top3_prob_sum"].fillna(0.0)

    class_name = df.groupby(race)["クラス名"].first().astype(str).rename("race_class_name").reset_index()
    by = by.merge(class_name, on=race, how="left")
    by["race_young_or_maiden_risk"] = by["race_class_name"].str.contains("新馬|未勝利", regex=True, na=False).astype(float)

    by["race_difficulty_score"] = (
        0.18 * _norm01(by["race_field_size"])
        + 0.16 * _norm01(by["race_favorite_danger"])
        + 0.14 * _norm01(by["race_pace_collapse"])
        + 0.12 * _norm01(by["race_slow_risk"])
        + 0.12 * _norm01(by["race_bias_volatility"])
        + 0.10 * by["race_young_or_maiden_risk"]
        + 0.10 * (1.0 - _norm01(by["race_market_top3_prob_sum"]))
        + 0.08 * _norm01(by["race_anchor_skip"])
    ).clip(0.0, 1.0)
    model_difficulty = _num(by.get("race_anchor_model_difficulty"), by.index, np.nan)
    by["race_difficulty_rule_score"] = by["race_difficulty_score"]
    by["race_difficulty_score"] = model_difficulty.fillna(by["race_difficulty_rule_score"]).clip(0.0, 1.0)
    return by


def _attach_race_features(universe: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    race = _race_level_features(scored)
    return universe.merge(race, on="race_id", how="left")


def _select_pairs(universe: pd.DataFrame, params: dict) -> pd.DataFrame:
    u = universe.copy()
    market_ok = _num(u["late_value_survives_score"]).ge(params["late_value_min"])
    if params["avoid_late_drift"]:
        market_ok &= _num(u["late_odds_drift_rate"] if "late_odds_drift_rate" in u else pd.Series(0, index=u.index)).le(0.55)

    mask = (
        _num(u["race_difficulty_score"]).le(params["race_difficulty_max"])
        & _num(u["wide_axis_score"]).ge(params["axis_min"])
        & _num(u["wide_partner_score"]).ge(params["partner_min"])
        & _num(u["partner_odds"]).between(params["partner_odds_min"], params["partner_odds_max"])
        & _num(u["projected_front5_prob"]).ge(params["front_min"])
        & _num(u["anchor_danger"]).le(params["anchor_danger_max"])
        & _num(u["partner_danger"]).le(params["partner_danger_max"])
        & market_ok
    )
    selected = u[mask].copy()
    if selected.empty:
        return selected
    return (
        selected.sort_values(
            ["race_id", "pair_quinella_score", "pair_score", "market_overlay_score"],
            ascending=[True, False, False, False],
        )
        .groupby("race_id", as_index=False)
        .head(params["pairs_per_race"])
    )


def _tickets_from_pairs(pairs: pd.DataFrame, params: dict) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    frames = []
    base = pairs.copy()
    base["pair_key"] = base["race_id"].astype(str) + ":" + base["anchor_no"].astype(str) + "-" + base["partner_no"].astype(str)

    if params["wide_base_stake"] > 0:
        wide = base.copy()
        wide["ticket_type"] = "wide"
        wide["stake_yen"] = np.where(
            (_num(wide["pair_score"]) >= params["wide_thick_pair_score"]) & (_num(wide["race_difficulty_score"]) <= params["wide_thick_difficulty_max"]),
            params["wide_thick_stake"],
            params["wide_base_stake"],
        )
        wide["hit"] = wide["wide_hit"]
        wide["return_yen"] = _num(wide["wide_pay"]).fillna(0.0) * wide["stake_yen"] / 100.0
        frames.append(wide)

    umaren = base[
        _num(base["pair_quinella_score"]).ge(params["umaren_quinella_min"])
        & _num(base["partner_quinella_score"]).ge(params["partner_quinella_min"])
        & _num(base["umaren_pay"]).ge(params["umaren_pay_min"])
        & _num(base["partner_odds"]).le(params["umaren_partner_odds_max"])
    ].copy()
    if not umaren.empty:
        umaren["ticket_type"] = "umaren"
        umaren["stake_yen"] = params["umaren_stake"]
        umaren["hit"] = umaren["umaren_hit"]
        umaren["return_yen"] = _num(umaren["umaren_pay"]).fillna(0.0).where(umaren["hit"], 0.0) * umaren["stake_yen"] / 100.0
        frames.append(umaren)

    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _metrics(tickets: pd.DataFrame, label: str) -> dict:
    if tickets.empty:
        return {"policy": label, "tickets": 0, "races": 0, "roi": 0.0, "race_hit_rate": 0.0}
    stake = float(_num(tickets["stake_yen"]).sum())
    ret = float(_num(tickets["return_yen"]).sum())
    by_race = tickets.groupby("race_id").agg(stake=("stake_yen", "sum"), ret=("return_yen", "sum"), hit=("hit", "max"))
    by_race = by_race.sort_index()
    equity = (by_race["ret"] - by_race["stake"]).cumsum()
    dd = equity - equity.cummax()
    type_roi = {}
    for ticket_type, g in tickets.groupby("ticket_type"):
        st = float(_num(g["stake_yen"]).sum())
        type_roi[f"{ticket_type}_roi"] = float(_num(g["return_yen"]).sum() / st) if st else 0.0
        type_roi[f"{ticket_type}_tickets"] = int(len(g))
    return {
        "policy": label,
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "avg_stake_per_race": float(by_race["stake"].mean()),
        "ticket_hit_rate": float(tickets["hit"].mean()),
        "race_hit_rate": float(by_race["hit"].mean()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "max_drawdown_yen": float(dd.min()) if not dd.empty else 0.0,
        **type_roi,
    }


def _grid() -> list[dict]:
    rows = []
    for race_diff, axis_min, partner_min, front_min, topn, late_min, wide_base_stake, thick_stake in product(
        [0.58, 0.66, 0.74],
        [0.62, 0.70],
        [0.60, 0.66],
        [0.45, 0.58],
        [1, 2],
        [0.45, 0.60],
        [0, 100],
        [100, 200, 300],
    ):
        rows.append(
            {
                "race_difficulty_max": race_diff,
                "axis_min": axis_min,
                "partner_min": partner_min,
                "partner_odds_min": 6.0,
                "partner_odds_max": 40.0,
                "front_min": front_min,
                "pairs_per_race": topn,
                "late_value_min": late_min,
                "avoid_late_drift": True,
                "anchor_danger_max": 0.50,
                "partner_danger_max": 0.35,
                "wide_base_stake": wide_base_stake,
                "wide_thick_stake": thick_stake,
                "wide_thick_pair_score": 0.78,
                "wide_thick_difficulty_max": 0.58,
                "umaren_stake": 100,
                "umaren_quinella_min": 0.58,
                "partner_quinella_min": 0.54,
                "umaren_pay_min": 1200.0,
                "umaren_partner_odds_max": 25.0,
            }
        )
    return rows


def _selection_score(metric: dict, min_races: int, min_hit: float) -> float:
    if metric["races"] < min_races or metric["race_hit_rate"] < min_hit:
        return -999.0
    return metric["roi"] * np.sqrt(max(metric["race_hit_rate"], 0.001)) * np.log1p(metric["races"])


def _walkforward(universe: pd.DataFrame, min_train_races: int, min_race_hit: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = sorted(universe["year"].dropna().astype(int).unique())
    grid = _grid()
    train_rows = []
    wf_rows = []
    ticket_frames = []
    for test_year in years[1:]:
        train = universe[universe["year"] < test_year]
        test = universe[universe["year"] == test_year]
        scored = []
        for i, params in enumerate(grid):
            tickets = _tickets_from_pairs(_select_pairs(train, params), params)
            metric = _metrics(tickets, f"grid_{i}")
            metric.update(params)
            metric["selection_score"] = _selection_score(metric, min_train_races, min_race_hit)
            scored.append(metric)
        train_grid = pd.DataFrame(scored).sort_values(["selection_score", "roi", "race_hit_rate"], ascending=[False, False, False])
        best = train_grid.iloc[0].to_dict()
        params = {k: best[k] for k in grid[0].keys()}
        train_grid["test_year"] = test_year
        train_rows.append(train_grid.head(40))

        tickets = _tickets_from_pairs(_select_pairs(test, params), params)
        metric = _metrics(tickets, f"wf_test_{test_year}")
        metric.update(params)
        metric["test_year"] = test_year
        metric["train_roi"] = float(best["roi"])
        metric["train_race_hit_rate"] = float(best["race_hit_rate"])
        metric["train_races"] = int(best["races"])
        wf_rows.append(metric)
        if not tickets.empty:
            tmp = tickets.copy()
            tmp["test_year"] = test_year
            ticket_frames.append(tmp)
    return (
        pd.concat(train_rows, ignore_index=True, sort=False) if train_rows else pd.DataFrame(),
        pd.DataFrame(wf_rows),
        pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize priority-S betting gates: late odds/value, ticket EV, danger favorites, race difficulty.")
    parser.add_argument("--scored-csv", default="outputs/analysis/quinella_top2_model_v1/investment_features_with_quinella_score.csv")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    parser.add_argument("--raw-csv", default="date/raw/蜈ｨ遶ｶ襍ｰ鬥ｬ謌千ｸｾ.csv")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--output-dir", default="outputs/analysis/priority_s_betting_policy_v1")
    parser.add_argument("--min-train-races", type=int, default=250)
    parser.add_argument("--min-race-hit", type=float, default=0.15)
    args = parser.parse_args()

    scored = pd.read_csv(project_path(args.scored_csv), low_memory=False)
    scored["race_id"] = scored["race_id"].astype(str)
    wide = _load_wide(project_path(args.wide_payoff_csv))
    pairs = _load_pairs(_existing_raw_csv(args.raw_csv), project_path(args.config))
    universe = _attach_race_features(_pair_universe(scored, wide, pairs), scored)
    train_grid, wf_summary, wf_tickets = _walkforward(universe, args.min_train_races, args.min_race_hit)

    out_dir = ensure_dir(project_path(args.output_dir))
    universe.to_csv(out_dir / "priority_s_pair_universe.csv", index=False, encoding="utf-8-sig")
    train_grid.to_csv(out_dir / "walkforward_train_grid_top40.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(out_dir / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    wf_tickets.to_csv(out_dir / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(out_dir),
        "pair_candidates": int(len(universe)),
        "walkforward_summary": wf_summary.to_dict(orient="records"),
        "walkforward_total": _metrics(wf_tickets, "walkforward_total"),
        "note": "Priority-S gate: race difficulty, late odds/value survival, ticket-specific EV, and danger favorite controls.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
