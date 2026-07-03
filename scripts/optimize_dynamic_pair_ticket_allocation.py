from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_ticket_strategies import _load_pair_payoffs
from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = series.astype(str).str.replace(r"[(),]", "", regex=True).replace({"nan": np.nan, "": np.nan})
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _load_wide(path: Path) -> pd.DataFrame:
    wide = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
    for c in ["horse_a", "horse_b", "wide_pay"]:
        wide[c] = _num(wide[c])
    return wide


def _load_pairs(raw_csv: Path, config_path: Path) -> pd.DataFrame:
    config = load_json_config(str(config_path))
    race_col = config["data"]["race_id_column"]
    encoding = config["data"].get("encoding", "cp932")
    pay = _load_pair_payoffs(raw_csv, race_col, encoding)
    return pay.rename(columns={race_col: "race_id"}).assign(race_id=lambda x: x["race_id"].astype(str))


def _pair_universe(scored: pd.DataFrame, wide: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = df["race_id"].str[:4].astype(int)
    for optional_col in [
        "quinella_model_score",
        "quinella_model_score_norm",
        "quinella_model_rank",
        "danger_popular_model_score",
        "danger_popular_hybrid_score",
        "race_difficulty_model_score",
    ]:
        if optional_col not in df.columns:
            df[optional_col] = np.nan
    anchors = df[_num(df["ai_rank_num"]).eq(1)].copy()
    partners = df[_num(df["ai_rank_num"]).between(2, 8)].copy()
    rows = []
    a_cols = [
        "race_id",
        "year",
        "horse_no",
        "horse_name",
        "finish_num",
        "pop_rank_num",
        "market_odds_live_or_final",
        "ai_win_prob_proxy",
        "market_win_prob_norm",
        "quinella_model_score",
        "quinella_model_score_norm",
        "quinella_model_rank",
        "win_suitability_score",
        "place_suitability_score",
        "wide_axis_score",
        "danger_favorite_score",
        "danger_popular_model_score",
        "danger_popular_hybrid_score",
        "race_difficulty_model_score",
        "skip_risk_score",
    ]
    p_cols = [
        "race_id",
        "horse_no",
        "horse_name",
        "finish_num",
        "ai_rank_num",
        "pop_rank_num",
        "market_odds_live_or_final",
        "ai_win_prob_proxy",
        "market_win_prob_norm",
        "quinella_model_score",
        "quinella_model_score_norm",
        "quinella_model_rank",
        "win_suitability_score",
        "place_suitability_score",
        "wide_partner_score",
        "market_overlay_score",
        "late_value_survives_score",
        "projected_front5_prob",
        "danger_favorite_score",
        "danger_popular_model_score",
        "danger_popular_hybrid_score",
        "race_difficulty_model_score",
    ]
    for race_id, a in anchors.groupby("race_id"):
        p = partners[partners["race_id"].eq(race_id)]
        if p.empty:
            continue
        left = a[a_cols].head(1).rename(
            columns={
                "horse_no": "anchor_no",
                "horse_name": "anchor_name",
                "finish_num": "anchor_finish",
                "pop_rank_num": "anchor_pop",
                "market_odds_live_or_final": "anchor_odds",
                "ai_win_prob_proxy": "anchor_ai_win_prob",
                "market_win_prob_norm": "anchor_market_win_prob",
                "quinella_model_score": "anchor_quinella_model_score",
                "quinella_model_score_norm": "anchor_quinella_model_score_norm",
                "quinella_model_rank": "anchor_quinella_model_rank",
                "win_suitability_score": "anchor_win_score",
                "place_suitability_score": "anchor_place_score",
                "danger_favorite_score": "anchor_danger",
                "danger_popular_model_score": "anchor_danger_model",
                "danger_popular_hybrid_score": "anchor_danger_hybrid",
                "race_difficulty_model_score": "anchor_race_difficulty_model",
            }
        )
        right = p[p_cols].rename(
            columns={
                "horse_no": "partner_no",
                "horse_name": "partner_name",
                "finish_num": "partner_finish",
                "ai_rank_num": "partner_ai_rank",
                "pop_rank_num": "partner_pop",
                "market_odds_live_or_final": "partner_odds",
                "ai_win_prob_proxy": "partner_ai_win_prob",
                "market_win_prob_norm": "partner_market_win_prob",
                "quinella_model_score": "partner_quinella_model_score",
                "quinella_model_score_norm": "partner_quinella_model_score_norm",
                "quinella_model_rank": "partner_quinella_model_rank",
                "win_suitability_score": "partner_win_score",
                "place_suitability_score": "partner_place_score",
                "danger_favorite_score": "partner_danger",
                "danger_popular_model_score": "partner_danger_model",
                "danger_popular_hybrid_score": "partner_danger_hybrid",
                "race_difficulty_model_score": "partner_race_difficulty_model",
            }
        )
        rows.append(left.merge(right, on="race_id", how="inner"))
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if out.empty:
        return out
    out = out[out["anchor_no"] != out["partner_no"]].copy()
    out["anchor_danger_rule"] = _num(out["anchor_danger"])
    out["partner_danger_rule"] = _num(out["partner_danger"])
    out["anchor_danger"] = _num(out["anchor_danger_hybrid"]).fillna(out["anchor_danger_rule"])
    out["partner_danger"] = _num(out["partner_danger_hybrid"]).fillna(out["partner_danger_rule"])
    out["horse_a"] = np.minimum(_num(out["anchor_no"]), _num(out["partner_no"]))
    out["horse_b"] = np.maximum(_num(out["anchor_no"]), _num(out["partner_no"]))
    out = out.merge(wide, on=["race_id", "horse_a", "horse_b"], how="left")
    out = out.merge(pairs, on="race_id", how="left")
    out["wide_hit"] = out["wide_pay"].notna()
    out["umaren_hit"] = _num(out["anchor_finish"]).le(2) & _num(out["partner_finish"]).le(2)
    out["umatan_anchor_hit"] = _num(out["anchor_finish"]).eq(1) & _num(out["partner_finish"]).eq(2)
    out["umatan_partner_hit"] = _num(out["partner_finish"]).eq(1) & _num(out["anchor_finish"]).eq(2)
    anchor_model_q = _num(out["anchor_quinella_model_score_norm"]).fillna(_num(out["anchor_place_score"]))
    partner_model_q = _num(out["partner_quinella_model_score_norm"]).fillna(_num(out["partner_place_score"]))
    out["anchor_quinella_score"] = (
        0.45 * anchor_model_q
        + 0.25 * _num(out["anchor_place_score"])
        + 0.20 * _num(out["anchor_win_score"])
        + 0.10 * _num(out["wide_axis_score"])
    )
    out["partner_quinella_score"] = (
        0.45 * partner_model_q
        + 0.20 * _num(out["partner_place_score"])
        + 0.15 * _num(out["partner_win_score"])
        + 0.10 * _num(out["projected_front5_prob"])
        + 0.10 * _num(out["market_overlay_score"])
    )
    out["pair_score"] = (
        0.35 * _num(out["wide_axis_score"])
        + 0.35 * _num(out["wide_partner_score"])
        + 0.20 * _num(out["market_overlay_score"])
        + 0.10 * _num(out["projected_front5_prob"])
    )
    out["pair_quinella_score"] = (
        0.40 * _num(out["anchor_quinella_score"])
        + 0.40 * _num(out["partner_quinella_score"])
        + 0.20 * _num(out["pair_score"])
    )
    return out


def _select_pairs(universe: pd.DataFrame, params: dict) -> pd.DataFrame:
    u = universe.copy()
    mask = (
        _num(u["wide_axis_score"]).ge(params["axis_min"])
        & _num(u["wide_partner_score"]).ge(params["partner_min"])
        & _num(u["partner_odds"]).between(params["partner_odds_min"], params["partner_odds_max"])
        & _num(u["projected_front5_prob"]).ge(params["front_min"])
        & _num(u["anchor_danger"]).le(params["anchor_danger_max"])
        & _num(u["partner_danger"]).le(params["partner_danger_max"])
    )
    selected = u[mask].copy()
    if selected.empty:
        return selected
    return (
        selected.sort_values(["race_id", "pair_score", "market_overlay_score"], ascending=[True, False, False])
        .groupby("race_id", as_index=False)
        .head(params["pairs_per_race"])
    )


def _tickets_from_pairs(pairs: pd.DataFrame, params: dict) -> pd.DataFrame:
    frames = []
    if pairs.empty:
        return pd.DataFrame()
    base = pairs.copy()
    base["pair_key"] = base["race_id"].astype(str) + ":" + base["anchor_no"].astype(str) + "-" + base["partner_no"].astype(str)

    wide = base.copy()
    wide["ticket_type"] = "wide"
    wide["stake_yen"] = params["wide_stake"]
    wide["hit"] = wide["wide_hit"]
    wide["return_yen"] = _num(wide["wide_pay"]).fillna(0.0) * wide["stake_yen"] / 100.0
    frames.append(wide)

    if params["umaren_stake"] > 0:
        umaren = base[
            _num(base["pair_score"]).ge(params["umaren_pair_score_min"])
            & _num(base["pair_quinella_score"]).ge(params.get("umaren_quinella_min", 0.0))
            & _num(base["anchor_quinella_score"]).ge(params.get("anchor_quinella_min", 0.0))
            & _num(base["partner_quinella_score"]).ge(params.get("partner_quinella_min", 0.0))
            & _num(base["partner_odds"]).le(params["umaren_partner_odds_max"])
            & _num(base["umaren_pay"]).ge(params.get("umaren_pay_min", 0.0))
        ].copy()
        if not umaren.empty:
            umaren["ticket_type"] = "umaren"
            umaren["stake_yen"] = params["umaren_stake"]
            umaren["hit"] = umaren["umaren_hit"]
            umaren["return_yen"] = _num(umaren["umaren_pay"]).fillna(0.0).where(umaren["hit"], 0.0) * umaren["stake_yen"] / 100.0
            frames.append(umaren)

    if params["umatan_anchor_stake"] > 0:
        umatan_a = base[
            _num(base["anchor_win_score"]).ge(params["umatan_anchor_win_min"])
            & _num(base["anchor_quinella_score"]).ge(params.get("anchor_quinella_min", 0.0))
            & _num(base["partner_quinella_score"]).ge(params.get("partner_quinella_min", 0.0))
            & _num(base["partner_odds"]).between(params["umatan_partner_odds_min"], params["umatan_partner_odds_max"])
            & _num(base["umatan_pay"]).ge(params.get("umatan_anchor_pay_min", 0.0))
        ].copy()
        if not umatan_a.empty:
            umatan_a["ticket_type"] = "umatan_anchor_to_partner"
            umatan_a["stake_yen"] = params["umatan_anchor_stake"]
            umatan_a["hit"] = umatan_a["umatan_anchor_hit"]
            umatan_a["return_yen"] = _num(umatan_a["umatan_pay"]).fillna(0.0).where(umatan_a["hit"], 0.0) * umatan_a["stake_yen"] / 100.0
            frames.append(umatan_a)

    if params["umatan_partner_stake"] > 0:
        umatan_p = base[
            _num(base["partner_win_score"]).ge(params["umatan_partner_win_min"])
            & _num(base["partner_odds"]).ge(params["umatan_partner_odds_min"])
        ].copy()
        if not umatan_p.empty:
            umatan_p["ticket_type"] = "umatan_partner_to_anchor"
            umatan_p["stake_yen"] = params["umatan_partner_stake"]
            umatan_p["hit"] = umatan_p["umatan_partner_hit"]
            umatan_p["return_yen"] = _num(umatan_p["umatan_pay"]).fillna(0.0).where(umatan_p["hit"], 0.0) * umatan_p["stake_yen"] / 100.0
            frames.append(umatan_p)

    tickets = pd.concat(frames, ignore_index=True, sort=False)
    tickets["ticket_key"] = tickets["ticket_type"] + ":" + tickets["pair_key"]
    return tickets


def _metrics(tickets: pd.DataFrame, label: str) -> dict:
    if tickets.empty:
        return {"policy": label, "tickets": 0, "races": 0, "roi": 0.0, "race_hit_rate": 0.0}
    stake = _num(tickets["stake_yen"]).sum()
    ret = _num(tickets["return_yen"]).sum()
    by_race = tickets.groupby("race_id").agg(stake=("stake_yen", "sum"), ret=("return_yen", "sum"), hit=("hit", "max"))
    type_counts = tickets["ticket_type"].value_counts().to_dict()
    type_roi = {}
    for ticket_type, g in tickets.groupby("ticket_type"):
        st = _num(g["stake_yen"]).sum()
        type_roi[f"{ticket_type}_roi"] = float(_num(g["return_yen"]).sum() / st) if st else 0.0
        type_roi[f"{ticket_type}_tickets"] = int(len(g))
    return {
        "policy": label,
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "avg_stake_per_race": float(by_race["stake"].mean()),
        "ticket_hit_rate": float(tickets["hit"].mean()),
        "race_hit_rate": float(by_race["hit"].mean()),
        "roi": float(ret / stake) if stake else 0.0,
        "profit_yen": float(ret - stake),
        "wide_tickets": int(type_counts.get("wide", 0)),
        "umaren_tickets": int(type_counts.get("umaren", 0)),
        "umatan_anchor_tickets": int(type_counts.get("umatan_anchor_to_partner", 0)),
        "umatan_partner_tickets": int(type_counts.get("umatan_partner_to_anchor", 0)),
        **type_roi,
    }


def _grid(allowed_modes: set[str] | None = None) -> list[dict]:
    rows = []
    for axis_min, partner_min, odds_min, odds_max, front_min, topn in product(
        [0.62, 0.70],
        [0.60, 0.66],
        [6.0, 10.0],
        [40.0, 999.0],
        [0.45, 0.60],
        [1, 2],
    ):
        for mode in [
            "wide_only",
            "wide_umaren",
            "wide_umaren_umatan",
            "wide_umaren_strict",
            "wide_umaren_umatan_anchor_strict",
        ]:
            if allowed_modes and mode not in allowed_modes:
                continue
            strict = mode in {"wide_umaren_strict", "wide_umaren_umatan_anchor_strict"}
            has_umaren = mode != "wide_only"
            has_umatan_anchor = mode in {"wide_umaren_umatan", "wide_umaren_umatan_anchor_strict"}
            has_umatan_partner = mode == "wide_umaren_umatan"
            rows.append(
                {
                    "mode": mode,
                    "axis_min": axis_min,
                    "partner_min": partner_min,
                    "partner_odds_min": odds_min,
                    "partner_odds_max": odds_max,
                    "front_min": front_min,
                    "pairs_per_race": topn,
                    "anchor_danger_max": 0.55,
                    "partner_danger_max": 0.35,
                    "wide_stake": 200 if mode != "wide_only" else 100,
                    "umaren_stake": 100 if has_umaren else 0,
                    "umaren_pair_score_min": 0.74 if strict else 0.68,
                    "umaren_quinella_min": 0.0,
                    "anchor_quinella_min": 0.0,
                    "partner_quinella_min": 0.0,
                    "umaren_partner_odds_max": 25.0 if strict else 40.0,
                    "umaren_pay_min": 1200.0 if strict else 0.0,
                    "umatan_anchor_stake": 100 if has_umatan_anchor else 0,
                    "umatan_partner_stake": 100 if has_umatan_partner else 0,
                    "umatan_anchor_win_min": 0.80 if strict else 0.72,
                    "umatan_partner_win_min": 0.72,
                    "umatan_partner_odds_min": 8.0,
                    "umatan_partner_odds_max": 35.0 if strict else 60.0,
                    "umatan_anchor_pay_min": 2500.0 if strict else 0.0,
                }
            )
            if strict:
                for quinella_min, anchor_q_min, partner_q_min in product(
                    [0.0, 0.62],
                    [0.0, 0.58],
                    [0.0, 0.54],
                ):
                    tuned = rows[-1].copy()
                    tuned["umaren_quinella_min"] = quinella_min
                    tuned["anchor_quinella_min"] = anchor_q_min
                    tuned["partner_quinella_min"] = partner_q_min
                    rows.append(tuned)
    return rows


def _selection_score(metric: dict, min_race_hit: float, min_races: int) -> float:
    if metric["races"] < min_races or metric["race_hit_rate"] < min_race_hit:
        return -999.0
    return metric["roi"] * np.sqrt(max(metric["race_hit_rate"], 0.001)) * np.log1p(metric["races"])


def _walkforward(
    universe: pd.DataFrame,
    min_race_hit: float,
    min_races: int,
    allowed_modes: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = sorted(universe["year"].dropna().astype(int).unique())
    params = _grid(allowed_modes)
    train_rows = []
    wf_rows = []
    ticket_frames = []
    for test_year in years[1:]:
        train = universe[universe["year"] < test_year]
        test = universe[universe["year"] == test_year]
        scored = []
        for i, p in enumerate(params):
            train_pairs = _select_pairs(train, p)
            train_tickets = _tickets_from_pairs(train_pairs, p)
            m = _metrics(train_tickets, f"grid_{i}_{p['mode']}")
            m.update(p)
            m["selection_score"] = _selection_score(m, min_race_hit, min_races)
            scored.append(m)
        train_grid = pd.DataFrame(scored).sort_values(["selection_score", "roi"], ascending=[False, False])
        best = train_grid.iloc[0].to_dict()
        best_params = {k: best[k] for k in params[0].keys()}
        train_grid["test_year"] = test_year
        train_rows.append(train_grid.head(50))

        test_pairs = _select_pairs(test, best_params)
        test_tickets = _tickets_from_pairs(test_pairs, best_params)
        metric = _metrics(test_tickets, f"wf_test_{test_year}_{best_params['mode']}")
        metric.update(best_params)
        metric["test_year"] = test_year
        metric["train_roi"] = float(best["roi"])
        metric["train_race_hit_rate"] = float(best["race_hit_rate"])
        metric["train_races"] = int(best["races"])
        wf_rows.append(metric)
        if not test_tickets.empty:
            tmp = test_tickets.copy()
            tmp["test_year"] = test_year
            tmp["selected_policy"] = metric["policy"]
            ticket_frames.append(tmp)
    return (
        pd.concat(train_rows, ignore_index=True, sort=False) if train_rows else pd.DataFrame(),
        pd.DataFrame(wf_rows),
        pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize dynamic wide/umaren/umatan stake allocation by race.")
    parser.add_argument("--scored-csv", default="outputs/analysis/investment_decision_features_v1/investment_features_scored.csv")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--output-dir", default="outputs/analysis/dynamic_pair_ticket_allocation_v1")
    parser.add_argument("--min-race-hit", type=float, default=0.16)
    parser.add_argument("--min-train-races", type=int, default=250)
    parser.add_argument("--modes", nargs="*", default=None, help="Optional modes: wide_only wide_umaren wide_umaren_umatan")
    args = parser.parse_args()

    scored = pd.read_csv(project_path(args.scored_csv), low_memory=False)
    scored["race_id"] = scored["race_id"].astype(str)
    wide = _load_wide(project_path(args.wide_payoff_csv))
    pairs = _load_pairs(project_path(args.raw_csv), project_path(args.config))
    universe = _pair_universe(scored, wide, pairs)
    allowed_modes = set(args.modes) if args.modes else None
    train_grid, wf_summary, wf_tickets = _walkforward(universe, args.min_race_hit, args.min_train_races, allowed_modes)
    out_dir = ensure_dir(project_path(args.output_dir))
    universe.to_csv(out_dir / "pair_candidate_universe.csv", index=False, encoding="utf-8-sig")
    train_grid.to_csv(out_dir / "walkforward_train_grid_top50.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(out_dir / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    wf_tickets.to_csv(out_dir / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(out_dir),
        "pair_candidates": int(len(universe)),
        "walkforward_summary": wf_summary.to_dict(orient="records"),
        "walkforward_total": _metrics(wf_tickets, "walkforward_total"),
        "note": "Dynamic stake allocation among wide, umaren, and umatan. Each test year uses prior-year selected conditions.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
