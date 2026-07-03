from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_ticket_portfolio import _pair_tickets, _trifecta_tickets, _trio_tickets
from scripts.evaluate_ticket_strategies import _add_model_columns, _col, _num
from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


def _existing_raw_csv(path_text: str) -> Path:
    path = project_path(path_text)
    if path.exists():
        return path
    files = list(project_path("date/raw").glob("*.csv"))
    if not files:
        raise FileNotFoundError(path_text)
    return files[0]


def _load_payoffs(raw_csv: Path, race_col: str, encoding: str) -> pd.DataFrame:
    columns = [
        race_col,
        "確定着順",
        "単勝配当",
        "複勝配当",
        "馬連",
        "馬単",
        "３連複",
        "３連単",
    ]
    raw = pd.read_csv(raw_csv, encoding=encoding, usecols=lambda c: c in columns, low_memory=False)
    raw["rank_num"] = _num(raw["確定着順"])
    for col in ["単勝配当", "複勝配当", "馬連", "馬単", "３連複", "３連単"]:
        raw[col] = _num(raw[col]).fillna(0.0)

    race_pay = (
        raw[raw["rank_num"].isin([1, 2, 3])]
        .groupby(race_col, as_index=False)
        .agg(
            umaren_pay=("馬連", "max"),
            umatan_pay=("馬単", "max"),
            trio_pay=("３連複", "max"),
            trifecta_pay=("３連単", "max"),
        )
    )
    horse_pay = raw[[race_col, "rank_num", "単勝配当", "複勝配当"]].copy()
    return race_pay, horse_pay


def _select_horse_cols(df: pd.DataFrame, race_col: str) -> list[str]:
    return [race_col, "horse_name_for_ticket", "ai_rank", "popularity_num", "odds_decimal", "rank_num", "target_top3"]


def _one_per_race(df: pd.DataFrame, mask: pd.Series, race_col: str) -> pd.DataFrame:
    return (
        df[mask]
        .sort_values([race_col, "ai_rank", "popularity_num", "odds_decimal"], ascending=[True, True, True, False])
        .groupby(race_col, as_index=False)
        .head(1)
    )


def _role_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "core_anchor": (df["ai_rank"] == 1) & (df["pop_rank"] <= 3) & (df["ai_score_gap_to_second"] >= 0.05),
        "clear_head": (df["ai_rank"] == 1) & (df["ai_score_gap_to_second"] >= 0.10),
        "head_or_anchor": (df["ai_rank"] == 1) & (df["ai_score_gap_to_second"] >= 0.05),
        "value_longshot": (df["ai_rank"] <= 5) & (df["popularity_num"] >= 10),
        "value_market_gap": (df["ai_rank"] <= 5) & (df["ai_pop_gap"] <= -5),
        "value_top3_gap": (df["ai_rank"] <= 3) & (df["ai_pop_gap"] <= -5),
    }


def _single_tickets(scored: pd.DataFrame, race_col: str) -> pd.DataFrame:
    masks = _role_masks(scored)
    frames = []
    specs = [
        ("win_core_anchor", "win", masks["core_anchor"]),
        ("place_core_anchor", "place", masks["core_anchor"]),
        ("win_clear_head", "win", masks["clear_head"]),
        ("place_clear_head", "place", masks["clear_head"]),
        ("win_value_top3_gap", "win", masks["value_top3_gap"]),
    ]
    for strategy, ticket_type, mask in specs:
        base = _one_per_race(scored, mask, race_col)[_select_horse_cols(scored, race_col)].copy()
        if base.empty:
            continue
        base = base.rename(
            columns={
                "horse_name_for_ticket": "a_horse",
                "ai_rank": "a_ai_rank",
                "popularity_num": "a_popularity",
                "odds_decimal": "a_odds",
                "rank_num": "a_finish",
                "target_top3": "a_top3",
            }
        )
        base["strategy"] = strategy
        base["ticket_type"] = ticket_type
        frames.append(base)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _candidate_tickets(scored: pd.DataFrame, payoffs: pd.DataFrame, horse_pay: pd.DataFrame, race_col: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    singles = _single_tickets(scored, race_col)
    if not singles.empty:
        singles = singles.merge(horse_pay, left_on=[race_col, "a_finish"], right_on=[race_col, "rank_num"], how="left")
        singles["hit"] = np.where(singles["ticket_type"].eq("win"), singles["a_finish"].eq(1), singles["a_finish"].le(3))
        singles["pay_per100"] = np.where(
            singles["ticket_type"].eq("win"),
            singles["単勝配当"].fillna(0.0),
            singles["複勝配当"].fillna(0.0),
        )
        frames.append(singles)

    pair_specs = [
        ("umaren_core_anchor_to_value_longshot", "umaren", "core_anchor", "value_longshot", 1),
        ("umatan_core_anchor_to_value_longshot", "umatan_a_to_b", "core_anchor", "value_longshot", 1),
        ("umatan_value_longshot_to_core_anchor", "umatan_b_to_a", "core_anchor", "value_longshot", 1),
        ("umaren_core_anchor_to_value_gap", "umaren", "core_anchor", "value_top3_gap", 1),
        ("umatan_core_anchor_to_value_gap", "umatan_a_to_b", "core_anchor", "value_top3_gap", 1),
        ("umaren_head_anchor_to_market_gap", "umaren", "head_or_anchor", "value_market_gap", 2),
    ]
    for strategy, ticket_type, anchor_role, partner_role, topn in pair_specs:
        tickets = _pair_tickets(scored, race_col, anchor_role, partner_role, partner_topn=topn)
        if tickets.empty:
            continue
        tickets = tickets.merge(payoffs, on=race_col, how="left")
        tickets["strategy"] = strategy
        tickets["ticket_type"] = ticket_type
        if ticket_type == "umaren":
            tickets["hit"] = tickets["umaren_hit"].eq(1)
            tickets["pay_per100"] = tickets["umaren_pay"].fillna(0.0)
        elif ticket_type == "umatan_a_to_b":
            tickets["hit"] = tickets["umatan_a_to_b_hit"].eq(1)
            tickets["pay_per100"] = tickets["umatan_pay"].fillna(0.0)
        else:
            tickets["hit"] = tickets["umatan_b_to_a_hit"].eq(1)
            tickets["pay_per100"] = tickets["umatan_pay"].fillna(0.0)
        frames.append(tickets)

    trio = _trio_tickets(scored, race_col, partner_topn=2, fill_topn=3)
    if not trio.empty:
        trio = trio.merge(payoffs, on=race_col, how="left")
        trio["strategy"] = "trio_core_anchor_value_fill_conservative"
        trio["ticket_type"] = "trio"
        trio["hit"] = trio["trio_hit"].eq(1)
        trio["pay_per100"] = trio["trio_pay"].fillna(0.0)
        frames.append(trio)

    trifecta = _trifecta_tickets(scored, race_col, value_first=False, partner_topn=2, fill_topn=4)
    if not trifecta.empty:
        trifecta = trifecta.merge(payoffs, on=race_col, how="left")
        trifecta["strategy"] = "trifecta_clear_head_value_fill"
        trifecta["ticket_type"] = "trifecta"
        trifecta["hit"] = trifecta["trifecta_hit"].eq(1)
        trifecta["pay_per100"] = trifecta["trifecta_pay"].fillna(0.0)
        frames.append(trifecta)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["return_per100"] = np.where(out["hit"], out["pay_per100"], 0.0)
    out["candidate_id"] = np.arange(len(out))
    return out


def _strategy_roi(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in candidates.groupby("strategy"):
        rows.append(
            {
                "strategy": strategy,
                "ticket_type": g["ticket_type"].iloc[0],
                "tickets": int(len(g)),
                "races": int(g["race_id"].nunique()) if "race_id" in g else int(g.iloc[:, 0].nunique()),
                "hit_rate": float(g["hit"].mean()),
                "roi": float(g["return_per100"].sum() / (len(g) * 100.0)) if len(g) else 0.0,
                "avg_pay_hit": float(g.loc[g["hit"], "pay_per100"].mean()) if int(g["hit"].sum()) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("roi", ascending=False)


def _build_portfolio(
    candidates: pd.DataFrame,
    race_col: str,
    strategy_rank: pd.DataFrame,
    *,
    budget_yen: int,
    force_full_budget: bool,
    min_calib_roi: float,
    min_hit_rate: float = 0.0,
    max_tickets_per_race: int | None = None,
    prefer_hit_rate: bool = False,
) -> pd.DataFrame:
    priority = strategy_rank[(strategy_rank["roi"] >= min_calib_roi) & (strategy_rank["hit_rate"] >= min_hit_rate)].copy()
    if priority.empty:
        priority = strategy_rank.head(3).copy()
    roi_map = priority.set_index("strategy")["roi"].to_dict()
    hit_map = priority.set_index("strategy")["hit_rate"].to_dict()
    selected = candidates[candidates["strategy"].isin(roi_map)].copy()
    selected["strategy_roi"] = selected["strategy"].map(roi_map).fillna(0.0)
    selected["strategy_hit_rate"] = selected["strategy"].map(hit_map).fillna(0.0)
    sort_cols = [race_col, "strategy_hit_rate", "strategy_roi", "ticket_type"] if prefer_hit_rate else [race_col, "strategy_roi", "strategy_hit_rate", "ticket_type"]
    selected = selected.sort_values(sort_cols, ascending=[True, False, False, True])

    rows = []
    for race_id, g in selected.groupby(race_col):
        g = g.drop_duplicates(["strategy", "ticket_type", "a_horse", "b_horse", "c_horse"], keep="first")
        max_tickets = max_tickets_per_race or max(1, budget_yen // 100)
        max_tickets = max(1, min(max_tickets, budget_yen // 100))
        g = g.head(max_tickets).copy()
        if g.empty:
            continue
        if force_full_budget:
            base = budget_yen // len(g)
            stake = max(100, int(base // 100) * 100)
            g["stake_yen"] = stake
            diff = budget_yen - int(g["stake_yen"].sum())
            if diff >= 100:
                add_count = diff // 100
                top_idx = g.index[:add_count]
                g.loc[top_idx, "stake_yen"] += 100
        else:
            g["stake_yen"] = 100
        g["return_yen"] = g["return_per100"] * (g["stake_yen"] / 100.0)
        rows.append(g)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _portfolio_metrics(tickets: pd.DataFrame, race_col: str, label: str) -> dict[str, object]:
    if tickets.empty:
        return {"policy": label, "tickets": 0, "races_bet": 0, "stake_yen": 0, "return_yen": 0, "roi": 0.0}
    race_ret = tickets.groupby(race_col).agg(stake_yen=("stake_yen", "sum"), return_yen=("return_yen", "sum"))
    return {
        "policy": label,
        "tickets": int(len(tickets)),
        "races_bet": int(race_ret.shape[0]),
        "avg_tickets_per_race": float(len(tickets) / race_ret.shape[0]),
        "stake_yen": float(race_ret["stake_yen"].sum()),
        "avg_stake_per_bet_race": float(race_ret["stake_yen"].mean()),
        "return_yen": float(race_ret["return_yen"].sum()),
        "profit_yen": float(race_ret["return_yen"].sum() - race_ret["stake_yen"].sum()),
        "roi": float(race_ret["return_yen"].sum() / race_ret["stake_yen"].sum()),
        "race_hit_rate": float((race_ret["return_yen"] > 0).mean()),
        "median_race_return_yen": float(race_ret["return_yen"].median()),
        "p90_race_return_yen": float(race_ret["return_yen"].quantile(0.90)),
        "max_race_return_yen": float(race_ret["return_yen"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed-budget multi-ticket portfolios.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default="models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv",
    )
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--budget-yen", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/analysis/fixed_budget_ticket_portfolio_5000")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    encoding = config["data"].get("encoding", "cp932")
    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = _add_model_columns(df, project_path(args.model), race_col)
    raw_csv = _existing_raw_csv(args.raw_csv)
    payoffs, horse_pay = _load_payoffs(raw_csv, race_col, encoding)
    candidates = _candidate_tickets(scored, payoffs, horse_pay, race_col)

    date_col = _col(scored, ["日付S", "譌･莉牢"])
    race_dates = scored[[race_col, date_col]].drop_duplicates().copy()
    race_dates["date_key"] = pd.to_datetime(race_dates[date_col].astype(str), errors="coerce")
    if race_dates["date_key"].isna().all():
        race_dates["date_key"] = pd.to_numeric(race_dates[date_col], errors="coerce")
    split_value = race_dates["date_key"].quantile(0.5)
    calib_races = set(race_dates.loc[race_dates["date_key"] <= split_value, race_col])
    valid_races = set(race_dates.loc[race_dates["date_key"] > split_value, race_col])

    calib = candidates[candidates[race_col].isin(calib_races)].copy()
    valid = candidates[candidates[race_col].isin(valid_races)].copy()
    calib_rank = _strategy_roi(calib)
    valid_rank = _strategy_roi(valid)

    portfolios = []
    ticket_frames = []
    policy_specs = []
    for min_roi in [1.00, 1.10, 1.25, 1.50]:
        for force in [False, True]:
            policy_specs.append(
                {
                    "label": f"{'fixed' + str(args.budget_yen) if force else 'flat100'}_calib_roi_{min_roi:.2f}",
                    "force": force,
                    "min_roi": min_roi,
                    "min_hit_rate": 0.0,
                    "max_tickets": None,
                    "prefer_hit": False,
                }
            )
    policy_specs.extend(
        [
            {
                "label": f"hit_balanced_fixed{args.budget_yen}_roi090_hit05_max8",
                "force": True,
                "min_roi": 0.90,
                "min_hit_rate": 0.05,
                "max_tickets": 8,
                "prefer_hit": True,
            },
            {
                "label": f"hit_balanced_fixed{args.budget_yen}_roi090_hit10_max6",
                "force": True,
                "min_roi": 0.90,
                "min_hit_rate": 0.10,
                "max_tickets": 6,
                "prefer_hit": True,
            },
            {
                "label": f"high_hit_fixed{args.budget_yen}_roi085_hit30_max4",
                "force": True,
                "min_roi": 0.85,
                "min_hit_rate": 0.30,
                "max_tickets": 4,
                "prefer_hit": True,
            },
            {
                "label": f"very_high_hit_fixed{args.budget_yen}_roi085_hit65_max3",
                "force": True,
                "min_roi": 0.85,
                "min_hit_rate": 0.65,
                "max_tickets": 3,
                "prefer_hit": True,
            },
        ]
    )
    for spec in policy_specs:
        label = spec["label"]
        force = spec["force"]
        min_roi = spec["min_roi"]
        tickets = _build_portfolio(
            valid,
            race_col,
            calib_rank,
            budget_yen=args.budget_yen,
            force_full_budget=force,
            min_calib_roi=min_roi,
            min_hit_rate=spec["min_hit_rate"],
            max_tickets_per_race=spec["max_tickets"],
            prefer_hit_rate=spec["prefer_hit"],
        )
        portfolios.append(_portfolio_metrics(tickets, race_col, label))
        if not tickets.empty:
            tmp = tickets.copy()
            tmp["policy"] = label
            ticket_frames.append(tmp)

    out_dir = ensure_dir(project_path(args.output_dir))
    summary = pd.DataFrame(portfolios).sort_values("roi", ascending=False)
    calib_rank.to_csv(out_dir / "calibration_strategy_roi.csv", index=False, encoding="utf-8-sig")
    valid_rank.to_csv(out_dir / "validation_strategy_roi.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "portfolio_5000_summary.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(out_dir / "candidate_tickets.csv", index=False, encoding="utf-8-sig")
    if ticket_frames:
        pd.concat(ticket_frames, ignore_index=True, sort=False).to_csv(out_dir / "portfolio_ticket_details.csv", index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(out_dir),
        "calibration_races": len(calib_races),
        "validation_races": len(valid_races),
        "budget_yen": args.budget_yen,
        "top_calibration_strategies": calib_rank.head(20).to_dict(orient="records"),
        "top_validation_strategies": valid_rank.head(20).to_dict(orient="records"),
        "portfolio_summary": summary.to_dict(orient="records"),
        "note": "Wide payoffs are not available in the raw CSV, so actual ROI is evaluated with win/place, umaren, umatan, trio, and trifecta.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
