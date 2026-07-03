from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_fixed_budget_ticket_portfolio import (
    _build_portfolio,
    _candidate_tickets,
    _existing_raw_csv,
    _load_payoffs,
    _portfolio_metrics,
    _strategy_roi,
)
from scripts.evaluate_ticket_strategies import _add_model_columns, _col, _num
from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


def _race_sets(scored: pd.DataFrame, race_col: str) -> dict[str, set]:
    top1 = scored[scored["ai_rank"].eq(1)].copy()
    return {
        "all": set(top1[race_col]),
        "core_anchor": set(top1.loc[top1["pop_rank"].le(3) & top1["ai_score_gap_to_second"].ge(0.05), race_col]),
        "clear_head": set(top1.loc[top1["ai_score_gap_to_second"].ge(0.10), race_col]),
        "clear_head_pop3": set(top1.loc[top1["pop_rank"].le(3) & top1["ai_score_gap_to_second"].ge(0.10), race_col]),
        "clear_head_odds5": set(top1.loc[top1["odds_decimal"].le(5.0) & top1["ai_score_gap_to_second"].ge(0.10), race_col]),
        "ai_gap015": set(top1.loc[top1["ai_score_gap_to_second"].ge(0.15), race_col]),
        "ai_gap020": set(top1.loc[top1["ai_score_gap_to_second"].ge(0.20), race_col]),
    }


def _race_confidence_summary(scored: pd.DataFrame, race_col: str, race_sets: dict[str, set]) -> pd.DataFrame:
    rank_col = _col(scored, ["確定着順", "遒ｺ螳夂捩鬆・"])
    top1 = scored[scored["ai_rank"].eq(1)].copy()
    top1["finish_num"] = _num(top1[rank_col])
    top1["win"] = top1["finish_num"].eq(1)
    top1["top3"] = top1["finish_num"].le(3)
    top1["win_pay"] = _num(top1[_col(top1, ["単勝配当", "蜊伜享驟榊ｽ・"])]).where(top1["win"], 0.0)
    top1["place_pay"] = _num(top1[_col(top1, ["複勝配当", "隍・享驟榊ｽ・"])]).where(top1["top3"], 0.0)
    rows = []
    for name, races in race_sets.items():
        part = top1[top1[race_col].isin(races)]
        if part.empty:
            rows.append({"confidence": name, "races": 0})
            continue
        rows.append(
            {
                "confidence": name,
                "races": int(part[race_col].nunique()),
                "top1_win_rate": float(part["win"].mean()),
                "top1_top3_rate": float(part["top3"].mean()),
                "top1_win_roi": float(part["win_pay"].sum() / (len(part) * 100.0)),
                "top1_place_roi": float(part["place_pay"].sum() / (len(part) * 100.0)),
                "avg_top1_pop": float(part["pop_rank"].mean()),
                "avg_top1_odds": float(part["odds_decimal"].mean()),
                "avg_gap": float(part["ai_score_gap_to_second"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("top1_win_roi", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ROI-focused ticket portfolio on AI-confidence races only.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default="models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv",
    )
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--budget-yen", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/analysis/confidence_only_ticket_portfolio")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    encoding = config["data"].get("encoding", "cp932")
    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = _add_model_columns(df, project_path(args.model), race_col)
    payoffs, horse_pay = _load_payoffs(_existing_raw_csv(args.raw_csv), race_col, encoding)
    candidates = _candidate_tickets(scored, payoffs, horse_pay, race_col)

    date_col = _col(scored, ["日付S", "譌･莉牢"])
    race_dates = scored[[race_col, date_col]].drop_duplicates().copy()
    race_dates["date_key"] = pd.to_datetime(race_dates[date_col].astype(str), errors="coerce")
    if race_dates["date_key"].isna().all():
        race_dates["date_key"] = pd.to_numeric(race_dates[date_col], errors="coerce")
    split_value = race_dates["date_key"].quantile(0.5)
    calib_races = set(race_dates.loc[race_dates["date_key"] <= split_value, race_col])
    valid_races = set(race_dates.loc[race_dates["date_key"] > split_value, race_col])

    race_sets = _race_sets(scored, race_col)
    confidence_summary = _race_confidence_summary(scored[scored[race_col].isin(valid_races)], race_col, race_sets)

    calib = candidates[candidates[race_col].isin(calib_races)].copy()
    valid_all = candidates[candidates[race_col].isin(valid_races)].copy()
    calib_rank = _strategy_roi(calib)

    rows = []
    ticket_frames = []
    for conf_name, conf_races in race_sets.items():
        valid = valid_all[valid_all[race_col].isin(valid_races & conf_races)].copy()
        for min_roi in [0.90, 1.00, 1.10]:
            label = f"{conf_name}_roi_focus_fixed{args.budget_yen}_calib_roi_{min_roi:.2f}"
            tickets = _build_portfolio(
                valid,
                race_col,
                calib_rank,
                budget_yen=args.budget_yen,
                force_full_budget=True,
                min_calib_roi=min_roi,
                min_hit_rate=0.0,
                prefer_hit_rate=False,
            )
            metric = _portfolio_metrics(tickets, race_col, label)
            metric["confidence"] = conf_name
            metric["min_calib_roi"] = min_roi
            rows.append(metric)
            if not tickets.empty:
                tmp = tickets.copy()
                tmp["policy"] = label
                tmp["confidence"] = conf_name
                ticket_frames.append(tmp)

    out_dir = ensure_dir(project_path(args.output_dir))
    portfolio = pd.DataFrame(rows).sort_values(["roi", "races_bet"], ascending=[False, False])
    confidence_summary.to_csv(out_dir / "confidence_top1_summary.csv", index=False, encoding="utf-8-sig")
    calib_rank.to_csv(out_dir / "calibration_strategy_roi.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(out_dir / f"confidence_portfolio_{args.budget_yen}_summary.csv", index=False, encoding="utf-8-sig")
    if ticket_frames:
        pd.concat(ticket_frames, ignore_index=True, sort=False).to_csv(out_dir / f"confidence_portfolio_{args.budget_yen}_tickets.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "budget_yen": args.budget_yen,
        "confidence_summary": confidence_summary.to_dict(orient="records"),
        "portfolio_summary": portfolio.head(30).to_dict(orient="records"),
    }
    with (out_dir / f"summary_{args.budget_yen}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
