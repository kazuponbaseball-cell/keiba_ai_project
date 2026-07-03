from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_fixed_budget_ticket_portfolio import _existing_raw_csv, _load_payoffs
from scripts.evaluate_hit_first_rollover_model import (
    _apply_policy,
    _consecutive_metrics,
    _daily_distribution,
    _flat_metrics,
    _pair_candidate_tickets,
    _select_hit_first_policy,
    _single_candidate_tickets,
    _strategy_stats,
)
from scripts.evaluate_market_edge_pair_strategy import _load_wide_payoffs
from scripts.evaluate_ticket_strategies import _add_model_columns, _col
from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


def _horse_no_col(df: pd.DataFrame) -> str:
    col = _col(df, ["馬番", "umaban", "horse_number"])
    if col:
        return col
    candidates = [c for c in df.columns if "馬番" in c or "umaban" in c.lower()]
    if candidates:
        return candidates[0]
    raise KeyError("horse number column not found")


def _feature_lookup(scored: pd.DataFrame, race_col: str) -> pd.DataFrame:
    horse_no = _horse_no_col(scored)
    useful = [
        race_col,
        horse_no,
        "クラス名",
        "頭数",
        "出走頭数",
        "枠番",
        "馬番",
        "人気",
        "単勝オッズ",
        "芝・ダ",
        "距離",
        "馬場状態",
        "間隔",
        "休み明け～戦目",
        "前走人気",
        "前走馬体重",
        "前走馬体重増減",
        "前4角",
        "前走頭数",
        "前走RPCI",
        "jockey_win_rate",
        "jockey_top3_rate",
        "trainer_win_rate",
        "trainer_top3_rate",
        "jockey_venue_top3_rate",
        "trainer_venue_top3_rate",
        "horse_front_run_rate_past5",
        "horse_closer_rate_past5",
        "horse_position_volatility_past5",
        "race_front_runner_count",
        "race_closer_count",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "pace_fit_score",
        "front_advantage_score",
        "positioning_advantage_score",
        "draw_pace_fit_score",
        "bias_adjusted_recent_score",
        "same_day_bias_ready",
        "same_day_bias_volatility",
        "same_day_bias_fit_score",
        "same_day_pop_adjusted_pace_fit_score",
        "same_day_projected_front_load_score",
        "same_day_projected_closer_load_score",
        "workout_latest_days_before_race",
        "workout_latest_total_vs_trainer_z",
        "workout_latest_final1_vs_trainer_z",
        "workout_trainer_pattern_top3_rate",
        "workout_course_pattern_top3_rate",
        "same_day_front_collapse_index",
        "same_day_closer_blocked_index",
    ]
    cols = []
    for c in useful:
        if c in scored.columns and c not in cols:
            cols.append(c)
    lookup = scored[cols].copy()
    lookup[race_col] = lookup[race_col].astype(str)
    lookup[horse_no] = pd.to_numeric(lookup[horse_no], errors="coerce")
    return lookup.rename(columns={horse_no: "horse_no_for_merge"})


def _attach_features(tickets: pd.DataFrame, lookup: pd.DataFrame, race_col: str) -> pd.DataFrame:
    out = tickets.copy()
    if "horse_no" in out.columns:
        out["horse_no_for_merge"] = pd.to_numeric(out["horse_no"], errors="coerce")
    elif "a_horse_no" in out.columns:
        out["horse_no_for_merge"] = pd.to_numeric(out["a_horse_no"], errors="coerce")
    else:
        out["horse_no_for_merge"] = np.nan
    out[race_col] = out["race_id"].astype(str)
    return out.merge(lookup, on=[race_col, "horse_no_for_merge"], how="left")


def _candidate_filters(df: pd.DataFrame) -> dict[str, pd.Series]:
    filters: dict[str, pd.Series] = {}

    def num(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)

    if "odds" in df.columns:
        odds = num("odds")
        filters["odds_le_1_6"] = odds.le(1.6)
        filters["odds_le_1_7"] = odds.le(1.7)
        filters["odds_1_4_to_1_7"] = odds.ge(1.4) & odds.le(1.7)
    if "ai_score_gap_to_second" in df.columns:
        gap = num("ai_score_gap_to_second")
        for th in [0.10, 0.15, 0.20, 0.25]:
            filters[f"gap_ge_{str(th).replace('.', '_')}"] = gap.ge(th)
    if "出走頭数" in df.columns or "頭数" in df.columns:
        field = num("出走頭数") if "出走頭数" in df.columns else num("頭数")
        for th in [12, 14, 16]:
            filters[f"field_le_{th}"] = field.le(th)
    if "クラス名" in df.columns:
        cls = df["クラス名"].astype(str)
        filters["not_maiden_newcomer"] = ~cls.str.contains("未勝利|新馬", na=False)
        filters["not_newcomer"] = ~cls.str.contains("新馬", na=False)
        filters["not_handicap_like"] = ~cls.str.contains("ハンデ", na=False)
    if "馬場状態" in df.columns:
        going = df["馬場状態"].astype(str)
        filters["going_good_or_yaya"] = going.isin(["良", "稍", "稍重"])
        filters["going_good_only"] = going.eq("良")
    if "芝・ダ" in df.columns:
        surface = df["芝・ダ"].astype(str)
        filters["surface_dirt"] = surface.str.contains("ダ", na=False)
        filters["surface_turf"] = surface.str.contains("芝", na=False)
    if "枠番" in df.columns:
        frame = num("枠番")
        filters["not_outer_frame"] = frame.le(6)
        filters["not_inner_frame"] = frame.ge(2)
    if "間隔" in df.columns:
        interval = num("間隔")
        filters["interval_3_to_16"] = interval.between(3, 16)
        filters["interval_le_12"] = interval.le(12)
    for col, thresholds, direction in [
        ("jockey_top3_rate", [0.25, 0.30, 0.35], "ge"),
        ("trainer_top3_rate", [0.25, 0.30, 0.35], "ge"),
        ("jockey_venue_top3_rate", [0.25, 0.30, 0.35], "ge"),
        ("trainer_venue_top3_rate", [0.25, 0.30, 0.35], "ge"),
        ("horse_closer_rate_past5", [0.20, 0.35, 0.50], "le"),
        ("horse_position_volatility_past5", [0.25, 0.40, 0.60], "le"),
        ("race_pace_collapse_risk", [0.30, 0.50, 0.70], "le"),
        ("same_day_bias_volatility", [0.30, 0.50, 0.70], "le"),
        ("same_day_bias_fit_score", [0.0, 0.1, 0.2], "ge"),
        ("pace_fit_score", [0.0, 0.1, 0.2], "ge"),
        ("draw_pace_fit_score", [0.0, 0.1, 0.2], "ge"),
        ("workout_trainer_pattern_top3_rate", [0.25, 0.30, 0.35], "ge"),
    ]:
        if col not in df.columns:
            continue
        s = num(col)
        for th in thresholds:
            name = f"{col}_{direction}_{str(th).replace('.', '_')}"
            filters[name] = s.ge(th) if direction == "ge" else s.le(th)
    return {k: v.fillna(False) for k, v in filters.items()}


def _evaluate_filter_set(df: pd.DataFrame, mask: pd.Series, label: str) -> dict[str, object]:
    part = df[mask].copy()
    if part.empty:
        return {"filter": label, "tickets": 0, "hit_rate": 0.0}
    hits = part[part["hit"]]
    return {
        "filter": label,
        "tickets": int(len(part)),
        "races": int(part["race_id"].nunique()),
        "days": int(part["date_key"].nunique()),
        "races_per_day": float(part.groupby("date_key")["race_id"].nunique().mean()),
        "hit_rate": float(part["hit"].mean()),
        "avg_hit_return": float(hits["return_per100"].mean()) if len(hits) else 0.0,
        "median_hit_return": float(hits["return_per100"].median()) if len(hits) else 0.0,
        "roi_reference": float(part["return_per100"].sum() / (len(part) * 100.0)),
    }


def _learn_filters(calib: pd.DataFrame, *, min_tickets: int, max_filters: int) -> tuple[list[str], pd.DataFrame]:
    candidates = _candidate_filters(calib)
    current = pd.Series(True, index=calib.index)
    chosen: list[str] = []
    rows = []
    baseline = _evaluate_filter_set(calib, current, "baseline")
    current_hit = float(baseline["hit_rate"])
    for _ in range(max_filters):
        best = None
        for name, mask in candidates.items():
            if name in chosen:
                continue
            test_mask = current & mask
            row = _evaluate_filter_set(calib, test_mask, name)
            if int(row["tickets"]) < min_tickets:
                continue
            lift = float(row["hit_rate"]) - current_hit
            row["lift_vs_current"] = lift
            row["chosen_step"] = len(chosen) + 1
            if best is None or (row["hit_rate"], row["tickets"]) > (best["hit_rate"], best["tickets"]):
                best = row
        if best is None or float(best["lift_vs_current"]) < 0.015:
            break
        chosen.append(str(best["filter"]))
        current = current & candidates[str(best["filter"])]
        current_hit = float(best["hit_rate"])
        rows.append(best)
    return chosen, pd.DataFrame(rows)


def _apply_named_filters(df: pd.DataFrame, names: list[str]) -> pd.Series:
    masks = _candidate_filters(df)
    out = pd.Series(True, index=df.index)
    for name in names:
        if name in masks:
            out = out & masks[name]
    return out


def _consecutive(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cm = _consecutive_metrics(df.sort_values("sort_key"), [2, 3, 4, 5, 6])
    if not cm.empty:
        cm["policy"] = label
    return cm


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn danger filters for hit-first 1.5x-ish rollover candidates.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default="models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--test-csv", default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv")
    parser.add_argument("--raw-csv", default="date/raw/蜈ｨ遶ｶ襍ｰ鬥ｬ謌千ｸｾ.csv")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/hit_first_danger_filters_v1")
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
    policy = _select_hit_first_policy(
        _strategy_stats(calib),
        min_tickets=args.min_tickets,
        min_hit_return=args.min_hit_return,
        max_hit_return=args.max_hit_return,
        target_hit_return=args.target_hit_return,
    )
    calib_selected = _apply_policy(calib, policy, max_per_day=None)
    valid_selected = _apply_policy(valid, policy, max_per_day=None)

    lookup = _feature_lookup(scored, race_col)
    calib_enriched = _attach_features(calib_selected, lookup, race_col)
    valid_enriched = _attach_features(valid_selected, lookup, race_col)

    chosen, learning_trace = _learn_filters(calib_enriched, min_tickets=max(40, args.min_tickets // 2), max_filters=4)
    valid_mask = _apply_named_filters(valid_enriched, chosen)
    valid_filtered = valid_enriched[valid_mask].copy()
    valid_max1 = (
        valid_filtered.sort_values(["date_key", "policy_score", "calib_hit_rate"], ascending=[True, False, False])
        .groupby("date_key", as_index=False)
        .head(1)
        .sort_values("sort_key")
        .reset_index(drop=True)
    )
    valid_max2 = (
        valid_filtered.sort_values(["date_key", "policy_score", "calib_hit_rate"], ascending=[True, False, False])
        .groupby("date_key", as_index=False)
        .head(2)
        .sort_values("sort_key")
        .reset_index(drop=True)
    )

    summaries = pd.DataFrame(
        [
            _flat_metrics(valid_enriched, "baseline_hit_first"),
            _flat_metrics(valid_filtered, "danger_filtered_all"),
            _flat_metrics(valid_max1, "danger_filtered_max1_per_day"),
            _flat_metrics(valid_max2, "danger_filtered_max2_per_day"),
        ]
    )
    consecutive = pd.concat(
        [
            _consecutive(valid_enriched, "baseline_hit_first"),
            _consecutive(valid_filtered, "danger_filtered_all"),
            _consecutive(valid_max1, "danger_filtered_max1_per_day"),
            _consecutive(valid_max2, "danger_filtered_max2_per_day"),
        ],
        ignore_index=True,
        sort=False,
    )
    daily = _daily_distribution(valid_filtered)

    out_dir = ensure_dir(project_path(args.output_dir))
    policy.to_csv(out_dir / "selected_hit_first_policy.csv", index=False, encoding="utf-8-sig")
    calib_enriched.to_csv(out_dir / "calibration_selected_enriched.csv", index=False, encoding="utf-8-sig")
    valid_enriched.to_csv(out_dir / "validation_selected_enriched.csv", index=False, encoding="utf-8-sig")
    valid_filtered.to_csv(out_dir / "validation_danger_filtered.csv", index=False, encoding="utf-8-sig")
    learning_trace.to_csv(out_dir / "danger_filter_learning_trace.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(out_dir / "danger_filter_summary.csv", index=False, encoding="utf-8-sig")
    consecutive.to_csv(out_dir / "danger_filter_consecutive_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out_dir / "danger_filter_daily_distribution.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "chosen_filters": chosen,
        "learning_trace": learning_trace.to_dict(orient="records"),
        "validation_summary": summaries.to_dict(orient="records"),
        "consecutive_summary": consecutive.to_dict(orient="records"),
        "daily_occurrence": {
            "days_with_candidates": int(daily.shape[0]) if not daily.empty else 0,
            "avg_per_day": float(daily["qualifying_races"].mean()) if not daily.empty else 0.0,
            "median_per_day": float(daily["qualifying_races"].median()) if not daily.empty else 0.0,
            "max_per_day": int(daily["qualifying_races"].max()) if not daily.empty else 0,
        },
        "notes": [
            "Filters are learned on the calibration half and evaluated on the validation half.",
            "ROI remains reference-only; filters optimize hit-rate lift subject to sample size.",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
