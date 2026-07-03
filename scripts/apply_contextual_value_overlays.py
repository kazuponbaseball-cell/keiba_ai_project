from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_operational_win_addon import _json_default, _metrics
from src.utils.paths import ensure_dir, project_path


RACE_COL = "レースID(新/馬番無)"
HORSE_NO_COL = "馬番"


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    if series.dtype == object or str(series.dtype).startswith("string"):
        series = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def _clip01(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _z_to_01(value: pd.Series, scale: float = 1.5) -> pd.Series:
    x = pd.to_numeric(value, errors="coerce").fillna(0.0).clip(-4.0, 4.0)
    return (0.5 + x / (2.0 * scale)).clip(0.0, 1.0)


def _race_rank01(values: pd.Series, race_ids: pd.Series, ascending: bool = True) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    pct = x.groupby(race_ids).rank(pct=True, ascending=ascending)
    return pct.fillna(0.5).clip(0.0, 1.0)


def _load_runner_context(paths: list[Path]) -> pd.DataFrame:
    needed = {
        RACE_COL,
        HORSE_NO_COL,
        "場所",
        "芝・ダ",
        "距離",
        "枠番",
        "年齢",
        "前距離",
        "前走上3F地点差",
        "前走上り3F順",
        "前4角",
        "前走RPCI",
        "distance_diff",
        "jockey_changed",
        "prev_corner4_position_rate",
        "front_running_tendency",
        "closing_tendency",
        "same_distance_category_top3_rate",
        "jockey_venue_top3_rate",
        "jockey_surface_top3_rate",
        "jockey_distance_top3_rate",
        "jockey_popularity_outperform_rate",
        "trainer_rotation_top3_rate",
        "trainer_rotation_popularity_outperform_rate",
        "jockey_trainer_combo_score",
        "rotation_fit_score",
        "rotation_fresh_start_flag",
        "rotation_second_after_layoff_flag",
        "rotation_third_after_layoff_flag",
        "rotation_distance_up_flag",
        "rotation_distance_down_flag",
        "rotation_big_distance_change_flag",
        "rotation_surface_switch_flag",
        "horse_late_gain_avg_past5",
        "prev_late_gain",
        "body_prev_weight",
        "body_prev_delta",
        "body_layoff_flag",
        "body_layoff_gain_flag",
        "body_layoff_loss_flag",
        "body_young_growth_gain_flag",
        "body_large_horse_flag",
        "body_very_large_horse_flag",
        "body_weight_z_in_race",
        "body_weight_percentile_in_race",
        "body_young_maturity_score",
        "body_layoff_workout_count_fit",
        "body_layoff_recent_workout_flag",
        "body_loss_with_strong_workout_flag",
        "owner_trainer_synergy_score",
        "owner_context_fit_score",
        "owner_trainer_pair_top3_rate",
        "owner_jockey_pair_top3_rate",
        "breeder_young_turf_fit_score",
        "breeder_context_fit_score",
        "breeder_northern_turf_young_flag",
        "breeder_shadai_turf_young_flag",
        "breeder_shadai_group_flag",
        "breeder_surface_top3_rate",
        "breeder_distance_top3_rate",
    }
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        header = pd.read_csv(path, nrows=0, low_memory=False)
        cols = [c for c in header.columns if c in needed]
        if RACE_COL not in cols or HORSE_NO_COL not in cols:
            continue
        frames.append(pd.read_csv(path, usecols=cols, dtype={RACE_COL: str}, low_memory=False))
    if not frames:
        return pd.DataFrame(columns=["race_id", "horse_no"])
    df = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates([RACE_COL, HORSE_NO_COL], keep="last")
    df = df.rename(columns={RACE_COL: "race_id", HORSE_NO_COL: "horse_no"})
    df["race_id"] = df["race_id"].astype(str)
    df["horse_no"] = _num(df["horse_no"], df.index).astype("Int64")
    return _add_runner_context_scores(df)


def _add_runner_context_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    race_id = out["race_id"].astype(str)
    age = _num(out.get("年齢"), idx, 0).fillna(0.0)
    distance = _num(out.get("距離"), idx, 0).fillna(0.0)
    surface = out.get("芝・ダ", pd.Series("", index=idx)).astype(str)
    venue = out.get("場所", pd.Series("", index=idx)).astype(str)
    frame_no = _num(out.get("枠番"), idx, np.nan)

    young = age.le(3).astype(float)
    route = distance.ge(1600).astype(float)
    dirt = surface.str.contains("ダ", regex=False, na=False).astype(float)
    turf = surface.str.contains("芝", regex=False, na=False).astype(float)
    small_course = venue.isin(["札幌", "函館", "福島", "小倉", "中山"]).astype(float)

    body_weight_rank = _race_rank01(out.get("body_prev_weight"), race_id, ascending=True)
    body_heavy = _clip01(
        0.45 * _clip01(out.get("body_weight_percentile_in_race"))
        + 0.35 * body_weight_rank
        + 0.20 * (_num(out.get("body_prev_weight"), idx, 0).ge(500).astype(float))
    )
    body_layoff_gain = _num(out.get("body_layoff_gain_flag"), idx, 0).fillna(0.0)
    body_layoff_loss = _num(out.get("body_layoff_loss_flag"), idx, 0).fillna(0.0)
    body_young_growth = _num(out.get("body_young_growth_gain_flag"), idx, 0).fillna(0.0)
    body_workout = _clip01(
        0.55 * _num(out.get("body_layoff_workout_count_fit"), idx, 0).fillna(0.0)
        + 0.45 * _num(out.get("body_layoff_recent_workout_flag"), idx, 0).fillna(0.0)
    )
    out["ctx_body_layoff_age_score"] = _clip01(
        0.30 * _clip01(out.get("body_young_maturity_score"))
        + 0.22 * body_heavy * (0.35 + 0.65 * young)
        + 0.18 * body_layoff_gain
        + 0.14 * body_young_growth
        + 0.12 * body_workout
        + 0.04 * _num(out.get("body_loss_with_strong_workout_flag"), idx, 0).fillna(0.0)
        - 0.20 * body_layoff_loss
    )

    out["ctx_stable_week_score"] = _clip01(
        0.28 * _clip01(out.get("trainer_rotation_top3_rate"))
        + 0.20 * _clip01(out.get("trainer_rotation_popularity_outperform_rate"))
        + 0.18 * _clip01(out.get("jockey_trainer_combo_score"))
        + 0.14 * _clip01(out.get("rotation_fit_score"))
        + 0.10 * _clip01(out.get("jockey_venue_top3_rate"))
        + 0.10 * _clip01(out.get("jockey_surface_top3_rate"))
    )

    owner_breeder = _clip01(
        0.25 * _clip01(out.get("owner_trainer_synergy_score"))
        + 0.20 * _clip01(out.get("owner_context_fit_score"))
        + 0.12 * _clip01(out.get("owner_trainer_pair_top3_rate"))
        + 0.10 * _clip01(out.get("owner_jockey_pair_top3_rate"))
        + 0.17 * _clip01(out.get("breeder_context_fit_score"))
        + 0.10 * _clip01(out.get("breeder_young_turf_fit_score"))
        + 0.06 * _clip01(out.get("breeder_surface_top3_rate"))
    )
    northern_young = _num(out.get("breeder_northern_turf_young_flag"), idx, 0).fillna(0.0).gt(0)
    shadai_young = _num(out.get("breeder_shadai_turf_young_flag"), idx, 0).fillna(0.0).gt(0)
    elite_young_turf = turf * young * (northern_young | shadai_young).astype(float)
    out["ctx_owner_breeder_synergy_score"] = _clip01(owner_breeder + 0.12 * elite_young_turf)

    fresh = _num(out.get("rotation_fresh_start_flag"), idx, 0).fillna(0.0)
    second = _num(out.get("rotation_second_after_layoff_flag"), idx, 0).fillna(0.0)
    third = _num(out.get("rotation_third_after_layoff_flag"), idx, 0).fillna(0.0)
    surface_switch = _num(out.get("rotation_surface_switch_flag"), idx, 0).fillna(0.0)
    prev_late_gain = _z_to_01(out.get("prev_late_gain"), scale=2.0)
    late_avg = _z_to_01(out.get("horse_late_gain_avg_past5"), scale=2.0)
    out["ctx_condition_rebound_score"] = _clip01(
        0.22 * second
        + 0.12 * third
        + 0.18 * _clip01(out.get("rotation_fit_score"))
        + 0.16 * prev_late_gain
        + 0.12 * late_avg
        + 0.10 * surface_switch
        + 0.10 * (1.0 - fresh)
    )

    front = _clip01(out.get("front_running_tendency"))
    prev_front = (1.0 - _clip01(out.get("prev_corner4_position_rate"))).clip(0.0, 1.0)
    prev_corner = _num(out.get("前4角"), idx, np.nan)
    gate_like = _clip01(0.55 * front + 0.30 * prev_front + 0.15 * _num(out.get("jockey_popularity_outperform_rate"), idx, 0))
    out["ctx_front_gate_reliability_score"] = _clip01(gate_like - 0.12 * prev_corner.gt(10).astype(float))

    outer = frame_no.ge(7).astype(float)
    inner = frame_no.le(2).astype(float)
    stalker = _clip01(1.0 - (front - _clip01(out.get("closing_tendency"))).abs())
    out["ctx_momingare_outside_score"] = _clip01(
        0.22 * dirt * outer
        + 0.18 * outer * small_course
        + 0.18 * stalker
        + 0.16 * _clip01(out.get("jockey_surface_top3_rate"))
        + 0.14 * body_heavy * dirt
        - 0.14 * dirt * inner * (1.0 - front)
    )

    prev_rpci = _num(out.get("前走RPCI"), idx, np.nan)
    low_rpci = ((50.0 - prev_rpci) / 8.0).clip(0.0, 1.0).fillna(0.0)
    late_rpci = ((prev_rpci - 50.0) / 8.0).clip(0.0, 1.0).fillna(0.0)
    prev_c4 = _clip01(out.get("prev_corner4_position_rate"))
    corner_gain_proxy = (1.0 - prev_c4).clip(0.0, 1.0)
    out["ctx_corner_rpci_score"] = _clip01(
        small_course
        * route
        * (
            0.34 * corner_gain_proxy
            + 0.22 * low_rpci * corner_gain_proxy
            + 0.16 * late_rpci * _clip01(out.get("horse_late_gain_avg_past5"))
            + 0.14 * _clip01(out.get("same_distance_category_top3_rate"))
            + 0.14 * _clip01(out.get("jockey_distance_top3_rate"))
        )
    )

    dist_diff = _num(out.get("distance_diff"), idx, 0.0).fillna(0.0)
    down = _num(out.get("rotation_distance_down_flag"), idx, 0).fillna(0.0)
    up = _num(out.get("rotation_distance_up_flag"), idx, 0).fillna(0.0)
    big = _num(out.get("rotation_big_distance_change_flag"), idx, 0).fillna(0.0)
    sharp_but_no_finish = _clip01(prev_late_gain * down)
    stamina_extension = _clip01(late_avg * up * route)
    out["ctx_distance_change_quality_score"] = _clip01(
        0.22 * _clip01(out.get("same_distance_category_top3_rate"))
        + 0.20 * _clip01(out.get("rotation_fit_score"))
        + 0.16 * sharp_but_no_finish
        + 0.16 * stamina_extension
        + 0.12 * _clip01(out.get("jockey_distance_top3_rate"))
        + 0.08 * _clip01(out.get("breeder_distance_top3_rate"))
        - 0.14 * big * (1.0 - _clip01(out.get("rotation_fit_score")))
        - 0.06 * (dist_diff.abs().ge(600).astype(float))
    )

    context_cols = [c for c in out.columns if c.startswith("ctx_")]
    out["ctx_runner_overlay_score"] = _clip01(out[context_cols].mean(axis=1))
    keep = ["race_id", "horse_no", *context_cols, "ctx_runner_overlay_score"]
    return out[keep]


def _merge_runner(tickets: pd.DataFrame, runners: pd.DataFrame, prefix: str, no_col: str) -> pd.DataFrame:
    runner = runners.copy()
    rename = {c: f"{prefix}_{c}" for c in runner.columns if c not in {"race_id", "horse_no"}}
    runner = runner.rename(columns=rename)
    left = tickets.copy()
    left[no_col] = _num(left.get(no_col), left.index).astype("Int64")
    return left.merge(runner, left_on=["race_id", no_col], right_on=["race_id", "horse_no"], how="left").drop(columns=["horse_no"], errors="ignore")


def _prepare_tickets(tickets: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = _num(df.get("year"), df.index, np.nan).fillna(df["race_id"].str[:4].astype(float)).astype(int)
    df["ticket_type"] = df.get("ticket_type", "").astype(str)
    df = _merge_runner(df, runners, "anchor", "anchor_no")
    df = _merge_runner(df, runners, "partner", "partner_no")

    ctx_names = [
        "ctx_body_layoff_age_score",
        "ctx_stable_week_score",
        "ctx_owner_breeder_synergy_score",
        "ctx_condition_rebound_score",
        "ctx_front_gate_reliability_score",
        "ctx_momingare_outside_score",
        "ctx_corner_rpci_score",
        "ctx_distance_change_quality_score",
    ]
    for name in ctx_names:
        a = _num(df.get(f"anchor_{name}"), df.index, 0.5).fillna(0.5)
        p = _num(df.get(f"partner_{name}"), df.index, 0.5).fillna(0.5)
        if name == "ctx_front_gate_reliability_score" and "projected_front5_prob" in df.columns:
            p = p.fillna(_num(df.get("projected_front5_prob"), df.index, 0.5))
        df[f"ticket_{name}"] = np.where(df["ticket_type"].eq("win"), a, 0.56 * a + 0.44 * p)

    existing_vertical = _clip01(
        0.30 * _num(df.get("anchor_vertical_condition_fit_score"), df.index, 0.5).fillna(0.5)
        + 0.20 * _num(df.get("anchor_surface_distance_vertical_fit_score"), df.index, 0.5).fillna(0.5)
        + 0.20 * _num(df.get("partner_vertical_condition_fit_score"), df.index, 0.5).fillna(0.5)
        + 0.15 * _num(df.get("partner_surface_distance_vertical_fit_score"), df.index, 0.5).fillna(0.5)
        + 0.15 * _num(df.get("projected_front5_prob"), df.index, 0.5).fillna(0.5)
    )
    df["ticket_existing_vertical_context_score"] = existing_vertical
    context_inputs = [f"ticket_{name}" for name in ctx_names] + ["ticket_existing_vertical_context_score"]
    weights = np.array([0.13, 0.13, 0.13, 0.16, 0.12, 0.10, 0.10, 0.15, 0.08])
    values = pd.concat([_num(df.get(c), df.index, 0.5).fillna(0.5) for c in context_inputs], axis=1)
    df["context_overlay_score"] = _clip01((values * weights).sum(axis=1) / weights.sum())
    df["context_negative_score"] = _clip01(
        0.35 * _num(df.get("anchor_vertical_overpopular_risk_score"), df.index, 0.0).fillna(0.0)
        + 0.25 * _num(df.get("anchor_vertical_condition_mismatch_score"), df.index, 0.0).fillna(0.0)
        + 0.20 * _num(df.get("partner_vertical_overpopular_risk_score"), df.index, 0.0).fillna(0.0)
        + 0.20 * _num(df.get("partner_vertical_condition_mismatch_score"), df.index, 0.0).fillna(0.0)
    )
    df["context_net_overlay_score"] = _clip01(df["context_overlay_score"] - 0.20 * df["context_negative_score"])
    return df


def _apply_policy(tickets: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = tickets.copy()
    score = _num(df.get("context_net_overlay_score"), df.index, 0.5).fillna(0.5)
    mask = score.ge(params["min_score"])
    if params["umaren_min_score"] is not None:
        mask &= (~df["ticket_type"].eq("umaren")) | score.ge(params["umaren_min_score"])
    df = df[mask].copy()
    if df.empty:
        return df

    original_stake = _num(df.get("stake_yen"), df.index, 100.0).replace(0, np.nan)
    payout_per100 = _num(df.get("return_yen"), df.index, 0.0) / original_stake * 100.0
    units = _num(df.get("stake_units"), df.index, 1.0).fillna(_num(df.get("stake_yen"), df.index, 100.0) / 100.0)
    mult = np.select(
        [score.loc[df.index].ge(params["top_score"]), score.loc[df.index].ge(params["high_score"])],
        [params["top_mult"], params["high_mult"]],
        default=params["base_mult"],
    )
    if params["reduce_low"]:
        mult = np.where(score.loc[df.index].lt(params["high_score"]), np.minimum(mult, 0.75), mult)
    df["stake_units"] = np.floor((units * mult).clip(0, params["max_units_per_ticket"]) + 1e-9)
    df = df[df["stake_units"].gt(0)].copy()
    if df.empty:
        return df
    df["stake_yen"] = df["stake_units"] * 100.0
    df["return_yen"] = np.where(df["hit"].astype(bool), payout_per100.loc[df.index] * df["stake_yen"] / 100.0, 0.0)

    df["_priority"] = (
        df["context_net_overlay_score"].rank(method="first", ascending=False)
        + _num(df.get("ticket_sizing_score"), df.index, 0.0).rank(method="first", ascending=False) / 10000.0
        + _num(df.get("late_value_survives_score"), df.index, 0.0).rank(method="first", ascending=False) / 100000000.0
    )
    keep: list[int] = []
    for _, group in df.sort_values(["race_id", "_priority"], ascending=[True, True]).groupby("race_id", sort=False):
        running = 0.0
        for idx, row in group.iterrows():
            next_units = float(row["stake_units"])
            if running + next_units > params["max_units_per_race"]:
                continue
            running += next_units
            keep.append(idx)
    df = df.loc[keep].copy().drop(columns=["_priority"], errors="ignore")
    df["operation_profile"] = df.get("operation_profile", "").astype(str) + "_context_overlay"
    df["operation_profile_label"] = df.get("operation_profile_label", "").astype(str) + "+文脈"
    return df


def _grid() -> list[dict]:
    out: list[dict] = []
    for min_score, umaren_min_score, high_score, top_score, base_mult, high_mult, top_mult, max_units, reduce_low in product(
        [0.0, 0.42, 0.48, 0.54],
        [None, 0.48, 0.54],
        [0.55, 0.62],
        [0.70, 0.78],
        [0.75, 1.0],
        [1.0, 1.25],
        [1.0, 1.5],
        [5, 7],
        [False, True],
    ):
        if high_score >= top_score:
            continue
        out.append(
            {
                "min_score": min_score,
                "umaren_min_score": umaren_min_score,
                "high_score": high_score,
                "top_score": top_score,
                "base_mult": base_mult,
                "high_mult": high_mult,
                "top_mult": top_mult,
                "max_units_per_ticket": max_units,
                "max_units_per_race": 10,
                "reduce_low": reduce_low,
            }
        )
    return out


def _score(metrics: dict) -> float:
    return metrics["profit_yen"] + 18000.0 * (metrics["roi"] - 1.0) + 6000.0 * metrics["race_hit_rate"] + metrics["max_drawdown_yen"] * 0.35


def _choose(train: pd.DataFrame, min_races: int) -> tuple[dict | None, pd.DataFrame]:
    rows: list[dict] = []
    best_params = None
    best_score = -np.inf
    for i, params in enumerate(_grid()):
        selected = _apply_policy(train, params)
        metrics = _metrics(selected, f"policy_{i}")
        if metrics["races"] < min_races or metrics["race_hit_rate"] < 0.12 or metrics["roi"] < 1.1:
            continue
        score = _score(metrics)
        rows.append({"policy_id": i, "score": score, **params, **metrics})
        if score > best_score:
            best_score = score
            best_params = params
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("score", ascending=False)
    return best_params, table


def _segment_report(tickets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    features = [
        "context_net_overlay_score",
        "ticket_ctx_body_layoff_age_score",
        "ticket_ctx_stable_week_score",
        "ticket_ctx_owner_breeder_synergy_score",
        "ticket_ctx_condition_rebound_score",
        "ticket_ctx_front_gate_reliability_score",
        "ticket_ctx_momingare_outside_score",
        "ticket_ctx_corner_rpci_score",
        "ticket_ctx_distance_change_quality_score",
    ]
    for feature in features:
        values = _num(tickets.get(feature), tickets.index)
        try:
            bins = pd.qcut(values, 4, duplicates="drop")
        except ValueError:
            continue
        tmp = tickets.assign(_bin=bins)
        for (ticket_type, bin_label), group in tmp.groupby(["ticket_type", "_bin"], observed=True):
            rows.append({"feature": feature, "ticket_type": ticket_type, "bin": str(bin_label), **_metrics(group, str(bin_label))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply contextual value overlays: body, stable/owner/breeder, rebound, gate, corner/RPCI, distance change.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/roi_mode_stake_sizing_v1/stake_sized_ticket_profiles.csv")
    parser.add_argument(
        "--runner-feature-csv",
        nargs="*",
        default=[
            "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_breeder_enriched/train_features_with_same_day_bias_v3_retro_body_breeder.csv",
            "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_breeder_enriched/test_features_with_same_day_bias_v3_retro_body_breeder.csv",
        ],
    )
    parser.add_argument("--output-dir", default="outputs/analysis/contextual_value_overlays_v1")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--min-train-races", type=int, default=220)
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    runners = _load_runner_context([project_path(p) for p in args.runner_feature_csv])
    enriched = _prepare_tickets(tickets, runners)
    train = enriched[enriched["year"].eq(args.train_year)].copy()
    test = enriched[enriched["year"].eq(args.test_year)].copy()
    params, candidates = _choose(train, args.min_train_races)
    selected = _apply_policy(enriched, params) if params else enriched.iloc[0:0].copy()

    out_dir = ensure_dir(project_path(args.output_dir))
    enriched.to_csv(out_dir / "context_enriched_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out_dir / "context_overlay_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    _segment_report(enriched).to_csv(out_dir / "context_overlay_segments.csv", index=False, encoding="utf-8-sig")
    if not candidates.empty:
        candidates.head(200).to_csv(out_dir / "candidate_policies.csv", index=False, encoding="utf-8-sig")

    coverage_cols = [c for c in enriched.columns if c.startswith("anchor_ctx_") or c.startswith("partner_ctx_")]
    summary = {
        "input": {
            "tickets_csv": args.tickets_csv,
            "runner_feature_csv": args.runner_feature_csv,
            "train_year": args.train_year,
            "test_year": args.test_year,
        },
        "runner_context_rows": int(len(runners)),
        "context_column_coverage": {c: float(enriched[c].notna().mean()) for c in coverage_cols[:20]},
        "selected_params": params,
        "base_all": _metrics(enriched, "base_all"),
        "context_all": _metrics(selected, "context_overlay_all"),
        "base_train": _metrics(train, "base_train"),
        "context_train": _metrics(selected[selected["year"].eq(args.train_year)], "context_overlay_train"),
        "base_test": _metrics(test, "base_test"),
        "context_test": _metrics(selected[selected["year"].eq(args.test_year)], "context_overlay_test"),
    }
    summary["adoption_check"] = {
        "all_roi_improves": summary["context_all"]["roi"] > summary["base_all"]["roi"],
        "all_profit_improves": summary["context_all"]["profit_yen"] > summary["base_all"]["profit_yen"],
        "test_roi_improves": summary["context_test"]["roi"] > summary["base_test"]["roi"],
        "test_profit_improves": summary["context_test"]["profit_yen"] > summary["base_test"]["profit_yen"],
    }
    pd.DataFrame(
        [
            summary["base_all"],
            summary["context_all"],
            summary["base_train"],
            summary["context_train"],
            summary["base_test"],
            summary["context_test"],
        ]
    ).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
