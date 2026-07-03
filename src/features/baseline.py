from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureContract:
    numeric_features: list[str]
    categorical_features: list[str]
    banned_keywords: list[str]
    allowed_prefixes: list[str]


def contract_from_config(config: dict[str, Any]) -> FeatureContract:
    return FeatureContract(
        numeric_features=list(config["numeric_features"]) + list(config.get("generated_numeric_features", [])),
        categorical_features=list(config["categorical_features"]) + list(config.get("generated_categorical_features", [])),
        banned_keywords=list(config.get("leakage_banned_feature_keywords", [])),
        allowed_prefixes=list(config.get("leakage_allowed_prefixes", [])),
    )


def assert_no_leakage(contract: FeatureContract) -> None:
    bad: list[str] = []
    for feature in [*contract.numeric_features, *contract.categorical_features]:
        allowed = any(feature.startswith(prefix) for prefix in contract.allowed_prefixes)
        if allowed:
            continue
        if any(keyword in feature for keyword in contract.banned_keywords):
            bad.append(feature)
    if bad:
        raise ValueError(f"Potential leakage features are not allowed: {bad}")


def prepare_training_frame(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    train_cfg = config["training"]
    rank_col = data_cfg["rank_column"]
    abnormal_col = data_cfg["abnormal_column"]
    race_col = data_cfg["race_id_column"]
    date_col = data_cfg["date_column"]

    out = df.copy()
    out[rank_col] = pd.to_numeric(out[rank_col], errors="coerce")
    out[abnormal_col] = pd.to_numeric(out[abnormal_col], errors="coerce")
    out[date_col] = pd.to_numeric(out[date_col], errors="coerce")

    valid_abnormal = set(train_cfg.get("valid_abnormal_values", [0]))
    out = out[out[abnormal_col].isin(valid_abnormal)]
    race_type_col = train_cfg.get("race_type_column")
    exclude_race_types = set(train_cfg.get("exclude_race_type_values", []))
    if race_type_col and race_type_col in out.columns and exclude_race_types:
        race_types = pd.to_numeric(out[race_type_col], errors="coerce")
        out = out[~race_types.isin(exclude_race_types)]
    out = out[out[rank_col].notna()]
    out = out[out[race_col].notna()]
    out = out[out[date_col].notna()]

    field_size = out.groupby(race_col)[rank_col].transform("max")
    out = out[field_size >= 2].copy()
    field_size = out.groupby(race_col)[rank_col].transform("max")
    out["target_score"] = (field_size + 1.0 - out[rank_col]) / field_size
    out["target_win"] = (out[rank_col] == 1).astype(int)
    out["target_top3"] = (out[rank_col] <= 3).astype(int)
    out = add_general_generated_features(out, config)
    out = add_local_racing_context_features(out, config)
    out = add_jockey_trainer_rotation_features(out, config)
    out = add_time_value_features(out, config)
    out = add_horse_surface_history(out, config)
    out = add_track_condition_features(out, config)
    out = add_race_pace_features(out, config)
    out = add_deep_pace_style_features(out, config)
    out = add_lap_aptitude_features(out, config)
    out = add_bloodline_features(out, config)
    out = add_draw_bias_features(out, config)
    out = add_member_level_features(out, config)
    out = add_confirmed_opponent_form_features(out, config)
    out = add_race_relative_features(out, config)
    return out


def add_race_relative_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    race_col = config["data"]["race_id_column"]
    out = df.copy()
    specs = [
        ("past3_avg_margin", "race_prev_margin_rank_score", False),
        ("horse_turf_top3_rate", "race_surface_top3_rank_score", True),
        ("horse_dirt_top3_rate", "race_surface_top3_rank_score", True),
        ("same_distance_category_top3_rate", "race_distance_top3_rank_score", True),
        ("斤量", "race_weight_light_rank_score", False),
    ]

    out["race_prev_margin_rank_score"] = 0.0
    out["race_surface_top3_rank_score"] = 0.0
    out["race_distance_top3_rank_score"] = 0.0
    out["race_weight_light_rank_score"] = 0.0

    surface_col = config["training"].get("surface_column", "芝・ダ")
    if surface_col in out.columns:
        turf_mask = out[surface_col] == "芝"
        dirt_mask = out[surface_col] == "ダ"
    else:
        turf_mask = pd.Series(False, index=out.index)
        dirt_mask = pd.Series(False, index=out.index)

    for source, dest, higher_is_better in specs:
        if source == "horse_turf_top3_rate":
            mask = turf_mask
        elif source == "horse_dirt_top3_rate":
            mask = dirt_mask
        else:
            mask = pd.Series(True, index=out.index)
        if source not in out.columns:
            continue
        values = _numeric(out, source)
        counts = values.notna().groupby(out[race_col]).transform("sum")
        ranks = values.groupby(out[race_col]).rank(ascending=not higher_is_better, method="average")
        ranked = ((counts - ranks) / (counts - 1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out.loc[mask, dest] = ranked.loc[mask].astype(float)
    return out


def add_race_pace_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    race_col = config["data"]["race_id_column"]
    out = df.copy()

    front_candidate = (out.get("front_running_tendency", 0.0) >= 0.45).astype(float)
    closer_candidate = (out.get("closing_tendency", 0.0) >= 0.45).astype(float)
    field_size = out.groupby(race_col)[race_col].transform("size").replace(0, np.nan)

    out["race_front_runner_count"] = front_candidate.groupby(out[race_col]).transform("sum").astype(float)
    out["race_front_runner_ratio"] = (out["race_front_runner_count"] / field_size).fillna(0.0)
    out["race_closer_count"] = closer_candidate.groupby(out[race_col]).transform("sum").astype(float)
    out["race_closer_ratio"] = (out["race_closer_count"] / field_size).fillna(0.0)
    out["race_early_pressure_score"] = (
        0.7 * out["race_front_runner_ratio"] + 0.3 * out["race_front_runner_count"].clip(upper=5) / 5.0
    ).fillna(0.0)

    values = _numeric(out, "front_running_tendency")
    counts = values.notna().groupby(out[race_col]).transform("sum")
    ranks = values.groupby(out[race_col]).rank(ascending=False, method="average")
    out["front_pressure_rank_score"] = ((counts - ranks) / (counts - 1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_deep_pace_style_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]
    race_col = data_cfg["race_id_column"]
    rank_col = data_cfg["rank_column"]

    out = df.copy()
    features = [
        "horse_front_run_rate_past5",
        "horse_stalker_rate_past5",
        "horse_midpack_rate_past5",
        "horse_closer_rate_past5",
        "horse_late_gain_avg_past5",
        "horse_early_move_avg_past5",
        "horse_position_volatility_past5",
        "horse_need_lead_rate",
        "horse_can_rate_rate",
        "prev_late_gain",
        "prev_early_move",
        "race_need_lead_count",
        "race_need_lead_ratio",
        "race_stalker_count_deep",
        "race_midpack_count_deep",
        "race_deep_closer_count",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "solo_lead_potential",
        "pace_fit_score",
        "front_advantage_score",
        "closer_advantage_score",
        "positioning_advantage_score",
    ]
    for col in features:
        out[col] = 0.0

    if horse_col not in out.columns or race_col not in out.columns:
        return out

    field = _numeric(out, "出走頭数").replace(0, np.nan)
    if field.isna().all():
        field = _numeric(out, "頭数").replace(0, np.nan)
    corner4 = _numeric(out, "4角.1")
    if corner4.isna().all():
        corner4 = _numeric(out, "4角")
    corner1 = _numeric(out, "1角")
    if corner1.isna().all():
        corner1 = _numeric(out, "2角")
    if corner1.isna().all():
        corner1 = corner4

    finish = _numeric(out, rank_col)
    corner4_rate = (corner4 / field).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
    front_flag = (corner4_rate <= 0.18).astype(float)
    stalker_flag = ((corner4_rate > 0.18) & (corner4_rate <= 0.40)).astype(float)
    midpack_flag = ((corner4_rate > 0.40) & (corner4_rate < 0.70)).astype(float)
    closer_flag = (corner4_rate >= 0.70).astype(float)
    late_gain = (corner4 - finish).replace([np.inf, -np.inf], np.nan)
    early_move = (corner1 - corner4).replace([np.inf, -np.inf], np.nan)

    out["_front_flag_for_style"] = front_flag
    out["_stalker_flag_for_style"] = stalker_flag
    out["_midpack_flag_for_style"] = midpack_flag
    out["_closer_flag_for_style"] = closer_flag
    out["_late_gain_for_style"] = late_gain
    out["_early_move_for_style"] = early_move
    out["_corner4_rate_for_style"] = corner4_rate

    ordered = out.sort_values([horse_col, date_col, race_col], kind="mergesort")
    rolling_specs = [
        ("_front_flag_for_style", "horse_front_run_rate_past5", "mean"),
        ("_stalker_flag_for_style", "horse_stalker_rate_past5", "mean"),
        ("_midpack_flag_for_style", "horse_midpack_rate_past5", "mean"),
        ("_closer_flag_for_style", "horse_closer_rate_past5", "mean"),
        ("_late_gain_for_style", "horse_late_gain_avg_past5", "mean"),
        ("_early_move_for_style", "horse_early_move_avg_past5", "mean"),
        ("_corner4_rate_for_style", "horse_position_volatility_past5", "std"),
        ("_late_gain_for_style", "prev_late_gain", "last"),
        ("_early_move_for_style", "prev_early_move", "last"),
    ]
    for source, dest, agg in rolling_specs:
        values = _numeric(ordered, source)
        if agg == "last":
            rolled = values.groupby(ordered[horse_col], sort=False).shift()
        elif agg == "std":
            rolled = values.groupby(ordered[horse_col], sort=False).transform(
                lambda s: s.shift().rolling(5, min_periods=2).std(ddof=0)
            )
        else:
            rolled = values.groupby(ordered[horse_col], sort=False).transform(
                lambda s: s.shift().rolling(5, min_periods=1).mean()
            )
        out.loc[ordered.index, dest] = pd.to_numeric(rolled, errors="coerce").fillna(0.0).astype(float)

    out["horse_need_lead_rate"] = (
        0.70 * out["horse_front_run_rate_past5"] + 0.30 * (out["prev_corner4_position_rate"] <= 0.18).astype(float)
    ).clip(0.0, 1.0)
    out["horse_can_rate_rate"] = (
        out["horse_front_run_rate_past5"] + out["horse_stalker_rate_past5"]
    ).clip(0.0, 1.0)

    need_lead_candidate = (out["horse_need_lead_rate"] >= 0.45).astype(float)
    stalker_candidate = (out["horse_stalker_rate_past5"] >= 0.35).astype(float)
    midpack_candidate = (out["horse_midpack_rate_past5"] >= 0.35).astype(float)
    deep_closer_candidate = (out["horse_closer_rate_past5"] >= 0.40).astype(float)
    race_size = out.groupby(race_col)[race_col].transform("size").replace(0, np.nan)

    out["race_need_lead_count"] = need_lead_candidate.groupby(out[race_col]).transform("sum").astype(float)
    out["race_need_lead_ratio"] = (out["race_need_lead_count"] / race_size).fillna(0.0)
    out["race_stalker_count_deep"] = stalker_candidate.groupby(out[race_col]).transform("sum").astype(float)
    out["race_midpack_count_deep"] = midpack_candidate.groupby(out[race_col]).transform("sum").astype(float)
    out["race_deep_closer_count"] = deep_closer_candidate.groupby(out[race_col]).transform("sum").astype(float)

    pressure = _numeric(out, "race_early_pressure_score").fillna(0.0)
    out["race_pace_collapse_risk"] = (
        0.45 * (out["race_need_lead_count"].clip(upper=5) / 5.0)
        + 0.35 * pressure
        + 0.20 * out["race_front_runner_ratio"].fillna(0.0)
    ).clip(0.0, 1.0)
    out["race_slow_pace_risk"] = (
        1.0
        - 0.55 * out["race_need_lead_count"].clip(upper=4) / 4.0
        - 0.30 * pressure
        - 0.15 * out["race_stalker_count_deep"].clip(upper=5) / 5.0
    ).clip(0.0, 1.0)
    out["solo_lead_potential"] = (
        need_lead_candidate
        * (out["race_need_lead_count"] == 1).astype(float)
        * (0.5 + 0.5 * out["race_slow_pace_risk"])
    ).fillna(0.0)

    late_score = (
        0.55 * out["horse_closer_rate_past5"]
        + 0.25 * (out["horse_late_gain_avg_past5"].clip(lower=-3, upper=6) + 3.0) / 9.0
        + 0.20 * out["closing_tendency"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    front_score = (
        0.45 * out["horse_front_run_rate_past5"]
        + 0.25 * out["horse_stalker_rate_past5"]
        + 0.20 * (1.0 - out["prev_corner4_position_rate"].clip(0.0, 1.0))
        + 0.10 * out["solo_lead_potential"]
    ).clip(0.0, 1.0)
    out["front_advantage_score"] = (
        front_score * (0.65 * out["race_slow_pace_risk"] + 0.35 * out["solo_lead_potential"])
    ).fillna(0.0)
    out["closer_advantage_score"] = (late_score * out["race_pace_collapse_risk"]).fillna(0.0)
    out["positioning_advantage_score"] = (
        out["front_advantage_score"] + out["closer_advantage_score"] - 0.20 * out["horse_position_volatility_past5"]
    ).fillna(0.0)
    out["pace_fit_score"] = (
        0.45 * out["front_advantage_score"]
        + 0.45 * out["closer_advantage_score"]
        + 0.10 * out["horse_can_rate_rate"] * (1.0 - out["race_pace_collapse_risk"])
    ).fillna(0.0)

    return out.drop(
        columns=[
            "_front_flag_for_style",
            "_stalker_flag_for_style",
            "_midpack_flag_for_style",
            "_closer_flag_for_style",
            "_late_gain_for_style",
            "_early_move_for_style",
            "_corner4_rate_for_style",
        ],
        errors="ignore",
    )


def add_pace_scenario_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base = _numeric(out, "ai_score").fillna(0.0)
    front = _numeric(out, "front_running_tendency").fillna(0.0)
    closer = _numeric(out, "closing_tendency").fillna(0.0)
    pressure = _numeric(out, "race_early_pressure_score").fillna(0.0)
    front_rank = _numeric(out, "front_pressure_rank_score").fillna(0.0)

    out["expected_pace"] = np.select(
        [pressure >= 0.48, pressure <= 0.24],
        ["fast", "slow"],
        default="middle",
    )
    out["slow_ai_score"] = base + 0.030 * front + 0.010 * front_rank - 0.015 * closer
    out["middle_ai_score"] = base
    out["fast_ai_score"] = base - 0.020 * front + 0.035 * closer - 0.010 * front_rank
    return out


def add_same_day_bias_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "same_day_top3_inner_rate",
        "same_day_top3_middle_rate",
        "same_day_top3_outer_rate",
        "same_day_front_advantage_index",
        "same_day_stalker_advantage_index",
        "same_day_midpack_advantage_index",
        "same_day_closer_advantage_index",
        "same_day_front_pop_outperform_index",
        "same_day_stalker_pop_outperform_index",
        "same_day_midpack_pop_outperform_index",
        "same_day_closer_pop_outperform_index",
    ]
    if not any(col in out.columns for col in required):
        return out

    frame = _numeric(out, "枠番")
    bucket = np.select(
        [frame <= 3, frame.between(4, 6), frame >= 7],
        ["inner", "middle", "outer"],
        default="unknown",
    )
    out["same_day_frame_bucket"] = bucket

    ready = _numeric(out, "same_day_bias_ready", default=0.0).fillna(0.0).gt(0)
    inner = _numeric(out, "same_day_top3_inner_rate").where(ready, 1.0 / 3.0).fillna(1.0 / 3.0)
    middle = _numeric(out, "same_day_top3_middle_rate").where(ready, 1.0 / 3.0).fillna(1.0 / 3.0)
    outer = _numeric(out, "same_day_top3_outer_rate").where(ready, 1.0 / 3.0).fillna(1.0 / 3.0)
    out["same_day_frame_bias_fit_score"] = np.select(
        [bucket == "inner", bucket == "middle", bucket == "outer"],
        [inner - 1.0 / 3.0, middle - 1.0 / 3.0, outer - 1.0 / 3.0],
        default=0.0,
    )

    front = _numeric(out, "horse_front_run_rate_past5").fillna(_numeric(out, "front_running_tendency").fillna(0.0))
    stalker = _numeric(out, "horse_stalker_rate_past5").fillna(0.0)
    midpack = _numeric(out, "horse_midpack_rate_past5").fillna(0.0)
    closer = _numeric(out, "horse_closer_rate_past5").fillna(_numeric(out, "closing_tendency").fillna(0.0))

    front_adv = _numeric(out, "same_day_front_advantage_index").where(ready, 0.0).fillna(0.0)
    stalker_adv = _numeric(out, "same_day_stalker_advantage_index").where(ready, 0.0).fillna(0.0)
    midpack_adv = _numeric(out, "same_day_midpack_advantage_index").where(ready, 0.0).fillna(0.0)
    closer_adv = _numeric(out, "same_day_closer_advantage_index").where(ready, 0.0).fillna(0.0)
    front_pop_adv = _numeric(out, "same_day_front_pop_outperform_index").where(ready, 0.0).fillna(0.0)
    stalker_pop_adv = _numeric(out, "same_day_stalker_pop_outperform_index").where(ready, 0.0).fillna(0.0)
    midpack_pop_adv = _numeric(out, "same_day_midpack_pop_outperform_index").where(ready, 0.0).fillna(0.0)
    closer_pop_adv = _numeric(out, "same_day_closer_pop_outperform_index").where(ready, 0.0).fillna(0.0)

    out["same_day_pace_bias_fit_score"] = (
        front_adv * front
        + stalker_adv * stalker
        + midpack_adv * midpack
        + closer_adv * closer
    )
    out["same_day_pop_adjusted_pace_fit_score"] = (
        front_pop_adv * front
        + stalker_pop_adv * stalker
        + midpack_pop_adv * midpack
        + closer_pop_adv * closer
    )
    direction = _numeric(out, "same_day_bias_pace_direction").where(ready, 0.0).fillna(front_adv - closer_adv)
    out["same_day_projected_front_load_score"] = (-direction).clip(lower=0.0) * front
    out["same_day_projected_closer_load_score"] = direction.clip(lower=0.0) * closer
    out["same_day_adversity_fit_score"] = (
        out["same_day_projected_front_load_score"] + out["same_day_projected_closer_load_score"]
    )
    out["same_day_bias_fit_score"] = (
        out["same_day_frame_bias_fit_score"]
        + out["same_day_pace_bias_fit_score"]
        + 0.5 * out["same_day_pop_adjusted_pace_fit_score"]
    )
    return out


def add_lap_aptitude_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]
    race_col = data_cfg["race_id_column"]

    out = df.copy()
    defaults = [
        "prev_lap_pace_index",
        "prev_lap_finish_index",
        "prev_lap_sustain_index",
        "prev_lap_long_spurt_index",
        "horse_fast_lap_count_past5",
        "horse_fast_lap_score_past5",
        "horse_slow_lap_count_past5",
        "horse_slow_lap_score_past5",
        "horse_instant_lap_count_past5",
        "horse_instant_lap_score_past5",
        "horse_sustain_lap_count_past5",
        "horse_sustain_lap_score_past5",
        "horse_long_spurt_lap_count_past5",
        "horse_long_spurt_lap_score_past5",
        "lap_pace_versatility_score",
        "lap_aptitude_fit_score",
        "lap_aptitude_reliability_score",
    ]
    for lag in (1, 2, 3):
        defaults.extend(
            [
                f"past{lag}_lap_fast_success",
                f"past{lag}_lap_slow_success",
                f"past{lag}_lap_instant_success",
                f"past{lag}_lap_sustain_success",
                f"past{lag}_lap_long_spurt_success",
                f"past{lag}_lap_fast_regime",
                f"past{lag}_lap_slow_regime",
                f"past{lag}_lap_instant_regime",
                f"past{lag}_lap_sustain_regime",
                f"past{lag}_lap_long_spurt_regime",
                f"past{lag}_lap_rpci",
                f"past{lag}_lap_pci",
                f"past{lag}_lap_pci3",
                f"past{lag}_lap_target_score",
            ]
        )
    for col in defaults:
        out[col] = 0.0

    if horse_col not in out.columns:
        return out

    prev_pci = _numeric(out, "前PCI")
    prev_pci3 = _numeric(out, "前走PCI3")
    prev_rpci = _numeric(out, "前走RPCI")
    out["prev_lap_pace_index"] = ((50.0 - prev_rpci) / 10.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["prev_lap_finish_index"] = ((prev_pci - prev_rpci) / 10.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["prev_lap_sustain_index"] = (1.0 - (prev_pci - prev_rpci).abs() / 10.0).clip(0.0, 1.0).fillna(0.0)
    out["prev_lap_long_spurt_index"] = ((prev_pci3 - prev_rpci) / 10.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    current_pci = _numeric(out, "PCI")
    current_pci3 = _numeric(out, "PCI3")
    current_rpci = _numeric(out, "RPCI")
    fast_flag = (current_rpci <= 48.0).astype(float)
    slow_flag = (current_rpci >= 52.0).astype(float)
    instant_flag = ((current_pci - current_rpci) >= 3.0).astype(float)
    sustain_flag = ((current_pci - current_rpci).abs() <= 2.0).astype(float)
    long_spurt_flag = ((current_pci3 - current_rpci) >= 2.0).astype(float)

    out["_fast_lap_flag"] = fast_flag
    out["_slow_lap_flag"] = slow_flag
    out["_instant_lap_flag"] = instant_flag
    out["_sustain_lap_flag"] = sustain_flag
    out["_long_spurt_lap_flag"] = long_spurt_flag
    out["_fast_lap_regime"] = ((50.0 - current_rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)
    out["_slow_lap_regime"] = ((current_rpci - 50.0) / 5.0).clip(0.0, 1.0).fillna(0.0)
    out["_instant_lap_regime"] = ((current_pci - current_rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)
    out["_sustain_lap_regime"] = (1.0 - (current_pci - current_rpci).abs() / 4.0).clip(0.0, 1.0).fillna(0.0)
    out["_long_spurt_lap_regime"] = ((current_pci3 - current_rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)
    score = _numeric(out, "target_score").fillna(0.0)
    for source in [
        "_fast_lap_flag",
        "_slow_lap_flag",
        "_instant_lap_flag",
        "_sustain_lap_flag",
        "_long_spurt_lap_flag",
    ]:
        out[f"{source}_score"] = out[source] * score

    for source in [
        "_fast_lap_regime",
        "_slow_lap_regime",
        "_instant_lap_regime",
        "_sustain_lap_regime",
        "_long_spurt_lap_regime",
    ]:
        out[f"{source}_success"] = (out[source] * score).clip(0.0, 1.0)

    ordered = out.sort_values([horse_col, date_col, race_col], kind="mergesort")
    lag_sources = {
        "_fast_lap_regime_success": "lap_fast_success",
        "_slow_lap_regime_success": "lap_slow_success",
        "_instant_lap_regime_success": "lap_instant_success",
        "_sustain_lap_regime_success": "lap_sustain_success",
        "_long_spurt_lap_regime_success": "lap_long_spurt_success",
        "_fast_lap_regime": "lap_fast_regime",
        "_slow_lap_regime": "lap_slow_regime",
        "_instant_lap_regime": "lap_instant_regime",
        "_sustain_lap_regime": "lap_sustain_regime",
        "_long_spurt_lap_regime": "lap_long_spurt_regime",
        "RPCI": "lap_rpci",
        "PCI": "lap_pci",
        "PCI3": "lap_pci3",
        "target_score": "lap_target_score",
    }
    for lag in (1, 2, 3):
        for source_col, dest_suffix in lag_sources.items():
            values = _numeric(ordered, source_col).fillna(0.0)
            out.loc[ordered.index, f"past{lag}_{dest_suffix}"] = values.groupby(ordered[horse_col], sort=False).shift(lag)

    specs = [
        ("_fast_lap_flag", "_fast_lap_flag_score", "horse_fast_lap_count_past5", "horse_fast_lap_score_past5"),
        ("_slow_lap_flag", "_slow_lap_flag_score", "horse_slow_lap_count_past5", "horse_slow_lap_score_past5"),
        ("_instant_lap_flag", "_instant_lap_flag_score", "horse_instant_lap_count_past5", "horse_instant_lap_score_past5"),
        ("_sustain_lap_flag", "_sustain_lap_flag_score", "horse_sustain_lap_count_past5", "horse_sustain_lap_score_past5"),
        ("_long_spurt_lap_flag", "_long_spurt_lap_flag_score", "horse_long_spurt_lap_count_past5", "horse_long_spurt_lap_score_past5"),
    ]
    for flag_col, score_col, count_dest, score_dest in specs:
        flag_values = _numeric(ordered, flag_col).fillna(0.0)
        score_values = _numeric(ordered, score_col).fillna(0.0)
        counts = flag_values.groupby(ordered[horse_col], sort=False).transform(
            lambda s: s.shift().rolling(5, min_periods=1).sum()
        )
        score_sums = score_values.groupby(ordered[horse_col], sort=False).transform(
            lambda s: s.shift().rolling(5, min_periods=1).sum()
        )
        denom = counts.replace(0, np.nan)
        out.loc[ordered.index, count_dest] = pd.to_numeric(counts, errors="coerce").fillna(0.0).astype(float)
        out.loc[ordered.index, score_dest] = (score_sums / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    out["lap_pace_versatility_score"] = (
        0.50 * out[["horse_fast_lap_score_past5", "horse_slow_lap_score_past5"]].max(axis=1)
        + 0.50 * out[["horse_instant_lap_score_past5", "horse_sustain_lap_score_past5", "horse_long_spurt_lap_score_past5"]].max(axis=1)
    ).fillna(0.0)
    out["lap_aptitude_reliability_score"] = (
        (
            out["horse_fast_lap_count_past5"]
            + out["horse_slow_lap_count_past5"]
            + out["horse_instant_lap_count_past5"]
            + out["horse_sustain_lap_count_past5"]
            + out["horse_long_spurt_lap_count_past5"]
        ).clip(upper=5)
        / 5.0
    ).fillna(0.0)

    collapse = _numeric(out, "race_pace_collapse_risk").fillna(0.0).clip(0.0, 1.0)
    slow = _numeric(out, "race_slow_pace_risk").fillna(0.0).clip(0.0, 1.0)
    pace_fit = _numeric(out, "pace_fit_score").fillna(0.0).clip(0.0, 1.0)
    out["lap_aptitude_fit_score"] = (
        0.30 * collapse * out["horse_fast_lap_score_past5"]
        + 0.22 * slow * out["horse_slow_lap_score_past5"]
        + 0.18 * pace_fit * out["horse_instant_lap_score_past5"]
        + 0.15 * out["horse_sustain_lap_score_past5"]
        + 0.15 * out["horse_long_spurt_lap_score_past5"]
    ).fillna(0.0)

    return out.drop(
        columns=[
            "_fast_lap_flag",
            "_slow_lap_flag",
            "_instant_lap_flag",
            "_sustain_lap_flag",
            "_long_spurt_lap_flag",
            "_fast_lap_flag_score",
            "_slow_lap_flag_score",
            "_instant_lap_flag_score",
            "_sustain_lap_flag_score",
            "_long_spurt_lap_flag_score",
            "_fast_lap_regime",
            "_slow_lap_regime",
            "_instant_lap_regime",
            "_sustain_lap_regime",
            "_long_spurt_lap_regime",
            "_fast_lap_regime_success",
            "_slow_lap_regime_success",
            "_instant_lap_regime_success",
            "_sustain_lap_regime_success",
            "_long_spurt_lap_regime_success",
        ],
        errors="ignore",
    )


def _numeric(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = df[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def _distance_category(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    bins = [-np.inf, 1300, 1600, 2000, 2400, np.inf]
    labels = ["sprint", "mile", "middle", "classic", "long"]
    return pd.cut(numeric, bins=bins, labels=labels).astype("string").fillna("unknown")


def _id_key(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = numeric.round(0).astype("Int64").astype("string")
    return out.where(numeric.notna(), pd.NA)


def _rolling_previous_mean(ordered: pd.DataFrame, group_col: str, value_col: str, window: int = 3) -> pd.Series:
    return ordered.groupby(group_col, sort=False)[value_col].transform(
        lambda s: s.shift().rolling(window, min_periods=1).mean()
    )


def _add_previous_group_rates(
    out: pd.DataFrame,
    *,
    group_cols: list[str],
    prefix: str,
    sort_cols: list[str],
) -> pd.DataFrame:
    out[f"{prefix}_starts"] = 0.0
    out[f"{prefix}_top3_rate"] = 0.0
    out[f"{prefix}_avg_score"] = 0.0
    if not {"target_top3", "target_score"}.issubset(out.columns):
        return out

    ordered = out.sort_values(sort_cols, kind="mergesort")
    grouped = ordered.groupby(group_cols, sort=False)
    starts = grouped.cumcount()
    top3 = grouped["target_top3"].cumsum().shift(fill_value=0)
    score = grouped["target_score"].cumsum().shift(fill_value=0.0)

    first_in_group = grouped.cumcount() == 0
    top3.loc[first_in_group] = 0
    score.loc[first_in_group] = 0.0
    denom = starts.replace(0, np.nan)

    out.loc[ordered.index, f"{prefix}_starts"] = starts.astype(float)
    out.loc[ordered.index, f"{prefix}_top3_rate"] = (top3 / denom).fillna(0.0).astype(float)
    out.loc[ordered.index, f"{prefix}_avg_score"] = (score / denom).fillna(0.0).astype(float)
    return out


def add_jockey_trainer_rotation_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]
    race_col = data_cfg["race_id_column"]

    out = df.copy()
    sort_cols = [horse_col, date_col, race_col]

    out = _add_entity_history_features(
        out,
        group_cols=["騎手コード"],
        prefix="jockey",
        sort_cols=[date_col, race_col],
    )
    out = _add_entity_history_features(
        out,
        group_cols=["調教師コード"],
        prefix="trainer",
        sort_cols=[date_col, race_col],
    )
    out = _add_entity_history_features(
        out,
        group_cols=["騎手コード", "場所"],
        prefix="jockey_venue",
        sort_cols=[date_col, race_col],
    )
    out = _add_entity_history_features(
        out,
        group_cols=["騎手コード", "芝・ダ"],
        prefix="jockey_surface",
        sort_cols=[date_col, race_col],
    )
    out = _add_entity_history_features(
        out,
        group_cols=["騎手コード", "distance_category"],
        prefix="jockey_distance",
        sort_cols=[date_col, race_col],
    )
    out = _add_entity_history_features(
        out,
        group_cols=["調教師コード", "場所"],
        prefix="trainer_venue",
        sort_cols=[date_col, race_col],
    )
    out = _add_entity_history_features(
        out,
        group_cols=["調教師コード", "芝・ダ"],
        prefix="trainer_surface",
        sort_cols=[date_col, race_col],
    )
    out = _add_entity_history_features(
        out,
        group_cols=["調教師コード", "distance_category"],
        prefix="trainer_distance",
        sort_cols=[date_col, race_col],
    )
    out = _add_entity_history_features(
        out,
        group_cols=["騎手コード", "調教師コード"],
        prefix="jockey_trainer_pair",
        sort_cols=[date_col, race_col],
    )
    out = _add_entity_history_features(
        out,
        group_cols=[horse_col, "騎手コード"],
        prefix="horse_jockey_pair",
        sort_cols=sort_cols,
    )

    interval = _numeric(out, "間隔").fillna(0.0)
    rest_start = _numeric(out, "休み明け～戦目").fillna(0.0)
    out["rotation_interval_weeks"] = interval
    out["rotation_short_rest_flag"] = ((interval > 0) & (interval <= 2)).astype(float)
    out["rotation_standard_rest_flag"] = ((interval >= 3) & (interval <= 8)).astype(float)
    out["rotation_layoff_9_16w_flag"] = ((interval >= 9) & (interval <= 16)).astype(float)
    out["rotation_long_layoff_17w_plus_flag"] = (interval >= 17).astype(float)
    out["rotation_fresh_start_flag"] = (rest_start == 1).astype(float)
    out["rotation_second_after_layoff_flag"] = (rest_start == 2).astype(float)
    out["rotation_third_after_layoff_flag"] = (rest_start == 3).astype(float)

    distance_diff = _numeric(out, "distance_diff").fillna(0.0)
    out["rotation_distance_up_flag"] = (distance_diff >= 200).astype(float)
    out["rotation_distance_down_flag"] = (distance_diff <= -200).astype(float)
    out["rotation_big_distance_change_flag"] = (distance_diff.abs() >= 400).astype(float)
    if "前芝・ダ" in out.columns and "芝・ダ" in out.columns:
        out["rotation_surface_switch_flag"] = (
            out["前芝・ダ"].astype("string") != out["芝・ダ"].astype("string")
        ).astype(float)
    else:
        out["rotation_surface_switch_flag"] = 0.0

    out["class_move_score"] = _class_level(out.get("クラス名", pd.Series("", index=out.index))) - _class_level(
        out.get("前クラス名", pd.Series("", index=out.index))
    )
    out["rotation_class_up_flag"] = (out["class_move_score"] > 0).astype(float)
    out["rotation_class_down_flag"] = (out["class_move_score"] < 0).astype(float)
    out["rotation_same_class_flag"] = (out["class_move_score"] == 0).astype(float)
    out["rotation_stress_score"] = (
        0.30 * out["rotation_long_layoff_17w_plus_flag"]
        + 0.20 * out["rotation_short_rest_flag"]
        + 0.20 * out["rotation_big_distance_change_flag"]
        + 0.15 * out["rotation_surface_switch_flag"]
        + 0.15 * out["rotation_class_up_flag"]
    ).fillna(0.0)

    out["rotation_bucket_code"] = np.select(
        [
            out["rotation_short_rest_flag"] == 1,
            out["rotation_standard_rest_flag"] == 1,
            out["rotation_layoff_9_16w_flag"] == 1,
            out["rotation_long_layoff_17w_plus_flag"] == 1,
        ],
        [1.0, 2.0, 3.0, 4.0],
        default=0.0,
    )
    out["_rotation_bucket"] = np.select(
        [
            out["rotation_short_rest_flag"] == 1,
            out["rotation_standard_rest_flag"] == 1,
            out["rotation_layoff_9_16w_flag"] == 1,
            out["rotation_long_layoff_17w_plus_flag"] == 1,
        ],
        ["short", "standard", "layoff", "long_layoff"],
        default="unknown",
    )
    out = _add_entity_history_features(
        out,
        group_cols=["調教師コード", "_rotation_bucket"],
        prefix="trainer_rotation",
        sort_cols=[date_col, race_col],
    )
    out = _add_entity_history_features(
        out,
        group_cols=["騎手コード", "_rotation_bucket"],
        prefix="jockey_rotation",
        sort_cols=[date_col, race_col],
    )

    out["jockey_trainer_combo_score"] = (
        0.40 * out["jockey_top3_rate"]
        + 0.30 * out["trainer_top3_rate"]
        + 0.20 * out["jockey_trainer_pair_top3_rate"]
        + 0.10 * out["jockey_popularity_outperform_rate"]
    ).fillna(0.0)
    out["rotation_fit_score"] = (
        0.45 * out["trainer_rotation_top3_rate"]
        + 0.25 * out["jockey_rotation_top3_rate"]
        + 0.20 * out["rotation_standard_rest_flag"]
        + 0.10 * out["rotation_class_down_flag"]
        - 0.20 * out["rotation_stress_score"]
    ).fillna(0.0)

    return out.drop(columns=["_rotation_bucket"], errors="ignore")


def add_draw_bias_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]
    race_col = data_cfg["race_id_column"]
    surface_col = config["training"].get("surface_column", "芝・ダ")

    out = df.copy()
    defaults = [
        "frame_bias_starts",
        "frame_bias_top3_rate",
        "frame_bias_avg_score",
        "horse_number_bias_starts",
        "horse_number_bias_top3_rate",
        "horse_number_bias_avg_score",
        "course_condition_draw_avg_score",
        "frame_bias_score",
        "horse_number_bias_score",
        "current_draw_advantage_score",
        "draw_bias_rank_score",
        "draw_pace_fit_score",
        "prev_draw_advantage_score",
        "past3_avg_draw_advantage",
        "past3_bias_resistant_score",
        "past3_bias_excuse_score",
        "past3_draw_pace_adversity_score",
        "bias_adjusted_recent_score",
    ]
    for col in defaults:
        out[col] = 0.0

    frame = _numeric(out, "枠番")
    horse_number = _numeric(out, "馬番")
    field_size = _numeric(out, "出走頭数").replace(0, np.nan)
    if field_size.isna().all():
        field_size = _numeric(out, "頭数").replace(0, np.nan)

    out["_frame_bucket"] = np.select(
        [frame <= 2, frame >= 7],
        ["inner", "outer"],
        default="middle",
    )
    horse_number_rate = (horse_number / field_size).replace([np.inf, -np.inf], np.nan)
    out["_horse_number_bucket"] = np.select(
        [horse_number_rate <= 0.33, horse_number_rate >= 0.67],
        ["inner", "outer"],
        default="middle",
    )

    condition_cols = [col for col in ["場所", surface_col, "distance_category", "馬場状態"] if col in out.columns]
    sort_cols = [date_col, race_col]
    out = _add_entity_history_features(
        out,
        group_cols=condition_cols,
        prefix="course_condition_draw",
        sort_cols=sort_cols,
    )
    out = _add_entity_history_features(
        out,
        group_cols=[*condition_cols, "_frame_bucket"],
        prefix="frame_bias",
        sort_cols=sort_cols,
    )
    out = _add_entity_history_features(
        out,
        group_cols=[*condition_cols, "_horse_number_bucket"],
        prefix="horse_number_bias",
        sort_cols=sort_cols,
    )

    out["course_condition_draw_avg_score"] = out["course_condition_draw_avg_score"].fillna(0.0)
    out["frame_bias_score"] = (
        out["frame_bias_avg_score"] - out["course_condition_draw_avg_score"]
    ).fillna(0.0)
    out["horse_number_bias_score"] = (
        out["horse_number_bias_avg_score"] - out["course_condition_draw_avg_score"]
    ).fillna(0.0)
    out["current_draw_advantage_score"] = (
        0.65 * out["frame_bias_score"] + 0.35 * out["horse_number_bias_score"]
    ).fillna(0.0)

    values = out["current_draw_advantage_score"]
    counts = values.notna().groupby(out[race_col]).transform("sum")
    ranks = values.groupby(out[race_col]).rank(ascending=False, method="average")
    out["draw_bias_rank_score"] = ((counts - ranks) / (counts - 1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    front = _numeric(out, "front_running_tendency").fillna(0.0).clip(0.0, 1.0)
    closer = _numeric(out, "closing_tendency").fillna(0.0).clip(0.0, 1.0)
    collapse = _numeric(out, "race_pace_collapse_risk").fillna(0.0).clip(0.0, 1.0)
    slow = _numeric(out, "race_slow_pace_risk").fillna(0.0).clip(0.0, 1.0)
    draw = out["current_draw_advantage_score"].clip(-0.25, 0.25)
    out["draw_pace_fit_score"] = (
        draw + 0.08 * front * slow + 0.08 * closer * collapse - 0.08 * front * collapse - 0.08 * closer * slow
    ).fillna(0.0)

    draw_adversity = (-out["current_draw_advantage_score"]).clip(lower=0.0, upper=0.25)
    pace_adversity = (0.5 * front * collapse + 0.5 * closer * slow).clip(0.0, 1.0)
    out["_draw_pace_adversity_for_history"] = (draw_adversity + 0.12 * pace_adversity).fillna(0.0)
    out["_bias_resistant_for_history"] = (
        out["_draw_pace_adversity_for_history"] * _numeric(out, "target_score").fillna(0.0)
    ).fillna(0.0)
    out["_bias_excuse_for_history"] = (
        out["_draw_pace_adversity_for_history"] * (1.0 - _numeric(out, "target_score").fillna(0.0))
    ).fillna(0.0)

    ordered = out.sort_values([horse_col, date_col, race_col], kind="mergesort")
    rolling_specs = [
        ("current_draw_advantage_score", "prev_draw_advantage_score", "last"),
        ("current_draw_advantage_score", "past3_avg_draw_advantage", "mean"),
        ("_bias_resistant_for_history", "past3_bias_resistant_score", "mean"),
        ("_bias_excuse_for_history", "past3_bias_excuse_score", "mean"),
        ("_draw_pace_adversity_for_history", "past3_draw_pace_adversity_score", "mean"),
    ]
    for source, dest, agg in rolling_specs:
        source_values = _numeric(ordered, source)
        if agg == "last":
            rolled = source_values.groupby(ordered[horse_col], sort=False).shift()
        else:
            rolled = source_values.groupby(ordered[horse_col], sort=False).transform(
                lambda s: s.shift().rolling(3, min_periods=1).mean()
            )
        out.loc[ordered.index, dest] = pd.to_numeric(rolled, errors="coerce").fillna(0.0).astype(float)

    out["bias_adjusted_recent_score"] = (
        _numeric(out, "past3_avg_score").fillna(0.0)
        + 0.50 * out["past3_bias_resistant_score"]
        + 0.25 * out["past3_bias_excuse_score"]
    ).fillna(0.0)

    return out.drop(
        columns=[
            "_frame_bucket",
            "_horse_number_bucket",
            "_draw_pace_adversity_for_history",
            "_bias_resistant_for_history",
            "_bias_excuse_for_history",
        ],
        errors="ignore",
    )


def add_member_level_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    horse_col = data_cfg["horse_id_column"]
    race_col = data_cfg["race_id_column"]
    date_col = data_cfg["date_column"]
    surface_col = config["training"].get("surface_column", "芝・ダ")

    out = df.copy()
    defaults = [
        "race_member_prior_strength_mean",
        "race_member_prior_strength_top3_mean",
        "race_member_prior_strength_max",
        "race_member_depth_score",
        "race_strong_opponent_count",
        "race_strong_opponent_ratio",
        "race_member_level_rank_score",
        "race_winner_prior_strength",
        "race_top3_prior_strength_mean",
        "prev_race_member_level",
        "past3_avg_race_member_level",
        "past3_max_race_member_level",
        "prev_performance_vs_member_level",
        "past3_avg_performance_vs_member_level",
        "past3_strong_race_good_run_score",
        "past5_strong_race_experience",
        "strong_race_bias_adjusted_score",
    ]
    for col in defaults:
        out[col] = 0.0

    surface = out[surface_col].astype("string") if surface_col in out.columns else pd.Series("", index=out.index)
    surface_top3 = pd.Series(0.0, index=out.index)
    turf_mask = surface == "芝"
    dirt_mask = surface == "ダ"
    surface_top3.loc[turf_mask] = _numeric(out.loc[turf_mask], "horse_turf_top3_rate").fillna(0.0)
    surface_top3.loc[dirt_mask] = _numeric(out.loc[dirt_mask], "horse_dirt_top3_rate").fillna(0.0)
    surface_avg = pd.Series(0.0, index=out.index)
    surface_avg.loc[turf_mask] = _numeric(out.loc[turf_mask], "horse_turf_avg_score").fillna(0.0)
    surface_avg.loc[dirt_mask] = _numeric(out.loc[dirt_mask], "horse_dirt_avg_score").fillna(0.0)

    out["_member_prior_strength"] = (
        0.34 * _numeric(out, "past3_avg_score").fillna(0.0)
        + 0.18 * _numeric(out, "same_distance_category_avg_score").fillna(0.0)
        + 0.16 * surface_avg
        + 0.14 * surface_top3
        + 0.10 * _numeric(out, "past3_avg_time_value").fillna(0.0).clip(-0.5, 0.5)
        + 0.08 * _numeric(out, "jockey_trainer_combo_score").fillna(0.0)
    ).fillna(0.0)

    race_group = out.groupby(race_col, sort=False)
    out["race_member_prior_strength_mean"] = race_group["_member_prior_strength"].transform("mean").fillna(0.0)
    out["race_member_prior_strength_max"] = race_group["_member_prior_strength"].transform("max").fillna(0.0)
    counts = race_group["_member_prior_strength"].transform("count").replace(0, np.nan)
    ranks = out["_member_prior_strength"].groupby(out[race_col]).rank(ascending=False, method="first")
    out["_member_prior_strength_rank"] = ranks
    top3_strength_sum = out["_member_prior_strength"].where(ranks <= 3, 0.0).groupby(out[race_col]).transform("sum")
    top3_counts = (ranks <= 3).astype(float).groupby(out[race_col]).transform("sum").replace(0, np.nan)
    out["race_member_prior_strength_top3_mean"] = (top3_strength_sum / top3_counts).fillna(0.0)
    out["race_member_depth_score"] = (
        0.55 * out["race_member_prior_strength_top3_mean"] + 0.45 * out["race_member_prior_strength_mean"]
    ).fillna(0.0)
    strong_cut = out["race_member_prior_strength_mean"] + 0.10
    strong_candidate = (out["_member_prior_strength"] >= strong_cut).astype(float)
    out["race_strong_opponent_count"] = strong_candidate.groupby(out[race_col]).transform("sum").astype(float)
    out["race_strong_opponent_ratio"] = (out["race_strong_opponent_count"] / counts).fillna(0.0)

    member_values = out["race_member_depth_score"]
    member_counts = member_values.notna().groupby(out[race_col]).transform("sum")
    member_ranks = member_values.groupby(out[race_col]).rank(ascending=False, method="average")
    out["race_member_level_rank_score"] = (
        (member_counts - member_ranks) / (member_counts - 1)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    finish = _numeric(out, data_cfg["rank_column"])
    out["_winner_strength_for_race"] = out["_member_prior_strength"].where(finish == 1, np.nan)
    out["race_winner_prior_strength"] = out.groupby(race_col)["_winner_strength_for_race"].transform("max").fillna(0.0)
    out["_top3_strength_for_race"] = out["_member_prior_strength"].where(finish <= 3, np.nan)
    out["race_top3_prior_strength_mean"] = out.groupby(race_col)["_top3_strength_for_race"].transform("mean").fillna(0.0)

    target_score = _numeric(out, "target_score").fillna(0.0)
    draw_excuse = _numeric(out, "past3_bias_excuse_score").fillna(0.0)
    out["_performance_vs_member_for_history"] = (
        target_score * (0.65 + out["race_member_depth_score"]) - out["_member_prior_strength"]
    ).fillna(0.0)
    out["_strong_race_good_run_for_history"] = (
        out["race_member_depth_score"].clip(lower=0.0)
        * (target_score + 0.25 * _numeric(out, "target_top3").fillna(0.0))
    ).fillna(0.0)
    out["_strong_race_experience_for_history"] = (out["race_member_depth_score"] >= 0.55).astype(float)
    out["_strong_race_bias_adjusted_for_history"] = (
        out["_strong_race_good_run_for_history"] + 0.30 * draw_excuse
    ).fillna(0.0)

    ordered = out.sort_values([horse_col, date_col, race_col], kind="mergesort")
    rolling_specs = [
        ("race_member_depth_score", "prev_race_member_level", "last", 3),
        ("race_member_depth_score", "past3_avg_race_member_level", "mean", 3),
        ("race_member_depth_score", "past3_max_race_member_level", "max", 3),
        ("_performance_vs_member_for_history", "prev_performance_vs_member_level", "last", 3),
        ("_performance_vs_member_for_history", "past3_avg_performance_vs_member_level", "mean", 3),
        ("_strong_race_good_run_for_history", "past3_strong_race_good_run_score", "mean", 3),
        ("_strong_race_experience_for_history", "past5_strong_race_experience", "sum", 5),
        ("_strong_race_bias_adjusted_for_history", "strong_race_bias_adjusted_score", "mean", 3),
    ]
    for source, dest, agg, window in rolling_specs:
        source_values = _numeric(ordered, source)
        if agg == "last":
            rolled = source_values.groupby(ordered[horse_col], sort=False).shift()
        elif agg == "max":
            rolled = source_values.groupby(ordered[horse_col], sort=False).transform(
                lambda s: s.shift().rolling(window, min_periods=1).max()
            )
        elif agg == "sum":
            rolled = source_values.groupby(ordered[horse_col], sort=False).transform(
                lambda s: s.shift().rolling(window, min_periods=1).sum()
            )
        else:
            rolled = source_values.groupby(ordered[horse_col], sort=False).transform(
                lambda s: s.shift().rolling(window, min_periods=1).mean()
            )
        out.loc[ordered.index, dest] = pd.to_numeric(rolled, errors="coerce").fillna(0.0).astype(float)

    return out.drop(
        columns=[
            "_member_prior_strength",
            "_member_prior_strength_rank",
            "_winner_strength_for_race",
            "_top3_strength_for_race",
            "_performance_vs_member_for_history",
            "_strong_race_good_run_for_history",
            "_strong_race_experience_for_history",
            "_strong_race_bias_adjusted_for_history",
        ],
        errors="ignore",
    )


def add_confirmed_opponent_form_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    race_col = data_cfg["race_id_column"]
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]
    prev_race_col = "前走レースID(新/馬番無)"

    out = df.copy()
    defaults = [
        "prev_race_next_starters_count",
        "prev_race_next_starters_ratio",
        "prev_race_next_avg_score",
        "prev_race_next_top3_rate",
        "prev_race_next_win_rate",
        "prev_race_confirmed_strength_score",
        "prev_race_confirmed_depth_score",
        "prev_target_score_for_context",
        "prev_confirmed_opponent_good_run_score",
        "prev_confirmed_opponent_excuse_score",
        "past3_confirmed_opponent_strength",
        "past3_confirmed_good_run_score",
        "confirmed_member_level_adjusted_score",
    ]
    for col in defaults:
        out[col] = 0.0

    if prev_race_col not in out.columns:
        return out

    ordered = out.sort_values([horse_col, date_col, race_col], kind="mergesort").copy()
    grouped = ordered.groupby(horse_col, sort=False)
    ordered["_next_date"] = grouped[date_col].shift(-1)
    ordered["_next_score"] = grouped["target_score"].shift(-1)
    ordered["_next_top3"] = grouped["target_top3"].shift(-1)
    ordered["_next_win"] = grouped["target_win"].shift(-1)

    member_next = ordered[
        [race_col, date_col, "_next_date", "_next_score", "_next_top3", "_next_win"]
    ].copy()
    member_next["_next_date"] = pd.to_numeric(member_next["_next_date"], errors="coerce")
    member_next = member_next[member_next["_next_date"].notna()].copy()

    out["_current_date_num"] = pd.to_numeric(out[date_col], errors="coerce")
    out["_prev_race_key"] = _id_key(out[prev_race_col])
    out["_race_key"] = _id_key(out[race_col])

    race_size_map = out.groupby("_race_key")[horse_col].count().to_dict()

    out["_own_next_date_for_context"] = np.nan
    out.loc[ordered.index, "_own_next_date_for_context"] = pd.to_numeric(ordered["_next_date"], errors="coerce")
    for col in [
        "_race_next_count_for_next",
        "_race_next_ratio_for_next",
        "_race_next_avg_score_for_next",
        "_race_next_top3_rate_for_next",
        "_race_next_win_rate_for_next",
    ]:
        out[col] = 0.0

    member_next["_race_key"] = _id_key(member_next[race_col])
    next_by_race = {str(race_id): part.sort_values("_next_date") for race_id, part in member_next.groupby("_race_key", sort=False)}
    queries_by_race = out[out["_own_next_date_for_context"].notna()].groupby("_race_key").groups
    for race_id, indices in queries_by_race.items():
        part = next_by_race.get(str(race_id))
        if part is None or part.empty:
            continue
        next_dates = part["_next_date"].to_numpy(dtype=float)
        order = np.argsort(next_dates)
        next_dates = next_dates[order]
        scores = pd.to_numeric(part["_next_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[order]
        top3 = pd.to_numeric(part["_next_top3"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[order]
        wins = pd.to_numeric(part["_next_win"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[order]
        score_csum = np.cumsum(scores)
        top3_csum = np.cumsum(top3)
        win_csum = np.cumsum(wins)

        own_next_dates = out.loc[indices, "_own_next_date_for_context"].to_numpy(dtype=float)
        counts = np.searchsorted(next_dates, own_next_dates, side="left")
        valid = counts > 0
        avg_score = np.zeros(len(indices), dtype=float)
        top3_rate = np.zeros(len(indices), dtype=float)
        win_rate = np.zeros(len(indices), dtype=float)
        avg_score[valid] = score_csum[counts[valid] - 1] / counts[valid]
        top3_rate[valid] = top3_csum[counts[valid] - 1] / counts[valid]
        win_rate[valid] = win_csum[counts[valid] - 1] / counts[valid]
        race_size = float(race_size_map.get(str(race_id), np.nan))
        ratio = counts / race_size if race_size and np.isfinite(race_size) else np.zeros(len(indices), dtype=float)

        out.loc[indices, "_race_next_count_for_next"] = counts.astype(float)
        out.loc[indices, "_race_next_ratio_for_next"] = ratio.astype(float)
        out.loc[indices, "_race_next_avg_score_for_next"] = avg_score
        out.loc[indices, "_race_next_top3_rate_for_next"] = top3_rate
        out.loc[indices, "_race_next_win_rate_for_next"] = win_rate

    queries_by_prev = out[out["_prev_race_key"].notna()].groupby("_prev_race_key").groups
    for prev_race_id, indices in queries_by_prev.items():
        part = next_by_race.get(str(prev_race_id))
        if part is None or part.empty:
            continue
        next_dates = part["_next_date"].to_numpy(dtype=float)
        order = np.argsort(next_dates)
        next_dates = next_dates[order]
        scores = pd.to_numeric(part["_next_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[order]
        top3 = pd.to_numeric(part["_next_top3"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[order]
        wins = pd.to_numeric(part["_next_win"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[order]
        score_csum = np.cumsum(scores)
        top3_csum = np.cumsum(top3)
        win_csum = np.cumsum(wins)

        current_dates = out.loc[indices, "_current_date_num"].to_numpy(dtype=float)
        counts = np.searchsorted(next_dates, current_dates, side="left")
        valid = counts > 0
        avg_score = np.zeros(len(indices), dtype=float)
        top3_rate = np.zeros(len(indices), dtype=float)
        win_rate = np.zeros(len(indices), dtype=float)
        avg_score[valid] = score_csum[counts[valid] - 1] / counts[valid]
        top3_rate[valid] = top3_csum[counts[valid] - 1] / counts[valid]
        win_rate[valid] = win_csum[counts[valid] - 1] / counts[valid]
        race_size = float(race_size_map.get(str(prev_race_id), np.nan))
        ratio = counts / race_size if race_size and np.isfinite(race_size) else np.zeros(len(indices), dtype=float)

        out.loc[indices, "prev_race_next_starters_count"] = counts.astype(float)
        out.loc[indices, "prev_race_next_starters_ratio"] = ratio.astype(float)
        out.loc[indices, "prev_race_next_avg_score"] = avg_score
        out.loc[indices, "prev_race_next_top3_rate"] = top3_rate
        out.loc[indices, "prev_race_next_win_rate"] = win_rate

    ordered_for_shift = out.sort_values([horse_col, date_col, race_col], kind="mergesort")
    for source, dest in [
        ("_race_next_count_for_next", "prev_race_next_starters_count"),
        ("_race_next_ratio_for_next", "prev_race_next_starters_ratio"),
        ("_race_next_avg_score_for_next", "prev_race_next_avg_score"),
        ("_race_next_top3_rate_for_next", "prev_race_next_top3_rate"),
        ("_race_next_win_rate_for_next", "prev_race_next_win_rate"),
    ]:
        shifted = _numeric(ordered_for_shift, source).groupby(ordered_for_shift[horse_col], sort=False).shift()
        existing = _numeric(out, dest).fillna(0.0)
        out.loc[ordered_for_shift.index, dest] = existing.loc[ordered_for_shift.index].where(
            existing.loc[ordered_for_shift.index] > 0,
            pd.to_numeric(shifted, errors="coerce").fillna(0.0),
        )

    out["prev_race_confirmed_strength_score"] = (
        0.55 * out["prev_race_next_avg_score"]
        + 0.25 * out["prev_race_next_top3_rate"]
        + 0.20 * out["prev_race_next_win_rate"]
    ).fillna(0.0)
    out["prev_race_confirmed_depth_score"] = (
        out["prev_race_confirmed_strength_score"] * out["prev_race_next_starters_ratio"].clip(0.0, 1.0)
    ).fillna(0.0)

    sorted_current = out.sort_values([horse_col, date_col, race_col], kind="mergesort")
    prev_target = sorted_current.groupby(horse_col, sort=False)["target_score"].shift()
    out.loc[sorted_current.index, "prev_target_score_for_context"] = pd.to_numeric(prev_target, errors="coerce").fillna(0.0)

    disadvantage = (-_numeric(out, "prev_draw_advantage_score").fillna(0.0)).clip(lower=0.0, upper=0.25)
    out["prev_confirmed_opponent_good_run_score"] = (
        out["prev_target_score_for_context"] * out["prev_race_confirmed_depth_score"]
    ).fillna(0.0)
    out["prev_confirmed_opponent_excuse_score"] = (
        (1.0 - out["prev_target_score_for_context"]).clip(0.0, 1.0)
        * out["prev_race_confirmed_depth_score"]
        * (0.50 + 2.0 * disadvantage)
    ).fillna(0.0)

    ordered2 = out.sort_values([horse_col, date_col, race_col], kind="mergesort")
    for source, dest in [
        ("prev_race_confirmed_depth_score", "past3_confirmed_opponent_strength"),
        ("prev_confirmed_opponent_good_run_score", "past3_confirmed_good_run_score"),
    ]:
        values = _numeric(ordered2, source)
        rolled = values.groupby(ordered2[horse_col], sort=False).transform(
            lambda s: s.shift().rolling(3, min_periods=1).mean()
        )
        out.loc[ordered2.index, dest] = pd.to_numeric(rolled, errors="coerce").fillna(0.0).astype(float)

    out["confirmed_member_level_adjusted_score"] = (
        _numeric(out, "bias_adjusted_recent_score").fillna(0.0)
        + 0.45 * out["prev_confirmed_opponent_good_run_score"]
        + 0.25 * out["prev_confirmed_opponent_excuse_score"]
        + 0.20 * out["past3_confirmed_good_run_score"]
    ).fillna(0.0)

    return out.drop(
        columns=[
            "_current_date_num",
            "_prev_race_key",
            "_race_key",
            "_own_next_date_for_context",
            "_race_next_count_for_next",
            "_race_next_ratio_for_next",
            "_race_next_avg_score_for_next",
            "_race_next_top3_rate_for_next",
            "_race_next_win_rate_for_next",
        ],
        errors="ignore",
    )


def add_bloodline_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    date_col = config["data"]["date_column"]
    race_col = config["data"]["race_id_column"]
    surface_col = config["training"].get("surface_column", "芝・ダ")

    out = df.copy()
    defaults = [
        "sire_starts",
        "sire_win_rate",
        "sire_top3_rate",
        "sire_avg_score",
        "sire_popularity_outperform_rate",
        "sire_surface_top3_rate",
        "sire_distance_top3_rate",
        "sire_venue_top3_rate",
        "sire_going_top3_rate",
        "bms_starts",
        "bms_win_rate",
        "bms_top3_rate",
        "bms_avg_score",
        "bms_popularity_outperform_rate",
        "bms_surface_top3_rate",
        "bms_distance_top3_rate",
        "bms_venue_top3_rate",
        "bms_going_top3_rate",
        "sire_type_top3_rate",
        "sire_type_surface_top3_rate",
        "sire_type_distance_top3_rate",
        "bms_type_top3_rate",
        "bms_type_surface_top3_rate",
        "bms_type_distance_top3_rate",
        "sire_bms_pair_starts",
        "sire_bms_pair_win_rate",
        "sire_bms_pair_top3_rate",
        "sire_bms_pair_avg_score",
        "sire_bms_pair_popularity_outperform_rate",
        "sire_surface_lift",
        "sire_distance_lift",
        "sire_venue_lift",
        "sire_going_lift",
        "bms_surface_lift",
        "bms_distance_lift",
        "bms_venue_lift",
        "bms_going_lift",
        "bloodline_pair_reliability_score",
        "bloodline_pair_fit_score",
        "bloodline_course_fit_score",
        "bloodline_surface_distance_fit_score",
        "bloodline_lift_fit_score",
        "bloodline_high_confidence_fit_score",
        "bloodline_low_sample_flag",
        "bloodline_reliability_score",
    ]
    for col in defaults:
        out[col] = 0.0

    sort_cols = [date_col, race_col]
    blood_specs = [
        (["種牡馬"], "sire"),
        (["種牡馬", surface_col], "sire_surface"),
        (["種牡馬", "distance_category"], "sire_distance"),
        (["種牡馬", "場所"], "sire_venue"),
        (["種牡馬", "馬場状態"], "sire_going"),
        (["母父馬"], "bms"),
        (["母父馬", surface_col], "bms_surface"),
        (["母父馬", "distance_category"], "bms_distance"),
        (["母父馬", "場所"], "bms_venue"),
        (["母父馬", "馬場状態"], "bms_going"),
        (["父タイプ名"], "sire_type"),
        (["父タイプ名", surface_col], "sire_type_surface"),
        (["父タイプ名", "distance_category"], "sire_type_distance"),
        (["母父タイプ名"], "bms_type"),
        (["母父タイプ名", surface_col], "bms_type_surface"),
        (["母父タイプ名", "distance_category"], "bms_type_distance"),
        (["種牡馬", "母父馬"], "sire_bms_pair"),
    ]
    for group_cols, prefix in blood_specs:
        if all(col in out.columns for col in group_cols):
            out = _add_entity_history_features(out, group_cols=group_cols, prefix=prefix, sort_cols=sort_cols)

    out["bloodline_course_fit_score"] = (
        0.24 * _numeric(out, "sire_surface_top3_rate").fillna(0.0)
        + 0.20 * _numeric(out, "sire_distance_top3_rate").fillna(0.0)
        + 0.16 * _numeric(out, "sire_venue_top3_rate").fillna(0.0)
        + 0.10 * _numeric(out, "sire_going_top3_rate").fillna(0.0)
        + 0.12 * _numeric(out, "bms_surface_top3_rate").fillna(0.0)
        + 0.10 * _numeric(out, "bms_distance_top3_rate").fillna(0.0)
        + 0.08 * _numeric(out, "bms_venue_top3_rate").fillna(0.0)
    ).fillna(0.0)
    out["bloodline_surface_distance_fit_score"] = (
        0.45 * _numeric(out, "sire_surface_top3_rate").fillna(0.0)
        + 0.35 * _numeric(out, "sire_distance_top3_rate").fillna(0.0)
        + 0.10 * _numeric(out, "bms_surface_top3_rate").fillna(0.0)
        + 0.10 * _numeric(out, "bms_distance_top3_rate").fillna(0.0)
    ).fillna(0.0)
    out["bloodline_reliability_score"] = (
        (_numeric(out, "sire_starts").fillna(0.0).clip(upper=50) / 50.0) * 0.65
        + (_numeric(out, "bms_starts").fillna(0.0).clip(upper=50) / 50.0) * 0.35
    ).fillna(0.0)
    out["bloodline_pair_reliability_score"] = (
        _numeric(out, "sire_bms_pair_starts").fillna(0.0).clip(upper=30) / 30.0
    ).fillna(0.0)
    out["sire_surface_lift"] = (_numeric(out, "sire_surface_top3_rate") - _numeric(out, "sire_top3_rate")).fillna(0.0)
    out["sire_distance_lift"] = (_numeric(out, "sire_distance_top3_rate") - _numeric(out, "sire_top3_rate")).fillna(0.0)
    out["sire_venue_lift"] = (_numeric(out, "sire_venue_top3_rate") - _numeric(out, "sire_top3_rate")).fillna(0.0)
    out["sire_going_lift"] = (_numeric(out, "sire_going_top3_rate") - _numeric(out, "sire_top3_rate")).fillna(0.0)
    out["bms_surface_lift"] = (_numeric(out, "bms_surface_top3_rate") - _numeric(out, "bms_top3_rate")).fillna(0.0)
    out["bms_distance_lift"] = (_numeric(out, "bms_distance_top3_rate") - _numeric(out, "bms_top3_rate")).fillna(0.0)
    out["bms_venue_lift"] = (_numeric(out, "bms_venue_top3_rate") - _numeric(out, "bms_top3_rate")).fillna(0.0)
    out["bms_going_lift"] = (_numeric(out, "bms_going_top3_rate") - _numeric(out, "bms_top3_rate")).fillna(0.0)
    out["bloodline_lift_fit_score"] = (
        0.30 * out["sire_surface_lift"]
        + 0.25 * out["sire_distance_lift"]
        + 0.15 * out["sire_venue_lift"]
        + 0.10 * out["sire_going_lift"]
        + 0.10 * out["bms_surface_lift"]
        + 0.10 * out["bms_distance_lift"]
    ).fillna(0.0)
    out["bloodline_pair_fit_score"] = (
        _numeric(out, "sire_bms_pair_top3_rate").fillna(0.0)
        * (0.50 + 0.50 * out["bloodline_pair_reliability_score"])
    ).fillna(0.0)
    out["bloodline_high_confidence_fit_score"] = (
        (
            0.60 * out["bloodline_course_fit_score"]
            + 0.25 * out["bloodline_lift_fit_score"]
            + 0.15 * out["bloodline_pair_fit_score"]
        )
        * out["bloodline_reliability_score"]
    ).fillna(0.0)
    out["bloodline_low_sample_flag"] = (
        (_numeric(out, "sire_starts").fillna(0.0) < 10) | (_numeric(out, "bms_starts").fillna(0.0) < 10)
    ).astype(float)
    return out


def _class_level(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("")
    levels = pd.Series(0.0, index=values.index)
    specs = [
        ("新馬", 1.0),
        ("未勝利", 1.0),
        ("1勝", 2.0),
        ("2勝", 3.0),
        ("3勝", 4.0),
        ("ｵｰﾌﾟﾝ", 5.0),
        ("オープン", 5.0),
        ("OP", 5.0),
        ("L", 5.5),
        ("Ｇ３", 6.0),
        ("G3", 6.0),
        ("Ｇ２", 7.0),
        ("G2", 7.0),
        ("Ｇ１", 8.0),
        ("G1", 8.0),
    ]
    for pattern, level in specs:
        levels = levels.mask(text.str.contains(pattern, regex=False, na=False), level)
    return levels


def _add_entity_history_features(
    out: pd.DataFrame,
    *,
    group_cols: list[str],
    prefix: str,
    sort_cols: list[str],
) -> pd.DataFrame:
    for col in [
        f"{prefix}_starts",
        f"{prefix}_win_rate",
        f"{prefix}_top3_rate",
        f"{prefix}_avg_score",
        f"{prefix}_popularity_outperform_rate",
    ]:
        out[col] = 0.0
    required = {"target_win", "target_top3", "target_score"}.union(group_cols)
    if not required.issubset(out.columns):
        return out

    ordered_keys = list(dict.fromkeys([*group_cols, *sort_cols]))
    ordered = out.sort_values(ordered_keys, kind="mergesort").copy()
    popularity = _numeric(ordered, "人気")
    finish = _numeric(ordered, "確定着順")
    ordered["_popularity_outperform"] = (popularity - finish > 0).astype(float)

    valid_mask = ordered[group_cols].notna().all(axis=1)
    if not valid_mask.any():
        return out

    valid = ordered.loc[valid_mask].copy()
    valid["_row_index_for_entity_history"] = valid.index
    race_keys = list(dict.fromkeys([*group_cols, *sort_cols]))
    race_agg = (
        valid.groupby(race_keys, sort=False, dropna=False)
        .agg(
            _entries=("target_win", "size"),
            _wins=("target_win", "sum"),
            _top3=("target_top3", "sum"),
            _score=("target_score", "sum"),
            _pop_out=("_popularity_outperform", "sum"),
        )
        .reset_index()
        .sort_values(race_keys, kind="mergesort")
    )
    grouped_races = race_agg.groupby(group_cols, sort=False, dropna=False)
    for source in ["_entries", "_wins", "_top3", "_score", "_pop_out"]:
        race_agg[f"_prior{source}"] = grouped_races[source].cumsum() - race_agg[source]

    valid = valid.merge(
        race_agg[
            [
                *race_keys,
                "_prior_entries",
                "_prior_wins",
                "_prior_top3",
                "_prior_score",
                "_prior_pop_out",
            ]
        ],
        on=race_keys,
        how="left",
        sort=False,
    ).set_index("_row_index_for_entity_history")

    starts = pd.to_numeric(valid["_prior_entries"], errors="coerce").fillna(0.0)
    denom = starts.replace(0, np.nan)
    out.loc[valid.index, f"{prefix}_starts"] = starts.astype(float)
    for source, dest in [
        ("_prior_wins", f"{prefix}_win_rate"),
        ("_prior_top3", f"{prefix}_top3_rate"),
        ("_prior_score", f"{prefix}_avg_score"),
        ("_prior_pop_out", f"{prefix}_popularity_outperform_rate"),
    ]:
        values = pd.to_numeric(valid[source], errors="coerce").fillna(0.0)
        out.loc[valid.index, dest] = (values / denom).fillna(0.0).astype(float)
    return out


def add_general_generated_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]
    race_col = data_cfg["race_id_column"]

    out = df.copy()
    out["distance_category"] = _distance_category(_numeric(out, "距離"))
    out["previous_distance_category"] = _distance_category(_numeric(out, "前距離"))
    out["distance_diff"] = (_numeric(out, "距離") - _numeric(out, "前距離")).fillna(0.0)
    out["weight_diff"] = (_numeric(out, "斤量") - _numeric(out, "前走斤量")).fillna(0.0)

    if "前クラス名" in out.columns and "クラス名" in out.columns:
        out["class_changed"] = (out["前クラス名"].astype("string") != out["クラス名"].astype("string")).astype(float)
    else:
        out["class_changed"] = 0.0
    if "前走騎手コード" in out.columns and "騎手コード" in out.columns:
        out["jockey_changed"] = (out["前走騎手コード"].astype("string") != out["騎手コード"].astype("string")).astype(float)
    else:
        out["jockey_changed"] = 0.0

    out["past3_avg_score"] = 0.0
    out["past3_avg_margin"] = 0.0
    out["past3_avg_final3f_rank"] = 0.0
    out["past3_avg_corner4_position_rate"] = 0.0
    out["prev_corner4_position_rate"] = 0.0
    out["past3_front_run_count"] = 0.0
    out["past3_stalker_count"] = 0.0
    out["past3_closer_count"] = 0.0
    out["front_running_tendency"] = 0.0
    out["closing_tendency"] = 0.0

    prev_corner4 = _numeric(out, "前4角.1")
    if prev_corner4.isna().all():
        prev_corner4 = _numeric(out, "前4角")
    prev_field = _numeric(out, "前走出走頭数").replace(0, np.nan)
    out["prev_corner4_position_rate"] = (prev_corner4 / prev_field).replace([np.inf, -np.inf], np.nan).fillna(0.5)

    if {"target_score", "target_top3"}.issubset(out.columns):
        margin = _numeric(out, "着差").fillna(0.0)
        final3f_rank = _numeric(out, "上り3F順")
        corner4 = _numeric(out, "4角.1")
        if corner4.isna().all():
            corner4 = _numeric(out, "4角")
        field_size = _numeric(out, "出走頭数").replace(0, np.nan)
        out["_margin_for_history"] = margin
        out["_final3f_rank_for_history"] = final3f_rank
        out["_corner4_rate_for_history"] = (corner4 / field_size).replace([np.inf, -np.inf], np.nan)
        out["_front_for_history"] = (out["_corner4_rate_for_history"] <= 0.25).astype(float)
        out["_stalker_for_history"] = (
            (out["_corner4_rate_for_history"] > 0.25) & (out["_corner4_rate_for_history"] <= 0.45)
        ).astype(float)
        out["_closer_for_history"] = (out["_corner4_rate_for_history"] >= 0.70).astype(float)

        ordered = out.sort_values([horse_col, date_col, race_col], kind="mergesort")
        for source, dest in [
            ("target_score", "past3_avg_score"),
            ("_margin_for_history", "past3_avg_margin"),
            ("_final3f_rank_for_history", "past3_avg_final3f_rank"),
            ("_corner4_rate_for_history", "past3_avg_corner4_position_rate"),
        ]:
            out.loc[ordered.index, dest] = _rolling_previous_mean(ordered, horse_col, source).fillna(0.0)

        for source, dest in [
            ("_front_for_history", "past3_front_run_count"),
            ("_stalker_for_history", "past3_stalker_count"),
            ("_closer_for_history", "past3_closer_count"),
        ]:
            out.loc[ordered.index, dest] = (
                ordered.groupby(horse_col, sort=False)[source]
                .transform(lambda s: s.shift().rolling(3, min_periods=1).sum())
                .fillna(0.0)
            )

        out = out.drop(
            columns=[
                "_margin_for_history",
                "_final3f_rank_for_history",
                "_corner4_rate_for_history",
                "_front_for_history",
                "_stalker_for_history",
                "_closer_for_history",
            ]
        )

    out["front_running_tendency"] = (
        0.45 * (1.0 - out["prev_corner4_position_rate"].clip(0.0, 1.0))
        + 0.40 * (out["past3_front_run_count"] / 3.0).clip(0.0, 1.0)
        + 0.15 * (out["past3_stalker_count"] / 3.0).clip(0.0, 1.0)
    ).fillna(0.0)
    out["closing_tendency"] = (
        0.55 * out["prev_corner4_position_rate"].clip(0.0, 1.0)
        + 0.45 * (out["past3_closer_count"] / 3.0).clip(0.0, 1.0)
    ).fillna(0.0)

    sort_cols = [horse_col, date_col, race_col]
    out = _add_previous_group_rates(
        out,
        group_cols=[horse_col, "distance_category"],
        prefix="same_distance_category",
        sort_cols=sort_cols,
    )
    out = _add_previous_group_rates(
        out,
        group_cols=[horse_col, "場所"],
        prefix="same_venue",
        sort_cols=sort_cols,
    )
    return out


JRA_VENUES = {"札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"}
SOUTH_KANTO_VENUES = {"大井", "船橋", "川崎", "浦和"}
HOKKAIDO_LOCAL_VENUES = {"門別"}


def _text_column(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype=str)
    return df[col].astype("string").fillna("").astype(str).str.strip()


def add_local_racing_context_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add conservative NAR/local-to-JRA transition signals from pre-race previous-run columns."""
    out = df.copy()
    feature_cols = [
        "prev_local_race_flag",
        "prev_local_south_kanto_flag",
        "prev_local_hokkaido_flag",
        "prev_local_other_flag",
        "prev_local_win_flag",
        "prev_local_top3_flag",
        "prev_local_bad_loss_flag",
        "prev_local_front_flag",
        "prev_local_popularity_outperform_flag",
        "prev_local_dirt_sprint_flag",
        "prev_local_surface_switch_flag",
        "prev_local_good_run_score",
        "prev_local_transition_risk_score",
        "prev_local_transition_value_score",
    ]
    for col in feature_cols:
        out[col] = 0.0

    prev_place = _text_column(out, "前走場所")
    has_prev_place = prev_place.ne("") & ~prev_place.str.lower().isin({"nan", "none", "<na>"})
    prev_local = has_prev_place & ~prev_place.isin(JRA_VENUES)
    if not prev_local.any():
        return out

    prev_rank = _numeric(out, "前走確定着順")
    prev_pop = _numeric(out, "前走人気")
    prev_margin = _numeric(out, "前走着差タイム")
    prev_c4 = _numeric(out, "前4角")
    prev_field = _numeric(out, "前走出走頭数").replace(0, np.nan)
    if prev_field.isna().all():
        prev_field = _numeric(out, "前走頭数").replace(0, np.nan)
    prev_surface = _text_column(out, "前芝・ダ")
    current_surface = _text_column(out, "芝・ダ")
    current_distance = _numeric(out, "距離")
    prev_distance = _numeric(out, "前距離")

    top3 = prev_rank.le(3)
    win = prev_rank.eq(1)
    bad_loss = prev_rank.ge(6) & (prev_margin.ge(1.0) | prev_margin.isna())
    front = (prev_c4 / prev_field).replace([np.inf, -np.inf], np.nan).le(0.35) | prev_c4.le(3)
    pop_outperform = top3 & prev_pop.ge(4)
    dirt_sprint = prev_surface.eq("ダ") & current_surface.eq("ダ") & current_distance.le(1400)
    surface_switch = prev_surface.ne("") & current_surface.ne("") & prev_surface.ne(current_surface)
    distance_shift = (current_distance - prev_distance).abs().fillna(0.0)

    out["prev_local_race_flag"] = prev_local.astype(float)
    out["prev_local_south_kanto_flag"] = (prev_local & prev_place.isin(SOUTH_KANTO_VENUES)).astype(float)
    out["prev_local_hokkaido_flag"] = (prev_local & prev_place.isin(HOKKAIDO_LOCAL_VENUES)).astype(float)
    out["prev_local_other_flag"] = (
        prev_local & ~prev_place.isin(SOUTH_KANTO_VENUES | HOKKAIDO_LOCAL_VENUES)
    ).astype(float)
    out["prev_local_win_flag"] = (prev_local & win).astype(float)
    out["prev_local_top3_flag"] = (prev_local & top3).astype(float)
    out["prev_local_bad_loss_flag"] = (prev_local & bad_loss).astype(float)
    out["prev_local_front_flag"] = (prev_local & front).astype(float)
    out["prev_local_popularity_outperform_flag"] = (prev_local & pop_outperform).astype(float)
    out["prev_local_dirt_sprint_flag"] = (prev_local & dirt_sprint).astype(float)
    out["prev_local_surface_switch_flag"] = (prev_local & surface_switch).astype(float)

    out["prev_local_good_run_score"] = (
        0.45 * out["prev_local_top3_flag"]
        + 0.25 * out["prev_local_win_flag"]
        + 0.15 * out["prev_local_front_flag"]
        + 0.10 * out["prev_local_popularity_outperform_flag"]
        + 0.05 * out["prev_local_dirt_sprint_flag"]
    ).clip(0.0, 1.0)
    out["prev_local_transition_risk_score"] = (
        0.55 * out["prev_local_bad_loss_flag"]
        + 0.18 * out["prev_local_surface_switch_flag"]
        + 0.12 * (prev_local & distance_shift.ge(600)).astype(float)
        + 0.10 * (prev_local & prev_rank.isna()).astype(float)
        - 0.20 * out["prev_local_good_run_score"]
    ).clip(0.0, 1.0)
    out["prev_local_transition_value_score"] = (
        0.70 * out["prev_local_good_run_score"]
        + 0.15 * out["prev_local_south_kanto_flag"]
        + 0.10 * out["prev_local_hokkaido_flag"]
        - 0.25 * out["prev_local_transition_risk_score"]
    ).clip(0.0, 1.0)
    return out


def add_time_value_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    train_cfg = config["training"]
    horse_col = data_cfg["horse_id_column"]
    race_col = data_cfg["race_id_column"]
    date_col = data_cfg["date_column"]
    surface_col = train_cfg.get("surface_column", "芝・ダ")

    out = df.copy()
    feature_defaults = [
        "prev_margin_sec",
        "past3_avg_margin_sec",
        "past3_best_margin_sec",
        "prev_race_time_value",
        "prev_time_z_course_distance",
        "prev_day_track_speed_index",
        "prev_time_adjusted_by_day_bias",
        "prev_class_time_value_score",
        "past3_avg_time_value",
        "past3_best_time_value",
        "past3_avg_time_z",
        "past3_avg_time_adjusted_by_day_bias",
        "horse_time_value_plus_margin",
    ]
    for col in feature_defaults:
        out[col] = 0.0

    avg1f = _numeric(out, "平均1Fタイム")
    distance = _numeric(out, "距離").replace(0, np.nan)
    margin = _numeric(out, "着差").fillna(0.0)
    out["prev_margin_sec"] = _numeric(out, "前走着差タイム").fillna(0.0)

    if avg1f.notna().sum() == 0:
        out["past3_avg_margin_sec"] = out["past3_avg_margin"]
        out["past3_best_margin_sec"] = out["past3_avg_margin"]
        return out

    race_keys = [race_col, date_col, "場所", surface_col, "距離", "馬場状態", "クラス名"]
    existing_keys = [col for col in race_keys if col in out.columns]
    race_frame = (
        out.assign(_avg1f=avg1f)
        .sort_values([date_col, race_col], kind="mergesort")
        .drop_duplicates(race_col)[existing_keys + ["_avg1f"]]
        .copy()
    )

    sort_cols = [date_col, race_col]
    condition_cols = [col for col in ["場所", surface_col, "距離", "馬場状態"] if col in race_frame.columns]
    class_condition_cols = [*condition_cols, "クラス名"] if "クラス名" in race_frame.columns else condition_cols

    race_frame["_condition_mean_prev"] = _expanding_previous_mean(race_frame, condition_cols, "_avg1f", sort_cols)
    race_frame["_condition_std_prev"] = _expanding_previous_std(race_frame, condition_cols, "_avg1f", sort_cols)
    race_frame["_class_mean_prev"] = _expanding_previous_mean(race_frame, class_condition_cols, "_avg1f", sort_cols)

    race_frame["_time_value"] = (race_frame["_condition_mean_prev"] - race_frame["_avg1f"]).fillna(0.0)
    race_frame["_time_z"] = (
        race_frame["_time_value"] / race_frame["_condition_std_prev"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    race_frame["_class_time_value"] = (race_frame["_class_mean_prev"] - race_frame["_avg1f"]).fillna(race_frame["_time_value"]).fillna(0.0)

    day_cols = [col for col in [date_col, "場所", surface_col] if col in race_frame.columns]
    if day_cols:
        race_frame["_day_track_speed"] = race_frame.groupby(day_cols)["_time_value"].transform("mean").fillna(0.0)
    else:
        race_frame["_day_track_speed"] = 0.0
    race_frame["_time_adjusted_by_day"] = (race_frame["_time_value"] - race_frame["_day_track_speed"]).fillna(0.0)

    race_metrics = race_frame.set_index(race_col)[
        ["_time_value", "_time_z", "_day_track_speed", "_time_adjusted_by_day", "_class_time_value"]
    ]
    out["_race_time_value_for_history"] = out[race_col].map(race_metrics["_time_value"]).fillna(0.0)
    out["_time_z_for_history"] = out[race_col].map(race_metrics["_time_z"]).fillna(0.0)
    out["_day_track_speed_for_history"] = out[race_col].map(race_metrics["_day_track_speed"]).fillna(0.0)
    out["_time_adjusted_for_history"] = out[race_col].map(race_metrics["_time_adjusted_by_day"]).fillna(0.0)
    out["_class_time_value_for_history"] = out[race_col].map(race_metrics["_class_time_value"]).fillna(0.0)

    furlongs = (distance / 200.0).replace(0, np.nan)
    out["_horse_time_value_for_history"] = (
        out["_time_adjusted_for_history"] - margin / furlongs
    ).replace([np.inf, -np.inf], np.nan).fillna(out["_time_adjusted_for_history"])

    ordered = out.sort_values([horse_col, date_col, race_col], kind="mergesort")
    rolling_specs = [
        ("着差", "past3_avg_margin_sec", "mean"),
        ("着差", "past3_best_margin_sec", "min"),
        ("_race_time_value_for_history", "prev_race_time_value", "last"),
        ("_time_z_for_history", "prev_time_z_course_distance", "last"),
        ("_day_track_speed_for_history", "prev_day_track_speed_index", "last"),
        ("_time_adjusted_for_history", "prev_time_adjusted_by_day_bias", "last"),
        ("_class_time_value_for_history", "prev_class_time_value_score", "last"),
        ("_horse_time_value_for_history", "past3_avg_time_value", "mean"),
        ("_horse_time_value_for_history", "past3_best_time_value", "max"),
        ("_time_z_for_history", "past3_avg_time_z", "mean"),
        ("_time_adjusted_for_history", "past3_avg_time_adjusted_by_day_bias", "mean"),
    ]
    for source, dest, agg in rolling_specs:
        source_values = _numeric(ordered, source) if source in ordered.columns else pd.Series(np.nan, index=ordered.index)
        if agg == "last":
            values = ordered.groupby(horse_col, sort=False)[source].shift()
        elif agg == "min":
            values = source_values.groupby(ordered[horse_col], sort=False).transform(
                lambda s: s.shift().rolling(3, min_periods=1).min()
            )
        elif agg == "max":
            values = source_values.groupby(ordered[horse_col], sort=False).transform(
                lambda s: s.shift().rolling(3, min_periods=1).max()
            )
        else:
            values = source_values.groupby(ordered[horse_col], sort=False).transform(
                lambda s: s.shift().rolling(3, min_periods=1).mean()
            )
        out.loc[ordered.index, dest] = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)

    out["horse_time_value_plus_margin"] = (
        0.7 * out["past3_avg_time_adjusted_by_day_bias"] - 0.3 * out["past3_avg_margin_sec"]
    ).fillna(0.0)

    return out.drop(
        columns=[
            "_race_time_value_for_history",
            "_time_z_for_history",
            "_day_track_speed_for_history",
            "_time_adjusted_for_history",
            "_class_time_value_for_history",
            "_horse_time_value_for_history",
        ],
        errors="ignore",
    )


def _expanding_previous_mean(df: pd.DataFrame, group_cols: list[str], value_col: str, sort_cols: list[str]) -> pd.Series:
    if not group_cols:
        ordered = df.sort_values(sort_cols, kind="mergesort")
        values = ordered[value_col].expanding().mean().shift()
        return values.reindex(df.index)
    ordered = df.sort_values([*group_cols, *sort_cols], kind="mergesort")
    values = ordered.groupby(group_cols, sort=False)[value_col].transform(
        lambda s: s.expanding().mean().shift()
    )
    return values.reindex(df.index)


def _expanding_previous_std(df: pd.DataFrame, group_cols: list[str], value_col: str, sort_cols: list[str]) -> pd.Series:
    if not group_cols:
        ordered = df.sort_values(sort_cols, kind="mergesort")
        values = ordered[value_col].expanding().std(ddof=0).shift()
        return values.reindex(df.index)
    ordered = df.sort_values([*group_cols, *sort_cols], kind="mergesort")
    values = ordered.groupby(group_cols, sort=False)[value_col].transform(
        lambda s: s.expanding().std(ddof=0).shift()
    )
    return values.reindex(df.index)


def add_horse_surface_history(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    train_cfg = config["training"]
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]
    race_col = data_cfg["race_id_column"]
    surface_col = train_cfg.get("surface_column", "芝・ダ")

    out = df.copy()
    surfaces = train_cfg.get("surface_values", ["芝", "ダ"])
    suffix_by_surface = {"芝": "turf", "ダ": "dirt"}

    for surface in surfaces:
        suffix = suffix_by_surface.get(str(surface), str(surface))
        starts_col = f"horse_{suffix}_starts"
        win_rate_col = f"horse_{suffix}_win_rate"
        top3_rate_col = f"horse_{suffix}_top3_rate"
        score_col = f"horse_{suffix}_avg_score"

        out[starts_col] = 0.0
        out[win_rate_col] = 0.0
        out[top3_rate_col] = 0.0
        out[score_col] = 0.0

    if not {"target_win", "target_top3", "target_score"}.issubset(out.columns):
        return out

    sort_cols = [horse_col, date_col, race_col]
    ordered = out.sort_values(sort_cols, kind="mergesort")
    for surface in surfaces:
        suffix = suffix_by_surface.get(str(surface), str(surface))
        starts_col = f"horse_{suffix}_starts"
        win_rate_col = f"horse_{suffix}_win_rate"
        top3_rate_col = f"horse_{suffix}_top3_rate"
        score_col = f"horse_{suffix}_avg_score"

        surface_rows = ordered[ordered[surface_col] == surface].copy()
        if surface_rows.empty:
            continue

        grouped = surface_rows.groupby(horse_col, sort=False)
        starts = grouped.cumcount()
        wins = grouped["target_win"].cumsum().shift(fill_value=0)
        top3 = grouped["target_top3"].cumsum().shift(fill_value=0)
        score = grouped["target_score"].cumsum().shift(fill_value=0.0)

        # The shift above must reset at each horse boundary.
        first_for_horse = surface_rows[horse_col] != surface_rows[horse_col].shift()
        wins.loc[first_for_horse] = 0
        top3.loc[first_for_horse] = 0
        score.loc[first_for_horse] = 0.0

        denom = starts.replace(0, np.nan)
        out.loc[surface_rows.index, starts_col] = starts.astype(float)
        out.loc[surface_rows.index, win_rate_col] = (wins / denom).fillna(0.0).astype(float)
        out.loc[surface_rows.index, top3_rate_col] = (top3 / denom).fillna(0.0).astype(float)
        out.loc[surface_rows.index, score_col] = (score / denom).fillna(0.0).astype(float)

    return out


TRACK_CONDITION_NUMERIC_FEATURES = [
    "track_condition_available",
    "race_cushion_value",
    "race_moisture_goal",
    "race_moisture_back",
    "race_moisture_avg",
    "race_cushion_z_by_venue",
    "race_moisture_z_by_venue_surface",
    "race_high_cushion_flag",
    "race_low_cushion_flag",
    "race_wet_moisture_flag",
    "race_dry_moisture_flag",
    "horse_high_cushion_starts",
    "horse_high_cushion_top3_rate",
    "horse_high_cushion_avg_score",
    "horse_high_cushion_popularity_outperform_rate",
    "horse_low_cushion_starts",
    "horse_low_cushion_top3_rate",
    "horse_low_cushion_avg_score",
    "horse_wet_moisture_starts",
    "horse_wet_moisture_top3_rate",
    "horse_wet_moisture_avg_score",
    "horse_dry_moisture_starts",
    "horse_dry_moisture_top3_rate",
    "horse_dry_moisture_avg_score",
    "horse_cushion_fit_score",
    "horse_moisture_fit_score",
    "horse_track_condition_fit_score",
]


def add_track_condition_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    train_cfg = config["training"]
    horse_col = data_cfg["horse_id_column"]
    race_col = data_cfg["race_id_column"]
    date_col = data_cfg["date_column"]
    surface_col = train_cfg.get("surface_column", "闃昴・繝")
    venue_col = train_cfg.get("venue_column", "蝣ｴ謇")

    out = df.copy()
    for col in TRACK_CONDITION_NUMERIC_FEATURES:
        out[col] = 0.0

    metrics_path = data_cfg.get("track_condition_csv")
    if not metrics_path:
        return out

    metrics_file = Path(str(metrics_path))
    if not metrics_file.is_absolute():
        metrics_file = Path.cwd() / metrics_file
    if not metrics_file.exists():
        return out

    metrics = _load_track_condition_metrics(metrics_file)
    required = {"_track_condition_date", "_track_condition_venue"}
    if metrics.empty or not required.issubset(metrics.columns):
        return out

    out["_track_condition_date"] = out[date_col].map(_normalize_ymd_key)
    out["_track_condition_venue"] = out[venue_col].astype("string").fillna("").str.strip() if venue_col in out.columns else ""
    if surface_col in out.columns:
        out["_track_condition_surface"] = out[surface_col].astype("string").fillna("").str.strip()
    else:
        out["_track_condition_surface"] = ""

    merge_cols = [
        "_track_condition_date",
        "_track_condition_venue",
        "race_cushion_value",
        "moisture_turf_goal",
        "moisture_turf_back",
        "moisture_dirt_goal",
        "moisture_dirt_back",
    ]
    metric_value_cols = [col for col in merge_cols if not col.startswith("_track_condition_")]
    metrics_for_merge = metrics[merge_cols].rename(columns={col: f"{col}__metric" for col in metric_value_cols})
    merged = out.merge(metrics_for_merge, on=["_track_condition_date", "_track_condition_venue"], how="left")
    out.index = merged.index
    out["race_cushion_value"] = _numeric(merged, "race_cushion_value__metric").fillna(0.0)

    surface = merged["_track_condition_surface"].astype("string").fillna("")
    turf_mask = surface.str.contains("闃|芝", regex=True, na=False)
    dirt_mask = surface.str.contains("繝|ダ", regex=True, na=False)
    turf_goal = _numeric(merged, "moisture_turf_goal__metric")
    turf_back = _numeric(merged, "moisture_turf_back__metric")
    dirt_goal = _numeric(merged, "moisture_dirt_goal__metric")
    dirt_back = _numeric(merged, "moisture_dirt_back__metric")
    out["race_moisture_goal"] = np.select([turf_mask, dirt_mask], [turf_goal, dirt_goal], default=np.nan)
    out["race_moisture_back"] = np.select([turf_mask, dirt_mask], [turf_back, dirt_back], default=np.nan)
    out["race_moisture_avg"] = out[["race_moisture_goal", "race_moisture_back"]].mean(axis=1).fillna(0.0)
    out["track_condition_available"] = (
        out["race_cushion_value"].gt(0) | out["race_moisture_avg"].gt(0)
    ).astype(float)

    condition_races = out.drop_duplicates(race_col).copy()
    condition_races["_venue_surface"] = (
        condition_races["_track_condition_venue"].astype(str)
        + "_"
        + condition_races["_track_condition_surface"].astype(str)
    )
    out["race_cushion_z_by_venue"] = _race_condition_z(
        condition_races, race_col, "race_cushion_value", ["_track_condition_venue"], [date_col, race_col]
    ).reindex(out[race_col]).to_numpy()
    out["race_moisture_z_by_venue_surface"] = _race_condition_z(
        condition_races, race_col, "race_moisture_avg", ["_venue_surface"], [date_col, race_col]
    ).reindex(out[race_col]).to_numpy()

    cushion = _numeric(out, "race_cushion_value")
    moisture = _numeric(out, "race_moisture_avg")
    out["race_high_cushion_flag"] = (turf_mask & cushion.ge(9.5)).astype(float)
    out["race_low_cushion_flag"] = (turf_mask & cushion.gt(0) & cushion.le(7.5)).astype(float)
    out["race_wet_moisture_flag"] = (moisture.ge(12.0)).astype(float)
    out["race_dry_moisture_flag"] = (moisture.gt(0) & moisture.le(8.0)).astype(float)

    if {"target_top3", "target_score"}.issubset(out.columns):
        popularity = _numeric(out, "人気")
        finish = _numeric(out, data_cfg["rank_column"])
        out["_track_condition_pop_outperform"] = (popularity - finish > 0).astype(float)
        out["_high_cushion_score"] = out["target_score"].where(out["race_high_cushion_flag"].gt(0))
        out["_high_cushion_top3"] = out["target_top3"].where(out["race_high_cushion_flag"].gt(0))
        out["_high_cushion_pop"] = out["_track_condition_pop_outperform"].where(out["race_high_cushion_flag"].gt(0))
        out["_low_cushion_score"] = out["target_score"].where(out["race_low_cushion_flag"].gt(0))
        out["_low_cushion_top3"] = out["target_top3"].where(out["race_low_cushion_flag"].gt(0))
        out["_wet_moisture_score"] = out["target_score"].where(out["race_wet_moisture_flag"].gt(0))
        out["_wet_moisture_top3"] = out["target_top3"].where(out["race_wet_moisture_flag"].gt(0))
        out["_dry_moisture_score"] = out["target_score"].where(out["race_dry_moisture_flag"].gt(0))
        out["_dry_moisture_top3"] = out["target_top3"].where(out["race_dry_moisture_flag"].gt(0))

        ordered = out.sort_values([horse_col, date_col, race_col], kind="mergesort")
        for source, dest, agg in [
            ("_high_cushion_top3", "horse_high_cushion_top3_rate", "mean"),
            ("_high_cushion_score", "horse_high_cushion_avg_score", "mean"),
            ("_high_cushion_pop", "horse_high_cushion_popularity_outperform_rate", "mean"),
            ("_low_cushion_top3", "horse_low_cushion_top3_rate", "mean"),
            ("_low_cushion_score", "horse_low_cushion_avg_score", "mean"),
            ("_wet_moisture_top3", "horse_wet_moisture_top3_rate", "mean"),
            ("_wet_moisture_score", "horse_wet_moisture_avg_score", "mean"),
            ("_dry_moisture_top3", "horse_dry_moisture_top3_rate", "mean"),
            ("_dry_moisture_score", "horse_dry_moisture_avg_score", "mean"),
        ]:
            out.loc[ordered.index, dest] = _rolling_previous_non_null(ordered, horse_col, source, agg).fillna(0.0)
        for source, dest in [
            ("_high_cushion_score", "horse_high_cushion_starts"),
            ("_low_cushion_score", "horse_low_cushion_starts"),
            ("_wet_moisture_score", "horse_wet_moisture_starts"),
            ("_dry_moisture_score", "horse_dry_moisture_starts"),
        ]:
            out.loc[ordered.index, dest] = _rolling_previous_non_null(ordered, horse_col, source, "count").fillna(0.0)

        out = out.drop(
            columns=[
                "_track_condition_pop_outperform",
                "_high_cushion_score",
                "_high_cushion_top3",
                "_high_cushion_pop",
                "_low_cushion_score",
                "_low_cushion_top3",
                "_wet_moisture_score",
                "_wet_moisture_top3",
                "_dry_moisture_score",
                "_dry_moisture_top3",
            ],
            errors="ignore",
        )

    out["horse_cushion_fit_score"] = (
        out["race_high_cushion_flag"] * out["horse_high_cushion_avg_score"]
        + out["race_low_cushion_flag"] * out["horse_low_cushion_avg_score"]
    ).fillna(0.0)
    out["horse_moisture_fit_score"] = (
        out["race_wet_moisture_flag"] * out["horse_wet_moisture_avg_score"]
        + out["race_dry_moisture_flag"] * out["horse_dry_moisture_avg_score"]
    ).fillna(0.0)
    out["horse_track_condition_fit_score"] = (
        0.6 * out["horse_cushion_fit_score"] + 0.4 * out["horse_moisture_fit_score"]
    ).fillna(0.0)

    return out.drop(
        columns=[
            "_track_condition_date",
            "_track_condition_venue",
            "_track_condition_surface",
            "_venue_surface",
            "moisture_turf_goal",
            "moisture_turf_back",
            "moisture_dirt_goal",
            "moisture_dirt_back",
        ],
        errors="ignore",
    )


def _load_track_condition_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path, encoding="utf-8-sig")
    aliases = {
        "date": ["date", "race_date", "日付", "年月日"],
        "venue": ["venue", "place", "場所", "競馬場"],
        "race_cushion_value": ["race_cushion_value", "cushion_value", "クッション値"],
        "moisture_turf_goal": ["moisture_turf_goal", "turf_moisture_goal", "芝含水率_ゴール前", "芝_ゴール前含水率"],
        "moisture_turf_back": ["moisture_turf_back", "turf_moisture_back", "芝含水率_4角", "芝_4角含水率"],
        "moisture_dirt_goal": ["moisture_dirt_goal", "dirt_moisture_goal", "ダート含水率_ゴール前", "ダート_ゴール前含水率"],
        "moisture_dirt_back": ["moisture_dirt_back", "dirt_moisture_back", "ダート含水率_4角", "ダート_4角含水率"],
    }
    rename: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in metrics.columns:
                rename[candidate] = canonical
                break
    metrics = metrics.rename(columns=rename)
    if "date" not in metrics.columns or "venue" not in metrics.columns:
        return pd.DataFrame()
    metrics["_track_condition_date"] = metrics["date"].map(_normalize_ymd_key)
    metrics["_track_condition_venue"] = metrics["venue"].astype("string").fillna("").str.strip()
    for col in [
        "race_cushion_value",
        "moisture_turf_goal",
        "moisture_turf_back",
        "moisture_dirt_goal",
        "moisture_dirt_back",
    ]:
        if col not in metrics.columns:
            metrics[col] = np.nan
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    return metrics.drop_duplicates(["_track_condition_date", "_track_condition_venue"], keep="last")


def _normalize_ymd_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 6:
        return "20" + digits
    if len(digits) >= 8:
        return digits[:8]
    return digits


def _race_condition_z(
    races: pd.DataFrame,
    race_col: str,
    value_col: str,
    group_cols: list[str],
    sort_cols: list[str],
) -> pd.Series:
    needed_cols = list(dict.fromkeys([race_col, value_col, *group_cols, *sort_cols]))
    frame = races[needed_cols].copy()
    values = _numeric(frame, value_col).replace(0, np.nan)
    frame[value_col] = values
    prev_mean = _expanding_previous_mean(frame, group_cols, value_col, sort_cols)
    prev_std = _expanding_previous_std(frame, group_cols, value_col, sort_cols).replace(0, np.nan)
    z = ((values - prev_mean) / prev_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.Series(z.to_numpy(), index=frame[race_col])


def _rolling_previous_non_null(df: pd.DataFrame, group_col: str, value_col: str, agg: str) -> pd.Series:
    values = _numeric(df, value_col)

    def calc(s: pd.Series) -> pd.Series:
        shifted = s.shift()
        if agg == "count":
            return shifted.notna().rolling(5, min_periods=1).sum()
        if agg == "sum":
            return shifted.rolling(5, min_periods=1).sum()
        return shifted.rolling(5, min_periods=1).mean()

    return values.groupby(df[group_col], sort=False).transform(calc)


def split_by_recent_dates(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    date_col = config["data"]["date_column"]
    frac = float(config["training"].get("test_recent_fraction", 0.2))
    dates = np.array(sorted(df[date_col].dropna().unique()))
    if len(dates) < 5:
        raise ValueError("Not enough unique race dates for a temporal split.")
    split_idx = max(1, min(len(dates) - 1, int(len(dates) * (1.0 - frac))))
    cutoff = int(dates[split_idx])
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    return train, test, cutoff
