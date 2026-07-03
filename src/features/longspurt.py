from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


RACE_REQUIRED_COLUMNS = [
    "race_id",
    "date",
    "venue",
    "surface",
    "distance",
    "going",
    "race_laps",
]

RUNNER_REQUIRED_COLUMNS = [
    "race_id",
    "horse_id",
    "date",
    "finish",
    "popularity",
    "field_size",
]


@dataclass(frozen=True)
class LongspurtConfig:
    group_cols: tuple[str, ...] = ("distance", "venue", "surface")
    fallback_group_cols: tuple[str, ...] = ("distance", "surface")
    min_group_size: int = 20
    fast_quantile: float = 0.25
    top_half_quantile: float = 0.50
    last2_bias_seconds: float = 0.35
    final1_deceleration_seconds: float = 0.50
    sustained_deceleration_limit: float = 0.30
    history_window: int | None = None
    min_longspurt_starts_for_flag: int = 2
    longspurt_advantage_threshold: float = 0.05


def parse_laps(value: object) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value if pd.notna(v)]
    return [float(part) for part in str(value).split("-") if part and part != "nan"]


def _numeric(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = df[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _percent_rank_fast(values: pd.Series) -> pd.Series:
    count = values.notna().sum()
    if count <= 1:
        return pd.Series(np.nan, index=values.index, dtype=float)
    ranks = values.rank(method="average", ascending=True)
    return (ranks - 1) / (count - 1)


def _add_group_stats(
    out: pd.DataFrame,
    config: LongspurtConfig,
    *,
    group_cols: tuple[str, ...],
    suffix: str,
) -> pd.DataFrame:
    grouped = out.groupby(list(group_cols), dropna=False)["last5f_sum"]
    out[f"last5f_mean_{suffix}"] = grouped.transform("mean")
    out[f"last5f_std_{suffix}"] = grouped.transform("std")
    out[f"last5f_group_count_{suffix}"] = grouped.transform("count")
    out[f"last5f_fast_percentile_{suffix}"] = grouped.transform(_percent_rank_fast)
    return out


def build_race_longspurt_features(
    races: pd.DataFrame,
    config: LongspurtConfig | None = None,
) -> pd.DataFrame:
    """Build race-level long-spurt shape and classification features.

    Lower last5f z-score/percentile means the final 5F was faster than peers.
    The primary peer group is distance x venue x surface; if the group is too
    sparse, stats fall back to distance x surface.
    """
    cfg = config or LongspurtConfig()
    _require_columns(races, RACE_REQUIRED_COLUMNS, "races")
    out = races.copy()

    laps = out["race_laps"].apply(parse_laps)
    out["lap_count"] = laps.apply(len)
    out["last5f_laps"] = laps.apply(lambda xs: xs[-5:] if len(xs) >= 5 else [])
    out["last5f_sum"] = out["last5f_laps"].apply(lambda xs: round(sum(xs), 1) if len(xs) == 5 else np.nan)
    out["last3f_sum"] = laps.apply(lambda xs: round(sum(xs[-3:]), 1) if len(xs) >= 3 else np.nan)

    out["fastest_lap_index_last5"] = out["last5f_laps"].apply(
        lambda xs: int(np.argmin(xs) + 1) if len(xs) == 5 else pd.NA
    )
    out["fastest_lap_value_last5"] = out["last5f_laps"].apply(lambda xs: min(xs) if len(xs) == 5 else np.nan)
    out["fastest_lap_remaining_f"] = out["fastest_lap_index_last5"].map(
        lambda idx: 6 - int(idx) if pd.notna(idx) else pd.NA
    )
    out["last5f_first3_avg"] = out["last5f_laps"].apply(
        lambda xs: float(np.mean(xs[:3])) if len(xs) == 5 else np.nan
    )
    out["last5f_last2_avg"] = out["last5f_laps"].apply(
        lambda xs: float(np.mean(xs[-2:])) if len(xs) == 5 else np.nan
    )
    out["last2_bias"] = out["last5f_first3_avg"] - out["last5f_last2_avg"]
    out["final1_deceleration"] = out["last5f_laps"].apply(
        lambda xs: xs[-1] - xs[-2] if len(xs) == 5 else np.nan
    )
    out["last5f_range"] = out["last5f_laps"].apply(
        lambda xs: max(xs) - min(xs) if len(xs) == 5 else np.nan
    )

    out = _add_group_stats(out, cfg, group_cols=cfg.group_cols, suffix="primary")
    out = _add_group_stats(out, cfg, group_cols=cfg.fallback_group_cols, suffix="fallback")
    sparse = out["last5f_group_count_primary"] < cfg.min_group_size
    out["last5f_mean"] = out["last5f_mean_primary"].where(~sparse, out["last5f_mean_fallback"])
    out["last5f_std"] = out["last5f_std_primary"].where(~sparse, out["last5f_std_fallback"])
    out["last5f_group_count"] = out["last5f_group_count_primary"].where(
        ~sparse, out["last5f_group_count_fallback"]
    )
    out["last5f_fast_percentile"] = out["last5f_fast_percentile_primary"].where(
        ~sparse, out["last5f_fast_percentile_fallback"]
    )
    out["last5f_vs_mean"] = out["last5f_sum"] - out["last5f_mean"]
    out["last5f_z"] = out["last5f_vs_mean"] / out["last5f_std"].replace(0, np.nan)

    out["last5f_top25"] = out["last5f_fast_percentile"].le(cfg.fast_quantile)
    out["last5f_top50"] = out["last5f_fast_percentile"].le(cfg.top_half_quantile)
    out["fastest_is_early_mid_last5"] = pd.to_numeric(out["fastest_lap_index_last5"], errors="coerce").between(1, 3)
    out["fastest_is_last2"] = pd.to_numeric(out["fastest_lap_index_last5"], errors="coerce").between(4, 5)
    out["last2_heavy"] = out["fastest_is_last2"] & out["last2_bias"].ge(cfg.last2_bias_seconds)
    out["strong_final_deceleration"] = out["final1_deceleration"].ge(cfg.final1_deceleration_seconds)
    out["sustained_shape"] = out["last5f_range"].le(0.7) & out["final1_deceleration"].le(
        cfg.sustained_deceleration_limit
    )

    conditions = [
        out["last5f_top25"] & out["fastest_is_early_mid_last5"] & ~out["last2_heavy"],
        out["last5f_top25"] & out["last2_heavy"],
        out["strong_final_deceleration"] & out["last5f_top50"],
        out["last5f_top50"] & out["sustained_shape"],
    ]
    choices = ["ロンスパ戦", "瞬発戦", "消耗戦", "持続戦"]
    out["race_longspurt_type"] = np.select(conditions, choices, default="標準戦")
    out["is_longspurt_race"] = out["race_longspurt_type"].eq("ロンスパ戦")
    out["is_instant_race"] = out["race_longspurt_type"].eq("瞬発戦")
    out["is_attrition_race"] = out["race_longspurt_type"].eq("消耗戦")
    out["is_sustained_race"] = out["race_longspurt_type"].eq("持続戦")
    return out


def add_runner_longspurt_outcomes(
    runners: pd.DataFrame,
    race_features: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(runners, RUNNER_REQUIRED_COLUMNS, "runners")
    needed_race_cols = [
        "race_id",
        "last5f_sum",
        "last5f_z",
        "last5f_fast_percentile",
        "last5f_top25",
        "race_longspurt_type",
        "is_longspurt_race",
    ]
    _require_columns(race_features, needed_race_cols, "race_features")

    out = runners.copy()
    out = out.merge(race_features[needed_race_cols], on="race_id", how="left")
    finish = _numeric(out, "finish")
    popularity = _numeric(out, "popularity")
    field_size = _numeric(out, "field_size").replace(0, np.nan)
    corner4 = _numeric(out, "corner4")
    corner3 = _numeric(out, "corner3")
    corner2 = _numeric(out, "corner2")
    corner1 = _numeric(out, "corner1")
    final3f_rank = _numeric(out, "final3f_rank")

    out["top3"] = finish.between(1, 3)
    out["win"] = finish.eq(1)
    out["finish_score"] = ((field_size + 1 - finish) / field_size).clip(0.0, 1.0)
    out["popularity_outperform"] = (popularity - finish).fillna(0.0)
    out["corner4_to_finish_gain"] = (corner4 - finish).fillna(0.0)
    out["corner4_position_rate"] = (corner4 / field_size).replace([np.inf, -np.inf], np.nan)

    first_corner = pd.concat([corner1, corner2, corner3, corner4], axis=1).bfill(axis=1).iloc[:, 0]
    out["early_position_gain_to_4c"] = (first_corner - corner4).fillna(0.0)
    out["early_move_flag"] = (
        out["early_position_gain_to_4c"].ge(2.0)
        & out["corner4_position_rate"].le(0.60)
    )
    out["late_burst_only_flag"] = (
        out["early_position_gain_to_4c"].le(0.0)
        & out["corner4_to_finish_gain"].ge(5.0)
    )
    out["final3f_rank_score"] = ((field_size + 1 - final3f_rank) / field_size).clip(0.0, 1.0).fillna(0.0)
    return out


def build_horse_longspurt_features(
    runner_features: pd.DataFrame,
    config: LongspurtConfig | None = None,
) -> pd.DataFrame:
    """Create pre-race horse long-spurt performance features.

    Every row receives features computed from the same horse's previous races
    only. This avoids using the target race result as a feature.
    """
    cfg = config or LongspurtConfig()
    _require_columns(
        runner_features,
        [
            "horse_id",
            "date",
            "race_id",
            "top3",
            "finish_score",
            "is_longspurt_race",
            "last5f_top25",
            "corner4_to_finish_gain",
            "early_move_flag",
            "late_burst_only_flag",
            "final3f_rank_score",
            "popularity_outperform",
        ],
        "runner_features",
    )
    out = runner_features.sort_values(["horse_id", "date", "race_id"], kind="mergesort").copy()
    grouped = out.groupby("horse_id", sort=False)

    history_cols = {
        "longspurt_starts": out["is_longspurt_race"].astype(float),
        "longspurt_top3": (out["is_longspurt_race"] & out["top3"]).astype(float),
        "fast_last5f_starts": out["last5f_top25"].astype(float),
        "fast_last5f_top3": (out["last5f_top25"] & out["top3"]).astype(float),
        "longspurt_finish_score_sum": out["finish_score"].where(out["is_longspurt_race"], 0.0),
        "fast_last5f_finish_score_sum": out["finish_score"].where(out["last5f_top25"], 0.0),
        "corner4_gain_sum": out["corner4_to_finish_gain"].where(out["is_longspurt_race"], 0.0),
        "early_move_sum": out["early_move_flag"].astype(float).where(out["is_longspurt_race"], 0.0),
        "late_burst_only_sum": out["late_burst_only_flag"].astype(float).where(out["is_longspurt_race"], 0.0),
        "final3f_rank_score_sum": out["final3f_rank_score"].where(out["is_longspurt_race"], 0.0),
        "popularity_outperform_sum": out["popularity_outperform"].where(out["is_longspurt_race"], 0.0),
    }
    hist = pd.DataFrame(history_cols, index=out.index)

    if cfg.history_window:
        rolled = hist.groupby(out["horse_id"], sort=False).transform(
            lambda s: s.shift().rolling(cfg.history_window, min_periods=1).sum()
        )
    else:
        rolled = hist.groupby(out["horse_id"], sort=False).cumsum()
        rolled = rolled.groupby(out["horse_id"], sort=False).shift(fill_value=0.0)

    for col in rolled.columns:
        out[f"horse_{col}_prev"] = rolled[col].astype(float)

    long_starts = out["horse_longspurt_starts_prev"].replace(0, np.nan)
    fast_starts = out["horse_fast_last5f_starts_prev"].replace(0, np.nan)
    out["horse_longspurt_top3_rate"] = (out["horse_longspurt_top3_prev"] / long_starts).fillna(0.0)
    out["horse_fast_last5f_top3_rate"] = (out["horse_fast_last5f_top3_prev"] / fast_starts).fillna(0.0)
    out["horse_longspurt_avg_finish_score"] = (
        out["horse_longspurt_finish_score_sum_prev"] / long_starts
    ).fillna(0.0)
    out["horse_fast_last5f_avg_finish_score"] = (
        out["horse_fast_last5f_finish_score_sum_prev"] / fast_starts
    ).fillna(0.0)
    out["horse_longspurt_corner4_gain_avg"] = (out["horse_corner4_gain_sum_prev"] / long_starts).fillna(0.0)
    out["horse_longspurt_early_move_rate"] = (out["horse_early_move_sum_prev"] / long_starts).fillna(0.0)
    out["horse_longspurt_late_burst_only_rate"] = (
        out["horse_late_burst_only_sum_prev"] / long_starts
    ).fillna(0.0)
    out["horse_longspurt_final3f_rank_score_avg"] = (
        out["horse_final3f_rank_score_sum_prev"] / long_starts
    ).fillna(0.0)
    out["horse_longspurt_popularity_outperform_avg"] = (
        out["horse_popularity_outperform_sum_prev"] / long_starts
    ).fillna(0.0)

    out["horse_longspurt_score"] = (
        35.0 * out["horse_longspurt_top3_rate"]
        + 20.0 * out["horse_fast_last5f_top3_rate"]
        + 12.0 * out["horse_longspurt_avg_finish_score"]
        + 10.0 * out["horse_longspurt_early_move_rate"]
        + 8.0 * out["horse_longspurt_final3f_rank_score_avg"]
        + 8.0 * (out["horse_longspurt_corner4_gain_avg"].clip(lower=-3.0, upper=6.0) + 3.0) / 9.0
        + 7.0 * ((out["horse_longspurt_popularity_outperform_avg"].clip(lower=-5.0, upper=5.0) + 5.0) / 10.0)
        - 10.0 * out["horse_longspurt_late_burst_only_rate"]
    ).clip(0.0, 100.0)
    out["horse_longspurt_aptitude_flag"] = (
        out["horse_longspurt_starts_prev"].ge(cfg.min_longspurt_starts_for_flag)
        & out["horse_longspurt_score"].ge(55.0)
        & out["horse_longspurt_late_burst_only_rate"].le(0.45)
    )
    return out.sort_index()


def build_longspurt_feature_set(
    races: pd.DataFrame,
    runners: pd.DataFrame,
    config: LongspurtConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    race_features = build_race_longspurt_features(races, config)
    runner_features = add_runner_longspurt_outcomes(runners, race_features)
    horse_features = build_horse_longspurt_features(runner_features, config)
    return race_features, horse_features
