from __future__ import annotations

import numpy as np
import pandas as pd


def add_workout_pattern_history_features(
    frame: pd.DataFrame,
    *,
    race_col: str,
    date_col: str,
    rank_col: str,
    odds_col: str | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    out["_workout_hist_race"] = out[race_col].astype("string")
    out["_workout_hist_date"] = pd.to_numeric(out[date_col], errors="coerce")
    rank = pd.to_numeric(out[rank_col], errors="coerce")
    out["_workout_hist_win"] = (rank == 1).astype(float)
    out["_workout_hist_top3"] = (rank <= 3).astype(float)
    if "target_score" in out.columns:
        out["_workout_hist_score"] = pd.to_numeric(out["target_score"], errors="coerce")
    else:
        field_size = rank.groupby(out["_workout_hist_race"]).transform("max").replace(0, np.nan)
        out["_workout_hist_score"] = ((field_size + 1.0 - rank) / field_size).replace([np.inf, -np.inf], np.nan)
    if odds_col and odds_col in out.columns:
        odds = pd.to_numeric(out[odds_col], errors="coerce")
        out["_workout_hist_win_return"] = np.where(rank == 1, odds, 0.0)
    else:
        out["_workout_hist_win_return"] = np.nan

    out["race_course_key"] = _race_course_key(out)
    out["trainer_key"] = _first_series(out, ["trainer_code", "調教師コード", "隱ｿ謨吝ｸｫ繧ｳ繝ｼ繝・"])
    out["horse_key"] = _first_series(out, ["horse_id", "血統登録番号", "陦邨ｱ逋ｻ骭ｲ逡ｪ蜿ｷ"])

    specs = [
        ("workout_course_pattern", ["race_course_key", "workout_latest_pattern_bucket"]),
        ("workout_course_lap", ["race_course_key", "workout_latest_lap_group"]),
        ("workout_trainer_pattern", ["trainer_key", "workout_latest_pattern_bucket"]),
        ("workout_trainer_lap", ["trainer_key", "workout_latest_lap_group"]),
        ("workout_horse_pattern", ["horse_key", "workout_latest_pattern_bucket"]),
        ("workout_horse_lap", ["horse_key", "workout_latest_lap_group"]),
    ]
    for prefix, group_cols in specs:
        if all(col in out.columns for col in group_cols):
            out = _add_prior_group_rates(out, group_cols, prefix)

    return out.drop(columns=[c for c in out.columns if c.startswith("_workout_hist_")], errors="ignore")


def _add_prior_group_rates(frame: pd.DataFrame, group_cols: list[str], prefix: str) -> pd.DataFrame:
    out = frame.copy()
    valid = out.dropna(subset=group_cols + ["_workout_hist_race", "_workout_hist_date"]).copy()
    if valid.empty:
        _fill_defaults(out, prefix)
        return out

    race_level = (
        valid.groupby(group_cols + ["_workout_hist_date", "_workout_hist_race"], dropna=False)
        .agg(
            _seg_starts=("_workout_hist_win", "size"),
            _seg_wins=("_workout_hist_win", "sum"),
            _seg_top3=("_workout_hist_top3", "sum"),
            _seg_score=("_workout_hist_score", "sum"),
            _seg_return=("_workout_hist_win_return", "sum"),
        )
        .reset_index()
        .sort_values(group_cols + ["_workout_hist_date", "_workout_hist_race"], kind="mergesort")
    )

    grouped = race_level.groupby(group_cols, dropna=False, sort=False)
    prior_starts = grouped["_seg_starts"].cumsum().groupby([race_level[col] for col in group_cols], sort=False).shift(fill_value=0)
    prior_wins = grouped["_seg_wins"].cumsum().groupby([race_level[col] for col in group_cols], sort=False).shift(fill_value=0.0)
    prior_top3 = grouped["_seg_top3"].cumsum().groupby([race_level[col] for col in group_cols], sort=False).shift(fill_value=0.0)
    prior_score = grouped["_seg_score"].cumsum().groupby([race_level[col] for col in group_cols], sort=False).shift(fill_value=0.0)
    prior_return = grouped["_seg_return"].cumsum().groupby([race_level[col] for col in group_cols], sort=False).shift(fill_value=0.0)

    race_level[f"{prefix}_starts"] = prior_starts.astype(float)
    denom = prior_starts.replace(0, np.nan)
    race_level[f"{prefix}_win_rate"] = (prior_wins / denom).fillna(0.0).astype(float)
    race_level[f"{prefix}_top3_rate"] = (prior_top3 / denom).fillna(0.0).astype(float)
    race_level[f"{prefix}_avg_score"] = (prior_score / denom).fillna(0.0).astype(float)
    race_level[f"{prefix}_win_roi"] = (prior_return / denom).fillna(0.0).astype(float)

    feature_cols = [
        f"{prefix}_starts",
        f"{prefix}_win_rate",
        f"{prefix}_top3_rate",
        f"{prefix}_avg_score",
        f"{prefix}_win_roi",
    ]
    keyed = race_level[group_cols + ["_workout_hist_race", *feature_cols]]
    out = out.merge(keyed, how="left", on=group_cols + ["_workout_hist_race"])
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _fill_defaults(frame: pd.DataFrame, prefix: str) -> None:
    for suffix in ["starts", "win_rate", "top3_rate", "avg_score", "win_roi"]:
        frame[f"{prefix}_{suffix}"] = 0.0


def _race_course_key(frame: pd.DataFrame) -> pd.Series:
    existing = _first_existing(frame, ["race_course_key"], required=False)
    if existing:
        return frame[existing].astype("string")
    venue = _first_series(frame, ["場所", "蝣ｴ謇", "venue"])
    surface = _first_series(frame, ["芝・ダ", "闃昴・繝", "surface"])
    distance = _first_series(frame, ["距離", "distance"])
    return venue.astype("string").fillna("") + "_" + surface.astype("string").fillna("") + "_" + distance.astype("string").fillna("")


def _first_existing(frame: pd.DataFrame, candidates: list[str], *, required: bool = True) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    if required:
        raise ValueError(f"None of these columns exist: {candidates}")
    return None


def _first_series(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    col = _first_existing(frame, candidates, required=False)
    if col:
        return frame[col]
    return pd.Series(pd.NA, index=frame.index, dtype="string")
