from __future__ import annotations

import numpy as np
import pandas as pd


def add_body_workout_context_features(
    frame: pd.DataFrame,
    *,
    race_col: str = "レースID(新/馬番無)",
    date_col: str = "日付",
    horse_col: str = "血統登録番号",
    rank_col: str = "確定着順",
) -> pd.DataFrame:
    out = frame.copy()
    _add_body_context(out, race_col=race_col)
    _add_good_workout_match_context(out, race_col=race_col, date_col=date_col, horse_col=horse_col, rank_col=rank_col)
    return out


def _add_body_context(out: pd.DataFrame, *, race_col: str) -> None:
    prev_weight = _num(out.get("前走馬体重", pd.Series(np.nan, index=out.index)))
    prev_delta = _num(out.get("前走馬体重増減", pd.Series(np.nan, index=out.index)))
    interval = _num(out.get("間隔", pd.Series(np.nan, index=out.index)))
    rest_start = _num(out.get("休み明け～戦目", pd.Series(np.nan, index=out.index)))
    age = _num(out.get("年齢", pd.Series(np.nan, index=out.index)))
    sex = out.get("性別", pd.Series("", index=out.index)).astype("string")

    out["body_prev_weight"] = prev_weight
    out["body_prev_delta"] = prev_delta
    out["body_prev_delta_pct"] = (prev_delta / prev_weight.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["body_prev_abs_delta"] = prev_delta.abs()
    out["body_prev_large_gain_flag"] = (prev_delta >= 10).astype(float)
    out["body_prev_large_loss_flag"] = (prev_delta <= -10).astype(float)
    out["body_prev_extreme_change_flag"] = (prev_delta.abs() >= 16).astype(float)
    out["body_layoff_flag"] = ((interval >= 9) | (rest_start == 1)).astype(float)
    out["body_layoff_gain_flag"] = ((out["body_layoff_flag"] == 1) & (prev_delta >= 8)).astype(float)
    out["body_layoff_loss_flag"] = ((out["body_layoff_flag"] == 1) & (prev_delta <= -8)).astype(float)
    out["body_young_growth_gain_flag"] = ((age <= 4) & (prev_delta >= 8)).astype(float)
    out["body_female_large_loss_flag"] = (sex.str.contains("牝", na=False) & (prev_delta <= -10)).astype(float)
    out["body_small_horse_flag"] = (prev_weight < 440).astype(float)
    out["body_large_horse_flag"] = (prev_weight >= 500).astype(float)
    out["body_very_large_horse_flag"] = (prev_weight >= 520).astype(float)

    if race_col in out.columns:
        race_mean = prev_weight.groupby(out[race_col]).transform("mean")
        race_std = prev_weight.groupby(out[race_col]).transform("std").replace(0, np.nan)
        race_count = prev_weight.groupby(out[race_col]).transform("count")
        body_rank = prev_weight.groupby(out[race_col]).rank(ascending=False, method="average")
        out["body_weight_z_in_race"] = ((prev_weight - race_mean) / race_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["body_weight_rank_in_race"] = body_rank
        out["body_weight_percentile_in_race"] = (1.0 - ((body_rank - 1.0) / (race_count - 1.0))).replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        out["body_weight_z_in_race"] = 0.0
        out["body_weight_rank_in_race"] = np.nan
        out["body_weight_percentile_in_race"] = np.nan

    out["body_race_heavy_top3_flag"] = (out["body_weight_rank_in_race"] <= 3).astype(float)
    out["body_race_heavy_top5_flag"] = (out["body_weight_rank_in_race"] <= 5).astype(float)
    out["body_age2_flag"] = (age == 2).astype(float)
    out["body_age3_flag"] = (age == 3).astype(float)
    out["body_age2_big500_flag"] = ((age == 2) & (prev_weight >= 500)).astype(float)
    out["body_age2_big520_flag"] = ((age == 2) & (prev_weight >= 520)).astype(float)
    out["body_age2_small460_flag"] = ((age == 2) & (prev_weight <= 460)).astype(float)
    out["body_age2_race_heavy_top3_flag"] = ((age == 2) & (out["body_weight_rank_in_race"] <= 3)).astype(float)
    out["body_age2_race_heavy_top5_flag"] = ((age == 2) & (out["body_weight_rank_in_race"] <= 5)).astype(float)
    out["body_age3_big500_flag"] = ((age == 3) & (prev_weight >= 500)).astype(float)
    out["body_age3_big520_flag"] = ((age == 3) & (prev_weight >= 520)).astype(float)
    out["body_age3_small460_flag"] = ((age == 3) & (prev_weight <= 460)).astype(float)
    out["body_age3_race_heavy_top3_flag"] = ((age == 3) & (out["body_weight_rank_in_race"] <= 3)).astype(float)
    out["body_age3_race_heavy_top5_flag"] = ((age == 3) & (out["body_weight_rank_in_race"] <= 5)).astype(float)
    out["body_young_maturity_score"] = np.where(
        age == 2,
        0.65 * out["body_large_horse_flag"] + 0.35 * out["body_race_heavy_top3_flag"],
        np.where(age == 3, 0.45 * out["body_large_horse_flag"] + 0.55 * out["body_race_heavy_top5_flag"], 0.0),
    )

    workout_count = _num(out.get("workout_count", pd.Series(np.nan, index=out.index))).fillna(0)
    workout_days = _num(out.get("workout_latest_days_before_race", pd.Series(np.nan, index=out.index)))
    strong_finish = _num(out.get("workout_strong_finish_flag", pd.Series(0, index=out.index))).fillna(0)
    out["body_layoff_workout_count_fit"] = np.where(
        out["body_layoff_flag"] == 1,
        np.minimum(workout_count / 8.0, 1.5),
        0.0,
    )
    out["body_layoff_recent_workout_flag"] = ((out["body_layoff_flag"] == 1) & (workout_days <= 7)).astype(float)
    out["body_loss_with_strong_workout_flag"] = ((prev_delta <= -8) & (strong_finish > 0)).astype(float)


def _add_good_workout_match_context(
    out: pd.DataFrame,
    *,
    race_col: str,
    date_col: str,
    horse_col: str,
    rank_col: str,
) -> None:
    for col in [
        "horse_goodrun_same_pattern_count_past5",
        "horse_goodrun_same_pattern_rate_past5",
        "horse_goodrun_same_lap_count_past5",
        "horse_goodrun_same_lap_rate_past5",
        "horse_last_goodrun_same_pattern_flag",
        "horse_last_goodrun_same_lap_flag",
        "horse_workout_past_goodrun_match_score",
    ]:
        out[col] = 0.0

    required = {horse_col, date_col, race_col, rank_col, "workout_latest_pattern_bucket", "workout_latest_lap_group"}
    if not required.issubset(out.columns):
        return

    ordered = out.sort_values([horse_col, date_col, race_col], kind="mergesort")
    rank = _num(ordered[rank_col])
    top3 = rank <= 3
    pattern = ordered["workout_latest_pattern_bucket"].astype("string").fillna("unknown")
    lap = ordered["workout_latest_lap_group"].astype("string").fillna("unknown")

    pattern_count = pd.Series(0.0, index=ordered.index)
    pattern_rate = pd.Series(0.0, index=ordered.index)
    lap_count = pd.Series(0.0, index=ordered.index)
    lap_rate = pd.Series(0.0, index=ordered.index)
    last_pattern_same = pd.Series(0.0, index=ordered.index)
    last_lap_same = pd.Series(0.0, index=ordered.index)

    for _, idx in ordered.groupby(horse_col, sort=False).groups.items():
        ids = list(idx)
        good_patterns: list[str] = []
        good_laps: list[str] = []
        for i in ids:
            recent_patterns = good_patterns[-5:]
            recent_laps = good_laps[-5:]
            current_pattern = str(pattern.loc[i])
            current_lap = str(lap.loc[i])
            pc = sum(x == current_pattern for x in recent_patterns)
            lc = sum(x == current_lap for x in recent_laps)
            denom_p = len(recent_patterns) if recent_patterns else np.nan
            denom_l = len(recent_laps) if recent_laps else np.nan
            pattern_count.loc[i] = float(pc)
            lap_count.loc[i] = float(lc)
            pattern_rate.loc[i] = float(pc / denom_p) if denom_p == denom_p else 0.0
            lap_rate.loc[i] = float(lc / denom_l) if denom_l == denom_l else 0.0
            last_pattern_same.loc[i] = float(bool(recent_patterns and recent_patterns[-1] == current_pattern))
            last_lap_same.loc[i] = float(bool(recent_laps and recent_laps[-1] == current_lap))
            if bool(top3.loc[i]):
                good_patterns.append(str(current_pattern))
                good_laps.append(str(current_lap))

    out.loc[ordered.index, "horse_goodrun_same_pattern_count_past5"] = pattern_count.loc[ordered.index]
    out.loc[ordered.index, "horse_goodrun_same_pattern_rate_past5"] = pattern_rate.loc[ordered.index]
    out.loc[ordered.index, "horse_goodrun_same_lap_count_past5"] = lap_count.loc[ordered.index]
    out.loc[ordered.index, "horse_goodrun_same_lap_rate_past5"] = lap_rate.loc[ordered.index]
    out.loc[ordered.index, "horse_last_goodrun_same_pattern_flag"] = last_pattern_same.loc[ordered.index]
    out.loc[ordered.index, "horse_last_goodrun_same_lap_flag"] = last_lap_same.loc[ordered.index]
    out["horse_workout_past_goodrun_match_score"] = (
        0.45 * out["horse_goodrun_same_pattern_rate_past5"]
        + 0.25 * out["horse_goodrun_same_lap_rate_past5"]
        + 0.20 * out["horse_last_goodrun_same_pattern_flag"]
        + 0.10 * out["horse_last_goodrun_same_lap_flag"]
    ).fillna(0.0)


def _num(values: pd.Series) -> pd.Series:
    if values.dtype == object or str(values.dtype).startswith("string"):
        values = values.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(values, errors="coerce")
