from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ENTRY_CSV = "data/datasets/inference/weekly/entry_snapshot.csv"
DEFAULT_RESULTS_CSV = "data/processed/normalized/results.csv"
DEFAULT_RUNNERS_CSV = "data/processed/normalized/runners.csv"
DEFAULT_RACES_CSV = "data/processed/normalized/races.csv"


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _first_existing(columns: list[str], candidates: list[str], *, required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def _normalize_date(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if "." in text and len(text.split(".")) == 3:
        y, m, d = text.split(".")
        y = y if len(y) == 4 else "20" + y.zfill(2)
        return f"{y}{int(m):02d}{int(d):02d}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 6:
        return "20" + digits
    if len(digits) >= 8:
        return digits[:8]
    return digits


def _race_no_from_id(value: Any) -> float:
    text = "" if pd.isna(value) else str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 16:
        try:
            return float(int(digits[14:16]))
        except ValueError:
            return np.nan
    return np.nan


def _frame_bucket(frame: pd.Series) -> pd.Series:
    value = _num(frame)
    return np.select(
        [value <= 3, value.between(4, 6), value >= 7],
        ["inner", "middle", "outer"],
        default="unknown",
    )


def _series_from_candidates(frame: pd.DataFrame, candidates: list[str], default: Any = np.nan) -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return frame[col]
    return pd.Series(default, index=frame.index)


def _rank01(values: pd.Series, *, ascending: bool) -> pd.Series:
    numeric = _num(values)
    count = numeric.groupby(level=0).transform("count")
    rank = numeric.groupby(level=0).rank(ascending=ascending, method="average")
    return ((count - rank) / (count - 1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_completed_results(results_csv: Path, runners_csv: Path | None, races_csv: Path | None) -> pd.DataFrame:
    results = pd.read_csv(results_csv, encoding="utf-8-sig", low_memory=False, dtype={"血統登録番号": str})
    frame = results.copy()
    race_col = _first_existing(frame.columns.tolist(), ["レースID(新/馬番無)", "race_id"])

    if runners_csv and runners_csv.exists():
        runners = pd.read_csv(runners_csv, encoding="utf-8-sig", low_memory=False, dtype={"血統登録番号": str})
        runner_cols = [
            col
            for col in [
                race_col,
                "血統登録番号",
                "馬名",
                "日付",
                "場所",
                "芝・ダ",
                "距離",
                "馬番",
                "枠番",
                "人気",
                "単勝オッズ",
            ]
            if col in runners.columns
        ]
        frame = frame.merge(
            runners[runner_cols].drop_duplicates([race_col, "血統登録番号"]),
            on=[race_col, "血統登録番号"],
            how="left",
            suffixes=("", "_runner"),
        )
        for col in ["馬名", "日付", "場所", "芝・ダ", "距離", "馬番", "枠番", "人気", "単勝オッズ"]:
            runner_col = f"{col}_runner"
            if runner_col in frame.columns:
                if col in frame.columns:
                    frame[col] = frame[col].combine_first(frame[runner_col])
                else:
                    frame[col] = frame[runner_col]
                frame = frame.drop(columns=[runner_col])

    if races_csv and races_csv.exists():
        races = pd.read_csv(races_csv, encoding="utf-8-sig", low_memory=False)
        race_cols = [
            col
            for col in [race_col, "日付", "場所", "Ｒ", "レース名", "芝・ダ", "距離", "馬場状態", "頭数", "出走頭数"]
            if col in races.columns
        ]
        frame = frame.merge(
            races[race_cols].drop_duplicates(race_col),
            on=race_col,
            how="left",
            suffixes=("", "_race"),
        )
        for col in ["日付", "場所", "Ｒ", "レース名", "芝・ダ", "距離", "馬場状態", "頭数", "出走頭数"]:
            race_side = f"{col}_race"
            if race_side in frame.columns:
                if col in frame.columns:
                    frame[col] = frame[col].combine_first(frame[race_side])
                else:
                    frame[col] = frame[race_side]
                frame = frame.drop(columns=[race_side])

    frame["date_norm"] = frame["日付"].map(_normalize_date)
    frame["race_no"] = _num(frame[_first_existing(frame.columns.tolist(), ["Ｒ", "R"])])
    frame["finish"] = _num(frame[_first_existing(frame.columns.tolist(), ["確定着順", "finish"])])
    frame["top3"] = frame["finish"].between(1, 3)
    frame["win"] = frame["finish"].eq(1)
    frame["frame_bucket"] = _frame_bucket(_series_from_candidates(frame, ["枠番", "譫逡ｪ"]))
    field = _num(frame.get("出走頭数", frame.get("頭数", pd.Series(np.nan, index=frame.index)))).replace(0, np.nan)
    if field.isna().all():
        field = frame.groupby(race_col)[race_col].transform("size").replace(0, np.nan)
    corner = _num(frame.get("前4角.1", frame.get("4角.1", pd.Series(np.nan, index=frame.index))))
    corner_rate = (corner / field).replace([np.inf, -np.inf], np.nan)
    frame["corner4_rate"] = corner_rate
    frame["frontish"] = corner.le(3) | corner_rate.le(0.25)
    frame["stalkerish"] = corner_rate.gt(0.25) & corner_rate.le(0.45)
    frame["midpackish"] = corner_rate.gt(0.45) & corner_rate.lt(0.70)
    frame["closerish"] = corner_rate.ge(0.70)
    frame["position_gain"] = (corner - frame["finish"]).replace([np.inf, -np.inf], np.nan)
    popularity = _num(frame.get("人気", pd.Series(np.nan, index=frame.index)))
    field_for_pop = field.fillna(frame.groupby(race_col)[race_col].transform("size").replace(0, np.nan))
    frame["popularity_outperform"] = ((popularity - frame["finish"]) / field_for_pop).replace([np.inf, -np.inf], np.nan)
    frame["popularity_underperform"] = ((frame["finish"] - popularity) / field_for_pop).replace([np.inf, -np.inf], np.nan)
    frame["longshot"] = popularity.ge(6) | popularity.ge((field * 0.45).fillna(np.inf))
    frame["longshot_good_run"] = frame["longshot"] & (
        frame["finish"].le(4) | frame["finish"].le((field * 0.35).fillna(0))
    )
    return frame


def summarize_prior_bias(prior: pd.DataFrame) -> dict[str, Any]:
    if prior.empty:
        return {
            "same_day_bias_ready": 0,
            "same_day_prior_races": 0,
            "same_day_prior_runners": 0,
        }
    top3 = prior[prior["top3"]].copy()
    winners = prior[prior["win"]].copy()
    overall_top3_rate = float(prior["top3"].mean()) if len(prior) else np.nan

    def _bucket_top3_rate(flag_col: str) -> float:
        part = prior[prior[flag_col]]
        return float(part["top3"].mean()) if len(part) else np.nan

    def _bucket_pop_outperform(flag_col: str) -> float:
        part = prior[prior[flag_col]]
        return float(_num(part["popularity_outperform"]).mean()) if len(part) else np.nan

    def _bucket_longshot_good_run_rate(flag_col: str) -> float:
        part = prior[prior[flag_col] & prior["longshot"]]
        return float(part["longshot_good_run"].mean()) if len(part) else np.nan

    front_survival = _bucket_top3_rate("frontish")
    stalker_success = _bucket_top3_rate("stalkerish")
    midpack_success = _bucket_top3_rate("midpackish")
    closer_success = _bucket_top3_rate("closerish")
    front_advantage = front_survival - overall_top3_rate if pd.notna(front_survival) and pd.notna(overall_top3_rate) else np.nan
    stalker_advantage = stalker_success - overall_top3_rate if pd.notna(stalker_success) and pd.notna(overall_top3_rate) else np.nan
    midpack_advantage = midpack_success - overall_top3_rate if pd.notna(midpack_success) and pd.notna(overall_top3_rate) else np.nan
    closer_advantage = closer_success - overall_top3_rate if pd.notna(closer_success) and pd.notna(overall_top3_rate) else np.nan
    advantage_values = {
        "front": front_advantage,
        "stalker": stalker_advantage,
        "midpack": midpack_advantage,
        "closer": closer_advantage,
    }
    front_pop_advantage = _bucket_pop_outperform("frontish")
    stalker_pop_advantage = _bucket_pop_outperform("stalkerish")
    midpack_pop_advantage = _bucket_pop_outperform("midpackish")
    closer_pop_advantage = _bucket_pop_outperform("closerish")
    pop_advantage_values = {
        "front": front_pop_advantage,
        "stalker": stalker_pop_advantage,
        "midpack": midpack_pop_advantage,
        "closer": closer_pop_advantage,
    }
    valid_advantages = {k: v for k, v in advantage_values.items() if pd.notna(v)}
    valid_pop_advantages = {k: v for k, v in pop_advantage_values.items() if pd.notna(v)}
    if valid_advantages:
        best_style, best_value = max(valid_advantages.items(), key=lambda item: item[1])
        worst_style, worst_value = min(valid_advantages.items(), key=lambda item: item[1])
        bias_shape = best_style if best_value >= 0.06 and (best_value - worst_value) >= 0.10 else "mixed"
        bias_adverse_style = worst_style if worst_value <= -0.06 and (best_value - worst_value) >= 0.10 else "none"
    else:
        bias_shape = "unknown"
        bias_adverse_style = "unknown"
    if valid_pop_advantages:
        pop_best_style, pop_best_value = max(valid_pop_advantages.items(), key=lambda item: item[1])
        pop_worst_style, pop_worst_value = min(valid_pop_advantages.items(), key=lambda item: item[1])
        popularity_adjusted_shape = (
            pop_best_style if pop_best_value >= 0.08 and (pop_best_value - pop_worst_value) >= 0.12 else "mixed"
        )
    else:
        popularity_adjusted_shape = "unknown"

    out: dict[str, Any] = {
        "same_day_bias_ready": 1,
        "same_day_prior_races": int(prior["レースID(新/馬番無)"].nunique()) if "レースID(新/馬番無)" in prior else int(prior["race_no"].nunique()),
        "same_day_prior_runners": int(len(prior)),
        "same_day_top3_inner_rate": float((top3["frame_bucket"] == "inner").mean()) if len(top3) else np.nan,
        "same_day_top3_middle_rate": float((top3["frame_bucket"] == "middle").mean()) if len(top3) else np.nan,
        "same_day_top3_outer_rate": float((top3["frame_bucket"] == "outer").mean()) if len(top3) else np.nan,
        "same_day_winner_inner_rate": float((winners["frame_bucket"] == "inner").mean()) if len(winners) else np.nan,
        "same_day_winner_outer_rate": float((winners["frame_bucket"] == "outer").mean()) if len(winners) else np.nan,
        "same_day_front_top3_rate": float(top3["frontish"].mean()) if len(top3) else np.nan,
        "same_day_closer_top3_rate": float(top3["closerish"].mean()) if len(top3) else np.nan,
        "same_day_overall_top3_rate": overall_top3_rate,
        "same_day_front_runner_ratio": float(prior["frontish"].mean()) if len(prior) else np.nan,
        "same_day_closer_runner_ratio": float(prior["closerish"].mean()) if len(prior) else np.nan,
        "same_day_front_survival_rate": front_survival,
        "same_day_stalker_success_rate": stalker_success,
        "same_day_midpack_success_rate": midpack_success,
        "same_day_closer_success_rate": closer_success,
        "same_day_front_advantage_index": front_advantage,
        "same_day_stalker_advantage_index": stalker_advantage,
        "same_day_midpack_advantage_index": midpack_advantage,
        "same_day_closer_advantage_index": closer_advantage,
        "same_day_front_pop_outperform_index": front_pop_advantage,
        "same_day_stalker_pop_outperform_index": stalker_pop_advantage,
        "same_day_midpack_pop_outperform_index": midpack_pop_advantage,
        "same_day_closer_pop_outperform_index": closer_pop_advantage,
        "same_day_front_longshot_good_run_rate": _bucket_longshot_good_run_rate("frontish"),
        "same_day_stalker_longshot_good_run_rate": _bucket_longshot_good_run_rate("stalkerish"),
        "same_day_midpack_longshot_good_run_rate": _bucket_longshot_good_run_rate("midpackish"),
        "same_day_closer_longshot_good_run_rate": _bucket_longshot_good_run_rate("closerish"),
        "same_day_longshot_good_run_count": int(prior["longshot_good_run"].sum()) if "longshot_good_run" in prior else 0,
        "same_day_longshot_good_run_rate": float(prior.loc[prior["longshot"], "longshot_good_run"].mean()) if prior["longshot"].any() else np.nan,
        "same_day_front_longshot_evidence_count": int((prior["frontish"] & prior["longshot_good_run"]).sum()),
        "same_day_closer_longshot_evidence_count": int((prior["closerish"] & prior["longshot_good_run"]).sum()),
        "same_day_bias_pace_direction": (front_advantage if pd.notna(front_advantage) else 0.0) - (closer_advantage if pd.notna(closer_advantage) else 0.0),
        "same_day_pop_adjusted_pace_direction": (front_pop_advantage if pd.notna(front_pop_advantage) else 0.0) - (closer_pop_advantage if pd.notna(closer_pop_advantage) else 0.0),
        "same_day_bias_volatility": float(max(valid_advantages.values()) - min(valid_advantages.values())) if valid_advantages else np.nan,
        "same_day_pop_adjusted_bias_volatility": float(max(valid_pop_advantages.values()) - min(valid_pop_advantages.values())) if valid_pop_advantages else np.nan,
        "same_day_position_gain_top3_avg": float(_num(top3.get("position_gain", pd.Series(dtype=float))).mean()) if len(top3) else np.nan,
        "same_day_position_gain_winner_avg": float(_num(winners.get("position_gain", pd.Series(dtype=float))).mean()) if len(winners) else np.nan,
        "same_day_front_collapse_index": (overall_top3_rate - front_survival) if pd.notna(front_survival) and pd.notna(overall_top3_rate) else np.nan,
        "same_day_closer_blocked_index": (overall_top3_rate - closer_success) if pd.notna(closer_success) and pd.notna(overall_top3_rate) else np.nan,
        "same_day_bias_shape": bias_shape,
        "same_day_bias_adverse_style": bias_adverse_style,
        "same_day_pop_adjusted_bias_shape": popularity_adjusted_shape,
        "same_day_avg_top3_frame": float(_num(top3.get("枠番", pd.Series(dtype=float))).mean()) if len(top3) else np.nan,
        "same_day_avg_top3_horse_number": float(_num(top3.get("馬番", pd.Series(dtype=float))).mean()) if len(top3) else np.nan,
        "same_day_avg_winner_popularity": float(_num(winners.get("人気", pd.Series(dtype=float))).mean()) if len(winners) else np.nan,
    }
    if "平均1Fタイム" in prior and "基準タイム" in prior:
        out["same_day_track_speed_index"] = float((_num(prior["基準タイム"]) - _num(prior["平均1Fタイム"])).mean())
    else:
        out["same_day_track_speed_index"] = np.nan
    return out


def add_runner_bias_fit(entry: pd.DataFrame, summary: dict[str, Any]) -> pd.DataFrame:
    out = entry.copy()
    for key, value in summary.items():
        out[key] = value
    bucket = _frame_bucket(_series_from_candidates(out, ["枠番", "譫逡ｪ"]))
    inner = float(summary.get("same_day_top3_inner_rate", np.nan) or 0.0)
    middle = float(summary.get("same_day_top3_middle_rate", np.nan) or 0.0)
    outer = float(summary.get("same_day_top3_outer_rate", np.nan) or 0.0)
    out["same_day_frame_bucket"] = bucket
    out["same_day_frame_bias_fit_score"] = np.select(
        [bucket == "inner", bucket == "middle", bucket == "outer"],
        [inner - 1.0 / 3.0, middle - 1.0 / 3.0, outer - 1.0 / 3.0],
        default=0.0,
    )
    front_rate = float(summary.get("same_day_front_top3_rate", np.nan) or 0.0)
    closer_rate = float(summary.get("same_day_closer_top3_rate", np.nan) or 0.0)
    frontish = _num(out.get("front_running_tendency", pd.Series(0, index=out.index))).fillna(0.0)
    closing = _num(out.get("closing_tendency", pd.Series(0, index=out.index))).fillna(0.0)
    out["same_day_pace_bias_fit_score"] = (front_rate - closer_rate) * frontish + (closer_rate - front_rate) * closing
    out["same_day_bias_fit_score"] = out["same_day_frame_bias_fit_score"] + out["same_day_pace_bias_fit_score"]
    return out


def add_runner_bias_fit_from_columns(entry: pd.DataFrame) -> pd.DataFrame:
    out = entry.copy()
    bucket = _frame_bucket(_series_from_candidates(out, ["枠番", "譫逡ｪ"]))
    out["same_day_frame_bucket"] = bucket

    for col in [
        "same_day_top3_inner_rate",
        "same_day_top3_middle_rate",
        "same_day_top3_outer_rate",
        "same_day_front_top3_rate",
        "same_day_closer_top3_rate",
        "same_day_front_advantage_index",
        "same_day_stalker_advantage_index",
        "same_day_midpack_advantage_index",
        "same_day_closer_advantage_index",
        "same_day_front_pop_outperform_index",
        "same_day_stalker_pop_outperform_index",
        "same_day_midpack_pop_outperform_index",
        "same_day_closer_pop_outperform_index",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    ready = _num(out.get("same_day_bias_ready", pd.Series(0, index=out.index))).fillna(0.0).gt(0)
    inner = _num(out["same_day_top3_inner_rate"]).where(ready, 1.0 / 3.0).fillna(1.0 / 3.0)
    middle = _num(out["same_day_top3_middle_rate"]).where(ready, 1.0 / 3.0).fillna(1.0 / 3.0)
    outer = _num(out["same_day_top3_outer_rate"]).where(ready, 1.0 / 3.0).fillna(1.0 / 3.0)
    out["same_day_frame_bias_fit_score"] = np.select(
        [bucket == "inner", bucket == "middle", bucket == "outer"],
        [inner - 1.0 / 3.0, middle - 1.0 / 3.0, outer - 1.0 / 3.0],
        default=0.0,
    )

    front_rate = _num(out["same_day_front_top3_rate"]).fillna(0.0)
    closer_rate = _num(out["same_day_closer_top3_rate"]).fillna(0.0)
    frontish = _num(out.get("horse_front_run_rate_past5", out.get("front_running_tendency", pd.Series(0, index=out.index)))).fillna(0.0)
    stalker = _num(out.get("horse_stalker_rate_past5", pd.Series(0, index=out.index))).fillna(0.0)
    midpack = _num(out.get("horse_midpack_rate_past5", pd.Series(0, index=out.index))).fillna(0.0)
    closing = _num(out.get("horse_closer_rate_past5", out.get("closing_tendency", pd.Series(0, index=out.index)))).fillna(0.0)
    front_adv = _num(out["same_day_front_advantage_index"]).where(ready, 0.0).fillna(front_rate - closer_rate)
    stalker_adv = _num(out["same_day_stalker_advantage_index"]).where(ready, 0.0).fillna(0.0)
    midpack_adv = _num(out["same_day_midpack_advantage_index"]).where(ready, 0.0).fillna(0.0)
    closer_adv = _num(out["same_day_closer_advantage_index"]).where(ready, 0.0).fillna(closer_rate - front_rate)
    front_pop_adv = _num(out["same_day_front_pop_outperform_index"]).where(ready, 0.0).fillna(0.0)
    stalker_pop_adv = _num(out["same_day_stalker_pop_outperform_index"]).where(ready, 0.0).fillna(0.0)
    midpack_pop_adv = _num(out["same_day_midpack_pop_outperform_index"]).where(ready, 0.0).fillna(0.0)
    closer_pop_adv = _num(out["same_day_closer_pop_outperform_index"]).where(ready, 0.0).fillna(0.0)
    out["same_day_pace_bias_fit_score"] = (
        front_adv * frontish
        + stalker_adv * stalker
        + midpack_adv * midpack
        + closer_adv * closing
    )
    out["same_day_pop_adjusted_pace_fit_score"] = (
        front_pop_adv * frontish
        + stalker_pop_adv * stalker
        + midpack_pop_adv * midpack
        + closer_pop_adv * closing
    )
    load_direction = _num(out.get("same_day_bias_pace_direction", pd.Series(0, index=out.index))).where(ready, 0.0).fillna(0.0)
    out["same_day_projected_front_load_score"] = (-load_direction).clip(lower=0.0) * frontish
    out["same_day_projected_closer_load_score"] = load_direction.clip(lower=0.0) * closing
    out["same_day_adversity_fit_score"] = out["same_day_projected_front_load_score"] + out["same_day_projected_closer_load_score"]
    out["same_day_bias_fit_score"] = (
        out["same_day_frame_bias_fit_score"]
        + out["same_day_pace_bias_fit_score"]
        + 0.5 * out["same_day_pop_adjusted_pace_fit_score"]
    )
    return out


def build_features(
    entry_csv: Path,
    results_csv: Path,
    runners_csv: Path | None,
    races_csv: Path | None,
    output_csv: Path,
) -> dict[str, Any]:
    entry = pd.read_csv(entry_csv, encoding="utf-8-sig", low_memory=False)
    cols = entry.columns.tolist()
    date_col = _first_existing(cols, ["date", "日付", "譌･莉・", "譌･莉牢"])
    venue_col = _first_existing(cols, ["場所", "蝣ｴ謇"])
    race_no_col = _first_existing(cols, ["R", "Ｒ", "・ｲ"], required=False)
    race_id_col = _first_existing(cols, ["race_id", "レースID(新/馬番無)"], required=False)
    if not race_no_col and race_id_col:
        race_no_col = "_same_day_bias_race_no"
        entry[race_no_col] = entry[race_id_col].map(_race_no_from_id)
    surface_col = _first_existing(cols, ["芝・ダ", "闃昴・繝"], required=False)

    completed = load_completed_results(results_csv, runners_csv, races_csv)
    race_summaries = []
    summary_by_key: dict[tuple[str, str, float, str], dict[str, Any]] = {}

    completed_groups: dict[tuple[str, str, str], pd.DataFrame] = {}
    if "芝・ダ" in completed:
        group_cols = ["date_norm", "場所", "芝・ダ"]
        for key, group in completed.groupby(group_cols, dropna=False):
            completed_groups[(str(key[0]), str(key[1]), str(key[2]))] = group.copy()
    else:
        for key, group in completed.groupby(["date_norm", "場所"], dropna=False):
            completed_groups[(str(key[0]), str(key[1]), "")] = group.copy()

    for keys, part in entry.groupby([date_col, venue_col, race_no_col] if race_no_col else [date_col, venue_col], dropna=False):
        if race_no_col:
            date_value, venue_value, race_no_value = keys
        else:
            date_value, venue_value = keys
            race_no_value = np.nan
        date_norm = _normalize_date(date_value)
        race_no = pd.to_numeric(pd.Series([race_no_value]), errors="coerce").iloc[0]
        prior = pd.DataFrame()
        surface_key = ""
        if surface_col and surface_col in part:
            surfaces = part[surface_col].dropna().astype(str).unique().tolist()
            if len(surfaces) == 1:
                surface_key = surfaces[0]
                prior = completed_groups.get((date_norm, str(venue_value), surface_key), pd.DataFrame()).copy()
        if prior.empty:
            prior = completed_groups.get((date_norm, str(venue_value), ""), pd.DataFrame()).copy()
            surface_key = ""
        if pd.notna(race_no) and not prior.empty:
            prior = prior[prior["race_no"] < float(race_no)]
        summary = summarize_prior_bias(prior)
        summary.update(
            {
                "date": date_norm,
                "venue": str(venue_value),
                "race_no": None if pd.isna(race_no) else float(race_no),
                "surface": surface_key,
            }
        )
        race_summaries.append(summary)
        summary_by_key[(date_norm, str(venue_value), float(race_no) if pd.notna(race_no) else np.nan, surface_key)] = summary

    summaries = pd.DataFrame(race_summaries)
    entry["_same_day_bias_date"] = entry[date_col].map(_normalize_date)
    entry["_same_day_bias_venue"] = entry[venue_col].astype(str)
    entry["_same_day_bias_race_no_merge"] = pd.to_numeric(entry[race_no_col], errors="coerce") if race_no_col else np.nan
    if surface_col and surface_col in entry:
        entry["_same_day_bias_surface"] = entry[surface_col].astype(str)
    else:
        entry["_same_day_bias_surface"] = ""
    if not summaries.empty:
        out = entry.merge(
            summaries,
            left_on=[
                "_same_day_bias_date",
                "_same_day_bias_venue",
                "_same_day_bias_race_no_merge",
                "_same_day_bias_surface",
            ],
            right_on=["date", "venue", "race_no", "surface"],
            how="left",
        )
        missing = out["same_day_bias_ready"].isna()
        if missing.any() and (summaries["surface"].astype(str) == "").any():
            fallback = entry.loc[missing, [
                "_same_day_bias_date",
                "_same_day_bias_venue",
                "_same_day_bias_race_no_merge",
            ]].merge(
                summaries[summaries["surface"].astype(str) == ""],
                left_on=["_same_day_bias_date", "_same_day_bias_venue", "_same_day_bias_race_no_merge"],
                right_on=["date", "venue", "race_no"],
                how="left",
            )
            for col in summaries.columns:
                if col in out.columns and col in fallback.columns:
                    out.loc[missing, col] = fallback[col].to_numpy()
    else:
        out = entry.copy()
    out = add_runner_bias_fit_from_columns(out)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    drop_cols = [col for col in out.columns if col.startswith("_same_day_bias_")]
    out = out.drop(columns=drop_cols, errors="ignore")
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary_path = output_csv.with_name(output_csv.stem + "_race_summary.csv")
    pd.DataFrame(race_summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {
        "entry_csv": str(entry_csv),
        "output_csv": str(output_csv),
        "race_summary_csv": str(summary_path),
        "rows": int(len(out)),
        "races": int(len(race_summaries)),
        "ready_races": int(sum(row.get("same_day_bias_ready", 0) for row in race_summaries)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Append same-day track/draw/pace bias features from completed same-day races.")
    parser.add_argument("--entry-csv", default=DEFAULT_ENTRY_CSV)
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--runners-csv", default=DEFAULT_RUNNERS_CSV)
    parser.add_argument("--races-csv", default=DEFAULT_RACES_CSV)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    entry_csv = Path(args.entry_csv)
    output_csv = Path(args.output_csv) if args.output_csv else entry_csv.with_name(entry_csv.stem + "_with_same_day_bias.csv")
    result = build_features(
        entry_csv=entry_csv,
        results_csv=Path(args.results_csv),
        runners_csv=Path(args.runners_csv) if args.runners_csv else None,
        races_csv=Path(args.races_csv) if args.races_csv else None,
        output_csv=output_csv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
