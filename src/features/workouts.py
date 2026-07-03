from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ALIASES: dict[str, tuple[str, ...]] = {
    "race_id": ("race_id", "レースID(新/馬番無)", "レースID"),
    "horse_id": ("horse_id", "血統登録番号"),
    "horse_number": ("horse_number", "馬番", "馬 番"),
    "trainer_code": ("trainer_code", "調教師コード"),
    "workout_date": ("workout_date", "追切日", "調教日"),
    "race_date": ("race_date", "日付", "レース日"),
    "course": ("course", "調教コース", "コース"),
    "distance_f": ("distance_f", "距離F", "調教距離F"),
    "total_time_sec": ("total_time_sec", "全体時計", "時計", "追切時計"),
    "final_1f_sec": ("final_1f_sec", "終い1F", "ラスト1F"),
    "final_2f_sec": ("final_2f_sec", "終い2F", "ラスト2F"),
    "final_3f_sec": ("final_3f_sec", "終い3F", "ラスト3F"),
    "intensity": ("intensity", "強度"),
    "partner_result": ("partner_result", "併せ結果"),
}


def build_workout_features_from_file(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    return build_workout_features(frame)


def build_workout_features(workouts: pd.DataFrame) -> pd.DataFrame:
    src = _canonicalize_columns(workouts)
    if "race_id" not in src.columns:
        raise ValueError("workout CSV requires race_id or レースID(新/馬番無)")
    if "horse_id" not in src.columns and "horse_number" not in src.columns:
        raise ValueError("workout CSV requires horse_id/血統登録番号 or horse_number/馬番")

    out = src.copy()
    out["race_id"] = out["race_id"].astype("string")
    for col in ["horse_id", "horse_number", "trainer_code", "course", "intensity", "partner_result"]:
        if col in out.columns:
            out[col] = out[col].astype("string")

    out["workout_date_dt"] = _to_datetime(out.get("workout_date"))
    out["race_date_dt"] = _to_datetime(out.get("race_date"))
    for col in ["distance_f", "total_time_sec", "final_1f_sec", "final_2f_sec", "final_3f_sec"]:
        if col in out.columns:
            out[col] = _seconds(out[col]) if col.endswith("_sec") else pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    out["course_bucket"] = _course_bucket(out.get("course", pd.Series("", index=out.index)))
    out["finish_gain_sec"] = _finish_gain(out["final_1f_sec"], out["final_2f_sec"])
    out["penultimate_1f_sec"] = out["final_2f_sec"] - out["final_1f_sec"]
    out["lap_group"] = _lap_group(out["penultimate_1f_sec"], out["final_1f_sec"])
    out["fast_final_flag"] = (out["final_1f_sec"] <= 12.2).astype(float)
    out["strong_finish_flag"] = (out["finish_gain_sec"] >= 0.2).astype(float)
    out["partner_win_flag"] = _partner_win_flag(out.get("partner_result", pd.Series("", index=out.index)))
    out["days_before_race"] = (out["race_date_dt"] - out["workout_date_dt"]).dt.days
    out["pattern_bucket"] = _pattern_bucket(out)

    out = _add_shifted_baselines(out, ["trainer_code", "course_bucket"], "trainer")
    out = _add_shifted_baselines(out, ["course_bucket"], "course")

    key_cols = ["race_id"]
    if "horse_id" in out.columns:
        key_cols.append("horse_id")
    elif "horse_number" in out.columns:
        key_cols.append("horse_number")

    sort_cols = key_cols + ["workout_date_dt"]
    out = out.sort_values(sort_cols, kind="mergesort")
    rows: list[dict[str, object]] = []
    for keys, group in out.groupby(key_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(key_cols, keys))
        latest = group.iloc[-1]
        row.update(_latest_features(latest))
        row.update(_aggregate_features(group))
        rows.append(row)

    features = pd.DataFrame(rows)
    for col in ["race_id", "horse_id", "horse_number"]:
        if col in features.columns:
            features[col] = features[col].astype("string")
    return features


def merge_workout_features(
    frame: pd.DataFrame,
    workout_features: pd.DataFrame,
    *,
    race_col: str,
    horse_id_col: str | None = None,
    horse_number_col: str | None = None,
) -> pd.DataFrame:
    left = frame.copy()
    right = workout_features.copy()
    left["_workout_race_id"] = left[race_col].astype("string")
    right["_workout_race_id"] = right["race_id"].astype("string")
    left_on = ["_workout_race_id"]
    right_on = ["_workout_race_id"]

    if horse_id_col and horse_id_col in left.columns and "horse_id" in right.columns:
        left["_workout_horse_id"] = left[horse_id_col].astype("string")
        right["_workout_horse_id"] = right["horse_id"].astype("string")
        left_on.append("_workout_horse_id")
        right_on.append("_workout_horse_id")
    elif horse_number_col and horse_number_col in left.columns and "horse_number" in right.columns:
        left["_workout_horse_number"] = left[horse_number_col].astype("string")
        right["_workout_horse_number"] = right["horse_number"].astype("string")
        left_on.append("_workout_horse_number")
        right_on.append("_workout_horse_number")
    else:
        raise ValueError("No merge key found. Provide horse_id_col or horse_number_col that exists in both frames.")

    drop_right = [c for c in ["race_id", "horse_id", "horse_number"] if c in right.columns]
    merged = left.merge(right.drop(columns=drop_right), how="left", left_on=left_on, right_on=right_on)
    return merged.drop(columns=[c for c in merged.columns if c.startswith("_workout_")])


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rename: dict[str, str] = {}
    existing = set(out.columns)
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in existing:
                rename[alias] = canonical
                break
    return out.rename(columns=rename)


def _to_datetime(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(pd.NaT)
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    text = text.where(~text.str.fullmatch(r"\d{6}", na=False), "20" + text)
    return pd.to_datetime(text, errors="coerce", format="mixed")


def _seconds(values: pd.Series) -> pd.Series:
    def parse(value: object) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if not text:
            return np.nan
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return float(minutes) * 60.0 + float(seconds)
        return float(text)

    return values.map(parse).astype(float)


def _course_bucket(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("")
    return np.select(
        [
            text.str.contains("坂|坂路|美坂|栗坂|hill", regex=True, case=False),
            text.str.contains("CW|ＣＷ|W|ウッド|南W|栗東CW|美浦W|wood", regex=True, case=False),
            text.str.contains("芝", regex=True),
            text.str.contains("ダ|D|ポリ|P", regex=True),
        ],
        ["hill", "wood", "turf", "dirt_poly"],
        default="other",
    )


def _finish_gain(final_1f: pd.Series, final_2f: pd.Series) -> pd.Series:
    prior_1f = final_2f - final_1f
    return (prior_1f - final_1f).replace([np.inf, -np.inf], np.nan)


def _partner_win_flag(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("")
    return text.str.contains("先着|優勢|勝", regex=True).astype(float)


def _lap_group(penultimate_1f: pd.Series, final_1f: pd.Series) -> pd.Series:
    second = pd.to_numeric(penultimate_1f, errors="coerce")
    last = pd.to_numeric(final_1f, errors="coerce")
    accel = last < second
    decel = last > second
    second_12 = second.between(12.0, 12.999, inclusive="both")
    last_12 = last.between(12.0, 12.999, inclusive="both")
    second_11 = second.between(11.0, 11.999, inclusive="both")
    last_11 = last.between(11.0, 11.999, inclusive="both")

    return pd.Series(
        np.select(
            [
                accel & last_11,
                decel & second_11 & last_11,
                accel & second_12 & last_12,
                decel & second_12 & last_12,
                accel & ~second_12 & ~second_11 & last_12,
                decel & second_12 & ~last_12 & ~last_11,
            ],
            ["A3", "B3", "A2", "B2", "A1", "B1"],
            default="other",
        ),
        index=final_1f.index,
    )


def _pattern_bucket(frame: pd.DataFrame) -> pd.Series:
    intensity = frame.get("intensity", pd.Series("", index=frame.index)).astype("string").fillna("")
    hard = intensity.str.contains("一杯|強め|末強|G前", regex=True)
    light = intensity.str.contains("馬なり|楽", regex=True)
    strength = np.select([hard, light], ["hard", "easy"], default="normal")
    finish = np.select(
        [frame["strong_finish_flag"] == 1, frame["fast_final_flag"] == 1],
        ["strong_finish", "fast_final"],
        default="steady",
    )
    return pd.Series(
        frame["course_bucket"].astype(str)
        + "_"
        + strength.astype(str)
        + "_"
        + finish.astype(str)
        + "_"
        + frame["lap_group"].astype(str),
        index=frame.index,
    )


def _add_shifted_baselines(frame: pd.DataFrame, group_cols: Iterable[str], prefix: str) -> pd.DataFrame:
    out = frame.copy()
    group_cols = [col for col in group_cols if col in out.columns]
    if not group_cols:
        return out
    ordered = out.sort_values([*group_cols, "workout_date_dt"], kind="mergesort")
    grouped = ordered.groupby(group_cols, dropna=False, sort=False)
    for source, stem in [("total_time_sec", "total"), ("final_1f_sec", "final1")]:
        values = pd.to_numeric(ordered[source], errors="coerce")
        valid = values.notna().astype(float)
        filled = values.fillna(0.0)
        prior_count = valid.groupby([ordered[col] for col in group_cols], sort=False).cumsum() - valid
        prior_sum = filled.groupby([ordered[col] for col in group_cols], sort=False).cumsum() - filled
        prior_sum_sq = (filled * filled).groupby([ordered[col] for col in group_cols], sort=False).cumsum() - (filled * filled)
        mean = prior_sum / prior_count.replace(0, np.nan)
        variance = (prior_sum_sq / prior_count.replace(0, np.nan)) - (mean * mean)
        std = variance.clip(lower=0.0).pow(0.5)
        mean = mean.where(prior_count >= 5)
        std = std.where(prior_count >= 5)
        z = ((values - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out.loc[ordered.index, f"{stem}_vs_{prefix}_z"] = z
    return out


def _latest_features(latest: pd.Series) -> dict[str, object]:
    return {
        "workout_latest_date": latest.get("workout_date"),
        "workout_latest_days_before_race": latest.get("days_before_race"),
        "workout_latest_course_bucket": latest.get("course_bucket"),
        "workout_latest_lap_group": latest.get("lap_group"),
        "workout_latest_pattern_bucket": latest.get("pattern_bucket"),
        "workout_latest_total_time_sec": latest.get("total_time_sec"),
        "workout_latest_penultimate_1f_sec": latest.get("penultimate_1f_sec"),
        "workout_latest_final_1f_sec": latest.get("final_1f_sec"),
        "workout_latest_final_2f_sec": latest.get("final_2f_sec"),
        "workout_latest_final_3f_sec": latest.get("final_3f_sec"),
        "workout_latest_finish_gain_sec": latest.get("finish_gain_sec"),
        "workout_latest_total_vs_trainer_z": latest.get("total_vs_trainer_z"),
        "workout_latest_final1_vs_trainer_z": latest.get("final1_vs_trainer_z"),
        "workout_latest_total_vs_course_z": latest.get("total_vs_course_z"),
        "workout_latest_final1_vs_course_z": latest.get("final1_vs_course_z"),
    }


def _aggregate_features(group: pd.DataFrame) -> dict[str, object]:
    ordered = group.sort_values("workout_date_dt", kind="mergesort")
    latest = ordered.iloc[-1]
    previous = ordered.iloc[-2] if len(ordered) >= 2 else None
    days = pd.to_numeric(group["days_before_race"], errors="coerce")
    total = pd.to_numeric(group["total_time_sec"], errors="coerce")
    final1 = pd.to_numeric(group["final_1f_sec"], errors="coerce")
    finish_gain = pd.to_numeric(group["finish_gain_sec"], errors="coerce")
    course = group["course_bucket"].astype("string")
    hill = course.str.contains("hill", case=False, na=False)
    wood = course.str.contains("wood", case=False, na=False)
    current_week = days.between(0, 6, inclusive="both")
    prev_week_window = days.between(7, 13, inclusive="both")
    prev_day = days == 1
    final_11 = final1 <= 11.9
    final_12 = final1.between(12.0, 12.999, inclusive="both")
    hill_fast_53 = hill & (total <= 53.9)
    hill_fast_51 = hill & (total <= 51.9)
    wood_fast_67 = wood & (total <= 67.0)
    wood_fast_53 = wood & (total <= 53.9)
    strong_finish = group["strong_finish_flag"] == 1
    strong_work = hill_fast_53 | wood_fast_67 | final_11 | strong_finish
    gaps = ordered["workout_date_dt"].diff().dt.days.dropna()
    latest_course = str(latest.get("course_bucket", ""))
    prev_course = str(previous.get("course_bucket", "")) if previous is not None else ""
    latest_total = _safe_float(latest.get("total_time_sec"))
    best_total = float(total.min()) if total.notna().any() else np.nan
    latest_slower_than_best = latest_total - best_total if latest_total == latest_total and best_total == best_total else np.nan
    prev_week_strong = bool((prev_week_window & strong_work).any())
    current_week_light = bool((current_week & ~strong_work).any())
    current_week_wood = bool((current_week & wood).any())
    prev_week_hill = bool((prev_week_window & hill).any())

    return {
        "workout_count": float(len(group)),
        "workout_days_span": float((group["workout_date_dt"].max() - group["workout_date_dt"].min()).days)
        if group["workout_date_dt"].notna().any()
        else np.nan,
        "workout_best_total_time_sec": group["total_time_sec"].min(),
        "workout_best_final_1f_sec": group["final_1f_sec"].min(),
        "workout_avg_final_1f_sec": group["final_1f_sec"].mean(),
        "workout_fast_final_flag": float((group["fast_final_flag"] == 1).any()),
        "workout_strong_finish_flag": float((group["strong_finish_flag"] == 1).any()),
        "workout_partner_win_flag": float((group["partner_win_flag"] == 1).any()),
        "workout_a1_flag": float((group["lap_group"] == "A1").any()),
        "workout_b1_flag": float((group["lap_group"] == "B1").any()),
        "workout_a2_flag": float((group["lap_group"] == "A2").any()),
        "workout_b2_flag": float((group["lap_group"] == "B2").any()),
        "workout_a3_flag": float((group["lap_group"] == "A3").any()),
        "workout_b3_flag": float((group["lap_group"] == "B3").any()),
        "workout_best_total_vs_trainer_z": group["total_vs_trainer_z"].min(),
        "workout_best_final1_vs_trainer_z": group["final1_vs_trainer_z"].min(),
        "workout_hill_count": float(hill.sum()),
        "workout_wood_count": float(wood.sum()),
        "workout_current_week_count": float(current_week.sum()),
        "workout_prev_week_count": float(prev_week_window.sum()),
        "workout_prev_day_count": float(prev_day.sum()),
        "workout_prev_day_hill_flag": float((prev_day & hill).any()),
        "workout_prev_day_wood_flag": float((prev_day & wood).any()),
        "workout_prev_week_hill_count": float((prev_week_window & hill).sum()),
        "workout_prev_week_wood_count": float((prev_week_window & wood).sum()),
        "workout_prev_weekend_hill_under60_flag": float((prev_week_window & hill & (total < 60.0)).any()),
        "workout_current_week_wood_after_prev_week_hill_flag": float(current_week_wood and prev_week_hill),
        "workout_hill_wood_mix_flag": float(hill.any() and wood.any()),
        "workout_latest_course_switch_flag": float(bool(prev_course) and latest_course != prev_course),
        "workout_latest_from_hill_to_wood_flag": float(prev_course == "hill" and latest_course == "wood"),
        "workout_latest_from_wood_to_hill_flag": float(prev_course == "wood" and latest_course == "hill"),
        "workout_strong_work_count": float(strong_work.sum()),
        "workout_strong_work_rate": float(strong_work.mean()) if len(strong_work) else np.nan,
        "workout_current_week_strong_count": float((current_week & strong_work).sum()),
        "workout_prev_week_strong_count": float((prev_week_window & strong_work).sum()),
        "workout_prev_week_strong_then_current_light_flag": float(prev_week_strong and current_week_light),
        "workout_hill_53_count": float(hill_fast_53.sum()),
        "workout_hill_51_count": float(hill_fast_51.sum()),
        "workout_wood_67_count": float(wood_fast_67.sum()),
        "workout_wood_53_count": float(wood_fast_53.sum()),
        "workout_final_11_count": float(final_11.sum()),
        "workout_final_12_count": float(final_12.sum()),
        "workout_strong_finish_count": float(strong_finish.sum()),
        "workout_avg_finish_gain_sec": finish_gain.mean(),
        "workout_best_finish_gain_sec": finish_gain.max(),
        "workout_latest_slower_than_best_total_sec": latest_slower_than_best,
        "workout_avg_gap_days": float(gaps.mean()) if len(gaps) else np.nan,
        "workout_max_gap_days": float(gaps.max()) if len(gaps) else np.nan,
        "workout_load_density_score": float(
            0.35 * np.log1p(len(group))
            + 0.25 * np.log1p(strong_work.sum())
            + 0.20 * float((current_week & strong_work).any())
            + 0.20 * float((prev_week_window & strong_work).any())
        ),
    }


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
