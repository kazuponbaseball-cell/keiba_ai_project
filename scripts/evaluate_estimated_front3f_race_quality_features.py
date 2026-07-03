from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_corner_accel_rpci_features import (  # noqa: E402
    CORNER_ACCEL_FEATURES,
    add_corner_accel_features,
)
from scripts.estimate_runner_front3f_from_laps import (  # noqa: E402
    extract_su_runners_for_races,
    normalize_runners,
    parse_int_bytes,
)
from scripts.evaluate_expected_lap_rpci_features import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    NEW_CATEGORICAL_FEATURES,
    NEW_NUMERIC_FEATURES,
    RACE_COL,
    HORSE_COL,
    DATE_COL,
    add_expected_lap_features,
    coefficient_importance,
    fit_ranker,
    metric_summary,
    num,
    race_z,
)
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("outputs/analysis/estimated_front3f_race_quality_v1")
DEFAULT_TARGET_ROOT = Path("C:/Users/kazup/Data Lab/SE_DATA")

RA_RACE_ID_OFFSET = 11
RA_RACE_ID_LEN = 16
RA_DISTANCE_OFFSET = 697
RA_DISTANCE_LEN = 4
RA_LAP_OFFSET = 890
RA_LAP_COUNT = 25
RA_LAP_WIDTH = 3

FRONT3F_NUMERIC_FEATURES = [
    "course_front3f_prior_sec",
    "course_front3f_prior_std",
    "course_front3f_prior_count",
    "horse_est_ten_speed_z_mean_past5",
    "horse_est_ten_speed_z_best_past5",
    "horse_est_ten_speed_goodrun_past5",
    "horse_est_fast_start_rate_past5",
    "horse_est_gap600_mean_past5",
    "horse_est_front3f_confidence_mean_past5",
    "horse_course_adj_ten_speed_mean_past5",
    "horse_course_adj_ten_speed_best_past5",
    "horse_course_adj_ten_speed_goodrun_past5",
    "horse_course_adj_fast_start_rate_past5",
    "race_est_ten_pressure_score",
    "race_est_fast_start_count",
    "race_est_ten_speed_std",
    "race_est_ten_speed_gap_top2",
    "race_est_queue_clarity_score",
    "race_course_adj_ten_pressure_score",
    "race_course_adj_fast_start_count",
    "race_course_adj_ten_speed_gap_top2",
    "race_course_adj_queue_clarity_score",
    "ten_speed_pressure_fit_score",
    "ten_speed_solo_front_fit_score",
    "ten_speed_expected_fast_fit_score",
    "ten_speed_expected_slow_penalty_score",
    "course_adjusted_ten_pressure_fit_score",
    "course_adjusted_solo_front_fit_score",
    "course_adjusted_unstable_ten_score",
    "race_quality_front_load_score",
    "race_quality_unstable_ten_score",
    "horse_front_load_goodrun_past5",
    "horse_front_load_forward_goodrun_past5",
    "horse_front_load_top3_resilience_past5",
    "horse_front_load_forward_fade_risk_past5",
    "front_load_retrospective_fit_score",
    "front_load_forward_resilience_score",
    "front_load_fade_risk_current_score",
]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _decode_laps(raw: bytes) -> list[float]:
    laps: list[float] = []
    for idx in range(RA_LAP_COUNT):
        start = RA_LAP_OFFSET + idx * RA_LAP_WIDTH
        value = raw[start : start + RA_LAP_WIDTH].decode("ascii", errors="ignore")
        if value.isdigit() and int(value) > 0:
            laps.append(int(value) / 10.0)
    return laps


def extract_ra_laps_for_races(target_root: Path, race_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    race_id_bytes = {race_id.encode("ascii") for race_id in race_ids if len(race_id) == RA_RACE_ID_LEN}
    years = sorted({race_id[:4] for race_id in race_ids if race_id[:4].isdigit()})
    for year in years:
        year_dir = target_root / year
        if not year_dir.exists():
            continue
        for path in sorted(year_dir.glob("SR*.DAT")):
            try:
                records = path.read_bytes().splitlines()
            except OSError:
                continue
            for raw in records:
                if not raw.startswith(b"RA") or raw[RA_RACE_ID_OFFSET : RA_RACE_ID_OFFSET + RA_RACE_ID_LEN] not in race_id_bytes:
                    continue
                race_id = raw[RA_RACE_ID_OFFSET : RA_RACE_ID_OFFSET + RA_RACE_ID_LEN].decode("ascii", errors="ignore")
                distance = parse_int_bytes(raw[RA_DISTANCE_OFFSET : RA_DISTANCE_OFFSET + RA_DISTANCE_LEN])
                laps = _decode_laps(raw)
                if distance is None or len(laps) < 3:
                    continue
                rows.append(
                    {
                        "race_id": race_id,
                        "distance_m": int(distance),
                        "laps_200m": laps,
                        "race_lap_string": "-".join(f"{lap:.1f}" for lap in laps),
                        "source_file": str(path),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["race_id", "distance_m", "laps_200m", "race_lap_string", "source_file"])
    out = pd.DataFrame(rows)
    return out.drop_duplicates("race_id", keep="last")


def build_estimated_runner_front3f(
    race_ids: set[str],
    *,
    target_root: Path,
    cache_csv: Path,
    force_rebuild: bool = False,
    use_optimizer: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cache_csv.exists() and not force_rebuild:
        cached = pd.read_csv(cache_csv, encoding="utf-8-sig", low_memory=False)
        cached["race_id"] = cached["race_id"].astype(str)
        coverage = len(set(cached["race_id"]).intersection(race_ids)) / max(len(race_ids), 1)
        if coverage >= 0.995:
            return cached[cached["race_id"].isin(race_ids)].copy(), {
                "estimate_source": "cache",
                "cache_csv": str(cache_csv),
                "coverage": coverage,
                "rows": int(len(cached)),
                "races": int(cached["race_id"].nunique()),
            }

    laps = extract_ra_laps_for_races(target_root, race_ids)
    runners_source = extract_su_runners_for_races(target_root, set(laps["race_id"].astype(str)))
    out = fast_estimate_runner_front3f(laps, runners_source)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_csv, index=False, encoding="utf-8-sig")
    summary = {
        "estimate_source": "rebuilt",
        "cache_csv": str(cache_csv),
        "requested_races": int(len(race_ids)),
        "lap_races": int(len(laps)),
        "runner_source_rows": int(len(runners_source)),
        "rows": int(len(out)),
        "races": int(out["race_id"].nunique()) if not out.empty else 0,
        "mean_abs_reconstruction_error": float(out["reconstruction_error_sec"].abs().mean()) if not out.empty else None,
        "use_optimizer": bool(use_optimizer),
    }
    return out, summary


def _lap_sum(laps: list[float], start: int | None = None, end: int | None = None) -> float:
    part = laps[slice(start, end)]
    return float(sum(part))


def _lap_segment_lengths(distance_m: float | int, laps: list[float]) -> list[float]:
    if not laps:
        return []
    distance = float(distance_m)
    first = distance - 200.0 * (len(laps) - 1)
    if first <= 0 or first > 200:
        first = 200.0
    return [first] + [200.0] * (len(laps) - 1)


def _sum_lap_distance(laps: list[float], distance_m: float | int, meters: float, *, from_start: bool) -> float:
    lengths = _lap_segment_lengths(distance_m, laps)
    if not lengths:
        return float("nan")
    paired = list(zip(lengths, laps))
    if not from_start:
        paired = list(reversed(paired))
    remaining = float(meters)
    total = 0.0
    for length, lap in paired:
        if remaining <= 0:
            break
        take = min(length, remaining)
        total += float(lap) * (take / length)
        remaining -= take
    return float(total)


def fast_estimate_runner_front3f(laps: pd.DataFrame, runners_source: pd.DataFrame) -> pd.DataFrame:
    """Vectorized approximation for large historical caches.

    For 1200m and shorter races, the 600m point is the start of the runner's final 3F,
    so the estimate is directly pinned by race last3F, runner final3F, and finish gap.
    For longer races, 600m gap is approximated with a blend of the inferred final-3F-start
    gap and the earliest available corner rank. The confidence feature downstream keeps
    route estimates from being over-trusted.
    """
    if laps.empty or runners_source.empty:
        return pd.DataFrame()
    lap_frame = laps.copy()
    lap_frame["race_id"] = lap_frame["race_id"].astype(str)
    lap_frame["race_first3f_sec"] = lap_frame.apply(
        lambda r: _sum_lap_distance(r["laps_200m"], r["distance_m"], 600.0, from_start=True), axis=1
    )
    lap_frame["race_last3f_sec"] = lap_frame.apply(
        lambda r: _sum_lap_distance(r["laps_200m"], r["distance_m"], 600.0, from_start=False), axis=1
    )
    lap_frame["race_total_time_sec"] = lap_frame["laps_200m"].map(lambda x: _lap_sum(x))
    lap_frame["last3f_start_m"] = (pd.to_numeric(lap_frame["distance_m"], errors="coerce") - 600).clip(lower=600)
    lap_frame = lap_frame[["race_id", "distance_m", "race_first3f_sec", "race_last3f_sec", "race_total_time_sec", "last3f_start_m"]]

    runners = normalize_runners(runners_source)
    if "horse_id" in runners_source.columns:
        runners["horse_id"] = runners_source["horse_id"].astype(str).str.strip()
    else:
        runners["horse_id"] = pd.NA
    runners["race_id"] = runners["race_id"].astype(str)
    merged = runners.merge(lap_frame, on="race_id", how="inner")
    if merged.empty:
        return merged

    finish_gap = pd.to_numeric(merged["finish_gap_sec"], errors="coerce")
    final3f = pd.to_numeric(merged["final_3f_sec"], errors="coerce")
    finish_position = pd.to_numeric(merged.get("finish_position"), errors="coerce")
    valid = (
        finish_gap.notna()
        & final3f.notna()
        & finish_gap.between(0.0, 30.0)
        & (finish_position.isna() | finish_position.between(1, 99))
    )
    out = merged[valid].copy()
    if out.empty:
        return out

    finish_gap = finish_gap[valid].astype(float)
    final3f = final3f[valid].astype(float)
    out["finish_gap_sec_used"] = finish_gap
    out["gap_last3f_start_sec"] = (finish_gap + pd.to_numeric(out["race_last3f_sec"], errors="coerce") - final3f).clip(lower=0.0)
    last3_start = pd.to_numeric(out["last3f_start_m"], errors="coerce").clip(lower=600.0)
    linear_prior = out["gap_last3f_start_sec"] * (600.0 / last3_start)

    early_rank = pd.Series(np.nan, index=out.index, dtype=float)
    for col in ["corner1", "corner2", "corner3", "corner4"]:
        if col in out.columns:
            early_rank = early_rank.fillna(pd.to_numeric(out[col], errors="coerce"))
    rank_prior = ((early_rank - 1.0).clip(lower=0.0) * 0.15).fillna(linear_prior)

    distance = pd.to_numeric(out["distance_m"], errors="coerce")
    direct = distance.le(1200)
    route_blend = np.select(
        [distance.le(1400), distance.le(1800), distance.gt(1800)],
        [0.50, 0.38, 0.28],
        default=0.35,
    )
    blended = (1.0 - route_blend) * linear_prior + route_blend * rank_prior
    out["estimated_gap_600m_sec"] = np.where(direct, out["gap_last3f_start_sec"], blended)
    out["estimated_gap_600m_sec"] = pd.to_numeric(out["estimated_gap_600m_sec"], errors="coerce").clip(lower=0.0)
    out["estimated_front3f_sec"] = pd.to_numeric(out["race_first3f_sec"], errors="coerce") + out["estimated_gap_600m_sec"]
    middle_base = pd.to_numeric(out["race_total_time_sec"], errors="coerce") - pd.to_numeric(out["race_first3f_sec"], errors="coerce") - pd.to_numeric(out["race_last3f_sec"], errors="coerce")
    out["estimated_middle_sec"] = middle_base + (out["gap_last3f_start_sec"] - out["estimated_gap_600m_sec"])
    out["finish_time_sec_used"] = pd.to_numeric(out["race_total_time_sec"], errors="coerce") + out["finish_gap_sec_used"]
    out["reconstructed_finish_time_sec"] = out["estimated_front3f_sec"] + out["estimated_middle_sec"] + final3f.to_numpy()
    out["reconstruction_error_sec"] = out["reconstructed_finish_time_sec"] - out["finish_time_sec_used"]
    out["front3f_method"] = np.where(direct, "direct_1200_from_final3f", "fast_rank_prior")
    out["front3f_confidence"] = np.select(
        [distance.le(1200), distance.le(1400), distance.le(2200)],
        ["high", "medium_high", "medium"],
        default="low",
    )
    return out


def confidence_weight(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .map({"high": 1.0, "medium_high": 0.78, "medium": 0.58, "low": 0.35})
        .fillna(0.35)
        .astype(float)
    )


def _sigmoid(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(values, -30, 30))))


def _first_existing_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def _prior_mean_std_count(
    races: pd.DataFrame,
    keys: list[str],
    value: str,
    *,
    min_periods: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    grouped = races.groupby(keys, sort=False)[value]
    prior_mean = grouped.transform(lambda s: s.shift().expanding(min_periods=min_periods).mean())
    prior_std = grouped.transform(lambda s: s.shift().expanding(min_periods=max(3, min_periods)).std())
    prior_count = grouped.cumcount()
    return prior_mean, prior_std, prior_count


def normalize_class_tier(series: pd.Series) -> pd.Series:
    raw = series.astype("string").fillna("").str.strip()
    out = pd.Series("unknown", index=series.index, dtype="object")
    out = out.where(~raw.str.contains("新馬", na=False), "newcomer")
    out = out.where(~raw.str.contains("未勝利", na=False), "maiden")
    out = out.where(~raw.str.contains("1勝|１勝", regex=True, na=False), "1win")
    out = out.where(~raw.str.contains("2勝|２勝", regex=True, na=False), "2win")
    out = out.where(~raw.str.contains("3勝|３勝", regex=True, na=False), "3win")
    out = out.where(~raw.str.contains("OP|ＯＰ|オープン|ｵｰﾌﾟﾝ", regex=True, na=False), "open")
    out = out.where(~raw.str.contains("G1|G2|G3|Ｇ１|Ｇ２|Ｇ３", regex=True, na=False), "grade")
    return out


def build_course_front3f_priors(all_df: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    """Time-safe course baseline for front 3F.

    The current race's front 3F is never used for its own prior. Baselines are
    backed off from venue/surface/distance/going to broader course buckets.
    """
    if estimates.empty or "race_first3f_sec" not in estimates.columns:
        return pd.DataFrame({"_race_key": all_df["_race_key"].drop_duplicates()})

    place_col = _first_existing_col(all_df, ["場所", "蝣ｴ謇"])
    surface_col = _first_existing_col(all_df, ["芝・ダ", "闃昴・繝"])
    distance_col = _first_existing_col(all_df, ["距離", "霍晞屬"])
    going_col = _first_existing_col(all_df, ["馬場状態", "鬥ｬ蝣ｴ迥ｶ諷・"])
    class_col = _first_existing_col(all_df, ["クラス名", "繧ｯ繝ｩ繧ｹ蜷・"])
    race_context_cols = ["_race_key", RACE_COL]
    for col in [place_col, surface_col, distance_col, going_col, class_col]:
        if col and col not in race_context_cols:
            race_context_cols.append(col)

    race_front = (
        estimates[["race_id", "race_first3f_sec"]]
        .dropna(subset=["race_first3f_sec"])
        .drop_duplicates("race_id")
        .rename(columns={"race_id": "_race_key"})
    )
    races = all_df[race_context_cols].drop_duplicates("_race_key", keep="first").copy()
    races["_race_key"] = races["_race_key"].astype(str)
    races = races.merge(race_front, on="_race_key", how="left")
    races["_date_order"] = pd.to_numeric(races["_race_key"].str.slice(0, 8), errors="coerce").fillna(0)
    races = races.sort_values(["_date_order", "_race_key"], kind="mergesort").copy()
    races["race_first3f_sec"] = pd.to_numeric(races["race_first3f_sec"], errors="coerce")
    races["_class_tier"] = normalize_class_tier(races[class_col]) if class_col else "unknown"

    global_mean = races["race_first3f_sec"].shift().expanding(min_periods=20).mean()
    global_std = races["race_first3f_sec"].shift().expanding(min_periods=20).std()
    global_count = pd.Series(np.arange(len(races)), index=races.index, dtype=float)
    races["course_front3f_prior_sec"] = np.nan
    races["course_front3f_prior_std"] = np.nan
    races["course_front3f_prior_count"] = np.nan

    groups: list[tuple[list[str], str, int]] = []
    if place_col and surface_col and distance_col and going_col and class_col:
        groups.append(([place_col, surface_col, distance_col, going_col, "_class_tier"], "full_class", 3))
    if place_col and surface_col and distance_col and class_col:
        groups.append(([place_col, surface_col, distance_col, "_class_tier"], "course_class", 5))
    if surface_col and distance_col and class_col:
        groups.append(([surface_col, distance_col, "_class_tier"], "surface_distance_class", 8))
    if place_col and surface_col and distance_col and going_col:
        groups.append(([place_col, surface_col, distance_col, going_col], "full", 3))
    if place_col and surface_col and distance_col:
        groups.append(([place_col, surface_col, distance_col], "course", 5))
    if surface_col and distance_col:
        groups.append(([surface_col, distance_col], "surface_distance", 10))

    for keys, prefix, min_periods in groups:
        mean, std, count = _prior_mean_std_count(races, keys, "race_first3f_sec", min_periods=min_periods)
        races[f"{prefix}_course_front3f_prior_sec"] = mean
        races[f"{prefix}_course_front3f_prior_std"] = std
        races[f"{prefix}_course_front3f_prior_count"] = count

    for prefix in ["full_class", "course_class", "surface_distance_class", "full", "course", "surface_distance"]:
        mean_col = f"{prefix}_course_front3f_prior_sec"
        if mean_col not in races.columns:
            continue
        races["course_front3f_prior_sec"] = races["course_front3f_prior_sec"].fillna(races[mean_col])
        races["course_front3f_prior_std"] = races["course_front3f_prior_std"].fillna(races[f"{prefix}_course_front3f_prior_std"])
        races["course_front3f_prior_count"] = races["course_front3f_prior_count"].fillna(races[f"{prefix}_course_front3f_prior_count"])

    global_fallback = float(races["race_first3f_sec"].dropna().mean()) if races["race_first3f_sec"].notna().any() else 36.0
    std_fallback = float(races["race_first3f_sec"].dropna().std()) if races["race_first3f_sec"].notna().sum() >= 3 else 0.8
    races["course_front3f_prior_sec"] = races["course_front3f_prior_sec"].fillna(global_mean).fillna(global_fallback)
    races["course_front3f_prior_std"] = races["course_front3f_prior_std"].fillna(global_std).fillna(std_fallback).clip(lower=0.35, upper=2.5)
    races["course_front3f_prior_count"] = races["course_front3f_prior_count"].fillna(global_count).fillna(0.0)
    return races[["_race_key", "_class_tier", "course_front3f_prior_sec", "course_front3f_prior_std", "course_front3f_prior_count"]]


def add_estimated_front3f_features(train: pd.DataFrame, test: pd.DataFrame, estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["_split"] = "train"
    test["_split"] = "test"
    train["_row_id"] = np.arange(len(train))
    test["_row_id"] = np.arange(len(test))
    all_df = pd.concat([train, test], ignore_index=True, sort=False)
    all_df["_race_key"] = all_df[RACE_COL].astype(str)
    all_df["_horse_key"] = all_df[HORSE_COL].astype(str).str.strip()

    if estimates.empty:
        for col in FRONT3F_NUMERIC_FEATURES:
            all_df[col] = 0.0
        train_out = all_df[all_df["_split"].eq("train")].drop(columns=["_split", "_row_id", "_race_key", "_horse_key"], errors="ignore")
        test_out = all_df[all_df["_split"].eq("test")].drop(columns=["_split", "_row_id", "_race_key", "_horse_key"], errors="ignore")
        return train_out, test_out, pd.DataFrame()

    est = estimates.copy()
    est["race_id"] = est["race_id"].astype(str)
    est["horse_id"] = est["horse_id"].astype(str).str.strip()
    est["estimated_front3f_sec"] = pd.to_numeric(est["estimated_front3f_sec"], errors="coerce")
    est["estimated_gap_600m_sec"] = pd.to_numeric(est["estimated_gap_600m_sec"], errors="coerce")
    course_priors = build_course_front3f_priors(all_df, est)
    all_df = all_df.merge(course_priors, on="_race_key", how="left")
    est = est.merge(
        course_priors.rename(columns={"_race_key": "race_id"}),
        on="race_id",
        how="left",
    )
    est["front3f_confidence_weight"] = confidence_weight(est["front3f_confidence"])
    race_mean = est.groupby("race_id")["estimated_front3f_sec"].transform("mean")
    race_std = est.groupby("race_id")["estimated_front3f_sec"].transform("std").replace(0, np.nan)
    est["estimated_ten_speed_z"] = ((race_mean - est["estimated_front3f_sec"]) / race_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    est["estimated_ten_speed_z_weighted"] = est["estimated_ten_speed_z"] * est["front3f_confidence_weight"]
    course_std = pd.to_numeric(est["course_front3f_prior_std"], errors="coerce").fillna(0.8).clip(lower=0.35, upper=2.5)
    course_prior = pd.to_numeric(est["course_front3f_prior_sec"], errors="coerce").fillna(pd.to_numeric(est["race_first3f_sec"], errors="coerce"))
    est["estimated_course_adj_ten_speed_z"] = ((course_prior - est["estimated_front3f_sec"]) / course_std).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0).clip(-4.0, 4.0)
    est["estimated_course_adj_ten_speed_z_weighted"] = est["estimated_course_adj_ten_speed_z"] * est["front3f_confidence_weight"]
    est["estimated_fast_start_flag"] = (
        (est["estimated_gap_600m_sec"].le(0.7)) | (est["estimated_ten_speed_z"].ge(0.65))
    ).astype(float)
    est["estimated_course_adj_fast_start_flag"] = (
        (est["estimated_course_adj_ten_speed_z"].ge(0.55))
        | (est["estimated_front3f_sec"].le(course_prior - 0.35))
    ).astype(float)
    est["_ten_pos"] = est["estimated_ten_speed_z"].clip(lower=0.0) * est["front3f_confidence_weight"]
    est["_field_size"] = est.groupby("race_id")["horse_id"].transform("size").replace(0, np.nan)
    est["estimated_race_front_load_est"] = (est.groupby("race_id")["_ten_pos"].transform("sum") / np.sqrt(est["_field_size"])).fillna(0.0)
    est["estimated_race_fast_start_count_est"] = est.groupby("race_id")["estimated_fast_start_flag"].transform("sum").fillna(0.0)
    est["estimated_race_ten_speed_std_est"] = est.groupby("race_id")["estimated_ten_speed_z"].transform("std").fillna(0.0)

    merge_cols = [
        "race_id",
        "horse_id",
        "estimated_front3f_sec",
        "estimated_gap_600m_sec",
        "front3f_confidence_weight",
        "estimated_ten_speed_z_weighted",
        "estimated_course_adj_ten_speed_z_weighted",
        "estimated_fast_start_flag",
        "estimated_course_adj_fast_start_flag",
        "estimated_race_front_load_est",
        "estimated_race_fast_start_count_est",
        "estimated_race_ten_speed_std_est",
    ]
    all_df = all_df.merge(
        est[merge_cols].rename(columns={"race_id": "_race_key", "horse_id": "_horse_key"}),
        on=["_race_key", "_horse_key"],
        how="left",
    )

    all_df["_date_order"] = pd.to_numeric(all_df["_race_key"].str.slice(0, 8), errors="coerce").fillna(0)
    ordered = all_df.sort_values(["_horse_key", "_date_order", "_race_key"], kind="mergesort")
    hist_specs = [
        ("estimated_ten_speed_z_weighted", "horse_est_ten_speed_z_mean_past5", "mean"),
        ("estimated_ten_speed_z_weighted", "horse_est_ten_speed_z_best_past5", "max"),
        ("estimated_fast_start_flag", "horse_est_fast_start_rate_past5", "mean"),
        ("estimated_gap_600m_sec", "horse_est_gap600_mean_past5", "mean"),
        ("front3f_confidence_weight", "horse_est_front3f_confidence_mean_past5", "mean"),
        ("estimated_course_adj_ten_speed_z_weighted", "horse_course_adj_ten_speed_mean_past5", "mean"),
        ("estimated_course_adj_ten_speed_z_weighted", "horse_course_adj_ten_speed_best_past5", "max"),
        ("estimated_course_adj_fast_start_flag", "horse_course_adj_fast_start_rate_past5", "mean"),
    ]
    target_score = pd.to_numeric(ordered.get("target_score"), errors="coerce").fillna(0.0).clip(lower=0.0)
    target_top3 = pd.to_numeric(ordered.get("target_top3"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    actual_load = pd.to_numeric(ordered.get("estimated_race_front_load_est"), errors="coerce").fillna(0.0).clip(lower=0.0)
    actual_ten_pos = pd.to_numeric(ordered.get("estimated_ten_speed_z_weighted"), errors="coerce").fillna(0.0).clip(lower=0.0)
    goodrun = pd.to_numeric(ordered["estimated_ten_speed_z_weighted"], errors="coerce").fillna(0.0) * target_score
    all_df.loc[ordered.index, "horse_est_ten_speed_goodrun_past5"] = (
        goodrun.groupby(ordered["_horse_key"], sort=False).transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    )
    course_adj_goodrun = pd.to_numeric(ordered["estimated_course_adj_ten_speed_z_weighted"], errors="coerce").fillna(0.0) * target_score
    all_df.loc[ordered.index, "horse_course_adj_ten_speed_goodrun_past5"] = (
        course_adj_goodrun.groupby(ordered["_horse_key"], sort=False).transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    )
    retrospective_sources = {
        "horse_front_load_goodrun_past5": actual_load * target_score,
        "horse_front_load_forward_goodrun_past5": actual_load * actual_ten_pos * target_score,
        "horse_front_load_top3_resilience_past5": actual_load * target_top3,
        "horse_front_load_forward_fade_risk_past5": actual_load * actual_ten_pos * (1.0 - target_score),
    }
    for dest, values in retrospective_sources.items():
        all_df.loc[ordered.index, dest] = values.groupby(ordered["_horse_key"], sort=False).transform(
            lambda s: s.shift().rolling(5, min_periods=1).mean()
        )
    for source, dest, agg in hist_specs:
        values = pd.to_numeric(ordered[source], errors="coerce")
        grouped = values.groupby(ordered["_horse_key"], sort=False)
        if agg == "max":
            rolled = grouped.transform(lambda s: s.shift().rolling(5, min_periods=1).max())
        else:
            rolled = grouped.transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
        all_df.loc[ordered.index, dest] = pd.to_numeric(rolled, errors="coerce")

    for col in [
        "horse_est_ten_speed_z_mean_past5",
        "horse_est_ten_speed_z_best_past5",
        "horse_est_ten_speed_goodrun_past5",
        "horse_est_fast_start_rate_past5",
        "horse_est_front3f_confidence_mean_past5",
        "horse_course_adj_ten_speed_mean_past5",
        "horse_course_adj_ten_speed_best_past5",
        "horse_course_adj_ten_speed_goodrun_past5",
        "horse_course_adj_fast_start_rate_past5",
        "horse_front_load_goodrun_past5",
        "horse_front_load_forward_goodrun_past5",
        "horse_front_load_top3_resilience_past5",
        "horse_front_load_forward_fade_risk_past5",
    ]:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce").fillna(0.0)
    all_df["horse_est_gap600_mean_past5"] = pd.to_numeric(all_df["horse_est_gap600_mean_past5"], errors="coerce").fillna(3.0)
    for col in ["course_front3f_prior_sec", "course_front3f_prior_std", "course_front3f_prior_count"]:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce").fillna(0.0)

    race = all_df[RACE_COL]
    ten = all_df["horse_est_ten_speed_z_mean_past5"].fillna(0.0)
    ten_pos = ten.clip(lower=0.0)
    field_size = ten.groupby(race).transform("size").replace(0, np.nan)
    all_df["race_est_ten_pressure_score"] = (ten_pos.groupby(race).transform("sum") / np.sqrt(field_size)).fillna(0.0)
    all_df["race_est_fast_start_count"] = (
        ((ten.ge(0.45)) | (all_df["horse_est_fast_start_rate_past5"].ge(0.34))).astype(float).groupby(race).transform("sum")
    )
    all_df["race_est_ten_speed_std"] = ten.groupby(race).transform("std").fillna(0.0)

    def top_gap(s: pd.Series) -> float:
        vals = np.sort(pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy())[::-1]
        if len(vals) < 2:
            return 0.0
        return float(vals[0] - vals[1])

    all_df["race_est_ten_speed_gap_top2"] = ten.groupby(race).transform(top_gap).fillna(0.0)
    all_df["race_est_queue_clarity_score"] = _sigmoid(
        1.4 * all_df["race_est_ten_speed_gap_top2"] - 0.35 * all_df["race_est_fast_start_count"] + 0.4
    ).to_numpy()
    course_adj_ten = all_df["horse_course_adj_ten_speed_mean_past5"].fillna(0.0)
    course_adj_pos = course_adj_ten.clip(lower=0.0)
    all_df["race_course_adj_ten_pressure_score"] = (course_adj_pos.groupby(race).transform("sum") / np.sqrt(field_size)).fillna(0.0)
    all_df["race_course_adj_fast_start_count"] = (
        ((course_adj_ten.ge(0.45)) | (all_df["horse_course_adj_fast_start_rate_past5"].ge(0.34))).astype(float).groupby(race).transform("sum")
    )
    all_df["race_course_adj_ten_speed_gap_top2"] = course_adj_ten.groupby(race).transform(top_gap).fillna(0.0)
    all_df["race_course_adj_queue_clarity_score"] = _sigmoid(
        1.5 * all_df["race_course_adj_ten_speed_gap_top2"] - 0.32 * all_df["race_course_adj_fast_start_count"] + 0.35
    ).to_numpy()

    ten_z_current = race_z(ten, race)
    expected_fast = pd.to_numeric(all_df.get("expected_lap_fast_weight"), errors="coerce").fillna(0.0)
    expected_slow = pd.to_numeric(all_df.get("expected_lap_slow_weight"), errors="coerce").fillna(0.0)
    collapse = pd.to_numeric(all_df.get("race_pace_collapse_risk"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    existing_pressure = pd.to_numeric(all_df.get("race_early_pressure_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)

    all_df["ten_speed_pressure_fit_score"] = (
        ten_z_current - 0.22 * all_df["race_est_ten_pressure_score"] - 0.18 * collapse
    ).fillna(0.0)
    all_df["ten_speed_solo_front_fit_score"] = (ten_z_current * all_df["race_est_queue_clarity_score"]).fillna(0.0)
    all_df["ten_speed_expected_fast_fit_score"] = (ten_z_current * expected_fast).fillna(0.0)
    all_df["ten_speed_expected_slow_penalty_score"] = (ten_z_current.clip(lower=0.0) * expected_slow).fillna(0.0)
    course_adj_current = race_z(course_adj_ten, race)
    all_df["course_adjusted_ten_pressure_fit_score"] = (
        course_adj_current - 0.22 * all_df["race_course_adj_ten_pressure_score"] - 0.16 * collapse
    ).fillna(0.0)
    all_df["course_adjusted_solo_front_fit_score"] = (
        course_adj_current * all_df["race_course_adj_queue_clarity_score"]
    ).fillna(0.0)
    all_df["course_adjusted_unstable_ten_score"] = (
        all_df["race_course_adj_ten_pressure_score"]
        * (0.58 * collapse + 0.42 * (1.0 - all_df["race_course_adj_queue_clarity_score"]))
    ).fillna(0.0)
    all_df["race_quality_front_load_score"] = (
        0.42 * all_df["race_est_ten_pressure_score"] + 0.28 * all_df["race_course_adj_ten_pressure_score"] + 0.30 * existing_pressure
    ).fillna(0.0)
    all_df["race_quality_unstable_ten_score"] = (
        all_df["race_quality_front_load_score"] * (0.65 * collapse + 0.35 * (1.0 - all_df["race_est_queue_clarity_score"]))
    ).fillna(0.0)
    all_df["front_load_retrospective_fit_score"] = (
        all_df["race_quality_front_load_score"] * race_z(all_df["horse_front_load_goodrun_past5"], race)
    ).fillna(0.0)
    all_df["front_load_forward_resilience_score"] = (
        all_df["race_quality_front_load_score"] * race_z(all_df["horse_front_load_forward_goodrun_past5"], race)
    ).fillna(0.0)
    all_df["front_load_fade_risk_current_score"] = (
        all_df["race_quality_front_load_score"] * race_z(all_df["horse_front_load_forward_fade_risk_past5"], race)
    ).fillna(0.0)

    diag = (
        all_df[
            [
                RACE_COL,
                "course_front3f_prior_sec",
                "course_front3f_prior_std",
                "course_front3f_prior_count",
                "race_est_ten_pressure_score",
                "race_est_fast_start_count",
                "race_est_ten_speed_gap_top2",
                "race_est_queue_clarity_score",
                "race_course_adj_ten_pressure_score",
                "race_course_adj_fast_start_count",
                "race_course_adj_ten_speed_gap_top2",
                "race_course_adj_queue_clarity_score",
            ]
        ]
        .drop_duplicates(RACE_COL)
        .copy()
    )

    all_df = all_df.drop(
        columns=[
            "_split",
            "_row_id",
            "_race_key",
            "_horse_key",
            "_date_order",
            "estimated_front3f_sec",
            "estimated_gap_600m_sec",
            "front3f_confidence_weight",
            "estimated_ten_speed_z_weighted",
            "estimated_course_adj_ten_speed_z_weighted",
            "estimated_fast_start_flag",
            "estimated_course_adj_fast_start_flag",
            "estimated_race_front_load_est",
            "estimated_race_fast_start_count_est",
            "estimated_race_ten_speed_std_est",
        ],
        errors="ignore",
    )
    train_out = all_df.iloc[: len(train)].copy()
    test_out = all_df.iloc[len(train) :].copy()
    return train_out, test_out, diag


def bet_metrics(part: pd.DataFrame) -> dict[str, Any]:
    if len(part) == 0:
        return {}
    win_pay = num(part, "単勝配当", 0.0).where(part["target_win"].eq(1), 0.0)
    place_pay = num(part, "複勝配当", 0.0).where(part["target_top3"].eq(1), 0.0)
    return {
        "bets": int(len(part)),
        "races": int(part[RACE_COL].nunique()),
        "win_rate": float(part["target_win"].mean()),
        "top3_rate": float(part["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (len(part) * 100.0)),
        "place_roi": float(place_pay.sum() / (len(part) * 100.0)),
        "avg_popularity": float(num(part, "人気").mean()),
        "avg_odds": float(num(part, "単勝オッズ").mean()),
    }


def segment_report(train: pd.DataFrame, test: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = test.copy()
    out["score"] = scores
    out["rank"] = out.groupby(RACE_COL)["score"].rank(ascending=False, method="first").astype(int)
    q = {
        "ten_hi": float(num(train, "horse_est_ten_speed_z_mean_past5").quantile(0.75)),
        "pressure_hi": float(num(train, "race_quality_front_load_score").quantile(0.75)),
        "clarity_hi": float(num(train, "race_est_queue_clarity_score").quantile(0.75)),
        "course_adj_ten_hi": float(num(train, "horse_course_adj_ten_speed_mean_past5").quantile(0.75)),
        "course_adj_clarity_hi": float(num(train, "race_course_adj_queue_clarity_score").quantile(0.75)),
        "course_adj_unstable_hi": float(num(train, "course_adjusted_unstable_ten_score").quantile(0.75)),
        "course_adj_solo_hi": float(num(train, "course_adjusted_solo_front_fit_score").quantile(0.75)),
        "unstable_hi": float(num(train, "race_quality_unstable_ten_score").quantile(0.75)),
        "solo_hi": float(num(train, "ten_speed_solo_front_fit_score").quantile(0.75)),
        "retro_hi": float(num(train, "front_load_retrospective_fit_score").quantile(0.75)),
        "resilience_hi": float(num(train, "front_load_forward_resilience_score").quantile(0.75)),
        "fade_hi": float(num(train, "front_load_fade_risk_current_score").quantile(0.75)),
    }
    checks = [
        ("top1_all", out["rank"].eq(1)),
        ("top1_ten_speed_hi", out["rank"].eq(1) & num(out, "horse_est_ten_speed_z_mean_past5").ge(q["ten_hi"])),
        ("top1_course_adj_ten_hi", out["rank"].eq(1) & num(out, "horse_course_adj_ten_speed_mean_past5").ge(q["course_adj_ten_hi"])),
        ("top1_front_load_hi", out["rank"].eq(1) & num(out, "race_quality_front_load_score").ge(q["pressure_hi"])),
        ("top1_queue_clarity_hi", out["rank"].eq(1) & num(out, "race_est_queue_clarity_score").ge(q["clarity_hi"])),
        ("top1_course_adj_queue_clarity_hi", out["rank"].eq(1) & num(out, "race_course_adj_queue_clarity_score").ge(q["course_adj_clarity_hi"])),
        ("top1_unstable_ten_hi", out["rank"].eq(1) & num(out, "race_quality_unstable_ten_score").ge(q["unstable_hi"])),
        ("top1_course_adj_unstable_ten_hi", out["rank"].eq(1) & num(out, "course_adjusted_unstable_ten_score").ge(q["course_adj_unstable_hi"])),
        ("top1_solo_front_fit_hi", out["rank"].eq(1) & num(out, "ten_speed_solo_front_fit_score").ge(q["solo_hi"])),
        ("top1_course_adj_solo_front_fit_hi", out["rank"].eq(1) & num(out, "course_adjusted_solo_front_fit_score").ge(q["course_adj_solo_hi"])),
        ("top1_retro_front_load_fit_hi", out["rank"].eq(1) & num(out, "front_load_retrospective_fit_score").ge(q["retro_hi"])),
        ("top1_forward_resilience_hi", out["rank"].eq(1) & num(out, "front_load_forward_resilience_score").ge(q["resilience_hi"])),
        ("top1_forward_fade_risk_hi", out["rank"].eq(1) & num(out, "front_load_fade_risk_current_score").ge(q["fade_hi"])),
        (
            "top3_pop5plus_ten_speed_hi",
            out["rank"].le(3) & num(out, "人気").ge(5) & num(out, "horse_est_ten_speed_z_mean_past5").ge(q["ten_hi"]),
        ),
        (
            "top3_pop5plus_course_adj_ten_hi",
            out["rank"].le(3) & num(out, "人気").ge(5) & num(out, "horse_course_adj_ten_speed_mean_past5").ge(q["course_adj_ten_hi"]),
        ),
        (
            "top3_pop5plus_solo_front_fit_hi",
            out["rank"].le(3) & num(out, "人気").ge(5) & num(out, "ten_speed_solo_front_fit_score").ge(q["solo_hi"]),
        ),
        (
            "top3_pop5plus_course_adj_solo_front_fit_hi",
            out["rank"].le(3) & num(out, "人気").ge(5) & num(out, "course_adjusted_solo_front_fit_score").ge(q["course_adj_solo_hi"]),
        ),
        (
            "top3_pop5plus_retro_front_load_fit_hi",
            out["rank"].le(3) & num(out, "人気").ge(5) & num(out, "front_load_retrospective_fit_score").ge(q["retro_hi"]),
        ),
    ]
    return pd.DataFrame([{"segment": name, **bet_metrics(out[mask])} for name, mask in checks])


def render_review(summary: pd.DataFrame, segments: pd.DataFrame, estimate_summary: dict[str, Any]) -> str:
    def pct(x: object) -> str:
        try:
            if pd.isna(x):
                return ""
            return f"{float(x) * 100:.1f}%"
        except Exception:
            return str(x)

    cols = [
        "variant",
        "races",
        "top1_win_rate",
        "top1_top3_rate",
        "top1_win_roi",
        "top1_place_roi",
        "top3_contains_winner_rate",
        "winner_mean_ai_rank",
    ]
    lines = [
        "# 推定テン3F × レース質 特徴量検証",
        "",
        "## Estimate Coverage",
        "```json",
        json.dumps(estimate_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Model Summary",
        "|variant|races|AI1位勝率|AI1位複勝率|AI1位単勝ROI|AI1位複勝ROI|上位3頭内勝ち馬率|勝ち馬平均AI順位|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary[cols].iterrows():
        lines.append(
            f"|{row['variant']}|{int(row['races'])}|{pct(row['top1_win_rate'])}|{pct(row['top1_top3_rate'])}|"
            f"{pct(row['top1_win_roi'])}|{pct(row['top1_place_roi'])}|{pct(row['top3_contains_winner_rate'])}|"
            f"{float(row['winner_mean_ai_rank']):.2f}|"
        )
    segment_cols = ["segment", "bets", "races", "win_rate", "top3_rate", "win_roi", "place_roi", "avg_popularity", "avg_odds"]
    lines.extend(
        [
            "",
            "## Segments",
            "|segment|bets|races|勝率|複勝率|単勝ROI|複勝ROI|平均人気|平均オッズ|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in segments[[c for c in segment_cols if c in segments.columns]].iterrows():
        lines.append(
            f"|{row.get('segment', '')}|{int(row.get('bets', 0) or 0)}|{int(row.get('races', 0) or 0)}|"
            f"{pct(row.get('win_rate'))}|{pct(row.get('top3_rate'))}|{pct(row.get('win_roi'))}|{pct(row.get('place_roi'))}|"
            f"{float(row.get('avg_popularity')):.2f}|{float(row.get('avg_odds')):.2f}|"
        )
    lines.append("")
    lines.extend(
        [
            "## Notes",
            "- 当該レースの推定テン3Fは使わず、馬ごとに過去走へshiftした履歴特徴量だけを使う。",
            "- 1200m以下は高信頼、1400m以上は通過順位ベースの推定なので信頼度重みを掛ける。",
            "- ここでのROIは単勝・複勝のモデル順位検証であり、最強BUYの馬連/ワイドROIへ直結させる前にペア側の検証が必要。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate estimated runner front-3F features as race-quality and pace-shape signals.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force-rebuild-estimates", action="store_true")
    parser.add_argument("--use-optimizer", action="store_true", help="Use slower scipy optimization for 1400m+ estimates.")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(project_path(args.train_csv), low_memory=False)
    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    base_model: SimpleRaceRanker = pickle.load(project_path(args.base_model).open("rb"))

    race_ids = set(train[RACE_COL].astype(str)).union(set(test[RACE_COL].astype(str)))
    estimates, estimate_summary = build_estimated_runner_front3f(
        race_ids,
        target_root=args.target_root,
        cache_csv=out_dir / "estimated_runner_front3f_cache.csv",
        force_rebuild=args.force_rebuild_estimates,
        use_optimizer=args.use_optimizer,
    )
    estimates.to_csv(out_dir / "estimated_runner_front3f_used.csv", index=False, encoding="utf-8-sig")

    train_x, test_x, race_diag = add_expected_lap_features(train, test)
    train_c, test_c = add_corner_accel_features(train_x, test_x)
    train_f, test_f, front_diag = add_estimated_front3f_features(train_c, test_c, estimates)
    light_cols = [
        RACE_COL,
        HORSE_COL,
        "馬番",
        "馬名",
        "人気",
        "単勝オッズ",
        *[col for col in FRONT3F_NUMERIC_FEATURES if col in train_f.columns],
    ]
    train_f[[col for col in light_cols if col in train_f.columns]].to_csv(
        out_dir / "train_front3f_feature_light.csv", index=False, encoding="utf-8-sig"
    )
    test_f[[col for col in light_cols if col in test_f.columns]].to_csv(
        out_dir / "test_front3f_feature_light.csv", index=False, encoding="utf-8-sig"
    )

    base_numeric = list(base_model.numeric_features) + [col for col in NEW_NUMERIC_FEATURES if col not in base_model.numeric_features]
    base_categorical = list(base_model.categorical_features) + [
        col for col in NEW_CATEGORICAL_FEATURES if col not in base_model.categorical_features
    ]
    corner_numeric = base_numeric + [col for col in CORNER_ACCEL_FEATURES if col not in base_numeric]
    front_numeric = corner_numeric + [col for col in FRONT3F_NUMERIC_FEATURES if col not in corner_numeric]

    alpha = float(base_model.ridge_alpha)
    top_k = int(base_model.categorical_top_k)
    expected_model = fit_ranker(train_c, base_numeric, base_categorical, alpha, top_k)
    corner_model = fit_ranker(train_c, corner_numeric, base_categorical, alpha, top_k)
    front_model = fit_ranker(train_f, front_numeric, base_categorical, alpha, top_k)

    expected_scores = expected_model.predict(test_c)
    corner_scores = corner_model.predict(test_c)
    front_scores = front_model.predict(test_f)

    summary = pd.DataFrame(
        [
            {"variant": "expected_lap_base", **metric_summary(test_c, expected_scores)},
            {"variant": "expected_lap_plus_corner_accel", **metric_summary(test_c, corner_scores)},
            {"variant": "estimated_front3f_race_quality", **metric_summary(test_f, front_scores)},
        ]
    )
    baseline = summary[summary["variant"].eq("expected_lap_plus_corner_accel")].iloc[0]
    for col in [
        "top1_win_rate",
        "top1_top3_rate",
        "top1_win_roi",
        "top1_place_roi",
        "top3_contains_winner_rate",
        "top3_win_roi",
        "top3_place_roi",
        "winner_mean_ai_rank",
    ]:
        summary[f"delta_vs_corner_{col}"] = summary[col] - baseline[col]

    segments = segment_report(train_f, test_f, front_scores)
    imps = coefficient_importance(front_model, "estimated_front3f_race_quality")

    summary.to_csv(out_dir / "estimated_front3f_race_quality_summary.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(out_dir / "estimated_front3f_race_quality_segments.csv", index=False, encoding="utf-8-sig")
    imps.to_csv(out_dir / "estimated_front3f_race_quality_importance.csv", index=False, encoding="utf-8-sig")
    race_diag.to_csv(out_dir / "race_expected_lap_diagnostics.csv", index=False, encoding="utf-8-sig")
    front_diag.to_csv(out_dir / "front3f_race_quality_diagnostics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "review.md").write_text(render_review(summary, segments, estimate_summary), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "estimate_summary": estimate_summary,
                "new_features": FRONT3F_NUMERIC_FEATURES,
                "output_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    show = summary.copy()
    pct_cols = [c for c in show.columns if c.endswith("_rate") or c.endswith("_roi")]
    show[pct_cols] = show[pct_cols] * 100.0
    print(show[["variant", "races", "top1_win_rate", "top1_top3_rate", "top1_win_roi", "top1_place_roi", "top3_contains_winner_rate", "winner_mean_ai_rank"]].to_string(index=False))
    print("\nFront3F segments:")
    show_seg = segments.copy()
    pct_cols = [c for c in show_seg.columns if c.endswith("_rate") or c.endswith("_roi")]
    show_seg[pct_cols] = show_seg[pct_cols] * 100.0
    print(show_seg.to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
