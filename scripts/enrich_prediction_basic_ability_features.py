from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_DIR = ROOT / "outputs/analysis/content_bridge_member_features_v1"
DEFAULT_RECENT_RESULT_GLOBS = [
    "outputs/analysis/race_day_review_*/parsed_results_all_horses.csv",
    "outputs/analysis/*bias_review/horse_results.csv",
]
DEFAULT_ENTRY_GLOBS = [
    "data/datasets/inference/weekly/entry_snapshot_netkeiba_*_target_de_overlay_enriched_workout_knowledge.csv",
    "data/datasets/inference/weekly/entry_snapshot_netkeiba_*_target_de_overlay_enriched_workout.csv",
    "data/datasets/inference/weekly/entry_snapshot_netkeiba_*_target_de_overlay_enriched.csv",
    "data/datasets/inference/weekly/entry_snapshot_netkeiba_*_target_de_overlay.csv",
    "data/datasets/inference/weekly/entry_snapshot_netkeiba_*_enriched_odds_workout_knowledge.csv",
    "data/datasets/inference/weekly/entry_snapshot_netkeiba_*_enriched_odds_workout.csv",
    "data/datasets/inference/weekly/entry_snapshot_netkeiba_*_enriched_odds.csv",
    "data/datasets/inference/weekly/entry_snapshot_netkeiba_*.csv",
]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def available_usecols(path: Path, wanted: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    return [col for col in wanted if col in header.columns]


def normalize_date(value: object) -> str:
    if pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 6:
        return "20" + digits
    if len(digits) >= 8:
        return digits[:8]
    return ""


def to_num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index if index is not None else None)
    return pd.to_numeric(series, errors="coerce")


def row_race_id(row: pd.Series) -> str:
    for col in ["レースID(新/馬番無)", "race_id"]:
        if col in row and not pd.isna(row[col]) and str(row[col]).strip():
            digits = re.sub(r"\D", "", str(row[col]))
            return digits
    return ""


def add_basic_history_features(history: pd.DataFrame) -> pd.DataFrame:
    out = history.copy()
    out["target_score_num"] = to_num(out.get("target_score"), out.index)
    out = out[out["horse_id"].ne("") & out["target_score_num"].notna()].copy()
    out = out.sort_values(["horse_id", "date_key", "race_no", "horse_no"]).reset_index(drop=True)
    grouped = out.groupby("horse_id", sort=False)
    lag_cols: list[str] = []
    for lag in range(1, 6):
        col = f"prev{lag}_target_score_from_history"
        out[col] = grouped["target_score_num"].shift(lag)
        lag_cols.append(col)
    past3 = out[lag_cols[:3]]
    past5 = out[lag_cols]
    weights = np.array([0.50, 0.30, 0.20])
    present = past3.notna().astype(float)
    weighted_sum = past3.fillna(0.0).to_numpy() @ weights
    weight_sum = present.to_numpy() @ weights
    out["recent_weighted_score_3"] = np.divide(
        weighted_sum,
        weight_sum,
        out=np.full_like(weighted_sum, np.nan, dtype=float),
        where=weight_sum > 0,
    )
    out["recent_score_mean_3"] = past3.mean(axis=1)
    out["recent_score_std_3"] = past3.std(axis=1, ddof=0)
    out["recent_score_slope_3"] = out["prev1_target_score_from_history"] - out["prev3_target_score_from_history"]
    out["recent_score_jump_vs_mean"] = out["prev1_target_score_from_history"] - out["recent_score_mean_3"]
    out["ability_stability_score_3"] = out["recent_score_mean_3"] - out["recent_score_std_3"].fillna(0.0)
    out["ability_ceiling_score_5"] = past5.max(axis=1)
    out["ability_floor_score_5"] = past5.min(axis=1)
    out["recent_score_count_3"] = past3.notna().sum(axis=1)
    out["recent_score_count_5"] = past5.notna().sum(axis=1)
    return out


def find_entry_file_for_date(date_key: str, entry_globs: list[str]) -> Path | None:
    candidates = expand_globs(entry_globs)
    matching = [p for p in candidates if date_key in p.name]
    if not matching:
        return None

    def priority(path: Path) -> tuple[int, float]:
        name = path.name
        score = 0
        if "target_de_overlay_enriched_workout_knowledge" in name:
            score += 100
        elif "target_de_overlay_enriched_workout" in name:
            score += 90
        elif "target_de_overlay_enriched" in name:
            score += 80
        elif "target_de_overlay" in name:
            score += 70
        elif "enriched_odds_workout_knowledge" in name:
            score += 60
        elif "enriched_odds" in name:
            score += 50
        return score, path.stat().st_mtime

    return sorted(matching, key=priority, reverse=True)[0]


def load_entry_keys(path: Path) -> pd.DataFrame:
    wanted = ["レースID(新/馬番無)", "race_id", "血統登録番号", "horse_id", "日付", "日付S", "場所", "Ｒ", "馬番"]
    usecols = available_usecols(path, wanted)
    if not usecols:
        return pd.DataFrame()
    entry = read_csv(path, dtype=str, usecols=usecols)
    entry["race_id"] = entry.apply(row_race_id, axis=1)
    entry["horse_no"] = to_num(entry.get("馬番"), entry.index)
    horse_raw = entry["horse_id"] if "horse_id" in entry.columns else pd.Series(np.nan, index=entry.index)
    entry["horse_id"] = (
        horse_raw.replace("", np.nan)
        .fillna(entry.get("血統登録番号", pd.Series("", index=entry.index)))
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    entry["date_key"] = entry.get("日付", pd.Series("", index=entry.index)).map(normalize_date)
    missing_date = entry["date_key"].eq("") & entry.get("日付S", pd.Series("", index=entry.index)).notna()
    entry.loc[missing_date, "date_key"] = entry.loc[missing_date, "日付S"].map(normalize_date)
    missing_date = entry["date_key"].eq("") & entry["race_id"].astype(str).str.len().ge(8)
    entry.loc[missing_date, "date_key"] = entry.loc[missing_date, "race_id"].astype(str).str.slice(0, 8)
    entry["race_no"] = to_num(entry.get("Ｒ"), entry.index)
    entry["venue"] = entry.get("場所", pd.Series("", index=entry.index)).fillna("").astype(str)
    entry = entry[entry["race_id"].ne("") & entry["horse_no"].notna() & entry["horse_id"].ne("")]
    entry["horse_no"] = entry["horse_no"].astype(float)
    return entry[["race_id", "horse_no", "horse_id", "date_key", "venue", "race_no"]].drop_duplicates(
        ["race_id", "horse_no"], keep="last"
    )


def expand_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        pat = Path(pattern)
        expanded_pattern = str(pat if pat.is_absolute() else ROOT / pat)
        paths.extend(Path(value) for value in glob.glob(expanded_pattern, recursive=True))
    return sorted({p.resolve(): p for p in paths if p.exists()}.values())


def load_recent_result_history(result_globs: list[str], entry_globs: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for result_path in expand_globs(result_globs):
        try:
            result = read_csv(result_path, dtype=str)
        except Exception:
            continue
        if "race_id" not in result.columns or "horse_no" not in result.columns or "finish" not in result.columns:
            continue
        keep_cols = ["race_id", "horse_no", "finish"]
        for optional in ["venue", "race_no"]:
            if optional in result.columns:
                keep_cols.append(optional)
        work = result[keep_cols].copy()
        work["race_id"] = work["race_id"].astype(str).map(lambda x: re.sub(r"\D", "", x))
        work["horse_no"] = to_num(work.get("horse_no"), work.index)
        work["finish"] = to_num(work.get("finish"), work.index)
        if "race_no" in work.columns:
            work["race_no"] = to_num(work.get("race_no"), work.index)
        else:
            work["race_no"] = np.nan
        if "venue" not in work.columns:
            work["venue"] = ""
        work["venue"] = work["venue"].fillna("").astype(str)
        work = work[work["race_id"].str.len().ge(8) & work["horse_no"].notna() & work["finish"].notna()].copy()
        if work.empty:
            continue
        work["date_key"] = work["race_id"].str.slice(0, 8)
        date_keys = sorted(work["date_key"].dropna().unique())
        merged_parts: list[pd.DataFrame] = []
        for date_key in date_keys:
            entry_path = find_entry_file_for_date(str(date_key), entry_globs)
            if entry_path is None:
                continue
            entry_keys = load_entry_keys(entry_path)
            if entry_keys.empty:
                continue
            part = work[work["date_key"].eq(date_key)].merge(
                entry_keys[["race_id", "horse_no", "horse_id", "race_no"]],
                on=["race_id", "horse_no"],
                how="left",
                suffixes=("", "_entry"),
            )
            missing_horse = part["horse_id"].fillna("").astype(str).eq("")
            if missing_horse.any() and {"venue", "race_no"}.issubset(part.columns):
                fallback = work[work["date_key"].eq(date_key)].loc[
                    missing_horse.to_numpy(), ["race_id", "horse_no", "finish", "date_key", "venue", "race_no"]
                ].merge(
                    entry_keys[["date_key", "venue", "race_no", "horse_no", "horse_id"]],
                    on=["date_key", "venue", "race_no", "horse_no"],
                    how="left",
                )
                part.loc[missing_horse, "horse_id"] = fallback["horse_id"].to_numpy()
            merged_parts.append(part)
        if not merged_parts:
            continue
        merged = pd.concat(merged_parts, ignore_index=True)
        merged = merged[merged["horse_id"].fillna("").astype(str).ne("")]
        if merged.empty:
            continue
        field = merged.groupby("race_id")["horse_no"].transform("count").replace(0, np.nan)
        merged["target_score"] = ((field + 1.0 - merged["finish"]) / field).clip(0.0, 1.0)
        missing_race_no = merged.get("race_no", pd.Series(np.nan, index=merged.index)).isna()
        merged.loc[missing_race_no, "race_no"] = to_num(merged.loc[missing_race_no, "race_id"].str.slice(-2), merged.loc[missing_race_no].index)
        frames.append(
            merged[
                ["race_id", "horse_id", "date_key", "race_no", "horse_no", "target_score"]
            ].assign(recent_result_source=str(result_path))
        )
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["race_id", "horse_id"], keep="last")
    return out


def load_history(
    history_dir: Path,
    extra_history_csv: Path | None = None,
    recent_result_globs: list[str] | None = None,
    entry_globs: list[str] | None = None,
) -> pd.DataFrame:
    wanted = [
        "レースID(新/馬番無)",
        "race_id",
        "血統登録番号",
        "horse_id",
        "日付",
        "日付S",
        "Ｒ",
        "馬番",
        "target_score",
    ]
    frames: list[pd.DataFrame] = []
    for name in ["train_features_with_content_bridge.csv", "test_features_with_content_bridge.csv"]:
        path = history_dir / name
        if not path.exists():
            continue
        usecols = available_usecols(path, wanted)
        if usecols:
            frames.append(read_csv(path, dtype=str, usecols=usecols))
    if extra_history_csv is not None and extra_history_csv.exists():
        usecols = available_usecols(extra_history_csv, wanted)
        if usecols:
            frames.append(read_csv(extra_history_csv, dtype=str, usecols=usecols))
    if not frames:
        raise FileNotFoundError(f"No history feature CSVs found under {history_dir}")
    out = pd.concat(frames, ignore_index=True)
    race_raw = out["race_id"] if "race_id" in out.columns else pd.Series(np.nan, index=out.index)
    out["race_id"] = (
        race_raw.replace("", np.nan)
        .fillna(out.get("レースID(新/馬番無)", pd.Series("", index=out.index)))
        .astype(str)
        .map(lambda x: re.sub(r"\D", "", x))
    )
    horse_raw = out["horse_id"] if "horse_id" in out.columns else pd.Series(np.nan, index=out.index)
    out["horse_id"] = (
        horse_raw.replace("", np.nan)
        .fillna(out.get("血統登録番号", pd.Series("", index=out.index)))
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    out["date_key"] = out.get("日付", pd.Series("", index=out.index)).map(normalize_date)
    missing_date = out["date_key"].eq("") & out.get("日付S", pd.Series("", index=out.index)).notna()
    out.loc[missing_date, "date_key"] = out.loc[missing_date, "日付S"].map(normalize_date)
    out["race_no"] = to_num(out.get("Ｒ"), out.index)
    out["horse_no"] = to_num(out.get("馬番"), out.index)
    out = out[out["race_id"].ne("") & out["horse_id"].ne("") & out["date_key"].ne("")]
    out = out.drop_duplicates(["race_id", "horse_id"], keep="last")
    recent = load_recent_result_history(
        recent_result_globs or DEFAULT_RECENT_RESULT_GLOBS,
        entry_globs or DEFAULT_ENTRY_GLOBS,
    )
    if not recent.empty:
        out = pd.concat([out, recent], ignore_index=True, sort=False)
        out = out.drop_duplicates(["race_id", "horse_id"], keep="last")
    return add_basic_history_features(out)


def prepare_current_keys(prediction: pd.DataFrame, entry: pd.DataFrame | None) -> pd.DataFrame:
    cur = prediction.copy()
    cur["race_id"] = cur.apply(row_race_id, axis=1)
    cur["horse_no"] = to_num(cur.get("馬番"), cur.index)
    cur["date_key"] = cur.get("日付", pd.Series("", index=cur.index)).map(normalize_date)
    missing_date = cur["date_key"].eq("") & cur.get("日付S", pd.Series("", index=cur.index)).notna()
    cur.loc[missing_date, "date_key"] = cur.loc[missing_date, "日付S"].map(normalize_date)
    missing_date = cur["date_key"].eq("") & cur["race_id"].astype(str).str.len().ge(8)
    cur.loc[missing_date, "date_key"] = cur.loc[missing_date, "race_id"].astype(str).str.slice(0, 8)
    cur["horse_id"] = (
        cur.get("血統登録番号", pd.Series("", index=cur.index))
        .fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    if entry is not None and not entry.empty:
        ent = entry.copy()
        ent["race_id"] = ent.apply(row_race_id, axis=1)
        ent["horse_no"] = to_num(ent.get("馬番"), ent.index)
        ent["entry_horse_id"] = (
            ent.get("血統登録番号", pd.Series("", index=ent.index))
            .fillna("")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )
        ent = ent[["race_id", "horse_no", "entry_horse_id"]].drop_duplicates(["race_id", "horse_no"], keep="last")
        cur = cur.merge(ent, on=["race_id", "horse_no"], how="left")
        empty_horse_id = cur["horse_id"].eq("") | cur["horse_id"].str.lower().eq("nan")
        cur.loc[empty_horse_id, "horse_id"] = cur.loc[empty_horse_id, "entry_horse_id"].fillna("")
        cur = cur.drop(columns=["entry_horse_id"])
    return cur


def latest_history_before(history: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "recent_weighted_score_3",
        "recent_score_slope_3",
        "recent_score_jump_vs_mean",
        "recent_score_std_3",
        "ability_stability_score_3",
        "ability_ceiling_score_5",
        "ability_floor_score_5",
        "recent_score_count_3",
        "recent_score_count_5",
    ]
    cur = current[["race_id", "horse_no", "horse_id", "date_key"]].copy()
    cur["_current_row_id"] = np.arange(len(cur))
    hist = history[["horse_id", "date_key", "race_no", "horse_no", *feature_cols]].copy()
    hist = hist.rename(columns={"date_key": "history_date_key", "horse_no": "history_horse_no"})
    joined = cur.merge(hist, on="horse_id", how="left")
    joined = joined[
        joined["history_date_key"].notna()
        & joined["date_key"].notna()
        & joined["history_date_key"].astype(str).lt(joined["date_key"].astype(str))
    ].copy()
    if joined.empty:
        out = cur[["race_id", "horse_no"]].copy()
        for col in feature_cols:
            out[col] = np.nan
        out["basic_ability_history_ready"] = 0.0
        return out
    joined = joined.sort_values(["_current_row_id", "history_date_key", "race_no", "history_horse_no"])
    latest = joined.drop_duplicates("_current_row_id", keep="last")
    out = cur.merge(latest[["_current_row_id", *feature_cols]], on="_current_row_id", how="left")
    out["basic_ability_history_ready"] = out["ability_floor_score_5"].notna().astype(float)
    return out.drop(columns=["_current_row_id", "horse_id", "date_key"])


def add_row_level_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    prev_field = to_num(out.get("前走出走頭数"), idx).fillna(to_num(out.get("前走頭数"), idx)).replace(0, np.nan)
    denom = (prev_field - 1.0).replace(0, np.nan)
    prev_pop = to_num(out.get("前走人気"), idx)
    prev_corner4 = to_num(out.get("前4角.1"), idx)
    prev_final3f_rank = to_num(out.get("前走上り3F順"), idx)
    out["prev_popularity_rate"] = prev_pop / prev_field
    out["prev_market_rank_score"] = (1.0 - (prev_pop - 1.0) / denom).clip(0.0, 1.0)
    out["prev_corner4_position_rate"] = out.get("prev_corner4_position_rate", prev_corner4 / prev_field)
    out["prev_corner4_front_rate"] = (1.0 - (prev_corner4 - 1.0) / denom).clip(0.0, 1.0)
    out["prev_final3f_rank_rate"] = prev_final3f_rank / prev_field
    out["prev_final3f_excellence_rate"] = (1.0 - (prev_final3f_rank - 1.0) / denom).clip(0.0, 1.0)
    prev_margin = to_num(out.get("prev_margin_sec"), idx).fillna(to_num(out.get("前走着差タイム"), idx))
    prev_3f_gap = to_num(out.get("前走上3F地点差"), idx)
    out["prev_stretch_gain_sec"] = prev_3f_gap - prev_margin
    out["prev_late_improvement_score"] = out["prev_stretch_gain_sec"] * out["prev_final3f_excellence_rate"].fillna(0.5)
    out["prev_market_underestimated_score"] = to_num(out.get("prev_target_score_for_context"), idx) - out[
        "prev_market_rank_score"
    ]
    out["prev_market_overestimated_risk"] = out["prev_market_rank_score"] - to_num(
        out.get("prev_target_score_for_context"), idx
    )
    current_weight = to_num(out.get("斤量"), idx)
    prev_weight = to_num(out.get("前走斤量"), idx)
    prev_body = to_num(out.get("前走馬体重"), idx)
    out["weight_burden_ratio_prev_body"] = current_weight / prev_body.replace(0, np.nan)
    out["prev_weight_burden_ratio"] = prev_weight / prev_body.replace(0, np.nan)
    out["weight_burden_ratio_change"] = out["weight_burden_ratio_prev_body"] - out["prev_weight_burden_ratio"]
    return out


def race_rank_score(df: pd.DataFrame, col: str, higher_is_better: bool = True) -> pd.Series:
    idx = df.index
    values = to_num(df.get(col), idx)
    if "race_id" not in df.columns:
        return pd.Series(np.nan, index=idx)
    counts = values.notna().groupby(df["race_id"]).transform("sum")
    ranks = values.groupby(df["race_id"]).rank(ascending=not higher_is_better, method="average")
    return ((counts - ranks) / (counts - 1)).replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0.0, 1.0)


def add_time_value_refinements(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    for col in [
        "prev_class_time_value_score",
        "prev_time_adjusted_by_day_bias",
        "past3_avg_time_value",
        "past3_best_time_value",
        "past3_avg_time_z",
        "horse_time_value_plus_margin",
    ]:
        if col in out.columns:
            out[f"{col}_race_rank"] = race_rank_score(out, col, True)

    time_rank_parts = [
        out.get("prev_class_time_value_score_race_rank"),
        out.get("past3_avg_time_value_race_rank"),
        out.get("past3_best_time_value_race_rank"),
        out.get("horse_time_value_plus_margin_race_rank"),
    ]
    out["time_value_relative_rank_score"] = pd.concat(time_rank_parts, axis=1).mean(axis=1)

    prev_class_rank = out.get("prev_class_time_value_score_race_rank", pd.Series(0.5, index=idx))
    prev_bias_rank = out.get("prev_time_adjusted_by_day_bias_race_rank", pd.Series(0.5, index=idx))
    avg3_rank = out.get("past3_avg_time_value_race_rank", pd.Series(0.5, index=idx))
    best3_rank = out.get("past3_best_time_value_race_rank", pd.Series(0.5, index=idx))
    out["recency_weighted_time_score"] = (
        0.38 * prev_class_rank + 0.27 * prev_bias_rank + 0.23 * avg3_rank + 0.12 * out.get("past3_avg_time_z_race_rank", avg3_rank)
    ).clip(0.0, 1.0)

    best_gap = to_num(out.get("past3_best_time_value"), idx) - to_num(out.get("past3_avg_time_value"), idx)
    out["best_time_gap_raw"] = best_gap
    gap_rank = best_gap.groupby(out["race_id"]).rank(pct=True).fillna(0.5) if "race_id" in out.columns else pd.Series(0.5, index=idx)
    out["best_time_reproducibility"] = (0.62 * best3_rank + 0.38 * (1.0 - gap_rank)).clip(0.0, 1.0)
    out["time_score_consistency"] = (0.62 * avg3_rank + 0.38 * (1.0 - gap_rank)).clip(0.0, 1.0)

    surface = out.get("芝・ダ", pd.Series("", index=idx)).fillna("").astype(str)
    turf_avg = to_num(out.get("horse_turf_avg_score"), idx)
    dirt_avg = to_num(out.get("horse_dirt_avg_score"), idx)
    surface_support = pd.Series(np.where(surface.eq("芝"), turf_avg, np.where(surface.eq("ダ"), dirt_avg, np.nan)), index=idx)
    condition_support = pd.concat(
        [
            to_num(out.get("same_distance_category_avg_score"), idx),
            to_num(out.get("same_venue_avg_score"), idx),
            surface_support,
            to_num(out.get("lap_aptitude_fit_score"), idx),
        ],
        axis=1,
    ).mean(axis=1).clip(0.0, 1.0)
    time_safe = pd.concat([prev_class_rank, prev_bias_rank, avg3_rank], axis=1).mean(axis=1)
    out["condition_matched_time_score"] = (time_safe * (0.58 + 0.42 * condition_support.fillna(0.5))).clip(0.0, 1.0)

    race_mean_time = time_safe.groupby(out["race_id"]).transform("mean") if "race_id" in out.columns else pd.Series(0.5, index=idx)
    race_top_time = time_safe.groupby(out["race_id"]).transform("max") if "race_id" in out.columns else pd.Series(0.5, index=idx)
    pressure = to_num(out.get("race_early_pressure_score"), idx).clip(0.0, 1.0).fillna(0.5)
    front_adv = to_num(out.get("front_advantage_score"), idx).clip(0.0, 1.0).fillna(0.5)
    out["today_fast_clock_likelihood"] = (
        0.38 * race_mean_time.fillna(0.5)
        + 0.28 * race_top_time.fillna(0.5)
        + 0.22 * pressure
        + 0.12 * front_adv
    ).clip(0.0, 1.0)
    fast_lap = to_num(out.get("horse_fast_lap_score_past5"), idx).clip(0.0, 1.0).fillna(0.5)
    out["fast_clock_x_today_likelihood"] = (fast_lap * out["today_fast_clock_likelihood"]).clip(0.0, 1.0)

    out["pace_tracking_score"] = pd.concat(
        [
            race_rank_score(out, "prev_lap_pace_index", True),
            race_rank_score(out, "horse_fast_lap_score_past5", True),
        ],
        axis=1,
    ).mean(axis=1)
    out["late_speed_value"] = pd.concat(
        [
            race_rank_score(out, "prev_lap_finish_index", True),
            race_rank_score(out, "horse_instant_lap_score_past5", True),
        ],
        axis=1,
    ).mean(axis=1)
    out["sustained_speed_score"] = pd.concat(
        [
            race_rank_score(out, "prev_lap_sustain_index", True),
            race_rank_score(out, "prev_lap_long_spurt_index", True),
            race_rank_score(out, "horse_sustain_lap_score_past5", True),
            race_rank_score(out, "horse_long_spurt_lap_score_past5", True),
        ],
        axis=1,
    ).mean(axis=1)
    out["time_refinement_composite"] = (
        0.30 * out["condition_matched_time_score"]
        + 0.24 * out["time_value_relative_rank_score"]
        + 0.18 * out["recency_weighted_time_score"]
        + 0.14 * out["best_time_reproducibility"]
        + 0.14 * out["fast_clock_x_today_likelihood"]
    ).clip(0.0, 1.0)
    return out


def enrich_prediction(
    prediction_path: Path,
    entry_path: Path | None,
    output_path: Path,
    history_dir: Path,
    recent_result_globs: list[str] | None = None,
    entry_globs: list[str] | None = None,
) -> dict[str, object]:
    effective_recent_result_globs = recent_result_globs or DEFAULT_RECENT_RESULT_GLOBS
    matched_recent_result_paths = expand_globs(effective_recent_result_globs)
    prediction = read_csv(prediction_path, dtype=str)
    entry = read_csv(entry_path, dtype=str) if entry_path is not None and entry_path.exists() else pd.DataFrame()
    current = prepare_current_keys(prediction, entry)
    history = load_history(
        history_dir,
        recent_result_globs=effective_recent_result_globs,
        entry_globs=entry_globs,
    )
    hist_features = latest_history_before(history, current)
    out = current.merge(hist_features, on=["race_id", "horse_no"], how="left", suffixes=("", "_hist"))
    out = add_time_value_refinements(add_row_level_transforms(out))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary = {
        "prediction_csv": str(prediction_path),
        "entry_csv": str(entry_path) if entry_path else "",
        "output_csv": str(output_path),
        "rows": int(len(out)),
        "race_count": int(out["race_id"].nunique()) if "race_id" in out else 0,
        "horse_id_filled": int(out["horse_id"].ne("").sum()) if "horse_id" in out else 0,
        "basic_ability_history_ready_rows": int(
            to_num(out.get("basic_ability_history_ready"), out.index, 0.0).fillna(0.0).sum()
        ),
        "history_rows": int(len(history)),
        "history_date_min": str(history["date_key"].min()) if not history.empty else "",
        "history_date_max": str(history["date_key"].max()) if not history.empty else "",
        "recent_result_history_rows": int(
            history.get("recent_result_source", pd.Series("", index=history.index)).fillna("").astype(str).ne("").sum()
        )
        if not history.empty
        else 0,
        "matched_recent_result_file_count": len(matched_recent_result_paths),
        "matched_recent_result_files": [str(path) for path in matched_recent_result_paths],
    }
    summary_path = output_path.with_suffix(".basic_ability_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Add basic ability transform features to a live prediction CSV.")
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--entry-csv", default="")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument(
        "--recent-result-glob",
        action="append",
        default=[],
        help="Optional glob for recent result CSVs to append to history. Defaults include race_day_review and bias_review.",
    )
    parser.add_argument(
        "--entry-glob",
        action="append",
        default=[],
        help="Optional glob for entry snapshots used to recover horse IDs for recent result CSVs.",
    )
    parser.add_argument("--skip-recent-results", action="store_true")
    args = parser.parse_args()

    prediction_path = Path(args.prediction_csv)
    entry_path = Path(args.entry_csv) if args.entry_csv else None
    output_path = Path(args.output_csv)
    history_dir = Path(args.history_dir)
    if not prediction_path.is_absolute():
        prediction_path = ROOT / prediction_path
    if entry_path is not None and not entry_path.is_absolute():
        entry_path = ROOT / entry_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if not history_dir.is_absolute():
        history_dir = ROOT / history_dir
    recent_globs = [] if args.skip_recent_results else (args.recent_result_glob or DEFAULT_RECENT_RESULT_GLOBS)
    entry_globs = args.entry_glob or DEFAULT_ENTRY_GLOBS
    summary = enrich_prediction(
        prediction_path,
        entry_path,
        output_path,
        history_dir,
        recent_result_globs=recent_globs,
        entry_globs=entry_globs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
