from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def normalize_date(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) == 6:
        return "20" + digits
    if len(digits) >= 8:
        return digits[:8]
    return ""


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_race_day_label(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"第\s*(\d+)\s*日", str(value))
    return float(match.group(1)) if match else np.nan


def clip01(value: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(value, 0.0, 1.0)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def summarize_bool_segment(df: pd.DataFrame, mask: pd.Series, name: str) -> dict[str, object]:
    seg = df.loc[mask.fillna(False)].copy()
    if seg.empty:
        return {
            "segment": name,
            "races": 0,
            "runners": 0,
            "front_top3_rate": np.nan,
            "closer_top3_rate": np.nan,
            "inner_top3_rate": np.nan,
            "outer_top3_rate": np.nan,
            "front_pop_outperform_rate": np.nan,
            "closer_pop_outperform_rate": np.nan,
            "front_longshot_good_run_rate": np.nan,
            "closer_longshot_good_run_rate": np.nan,
        }
    front = seg["frontish"]
    closer = seg["closerish"]
    inner = seg["inner_gate"]
    outer = seg["outer_gate"]
    return {
        "segment": name,
        "races": int(seg["race_id"].nunique()),
        "runners": int(len(seg)),
        "front_top3_rate": safe_div(float(seg.loc[front, "top3"].sum()), int(front.sum())),
        "closer_top3_rate": safe_div(float(seg.loc[closer, "top3"].sum()), int(closer.sum())),
        "inner_top3_rate": safe_div(float(seg.loc[inner, "top3"].sum()), int(inner.sum())),
        "outer_top3_rate": safe_div(float(seg.loc[outer, "top3"].sum()), int(outer.sum())),
        "front_pop_outperform_rate": safe_div(float(seg.loc[front, "popularity_outperform"].sum()), int(front.sum())),
        "closer_pop_outperform_rate": safe_div(float(seg.loc[closer, "popularity_outperform"].sum()), int(closer.sum())),
        "front_longshot_good_run_rate": safe_div(
            float(seg.loc[front & seg["longshot"], "good_run"].sum()),
            int((front & seg["longshot"]).sum()),
        ),
        "closer_longshot_good_run_rate": safe_div(
            float(seg.loc[closer & seg["longshot"], "good_run"].sum()),
            int((closer & seg["longshot"]).sum()),
        ),
    }


def summarize_runner_roi(df: pd.DataFrame, mask: pd.Series, name: str) -> dict[str, object]:
    seg = df.loc[mask.fillna(False)].copy()
    if seg.empty:
        return {
            "segment": name,
            "races": 0,
            "horses": 0,
            "win_rate": np.nan,
            "top3_rate": np.nan,
            "win_roi": np.nan,
            "place_roi": np.nan,
            "avg_popularity": np.nan,
            "avg_odds": np.nan,
        }
    stake = len(seg) * 100.0
    return {
        "segment": name,
        "races": int(seg["race_id"].nunique()),
        "horses": int(len(seg)),
        "win_rate": float(seg["is_win"].mean()),
        "top3_rate": float(seg["is_top3"].mean()),
        "win_roi": safe_div(float(seg["win_return_100"].sum()), stake),
        "place_roi": safe_div(float(seg["place_return_100"].sum()), stake),
        "avg_popularity": float(seg["popularity"].mean()),
        "avg_odds": float(seg["odds"].mean()),
    }


def summarize_ticket_roi(df: pd.DataFrame, mask: pd.Series, name: str) -> dict[str, object]:
    seg = df.loc[mask.fillna(False)].copy()
    if seg.empty:
        return {
            "segment": name,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "roi": np.nan,
            "hit_rate": np.nan,
        }
    stake = float(seg["stake_yen"].sum())
    ret = float(seg["return_yen"].sum())
    return {
        "segment": name,
        "tickets": int(len(seg)),
        "races": int(seg["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "roi": safe_div(ret, stake),
        "hit_rate": float((seg["return_yen"] > 0).mean()),
    }


def build_track_regime_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    track_path = ROOT / "data/raw/track_condition_metrics.csv"
    races_path = ROOT / "data/processed/normalized/races.csv"

    track = read_csv(track_path, dtype=str)
    track["date_key"] = track["date"].map(normalize_date)
    track["venue"] = track["venue"].astype(str).str.strip()
    track["course"] = track["course"].astype(str).str.strip().replace({"nan": ""})
    track = track[(track["date_key"] >= "20250101") & (track["course"] != "")].copy()

    for col in [
        "cushion_value",
        "moisture_turf_goal",
        "moisture_turf_back",
        "moisture_dirt_goal",
        "moisture_dirt_back",
    ]:
        track[col] = to_num(track[col])
    track["turf_moisture_avg"] = track[["moisture_turf_goal", "moisture_turf_back"]].mean(axis=1)
    track["race_day_label_num"] = track["race_day_label"].map(parse_race_day_label)

    races = read_csv(
        races_path,
        dtype=str,
        usecols=["レースID(新/馬番無)", "日付", "場所", "Ｒ", "芝・ダ", "距離", "クラス名", "頭数", "出走頭数", "馬場状態"],
    )
    races = races.rename(
        columns={
            "レースID(新/馬番無)": "race_id",
            "場所": "venue",
            "Ｒ": "race_no",
            "芝・ダ": "surface",
            "距離": "distance",
            "クラス名": "class_name",
            "頭数": "field_size",
            "出走頭数": "starter_count",
            "馬場状態": "going",
        }
    )
    races["race_id"] = races["race_id"].astype(str)
    races["date_key"] = races["日付"].map(normalize_date)
    races["venue"] = races["venue"].astype(str).str.strip()
    races["race_no"] = to_num(races["race_no"])
    races["distance"] = to_num(races["distance"])
    races["field_size"] = to_num(races["field_size"])
    races["starter_count"] = to_num(races["starter_count"])
    races = races[races["date_key"] >= "20250101"].copy()

    race_days = races[["date_key", "venue"]].drop_duplicates()
    day = race_days.merge(track, on=["date_key", "venue"], how="left")
    day = day[day["course"].notna() & (day["course"] != "")].copy()
    day["date_dt"] = pd.to_datetime(day["date_key"], format="%Y%m%d")
    day = day.sort_values(["venue", "date_dt"]).reset_index(drop=True)

    records: list[dict[str, object]] = []
    for venue, group in day.groupby("venue", sort=False):
        meet_id = 0
        inferred_day = 0
        prev_label = np.nan
        prev_date = None
        prev_course = ""
        current_segment_id = 0
        last_change_date = None
        last_change_raceday_ordinal = 0
        for ordinal, (_, row) in enumerate(group.iterrows(), start=1):
            date_dt = row["date_dt"]
            label = row["race_day_label_num"]
            gap = (date_dt - prev_date).days if prev_date is not None else 999
            reset_by_label = pd.notna(label) and pd.notna(prev_label) and label <= prev_label
            reset_by_gap = gap > 21
            if prev_date is None or reset_by_label or reset_by_gap:
                meet_id += 1
                inferred_day = 1
                current_segment_id += 1
                meet_start = True
                last_change_date = date_dt
                last_change_raceday_ordinal = ordinal
            else:
                inferred_day += 1
                meet_start = False

            meet_day_index = int(label) if pd.notna(label) else inferred_day
            course_change = (
                (not meet_start)
                and bool(prev_course)
                and str(row["course"]) != str(prev_course)
            )
            if course_change:
                current_segment_id += 1
                last_change_date = date_dt
                last_change_raceday_ordinal = ordinal

            days_since_change = (date_dt - last_change_date).days if last_change_date is not None else 0
            race_days_since_change = ordinal - last_change_raceday_ordinal
            records.append(
                {
                    "date_key": row["date_key"],
                    "venue": venue,
                    "track_meet_id": f"{venue}_{meet_id}",
                    "track_course_segment_id": f"{venue}_{meet_id}_{current_segment_id}",
                    "meet_day_index": meet_day_index,
                    "meet_week_index": int(math.ceil(meet_day_index / 2.0)),
                    "opening_week_flag": int(meet_day_index <= 2),
                    "late_meet_flag": int(meet_day_index >= 6),
                    "meet_start_flag": int(meet_start),
                    "turf_course_code": row["course"],
                    "course_change_flag": int(course_change),
                    "days_since_course_change": int(days_since_change),
                    "race_days_since_course_change": int(race_days_since_change),
                    "cushion_value": row["cushion_value"],
                    "turf_moisture_avg": row["turf_moisture_avg"],
                    "moisture_turf_goal": row["moisture_turf_goal"],
                    "moisture_turf_back": row["moisture_turf_back"],
                }
            )
            prev_label = label
            prev_date = date_dt
            prev_course = str(row["course"])

    day_features = pd.DataFrame(records)
    rail_map = {"A": 0.0, "B": 3.0, "C": 6.0, "D": 9.0}
    width_map = {"A": 1.0, "B": 0.75, "C": 0.50, "D": 0.25}
    day_features["rail_offset_m"] = day_features["turf_course_code"].map(rail_map).astype(float)
    day_features["course_width_proxy"] = day_features["turf_course_code"].map(width_map).astype(float)

    race_features = races.merge(day_features, on=["date_key", "venue"], how="left")
    race_features = race_features[race_features["turf_course_code"].notna()].copy()
    race_features = race_features.sort_values(["venue", "date_key", "race_no"])
    race_features["surface_is_turf"] = race_features["surface"].astype(str).str.contains("芝").astype(int)
    turf = race_features["surface_is_turf"] == 1
    race_features["turf_races_since_meet_start"] = np.nan
    race_features["turf_races_since_course_change"] = np.nan
    race_features.loc[turf, "turf_races_since_meet_start"] = (
        race_features.loc[turf].groupby(["venue", "track_meet_id"]).cumcount()
    )
    race_features.loc[turf, "turf_races_since_course_change"] = (
        race_features.loc[turf].groupby(["venue", "track_course_segment_id"]).cumcount()
    )

    for col in [
        "turf_races_since_meet_start",
        "turf_races_since_course_change",
        "cushion_value",
        "turf_moisture_avg",
    ]:
        race_features[col] = to_num(race_features[col])
    cushion_q = race_features["cushion_value"].rank(pct=True).fillna(0.5)
    moisture_excess = clip01((race_features["turf_moisture_avg"].fillna(12.5) - 13.0) / 7.0)
    wear = clip01(
        race_features["turf_races_since_meet_start"].fillna(0) / 70.0 * 0.65
        + race_features["turf_races_since_course_change"].fillna(0) / 35.0 * 0.20
        + moisture_excess * 0.25
    )
    freshness_decay = np.exp(-race_features["turf_races_since_course_change"].fillna(0) / 14.0)
    recency = clip01(1.0 - race_features["days_since_course_change"].fillna(0) / 14.0)
    freshness = clip01(
        0.50 * freshness_decay
        + 0.20 * race_features["opening_week_flag"].fillna(0)
        + 0.20 * race_features["course_change_flag"].fillna(0)
        + 0.10 * recency
    )
    inside_front = clip01(
        0.25 * race_features["opening_week_flag"].fillna(0)
        + 0.25 * race_features["course_change_flag"].fillna(0)
        + 0.25 * freshness
        + 0.15 * cushion_q
        - 0.30 * wear
        - 0.15 * moisture_excess
    )
    outer_late = clip01(
        0.25 * race_features["late_meet_flag"].fillna(0)
        + 0.35 * wear
        + 0.20 * moisture_excess
        + 0.10 * (1.0 - race_features["course_width_proxy"].fillna(0.65))
        - 0.25 * freshness
    )
    speed_prior = clip01(
        0.25 * race_features["opening_week_flag"].fillna(0)
        + 0.20 * race_features["course_change_flag"].fillna(0)
        + 0.25 * cushion_q
        + 0.20 * freshness
        - 0.25 * moisture_excess
    )

    race_features["inner_wear_proxy"] = wear
    race_features["course_freshness_score"] = freshness
    race_features["opening_speed_bias_prior"] = speed_prior
    race_features["inside_front_bias_prior"] = inside_front
    race_features["outer_late_bias_prior"] = outer_late
    race_features["track_regime_confidence"] = np.maximum(inside_front, outer_late)
    race_features["track_regime_direction"] = np.select(
        [
            inside_front >= outer_late + 0.15,
            outer_late >= inside_front + 0.15,
        ],
        ["inside_front", "outer_late"],
        default="neutral",
    )

    return day_features, race_features


def evaluate_bias_outcomes(race_features: pd.DataFrame) -> pd.DataFrame:
    rf = race_features[race_features["surface_is_turf"] == 1].copy()
    race_ids = set(rf["race_id"].astype(str))
    feature_paths = [
        ROOT / "outputs/analysis/content_bridge_member_features_v1/train_features_with_content_bridge.csv",
        ROOT / "outputs/analysis/content_bridge_member_features_v1/test_features_with_content_bridge.csv",
    ]
    frames: list[pd.DataFrame] = []
    wanted = ["レースID(新/馬番無)", "枠番", "人気", "確定着順", "4角.1", "前4角.1", "出走頭数"]
    for path in feature_paths:
        if not path.exists():
            continue
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        usecols = [c for c in wanted if c in header.columns]
        tmp = read_csv(path, dtype=str, usecols=usecols).rename(columns={"レースID(新/馬番無)": "race_id"})
        tmp["race_id"] = tmp["race_id"].astype(str)
        tmp = tmp[tmp["race_id"].isin(race_ids)].copy()
        if not tmp.empty:
            frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "4角.1" in df.columns:
        df["corner4_source"] = df["4角.1"]
    elif "前4角.1" in df.columns:
        df["corner4_source"] = df["前4角.1"]
    else:
        df["corner4_source"] = np.nan

    keep_cols = [
        "race_id",
        "date_key",
        "venue",
        "race_no",
        "starter_count",
        "turf_course_code",
        "opening_week_flag",
        "late_meet_flag",
        "course_change_flag",
        "race_days_since_course_change",
        "turf_races_since_meet_start",
        "turf_races_since_course_change",
        "inner_wear_proxy",
        "course_freshness_score",
        "inside_front_bias_prior",
        "outer_late_bias_prior",
        "opening_speed_bias_prior",
        "track_regime_direction",
    ]
    df = df.merge(rf[keep_cols], on="race_id", how="inner")
    df["finish"] = to_num(df["確定着順"])
    df["corner4"] = to_num(df["corner4_source"])
    df["field"] = to_num(df["starter_count"]).fillna(to_num(df.get("出走頭数", pd.Series(index=df.index))))
    df["popularity"] = to_num(df["人気"])
    df["frame"] = to_num(df["枠番"])
    df = df[df["finish"].notna() & df["corner4"].notna() & df["field"].notna()].copy()
    df["top3"] = df["finish"] <= 3
    df["frontish"] = df["corner4"] <= np.maximum(3, np.ceil(df["field"] * 0.25))
    df["closerish"] = df["corner4"] >= np.ceil(df["field"] * 0.70)
    df["inner_gate"] = df["frame"] <= 3
    df["outer_gate"] = df["frame"] >= 7
    df["longshot"] = df["popularity"] >= 6
    df["good_run"] = (df["top3"]) | (df["finish"] <= 5) | (df["finish"] < df["popularity"])
    df["popularity_outperform"] = df["finish"] < df["popularity"]

    inside_q80 = df["inside_front_bias_prior"].quantile(0.80)
    inside_q90 = df["inside_front_bias_prior"].quantile(0.90)
    outer_q80 = df["outer_late_bias_prior"].quantile(0.80)
    outer_q90 = df["outer_late_bias_prior"].quantile(0.90)
    fresh_q80 = df["course_freshness_score"].quantile(0.80)
    wear_q80 = df["inner_wear_proxy"].quantile(0.80)

    rows = [
        summarize_bool_segment(df, pd.Series(True, index=df.index), "all_2025plus_turf_with_course"),
        summarize_bool_segment(df, df["opening_week_flag"] == 1, "opening_week"),
        summarize_bool_segment(df, df["late_meet_flag"] == 1, "late_meet"),
        summarize_bool_segment(df, df["course_change_flag"] == 1, "course_change_day"),
        summarize_bool_segment(df, df["race_days_since_course_change"] <= 1, "course_change_first2_racedays"),
        summarize_bool_segment(df, df["inner_wear_proxy"] >= 0.55, "high_inner_wear_proxy"),
        summarize_bool_segment(df, df["inside_front_bias_prior"] >= 0.55, "high_inside_front_prior"),
        summarize_bool_segment(df, df["outer_late_bias_prior"] >= 0.45, "high_outer_late_prior"),
        summarize_bool_segment(df, df["inside_front_bias_prior"] >= inside_q80, "inside_front_prior_top20pct"),
        summarize_bool_segment(df, df["inside_front_bias_prior"] >= inside_q90, "inside_front_prior_top10pct"),
        summarize_bool_segment(df, df["outer_late_bias_prior"] >= outer_q80, "outer_late_prior_top20pct"),
        summarize_bool_segment(df, df["outer_late_bias_prior"] >= outer_q90, "outer_late_prior_top10pct"),
        summarize_bool_segment(df, df["course_freshness_score"] >= fresh_q80, "course_freshness_top20pct"),
        summarize_bool_segment(df, df["inner_wear_proxy"] >= wear_q80, "inner_wear_top20pct"),
    ]
    return pd.DataFrame(rows)


def load_feature_rows(path: Path, source: str, race_ids: set[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    wanted = [
        "レースID(新/馬番無)",
        "枠番",
        "人気",
        "単勝オッズ",
        "確定着順",
        "単勝配当",
        "複勝配当",
        "horse_front_run_rate_past5",
        "horse_closer_rate_past5",
        "front_running_tendency",
        "closing_tendency",
    ]
    usecols = [c for c in wanted if c in header.columns]
    df = read_csv(path, dtype=str, usecols=usecols)
    df = df.rename(columns={"レースID(新/馬番無)": "race_id"})
    df["race_id"] = df["race_id"].astype(str)
    df = df[df["race_id"].isin(race_ids)].copy()
    df["source"] = source
    return df


def evaluate_runner_fit(race_features: pd.DataFrame) -> pd.DataFrame:
    rf = race_features[race_features["surface_is_turf"] == 1].copy()
    race_ids = set(rf["race_id"].astype(str))
    paths = [
        (ROOT / "outputs/analysis/content_bridge_member_features_v1/train_features_with_content_bridge.csv", "train"),
        (ROOT / "outputs/analysis/content_bridge_member_features_v1/test_features_with_content_bridge.csv", "test"),
    ]
    frames = [load_feature_rows(path, source, race_ids) for path, source in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.merge(
        rf[
            [
                "race_id",
                "date_key",
                "venue",
                "inside_front_bias_prior",
                "outer_late_bias_prior",
                "course_freshness_score",
                "inner_wear_proxy",
                "track_regime_direction",
            ]
        ],
        on="race_id",
        how="inner",
    )
    df["frame"] = to_num(df.get("枠番"))
    df["popularity"] = to_num(df.get("人気"))
    df["odds"] = to_num(df.get("単勝オッズ"))
    df["finish"] = to_num(df.get("確定着順"))
    df["front_rate"] = to_num(df.get("horse_front_run_rate_past5")).fillna(
        to_num(df.get("front_running_tendency")).fillna(0.0)
    )
    df["closer_rate"] = to_num(df.get("horse_closer_rate_past5")).fillna(
        to_num(df.get("closing_tendency")).fillna(0.0)
    )
    df["inner_gate_score"] = np.where(df["frame"] <= 2, 1.0, np.where(df["frame"] <= 3, 0.7, 0.0))
    df["outer_gate_score"] = np.where(df["frame"] >= 7, 1.0, np.where(df["frame"] >= 6, 0.7, 0.0))
    df["track_regime_fit_score"] = (
        df["inside_front_bias_prior"].fillna(0)
        * (0.55 * df["front_rate"].fillna(0) + 0.45 * df["inner_gate_score"])
        + df["outer_late_bias_prior"].fillna(0)
        * (0.55 * df["closer_rate"].fillna(0) + 0.45 * df["outer_gate_score"])
    )
    df["track_regime_mismatch_score"] = (
        df["inside_front_bias_prior"].fillna(0)
        * (0.55 * df["closer_rate"].fillna(0) + 0.45 * df["outer_gate_score"])
        + df["outer_late_bias_prior"].fillna(0)
        * (0.55 * df["front_rate"].fillna(0) + 0.45 * df["inner_gate_score"])
    )
    df["is_win"] = df["finish"] == 1
    df["is_top3"] = df["finish"] <= 3
    df["win_return_100"] = np.where(df["is_win"], to_num(df.get("単勝配当")).fillna(0), 0.0)
    df["place_return_100"] = np.where(df["is_top3"], to_num(df.get("複勝配当")).fillna(0), 0.0)
    df = df[df["finish"].notna()].copy()

    rows: list[dict[str, object]] = []
    for source, group in df.groupby("source"):
        q90 = group["track_regime_fit_score"].quantile(0.90)
        q80 = group["track_regime_fit_score"].quantile(0.80)
        q20 = group["track_regime_fit_score"].quantile(0.20)
        rows.extend(
            [
                summarize_runner_roi(group, pd.Series(True, index=group.index), f"{source}:all_turf_course_feature"),
                summarize_runner_roi(group, group["track_regime_fit_score"] >= q90, f"{source}:fit_top10pct"),
                summarize_runner_roi(group, group["track_regime_fit_score"] >= q80, f"{source}:fit_top20pct"),
                summarize_runner_roi(group, group["track_regime_fit_score"] <= q20, f"{source}:fit_bottom20pct"),
                summarize_runner_roi(
                    group,
                    (group["track_regime_fit_score"] >= q80) & (group["popularity"] >= 4),
                    f"{source}:fit_top20pct_pop4plus",
                ),
                summarize_runner_roi(
                    group,
                    (group["track_regime_fit_score"] >= q80) & (group["popularity"] >= 6),
                    f"{source}:fit_top20pct_pop6plus",
                ),
                summarize_runner_roi(
                    group,
                    (group["track_regime_mismatch_score"] >= group["track_regime_mismatch_score"].quantile(0.80)),
                    f"{source}:mismatch_top20pct",
                ),
            ]
        )
    return pd.DataFrame(rows)


def evaluate_selected_tickets(race_features: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "outputs/analysis/optimized_stack_ab_s_gate_v1/s_priority_selected_tickets.csv"
    if not path.exists():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    usecols = [
        c
        for c in [
            "race_id",
            "date_key",
            "ticket_type",
            "stake_yen",
            "return_yen",
            "operation_profile_label",
            "runtime_ticket_status",
        ]
        if c in header.columns
    ]
    df = read_csv(path, dtype=str, usecols=usecols)
    df["race_id"] = df["race_id"].astype(str)
    df["stake_yen"] = to_num(df["stake_yen"]).fillna(0)
    df["return_yen"] = to_num(df["return_yen"]).fillna(0)
    rf = race_features[race_features["surface_is_turf"] == 1].copy()
    df = df.merge(
        rf[
            [
                "race_id",
                "date_key",
                "venue",
                "inside_front_bias_prior",
                "outer_late_bias_prior",
                "course_freshness_score",
                "inner_wear_proxy",
                "track_regime_direction",
            ]
        ],
        on="race_id",
        how="inner",
        suffixes=("", "_race"),
    )
    rows = [
        summarize_ticket_roi(df, pd.Series(True, index=df.index), "selected_tickets:turf_course_feature"),
        summarize_ticket_roi(df, df["inside_front_bias_prior"] >= 0.55, "selected_tickets:high_inside_front_prior"),
        summarize_ticket_roi(df, df["outer_late_bias_prior"] >= 0.45, "selected_tickets:high_outer_late_prior"),
        summarize_ticket_roi(df, df["course_freshness_score"] >= 0.60, "selected_tickets:high_course_freshness"),
        summarize_ticket_roi(df, df["inner_wear_proxy"] >= 0.55, "selected_tickets:high_inner_wear"),
        summarize_ticket_roi(df, df["track_regime_direction"] == "inside_front", "selected_tickets:direction_inside_front"),
        summarize_ticket_roi(df, df["track_regime_direction"] == "outer_late", "selected_tickets:direction_outer_late"),
        summarize_ticket_roi(df, df["track_regime_direction"] == "neutral", "selected_tickets:direction_neutral"),
        summarize_ticket_roi(
            df,
            df["inside_front_bias_prior"] >= df["inside_front_bias_prior"].quantile(0.80),
            "selected_tickets:inside_front_prior_top20pct",
        ),
        summarize_ticket_roi(
            df,
            df["outer_late_bias_prior"] >= df["outer_late_bias_prior"].quantile(0.80),
            "selected_tickets:outer_late_prior_top20pct",
        ),
        summarize_ticket_roi(
            df,
            df["course_freshness_score"] >= df["course_freshness_score"].quantile(0.80),
            "selected_tickets:course_freshness_top20pct",
        ),
        summarize_ticket_roi(
            df,
            df["inner_wear_proxy"] >= df["inner_wear_proxy"].quantile(0.80),
            "selected_tickets:inner_wear_top20pct",
        ),
    ]
    return pd.DataFrame(rows)


def write_outputs(out_dir: Path, day_features: pd.DataFrame, race_features: pd.DataFrame) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    day_path = out_dir / "track_day_regime_features.csv"
    race_path = out_dir / "race_track_regime_features.csv"
    bias_path = out_dir / "bias_outcome_summary.csv"
    runner_path = out_dir / "runner_track_regime_fit_roi.csv"
    ticket_path = out_dir / "selected_ticket_track_regime_roi.csv"
    summary_path = out_dir / "summary.json"

    day_features.to_csv(day_path, index=False, encoding="utf-8-sig")
    race_features.to_csv(race_path, index=False, encoding="utf-8-sig")
    bias = evaluate_bias_outcomes(race_features)
    runner = evaluate_runner_fit(race_features)
    tickets = evaluate_selected_tickets(race_features)
    bias.to_csv(bias_path, index=False, encoding="utf-8-sig")
    runner.to_csv(runner_path, index=False, encoding="utf-8-sig")
    tickets.to_csv(ticket_path, index=False, encoding="utf-8-sig")

    def record_lookup(df: pd.DataFrame, segment: str, metric: str) -> float | None:
        if df.empty or "segment" not in df.columns:
            return None
        hit = df[df["segment"] == segment]
        if hit.empty or metric not in hit.columns:
            return None
        value = hit.iloc[0][metric]
        if pd.isna(value):
            return None
        return float(value)

    summary = {
        "output_dir": str(out_dir),
        "coverage": {
            "track_days_with_course": int(len(day_features)),
            "races_with_course": int(len(race_features)),
            "turf_races_with_course": int((race_features["surface_is_turf"] == 1).sum()),
            "date_min": str(race_features["date_key"].min()) if not race_features.empty else None,
            "date_max": str(race_features["date_key"].max()) if not race_features.empty else None,
        },
        "bias_read": {
            "all_front_top3_rate": record_lookup(bias, "all_2025plus_turf_with_course", "front_top3_rate"),
            "inside_prior_front_top3_rate": record_lookup(bias, "high_inside_front_prior", "front_top3_rate"),
            "inside_prior_closer_top3_rate": record_lookup(bias, "high_inside_front_prior", "closer_top3_rate"),
            "outer_prior_front_top3_rate": record_lookup(bias, "high_outer_late_prior", "front_top3_rate"),
            "outer_prior_closer_top3_rate": record_lookup(bias, "high_outer_late_prior", "closer_top3_rate"),
        },
        "runner_fit_read": {
            "test_all_win_roi": record_lookup(runner, "test:all_turf_course_feature", "win_roi"),
            "test_fit_top20_win_roi": record_lookup(runner, "test:fit_top20pct", "win_roi"),
            "test_fit_top20_place_roi": record_lookup(runner, "test:fit_top20pct", "place_roi"),
            "test_fit_top20_pop4_win_roi": record_lookup(runner, "test:fit_top20pct_pop4plus", "win_roi"),
            "test_mismatch_top20_win_roi": record_lookup(runner, "test:mismatch_top20pct", "win_roi"),
        },
        "selected_ticket_read": {
            "all_turf_course_roi": record_lookup(tickets, "selected_tickets:turf_course_feature", "roi"),
            "inside_prior_roi": record_lookup(tickets, "selected_tickets:high_inside_front_prior", "roi"),
            "outer_late_prior_roi": record_lookup(tickets, "selected_tickets:high_outer_late_prior", "roi"),
            "freshness_roi": record_lookup(tickets, "selected_tickets:high_course_freshness", "roi"),
            "inner_wear_roi": record_lookup(tickets, "selected_tickets:high_inner_wear", "roi"),
        },
        "notes": [
            "A/B/C/D course coverage is mostly 2025+ because earlier JRA PDF parsing leaves course blank.",
            "This is a pre-race track-regime prior verification, not a model retrain.",
            "Same-day live bias already exists elsewhere; this test checks opening week / course-change / inner-wear priors.",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs/analysis/track_regime_course_change_v1"),
        help="Output directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    day_features, race_features = build_track_regime_features()
    summary = write_outputs(Path(args.output_dir), day_features, race_features)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
