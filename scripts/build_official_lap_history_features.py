from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER_CSVS = [
    ROOT
    / "data"
    / "datasets"
    / "cache"
    / "workout_lap_pedigree_interactions_confirmed_opponent_2023plus"
    / "train_features.csv",
    ROOT
    / "data"
    / "datasets"
    / "cache"
    / "workout_lap_pedigree_interactions_confirmed_opponent_2023plus"
    / "test_features.csv",
]
DEFAULT_LAPS = ROOT / "data" / "processed" / "jra_official_race_laps" / "race_laps.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "jra_official_race_laps" / "official_lap_history_features.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "analysis" / "official_lap_history_features_v1" / "summary.json"


J_RACE_ID = "レースID(新/馬番無)"
J_HORSE_ID = "血統登録番号"
J_HORSE_NO = "馬番"
J_DATE = "日付"
J_DATE_S = "日付S"
J_FINISH = "確定着順"
J_POP = "人気"
J_TARGET_SCORE = "target_score"
J_SURFACE = "芝・ダ"
J_DISTANCE = "距離"
J_CLASS = "クラス名"
J_GOING = "馬場状態"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv_any(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def clean_id(s: pd.Series, width: int | None = None) -> pd.Series:
    out = s.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip()
    return out.str.zfill(width) if width else out


def parse_date(s: pd.Series) -> pd.Series:
    raw = s.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip()
    out = pd.to_datetime(raw, errors="coerce")
    mask6 = raw.str.fullmatch(r"\d{6}")
    if mask6.any():
        out.loc[mask6] = pd.to_datetime(raw.loc[mask6], format="%y%m%d", errors="coerce")
    mask8 = raw.str.fullmatch(r"\d{8}")
    if mask8.any():
        out.loc[mask8] = pd.to_datetime(raw.loc[mask8], format="%Y%m%d", errors="coerce")
    return out


def first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def load_runner_rows(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    wanted = [
        J_RACE_ID,
        J_HORSE_ID,
        J_HORSE_NO,
        J_DATE,
        J_DATE_S,
        J_FINISH,
        J_POP,
        J_TARGET_SCORE,
        J_SURFACE,
        J_DISTANCE,
        J_CLASS,
        J_GOING,
        "馬名",
    ]
    for path in paths:
        if not path.exists():
            continue
        header = read_csv_any(path, nrows=0)
        header_cols = set(header.columns)
        if {"race_id", "horse_id", "horse_no"}.issubset(header_cols):
            usecols = [
                c
                for c in [
                    "race_id",
                    "horse_id",
                    "horse_no",
                    "horse_name",
                    "finish",
                    "popularity",
                    "surface",
                    "distance_m",
                    "distance",
                    "course_detail",
                ]
                if c in header_cols
            ]
            frame = read_csv_any(path, usecols=usecols)
            frame["_source_csv"] = str(path)
            frame["race_id"] = clean_id(frame["race_id"], 16)
            frame["horse_id"] = clean_id(frame["horse_id"])
            frame["horse_no"] = pd.to_numeric(frame["horse_no"], errors="coerce").astype("Int64")
            frame["race_date"] = pd.to_datetime(frame["race_id"].str[:8], format="%Y%m%d", errors="coerce")
            frame["finish"] = pd.to_numeric(frame.get("finish"), errors="coerce")
            frame["popularity"] = pd.to_numeric(frame.get("popularity"), errors="coerce")
            frame["target_score"] = np.where(frame["finish"].le(1), 1.0, np.where(frame["finish"].le(3), 0.72, 0.36))
            frame["surface"] = frame.get("surface", pd.Series("", index=frame.index)).astype("string").fillna("")
            if "distance" in frame.columns:
                frame["distance"] = pd.to_numeric(frame["distance"], errors="coerce")
            else:
                frame["distance"] = pd.to_numeric(frame.get("distance_m"), errors="coerce")
            frame["class_name"] = ""
            frame["going"] = ""
            frames.append(frame)
            continue
        usecols = [c for c in wanted if c in header.columns]
        if not {J_RACE_ID, J_HORSE_ID, J_HORSE_NO}.issubset(usecols):
            continue
        frame = read_csv_any(path, usecols=usecols)
        frame["_source_csv"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True, sort=False)
    if "race_id" not in raw.columns:
        raw["race_id"] = ""
    if J_RACE_ID in raw.columns:
        raw["race_id"] = raw["race_id"].astype("string").fillna("").mask(
            raw["race_id"].astype("string").fillna("").eq(""),
            clean_id(raw[J_RACE_ID], 16),
        )
    raw["race_id"] = clean_id(raw["race_id"], 16)

    if "horse_id" not in raw.columns:
        raw["horse_id"] = ""
    if J_HORSE_ID in raw.columns:
        raw["horse_id"] = raw["horse_id"].astype("string").fillna("").mask(
            raw["horse_id"].astype("string").fillna("").eq(""),
            clean_id(raw[J_HORSE_ID]),
        )
    raw["horse_id"] = clean_id(raw["horse_id"])

    if "horse_no" not in raw.columns:
        raw["horse_no"] = pd.NA
    if J_HORSE_NO in raw.columns:
        raw["horse_no"] = pd.to_numeric(raw["horse_no"], errors="coerce").fillna(
            pd.to_numeric(raw[J_HORSE_NO], errors="coerce")
        )
    raw["horse_no"] = pd.to_numeric(raw["horse_no"], errors="coerce").astype("Int64")

    if "race_date" not in raw.columns:
        raw["race_date"] = pd.NaT
    if J_DATE in raw.columns:
        raw["race_date"] = pd.to_datetime(raw["race_date"], errors="coerce").fillna(parse_date(raw[J_DATE]))
    elif J_DATE_S in raw.columns:
        raw["race_date"] = pd.to_datetime(raw["race_date"], errors="coerce").fillna(parse_date(raw[J_DATE_S]))
    raw["race_date"] = pd.to_datetime(raw["race_date"], errors="coerce").fillna(
        pd.to_datetime(raw["race_id"].str[:8], format="%Y%m%d", errors="coerce")
    )

    if "finish" not in raw.columns:
        raw["finish"] = pd.NA
    if J_FINISH in raw.columns:
        raw["finish"] = pd.to_numeric(raw["finish"], errors="coerce").fillna(pd.to_numeric(raw[J_FINISH], errors="coerce"))
    raw["finish"] = pd.to_numeric(raw["finish"], errors="coerce")

    if "popularity" not in raw.columns:
        raw["popularity"] = pd.NA
    if J_POP in raw.columns:
        raw["popularity"] = pd.to_numeric(raw["popularity"], errors="coerce").fillna(pd.to_numeric(raw[J_POP], errors="coerce"))
    raw["popularity"] = pd.to_numeric(raw["popularity"], errors="coerce")

    if "target_score" not in raw.columns:
        raw["target_score"] = pd.NA
    if J_TARGET_SCORE in raw.columns:
        raw["target_score"] = pd.to_numeric(raw["target_score"], errors="coerce").fillna(
            pd.to_numeric(raw[J_TARGET_SCORE], errors="coerce")
        )
    raw["target_score"] = pd.to_numeric(raw["target_score"], errors="coerce")

    if "surface" not in raw.columns:
        raw["surface"] = ""
    if J_SURFACE in raw.columns:
        raw["surface"] = raw["surface"].astype("string").fillna("").mask(
            raw["surface"].astype("string").fillna("").eq(""),
            raw[J_SURFACE].astype("string").fillna(""),
        )
    raw["surface"] = raw["surface"].astype("string").fillna("")

    if "distance" not in raw.columns:
        raw["distance"] = pd.NA
    if J_DISTANCE in raw.columns:
        raw["distance"] = pd.to_numeric(raw["distance"], errors="coerce").fillna(pd.to_numeric(raw[J_DISTANCE], errors="coerce"))
    raw["distance"] = pd.to_numeric(raw["distance"], errors="coerce")

    if "class_name" not in raw.columns:
        raw["class_name"] = ""
    if J_CLASS in raw.columns:
        raw["class_name"] = raw["class_name"].astype("string").fillna("").mask(
            raw["class_name"].astype("string").fillna("").eq(""),
            raw[J_CLASS].astype("string").fillna(""),
        )
    raw["class_name"] = raw["class_name"].astype("string").fillna("")

    if "going" not in raw.columns:
        raw["going"] = ""
    if J_GOING in raw.columns:
        raw["going"] = raw["going"].astype("string").fillna("").mask(
            raw["going"].astype("string").fillna("").eq(""),
            raw[J_GOING].astype("string").fillna(""),
        )
    raw["going"] = raw["going"].astype("string").fillna("")
    raw = raw.dropna(subset=["race_id", "horse_id", "horse_no", "race_date"]).copy()
    raw = raw.sort_values(["horse_id", "race_date", "race_id"], kind="mergesort")
    raw = raw.drop_duplicates(["race_id", "horse_no"], keep="last")
    return raw


def load_laps(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    laps = read_csv_any(path)
    if "race_id" not in laps.columns:
        return pd.DataFrame()
    laps["race_id"] = clean_id(laps["race_id"], 16)
    for col in [
        "first_3f_sec",
        "last_3f_sec",
        "front_back_3f_diff_sec",
        "last_4f_sec",
        "finish_1f_accel_sec",
        "l1_vs_l2_prev_accel_sec",
        "l2_vs_l3_prev_accel_sec",
        "l3_vs_l4_prev_accel_sec",
        "lap_std_sec",
        "lap_range_sec",
        "distance_m",
    ]:
        if col in laps.columns:
            laps[col] = pd.to_numeric(laps[col], errors="coerce")
    return laps.drop_duplicates("race_id", keep="last")


def add_official_run_scores(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    score = pd.to_numeric(out["target_score"], errors="coerce")
    # Keep missing target_score neutral rather than making unknown races look bad.
    fallback_score = pd.Series(
        np.where(pd.to_numeric(out["finish"], errors="coerce").le(3), 0.72, 0.36),
        index=out.index,
        dtype=float,
    )
    score = score.fillna(fallback_score).clip(0.0, 1.0)
    diff = pd.to_numeric(out.get("front_back_3f_diff_sec"), errors="coerce")
    accel1 = pd.to_numeric(out.get("finish_1f_accel_sec"), errors="coerce")
    accel2 = pd.to_numeric(out.get("l2_vs_l3_prev_accel_sec"), errors="coerce")
    accel3 = pd.to_numeric(out.get("l3_vs_l4_prev_accel_sec"), errors="coerce")
    lap_std = pd.to_numeric(out.get("lap_std_sec"), errors="coerce")

    out["official_front_load_need"] = ((-diff) / 3.0).clip(0.0, 1.0).fillna(0.0)
    out["official_slow_finish_need"] = (diff / 3.0).clip(0.0, 1.0).fillna(0.0)
    out["official_l1_instant_need"] = ((accel1 + diff.clip(lower=0.0) * 0.18) / 1.0).clip(0.0, 1.0).fillna(0.0)
    out["official_l2_sustain_need"] = (
        1.0
        - (
            out["official_front_load_need"].fillna(0.0)
            + out["official_l1_instant_need"].fillna(0.0)
        )
        / 2.0
    ).clip(0.0, 1.0)
    out["official_l3_long_spurt_need"] = ((accel3 + accel2.fillna(0.0) * 0.35 + (-diff).clip(lower=0.0) * 0.10) / 1.2).clip(0.0, 1.0).fillna(0.0)
    out["official_lap_wave_volatility"] = (lap_std / 1.0).clip(0.0, 1.0).fillna(0.0)

    for key in ["front_load", "slow_finish", "l1_instant", "l2_sustain", "l3_long_spurt"]:
        out[f"official_{key}_goodrun_score"] = (out[f"official_{key}_need"] * score).clip(0.0, 1.0)
    out["official_lap_goodrun_score"] = score
    out["official_lap_available"] = out["front_back_3f_diff_sec"].notna().astype(float)
    available = out["official_lap_available"].astype(float)
    for col in [
        "official_front_load_need",
        "official_slow_finish_need",
        "official_l1_instant_need",
        "official_l2_sustain_need",
        "official_l3_long_spurt_need",
        "official_lap_wave_volatility",
        "official_front_load_goodrun_score",
        "official_slow_finish_goodrun_score",
        "official_l1_instant_goodrun_score",
        "official_l2_sustain_goodrun_score",
        "official_l3_long_spurt_goodrun_score",
        "official_lap_goodrun_score",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").where(available.gt(0))
    return out


def rolling_prior_features(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out = rows.sort_values(["horse_id", "race_date", "race_id"], kind="mergesort").copy()
    group = out.groupby("horse_id", sort=False)
    source_cols = [
        "official_front_load_goodrun_score",
        "official_slow_finish_goodrun_score",
        "official_l1_instant_goodrun_score",
        "official_l2_sustain_goodrun_score",
        "official_l3_long_spurt_goodrun_score",
        "official_lap_goodrun_score",
        "official_front_load_need",
        "official_slow_finish_need",
        "official_l1_instant_need",
        "official_l2_sustain_need",
        "official_l3_long_spurt_need",
        "official_lap_wave_volatility",
        "first_3f_sec",
        "last_3f_sec",
        "front_back_3f_diff_sec",
        "last_4f_sec",
        "finish_1f_accel_sec",
    ]
    for col in source_cols:
        if col not in out.columns:
            out[col] = np.nan
        shifted = group[col].shift()
        out[f"{col}_past3_mean"] = shifted.groupby(out["horse_id"], sort=False).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"{col}_past3_max"] = shifted.groupby(out["horse_id"], sort=False).rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    out["official_lap_history_count_past3"] = (
        group["official_lap_available"]
        .shift()
        .groupby(out["horse_id"], sort=False)
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    out["official_lap_history_ready"] = out["official_lap_history_count_past3"].gt(0).astype(float)
    out["official_lap_profile_strength"] = out[
        [
            "official_front_load_goodrun_score_past3_max",
            "official_slow_finish_goodrun_score_past3_max",
            "official_l1_instant_goodrun_score_past3_max",
            "official_l2_sustain_goodrun_score_past3_max",
            "official_l3_long_spurt_goodrun_score_past3_max",
        ]
    ].max(axis=1, skipna=True)
    out["official_lap_profile_versatility"] = out[
        [
            "official_front_load_goodrun_score_past3_mean",
            "official_slow_finish_goodrun_score_past3_mean",
            "official_l1_instant_goodrun_score_past3_mean",
            "official_l2_sustain_goodrun_score_past3_mean",
            "official_l3_long_spurt_goodrun_score_past3_mean",
        ]
    ].gt(0.20).sum(axis=1)

    keep = [
        "race_id",
        "horse_no",
        "horse_id",
        "race_date",
        "official_lap_history_count_past3",
        "official_lap_history_ready",
        "official_lap_profile_strength",
        "official_lap_profile_versatility",
    ]
    keep.extend([c for c in out.columns if c.endswith("_past3_mean") or c.endswith("_past3_max")])
    return out[keep].drop_duplicates(["race_id", "horse_no"], keep="last")


def summarize(enriched: pd.DataFrame, features: pd.DataFrame, out_csv: Path) -> dict[str, Any]:
    matched = enriched["official_lap_available"].fillna(0).gt(0)
    ready = features["official_lap_history_ready"].fillna(0).gt(0) if not features.empty else pd.Series(dtype=bool)
    by_year = {}
    if not enriched.empty:
        tmp = enriched.copy()
        tmp["year"] = tmp["race_date"].dt.year
        by_year = (
            tmp.groupby("year")["official_lap_available"]
            .agg(rows="size", official_lap_rows="sum")
            .reset_index()
            .to_dict(orient="records")
        )
    return {
        "output_csv": str(out_csv),
        "runner_rows": int(len(enriched)),
        "races": int(enriched["race_id"].nunique()) if not enriched.empty else 0,
        "official_lap_matched_rows": int(matched.sum()) if not enriched.empty else 0,
        "official_lap_matched_races": int(enriched.loc[matched, "race_id"].nunique()) if not enriched.empty else 0,
        "history_feature_rows": int(len(features)),
        "history_ready_rows": int(ready.sum()) if not features.empty else 0,
        "history_ready_races": int(features.loc[ready, "race_id"].nunique()) if not features.empty else 0,
        "by_year": by_year,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build horse prior official-lap history features from JRA official result laps.")
    parser.add_argument("--runner-csv", action="append", type=Path, default=None)
    parser.add_argument("--laps-csv", type=Path, default=DEFAULT_LAPS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    runner_paths = [project_path(p) for p in (args.runner_csv or DEFAULT_RUNNER_CSVS)]
    laps_path = project_path(args.laps_csv)
    out_csv = project_path(args.output_csv)
    summary_json = project_path(args.summary_json)

    runners = load_runner_rows(runner_paths)
    laps = load_laps(laps_path)
    if runners.empty:
        raise SystemExit("No runner rows loaded.")
    if laps.empty:
        raise SystemExit("No official lap rows loaded.")
    enriched = runners.merge(laps, on="race_id", how="left", suffixes=("", "_official"))
    enriched = add_official_run_scores(enriched)
    features = rolling_prior_features(enriched)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_csv, index=False, encoding="utf-8-sig")
    summary = summarize(enriched, features, out_csv)
    summary["runner_csvs"] = [str(p) for p in runner_paths]
    summary["laps_csv"] = str(laps_path)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not features.empty:
        print(features.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
