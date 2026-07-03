from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.workout_history import add_workout_pattern_history_features
from src.features.workouts import build_workout_features, merge_workout_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/test feature cache enriched with TARGET workout features.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--workouts-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lookback-days", type=int, default=21)
    parser.add_argument("--race-col", default="レースID(新/馬番無)")
    parser.add_argument("--date-col", default="日付")
    parser.add_argument("--horse-id-col", default="血統登録番号")
    parser.add_argument("--trainer-col", default="調教師コード")
    parser.add_argument("--rank-col", default="確定着順")
    parser.add_argument("--odds-col", default="単勝オッズ")
    parser.add_argument("--min-race-date", default=None, help="Optional minimum race date, e.g. 230101 or 20230101.")
    parser.add_argument("--max-race-date", default=None, help="Optional maximum race date, e.g. 260613 or 20260613.")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, encoding="utf-8-sig", low_memory=False).copy()
    test = pd.read_csv(args.test_csv, encoding="utf-8-sig", low_memory=False).copy()
    train = _filter_by_race_date(train, args.date_col, args.min_race_date, args.max_race_date)
    test = _filter_by_race_date(test, args.date_col, args.min_race_date, args.max_race_date)
    train = train.assign(_cache_split="train", _cache_order=np.arange(len(train)))
    test = test.assign(_cache_split="test", _cache_order=np.arange(len(test)))
    frame = pd.concat([train, test], ignore_index=True, sort=False)

    workouts = pd.read_csv(args.workouts_csv, encoding="utf-8-sig", low_memory=False)
    selected = select_workouts_for_frame(
        frame,
        workouts,
        race_col=args.race_col,
        date_col=args.date_col,
        horse_id_col=args.horse_id_col,
        trainer_col=args.trainer_col,
        lookback_days=args.lookback_days,
    )
    workout_features = build_workout_features(selected) if not selected.empty else pd.DataFrame(columns=["race_id", "horse_id"])
    enriched = merge_workout_features(
        frame,
        workout_features,
        race_col=args.race_col,
        horse_id_col=args.horse_id_col,
    )
    enriched = add_workout_pattern_history_features(
        enriched,
        race_col=args.race_col,
        date_col=args.date_col,
        rank_col=args.rank_col,
        odds_col=args.odds_col if args.odds_col in enriched.columns else None,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = (
        enriched[enriched["_cache_split"] == "train"]
        .sort_values("_cache_order", kind="mergesort")
        .drop(columns=["_cache_split", "_cache_order"], errors="ignore")
    )
    test_out = (
        enriched[enriched["_cache_split"] == "test"]
        .sort_values("_cache_order", kind="mergesort")
        .drop(columns=["_cache_split", "_cache_order"], errors="ignore")
    )
    train_path = output_dir / "train_features.csv"
    test_path = output_dir / "test_features.csv"
    metadata_path = output_dir / "metadata.json"
    train_out.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_out.to_csv(test_path, index=False, encoding="utf-8-sig")

    workout_cols = [col for col in enriched.columns if col.startswith("workout_")]
    matched_rows = int(enriched[workout_cols].notna().any(axis=1).sum()) if workout_cols else 0
    metadata = {
        "train_rows": int(len(train_out)),
        "test_rows": int(len(test_out)),
        "selected_workout_rows": int(len(selected)),
        "workout_feature_columns": workout_cols,
        "matched_rows": matched_rows,
        "matched_rate": float(matched_rows / len(enriched)) if len(enriched) else 0.0,
        "lookback_days": args.lookback_days,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "workouts_csv": args.workouts_csv,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def select_workouts_for_frame(
    frame: pd.DataFrame,
    workouts: pd.DataFrame,
    *,
    race_col: str,
    date_col: str,
    horse_id_col: str,
    trainer_col: str,
    lookback_days: int,
) -> pd.DataFrame:
    entries = frame[[race_col, date_col, horse_id_col, trainer_col]].copy()
    entries["_race_id"] = entries[race_col].astype("string")
    entries["_horse_id"] = entries[horse_id_col].astype("string")
    entries["_race_date_dt"] = _to_datetime(entries[date_col])
    entries["_trainer_code"] = entries[trainer_col].astype("string") if trainer_col in entries.columns else pd.NA
    entries = entries.dropna(subset=["_horse_id", "_race_date_dt"]).reset_index(drop=True)
    entries["_entry_index"] = np.arange(len(entries))

    w = workouts.copy()
    w["_horse_id"] = w["horse_id"].astype("string")
    w["_workout_date_dt"] = _to_datetime(w["workout_date"])
    w = w.dropna(subset=["_horse_id", "_workout_date_dt"]).sort_values(["_horse_id", "_workout_date_dt"], kind="mergesort").reset_index(drop=True)
    w["_workout_index"] = np.arange(len(w))

    selected_workout_indices: list[np.ndarray] = []
    selected_entry_indices: list[np.ndarray] = []
    workout_groups = {horse_id: group for horse_id, group in w.groupby("_horse_id", sort=False, observed=True)}
    for horse_id, entry_group in entries.groupby("_horse_id", sort=False, observed=True):
        horse_workouts = workout_groups.get(horse_id)
        if horse_workouts is None:
            continue
        dates = horse_workouts["_workout_date_dt"].to_numpy(dtype="datetime64[ns]")
        race_dates = entry_group["_race_date_dt"].to_numpy(dtype="datetime64[ns]")
        start_dates = race_dates - np.timedelta64(lookback_days, "D")
        lefts = np.searchsorted(dates, start_dates, side="left")
        rights = np.searchsorted(dates, race_dates, side="right")
        workout_index_values = horse_workouts["_workout_index"].to_numpy()
        entry_index_values = entry_group["_entry_index"].to_numpy()
        for entry_index, left, right in zip(entry_index_values, lefts, rights):
            if right <= left:
                continue
            selected_workout_indices.append(workout_index_values[left:right])
            selected_entry_indices.append(np.full(right - left, entry_index, dtype=np.int64))

    if not selected_workout_indices:
        return pd.DataFrame()
    workout_idx = np.concatenate(selected_workout_indices)
    entry_idx = np.concatenate(selected_entry_indices)
    selected = w.iloc[workout_idx].copy().reset_index(drop=True)
    matched_entries = entries.iloc[entry_idx].reset_index(drop=True)
    selected["race_id"] = matched_entries["_race_id"].to_numpy()
    selected["race_date"] = matched_entries["_race_date_dt"].dt.strftime("%Y%m%d").to_numpy()
    selected["horse_id"] = matched_entries["_horse_id"].to_numpy()
    selected["trainer_code"] = matched_entries["_trainer_code"].to_numpy()
    return selected.drop(columns=["_horse_id", "_workout_date_dt", "_workout_index"], errors="ignore")


def _to_datetime(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    text = text.where(~text.str.fullmatch(r"\d{6}", na=False), "20" + text)
    return pd.to_datetime(text, errors="coerce", format="mixed")


def _filter_by_race_date(frame: pd.DataFrame, date_col: str, min_date: str | None, max_date: str | None) -> pd.DataFrame:
    if not min_date and not max_date:
        return frame
    dates = pd.to_numeric(frame[date_col], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if min_date:
        mask &= dates >= _normalize_compare_date(min_date)
    if max_date:
        mask &= dates <= _normalize_compare_date(max_date)
    return frame[mask].copy()


def _normalize_compare_date(value: str) -> int:
    text = str(value).strip()
    if len(text) == 8 and text.startswith("20"):
        text = text[2:]
    return int(text)


if __name__ == "__main__":
    main()
