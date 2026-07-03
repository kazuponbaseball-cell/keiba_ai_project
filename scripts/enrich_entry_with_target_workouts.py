from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.workouts import build_workout_features, merge_workout_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach TARGET workout records to entries and build workout features.")
    parser.add_argument("--entry-csv", required=True)
    parser.add_argument("--workouts-csv", default="data/processed/target/workouts.csv")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--lookback-days", type=int, default=21)
    parser.add_argument("--race-col", default=None)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--horse-id-col", default=None)
    parser.add_argument("--trainer-col", default=None)
    args = parser.parse_args()

    entry_path = Path(args.entry_csv)
    workouts_path = Path(args.workouts_csv)
    output_path = Path(args.output_csv) if args.output_csv else entry_path.with_name(f"{entry_path.stem}_with_target_workouts.csv")

    entry = pd.read_csv(entry_path, encoding="utf-8-sig", low_memory=False)
    workouts = pd.read_csv(workouts_path, encoding="utf-8-sig", low_memory=False)

    race_col = args.race_col or _first_existing(entry, ["race_id", "レースID(新/馬番無)", "繝ｬ繝ｼ繧ｹID(譁ｰ/鬥ｬ逡ｪ辟｡)"])
    date_col = args.date_col or _first_existing(entry, ["date", "日付", "譌･莉・"])
    horse_id_col = args.horse_id_col or _first_existing(entry, ["horse_id", "血統登録番号", "陦邨ｱ逋ｻ骭ｲ逡ｪ蜿ｷ"])
    trainer_col = args.trainer_col or _first_existing(entry, ["trainer_code", "調教師コード", "隱ｿ謨吝ｸｫ繧ｳ繝ｼ繝・"], required=False)

    selected = _select_workouts_for_entries(
        entry,
        workouts,
        race_col=race_col,
        date_col=date_col,
        horse_id_col=horse_id_col,
        trainer_col=trainer_col,
        lookback_days=args.lookback_days,
    )
    features = build_workout_features(selected) if not selected.empty else pd.DataFrame(columns=["race_id", "horse_id"])
    enriched = merge_workout_features(entry, features, race_col=race_col, horse_id_col=horse_id_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False, encoding="utf-8-sig")
    workout_cols = [col for col in enriched.columns if col.startswith("workout_")]
    matched = int(enriched[workout_cols].notna().any(axis=1).sum()) if workout_cols else 0
    print(
        json.dumps(
            {
                "entry_rows": int(len(entry)),
                "selected_workout_rows": int(len(selected)),
                "matched_entry_rows": matched,
                "lookback_days": args.lookback_days,
                "output_csv": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _select_workouts_for_entries(
    entry: pd.DataFrame,
    workouts: pd.DataFrame,
    *,
    race_col: str,
    date_col: str,
    horse_id_col: str,
    trainer_col: str | None,
    lookback_days: int,
) -> pd.DataFrame:
    e = entry[[race_col, date_col, horse_id_col] + ([trainer_col] if trainer_col else [])].copy()
    e["_race_id"] = e[race_col].astype("string")
    e["_horse_id"] = e[horse_id_col].astype("string")
    e["_race_date_dt"] = _to_datetime(e[date_col])
    if trainer_col:
        e["_trainer_code"] = e[trainer_col].astype("string")
    w = workouts.copy()
    w["_horse_id"] = w["horse_id"].astype("string")
    w["_workout_date_dt"] = _to_datetime(w["workout_date"])

    merged = e.merge(w, how="left", on="_horse_id")
    days = (merged["_race_date_dt"] - merged["_workout_date_dt"]).dt.days
    selected = merged[(days >= 0) & (days <= lookback_days)].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["race_id"] = selected["_race_id"]
    selected["race_date"] = selected["_race_date_dt"].dt.strftime("%Y%m%d")
    selected["horse_id"] = selected["_horse_id"]
    if trainer_col:
        selected["trainer_code"] = selected["_trainer_code"]
    return selected.drop(columns=[c for c in selected.columns if c.startswith("_")], errors="ignore")


def _to_datetime(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    text = text.where(~text.str.fullmatch(r"\d{6}", na=False), "20" + text)
    return pd.to_datetime(text, errors="coerce", format="mixed")


def _first_existing(frame: pd.DataFrame, candidates: list[str], *, required: bool = True) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    if required:
        raise ValueError(f"None of these columns exist: {candidates}")
    return None


if __name__ == "__main__":
    main()
