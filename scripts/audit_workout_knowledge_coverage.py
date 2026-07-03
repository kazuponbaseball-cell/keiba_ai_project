from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.workout_knowledge import TRAINER_NAMES


DEFAULT_INPUTS = [
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/train_features.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/test_features.csv",
]


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of these columns exist: {list(candidates)}")


def to_num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def load_inputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, low_memory=False)
        df["_source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("No input files were provided.")
    return pd.concat(frames, ignore_index=True, sort=False)


def build_coverage(df: pd.DataFrame) -> pd.DataFrame:
    trainer_col = first_existing_column(df, ["調教師コード", "trainer_code"])
    rank_col = first_existing_column(df, ["確定着順", "finish_position", "rank"])

    work_count = to_num(df.get("workout_count", pd.Series(0, index=df.index)))
    pattern_avg = to_num(df.get("workout_trainer_pattern_avg_score", pd.Series(0, index=df.index)))
    lap_avg = to_num(df.get("workout_trainer_lap_avg_score", pd.Series(0, index=df.index)))
    finish = to_num(df[rank_col], default=999.0)

    tmp = pd.DataFrame(
        {
            "trainer_code": to_num(df[trainer_col], default=-1).astype("Int64"),
            "workout_present": work_count.gt(0.0),
            "trainer_pattern_avg": pattern_avg,
            "trainer_lap_avg": lap_avg,
            "win": finish.eq(1.0),
            "top3": finish.between(1.0, 3.0),
        }
    )
    tmp = tmp[tmp["trainer_code"].ge(0)].copy()
    registered_codes = {int(code) for code in TRAINER_NAMES.keys()}
    tmp["registered"] = tmp["trainer_code"].astype(int).isin(registered_codes)

    grouped = (
        tmp.groupby(["trainer_code", "registered"], dropna=False)
        .agg(
            rows=("trainer_code", "size"),
            workout_rows=("workout_present", "sum"),
            trainer_pattern_avg=("trainer_pattern_avg", "mean"),
            trainer_lap_avg=("trainer_lap_avg", "mean"),
            win_rate=("win", "mean"),
            top3_rate=("top3", "mean"),
        )
        .reset_index()
    )
    grouped["workout_row_pct"] = grouped["workout_rows"] / grouped["rows"] * 100.0
    grouped["priority_score"] = (
        grouped["rows"].rank(pct=True)
        + grouped["workout_row_pct"].fillna(0.0).rank(pct=True)
        + grouped["top3_rate"].fillna(0.0).rank(pct=True)
    ) / 3.0
    return grouped.sort_values(["registered", "rows"], ascending=[True, False]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit stable-specific workout knowledge coverage.")
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", default="outputs/analysis/current_strongest_runtime_v1")
    args = parser.parse_args()

    df = load_inputs(args.inputs)
    coverage = build_coverage(df)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = out_dir / "trainer_workout_knowledge_coverage.csv"
    unregistered_path = out_dir / "trainer_workout_knowledge_unregistered_priority.csv"
    summary_path = out_dir / "trainer_workout_knowledge_coverage_summary.json"

    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    unregistered = coverage[~coverage["registered"]].sort_values(
        ["priority_score", "rows"], ascending=[False, False]
    )
    unregistered.to_csv(unregistered_path, index=False, encoding="utf-8-sig")

    total_rows = int(coverage["rows"].sum())
    registered_rows = int(coverage.loc[coverage["registered"], "rows"].sum())
    workout_rows = int(coverage["workout_rows"].sum())
    summary = {
        "input_rows": int(len(df)),
        "trainers": int(len(coverage)),
        "registered_trainers": int(coverage["registered"].sum()),
        "unregistered_trainers": int((~coverage["registered"]).sum()),
        "registered_row_pct": round(registered_rows / total_rows * 100.0, 1) if total_rows else 0.0,
        "unregistered_row_pct": round((total_rows - registered_rows) / total_rows * 100.0, 1) if total_rows else 0.0,
        "workout_row_pct": round(workout_rows / total_rows * 100.0, 1) if total_rows else 0.0,
        "coverage_csv": str(coverage_path),
        "unregistered_priority_csv": str(unregistered_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
