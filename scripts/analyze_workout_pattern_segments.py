from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze workout pattern performance segments.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/workout_segments")
    parser.add_argument("--min-starts", type=int, default=5)
    parser.add_argument("--rank-col", default=None)
    parser.add_argument("--score-col", default=None)
    parser.add_argument("--odds-col", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv, encoding="utf-8-sig", low_memory=False)
    rank_col = args.rank_col or _first_existing(frame, ["確定着順", "遒ｺ螳夂捩鬆・", "rank", "finish_rank"])
    score_col = args.score_col or _first_existing(frame, ["target_score"], required=False)
    odds_col = args.odds_col or _first_existing(frame, ["単勝オッズ", "odds", "win_odds"], required=False)
    frame = _prepare(frame, rank_col=rank_col, score_col=score_col, odds_col=odds_col)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "race_course_pattern": ["race_course_key", "workout_latest_pattern_bucket"],
        "race_course_lap_group": ["race_course_key", "workout_latest_lap_group"],
        "trainer_pattern": ["trainer_key", "workout_latest_pattern_bucket"],
        "trainer_lap_group": ["trainer_key", "workout_latest_lap_group"],
        "horse_pattern": ["horse_key", "workout_latest_pattern_bucket"],
        "horse_lap_group": ["horse_key", "workout_latest_lap_group"],
    }

    outputs: dict[str, str] = {}
    for name, group_cols in specs.items():
        available = [col for col in group_cols if col in frame.columns]
        if len(available) != len(group_cols):
            continue
        summary = _segment_summary(frame, group_cols, min_starts=args.min_starts)
        path = output_dir / f"workout_segment_{name}.csv"
        summary.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[name] = str(path)

    print(json.dumps({"rows": int(len(frame)), "rank_col": rank_col, "outputs": outputs}, ensure_ascii=False, indent=2))


def _prepare(frame: pd.DataFrame, *, rank_col: str, score_col: str | None, odds_col: str | None) -> pd.DataFrame:
    out = frame.copy()
    out["_rank"] = pd.to_numeric(out[rank_col], errors="coerce")
    out["_win"] = (out["_rank"] == 1).astype(float)
    out["_top3"] = (out["_rank"] <= 3).astype(float)
    if score_col and score_col in out.columns:
        out["_score"] = pd.to_numeric(out[score_col], errors="coerce")
    else:
        field_size = out.groupby(_race_col(out))["_rank"].transform("max").replace(0, np.nan)
        out["_score"] = ((field_size + 1.0 - out["_rank"]) / field_size).replace([np.inf, -np.inf], np.nan)
    if odds_col and odds_col in out.columns:
        out["_win_odds"] = pd.to_numeric(out[odds_col], errors="coerce")
        out["_win_return"] = np.where(out["_rank"] == 1, out["_win_odds"], 0.0)
    else:
        out["_win_odds"] = np.nan
        out["_win_return"] = np.nan

    out["race_course_key"] = _race_course_key(out)
    out["trainer_key"] = _first_series(out, ["trainer_code", "調教師コード", "隱ｿ謨吝ｸｫ繧ｳ繝ｼ繝・"])
    out["horse_key"] = _first_series(out, ["horse_id", "血統登録番号", "陦邨ｱ逋ｻ骭ｲ逡ｪ蜿ｷ"])
    return out


def _segment_summary(frame: pd.DataFrame, group_cols: list[str], *, min_starts: int) -> pd.DataFrame:
    valid = frame.dropna(subset=group_cols + ["_rank"]).copy()
    grouped = valid.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        starts=("_rank", "size"),
        win_rate=("_win", "mean"),
        top3_rate=("_top3", "mean"),
        avg_score=("_score", "mean"),
        avg_win_odds=("_win_odds", "mean"),
        win_roi=("_win_return", "mean"),
    ).reset_index()
    summary = summary[summary["starts"] >= min_starts].copy()

    base_win = valid["_win"].mean()
    base_top3 = valid["_top3"].mean()
    base_score = valid["_score"].mean()
    summary["win_lift"] = summary["win_rate"] - base_win
    summary["top3_lift"] = summary["top3_rate"] - base_top3
    summary["score_lift"] = summary["avg_score"] - base_score
    summary["segment_value_score"] = (
        summary["score_lift"].fillna(0.0) * 100.0
        + summary["top3_lift"].fillna(0.0) * 30.0
        + np.log1p(summary["starts"]) * 0.5
    )
    return summary.sort_values(["segment_value_score", "starts"], ascending=[False, False])


def _race_col(frame: pd.DataFrame) -> str:
    return _first_existing(frame, ["race_id", "レースID(新/馬番無)", "繝ｬ繝ｼ繧ｹID(譁ｰ/鬥ｬ逡ｪ辟｡)"])


def _race_course_key(frame: pd.DataFrame) -> pd.Series:
    existing = _first_existing(frame, ["race_course_key"], required=False)
    if existing:
        return frame[existing].astype("string")
    venue = _first_series(frame, ["場所", "蝣ｴ謇", "venue"])
    surface = _first_series(frame, ["芝・ダ", "闃昴・繝", "surface"])
    distance = _first_series(frame, ["距離", "distance"])
    return venue.astype("string").fillna("") + "_" + surface.astype("string").fillna("") + "_" + distance.astype("string").fillna("")


def _first_existing(frame: pd.DataFrame, candidates: list[str], *, required: bool = True) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    if required:
        raise ValueError(f"None of these columns exist: {candidates}")
    return None


def _first_series(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    col = _first_existing(frame, candidates, required=False)
    if col:
        return frame[col]
    return pd.Series(pd.NA, index=frame.index, dtype="string")


if __name__ == "__main__":
    main()
