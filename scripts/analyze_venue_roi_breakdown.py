from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_expected_lap_rpci_features import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    NEW_CATEGORICAL_FEATURES,
    NEW_NUMERIC_FEATURES,
    RACE_COL,
    add_expected_lap_features,
    fit_ranker,
    metric_summary,
    num,
)
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


DEFAULT_OUT = Path("outputs/analysis/venue_roi_breakdown")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def score_frame(df: pd.DataFrame, scores: np.ndarray, score_name: str) -> pd.DataFrame:
    out = df.copy()
    out[score_name] = scores
    out[f"{score_name}_rank"] = out.groupby(RACE_COL)[score_name].rank(ascending=False, method="first").astype(int)
    return out


def topn_metrics(scored: pd.DataFrame, rank_col: str, n: int) -> dict[str, Any]:
    frame = scored[scored[rank_col].le(n)]
    if len(frame) == 0:
        return {
            f"top{n}_bets": 0,
            f"top{n}_win_rate": np.nan,
            f"top{n}_top3_rate": np.nan,
            f"top{n}_win_roi": np.nan,
            f"top{n}_place_roi": np.nan,
            f"top{n}_avg_popularity": np.nan,
            f"top{n}_avg_odds": np.nan,
        }
    win_pay = num(frame, "単勝配当", 0.0).fillna(0.0).where(frame["target_win"].eq(1), 0.0)
    place_pay = num(frame, "複勝配当", 0.0).fillna(0.0).where(frame["target_top3"].eq(1), 0.0)
    return {
        f"top{n}_bets": int(len(frame)),
        f"top{n}_win_rate": float(frame["target_win"].mean()),
        f"top{n}_top3_rate": float(frame["target_top3"].mean()),
        f"top{n}_win_roi": float(win_pay.sum() / (len(frame) * 100.0)),
        f"top{n}_place_roi": float(place_pay.sum() / (len(frame) * 100.0)),
        f"top{n}_avg_popularity": float(num(frame, "人気").mean()),
        f"top{n}_avg_odds": float(num(frame, "単勝オッズ").mean()),
    }


def by_group_metrics(df: pd.DataFrame, score_col: str, group_cols: list[str]) -> pd.DataFrame:
    rank_col = f"{score_col}_rank"
    rows = []
    for keys, part in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["rows"] = int(len(part))
        row["races"] = int(part[RACE_COL].nunique())
        row.update(topn_metrics(part, rank_col, 1))
        row.update(topn_metrics(part, rank_col, 3))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["races", *group_cols], ascending=[False, *([True] * len(group_cols))])


def add_deltas(base: pd.DataFrame, plus: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    merged = base.merge(plus, on=group_cols, suffixes=("_base", "_expected_lap"))
    for col in [
        "top1_win_rate",
        "top1_top3_rate",
        "top1_win_roi",
        "top1_place_roi",
        "top3_win_rate",
        "top3_top3_rate",
        "top3_win_roi",
        "top3_place_roi",
    ]:
        if f"{col}_base" in merged.columns and f"{col}_expected_lap" in merged.columns:
            merged[f"delta_{col}"] = merged[f"{col}_expected_lap"] - merged[f"{col}_base"]
    return merged


def pct_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col.endswith("_roi") or col.startswith("delta_"):
            out[col] = pd.to_numeric(out[col], errors="coerce") * 100.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Venue-level ROI breakdown for base and expected-lap models.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(project_path(args.train_csv), low_memory=False)
    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    base_model: SimpleRaceRanker = pickle.load(project_path(args.base_model).open("rb"))
    train_x, test_x, _ = add_expected_lap_features(train, test)

    base_numeric = list(base_model.numeric_features)
    base_categorical = list(base_model.categorical_features)
    plus_numeric = base_numeric + [col for col in NEW_NUMERIC_FEATURES if col not in base_numeric]
    plus_categorical = base_categorical + [col for col in NEW_CATEGORICAL_FEATURES if col not in base_categorical]
    plus_model = fit_ranker(
        train_x,
        plus_numeric,
        plus_categorical,
        float(base_model.ridge_alpha),
        int(base_model.categorical_top_k),
    )

    scored = score_frame(test_x, base_model.predict(test_x), "base_score")
    scored = score_frame(scored, plus_model.predict(test_x), "expected_lap_score")

    venue_base = by_group_metrics(scored, "base_score", ["場所"])
    venue_plus = by_group_metrics(scored, "expected_lap_score", ["場所"])
    venue_delta = add_deltas(venue_base, venue_plus, ["場所"])

    venue_surface_base = by_group_metrics(scored, "base_score", ["場所", "芝・ダ"])
    venue_surface_plus = by_group_metrics(scored, "expected_lap_score", ["場所", "芝・ダ"])
    venue_surface_delta = add_deltas(venue_surface_base, venue_surface_plus, ["場所", "芝・ダ"])

    venue_base.to_csv(out_dir / "venue_base_metrics.csv", index=False, encoding="utf-8-sig")
    venue_plus.to_csv(out_dir / "venue_expected_lap_metrics.csv", index=False, encoding="utf-8-sig")
    venue_delta.to_csv(out_dir / "venue_delta_metrics.csv", index=False, encoding="utf-8-sig")
    venue_surface_delta.to_csv(out_dir / "venue_surface_delta_metrics.csv", index=False, encoding="utf-8-sig")

    overall = pd.DataFrame(
        [
            {"variant": "base", **metric_summary(scored, scored["base_score"].to_numpy())},
            {"variant": "expected_lap", **metric_summary(scored, scored["expected_lap_score"].to_numpy())},
        ]
    )
    overall.to_csv(out_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")

    show_cols = [
        "場所",
        "races_base",
        "top1_win_rate_base",
        "top1_top3_rate_base",
        "top1_win_roi_base",
        "top1_place_roi_base",
        "top1_win_rate_expected_lap",
        "top1_top3_rate_expected_lap",
        "top1_win_roi_expected_lap",
        "top1_place_roi_expected_lap",
        "delta_top1_win_roi",
        "delta_top1_place_roi",
        "top3_win_roi_base",
        "top3_place_roi_base",
        "top3_win_roi_expected_lap",
        "top3_place_roi_expected_lap",
    ]
    display = pct_cols(venue_delta[show_cols].copy())
    print("Overall")
    print(pct_cols(overall).to_string(index=False))
    print("\nVenue")
    print(display.to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
