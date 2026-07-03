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


DEFAULT_OUT = Path("outputs/analysis/corner_accel_rpci")
HORSE_COL = "血統登録番号"
DATE_COL = "日付"

CORNER_ACCEL_FEATURES = [
    "horse_corner_gain_3to4_avg_past5",
    "horse_corner_gain_3to4_goodrun_past5",
    "horse_fast_rpci_corner_gain_score_past5",
    "horse_late_speed_corner_gain_score_past5",
    "horse_long_spurt_corner_gain_score_past5",
    "small_course_corner_accel_fit_score",
    "small_course_fast_rpci_accel_fit_score",
    "small_course_late_speed_accel_fit_score",
    "corner_accel_rpci_composite_score",
]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def race_z(values: pd.Series, race_ids: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    mean = values.groupby(race_ids).transform("mean")
    std = values.groupby(race_ids).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_corner_accel_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["_split"] = "train"
    test["_split"] = "test"
    train["_row_id"] = np.arange(len(train))
    test["_row_id"] = np.arange(len(test))
    all_df = pd.concat([train, test], ignore_index=True, sort=False)

    for col in CORNER_ACCEL_FEATURES:
        all_df[col] = 0.0

    c3 = num(all_df, "3角", np.nan)
    c4 = num(all_df, "4角", np.nan).fillna(num(all_df, "4角.1", np.nan))
    field = num(all_df, "頭数", np.nan).replace(0, np.nan)
    corner_gain = ((c3 - c4) / field).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.60, 0.60)
    rpci = num(all_df, "RPCI", np.nan)
    pci3 = num(all_df, "PCI3", np.nan)
    # Low RPCI = pace/load tends to be stronger; PCI3-RPCI high = long-spurt/late-speed shape.
    fast_rpci_pressure = ((50.0 - rpci) / 6.0).clip(0.0, 1.0).fillna(0.0)
    late_speed_context = ((rpci - 50.0) / 6.0).clip(0.0, 1.0).fillna(0.0)
    long_spurt_context = ((pci3 - rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)
    target_score = num(all_df, "target_score", 0.0).fillna(0.0).clip(lower=0.0)
    positive_gain = corner_gain.clip(lower=0.0)

    all_df["_corner_gain"] = corner_gain
    all_df["_corner_gain_goodrun"] = positive_gain * target_score
    all_df["_fast_rpci_corner_gain"] = positive_gain * fast_rpci_pressure * target_score
    all_df["_late_speed_corner_gain"] = positive_gain * late_speed_context * target_score
    all_df["_long_spurt_corner_gain"] = positive_gain * long_spurt_context * target_score

    ordered = all_df.sort_values([HORSE_COL, DATE_COL, RACE_COL], kind="mergesort")
    specs = [
        ("_corner_gain", "horse_corner_gain_3to4_avg_past5"),
        ("_corner_gain_goodrun", "horse_corner_gain_3to4_goodrun_past5"),
        ("_fast_rpci_corner_gain", "horse_fast_rpci_corner_gain_score_past5"),
        ("_late_speed_corner_gain", "horse_late_speed_corner_gain_score_past5"),
        ("_long_spurt_corner_gain", "horse_long_spurt_corner_gain_score_past5"),
    ]
    for source, dest in specs:
        values = pd.to_numeric(ordered[source], errors="coerce").fillna(0.0)
        rolled = values.groupby(ordered[HORSE_COL], sort=False).transform(
            lambda s: s.shift().rolling(5, min_periods=1).mean()
        )
        all_df.loc[ordered.index, dest] = pd.to_numeric(rolled, errors="coerce").fillna(0.0).astype(float)

    venue = all_df["場所"].astype(str)
    surface = all_df["芝・ダ"].astype(str)
    track_code = num(all_df, "トラックコード", np.nan)
    distance = num(all_df, "距離", 0.0)
    small_course = (
        venue.isin(["札幌", "函館", "福島", "小倉", "中山"])
        | (venue.eq("京都") & surface.str.contains("芝", regex=False, na=False) & track_code.eq(0))
        | (venue.eq("阪神") & surface.str.contains("芝", regex=False, na=False) & distance.between(1800, 2200))
    ).astype(float)
    route_small = small_course * distance.ge(1500).astype(float)
    all_df["small_course_corner_accel_fit_score"] = route_small * race_z(all_df["horse_corner_gain_3to4_goodrun_past5"], all_df[RACE_COL])
    all_df["small_course_fast_rpci_accel_fit_score"] = route_small * race_z(all_df["horse_fast_rpci_corner_gain_score_past5"], all_df[RACE_COL])
    all_df["small_course_late_speed_accel_fit_score"] = route_small * race_z(all_df["horse_late_speed_corner_gain_score_past5"], all_df[RACE_COL])
    all_df["corner_accel_rpci_composite_score"] = (
        0.35 * all_df["small_course_corner_accel_fit_score"]
        + 0.30 * all_df["small_course_fast_rpci_accel_fit_score"]
        + 0.20 * all_df["small_course_late_speed_accel_fit_score"]
        + 0.15 * route_small * race_z(all_df["horse_long_spurt_corner_gain_score_past5"], all_df[RACE_COL])
    ).fillna(0.0)

    all_df = all_df.drop(
        columns=[
            "_corner_gain",
            "_corner_gain_goodrun",
            "_fast_rpci_corner_gain",
            "_late_speed_corner_gain",
            "_long_spurt_corner_gain",
        ],
        errors="ignore",
    )
    train_out = all_df[all_df["_split"].eq("train")].sort_values("_row_id", kind="mergesort").drop(columns=["_split", "_row_id"])
    test_out = all_df[all_df["_split"].eq("test")].sort_values("_row_id", kind="mergesort").drop(columns=["_split", "_row_id"])
    return train_out, test_out


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
        "composite_hi": float(num(train, "corner_accel_rpci_composite_score").quantile(0.75)),
        "fast_hi": float(num(train, "small_course_fast_rpci_accel_fit_score").quantile(0.75)),
        "late_hi": float(num(train, "small_course_late_speed_accel_fit_score").quantile(0.75)),
    }
    venue = out["場所"].astype(str)
    checks = [
        ("top1_all", out["rank"].eq(1)),
        ("top1_composite_hi", out["rank"].eq(1) & num(out, "corner_accel_rpci_composite_score").ge(q["composite_hi"])),
        ("top1_fast_rpci_accel_hi", out["rank"].eq(1) & num(out, "small_course_fast_rpci_accel_fit_score").ge(q["fast_hi"])),
        ("top1_late_speed_accel_hi", out["rank"].eq(1) & num(out, "small_course_late_speed_accel_fit_score").ge(q["late_hi"])),
        ("top3_pop5plus_composite_hi", out["rank"].le(3) & num(out, "人気").ge(5) & num(out, "corner_accel_rpci_composite_score").ge(q["composite_hi"])),
        ("sapporo_top1_composite_hi", venue.eq("札幌") & out["rank"].eq(1) & num(out, "corner_accel_rpci_composite_score").ge(q["composite_hi"])),
        ("hakodate_top1_composite_hi", venue.eq("函館") & out["rank"].eq(1) & num(out, "corner_accel_rpci_composite_score").ge(q["composite_hi"])),
        ("fukushima_top1_composite_hi", venue.eq("福島") & out["rank"].eq(1) & num(out, "corner_accel_rpci_composite_score").ge(q["composite_hi"])),
        ("kokura_top1_composite_hi", venue.eq("小倉") & out["rank"].eq(1) & num(out, "corner_accel_rpci_composite_score").ge(q["composite_hi"])),
        ("nakayama_top1_composite_hi", venue.eq("中山") & out["rank"].eq(1) & num(out, "corner_accel_rpci_composite_score").ge(q["composite_hi"])),
    ]
    rows = []
    for name, mask in checks:
        rows.append({"segment": name, **bet_metrics(out[mask])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3C-to-4C acceleration adjusted by RPCI/lap context.")
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
    train_x, test_x = add_corner_accel_features(train_x, test_x)

    base_numeric = list(base_model.numeric_features) + [col for col in NEW_NUMERIC_FEATURES if col not in base_model.numeric_features]
    base_categorical = list(base_model.categorical_features) + [
        col for col in NEW_CATEGORICAL_FEATURES if col not in base_model.categorical_features
    ]
    plus_numeric = base_numeric + [col for col in CORNER_ACCEL_FEATURES if col not in base_numeric]

    base_ranker = fit_ranker(train_x, base_numeric, base_categorical, float(base_model.ridge_alpha), int(base_model.categorical_top_k))
    plus_ranker = fit_ranker(train_x, plus_numeric, base_categorical, float(base_model.ridge_alpha), int(base_model.categorical_top_k))
    base_scores = base_ranker.predict(test_x)
    plus_scores = plus_ranker.predict(test_x)

    summary = pd.DataFrame(
        [
            {"variant": "expected_lap_base", **metric_summary(test_x, base_scores)},
            {"variant": "corner_accel_rpci_features", **metric_summary(test_x, plus_scores)},
        ]
    )
    segs = segment_report(train_x, test_x, plus_scores)
    imps = pd.DataFrame(
        {
            "feature": plus_ranker.feature_names_,
            "coef": plus_ranker.coefficients_,
            "abs_coef": np.abs(plus_ranker.coefficients_),
        }
    ).sort_values("abs_coef", ascending=False)

    summary.to_csv(out_dir / "corner_accel_rpci_summary.csv", index=False, encoding="utf-8-sig")
    segs.to_csv(out_dir / "corner_accel_rpci_segments.csv", index=False, encoding="utf-8-sig")
    imps.to_csv(out_dir / "corner_accel_rpci_importance.csv", index=False, encoding="utf-8-sig")

    show = summary.copy()
    pct = [c for c in show.columns if c.endswith("_rate") or c.endswith("_roi")]
    show[pct] = show[pct] * 100.0
    print("Summary")
    print(show.to_string(index=False))
    show_seg = segs.copy()
    pct = [c for c in show_seg.columns if c.endswith("_rate") or c.endswith("_roi")]
    show_seg[pct] = show_seg[pct] * 100.0
    print("\nSegments")
    print(show_seg.to_string(index=False))
    print("\nCorner accel coefficients")
    print(imps[imps["feature"].isin(CORNER_ACCEL_FEATURES)].to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
