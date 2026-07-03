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


DEFAULT_OUT = Path("outputs/analysis/sapporo_special_logic")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def z(values: pd.Series, race: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    mean = values.groupby(race).transform("mean")
    std = values.groupby(race).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_sapporo_components(df: pd.DataFrame, base_score_col: str) -> pd.DataFrame:
    out = df.copy()
    race = out[RACE_COL]
    surface = out["芝・ダ"].astype(str)
    turf = surface.str.contains("芝", regex=False, na=False).astype(float)
    dirt = surface.str.contains("ダ", regex=False, na=False).astype(float)
    distance = num(out, "距離", 0.0).fillna(0.0)
    sprint = distance.le(1200).astype(float)
    route = distance.between(1500, 2000).astype(float)
    long = distance.ge(2400).astype(float)
    dirt1700 = (dirt.eq(1.0) & distance.eq(1700)).astype(float)
    dirt1000 = (dirt.eq(1.0) & distance.eq(1000)).astype(float)

    frame = num(out, "枠番", 0.0).fillna(0.0)
    horse_no = num(out, "馬番", 0.0).fillna(0.0)
    field = num(out, "頭数", 0.0).replace(0, np.nan)
    outer_pos = (horse_no / field).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    inner = frame.le(3).astype(float)
    middle = frame.between(4, 6).astype(float)
    outer = frame.ge(7).astype(float)

    front = num(out, "horse_front_run_rate_past5", 0.0).fillna(num(out, "front_running_tendency", 0.0)).fillna(0.0)
    stalker = num(out, "horse_stalker_rate_past5", 0.0).fillna(0.0)
    midpack = num(out, "horse_midpack_rate_past5", 0.0).fillna(0.0)
    closer = num(out, "horse_closer_rate_past5", 0.0).fillna(num(out, "closing_tendency", 0.0)).fillna(0.0)
    can_rate = num(out, "horse_can_rate_rate", 0.0).fillna(front + stalker).clip(0.0, 1.0)
    collapse = num(out, "race_pace_collapse_risk", 0.0).fillna(0.0).clip(0.0, 1.0)
    slow = num(out, "race_slow_pace_risk", 0.0).fillna(0.0).clip(0.0, 1.0)
    sustain = num(out, "horse_sustain_lap_score_past5", 0.0).fillna(0.0)
    long_spurt = num(out, "horse_long_spurt_lap_score_past5", 0.0).fillna(0.0)
    instant = num(out, "horse_instant_lap_score_past5", 0.0).fillna(0.0)
    expected_fit = num(out, "expected_lap_total_fit_score", 0.0).fillna(0.0)
    rpci_fit = num(out, "horse_expected_rpci_fit_score", 0.0).fillna(0.0)
    body = num(out, "body_prev_weight", np.nan).fillna(num(out, "前走馬体重", np.nan)).fillna(470.0)
    body_power = ((body - 440.0) / 80.0).clip(-0.5, 1.0)

    venue_fit = (
        0.18 * num(out, "same_venue_avg_score", 0.0).fillna(0.0)
        + 0.14 * num(out, "same_venue_top3_rate", 0.0).fillna(0.0)
        + 0.13 * num(out, "jockey_venue_avg_score", 0.0).fillna(0.0)
        + 0.12 * num(out, "trainer_venue_avg_score", 0.0).fillna(0.0)
        + 0.12 * num(out, "sire_venue_avg_score", 0.0).fillna(0.0)
        + 0.10 * num(out, "bms_venue_avg_score", 0.0).fillna(0.0)
        + 0.08 * num(out, "owner_venue_top3_rate", 0.0).fillna(0.0)
        + 0.08 * num(out, "breeder_venue_top3_rate", 0.0).fillna(0.0)
        + 0.05 * num(out, "bloodline_lift_fit_score", 0.0).fillna(0.0)
    )
    out["sapporo_venue_specialist_fit"] = z(venue_fit, race)
    out["sapporo_turf_power_stay_fit"] = z(
        turf
        * (0.24 * venue_fit + 0.22 * sustain + 0.18 * long_spurt + 0.16 * body_power + 0.12 * rpci_fit + 0.08 * expected_fit)
        * (0.55 + 0.45 * route + 0.30 * long),
        race,
    )
    out["sapporo_turf_1500_fit"] = z(
        turf
        * distance.eq(1500).astype(float)
        * (0.26 * can_rate + 0.22 * venue_fit + 0.18 * sustain + 0.14 * middle + 0.12 * slow + 0.08 * instant),
        race,
    )
    out["sapporo_turf_sprint_fit"] = z(
        turf
        * sprint
        * (0.30 * can_rate + 0.20 * front + 0.16 * num(out, "draw_pace_fit_score", 0.0).fillna(0.0) + 0.14 * expected_fit + 0.12 * venue_fit + 0.08 * body_power),
        race,
    )
    out["sapporo_dirt1700_position_fit"] = z(
        dirt1700
        * (0.30 * can_rate + 0.22 * stalker + 0.16 * (inner + 0.55 * middle) + 0.14 * venue_fit + 0.10 * body_power + 0.08 * slow),
        race,
    )
    out["sapporo_dirt1000_speed_fit"] = z(
        dirt1000
        * (0.34 * front + 0.24 * can_rate + 0.18 * num(out, "front_pressure_rank_score", 0.0).fillna(0.0) + 0.14 * inner + 0.10 * body_power),
        race,
    )
    out["sapporo_outer_loss_risk"] = z((outer_pos * front * slow + outer * can_rate * slow) * (turf + 0.7 * dirt), race)
    out["sapporo_front_collapse_risk"] = z(front * collapse * (0.65 * turf + 0.50 * dirt), race)
    out["sapporo_late_stuck_risk"] = z(closer * slow * (0.70 * turf + 0.55 * dirt), race)
    out["sapporo_bias_fit"] = z(
        0.22 * num(out, "same_day_frame_bias_fit_score", 0.0).fillna(0.0)
        + 0.22 * num(out, "same_day_pace_bias_fit_score", 0.0).fillna(0.0)
        + 0.18 * num(out, "draw_pace_fit_score", 0.0).fillna(0.0)
        + 0.16 * num(out, "current_draw_advantage_score", 0.0).fillna(0.0)
        + 0.12 * num(out, "bias_adjusted_recent_score", 0.0).fillna(0.0)
        + 0.10 * num(out, "same_day_pop_adjusted_pace_fit_score", 0.0).fillna(0.0),
        race,
    )
    out["sapporo_member_form_fit"] = z(
        0.24 * num(out, "confirmed_member_level_adjusted_score", 0.0).fillna(0.0)
        + 0.20 * num(out, "past3_avg_score", 0.0).fillna(0.0)
        + 0.18 * num(out, "prev_class_time_value_score", 0.0).fillna(0.0)
        + 0.14 * num(out, "same_distance_category_avg_score", 0.0).fillna(0.0)
        + 0.12 * venue_fit
        + 0.12 * num(out, "rotation_fit_score", 0.0).fillna(0.0),
        race,
    )

    out["sapporo_base_rank"] = out.groupby(RACE_COL)[base_score_col].rank(ascending=False, method="first").astype(int)
    out["sapporo_popularity"] = num(out, "人気", np.nan)
    out["sapporo_pop_rank"] = out.groupby(RACE_COL)["sapporo_popularity"].rank(ascending=True, method="first")
    out["sapporo_market_gap"] = z(out["sapporo_pop_rank"] - out["sapporo_base_rank"], race)
    return out


def score_with_weights(df: pd.DataFrame, base_col: str, weights: dict[str, float]) -> pd.Series:
    score = df[base_col].copy()
    risk_keys = {"outer_loss_risk", "front_collapse_risk", "late_stuck_risk"}
    for key, weight in weights.items():
        col = f"sapporo_{key}"
        if col not in df.columns:
            continue
        if key in risk_keys:
            score = score - weight * df[col]
        else:
            score = score + weight * df[col]
    return score


def candidate_weights(segment: str) -> list[dict[str, float]]:
    if segment == "turf":
        return [
            {
                "turf_power_stay_fit": power,
                "turf_1500_fit": fit1500,
                "turf_sprint_fit": sprint,
                "venue_specialist_fit": venue,
                "bias_fit": bias,
                "late_stuck_risk": stuck,
            }
            for power in [0.0, 0.03, 0.05]
            for fit1500 in [0.0, 0.03]
            for sprint in [0.0, 0.03]
            for venue in [0.0, 0.03]
            for bias in [0.0, 0.02]
            for stuck in [0.0, 0.02]
        ]
    if segment == "dirt":
        return [
            {
                "dirt1700_position_fit": d1700,
                "dirt1000_speed_fit": d1000,
                "venue_specialist_fit": venue,
                "member_form_fit": member,
                "outer_loss_risk": outer_risk,
                "front_collapse_risk": collapse,
            }
            for d1700 in [0.0, 0.03, 0.05]
            for d1000 in [0.0, 0.03]
            for venue in [0.0, 0.03]
            for member in [0.0, 0.03]
            for outer_risk in [0.0, 0.02]
            for collapse in [0.0, 0.02]
        ]
    return [
        {"venue_specialist_fit": venue, "member_form_fit": member, "bias_fit": bias}
        for venue in [0.0, 0.03]
        for member in [0.0, 0.03]
        for bias in [0.0, 0.02]
    ]


def selection_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["top1_win_roi"]
        + 0.42 * metrics["top1_place_roi"]
        + 0.42 * metrics["top1_win_rate"]
        + 0.22 * metrics["top1_top3_rate"]
        - 0.014 * metrics["winner_mean_ai_rank"]
    )


def topn_segment_metrics(scored: pd.DataFrame, score_col: str, group_col: str) -> pd.DataFrame:
    out = scored.copy()
    out["rank"] = out.groupby(RACE_COL)[score_col].rank(ascending=False, method="first").astype(int)
    rows = []
    for name, part in out.groupby(group_col, dropna=False, sort=True):
        if part[RACE_COL].nunique() < 15:
            continue
        top1 = part[part["rank"].eq(1)]
        top3 = part[part["rank"].le(3)]
        win_pay = num(top1, "単勝配当", 0.0).where(top1["target_win"].eq(1), 0.0)
        place_pay = num(top1, "複勝配当", 0.0).where(top1["target_top3"].eq(1), 0.0)
        top3_win = num(top3, "単勝配当", 0.0).where(top3["target_win"].eq(1), 0.0)
        top3_place = num(top3, "複勝配当", 0.0).where(top3["target_top3"].eq(1), 0.0)
        rows.append(
            {
                group_col: name,
                "races": int(part[RACE_COL].nunique()),
                "top1_win_rate": float(top1["target_win"].mean()),
                "top1_top3_rate": float(top1["target_top3"].mean()),
                "top1_win_roi": float(win_pay.sum() / (len(top1) * 100.0)),
                "top1_place_roi": float(place_pay.sum() / (len(top1) * 100.0)),
                "top3_win_roi": float(top3_win.sum() / (len(top3) * 100.0)),
                "top3_place_roi": float(top3_place.sum() / (len(top3) * 100.0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Sapporo-specific score logic.")
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
    plus_numeric = list(base_model.numeric_features) + [
        col for col in NEW_NUMERIC_FEATURES if col not in base_model.numeric_features
    ]
    plus_categorical = list(base_model.categorical_features) + [
        col for col in NEW_CATEGORICAL_FEATURES if col not in base_model.categorical_features
    ]
    expected_model = fit_ranker(
        train_x,
        plus_numeric,
        plus_categorical,
        float(base_model.ridge_alpha),
        int(base_model.categorical_top_k),
    )
    train_x["base_score"] = base_model.predict(train_x)
    test_x["base_score"] = base_model.predict(test_x)
    train_x["expected_lap_score"] = expected_model.predict(train_x)
    test_x["expected_lap_score"] = expected_model.predict(test_x)

    sapporo_train = train_x[train_x["場所"].astype(str).eq("札幌")].copy()
    sapporo_test = test_x[test_x["場所"].astype(str).eq("札幌")].copy()
    sapporo_train = add_sapporo_components(sapporo_train, "expected_lap_score")
    sapporo_test = add_sapporo_components(sapporo_test, "expected_lap_score")
    sapporo_train["surface_group"] = np.where(sapporo_train["芝・ダ"].astype(str).str.contains("芝", na=False), "turf", "dirt")
    sapporo_test["surface_group"] = np.where(sapporo_test["芝・ダ"].astype(str).str.contains("芝", na=False), "turf", "dirt")
    sapporo_test["sapporo_surface_score"] = sapporo_test["expected_lap_score"]

    best_rows = []
    for segment, train_part in sapporo_train.groupby("surface_group", sort=True):
        test_mask = sapporo_test["surface_group"].eq(segment)
        rows = []
        for weights in candidate_weights(str(segment)):
            score = score_with_weights(train_part, "expected_lap_score", weights)
            metrics = metric_summary(train_part, score.to_numpy())
            rows.append(
                {
                    "segment": segment,
                    "weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                    "selection_score": selection_score(metrics),
                    **metrics,
                }
            )
        grid = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
        best = grid.iloc[0].to_dict()
        best_rows.append(best)
        best_weights = json.loads(best["weights"])
        sapporo_test.loc[test_mask, "sapporo_surface_score"] = score_with_weights(
            sapporo_test.loc[test_mask], "expected_lap_score", best_weights
        )

    summary = pd.DataFrame(
        [
            {"variant": "base", **metric_summary(sapporo_test, sapporo_test["base_score"].to_numpy())},
            {"variant": "expected_lap", **metric_summary(sapporo_test, sapporo_test["expected_lap_score"].to_numpy())},
            {"variant": "sapporo_surface_overlay", **metric_summary(sapporo_test, sapporo_test["sapporo_surface_score"].to_numpy())},
        ]
    )
    baseline = summary[summary["variant"].eq("expected_lap")].iloc[0]
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
        summary[f"delta_vs_expected_{col}"] = summary[col] - baseline[col]

    sapporo_test["distance_group"] = np.select(
        [
            num(sapporo_test, "距離").le(1200),
            num(sapporo_test, "距離").eq(1500),
            num(sapporo_test, "距離").between(1700, 1800),
            num(sapporo_test, "距離").eq(2000),
            num(sapporo_test, "距離").ge(2400),
        ],
        ["<=1200", "1500", "1700-1800", "2000", "2400+"],
        default="other",
    )
    by_surface = topn_segment_metrics(sapporo_test, "sapporo_surface_score", "surface_group")
    by_distance = topn_segment_metrics(sapporo_test, "sapporo_surface_score", "distance_group")

    pd.DataFrame(best_rows).to_csv(out_dir / "sapporo_surface_best_weights.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "sapporo_special_summary.csv", index=False, encoding="utf-8-sig")
    by_surface.to_csv(out_dir / "sapporo_special_by_surface.csv", index=False, encoding="utf-8-sig")
    by_distance.to_csv(out_dir / "sapporo_special_by_distance.csv", index=False, encoding="utf-8-sig")

    show = summary.copy()
    pct = [c for c in show.columns if c.endswith("_rate") or c.endswith("_roi") or c.startswith("delta_")]
    show[pct] = show[pct] * 100.0
    print("Best weights")
    print(pd.DataFrame(best_rows)[["segment", "weights", "selection_score", "top1_win_rate", "top1_top3_rate", "top1_win_roi", "top1_place_roi"]].to_string(index=False))
    print("\nSummary")
    print(show.to_string(index=False))
    for label, frame in [("Surface", by_surface), ("Distance", by_distance)]:
        tmp = frame.copy()
        pct = [c for c in tmp.columns if c.endswith("_rate") or c.endswith("_roi")]
        tmp[pct] = tmp[pct] * 100.0
        print(f"\n{label}")
        print(tmp.to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
