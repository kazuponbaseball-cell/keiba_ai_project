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


DEFAULT_OUT = Path("outputs/analysis/kyoto_special_logic")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def z(values: pd.Series, race: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    mean = values.groupby(race).transform("mean")
    std = values.groupby(race).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_kyoto_components(df: pd.DataFrame, base_score_col: str) -> pd.DataFrame:
    out = df.copy()
    race = out[RACE_COL]
    frame = num(out, "枠番", 0.0).fillna(0.0)
    horse_no = num(out, "馬番", 0.0).fillna(0.0)
    field = num(out, "頭数", 0.0).replace(0, np.nan)
    outer_pos = (horse_no / field).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    surface = out["芝・ダ"].astype(str) if "芝・ダ" in out.columns else pd.Series("", index=out.index)
    turf = surface.str.contains("芝", regex=False).astype(float)
    dirt = surface.str.contains("ダ", regex=False).astype(float)
    distance = num(out, "距離", 0.0).fillna(0.0)
    sprint = distance.le(1400).astype(float)
    mile_mid = distance.between(1500, 2000).astype(float)
    long = distance.ge(2200).astype(float)

    front = num(out, "horse_front_run_rate_past5", 0.0).fillna(num(out, "front_running_tendency", 0.0)).fillna(0.0)
    stalker = num(out, "horse_stalker_rate_past5", 0.0).fillna(0.0)
    closer = num(out, "horse_closer_rate_past5", 0.0).fillna(num(out, "closing_tendency", 0.0)).fillna(0.0)
    can_rate = num(out, "horse_can_rate_rate", 0.0).fillna(front + stalker).clip(0.0, 1.0)
    collapse = num(out, "race_pace_collapse_risk", 0.0).fillna(0.0).clip(0.0, 1.0)
    slow = num(out, "race_slow_pace_risk", 0.0).fillna(0.0).clip(0.0, 1.0)
    sustain = num(out, "horse_sustain_lap_score_past5", 0.0).fillna(0.0)
    long_spurt = num(out, "horse_long_spurt_lap_score_past5", 0.0).fillna(0.0)
    instant = num(out, "horse_instant_lap_score_past5", 0.0).fillna(0.0)
    expected_fit = num(out, "expected_lap_total_fit_score", 0.0).fillna(0.0)
    rpci_fit = num(out, "horse_expected_rpci_fit_score", 0.0).fillna(0.0)

    inner = frame.le(3).astype(float)
    middle = frame.between(4, 6).astype(float)
    outer = frame.ge(7).astype(float)
    out["kyoto_inner_front_fit"] = z((inner + 0.4 * middle) * can_rate * (0.45 + 0.55 * slow), race)
    out["kyoto_outer_closer_fit"] = z((outer + 0.35 * middle) * closer * collapse, race)
    out["kyoto_sustain_downhill_fit"] = z(
        turf
        * (0.34 * sustain + 0.30 * long_spurt + 0.18 * instant + 0.18 * expected_fit)
        * (0.55 + 0.45 * mile_mid + 0.35 * long),
        race,
    )
    out["kyoto_dirt_position_fit"] = z(
        dirt
        * (0.45 * can_rate + 0.20 * front + 0.20 * num(out, "draw_pace_fit_score", 0.0).fillna(0.0) + 0.15 * slow)
        * (0.55 + 0.45 * sprint + 0.25 * mile_mid),
        race,
    )
    out["kyoto_draw_bias_fit"] = z(
        0.25 * num(out, "current_draw_advantage_score", 0.0).fillna(0.0)
        + 0.25 * num(out, "draw_pace_fit_score", 0.0).fillna(0.0)
        + 0.18 * num(out, "same_day_frame_bias_fit_score", 0.0).fillna(0.0)
        + 0.18 * num(out, "same_day_pace_bias_fit_score", 0.0).fillna(0.0)
        + 0.14 * num(out, "same_day_pop_adjusted_pace_fit_score", 0.0).fillna(0.0),
        race,
    )
    out["kyoto_lap_fit"] = z(0.55 * expected_fit + 0.30 * rpci_fit + 0.15 * sustain, race)
    out["kyoto_member_form_fit"] = z(
        0.25 * num(out, "confirmed_member_level_adjusted_score", 0.0).fillna(0.0)
        + 0.20 * num(out, "past3_avg_score", 0.0).fillna(0.0)
        + 0.20 * num(out, "prev_class_time_value_score", 0.0).fillna(0.0)
        + 0.15 * num(out, "bias_adjusted_recent_score", 0.0).fillna(0.0)
        + 0.10 * num(out, "jockey_venue_avg_score", 0.0).fillna(0.0)
        + 0.10 * num(out, "trainer_venue_avg_score", 0.0).fillna(0.0),
        race,
    )
    out["kyoto_outer_loss_risk"] = z(outer_pos * can_rate * slow - outer_pos * closer * collapse, race)
    out["kyoto_front_collapse_risk"] = z(front * collapse * (0.6 + 0.4 * turf), race)

    front_position = (1.0 - num(out, "prev_corner4_position_rate", 0.5).fillna(0.5)).clip(0.0, 1.0)
    early_move = num(out, "horse_early_move_avg_past5", 0.0).fillna(0.0).clip(lower=-2.0, upper=4.0)
    late_gain = num(out, "horse_late_gain_avg_past5", 0.0).fillna(0.0).clip(lower=-3.0, upper=6.0)
    final3_rank = num(out, "past3_avg_final3f_rank", 0.0).fillna(0.0)
    final3_score = (1.0 - final3_rank / num(out, "頭数", 16.0).replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)

    out["kyoto_inner_position_fit"] = z(
        turf
        * (inner + 0.55 * middle)
        * (0.38 * front_position + 0.28 * can_rate + 0.18 * slow + 0.16 * num(out, "draw_pace_fit_score", 0.0).fillna(0.0)),
        race,
    )
    out["kyoto_inner_mobility_fit"] = z(
        turf
        * (inner + 0.45 * middle)
        * (0.36 * ((early_move + 2.0) / 6.0) + 0.26 * sustain + 0.22 * long_spurt + 0.16 * expected_fit),
        race,
    )
    out["kyoto_inner_front_risk"] = z(turf * front * collapse * (outer + 0.4 * middle), race)
    out["kyoto_outer_late_sustain_fit"] = z(
        turf
        * (0.30 * ((late_gain + 3.0) / 9.0) + 0.25 * sustain + 0.22 * long_spurt + 0.13 * final3_score + 0.10 * expected_fit),
        race,
    )
    out["kyoto_outer_closer_flow_fit"] = z(
        turf * (0.48 * closer + 0.22 * num(out, "horse_midpack_rate_past5", 0.0).fillna(0.0)) * (0.55 * collapse + 0.45 * rpci_fit),
        race,
    )
    out["kyoto_outer_instant_slow_fit"] = z(turf * (0.52 * instant + 0.22 * final3_score + 0.26 * rpci_fit) * slow, race)
    out["kyoto_outer_front_overtrust_risk"] = z(turf * front * (0.55 * collapse + 0.25 * outer + 0.20 * num(out, "same_day_front_collapse_index", 0.0).fillna(0.0)), race)

    out["kyoto_base_rank"] = out.groupby(RACE_COL)[base_score_col].rank(ascending=False, method="first").astype(int)
    out["kyoto_popularity"] = num(out, "人気", np.nan)
    out["kyoto_pop_rank"] = out.groupby(RACE_COL)["kyoto_popularity"].rank(ascending=True, method="first")
    out["kyoto_market_gap"] = z(out["kyoto_pop_rank"] - out["kyoto_base_rank"], race)
    out["kyoto_favorite_penalty"] = z(
        out["kyoto_base_rank"].eq(1).astype(float)
        * (out["kyoto_popularity"].le(1).astype(float) + num(out, "単勝オッズ", 99.0).lt(2.2).astype(float)),
        race,
    ).clip(lower=0.0)
    return out


def score_with_weights(df: pd.DataFrame, base_col: str, weights: dict[str, float]) -> pd.Series:
    score = df[base_col].copy()
    for key, weight in weights.items():
        if key == "favorite_penalty":
            score = score - weight * df["kyoto_favorite_penalty"]
        else:
            score = score + weight * df[f"kyoto_{key}"]
    return score


def candidate_weights() -> list[dict[str, float]]:
    rows = []
    for sustain in [0.0, 0.03, 0.05]:
        for draw in [0.0, 0.03]:
            for dirt_pos in [0.0, 0.04]:
                for inner_front in [0.0, 0.03]:
                    for outer_closer in [0.0, 0.03]:
                        for member in [0.0, 0.03]:
                            for market in [0.0]:
                                rows.append(
                                    {
                                        "sustain_downhill_fit": sustain,
                                        "draw_bias_fit": draw,
                                        "dirt_position_fit": dirt_pos,
                                        "inner_front_fit": inner_front,
                                        "outer_closer_fit": outer_closer,
                                        "member_form_fit": member,
                                        "market_gap": market,
                                        "outer_loss_risk": 0.0,
                                        "front_collapse_risk": 0.0,
                                        "favorite_penalty": 0.0,
                                    }
                                )
    return rows


def candidate_layout_weights(layout: str) -> list[dict[str, float]]:
    if layout == "turf_inner":
        keys = {
            "inner_position_fit": [0.0, 0.03, 0.06],
            "inner_mobility_fit": [0.0, 0.03, 0.05],
            "draw_bias_fit": [0.0, 0.02],
            "member_form_fit": [0.0, 0.02],
            "inner_front_risk": [0.0, 0.02],
        }
        rows = []
        for pos in keys["inner_position_fit"]:
            for mob in keys["inner_mobility_fit"]:
                for draw in keys["draw_bias_fit"]:
                    for member in keys["member_form_fit"]:
                        for risk in keys["inner_front_risk"]:
                            rows.append(
                                {
                                    "inner_position_fit": pos,
                                    "inner_mobility_fit": mob,
                                    "draw_bias_fit": draw,
                                    "member_form_fit": member,
                                    "inner_front_risk": risk,
                                }
                            )
        return rows
    if layout == "turf_outer":
        rows = []
        for late in [0.0, 0.03, 0.06]:
            for flow in [0.0, 0.03, 0.05]:
                for instant_slow in [0.0, 0.02, 0.04]:
                    for draw in [0.0, 0.02]:
                        for risk in [0.0, 0.02]:
                            rows.append(
                                {
                                    "outer_late_sustain_fit": late,
                                    "outer_closer_flow_fit": flow,
                                    "outer_instant_slow_fit": instant_slow,
                                    "draw_bias_fit": draw,
                                    "outer_front_overtrust_risk": risk,
                                }
                            )
        return rows
    return [
        {"dirt_position_fit": dirt, "outer_closer_fit": outer, "member_form_fit": member}
        for dirt in [0.0, 0.03, 0.05]
        for outer in [0.0, 0.03]
        for member in [0.0, 0.03]
    ]


def score_with_layout_weights(df: pd.DataFrame, base_col: str, weights: dict[str, float]) -> pd.Series:
    score = df[base_col].copy()
    risk_keys = {"inner_front_risk", "outer_front_overtrust_risk", "favorite_penalty"}
    for key, weight in weights.items():
        col = f"kyoto_{key}"
        if col not in df.columns:
            continue
        if key in risk_keys:
            score = score - weight * df[col]
        else:
            score = score + weight * df[col]
    return score


def selection_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["top1_win_roi"] * 1.00
        + metrics["top1_place_roi"] * 0.38
        + metrics["top1_win_rate"] * 0.45
        + metrics["top1_top3_rate"] * 0.18
        - metrics["winner_mean_ai_rank"] * 0.012
    )


def topn_segment_metrics(scored: pd.DataFrame, score_col: str, group_col: str) -> pd.DataFrame:
    out = scored.copy()
    out["rank"] = out.groupby(RACE_COL)[score_col].rank(ascending=False, method="first").astype(int)
    rows = []
    for name, part in out.groupby(group_col, dropna=False, sort=True):
        if part[RACE_COL].nunique() < 20:
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


def add_kyoto_layout_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    surface = out["芝・ダ"].astype(str) if "芝・ダ" in out.columns else pd.Series("", index=out.index)
    track_code = num(out, "トラックコード", np.nan)
    out["kyoto_layout_group"] = np.select(
        [
            surface.str.contains("ダ", regex=False, na=False),
            surface.str.contains("芝", regex=False, na=False) & track_code.eq(8),
            surface.str.contains("芝", regex=False, na=False) & track_code.eq(0),
        ],
        ["dirt", "turf_outer", "turf_inner"],
        default="unknown",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Kyoto-specific score logic.")
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

    kyoto_train = train_x[train_x["場所"].astype(str).eq("京都")].copy()
    kyoto_test = test_x[test_x["場所"].astype(str).eq("京都")].copy()
    kyoto_train = add_kyoto_components(kyoto_train, "expected_lap_score")
    kyoto_test = add_kyoto_components(kyoto_test, "expected_lap_score")
    kyoto_train = add_kyoto_layout_group(kyoto_train)
    kyoto_test = add_kyoto_layout_group(kyoto_test)

    grid_rows = []
    for weights in candidate_weights():
        score = score_with_weights(kyoto_train, "expected_lap_score", weights)
        metrics = metric_summary(kyoto_train, score.to_numpy())
        grid_rows.append(
            {
                "weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                "selection_score": selection_score(metrics),
                **metrics,
            }
        )
    grid = pd.DataFrame(grid_rows).sort_values("selection_score", ascending=False)
    best_weights = json.loads(grid.iloc[0]["weights"])

    kyoto_test["kyoto_special_score"] = score_with_weights(kyoto_test, "expected_lap_score", best_weights)
    kyoto_test["kyoto_surface_score"] = kyoto_test["expected_lap_score"]
    kyoto_test["kyoto_layout_score"] = kyoto_test["expected_lap_score"]
    surface_best_rows = []
    for surface_name, train_part in kyoto_train.groupby("芝・ダ", dropna=False, sort=True):
        test_mask = kyoto_test["芝・ダ"].astype(str).eq(str(surface_name))
        if train_part[RACE_COL].nunique() < 80 or test_mask.sum() == 0:
            continue
        surface_rows = []
        for weights in candidate_weights():
            score = score_with_weights(train_part, "expected_lap_score", weights)
            metrics = metric_summary(train_part, score.to_numpy())
            surface_rows.append(
                {
                    "surface": surface_name,
                    "weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                    "selection_score": selection_score(metrics),
                    **metrics,
                }
            )
        surface_grid = pd.DataFrame(surface_rows).sort_values("selection_score", ascending=False)
        best_surface_weights = json.loads(surface_grid.iloc[0]["weights"])
        kyoto_test.loc[test_mask, "kyoto_surface_score"] = score_with_weights(
            kyoto_test.loc[test_mask], "expected_lap_score", best_surface_weights
        )
        surface_best_rows.append(surface_grid.iloc[0].to_dict())

    layout_best_rows = []
    for layout_name, train_part in kyoto_train.groupby("kyoto_layout_group", dropna=False, sort=True):
        test_mask = kyoto_test["kyoto_layout_group"].astype(str).eq(str(layout_name))
        if train_part[RACE_COL].nunique() < 40 or test_mask.sum() == 0:
            continue
        layout_rows = []
        for weights in candidate_layout_weights(str(layout_name)):
            score = score_with_layout_weights(train_part, "expected_lap_score", weights)
            metrics = metric_summary(train_part, score.to_numpy())
            layout_rows.append(
                {
                    "layout": layout_name,
                    "weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                    "selection_score": selection_score(metrics),
                    **metrics,
                }
            )
        layout_grid = pd.DataFrame(layout_rows).sort_values("selection_score", ascending=False)
        best_layout_weights = json.loads(layout_grid.iloc[0]["weights"])
        kyoto_test.loc[test_mask, "kyoto_layout_score"] = score_with_layout_weights(
            kyoto_test.loc[test_mask], "expected_lap_score", best_layout_weights
        )
        layout_best_rows.append(layout_grid.iloc[0].to_dict())

    summary = pd.DataFrame(
        [
            {"variant": "base", **metric_summary(kyoto_test, kyoto_test["base_score"].to_numpy())},
            {"variant": "expected_lap", **metric_summary(kyoto_test, kyoto_test["expected_lap_score"].to_numpy())},
            {"variant": "kyoto_special_overlay", **metric_summary(kyoto_test, kyoto_test["kyoto_special_score"].to_numpy())},
            {"variant": "kyoto_surface_overlay", **metric_summary(kyoto_test, kyoto_test["kyoto_surface_score"].to_numpy())},
            {"variant": "kyoto_layout_overlay", **metric_summary(kyoto_test, kyoto_test["kyoto_layout_score"].to_numpy())},
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

    kyoto_test["surface_group"] = kyoto_test["芝・ダ"].astype(str)
    surface = topn_segment_metrics(kyoto_test, "kyoto_special_score", "surface_group")
    layout = topn_segment_metrics(kyoto_test, "kyoto_special_score", "kyoto_layout_group")
    kyoto_test["distance_group"] = pd.cut(
        num(kyoto_test, "距離", 0.0),
        bins=[0, 1400, 1800, 2200, 10000],
        labels=["<=1400", "1500-1800", "1900-2200", "2300+"],
    ).astype(str)
    distance = topn_segment_metrics(kyoto_test, "kyoto_special_score", "distance_group")

    grid.head(100).to_csv(out_dir / "kyoto_overlay_grid_top100.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(surface_best_rows).to_csv(out_dir / "kyoto_surface_best_weights.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(layout_best_rows).to_csv(out_dir / "kyoto_layout_best_weights.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "kyoto_special_summary.csv", index=False, encoding="utf-8-sig")
    surface.to_csv(out_dir / "kyoto_special_by_surface.csv", index=False, encoding="utf-8-sig")
    layout.to_csv(out_dir / "kyoto_special_by_layout.csv", index=False, encoding="utf-8-sig")
    layout_overlay = topn_segment_metrics(kyoto_test, "kyoto_layout_score", "kyoto_layout_group")
    layout_overlay.to_csv(out_dir / "kyoto_layout_overlay_by_layout.csv", index=False, encoding="utf-8-sig")
    distance.to_csv(out_dir / "kyoto_special_by_distance.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "kyoto_special_best_weights.json").open("w", encoding="utf-8") as f:
        json.dump(best_weights, f, ensure_ascii=False, indent=2)

    show = summary.copy()
    pct_cols = [c for c in show.columns if c.endswith("_rate") or c.endswith("_roi") or c.startswith("delta_")]
    show[pct_cols] = show[pct_cols] * 100.0
    print("Best weights")
    print(json.dumps(best_weights, ensure_ascii=False, indent=2))
    print("\nSummary")
    print(show.to_string(index=False))
    for label, frame in [("Surface", surface), ("Layout", layout), ("Distance", distance)]:
        tmp = frame.copy()
        pct = [c for c in tmp.columns if c.endswith("_rate") or c.endswith("_roi")]
        tmp[pct] = tmp[pct] * 100.0
        print(f"\n{label}")
        print(tmp.to_string(index=False))
    tmp = layout_overlay.copy()
    pct = [c for c in tmp.columns if c.endswith("_rate") or c.endswith("_roi")]
    tmp[pct] = tmp[pct] * 100.0
    print("\nLayout overlay")
    print(tmp.to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
