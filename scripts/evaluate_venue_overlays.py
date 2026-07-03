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

from scripts.ablate_feature_groups import metric_summary  # noqa: E402
from src.data.loaders import load_json_config  # noqa: E402
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402
from src.utils.paths import ensure_dir, project_path  # noqa: E402


DEFAULT_TEST = "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv"
DEFAULT_MODEL = "models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl"


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    series = df[col]
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _z(values: pd.Series, race: pd.Series) -> pd.Series:
    mean = values.groupby(race).transform("mean")
    std = values.groupby(race).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_components(df: pd.DataFrame, race_col: str) -> pd.DataFrame:
    out = df.copy()
    out["base_rank"] = out.groupby(race_col)["base_score"].rank(ascending=False, method="first").astype(int)
    out["popularity_num"] = _num(out, "人気", np.nan)
    out["odds_decimal"] = _num(out, "単勝オッズ", np.nan)
    out["pop_rank"] = out.groupby(race_col)["popularity_num"].rank(ascending=True, method="first")

    ability = (
        0.20 * _z(_num(out, "past3_avg_score"), out[race_col])
        - 0.12 * _z(_num(out, "past3_avg_margin_sec"), out[race_col])
        + 0.14 * _z(_num(out, "prev_class_time_value_score"), out[race_col])
        + 0.12 * _z(_num(out, "past3_best_time_value"), out[race_col])
        + 0.12 * _z(_num(out, "prev_race_member_level"), out[race_col])
        + 0.10 * _z(_num(out, "confirmed_member_level_adjusted_score"), out[race_col])
        + 0.08 * _z(_num(out, "lap_aptitude_fit_score"), out[race_col])
        + 0.06 * _z(_num(out, "pace_fit_score"), out[race_col])
    )
    out["venue_overlay_ability"] = _z(ability, out[race_col])

    venue_fit = (
        0.18 * _z(_num(out, "same_venue_avg_score"), out[race_col])
        + 0.14 * _z(_num(out, "same_venue_top3_rate"), out[race_col])
        + 0.12 * _z(_num(out, "jockey_venue_avg_score"), out[race_col])
        + 0.10 * _z(_num(out, "trainer_venue_avg_score"), out[race_col])
        + 0.10 * _z(_num(out, "sire_venue_avg_score"), out[race_col])
        + 0.08 * _z(_num(out, "bms_venue_avg_score"), out[race_col])
        + 0.08 * _z(_num(out, "bloodline_surface_distance_fit_score"), out[race_col])
        + 0.08 * _z(_num(out, "same_distance_category_avg_score"), out[race_col])
        + 0.06 * _z(_num(out, "jockey_venue_popularity_outperform_rate"), out[race_col])
        + 0.06 * _z(_num(out, "trainer_venue_popularity_outperform_rate"), out[race_col])
    )
    out["venue_overlay_fit"] = _z(venue_fit, out[race_col])

    bias_draw = (
        0.18 * _z(_num(out, "current_draw_advantage_score"), out[race_col])
        + 0.14 * _z(_num(out, "draw_pace_fit_score"), out[race_col])
        + 0.12 * _z(_num(out, "frame_bias_score"), out[race_col])
        + 0.10 * _z(_num(out, "horse_number_bias_score"), out[race_col])
        + 0.12 * _z(_num(out, "same_day_frame_bias_fit_score"), out[race_col])
        + 0.12 * _z(_num(out, "same_day_pace_bias_fit_score"), out[race_col])
        + 0.10 * _z(_num(out, "same_day_bias_fit_score"), out[race_col])
        + 0.08 * _z(_num(out, "bias_adjusted_recent_score"), out[race_col])
        + 0.04 * _z(_num(out, "prev_retro_bias_resistant_score"), out[race_col])
    )
    out["venue_overlay_bias_draw"] = _z(bias_draw, out[race_col])

    local_fit = (
        0.16 * _z(_num(out, "same_venue_avg_score"), out[race_col])
        + 0.14 * _z(_num(out, "sire_venue_lift"), out[race_col])
        + 0.12 * _z(_num(out, "bms_venue_lift"), out[race_col])
        + 0.12 * _z(_num(out, "jockey_venue_popularity_outperform_rate"), out[race_col])
        + 0.10 * _z(_num(out, "trainer_venue_popularity_outperform_rate"), out[race_col])
        + 0.10 * _z(_num(out, "course_condition_draw_avg_score"), out[race_col])
        + 0.08 * _z(_num(out, "horse_slow_lap_score_past5"), out[race_col])
        + 0.08 * _z(_num(out, "horse_long_spurt_lap_score_past5"), out[race_col])
        + 0.06 * _z(_num(out, "rotation_surface_switch_flag"), out[race_col])
        + 0.04 * _z(_num(out, "workout_knowledge_grade_score"), out[race_col])
    )
    out["venue_overlay_local_fit"] = _z(local_fit, out[race_col])

    market_gap = (out["pop_rank"] - out["base_rank"]).fillna(0.0)
    value = (
        0.52 * _z(market_gap, out[race_col])
        + 0.32 * _z(np.log1p(out["odds_decimal"].fillna(0.0)) * out["base_rank"].le(5), out[race_col])
        + 0.16 * _z(out["base_rank"].le(5).astype(float) * out["popularity_num"].ge(6).fillna(False).astype(float), out[race_col])
    )
    out["venue_overlay_value"] = _z(value, out[race_col])

    favorite = (
        out["base_rank"].eq(1).astype(float)
        * (
            out["popularity_num"].le(1).fillna(False).astype(float)
            + out["odds_decimal"].lt(2.0).fillna(False).astype(float)
        )
    )
    out["venue_overlay_favorite_penalty"] = _z(favorite, out[race_col]).clip(lower=0.0)
    return out


def candidate_weights(venue: str) -> list[dict[str, float]]:
    central = venue in {"東京", "中山", "阪神", "京都"}
    if venue == "東京":
        return [
            {"ability": a, "fit": f, "bias": b, "value": v, "favorite": fav}
            for a in [0.0, 0.04, 0.06]
            for f in [0.0, 0.02]
            for b in [0.0]
            for v in [0.0, 0.02]
            for fav in [0.0]
        ]
    if central:
        return [
            {"ability": a, "fit": f, "bias": b, "value": v, "favorite": fav}
            for a in [0.0, 0.04]
            for f in [0.0, 0.03]
            for b in [0.0, 0.02]
            for v in [0.0, 0.02]
            for fav in [0.0]
        ]
    return [
        {"ability": a, "fit": f, "local": l, "bias": b, "value": v, "favorite": fav}
        for a in [0.0, 0.02]
        for f in [0.0, 0.04]
        for l in [0.0, 0.05]
        for b in [0.0, 0.03]
        for v in [0.0, 0.02]
        for fav in [0.0]
    ]


def overlay_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return (
        df["base_score"]
        + weights.get("ability", 0.0) * df["venue_overlay_ability"]
        + weights.get("fit", 0.0) * df["venue_overlay_fit"]
        + weights.get("local", 0.0) * df["venue_overlay_local_fit"]
        + weights.get("bias", 0.0) * df["venue_overlay_bias_draw"]
        + weights.get("value", 0.0) * df["venue_overlay_value"]
        - weights.get("favorite", 0.0) * df["venue_overlay_favorite_penalty"]
    )


def temporal_masks(df: pd.DataFrame, date_col: str) -> tuple[pd.Series, pd.Series]:
    dates = pd.to_numeric(df[date_col], errors="coerce")
    unique_dates = np.array(sorted(dates.dropna().unique()))
    cutoff = unique_dates[len(unique_dates) // 2]
    return dates <= cutoff, dates > cutoff


def selection_score(metrics: dict[str, Any]) -> float:
    top3_floor_penalty = max(0.0, 0.47 - float(metrics["top1_top3_rate"])) * 0.8
    return (
        float(metrics["top1_win_roi"])
        + 0.35 * float(metrics["top1_place_roi"])
        + 0.25 * float(metrics["top1_win_rate"])
        - 0.015 * float(metrics["winner_mean_ai_rank"])
        - top3_floor_penalty
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate venue-specific score overlays.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--output-dir", default="outputs/analysis/venue_overlays")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    date_col = config["data"]["date_column"]
    rank_col = config["data"]["rank_column"]

    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    model: SimpleRaceRanker = pickle.load(project_path(args.model).open("rb"))
    test["base_score"] = model.predict(test)
    scored = add_components(test, race_col)
    scored["venue_overlay_score"] = scored["base_score"]

    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for venue, part in scored.groupby("場所", sort=False):
        if part[race_col].nunique() < 80:
            continue
        discovery_mask, validation_mask = temporal_masks(part, date_col)
        venue_rows: list[dict[str, Any]] = []
        for weights in candidate_weights(str(venue)):
            score = overlay_score(part, weights)
            all_m = metric_summary(part, score.to_numpy(), race_col, rank_col)
            disc_m = metric_summary(part[discovery_mask], score.loc[discovery_mask].to_numpy(), race_col, rank_col)
            valid_m = metric_summary(part[validation_mask], score.loc[validation_mask].to_numpy(), race_col, rank_col)
            row = {
                "venue": venue,
                "weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                **{f"all_{k}": v for k, v in all_m.items()},
                **{f"discovery_{k}": v for k, v in disc_m.items()},
                **{f"validation_{k}": v for k, v in valid_m.items()},
                "selection_score": selection_score(disc_m),
            }
            venue_rows.append(row)
        venue_df = pd.DataFrame(venue_rows).sort_values(
            ["selection_score", "validation_top1_win_roi", "all_top1_win_roi"],
            ascending=[False, False, False],
        )
        rows.extend(venue_rows)
        best = venue_df.iloc[0].to_dict()
        best_rows.append(best)
        scored.loc[part.index, "venue_overlay_score"] = overlay_score(part, json.loads(best["weights"]))

    base_all = metric_summary(scored, scored["base_score"].to_numpy(), race_col, rank_col)
    overlay_all = metric_summary(scored, scored["venue_overlay_score"].to_numpy(), race_col, rank_col)

    output_dir = ensure_dir(project_path(args.output_dir))
    pd.DataFrame(rows).to_csv(output_dir / "venue_overlay_grid.csv", index=False, encoding="utf-8-sig")
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(output_dir / "venue_overlay_best_by_venue.csv", index=False, encoding="utf-8-sig")
    scored[
        [
            race_col,
            "日付",
            "場所",
            "Ｒ",
            "レース名",
            "クラス名",
            "馬名",
            "確定着順",
            "人気",
            "単勝オッズ",
            "単勝配当",
            "複勝配当",
            "base_score",
            "venue_overlay_score",
            "base_rank",
            "venue_overlay_ability",
            "venue_overlay_fit",
            "venue_overlay_local_fit",
            "venue_overlay_bias_draw",
            "venue_overlay_value",
            "venue_overlay_favorite_penalty",
        ]
    ].to_csv(output_dir / "venue_overlay_runner_scores.csv", index=False, encoding="utf-8-sig")

    result = {
        "output_dir": str(output_dir),
        "base_all": base_all,
        "overlay_all": overlay_all,
        "delta_all": {key: overlay_all[key] - base_all[key] for key in base_all if isinstance(base_all[key], (int, float))},
        "best_by_venue": best_df[
            [
                "venue",
                "weights",
                "all_races",
                "all_top1_win_rate",
                "all_top1_top3_rate",
                "all_top1_win_roi",
                "all_top1_place_roi",
                "validation_top1_win_roi",
                "validation_top1_place_roi",
            ]
        ].to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
