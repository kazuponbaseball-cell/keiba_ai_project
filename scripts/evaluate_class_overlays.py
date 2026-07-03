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


def class_bucket(class_name: object) -> str:
    text = str(class_name)
    if "新馬" in text:
        return "newcomer"
    if "未勝利" in text:
        return "maiden"
    if "1勝" in text:
        return "class_1win"
    if "2勝" in text:
        return "class_2win"
    if "3勝" in text:
        return "class_3win"
    if "Ｇ" in text or "G" in text:
        return "graded"
    if "OP" in text or "ｵｰﾌﾟﾝ" in text or "オープン" in text:
        return "open"
    return "other"


def add_components(df: pd.DataFrame, race_col: str) -> pd.DataFrame:
    out = df.copy()
    out["base_rank"] = out.groupby(race_col)["base_score"].rank(ascending=False, method="first").astype(int)
    out["popularity_num"] = _num(out, "人気", np.nan)
    out["odds_decimal"] = _num(out, "単勝オッズ", np.nan)
    out["pop_rank"] = out.groupby(race_col)["popularity_num"].rank(ascending=True, method="first")

    ability = (
        0.18 * _z(_num(out, "past3_avg_score"), out[race_col])
        - 0.12 * _z(_num(out, "past3_avg_margin_sec"), out[race_col])
        + 0.12 * _z(_num(out, "prev_class_time_value_score"), out[race_col])
        + 0.10 * _z(_num(out, "past3_best_time_value"), out[race_col])
        + 0.10 * _z(_num(out, "lap_aptitude_fit_score"), out[race_col])
        + 0.08 * _z(_num(out, "pace_fit_score"), out[race_col])
        + 0.06 * _z(_num(out, "same_day_bias_fit_score"), out[race_col])
        + 0.04 * _z(_num(out, "draw_pace_fit_score"), out[race_col])
    )
    out["overlay_ability"] = _z(ability, out[race_col])

    potential = (
        0.18 * _z(_num(out, "bloodline_high_confidence_fit_score"), out[race_col])
        + 0.12 * _z(_num(out, "bloodline_lift_fit_score"), out[race_col])
        + 0.16 * _z(_num(out, "workout_knowledge_grade_score"), out[race_col])
        + 0.12 * _z(_num(out, "workout_load_density_score"), out[race_col])
        - 0.10 * _z(_num(out, "workout_latest_total_vs_course_z"), out[race_col])
        - 0.08 * _z(_num(out, "workout_latest_finish_gain_sec"), out[race_col])
        + 0.08 * _z(_num(out, "trainer_win_rate"), out[race_col])
        + 0.08 * _z(_num(out, "jockey_win_rate"), out[race_col])
    )
    out["overlay_potential"] = _z(potential, out[race_col])

    opponent = (
        0.22 * _z(_num(out, "prev_race_member_level"), out[race_col])
        + 0.18 * _z(_num(out, "past3_max_race_member_level"), out[race_col])
        + 0.16 * _z(_num(out, "confirmed_member_level_adjusted_score"), out[race_col])
        + 0.12 * _z(_num(out, "prev_confirmed_opponent_good_run_score"), out[race_col])
        - 0.12 * _z(_num(out, "prev_performance_vs_member_level"), out[race_col])
    )
    out["overlay_opponent"] = _z(opponent, out[race_col])

    weight = (
        0.35 * _z(_num(out, "race_weight_light_rank_score"), out[race_col])
        - 0.20 * _z(_num(out, "斤量"), out[race_col])
        - 0.08 * _z(_num(out, "weight_diff"), out[race_col])
    )
    out["overlay_weight"] = _z(weight, out[race_col])

    market_gap = (out["pop_rank"] - out["base_rank"]).fillna(0.0)
    value = (
        0.52 * _z(market_gap, out[race_col])
        + 0.32 * _z(np.log1p(out["odds_decimal"].fillna(0.0)) * out["base_rank"].le(5), out[race_col])
        + 0.16 * _z(out["base_rank"].le(5).astype(float) * out["popularity_num"].ge(6).fillna(False).astype(float), out[race_col])
    )
    out["overlay_value"] = _z(value, out[race_col])

    favorite = (
        out["base_rank"].eq(1).astype(float)
        * (
            out["popularity_num"].le(1).fillna(False).astype(float)
            + out["odds_decimal"].lt(2.0).fillna(False).astype(float)
        )
    )
    out["overlay_favorite_penalty"] = _z(favorite, out[race_col]).clip(lower=0.0)
    return out


def overlay_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return (
        df["base_score"]
        + weights.get("ability", 0.0) * df["overlay_ability"]
        + weights.get("potential", 0.0) * df["overlay_potential"]
        + weights.get("opponent", 0.0) * df["overlay_opponent"]
        + weights.get("weight", 0.0) * df["overlay_weight"]
        + weights.get("value", 0.0) * df["overlay_value"]
        - weights.get("favorite", 0.0) * df["overlay_favorite_penalty"]
    )


def candidate_weights(bucket: str) -> list[dict[str, float]]:
    profiles: dict[str, list[dict[str, float]]] = {
        "newcomer": [
            {"potential": p, "value": v, "favorite": f}
            for p in [0.0, 0.02, 0.04, 0.06]
            for v in [0.0, 0.01, 0.03]
            for f in [0.0, 0.01]
        ],
        "maiden": [
            {"ability": a, "value": v, "favorite": f}
            for a in [0.0, 0.03, 0.05]
            for v in [0.0, 0.03, 0.05]
            for f in [0.0, 0.01, 0.03]
        ],
        "class_1win": [
            {"ability": a, "opponent": o, "value": v, "favorite": f}
            for a in [0.0, 0.02, 0.04]
            for o in [0.0, 0.02, 0.04]
            for v in [0.0, 0.02]
            for f in [0.0, 0.01]
        ],
        "class_2win": [
            {"ability": a, "opponent": o, "weight": w, "value": v}
            for a in [0.0, 0.02, 0.04]
            for o in [0.0, 0.03, 0.05]
            for w in [0.0, 0.02]
            for v in [0.0, 0.02]
        ],
        "class_3win": [
            {"ability": a, "opponent": o, "weight": w, "value": v}
            for a in [0.0, 0.02, 0.04]
            for o in [0.0, 0.03, 0.06]
            for w in [0.0, 0.02]
            for v in [0.0, 0.02]
        ],
        "open": [
            {"ability": a, "opponent": o, "value": v, "favorite": f}
            for a in [0.0, 0.02, 0.04]
            for o in [0.0, 0.04, 0.07]
            for v in [0.0, 0.02]
            for f in [0.0, 0.01]
        ],
        "graded": [
            {"ability": a, "opponent": o, "value": v, "favorite": f}
            for a in [0.0, 0.02, 0.04]
            for o in [0.0, 0.04, 0.08]
            for v in [0.0, 0.02]
            for f in [0.0, 0.01]
        ],
    }
    return profiles.get(bucket, [{}])


def temporal_masks(df: pd.DataFrame, date_col: str) -> tuple[pd.Series, pd.Series]:
    dates = pd.to_numeric(df[date_col], errors="coerce")
    unique_dates = np.array(sorted(dates.dropna().unique()))
    cutoff = unique_dates[len(unique_dates) // 2]
    return dates <= cutoff, dates > cutoff


def selection_score(metrics: dict[str, Any]) -> float:
    top3_floor_penalty = max(0.0, 0.48 - float(metrics["top1_top3_rate"])) * 0.8
    return (
        float(metrics["top1_win_roi"])
        + 0.35 * float(metrics["top1_place_roi"])
        + 0.25 * float(metrics["top1_win_rate"])
        - 0.015 * float(metrics["winner_mean_ai_rank"])
        - top3_floor_penalty
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate class-specific score overlays.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--output-dir", default="outputs/analysis/class_overlays")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    date_col = config["data"]["date_column"]
    rank_col = config["data"]["rank_column"]

    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    model: SimpleRaceRanker = pickle.load(project_path(args.model).open("rb"))
    test["base_score"] = model.predict(test)
    scored = add_components(test, race_col)
    scored["class_bucket"] = scored["クラス名"].map(class_bucket)
    scored["class_overlay_score"] = scored["base_score"]

    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for bucket, part in scored.groupby("class_bucket", sort=False):
        if part[race_col].nunique() < 50:
            continue
        discovery_mask, validation_mask = temporal_masks(part, date_col)
        bucket_rows: list[dict[str, Any]] = []
        for weights in candidate_weights(bucket):
            score = overlay_score(part, weights)
            all_m = metric_summary(part, score.to_numpy(), race_col, rank_col)
            disc_m = metric_summary(part[discovery_mask], score.loc[discovery_mask].to_numpy(), race_col, rank_col)
            valid_m = metric_summary(part[validation_mask], score.loc[validation_mask].to_numpy(), race_col, rank_col)
            row = {
                "class_bucket": bucket,
                "weights": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                **{f"all_{k}": v for k, v in all_m.items()},
                **{f"discovery_{k}": v for k, v in disc_m.items()},
                **{f"validation_{k}": v for k, v in valid_m.items()},
                "selection_score": selection_score(disc_m),
            }
            bucket_rows.append(row)
        bucket_df = pd.DataFrame(bucket_rows).sort_values(
            ["selection_score", "validation_top1_win_roi", "all_top1_win_roi"],
            ascending=[False, False, False],
        )
        rows.extend(bucket_rows)
        best = bucket_df.iloc[0].to_dict()
        best_rows.append(best)
        best_weights = json.loads(best["weights"])
        scored.loc[part.index, "class_overlay_score"] = overlay_score(part, best_weights)

    base_all = metric_summary(scored, scored["base_score"].to_numpy(), race_col, rank_col)
    overlay_all = metric_summary(scored, scored["class_overlay_score"].to_numpy(), race_col, rank_col)

    output_dir = ensure_dir(project_path(args.output_dir))
    pd.DataFrame(rows).to_csv(output_dir / "class_overlay_grid.csv", index=False, encoding="utf-8-sig")
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(output_dir / "class_overlay_best_by_bucket.csv", index=False, encoding="utf-8-sig")
    scored[
        [
            race_col,
            "日付",
            "場所",
            "Ｒ",
            "レース名",
            "クラス名",
            "class_bucket",
            "馬名",
            "確定着順",
            "人気",
            "単勝オッズ",
            "単勝配当",
            "複勝配当",
            "base_score",
            "class_overlay_score",
            "base_rank",
            "overlay_ability",
            "overlay_potential",
            "overlay_opponent",
            "overlay_weight",
            "overlay_value",
            "overlay_favorite_penalty",
        ]
    ].to_csv(output_dir / "class_overlay_runner_scores.csv", index=False, encoding="utf-8-sig")

    result = {
        "output_dir": str(output_dir),
        "base_all": base_all,
        "overlay_all": overlay_all,
        "delta_all": {key: overlay_all[key] - base_all[key] for key in base_all if isinstance(base_all[key], (int, float))},
        "best_by_bucket": best_df[
            [
                "class_bucket",
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
