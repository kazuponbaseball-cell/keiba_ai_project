from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ablate_feature_groups import metric_summary  # noqa: E402
from src.data.loaders import load_json_config  # noqa: E402
from src.features.baseline import TRACK_CONDITION_NUMERIC_FEATURES, add_track_condition_features  # noqa: E402
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


DEFAULT_TRAIN = "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features_with_same_day_bias_v3_retro.csv"
DEFAULT_TEST = "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv"
DEFAULT_MODEL = "models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl"


def fit_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict[str, Any],
    label: str,
) -> tuple[SimpleRaceRanker, dict[str, Any]]:
    model = SimpleRaceRanker(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categorical_top_k=int(config["training"].get("categorical_top_k", 80)),
        ridge_alpha=float(config["training"].get("ridge_alpha", 10.0)),
    ).fit(train, "target_score")
    metrics = metric_summary(
        test,
        model.predict(test),
        config["data"]["race_id_column"],
        config["data"]["rank_column"],
    )
    return model, {"variant": label, "numeric_features": len(numeric_features), "categorical_features": len(categorical_features), **metrics}


def coefficient_rows(model: SimpleRaceRanker, variant: str, features: list[str]) -> list[dict[str, object]]:
    wanted = set(features)
    rows = []
    coefficients = model.coefficients_ if model.coefficients_ is not None else []
    for name, coef in zip(model.feature_names_ or [], coefficients):
        if name in wanted:
            rows.append({"variant": variant, "feature": name, "coef": float(coef), "abs_coef": abs(float(coef))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline model with and without JRA track-condition features.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN)
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--output-dir", default="outputs/analysis/track_condition_feature_eval")
    parser.add_argument("--save-enriched", action="store_true", help="Also save enriched train/test feature CSVs.")
    args = parser.parse_args()

    config = load_json_config(args.config)
    print("loading train/test...", flush=True)
    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)
    print("adding track-condition features...", flush=True)
    train_tc = add_track_condition_features(train, config)
    test_tc = add_track_condition_features(test, config)

    base_model: SimpleRaceRanker = pickle.load(Path(args.base_model).open("rb"))
    base_numeric = list(base_model.numeric_features)
    base_categorical = list(base_model.categorical_features)
    track_numeric = [feature for feature in TRACK_CONDITION_NUMERIC_FEATURES if feature in train_tc.columns]
    plus_numeric = list(dict.fromkeys([*base_numeric, *track_numeric]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_enriched:
        train_tc.to_csv(output_dir / "train_features_with_track_condition.csv", index=False, encoding="utf-8-sig")
        test_tc.to_csv(output_dir / "test_features_with_track_condition.csv", index=False, encoding="utf-8-sig")

    rows = []
    coef_rows = []
    print("fitting baseline retrain...", flush=True)
    model, metrics = fit_eval(train_tc, test_tc, base_numeric, base_categorical, config, "baseline_retrained")
    rows.append(metrics)
    coef_rows.extend(coefficient_rows(model, "baseline_retrained", track_numeric))

    variants = {
        "plus_track_condition": track_numeric,
        "track_fit_only": [
            "horse_cushion_fit_score",
            "horse_moisture_fit_score",
            "horse_track_condition_fit_score",
        ],
        "track_horse_history_only": [
            feature for feature in track_numeric if feature.startswith("horse_")
        ],
        "track_race_context_only": [
            feature for feature in track_numeric if feature.startswith("track_") or feature.startswith("race_")
        ],
        "track_flags_plus_fit": [
            "track_condition_available",
            "race_high_cushion_flag",
            "race_low_cushion_flag",
            "race_wet_moisture_flag",
            "race_dry_moisture_flag",
            "horse_cushion_fit_score",
            "horse_moisture_fit_score",
            "horse_track_condition_fit_score",
        ],
    }
    for variant, extra_features in variants.items():
        print(f"fitting {variant}...", flush=True)
        numeric = list(dict.fromkeys([*base_numeric, *extra_features]))
        model_tc, metrics_tc = fit_eval(train_tc, test_tc, numeric, base_categorical, config, variant)
        rows.append(metrics_tc)
        coef_rows.extend(coefficient_rows(model_tc, variant, track_numeric))

    summary = pd.DataFrame(rows)
    base = summary.loc[summary["variant"].eq("baseline_retrained")].iloc[0]
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
        summary[f"delta_{col}"] = summary[col] - base[col]

    coverage = {
        "train_rows": int(len(train_tc)),
        "test_rows": int(len(test_tc)),
        "train_available_rate": float(train_tc["track_condition_available"].mean()),
        "test_available_rate": float(test_tc["track_condition_available"].mean()),
        "train_race_available_rate": float(train_tc.groupby(config["data"]["race_id_column"])["track_condition_available"].max().mean()),
        "test_race_available_rate": float(test_tc.groupby(config["data"]["race_id_column"])["track_condition_available"].max().mean()),
        "track_numeric_features": track_numeric,
    }
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coef_rows).sort_values(["variant", "abs_coef"], ascending=[True, False]).to_csv(
        output_dir / "track_condition_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    with (output_dir / "coverage.json").open("w", encoding="utf-8") as f:
        json.dump(coverage, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_dir": str(output_dir), **coverage}, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
