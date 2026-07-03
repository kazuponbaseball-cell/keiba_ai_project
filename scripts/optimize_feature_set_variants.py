from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ablate_feature_groups import GROUPS, drop_group, matches_group, metric_summary  # noqa: E402
from src.data.loaders import load_json_config  # noqa: E402
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


CONFIG = Path("config/baseline_features_workout.json")
TRAIN_CSV = Path("data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features.csv")
TEST_CSV = Path("data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features.csv")
BASE_MODEL = Path("models/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/baseline_ranker.pkl")
OUT_DIR = Path("outputs/analysis/feature_set_optimization")


VARIANTS: dict[str, list[str]] = {
    "all_features": [],
    "drop_condition": ["condition_aptitude"],
    "drop_condition_jt": ["condition_aptitude", "jockey_trainer_codes"],
    "drop_condition_pedigree": ["condition_aptitude", "pedigree"],
    "drop_condition_workout": ["condition_aptitude", "workout"],
    "drop_condition_jt_pedigree": ["condition_aptitude", "jockey_trainer_codes", "pedigree"],
    "drop_condition_jt_workout": ["condition_aptitude", "jockey_trainer_codes", "workout"],
    "drop_condition_pedigree_workout": ["condition_aptitude", "pedigree", "workout"],
    "drop_jt_pedigree_workout": ["jockey_trainer_codes", "pedigree", "workout"],
    "drop_condition_jt_pedigree_workout": ["condition_aptitude", "jockey_trainer_codes", "pedigree", "workout"],
    "core_no_side_layers": ["condition_aptitude", "jockey_trainer_codes", "pedigree", "workout"],
    "core_plus_workout": ["condition_aptitude", "jockey_trainer_codes", "pedigree"],
    "core_plus_jt": ["condition_aptitude", "pedigree", "workout"],
    "core_plus_pedigree": ["condition_aptitude", "jockey_trainer_codes", "workout"],
}


def remove_groups(features: list[str], groups: list[str]) -> list[str]:
    kept = list(features)
    for group in groups:
        kept = drop_group(kept, group)
    return kept


def removed_examples(base: list[str], kept: list[str], limit: int = 30) -> str:
    kept_set = set(kept)
    return "; ".join([feature for feature in base if feature not in kept_set][:limit])


def fit_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict[str, Any],
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
    return model, metrics


def model_importance(model: SimpleRaceRanker, label: str) -> pd.DataFrame:
    rows = []
    coefficients = model.coefficients_ if model.coefficients_ is not None else []
    for feature, coef in zip(model.feature_names_ or [], coefficients):
        matched = [group for group, patterns in GROUPS.items() if matches_group(feature, patterns)]
        rows.append(
            {
                "variant": label,
                "feature": feature,
                "coef": float(coef),
                "abs_coef": abs(float(coef)),
                "group": matched[0] if matched else "other",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_json_config(CONFIG)
    train = pd.read_csv(TRAIN_CSV, low_memory=False)
    test = pd.read_csv(TEST_CSV, low_memory=False)
    base_model: SimpleRaceRanker = pickle.load(BASE_MODEL.open("rb"))
    base_numeric = list(base_model.numeric_features)
    base_categorical = list(base_model.categorical_features)

    rows = []
    imps = []
    models_dir = OUT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    for name, groups in VARIANTS.items():
        numeric = remove_groups(base_numeric, groups)
        categorical = remove_groups(base_categorical, groups)
        model, metrics = fit_eval(train, test, numeric, categorical, config)
        with (models_dir / f"{name}.pkl").open("wb") as f:
            pickle.dump(model, f)
        rows.append(
            {
                "variant": name,
                "removed_groups": ",".join(groups),
                "numeric_features": len(numeric),
                "categorical_features": len(categorical),
                "removed_numeric": len(base_numeric) - len(numeric),
                "removed_categorical": len(base_categorical) - len(categorical),
                "removed_numeric_examples": removed_examples(base_numeric, numeric),
                "removed_categorical_examples": removed_examples(base_categorical, categorical),
                **metrics,
            }
        )
        imps.append(model_importance(model, name))

    summary = pd.DataFrame(rows)
    baseline = summary[summary["variant"] == "all_features"].iloc[0]
    metric_cols = [
        "top1_win_rate",
        "top1_top3_rate",
        "top1_win_roi",
        "top1_place_roi",
        "top3_contains_winner_rate",
        "top3_win_roi",
        "top3_place_roi",
        "winner_mean_ai_rank",
    ]
    for col in metric_cols:
        summary[f"delta_{col}"] = summary[col] - baseline[col]
    summary["balanced_score"] = (
        summary["top1_win_rate"] * 100
        + summary["top1_top3_rate"] * 25
        + summary["top1_win_roi"] * 20
        + summary["top1_place_roi"] * 10
        + summary["top3_contains_winner_rate"] * 15
        - summary["winner_mean_ai_rank"] * 0.8
    )
    summary = summary.sort_values("balanced_score", ascending=False)
    summary.to_csv(OUT_DIR / "optimization_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(imps, ignore_index=True).to_csv(OUT_DIR / "optimization_importance.csv", index=False, encoding="utf-8-sig")

    best = summary.iloc[0].to_dict()
    with (OUT_DIR / "best_variant.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_dir": str(OUT_DIR), "best_variant": best["variant"]}, ensure_ascii=False, indent=2))
    print(
        summary[
            [
                "variant",
                "removed_groups",
                "numeric_features",
                "categorical_features",
                "top1_win_rate",
                "top1_top3_rate",
                "top1_win_roi",
                "top1_place_roi",
                "top3_contains_winner_rate",
                "top3_win_roi",
                "top3_place_roi",
                "balanced_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
