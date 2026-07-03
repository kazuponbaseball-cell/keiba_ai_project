from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config  # noqa: E402
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


CONFIG = Path("config/baseline_features_workout.json")
TRAIN_CSV = Path("data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features.csv")
TEST_CSV = Path("data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features.csv")
BASE_MODEL = Path("models/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/baseline_ranker.pkl")
OUT_DIR = Path("outputs/analysis/feature_group_ablation")


GROUPS = {
    "recent_form_rotation": [
        "past3_",
        "prev_",
        "前走",
        "蜑崎",
        "キャリア",
        "繧ｭ繝｣繝ｪ繧｢",
        "間隔",
        "髢馴囈",
        "休み",
        "莨代",
        "margin",
    ],
    "member_level_class": [
        "race_member_",
        "opponent",
        "member",
        "class",
        "クラス",
        "繧ｯ繝ｩ繧ｹ",
        "next_starters",
    ],
    "time_lap_pace": [
        "time_value",
        "time_z",
        "lap",
        "pci",
        "rpci",
        "pace",
        "Ave",
        "average",
        "平均",
        "基準",
        "front_runner",
        "closer",
        "midpack",
        "stalker",
    ],
    "draw_bias": [
        "draw",
        "bias",
        "枠",
        "馬番",
        "譫",
        "鬥ｬ逡ｪ",
    ],
    "jockey_trainer_codes": [
        "騎手コード",
        "調教師コード",
        "jockey",
        "trainer",
        "鬨取焔",
        "隱ｿ謨吝ｸｫ",
    ],
    "pedigree": [
        "sire",
        "broodmare",
        "pedigree",
        "blood",
        "血統",
    ],
    "workout": [
        "workout",
        "調教",
    ],
    "condition_aptitude": [
        "distance",
        "surface",
        "venue",
        "track",
        "going",
        "距離",
        "芝・ダ",
        "馬場",
        "場所",
        "霍晞屬",
        "闃昴",
        "蝣ｴ謇",
    ],
    "horse_history_stats": [
        "horse_",
        "same_",
    ],
}


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def matches_group(feature: str, patterns: list[str]) -> bool:
    lower = feature.lower()
    return any(pattern.lower() in lower or pattern in feature for pattern in patterns)


def drop_group(features: list[str], group: str) -> list[str]:
    patterns = GROUPS[group]
    return [feature for feature in features if not matches_group(feature, patterns)]


def metric_summary(df: pd.DataFrame, scores: np.ndarray, race_col: str, rank_col: str) -> dict[str, Any]:
    scored = df.copy()
    scored["ai_score"] = scores
    scored["ai_rank"] = scored.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    top1 = scored[scored["ai_rank"] == 1]
    top3 = scored[scored["ai_rank"] <= 3]

    win_pay_col = "単勝配当" if "単勝配当" in scored.columns else "蜊伜享驟榊ｽ・"
    place_pay_col = "複勝配当" if "複勝配当" in scored.columns else "隍・享驟榊ｽ・"
    win_pay = _num(top1.get(win_pay_col, pd.Series(0, index=top1.index))).fillna(0).where(top1["target_win"] == 1, 0.0)
    place_pay = _num(top1.get(place_pay_col, pd.Series(0, index=top1.index))).fillna(0).where(top1["target_top3"] == 1, 0.0)
    top3_win_pay = _num(top3.get(win_pay_col, pd.Series(0, index=top3.index))).fillna(0).where(top3["target_win"] == 1, 0.0)
    top3_place_pay = _num(top3.get(place_pay_col, pd.Series(0, index=top3.index))).fillna(0).where(top3["target_top3"] == 1, 0.0)

    return {
        "rows": int(len(scored)),
        "races": int(scored[race_col].nunique()),
        "top1_win_rate": float(top1["target_win"].mean()),
        "top1_top3_rate": float(top1["target_top3"].mean()),
        "top1_win_roi": float(win_pay.sum() / (len(top1) * 100.0)),
        "top1_place_roi": float(place_pay.sum() / (len(top1) * 100.0)),
        "top3_contains_winner_rate": float(top3.groupby(race_col)["target_win"].max().mean()),
        "top3_win_roi": float(top3_win_pay.sum() / (len(top3) * 100.0)),
        "top3_place_roi": float(top3_place_pay.sum() / (len(top3) * 100.0)),
        "winner_mean_ai_rank": float(scored.loc[scored[rank_col] == 1, "ai_rank"].mean()),
    }


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
    race_col = config["data"]["race_id_column"]
    rank_col = config["data"]["rank_column"]
    return model, metric_summary(test, model.predict(test), race_col, rank_col)


def coefficient_importance(model: SimpleRaceRanker, label: str) -> pd.DataFrame:
    rows = []
    feature_names = model.feature_names_ or []
    coefficients = model.coefficients_ if model.coefficients_ is not None else []
    for feature, coef in zip(feature_names, coefficients):
        if feature == "intercept":
            group = "intercept"
        else:
            matched = [name for name in GROUPS if matches_group(feature, GROUPS[name])]
            group = matched[0] if matched else "other"
        rows.append(
            {
                "model": label,
                "feature": feature,
                "coef": float(coef),
                "abs_coef": abs(float(coef)),
                "group": group,
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
    importances = []

    base_retrained, base_metrics = fit_eval(train, test, base_numeric, base_categorical, config)
    rows.append(
        {
            "variant": "all_features_retrained",
            "removed_group": "",
            "numeric_features": len(base_numeric),
            "categorical_features": len(base_categorical),
            **base_metrics,
        }
    )
    importances.append(coefficient_importance(base_retrained, "all_features_retrained"))

    original_metrics = metric_summary(
        test,
        base_model.predict(test),
        config["data"]["race_id_column"],
        config["data"]["rank_column"],
    )
    rows.append(
        {
            "variant": "saved_model_reference",
            "removed_group": "",
            "numeric_features": len(base_numeric),
            "categorical_features": len(base_categorical),
            **original_metrics,
        }
    )
    importances.append(coefficient_importance(base_model, "saved_model_reference"))

    for group in GROUPS:
        numeric = drop_group(base_numeric, group)
        categorical = drop_group(base_categorical, group)
        model, metrics = fit_eval(train, test, numeric, categorical, config)
        rows.append(
            {
                "variant": f"drop_{group}",
                "removed_group": group,
                "numeric_features": len(numeric),
                "categorical_features": len(categorical),
                "removed_numeric": len(base_numeric) - len(numeric),
                "removed_categorical": len(base_categorical) - len(categorical),
                **metrics,
            }
        )
        importances.append(coefficient_importance(model, f"drop_{group}"))

    summary = pd.DataFrame(rows)
    baseline = summary[summary["variant"] == "all_features_retrained"].iloc[0]
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
        summary[f"delta_{col}"] = summary[col] - baseline[col]

    summary.to_csv(OUT_DIR / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(importances, ignore_index=True).to_csv(OUT_DIR / "coefficient_importance_by_variant.csv", index=False, encoding="utf-8-sig")

    group_rows = []
    for group, patterns in GROUPS.items():
        removed_num = [feature for feature in base_numeric if matches_group(feature, patterns)]
        removed_cat = [feature for feature in base_categorical if matches_group(feature, patterns)]
        group_rows.append(
            {
                "group": group,
                "numeric_features": len(removed_num),
                "categorical_features": len(removed_cat),
                "examples": "; ".join((removed_num + removed_cat)[:20]),
            }
        )
    pd.DataFrame(group_rows).to_csv(OUT_DIR / "feature_group_inventory.csv", index=False, encoding="utf-8-sig")

    print(json.dumps({"output_dir": str(OUT_DIR), "variants": len(summary)}, ensure_ascii=False, indent=2))
    print(summary.sort_values("delta_top1_win_roi")[[
        "variant",
        "removed_group",
        "top1_win_rate",
        "top1_top3_rate",
        "top1_win_roi",
        "top1_place_roi",
        "top3_contains_winner_rate",
        "delta_top1_win_roi",
        "delta_top1_win_rate",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
