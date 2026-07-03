from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config, model_categorical_features, model_numeric_features
from src.train.simple_ranker import SimpleRaceRanker
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _read_needed_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0, low_memory=False)
    usecols = [c for c in columns if c in header.columns]
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _softmax_by_race(score: pd.Series, race: pd.Series, temperature: float = 0.55) -> pd.Series:
    s = _num(score).fillna(0.0)
    out = pd.Series(index=s.index, dtype=float)
    for _, idx in race.groupby(race).groups.items():
        vals = s.loc[idx].astype(float)
        centered = (vals - vals.max()) / max(temperature, 1e-6)
        exp = np.exp(centered.clip(-30, 30))
        denom = exp.sum()
        out.loc[idx] = exp / denom if denom else 0.0
    return out.fillna(0.0)


def _add_prediction_columns(df: pd.DataFrame, model: SimpleRaceRanker, race_col: str) -> pd.DataFrame:
    out = df.copy()
    score = pd.Series(model.predict(out), index=out.index)
    out["quinella_model_score"] = score
    out["quinella_model_prob"] = _softmax_by_race(score, out[race_col])
    out["quinella_model_rank"] = out.groupby(race_col)["quinella_model_score"].rank(ascending=False, method="first")
    field = out.groupby(race_col)[race_col].transform("size").replace(1, np.nan)
    out["quinella_model_score_norm"] = ((field + 1.0 - out["quinella_model_rank"]) / field).clip(0.0, 1.0).fillna(0.0)
    return out


def _evaluate(df: pd.DataFrame, race_col: str, rank_col: str) -> dict:
    work = df.copy()
    work["target_quinella"] = _num(work[rank_col]).le(2).astype(float)
    top1 = work[work["quinella_model_rank"].eq(1)]
    top2 = work[work["quinella_model_rank"].le(2)]
    top3 = work[work["quinella_model_rank"].le(3)]
    return {
        "races": int(work[race_col].nunique()),
        "rows": int(len(work)),
        "top1_quinella_rate": float(top1["target_quinella"].mean()) if len(top1) else 0.0,
        "top2_contains_both_quinella_rate": float(
            top2.groupby(race_col)["target_quinella"].sum().ge(2).mean()
        )
        if len(top2)
        else 0.0,
        "top3_contains_both_quinella_rate": float(
            top3.groupby(race_col)["target_quinella"].sum().ge(2).mean()
        )
        if len(top3)
        else 0.0,
        "mean_quinella_horse_rank": float(work.loc[work["target_quinella"].eq(1), "quinella_model_rank"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dedicated top-2/quinella model and score investment features.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument(
        "--train-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features_with_same_day_bias_v3_retro.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv",
    )
    parser.add_argument(
        "--investment-scored-csv",
        default="outputs/analysis/investment_decision_features_v1/investment_features_scored.csv",
    )
    parser.add_argument("--model-dir", default="models/quinella_top2_v1")
    parser.add_argument("--output-dir", default="outputs/analysis/quinella_top2_model_v1")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    rank_col = config["data"]["rank_column"]
    numeric = model_numeric_features(config)
    categorical = model_categorical_features(config)
    needed = list(dict.fromkeys([race_col, rank_col, *numeric, *categorical]))

    train = _read_needed_csv(project_path(args.train_csv), needed)
    test = _read_needed_csv(project_path(args.test_csv), needed)
    train["target_quinella"] = _num(train[rank_col]).le(2).astype(float)
    test["target_quinella"] = _num(test[rank_col]).le(2).astype(float)

    model = SimpleRaceRanker(
        numeric_features=numeric,
        categorical_features=categorical,
        categorical_top_k=int(config["training"].get("categorical_top_k", 80)),
        ridge_alpha=float(config["training"].get("ridge_alpha", 10.0)),
    ).fit(train, "target_quinella")

    train_scored = _add_prediction_columns(train, model, race_col)
    test_scored = _add_prediction_columns(test, model, race_col)

    investment = pd.read_csv(project_path(args.investment_scored_csv), low_memory=False)
    investment_race_col = "race_id"
    investment_scored = _add_prediction_columns(investment, model, investment_race_col)

    model_dir = ensure_dir(project_path(args.model_dir))
    out_dir = ensure_dir(project_path(args.output_dir))
    model_path = model_dir / "quinella_top2_ranker.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    metadata = {
        **model.to_metadata(),
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "target": "target_quinella_rank_le_2",
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "investment_scored_csv": args.investment_scored_csv,
        "model_path": str(model_path),
        "train": _evaluate(train_scored, race_col, rank_col),
        "test": _evaluate(test_scored, race_col, rank_col),
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    keep_cols = [
        "race_id",
        "horse_no",
        "horse_name",
        "quinella_model_score",
        "quinella_model_prob",
        "quinella_model_rank",
        "quinella_model_score_norm",
    ]
    investment_scored.to_csv(out_dir / "investment_features_with_quinella_score.csv", index=False, encoding="utf-8-sig")
    investment_scored[[c for c in keep_cols if c in investment_scored.columns]].to_csv(
        out_dir / "quinella_scores_for_investment.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
