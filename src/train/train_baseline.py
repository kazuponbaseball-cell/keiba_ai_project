from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from src.data.loaders import (
    load_historical_csv,
    load_json_config,
    model_categorical_features,
    model_numeric_features,
    required_columns,
)
from src.features.baseline import (
    assert_no_leakage,
    contract_from_config,
    prepare_training_frame,
    split_by_recent_dates,
)
from src.train.simple_ranker import SimpleRaceRanker
from src.utils.paths import ensure_dir, project_path


def evaluate(df: pd.DataFrame, scores: np.ndarray, config: dict) -> dict[str, float]:
    race_col = config["data"]["race_id_column"]
    rank_col = config["data"]["rank_column"]
    scored = df[[race_col, rank_col, "target_win", "target_top3"]].copy()
    scored["score"] = scores
    scored["pred_rank"] = scored.groupby(race_col)["score"].rank(ascending=False, method="first")
    top1 = scored[scored["pred_rank"] == 1]
    top3 = scored[scored["pred_rank"] <= 3]
    return {
        "races": float(scored[race_col].nunique()),
        "rows": float(len(scored)),
        "top1_win_rate": float(top1["target_win"].mean()) if len(top1) else 0.0,
        "top1_top3_rate": float(top1["target_top3"].mean()) if len(top1) else 0.0,
        "top3_contains_winner_rate": float(top3.groupby(race_col)["target_win"].max().mean()) if len(top3) else 0.0,
        "mean_winner_pred_rank": float(scored.loc[scored[rank_col] == 1, "pred_rank"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first leakage-guarded baseline ranker.")
    parser.add_argument("--config", default="config/baseline_features.json")
    parser.add_argument("--output-dir", default="models/baseline")
    parser.add_argument("--train-csv", default=None, help="Use a prepared training feature CSV instead of rebuilding features.")
    parser.add_argument("--test-csv", default=None, help="Use a prepared temporal test feature CSV instead of rebuilding features.")
    args = parser.parse_args()

    config = load_json_config(args.config)
    contract = contract_from_config(config)
    assert_no_leakage(contract)

    date_col = config["data"]["date_column"]
    if args.train_csv and args.test_csv:
        train_df = pd.read_csv(project_path(args.train_csv), low_memory=False)
        test_df = pd.read_csv(project_path(args.test_csv), low_memory=False)
        cutoff = int(pd.to_numeric(test_df[date_col], errors="coerce").min())
    else:
        columns = required_columns(config)
        raw = load_historical_csv(config, columns=columns)
        frame = prepare_training_frame(raw, config)
        train_df, test_df, cutoff = split_by_recent_dates(frame, config)

    model = SimpleRaceRanker(
        numeric_features=model_numeric_features(config),
        categorical_features=model_categorical_features(config),
        categorical_top_k=int(config["training"].get("categorical_top_k", 80)),
        ridge_alpha=float(config["training"].get("ridge_alpha", 10.0)),
    ).fit(train_df, "target_score")

    train_scores = model.predict(train_df)
    test_scores = model.predict(test_df)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(project_path(args.output_dir))
    model_path = output_dir / "baseline_ranker.pkl"
    metadata_path = output_dir / "baseline_metadata.json"

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    metadata = {
        **model.to_metadata(),
        "run_id": run_id,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "temporal_test_cutoff_date": cutoff,
        "train": evaluate(train_df, train_scores, config),
        "test": evaluate(test_df, test_scores, config),
        "leakage_policy": {
            "banned_keywords": config.get("leakage_banned_feature_keywords", []),
            "allowed_prefixes": config.get("leakage_allowed_prefixes", []),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
