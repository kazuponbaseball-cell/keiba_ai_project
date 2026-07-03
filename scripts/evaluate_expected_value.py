from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def _softmax_win_prob(df: pd.DataFrame, race_col: str, alpha: float) -> pd.Series:
    scaled = df["ai_score"].astype(float) * alpha
    max_by_race = scaled.groupby(df[race_col]).transform("max")
    exp_score = np.exp((scaled - max_by_race).clip(-50, 50))
    denom = exp_score.groupby(df[race_col]).transform("sum").replace(0, np.nan)
    return (exp_score / denom).fillna(0.0)


def _logloss(y_true: pd.Series, prob: pd.Series) -> float:
    p = prob.clip(1e-6, 1.0 - 1e-6)
    y = y_true.astype(float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _calibrate_alpha(train: pd.DataFrame, race_col: str) -> dict[str, float]:
    candidates = np.concatenate(
        [
            np.linspace(0.25, 5.0, 20),
            np.linspace(5.5, 30.0, 50),
            np.linspace(32.0, 80.0, 25),
        ]
    )
    rows = []
    for alpha in candidates:
        prob = _softmax_win_prob(train, race_col, float(alpha))
        rows.append((float(alpha), _logloss(train["target_win"], prob)))
    best_alpha, best_loss = min(rows, key=lambda x: x[1])
    return {"alpha": best_alpha, "train_logloss": best_loss}


def _odds_bucket(values: pd.Series) -> pd.Series:
    bins = [-np.inf, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0, 50.0, 100.0, np.inf]
    labels = ["<=2", "2-3", "3-5", "5-7", "7-10", "10-15", "15-30", "30-50", "50-100", "100+"]
    return pd.cut(values, bins=bins, labels=labels).astype("string").fillna("unknown")


def _rank_bucket(values: pd.Series) -> pd.Series:
    bins = [-np.inf, 1, 2, 3, 5, 8, np.inf]
    labels = ["1", "2", "3", "4-5", "6-8", "9+"]
    return pd.cut(values, bins=bins, labels=labels).astype("string").fillna("unknown")


def _fit_bucket_probability(train: pd.DataFrame, *, smoothing: float = 80.0) -> dict[str, Any]:
    train = train.copy()
    train["odds_decimal"] = _num(train["単勝オッズ"]).replace(0, np.nan)
    train["odds_bucket"] = _odds_bucket(train["odds_decimal"])
    train["rank_bucket"] = _rank_bucket(train["ai_rank"])
    prior = float(train["target_win"].mean())

    grouped = train.groupby(["rank_bucket", "odds_bucket"], dropna=False)["target_win"].agg(["sum", "count"]).reset_index()
    grouped["prob"] = (grouped["sum"] + prior * smoothing) / (grouped["count"] + smoothing)

    rank_grouped = train.groupby(["rank_bucket"], dropna=False)["target_win"].agg(["sum", "count"]).reset_index()
    rank_grouped["prob"] = (rank_grouped["sum"] + prior * smoothing) / (rank_grouped["count"] + smoothing)

    return {
        "prior": prior,
        "smoothing": smoothing,
        "joint": {
            (str(row.rank_bucket), str(row.odds_bucket)): float(row.prob)
            for row in grouped.itertuples(index=False)
        },
        "rank": {str(row.rank_bucket): float(row.prob) for row in rank_grouped.itertuples(index=False)},
    }


def _temporal_calibration_frame(train: pd.DataFrame, model: Any, config: dict[str, Any], fraction: float) -> pd.DataFrame:
    date_col = config["data"]["date_column"]
    dates = np.array(sorted(pd.to_numeric(train[date_col], errors="coerce").dropna().unique()))
    if len(dates) < 5:
        scored = train.copy()
        scored["ai_score"] = model.predict(scored)
        return scored
    split_idx = max(1, min(len(dates) - 1, int(len(dates) * (1.0 - fraction))))
    cutoff = dates[split_idx]
    fit_df = train[pd.to_numeric(train[date_col], errors="coerce") < cutoff].copy()
    calib_df = train[pd.to_numeric(train[date_col], errors="coerce") >= cutoff].copy()
    cal_model = model.__class__(
        numeric_features=list(model.numeric_features),
        categorical_features=list(model.categorical_features),
        categorical_top_k=int(model.categorical_top_k),
        ridge_alpha=float(model.ridge_alpha),
    ).fit(fit_df, "target_score")
    calib_df["ai_score"] = cal_model.predict(calib_df)
    calib_df["calibration_cutoff_date"] = cutoff
    return calib_df


def _apply_bucket_probability(df: pd.DataFrame, calibration: dict[str, Any]) -> pd.Series:
    rank_buckets = _rank_bucket(df["ai_rank"])
    odds_buckets = _odds_bucket(df["odds_decimal"])
    joint = calibration["joint"]
    rank = calibration["rank"]
    prior = float(calibration["prior"])
    probs = []
    for rb, ob in zip(rank_buckets.astype(str), odds_buckets.astype(str)):
        probs.append(joint.get((rb, ob), rank.get(rb, prior)))
    return pd.Series(probs, index=df.index, dtype=float)


def _prepare_scored_frame(
    df: pd.DataFrame,
    model: Any,
    race_col: str,
    alpha: float,
    bucket_calibration: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    out["win_prob_softmax"] = _softmax_win_prob(out, race_col, alpha)
    out["pop_rank"] = out.groupby(race_col)["人気"].rank(ascending=True, method="first").astype(int)
    out["odds_decimal"] = _num(out["単勝オッズ"]).replace(0, np.nan)
    out["odds_bucket"] = _odds_bucket(out["odds_decimal"])
    out["rank_bucket"] = _rank_bucket(out["ai_rank"])
    if bucket_calibration is None:
        out["win_prob"] = out["win_prob_softmax"]
    else:
        out["win_prob"] = _apply_bucket_probability(out, bucket_calibration)
    out["win_ev"] = (out["win_prob"] * out["odds_decimal"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["win_edge"] = out["win_ev"] - 1.0
    out["kelly_fraction"] = ((out["win_prob"] * out["odds_decimal"] - 1.0) / (out["odds_decimal"] - 1.0)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0).clip(lower=0.0, upper=0.25)
    out["win_pay"] = _num(out["単勝配当"]).fillna(0.0).where(out["target_win"] == 1, 0.0)
    return out


def _bet_metrics(df: pd.DataFrame, race_col: str, label: str) -> dict[str, Any]:
    rows = len(df)
    if rows == 0:
        return {
            "segment": label,
            "bets": 0,
            "races": 0,
            "win_rate": 0.0,
            "avg_odds": None,
            "avg_win_prob": None,
            "avg_win_ev": None,
            "win_roi": 0.0,
            "profit_yen_flat100": 0.0,
        }
    pay = df["win_pay"].sum()
    stake = rows * 100.0
    return {
        "segment": label,
        "bets": int(rows),
        "races": int(df[race_col].nunique()),
        "win_rate": float(df["target_win"].mean()),
        "avg_odds": float(df["odds_decimal"].mean()),
        "avg_win_prob": float(df["win_prob"].mean()),
        "avg_win_ev": float(df["win_ev"].mean()),
        "win_roi": float(pay / stake),
        "profit_yen_flat100": float(pay - stake),
    }


def _summarize_ev_thresholds(test: pd.DataFrame, race_col: str) -> pd.DataFrame:
    rows = []
    thresholds = [0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]
    for threshold in thresholds:
        rows.append(_bet_metrics(test[test["win_ev"] >= threshold], race_col, f"all_ev>={threshold:.2f}"))
        top_per_race = (
            test[test["win_ev"] >= threshold]
            .sort_values([race_col, "win_ev"], ascending=[True, False])
            .groupby(race_col, as_index=False)
            .head(1)
        )
        rows.append(_bet_metrics(top_per_race, race_col, f"top1_ev_per_race_ev>={threshold:.2f}"))

    rows.extend(
        [
            _bet_metrics(test[test["ai_rank"] == 1], race_col, "ai_rank_1"),
            _bet_metrics(test[test["ai_rank"] <= 3], race_col, "ai_rank_1_3"),
            _bet_metrics(test[(test["ai_rank"] <= 3) & (test["win_ev"] >= 1.0)], race_col, "ai_rank_1_3_ev>=1.00"),
            _bet_metrics(test[(test["ai_rank"] <= 3) & (test["win_ev"] >= 1.1)], race_col, "ai_rank_1_3_ev>=1.10"),
            _bet_metrics(test[(test["ai_rank"] <= 5) & (test["win_ev"] >= 1.0)], race_col, "ai_rank_1_5_ev>=1.00"),
        ]
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate win expected value from calibrated AI scores and win odds.")
    parser.add_argument("--config", default="config/baseline_features.json")
    parser.add_argument("--model", default="models/jockey_trainer_rotation_baseline/baseline_ranker.pkl")
    parser.add_argument("--train-csv", default="data/datasets/cache/jockey_trainer_rotation/train_features.csv")
    parser.add_argument("--test-csv", default="data/datasets/cache/jockey_trainer_rotation/test_features.csv")
    parser.add_argument("--output-dir", default="outputs/evaluation_expected_value")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]

    train = pd.read_csv(project_path(args.train_csv), low_memory=False)
    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    with project_path(args.model).open("rb") as f:
        model = pickle.load(f)

    calib_scored = _temporal_calibration_frame(train, model, config, args.calibration_fraction)
    calib_scored["ai_rank"] = calib_scored.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    calibration = _calibrate_alpha(calib_scored, race_col)
    softmax_calib_prob = _softmax_win_prob(calib_scored, race_col, calibration["alpha"])
    calibration["softmax_calibration_logloss"] = _logloss(calib_scored["target_win"], softmax_calib_prob)
    calibration["calibration_rows"] = int(len(calib_scored))
    if "calibration_cutoff_date" in calib_scored.columns:
        calibration["calibration_cutoff_date"] = int(calib_scored["calibration_cutoff_date"].iloc[0])
    bucket_calibration = _fit_bucket_probability(calib_scored)
    test_scored = _prepare_scored_frame(test, model, race_col, calibration["alpha"], bucket_calibration)

    output_dir = ensure_dir(project_path(args.output_dir))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(output_dir / stamp)

    threshold_summary = _summarize_ev_thresholds(test_scored, race_col)
    detail_cols = [
        race_col,
        "日付S",
        "場所",
        "Ｒ",
        "レース名",
        "馬番",
        "馬名",
        "ai_rank",
        "ai_score",
        "win_prob",
        "単勝オッズ",
        "odds_decimal",
        "win_ev",
        "win_edge",
        "kelly_fraction",
        "人気",
        "pop_rank",
        "確定着順",
        "単勝配当",
        "win_pay",
    ]
    detail = test_scored[[col for col in detail_cols if col in test_scored.columns]].sort_values(
        [race_col, "win_ev"], ascending=[True, False]
    )

    threshold_summary.to_csv(run_dir / "ev_threshold_summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(run_dir / "ev_prediction_detail.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "run_dir": str(run_dir),
        "config": args.config,
        "model": args.model,
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "calibration": calibration,
        "bucket_probability": {
            "prior": bucket_calibration["prior"],
            "smoothing": bucket_calibration["smoothing"],
            "method": "empirical_win_rate_by_ai_rank_bucket_and_odds_bucket",
        },
        "top_segments": threshold_summary.to_dict(orient="records"),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
