from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


NUMERIC_FEATURES = [
    "horse_no",
    "枠番",
    "頭数",
    "出走頭数",
    "popularity",
    "pop_rank_num",
    "odds",
    "odds_num",
    "market_win_prob_norm",
    "front_running_tendency_x",
    "front_running_tendency_y",
    "front_running_tendency",
    "closing_tendency_x",
    "closing_tendency_y",
    "race_front_runner_count_x",
    "race_front_runner_count_y",
    "race_front_runner_ratio",
    "race_early_pressure_score",
    "race_closer_count",
    "race_closer_ratio",
    "race_pace_collapse_risk",
    "race_slow_pace_risk",
    "pace_fit_score",
    "front_advantage_score",
    "positioning_advantage_score",
    "draw_pace_fit_score",
    "horse_front_run_rate_past5",
    "horse_front_run_rate_past5_feature",
    "horse_stalker_rate_past5",
    "horse_closer_rate_past5",
    "horse_closer_rate_past5_feature",
    "prev_corner4_position_rate",
    "front_pressure_rank_score",
    "solo_lead_potential",
]


CATEGORICAL_FEATURES = [
    "venue",
    "surface",
    "distance_bin",
    "distance_category_eval",
    "class_group",
    "field_bin",
    "expected_pace",
    "馬場状態",
]


LEAK_COLUMNS = {
    "finish",
    "finish_num",
    "win_pay",
    "place_pay",
    "win_return",
    "place_return",
    "is_win",
    "is_place",
    "is_quinella",
    "4角",
    "4角.1",
    "actual_front5",
}


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def parse_date(df: pd.DataFrame) -> pd.Series:
    if "日付S" in df.columns:
        parsed = pd.to_datetime(df["日付S"].astype(str), errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=df.index)
    missing = parsed.isna()
    if missing.any() and "race_id" in df.columns:
        digits = df.loc[missing, "race_id"].astype(str).str.extract(r"(\d{8})", expand=False)
        parsed.loc[missing] = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    return parsed


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos_rank_sum = ranks[y == 1].sum()
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def metric_block(y: np.ndarray, p: np.ndarray, label: str) -> dict:
    p = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)
    y = y.astype(float)
    top_n = max(1, int(np.ceil(len(y) * 0.10)))
    top_idx = np.argsort(-p)[:top_n]
    base_rate = float(y.mean()) if len(y) else 0.0
    top_rate = float(y[top_idx].mean()) if len(top_idx) else 0.0
    return {
        "label": label,
        "rows": int(len(y)),
        "actual_front5_rate": base_rate,
        "auc": auc_score(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "top10pct_front5_rate": top_rate,
        "top10pct_lift": top_rate / base_rate if base_rate else 0.0,
        "avg_pred": float(p.mean()),
    }


def build_global_design(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[dict[str, str]], np.ndarray]:
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns and c not in LEAK_COLUMNS]
    numeric = []
    for col in numeric_cols:
        numeric.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
    x_num = np.vstack(numeric).T if numeric else np.empty((len(df), 0))

    cat_parts = []
    cat_specs: list[dict[str, str]] = []
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns or col in LEAK_COLUMNS:
            continue
        values = df[col].fillna("NA").astype(str)
        top = values.value_counts().head(20).index.tolist()
        for val in top:
            cat_parts.append((values == val).to_numpy(dtype=float))
            cat_specs.append({"column": col, "value": str(val), "feature": f"{col}={val}"})
    x_cat = np.vstack(cat_parts).T if cat_parts else np.empty((len(df), 0))
    return x_num, numeric_cols, cat_specs, x_cat


def standardization_params(x_num: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_num.shape[1] == 0:
        return np.array([]), np.array([]), np.array([])
    fill = np.nanmedian(x_num, axis=0)
    filled = np.where(np.isnan(x_num), fill, x_num)
    mean = np.mean(filled, axis=0)
    scale = np.std(filled, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return fill, mean, scale


def apply_standardization(x_num: np.ndarray, x_cat: np.ndarray, fill: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if x_num.shape[1] == 0:
        num_std = np.empty((len(x_num), 0))
    else:
        filled = np.where(np.isnan(x_num), fill, x_num)
        num_std = (filled - mean) / scale
    intercept = np.ones((len(x_num), 1), dtype=float)
    return np.hstack([intercept, num_std, x_cat])


def standardize(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    train_idx: np.ndarray,
    apply_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_num.shape[1] == 0:
        num_std = np.empty((len(apply_idx), 0))
        med = np.array([])
        scale = np.array([])
    else:
        train = x_num[train_idx]
        med = np.nanmedian(train, axis=0)
        filled_train = np.where(np.isnan(train), med, train)
        mean = np.mean(filled_train, axis=0)
        scale = np.std(filled_train, axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        apply = x_num[apply_idx]
        filled_apply = np.where(np.isnan(apply), med, apply)
        num_std = (filled_apply - mean) / scale
        med = mean
    intercept = np.ones((len(apply_idx), 1), dtype=float)
    x = np.hstack([intercept, num_std, x_cat[apply_idx]])
    return x, med, scale


def fit_logistic_irls(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    max_iter: int,
) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=float)
    ridge = np.eye(x.shape[1]) * l2
    ridge[0, 0] = 0.0
    n = max(1, len(y))
    for _ in range(max_iter):
        p = sigmoid(x @ beta)
        w = np.clip(p * (1.0 - p), 1e-4, None)
        grad = (x.T @ (p - y)) / n + ridge @ beta
        hess = (x.T @ (x * w[:, None])) / n + ridge
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta -= step
        if float(np.max(np.abs(step))) < 1e-5:
            break
    return beta


def fit_bin_calibrator(p: np.ndarray, y: np.ndarray, bins: int = 10, smoothing: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        return np.array([0.0, 1.0]), np.array([float(y.mean()), float(y.mean())])
    global_rate = float(y.mean())
    values = []
    mids = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi if hi == edges[-1] else p < hi)
        n = int(mask.sum())
        rate = float(y[mask].mean()) if n else global_rate
        smooth = (rate * n + global_rate * smoothing) / (n + smoothing)
        values.append(smooth)
        mids.append((lo + hi) / 2.0)
    return np.array(mids), np.array(values)


def apply_calibrator(p: np.ndarray, mids: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.clip(np.interp(p, mids, values, left=values[0], right=values[-1]), 1e-4, 1.0 - 1e-4)


def month_starts(dates: pd.Series) -> list[pd.Timestamp]:
    periods = dates.dt.to_period("M").dropna().sort_values().unique()
    return [pd.Timestamp(period.start_time) for period in periods]


def summarize_longshots(pred: pd.DataFrame) -> pd.DataFrame:
    odds = pd.to_numeric(pred.get("odds", np.nan), errors="coerce")
    pop = pd.to_numeric(pred.get("popularity", np.nan), errors="coerce")
    y = pred["actual_front5"].astype(float)
    out = pred[(odds.ge(10.0) | pop.ge(5.0))].copy()
    if out.empty:
        return pd.DataFrame()
    out["front5_model_bin"] = pd.qcut(out["front5_model_prob"], q=5, labels=False, duplicates="drop")
    rows = []
    for b, g in out.groupby("front5_model_bin", dropna=True):
        rows.append(
            {
                "bin": int(b),
                "rows": int(len(g)),
                "avg_model_prob": float(g["front5_model_prob"].mean()),
                "avg_heuristic_prob": float(g["projected_front5_prob"].mean()),
                "actual_front5_rate": float(g["actual_front5"].astype(float).mean()),
                "avg_odds": float(pd.to_numeric(g.get("odds"), errors="coerce").mean()),
                "avg_popularity": float(pd.to_numeric(g.get("popularity"), errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("bin")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an OOS front-5 position model using pre-race columns only.")
    parser.add_argument("--input-csv", default="outputs/analysis/risk_models_v1/investment_features_with_risk_models.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/front5_position_model_v1")
    parser.add_argument("--purge-days", type=int, default=7)
    parser.add_argument("--min-train-rows", type=int, default=15000)
    parser.add_argument("--l2", type=float, default=0.20)
    parser.add_argument("--max-iter", type=int, default=30)
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(project_path(args.input_csv), dtype={"race_id": str}, low_memory=False)
    df["_date"] = parse_date(df)
    df = df[df["_date"].notna() & df["actual_front5"].notna()].copy()
    df["actual_front5"] = df["actual_front5"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"]).astype(float)
    df = df.sort_values(["_date", "race_id", "horse_no"], kind="mergesort").reset_index(drop=True)

    x_num, numeric_cols, cat_specs, x_cat = build_global_design(df)
    cat_names = [spec["feature"] for spec in cat_specs]
    y = df["actual_front5"].to_numpy(dtype=float)
    dates = df["_date"]
    months = month_starts(dates)

    predictions = []
    fold_rows = []
    for test_start in months:
        test_end = test_start + pd.offsets.MonthBegin(1)
        train_end = test_start - pd.Timedelta(days=args.purge_days)
        train_idx = np.flatnonzero(dates.lt(train_end).to_numpy())
        test_idx = np.flatnonzero((dates.ge(test_start) & dates.lt(test_end)).to_numpy())
        if len(test_idx) == 0 or len(train_idx) < args.min_train_rows:
            continue

        x_train, _, _ = standardize(x_num, x_cat, train_idx, train_idx)
        x_test, _, _ = standardize(x_num, x_cat, train_idx, test_idx)
        beta = fit_logistic_irls(x_train, y[train_idx], l2=args.l2, max_iter=args.max_iter)
        train_raw = sigmoid(x_train @ beta)
        test_raw = sigmoid(x_test @ beta)
        mids, cal_values = fit_bin_calibrator(train_raw, y[train_idx])
        test_prob = apply_calibrator(test_raw, mids, cal_values)

        fold = df.iloc[test_idx][
            [
                "race_id",
                "日付S",
                "venue",
                "Ｒ",
                "horse_no",
                "horse_name",
                "popularity",
                "odds",
                "枠番",
                "surface",
                "distance",
                "expected_pace",
                "projected_front5_prob",
                "actual_front5",
            ]
        ].copy()
        fold["fold_test_month"] = test_start.strftime("%Y-%m")
        fold["front5_model_prob_raw"] = test_raw
        fold["front5_model_prob"] = test_prob
        predictions.append(fold)

        fold_rows.append(metric_block(y[test_idx], test_prob, f"{test_start:%Y-%m}_model"))
        fold_rows[-1]["test_month"] = test_start.strftime("%Y-%m")
        fold_rows[-1]["train_rows"] = int(len(train_idx))
        fold_rows[-1]["test_rows"] = int(len(test_idx))
        heuristic = pd.to_numeric(df.iloc[test_idx]["projected_front5_prob"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
        h = metric_block(y[test_idx], heuristic, f"{test_start:%Y-%m}_heuristic")
        h["test_month"] = test_start.strftime("%Y-%m")
        h["train_rows"] = int(len(train_idx))
        h["test_rows"] = int(len(test_idx))
        fold_rows.append(h)

    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    pred.to_csv(out_dir / "front5_oos_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(out_dir / "front5_fold_metrics.csv", index=False, encoding="utf-8-sig")

    overall_rows = []
    if not pred.empty:
        yy = pred["actual_front5"].astype(float).to_numpy()
        overall_rows.append(metric_block(yy, pred["front5_model_prob"].to_numpy(dtype=float), "oos_model"))
        overall_rows.append(metric_block(yy, pred["projected_front5_prob"].to_numpy(dtype=float), "existing_heuristic"))
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(out_dir / "front5_overall_metrics.csv", index=False, encoding="utf-8-sig")

    longshot = summarize_longshots(pred)
    longshot.to_csv(out_dir / "front5_longshot_lift.csv", index=False, encoding="utf-8-sig")

    # Fit one final model for coefficient inspection and runtime scoring. OOS
    # metrics above remain the validation source.
    fill, mean, scale = standardization_params(x_num)
    x_all = apply_standardization(x_num, x_cat, fill, mean, scale)
    beta = fit_logistic_irls(x_all, y, l2=args.l2, max_iter=args.max_iter)
    raw_all = sigmoid(x_all @ beta)
    cal_mids, cal_values = fit_bin_calibrator(raw_all, y)
    feature_names = ["intercept"] + numeric_cols + cat_names
    coef = pd.DataFrame({"feature": feature_names, "coefficient": beta})
    coef["abs_coefficient"] = coef["coefficient"].abs()
    coef.sort_values("abs_coefficient", ascending=False).to_csv(out_dir / "front5_feature_coefficients.csv", index=False, encoding="utf-8-sig")

    model_params = {
        "model_type": "logistic_irls_bin_calibrated",
        "target": "actual_front5",
        "numeric_features": numeric_cols,
        "categorical_specs": cat_specs,
        "feature_names": feature_names,
        "numeric_fill": fill.tolist(),
        "numeric_mean": mean.tolist(),
        "numeric_scale": scale.tolist(),
        "beta": beta.tolist(),
        "calibrator_mids": cal_mids.tolist(),
        "calibrator_values": cal_values.tolist(),
        "notes": [
            "Runtime use is allowed because the model uses only pre-race columns listed in NUMERIC_FEATURES/CATEGORICAL_FEATURES.",
            "OOS validation metrics are in front5_overall_metrics.csv; this artifact is the final all-history fit for live scoring.",
        ],
    }
    (out_dir / "front5_model_params.json").write_text(json.dumps(model_params, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input_csv": args.input_csv,
        "output_dir": str(out_dir),
        "rows": int(len(df)),
        "oos_rows": int(len(pred)),
        "folds": int(fold_metrics["test_month"].nunique()) if not fold_metrics.empty else 0,
        "purge_days": args.purge_days,
        "min_train_rows": args.min_train_rows,
        "numeric_features": numeric_cols,
        "categorical_features": CATEGORICAL_FEATURES,
        "overall_metrics": overall.to_dict(orient="records"),
        "longshot_lift": longshot.to_dict(orient="records") if not longshot.empty else [],
        "notes": [
            "The target is actual_front5, but target/corner/result/payoff columns are excluded from features.",
            "OOS folds are monthly, using only earlier dates with a purge gap.",
            "This validates the pre-race front-position model before feeding it into ROI ticket selection.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
