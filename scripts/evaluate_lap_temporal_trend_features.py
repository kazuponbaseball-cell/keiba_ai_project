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
    DATE_COL,
    DEFAULT_MODEL,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    HORSE_COL,
    RACE_COL,
    coefficient_importance,
    fit_ranker,
    metric_summary,
    num,
    race_z,
)
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "outputs" / "analysis" / "lap_temporal_trend_features_v1"

NEW_NUMERIC_FEATURES = [
    "lap_temporal_ready_count",
    "lap_temporal_recent_score",
    "lap_temporal_score_trend",
    "lap_temporal_score_stability",
    "lap_temporal_fit_recent",
    "lap_temporal_fit_trend",
    "lap_temporal_fit_stability",
    "lap_temporal_fit_best",
    "lap_temporal_fit_worst",
    "lap_temporal_fit_volatility",
    "lap_temporal_expected_fit_score",
    "lap_temporal_improving_fit_score",
    "lap_temporal_stable_axis_score",
    "lap_temporal_blowup_risk_score",
    "lap_temporal_rpci_recent",
    "lap_temporal_rpci_trend",
    "lap_temporal_pci3_recent",
    "lap_temporal_pci3_trend",
]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_date_series(series: pd.Series) -> pd.Series:
    raw = series.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip()
    out = pd.to_datetime(raw, errors="coerce")
    yymmdd = raw.str.fullmatch(r"\d{6}")
    if yymmdd.any():
        out.loc[yymmdd] = pd.to_datetime(raw.loc[yymmdd], format="%y%m%d", errors="coerce")
    yyyymmdd = raw.str.fullmatch(r"\d{8}")
    if yyyymmdd.any():
        out.loc[yyyymmdd] = pd.to_datetime(raw.loc[yyyymmdd], format="%Y%m%d", errors="coerce")
    return out


def clip01(series: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return pd.Series(series).fillna(0.0).clip(0.0, 1.0)


def _weighted_mean(values: list[pd.Series], weights: list[float]) -> pd.Series:
    idx = values[0].index
    numerator = pd.Series(0.0, index=idx, dtype=float)
    denominator = pd.Series(0.0, index=idx, dtype=float)
    for value, weight in zip(values, weights):
        val = pd.to_numeric(value, errors="coerce")
        ok = val.notna().astype(float)
        numerator = numerator + val.fillna(0.0) * weight * ok
        denominator = denominator + weight * ok
    return (numerator / denominator.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _row_std(values: list[pd.Series]) -> pd.Series:
    frame = pd.concat([pd.to_numeric(v, errors="coerce") for v in values], axis=1)
    return frame.std(axis=1, skipna=True).replace([np.inf, -np.inf], np.nan)


def add_lap_temporal_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["_split"] = "train"
    test["_split"] = "test"
    all_df = pd.concat([train, test], ignore_index=True, sort=False)
    all_df["_row_id"] = np.arange(len(all_df))
    all_df["_race_date_dt"] = parse_date_series(all_df[DATE_COL]) if DATE_COL in all_df.columns else pd.NaT
    all_df["_horse_key"] = all_df[HORSE_COL].astype("string").fillna("").astype(str)
    all_df["_race_key"] = all_df[RACE_COL].astype("string").fillna("").astype(str)

    rpci = num(all_df, "RPCI")
    pci = num(all_df, "PCI")
    pci3 = num(all_df, "PCI3")
    score = num(all_df, "target_score", 0.0).fillna(0.0).clip(0.0, 1.0)

    all_df["_lap_fast_regime"] = ((50.0 - rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)
    all_df["_lap_slow_regime"] = ((rpci - 50.0) / 5.0).clip(0.0, 1.0).fillna(0.0)
    all_df["_lap_instant_regime"] = ((pci - rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)
    all_df["_lap_sustain_regime"] = (1.0 - (pci - rpci).abs() / 4.0).clip(0.0, 1.0).fillna(0.0)
    all_df["_lap_long_spurt_regime"] = ((pci3 - rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)
    for kind in ["fast", "slow", "instant", "sustain", "long_spurt"]:
        all_df[f"_lap_{kind}_success"] = (all_df[f"_lap_{kind}_regime"] * score).clip(0.0, 1.0)

    all_df = all_df.sort_values(["_horse_key", "_race_date_dt", "_race_key", "_row_id"], kind="mergesort")
    grouped = all_df.groupby("_horse_key", sort=False)
    lag_sources = [
        "target_score",
        "RPCI",
        "PCI",
        "PCI3",
        "_lap_fast_success",
        "_lap_slow_success",
        "_lap_instant_success",
        "_lap_sustain_success",
        "_lap_long_spurt_success",
    ]
    for lag in range(1, 6):
        for col in lag_sources:
            all_df[f"lap_t{lag}_{col.removeprefix('_')}"] = grouped[col].shift(lag)
        all_df[f"lap_t{lag}_race_id"] = grouped["_race_key"].shift(lag)

    weights = [1.00, 0.86, 0.72, 0.58, 0.46]
    lag_score = [num(all_df, f"lap_t{lag}_target_score") for lag in range(1, 6)]
    lag_rpci = [num(all_df, f"lap_t{lag}_RPCI") for lag in range(1, 6)]
    lag_pci3 = [num(all_df, f"lap_t{lag}_PCI3") for lag in range(1, 6)]
    ready_count = pd.concat([v.notna().astype(float) for v in lag_score], axis=1).sum(axis=1)

    all_df["lap_temporal_ready_count"] = ready_count
    all_df["lap_temporal_recent_score"] = _weighted_mean(lag_score, weights).fillna(0.0).clip(0.0, 1.0)
    older_score = _weighted_mean(lag_score[2:], weights[2:]).fillna(all_df["lap_temporal_recent_score"])
    all_df["lap_temporal_score_trend"] = (lag_score[0].fillna(older_score) - older_score).fillna(0.0).clip(-1.0, 1.0)
    score_std = _row_std(lag_score).fillna(0.0).clip(0.0, 0.6)
    all_df["lap_temporal_score_stability"] = (
        all_df["lap_temporal_recent_score"] - 0.45 * score_std
    ).clip(0.0, 1.0)

    fast_need = num(all_df, "race_quality_fast_need_score", 0.0).fillna(num(all_df, "expected_lap_fast_weight", 0.0)).clip(0.0, 1.0)
    slow_need = num(all_df, "race_quality_slow_need_score", 0.0).fillna(num(all_df, "expected_lap_slow_weight", 0.0)).clip(0.0, 1.0)
    sustain_need = num(all_df, "race_quality_sustain_need_score", np.nan)
    if sustain_need.isna().all():
        sustain_need = (1.0 - np.maximum(fast_need, slow_need)).clip(0.0, 1.0)
    sustain_need = sustain_need.fillna((1.0 - np.maximum(fast_need, slow_need)).clip(0.0, 1.0)).clip(0.0, 1.0)

    fit_lags: list[pd.Series] = []
    for lag in range(1, 6):
        fast = num(all_df, f"lap_t{lag}_lap_fast_success", 0.0).fillna(0.0).clip(0.0, 1.0)
        slow = num(all_df, f"lap_t{lag}_lap_slow_success", 0.0).fillna(0.0).clip(0.0, 1.0)
        instant = num(all_df, f"lap_t{lag}_lap_instant_success", 0.0).fillna(0.0).clip(0.0, 1.0)
        sustain = num(all_df, f"lap_t{lag}_lap_sustain_success", 0.0).fillna(0.0).clip(0.0, 1.0)
        long_spurt = num(all_df, f"lap_t{lag}_lap_long_spurt_success", 0.0).fillna(0.0).clip(0.0, 1.0)
        lag_fit = (
            fast_need * (0.72 * fast + 0.18 * sustain + 0.10 * long_spurt)
            + slow_need * (0.58 * slow + 0.24 * instant + 0.18 * sustain)
            + sustain_need * (0.42 * sustain + 0.34 * long_spurt + 0.24 * instant)
        ).clip(0.0, 1.0)
        has_lag = all_df[f"lap_t{lag}_race_id"].astype("string").fillna("").ne("").astype(float)
        fit_lags.append(lag_fit.where(has_lag.gt(0), np.nan))

    all_df["lap_temporal_fit_recent"] = _weighted_mean(fit_lags, weights).fillna(0.0).clip(0.0, 1.0)
    older_fit = _weighted_mean(fit_lags[2:], weights[2:]).fillna(all_df["lap_temporal_fit_recent"])
    all_df["lap_temporal_fit_trend"] = (fit_lags[0].fillna(older_fit) - older_fit).fillna(0.0).clip(-1.0, 1.0)
    fit_frame = pd.concat(fit_lags, axis=1)
    fit_std = fit_frame.std(axis=1, skipna=True).fillna(0.0).clip(0.0, 0.6)
    all_df["lap_temporal_fit_volatility"] = fit_std
    all_df["lap_temporal_fit_stability"] = (
        all_df["lap_temporal_fit_recent"] - 0.50 * fit_std
    ).clip(0.0, 1.0)
    all_df["lap_temporal_fit_best"] = fit_frame.max(axis=1, skipna=True).fillna(0.0).clip(0.0, 1.0)
    all_df["lap_temporal_fit_worst"] = fit_frame.min(axis=1, skipna=True).fillna(0.0).clip(0.0, 1.0)

    all_df["lap_temporal_expected_fit_score"] = (
        0.44 * all_df["lap_temporal_fit_recent"]
        + 0.24 * all_df["lap_temporal_fit_stability"]
        + 0.18 * all_df["lap_temporal_fit_best"]
        + 0.14 * (0.5 + 0.5 * all_df["lap_temporal_fit_trend"]).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    all_df["lap_temporal_improving_fit_score"] = (
        all_df["lap_temporal_fit_recent"] + 0.55 * all_df["lap_temporal_fit_trend"].clip(lower=0.0)
    ).clip(0.0, 1.0)
    all_df["lap_temporal_stable_axis_score"] = (
        0.62 * all_df["lap_temporal_fit_stability"] + 0.38 * all_df["lap_temporal_score_stability"]
    ).clip(0.0, 1.0)
    all_df["lap_temporal_blowup_risk_score"] = (
        0.55 * all_df["lap_temporal_fit_volatility"]
        + 0.30 * (1.0 - all_df["lap_temporal_fit_worst"])
        + 0.15 * ready_count.lt(2).astype(float)
    ).clip(0.0, 1.0)

    all_df["lap_temporal_rpci_recent"] = _weighted_mean(lag_rpci, weights).fillna(50.0)
    older_rpci = _weighted_mean(lag_rpci[2:], weights[2:]).fillna(all_df["lap_temporal_rpci_recent"])
    all_df["lap_temporal_rpci_trend"] = (lag_rpci[0].fillna(older_rpci) - older_rpci).fillna(0.0).clip(-15.0, 15.0)
    all_df["lap_temporal_pci3_recent"] = _weighted_mean(lag_pci3, weights).fillna(50.0)
    older_pci3 = _weighted_mean(lag_pci3[2:], weights[2:]).fillna(all_df["lap_temporal_pci3_recent"])
    all_df["lap_temporal_pci3_trend"] = (lag_pci3[0].fillna(older_pci3) - older_pci3).fillna(0.0).clip(-15.0, 15.0)

    for col in NEW_NUMERIC_FEATURES:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce").fillna(0.0)

    diag_cols = [RACE_COL, DATE_COL, HORSE_COL, "馬番", *NEW_NUMERIC_FEATURES]
    diagnostics = all_df[[c for c in diag_cols if c in all_df.columns]].copy()
    all_df = all_df.sort_values("_row_id", kind="mergesort")
    train_x = all_df[all_df["_split"].eq("train")].drop(columns=["_split", "_row_id", "_race_date_dt", "_horse_key", "_race_key"], errors="ignore")
    test_x = all_df[all_df["_split"].eq("test")].drop(columns=["_split", "_row_id", "_race_date_dt", "_horse_key", "_race_key"], errors="ignore")
    return train_x, test_x, diagnostics


def bet_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "bets": 0,
            "races": 0,
            "win_rate": np.nan,
            "top3_rate": np.nan,
            "win_roi": np.nan,
            "place_roi": np.nan,
            "avg_popularity": np.nan,
            "avg_odds": np.nan,
        }
    win_pay = num(frame, "単勝配当", 0.0).fillna(0.0).where(frame["target_win"].eq(1), 0.0)
    place_pay = num(frame, "複勝配当", 0.0).fillna(0.0).where(frame["target_top3"].eq(1), 0.0)
    return {
        "bets": int(len(frame)),
        "races": int(frame[RACE_COL].nunique()),
        "win_rate": float(frame["target_win"].mean()),
        "top3_rate": float(frame["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (len(frame) * 100.0)),
        "place_roi": float(place_pay.sum() / (len(frame) * 100.0)),
        "avg_popularity": float(num(frame, "人気", np.nan).mean()),
        "avg_odds": float(num(frame, "単勝オッズ", np.nan).mean()),
    }


def optimize_overlay(train: pd.DataFrame, test: pd.DataFrame, base_model: SimpleRaceRanker) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_train = base_model.predict(train)
    base_test = base_model.predict(test)
    fit_train = race_z(train["lap_temporal_expected_fit_score"], train[RACE_COL]).to_numpy()
    trend_train = race_z(train["lap_temporal_improving_fit_score"], train[RACE_COL]).to_numpy()
    stable_train = race_z(train["lap_temporal_stable_axis_score"], train[RACE_COL]).to_numpy()
    risk_train = race_z(train["lap_temporal_blowup_risk_score"], train[RACE_COL]).to_numpy()
    fit_test = race_z(test["lap_temporal_expected_fit_score"], test[RACE_COL]).to_numpy()
    trend_test = race_z(test["lap_temporal_improving_fit_score"], test[RACE_COL]).to_numpy()
    stable_test = race_z(test["lap_temporal_stable_axis_score"], test[RACE_COL]).to_numpy()
    risk_test = race_z(test["lap_temporal_blowup_risk_score"], test[RACE_COL]).to_numpy()

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    # Keep this grid intentionally compact. The purpose is to test whether the
    # temporal lap signal is directionally useful, not to overfit a tiny weight.
    for fit_w in [-0.02, 0.0, 0.02, 0.04, 0.06]:
        for trend_w in [-0.02, 0.0, 0.02, 0.04]:
            for stable_w in [-0.02, 0.0, 0.02, 0.04]:
                for risk_w in [-0.04, -0.02, 0.0, 0.02]:
                    scores = base_train + fit_w * fit_train + trend_w * trend_train + stable_w * stable_train + risk_w * risk_train
                    m = metric_summary(train, scores)
                    score = (
                        m["top1_win_rate"] * 100
                        + m["top1_top3_rate"] * 35
                        + m["top1_win_roi"] * 20
                        + m["top1_place_roi"] * 14
                        - m["winner_mean_ai_rank"] * 0.8
                    )
                    row = {
                        "fit_weight": float(fit_w),
                        "trend_weight": float(trend_w),
                        "stable_weight": float(stable_w),
                        "risk_weight": float(risk_w),
                        "selection_score": float(score),
                        **m,
                    }
                    rows.append(row)
                    if best is None or score > best["selection_score"]:
                        best = row
    assert best is not None
    test_scores = (
        base_test
        + best["fit_weight"] * fit_test
        + best["trend_weight"] * trend_test
        + best["stable_weight"] * stable_test
        + best["risk_weight"] * risk_test
    )
    test_metrics = metric_summary(test, test_scores)
    best = {**best, **{f"test_{k}": v for k, v in test_metrics.items()}}
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False), best


def temporal_segments(train: pd.DataFrame, test: pd.DataFrame, base_scores: np.ndarray, plus_scores: np.ndarray) -> pd.DataFrame:
    q = {
        "fit_hi": float(num(train, "lap_temporal_expected_fit_score").quantile(0.75)),
        "fit_top": float(num(train, "lap_temporal_expected_fit_score").quantile(0.90)),
        "trend_hi": float(num(train, "lap_temporal_improving_fit_score").quantile(0.75)),
        "stable_hi": float(num(train, "lap_temporal_stable_axis_score").quantile(0.75)),
        "risk_hi": float(num(train, "lap_temporal_blowup_risk_score").quantile(0.75)),
    }
    scored = test.copy()
    scored["base_score_eval"] = base_scores
    scored["plus_score_eval"] = plus_scores
    scored["base_rank_eval"] = scored.groupby(RACE_COL)["base_score_eval"].rank(ascending=False, method="first").astype(int)
    scored["plus_rank_eval"] = scored.groupby(RACE_COL)["plus_score_eval"].rank(ascending=False, method="first").astype(int)

    checks: list[tuple[str, pd.Series]] = [
        ("plus_top1", scored["plus_rank_eval"].eq(1)),
        ("plus_top1_fit_hi", scored["plus_rank_eval"].eq(1) & num(scored, "lap_temporal_expected_fit_score").ge(q["fit_hi"])),
        ("plus_top1_fit_top10", scored["plus_rank_eval"].eq(1) & num(scored, "lap_temporal_expected_fit_score").ge(q["fit_top"])),
        ("plus_top1_trend_hi", scored["plus_rank_eval"].eq(1) & num(scored, "lap_temporal_improving_fit_score").ge(q["trend_hi"])),
        ("plus_top1_stable_hi", scored["plus_rank_eval"].eq(1) & num(scored, "lap_temporal_stable_axis_score").ge(q["stable_hi"])),
        ("plus_top1_risk_hi", scored["plus_rank_eval"].eq(1) & num(scored, "lap_temporal_blowup_risk_score").ge(q["risk_hi"])),
        ("plus_top3_pop5plus_fit_hi", scored["plus_rank_eval"].le(3) & num(scored, "人気", 99).ge(5) & num(scored, "lap_temporal_expected_fit_score").ge(q["fit_hi"])),
        ("plus_top3_pop5plus_trend_hi", scored["plus_rank_eval"].le(3) & num(scored, "人気", 99).ge(5) & num(scored, "lap_temporal_improving_fit_score").ge(q["trend_hi"])),
        ("new_top1_changed_from_base", scored["plus_rank_eval"].eq(1) & scored["base_rank_eval"].ne(1)),
        ("base_top1_lost_by_plus", scored["base_rank_eval"].eq(1) & scored["plus_rank_eval"].ne(1)),
    ]
    rows = []
    for name, mask in checks:
        rows.append({"segment": name, **bet_metrics(scored[mask])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3-5 race temporal lap trend/stability features.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--write-feature-csv", action="store_true")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(project_path(args.train_csv), low_memory=False)
    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    base_model: SimpleRaceRanker = pickle.load(project_path(args.base_model).open("rb"))

    train_x, test_x, diag = add_lap_temporal_features(train, test)
    if args.write_feature_csv:
        train_x.to_csv(out_dir / "train_features_lap_temporal.csv", index=False, encoding="utf-8-sig")
        test_x.to_csv(out_dir / "test_features_lap_temporal.csv", index=False, encoding="utf-8-sig")
    diag.to_csv(out_dir / "lap_temporal_runner_diagnostics.csv", index=False, encoding="utf-8-sig")

    base_numeric = list(base_model.numeric_features)
    base_categorical = list(base_model.categorical_features)
    plus_numeric = base_numeric + [col for col in NEW_NUMERIC_FEATURES if col not in base_numeric]
    plus_categorical = base_categorical
    alpha = float(base_model.ridge_alpha)
    top_k = int(base_model.categorical_top_k)

    base_retrained = fit_ranker(train_x, base_numeric, base_categorical, alpha, top_k)
    plus_model = fit_ranker(train_x, plus_numeric, plus_categorical, alpha, top_k)
    base_test_scores = base_retrained.predict(test_x)
    plus_test_scores = plus_model.predict(test_x)

    rows = [
        {"variant": "saved_model_reference", **metric_summary(test_x, base_model.predict(test_x))},
        {"variant": "base_retrained_same_features", **metric_summary(test_x, base_test_scores)},
        {"variant": "lap_temporal_features_retrained", **metric_summary(test_x, plus_test_scores)},
    ]
    overlay_grid, overlay_best = optimize_overlay(train_x, test_x, base_retrained)
    rows.append(
        {
            "variant": "lap_temporal_overlay_on_base_retrained",
            "fit_weight": overlay_best["fit_weight"],
            "trend_weight": overlay_best["trend_weight"],
            "stable_weight": overlay_best["stable_weight"],
            "risk_weight": overlay_best["risk_weight"],
            **{k.replace("test_", ""): v for k, v in overlay_best.items() if k.startswith("test_")},
        }
    )
    summary = pd.DataFrame(rows)
    baseline = summary[summary["variant"].eq("base_retrained_same_features")].iloc[0]
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

    summary.to_csv(out_dir / "lap_temporal_summary.csv", index=False, encoding="utf-8-sig")
    overlay_grid.to_csv(out_dir / "lap_temporal_overlay_grid_train.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "overlay_best.json").open("w", encoding="utf-8") as f:
        json.dump(overlay_best, f, ensure_ascii=False, indent=2)

    imps = pd.concat(
        [
            coefficient_importance(base_retrained, "base_retrained_same_features"),
            coefficient_importance(plus_model, "lap_temporal_features_retrained"),
        ],
        ignore_index=True,
    )
    imps.to_csv(out_dir / "lap_temporal_importance.csv", index=False, encoding="utf-8-sig")
    segments = temporal_segments(train_x, test_x, base_test_scores, plus_test_scores)
    segments.to_csv(out_dir / "lap_temporal_segments.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "new_features": NEW_NUMERIC_FEATURES,
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "test_races": int(test_x[RACE_COL].nunique()),
        "ready_rate_pct": float(num(test_x, "lap_temporal_ready_count").ge(2).mean() * 100.0),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(summary[[
        "variant",
        "top1_win_rate",
        "top1_top3_rate",
        "top1_win_roi",
        "top1_place_roi",
        "top3_contains_winner_rate",
        "top3_win_roi",
        "top3_place_roi",
        "winner_mean_ai_rank",
    ]].to_string(index=False))
    print("\nTemporal lap feature coefficients:")
    print(
        imps[
            imps["model"].eq("lap_temporal_features_retrained")
            & imps["feature"].isin(NEW_NUMERIC_FEATURES)
        ]
        .sort_values("abs_coef", ascending=False)
        .to_string(index=False)
    )
    print("\nTemporal lap segments:")
    print(segments.to_string(index=False))


if __name__ == "__main__":
    main()
