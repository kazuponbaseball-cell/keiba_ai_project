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
    DEFAULT_MODEL,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    RACE_COL,
    coefficient_importance,
    fit_ranker,
    metric_summary,
    num,
    race_z,
)
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "analysis" / "adjusted_final3f_value_v1"

NEW_NUMERIC_FEATURES = [
    "prev_final3f_adj_value",
    "prev_final3f_rank_score",
    "prev_final3f_from_back_value",
    "prev_final3f_highpace_value",
    "prev_final3f_slow_instant_value",
    "past3_final3f_rank_score",
    "final3f_closer_fit_score",
    "final3f_collapse_fit_score",
    "final3f_slow_instant_fit_score",
    "final3f_front_hold_fit_score",
    "final3f_total_fit_score",
    "final3f_uncertainty_score",
]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sigmoid(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(values, -30, 30))))


def _first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _context_key(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series("__global__", index=frame.index)
    parts = []
    for col in cols:
        parts.append(frame[col].astype("string").fillna("__NA__").astype(str))
    out = parts[0]
    for part in parts[1:]:
        out = out + "|" + part
    return out


def _fit_prior_stats(train: pd.DataFrame) -> dict[str, Any]:
    prev_final = num(train, "前走上り3F")
    work = train.copy()
    work["_prev_final3f"] = prev_final
    context_cols = [
        c
        for c in ["前芝・ダ", "前距離", "前走馬場状態", "前クラス名"]
        if c in train.columns
    ]
    coarse_cols = [c for c in ["前芝・ダ", "前距離"] if c in train.columns]
    work["_context"] = _context_key(work, context_cols)
    work["_coarse"] = _context_key(work, coarse_cols)

    def stats_by(key_col: str) -> pd.DataFrame:
        g = work.dropna(subset=["_prev_final3f"]).groupby(key_col)["_prev_final3f"]
        stats = g.agg(["mean", "std", "count"]).reset_index()
        stats["std"] = stats["std"].fillna(0.0).clip(lower=0.25, upper=3.0)
        return stats

    global_mean = float(prev_final.dropna().mean()) if prev_final.notna().any() else 36.0
    global_std = float(prev_final.dropna().std()) if prev_final.notna().sum() >= 3 else 0.8
    if not np.isfinite(global_std) or global_std <= 0:
        global_std = 0.8
    return {
        "context_cols": context_cols,
        "coarse_cols": coarse_cols,
        "full": stats_by("_context"),
        "coarse": stats_by("_coarse"),
        "global_mean": global_mean,
        "global_std": float(np.clip(global_std, 0.25, 3.0)),
    }


def _apply_prior_stats(frame: pd.DataFrame, priors: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    out["_context"] = _context_key(out, priors["context_cols"])
    out["_coarse"] = _context_key(out, priors["coarse_cols"])
    full = priors["full"].rename(
        columns={"_context": "_context", "mean": "_final3f_full_mean", "std": "_final3f_full_std", "count": "_final3f_full_count"}
    )
    coarse = priors["coarse"].rename(
        columns={"_coarse": "_coarse", "mean": "_final3f_coarse_mean", "std": "_final3f_coarse_std", "count": "_final3f_coarse_count"}
    )
    out = out.merge(full, on="_context", how="left").merge(coarse, on="_coarse", how="left")
    out["_final3f_prior_mean"] = out["_final3f_full_mean"].where(out["_final3f_full_count"].fillna(0) >= 8)
    out["_final3f_prior_std"] = out["_final3f_full_std"].where(out["_final3f_full_count"].fillna(0) >= 8)
    out["_final3f_prior_count"] = out["_final3f_full_count"].where(out["_final3f_full_count"].fillna(0) >= 8)
    out["_final3f_prior_mean"] = out["_final3f_prior_mean"].fillna(out["_final3f_coarse_mean"]).fillna(priors["global_mean"])
    out["_final3f_prior_std"] = out["_final3f_prior_std"].fillna(out["_final3f_coarse_std"]).fillna(priors["global_std"]).clip(0.25, 3.0)
    out["_final3f_prior_count"] = out["_final3f_prior_count"].fillna(out["_final3f_coarse_count"]).fillna(0.0)
    return out


def add_adjusted_final3f_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    priors = _fit_prior_stats(train)
    train_out = _apply_prior_stats(train, priors)
    test_out = _apply_prior_stats(test, priors)
    combined = pd.concat(
        [train_out.assign(_split="train"), test_out.assign(_split="test")],
        ignore_index=True,
        sort=False,
    )

    prev_final = num(combined, "前走上り3F")
    prev_rank = num(combined, "前走上り3F順")
    prev_gap_3f = num(combined, "前走上3F地点差", 0.0).fillna(0.0).clip(lower=0.0, upper=8.0)
    prev_field = num(combined, "前走出走頭数", np.nan).fillna(num(combined, "前走頭数", np.nan)).replace(0, np.nan)
    prev_rpci = num(combined, "前走RPCI")
    prev_pci3 = num(combined, "前走PCI3")

    combined["prev_final3f_adj_value"] = (
        (combined["_final3f_prior_mean"] - prev_final) / combined["_final3f_prior_std"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-4.0, 4.0)
    combined["prev_final3f_rank_score"] = ((prev_field + 1 - prev_rank) / prev_field).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    back_pressure = (prev_gap_3f / 4.0).clip(0.0, 1.0)
    highpace = _sigmoid((47.0 - prev_rpci.fillna(50.0)) / 3.5).to_numpy()
    slow_instant = _sigmoid((prev_rpci.fillna(50.0) - 54.0) / 3.5).to_numpy()
    finish_pci = _sigmoid((prev_pci3.fillna(50.0) - 52.0) / 3.0).to_numpy()

    combined["prev_final3f_from_back_value"] = (
        combined["prev_final3f_adj_value"].clip(lower=0.0) * (0.50 + 0.50 * back_pressure)
    ).fillna(0.0)
    combined["prev_final3f_highpace_value"] = (
        combined["prev_final3f_adj_value"].clip(lower=0.0) * highpace
    ).fillna(0.0)
    combined["prev_final3f_slow_instant_value"] = (
        combined["prev_final3f_adj_value"].clip(lower=0.0) * np.maximum(slow_instant, finish_pci)
    ).fillna(0.0)

    rank_score = combined["prev_final3f_rank_score"]
    combined["past3_final3f_rank_score"] = (1.0 - (num(combined, "past3_avg_final3f_rank", 9.0).fillna(9.0) - 1.0) / 8.0).clip(0.0, 1.0)

    # Current-race fit: do not make "fast final3F" a blanket bonus. Tie it to the expected race shape.
    closer = num(combined, "horse_closer_rate_past5", 0.0).fillna(num(combined, "closing_tendency", 0.0)).fillna(0.0).clip(0.0, 1.0)
    front = num(combined, "horse_front_run_rate_past5", 0.0).fillna(num(combined, "front_running_tendency", 0.0)).fillna(0.0).clip(0.0, 1.0)
    collapse = num(combined, "race_pace_collapse_risk", 0.0).fillna(0.0).clip(0.0, 1.0)
    slow = num(combined, "race_slow_pace_risk", 0.0).fillna(0.0).clip(0.0, 1.0)
    combined["final3f_closer_fit_score"] = (
        race_z(combined["prev_final3f_from_back_value"], combined[RACE_COL]) * (0.35 + 0.65 * closer)
    ).fillna(0.0)
    combined["final3f_collapse_fit_score"] = (
        race_z(combined["prev_final3f_highpace_value"], combined[RACE_COL]) * (0.30 + 0.70 * collapse)
    ).fillna(0.0)
    combined["final3f_slow_instant_fit_score"] = (
        race_z(combined["prev_final3f_slow_instant_value"], combined[RACE_COL]) * (0.30 + 0.70 * slow)
    ).fillna(0.0)
    combined["final3f_front_hold_fit_score"] = (
        race_z(combined["prev_final3f_adj_value"].clip(lower=0.0) * front, combined[RACE_COL]) * (0.35 + 0.65 * front)
    ).fillna(0.0)
    combined["final3f_total_fit_score"] = (
        0.32 * combined["final3f_closer_fit_score"]
        + 0.26 * combined["final3f_collapse_fit_score"]
        + 0.20 * combined["final3f_slow_instant_fit_score"]
        + 0.14 * combined["final3f_front_hold_fit_score"]
        + 0.08 * race_z(rank_score, combined[RACE_COL])
    ).fillna(0.0)
    combined["final3f_uncertainty_score"] = (
        prev_final.isna().astype(float)
        + (combined["_final3f_prior_count"].fillna(0.0).lt(8).astype(float) * 0.5)
    ).clip(0.0, 1.0)

    for col in NEW_NUMERIC_FEATURES:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)

    train_x = combined[combined["_split"].eq("train")].drop(columns=["_split"], errors="ignore").copy()
    test_x = combined[combined["_split"].eq("test")].drop(columns=["_split"], errors="ignore").copy()
    diag_cols = [RACE_COL, "_final3f_prior_mean", "_final3f_prior_std", "_final3f_prior_count"]
    diag = combined[diag_cols].drop_duplicates(RACE_COL, keep="first").copy()
    return train_x, test_x, diag


def bet_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"bets": 0, "races": 0, "win_rate": 0.0, "top3_rate": 0.0, "win_roi": 0.0, "place_roi": 0.0}
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


def segment_report(train_x: pd.DataFrame, test_x: pd.DataFrame, base_scores: np.ndarray, plus_scores: np.ndarray) -> pd.DataFrame:
    scored = test_x.copy()
    scored["base_score"] = base_scores
    scored["plus_score"] = plus_scores
    scored["base_rank"] = scored.groupby(RACE_COL)["base_score"].rank(ascending=False, method="first").astype(int)
    scored["plus_rank"] = scored.groupby(RACE_COL)["plus_score"].rank(ascending=False, method="first").astype(int)
    q = {
        "fit_hi": float(num(train_x, "final3f_total_fit_score").quantile(0.75)),
        "fit_lo": float(num(train_x, "final3f_total_fit_score").quantile(0.25)),
        "closer_hi": float(num(train_x, "final3f_closer_fit_score").quantile(0.75)),
        "collapse_hi": float(num(train_x, "final3f_collapse_fit_score").quantile(0.75)),
        "instant_hi": float(num(train_x, "final3f_slow_instant_fit_score").quantile(0.75)),
        "prev_adj_hi": float(num(train_x, "prev_final3f_adj_value").quantile(0.75)),
    }
    pop5 = num(scored, "人気", np.nan).ge(5)
    checks = [
        ("base_top1", scored["base_rank"].eq(1)),
        ("plus_top1", scored["plus_rank"].eq(1)),
        ("plus_top1_final3f_fit_hi", scored["plus_rank"].eq(1) & num(scored, "final3f_total_fit_score").ge(q["fit_hi"])),
        ("plus_top1_final3f_fit_lo", scored["plus_rank"].eq(1) & num(scored, "final3f_total_fit_score").le(q["fit_lo"])),
        ("plus_top1_closer_hi", scored["plus_rank"].eq(1) & num(scored, "final3f_closer_fit_score").ge(q["closer_hi"])),
        ("plus_top1_collapse_hi", scored["plus_rank"].eq(1) & num(scored, "final3f_collapse_fit_score").ge(q["collapse_hi"])),
        ("plus_top1_slow_instant_hi", scored["plus_rank"].eq(1) & num(scored, "final3f_slow_instant_fit_score").ge(q["instant_hi"])),
        ("plus_top1_prev_adj_hi", scored["plus_rank"].eq(1) & num(scored, "prev_final3f_adj_value").ge(q["prev_adj_hi"])),
        ("plus_top3_pop5plus_final3f_fit_hi", scored["plus_rank"].le(3) & pop5 & num(scored, "final3f_total_fit_score").ge(q["fit_hi"])),
        ("plus_top3_pop5plus_closer_hi", scored["plus_rank"].le(3) & pop5 & num(scored, "final3f_closer_fit_score").ge(q["closer_hi"])),
        ("new_top1_changed_from_base", scored["plus_rank"].eq(1) & scored["base_rank"].ne(1)),
        ("base_top1_lost_by_plus", scored["base_rank"].eq(1) & scored["plus_rank"].ne(1)),
    ]
    return pd.DataFrame([{"segment": name, **bet_metrics(scored[mask])} for name, mask in checks])


def optimize_overlay(train_x: pd.DataFrame, test_x: pd.DataFrame, base_model: SimpleRaceRanker) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_train = base_model.predict(train_x)
    base_test = base_model.predict(test_x)
    total_train = race_z(train_x["final3f_total_fit_score"], train_x[RACE_COL]).to_numpy()
    risk_train = race_z(train_x["final3f_uncertainty_score"], train_x[RACE_COL]).to_numpy()
    total_test = race_z(test_x["final3f_total_fit_score"], test_x[RACE_COL]).to_numpy()
    risk_test = race_z(test_x["final3f_uncertainty_score"], test_x[RACE_COL]).to_numpy()
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for fit_w in np.arange(-0.04, 0.071, 0.01):
        for risk_w in np.arange(-0.04, 0.031, 0.01):
            scores = base_train + fit_w * total_train + risk_w * risk_train
            metrics = metric_summary(train_x, scores)
            selection_score = (
                metrics["top1_win_rate"] * 100
                + metrics["top1_top3_rate"] * 35
                + metrics["top1_win_roi"] * 20
                + metrics["top1_place_roi"] * 12
                - metrics["winner_mean_ai_rank"] * 0.8
            )
            row = {"fit_weight": float(fit_w), "risk_weight": float(risk_w), "selection_score": float(selection_score), **metrics}
            rows.append(row)
            if best is None or selection_score > best["selection_score"]:
                best = row
    assert best is not None
    test_scores = base_test + best["fit_weight"] * total_test + best["risk_weight"] * risk_test
    test_metrics = metric_summary(test_x, test_scores)
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False), {**best, **{f"test_{k}": v for k, v in test_metrics.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write-feature-csv", action="store_true")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(project_path(args.train_csv), low_memory=False)
    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    base_model: SimpleRaceRanker = pickle.load(project_path(args.base_model).open("rb"))
    train_x, test_x, diag = add_adjusted_final3f_features(train, test)
    if args.write_feature_csv:
        train_x.to_csv(out_dir / "train_adjusted_final3f_features.csv", index=False, encoding="utf-8-sig")
        test_x.to_csv(out_dir / "test_adjusted_final3f_features.csv", index=False, encoding="utf-8-sig")
    diag.to_csv(out_dir / "adjusted_final3f_diagnostics.csv", index=False, encoding="utf-8-sig")

    base_numeric = list(base_model.numeric_features)
    base_categorical = list(base_model.categorical_features)
    plus_numeric = base_numeric + [c for c in NEW_NUMERIC_FEATURES if c not in base_numeric]
    alpha = float(base_model.ridge_alpha)
    top_k = int(base_model.categorical_top_k)
    base_retrained = fit_ranker(train_x, base_numeric, base_categorical, alpha, top_k)
    plus_model = fit_ranker(train_x, plus_numeric, base_categorical, alpha, top_k)
    base_scores = base_retrained.predict(test_x)
    plus_scores = plus_model.predict(test_x)
    overlay_grid, overlay_best = optimize_overlay(train_x, test_x, base_retrained)

    rows = [
        {"variant": "saved_model_reference", **metric_summary(test_x, base_model.predict(test_x))},
        {"variant": "base_retrained_same_features", **metric_summary(test_x, base_scores)},
        {"variant": "adjusted_final3f_features_retrained", **metric_summary(test_x, plus_scores)},
        {
            "variant": "adjusted_final3f_overlay_on_base",
            "fit_weight": overlay_best["fit_weight"],
            "risk_weight": overlay_best["risk_weight"],
            **{k.replace("test_", ""): v for k, v in overlay_best.items() if k.startswith("test_")},
        },
    ]
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
    summary.to_csv(out_dir / "adjusted_final3f_summary.csv", index=False, encoding="utf-8-sig")
    segments = segment_report(train_x, test_x, base_scores, plus_scores)
    segments.to_csv(out_dir / "adjusted_final3f_segments.csv", index=False, encoding="utf-8-sig")
    overlay_grid.to_csv(out_dir / "adjusted_final3f_overlay_grid_train.csv", index=False, encoding="utf-8-sig")
    coefficient_importance(plus_model, "adjusted_final3f_features_retrained").to_csv(
        out_dir / "adjusted_final3f_importance.csv", index=False, encoding="utf-8-sig"
    )
    (out_dir / "overlay_best.json").write_text(json.dumps(overlay_best, ensure_ascii=False, indent=2), encoding="utf-8")

    show = summary.copy()
    for col in [
        "top1_win_rate",
        "top1_top3_rate",
        "top1_win_roi",
        "top1_place_roi",
        "top3_contains_winner_rate",
        "top3_win_roi",
        "top3_place_roi",
    ]:
        show[col] = show[col] * 100.0
    print(show[["variant", "top1_win_rate", "top1_top3_rate", "top1_win_roi", "top1_place_roi", "top3_contains_winner_rate", "winner_mean_ai_rank"]].to_string(index=False))
    print("\nsegments")
    seg_show = segments.copy()
    for col in ["win_rate", "top3_rate", "win_roi", "place_roi"]:
        seg_show[col] = seg_show[col] * 100.0
    print(seg_show.to_string(index=False))
    print(json.dumps({"output_dir": str(out_dir), "new_features": NEW_NUMERIC_FEATURES}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
