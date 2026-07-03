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

from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


DEFAULT_TRAIN = Path(
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
    "body_weight_backfilled/owner_breeder_enriched/train_features_with_same_day_bias_v3_retro_body_breeder.csv"
)
DEFAULT_TEST = Path(
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
    "body_weight_backfilled/owner_breeder_enriched/test_features_with_same_day_bias_v3_retro_body_breeder.csv"
)
DEFAULT_MODEL = Path("models/body_owner_numeric_breeder_context_same_day_bias_v3_retro/baseline_ranker.pkl")
DEFAULT_OUT_DIR = Path("outputs/analysis/expected_lap_rpci")

RACE_COL = "レースID(新/馬番無)"
HORSE_COL = "血統登録番号"
DATE_COL = "日付"
RANK_COL = "確定着順"

NEW_NUMERIC_FEATURES = [
    "course_base_rpci_prior",
    "course_base_pci3_prior",
    "course_base_rpci_count",
    "expected_rpci",
    "expected_rpci_delta_from_course",
    "expected_pace_pressure_adjustment",
    "expected_lap_fast_weight",
    "expected_lap_slow_weight",
    "expected_lap_sustain_weight",
    "horse_rpci_mean_past5",
    "horse_rpci_std_past5",
    "horse_rpci_count_past5",
    "horse_expected_rpci_abs_gap",
    "horse_expected_rpci_fit_score",
    "expected_fast_lap_fit_score",
    "expected_slow_lap_fit_score",
    "expected_instant_lap_fit_score",
    "expected_sustain_lap_fit_score",
    "expected_long_spurt_lap_fit_score",
    "expected_lap_shape_fit_score",
    "expected_lap_total_fit_score",
    "expected_lap_front_adversity_score",
    "expected_lap_closer_adversity_score",
]
NEW_CATEGORICAL_FEATURES = ["expected_lap_type"]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = df[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def metric_summary(df: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    scored = df.copy()
    scored["ai_score_eval"] = scores
    scored["ai_rank_eval"] = scored.groupby(RACE_COL)["ai_score_eval"].rank(ascending=False, method="first").astype(int)
    top1 = scored[scored["ai_rank_eval"] == 1]
    top3 = scored[scored["ai_rank_eval"] <= 3]

    win_pay = num(top1, "単勝配当", 0.0).fillna(0.0).where(top1["target_win"].eq(1), 0.0)
    place_pay = num(top1, "複勝配当", 0.0).fillna(0.0).where(top1["target_top3"].eq(1), 0.0)
    top3_win_pay = num(top3, "単勝配当", 0.0).fillna(0.0).where(top3["target_win"].eq(1), 0.0)
    top3_place_pay = num(top3, "複勝配当", 0.0).fillna(0.0).where(top3["target_top3"].eq(1), 0.0)

    return {
        "rows": int(len(scored)),
        "races": int(scored[RACE_COL].nunique()),
        "top1_win_rate": float(top1["target_win"].mean()),
        "top1_top3_rate": float(top1["target_top3"].mean()),
        "top1_win_roi": float(win_pay.sum() / (len(top1) * 100.0)),
        "top1_place_roi": float(place_pay.sum() / (len(top1) * 100.0)),
        "top3_contains_winner_rate": float(top3.groupby(RACE_COL)["target_win"].max().mean()),
        "top3_win_roi": float(top3_win_pay.sum() / (len(top3) * 100.0)),
        "top3_place_roi": float(top3_place_pay.sum() / (len(top3) * 100.0)),
        "winner_mean_ai_rank": float(scored.loc[scored[RANK_COL].eq(1), "ai_rank_eval"].mean()),
    }


def _prior_group_stats(races: pd.DataFrame, keys: list[str], value: str, min_periods: int) -> tuple[pd.Series, pd.Series]:
    grouped = races.groupby(keys, sort=False)[value]
    prior_mean = grouped.transform(lambda s: s.shift().expanding(min_periods=min_periods).mean())
    prior_count = grouped.cumcount()
    return prior_mean, prior_count


def add_expected_lap_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["_split"] = "train"
    test["_split"] = "test"
    all_df = pd.concat([train, test], ignore_index=True, sort=False)
    all_df["_row_id"] = np.arange(len(all_df))

    race_cols = [
        RACE_COL,
        DATE_COL,
        "Ｒ",
        "場所",
        "芝・ダ",
        "距離",
        "馬場状態",
        "クラス名",
        "RPCI",
        "PCI3",
    ]
    races = (
        all_df[[col for col in race_cols if col in all_df.columns]]
        .drop_duplicates(RACE_COL, keep="first")
        .copy()
        .sort_values([DATE_COL, "Ｒ", RACE_COL], kind="mergesort")
    )
    races["RPCI"] = num(races, "RPCI")
    races["PCI3"] = num(races, "PCI3")

    full_keys = ["場所", "芝・ダ", "距離", "馬場状態"]
    mid_keys = ["芝・ダ", "距離", "馬場状態"]
    coarse_keys = ["芝・ダ", "距離"]
    for keys, prefix, min_periods in [
        (full_keys, "full", 3),
        (mid_keys, "mid", 5),
        (coarse_keys, "coarse", 8),
    ]:
        if all(col in races.columns for col in keys):
            races[f"{prefix}_rpci_prior"], races[f"{prefix}_rpci_count"] = _prior_group_stats(races, keys, "RPCI", min_periods)
            races[f"{prefix}_pci3_prior"], races[f"{prefix}_pci3_count"] = _prior_group_stats(races, keys, "PCI3", min_periods)

    races["global_rpci_prior"] = races["RPCI"].shift().expanding(min_periods=20).mean()
    races["global_pci3_prior"] = races["PCI3"].shift().expanding(min_periods=20).mean()
    races["global_count"] = np.arange(len(races))

    races["course_base_rpci_prior"] = races.get("full_rpci_prior")
    races["course_base_pci3_prior"] = races.get("full_pci3_prior")
    races["course_base_rpci_count"] = races.get("full_rpci_count")
    for prefix in ["mid", "coarse"]:
        if f"{prefix}_rpci_prior" in races.columns:
            races["course_base_rpci_prior"] = races["course_base_rpci_prior"].fillna(races[f"{prefix}_rpci_prior"])
            races["course_base_pci3_prior"] = races["course_base_pci3_prior"].fillna(races[f"{prefix}_pci3_prior"])
            races["course_base_rpci_count"] = races["course_base_rpci_count"].fillna(races[f"{prefix}_rpci_count"])
    races["course_base_rpci_prior"] = races["course_base_rpci_prior"].fillna(races["global_rpci_prior"]).fillna(50.0)
    races["course_base_pci3_prior"] = races["course_base_pci3_prior"].fillna(races["global_pci3_prior"]).fillna(50.0)
    races["course_base_rpci_count"] = races["course_base_rpci_count"].fillna(races["global_count"]).fillna(0.0)

    race_features = races[[RACE_COL, "course_base_rpci_prior", "course_base_pci3_prior", "course_base_rpci_count"]]
    all_df = all_df.merge(race_features, on=RACE_COL, how="left")

    collapse = num(all_df, "race_pace_collapse_risk", 0.0).fillna(0.0).clip(0.0, 1.0)
    slow = num(all_df, "race_slow_pace_risk", 0.0).fillna(0.0).clip(0.0, 1.0)
    pressure = num(all_df, "race_early_pressure_score", 0.0).fillna(0.0).clip(0.0, 1.0)
    need_lead = num(all_df, "race_need_lead_count", 0.0).fillna(0.0)
    stalkers = num(all_df, "race_stalker_count_deep", 0.0).fillna(0.0)
    front_ratio = num(all_df, "race_front_runner_ratio", 0.0).fillna(0.0).clip(0.0, 1.0)

    all_df["expected_pace_pressure_adjustment"] = (
        3.0 * slow
        - 3.2 * collapse
        - 1.1 * pressure
        - 0.22 * (need_lead - 1.0).clip(lower=0.0)
        - 0.12 * stalkers.clip(upper=6.0)
        + 0.60 * (front_ratio <= 0.12).astype(float)
    )
    all_df["expected_rpci"] = (
        num(all_df, "course_base_rpci_prior", 50.0).fillna(50.0)
        + all_df["expected_pace_pressure_adjustment"]
    ).clip(42.0, 58.0)
    all_df["expected_rpci_delta_from_course"] = all_df["expected_rpci"] - num(all_df, "course_base_rpci_prior", 50.0)

    all_df["expected_lap_fast_weight"] = ((50.0 - all_df["expected_rpci"]) / 6.0).clip(0.0, 1.0)
    all_df["expected_lap_slow_weight"] = ((all_df["expected_rpci"] - 50.0) / 6.0).clip(0.0, 1.0)
    all_df["expected_lap_sustain_weight"] = (1.0 - (all_df["expected_rpci"] - 50.0).abs() / 6.0).clip(0.0, 1.0)
    all_df["expected_lap_type"] = np.select(
        [all_df["expected_rpci"] <= 48.0, all_df["expected_rpci"] >= 52.0],
        ["fast", "slow"],
        default="middle",
    )

    actual_rpci = num(all_df, "RPCI")
    actual_pci3 = num(all_df, "PCI3")
    ordered = all_df.sort_values([HORSE_COL, DATE_COL, RACE_COL], kind="mergesort")
    rpci_mean = actual_rpci.loc[ordered.index].groupby(ordered[HORSE_COL], sort=False).transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean()
    )
    rpci_std = actual_rpci.loc[ordered.index].groupby(ordered[HORSE_COL], sort=False).transform(
        lambda s: s.shift().rolling(5, min_periods=2).std(ddof=0)
    )
    rpci_count = actual_rpci.loc[ordered.index].notna().astype(float).groupby(ordered[HORSE_COL], sort=False).transform(
        lambda s: s.shift().rolling(5, min_periods=1).sum()
    )
    all_df.loc[ordered.index, "horse_rpci_mean_past5"] = pd.to_numeric(rpci_mean, errors="coerce")
    all_df.loc[ordered.index, "horse_rpci_std_past5"] = pd.to_numeric(rpci_std, errors="coerce")
    all_df.loc[ordered.index, "horse_rpci_count_past5"] = pd.to_numeric(rpci_count, errors="coerce")

    all_df["horse_rpci_mean_past5"] = all_df["horse_rpci_mean_past5"].fillna(num(all_df, "前走RPCI"))
    all_df["horse_rpci_std_past5"] = all_df["horse_rpci_std_past5"].fillna(3.0)
    all_df["horse_rpci_count_past5"] = all_df["horse_rpci_count_past5"].fillna(0.0)
    all_df["horse_expected_rpci_abs_gap"] = (
        all_df["expected_rpci"] - all_df["horse_rpci_mean_past5"].fillna(all_df["expected_rpci"])
    ).abs()
    reliability = (all_df["horse_rpci_count_past5"].clip(upper=5.0) / 5.0).fillna(0.0)
    all_df["horse_expected_rpci_fit_score"] = (
        (1.0 - all_df["horse_expected_rpci_abs_gap"] / 8.0).clip(-0.5, 1.0) * (0.35 + 0.65 * reliability)
    ).fillna(0.0)

    fast_score = num(all_df, "horse_fast_lap_score_past5", 0.0).fillna(0.0)
    slow_score = num(all_df, "horse_slow_lap_score_past5", 0.0).fillna(0.0)
    instant_score = num(all_df, "horse_instant_lap_score_past5", 0.0).fillna(0.0)
    sustain_score = num(all_df, "horse_sustain_lap_score_past5", 0.0).fillna(0.0)
    long_score = num(all_df, "horse_long_spurt_lap_score_past5", 0.0).fillna(0.0)
    pace_fit = num(all_df, "pace_fit_score", 0.0).fillna(0.0)

    all_df["expected_fast_lap_fit_score"] = all_df["expected_lap_fast_weight"] * fast_score
    all_df["expected_slow_lap_fit_score"] = all_df["expected_lap_slow_weight"] * slow_score
    all_df["expected_instant_lap_fit_score"] = (0.65 * all_df["expected_lap_slow_weight"] + 0.20 * pace_fit) * instant_score
    all_df["expected_sustain_lap_fit_score"] = all_df["expected_lap_sustain_weight"] * sustain_score
    all_df["expected_long_spurt_lap_fit_score"] = (0.55 * collapse + 0.25 * all_df["expected_lap_sustain_weight"]) * long_score
    all_df["expected_lap_shape_fit_score"] = (
        all_df["expected_fast_lap_fit_score"]
        + all_df["expected_slow_lap_fit_score"]
        + all_df["expected_instant_lap_fit_score"]
        + all_df["expected_sustain_lap_fit_score"]
        + all_df["expected_long_spurt_lap_fit_score"]
    ).fillna(0.0)
    all_df["expected_lap_total_fit_score"] = (
        0.45 * all_df["expected_lap_shape_fit_score"]
        + 0.35 * all_df["horse_expected_rpci_fit_score"]
        + 0.20 * pace_fit
    ).fillna(0.0)

    front = num(all_df, "horse_front_run_rate_past5", 0.0).fillna(num(all_df, "front_running_tendency", 0.0)).fillna(0.0)
    closer = num(all_df, "horse_closer_rate_past5", 0.0).fillna(num(all_df, "closing_tendency", 0.0)).fillna(0.0)
    all_df["expected_lap_front_adversity_score"] = (collapse * front - slow * front * 0.35).fillna(0.0)
    all_df["expected_lap_closer_adversity_score"] = (slow * closer - collapse * closer * 0.25).fillna(0.0)

    all_df = all_df.sort_values("_row_id", kind="mergesort")
    out_train = all_df[all_df["_split"].eq("train")].drop(columns=["_split", "_row_id"], errors="ignore")
    out_test = all_df[all_df["_split"].eq("test")].drop(columns=["_split", "_row_id"], errors="ignore")
    diagnostics = races[[RACE_COL, DATE_COL, "Ｒ", "場所", "芝・ダ", "距離", "馬場状態", "RPCI", "PCI3", "course_base_rpci_prior", "course_base_pci3_prior", "course_base_rpci_count"]]
    return out_train, out_test, diagnostics


def fit_ranker(train: pd.DataFrame, numeric: list[str], categorical: list[str], alpha: float, top_k: int) -> SimpleRaceRanker:
    return SimpleRaceRanker(
        numeric_features=numeric,
        categorical_features=categorical,
        categorical_top_k=top_k,
        ridge_alpha=alpha,
    ).fit(train, "target_score")


def race_z(values: pd.Series, race_ids: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    mean = values.groupby(race_ids).transform("mean")
    std = values.groupby(race_ids).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def optimize_overlay(train: pd.DataFrame, test: pd.DataFrame, base_model: SimpleRaceRanker) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_train = base_model.predict(train)
    base_test = base_model.predict(test)
    fit_train = race_z(train["expected_lap_total_fit_score"], train[RACE_COL]).to_numpy()
    adv_train = race_z(train["expected_lap_front_adversity_score"] + train["expected_lap_closer_adversity_score"], train[RACE_COL]).to_numpy()
    fit_test = race_z(test["expected_lap_total_fit_score"], test[RACE_COL]).to_numpy()
    adv_test = race_z(test["expected_lap_front_adversity_score"] + test["expected_lap_closer_adversity_score"], test[RACE_COL]).to_numpy()

    rows = []
    best = None
    for fit_w in np.arange(-0.04, 0.081, 0.01):
        for adv_w in np.arange(-0.03, 0.041, 0.01):
            train_scores = base_train + fit_w * fit_train + adv_w * adv_train
            metrics = metric_summary(train, train_scores)
            score = (
                metrics["top1_win_rate"] * 100
                + metrics["top1_top3_rate"] * 35
                + metrics["top1_win_roi"] * 18
                + metrics["top1_place_roi"] * 12
                - metrics["winner_mean_ai_rank"] * 0.8
            )
            row = {"fit_weight": float(fit_w), "adversity_weight": float(adv_w), "selection_score": float(score), **metrics}
            rows.append(row)
            if best is None or score > best["selection_score"]:
                best = row

    assert best is not None
    test_scores = base_test + best["fit_weight"] * fit_test + best["adversity_weight"] * adv_test
    test_metrics = metric_summary(test, test_scores)
    best = {**best, **{f"test_{k}": v for k, v in test_metrics.items()}}
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False), best


def coefficient_importance(model: SimpleRaceRanker, label: str) -> pd.DataFrame:
    rows = []
    for feature, coef in zip(model.feature_names_ or [], model.coefficients_ if model.coefficients_ is not None else []):
        rows.append({"model": label, "feature": feature, "coef": float(coef), "abs_coef": abs(float(coef))})
    return pd.DataFrame(rows)


def bet_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) == 0:
        return {
            "bets": 0,
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
        "win_rate": float(frame["target_win"].mean()),
        "top3_rate": float(frame["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (len(frame) * 100.0)),
        "place_roi": float(place_pay.sum() / (len(frame) * 100.0)),
        "avg_popularity": float(num(frame, "人気").mean()),
        "avg_odds": float(num(frame, "単勝オッズ").mean()),
    }


def expected_lap_segments(train: pd.DataFrame, test: pd.DataFrame, base_scores: np.ndarray, plus_scores: np.ndarray) -> pd.DataFrame:
    q = {
        "fit_hi": float(num(train, "expected_lap_total_fit_score").quantile(0.75)),
        "fit_lo": float(num(train, "expected_lap_total_fit_score").quantile(0.25)),
        "rpci_fit_hi": float(num(train, "horse_expected_rpci_fit_score").quantile(0.75)),
        "closer_adv_hi": float(num(train, "expected_lap_closer_adversity_score").quantile(0.75)),
        "front_adv_hi": float(num(train, "expected_lap_front_adversity_score").quantile(0.75)),
    }
    scored = test.copy()
    scored["base_score"] = base_scores
    scored["plus_score"] = plus_scores
    scored["base_rank"] = scored.groupby(RACE_COL)["base_score"].rank(ascending=False, method="first").astype(int)
    scored["plus_rank"] = scored.groupby(RACE_COL)["plus_score"].rank(ascending=False, method="first").astype(int)

    checks: list[tuple[str, pd.Series]] = [
        ("plus_top1", scored["plus_rank"].eq(1)),
        ("plus_top1_fit_hi", scored["plus_rank"].eq(1) & num(scored, "expected_lap_total_fit_score").ge(q["fit_hi"])),
        ("plus_top1_fit_lo", scored["plus_rank"].eq(1) & num(scored, "expected_lap_total_fit_score").le(q["fit_lo"])),
        ("plus_top1_rpci_fit_hi", scored["plus_rank"].eq(1) & num(scored, "horse_expected_rpci_fit_score").ge(q["rpci_fit_hi"])),
        ("plus_top1_closer_adversity_hi", scored["plus_rank"].eq(1) & num(scored, "expected_lap_closer_adversity_score").ge(q["closer_adv_hi"])),
        ("plus_top1_front_adversity_hi", scored["plus_rank"].eq(1) & num(scored, "expected_lap_front_adversity_score").ge(q["front_adv_hi"])),
        ("plus_top1_fast_expected", scored["plus_rank"].eq(1) & scored["expected_lap_type"].eq("fast")),
        ("plus_top1_middle_expected", scored["plus_rank"].eq(1) & scored["expected_lap_type"].eq("middle")),
        ("plus_top1_slow_expected", scored["plus_rank"].eq(1) & scored["expected_lap_type"].eq("slow")),
        ("new_top1_changed_from_base", scored["plus_rank"].eq(1) & scored["base_rank"].ne(1)),
        ("base_top1_lost_by_plus", scored["base_rank"].eq(1) & scored["plus_rank"].ne(1)),
        ("plus_top3_pop5plus_fit_hi", scored["plus_rank"].le(3) & num(scored, "人気").ge(5) & num(scored, "expected_lap_total_fit_score").ge(q["fit_hi"])),
        ("plus_top3_pop5plus_rpci_fit_hi", scored["plus_rank"].le(3) & num(scored, "人気").ge(5) & num(scored, "horse_expected_rpci_fit_score").ge(q["rpci_fit_hi"])),
    ]
    rows = []
    for name, mask in checks:
        rows.append({"segment": name, **bet_metrics(scored[mask])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate expected RPCI/lap scenario features.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--write-feature-csv", action="store_true", help="Write full enriched train/test feature CSVs.")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(project_path(args.train_csv), low_memory=False)
    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    base_model: SimpleRaceRanker = pickle.load(project_path(args.base_model).open("rb"))

    train_x, test_x, race_diag = add_expected_lap_features(train, test)
    if args.write_feature_csv:
        train_x.to_csv(out_dir / "train_features_expected_lap.csv", index=False, encoding="utf-8-sig")
        test_x.to_csv(out_dir / "test_features_expected_lap.csv", index=False, encoding="utf-8-sig")
    race_diag.to_csv(out_dir / "race_expected_lap_diagnostics.csv", index=False, encoding="utf-8-sig")

    base_numeric = list(base_model.numeric_features)
    base_categorical = list(base_model.categorical_features)
    plus_numeric = base_numeric + [col for col in NEW_NUMERIC_FEATURES if col not in base_numeric]
    plus_categorical = base_categorical + [col for col in NEW_CATEGORICAL_FEATURES if col not in base_categorical]

    alpha = float(base_model.ridge_alpha)
    top_k = int(base_model.categorical_top_k)
    base_retrained = fit_ranker(train_x, base_numeric, base_categorical, alpha, top_k)
    plus_model = fit_ranker(train_x, plus_numeric, plus_categorical, alpha, top_k)
    base_test_scores = base_retrained.predict(test_x)
    plus_test_scores = plus_model.predict(test_x)

    rows = [
        {"variant": "saved_model_reference", **metric_summary(test_x, base_model.predict(test_x))},
        {"variant": "base_retrained_same_features", **metric_summary(test_x, base_test_scores)},
        {"variant": "expected_lap_features_retrained", **metric_summary(test_x, plus_test_scores)},
    ]

    overlay_grid, overlay_best = optimize_overlay(train_x, test_x, base_retrained)
    rows.append(
        {
            "variant": "expected_lap_overlay_on_base_retrained",
            "fit_weight": overlay_best["fit_weight"],
            "adversity_weight": overlay_best["adversity_weight"],
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

    summary.to_csv(out_dir / "expected_lap_rpci_summary.csv", index=False, encoding="utf-8-sig")
    overlay_grid.to_csv(out_dir / "expected_lap_overlay_grid_train.csv", index=False, encoding="utf-8-sig")
    imps = pd.concat(
        [
            coefficient_importance(base_retrained, "base_retrained_same_features"),
            coefficient_importance(plus_model, "expected_lap_features_retrained"),
        ],
        ignore_index=True,
    )
    imps.to_csv(out_dir / "expected_lap_rpci_importance.csv", index=False, encoding="utf-8-sig")
    segments = expected_lap_segments(train_x, test_x, base_test_scores, plus_test_scores)
    segments.to_csv(out_dir / "expected_lap_segments.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "overlay_best.json").open("w", encoding="utf-8") as f:
        json.dump(overlay_best, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_dir": str(out_dir), "new_features": NEW_NUMERIC_FEATURES + NEW_CATEGORICAL_FEATURES}, ensure_ascii=False, indent=2))
    print(
        summary[
            [
                "variant",
                "top1_win_rate",
                "top1_top3_rate",
                "top1_win_roi",
                "top1_place_roi",
                "top3_contains_winner_rate",
                "top3_win_roi",
                "top3_place_roi",
                "winner_mean_ai_rank",
            ]
        ].to_string(index=False)
    )
    print("\nExpected-lap coefficients:")
    print(
        imps[
            imps["model"].eq("expected_lap_features_retrained")
            & imps["feature"].isin(NEW_NUMERIC_FEATURES + [f"expected_lap_type={x}" for x in ["fast", "middle", "slow"]])
        ]
        .sort_values("abs_coef", ascending=False)
        .head(30)
        .to_string(index=False)
    )
    print("\nExpected-lap segments:")
    print(segments.to_string(index=False))


if __name__ == "__main__":
    main()
