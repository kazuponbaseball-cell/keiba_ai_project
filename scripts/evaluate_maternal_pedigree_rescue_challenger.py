from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_pedigree_asof_pure_prior import (  # noqa: E402
    json_ready,
    num,
    payout_metrics,
    project_path,
    read_needed,
    strict_group_history,
)
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


DEFAULT_HISTORY_TRAIN = "data/datasets/cache/target_pedigree_interactions_confirmed_opponent/train_features.csv"
DEFAULT_HISTORY_TEST = "data/datasets/cache/target_pedigree_interactions_confirmed_opponent/test_features.csv"
DEFAULT_SCORE_TEST = (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
    "test_features.csv"
)
DEFAULT_MODEL = "models/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/baseline_ranker.pkl"
DEFAULT_OUT = "outputs/analysis/maternal_pedigree_rescue_challenger_v1"
DEFAULT_BLOODLINE_PRIOR_SCORES = "outputs/analysis/bloodline_asof_pure_prior_audit_v1/test_pure_prior_scores.csv"


def zscore_by_race(values: pd.Series, race_ids: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").fillna(0.0)
    mean = x.groupby(race_ids).transform("mean")
    std = x.groupby(race_ids).transform("std").replace(0, np.nan)
    return ((x - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def norm01(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo = x.quantile(0.05)
    hi = x.quantile(0.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.0)


def add_unique_prior_counts(
    frame: pd.DataFrame,
    *,
    group_col: str,
    prefix: str,
    date_col: str = "日付",
    race_col: str = "レースID(新/馬番無)",
    horse_col: str = "血統登録番号",
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out[f"{prefix}_pure_unique_horses"] = 0.0
    if group_col not in frame.columns:
        return out

    valid = frame[[group_col, date_col, race_col, horse_col]].dropna(subset=[group_col]).copy()
    if valid.empty:
        return out
    valid["_row_index_for_unique_prior"] = valid.index
    valid = valid.sort_values([group_col, date_col, race_col], kind="mergesort")

    result = pd.Series(0.0, index=valid.index, dtype=float)
    for _group, part in valid.groupby(group_col, sort=False):
        seen: set[str] = set()
        for (_date, _race), race_part in part.groupby([date_col, race_col], sort=False):
            race_horses = set(race_part[horse_col].astype(str))
            count_for_rows = [len(seen.difference({str(h)})) for h in race_part[horse_col]]
            result.loc[race_part.index] = count_for_rows
            seen.update(race_horses)
    out.loc[valid.index, f"{prefix}_pure_unique_horses"] = result
    return out


def add_maternal_priors(frame: pd.DataFrame, *, prior_strength: float, include_unique_counts: bool = False) -> pd.DataFrame:
    out = frame.copy()
    specs = [
        (["母馬"], "dam"),
        (["母馬", "芝・ダ"], "dam_surface"),
        (["母馬", "distance_category"], "dam_distance"),
        (["母馬", "馬場状態"], "dam_going"),
        (["母母馬"], "second_dam"),
        (["母母馬", "芝・ダ"], "second_dam_surface"),
        (["母母馬", "distance_category"], "second_dam_distance"),
        (["母母馬", "馬場状態"], "second_dam_going"),
    ]
    for group_cols, prefix in specs:
        if all(c in out.columns for c in group_cols):
            hist = strict_group_history(
                out,
                group_cols,
                prefix,
                date_col="日付",
                race_col="レースID(新/馬番無)",
                horse_col="血統登録番号",
                prior_strength=prior_strength,
            )
            out = pd.concat([out, hist], axis=1)

    if include_unique_counts:
        out = pd.concat(
            [
                out,
                add_unique_prior_counts(out, group_col="母馬", prefix="maternal_sibling"),
                add_unique_prior_counts(out, group_col="母母馬", prefix="second_dam_family"),
            ],
            axis=1,
        )
    else:
        out["maternal_sibling_pure_unique_horses"] = 0.0
        out["second_dam_family_pure_unique_horses"] = 0.0

    out["dam_surface_lift"] = num(out, "dam_surface_pure_top3_eb", 0) - num(out, "dam_pure_top3_eb", 0)
    out["dam_distance_lift"] = num(out, "dam_distance_pure_top3_eb", 0) - num(out, "dam_pure_top3_eb", 0)
    out["dam_going_lift"] = num(out, "dam_going_pure_top3_eb", 0) - num(out, "dam_pure_top3_eb", 0)
    out["second_dam_surface_lift"] = num(out, "second_dam_surface_pure_top3_eb", 0) - num(out, "second_dam_pure_top3_eb", 0)
    out["second_dam_distance_lift"] = num(out, "second_dam_distance_pure_top3_eb", 0) - num(out, "second_dam_pure_top3_eb", 0)
    out["second_dam_going_lift"] = num(out, "second_dam_going_pure_top3_eb", 0) - num(out, "second_dam_pure_top3_eb", 0)
    out["maternal_family_lift_fit_score"] = (
        0.32 * out["dam_surface_lift"]
        + 0.28 * out["dam_distance_lift"]
        + 0.18 * out["dam_going_lift"]
        + 0.10 * out["second_dam_surface_lift"]
        + 0.08 * out["second_dam_distance_lift"]
        + 0.04 * out["second_dam_going_lift"]
    ).fillna(0.0)
    out["maternal_family_lower_bound_score"] = (
        0.42 * num(out, "dam_surface_pure_top3_lower10", 0)
        + 0.28 * num(out, "dam_distance_pure_top3_lower10", 0)
        + 0.15 * num(out, "second_dam_surface_pure_top3_lower10", 0)
        + 0.15 * num(out, "second_dam_distance_pure_top3_lower10", 0)
    ).fillna(0.0)
    out["maternal_family_reliability_score"] = (
        0.60 * (num(out, "dam_pure_starts", 0).clip(upper=20) / 20.0)
        + 0.25 * (num(out, "second_dam_pure_starts", 0).clip(upper=50) / 50.0)
        + 0.15 * (num(out, "maternal_sibling_pure_unique_horses", 0).clip(upper=4) / 4.0)
    ).fillna(0.0)
    out["maternal_rescue_fit_score"] = (
        out["maternal_family_lift_fit_score"] * (0.50 + 0.50 * out["maternal_family_reliability_score"])
    ).fillna(0.0)
    return out


def first_condition_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    career = num(out, "キャリア", 99).fillna(99)
    surface = out.get("芝・ダ", pd.Series("", index=out.index)).astype(str)
    turf_starts = num(out, "horse_turf_starts", 0).fillna(0)
    dirt_starts = num(out, "horse_dirt_starts", 0).fillna(0)
    current_surface_starts = pd.Series(0.0, index=out.index)
    current_surface_starts.loc[surface.str.contains("芝", regex=False)] = turf_starts.loc[
        surface.str.contains("芝", regex=False)
    ]
    current_surface_starts.loc[surface.str.contains("ダ", regex=False)] = dirt_starts.loc[
        surface.str.contains("ダ", regex=False)
    ]
    same_distance = num(out, "same_distance_category_starts", 99).fillna(99)
    out["fc_low_career_flag"] = career.le(3).astype(int)
    out["fc_very_low_career_flag"] = career.le(2).astype(int)
    out["fc_first_surface_flag"] = current_surface_starts.le(0).astype(int)
    out["fc_low_surface_sample_flag"] = current_surface_starts.le(1).astype(int)
    out["fc_first_distance_category_flag"] = same_distance.le(0).astype(int)
    out["fc_any_first_condition_flag"] = out[
        ["fc_first_surface_flag", "fc_first_distance_category_flag"]
    ].max(axis=1)
    out["fc_condition_uncertainty_score"] = (
        0.35 * out["fc_low_career_flag"]
        + 0.30 * out["fc_first_surface_flag"]
        + 0.25 * out["fc_first_distance_category_flag"]
        + 0.10 * out["fc_low_surface_sample_flag"]
    ).clip(0.0, 1.0)
    return out


def add_market_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    race_col = "レースID(新/馬番無)"
    odds = num(out, "単勝オッズ", np.nan)
    implied = (1.0 / odds.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    implied_sum = implied.groupby(out[race_col]).transform("sum").replace(0, np.nan)
    out["market_win_prob_norm"] = (implied / implied_sum).fillna(0.0)
    out["ai_score_z"] = zscore_by_race(num(out, "ai_score", 0), out[race_col])
    out["market_prob_z"] = zscore_by_race(out["market_win_prob_norm"], out[race_col])
    out["ai_market_residual_z"] = out["ai_score_z"] - out["market_prob_z"]
    out["ai_rank_market_residual"] = num(out, "人気", 99).fillna(99) - num(out, "ai_rank", 99).fillna(99)
    return out


def segment_rows(test: pd.DataFrame) -> list[dict[str, Any]]:
    q_maternal_hi = float(num(test, "maternal_rescue_fit_score", 0).quantile(0.75))
    q_maternal_top = float(num(test, "maternal_rescue_fit_score", 0).quantile(0.90))
    q_maternal_lower_hi = float(num(test, "maternal_family_lower_bound_score", 0).quantile(0.75))
    q_combined_hi = float(num(test, "combined_pedigree_rescue_score", 0).quantile(0.75))
    surface = test.get("芝・ダ", pd.Series("", index=test.index)).astype(str)
    going = test.get("馬場状態", pd.Series("", index=test.index)).astype(str)
    age = num(test, "年齢", 0)
    ai_top1 = num(test, "ai_rank", 99).eq(1)
    ai_top3 = num(test, "ai_rank", 99).le(3)
    ai_top5 = num(test, "ai_rank", 99).le(5)
    ai_2to5 = num(test, "ai_rank", 99).between(2, 5)
    dirt = surface.str.contains("ダ", regex=False)
    turf = surface.str.contains("芝", regex=False)
    wetish = going.str.contains("稍|重|不", regex=True)
    low_career = num(test, "fc_low_career_flag", 0).eq(1)
    first_condition = num(test, "fc_any_first_condition_flag", 0).eq(1)
    residual_pos = num(test, "ai_market_residual_z", 0).ge(0)
    residual_rank_pos = num(test, "ai_rank_market_residual", 0).ge(1)
    maternal_hi = num(test, "maternal_rescue_fit_score", 0).ge(q_maternal_hi)
    maternal_top = num(test, "maternal_rescue_fit_score", 0).ge(q_maternal_top)
    maternal_lower_hi = num(test, "maternal_family_lower_bound_score", 0).ge(q_maternal_lower_hi)
    combined_hi = num(test, "combined_pedigree_rescue_score", 0).ge(q_combined_hi)
    specs = [
        ("ai_top1_maternal_hi_age2_dirt", ai_top1 & age.eq(2) & dirt & maternal_hi),
        ("ai_top1_maternal_top_age2_dirt", ai_top1 & age.eq(2) & dirt & maternal_top),
        ("ai_top3_maternal_hi_age2_dirt", ai_top3 & age.eq(2) & dirt & maternal_hi),
        ("ai_top5_maternal_hi_low_career_dirt", ai_top5 & low_career & dirt & maternal_hi),
        ("ai_2to5_maternal_hi_low_career_dirt_value", ai_2to5 & low_career & dirt & maternal_hi & residual_rank_pos),
        ("ai_top5_maternal_top_low_career_dirt_value", ai_top5 & low_career & dirt & maternal_top & residual_rank_pos),
        ("ai_top5_maternal_lower_hi_first_condition", ai_top5 & first_condition & maternal_lower_hi),
        ("ai_top5_combined_hi_first_condition_value", ai_top5 & first_condition & combined_hi & residual_rank_pos),
        ("ai_top5_combined_hi_low_career_value", ai_top5 & low_career & combined_hi & residual_rank_pos),
        ("ai_top3_combined_hi_wetish", ai_top3 & wetish & combined_hi),
        ("ai_top5_maternal_hi_wetish_dirt_value", ai_top5 & wetish & dirt & maternal_hi & residual_rank_pos),
        ("ai_top5_maternal_hi_turf_value", ai_top5 & turf & maternal_hi & residual_rank_pos),
        ("ai_top5_maternal_hi_market_residual_pos", ai_top5 & maternal_hi & residual_pos),
    ]
    rows = []
    for label, mask in specs:
        rows.append({"segment": label, **payout_metrics(test[mask].copy())})
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}" if math.isfinite(value) else ""
            vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate maternal pedigree rescue challenger segments.")
    parser.add_argument("--history-train-csv", default=DEFAULT_HISTORY_TRAIN)
    parser.add_argument("--history-test-csv", default=DEFAULT_HISTORY_TEST)
    parser.add_argument("--score-test-csv", default=DEFAULT_SCORE_TEST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--bloodline-prior-score-csv", default=DEFAULT_BLOODLINE_PRIOR_SCORES)
    parser.add_argument("--prior-strength", type=float, default=30.0)
    parser.add_argument("--include-unique-counts", action="store_true")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model: SimpleRaceRanker = pickle.load(project_path(args.model).open("rb"))
    extra = list(model.numeric_features) + list(model.categorical_features)
    history_train = read_needed(project_path(args.history_train_csv), [])
    history_test = read_needed(project_path(args.history_test_csv), [])
    history_train["_split"] = "train"
    history_test["_split"] = "test"
    history = pd.concat([history_train, history_test], ignore_index=True)
    history["レースID(新/馬番無)"] = history["レースID(新/馬番無)"].astype(str)
    history["血統登録番号"] = history["血統登録番号"].astype(str)
    history = add_maternal_priors(
        history, prior_strength=args.prior_strength, include_unique_counts=args.include_unique_counts
    )
    history_test_enriched = history[history["_split"] == "test"].copy()
    transfer_cols = [
        c
        for c in history_test_enriched.columns
        if c.startswith("pure_")
        or c.startswith("dam_")
        or c.startswith("second_dam_")
        or c.startswith("maternal_")
    ]
    transfer = history_test_enriched[
        ["レースID(新/馬番無)", "血統登録番号", *transfer_cols]
    ].drop_duplicates(["レースID(新/馬番無)", "血統登録番号"], keep="last")

    score_test = read_needed(project_path(args.score_test_csv), extra)
    score_test["レースID(新/馬番無)"] = score_test["レースID(新/馬番無)"].astype(str)
    score_test["血統登録番号"] = score_test["血統登録番号"].astype(str)
    test = score_test.merge(transfer, on=["レースID(新/馬番無)", "血統登録番号"], how="left")
    bloodline_prior_path = project_path(args.bloodline_prior_score_csv)
    if bloodline_prior_path.exists():
        blood = pd.read_csv(bloodline_prior_path, encoding="utf-8-sig", low_memory=False)
        blood["レースID(新/馬番無)"] = blood["レースID(新/馬番無)"].astype(str)
        blood_cols = [
            c
            for c in [
                "レースID(新/馬番無)",
                "馬名",
                "pure_bloodline_lift_fit_score",
                "pure_bloodline_lower_bound_score",
            ]
            if c in blood.columns
        ]
        if {"レースID(新/馬番無)", "馬名"}.issubset(blood_cols):
            blood = blood[blood_cols].drop_duplicates(["レースID(新/馬番無)", "馬名"], keep="last")
            test = test.merge(blood, on=["レースID(新/馬番無)", "馬名"], how="left", suffixes=("", "__blood"))
    test["ai_score"] = model.predict(test)
    test["ai_rank"] = test.groupby("レースID(新/馬番無)")["ai_score"].rank(ascending=False, method="first").astype(int)
    test = first_condition_flags(test)
    test = add_market_residuals(test)
    test["combined_pedigree_rescue_score"] = (
        0.45 * norm01(num(test, "maternal_rescue_fit_score", 0))
        + 0.25 * norm01(num(test, "pure_bloodline_lift_fit_score", 0))
        + 0.15 * norm01(num(test, "maternal_family_lower_bound_score", 0))
        + 0.15 * norm01(num(test, "ai_market_residual_z", 0))
    ).fillna(0.0)
    test["low_career_pedigree_rescue_score"] = (
        test["combined_pedigree_rescue_score"] * (0.50 + 0.50 * num(test, "fc_condition_uncertainty_score", 0))
    ).fillna(0.0)

    rows = segment_rows(test)
    summary = {
        "output_dir": str(out_dir),
        "history_train_rows": int(len(history_train)),
        "history_test_rows": int(len(history_test)),
        "score_test_rows": int(len(score_test)),
        "prior_strength": float(args.prior_strength),
        "segments": rows,
        "notes": [
            "Maternal features are race-boundary as-of and subtract the target horse's own prior records.",
            "This is a shadow challenger; production BUY logic is not changed.",
            "Use positive segments as candidates for T-5/T-3 odds-survival shadowing, not immediate live BUY expansion.",
        ],
    }
    pd.DataFrame(rows).to_csv(out_dir / "segment_summary.csv", index=False, encoding="utf-8-sig")
    keep_cols = [
        "日付",
        "場所",
        "レースID(新/馬番無)",
        "馬名",
        "年齢",
        "芝・ダ",
        "距離",
        "馬場状態",
        "キャリア",
        "人気",
        "単勝オッズ",
        "ai_rank",
        "maternal_rescue_fit_score",
        "maternal_family_lower_bound_score",
        "maternal_family_reliability_score",
        "maternal_sibling_pure_unique_horses",
        "second_dam_family_pure_unique_horses",
        "pure_bloodline_lift_fit_score",
        "combined_pedigree_rescue_score",
        "low_career_pedigree_rescue_score",
        "ai_market_residual_z",
        "ai_rank_market_residual",
        "fc_condition_uncertainty_score",
        "target_win",
        "target_top3",
        "単勝配当",
        "複勝配当",
    ]
    test[[c for c in keep_cols if c in test.columns]].to_csv(
        out_dir / "runner_scores.csv", index=False, encoding="utf-8-sig"
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(summary), f, ensure_ascii=False, indent=2)

    review = [
        "# Maternal pedigree rescue challenger",
        "",
        "## Segment results",
        "",
        markdown_table(rows),
        "",
        "## Reading",
        "",
        "- Positive segments are only challenger candidates.",
        "- Prefer segments with enough bets/races and both win/place ROI not collapsing.",
        "- Low-career expansion should remain shadow-only until T-5/T-3 odds-survival is added.",
    ]
    (out_dir / "review.md").write_text("\n".join(review), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
