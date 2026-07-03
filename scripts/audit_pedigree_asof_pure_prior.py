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

from src.data.loaders import load_json_config  # noqa: E402
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HISTORY_TRAIN = "data/datasets/cache/target_pedigree_interactions_confirmed_opponent/train_features.csv"
DEFAULT_HISTORY_TEST = "data/datasets/cache/target_pedigree_interactions_confirmed_opponent/test_features.csv"
DEFAULT_SCORE_TEST = (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
    "test_features.csv"
)
DEFAULT_CONFIG = "config/baseline_features_workout.json"
DEFAULT_MODEL = "models/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/baseline_ranker.pkl"
DEFAULT_OUT = "outputs/analysis/bloodline_asof_pure_prior_audit_v1"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    values = frame[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if pd.isna(value):
        return None
    return value


def beta_binomial_mean_lower(
    wins: pd.Series,
    starts: pd.Series,
    *,
    base_rate: float,
    prior_strength: float,
) -> tuple[pd.Series, pd.Series]:
    alpha = max(base_rate * prior_strength, 1e-6)
    beta = max((1.0 - base_rate) * prior_strength, 1e-6)
    a = wins.astype(float).fillna(0.0) + alpha
    b = (starts.astype(float).fillna(0.0) - wins.astype(float).fillna(0.0)).clip(lower=0.0) + beta
    total = a + b
    mean = a / total
    var = (a * b) / ((total**2) * (total + 1.0))
    lower10 = (mean - 1.2815515655446004 * np.sqrt(var)).clip(lower=0.0, upper=1.0)
    return mean.astype(float), lower10.astype(float)


def read_needed(path: Path, extra_cols: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    needed = [
        "日付",
        "レースID(新/馬番無)",
        "血統登録番号",
        "馬名",
        "年齢",
        "人気",
        "単勝オッズ",
        "単勝配当",
        "複勝配当",
        "確定着順",
        "target_win",
        "target_top3",
        "target_score",
        "場所",
        "芝・ダ",
        "距離",
        "馬場状態",
        "キャリア",
        "distance_category",
        "種牡馬",
        "母父馬",
        "母馬",
        "母母馬",
        "父父馬",
        "父母馬",
        "sire_starts",
        "sire_top3_rate",
        "sire_surface_top3_rate",
        "sire_distance_top3_rate",
        "sire_going_top3_rate",
        "bms_starts",
        "bms_top3_rate",
        "bms_surface_top3_rate",
        "bms_distance_top3_rate",
        "bms_going_top3_rate",
        "sire_bms_pair_starts",
        "sire_bms_pair_top3_rate",
        "bloodline_lift_fit_score",
        "bloodline_high_confidence_fit_score",
        "same_distance_category_starts",
        "horse_turf_starts",
        "horse_dirt_starts",
    ]
    usecols = [c for c in dict.fromkeys([*needed, *extra_cols]) if c in header]
    out = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    out["_source_path"] = str(path)
    return out


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}" if math.isfinite(value) else ""
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def strict_group_history(
    frame: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
    *,
    date_col: str,
    race_col: str,
    horse_col: str,
    prior_strength: float,
) -> pd.DataFrame:
    valid = frame.dropna(subset=group_cols).copy()
    if valid.empty:
        return pd.DataFrame(index=frame.index)

    key_cols = [*group_cols, date_col, race_col]
    race_agg = (
        valid.groupby(key_cols, dropna=False)
        .agg(
            entries=("target_top3", "size"),
            wins=("target_win", "sum"),
            top3=("target_top3", "sum"),
            score=("target_score", "sum"),
        )
        .reset_index()
        .sort_values([*group_cols, date_col, race_col], kind="mergesort")
    )
    grouped = race_agg.groupby(group_cols, dropna=False, sort=False)
    for source in ["entries", "wins", "top3", "score"]:
        race_agg[f"prior_{source}"] = grouped[source].cumsum() - race_agg[source]

    mapped = frame[[*group_cols, date_col, race_col, horse_col]].merge(
        race_agg[[*key_cols, "prior_entries", "prior_wins", "prior_top3", "prior_score"]],
        on=key_cols,
        how="left",
    )
    starts = mapped["prior_entries"].fillna(0.0)
    wins = mapped["prior_wins"].fillna(0.0)
    top3 = mapped["prior_top3"].fillna(0.0)
    score = mapped["prior_score"].fillna(0.0)

    out = pd.DataFrame(index=frame.index)
    out[f"{prefix}_strict_starts"] = starts.astype(float)
    out[f"{prefix}_strict_win_rate"] = (wins / starts.replace(0, np.nan)).fillna(0.0).astype(float)
    out[f"{prefix}_strict_top3_rate"] = (top3 / starts.replace(0, np.nan)).fillna(0.0).astype(float)
    out[f"{prefix}_strict_avg_score"] = (score / starts.replace(0, np.nan)).fillna(0.0).astype(float)

    pure_key_cols = [*group_cols, horse_col, date_col, race_col]
    horse_agg = (
        valid.groupby(pure_key_cols, dropna=False)
        .agg(
            entries=("target_top3", "size"),
            wins=("target_win", "sum"),
            top3=("target_top3", "sum"),
            score=("target_score", "sum"),
        )
        .reset_index()
        .sort_values([*group_cols, horse_col, date_col, race_col], kind="mergesort")
    )
    horse_grouped = horse_agg.groupby([*group_cols, horse_col], dropna=False, sort=False)
    for source in ["entries", "wins", "top3", "score"]:
        horse_agg[f"prior_{source}"] = horse_grouped[source].cumsum() - horse_agg[source]

    mapped_horse = frame[[*group_cols, horse_col, date_col, race_col]].merge(
        horse_agg[[*pure_key_cols, "prior_entries", "prior_wins", "prior_top3", "prior_score"]],
        on=pure_key_cols,
        how="left",
    )
    pure_starts = (starts - mapped_horse["prior_entries"].fillna(0.0)).clip(lower=0.0)
    pure_wins = (wins - mapped_horse["prior_wins"].fillna(0.0)).clip(lower=0.0)
    pure_top3 = (top3 - mapped_horse["prior_top3"].fillna(0.0)).clip(lower=0.0)
    pure_score = (score - mapped_horse["prior_score"].fillna(0.0)).clip(lower=0.0)
    global_win = float(valid["target_win"].mean())
    global_top3 = float(valid["target_top3"].mean())
    win_eb, win_lower = beta_binomial_mean_lower(
        pure_wins, pure_starts, base_rate=global_win, prior_strength=prior_strength
    )
    top3_eb, top3_lower = beta_binomial_mean_lower(
        pure_top3, pure_starts, base_rate=global_top3, prior_strength=prior_strength
    )

    out[f"{prefix}_pure_starts"] = pure_starts.astype(float)
    out[f"{prefix}_pure_win_rate"] = (pure_wins / pure_starts.replace(0, np.nan)).fillna(0.0).astype(float)
    out[f"{prefix}_pure_top3_rate"] = (pure_top3 / pure_starts.replace(0, np.nan)).fillna(0.0).astype(float)
    out[f"{prefix}_pure_avg_score"] = (pure_score / pure_starts.replace(0, np.nan)).fillna(0.0).astype(float)
    out[f"{prefix}_pure_win_eb"] = win_eb
    out[f"{prefix}_pure_win_lower10"] = win_lower
    out[f"{prefix}_pure_top3_eb"] = top3_eb
    out[f"{prefix}_pure_top3_lower10"] = top3_lower
    out[f"{prefix}_pure_effective_n"] = (pure_starts + prior_strength).astype(float)
    return out


def add_pure_priors(frame: pd.DataFrame, *, prior_strength: float) -> pd.DataFrame:
    date_col = "日付"
    race_col = "レースID(新/馬番無)"
    horse_col = "血統登録番号"
    out = frame.copy()
    specs = [
        (["種牡馬"], "sire"),
        (["種牡馬", "芝・ダ"], "sire_surface"),
        (["種牡馬", "distance_category"], "sire_distance"),
        (["種牡馬", "馬場状態"], "sire_going"),
        (["母父馬"], "bms"),
        (["母父馬", "芝・ダ"], "bms_surface"),
        (["母父馬", "distance_category"], "bms_distance"),
        (["母父馬", "馬場状態"], "bms_going"),
        (["種牡馬", "母父馬"], "sire_bms_pair"),
    ]
    for group_cols, prefix in specs:
        if all(c in out.columns for c in group_cols):
            hist = strict_group_history(
                out,
                group_cols,
                prefix,
                date_col=date_col,
                race_col=race_col,
                horse_col=horse_col,
                prior_strength=prior_strength,
            )
            out = pd.concat([out, hist], axis=1)

    out["pure_sire_surface_lift"] = num(out, "sire_surface_pure_top3_eb", 0) - num(out, "sire_pure_top3_eb", 0)
    out["pure_sire_distance_lift"] = num(out, "sire_distance_pure_top3_eb", 0) - num(out, "sire_pure_top3_eb", 0)
    out["pure_sire_going_lift"] = num(out, "sire_going_pure_top3_eb", 0) - num(out, "sire_pure_top3_eb", 0)
    out["pure_bms_surface_lift"] = num(out, "bms_surface_pure_top3_eb", 0) - num(out, "bms_pure_top3_eb", 0)
    out["pure_bms_distance_lift"] = num(out, "bms_distance_pure_top3_eb", 0) - num(out, "bms_pure_top3_eb", 0)
    out["pure_bms_going_lift"] = num(out, "bms_going_pure_top3_eb", 0) - num(out, "bms_pure_top3_eb", 0)
    out["pure_bloodline_lift_fit_score"] = (
        0.30 * out["pure_sire_surface_lift"]
        + 0.25 * out["pure_sire_distance_lift"]
        + 0.15 * out["pure_sire_going_lift"]
        + 0.15 * out["pure_bms_surface_lift"]
        + 0.10 * out["pure_bms_distance_lift"]
        + 0.05 * out["pure_bms_going_lift"]
    ).fillna(0.0)
    out["pure_bloodline_lower_bound_score"] = (
        0.55 * num(out, "sire_surface_pure_top3_lower10", 0)
        + 0.25 * num(out, "sire_distance_pure_top3_lower10", 0)
        + 0.20 * num(out, "bms_surface_pure_top3_lower10", 0)
    ).fillna(0.0)
    return out


def payout_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "bets": 0,
            "races": 0,
            "win_rate": 0.0,
            "top3_rate": 0.0,
            "win_roi": 0.0,
            "place_roi": 0.0,
            "avg_popularity": None,
            "avg_odds": None,
        }
    win_pay = num(frame, "単勝配当", 0).fillna(0).where(num(frame, "target_win", 0).eq(1), 0.0)
    place_pay = num(frame, "複勝配当", 0).fillna(0).where(num(frame, "target_top3", 0).eq(1), 0.0)
    return {
        "bets": int(len(frame)),
        "races": int(frame["レースID(新/馬番無)"].nunique()),
        "win_rate": float(num(frame, "target_win", 0).mean()),
        "top3_rate": float(num(frame, "target_top3", 0).mean()),
        "win_roi": float(win_pay.sum() / (len(frame) * 100.0)),
        "place_roi": float(place_pay.sum() / (len(frame) * 100.0)),
        "avg_popularity": float(num(frame, "人気", np.nan).mean()),
        "avg_odds": float(num(frame, "単勝オッズ", np.nan).mean()),
    }


def segment_report(test: pd.DataFrame) -> list[dict[str, Any]]:
    q_current_hi = float(num(test, "bloodline_lift_fit_score", 0).quantile(0.75))
    q_current_top = float(num(test, "bloodline_lift_fit_score", 0).quantile(0.90))
    q_pure_hi = float(num(test, "pure_bloodline_lift_fit_score", 0).quantile(0.75))
    q_pure_top = float(num(test, "pure_bloodline_lift_fit_score", 0).quantile(0.90))
    q_lower_hi = float(num(test, "pure_bloodline_lower_bound_score", 0).quantile(0.75))

    surface = test.get("芝・ダ", pd.Series("", index=test.index)).astype(str)
    going = test.get("馬場状態", pd.Series("", index=test.index)).astype(str)
    career = num(test, "キャリア", 99).fillna(99)
    same_dist = num(test, "same_distance_category_starts", 99).fillna(99)
    is_dirt = surface.str.contains("ダ", regex=False)
    is_turf = surface.str.contains("芝", regex=False)
    is_young2 = num(test, "年齢", 0).eq(2) if "年齢" in test.columns else pd.Series(False, index=test.index)
    low_career = career.le(3)
    first_distance = same_dist.le(0)
    wetish = going.astype(str).str.contains("稍|重|不", regex=True)
    ai_top1 = num(test, "ai_rank", 99).eq(1)
    ai_top3 = num(test, "ai_rank", 99).le(3)
    ai_top5 = num(test, "ai_rank", 99).le(5)

    specs = [
        ("ai_top1_current_blood_lift_hi_age2_dirt", ai_top1 & is_young2 & is_dirt & num(test, "bloodline_lift_fit_score", 0).ge(q_current_hi)),
        ("ai_top1_pure_blood_lift_hi_age2_dirt", ai_top1 & is_young2 & is_dirt & num(test, "pure_bloodline_lift_fit_score", 0).ge(q_pure_hi)),
        ("ai_top3_pure_blood_lift_hi_age2_dirt", ai_top3 & is_young2 & is_dirt & num(test, "pure_bloodline_lift_fit_score", 0).ge(q_pure_hi)),
        ("ai_top5_pure_blood_lift_hi_age2_dirt", ai_top5 & is_young2 & is_dirt & num(test, "pure_bloodline_lift_fit_score", 0).ge(q_pure_hi)),
        ("ai_top3_pure_lower_hi_low_career_first_distance", ai_top3 & low_career & first_distance & num(test, "pure_bloodline_lower_bound_score", 0).ge(q_lower_hi)),
        ("ai_top5_pure_lift_top_low_career_dirt", ai_top5 & low_career & is_dirt & num(test, "pure_bloodline_lift_fit_score", 0).ge(q_pure_top)),
        ("ai_top3_pure_lift_hi_wetish", ai_top3 & wetish & num(test, "pure_bloodline_lift_fit_score", 0).ge(q_pure_hi)),
        ("ai_top3_pure_lift_hi_wetish_dirt", ai_top3 & wetish & is_dirt & num(test, "pure_bloodline_lift_fit_score", 0).ge(q_pure_hi)),
        ("ai_top3_pure_lift_hi_turf", ai_top3 & is_turf & num(test, "pure_bloodline_lift_fit_score", 0).ge(q_pure_hi)),
    ]
    rows = []
    for label, mask in specs:
        rows.append({"segment": label, **payout_metrics(test[mask].copy())})
    return rows


def make_audit_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    checks = [
        ("sire", "sire_starts", "sire_strict_starts", "sire_pure_starts"),
        ("bms", "bms_starts", "bms_strict_starts", "bms_pure_starts"),
        ("sire_bms_pair", "sire_bms_pair_starts", "sire_bms_pair_strict_starts", "sire_bms_pair_pure_starts"),
    ]
    for label, existing, strict, pure in checks:
        if existing not in frame.columns or strict not in frame.columns:
            continue
        diff = (num(frame, existing, 0).fillna(0) - num(frame, strict, 0).fillna(0)).abs()
        own = (num(frame, strict, 0).fillna(0) - num(frame, pure, 0).fillna(0)).clip(lower=0)
        rows.append(
            {
                "group": label,
                "rows": int(len(frame)),
                "strict_mismatch_rows": int(diff.gt(0.0001).sum()),
                "strict_mismatch_rate": float(diff.gt(0.0001).mean()),
                "max_abs_starts_diff": float(diff.max()),
                "rows_with_own_history_in_group": int(own.gt(0.0001).sum()),
                "own_history_rate": float(own.gt(0.0001).mean()),
                "avg_own_history_subtracted": float(own.mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit bloodline as-of leakage and pure-prior challenger segments.")
    parser.add_argument("--history-train-csv", default=DEFAULT_HISTORY_TRAIN)
    parser.add_argument("--history-test-csv", default=DEFAULT_HISTORY_TEST)
    parser.add_argument("--score-test-csv", default=DEFAULT_SCORE_TEST)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--prior-strength", type=float, default=30.0)
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json_config(project_path(args.config))
    model: SimpleRaceRanker = pickle.load(project_path(args.model).open("rb"))
    train = read_needed(project_path(args.history_train_csv), [])
    test = read_needed(project_path(args.history_test_csv), [])
    train["_split"] = "train"
    test["_split"] = "test"
    frame = pd.concat([train, test], ignore_index=True)
    frame["レースID(新/馬番無)"] = frame["レースID(新/馬番無)"].astype(str)
    frame["血統登録番号"] = frame["血統登録番号"].astype(str)

    enriched = add_pure_priors(frame, prior_strength=args.prior_strength)
    history_test_enriched = enriched[enriched["_split"] == "test"].copy()
    pure_transfer_cols = [
        c
        for c in history_test_enriched.columns
        if c.endswith("_strict_starts")
        or c.endswith("_pure_starts")
        or c.endswith("_pure_win_rate")
        or c.endswith("_pure_top3_rate")
        or c.endswith("_pure_avg_score")
        or c.endswith("_pure_win_eb")
        or c.endswith("_pure_win_lower10")
        or c.endswith("_pure_top3_eb")
        or c.endswith("_pure_top3_lower10")
        or c.endswith("_pure_effective_n")
        or c.startswith("pure_")
    ]
    score_test = read_needed(project_path(args.score_test_csv), list(model.numeric_features) + list(model.categorical_features))
    score_test["レースID(新/馬番無)"] = score_test["レースID(新/馬番無)"].astype(str)
    score_test["血統登録番号"] = score_test["血統登録番号"].astype(str)
    transfer = history_test_enriched[
        ["レースID(新/馬番無)", "血統登録番号", *pure_transfer_cols]
    ].drop_duplicates(["レースID(新/馬番無)", "血統登録番号"], keep="last")
    test_enriched = score_test.merge(transfer, on=["レースID(新/馬番無)", "血統登録番号"], how="left")
    test_enriched["ai_score"] = model.predict(test_enriched)
    test_enriched["ai_rank"] = (
        test_enriched.groupby("レースID(新/馬番無)")["ai_score"].rank(ascending=False, method="first").astype(int)
    )

    audit_rows = make_audit_rows(enriched)
    segment_rows = segment_report(test_enriched)
    pd.DataFrame(audit_rows).to_csv(out_dir / "asof_audit_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(segment_rows).to_csv(out_dir / "pure_prior_segment_summary.csv", index=False, encoding="utf-8-sig")

    pure_cols = [
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
        "bloodline_lift_fit_score",
        "pure_bloodline_lift_fit_score",
        "pure_bloodline_lower_bound_score",
        "sire_pure_starts",
        "sire_pure_top3_eb",
        "sire_surface_pure_top3_eb",
        "sire_distance_pure_top3_eb",
        "bms_pure_starts",
        "bms_pure_top3_eb",
        "bms_surface_pure_top3_eb",
        "target_win",
        "target_top3",
        "単勝配当",
        "複勝配当",
    ]
    sample = test_enriched[[c for c in pure_cols if c in test_enriched.columns]].copy()
    sample.to_csv(out_dir / "test_pure_prior_scores.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(out_dir),
        "history_train_rows": int(len(train)),
        "history_test_rows": int(len(test)),
        "score_test_rows": int(len(score_test)),
        "prior_strength": float(args.prior_strength),
        "audit": audit_rows,
        "segments": segment_rows,
        "notes": [
            "strict_* excludes same-race records and future records.",
            "pure_* additionally subtracts the target horse's own prior records from the pedigree group.",
            "This is an audit/challenger output. It does not replace the production model yet.",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(summary), f, ensure_ascii=False, indent=2)

    lines = [
        "# Bloodline as-of / pure-prior audit",
        "",
        "## What was checked",
        "",
        "- Recomputed strict as-of bloodline starts/rates at race boundary.",
        "- Built pure pedigree priors by subtracting the target horse's own prior records from sire/bms/pair histories.",
        "- Added Beta-Binomial style shrinkage using a normal approximation for the 10% lower bound.",
        "- Tested pure-prior challenger segments on the test period without changing the saved model.",
        "",
        "## As-of audit",
        "",
        markdown_table(audit_rows),
        "",
        "## Pure-prior challenger segments",
        "",
        markdown_table(segment_rows),
        "",
        "## Interpretation",
        "",
        "- If strict mismatch rows are non-zero, current bloodline starts can include same-race records for duplicated sire/bms groups.",
        "- If own-history rate is high, current pedigree stats partly contain the target horse's own ability history.",
        "- Segments that remain positive under pure priors are safer candidates for low-career / first-condition rescue.",
    ]
    (out_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
