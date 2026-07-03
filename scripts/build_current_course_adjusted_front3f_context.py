from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_course_adjusted_front3f_signal import (
    DEFAULT_ESTIMATES,
    enrich_course_adjusted_front3f,
    race_z,
    read_csv,
    sigmoid,
    top_gap,
)
from scripts.evaluate_expected_lap_rpci_features import DEFAULT_TEST, DEFAULT_TRAIN, HORSE_COL, RACE_COL


DEFAULT_OUT = ROOT / "outputs" / "analysis" / "current_strongest_runtime_v1" / "current_course_adjusted_front3f_context.csv"
DEFAULT_HISTORICAL_CONTEXT = (
    ROOT / "outputs" / "analysis" / "course_adjusted_front3f_signal_v1" / "course_adjusted_front3f_enriched_light.csv"
)


def project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def latest_file(pattern: str) -> Path | None:
    files = list(ROOT.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def ncol(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = df[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce").fillna(default)


def quantiles(history: pd.DataFrame) -> dict[str, float]:
    def q(col: str, p: float, default: float) -> float:
        if col not in history.columns:
            return default
        values = pd.to_numeric(history[col], errors="coerce").dropna()
        if values.empty:
            return default
        return float(values.quantile(p))

    return {
        "runner_hi": max(q("horse_course_adj_ten_speed_mean_past5", 0.75, 0.45), 0.35),
        "runner_lo": q("horse_course_adj_ten_speed_mean_past5", 0.25, -0.25),
        "best_hi": max(q("horse_course_adj_ten_speed_best_past5", 0.75, 0.55), 0.55),
        "pressure_hi": q("race_course_adj_ten_pressure_score", 0.75, 0.65),
        "pressure_lo": q("race_course_adj_ten_pressure_score", 0.25, 0.25),
        "clarity_hi": max(q("race_course_adj_queue_clarity_score", 0.75, 0.65), 0.62),
        "clarity_lo": q("race_course_adj_queue_clarity_score", 0.25, 0.35),
    }


def race_label(row: pd.Series, q: dict[str, float]) -> str:
    pressure = float(row.get("race_course_adj_ten_pressure_score") or 0.0)
    clarity = float(row.get("race_course_adj_queue_clarity_score") or 0.0)
    fast_count = float(row.get("race_course_adj_fast_start_count") or 0.0)
    history_count = float(row.get("course_ten_history_count") or 0.0)
    gap = float(row.get("race_course_adj_ten_speed_gap_top2") or 0.0)
    if history_count < 3:
        return "クラス×コース基準: 履歴不足"
    if pressure >= q["pressure_hi"] and clarity <= q["clarity_lo"]:
        return "クラス×コース基準: 先行競り合い寄り"
    if clarity >= q["clarity_hi"] and fast_count <= 3 and (gap >= 0.15 or pressure >= 0.10):
        return "クラス×コース基準: 隊列明確寄り"
    if pressure <= q["pressure_lo"]:
        return "クラス×コース基準: テン落ち着き寄り"
    return "クラス×コース基準: 標準"


def runner_note(row: pd.Series, q: dict[str, float]) -> str:
    mean = float(row.get("horse_course_adj_ten_speed_mean_past5") or 0.0)
    best = float(row.get("horse_course_adj_ten_speed_best_past5") or 0.0)
    fast_rate = float(row.get("horse_course_adj_fast_start_rate_past5") or 0.0)
    if float(row.get("course_ten_history_available") or 0.0) < 1:
        return "補正テン履歴不足"
    if mean >= q["runner_hi"] or best >= q["best_hi"] or fast_rate >= 0.34:
        return "コース基準でテン速い"
    if mean <= q["runner_lo"] and fast_rate <= 0.10:
        return "コース基準ではテン不足"
    return "コース基準テンは標準"


def _race_horse_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_race_key_join"] = out[RACE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    if "馬番" in out.columns:
        out["_horse_no_join"] = pd.to_numeric(out["馬番"], errors="coerce").astype("Int64")
    else:
        out["_horse_no_join"] = pd.NA
    return out


def prepare_current_frame(prediction_csv: Path, entry_csv: Path | None) -> pd.DataFrame:
    prediction = read_csv(prediction_csv)
    if prediction.empty:
        raise ValueError(f"prediction CSV is empty: {prediction_csv}")
    if RACE_COL not in prediction.columns:
        raise ValueError(f"prediction CSV must contain {RACE_COL!r}: {prediction_csv}")
    if HORSE_COL in prediction.columns:
        return prediction
    if entry_csv is None or not entry_csv.exists():
        raise ValueError(
            f"prediction CSV does not contain {HORSE_COL!r}; pass --entry-csv with bloodline IDs: {prediction_csv}"
        )
    entry = read_csv(entry_csv)
    if entry.empty:
        raise ValueError(f"entry CSV is empty: {entry_csv}")
    if RACE_COL not in entry.columns or HORSE_COL not in entry.columns:
        raise ValueError(f"entry CSV must contain {RACE_COL!r} and {HORSE_COL!r}: {entry_csv}")
    entry = _race_horse_key(entry)
    prediction = _race_horse_key(prediction)
    if "_horse_no_join" not in prediction.columns or prediction["_horse_no_join"].isna().all():
        raise ValueError("prediction CSV needs 馬番 when bloodline IDs are supplied from entry CSV.")

    pred_cols = [
        c
        for c in prediction.columns
        if c not in entry.columns or c in {"ai_rank", "ai_score", "expected_pace", "front_running_tendency", "closing_tendency"}
    ]
    pred_cols = ["_race_key_join", "_horse_no_join"] + [c for c in pred_cols if c not in {"_race_key_join", "_horse_no_join"}]
    current = entry.merge(
        prediction[pred_cols].drop_duplicates(["_race_key_join", "_horse_no_join"], keep="last"),
        on=["_race_key_join", "_horse_no_join"],
        how="left",
        suffixes=("", "_pred"),
    )
    return current.drop(columns=["_race_key_join", "_horse_no_join"], errors="ignore")


def finalize_current_context(
    current_enriched: pd.DataFrame,
    q: dict[str, float],
    prediction_csv: Path,
    entry_csv: Path | None,
    context_source_csv: Path,
    *,
    fast_mode: bool,
    history_rows: int | None = None,
    estimate_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keep_candidates = [
        RACE_COL,
        HORSE_COL,
        "馬番",
        "馬名",
        "場所",
        "芝・ダ",
        "距離",
        "馬場状態",
        "クラス名",
        "日付S",
        "Ｒ",
        "発走時刻",
        "レース名",
        "ai_rank",
        "ai_score",
        "人気",
        "単勝オッズ",
        "course_front3f_prior_sec",
        "course_front3f_prior_std",
        "course_front3f_prior_count",
        "course_ten_history_available",
        "course_ten_history_count",
        "horse_course_adj_ten_speed_mean_past5",
        "horse_course_adj_ten_speed_best_past5",
        "horse_course_adj_fast_start_rate_past5",
        "course_adj_ten_race_z",
        "race_course_adj_ten_pressure_score",
        "race_course_adj_fast_start_count",
        "race_course_adj_ten_speed_gap_top2",
        "race_course_adj_queue_clarity_score",
    ]
    out = current_enriched[[c for c in keep_candidates if c in current_enriched.columns]].copy()
    out[RACE_COL] = out[RACE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    out[HORSE_COL] = out[HORSE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    out["course_ten_runner_note"] = out.apply(lambda r: runner_note(r, q), axis=1)
    race_notes = (
        out[
            [
                RACE_COL,
                "race_course_adj_ten_pressure_score",
                "race_course_adj_fast_start_count",
                "race_course_adj_ten_speed_gap_top2",
                "race_course_adj_queue_clarity_score",
                "course_ten_history_count",
            ]
        ]
        .drop_duplicates(RACE_COL)
        .copy()
    )
    race_notes["course_ten_race_label"] = race_notes.apply(lambda r: race_label(r, q), axis=1)
    out = out.merge(race_notes[[RACE_COL, "course_ten_race_label"]], on=RACE_COL, how="left")

    summary = {
        "prediction_csv": str(prediction_csv),
        "entry_csv": str(entry_csv) if entry_csv else "",
        "context_source_csv": str(context_source_csv),
        "rows": int(len(out)),
        "races": int(out[RACE_COL].nunique()),
        "fast_mode": bool(fast_mode),
        "history_rows": int(history_rows) if history_rows is not None else None,
        "estimate_rows": int(estimate_rows) if estimate_rows is not None else None,
        "thresholds": q,
        "notes": [
            "Class-tier course front3F priors are built from historical races only.",
            "Current rows use past horse estimated front3F history; missing recent history is treated as neutral.",
            "This output is for dashboard context and shadow review. It does not change formal BUY gates.",
        ],
    }
    return out, summary


def build_context_fast_from_historical(
    current: pd.DataFrame,
    prediction_csv: Path,
    entry_csv: Path | None,
    historical_context_csv: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    history = read_csv(historical_context_csv)
    if history.empty or RACE_COL not in history.columns or HORSE_COL not in history.columns:
        raise ValueError(f"historical context CSV is not usable: {historical_context_csv}")
    history = history.copy()
    history[RACE_COL] = history[RACE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    history[HORSE_COL] = history[HORSE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    history["_date_order"] = pd.to_numeric(history[RACE_COL].str.slice(0, 8), errors="coerce").fillna(0)
    history = history.sort_values([HORSE_COL, "_date_order", RACE_COL], kind="mergesort")
    feature_cols = [
        "horse_course_adj_ten_speed_mean_past5",
        "horse_course_adj_ten_speed_best_past5",
        "horse_course_adj_fast_start_rate_past5",
    ]
    latest = history.drop_duplicates(HORSE_COL, keep="last")[[HORSE_COL] + [c for c in feature_cols if c in history.columns]].copy()
    out = current.copy()
    out[RACE_COL] = out[RACE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    out[HORSE_COL] = out[HORSE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    out = out.merge(latest, on=HORSE_COL, how="left")
    present_feature_cols = [c for c in feature_cols if c in out.columns]
    out["course_ten_history_available"] = (
        out[present_feature_cols].notna().any(axis=1).astype(float) if present_feature_cols else 0.0
    )
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["course_ten_history_count"] = out["course_ten_history_available"].groupby(out[RACE_COL]).transform("sum").fillna(0.0)
    out["course_adj_ten_race_z"] = race_z(out["horse_course_adj_ten_speed_mean_past5"], out[RACE_COL])
    course_pos = out["horse_course_adj_ten_speed_mean_past5"].clip(lower=0.0)
    field = course_pos.groupby(out[RACE_COL]).transform("size").replace(0, np.nan)
    out["race_course_adj_ten_pressure_score"] = (course_pos.groupby(out[RACE_COL]).transform("sum") / np.sqrt(field)).fillna(0.0)
    out["race_course_adj_fast_start_count"] = (
        ((out["horse_course_adj_ten_speed_mean_past5"].ge(0.45)) | (out["horse_course_adj_fast_start_rate_past5"].ge(0.34)))
        .astype(float)
        .groupby(out[RACE_COL])
        .transform("sum")
    )
    out["race_course_adj_ten_speed_gap_top2"] = out["horse_course_adj_ten_speed_mean_past5"].groupby(out[RACE_COL]).transform(top_gap)
    out["race_course_adj_queue_clarity_score"] = sigmoid(
        1.5 * out["race_course_adj_ten_speed_gap_top2"] - 0.32 * out["race_course_adj_fast_start_count"] + 0.35
    )
    out["course_front3f_prior_sec"] = np.nan
    out["course_front3f_prior_std"] = np.nan
    out["course_front3f_prior_count"] = float(history[RACE_COL].nunique())
    q = quantiles(history)
    return finalize_current_context(out, q, prediction_csv, entry_csv, historical_context_csv, fast_mode=True, history_rows=len(history))


def build_context(
    prediction_csv: Path,
    estimates_csv: Path,
    train_csv: Path,
    test_csv: Path,
    entry_csv: Path | None = None,
    historical_context_csv: Path | None = DEFAULT_HISTORICAL_CONTEXT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    current = prepare_current_frame(prediction_csv, entry_csv)
    if historical_context_csv and historical_context_csv.exists():
        return build_context_fast_from_historical(current, prediction_csv, entry_csv, historical_context_csv)

    train = read_csv(train_csv)
    test = read_csv(test_csv)
    history = pd.concat([train, test], ignore_index=True, sort=False)
    history["_split"] = "history"
    current = current.copy()
    current["_split"] = "current"
    base = pd.concat([history, current], ignore_index=True, sort=False)

    estimates = read_csv(estimates_csv, dtype={"race_id": str})
    enriched, _diag = enrich_course_adjusted_front3f(base, estimates)
    history_enriched = enriched[enriched["_split"].eq("history")].copy()
    current_enriched = enriched[enriched["_split"].eq("current")].copy()
    q = quantiles(history_enriched)
    return finalize_current_context(
        current_enriched,
        q,
        prediction_csv,
        entry_csv,
        estimates_csv,
        fast_mode=False,
        history_rows=len(history),
        estimate_rows=len(estimates),
    )

    keep_candidates = [
        RACE_COL,
        HORSE_COL,
        "馬番",
        "馬名",
        "場所",
        "芝・ダ",
        "距離",
        "馬場状態",
        "クラス名",
        "日付S",
        "Ｒ",
        "発走時刻",
        "レース名",
        "ai_rank",
        "ai_score",
        "人気",
        "単勝オッズ",
        "course_front3f_prior_sec",
        "course_front3f_prior_std",
        "course_front3f_prior_count",
        "horse_course_adj_ten_speed_mean_past5",
        "horse_course_adj_ten_speed_best_past5",
        "horse_course_adj_fast_start_rate_past5",
        "course_adj_ten_race_z",
        "race_course_adj_ten_pressure_score",
        "race_course_adj_fast_start_count",
        "race_course_adj_ten_speed_gap_top2",
        "race_course_adj_queue_clarity_score",
    ]
    out = current_enriched[[c for c in keep_candidates if c in current_enriched.columns]].copy()
    out[RACE_COL] = out[RACE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    out[HORSE_COL] = out[HORSE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    out["course_ten_runner_note"] = out.apply(lambda r: runner_note(r, q), axis=1)
    race_notes = (
        out[
            [
                RACE_COL,
                "race_course_adj_ten_pressure_score",
                "race_course_adj_fast_start_count",
                "race_course_adj_ten_speed_gap_top2",
                "race_course_adj_queue_clarity_score",
            ]
        ]
        .drop_duplicates(RACE_COL)
        .copy()
    )
    race_notes["course_ten_race_label"] = race_notes.apply(lambda r: race_label(r, q), axis=1)
    out = out.merge(race_notes[[RACE_COL, "course_ten_race_label"]], on=RACE_COL, how="left")

    summary = {
        "prediction_csv": str(prediction_csv),
        "entry_csv": str(entry_csv) if entry_csv else "",
        "estimates_csv": str(estimates_csv),
        "rows": int(len(out)),
        "races": int(out[RACE_COL].nunique()),
        "history_rows": int(len(history)),
        "estimate_rows": int(len(estimates)),
        "thresholds": q,
        "notes": [
            "Class-tier course front3F priors are built from historical races only.",
            "Current rows use past horse estimated front3F history; missing recent history is treated as neutral.",
            "This output is for dashboard context and shadow review. It does not change formal BUY gates.",
        ],
    }
    return out, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-csv",
        default="",
        help="Current prediction CSV. Defaults to the latest preday prediction file.",
    )
    parser.add_argument(
        "--entry-csv",
        default="",
        help="Current entry CSV used to supply bloodline IDs when the prediction CSV does not include them.",
    )
    parser.add_argument("--estimates-csv", default=str(DEFAULT_ESTIMATES))
    parser.add_argument("--historical-context-csv", default=str(DEFAULT_HISTORICAL_CONTEXT))
    parser.add_argument("--train-csv", default=str(DEFAULT_TRAIN))
    parser.add_argument("--test-csv", default=str(DEFAULT_TEST))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    prediction_csv = project_path(args.prediction_csv) if args.prediction_csv else (
        latest_file("outputs/predictions/preday_target_de_overlay_*/baseline_predictions_*.csv")
        or latest_file("outputs/predictions/preday_strongest_feature_parity/baseline_predictions_*.csv")
    )
    entry_csv = project_path(args.entry_csv) if args.entry_csv else (
        latest_file("data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched_workout_knowledge.csv")
        or latest_file("data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched_workout.csv")
        or latest_file("data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched.csv")
        or latest_file("data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay.csv")
    )
    if prediction_csv is None:
        raise FileNotFoundError("No prediction CSV found.")
    output_csv = project_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    out, summary = build_context(
        prediction_csv=prediction_csv,
        estimates_csv=project_path(args.estimates_csv),
        train_csv=project_path(args.train_csv),
        test_csv=project_path(args.test_csv),
        entry_csv=entry_csv,
        historical_context_csv=project_path(args.historical_context_csv) if args.historical_context_csv else None,
    )
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_csv": str(output_csv), **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
