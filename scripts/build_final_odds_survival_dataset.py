from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={"race_id": str}, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False)


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def first_existing(frame: pd.DataFrame, names: list[str], default: float = np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def candidate_numbers(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    a = first_existing(frame, ["anchor_horse_no", "anchor_no", "horse_a", "a_no"])
    b = first_existing(frame, ["partner_horse_no", "partner_no", "horse_b", "b_no"])
    return a, b


def pair_key(race_id: pd.Series, a: pd.Series, b: pd.Series, ticket_type: pd.Series | str = "umaren") -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    lo = np.minimum(aa, bb)
    hi = np.maximum(aa, bb)
    if isinstance(ticket_type, pd.Series):
        tt = ticket_type.astype(str)
    else:
        tt = pd.Series(ticket_type, index=race_id.index)
    return race_id.astype(str) + ":" + tt + ":" + lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)


def latest_by_label(timeline: pd.DataFrame, label_match: str) -> pd.DataFrame:
    if timeline.empty:
        return timeline
    label = timeline.get("decision_label", pd.Series("", index=timeline.index)).astype(str)
    part = timeline[label.str.upper().str.contains(label_match.upper(), regex=False)].copy()
    if part.empty:
        return part
    part["_sort_time"] = part.get("captured_at", part.get("snapshot_at", "")).astype(str)
    part = part.sort_values(["pair_id", "_sort_time"])
    return part.groupby("pair_id", as_index=False).tail(1)


def final_snapshot(timeline: pd.DataFrame) -> pd.DataFrame:
    if timeline.empty:
        return timeline
    part = timeline.copy()
    part["_sort_time"] = part.get("captured_at", part.get("snapshot_at", "")).astype(str)
    part = part.sort_values(["pair_id", "_sort_time"])
    return part.groupby("pair_id", as_index=False).tail(1)


def add_quantile_predictions(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    valid = out[out["t5_odds"].gt(0) & out["final_odds"].gt(0)].copy()
    if valid.empty:
        out["final_odds_pred_p10"] = np.nan
        out["final_odds_pred_p20"] = np.nan
        out["final_odds_pred_p50"] = np.nan
        out["final_odds_pred_p80"] = np.nan
        out["final_odds_pred_p90"] = np.nan
        out["conservative_expected_roi"] = np.nan
        return out

    valid["log_final_over_t5"] = np.log(valid["final_odds"] / valid["t5_odds"])
    valid["t5_odds_bin"] = pd.cut(
        valid["t5_odds"],
        bins=[0, 5, 10, 20, 40, 80, 120, 200, 500, np.inf],
        include_lowest=True,
    ).astype(str)
    quantile_points = [0.10, 0.20, 0.50, 0.80, 0.90]
    global_q = valid["log_final_over_t5"].quantile(quantile_points).to_dict()
    q = (
        valid.groupby("t5_odds_bin")["log_final_over_t5"]
        .quantile(quantile_points)
        .unstack()
        .rename(columns={0.10: "q10", 0.20: "q20", 0.50: "q50", 0.80: "q80", 0.90: "q90"})
        .reset_index()
    )
    out["t5_odds_bin"] = pd.cut(
        out["t5_odds"],
        bins=[0, 5, 10, 20, 40, 80, 120, 200, 500, np.inf],
        include_lowest=True,
    ).astype(str)
    out = out.merge(q, on="t5_odds_bin", how="left")
    for src, fallback, dest in [
        ("q10", global_q.get(0.10, 0.0), "final_odds_pred_p10"),
        ("q20", global_q.get(0.20, 0.0), "final_odds_pred_p20"),
        ("q50", global_q.get(0.50, 0.0), "final_odds_pred_p50"),
        ("q80", global_q.get(0.80, 0.0), "final_odds_pred_p80"),
        ("q90", global_q.get(0.90, 0.0), "final_odds_pred_p90"),
    ]:
        log_ratio = pd.to_numeric(out[src], errors="coerce").fillna(fallback)
        out[dest] = out["t5_odds"] * np.exp(log_ratio)
    prob = first_existing(out, ["ticket_hit_prob", "pair_calibrated_hit_prob", "umaren_hit_prob_cal"], 0.0).clip(0, 1)
    out["conservative_expected_roi"] = prob * out["final_odds_pred_p10"]
    out["conservative_expected_roi_p20"] = prob * out["final_odds_pred_p20"]
    out["median_expected_roi"] = prob * out["final_odds_pred_p50"]
    out["final_odds_log_ratio_actual"] = np.where(
        out["t5_odds"].gt(0) & out["final_odds"].gt(0),
        np.log(out["final_odds"] / out["t5_odds"]),
        np.nan,
    )
    return out.drop(columns=[c for c in ["q10", "q20", "q50", "q80", "q90"] if c in out.columns])


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T-5/T-3 to final pair-odds survival dataset.")
    parser.add_argument("--candidates-csv", default="outputs/analysis/current_strongest_runtime_v1/current_strongest_all_candidates.csv")
    parser.add_argument("--pair-timeline-csv", default="data/processed/live_odds/realtime_pair_odds_timeline.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/final_odds_survival_model_v1")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_csv(project_path(args.candidates_csv))
    timeline = read_csv(project_path(args.pair_timeline_csv))
    if candidates.empty or timeline.empty:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "rows": 0,
            "reason": "missing candidates or pair timeline",
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    a, b = candidate_numbers(candidates)
    candidates = candidates.copy()
    candidates["pair_id"] = pair_key(candidates["race_id"], a, b, "umaren")
    candidates["candidate_current_odds"] = first_existing(candidates, ["live_odds", "runtime_odds", "quote_odds_proxy"])

    timeline = timeline.copy()
    timeline = timeline[timeline.get("ticket_type", "umaren").astype(str).eq("umaren")].copy()
    timeline["pair_id"] = pair_key(timeline["race_id"], timeline["a_no"], timeline["b_no"], timeline["ticket_type"])
    timeline["live_odds"] = num(timeline, "live_odds")
    timeline = timeline[timeline["live_odds"].gt(0)].copy()

    base_cols = [
        "race_id",
        "pair_id",
        "anchor_horse_no",
        "partner_horse_no",
        "candidate_current_odds",
        "ticket_hit_prob",
        "min_odds_margin_ratio",
        "runtime_expected_roi",
        "strongest_current_score",
        "projected_front5_prob",
        "ticket_danger_popular_in_pair_score",
        "race_difficulty_score",
        "skip_risk_score",
    ]
    base_cols = [c for c in base_cols if c in candidates.columns]
    dataset = candidates[base_cols].drop_duplicates("pair_id").copy()

    for label, prefix in [("T-10", "t10"), ("T-5", "t5"), ("T-3", "t3")]:
        snap = latest_by_label(timeline, label)
        if snap.empty:
            dataset[f"{prefix}_odds"] = np.nan
            dataset[f"{prefix}_snapshot_at"] = ""
            continue
        snap = snap[["pair_id", "live_odds", "snapshot_at", "captured_at", "decision_label"]].rename(
            columns={
                "live_odds": f"{prefix}_odds",
                "snapshot_at": f"{prefix}_snapshot_at",
                "captured_at": f"{prefix}_captured_at",
                "decision_label": f"{prefix}_label",
            }
        )
        dataset = dataset.merge(snap, on="pair_id", how="left")

    final = final_snapshot(timeline)[["pair_id", "live_odds", "snapshot_at", "captured_at", "decision_label"]].rename(
        columns={
            "live_odds": "final_odds",
            "snapshot_at": "final_snapshot_at",
            "captured_at": "final_captured_at",
            "decision_label": "final_label",
        }
    )
    dataset = dataset.merge(final, on="pair_id", how="left")
    for src, dest in [
        ("t10_odds", "log_t5_over_t10"),
        ("t5_odds", "log_t3_over_t5"),
        ("t5_odds", "log_final_over_t5"),
        ("t3_odds", "log_final_over_t3"),
    ]:
        dataset[dest] = np.nan
    dataset["log_t5_over_t10"] = np.where(
        dataset["t10_odds"].gt(0) & dataset["t5_odds"].gt(0),
        np.log(dataset["t5_odds"] / dataset["t10_odds"]),
        np.nan,
    )
    dataset["log_t3_over_t5"] = np.where(
        dataset["t5_odds"].gt(0) & dataset["t3_odds"].gt(0),
        np.log(dataset["t3_odds"] / dataset["t5_odds"]),
        np.nan,
    )
    dataset["log_final_over_t5"] = np.where(
        dataset["t5_odds"].gt(0) & dataset["final_odds"].gt(0),
        np.log(dataset["final_odds"] / dataset["t5_odds"]),
        np.nan,
    )
    dataset["log_final_over_t3"] = np.where(
        dataset["t3_odds"].gt(0) & dataset["final_odds"].gt(0),
        np.log(dataset["final_odds"] / dataset["t3_odds"]),
        np.nan,
    )
    dataset = add_quantile_predictions(dataset)

    dataset.to_csv(out_dir / "final_odds_survival_dataset.csv", index=False, encoding="utf-8-sig")
    valid = dataset[dataset["t5_odds"].gt(0) & dataset["final_odds"].gt(0)].copy()
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidates": int(len(candidates)),
        "dataset_rows": int(len(dataset)),
        "with_t5_and_final": int(len(valid)),
        "with_t3_and_final": int((dataset["t3_odds"].gt(0) & dataset["final_odds"].gt(0)).sum()),
        "global_log_final_over_t5_quantiles": valid["log_final_over_t5"].quantile([0.10, 0.20, 0.50, 0.80, 0.90]).to_dict() if not valid.empty else {},
        "median_t5_odds": float(valid["t5_odds"].median()) if not valid.empty else None,
        "median_final_odds": float(valid["final_odds"].median()) if not valid.empty else None,
        "note": "Use P20 final-odds prediction and survival probability as Challenger-only conservative expected value until enough live weeks are accumulated.",
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
