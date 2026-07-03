from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_COLUMNS = [
    "decision_snapshot_id",
    "captured_at",
    "decision_label",
    "model_version",
    "race_id",
    "date_key",
    "post_time",
    "minutes_to_post",
    "decision_key",
    "decision_reason",
    "live_odds",
    "min_odds_margin_ratio",
    "runtime_expected_roi",
    "ticket_hit_prob",
    "projected_front5_prob",
    "pair_quinella_score",
    "skip_risk_score",
    "race_difficulty_score",
    "late_value_survives_score",
    "track_available",
    "going_observed_at",
    "minutes_since_going_update",
    "track_state_uncertainty",
    "final_buy_tickets",
    "falloff_reasons",
]


CRITICAL_NUMERIC_COLUMNS = [
    "live_odds",
    "min_odds_margin_ratio",
    "runtime_expected_roi",
    "ticket_hit_prob",
    "projected_front5_prob",
    "skip_risk_score",
    "race_difficulty_score",
]


FIXED_TIME_LABELS = ["T-10", "T-5", "T-3", "final_check"]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


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
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"race_id": str, "decision_snapshot_id": str}, low_memory=False)


def row_hashes(frame: pd.DataFrame) -> pd.Series:
    stable = frame.copy()
    cols = [c for c in stable.columns if c != "_row_hash"]
    stable = stable[cols].fillna("")
    return stable.astype(str).agg("\x1f".join, axis=1).map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())


def label_family(label: str) -> str:
    text = str(label)
    for fixed in FIXED_TIME_LABELS:
        if fixed in text:
            return fixed
    if "manual" in text:
        return "manual"
    if "post" in text:
        return "post"
    return text or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit immutable decision snapshot completeness for live betting review.")
    parser.add_argument(
        "--snapshots-csv",
        default="data/processed/live_decision_snapshots/current_strongest_decision_snapshots.csv",
    )
    parser.add_argument("--output-dir", default="outputs/analysis/decision_snapshot_completeness_v1")
    parser.add_argument("--max-critical-null-rate", type=float, default=0.02)
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots_path = project_path(args.snapshots_csv)
    df = read_csv(snapshots_path)
    if df.empty:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "snapshots_csv": str(snapshots_path),
            "rows": 0,
            "status": "FAIL",
            "failures": ["snapshot file missing or empty"],
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    null_rows = []
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            series = df[col]
            empty = series.isna() | series.astype(str).str.strip().isin(["", "nan", "None"])
            null_rows.append({"column": col, "null_rate": float(empty.mean()), "null_count": int(empty.sum())})
    null_report = pd.DataFrame(null_rows).sort_values(["null_rate", "column"], ascending=[False, True])
    null_report.to_csv(out_dir / "required_column_null_report.csv", index=False, encoding="utf-8-sig")

    critical_failures = []
    for col in CRITICAL_NUMERIC_COLUMNS:
        if col in df.columns:
            rate = float(pd.to_numeric(df[col], errors="coerce").isna().mean())
            if rate > args.max_critical_null_rate:
                critical_failures.append(f"{col} null/non-numeric rate {rate:.3f} > {args.max_critical_null_rate:.3f}")

    label_counts = (
        df["decision_label"].astype(str).map(label_family).value_counts(dropna=False).rename_axis("label_family").reset_index(name="rows")
        if "decision_label" in df.columns
        else pd.DataFrame(columns=["label_family", "rows"])
    )
    label_counts.to_csv(out_dir / "decision_label_family_counts.csv", index=False, encoding="utf-8-sig")
    label_set = set(label_counts["label_family"].astype(str))
    missing_fixed_labels = [label for label in FIXED_TIME_LABELS if label not in label_set]

    duplicate_snapshot_ids = 0
    conflicting_snapshot_ids = 0
    if "decision_snapshot_id" in df.columns:
        duplicate_snapshot_ids = int(df["decision_snapshot_id"].duplicated().sum())
        work = df.copy()
        work["_row_hash"] = row_hashes(work)
        conflicts = work.groupby("decision_snapshot_id", dropna=False)["_row_hash"].nunique()
        conflicting_snapshot_ids = int(conflicts.gt(1).sum())

    race_coverage = (
        df.groupby("race_id", dropna=False)
        .agg(
            snapshots=("decision_snapshot_id", "count") if "decision_snapshot_id" in df.columns else ("race_id", "count"),
            label_families=("decision_label", lambda s: ",".join(sorted(set(label_family(v) for v in s)))),
            first_captured_at=("captured_at", "min") if "captured_at" in df.columns else ("race_id", "first"),
            last_captured_at=("captured_at", "max") if "captured_at" in df.columns else ("race_id", "first"),
        )
        .reset_index()
    )
    race_coverage.to_csv(out_dir / "race_snapshot_coverage.csv", index=False, encoding="utf-8-sig")

    failures = []
    warnings = []
    if missing_columns:
        failures.append(f"missing required columns: {', '.join(missing_columns)}")
    if critical_failures:
        failures.extend(critical_failures)
    if conflicting_snapshot_ids:
        failures.append(f"same decision_snapshot_id has conflicting rows: {conflicting_snapshot_ids}")
    if duplicate_snapshot_ids:
        warnings.append(f"duplicate decision_snapshot_id rows: {duplicate_snapshot_ids}")
    if missing_fixed_labels:
        warnings.append(f"fixed-time labels not yet accumulated: {', '.join(missing_fixed_labels)}")

    fixed_time_ready = not missing_fixed_labels
    immutable_ready = not missing_columns and not critical_failures and conflicting_snapshot_ids == 0
    status = "PASS" if immutable_ready and fixed_time_ready else ("PARTIAL" if immutable_ready else "FAIL")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "snapshots_csv": str(snapshots_path),
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()) if "race_id" in df.columns else 0,
        "label_counts": label_counts.to_dict(orient="records"),
        "missing_required_columns": missing_columns,
        "missing_fixed_time_labels": missing_fixed_labels,
        "duplicate_snapshot_ids": duplicate_snapshot_ids,
        "conflicting_snapshot_ids": conflicting_snapshot_ids,
        "immutable_snapshot_ready": immutable_ready,
        "fixed_time_timeline_ready": fixed_time_ready,
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "next_action": (
            "Keep live operation in shadow/champion-only mode until T-10/T-5/T-3/final_check snapshots are accumulated."
            if not fixed_time_ready
            else "Snapshot layer is ready for risk-coverage review."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
