from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


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
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False)


def parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.replace("/", "-", regex=False), errors="coerce")


def classify_staleness(minutes: pd.Series, fresh_max: float, stale_min: float) -> pd.Series:
    out = pd.Series("unknown", index=minutes.index, dtype=object)
    out.loc[minutes.le(fresh_max)] = "fresh"
    out.loc[minutes.gt(fresh_max) & minutes.lt(stale_min)] = "aging"
    out.loc[minutes.ge(stale_min)] = "stale"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit live track-condition freshness and regime transition risk.")
    parser.add_argument("--track-csv", default="data/processed/live_track_conditions/current_track_conditions.csv")
    parser.add_argument(
        "--snapshots-csv",
        default="data/processed/live_decision_snapshots/current_strongest_decision_snapshots.csv",
    )
    parser.add_argument("--output-dir", default="outputs/analysis/track_regime_freshness_v1")
    parser.add_argument("--fresh-max-minutes", type=float, default=30.0)
    parser.add_argument("--stale-min-minutes", type=float, default=90.0)
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    track = read_csv(project_path(args.track_csv))
    snapshots = read_csv(project_path(args.snapshots_csv))

    venue_rows = []
    if not track.empty:
        work = track.copy()
        for col in ["fetched_at", "observed_date", "effective_date"]:
            if col not in work.columns:
                work[col] = ""
        work["_fetched_at_dt"] = parse_time(work["fetched_at"])
        latest_by_venue = work.sort_values("_fetched_at_dt").drop_duplicates("venue", keep="last")
        for _, row in latest_by_venue.iterrows():
            venue_rows.append(
                {
                    "venue": row.get("venue", ""),
                    "weather": row.get("weather", ""),
                    "turf_going": row.get("turf_going", ""),
                    "dirt_going": row.get("dirt_going", ""),
                    "fetched_at": row.get("fetched_at", ""),
                    "timing": row.get("timing", ""),
                    "source_url": row.get("source_url", ""),
                    "complete": bool(str(row.get("turf_going", "")).strip() and str(row.get("dirt_going", "")).strip()),
                }
            )
    venue_report = pd.DataFrame(venue_rows)
    venue_report.to_csv(out_dir / "venue_track_condition_latest.csv", index=False, encoding="utf-8-sig")

    snapshot_report = pd.DataFrame()
    if not snapshots.empty:
        snap = snapshots.copy()
        if "minutes_since_going_update" in snap.columns:
            minutes = pd.to_numeric(snap["minutes_since_going_update"], errors="coerce")
        else:
            captured = parse_time(snap.get("captured_at", pd.Series("", index=snap.index)))
            observed = parse_time(snap.get("going_observed_at", pd.Series("", index=snap.index)))
            minutes = (captured - observed).dt.total_seconds() / 60.0
        snap["track_freshness_bucket"] = classify_staleness(minutes, args.fresh_max_minutes, args.stale_min_minutes)
        snap["minutes_since_going_update_num"] = minutes
        changed = pd.to_numeric(snap.get("going_changed_since_previous_observation", pd.Series(0, index=snap.index)), errors="coerce").fillna(0)
        uncertainty = pd.to_numeric(snap.get("track_state_uncertainty", pd.Series(np.nan, index=snap.index)), errors="coerce")
        snap["track_transition_alert"] = changed.gt(0) | uncertainty.ge(0.45) | snap["track_freshness_bucket"].eq("stale")
        group_cols = ["decision_label", "venue", "track_freshness_bucket"]
        existing_group_cols = [c for c in group_cols if c in snap.columns]
        snapshot_report = (
            snap.groupby(existing_group_cols, dropna=False)
            .agg(
                rows=("race_id", "size"),
                races=("race_id", "nunique"),
                avg_minutes_since_update=("minutes_since_going_update_num", "mean"),
                max_minutes_since_update=("minutes_since_going_update_num", "max"),
                transition_alerts=("track_transition_alert", "sum"),
            )
            .reset_index()
            if existing_group_cols
            else pd.DataFrame()
        )
        snap.to_csv(out_dir / "decision_snapshot_track_freshness_rows.csv", index=False, encoding="utf-8-sig")
        snapshot_report.to_csv(out_dir / "decision_snapshot_track_freshness_summary.csv", index=False, encoding="utf-8-sig")

    failures = []
    warnings = []
    if track.empty:
        failures.append("live track condition csv missing or empty")
    elif venue_report.empty or not bool(venue_report["complete"].all()):
        failures.append("latest track condition is incomplete for one or more venues")
    if not snapshots.empty and "track_freshness_bucket" in snapshots.columns:
        # Kept for backward compatibility if a caller already supplies the bucket.
        pass
    stale_snapshot_rows = 0
    transition_alert_rows = 0
    if not snapshot_report.empty:
        stale_snapshot_rows = int(snapshot_report.loc[snapshot_report["track_freshness_bucket"].eq("stale"), "rows"].sum())
        transition_alert_rows = int(snapshot_report["transition_alerts"].sum())
        if stale_snapshot_rows:
            warnings.append(f"stale track state used by {stale_snapshot_rows} snapshot rows")
        if transition_alert_rows:
            warnings.append(f"track transition/staleness alerts in {transition_alert_rows} snapshot rows")

    status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "track_csv": str(project_path(args.track_csv)),
        "snapshots_csv": str(project_path(args.snapshots_csv)),
        "track_rows": int(len(track)),
        "snapshot_rows": int(len(snapshots)),
        "venue_latest": venue_report.to_dict(orient="records"),
        "snapshot_freshness_summary": snapshot_report.to_dict(orient="records") if not snapshot_report.empty else [],
        "fresh_max_minutes": args.fresh_max_minutes,
        "stale_min_minutes": args.stale_min_minutes,
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "next_action": "For live BUY, require fresh/aging track state; stale or changed regimes should stay WAIT/SHADOW unless explicitly tested.",
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
