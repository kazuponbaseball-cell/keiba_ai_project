from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def latest_file(pattern: str) -> Path | None:
    files = list(ROOT.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def read_csv_safe(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def column_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        if col not in df.columns:
            out[col] = {"present": False}
            continue
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            out[col] = {
                "present": True,
                "non_null": int(series.notna().sum()),
                "nunique": int(series.nunique(dropna=True)),
                "min": float(numeric.min()),
                "mean": float(numeric.mean()),
                "max": float(numeric.max()),
            }
        else:
            out[col] = {
                "present": True,
                "non_null": int(series.notna().sum()),
                "nunique": int(series.nunique(dropna=True)),
                "top_values": series.value_counts(dropna=False).head(5).to_dict(),
            }
    return out


def any_columns(df: pd.DataFrame, names: list[str], contains: list[str] | None = None) -> list[str]:
    found = [name for name in names if name in df.columns]
    for token in contains or []:
        found.extend([col for col in df.columns if token.lower() in str(col).lower()])
    return sorted(dict.fromkeys(found))


def status_from_columns(df: pd.DataFrame, columns: list[str], *, require_variation: bool = False) -> str:
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return "missing"
    if require_variation:
        varied = [col for col in existing if df[col].nunique(dropna=True) > 1]
        return "ok" if varied else "weak_constant"
    non_null = [col for col in existing if df[col].notna().any()]
    return "ok" if non_null else "present_empty"


def date_keys_from_entry(entry: pd.DataFrame) -> set[str]:
    keys: set[str] = set()
    for col in ["日付S", "日付"]:
        if col not in entry.columns:
            continue
        raw = entry[col].astype("string").dropna().str.strip()
        for value in raw:
            for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y%m%d", "%y%m%d"):
                try:
                    parsed = datetime.strptime(value, fmt)
                    if 2000 <= parsed.year <= 2099:
                        keys.add(parsed.strftime("%Y%m%d"))
                    break
                except Exception:
                    pass
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit current live/preday inputs against tested betting conditions.")
    parser.add_argument("--entry-csv", default="")
    parser.add_argument("--prediction-csv", default="")
    parser.add_argument("--single-odds-csv", default="data/processed/live_odds/realtime_single_odds_latest.csv")
    parser.add_argument("--pair-odds-csv", default="data/processed/live_odds/realtime_pair_odds_latest.csv")
    parser.add_argument("--single-timeline-csv", default="data/processed/live_odds/realtime_single_odds_timeline.csv")
    parser.add_argument("--pair-timeline-csv", default="data/processed/live_odds/realtime_pair_odds_timeline.csv")
    parser.add_argument(
        "--track-condition-csv",
        default="data/processed/live_track_conditions/current_track_conditions.csv",
    )
    parser.add_argument(
        "--tickets-csv",
        default="outputs/analysis/current_strongest_runtime_v1/selected_after_live_safety.csv",
    )
    parser.add_argument("--output-json", default="outputs/analysis/live_feature_parity_audit/summary.json")
    args = parser.parse_args()

    entry_path = project_path(args.entry_csv) if args.entry_csv else latest_file(
        "data/datasets/inference/weekly/entry_snapshot_*_enriched_odds_workout_knowledge.csv"
    ) or latest_file(
        "data/datasets/inference/weekly/entry_snapshot_*_enriched_odds.csv"
    )
    prediction_path = project_path(args.prediction_csv) if args.prediction_csv else (
        latest_file("outputs/predictions/preday_strongest_feature_parity/baseline_predictions_*.csv")
        or latest_file("outputs/predictions/preday_netkeiba_enriched_odds_history_context/baseline_predictions_*.csv")
    )
    single_path = project_path(args.single_odds_csv)
    pair_path = project_path(args.pair_odds_csv)
    single_timeline_path = project_path(args.single_timeline_csv)
    pair_timeline_path = project_path(args.pair_timeline_csv)
    track_path = project_path(args.track_condition_csv)
    tickets_path = project_path(args.tickets_csv)

    entry = read_csv_safe(entry_path)
    prediction = read_csv_safe(prediction_path)
    single = read_csv_safe(single_path)
    pair = read_csv_safe(pair_path)
    single_timeline = read_csv_safe(single_timeline_path)
    pair_timeline = read_csv_safe(pair_timeline_path)
    track = read_csv_safe(track_path)
    tickets = read_csv_safe(tickets_path)

    current_date_keys = date_keys_from_entry(entry)
    ticket_race_ids = tickets.get("race_id", pd.Series(dtype="string")).astype("string").dropna()
    current_ticket_rows = int(ticket_race_ids.str.slice(0, 8).isin(current_date_keys).sum()) if current_date_keys else 0

    groups: dict[str, Any] = {}
    pace_cols = [
        "expected_pace",
        "front_running_tendency",
        "closing_tendency",
        "race_front_runner_count",
        "race_early_pressure_score",
    ]
    groups["pace_position_baseline"] = {
        "status": status_from_columns(prediction, pace_cols, require_variation=True),
        "columns": column_summary(prediction, pace_cols),
        "note": "Preday-safe pace and position features generated from historical context.",
    }

    deep_cols = [
        "projected_front5_prob",
        "horse_front_run_rate_past5",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "pace_fit_score",
        "front_advantage_score",
    ]
    ticket_deep_cols = []
    for col in deep_cols:
        ticket_deep_cols.extend([col, f"anchor_{col}", f"partner_{col}"])
    ticket_deep_cols.extend(["pace_fit_pair_score", "workout_pair_score"])
    ticket_deep_cols = [col for col in ticket_deep_cols if col in tickets.columns]
    groups["front5_pair_model_inputs"] = {
        "status": status_from_columns(tickets, ticket_deep_cols, require_variation=True),
        "current_ticket_rows": current_ticket_rows,
        "columns": column_summary(tickets, ticket_deep_cols),
        "note": (
            "High-ROI front-position proxy is present in current-week final ticket rows."
            if current_ticket_rows
            else "High-ROI ticket model features exist in the tested ticket universe; current-week final ticket rows are required before purchase."
        ),
    }

    groups["official_live_odds"] = {
        "status": "ok" if len(single) > 0 and len(pair) > 0 else "missing",
        "single_rows": int(len(single)),
        "pair_rows": int(len(pair)),
        "single_timeline_rows": int(len(single_timeline)),
        "pair_timeline_rows": int(len(pair_timeline)),
        "note": "Current official single/pair odds are available; timeline strength depends on repeated refresh snapshots.",
    }

    same_day_cols = any_columns(prediction, ["same_day_bias_ready"], contains=["same_day_"])
    prior_races = pd.to_numeric(prediction.get("same_day_prior_races"), errors="coerce")
    same_day_ready = bool(prior_races.notna().any() and prior_races.max() > 0)
    groups["same_day_bias"] = {
        "status": status_from_columns(prediction, same_day_cols) if same_day_ready else "pending_until_prior_races",
        "columns": same_day_cols[:30],
        "note": "This cannot be fully known before same-day prior races finish. It should stay pending in preday mode.",
    }

    body_cols = any_columns(entry, ["馬体重", "馬体重増減"], contains=["body_weight", "live_body_weight"])
    groups["live_body_weight"] = {
        "status": "pending_until_body_weight_publish" if not body_cols else status_from_columns(entry, body_cols),
        "columns": body_cols[:30],
        "note": "Final safety overlay should run after official body-weight publication.",
    }

    track_cols = any_columns(entry, ["馬場状態"], contains=["cushion", "moisture", "含水", "クッション"])
    ticket_track_cols = any_columns(
        tickets,
        ["runtime_policy_gate"],
        contains=[
            "runtime_going",
            "runtime_track_condition",
            "runtime_soft_heavy",
            "runtime_hakodate",
        ],
    )
    required_track_cols = {"venue", "turf_going", "dirt_going"}
    live_track_complete = False
    if not track.empty and required_track_cols.issubset(track.columns):
        turf_ok = track["turf_going"].astype("string").fillna("").str.strip().ne("").all()
        dirt_ok = track["dirt_going"].astype("string").fillna("").str.strip().ne("").all()
        live_track_complete = bool(turf_ok and dirt_ok)
    entry_track_status = status_from_columns(entry, track_cols) if track_cols else "missing"
    ticket_track_status = (
        status_from_columns(tickets, ticket_track_cols)
        if ticket_track_cols
        else "missing"
    )
    if live_track_complete and ticket_track_status == "ok":
        track_status = "ok_runtime"
    elif live_track_complete:
        track_status = "live_source_ok_ticket_missing"
    elif entry_track_status == "ok":
        track_status = "ok_entry"
    else:
        track_status = entry_track_status
    groups["track_condition"] = {
        "status": track_status,
        "entry_status": entry_track_status,
        "ticket_status": ticket_track_status,
        "live_track_rows": int(len(track)),
        "live_track_complete": live_track_complete,
        "entry_columns": column_summary(entry, track_cols[:20]) if track_cols else {},
        "ticket_columns": column_summary(tickets, ticket_track_cols[:40]) if ticket_track_cols else {},
        "live_track_sample": track.head(10).to_dict(orient="records") if not track.empty else [],
        "note": (
            "Current operation uses JRA live/current going at runtime. Entry 馬場状態 may be empty in preday snapshots; "
            "the final ticket layer must carry runtime_going/runtime_going_class and skip_soft_heavy policy flags."
        ),
    }

    workout_cols = any_columns(entry, [], contains=["workout", "追切", "追い切"])
    ticket_workout_cols = any_columns(tickets, [], contains=["workout"])
    workout_status = (
        "ok"
        if workout_cols and ticket_workout_cols
        else "ok_in_ticket_layer_missing_in_current_entry"
        if ticket_workout_cols and not workout_cols
        else status_from_columns(entry, workout_cols)
    )
    groups["workout_overlay"] = {
        "status": workout_status,
        "entry_columns": workout_cols[:30],
        "ticket_columns": ticket_workout_cols[:30],
        "note": (
            "Trainer-specific workout factors are present in both current entry and ticket rows."
            if workout_status == "ok"
            else "Trainer-specific workout factors are present in ticket rows, but the current entry snapshot does not carry them."
        ),
    }

    final_ticket_status = "missing_current_week" if current_ticket_rows == 0 else "ok"
    groups["final_ticket_layer_for_current_week"] = {
        "status": final_ticket_status,
        "tested_ticket_rows": int(len(tickets)),
        "current_ticket_rows": current_ticket_rows,
        "current_date_keys": sorted(current_date_keys),
        "note": (
            "Current-week final ticket rows are available for the dashboard."
            if final_ticket_status == "ok"
            else "Dashboard final tickets remain 0 until the runtime ticket pipeline is run for the current race IDs."
        ),
    }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": {
            "entry_csv": str(entry_path) if entry_path else "",
            "prediction_csv": str(prediction_path) if prediction_path else "",
            "single_odds_csv": str(single_path),
            "pair_odds_csv": str(pair_path),
            "track_condition_csv": str(track_path),
            "tickets_csv": str(tickets_path),
        },
        "counts": {
            "entry_rows": int(len(entry)),
            "prediction_rows": int(len(prediction)),
            "single_odds_rows": int(len(single)),
            "pair_odds_rows": int(len(pair)),
            "track_condition_rows": int(len(track)),
            "ticket_rows": int(len(tickets)),
            "current_ticket_rows": current_ticket_rows,
        },
        "groups": groups,
    }

    output_path = project_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
