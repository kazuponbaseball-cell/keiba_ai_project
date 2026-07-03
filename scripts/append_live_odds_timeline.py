from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False)


def _append_unique(existing_path: Path, new_rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    ensure_dir(existing_path.parent)
    if existing_path.exists() and existing_path.stat().st_size > 0:
        existing = pd.read_csv(existing_path, dtype={"race_id": str}, low_memory=False)
        combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        combined = new_rows.copy()
    if not combined.empty:
        combined = combined.drop_duplicates(keys, keep="last").sort_values(keys)
    combined.to_csv(existing_path, index=False, encoding="utf-8-sig")
    return combined


def _append_pair(pair_latest: pd.DataFrame, output_csv: Path, decision_label: str, captured_at: str) -> dict:
    cols = [
        "race_id",
        "ticket_type",
        "a_no",
        "b_no",
        "live_pay_per100",
        "live_odds",
        "popularity",
        "snapshot_at",
        "decision_label",
        "captured_at",
        "parser_mode",
    ]
    if pair_latest.empty:
        if not output_csv.exists():
            ensure_dir(output_csv.parent)
            pd.DataFrame(columns=cols).to_csv(output_csv, index=False, encoding="utf-8-sig")
        return {"output_csv": str(output_csv), "new_rows": 0, "total_rows": int(len(_read_optional_csv(output_csv)))}
    out = pair_latest.copy()
    for col in ["race_id", "ticket_type", "a_no", "b_no", "live_pay_per100"]:
        if col not in out.columns:
            raise ValueError(f"pair odds csv missing column: {col}")
    if "snapshot_at" not in out.columns:
        out["snapshot_at"] = captured_at
    out["decision_label"] = decision_label
    out["captured_at"] = captured_at
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[cols].copy()
    for col in ["a_no", "b_no", "live_pay_per100", "live_odds", "popularity"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["race_id"].notna() & out["ticket_type"].notna() & out["a_no"].notna() & out["b_no"].notna()].copy()
    combined = _append_unique(output_csv, out, ["race_id", "ticket_type", "a_no", "b_no", "snapshot_at", "decision_label"])
    return {"output_csv": str(output_csv), "new_rows": int(len(out)), "total_rows": int(len(combined))}


def _append_single(single_latest: pd.DataFrame, output_csv: Path, decision_label: str, captured_at: str) -> dict:
    cols = [
        "race_id",
        "horse_number",
        "snapshot_at",
        "win_odds",
        "place_odds_min",
        "place_odds_max",
        "popularity_estimated",
        "decision_label",
        "captured_at",
        "parser_mode",
    ]
    if single_latest.empty:
        if not output_csv.exists():
            ensure_dir(output_csv.parent)
            pd.DataFrame(columns=cols).to_csv(output_csv, index=False, encoding="utf-8-sig")
        return {"output_csv": str(output_csv), "new_rows": 0, "total_rows": int(len(_read_optional_csv(output_csv)))}
    out = single_latest.copy()
    rename = {
        "horse_no": "horse_number",
        "live_win_odds": "win_odds",
        "live_place_odds_min": "place_odds_min",
        "live_place_odds_max": "place_odds_max",
        "live_popularity": "popularity_estimated",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    for col in ["race_id", "horse_number"]:
        if col not in out.columns:
            raise ValueError(f"single odds csv missing column: {col}")
    if "snapshot_at" not in out.columns:
        out["snapshot_at"] = captured_at
    out["decision_label"] = decision_label
    out["captured_at"] = captured_at
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[cols].copy()
    for col in ["horse_number", "win_odds", "place_odds_min", "place_odds_max", "popularity_estimated"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["race_id"].notna() & out["horse_number"].notna()].copy()
    combined = _append_unique(output_csv, out, ["race_id", "horse_number", "snapshot_at", "decision_label"])
    return {"output_csv": str(output_csv), "new_rows": int(len(out)), "total_rows": int(len(combined))}


def _filter_race_ids(df: pd.DataFrame, race_ids: list[str]) -> pd.DataFrame:
    if df.empty or not race_ids or "race_id" not in df.columns:
        return df
    allowed = {str(race_id).strip() for race_id in race_ids if str(race_id).strip()}
    if not allowed:
        return df
    return df[df["race_id"].astype(str).isin(allowed)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Append latest live odds CSVs to timestamped odds timeline CSVs.")
    parser.add_argument("--pair-latest-csv", default="data/processed/live_odds/realtime_pair_odds_latest.csv")
    parser.add_argument("--single-latest-csv", default="data/processed/live_odds/realtime_single_odds_latest.csv")
    parser.add_argument("--pair-timeline-csv", default="data/processed/live_odds/realtime_pair_odds_timeline.csv")
    parser.add_argument("--single-timeline-csv", default="data/processed/live_odds/realtime_single_odds_timeline.csv")
    parser.add_argument("--decision-label", default="manual", help="Example: T-10, T-5, T-3, final_check")
    parser.add_argument("--race-ids", nargs="*", default=[], help="Optional race IDs to append. Defaults to all latest rows.")
    parser.add_argument("--captured-at", default="", help="Optional yyyyMMdd_HHMMSS. Defaults to now.")
    parser.add_argument("--summary-json", default="outputs/analysis/live_odds_timeline/append_summary.json")
    args = parser.parse_args()

    captured_at = args.captured_at or _now_stamp()
    pair_latest = _read_optional_csv(project_path(args.pair_latest_csv))
    single_latest = _read_optional_csv(project_path(args.single_latest_csv))
    pair_latest = _filter_race_ids(pair_latest, args.race_ids)
    single_latest = _filter_race_ids(single_latest, args.race_ids)
    summary = {
        "decision_label": args.decision_label,
        "race_ids": args.race_ids,
        "captured_at": captured_at,
        "pair": _append_pair(pair_latest, project_path(args.pair_timeline_csv), args.decision_label, captured_at),
        "single": _append_single(single_latest, project_path(args.single_timeline_csv), args.decision_label, captured_at),
    }
    summary_path = project_path(args.summary_json)
    ensure_dir(summary_path.parent)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
