from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_operational_win_addon import _json_default
from src.utils.paths import ensure_dir, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze pre-race tickets for true external validation.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/final_operational_quality_v1/standard_explained_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/validation/external_freeze")
    parser.add_argument("--batch-name", default="")
    parser.add_argument("--max-races", type=int, default=30)
    parser.add_argument("--date-from", default="")
    parser.add_argument("--date-to", default="")
    args = parser.parse_args()

    df = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    if "date_key" not in df.columns:
        df["date_key"] = df["race_id"].astype(str).str[:8]
    df["date"] = pd.to_datetime(df["date_key"], errors="coerce")
    stake = pd.to_numeric(df.get("runtime_stake_yen"), errors="coerce").fillna(0)
    df = df[stake.gt(0)].copy()
    if args.date_from:
        df = df[df["date"].ge(pd.to_datetime(args.date_from))]
    if args.date_to:
        df = df[df["date"].le(pd.to_datetime(args.date_to))]

    race_order = df[["race_id", "date"]].drop_duplicates().sort_values(["date", "race_id"]).head(args.max_races)
    frozen = df[df["race_id"].isin(set(race_order["race_id"]))].copy()
    frozen["freeze_created_at"] = datetime.now().isoformat(timespec="seconds")
    frozen["freeze_batch_name"] = args.batch_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    frozen["validation_status"] = "frozen_pre_result"

    out_dir = ensure_dir(project_path(args.output_dir))
    batch = frozen["freeze_batch_name"].iloc[0] if len(frozen) else (args.batch_name or "empty_batch")
    out_csv = out_dir / f"{batch}.csv"
    out_json = out_dir / f"{batch}.json"
    frozen.to_csv(out_csv, index=False, encoding="utf-8-sig")
    payload = {
        "batch": batch,
        "source": args.tickets_csv,
        "output_csv": str(out_csv),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tickets": int(len(frozen)),
        "races": int(frozen["race_id"].nunique()) if len(frozen) else 0,
        "date_from": str(frozen["date"].min().date()) if len(frozen) else "",
        "date_to": str(frozen["date"].max().date()) if len(frozen) else "",
        "total_stake_yen": float(pd.to_numeric(frozen.get("runtime_stake_yen"), errors="coerce").fillna(0).sum()) if len(frozen) else 0.0,
        "rules": [
            "Do not tune model using this batch before all results are known.",
            "Do not edit frozen tickets after creation.",
            "Evaluate only after race results/payoffs are appended.",
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
