from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.finalize_operational_quality import _reason_columns
from scripts.optimize_operational_win_addon import _json_default
from src.utils.paths import ensure_dir, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Add dashboard-facing buy/risk/stake explanation columns to ticket CSV.")
    parser.add_argument("--tickets-csv", required=True)
    parser.add_argument("--output-csv", default="outputs/analysis/live_optimized_strategy_v1/standard_explained_tickets.csv")
    parser.add_argument("--mode-label", default="standard")
    args = parser.parse_args()

    df = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    out = _reason_columns(df, args.mode_label)
    output = project_path(args.output_csv)
    ensure_dir(output.parent)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    payload = {
        "input": args.tickets_csv,
        "output": str(output),
        "rows": int(len(out)),
        "races": int(out["race_id"].nunique()) if "race_id" in out.columns else 0,
        "runtime_stake_yen": float(pd.to_numeric(out.get("runtime_stake_yen"), errors="coerce").fillna(0).sum()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
