from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_equipment_overlay import enrich, _metric
from scripts.optimize_operational_win_addon import _json_default
from src.utils.paths import ensure_dir, project_path


def apply_policy(tickets: pd.DataFrame, equipment: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = enrich(tickets, equipment)
    pair_first = out["ticket_type"].isin(["wide", "umaren"]) & pd.to_numeric(
        out["ticket_equipment_first_or_reapply_flag"], errors="coerce"
    ).fillna(0).eq(1)
    out["equipment_overlay_pair_first_blinker_flag"] = pair_first.astype(int)
    out["equipment_overlay_action"] = "KEEP"
    out["pre_equipment_overlay_stake_yen"] = out["runtime_stake_yen"]
    if mode == "exclude_pair_first_blinker":
        out.loc[pair_first, "runtime_stake_yen"] = 0.0
        out.loc[pair_first, "equipment_overlay_action"] = "SKIP_PAIR_FIRST_BLINKER"
    elif mode == "reduce_pair_first_blinker_50":
        out.loc[pair_first, "runtime_stake_yen"] = (
            np.floor(out.loc[pair_first, "runtime_stake_yen"] * 0.5 / 100.0) * 100.0
        ).clip(lower=100.0)
        out.loc[pair_first, "equipment_overlay_action"] = "REDUCE_PAIR_FIRST_BLINKER"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    pay = pd.to_numeric(out.get("runtime_backtest_pay_per100"), errors="coerce").fillna(
        pd.to_numeric(out.get("quote_pay_proxy_per100"), errors="coerce").fillna(0.0)
    )
    out["runtime_return_yen"] = np.where(out["hit"].astype(bool), pay * out["runtime_stake_yen"] / 100.0, 0.0)
    out["runtime_reason"] = out.get("runtime_reason", "").astype(str) + "|equipment_pair_blinker_overlay"
    out = out[out["runtime_stake_yen"].gt(0)].copy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a pair-ticket blinker overlay based on TARGET equipment data.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/live_runtime_safety_overlay_v1/live_safety_overlaid_tickets.csv")
    parser.add_argument("--equipment-csv", default="data/processed/target/equipment_features.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/equipment_pair_blinker_overlay_v1")
    parser.add_argument("--mode", choices=["exclude_pair_first_blinker", "reduce_pair_first_blinker_50"], default="reduce_pair_first_blinker_50")
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    equipment = pd.read_csv(project_path(args.equipment_csv), dtype={"race_id": str}, low_memory=False)
    selected = apply_policy(tickets, equipment, args.mode)
    out_dir = ensure_dir(project_path(args.output_dir))
    selected.to_csv(out_dir / f"{args.mode}_tickets.csv", index=False, encoding="utf-8-sig")
    metrics = pd.DataFrame([_metric(tickets, "base"), _metric(selected, args.mode)])
    metrics.to_csv(out_dir / f"{args.mode}_metrics.csv", index=False, encoding="utf-8-sig")
    payload = {
        "mode": args.mode,
        "output_tickets": str(out_dir / f"{args.mode}_tickets.csv"),
        "metrics": metrics.to_dict(orient="records"),
    }
    with (out_dir / f"{args.mode}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
