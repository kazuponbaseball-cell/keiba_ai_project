from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_context_cut_and_monthly_plan import _scale_tickets, _scaled_metrics
from scripts.optimize_operational_win_addon import _json_default
from src.utils.paths import ensure_dir, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the standard staking plan to final selected tickets.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/priority_context_factor_overlay_v1/priority_context_selected_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/standard_staking_plan_v1")
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--race-cap", type=float, default=3000.0)
    parser.add_argument("--day-cap", type=float, default=8000.0)
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    scaled = _scale_tickets(tickets, multiplier=args.multiplier, race_cap=args.race_cap, day_cap=args.day_cap)
    # Make downstream dashboard/export scripts use the standard stake as the runtime stake.
    scaled["base_runtime_stake_yen"] = scaled.get("runtime_stake_yen", 0)
    scaled["base_runtime_return_yen"] = scaled.get("runtime_return_yen", 0)
    scaled["runtime_stake_yen"] = scaled["scaled_stake_yen"]
    scaled["runtime_return_yen"] = scaled["scaled_return_yen"]
    scaled["runtime_ticket_status"] = scaled.get("runtime_ticket_status", "買い").astype(str) + f" x{args.multiplier:g}"
    scaled["standard_staking_multiplier"] = args.multiplier
    scaled["standard_race_cap_yen"] = args.race_cap
    scaled["standard_day_cap_yen"] = args.day_cap

    out_dir = ensure_dir(project_path(args.output_dir))
    ticket_path = out_dir / "standard_staked_tickets.csv"
    scaled.to_csv(ticket_path, index=False, encoding="utf-8-sig")
    metrics = _scaled_metrics(scaled.rename(columns={"runtime_stake_yen": "_runtime_stake_yen_original"}), f"standard_x{args.multiplier:g}")
    # _scaled_metrics reads scaled_* columns, so also provide a plain runtime metric style summary.
    runtime_like = scaled.copy()
    runtime_like["scaled_stake_yen"] = runtime_like["runtime_stake_yen"]
    runtime_like["scaled_return_yen"] = runtime_like["runtime_return_yen"]
    metrics = _scaled_metrics(runtime_like, f"standard_x{args.multiplier:g}")
    pd.DataFrame([metrics]).to_csv(out_dir / "standard_staking_metrics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "tickets_csv": args.tickets_csv,
        "output_tickets": str(ticket_path),
        "multiplier": args.multiplier,
        "race_cap_yen": args.race_cap,
        "day_cap_yen": args.day_cap,
        "metrics": metrics,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
