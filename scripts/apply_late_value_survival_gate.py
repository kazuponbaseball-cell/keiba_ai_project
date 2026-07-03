from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_operational_win_addon import _json_default, _metrics
from src.utils.paths import ensure_dir, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a late-value-survival gate to operational tickets.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/operational_win_addon_1pt_v1/combined_ticket_profiles.csv")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--output-dir", default="outputs/analysis/late_value_survival_gate_v1")
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    score = pd.to_numeric(tickets.get("late_value_survives_score"), errors="coerce").fillna(1.0)
    gated = tickets[score.ge(args.threshold)].copy()
    out_dir = ensure_dir(project_path(args.output_dir))
    gated.to_csv(out_dir / "gated_ticket_profiles.csv", index=False, encoding="utf-8-sig")

    yearly = []
    if "year" in gated.columns:
        for year, group in gated.groupby("year"):
            row = _metrics(group, f"year_{int(year)}")
            row["year"] = int(year)
            yearly.append(row)
    summary = {
        "input": {
            "tickets_csv": args.tickets_csv,
            "threshold": args.threshold,
            "note": "This gate uses late_value_survives_score. Actual late odds drop/drift columns are currently not populated, so this is a value-survival proxy rather than true live odds movement.",
        },
        "ungated": _metrics(tickets, "ungated"),
        "gated": _metrics(gated, "late_value_survival_gated"),
        "yearly": yearly,
    }
    summary["delta_roi"] = summary["gated"]["roi"] - summary["ungated"]["roi"]
    summary["delta_profit_yen"] = summary["gated"]["profit_yen"] - summary["ungated"]["profit_yen"]
    pd.DataFrame([summary["ungated"], summary["gated"]]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(out_dir / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
