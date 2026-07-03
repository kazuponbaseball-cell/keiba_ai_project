from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_priority_s_betting_policy import _metrics
from scripts.summarize_factor_roi_effectiveness import _apply_gate
from src.utils.paths import ensure_dir, project_path


DEFAULT_PARAMS = {
    "race_difficulty_max": None,
    "late_value_min": None,
    "pair_quinella_min": 0.58,
    "partner_quinella_min": 0.54,
    "market_overlay_min": 0.55,
    "front_min": None,
    "anchor_overpop_max": 0.66,
    "partner_value_min": 0.50,
    "partner_overpop_max": None,
    "partner_odds_max": None,
}


def _yearly(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "test_year" not in df.columns:
        return pd.DataFrame()
    for year, group in df.groupby("test_year"):
        row = _metrics(group, f"optimized_gate_{int(year)}")
        row["test_year"] = int(year)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the current optimized ROI gate to already-selected ticket candidates.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/vertical_context_roi_v1/baseline_selected_tickets_with_vertical_scores.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/optimized_factor_gate_v1")
    parser.add_argument("--params-json", default=None, help="Optional JSON object overriding DEFAULT_PARAMS.")
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    params = dict(DEFAULT_PARAMS)
    if args.params_json:
        params.update(json.loads(args.params_json))

    gated = tickets[_apply_gate(tickets, params)].copy()
    out_dir = ensure_dir(project_path(args.output_dir))
    gated.to_csv(out_dir / "optimized_gate_tickets.csv", index=False, encoding="utf-8-sig")
    yearly = _yearly(gated)
    yearly.to_csv(out_dir / "optimized_gate_yearly.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "params": params,
        "baseline": _metrics(tickets, "baseline_selected"),
        "optimized": _metrics(gated, "optimized_gate"),
        "yearly": yearly.to_dict(orient="records"),
        "note": "Fixed gate from factor_roi_effectiveness_v1: quinella + market overlay + vertical over/under-popularity.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
