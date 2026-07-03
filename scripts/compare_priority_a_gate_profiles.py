from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_priority_s_betting_policy import _metrics
from scripts.evaluate_priority_a_context_gates import _between, _num, _yearly
from src.utils.paths import ensure_dir, project_path


PROFILES = [
    {
        "profile": "base_optimized_156",
        "race_difficulty_min": None,
        "race_difficulty_max": None,
        "pace_collapse_min": None,
        "pace_collapse_max": None,
        "description": "Optimized factor gate only.",
    },
    {
        "profile": "strict_190",
        "race_difficulty_min": 0.35,
        "race_difficulty_max": None,
        "pace_collapse_min": None,
        "pace_collapse_max": 0.80,
        "description": "Priority-A best profile.",
    },
    {
        "profile": "middle_175_candidate",
        "race_difficulty_min": 0.30,
        "race_difficulty_max": None,
        "pace_collapse_min": None,
        "pace_collapse_max": 0.85,
        "description": "Loosen both difficulty floor and collapse cap.",
    },
    {
        "profile": "middle_more_races",
        "race_difficulty_min": 0.25,
        "race_difficulty_max": None,
        "pace_collapse_min": None,
        "pace_collapse_max": 0.85,
        "description": "More tickets while keeping extreme collapses out.",
    },
    {
        "profile": "difficulty_only",
        "race_difficulty_min": 0.35,
        "race_difficulty_max": None,
        "pace_collapse_min": None,
        "pace_collapse_max": None,
        "description": "Only avoid too-easy/no-edge races.",
    },
    {
        "profile": "pace_only",
        "race_difficulty_min": None,
        "race_difficulty_max": None,
        "pace_collapse_min": None,
        "pace_collapse_max": 0.80,
        "description": "Only avoid extreme pace-collapse races.",
    },
]


def _mask(tickets: pd.DataFrame, profile: dict) -> pd.Series:
    mask = pd.Series(True, index=tickets.index)
    mask &= _between(tickets["race_difficulty_score"], profile["race_difficulty_min"], profile["race_difficulty_max"])
    mask &= _between(tickets["race_pace_collapse"], profile["pace_collapse_min"], profile["pace_collapse_max"])
    return mask.fillna(False)


def _score_profile(tickets: pd.DataFrame, profile: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    subset = tickets[_mask(tickets, profile)].copy()
    metric = _metrics(subset, profile["profile"])
    yearly = pd.DataFrame(_yearly(subset, profile["profile"]))
    min_year_roi = float(yearly["roi"].min()) if not yearly.empty else 0.0
    max_year_roi = float(yearly["roi"].max()) if not yearly.empty else 0.0
    metric.update(profile)
    metric["keep_rate"] = len(subset) / len(tickets) if len(tickets) else 0.0
    metric["min_year_roi"] = min_year_roi
    metric["max_year_roi"] = max_year_roi
    metric["year_roi_spread"] = max_year_roi - min_year_roi
    metric["robust_score"] = (
        metric["roi"] * 0.42
        + min_year_roi * 0.35
        + metric["race_hit_rate"] * 0.50
        + metric["keep_rate"] * 0.28
        - metric["year_roi_spread"] * 0.12
    )
    return metric, subset, yearly


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare practical Priority-A profiles between the 156% and 190% gates.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/optimized_factor_gate_v1/optimized_gate_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/priority_a_gate_profiles_v1")
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    out_dir = ensure_dir(project_path(args.output_dir))
    rows = []
    yearly_frames = []
    best_metric = None
    best_tickets = None
    best_yearly = None
    for profile in PROFILES:
        metric, subset, yearly = _score_profile(tickets, profile)
        rows.append(metric)
        if not yearly.empty:
            yearly["profile"] = profile["profile"]
            yearly_frames.append(yearly)
        if best_metric is None or metric["robust_score"] > best_metric["robust_score"]:
            best_metric = metric
            best_tickets = subset
            best_yearly = yearly

    summary = pd.DataFrame(rows).sort_values(["robust_score", "roi", "races"], ascending=[False, False, False])
    summary.to_csv(out_dir / "priority_a_profile_summary.csv", index=False, encoding="utf-8-sig")
    if yearly_frames:
        pd.concat(yearly_frames, ignore_index=True, sort=False).to_csv(out_dir / "priority_a_profile_yearly.csv", index=False, encoding="utf-8-sig")
    if best_tickets is not None:
        best_tickets.to_csv(out_dir / "best_profile_tickets.csv", index=False, encoding="utf-8-sig")
    if best_yearly is not None and not best_yearly.empty:
        best_yearly.to_csv(out_dir / "best_profile_yearly.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "best_profile": best_metric,
        "profiles": summary.to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
