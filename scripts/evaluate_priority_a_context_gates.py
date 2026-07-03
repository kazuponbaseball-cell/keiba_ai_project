from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_priority_s_betting_policy import _metrics
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _between(series: pd.Series, lo: float | None, hi: float | None) -> pd.Series:
    values = _num(series)
    mask = pd.Series(True, index=series.index)
    if lo is not None:
        mask &= values.ge(lo)
    if hi is not None:
        mask &= values.le(hi)
    return mask.fillna(False)


def _yearly(df: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    if "test_year" not in df.columns:
        return rows
    for year, group in df.groupby("test_year"):
        row = _metrics(group, f"{label}_{int(year)}")
        row["test_year"] = int(year)
        rows.append(row)
    return rows


def _evaluate(tickets: pd.DataFrame, label: str, mask: pd.Series, params: dict) -> dict:
    subset = tickets[mask].copy()
    row = _metrics(subset, label)
    row.update(params)
    row["keep_rate"] = len(subset) / len(tickets) if len(tickets) else 0.0
    yearly = _yearly(subset, label)
    row["min_year_roi"] = min((y["roi"] for y in yearly), default=0.0)
    row["max_year_roi"] = max((y["roi"] for y in yearly), default=0.0)
    row["year_roi_spread"] = row["max_year_roi"] - row["min_year_roi"]
    row["robust_score"] = (
        row["roi"] * 0.45
        + row["min_year_roi"] * 0.35
        + row["race_hit_rate"] * 0.55
        + row["keep_rate"] * 0.20
        - row["year_roi_spread"] * 0.12
    )
    return row


def _param(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate priority-A context gates after optimized ROI gate.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/optimized_factor_gate_v1/optimized_gate_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/priority_a_context_gates_v1")
    parser.add_argument("--min-races", type=int, default=250)
    parser.add_argument("--min-keep-rate", type=float, default=0.40)
    parser.add_argument("--min-hit", type=float, default=0.17)
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    out_dir = ensure_dir(project_path(args.output_dir))

    rows = [_evaluate(tickets, "optimized_gate_baseline", pd.Series(True, index=tickets.index), {})]

    single_specs = [
        ("race_difficulty_mid", _between(tickets["race_difficulty_score"], 0.40, 0.74), {"race_difficulty_band": "0.40-0.74"}),
        ("race_difficulty_not_low", _num(tickets["race_difficulty_score"]).ge(0.40), {"race_difficulty_min": 0.40}),
        ("race_difficulty_not_high", _num(tickets["race_difficulty_score"]).le(0.74), {"race_difficulty_max": 0.74}),
        ("pace_collapse_mid_high", _num(tickets["race_pace_collapse"]).ge(0.40), {"pace_collapse_min": 0.40}),
        ("pace_collapse_mid", _between(tickets["race_pace_collapse"], 0.40, 0.80), {"pace_collapse_band": "0.40-0.80"}),
        ("slow_risk_not_extreme", _num(tickets["race_slow_risk"]).le(0.85), {"slow_risk_max": 0.85}),
        ("bias_volatility_known", _num(tickets["race_bias_volatility"]).notna(), {"bias_volatility": "known"}),
        ("bias_volatility_low_mid", _between(tickets["race_bias_volatility"], 0.0, 0.70), {"bias_volatility_max": 0.70}),
        ("front_pressure_mid_high", _num(tickets["race_front_pressure"]).ge(0.40), {"front_pressure_min": 0.40}),
    ]
    for label, mask, params in single_specs:
        rows.append(_evaluate(tickets, label, mask, params))

    for diff_lo, diff_hi, pace_lo, pace_hi, bias_hi, slow_hi in product(
        [None, 0.35, 0.45],
        [None, 0.74, 0.85],
        [None, 0.35, 0.45],
        [None, 0.80],
        [None, 0.70, 0.90],
        [None, 0.80, 0.90],
    ):
        params = {
            "race_difficulty_min": diff_lo,
            "race_difficulty_max": diff_hi,
            "pace_collapse_min": pace_lo,
            "pace_collapse_max": pace_hi,
            "bias_volatility_max": bias_hi,
            "slow_risk_max": slow_hi,
        }
        mask = pd.Series(True, index=tickets.index)
        mask &= _between(tickets["race_difficulty_score"], diff_lo, diff_hi)
        mask &= _between(tickets["race_pace_collapse"], pace_lo, pace_hi)
        if bias_hi is not None:
            mask &= _num(tickets["race_bias_volatility"]).le(bias_hi).fillna(False)
        if slow_hi is not None:
            mask &= _num(tickets["race_slow_risk"]).le(slow_hi).fillna(False)
        metric = _evaluate(tickets, "priority_a_grid", mask, params)
        if metric["races"] >= args.min_races and metric["keep_rate"] >= args.min_keep_rate and metric["race_hit_rate"] >= args.min_hit:
            rows.append(metric)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["robust_score", "roi", "races"], ascending=[False, False, False])
    summary.to_csv(out_dir / "priority_a_context_gate_summary.csv", index=False, encoding="utf-8-sig")

    best = summary.iloc[0].to_dict() if not summary.empty else None
    best_tickets = pd.DataFrame()
    best_yearly = pd.DataFrame()
    if best and best["policy"] == "priority_a_grid":
        mask = pd.Series(True, index=tickets.index)
        mask &= _between(tickets["race_difficulty_score"], _param(best.get("race_difficulty_min")), _param(best.get("race_difficulty_max")))
        mask &= _between(tickets["race_pace_collapse"], _param(best.get("pace_collapse_min")), _param(best.get("pace_collapse_max")))
        bias_max = _param(best.get("bias_volatility_max"))
        slow_max = _param(best.get("slow_risk_max"))
        if bias_max is not None:
            mask &= _num(tickets["race_bias_volatility"]).le(bias_max).fillna(False)
        if slow_max is not None:
            mask &= _num(tickets["race_slow_risk"]).le(slow_max).fillna(False)
        best_tickets = tickets[mask].copy()
    elif best:
        # Single-condition labels are analysis-only; keep the fixed optimized gate as production tickets.
        best_tickets = tickets.copy()

    if not best_tickets.empty:
        best_tickets.to_csv(out_dir / "priority_a_best_tickets.csv", index=False, encoding="utf-8-sig")
        best_yearly = pd.DataFrame(_yearly(best_tickets, "priority_a_best"))
        best_yearly.to_csv(out_dir / "priority_a_best_yearly.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "baseline": _metrics(tickets, "optimized_gate_baseline"),
        "best": best,
        "best_yearly": best_yearly.to_dict(orient="records"),
        "note": "Priority-A checks whether race difficulty, pace collapse, slow risk, and bias volatility improve the already-optimized factor gate.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
