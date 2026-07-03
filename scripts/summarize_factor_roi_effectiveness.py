from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_priority_s_betting_policy import _metrics
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _metric_row(tickets: pd.DataFrame, label: str, *, factor: str = "", condition: str = "") -> dict:
    row = _metrics(tickets, label)
    row["factor"] = factor
    row["condition"] = condition
    return row


def _quantile_segments(tickets: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    rows = [_metric_row(tickets, "all_selected", factor="baseline", condition="all")]
    for spec in specs:
        col = spec["column"]
        if col not in tickets.columns:
            continue
        values = _num(tickets[col])
        if values.notna().sum() == 0:
            continue
        direction = spec.get("direction", "high")
        q20 = values.quantile(0.20)
        q40 = values.quantile(0.40)
        q60 = values.quantile(0.60)
        q80 = values.quantile(0.80)
        cuts = [
            ("bottom20", values <= q20),
            ("low40", values <= q40),
            ("mid40_60", values.between(q40, q60, inclusive="both")),
            ("high40", values >= q60),
            ("top20", values >= q80),
        ]
        if direction == "low":
            cuts.extend(
                [
                    ("good_bottom20", values <= q20),
                    ("bad_top20", values >= q80),
                ]
            )
        else:
            cuts.extend(
                [
                    ("bad_bottom20", values <= q20),
                    ("good_top20", values >= q80),
                ]
            )
        for name, mask in cuts:
            subset = tickets[mask.fillna(False)].copy()
            if subset.empty:
                continue
            rows.append(
                _metric_row(
                    subset,
                    f"{spec['name']}:{name}",
                    factor=spec["name"],
                    condition=f"{col} {name}",
                )
            )
    out = pd.DataFrame(rows)
    base = out[out["policy"].eq("all_selected")].iloc[0]
    for col in ["roi", "race_hit_rate", "ticket_hit_rate", "wide_roi", "umaren_roi", "max_drawdown_yen"]:
        if col in out.columns:
            out[f"delta_{col}"] = out[col] - base[col]
    return out.sort_values(["roi", "races"], ascending=[False, False])


def _apply_gate(tickets: pd.DataFrame, params: dict) -> pd.Series:
    mask = pd.Series(True, index=tickets.index)
    if params["race_difficulty_max"] is not None:
        mask &= _num(tickets["race_difficulty_score"]).le(params["race_difficulty_max"])
    if params["late_value_min"] is not None:
        mask &= _num(tickets["late_value_survives_score"]).ge(params["late_value_min"])
    if params["pair_quinella_min"] is not None:
        mask &= _num(tickets["pair_quinella_score"]).ge(params["pair_quinella_min"])
    if params["partner_quinella_min"] is not None:
        mask &= _num(tickets["partner_quinella_score"]).ge(params["partner_quinella_min"])
    if params["market_overlay_min"] is not None:
        mask &= _num(tickets["market_overlay_score"]).ge(params["market_overlay_min"])
    if params["front_min"] is not None:
        mask &= _num(tickets["projected_front5_prob"]).ge(params["front_min"])
    if params["anchor_overpop_max"] is not None:
        mask &= _num(tickets["anchor_vertical_overpopular_risk_score"]).lt(params["anchor_overpop_max"])
    if params["partner_value_min"] is not None:
        mask &= _num(tickets["partner_vertical_underpopular_value_score"]).ge(params["partner_value_min"])
    if params["partner_overpop_max"] is not None:
        mask &= _num(tickets["partner_vertical_overpopular_risk_score"]).lt(params["partner_overpop_max"])
    if params["partner_odds_max"] is not None:
        mask &= _num(tickets["partner_odds"]).le(params["partner_odds_max"])
    return mask.fillna(False)


def _yearly_metrics(tickets: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for year, group in tickets.groupby("test_year"):
        row = _metrics(group, f"{label}_{int(year)}")
        row["test_year"] = int(year)
        rows.append(row)
    return pd.DataFrame(rows)


def _grid_search(tickets: pd.DataFrame, min_races: int, min_hit: float, min_keep_rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = []
    for race_difficulty_max, late_value_min, pair_q_min, partner_q_min, market_min, front_min, anchor_overpop, partner_value, partner_overpop, partner_odds_max in product(
        [None, 0.58, 0.66],
        [None, 0.45, 0.55],
        [None, 0.58, 0.62],
        [None, 0.54],
        [None, 0.55],
        [None, 0.45],
        [None, 0.66],
        [None, 0.50, 0.58],
        [None, 0.66],
        [None, 25.0, 40.0],
    ):
        params = {
            "race_difficulty_max": race_difficulty_max,
            "late_value_min": late_value_min,
            "pair_quinella_min": pair_q_min,
            "partner_quinella_min": partner_q_min,
            "market_overlay_min": market_min,
            "front_min": front_min,
            "anchor_overpop_max": anchor_overpop,
            "partner_value_min": partner_value,
            "partner_overpop_max": partner_overpop,
            "partner_odds_max": partner_odds_max,
        }
        mask = _apply_gate(tickets, params)
        subset = tickets[mask].copy()
        if subset.empty:
            continue
        metric = _metrics(subset, "grid")
        keep_rate = len(subset) / len(tickets) if len(tickets) else 0.0
        if metric["races"] < min_races or metric["race_hit_rate"] < min_hit or keep_rate < min_keep_rate:
            continue
        yearly = _yearly_metrics(subset, "grid")
        metric.update(params)
        metric["keep_rate"] = keep_rate
        metric["min_year_roi"] = float(yearly["roi"].min()) if not yearly.empty else 0.0
        metric["max_year_roi"] = float(yearly["roi"].max()) if not yearly.empty else 0.0
        metric["year_roi_spread"] = metric["max_year_roi"] - metric["min_year_roi"]
        metric["robust_score"] = (
            metric["roi"] * 0.45
            + metric["min_year_roi"] * 0.35
            + metric["race_hit_rate"] * 0.60
            + np.log1p(metric["races"]) * 0.03
            - metric["year_roi_spread"] * 0.12
        )
        grid.append(metric)
    out = pd.DataFrame(grid)
    if out.empty:
        return out, pd.DataFrame()
    out = out.sort_values(["robust_score", "roi", "races"], ascending=[False, False, False])
    best = out.head(1).copy()
    return out, best


def _classify_segments(segment_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_row = segment_summary[segment_summary["policy"].eq("all_selected")].iloc[0]
    base_roi = float(all_row["roi"])
    base_hit = float(all_row["race_hit_rate"])
    for factor, group in segment_summary[~segment_summary["policy"].eq("all_selected")].groupby("factor"):
        top = group.sort_values(["roi", "races"], ascending=[False, False]).iloc[0]
        worst = group.sort_values(["roi", "races"], ascending=[True, False]).iloc[0]
        if top["roi"] >= base_roi + 0.10 and top["races"] >= 300:
            verdict = "keep_as_gate"
        elif top["roi"] >= base_roi + 0.03 and worst["roi"] <= base_roi - 0.05:
            verdict = "use_limited"
        elif abs(top["roi"] - base_roi) < 0.03 and abs(worst["roi"] - base_roi) < 0.05:
            verdict = "weak_or_redundant"
        else:
            verdict = "monitor_only"
        rows.append(
            {
                "factor": factor,
                "verdict": verdict,
                "best_condition": top["condition"],
                "best_races": int(top["races"]),
                "best_roi": float(top["roi"]),
                "best_race_hit_rate": float(top["race_hit_rate"]),
                "worst_condition": worst["condition"],
                "worst_roi": float(worst["roi"]),
                "delta_best_roi": float(top["roi"] - base_roi),
                "delta_best_hit": float(top["race_hit_rate"] - base_hit),
            }
        )
    return pd.DataFrame(rows).sort_values(["verdict", "delta_best_roi"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ROI effectiveness of current betting-factor groups and optimize post gates.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/vertical_context_roi_v1/baseline_selected_tickets_with_vertical_scores.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/factor_roi_effectiveness_v1")
    parser.add_argument("--min-races", type=int, default=500)
    parser.add_argument("--min-hit", type=float, default=0.17)
    parser.add_argument("--min-keep-rate", type=float, default=0.20)
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    out_dir = ensure_dir(project_path(args.output_dir))

    specs = [
        {"name": "race_difficulty", "column": "race_difficulty_score", "direction": "low"},
        {"name": "danger_popular_anchor", "column": "anchor_danger", "direction": "low"},
        {"name": "danger_popular_partner", "column": "partner_danger", "direction": "low"},
        {"name": "late_value_odds", "column": "late_value_survives_score", "direction": "high"},
        {"name": "market_overlay", "column": "market_overlay_score", "direction": "high"},
        {"name": "quinella_pair", "column": "pair_quinella_score", "direction": "high"},
        {"name": "quinella_partner", "column": "partner_quinella_score", "direction": "high"},
        {"name": "front_position_probability", "column": "projected_front5_prob", "direction": "high"},
        {"name": "vertical_anchor_overpop", "column": "anchor_vertical_overpopular_risk_score", "direction": "low"},
        {"name": "vertical_partner_value", "column": "partner_vertical_underpopular_value_score", "direction": "high"},
        {"name": "vertical_partner_overpop", "column": "partner_vertical_overpopular_risk_score", "direction": "low"},
        {"name": "bias_volatility", "column": "race_bias_volatility", "direction": "low"},
        {"name": "pace_collapse_risk", "column": "race_pace_collapse", "direction": "low"},
        {"name": "partner_odds", "column": "partner_odds", "direction": "low"},
    ]

    segment_summary = _quantile_segments(tickets, specs)
    classification = _classify_segments(segment_summary)
    grid, best = _grid_search(tickets, args.min_races, args.min_hit, args.min_keep_rate)

    segment_summary.to_csv(out_dir / "factor_segment_roi_summary.csv", index=False, encoding="utf-8-sig")
    classification.to_csv(out_dir / "factor_role_classification.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(out_dir / "post_gate_optimization_grid.csv", index=False, encoding="utf-8-sig")

    best_yearly = pd.DataFrame()
    best_tickets = pd.DataFrame()
    if not best.empty:
        params = {k: best.iloc[0][k] for k in [
            "race_difficulty_max",
            "late_value_min",
            "pair_quinella_min",
            "partner_quinella_min",
            "market_overlay_min",
            "front_min",
            "anchor_overpop_max",
            "partner_value_min",
            "partner_overpop_max",
            "partner_odds_max",
        ]}
        params = {k: (None if pd.isna(v) else v) for k, v in params.items()}
        mask = _apply_gate(tickets, params)
        best_tickets = tickets[mask].copy()
        best_yearly = _yearly_metrics(best_tickets, "best_post_gate")
        best_tickets.to_csv(out_dir / "best_post_gate_tickets.csv", index=False, encoding="utf-8-sig")
        best_yearly.to_csv(out_dir / "best_post_gate_yearly.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "baseline": _metrics(tickets, "baseline_selected"),
        "best_post_gate": best.head(1).to_dict(orient="records")[0] if not best.empty else None,
        "best_post_gate_yearly": best_yearly.to_dict(orient="records"),
        "classification": classification.to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
