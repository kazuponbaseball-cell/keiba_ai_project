from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from build_selective_expansion_policy import metrics, top_removed
from diagnose_expansion_roi_dilution import normalize, num


def score_candidates(extra: pd.DataFrame) -> pd.DataFrame:
    out = extra.copy()
    out["partner_odds_num"] = num(out.get("partner_odds"), out.index, np.nan)
    out["anchor_pop_num"] = num(out.get("anchor_pop"), out.index, np.nan)
    out["partner_pop_num"] = num(out.get("partner_pop"), out.index, np.nan)
    out["overlay_num"] = num(out.get("market_overlay_score"), out.index, 0.0).fillna(0.0)
    out["front5_num"] = num(out.get("projected_front5_prob"), out.index, 0.0).fillna(0.0)
    out["pair_score_num"] = num(out.get("pair_score"), out.index, 0.0).fillna(0.0)
    out["pair_q_num"] = num(out.get("pair_quinella_score"), out.index, 0.0).fillna(0.0)
    out["danger_sum"] = (
        num(out.get("anchor_danger"), out.index, 0.0).fillna(0.0)
        + num(out.get("partner_danger"), out.index, 0.0).fillna(0.0)
    )
    return out


def build_policy(extra: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = extra.copy()
    umaren = out["ticket_type"].eq("umaren")
    base = (
        umaren
        & out["partner_odds_num"].between(params["partner_odds_min"], params["partner_odds_max"])
        & out["overlay_num"].ge(params["overlay_min"])
        & out["pair_score_num"].ge(params["pair_score_min"])
        & out["pair_q_num"].ge(params["pair_q_min"])
        & out["front5_num"].ge(params["front5_min"])
        & out["danger_sum"].le(params["danger_max"])
    )
    shape = (
        out["anchor_pop_num"].between(params["anchor_pop_min"], params["anchor_pop_max"])
        | (out["venue_eval"].eq("Niigata") & out["anchor_pop_num"].le(params["niigata_anchor_pop_max"]))
    )
    selected = out[base & shape].copy()
    if selected.empty:
        return selected

    raw = params["base_stake"] + params["quality_stake"] * (
        0.40 * ((selected["pair_score_num"] - params["pair_score_min"]) / max(1.0 - params["pair_score_min"], 0.001)).clip(0.0, 1.0)
        + 0.35 * ((selected["overlay_num"] - params["overlay_min"]) / max(1.0 - params["overlay_min"], 0.001)).clip(0.0, 1.0)
        + 0.25 * ((selected["front5_num"] - params["front5_min"]) / max(1.0 - params["front5_min"], 0.001)).clip(0.0, 1.0)
    )
    selected["eval_stake_yen"] = (np.floor(raw / 100.0) * 100.0).clip(lower=100.0, upper=params["max_stake"])
    selected["eval_return_yen"] = np.where(
        selected["hit_eval"],
        num(selected.get("umaren_pay"), selected.index, 0.0).fillna(0.0) * selected["eval_stake_yen"] / 100.0,
        0.0,
    )
    selected["eval_profit_yen"] = selected["eval_return_yen"] - selected["eval_stake_yen"]
    return selected


def grid() -> list[dict]:
    rows = []
    for (
        partner_odds_max,
        overlay_min,
        pair_score_min,
        pair_q_min,
        front5_min,
        danger_max,
        anchor_pop_max,
        niigata_anchor_pop_max,
    ) in product(
        [40.0, 60.0],
        [0.70, 0.78],
        [0.72, 0.80],
        [0.50, 0.58],
        [0.50, 0.60],
        [0.55, 0.70],
        [3.0, 5.0],
        [5.0],
    ):
        rows.append(
            {
                "partner_odds_min": 5.0,
                "partner_odds_max": partner_odds_max,
                "overlay_min": overlay_min,
                "pair_score_min": pair_score_min,
                "pair_q_min": pair_q_min,
                "front5_min": front5_min,
                "danger_max": danger_max,
                "anchor_pop_min": 2.0,
                "anchor_pop_max": anchor_pop_max,
                "niigata_anchor_pop_max": niigata_anchor_pop_max,
                "base_stake": 200.0,
                "quality_stake": 300.0,
                "max_stake": 500.0,
            }
        )
    return rows


def robust_score(row: dict) -> float:
    if row["tickets"] < 30 or row["races"] < 25:
        return -1e9
    if row["minus_top5_roi"] < 0.90:
        return -1e9
    if row["year_count"] < 2:
        return -1e9
    return float(
        row["roi"] * 0.50
        + row["minus_top5_roi"] * 1.10
        + row["minus_top10_roi"] * 0.70
        + row["race_hit_rate"] * 2.0
        + np.log1p(row["races"]) * 0.08
        - abs(row["max_drawdown_yen"]) / 30000.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine selective expansion policy using top-hit removal robustness.")
    parser.add_argument("--expanded-csv", default="outputs/analysis/extended_period_validation_v1/fixed_proxy_selected_tickets_2024_2026.csv")
    parser.add_argument("--standard-csv", default="outputs/analysis/final_operational_quality_v1/standard_explained_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/selective_expansion_robust_v1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expanded = normalize(pd.read_csv(args.expanded_csv, dtype={"race_id": str}, low_memory=False), "expanded")
    standard = normalize(pd.read_csv(args.standard_csv, dtype={"race_id": str}, low_memory=False), "standard")
    extra = expanded[~expanded["ticket_key_eval"].isin(set(standard["ticket_key_eval"]))].copy()
    extra = score_candidates(extra)

    rows = []
    selected_by_grid: dict[int, pd.DataFrame] = {}
    for i, params in enumerate(grid()):
        selected = build_policy(extra, params)
        m = metrics(selected, f"grid_{i}")
        mt5 = top_removed(selected, 5)
        mt10 = top_removed(selected, 10)
        year_count = int(selected["year"].nunique()) if not selected.empty else 0
        row = params | {
            "grid_id": i,
            **m,
            "minus_top5_roi": mt5["roi"],
            "minus_top10_roi": mt10["roi"],
            "minus_top5_profit_yen": mt5["profit_yen"],
            "minus_top10_profit_yen": mt10["profit_yen"],
            "year_count": year_count,
        }
        row["robust_score"] = robust_score(row)
        rows.append(row)
        selected_by_grid[i] = selected

    results = pd.DataFrame(rows).sort_values(["robust_score", "roi"], ascending=[False, False])
    results.to_csv(out_dir / "robust_expansion_grid.csv", index=False, encoding="utf-8-sig")
    best = results.iloc[0]
    best_selected = selected_by_grid[int(best["grid_id"])].copy()
    best_selected.to_csv(out_dir / "robust_expansion_addon_tickets.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat([standard, best_selected], ignore_index=True, sort=False)
    combined.to_csv(out_dir / "standard_plus_robust_expansion_tickets.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            metrics(standard, "standard"),
            metrics(best_selected, "robust_expansion_addon"),
            metrics(combined, "standard_plus_robust_expansion"),
            top_removed(standard, 10) | {"label": "standard_minus_top10"},
            top_removed(best_selected, 10) | {"label": "addon_minus_top10"},
            top_removed(combined, 10) | {"label": "combined_minus_top10"},
        ]
    )
    summary.to_csv(out_dir / "robust_expansion_summary.csv", index=False, encoding="utf-8-sig")
    by_year = pd.DataFrame([metrics(g, f"year_{year}") | {"year": year} for year, g in best_selected.groupby("year")])
    by_year.to_csv(out_dir / "robust_addon_by_year.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "best_grid_id": int(best["grid_id"]),
        "best_params": {k: best[k].item() if hasattr(best[k], "item") else best[k] for k in grid()[0].keys()},
        "summary": summary.to_dict(orient="records"),
        "best_top_grid": results.head(20).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
