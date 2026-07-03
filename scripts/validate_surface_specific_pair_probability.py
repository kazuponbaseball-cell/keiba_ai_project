from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.strict_pair_probability_roi_protocol import (  # noqa: E402
    add_calibrated_probs,
    build_raw_probability_features,
    evaluate,
    metrics,
    policy_grid,
    policy_score,
    threshold_from_coverage,
    train_val_holdout,
    walkforward,
)


PAIR_UNIVERSE = Path("outputs/analysis/horse_lap_decomp_pair_probability_v1/pair_universe_with_lap_decomp_features.csv")
RACE_CONTEXT = Path("outputs/analysis/high_pressure_front_survival_context_v1/race_high_pressure_front_survival_context.csv")
OUT = Path("outputs/analysis/surface_specific_pair_probability_v1")

SURFACES = ["芝", "ダ"]


def read_pair_universe() -> pd.DataFrame:
    if not PAIR_UNIVERSE.exists():
        raise FileNotFoundError(f"missing pair universe: {PAIR_UNIVERSE}")
    df = pd.read_csv(PAIR_UNIVERSE, dtype={"race_id": str}, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    if "year" not in df.columns:
        df["year"] = df["race_id"].str[:4].astype(int)
    return df


def attach_surface(df: pd.DataFrame) -> pd.DataFrame:
    if not RACE_CONTEXT.exists():
        raise FileNotFoundError(f"missing race context: {RACE_CONTEXT}")
    ctx = pd.read_csv(
        RACE_CONTEXT,
        dtype={"race_id": str},
        usecols=lambda c: c in {"race_id", "surface", "venue_code", "distance_bin", "class_tier", "going"},
        low_memory=False,
    )
    ctx["race_id"] = ctx["race_id"].astype(str)
    ctx = ctx.drop_duplicates("race_id", keep="last").rename(
        columns={
            "surface": "surface_clean",
            "distance_bin": "surface_distance_bin",
            "class_tier": "surface_class_tier",
            "going": "surface_going_raw",
        }
    )
    out = df.merge(ctx, on="race_id", how="left")
    if "surface_clean" not in out.columns:
        out["surface_clean"] = ""
    out["surface_clean"] = out["surface_clean"].fillna("").astype(str)
    # Keep only races where the clean TARGET/JRA context says turf or dirt.
    out = out[out["surface_clean"].isin(SURFACES)].copy()
    return out


def _grid_search_on_train(train_scored: pd.DataFrame, test_year: int, surface: str, mode: str) -> tuple[pd.DataFrame, dict]:
    rows: list[dict[str, Any]] = []
    grids = policy_grid()
    for i, params in enumerate(grids):
        threshold = threshold_from_coverage(train_scored, params["coverage"])
        m, _ = evaluate(train_scored, params, threshold, f"train_{mode}_{surface}_{test_year}_{i}")
        row = m | {k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}}
        row["grid_id"] = i
        row["surface"] = surface
        row["mode"] = mode
        row["test_year"] = test_year
        row["score_threshold"] = threshold
        row["selection_score"] = policy_score(m)
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values(["selection_score", "roi"], ascending=[False, False])
    best = grid.iloc[0].to_dict()
    return grid, best


def _evaluate_surface(
    train_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    test_year: int,
    surface: str,
    mode: str,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    grid, best = _grid_search_on_train(train_scored, test_year, surface, mode)
    params = policy_grid()[int(best["grid_id"])]
    m, tickets = evaluate(test_scored, params, float(best["score_threshold"]), f"{mode}_{surface}_{test_year}")
    m.update({k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}})
    m["surface"] = surface
    m["mode"] = mode
    m["test_year"] = test_year
    m["train_roi"] = float(best["roi"])
    m["train_races"] = int(best["races"])
    m["train_profit_yen"] = float(best["profit_yen"])
    m["score_threshold"] = float(best["score_threshold"])
    if not tickets.empty:
        tickets = tickets.copy()
        tickets["surface"] = surface
        tickets["mode"] = mode
        tickets["test_year"] = test_year
        tickets["selected_grid_id"] = int(best["grid_id"])
    return grid, m, tickets


def walkforward_surface_threshold_only(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    raw = build_raw_probability_features(df)
    for test_year in [2025, 2026]:
        train_raw = raw[raw["year"] < test_year].copy()
        test_raw = raw[raw["year"] == test_year].copy()
        train_scored_all, _ = add_calibrated_probs(train_raw, train_raw)
        test_scored_all, _ = add_calibrated_probs(train_raw, test_raw)
        for surface in SURFACES:
            train_scored = train_scored_all[train_scored_all["surface_clean"].eq(surface)].copy()
            test_scored = test_scored_all[test_scored_all["surface_clean"].eq(surface)].copy()
            if len(train_scored) < 200 or test_scored.empty:
                continue
            grid, m, tickets = _evaluate_surface(train_scored, test_scored, test_year, surface, "surface_threshold_only")
            train_rows.append(grid.head(100))
            summary_rows.append(m)
            if not tickets.empty:
                ticket_frames.append(tickets)
    return (
        pd.concat(train_rows, ignore_index=True, sort=False) if train_rows else pd.DataFrame(),
        pd.DataFrame(summary_rows),
        pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame(),
    )


def walkforward_surface_full(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    raw = build_raw_probability_features(df)
    for test_year in [2025, 2026]:
        for surface in SURFACES:
            train_raw = raw[(raw["year"] < test_year) & raw["surface_clean"].eq(surface)].copy()
            test_raw = raw[(raw["year"] == test_year) & raw["surface_clean"].eq(surface)].copy()
            if len(train_raw) < 200 or test_raw.empty:
                continue
            train_scored, _ = add_calibrated_probs(train_raw, train_raw)
            test_scored, _ = add_calibrated_probs(train_raw, test_raw)
            grid, m, tickets = _evaluate_surface(train_scored, test_scored, test_year, surface, "surface_full")
            train_rows.append(grid.head(100))
            summary_rows.append(m)
            if not tickets.empty:
                ticket_frames.append(tickets)
    return (
        pd.concat(train_rows, ignore_index=True, sort=False) if train_rows else pd.DataFrame(),
        pd.DataFrame(summary_rows),
        pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame(),
    )


def total_metrics(tickets: pd.DataFrame, label: str) -> dict:
    m = metrics(tickets, label)
    m["mode"] = label
    if not tickets.empty:
        years = []
        for year, part in tickets.groupby("year", sort=True):
            y = metrics(part, f"{label}_{year}")
            years.append({"year": int(year), "roi": y["roi"], "races": y["races"], "profit_yen": y["profit_yen"], "mode": label})
        m["min_year_roi"] = min((x["roi"] for x in years), default=0.0)
        m["year_metrics"] = years
    else:
        m["min_year_roi"] = 0.0
        m["year_metrics"] = []
    return m


def surface_metrics(tickets: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    if tickets.empty:
        return pd.DataFrame()
    for keys in [["surface"], ["surface", "ticket_type"], ["surface", "venue"], ["surface", "going"]]:
        for values, group in tickets.groupby(keys, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            m = metrics(group, label)
            rows.append(
                {
                    "mode": label,
                    "segment": " x ".join(keys),
                    "value": " x ".join(str(v) for v in values),
                    "tickets": m["tickets"],
                    "races": m["races"],
                    "stake_yen": m["stake_yen"],
                    "return_yen": m["return_yen"],
                    "profit_yen": m["profit_yen"],
                    "roi": m["roi"],
                    "race_hit_rate": m["race_hit_rate"],
                }
            )
    return pd.DataFrame(rows)


def run_common_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = build_raw_probability_features(df)
    train_grid, wf_summary, wf_tickets = walkforward(raw)
    holdout = train_val_holdout(raw)
    if not wf_tickets.empty:
        if "surface_clean" not in wf_tickets.columns:
            surf = df[["race_id", "surface_clean"]].drop_duplicates("race_id")
            wf_tickets = wf_tickets.merge(surf, on="race_id", how="left")
        wf_tickets["surface"] = wf_tickets["surface_clean"]
        wf_tickets["mode"] = "common_baseline"
    return train_grid, wf_summary, wf_tickets, holdout


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = attach_surface(read_pair_universe())
    source.to_csv(OUT / "pair_universe_surface_clean.csv", index=False, encoding="utf-8-sig")

    common_grid, common_wf_summary, common_tickets, common_holdout = run_common_baseline(source)
    th_grid, th_wf_summary, th_tickets = walkforward_surface_threshold_only(source)
    full_grid, full_wf_summary, full_tickets = walkforward_surface_full(source)

    common_grid.to_csv(OUT / "common_baseline_walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
    common_wf_summary.to_csv(OUT / "common_baseline_walkforward_summary.csv", index=False, encoding="utf-8-sig")
    common_tickets.to_csv(OUT / "common_baseline_walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
    common_holdout.to_csv(OUT / "common_baseline_train2024_val2025_hold2026_grid.csv", index=False, encoding="utf-8-sig")

    th_grid.to_csv(OUT / "surface_threshold_only_walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
    th_wf_summary.to_csv(OUT / "surface_threshold_only_walkforward_summary.csv", index=False, encoding="utf-8-sig")
    th_tickets.to_csv(OUT / "surface_threshold_only_walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")

    full_grid.to_csv(OUT / "surface_full_walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
    full_wf_summary.to_csv(OUT / "surface_full_walkforward_summary.csv", index=False, encoding="utf-8-sig")
    full_tickets.to_csv(OUT / "surface_full_walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")

    totals = [
        total_metrics(common_tickets, "common_baseline"),
        total_metrics(th_tickets, "surface_threshold_only"),
        total_metrics(full_tickets, "surface_full"),
    ]
    comparison = pd.DataFrame(totals).drop(columns=["year_metrics"], errors="ignore")
    comparison = comparison.sort_values(["roi", "profit_yen"], ascending=False)
    comparison.to_csv(OUT / "surface_model_comparison.csv", index=False, encoding="utf-8-sig")

    year_rows = []
    for row in totals:
        year_rows.extend(row.get("year_metrics", []))
    pd.DataFrame(year_rows).to_csv(OUT / "surface_model_yearly_summary.csv", index=False, encoding="utf-8-sig")

    seg = pd.concat(
        [
            surface_metrics(common_tickets, "common_baseline"),
            surface_metrics(th_tickets, "surface_threshold_only"),
            surface_metrics(full_tickets, "surface_full"),
        ],
        ignore_index=True,
        sort=False,
    )
    seg.to_csv(OUT / "surface_model_segment_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(OUT),
        "pair_universe_rows": int(len(source)),
        "race_count": int(source["race_id"].nunique()),
        "modes": ["common_baseline", "surface_threshold_only", "surface_full"],
        "comparison": comparison.to_dict(orient="records"),
        "note": (
            "surface_threshold_only uses common calibration but separately optimizes policy thresholds by turf/dirt. "
            "surface_full calibrates and optimizes separately by turf/dirt. This tests whether surface-specific race-quality models improve ROI."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Surface Specific Pair Probability v1",
                "",
                "Purpose: compare the current common strict pair-probability model with turf/dirt-specific thresholding",
                "and turf/dirt-specific calibration + thresholding.",
                "",
                "Key outputs:",
                "- surface_model_comparison.csv",
                "- surface_model_yearly_summary.csv",
                "- surface_model_segment_metrics.csv",
                "- common_baseline_walkforward_selected_tickets.csv",
                "- surface_threshold_only_walkforward_selected_tickets.csv",
                "- surface_full_walkforward_selected_tickets.csv",
            ]
        ),
        encoding="utf-8",
    )

    display_cols = [
        "mode",
        "races",
        "tickets",
        "stake_yen",
        "return_yen",
        "profit_yen",
        "roi",
        "race_hit_rate",
        "max_drawdown_yen",
        "min_year_roi",
    ]
    print(comparison[display_cols].to_string(index=False))
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
