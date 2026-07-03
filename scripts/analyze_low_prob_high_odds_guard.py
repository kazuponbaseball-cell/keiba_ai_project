from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_calibration_and_fractional_kelly import prepare_tickets  # noqa: E402


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def max_drawdown(race: pd.DataFrame) -> float:
    if race.empty:
        return 0.0
    equity = race["profit_yen"].cumsum()
    peak = equity.cummax().clip(lower=0.0)
    return float((equity - peak).min())


def summarize(rows: pd.DataFrame, label: str) -> dict[str, Any]:
    if rows.empty:
        return {
            "label": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
            "top5_removed_roi": 0.0,
            "top10_removed_roi": 0.0,
        }

    race = (
        rows.groupby("race_id", sort=False)
        .agg(
            date=("_date", "min"),
            stake_yen=("_current_stake", "sum"),
            return_yen=("_actual_return_yen_current", "sum"),
            hit=("_actual_return_yen_current", lambda s: bool((s > 0).any())),
        )
        .reset_index()
        .sort_values(["date", "race_id"])
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    stake = float(race["stake_yen"].sum())
    ret = float(race["return_yen"].sum())
    top_returns = race["return_yen"].sort_values(ascending=False)
    ex_top5 = race.copy()
    ex_top10 = race.copy()
    ex_top5.loc[top_returns.index[:5], "return_yen"] = 0.0
    ex_top10.loc[top_returns.index[:10], "return_yen"] = 0.0
    return {
        "label": label,
        "tickets": int(len(rows)),
        "races": int(rows["race_id"].nunique()),
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(rows["_hit"].mean()),
        "race_hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "max_drawdown_yen": max_drawdown(race),
        "top5_removed_roi": float(ex_top5["return_yen"].sum() / stake) if stake else 0.0,
        "top10_removed_roi": float(ex_top10["return_yen"].sum() / stake) if stake else 0.0,
    }


def evaluate_rules(df: pd.DataFrame, min_races: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = summarize(df, "baseline")
    rows.append({**baseline, "guard_type": "none", "prob_max": None, "odds_min": None, "removed_tickets": 0, "removed_hits": 0})

    prob_grid = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
    odds_grid = [40, 60, 80, 100, 120, 150, 200]
    for prob_max, odds_min in product(prob_grid, odds_grid):
        remove = df["_model_prob"].le(prob_max) & df["_decimal_odds"].ge(odds_min)
        kept = df[~remove].copy()
        if kept["race_id"].nunique() < min_races:
            continue
        metric = summarize(kept, f"drop_prob_le_{prob_max:.2f}_odds_ge_{odds_min:g}")
        metric.update(
            {
                "guard_type": "drop_low_prob_high_odds",
                "prob_max": prob_max,
                "odds_min": odds_min,
                "removed_tickets": int(remove.sum()),
                "removed_hits": int(df.loc[remove, "_hit"].sum()),
                "removed_stake_yen": float(df.loc[remove, "_current_stake"].sum()),
                "removed_return_yen": float(df.loc[remove, "_actual_return_yen_current"].sum()),
                "delta_roi": metric["roi"] - baseline["roi"],
                "delta_top10_removed_roi": metric["top10_removed_roi"] - baseline["top10_removed_roi"],
                "delta_max_drawdown_yen": metric["max_drawdown_yen"] - baseline["max_drawdown_yen"],
            }
        )
        metric["robust_score"] = (
            metric["top10_removed_roi"] * 0.50
            + metric["top5_removed_roi"] * 0.20
            + metric["roi"] * 0.18
            + metric["race_hit_rate"] * 0.50
            - max(0, baseline["races"] - metric["races"]) / max(baseline["races"], 1) * 0.25
        )
        rows.append(metric)

    # Also test simple hard ceilings/floors to prove whether the interaction guard
    # is better than bluntly cutting high odds or low probability.
    for prob_min in [0.07, 0.08, 0.09, 0.10, 0.11, 0.12]:
        kept = df[df["_model_prob"].ge(prob_min)].copy()
        if kept["race_id"].nunique() < min_races:
            continue
        metric = summarize(kept, f"keep_prob_ge_{prob_min:.2f}")
        metric.update(
            {
                "guard_type": "keep_prob_floor",
                "prob_max": prob_min,
                "odds_min": None,
                "removed_tickets": int(len(df) - len(kept)),
                "removed_hits": int(df.loc[~df.index.isin(kept.index), "_hit"].sum()),
                "removed_stake_yen": float(df.loc[~df.index.isin(kept.index), "_current_stake"].sum()),
                "removed_return_yen": float(df.loc[~df.index.isin(kept.index), "_actual_return_yen_current"].sum()),
                "delta_roi": metric["roi"] - baseline["roi"],
                "delta_top10_removed_roi": metric["top10_removed_roi"] - baseline["top10_removed_roi"],
                "delta_max_drawdown_yen": metric["max_drawdown_yen"] - baseline["max_drawdown_yen"],
            }
        )
        metric["robust_score"] = metric["top10_removed_roi"] * 0.55 + metric["roi"] * 0.25 + metric["race_hit_rate"] * 0.50
        rows.append(metric)

    for odds_max in [40, 60, 80, 100, 120, 150, 200]:
        kept = df[df["_decimal_odds"].le(odds_max)].copy()
        if kept["race_id"].nunique() < min_races:
            continue
        metric = summarize(kept, f"keep_odds_le_{odds_max:g}")
        metric.update(
            {
                "guard_type": "keep_odds_ceiling",
                "prob_max": None,
                "odds_min": odds_max,
                "removed_tickets": int(len(df) - len(kept)),
                "removed_hits": int(df.loc[~df.index.isin(kept.index), "_hit"].sum()),
                "removed_stake_yen": float(df.loc[~df.index.isin(kept.index), "_current_stake"].sum()),
                "removed_return_yen": float(df.loc[~df.index.isin(kept.index), "_actual_return_yen_current"].sum()),
                "delta_roi": metric["roi"] - baseline["roi"],
                "delta_top10_removed_roi": metric["top10_removed_roi"] - baseline["top10_removed_roi"],
                "delta_max_drawdown_yen": metric["max_drawdown_yen"] - baseline["max_drawdown_yen"],
            }
        )
        metric["robust_score"] = metric["top10_removed_roi"] * 0.55 + metric["roi"] * 0.25 + metric["race_hit_rate"] * 0.50
        rows.append(metric)

    out = pd.DataFrame(rows)
    return out.sort_values(["robust_score", "top10_removed_roi", "races"], ascending=[False, False, False])


def odds_ceiling_sensitivity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    for cap in [80.0, 100.0, 120.0, 150.0, None]:
        label = "odds_cap_none" if cap is None else f"odds_cap_{cap:g}"
        subset = df.copy() if cap is None else df[df["_decimal_odds"].le(cap)].copy()
        row = summarize(subset, label)
        row["odds_cap"] = cap
        row["removed_tickets"] = int(len(df) - len(subset))
        row["removed_hits"] = int(df.loc[~df.index.isin(subset.index), "_hit"].sum())
        rows.append(row)
        for year, group in subset.groupby("_year"):
            y = summarize(group, f"{label}_{int(year)}")
            y["odds_cap"] = cap
            y["year"] = int(year)
            yearly_rows.append(y)
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate low-probability high-odds guard rules for strongest runtime tickets.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates_default/recommended_runtime_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/low_prob_high_odds_guard_v1")
    parser.add_argument("--min-races", type=int, default=220)
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_tickets(project_path(args.tickets_csv))
    grid = evaluate_rules(df, args.min_races)
    best = grid[grid["guard_type"].ne("none")].head(1).copy()
    cap_sensitivity, cap_yearly = odds_ceiling_sensitivity(df)

    df.to_csv(out_dir / "prepared_tickets.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(out_dir / "guard_grid.csv", index=False, encoding="utf-8-sig")
    cap_sensitivity.to_csv(out_dir / "odds_ceiling_sensitivity.csv", index=False, encoding="utf-8-sig")
    cap_yearly.to_csv(out_dir / "odds_ceiling_sensitivity_yearly.csv", index=False, encoding="utf-8-sig")
    if not best.empty:
        row = best.iloc[0]
        if row["guard_type"] == "drop_low_prob_high_odds":
            keep = ~(df["_model_prob"].le(float(row["prob_max"])) & df["_decimal_odds"].ge(float(row["odds_min"])))
        elif row["guard_type"] == "keep_prob_floor":
            keep = df["_model_prob"].ge(float(row["prob_max"]))
        elif row["guard_type"] == "keep_odds_ceiling":
            keep = df["_decimal_odds"].le(float(row["odds_min"]))
        else:
            keep = pd.Series(True, index=df.index)
        df[keep].to_csv(out_dir / "best_guard_tickets.csv", index=False, encoding="utf-8-sig")

    payload = {
        "tickets_csv": str(project_path(args.tickets_csv)),
        "output_dir": str(out_dir),
        "baseline": grid[grid["guard_type"].eq("none")].head(1).to_dict(orient="records")[0],
        "best_guard": best.to_dict(orient="records")[0] if not best.empty else None,
        "odds_ceiling_sensitivity": cap_sensitivity.to_dict(orient="records"),
        "top_rules": grid.head(12).to_dict(orient="records"),
        "note": "This is a post-selection safety guard. Adopt only if it improves robustness without cutting too many races.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, ensure_ascii=False, indent=2)
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
