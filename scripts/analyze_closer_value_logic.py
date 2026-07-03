from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def rank01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() <= 1 or x.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=x.index, dtype=float)
    return x.rank(pct=True).fillna(0.5).clip(0.0, 1.0)


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    equity = profit.cumsum()
    return float((equity - equity.cummax()).min())


def metrics(df: pd.DataFrame, mask: pd.Series | np.ndarray, label: str) -> dict[str, Any]:
    part = df.loc[np.asarray(mask)].copy()
    if part.empty:
        return {
            "label": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
            "top5_removed_roi": 0.0,
            "top10_removed_roi": 0.0,
            "roi_2025": np.nan,
            "roi_2026": np.nan,
            "min_year_roi": np.nan,
        }
    stake = num(part, "stake_yen", 0.0).fillna(0.0)
    ret = num(part, "return_yen", 0.0).fillna(0.0)
    profit = ret - stake
    part["_stake"] = stake
    part["_return"] = ret
    part["_profit"] = profit

    def removed_roi(n: int) -> float:
        if len(part) <= n:
            return 0.0
        kept = part.sort_values("_profit", ascending=False).iloc[n:]
        kept_stake = float(kept["_stake"].sum())
        return float(kept["_return"].sum() / kept_stake) if kept_stake else 0.0

    yearly: dict[str, float] = {}
    for year, g in part.groupby("year"):
        y_stake = float(g["_stake"].sum())
        if y_stake:
            yearly[f"roi_{int(year)}"] = float(g["_return"].sum() / y_stake)
    year_vals = [v for v in yearly.values() if np.isfinite(v)]
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    return {
        "label": label,
        "tickets": int(len(part)),
        "races": int(part["race_id"].nunique()),
        "stake_yen": stake_sum,
        "return_yen": ret_sum,
        "profit_yen": ret_sum - stake_sum,
        "roi": float(ret_sum / stake_sum) if stake_sum else 0.0,
        "hit_rate": float(ret.gt(0).mean()),
        "max_drawdown_yen": max_drawdown(part.sort_values(["year", "race_id"])["_profit"]),
        "top5_removed_roi": removed_roi(5),
        "top10_removed_roi": removed_roi(10),
        "roi_2025": yearly.get("roi_2025", np.nan),
        "roi_2026": yearly.get("roi_2026", np.nan),
        "min_year_roi": min(year_vals) if year_vals else np.nan,
    }


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "position_closer_value_score",
        "position_front_value_score",
        "collapse_fit",
        "closer_pair_max",
        "style_diversity",
        "front_front_slow_fit",
        "front_front_clash",
        "front_pair_max",
        "projected_front5_prob",
        "market_overlay_score",
        "pair_quinella_score",
        "danger_sum",
        "skip_risk_score",
    ]:
        out[f"{col}_pct"] = rank01(num(out, col, 0.0))
    out["front_minus_closer_value"] = num(out, "position_front_value_score", 0.0) - num(
        out, "position_closer_value_score", 0.0
    )
    out["front_minus_closer_value_pct"] = rank01(out["front_minus_closer_value"])
    return out


def rule_grid(df: pd.DataFrame, *, min_races: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    collapse_lows = [0.25, 0.33, 0.40]
    collapse_highs = [0.67, 0.75, 0.85]
    closer_lows = [0.20, 0.33]
    closer_highs = [0.60, 0.75, 1.00]
    slow_highs = [0.50, 0.60, 0.75]
    clash_lows = [0.25, 0.33, 0.40]
    front_value_highs = [0.50, 0.60, 0.75]
    overlay_lows = [0.00, 0.33]
    danger_highs = [0.60, 1.00]

    base = pd.Series(True, index=df.index)
    for collapse_lo in collapse_lows:
        for collapse_hi in collapse_highs:
            if collapse_hi <= collapse_lo:
                continue
            collapse_mask = df["collapse_fit_pct"].between(collapse_lo, collapse_hi, inclusive="both")
            for closer_lo in closer_lows:
                for closer_hi in closer_highs:
                    if closer_hi <= closer_lo:
                        continue
                    closer_mask = df["closer_pair_max_pct"].between(closer_lo, closer_hi, inclusive="both")
                    for slow_hi in slow_highs:
                        slow_mask = df["front_front_slow_fit_pct"].le(slow_hi)
                        for clash_lo in clash_lows:
                            clash_mask = df["front_front_clash_pct"].ge(clash_lo)
                            for front_hi in front_value_highs:
                                front_mask = df["position_front_value_score_pct"].le(front_hi)
                                for overlay_lo in overlay_lows:
                                    overlay_mask = df["market_overlay_score_pct"].ge(overlay_lo)
                                    for danger_hi in danger_highs:
                                        mask = (
                                            base
                                            & collapse_mask
                                            & closer_mask
                                            & slow_mask
                                            & clash_mask
                                            & front_mask
                                            & overlay_mask
                                            & df["danger_sum_pct"].le(danger_hi)
                                        )
                                        if int(df.loc[mask, "race_id"].nunique()) < min_races:
                                            continue
                                        label = (
                                            f"closer_logic_collapse{collapse_lo:.2f}-{collapse_hi:.2f}"
                                            f"_closer{closer_lo:.2f}-{closer_hi:.2f}"
                                            f"_slow_le{slow_hi:.2f}_clash_ge{clash_lo:.2f}"
                                            f"_front_le{front_hi:.2f}_overlay_ge{overlay_lo:.2f}"
                                            f"_danger_le{danger_hi:.2f}"
                                        )
                                        row = metrics(df, mask, label)
                                        row.update(
                                            {
                                                "collapse_lo": collapse_lo,
                                                "collapse_hi": collapse_hi,
                                                "closer_lo": closer_lo,
                                                "closer_hi": closer_hi,
                                                "slow_hi": slow_hi,
                                                "clash_lo": clash_lo,
                                                "front_value_hi": front_hi,
                                                "overlay_lo": overlay_lo,
                                                "danger_hi": danger_hi,
                                            }
                                        )
                                        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    stability = out["top10_removed_roi"].fillna(0.0).clip(0.0, 3.0)
    min_year = out["min_year_roi"].fillna(0.0).clip(0.0, 3.0)
    coverage = np.log1p(out["races"]) / np.log1p(max(float(out["races"].max()), 1.0))
    out["robust_score"] = (
        0.32 * out["roi"].clip(0.0, 4.0)
        + 0.32 * stability
        + 0.22 * min_year
        + 0.08 * out["hit_rate"].fillna(0.0) * 10.0
        + 0.06 * coverage
    )
    return out.sort_values(
        ["robust_score", "top10_removed_roi", "roi", "races"],
        ascending=[False, False, False, False],
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep-dive closer-position value logic.")
    parser.add_argument(
        "--scored-csv",
        default="outputs/analysis/position_value_gate_v2_with_closer/position_value_scored_tickets.csv",
    )
    parser.add_argument("--analysis-source", default="recommended_joint_guard")
    parser.add_argument("--output-dir", default="outputs/analysis/closer_value_logic_v1")
    parser.add_argument("--min-races", type=int, default=80)
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project_path(args.scored_csv), encoding="utf-8-sig", low_memory=False)
    if "analysis_source" in raw.columns:
        raw = raw[raw["analysis_source"].astype(str).eq(args.analysis_source)].copy()
    raw["race_id"] = raw["race_id"].astype(str)
    raw["year"] = pd.to_numeric(raw.get("year", raw["race_id"].str[:4]), errors="coerce").astype("Int64")
    scored = add_rank_features(raw)

    baseline = metrics(scored, pd.Series(True, index=scored.index), "baseline")
    grid = rule_grid(scored, min_races=args.min_races)
    grid.to_csv(out_dir / "closer_logic_grid.csv", index=False, encoding="utf-8-sig")

    top = grid.head(30).copy() if not grid.empty else pd.DataFrame()
    top.to_csv(out_dir / "top_closer_logic_rules.csv", index=False, encoding="utf-8-sig")

    payload = {
        "inputs": {
            "scored_csv": str(project_path(args.scored_csv)),
            "analysis_source": args.analysis_source,
            "min_races": args.min_races,
        },
        "baseline": baseline,
        "rules_tested": int(len(grid)),
        "best_rules": top.head(10).to_dict(orient="records") if not top.empty else [],
        "outputs": {
            "grid_csv": str(out_dir / "closer_logic_grid.csv"),
            "top_rules_csv": str(out_dir / "top_closer_logic_rules.csv"),
        },
        "note": "Use these closer rules for shadow/watch ranking first; do not promote to final BUY without OOS accumulation.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
