from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


def rank01_by_train(train: pd.Series, target: pd.Series) -> pd.Series:
    train_x = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    target_x = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if train_x.empty or train_x.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=target.index, dtype=float)
    sorted_train = np.sort(train_x.to_numpy(dtype=float))
    vals = target_x.to_numpy(dtype=float)
    ranks = np.searchsorted(sorted_train, vals, side="right") / len(sorted_train)
    out = pd.Series(ranks, index=target.index, dtype=float)
    return out.fillna(0.5).clip(0.0, 1.0)


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    equity = profit.cumsum()
    return float((equity - equity.cummax()).min())


def removed_roi(part: pd.DataFrame, n: int) -> float:
    if part.empty or len(part) <= n:
        return 0.0
    kept = part.sort_values("_profit", ascending=False).iloc[n:]
    stake = float(kept["_stake"].sum())
    return float(kept["_return"].sum() / stake) if stake else 0.0


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
            "roi_pct": 0.0,
            "hit_rate_pct": 0.0,
            "max_drawdown_yen": 0.0,
            "top1_removed_roi_pct": 0.0,
            "top3_removed_roi_pct": 0.0,
            "top5_removed_roi_pct": 0.0,
            "top10_removed_roi_pct": 0.0,
        }
    part["_stake"] = num(part, "stake_yen", 0.0).fillna(0.0)
    part["_return"] = num(part, "return_yen", 0.0).fillna(0.0)
    part["_profit"] = part["_return"] - part["_stake"]
    stake = float(part["_stake"].sum())
    ret = float(part["_return"].sum())
    ordered = part.sort_values(["year", "race_id", "ticket_type", "horse_a", "horse_b"], kind="mergesort")
    return {
        "label": label,
        "tickets": int(len(part)),
        "races": int(part["race_id"].nunique()) if "race_id" in part.columns else int(len(part)),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi_pct": float(ret / stake * 100.0) if stake else 0.0,
        "hit_rate_pct": float(part["_return"].gt(0).mean() * 100.0),
        "max_drawdown_yen": max_drawdown(ordered["_profit"]),
        "top1_removed_roi_pct": removed_roi(part, 1) * 100.0,
        "top3_removed_roi_pct": removed_roi(part, 3) * 100.0,
        "top5_removed_roi_pct": removed_roi(part, 5) * 100.0,
        "top10_removed_roi_pct": removed_roi(part, 10) * 100.0,
    }


def add_train_rank_features(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    out = df.copy()
    cols = [
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
    ]
    for col in cols:
        s = num(out, col, 0.0).fillna(0.0)
        out[f"{col}_train_pct"] = rank01_by_train(s.loc[train_mask], s)
    out["front_minus_closer_value"] = num(out, "position_front_value_score", 0.0).fillna(0.0) - num(
        out, "position_closer_value_score", 0.0
    ).fillna(0.0)
    out["front_minus_closer_value_train_pct"] = rank01_by_train(
        out.loc[train_mask, "front_minus_closer_value"], out["front_minus_closer_value"]
    )
    return out


@dataclass(frozen=True)
class Gate:
    name: str
    collapse_lo: float | None = None
    collapse_hi: float | None = None
    closer_lo: float | None = None
    closer_hi: float | None = None
    slow_hi: float | None = None
    clash_lo: float | None = None
    front_hi: float | None = None
    overlay_lo: float | None = None
    danger_hi: float | None = None
    skip_hi: float | None = None

    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=df.index)
        if self.collapse_lo is not None:
            m &= df["collapse_fit_train_pct"].ge(self.collapse_lo)
        if self.collapse_hi is not None:
            m &= df["collapse_fit_train_pct"].le(self.collapse_hi)
        if self.closer_lo is not None:
            m &= df["closer_pair_max_train_pct"].ge(self.closer_lo)
        if self.closer_hi is not None:
            m &= df["closer_pair_max_train_pct"].le(self.closer_hi)
        if self.slow_hi is not None:
            m &= df["front_front_slow_fit_train_pct"].le(self.slow_hi)
        if self.clash_lo is not None:
            m &= df["front_front_clash_train_pct"].ge(self.clash_lo)
        if self.front_hi is not None:
            m &= df["position_front_value_score_train_pct"].le(self.front_hi)
        if self.overlay_lo is not None:
            m &= df["market_overlay_score_train_pct"].ge(self.overlay_lo)
        if self.danger_hi is not None:
            m &= df["danger_sum_train_pct"].le(self.danger_hi)
        if self.skip_hi is not None:
            m &= df["skip_risk_score_train_pct"].le(self.skip_hi)
        return m


def fixed_gates() -> list[Gate]:
    return [
        Gate("baseline"),
        Gate(
            "closer_shadow_existing_like",
            collapse_lo=0.33,
            collapse_hi=0.85,
            closer_lo=0.33,
            slow_hi=0.75,
            clash_lo=0.25,
            front_hi=0.75,
            overlay_lo=0.33,
            danger_hi=1.00,
        ),
        Gate(
            "closer_shadow_safer",
            collapse_lo=0.33,
            collapse_hi=0.85,
            closer_lo=0.33,
            slow_hi=0.60,
            clash_lo=0.25,
            front_hi=0.60,
            overlay_lo=0.33,
            danger_hi=0.80,
        ),
        Gate(
            "closer_shadow_value_strict",
            collapse_lo=0.33,
            collapse_hi=0.85,
            closer_lo=0.33,
            slow_hi=0.75,
            clash_lo=0.25,
            front_hi=0.75,
            overlay_lo=0.50,
            danger_hi=0.80,
        ),
        Gate(
            "closer_lap_midcollapse_only",
            collapse_lo=0.40,
            collapse_hi=0.75,
            closer_lo=0.33,
            slow_hi=0.75,
            clash_lo=0.33,
            front_hi=0.75,
            overlay_lo=0.33,
            danger_hi=1.00,
        ),
        Gate(
            "closer_lap_no_front_bias",
            collapse_lo=0.33,
            collapse_hi=0.85,
            closer_lo=0.33,
            slow_hi=0.60,
            clash_lo=0.33,
            front_hi=0.50,
            overlay_lo=0.33,
            danger_hi=1.00,
        ),
        Gate(
            "closer_lap_quality_guard",
            collapse_lo=0.33,
            collapse_hi=0.85,
            closer_lo=0.33,
            slow_hi=0.75,
            clash_lo=0.25,
            front_hi=0.75,
            overlay_lo=0.33,
            danger_hi=0.80,
            skip_hi=0.80,
        ),
    ]


def grid_gates() -> list[Gate]:
    gates: list[Gate] = []
    # Keep this intentionally compact: a huge grid creates model-selection bias and is slow.
    collapse_lows = [0.25, 0.33]
    collapse_highs = [0.75, 0.85, 1.00]
    closer_lows = [0.20, 0.33]
    slow_highs = [0.60, 0.75, 1.00]
    clash_lows = [0.00, 0.25, 0.33]
    front_value_highs = [0.60, 0.75, 1.00]
    overlay_lows = [0.25, 0.33, 0.50]
    danger_highs = [0.80, 1.00]
    skip_highs = [1.00]
    for collapse_lo in collapse_lows:
        for collapse_hi in collapse_highs:
            if collapse_hi <= collapse_lo:
                continue
            for closer_lo in closer_lows:
                for slow_hi in slow_highs:
                    for clash_lo in clash_lows:
                        for front_hi in front_value_highs:
                            for overlay_lo in overlay_lows:
                                for danger_hi in danger_highs:
                                    for skip_hi in skip_highs:
                                        name = (
                                            f"grid_collapse{collapse_lo:.2f}-{collapse_hi:.2f}"
                                            f"_closer_ge{closer_lo:.2f}"
                                            f"_slow_le{slow_hi:.2f}"
                                            f"_clash_ge{clash_lo:.2f}"
                                            f"_front_le{front_hi:.2f}"
                                            f"_overlay_ge{overlay_lo:.2f}"
                                            f"_danger_le{danger_hi:.2f}"
                                            f"_skip_le{skip_hi:.2f}"
                                        )
                                        gates.append(
                                            Gate(
                                                name,
                                                collapse_lo=collapse_lo,
                                                collapse_hi=collapse_hi,
                                                closer_lo=closer_lo,
                                                slow_hi=slow_hi,
                                                clash_lo=clash_lo,
                                                front_hi=front_hi,
                                                overlay_lo=overlay_lo,
                                                danger_hi=danger_hi,
                                                skip_hi=skip_hi,
                                            )
                                        )
    return gates


def adoption_level(row: pd.Series, min_train_races: int, min_oos_races: int) -> str:
    train_races = int(row.get("train_races", 0) or 0)
    oos_races = int(row.get("oos_races", 0) or 0)
    train_roi = float(row.get("train_roi_pct", 0.0) or 0.0)
    oos_roi = float(row.get("oos_roi_pct", 0.0) or 0.0)
    train_top5 = float(row.get("train_top5_removed_roi_pct", 0.0) or 0.0)
    oos_top5 = float(row.get("oos_top5_removed_roi_pct", 0.0) or 0.0)
    oos_top10 = float(row.get("oos_top10_removed_roi_pct", 0.0) or 0.0)
    if (
        train_races >= min_train_races
        and oos_races >= min_oos_races
        and train_roi >= 110.0
        and train_top5 >= 100.0
        and oos_roi >= 120.0
        and oos_top5 >= 100.0
        and oos_top10 >= 85.0
    ):
        return "buy_gate_candidate"
    if oos_races >= min_oos_races and oos_roi >= 150.0:
        return "fragile_high_roi"
    return "shadow_only"


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
    parser = argparse.ArgumentParser(description="Validate closer-lap gates with train-derived thresholds and 2026 OOS.")
    parser.add_argument(
        "--scored-csv",
        default="outputs/analysis/position_value_gate_v2_with_closer/position_value_scored_tickets.csv",
    )
    parser.add_argument("--analysis-source", default="recommended_joint_guard")
    parser.add_argument("--train-max-year", type=int, default=2025)
    parser.add_argument("--oos-min-year", type=int, default=2026)
    parser.add_argument("--min-train-races", type=int, default=80)
    parser.add_argument("--min-oos-races", type=int, default=20)
    parser.add_argument("--include-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs/analysis/closer_lap_buy_gate_decision_v1")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project_path(args.scored_csv), encoding="utf-8-sig", low_memory=False)
    if "analysis_source" in raw.columns:
        raw = raw[raw["analysis_source"].astype(str).eq(args.analysis_source)].copy()
    raw["race_id"] = raw["race_id"].astype(str)
    raw["year"] = pd.to_numeric(raw.get("year", raw["race_id"].str[:4]), errors="coerce").astype("Int64")
    raw["ticket_type"] = raw.get("ticket_type", "unknown").astype(str)

    train_mask = raw["year"].le(args.train_max_year).fillna(False)
    oos_mask = raw["year"].ge(args.oos_min_year).fillna(False)
    scored = add_train_rank_features(raw, train_mask)

    rows: list[dict[str, Any]] = []
    baseline_by_type: list[dict[str, Any]] = []
    gates = fixed_gates()
    if args.include_grid:
        gates.extend(grid_gates())

    for ticket_type in ["all", *sorted(scored["ticket_type"].dropna().astype(str).unique())]:
        type_mask = pd.Series(True, index=scored.index) if ticket_type == "all" else scored["ticket_type"].eq(ticket_type)
        for gate in gates:
            gate_mask = gate.mask(scored) & type_mask
            train = metrics(scored, gate_mask & train_mask, "train")
            if gate.name.startswith("grid_") and train["races"] < args.min_train_races:
                continue
            oos = metrics(scored, gate_mask & oos_mask, "oos")
            row = {
                "ticket_type": ticket_type,
                "gate": gate.name,
                "train_tickets": train["tickets"],
                "train_races": train["races"],
                "train_roi_pct": train["roi_pct"],
                "train_hit_rate_pct": train["hit_rate_pct"],
                "train_profit_yen": train["profit_yen"],
                "train_max_drawdown_yen": train["max_drawdown_yen"],
                "train_top1_removed_roi_pct": train["top1_removed_roi_pct"],
                "train_top3_removed_roi_pct": train["top3_removed_roi_pct"],
                "train_top5_removed_roi_pct": train["top5_removed_roi_pct"],
                "train_top10_removed_roi_pct": train["top10_removed_roi_pct"],
                "oos_tickets": oos["tickets"],
                "oos_races": oos["races"],
                "oos_roi_pct": oos["roi_pct"],
                "oos_hit_rate_pct": oos["hit_rate_pct"],
                "oos_profit_yen": oos["profit_yen"],
                "oos_max_drawdown_yen": oos["max_drawdown_yen"],
                "oos_top1_removed_roi_pct": oos["top1_removed_roi_pct"],
                "oos_top3_removed_roi_pct": oos["top3_removed_roi_pct"],
                "oos_top5_removed_roi_pct": oos["top5_removed_roi_pct"],
                "oos_top10_removed_roi_pct": oos["top10_removed_roi_pct"],
                "coverage_vs_baseline_oos_pct": 0.0,
            }
            row["adoption_level"] = adoption_level(row, args.min_train_races, args.min_oos_races)
            rows.append(row)
            if gate.name == "baseline":
                baseline_by_type.append(row)

    out = pd.DataFrame(rows)
    baseline_oos_races = {
        r["ticket_type"]: max(int(r["oos_races"]), 1)
        for r in baseline_by_type
    }
    out["coverage_vs_baseline_oos_pct"] = out.apply(
        lambda r: float(r["oos_races"]) / baseline_oos_races.get(str(r["ticket_type"]), 1) * 100.0,
        axis=1,
    )
    out["robust_score"] = (
        0.26 * (out["train_roi_pct"].clip(0.0, 250.0) / 100.0)
        + 0.30 * (out["oos_roi_pct"].clip(0.0, 300.0) / 100.0)
        + 0.20 * (out["oos_top5_removed_roi_pct"].clip(0.0, 220.0) / 100.0)
        + 0.12 * (out["train_top5_removed_roi_pct"].clip(0.0, 180.0) / 100.0)
        + 0.08 * (out["oos_hit_rate_pct"].clip(0.0, 50.0) / 10.0)
        + 0.04 * np.log1p(out["oos_races"].clip(lower=0)) / np.log1p(max(float(out["oos_races"].max()), 1.0))
    )
    out = out.sort_values(
        ["adoption_level", "robust_score", "oos_top5_removed_roi_pct", "oos_roi_pct", "oos_races"],
        ascending=[True, False, False, False, False],
    )

    metrics_path = out_dir / "gate_metrics_by_split.csv"
    rec_path = out_dir / "gate_recommendations.csv"
    out.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    rec = out[out["gate"].ne("baseline")].sort_values(
        ["robust_score", "oos_top5_removed_roi_pct", "oos_roi_pct"],
        ascending=[False, False, False],
    )
    rec.to_csv(rec_path, index=False, encoding="utf-8-sig")

    level_counts = out["adoption_level"].value_counts(dropna=False).to_dict()
    summary = {
        "inputs": {
            "scored_csv": str(project_path(args.scored_csv)),
            "analysis_source": args.analysis_source,
            "train_max_year": args.train_max_year,
            "oos_min_year": args.oos_min_year,
            "min_train_races": args.min_train_races,
            "min_oos_races": args.min_oos_races,
        },
        "rows": int(len(scored)),
        "years": [int(y) for y in sorted(scored["year"].dropna().unique())],
        "adoption_level_counts": level_counts,
        "top_recommendations": rec.head(12).to_dict(orient="records"),
        "outputs": {
            "gate_metrics_by_split_csv": str(metrics_path),
            "gate_recommendations_csv": str(rec_path),
        },
        "decision_note": (
            "Promote only buy_gate_candidate rows. fragile_high_roi rows are useful for display/shadow but "
            "not enough for formal BUY because payoff concentration or train/OOS stability is insufficient."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
