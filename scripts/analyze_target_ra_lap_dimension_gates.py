from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


DIMENSIONS = {
    "front_load": {
        "need": "official_need_front_load",
        "anchor_mean": "anchor_official_front_load_goodrun_score_past3_mean",
        "partner_mean": "partner_official_front_load_goodrun_score_past3_mean",
        "anchor_max": "anchor_official_front_load_goodrun_score_past3_max",
        "partner_max": "partner_official_front_load_goodrun_score_past3_max",
    },
    "slow_finish": {
        "need": "official_need_slow_finish",
        "anchor_mean": "anchor_official_slow_finish_goodrun_score_past3_mean",
        "partner_mean": "partner_official_slow_finish_goodrun_score_past3_mean",
        "anchor_max": "anchor_official_slow_finish_goodrun_score_past3_max",
        "partner_max": "partner_official_slow_finish_goodrun_score_past3_max",
    },
    "l1_instant": {
        "need": "official_need_l1_instant",
        "anchor_mean": "anchor_official_l1_instant_goodrun_score_past3_mean",
        "partner_mean": "partner_official_l1_instant_goodrun_score_past3_mean",
        "anchor_max": "anchor_official_l1_instant_goodrun_score_past3_max",
        "partner_max": "partner_official_l1_instant_goodrun_score_past3_max",
    },
    "l2_sustain": {
        "need": "official_need_l2_sustain",
        "anchor_mean": "anchor_official_l2_sustain_goodrun_score_past3_mean",
        "partner_mean": "partner_official_l2_sustain_goodrun_score_past3_mean",
        "anchor_max": "anchor_official_l2_sustain_goodrun_score_past3_max",
        "partner_max": "partner_official_l2_sustain_goodrun_score_past3_max",
    },
    "l3_long_spurt": {
        "need": "official_need_l3_long_spurt",
        "anchor_mean": "anchor_official_l3_long_spurt_goodrun_score_past3_mean",
        "partner_mean": "partner_official_l3_long_spurt_goodrun_score_past3_mean",
        "anchor_max": "anchor_official_l3_long_spurt_goodrun_score_past3_max",
        "partner_max": "partner_official_l3_long_spurt_goodrun_score_past3_max",
    },
}


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = profit.cumsum()
    return float((curve - curve.cummax()).min())


def removed_roi(part: pd.DataFrame, n: int) -> float:
    if part.empty or len(part) <= n:
        return 0.0
    kept = part.sort_values("_profit", ascending=False).iloc[n:]
    stake = float(kept["_stake"].sum())
    return float(kept["_return"].sum() / stake * 100.0) if stake else 0.0


def metrics(df: pd.DataFrame, mask: pd.Series, label: str) -> dict[str, Any]:
    part = df.loc[mask].copy()
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
    part["_stake"] = num(part, "stake_yen", 100.0)
    part["_return"] = num(part, "return_yen", 0.0)
    part["_profit"] = part["_return"] - part["_stake"]
    stake = float(part["_stake"].sum())
    ret = float(part["_return"].sum())
    ordered = part.sort_values(["year", "race_id"], kind="mergesort")
    return {
        "label": label,
        "tickets": int(len(part)),
        "races": int(part["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi_pct": float(ret / stake * 100.0) if stake else 0.0,
        "hit_rate_pct": float(part["_return"].gt(0).mean() * 100.0),
        "max_drawdown_yen": max_drawdown(ordered["_profit"]),
        "top1_removed_roi_pct": removed_roi(part, 1),
        "top3_removed_roi_pct": removed_roi(part, 3),
        "top5_removed_roi_pct": removed_roi(part, 5),
        "top10_removed_roi_pct": removed_roi(part, 10),
    }


def add_dimension_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["official_pair_ready_both"] = num(out, "official_pair_ready_both", 0.0).ge(0.5)
    for dim, cols in DIMENSIONS.items():
        need = num(out, cols["need"], 0.0).clip(0.0, 1.0)
        a_mean = num(out, cols["anchor_mean"], 0.0).clip(0.0, 1.0)
        p_mean = num(out, cols["partner_mean"], 0.0).clip(0.0, 1.0)
        a_max = num(out, cols["anchor_max"], 0.0).clip(0.0, 1.0)
        p_max = num(out, cols["partner_max"], 0.0).clip(0.0, 1.0)
        pair_mean_avg = (a_mean + p_mean) / 2.0
        pair_mean_min = np.minimum(a_mean, p_mean)
        pair_best_avg = (np.maximum(a_mean, a_max) + np.maximum(p_mean, p_max)) / 2.0
        out[f"{dim}_need"] = need
        out[f"{dim}_pair_mean_avg"] = pair_mean_avg
        out[f"{dim}_pair_mean_min"] = pair_mean_min
        out[f"{dim}_pair_best_avg"] = pair_best_avg
        out[f"{dim}_match_score"] = (need * (0.65 * pair_mean_avg + 0.35 * pair_best_avg)).clip(0.0, 1.0)
    return out


def quantile(frame: pd.DataFrame, col: str, q: float, default: float) -> float:
    values = pd.to_numeric(frame.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.quantile(q))


def adoption_level(row: pd.Series) -> str:
    if (
        int(row["train_races"]) >= 80
        and int(row["oos_races"]) >= 20
        and float(row["train_roi_pct"]) >= 110.0
        and float(row["train_top5_removed_roi_pct"]) >= 100.0
        and float(row["oos_roi_pct"]) >= 120.0
        and float(row["oos_top5_removed_roi_pct"]) >= 100.0
    ):
        return "buy_gate_candidate"
    if int(row["oos_races"]) >= 10 and float(row["oos_roi_pct"]) >= 150.0:
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
    parser = argparse.ArgumentParser(description="Analyze TARGET RA official lap dimension gates.")
    parser.add_argument(
        "--input-csv",
        default="outputs/analysis/target_ra_official_lap_history_overlay_v1/tickets_with_target_ra_official_lap_overlay.csv",
    )
    parser.add_argument("--train-max-year", type=int, default=2025)
    parser.add_argument("--oos-min-year", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/analysis/target_ra_lap_dimension_gates_v1")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project_path(args.input_csv), encoding="utf-8-sig", low_memory=False)
    raw["race_id"] = raw["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    raw["year"] = pd.to_numeric(raw.get("year", raw["race_id"].str[:4]), errors="coerce").astype("Int64")
    raw["policy"] = raw.get("policy", "unknown").astype(str)
    raw["ticket_type"] = raw.get("ticket_type", "unknown").astype(str)
    scored = add_dimension_scores(raw)

    rows: list[dict[str, Any]] = []
    for (policy, ticket_type), frame in scored.groupby(["policy", "ticket_type"], dropna=False):
        train_mask = frame["year"].le(args.train_max_year).fillna(False)
        oos_mask = frame["year"].ge(args.oos_min_year).fillna(False)
        train_frame = frame.loc[train_mask]
        gates: list[tuple[str, pd.Series]] = [
            ("base", pd.Series(True, index=frame.index)),
            ("ready_both", frame["official_pair_ready_both"]),
            (
                "overall_fit_q60_risk_q60",
                num(frame, "official_pair_fit_max_min", 0.0).ge(
                    quantile(train_frame, "official_pair_fit_max_min", 0.60, 0.0)
                )
                & num(frame, "official_pair_mismatch_risk", 1.0).le(
                    quantile(train_frame, "official_pair_mismatch_risk", 0.60, 1.0)
                ),
            ),
        ]
        for dim in DIMENSIONS:
            need_q60 = quantile(train_frame, f"{dim}_need", 0.60, 0.0)
            need_q70 = quantile(train_frame, f"{dim}_need", 0.70, 0.0)
            match_q60 = quantile(train_frame, f"{dim}_match_score", 0.60, 0.0)
            match_q70 = quantile(train_frame, f"{dim}_match_score", 0.70, 0.0)
            min_q50 = quantile(train_frame, f"{dim}_pair_mean_min", 0.50, 0.0)
            gates.extend(
                [
                    (
                        f"{dim}_need60_match60",
                        frame[f"{dim}_need"].ge(need_q60) & frame[f"{dim}_match_score"].ge(match_q60),
                    ),
                    (
                        f"{dim}_need70_match60",
                        frame[f"{dim}_need"].ge(need_q70) & frame[f"{dim}_match_score"].ge(match_q60),
                    ),
                    (
                        f"{dim}_need60_match70",
                        frame[f"{dim}_need"].ge(need_q60) & frame[f"{dim}_match_score"].ge(match_q70),
                    ),
                    (
                        f"{dim}_need60_match60_pairmin50",
                        frame[f"{dim}_need"].ge(need_q60)
                        & frame[f"{dim}_match_score"].ge(match_q60)
                        & frame[f"{dim}_pair_mean_min"].ge(min_q50),
                    ),
                ]
            )
        for gate_name, mask in gates:
            train = metrics(frame, mask & train_mask, "train")
            oos = metrics(frame, mask & oos_mask, "oos")
            row = {
                "policy": policy,
                "ticket_type": ticket_type,
                "gate": gate_name,
                "train_tickets": train["tickets"],
                "train_races": train["races"],
                "train_roi_pct": train["roi_pct"],
                "train_hit_rate_pct": train["hit_rate_pct"],
                "train_profit_yen": train["profit_yen"],
                "train_top1_removed_roi_pct": train["top1_removed_roi_pct"],
                "train_top3_removed_roi_pct": train["top3_removed_roi_pct"],
                "train_top5_removed_roi_pct": train["top5_removed_roi_pct"],
                "oos_tickets": oos["tickets"],
                "oos_races": oos["races"],
                "oos_roi_pct": oos["roi_pct"],
                "oos_hit_rate_pct": oos["hit_rate_pct"],
                "oos_profit_yen": oos["profit_yen"],
                "oos_top1_removed_roi_pct": oos["top1_removed_roi_pct"],
                "oos_top3_removed_roi_pct": oos["top3_removed_roi_pct"],
                "oos_top5_removed_roi_pct": oos["top5_removed_roi_pct"],
            }
            row["adoption_level"] = adoption_level(pd.Series(row))
            rows.append(row)

    out = pd.DataFrame(rows)
    out["robust_score"] = (
        0.25 * (out["train_roi_pct"].clip(0.0, 250.0) / 100.0)
        + 0.30 * (out["oos_roi_pct"].clip(0.0, 300.0) / 100.0)
        + 0.20 * (out["train_top5_removed_roi_pct"].clip(0.0, 200.0) / 100.0)
        + 0.20 * (out["oos_top5_removed_roi_pct"].clip(0.0, 200.0) / 100.0)
        + 0.05 * np.log1p(out["oos_races"].clip(lower=0)) / np.log1p(max(float(out["oos_races"].max()), 1.0))
    )
    out = out.sort_values(
        ["adoption_level", "robust_score", "oos_top5_removed_roi_pct", "oos_roi_pct", "oos_races"],
        ascending=[True, False, False, False, False],
    )
    metrics_path = out_dir / "dimension_gate_metrics.csv"
    rec_path = out_dir / "dimension_gate_recommendations.csv"
    out.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    out[out["gate"].ne("base")].to_csv(rec_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_csv": str(project_path(args.input_csv)),
        "output_dir": str(out_dir),
        "rows": int(len(scored)),
        "years": [int(y) for y in sorted(scored["year"].dropna().unique())],
        "adoption_level_counts": out["adoption_level"].value_counts(dropna=False).to_dict(),
        "top_recommendations": out[out["gate"].ne("base")].head(20).to_dict(orient="records"),
        "outputs": {
            "metrics_csv": str(metrics_path),
            "recommendations_csv": str(rec_path),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
