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


def pct_rank(s: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() <= 1:
        return pd.Series(0.5, index=x.index, dtype=float)
    rank = x.rank(pct=True, ascending=True)
    if not higher_is_better:
        rank = 1.0 - rank
    return rank.fillna(0.5).clip(0.0, 1.0)


def add_position_value_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["year"] = pd.to_numeric(out.get("year", out["race_id"].str[:4]), errors="coerce").astype("Int64")

    max_front_adv = np.maximum(
        num(out, "anchor_front_advantage_score", 0.0).fillna(0.0),
        num(out, "partner_front_advantage_score", 0.0).fillna(0.0),
    )
    max_draw_pace = np.maximum(
        num(out, "anchor_draw_pace_fit_score", 0.0).fillna(0.0),
        num(out, "partner_draw_pace_fit_score", 0.0).fillna(0.0),
    )
    max_same_day_bias = np.maximum(
        num(out, "anchor_same_day_bias_fit_score", 0.0).fillna(0.0),
        num(out, "partner_same_day_bias_fit_score", 0.0).fillna(0.0),
    )
    front_risk = (
        num(out, "front_front_clash", 0.0).fillna(0.0)
        + num(out, "collapse_fit", 0.0).fillna(0.0)
    )

    out["position_front_value_score"] = (
        0.34 * pct_rank(num(out, "front_front_slow_fit", 0.0))
        + 0.22 * pct_rank(max_front_adv)
        + 0.16 * pct_rank(max_draw_pace)
        + 0.12 * pct_rank(max_same_day_bias)
        + 0.16 * pct_rank(front_risk, higher_is_better=False)
    ).clip(0.0, 1.0)

    out["position_closer_value_score"] = (
        0.38 * pct_rank(num(out, "collapse_fit", 0.0))
        + 0.24 * pct_rank(num(out, "closer_pair_max", 0.0))
        + 0.18 * pct_rank(num(out, "style_diversity", 0.0))
        + 0.12 * pct_rank(max_draw_pace)
        + 0.08 * pct_rank(num(out, "front_front_slow_fit", 0.0), higher_is_better=False)
    ).clip(0.0, 1.0)

    out["position_value_score"] = np.maximum(
        out["position_front_value_score"],
        out["position_closer_value_score"],
    )
    out["front_value_mismatch_risk"] = (
        pct_rank(num(out, "projected_front5_prob", 0.0)) * (1.0 - out["position_front_value_score"])
    ).clip(0.0, 1.0)
    out["front_value_gap_vs_closer"] = out["position_front_value_score"] - out["position_closer_value_score"]
    return out


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
        }
    stake = num(part, "stake_yen", 0.0).fillna(0.0)
    ret = num(part, "return_yen", 0.0).fillna(0.0)
    profit = ret - stake
    part["_stake"] = stake
    part["_return"] = ret
    part["_profit"] = profit
    hit = ret.gt(0)

    def removed_roi(n: int) -> float:
        if len(part) <= n:
            return 0.0
        kept = part.sort_values("_profit", ascending=False).iloc[n:]
        kept_stake = float(kept["_stake"].sum())
        return float(kept["_return"].sum() / kept_stake) if kept_stake else 0.0

    yearly = {}
    for year, g in part.groupby("year"):
        y_stake = float(g["_stake"].sum())
        if y_stake:
            yearly[f"roi_{int(year)}"] = float(g["_return"].sum() / y_stake)

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
        "hit_rate": float(hit.mean()),
        "max_drawdown_yen": max_drawdown(part.sort_values(["year", "race_id"])["_profit"]),
        "top5_removed_roi": removed_roi(5),
        "top10_removed_roi": removed_roi(10),
        "roi_2025": yearly.get("roi_2025", np.nan),
        "roi_2026": yearly.get("roi_2026", np.nan),
    }


def threshold_rows(df: pd.DataFrame, label_prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [metrics(df, pd.Series(True, index=df.index), f"{label_prefix}_all")]
    for col in [
        "position_front_value_score",
        "position_closer_value_score",
        "position_value_score",
        "front_value_mismatch_risk",
    ]:
        for q in [0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75]:
            threshold = float(df[col].quantile(q))
            if col == "front_value_mismatch_risk":
                mask = df[col].le(threshold)
                suffix = f"{col}_le_q{int(q*100):02d}"
            else:
                mask = df[col].ge(threshold)
                suffix = f"{col}_ge_q{int(q*100):02d}"
            row = metrics(df, mask, f"{label_prefix}_{suffix}")
            row["threshold_col"] = col
            row["threshold_quantile"] = q
            row["threshold_value"] = threshold
            rows.append(row)
    return rows


def segment_bins(df: pd.DataFrame, col: str, label_prefix: str) -> pd.DataFrame:
    out = df.copy()
    try:
        out["_bin"] = pd.qcut(out[col].rank(method="first"), 5, labels=[f"q{i}" for i in range(1, 6)])
    except ValueError:
        return pd.DataFrame()
    rows = []
    for bucket, part in out.groupby("_bin", observed=True):
        row = metrics(part, pd.Series(True, index=part.index), f"{label_prefix}_{col}_{bucket}")
        row["score_col"] = col
        row["bucket"] = str(bucket)
        row["avg_score"] = float(pd.to_numeric(part[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


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
    parser = argparse.ArgumentParser(description="Evaluate course/pace position-value gates for current pair tickets.")
    parser.add_argument(
        "--recommended-csv",
        default="outputs/analysis/pair_joint_probability_v2_rebuilt_20260623/recommended_joint_guard_tickets.csv",
    )
    parser.add_argument(
        "--policy-csv",
        default="outputs/analysis/pair_joint_probability_v2_rebuilt_20260623/pair_joint_v2_policy_tickets.csv",
    )
    parser.add_argument("--output-dir", default="outputs/analysis/position_value_gate_v1")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[tuple[str, pd.DataFrame]] = []
    recommended = pd.read_csv(project_path(args.recommended_csv), encoding="utf-8-sig", low_memory=False)
    frames.append(("recommended_joint_guard", add_position_value_scores(recommended)))

    policy = pd.read_csv(project_path(args.policy_csv), encoding="utf-8-sig", low_memory=False)
    if "policy" in policy.columns:
        base = policy[(policy["policy"].astype(str).eq("existing_quality_wide_cov10")) & (policy["ticket_type"].astype(str).eq("wide"))].copy()
        if not base.empty:
            frames.append(("existing_quality_wide_cov10", add_position_value_scores(base)))

    all_rows: list[dict[str, Any]] = []
    segment_frames: list[pd.DataFrame] = []
    scored_outputs = []
    for label, frame in frames:
        rows = threshold_rows(frame, label)
        all_rows.extend(rows)
        for score_col in ["position_front_value_score", "position_closer_value_score", "position_value_score", "front_value_mismatch_risk"]:
            seg = segment_bins(frame, score_col, label)
            if not seg.empty:
                segment_frames.append(seg)
        scored = frame.copy()
        scored["analysis_source"] = label
        scored_outputs.append(scored)

    comparison = pd.DataFrame(all_rows)
    comparison = comparison.sort_values(["label"]).reset_index(drop=True)
    segments = pd.concat(segment_frames, ignore_index=True, sort=False) if segment_frames else pd.DataFrame()
    scored_all = pd.concat(scored_outputs, ignore_index=True, sort=False)

    comparison.to_csv(out_dir / "position_value_gate_comparison.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(out_dir / "position_value_segments.csv", index=False, encoding="utf-8-sig")
    scored_all.to_csv(out_dir / "position_value_scored_tickets.csv", index=False, encoding="utf-8-sig")

    best = comparison[comparison["races"].ge(50)].sort_values(["roi", "races"], ascending=[False, False]).head(12)
    summary = {
        "inputs": {
            "recommended_csv": str(project_path(args.recommended_csv)),
            "policy_csv": str(project_path(args.policy_csv)),
        },
        "rows": {
            "comparison": int(len(comparison)),
            "segments": int(len(segments)),
            "scored_tickets": int(len(scored_all)),
        },
        "baseline": comparison[comparison["label"].str.endswith("_all")].to_dict(orient="records"),
        "best_min50_races": best.to_dict(orient="records"),
        "outputs": {
            "comparison_csv": str(out_dir / "position_value_gate_comparison.csv"),
            "segments_csv": str(out_dir / "position_value_segments.csv"),
            "scored_tickets_csv": str(out_dir / "position_value_scored_tickets.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
