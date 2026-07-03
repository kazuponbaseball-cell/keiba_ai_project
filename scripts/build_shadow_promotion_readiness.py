from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RNG_SEED = 20260622


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={"race_id": str, "raceId": str}, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype={"race_id": str, "raceId": str}, low_memory=False)


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def text(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=str)
    return frame[col].astype("string").fillna(default).astype(str)


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


def max_drawdown(profits: pd.Series) -> float:
    if profits.empty:
        return 0.0
    curve = pd.to_numeric(profits, errors="coerce").fillna(0.0).cumsum()
    peak = curve.cummax()
    return float((curve - peak).min())


def top_removed_roi(rows: pd.DataFrame, top_n: int) -> float | None:
    if rows.empty:
        return None
    stake = float(num(rows, "proxy_stake_yen", 100.0).sum())
    if stake <= 0:
        return None
    work = rows.copy()
    top_idx = work["proxy_return_yen"].sort_values(ascending=False).index[:top_n]
    work.loc[top_idx, "proxy_return_yen"] = 0.0
    return float(work["proxy_return_yen"].sum() / stake * 100.0)


def winsorized_roi(rows: pd.DataFrame, percentile: float = 0.95) -> float | None:
    if rows.empty:
        return None
    stake = float(num(rows, "proxy_stake_yen", 100.0).sum())
    if stake <= 0:
        return None
    returns = num(rows, "proxy_return_yen", 0.0)
    positive = returns[returns.gt(0)]
    if positive.empty:
        capped = returns
    else:
        cap = float(positive.quantile(percentile))
        capped = returns.clip(upper=cap)
    return float(capped.sum() / stake * 100.0)


def race_level(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["race_id", "date_key", "stake_yen", "return_yen", "profit_yen", "hit"])
    work = rows.copy()
    work["date_key"] = text(work, "race_id").str.slice(0, 8)
    grouped = (
        work.groupby("race_id", dropna=False)
        .agg(
            date_key=("date_key", "first"),
            stake_yen=("proxy_stake_yen", "sum"),
            return_yen=("proxy_return_yen", "sum"),
            profit_yen=("proxy_profit_yen", "sum"),
            hit=("proxy_return_yen", lambda s: bool((pd.to_numeric(s, errors="coerce").fillna(0) > 0).any())),
        )
        .reset_index()
        .sort_values(["date_key", "race_id"])
    )
    return grouped


def bootstrap_prob_true_roi_gt_100(rows: pd.DataFrame, iterations: int = 2000) -> float | None:
    races = race_level(rows)
    if races.empty or len(races) < 2:
        return None
    stake = pd.to_numeric(races["stake_yen"], errors="coerce").fillna(0).to_numpy(dtype=float)
    returns = pd.to_numeric(races["return_yen"], errors="coerce").fillna(0).to_numpy(dtype=float)
    if stake.sum() <= 0:
        return None
    rng = np.random.default_rng(RNG_SEED)
    count = 0
    n = len(races)
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        sample_stake = float(stake[idx].sum())
        sample_return = float(returns[idx].sum())
        if sample_stake > 0 and sample_return / sample_stake > 1.0:
            count += 1
    return count / iterations


def chronological_blocks(rows: pd.DataFrame, target_blocks: int = 3) -> list[pd.DataFrame]:
    races = race_level(rows)
    if races.empty:
        return []
    unique_dates = sorted(races["date_key"].dropna().unique())
    if len(unique_dates) < target_blocks:
        return [races[races["date_key"].eq(date)].copy() for date in unique_dates]
    date_groups = np.array_split(unique_dates, target_blocks)
    blocks = []
    for dates in date_groups:
        blocks.append(races[races["date_key"].isin(list(dates))].copy())
    return blocks


def block_profitability(rows: pd.DataFrame) -> tuple[int, int]:
    blocks = chronological_blocks(rows)
    if not blocks:
        return 0, 0
    profitable = 0
    for block in blocks:
        if float(block["return_yen"].sum() - block["stake_yen"].sum()) > 0:
            profitable += 1
    return profitable, len(blocks)


def metric_row(rows: pd.DataFrame, label: str, pool_rows: int) -> dict[str, Any]:
    if rows.empty:
        return {
            "label": label,
            "pool_rows": int(pool_rows),
            "candidates": 0,
            "races": 0,
            "race_days": 0,
            "hits": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": None,
        }
    ordered = rows.sort_values(["race_id", "shadow_rank_score"], ascending=[True, False]).copy()
    stake = float(num(ordered, "proxy_stake_yen", 100.0).sum())
    ret = float(num(ordered, "proxy_return_yen", 0.0).sum())
    positive_returns = num(ordered, "proxy_return_yen", 0.0)
    top_return = float(positive_returns.max()) if len(positive_returns) else 0.0
    profitable_blocks, block_count = block_profitability(ordered)
    bootstrap = bootstrap_prob_true_roi_gt_100(ordered)
    return {
        "label": label,
        "pool_rows": int(pool_rows),
        "candidates": int(len(ordered)),
        "races": int(ordered["race_id"].nunique()),
        "race_days": int(text(ordered, "race_id").str.slice(0, 8).nunique()),
        "hits": int(num(ordered, "proxy_hit_umaren_top2", 0).sum()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi_pct": (ret / stake * 100.0) if stake > 0 else None,
        "max_drawdown_yen": max_drawdown(ordered["proxy_profit_yen"]),
        "avg_live_odds": float(num(ordered, "live_odds", np.nan).mean()),
        "avg_score": float(num(ordered, "strongest_current_score", np.nan).mean()),
        "avg_score_gap_z": float(num(ordered, "score_gap_z", np.nan).mean()),
        "avg_conservative_ev_q20": float(num(ordered, "conservative_expected_roi_q20", np.nan).mean()),
        "avg_value_survival_probability": float(num(ordered, "value_survival_probability", np.nan).mean()),
        "top1_return_concentration_pct": (top_return / ret * 100.0) if ret > 0 else 0.0,
        "top1_removed_roi_pct": top_removed_roi(ordered, 1),
        "top3_removed_roi_pct": top_removed_roi(ordered, 3),
        "winsorized_95p_roi_pct": winsorized_roi(ordered, 0.95),
        "profitable_blocks": profitable_blocks,
        "block_count": block_count,
        "bootstrap_prob_roi_gt_100": bootstrap,
    }


def add_shadow_rank_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score_gap = num(out, "score_gap_z", 9.0)
    score_near = (1.0 - (score_gap / 0.50)).clip(0.0, 1.0)
    ev = np.log1p(num(out, "conservative_expected_roi_q20", 0.0)).clip(0, math.log1p(12.0)) / math.log1p(12.0)
    survival = num(out, "value_survival_probability", 0.0).clip(0.0, 1.0)
    fragility = (1.0 - num(out, "pair_fragility_proxy", 1.0)).clip(0.0, 1.0)
    dispersion = (1.0 - num(out, "model_dispersion_proxy", 1.0)).clip(0.0, 1.0)
    front5 = num(out, "projected_front5_prob", 0.0).clip(0.0, 1.0)
    out["shadow_rank_score"] = (
        0.22 * score_near
        + 0.22 * ev
        + 0.18 * survival
        + 0.16 * fragility
        + 0.12 * front5
        + 0.10 * dispersion
    )
    return out


def build_pool_map(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    reasons = text(df, "rejection_reasons")
    reason_count = num(df, "rejection_reason_count", 99)
    pools = {
        "challenger_s_score_only_green": df[df["shadow_action"].eq("CHALLENGER_S_SCORE_ONLY_GREEN_SHADOW")].copy(),
        "watch_score_only_near_not_green": df[df["shadow_action"].eq("WATCH_SCORE_ONLY_NEAR_FAIL_NOT_GREEN")].copy(),
        "score_only_single_fail_all": df[reasons.eq("SCORE_FAIL") & reason_count.eq(1)].copy(),
        "score_only_gap_le_050": df[reasons.eq("SCORE_FAIL") & reason_count.eq(1) & num(df, "score_gap_z", 9).between(0, 0.50)].copy(),
        "single_fail_watchable": df[df["shadow_action"].isin(["CHALLENGER_S_SCORE_ONLY_GREEN_SHADOW", "WATCH_SCORE_ONLY_NEAR_FAIL_NOT_GREEN", "WATCH_ONLY_SINGLE_FAIL"])].copy(),
        "focus_champion_or_challenger": df[
            df["shadow_action"].str.startswith("CHAMPION_", na=False)
            | df["shadow_action"].str.startswith("CHALLENGER_", na=False)
        ].copy(),
    }
    return {name: part for name, part in pools.items() if not part.empty}


def risk_coverage_rows(df: pd.DataFrame) -> pd.DataFrame:
    coverages = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
    rows: list[dict[str, Any]] = []
    for pool_name, pool in build_pool_map(df).items():
        ordered = pool.sort_values("shadow_rank_score", ascending=False).reset_index(drop=True)
        for cov in coverages:
            n = max(1, int(math.ceil(len(ordered) * cov)))
            rows.append(metric_row(ordered.head(n), f"{pool_name}_top{int(cov * 100)}pct", len(ordered)))
    return pd.DataFrame(rows)


def sensitivity_rows(df: pd.DataFrame) -> pd.DataFrame:
    reasons = text(df, "rejection_reasons")
    reason_count = num(df, "rejection_reason_count", 99)
    base = df[reasons.eq("SCORE_FAIL") & reason_count.eq(1)].copy()
    rows: list[dict[str, Any]] = []
    if base.empty:
        return pd.DataFrame()
    for gap in [0.20, 0.25, 0.30, 0.40, 0.50]:
        subset = base[
            num(base, "score_gap_z", 9).gt(0)
            & num(base, "score_gap_z", 9).le(gap)
            & num(base, "live_odds", 9999).le(120)
            & num(base, "value_survival_probability", 0).ge(0.80)
            & num(base, "conservative_expected_roi_q20", 0).ge(1.10)
        ].copy()
        rows.append(metric_row(subset.sort_values("shadow_rank_score", ascending=False), f"score_gap_z_le_{gap:.2f}", len(base)))
    return pd.DataFrame(rows)


def promotion_gate_rows(pool_name: str, rows: pd.DataFrame) -> pd.DataFrame:
    m = metric_row(rows, pool_name, len(rows))
    profitable_blocks = int(m.get("profitable_blocks") or 0)
    block_count = int(m.get("block_count") or 0)
    canary_checks = {
        "tickets_ge_120": m["candidates"] >= 120,
        "races_ge_80": m["races"] >= 80,
        "race_days_ge_12": m["race_days"] >= 12,
        "hits_ge_10": m["hits"] >= 10,
        "blocks_ge_3": block_count >= 3,
        "final_odds_roi_ge_120": (m.get("roi_pct") or 0) >= 120.0,
        "flat_100yen_roi_ge_110": (m.get("roi_pct") or 0) >= 110.0,
        "two_of_three_blocks_profitable": block_count >= 3 and profitable_blocks >= 2,
        "top1_return_concentration_le_30": (m.get("top1_return_concentration_pct") or 999) <= 30.0,
        "top3_removed_roi_ge_100": (m.get("top3_removed_roi_pct") or 0) >= 100.0,
        "bootstrap_prob_roi_gt_100_ge_80": (m.get("bootstrap_prob_roi_gt_100") or 0) >= 0.80,
    }
    formal_checks = {
        "tickets_ge_250": m["candidates"] >= 250,
        "races_ge_150": m["races"] >= 150,
        "race_days_ge_24": m["race_days"] >= 24,
        "hits_ge_20": m["hits"] >= 20,
        "final_odds_roi_ge_150": (m.get("roi_pct") or 0) >= 150.0,
        "flat_100yen_roi_ge_120": (m.get("roi_pct") or 0) >= 120.0,
        "bootstrap_prob_roi_gt_100_ge_90": (m.get("bootstrap_prob_roi_gt_100") or 0) >= 0.90,
        "top3_removed_roi_ge_100": (m.get("top3_removed_roi_pct") or 0) >= 100.0,
        "winsorized_95p_roi_ge_100": (m.get("winsorized_95p_roi_pct") or 0) >= 100.0,
        "threshold_stability_verified": False,
        "venue_month_going_dependence_cleared": False,
        "champion_combined_downside_not_worse": False,
    }
    out = []
    for tier, checks in [("canary", canary_checks), ("formal_buy", formal_checks)]:
        for check, ok in checks.items():
            out.append(
                {
                    "pool": pool_name,
                    "tier": tier,
                    "check": check,
                    "passed": bool(ok),
                }
            )
        out.append(
            {
                "pool": pool_name,
                "tier": tier,
                "check": "ALL_CHECKS",
                "passed": all(checks.values()),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shadow Challenger promotion readiness and risk-coverage report.")
    parser.add_argument("--shadow-csv", default="outputs/analysis/shadow_challenger_candidates_v1/shadow_challenger_candidates.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/shadow_promotion_readiness_v1")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shadow = read_csv(project_path(args.shadow_csv))
    if shadow.empty:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "rows": 0,
            "reason": "missing shadow candidates",
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    shadow = add_shadow_rank_score(shadow)
    coverage = risk_coverage_rows(shadow)
    sensitivity = sensitivity_rows(shadow)
    pools = build_pool_map(shadow)
    pool_summary = pd.DataFrame([metric_row(part, name, len(part)) for name, part in pools.items()])
    gate_checks = pd.concat(
        [promotion_gate_rows(name, part) for name, part in pools.items()],
        ignore_index=True,
    )
    top_candidates = (
        shadow.sort_values("shadow_rank_score", ascending=False)
        .head(80)
        .copy()
    )

    shadow.to_csv(out_dir / "shadow_candidates_scored.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(out_dir / "risk_coverage.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(out_dir / "score_gap_sensitivity.csv", index=False, encoding="utf-8-sig")
    pool_summary.to_csv(out_dir / "pool_summary.csv", index=False, encoding="utf-8-sig")
    gate_checks.to_csv(out_dir / "promotion_gate_check.csv", index=False, encoding="utf-8-sig")
    top_candidates.to_csv(out_dir / "top_shadow_candidates.csv", index=False, encoding="utf-8-sig")

    all_checks = gate_checks[gate_checks["check"].eq("ALL_CHECKS")].copy()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(shadow)),
        "races": int(shadow["race_id"].nunique()),
        "pool_summary": pool_summary.to_dict(orient="records"),
        "promotion_all_checks": all_checks.to_dict(orient="records"),
        "score_gap_sensitivity": sensitivity.to_dict(orient="records"),
        "policy": {
            "champion_changed": False,
            "live_buy_change": False,
            "promotion_allowed_now": bool(all_checks["passed"].any()) if not all_checks.empty else False,
            "note": "Promotion requires fixed OOS evidence. Current report may rank shadow pools, but cannot authorize live BUY without all gates passing.",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
