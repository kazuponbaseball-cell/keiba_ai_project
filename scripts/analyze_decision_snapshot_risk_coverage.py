from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, low_memory=False)


def text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def num_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def bool_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    raw = frame[col].astype(str).str.lower()
    numeric = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return raw.isin(["true", "1", "1.0", "yes"]) | numeric.gt(0)


def max_drawdown(profits: pd.Series) -> float:
    if profits.empty:
        return 0.0
    equity = profits.cumsum()
    peak = equity.cummax()
    return float((equity - peak).min())


def race_pnl(detail: pd.DataFrame, payout_cap: float | None) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=["race_id", "tickets", "stake_yen", "return_yen", "profit_yen", "hit_tickets"])
    work = detail.copy()
    work["race_id"] = work["raceId"].astype(str) if "raceId" in work.columns else work.get("race_id", "").astype(str)
    work["stake_yen"] = num_series(work, "stakeYen")
    work["return_yen"] = num_series(work, "payoutYen")
    if payout_cap is not None:
        work["return_capped_yen"] = work["return_yen"].clip(upper=payout_cap)
    else:
        work["return_capped_yen"] = work["return_yen"]
    work["profit_yen"] = work["return_yen"] - work["stake_yen"]
    work["profit_capped_yen"] = work["return_capped_yen"] - work["stake_yen"]
    work["hit_bool"] = bool_series(work, "hit")
    work["start_sort"] = work.get("startTime", "").astype(str) if "startTime" in work.columns else ""
    grouped = (
        work.groupby("race_id", dropna=False)
        .agg(
            tickets=("race_id", "size"),
            stake_yen=("stake_yen", "sum"),
            return_yen=("return_yen", "sum"),
            return_capped_yen=("return_capped_yen", "sum"),
            profit_yen=("profit_yen", "sum"),
            profit_capped_yen=("profit_capped_yen", "sum"),
            hit_tickets=("hit_bool", "sum"),
            start_sort=("start_sort", "first"),
        )
        .reset_index()
    )
    grouped["race_hit"] = grouped["return_yen"].gt(0)
    return grouped


def promotion_score(snap: pd.DataFrame) -> pd.Series:
    score = (
        0.20 * num_series(snap, "strongest_current_score")
        + 0.12 * np.log1p(num_series(snap, "min_odds_margin_ratio")).clip(0, math.log1p(15)) / math.log1p(15)
        + 0.12 * np.log1p(num_series(snap, "runtime_expected_roi")).clip(0, math.log1p(20)) / math.log1p(20)
        + 0.12 * num_series(snap, "projected_front5_prob")
        + 0.10 * num_series(snap, "pair_quinella_score")
        + 0.08 * num_series(snap, "late_value_survives_score")
        + 0.05 * num_series(snap, "pace_fit_pair_score")
        + 0.04 * num_series(snap, "workout_pair_score")
        - 0.13 * num_series(snap, "skip_risk_score")
        - 0.10 * num_series(snap, "ticket_danger_popular_in_pair_score")
        - 0.08 * num_series(snap, "race_difficulty_score")
        - 0.08 * num_series(snap, "first_condition_pair_uncertainty_score")
        - 0.08 * num_series(snap, "track_state_uncertainty")
        - 0.025 * num_series(snap, "falloff_reason_count")
    )
    decision_bonus = snap.get("decision_key", pd.Series("", index=snap.index)).map(
        {"candidate": 0.18, "watch": 0.07, "weak": 0.0, "skip": -0.08, "buy": 0.25}
    ).fillna(0.0)
    single_gate_bonus = num_series(snap, "single_gate_failure") * 0.05
    return (score + decision_bonus + single_gate_bonus).astype(float)


def latest_per_race(snapshots: pd.DataFrame, decision_label: str) -> pd.DataFrame:
    work = snapshots.copy()
    if decision_label:
        work = work[work["decision_label"].astype(str).eq(decision_label)].copy()
    if work.empty:
        return work
    work["_captured_sort"] = pd.to_datetime(work.get("captured_at", ""), errors="coerce")
    work = work.sort_values(["race_id", "_captured_sort", "decision_snapshot_id"], na_position="last")
    return work.drop_duplicates("race_id", keep="last").drop(columns=["_captured_sort"], errors="ignore")


def metrics(rows: pd.DataFrame, label: str, pool_size: int) -> dict[str, Any]:
    if rows.empty:
        return {
            "label": label,
            "pool_races": pool_size,
            "selected_races": 0,
            "selected_tickets": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": None,
            "hit_races": 0,
            "race_hit_rate_pct": None,
            "max_drawdown_yen": 0.0,
        }
    stake = float(rows["stake_yen"].sum())
    ret = float(rows["return_yen"].sum())
    capped_ret = float(rows["return_capped_yen"].sum())
    profit = ret - stake
    capped_profit = capped_ret - stake
    hit_races = int(rows["race_hit"].sum())
    top_returns = rows["return_yen"].sort_values(ascending=False)
    excl1 = rows.copy()
    if not excl1.empty:
        excl1.loc[top_returns.index[:1], "return_yen"] = 0.0
    excl3 = rows.copy()
    if not excl3.empty:
        excl3.loc[top_returns.index[:3], "return_yen"] = 0.0
    top1_concentration = float(top_returns.iloc[0] / ret) if ret > 0 and len(top_returns) else 0.0
    return {
        "label": label,
        "pool_races": int(pool_size),
        "selected_races": int(rows["race_id"].nunique()),
        "selected_tickets": int(rows["tickets"].sum()),
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(profit, 1),
        "roi_pct": round(ret / stake * 100, 1) if stake > 0 else None,
        "hit_races": hit_races,
        "race_hit_rate_pct": round(hit_races / max(int(rows["race_id"].nunique()), 1) * 100, 1),
        "max_drawdown_yen": round(max_drawdown(rows.sort_values("start_sort")["profit_yen"]), 1),
        "capped_return_yen": round(capped_ret, 1),
        "capped_profit_yen": round(capped_profit, 1),
        "capped_roi_pct": round(capped_ret / stake * 100, 1) if stake > 0 else None,
        "roi_excluding_top1_hit_pct": round(excl1["return_yen"].sum() / stake * 100, 1) if stake > 0 else None,
        "roi_excluding_top3_hits_pct": round(excl3["return_yen"].sum() / stake * 100, 1) if stake > 0 else None,
        "top1_return_concentration_pct": round(top1_concentration * 100, 1),
        "avg_promotion_score": round(float(rows["promotion_score"].mean()), 4),
        "avg_margin": round(float(rows["min_odds_margin_ratio"].mean()), 2),
        "avg_expected_roi": round(float(rows["runtime_expected_roi"].mean()), 2),
        "avg_skip_risk": round(float(rows["skip_risk_score"].mean()), 2),
    }


def coverage_rows(pool: pd.DataFrame, pool_name: str, coverages: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ordered = pool.sort_values("promotion_score", ascending=False).reset_index(drop=True)
    for cov in coverages:
        n = max(1, int(math.ceil(len(ordered) * cov))) if len(ordered) else 0
        selected = ordered.head(n)
        out.append(metrics(selected, f"{pool_name}_top{int(cov * 100)}pct", len(ordered)))
    out.append(metrics(ordered, f"{pool_name}_all", len(ordered)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze risk-coverage for frozen decision snapshots.")
    parser.add_argument("--snapshots-csv", default="data/processed/live_decision_snapshots/current_strongest_decision_snapshots.csv")
    parser.add_argument("--pnl-detail-csv", default="outputs/analysis/current_live_pnl/current_live_pnl_detail.csv")
    parser.add_argument("--decision-label", default="")
    parser.add_argument("--output-dir", default="outputs/analysis/decision_snapshot_risk_coverage")
    parser.add_argument("--payout-cap-yen", type=float, default=2000.0)
    args = parser.parse_args()

    snapshots = read_csv_safe(project_path(args.snapshots_csv))
    detail = read_csv_safe(project_path(args.pnl_detail_csv))
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if snapshots.empty:
        raise SystemExit("No snapshots found.")
    snap = latest_per_race(snapshots, args.decision_label)
    if snap.empty:
        raise SystemExit(f"No snapshots found for decision label: {args.decision_label}")

    pnl = race_pnl(detail, payout_cap=args.payout_cap_yen)
    work = snap.merge(pnl, on="race_id", how="left")
    for col in ["tickets", "stake_yen", "return_yen", "return_capped_yen", "profit_yen", "profit_capped_yen", "hit_tickets"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce").fillna(0)
    work["race_hit"] = work.get("race_hit", False).fillna(False).astype(bool)
    work["start_sort"] = work.get("start_sort", "").fillna("")
    work["promotion_score"] = promotion_score(work)
    for col in ["min_odds_margin_ratio", "runtime_expected_roi", "skip_risk_score"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")

    pools = {
        "candidate_only": work[work["decision_key"].eq("candidate")].copy(),
        "candidate_watch": work[work["decision_key"].isin(["candidate", "watch"])].copy(),
        "shadow_non_skip": work[work["decision_key"].isin(["candidate", "watch", "weak"])].copy(),
        "single_gate": work[pd.to_numeric(work.get("single_gate_failure"), errors="coerce").fillna(0).gt(0)].copy(),
        "all_non_buy": work[~work["decision_key"].eq("buy")].copy(),
    }
    coverages = [0.05, 0.10, 0.20, 0.30, 0.50]
    rows: list[dict[str, Any]] = []
    for name, pool in pools.items():
        if pool.empty:
            continue
        rows.extend(coverage_rows(pool, name, coverages))
    coverage = pd.DataFrame(rows)

    reason_rows = []
    for _, row in work.iterrows():
        reasons = [r for r in text(row.get("falloff_reasons")).split("|") if r]
        for reason in reasons:
            reason_rows.append(
                {
                    "reason": reason,
                    "race_id": row.get("race_id"),
                    "decision_key": row.get("decision_key"),
                    "promotion_score": row.get("promotion_score"),
                    "stake_yen": row.get("stake_yen"),
                    "return_yen": row.get("return_yen"),
                    "profit_yen": row.get("profit_yen"),
                    "race_hit": row.get("race_hit"),
                }
            )
    reason = pd.DataFrame(reason_rows)
    if not reason.empty:
        reason_summary = (
            reason.groupby("reason")
            .agg(
                races=("race_id", "nunique"),
                avg_score=("promotion_score", "mean"),
                stake_yen=("stake_yen", "sum"),
                return_yen=("return_yen", "sum"),
                profit_yen=("profit_yen", "sum"),
                hit_races=("race_hit", "sum"),
            )
            .reset_index()
        )
        reason_summary["roi_pct"] = np.where(
            reason_summary["stake_yen"].gt(0),
            reason_summary["return_yen"] / reason_summary["stake_yen"] * 100,
            np.nan,
        )
        reason_summary = reason_summary.sort_values(["races", "roi_pct"], ascending=[False, False])
    else:
        reason_summary = pd.DataFrame()

    work_out = work.sort_values("promotion_score", ascending=False)
    work_out.to_csv(out_dir / "race_scores.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(out_dir / "coverage_summary.csv", index=False, encoding="utf-8-sig")
    reason_summary.to_csv(out_dir / "falloff_reason_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "decision_label": args.decision_label or "ALL_LABELS_LATEST_PER_RACE",
        "snapshot_races": int(len(work)),
        "pnl_races_joined": int(work["stake_yen"].gt(0).sum()),
        "output_dir": str(out_dir),
        "top_coverage_rows": coverage.head(20).to_dict(orient="records"),
        "warning": "Small-sample shadow analysis. Do not promote gates from this alone.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not coverage.empty:
        print(coverage.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
