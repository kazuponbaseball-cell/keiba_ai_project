from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_high_pressure_front_survival_context import (
    DEFAULT_OUT as DEFAULT_CONTEXT_OUT,
    DEFAULT_PACE,
    DEFAULT_RUNNERS,
    DEFAULT_TICKETS,
    build_race_table,
    build_ticket_context,
    make_policy_masks,
    norm_race_id,
    read_csv,
    rolling_context_priors,
)


DEFAULT_OUT = ROOT / "outputs/analysis/high_pressure_front_survival_context_v1/mcs_pbo_v1"


FOCUS_POLICIES = [
    ("wide", "base_all"),
    ("wide", "pressure60_pair_front_any"),
    ("wide", "pressure70_pair_front_any"),
    ("wide", "pressure60_survival70_pair_front_any"),
    ("wide", "pressure60_avoid_collapse80_front_any"),
    ("wide", "pressure60_front_complement"),
    ("wide", "mixed_queue_front_any"),
    ("umaren", "base_all"),
    ("umaren", "pressure60_survival70_pair_front_any"),
    ("umaren", "pressure60_survival70_pair_front_both"),
    ("umaren", "pressure60_avoid_collapse80_front_any"),
]


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def metrics(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": np.nan,
            "hit_rate_pct": np.nan,
            "roi_ex_top1_pct": np.nan,
            "roi_ex_top5_pct": np.nan,
            "top_return_share_pct": np.nan,
            "max_drawdown_yen": np.nan,
        }
    stake = pd.to_numeric(frame["stake_yen"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(frame["return_yen"], errors="coerce").fillna(0.0)
    profit = ret - stake
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    ordered = ret.sort_values(ascending=False)
    top1_idx = ordered.index[:1]
    top5_idx = ordered.index[:5]
    equity = profit.cumsum()
    drawdown = (equity.cummax() - equity).max() if len(equity) else 0.0
    return {
        "policy": label,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": stake_sum,
        "return_yen": ret_sum,
        "profit_yen": ret_sum - stake_sum,
        "roi_pct": safe_div(ret_sum, stake_sum) * 100,
        "hit_rate_pct": float(ret.gt(0).mean() * 100),
        "roi_ex_top1_pct": safe_div(ret_sum - float(ret.loc[top1_idx].sum()), stake_sum - float(stake.loc[top1_idx].sum())) * 100,
        "roi_ex_top5_pct": safe_div(ret_sum - float(ret.loc[top5_idx].sum()), stake_sum - float(stake.loc[top5_idx].sum())) * 100,
        "top_return_share_pct": safe_div(float(ordered.iloc[0]) if len(ordered) else 0.0, ret_sum) * 100,
        "max_drawdown_yen": float(drawdown),
    }


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


def load_focus_rows(runners_path: Path, tickets_path: Path, pace_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runner_cols = {
        "race_id",
        "芝・ダ",
        "距離",
        "クラス名",
        "馬場状態",
        "queue_type",
        "race_est_ten_pressure_score",
        "race_est_fast_start_count",
        "race_est_ten_speed_gap_top2",
        "race_est_queue_clarity_score",
        "actual_front5",
        "target_top3",
        "target_win",
    }
    pace_cols = {
        "race_id",
        "venue_code",
        "race_no",
        "actual_lap_mode",
        "cont_predicted_lap_mode",
        "cont_confidence",
        "cont_margin",
        "front3f_sec",
        "last3f_sec",
        "rpci",
        "pci3",
        "course_front3f_prior_sec",
        "course_front3f_prior_std",
        "course_front3f_prior_count",
        "pred_front3f_sec",
        "pred_rpci",
        "pred_pci3",
        "cont_front_delta_z",
        "cont_rpci_delta",
    }
    ticket_cols = {
        "race_id",
        "year",
        "ticket_type",
        "stake_yen",
        "return_yen",
        "pair_pred_front5_any",
        "pair_pred_front5_both",
        "pair_pred_leader_any",
        "pair_pred_front_complement",
        "queue_type",
        "queue_shape_label",
    }
    runners = read_csv(runners_path, usecols=lambda c: c in runner_cols)
    pace = read_csv(pace_path, usecols=lambda c: c in pace_cols)
    tickets = read_csv(tickets_path, usecols=lambda c: c in ticket_cols)
    races = rolling_context_priors(build_race_table(runners, pace))
    t, thresholds = build_ticket_context(tickets, races)
    masks = make_policy_masks(t, thresholds)
    t["race_id"] = norm_race_id(t["race_id"])
    t["_date"] = pd.to_datetime(t["race_id"].str.slice(0, 8), format="%Y%m%d", errors="coerce")
    t["_month"] = t["_date"].dt.to_period("M").astype(str)
    t = t[t["_date"].notna()].copy()

    rows = []
    for ticket_type, policy_name in FOCUS_POLICIES:
        mask = t["ticket_type"].eq(ticket_type) & masks[policy_name].fillna(False)
        sub = t.loc[mask, ["race_id", "year", "_date", "_month", "ticket_type", "stake_yen", "return_yen"]].copy()
        sub["policy"] = f"{ticket_type}::{policy_name}"
        rows.append(sub)
    selected = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    selected["profit_yen"] = pd.to_numeric(selected["return_yen"], errors="coerce").fillna(0.0) - pd.to_numeric(selected["stake_yen"], errors="coerce").fillna(0.0)
    selected = selected.sort_values(["_date", "race_id", "policy"], kind="mergesort").reset_index(drop=True)
    return selected, races


def overall_and_year(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame([metrics(g, p) for p, g in selected.groupby("policy", sort=False)])
    years = []
    for policy, gp in selected.groupby("policy", sort=False):
        for year, gy in gp.groupby("year", sort=True):
            row = metrics(gy, policy)
            row["year"] = int(year)
            years.append(row)
    return overall, pd.DataFrame(years)


def walk_forward(selected: pd.DataFrame, min_train_tickets: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(selected["_month"].dropna().unique())
    rows = []
    for idx in range(6, len(months)):
        train_months = months[: idx - 1]
        test_month = months[idx]
        train = selected[selected["_month"].isin(train_months)]
        test_pool = selected[selected["_month"].eq(test_month)]
        scores = []
        for policy, group in train.groupby("policy"):
            m = metrics(group, policy)
            if m["tickets"] < min_train_tickets:
                continue
            score = m["roi_ex_top5_pct"] + 0.20 * m["hit_rate_pct"] - max(0.0, m["top_return_share_pct"] - 3.0)
            scores.append((score, m["roi_ex_top5_pct"], m["roi_pct"], policy))
        if not scores:
            continue
        scores.sort(reverse=True)
        chosen = scores[0][-1]
        test = test_pool[test_pool["policy"].eq(chosen)]
        test_m = metrics(test, chosen)
        test_m["test_month"] = test_month
        test_m["chosen_policy"] = chosen
        rows.append(test_m)
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    stake = float(detail["stake_yen"].sum())
    ret = float(detail["return_yen"].sum())
    summary = pd.DataFrame(
        [
            {
                "months": int(len(detail)),
                "tickets": int(detail["tickets"].sum()),
                "races": int(detail["races"].sum()),
                "stake_yen": stake,
                "return_yen": ret,
                "profit_yen": ret - stake,
                "roi_pct": safe_div(ret, stake) * 100,
                "positive_month_rate_pct": float(detail["profit_yen"].gt(0).mean() * 100),
                "chosen_policy_counts": json.dumps(detail["chosen_policy"].value_counts().to_dict(), ensure_ascii=False),
            }
        ]
    )
    return detail, summary


def assign_blocks(selected: pd.DataFrame, n_blocks: int = 8) -> pd.Series:
    dates = selected[["_date"]].drop_duplicates().sort_values("_date").reset_index(drop=True)
    dates["block"] = pd.qcut(dates.index, q=n_blocks, labels=False, duplicates="drop")
    return selected["_date"].map(dates.set_index("_date")["block"]).astype(int)


def pbo(selected: pd.DataFrame, min_train_tickets: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = selected.copy()
    work["_block"] = assign_blocks(work, 8)
    blocks = sorted(work["_block"].unique())
    rows = []
    for split_id, test_blocks in enumerate(itertools.combinations(blocks, len(blocks) // 2), start=1):
        test_blocks = set(test_blocks)
        train = work[~work["_block"].isin(test_blocks)]
        test = work[work["_block"].isin(test_blocks)]
        train_scores = []
        test_scores = []
        for policy, group in train.groupby("policy"):
            m = metrics(group, policy)
            if m["tickets"] < min_train_tickets:
                continue
            score = m["roi_ex_top5_pct"] + 0.20 * m["hit_rate_pct"] - max(0.0, m["top_return_share_pct"] - 3.0)
            train_scores.append((score, policy))
        if not train_scores:
            continue
        for policy, group in test.groupby("policy"):
            m = metrics(group, policy)
            test_scores.append((m["roi_ex_top5_pct"], policy))
        if not test_scores:
            continue
        train_scores.sort(reverse=True)
        chosen = train_scores[0][1]
        ranked_test = pd.DataFrame(test_scores, columns=["test_score", "policy"]).sort_values("test_score")
        rank_map = {p: i / max(len(ranked_test) - 1, 1) for i, p in enumerate(ranked_test["policy"])}
        chosen_test = metrics(test[test["policy"].eq(chosen)], chosen)
        rows.append(
            {
                "split_id": split_id,
                "chosen_policy": chosen,
                "test_percentile": rank_map.get(chosen, np.nan),
                "test_roi_pct": chosen_test["roi_pct"],
                "test_roi_ex_top5_pct": chosen_test["roi_ex_top5_pct"],
                "test_tickets": chosen_test["tickets"],
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = pd.DataFrame(
        [
            {
                "splits": int(len(detail)),
                "pbo_below_median_rate": float(detail["test_percentile"].lt(0.5).mean()),
                "median_test_percentile": float(detail["test_percentile"].median()),
                "median_test_roi_pct": float(detail["test_roi_pct"].median()),
                "median_test_roi_ex_top5_pct": float(detail["test_roi_ex_top5_pct"].median()),
                "chosen_policy_counts": json.dumps(detail["chosen_policy"].value_counts().to_dict(), ensure_ascii=False),
            }
        ]
    )
    return detail, summary


def bootstrap_by_month(selected: pd.DataFrame, iterations: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    monthly = (
        selected.groupby(["policy", "_month"], dropna=False)
        .agg(stake_yen=("stake_yen", "sum"), return_yen=("return_yen", "sum"))
        .reset_index()
    )
    months = sorted(monthly["_month"].unique())
    for policy, group in monthly.groupby("policy"):
        g = group.set_index("_month").reindex(months).fillna(0.0)
        stakes = g["stake_yen"].to_numpy(dtype=float)
        returns = g["return_yen"].to_numpy(dtype=float)
        rois = []
        for _ in range(iterations):
            idx = rng.integers(0, len(months), len(months))
            st = float(stakes[idx].sum())
            rt = float(returns[idx].sum())
            if st > 0:
                rois.append(rt / st * 100)
        arr = np.array(rois, dtype=float)
        rows.append(
            {
                "policy": policy,
                "months": int(len(months)),
                "bootstrap_roi_p05": float(np.percentile(arr, 5)) if len(arr) else np.nan,
                "bootstrap_roi_p50": float(np.percentile(arr, 50)) if len(arr) else np.nan,
                "bootstrap_roi_p95": float(np.percentile(arr, 95)) if len(arr) else np.nan,
                "bootstrap_prob_roi_gt_100": float((arr > 100).mean()) if len(arr) else np.nan,
                "bootstrap_prob_roi_gt_120": float((arr > 120).mean()) if len(arr) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["bootstrap_prob_roi_gt_100", "bootstrap_roi_p50"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runners", type=Path, default=DEFAULT_RUNNERS)
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    parser.add_argument("--pace", type=Path, default=DEFAULT_PACE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected, races = load_focus_rows(args.runners, args.tickets, args.pace)
    overall, by_year = overall_and_year(selected)
    wf_detail, wf_summary = walk_forward(selected)
    pbo_detail, pbo_summary = pbo(selected)
    boot = bootstrap_by_month(selected)

    selected.to_csv(args.out_dir / "focused_policy_selected_tickets.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(args.out_dir / "focused_policy_overall_metrics.csv", index=False, encoding="utf-8-sig")
    by_year.to_csv(args.out_dir / "focused_policy_year_metrics.csv", index=False, encoding="utf-8-sig")
    wf_detail.to_csv(args.out_dir / "walk_forward_detail.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(args.out_dir / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")
    pbo_detail.to_csv(args.out_dir / "pbo_detail.csv", index=False, encoding="utf-8-sig")
    pbo_summary.to_csv(args.out_dir / "pbo_summary.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(args.out_dir / "monthly_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    key_policy = "wide::pressure60_survival70_pair_front_any"
    payload = {
        "output_dir": str(args.out_dir.relative_to(ROOT)),
        "selected_rows": int(len(selected)),
        "focused_policies": [f"{t}::{p}" for t, p in FOCUS_POLICIES],
        "key_policy": key_policy,
        "key_policy_overall": overall[overall["policy"].eq(key_policy)].to_dict(orient="records"),
        "key_policy_by_year": by_year[by_year["policy"].eq(key_policy)].to_dict(orient="records"),
        "key_policy_bootstrap": boot[boot["policy"].eq(key_policy)].to_dict(orient="records"),
        "walk_forward_summary": wf_summary.to_dict(orient="records"),
        "pbo_summary": pbo_summary.to_dict(orient="records"),
        "best_bootstrap": boot.head(5).to_dict(orient="records"),
        "note": (
            "This is a focused robustness check for high-pressure front-survival policies. "
            "It is intentionally conservative: policies are fixed before validation, and the key policy remains a watchlist gate unless live shadow accumulation confirms it."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
