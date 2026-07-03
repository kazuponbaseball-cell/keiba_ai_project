from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_priority_a_non_day_factors import _metric, _num
from scripts.optimize_operational_win_addon import _json_default
from src.utils.paths import ensure_dir, project_path


def _round_stake(stake: pd.Series) -> pd.Series:
    return (np.floor(stake.clip(lower=0.0) / 100.0) * 100.0).clip(lower=100.0)


def _reprice(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pay = _num(
        out.get("runtime_backtest_pay_per100"),
        out.index,
        _num(out.get("quote_pay_proxy_per100"), out.index, 0.0),
    ).fillna(0.0)
    out["runtime_return_yen"] = np.where(out.get("hit", False).astype(bool), pay * out["runtime_stake_yen"] / 100.0, 0.0)
    return out


def apply_overlay(df: pd.DataFrame, mode: str, train_year: int, max_stake: float) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    if "year" not in out.columns:
        out["year"] = out["race_id"].astype(str).str[:4].astype(int)

    train = out[out["year"].eq(train_year)].copy()
    umaren_train = train[train["ticket_type"].eq("umaren")]
    wide_train = train[train["ticket_type"].eq("wide")]
    umaren_top = float(_num(umaren_train.get("a_priority_net_score"), umaren_train.index, np.nan).quantile(0.75))
    wide_low = float(_num(wide_train.get("a_priority_net_score"), wide_train.index, np.nan).quantile(0.25))

    out["pre_priority_a_stake_yen"] = _num(out.get("runtime_stake_yen"), out.index, 0.0).fillna(0.0)
    out["priority_a_ticket_overlay_action"] = "KEEP"
    stake = out["pre_priority_a_stake_yen"].copy()

    a_score = _num(out.get("a_priority_net_score"), out.index, 0.0).fillna(0.0)
    boost_umaren = out["ticket_type"].eq("umaren") & a_score.ge(umaren_top) & stake.gt(0)
    reduce_wide = out["ticket_type"].eq("wide") & a_score.lt(wide_low) & stake.gt(0)

    if mode in {"boost_umaren_a_top_115", "both"}:
        stake.loc[boost_umaren] = (stake.loc[boost_umaren] * 1.15).clip(upper=max_stake)
        out.loc[boost_umaren, "priority_a_ticket_overlay_action"] = "BOOST_UMAREN_A_TOP"
    if mode in {"reduce_wide_a_low_50", "both"}:
        stake.loc[reduce_wide] = stake.loc[reduce_wide] * 0.50
        out.loc[reduce_wide, "priority_a_ticket_overlay_action"] = np.where(
            out.loc[reduce_wide, "priority_a_ticket_overlay_action"].eq("KEEP"),
            "REDUCE_WIDE_A_LOW",
            out.loc[reduce_wide, "priority_a_ticket_overlay_action"] + "+REDUCE_WIDE_A_LOW",
        )

    out["runtime_stake_yen"] = _round_stake(stake).where(stake.gt(0), 0.0)
    out["runtime_reason"] = out.get("runtime_reason", "").astype(str) + "|priority_a_ticket_type_overlay:" + mode
    out = _reprice(out)
    meta = {
        "mode": mode,
        "train_year": train_year,
        "max_stake": max_stake,
        "umaren_top_a_threshold": umaren_top,
        "wide_low_a_threshold": wide_low,
        "boost_umaren_count": int(boost_umaren.sum()),
        "reduce_wide_count": int(reduce_wide.sum()),
    }
    return out, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ticket-type overlay for priority-A factors.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/priority_a_non_day_factors_v1/a_priority_enriched_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/priority_a_ticket_type_overlay_v1")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--mode", choices=["boost_umaren_a_top_115", "reduce_wide_a_low_50", "both"], default="boost_umaren_a_top_115")
    parser.add_argument("--max-stake", type=float, default=3000.0)
    args = parser.parse_args()

    df = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    base_metrics = [_metric(df, "base_all")]
    for year, g in df.groupby("year"):
        base_metrics.append(_metric(g, f"base_{int(year)}"))

    overlaid, meta = apply_overlay(df, args.mode, args.train_year, args.max_stake)
    overlay_metrics = [_metric(overlaid, f"{args.mode}_all")]
    for year, g in overlaid.groupby("year"):
        overlay_metrics.append(_metric(g, f"{args.mode}_{int(year)}"))

    out_dir = ensure_dir(project_path(args.output_dir))
    overlaid.to_csv(out_dir / "priority_a_ticket_type_overlaid_tickets.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(base_metrics + overlay_metrics).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"input": args.tickets_csv, "output_dir": str(out_dir), **meta}, f, ensure_ascii=False, indent=2, default=_json_default)

    print(json.dumps({"summary": meta, "base_all": base_metrics[0], "overlay_all": overlay_metrics[0]}, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
