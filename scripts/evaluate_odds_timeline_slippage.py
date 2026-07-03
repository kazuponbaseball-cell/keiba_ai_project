from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False)


def _pair_ticket_frame(tickets: pd.DataFrame) -> pd.DataFrame:
    df = tickets[tickets["ticket_type"].astype(str).isin(["wide", "umaren"])].copy()
    if df.empty:
        return df
    anchor = _num(df.get("anchor_no"), df.index)
    partner = _num(df.get("partner_no"), df.index)
    df["a_no"] = np.minimum(anchor, partner).astype("Int64")
    df["b_no"] = np.maximum(anchor, partner).astype("Int64")
    df["final_pay_per100"] = np.where(
        df["ticket_type"].astype(str).eq("wide"),
        _num(df.get("wide_pay"), df.index, np.nan),
        _num(df.get("umaren_pay"), df.index, np.nan),
    )
    return df


def _single_ticket_frame(tickets: pd.DataFrame) -> pd.DataFrame:
    df = tickets[tickets["ticket_type"].astype(str).eq("win")].copy()
    if df.empty:
        return df
    df["horse_number"] = _num(df.get("anchor_no"), df.index).astype("Int64")
    df["final_pay_per100"] = _num(df.get("win_pay"), df.index, np.nan)
    if "final_pay_per100" in df.columns:
        df["final_pay_per100"] = df["final_pay_per100"].where(df.get("hit", False).astype(bool), np.nan)
    return df


def _evaluate_pair(tickets: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty or timeline.empty:
        return pd.DataFrame()
    t = _pair_ticket_frame(tickets)
    live = timeline.copy()
    for col in ["a_no", "b_no", "live_pay_per100"]:
        live[col] = _num(live.get(col), live.index)
    merged = t.merge(
        live,
        on=["race_id", "ticket_type", "a_no", "b_no"],
        how="inner",
        suffixes=("", "_live"),
    )
    if merged.empty:
        return merged
    merged["final_pay_per100"] = _num(merged.get("final_pay_per100"), merged.index)
    merged["live_pay_per100"] = _num(merged.get("live_pay_per100"), merged.index)
    merged["final_vs_live_ratio"] = merged["final_pay_per100"] / merged["live_pay_per100"].replace(0, np.nan)
    merged["live_to_final_drop_pct"] = 1.0 - merged["final_vs_live_ratio"]
    merged["effective_return_yen_if_live_quote"] = np.where(
        merged.get("hit", False).astype(bool),
        merged["live_pay_per100"] * _num(merged.get("stake_yen"), merged.index, 0.0) / 100.0,
        0.0,
    )
    return merged


def _evaluate_single(tickets: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty or timeline.empty:
        return pd.DataFrame()
    t = _single_ticket_frame(tickets)
    live = timeline.copy()
    live["horse_number"] = _num(live.get("horse_number"), live.index).astype("Int64")
    live["live_pay_per100"] = _num(live.get("win_odds"), live.index) * 100.0
    merged = t.merge(live, on=["race_id", "horse_number"], how="inner", suffixes=("", "_live"))
    if merged.empty:
        return merged
    merged["final_pay_per100"] = _num(merged.get("final_pay_per100"), merged.index)
    merged["final_vs_live_ratio"] = merged["final_pay_per100"] / merged["live_pay_per100"].replace(0, np.nan)
    merged["live_to_final_drop_pct"] = 1.0 - merged["final_vs_live_ratio"]
    merged["effective_return_yen_if_live_quote"] = np.where(
        merged.get("hit", False).astype(bool),
        merged["live_pay_per100"] * _num(merged.get("stake_yen"), merged.index, 0.0) / 100.0,
        0.0,
    )
    return merged


def _summary(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {
            "scope": label,
            "rows": 0,
            "races": 0,
            "hit_rows": 0,
            "avg_final_vs_live_ratio_hit": np.nan,
            "median_final_vs_live_ratio_hit": np.nan,
            "p10_final_vs_live_ratio_hit": np.nan,
            "p25_final_vs_live_ratio_hit": np.nan,
            "pct_hit_rows_final_below_live": np.nan,
            "pct_hit_rows_final_below_85pct_live": np.nan,
        }
    hit = frame[frame.get("hit", False).astype(bool)].copy()
    ratio = _num(hit.get("final_vs_live_ratio"), hit.index)
    return {
        "scope": label,
        "rows": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0,
        "hit_rows": int(len(hit)),
        "avg_final_vs_live_ratio_hit": float(ratio.mean()) if len(ratio.dropna()) else np.nan,
        "median_final_vs_live_ratio_hit": float(ratio.median()) if len(ratio.dropna()) else np.nan,
        "p10_final_vs_live_ratio_hit": float(ratio.quantile(0.10)) if len(ratio.dropna()) else np.nan,
        "p25_final_vs_live_ratio_hit": float(ratio.quantile(0.25)) if len(ratio.dropna()) else np.nan,
        "pct_hit_rows_final_below_live": float(ratio.lt(1.0).mean()) if len(ratio.dropna()) else np.nan,
        "pct_hit_rows_final_below_85pct_live": float(ratio.lt(0.85).mean()) if len(ratio.dropna()) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare captured live odds timeline against final payoffs for selected tickets.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/roi_mode_stake_sizing_v1/stake_sized_ticket_profiles.csv")
    parser.add_argument("--pair-timeline-csv", default="data/processed/live_odds/realtime_pair_odds_timeline.csv")
    parser.add_argument("--single-timeline-csv", default="data/processed/live_odds/realtime_single_odds_timeline.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/live_odds_slippage_v1")
    args = parser.parse_args()

    tickets = _read(project_path(args.tickets_csv))
    pair_timeline = _read(project_path(args.pair_timeline_csv))
    single_timeline = _read(project_path(args.single_timeline_csv))

    pair_eval = _evaluate_pair(tickets, pair_timeline)
    single_eval = _evaluate_single(tickets, single_timeline)
    all_eval = pd.concat([pair_eval, single_eval], ignore_index=True, sort=False) if not pair_eval.empty or not single_eval.empty else pd.DataFrame()

    out_dir = ensure_dir(project_path(args.output_dir))
    pair_eval.to_csv(out_dir / "pair_slippage_rows.csv", index=False, encoding="utf-8-sig")
    single_eval.to_csv(out_dir / "single_slippage_rows.csv", index=False, encoding="utf-8-sig")
    all_eval.to_csv(out_dir / "all_slippage_rows.csv", index=False, encoding="utf-8-sig")

    rows = [_summary(all_eval, "all"), _summary(pair_eval, "pair"), _summary(single_eval, "single")]
    if not all_eval.empty and "decision_label" in all_eval.columns:
        for label, g in all_eval.groupby("decision_label", dropna=False):
            rows.append(_summary(g, f"decision_{label}"))
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "slippage_summary.csv", index=False, encoding="utf-8-sig")

    payload = {
        "tickets_csv": args.tickets_csv,
        "pair_timeline_csv": args.pair_timeline_csv,
        "single_timeline_csv": args.single_timeline_csv,
        "output_dir": str(out_dir),
        "summary": rows,
        "note": "final_vs_live_ratio < 1.0 means final payout became worse than the captured live quote. No rows means live timeline has not matched selected tickets yet.",
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
