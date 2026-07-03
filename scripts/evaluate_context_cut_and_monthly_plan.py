from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_operational_win_addon import _json_default, _metrics
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _ticket_key(df: pd.DataFrame) -> pd.Series:
    parts = [
        df["race_id"].astype(str),
        df.get("ticket_type", "").astype(str),
        _num(df.get("anchor_no"), df.index, -1).fillna(-1).astype(int).astype(str),
        _num(df.get("partner_no"), df.index, -1).fillna(-1).astype(int).astype(str),
        _num(df.get("third_no"), df.index, -1).fillna(-1).astype(int).astype(str) if "third_no" in df.columns else pd.Series("-1", index=df.index),
    ]
    return parts[0] + ":" + parts[1] + ":" + parts[2] + ":" + parts[3] + ":" + parts[4]


def _runtime_metric(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"policy": label, "tickets": 0, "races": 0, "roi": 0.0}
    tmp = df.copy()
    tmp["stake_yen"] = _num(tmp.get("runtime_stake_yen"), tmp.index, 0.0).fillna(0.0)
    tmp["return_yen"] = _num(tmp.get("runtime_return_yen"), tmp.index, 0.0).fillna(0.0)
    return _metrics(tmp, label)


def _scale_tickets(df: pd.DataFrame, multiplier: float, race_cap: float, day_cap: float) -> pd.DataFrame:
    out = df.copy()
    out["date_key"] = _date_key(out)
    base_stake = _num(out.get("runtime_stake_yen"), out.index, 0.0).fillna(0.0)
    out["scaled_stake_yen"] = (np.floor((base_stake * multiplier).clip(lower=0.0) / 100.0) * 100.0).clip(lower=0.0)
    out["scaled_stake_yen"] = np.minimum(out["scaled_stake_yen"], race_cap)

    keep_frames: list[pd.DataFrame] = []
    for _, day in out.sort_values(["date_key", "race_id", "runtime_odds_margin_ratio"], ascending=[True, True, False]).groupby("date_key", sort=False):
        running = 0.0
        rows = []
        for idx, row in day.iterrows():
            stake = float(row["scaled_stake_yen"])
            if stake <= 0:
                continue
            if running + stake > day_cap:
                continue
            running += stake
            rows.append(idx)
        keep_frames.append(day.loc[rows])
    out = pd.concat(keep_frames, ignore_index=False, sort=False) if keep_frames else out.iloc[0:0].copy()
    pay = _num(out.get("runtime_backtest_pay_per100"), out.index, 0.0).fillna(0.0)
    out["scaled_return_yen"] = np.where(out["hit"].astype(bool), pay * out["scaled_stake_yen"] / 100.0, 0.0)
    return out


def _date_key(df: pd.DataFrame) -> pd.Series:
    if "日付S" in df.columns:
        text = df["日付S"].astype(str)
    elif "date" in df.columns:
        text = df["date"].astype(str)
    else:
        text = df["race_id"].astype(str).str.slice(0, 8)
    parsed = pd.to_datetime(text, errors="coerce")
    fallback = pd.to_datetime(df["race_id"].astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce")
    parsed = parsed.fillna(fallback)
    return parsed.dt.strftime("%Y-%m-%d").fillna(df["race_id"].astype(str).str.slice(0, 8))


def _month_key(date_key: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(date_key, errors="coerce")
    return parsed.dt.strftime("%Y-%m").fillna(date_key.astype(str).str.slice(0, 7))


def _scaled_metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"policy": label, "tickets": 0, "races": 0, "roi": 0.0}
    tmp = df.copy()
    tmp["stake_yen"] = _num(tmp.get("scaled_stake_yen"), tmp.index, 0.0).fillna(0.0)
    tmp["return_yen"] = _num(tmp.get("scaled_return_yen"), tmp.index, 0.0).fillna(0.0)
    return _metrics(tmp, label)


def _monthly_report(df: pd.DataFrame, multiplier: float, race_cap: float, day_cap: float) -> tuple[pd.DataFrame, dict]:
    scaled = _scale_tickets(df, multiplier, race_cap, day_cap)
    if scaled.empty:
        return pd.DataFrame(), _scaled_metrics(scaled, f"x{multiplier}")
    scaled["date_key"] = _date_key(scaled)
    scaled["month"] = _month_key(scaled["date_key"])
    rows = []
    for month, g in scaled.groupby("month"):
        stake = float(g["scaled_stake_yen"].sum())
        ret = float(g["scaled_return_yen"].sum())
        by_day = g.groupby("date_key").agg(stake=("scaled_stake_yen", "sum"), ret=("scaled_return_yen", "sum"))
        by_day["profit"] = by_day["ret"] - by_day["stake"]
        equity = by_day["profit"].cumsum()
        dd = equity - equity.cummax()
        rows.append(
            {
                "month": month,
                "days": int(g["date_key"].nunique()),
                "races": int(g["race_id"].nunique()),
                "tickets": int(len(g)),
                "stake_yen": stake,
                "return_yen": ret,
                "profit_yen": ret - stake,
                "roi": ret / stake if stake else 0.0,
                "max_day_loss_yen": float(by_day["profit"].min()) if not by_day.empty else 0.0,
                "month_max_drawdown_yen": float(dd.min()) if not dd.empty else 0.0,
            }
        )
    return pd.DataFrame(rows), _scaled_metrics(scaled, f"x{multiplier}_racecap{race_cap}_daycap{day_cap}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate context-cut tickets and monthly staking plan.")
    parser.add_argument("--base-csv", default="outputs/analysis/runtime_odds_decision_rules_v1/runtime_selected_tickets.csv")
    parser.add_argument("--selected-csv", default="outputs/analysis/priority_context_factor_overlay_v1/priority_context_selected_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/context_cut_monthly_plan_v1")
    args = parser.parse_args()

    base = pd.read_csv(project_path(args.base_csv), dtype={"race_id": str}, low_memory=False)
    selected = pd.read_csv(project_path(args.selected_csv), dtype={"race_id": str}, low_memory=False)
    base["_ticket_key"] = _ticket_key(base)
    selected["_ticket_key"] = _ticket_key(selected)
    cut = base[~base["_ticket_key"].isin(set(selected["_ticket_key"]))].copy()
    kept = selected.copy()

    out_dir = ensure_dir(project_path(args.output_dir))
    cut.to_csv(out_dir / "context_cut_tickets.csv", index=False, encoding="utf-8-sig")
    kept.to_csv(out_dir / "context_kept_tickets.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(
        [
            _runtime_metric(base, "base_before_context"),
            _runtime_metric(kept, "kept_after_context"),
            _runtime_metric(cut, "cut_by_context"),
        ]
    )
    metrics.to_csv(out_dir / "cut_vs_kept_metrics.csv", index=False, encoding="utf-8-sig")

    by_type = []
    for name, frame in [("kept", kept), ("cut", cut)]:
        for ticket_type, g in frame.groupby("ticket_type"):
            row = _runtime_metric(g, f"{name}_{ticket_type}")
            row.update({"group": name, "ticket_type": ticket_type})
            by_type.append(row)
    pd.DataFrame(by_type).to_csv(out_dir / "cut_vs_kept_by_ticket_type.csv", index=False, encoding="utf-8-sig")

    monthly_rows = []
    overall_rows = []
    for multiplier in [1, 2, 3, 4, 5]:
        monthly, overall = _monthly_report(kept, multiplier=multiplier, race_cap=3000.0, day_cap=8000.0)
        monthly["multiplier"] = multiplier
        monthly_rows.append(monthly)
        overall["multiplier"] = multiplier
        overall_rows.append(overall)
    monthly_df = pd.concat(monthly_rows, ignore_index=True, sort=False) if monthly_rows else pd.DataFrame()
    overall_df = pd.DataFrame(overall_rows)
    monthly_df.to_csv(out_dir / "monthly_scaled_results.csv", index=False, encoding="utf-8-sig")
    overall_df.to_csv(out_dir / "scaled_overall_metrics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "base_tickets": int(len(base)),
        "kept_tickets": int(len(kept)),
        "cut_tickets": int(len(cut)),
        "cut_metrics": _runtime_metric(cut, "cut_by_context"),
        "kept_metrics": _runtime_metric(kept, "kept_after_context"),
        "monthly_plan_note": "Scaled results use race cap 3000 yen and day cap 8000 yen.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
