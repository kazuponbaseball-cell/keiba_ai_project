from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series | None, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _ticket_pair_key(df: pd.DataFrame) -> pd.Series:
    a = pd.to_numeric(df["anchor_no"], errors="coerce").astype("Int64").astype(str)
    b = pd.to_numeric(df["partner_no"], errors="coerce").astype("Int64").astype(str)
    lo = np.minimum(pd.to_numeric(df["anchor_no"], errors="coerce"), pd.to_numeric(df["partner_no"], errors="coerce"))
    hi = np.maximum(pd.to_numeric(df["anchor_no"], errors="coerce"), pd.to_numeric(df["partner_no"], errors="coerce"))
    return df["race_id"].astype(str) + ":" + lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)


def _normalize_live_odds(path: Path) -> pd.DataFrame:
    live = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
    rename = {
        "bet": "ticket_type",
        "type": "ticket_type",
        "horse_a": "a_no",
        "horse_b": "b_no",
        "umaban_a": "a_no",
        "umaban_b": "b_no",
        "pay": "live_pay_per100",
        "odds": "live_odds",
    }
    live = live.rename(columns={k: v for k, v in rename.items() if k in live.columns})
    required = {"race_id", "ticket_type", "a_no", "b_no"}
    missing = sorted(required - set(live.columns))
    if missing:
        raise ValueError(f"live odds csv missing columns: {missing}")
    if "live_pay_per100" not in live.columns:
        if "live_odds" not in live.columns:
            raise ValueError("live odds csv needs live_pay_per100 or live_odds")
        live["live_pay_per100"] = pd.to_numeric(live["live_odds"], errors="coerce") * 100.0
    live["a_no"] = pd.to_numeric(live["a_no"], errors="coerce")
    live["b_no"] = pd.to_numeric(live["b_no"], errors="coerce")
    lo = np.minimum(live["a_no"], live["b_no"])
    hi = np.maximum(live["a_no"], live["b_no"])
    live["pair_lookup_key"] = live["race_id"].astype(str) + ":" + lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)
    live["ticket_type"] = live["ticket_type"].astype(str)
    return live[["race_id", "ticket_type", "pair_lookup_key", "live_pay_per100"]].copy()


def _historical_pay_col(df: pd.DataFrame) -> pd.Series:
    return np.select(
        [df["ticket_type"].eq("wide"), df["ticket_type"].eq("umaren")],
        [_num(df.get("wide_pay"), 0.0), _num(df.get("umaren_pay"), 0.0)],
        default=0.0,
    )


def _apply_gate(tickets: pd.DataFrame, live: pd.DataFrame | None, margin: float, allow_missing_live: bool) -> pd.DataFrame:
    out = tickets.copy()
    out["pair_lookup_key"] = _ticket_pair_key(out)
    out["historical_pay_per100"] = _historical_pay_col(out)
    if live is not None:
        out = out.merge(live, on=["race_id", "ticket_type", "pair_lookup_key"], how="left")
    else:
        out["live_pay_per100"] = np.nan

    out["effective_pay_per100"] = _num(out["live_pay_per100"]).where(_num(out["live_pay_per100"]).gt(0), out["historical_pay_per100"])
    out["pay_source"] = np.where(_num(out["live_pay_per100"]).gt(0), "live", "historical_proxy")

    # These are payout floors, not result-derived expectations. In live use, the same columns are checked against JV/Data Lab odds.
    wide_floor = np.where(_num(out["pair_score"]).ge(0.78), 280.0, 350.0)
    umaren_floor = np.where(_num(out["pair_quinella_score"]).ge(0.66), 1000.0, 1200.0)
    out["required_pay_per100"] = np.where(out["ticket_type"].eq("wide"), wide_floor, umaren_floor) * margin
    has_live = _num(out["live_pay_per100"]).gt(0)
    out["live_available"] = has_live
    out["odds_gate_pass"] = out["effective_pay_per100"].ge(out["required_pay_per100"]) & (has_live | allow_missing_live)
    gated = out[out["odds_gate_pass"]].copy()
    gated["return_yen"] = np.where(gated["hit"].astype(bool), gated["effective_pay_per100"], 0.0) * _num(gated["stake_yen"]) / 100.0
    return gated


def _metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"policy": label, "tickets": 0, "races": 0, "roi": 0.0}
    stake = float(_num(df["stake_yen"]).sum())
    ret = float(_num(df["return_yen"]).sum())
    by_race = df.groupby("race_id").agg(stake=("stake_yen", "sum"), ret=("return_yen", "sum"), hit=("hit", "max")).sort_index()
    equity = (by_race["ret"] - by_race["stake"]).cumsum()
    dd = equity - equity.cummax()
    rows = {
        "policy": label,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "race_hit_rate": float(by_race["hit"].mean()),
        "ticket_hit_rate": float(df["hit"].mean()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "max_drawdown_yen": float(dd.min()) if not dd.empty else 0.0,
        "live_ticket_rate": float(df["live_available"].mean()) if "live_available" in df else 0.0,
    }
    for ticket_type, g in df.groupby("ticket_type"):
        st = float(_num(g["stake_yen"]).sum())
        rows[f"{ticket_type}_tickets"] = int(len(g))
        rows[f"{ticket_type}_roi"] = float(_num(g["return_yen"]).sum() / st) if st else 0.0
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply live odds payout gates to priority-S selected tickets.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/priority_s_betting_policy_ticket_choice_v1/walkforward_selected_tickets.csv")
    parser.add_argument("--live-odds-csv", default=None, help="Optional normalized live odds csv: race_id,ticket_type,a_no,b_no,live_pay_per100")
    parser.add_argument("--output-dir", default="outputs/analysis/priority_s_live_odds_gate_v1")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--allow-missing-live", action="store_true", help="Use historical payoffs as a proxy when live odds are missing.")
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    live = _normalize_live_odds(project_path(args.live_odds_csv)) if args.live_odds_csv else None
    gated = _apply_gate(tickets, live, args.margin, args.allow_missing_live)

    out_dir = ensure_dir(project_path(args.output_dir))
    gated.to_csv(out_dir / "live_odds_gated_tickets.csv", index=False, encoding="utf-8-sig")
    yearly = []
    if "test_year" in gated.columns:
        for year, g in gated.groupby("test_year"):
            row = _metrics(g, f"year_{year}")
            row["year"] = int(year)
            yearly.append(row)
    yearly_df = pd.DataFrame(yearly)
    yearly_df.to_csv(out_dir / "live_odds_gated_yearly.csv", index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(out_dir),
        "margin": args.margin,
        "allow_missing_live": args.allow_missing_live,
        "summary": _metrics(gated, "live_odds_gated_total"),
        "yearly": yearly_df.to_dict(orient="records"),
        "live_odds_csv": args.live_odds_csv,
        "note": "In production, pass JV/Data Lab normalized wide/umaren odds. Without --allow-missing-live, tickets without live odds are skipped.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
