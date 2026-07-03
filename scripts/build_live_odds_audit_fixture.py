from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


PAIR_TYPES = {"wide", "umaren", "umatan"}


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _live_pay_proxy(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    quote = _num(df.get("quote_pay_proxy_per100"), idx, np.nan)
    runtime = _num(df.get("runtime_pay_per100"), idx, np.nan)
    anchor_odds = _num(df.get("anchor_odds"), idx, np.nan) * 100.0
    return quote.where(quote.gt(0), runtime.where(runtime.gt(0), anchor_odds))


def build_pair_fixture(tickets: pd.DataFrame) -> pd.DataFrame:
    pair = tickets[tickets["ticket_type"].astype(str).isin(PAIR_TYPES)].copy()
    if pair.empty:
        return pd.DataFrame(columns=["race_id", "ticket_type", "a_no", "b_no", "live_pay_per100", "live_odds", "popularity", "snapshot_at"])
    a = _num(pair.get("anchor_no"), pair.index, np.nan)
    b = _num(pair.get("partner_no"), pair.index, np.nan)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    out = pd.DataFrame(
        {
            "race_id": pair["race_id"].astype(str),
            "ticket_type": pair["ticket_type"].astype(str),
            "a_no": lo.astype("Int64"),
            "b_no": hi.astype("Int64"),
            "live_pay_per100": _live_pay_proxy(pair),
            "popularity": pd.NA,
            "snapshot_at": "audit_fixture",
            "parser_mode": "audit_fixture_from_ticket_proxy",
        }
    )
    out["live_odds"] = out["live_pay_per100"] / 100.0
    out = out[out["a_no"].notna() & out["b_no"].notna() & out["live_pay_per100"].gt(0)].copy()
    return out.drop_duplicates(["race_id", "ticket_type", "a_no", "b_no"], keep="last")


def build_single_fixture(tickets: pd.DataFrame) -> pd.DataFrame:
    single = tickets[tickets["ticket_type"].astype(str).eq("win")].copy()
    if single.empty:
        return pd.DataFrame(
            columns=[
                "race_id",
                "horse_no",
                "live_win_odds",
                "live_popularity",
                "live_place_odds_min",
                "live_place_odds_max",
                "snapshot_at",
                "parser_mode",
            ]
        )
    pay = _live_pay_proxy(single)
    out = pd.DataFrame(
        {
            "race_id": single["race_id"].astype(str),
            "horse_no": _num(single.get("anchor_no"), single.index, np.nan).astype("Int64"),
            "live_win_odds": pay / 100.0,
            "live_popularity": pd.NA,
            "live_place_odds_min": pd.NA,
            "live_place_odds_max": pd.NA,
            "snapshot_at": "audit_fixture",
            "parser_mode": "audit_fixture_from_ticket_proxy",
        }
    )
    out = out[out["horse_no"].notna() & out["live_win_odds"].gt(0)].copy()
    return out.drop_duplicates(["race_id", "horse_no"], keep="last")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build manual live-odds CSV fixtures from ticket proxy odds to audit strict live-odds wiring. Not for ROI evaluation."
    )
    parser.add_argument("--tickets-csv", default="outputs/analysis/pair_probability_runtime_v1/pair_calibrated_runtime_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/live_odds_audit_fixture_v1")
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    out_dir = ensure_dir(project_path(args.output_dir))
    pair = build_pair_fixture(tickets)
    single = build_single_fixture(tickets)
    pair_csv = out_dir / "manual_pair_odds_fixture.csv"
    single_csv = out_dir / "manual_single_odds_fixture.csv"
    pair.to_csv(pair_csv, index=False, encoding="utf-8-sig")
    single.to_csv(single_csv, index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(out_dir),
        "tickets_csv": args.tickets_csv,
        "pair_csv": str(pair_csv),
        "single_csv": str(single_csv),
        "pair_rows": int(len(pair)),
        "single_rows": int(len(single)),
        "pair_ticket_types": pair["ticket_type"].value_counts().to_dict() if not pair.empty else {},
        "note": "Audit fixture only. It proves the strict live-odds path can consume live-like CSVs; it must not be used as real betting odds.",
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
