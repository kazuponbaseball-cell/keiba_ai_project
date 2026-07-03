from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic pseudo T-5/T-3 pair odds timeline for wiring tests.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/mcs_pbo_runtime_overlay_v3/mcs_full_margin095_s0304_selected_tickets.csv")
    parser.add_argument("--output-csv", default="outputs/analysis/fixed_time_pair_edge_fixture_v1/pseudo_pair_odds_timeline.csv")
    parser.add_argument("--max-rows", type=int, default=0, help="0 keeps all selected umaren tickets.")
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    tickets = tickets[tickets.get("ticket_type", "").astype(str).eq("umaren")].copy()
    tickets = tickets[num(tickets, "runtime_stake_yen", 0.0).fillna(0.0).gt(0)].copy()
    if args.max_rows > 0:
        tickets = tickets.head(args.max_rows).copy()

    a = num(tickets, "horse_a").fillna(num(tickets, "anchor_no"))
    b = num(tickets, "horse_b").fillna(num(tickets, "partner_no"))
    lo = np.minimum(a, b).astype(int)
    hi = np.maximum(a, b).astype(int)
    pay = num(tickets, "quote_pay_proxy_per100").fillna(num(tickets, "runtime_pay_per100")).fillna(num(tickets, "umaren_pay"))
    pay = pay.clip(lower=110.0)

    rows = []
    for i, (_, row) in enumerate(tickets.iterrows()):
        base_pay = float(pay.loc[row.name])
        # Stable deterministic movement: T-3 is sometimes shorter and
        # sometimes wider than T-5, so the validation can test both paths.
        drift = 0.92 + (i % 9) * 0.02
        for label, mult in [("T-5", 1.00), ("T-3", drift)]:
            live_pay = round(base_pay * mult, 2)
            rows.append(
                {
                    "race_id": str(row["race_id"]),
                    "ticket_type": "umaren",
                    "a_no": int(lo.loc[row.name]),
                    "b_no": int(hi.loc[row.name]),
                    "live_pay_per100": live_pay,
                    "live_odds": live_pay / 100.0,
                    "popularity": pd.NA,
                    "snapshot_at": f"fixture_{label}",
                    "decision_label": label,
                    "captured_at": f"20260619_{120000 + i:06d}",
                    "parser_mode": "fixture",
                }
            )
    out = pd.DataFrame(rows)
    output = project_path(args.output_csv)
    ensure_dir(output.parent)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    payload = {
        "tickets_csv": str(project_path(args.tickets_csv)),
        "output_csv": str(output),
        "rows": int(len(out)),
        "races": int(out["race_id"].nunique()) if not out.empty else 0,
        "labels": out["decision_label"].value_counts().to_dict() if not out.empty else {},
        "note": "Fixture only. Do not use for performance claims.",
    }
    (output.parent / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
