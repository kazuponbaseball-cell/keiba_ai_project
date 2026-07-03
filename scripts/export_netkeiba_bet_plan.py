from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


TICKET_LABELS = {
    "win": "単勝",
    "wide": "ワイド",
    "umaren": "馬連",
    "umatan": "馬単",
    "sanrenpuku": "3連複",
    "trio": "3連複",
}

VENUE_BY_CODE = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


PLAN_COLUMNS = [
    "race_id",
    "date",
    "venue",
    "race_no",
    "ticket_type",
    "ticket_label",
    "numbers",
    "amount_yen",
    "anchor_no",
    "anchor_name",
    "partner_no",
    "partner_name",
    "third_no",
    "third_name",
    "runtime_action",
    "runtime_margin",
    "min_acceptable_odds",
    "runtime_odds",
]


def _num(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _first_value(row: pd.Series, names: list[str], default: object = "") -> object:
    for name in names:
        if name in row.index and not pd.isna(row.get(name)):
            value = row.get(name)
            if str(value).strip() != "":
                return value
    return default


def _horse_no(value) -> str:
    try:
        if pd.isna(value):
            return ""
        return str(int(float(value)))
    except Exception:
        return str(value or "")


def _race_no(row: pd.Series) -> int | str:
    value = _first_value(row, ["race_no", "Ｒ", "R"], "")
    if str(value).strip() == "":
        race_id = str(row.get("race_id", ""))
        if len(race_id) >= 2 and race_id[-2:].isdigit():
            value = race_id[-2:]
    try:
        return int(float(value))
    except Exception:
        return ""


def _date(row: pd.Series) -> str:
    value = _first_value(row, ["date", "date_key", "日付S"], "")
    if str(value).strip() and str(value).lower() != "nan":
        return str(value)
    race_id = str(row.get("race_id", ""))
    if len(race_id) >= 8 and race_id[:8].isdigit():
        return f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}"
    return ""


def _venue(row: pd.Series) -> str:
    value = row.get("venue", "")
    if not pd.isna(value) and str(value).strip():
        return str(value)
    race_id = str(row.get("race_id", "")).zfill(16)
    return VENUE_BY_CODE.get(race_id[8:10], "")


def _bet_numbers(row: pd.Series, ticket_type: str) -> str:
    a_no = _horse_no(row.get("anchor_no"))
    b_no = _horse_no(row.get("partner_no"))
    c_no = _horse_no(row.get("third_no"))
    if ticket_type == "win":
        return a_no
    if c_no:
        return f"{a_no}-{b_no}-{c_no}"
    if b_no:
        return f"{a_no}-{b_no}"
    return a_no


def _build_rows(tickets: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    df["race_id"] = df["race_id"].astype(str)
    if "runtime_action" in df.columns:
        df = df[df["runtime_action"].astype(str).isin(["BUY", "BUY_CONTEXT_BOOST", "REDUCE", "REDUCE_ALERT"])].copy()
    if "runtime_stake_yen" in df.columns:
        df = df[pd.to_numeric(df["runtime_stake_yen"], errors="coerce").fillna(0).gt(0)].copy()

    rows: list[dict] = []
    for _, row in df.iterrows():
        ticket_type = str(row.get("ticket_type", ""))
        stake = int(_num(row.get("runtime_stake_yen", row.get("stake_yen")), 0))
        rows.append(
            {
                "race_id": row.get("race_id", ""),
                "date": _date(row),
                "venue": _venue(row),
                "race_no": _race_no(row),
                "ticket_type": ticket_type,
                "ticket_label": TICKET_LABELS.get(ticket_type, ticket_type),
                "numbers": _bet_numbers(row, ticket_type),
                "amount_yen": stake,
                "anchor_no": _horse_no(row.get("anchor_no")),
                "anchor_name": row.get("anchor_name", ""),
                "partner_no": _horse_no(row.get("partner_no")),
                "partner_name": row.get("partner_name", ""),
                "third_no": _horse_no(row.get("third_no")),
                "third_name": row.get("third_name", ""),
                "runtime_action": row.get("runtime_action", ""),
                "runtime_margin": round(_num(row.get("runtime_odds_margin_ratio"), 0), 2),
                "min_acceptable_odds": round(_num(row.get("min_acceptable_odds"), 0), 2),
                "runtime_odds": round(_num(row.get("runtime_odds"), 0), 2),
            }
        )
    plan = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    if plan.empty:
        return plan
    return plan.sort_values(["date", "venue", "race_no", "ticket_label", "numbers"], na_position="last")


def _write_text_plan(plan: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    if plan.empty:
        lines.append("買い目なし")
    else:
        for (date, venue, race_no), group in plan.groupby(["date", "venue", "race_no"], dropna=False):
            date_text = "" if pd.isna(date) else str(date)
            venue_text = "" if pd.isna(venue) else str(venue)
            race_no_text = "" if pd.isna(race_no) else str(race_no)
            lines.append(f"{date_text} {venue_text}{race_no_text}R")
            for _, row in group.iterrows():
                lines.append(
                    f"  {row['ticket_label']} {row['numbers']} {int(row['amount_yen'])}円 "
                    f"(margin {row['runtime_margin']}x / odds {row['runtime_odds']} / min {row['min_acceptable_odds']})"
                )
            lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export runtime tickets as a netkeiba-friendly bet plan.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/runtime_odds_decision_rules_v1/runtime_selected_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/integration/netkeiba_bet_plan")
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    plan = _build_rows(tickets)
    out_dir = ensure_dir(project_path(args.output_dir))
    csv_path = out_dir / "netkeiba_bet_plan.csv"
    json_path = out_dir / "netkeiba_bet_plan.json"
    txt_path = out_dir / "netkeiba_bet_plan.txt"
    plan.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(plan.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_text_plan(plan, txt_path)
    payload = {
        "rows": int(len(plan)),
        "races": int(plan["race_id"].nunique()) if not plan.empty else 0,
        "csv": str(csv_path),
        "json": str(json_path),
        "text": str(txt_path),
        "note": "This is a safe handoff file for netkeiba/manual entry. It does not log in, submit, or purchase tickets.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
