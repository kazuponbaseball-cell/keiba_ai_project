from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


VENUE_CODE_BY_NAME = {
    "札幌": "01",
    "函館": "02",
    "福島": "03",
    "新潟": "04",
    "東京": "05",
    "中山": "06",
    "中京": "07",
    "京都": "08",
    "阪神": "09",
    "小倉": "10",
}


def _project_path(path: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    p = Path(path)
    return p if p.is_absolute() else root / p


def _date_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y%m%d")
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 6:
        # TARGET-style yymmdd.
        yy = int(digits[:2])
        yyyy = 2000 + yy if yy < 80 else 1900 + yy
        return f"{yyyy}{digits[2:]}"
    return digits[:8] if len(digits) >= 8 else ""


def _post_time(date_key: str, value: object) -> str:
    text = str(value or "").strip()
    if not date_key or not text:
        return ""
    for fmt in ("%Y%m%d %H:%M", "%Y%m%d %H%M", "%Y%m%d %H:%M:%S"):
        try:
            normalized = text if ":" in text else text.zfill(4)
            candidate = f"{date_key} {normalized}"
            return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    parsed = pd.to_datetime(f"{date_key} {text}", errors="coerce")
    return parsed.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(parsed) else ""


def _race_key(date_key: str, venue: str, race_no: int) -> str:
    venue_code = VENUE_CODE_BY_NAME.get(str(venue).strip(), "")
    return f"{date_key}{venue_code}{race_no:02d}" if date_key and venue_code and race_no else ""


def _race_key_from_target_id(value: object) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:16] if len(digits) >= 16 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a live-odds automation schedule from a TARGET entry snapshot.")
    parser.add_argument("--entry-csv", default="data/datasets/inference/weekly/entry_snapshot.csv")
    parser.add_argument("--date", default="", help="Target date such as 2026-06-07, 20260607, or 260607. Empty keeps all dates.")
    parser.add_argument("--venues", nargs="*", default=[], help="Venue names. Empty keeps all venues.")
    parser.add_argument("--races", nargs="*", type=int, default=[], help="Race numbers. Empty keeps all races.")
    parser.add_argument("--output-csv", default="data/processed/live_odds/live_odds_race_schedule.csv")
    parser.add_argument("--output-json", default="data/processed/live_odds/live_odds_race_schedule.json")
    args = parser.parse_args()

    source = pd.read_csv(_project_path(args.entry_csv), dtype=str, low_memory=False)
    required = ["日付S", "場所", "Ｒ", "発走時刻"]
    missing = [col for col in required if col not in source.columns]
    if missing:
        raise ValueError(f"entry csv missing columns: {missing}")

    race_name_col = "レース名" if "レース名" in source.columns else None
    race_id_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in source.columns else None
    cols = [*required]
    if race_name_col:
        cols.append(race_name_col)
    if race_id_col:
        cols.append(race_id_col)
    races = source[cols].drop_duplicates().copy()
    races["date_key"] = races["日付S"].map(_date_key)
    races["venue"] = races["場所"].astype(str).str.strip()
    races["race_no"] = pd.to_numeric(races["Ｒ"], errors="coerce").fillna(0).astype(int)
    races["post_time"] = [_post_time(row.date_key, row.発走時刻) for row in races.itertuples(index=False)]
    if race_name_col:
        races["race_name"] = races[race_name_col]
    else:
        races["race_name"] = ""
    if race_id_col:
        races["target_race_id"] = races[race_id_col].astype(str)
    else:
        races["target_race_id"] = ""
    races["race_key"] = [
        _race_key_from_target_id(row.target_race_id) or _race_key(row.date_key, row.venue, row.race_no)
        for row in races.itertuples(index=False)
    ]

    if args.date:
        races = races[races["date_key"].eq(_date_key(args.date))].copy()
    if args.venues:
        races = races[races["venue"].isin(args.venues)].copy()
    if args.races:
        races = races[races["race_no"].isin(args.races)].copy()
    races = races[races["race_key"].ne("") & races["post_time"].ne("")].copy()
    races = races.sort_values(["post_time", "venue", "race_no"])

    out_cols = ["race_key", "target_race_id", "date_key", "venue", "race_no", "race_name", "post_time"]
    output_csv = _project_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    races[out_cols].to_csv(output_csv, index=False, encoding="utf-8-sig")

    payload = races[out_cols].to_dict(orient="records")
    output_json = _project_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_csv": str(output_csv), "output_json": str(output_json), "races": len(payload)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
