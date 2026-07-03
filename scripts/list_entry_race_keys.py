from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def date_short(value: str) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) == 8 and digits.startswith("20"):
        return digits[2:]
    if len(digits) == 6:
        return digits
    return digits


def first_existing(columns: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="List 16-digit race IDs from an entry snapshot for a target date.")
    parser.add_argument("--entry-csv", required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    frame = read_csv(Path(args.entry_csv))
    date_col = first_existing(list(frame.columns), ["日付", "date", "race_date"])
    race_col = first_existing(list(frame.columns), ["レースID(新/馬番無)", "race_id"])
    if not date_col or not race_col or frame.empty:
        return 0

    wanted = date_short(args.date)
    dates = frame[date_col].astype(str).map(date_short)
    values = frame.loc[dates.eq(wanted), race_col].dropna().astype(str)
    race_ids = sorted(
        {
            re.sub(r"\.0$", "", value.strip())
            for value in values
            if re.match(r"^20\d{14}(?:\.0)?$", value.strip())
        }
    )
    for race_id in race_ids:
        print(race_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
