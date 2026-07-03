from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def read_csv_safe(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def date_key(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    for fmt in ("%Y%m%d", "%y%m%d", "%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y%m%d")
        except Exception:
            pass
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 6:
        return "20" + digits
    return digits[:8]


def find_date_col(frame: pd.DataFrame) -> str | None:
    for candidate in ["\u65e5\u4ed8", "date"]:
        if candidate in frame.columns:
            return candidate
    for column in frame.columns:
        sample = frame[column].dropna().astype(str).head(20)
        if len(sample) and sample.str.match(r"^\d{6}$|^\d{8}$|^\d{4}[./-]\d{1,2}[./-]\d{1,2}$").mean() >= 0.5:
            return column
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether an entry snapshot contains a target date.")
    parser.add_argument("--entry-csv", required=True)
    parser.add_argument("--target-date", required=True)
    args = parser.parse_args()

    path = Path(args.entry_csv)
    if not path.exists():
        print("missing")
        raise SystemExit(1)
    frame = read_csv_safe(path)
    date_col = find_date_col(frame)
    if date_col is None:
        print("date_column_missing")
        raise SystemExit(1)
    dates = sorted({date_key(value) for value in frame[date_col].dropna().unique() if date_key(value)})
    print(",".join(dates))
    raise SystemExit(0 if args.target_date in dates else 1)


if __name__ == "__main__":
    main()
