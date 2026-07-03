from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TARGET CK_DATA hill/wood workout records.")
    parser.add_argument("--target-root", default=r"C:\Users\kazup\Data Lab")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-csv", default="data/processed/target/workouts.csv")
    args = parser.parse_args()

    ck_root = Path(args.target_root) / "CK_DATA"
    rows: list[dict[str, object]] = []
    for path in _iter_source_files(ck_root, args.start_date, args.end_date):
        date = _date_from_name(path.name)
        if not date:
            continue
        if args.start_date and date < args.start_date:
            continue
        if args.end_date and date > args.end_date:
            continue
        if path.name.startswith("HC"):
            rows.extend(_parse_hc(path))
        elif path.name.startswith("WC"):
            rows.extend(_parse_wc(path))

    out = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "rows": int(len(out)),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "output_csv": str(output_path),
                "source_root": str(ck_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _date_from_name(name: str) -> str:
    stem = Path(name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    for i in range(0, max(0, len(digits) - 7)):
        value = digits[i : i + 8]
        if value.startswith("20"):
            return value
    return ""


def _iter_source_files(ck_root: Path, start_date: str | None, end_date: str | None) -> list[Path]:
    if start_date and end_date:
        months = _month_range(start_date[:6], end_date[:6])
        files: list[Path] = []
        for month in months:
            year_dir = ck_root / month[:4] / month
            if year_dir.exists():
                files.extend(year_dir.glob("*.DAT"))
        return sorted(files)
    return sorted(ck_root.rglob("*.DAT"))


def _month_range(start_yyyymm: str, end_yyyymm: str) -> list[str]:
    current = datetime.strptime(start_yyyymm + "01", "%Y%m%d")
    end = datetime.strptime(end_yyyymm + "01", "%Y%m%d")
    months = []
    while current <= end:
        months.append(current.strftime("%Y%m"))
        year = current.year + (1 if current.month == 12 else 0)
        month = 1 if current.month == 12 else current.month + 1
        current = current.replace(year=year, month=month)
    return months


def _parse_hc(path: Path) -> list[dict[str, object]]:
    rows = []
    for raw in path.read_bytes().splitlines():
        if len(raw) < 47:
            continue
        text = raw.decode("ascii", errors="ignore")
        row = {
            "source_file": str(path),
            "source_type": "HC",
            "course": "hill",
            "tracen_kubun": text[0:1],
            "workout_date": text[1:9],
            "workout_time": text[9:13],
            "horse_id": text[13:23],
            "distance_f": 4,
            "total_time_sec": _tenths(text[23:27]),
            "lap_4f_sec": _tenths(text[27:30]),
            "final_3f_sec": _tenths(text[30:34]),
            "lap_3f_sec": _tenths(text[34:37]),
            "final_2f_sec": _tenths(text[37:41]),
            "lap_2f_sec": _tenths(text[41:44]),
            "final_1f_sec": _tenths(text[44:47]),
        }
        rows.append(row)
    return rows


def _parse_wc(path: Path) -> list[dict[str, object]]:
    rows = []
    for raw in path.read_bytes().splitlines():
        if len(raw) < 92:
            continue
        text = raw.decode("ascii", errors="ignore")
        pos = 23
        row: dict[str, object] = {
            "source_file": str(path),
            "source_type": "WC",
            "course": "wood",
            "tracen_kubun": text[0:1],
            "workout_date": text[1:9],
            "workout_time": text[9:13],
            "horse_id": text[13:23],
            "wood_course_code": text[pos : pos + 1],
            "wood_babamawari": text[pos + 1 : pos + 2],
        }
        pos += 3
        total_by_f: dict[int, float] = {}
        lap_by_f: dict[int, float] = {}
        for f in range(10, 1, -1):
            total_by_f[f] = _tenths(text[pos : pos + 4])
            pos += 4
            lap_by_f[f] = _tenths(text[pos : pos + 3])
            pos += 3
        lap_by_f[1] = _tenths(text[pos : pos + 3])

        available = [(f, value) for f, value in total_by_f.items() if value and value > 0]
        if available:
            distance_f, total_time = max(available, key=lambda item: item[0])
        else:
            distance_f, total_time = 0, None

        row.update(
            {
                "distance_f": distance_f,
                "total_time_sec": total_time,
                "final_3f_sec": total_by_f.get(3),
                "final_2f_sec": total_by_f.get(2),
                "final_1f_sec": lap_by_f.get(1),
                "lap_10f_sec": lap_by_f.get(10),
                "lap_9f_sec": lap_by_f.get(9),
                "lap_8f_sec": lap_by_f.get(8),
                "lap_7f_sec": lap_by_f.get(7),
                "lap_6f_sec": lap_by_f.get(6),
                "lap_5f_sec": lap_by_f.get(5),
                "lap_4f_sec": lap_by_f.get(4),
                "lap_3f_sec": lap_by_f.get(3),
                "lap_2f_sec": lap_by_f.get(2),
            }
        )
        rows.append(row)
    return rows


def _tenths(value: str) -> float | None:
    if not value or set(value) <= {"0", " "}:
        return None
    try:
        return int(value) / 10.0
    except ValueError:
        return None


if __name__ == "__main__":
    main()
