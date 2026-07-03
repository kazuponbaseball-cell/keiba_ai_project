from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


TARGET_ROOT = Path("C:/Users/kazup/Data Lab")
SE_RACE_ID_OFFSET = 11
SE_RACE_ID_LEN = 16
DATE_RE = re.compile(rb"20\d{6}")
DATE_TEXT_RE = re.compile(r"20\d{6}")


def yyyymmdd(value: datetime) -> str:
    return value.strftime("%Y%m%d")


def safe_read_lines(path: Path) -> list[bytes]:
    try:
        return path.read_bytes().splitlines()
    except OSError:
        return []


def scan_se_ra(se_root: Path, start_date: str, end_date: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        year_dir = se_root / str(year)
        if not year_dir.exists():
            continue
        for path in year_dir.glob("SR*.DAT"):
            for raw in safe_read_lines(path):
                if not raw.startswith(b"RA"):
                    continue
                race_id = raw[SE_RACE_ID_OFFSET : SE_RACE_ID_OFFSET + SE_RACE_ID_LEN].decode(
                    "ascii", errors="ignore"
                )
                if len(race_id) != SE_RACE_ID_LEN or not race_id[:8].isdigit():
                    continue
                date = race_id[:8]
                if start_date <= date <= end_date:
                    counts[date] += 1
    return counts


def scan_se_su(se_root: Path, start_date: str, end_date: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        year_dir = se_root / str(year)
        if not year_dir.exists():
            continue
        for path in year_dir.glob("SU*.DAT"):
            for raw in safe_read_lines(path):
                if not raw.startswith(b"SE"):
                    continue
                race_id = raw[SE_RACE_ID_OFFSET : SE_RACE_ID_OFFSET + SE_RACE_ID_LEN].decode(
                    "ascii", errors="ignore"
                )
                if len(race_id) != SE_RACE_ID_LEN or not race_id[:8].isdigit():
                    continue
                date = race_id[:8]
                if start_date <= date <= end_date:
                    counts[date] += 1
    return counts


def scan_date_mentions(root: Path, pattern: str, start_date: str, end_date: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not root.exists():
        return counts
    for path in root.rglob(pattern):
        for hit in DATE_RE.findall(path.read_bytes() if path.is_file() else b""):
            date = hit.decode("ascii", errors="ignore")
            if start_date <= date <= end_date:
                counts[date] += 1
    return counts


def scan_files_by_date(root: Path, start_date: str, end_date: str) -> tuple[Counter[str], dict[str, list[str]]]:
    counts: Counter[str] = Counter()
    files: dict[str, list[str]] = defaultdict(list)
    if not root.exists():
        return counts, files
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        matches = DATE_TEXT_RE.findall(path.name)
        for date in matches:
            if start_date <= date <= end_date:
                counts[date] += 1
                files[date].append(str(path))
    return counts, files


def date_range(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    days = []
    current = start
    while current <= end:
        days.append(yyyymmdd(current))
        current += timedelta(days=1)
    return days


def latest_nonzero(counter: Counter[str]) -> str:
    dates = [date for date, count in counter.items() if count > 0]
    return max(dates) if dates else ""


def build_rows(target_root: Path, start_date: str, end_date: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    se_root = target_root / "SE_DATA"
    ra_counts = scan_se_ra(se_root, start_date, end_date)
    su_counts = scan_se_su(se_root, start_date, end_date)
    schd_counts = scan_date_mentions(se_root, "SCHD*.DAT", start_date, end_date)
    ck_counts, ck_files = scan_files_by_date(target_root / "CK_DATA", start_date, end_date)
    de_counts, de_files = scan_files_by_date(target_root / "DE_DATA", start_date, end_date)
    jg_counts, jg_files = scan_files_by_date(target_root / "JG_DATA", start_date, end_date)

    rows: list[dict[str, object]] = []
    for date in date_range(start_date, end_date):
        analysis_ready = ra_counts[date] > 0 and su_counts[date] > 0
        rows.append(
            {
                "date": date,
                "se_ra_races": ra_counts[date],
                "se_su_starters": su_counts[date],
                "se_schd_mentions": schd_counts[date],
                "ck_files": ck_counts[date],
                "de_files": de_counts[date],
                "jg_files": jg_counts[date],
                "analysis_ready": analysis_ready,
                "status": "ready" if analysis_ready else "not_ready",
                "ck_file_examples": " | ".join(ck_files.get(date, [])[:4]),
                "de_file_examples": " | ".join(de_files.get(date, [])[:4]),
                "jg_file_examples": " | ".join(jg_files.get(date, [])[:4]),
            }
        )

    summary = {
        "target_root": str(target_root),
        "start_date": start_date,
        "end_date": end_date,
        "latest_se_ra_date": latest_nonzero(ra_counts),
        "latest_se_su_date": latest_nonzero(su_counts),
        "latest_schd_date": latest_nonzero(schd_counts),
        "latest_ck_date": latest_nonzero(ck_counts),
        "latest_de_date": latest_nonzero(de_counts),
        "latest_jg_date": latest_nonzero(jg_counts),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]], summary: dict[str, object], target_date: str) -> None:
    target = next((row for row in rows if row["date"] == target_date), None)
    lines = [
        "# TARGET data availability diagnosis",
        "",
        "## Summary",
        "",
        f"- target root: `{summary['target_root']}`",
        f"- checked range: `{summary['start_date']}` - `{summary['end_date']}`",
        f"- latest SE RA race-lap date: `{summary['latest_se_ra_date'] or 'none'}`",
        f"- latest SE SU starter/result date: `{summary['latest_se_su_date'] or 'none'}`",
        f"- latest SE schedule mention date: `{summary['latest_schd_date'] or 'none'}`",
        f"- latest CK file date: `{summary['latest_ck_date'] or 'none'}`",
        f"- latest DE file date: `{summary['latest_de_date'] or 'none'}`",
        f"- latest JG file date: `{summary['latest_jg_date'] or 'none'}`",
        "",
        "## Target Date",
        "",
    ]
    if target:
        lines.extend(
            [
                f"- date: `{target_date}`",
                f"- SE RA races: `{target['se_ra_races']}`",
                f"- SE SU starters: `{target['se_su_starters']}`",
                f"- SE schedule mentions: `{target['se_schd_mentions']}`",
                f"- CK files: `{target['ck_files']}`",
                f"- DE files: `{target['de_files']}`",
                f"- JG files: `{target['jg_files']}`",
                f"- analysis ready: `{target['analysis_ready']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- This project's lap and horse-result analyses currently require SE_DATA `SR*.DAT` RA records and `SU*.DAT` SE records.",
            "- CK_DATA/DE_DATA presence means TARGET has some data for the date, but those files are not yet parsed by the lap-analysis scripts.",
            "- If CK_DATA is present while SE_DATA RA/SU is absent, TARGET can show data registration as complete while the current analysis pipeline still cannot use that date.",
            "",
            "## Recent Rows",
            "",
            "|date|SE RA|SE SU|SCHD|CK|DE|JG|ready|",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows[-21:]:
        lines.append(
            f"|{row['date']}|{row['se_ra_races']}|{row['se_su_starters']}|"
            f"{row['se_schd_mentions']}|{row['ck_files']}|{row['de_files']}|"
            f"{row['jg_files']}|{row['analysis_ready']}|"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row_for_date(rows: list[dict[str, object]], target_date: str) -> dict[str, object] | None:
    return next((row for row in rows if row["date"] == target_date), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose TARGET/JV local data availability by date.")
    parser.add_argument("--target-root", default=str(TARGET_ROOT))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--output-csv", default="outputs/analysis/target_data_availability.csv")
    parser.add_argument("--output-report", default="outputs/analysis/target_data_availability_report.md")
    parser.add_argument(
        "--require-target-ready",
        action="store_true",
        help="Exit with status 2 when the target date is not ready for current SE_DATA-based analyses.",
    )
    args = parser.parse_args()

    today = datetime.now()
    end_date = args.end_date or yyyymmdd(today)
    start_date = args.start_date or yyyymmdd(today - timedelta(days=45))
    target_date = args.target_date or yyyymmdd(today - timedelta(days=1))

    rows, summary = build_rows(Path(args.target_root), start_date, end_date)
    write_csv(Path(args.output_csv), rows)
    write_report(Path(args.output_report), rows, summary, target_date)
    target_row = row_for_date(rows, target_date)
    result = {
        "summary": summary,
        "target_date": target_date,
        "target": target_row,
        "ready": bool(target_row and target_row["analysis_ready"]),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_target_ready and not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
