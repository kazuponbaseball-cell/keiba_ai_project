from __future__ import annotations

import argparse
import csv
import html
import json
import re
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

PAGES = [
    "https://www.jra.go.jp/keiba/baba/",
    "https://www.jra.go.jp/keiba/baba/index2.html",
    "https://www.jra.go.jp/keiba/baba/index3.html",
]
ACCESS_J_URL = "https://www.jra.go.jp/JRADB/accessJ.html"
ACCESS_J_CNAME = "pw01iwtS3/CD"
GOING_VALUES = {"良", "稍重", "稍", "重", "不良", "不"}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    for enc in ("utf-8", "shift_jis", "cp932"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def decode_text(data: bytes) -> str:
    for enc in ("utf-8", "shift_jis", "cp932"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def fetch_access_j_rows() -> list[dict[str, Any]]:
    data = f"CNAME={ACCESS_J_CNAME}".encode("ascii")
    request = urllib.request.Request(
        ACCESS_J_URL,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(decode_text(response.read()))

    fetched_at = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    for row in payload.get("kaisai_info", []):
        venue = str(row.get("jyoname") or "").strip()
        effective_date = str(row.get("kaisai_ymd") or "").strip()
        turf_going = normalize_going(str(row.get("ba_s") or "").strip())
        dirt_going = normalize_going(str(row.get("ba_d") or "").strip())
        if not venue or not effective_date or not (turf_going or dirt_going):
            continue
        rows.append(
            {
                "effective_date": effective_date,
                "observed_date": effective_date,
                "timing": "race_day_api",
                "venue": venue,
                "weather": str(row.get("weather") or "").strip(),
                "turf_going": turf_going,
                "dirt_going": dirt_going,
                "source_url": ACCESS_J_URL,
                "fetched_at": fetched_at,
            }
        )
    return rows


def html_lines(source: str) -> list[str]:
    source = re.sub(r"<(br|BR)\s*/?>", "\n", source)
    source = re.sub(r"</(p|div|li|h[1-6]|td|th|tr|dt|dd)>", "\n", source)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", source))
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_going(value: str) -> str:
    value = value.strip()
    if value == "稍":
        return "稍重"
    if value == "不":
        return "不良"
    return value


def find_next_going(lines: list[str], start: int) -> str:
    for line in lines[start + 1 : start + 12]:
        value = normalize_going(line)
        if value in GOING_VALUES:
            return value
    return ""


def parse_status_date(lines: list[str]) -> tuple[str, str, str]:
    joined = "\n".join(lines)
    match = re.search(r"第[^\n（]+?(前日)?（(\d{4})年(\d{1,2})月(\d{1,2})日", joined)
    if not match:
        return "", "", ""
    is_preday, year, month, day = match.groups()
    observed = datetime(int(year), int(month), int(day))
    effective = observed + timedelta(days=1) if is_preday else observed
    return (
        observed.strftime("%Y%m%d"),
        effective.strftime("%Y%m%d"),
        "preday_noon" if is_preday else "race_day",
    )


def parse_page(url: str) -> dict[str, Any]:
    source = fetch_text(url)
    lines = html_lines(source)
    joined = "\n".join(lines)
    venue = ""
    match = re.search(r"馬場情報（(.+?)競馬場）", joined)
    if match:
        venue = match.group(1)

    observed_date, effective_date, timing = parse_status_date(lines)
    if not effective_date:
        effective_date = datetime.now().strftime("%Y%m%d")
        timing = "current_page"
    weather = ""
    weather_match = re.search(r"天候：([^\n]+)", joined)
    if weather_match:
        weather = weather_match.group(1).strip()

    turf_going = ""
    dirt_going = ""
    for i, line in enumerate(lines):
        if line == "芝" and not turf_going:
            turf_going = find_next_going(lines, i)
        elif line == "ダート" and not dirt_going:
            dirt_going = find_next_going(lines, i)

    return {
        "effective_date": effective_date,
        "observed_date": observed_date,
        "timing": timing,
        "venue": venue,
        "weather": weather,
        "turf_going": turf_going,
        "dirt_going": dirt_going,
        "source_url": url,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch current JRA going information from the public baba pages.")
    parser.add_argument("--output-csv", default="data/processed/live_track_conditions/current_track_conditions.csv")
    parser.add_argument("--summary-json", default="outputs/analysis/live_track_conditions/current_track_conditions_summary.json")
    args = parser.parse_args()

    rows = fetch_access_j_rows()
    if not rows:
        rows = [row for row in (parse_page(url) for url in PAGES) if row.get("venue")]
    output = project_path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "effective_date",
        "observed_date",
        "timing",
        "venue",
        "weather",
        "turf_going",
        "dirt_going",
        "source_url",
        "fetched_at",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_csv": str(output),
        "rows": len(rows),
        "venues": [row["venue"] for row in rows],
        "complete": all(row.get("turf_going") and row.get("dirt_going") for row in rows),
        "rows_data": rows,
    }
    summary_path = project_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(__import__("json").dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
