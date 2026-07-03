from __future__ import annotations

import argparse
import csv
import re
import urllib.request
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

from pypdf import PdfReader


BASE_URL = "https://www.jra.go.jp/keiba/baba/archive/"


class ArchiveLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        self._href = attrs_dict.get("href")
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href:
            text = " ".join(" ".join(self._parts).split())
            self.links.append((self._href, text))
            self._href = None
            self._parts = []


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def parse_archive_links(html: str, page_url: str) -> list[tuple[str, str]]:
    parser = ArchiveLinkParser()
    parser.feed(html)
    return [
        (urljoin(page_url, href), text)
        for href, text in parser.links
        if href.lower().endswith(".pdf") and "/keiba/baba/archive/" in urljoin(page_url, href)
    ]


def parse_pdf_rows(pdf_path: Path, source_url: str) -> list[dict[str, object]]:
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    title_match = re.search(r"(\d{4})年\s+(?:第)?\d+回\s*([^\s　]+?)競馬", text)
    if not title_match:
        raise ValueError(f"Could not read year/venue from {pdf_path}")
    year = int(title_match.group(1))
    venue = title_match.group(2)

    rows: list[dict[str, object]] = []
    row_pattern = re.compile(
        r"(?:(第\s*\d+日)\s+)?"
        r"(\d{1,2})月\s*(\d{1,2})日\s+"
        r"(\S+曜日)\s+"
        r"([A-D])\s+"
        r"(\d{1,2}:\d{2})\s+"
        r"([0-9.]+)\s+"
        r"(\d{1,2}:\d{2})\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
    )
    for line in text.splitlines():
        line = " ".join(line.split())
        match = row_pattern.search(line)
        if not match:
            continue
        day_label, month, day, weekday, course, cushion_time, cushion, moisture_time, turf_goal, turf_back, dirt_goal, dirt_back = match.groups()
        rows.append(
            {
                "date": f"{year}{int(month):02d}{int(day):02d}",
                "venue": venue,
                "course": course,
                "weekday": weekday,
                "race_day_label": (day_label or "").replace(" ", ""),
                "cushion_value": float(cushion),
                "cushion_measured_at": cushion_time,
                "moisture_turf_goal": float(turf_goal),
                "moisture_turf_back": float(turf_back),
                "moisture_dirt_goal": float(dirt_goal),
                "moisture_dirt_back": float(dirt_back),
                "moisture_measured_at": moisture_time,
                "source_url": source_url,
                "source_pdf": str(pdf_path),
            }
        )
    if rows:
        return rows
    return parse_legacy_weekly_pdf_rows(text, pdf_path, source_url, year, venue)


def parse_legacy_weekly_pdf_rows(
    text: str,
    pdf_path: Path,
    source_url: str,
    year: int,
    venue: str,
) -> list[dict[str, object]]:
    lines = [" ".join(line.split()) for line in text.splitlines() if " ".join(line.split())]
    rows: list[dict[str, object]] = []
    for i, line in enumerate(lines):
        if not (line.startswith("第") and "（" in line and "）" in line):
            continue
        date_match = re.search(
            r"（(\d{4})年(\d{1,2})月(\d{1,2})日[～〜](?:(\d{1,2})月)?(\d{1,2})日）",
            line,
        )
        if not date_match or i + 2 >= len(lines):
            continue
        block_year, start_month, start_day, end_month, end_day = date_match.groups()
        start = date(int(block_year), int(start_month), int(start_day))
        end = date(int(block_year), int(end_month or start_month), int(end_day))
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)

        weekdays = [part for part in lines[i + 1].split() if part.endswith("曜日")]
        cushions = _float_values(lines[i + 2])
        if not weekdays or len(cushions) < len(weekdays):
            continue
        turf_goal = turf_back = dirt_goal = dirt_back = []
        for j in range(i + 3, min(i + 12, len(lines))):
            target = lines[j]
            values = _float_values(target)
            if "芝コース含水率" in target and "ゴール前" in target:
                turf_goal = values
            elif "４コーナー" in target and not turf_back:
                turf_back = values
            elif "ダートコース含水率" in target and "ゴール前" in target:
                dirt_goal = values
            elif "４コーナー" in target and turf_back and not dirt_back:
                dirt_back = values

        count = min(len(weekdays), len(dates), len(cushions), len(turf_goal), len(turf_back), len(dirt_goal), len(dirt_back))
        for offset in range(count):
            day = dates[offset]
            rows.append(
                {
                    "date": day.strftime("%Y%m%d"),
                    "venue": venue,
                    "course": "",
                    "weekday": weekdays[offset],
                    "race_day_label": "",
                    "cushion_value": float(cushions[offset]),
                    "cushion_measured_at": "",
                    "moisture_turf_goal": float(turf_goal[offset]),
                    "moisture_turf_back": float(turf_back[offset]),
                    "moisture_dirt_goal": float(dirt_goal[offset]),
                    "moisture_dirt_back": float(dirt_back[offset]),
                    "moisture_measured_at": "",
                    "source_url": source_url,
                    "source_pdf": str(pdf_path),
                }
            )
    return rows


def _float_values(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"(?<!\d)(\d{1,2}\.\d)(?!\d)", text)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch JRA official cushion/moisture archive PDFs and build a CSV.")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    parser.add_argument("--pdf-dir", default="data/raw/jra_track_condition_pdfs")
    parser.add_argument("--output-csv", default="data/raw/track_condition_metrics.csv")
    args = parser.parse_args()

    pdf_root = Path(args.pdf_dir)
    pdf_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    for year in args.years:
        page_url = urljoin(BASE_URL, f"{year}.html")
        html = fetch_bytes(page_url).decode("shift_jis", errors="replace")
        links = parse_archive_links(html, page_url)
        year_dir = pdf_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        for url, _text in links:
            pdf_path = year_dir / Path(url).name
            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                pdf_path.write_bytes(fetch_bytes(url))
            all_rows.extend(parse_pdf_rows(pdf_path, url))

    fieldnames = [
        "date",
        "venue",
        "course",
        "weekday",
        "race_day_label",
        "cushion_value",
        "cushion_measured_at",
        "moisture_turf_goal",
        "moisture_turf_back",
        "moisture_dirt_goal",
        "moisture_dirt_back",
        "moisture_measured_at",
        "source_url",
        "source_pdf",
    ]
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: (str(row["date"]), str(row["venue"]))))

    unique_dates = {row["date"] for row in all_rows}
    print(
        {
            "output_csv": str(output),
            "rows": len(all_rows),
            "dates": len(unique_dates),
            "years": args.years,
        }
    )


if __name__ == "__main__":
    main()
