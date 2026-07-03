from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from extract_jra_official_race_laps import clean_html_text, parse_course, race_id_from_path, read_text, snapshot_key


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "jra_official_results"
DEFAULT_OUT = ROOT / "data" / "processed" / "jra_official_results" / "result_runners.csv"


TR_RE = re.compile(r"<tr>\s*(.*?)\s*</tr>", re.S)
TD_RE_TEMPLATE = r'<td class="{klass}[^"]*"[^>]*>(.*?)</td>'
HORSE_RE = re.compile(r'accessU\.html\?CNAME=pw01dud0*(\d{10})/[^"]*">([^<]+)</a>', re.S)
TAG_RE = re.compile(r"<[^>]+>")
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def latest_files(raw_dir: Path) -> list[Path]:
    files = list(raw_dir.rglob("*.html"))
    by_race: dict[str, Path] = {}
    for path in files:
        rid = race_id_from_path(path)
        if not rid:
            continue
        cur = by_race.get(rid)
        if cur is None or snapshot_key(path) > snapshot_key(cur):
            by_race[rid] = path
    return sorted(by_race.values(), key=lambda p: race_id_from_path(p))


def td_html(row_html: str, klass: str) -> str:
    m = re.search(TD_RE_TEMPLATE.format(klass=re.escape(klass)), row_html, re.S)
    return m.group(1) if m else ""


def td_text(row_html: str, klass: str) -> str:
    return clean_html_text(td_html(row_html, klass))


def parse_time_to_sec(value: str) -> float | None:
    raw = value.strip()
    if not raw:
        return None
    m = re.match(r"(?:(\d+):)?(\d+(?:\.\d+)?)", raw)
    if not m:
        return None
    return float(m.group(1) or 0) * 60.0 + float(m.group(2))


def parse_int_text(value: str) -> int | None:
    m = re.search(r"\d+", value)
    return int(m.group(0)) if m else None


def parse_body_weight(value: str) -> tuple[int | None, int | None]:
    raw = value.replace(" ", "")
    m = re.search(r"(\d+)(?:\(([+-]?\d+)\))?", raw)
    if not m:
        return None, None
    body = int(m.group(1))
    diff = int(m.group(2)) if m.group(2) else None
    return body, diff


def parse_corner(row_html: str) -> list[int | None]:
    corners: list[int | None] = []
    for title in ["1コーナー", "2コーナー", "3コーナー", "4コーナー"]:
        m = re.search(rf'<li title="{title}通過順位">\s*(\d+)\s*</li>', row_html)
        corners.append(int(m.group(1)) if m else None)
    return corners


def parse_sex_age(value: str) -> tuple[str, int | None]:
    raw = value.strip()
    if not raw:
        return "", None
    sex = raw[0]
    age = parse_int_text(raw)
    return sex, age


def parse_row(row_html: str, race_id: str, source_html: Path) -> dict[str, Any] | None:
    hm = HORSE_RE.search(row_html)
    if not hm:
        return None
    horse_id = hm.group(1)
    horse_name = clean_html_text(hm.group(2))
    body, body_diff = parse_body_weight(td_text(row_html, "h_weight"))
    c1, c2, c3, c4 = parse_corner(row_html)
    sex, age = parse_sex_age(td_text(row_html, "age"))
    return {
        "race_id": race_id,
        "source_html": str(source_html),
        "snapshot_key": snapshot_key(source_html),
        "finish": parse_int_text(td_text(row_html, "place")),
        "frame_no": parse_int_text(td_html(row_html, "waku")),
        "horse_no": parse_int_text(td_text(row_html, "num")),
        "horse_id": horse_id,
        "horse_name": horse_name,
        "sex": sex,
        "age": age,
        "carried_weight": float(FLOAT_RE.search(td_text(row_html, "weight")).group(0)) if FLOAT_RE.search(td_text(row_html, "weight")) else None,
        "jockey": td_text(row_html, "jockey"),
        "race_time": td_text(row_html, "time"),
        "race_time_sec": parse_time_to_sec(td_text(row_html, "time")),
        "margin": td_text(row_html, "margin"),
        "corner1": c1,
        "corner2": c2,
        "corner3": c3,
        "corner4": c4,
        "estimated_last3f_sec": parse_time_to_sec(td_text(row_html, "f_time")),
        "body_weight": body,
        "body_weight_diff": body_diff,
        "trainer": td_text(row_html, "trainer"),
        "popularity": parse_int_text(td_text(row_html, "pop")),
    }


def parse_one(path: Path) -> list[dict[str, Any]]:
    text = read_text(path)
    race_id = race_id_from_path(path)
    course = parse_course(text)
    rows: list[dict[str, Any]] = []
    for tr in TR_RE.findall(text):
        row = parse_row(tr, race_id, path)
        if row is None:
            continue
        row.update(course)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract runner result rows from saved JRA official result HTML.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else ROOT / args.raw_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else ROOT / args.output_csv
    rows: list[dict[str, Any]] = []
    for path in latest_files(raw_dir):
        rows.extend(parse_one(path))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["race_id", "horse_no", "horse_id"]).sort_values(["race_id", "finish", "horse_no"])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = {
        "raw_dir": str(raw_dir),
        "output_csv": str(output_csv),
        "races": int(df["race_id"].nunique()) if not df.empty else 0,
        "rows": int(len(df)),
        "horse_ids": int(df["horse_id"].nunique()) if not df.empty else 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not df.empty:
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
