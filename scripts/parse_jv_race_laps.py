from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extract_jra_official_race_laps import lap_shape_features, section_lengths


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "processed" / "jv_race_laps" / "race_laps.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "analysis" / "jv_race_lap_parse_v1" / "summary.json"


TRACK_SURFACE = {
    "10": "芝",
    "11": "芝",
    "12": "芝",
    "13": "芝",
    "14": "芝",
    "15": "芝",
    "16": "芝",
    "17": "芝",
    "18": "芝",
    "19": "芝",
    "20": "ダート",
    "21": "ダート",
    "22": "ダート",
    "23": "ダート",
    "24": "ダート",
    "25": "ダート",
    "26": "ダート",
    "27": "ダート",
    "28": "ダート",
    "29": "ダート",
    "51": "障害",
    "52": "障害",
    "53": "障害",
    "54": "障害",
    "55": "障害",
    "56": "障害",
    "57": "障害",
    "58": "障害",
    "59": "障害",
}

GOING_CODE = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}
WEATHER_CODE = {"1": "晴", "2": "曇", "3": "雨", "4": "小雨", "5": "雪", "6": "小雪"}


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def field(record: str, start: int, length: int) -> str:
    # JVData positions are 1-based byte positions. Japanese fields before the
    # lap block are fixed-length Shift_JIS, so we slice the decoded string after
    # reading with cp932; ASCII numeric fields keep the same offsets in practice
    # because Python string positions correspond to displayed characters for
    # these fixed fields. Use bytes fallback when records contain wide chars.
    return record[start - 1 : start - 1 + length]


def bfield(raw: bytes, start: int, length: int) -> str:
    return raw[start - 1 : start - 1 + length].decode("cp932", errors="ignore")


def parse_int(value: str) -> int | None:
    value = value.strip()
    if not value or not value.isdigit():
        return None
    return int(value)


def parse_tenth(value: str) -> float | None:
    value = value.strip()
    if not value or not value.isdigit():
        return None
    num = int(value)
    if num <= 0 or num >= 999:
        return None
    return round(num / 10.0, 1)


def parse_ra_record_bytes(raw: bytes, source_path: Path, line_no: int) -> dict[str, Any] | None:
    rec_id = bfield(raw, 1, 2)
    if rec_id != "RA":
        return None
    data_kbn = bfield(raw, 3, 1)
    year = bfield(raw, 12, 4)
    month_day = bfield(raw, 16, 4)
    venue_code = bfield(raw, 20, 2)
    meeting_no = bfield(raw, 22, 2)
    day_no = bfield(raw, 24, 2)
    race_no = bfield(raw, 26, 2)
    if not (year + month_day + venue_code + meeting_no + day_no + race_no).isdigit():
        return None
    race_id = year + month_day + venue_code + meeting_no + day_no + race_no
    distance_m = parse_int(bfield(raw, 698, 4))
    track_code = bfield(raw, 706, 2)
    course_code = bfield(raw, 710, 2).strip()
    weather_code = bfield(raw, 888, 1)
    turf_going_code = bfield(raw, 889, 1)
    dirt_going_code = bfield(raw, 890, 1)

    lap_values = [parse_tenth(bfield(raw, 891 + i * 3, 3)) for i in range(25)]
    if distance_m:
        lap_count = int(np.ceil(distance_m / 200.0))
        laps = [x for x in lap_values[:lap_count] if x is not None]
    else:
        laps = [x for x in lap_values if x is not None]
    if not laps:
        return None

    row: dict[str, Any] = {
        "race_id": race_id,
        "data_kbn": data_kbn,
        "race_date": f"{year}-{month_day[:2]}-{month_day[2:]}",
        "venue_code": venue_code,
        "meeting_no": meeting_no,
        "day_no": day_no,
        "race_no": race_no,
        "distance_m": distance_m,
        "track_code": track_code,
        "surface": TRACK_SURFACE.get(track_code, ""),
        "course_code": course_code,
        "weather_code": weather_code,
        "weather": WEATHER_CODE.get(weather_code, ""),
        "turf_going_code": turf_going_code,
        "turf_going": GOING_CODE.get(turf_going_code, ""),
        "dirt_going_code": dirt_going_code,
        "dirt_going": GOING_CODE.get(dirt_going_code, ""),
        "jv_front_3f_sec": parse_tenth(bfield(raw, 970, 3)),
        "jv_front_4f_sec": parse_tenth(bfield(raw, 973, 3)),
        "jv_last_3f_sec": parse_tenth(bfield(raw, 976, 3)),
        "jv_last_4f_sec": parse_tenth(bfield(raw, 979, 3)),
        "source_records": str(source_path),
        "source_line_no": line_no,
    }
    row.update(lap_shape_features(laps, distance_m))
    if row.get("jv_front_3f_sec") is not None and pd.notna(row.get("first_3f_sec")):
        row["jv_front_3f_delta_sec"] = round(float(row["jv_front_3f_sec"]) - float(row["first_3f_sec"]), 3)
    if row.get("jv_last_3f_sec") is not None and pd.notna(row.get("last_3f_sec")):
        row["jv_last_3f_delta_sec"] = round(float(row["jv_last_3f_sec"]) - float(row["last_3f_sec"]), 3)
    return row


def iter_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # fetch_jv_historical_race_data.ps1 writes each JVRead buffer as a line.
    # Read as bytes so fixed-position offsets remain JVData byte offsets.
    for line_no, raw in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw:
            continue
        row = parse_ra_record_bytes(raw, path, line_no)
        if row is not None:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse JVData RA race-detail records into race-level lap features.")
    parser.add_argument("records", nargs="+", type=Path, help="records.txt files produced by fetch_jv_historical_race_data.ps1")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    records_paths = [project_path(p) for p in args.records]
    output_csv = project_path(args.output_csv)
    summary_json = project_path(args.summary_json)
    rows: list[dict[str, Any]] = []
    scanned_files = 0
    for path in records_paths:
        if not path.exists():
            continue
        scanned_files += 1
        rows.extend(iter_records(path))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["race_id", "data_kbn", "source_records", "source_line_no"], kind="mergesort")
        # Prefer the most final data category for duplicate races.
        final_order = {"7": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2, "1": 1}
        df["_data_kbn_order"] = df["data_kbn"].map(final_order).fillna(0)
        df = df.sort_values(["race_id", "_data_kbn_order"], ascending=[True, False], kind="mergesort")
        df = df.drop_duplicates("race_id", keep="first").drop(columns=["_data_kbn_order"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = {
        "records_files_requested": len(records_paths),
        "records_files_scanned": scanned_files,
        "output_csv": str(output_csv),
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()) if not df.empty else 0,
        "shape_counts": df.get("official_lap_shape", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
        if not df.empty
        else {},
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not df.empty:
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
