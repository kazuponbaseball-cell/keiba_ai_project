from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from extract_jra_official_race_laps import lap_shape_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_ROOT = Path(r"C:\Users\kazup\Data Lab\SE_DATA")
DEFAULT_OUT = ROOT / "data" / "processed" / "target_ra_race_laps" / "race_laps.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "analysis" / "target_ra_race_laps_v1" / "summary.json"


# TARGET/JRA-VAN RA records are fixed-width bytes. These offsets are zero-based
# for a single record line without CRLF.
RA_RACE_ID_OFFSET = 11
RA_RACE_ID_LEN = 16
RA_RACE_NAME_OFFSET = 32
RA_RACE_NAME_LEN = 60
RA_DISTANCE_OFFSET = 697
RA_DISTANCE_LEN = 4
RA_TRACK_CODE_OFFSET = 705
RA_TRACK_CODE_LEN = 2
RA_COURSE_CODE_OFFSET = 709
RA_COURSE_CODE_LEN = 2
RA_WEATHER_OFFSET = 887
RA_TURF_GOING_OFFSET = 888
RA_DIRT_GOING_OFFSET = 889
RA_LAP_OFFSET = 890
RA_LAP_COUNT = 25
RA_LAP_WIDTH = 3
RA_FRONT_3F_OFFSET = 969
RA_FRONT_4F_OFFSET = 972
RA_LAST_3F_OFFSET = 975
RA_LAST_4F_OFFSET = 978


TRACK_SURFACE = {
    **{f"{i:02d}": "芝" for i in range(10, 20)},
    **{f"{i:02d}": "ダート" for i in range(20, 30)},
    **{f"{i:02d}": "障害" for i in range(51, 60)},
}

WEATHER_CODE = {
    "1": "晴",
    "2": "曇",
    "3": "雨",
    "4": "小雨",
    "5": "雪",
    "6": "小雪",
}

GOING_CODE = {
    "1": "良",
    "2": "稍重",
    "3": "重",
    "4": "不良",
}


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def decode_text(raw: bytes) -> str:
    return raw.decode("cp932", errors="replace").strip().replace("\u3000", "")


def parse_int(raw: bytes) -> int | None:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text.isdigit():
        return None
    return int(text)


def parse_tenth(raw: bytes) -> float | None:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text.isdigit():
        return None
    value = int(text)
    if value <= 0 or value >= 999:
        return None
    return round(value / 10.0, 1)


def parse_laps(raw: bytes) -> list[float]:
    laps: list[float] = []
    for idx in range(RA_LAP_COUNT):
        start = RA_LAP_OFFSET + idx * RA_LAP_WIDTH
        lap = parse_tenth(raw[start : start + RA_LAP_WIDTH])
        if lap is None:
            continue
        laps.append(lap)
    return laps


def parse_ra_record(raw: bytes, source_file: Path, source_line_no: int) -> dict[str, Any] | None:
    if not raw.startswith(b"RA"):
        return None
    race_id = raw[RA_RACE_ID_OFFSET : RA_RACE_ID_OFFSET + RA_RACE_ID_LEN].decode("ascii", errors="ignore")
    if len(race_id) != RA_RACE_ID_LEN or not race_id.isdigit():
        return None

    distance_m = parse_int(raw[RA_DISTANCE_OFFSET : RA_DISTANCE_OFFSET + RA_DISTANCE_LEN])
    track_code = raw[RA_TRACK_CODE_OFFSET : RA_TRACK_CODE_OFFSET + RA_TRACK_CODE_LEN].decode("ascii", errors="ignore")
    course_code = raw[RA_COURSE_CODE_OFFSET : RA_COURSE_CODE_OFFSET + RA_COURSE_CODE_LEN].decode("ascii", errors="ignore").strip()
    turf_going_code = raw[RA_TURF_GOING_OFFSET : RA_TURF_GOING_OFFSET + 1].decode("ascii", errors="ignore")
    dirt_going_code = raw[RA_DIRT_GOING_OFFSET : RA_DIRT_GOING_OFFSET + 1].decode("ascii", errors="ignore")
    weather_code = raw[RA_WEATHER_OFFSET : RA_WEATHER_OFFSET + 1].decode("ascii", errors="ignore")

    laps = parse_laps(raw)
    if not laps:
        return None

    surface = TRACK_SURFACE.get(track_code, "")
    going_code = dirt_going_code if surface == "ダート" else turf_going_code
    row: dict[str, Any] = {
        "race_id": race_id,
        "race_date": f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}",
        "venue_code": race_id[8:10],
        "meeting_no": race_id[10:12],
        "day_no": race_id[12:14],
        "race_no": int(race_id[14:16]),
        "race_name": decode_text(raw[RA_RACE_NAME_OFFSET : RA_RACE_NAME_OFFSET + RA_RACE_NAME_LEN]),
        "distance_m": distance_m,
        "track_code": track_code,
        "surface": surface,
        "course_code": course_code,
        "weather_code": weather_code,
        "weather": WEATHER_CODE.get(weather_code, ""),
        "turf_going_code": turf_going_code,
        "turf_going": GOING_CODE.get(turf_going_code, ""),
        "dirt_going_code": dirt_going_code,
        "dirt_going": GOING_CODE.get(dirt_going_code, ""),
        "going_code": going_code,
        "going": GOING_CODE.get(going_code, ""),
        "jv_front_3f_sec": parse_tenth(raw[RA_FRONT_3F_OFFSET : RA_FRONT_3F_OFFSET + 3]),
        "jv_front_4f_sec": parse_tenth(raw[RA_FRONT_4F_OFFSET : RA_FRONT_4F_OFFSET + 3]),
        "jv_last_3f_sec": parse_tenth(raw[RA_LAST_3F_OFFSET : RA_LAST_3F_OFFSET + 3]),
        "jv_last_4f_sec": parse_tenth(raw[RA_LAST_4F_OFFSET : RA_LAST_4F_OFFSET + 3]),
        "source_file": str(source_file),
        "source_line_no": source_line_no,
    }
    row.update(lap_shape_features(laps, distance_m))
    if row.get("jv_front_3f_sec") is not None and pd.notna(row.get("first_3f_sec")):
        row["jv_front_3f_delta_sec"] = round(float(row["jv_front_3f_sec"]) - float(row["first_3f_sec"]), 3)
    if row.get("jv_last_3f_sec") is not None and pd.notna(row.get("last_3f_sec")):
        row["jv_last_3f_delta_sec"] = round(float(row["jv_last_3f_sec"]) - float(row["last_3f_sec"]), 3)
    return row


def iter_sr_files(target_root: Path, start_year: int | None, end_year: int | None) -> list[Path]:
    paths: list[Path] = []
    if start_year is None and end_year is None:
        return sorted(target_root.rglob("SR*.DAT"))
    for year_dir in sorted(p for p in target_root.iterdir() if p.is_dir() and p.name.isdigit()):
        year = int(year_dir.name)
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        paths.extend(sorted(year_dir.glob("SR*.DAT")))
    return paths


def extract(target_root: Path, start_year: int | None, end_year: int | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in iter_sr_files(target_root, start_year, end_year):
        try:
            records = path.read_bytes().splitlines()
        except OSError:
            continue
        for line_no, raw in enumerate(records, start=1):
            row = parse_ra_record(raw, path, line_no)
            if row is not None:
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["race_id", "source_file", "source_line_no"], kind="mergesort")
        df = df.drop_duplicates("race_id", keep="last")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract all race-level lap features from TARGET/JRA-VAN RA SR*.DAT files.")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    target_root = project_path(args.target_root)
    output_csv = project_path(args.output_csv)
    summary_json = project_path(args.summary_json)
    df = extract(target_root, args.start_year, args.end_year)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "target_root": str(target_root),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "output_csv": str(output_csv),
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()) if not df.empty else 0,
        "date_min": str(df["race_date"].min()) if not df.empty else None,
        "date_max": str(df["race_date"].max()) if not df.empty else None,
        "surface_counts": df.get("surface", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
        if not df.empty
        else {},
        "shape_counts": df.get("official_lap_shape", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
        if not df.empty
        else {},
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not df.empty:
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
