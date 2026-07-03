from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "jra_official_results"
DEFAULT_OUT = ROOT / "data" / "processed" / "jra_official_race_laps" / "race_laps.csv"


LAP_RE = re.compile(r"<th[^>]*>\s*ハロンタイム\s*</th>\s*<td[^>]*>(.*?)</td>", re.S)
AGARI_RE = re.compile(r"<th[^>]*>\s*上り\s*</th>\s*<td[^>]*>(.*?)</td>", re.S)
COURSE_RE = re.compile(
    r'<span class="cap">\s*コース：\s*</span>\s*([\d,]+)\s*<span class="unit">\s*メートル\s*</span>\s*<span class="detail">\s*（([^）]+)）',
    re.S,
)
TAG_RE = re.compile(r"<[^>]+>")
FLOAT_RE = re.compile(r"\d{1,2}\.\d")


def clean_html_text(value: str) -> str:
    value = TAG_RE.sub("", value)
    value = value.replace("&nbsp;", " ").replace("\u3000", " ")
    return re.sub(r"\s+", " ", value).strip()


def read_text(path: Path) -> str:
    for enc in ["cp932", "shift_jis", "utf-8-sig", "utf-8"]:
        try:
            return path.read_text(encoding=enc, errors="strict")
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="cp932", errors="ignore")


def parse_laps(text: str) -> list[float]:
    m = LAP_RE.search(text)
    if not m:
        return []
    lap_text = clean_html_text(m.group(1))
    return [float(x) for x in FLOAT_RE.findall(lap_text)]


def parse_agari(text: str) -> dict[str, float]:
    m = AGARI_RE.search(text)
    if not m:
        return {}
    agari_text = clean_html_text(m.group(1))
    values: dict[str, float] = {}
    for label, sec in re.findall(r"(\dF)\s*(\d{1,2}\.\d)", agari_text):
        values[f"official_last_{label.lower()}_sec"] = float(sec)
    return values


def parse_course(text: str) -> dict[str, Any]:
    m = COURSE_RE.search(text)
    if not m:
        return {}
    distance = int(m.group(1).replace(",", ""))
    detail = clean_html_text(m.group(2))
    surface = "芝" if "芝" in detail else "ダート" if "ダート" in detail else "障害" if "障害" in detail else ""
    return {"distance_m": distance, "surface": surface, "course_detail": detail}


def race_id_from_path(path: Path) -> str:
    for parent in [path.parent, *path.parents]:
        name = parent.name
        if re.fullmatch(r"\d{16}", name):
            return name
    m = re.search(r"(\d{16})", str(path))
    return m.group(1) if m else ""


def snapshot_key(path: Path) -> str:
    m = re.search(r"(\d{8}_\d{6})", path.name)
    return m.group(1) if m else path.stat().st_mtime_ns.__str__()


def section_lengths(distance_m: int | None, lap_count: int) -> list[int]:
    if not distance_m or lap_count <= 0:
        return [200] * lap_count
    first = distance_m - 200 * (lap_count - 1)
    if first <= 0 or first > 200:
        return [200] * lap_count
    return [int(first)] + [200] * (lap_count - 1)


def time_from_start(laps: list[float], lengths: list[int], meters: int) -> float:
    remaining = float(meters)
    total = 0.0
    for lap, length in zip(laps, lengths):
        if remaining <= 0:
            break
        take = min(float(length), remaining)
        total += float(lap) * (take / float(length))
        remaining -= take
    return float(total) if remaining <= 1e-9 else np.nan


def time_from_finish(laps: list[float], lengths: list[int], meters: int) -> float:
    remaining = float(meters)
    total = 0.0
    for lap, length in zip(reversed(laps), reversed(lengths)):
        if remaining <= 0:
            break
        take = min(float(length), remaining)
        total += float(lap) * (take / float(length))
        remaining -= take
    return float(total) if remaining <= 1e-9 else np.nan


def lap_shape_features(laps: list[float], distance_m: int | None = None) -> dict[str, Any]:
    if not laps:
        return {}
    arr = np.array(laps, dtype=float)
    lengths = section_lengths(distance_m, len(laps))
    out: dict[str, Any] = {
        "lap_count": int(len(laps)),
        "distance_from_lap_m": int(sum(lengths)),
        "furlong_laps_json": json.dumps(laps, ensure_ascii=False),
        "lap_section_m_json": json.dumps(lengths, ensure_ascii=False),
        "first_1f_sec": float(arr[0]),
        "last_1f_sec": float(arr[-1]),
        "lap_mean_sec": float(arr.mean()),
        "lap_std_sec": float(arr.std(ddof=0)),
        "lap_min_sec": float(arr.min()),
        "lap_max_sec": float(arr.max()),
        "lap_range_sec": float(arr.max() - arr.min()),
    }
    for i, lap in enumerate(laps, start=1):
        out[f"lap_{i}_sec"] = float(lap)
        out[f"lap_{i}_m"] = int(lengths[i - 1])
    if len(arr) >= 2:
        out["last_2f_sec"] = time_from_finish(laps, lengths, 400)
        out["finish_1f_accel_sec"] = float(arr[-2] - arr[-1])
    if len(arr) >= 3:
        out["first_3f_sec"] = time_from_start(laps, lengths, 600)
        out["last_3f_sec"] = time_from_finish(laps, lengths, 600)
        out["front_back_3f_diff_sec"] = float(out["first_3f_sec"] - out["last_3f_sec"])
        out["finish_3f_accel_sec"] = float(arr[-3] - arr[-1])
    if len(arr) >= 4:
        out["first_4f_sec"] = time_from_start(laps, lengths, 800)
        out["last_4f_sec"] = time_from_finish(laps, lengths, 800)
        out["l1_final_1f_sec"] = time_from_finish(laps, lengths, 200)
        out["l2_final_2f_sec"] = time_from_finish(laps, lengths, 400)
        out["l3_final_3f_sec"] = time_from_finish(laps, lengths, 600)
        out["l4_final_4f_sec"] = time_from_finish(laps, lengths, 800)
        out["l1_vs_l2_prev_accel_sec"] = float(arr[-2] - arr[-1])
        out["l2_vs_l3_prev_accel_sec"] = float(arr[-3] - arr[-2])
        out["l3_vs_l4_prev_accel_sec"] = float(arr[-4] - arr[-3])
    if len(arr) >= 5:
        out["first_5f_sec"] = time_from_start(laps, lengths, 1000)
    if len(arr) >= 6:
        out["middle_lap_mean_sec"] = float(arr[3:-3].mean()) if len(arr[3:-3]) else np.nan
    # Lower front/back diff means the first 3F was faster than the last 3F.
    fb = out.get("front_back_3f_diff_sec", np.nan)
    final_accel = out.get("finish_1f_accel_sec", 0.0)
    if pd.notna(fb):
        if fb <= -1.0:
            shape = "front_loaded"
        elif fb >= 1.0 and final_accel >= 0.2:
            shape = "slow_instant"
        elif fb >= 1.0:
            shape = "slow_sustain"
        else:
            shape = "balanced"
        out["official_lap_shape"] = shape
    return out


def extract_one(path: Path) -> dict[str, Any] | None:
    text = read_text(path)
    laps = parse_laps(text)
    if not laps:
        return None
    course = parse_course(text)
    row = {
        "race_id": race_id_from_path(path),
        "source_html": str(path),
        "snapshot_key": snapshot_key(path),
    }
    row.update(course)
    row.update(lap_shape_features(laps, course.get("distance_m")))
    row.update(parse_agari(text))
    return row


def latest_files(raw_dir: Path) -> list[Path]:
    files = list(raw_dir.rglob("*.html"))
    by_race: dict[str, Path] = {}
    for path in files:
        rid = race_id_from_path(path)
        if not rid:
            continue
        current = by_race.get(rid)
        if current is None or snapshot_key(path) > snapshot_key(current):
            by_race[rid] = path
    return sorted(by_race.values(), key=lambda p: race_id_from_path(p))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract race-level furlong laps from saved JRA official result HTML.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else ROOT / args.raw_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else ROOT / args.output_csv
    rows = []
    for path in latest_files(raw_dir):
        row = extract_one(path)
        if row is not None:
            rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("race_id") if rows else pd.DataFrame()
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = {
        "raw_dir": str(raw_dir),
        "output_csv": str(output_csv),
        "html_races": len(latest_files(raw_dir)),
        "lap_rows": int(len(df)),
        "shape_counts": df.get("official_lap_shape", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
        if not df.empty
        else {},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not df.empty:
        keep = [
            "race_id",
            "lap_count",
            "distance_from_lap_m",
            "first_3f_sec",
            "last_3f_sec",
            "front_back_3f_diff_sec",
            "last_4f_sec",
            "official_lap_shape",
        ]
        print(df[[c for c in keep if c in df.columns]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
