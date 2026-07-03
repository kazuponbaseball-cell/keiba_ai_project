from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.paths import ensure_dir, project_path


RA_RACE_ID_OFFSET = 11
RA_RACE_ID_LEN = 16
RA_LAP_OFFSET = 890
RA_LAP_COUNT = 25
RA_LAP_WIDTH = 3


def _decode_name(raw: bytes) -> str:
    return raw.decode("cp932", errors="replace").strip().replace("\u3000", "")


def _decode_laps(raw: bytes) -> list[float]:
    laps: list[float] = []
    for idx in range(RA_LAP_COUNT):
        start = RA_LAP_OFFSET + idx * RA_LAP_WIDTH
        value = raw[start : start + RA_LAP_WIDTH].decode("ascii", errors="ignore")
        if not value.isdigit():
            continue
        number = int(value)
        if number <= 0:
            continue
        laps.append(number / 10.0)
    return laps


def _race_lap_string(laps: list[float]) -> str:
    return "-".join(f"{lap:.1f}" for lap in laps)


def _cumulative_string(laps: list[float], points: tuple[int, ...]) -> str:
    values = []
    for point in points:
        if len(laps) >= point:
            values.append(f"{sum(laps[:point]):.1f}")
    return "-".join(values)


def _last_string(laps: list[float], points: tuple[int, ...]) -> str:
    values = []
    for point in points:
        if len(laps) >= point:
            values.append(f"{sum(laps[-point:]):.1f}")
    return "-".join(values)


def extract_ra_laps(root: Path, race_name_contains: str, start_year: int, end_year: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paths: list[Path] = []
    for year in range(start_year, end_year + 1):
        year_dir = root / str(year)
        if year_dir.exists():
            paths.extend(year_dir.glob("SR*.DAT"))
    for path in sorted(paths):
        try:
            records = path.read_bytes().splitlines()
        except OSError:
            continue
        for raw in records:
            if not raw.startswith(b"RA"):
                continue
            race_id = raw[RA_RACE_ID_OFFSET : RA_RACE_ID_OFFSET + RA_RACE_ID_LEN].decode("ascii", errors="ignore")
            if len(race_id) != RA_RACE_ID_LEN or not race_id[:8].isdigit():
                continue
            year = int(race_id[:4])
            if year < start_year or year > end_year:
                continue
            race_name = _decode_name(raw[32:92])
            if race_name_contains not in race_name:
                continue
            laps = _decode_laps(raw)
            rows.append(
                {
                    "年": year,
                    "日付": race_id[:8],
                    "レースID": race_id,
                    "場所コード": race_id[8:10],
                    "回": race_id[10:12],
                    "日目": race_id[12:14],
                    "R": int(race_id[14:16]),
                    "レース名": race_name,
                    "ラップ数": len(laps),
                    "レースラップタイム": _race_lap_string(laps),
                    "レース通過タイム": _cumulative_string(laps, (3, 4, 5, 6)),
                    "レース後5Fラップタイム": _race_lap_string(laps[-5:]) if len(laps) >= 5 else "",
                    "レース上りタイム": _last_string(laps, (6, 5, 4, 3)),
                    "source_file": str(path),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(["日付", "レースID"]).drop_duplicates("レースID", keep="last")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract race lap times from TARGET RA records.")
    parser.add_argument("--target-root", default="C:/Users/kazup/Data Lab/SE_DATA")
    parser.add_argument("--race-name-contains", default="安田記念")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-csv", default="outputs/analysis/yasuda_kinen_10y_race_laps.csv")
    args = parser.parse_args()

    out = extract_ra_laps(
        Path(args.target_root),
        args.race_name_contains,
        args.start_year,
        args.end_year,
    )
    output_path = project_path(args.output_csv)
    ensure_dir(output_path.parent)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary = {
        "rows": int(len(out)),
        "output_csv": str(output_path),
        "years": out["年"].astype(int).tolist() if not out.empty and "年" in out.columns else [],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
