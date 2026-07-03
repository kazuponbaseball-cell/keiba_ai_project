from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.utils.paths import ensure_dir, project_path


LAP_COLUMNS = {
    "レース通過タイム",
    "レース上りタイム",
    "レースラップタイム",
    "レース後5Fラップタイム",
}


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _horse_name_from_html(path: Path) -> str:
    text = path.read_text(encoding="cp932", errors="replace")
    match = re.search(r"<B>([^<]+)</B>", text)
    return _clean_text(match.group(1)) if match else path.stem


def extract_laps(path: Path, race_name_contains: str | None = None) -> pd.DataFrame:
    try:
        tables = pd.read_html(path, encoding="cp932", flavor="lxml")
    except ValueError:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    horse_name = _horse_name_from_html(path)
    for table in tables:
        if not LAP_COLUMNS.issubset(set(map(str, table.columns))):
            continue
        out = table.copy()
        out.insert(0, "抽出元馬名", horse_name)
        out.insert(1, "source_file", str(path))
        if race_name_contains:
            out = out[out["レース名"].astype(str).str.contains(race_name_contains, na=False)]
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TARGET horse-page race lap tables from HTML files.")
    parser.add_argument("--input", action="append", default=None, help="HTML file or directory. Can be repeated.")
    parser.add_argument("--race-name-contains", default=None)
    parser.add_argument("--output-csv", default="outputs/analysis/target_lap_html_extract.csv")
    args = parser.parse_args()

    inputs = [Path(p) for p in (args.input or ["C:/Users/kazup/Data Lab/TXT"])]
    html_files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            html_files.extend(item.glob("*.html"))
        elif item.is_file():
            html_files.append(item)

    frames = [extract_laps(path, args.race_name_contains) for path in html_files]
    frames = [frame for frame in frames if not frame.empty]
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(
            subset=["日付", "開催", "レース名", "TR", "距離", "レースラップタイム"],
            keep="first",
        )
    else:
        out = pd.DataFrame()

    output_path = project_path(args.output_csv)
    ensure_dir(output_path.parent)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary = {
        "html_files_checked": len(html_files),
        "rows": int(len(out)),
        "output_csv": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
