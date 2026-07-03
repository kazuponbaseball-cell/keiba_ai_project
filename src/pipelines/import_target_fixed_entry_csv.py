from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from src.data.loaders import (
    inference_optional_columns,
    inference_required_columns,
    load_json_config,
)
from src.utils.paths import ensure_dir, project_path
from src.utils.runtime_config import load_runtime_config


FIXED_ENTRY_COLUMNS = {
    0: "日付",
    1: "場所",
    2: "Ｒ",
    3: "馬番",
    4: "レース名",
    5: "芝・ダ",
    6: "距離",
    7: "馬名",
    8: "性別",
    9: "年齢",
    10: "騎手",
    11: "斤量",
    12: "調教師",
    18: "血統登録番号",
    22: "枠番",
    23: "人気",
    26: "頭数",
    30: "単勝オッズ",
    32: "レースID(新)",
}


def looks_like_fixed_entry_csv(path: Path, encoding: str = "cp932") -> bool:
    try:
        with path.open("r", encoding=encoding, newline="") as f:
            row = next(csv.reader(f))
    except (OSError, UnicodeDecodeError, StopIteration, csv.Error):
        return False
    if len(row) < 33:
        return False
    return (
        row[1] in {"札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"}
        and row[5] in {"芝", "ダ", "障"}
        and bool(str(row[7]).strip())
        and str(row[18]).strip().isdigit()
        and str(row[32]).strip().isdigit()
    )


def _date_s(value: object) -> str | pd.NA:
    text = str(value).strip()
    if len(text) != 6 or not text.isdigit():
        return pd.NA
    year = 2000 + int(text[:2])
    return f"{year}.{int(text[2:4])}.{int(text[4:6])}"


def build_snapshot(source: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=source.index, columns=columns)
    for source_idx, dest_col in FIXED_ENTRY_COLUMNS.items():
        if source_idx in source.columns and dest_col in out.columns:
            out[dest_col] = source[source_idx]

    if "レースID(新/馬番無)" in out.columns and 32 in source.columns:
        out["レースID(新/馬番無)"] = source[32].astype("string").str.strip().str[:-2]
    if "日付S" in out.columns and "日付" in out.columns:
        out["日付S"] = out["日付"].map(_date_s)
    if "出走頭数" in out.columns and "頭数" in out.columns:
        out["出走頭数"] = out["頭数"]
    if "異常コード" in out.columns:
        out["異常コード"] = 0
    if "確定着順" in out.columns:
        out["確定着順"] = pd.NA
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a TARGET no-header fixed entry CSV into entry_snapshot.csv.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--runtime-config", default="config/data_pipeline.json")
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    runtime = load_runtime_config(args.runtime_config)
    feature_config_path = args.feature_config or runtime["pipeline"]["baseline_feature_config"]
    feature_config = load_json_config(feature_config_path)
    columns = list(dict.fromkeys([
        *inference_required_columns(feature_config),
        *inference_optional_columns(feature_config),
    ]))

    source_path = project_path(args.input_csv)
    source = pd.read_csv(source_path, header=None, encoding="cp932", low_memory=False)
    snapshot = build_snapshot(source, columns)
    output_path = project_path(args.output_csv or runtime["datasets"]["weekly_entry_file"])
    ensure_dir(output_path.parent)
    snapshot.to_csv(output_path, index=False, encoding="utf-8-sig")

    missing_required = [
        col for col in inference_required_columns(feature_config)
        if col in snapshot.columns and snapshot[col].isna().all()
    ]
    summary = {
        "input_csv": str(source_path),
        "output_csv": str(output_path),
        "rows": int(len(snapshot)),
        "missing_required_values": missing_required,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
