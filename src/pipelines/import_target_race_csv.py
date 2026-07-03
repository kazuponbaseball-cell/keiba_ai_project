from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.paths import ensure_dir, project_path


TARGET_RACE_COLUMNS = [
    "レースID(新/馬番無)",
    "日付",
    "日付S",
    "場所",
    "Ｒ",
    "レース名",
    "クラス名",
    "芝・ダ",
    "距離",
    "馬場状態",
    "天気",
    "頭数",
    "出走頭数",
    "トラックコード",
    "走破タイム",
    "Ave-3F",
    "上り3F",
    "PCI",
    "PCI3",
    "RPCI",
    "前3F",
    "前4F",
    "後3F",
    "後4F",
    "1角通過順",
    "2角通過順",
    "3角通過順",
    "4角通過順",
]


def load_aliases(path: str | Path) -> dict[str, list[str]]:
    alias_path = Path(path)
    if not alias_path.is_absolute():
        alias_path = project_path(str(alias_path))
    with alias_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return None


def build_race_snapshot(source: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    out = pd.DataFrame(index=source.index)
    for col in TARGET_RACE_COLUMNS:
        if col in source.columns:
            out[col] = source[col]
            continue
        picked = _pick_column(source, aliases.get(col, []))
        out[col] = picked if picked is not None else pd.NA
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TARGET race-detail CSV into a race-level snapshot.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--alias-config", default="config/target_race_aliases.json")
    parser.add_argument("--output-csv", default="data/datasets/inference/weekly/race_snapshot.csv")
    parser.add_argument("--filter-date", default=None)
    parser.add_argument("--filter-race-id", default=None)
    args = parser.parse_args()

    source_path = project_path(args.input_csv)
    source = pd.read_csv(source_path, encoding="cp932", low_memory=False)
    aliases = load_aliases(args.alias_config)
    snapshot = build_race_snapshot(source, aliases)

    if args.filter_date and "日付" in snapshot.columns:
        snapshot = snapshot[snapshot["日付"].astype("string") == str(args.filter_date)].copy()
    if args.filter_race_id and "レースID(新/馬番無)" in snapshot.columns:
        snapshot = snapshot[snapshot["レースID(新/馬番無)"].astype("string") == str(args.filter_race_id)].copy()

    if snapshot.empty:
        raise ValueError("No rows matched the provided filters.")

    output_path = project_path(args.output_csv)
    ensure_dir(output_path.parent)
    snapshot.to_csv(output_path, index=False, encoding="utf-8-sig")

    all_null = [col for col in TARGET_RACE_COLUMNS if snapshot[col].isna().all()]
    summary = {
        "input_csv": str(source_path),
        "output_csv": str(output_path),
        "rows": int(len(snapshot)),
        "all_null_columns": all_null,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
