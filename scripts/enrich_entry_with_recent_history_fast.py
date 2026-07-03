from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_date(value: Any) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if "." in text and len(text.split(".")) == 3:
        y, m, d = text.split(".")
        year = int(y) if len(y) == 4 else 2000 + int(y)
        return int(f"{year}{int(m):02d}{int(d):02d}")
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 6:
        return int("20" + digits)
    if len(digits) >= 8:
        return int(digits[:8])
    return None


def _blank_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return series.isna() | text.str.strip().isin(["", "nan", "NaN", "<NA>"])


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast previous-race enrichment for provisional entry snapshots.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--config", default="config/baseline_features.json")
    parser.add_argument("--historical-csv", default=None)
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    config = _load_config(args.config)
    data_cfg = config["data"]
    encoding = data_cfg.get("encoding", "cp932")
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]
    history_path = Path(args.historical_csv or data_cfg["historical_csv"])

    entry = pd.read_csv(args.input_csv, encoding="utf-8-sig", low_memory=False, dtype={horse_col: "string"})
    entry["_entry_date_norm"] = entry[date_col].map(_normalize_date)
    target_horses = set(entry[horse_col].dropna().astype("string"))
    max_entry_date = int(entry["_entry_date_norm"].max())

    source_to_dest = {
        "レースID(新/馬番無)": "前走レースID(新/馬番無)",
        "走破タイム": "前走走破タイム",
        "平均1Fタイム": "前走平均1Fタイム",
        "基準タイム": "前走基準タイム",
        "着差タイム": "前走着差タイム",
        "確定着順": "前走確定着順",
        "人気": "前走人気",
        "頭数": "前走頭数",
        "出走頭数": "前走出走頭数",
        "馬番": "前走馬番",
        "枠番": "前走枠番",
        "斤量": "前走斤量",
        "芝・ダ": "前芝・ダ",
        "距離": "前距離",
        "馬場状態": "前走馬場状態",
        "上3F地点差": "前走上3F地点差",
        "Ave-3F": "前走Ave-3F",
        "上り3F": "前走上り3F",
        "上り3F順": "前走上り3F順",
        "PCI": "前PCI",
        "PCI3": "前走PCI3",
        "RPCI": "前走RPCI",
        "馬体重": "前走馬体重",
        "馬体重増減": "前走馬体重増減",
        "騎手コード": "前走騎手コード",
        "トラックコード": "前走トラックコード",
    }
    usecols = list(dict.fromkeys([horse_col, date_col, *source_to_dest.keys()]))
    best_parts: list[pd.DataFrame] = []
    read_cols: list[str] | None = None

    for chunk in pd.read_csv(
        history_path,
        encoding=encoding,
        low_memory=False,
        dtype={horse_col: "string"},
        chunksize=args.chunksize,
    ):
        if read_cols is None:
            read_cols = [col for col in usecols if col in chunk.columns]
        chunk = chunk[[col for col in read_cols if col in chunk.columns]].copy()
        if horse_col not in chunk.columns or date_col not in chunk.columns:
            continue
        chunk = chunk[chunk[horse_col].astype("string").isin(target_horses)].copy()
        if chunk.empty:
            continue
        chunk["_history_date_norm"] = chunk[date_col].map(_normalize_date)
        chunk = chunk[chunk["_history_date_norm"].notna() & (chunk["_history_date_norm"] < max_entry_date)]
        if chunk.empty:
            continue
        chunk = chunk.sort_values([horse_col, "_history_date_norm"], kind="mergesort")
        best_parts.append(chunk.groupby(horse_col, as_index=False).tail(1))

    if best_parts:
        history = pd.concat(best_parts, ignore_index=True)
        history = history.sort_values([horse_col, "_history_date_norm"], kind="mergesort")
        latest = history.groupby(horse_col, as_index=False).tail(1)
    else:
        latest = pd.DataFrame(columns=[horse_col])

    out = entry.drop(columns=["_entry_date_norm"], errors="ignore").copy()
    latest = latest.set_index(horse_col, drop=False)
    filled_counts: dict[str, int] = {}
    for source, dest in source_to_dest.items():
        if source not in latest.columns:
            continue
        values = out[horse_col].astype("string").map(latest[source])
        if dest not in out.columns:
            out[dest] = pd.NA
        out[dest] = out[dest].astype("object")
        before = int(out[dest].notna().sum())
        mask = _blank_mask(out[dest])
        out.loc[mask, dest] = values[mask]
        filled_counts[dest] = int(out[dest].notna().sum() - before)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "input_rows": int(len(entry)),
                "matched_horses": int(latest[horse_col].nunique()) if horse_col in latest.columns else 0,
                "output_csv": str(output_path),
                "filled_counts": filled_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
