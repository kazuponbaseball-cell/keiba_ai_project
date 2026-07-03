from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_retrospective_bias_features import add_retrospective_bias_features


DEFAULT_CONFIG = "config/baseline_features_workout_optimized_core_same_day_bias.json"


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


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path, encoding: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={"血統登録番号": str})
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding=encoding, low_memory=False, dtype={"血統登録番号": str})


def _coalesce_blank(target: pd.Series, source: pd.Series) -> pd.Series:
    blank = target.isna() | target.astype("string").str.strip().isin(["", "nan", "NaN", "<NA>"])
    return target.where(~blank, source)


def build_recent_history_lookup(history: pd.DataFrame, *, horse_col: str, date_col: str) -> pd.DataFrame:
    hist = history.copy()
    hist["_history_date_norm"] = hist[date_col].map(_normalize_date)
    hist = hist[hist[horse_col].notna() & hist["_history_date_norm"].notna()].copy()
    hist[horse_col] = hist[horse_col].astype("string")
    hist = hist.sort_values([horse_col, "_history_date_norm", "レースID(新/馬番無)"], kind="mergesort")
    return hist


def enrich_entry(entry: pd.DataFrame, history: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_cfg = config["data"]
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]

    entry_work = entry.copy()
    entry_work["_entry_row_id"] = range(len(entry_work))
    entry_work["_entry_date_norm"] = entry_work[date_col].map(_normalize_date)
    entry_work[horse_col] = entry_work[horse_col].astype("string")

    hist_cols = [
        horse_col,
        "_history_date_norm",
        "レースID(新/馬番無)",
        "走破タイム",
        "平均1Fタイム",
        "基準タイム",
        "着差タイム",
        "確定着順",
        "人気",
        "頭数",
        "出走頭数",
        "馬番",
        "枠番",
        "斤量",
        "場所",
        "レース名",
        "クラス名",
        "所属",
        "芝・ダ",
        "距離",
        "馬場状態",
        "上3F地点差",
        "Ave-3F",
        "上り3F",
        "上り3F順",
        "PCI",
        "PCI3",
        "RPCI",
        "馬体重",
        "馬体重増減",
        "騎手コード",
        "トラックコード",
        "競走種別",
        "retro_bias_help_score",
        "retro_bias_adversity_score",
        "retro_bias_resistant_score",
        "retro_bias_excuse_score",
        "retro_bias_overhelped_score",
        "past3_retro_bias_help_score",
        "past3_retro_bias_adversity_score",
        "past3_retro_bias_resistant_score",
        "past3_retro_bias_excuse_score",
        "past3_retro_bias_overhelped_score",
    ]
    hist_cols = [col for col in hist_cols if col in history.columns]
    merged = entry_work[["_entry_row_id", horse_col, "_entry_date_norm"]].merge(
        history[hist_cols],
        on=horse_col,
        how="left",
    )
    merged = merged[merged["_history_date_norm"] < merged["_entry_date_norm"]].copy()
    merged = merged.sort_values(["_entry_row_id", "_history_date_norm"], kind="mergesort")
    latest = merged.groupby("_entry_row_id", as_index=False).tail(1).set_index("_entry_row_id")

    out = entry.copy()
    mapping = {
        "前走レースID(新/馬番無)": "レースID(新/馬番無)",
        "前走走破タイム": "走破タイム",
        "前走平均1Fタイム": "平均1Fタイム",
        "前走基準タイム": "基準タイム",
        "前走着差タイム": "着差タイム",
        "前走確定着順": "確定着順",
        "前走人気": "人気",
        "前走頭数": "頭数",
        "前走出走頭数": "出走頭数",
        "前走馬番": "馬番",
        "前走枠番": "枠番",
        "前走斤量": "斤量",
        "前走場所": "場所",
        "前走レース名": "レース名",
        "前クラス名": "クラス名",
        "前走所属": "所属",
        "前芝・ダ": "芝・ダ",
        "前距離": "距離",
        "前走馬場状態": "馬場状態",
        "前走上3F地点差": "上3F地点差",
        "前走Ave-3F": "Ave-3F",
        "前走上り3F": "上り3F",
        "前走上り3F順": "上り3F順",
        "前PCI": "PCI",
        "前走PCI3": "PCI3",
        "前走RPCI": "RPCI",
        "前走馬体重": "馬体重",
        "前走馬体重増減": "馬体重増減",
        "前走騎手コード": "騎手コード",
        "前走トラックコード": "トラックコード",
        "前走競走種別": "競走種別",
    }
    retro_mapping = {
        "prev_retro_bias_help_score": "retro_bias_help_score",
        "prev_retro_bias_adversity_score": "retro_bias_adversity_score",
        "prev_retro_bias_resistant_score": "retro_bias_resistant_score",
        "prev_retro_bias_excuse_score": "retro_bias_excuse_score",
        "prev_retro_bias_overhelped_score": "retro_bias_overhelped_score",
        "past3_retro_bias_help_score": "past3_retro_bias_help_score",
        "past3_retro_bias_adversity_score": "past3_retro_bias_adversity_score",
        "past3_retro_bias_resistant_score": "past3_retro_bias_resistant_score",
        "past3_retro_bias_excuse_score": "past3_retro_bias_excuse_score",
        "past3_retro_bias_overhelped_score": "past3_retro_bias_overhelped_score",
    }

    filled_counts: dict[str, int] = {}
    for dest, source in mapping.items():
        if source not in latest.columns:
            continue
        values = out.index.to_series().map(latest[source])
        if dest not in out.columns:
            out[dest] = pd.NA
        before = out[dest].notna().sum()
        out[dest] = _coalesce_blank(out[dest], values)
        filled_counts[dest] = int(out[dest].notna().sum() - before)

    for dest, source in retro_mapping.items():
        if source not in latest.columns:
            continue
        values = out.index.to_series().map(latest[source]).fillna(0.0)
        if dest not in out.columns:
            out[dest] = pd.NA
        before = out[dest].notna().sum()
        out[dest] = _coalesce_blank(out[dest], values)
        filled_counts[dest] = int(out[dest].notna().sum() - before)

    summary = {
        "rows": int(len(out)),
        "matched_rows": int(out.index.to_series().isin(latest.index).sum()),
        "filled_counts": filled_counts,
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill blank previous-race columns in an entry CSV from historical race results.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--historical-csv", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    encoding = config["data"].get("encoding", "cp932")
    input_path = Path(args.input_csv)
    history_path = Path(args.historical_csv or config["data"]["historical_csv"])
    entry = _read_csv(input_path, encoding)
    history_raw = pd.read_csv(history_path, encoding=encoding, low_memory=False, dtype={"血統登録番号": str})
    history_raw = add_retrospective_bias_features(history_raw, config)
    history = build_recent_history_lookup(
        history_raw,
        horse_col=config["data"]["horse_id_column"],
        date_col=config["data"]["date_column"],
    )
    enriched, summary = enrich_entry(entry, history, config)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary["input_csv"] = str(input_path)
    summary["output_csv"] = str(output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
