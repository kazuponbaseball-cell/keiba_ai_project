from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _flag(value: pd.Series) -> pd.Series:
    text = value.astype("string").fillna("").str.strip()
    return text.ne("").astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract blinker/equipment features from TARGET horse-result CSV export.")
    parser.add_argument("--source-csv", default=r"C:\Users\kazup\Data Lab\TXT\全競走馬成績_utf8.csv")
    parser.add_argument("--output-csv", default="data/processed/target/equipment_features.csv")
    args = parser.parse_args()

    needed = [
        "日付S",
        "レースID(新/馬番無)",
        "馬番",
        "馬名",
        "ブリンカー",
        "前走B",
        "人気",
        "単勝オッズ",
        "確定着順",
    ]
    source = Path(args.source_csv)
    if not source.exists():
        raise FileNotFoundError(source)

    df = pd.read_csv(source, usecols=lambda c: c in needed, dtype={"レースID(新/馬番無)": str}, low_memory=False)
    df = df.rename(
        columns={
            "レースID(新/馬番無)": "race_id",
            "馬番": "horse_no",
            "馬名": "horse_name",
            "ブリンカー": "equipment_blinker_raw",
            "前走B": "prev_equipment_blinker_raw",
            "人気": "popularity",
            "単勝オッズ": "win_odds",
            "確定着順": "finish",
        }
    )
    df["race_id"] = df["race_id"].astype("string").str.replace(".0", "", regex=False).str.zfill(16)
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    df["equipment_blinker_flag"] = _flag(df.get("equipment_blinker_raw", pd.Series("", index=df.index)))
    df["prev_equipment_blinker_flag"] = _flag(df.get("prev_equipment_blinker_raw", pd.Series("", index=df.index)))
    df["equipment_first_or_reapply_blinker_flag"] = ((df["equipment_blinker_flag"] == 1) & (df["prev_equipment_blinker_flag"] == 0)).astype(int)
    df["equipment_continue_blinker_flag"] = ((df["equipment_blinker_flag"] == 1) & (df["prev_equipment_blinker_flag"] == 1)).astype(int)
    df["equipment_remove_blinker_flag"] = ((df["equipment_blinker_flag"] == 0) & (df["prev_equipment_blinker_flag"] == 1)).astype(int)
    df["equipment_change_flag"] = (df["equipment_blinker_flag"] != df["prev_equipment_blinker_flag"]).astype(int)
    df["equipment_any_signal_flag"] = (
        df["equipment_blinker_flag"]
        | df["prev_equipment_blinker_flag"]
        | df["equipment_change_flag"]
    ).astype(int)
    if "日付S" in df.columns:
        parsed_date = pd.to_datetime(df["日付S"], errors="coerce")
        df["date_key"] = parsed_date.dt.strftime("%Y%m%d")
    df["finish_num"] = pd.to_numeric(df.get("finish"), errors="coerce")
    df["equipment_is_win"] = df["finish_num"].eq(1).astype(int)
    df["equipment_is_place"] = df["finish_num"].le(3).astype(int)

    keep = [
        "race_id",
        "horse_no",
        "horse_name",
        "date_key",
        "equipment_blinker_raw",
        "prev_equipment_blinker_raw",
        "equipment_blinker_flag",
        "prev_equipment_blinker_flag",
        "equipment_first_or_reapply_blinker_flag",
        "equipment_continue_blinker_flag",
        "equipment_remove_blinker_flag",
        "equipment_change_flag",
        "equipment_any_signal_flag",
        "popularity",
        "win_odds",
        "finish_num",
        "equipment_is_win",
        "equipment_is_place",
    ]
    out = df[[c for c in keep if c in df.columns]].dropna(subset=["race_id", "horse_no"]).drop_duplicates(["race_id", "horse_no"], keep="last")
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    summary = {
        "source_csv": str(source),
        "output_csv": str(output),
        "rows": int(len(out)),
        "races": int(out["race_id"].nunique()),
        "current_blinker_rows": int(out["equipment_blinker_flag"].sum()),
        "first_or_reapply_rows": int(out["equipment_first_or_reapply_blinker_flag"].sum()),
        "remove_blinker_rows": int(out["equipment_remove_blinker_flag"].sum()),
        "date_min": str(out.get("date_key", pd.Series(dtype=str)).min()),
        "date_max": str(out.get("date_key", pd.Series(dtype=str)).max()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
