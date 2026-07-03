from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def pick_col(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"Missing column. Tried: {names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely overlay TARGET DE entry-card keys onto an existing entry CSV.")
    parser.add_argument("--entry-csv", required=True)
    parser.add_argument("--target-card-csv", required=True)
    parser.add_argument("--target-date", required=True, help="YYMMDD or YYYYMMDD accepted for filtering entry rows.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    entry_path = project_path(args.entry_csv)
    target_path = project_path(args.target_card_csv)
    out_path = project_path(args.output_csv)
    summary_path = project_path(args.summary_json or str(out_path.with_suffix(".summary.json")))

    entry = read_csv(entry_path)
    target = read_csv(target_path)
    target = target[target["is_active_runner"].astype(str).str.lower().isin(["true", "1"]) | target["record_type"].eq("SE1")].copy()

    race_col = pick_col(entry, ["レースID(新/馬番無)", "race_id"])
    place_col = pick_col(entry, ["場所", "venue"])
    date_col = pick_col(entry, ["日付", "date"])
    horse_name_col = pick_col(entry, ["馬名", "horse_name"])
    race_no_col = pick_col(entry, ["Ｒ", "R", "race_no"])
    horse_no_col = pick_col(entry, ["馬番", "horse_no"])
    frame_col = pick_col(entry, ["枠番", "frame_no"])
    horse_id_col = pick_col(entry, ["血統登録番号", "horse_id"])
    jockey_code_col = pick_col(entry, ["騎手コード", "jockey_code"])
    jockey_col = pick_col(entry, ["騎手", "jockey"])
    trainer_col = pick_col(entry, ["調教師コード", "trainer_code"])
    weight_col = pick_col(entry, ["斤量", "assigned_weight_kg"])

    date_text = str(args.target_date)
    target_date_yy = date_text[2:] if len(date_text) == 8 else date_text
    work = entry.copy()
    mask = work[date_col].astype(str).str.replace(r"\.0$", "", regex=True).eq(target_date_yy)
    scoped = work[mask].copy()
    scoped["_entry_order"] = scoped.index
    scoped[race_no_col] = pd.to_numeric(scoped[race_no_col], errors="coerce")

    target["race_no"] = pd.to_numeric(target["race_no"], errors="coerce")
    merged = scoped.merge(
        target,
        left_on=[place_col, race_no_col, horse_name_col],
        right_on=["venue", "race_no", "horse_name"],
        how="left",
        suffixes=("", "_target"),
    )

    matched = merged["race_id"].notna()
    for out_col, target_col in [
        (race_col, "race_id"),
        (horse_no_col, "horse_no"),
        (frame_col, "frame_no"),
        (horse_id_col, "horse_id"),
        (jockey_code_col, "jockey_code"),
        (jockey_col, "jockey_name"),
        (trainer_col, "trainer_code"),
        (weight_col, "assigned_weight_kg"),
    ]:
        values = merged.set_index("_entry_order")[target_col]
        work.loc[values.index, out_col] = values

    out_path.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "entry_csv": str(entry_path),
        "target_card_csv": str(target_path),
        "output_csv": str(out_path),
        "target_date": args.target_date,
        "entry_rows": int(len(entry)),
        "target_date_rows": int(len(scoped)),
        "matched_rows": int(matched.sum()),
        "unmatched_rows": int((~matched).sum()),
        "target_card_rows": int(len(target)),
        "target_card_races": int(target["race_id"].nunique()) if "race_id" in target else 0,
        "note": "Overlay is written to a new CSV only; source entry CSV is not modified.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
