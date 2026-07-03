from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from src.utils.paths import ensure_dir, project_path


COLUMNS = [
    "日付",
    "日付S",
    "レース名",
    "クラス名",
    "馬名",
    "血統登録番号",
    "確定着順",
    "人気",
    "年齢",
    "場所",
    "芝・ダ",
    "距離",
    "単勝配当",
    "複勝配当",
]


def load_source(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(project_path(csv_path), encoding="cp932", usecols=COLUMNS, low_memory=False)
    for col in ["日付", "確定着順", "人気", "年齢", "距離", "単勝配当", "複勝配当"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["確定着順"].notna()].copy()


def bucket_of_tokyo1600_race(race_name: str, class_name: str) -> str:
    text = f"{race_name} {class_name}"
    if "G1" in text:
        return "tokyo1600_g1"
    if "G2" in text or "G3" in text:
        if race_name.startswith("クイーン") or race_name.startswith("アルテミ"):
            return "tokyo1600_2yo3yo_graded"
        return "tokyo1600_g2g3"
    if "新馬" in text or "未勝利" in text:
        return "tokyo1600_maiden"
    if "1勝" in text or "2勝" in text or "3勝" in text or "1600" in text or "L" in text:
        return "tokyo1600_conditions"
    return "tokyo1600_other"


def summarize(part: pd.DataFrame, label: str) -> dict:
    rows = len(part)
    win_pay = part["単勝配当"].fillna(0.0).sum()
    place_pay = part["複勝配当"].fillna(0.0).sum()
    return {
        "segment": label,
        "rows": int(rows),
        "wins": int((part["確定着順"] == 1).sum()),
        "top2": int((part["確定着順"] <= 2).sum()),
        "top3": int((part["確定着順"] <= 3).sum()),
        "win_rate": float((part["確定着順"] == 1).mean()) if rows else 0.0,
        "top2_rate": float((part["確定着順"] <= 2).mean()) if rows else 0.0,
        "top3_rate": float((part["確定着順"] <= 3).mean()) if rows else 0.0,
        "avg_popularity": float(part["人気"].mean()) if rows else None,
        "avg_age": float(part["年齢"].mean()) if rows else None,
        "tansho_roi": float(win_pay / (rows * 100.0)) if rows else 0.0,
        "fukusho_roi": float(place_pay / (rows * 100.0)) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Break down Victoria Mile results by type of past Tokyo turf 1600 top-3 history.")
    parser.add_argument("--csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    df = load_source(args.csv)
    vm = df[df["レース名"].astype(str) == "ヴィクトG1"].copy()
    target_ids = set(vm["血統登録番号"].dropna().astype(str))

    tokyo1600 = df[
        (df["場所"].astype(str) == "東京")
        & (df["芝・ダ"].astype(str) == "芝")
        & (df["距離"] == 1600)
        & (df["確定着順"] <= 3)
        & (df["血統登録番号"].astype(str).isin(target_ids))
        & (df["レース名"].astype(str) != "ヴィクトG1")
    ].copy()
    tokyo1600["bucket"] = [
        bucket_of_tokyo1600_race(str(race_name), "" if pd.isna(class_name) else str(class_name))
        for race_name, class_name in zip(tokyo1600["レース名"], tokyo1600["クラス名"])
    ]

    history_by_horse: dict[str, set[str]] = {}
    for horse_id, part in tokyo1600.groupby(tokyo1600["血統登録番号"].astype(str)):
        history_by_horse[horse_id] = set(part["bucket"].tolist())

    bucket_names = [
        "tokyo1600_g1",
        "tokyo1600_g2g3",
        "tokyo1600_2yo3yo_graded",
        "tokyo1600_conditions",
        "tokyo1600_maiden",
        "tokyo1600_other",
    ]

    for bucket in bucket_names:
        vm[bucket] = vm["血統登録番号"].astype(str).map(lambda x: bucket in history_by_horse.get(x, set()))

    summaries = [summarize(vm[vm[bucket]], bucket) for bucket in bucket_names]
    summaries.append(summarize(vm[vm[bucket_names].any(axis=1)], "tokyo1600_any"))
    summaries.append(summarize(vm[~vm[bucket_names].any(axis=1)], "tokyo1600_none"))

    detail_rows = []
    for _, row in vm.iterrows():
        horse_id = str(row["血統登録番号"])
        horse_hist = tokyo1600[tokyo1600["血統登録番号"].astype(str) == horse_id].copy()
        if horse_hist.empty:
            continue
        detail_rows.append(
            {
                "日付S": row["日付S"],
                "馬名": row["馬名"],
                "人気": row["人気"],
                "確定着順": row["確定着順"],
                "tokyo1600_buckets": ",".join(sorted(history_by_horse.get(horse_id, set()))),
                "tokyo1600_races": " / ".join(sorted(set(horse_hist["レース名"].astype(str)))),
            }
        )
    detail = pd.DataFrame(detail_rows).sort_values(["日付S", "確定着順", "人気"])

    output_dir = ensure_dir(project_path(args.output_dir))
    run_dir = ensure_dir(output_dir / f"victoria_mile_tokyo1600_buckets_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    pd.DataFrame(summaries).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(run_dir / "detail.csv", index=False, encoding="utf-8-sig")
    tokyo1600[["日付S", "馬名", "レース名", "クラス名", "確定着順", "bucket"]].to_csv(
        run_dir / "tokyo1600_source_rows.csv", index=False, encoding="utf-8-sig"
    )

    result = {"run_dir": str(run_dir), "summary": summaries}
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
