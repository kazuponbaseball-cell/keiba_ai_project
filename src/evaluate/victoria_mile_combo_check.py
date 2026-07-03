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
    "馬名",
    "血統登録番号",
    "確定着順",
    "人気",
    "年齢",
    "場所",
    "芝・ダ",
    "距離",
]


def load_source(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(project_path(csv_path), encoding="cp932", usecols=COLUMNS, low_memory=False)
    for col in ["日付", "確定着順", "人気", "年齢", "距離"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["確定着順"].notna()].copy()


def summarize(part: pd.DataFrame, label: str) -> dict:
    return {
        "segment": label,
        "rows": int(len(part)),
        "wins": int((part["確定着順"] == 1).sum()),
        "top2": int((part["確定着順"] <= 2).sum()),
        "top3": int((part["確定着順"] <= 3).sum()),
        "win_rate": float((part["確定着順"] == 1).mean()) if len(part) else 0.0,
        "top2_rate": float((part["確定着順"] <= 2).mean()) if len(part) else 0.0,
        "top3_rate": float((part["確定着順"] <= 3).mean()) if len(part) else 0.0,
        "avg_popularity": float(part["人気"].mean()) if len(part) else None,
        "avg_age": float(part["年齢"].mean()) if len(part) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Victoria Mile combo of Oka Sho top3 history and Tokyo turf 1600 top3 history.")
    parser.add_argument("--csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    df = load_source(args.csv)

    ouka_top3_ids = set(
        df[(df["レース名"].astype(str) == "桜花賞G1") & (df["確定着順"] <= 3)]["血統登録番号"].dropna().astype(str)
    )
    tokyo1600_top3_ids = set(
        df[
            (df["場所"].astype(str) == "東京")
            & (df["芝・ダ"].astype(str) == "芝")
            & (df["距離"] == 1600)
            & (df["確定着順"] <= 3)
        ]["血統登録番号"].dropna().astype(str)
    )

    vm = df[df["レース名"].astype(str) == "ヴィクトG1"].copy()
    vm["has_ouka_top3"] = vm["血統登録番号"].astype(str).isin(ouka_top3_ids)
    vm["has_tokyo1600_top3"] = vm["血統登録番号"].astype(str).isin(tokyo1600_top3_ids)
    vm["has_both"] = vm["has_ouka_top3"] & vm["has_tokyo1600_top3"]
    vm["has_ouka_only"] = vm["has_ouka_top3"] & ~vm["has_tokyo1600_top3"]
    vm["has_tokyo1600_only"] = ~vm["has_ouka_top3"] & vm["has_tokyo1600_top3"]
    vm["has_neither"] = ~vm["has_ouka_top3"] & ~vm["has_tokyo1600_top3"]

    summaries = [
        summarize(vm[vm["has_both"]], "ouka_top3_and_tokyo1600_top3"),
        summarize(vm[vm["has_ouka_only"]], "ouka_top3_only"),
        summarize(vm[vm["has_tokyo1600_only"]], "tokyo1600_top3_only"),
        summarize(vm[vm["has_neither"]], "neither"),
        summarize(vm[vm["has_ouka_top3"]], "has_ouka_top3"),
        summarize(vm[vm["has_tokyo1600_top3"]], "has_tokyo1600_top3"),
    ]

    detail_cols = [
        "日付S",
        "馬名",
        "年齢",
        "人気",
        "確定着順",
        "has_ouka_top3",
        "has_tokyo1600_top3",
        "has_both",
    ]
    detail = vm[detail_cols].sort_values(["日付S", "確定着順", "人気"])

    output_dir = ensure_dir(project_path(args.output_dir))
    run_dir = ensure_dir(output_dir / f"victoria_mile_combo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    pd.DataFrame(summaries).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(run_dir / "detail.csv", index=False, encoding="utf-8-sig")

    result = {"run_dir": str(run_dir), "summary": summaries}
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
