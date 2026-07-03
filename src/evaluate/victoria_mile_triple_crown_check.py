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
    "単勝オッズ",
    "性別",
    "年齢",
    "場所",
    "芝・ダ",
    "距離",
]

TRIPLE_CROWN_RACES = {"桜花賞G1", "優駿牝馬G1", "秋華賞G1"}


def load_source(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(project_path(csv_path), encoding="cp932", usecols=COLUMNS, low_memory=False)
    for col in ["日付", "確定着順", "人気", "単勝オッズ", "年齢", "距離"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["確定着順"].notna()].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Victoria Mile results for horses with past fillies triple crown top-3 finishes.")
    parser.add_argument("--csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    df = load_source(args.csv)
    triple_top3 = df[df["レース名"].isin(TRIPLE_CROWN_RACES) & (df["確定着順"] <= 3)].copy()
    qualified_ids = set(triple_top3["血統登録番号"].dropna().astype(str))

    vm = df[df["レース名"].astype(str) == "ヴィクトG1"].copy()
    vm["has_triple_crown_top3"] = vm["血統登録番号"].astype(str).isin(qualified_ids)
    vm["win"] = (vm["確定着順"] == 1).astype(int)
    vm["top2"] = (vm["確定着順"] <= 2).astype(int)
    vm["top3"] = (vm["確定着順"] <= 3).astype(int)

    summary_rows = []
    for label, part in [
        ("has_triple_crown_top3", vm[vm["has_triple_crown_top3"]]),
        ("no_triple_crown_top3", vm[~vm["has_triple_crown_top3"]]),
    ]:
        summary_rows.append(
            {
                "segment": label,
                "rows": int(len(part)),
                "races": int(part["日付S"].nunique()),
                "wins": int(part["win"].sum()),
                "top2": int(part["top2"].sum()),
                "top3": int(part["top3"].sum()),
                "win_rate": float(part["win"].mean()) if len(part) else 0.0,
                "top2_rate": float(part["top2"].mean()) if len(part) else 0.0,
                "top3_rate": float(part["top3"].mean()) if len(part) else 0.0,
                "avg_popularity": float(part["人気"].mean()) if len(part) else None,
            }
        )

    detail = (
        vm[vm["has_triple_crown_top3"]][["日付", "日付S", "確定着順", "人気", "馬名", "年齢"]]
        .sort_values(["日付", "確定着順"])
        .drop(columns=["日付"])
    )
    winners = (
        vm[vm["win"] == 1][["日付", "日付S", "馬名", "人気", "年齢", "has_triple_crown_top3"]]
        .sort_values("日付")
        .drop(columns=["日付"])
    )

    output_dir = ensure_dir(project_path(args.output_dir))
    run_dir = ensure_dir(output_dir / f"victoria_mile_triple_crown_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(run_dir / "qualified_horses_in_vm.csv", index=False, encoding="utf-8-sig")
    winners.to_csv(run_dir / "winners.csv", index=False, encoding="utf-8-sig")

    result = {
        "run_dir": str(run_dir),
        "summary": summary_rows,
        "winner_has_triple_crown_top3_count": int(winners["has_triple_crown_top3"].sum()),
        "winner_total_count": int(len(winners)),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
