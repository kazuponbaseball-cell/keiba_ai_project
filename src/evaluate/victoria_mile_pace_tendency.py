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
    "確定着順",
    "人気",
    "脚質",
    "4角.1",
    "上り3F",
    "上り3F順",
    "PCI",
    "RPCI",
    "Ave-3F",
    "走破タイム.1",
]


def load_source(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(project_path(csv_path), encoding="cp932", usecols=COLUMNS, low_memory=False)
    for col in ["日付", "確定着順", "人気", "4角.1", "上り3F", "上り3F順", "PCI", "RPCI", "Ave-3F"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["確定着順"].notna()].copy()


def bias_label(avg_corner4: float, winner_style: str) -> str:
    if pd.notna(avg_corner4):
        if avg_corner4 >= 8.5:
            return "差し寄り"
        if avg_corner4 <= 5.0:
            return "前寄り"
    if winner_style == "後方":
        return "差し寄り"
    if winner_style in {"逃げ", "先行"}:
        return "前寄り"
    return "中団差し"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Victoria Mile PCI/RPCI and running-style tendency.")
    parser.add_argument("--csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    df = load_source(args.csv)
    vm = df[df["レース名"].astype(str) == "ヴィクトG1"].copy()
    vm = vm.sort_values(["日付", "確定着順"])

    yearly_rows = []
    for date_s, part in vm.groupby("日付S", sort=True):
        winner = part[part["確定着順"] == 1].iloc[0]
        top3 = part[part["確定着順"] <= 3]
        yearly_rows.append(
            {
                "日付S": date_s,
                "winner": winner["馬名"],
                "winner_popularity": winner["人気"],
                "winner_style": winner["脚質"],
                "winner_corner4": winner["4角.1"],
                "winner_agari3f_rank": winner["上り3F順"],
                "race_rpci": float(part["RPCI"].dropna().iloc[0]) if part["RPCI"].notna().any() else None,
                "winner_pci": winner["PCI"],
                "winner_ave3f": winner["Ave-3F"],
                "winner_agari3f": winner["上り3F"],
                "winning_time": winner["走破タイム.1"],
                "top3_avg_corner4": float(top3["4角.1"].mean()),
                "top3_avg_agari3f_rank": float(top3["上り3F順"].mean()),
                "top3_style_mix": " / ".join(top3["脚質"].fillna("NA").astype(str).tolist()),
            }
        )

    yearly = pd.DataFrame(yearly_rows)
    yearly["tendency"] = [bias_label(c4, style) for c4, style in zip(yearly["top3_avg_corner4"], yearly["winner_style"])]

    overall = {
        "avg_rpci": float(yearly["race_rpci"].mean()),
        "avg_winner_pci": float(yearly["winner_pci"].mean()),
        "avg_top3_corner4": float(yearly["top3_avg_corner4"].mean()),
        "avg_top3_agari3f_rank": float(yearly["top3_avg_agari3f_rank"].mean()),
        "tendency_counts": yearly["tendency"].value_counts().to_dict(),
    }

    output_dir = ensure_dir(project_path(args.output_dir))
    run_dir = ensure_dir(output_dir / f"victoria_mile_pace_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    yearly.to_csv(run_dir / "victoria_mile_pace_yearly.csv", index=False, encoding="utf-8-sig")
    with (run_dir / "victoria_mile_pace_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"run_dir": str(run_dir), "overall": overall}, f, ensure_ascii=False, indent=2)

    print(json.dumps({"run_dir": str(run_dir), "overall": overall}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
