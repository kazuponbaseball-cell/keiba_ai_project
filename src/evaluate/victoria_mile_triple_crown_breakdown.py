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
]

TRIPLE_RACES = {
    "桜花賞G1": "ouka",
    "優駿牝馬G1": "oaks",
    "秋華賞G1": "shuka",
}


def load_source(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(project_path(csv_path), encoding="cp932", usecols=COLUMNS, low_memory=False)
    for col in ["日付", "確定着順", "人気", "年齢"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["確定着順"].notna()].copy()


def summarize_segment(part: pd.DataFrame, label: str) -> dict:
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Break down Victoria Mile results by fillies triple crown history.")
    parser.add_argument("--csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    df = load_source(args.csv)
    vm = df[df["レース名"].astype(str) == "ヴィクトG1"].copy()
    vm["vm_year"] = (vm["日付"] // 10000).astype("Int64")

    for race_name, short in TRIPLE_RACES.items():
        race_top3 = df[(df["レース名"].astype(str) == race_name) & (df["確定着順"] <= 3)].copy()
        ids_all = set(race_top3["血統登録番号"].dropna().astype(str))
        race_top3["race_year"] = (race_top3["日付"] // 10000).astype("Int64")
        prev_year_keys = set(
            zip(
                race_top3["血統登録番号"].dropna().astype(str),
                race_top3["race_year"].dropna().astype(int) + 1,
            )
        )
        vm[f"has_{short}_top3"] = vm["血統登録番号"].astype(str).isin(ids_all)
        vm[f"has_prev_year_{short}_top3"] = [
            (str(hid), int(year)) in prev_year_keys if pd.notna(year) else False
            for hid, year in zip(vm["血統登録番号"], vm["vm_year"])
        ]

    vm["has_any_triple_top3"] = vm[[f"has_{s}_top3" for s in TRIPLE_RACES.values()]].any(axis=1)
    vm["has_prev_year_any_triple_top3"] = vm[[f"has_prev_year_{s}_top3" for s in TRIPLE_RACES.values()]].any(axis=1)

    summaries = []
    for race_name, short in TRIPLE_RACES.items():
        summaries.append(summarize_segment(vm[vm[f"has_{short}_top3"]], f"{race_name}_top3"))
        summaries.append(summarize_segment(vm[vm[f"has_prev_year_{short}_top3"]], f"prev_year_{race_name}_top3"))
    summaries.append(summarize_segment(vm[vm["has_any_triple_top3"]], "any_triple_top3"))
    summaries.append(summarize_segment(vm[vm["has_prev_year_any_triple_top3"]], "prev_year_any_triple_top3"))
    summaries.append(summarize_segment(vm[(vm["年齢"] == 4) & (vm["has_prev_year_any_triple_top3"])], "4yo_prev_year_any_triple_top3"))

    detail_cols = [
        "日付S",
        "馬名",
        "年齢",
        "人気",
        "確定着順",
        "has_ouka_top3",
        "has_oaks_top3",
        "has_shuka_top3",
        "has_prev_year_ouka_top3",
        "has_prev_year_oaks_top3",
        "has_prev_year_shuka_top3",
    ]
    detail = vm[detail_cols].sort_values(["日付S", "確定着順", "人気"])

    output_dir = ensure_dir(project_path(args.output_dir))
    run_dir = ensure_dir(output_dir / f"victoria_mile_triple_crown_breakdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    pd.DataFrame(summaries).to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(run_dir / "detail.csv", index=False, encoding="utf-8-sig")

    result = {"run_dir": str(run_dir), "summary": summaries}
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
