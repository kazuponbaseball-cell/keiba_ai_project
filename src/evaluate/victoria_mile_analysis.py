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
    "場所",
    "芝・ダ",
    "距離",
    "性別",
    "年齢",
    "人気",
    "単勝オッズ",
    "確定着順",
    "枠番",
    "馬番",
    "馬名",
    "騎手",
    "脚質",
    "4角.1",
    "上り3F",
    "上り3F順",
    "血統登録番号",
    "レースID(新/馬番無)",
]


def load_source_frame(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(project_path(csv_path), encoding="cp932", usecols=COLUMNS, low_memory=False)
    for col in ["日付", "距離", "年齢", "人気", "単勝オッズ", "確定着順", "枠番", "馬番", "4角.1", "上り3F", "上り3F順"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_derived_previous_race(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["血統登録番号", "日付", "レースID(新/馬番無)", "馬番"], kind="mergesort").copy()
    grouped = ordered.groupby("血統登録番号", sort=False)
    for source, target in [
        ("レース名", "derived_prev_race_name"),
        ("場所", "derived_prev_place"),
        ("芝・ダ", "derived_prev_surface"),
        ("距離", "derived_prev_distance"),
        ("確定着順", "derived_prev_finish"),
        ("人気", "derived_prev_popularity"),
        ("単勝オッズ", "derived_prev_odds"),
        ("脚質", "derived_prev_style"),
        ("4角.1", "derived_prev_corner4"),
        ("上り3F順", "derived_prev_final3f_rank"),
        ("日付", "derived_prev_date"),
    ]:
        ordered[target] = grouped[source].shift(1)
    return ordered


def build_report(vm: pd.DataFrame) -> dict:
    top3 = vm[vm["確定着順"] <= 3].copy()
    winners = vm[vm["確定着順"] == 1].copy()

    return {
        "years": sorted(vm["日付S"].unique().tolist()),
        "field_sizes": vm.groupby("日付S").size().to_dict(),
        "averages": {
            "winner_popularity_mean": float(winners["人気"].mean()),
            "top3_popularity_mean": float(top3["人気"].mean()),
            "winner_age_mean": float(winners["年齢"].mean()),
            "top3_age_mean": float(top3["年齢"].mean()),
            "winner_corner4_mean": float(winners["4角.1"].mean()),
            "top3_corner4_mean": float(top3["4角.1"].mean()),
            "winner_final3f_rank_mean": float(winners["上り3F順"].mean()),
            "top3_final3f_rank_mean": float(top3["上り3F順"].mean()),
        },
        "winner_table": winners[["日付S", "馬名", "人気", "年齢", "枠番", "馬番", "脚質", "4角.1", "上り3F順"]].to_dict("records"),
        "top3_style_counts": {str(k): int(v) for k, v in top3["脚質"].value_counts(dropna=False).items()},
        "top3_frame_zone_counts": {
            str(k): int(v)
            for k, v in pd.cut(top3["枠番"], bins=[0, 2, 5, 8], labels=["inner", "middle", "outer"]).astype(str).value_counts().items()
        },
        "top3_popularity_zone_counts": {
            str(k): int(v)
            for k, v in pd.cut(top3["人気"], bins=[0, 3, 6, 18], labels=["fav1-3", "mid4-6", "long7+"]).astype(str).value_counts().items()
        },
        "winner_prev_race_table": winners[
            [
                "日付S",
                "馬名",
                "derived_prev_race_name",
                "derived_prev_place",
                "derived_prev_surface",
                "derived_prev_distance",
                "derived_prev_finish",
                "derived_prev_popularity",
            ]
        ].to_dict("records"),
        "top3_prev_race_counts": {str(k): int(v) for k, v in top3["derived_prev_race_name"].astype(str).value_counts().head(12).items()},
        "winner_prev_finish_counts": {str(k): int(v) for k, v in winners["derived_prev_finish"].value_counts(dropna=False).items()},
        "top3_prev_finish_counts": {str(k): int(v) for k, v in top3["derived_prev_finish"].value_counts(dropna=False).head(12).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Victoria Mile tendencies from local historical CSV.")
    parser.add_argument("--csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    df = load_source_frame(args.csv)
    df = df[df["確定着順"].notna()].copy()
    df = add_derived_previous_race(df)

    vm = df[df["レース名"].astype(str) == "ヴィクトG1"].copy()
    if vm.empty:
        raise ValueError("Victoria Mile rows were not found in the source CSV.")

    report = build_report(vm)
    output_dir = ensure_dir(project_path(args.output_dir))
    stem = f"victoria_mile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(output_dir / stem)

    vm.sort_values(["日付", "確定着順"]).to_csv(run_dir / "victoria_mile_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report["winner_prev_race_table"]).to_csv(run_dir / "victoria_mile_winner_prev_races.csv", index=False, encoding="utf-8-sig")
    with (run_dir / "victoria_mile_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({"run_dir": str(run_dir), "averages": report["averages"], "top3_prev_race_counts": report["top3_prev_race_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
