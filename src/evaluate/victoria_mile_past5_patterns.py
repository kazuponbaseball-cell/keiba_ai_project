from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime

import pandas as pd

from src.utils.paths import ensure_dir, project_path


COLUMNS = [
    "日付",
    "日付S",
    "レース名",
    "クラス名",
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


def add_history_columns(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    ordered = df.sort_values(["血統登録番号", "日付", "レースID(新/馬番無)", "馬番"], kind="mergesort").copy()
    grouped = ordered.groupby("血統登録番号", sort=False)
    base_cols = ["レース名", "クラス名", "場所", "芝・ダ", "距離", "確定着順", "人気", "脚質", "4角.1", "上り3F順", "日付"]
    for lag in range(1, max_lag + 1):
        for source in base_cols:
            ordered[f"{source}_lag{lag}"] = grouped[source].shift(lag)
    return ordered


def _grade_of(name: str, class_name: str) -> str:
    s = f"{name} {class_name}"
    if "G1" in s:
        return "G1"
    if "G2" in s:
        return "G2"
    if "G3" in s:
        return "G3"
    return "non-graded"


def build_past5_features(vm: pd.DataFrame) -> pd.DataFrame:
    out = vm.copy()
    out["top3"] = (out["確定着順"] <= 3).astype(int)
    out["win"] = (out["確定着順"] == 1).astype(int)

    feature_rows: list[dict] = []
    for _, row in out.iterrows():
        races = []
        for lag in range(1, 6):
            name = row.get(f"レース名_lag{lag}")
            if pd.isna(name):
                continue
            class_name = row.get(f"クラス名_lag{lag}")
            place = row.get(f"場所_lag{lag}")
            surface = row.get(f"芝・ダ_lag{lag}")
            distance = row.get(f"距離_lag{lag}")
            finish = row.get(f"確定着順_lag{lag}")
            pop = row.get(f"人気_lag{lag}")
            style = row.get(f"脚質_lag{lag}")
            corner4 = row.get(f"4角.1_lag{lag}")
            final3f_rank = row.get(f"上り3F順_lag{lag}")
            races.append(
                {
                    "name": str(name),
                    "class_name": "" if pd.isna(class_name) else str(class_name),
                    "place": "" if pd.isna(place) else str(place),
                    "surface": "" if pd.isna(surface) else str(surface),
                    "distance": None if pd.isna(distance) else float(distance),
                    "finish": None if pd.isna(finish) else float(finish),
                    "popularity": None if pd.isna(pop) else float(pop),
                    "style": "" if pd.isna(style) else str(style),
                    "corner4": None if pd.isna(corner4) else float(corner4),
                    "final3f_rank": None if pd.isna(final3f_rank) else float(final3f_rank),
                    "grade": _grade_of("" if pd.isna(name) else str(name), "" if pd.isna(class_name) else str(class_name)),
                }
            )

        runs = len(races)
        top3_count = sum(1 for r in races if r["finish"] is not None and r["finish"] <= 3)
        win_count = sum(1 for r in races if r["finish"] == 1)
        tokyo_runs = sum(1 for r in races if r["place"] == "東京")
        tokyo_top3 = sum(1 for r in races if r["place"] == "東京" and r["finish"] is not None and r["finish"] <= 3)
        turf1600_runs = sum(1 for r in races if r["surface"] == "芝" and r["distance"] == 1600)
        turf1600_top3 = sum(1 for r in races if r["surface"] == "芝" and r["distance"] == 1600 and r["finish"] is not None and r["finish"] <= 3)
        tokyo1600_runs = sum(1 for r in races if r["place"] == "東京" and r["surface"] == "芝" and r["distance"] == 1600)
        tokyo1600_top3 = sum(1 for r in races if r["place"] == "東京" and r["surface"] == "芝" and r["distance"] == 1600 and r["finish"] is not None and r["finish"] <= 3)
        graded_runs = sum(1 for r in races if r["grade"] in {"G1", "G2", "G3"})
        g1_runs = sum(1 for r in races if r["grade"] == "G1")
        g1_top3 = sum(1 for r in races if r["grade"] == "G1" and r["finish"] is not None and r["finish"] <= 3)
        g2_runs = sum(1 for r in races if r["grade"] == "G2")
        g2_top3 = sum(1 for r in races if r["grade"] == "G2" and r["finish"] is not None and r["finish"] <= 3)
        best_finish = min((r["finish"] for r in races if r["finish"] is not None), default=None)
        best_final3f_rank = min((r["final3f_rank"] for r in races if r["final3f_rank"] is not None), default=None)
        avg_pop = pd.Series([r["popularity"] for r in races]).dropna().mean() if races else None

        feature_rows.append(
            {
                "past5_runs": runs,
                "past5_top3_count": top3_count,
                "past5_win_count": win_count,
                "past5_top3_rate": 0.0 if runs == 0 else top3_count / runs,
                "past5_tokyo_runs": tokyo_runs,
                "past5_tokyo_top3": tokyo_top3,
                "past5_turf1600_runs": turf1600_runs,
                "past5_turf1600_top3": turf1600_top3,
                "past5_tokyo1600_runs": tokyo1600_runs,
                "past5_tokyo1600_top3": tokyo1600_top3,
                "past5_graded_runs": graded_runs,
                "past5_g1_runs": g1_runs,
                "past5_g1_top3": g1_top3,
                "past5_g2_runs": g2_runs,
                "past5_g2_top3": g2_top3,
                "past5_best_finish": best_finish,
                "past5_best_final3f_rank": best_final3f_rank,
                "past5_avg_popularity": avg_pop,
                "has_tokyo_top3": int(tokyo_top3 > 0),
                "has_turf1600_top3": int(turf1600_top3 > 0),
                "has_tokyo1600_top3": int(tokyo1600_top3 > 0),
                "has_g1_top3": int(g1_top3 > 0),
                "has_g2_top3": int(g2_top3 > 0),
            }
        )

    feature_df = pd.DataFrame(feature_rows, index=out.index)
    return pd.concat([out, feature_df], axis=1)


def segment_rates(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    overall_top3_rate = df["top3"].mean()
    overall_win_rate = df["win"].mean()
    for col in feature_cols:
        if df[col].dropna().nunique() <= 1:
            continue
        if set(df[col].dropna().unique()).issubset({0, 1}):
            groups = [(1, df[df[col] == 1]), (0, df[df[col] == 0])]
        else:
            median = df[col].median()
            groups = [(f">={median:g}", df[df[col] >= median]), (f"<{median:g}", df[df[col] < median])]
        for label, part in groups:
            if len(part) == 0:
                continue
            rows.append(
                {
                    "feature": col,
                    "segment": str(label),
                    "rows": int(len(part)),
                    "top3_rate": float(part["top3"].mean()),
                    "win_rate": float(part["win"].mean()),
                    "top3_lift": float(part["top3"].mean() / overall_top3_rate) if overall_top3_rate else None,
                    "win_lift": float(part["win"].mean() / overall_win_rate) if overall_win_rate else None,
                }
            )
    return pd.DataFrame(rows).sort_values(["win_lift", "top3_lift"], ascending=False)


def past_race_name_lift(df: pd.DataFrame) -> pd.DataFrame:
    all_counter: Counter[str] = Counter()
    top3_counter: Counter[str] = Counter()
    win_counter: Counter[str] = Counter()

    for _, row in df.iterrows():
        names = set()
        for lag in range(1, 6):
            name = row.get(f"レース名_lag{lag}")
            if pd.isna(name):
                continue
            names.add(str(name))
        for name in names:
            all_counter[name] += 1
            if row["top3"] == 1:
                top3_counter[name] += 1
            if row["win"] == 1:
                win_counter[name] += 1

    all_n = len(df)
    top3_n = int(df["top3"].sum())
    win_n = int(df["win"].sum())
    rows = []
    for name, support in all_counter.items():
        if support < 3:
            continue
        top3_rate = top3_counter[name] / support
        win_rate = win_counter[name] / support
        rows.append(
            {
                "past_race_name": name,
                "support": support,
                "top3_count": top3_counter[name],
                "win_count": win_counter[name],
                "top3_rate": top3_rate,
                "win_rate": win_rate,
                "top3_lift": top3_rate / (top3_n / all_n),
                "win_lift": win_rate / (win_n / all_n),
            }
        )
    return pd.DataFrame(rows).sort_values(["win_lift", "top3_lift", "support"], ascending=[False, False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore Victoria Mile patterns using past five starts.")
    parser.add_argument("--csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    df = load_source_frame(args.csv)
    df = df[df["確定着順"].notna()].copy()
    df = add_history_columns(df, max_lag=5)

    vm = df[df["レース名"].astype(str) == "ヴィクトG1"].copy()
    if vm.empty:
        raise ValueError("Victoria Mile rows were not found in the source CSV.")
    vm = build_past5_features(vm)

    feature_cols = [
        "past5_runs",
        "past5_top3_count",
        "past5_top3_rate",
        "past5_tokyo_runs",
        "past5_tokyo_top3",
        "past5_turf1600_runs",
        "past5_turf1600_top3",
        "past5_tokyo1600_runs",
        "past5_tokyo1600_top3",
        "past5_graded_runs",
        "past5_g1_runs",
        "past5_g1_top3",
        "past5_g2_runs",
        "past5_g2_top3",
        "past5_best_finish",
        "past5_best_final3f_rank",
        "has_tokyo_top3",
        "has_turf1600_top3",
        "has_tokyo1600_top3",
        "has_g1_top3",
        "has_g2_top3",
    ]
    feature_summary = segment_rates(vm, feature_cols)
    race_lift = past_race_name_lift(vm)

    output_dir = ensure_dir(project_path(args.output_dir))
    stem = f"victoria_mile_past5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(output_dir / stem)

    vm.sort_values(["日付", "確定着順"]).to_csv(run_dir / "victoria_mile_past5_rows.csv", index=False, encoding="utf-8-sig")
    feature_summary.to_csv(run_dir / "victoria_mile_past5_feature_summary.csv", index=False, encoding="utf-8-sig")
    race_lift.to_csv(run_dir / "victoria_mile_past5_race_lift.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_dir": str(run_dir),
        "sample_rows": int(len(vm)),
        "sample_races": int(vm["日付S"].nunique()),
        "top_feature_segments": feature_summary.head(20).to_dict("records"),
        "top_past_races": race_lift.head(20).to_dict("records"),
    }
    with (run_dir / "victoria_mile_past5_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
