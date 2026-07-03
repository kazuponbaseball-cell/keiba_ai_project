from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_takarazuka_longspurt_aptitude import (  # noqa: E402
    load_ra_records,
    load_su_records_for_horses,
    load_su_records_for_races,
    load_takarazuka_race_ids,
    pct,
    record_text,
)
from src.features.longspurt import parse_laps  # noqa: E402


def last7_features(races: pd.DataFrame) -> pd.DataFrame:
    out = races.copy()
    laps = out["race_laps"].apply(parse_laps)
    out["lap_count"] = laps.apply(len)
    out["last7_laps_list"] = laps.apply(lambda xs: xs[-7:] if len(xs) >= 7 else [])
    out["last7_laps"] = out["last7_laps_list"].apply(
        lambda xs: "-".join(f"{x:.1f}" for x in xs) if len(xs) == 7 else ""
    )
    out["last7_sum"] = out["last7_laps_list"].apply(lambda xs: round(sum(xs), 1) if len(xs) == 7 else np.nan)

    labels = ["R7", "R6", "R5", "R4", "R3", "R2", "R1"]
    for idx, label in enumerate(labels):
        out[f"lap_{label}"] = out["last7_laps_list"].apply(lambda xs, i=idx: xs[i] if len(xs) == 7 else np.nan)

    def shape(xs: list[float]) -> pd.Series:
        if len(xs) != 7:
            return pd.Series(
                {
                    "peak_remaining_f": pd.NA,
                    "peak_lap": np.nan,
                    "r7_to_peak_accel": np.nan,
                    "first_accel": False,
                    "smooth_to_peak_steps": 0,
                    "accel_to_peak_steps": 0,
                    "final_deceleration": np.nan,
                    "late_peak7_race": False,
                    "late_peak7_r2_decel_race": False,
                }
            )

        min_value = min(xs)
        min_indices = [idx for idx, value in enumerate(xs) if abs(value - min_value) <= 0.0001]
        peak_idx = max(min_indices)
        peak_remaining = 7 - peak_idx
        diffs = [xs[idx] - xs[idx + 1] for idx in range(6)]
        to_peak = diffs[:peak_idx]
        early_accel_window = xs[0] - min(xs[1:3])
        first_accel = xs[1] <= xs[0] + 0.05 and early_accel_window >= 0.1
        smooth_steps = sum(diff >= -0.1 for diff in to_peak)
        accel_steps = sum(diff >= 0.05 for diff in to_peak)
        required_smooth_steps = max(peak_idx - 1, 0)
        required_accel_steps = 3 if peak_remaining == 2 else 4
        r7_to_peak_accel = xs[0] - xs[peak_idx]
        final_deceleration = xs[-1] - xs[-2]

        peak_is_late = peak_remaining in (1, 2)
        smooth_to_peak = smooth_steps >= required_smooth_steps
        enough_accel_steps = accel_steps >= required_accel_steps
        enough_total_accel = r7_to_peak_accel >= 0.5
        r2_peak_decelerates = peak_remaining == 2 and final_deceleration >= 0.1
        r1_finish_peak = peak_remaining == 1
        late_peak = (
            first_accel
            and peak_is_late
            and smooth_to_peak
            and enough_accel_steps
            and enough_total_accel
            and (r2_peak_decelerates or r1_finish_peak)
        )
        return pd.Series(
            {
                "peak_remaining_f": peak_remaining,
                "peak_lap": min_value,
                "r7_to_peak_accel": round(r7_to_peak_accel, 1),
                "first_accel": first_accel,
                "smooth_to_peak_steps": smooth_steps,
                "accel_to_peak_steps": accel_steps,
                "final_deceleration": round(final_deceleration, 1),
                "late_peak7_race": bool(late_peak),
                "late_peak7_r2_decel_race": bool(late_peak and r2_peak_decelerates),
            }
        )

    shape_features = out["last7_laps_list"].apply(shape)
    out = pd.concat([out, shape_features], axis=1)
    out["late_peak7_type"] = np.select(
        [
            out["late_peak7_r2_decel_race"],
            out["late_peak7_race"] & out["peak_remaining_f"].eq(1),
        ],
        ["R2最速→R1減速", "R1最速フィニッシュ"],
        default="非該当",
    )
    return out


def summarize_results(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {"starts": 0, "unique_horses": 0, "wins": 0, "top3": 0, "avg_finish": math.nan, "top3_rate": math.nan}
    finish = pd.to_numeric(df["finish"], errors="coerce")
    return {
        "starts": len(df),
        "unique_horses": df["horse_id"].nunique(),
        "wins": int(finish.eq(1).sum()),
        "top3": int(finish.between(1, 3).sum()),
        "avg_finish": finish.mean(),
        "top3_rate": finish.between(1, 3).mean(),
    }


def summary_line(label: str, df: pd.DataFrame) -> str:
    stats = summarize_results(df)
    if not stats["starts"]:
        return f"- {label}: 該当なし"
    return (
        f"- {label}: のべ{stats['starts']}頭（ユニーク{stats['unique_horses']}頭）"
        f" / {stats['starts']}戦{stats['wins']}勝・馬券内{stats['top3']}回"
        f" / 平均{stats['avg_finish']:.1f}着 / 馬券内率{stats['top3_rate'] * 100:.1f}%"
    )


def main() -> None:
    out_dir = ROOT / "outputs/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    takarazuka_ids = set(load_takarazuka_race_ids())
    takarazuka = load_su_records_for_races(takarazuka_ids)
    takarazuka = takarazuka[pd.to_numeric(takarazuka["finish"], errors="coerce").between(1, 99)].copy()
    horse_ids = set(takarazuka["horse_id"].dropna().astype(str))
    start_year = max(2000, int(takarazuka["year"].min()) - 10)
    end_year = int(takarazuka["year"].max())

    histories = load_su_records_for_horses(horse_ids, start_year, end_year)
    races = load_ra_records(start_year, end_year)
    race_features = last7_features(races)
    race_cols = [
        "race_id",
        "venue",
        "surface",
        "distance",
        "race_name",
        "last7_laps",
        "last7_sum",
        "peak_remaining_f",
        "peak_lap",
        "r7_to_peak_accel",
        "first_accel",
        "smooth_to_peak_steps",
        "accel_to_peak_steps",
        "final_deceleration",
        "late_peak7_type",
        "late_peak7_race",
        "late_peak7_r2_decel_race",
    ]

    histories = histories.merge(race_features[race_cols], on="race_id", how="left")
    histories["有効出走"] = pd.to_numeric(histories["finish"], errors="coerce").between(1, 99)
    histories["終盤ピーク判定対象"] = (
        histories["有効出走"]
        & histories["surface"].astype(str).eq("芝")
        & pd.to_numeric(histories["distance"], errors="coerce").ge(1800)
        & histories["last7_laps"].astype(str).ne("")
    )
    histories["終盤ピーク型"] = histories["終盤ピーク判定対象"] & histories["late_peak7_race"].fillna(False)
    histories["R2最速減速型"] = histories["終盤ピーク判定対象"] & histories["late_peak7_r2_decel_race"].fillna(False)
    histories["好走"] = pd.to_numeric(histories["finish"], errors="coerce").between(1, 3)

    judged = histories[histories["終盤ピーク判定対象"]].copy()
    rows: list[dict[str, object]] = []
    for _, entry in takarazuka.sort_values(["year", "race_id", "finish"]).iterrows():
        horse_id = str(entry["horse_id"])
        target_date = str(entry["date"])
        group = judged[judged["horse_id"].astype(str).eq(horse_id) & judged["date"].astype(str).lt(target_date)].copy()
        target = group[group["終盤ピーク型"]]
        non = group[~group["終盤ピーク型"]]
        strict = group[group["R2最速減速型"]]
        target_starts = len(target)
        non_starts = len(non)
        strict_starts = len(strict)
        target_good = int(target["好走"].sum())
        non_good = int(non["好走"].sum())
        strict_good = int(strict["好走"].sum())
        target_rate = target_good / target_starts if target_starts else math.nan
        non_rate = non_good / non_starts if non_starts else math.nan
        strict_rate = strict_good / strict_starts if strict_starts else math.nan
        rate_advantage = (
            target_starts > 0
            and non_starts > 0
            and pd.notna(target_rate)
            and pd.notna(non_rate)
            and target_rate > non_rate
        )
        rows.append(
            {
                "race_id": entry["race_id"],
                "year": entry["year"],
                "date": entry["date"],
                "horse_id": horse_id,
                "馬名": entry["horse_name"],
                "宝塚記念着順": entry["finish"],
                "宝塚記念人気": entry["popularity"],
                "宝塚記念位置取り": entry["position"],
                "判定対象戦数": len(group),
                "終盤ピーク型戦数": target_starts,
                "終盤ピーク型好走数": target_good,
                "終盤ピーク型好走率": round(target_rate, 4) if pd.notna(target_rate) else pd.NA,
                "非終盤ピーク型戦数": non_starts,
                "非終盤ピーク型好走数": non_good,
                "非終盤ピーク型好走率": round(non_rate, 4) if pd.notna(non_rate) else pd.NA,
                "好走率差": round(target_rate - non_rate, 4)
                if pd.notna(target_rate) and pd.notna(non_rate)
                else pd.NA,
                "R2最速減速型戦数": strict_starts,
                "R2最速減速型好走数": strict_good,
                "R2最速減速型好走率": round(strict_rate, 4) if pd.notna(strict_rate) else pd.NA,
                "終盤ピーク対応実績": "Y" if target_good > 0 else "",
                "終盤ピーク得意": "Y" if rate_advantage else "",
                "R2最速減速対応実績": "Y" if strict_good > 0 else "",
            }
        )

    aptitude = pd.DataFrame(rows).sort_values(
        ["終盤ピーク得意", "終盤ピーク対応実績", "好走率差", "終盤ピーク型戦数"],
        ascending=[False, False, False, False],
    )

    takarazuka = takarazuka.merge(
        race_features[
            [
                "race_id",
                "last7_laps",
                "last7_sum",
                "peak_remaining_f",
                "r7_to_peak_accel",
                "final_deceleration",
                "late_peak7_type",
                "late_peak7_race",
                "late_peak7_r2_decel_race",
            ]
        ],
        on="race_id",
        how="left",
    )
    takarazuka = takarazuka.merge(
        aptitude[
            [
                "race_id",
                "horse_id",
                "判定対象戦数",
                "終盤ピーク型戦数",
                "終盤ピーク型好走数",
                "終盤ピーク型好走率",
                "非終盤ピーク型戦数",
                "非終盤ピーク型好走数",
                "非終盤ピーク型好走率",
                "好走率差",
                "R2最速減速型戦数",
                "R2最速減速型好走数",
                "R2最速減速型好走率",
                "終盤ピーク対応実績",
                "終盤ピーク得意",
                "R2最速減速対応実績",
            ]
        ],
        on=["race_id", "horse_id"],
        how="left",
    )

    response_any = takarazuka[takarazuka["終盤ピーク対応実績"].eq("Y")].copy()
    response_rate = takarazuka[takarazuka["終盤ピーク得意"].eq("Y")].copy()
    response_strict = takarazuka[takarazuka["R2最速減速対応実績"].eq("Y")].copy()
    no_response = takarazuka[~takarazuka["終盤ピーク対応実績"].eq("Y")].copy()
    takarazuka_shape = takarazuka[takarazuka["late_peak7_race"].fillna(False)].copy()
    takarazuka_non_shape = takarazuka[~takarazuka["late_peak7_race"].fillna(False)].copy()

    histories.sort_values(["horse_id", "date", "race_id"]).to_csv(
        out_dir / "takarazuka_kinen_hanshin_10_running_lines_last7_peak_flags.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aptitude.to_csv(
        out_dir / "takarazuka_kinen_hanshin_10_last7_peak_aptitude_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    response_any.to_csv(
        out_dir / "takarazuka_kinen_hanshin_10_last7_peak_response_takarazuka_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    race_features[race_features["race_id"].isin(takarazuka_ids)].sort_values("date").to_csv(
        out_dir / "takarazuka_kinen_hanshin_10_last7_peak_race_shapes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    target_races = race_features[
        race_features["surface"].eq("芝")
        & pd.to_numeric(race_features["distance"], errors="coerce").ge(1800)
        & race_features["last7_laps"].astype(str).ne("")
    ]
    lines = [
        "# 宝塚記念 阪神開催10回 残り7F終盤ピーク型 適性分析",
        "",
        "定義: TARGET RAレコードの末尾7Fを `R7-R6-R5-R4-R3-R2-R1` として見る。終盤ピーク型は、R7直後から加速し始め、R7から最速ラップまで0.5秒以上速くなり、最速がR2またはR1にあり、最速までの途中で大きく緩まないレース。R2最速の場合はR1で0.1秒以上減速していることも条件にした。判定対象は芝1800m以上、好走は3着以内。",
        "",
        f"- 判定対象レース数: {len(target_races)}",
        f"- 終盤ピーク型レース数: {int(target_races['late_peak7_race'].sum())}",
        f"- R2最速→R1減速型レース数: {int(target_races['late_peak7_r2_decel_race'].sum())}",
        "",
        "## 宝塚記念自体の残り7F形状",
        "",
        "|年|残り7F|型|R7→最速加速|最速残りF|R1減速|",
        "|---:|---|---|---:|---:|---:|",
    ]
    for _, row in (
        race_features[race_features["race_id"].isin(takarazuka_ids)]
        .assign(year=lambda df: df["date"].str.slice(0, 4).astype(int))
        .sort_values("year")
        .iterrows()
    ):
        lines.append(
            f"|{int(row['year'])}|{row['last7_laps']}|{row['late_peak7_type']}|"
            f"{float(row['r7_to_peak_accel']):.1f}|{int(row['peak_remaining_f'])}|"
            f"{float(row['final_deceleration']):.1f}|"
        )

    lines.extend(
        [
            "",
            "## 対応実績馬の宝塚記念成績",
            "",
            summary_line("終盤ピーク型で好走歴あり", response_any),
            summary_line("終盤ピーク型の好走率が非該当戦より高い", response_rate),
            summary_line("R2最速→R1減速型で好走歴あり", response_strict),
            summary_line("終盤ピーク型で好走歴なし", no_response),
            "",
            "## 宝塚記念が終盤ピーク型だった年だけ",
            "",
            summary_line(
                "終盤ピーク型で好走歴あり",
                takarazuka_shape[takarazuka_shape["終盤ピーク対応実績"].eq("Y")],
            ),
            summary_line(
                "終盤ピーク型の好走率が非該当戦より高い",
                takarazuka_shape[takarazuka_shape["終盤ピーク得意"].eq("Y")],
            ),
            summary_line(
                "終盤ピーク型で好走歴なし",
                takarazuka_shape[~takarazuka_shape["終盤ピーク対応実績"].eq("Y")],
            ),
            "",
            "### 同型年で拾えなかった馬券内馬",
            "",
            "|年|馬名|着順|人気|判定対象|終盤ピーク型出走|終盤ピーク型好走|非該当好走|",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    missed_top3 = takarazuka_shape[
        ~takarazuka_shape["終盤ピーク対応実績"].eq("Y")
        & pd.to_numeric(takarazuka_shape["finish"], errors="coerce").between(1, 3)
    ].sort_values(["year", "finish", "horse_name"])
    if missed_top3.empty:
        lines.append("|-|該当なし||||||")
    else:
        for _, row in missed_top3.iterrows():
            lines.append(
                f"|{int(row['year'])}|{row['horse_name']}|{int(row['finish'])}|"
                f"{int(row['popularity']) if pd.notna(row['popularity']) else ''}|"
                f"{int(row['判定対象戦数']) if pd.notna(row['判定対象戦数']) else 0}|"
                f"{int(row['終盤ピーク型戦数']) if pd.notna(row['終盤ピーク型戦数']) else 0}|"
                f"{int(row['終盤ピーク型好走数']) if pd.notna(row['終盤ピーク型好走数']) else 0}|"
                f"{int(row['非終盤ピーク型好走数']) if pd.notna(row['非終盤ピーク型好走数']) else 0}|"
            )

    lines.extend(
        [
            "",
            "## 宝塚記念が終盤ピーク型ではなかった年",
            "",
            summary_line(
                "終盤ピーク型で好走歴あり",
                takarazuka_non_shape[takarazuka_non_shape["終盤ピーク対応実績"].eq("Y")],
            ),
            summary_line(
                "終盤ピーク型で好走歴なし",
                takarazuka_non_shape[~takarazuka_non_shape["終盤ピーク対応実績"].eq("Y")],
            ),
            "",
            "## 終盤ピーク型で好走歴あり",
            "",
            "|年|馬名|着順|人気|位置取り|宝塚7F型|過去該当成績|非該当成績|差|R2減速型成績|",
            "|---:|---|---:|---:|---|---|---|---|---:|---|",
        ]
    )
    for _, row in response_any.sort_values(["year", "finish", "horse_name"]).iterrows():
        rate_diff = float(row["好走率差"]) * 100 if pd.notna(row["好走率差"]) else math.nan
        diff_text = f"{rate_diff:.1f}pt" if pd.notna(rate_diff) else ""
        strict_text = (
            record_text(
                int(row["R2最速減速型戦数"]),
                int(row["R2最速減速型好走数"]),
                float(row["R2最速減速型好走率"]),
            )
            if pd.notna(row["R2最速減速型好走率"])
            else "0戦"
        )
        lines.append(
            f"|{int(row['year'])}|{row['horse_name']}|{int(row['finish'])}|"
            f"{int(row['popularity']) if pd.notna(row['popularity']) else ''}|{row['position']}|"
            f"{row['late_peak7_type']}|"
            f"{record_text(int(row['終盤ピーク型戦数']), int(row['終盤ピーク型好走数']), float(row['終盤ピーク型好走率']))}|"
            f"{record_text(int(row['非終盤ピーク型戦数']), int(row['非終盤ピーク型好走数']), float(row['非終盤ピーク型好走率']))}|"
            f"{diff_text}|{strict_text}|"
        )

    lines.extend(
        [
            "",
            "## 出力ファイル",
            "",
            f"- 馬柱明細: `{(out_dir / 'takarazuka_kinen_hanshin_10_running_lines_last7_peak_flags.csv').as_posix()}`",
            f"- 馬別集計: `{(out_dir / 'takarazuka_kinen_hanshin_10_last7_peak_aptitude_summary.csv').as_posix()}`",
            f"- 対応実績馬の宝塚成績: `{(out_dir / 'takarazuka_kinen_hanshin_10_last7_peak_response_takarazuka_results.csv').as_posix()}`",
            f"- 宝塚記念レース形状: `{(out_dir / 'takarazuka_kinen_hanshin_10_last7_peak_race_shapes.csv').as_posix()}`",
        ]
    )
    report = out_dir / "takarazuka_kinen_hanshin_10_last7_peak_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"histories={len(histories)}")
    print(f"judged={len(judged)}")
    print(f"late_peak_response={len(response_any)}")
    print(f"late_peak_advantage={len(response_rate)}")
    print(f"strict_response={len(response_strict)}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
