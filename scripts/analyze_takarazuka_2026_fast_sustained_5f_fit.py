from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_takarazuka_2025_benchmark import add_2025_benchmark_flags  # noqa: E402
from scripts.analyze_takarazuka_2026_expected_runners import (  # noqa: E402
    EXPECTED_RUNNERS,
    TARGET_DATE,
    grade_score_label,
    short_examples,
    supplemental_target_rows,
)
from scripts.analyze_takarazuka_character_fit import (  # noqa: E402
    build_history_features,
    load_raw_results,
    record_text,
)
from scripts.analyze_takarazuka_longspurt_aptitude import load_ra_records  # noqa: E402
from src.features.longspurt import parse_laps  # noqa: E402


def add_fast_sustained_5f(history: pd.DataFrame) -> pd.DataFrame:
    races = load_ra_records(2016, 2026)
    rows: list[dict[str, object]] = []
    for _, row in races.iterrows():
        laps = parse_laps(row.get("race_laps", ""))
        last5 = laps[-5:] if len(laps) >= 5 else []
        eleven_count = sum(1 for lap in last5 if 11.0 <= lap < 12.0)
        rows.append(
            {
                "race_id": row["race_id"],
                "last5f_11sec_count": eleven_count,
                "fast_sustained_5f_race": bool(
                    len(last5) == 5 and round(sum(last5), 1) <= 60.0 and eleven_count >= 4
                ),
                "last5f_lap_shape": "-".join(f"{lap:.1f}" for lap in last5),
            }
        )
    quality = pd.DataFrame(rows).drop_duplicates("race_id")
    out = history.merge(quality, on="race_id", how="left")
    out["fast_sustained_5f_race"] = out["fast_sustained_5f_race"].fillna(False)
    out["fast_sustained_5f_good"] = out["fast_sustained_5f_race"] & out["top3"]
    return out


def race_record(starts: int, top3: int) -> str:
    rate = top3 / starts if starts else math.nan
    return record_text(starts, top3, rate)


def build_summary(history: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    name_map = (
        raw[raw["horse_name"].isin(EXPECTED_RUNNERS)][["horse_name", "horse_id"]]
        .drop_duplicates()
        .sort_values(["horse_name", "horse_id"])
    )
    latest_name_map = name_map.groupby("horse_name", as_index=False).tail(1)
    rows: list[dict[str, object]] = []

    for horse_name in EXPECTED_RUNNERS:
        match = latest_name_map[latest_name_map["horse_name"].eq(horse_name)]
        if match.empty:
            rows.append({"馬名": horse_name, "データ有無": "未照合"})
            continue

        horse_id = str(match.iloc[0]["horse_id"])
        group = history[
            history["horse_id"].astype(str).eq(horse_id)
            & history["date"].astype(str).lt(TARGET_DATE)
            & history["valid_finish"]
            & history["turf_mid"]
        ].copy()

        fast_mask = group["fast_sustained_5f_race"].fillna(False).astype(bool)
        fast = group[fast_mask].copy()
        non_fast = group[~fast_mask].copy()
        fast_good = fast[fast["top3"]].copy()
        graded_fast_good = fast_good[fast_good["graded_race"].fillna(False)].copy()
        no_breather_fast_good = fast_good[fast_good["no_breather_race"].fillna(False)].copy()
        scope_fast_good = fast_good[
            fast_good["corner4_rate"].le(0.65).fillna(False)
            | fast_good["corner3_to_4_gain"].ge(2).fillna(False)
            | fast_good["final3f_rank"].le(5).fillna(False)
        ].copy()

        starts = len(fast)
        top3 = int(fast["top3"].sum()) if starts else 0
        non_top3 = int(non_fast["top3"].sum())
        rate = top3 / starts if starts else math.nan
        non_rate = non_top3 / len(non_fast) if len(non_fast) else math.nan
        avg_finish = fast["finish"].mean() if starts else math.nan
        avg_pop = fast["popularity"].mean() if starts else math.nan
        pop_gain = (fast["popularity"] - fast["finish"]).mean() if starts else math.nan
        weighted_score = float(fast_good["grade_weight"].sum()) if not fast_good.empty else 0.0
        graded_score = float(graded_fast_good["grade_weight"].sum()) if not graded_fast_good.empty else 0.0
        no_breather_score = (
            float(no_breather_fast_good["grade_weight"].sum()) if not no_breather_fast_good.empty else 0.0
        )

        score = 0.0
        score += min(graded_score, 7.0) * 0.85
        score += min(no_breather_score, 5.0) * 0.50
        score += min(max(weighted_score - graded_score, 0.0), 4.0) * 0.20
        score += 0.7 if len(scope_fast_good) >= 2 else 0.0
        score += 0.5 if starts >= 3 and pd.notna(rate) and rate >= 0.50 else 0.0
        if pd.notna(rate) and pd.notna(non_rate) and rate > non_rate:
            score += 0.4

        rows.append(
            {
                "馬名": horse_name,
                "horse_id": horse_id,
                "判定対象戦数": len(group),
                "高速持続5F戦数": starts,
                "高速持続5F馬券内数": top3,
                "高速持続5F馬券内率": round(rate, 4) if pd.notna(rate) else pd.NA,
                "非高速持続5F馬券内率": round(non_rate, 4) if pd.notna(non_rate) else pd.NA,
                "高速持続5F平均着順": round(avg_finish, 2) if pd.notna(avg_finish) else pd.NA,
                "高速持続5F平均人気": round(avg_pop, 2) if pd.notna(avg_pop) else pd.NA,
                "高速持続5F人気着順差": round(pop_gain, 2) if pd.notna(pop_gain) else pd.NA,
                "重賞高速持続好走数": len(graded_fast_good),
                "重賞高速持続重み": round(graded_score, 2),
                "息入りにくい高速持続重み": round(no_breather_score, 2),
                "射程/押し上げ/上がり対応数": len(scope_fast_good),
                "適性スコア": round(score, 2),
                "評価": grade_score_label(score),
                "根拠レース": short_examples(
                    fast_good.sort_values(["grade_weight", "date"], ascending=[False, False])
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    out_dir = ROOT / "outputs/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_results()
    supplemental = supplemental_target_rows(raw)
    if not supplemental.empty:
        raw = (
            pd.concat([raw, supplemental], ignore_index=True)
            .drop_duplicates(["race_id", "horse_id"], keep="last")
            .copy()
        )
    history = add_fast_sustained_5f(add_2025_benchmark_flags(build_history_features(raw, 2016, 2026)))
    summary = build_summary(history, raw)

    detail_path = out_dir / "takarazuka_kinen_2026_fast_sustained_5f_fit_summary.csv"
    history_path = out_dir / "takarazuka_kinen_2026_fast_sustained_5f_fit_histories.csv"
    report_path = out_dir / "takarazuka_kinen_2026_fast_sustained_5f_fit_report.md"

    summary.sort_values(["適性スコア", "馬名"], ascending=[False, True]).to_csv(
        detail_path, index=False, encoding="utf-8-sig"
    )

    horse_ids = set(summary["horse_id"].dropna().astype(str))
    history[
        history["horse_id"].astype(str).isin(horse_ids)
        & history["valid_finish"]
        & history["turf_mid"]
        & history["date"].astype(str).lt(TARGET_DATE)
    ].sort_values(["horse_name", "date", "race_id"]).to_csv(
        history_path, index=False, encoding="utf-8-sig"
    )

    ranked = summary.sort_values(["適性スコア", "馬名"], ascending=[False, True])
    lines = [
        "# 宝塚記念2026 想定馬 高速持続5F適性",
        "",
        "判定: 芝1800m以上で、レース後半5Fが60.0秒以内、かつ後半5F内に11秒台ラップが4つ以上あるレース。",
        "狙い: 一瞬の切れではなく、11秒台を長く並べる高速持続力への対応を見る。",
        "",
        "## ランキング",
        "",
        "|評価|馬名|スコア|高速持続5F成績|重賞高速持続|息入り高速持続|平均着順|人気着順差|根拠|",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for _, row in ranked.iterrows():
        avg_finish = f"{float(row['高速持続5F平均着順']):.2f}" if pd.notna(row["高速持続5F平均着順"]) else ""
        pop_gain = f"{float(row['高速持続5F人気着順差']):.2f}" if pd.notna(row["高速持続5F人気着順差"]) else ""
        lines.append(
            f"|{row['評価']}|{row['馬名']}|{float(row['適性スコア']):.2f}|"
            f"{race_record(int(row['高速持続5F戦数']), int(row['高速持続5F馬券内数']))}|"
            f"{float(row['重賞高速持続重み']):.1f}|{float(row['息入りにくい高速持続重み']):.1f}|"
            f"{avg_finish}|{pop_gain}|{str(row['根拠レース']).replace('|', '/')[:160]}|"
        )

    lines.extend(
        [
            "",
            "## 読み取り",
            "",
            "- 11秒台が4つ以上並ぶため、単なる消耗戦ではなく、速い区間を長く踏まされるレース質として扱う。",
            "- 重賞高速持続重みが高い馬は、宝塚で後半5Fが締まった場合に信頼しやすい。",
            "- 補完分はSE_DATA由来のためPCI/RPCIは未補完。大阪杯・天皇賞春はラップ、着順、人気、4角、上がり順位、G1重みで反映した。",
            "",
            "## 出力ファイル",
            "",
            f"- 集計CSV: `{detail_path.as_posix()}`",
            f"- 過去走明細CSV: `{history_path.as_posix()}`",
            f"- レポート: `{report_path.as_posix()}`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"runners={len(summary)}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
