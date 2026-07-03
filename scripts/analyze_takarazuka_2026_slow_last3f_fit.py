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

        slow = group[group["last3f"].ge(35.0).fillna(False)].copy()
        non_slow = group[~group["last3f"].ge(35.0).fillna(False)].copy()
        very_slow = group[group["last3f"].ge(36.0).fillna(False)].copy()
        slow_good = slow[slow["top3"]].copy()
        slow_scope_good = slow_good[
            slow_good["corner4_rate"].le(0.65).fillna(False)
            | slow_good["corner3_to_4_gain"].ge(2).fillna(False)
            | slow_good["final3f_rank"].le(5).fillna(False)
        ].copy()
        graded_slow_good = slow_good[slow_good["graded_race"].fillna(False)].copy()
        graded_very_slow_good = very_slow[
            very_slow["top3"] & very_slow["graded_race"].fillna(False)
        ].copy()

        slow_starts = len(slow)
        slow_top3 = int(slow["top3"].sum()) if slow_starts else 0
        non_slow_top3 = int(non_slow["top3"].sum())
        slow_rate = slow_top3 / slow_starts if slow_starts else math.nan
        non_slow_rate = non_slow_top3 / len(non_slow) if len(non_slow) else math.nan
        weighted_slow_score = float(slow_good["grade_weight"].sum()) if not slow_good.empty else 0.0
        graded_slow_score = (
            float(graded_slow_good["grade_weight"].sum()) if not graded_slow_good.empty else 0.0
        )
        no_breather_slow_score = float(
            slow_good.loc[slow_good["no_breather_race"].fillna(False), "grade_weight"].sum()
        )

        score = 0.0
        score += min(graded_slow_score, 7.0) * 0.80
        score += min(no_breather_slow_score, 5.0) * 0.55
        score += min(max(weighted_slow_score - graded_slow_score, 0.0), 4.0) * 0.18
        score += 0.9 if len(graded_very_slow_good) else 0.0
        score += 0.6 if len(slow_scope_good) >= 2 else 0.0
        if pd.notna(slow_rate) and pd.notna(non_slow_rate) and slow_rate > non_slow_rate:
            score += 0.5

        rows.append(
            {
                "馬名": horse_name,
                "horse_id": horse_id,
                "判定対象戦数": len(group),
                "上がり35秒以上戦数": slow_starts,
                "上がり35秒以上馬券内数": slow_top3,
                "上がり35秒以上馬券内率": round(slow_rate, 4) if pd.notna(slow_rate) else pd.NA,
                "非35秒以上馬券内率": round(non_slow_rate, 4) if pd.notna(non_slow_rate) else pd.NA,
                "上がり36秒以上好走数": int(very_slow["top3"].sum()) if len(very_slow) else 0,
                "重賞35秒以上好走数": len(graded_slow_good),
                "重賞35秒以上重み": round(graded_slow_score, 2),
                "息入りにくい35秒以上重み": round(no_breather_slow_score, 2),
                "射程/押し上げ/上がり対応数": len(slow_scope_good),
                "適性スコア": round(score, 2),
                "評価": grade_score_label(score),
                "根拠レース": short_examples(
                    slow_good.sort_values(["grade_weight", "date"], ascending=[False, False])
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
    history = add_2025_benchmark_flags(build_history_features(raw, 2016, 2026))
    summary = build_summary(history, raw)

    detail_path = out_dir / "takarazuka_kinen_2026_slow_last3f_fit_summary.csv"
    history_path = out_dir / "takarazuka_kinen_2026_slow_last3f_fit_histories.csv"
    report_path = out_dir / "takarazuka_kinen_2026_slow_last3f_fit_report.md"

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
        "# 宝塚記念2026 想定馬 レース上がり3F 35秒以上適性",
        "",
        "判定: 芝1800m以上で、レースラップのラスト3F合計が35.0秒以上。馬自身の上がり3Fではなく、レース全体の上がり3Fで判定した。",
        "加点: 35秒以上戦での馬券内、重賞での発揮、36秒以上のさらにタフな終い、息入りにくい35秒以上戦、4角射程・押し上げ・上がり順位の対応。",
        "",
        "## ランキング",
        "",
        "|評価|馬名|スコア|35秒以上成績|重賞35秒以上|息入り35秒重み|36秒以上好走|根拠|",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for _, row in ranked.iterrows():
        lines.append(
            f"|{row['評価']}|{row['馬名']}|{float(row['適性スコア']):.2f}|"
            f"{race_record(int(row['上がり35秒以上戦数']), int(row['上がり35秒以上馬券内数']))}|"
            f"{float(row['重賞35秒以上重み']):.1f}|{float(row['息入りにくい35秒以上重み']):.1f}|"
            f"{int(row['上がり36秒以上好走数'])}|{str(row['根拠レース']).replace('|', '/')[:160]}|"
        )

    lines.extend(
        [
            "",
            "## 読み取り",
            "",
            "- 宝塚記念がレース上がり35秒以上に寄るなら、速い上がり勝負よりも、G1/G2級のタフな終いで崩れず好走した馬を上に取る。",
            "- スコア上位でも条件戦中心の馬は格上げ課題を残す。重賞35秒以上重みが高い馬を本線寄りに見る。",
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
