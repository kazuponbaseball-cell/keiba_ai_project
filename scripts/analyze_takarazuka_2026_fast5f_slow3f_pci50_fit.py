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
from scripts.analyze_takarazuka_2026_fast_sustained_5f_fit import add_fast_sustained_5f  # noqa: E402
from scripts.analyze_takarazuka_character_fit import (  # noqa: E402
    build_history_features,
    load_raw_results,
    record_text,
)


PCI_MIN = 50.0
PCI_MAX = 52.9


def record(starts: int, top3: int) -> str:
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
        slow3_mask = group["last3f"].ge(35.0).fillna(False).astype(bool)
        pci_mask = group["pci"].between(PCI_MIN, PCI_MAX).fillna(False)
        combo_mask = fast_mask & slow3_mask & pci_mask
        combo = group[combo_mask].copy()
        non_combo = group[~combo_mask].copy()

        fast_slow = group[fast_mask & slow3_mask].copy()
        pci_like = group[pci_mask].copy()
        combo_good = combo[combo["top3"]].copy()
        graded_combo_good = combo_good[combo_good["graded_race"].fillna(False)].copy()
        no_breather_combo_good = combo_good[combo_good["no_breather_race"].fillna(False)].copy()
        scope_combo_good = combo_good[
            combo_good["corner4_rate"].le(0.65).fillna(False)
            | combo_good["corner3_to_4_gain"].ge(2).fillna(False)
            | combo_good["final3f_rank"].le(5).fillna(False)
        ].copy()

        starts = len(combo)
        top3 = int(combo["top3"].sum()) if starts else 0
        rate = top3 / starts if starts else math.nan
        non_rate = int(non_combo["top3"].sum()) / len(non_combo) if len(non_combo) else math.nan
        fast_slow_rate = (
            int(fast_slow["top3"].sum()) / len(fast_slow) if len(fast_slow) else math.nan
        )
        pci_rate = int(pci_like["top3"].sum()) / len(pci_like) if len(pci_like) else math.nan
        avg_finish = combo["finish"].mean() if starts else math.nan
        avg_pop = combo["popularity"].mean() if starts else math.nan
        pop_gain = (combo["popularity"] - combo["finish"]).mean() if starts else math.nan
        weighted_score = float(combo_good["grade_weight"].sum()) if not combo_good.empty else 0.0
        graded_score = float(graded_combo_good["grade_weight"].sum()) if not graded_combo_good.empty else 0.0
        no_breather_score = (
            float(no_breather_combo_good["grade_weight"].sum()) if not no_breather_combo_good.empty else 0.0
        )

        score = 0.0
        score += min(graded_score, 7.0) * 0.95
        score += min(no_breather_score, 5.0) * 0.60
        score += min(max(weighted_score - graded_score, 0.0), 4.0) * 0.20
        score += 0.7 if len(scope_combo_good) >= 2 else 0.0
        score += 0.5 if starts >= 2 and pd.notna(rate) and rate >= 0.50 else 0.0
        if pd.notna(rate) and pd.notna(non_rate) and rate > non_rate:
            score += 0.4

        rows.append(
            {
                "馬名": horse_name,
                "horse_id": horse_id,
                "判定対象戦数": len(group),
                "複合PCI条件戦数": starts,
                "複合PCI条件馬券内数": top3,
                "複合PCI条件馬券内率": round(rate, 4) if pd.notna(rate) else pd.NA,
                "非複合PCI条件馬券内率": round(non_rate, 4) if pd.notna(non_rate) else pd.NA,
                "高速5F×35秒条件馬券内率": (
                    round(fast_slow_rate, 4) if pd.notna(fast_slow_rate) else pd.NA
                ),
                "PCI50-52台馬券内率": round(pci_rate, 4) if pd.notna(pci_rate) else pd.NA,
                "複合PCI条件平均着順": round(avg_finish, 2) if pd.notna(avg_finish) else pd.NA,
                "複合PCI条件平均人気": round(avg_pop, 2) if pd.notna(avg_pop) else pd.NA,
                "複合PCI条件人気着順差": round(pop_gain, 2) if pd.notna(pop_gain) else pd.NA,
                "重賞複合PCI好走数": len(graded_combo_good),
                "重賞複合PCI重み": round(graded_score, 2),
                "息入りにくい複合PCI重み": round(no_breather_score, 2),
                "射程/押し上げ/上がり対応数": len(scope_combo_good),
                "適性スコア": round(score, 2),
                "評価": grade_score_label(score),
                "根拠レース": short_examples(
                    combo_good.sort_values(["grade_weight", "date"], ascending=[False, False])
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

    detail_path = out_dir / "takarazuka_kinen_2026_fast5f_slow3f_pci50_fit_summary.csv"
    history_path = out_dir / "takarazuka_kinen_2026_fast5f_slow3f_pci50_fit_histories.csv"
    report_path = out_dir / "takarazuka_kinen_2026_fast5f_slow3f_pci50_fit_report.md"

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
        "# 宝塚記念2026 想定馬 高速持続5F・上がり35秒以上・PCI50-52台適性",
        "",
        f"複合条件: 芝1800m以上で、後半5Fが60.0秒以内、後半5F内に11秒台が4つ以上、レース上がり3Fが35.0秒以上、かつPCI {PCI_MIN:.1f}-{PCI_MAX:.1f}。",
        "狙い: 速いロンスパを踏み、最後はタフになりつつ、馬自身の走破バランスが極端な差し/消耗に寄りすぎないレースでの対応を見る。",
        "",
        "## ランキング",
        "",
        "|評価|馬名|スコア|複合PCI成績|重賞重み|息入り重み|平均着順|人気着順差|根拠|",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for _, row in ranked.iterrows():
        avg_finish = f"{float(row['複合PCI条件平均着順']):.2f}" if pd.notna(row["複合PCI条件平均着順"]) else ""
        pop_gain = f"{float(row['複合PCI条件人気着順差']):.2f}" if pd.notna(row["複合PCI条件人気着順差"]) else ""
        lines.append(
            f"|{row['評価']}|{row['馬名']}|{float(row['適性スコア']):.2f}|"
            f"{record(int(row['複合PCI条件戦数']), int(row['複合PCI条件馬券内数']))}|"
            f"{float(row['重賞複合PCI重み']):.1f}|{float(row['息入りにくい複合PCI重み']):.1f}|"
            f"{avg_finish}|{pop_gain}|{str(row['根拠レース']).replace('|', '/')[:160]}|"
        )

    lines.extend(
        [
            "",
            "## 読み取り",
            "",
            "- PCI条件まで加えると、2025宝塚のメイショウタバルや有馬記念のダノンデサイルはPCI帯から外れるため、該当から落ちる。",
            "- 残るのは、レース質はタフ持続だが馬自身のPCIが50-52台に収まったタイプ。条件適性の純度は上がる一方、サンプルはかなり小さくなる。",
            "- SE_DATA補完分はPCI/RPCI未補完のため、大阪杯・天皇賞春はこのPCI条件では加点されない。",
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
