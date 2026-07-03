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
    supplemental_target_rows,
)
from scripts.analyze_takarazuka_character_fit import (  # noqa: E402
    build_history_features,
    load_raw_results,
    record_text,
)
from scripts.analyze_takarazuka_longspurt_aptitude import load_ra_records  # noqa: E402
from src.features.longspurt import parse_laps  # noqa: E402


def add_no_slack_rhythm(history: pd.DataFrame) -> pd.DataFrame:
    races = load_ra_records(2016, 2026)
    rows: list[dict[str, object]] = []
    for _, row in races.iterrows():
        laps = parse_laps(str(row.get("race_laps", "")))
        body = laps[1:] if len(laps) >= 2 else []
        no_slack = bool(body and max(body) < 12.5)
        rows.append(
            {
                "race_id": str(row["race_id"]),
                "no_slack_rhythm_race": no_slack,
                "max_lap_after_first": round(max(body), 1) if body else math.nan,
                "slow_lap_after_first_count": sum(1 for lap in body if lap >= 12.5),
                "race_lap_shape": "-".join(f"{lap:.1f}" for lap in laps),
            }
        )
    quality = pd.DataFrame(rows).drop_duplicates("race_id")
    out = history.copy()
    out["race_id"] = out["race_id"].astype(str)
    out = out.merge(quality, on="race_id", how="left")
    out["no_slack_rhythm_race"] = out["no_slack_rhythm_race"].fillna(False).astype(bool)
    out["no_slack_rhythm_good"] = out["no_slack_rhythm_race"] & out["top3"]
    return out


def examples(frame: pd.DataFrame, n: int = 2) -> str:
    if frame.empty:
        return ""
    scored = frame.copy()
    scored["example_score"] = (
        scored["top3"].astype(int) * 5
        + scored["graded_race"].fillna(False).astype(int) * 3
        + scored["no_breather_race"].fillna(False).astype(int) * 2
        + scored["date"].astype(str).rank(method="dense").astype(int) / 1000
    )
    rows = scored.sort_values(["example_score", "date"], ascending=[False, False]).head(n)
    parts: list[str] = []
    for _, row in rows.iterrows():
        pci = "" if pd.isna(row["pci"]) else f"PCI{row['pci']:.1f}"
        rpci = "" if pd.isna(row["rpci"]) else f"RPCI{row['rpci']:.1f}"
        c4 = "-" if pd.isna(row["corner4"]) else str(int(row["corner4"]))
        parts.append(
            f"{row['date']} {row['race_name']} {int(row['distance'])}m "
            f"{int(row['finish'])}着 {pci} {rpci} 4角{c4}"
        )
    return " / ".join(parts)


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
        rhythm = group[group["no_slack_rhythm_race"]].copy()
        non_rhythm = group[~group["no_slack_rhythm_race"]].copy()
        good = rhythm[rhythm["top3"]].copy()
        graded_good = good[good["graded_race"].fillna(False)].copy()
        no_breather_good = good[good["no_breather_race"].fillna(False)].copy()
        scope_good = good[
            good["corner4_rate"].le(0.65).fillna(False)
            | good["corner3_to_4_gain"].ge(2).fillna(False)
            | good["final3f_rank"].le(5).fillna(False)
        ].copy()

        starts = len(rhythm)
        top3 = int(rhythm["top3"].sum()) if starts else 0
        rate = top3 / starts if starts else math.nan
        non_rate = int(non_rhythm["top3"].sum()) / len(non_rhythm) if len(non_rhythm) else math.nan
        avg_finish = rhythm["finish"].mean() if starts else math.nan
        avg_pop = rhythm["popularity"].mean() if starts else math.nan
        pop_gain = (rhythm["popularity"] - rhythm["finish"]).mean() if starts else math.nan
        weighted_score = float(good["grade_weight"].sum()) if not good.empty else 0.0
        graded_score = float(graded_good["grade_weight"].sum()) if not graded_good.empty else 0.0
        no_breather_score = float(no_breather_good["grade_weight"].sum()) if not no_breather_good.empty else 0.0

        score = 0.0
        score += min(graded_score, 7.0) * 0.90
        score += min(no_breather_score, 5.0) * 0.55
        score += min(max(weighted_score - graded_score, 0.0), 4.0) * 0.18
        score += 0.7 if len(scope_good) >= 2 else 0.0
        score += 0.5 if starts >= 3 and pd.notna(rate) and rate >= 0.50 else 0.0
        if pd.notna(rate) and pd.notna(non_rate) and rate > non_rate:
            score += 0.4

        rows.append(
            {
                "馬名": horse_name,
                "horse_id": horse_id,
                "判定対象戦数": len(group),
                "淡々リズム戦数": starts,
                "淡々リズム馬券内数": top3,
                "淡々リズム馬券内率": round(rate, 4) if pd.notna(rate) else pd.NA,
                "非淡々リズム馬券内率": round(non_rate, 4) if pd.notna(non_rate) else pd.NA,
                "淡々リズム平均着順": round(avg_finish, 2) if pd.notna(avg_finish) else pd.NA,
                "淡々リズム平均人気": round(avg_pop, 2) if pd.notna(avg_pop) else pd.NA,
                "淡々リズム人気着順差": round(pop_gain, 2) if pd.notna(pop_gain) else pd.NA,
                "重賞淡々リズム好走数": len(graded_good),
                "重賞淡々リズム重み": round(graded_score, 2),
                "息入りにくい淡々重み": round(no_breather_score, 2),
                "射程/押し上げ/上がり対応数": len(scope_good),
                "適性スコア": round(score, 2),
                "評価": grade_score_label(score),
                "根拠レース": examples(good.sort_values(["grade_weight", "date"], ascending=[False, False])),
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

    history = add_no_slack_rhythm(add_2025_benchmark_flags(build_history_features(raw, 2016, 2026)))
    summary = build_summary(history, raw)

    detail_path = out_dir / "takarazuka_kinen_2026_no_slack_rhythm_fit_summary.csv"
    history_path = out_dir / "takarazuka_kinen_2026_no_slack_rhythm_fit_histories.csv"
    report_path = out_dir / "takarazuka_kinen_2026_no_slack_rhythm_fit_report.md"

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
        "# 宝塚記念2026 想定馬 淡々リズム適性",
        "",
        "定義: 芝1800m以上で、最初の1ハロンを除く全ラップが12.5秒未満。12.5秒以上の緩みが一度もないレースを「淡々リズム戦」とした。",
        "",
        "## ランキング",
        "",
        "|評価|馬名|スコア|淡々リズム成績|重賞淡々|息入り淡々|平均着順|人気着順差|根拠|",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for _, row in ranked.iterrows():
        avg_finish = f"{float(row['淡々リズム平均着順']):.2f}" if pd.notna(row["淡々リズム平均着順"]) else ""
        pop_gain = f"{float(row['淡々リズム人気着順差']):.2f}" if pd.notna(row["淡々リズム人気着順差"]) else ""
        lines.append(
            f"|{row['評価']}|{row['馬名']}|{float(row['適性スコア']):.2f}|"
            f"{record(int(row['淡々リズム戦数']), int(row['淡々リズム馬券内数']))}|"
            f"{float(row['重賞淡々リズム重み']):.1f}|{float(row['息入りにくい淡々重み']):.1f}|"
            f"{avg_finish}|{pop_gain}|{str(row['根拠レース']).replace('|', '/')[:160]}|"
        )

    lines.extend(
        [
            "",
            "## 読み取り",
            "",
            "- この条件はかなり厳しく、長距離ほど該当しにくい。該当数より、重賞で該当して好走したかを優先する。",
            "- 12.5秒以上の息が入らないため、後方待機の瞬発力だけでなく、道中の追走耐性と長く脚を使う性能を見やすい。",
            "- SE_DATA補完分はPCI/RPCI未補完。大阪杯・天皇賞春・日経賞はラップ、着順、人気、4角、上がり順位、重みで反映した。",
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
