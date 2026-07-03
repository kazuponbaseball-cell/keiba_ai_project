from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_takarazuka_character_fit import (  # noqa: E402
    build_history_features,
    load_raw_results,
    record_text,
    result_line,
    summarize_result,
)
from scripts.analyze_takarazuka_longspurt_aptitude import (  # noqa: E402
    load_su_records_for_races,
    load_takarazuka_race_ids,
)


BENCHMARK = {
    "year": 2025,
    "date": "20250615",
    "going": "稍",
    "first3f": 34.8,
    "last3f": 36.0,
    "last5f": 59.8,
    "front_minus_last": -1.2,
    "pci3": 50.1,
    "rpci": 49.1,
    "last7_type": "R2最速→R1減速",
}


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def add_2025_benchmark_flags(history: pd.DataFrame) -> pd.DataFrame:
    out = history.copy()
    out["bench2025_rpci_band"] = out["rpci"].between(46.5, 51.5)
    out["bench2025_pci_band"] = out["pci"].between(46.0, 53.5)
    out["bench2025_front_loaded"] = out["front_minus_last"].between(-2.4, -0.6)
    out["bench2025_last5f_ok"] = out["last5f_fast_percentile"].le(0.55)
    out["bench2025_late_peak_ok"] = out["late_peak7_r2_decel_race"].fillna(False)
    out["bench2025_scope_or_finish"] = (
        out["corner4_rate"].le(0.65)
        | out["corner3_to_4_gain"].ge(2)
        | out["final3f_rank"].le(5)
    )
    out["bench2025_like_race"] = (
        out["turf_mid"]
        & out["bench2025_rpci_band"].fillna(False)
        & out["bench2025_front_loaded"].fillna(False)
        & (
            out["bench2025_last5f_ok"].fillna(False)
            | out["bench2025_late_peak_ok"].fillna(False)
        )
    )
    out["bench2025_good"] = (
        out["bench2025_like_race"]
        & out["top3"]
        & out["bench2025_pci_band"].fillna(False)
        & out["bench2025_scope_or_finish"].fillna(False)
    )
    out["bench2025_strict_good"] = (
        out["bench2025_good"]
        & out["bench2025_late_peak_ok"].fillna(False)
    )
    return out


def example_text(frame: pd.DataFrame, n: int = 2) -> str:
    if frame.empty:
        return ""
    scored = frame.copy()
    scored["example_score"] = (
        scored["bench2025_good"].astype(int) * 5
        + scored["bench2025_strict_good"].astype(int) * 3
        + scored["bench2025_last5f_ok"].astype(int)
        + scored["bench2025_scope_or_finish"].astype(int)
    )
    rows = scored.sort_values(["example_score", "date"], ascending=[False, False]).head(n)
    parts = []
    for _, row in rows.iterrows():
        rpci = "" if pd.isna(row["rpci"]) else f"RPCI{row['rpci']:.1f}"
        pci = "" if pd.isna(row["pci"]) else f"PCI{row['pci']:.1f}"
        c4 = "-" if pd.isna(row["corner4"]) else str(int(row["corner4"]))
        parts.append(
            f"{row['date']} {row['race_name']} {int(row['distance'])}m "
            f"{int(row['finish'])}着 {pci} {rpci} 4角{c4}"
        )
    return " / ".join(parts)


def main() -> None:
    out_dir = ROOT / "outputs/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    takarazuka_ids = set(load_takarazuka_race_ids())
    takarazuka = load_su_records_for_races(takarazuka_ids)
    takarazuka = takarazuka[pd.to_numeric(takarazuka["finish"], errors="coerce").between(1, 99)].copy()
    takarazuka["horse_id"] = takarazuka["horse_id"].astype(str)
    takarazuka["date"] = takarazuka["date"].astype(str)

    raw = load_raw_results()
    start_year = max(2016, int(takarazuka["year"].min()) - 10)
    end_year = int(takarazuka["year"].max())
    history = add_2025_benchmark_flags(build_history_features(raw, start_year, end_year))

    rows: list[dict[str, object]] = []
    for _, entry in takarazuka.sort_values(["year", "race_id", "finish"]).iterrows():
        horse_id = str(entry["horse_id"])
        target_date = str(entry["date"])
        group = history[
            history["horse_id"].astype(str).eq(horse_id)
            & history["date"].astype(str).lt(target_date)
            & history["valid_finish"]
            & history["turf_mid"]
        ].copy()
        like = group[group["bench2025_like_race"]]
        non_like = group[~group["bench2025_like_race"]]
        good = group[group["bench2025_good"]]
        strict_good = group[group["bench2025_strict_good"]]
        starts = len(like)
        good_count = int(like["bench2025_good"].sum()) if starts else 0
        top3_count = int(like["top3"].sum()) if starts else 0
        non_top3 = int(non_like["top3"].sum())
        rate = good_count / starts if starts else math.nan
        top3_rate = top3_count / starts if starts else math.nan
        non_rate = non_top3 / len(non_like) if len(non_like) else math.nan
        graded_bench_good = int(good["graded_race"].sum()) if not good.empty else 0
        weighted_bench_score = float(good["grade_weight"].sum()) if not good.empty else 0.0
        no_breather_good = int(group["no_breather_good"].sum())
        weighted_no_breather_score = float(group["weighted_no_breather_good"].sum())
        graded_tough_good = int(group["graded_tough_good"].sum())
        weighted_tough_score = float(group["weighted_tough_good"].sum())

        fit = (
            len(group) >= 3
            and starts >= 1
            and good_count >= 1
            and (pd.isna(non_rate) or top3_rate >= non_rate)
        )
        strong_fit = len(group) >= 3 and good_count >= 2
        weighted_fit = (
            len(group) >= 3
            and (
                weighted_bench_score >= 1.7
                or (good_count >= 1 and no_breather_good >= 1 and weighted_tough_score >= 2.4)
            )
        )
        graded_fit = len(group) >= 3 and (graded_bench_good >= 1 or graded_tough_good >= 2)

        rows.append(
            {
                "race_id": entry["race_id"],
                "year": int(entry["year"]),
                "date": entry["date"],
                "horse_id": horse_id,
                "馬名": entry["horse_name"],
                "宝塚着順": entry["finish"],
                "宝塚人気": entry["popularity"],
                "宝塚位置取り": entry["position"],
                "判定対象戦数": len(group),
                "2025型戦数": starts,
                "2025型好走数": good_count,
                "2025型馬券内数": top3_count,
                "2025型好走率": round(rate, 4) if pd.notna(rate) else pd.NA,
                "2025型馬券内率": round(top3_rate, 4) if pd.notna(top3_rate) else pd.NA,
                "非2025型馬券内率": round(non_rate, 4) if pd.notna(non_rate) else pd.NA,
                "R2減速型好走数": int(strict_good.shape[0]),
                "重賞2025型好走数": graded_bench_good,
                "2025型重みスコア": round(weighted_bench_score, 2),
                "息入りにくい好走数": no_breather_good,
                "息入りにくい重みスコア": round(weighted_no_breather_score, 2),
                "重賞タフ持続好走数": graded_tough_good,
                "タフ持続重みスコア": round(weighted_tough_score, 2),
                "2025ベンチ該当": "Y" if fit else "",
                "2025ベンチ強該当": "Y" if strong_fit else "",
                "重み付き2025該当": "Y" if weighted_fit else "",
                "重賞実績該当": "Y" if graded_fit else "",
                "参考2025型好走": example_text(good if not good.empty else like),
            }
        )

    summary = pd.DataFrame(rows)
    character_path = out_dir / "takarazuka_kinen_hanshin_10_character_fit_summary.csv"
    if character_path.exists():
        character = pd.read_csv(character_path, dtype={"race_id": str, "horse_id": str})
        character_cols = ["race_id", "horse_id", "宝塚キャラ該当", "持続高打率型"]
        summary = summary.merge(character[character_cols], on=["race_id", "horse_id"], how="left")
    else:
        summary["宝塚キャラ該当"] = ""
        summary["持続高打率型"] = ""
    summary["2025_OR_宝塚キャラ"] = (
        summary["重み付き2025該当"].eq("Y") | summary["宝塚キャラ該当"].eq("Y")
    )
    summary["2025_OR_持続高打率"] = (
        summary["重み付き2025該当"].eq("Y") | summary["持続高打率型"].eq("Y")
    )
    judged = summary[summary["判定対象戦数"].ge(3)].copy()
    fit = judged[judged["2025ベンチ該当"].eq("Y")].copy()
    non_fit = judged[~judged["2025ベンチ該当"].eq("Y")].copy()
    strong = judged[judged["2025ベンチ強該当"].eq("Y")].copy()
    weak = judged[~judged["2025ベンチ強該当"].eq("Y")].copy()
    weighted = judged[judged["重み付き2025該当"].eq("Y")].copy()
    non_weighted = judged[~judged["重み付き2025該当"].eq("Y")].copy()
    graded = judged[judged["重賞実績該当"].eq("Y")].copy()
    non_graded = judged[~judged["重賞実績該当"].eq("Y")].copy()
    no_breather_weight2 = judged[judged["息入りにくい重みスコア"].ge(2.0)].copy()
    non_no_breather_weight2 = judged[~judged["息入りにくい重みスコア"].ge(2.0)].copy()
    no_breather_weight3 = judged[judged["息入りにくい重みスコア"].ge(3.0)].copy()
    non_no_breather_weight3 = judged[~judged["息入りにくい重みスコア"].ge(3.0)].copy()
    composite = judged[judged["2025_OR_宝塚キャラ"]].copy()
    non_composite = judged[~judged["2025_OR_宝塚キャラ"]].copy()
    composite_sustain = judged[judged["2025_OR_持続高打率"]].copy()
    non_composite_sustain = judged[~judged["2025_OR_持続高打率"]].copy()
    actual_2025 = summary[summary["year"].eq(2025)].copy()

    detail_path = out_dir / "takarazuka_kinen_hanshin_10_2025_benchmark_summary.csv"
    result_path = out_dir / "takarazuka_kinen_hanshin_10_2025_benchmark_results.csv"
    history_path = out_dir / "takarazuka_kinen_hanshin_10_2025_benchmark_histories.csv"
    report_path = out_dir / "takarazuka_kinen_hanshin_10_2025_benchmark_report.md"

    summary.sort_values(["year", "宝塚着順", "馬名"]).to_csv(detail_path, index=False, encoding="utf-8-sig")
    weighted.sort_values(["year", "宝塚着順", "馬名"]).to_csv(result_path, index=False, encoding="utf-8-sig")
    history[
        history["horse_id"].isin(set(takarazuka["horse_id"]))
        & history["valid_finish"]
        & history["turf_mid"]
    ].sort_values(["horse_id", "date", "race_id"]).to_csv(history_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 宝塚記念 2025年展開ベンチマーク分析",
        "",
        "ベンチマーク: 2025年6月15日 阪神芝2200m・稍重。前半3F34.8、後半3F36.0、前後差-1.2、後半5F59.8、PCI3 50.1、RPCI 49.1、残り7FはR2最速→R1減速。",
        "",
        "2025型レース定義: 芝1800m以上、RPCI46.5-51.5、前後差-2.4から-0.6、かつ後半5Fが同条件上位55%以内またはR2最速→R1減速型。好走は3着以内かつPCI46.0-53.5、さらに4角射程圏・押し上げ・上がり5位以内のいずれかを満たすもの。G1=2.0、G2=1.7、G3=1.4、OP/L=1.15で重み付けした。",
        "",
        "息入りにくいペース: 芝1800m以上でRPCI51.5以下または前傾、かつ後半5F上位55%以内・持続形状・ロンスパ/持続/消耗戦のいずれかに該当するレース。",
        "",
        "## 2025年本番",
        "",
        "|馬名|着順|人気|位置取り|2025型好走|重み|重賞2025型|息入り好走|重み付き|汎用宝塚|複合|",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---|---|",
    ]
    for _, row in actual_2025.sort_values("宝塚着順").iterrows():
        lines.append(
            f"|{row['馬名']}|{int(row['宝塚着順'])}|{int(row['宝塚人気'])}|{row['宝塚位置取り']}|"
            f"{int(row['2025型好走数'])}|{float(row['2025型重みスコア']):.1f}|"
            f"{int(row['重賞2025型好走数'])}|{int(row['息入りにくい好走数'])}|"
            f"{row['重み付き2025該当'] if pd.notna(row['重み付き2025該当']) else ''}|"
            f"{row['宝塚キャラ該当'] if pd.notna(row['宝塚キャラ該当']) else ''}|"
            f"{'Y' if row['2025_OR_宝塚キャラ'] else ''}|"
        )

    lines.extend(
        [
            "",
            "## 過去10回での直結度",
            "",
            result_line("判定対象全体", judged),
            result_line("2025ベンチ該当", fit),
            result_line("2025ベンチ非該当", non_fit),
            result_line("2025ベンチ強該当（好走2回以上）", strong),
            result_line("2025ベンチ強非該当", weak),
            result_line("重み付き2025該当", weighted),
            result_line("重み付き2025非該当", non_weighted),
            result_line("重賞実績該当", graded),
            result_line("重賞実績非該当", non_graded),
            result_line("息入りにくい重み2.0以上", no_breather_weight2),
            result_line("息入りにくい重み2.0未満", non_no_breather_weight2),
            result_line("息入りにくい重み3.0以上", no_breather_weight3),
            result_line("息入りにくい重み3.0未満", non_no_breather_weight3),
            "",
            "## 2025型と汎用宝塚キャラの複合",
            "",
            result_line("2025型 OR 汎用宝塚キャラ", composite),
            result_line("どちらも非該当", non_composite),
            result_line("2025型 OR 持続高打率型", composite_sustain),
            result_line("どちらも非該当", non_composite_sustain),
            "",
            "## 重み付き2025該当馬",
            "",
            "|年|馬名|着順|人気|位置取り|2025型成績|重み|重賞2025型|息入り好走|息入り重み|参考好走|",
            "|---:|---|---:|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in weighted.sort_values(["year", "宝塚着順", "馬名"]).iterrows():
        like_rate = float(row["2025型馬券内率"]) if pd.notna(row["2025型馬券内率"]) else math.nan
        lines.append(
            f"|{int(row['year'])}|{row['馬名']}|{int(row['宝塚着順'])}|"
            f"{int(row['宝塚人気']) if pd.notna(row['宝塚人気']) else ''}|{row['宝塚位置取り']}|"
            f"{record_text(int(row['2025型戦数']), int(row['2025型好走数']), like_rate)}|"
            f"{float(row['2025型重みスコア']):.1f}|{int(row['重賞2025型好走数'])}|"
            f"{int(row['息入りにくい好走数'])}|{float(row['息入りにくい重みスコア']):.1f}|"
            f"{str(row['参考2025型好走']).replace('|', '/')[:140]}|"
        )

    missed = judged[judged["宝塚着順"].between(1, 3) & ~judged["2025ベンチ該当"].eq("Y")].copy()
    lines.extend(
        [
            "",
            "## 拾えなかった馬券内馬",
            "",
            "|年|馬名|着順|人気|2025型戦数|2025型好走|非2025型馬券内率|",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in missed.sort_values(["year", "宝塚着順", "馬名"]).iterrows():
        lines.append(
            f"|{int(row['year'])}|{row['馬名']}|{int(row['宝塚着順'])}|"
            f"{int(row['宝塚人気']) if pd.notna(row['宝塚人気']) else ''}|"
            f"{int(row['2025型戦数'])}|{int(row['2025型好走数'])}|"
            f"{pct(float(row['非2025型馬券内率'])) if pd.notna(row['非2025型馬券内率']) else ''}|"
        )

    lines.extend(
        [
            "",
            "## 読み取り",
            "",
            "- 2025年型を重賞実績で重くしても、単体では馬券内率を大きく押し上げなかった。昨年の展開そのものは再現シナリオ専用の補助特徴として扱う。",
            "- 息入りにくい重みスコアは単体で少し差が出た。特に3.0以上では平均着順が改善し、タフな流れへの耐性を見る材料になる。",
            "- 2025年本番は、メイショウタバルとジャスティンパレスを重み付き2025型で拾い、ベラジオオペラは汎用宝塚キャラで拾う形。2025型だけでなく汎用宝塚キャラとの複合で見るのが妥当。",
            "",
            "## 出力ファイル",
            "",
            f"- 集計CSV: `{detail_path.as_posix()}`",
            f"- 該当馬CSV: `{result_path.as_posix()}`",
            f"- 過去走明細CSV: `{history_path.as_posix()}`",
            f"- レポート: `{report_path.as_posix()}`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"judged={len(judged)}")
    print(f"fit={len(fit)}")
    print(f"strong={len(strong)}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
