from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_takarazuka_2025_benchmark import add_2025_benchmark_flags  # noqa: E402
from scripts.analyze_takarazuka_character_fit import (  # noqa: E402
    build_history_features,
    load_raw_results,
    record_text,
)
from scripts.analyze_takarazuka_longspurt_aptitude import (  # noqa: E402
    load_ra_map,
    load_su_records_for_horses,
    load_su_records_for_races,
)


EXPECTED_RUNNERS = [
    "クロワデュノール",
    "メイショウタバル",
    "ミュージアムマイル",
    "ダノンデサイル",
    "レガレイラ",
    "ビザンチンドリーム",
    "シェイクユアハート",
    "タガノデュード",
    "スティンガーグラス",
    "ジューンテイク",
    "マイユニバース",
    "シンエンペラー",
    "ミステリーウェイ",
    "ミクニインスパイア",
    "コスモキュランダ",
    "ファミリータイム",
    "シュガークン",
]

TARGET_DATE = "20260614"


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def y(flag: bool) -> str:
    return "Y" if flag else ""


def short_examples(frame: pd.DataFrame, n: int = 2) -> str:
    if frame.empty:
        return ""
    scored = frame.copy()
    scored["example_score"] = (
        scored.get("bench2025_good", False).astype(int) * 6
        + scored.get("no_breather_good", False).astype(int) * 3
        + scored.get("graded_race", False).astype(int) * 2
        + scored.get("tough_sustain_good", False).astype(int)
        + scored["date"].astype(str).rank(method="dense").astype(int) / 1000
    )
    rows = scored.sort_values(["example_score", "date"], ascending=[False, False]).head(n)
    parts: list[str] = []
    for _, row in rows.iterrows():
        pci = "" if pd.isna(row["pci"]) else f"PCI{row['pci']:.1f}"
        rpci = "" if pd.isna(row["rpci"]) else f"RPCI{row['rpci']:.1f}"
        c4 = "-" if pd.isna(row["corner4"]) else str(int(row["corner4"]))
        grade = ""
        if row.get("grade_weight", 1.0) >= 2.0:
            grade = "G1 "
        elif row.get("grade_weight", 1.0) >= 1.7:
            grade = "G2 "
        elif row.get("grade_weight", 1.0) >= 1.4:
            grade = "G3 "
        parts.append(
            f"{row['date']} {grade}{row['race_name']} {int(row['distance'])}m "
            f"{int(row['finish'])}着 {pci} {rpci} 4角{c4}"
        )
    return " / ".join(parts)


def grade_score_label(score: float) -> str:
    if score >= 7.0:
        return "A"
    if score >= 5.0:
        return "B+"
    if score >= 3.5:
        return "B"
    if score >= 2.0:
        return "C"
    return "D"


def split_position(position: object) -> tuple[float, float, float, float]:
    values = [math.nan, math.nan, math.nan, math.nan]
    if pd.isna(position):
        return tuple(values)
    parts = [part for part in str(position).split("-") if part.strip().isdigit()]
    for idx, part in enumerate(parts[:4]):
        values[idx] = float(part)
    return tuple(values)


def supplemental_target_rows(raw: pd.DataFrame) -> pd.DataFrame:
    name_map = raw[raw["horse_name"].isin(EXPECTED_RUNNERS)][["horse_name", "horse_id"]].drop_duplicates()
    horse_ids = set(name_map["horse_id"].astype(str))
    if not horse_ids:
        return pd.DataFrame(columns=raw.columns)

    raw_max_date = str(raw["date"].max())
    target_su = load_su_records_for_horses(horse_ids, 2026, 2026)
    if target_su.empty:
        return pd.DataFrame(columns=raw.columns)
    target_su = target_su[target_su["date"].astype(str).gt(raw_max_date)].copy()
    target_su = target_su[target_su["date"].astype(str).lt(TARGET_DATE)].copy()
    if target_su.empty:
        return pd.DataFrame(columns=raw.columns)

    race_ids = set(target_su["race_id"].astype(str))
    all_su = load_su_records_for_races(race_ids)
    if all_su.empty:
        all_su = target_su.copy()
    field_size = all_su.groupby("race_id")["horse_id"].nunique().rename("field_size")
    final_rank = all_su.copy()
    final_rank["final3f_rank"] = final_rank.groupby("race_id")["final3f"].rank(method="min")
    final_rank = final_rank[["race_id", "horse_id", "final3f_rank"]]

    races = load_ra_map(race_ids)
    out = target_su.merge(races, on="race_id", how="left", suffixes=("", "_race"))
    out = out.merge(field_size, on="race_id", how="left")
    out = out.merge(final_rank, on=["race_id", "horse_id"], how="left")
    corners = out["position"].apply(split_position).apply(pd.Series)
    corners.columns = ["corner1", "corner2", "corner3", "corner4"]
    out = pd.concat([out, corners], axis=1)
    out["class_name"] = out["race_name"].fillna("").apply(
        lambda name: "G1" if name in {"大阪杯", "天皇賞（春）"} else ""
    )
    venue_by_code = {
        "01": "札幌",
        "02": "函館",
        "03": "福島",
        "04": "新潟",
        "05": "東京",
        "06": "中山",
        "07": "中京",
        "08": "京都",
        "09": "阪神",
        "10": "小倉",
    }
    out["venue"] = out["race_id"].astype(str).str[8:10].map(venue_by_code).fillna("")
    out["date_raw"] = out["date"].astype(str).str[2:]
    out["going"] = ""
    out["margin"] = pd.NA
    out["running_style"] = ""
    out["ave3f"] = pd.NA
    out["pci"] = pd.NA
    out["pci3"] = pd.NA
    out["rpci"] = pd.NA
    keep = [
        "date_raw",
        "venue",
        "race_name",
        "class_name",
        "horse_name",
        "field_size",
        "popularity",
        "finish",
        "surface",
        "distance",
        "going",
        "margin",
        "corner1",
        "corner2",
        "corner3",
        "corner4",
        "running_style",
        "ave3f",
        "final3f",
        "final3f_rank",
        "pci",
        "pci3",
        "rpci",
        "race_id",
        "horse_id",
        "date",
    ]
    supplemental = out[keep].copy()
    for col in raw.columns:
        if col not in supplemental.columns:
            supplemental[col] = pd.NA
    return supplemental[raw.columns]


def build_runner_summary(history: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
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

        like = group[group["bench2025_like_race"].fillna(False)]
        non_like = group[~group["bench2025_like_race"].fillna(False)]
        good = group[group["bench2025_good"].fillna(False)]
        strict_good = group[group["bench2025_strict_good"].fillna(False)]
        tough = group[group["tough_sustain_race"].fillna(False)]

        starts = len(like)
        good_count = int(good.shape[0])
        top3_count = int(like["top3"].sum()) if starts else 0
        non_top3 = int(non_like["top3"].sum())
        rate = good_count / starts if starts else math.nan
        top3_rate = top3_count / starts if starts else math.nan
        non_rate = non_top3 / len(non_like) if len(non_like) else math.nan

        tough_starts = len(tough)
        tough_top3 = int(tough["top3"].sum())
        tough_rate = tough_top3 / tough_starts if tough_starts else math.nan
        low_pci_good = int(group["low_rpci_pci_fit_good"].sum())
        scope_good = int(group["scope_move_good"].sum())
        graded_bench_good = int(good["graded_race"].sum()) if not good.empty else 0
        weighted_bench_score = float(good["grade_weight"].sum()) if not good.empty else 0.0
        no_breather_good = int(group["no_breather_good"].sum())
        weighted_no_breather_score = float(group["weighted_no_breather_good"].sum())
        graded_tough_good = int(group["graded_tough_good"].sum())
        weighted_tough_score = float(group["weighted_tough_good"].sum())
        graded_no_breather_score = float(
            group.loc[
                group["no_breather_good"].fillna(False) & group["graded_race"].fillna(False),
                "grade_weight",
            ].sum()
        )
        graded_tough_score = float(
            group.loc[
                group["tough_sustain_good"].fillna(False) & group["graded_race"].fillna(False),
                "grade_weight",
            ].sum()
        )

        generic_fit = (
            len(group) >= 3
            and tough_starts >= 3
            and pd.notna(tough_rate)
            and tough_rate >= 0.55
            and low_pci_good >= 2
            and scope_good >= 2
        )
        sustain_high_rate = (
            len(group) >= 3
            and tough_starts >= 3
            and pd.notna(tough_rate)
            and tough_rate >= 0.60
        )
        weighted_2025_fit = (
            len(group) >= 3
            and (
                weighted_bench_score >= 1.7
                or (good_count >= 1 and no_breather_good >= 1 and weighted_tough_score >= 2.4)
            )
        )
        graded_fit = len(group) >= 3 and (graded_bench_good >= 1 or graded_tough_good >= 2)
        composite_fit = weighted_2025_fit or generic_fit

        grade_core = (
            min(weighted_bench_score, 4.0) * 0.60
            + min(graded_no_breather_score, 5.0) * 0.70
            + min(graded_tough_score, 7.0) * 0.35
        )
        lower_support = (
            min(max(weighted_no_breather_score - graded_no_breather_score, 0.0), 3.0) * 0.20
            + min(max(weighted_tough_score - graded_tough_score, 0.0), 4.0) * 0.12
        )
        score = grade_core + lower_support
        score += 1.2 if weighted_2025_fit else 0.0
        score += 0.8 if generic_fit else 0.0
        score += 0.8 if graded_fit else 0.0
        score += 0.3 if sustain_high_rate else 0.0
        if starts and pd.notna(non_rate) and top3_rate > non_rate:
            score += 0.3

        rows.append(
            {
                "馬名": horse_name,
                "horse_id": horse_id,
                "データ有無": "OK",
                "判定対象戦数": len(group),
                "2025型戦数": starts,
                "2025型好走数": good_count,
                "2025型馬券内数": top3_count,
                "2025型馬券内率": round(top3_rate, 4) if pd.notna(top3_rate) else pd.NA,
                "非2025型馬券内率": round(non_rate, 4) if pd.notna(non_rate) else pd.NA,
                "R2減速型好走数": int(strict_good.shape[0]),
                "重賞2025型好走数": graded_bench_good,
                "2025型重みスコア": round(weighted_bench_score, 2),
                "息入りにくい好走数": no_breather_good,
                "息入りにくい重みスコア": round(weighted_no_breather_score, 2),
                "重賞息入り重みスコア": round(graded_no_breather_score, 2),
                "タフ持続戦数": tough_starts,
                "タフ持続馬券内数": tough_top3,
                "タフ持続馬券内率": round(tough_rate, 4) if pd.notna(tough_rate) else pd.NA,
                "重賞タフ持続好走数": graded_tough_good,
                "タフ持続重みスコア": round(weighted_tough_score, 2),
                "重賞タフ持続重みスコア": round(graded_tough_score, 2),
                "低RPCI×PCI適性好走数": low_pci_good,
                "射程圏/押し上げ好走数": scope_good,
                "汎用宝塚キャラ": y(generic_fit),
                "持続高打率型": y(sustain_high_rate),
                "重み付き2025該当": y(weighted_2025_fit),
                "重賞実績該当": y(graded_fit),
                "複合該当": y(composite_fit),
                "適性スコア": round(score, 2),
                "評価": grade_score_label(score),
                "根拠レース": short_examples(
                    group[
                        group["bench2025_good"].fillna(False)
                        | group["no_breather_good"].fillna(False)
                        | group["tough_sustain_good"].fillna(False)
                    ]
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
    summary = build_runner_summary(history, raw)
    raw_max_date = str(raw["date"].max())

    detail_path = out_dir / "takarazuka_kinen_2026_expected_runners_benchmark_summary.csv"
    report_path = out_dir / "takarazuka_kinen_2026_expected_runners_benchmark_report.md"
    history_path = out_dir / "takarazuka_kinen_2026_expected_runners_benchmark_histories.csv"

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

    lines = [
        "# 宝塚記念2026 想定馬の重み付き適性分析",
        "",
        "対象: netkeiba掲載の想定17頭。正式出馬表公開前のため、出走予定の変更は別途反映が必要。",
        f"TARGET由来履歴データの最新日: {raw_max_date}。この日以降の出走内容は今回のスコアに未反映。",
        f"SE_DATAから2/15以降の対象馬履歴を{len(supplemental)}件補完。PCI/RPCIは未補完のため、補完分は着順・人気・4角・上がり順位・レースラップ・G1/G2/G3重みを中心に反映。",
        "",
        "評価軸: 2025年宝塚型への対応、G1/G2/G3での発揮重み、息が入りにくいペースでの好走、汎用宝塚キャラ、タフ持続戦の安定度。",
        "",
        "## 総合ランキング",
        "",
        "|評価|馬名|スコア|複合|汎用宝塚|重み付き2025|息入り重み|重賞息入り|タフ持続|重賞タフ|根拠|",
        "|---|---|---:|---|---|---|---:|---:|---|---:|---|",
    ]

    ranked = summary.sort_values(["適性スコア", "馬名"], ascending=[False, True]).copy()
    for _, row in ranked.iterrows():
        tough_record = record_text(
            int(row.get("タフ持続戦数", 0) or 0),
            int(row.get("タフ持続馬券内数", 0) or 0),
            float(row["タフ持続馬券内率"]) if pd.notna(row.get("タフ持続馬券内率")) else math.nan,
        )
        lines.append(
            f"|{row['評価']}|{row['馬名']}|{float(row['適性スコア']):.2f}|"
            f"{row['複合該当']}|{row['汎用宝塚キャラ']}|{row['重み付き2025該当']}|"
            f"{float(row['息入りにくい重みスコア']):.1f}|{float(row['重賞息入り重みスコア']):.1f}|{tough_record}|"
            f"{int(row['重賞タフ持続好走数'])}|{str(row['根拠レース']).replace('|', '/')[:150]}|"
        )

    lines.extend(
        [
            "",
            "## 今年の見立て",
            "",
            "- A/B+評価は、単に速い上がりを使った馬ではなく、上位クラスまたは息が入りにくい流れで好走実績がある馬。",
            "- C以下は能力否定ではなく、今回のラップキャラクター条件に対するTARGET履歴上の裏付けが薄い馬。特に3歳馬は履歴戦数が少ないため過小評価になりやすい。",
            "- 2026年春の戦績がデータ未反映なので、正式出馬表確定後にTARGETを更新して再実行すると評価が動く可能性がある。",
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
    print(f"raw_max_date={raw_max_date}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
