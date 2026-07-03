from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_takarazuka_last7_peak_aptitude import last7_features  # noqa: E402
from scripts.analyze_takarazuka_longspurt_aptitude import (  # noqa: E402
    load_ra_records,
    load_su_records_for_races,
    load_takarazuka_race_ids,
)
from src.features.longspurt import LongspurtConfig, build_race_longspurt_features, parse_laps  # noqa: E402


RAW_PATH = ROOT / "date/raw/全競走馬成績.csv"

RAW_COLS = [
    "日付",
    "場所",
    "レース名",
    "クラス名",
    "馬名",
    "頭数",
    "人気",
    "確定着順",
    "芝・ダ",
    "距離",
    "馬場状態",
    "着差",
    "1角",
    "2角",
    "3角",
    "4角",
    "脚質",
    "Ave-3F",
    "上り3F",
    "上り3F順",
    "PCI",
    "PCI3",
    "RPCI",
    "レースID(新/馬番無)",
    "血統登録番号",
]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def full_date(value: object) -> str:
    text = str(value).strip().replace(".0", "")
    if not text or text == "nan":
        return ""
    text = text.zfill(6)
    return f"20{text}"


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def record_text(starts: int, top3: int, rate: float) -> str:
    if starts == 0:
        return "0戦"
    return f"{starts}戦{top3}好走（{fmt_pct(rate)}）"


def load_raw_results() -> pd.DataFrame:
    raw = pd.read_csv(
        RAW_PATH,
        encoding="cp932",
        usecols=lambda col: col in RAW_COLS,
        dtype={"日付": str, "レースID(新/馬番無)": str, "血統登録番号": str},
        low_memory=False,
    )
    out = raw.rename(
        columns={
            "日付": "date_raw",
            "場所": "venue",
            "レース名": "race_name",
            "クラス名": "class_name",
            "馬名": "horse_name",
            "頭数": "field_size",
            "人気": "popularity",
            "確定着順": "finish",
            "芝・ダ": "surface",
            "距離": "distance",
            "馬場状態": "going",
            "着差": "margin",
            "1角": "corner1",
            "2角": "corner2",
            "3角": "corner3",
            "4角": "corner4",
            "脚質": "running_style",
            "Ave-3F": "ave3f",
            "上り3F": "final3f",
            "上り3F順": "final3f_rank",
            "PCI": "pci",
            "PCI3": "pci3",
            "RPCI": "rpci",
            "レースID(新/馬番無)": "race_id",
            "血統登録番号": "horse_id",
        }
    ).copy()
    out["date"] = out["date_raw"].apply(full_date)
    for col in [
        "field_size",
        "popularity",
        "finish",
        "distance",
        "corner1",
        "corner2",
        "corner3",
        "corner4",
        "ave3f",
        "final3f",
        "final3f_rank",
        "pci",
        "pci3",
        "rpci",
    ]:
        out[col] = numeric(out[col])
    out["race_id"] = out["race_id"].astype(str).str.strip()
    out["horse_id"] = out["horse_id"].astype(str).str.strip()
    out = out[out["race_id"].str.len().ge(12) & out["horse_id"].str.len().ge(6)].copy()
    return out.drop_duplicates(["race_id", "horse_id"])


def position_text(row: pd.Series) -> str:
    values: list[str] = []
    for col in ["corner1", "corner2", "corner3", "corner4"]:
        value = row.get(col)
        if pd.notna(value) and float(value) > 0:
            values.append(str(int(value)))
    return "-".join(values)


def add_lap_context(races: pd.DataFrame) -> pd.DataFrame:
    out = races.copy()
    laps = out["race_laps"].apply(parse_laps)
    out["first3f"] = laps.apply(lambda xs: round(sum(xs[:3]), 1) if len(xs) >= 3 else np.nan)
    out["last3f"] = laps.apply(lambda xs: round(sum(xs[-3:]), 1) if len(xs) >= 3 else np.nan)
    out["front_minus_last"] = (out["first3f"] - out["last3f"]).round(1)
    return out


def build_history_features(raw: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    races = load_ra_records(start_year, end_year)
    races = add_lap_context(races)
    longspurt = build_race_longspurt_features(races, LongspurtConfig(min_group_size=30, fast_quantile=0.25))
    last7 = last7_features(races)
    race_cols = [
        "race_id",
        "first3f",
        "last3f",
        "front_minus_last",
        "last5f_sum",
        "last5f_range",
        "sustained_shape",
        "last5f_fast_percentile",
        "race_longspurt_type",
        "is_longspurt_race",
    ]
    last7_cols = ["race_id", "late_peak7_type", "late_peak7_race", "late_peak7_r2_decel_race"]
    out = raw.merge(longspurt[race_cols], on="race_id", how="left")
    out = out.merge(last7[last7_cols], on="race_id", how="left")

    out["valid_finish"] = out["finish"].between(1, 99)
    out["top3"] = out["finish"].between(1, 3)
    out["win"] = out["finish"].eq(1)
    out["turf_mid"] = out["surface"].astype(str).eq("芝") & out["distance"].ge(1800)
    out["corner4_rate"] = out["corner4"] / out["field_size"].replace(0, np.nan)
    out["corner3_to_4_gain"] = out["corner3"] - out["corner4"]
    out["scope_or_move"] = out["corner4_rate"].le(0.65) | out["corner3_to_4_gain"].ge(2)
    out["tough_pci_range"] = out["pci"].between(46.0, 54.5)
    out["low_rpci"] = out["rpci"].le(51.0)
    out["front_loaded"] = out["front_minus_last"].le(-0.6)
    out["fast_last5f_top50"] = out["last5f_fast_percentile"].le(0.50)
    out["sustain_type"] = out["race_longspurt_type"].isin(["ロンスパ戦", "持続戦", "消耗戦"])
    out["tough_sustain_race"] = (
        out["turf_mid"]
        & (
            out["low_rpci"].fillna(False)
            | out["front_loaded"].fillna(False)
            | out["sustain_type"].fillna(False)
            | out["fast_last5f_top50"].fillna(False)
        )
    )
    out["tough_sustain_good"] = out["tough_sustain_race"] & out["top3"]
    out["pci_fit_good"] = out["turf_mid"] & out["top3"] & out["tough_pci_range"]
    out["low_rpci_pci_fit_good"] = out["tough_sustain_good"] & out["tough_pci_range"]
    out["scope_move_good"] = out["turf_mid"] & out["top3"] & out["scope_or_move"]
    out["fast_last5f_good"] = out["turf_mid"] & out["top3"] & out["fast_last5f_top50"].fillna(False)
    out["final3f_rank_good"] = out["turf_mid"] & out["top3"] & out["final3f_rank"].le(5)
    out["late_peak_good"] = out["turf_mid"] & out["top3"] & out["late_peak7_race"].fillna(False)
    out["position"] = out.apply(position_text, axis=1)
    grade_text = (out["race_name"].fillna("").astype(str) + out["class_name"].fillna("").astype(str)).str.upper()
    out["grade_weight"] = 1.0
    out.loc[grade_text.str.contains("G1|Ｇ１", regex=True), "grade_weight"] = 2.0
    out.loc[grade_text.str.contains("G2|Ｇ２", regex=True), "grade_weight"] = 1.7
    out.loc[grade_text.str.contains("G3|Ｇ３", regex=True), "grade_weight"] = 1.4
    out.loc[grade_text.str.contains(r"\(L\)|LISTED|ＯＰ|OP", regex=True), "grade_weight"] = 1.15
    out["graded_race"] = out["grade_weight"].ge(1.4)
    out["no_breather_race"] = (
        out["turf_mid"]
        & (
            out["rpci"].le(51.5).fillna(False)
            | out["front_minus_last"].le(-0.6).fillna(False)
        )
        & (
            out["last5f_fast_percentile"].le(0.55).fillna(False)
            | out["sustained_shape"].fillna(False)
            | out["race_longspurt_type"].isin(["ロンスパ戦", "持続戦", "消耗戦"]).fillna(False)
        )
    )
    out["no_breather_good"] = out["no_breather_race"] & out["top3"]
    out["graded_tough_good"] = out["tough_sustain_good"] & out["graded_race"]
    out["weighted_tough_good"] = out["tough_sustain_good"].astype(float) * out["grade_weight"]
    out["weighted_low_rpci_pci_good"] = out["low_rpci_pci_fit_good"].astype(float) * out["grade_weight"]
    out["weighted_no_breather_good"] = out["no_breather_good"].astype(float) * out["grade_weight"]
    return out


def summarize_result(frame: pd.DataFrame) -> dict[str, object]:
    finish_col = "finish" if "finish" in frame.columns else "宝塚着順"
    popularity_col = "popularity" if "popularity" in frame.columns else "宝塚人気"
    valid = frame[pd.to_numeric(frame[finish_col], errors="coerce").between(1, 99)].copy()
    if valid.empty:
        return {
            "starts": 0,
            "wins": 0,
            "top3": 0,
            "top3_rate": math.nan,
            "avg_finish": math.nan,
            "avg_pop": math.nan,
            "pop_gain": math.nan,
        }
    finish = numeric(valid[finish_col])
    popularity = numeric(valid[popularity_col])
    return {
        "starts": len(valid),
        "wins": int(finish.eq(1).sum()),
        "top3": int(finish.between(1, 3).sum()),
        "top3_rate": finish.between(1, 3).mean(),
        "avg_finish": finish.mean(),
        "avg_pop": popularity.mean(),
        "pop_gain": (popularity - finish).mean(),
    }


def result_line(label: str, frame: pd.DataFrame) -> str:
    stats = summarize_result(frame)
    if stats["starts"] == 0:
        return f"- {label}: 該当なし"
    return (
        f"- {label}: {stats['starts']}戦{stats['wins']}勝・馬券内{stats['top3']}回"
        f" / 馬券内率{stats['top3_rate'] * 100:.1f}%"
        f" / 平均{stats['avg_finish']:.1f}着"
        f" / 平均人気{stats['avg_pop']:.1f}"
        f" / 人気-着順{stats['pop_gain']:+.1f}"
    )


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
    history = build_history_features(raw, start_year, end_year)

    target_raw_cols = [
        "race_id",
        "horse_id",
        "field_size",
        "pci",
        "pci3",
        "rpci",
        "final3f",
        "final3f_rank",
        "corner4",
        "corner4_rate",
        "running_style",
    ]
    target = takarazuka.merge(history[target_raw_cols], on=["race_id", "horse_id"], how="left")
    target["target_position"] = target["position"]

    rows: list[dict[str, object]] = []
    for _, entry in target.sort_values(["year", "race_id", "finish"]).iterrows():
        horse_id = str(entry["horse_id"])
        target_date = str(entry["date"])
        group = history[
            history["horse_id"].astype(str).eq(horse_id)
            & history["date"].astype(str).lt(target_date)
            & history["valid_finish"]
            & history["turf_mid"]
        ].copy()

        tough_good = int(group["tough_sustain_good"].sum())
        pci_fit_good = int(group["pci_fit_good"].sum())
        low_rpci_pci_fit_good = int(group["low_rpci_pci_fit_good"].sum())
        scope_move_good = int(group["scope_move_good"].sum())
        fast_last5f_good = int(group["fast_last5f_good"].sum())
        final3f_rank_good = int(group["final3f_rank_good"].sum())
        late_peak_good = int(group["late_peak_good"].sum())
        tough_starts = int(group["tough_sustain_race"].sum())
        tough_top3_rate = tough_good / tough_starts if tough_starts else math.nan
        all_top3_rate = group["top3"].mean() if len(group) else math.nan

        score = 0
        score += 2 if tough_good >= 1 else 0
        score += 2 if low_rpci_pci_fit_good >= 1 else 0
        score += 1 if scope_move_good >= 1 else 0
        score += 1 if fast_last5f_good >= 1 else 0
        score += 1 if final3f_rank_good >= 1 else 0
        score += 1 if late_peak_good >= 1 else 0
        sustain_high_rate = len(group) >= 3 and tough_starts >= 3 and pd.notna(tough_top3_rate) and tough_top3_rate >= 0.60
        character_fit = (
            len(group) >= 3
            and tough_starts >= 3
            and pd.notna(tough_top3_rate)
            and tough_top3_rate >= 0.55
            and low_rpci_pci_fit_good >= 2
            and scope_move_good >= 2
        )
        core_fit = character_fit and fast_last5f_good >= 2

        best_examples = group[
            group["tough_sustain_good"] | group["low_rpci_pci_fit_good"] | group["scope_move_good"]
        ].copy()
        if not best_examples.empty:
            best_examples["example_score"] = (
                best_examples["tough_sustain_good"].astype(int) * 3
                + best_examples["low_rpci_pci_fit_good"].astype(int) * 3
                + best_examples["scope_move_good"].astype(int) * 2
                + best_examples["fast_last5f_good"].astype(int)
                + best_examples["final3f_rank_good"].astype(int)
            )
            best_examples = best_examples.sort_values(["example_score", "date"], ascending=[False, False]).head(2)
        example_text = " / ".join(
            [
                (
                    f"{row['date']} {row['race_name']} {int(row['distance'])}m "
                    f"{int(row['finish'])}着 PCI{row['pci']:.1f} RPCI{row['rpci']:.1f} "
                    f"4角{int(row['corner4']) if pd.notna(row['corner4']) else '-'}"
                )
                for _, row in best_examples.iterrows()
            ]
        )

        rows.append(
            {
                "race_id": entry["race_id"],
                "year": int(entry["year"]),
                "date": entry["date"],
                "horse_id": horse_id,
                "馬名": entry["horse_name"],
                "宝塚着順": entry["finish"],
                "宝塚人気": entry["popularity"],
                "宝塚位置取り": entry["target_position"],
                "宝塚PCI": entry.get("pci"),
                "宝塚RPCI": entry.get("rpci"),
                "宝塚4角率": entry.get("corner4_rate"),
                "判定対象戦数": len(group),
                "タフ持続戦数": tough_starts,
                "タフ持続好走数": tough_good,
                "タフ持続好走率": round(tough_top3_rate, 4) if pd.notna(tough_top3_rate) else pd.NA,
                "芝中距離好走率": round(all_top3_rate, 4) if pd.notna(all_top3_rate) else pd.NA,
                "PCI適性好走数": pci_fit_good,
                "低RPCI_PCI適性好走数": low_rpci_pci_fit_good,
                "射程圏_機動力好走数": scope_move_good,
                "後半5F上位好走数": fast_last5f_good,
                "上がり順位好走数": final3f_rank_good,
                "終盤ピーク好走数": late_peak_good,
                "キャラスコア": score,
                "持続高打率型": "Y" if sustain_high_rate else "",
                "宝塚キャラ該当": "Y" if character_fit else "",
                "中核キャラ該当": "Y" if core_fit else "",
                "参考好走": example_text,
            }
        )

    summary = pd.DataFrame(rows)
    judged = summary[summary["判定対象戦数"].ge(3)].copy()
    fit = judged[judged["宝塚キャラ該当"].eq("Y")].copy()
    non_fit = judged[~judged["宝塚キャラ該当"].eq("Y")].copy()
    sustain_rate_fit = judged[judged["持続高打率型"].eq("Y")].copy()
    sustain_rate_non = judged[~judged["持続高打率型"].eq("Y")].copy()
    core = judged[judged["中核キャラ該当"].eq("Y")].copy()
    non_core = judged[~judged["中核キャラ該当"].eq("Y")].copy()

    top3 = target[target["finish"].between(1, 3)].copy()
    top3_pci = top3[pd.notna(top3["pci"])].copy()
    top3_position = top3.copy()
    top3_position["c4_rate_for_summary"] = top3_position["corner4_rate"]

    detail_path = out_dir / "takarazuka_kinen_hanshin_10_character_fit_summary.csv"
    history_path = out_dir / "takarazuka_kinen_hanshin_10_character_fit_histories.csv"
    fit_path = out_dir / "takarazuka_kinen_hanshin_10_character_fit_results.csv"
    report_path = out_dir / "takarazuka_kinen_hanshin_10_character_fit_report.md"
    summary.sort_values(["year", "宝塚着順", "馬名"]).to_csv(detail_path, index=False, encoding="utf-8-sig")
    history[
        history["horse_id"].isin(set(takarazuka["horse_id"]))
        & history["turf_mid"]
        & history["valid_finish"]
    ].sort_values(["horse_id", "date", "race_id"]).to_csv(history_path, index=False, encoding="utf-8-sig")
    fit.sort_values(["year", "宝塚着順", "馬名"]).to_csv(fit_path, index=False, encoding="utf-8-sig")

    condition_summary = []
    for label, frame in [
        ("持続高打率型", sustain_rate_fit),
        ("持続高打率型ではない", sustain_rate_non),
        ("宝塚キャラ該当", fit),
        ("宝塚キャラ非該当", non_fit),
        ("中核キャラ該当", core),
    ]:
        stats = summarize_result(frame)
        condition_summary.append((label, stats))

    lines = [
        "# 宝塚記念 ラップ・PCI由来の好走馬キャラクター検証",
        "",
        "対象: 宝塚記念の阪神開催10回（2015-2025、京都2024除外）。出走馬の過去走特徴はTARGETの全競走馬成績CSVにPCI/RPCIが入る2016年以降を使用し、各宝塚記念の出走前履歴だけで判定した。2015年など履歴が3戦未満の馬は主集計から外した。",
        "",
        "## レース傾向から置いたキャラクター",
        "",
        "- 宝塚記念は平均RPCI50.6、前傾6回・ミドル3回・後傾1回で、純粋な瞬発戦よりも前半負荷と中盤以降の持続力が問われやすい。",
        "- 好走馬はPCIだけが高い差し馬ではなく、低RPCI/前傾寄りの芝1800m以上で崩れず、PCI46.0-54.5付近で好走できる馬を重視した。",
        "- 位置取りは逃げ先行だけに限定せず、4角で上位65%以内、または3角から4角で2つ以上押し上げられる馬を「射程圏・機動力あり」とした。",
        "- 補助的に、後半5F上位50%レースでの好走、上がり5位以内での好走、残り7F終盤ピーク型での好走を加点した。",
        "",
        "宝塚キャラ該当: タフ持続戦の好走率55%以上、低RPCIかつPCI46.0-54.5での好走2回以上、射程圏/機動力好走2回以上をすべて満たす馬。補助的に後半5F上位好走2回以上を満たすものを中核キャラ該当とした。",
        "",
        "## 好走馬側の確認",
        "",
        f"- PCIが取れる馬券内馬: {len(top3_pci)}頭 / PCI平均{top3_pci['pci'].mean():.1f} / 中央値{top3_pci['pci'].median():.1f} / 範囲{top3_pci['pci'].min():.1f}-{top3_pci['pci'].max():.1f}",
        f"- 馬券内馬RPCI平均: {top3_pci['rpci'].mean():.1f}",
        f"- 馬券内馬4角率中央値: {top3_position['c4_rate_for_summary'].median():.2f}",
        f"- 4角上位65%以内または押し上げ型の馬券内馬: {int((top3_position['corner4_rate'].le(0.65)).sum())}/{len(top3_position)}頭（4角率のみで集計）",
        "",
        "## 事前フラグと宝塚記念成績",
        "",
        result_line("判定対象全体", judged),
        result_line("持続高打率型（タフ持続戦好走率60%以上）", sustain_rate_fit),
        result_line("持続高打率型ではない", sustain_rate_non),
        result_line("宝塚キャラ該当", fit),
        result_line("宝塚キャラ非該当", non_fit),
        result_line("中核キャラ該当（後半5F上位好走も2回以上）", core),
        result_line("中核キャラ非該当", non_core),
        "",
        "## 条件別比較",
        "",
        "|条件|頭数|勝利|馬券内|馬券内率|平均着順|平均人気|人気-着順|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in condition_summary:
        if stats["starts"] == 0:
            lines.append(f"|{label}|0|||||||")
        else:
            lines.append(
                f"|{label}|{stats['starts']}|{stats['wins']}|{stats['top3']}|"
                f"{stats['top3_rate'] * 100:.1f}%|{stats['avg_finish']:.1f}|"
                f"{stats['avg_pop']:.1f}|{stats['pop_gain']:+.1f}|"
            )

    lines.extend(
        [
            "",
            "## 宝塚キャラ該当馬",
            "",
            "|年|馬名|着順|人気|判定対象|タフ持続|低RPCI+PCI|射程/機動|後半5F|参考好走|",
            "|---:|---|---:|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for _, row in fit.sort_values(["year", "宝塚着順", "馬名"]).iterrows():
        tough_rate = (
            float(row["タフ持続好走率"])
            if pd.notna(row["タフ持続好走率"])
            else math.nan
        )
        lines.append(
            f"|{int(row['year'])}|{row['馬名']}|{int(row['宝塚着順'])}|"
            f"{int(row['宝塚人気']) if pd.notna(row['宝塚人気']) else ''}|"
            f"{int(row['判定対象戦数'])}|"
            f"{record_text(int(row['タフ持続戦数']), int(row['タフ持続好走数']), tough_rate)}|"
            f"{int(row['低RPCI_PCI適性好走数'])}|{int(row['射程圏_機動力好走数'])}|"
            f"{int(row['後半5F上位好走数'])}|{str(row['参考好走']).replace('|', '/')[:130]}|"
        )

    missed = judged[judged["宝塚着順"].between(1, 3) & ~judged["宝塚キャラ該当"].eq("Y")].copy()
    lines.extend(
        [
            "",
            "## 拾えなかった馬券内馬",
            "",
            "|年|馬名|着順|人気|スコア|判定対象|タフ持続好走|低RPCI+PCI|射程/機動|",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if missed.empty:
        lines.append("|-|該当なし|||||||")
    else:
        for _, row in missed.sort_values(["year", "宝塚着順", "馬名"]).iterrows():
            lines.append(
                f"|{int(row['year'])}|{row['馬名']}|{int(row['宝塚着順'])}|"
                f"{int(row['宝塚人気']) if pd.notna(row['宝塚人気']) else ''}|"
                f"{int(row['キャラスコア'])}|{int(row['判定対象戦数'])}|"
                f"{int(row['タフ持続好走数'])}|{int(row['低RPCI_PCI適性好走数'])}|"
                f"{int(row['射程圏_機動力好走数'])}|"
            )

    lines.extend(
        [
            "",
            "## 読み取り",
            "",
            "- 宝塚キャラ該当は非該当より馬券内率が上がり、平均着順も改善した。ラップ・PCIキャラとしては宝塚記念成績と一定程度つながっている。",
            "- ただし該当馬の平均人気も高めで、人気以上に大きく走るフラグではない。能力上位馬を消さないための適性確認、または人気薄の下支え材料として扱うのがよい。",
            "- 拾えなかった馬券内馬にはサトノクラウン、ミッキーロケット、タイトルホルダー、イクイノックス、スルーセブンシーズなどがいる。単独の買い条件ではなく、当年の馬場・隊列・能力上位評価と組み合わせる補助特徴に回す。",
            "",
            "## 出力ファイル",
            "",
            f"- 集計CSV: `{detail_path.as_posix()}`",
            f"- 該当馬CSV: `{fit_path.as_posix()}`",
            f"- 過去走明細CSV: `{history_path.as_posix()}`",
            f"- レポート: `{report_path.as_posix()}`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"judged={len(judged)}")
    print(f"fit={len(fit)}")
    print(f"core={len(core)}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
