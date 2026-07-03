from __future__ import annotations

import io
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "outputs/analysis/hakodate_sprint_2026_shutuba.html"
RAW = ROOT / "date/raw/全競走馬成績.csv"
OUT_DIR = ROOT / "outputs/analysis"

RACE_URL = "https://race.netkeiba.com/race/shutuba.html?race_id=202602010111"
RACE_AVG_RPCI = 46.1
RACE_AVG_FIRST3F = 33.1
RACE_AVG_LAST3F = 34.5


def fmt_num(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    try:
        number = float(value)
        if not math.isfinite(number):
            return "-"
        if number.is_integer():
            return str(int(number))
        return f"{number:.{digits}f}"
    except Exception:
        return str(value)


def fmt_record(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "0-0-0-0"
    finish = pd.to_numeric(frame["確定着順"], errors="coerce")
    starts = int(finish.notna().sum())
    wins = int((finish == 1).sum())
    seconds = int((finish == 2).sum())
    thirds = int((finish == 3).sum())
    others = starts - wins - seconds - thirds
    return f"{wins}-{seconds}-{thirds}-{others}"


def fmt_time(raw: object) -> str:
    if pd.isna(raw):
        return "-"
    text = str(raw).strip()
    if not text:
        return "-"
    try:
        value = int(float(text))
    except ValueError:
        return text
    minutes = value // 1000
    remain = (value % 1000) / 10
    return f"{minutes}:{remain:04.1f}"


def score_label(score: int) -> str:
    if score >= 8:
        return "A"
    if score >= 6:
        return "B"
    if score >= 4:
        return "C"
    return "D"


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(part) for part in col if str(part) != "nan").strip("_")
            for col in out.columns
        ]
    return out


def load_shutuba() -> pd.DataFrame:
    html = HTML.read_bytes().decode("euc_jp", errors="ignore")
    table = flatten_columns(pd.read_html(io.StringIO(html))[0])
    table = table.rename(
        columns={
            "枠_枠": "枠",
            "馬 番_馬 番": "馬番",
            "馬名_馬名": "馬名",
            "性齢_性齢": "性齢",
            "斤量_斤量": "斤量",
            "騎手_騎手": "騎手",
            "厩舎_厩舎": "厩舎",
            "人気_人気": "人気",
        }
    )
    cols = ["枠", "馬番", "馬名", "性齢", "斤量", "騎手", "厩舎", "人気"]
    table = table[cols].copy()
    table["馬名"] = table["馬名"].astype(str).str.strip()

    horse_ids = []
    for horse_id in re.findall(r"https://db\.netkeiba\.com/horse/(\d+)", html):
        if horse_id not in horse_ids:
            horse_ids.append(horse_id)
    table["netkeiba_horse_id"] = horse_ids[: len(table)]
    return table


def load_history(names: list[str]) -> pd.DataFrame:
    cols = [
        "日付",
        "場所",
        "Ｒ",
        "レース名",
        "馬名",
        "人気",
        "芝・ダ",
        "距離",
        "馬場状態",
        "走破タイム",
        "着差",
        "3角.1",
        "4角.1",
        "Ave-3F",
        "上り3F",
        "上り3F順",
        "PCI",
        "PCI3",
        "RPCI",
        "確定着順",
    ]
    hist = pd.read_csv(RAW, encoding="cp932", usecols=lambda c: c in cols, low_memory=False)
    hist = hist[hist["馬名"].isin(names)].copy()
    for col in ["日付", "距離", "人気", "走破タイム", "着差", "3角.1", "4角.1", "Ave-3F", "上り3F", "PCI", "PCI3", "RPCI", "確定着順"]:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    return hist.sort_values(["馬名", "日付"])


def latest_lines(frame: pd.DataFrame, n: int = 5) -> str:
    if frame.empty:
        return "履歴なし"
    parts = []
    for _, row in frame.sort_values("日付", ascending=False).head(n).iterrows():
        parts.append(
            f"{int(row['日付'])} {row['場所']}{int(row['Ｒ'])}R {row['レース名']} "
            f"{fmt_num(row['確定着順'])}着/{fmt_num(row['人気'])}人気 "
            f"{row['芝・ダ']}{fmt_num(row['距離'], 0)}m "
            f"4角{fmt_num(row['4角.1'])} 上り{fmt_num(row['上り3F'])} RPCI{fmt_num(row['RPCI'])}"
        )
    return " / ".join(parts)


def analyze_one(entry: pd.Series, hist: pd.DataFrame) -> dict[str, object]:
    name = entry["馬名"]
    h = hist[hist["馬名"].eq(name)].copy()
    turf1200 = h[h["芝・ダ"].eq("芝") & h["距離"].eq(1200)]
    hak_sap_1200 = turf1200[turf1200["場所"].isin(["函館", "札幌"])]
    fast_rpci = turf1200[turf1200["RPCI"].le(RACE_AVG_RPCI)]
    fast_clock = turf1200[turf1200["走破タイム"].le(1080)]
    recent = h.sort_values("日付", ascending=False).head(5)
    recent_turf1200 = turf1200.sort_values("日付", ascending=False).head(5)

    finish = pd.to_numeric(turf1200["確定着順"], errors="coerce")
    top3_rate = float((finish <= 3).mean()) if finish.notna().any() else math.nan
    win_rate = float((finish == 1).mean()) if finish.notna().any() else math.nan
    avg_4c = float(turf1200["4角.1"].dropna().mean()) if turf1200["4角.1"].notna().any() else math.nan
    avg_rpci = float(turf1200["RPCI"].dropna().mean()) if turf1200["RPCI"].notna().any() else math.nan
    avg_last = float(turf1200["上り3F"].dropna().mean()) if turf1200["上り3F"].notna().any() else math.nan
    best_time = turf1200["走破タイム"].dropna().min() if turf1200["走破タイム"].notna().any() else math.nan

    score = 0
    reasons = []
    cautions = []

    if len(turf1200) >= 3:
        score += 1
        reasons.append("芝1200の経験量は十分")
    else:
        cautions.append("芝1200のサンプルが少ない")
    if not pd.isna(top3_rate) and top3_rate >= 0.45:
        score += 2
        reasons.append("芝1200で馬券圏内率が高い")
    elif not pd.isna(top3_rate) and top3_rate >= 0.25:
        score += 1
        reasons.append("芝1200で一定の安定感")
    else:
        cautions.append("芝1200の安定感は強調しづらい")
    if len(fast_rpci):
        score += 2
        reasons.append(f"RPCI{RACE_AVG_RPCI:.1f}以下の前傾1200経験あり")
    else:
        cautions.append("函館SS級の前傾RPCI経験が薄い")
    if len(fast_clock):
        score += 1
        reasons.append("1分08秒0以内の時計対応あり")
    else:
        cautions.append("高速時計の裏付けが弱い")
    if not pd.isna(avg_4c):
        if avg_4c <= 4.5:
            score += 2
            reasons.append("4角好位を取れる")
        elif avg_4c <= 9:
            score += 1
            reasons.append("差しの受け皿になりやすい中団型")
        else:
            cautions.append("位置が後ろになりやすい")
    if not hak_sap_1200.empty:
        score += 1
        reasons.append("函館/札幌の芝1200経験あり")
    else:
        cautions.append("洋芝1200の実戦経験が乏しい")
    if not recent.empty and (pd.to_numeric(recent["確定着順"], errors="coerce") <= 3).head(3).any():
        score += 1
        reasons.append("近走で3着以内あり")

    style = "不明"
    if not pd.isna(avg_4c):
        if avg_4c <= 3.5:
            style = "逃げ先行"
        elif avg_4c <= 7.5:
            style = "好位差し"
        elif avg_4c <= 11:
            style = "中団差し"
        else:
            style = "後方差し"

    return {
        "枠": entry["枠"],
        "馬番": entry["馬番"],
        "馬名": name,
        "性齢": entry["性齢"],
        "斤量": entry["斤量"],
        "騎手": entry["騎手"],
        "厩舎": entry["厩舎"],
        "netkeiba_horse_id": entry["netkeiba_horse_id"],
        "適性評価": score_label(score),
        "適性点": score,
        "脚質推定": style,
        "芝1200成績": fmt_record(turf1200),
        "函館札幌芝1200成績": fmt_record(hak_sap_1200),
        "前傾1200成績": fmt_record(fast_rpci),
        "芝1200最高時計": fmt_time(best_time),
        "芝1200平均4角": round(avg_4c, 1) if not pd.isna(avg_4c) else pd.NA,
        "芝1200平均上り": round(avg_last, 1) if not pd.isna(avg_last) else pd.NA,
        "芝1200平均RPCI": round(avg_rpci, 1) if not pd.isna(avg_rpci) else pd.NA,
        "評価材料": "、".join(reasons[:4]) if reasons else "-",
        "懸念材料": "、".join(cautions[:3]) if cautions else "-",
        "近5走": latest_lines(h, 5),
        "近芝1200": latest_lines(recent_turf1200, 5),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    cols = ["枠", "馬番", "馬名", "性齢", "斤量", "騎手", "厩舎", "適性評価", "適性点", "脚質推定", "芝1200成績", "函館札幌芝1200成績", "前傾1200成績", "芝1200最高時計"]
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in frame[cols].iterrows():
        lines.append("|" + "|".join(str(row[col]).replace("|", "/") for col in cols) + "|")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame) -> None:
    out_csv = OUT_DIR / "hakodate_sprint_2026_field_analysis.csv"
    out_md = OUT_DIR / "hakodate_sprint_2026_field_analysis.md"
    card_csv = OUT_DIR / "hakodate_sprint_2026_shutuba_clean.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    summary[["枠", "馬番", "馬名", "性齢", "斤量", "騎手", "厩舎", "netkeiba_horse_id"]].to_csv(card_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# 2026 函館スプリントS 出馬表・個別適性分析",
        "",
        f"出馬表ソース: {RACE_URL}",
        "レース条件: 2026年6月13日 函館11R、芝1200m Aコース、13頭。netkeiba取得時点では天候:晴、馬場:稍。",
        "",
        "過去10年の基準: 函館開催9年平均で前半3F 33.1、後半3F 34.5、RPCI 46.1、PCI3 48.4。基本は前傾持続戦で、4角4番手以内が馬券内30頭中16頭。ただし中団差しも届く。",
        "",
        "## 出馬表と適性サマリー",
        "",
        markdown_table(summary),
        "",
        "## 個別分析",
        "",
    ]
    for _, row in summary.sort_values(["馬番"]).iterrows():
        lines.extend(
            [
                f"### {int(row['馬番'])}. {row['馬名']}（{row['枠']}枠、{row['性齢']}、{row['斤量']}kg、{row['騎手']}）",
                "",
                f"- 総合: {row['適性評価']}（{row['適性点']}点）。脚質推定は{row['脚質推定']}。",
                f"- 芝1200: {row['芝1200成績']}、函館/札幌芝1200: {row['函館札幌芝1200成績']}、前傾1200: {row['前傾1200成績']}。",
                f"- 指標: 最高時計 {row['芝1200最高時計']}、平均4角 {fmt_num(row['芝1200平均4角'])}、平均上り {fmt_num(row['芝1200平均上り'])}、平均RPCI {fmt_num(row['芝1200平均RPCI'])}。",
                f"- 合う材料: {row['評価材料']}。",
                f"- 気になる材料: {row['懸念材料']}。",
                f"- 近5走: {row['近5走']}",
                f"- 近芝1200: {row['近芝1200']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 現時点の見立て",
            "",
            "- 最も傾向に噛み合うのは、前傾1200の経験と好位性能を両立する馬。",
            "- 今年は稍重表示なので、単純な上がり最速よりも、前半33秒台前半から粘れる馬を上に取りたい。",
            "- 後方一気型は、過去傾向上まったく消しではないが、13頭立てでも前が完全に潰れる読みが必要。",
            "",
            "## 出力ファイル",
            "",
            f"- 出馬表CSV: `{card_csv.as_posix()}`",
            f"- 個別分析CSV: `{out_csv.as_posix()}`",
            f"- レポート: `{out_md.as_posix()}`",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"saved_card={card_csv}")
    print(f"saved_csv={out_csv}")
    print(f"saved_report={out_md}")


def main() -> None:
    entries = load_shutuba()
    hist = load_history(entries["馬名"].tolist())
    summary = pd.DataFrame([analyze_one(row, hist) for _, row in entries.iterrows()])
    summary = summary.sort_values(["馬番"])
    write_report(summary)
    print(summary[["馬番", "馬名", "適性評価", "適性点", "脚質推定", "芝1200成績", "前傾1200成績", "芝1200最高時計", "評価材料", "懸念材料"]].to_string(index=False))


if __name__ == "__main__":
    main()
