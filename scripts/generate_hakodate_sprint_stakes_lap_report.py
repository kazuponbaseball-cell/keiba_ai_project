from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAP_CSV = ROOT / "outputs/analysis/hakodate_sprint_stakes_10y_race_laps.csv"
RAW_CSV = ROOT / "date/raw/全競走馬成績.csv"
OUT_DIR = ROOT / "outputs/analysis"


def parse_laps(value: object) -> list[float]:
    return [float(part) for part in str(value).split("-") if part and part != "nan"]


def fmt_num(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        number = float(value)
        if not math.isfinite(number):
            return ""
        return str(int(number)) if number.is_integer() else f"{number:.1f}"
    except Exception:
        return str(value)


def fmt_time(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        raw = int(float(text))
    except ValueError:
        return text
    minutes = raw // 1000
    remain = (raw % 1000) / 10
    return f"{minutes}:{remain:04.1f}"


def corner_position(row: pd.Series) -> str:
    values = []
    for col in ["3角.1", "4角.1"]:
        value = fmt_num(row.get(col))
        if value:
            values.append(value)
    return "-".join(values)


def pace_label(first3f: float, last3f: float) -> str:
    diff = round(first3f - last3f, 1)
    if diff <= -0.6:
        return "前傾"
    if diff >= 0.6:
        return "後傾"
    return "ミドル"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "年",
        "日付",
        "場所",
        "馬場",
        "勝ち時計",
        "レースラップ",
        "前半3F",
        "後半3F",
        "前後差",
        "傾向",
        "RPCI",
        "PCI3",
        "出走馬上り3F平均",
        "馬券内上り3F平均",
        "馬券内の位置取り",
    ]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.1f}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def summarize_average(label: str, frame: pd.DataFrame) -> dict[str, object]:
    lap_lists = [parse_laps(value) for value in frame["レースラップ"]]
    avg_laps = [round(sum(laps[idx] for laps in lap_lists) / len(lap_lists), 2) for idx in range(6)]
    return {
        "対象": label,
        "件数": len(frame),
        "平均ラップ": "-".join(f"{lap:.2f}" for lap in avg_laps),
        "前半3F平均": round(frame["前半3F"].mean(), 1),
        "後半3F平均": round(frame["後半3F"].mean(), 1),
        "前後差平均": round(frame["前後差"].mean(), 1),
        "RPCI平均": round(pd.to_numeric(frame["RPCI"], errors="coerce").mean(), 1),
        "PCI3平均": round(pd.to_numeric(frame["PCI3"], errors="coerce").mean(), 1),
        "出走馬上り3F平均": round(frame["出走馬上り3F平均"].mean(), 1),
        "馬券内上り3F平均": round(frame["馬券内上り3F平均"].mean(), 1),
    }


def main() -> None:
    laps = pd.read_csv(LAP_CSV, encoding="utf-8-sig")
    raw_cols = [
        "日付",
        "場所",
        "Ｒ",
        "レース名",
        "馬名",
        "人気",
        "馬場状態",
        "走破タイム",
        "3角.1",
        "4角.1",
        "Ave-3F",
        "上り3F",
        "PCI",
        "PCI3",
        "RPCI",
        "レースID(新/馬番無)",
        "確定着順",
    ]
    raw = pd.read_csv(RAW_CSV, encoding="cp932", usecols=raw_cols, low_memory=False)
    race_ids = set(laps["レースID"].astype(str))
    races = raw[raw["レースID(新/馬番無)"].astype(str).isin(race_ids)].copy()
    races["確定着順_num"] = pd.to_numeric(races["確定着順"], errors="coerce")
    races["日付_num"] = pd.to_numeric(races["日付"], errors="coerce")
    races["年"] = 2000 + (races["日付_num"] // 10000).astype("Int64")

    rows: list[dict[str, object]] = []
    for _, lap_row in laps.sort_values("日付").iterrows():
        year = int(lap_row["年"])
        race_rows = races[races["レースID(新/馬番無)"].astype(str).eq(str(lap_row["レースID"]))].copy()
        race_rows = race_rows.sort_values("確定着順_num")
        top3 = race_rows[race_rows["確定着順_num"].between(1, 3)]
        winner = race_rows[race_rows["確定着順_num"].eq(1)].head(1)
        lap_values = parse_laps(lap_row["レースラップタイム"])
        first3 = round(sum(lap_values[:3]), 1)
        last3 = round(sum(lap_values[-3:]), 1)

        top3_text = []
        for _, top_row in top3.iterrows():
            popularity = fmt_num(top_row.get("人気"))
            top3_text.append(
                f"{fmt_num(top_row['確定着順_num'])}着 {top_row['馬名']}"
                f"({corner_position(top_row)}/{popularity}人気)"
            )

        rows.append(
            {
                "年": year,
                "日付": str(lap_row["日付"]),
                "場所": race_rows["場所"].dropna().iloc[0] if race_rows["場所"].notna().any() else "",
                "馬場": race_rows["馬場状態"].dropna().iloc[0] if race_rows["馬場状態"].notna().any() else "",
                "レース名": race_rows["レース名"].dropna().iloc[0] if race_rows["レース名"].notna().any() else lap_row["レース名"],
                "勝ち時計": fmt_time(winner["走破タイム"].iloc[0]) if not winner.empty else "",
                "レースラップ": lap_row["レースラップタイム"],
                "前半3F": first3,
                "後半3F": last3,
                "前後差": round(first3 - last3, 1),
                "傾向": pace_label(first3, last3),
                "後半5F": round(sum(lap_values[-5:]), 1),
                "RPCI": round(float(race_rows["RPCI"].dropna().iloc[0]), 1) if race_rows["RPCI"].notna().any() else pd.NA,
                "PCI3": round(float(race_rows["PCI3"].dropna().iloc[0]), 1) if race_rows["PCI3"].notna().any() else pd.NA,
                "出走馬上り3F平均": round(pd.to_numeric(race_rows["上り3F"], errors="coerce").mean(), 1),
                "馬券内上り3F平均": round(pd.to_numeric(top3["上り3F"], errors="coerce").mean(), 1),
                "馬券内Ave-3F平均": round(pd.to_numeric(top3["Ave-3F"], errors="coerce").mean(), 1),
                "馬券内の位置取り": " / ".join(top3_text),
                "レース通過タイム": lap_row["レース通過タイム"],
                "レース上りタイム": lap_row["レース上りタイム"],
            }
        )

    summary = pd.DataFrame(rows)
    averages = pd.DataFrame(
        [
            summarize_average("全10年（2021札幌含む）", summary),
            summarize_average("函館開催のみ（2021札幌除外）", summary[summary["場所"].eq("函館")]),
        ]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "hakodate_sprint_stakes_10y_lap_report_table.csv"
    avg_csv = OUT_DIR / "hakodate_sprint_stakes_10y_lap_averages.csv"
    report = OUT_DIR / "hakodate_sprint_stakes_10y_lap_report.md"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    averages.to_csv(avg_csv, index=False, encoding="utf-8-sig")

    tendency_counts = summary["傾向"].value_counts().to_dict()
    front_inside_top3 = sum(
        1
        for text in summary["馬券内の位置取り"]
        for part in str(text).split(" / ")
        if part and part.split("(")[-1].split("/")[0].split("-")[-1].isdigit() and int(part.split("(")[-1].split("/")[0].split("-")[-1]) <= 4
    )
    total_top3 = len(summary) * 3
    avg_all = averages.iloc[0]
    avg_hakodate = averages.iloc[1]
    latest = summary.sort_values("年").iloc[-1]

    report.write_text(
        f"""# 函館スプリントS 過去10年ラップ傾向（2016-2025）

対象: 函館スプリントステークスの2016-2025年。2021年は札幌開催のため、全10年集計と函館開催のみ集計を分けた。ラップはTARGET内部RAレコード、PCI3/RPCI・通過順・上り3FはTARGET由来の全競走馬成績CSVから結合。

## サマリー

- 全10年の平均ラップ: {avg_all['平均ラップ']}
- 函館開催9年の平均ラップ: {avg_hakodate['平均ラップ']}
- 全10年前半3F平均: {avg_all['前半3F平均']:.1f}秒 / 後半3F平均: {avg_all['後半3F平均']:.1f}秒 / 前後差: {avg_all['前後差平均']:.1f}秒
- 函館開催のみ前半3F平均: {avg_hakodate['前半3F平均']:.1f}秒 / 後半3F平均: {avg_hakodate['後半3F平均']:.1f}秒 / 前後差: {avg_hakodate['前後差平均']:.1f}秒
- RPCI平均: 全10年 {avg_all['RPCI平均']:.1f} / 函館開催のみ {avg_hakodate['RPCI平均']:.1f}
- PCI3平均: 全10年 {avg_all['PCI3平均']:.1f} / 函館開催のみ {avg_hakodate['PCI3平均']:.1f}
- 出走馬上り3F平均: 全10年 {avg_all['出走馬上り3F平均']:.1f}秒 / 馬券内上り3F平均: {avg_all['馬券内上り3F平均']:.1f}秒
- 傾向内訳: 前傾 {tendency_counts.get('前傾', 0)}年 / ミドル {tendency_counts.get('ミドル', 0)}年 / 後傾 {tendency_counts.get('後傾', 0)}年
- 馬券内30頭中、4角4番手以内は {front_inside_top3}頭。短距離重賞だが、差しも毎年のように絡む。

## 年度別

{markdown_table(summary)}

## 平均

|対象|件数|平均ラップ|前半3F平均|後半3F平均|前後差平均|RPCI平均|PCI3平均|出走馬上り3F平均|馬券内上り3F平均|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
|{avg_all['対象']}|{int(avg_all['件数'])}|{avg_all['平均ラップ']}|{avg_all['前半3F平均']:.1f}|{avg_all['後半3F平均']:.1f}|{avg_all['前後差平均']:.1f}|{avg_all['RPCI平均']:.1f}|{avg_all['PCI3平均']:.1f}|{avg_all['出走馬上り3F平均']:.1f}|{avg_all['馬券内上り3F平均']:.1f}|
|{avg_hakodate['対象']}|{int(avg_hakodate['件数'])}|{avg_hakodate['平均ラップ']}|{avg_hakodate['前半3F平均']:.1f}|{avg_hakodate['後半3F平均']:.1f}|{avg_hakodate['前後差平均']:.1f}|{avg_hakodate['RPCI平均']:.1f}|{avg_hakodate['PCI3平均']:.1f}|{avg_hakodate['出走馬上り3F平均']:.1f}|{avg_hakodate['馬券内上り3F平均']:.1f}|

## 読み取りメモ

- 基本は明確な前傾戦。全10年で前半3F平均{avg_all['前半3F平均']:.1f}秒、後半3F平均{avg_all['後半3F平均']:.1f}秒。函館開催だけでも前半{avg_hakodate['前半3F平均']:.1f}秒、後半{avg_hakodate['後半3F平均']:.1f}秒でほぼ同じ。
- RPCI平均{avg_all['RPCI平均']:.1f}、函館開催のみ{avg_hakodate['RPCI平均']:.1f}。50を下回る年が大半で、終いの瞬発力勝負というより前半負荷と持続力のレース。
- ただし馬券内は逃げ・先行だけに偏らない。2017、2019、2022、2024は4角5番手以下の馬が複数絡んでおり、速すぎる入りでは差しの受け皿ができる。
- 4角4番手以内の馬券内率は30頭中{front_inside_top3}頭。軸は先行力を重視しつつ、相手には4角5-9番手あたりで上りをまとめる馬を残したい。
- {int(latest['年'])}年は前半{latest['前半3F']:.1f}、後半{latest['後半3F']:.1f}、RPCI{latest['RPCI']:.1f}で速い入りでも止まり切らない高速決着。時計対応力の下限は1分6秒台後半まで見たい。

## 出力ファイル

- 詳細CSV: `{out_csv.as_posix()}`
- 平均CSV: `{avg_csv.as_posix()}`
- レポート: `{report.as_posix()}`
""",
        encoding="utf-8-sig",
    )

    print(f"saved_report={report}")
    print(f"saved_table={out_csv}")
    print(f"saved_averages={avg_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
