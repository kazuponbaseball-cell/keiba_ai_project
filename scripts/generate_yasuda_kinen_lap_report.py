from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


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


def corner_position(row: pd.Series) -> str:
    values = []
    for col in ["2角", "3角", "4角"]:
        value = fmt_num(row.get(col))
        if value:
            values.append(value)
    return "-".join(values)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = ["年", "レースラップ", "前半3F", "後半3F", "後半5F", "PCI3", "RPCI", "馬券内の位置取り"]
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


def main() -> None:
    laps = pd.read_csv(ROOT / "outputs/analysis/yasuda_kinen_10y_race_laps.csv", encoding="utf-8-sig")
    raw_cols = [
        "日付",
        "日付S",
        "場所",
        "Ｒ",
        "レース名",
        "馬名",
        "馬番",
        "人気",
        "確定着順",
        "2角",
        "3角",
        "4角",
        "PCI3",
        "RPCI",
        "レースID(新/馬番無)",
    ]
    raw = pd.read_csv(
        ROOT / "date/raw/全競走馬成績.csv",
        encoding="cp932",
        usecols=lambda col: col in raw_cols,
        low_memory=False,
    )
    mask = (
        raw["レース名"].astype(str).str.contains("安田記念", na=False)
        & raw["場所"].astype(str).eq("東京")
        & pd.to_numeric(raw["日付"], errors="coerce").between(160000, 259999)
    )
    races = raw[mask].copy()
    races["確定着順_num"] = pd.to_numeric(races["確定着順"], errors="coerce")
    races["年"] = (2000 + (pd.to_numeric(races["日付"], errors="coerce") // 10000)).astype("Int64")

    rows = []
    for _, lap_row in laps.sort_values("年").iterrows():
        year = int(lap_row["年"])
        lap_values = parse_laps(lap_row["レースラップタイム"])
        race_rows = races[races["年"].astype("Int64") == year].sort_values("確定着順_num")
        top3 = race_rows[race_rows["確定着順_num"].between(1, 3)]
        top3_text = []
        for _, top_row in top3.iterrows():
            popularity = fmt_num(top_row.get("人気"))
            popularity_text = f"/{popularity}人気" if popularity else ""
            top3_text.append(
                f"{fmt_num(top_row['確定着順_num'])}着 {top_row['馬名']}({corner_position(top_row)}{popularity_text})"
            )

        rpci_values = race_rows["RPCI"].dropna()
        pci3_values = race_rows["PCI3"].dropna()
        rows.append(
            {
                "年": year,
                "日付": str(lap_row["日付"]),
                "レース名": lap_row["レース名"],
                "レースラップ": lap_row["レースラップタイム"],
                "前半3F": round(sum(lap_values[:3]), 1),
                "後半3F": round(sum(lap_values[-3:]), 1),
                "後半5F": round(sum(lap_values[-5:]), 1),
                "PCI3": round(float(pci3_values.iloc[0]), 1) if len(pci3_values) else pd.NA,
                "RPCI": round(float(rpci_values.iloc[0]), 1) if len(rpci_values) else pd.NA,
                "馬券内の位置取り": " / ".join(top3_text),
                "レース通過タイム": lap_row["レース通過タイム"],
                "レース上りタイム": lap_row["レース上りタイム"],
            }
        )

    summary = pd.DataFrame(rows)
    out_dir = ROOT / "outputs/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "yasuda_kinen_10y_lap_report_table.csv"
    report = out_dir / "yasuda_kinen_10y_lap_report.md"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    avg_first = summary["前半3F"].mean()
    avg_last = summary["後半3F"].mean()
    avg_5 = summary["後半5F"].mean()
    avg_rpci = pd.to_numeric(summary["RPCI"], errors="coerce").mean()
    avg_pci3 = pd.to_numeric(summary["PCI3"], errors="coerce").mean()
    fastest_first = summary.loc[summary["前半3F"].idxmin()]
    fastest_last = summary.loc[summary["後半3F"].idxmin()]
    slowest_last = summary.loc[summary["後半3F"].idxmax()]

    report.write_text(
        f"""# 安田記念 過去10年ラップレポート（2016-2025）

対象: 東京芝1600m・安田記念の2016-2025年。TARGET内部RAレコードからレースラップを抽出し、全競走馬成績CSVからPCI3/RPCIと馬券内馬の道中位置取りを結合した。

## サマリー

- 前半3F平均: {avg_first:.1f}秒
- 後半3F平均: {avg_last:.1f}秒
- 後半5F平均: {avg_5:.1f}秒
- PCI3平均: {avg_pci3:.1f}
- RPCI平均: {avg_rpci:.1f}
- 最速前半3F: {int(fastest_first['年'])}年 {fastest_first['前半3F']:.1f}秒
- 最速後半3F: {int(fastest_last['年'])}年 {fastest_last['後半3F']:.1f}秒
- 最も上がりが掛かった後半3F: {int(slowest_last['年'])}年 {slowest_last['後半3F']:.1f}秒

## 年別一覧

{markdown_table(summary)}

## 読み取りメモ

- 過去10年の前半3Fは33.9-35.0秒の範囲で、極端な超スローは少ない。
- 後半3Fは33.6-34.5秒。勝負所は残り3Fからの瞬発力だけでなく、後半5F全体の持続力も問われやすい。
- RPCIは48.8-54.8まで幅があり、年によって前傾寄り・後傾寄りが変わる。RPCIが低い年は前半負荷が高く、差し・持続型が届きやすい傾向として読む。
- 馬券内の位置取りは中団から差しの好走が多い一方、2016年のロゴタイプ、2025年のジャンタルマンタルのように好位で運んだ馬も勝ち切っている。
- 東京1600mのため、位置取り列は主に3角-4角で評価している。

## 出力ファイル

- 詳細CSV: `{out_csv.as_posix()}`
- 本レポート: `{report.as_posix()}`
""",
        encoding="utf-8",
    )
    print(f"saved_report={report}")
    print(f"saved_table={out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
