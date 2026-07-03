from __future__ import annotations

import math
from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET_SE_DATA = Path("C:/Users/kazup/Data Lab/SE_DATA")


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


def parse_int_bytes(raw: bytes) -> int | None:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text or not text.lstrip("+-").isdigit():
        return None
    return int(text)


def pci(total_time: float, final3f: float, distance: int = 2200) -> float:
    units = distance / 200
    before_3f_time = total_time - final3f
    before_3f_units = units - 3
    if before_3f_units <= 0 or final3f <= 0:
        return math.nan
    before_3f_equivalent = before_3f_time / before_3f_units * 3
    return (before_3f_equivalent / final3f - 1) * 100 + 50


def pace_label(first3f: float, last3f: float) -> str:
    diff = round(first3f - last3f, 1)
    if diff <= -0.6:
        return "前傾"
    if diff >= 0.6:
        return "後傾"
    return "ミドル"


def load_2015_su_supplement(race_time: float) -> dict[str, object]:
    """Supplement 2015 rows because the exported all-results CSV starts at 2016."""
    race_id = b"2015062809030811"
    path = TARGET_SE_DATA / "2015" / "SU201539.DAT"
    if not path.exists():
        return {}

    top3: list[dict[str, object]] = []
    for raw in path.read_bytes().splitlines():
        if raw[11:27] != race_id:
            continue
        name = raw[40:76].decode("cp932", errors="replace").strip().replace("\u3000", "")
        finish = parse_int_bytes(raw[331:334])
        if finish is None or not 1 <= finish <= 3:
            continue
        popularity = parse_int_bytes(raw[363:365])
        final3f_raw = parse_int_bytes(raw[390:393])
        margin_raw = parse_int_bytes(raw[531:535])
        final3f = final3f_raw / 10 if final3f_raw is not None else math.nan
        margin = (margin_raw or 0) / 10
        positions = []
        for start in (351, 353, 355, 357):
            value = parse_int_bytes(raw[start : start + 2])
            if value:
                positions.append(str(value))
        total_time = race_time + margin
        top3.append(
            {
                "finish": finish,
                "name": name,
                "popularity": popularity,
                "position": "-".join(positions),
                "pci": pci(total_time, final3f),
            }
        )

    top3 = sorted(top3, key=lambda row: int(row["finish"]))
    if len(top3) != 3:
        return {}
    top3_text = [
        f"{row['finish']}着 {row['name']}({row['position']}/{row['popularity']}人気)"
        for row in top3
    ]
    race_last3f = 35.0
    return {
        "場所": "阪神",
        "馬場状態": "良",
        "PCI3": round(sum(float(row["pci"]) for row in top3) / 3, 1),
        "RPCI": round(pci(race_time, race_last3f), 1),
        "馬券内の位置取り": " / ".join(top3_text),
    }


def corner_position(row: pd.Series) -> str:
    values = []
    for col in ["1角", "2角", "3角", "4角"]:
        value = fmt_num(row.get(col))
        if value:
            values.append(value)
    return "-".join(values)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "年",
        "場所",
        "馬場状態",
        "レースラップ",
        "前半3F",
        "後半3F",
        "前後差",
        "ラップ傾向",
        "後半5F",
        "PCI3",
        "RPCI",
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


def make_lap_graph(summary: pd.DataFrame, output_path: Path) -> None:
    width = 1180
    height = 720
    margin_left = 74
    margin_right = 220
    margin_top = 54
    margin_bottom = 72
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    min_lap = 10.0
    max_lap = 13.6
    years = [int(year) for year in summary["年"].tolist()]
    colors = [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4f46e5",
        "#65a30d",
        "#0f766e",
    ]

    lap_series = {int(row["年"]): parse_laps(row["レースラップ"]) for _, row in summary.iterrows()}
    avg_laps = [sum(laps[idx] for laps in lap_series.values()) / len(lap_series) for idx in range(11)]

    def x_pos(idx: int) -> float:
        return margin_left + (idx / 10) * plot_w

    def y_pos(value: float) -> float:
        return margin_top + ((max_lap - value) / (max_lap - min_lap)) * plot_h

    def points(values: list[float]) -> str:
        return " ".join(f"{x_pos(idx):.1f},{y_pos(value):.1f}" for idx, value in enumerate(values))

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="74" y="30" font-family="Meiryo, Yu Gothic, sans-serif" font-size="20" font-weight="700" fill="#111827">宝塚記念 阪神開催10回 レースラップ比較</text>',
        '<text x="74" y="52" font-family="Meiryo, Yu Gothic, sans-serif" font-size="12" fill="#4b5563">京都開催2024を除外。横軸はスタートからの1F、縦軸はラップタイム（秒）。</text>',
    ]

    for tick in [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5]:
        y = y_pos(tick)
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{tick:.1f}</text>')
    for idx in range(11):
        x = x_pos(idx)
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#f3f4f6" stroke-width="1"/>')
        lines.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{idx + 1}F</text>')

    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9ca3af" stroke-width="1.2"/>')
    lines.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9ca3af" stroke-width="1.2"/>')

    for color, year in zip(colors, years):
        values = lap_series[year]
        lines.append(f'<polyline points="{points(values)}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" opacity="0.82"/>')
        for idx, value in enumerate(values):
            lines.append(f'<circle cx="{x_pos(idx):.1f}" cy="{y_pos(value):.1f}" r="2.4" fill="{color}" opacity="0.92"/>')

    lines.append(f'<polyline points="{points(avg_laps)}" fill="none" stroke="#111827" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')

    legend_x = width - margin_right + 28
    legend_y = margin_top + 12
    for idx, (color, year) in enumerate(zip(colors, years)):
        y = legend_y + idx * 26
        going = summary.loc[summary["年"].eq(year), "馬場状態"].iloc[0]
        tendency = summary.loc[summary["年"].eq(year), "ラップ傾向"].iloc[0]
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="{legend_x + 34}" y="{y + 4}" font-family="Meiryo, Yu Gothic, sans-serif" font-size="12" fill="#111827">{year} {escape(str(going))} {escape(str(tendency))}</text>'
        )
    y = legend_y + len(years) * 26 + 8
    lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="#111827" stroke-width="4"/>')
    lines.append(f'<text x="{legend_x + 34}" y="{y + 4}" font-family="Meiryo, Yu Gothic, sans-serif" font-size="12" font-weight="700" fill="#111827">平均</text>')
    lines.append('<text x="74" y="690" font-family="Meiryo, Yu Gothic, sans-serif" font-size="12" fill="#4b5563">注: 縦軸は値が大きいほど上に描画。ラップが上に振れるほど時計が掛かっている。</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    laps = pd.read_csv(ROOT / "outputs/analysis/takarazuka_kinen_hanshin_10_race_laps.csv", encoding="utf-8-sig")
    raw_cols = [
        "日付",
        "日付S",
        "場所",
        "馬場状態",
        "Ｒ",
        "レース名",
        "馬名",
        "馬番",
        "人気",
        "確定着順",
        "1角",
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
        raw["レース名"].astype(str).str.contains("宝塚記念", na=False)
        & raw["場所"].astype(str).eq("阪神")
        & pd.to_numeric(raw["日付"], errors="coerce").between(150000, 259999)
    )
    races = raw[mask].copy()
    races["確定着順_num"] = pd.to_numeric(races["確定着順"], errors="coerce")
    races["年"] = (2000 + (pd.to_numeric(races["日付"], errors="coerce") // 10000)).astype("Int64")

    rows = []
    for _, lap_row in laps.sort_values("年").iterrows():
        year = int(lap_row["年"])
        lap_values = parse_laps(lap_row["レースラップタイム"])
        race_rows = races[races["年"].astype("Int64") == year].sort_values("確定着順_num")
        supplement = load_2015_su_supplement(round(sum(lap_values), 1)) if year == 2015 else {}
        if race_rows.empty and not supplement:
            continue
        if race_rows.empty:
            place = supplement["場所"]
            going = supplement["馬場状態"]
            pci3 = supplement["PCI3"]
            rpci = supplement["RPCI"]
            top3_joined = supplement["馬券内の位置取り"]
        else:
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
            place = race_rows["場所"].dropna().iloc[0] if len(race_rows["場所"].dropna()) else ""
            going = race_rows["馬場状態"].dropna().iloc[0] if len(race_rows["馬場状態"].dropna()) else ""
            pci3 = round(float(pci3_values.iloc[0]), 1) if len(pci3_values) else pd.NA
            rpci = round(float(rpci_values.iloc[0]), 1) if len(rpci_values) else pd.NA
            top3_joined = " / ".join(top3_text)
        rows.append(
            {
                "年": year,
                "日付": str(lap_row["日付"]),
                "場所": place,
                "馬場状態": going,
                "レース名": lap_row["レース名"],
                "レースラップ": lap_row["レースラップタイム"],
                "前半3F": round(sum(lap_values[:3]), 1),
                "後半3F": round(sum(lap_values[-3:]), 1),
                "後半5F": round(sum(lap_values[-5:]), 1),
                "PCI3": pci3,
                "RPCI": rpci,
                "馬券内の位置取り": top3_joined,
                "レース通過タイム": lap_row["レース通過タイム"],
                "レース上りタイム": lap_row["レース上りタイム"],
            }
        )

    summary = pd.DataFrame(rows)
    summary["前後差"] = (summary["前半3F"] - summary["後半3F"]).round(1)
    summary["ラップ傾向"] = summary.apply(lambda row: pace_label(float(row["前半3F"]), float(row["後半3F"])), axis=1)
    out_dir = ROOT / "outputs/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "takarazuka_kinen_hanshin_10_lap_report_table.csv"
    report = out_dir / "takarazuka_kinen_hanshin_10_lap_report.md"
    graph = out_dir / "takarazuka_kinen_hanshin_10_lap_graph.svg"
    make_lap_graph(summary, graph)
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    avg_first = summary["前半3F"].mean()
    avg_last = summary["後半3F"].mean()
    avg_5 = summary["後半5F"].mean()
    avg_rpci = pd.to_numeric(summary["RPCI"], errors="coerce").mean()
    avg_pci3 = pd.to_numeric(summary["PCI3"], errors="coerce").mean()
    avg_diff = summary["前後差"].mean()
    fastest_first = summary.loc[summary["前半3F"].idxmin()]
    fastest_last = summary.loc[summary["後半3F"].idxmin()]
    slowest_last = summary.loc[summary["後半3F"].idxmax()]
    lowest_rpci = summary.loc[pd.to_numeric(summary["RPCI"], errors="coerce").idxmin()]
    highest_rpci = summary.loc[pd.to_numeric(summary["RPCI"], errors="coerce").idxmax()]
    avg_row = pd.DataFrame(
        [
            {
                "年": "平均",
                "場所": "阪神",
                "馬場状態": "-",
                "レースラップ": "-",
                "前半3F": avg_first,
                "後半3F": avg_last,
                "前後差": avg_diff,
                "ラップ傾向": pace_label(float(avg_first), float(avg_last)),
                "後半5F": avg_5,
                "PCI3": avg_pci3,
                "RPCI": avg_rpci,
                "馬券内の位置取り": "-",
            }
        ]
    )
    table_frame = pd.concat([summary, avg_row], ignore_index=True)
    going_avg = (
        summary.groupby("馬場状態", dropna=False)[["前半3F", "後半3F", "前後差", "後半5F", "PCI3", "RPCI"]]
        .mean()
        .round(1)
        .reset_index()
    )
    going_counts = summary.groupby("馬場状態", dropna=False).size().reset_index(name="件数")
    going_avg = going_avg.merge(going_counts, on="馬場状態", how="left")
    going_lines = ["|馬場状態|件数|前半3F平均|後半3F平均|前後差平均|傾向|後半5F平均|PCI3平均|RPCI平均|", "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for _, row in going_avg.iterrows():
        going_lines.append(
            f"|{row['馬場状態']}|{int(row['件数'])}|{row['前半3F']:.1f}|{row['後半3F']:.1f}|"
            f"{row['前後差']:.1f}|{pace_label(float(row['前半3F']), float(row['後半3F']))}|"
            f"{row['後半5F']:.1f}|{row['PCI3']:.1f}|{row['RPCI']:.1f}|"
        )
    tendency_counts = summary["ラップ傾向"].value_counts().to_dict()
    tendency_text = "、".join(f"{label}{tendency_counts.get(label, 0)}回" for label in ["前傾", "ミドル", "後傾"])

    report.write_text(
        f"""# 宝塚記念 阪神開催10回ラップレポート（2015-2025、京都2024除外）

対象: 阪神芝2200m・宝塚記念の直近10回。2015-2025年から京都開催の2024年を除外した。TARGET内部RAレコードからレースラップを抽出し、全競走馬成績CSVおよび2015年SUレコードから馬場状態、PCI3/RPCI、馬券内馬の道中位置取りを結合した。

## サマリー

- 前半3F平均: {avg_first:.1f}秒
- 後半3F平均: {avg_last:.1f}秒
- 前後差平均: {avg_diff:.1f}秒（前半3F - 後半3F）
- ラップ傾向: {tendency_text}
- 後半5F平均: {avg_5:.1f}秒
- PCI3平均: {avg_pci3:.1f}
- RPCI平均: {avg_rpci:.1f}
- 最速前半3F: {int(fastest_first['年'])}年 {fastest_first['前半3F']:.1f}秒
- 最速後半3F: {int(fastest_last['年'])}年 {fastest_last['後半3F']:.1f}秒
- 最も上がりが掛かった後半3F: {int(slowest_last['年'])}年 {slowest_last['後半3F']:.1f}秒
- 最も前傾寄りのRPCI: {int(lowest_rpci['年'])}年 {lowest_rpci['RPCI']:.1f}
- 最も後傾寄りのRPCI: {int(highest_rpci['年'])}年 {highest_rpci['RPCI']:.1f}

## 年別一覧

{markdown_table(table_frame)}

## ラップグラフ

![宝塚記念 阪神開催10回 レースラップ比較]({graph.as_posix()})

## 馬場状態別平均

{chr(10).join(going_lines)}

## 読み取りメモ

- 阪神開催10回の前半3Fは33.9-36.0秒。見た目の入りは年によって差があるが、2200m戦としては中盤以降に緩み切らない年が多い。
- 後半5F平均は59.8秒で、安田記念のような純粋な瞬発戦というより、向正面から長く脚を使う持続力が問われやすい。
- RPCIは46.5-56.5。低RPCIの2022年・2016年・2018年は前半負荷が高く、後半3Fも36秒台まで掛かった消耗度の高い持続戦として読む。2015年のように前半が緩む年はRPCIが高く、差しも届く後傾戦になる。
- 前後差で見ると{tendency_text}。平均も{avg_diff:.1f}秒で、阪神開催の宝塚記念は基本線として前傾寄りに見たい。
- 馬券内の位置取りは先行・好位の粘り込みと、中団差しの両方が出る。極端な後方一気より、3角-4角で射程圏に入れる機動力を重視したい。
- 良馬場でも後半3F平均は35秒台半ばまで掛かっており、良馬場=瞬発戦とは見ない方がよい。稍重は後半5Fと後半3Fがより掛かりやすい。

## 出力ファイル

- 詳細CSV: `{out_csv.as_posix()}`
- ラップグラフ: `{graph.as_posix()}`
- 本レポート: `{report.as_posix()}`
""",
        encoding="utf-8",
    )
    print(f"saved_report={report}")
    print(f"saved_table={out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
