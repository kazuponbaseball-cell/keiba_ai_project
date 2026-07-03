from __future__ import annotations

import io
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "outputs/analysis/tokyo_20260613_12_shutuba.html"
RAW = ROOT / "date/raw/全競走馬成績.csv"
OUT_DIR = ROOT / "outputs/analysis"


def fmt_num(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    try:
        number = float(value)
        if not math.isfinite(number):
            return "-"
        return str(int(number)) if number.is_integer() else f"{number:.{digits}f}"
    except Exception:
        return str(value)


def fmt_record(frame: pd.DataFrame) -> str:
    finish = pd.to_numeric(frame["確定着順"], errors="coerce")
    starts = int(finish.notna().sum())
    if starts == 0:
        return "0-0-0-0"
    wins = int((finish == 1).sum())
    seconds = int((finish == 2).sum())
    thirds = int((finish == 3).sum())
    return f"{wins}-{seconds}-{thirds}-{starts - wins - seconds - thirds}"


def load_entries() -> pd.DataFrame:
    html = HTML.read_bytes().decode("euc_jp", errors="ignore")
    table = pd.read_html(io.StringIO(html))[0]
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [
            "_".join(str(part) for part in col if str(part) != "nan").strip("_")
            for col in table.columns
        ]
    table = table.rename(
        columns={
            "枠_枠": "枠",
            "馬 番_馬 番": "馬番",
            "馬名_馬名": "馬名",
            "性齢_性齢": "性齢",
            "斤量_斤量": "斤量",
            "騎手_騎手": "騎手",
            "厩舎_厩舎": "厩舎",
        }
    )
    entries = table[["枠", "馬番", "馬名", "性齢", "斤量", "騎手", "厩舎"]].copy()
    entries["馬名"] = entries["馬名"].astype(str).str.strip()
    html_ids = []
    for horse_id in re.findall(r"https://db\.netkeiba\.com/horse/(\d+)", html):
        if horse_id not in html_ids:
            html_ids.append(horse_id)
    entries["netkeiba_horse_id"] = html_ids[: len(entries)]
    return entries


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
    for col in [
        "日付",
        "人気",
        "距離",
        "走破タイム",
        "着差",
        "3角.1",
        "4角.1",
        "Ave-3F",
        "上り3F",
        "PCI",
        "PCI3",
        "RPCI",
        "確定着順",
    ]:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    return hist.sort_values(["馬名", "日付"])


def latest_lines(frame: pd.DataFrame) -> str:
    parts = []
    for _, row in frame.sort_values("日付", ascending=False).head(5).iterrows():
        parts.append(
            f"{fmt_num(row['日付'], 0)} {row['場所']}{fmt_num(row['Ｒ'], 0)}R {row['レース名']} "
            f"{fmt_num(row['確定着順'], 0)}着/{fmt_num(row['人気'], 0)}人気 "
            f"{row['芝・ダ']}{fmt_num(row['距離'], 0)} 4角{fmt_num(row['4角.1'], 0)} 上り{fmt_num(row['上り3F'])}"
        )
    return " / ".join(parts) if parts else "-"


def analyze_one(entry: pd.Series, hist: pd.DataFrame) -> dict[str, object]:
    name = entry["馬名"]
    h = hist[hist["馬名"].eq(name)].copy()
    dirt = h[h["芝・ダ"].eq("ダ")]
    d1400 = dirt[dirt["距離"].eq(1400)]
    tokyo_dirt = dirt[dirt["場所"].eq("東京")]
    tokyo_1400 = d1400[d1400["場所"].eq("東京")]
    recent = h.sort_values("日付", ascending=False).head(5)

    score = 0
    plus = []
    minus = []
    if len(d1400) >= 2:
        score += 2
        plus.append("ダ1400経験十分")
    elif len(d1400) == 1:
        score += 1
        plus.append("ダ1400経験あり")
    else:
        minus.append("ダ1400未知")

    if len(tokyo_1400):
        score += 2
        plus.append("東京ダ1400経験")
    elif len(tokyo_dirt):
        score += 1
        plus.append("東京ダ経験")

    d1400_finish = pd.to_numeric(d1400["確定着順"], errors="coerce")
    tokyo_1400_finish = pd.to_numeric(tokyo_1400["確定着順"], errors="coerce")
    recent_finish = pd.to_numeric(recent["確定着順"], errors="coerce")
    if not d1400.empty and d1400_finish.between(1, 3).any():
        score += 2
        plus.append("ダ1400好走あり")
    if not tokyo_1400.empty and tokyo_1400_finish.between(1, 3).any():
        score += 2
        plus.append("東京ダ1400好走")
    if not recent.empty and recent_finish.head(3).between(1, 3).any():
        score += 1
        plus.append("近走3着以内")
    try:
        if float(entry["斤量"]) <= 53:
            score += 1
            plus.append("軽斤量")
    except Exception:
        pass

    base_for_style = d1400 if not d1400.empty else dirt
    avg_4c = base_for_style["4角.1"].dropna().mean() if not base_for_style.empty else math.nan
    if pd.isna(avg_4c):
        style = "不明"
    elif avg_4c <= 4:
        style = "先行"
    elif avg_4c <= 9:
        style = "中団"
    else:
        style = "差し追込"

    if d1400.empty:
        minus.append("距離適性の裏付け不足")
    if len(tokyo_1400) == 0:
        minus.append("東京ダ1400の実績不足")

    return {
        **entry.to_dict(),
        "適性点": score,
        "脚質": style,
        "ダ成績": fmt_record(dirt),
        "ダ1400成績": fmt_record(d1400),
        "東京ダ成績": fmt_record(tokyo_dirt),
        "東京ダ1400成績": fmt_record(tokyo_1400),
        "平均4角": round(avg_4c, 1) if not pd.isna(avg_4c) else pd.NA,
        "ダ1400平均上り": round(d1400["上り3F"].dropna().mean(), 1) if not d1400.empty else pd.NA,
        "評価材料": "、".join(plus) if plus else "-",
        "懸念材料": "、".join(minus) if minus else "-",
        "近5走": latest_lines(recent),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = load_entries()
    hist = load_history(entries["馬名"].tolist())
    summary = pd.DataFrame([analyze_one(row, hist) for _, row in entries.iterrows()])
    summary = summary.sort_values(["適性点", "馬番"], ascending=[False, True])
    entries.to_csv(OUT_DIR / "tokyo_20260613_12_shutuba_clean.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "tokyo_20260613_12_field_analysis.csv", index=False, encoding="utf-8-sig")
    print(summary[["馬番", "馬名", "性齢", "斤量", "騎手", "適性点", "脚質", "ダ1400成績", "東京ダ1400成績", "評価材料", "懸念材料"]].to_string(index=False))


if __name__ == "__main__":
    main()
