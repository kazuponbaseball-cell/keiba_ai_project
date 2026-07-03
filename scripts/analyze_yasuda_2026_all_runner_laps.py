from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET_SE_DATA = Path("C:/Users/kazup/Data Lab/SE_DATA")

RAW_USECOLS = [
    1,
    2,
    3,
    6,
    7,
    9,
    10,
    12,
    14,
    15,
    16,
    20,
    23,
    27,
    28,
    34,
    35,
    60,
    61,
    62,
    66,
    82,
    84,
    85,
    86,
    87,
    91,
    94,
    95,
    96,
    97,
    99,
    100,
    129,
    152,
    154,
    157,
    183,
    184,
]

RAW_NAMES = [
    "date_full",
    "date_s",
    "date",
    "venue",
    "r",
    "race_name",
    "class_name",
    "horse_name",
    "sex",
    "age",
    "jockey",
    "weight",
    "field",
    "frame",
    "horse_no",
    "pop",
    "odds",
    "finish_raw",
    "surface",
    "distance",
    "going",
    "margin",
    "diff_at_3f",
    "c2",
    "c3",
    "c4",
    "c4b",
    "ave3f",
    "final3f",
    "final3f_rank",
    "pci",
    "pci3",
    "rpci",
    "career",
    "race_id",
    "horse_id",
    "finish",
    "interval",
    "fresh_run",
]

RA_RACE_ID_OFFSET = 11
RA_RACE_ID_LEN = 16
RA_LAP_OFFSET = 890
RA_LAP_COUNT = 25
RA_LAP_WIDTH = 3


def load_entries() -> pd.DataFrame:
    entries = pd.read_csv(
        ROOT / "data/datasets/inference/weekly/entry_snapshot.csv",
        encoding="utf-8-sig",
        dtype=str,
    )
    return entries[["枠番", "馬番", "馬名", "血統登録番号", "騎手", "性別", "年齢"]].copy()


def load_raw_results() -> pd.DataFrame:
    raw_path = next((ROOT / "date/raw").glob("*.csv"))
    raw = pd.read_csv(raw_path, encoding="cp932", usecols=RAW_USECOLS, low_memory=False)
    raw.columns = RAW_NAMES
    return raw


def decode_laps(raw: bytes) -> list[float]:
    laps: list[float] = []
    for idx in range(RA_LAP_COUNT):
        start = RA_LAP_OFFSET + idx * RA_LAP_WIDTH
        value = raw[start : start + RA_LAP_WIDTH].decode("ascii", errors="ignore")
        if value.isdigit() and int(value) > 0:
            laps.append(int(value) / 10.0)
    return laps


def load_laps(race_ids: set[str]) -> dict[str, list[float]]:
    laps_by_id: dict[str, list[float]] = {}
    years = sorted({race_id[:4] for race_id in race_ids if race_id[:4].isdigit()})
    for year in years:
        year_dir = TARGET_SE_DATA / year
        if not year_dir.exists():
            continue
        for path in year_dir.glob("SR*.DAT"):
            try:
                records = path.read_bytes().splitlines()
            except OSError:
                continue
            for raw in records:
                if not raw.startswith(b"RA"):
                    continue
                race_id = raw[RA_RACE_ID_OFFSET : RA_RACE_ID_OFFSET + RA_RACE_ID_LEN].decode(
                    "ascii", errors="ignore"
                )
                if race_id in race_ids:
                    laps_by_id[race_id] = decode_laps(raw)
    return laps_by_id


def pace_label(first3f: float, last3f: float) -> str:
    if pd.isna(first3f) or pd.isna(last3f):
        return ""
    diff = round(first3f - last3f, 1)
    if abs(diff) <= 0.5:
        return "ミドル"
    if diff > 0.5:
        return "後傾"
    return "前傾"


def position_text(row: pd.Series) -> str:
    values: list[str] = []
    for col in ["c2", "c3", "c4"]:
        value = row.get(col)
        if pd.notna(value):
            try:
                values.append(str(int(float(value))))
            except (TypeError, ValueError):
                values.append(str(value))
    if values:
        return "-".join(values)
    value = row.get("c4b")
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def build_profiles(entries: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    horse_ids = set(entries["血統登録番号"].dropna().astype(str))
    rows = raw[raw["horse_id"].astype(str).isin(horse_ids)].copy()
    rows = rows.drop_duplicates(["horse_id", "race_id"]).sort_values(["horse_id", "date_full"])
    race_ids = set(rows["race_id"].dropna().astype(str))
    laps_by_id = load_laps(race_ids)

    out: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        race_id = str(row["race_id"])
        laps = laps_by_id.get(race_id, [])
        first3f = round(sum(laps[:3]), 1) if len(laps) >= 3 else math.nan
        last3f = round(sum(laps[-3:]), 1) if len(laps) >= 3 else math.nan
        last5f = round(sum(laps[-5:]), 1) if len(laps) >= 5 else math.nan
        diff = round(first3f - last3f, 1) if not math.isnan(first3f) else math.nan
        out.append(
            {
                "horse_id": str(row["horse_id"]),
                "horse_name": row["horse_name"],
                "date": row["date_full"],
                "venue": row["venue"],
                "race_name": row["race_name"],
                "class": row["class_name"],
                "surface": row["surface"],
                "distance": row["distance"],
                "going": row["going"],
                "finish": row["finish"],
                "popularity": row["pop"],
                "position": position_text(row),
                "final3f": row["final3f"],
                "final3f_rank": row["final3f_rank"],
                "pci": row["pci"],
                "pci3": row["pci3"],
                "rpci": row["rpci"],
                "first3f": first3f,
                "last3f": last3f,
                "front_minus_last": diff,
                "last5f": last5f,
                "pace_type": pace_label(first3f, last3f),
                "laps": "-".join(f"{lap:.1f}" for lap in laps),
                "race_id": race_id,
            }
        )
    profiles = pd.DataFrame(out)
    numeric_cols = [
        "distance",
        "finish",
        "popularity",
        "final3f",
        "final3f_rank",
        "pci",
        "pci3",
        "rpci",
        "first3f",
        "last3f",
        "front_minus_last",
        "last5f",
    ]
    for col in numeric_cols:
        profiles[col] = pd.to_numeric(profiles[col], errors="coerce")
    profiles["year"] = pd.to_numeric(profiles["race_id"].str[:4], errors="coerce")
    profiles["win"] = profiles["finish"].eq(1)
    profiles["top3"] = profiles["finish"].le(3)
    return profiles


def fmt_record(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "該当なし"
    starts = len(frame)
    wins = int(frame["win"].sum())
    top3 = int(frame["top3"].sum())
    avg = frame["finish"].mean()
    return f"{starts}戦{wins}勝・馬券内{top3}回 / 平均{avg:.1f}着"


def fmt_num(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.{digits}f}"


def fit_score(row: pd.Series) -> float:
    score = 0.0
    if pd.notna(row["first3f"]):
        score += max(0.0, 20.0 - abs(row["first3f"] - 34.5) * 10.0)
    if pd.notna(row["front_minus_last"]):
        score += max(0.0, 20.0 - abs(row["front_minus_last"] - 0.5) * 10.0)
    if pd.notna(row["last5f"]):
        score += max(0.0, 20.0 - abs(row["last5f"] - 57.5) * 10.0)
    if pd.notna(row["rpci"]):
        score += max(0.0, 20.0 - abs(row["rpci"] - 52.0) * 4.0)
    finish = row.get("finish")
    if pd.notna(finish):
        score += max(0.0, 15.0 - max(0.0, float(finish) - 1.0) * 2.0)
    if row.get("top3"):
        score += 10.0
    return score


def best_fit_rows(frame: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    scored = frame.dropna(subset=["first3f", "last3f", "last5f"]).copy()
    if scored.empty:
        return scored
    scored["fit_score"] = scored.apply(fit_score, axis=1)
    return scored.sort_values(["fit_score", "year"], ascending=[False, False]).head(n)


def summarize_horse(entry: pd.Series, frame: pd.DataFrame) -> dict[str, object]:
    turf = frame[frame["surface"].astype(str).eq("芝")].copy()
    recent = turf[turf["year"].ge(2023)].copy()
    if recent.empty:
        recent = turf.copy()

    fit = best_fit_rows(recent)
    target_like = recent[
        recent["first3f"].between(34.3, 34.7)
        & recent["last5f"].between(57.0, 57.8)
        & recent["front_minus_last"].between(-0.2, 0.9)
    ]
    tokyo1600 = recent[
        recent["venue"].astype(str).eq("東京") & recent["distance"].eq(1600)
    ]
    pace_parts = []
    for label in ["前傾", "ミドル", "後傾"]:
        pace_parts.append(f"{label}: {fmt_record(recent[recent['pace_type'].eq(label)])}")

    best_text_parts = []
    for _, row in fit.iterrows():
        best_text_parts.append(
            f"{row['date']} {row['venue']} {row['race_name']} {fmt_num(row['distance'], 0)}m "
            f"{fmt_num(row['finish'], 0)}着 "
            f"前{fmt_num(row['first3f'])}/後{fmt_num(row['last3f'])}/5F{fmt_num(row['last5f'])} "
            f"RPCI{fmt_num(row['rpci'])} 位置{row['position']}"
        )

    recent_top3 = recent[recent["top3"]].copy()
    recent_bad_target = target_like[target_like["finish"].gt(5)]

    if not target_like.empty:
        target_record = fmt_record(target_like)
    else:
        target_record = "近いラップ経験が薄い"

    if not tokyo1600.empty:
        tokyo_record = fmt_record(tokyo1600)
    else:
        tokyo_record = "近年該当なし"

    score = 0.0
    if not fit.empty:
        score += min(100.0, float(fit["fit_score"].max()))
    if not target_like.empty:
        score += min(20.0, 5.0 * int(target_like["top3"].sum()))
    if not recent_bad_target.empty:
        score -= min(15.0, 3.0 * len(recent_bad_target))
    if not tokyo1600.empty:
        score += 8.0 * tokyo1600["top3"].mean()
        score -= 4.0 * tokyo1600["finish"].gt(8).mean()
    score = round(score, 1)

    return {
        "枠番": entry["枠番"],
        "馬番": entry["馬番"],
        "馬名": entry["馬名"],
        "血統登録番号": entry["血統登録番号"],
        "近似ラップ成績": target_record,
        "東京1600近況": tokyo_record,
        "前傾ミドル後傾": " / ".join(pace_parts),
        "近い好走・参考レース": " | ".join(best_text_parts),
        "fit_score": score,
        "recent_starts": len(recent),
    }


def write_report(summary: pd.DataFrame, profiles: pd.DataFrame) -> None:
    out_dir = ROOT / "outputs/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "yasuda_2026_all_runners_lap_summary.csv"
    profiles_csv = out_dir / "yasuda_2026_all_runners_lap_profiles.csv"
    report_path = out_dir / "yasuda_2026_all_runners_lap_summary.md"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    profiles.to_csv(profiles_csv, index=False, encoding="utf-8-sig")

    ranked = summary.sort_values("fit_score", ascending=False)
    lines = [
        "# 2026 安田記念 出走馬ラップ適性まとめ",
        "",
        "想定: 前半3F 34.3-34.7、後半5F 57.0-57.8、前後差はミドルからやや後傾、RPCI 50-53台。",
        "",
        "## 適性スコア順",
        "",
        "|順位|馬番|馬名|スコア|近似ラップ成績|東京1600近況|",
        "|---:|---:|---|---:|---|---|",
    ]
    for idx, (_, row) in enumerate(ranked.iterrows(), start=1):
        lines.append(
            f"|{idx}|{row['馬番']}|{row['馬名']}|{row['fit_score']:.1f}|"
            f"{row['近似ラップ成績']}|{row['東京1600近況']}|"
        )

    lines.extend(["", "## 各馬メモ", ""])
    for _, row in summary.sort_values(pd.to_numeric(summary["馬番"], errors="coerce").index.name or "馬番").iterrows():
        lines.extend(
            [
                f"### {row['馬番']} {row['馬名']}",
                "",
                f"- 近似ラップ成績: {row['近似ラップ成績']}",
                f"- 東京1600近況: {row['東京1600近況']}",
                f"- 前傾/ミドル/後傾: {row['前傾ミドル後傾']}",
                f"- 参考レース: {row['近い好走・参考レース'] or '該当なし'}",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"summary_csv={summary_csv}")
    print(f"profiles_csv={profiles_csv}")
    print(f"report={report_path}")


def main() -> None:
    entries = load_entries()
    raw = load_raw_results()
    profiles = build_profiles(entries, raw)

    rows = []
    for _, entry in entries.iterrows():
        horse_frame = profiles[profiles["horse_id"].astype(str).eq(str(entry["血統登録番号"]))]
        rows.append(summarize_horse(entry, horse_frame))
    summary = pd.DataFrame(rows)
    write_report(summary, profiles)
    print(summary.sort_values("fit_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
