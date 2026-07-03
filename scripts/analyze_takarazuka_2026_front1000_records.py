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
    supplemental_target_rows,
)
from scripts.analyze_takarazuka_character_fit import build_history_features, load_raw_results  # noqa: E402
from scripts.analyze_takarazuka_longspurt_aptitude import load_ra_records  # noqa: E402
from src.features.longspurt import parse_laps  # noqa: E402


def split_lengths(distance: object, n_laps: int) -> list[int]:
    distance_int = int(pd.to_numeric(pd.Series([distance]), errors="coerce").iloc[0])
    remainder = distance_int % 200
    if remainder and n_laps > 1:
        return [remainder] + [200] * (n_laps - 1)
    return [200] * n_laps


def first_n_time(laps_text: object, distance: object, meters: int = 1000) -> float:
    laps = parse_laps(str(laps_text))
    if not laps:
        return math.nan
    lengths = split_lengths(distance, len(laps))
    need = meters
    total = 0.0
    for lap, length in zip(laps, lengths):
        if need <= 0:
            break
        use = min(need, length)
        total += lap * (use / length)
        need -= use
    return round(total, 1) if need <= 0 else math.nan


def add_front1000(history: pd.DataFrame) -> pd.DataFrame:
    races = load_ra_records(2016, 2026)
    rows: list[dict[str, object]] = []
    for _, row in races.iterrows():
        rows.append(
            {
                "race_id": str(row["race_id"]),
                "front1000": first_n_time(row["race_laps"], row["distance"]),
                "race_lap_shape": row["race_laps"],
            }
        )
    front = pd.DataFrame(rows).drop_duplicates("race_id")
    out = history.copy()
    out["race_id"] = out["race_id"].astype(str)
    out = out.merge(front, on="race_id", how="left")
    return out


def bucket(value: object) -> str:
    front1000 = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(front1000):
        return "不明"
    if front1000 <= 59.0:
        return "59.0秒以下"
    if front1000 <= 60.0:
        return "59.1-60.0秒"
    return "60.1秒以上"


def record_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "0-0-0-0"
    finish = pd.to_numeric(frame["finish"], errors="coerce")
    return (
        f"{int(finish.eq(1).sum())}-"
        f"{int(finish.eq(2).sum())}-"
        f"{int(finish.eq(3).sum())}-"
        f"{int((finish.ge(4) & finish.le(99)).sum())}"
    )


def avg_finish(frame: pd.DataFrame) -> object:
    if frame.empty:
        return pd.NA
    return round(pd.to_numeric(frame["finish"], errors="coerce").mean(), 2)


def examples(frame: pd.DataFrame, n: int = 2) -> str:
    if frame.empty:
        return ""
    rows = frame.copy()
    rows["finish_num"] = pd.to_numeric(rows["finish"], errors="coerce")
    rows = rows.sort_values(["finish_num", "date"]).head(n)
    parts: list[str] = []
    for _, row in rows.iterrows():
        pci = "" if pd.isna(row.get("pci")) else f"PCI{float(row['pci']):.1f}"
        rpci = "" if pd.isna(row.get("rpci")) else f"RPCI{float(row['rpci']):.1f}"
        parts.append(
            f"{row['date']} {row['race_name']} {int(row['distance'])}m "
            f"{int(row['finish'])}着 {float(row['front1000']):.1f}s {pci} {rpci}".strip()
        )
    return " / ".join(parts)


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
            rows.append({"馬名": horse_name})
            continue
        horse_id = str(match.iloc[0]["horse_id"])
        group = history[
            history["horse_id"].astype(str).eq(horse_id)
            & history["date"].astype(str).lt(TARGET_DATE)
            & history["valid_finish"]
            & history["turf_mid"]
        ].copy()
        group["前半1000区分"] = group["front1000"].apply(bucket)
        fast = group[group["前半1000区分"].eq("59.0秒以下")]
        mid = group[group["前半1000区分"].eq("59.1-60.0秒")]
        slow = group[group["前半1000区分"].eq("60.1秒以上")]
        rows.append(
            {
                "馬名": horse_name,
                "判定対象": len(group),
                "59.0秒以下": record_text(fast),
                "59.0秒以下平均着順": avg_finish(fast),
                "59.1-60.0秒": record_text(mid),
                "59.1-60.0秒平均着順": avg_finish(mid),
                "60.1秒以上": record_text(slow),
                "60.1秒以上平均着順": avg_finish(slow),
                "59.0秒以下好走例": examples(fast[fast["finish"].between(1, 3)]),
                "59.1-60.0秒好走例": examples(mid[mid["finish"].between(1, 3)]),
                "60.1秒以上好走例": examples(slow[slow["finish"].between(1, 3)]),
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
    history = add_front1000(add_2025_benchmark_flags(build_history_features(raw, 2016, 2026)))
    summary = build_summary(history, raw)

    detail_path = out_dir / "takarazuka_kinen_2026_front1000_records_summary.csv"
    history_path = out_dir / "takarazuka_kinen_2026_front1000_records_histories.csv"
    report_path = out_dir / "takarazuka_kinen_2026_front1000_records_report.md"

    summary.to_csv(detail_path, index=False, encoding="utf-8-sig")
    ids = set(raw[raw["horse_name"].isin(EXPECTED_RUNNERS)]["horse_id"].astype(str))
    history[
        history["horse_id"].astype(str).isin(ids)
        & history["valid_finish"]
        & history["turf_mid"]
        & history["date"].astype(str).lt(TARGET_DATE)
    ].sort_values(["horse_name", "date", "race_id"]).to_csv(
        history_path, index=False, encoding="utf-8-sig"
    )

    lines = [
        "# 宝塚記念2026 想定馬 前半1000m通過別成績",
        "",
        "対象: 芝1800m以上。前半1000mはレースラップの先頭から1000m分を集計し、2500mなどの端数距離は線形補間した。",
        "成績表記は `(1着-2着-3着-圏外)`。",
        "",
        "|馬名|対象|59.0秒以下|平均|59.1-60.0秒|平均|60.1秒以上|平均|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"|{row['馬名']}|{int(row['判定対象'])}|{row['59.0秒以下']}|{row['59.0秒以下平均着順']}|"
            f"{row['59.1-60.0秒']}|{row['59.1-60.0秒平均着順']}|"
            f"{row['60.1秒以上']}|{row['60.1秒以上平均着順']}|"
        )
    lines.extend(
        [
            "",
            "## 好走例",
            "",
            "|馬名|59.0秒以下|59.1-60.0秒|60.1秒以上|",
            "|---|---|---|---|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"|{row['馬名']}|{row['59.0秒以下好走例']}|"
            f"{row['59.1-60.0秒好走例']}|{row['60.1秒以上好走例']}|"
        )
    lines.extend(
        [
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
