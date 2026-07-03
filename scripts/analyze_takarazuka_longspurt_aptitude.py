from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.longspurt import LongspurtConfig, build_race_longspurt_features

TARGET_SE_DATA = Path("C:/Users/kazup/Data Lab/SE_DATA")

RA_RACE_ID_OFFSET = 11
RA_RACE_ID_LEN = 16
RA_LAP_OFFSET = 890
RA_LAP_COUNT = 25
RA_LAP_WIDTH = 3

VENUE_BY_CODE = {
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


def parse_int(raw: bytes) -> int | None:
    text = raw.decode("ascii", errors="ignore").strip()
    if not text or not text.lstrip("+-").isdigit():
        return None
    return int(text)


def decode_name(raw: bytes) -> str:
    return raw.decode("cp932", errors="replace").strip().replace("\u3000", "")


def decode_laps(raw: bytes) -> list[float]:
    laps: list[float] = []
    for idx in range(RA_LAP_COUNT):
        start = RA_LAP_OFFSET + idx * RA_LAP_WIDTH
        value = raw[start : start + RA_LAP_WIDTH].decode("ascii", errors="ignore")
        if value.isdigit() and int(value) > 0:
            laps.append(int(value) / 10)
    return laps


def position_text(raw: bytes) -> str:
    positions = []
    for start in (351, 353, 355, 357):
        value = parse_int(raw[start : start + 2])
        if value:
            positions.append(str(value))
    return "-".join(positions)


def parse_su_record(raw: bytes) -> dict[str, object] | None:
    if not raw.startswith(b"SE") or len(raw) < 553:
        return None
    race_id = raw[11:27].decode("ascii", errors="ignore")
    if len(race_id) != 16 or not race_id[:8].isdigit():
        return None
    horse_id = raw[30:40].decode("ascii", errors="ignore")
    finish = parse_int(raw[331:334])
    final3f = parse_int(raw[390:393])
    return {
        "race_id": race_id,
        "year": int(race_id[:4]),
        "date": race_id[:8],
        "horse_id": horse_id,
        "horse_name": decode_name(raw[40:76]),
        "finish": finish,
        "popularity": parse_int(raw[363:365]),
        "position": position_text(raw),
        "final3f": final3f / 10 if final3f is not None else math.nan,
    }


def load_takarazuka_race_ids() -> list[str]:
    table = pd.read_csv(
        ROOT / "outputs/analysis/takarazuka_kinen_hanshin_10_lap_report_table.csv",
        encoding="utf-8-sig",
        dtype={"日付": str},
    )
    table = table[pd.to_numeric(table["年"], errors="coerce").notna()].copy()
    dates = set(table["日付"].astype(str))
    laps = pd.read_csv(
        ROOT / "outputs/analysis/takarazuka_kinen_hanshin_10_race_laps.csv",
        encoding="utf-8-sig",
        dtype={"日付": str, "レースID": str},
    )
    return laps[laps["日付"].astype(str).isin(dates)]["レースID"].tolist()


def load_su_records_for_races(race_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    years = sorted({int(race_id[:4]) for race_id in race_ids})
    race_id_bytes = {race_id.encode("ascii") for race_id in race_ids}
    for year in years:
        year_dir = TARGET_SE_DATA / str(year)
        for path in sorted(year_dir.glob("SU*.DAT")):
            for raw in path.read_bytes().splitlines():
                if raw[11:27] not in race_id_bytes:
                    continue
                row = parse_su_record(raw)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def load_su_records_for_horses(horse_ids: set[str], start_year: int, end_year: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    horse_id_bytes = {horse_id.encode("ascii") for horse_id in horse_ids}
    for year in range(start_year, end_year + 1):
        year_dir = TARGET_SE_DATA / str(year)
        if not year_dir.exists():
            continue
        for path in sorted(year_dir.glob("SU*.DAT")):
            for raw in path.read_bytes().splitlines():
                if raw[30:40] not in horse_id_bytes:
                    continue
                row = parse_su_record(raw)
                if row:
                    rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["race_id", "horse_id"])


def load_ra_map(race_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    years = sorted({int(race_id[:4]) for race_id in race_ids if race_id[:4].isdigit()})
    race_id_bytes = {race_id.encode("ascii") for race_id in race_ids}
    for year in years:
        year_dir = TARGET_SE_DATA / str(year)
        if not year_dir.exists():
            continue
        for path in sorted(year_dir.glob("SR*.DAT")):
            for raw in path.read_bytes().splitlines():
                if not raw.startswith(b"RA"):
                    continue
                if raw[RA_RACE_ID_OFFSET : RA_RACE_ID_OFFSET + RA_RACE_ID_LEN] not in race_id_bytes:
                    continue
                race_id = raw[RA_RACE_ID_OFFSET : RA_RACE_ID_OFFSET + RA_RACE_ID_LEN].decode("ascii")
                laps = decode_laps(raw)
                distance = parse_int(raw[697:701])
                surface_code = raw[705:706].decode("ascii", errors="ignore")
                surface = {"1": "芝", "2": "ダ"}.get(surface_code, "")
                rows.append(
                    {
                        "race_id": race_id,
                        "race_name": decode_name(raw[32:92]),
                        "surface": surface,
                        "distance": distance,
                        "laps": "-".join(f"{lap:.1f}" for lap in laps),
                        "lap_count": len(laps),
                        "last5f": round(sum(laps[-5:]), 1) if len(laps) >= 5 else math.nan,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("race_id")


def load_ra_records(start_year: int, end_year: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        year_dir = TARGET_SE_DATA / str(year)
        if not year_dir.exists():
            continue
        for path in sorted(year_dir.glob("SR*.DAT")):
            for raw in path.read_bytes().splitlines():
                if not raw.startswith(b"RA"):
                    continue
                race_id = raw[RA_RACE_ID_OFFSET : RA_RACE_ID_OFFSET + RA_RACE_ID_LEN].decode(
                    "ascii", errors="ignore"
                )
                if len(race_id) != RA_RACE_ID_LEN or not race_id[:8].isdigit():
                    continue
                laps = decode_laps(raw)
                if len(laps) < 5:
                    continue
                surface_code = raw[705:706].decode("ascii", errors="ignore")
                surface = {"1": "芝", "2": "ダ"}.get(surface_code, "")
                distance = parse_int(raw[697:701])
                rows.append(
                    {
                        "race_id": race_id,
                        "date": race_id[:8],
                        "venue": VENUE_BY_CODE.get(race_id[8:10], race_id[8:10]),
                        "surface": surface,
                        "distance": distance,
                        "going": "",
                        "race_name": decode_name(raw[32:92]),
                        "race_laps": "-".join(f"{lap:.1f}" for lap in laps),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("race_id")


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def record_text(starts: int, good: int, rate: float) -> str:
    return f"{starts}戦{good}好走（{pct(rate)}）"


def main() -> None:
    out_dir = ROOT / "outputs/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    takarazuka_ids = set(load_takarazuka_race_ids())
    takarazuka = load_su_records_for_races(takarazuka_ids)
    takarazuka = takarazuka[pd.to_numeric(takarazuka["finish"], errors="coerce").between(1, 99)].copy()
    horse_ids = set(takarazuka["horse_id"].dropna().astype(str))
    start_year = max(2000, int(takarazuka["year"].min()) - 10)
    end_year = int(takarazuka["year"].max())

    histories = load_su_records_for_horses(horse_ids, start_year, end_year)
    ra_all = load_ra_records(start_year, end_year)
    race_features = build_race_longspurt_features(
        ra_all,
        LongspurtConfig(min_group_size=30, fast_quantile=0.25),
    )
    race_cols = [
        "race_id",
        "surface",
        "distance",
        "last5f_sum",
        "last5f_z",
        "last5f_fast_percentile",
        "last5f_top25",
        "fastest_lap_index_last5",
        "last2_bias",
        "final1_deceleration",
        "race_longspurt_type",
        "is_longspurt_race",
    ]
    histories = histories.merge(race_features[race_cols], on="race_id", how="left")
    histories["last5f"] = histories["last5f_sum"]
    histories["有効出走"] = pd.to_numeric(histories["finish"], errors="coerce").between(1, 99)
    histories["ロンスパ判定対象"] = (
        histories["有効出走"]
        &
        histories["surface"].astype(str).eq("芝")
        & pd.to_numeric(histories["distance"], errors="coerce").ge(1800)
        & histories["race_longspurt_type"].notna()
    )
    histories["ロンスパ戦"] = histories["ロンスパ判定対象"] & histories["is_longspurt_race"].fillna(False)
    histories["好走"] = pd.to_numeric(histories["finish"], errors="coerce").between(1, 3)

    judged = histories[histories["ロンスパ判定対象"]].copy()
    rows: list[dict[str, object]] = []
    for _, entry in takarazuka.sort_values(["year", "race_id", "finish"]).iterrows():
        horse_id = str(entry["horse_id"])
        target_date = str(entry["date"])
        group = judged[judged["horse_id"].astype(str).eq(horse_id) & judged["date"].astype(str).lt(target_date)].copy()
        long = group[group["ロンスパ戦"]]
        non = group[~group["ロンスパ戦"]]
        long_starts = len(long)
        non_starts = len(non)
        long_good = int(long["好走"].sum())
        non_good = int(non["好走"].sum())
        long_rate = long_good / long_starts if long_starts else math.nan
        non_rate = non_good / non_starts if non_starts else math.nan
        longspurt_good = (
            long_starts > 0
            and non_starts > 0
            and pd.notna(long_rate)
            and pd.notna(non_rate)
            and long_rate > non_rate
        )
        rows.append(
            {
                "race_id": entry["race_id"],
                "year": entry["year"],
                "date": entry["date"],
                "horse_id": horse_id,
                "馬名": entry["horse_name"],
                "宝塚記念着順": entry["finish"],
                "宝塚記念人気": entry["popularity"],
                "宝塚記念位置取り": entry["position"],
                "判定対象戦数": len(group),
                "ロンスパ戦数": long_starts,
                "ロンスパ好走数": long_good,
                "ロンスパ好走率": round(long_rate, 4) if pd.notna(long_rate) else pd.NA,
                "非ロンスパ戦数": non_starts,
                "非ロンスパ好走数": non_good,
                "非ロンスパ好走率": round(non_rate, 4) if pd.notna(non_rate) else pd.NA,
                "好走率差": round(long_rate - non_rate, 4) if pd.notna(long_rate) and pd.notna(non_rate) else pd.NA,
                "ロンスパ得意": "Y" if longspurt_good else "",
            }
        )
    aptitude = pd.DataFrame(rows).sort_values(["ロンスパ得意", "好走率差", "ロンスパ戦数"], ascending=[False, False, False])

    takarazuka = takarazuka.merge(
        race_features[
            [
                "race_id",
                "last5f_sum",
                "last5f_z",
                "last5f_fast_percentile",
                "fastest_lap_index_last5",
                "race_longspurt_type",
                "is_longspurt_race",
            ]
        ],
        on="race_id",
        how="left",
    )
    takarazuka["last5f"] = takarazuka["last5f_sum"]
    takarazuka["宝塚記念ロンスパ戦"] = takarazuka["is_longspurt_race"].fillna(False)
    takarazuka = takarazuka.merge(
        aptitude[
            [
                "race_id",
                "horse_id",
                "判定対象戦数",
                "ロンスパ戦数",
                "ロンスパ好走数",
                "ロンスパ好走率",
                "非ロンスパ戦数",
                "非ロンスパ好走数",
                "非ロンスパ好走率",
                "好走率差",
                "ロンスパ得意",
            ]
        ],
        on=["race_id", "horse_id"],
        how="left",
    )
    takarazuka_favorites = takarazuka[takarazuka["ロンスパ得意"].eq("Y")].copy()

    histories_out = histories.sort_values(["horse_id", "date", "race_id"])
    histories_out.to_csv(out_dir / "takarazuka_kinen_hanshin_10_running_lines_longspurt_flags.csv", index=False, encoding="utf-8-sig")
    aptitude.to_csv(out_dir / "takarazuka_kinen_hanshin_10_longspurt_aptitude_summary.csv", index=False, encoding="utf-8-sig")
    takarazuka_favorites.to_csv(out_dir / "takarazuka_kinen_hanshin_10_longspurt_good_takarazuka_results.csv", index=False, encoding="utf-8-sig")

    ranked = aptitude[aptitude["ロンスパ得意"].eq("Y")].copy()
    lines = [
        "# 宝塚記念 阪神開催10回 ロンスパ適性分析",
        "",
        "定義: TARGET RAレコードで後半5Fが取得できる芝1800m以上のレースを判定対象とした。ロンスパ戦は `src.features.longspurt` のレース分類ロジックを使用し、同距離・同競馬場・同芝ダートで後半5F上位25%かつ最速ラップ位置が後半5F前半から中盤、ラスト2F偏重でないレース。好走は3着以内。",
        "",
        "ロンスパ得意フラグ: 各宝塚記念の出走時点より前の馬柱だけを対象に、ロンスパ戦と非ロンスパ戦の両方に出走歴があり、ロンスパ戦の好走率が非ロンスパ戦の好走率を上回る馬。",
        "",
        "## ロンスパ得意馬",
        "",
        "|年|馬名|宝塚着順|人気|判定対象|ロンスパ戦|非ロンスパ戦|差|",
        "|---:|---|---:|---:|---:|---|---|---:|",
    ]
    for _, row in ranked.iterrows():
        lines.append(
            f"|{int(row['year'])}|{row['馬名']}|{int(row['宝塚記念着順']) if pd.notna(row['宝塚記念着順']) else ''}|"
            f"{int(row['宝塚記念人気']) if pd.notna(row['宝塚記念人気']) else ''}|{int(row['判定対象戦数'])}|"
            f"{record_text(int(row['ロンスパ戦数']), int(row['ロンスパ好走数']), float(row['ロンスパ好走率']))}|"
            f"{record_text(int(row['非ロンスパ戦数']), int(row['非ロンスパ好走数']), float(row['非ロンスパ好走率']))}|"
            f"{float(row['好走率差']) * 100:.1f}pt|"
        )

    lines.extend(["", "## ロンスパ得意馬の宝塚記念成績", ""])
    if takarazuka_favorites.empty:
        lines.append("該当なし")
    else:
        starts = len(takarazuka_favorites)
        unique_horses = takarazuka_favorites["horse_id"].nunique()
        wins = int(pd.to_numeric(takarazuka_favorites["finish"], errors="coerce").eq(1).sum())
        top3 = int(pd.to_numeric(takarazuka_favorites["finish"], errors="coerce").between(1, 3).sum())
        avg_finish = pd.to_numeric(takarazuka_favorites["finish"], errors="coerce").mean()
        lines.extend(
            [
                f"- 対象: のべ{starts}頭（ユニーク{unique_horses}頭）",
                f"- 成績: {starts}戦{wins}勝・馬券内{top3}回 / 平均{avg_finish:.1f}着",
                f"- 馬券内率: {top3 / starts * 100:.1f}%",
                "",
            ]
        )
        lines.extend(
            [
                "|年|馬名|着順|人気|位置取り|宝塚後半5F|宝塚分類|宝塚ロンスパ戦|ロンスパ戦成績|非ロンスパ戦成績|",
                "|---:|---|---:|---:|---|---:|---|---|---|---|",
            ]
        )
        for _, row in takarazuka_favorites.sort_values(["year", "finish", "horse_name"]).iterrows():
            lines.append(
                f"|{int(row['year'])}|{row['horse_name']}|{int(row['finish']) if pd.notna(row['finish']) else ''}|"
                f"{int(row['popularity']) if pd.notna(row['popularity']) else ''}|{row['position']}|"
                f"{float(row['last5f']):.1f}|{row['race_longspurt_type']}|"
                f"{'Y' if row['宝塚記念ロンスパ戦'] else ''}|"
                f"{record_text(int(row['ロンスパ戦数']), int(row['ロンスパ好走数']), float(row['ロンスパ好走率']))}|"
                f"{record_text(int(row['非ロンスパ戦数']), int(row['非ロンスパ好走数']), float(row['非ロンスパ好走率']))}|"
            )

    lines.extend(
        [
            "",
            "## 出力ファイル",
            "",
            f"- 馬柱明細: `{(out_dir / 'takarazuka_kinen_hanshin_10_running_lines_longspurt_flags.csv').as_posix()}`",
            f"- 馬別集計: `{(out_dir / 'takarazuka_kinen_hanshin_10_longspurt_aptitude_summary.csv').as_posix()}`",
            f"- ロンスパ得意馬の宝塚記念成績: `{(out_dir / 'takarazuka_kinen_hanshin_10_longspurt_good_takarazuka_results.csv').as_posix()}`",
        ]
    )
    report = out_dir / "takarazuka_kinen_hanshin_10_longspurt_aptitude_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"histories={len(histories_out)}")
    print(f"judged={len(judged)}")
    print(f"horses={len(aptitude)}")
    print(f"longspurt_good={len(ranked)}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
