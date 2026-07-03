from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_takarazuka_character_fit import load_raw_results, numeric  # noqa: E402
from scripts.analyze_takarazuka_longspurt_aptitude import (  # noqa: E402
    load_ra_records,
    load_su_records_for_horses,
    load_su_records_for_races,
    load_takarazuka_race_ids,
)
from src.features.longspurt import parse_laps  # noqa: E402


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    valid = frame[pd.to_numeric(frame["takarazuka_finish"], errors="coerce").between(1, 99)].copy()
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
    finish = numeric(valid["takarazuka_finish"])
    pop = numeric(valid["takarazuka_popularity"])
    return {
        "starts": len(valid),
        "wins": int(finish.eq(1).sum()),
        "top3": int(finish.between(1, 3).sum()),
        "top3_rate": float(finish.between(1, 3).mean()),
        "avg_finish": float(finish.mean()),
        "avg_pop": float(pop.mean()),
        "pop_gain": float((pop - finish).mean()),
    }


def result_line(label: str, frame: pd.DataFrame) -> str:
    s = summarize(frame)
    if s["starts"] == 0:
        return f"- {label}: 0戦"
    return (
        f"- {label}: {s['starts']}戦{s['wins']}勝・馬券内{s['top3']}回 / "
        f"馬券内率{pct(s['top3_rate'])} / 平均着順{s['avg_finish']:.1f} / "
        f"平均人気{s['avg_pop']:.1f} / 人気-着順{s['pop_gain']:+.1f}"
    )


def add_lap_context(races: pd.DataFrame) -> pd.DataFrame:
    out = races.copy()
    laps = out["race_laps"].apply(parse_laps)
    out["first3f"] = laps.apply(lambda xs: round(sum(xs[:3]), 1) if len(xs) >= 3 else math.nan)
    out["last3f"] = laps.apply(lambda xs: round(sum(xs[-3:]), 1) if len(xs) >= 3 else math.nan)
    out["last5f"] = laps.apply(lambda xs: round(sum(xs[-5:]), 1) if len(xs) >= 5 else math.nan)
    return out


def build_history(takarazuka: pd.DataFrame) -> pd.DataFrame:
    horse_ids = set(takarazuka["horse_id"].astype(str))
    history = load_su_records_for_horses(horse_ids, 2015, 2025)
    races = add_lap_context(load_ra_records(2015, 2025))
    history = history.merge(
        races[
            [
                "race_id",
                "race_name",
                "venue",
                "surface",
                "distance",
                "first3f",
                "last3f",
                "last5f",
                "race_laps",
            ]
        ],
        on="race_id",
        how="left",
    )

    raw = load_raw_results()
    raw_cols = [
        "race_id",
        "horse_id",
        "pci",
        "pci3",
        "rpci",
        "corner1",
        "corner2",
        "corner3",
        "corner4",
        "final3f_rank",
    ]
    history = history.merge(raw[raw_cols], on=["race_id", "horse_id"], how="left")
    return history.drop_duplicates(["race_id", "horse_id"])


def main() -> None:
    out_dir = ROOT / "outputs/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    takarazuka_ids = set(load_takarazuka_race_ids())
    takarazuka = load_su_records_for_races(takarazuka_ids)
    takarazuka = takarazuka[
        pd.to_numeric(takarazuka["finish"], errors="coerce").between(1, 99)
    ].copy()
    takarazuka["horse_id"] = takarazuka["horse_id"].astype(str)
    takarazuka["date"] = takarazuka["date"].astype(str)

    history = build_history(takarazuka)
    rows: list[dict[str, object]] = []
    for _, entry in takarazuka.sort_values(["year", "finish"]).iterrows():
        horse_id = str(entry["horse_id"])
        year = int(entry["year"])
        target_date = str(entry["date"])
        group = history[
            history["horse_id"].astype(str).eq(horse_id)
            & history["date"].astype(str).lt(target_date)
        ].sort_values(["date", "race_id"])
        if group.empty:
            continue
        prev = group.iloc[-1]
        same_year_nikkei = group[
            group["date"].astype(str).str.startswith(str(year))
            & group["race_name"].fillna("").astype(str).str.contains("日経賞", na=False)
        ].copy()
        if same_year_nikkei.empty and not str(prev.get("race_name", "")).find("日経賞") >= 0:
            continue
        nikkei = same_year_nikkei.iloc[-1] if not same_year_nikkei.empty else prev
        direct = "Y" if str(prev.get("race_name", "")).find("日経賞") >= 0 else ""
        rows.append(
            {
                "year": year,
                "horse_id": horse_id,
                "horse_name": entry["horse_name"],
                "takarazuka_date": entry["date"],
                "takarazuka_finish": entry["finish"],
                "takarazuka_popularity": entry["popularity"],
                "takarazuka_position": entry["position"],
                "direct_from_nikkei": direct,
                "prev_race": prev.get("race_name", ""),
                "prev_finish": prev.get("finish", pd.NA),
                "prev_popularity": prev.get("popularity", pd.NA),
                "nikkei_date": nikkei.get("date", ""),
                "nikkei_race_name": nikkei.get("race_name", ""),
                "nikkei_finish": nikkei.get("finish", pd.NA),
                "nikkei_popularity": nikkei.get("popularity", pd.NA),
                "nikkei_position": nikkei.get("position", ""),
                "nikkei_pci": nikkei.get("pci", pd.NA),
                "nikkei_rpci": nikkei.get("rpci", pd.NA),
                "nikkei_first3f": nikkei.get("first3f", pd.NA),
                "nikkei_last3f": nikkei.get("last3f", pd.NA),
                "nikkei_last5f": nikkei.get("last5f", pd.NA),
                "nikkei_final3f": nikkei.get("final3f", pd.NA),
                "nikkei_final3f_rank": nikkei.get("final3f_rank", pd.NA),
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise SystemExit("no nikkei rotation rows")
    detail["takarazuka_top3"] = pd.to_numeric(detail["takarazuka_finish"], errors="coerce").between(1, 3)
    detail["nikkei_top3"] = pd.to_numeric(detail["nikkei_finish"], errors="coerce").between(1, 3)
    detail["nikkei_finish_bucket"] = pd.cut(
        pd.to_numeric(detail["nikkei_finish"], errors="coerce"),
        bins=[0, 3, 5, 99],
        labels=["日経賞1-3着", "日経賞4-5着", "日経賞6着以下"],
    ).astype(str)
    detail["nikkei_pop_bucket"] = pd.cut(
        pd.to_numeric(detail["nikkei_popularity"], errors="coerce"),
        bins=[0, 3, 6, 99],
        labels=["日経賞1-3人気", "日経賞4-6人気", "日経賞7人気以下"],
    ).astype(str)

    detail_path = out_dir / "takarazuka_kinen_nikkei_sho_rotation_detail.csv"
    report_path = out_dir / "takarazuka_kinen_nikkei_sho_rotation_report.md"
    detail.sort_values(["year", "takarazuka_finish", "horse_name"]).to_csv(
        detail_path, index=False, encoding="utf-8-sig"
    )

    direct = detail[detail["direct_from_nikkei"].eq("Y")].copy()
    not_direct = detail[~detail["direct_from_nikkei"].eq("Y")].copy()
    nikkei_top3 = detail[detail["nikkei_top3"]].copy()
    nikkei_out = detail[~detail["nikkei_top3"]].copy()
    c4_front = detail[
        pd.to_numeric(detail["nikkei_position"].astype(str).str.split("-").str[-1], errors="coerce").le(5)
    ].copy()
    c4_back = detail[
        pd.to_numeric(detail["nikkei_position"].astype(str).str.split("-").str[-1], errors="coerce").gt(5)
    ].copy()

    lines = [
        "# 日経賞から宝塚記念ローテ分析",
        "",
        "対象: 宝塚記念の阪神開催過去10回（2015-2023, 2025。2024京都開催は除外）。",
        "定義: 「直行」は直前走が日経賞。「同年日経賞組」は同年の日経賞を使ってから宝塚へ出走した馬で、間に天皇賞春などを挟む馬も含む。",
        "",
        "## 成績サマリ",
        "",
        result_line("同年日経賞組", detail),
        result_line("日経賞から直行", direct),
        result_line("日経賞後に別レースを挟む", not_direct),
        result_line("日経賞1-3着", nikkei_top3),
        result_line("日経賞4着以下", nikkei_out),
        result_line("日経賞4角5番手以内", c4_front),
        result_line("日経賞4角6番手以下", c4_back),
        "",
        "## 該当馬一覧",
        "",
        "|年|馬名|宝塚着|宝塚人気|直行|前走|日経賞着|日経賞人気|日経賞位置|日経賞PCI|日経賞RPCI|",
        "|---:|---|---:|---:|---|---|---:|---:|---|---:|---:|",
    ]
    for _, row in detail.sort_values(["year", "takarazuka_finish", "horse_name"]).iterrows():
        pci = "" if pd.isna(row["nikkei_pci"]) else f"{float(row['nikkei_pci']):.1f}"
        rpci = "" if pd.isna(row["nikkei_rpci"]) else f"{float(row['nikkei_rpci']):.1f}"
        lines.append(
            f"|{int(row['year'])}|{row['horse_name']}|{int(row['takarazuka_finish'])}|"
            f"{int(row['takarazuka_popularity']) if pd.notna(row['takarazuka_popularity']) else ''}|"
            f"{row['direct_from_nikkei']}|{row['prev_race']}|"
            f"{int(row['nikkei_finish']) if pd.notna(row['nikkei_finish']) else ''}|"
            f"{int(row['nikkei_popularity']) if pd.notna(row['nikkei_popularity']) else ''}|"
            f"{row['nikkei_position']}|{pci}|{rpci}|"
        )

    lines.extend(
        [
            "",
            "## 読み取り",
            "",
            "- 日経賞組全体はサンプルが少なく、宝塚で強いプラスローテとは言いにくい。",
            "- 直行組はさらに少数。好走するには日経賞で好位から崩れず、宝塚でも前受けまたは早め進出できることが条件になりやすい。",
            "- 日経賞で負けていても巻き返し余地はあるが、日経賞4角で前にいた馬が宝塚で残す形はかなり限定的。前が厳しい想定なら過信しない。",
            "- 今年の日経賞組を見るなら、日経賞の着順だけでなく、4角位置、上がり順位、宝塚想定ラップとの一致度を優先して取捨する。",
            "",
            "## 出力ファイル",
            "",
            f"- 明細CSV: `{detail_path.as_posix()}`",
            f"- レポート: `{report_path.as_posix()}`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"rows={len(detail)}")
    print(f"direct={len(direct)}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
