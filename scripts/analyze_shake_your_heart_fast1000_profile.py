from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis"
TARGET_DATE = "20260614"


def record(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "0-0-0-0"
    finish = pd.to_numeric(frame["finish"], errors="coerce")
    return (
        f"{int(finish.eq(1).sum())}-"
        f"{int(finish.eq(2).sum())}-"
        f"{int(finish.eq(3).sum())}-"
        f"{int(finish.ge(4).sum())}"
    )


def rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float(pd.to_numeric(frame["finish"], errors="coerce").le(3).mean())


def mean(frame: pd.DataFrame, col: str) -> str:
    values = pd.to_numeric(frame.get(col), errors="coerce")
    if values.dropna().empty:
        return ""
    return f"{values.mean():.2f}"


def examples(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    parts = []
    for _, row in frame.sort_values(["finish", "date"]).iterrows():
        pci = "" if pd.isna(row.get("pci")) else f" PCI{float(row['pci']):.1f}"
        pci3 = "" if pd.isna(row.get("pci3")) else f" PCI3{float(row['pci3']):.1f}"
        rpci = "" if pd.isna(row.get("rpci")) else f" RPCI{float(row['rpci']):.1f}"
        parts.append(
            f"{row['date']} {row['race_name']} {int(row['distance'])}m "
            f"{int(row['finish'])}着 人気{int(row['popularity']) if pd.notna(row['popularity']) else ''} "
            f"4角{int(row['corner4']) if pd.notna(row['corner4']) else ''} "
            f"1000m{float(row['front1000']):.1f} 後5F{float(row['last5f_sum']):.1f} "
            f"上3F{float(row['last3f']):.1f}{pci}{pci3}{rpci} {row['race_longspurt_type']}"
        )
    return " / ".join(parts)


def group_summary(history: pd.DataFrame, label: str, groups: list[tuple[str, pd.Series]]) -> pd.DataFrame:
    rows = []
    for name, mask in groups:
        frame = history[mask.fillna(False)].copy()
        rows.append(
            {
                "観点": label,
                "区分": name,
                "該当数": len(frame),
                "成績": record(frame),
                "複勝率": round(rate(frame), 4),
                "平均着順": mean(frame, "finish"),
                "平均人気": mean(frame, "popularity"),
                "平均1000m": mean(frame, "front1000"),
                "平均後半5F": mean(frame, "last5f_sum"),
                "平均上がり3F": mean(frame, "last3f"),
                "平均PCI": mean(frame, "pci"),
                "平均PCI3": mean(frame, "pci3"),
                "平均RPCI": mean(frame, "rpci"),
                "該当レース": examples(frame),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one horse under front-1000m <= 59.0s races.")
    parser.add_argument("--horse-id", default="2020103101")
    parser.add_argument("--horse-name", default="シェイクユアハート")
    parser.add_argument("--output-prefix", default="shake_your_heart_fast1000")
    args = parser.parse_args()

    history = pd.read_csv(
        OUT / "takarazuka_kinen_2026_front1000_records_histories.csv",
        encoding="utf-8-sig",
        dtype={"horse_id": str},
    )
    h = history[
        history["horse_id"].eq(str(args.horse_id))
        & history["date"].astype(str).lt(TARGET_DATE)
        & history["valid_finish"].fillna(False).astype(bool)
        & history["turf_mid"].fillna(False).astype(bool)
    ].copy()
    h["front1000"] = pd.to_numeric(h["front1000"], errors="coerce")
    fast = h[h["front1000"].le(59.0)].copy().sort_values("date")

    fast["pci_band"] = pd.cut(
        pd.to_numeric(fast["pci"], errors="coerce"),
        [-999, 50, 52, 55, 999],
        labels=["PCI<50", "PCI50-52", "PCI52-55", "PCI55+"],
    )
    fast["rpci_band"] = pd.cut(
        pd.to_numeric(fast["rpci"], errors="coerce"),
        [-999, 50, 52, 55, 999],
        labels=["RPCI<50", "RPCI50-52", "RPCI52-55", "RPCI55+"],
    )
    fast["front_type"] = pd.cut(
        pd.to_numeric(fast["front_minus_last"], errors="coerce"),
        [-999, -0.6, 0.6, 999],
        labels=["前傾", "ミドル", "後傾"],
    )

    summary = pd.concat(
        [
            group_summary(
                fast,
                "レース質",
                [
                    (name, fast["race_longspurt_type"].eq(name))
                    for name in ["ロンスパ戦", "持続戦", "瞬発戦", "消耗戦", "標準戦"]
                ],
            ),
            group_summary(fast, "PCI", [(str(cat), fast["pci_band"].eq(cat)) for cat in fast["pci_band"].cat.categories]),
            group_summary(fast, "RPCI", [(str(cat), fast["rpci_band"].eq(cat)) for cat in fast["rpci_band"].cat.categories]),
            group_summary(
                fast,
                "前傾/後傾",
                [(str(cat), fast["front_type"].eq(cat)) for cat in fast["front_type"].cat.categories],
            ),
            group_summary(
                fast,
                "後半条件",
                [
                    ("後半5F<=60.0", fast["last5f_sum"].le(60.0)),
                    ("上がり3F>=35.0", fast["last3f"].ge(35.0)),
                ],
            ),
        ],
        ignore_index=True,
    )

    detail_path = OUT / f"{args.output_prefix}_detail.csv"
    summary_path = OUT / f"{args.output_prefix}_summary.csv"
    report_path = OUT / f"{args.output_prefix}_report.md"
    fast.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    lines = [
        f"# {args.horse_name} 1000m通過59秒以下限定分析",
        "",
        f"対象: 芝1800m以上、{TARGET_DATE}以前、1000m通過59.0秒以下。該当 {len(fast)}戦。",
        "",
        f"限定成績: {record(fast)} / 複勝率 {rate(fast) * 100:.1f}% / 平均着順 {mean(fast, 'finish')} / 平均人気 {mean(fast, 'popularity')}",
        "",
        "## 該当レース",
        "",
        "|日付|レース|距離|着|人気|4角|1000m|後5F|上3F|PCI|PCI3|RPCI|分類|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in fast.iterrows():
        lines.append(
            f"|{row['date']}|{row['race_name']}|{int(row['distance'])}|{int(row['finish'])}|"
            f"{'' if pd.isna(row['popularity']) else int(row['popularity'])}|"
            f"{'' if pd.isna(row['corner4']) else int(row['corner4'])}|"
            f"{float(row['front1000']):.1f}|{float(row['last5f_sum']):.1f}|{float(row['last3f']):.1f}|"
            f"{'' if pd.isna(row['pci']) else f'{float(row['pci']):.1f}'}|"
            f"{'' if pd.isna(row['pci3']) else f'{float(row['pci3']):.1f}'}|"
            f"{'' if pd.isna(row['rpci']) else f'{float(row['rpci']):.1f}'}|"
            f"{row['race_longspurt_type']}|"
        )
    lines.extend(
        [
            "",
            "## 集計",
            "",
            "|観点|区分|成績|複勝率|平均着順|平均PCI|平均PCI3|平均RPCI|平均後半5F|平均上がり3F|",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"|{row['観点']}|{row['区分']}|{row['成績']}|{row['複勝率']:.1%}|{row['平均着順']}|"
            f"{row['平均PCI']}|{row['平均PCI3']}|{row['平均RPCI']}|{row['平均後半5F']}|{row['平均上がり3F']}|"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path)
    print(detail_path)
    print(report_path)


if __name__ == "__main__":
    main()
