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
    return (
        f"{int((frame['finish'] == 1).sum())}-"
        f"{int((frame['finish'] == 2).sum())}-"
        f"{int((frame['finish'] == 3).sum())}-"
        f"{int((frame['finish'] > 3).sum())}"
    )


def rate(frame: pd.DataFrame) -> float:
    return float((frame["finish"] <= 3).mean()) if not frame.empty else 0.0


def mean(frame: pd.DataFrame, col: str) -> str:
    if frame.empty or col not in frame or pd.to_numeric(frame[col], errors="coerce").dropna().empty:
        return ""
    return f"{pd.to_numeric(frame[col], errors='coerce').mean():.2f}"


def examples(frame: pd.DataFrame, n: int = 3) -> str:
    if frame.empty:
        return ""
    good = frame[frame["finish"] <= 3].copy()
    if good.empty:
        good = frame.copy()
    good = good.sort_values(["finish", "date"]).head(n)
    parts = []
    for _, row in good.iterrows():
        pci = "" if pd.isna(row["pci"]) else f" PCI{float(row['pci']):.1f}"
        rpci = "" if pd.isna(row["rpci"]) else f" RPCI{float(row['rpci']):.1f}"
        parts.append(
            f"{row['date']} {row['race_name']} {int(row['distance'])}m "
            f"{int(row['finish'])}着{pci}{rpci}"
        )
    return " / ".join(parts)


def summarize_group(history: pd.DataFrame, label: str, groups: list[tuple[str, pd.Series]]) -> pd.DataFrame:
    rows = []
    for name, mask in groups:
        frame = history[mask.fillna(False) if hasattr(mask, "fillna") else mask].copy()
        rows.append(
            {
                "観点": label,
                "区分": name,
                "成績": record(frame),
                "複勝率": round(rate(frame), 4),
                "平均着順": mean(frame, "finish"),
                "平均人気": mean(frame, "popularity"),
                "平均PCI": mean(frame, "pci"),
                "平均PCI3": mean(frame, "pci3"),
                "平均RPCI": mean(frame, "rpci"),
                "平均後半5F": mean(frame, "last5f_sum"),
                "平均上がり3F": mean(frame, "last3f"),
                "好走例": examples(frame),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one horse's multifactor lap profile.")
    parser.add_argument("--horse-id", default="2020103101")
    parser.add_argument("--horse-name", default="シェイクユアハート")
    parser.add_argument("--output-prefix", default="shake_your_heart_multifactor")
    args = parser.parse_args()

    hist = pd.read_csv(
        OUT / "takarazuka_kinen_2026_expected_runners_benchmark_histories.csv",
        encoding="utf-8-sig",
        dtype={"horse_id": str},
    )
    h = hist[
        hist["horse_id"].eq(str(args.horse_id))
        & hist["date"].astype(str).lt(TARGET_DATE)
        & hist["valid_finish"].fillna(False).astype(bool)
        & hist["turf_mid"].fillna(False).astype(bool)
    ].copy()
    h = h.sort_values("date")

    h["pci_band"] = pd.cut(
        pd.to_numeric(h["pci"], errors="coerce"),
        [-999, 50, 52, 55, 999],
        labels=["PCI<50", "PCI50-52", "PCI52-55", "PCI55+"],
    )
    h["rpci_band"] = pd.cut(
        pd.to_numeric(h["rpci"], errors="coerce"),
        [-999, 50, 52, 55, 999],
        labels=["RPCI<50", "RPCI50-52", "RPCI52-55", "RPCI55+"],
    )
    h["front_type"] = pd.cut(
        pd.to_numeric(h["front_minus_last"], errors="coerce"),
        [-999, -0.6, 0.6, 999],
        labels=["前傾", "ミドル", "後傾"],
    )
    h["front1000_type"] = pd.cut(
        pd.to_numeric(h["first3f"], errors="coerce") + 24.0,
        [-999, 59.0, 60.0, 999],
        labels=["推定1000m高速", "推定1000m標準", "推定1000mスロー"],
    )

    tables = [
        summarize_group(
            h,
            "レース質",
            [(str(k), h["race_longspurt_type"].eq(k)) for k in ["ロンスパ戦", "持続戦", "瞬発戦", "消耗戦", "標準戦"]],
        ),
        summarize_group(h, "PCI", [(str(k), h["pci_band"].eq(k)) for k in h["pci_band"].cat.categories]),
        summarize_group(h, "RPCI", [(str(k), h["rpci_band"].eq(k)) for k in h["rpci_band"].cat.categories]),
        summarize_group(h, "前傾/後傾", [(str(k), h["front_type"].eq(k)) for k in h["front_type"].cat.categories]),
        summarize_group(h, "後半条件", [("後半5F<=60.0", h["last5f_sum"] <= 60), ("上がり3F>=35.0", h["last3f"] >= 35)]),
    ]
    summary = pd.concat(tables, ignore_index=True)
    summary_path = OUT / f"{args.output_prefix}_summary.csv"
    detail_path = OUT / f"{args.output_prefix}_detail.csv"
    report_path = OUT / f"{args.output_prefix}_report.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    detail_cols = [
        "date",
        "race_name",
        "class_name",
        "distance",
        "finish",
        "popularity",
        "corner4",
        "final3f",
        "final3f_rank",
        "pci",
        "pci3",
        "rpci",
        "first3f",
        "last5f_sum",
        "last3f",
        "front_minus_last",
        "race_longspurt_type",
        "late_peak7_type",
        "tough_sustain_race",
        "no_breather_race",
    ]
    h[detail_cols].to_csv(detail_path, index=False, encoding="utf-8-sig")

    lines = [
        f"# {args.horse_name} 多角ラップ分析",
        "",
        f"対象: 芝1800m以上、{TARGET_DATE}以前、TARGET履歴 {len(h)}戦。",
        "",
        f"総成績: {record(h)} / 複勝率 {rate(h) * 100:.1f}% / 平均着順 {mean(h, 'finish')}",
        "",
        "## 要約",
        "",
        "- 下表はレース質、PCI/RPCI帯、前傾/後傾、後半条件ごとの成績を同一母集団で再集計したもの。",
        "- 複勝率だけでなく、平均着順、平均人気、平均PCI/RPCI、後半5F、上がり3Fを合わせて見る。",
        "- 宝塚記念向けには、前傾/ミドル、上がり35秒以上、後半5Fが締まるレースでの安定度を重視する。",
        "",
        "## 集計",
        "",
        "|観点|区分|成績|複勝率|平均着順|平均PCI|平均PCI3|平均RPCI|平均後半5F|平均上がり3F|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
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
