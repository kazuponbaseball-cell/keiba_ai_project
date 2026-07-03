from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "analysis"
TARGET_DATE = "20260614"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / name, encoding="utf-8-sig")


def norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    lo = values.min()
    hi = values.max()
    if math.isclose(float(lo), float(hi)):
        return pd.Series(0.5, index=series.index)
    return ((values - lo) / (hi - lo)).fillna(0.0)


def grade_weight_label(value: float) -> str:
    if value >= 2.0:
        return "G1"
    if value >= 1.7:
        return "G2"
    if value >= 1.4:
        return "G3"
    return "OP/条件"


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).lower() in {"true", "1", "y", "yes"}


def record_win_rate(record: object) -> float:
    if pd.isna(record):
        return 0.0
    parts = str(record).split("-")
    if len(parts) < 4:
        return 0.0
    nums = []
    for part in parts[:4]:
        try:
            nums.append(int(part))
        except ValueError:
            nums.append(0)
    total = sum(nums)
    return (nums[0] + nums[1] + nums[2]) / total if total else 0.0


def previous_race_features(histories: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    histories = histories[
        histories["date"].astype(str).lt(TARGET_DATE)
        & histories["valid_finish"].fillna(False).astype(bool)
        & histories["turf_mid"].fillna(False).astype(bool)
    ].copy()

    for horse_id, group in histories.groupby("horse_id"):
        group = group.sort_values("date")
        prev = group.iloc[-1]
        recent3 = group.tail(3)
        prev_top3 = int(prev["finish"]) <= 3
        prev_grade_weight = float(prev.get("grade_weight", 1.0) or 1.0)
        prev_tough = (
            to_bool(prev.get("tough_sustain_race"))
            or to_bool(prev.get("no_breather_race"))
            or to_bool(prev.get("bench2025_like_race"))
            or (pd.notna(prev.get("last3f")) and float(prev["last3f"]) >= 35.0)
        )
        prev_burden = 0.0
        if prev_tough:
            prev_burden += 1.0
        if pd.notna(prev.get("last5f_sum")) and float(prev["last5f_sum"]) <= 60.0:
            prev_burden += 0.5
        if pd.notna(prev.get("rpci")) and float(prev["rpci"]) <= 52.0:
            prev_burden += 0.5
        if pd.notna(prev.get("corner4_rate")) and float(prev["corner4_rate"]) <= 0.35:
            prev_burden += 0.4
        if pd.notna(prev.get("final3f_rank")) and float(prev["final3f_rank"]) <= 3:
            prev_burden += 0.3
        prev_burden = min(prev_burden, 2.7)

        route_bonus = 0.0
        if prev_top3 and prev_tough:
            route_bonus += 1.2
        if prev_top3 and prev_grade_weight >= 1.7:
            route_bonus += 1.0
        if prev_top3 and prev_grade_weight >= 2.0:
            route_bonus += 0.5
        if str(prev.get("race_name", "")).find("日経賞") >= 0:
            route_bonus += 0.4 if prev_top3 else -0.3
        if int(prev["finish"]) >= 10 and prev_burden >= 1.5:
            route_bonus -= 0.5
        if int(prev["finish"]) >= 10 and prev_burden < 1.0:
            route_bonus -= 0.8

        recent_link = (
            recent3["tough_sustain_good"].fillna(False).astype(bool).sum() * 0.5
            + recent3["no_breather_good"].fillna(False).astype(bool).sum() * 0.4
            + recent3["bench2025_good"].fillna(False).astype(bool).sum() * 0.5
            + recent3["top3"].fillna(False).astype(bool).sum() * 0.2
        )

        prev_race_note = []
        if prev_top3:
            prev_race_note.append("前走馬券内")
        if prev_tough:
            prev_race_note.append("前走タフ質")
        if prev_grade_weight >= 1.7:
            prev_race_note.append(f"前走{grade_weight_label(prev_grade_weight)}")
        if str(prev.get("race_name", "")).find("日経賞") >= 0:
            prev_race_note.append("日経賞組")

        rows.append(
            {
                "horse_id": str(horse_id),
                "前走日": str(prev["date"]),
                "前走": prev["race_name"],
                "前走格": grade_weight_label(prev_grade_weight),
                "前走距離": prev["distance"],
                "前走着順": int(prev["finish"]),
                "前走人気": "" if pd.isna(prev.get("popularity")) else int(prev["popularity"]),
                "前走4角": "" if pd.isna(prev.get("corner4")) else int(prev["corner4"]),
                "前走PCI": "" if pd.isna(prev.get("pci")) else round(float(prev["pci"]), 1),
                "前走RPCI": "" if pd.isna(prev.get("rpci")) else round(float(prev["rpci"]), 1),
                "前走前半3F": "" if pd.isna(prev.get("first3f")) else round(float(prev["first3f"]), 1),
                "前走後半5F": "" if pd.isna(prev.get("last5f_sum")) else round(float(prev["last5f_sum"]), 1),
                "前走上がり3F": "" if pd.isna(prev.get("last3f")) else round(float(prev["last3f"]), 1),
                "前走負荷": round(prev_burden, 2),
                "前走ローテ加点": round(route_bonus, 2),
                "近3走連関加点": round(float(recent_link), 2),
                "前走メモ": "・".join(prev_race_note),
            }
        )
    return pd.DataFrame(rows)


def reason(row: pd.Series) -> str:
    parts: list[str] = []
    if row["ラップ総合"] >= 75:
        parts.append("ラップ適性上位")
    if row.get("前走ローテ加点", 0) >= 1.5:
        parts.append("前走内容が直結")
    elif row.get("前走負荷", 0) >= 1.5 and row.get("前走着順", 99) <= 3:
        parts.append("前走で負荷を受けて好走")
    if row.get("淡々リズムスコア", 0) >= 60:
        parts.append("息が入りにくい流れに強い")
    if row.get("上がり35秒以上スコア", 0) >= 70:
        parts.append("上がりを要す形に強い")
    if row.get("59秒以下複勝率", 0) >= 0.5:
        parts.append("速い1000m通過にも対応")
    if not parts:
        parts.append("条件適性は部分的")
    return " / ".join(parts[:4])


def final_label(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B+"
    if score >= 55:
        return "B"
    if score >= 40:
        return "C+"
    if score >= 30:
        return "C"
    return "D"


def main() -> None:
    base = read_csv("takarazuka_kinen_2026_expected_runners_benchmark_summary.csv")
    slow = read_csv("takarazuka_kinen_2026_slow_last3f_fit_summary.csv")
    combo = read_csv("takarazuka_kinen_2026_fast5f_slow3f_fit_summary.csv")
    rhythm = read_csv("takarazuka_kinen_2026_no_slack_rhythm_fit_summary.csv")
    front1000 = read_csv("takarazuka_kinen_2026_front1000_records_summary.csv")
    frontback = read_csv("takarazuka_kinen_2026_front_back_records_summary.csv")
    histories = read_csv("takarazuka_kinen_2026_expected_runners_benchmark_histories.csv")

    prev = previous_race_features(histories)

    df = base[["馬名", "horse_id", "判定対象戦数", "適性スコア", "評価", "根拠レース"]].copy()
    df["horse_id"] = df["horse_id"].astype(str)
    prev["horse_id"] = prev["horse_id"].astype(str)
    df = df.rename(columns={"適性スコア": "宝塚キャラスコア", "評価": "宝塚キャラ評価", "根拠レース": "主要根拠"})
    df = df.merge(
        slow[["馬名", "適性スコア", "上がり35秒以上馬券内率", "重賞35秒以上重み"]],
        on="馬名",
        how="left",
        suffixes=("", "_slow"),
    ).rename(columns={"適性スコア": "上がり35秒以上スコア"})
    df = df.merge(
        combo[["馬名", "適性スコア", "複合条件馬券内率", "重賞複合条件重み"]],
        on="馬名",
        how="left",
    ).rename(columns={"適性スコア": "高速5F×上がり掛かるスコア"})
    df = df.merge(
        rhythm[["馬名", "適性スコア", "淡々リズム馬券内率", "重賞淡々リズム重み"]],
        on="馬名",
        how="left",
    ).rename(columns={"適性スコア": "淡々リズムスコア_raw"})
    df = df.merge(
        front1000[["馬名", "59.0秒以下", "59.1-60.0秒", "60.1秒以上"]],
        on="馬名",
        how="left",
    )
    df = df.merge(
        frontback[["馬名", "前傾成績", "後傾成績"]],
        on="馬名",
        how="left",
    )
    df = df.merge(prev, on="horse_id", how="left")

    df["59秒以下複勝率"] = df["59.0秒以下"].apply(record_win_rate)
    df["前傾複勝率"] = df["前傾成績"].apply(record_win_rate)
    df["後傾複勝率"] = df["後傾成績"].apply(record_win_rate)

    df["宝塚キャラスコア_n"] = norm(df["宝塚キャラスコア"]) * 100
    df["上がり35秒以上スコア_n"] = norm(df["上がり35秒以上スコア"]) * 100
    df["高速5F×上がり掛かるスコア_n"] = norm(df["高速5F×上がり掛かるスコア"]) * 100
    df["淡々リズムスコア"] = norm(df["淡々リズムスコア_raw"]) * 100
    df["ローテスコア"] = (
        norm(df["前走ローテ加点"]) * 55
        + norm(df["前走負荷"]) * 20
        + norm(df["近3走連関加点"]) * 25
    )
    df["ラップ総合"] = (
        df["宝塚キャラスコア_n"] * 0.35
        + df["上がり35秒以上スコア_n"] * 0.25
        + df["高速5F×上がり掛かるスコア_n"] * 0.2
        + df["淡々リズムスコア"] * 0.12
        + df["59秒以下複勝率"].fillna(0) * 8
    )
    df["総合スコア"] = df["ラップ総合"] * 0.72 + df["ローテスコア"] * 0.28
    df["総合評価"] = df["総合スコア"].apply(final_label)
    df["評価理由"] = df.apply(reason, axis=1)

    out_cols = [
        "馬名",
        "総合評価",
        "総合スコア",
        "ラップ総合",
        "ローテスコア",
        "宝塚キャラ評価",
        "宝塚キャラスコア",
        "上がり35秒以上馬券内率",
        "複合条件馬券内率",
        "淡々リズム馬券内率",
        "59.0秒以下",
        "前傾成績",
        "前走日",
        "前走",
        "前走格",
        "前走距離",
        "前走着順",
        "前走人気",
        "前走4角",
        "前走PCI",
        "前走RPCI",
        "前走後半5F",
        "前走上がり3F",
        "前走負荷",
        "前走ローテ加点",
        "近3走連関加点",
        "前走メモ",
        "評価理由",
        "主要根拠",
    ]
    out = df[out_cols].sort_values(["総合スコア", "ラップ総合"], ascending=False).copy()
    for col in ["総合スコア", "ラップ総合", "ローテスコア"]:
        out[col] = out[col].round(1)

    csv_path = OUT_DIR / "takarazuka_kinen_2026_combined_lap_rotation_summary.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 宝塚記念2026 ラップ適性×ローテ総合評価",
        "",
        "前提: 現在のTARGET `SE_DATA`で正式成績・ラップとして読める最新は2026-05-31。6/7分はCK/DEのみで、今回の正式ラップ集計には未反映。",
        "",
        "スコア配分: ラップ総合72%、ローテ/前走28%。ラップ総合は2025型、上がり35秒以上、高速5F×上がり掛かる条件、淡々リズム、59秒以下1000m通過対応を合成。",
        "",
        "|順位|馬名|評価|総合|ラップ|ローテ|前走|前走内容|評価理由|",
        "|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for idx, row in out.head(12).reset_index(drop=True).iterrows():
        prev_text = f"{row['前走日']} {row['前走']} {row['前走着順']}着"
        prev_detail = f"{row['前走格']} 後5F{row['前走後半5F']} 上3F{row['前走上がり3F']} 負荷{row['前走負荷']}"
        lines.append(
            f"|{idx + 1}|{row['馬名']}|{row['総合評価']}|{row['総合スコア']}|"
            f"{row['ラップ総合']}|{row['ローテスコア']}|{prev_text}|{prev_detail}|{row['評価理由']}|"
        )
    lines.extend(
        [
            "",
            "## 取捨メモ",
            "",
        ]
    )
    for _, row in out.iterrows():
        lines.append(
            f"- {row['馬名']}: {row['総合評価']}。{row['評価理由']}。"
            f" 前走は{row['前走日']} {row['前走']} {row['前走着順']}着、"
            f"前走メモ={row['前走メモ'] or '特記事項薄め'}。"
        )

    report_path = OUT_DIR / "takarazuka_kinen_2026_combined_lap_rotation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
