from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.workout_knowledge import (  # noqa: E402
    evaluate_workout_knowledge,
    prepare_workouts_for_knowledge,
    select_entry_workouts,
)


TARGET_DU = Path(r"C:\Users\kazup\Data Lab\DE_DATA\2026\DU20260614.DAT")
WORKOUT_CSV = Path("data/processed/target/workouts_20260601_20260614.csv")
OUT_CSV = Path("outputs/analysis/hanshin_20260614_12_heavy_target_card_workout_history_check.csv")
RACE_ID = "2026061409030412"

TRAINERS = {
    "1157": "杉山晴紀",
    "1129": "高橋義忠",
    "1109": "伊藤大士",
    "1115": "菊沢隆徳",
    "1144": "池添学",
    "1095": "荒川義之",
    "1214": "井上智史",
    "1183": "辻野泰之",
    "1023": "伊藤圭三",
    "1065": "大橋勇樹",
    "1117": "高野友和",
    "1140": "石橋守",
    "1113": "牧浦充徳",
    "1102": "大竹正博",
    "1136": "高橋亮",
    "1166": "石坂公一",
}


def dec(raw: bytes, start: int, end: int) -> str:
    return raw[start:end].decode("cp932", errors="replace").replace("\u3000", "").strip()


def horse_id(short: str) -> str:
    return "20" + short if len(short) == 8 else short


def load_card() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw in TARGET_DU.read_bytes().splitlines():
        if RACE_ID.encode("ascii") not in raw:
            continue
        trainer_code = dec(raw, 86, 90)
        rows.append(
            {
                "race_id": RACE_ID,
                "date": "20260614",
                "場所": "阪神",
                "R": 12,
                "レース名": "リボン賞",
                "芝・ダ": "ダ",
                "距離": 1200,
                "馬場状態": "重",
                "枠番": int(dec(raw, 27, 28)),
                "馬番": int(dec(raw, 28, 30)),
                "horse_id": horse_id(dec(raw, 32, 40)),
                "馬名": dec(raw, 40, 76),
                "年齢": int(dec(raw, 82, 84)),
                "trainer_code": trainer_code,
                "厩舎": TRAINERS.get(trainer_code, dec(raw, 90, 100)),
                "斤量": int(dec(raw, 104, 109)) / 1000,
                "jockey_code": dec(raw, 112, 117),
                "騎手": dec(raw, 122, 132),
            }
        )
    return pd.DataFrame(rows).sort_values("馬番")


def add_workouts(card: pd.DataFrame) -> pd.DataFrame:
    workouts = pd.read_csv(WORKOUT_CSV, encoding="utf-8-sig", dtype={"horse_id": str, "workout_date": str})
    workouts = prepare_workouts_for_knowledge(workouts)
    rows = []
    for _, row in card.iterrows():
        entry = pd.Series(
            {
                "horse_id": row["horse_id"],
                "date": row["date"],
                "trainer_code": row["trainer_code"],
                "horse_name": row["馬名"],
                "surface": "ダ",
                "jockey": row["騎手"],
                "age": row["年齢"],
                "芝・ダ": "ダ",
                "騎手": row["騎手"],
                "馬名": row["馬名"],
                "人気": np.nan,
            }
        )
        selected = select_entry_workouts(entry, workouts, lookback_days=21)
        evaluation = evaluate_workout_knowledge(entry, selected)
        latest = selected.sort_values(["workout_date_dt", "workout_time"]).tail(1)
        latest_values = {
            "最新追切日": "",
            "最新コース": "",
            "最新総時計": np.nan,
            "最新終い1F": np.nan,
            "最新ラップ分類": "",
        }
        if not latest.empty:
            latest_row = latest.iloc[0]
            latest_values = {
                "最新追切日": str(latest_row.get("workout_date", "")),
                "最新コース": latest_row.get("course_bucket", ""),
                "最新総時計": latest_row.get("total_time_sec", np.nan),
                "最新終い1F": latest_row.get("final_1f_sec", np.nan),
                "最新ラップ分類": latest_row.get("lap_group", ""),
            }
        out = row.to_dict()
        out.update(latest_values)
        out.update(
            {
                "調教評価": evaluation.get("grade"),
                "調教点": evaluation.get("grade_score"),
                "調教パターン": evaluation.get("matched_pattern"),
                "調教加点": "; ".join(evaluation.get("plus_factors", []) or []),
                "調教本数_21日": len(selected),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def load_history(ids: set[str]) -> pd.DataFrame:
    paths = [
        Path("data/datasets/cache/lap_pedigree_interactions_confirmed_opponent/train_features.csv"),
        Path("data/datasets/cache/lap_pedigree_interactions_confirmed_opponent/test_features.csv"),
    ]
    use_cols = [
        "日付",
        "場所",
        "Ｒ",
        "レース名",
        "馬名",
        "血統登録番号",
        "確定着順",
        "人気",
        "芝・ダ",
        "距離",
        "馬場状態",
        "target_score",
        "past3_avg_score",
        "horse_dirt_avg_score",
        "horse_dirt_top3_rate",
        "horse_time_value_plus_margin",
    ]
    frames = []
    for path in paths:
        available = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
        cols = [col for col in use_cols if col in available]
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=cols, dtype={"血統登録番号": str})
        frames.append(frame[frame["血統登録番号"].isin(ids)])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["血統登録番号", "日付", "場所", "Ｒ", "馬名"])


def add_history(card: pd.DataFrame) -> pd.DataFrame:
    history = load_history(set(card["horse_id"].astype(str)))
    history["日付_num"] = pd.to_numeric(history["日付"], errors="coerce")
    latest = history.sort_values(["血統登録番号", "日付_num"]).groupby("血統登録番号").tail(1).copy()
    latest = latest.rename(
        columns={
            "血統登録番号": "horse_id",
            "target_score": "前走スコア",
            "past3_avg_score": "近3走平均スコア",
            "horse_dirt_avg_score": "ダート平均スコア",
            "horse_dirt_top3_rate": "ダート複勝率",
            "horse_time_value_plus_margin": "タイム価値余力",
        }
    )
    latest["前走概要"] = latest.apply(
        lambda row: (
            f"{int(row['日付'])} {row['場所']}{int(row['Ｒ'])}R {row['レース名']} "
            f"{int(row['確定着順'])}着/{int(row['人気']) if pd.notna(row['人気']) else ''}人気 "
            f"{row['芝・ダ']}{int(row['距離']) if pd.notna(row['距離']) else ''} {row.get('馬場状態', '')}"
        ),
        axis=1,
    )
    bad = history[(history["芝・ダ"] == "ダ") & (history["馬場状態"].isin(["稍重", "重", "不良"]))].copy()
    if not bad.empty:
        bad["top3"] = (pd.to_numeric(bad["確定着順"], errors="coerce") <= 3).astype(float)
        bad_summary = (
            bad.groupby("血統登録番号")
            .agg(
                道悪ダ戦数=("確定着順", "count"),
                道悪ダ複勝率=("top3", "mean"),
                道悪ダ平均スコア=("target_score", "mean"),
            )
            .reset_index()
            .rename(columns={"血統登録番号": "horse_id"})
        )
    else:
        bad_summary = pd.DataFrame(columns=["horse_id", "道悪ダ戦数", "道悪ダ複勝率", "道悪ダ平均スコア"])
    keep = [
        "horse_id",
        "前走概要",
        "前走スコア",
        "近3走平均スコア",
        "ダート平均スコア",
        "ダート複勝率",
        "タイム価値余力",
    ]
    return card.merge(latest[keep], on="horse_id", how="left").merge(bad_summary, on="horse_id", how="left")


def score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in [
        "前走スコア",
        "近3走平均スコア",
        "ダート平均スコア",
        "ダート複勝率",
        "調教点",
        "道悪ダ複勝率",
        "道悪ダ平均スコア",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["重馬場補正"] = (
        out["道悪ダ平均スコア"].fillna(out["ダート平均スコア"]).fillna(0.5) * 12
        + out["道悪ダ複勝率"].fillna(out["ダート複勝率"]).fillna(0.3) * 8
    )
    out["斤量補正"] = (58 - out["斤量"]).clip(lower=0) * 1.2
    out["枠補正"] = np.where(out["枠番"] <= 3, 2.0, np.where(out["枠番"] >= 7, -0.5, 0.5))
    out["総合仮スコア"] = (
        out["前走スコア"].fillna(0.5) * 25
        + out["近3走平均スコア"].fillna(0.5) * 20
        + out["ダート平均スコア"].fillna(0.5) * 20
        + out["重馬場補正"]
        + out["調教点"].fillna(2) / 5 * 8
        + out["斤量補正"]
        + out["枠補正"]
    )
    return out.sort_values("総合仮スコア", ascending=False)


def main() -> None:
    out = score(add_history(add_workouts(load_card())))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(OUT_CSV)
    print(
        out[
            [
                "馬番",
                "馬名",
                "枠番",
                "斤量",
                "騎手",
                "厩舎",
                "総合仮スコア",
                "前走概要",
                "前走スコア",
                "近3走平均スコア",
                "ダート平均スコア",
                "道悪ダ戦数",
                "道悪ダ複勝率",
                "調教評価",
                "調教パターン",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
