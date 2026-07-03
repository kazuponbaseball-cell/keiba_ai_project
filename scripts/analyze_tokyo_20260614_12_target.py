from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.workout_knowledge import (
    evaluate_workout_knowledge,
    prepare_workouts_for_knowledge,
    select_entry_workouts,
)


TARGET_DU = Path(r"C:\Users\kazup\Data Lab\DE_DATA\2026\DU20260614.DAT")
WORKOUT_CSV = Path("data/processed/target/workouts_20260601_20260614.csv")
OUT_CSV = Path("outputs/analysis/tokyo_20260614_12_target_card_workout_history_check.csv")
RACE_ID = "2026061405030412"

TRAINER_NAMES = {
    "1088": "高木登",
    "1132": "金成貴史",
    "1199": "千葉直人",
    "1131": "蛯名利弘",
    "1071": "池江泰寿",
    "1145": "奥村武",
    "1151": "斉藤崇史",
    "1091": "羽月友彦",
    "1031": "伊藤伸一",
    "1127": "栗田徹",
    "1102": "大竹正博",
    "1177": "伊坂重信",
}


def dec(raw: bytes, start: int, end: int) -> str:
    return raw[start:end].decode("cp932", errors="replace").replace("\u3000", "").strip()


def full_horse_id(short_id: str) -> str:
    return "20" + short_id if len(short_id) == 8 else short_id


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
                "場所": "東京",
                "R": 12,
                "芝・ダ": "ダ",
                "距離": 1600,
                "枠番": int(dec(raw, 27, 28)),
                "馬番": int(dec(raw, 28, 30)),
                "horse_id": full_horse_id(dec(raw, 32, 40)),
                "馬名": dec(raw, 40, 76),
                "年齢": int(dec(raw, 82, 84)),
                "trainer_code": trainer_code,
                "厩舎": TRAINER_NAMES.get(trainer_code, dec(raw, 90, 100)),
                "斤量": int(dec(raw, 104, 109)) / 1000,
                "jockey_code": dec(raw, 112, 117),
                "騎手": dec(raw, 122, 132),
            }
        )
    return pd.DataFrame(rows).sort_values("馬番")


def add_workouts(card: pd.DataFrame) -> pd.DataFrame:
    workouts = pd.read_csv(
        WORKOUT_CSV,
        encoding="utf-8-sig",
        dtype={"horse_id": str, "workout_date": str},
    )
    workouts = prepare_workouts_for_knowledge(workouts)
    rows: list[dict[str, object]] = []
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
                "調教短評": evaluation.get("comment"),
                "調教本数_21日": len(selected),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def latest_history(ids: set[str]) -> pd.DataFrame:
    frames = []
    source_paths = [
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
        "target_score",
        "past3_avg_score",
        "horse_dirt_avg_score",
        "horse_dirt_top3_rate",
        "horse_time_value_plus_margin",
    ]
    for path in source_paths:
        available = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
        cols = [col for col in use_cols if col in available]
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=cols, dtype={"血統登録番号": str})
        frames.append(frame[frame["血統登録番号"].isin(ids)])
    history = pd.concat(frames, ignore_index=True).drop_duplicates(["血統登録番号", "日付", "場所", "Ｒ", "馬名"])
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
            f"{row['芝・ダ']}{int(row['距離']) if pd.notna(row['距離']) else ''}"
        ),
        axis=1,
    )
    return latest[
        [
            "horse_id",
            "前走概要",
            "前走スコア",
            "近3走平均スコア",
            "ダート平均スコア",
            "ダート複勝率",
            "タイム価値余力",
        ]
    ]


def main() -> None:
    card = add_workouts(load_card())
    history = latest_history(set(card["horse_id"].astype(str)))
    out = card.merge(history, on="horse_id", how="left")
    for col in ["前走スコア", "近3走平均スコア", "ダート平均スコア", "ダート複勝率", "調教点"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["総合仮スコア"] = (
        out["前走スコア"].fillna(0.5) * 30
        + out["近3走平均スコア"].fillna(0.5) * 25
        + out["ダート平均スコア"].fillna(0.5) * 25
        + out["ダート複勝率"].fillna(0.3) * 10
        + out["調教点"].fillna(2) / 5 * 10
    )
    out = out.sort_values("総合仮スコア", ascending=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(OUT_CSV)
    print(
        out[
            [
                "馬番",
                "馬名",
                "枠番",
                "騎手",
                "厩舎",
                "総合仮スコア",
                "前走概要",
                "前走スコア",
                "近3走平均スコア",
                "ダート平均スコア",
                "調教評価",
                "調教パターン",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
