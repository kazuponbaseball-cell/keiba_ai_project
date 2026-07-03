from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from src.utils.paths import ensure_dir, project_path


COLUMNS = [
    "日付",
    "日付S",
    "レース名",
    "クラス名",
    "馬名",
    "血統登録番号",
    "確定着順",
    "人気",
    "年齢",
    "場所",
    "芝・ダ",
    "距離",
]

ASSUMED_2026_ENTRANTS = [
    "エンブロイダリー",
    "カムニャック",
    "チェルヴィニア",
    "クイーンズウォーク",
    "ボンドガール",
    "ラヴァンダ",
    "ドロップオブライト",
    "パラディレーヌ",
    "カピリナ",
    "エリカエクスプレス",
    "ニシノティアモ",
    "アイサンサン",
    "カナテープ",
    "ココナッツブラウン",
    "ジョスラン",
    "ワイドラトゥール",
    "マピュース",
    "ケリフレッドアスク",
    "サフィラ",
    "チェルビアット",
]


def load_source(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(project_path(csv_path), encoding="cp932", usecols=COLUMNS, low_memory=False)
    for col in ["日付", "確定着順", "人気", "年齢", "距離"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["確定着順"].notna()].copy()


def bucket_of_tokyo1600_race(race_name: str, class_name: str) -> str:
    text = f"{race_name} {class_name}"
    if "G1" in text:
        return "G1"
    if "G2" in text or "G3" in text:
        if race_name.startswith("クイーン") or race_name.startswith("アルテミ"):
            return "2yo3yo重賞"
        return "G2G3"
    if "新馬" in text or "未勝利" in text:
        return "新馬未勝利"
    return "条件戦/L"


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Victoria Mile tendencies to the assumed 2026 field.")
    parser.add_argument("--csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    df = load_source(args.csv)
    df = df.sort_values(["血統登録番号", "日付"], kind="mergesort")

    ouka_top3_ids = set(df[(df["レース名"].astype(str) == "桜花賞G1") & (df["確定着順"] <= 3)]["血統登録番号"].dropna().astype(str))
    tokyo_top3_ids = set(df[(df["場所"].astype(str) == "東京") & (df["確定着順"] <= 3)]["血統登録番号"].dropna().astype(str))
    turf1600_top3_ids = set(
        df[(df["芝・ダ"].astype(str) == "芝") & (df["距離"] == 1600) & (df["確定着順"] <= 3)]["血統登録番号"].dropna().astype(str)
    )
    tokyo1600_top3 = df[
        (df["場所"].astype(str) == "東京")
        & (df["芝・ダ"].astype(str) == "芝")
        & (df["距離"] == 1600)
        & (df["確定着順"] <= 3)
    ].copy()

    rows = []
    for horse in ASSUMED_2026_ENTRANTS:
        horse_df = df[df["馬名"].astype(str) == horse].copy()
        if horse_df.empty:
            rows.append(
                {
                    "馬名": horse,
                    "local_data_found": False,
                    "コメント": "手元CSVに十分な履歴なし",
                }
            )
            continue

        horse_id = str(horse_df["血統登録番号"].dropna().astype(str).iloc[0])
        latest = horse_df.iloc[-1]
        tokyo1600_rows = tokyo1600_top3[tokyo1600_top3["血統登録番号"].astype(str) == horse_id].copy()
        tokyo1600_buckets = sorted(
            {bucket_of_tokyo1600_race(str(r["レース名"]), "" if pd.isna(r["クラス名"]) else str(r["クラス名"])) for _, r in tokyo1600_rows.iterrows()}
        )
        tokyo1600_races = sorted(set(tokyo1600_rows["レース名"].astype(str).tolist()))

        flags = {
            "has_ouka_top3": horse_id in ouka_top3_ids,
            "has_tokyo_top3": horse_id in tokyo_top3_ids,
            "has_turf1600_top3": horse_id in turf1600_top3_ids,
            "has_tokyo1600_top3": not tokyo1600_rows.empty,
            "ouka_and_tokyo1600": (horse_id in ouka_top3_ids) and (not tokyo1600_rows.empty),
        }

        score = (
            3 * int(flags["ouka_and_tokyo1600"])
            + 2 * int(flags["has_tokyo1600_top3"])
            + 1 * int(flags["has_tokyo_top3"])
            + 1 * int(flags["has_turf1600_top3"])
        )

        if flags["ouka_and_tokyo1600"]:
            note = "桜花賞実績と東京芝1600実績が両立。過去傾向上はかなり強い。"
        elif flags["has_tokyo1600_top3"]:
            note = "東京芝1600好走歴あり。少なくとも舞台接続は強め。"
        elif flags["has_turf1600_top3"]:
            note = "芝1600好走歴はあるが、東京芝1600裏付けはまだ弱い。"
        else:
            note = "手元データ上は東京芝1600接続が薄い。"

        rows.append(
            {
                "馬名": horse,
                "local_data_found": True,
                "年齢_最新": latest["年齢"],
                "人気_最新": latest["人気"],
                "最終収録日": latest["日付S"],
                "最終収録レース": latest["レース名"],
                **flags,
                "tokyo1600_bucket_count": len(tokyo1600_buckets),
                "tokyo1600_buckets": ",".join(tokyo1600_buckets),
                "tokyo1600_races": " / ".join(tokyo1600_races),
                "projection_score": score,
                "コメント": note,
            }
        )

    out = pd.DataFrame(rows).sort_values(["projection_score", "local_data_found", "年齢_最新"], ascending=[False, False, True])
    output_dir = ensure_dir(project_path(args.output_dir))
    run_dir = ensure_dir(output_dir / f"victoria_mile_2026_projection_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out.to_csv(run_dir / "projection.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_dir": str(run_dir),
        "note": "想定登録馬は外部ソースベース。手元CSVは2026-02-15までで、春ローテの最新結果は未反映。",
        "top_candidates_by_profile": out.head(10).to_dict("records"),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
