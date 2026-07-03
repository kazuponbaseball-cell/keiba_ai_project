from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


RACE_COL = "\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)"
HORSE_COL = "\u8840\u7d71\u767b\u9332\u756a\u53f7"
AGE_COL = "\u5e74\u9f62"
SURFACE_COL = "\u829d\u30fb\u30c0"
CLASS_COL = "\u30af\u30e9\u30b9\u540d"
POPULARITY_COL = "\u4eba\u6c17"
WIN_PAY_COL = "\u5358\u52dd\u914d\u5f53"
PLACE_PAY_COL = "\u8907\u52dd\u914d\u5f53"
ODDS_COL = "\u5358\u52dd\u30aa\u30c3\u30ba"

NORTHERN = "\u30ce\u30fc\u30b6\u30f3\u30d5\u30a1\u30fc\u30e0"
SHADAI = "\u793e\u53f0\u30d5\u30a1\u30fc\u30e0"
SHADAI_OLD = "\u793e\u53f0\u30d5\u30a2\u30fc\u30e0"
SHIRAOI = "\u793e\u53f0\u30b3\u30fc\u30dd\u30ec\u30fc\u30b7\u30e7\u30f3\u767d\u8001\u30d5\u30a1\u30fc\u30e0"
SHIRAOI_SHORT = "\u767d\u8001\u30d5\u30a1\u30fc\u30e0"
OIWAKAKE = "\u8ffd\u5206\u30d5\u30a1\u30fc\u30e0"


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index)
    if series.dtype == object or str(series.dtype).startswith("string"):
        series = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def _breeder_group(name: pd.Series) -> pd.Series:
    text = name.astype("string").fillna("")
    return pd.Series(
        np.select(
            [
                text.eq(NORTHERN),
                text.isin([SHADAI, SHADAI_OLD]),
                text.isin([SHIRAOI, SHIRAOI_SHORT]),
                text.eq(OIWAKAKE),
                text.str.contains("\u793e\u53f0|\u767d\u8001|\u8ffd\u5206", regex=True, na=False),
            ],
            ["northern_farm", "shadai_farm", "shiraoi_farm", "oiwake_farm", "other_shadai_group"],
            default="other",
        ),
        index=name.index,
    )


def _class_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    class_name = frame[CLASS_COL].astype("string").fillna("")
    out["is_newcomer"] = class_name.str.contains("\u65b0\u99ac", na=False)
    out["is_maiden"] = class_name.str.contains("\u672a\u52dd\u5229", na=False)
    out["is_1win"] = class_name.str.contains("1\u52dd|500\u4e07", regex=True, na=False)
    out["is_open_plus"] = class_name.str.contains("\u30aa\u30fc\u30d7\u30f3|OP|L|G", regex=True, na=False)
    return out


def _metrics(frame: pd.DataFrame, label: str) -> dict[str, object] | None:
    rows = len(frame)
    if rows == 0:
        return None
    win_pay = _num(frame.get(WIN_PAY_COL), frame.index, 0.0).fillna(0.0).where(frame["target_win"].eq(1), 0.0)
    place_pay = _num(frame.get(PLACE_PAY_COL), frame.index, 0.0).fillna(0.0).where(frame["target_top3"].eq(1), 0.0)
    return {
        "label": label,
        "bets": int(rows),
        "races": int(frame[RACE_COL].nunique()),
        "win_rate": float(frame["target_win"].mean()),
        "top3_rate": float(frame["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (rows * 100.0)),
        "place_roi": float(place_pay.sum() / (rows * 100.0)),
        "avg_popularity": float(_num(frame.get(POPULARITY_COL), frame.index).mean()),
        "avg_odds": float(_num(frame.get(ODDS_COL), frame.index).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_enriched/test_features_with_same_day_bias_v3_retro_body_owner.csv",
    )
    parser.add_argument("--breeder-master-csv", default="data/processed/target/breeder_master.csv")
    parser.add_argument("--model", default="models/body_context_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--output-dir", default="outputs/analysis/breeder_group_segments")
    args = parser.parse_args()

    frame = pd.read_csv(args.test_csv, low_memory=False)
    breeder = pd.read_csv(args.breeder_master_csv, low_memory=False)
    breeder = breeder[[HORSE_COL, "breeder_name", "breeder_code", "birthplace"]].drop_duplicates(HORSE_COL)
    frame = frame.merge(breeder, on=HORSE_COL, how="left")

    with Path(args.model).open("rb") as f:
        model = pickle.load(f)
    frame["ai_score"] = model.predict(frame)
    frame["ai_rank"] = frame.groupby(RACE_COL)["ai_score"].rank(ascending=False, method="first").astype(int)

    frame["breeder_group"] = _breeder_group(frame["breeder_name"])
    frame["is_turf"] = frame[SURFACE_COL].astype("string").str.contains("\u829d", na=False)
    frame["is_dirt"] = frame[SURFACE_COL].astype("string").str.contains("\u30c0", na=False)
    age = _num(frame[AGE_COL], frame.index)
    frame["is_young"] = age.le(3)
    frame["is_2yo"] = age.eq(2)
    frame["is_3yo"] = age.eq(3)
    frame = pd.concat([frame, _class_flags(frame)], axis=1)

    conditions: list[tuple[pd.Series, str]] = [
        (pd.Series(True, index=frame.index), "all"),
        (frame["is_turf"], "turf"),
        (frame["is_dirt"], "dirt"),
        (frame["is_young"], "young_le3"),
        (frame["is_turf"] & frame["is_young"], "turf_young_le3"),
        (frame["is_turf"] & frame["is_2yo"], "turf_2yo"),
        (frame["is_turf"] & frame["is_3yo"], "turf_3yo"),
        (frame["is_turf"] & frame["is_newcomer"], "turf_newcomer"),
        (frame["is_turf"] & frame["is_maiden"], "turf_maiden"),
        (frame["is_turf"] & frame["is_1win"], "turf_1win"),
        (frame["is_turf"] & frame["is_open_plus"], "turf_open_plus"),
    ]

    rows: list[dict[str, object]] = []
    groups = ["northern_farm", "shadai_farm", "shiraoi_farm", "oiwake_farm", "other_shadai_group", "other"]
    for group in groups:
        base = frame["breeder_group"].eq(group)
        for mask, suffix in conditions:
            result = _metrics(frame[base & mask], f"{group}:{suffix}")
            if result:
                rows.append(result)
        for rank_mask, suffix in [
            (frame["ai_rank"].eq(1), "ai_top1_all"),
            (frame["ai_rank"].eq(1) & frame["is_turf"] & frame["is_young"], "ai_top1_turf_young"),
            (frame["ai_rank"].le(3) & frame["is_turf"] & frame["is_young"], "ai_top3_turf_young"),
            (frame["ai_rank"].eq(1) & frame["is_turf"] & frame["is_2yo"], "ai_top1_turf_2yo"),
        ]:
            result = _metrics(frame[base & rank_mask], f"{group}:{suffix}")
            if result:
                rows.append(result)

    summary = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "breeder_group_segments.csv", index=False, encoding="utf-8-sig")
    focused = summary[
        summary["label"].str.contains("turf_young|turf_2yo|turf_newcomer|ai_top1_turf_young|ai_top1_turf_2yo", regex=True)
    ].copy()
    focused.to_csv(output_dir / "breeder_group_focused_segments.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "output_dir": str(output_dir),
        "breeder_coverage": float(frame["breeder_name"].notna().mean()),
        "group_counts": frame["breeder_group"].value_counts().to_dict(),
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps({"output_dir": str(output_dir), "breeder_coverage": metadata["breeder_coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
