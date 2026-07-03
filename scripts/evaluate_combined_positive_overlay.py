from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index)
    if series.dtype == object or str(series.dtype).startswith("string"):
        series = series.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _score(frame: pd.DataFrame, model: object, race_col: str, label: str) -> pd.DataFrame:
    out = frame.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    out["model_label"] = label
    return out


def _metrics(frame: pd.DataFrame, race_col: str, label: str) -> dict[str, object]:
    rows = len(frame)
    if rows == 0:
        return {
            "label": label,
            "bets": 0,
            "races": 0,
            "win_rate": 0.0,
            "top3_rate": 0.0,
            "win_roi": 0.0,
            "place_roi": 0.0,
            "avg_popularity": None,
            "avg_odds": None,
        }
    win_pay = _num(frame.get("単勝配当"), frame.index, 0.0).fillna(0.0).where(frame["target_win"].eq(1), 0.0)
    place_pay = _num(frame.get("複勝配当"), frame.index, 0.0).fillna(0.0).where(frame["target_top3"].eq(1), 0.0)
    stake = rows * 100.0
    return {
        "label": label,
        "bets": int(rows),
        "races": int(frame[race_col].nunique()),
        "win_rate": float(frame["target_win"].mean()),
        "top3_rate": float(frame["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / stake),
        "place_roi": float(place_pay.sum() / stake),
        "avg_popularity": float(_num(frame.get("人気"), frame.index).mean()) if "人気" in frame.columns else None,
        "avg_odds": float(_num(frame.get("単勝オッズ"), frame.index).mean()) if "単勝オッズ" in frame.columns else None,
    }


def _add_context(out: pd.DataFrame) -> pd.DataFrame:
    surface = out.get("芝・ダ", pd.Series("", index=out.index)).astype("string")
    out["surface_simple"] = np.select(
        [surface.str.contains("芝", na=False), surface.str.contains("ダ", na=False)],
        ["turf", "dirt"],
        default="other",
    )
    distance = _num(out.get("距離"), out.index)
    out["distance_bucket_simple"] = np.select(
        [distance <= 1300, distance.between(1400, 1800), distance.between(1900, 2400), distance >= 2500],
        ["sprint_1300", "mile_1600", "middle_2000", "long_2600"],
        default="unknown",
    )
    class_name = out.get("クラス名", pd.Series("", index=out.index)).astype("string")
    out["class_bucket"] = np.select(
        [
            class_name.str.contains("新馬", na=False),
            class_name.str.contains("未勝利", na=False),
            class_name.str.contains("１勝|1勝|500万", regex=True, na=False),
            class_name.str.contains("２勝|2勝|1000万", regex=True, na=False),
            class_name.str.contains("３勝|3勝|1600万", regex=True, na=False),
            class_name.str.contains("オープン|OP|Ｌ|L", regex=True, na=False),
            class_name.str.contains("Ｇ|G", regex=True, na=False),
        ],
        ["newcomer", "maiden", "class_1win", "class_2win", "class_3win", "open", "graded"],
        default="other",
    )
    going = out.get("馬場状態", pd.Series("", index=out.index)).astype("string")
    out["going_bucket"] = np.select(
        [
            going.str.contains("良|稍", regex=True, na=False),
            going.str.contains("重", regex=True, na=False),
            going.str.contains("不", regex=True, na=False),
        ],
        ["firm_good", "yielding", "muddy"],
        default="unknown",
    )
    return out


def _build_overlay(out: pd.DataFrame) -> pd.DataFrame:
    out = _add_context(out)
    flags: dict[str, pd.Series] = {}
    flags["age2_ai1_big500"] = out["ai_rank"].eq(1) & out["body_age2_big500_flag"].eq(1)
    flags["age2_ai1_heavy_dirt"] = (
        out["ai_rank"].eq(1) & out["body_age2_race_heavy_top3_flag"].eq(1) & out["surface_simple"].eq("dirt")
    )
    flags["age2_ai1_heavy_mid2000"] = (
        out["ai_rank"].eq(1)
        & out["body_age2_race_heavy_top3_flag"].eq(1)
        & out["distance_bucket_simple"].eq("middle_2000")
    )
    flags["age3_ai1_heavy_long"] = (
        out["ai_rank"].eq(1)
        & out["body_age3_race_heavy_top5_flag"].eq(1)
        & out["distance_bucket_simple"].eq("long_2600")
    )
    flags["firm_turf_ai1_big520"] = (
        out["ai_rank"].eq(1)
        & out["body_very_large_horse_flag"].eq(1)
        & out["going_bucket"].eq("firm_good")
        & out["surface_simple"].eq("turf")
    )
    flags["yielding_dirt_ai3_big520"] = (
        out["ai_rank"].le(3)
        & out["body_very_large_horse_flag"].eq(1)
        & out["going_bucket"].eq("yielding")
        & out["surface_simple"].eq("dirt")
    )
    flags["class_1win_young_body_ai5"] = (
        out["ai_rank"].le(5)
        & out["class_bucket"].eq("class_1win")
        & (out["body_age2_race_heavy_top5_flag"].eq(1) | out["body_age3_race_heavy_top5_flag"].eq(1))
    )
    if "course_size" in out.columns:
        frame = _num(out.get("枠番"), out.index)
        front = _num(out.get("front_running_tendency"), out.index).fillna(0.0)
        flags["small_tight_turf_inner_front_ai1"] = (
            out["ai_rank"].eq(1)
            & out["course_size"].eq("small_tight")
            & out["surface_simple"].eq("turf")
            & frame.le(3)
            & front.ge(0.45)
        )
        flags["local_dirt_outer_front_ai3"] = (
            out["ai_rank"].le(3)
            & out["course_size"].eq("local")
            & out["surface_simple"].eq("dirt")
            & frame.ge(7)
            & front.ge(0.45)
        )
    flag_frame = pd.DataFrame({name: value.fillna(False).astype(bool) for name, value in flags.items()}, index=out.index)
    out["positive_overlay_count"] = flag_frame.sum(axis=1)
    out["positive_overlay_names"] = flag_frame.apply(lambda row: "|".join(flag_frame.columns[row.to_numpy()]), axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/test_features_with_same_day_bias_v3_retro_body_context.csv",
    )
    parser.add_argument("--new-model", default="models/body_context_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--old-model", default="models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--output-dir", default="outputs/analysis/combined_positive_overlay")
    parser.add_argument("--race-col", default="レースID(新/馬番無)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.test_csv, encoding="utf-8-sig", low_memory=False)
    with Path(args.new_model).open("rb") as f:
        new_model = pickle.load(f)
    with Path(args.old_model).open("rb") as f:
        old_model = pickle.load(f)

    old = _score(frame, old_model, args.race_col, "old")
    new = _build_overlay(_score(frame, new_model, args.race_col, "new"))

    rows = [
        _metrics(old[old["ai_rank"].eq(1)], args.race_col, "old_base: ai_top1"),
        _metrics(old[old["ai_rank"].le(3)], args.race_col, "old_base: ai_top3_all"),
        _metrics(new[new["ai_rank"].eq(1)], args.race_col, "new_body_age_features: ai_top1"),
        _metrics(new[new["ai_rank"].le(3)], args.race_col, "new_body_age_features: ai_top3_all"),
    ]
    overlay = new[new["positive_overlay_count"].gt(0)].copy()
    rows.extend(
        [
            _metrics(overlay, args.race_col, "new+positive_overlay: all_flagged"),
            _metrics(overlay[overlay["ai_rank"].eq(1)], args.race_col, "new+positive_overlay: ai_top1_flagged"),
            _metrics(overlay[overlay["ai_rank"].le(3)], args.race_col, "new+positive_overlay: ai_top3_flagged"),
        ]
    )
    best = (
        overlay.sort_values([args.race_col, "positive_overlay_count", "ai_score"], ascending=[True, False, False])
        .groupby(args.race_col, as_index=False)
        .head(1)
    )
    rows.append(_metrics(best, args.race_col, "new+positive_overlay: best1_per_race"))
    conservative = new[
        (new["ai_rank"].eq(1) & new["positive_overlay_count"].gt(0))
        | (new["ai_rank"].le(3) & new["positive_overlay_count"].ge(2))
    ].copy()
    rows.append(_metrics(conservative, args.race_col, "new+positive_overlay: conservative_all"))
    conservative_best = (
        conservative.sort_values([args.race_col, "positive_overlay_count", "ai_score"], ascending=[True, False, False])
        .groupby(args.race_col, as_index=False)
        .head(1)
    )
    rows.append(_metrics(conservative_best, args.race_col, "new+positive_overlay: conservative_best1"))

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "combined_overlay_summary.csv", index=False, encoding="utf-8-sig")
    detail_cols = [
        args.race_col,
        "日付",
        "場所",
        "Ｒ",
        "レース名",
        "馬番",
        "馬名",
        "ai_rank",
        "ai_score",
        "positive_overlay_count",
        "positive_overlay_names",
        "target_win",
        "target_top3",
        "単勝オッズ",
        "単勝配当",
        "複勝配当",
    ]
    new[[col for col in detail_cols if col in new.columns]].to_csv(
        output_dir / "scored_with_overlay_flags.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {"output_dir": str(output_dir), "summary": summary.to_dict(orient="records")}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(summary.to_string(index=False))
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
