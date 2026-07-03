from __future__ import annotations

import argparse
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config  # noqa: E402
from src.utils.paths import ensure_dir, project_path  # noqa: E402


DEFAULT_TEST = "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv"
DEFAULT_MODEL = "models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl"


MaskFunc = Callable[[pd.DataFrame], pd.Series]


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    series = df[col]
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _pay(df: pd.DataFrame, col: str) -> pd.Series:
    return _num(df, col, 0.0).fillna(0.0)


def add_scores(df: pd.DataFrame, model_path: Path, race_col: str) -> pd.DataFrame:
    with model_path.open("rb") as f:
        model = pickle.load(f)
    out = df.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    out["popularity_num"] = _num(out, "人気")
    out["odds_decimal"] = _num(out, "単勝オッズ")
    out["pop_rank"] = out.groupby(race_col)["popularity_num"].rank(ascending=True, method="first")
    out["ai_pop_gap"] = out["ai_rank"] - out["pop_rank"]
    return out


def metrics(df: pd.DataFrame, race_col: str, label: str) -> dict[str, object]:
    rows = len(df)
    if rows == 0:
        return {
            "segment": label,
            "bets": 0,
            "races": 0,
            "win_rate": 0.0,
            "top3_rate": 0.0,
            "win_roi": 0.0,
            "place_roi": 0.0,
            "avg_popularity": None,
            "avg_odds": None,
            "avg_ai_rank": None,
            "avg_prev_body_weight": None,
        }
    win_pay = _pay(df, "単勝配当").where(df["target_win"].eq(1), 0.0)
    place_pay = _pay(df, "複勝配当").where(df["target_top3"].eq(1), 0.0)
    stake = rows * 100.0
    return {
        "segment": label,
        "bets": int(rows),
        "races": int(df[race_col].nunique()),
        "win_rate": float(df["target_win"].mean()),
        "top3_rate": float(df["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / stake),
        "place_roi": float(place_pay.sum() / stake),
        "avg_popularity": float(_num(df, "人気").mean()),
        "avg_odds": float(_num(df, "単勝オッズ").mean()),
        "avg_ai_rank": float(df["ai_rank"].mean()) if "ai_rank" in df.columns else None,
        "avg_prev_body_weight": float(_num(df, "前走馬体重").mean()),
    }


def temporal_split(df: pd.DataFrame, date_col: str) -> tuple[pd.Series, pd.Series]:
    dates = pd.to_numeric(df[date_col], errors="coerce")
    unique_dates = np.array(sorted(dates.dropna().unique()))
    cutoff = unique_dates[len(unique_dates) // 2]
    return dates <= cutoff, dates > cutoff


def class_bucket(class_name: object) -> str:
    text = str(class_name)
    if "新馬" in text:
        return "newcomer"
    if "未勝利" in text:
        return "maiden"
    if "1勝" in text:
        return "class_1win"
    if "2勝" in text:
        return "class_2win"
    if "3勝" in text:
        return "class_3win"
    if "Ｇ" in text or "G" in text:
        return "graded"
    if "OP" in text or "ｵｰﾌﾟﾝ" in text or "オープン" in text:
        return "open"
    return "other"


def add_body_columns(df: pd.DataFrame, race_col: str) -> pd.DataFrame:
    out = df.copy()
    weight = _num(out, "前走馬体重")
    out["prev_body_weight_num"] = weight
    out["body_weight_bucket"] = pd.cut(
        weight,
        bins=[0, 439, 459, 479, 499, 519, 539, 1000],
        labels=["<=439", "440-459", "460-479", "480-499", "500-519", "520-539", "540+"],
        include_lowest=True,
    ).astype(str)
    out.loc[weight.isna(), "body_weight_bucket"] = "missing"
    mean = weight.groupby(out[race_col]).transform("mean")
    std = weight.groupby(out[race_col]).transform("std").replace(0, np.nan)
    out["body_weight_z_in_race"] = ((weight - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["body_weight_rank_in_race"] = weight.groupby(out[race_col]).rank(ascending=False, method="average")
    out["class_bucket"] = out["クラス名"].map(class_bucket)
    return out


def segment_rows(scored: pd.DataFrame, race_col: str, date_col: str) -> pd.DataFrame:
    discovery_mask, validation_mask = temporal_split(scored, date_col)
    rows: list[dict[str, object]] = []

    masks: dict[str, pd.Series] = {
        "all": pd.Series(True, index=scored.index),
        "ai_top1": scored["ai_rank"].eq(1),
        "ai_top3": scored["ai_rank"].le(3),
        "ai_top5": scored["ai_rank"].le(5),
        "pop5plus": scored["popularity_num"].ge(5),
        "market_gap3plus": scored["ai_pop_gap"].le(-3),
        "turf": scored["芝・ダ"].astype(str).str.contains("芝", na=False),
        "dirt": scored["芝・ダ"].astype(str).str.contains("ダ", na=False),
        "local": scored["場所"].isin({"福島", "小倉", "函館", "札幌"}),
        "central": scored["場所"].isin({"東京", "中山", "阪神", "京都"}),
        "big_500": scored["prev_body_weight_num"].ge(500),
        "big_520": scored["prev_body_weight_num"].ge(520),
        "small_460": scored["prev_body_weight_num"].le(460),
        "race_heavy_top3": scored["body_weight_rank_in_race"].le(3),
        "race_light_bottom3": scored["body_weight_rank_in_race"].ge(scored.groupby(race_col)["body_weight_rank_in_race"].transform("max") - 2),
    }
    for bucket in sorted(scored["body_weight_bucket"].dropna().unique()):
        masks[f"bw_{bucket}"] = scored["body_weight_bucket"].eq(bucket)
    for cls in sorted(scored["class_bucket"].dropna().unique()):
        masks[f"class_{cls}"] = scored["class_bucket"].eq(cls)

    single_names = [
        "all",
        "ai_top1",
        "ai_top3",
        "ai_top5",
        "big_500",
        "big_520",
        "small_460",
        "race_heavy_top3",
        "race_light_bottom3",
    ] + [name for name in masks if name.startswith("bw_")]

    combos = []
    for name in single_names:
        combos.append((name,))
    base_combo_names = [
        "ai_top1",
        "ai_top3",
        "ai_top5",
        "pop5plus",
        "market_gap3plus",
        "turf",
        "dirt",
        "local",
        "central",
        "big_500",
        "big_520",
        "small_460",
        "race_heavy_top3",
    ]
    for size in [2, 3, 4]:
        for combo in combinations(base_combo_names, size):
            if not any(name in combo for name in ["big_500", "big_520", "small_460", "race_heavy_top3"]):
                continue
            if not any(name.startswith("ai_top") for name in combo):
                continue
            combos.append(combo)
    for cls in [name for name in masks if name.startswith("class_")]:
        for bw in ["big_500", "big_520", "small_460", "race_heavy_top3"]:
            for ai in ["ai_top1", "ai_top3"]:
                combos.append((cls, bw, ai))

    seen = set()
    for combo in combos:
        if combo in seen:
            continue
        seen.add(combo)
        label = "&".join(combo)
        mask = pd.Series(True, index=scored.index)
        for name in combo:
            mask &= masks[name].fillna(False)
        all_part = scored[mask]
        if len(all_part) < 40:
            continue
        discovery_part = scored[mask & discovery_mask]
        validation_part = scored[mask & validation_mask]
        if len(discovery_part) < 20 or len(validation_part) < 20:
            continue
        rows.append(
            {
                "segment": label,
                **{f"all_{k}": v for k, v in metrics(all_part, race_col, label).items() if k != "segment"},
                **{f"discovery_{k}": v for k, v in metrics(discovery_part, race_col, label).items() if k != "segment"},
                **{f"validation_{k}": v for k, v in metrics(validation_part, race_col, label).items() if k != "segment"},
            }
        )

    out = pd.DataFrame(rows)
    out["validation_edge_score"] = (
        out["validation_win_roi"] + 0.35 * out["validation_place_roi"] + 0.2 * out["validation_top3_rate"]
    )
    out["roi_stability_win"] = out["validation_win_roi"] - out["discovery_win_roi"]
    out["roi_stability_place"] = out["validation_place_roi"] - out["discovery_place_roi"]
    return out.sort_values(["validation_edge_score", "validation_win_roi", "validation_place_roi"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze previous body-weight segments.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--output-dir", default="outputs/analysis/body_weight_segments")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    date_col = config["data"]["date_column"]

    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = add_scores(df, project_path(args.model), race_col)
    scored = add_body_columns(scored, race_col)
    out = segment_rows(scored, race_col, date_col)

    output_dir = ensure_dir(project_path(args.output_dir))
    out.to_csv(output_dir / "body_weight_segments.csv", index=False, encoding="utf-8-sig")
    out.head(80).to_csv(output_dir / "top_body_weight_segments.csv", index=False, encoding="utf-8-sig")

    bucket_rows = []
    for bucket, part in scored.groupby("body_weight_bucket", sort=False):
        bucket_rows.append(metrics(part, race_col, str(bucket)))
    bucket_df = pd.DataFrame(bucket_rows).sort_values("segment")
    bucket_df.to_csv(output_dir / "body_weight_bucket_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(output_dir),
        "rows": int(len(scored)),
        "races": int(scored[race_col].nunique()),
        "bucket_metrics": bucket_df.to_dict(orient="records"),
        "top_segments": out.head(30).to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
