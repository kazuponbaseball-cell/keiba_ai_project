from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config  # noqa: E402
from src.utils.paths import ensure_dir, project_path  # noqa: E402


DEFAULT_TEST = "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/test_features_with_same_day_bias_v3_retro_body_weight_backfilled.csv"
DEFAULT_MODEL = "models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl"


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    series = df[col]
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


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


def add_context(df: pd.DataFrame, race_col: str) -> pd.DataFrame:
    out = df.copy()
    weight = _num(out, "前走馬体重")
    out["prev_body_weight_num"] = weight
    out["age_num"] = _num(out, "年齢")
    out["age_bucket"] = np.select(
        [
            out["age_num"].eq(2),
            out["age_num"].eq(3),
            out["age_num"].eq(4),
            out["age_num"].ge(5),
        ],
        ["age2", "age3", "age4", "age5plus"],
        default="unknown",
    )
    out["body_weight_bucket"] = pd.cut(
        weight,
        bins=[0, 439, 459, 479, 499, 519, 539, 1000],
        labels=["<=439", "440-459", "460-479", "480-499", "500-519", "520-539", "540+"],
        include_lowest=True,
    ).astype(str)
    out.loc[weight.isna(), "body_weight_bucket"] = "missing"
    out["maturity_size_bucket"] = np.select(
        [
            weight.le(439),
            weight.between(440, 459, inclusive="both"),
            weight.between(460, 499, inclusive="both"),
            weight.between(500, 519, inclusive="both"),
            weight.ge(520),
        ],
        ["very_small", "small", "mid_460_499", "big_500_519", "very_big_520"],
        default="missing",
    )
    out["surface_simple"] = np.select(
        [
            out["芝・ダ"].astype(str).str.contains("芝", na=False),
            out["芝・ダ"].astype(str).str.contains("ダ", na=False),
        ],
        ["turf", "dirt"],
        default="other",
    )
    out["distance_bucket_simple"] = pd.cut(
        _num(out, "距離"),
        bins=[0, 1300, 1600, 2000, 2600, 10000],
        labels=["sprint_1300", "mile_1600", "middle_2000", "long_2600", "stayer_2600plus"],
        include_lowest=True,
    ).astype(str)
    out["class_bucket"] = out["クラス名"].map(class_bucket)
    out["body_weight_rank_in_race"] = weight.groupby(out[race_col]).rank(ascending=False, method="average")
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
    win_pay = _num(df, "単勝配当", 0.0).fillna(0.0).where(df["target_win"].eq(1), 0.0)
    place_pay = _num(df, "複勝配当", 0.0).fillna(0.0).where(df["target_top3"].eq(1), 0.0)
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
        "avg_ai_rank": float(df["ai_rank"].mean()),
        "avg_prev_body_weight": float(_num(df, "前走馬体重").mean()),
    }


def temporal_masks(df: pd.DataFrame, date_col: str) -> tuple[pd.Series, pd.Series]:
    dates = pd.to_numeric(df[date_col], errors="coerce")
    unique_dates = np.array(sorted(dates.dropna().unique()))
    cutoff = unique_dates[len(unique_dates) // 2]
    return dates <= cutoff, dates > cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze previous body weight by age/maturity.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--output-dir", default="outputs/analysis/body_weight_age_segments")
    parser.add_argument("--min-bets", type=int, default=80)
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    date_col = config["data"]["date_column"]

    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = add_context(add_scores(df, project_path(args.model), race_col), race_col)
    discovery_mask, validation_mask = temporal_masks(scored, date_col)

    masks: dict[str, pd.Series] = {
        "all": pd.Series(True, index=scored.index),
        "ai_top1": scored["ai_rank"].eq(1),
        "ai_top3": scored["ai_rank"].le(3),
        "ai_top5": scored["ai_rank"].le(5),
        "big_500": scored["prev_body_weight_num"].ge(500),
        "big_520": scored["prev_body_weight_num"].ge(520),
        "mid_460_499": scored["prev_body_weight_num"].between(460, 499, inclusive="both"),
        "small_460": scored["prev_body_weight_num"].le(460),
        "race_heavy_top3": scored["body_weight_rank_in_race"].le(3),
        "race_heavy_top5": scored["body_weight_rank_in_race"].le(5),
    }
    for col in ["age_bucket", "maturity_size_bucket", "body_weight_bucket", "surface_simple", "distance_bucket_simple", "class_bucket"]:
        for value in sorted(scored[col].dropna().unique()):
            masks[f"{col}={value}"] = scored[col].eq(value)

    rows: list[dict[str, object]] = []
    combos: list[tuple[str, ...]] = []
    age_terms = ["age_bucket=age2", "age_bucket=age3", "age_bucket=age4", "age_bucket=age5plus"]
    body_terms = ["big_500", "big_520", "mid_460_499", "small_460", "race_heavy_top3", "race_heavy_top5"]
    for age in age_terms:
        for body in body_terms:
            combos.append((age, body))
            for ai in ["ai_top1", "ai_top3", "ai_top5"]:
                combos.append((ai, age, body))
                for surface in ["surface_simple=turf", "surface_simple=dirt"]:
                    combos.append((ai, age, body, surface))
                for cls in ["class_bucket=newcomer", "class_bucket=maiden", "class_bucket=class_1win", "class_bucket=graded"]:
                    combos.append((ai, age, body, cls))
                for distance in [
                    "distance_bucket_simple=sprint_1300",
                    "distance_bucket_simple=mile_1600",
                    "distance_bucket_simple=middle_2000",
                    "distance_bucket_simple=long_2600",
                ]:
                    combos.append((ai, age, body, distance))

    seen = set()
    for combo in combos:
        if combo in seen or any(name not in masks for name in combo):
            continue
        seen.add(combo)
        mask = pd.Series(True, index=scored.index)
        for name in combo:
            mask &= masks[name].fillna(False)
        all_part = scored[mask]
        if len(all_part) < args.min_bets:
            continue
        discovery_part = scored[mask & discovery_mask]
        validation_part = scored[mask & validation_mask]
        if len(discovery_part) < max(25, args.min_bets // 3) or len(validation_part) < max(25, args.min_bets // 3):
            continue
        label = "&".join(combo)
        rows.append(
            {
                "segment": label,
                **{f"all_{k}": v for k, v in metrics(all_part, race_col, label).items() if k != "segment"},
                **{f"discovery_{k}": v for k, v in metrics(discovery_part, race_col, label).items() if k != "segment"},
                **{f"validation_{k}": v for k, v in metrics(validation_part, race_col, label).items() if k != "segment"},
            }
        )

    out = pd.DataFrame(rows)
    out["validation_edge_score"] = out["validation_win_roi"] + 0.35 * out["validation_place_roi"] + 0.2 * out["validation_top3_rate"]
    out["roi_stability_win"] = out["validation_win_roi"] - out["discovery_win_roi"]
    out["roi_stability_place"] = out["validation_place_roi"] - out["discovery_place_roi"]
    out = out.sort_values(["validation_edge_score", "validation_win_roi", "validation_place_roi"], ascending=False)

    bucket_rows = []
    for group_cols in [
        ["age_bucket", "body_weight_bucket"],
        ["age_bucket", "maturity_size_bucket"],
        ["age_bucket", "surface_simple", "maturity_size_bucket"],
        ["age_bucket", "class_bucket", "maturity_size_bucket"],
    ]:
        for keys, part in scored.groupby(group_cols, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            label = "&".join(f"{col}={value}" for col, value in zip(group_cols, keys))
            bucket_rows.append({"group": "+".join(group_cols), "value": label, **metrics(part, race_col, label)})
    bucket_df = pd.DataFrame(bucket_rows)

    output_dir = ensure_dir(project_path(args.output_dir))
    out.to_csv(output_dir / "body_weight_age_segments.csv", index=False, encoding="utf-8-sig")
    out.head(100).to_csv(output_dir / "top_body_weight_age_segments.csv", index=False, encoding="utf-8-sig")
    bucket_df.to_csv(output_dir / "body_weight_age_bucket_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(output_dir),
        "rows": int(len(scored)),
        "races": int(scored[race_col].nunique()),
        "prev_body_weight_nonnull": int(scored["prev_body_weight_num"].notna().sum()),
        "top_segments": out.head(30).to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
