from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


LEFT_VENUES = {"東京", "中京", "新潟"}
RIGHT_VENUES = {"中山", "京都", "阪神", "札幌", "函館", "福島", "小倉"}
LOCAL_SMALL_VENUES = {"札幌", "函館", "福島", "小倉"}


def _num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _id_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def _text(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series("", index=index, dtype="string")
    return series.astype("string").fillna("").str.strip()


def _course_direction(venue: pd.Series) -> pd.Series:
    v = venue.astype("string").fillna("").str.strip()
    out = pd.Series("unknown", index=venue.index, dtype="string")
    out[v.isin(LEFT_VENUES)] = "left"
    out[v.isin(RIGHT_VENUES)] = "right"
    return out


def _rotation_bucket(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    interval = _num(df.get("rotation_interval_weeks"), idx, np.nan)
    starts_after_break = _num(df.get("休み明け～戦目"), idx, np.nan)
    out = pd.Series("unknown", index=idx, dtype="string")
    out[interval <= 2] = "short"
    out[(interval > 2) & (interval <= 8)] = "standard"
    out[(interval > 8) & (interval <= 16)] = "layoff_9_16w"
    out[interval > 16] = "layoff_17w_plus"
    out[starts_after_break.eq(1)] = "fresh"
    out[starts_after_break.eq(2)] = "second_after_layoff"
    out[starts_after_break.eq(3)] = "third_after_layoff"
    return out


def _prior_stats(
    ordered: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    group = ordered.groupby(group_cols, sort=False, dropna=False)[value_col]
    count = group.cumcount()
    csum = group.cumsum() - ordered[value_col]
    out = pd.DataFrame(index=ordered.index)
    out[f"{prefix}_starts"] = count.astype(float)
    out[f"{prefix}_avg_score"] = (csum / count.replace(0, np.nan)).fillna(0.0)
    return out


def _prior_binary_rate(
    ordered: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    group = ordered.groupby(group_cols, sort=False, dropna=False)[value_col]
    count = group.cumcount()
    csum = group.cumsum() - ordered[value_col]
    out = pd.DataFrame(index=ordered.index)
    out[f"{prefix}_top3_rate"] = (csum / count.replace(0, np.nan)).fillna(0.0)
    return out


def _prior_min(ordered: pd.DataFrame, group_col: str, value_col: str, out_col: str) -> pd.Series:
    values = _num(ordered.get(value_col), ordered.index, np.nan)
    sentinel = values.fillna(np.inf)
    prior = sentinel.groupby(ordered[group_col], sort=False).cummin().groupby(ordered[group_col], sort=False).shift()
    return prior.replace(np.inf, np.nan).rename(out_col)


def add_vertical_context_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    race_col = config["data"]["race_id_column"]
    horse_col = config["data"]["horse_id_column"]
    date_col = config["data"]["date_column"]
    venue_col = "場所"
    surface_col = "芝・ダ"
    distance_col = "距離"

    out = frame.copy()
    out["_race_key"] = _id_key(out[race_col])
    out["_horse_key"] = _id_key(out[horse_col])
    out["_date_num"] = _num(out[date_col], out.index, np.nan)
    out["_venue_key"] = _text(out.get(venue_col), out.index)
    out["_surface_key"] = _text(out.get(surface_col), out.index)
    out["_direction_key"] = _course_direction(out["_venue_key"])
    out["_is_local_small"] = out["_venue_key"].isin(LOCAL_SMALL_VENUES).astype(float)
    out["_distance_category_key"] = _text(out.get("distance_category"), out.index)
    out["_rotation_bucket_key"] = _rotation_bucket(out)
    out["_workout_lap_key"] = _text(out.get("workout_latest_lap_group"), out.index)
    out["_workout_pattern_key"] = _text(out.get("workout_latest_pattern_bucket"), out.index)
    out["_surface_distance_key"] = out["_surface_key"] + "_" + out["_distance_category_key"]

    out["_target_score_hist"] = _num(out.get("target_score"), out.index, 0.0).fillna(0.0)
    out["_target_top3_hist"] = _num(out.get("target_top3"), out.index, 0.0).fillna(0.0)

    ordered = out.sort_values(["_horse_key", "_date_num", "_race_key"], kind="mergesort").copy()

    # Horse baseline: the anchor for vertical comparison.
    baseline = _prior_stats(ordered, ["_horse_key"], "_target_score_hist", "horse_vertical_overall")
    baseline = baseline.join(_prior_binary_rate(ordered, ["_horse_key"], "_target_top3_hist", "horse_vertical_overall"))

    pieces = [baseline]
    contexts = [
        ("horse_direction", ["_horse_key", "_direction_key"]),
        ("horse_venue_vertical", ["_horse_key", "_venue_key"]),
        ("horse_surface_distance_vertical", ["_horse_key", "_surface_distance_key"]),
        ("horse_rotation_vertical", ["_horse_key", "_rotation_bucket_key"]),
        ("horse_workout_lap_vertical", ["_horse_key", "_workout_lap_key"]),
        ("horse_workout_pattern_vertical", ["_horse_key", "_workout_pattern_key"]),
    ]
    for prefix, cols in contexts:
        stats = _prior_stats(ordered, cols, "_target_score_hist", prefix)
        stats = stats.join(_prior_binary_rate(ordered, cols, "_target_top3_hist", prefix))
        pieces.append(stats)

    prior_best_total = _prior_min(ordered, "_horse_key", "workout_latest_total_time_sec", "horse_prior_best_workout_total_sec")
    prior_best_final = _prior_min(ordered, "_horse_key", "workout_latest_final_1f_sec", "horse_prior_best_workout_final1_sec")
    pieces.append(pd.DataFrame({prior_best_total.name: prior_best_total, prior_best_final.name: prior_best_final}, index=ordered.index))

    features = pd.concat(pieces, axis=1)
    out.loc[ordered.index, features.columns] = features

    overall = _num(out.get("horse_vertical_overall_avg_score"), out.index, 0.0).fillna(0.0)
    overall_starts = _num(out.get("horse_vertical_overall_starts"), out.index, 0.0).fillna(0.0)

    fit_specs = [
        ("horse_direction", 0.18),
        ("horse_venue_vertical", 0.16),
        ("horse_surface_distance_vertical", 0.22),
        ("horse_rotation_vertical", 0.18),
        ("horse_workout_lap_vertical", 0.13),
        ("horse_workout_pattern_vertical", 0.13),
    ]
    fit = pd.Series(0.0, index=out.index)
    mismatch = pd.Series(0.0, index=out.index)
    reliability = pd.Series(0.0, index=out.index)
    for prefix, weight in fit_specs:
        starts = _num(out.get(f"{prefix}_starts"), out.index, 0.0).fillna(0.0)
        avg = _num(out.get(f"{prefix}_avg_score"), out.index, 0.0).fillna(0.0)
        rel = (starts / (starts + 3.0)).clip(0.0, 1.0)
        delta = (avg - overall).fillna(0.0)
        fit += weight * rel * delta.clip(-0.5, 0.5)
        mismatch += weight * rel * (-delta).clip(0.0, 0.5)
        reliability += weight * rel

    out["vertical_condition_fit_score"] = fit.fillna(0.0)
    out["vertical_condition_mismatch_score"] = mismatch.fillna(0.0)
    out["vertical_condition_reliability_score"] = reliability.clip(0.0, 1.0).fillna(0.0)
    out["vertical_condition_positive_flag"] = ((out["vertical_condition_fit_score"] > 0.04) & (overall_starts >= 3)).astype(float)
    out["vertical_condition_negative_flag"] = ((out["vertical_condition_mismatch_score"] > 0.06) & (overall_starts >= 3)).astype(float)

    latest_total = _num(out.get("workout_latest_total_time_sec"), out.index, np.nan)
    latest_final = _num(out.get("workout_latest_final_1f_sec"), out.index, np.nan)
    out["workout_vs_horse_best_total_gap_sec"] = (latest_total - _num(out.get("horse_prior_best_workout_total_sec"), out.index, np.nan)).fillna(0.0)
    out["workout_vs_horse_best_final1_gap_sec"] = (latest_final - _num(out.get("horse_prior_best_workout_final1_sec"), out.index, np.nan)).fillna(0.0)
    out["workout_horse_regression_flag"] = (
        (out["workout_vs_horse_best_total_gap_sec"] > 2.0) & (out["workout_vs_horse_best_final1_gap_sec"] > 0.4)
    ).astype(float)

    out["layoff_vertical_fit_score"] = (
        _num(out.get("horse_rotation_vertical_avg_score"), out.index, 0.0).fillna(0.0) - overall
    ).fillna(0.0)
    out["direction_vertical_fit_score"] = (
        _num(out.get("horse_direction_avg_score"), out.index, 0.0).fillna(0.0) - overall
    ).fillna(0.0)
    out["course_vertical_fit_score"] = (
        _num(out.get("horse_venue_vertical_avg_score"), out.index, 0.0).fillna(0.0) - overall
    ).fillna(0.0)
    out["surface_distance_vertical_fit_score"] = (
        _num(out.get("horse_surface_distance_vertical_avg_score"), out.index, 0.0).fillna(0.0) - overall
    ).fillna(0.0)
    out["workout_vertical_fit_score"] = (
        0.55 * (_num(out.get("horse_workout_lap_vertical_avg_score"), out.index, 0.0).fillna(0.0) - overall)
        + 0.45 * (_num(out.get("horse_workout_pattern_vertical_avg_score"), out.index, 0.0).fillna(0.0) - overall)
    ).fillna(0.0)

    return out.drop(
        columns=[
            "_race_key",
            "_horse_key",
            "_date_num",
            "_venue_key",
            "_surface_key",
            "_direction_key",
            "_distance_category_key",
            "_rotation_bucket_key",
            "_workout_lap_key",
            "_workout_pattern_key",
            "_surface_distance_key",
            "_target_score_hist",
            "_target_top3_hist",
            "_is_local_small",
        ],
        errors="ignore",
    )


def _diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = _num(df.get("target_top3"), df.index, 0.0).fillna(0.0)
    win = _num(df.get("target_win"), df.index, 0.0).fillna(0.0)
    for col in [
        "vertical_condition_fit_score",
        "vertical_condition_mismatch_score",
        "layoff_vertical_fit_score",
        "direction_vertical_fit_score",
        "course_vertical_fit_score",
        "surface_distance_vertical_fit_score",
        "workout_vertical_fit_score",
    ]:
        if col not in df.columns:
            continue
        work = df[[col]].copy()
        work["target_top3"] = target
        work["target_win"] = win
        work["bin"] = pd.qcut(work[col], q=5, labels=False, duplicates="drop")
        part = (
            work.groupby("bin", dropna=False)
            .agg(rows=(col, "size"), avg_score=(col, "mean"), win_rate=("target_win", "mean"), top3_rate=("target_top3", "mean"))
            .reset_index()
        )
        part.insert(0, "feature", col)
        rows.append(part)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Add vertical horse-context fit features from prior races only.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias_content_bridge_safe.json")
    parser.add_argument("--train-csv", default="outputs/analysis/content_bridge_member_features_v1/train_features_with_content_bridge.csv")
    parser.add_argument("--test-csv", default="outputs/analysis/content_bridge_member_features_v1/test_features_with_content_bridge.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/vertical_horse_context_v1")
    args = parser.parse_args()

    config = load_json_config(args.config)
    train = pd.read_csv(project_path(args.train_csv), low_memory=False)
    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    train["_split"] = "train"
    test["_split"] = "test"
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    enriched = add_vertical_context_features(combined, config)
    out_dir = ensure_dir(project_path(args.output_dir))

    train_out = enriched[enriched["_split"].eq("train")].drop(columns=["_split"], errors="ignore")
    test_out = enriched[enriched["_split"].eq("test")].drop(columns=["_split"], errors="ignore")
    train_path = out_dir / "train_features_with_vertical_context.csv"
    test_path = out_dir / "test_features_with_vertical_context.csv"
    train_out.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_out.to_csv(test_path, index=False, encoding="utf-8-sig")

    diag = _diagnostics(test_out)
    diag_path = out_dir / "vertical_context_diagnostics.csv"
    diag.to_csv(diag_path, index=False, encoding="utf-8-sig")

    new_features = [
        c
        for c in train_out.columns
        if c.startswith("horse_vertical_")
        or c.startswith("horse_direction_")
        or c.startswith("horse_venue_vertical_")
        or c.startswith("horse_surface_distance_vertical_")
        or c.startswith("horse_rotation_vertical_")
        or c.startswith("horse_workout_lap_vertical_")
        or c.startswith("horse_workout_pattern_vertical_")
        or c
        in {
            "vertical_condition_fit_score",
            "vertical_condition_mismatch_score",
            "vertical_condition_reliability_score",
            "vertical_condition_positive_flag",
            "vertical_condition_negative_flag",
            "horse_prior_best_workout_total_sec",
            "horse_prior_best_workout_final1_sec",
            "workout_vs_horse_best_total_gap_sec",
            "workout_vs_horse_best_final1_gap_sec",
            "workout_horse_regression_flag",
            "layoff_vertical_fit_score",
            "direction_vertical_fit_score",
            "course_vertical_fit_score",
            "surface_distance_vertical_fit_score",
            "workout_vertical_fit_score",
        }
    ]
    payload = {
        "output_dir": str(out_dir),
        "train_csv": str(train_path),
        "test_csv": str(test_path),
        "diagnostics_csv": str(diag_path),
        "new_features": new_features,
        "diagnostics": diag.to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
