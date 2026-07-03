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


def _num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _z_by_race(values: pd.Series, race: pd.Series) -> pd.Series:
    v = _num(values).fillna(0.0)
    mean = v.groupby(race).transform("mean")
    std = v.groupby(race).transform("std").replace(0, np.nan)
    return ((v - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _id_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def _content_performance_score(df: pd.DataFrame, race_col: str, rank_col: str) -> pd.Series:
    idx = df.index
    target_score = _num(df.get("target_score"), idx, 0.0).fillna(0.0)
    target_top3 = _num(df.get("target_top3"), idx, 0.0).fillna(0.0)
    finish = _num(df.get(rank_col), idx, np.nan)
    field = _num(df.get("出走頭数", df.get("頭数")), idx, np.nan).fillna(df.groupby(race_col)[race_col].transform("size"))
    finish_score = ((field + 1.0 - finish) / field.replace(0, np.nan)).clip(0.0, 1.0).fillna(target_score)

    member = _num(df.get("race_member_depth_score"), idx, 0.0).fillna(0.0)
    time_value = _z_by_race(_num(df.get("prev_class_time_value_score"), idx, 0.0), df[race_col])
    bias_resist = _num(df.get("prev_retro_bias_resistant_score"), idx, 0.0).fillna(0.0)
    bias_excuse = _num(df.get("prev_retro_bias_excuse_score"), idx, 0.0).fillna(0.0)
    adversity = _num(df.get("prev_retro_bias_adversity_score"), idx, 0.0).fillna(0.0)
    overhelped = _num(df.get("prev_retro_bias_overhelped_score"), idx, 0.0).fillna(0.0)
    position_fit = _num(df.get("pace_fit_score"), idx, 0.0).fillna(0.0)
    corner_rate = _num(df.get("4角.1", df.get("4角")), idx, np.nan)
    frontish_bonus = (1.0 - (corner_rate / field.replace(0, np.nan))).clip(0.0, 1.0).fillna(0.0)

    score = (
        0.46 * finish_score
        + 0.18 * target_score
        + 0.10 * target_top3
        + 0.10 * member.clip(0.0, 1.0)
        + 0.06 * ((time_value + 2.0) / 4.0).clip(0.0, 1.0)
        + 0.06 * bias_resist.clip(0.0, 1.0)
        + 0.06 * bias_excuse.clip(0.0, 1.0)
        + 0.04 * adversity.clip(0.0, 1.0)
        + 0.03 * position_fit.clip(0.0, 1.0)
        + 0.02 * frontish_bonus
        - 0.10 * overhelped.clip(0.0, 1.0)
    )
    return score.clip(0.0, 1.25).fillna(0.0)


def add_content_bridge_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    race_col = config["data"]["race_id_column"]
    horse_col = config["data"]["horse_id_column"]
    date_col = config["data"]["date_column"]
    rank_col = config["data"]["rank_column"]
    prev_race_col = "前走レースID(新/馬番無)"

    out = frame.copy()
    out["_race_key"] = _id_key(out[race_col])
    out["_prev_race_key"] = _id_key(out[prev_race_col]) if prev_race_col in out.columns else pd.Series(pd.NA, index=out.index)
    out["_date_num"] = _num(out[date_col], out.index, np.nan)
    out["content_performance_score"] = _content_performance_score(out, race_col, rank_col)

    race_group = out.groupby("_race_key", sort=False)
    out["race_content_top_score"] = race_group["content_performance_score"].transform("max").fillna(0.0)
    out["race_content_top3_mean"] = (
        out["content_performance_score"]
        .where(out["content_performance_score"].groupby(out["_race_key"]).rank(ascending=False, method="first").le(3), 0.0)
        .groupby(out["_race_key"])
        .transform("sum")
        / out.groupby("_race_key")["_race_key"].transform(lambda s: min(3, len(s))).replace(0, np.nan)
    ).fillna(0.0)
    out["race_content_depth_score"] = (0.60 * out["race_content_top3_mean"] + 0.40 * race_group["content_performance_score"].transform("mean")).fillna(0.0)

    ordered = out.sort_values([horse_col, "_date_num", "_race_key"], kind="mergesort").copy()
    # A horse's own previous content lets us tell whether it was quietly strong
    # despite not necessarily winning.
    out.loc[ordered.index, "prev_content_performance_score"] = (
        ordered.groupby(horse_col, sort=False)["content_performance_score"].shift().fillna(0.0)
    )
    out.loc[ordered.index, "past3_content_performance_score"] = (
        ordered.groupby(horse_col, sort=False)["content_performance_score"]
        .transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
        .fillna(0.0)
    )

    # Build a chronological lookup: for each race, what did its members prove
    # afterwards before the current query date, using content score rather than
    # plain finishing position.
    member_next = ordered[["_race_key", horse_col, "_date_num", "content_performance_score", "target_top3", "target_win"]].copy()
    member_next["_next_date"] = member_next.groupby(horse_col, sort=False)["_date_num"].shift(-1)
    member_next["_next_content"] = member_next.groupby(horse_col, sort=False)["content_performance_score"].shift(-1)
    member_next["_next_top3"] = member_next.groupby(horse_col, sort=False)["target_top3"].shift(-1)
    member_next["_next_win"] = member_next.groupby(horse_col, sort=False)["target_win"].shift(-1)
    member_next = member_next[member_next["_next_date"].notna()].copy()

    next_by_race = {
        str(race_id): part.sort_values("_next_date", kind="mergesort")
        for race_id, part in member_next.groupby("_race_key", sort=False)
    }
    race_size = out.groupby("_race_key")[horse_col].count().to_dict()

    for col in [
        "prev_race_content_next_starters_count",
        "prev_race_content_next_starters_ratio",
        "prev_race_content_next_avg_score",
        "prev_race_content_next_top_score",
        "prev_race_content_next_top3_rate",
        "prev_race_content_next_win_rate",
    ]:
        out[col] = 0.0

    queries = out[out["_prev_race_key"].notna()].groupby("_prev_race_key").groups
    for prev_race_id, indices in queries.items():
        part = next_by_race.get(str(prev_race_id))
        if part is None or part.empty:
            continue
        dates = part["_next_date"].to_numpy(dtype=float)
        order = np.argsort(dates)
        dates = dates[order]
        content = _num(part["_next_content"]).fillna(0.0).to_numpy(dtype=float)[order]
        top3 = _num(part["_next_top3"]).fillna(0.0).to_numpy(dtype=float)[order]
        wins = _num(part["_next_win"]).fillna(0.0).to_numpy(dtype=float)[order]
        content_csum = np.cumsum(content)
        top3_csum = np.cumsum(top3)
        win_csum = np.cumsum(wins)
        content_cmax = np.maximum.accumulate(content)

        current_dates = out.loc[indices, "_date_num"].to_numpy(dtype=float)
        counts = np.searchsorted(dates, current_dates, side="left")
        valid = counts > 0
        avg_content = np.zeros(len(indices), dtype=float)
        top_content = np.zeros(len(indices), dtype=float)
        top3_rate = np.zeros(len(indices), dtype=float)
        win_rate = np.zeros(len(indices), dtype=float)
        avg_content[valid] = content_csum[counts[valid] - 1] / counts[valid]
        top_content[valid] = content_cmax[counts[valid] - 1]
        top3_rate[valid] = top3_csum[counts[valid] - 1] / counts[valid]
        win_rate[valid] = win_csum[counts[valid] - 1] / counts[valid]
        size = float(race_size.get(str(prev_race_id), np.nan))
        ratio = counts / size if size and np.isfinite(size) else np.zeros(len(indices), dtype=float)

        out.loc[indices, "prev_race_content_next_starters_count"] = counts.astype(float)
        out.loc[indices, "prev_race_content_next_starters_ratio"] = ratio.astype(float)
        out.loc[indices, "prev_race_content_next_avg_score"] = avg_content
        out.loc[indices, "prev_race_content_next_top_score"] = top_content
        out.loc[indices, "prev_race_content_next_top3_rate"] = top3_rate
        out.loc[indices, "prev_race_content_next_win_rate"] = win_rate

    out["prev_race_content_confirmed_strength"] = (
        0.45 * out["prev_race_content_next_avg_score"]
        + 0.25 * out["prev_race_content_next_top_score"]
        + 0.18 * out["prev_race_content_next_top3_rate"]
        + 0.12 * out["prev_race_content_next_win_rate"]
    ).fillna(0.0)
    out["prev_race_content_bridge_score"] = (
        out["prev_race_content_confirmed_strength"]
        * out["prev_race_content_next_starters_ratio"].clip(0.0, 1.0)
    ).fillna(0.0)
    out["prev_content_vs_bridge_score"] = (
        out["prev_content_performance_score"] * (0.65 + out["prev_race_content_bridge_score"])
    ).fillna(0.0)
    out["content_common_opponent_adjusted_score"] = (
        0.38 * _num(out.get("confirmed_member_level_adjusted_score"), out.index, 0.0).fillna(0.0)
        + 0.32 * out["prev_content_vs_bridge_score"]
        + 0.20 * out["past3_content_performance_score"]
        + 0.10 * out["prev_race_content_bridge_score"]
    ).fillna(0.0)

    return out.drop(columns=["_race_key", "_prev_race_key", "_date_num"], errors="ignore")


def _diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = _num(df.get("target_top3"), df.index, 0.0).fillna(0.0)
    win = _num(df.get("target_win"), df.index, 0.0).fillna(0.0)
    for col in [
        "content_common_opponent_adjusted_score",
        "prev_race_content_bridge_score",
        "prev_content_vs_bridge_score",
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
    parser = argparse.ArgumentParser(description="Add content-based common-opponent bridge member-level features.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument(
        "--train-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features_with_same_day_bias_v3_retro.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv",
    )
    parser.add_argument("--output-dir", default="outputs/analysis/content_bridge_member_features_v1")
    args = parser.parse_args()

    config = load_json_config(args.config)
    train = pd.read_csv(project_path(args.train_csv), low_memory=False)
    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    train["_split"] = "train"
    test["_split"] = "test"
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    enriched = add_content_bridge_features(combined, config)
    out_dir = ensure_dir(project_path(args.output_dir))

    train_out = enriched[enriched["_split"].eq("train")].drop(columns=["_split"], errors="ignore")
    test_out = enriched[enriched["_split"].eq("test")].drop(columns=["_split"], errors="ignore")
    train_path = out_dir / "train_features_with_content_bridge.csv"
    test_path = out_dir / "test_features_with_content_bridge.csv"
    train_out.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_out.to_csv(test_path, index=False, encoding="utf-8-sig")
    diag = _diagnostics(test_out)
    diag_path = out_dir / "content_bridge_diagnostics.csv"
    diag.to_csv(diag_path, index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(out_dir),
        "train_csv": str(train_path),
        "test_csv": str(test_path),
        "diagnostics_csv": str(diag_path),
        "new_features": [
            "content_performance_score",
            "race_content_depth_score",
            "prev_content_performance_score",
            "past3_content_performance_score",
            "prev_race_content_confirmed_strength",
            "prev_race_content_bridge_score",
            "prev_content_vs_bridge_score",
            "content_common_opponent_adjusted_score",
        ],
        "diagnostics": diag.to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
