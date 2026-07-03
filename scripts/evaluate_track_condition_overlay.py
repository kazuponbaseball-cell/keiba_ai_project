from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ablate_feature_groups import metric_summary  # noqa: E402
from src.data.loaders import load_json_config  # noqa: E402
from src.features.baseline import add_track_condition_features  # noqa: E402
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


DEFAULT_TEST = "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv"
DEFAULT_MODEL = "models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl"


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def z_by_race(values: pd.Series, race: pd.Series) -> pd.Series:
    mean = values.groupby(race).transform("mean")
    std = values.groupby(race).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_overlay_components(df: pd.DataFrame, race_col: str) -> pd.DataFrame:
    out = df.copy()
    high_fit = _num(out, "race_high_cushion_flag") * _num(out, "horse_high_cushion_avg_score")
    low_fit = _num(out, "race_low_cushion_flag") * _num(out, "horse_low_cushion_avg_score")
    wet_fit = _num(out, "race_wet_moisture_flag") * _num(out, "horse_wet_moisture_avg_score")
    dry_fit = _num(out, "race_dry_moisture_flag") * _num(out, "horse_dry_moisture_avg_score")

    # Keep the overlay as a small within-race preference, not a new ability model.
    out["track_overlay_fit"] = z_by_race(0.45 * high_fit + 0.25 * low_fit + 0.20 * dry_fit + 0.10 * wet_fit, out[race_col])
    out["track_overlay_experience"] = z_by_race(
        (
            _num(out, "race_high_cushion_flag") * np.log1p(_num(out, "horse_high_cushion_starts"))
            + _num(out, "race_low_cushion_flag") * np.log1p(_num(out, "horse_low_cushion_starts"))
            + _num(out, "race_wet_moisture_flag") * np.log1p(_num(out, "horse_wet_moisture_starts"))
            + _num(out, "race_dry_moisture_flag") * np.log1p(_num(out, "horse_dry_moisture_starts"))
        ),
        out[race_col],
    )
    out["track_overlay_pop_value"] = z_by_race(
        _num(out, "race_high_cushion_flag") * _num(out, "horse_high_cushion_popularity_outperform_rate"),
        out[race_col],
    )
    out["track_overlay_context"] = z_by_race(
        0.25 * _num(out, "race_cushion_z_by_venue") - 0.15 * _num(out, "race_moisture_z_by_venue_surface"),
        out[race_col],
    )
    out["track_overlay_score"] = (
        0.60 * out["track_overlay_fit"]
        + 0.20 * out["track_overlay_experience"]
        + 0.15 * out["track_overlay_pop_value"]
        + 0.05 * out["track_overlay_context"]
    ).fillna(0.0)
    return out


def longshot_metrics(df: pd.DataFrame, race_col: str, rank_col: str) -> dict[str, Any]:
    scored = df.copy()
    scored["overlay_rank"] = scored.groupby(race_col)["overlay_score"].rank(ascending=False, method="first")
    popularity = _num(scored, "人気", default=np.nan)
    longshot = scored[(scored["overlay_rank"] <= 3) & (popularity >= 6)]
    if longshot.empty:
        return {
            "top3_longshot_bets": 0,
            "top3_longshot_top3_rate": 0.0,
            "top3_longshot_place_roi": 0.0,
        }
    place_pay = _num(longshot, "複勝配当").where(longshot["target_top3"].eq(1), 0.0)
    return {
        "top3_longshot_bets": int(len(longshot)),
        "top3_longshot_top3_rate": float(longshot["target_top3"].mean()),
        "top3_longshot_place_roi": float(place_pay.sum() / (len(longshot) * 100.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate track-condition features as a small score overlay.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--output-dir", default="outputs/analysis/track_condition_overlay")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    rank_col = config["data"]["rank_column"]

    test = pd.read_csv(args.test_csv, low_memory=False)
    test = add_track_condition_features(test, config)
    model: SimpleRaceRanker = pickle.load(Path(args.model).open("rb"))
    test["base_score"] = model.predict(test)
    test = build_overlay_components(test, race_col)

    rows = []
    for alpha in [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]:
        score = test["base_score"] + alpha * test["track_overlay_score"]
        metrics = metric_summary(test, score.to_numpy(), race_col, rank_col)
        tmp = test.copy()
        tmp["overlay_score"] = score
        rows.append({"alpha": alpha, **metrics, **longshot_metrics(tmp, race_col, rank_col)})

    summary = pd.DataFrame(rows)
    base = summary.loc[summary["alpha"].eq(0.0)].iloc[0]
    for col in [
        "top1_win_rate",
        "top1_top3_rate",
        "top1_win_roi",
        "top1_place_roi",
        "top3_contains_winner_rate",
        "top3_win_roi",
        "top3_place_roi",
        "winner_mean_ai_rank",
        "top3_longshot_top3_rate",
        "top3_longshot_place_roi",
    ]:
        summary[f"delta_{col}"] = summary[col] - base[col]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "overlay_summary.csv", index=False, encoding="utf-8-sig")
    test[
        [
            race_col,
            "馬名",
            "人気",
            "base_score",
            "track_overlay_score",
            "track_overlay_fit",
            "track_overlay_experience",
            "track_overlay_pop_value",
            "track_overlay_context",
            "race_cushion_value",
            "race_moisture_avg",
            "horse_high_cushion_avg_score",
            "horse_low_cushion_avg_score",
            "horse_wet_moisture_avg_score",
            "horse_dry_moisture_avg_score",
        ]
    ].to_csv(output_dir / "runner_overlay_components.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
