from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_expected_lap_rpci_features import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    NEW_CATEGORICAL_FEATURES,
    NEW_NUMERIC_FEATURES,
    RACE_COL,
    add_expected_lap_features,
    fit_ranker,
    num,
)
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


DEFAULT_OUT = Path("outputs/analysis/venue_roi_breakdown")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def z(values: pd.Series, race: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    mean = values.groupby(race).transform("mean")
    std = values.groupby(race).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def summarize_venue(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scored = scored.copy()
    scored["ai_rank"] = scored.groupby(RACE_COL)["expected_lap_score"].rank(ascending=False, method="first").astype(int)
    scored["pop_rank"] = num(scored, "人気").groupby(scored[RACE_COL]).rank(ascending=True, method="first")
    scored["market_gap"] = scored["pop_rank"] - scored["ai_rank"]
    scored["fit_z"] = z(scored["expected_lap_total_fit_score"], scored[RACE_COL])
    scored["rpci_fit_z"] = z(scored["horse_expected_rpci_fit_score"], scored[RACE_COL])
    scored["bias_fit_z"] = z(num(scored, "same_day_bias_fit_score", 0.0), scored[RACE_COL])
    scored["draw_pace_z"] = z(num(scored, "draw_pace_fit_score", 0.0), scored[RACE_COL])
    scored["member_z"] = z(num(scored, "confirmed_member_level_adjusted_score", 0.0), scored[RACE_COL])

    top1 = scored[scored["ai_rank"].eq(1)].copy()
    for venue, part in top1.groupby("場所", dropna=False, sort=True):
        all_part = scored[scored["場所"].eq(venue)]
        races = int(part[RACE_COL].nunique())
        win = part[part["target_win"].eq(1)]
        place = part[part["target_top3"].eq(1)]
        miss = part[part["target_top3"].eq(0)]
        win_pay = num(part, "単勝配当", 0.0).where(part["target_win"].eq(1), 0.0)
        place_pay = num(part, "複勝配当", 0.0).where(part["target_top3"].eq(1), 0.0)
        favorite = part[num(part, "人気").le(1)]
        pop3 = part[num(part, "人気").le(3)]
        pop5plus = part[num(part, "人気").ge(5)]
        row = {
            "場所": venue,
            "races": races,
            "top1_win_rate": part["target_win"].mean(),
            "top1_top3_rate": part["target_top3"].mean(),
            "top1_win_roi": win_pay.sum() / (len(part) * 100.0),
            "top1_place_roi": place_pay.sum() / (len(part) * 100.0),
            "avg_field_size": num(part, "頭数").mean(),
            "avg_popularity": num(part, "人気").mean(),
            "avg_odds": num(part, "単勝オッズ").mean(),
            "top1_favorite_share": len(favorite) / len(part),
            "top1_pop3_share": len(pop3) / len(part),
            "top1_pop5plus_share": len(pop5plus) / len(part),
            "favorite_win_rate_when_ai_top1": favorite["target_win"].mean() if len(favorite) else np.nan,
            "favorite_top3_rate_when_ai_top1": favorite["target_top3"].mean() if len(favorite) else np.nan,
            "pop5plus_win_rate_when_ai_top1": pop5plus["target_win"].mean() if len(pop5plus) else np.nan,
            "pop5plus_top3_rate_when_ai_top1": pop5plus["target_top3"].mean() if len(pop5plus) else np.nan,
            "avg_market_gap": part["market_gap"].mean(),
            "win_avg_odds": num(win, "単勝オッズ").mean(),
            "miss_avg_odds": num(miss, "単勝オッズ").mean(),
            "avg_expected_rpci": part["expected_rpci"].mean(),
            "fast_expected_share": part["expected_lap_type"].eq("fast").mean(),
            "middle_expected_share": part["expected_lap_type"].eq("middle").mean(),
            "slow_expected_share": part["expected_lap_type"].eq("slow").mean(),
            "avg_lap_fit_z": part["fit_z"].mean(),
            "avg_rpci_fit_z": part["rpci_fit_z"].mean(),
            "avg_bias_fit_z": part["bias_fit_z"].mean(),
            "avg_draw_pace_z": part["draw_pace_z"].mean(),
            "avg_member_z": part["member_z"].mean(),
            "all_avg_pace_collapse": num(all_part, "race_pace_collapse_risk", 0.0).mean(),
            "all_avg_slow_pace": num(all_part, "race_slow_pace_risk", 0.0).mean(),
            "all_avg_same_day_ready": num(all_part, "same_day_bias_ready", 0.0).mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("top1_win_roi", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze venue ROI drivers.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(project_path(args.train_csv), low_memory=False)
    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    base_model: SimpleRaceRanker = pickle.load(project_path(args.base_model).open("rb"))
    train_x, test_x, _ = add_expected_lap_features(train, test)

    plus_numeric = list(base_model.numeric_features) + [
        col for col in NEW_NUMERIC_FEATURES if col not in base_model.numeric_features
    ]
    plus_categorical = list(base_model.categorical_features) + [
        col for col in NEW_CATEGORICAL_FEATURES if col not in base_model.categorical_features
    ]
    model = fit_ranker(train_x, plus_numeric, plus_categorical, float(base_model.ridge_alpha), int(base_model.categorical_top_k))
    scored = test_x.copy()
    scored["expected_lap_score"] = model.predict(test_x)
    drivers = summarize_venue(scored)
    drivers.to_csv(out_dir / "venue_roi_drivers.csv", index=False, encoding="utf-8-sig")

    show = drivers.copy()
    pct_cols = [c for c in show.columns if c.endswith("_rate") or c.endswith("_roi") or c.endswith("_share")]
    show[pct_cols] = show[pct_cols] * 100.0
    cols = [
        "場所",
        "races",
        "top1_win_rate",
        "top1_top3_rate",
        "top1_win_roi",
        "top1_place_roi",
        "avg_popularity",
        "avg_odds",
        "top1_favorite_share",
        "top1_pop5plus_share",
        "favorite_win_rate_when_ai_top1",
        "pop5plus_win_rate_when_ai_top1",
        "avg_market_gap",
        "win_avg_odds",
        "fast_expected_share",
        "slow_expected_share",
        "avg_lap_fit_z",
        "avg_rpci_fit_z",
        "avg_bias_fit_z",
        "all_avg_same_day_ready",
    ]
    print(show[cols].to_string(index=False))
    print(f"\nOutput: {out_dir / 'venue_roi_drivers.csv'}")


if __name__ == "__main__":
    main()
