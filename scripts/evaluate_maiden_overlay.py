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
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402
from src.utils.paths import ensure_dir, project_path  # noqa: E402


DEFAULT_TEST = "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv"
DEFAULT_MODEL = "models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl"


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    series = frame[col]
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _z(values: pd.Series, race: pd.Series) -> pd.Series:
    mean = values.groupby(race).transform("mean")
    std = values.groupby(race).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _payoff_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _flat_metrics(df: pd.DataFrame, race_col: str, label: str) -> dict[str, Any]:
    rows = len(df)
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
            "avg_base_rank": None,
        }
    win_col = _payoff_col(df, ["単勝配当", "蜊伜享驟榊ｽ・", "陷贋ｼ應ｺｫ鬩滓ｦ奇ｽｽ繝ｻ"])
    place_col = _payoff_col(df, ["複勝配当", "隍・享驟榊ｽ・", "髫阪・莠ｫ鬩滓ｦ奇ｽｽ繝ｻ"])
    win_pay = _num(df, win_col or "__missing__").where(df["target_win"].eq(1), 0.0)
    place_pay = _num(df, place_col or "__missing__").where(df["target_top3"].eq(1), 0.0)
    return {
        "label": label,
        "bets": int(rows),
        "races": int(df[race_col].nunique()),
        "win_rate": float(df["target_win"].mean()),
        "top3_rate": float(df["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (rows * 100.0)),
        "place_roi": float(place_pay.sum() / (rows * 100.0)),
        "avg_popularity": float(_num(df, "人気", np.nan).mean()),
        "avg_odds": float(_num(df, "単勝オッズ", np.nan).mean()),
        "avg_base_rank": float(_num(df, "base_rank", np.nan).mean()),
    }


def _add_components(df: pd.DataFrame, race_col: str) -> pd.DataFrame:
    out = df.copy()
    out["base_rank"] = out.groupby(race_col)["base_score"].rank(ascending=False, method="first").astype(int)
    out["popularity_num"] = _num(out, "人気", np.nan)
    out["odds_decimal"] = _num(out, "単勝オッズ", np.nan)
    out["pop_rank"] = out.groupby(race_col)["popularity_num"].rank(ascending=True, method="first")

    ability_raw = (
        0.22 * _z(_num(out, "prev_race_member_level"), out[race_col])
        + 0.18 * _z(_num(out, "prev_class_time_value_score"), out[race_col])
        + 0.12 * _z(_num(out, "prev_race_time_value"), out[race_col])
        - 0.12 * _z(_num(out, "prev_performance_vs_member_level"), out[race_col])
        - 0.10 * _z(_num(out, "workout_latest_total_vs_course_z"), out[race_col])
        - 0.08 * _z(_num(out, "workout_latest_finish_gain_sec"), out[race_col])
        + 0.08 * _z(_num(out, "lap_aptitude_fit_score"), out[race_col])
        + 0.06 * _z(_num(out, "pace_fit_score"), out[race_col])
        + 0.06 * _z(_num(out, "bloodline_high_confidence_fit_score"), out[race_col])
        + 0.05 * _z(_num(out, "same_day_bias_fit_score"), out[race_col])
        + 0.03 * _z(_num(out, "same_day_pop_adjusted_pace_fit_score"), out[race_col])
    )
    out["maiden_ability_overlay"] = _z(ability_raw, out[race_col])

    market_gap = (out["pop_rank"] - out["base_rank"]).fillna(0.0)
    supported_longshot = out["base_rank"].le(5).astype(float) * out["popularity_num"].ge(6).fillna(False).astype(float)
    value_raw = (
        0.55 * _z(market_gap, out[race_col])
        + 0.30 * _z(np.log1p(out["odds_decimal"].fillna(0.0)) * out["base_rank"].le(5), out[race_col])
        + 0.15 * _z(supported_longshot, out[race_col])
    )
    out["maiden_value_overlay"] = _z(value_raw, out[race_col])

    favorite_raw = (
        out["base_rank"].eq(1).astype(float)
        * (
            out["popularity_num"].le(1).fillna(False).astype(float)
            + out["odds_decimal"].lt(2.0).fillna(False).astype(float)
        )
    )
    out["maiden_favorite_penalty"] = _z(favorite_raw, out[race_col]).clip(lower=0.0)
    return out


def _temporal_split(df: pd.DataFrame, date_col: str) -> tuple[pd.Series, pd.Series]:
    dates = pd.to_numeric(df[date_col], errors="coerce")
    unique_dates = np.array(sorted(dates.dropna().unique()))
    cutoff = unique_dates[len(unique_dates) // 2]
    return dates <= cutoff, dates > cutoff


def _with_overlay_score(df: pd.DataFrame, ability_w: float, value_w: float, favorite_w: float) -> pd.Series:
    return (
        df["base_score"]
        + ability_w * df["maiden_ability_overlay"]
        + value_w * df["maiden_value_overlay"]
        - favorite_w * df["maiden_favorite_penalty"]
    )


def _top_sets(df: pd.DataFrame, score_col: str, race_col: str) -> dict[str, pd.DataFrame]:
    ranked = df.copy()
    ranked["overlay_rank"] = ranked.groupby(race_col)[score_col].rank(ascending=False, method="first").astype(int)
    return {
        "top1": ranked[ranked["overlay_rank"].eq(1)].copy(),
        "top3": ranked[ranked["overlay_rank"].le(3)].copy(),
        "top3_pop6plus": ranked[ranked["overlay_rank"].le(3) & ranked["popularity_num"].ge(6)].copy(),
        "top3_pop10plus": ranked[ranked["overlay_rank"].le(3) & ranked["popularity_num"].ge(10)].copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a maiden-only score overlay.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--output-dir", default="outputs/analysis/maiden_overlay")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    date_col = config["data"]["date_column"]
    rank_col = config["data"]["rank_column"]

    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    model: SimpleRaceRanker = pickle.load(project_path(args.model).open("rb"))
    test["base_score"] = model.predict(test)
    maiden = test[test["クラス名"].astype(str).str.contains("未勝利", na=False)].copy()
    maiden = _add_components(maiden, race_col)

    discovery_mask, validation_mask = _temporal_split(maiden, date_col)
    weights = [0.0, 0.01, 0.03, 0.05]
    rows: list[dict[str, Any]] = []
    for ability_w in weights:
        for value_w in weights:
            for favorite_w in [0.0, 0.01, 0.03]:
                label = f"a{ability_w:.3f}_v{value_w:.3f}_f{favorite_w:.3f}"
                maiden["overlay_score"] = _with_overlay_score(maiden, ability_w, value_w, favorite_w)
                all_metrics = metric_summary(maiden, maiden["overlay_score"].to_numpy(), race_col, rank_col)
                disc_metrics = metric_summary(maiden[discovery_mask], maiden.loc[discovery_mask, "overlay_score"].to_numpy(), race_col, rank_col)
                valid_metrics = metric_summary(maiden[validation_mask], maiden.loc[validation_mask, "overlay_score"].to_numpy(), race_col, rank_col)
                top_sets = _top_sets(maiden, "overlay_score", race_col)
                rows.append(
                    {
                        "label": label,
                        "ability_w": ability_w,
                        "value_w": value_w,
                        "favorite_w": favorite_w,
                        **{f"all_{k}": v for k, v in all_metrics.items()},
                        **{f"discovery_{k}": v for k, v in disc_metrics.items()},
                        **{f"validation_{k}": v for k, v in valid_metrics.items()},
                        **{f"{name}_{k}": v for name, part in top_sets.items() for k, v in _flat_metrics(part, race_col, name).items() if k != "label"},
                    }
                )

    summary = pd.DataFrame(rows)
    base = summary[(summary["ability_w"].eq(0.0)) & (summary["value_w"].eq(0.0)) & (summary["favorite_w"].eq(0.0))].iloc[0]
    for col in [
        "all_top1_win_rate",
        "all_top1_top3_rate",
        "all_top1_win_roi",
        "all_top1_place_roi",
        "all_top3_contains_winner_rate",
        "all_top3_win_roi",
        "all_top3_place_roi",
        "validation_top1_win_roi",
        "validation_top1_place_roi",
    ]:
        summary[f"delta_{col}"] = summary[col] - base[col]

    summary["selection_score"] = (
        summary["discovery_top1_win_roi"] * 1.0
        + summary["discovery_top1_place_roi"] * 0.4
        + summary["discovery_top1_win_rate"] * 0.4
        - summary["discovery_winner_mean_ai_rank"] * 0.02
    )
    ranked = summary.sort_values(
        ["selection_score", "validation_top1_win_roi", "all_top1_win_roi"],
        ascending=[False, False, False],
    )

    output_dir = ensure_dir(project_path(args.output_dir))
    summary.to_csv(output_dir / "maiden_overlay_grid.csv", index=False, encoding="utf-8-sig")
    ranked.head(50).to_csv(output_dir / "maiden_overlay_top50.csv", index=False, encoding="utf-8-sig")

    best = ranked.iloc[0]
    maiden["overlay_score"] = _with_overlay_score(maiden, float(best["ability_w"]), float(best["value_w"]), float(best["favorite_w"]))
    out = maiden.copy()
    out["overlay_rank"] = out.groupby(race_col)["overlay_score"].rank(ascending=False, method="first").astype(int)
    out["base_rank"] = out.groupby(race_col)["base_score"].rank(ascending=False, method="first").astype(int)
    keep = [
        race_col,
        "日付",
        "場所",
        "Ｒ",
        "レース名",
        "クラス名",
        "馬名",
        "確定着順",
        "人気",
        "単勝オッズ",
        "単勝配当",
        "複勝配当",
        "base_score",
        "base_rank",
        "overlay_score",
        "overlay_rank",
        "maiden_ability_overlay",
        "maiden_value_overlay",
        "maiden_favorite_penalty",
        "prev_race_member_level",
        "prev_class_time_value_score",
        "prev_race_time_value",
        "workout_latest_total_vs_course_z",
        "workout_latest_finish_gain_sec",
        "lap_aptitude_fit_score",
        "pace_fit_score",
        "bloodline_high_confidence_fit_score",
        "same_day_bias_fit_score",
    ]
    out[[c for c in keep if c in out.columns]].sort_values([race_col, "overlay_rank"]).to_csv(
        output_dir / "maiden_overlay_runner_scores.csv", index=False, encoding="utf-8-sig"
    )

    result = {
        "output_dir": str(output_dir),
        "maiden_rows": int(len(maiden)),
        "maiden_races": int(maiden[race_col].nunique()),
        "base": base.to_dict(),
        "best": best.to_dict(),
        "top10": ranked.head(10)[
            [
                "label",
                "ability_w",
                "value_w",
                "favorite_w",
                "all_top1_win_rate",
                "all_top1_top3_rate",
                "all_top1_win_roi",
                "all_top1_place_roi",
                "validation_top1_win_roi",
                "validation_top1_place_roi",
                "top3_pop6plus_bets",
                "top3_pop6plus_win_roi",
                "top3_pop6plus_place_roi",
                "top3_pop10plus_bets",
                "top3_pop10plus_win_roi",
                "top3_pop10plus_place_roi",
            ]
        ].to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
