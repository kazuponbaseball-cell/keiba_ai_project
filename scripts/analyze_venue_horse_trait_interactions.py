from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Condition:
    name: str
    func: MaskFunc


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    series = df[col]
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _q(df: pd.DataFrame, col: str, q: float, default: float) -> float:
    values = _num(df, col).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return default
    return float(values.quantile(q))


def _pay(df: pd.DataFrame, col: str) -> pd.Series:
    return _num(df, col, 0.0).fillna(0.0)


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
        "avg_ai_rank": float(df["ai_rank"].mean()),
    }


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


def temporal_split(df: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_numeric(df[date_col], errors="coerce")
    unique_dates = np.array(sorted(dates.dropna().unique()))
    cutoff = unique_dates[len(unique_dates) // 2]
    return df[dates <= cutoff].copy(), df[dates > cutoff].copy()


def build_conditions(df: pd.DataFrame) -> list[Condition]:
    front = pd.concat(
        [
            _num(df, "front_running_tendency").fillna(0.0),
            _num(df, "horse_front_run_rate_past5").fillna(0.0),
            _num(df, "horse_stalker_rate_past5").fillna(0.0) * 0.75,
        ],
        axis=1,
    ).max(axis=1)
    front_hi = float(front.quantile(0.70))
    front_top = float(front.quantile(0.85))

    local_tight = {"福島", "小倉", "函館", "札幌"}
    tight_plus_nakayama = {"福島", "小倉", "函館", "札幌", "中山"}
    central_big = {"東京", "阪神", "京都", "中山"}

    conditions = [
        Condition("ai_top1", lambda x: x["ai_rank"].eq(1)),
        Condition("ai_top3", lambda x: x["ai_rank"].le(3)),
        Condition("ai_top5", lambda x: x["ai_rank"].le(5)),
        Condition("pop5plus", lambda x: _num(x, "人気").ge(5)),
        Condition("pop7plus", lambda x: _num(x, "人気").ge(7)),
        Condition("odds10plus", lambda x: _num(x, "単勝オッズ").ge(10.0)),
        Condition("market_gap3plus", lambda x: x["ai_pop_gap"].le(-3)),
        Condition("local_tight", lambda x: x["場所"].isin(local_tight)),
        Condition("tight_plus_nakayama", lambda x: x["場所"].isin(tight_plus_nakayama)),
        Condition("central_big4", lambda x: x["場所"].isin(central_big)),
        Condition("tokyo", lambda x: x["場所"].eq("東京")),
        Condition("fukushima", lambda x: x["場所"].eq("福島")),
        Condition("kokura", lambda x: x["場所"].eq("小倉")),
        Condition("hakodate_sapporo", lambda x: x["場所"].isin({"函館", "札幌"})),
        Condition("turf", lambda x: x["芝・ダ"].astype(str).str.contains("芝", na=False)),
        Condition("dirt", lambda x: x["芝・ダ"].astype(str).str.contains("ダ", na=False)),
        Condition("inner_frame", lambda x: _num(x, "枠番").le(3)),
        Condition("middle_frame", lambda x: _num(x, "枠番").between(4, 6, inclusive="both")),
        Condition("outer_frame", lambda x: _num(x, "枠番").ge(7)),
        Condition("inner_horse_no", lambda x: _num(x, "馬番").le(5)),
        Condition("big_500", lambda x: _num(x, "前走馬体重").ge(500)),
        Condition("big_520", lambda x: _num(x, "前走馬体重").ge(520)),
        Condition("small_460", lambda x: _num(x, "前走馬体重").le(460)),
        Condition("front_hi", lambda x, t=front_hi: pd.concat(
            [
                _num(x, "front_running_tendency").fillna(0.0),
                _num(x, "horse_front_run_rate_past5").fillna(0.0),
                _num(x, "horse_stalker_rate_past5").fillna(0.0) * 0.75,
            ],
            axis=1,
        ).max(axis=1).ge(t)),
        Condition("front_top", lambda x, t=front_top: pd.concat(
            [
                _num(x, "front_running_tendency").fillna(0.0),
                _num(x, "horse_front_run_rate_past5").fillna(0.0),
                _num(x, "horse_stalker_rate_past5").fillna(0.0) * 0.75,
            ],
            axis=1,
        ).max(axis=1).ge(t)),
    ]

    high_cols = [
        ("draw_pace_hi", "draw_pace_fit_score", 0.70),
        ("draw_adv_hi", "current_draw_advantage_score", 0.70),
        ("pace_fit_hi", "pace_fit_score", 0.70),
        ("lap_fit_hi", "lap_aptitude_fit_score", 0.70),
        ("blood_hi", "bloodline_high_confidence_fit_score", 0.70),
        ("venue_fit_hi", "same_venue_avg_score", 0.70),
        ("bias_recent_hi", "bias_adjusted_recent_score", 0.70),
        ("same_day_bias_hi", "same_day_bias_fit_score", 0.60),
    ]
    for name, col, quantile in high_cols:
        if col in df.columns:
            threshold = _q(df, col, quantile, 0.0)
            conditions.append(Condition(name, lambda x, c=col, t=threshold: _num(x, c).ge(t)))

    return conditions


def mask_for(df: pd.DataFrame, conds: tuple[Condition, ...]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for cond in conds:
        mask &= cond.func(df).fillna(False).astype(bool)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze venue x body size x draw x running-style interactions.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--output-dir", default="outputs/analysis/venue_horse_trait_interactions")
    parser.add_argument("--min-discovery-bets", type=int, default=80)
    parser.add_argument("--min-validation-bets", type=int, default=40)
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    date_col = config["data"]["date_column"]

    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = add_scores(df, project_path(args.model), race_col)
    discovery, validation = temporal_split(scored, date_col)
    conditions = build_conditions(scored)
    by_name = {cond.name: cond for cond in conditions}

    required_patterns = [
        ("local_big_inner_front_top3", ["ai_top3", "local_tight", "big_500", "inner_frame", "front_hi"]),
        ("local_big_inner_front_top5", ["ai_top5", "local_tight", "big_500", "inner_frame", "front_hi"]),
        ("tight_big_inner_front_top3", ["ai_top3", "tight_plus_nakayama", "big_500", "inner_frame", "front_hi"]),
        ("local_big_inner_draw_front_top5", ["ai_top5", "local_tight", "big_500", "inner_frame", "front_hi", "draw_pace_hi"]),
        ("local_big_outer_front_top5", ["ai_top5", "local_tight", "big_500", "outer_frame", "front_hi"]),
        ("local_small_inner_front_top5", ["ai_top5", "local_tight", "small_460", "inner_frame", "front_hi"]),
        ("tokyo_big_outer_ability_top5", ["ai_top5", "tokyo", "big_500", "outer_frame", "lap_fit_hi"]),
        ("fukushima_big_inner_front_top5", ["ai_top5", "fukushima", "big_500", "inner_frame", "front_hi"]),
        ("kokura_big_inner_front_top5", ["ai_top5", "kokura", "big_500", "inner_frame", "front_hi"]),
    ]

    rows: list[dict[str, object]] = []
    for label, names in required_patterns:
        conds = tuple(by_name[name] for name in names if name in by_name)
        d = discovery[mask_for(discovery, conds)]
        v = validation[mask_for(validation, conds)]
        a = scored[mask_for(scored, conds)]
        rows.append(
            {
                "source": "hypothesis",
                "segment": label,
                "conditions": "&".join(names),
                **{f"all_{k}": val for k, val in metrics(a, race_col, label).items() if k != "segment"},
                **{f"discovery_{k}": val for k, val in metrics(d, race_col, label).items() if k != "segment"},
                **{f"validation_{k}": val for k, val in metrics(v, race_col, label).items() if k != "segment"},
            }
        )

    search_names = [
        "ai_top1",
        "ai_top3",
        "ai_top5",
        "pop5plus",
        "odds10plus",
        "market_gap3plus",
        "local_tight",
        "tight_plus_nakayama",
        "tokyo",
        "fukushima",
        "kokura",
        "hakodate_sapporo",
        "turf",
        "dirt",
        "inner_frame",
        "outer_frame",
        "big_500",
        "big_520",
        "small_460",
        "front_hi",
        "front_top",
        "draw_pace_hi",
        "draw_adv_hi",
        "pace_fit_hi",
        "lap_fit_hi",
        "blood_hi",
        "venue_fit_hi",
        "same_day_bias_hi",
    ]
    search_conditions = [by_name[name] for name in search_names if name in by_name]
    must_have = {"local_tight", "tight_plus_nakayama", "tokyo", "fukushima", "kokura", "hakodate_sapporo"}
    interaction_markers = {"big_500", "big_520", "small_460", "inner_frame", "outer_frame", "front_hi", "front_top"}

    for size in range(3, 6):
        for combo in combinations(search_conditions, size):
            names = {c.name for c in combo}
            if not names & must_have:
                continue
            if len(names & interaction_markers) < 2:
                continue
            if not any(name.startswith("ai_top") for name in names):
                continue
            label = "&".join(c.name for c in combo)
            d = discovery[mask_for(discovery, combo)]
            if len(d) < args.min_discovery_bets:
                continue
            v = validation[mask_for(validation, combo)]
            if len(v) < args.min_validation_bets:
                continue
            a = scored[mask_for(scored, combo)]
            dm = metrics(d, race_col, label)
            vm = metrics(v, race_col, label)
            am = metrics(a, race_col, label)
            rows.append(
                {
                    "source": "search",
                    "segment": label,
                    "conditions": label,
                    **{f"all_{k}": val for k, val in am.items() if k != "segment"},
                    **{f"discovery_{k}": val for k, val in dm.items() if k != "segment"},
                    **{f"validation_{k}": val for k, val in vm.items() if k != "segment"},
                    "validation_edge_score": float(vm["win_roi"]) + 0.35 * float(vm["place_roi"]) + 0.2 * float(vm["top3_rate"]),
                }
            )

    out = pd.DataFrame(rows)
    if "validation_edge_score" not in out.columns:
        out["validation_edge_score"] = out["validation_win_roi"] + 0.35 * out["validation_place_roi"] + 0.2 * out["validation_top3_rate"]
    out["roi_stability_win"] = out["validation_win_roi"] - out["discovery_win_roi"]
    out["roi_stability_place"] = out["validation_place_roi"] - out["discovery_place_roi"]
    out = out.sort_values(["validation_edge_score", "validation_win_roi", "validation_place_roi"], ascending=[False, False, False])

    output_dir = ensure_dir(project_path(args.output_dir))
    out.to_csv(output_dir / "interaction_segments.csv", index=False, encoding="utf-8-sig")
    out[out["source"].eq("hypothesis")].to_csv(output_dir / "hypothesis_segments.csv", index=False, encoding="utf-8-sig")
    out.head(80).to_csv(output_dir / "top_interaction_segments.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(output_dir),
        "rows": int(len(scored)),
        "races": int(scored[race_col].nunique()),
        "top_hypotheses": out[out["source"].eq("hypothesis")].to_dict(orient="records"),
        "top_search": out[out["source"].eq("search")].head(20).to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
