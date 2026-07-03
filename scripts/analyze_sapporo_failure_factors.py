from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_sapporo_special_logic import add_sapporo_components, score_with_weights  # noqa: E402
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


DEFAULT_OUT = Path("outputs/analysis/sapporo_failure_factors")

SAPPORO_SURFACE_WEIGHTS = {
    "dirt": {
        "dirt1000_speed_fit": 0.03,
        "front_collapse_risk": 0.02,
        "member_form_fit": 0.03,
        "outer_loss_risk": 0.02,
    },
    "turf": {
        "bias_fit": 0.02,
    },
}


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def rank_scores(df: pd.DataFrame, score_col: str) -> pd.Series:
    return df.groupby(RACE_COL)[score_col].rank(ascending=False, method="first").astype(int)


def add_distance_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dist = num(out, "距離")
    out["sapporo_distance_group"] = np.select(
        [
            dist.le(1200),
            dist.eq(1500),
            dist.between(1700, 1800),
            dist.eq(2000),
            dist.ge(2400),
        ],
        ["<=1200", "1500", "1700-1800", "2000", "2400+"],
        default="other",
    )
    return out


def aggregate_role_metrics(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if len(frame) == 0:
        return {"role": label, "rows": 0}
    corner4 = num(frame, "4角.1", np.nan)
    prev_corner_rate = num(frame, "prev_corner4_position_rate", np.nan)
    front = num(frame, "horse_front_run_rate_past5", 0.0)
    stalker = num(frame, "horse_stalker_rate_past5", 0.0)
    closer = num(frame, "horse_closer_rate_past5", 0.0)
    odds = num(frame, "単勝オッズ", np.nan)
    popularity = num(frame, "人気", np.nan)
    win_pay = num(frame, "単勝配当", 0.0).where(frame["target_win"].eq(1), 0.0)
    place_pay = num(frame, "複勝配当", 0.0).where(frame["target_top3"].eq(1), 0.0)
    return {
        "role": label,
        "rows": int(len(frame)),
        "races": int(frame[RACE_COL].nunique()),
        "win_rate": float(frame["target_win"].mean()),
        "top3_rate": float(frame["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (len(frame) * 100.0)),
        "place_roi": float(place_pay.sum() / (len(frame) * 100.0)),
        "avg_actual_4c": float(corner4.mean()),
        "median_actual_4c": float(corner4.median()),
        "front_4c_share": float(corner4.le(4).mean()),
        "avg_prev_corner4_rate": float(prev_corner_rate.mean()),
        "avg_front_rate": float(front.mean()),
        "avg_stalker_rate": float(stalker.mean()),
        "avg_closer_rate": float(closer.mean()),
        "avg_popularity": float(popularity.mean()),
        "avg_odds": float(odds.mean()),
    }


def role_comparison(scored: pd.DataFrame, score_col: str, group_cols: list[str]) -> pd.DataFrame:
    out = scored.copy()
    out["rank"] = rank_scores(out, score_col)
    rows = []
    grouped = out.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), out)]
    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {col: key for col, key in zip(group_cols, keys)}
        roles = [
            ("ai_top1", part[part["rank"].eq(1)]),
            ("winner", part[part["target_win"].eq(1)]),
            ("top3", part[part["target_top3"].eq(1)]),
            ("ai_top1_miss", part[part["rank"].eq(1) & part["target_top3"].eq(0)]),
        ]
        for label, frame in roles:
            if len(frame) == 0:
                continue
            rows.append({**base, **aggregate_role_metrics(frame, label)})
    return pd.DataFrame(rows)


def missed_winner_pairs(scored: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = scored.copy()
    out["rank"] = rank_scores(out, score_col)
    ai = out[out["rank"].eq(1)][
        [
            RACE_COL,
            "馬名",
            "人気",
            "単勝オッズ",
            "4角.1",
            "prev_corner4_position_rate",
            "horse_front_run_rate_past5",
            "horse_closer_rate_past5",
            score_col,
        ]
    ].rename(
        columns={
            "馬名": "ai_horse",
            "人気": "ai_popularity",
            "単勝オッズ": "ai_odds",
            "4角.1": "ai_actual_4c",
            "prev_corner4_position_rate": "ai_prev_corner4_rate",
            "horse_front_run_rate_past5": "ai_front_rate",
            "horse_closer_rate_past5": "ai_closer_rate",
            score_col: "ai_score",
        }
    )
    winners = out[out["target_win"].eq(1)][
        [
            RACE_COL,
            "場所",
            "Ｒ",
            "レース名",
            "芝・ダ",
            "距離",
            "馬名",
            "人気",
            "単勝オッズ",
            "4角.1",
            "prev_corner4_position_rate",
            "horse_front_run_rate_past5",
            "horse_closer_rate_past5",
            score_col,
        ]
    ].rename(
        columns={
            "馬名": "winner_horse",
            "人気": "winner_popularity",
            "単勝オッズ": "winner_odds",
            "4角.1": "winner_actual_4c",
            "prev_corner4_position_rate": "winner_prev_corner4_rate",
            "horse_front_run_rate_past5": "winner_front_rate",
            "horse_closer_rate_past5": "winner_closer_rate",
            score_col: "winner_score",
        }
    )
    pairs = winners.merge(ai, on=RACE_COL, how="left")
    pairs = pairs[pairs["winner_horse"].ne(pairs["ai_horse"])].copy()
    pairs["winner_more_forward_actual_4c"] = num(pairs, "winner_actual_4c").lt(num(pairs, "ai_actual_4c"))
    pairs["winner_more_forward_prev_rate"] = num(pairs, "winner_prev_corner4_rate").lt(num(pairs, "ai_prev_corner4_rate"))
    pairs["winner_odds_over_ai"] = num(pairs, "winner_odds") - num(pairs, "ai_odds")
    return pairs


def segment_roi(scored: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = scored.copy()
    out["rank"] = rank_scores(out, score_col)
    corner4 = num(out, "4角.1", np.nan)
    prev_corner = num(out, "prev_corner4_position_rate", np.nan)
    front = num(out, "horse_front_run_rate_past5", 0.0)
    stalker = num(out, "horse_stalker_rate_past5", 0.0)
    closer = num(out, "horse_closer_rate_past5", 0.0)
    venue_fit = num(out, "sapporo_venue_specialist_fit", 0.0)
    turf_power = num(out, "sapporo_turf_power_stay_fit", 0.0)
    dirt1700 = num(out, "sapporo_dirt1700_position_fit", 0.0)
    bias = num(out, "sapporo_bias_fit", 0.0)
    body = num(out, "body_prev_weight", np.nan).fillna(num(out, "前走馬体重", np.nan))
    odds = num(out, "単勝オッズ", np.nan)
    popularity = num(out, "人気", np.nan)

    q = {
        "venue_hi": float(venue_fit.quantile(0.70)),
        "turf_power_hi": float(turf_power.quantile(0.70)),
        "dirt1700_hi": float(dirt1700.quantile(0.70)),
        "bias_hi": float(bias.quantile(0.70)),
    }
    checks: list[tuple[str, pd.Series]] = [
        ("top1_all", out["rank"].eq(1)),
        ("top1_actual_4c_le4", out["rank"].eq(1) & corner4.le(4)),
        ("top1_actual_4c_gt4", out["rank"].eq(1) & corner4.gt(4)),
        ("top1_front_or_stalker_hi", out["rank"].eq(1) & (front + stalker).ge(0.50)),
        ("top1_closer_hi", out["rank"].eq(1) & closer.ge(0.35)),
        ("top1_pop1_3", out["rank"].eq(1) & popularity.between(1, 3)),
        ("top1_pop4_6", out["rank"].eq(1) & popularity.between(4, 6)),
        ("top1_pop7plus", out["rank"].eq(1) & popularity.ge(7)),
        ("top1_odds_lt3", out["rank"].eq(1) & odds.lt(3.0)),
        ("top1_odds_3_7", out["rank"].eq(1) & odds.between(3.0, 7.0, inclusive="left")),
        ("top1_odds_7plus", out["rank"].eq(1) & odds.ge(7.0)),
        ("top1_venue_fit_hi", out["rank"].eq(1) & venue_fit.ge(q["venue_hi"])),
        ("top1_bias_fit_hi", out["rank"].eq(1) & bias.ge(q["bias_hi"])),
        ("top1_body_480plus", out["rank"].eq(1) & body.ge(480)),
        ("top1_body_500plus", out["rank"].eq(1) & body.ge(500)),
        ("top3_pop5plus", out["rank"].le(3) & popularity.ge(5)),
        ("top3_pop5plus_front_stalker", out["rank"].le(3) & popularity.ge(5) & (front + stalker).ge(0.50)),
        ("top3_pop5plus_venue_fit_hi", out["rank"].le(3) & popularity.ge(5) & venue_fit.ge(q["venue_hi"])),
        ("turf_top1_power_hi", out["rank"].eq(1) & out["surface_group"].eq("turf") & turf_power.ge(q["turf_power_hi"])),
        ("dirt_top1_1700_position_hi", out["rank"].eq(1) & out["surface_group"].eq("dirt") & dirt1700.ge(q["dirt1700_hi"])),
        ("dirt_top1_actual_4c_le4", out["rank"].eq(1) & out["surface_group"].eq("dirt") & corner4.le(4)),
        ("turf_top1_actual_4c_le4", out["rank"].eq(1) & out["surface_group"].eq("turf") & corner4.le(4)),
    ]
    rows = []
    for name, mask in checks:
        part = out[mask].copy()
        if len(part) == 0:
            continue
        win_pay = num(part, "単勝配当", 0.0).where(part["target_win"].eq(1), 0.0)
        place_pay = num(part, "複勝配当", 0.0).where(part["target_top3"].eq(1), 0.0)
        rows.append(
            {
                "segment": name,
                "bets": int(len(part)),
                "races": int(part[RACE_COL].nunique()),
                "win_rate": float(part["target_win"].mean()),
                "top3_rate": float(part["target_top3"].mean()),
                "win_roi": float(win_pay.sum() / (len(part) * 100.0)),
                "place_roi": float(place_pay.sum() / (len(part) * 100.0)),
                "avg_popularity": float(popularity.loc[part.index].mean()),
                "avg_odds": float(odds.loc[part.index].mean()),
                "avg_actual_4c": float(corner4.loc[part.index].mean()),
                "avg_front_stalker": float((front + stalker).loc[part.index].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["win_roi", "place_roi"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why Sapporo ROI is weak.")
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

    sapporo = test_x[test_x["場所"].astype(str).eq("札幌")].copy()
    sapporo["expected_lap_score"] = model.predict(sapporo)
    sapporo = add_sapporo_components(sapporo, "expected_lap_score")
    sapporo = add_distance_group(sapporo)
    sapporo["surface_group"] = np.where(sapporo["芝・ダ"].astype(str).str.contains("芝", na=False), "turf", "dirt")
    sapporo["sapporo_surface_score"] = sapporo["expected_lap_score"]
    for segment, weights in SAPPORO_SURFACE_WEIGHTS.items():
        mask = sapporo["surface_group"].eq(segment)
        sapporo.loc[mask, "sapporo_surface_score"] = score_with_weights(sapporo.loc[mask], "expected_lap_score", weights)

    overall = role_comparison(sapporo, "sapporo_surface_score", [])
    by_surface = role_comparison(sapporo, "sapporo_surface_score", ["surface_group"])
    by_distance = role_comparison(sapporo, "sapporo_surface_score", ["sapporo_distance_group"])
    pairs = missed_winner_pairs(sapporo, "sapporo_surface_score")
    segments = segment_roi(sapporo, "sapporo_surface_score")

    overall.to_csv(out_dir / "sapporo_role_comparison_overall.csv", index=False, encoding="utf-8-sig")
    by_surface.to_csv(out_dir / "sapporo_role_comparison_by_surface.csv", index=False, encoding="utf-8-sig")
    by_distance.to_csv(out_dir / "sapporo_role_comparison_by_distance.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(out_dir / "sapporo_missed_winner_pairs.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(out_dir / "sapporo_segment_roi.csv", index=False, encoding="utf-8-sig")

    pair_summary = {
        "missed_winner_pairs": int(len(pairs)),
        "winner_more_forward_actual_4c_rate": float(pairs["winner_more_forward_actual_4c"].mean()),
        "winner_more_forward_prev_rate": float(pairs["winner_more_forward_prev_rate"].mean()),
        "winner_avg_odds_minus_ai_avg_odds": float(pairs["winner_odds_over_ai"].mean()),
        "winner_avg_actual_4c": float(num(pairs, "winner_actual_4c").mean()),
        "ai_avg_actual_4c": float(num(pairs, "ai_actual_4c").mean()),
    }
    pd.DataFrame([pair_summary]).to_csv(out_dir / "sapporo_missed_winner_summary.csv", index=False, encoding="utf-8-sig")

    def show_pct(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        pct_cols = [c for c in out.columns if c.endswith("_rate") or c.endswith("_roi") or c.endswith("_share")]
        out[pct_cols] = out[pct_cols] * 100.0
        return out

    print("Overall roles")
    print(show_pct(overall).to_string(index=False))
    print("\nBy surface")
    print(show_pct(by_surface).to_string(index=False))
    print("\nMissed winner summary")
    print(pd.DataFrame([pair_summary]).to_string(index=False))
    print("\nSegments")
    show_segments = segments.copy()
    pct_cols = [c for c in show_segments.columns if c.endswith("_rate") or c.endswith("_roi")]
    show_segments[pct_cols] = show_segments[pct_cols] * 100.0
    print(show_segments.head(30).to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
