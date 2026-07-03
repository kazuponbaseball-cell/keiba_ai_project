from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_kyoto_special_logic import (  # noqa: E402
    add_kyoto_components,
    add_kyoto_layout_group,
    score_with_layout_weights,
)
from scripts.analyze_sapporo_special_logic import add_sapporo_components, score_with_weights as score_sapporo_weights  # noqa: E402
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
from scripts.evaluate_layout_aware_global_policy import (  # noqa: E402
    INNER_HEAD_WEIGHTS,
    OUTER_PLACE_WEIGHTS,
    policy_metrics,
    simple_base_stakes,
)
from scripts.evaluate_sapporo_dirt1700_policy import (  # noqa: E402
    SAPPORO_D1700_WEIGHTS,
    fit_position_model,
    is_sapporo_dirt1700,
    predict_position,
)
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


DEFAULT_OUT = Path("outputs/analysis/venue_special_global_policy")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def add_rank(df: pd.DataFrame, score_col: str) -> pd.Series:
    return df.groupby(RACE_COL)[score_col].rank(ascending=False, method="first").astype(int)


def distance_group(df: pd.DataFrame) -> pd.Series:
    dist = num(df, "距離")
    return pd.Series(
        np.select(
            [dist.le(1200), dist.eq(1500), dist.between(1700, 1800), dist.eq(2000), dist.ge(2400)],
            ["<=1200", "1500", "1700-1800", "2000", "2400+"],
            default="other",
        ),
        index=df.index,
    )


def prepare_scores(train: pd.DataFrame, test: pd.DataFrame, base_model: SimpleRaceRanker) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x, test_x, _ = add_expected_lap_features(train, test)
    plus_numeric = list(base_model.numeric_features) + [
        col for col in NEW_NUMERIC_FEATURES if col not in base_model.numeric_features
    ]
    plus_categorical = list(base_model.categorical_features) + [
        col for col in NEW_CATEGORICAL_FEATURES if col not in base_model.categorical_features
    ]
    model = fit_ranker(train_x, plus_numeric, plus_categorical, float(base_model.ridge_alpha), int(base_model.categorical_top_k))
    train_x["expected_lap_score"] = model.predict(train_x)
    test_x["expected_lap_score"] = model.predict(test_x)
    return train_x, test_x


def add_kyoto_policy_scores(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["win_score"] = out["expected_lap_score"]
    out["place_score"] = out["expected_lap_score"]
    out["kyoto_layout_group"] = "not_kyoto"

    kyoto_mask = out["場所"].astype(str).eq("京都")
    kyoto = out[kyoto_mask].copy()
    if len(kyoto) == 0:
        return out
    kyoto = add_kyoto_components(kyoto, "expected_lap_score")
    kyoto = add_kyoto_layout_group(kyoto)
    inner = kyoto["kyoto_layout_group"].eq("turf_inner")
    outer = kyoto["kyoto_layout_group"].eq("turf_outer")
    kyoto.loc[inner, "win_score"] = score_with_layout_weights(kyoto.loc[inner], "expected_lap_score", INNER_HEAD_WEIGHTS)
    kyoto.loc[outer, "place_score"] = score_with_layout_weights(kyoto.loc[outer], "expected_lap_score", OUTER_PLACE_WEIGHTS)
    out.loc[kyoto.index, "win_score"] = kyoto["win_score"]
    out.loc[kyoto.index, "place_score"] = kyoto["place_score"]
    out.loc[kyoto.index, "kyoto_layout_group"] = kyoto["kyoto_layout_group"]
    return out


def add_sapporo_policy_scores(train: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["sapporo_group"] = "not_sapporo"
    out["sapporo_d1700_score"] = out["expected_lap_score"]
    out["projected_4c_front_rank"] = np.nan
    out["sapporo_distance_group"] = distance_group(out)

    sapporo_mask = out["場所"].astype(str).eq("札幌")
    sapporo = out[sapporo_mask].copy()
    if len(sapporo) == 0:
        return out

    sapporo = add_sapporo_components(sapporo, "expected_lap_score")
    sapporo["sapporo_group"] = np.where(sapporo["芝・ダ"].astype(str).str.contains("芝", na=False), "sapporo_turf", "sapporo_dirt")
    sapporo["sapporo_distance_group"] = distance_group(sapporo)
    sapporo["sapporo_d1700_score"] = sapporo["expected_lap_score"]

    train_d1700 = train[is_sapporo_dirt1700(train)].copy()
    test_d1700_mask = is_sapporo_dirt1700(sapporo)
    if len(train_d1700) and test_d1700_mask.any():
        train_d1700 = add_sapporo_components(train_d1700, "expected_lap_score")
        pos_model = fit_position_model(train_d1700)
        d1700 = sapporo[test_d1700_mask].copy()
        d1700["sapporo_d1700_score"] = score_sapporo_weights(d1700, "expected_lap_score", SAPPORO_D1700_WEIGHTS)
        d1700["projected_4c_rate"] = predict_position(pos_model, d1700)
        d1700["projected_4c_pos"] = d1700["projected_4c_rate"] * num(d1700, "頭数", 14.0)
        d1700["projected_4c_front_rank"] = d1700.groupby(RACE_COL)["projected_4c_pos"].rank(ascending=True, method="first")
        sapporo.loc[d1700.index, "sapporo_d1700_score"] = d1700["sapporo_d1700_score"]
        sapporo.loc[d1700.index, "projected_4c_front_rank"] = d1700["projected_4c_front_rank"]
        sapporo.loc[d1700.index, "sapporo_group"] = "sapporo_dirt1700"

    out.loc[sapporo.index, "sapporo_group"] = sapporo["sapporo_group"]
    out.loc[sapporo.index, "sapporo_distance_group"] = sapporo["sapporo_distance_group"]
    out.loc[sapporo.index, "sapporo_d1700_score"] = sapporo["sapporo_d1700_score"]
    out.loc[sapporo.index, "projected_4c_front_rank"] = sapporo["projected_4c_front_rank"]
    return out


def assign_venue_special_stakes(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["expected_rank"] = add_rank(out, "expected_lap_score")
    out["win_rank"] = add_rank(out, "win_score")
    out["place_rank"] = add_rank(out, "place_score")
    out["d1700_rank"] = add_rank(out, "sapporo_d1700_score")
    out["win_stake"] = 0.0
    out["place_stake"] = 0.0

    odds = num(out, "単勝オッズ", np.nan)
    venue = out["場所"].astype(str)

    # Default outside special venues.
    default_top = out["expected_rank"].eq(1)
    out.loc[default_top & odds.ge(2.0), "win_stake"] = 100.0
    out.loc[default_top, "place_stake"] = 100.0

    # Kyoto layout rules.
    kyoto_inner = venue.eq("京都") & out["kyoto_layout_group"].eq("turf_inner")
    kyoto_outer = venue.eq("京都") & out["kyoto_layout_group"].eq("turf_outer")
    out.loc[kyoto_inner | kyoto_outer, ["win_stake", "place_stake"]] = 0.0
    inner_top = kyoto_inner & out["win_rank"].eq(1)
    out.loc[inner_top & odds.ge(2.5), "win_stake"] = 150.0
    out.loc[inner_top & odds.between(1.8, 2.5, inclusive="left"), "win_stake"] = 80.0
    outer_top = kyoto_outer & out["place_rank"].eq(1)
    out.loc[outer_top & odds.ge(3.0), "win_stake"] = 70.0
    out.loc[outer_top, "place_stake"] = 140.0

    # Sapporo rules.
    sapporo = venue.eq("札幌")
    out.loc[sapporo, ["win_stake", "place_stake"]] = 0.0

    # Sapporo dirt 1700: projected-position front rank is the key.
    d1700 = sapporo & out["sapporo_group"].eq("sapporo_dirt1700")
    d1700_top = d1700 & out["d1700_rank"].eq(1) & num(out, "projected_4c_front_rank").le(4)
    out.loc[d1700_top & odds.ge(2.0), "win_stake"] = 100.0
    out.loc[d1700_top, "place_stake"] = 100.0
    d1700_hole = d1700 & out["d1700_rank"].le(3) & num(out, "projected_4c_front_rank").le(4) & num(out, "人気").ge(5)
    out.loc[d1700_hole, "win_stake"] = out.loc[d1700_hole, "win_stake"] + 50.0

    # Sapporo turf: place/axis leaning. Avoid short win, suppress bad distance bands.
    turf = sapporo & out["sapporo_group"].eq("sapporo_turf")
    turf_top = turf & out["expected_rank"].eq(1)
    good_turf = ~out["sapporo_distance_group"].isin(["<=1200", "2000"])
    out.loc[turf_top & good_turf, "place_stake"] = 120.0
    out.loc[turf_top & good_turf & odds.ge(4.0), "win_stake"] = 50.0
    # Long turf is place-only.
    out.loc[turf_top & out["sapporo_distance_group"].eq("2400+"), "win_stake"] = 0.0

    # Sapporo non-1700 dirt and <=1200 are alert zones: only tiny place on strong top1.
    other_sapporo = sapporo & out["win_stake"].eq(0) & out["place_stake"].eq(0)
    out.loc[other_sapporo & out["expected_rank"].eq(1) & odds.lt(3.0), "place_stake"] = 50.0
    return out


def group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, part in df.groupby(group_col, dropna=False, sort=True):
        rows.append({group_col: group, "races": int(part[RACE_COL].nunique()), **policy_metrics(part)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate global policy with Kyoto and Sapporo special rules.")
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
    train_x, scored = prepare_scores(train, test, base_model)
    scored = add_kyoto_policy_scores(scored)
    scored = add_sapporo_policy_scores(train_x, scored)

    base_policy = simple_base_stakes(scored.assign(expected_rank=add_rank(scored, "expected_lap_score")))
    special_policy = assign_venue_special_stakes(scored)

    summary = pd.DataFrame(
        [
            {"policy": "expected_lap_top1_win_place", **policy_metrics(base_policy)},
            {"policy": "kyoto_sapporo_special_policy", **policy_metrics(special_policy)},
        ]
    )
    summary.to_csv(out_dir / "venue_special_global_policy_summary.csv", index=False, encoding="utf-8-sig")
    group_metrics(special_policy, "場所").to_csv(out_dir / "venue_special_policy_by_venue.csv", index=False, encoding="utf-8-sig")
    group_metrics(special_policy, "sapporo_group").to_csv(out_dir / "venue_special_policy_by_sapporo_group.csv", index=False, encoding="utf-8-sig")
    group_metrics(special_policy, "kyoto_layout_group").to_csv(out_dir / "venue_special_policy_by_kyoto_layout.csv", index=False, encoding="utf-8-sig")
    special_policy.to_csv(out_dir / "venue_special_policy_bets.csv", index=False, encoding="utf-8-sig")

    show = summary.copy()
    pct = [c for c in show.columns if c.endswith("_rate") or c.endswith("_roi")]
    show[pct] = show[pct] * 100.0
    print(show.to_string(index=False))
    by_sap = group_metrics(special_policy, "sapporo_group")
    pct = [c for c in by_sap.columns if c.endswith("_rate") or c.endswith("_roi")]
    by_sap[pct] = by_sap[pct] * 100.0
    print("\nBy Sapporo group")
    print(by_sap.to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
