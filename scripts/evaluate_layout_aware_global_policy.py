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

from scripts.analyze_kyoto_special_logic import (  # noqa: E402
    add_kyoto_components,
    add_kyoto_layout_group,
    score_with_layout_weights,
)
from scripts.evaluate_expected_lap_rpci_features import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    NEW_CATEGORICAL_FEATURES,
    NEW_NUMERIC_FEATURES,
    RACE_COL,
    add_expected_lap_features,
    fit_ranker,
    metric_summary,
    num,
)
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


DEFAULT_OUT = Path("outputs/analysis/layout_aware_global_policy")

INNER_HEAD_WEIGHTS = {
    "inner_position_fit": 0.06,
    "draw_bias_fit": 0.02,
    "inner_front_risk": 0.02,
}
OUTER_PLACE_WEIGHTS = {
    "outer_front_overtrust_risk": 0.02,
}


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def add_ranks(df: pd.DataFrame, score_col: str) -> pd.Series:
    return df.groupby(RACE_COL)[score_col].rank(ascending=False, method="first").astype(int)


def bet_return(frame: pd.DataFrame, bet_col: str, pay_col: str, hit_col: str) -> dict[str, Any]:
    bets = frame[frame[bet_col].gt(0)].copy()
    if len(bets) == 0:
        return {"bets": 0, "stake": 0.0, "return": 0.0, "roi": np.nan, "hit_rate": np.nan}
    stake = bets[bet_col].sum()
    ret = (bets[bet_col] / 100.0 * num(bets, pay_col, 0.0).where(bets[hit_col].eq(1), 0.0)).sum()
    return {
        "bets": int(len(bets)),
        "stake": float(stake),
        "return": float(ret),
        "roi": float(ret / stake) if stake > 0 else np.nan,
        "hit_rate": float(bets[hit_col].mean()),
    }


def policy_metrics(df: pd.DataFrame) -> dict[str, Any]:
    win = bet_return(df, "win_stake", "単勝配当", "target_win")
    place = bet_return(df, "place_stake", "複勝配当", "target_top3")
    stake = win["stake"] + place["stake"]
    ret = win["return"] + place["return"]
    return {
        "win_bets": win["bets"],
        "win_hit_rate": win["hit_rate"],
        "win_roi": win["roi"],
        "place_bets": place["bets"],
        "place_hit_rate": place["hit_rate"],
        "place_roi": place["roi"],
        "total_stake": stake,
        "total_return": ret,
        "total_roi": float(ret / stake) if stake > 0 else np.nan,
    }


def make_policy_scores(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["win_score"] = out["expected_lap_score"]
    out["place_score"] = out["expected_lap_score"]

    kyoto_mask = out["場所"].astype(str).eq("京都")
    kyoto = out[kyoto_mask].copy()
    kyoto = add_kyoto_components(kyoto, "expected_lap_score")
    kyoto = add_kyoto_layout_group(kyoto)
    inner = kyoto["kyoto_layout_group"].eq("turf_inner")
    outer = kyoto["kyoto_layout_group"].eq("turf_outer")
    kyoto.loc[inner, "win_score"] = score_with_layout_weights(kyoto.loc[inner], "expected_lap_score", INNER_HEAD_WEIGHTS)
    kyoto.loc[outer, "place_score"] = score_with_layout_weights(kyoto.loc[outer], "expected_lap_score", OUTER_PLACE_WEIGHTS)

    out.loc[kyoto.index, "win_score"] = kyoto["win_score"]
    out.loc[kyoto.index, "place_score"] = kyoto["place_score"]
    out.loc[kyoto.index, "kyoto_layout_group"] = kyoto["kyoto_layout_group"]
    out["kyoto_layout_group"] = out["kyoto_layout_group"].fillna("not_kyoto")

    out["expected_rank"] = add_ranks(out, "expected_lap_score")
    out["win_rank"] = add_ranks(out, "win_score")
    out["place_rank"] = add_ranks(out, "place_score")
    return out


def assign_stakes(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["win_stake"] = 0.0
    out["place_stake"] = 0.0

    odds = num(out, "単勝オッズ", np.nan)
    venue = out["場所"].astype(str)
    layout = out["kyoto_layout_group"].astype(str)

    # Baseline practical policy: AI top1 win/place. Avoid very short prices for win.
    base_candidate = out["expected_rank"].eq(1)
    out.loc[base_candidate & odds.ge(2.0), "win_stake"] = 100.0
    out.loc[base_candidate, "place_stake"] = 100.0

    # Kyoto inner: head/value mode. Use the inner-course head score for win,
    # and reduce place exposure because the validation pattern was volatile.
    inner_top = venue.eq("京都") & layout.eq("turf_inner") & out["win_rank"].eq(1)
    out.loc[venue.eq("京都") & layout.eq("turf_inner"), ["win_stake", "place_stake"]] = 0.0
    out.loc[inner_top & odds.ge(2.5), "win_stake"] = 150.0
    out.loc[inner_top & odds.between(1.8, 2.5, inclusive="left"), "win_stake"] = 80.0

    # Kyoto outer: place/axis mode. Keep win only for non-short prices, strengthen place.
    outer_top = venue.eq("京都") & layout.eq("turf_outer") & out["place_rank"].eq(1)
    out.loc[venue.eq("京都") & layout.eq("turf_outer"), ["win_stake", "place_stake"]] = 0.0
    out.loc[outer_top & odds.ge(3.0), "win_stake"] = 70.0
    out.loc[outer_top, "place_stake"] = 140.0

    return out


def simple_base_stakes(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["win_stake"] = 0.0
    out["place_stake"] = 0.0
    odds = num(out, "単勝オッズ", np.nan)
    top = out["expected_rank"].eq(1)
    out.loc[top & odds.ge(2.0), "win_stake"] = 100.0
    out.loc[top, "place_stake"] = 100.0
    return out


def group_policy(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, part in df.groupby(group_col, dropna=False, sort=True):
        row = {group_col: group, "races": int(part[RACE_COL].nunique()), **policy_metrics(part)}
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate global policy with Kyoto layout-aware ticket overlay.")
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
    scored = make_policy_scores(scored)

    base_policy = simple_base_stakes(scored)
    layout_policy = assign_stakes(scored)

    summary = pd.DataFrame(
        [
            {"policy": "expected_lap_top1_win_place", **policy_metrics(base_policy)},
            {"policy": "kyoto_layout_aware_policy", **policy_metrics(layout_policy)},
        ]
    )
    summary.to_csv(out_dir / "layout_aware_global_policy_summary.csv", index=False, encoding="utf-8-sig")
    group_policy(layout_policy, "場所").to_csv(out_dir / "layout_aware_policy_by_venue.csv", index=False, encoding="utf-8-sig")
    group_policy(layout_policy, "kyoto_layout_group").to_csv(
        out_dir / "layout_aware_policy_by_kyoto_layout.csv", index=False, encoding="utf-8-sig"
    )
    layout_policy[
        [
            RACE_COL,
            "場所",
            "Ｒ",
            "レース名",
            "馬名",
            "芝・ダ",
            "距離",
            "トラックコード",
            "kyoto_layout_group",
            "人気",
            "単勝オッズ",
            "確定着順",
            "target_win",
            "target_top3",
            "expected_rank",
            "win_rank",
            "place_rank",
            "win_stake",
            "place_stake",
            "単勝配当",
            "複勝配当",
        ]
    ].to_csv(out_dir / "layout_aware_policy_bets.csv", index=False, encoding="utf-8-sig")

    show = summary.copy()
    pct = [c for c in show.columns if c.endswith("_rate") or c.endswith("_roi")]
    show[pct] = show[pct] * 100.0
    print(show.to_string(index=False))
    by_layout = group_policy(layout_policy, "kyoto_layout_group")
    pct = [c for c in by_layout.columns if c.endswith("_rate") or c.endswith("_roi")]
    by_layout[pct] = by_layout[pct] * 100.0
    print("\nBy Kyoto layout group")
    print(by_layout.to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
