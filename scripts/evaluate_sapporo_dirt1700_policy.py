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


DEFAULT_OUT = Path("outputs/analysis/sapporo_dirt1700_policy")

SAPPORO_D1700_WEIGHTS = {
    "dirt1000_speed_fit": 0.03,
    "front_collapse_risk": 0.02,
    "member_form_fit": 0.03,
    "outer_loss_risk": 0.02,
}

D1700_POSITION_FEATURES = [
    "prev_corner4_position_rate",
    "past3_avg_corner4_position_rate",
    "前4角.1",
    "前走馬番",
    "horse_front_run_rate_past5",
    "horse_stalker_rate_past5",
    "horse_midpack_rate_past5",
    "horse_closer_rate_past5",
    "front_running_tendency",
    "closing_tendency",
    "horse_need_lead_rate",
    "horse_can_rate_rate",
    "prev_early_move",
    "horse_early_move_avg_past5",
    "horse_late_gain_avg_past5",
    "枠番",
    "馬番",
    "頭数",
    "斤量",
    "前走斤量",
    "weight_diff",
    "間隔",
    "race_need_lead_count",
    "race_front_runner_ratio",
    "race_early_pressure_score",
    "race_slow_pace_risk",
    "race_pace_collapse_risk",
    "draw_pace_fit_score",
    "front_pressure_rank_score",
    "jockey_venue_avg_score",
    "jockey_venue_popularity_outperform_rate",
    "trainer_venue_avg_score",
    "same_venue_avg_score",
    "same_venue_top3_rate",
    "body_prev_weight",
    "前走馬体重",
]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def is_sapporo_dirt1700(df: pd.DataFrame) -> pd.Series:
    return (
        df["場所"].astype(str).eq("札幌")
        & df["芝・ダ"].astype(str).str.contains("ダ", regex=False, na=False)
        & num(df, "距離").eq(1700)
    )


def add_surface_score(df: pd.DataFrame) -> pd.DataFrame:
    out = add_sapporo_components(df.copy(), "expected_lap_score")
    out["sapporo_d1700_score"] = score_with_weights(out, "expected_lap_score", SAPPORO_D1700_WEIGHTS)
    return out


def build_design(df: pd.DataFrame, stats: dict[str, float] | None = None) -> tuple[np.ndarray, dict[str, float], list[str]]:
    parts = [np.ones((len(df), 1), dtype=float)]
    names = ["intercept"]
    out_stats = {} if stats is None else dict(stats)
    field = num(df, "頭数", 14.0).replace(0, np.nan).fillna(14.0)
    horse_no = num(df, "馬番", np.nan).fillna(field / 2.0)
    frame = num(df, "枠番", np.nan).fillna(4.0)
    engineered = {
        "relative_horse_no": horse_no / field,
        "inner_gate": frame.le(3).astype(float),
        "middle_gate": frame.between(4, 6).astype(float),
        "outer_gate": frame.ge(7).astype(float),
        "front_stalker_sum": num(df, "horse_front_run_rate_past5", 0.0) + num(df, "horse_stalker_rate_past5", 0.0),
        "front_need_vs_field": num(df, "horse_need_lead_rate", 0.0) * (1.0 + num(df, "race_need_lead_count", 0.0) / 5.0),
        "body_power": ((num(df, "body_prev_weight", np.nan).fillna(num(df, "前走馬体重", np.nan)).fillna(470.0) - 460.0) / 70.0).clip(-0.8, 1.2),
    }
    for col in D1700_POSITION_FEATURES:
        values = num(df, col, np.nan)
        if stats is None:
            median = float(values.median()) if values.notna().any() else 0.0
            out_stats[col] = median
            filled_for_stats = values.fillna(median).astype(float)
            out_stats[f"{col}__mean"] = float(filled_for_stats.mean())
            std = float(filled_for_stats.std(ddof=0))
            out_stats[f"{col}__std"] = std if np.isfinite(std) and std > 0 else 1.0
        median = out_stats[col]
        mean = out_stats[f"{col}__mean"]
        std = out_stats[f"{col}__std"]
        filled = values.fillna(median).astype(float)
        parts.append(((filled - mean) / std).to_numpy()[:, None])
        names.append(col)
    for col, values in engineered.items():
        values = pd.to_numeric(values, errors="coerce")
        if stats is None:
            median = float(values.median()) if values.notna().any() else 0.0
            out_stats[col] = median
            filled_for_stats = values.fillna(median).astype(float)
            out_stats[f"{col}__mean"] = float(filled_for_stats.mean())
            std = float(filled_for_stats.std(ddof=0))
            out_stats[f"{col}__std"] = std if np.isfinite(std) and std > 0 else 1.0
        median = out_stats[col]
        mean = out_stats[f"{col}__mean"]
        std = out_stats[f"{col}__std"]
        filled = values.fillna(median).astype(float)
        parts.append(((filled - mean) / std).to_numpy()[:, None])
        names.append(col)
    return np.hstack(parts), out_stats, names


def fit_position_model(train: pd.DataFrame, alpha: float = 18.0) -> dict[str, Any]:
    target = num(train, "4角.1", np.nan) / num(train, "頭数", np.nan).replace(0, np.nan)
    mask = target.notna() & np.isfinite(target)
    x, stats, names = build_design(train.loc[mask])
    y = target.loc[mask].clip(0.03, 1.0).to_numpy(dtype=float)
    reg = np.eye(x.shape[1]) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + reg, x.T @ y)
    return {"coef": coef, "stats": stats, "names": names}


def predict_position(model: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x, _, _ = build_design(df, model["stats"])
    return np.clip(x @ model["coef"], 0.03, 1.0)


def rank_by_score(df: pd.DataFrame, score_col: str) -> pd.Series:
    return df.groupby(RACE_COL)[score_col].rank(ascending=False, method="first").astype(int)


def bet_metrics(part: pd.DataFrame) -> dict[str, Any]:
    if len(part) == 0:
        return {
            "bets": 0,
            "win_rate": np.nan,
            "top3_rate": np.nan,
            "win_roi": np.nan,
            "place_roi": np.nan,
        }
    win_pay = num(part, "単勝配当", 0.0).where(part["target_win"].eq(1), 0.0)
    place_pay = num(part, "複勝配当", 0.0).where(part["target_top3"].eq(1), 0.0)
    return {
        "bets": int(len(part)),
        "races": int(part[RACE_COL].nunique()),
        "win_rate": float(part["target_win"].mean()),
        "top3_rate": float(part["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (len(part) * 100.0)),
        "place_roi": float(place_pay.sum() / (len(part) * 100.0)),
        "avg_popularity": float(num(part, "人気").mean()),
        "avg_odds": float(num(part, "単勝オッズ").mean()),
        "avg_projected_4c": float(num(part, "projected_4c_pos").mean()),
        "avg_projected_front_rank": float(num(part, "projected_4c_front_rank").mean()),
        "avg_actual_4c": float(num(part, "4角.1").mean()),
    }


def segment_metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.copy()
    out["score_rank"] = rank_by_score(out, score_col)
    out["projected_4c_front_rank"] = out.groupby(RACE_COL)["projected_4c_pos"].rank(ascending=True, method="first")
    odds = num(out, "単勝オッズ", np.nan)
    pop = num(out, "人気", np.nan)
    proj = num(out, "projected_4c_pos", np.nan)
    front_rank = num(out, "projected_4c_front_rank", np.nan)
    actual_4c = num(out, "4角.1", np.nan)
    checks = [
        ("top1_all", out["score_rank"].eq(1)),
        ("top1_projected_pos_le4", out["score_rank"].eq(1) & proj.le(4.0)),
        ("top1_projected_pos_le5", out["score_rank"].eq(1) & proj.le(5.0)),
        ("top1_projected_rank_le3", out["score_rank"].eq(1) & front_rank.le(3)),
        ("top1_projected_rank_le4", out["score_rank"].eq(1) & front_rank.le(4)),
        ("top1_projected_rank_gt4", out["score_rank"].eq(1) & front_rank.gt(4)),
        ("top1_rank_le4_odds2plus", out["score_rank"].eq(1) & front_rank.le(4) & odds.ge(2.0)),
        ("top1_rank_le4_pop4_6", out["score_rank"].eq(1) & front_rank.le(4) & pop.between(4, 6)),
        ("top1_actual_4c_le4_reference", out["score_rank"].eq(1) & actual_4c.le(4)),
        ("top3_pop5plus_projected_rank_le4", out["score_rank"].le(3) & pop.ge(5) & front_rank.le(4)),
        ("top3_pop5plus_projected_pos_le5", out["score_rank"].le(3) & pop.ge(5) & proj.le(5.0)),
    ]
    rows = []
    for name, mask in checks:
        rows.append({"segment": name, **bet_metrics(out[mask])})
    return pd.DataFrame(rows)


def ticket_policy(df: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["score_rank"] = rank_by_score(out, score_col)
    out["projected_4c_front_rank"] = out.groupby(RACE_COL)["projected_4c_pos"].rank(ascending=True, method="first")
    out["win_stake"] = 0.0
    out["place_stake"] = 0.0
    odds = num(out, "単勝オッズ", np.nan)
    top = out["score_rank"].eq(1)
    frontish = num(out, "projected_4c_front_rank").le(4)
    value = odds.ge(2.0)
    out.loc[top & frontish & value, "win_stake"] = 100.0
    out.loc[top & frontish, "place_stake"] = 100.0

    # Add small head-shot stakes for projected-front longshots in AI top3.
    pop = num(out, "人気", np.nan)
    top3_hole = out["score_rank"].le(3) & frontish & pop.ge(5)
    out.loc[top3_hole, "win_stake"] = out.loc[top3_hole, "win_stake"] + 50.0

    win_bets = out[out["win_stake"].gt(0)]
    place_bets = out[out["place_stake"].gt(0)]
    win_return = (win_bets["win_stake"] / 100.0 * num(win_bets, "単勝配当", 0.0).where(win_bets["target_win"].eq(1), 0.0)).sum()
    place_return = (
        place_bets["place_stake"] / 100.0 * num(place_bets, "複勝配当", 0.0).where(place_bets["target_top3"].eq(1), 0.0)
    ).sum()
    stake = out["win_stake"].sum() + out["place_stake"].sum()
    summary = pd.DataFrame(
        [
            {
                "policy": "dirt1700_projected_front_ticket",
                "win_bets": int(len(win_bets)),
                "win_hit_rate": float(win_bets["target_win"].mean()) if len(win_bets) else np.nan,
                "win_roi": float(win_return / out["win_stake"].sum()) if out["win_stake"].sum() > 0 else np.nan,
                "place_bets": int(len(place_bets)),
                "place_hit_rate": float(place_bets["target_top3"].mean()) if len(place_bets) else np.nan,
                "place_roi": float(place_return / out["place_stake"].sum()) if out["place_stake"].sum() > 0 else np.nan,
                "total_stake": float(stake),
                "total_return": float(win_return + place_return),
                "total_roi": float((win_return + place_return) / stake) if stake > 0 else np.nan,
            }
        ]
    )
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Sapporo dirt 1700 projected-position policy.")
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
    expected_model = fit_ranker(train_x, plus_numeric, plus_categorical, float(base_model.ridge_alpha), int(base_model.categorical_top_k))
    train_x["expected_lap_score"] = expected_model.predict(train_x)
    test_x["expected_lap_score"] = expected_model.predict(test_x)

    d1700_train = add_surface_score(train_x[is_sapporo_dirt1700(train_x)].copy())
    d1700_test = add_surface_score(test_x[is_sapporo_dirt1700(test_x)].copy())
    pos_model = fit_position_model(d1700_train)
    d1700_train["projected_4c_rate"] = predict_position(pos_model, d1700_train)
    d1700_test["projected_4c_rate"] = predict_position(pos_model, d1700_test)
    d1700_train["projected_4c_pos"] = d1700_train["projected_4c_rate"] * num(d1700_train, "頭数", 14.0)
    d1700_test["projected_4c_pos"] = d1700_test["projected_4c_rate"] * num(d1700_test, "頭数", 14.0)
    d1700_train["projected_4c_front_rank"] = d1700_train.groupby(RACE_COL)["projected_4c_pos"].rank(ascending=True, method="first")
    d1700_test["projected_4c_front_rank"] = d1700_test.groupby(RACE_COL)["projected_4c_pos"].rank(ascending=True, method="first")

    segments = segment_metrics(d1700_test, "sapporo_d1700_score")
    bets, policy_summary = ticket_policy(d1700_test, "sapporo_d1700_score")
    coef = pd.DataFrame({"feature": pos_model["names"], "coef": pos_model["coef"], "abs_coef": np.abs(pos_model["coef"])}).sort_values(
        "abs_coef", ascending=False
    )

    base_summary = pd.DataFrame(
        [
            {"segment": "d1700_top1_baseline", **bet_metrics(d1700_test[rank_by_score(d1700_test, "sapporo_d1700_score").eq(1)])}
        ]
    )
    base_summary.to_csv(out_dir / "sapporo_dirt1700_baseline_summary.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(out_dir / "sapporo_dirt1700_segments.csv", index=False, encoding="utf-8-sig")
    policy_summary.to_csv(out_dir / "sapporo_dirt1700_ticket_policy_summary.csv", index=False, encoding="utf-8-sig")
    bets.to_csv(out_dir / "sapporo_dirt1700_scored_bets.csv", index=False, encoding="utf-8-sig")
    coef.to_csv(out_dir / "sapporo_dirt1700_position_coefficients.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "sapporo_dirt1700_position_model.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "features": pos_model["names"],
                "coef": [float(x) for x in pos_model["coef"]],
                "stats": {k: float(v) for k, v in pos_model["stats"].items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    def pct(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = [c for c in out.columns if c.endswith("_rate") or c.endswith("_roi")]
        out[cols] = out[cols] * 100.0
        return out

    print("Baseline")
    print(pct(base_summary).to_string(index=False))
    print("\nSegments")
    print(pct(segments).to_string(index=False))
    print("\nTicket policy")
    print(pct(policy_summary).to_string(index=False))
    print("\nTop coefficients")
    print(coef.head(20).to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
