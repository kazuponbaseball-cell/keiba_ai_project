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


DEFAULT_OUT = Path("outputs/analysis/sapporo_projected_4c")

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

PROJECTED_4C_FEATURES = [
    "prev_corner4_position_rate",
    "past3_avg_corner4_position_rate",
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
    "枠番",
    "馬番",
    "頭数",
    "距離",
    "race_need_lead_count",
    "race_front_runner_ratio",
    "race_early_pressure_score",
    "race_slow_pace_risk",
    "race_pace_collapse_risk",
    "draw_pace_fit_score",
]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


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


def build_design(df: pd.DataFrame, medians: dict[str, float] | None = None) -> tuple[np.ndarray, dict[str, float], list[str]]:
    parts = [np.ones((len(df), 1), dtype=float)]
    names = ["intercept"]
    out_medians: dict[str, float] = {} if medians is None else dict(medians)
    for col in PROJECTED_4C_FEATURES:
        values = num(df, col, np.nan)
        if medians is None:
            median = float(values.median()) if values.notna().any() else 0.0
            out_medians[col] = median
        else:
            median = out_medians[col]
        filled = values.fillna(median).astype(float)
        std = float(filled.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        mean = float(filled.mean())
        if medians is None:
            out_medians[f"{col}__mean"] = mean
            out_medians[f"{col}__std"] = std
        else:
            mean = out_medians.get(f"{col}__mean", mean)
            std = out_medians.get(f"{col}__std", std)
        parts.append(((filled - mean) / std).to_numpy()[:, None])
        names.append(col)

    surface = df["芝・ダ"].astype(str) if "芝・ダ" in df.columns else pd.Series("", index=df.index)
    for name, mask in {
        "surface_turf": surface.str.contains("芝", regex=False, na=False),
        "surface_dirt": surface.str.contains("ダ", regex=False, na=False),
    }.items():
        parts.append(mask.astype(float).to_numpy()[:, None])
        names.append(name)
    for value in ["<=1200", "1500", "1700-1800", "2000", "2400+"]:
        parts.append(distance_group(df).eq(value).astype(float).to_numpy()[:, None])
        names.append(f"distance_group={value}")
    return np.hstack(parts), out_medians, names


def fit_projected_4c(train: pd.DataFrame, alpha: float = 20.0) -> dict[str, Any]:
    target = num(train, "4角.1", np.nan) / num(train, "頭数", np.nan).replace(0, np.nan)
    mask = target.notna() & np.isfinite(target)
    x, medians, names = build_design(train.loc[mask])
    y = target.loc[mask].clip(0.03, 1.0).to_numpy(dtype=float)
    reg = np.eye(x.shape[1]) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + reg, x.T @ y)
    return {"coef": coef, "medians": medians, "names": names}


def predict_projected_4c(model: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    x, _, _ = build_design(df, model["medians"])
    return np.clip(x @ model["coef"], 0.03, 1.0)


def add_scores(train: pd.DataFrame, test: pd.DataFrame, base_model: SimpleRaceRanker) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x, test_x, _ = add_expected_lap_features(train, test)
    plus_numeric = list(base_model.numeric_features) + [
        col for col in NEW_NUMERIC_FEATURES if col not in base_model.numeric_features
    ]
    plus_categorical = list(base_model.categorical_features) + [
        col for col in NEW_CATEGORICAL_FEATURES if col not in base_model.categorical_features
    ]
    expected_model = fit_ranker(
        train_x,
        plus_numeric,
        plus_categorical,
        float(base_model.ridge_alpha),
        int(base_model.categorical_top_k),
    )
    train_x["expected_lap_score"] = expected_model.predict(train_x)
    test_x["expected_lap_score"] = expected_model.predict(test_x)
    return train_x, test_x


def prepare_sapporo(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["場所"].astype(str).eq("札幌")].copy()
    out = add_sapporo_components(out, "expected_lap_score")
    out["surface_group"] = np.where(out["芝・ダ"].astype(str).str.contains("芝", na=False), "turf", "dirt")
    out["sapporo_surface_score"] = out["expected_lap_score"]
    for segment, weights in SAPPORO_SURFACE_WEIGHTS.items():
        mask = out["surface_group"].eq(segment)
        out.loc[mask, "sapporo_surface_score"] = score_with_weights(out.loc[mask], "expected_lap_score", weights)
    out["sapporo_distance_group"] = distance_group(out)
    return out


def metric_for_rank(df: pd.DataFrame, score_col: str, mask: pd.Series | None = None) -> dict[str, Any]:
    out = df.copy()
    out["rank"] = out.groupby(RACE_COL)[score_col].rank(ascending=False, method="first").astype(int)
    part = out[out["rank"].eq(1)].copy()
    if mask is not None:
        part = part[mask.loc[part.index]]
    if len(part) == 0:
        return {"bets": 0, "win_rate": np.nan, "top3_rate": np.nan, "win_roi": np.nan, "place_roi": np.nan}
    win_pay = num(part, "単勝配当", 0.0).where(part["target_win"].eq(1), 0.0)
    place_pay = num(part, "複勝配当", 0.0).where(part["target_top3"].eq(1), 0.0)
    return {
        "bets": int(len(part)),
        "win_rate": float(part["target_win"].mean()),
        "top3_rate": float(part["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (len(part) * 100.0)),
        "place_roi": float(place_pay.sum() / (len(part) * 100.0)),
        "avg_popularity": float(num(part, "人気").mean()),
        "avg_odds": float(num(part, "単勝オッズ").mean()),
        "avg_projected_4c": float(num(part, "projected_4c_pos").mean()),
        "avg_actual_4c": float(num(part, "4角.1").mean()),
    }


def optimize_threshold(train: pd.DataFrame, score_col: str, min_bets: int = 80) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.arange(3.0, 6.01, 0.25):
        mask = num(train, "projected_4c_pos").le(threshold)
        metrics = metric_for_rank(train, score_col, mask)
        too_few_penalty = max(0, min_bets - metrics["bets"]) * 0.015
        score = (
            metrics["win_roi"]
            + 0.45 * metrics["place_roi"]
            + 0.35 * metrics["win_rate"]
            + 0.25 * metrics["top3_rate"]
            + 0.0009 * metrics["bets"]
            - too_few_penalty
        )
        rows.append({"threshold": float(threshold), "selection_score": float(score), **metrics})
    grid = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    return float(grid.iloc[0]["threshold"]), grid


def threshold_grid(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(3.0, 7.51, 0.25):
        mask = num(df, "projected_4c_pos").le(threshold)
        metrics = metric_for_rank(df, score_col, mask)
        rows.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame(rows)


def segment_table(df: pd.DataFrame, score_col: str, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["rank"] = out.groupby(RACE_COL)[score_col].rank(ascending=False, method="first").astype(int)
    odds = num(out, "単勝オッズ", np.nan)
    pop = num(out, "人気", np.nan)
    proj = num(out, "projected_4c_pos", np.nan)
    checks = [
        ("top1_all", out["rank"].eq(1)),
        ("top1_projected_front", out["rank"].eq(1) & proj.le(threshold)),
        ("top1_projected_back", out["rank"].eq(1) & proj.gt(threshold)),
        ("top1_projected_front_odds2plus", out["rank"].eq(1) & proj.le(threshold) & odds.ge(2.0)),
        ("top1_projected_front_pop4_6", out["rank"].eq(1) & proj.le(threshold) & pop.between(4, 6)),
        ("top1_projected_front_turf", out["rank"].eq(1) & proj.le(threshold) & out["surface_group"].eq("turf")),
        ("top1_projected_front_dirt", out["rank"].eq(1) & proj.le(threshold) & out["surface_group"].eq("dirt")),
        ("top1_projected_front_1700_1800", out["rank"].eq(1) & proj.le(threshold) & out["sapporo_distance_group"].eq("1700-1800")),
        ("top1_projected_front_1200less", out["rank"].eq(1) & proj.le(threshold) & out["sapporo_distance_group"].eq("<=1200")),
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
                "win_rate": float(part["target_win"].mean()),
                "top3_rate": float(part["target_top3"].mean()),
                "win_roi": float(win_pay.sum() / (len(part) * 100.0)),
                "place_roi": float(place_pay.sum() / (len(part) * 100.0)),
                "avg_projected_4c": float(proj.loc[part.index].mean()),
                "avg_actual_4c": float(num(part, "4角.1").mean()),
                "avg_popularity": float(pop.loc[part.index].mean()),
                "avg_odds": float(odds.loc[part.index].mean()),
            }
        )
    return pd.DataFrame(rows)


def projected_rank_segments(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.copy()
    out["rank"] = out.groupby(RACE_COL)[score_col].rank(ascending=False, method="first").astype(int)
    out["projected_4c_front_rank"] = out.groupby(RACE_COL)["projected_4c_pos"].rank(ascending=True, method="first")
    odds = num(out, "単勝オッズ", np.nan)
    pop = num(out, "人気", np.nan)
    checks = [
        ("top1_projected_front_rank_le3", out["rank"].eq(1) & out["projected_4c_front_rank"].le(3)),
        ("top1_projected_front_rank_le4", out["rank"].eq(1) & out["projected_4c_front_rank"].le(4)),
        ("top1_projected_front_rank_le5", out["rank"].eq(1) & out["projected_4c_front_rank"].le(5)),
        ("top1_projected_front_rank_gt5", out["rank"].eq(1) & out["projected_4c_front_rank"].gt(5)),
        ("top1_front_rank_le4_odds2plus", out["rank"].eq(1) & out["projected_4c_front_rank"].le(4) & odds.ge(2.0)),
        ("top1_front_rank_le4_pop4_6", out["rank"].eq(1) & out["projected_4c_front_rank"].le(4) & pop.between(4, 6)),
        ("top1_front_rank_le4_turf", out["rank"].eq(1) & out["projected_4c_front_rank"].le(4) & out["surface_group"].eq("turf")),
        ("top1_front_rank_le4_dirt", out["rank"].eq(1) & out["projected_4c_front_rank"].le(4) & out["surface_group"].eq("dirt")),
        ("top3_pop5plus_front_rank_le4", out["rank"].le(3) & pop.ge(5) & out["projected_4c_front_rank"].le(4)),
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
                "win_rate": float(part["target_win"].mean()),
                "top3_rate": float(part["target_top3"].mean()),
                "win_roi": float(win_pay.sum() / (len(part) * 100.0)),
                "place_roi": float(place_pay.sum() / (len(part) * 100.0)),
                "avg_projected_front_rank": float(out.loc[part.index, "projected_4c_front_rank"].mean()),
                "avg_projected_4c": float(num(part, "projected_4c_pos").mean()),
                "avg_actual_4c": float(num(part, "4角.1").mean()),
                "avg_popularity": float(pop.loc[part.index].mean()),
                "avg_odds": float(odds.loc[part.index].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Sapporo projected-4C pre-race filter/policy.")
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

    train_x, test_x = add_scores(train, test, base_model)
    sapporo_train = prepare_sapporo(train_x)
    sapporo_test = prepare_sapporo(test_x)

    fourc_model = fit_projected_4c(sapporo_train)
    sapporo_train["projected_4c_rate"] = predict_projected_4c(fourc_model, sapporo_train)
    sapporo_test["projected_4c_rate"] = predict_projected_4c(fourc_model, sapporo_test)
    sapporo_train["projected_4c_pos"] = sapporo_train["projected_4c_rate"] * num(sapporo_train, "頭数", 14.0)
    sapporo_test["projected_4c_pos"] = sapporo_test["projected_4c_rate"] * num(sapporo_test, "頭数", 14.0)

    threshold, grid = optimize_threshold(sapporo_train, "sapporo_surface_score")
    test_grid = threshold_grid(sapporo_test, "sapporo_surface_score")
    base_metrics = metric_for_rank(sapporo_test, "sapporo_surface_score")
    filtered_metrics = metric_for_rank(
        sapporo_test,
        "sapporo_surface_score",
        num(sapporo_test, "projected_4c_pos").le(threshold),
    )
    segments = segment_table(sapporo_test, "sapporo_surface_score", threshold)
    rank_segments = projected_rank_segments(sapporo_test, "sapporo_surface_score")
    coef = pd.DataFrame(
        {"feature": fourc_model["names"], "coef": fourc_model["coef"], "abs_coef": np.abs(fourc_model["coef"])}
    ).sort_values("abs_coef", ascending=False)

    summary = pd.DataFrame(
        [
            {"policy": "sapporo_surface_top1", "threshold": np.nan, **base_metrics},
            {"policy": "projected_4c_front_filter", "threshold": threshold, **filtered_metrics},
        ]
    )
    summary.to_csv(out_dir / "sapporo_projected_4c_summary.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(out_dir / "sapporo_projected_4c_threshold_grid_train.csv", index=False, encoding="utf-8-sig")
    test_grid.to_csv(out_dir / "sapporo_projected_4c_threshold_grid_test_reference.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(out_dir / "sapporo_projected_4c_segments.csv", index=False, encoding="utf-8-sig")
    rank_segments.to_csv(out_dir / "sapporo_projected_4c_rank_segments.csv", index=False, encoding="utf-8-sig")
    sapporo_test.to_csv(out_dir / "sapporo_projected_4c_scored_test.csv", index=False, encoding="utf-8-sig")
    coef.to_csv(out_dir / "sapporo_projected_4c_model_coefficients.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "sapporo_projected_4c_model.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": threshold,
                "features": fourc_model["names"],
                "coef": [float(x) for x in fourc_model["coef"]],
                "medians": {k: float(v) for k, v in fourc_model["medians"].items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    show = summary.copy()
    pct = [c for c in show.columns if c.endswith("_rate") or c.endswith("_roi")]
    show[pct] = show[pct] * 100.0
    print("Summary")
    print(show.to_string(index=False))
    show_seg = segments.copy()
    pct = [c for c in show_seg.columns if c.endswith("_rate") or c.endswith("_roi")]
    show_seg[pct] = show_seg[pct] * 100.0
    print("\nSegments")
    print(show_seg.to_string(index=False))
    print("\nTop projected 4C coefficients")
    print(coef.head(20).to_string(index=False))
    show_rank = rank_segments.copy()
    pct = [c for c in show_rank.columns if c.endswith("_rate") or c.endswith("_roi")]
    show_rank[pct] = show_rank[pct] * 100.0
    print("\nProjected rank segments")
    print(show_rank.to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
