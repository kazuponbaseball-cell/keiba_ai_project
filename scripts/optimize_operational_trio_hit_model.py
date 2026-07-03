from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_operational_trio_addon import _candidate_universe, _load_trio_payoffs, _prepare_scored
from scripts.optimize_operational_win_addon import _json_default, _load_base_tickets, _metrics, _num
from src.utils.paths import ensure_dir, project_path


FEATURES = [
    "anchor_place_score",
    "anchor_danger",
    "partner_max_danger",
    "partner_max_rank",
    "partner_max_order",
    "partner_max_odds_log",
    "partner_min_score",
    "pair_quinella_score",
    "market_overlay_score",
    "race_difficulty_score",
    "b_partner_score",
    "c_partner_score",
    "partner_score_mean",
    "partner_score_gap",
    "combo_quality",
    "value_shape",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["partner_max_odds_log"] = np.log1p(_num(out.get("partner_max_odds"), out.index, 0.0).clip(lower=0.0))
    out["partner_score_mean"] = (_num(out.get("b_partner_score"), out.index, 0.0) + _num(out.get("c_partner_score"), out.index, 0.0)) / 2.0
    out["partner_score_gap"] = (_num(out.get("b_partner_score"), out.index, 0.0) - _num(out.get("c_partner_score"), out.index, 0.0)).abs()
    out["combo_quality"] = (
        0.34 * _num(out.get("anchor_place_score"), out.index, 0.0)
        + 0.28 * _num(out.get("partner_score_mean"), out.index, 0.0)
        + 0.22 * _num(out.get("pair_quinella_score"), out.index, 0.0)
        + 0.16 * _num(out.get("market_overlay_score"), out.index, 0.0)
    )
    out["value_shape"] = (
        _num(out.get("market_overlay_score"), out.index, 0.0)
        * (1.0 - _num(out.get("race_difficulty_score"), out.index, 0.0).clip(0.0, 1.0))
        * (1.0 - _num(out.get("partner_max_danger"), out.index, 0.0).clip(0.0, 1.0))
    )
    for col in FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = _num(out[col], out.index, 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _fit_logistic(
    train: pd.DataFrame,
    *,
    lr: float = 0.045,
    epochs: int = 900,
    l2: float = 0.03,
    positive_payoff_weight: bool = False,
) -> dict:
    x = train[FEATURES].to_numpy(dtype=float)
    y = train["hit"].astype(float).to_numpy()
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std <= 1e-9, 1.0, std)
    z = (x - mean) / std
    z = np.c_[np.ones(len(z)), z]
    w = np.zeros(z.shape[1], dtype=float)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    pos_weight = min(neg / pos, 24.0)
    sample_weight = np.where(y > 0, pos_weight, 1.0)
    if positive_payoff_weight and y.sum() > 0:
        hit_pay = _num(train.get("trio_pay"), train.index, 0.0).to_numpy(dtype=float)
        median_hit_pay = np.nanmedian(hit_pay[(y > 0) & np.isfinite(hit_pay) & (hit_pay > 0)])
        if np.isfinite(median_hit_pay) and median_hit_pay > 0:
            payoff_boost = np.sqrt(np.clip(hit_pay / median_hit_pay, 0.25, 9.0))
            sample_weight = np.where(y > 0, sample_weight * payoff_boost, sample_weight)
    sample_weight = sample_weight / sample_weight.mean()

    for _ in range(epochs):
        p = _sigmoid(z @ w)
        grad = (z.T @ ((p - y) * sample_weight)) / len(y)
        grad[1:] += l2 * w[1:] / len(y)
        w -= lr * grad
    return {"weights": w, "mean": mean, "std": std, "median": med, "pos_rate": float(y.mean())}


def _predict(model: dict, df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURES].to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, model["median"])
    z = (x - model["mean"]) / model["std"]
    z = np.c_[np.ones(len(z)), z]
    raw = _sigmoid(z @ model["weights"])
    # Light shrinkage keeps rare-combo probabilities from becoming too brave.
    base = float(model.get("pos_rate", 0.03))
    return 0.75 * raw + 0.25 * base


def _select_by_policy(df: pd.DataFrame, params: dict, stake_yen: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    selected = df[
        _num(df.get("trio_model_prob"), df.index, 0.0).ge(params["prob_min"])
        & _num(df.get("race_difficulty_score"), df.index, 1.0).le(params["difficulty_max"])
        & _num(df.get("partner_max_order"), df.index, 999).le(params["partner_order_max"])
        & _num(df.get("partner_max_rank"), df.index, 999).le(params["partner_rank_max"])
        & _num(df.get("partner_max_odds"), df.index, 9999).le(params["partner_odds_max"])
    ].copy()
    if selected.empty:
        return selected
    selected = (
        selected.sort_values(["race_id", "trio_model_prob", "combo_quality", "market_overlay_score"], ascending=[True, False, False, False])
        .groupby("race_id", as_index=False)
        .head(params["tickets_per_race"])
    )
    selected["stake_yen"] = float(stake_yen)
    selected["return_yen"] = selected["trio_pay"].where(selected["hit"], 0.0) * stake_yen / 100.0
    selected["operation_profile"] = "trio_model"
    selected["operation_profile_label"] = "3連複モデル"
    selected["operation_strength_rank"] = 2
    return selected


def _policy_grid(train: pd.DataFrame) -> list[dict]:
    probs = _num(train.get("trio_model_prob"), train.index, 0.0)
    quantile_thresholds = [float(probs.quantile(q)) for q in [0.84, 0.88, 0.92, 0.95]]
    fixed_thresholds = [0.035, 0.045, 0.055, 0.070]
    thresholds = sorted({round(x, 5) for x in quantile_thresholds + fixed_thresholds if np.isfinite(x) and x > 0})
    rows: list[dict] = []
    for prob_min, tickets_per_race, difficulty_max, partner_order_max, partner_rank_max, partner_odds_max in product(
        thresholds,
        [1, 2],
        [0.60, 0.75, 1.01],
        [3, 4],
        [5, 7],
        [80.0, 300.0],
    ):
        rows.append(
            {
                "prob_min": prob_min,
                "tickets_per_race": tickets_per_race,
                "difficulty_max": difficulty_max,
                "partner_order_max": partner_order_max,
                "partner_rank_max": partner_rank_max,
                "partner_odds_max": partner_odds_max,
            }
        )
    return rows


def _choose_policy(train: pd.DataFrame, stake_yen: int, min_train_races: int, min_race_hit_rate: float) -> tuple[dict | None, dict | None]:
    best_params: dict | None = None
    best_metrics: dict | None = None
    best_score = -np.inf
    for params in _policy_grid(train):
        selected = _select_by_policy(train, params, stake_yen)
        metrics = _metrics(selected, "train_trio_model")
        if metrics["races"] < min_train_races or metrics["race_hit_rate"] < min_race_hit_rate:
            continue
        score = (
            (metrics["roi"] - 1.0) * 100.0
            + metrics["race_hit_rate"] * 18.0
            + np.log1p(metrics["races"]) * 0.35
            - max(0.0, abs(metrics["max_drawdown_yen"]) / 10000.0) * 0.20
            - max(0.0, metrics["avg_stake_per_race"] - 300.0) / 1200.0
        )
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics
    return best_params, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a separate 3-renpuku hit-probability model and backtest as an addon.")
    parser.add_argument("--scored-csv", default="outputs/analysis/risk_models_v1/investment_features_with_risk_models.csv")
    parser.add_argument("--base-tickets-csv", default="outputs/analysis/operational_win_addon_1pt_v1/combined_ticket_profiles.csv")
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--race-col", default="レースID(新/馬番無)")
    parser.add_argument("--encoding", default="cp932")
    parser.add_argument("--output-dir", default="outputs/analysis/operational_trio_hit_model_v1")
    parser.add_argument("--stake-yen", type=int, default=100)
    parser.add_argument("--min-train-races", type=int, default=50)
    parser.add_argument("--min-race-hit-rate", type=float, default=0.035)
    parser.add_argument("--positive-payoff-weight", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(project_path(args.output_dir))
    trio_payoffs = _load_trio_payoffs(project_path(args.raw_csv), args.race_col, args.encoding)
    scored = pd.read_csv(project_path(args.scored_csv), dtype={"race_id": str}, low_memory=False)
    universe = _feature_frame(_candidate_universe(_prepare_scored(scored, trio_payoffs)))
    base = _load_base_tickets(project_path(args.base_tickets_csv))

    years = sorted(int(y) for y in universe["year"].dropna().unique())
    wf_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    coefficient_rows: list[dict] = []

    for year in years[1:]:
        train = universe[universe["year"].lt(year)].copy()
        test = universe[universe["year"].eq(year)].copy()
        if train.empty or test.empty or int(train["hit"].sum()) == 0:
            wf_rows.append({"year": year, "selected": False, "reason": "insufficient_data"})
            continue
        model = _fit_logistic(train, positive_payoff_weight=args.positive_payoff_weight)
        train["trio_model_prob"] = _predict(model, train)
        test["trio_model_prob"] = _predict(model, test)
        params, train_metrics = _choose_policy(train, args.stake_yen, args.min_train_races, args.min_race_hit_rate)
        if params is None:
            wf_rows.append({"year": year, "selected": False, "reason": "no_policy"})
            continue
        test_selected = _select_by_policy(test, params, args.stake_yen)
        test_metrics = _metrics(test_selected, f"test_{year}_trio_model")
        ticket_frames.append(test_selected.assign(test_year=year))
        for name, weight in zip(["intercept", *FEATURES], model["weights"]):
            coefficient_rows.append({"test_year": year, "feature": name, "weight": float(weight)})
        wf_rows.append(
            {
                "year": year,
                "selected": True,
                "train_candidates": int(len(train)),
                "train_hit_rate_all_candidates": float(train["hit"].mean()),
                "test_candidates": int(len(test)),
                "test_hit_rate_all_candidates": float(test["hit"].mean()),
                **{f"param_{k}": v for k, v in params.items()},
                **{f"train_{k}": v for k, v in (train_metrics or {}).items() if k != "policy"},
                **{f"test_{k}": v for k, v in test_metrics.items() if k != "policy"},
            }
        )

    trio_tickets = pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame()
    test_years = sorted(trio_tickets["year"].dropna().astype(int).unique()) if not trio_tickets.empty else years[1:]
    base_test = base[base["year"].isin(test_years)].copy() if test_years else base.iloc[0:0].copy()
    combined = pd.concat([base_test, trio_tickets], ignore_index=True, sort=False)
    summary = {
        "input": {
            "scored_csv": args.scored_csv,
            "base_tickets_csv": args.base_tickets_csv,
            "raw_csv": args.raw_csv,
            "stake_yen": args.stake_yen,
            "test_years": test_years,
            "model": "custom_l2_logistic_no_sklearn",
            "positive_payoff_weight": args.positive_payoff_weight,
            "features": FEATURES,
            "note": "Selection uses pre-race candidate features and predicted hit probability. Actual trio payoff is used only for backtest returns.",
        },
        "base_operational": _metrics(base_test, "base_operational"),
        "trio_model": _metrics(trio_tickets, "trio_model"),
        "combined": _metrics(combined, "base_plus_trio_model"),
    }
    summary["delta_roi"] = summary["combined"]["roi"] - summary["base_operational"]["roi"]
    summary["delta_profit_yen"] = summary["combined"]["profit_yen"] - summary["base_operational"]["profit_yen"]

    pd.DataFrame(wf_rows).to_csv(out_dir / "walkforward_trio_model_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefficient_rows).to_csv(out_dir / "trio_model_coefficients.csv", index=False, encoding="utf-8-sig")
    trio_tickets.to_csv(out_dir / "trio_model_tickets.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(out_dir / "combined_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary["base_operational"], summary["trio_model"], summary["combined"]]).to_csv(
        out_dir / "metrics.csv", index=False, encoding="utf-8-sig"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
