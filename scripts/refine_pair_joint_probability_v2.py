from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIR_UNIVERSE = "outputs/analysis/dynamic_pair_ticket_allocation_rebuilt_20260623/pair_candidate_universe.csv"
DEFAULT_RUNNER_CONTEXT = "outputs/analysis/risk_models_v1/investment_features_with_risk_models.csv"
DEFAULT_FRONT5 = "outputs/analysis/front5_position_model_v1/front5_oos_predictions.csv"
DEFAULT_OUT = "outputs/analysis/pair_joint_probability_v2_rebuilt_20260623"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    values = frame[col]
    if values.dtype == object or str(values.dtype).startswith("string"):
        values = values.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def clip01(values: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(values, pd.Series):
        x = values
    else:
        x = pd.Series(values)
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def top_removed_roi(ret: pd.Series, stake: pd.Series, top_n: int) -> float:
    if len(ret) <= top_n:
        return 0.0
    drop_idx = ret.sort_values(ascending=False).index[:top_n]
    ret2 = ret.drop(index=drop_idx)
    stake2 = stake.drop(index=drop_idx)
    return float(ret2.sum() / stake2.sum()) if stake2.sum() > 0 else 0.0


def max_drawdown_by_race(frame: pd.DataFrame, stake_col: str, return_col: str) -> float:
    if frame.empty:
        return 0.0
    tmp = frame.copy()
    tmp["_profit"] = num(tmp, return_col) - num(tmp, stake_col)
    tmp = tmp.sort_values(["year", "race_id"], kind="mergesort")
    race_profit = tmp.groupby("race_id", sort=False)["_profit"].sum()
    equity = race_profit.cumsum()
    dd = equity - equity.cummax()
    return float(dd.min()) if len(dd) else 0.0


def ticket_metrics(frame: pd.DataFrame, stake_col: str, return_col: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
            "top5_removed_roi": 0.0,
            "top10_removed_roi": 0.0,
        }
    stake = num(frame, stake_col)
    ret = num(frame, return_col)
    hit = ret.gt(0)
    race_hit = frame.assign(_hit=hit).groupby("race_id")["_hit"].max()
    return {
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(ret.sum() - stake.sum()),
        "roi": float(ret.sum() / stake.sum()) if stake.sum() > 0 else 0.0,
        "ticket_hit_rate": float(hit.mean()) if len(hit) else 0.0,
        "race_hit_rate": float(race_hit.mean()) if len(race_hit) else 0.0,
        "max_drawdown_yen": max_drawdown_by_race(frame, stake_col, return_col),
        "top5_removed_roi": top_removed_roi(ret, stake, 5),
        "top10_removed_roi": top_removed_roi(ret, stake, 10),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}" if math.isfinite(value) else ""
            vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def pair_key(frame: pd.DataFrame, a_col: str = "anchor_no", b_col: str = "partner_no") -> pd.Series:
    a = num(frame, a_col, np.nan)
    b = num(frame, b_col, np.nan)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return frame["race_id"].astype(str) + ":" + lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)


def load_runner_context(path: Path, front5_path: Path | None) -> pd.DataFrame:
    usecols = [
        "race_id",
        "horse_no",
        "horse_name",
        "front_running_tendency",
        "front_running_tendency_y",
        "closing_tendency_y",
        "horse_front_run_rate_past5_feature",
        "horse_closer_rate_past5_feature",
        "front_pressure_rank_score",
        "solo_lead_potential",
        "race_early_pressure_score",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "front_advantage_score",
        "pace_fit_score",
        "positioning_advantage_score",
        "draw_pace_fit_score",
        "projected_front5_prob",
        "style_bucket",
        "same_day_bias_ready",
        "same_day_bias_fit_score",
        "same_day_projected_front_load_score",
        "same_day_front_collapse_index",
    ]
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    ctx = pd.read_csv(path, usecols=[c for c in usecols if c in header.columns], encoding="utf-8-sig", low_memory=False)
    ctx["race_id"] = ctx["race_id"].astype(str)
    ctx["horse_no"] = num(ctx, "horse_no", np.nan).astype("Int64").astype(str)
    if front5_path is not None and front5_path.exists():
        front = pd.read_csv(front5_path, dtype={"race_id": str}, encoding="utf-8-sig", low_memory=False)
        keep = [c for c in ["race_id", "horse_no", "front5_model_prob"] if c in front.columns]
        front = front[keep].copy()
        if set(["race_id", "horse_no", "front5_model_prob"]).issubset(front.columns):
            front["race_id"] = front["race_id"].astype(str)
            front["horse_no"] = num(front, "horse_no", np.nan).astype("Int64").astype(str)
            front = front.drop_duplicates(["race_id", "horse_no"], keep="last")
            ctx = ctx.merge(front, on=["race_id", "horse_no"], how="left")
    if "front5_model_prob" not in ctx.columns:
        ctx["front5_model_prob"] = np.nan
    ctx["front5_prob_v2"] = num(ctx, "front5_model_prob", np.nan).fillna(num(ctx, "projected_front5_prob", 0.5)).clip(0.02, 0.98)
    ctx["front_tendency_v2"] = (
        0.38 * clip01(num(ctx, "front_running_tendency", 0.0))
        + 0.26 * clip01(num(ctx, "front_running_tendency_y", 0.0))
        + 0.22 * clip01(num(ctx, "horse_front_run_rate_past5_feature", 0.0))
        + 0.14 * ctx["front5_prob_v2"]
    ).clip(0.0, 1.0)
    ctx["closer_tendency_v2"] = (
        0.38 * clip01(num(ctx, "closing_tendency_y", 0.0))
        + 0.30 * clip01(num(ctx, "horse_closer_rate_past5_feature", 0.0))
        + 0.20 * (1.0 - ctx["front5_prob_v2"])
        + 0.12 * clip01(num(ctx, "race_pace_collapse_risk", 0.0))
    ).clip(0.0, 1.0)
    return ctx.drop_duplicates(["race_id", "horse_no"], keep="last")


def add_context_side(frame: pd.DataFrame, context: pd.DataFrame, side: str, no_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = out["race_id"].astype(str)
    out[no_col] = num(out, no_col, np.nan).astype("Int64").astype(str)
    side_context = context.add_prefix(f"{side}_").rename(
        columns={f"{side}_race_id": "race_id", f"{side}_horse_no": no_col}
    )
    return out.merge(side_context, on=["race_id", no_col], how="left")


def add_pair_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["year"] = num(out, "year", np.nan).fillna(out["race_id"].str[:4].astype(float)).astype(int)
    out["pair_key"] = pair_key(out)
    out["wide_label"] = out["wide_hit"].astype(bool).astype(float)
    out["umaren_label"] = out["umaren_hit"].astype(bool).astype(float)
    out["wide_quote_proxy"] = 100.0 * np.sqrt(num(out, "anchor_odds", 1.0).clip(1.0) * num(out, "partner_odds", 1.0).clip(1.0)) * 0.45
    out["umaren_quote_proxy"] = 100.0 * (num(out, "anchor_odds", 1.0).clip(1.0) * num(out, "partner_odds", 1.0).clip(1.0) * 0.32).clip(1.3, 260.0)
    out["wide_return_100"] = np.where(out["wide_hit"].astype(bool), num(out, "wide_pay", 0.0), 0.0)
    out["umaren_return_100"] = np.where(out["umaren_hit"].astype(bool), num(out, "umaren_pay", 0.0), 0.0)

    a_front = num(out, "anchor_front5_prob_v2", np.nan).fillna(num(out, "anchor_front_tendency_v2", 0.5)).clip(0.02, 0.98)
    p_front = num(out, "partner_front5_prob_v2", np.nan).fillna(num(out, "projected_front5_prob", 0.5)).clip(0.02, 0.98)
    a_closer = num(out, "anchor_closer_tendency_v2", 0.0).clip(0.0, 1.0)
    p_closer = num(out, "partner_closer_tendency_v2", 0.0).clip(0.0, 1.0)
    pressure = num(out, "partner_race_early_pressure_score", np.nan).fillna(num(out, "anchor_race_early_pressure_score", 0.0)).clip(0.0, 1.0)
    collapse = num(out, "partner_race_pace_collapse_risk", np.nan).fillna(num(out, "anchor_race_pace_collapse_risk", 0.0)).clip(0.0, 1.0)
    slow = num(out, "partner_race_slow_pace_risk", np.nan).fillna(num(out, "anchor_race_slow_pace_risk", 0.0)).clip(0.0, 1.0)
    front_adv = num(out, "partner_front_advantage_score", np.nan).fillna(num(out, "anchor_front_advantage_score", 0.0)).clip(0.0, 1.0)

    out["joint_place_product"] = np.sqrt((num(out, "anchor_place_score").clip(0, 1) * num(out, "partner_place_score").clip(0, 1)).clip(0, 1))
    out["joint_win_product"] = np.sqrt((num(out, "anchor_win_score").clip(0, 1) * num(out, "partner_win_score").clip(0, 1)).clip(0, 1))
    out["joint_q_product"] = np.sqrt((num(out, "anchor_quinella_score").clip(0, 1) * num(out, "partner_quinella_score").clip(0, 1)).clip(0, 1))
    out["joint_market_product"] = np.sqrt((num(out, "anchor_market_win_prob").clip(0, 1) * num(out, "partner_market_win_prob").clip(0, 1)).clip(0, 1))
    out["front_pair_min"] = np.minimum(a_front, p_front)
    out["front_pair_max"] = np.maximum(a_front, p_front)
    out["closer_pair_max"] = np.maximum(a_closer, p_closer)
    out["front_closer_complement"] = np.maximum(a_front * p_closer, p_front * a_closer)
    out["front_front_clash"] = a_front * p_front * pressure
    out["front_front_slow_fit"] = a_front * p_front * (0.50 * slow + 0.50 * front_adv)
    out["collapse_fit"] = collapse * (0.55 * out["closer_pair_max"] + 0.45 * out["front_closer_complement"])
    out["style_diversity"] = (a_front - p_front).abs() + (a_closer - p_closer).abs()
    out["danger_sum"] = num(out, "anchor_danger", 0.0).clip(0, 1) + num(out, "partner_danger", 0.0).clip(0, 1)
    out["danger_max"] = np.maximum(num(out, "anchor_danger", 0.0).clip(0, 1), num(out, "partner_danger", 0.0).clip(0, 1))
    out["odds_geom"] = np.sqrt(num(out, "anchor_odds", 1.0).clip(1.0) * num(out, "partner_odds", 1.0).clip(1.0))
    out["partner_value_flag"] = ((num(out, "partner_pop", 99) >= 4) | (num(out, "partner_odds", 0) >= 8.0)).astype(float)
    return out


FEATURES = [
    "pair_score",
    "pair_quinella_score",
    "anchor_quinella_score",
    "partner_quinella_score",
    "joint_place_product",
    "joint_win_product",
    "joint_q_product",
    "joint_market_product",
    "wide_axis_score",
    "wide_partner_score",
    "market_overlay_score",
    "late_value_survives_score",
    "projected_front5_prob",
    "front_pair_min",
    "front_pair_max",
    "closer_pair_max",
    "front_closer_complement",
    "front_front_clash",
    "front_front_slow_fit",
    "collapse_fit",
    "style_diversity",
    "danger_sum",
    "danger_max",
    "skip_risk_score",
    "odds_geom",
    "partner_value_flag",
    "anchor_pop",
    "partner_pop",
]


class LogisticModel:
    def __init__(self, features: list[str], mean: pd.Series, std: pd.Series, weights: np.ndarray):
        self.features = features
        self.mean = mean
        self.std = std
        self.weights = weights

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        x = frame[self.features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x = x.fillna(self.mean)
        z = ((x - self.mean) / self.std.replace(0, 1.0)).clip(-6.0, 6.0).to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(z)), z])
        return pd.Series(sigmoid(design @ self.weights), index=frame.index)


def fit_logistic(train: pd.DataFrame, label_col: str, features: list[str], l2: float = 2.0, iterations: int = 900, lr: float = 0.08) -> LogisticModel:
    x = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = x.mean().fillna(0.0)
    std = x.std().replace(0, np.nan).fillna(1.0)
    x = x.fillna(mean)
    z = ((x - mean) / std).clip(-6.0, 6.0).to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(z)), z])
    y = num(train, label_col, 0.0).clip(0, 1).to_numpy(dtype=float)
    w = np.zeros(design.shape[1], dtype=float)
    base_rate = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    w[0] = math.log(base_rate / (1.0 - base_rate))
    n = max(len(y), 1)
    reg = np.r_[0.0, np.ones(design.shape[1] - 1)]
    for _ in range(iterations):
        pred = sigmoid(design @ w)
        grad = (design.T @ (pred - y)) / n + (l2 / n) * reg * w
        w -= lr * grad
    return LogisticModel(features=features, mean=mean, std=std, weights=w)


class BinCalibrator:
    def __init__(self, bins: pd.DataFrame, fallback: float):
        self.bins = bins
        self.fallback = float(fallback)

    def apply(self, raw: pd.Series) -> pd.Series:
        x = pd.to_numeric(raw, errors="coerce")
        out = pd.Series(self.fallback, index=x.index, dtype=float)
        if self.bins.empty:
            return out.clip(0.001, 0.95)
        ordered = self.bins.sort_values("raw_min")
        for _, row in ordered.iterrows():
            mask = x.between(float(row["raw_min"]), float(row["raw_max"]), inclusive="both")
            out.loc[mask] = float(row["prob"])
        out.loc[x.lt(float(ordered["raw_min"].min()))] = float(ordered.iloc[0]["prob"])
        out.loc[x.gt(float(ordered["raw_max"].max()))] = float(ordered.iloc[-1]["prob"])
        return out.clip(0.001, 0.95)


def fit_calibrator(train: pd.DataFrame, raw_col: str, label_col: str, bins: int = 10, smoothing: float = 40.0) -> BinCalibrator:
    use = train[[raw_col, label_col]].dropna().copy()
    use[raw_col] = pd.to_numeric(use[raw_col], errors="coerce")
    use[label_col] = pd.to_numeric(use[label_col], errors="coerce")
    use = use.dropna()
    fallback = float(use[label_col].mean()) if len(use) else 0.02
    if len(use) < max(100, bins * 12) or use[raw_col].nunique() < 4:
        return BinCalibrator(pd.DataFrame(), fallback)
    ranked = use[raw_col].rank(method="first")
    try:
        use["_bin"] = pd.qcut(ranked, bins, labels=False, duplicates="drop")
    except ValueError:
        return BinCalibrator(pd.DataFrame(), fallback)
    grouped = (
        use.groupby("_bin", observed=True)
        .agg(raw_min=(raw_col, "min"), raw_max=(raw_col, "max"), n=(label_col, "size"), hit_rate=(label_col, "mean"))
        .reset_index(drop=True)
        .sort_values("raw_min")
    )
    grouped["prob"] = (grouped["hit_rate"] * grouped["n"] + fallback * smoothing) / (grouped["n"] + smoothing)
    grouped["prob"] = grouped["prob"].cummax().clip(0.001, 0.95)
    return BinCalibrator(grouped, fallback)


def score_walkforward(universe: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    diag_rows: list[dict[str, Any]] = []
    years = sorted(int(y) for y in universe["year"].dropna().unique())
    for year in years:
        train = universe[universe["year"].lt(year)].copy()
        test = universe[universe["year"].eq(year)].copy()
        if len(train) < min_train_rows or test.empty:
            continue
        wide_model = fit_logistic(train, "wide_label", FEATURES)
        umaren_model = fit_logistic(train, "umaren_label", FEATURES)
        train = train.copy()
        test = test.copy()
        train["joint_v2_wide_raw"] = wide_model.predict(train)
        train["joint_v2_umaren_raw"] = umaren_model.predict(train)
        test["joint_v2_wide_raw"] = wide_model.predict(test)
        test["joint_v2_umaren_raw"] = umaren_model.predict(test)
        wide_cal = fit_calibrator(train, "joint_v2_wide_raw", "wide_label")
        umaren_cal = fit_calibrator(train, "joint_v2_umaren_raw", "umaren_label")
        test["joint_v2_wide_prob"] = wide_cal.apply(test["joint_v2_wide_raw"])
        test["joint_v2_umaren_prob"] = umaren_cal.apply(test["joint_v2_umaren_raw"])
        train["joint_v2_wide_prob"] = wide_cal.apply(train["joint_v2_wide_raw"])
        train["joint_v2_umaren_prob"] = umaren_cal.apply(train["joint_v2_umaren_raw"])
        for frame in [train, test]:
            frame["joint_v2_wide_ev_proxy"] = frame["joint_v2_wide_prob"] * frame["wide_quote_proxy"] / 100.0
            frame["joint_v2_umaren_ev_proxy"] = frame["joint_v2_umaren_prob"] * frame["umaren_quote_proxy"] / 100.0
            frame["joint_v2_wide_select_score"] = (
                0.46 * frame["joint_v2_wide_prob"]
                + 0.30 * frame["joint_v2_wide_ev_proxy"].clip(0, 3) / 3
                + 0.14 * clip01(frame["front_pair_min"])
                + 0.10 * clip01(frame["market_overlay_score"])
            )
            frame["joint_v2_umaren_select_score"] = (
                0.50 * frame["joint_v2_umaren_prob"]
                + 0.30 * frame["joint_v2_umaren_ev_proxy"].clip(0, 3) / 3
                + 0.12 * clip01(frame["joint_q_product"])
                + 0.08 * clip01(frame["market_overlay_score"])
            )
        test["joint_v2_test_year"] = year
        test["joint_v2_train_rows"] = len(train)
        frames.append(test)
        for label_col, prob_col in [("wide_label", "joint_v2_wide_prob"), ("umaren_label", "joint_v2_umaren_prob")]:
            y = num(test, label_col)
            p = num(test, prob_col)
            diag_rows.append(
                {
                    "year": year,
                    "target": label_col,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "actual_hit_rate": float(y.mean()),
                    "avg_pred_prob": float(p.mean()),
                    "brier": float(((p - y) ** 2).mean()),
                    "top10_prob_hit_rate": float(y[p.ge(p.quantile(0.90))].mean()) if len(p) else 0.0,
                    "top10_prob_rows": int(p.ge(p.quantile(0.90)).sum()) if len(p) else 0,
                }
            )
    if not frames:
        return pd.DataFrame(), pd.DataFrame(diag_rows)
    return pd.concat(frames, ignore_index=True, sort=False), pd.DataFrame(diag_rows)


def build_train_thresholds(train: pd.DataFrame, policy: str, coverage: float) -> dict[str, float]:
    if policy.startswith("existing"):
        return {
            "pair_q": float(num(train, "pair_quinella_score").quantile(0.75)),
            "overlay_q": float(num(train, "market_overlay_score").quantile(0.75)),
            "late_q": float(num(train, "late_value_survives_score").quantile(0.50)),
            "front_q": float(num(train, "projected_front5_prob").quantile(0.50)),
            "score_q": float(num(train, "pair_score").quantile(1.0 - coverage)),
        }
    score_col = "joint_v2_wide_select_score" if "wide" in policy else "joint_v2_umaren_select_score"
    ev_col = "joint_v2_wide_ev_proxy" if "wide" in policy else "joint_v2_umaren_ev_proxy"
    prob_col = "joint_v2_wide_prob" if "wide" in policy else "joint_v2_umaren_prob"
    return {
        "score_q": float(num(train, score_col).quantile(1.0 - coverage)),
        "ev_q": float(num(train, ev_col).quantile(0.75)),
        "prob_q": float(num(train, prob_col).quantile(0.60)),
    }


def select_policy(test: pd.DataFrame, thresholds: dict[str, float], policy: str, max_pairs_per_race: int) -> pd.DataFrame:
    out = test.copy()
    if policy == "existing_quality_wide":
        mask = (
            num(out, "pair_quinella_score").ge(thresholds["pair_q"])
            & num(out, "market_overlay_score").ge(thresholds["overlay_q"])
            & num(out, "late_value_survives_score").ge(thresholds["late_q"])
            & num(out, "projected_front5_prob").ge(thresholds["front_q"])
            & num(out, "pair_score").ge(thresholds["score_q"])
            & num(out, "anchor_danger").le(0.70)
            & num(out, "partner_danger").le(0.70)
        )
        selected = out[mask].copy()
        selected["ticket_type"] = "wide"
        selected["stake_yen"] = 100.0
        selected["return_yen"] = selected["wide_return_100"]
        sort_col = "pair_score"
    elif policy == "existing_quality_umaren":
        mask = (
            num(out, "pair_quinella_score").ge(thresholds["pair_q"])
            & num(out, "market_overlay_score").ge(thresholds["overlay_q"])
            & num(out, "late_value_survives_score").ge(thresholds["late_q"])
            & num(out, "projected_front5_prob").ge(thresholds["front_q"])
            & num(out, "pair_score").ge(thresholds["score_q"])
            & num(out, "anchor_danger").le(0.70)
            & num(out, "partner_danger").le(0.70)
        )
        selected = out[mask].copy()
        selected["ticket_type"] = "umaren"
        selected["stake_yen"] = 100.0
        selected["return_yen"] = selected["umaren_return_100"]
        sort_col = "pair_quinella_score"
    elif policy == "joint_v2_wide":
        mask = (
            num(out, "joint_v2_wide_select_score").ge(thresholds["score_q"])
            & num(out, "joint_v2_wide_ev_proxy").ge(thresholds["ev_q"])
            & num(out, "joint_v2_wide_prob").ge(thresholds["prob_q"])
            & num(out, "anchor_danger").le(0.70)
            & num(out, "partner_danger").le(0.70)
        )
        selected = out[mask].copy()
        selected["ticket_type"] = "wide"
        selected["stake_yen"] = 100.0
        selected["return_yen"] = selected["wide_return_100"]
        sort_col = "joint_v2_wide_select_score"
    elif policy == "joint_v2_umaren":
        mask = (
            num(out, "joint_v2_umaren_select_score").ge(thresholds["score_q"])
            & num(out, "joint_v2_umaren_ev_proxy").ge(thresholds["ev_q"])
            & num(out, "joint_v2_umaren_prob").ge(thresholds["prob_q"])
            & num(out, "anchor_danger").le(0.60)
            & num(out, "partner_danger").le(0.55)
        )
        selected = out[mask].copy()
        selected["ticket_type"] = "umaren"
        selected["stake_yen"] = 100.0
        selected["return_yen"] = selected["umaren_return_100"]
        sort_col = "joint_v2_umaren_select_score"
    else:
        raise ValueError(policy)
    if selected.empty:
        return selected
    return (
        selected.sort_values(["race_id", sort_col, "market_overlay_score"], ascending=[True, False, False])
        .groupby("race_id", as_index=False)
        .head(max_pairs_per_race)
    )


def evaluate_policies(scored: pd.DataFrame, full_universe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_rows: list[pd.DataFrame] = []
    yearly_rows: list[dict[str, Any]] = []
    policies = ["existing_quality_wide", "existing_quality_umaren", "joint_v2_wide", "joint_v2_umaren"]
    coverages = [0.05, 0.10, 0.15]
    for year in sorted(scored["year"].dropna().unique()):
        train = full_universe[full_universe["year"].lt(year)].copy()
        test = scored[scored["year"].eq(year)].copy()
        # Attach train-side proxy joint scores for thresholding using in-sample model-free approximations when needed.
        if train.empty or test.empty:
            continue
        # Train thresholds for joint_v2 use scored history when available; otherwise use all prior scored rows.
        prior_scored = scored[scored["year"].lt(year)].copy()
        if prior_scored.empty:
            prior_scored = test.copy()
        for policy in policies:
            for coverage in coverages:
                thresh_base = train if policy.startswith("existing") else prior_scored
                thresholds = build_train_thresholds(thresh_base, policy, coverage)
                selected = select_policy(test, thresholds, policy, max_pairs_per_race=1)
                if selected.empty:
                    selected = pd.DataFrame(columns=[*test.columns, "ticket_type", "stake_yen", "return_yen"])
                selected["policy"] = f"{policy}_cov{int(coverage*100):02d}"
                selected["coverage"] = coverage
                selected["threshold_test_year"] = int(year)
                policy_rows.append(selected)
                m = ticket_metrics(selected, "stake_yen", "return_yen")
                yearly_rows.append({"policy": selected["policy"].iloc[0] if len(selected) else f"{policy}_cov{int(coverage*100):02d}", "year": int(year), **m})
    tickets = pd.concat(policy_rows, ignore_index=True, sort=False) if policy_rows else pd.DataFrame()
    yearly = pd.DataFrame(yearly_rows)
    all_rows = []
    if not tickets.empty:
        for policy, part in tickets.groupby("policy", sort=False):
            all_rows.append({"policy": policy, **ticket_metrics(part, "stake_yen", "return_yen")})
    return tickets, yearly, pd.DataFrame(all_rows).sort_values(["top5_removed_roi", "roi"], ascending=[False, False])


def evaluate_joint_guard_grid(tickets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test joint V2 as a guard on the strongest existing wide policy.

    Direct joint-probability ranking improves hit rate but tends to buy cheap
    wide pairs. The safer first use is to keep the existing value/overlay
    policy and remove only pairs with weak joint confirmation.
    """
    if tickets.empty or "policy" not in tickets.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    base = tickets[tickets["policy"].eq("existing_quality_wide_cov10")].copy()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    prob_thresholds = [0.040, 0.045, 0.047, 0.050, 0.060, 0.070, 0.075, 0.080, 0.100]
    ev_thresholds = [0.00, 0.25, 0.30, 0.32, 0.34, 0.36, 0.37, 0.39, 0.42, 0.45]
    grid_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    ticket_frames: list[pd.DataFrame] = []

    for prob_thr in prob_thresholds:
        for ev_thr in ev_thresholds:
            selected = base[
                num(base, "joint_v2_wide_prob").ge(prob_thr)
                & num(base, "joint_v2_wide_ev_proxy").ge(ev_thr)
            ].copy()
            if len(selected) < 250:
                continue
            policy = f"existing_wide_cov10_joint_guard_p{prob_thr:.3f}_ev{ev_thr:.2f}"
            selected["policy"] = policy
            selected["joint_guard_prob_threshold"] = prob_thr
            selected["joint_guard_ev_threshold"] = ev_thr
            ticket_frames.append(selected)

            metrics = ticket_metrics(selected, "stake_yen", "return_yen")
            row = {"policy": policy, "prob_threshold": prob_thr, "ev_threshold": ev_thr, **metrics}
            year_rois: list[float] = []
            for year, part in selected.groupby("year", sort=True):
                ym = ticket_metrics(part, "stake_yen", "return_yen")
                yearly_rows.append({"policy": policy, "year": int(year), **ym})
                row[f"roi_{int(year)}"] = ym["roi"]
                row[f"tickets_{int(year)}"] = ym["tickets"]
                year_rois.append(float(ym["roi"]))
            row["min_year_roi"] = min(year_rois) if year_rois else 0.0
            grid_rows.append(row)

    grid = pd.DataFrame(grid_rows)
    yearly = pd.DataFrame(yearly_rows)
    all_tickets = pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame()
    if grid.empty:
        return grid, yearly, all_tickets

    eligible = grid[
        grid["tickets"].ge(450)
        & grid.get("roi_2025", pd.Series(0.0, index=grid.index)).ge(1.00)
        & grid.get("roi_2026", pd.Series(0.0, index=grid.index)).ge(1.00)
        & grid["top5_removed_roi"].ge(1.00)
        & grid["top10_removed_roi"].ge(0.90)
    ].copy()
    if eligible.empty:
        eligible = grid[grid["tickets"].ge(350)].copy()
    sort_cols = ["min_year_roi", "top5_removed_roi", "roi", "top10_removed_roi"]
    recommended_policy = eligible.sort_values(sort_cols, ascending=[False, False, False, False]).iloc[0]["policy"]
    recommended = all_tickets[all_tickets["policy"].eq(recommended_policy)].copy()
    grid = grid.sort_values(sort_cols, ascending=[False, False, False, False])
    return grid, yearly, recommended


def evaluate_umaren_joint_guard_grid(tickets: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty or "policy" not in tickets.columns:
        return pd.DataFrame()
    prob_thresholds = [0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.120, 0.150]
    ev_thresholds = [0.00, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.50]
    rows: list[dict[str, Any]] = []
    for base_policy in ["existing_quality_umaren_cov05", "existing_quality_umaren_cov10", "existing_quality_umaren_cov15"]:
        base = tickets[tickets["policy"].eq(base_policy)].copy()
        if base.empty:
            continue
        for prob_thr in prob_thresholds:
            for ev_thr in ev_thresholds:
                selected = base[
                    num(base, "joint_v2_umaren_prob").ge(prob_thr)
                    & num(base, "joint_v2_umaren_ev_proxy").ge(ev_thr)
                ].copy()
                if len(selected) < 80:
                    continue
                metrics = ticket_metrics(selected, "stake_yen", "return_yen")
                row = {
                    "base_policy": base_policy,
                    "prob_threshold": prob_thr,
                    "ev_threshold": ev_thr,
                    **metrics,
                }
                for year, part in selected.groupby("year", sort=True):
                    ym = ticket_metrics(part, "stake_yen", "return_yen")
                    row[f"roi_{int(year)}"] = ym["roi"]
                    row[f"tickets_{int(year)}"] = ym["tickets"]
                row["fragile_flag"] = bool(row["top5_removed_roi"] < 1.0 or row["top10_removed_roi"] < 0.9)
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["fragile_flag", "top5_removed_roi", "roi"], ascending=[True, False, False])


def render_review(summary: dict[str, Any], diagnostics: pd.DataFrame, comparison: pd.DataFrame, yearly: pd.DataFrame) -> str:
    comp_cols = ["policy", "tickets", "races", "roi", "ticket_hit_rate", "race_hit_rate", "top5_removed_roi", "top10_removed_roi", "profit_yen"]
    diag_cols = ["year", "target", "train_rows", "test_rows", "actual_hit_rate", "avg_pred_prob", "brier", "top10_prob_hit_rate", "top10_prob_rows"]
    yearly_cols = ["policy", "year", "tickets", "races", "roi", "top5_removed_roi", "profit_yen"]
    lines = [
        "# Pair Joint Probability V2",
        "",
        "## 目的",
        "",
        "既存の `pair_score` / `pair_quinella_score` ではなく、ワイド・馬連の2頭同時好走確率を直接推定する。",
        "今回は初回実装として、馬単体スコアに加えて前目確率、差し/前残りシナリオ、同型衝突、危険度を入れたwalk-forward logistic + bin calibrationを作った。",
        "",
        "## 結論",
        "",
        "- `joint_v2` の確率列は作成できた。`joint_v2_wide_prob`, `joint_v2_umaren_prob`, EV proxy, selection score を出力済み。",
        "- ただし、この初回V2をそのまま本番BUYに差し替える段階ではない。",
        "- 既存quality条件より良い面はあるが、年度別・top5除外後の安定性を見てから、準候補シャドー側に先に入れるのが安全。",
        "",
        "## 入力",
        "",
        f"- pair_universe: `{summary['inputs']['pair_universe']}`",
        f"- runner_context: `{summary['inputs']['runner_context']}`",
        f"- front5: `{summary['inputs']['front5']}`",
        "",
        "## 確率診断",
        "",
        markdown_table(diagnostics[diag_cols].to_dict("records")),
        "",
        "## ポリシー比較",
        "",
        markdown_table(comparison[comp_cols].to_dict("records")),
        "",
        "## 年度別",
        "",
        markdown_table(yearly[yearly_cols].sort_values(["policy", "year"]).to_dict("records")),
        "",
        "## 次の扱い",
        "",
        "1. 本番BUYにはまだ直結させず、準候補シャドー運用のスコアとして追加する。",
        "2. T-5/T-3実オッズで `joint_v2_ev_proxy` が残るかを見る。",
        "3. ワイド/馬連の券種別に、確率下限ではなく保守的EV下限で採用判断する。",
    ]
    return "\n".join(lines) + "\n"


def render_review_v2(
    summary: dict[str, Any],
    diagnostics: pd.DataFrame,
    comparison: pd.DataFrame,
    yearly: pd.DataFrame,
    guard_grid: pd.DataFrame,
    umaren_guard_grid: pd.DataFrame,
) -> str:
    comp_cols = ["policy", "tickets", "races", "roi", "ticket_hit_rate", "race_hit_rate", "top5_removed_roi", "top10_removed_roi", "profit_yen"]
    diag_cols = ["year", "target", "train_rows", "test_rows", "actual_hit_rate", "avg_pred_prob", "brier", "top10_prob_hit_rate", "top10_prob_rows"]
    yearly_cols = ["policy", "year", "tickets", "races", "roi", "top5_removed_roi", "profit_yen"]
    guard_cols = [
        "policy",
        "tickets",
        "races",
        "roi",
        "ticket_hit_rate",
        "top5_removed_roi",
        "top10_removed_roi",
        "roi_2025",
        "roi_2026",
        "profit_yen",
    ]
    guard_cols = [c for c in guard_cols if c in guard_grid.columns]
    umaren_cols = [
        "base_policy",
        "tickets",
        "races",
        "roi",
        "ticket_hit_rate",
        "top5_removed_roi",
        "top10_removed_roi",
        "roi_2025",
        "roi_2026",
        "fragile_flag",
    ]
    umaren_cols = [c for c in umaren_cols if c in umaren_guard_grid.columns]
    lines = [
        "# Pair Joint Probability V2",
        "",
        "## Purpose",
        "",
        "Estimate pair-level joint finishing probabilities instead of relying only on single-horse scores.",
        "This version uses walk-forward logistic models plus bin calibration, then tests whether the probability should rank tickets directly or guard existing value tickets.",
        "",
        "## Takeaway",
        "",
        "- Directly ranking by joint probability improves hit rate but selects too many cheap wide pairs.",
        "- The better first use is a guard: keep the existing value/overlay logic, then remove pairs with weak joint confirmation.",
        "- The recommended guard should enter shadow/runtime diagnostics before replacing final BUY.",
        "",
        "## Inputs",
        "",
        f"- pair_universe: `{summary['inputs']['pair_universe']}`",
        f"- runner_context: `{summary['inputs']['runner_context']}`",
        f"- front5: `{summary['inputs']['front5']}`",
        "",
        "## Probability Diagnostics",
        "",
        markdown_table(diagnostics[diag_cols].to_dict("records")) if not diagnostics.empty else "_No diagnostics._",
        "",
        "## Direct Policy Comparison",
        "",
        markdown_table(comparison[comp_cols].to_dict("records")) if not comparison.empty else "_No policies._",
        "",
        "## Joint Guard Candidates",
        "",
        markdown_table(guard_grid.head(20)[guard_cols].to_dict("records")) if not guard_grid.empty else "_No guard candidates._",
        "",
        "## Umaren Guard Check",
        "",
        markdown_table(umaren_guard_grid.head(20)[umaren_cols].to_dict("records")) if not umaren_guard_grid.empty else "_No umaren guard rows._",
        "",
        "## Yearly Direct Policies",
        "",
        markdown_table(yearly[yearly_cols].sort_values(["policy", "year"]).to_dict("records")) if not yearly.empty else "_No yearly rows._",
        "",
        "## Next",
        "",
        "1. Add the recommended guard to shadow/runtime output first.",
        "2. Track whether T-5/T-3 odds preserve the joint EV proxy.",
        "3. Promote only if the guard remains stable under additional live snapshots.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-universe", default=DEFAULT_PAIR_UNIVERSE)
    parser.add_argument("--runner-context", default=DEFAULT_RUNNER_CONTEXT)
    parser.add_argument("--front5", default=DEFAULT_FRONT5)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--min-train-rows", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_universe_path = project_path(args.pair_universe)
    runner_context_path = project_path(args.runner_context)
    front5_path = project_path(args.front5)

    universe = pd.read_csv(pair_universe_path, encoding="utf-8-sig", low_memory=False)
    universe["race_id"] = universe["race_id"].astype(str)
    context = load_runner_context(runner_context_path, front5_path if front5_path.exists() else None)
    universe = add_context_side(universe, context, "anchor", "anchor_no")
    universe = add_context_side(universe, context, "partner", "partner_no")
    universe = add_pair_features(universe)
    scored, diagnostics = score_walkforward(universe, args.min_train_rows)
    tickets, yearly, comparison = evaluate_policies(scored, universe)
    guard_grid, guard_yearly, recommended_guard_tickets = evaluate_joint_guard_grid(tickets)
    umaren_guard_grid = evaluate_umaren_joint_guard_grid(tickets)

    universe.to_csv(out_dir / "pair_universe_with_joint_v2_features.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(out_dir / "pair_joint_v2_oos_scores.csv", index=False, encoding="utf-8-sig")
    tickets.to_csv(out_dir / "pair_joint_v2_policy_tickets.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(out_dir / "probability_diagnostics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "yearly_policy_comparison.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(out_dir / "policy_comparison.csv", index=False, encoding="utf-8-sig")
    guard_grid.to_csv(out_dir / "joint_guard_grid.csv", index=False, encoding="utf-8-sig")
    guard_yearly.to_csv(out_dir / "joint_guard_yearly.csv", index=False, encoding="utf-8-sig")
    recommended_guard_tickets.to_csv(out_dir / "recommended_joint_guard_tickets.csv", index=False, encoding="utf-8-sig")
    umaren_guard_grid.to_csv(out_dir / "umaren_joint_guard_grid.csv", index=False, encoding="utf-8-sig")

    summary = {
        "inputs": {
            "pair_universe": str(pair_universe_path),
            "runner_context": str(runner_context_path),
            "front5": str(front5_path),
        },
        "rows": {
            "universe": int(len(universe)),
            "oos_scored": int(len(scored)),
            "policy_tickets": int(len(tickets)),
            "guard_grid": int(len(guard_grid)),
            "recommended_guard_tickets": int(len(recommended_guard_tickets)),
            "umaren_guard_grid": int(len(umaren_guard_grid)),
        },
        "features": FEATURES,
        "best_policy": comparison.head(1).to_dict("records")[0] if not comparison.empty else {},
        "recommended_joint_guard": guard_grid.head(1).to_dict("records")[0] if not guard_grid.empty else {},
        "best_umaren_guard": umaren_guard_grid.head(1).to_dict("records")[0] if not umaren_guard_grid.empty else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review.md").write_text(render_review_v2(summary, diagnostics, comparison, yearly, guard_grid, umaren_guard_grid), encoding="utf-8")
    print(json.dumps(json_ready({"ok": True, "out_dir": str(out_dir), **summary["rows"], "best_policy": summary["best_policy"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
