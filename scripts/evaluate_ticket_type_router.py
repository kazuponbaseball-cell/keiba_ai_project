from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "outputs/analysis/investment_decision_features_rebuilt_20260623/investment_features_scored.csv"
PAIR_PATH = ROOT / "outputs/analysis/dynamic_pair_ticket_allocation_rebuilt_20260623/pair_candidate_universe.csv"
OUT = ROOT / "outputs/analysis/ticket_type_router_v1"


def num(value, default: float = np.nan) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce")
    return pd.Series(default)


def clip01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def norm01(s: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if lo is None:
        lo = float(x.quantile(0.05)) if x.notna().any() else 0.0
    if hi is None:
        hi = float(x.quantile(0.95)) if x.notna().any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=x.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5).clip(0.0, 1.0)


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    equity = profit.cumsum()
    return float((equity - equity.cummax()).min())


@dataclass
class Calibrator:
    feature: str
    label: str
    bins: pd.DataFrame
    fallback: float

    def apply(self, df: pd.DataFrame) -> pd.Series:
        x = pd.to_numeric(df[self.feature], errors="coerce").fillna(0.0)
        out = pd.Series(self.fallback, index=df.index, dtype=float)
        if self.bins.empty:
            return out.clip(0.001, 0.98)
        bins = self.bins.sort_values("raw_min")
        for _, row in bins.iterrows():
            mask = x.between(float(row["raw_min"]), float(row["raw_max"]), inclusive="both")
            out.loc[mask] = float(row["prob"])
        out.loc[x.lt(float(bins["raw_min"].min()))] = float(bins.iloc[0]["prob"])
        out.loc[x.gt(float(bins["raw_max"].max()))] = float(bins.iloc[-1]["prob"])
        return out.clip(0.001, 0.98)


def fit_calibrator(train: pd.DataFrame, feature: str, label: str, bins: int = 10, smoothing: float = 40.0) -> Calibrator:
    use = train[[feature, label]].copy()
    use[feature] = pd.to_numeric(use[feature], errors="coerce").fillna(0.0)
    use[label] = pd.to_numeric(use[label], errors="coerce").fillna(0.0)
    fallback = float(use[label].mean()) if len(use) else 0.01
    if len(use) < max(80, bins * 10) or use[feature].nunique() < 5:
        return Calibrator(feature, label, pd.DataFrame(), fallback)
    try:
        use["_bin"] = pd.qcut(use[feature].rank(method="first"), bins, labels=False, duplicates="drop")
    except ValueError:
        return Calibrator(feature, label, pd.DataFrame(), fallback)
    grouped = (
        use.groupby("_bin", observed=True)
        .agg(raw_min=(feature, "min"), raw_max=(feature, "max"), n=(label, "size"), hit_rate=(label, "mean"))
        .reset_index(drop=True)
        .sort_values("raw_min")
    )
    grouped["prob"] = (grouped["hit_rate"] * grouped["n"] + fallback * smoothing) / (grouped["n"] + smoothing)
    grouped["prob"] = grouped["prob"].cummax()
    return Calibrator(feature, label, grouped, fallback)


def load_runners() -> pd.DataFrame:
    cols = [
        "race_id",
        "日付S",
        "venue",
        "Ｒ",
        "レース名",
        "horse_no",
        "horse_name",
        "ai_rank_num",
        "ai_score",
        "popularity",
        "odds",
        "finish_num",
        "win_return",
        "place_return",
        "is_win",
        "is_place",
        "market_overlay_score",
        "late_value_survives_score",
        "projected_front5_prob",
        "danger_favorite_score",
        "win_suitability_score",
        "place_suitability_score",
        "skip_risk_score",
        "race_early_pressure_score",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "馬場状態",
        "surface",
        "distance",
        "クラス名",
    ]
    df = pd.read_csv(RUNNER_PATH, usecols=lambda c: c in cols, dtype={"race_id": str}, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = df["race_id"].str[:4].astype(int)
    for col in [
        "horse_no",
        "ai_rank_num",
        "ai_score",
        "popularity",
        "odds",
        "finish_num",
        "win_return",
        "place_return",
        "market_overlay_score",
        "late_value_survives_score",
        "projected_front5_prob",
        "danger_favorite_score",
        "win_suitability_score",
        "place_suitability_score",
        "skip_risk_score",
        "race_early_pressure_score",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_win"] = pd.to_numeric(df["is_win"], errors="coerce").fillna(0).astype(float)
    df["is_place"] = pd.to_numeric(df["is_place"], errors="coerce").fillna(0).astype(float)

    ordered = df.sort_values(["race_id", "ai_score"], ascending=[True, False])
    top = ordered.groupby("race_id", as_index=False).head(3).copy()
    top["_rank_by_score"] = top.groupby("race_id")["ai_score"].rank(method="first", ascending=False)
    gap = (
        ordered.groupby("race_id")["ai_score"]
        .apply(lambda s: float(s.iloc[0] - s.iloc[1]) if len(s) > 1 else float(s.iloc[0]))
        .rename("top_gap")
        .reset_index()
    )
    race_difficulty = (
        df.groupby("race_id", as_index=False)
        .agg(
            race_skip=("skip_risk_score", "mean"),
            race_pressure=("race_early_pressure_score", "mean"),
            race_collapse=("race_pace_collapse_risk", "mean"),
            race_slow=("race_slow_pace_risk", "mean"),
        )
    )
    top = top.merge(gap, on="race_id", how="left").merge(race_difficulty, on="race_id", how="left")

    top["win_pay_per100"] = top["win_return"].fillna(0.0)
    missing_win_pay = top["win_pay_per100"].le(0) & top["is_win"].eq(1)
    top.loc[missing_win_pay, "win_pay_per100"] = top.loc[missing_win_pay, "odds"].fillna(0.0) * 100.0
    top["place_pay_per100"] = top["place_return"].fillna(0.0)
    top["win_quote_per100"] = top["odds"].fillna(0.0).clip(lower=1.0) * 100.0
    # Historical pre-race place odds are not in this file, so this is a conservative proxy for selection only.
    top["place_quote_proxy_per100"] = (100.0 + (top["odds"].fillna(1.0).clip(lower=1.0) - 1.0) * 22.0).clip(110.0, 780.0)

    top["win_raw"] = (
        0.34 * norm01(top["ai_score"])
        + 0.22 * clip01(top["win_suitability_score"])
        + 0.18 * clip01(top["market_overlay_score"])
        + 0.12 * norm01(top["top_gap"], lo=0.0, hi=0.18)
        + 0.08 * clip01(top["late_value_survives_score"])
        + 0.06 * (1.0 - clip01(top["danger_favorite_score"]))
    ).clip(0.0, 1.0)
    top["place_raw"] = (
        0.32 * clip01(top["place_suitability_score"])
        + 0.24 * norm01(top["ai_score"])
        + 0.14 * clip01(top["projected_front5_prob"])
        + 0.12 * clip01(top["market_overlay_score"])
        + 0.10 * (1.0 - clip01(top["danger_favorite_score"]))
        + 0.08 * (1.0 - norm01(top["race_skip"], lo=0.20, hi=0.65))
    ).clip(0.0, 1.0)
    return top


def load_pairs() -> pd.DataFrame:
    df = pd.read_csv(PAIR_PATH, dtype={"race_id": str}, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = df["race_id"].str[:4].astype(int)
    for col in [
        "anchor_no",
        "partner_no",
        "anchor_finish",
        "partner_finish",
        "anchor_pop",
        "partner_pop",
        "anchor_odds",
        "partner_odds",
        "anchor_win_score",
        "partner_win_score",
        "anchor_place_score",
        "partner_place_score",
        "wide_axis_score",
        "wide_partner_score",
        "market_overlay_score",
        "late_value_survives_score",
        "projected_front5_prob",
        "anchor_danger",
        "partner_danger",
        "skip_risk_score",
        "pair_score",
        "pair_quinella_score",
        "anchor_quinella_score",
        "partner_quinella_score",
        "wide_pay",
        "umaren_pay",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["wide_hit"] = df["wide_hit"].astype(bool).astype(float)
    df["umaren_hit"] = df["umaren_hit"].astype(bool).astype(float)
    df["wide_quote_proxy_per100"] = (
        100.0 * (np.sqrt(df["anchor_odds"].clip(lower=1.0) * df["partner_odds"].clip(lower=1.0)) * 0.45)
    ).clip(110.0, 12000.0)
    df["umaren_quote_proxy_per100"] = (
        100.0 * (df["anchor_odds"].clip(lower=1.0) * df["partner_odds"].clip(lower=1.0) * 0.32)
    ).clip(130.0, 26000.0)
    place_joint = np.sqrt((clip01(df["anchor_place_score"]) * clip01(df["partner_place_score"])).clip(0.0, 1.0))
    danger_sum = clip01(df["anchor_danger"]) + clip01(df["partner_danger"])
    df["wide_raw"] = (
        0.24 * clip01(df["pair_score"])
        + 0.18 * clip01(df["wide_axis_score"])
        + 0.18 * clip01(df["wide_partner_score"])
        + 0.16 * place_joint
        + 0.10 * clip01(df["projected_front5_prob"])
        + 0.09 * clip01(df["market_overlay_score"])
        + 0.05 * clip01(df["late_value_survives_score"])
        - 0.08 * danger_sum
    ).clip(0.0, 1.0)
    df["umaren_raw"] = (
        0.30 * clip01(df["pair_quinella_score"])
        + 0.19 * clip01(df["anchor_quinella_score"])
        + 0.19 * clip01(df["partner_quinella_score"])
        + 0.11 * clip01(df["anchor_win_score"])
        + 0.08 * clip01(df["partner_win_score"])
        + 0.08 * clip01(df["market_overlay_score"])
        + 0.05 * clip01(df["late_value_survives_score"])
        - 0.10 * danger_sum
    ).clip(0.0, 1.0)
    df["pair_label"] = df["anchor_name"].astype(str) + "-" + df["partner_name"].astype(str)
    return df


def calibrate_and_build_candidates(train_years: Iterable[int], test_year: int, runners: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    train_r = runners[runners["year"].isin(train_years)].copy()
    test_r = runners[runners["year"].eq(test_year)].copy()
    train_p = pairs[pairs["year"].isin(train_years)].copy()
    test_p = pairs[pairs["year"].eq(test_year)].copy()

    win_cal = fit_calibrator(train_r, "win_raw", "is_win", bins=10, smoothing=45.0)
    place_cal = fit_calibrator(train_r, "place_raw", "is_place", bins=10, smoothing=45.0)
    wide_cal = fit_calibrator(train_p, "wide_raw", "wide_hit", bins=10, smoothing=45.0)
    umaren_cal = fit_calibrator(train_p, "umaren_raw", "umaren_hit", bins=10, smoothing=45.0)

    frames: list[pd.DataFrame] = []
    if not test_r.empty:
        r = test_r.copy()
        r["win_prob_cal"] = win_cal.apply(r)
        r["place_prob_cal"] = place_cal.apply(r)
        for ticket_type in ["win", "place"]:
            tmp = r.copy()
            tmp["ticket_type"] = ticket_type
            tmp["ticket_key"] = ticket_type + ":" + tmp["race_id"] + ":" + tmp["horse_no"].astype("Int64").astype(str)
            tmp["name"] = tmp["horse_name"]
            tmp["combo"] = tmp["horse_no"].astype("Int64").astype(str) + " " + tmp["horse_name"].astype(str)
            if ticket_type == "win":
                tmp["hit"] = tmp["is_win"].astype(bool)
                tmp["return_yen"] = tmp["win_pay_per100"].where(tmp["hit"], 0.0)
                tmp["hit_prob"] = tmp["win_prob_cal"]
                tmp["quote_pay_proxy_per100"] = tmp["win_quote_per100"]
                tmp["structure_score"] = (
                    0.34 * norm01(tmp["top_gap"], lo=0.0, hi=0.18)
                    + 0.28 * clip01(tmp["win_suitability_score"])
                    + 0.18 * (1.0 - norm01(tmp["race_skip"], lo=0.20, hi=0.65))
                    + 0.12 * clip01(tmp["market_overlay_score"])
                    + 0.08 * (1.0 - clip01(tmp["danger_favorite_score"]))
                ).clip(0.0, 1.0)
            else:
                tmp["hit"] = tmp["is_place"].astype(bool)
                tmp["return_yen"] = tmp["place_pay_per100"].where(tmp["hit"], 0.0)
                tmp["hit_prob"] = tmp["place_prob_cal"]
                tmp["quote_pay_proxy_per100"] = tmp["place_quote_proxy_per100"]
                tmp["structure_score"] = (
                    0.34 * clip01(tmp["place_suitability_score"])
                    + 0.22 * (1.0 - norm01(tmp["race_skip"], lo=0.20, hi=0.65))
                    + 0.18 * clip01(tmp["projected_front5_prob"])
                    + 0.14 * norm01(tmp["top_gap"], lo=0.0, hi=0.18)
                    + 0.12 * (1.0 - clip01(tmp["danger_favorite_score"]))
                ).clip(0.0, 1.0)
            tmp["quote_odds_proxy"] = tmp["quote_pay_proxy_per100"] / 100.0
            market_prob = (100.0 / tmp["quote_pay_proxy_per100"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            cap = 3.0 if ticket_type == "win" else 2.2
            tmp["hit_prob"] = np.minimum(tmp["hit_prob"], market_prob * cap).clip(0.001, 0.98)
            tmp["expected_roi"] = tmp["hit_prob"] * tmp["quote_pay_proxy_per100"] / 100.0
            tmp["difficulty"] = tmp["race_skip"].fillna(0.5)
            tmp["danger"] = tmp["danger_favorite_score"].fillna(0.0)
            frames.append(tmp[[
                "race_id",
                "year",
                "ticket_type",
                "ticket_key",
                "combo",
                "hit",
                "return_yen",
                "hit_prob",
                "quote_pay_proxy_per100",
                "quote_odds_proxy",
                "expected_roi",
                "structure_score",
                "difficulty",
                "danger",
                "top_gap",
                "market_overlay_score",
                "late_value_survives_score",
            ]])

    if not test_p.empty:
        p = test_p.copy()
        p["wide_prob_cal"] = wide_cal.apply(p)
        p["umaren_prob_cal"] = umaren_cal.apply(p)
        p["difficulty"] = p["skip_risk_score"].fillna(0.5)
        p["danger"] = (clip01(p["anchor_danger"]) + clip01(p["partner_danger"])).clip(0.0, 2.0)
        p["pair_key"] = p["race_id"] + ":" + p["anchor_no"].astype("Int64").astype(str) + "-" + p["partner_no"].astype("Int64").astype(str)
        for ticket_type in ["wide", "umaren"]:
            tmp = p.copy()
            tmp["ticket_type"] = ticket_type
            tmp["ticket_key"] = ticket_type + ":" + tmp["pair_key"]
            tmp["combo"] = (
                tmp["anchor_no"].astype("Int64").astype(str)
                + "-"
                + tmp["partner_no"].astype("Int64").astype(str)
                + " "
                + tmp["pair_label"]
            )
            if ticket_type == "wide":
                tmp["hit"] = tmp["wide_hit"].astype(bool)
                tmp["return_yen"] = tmp["wide_pay"].fillna(0.0).where(tmp["hit"], 0.0)
                tmp["hit_prob"] = tmp["wide_prob_cal"]
                tmp["quote_pay_proxy_per100"] = tmp["wide_quote_proxy_per100"]
                tmp["structure_score"] = (
                    0.30 * clip01(tmp["pair_score"])
                    + 0.22 * clip01(tmp["wide_axis_score"])
                    + 0.22 * clip01(tmp["wide_partner_score"])
                    + 0.14 * clip01(tmp["projected_front5_prob"])
                    + 0.12 * (1.0 - norm01(tmp["difficulty"], lo=0.20, hi=0.65))
                ).clip(0.0, 1.0)
            else:
                tmp["hit"] = tmp["umaren_hit"].astype(bool)
                tmp["return_yen"] = tmp["umaren_pay"].fillna(0.0).where(tmp["hit"], 0.0)
                tmp["hit_prob"] = tmp["umaren_prob_cal"]
                tmp["quote_pay_proxy_per100"] = tmp["umaren_quote_proxy_per100"]
                tmp["structure_score"] = (
                    0.34 * clip01(tmp["pair_quinella_score"])
                    + 0.20 * clip01(tmp["anchor_quinella_score"])
                    + 0.20 * clip01(tmp["partner_quinella_score"])
                    + 0.14 * clip01(tmp["market_overlay_score"])
                    + 0.12 * (1.0 - norm01(tmp["difficulty"], lo=0.20, hi=0.65))
                ).clip(0.0, 1.0)
            tmp["quote_odds_proxy"] = tmp["quote_pay_proxy_per100"] / 100.0
            market_prob = (100.0 / tmp["quote_pay_proxy_per100"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            cap = 3.2 if ticket_type == "wide" else 4.0
            tmp["hit_prob"] = np.minimum(tmp["hit_prob"], market_prob * cap).clip(0.001, 0.98)
            tmp["expected_roi"] = tmp["hit_prob"] * tmp["quote_pay_proxy_per100"] / 100.0
            tmp["top_gap"] = np.nan
            frames.append(tmp[[
                "race_id",
                "year",
                "ticket_type",
                "ticket_key",
                "combo",
                "hit",
                "return_yen",
                "hit_prob",
                "quote_pay_proxy_per100",
                "quote_odds_proxy",
                "expected_roi",
                "structure_score",
                "difficulty",
                "danger",
                "top_gap",
                "market_overlay_score",
                "late_value_survives_score",
            ]])

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["stake_yen"] = 100.0
    out["return_yen"] = pd.to_numeric(out["return_yen"], errors="coerce").fillna(0.0)
    out["hit_prob"] = pd.to_numeric(out["hit_prob"], errors="coerce").fillna(0.0).clip(0.001, 0.98)
    out["expected_roi"] = pd.to_numeric(out["expected_roi"], errors="coerce").fillna(0.0)
    out["router_score"] = (
        out["expected_roi"].clip(0.0, 20.0)
        * np.sqrt(out["hit_prob"].clip(0.001, 0.98))
        * (0.70 + 0.30 * out["structure_score"].fillna(0.0).clip(0.0, 1.0))
        * (1.0 - 0.20 * norm01(out["difficulty"].fillna(0.5), lo=0.25, hi=0.70))
    )
    return out


PROFILES = {
    "balanced": {
        "win": (1.10, 0.13),
        "place": (0.95, 0.36),
        "wide": (1.05, 0.08),
        "umaren": (1.25, 0.025),
    },
    "roi_strict": {
        "win": (1.35, 0.10),
        "place": (1.05, 0.32),
        "wide": (1.25, 0.06),
        "umaren": (1.55, 0.020),
    },
    "hit_lean": {
        "win": (1.00, 0.18),
        "place": (0.90, 0.44),
        "wide": (0.98, 0.12),
        "umaren": (1.20, 0.035),
    },
}


def apply_profile(cands: pd.DataFrame, profile_name: str, max_difficulty: float, max_danger: float, allowed_types: set[str]) -> pd.DataFrame:
    if cands.empty:
        return cands
    profile = PROFILES[profile_name]
    mask = pd.Series(False, index=cands.index)
    for ticket_type, (min_ev, min_hit) in profile.items():
        type_mask = (
            cands["ticket_type"].eq(ticket_type)
            & cands["ticket_type"].isin(allowed_types)
            & cands["expected_roi"].ge(min_ev)
            & cands["hit_prob"].ge(min_hit)
        )
        mask |= type_mask
    work = cands[mask].copy()
    if work.empty:
        return work
    work = work[work["difficulty"].fillna(0.5).le(max_difficulty) & work["danger"].fillna(0.0).le(max_danger)].copy()
    return work


def select_by_policy(cands: pd.DataFrame, profile_name: str, coverage: float, max_difficulty: float, max_danger: float, allowed_types: set[str], threshold: float | None = None) -> tuple[pd.DataFrame, float]:
    work = apply_profile(cands, profile_name, max_difficulty, max_danger, allowed_types)
    if work.empty:
        return work, float("inf") if threshold is None else threshold
    race_best = (
        work.sort_values(["race_id", "router_score", "expected_roi", "hit_prob"], ascending=[True, False, False, False])
        .groupby("race_id", as_index=False)
        .head(1)
        .copy()
    )
    if threshold is None:
        threshold = float(race_best["router_score"].quantile(1.0 - coverage))
    picked = race_best[race_best["router_score"].ge(threshold)].copy()
    return picked, threshold


def metrics(tickets: pd.DataFrame, label: str) -> dict:
    if tickets.empty:
        return {
            "label": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
        }
    t = tickets.copy().sort_values(["year", "race_id", "ticket_type"])
    t["profit"] = t["return_yen"] - t["stake_yen"]
    out = {
        "label": label,
        "tickets": int(len(t)),
        "races": int(t["race_id"].nunique()),
        "stake_yen": float(t["stake_yen"].sum()),
        "return_yen": float(t["return_yen"].sum()),
        "profit_yen": float(t["profit"].sum()),
        "roi": float(t["return_yen"].sum() / t["stake_yen"].sum()) if t["stake_yen"].sum() else 0.0,
        "hit_rate": float(t["hit"].mean()),
        "max_drawdown_yen": max_drawdown(t["profit"]),
    }
    for ticket_type in ["win", "place", "wide", "umaren"]:
        out[f"{ticket_type}_tickets"] = int(t["ticket_type"].eq(ticket_type).sum())
    return out


def policy_score(row: dict) -> float:
    if row["races"] < 35:
        return -1e9
    if row["hit_rate"] < 0.035:
        return -1e9
    # ROI is the objective, but avoid selecting one-lucky-hit profiles only.
    return float(row["roi"]) * np.log1p(float(row["races"])) * np.sqrt(max(float(row["hit_rate"]), 0.001))


def grid_search(train_cands: pd.DataFrame, allowed_types: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for profile_name, coverage, max_difficulty, max_danger in product(
        PROFILES.keys(),
        [0.02, 0.05, 0.08],
        [0.46, 0.58, 0.70],
        [0.70, 1.05],
    ):
        picked, threshold = select_by_policy(train_cands, profile_name, coverage, max_difficulty, max_danger, allowed_types)
        m = metrics(picked, "train")
        m.update(
            {
                "profile": profile_name,
                "coverage": coverage,
                "max_difficulty": max_difficulty,
                "max_danger": max_danger,
                "router_threshold": threshold,
                "allowed_types": "+".join(sorted(allowed_types)),
                "selection_score": policy_score(m),
            }
        )
        rows.append(m)
    return pd.DataFrame(rows).sort_values(["selection_score", "roi"], ascending=[False, False])


def evaluate_strategy(name: str, allowed_types: set[str], all_cands: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    ticket_rows: list[pd.DataFrame] = []
    grid_rows: list[pd.DataFrame] = []
    for test_year in [2025, 2026]:
        train = all_cands[all_cands["year"].lt(test_year)].copy()
        test = all_cands[all_cands["year"].eq(test_year)].copy()
        grid = grid_search(train, allowed_types)
        grid["strategy"] = name
        grid["test_year"] = test_year
        grid_rows.append(grid.head(50))
        best = grid.iloc[0].to_dict()
        picked, _ = select_by_policy(
            test,
            str(best["profile"]),
            float(best["coverage"]),
            float(best["max_difficulty"]),
            float(best["max_danger"]),
            allowed_types,
            threshold=float(best["router_threshold"]),
        )
        m = metrics(picked, f"{name}_{test_year}")
        m.update(
            {
                "strategy": name,
                "test_year": test_year,
                "train_roi": float(best["roi"]),
                "train_races": int(best["races"]),
                "profile": best["profile"],
                "coverage": float(best["coverage"]),
                "max_difficulty": float(best["max_difficulty"]),
                "max_danger": float(best["max_danger"]),
                "router_threshold": float(best["router_threshold"]),
                "candidate_races": int(test["race_id"].nunique()),
                "selection_rate": float(picked["race_id"].nunique() / test["race_id"].nunique()) if test["race_id"].nunique() else 0.0,
            }
        )
        summary_rows.append(m)
        if not picked.empty:
            tmp = picked.copy()
            tmp["strategy"] = name
            tmp["test_year"] = test_year
            ticket_rows.append(tmp)
    return (
        pd.DataFrame(summary_rows),
        pd.concat(ticket_rows, ignore_index=True, sort=False) if ticket_rows else pd.DataFrame(),
        pd.concat(grid_rows, ignore_index=True, sort=False) if grid_rows else pd.DataFrame(),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runners = load_runners()
    pairs = load_pairs()

    all_frames: list[pd.DataFrame] = []
    years = sorted(set(runners["year"].unique()) | set(pairs["year"].unique()))
    for candidate_year in years:
        train_years = sorted(
            set(runners.loc[runners["year"].lt(candidate_year), "year"])
            | set(pairs.loc[pairs["year"].lt(candidate_year), "year"])
        )
        # The first available year is used only to tune later walk-forward rules.
        # It never appears as a reported OOS test year, so fitting its candidate
        # calibration on itself is acceptable for policy-search bootstrapping.
        if not train_years:
            train_years = [candidate_year]
        all_frames.append(calibrate_and_build_candidates(train_years, candidate_year, runners, pairs))
    candidates = pd.concat(all_frames, ignore_index=True, sort=False)
    candidates.to_csv(OUT / "ticket_type_router_candidates.csv", index=False, encoding="utf-8-sig")

    strategies = {
        "router_all_types": {"win", "place", "wide", "umaren"},
        "pair_only": {"wide", "umaren"},
        "single_only": {"win", "place"},
    }
    summaries: list[pd.DataFrame] = []
    tickets: list[pd.DataFrame] = []
    grids: list[pd.DataFrame] = []
    for name, allowed in strategies.items():
        s, t, g = evaluate_strategy(name, allowed, candidates)
        summaries.append(s)
        tickets.append(t)
        grids.append(g)

    summary = pd.concat(summaries, ignore_index=True, sort=False)
    ticket_df = pd.concat(tickets, ignore_index=True, sort=False)
    grid_df = pd.concat(grids, ignore_index=True, sort=False)

    overall = []
    for strategy, group in ticket_df.groupby("strategy", sort=False):
        m = metrics(group, strategy)
        m["strategy"] = strategy
        m["test_year"] = "2025-2026"
        overall.append(m)
    overall_df = pd.DataFrame(overall)

    by_type = (
        ticket_df.groupby(["strategy", "ticket_type"], dropna=False)
        .agg(
            tickets=("ticket_key", "size"),
            races=("race_id", "nunique"),
            hit_rate=("hit", "mean"),
            stake_yen=("stake_yen", "sum"),
            return_yen=("return_yen", "sum"),
        )
        .reset_index()
    )
    by_type["roi"] = by_type["return_yen"] / by_type["stake_yen"]
    by_type["profit_yen"] = by_type["return_yen"] - by_type["stake_yen"]

    summary.to_csv(OUT / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    overall_df.to_csv(OUT / "overall_summary.csv", index=False, encoding="utf-8-sig")
    ticket_df.to_csv(OUT / "selected_tickets.csv", index=False, encoding="utf-8-sig")
    by_type.to_csv(OUT / "ticket_type_breakdown.csv", index=False, encoding="utf-8-sig")
    grid_df.to_csv(OUT / "train_grid_top50.csv", index=False, encoding="utf-8-sig")

    display_cols = [
        "strategy",
        "test_year",
        "races",
        "tickets",
        "hit_rate",
        "roi",
        "profit_yen",
        "max_drawdown_yen",
        "win_tickets",
        "place_tickets",
        "wide_tickets",
        "umaren_tickets",
        "selection_rate",
        "profile",
        "train_roi",
    ]
    print("TICKET TYPE ROUTER WALKFORWARD")
    print(summary[[c for c in display_cols if c in summary.columns]].to_string(index=False))
    print("\nOVERALL")
    print(overall_df[["strategy", "races", "tickets", "hit_rate", "roi", "profit_yen", "max_drawdown_yen"]].to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
