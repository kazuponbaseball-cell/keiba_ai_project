from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def parse_dates(df: pd.DataFrame) -> pd.Series:
    if "date_key" in df.columns:
        parsed = pd.to_datetime(df["date_key"], errors="coerce")
    elif "日付S" in df.columns:
        parsed = pd.to_datetime(df["日付S"], errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=df.index)

    missing = parsed.isna()
    if missing.any() and "race_id" in df.columns:
        race_digits = df.loc[missing, "race_id"].astype(str).str.extract(r"(\d{8})", expand=False)
        parsed.loc[missing] = pd.to_datetime(race_digits, format="%Y%m%d", errors="coerce")

    missing = parsed.isna()
    if missing.any() and "year" in df.columns:
        parsed.loc[missing] = pd.to_datetime(num(df.loc[missing], "year", 1970).astype("Int64").astype(str) + "-01-01", errors="coerce")
    return parsed


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if pd.isna(value):
        return None
    return value


def prepare_tickets(df: pd.DataFrame, stake_col: str, return_col: str) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["_date"] = parse_dates(out)
    out = out[out["_date"].notna()].copy()
    if "year" not in out.columns:
        out["year"] = out["_date"].dt.year
    out["year"] = num(out, "year", np.nan).astype("Int64")

    if stake_col not in out.columns:
        for fallback in ["runtime_stake_yen", "eval_stake_yen", "scaled_stake_yen", "stake_yen"]:
            if fallback in out.columns:
                stake_col = fallback
                break
    if return_col not in out.columns:
        for fallback in ["runtime_return_yen", "eval_return_yen", "scaled_return_yen", "return_yen"]:
            if fallback in out.columns:
                return_col = fallback
                break

    out["_stake"] = num(out, stake_col, 0.0).fillna(0.0)
    out["_return"] = num(out, return_col, 0.0).fillna(0.0)
    out["_profit"] = out["_return"] - out["_stake"]
    out["_hit"] = np.where(out["_return"].gt(0), 1.0, num(out, "hit", 0.0).fillna(0.0))
    out["_period"] = out["_date"].dt.to_period("M").astype(str)
    out["_ticket_order"] = np.arange(len(out))

    # Normalize pre-race/runtime quality columns used by strategy rules.
    out["_margin"] = num(out, "min_odds_margin_ratio", 0.0).fillna(0.0)
    out["_expected_roi"] = num(out, "runtime_expected_roi", np.nan)
    out["_expected_roi"] = out["_expected_roi"].fillna(num(out, "expected_roi_after_slippage", 0.0)).fillna(0.0)
    out["_hit_prob"] = num(out, "ticket_hit_prob", 0.0).fillna(0.0)
    out["_overlay"] = num(out, "market_overlay_score", 0.0).fillna(0.0)
    out["_front5"] = num(out, "projected_front5_prob", 0.0).fillna(0.0)
    out["_pair_score"] = num(out, "pair_score", 0.0).fillna(0.0)
    out["_pair_q"] = num(out, "pair_quinella_score", 0.0).fillna(0.0)
    out["_late_value"] = num(out, "late_value_survives_score", 0.0).fillna(0.0)
    out["_priority_net"] = num(out, "priority_context_net_score", 0.0).fillna(0.0)
    out["_ticket_quality"] = num(out, "b_ticket_quality_score", 0.0).fillna(0.0)
    out["_front_reliability"] = num(out, "ticket_front_position_reliability_score", 0.0).fillna(0.0)
    front_context_collapse = num(out, "front_context_collapse_risk_score", np.nan)
    front_context_collapse = front_context_collapse.fillna(num(out, "front_collapse_reinforced_score", np.nan))
    out["_front_context_available"] = front_context_collapse.notna().astype(float)
    out["_front_context_collapse"] = front_context_collapse.fillna(999.0)
    front_context_survival = num(out, "front_context_survival_support_score", np.nan)
    front_context_survival = front_context_survival.fillna(num(out, "front_survival_despite_pressure_score", 0.0))
    out["_front_context_survival"] = front_context_survival.fillna(0.0)
    out["_front_context_readability"] = num(out, "front_context_readability_score", 0.0).fillna(0.0)
    danger_sum = num(out, "danger_sum", np.nan)
    danger_sum = danger_sum.fillna(num(out, "anchor_danger", 0.0).fillna(0.0) + num(out, "partner_danger", 0.0).fillna(0.0))
    out["_danger_sum"] = danger_sum.fillna(0.0)
    out["_difficulty"] = num(out, "race_difficulty_score", np.nan)
    out["_difficulty"] = out["_difficulty"].fillna(num(out, "race_difficulty_model_score", np.nan)).fillna(num(out, "difficulty", 0.0)).fillna(0.0)
    if "source_label" not in out.columns:
        out["source_label"] = "unknown"
    if "ticket_type" not in out.columns:
        out["ticket_type"] = "unknown"
    return out[out["_stake"].gt(0)].reset_index(drop=True)


@dataclass(frozen=True)
class StrategyRule:
    strategy_id: str
    ticket_type: str = "all"
    source_label: str = "all"
    min_margin: float = 0.0
    min_expected_roi: float = 0.0
    min_hit_prob: float = 0.0
    min_overlay: float = 0.0
    min_front5: float = 0.0
    min_pair_score: float = 0.0
    min_pair_q: float = 0.0
    min_late_value: float = 0.0
    min_priority_net: float = -999.0
    min_ticket_quality: float = 0.0
    max_danger_sum: float = 999.0
    max_difficulty: float = 999.0
    min_front_context_available: float = 0.0
    max_front_context_collapse: float = 999.0
    min_front_context_survival: float = 0.0
    min_front_context_readability: float = 0.0


def candidate_rules(preset: str = "lean") -> list[StrategyRule]:
    rows: list[StrategyRule] = []
    seen: set[tuple] = set()

    def add(**kwargs) -> None:
        key = tuple(sorted(kwargs.items()))
        if key in seen:
            return
        seen.add(key)
        rows.append(StrategyRule(strategy_id=f"s{len(rows):04d}", **kwargs))

    if preset not in {"lean", "full"}:
        raise ValueError(f"unknown candidate preset: {preset}")

    if preset == "full":
        base_ticket_types = ["all", "umaren", "wide", "win"]
        base_sources = ["all", "standard", "expanded"]
        base_margins = [0.0, 0.95, 1.0, 1.15, 1.35, 1.75]
        base_expected_rois = [0.0, 1.10, 1.35, 1.75]
        base_hit_probs = [0.0, 0.08, 0.12, 0.18]
        structural_margins = [1.0, 1.15, 1.35]
        structural_overlays = [0.70, 0.78, 0.85]
        structural_front5 = [0.50, 0.60, 0.70]
        structural_dangers = [0.55, 0.70, 999.0]
        pair_scores = [0.72, 0.80, 0.86]
        pair_qs = [0.50, 0.58, 0.66]
        late_values = [0.0, 0.65, 0.75]
        priority_nets = [0.0, 0.10, 0.20]
        ticket_qualities = [0.0, 0.45, 0.55]
        max_difficulties = [0.55, 0.70, 999.0]
        front_context_max_collapse = [0.012, 0.018]
        front_context_min_readability = [0.0, 0.46]
    else:
        base_ticket_types = ["all", "umaren", "wide", "win"]
        base_sources = ["all", "standard", "expanded"]
        base_margins = [0.0, 1.0, 1.15, 1.35]
        base_expected_rois = [0.0, 1.10, 1.35]
        base_hit_probs = [0.0, 0.12]
        structural_margins = [1.0, 1.20, 1.50]
        structural_overlays = [0.70, 0.82]
        structural_front5 = [0.50, 0.65]
        structural_dangers = [0.65, 999.0]
        pair_scores = [0.72, 0.82]
        pair_qs = [0.50, 0.62]
        late_values = [0.0, 0.70]
        priority_nets = [0.0, 0.18]
        ticket_qualities = [0.0, 0.52]
        max_difficulties = [0.65, 999.0]
        front_context_max_collapse = [0.012, 0.018]
        front_context_min_readability = [0.0, 0.46]

    # Current-ish operational gates and close variants.
    for ticket_type, source_label, min_margin, min_expected_roi, min_hit_prob in itertools.product(
        base_ticket_types,
        base_sources,
        base_margins,
        base_expected_rois,
        base_hit_probs,
    ):
        add(
            ticket_type=ticket_type,
            source_label=source_label,
            min_margin=min_margin,
            min_expected_roi=min_expected_roi,
            min_hit_prob=min_hit_prob,
        )

    # Structural value variants: front-running, pair quality, overlay, and risk.
    for ticket_type, min_margin, min_overlay, min_front5, max_danger_sum in itertools.product(
        ["all", "umaren", "wide"],
        structural_margins,
        structural_overlays,
        structural_front5,
        structural_dangers,
    ):
        add(
            ticket_type=ticket_type,
            min_margin=min_margin,
            min_expected_roi=1.10,
            min_overlay=min_overlay,
            min_front5=min_front5,
            max_danger_sum=max_danger_sum,
        )

    for ticket_type, min_margin, min_pair_score, min_pair_q, min_late_value in itertools.product(
        ["all", "umaren", "wide"],
        structural_margins,
        pair_scores,
        pair_qs,
        late_values,
    ):
        add(
            ticket_type=ticket_type,
            min_margin=min_margin,
            min_expected_roi=1.10,
            min_pair_score=min_pair_score,
            min_pair_q=min_pair_q,
            min_late_value=min_late_value,
        )

    for ticket_type, min_margin, min_priority_net, min_ticket_quality, max_difficulty in itertools.product(
        ["all", "umaren", "wide"],
        structural_margins,
        priority_nets,
        ticket_qualities,
        max_difficulties,
    ):
        add(
            ticket_type=ticket_type,
            min_margin=min_margin,
            min_expected_roi=1.10,
            min_priority_net=min_priority_net,
            min_ticket_quality=min_ticket_quality,
            max_difficulty=max_difficulty,
        )

    for ticket_type, min_margin, max_front_context_collapse, min_front_context_readability in itertools.product(
        ["all", "umaren", "wide"],
        structural_margins,
        front_context_max_collapse,
        front_context_min_readability,
    ):
        add(
            ticket_type=ticket_type,
            min_margin=min_margin,
            min_expected_roi=1.10,
            min_front_context_available=1.0,
            max_front_context_collapse=max_front_context_collapse,
            min_front_context_readability=min_front_context_readability,
        )

    return rows


def apply_rule(df: pd.DataFrame, rule: StrategyRule) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if rule.ticket_type != "all":
        mask &= df["ticket_type"].astype(str).eq(rule.ticket_type)
    if rule.source_label != "all":
        mask &= df["source_label"].astype(str).eq(rule.source_label)
    mask &= df["_margin"].ge(rule.min_margin)
    mask &= df["_expected_roi"].ge(rule.min_expected_roi)
    mask &= df["_hit_prob"].ge(rule.min_hit_prob)
    mask &= df["_overlay"].ge(rule.min_overlay)
    mask &= df["_front5"].ge(rule.min_front5)
    mask &= df["_pair_score"].ge(rule.min_pair_score)
    mask &= df["_pair_q"].ge(rule.min_pair_q)
    mask &= df["_late_value"].ge(rule.min_late_value)
    mask &= df["_priority_net"].ge(rule.min_priority_net)
    mask &= df["_ticket_quality"].ge(rule.min_ticket_quality)
    mask &= df["_danger_sum"].le(rule.max_danger_sum)
    mask &= df["_difficulty"].le(rule.max_difficulty)
    mask &= df["_front_context_available"].ge(rule.min_front_context_available)
    mask &= df["_front_context_collapse"].le(rule.max_front_context_collapse)
    mask &= df["_front_context_survival"].ge(rule.min_front_context_survival)
    mask &= df["_front_context_readability"].ge(rule.min_front_context_readability)
    return mask.fillna(False)


def max_drawdown(profit_by_order: pd.Series) -> float:
    if profit_by_order.empty:
        return 0.0
    equity = profit_by_order.cumsum()
    peak = equity.cummax().clip(lower=0.0)
    drawdown = equity - peak
    return float(drawdown.min())


def metrics(df: pd.DataFrame, mask: pd.Series | np.ndarray, label: str = "") -> dict:
    part = df.loc[np.asarray(mask)].copy()
    if part.empty:
        return {
            "label": label,
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

    stake = float(part["_stake"].sum())
    ret = float(part["_return"].sum())
    race = (
        part.groupby("race_id", sort=False)
        .agg(
            date=("_date", "min"),
            stake_yen=("_stake", "sum"),
            return_yen=("_return", "sum"),
            hit=("_hit", "max"),
        )
        .reset_index()
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    race = race.sort_values(["date", "race_id"])

    def removed_roi(n: int) -> float:
        if len(race) <= n:
            return 0.0
        kept = race.sort_values("profit_yen", ascending=False).iloc[n:]
        kept_stake = float(kept["stake_yen"].sum())
        return float(kept["return_yen"].sum() / kept_stake) if kept_stake else 0.0

    return {
        "label": label,
        "tickets": int(len(part)),
        "races": int(part["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(part["_hit"].mean()) if len(part) else 0.0,
        "race_hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "max_drawdown_yen": max_drawdown(race["profit_yen"]),
        "top5_removed_roi": removed_roi(5),
        "top10_removed_roi": removed_roi(10),
    }


def robust_score(row: dict, min_races: int, min_stake: float) -> float:
    if row["races"] < min_races or row["stake_yen"] < min_stake:
        return -1e9
    if row["race_hit_rate"] <= 0:
        return -1e9
    roi = float(row["roi"])
    top5 = float(row["top5_removed_roi"])
    top10 = float(row["top10_removed_roi"])
    concentration = max(roi - top5, 0.0)
    dd_penalty = abs(float(row["max_drawdown_yen"])) / max(float(row["stake_yen"]), 1.0)
    return float(
        0.50 * roi
        + 0.95 * top5
        + 0.45 * top10
        + 1.75 * float(row["race_hit_rate"])
        + 0.06 * math.log1p(float(row["races"]))
        - 0.75 * concentration
        - 0.35 * dd_penalty
    )


def fast_metrics(df: pd.DataFrame, mask: pd.Series | np.ndarray, label: str = "") -> dict:
    part = df.loc[np.asarray(mask)]
    if part.empty:
        return {
            "label": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
        }
    stake = float(part["_stake"].sum())
    ret = float(part["_return"].sum())
    ticket_hit_rate = float(part["_hit"].mean()) if len(part) else 0.0
    return {
        "label": label,
        "tickets": int(len(part)),
        "races": int(part["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": ticket_hit_rate,
        "race_hit_rate": ticket_hit_rate,
    }


def fast_selection_score(row: dict, min_races: int, min_stake: float) -> float:
    if row["races"] < min_races or row["stake_yen"] < min_stake:
        return -1e9
    if row["race_hit_rate"] <= 0:
        return -1e9
    return float(
        0.85 * row["roi"]
        + 1.50 * row["race_hit_rate"]
        + 0.05 * math.log1p(row["races"])
        + 0.000002 * row["profit_yen"]
    )


def select_fast_candidate(
    df: pd.DataFrame,
    rules: list[StrategyRule],
    candidate_masks: dict[str, np.ndarray],
    row_mask: pd.Series | np.ndarray,
    *,
    min_races: int,
    min_stake: float,
) -> tuple[StrategyRule | None, pd.DataFrame]:
    base_mask = np.asarray(row_mask)
    rows = []
    best_rule: StrategyRule | None = None
    best_score = -1e18
    for rule in rules:
        mask = candidate_masks[rule.strategy_id] & base_mask
        m = fast_metrics(df, mask, rule.strategy_id)
        score = fast_selection_score(m, min_races=min_races, min_stake=min_stake)
        rows.append(asdict(rule) | m | {"fast_selection_score": score})
        if score > best_score:
            best_score = score
            best_rule = rule
    grid = pd.DataFrame(rows).sort_values(["fast_selection_score", "roi"], ascending=[False, False])
    if best_score <= -1e8:
        return None, grid
    return best_rule, grid


def select_fast_candidate_matrix(
    df: pd.DataFrame,
    rules: list[StrategyRule],
    mask_matrix: np.ndarray,
    row_mask: pd.Series | np.ndarray,
    *,
    min_races: int,
    min_stake: float,
) -> tuple[StrategyRule | None, pd.DataFrame]:
    base = np.asarray(row_mask, dtype=bool)
    active = mask_matrix & base.reshape(1, -1)
    tickets = active.sum(axis=1).astype(float)
    stake_values = df["_stake"].to_numpy(dtype=float)
    return_values = df["_return"].to_numpy(dtype=float)
    hit_values = df["_hit"].to_numpy(dtype=float)
    stakes = active @ stake_values
    returns = active @ return_values
    hits = active @ hit_values
    profits = returns - stakes
    rois = np.divide(returns, stakes, out=np.zeros_like(returns, dtype=float), where=stakes > 0)
    hit_rates = np.divide(hits, tickets, out=np.zeros_like(hits, dtype=float), where=tickets > 0)
    # Fast selection uses ticket count as a conservative coverage proxy. Exact
    # race-level metrics are recomputed for the selected policy only.
    scores = (
        0.85 * rois
        + 1.50 * hit_rates
        + 0.05 * np.log1p(tickets)
        + 0.000002 * profits
    )
    invalid = (tickets < min_races) | (stakes < min_stake) | (hit_rates <= 0)
    scores = np.where(invalid, -1e9, scores)
    rows = []
    for i, rule in enumerate(rules):
        rows.append(
            asdict(rule)
            | {
                "label": rule.strategy_id,
                "tickets": int(tickets[i]),
                "races": int(tickets[i]),
                "stake_yen": float(stakes[i]),
                "return_yen": float(returns[i]),
                "profit_yen": float(profits[i]),
                "roi": float(rois[i]),
                "ticket_hit_rate": float(hit_rates[i]),
                "race_hit_rate": float(hit_rates[i]),
                "fast_selection_score": float(scores[i]),
            }
        )
    grid = pd.DataFrame(rows).sort_values(["fast_selection_score", "roi"], ascending=[False, False])
    if len(grid) == 0 or float(grid.iloc[0]["fast_selection_score"]) <= -1e8:
        return None, grid
    best_idx = int(str(grid.iloc[0]["strategy_id"])[1:])
    return rules[best_idx], grid


def evaluate_candidates(df: pd.DataFrame, rules: list[StrategyRule], row_mask: pd.Series | np.ndarray, *, min_races: int, min_stake: float) -> pd.DataFrame:
    rows = []
    base_mask = np.asarray(row_mask)
    for rule in rules:
        mask = np.asarray(apply_rule(df, rule)) & base_mask
        m = metrics(df, mask, rule.strategy_id)
        row = asdict(rule) | m
        row["robust_score"] = robust_score(row, min_races=min_races, min_stake=min_stake)
        rows.append(row)
    return pd.DataFrame(rows)


def purged_walk_forward(
    df: pd.DataFrame,
    rules: list[StrategyRule],
    *,
    purge_days: int,
    embargo_days: int,
    min_train_periods: int,
    min_train_races: int,
    min_train_stake: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = (
        df[["_period", "_date"]]
        .drop_duplicates("_period")
        .sort_values("_date")["_period"]
        .tolist()
    )
    folds = []
    selected_parts = []
    candidate_masks = {rule.strategy_id: np.asarray(apply_rule(df, rule)) for rule in rules}
    mask_matrix = np.vstack([candidate_masks[rule.strategy_id] for rule in rules]).astype(bool)

    for idx, period in enumerate(periods):
        if idx < min_train_periods:
            continue
        test_mask = df["_period"].eq(period)
        if not test_mask.any():
            continue
        test_start = df.loc[test_mask, "_date"].min()
        test_end = df.loc[test_mask, "_date"].max()
        train_mask = df["_date"].lt(test_start - pd.Timedelta(days=purge_days))
        train_mask &= ~df["_date"].between(test_start - pd.Timedelta(days=purge_days), test_end + pd.Timedelta(days=embargo_days))
        if df.loc[train_mask, "race_id"].nunique() < min_train_races:
            continue

        rule, train_grid = select_fast_candidate_matrix(
            df,
            rules,
            mask_matrix,
            train_mask,
            min_races=min_train_races,
            min_stake=min_train_stake,
        )
        if rule is None:
            continue
        best = train_grid.iloc[0]
        test_rule_mask = candidate_masks[rule.strategy_id] & np.asarray(test_mask)
        test_m = metrics(df, test_rule_mask, f"test_{period}")
        train_m = metrics(df, candidate_masks[rule.strategy_id] & np.asarray(train_mask), f"train_{period}")
        folds.append(
            {
                "test_period": period,
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "selected_strategy_id": rule.strategy_id,
                "selected_ticket_type": rule.ticket_type,
                "selected_source_label": rule.source_label,
                "selected_min_margin": rule.min_margin,
                "selected_min_expected_roi": rule.min_expected_roi,
                "selected_min_hit_prob": rule.min_hit_prob,
                "selected_min_overlay": rule.min_overlay,
                "selected_min_front5": rule.min_front5,
                "selected_max_danger_sum": rule.max_danger_sum,
                "selected_max_front_context_collapse": rule.max_front_context_collapse,
                "selected_min_front_context_readability": rule.min_front_context_readability,
                "train_fast_selection_score": float(best["fast_selection_score"]),
                **{f"train_{k}": v for k, v in train_m.items() if k != "label"},
                **{f"test_{k}": v for k, v in test_m.items() if k != "label"},
            }
        )
        if test_rule_mask.any():
            part = df.loc[test_rule_mask].copy()
            part["walkforward_test_period"] = period
            part["walkforward_strategy_id"] = rule.strategy_id
            selected_parts.append(part)

    selected = pd.concat(selected_parts, ignore_index=True, sort=False) if selected_parts else pd.DataFrame()
    return pd.DataFrame(folds), selected


def assign_equal_blocks(df: pd.DataFrame, n_blocks: int) -> pd.Series:
    races = (
        df[["race_id", "_date"]]
        .drop_duplicates("race_id")
        .sort_values(["_date", "race_id"])
        .reset_index(drop=True)
    )
    races["_block"] = (np.arange(len(races)) * n_blocks // max(len(races), 1)).clip(0, n_blocks - 1)
    return df["race_id"].map(dict(zip(races["race_id"], races["_block"]))).astype(int)


def pbo_analysis(
    df: pd.DataFrame,
    rules: list[StrategyRule],
    *,
    n_blocks: int,
    purge_days: int,
    embargo_days: int,
    min_train_races: int,
    min_train_stake: float,
    max_splits: int,
) -> pd.DataFrame:
    work = df.copy()
    work["_pbo_block"] = assign_equal_blocks(work, n_blocks)
    candidate_masks = {rule.strategy_id: np.asarray(apply_rule(work, rule)) for rule in rules}
    mask_matrix = np.vstack([candidate_masks[rule.strategy_id] for rule in rules]).astype(bool)
    stake_values = work["_stake"].to_numpy(dtype=float)
    return_values = work["_return"].to_numpy(dtype=float)
    block_dates = work.groupby("_pbo_block")["_date"].agg(["min", "max"]).to_dict("index")
    blocks = list(range(n_blocks))
    combos = list(itertools.combinations(blocks, n_blocks // 2))
    if max_splits and len(combos) > max_splits:
        rng = np.random.default_rng(20260619)
        combos = [combos[i] for i in sorted(rng.choice(len(combos), size=max_splits, replace=False).tolist())]

    rows = []
    for split_id, test_blocks_tuple in enumerate(combos):
        test_blocks = set(test_blocks_tuple)
        test_mask = work["_pbo_block"].isin(test_blocks)
        train_mask = ~test_mask

        # Purge/embargo any training rows close to a test interval.
        for block in test_blocks:
            start = block_dates[block]["min"]
            end = block_dates[block]["max"]
            close = work["_date"].between(start - pd.Timedelta(days=purge_days), end + pd.Timedelta(days=embargo_days))
            train_mask &= ~close

        if work.loc[train_mask, "race_id"].nunique() < min_train_races:
            continue
        selected_rule, train_grid = select_fast_candidate_matrix(
            work,
            rules,
            mask_matrix,
            train_mask,
            min_races=min_train_races,
            min_stake=min_train_stake,
        )
        if selected_rule is None:
            continue
        best = train_grid.iloc[0]

        test_base = np.asarray(test_mask, dtype=bool)
        test_active = mask_matrix & test_base.reshape(1, -1)
        test_tickets = test_active.sum(axis=1).astype(float)
        test_stakes = test_active @ stake_values
        test_returns = test_active @ return_values
        test_profits = test_returns - test_stakes
        test_rois = np.divide(test_returns, test_stakes, out=np.zeros_like(test_returns, dtype=float), where=test_stakes > 0)
        valid = test_stakes > 0
        test_grid = pd.DataFrame(
            {
                "strategy_id": [rule.strategy_id for rule in rules],
                "test_roi": test_rois,
                "test_profit_yen": test_profits,
                "test_races": test_tickets.astype(int),
                "valid": valid,
            }
        )
        test_grid = test_grid[test_grid["valid"]].drop(columns=["valid"])
        if test_grid.empty or best["strategy_id"] not in set(test_grid["strategy_id"]):
            continue
        selected_test = test_grid[test_grid["strategy_id"].eq(best["strategy_id"])].iloc[0]
        selected_train_exact = fast_metrics(work, candidate_masks[selected_rule.strategy_id] & np.asarray(train_mask), selected_rule.strategy_id)
        test_grid["rank"] = test_grid["test_roi"].rank(method="average", ascending=True)
        percentile = float(test_grid.loc[test_grid["strategy_id"].eq(best["strategy_id"]), "rank"].iloc[0] / len(test_grid))
        percentile = min(max(percentile, 1e-6), 1 - 1e-6)
        rows.append(
            {
                "split_id": split_id,
                "test_blocks": ",".join(map(str, sorted(test_blocks))),
                "selected_strategy_id": best["strategy_id"],
                "train_fast_selection_score": float(best["fast_selection_score"]),
                "train_roi": float(selected_train_exact["roi"]),
                "train_races": int(selected_train_exact["races"]),
                "test_roi": float(selected_test["test_roi"]),
                "test_profit_yen": float(selected_test["test_profit_yen"]),
                "test_races": int(selected_test["test_races"]),
                "test_percentile": percentile,
                "test_logit_percentile": float(math.log(percentile / (1.0 - percentile))),
                "is_overfit": bool(percentile < 0.5),
                "test_candidate_count": int(len(test_grid)),
            }
        )
    return pd.DataFrame(rows)


def period_profit_matrix(df: pd.DataFrame, rules: list[StrategyRule], candidate_ids: Iterable[str]) -> pd.DataFrame:
    ids = list(candidate_ids)
    periods = sorted(df["_period"].unique().tolist())
    matrix = pd.DataFrame(0.0, index=periods, columns=ids)
    for rule in rules:
        if rule.strategy_id not in ids:
            continue
        part = df.loc[apply_rule(df, rule)]
        if part.empty:
            continue
        profit = part.groupby("_period")["_profit"].sum()
        for period, value in profit.items():
            matrix.loc[period, rule.strategy_id] = float(value)
    return matrix


def simplified_mcs(
    profit_matrix: pd.DataFrame,
    *,
    alpha: float,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    active = list(profit_matrix.columns)
    losses = -profit_matrix.astype(float)
    path = []
    step = 0

    while len(active) > 1:
        active_losses = losses[active]
        observed_mean = active_losses.mean(axis=0)
        best = observed_mean.idxmin()
        p_values = {}
        t = len(active_losses)
        for candidate in active:
            if candidate == best:
                p_values[candidate] = 1.0
                continue
            observed_diff = float(observed_mean[candidate] - observed_mean[best])
            diffs = []
            values = active_losses[[candidate, best]].to_numpy()
            for _ in range(bootstrap_samples):
                idx = rng.integers(0, t, size=t)
                sample = values[idx]
                diffs.append(float(sample[:, 0].mean() - sample[:, 1].mean()))
            # Probability that the candidate is not worse than the current best.
            p_values[candidate] = float(np.mean(np.asarray(diffs) <= 0.0)) if observed_diff >= 0 else 1.0

        removable = [(c, p) for c, p in p_values.items() if c != best]
        worst, p_no_worse = min(removable, key=lambda item: item[1])
        path.append(
            {
                "step": step,
                "active_count": len(active),
                "current_best": best,
                "removed_strategy_id": worst if p_no_worse < alpha else "",
                "removed_p_no_worse_than_best": p_no_worse,
                "stop": bool(p_no_worse >= alpha),
            }
        )
        if p_no_worse >= alpha:
            break
        active.remove(worst)
        step += 1

    survivors = pd.DataFrame(
        {
            "strategy_id": active,
            "mcs_survivor": True,
            "mean_period_profit_yen": [float(profit_matrix[c].mean()) for c in active],
            "total_profit_yen": [float(profit_matrix[c].sum()) for c in active],
        }
    ).sort_values(["mean_period_profit_yen", "total_profit_yen"], ascending=[False, False])
    return survivors, pd.DataFrame(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ticket strategy selection with purged walk-forward, simplified MCS, and PBO diagnostics.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/pair_probability_runtime_v1/pair_calibrated_runtime_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/purged_walkforward_mcs_pbo_v1")
    parser.add_argument("--stake-col", default="runtime_stake_yen")
    parser.add_argument("--return-col", default="runtime_return_yen")
    parser.add_argument("--purge-days", type=int, default=7)
    parser.add_argument("--embargo-days", type=int, default=7)
    parser.add_argument("--min-train-periods", type=int, default=3)
    parser.add_argument("--min-train-races", type=int, default=50)
    parser.add_argument("--min-train-stake", type=float, default=10000.0)
    parser.add_argument("--pbo-blocks", type=int, default=8)
    parser.add_argument("--pbo-max-splits", type=int, default=0)
    parser.add_argument("--mcs-alpha", type=float, default=0.10)
    parser.add_argument("--mcs-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--mcs-max-candidates", type=int, default=80)
    parser.add_argument("--candidate-preset", choices=["lean", "full"], default="lean")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tickets = prepare_tickets(pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False), args.stake_col, args.return_col)
    rules = candidate_rules(args.candidate_preset)
    catalog = pd.DataFrame([asdict(rule) for rule in rules])
    catalog.to_csv(out_dir / "candidate_catalog.csv", index=False, encoding="utf-8-sig")

    all_mask = pd.Series(True, index=tickets.index)
    overall = evaluate_candidates(tickets, rules, all_mask, min_races=max(10, args.min_train_races // 2), min_stake=max(1000.0, args.min_train_stake / 2))
    overall = overall.sort_values(["robust_score", "top5_removed_roi", "roi"], ascending=[False, False, False])
    overall.to_csv(out_dir / "candidate_overall_metrics.csv", index=False, encoding="utf-8-sig")

    folds, wf_selected = purged_walk_forward(
        tickets,
        rules,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        min_train_periods=args.min_train_periods,
        min_train_races=args.min_train_races,
        min_train_stake=args.min_train_stake,
    )
    folds.to_csv(out_dir / "purged_walkforward_folds.csv", index=False, encoding="utf-8-sig")
    if not wf_selected.empty:
        wf_selected.to_csv(out_dir / "purged_walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
    wf_metrics = metrics(wf_selected, pd.Series(True, index=wf_selected.index), "purged_walkforward_selected") if not wf_selected.empty else metrics(tickets, pd.Series(False, index=tickets.index), "purged_walkforward_selected")

    pbo = pbo_analysis(
        tickets,
        rules,
        n_blocks=args.pbo_blocks,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        min_train_races=args.min_train_races,
        min_train_stake=args.min_train_stake,
        max_splits=args.pbo_max_splits,
    )
    pbo.to_csv(out_dir / "pbo_splits.csv", index=False, encoding="utf-8-sig")

    eligible_mcs = overall[overall["robust_score"].gt(-1e8)].head(args.mcs_max_candidates).copy()
    profit_matrix = period_profit_matrix(tickets, rules, eligible_mcs["strategy_id"].tolist())
    profit_matrix.to_csv(out_dir / "mcs_period_profit_matrix.csv", encoding="utf-8-sig")
    if profit_matrix.shape[1] >= 2 and profit_matrix.shape[0] >= 2:
        mcs_survivors, mcs_path = simplified_mcs(
            profit_matrix,
            alpha=args.mcs_alpha,
            bootstrap_samples=args.mcs_bootstrap_samples,
            random_seed=20260619,
        )
    else:
        mcs_survivors = pd.DataFrame(columns=["strategy_id", "mcs_survivor", "mean_period_profit_yen", "total_profit_yen"])
        mcs_path = pd.DataFrame()
    mcs_survivors = mcs_survivors.merge(catalog, on="strategy_id", how="left")
    mcs_survivors = mcs_survivors.merge(overall.add_prefix("overall_"), left_on="strategy_id", right_on="overall_strategy_id", how="left")
    mcs_survivors.to_csv(out_dir / "mcs_survivors.csv", index=False, encoding="utf-8-sig")
    mcs_path.to_csv(out_dir / "mcs_elimination_path.csv", index=False, encoding="utf-8-sig")

    pbo_summary = {
        "splits": int(len(pbo)),
        "pbo": float(pbo["is_overfit"].mean()) if not pbo.empty else None,
        "median_test_percentile": float(pbo["test_percentile"].median()) if not pbo.empty else None,
        "median_test_logit_percentile": float(pbo["test_logit_percentile"].median()) if not pbo.empty else None,
        "avg_selected_train_roi": float(pbo["train_roi"].mean()) if not pbo.empty else None,
        "avg_selected_test_roi": float(pbo["test_roi"].mean()) if not pbo.empty else None,
    }
    summary = {
        "tickets_csv": args.tickets_csv,
        "rows": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "periods": int(tickets["_period"].nunique()),
        "date_min": str(tickets["_date"].min().date()) if not tickets.empty else "",
        "date_max": str(tickets["_date"].max().date()) if not tickets.empty else "",
        "candidate_count": int(len(rules)),
        "candidate_preset": args.candidate_preset,
        "purge_days": args.purge_days,
        "embargo_days": args.embargo_days,
        "walkforward_folds": int(len(folds)),
        "walkforward_metrics": wf_metrics,
        "pbo_summary": pbo_summary,
        "mcs_alpha": args.mcs_alpha,
        "mcs_survivor_count": int(len(mcs_survivors)),
        "top_overall_candidates": overall.head(15).to_dict(orient="records"),
        "top_mcs_survivors": mcs_survivors.head(15).to_dict(orient="records"),
        "notes": [
            "Candidate rules use only pre-race/runtime columns, then select by train-period results.",
            "Purged walk-forward uses only dates before each test period after the purge window.",
            "MCS is a simplified operational bootstrap over monthly period profits, not a full Hansen-MCS proof.",
            "PBO uses combinatorial chronological blocks and reports the fraction of selected strategies that fall below the median test percentile.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
