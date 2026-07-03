from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ensure_dir(path: str | Path) -> Path:
    p = project_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if pd.isna(value):
        return None
    return value


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    series = df[col]
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False).str.replace("+", "", regex=False).str.strip()
    return pd.to_numeric(series, errors="coerce")


def norm01(series: pd.Series, lo: float, hi: float) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if hi <= lo:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0)


def read_feature_file(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    by_pos = {
        "date_raw": 0,
        "date_s": 1,
        "venue": 2,
        "race_no": 3,
        "class_name": 6,
        "horse_name": 7,
        "sex": 8,
        "age": 9,
        "carried_weight": 11,
        "field_size": 12,
        "horse_no": 15,
        "popularity": 16,
        "win_odds": 17,
        "surface": 18,
        "distance": 19,
        "going": 20,
        "race_id": 34,
        "finish": 38,
        "prev_carried_weight": 45,
        "weight_diff": 77,
    }
    resolved = {k: header[v] for k, v in by_pos.items() if v < len(header)}
    ascii_cols = ["race_weight_light_rank_score"]
    usecols = list(resolved.values()) + [c for c in ascii_cols if c in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns={v: k for k, v in resolved.items()})
    for col in ascii_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_runner_context(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw in paths:
        path = project_path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(read_feature_file(path))
    out = pd.concat(frames, ignore_index=True)
    out["race_id"] = out["race_id"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(16)
    out["horse_no"] = num(out, "horse_no").astype("Int64")
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out = out.drop_duplicates(["race_id", "horse_no"], keep="last")

    out["year"] = pd.to_numeric(out["race_id"].astype(str).str.slice(0, 4), errors="coerce")
    out["month"] = pd.to_numeric(out["race_id"].astype(str).str.slice(4, 6), errors="coerce")
    out["age_num"] = num(out, "age")
    out["carried_weight_num"] = num(out, "carried_weight")
    out["prev_carried_weight_num"] = num(out, "prev_carried_weight")
    out["weight_diff_num"] = num(out, "weight_diff", 0).fillna(0)
    out["popularity_num"] = num(out, "popularity")
    out["win_odds_num"] = num(out, "win_odds")
    out["finish_num"] = num(out, "finish")
    out["distance_num"] = num(out, "distance")
    out["race_weight_light_rank_score_num"] = num(out, "race_weight_light_rank_score", 0.5).fillna(0.5)

    surface_text = out.get("surface", pd.Series("", index=out.index)).astype(str)
    out["surface_simple"] = np.select(
        [
            surface_text.str.contains("\u829d", regex=False, na=False),
            surface_text.str.contains("\u30c0", regex=False, na=False),
        ],
        ["turf", "dirt"],
        default="other",
    )
    out["age_bucket"] = np.select(
        [
            out["age_num"].eq(2),
            out["age_num"].eq(3),
            out["age_num"].eq(4),
            out["age_num"].eq(5),
            out["age_num"].ge(6),
        ],
        ["age2", "age3", "age4", "age5", "age6plus"],
        default="unknown",
    )

    race = out.groupby("race_id", sort=False)
    out["race_weight_median"] = race["carried_weight_num"].transform("median")
    out["race_weight_min"] = race["carried_weight_num"].transform("min")
    out["race_weight_max"] = race["carried_weight_num"].transform("max")
    out["race_weight_range"] = (out["race_weight_max"] - out["race_weight_min"]).fillna(0)
    out["race_age_min"] = race["age_num"].transform("min")
    out["race_age_max"] = race["age_num"].transform("max")
    out["race_age_range"] = (out["race_age_max"] - out["race_age_min"]).fillna(0)
    out["race_has_age3"] = race["age_num"].transform(lambda s: bool((s == 3).any())).astype(int)
    out["race_has_older4plus"] = race["age_num"].transform(lambda s: bool((s >= 4).any())).astype(int)
    out["race_mixed_3yo_older"] = (out["race_has_age3"].eq(1) & out["race_has_older4plus"].eq(1)).astype(int)
    out["runner_weight_advantage_kg"] = (out["race_weight_median"] - out["carried_weight_num"]).fillna(0)
    out["runner_weight_from_high_kg"] = (out["race_weight_max"] - out["carried_weight_num"]).fillna(0)
    out["runner_heavy_vs_race_kg"] = (out["carried_weight_num"] - out["race_weight_median"]).fillna(0)

    summer = out["month"].between(6, 9, inclusive="both")
    out["age_weight_summer_3yo_allowance_flag"] = (
        out["age_num"].eq(3)
        & summer
        & out["race_mixed_3yo_older"].eq(1)
        & (out["runner_weight_advantage_kg"].ge(1.0) | out["runner_weight_from_high_kg"].ge(2.0))
    ).astype(int)
    out["age_weight_big_3yo_allowance_flag"] = (
        out["age_num"].eq(3)
        & out["race_mixed_3yo_older"].eq(1)
        & out["runner_weight_from_high_kg"].ge(2.0)
    ).astype(int)
    out["age_weight_dirt_veteran_flag"] = (out["surface_simple"].eq("dirt") & out["age_num"].ge(5)).astype(int)
    out["age_weight_dirt_veteran_fit_flag"] = (
        out["age_weight_dirt_veteran_flag"].eq(1)
        & (out["runner_heavy_vs_race_kg"].le(1.0) | out["race_weight_light_rank_score_num"].ge(0.45))
    ).astype(int)
    out["age_weight_turf_young_flag"] = (out["surface_simple"].eq("turf") & out["age_num"].le(3)).astype(int)
    out["age_weight_old_heavy_risk_flag"] = (
        out["age_num"].ge(6)
        & out["runner_heavy_vs_race_kg"].ge(1.0)
        & out["race_weight_range"].ge(2.0)
    ).astype(int)
    out["age_weight_immature_heavy_risk_flag"] = (
        out["age_num"].le(3)
        & out["runner_heavy_vs_race_kg"].ge(1.0)
        & out["surface_simple"].eq("dirt")
    ).astype(int)
    out["age_weight_positive_score"] = (
        0.50 * out["age_weight_summer_3yo_allowance_flag"]
        + 0.30 * out["age_weight_big_3yo_allowance_flag"]
        + 0.42 * out["age_weight_dirt_veteran_fit_flag"]
        + 0.18 * out["age_weight_turf_young_flag"]
        + 0.18 * norm01(out["runner_weight_from_high_kg"], 0.0, 4.0)
    ).clip(0.0, 1.0)
    out["age_weight_risk_score"] = (
        0.48 * out["age_weight_old_heavy_risk_flag"]
        + 0.35 * out["age_weight_immature_heavy_risk_flag"]
        + 0.14 * norm01(out["runner_heavy_vs_race_kg"], 0.0, 3.0)
    ).clip(0.0, 1.0)
    return out


def merge_runner(tickets: pd.DataFrame, runners: pd.DataFrame, prefix: str, no_col: str) -> pd.DataFrame:
    cols = [
        "race_id",
        "horse_no",
        "horse_name",
        "age_num",
        "age_bucket",
        "sex",
        "carried_weight_num",
        "prev_carried_weight_num",
        "weight_diff_num",
        "surface_simple",
        "distance_num",
        "going",
        "month",
        "popularity_num",
        "win_odds_num",
        "race_weight_median",
        "race_weight_min",
        "race_weight_max",
        "race_weight_range",
        "race_age_range",
        "race_mixed_3yo_older",
        "runner_weight_advantage_kg",
        "runner_weight_from_high_kg",
        "runner_heavy_vs_race_kg",
        "race_weight_light_rank_score_num",
        "age_weight_summer_3yo_allowance_flag",
        "age_weight_big_3yo_allowance_flag",
        "age_weight_dirt_veteran_flag",
        "age_weight_dirt_veteran_fit_flag",
        "age_weight_turf_young_flag",
        "age_weight_old_heavy_risk_flag",
        "age_weight_immature_heavy_risk_flag",
        "age_weight_positive_score",
        "age_weight_risk_score",
    ]
    use = runners[[c for c in cols if c in runners.columns]].copy()
    use = use.rename(columns={c: f"{prefix}_{c}" for c in use.columns if c not in {"race_id", "horse_no"}})
    out = tickets.copy()
    out["race_id"] = out["race_id"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(16)
    out[no_col] = num(out, no_col).astype("Int64")
    return out.merge(use, left_on=["race_id", no_col], right_on=["race_id", "horse_no"], how="left").drop(
        columns=["horse_no"], errors="ignore"
    )


def enrich_tickets(tickets: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = out["race_id"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(16)
    if "year" not in out.columns:
        out["year"] = pd.to_numeric(out["race_id"].str.slice(0, 4), errors="coerce")
    out = merge_runner(out, runners, "anchor", "anchor_no")
    out = merge_runner(out, runners, "partner", "partner_no")

    a_pos = num(out, "anchor_age_weight_positive_score", 0).fillna(0)
    p_pos = num(out, "partner_age_weight_positive_score", 0).fillna(0)
    a_risk = num(out, "anchor_age_weight_risk_score", 0).fillna(0)
    p_risk = num(out, "partner_age_weight_risk_score", 0).fillna(0)
    out["ticket_age_weight_positive_score"] = np.maximum(a_pos, p_pos)
    out["ticket_age_weight_risk_score"] = np.maximum(a_risk, p_risk)
    out["ticket_has_summer_3yo_allowance"] = (
        num(out, "anchor_age_weight_summer_3yo_allowance_flag", 0).fillna(0).eq(1)
        | num(out, "partner_age_weight_summer_3yo_allowance_flag", 0).fillna(0).eq(1)
    ).astype(int)
    out["ticket_has_big_3yo_allowance"] = (
        num(out, "anchor_age_weight_big_3yo_allowance_flag", 0).fillna(0).eq(1)
        | num(out, "partner_age_weight_big_3yo_allowance_flag", 0).fillna(0).eq(1)
    ).astype(int)
    out["ticket_has_dirt_veteran_fit"] = (
        num(out, "anchor_age_weight_dirt_veteran_fit_flag", 0).fillna(0).eq(1)
        | num(out, "partner_age_weight_dirt_veteran_fit_flag", 0).fillna(0).eq(1)
    ).astype(int)
    out["ticket_has_old_heavy_risk"] = (
        num(out, "anchor_age_weight_old_heavy_risk_flag", 0).fillna(0).eq(1)
        | num(out, "partner_age_weight_old_heavy_risk_flag", 0).fillna(0).eq(1)
    ).astype(int)
    out["ticket_pair_age_gap"] = (num(out, "anchor_age_num") - num(out, "partner_age_num")).abs()
    a_age = num(out, "anchor_age_num")
    p_age = num(out, "partner_age_num")
    out["ticket_pair_age_shape"] = np.select(
        [
            a_age.eq(3) & p_age.ge(4) | p_age.eq(3) & a_age.ge(4),
            a_age.ge(5) & p_age.ge(5),
            a_age.le(3) & p_age.le(3),
            a_age.ge(6) | p_age.ge(6),
        ],
        ["age3_vs_older", "both_veteran", "both_young", "has_age6plus"],
        default="mixed_other",
    )
    for col in ["runtime_stake_yen", "runtime_return_yen", "runtime_backtest_pay_per100"]:
        if col not in out.columns:
            out[col] = 0.0
    out["_base_stake"] = num(out, "runtime_stake_yen", 0).fillna(0)
    out["_base_return"] = num(out, "runtime_return_yen", 0).fillna(0)
    return out


def recalc_return(df: pd.DataFrame, stake: pd.Series) -> pd.Series:
    pay = num(df, "runtime_backtest_pay_per100", np.nan)
    pay = pay.fillna(num(df, "runtime_pay_per100", np.nan)).fillna(num(df, "quote_pay_proxy_per100", 0)).fillna(0)
    hit = df.get("hit", pd.Series(False, index=df.index)).astype(bool)
    if "runtime_return_yen" in df.columns:
        hit = hit | num(df, "runtime_return_yen", 0).fillna(0).gt(0)
    return np.where(hit, pay * stake / 100.0, 0.0)


def metrics(df: pd.DataFrame, label: str, stake_col: str = "_eval_stake", return_col: str = "_eval_return") -> dict:
    selected = df[num(df, stake_col, 0).fillna(0).gt(0)].copy()
    stake = float(num(selected, stake_col, 0).fillna(0).sum())
    ret = float(num(selected, return_col, 0).fillna(0).sum())
    races = int(selected["race_id"].nunique()) if len(selected) else 0
    hit_races = int(selected.loc[num(selected, return_col, 0).fillna(0).gt(0), "race_id"].nunique()) if len(selected) else 0
    if len(selected):
        by_race = (
            selected.sort_values(["year", "race_id"])
            .groupby("race_id", sort=False)
            .agg(stake=(stake_col, "sum"), ret=(return_col, "sum"))
        )
        pnl = by_race["ret"] - by_race["stake"]
        eq = pnl.cumsum()
        dd = eq - eq.cummax()
        max_dd = float(dd.min()) if len(dd) else 0.0
    else:
        max_dd = 0.0
    return {
        "policy": label,
        "tickets": int(len(selected)),
        "races": races,
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(num(selected, return_col, 0).fillna(0).gt(0).mean()) if len(selected) else 0.0,
        "race_hit_rate": hit_races / races if races else 0.0,
        "max_drawdown_yen": round(max_dd, 1),
    }


def apply_policy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    base_stake = num(out, "_base_stake", 0).fillna(0)
    keep = base_stake.gt(0)
    pos = num(out, "ticket_age_weight_positive_score", 0).fillna(0)
    risk = num(out, "ticket_age_weight_risk_score", 0).fillna(0)
    margin = num(out, "min_odds_margin_ratio", 0).fillna(0)
    expected_roi = num(out, "runtime_expected_roi", 0).fillna(0)
    strong_edge = margin.ge(params.get("strong_min_margin", 2.0)) & expected_roi.ge(
        params.get("strong_min_expected_roi", 1.55)
    )

    if params.get("strong_edge_only", False):
        keep &= strong_edge
    if params.get("positive_context_only", False):
        keep &= pos.ge(params.get("positive_threshold", 0.45))
    if params.get("require_extra_edge_for_age_risk", False):
        weak_edge = margin.lt(params.get("risk_min_margin", 2.0)) | expected_roi.lt(params.get("risk_min_expected_roi", 1.5))
        keep &= ~(risk.ge(params.get("risk_threshold", 0.45)) & weak_edge)
    if params.get("keep_positive_or_strong_edge", False):
        keep &= pos.ge(params.get("positive_threshold", 0.45)) | strong_edge
    if params.get("skip_old_heavy_risk", False):
        weak_edge = margin.lt(params.get("old_heavy_min_margin", 2.5)) | expected_roi.lt(
            params.get("old_heavy_min_expected_roi", 1.6)
        )
        keep &= ~(num(out, "ticket_has_old_heavy_risk", 0).fillna(0).eq(1) & weak_edge)
    if params.get("require_3yo_or_veteran_context", False):
        good = (
            num(out, "ticket_has_summer_3yo_allowance", 0).fillna(0).eq(1)
            | num(out, "ticket_has_dirt_veteran_fit", 0).fillna(0).eq(1)
            | margin.ge(params.get("context_min_margin", 2.5))
        )
        keep &= good

    stake = base_stake.where(keep, 0.0)
    if params.get("halve_age_risk", False):
        stake = np.where(risk.ge(params.get("halve_risk_threshold", 0.45)), np.floor(stake * 0.5 / 100.0) * 100.0, stake)
    if params.get("boost_positive", False):
        mult = np.where(pos.ge(params.get("boost_positive_threshold", 0.55)), params.get("positive_multiplier", 1.15), 1.0)
        stake = np.floor(stake * mult / 100.0) * 100.0

    out["_eval_stake"] = pd.Series(stake, index=out.index).clip(lower=0)
    out["_eval_return"] = recalc_return(out, out["_eval_stake"])
    return out


def build_policy_grid() -> list[dict]:
    rows: list[dict] = [{"name": "baseline_current"}]
    for margin in [1.5, 2.0, 2.5, 3.0]:
        rows.append(
            {
                "name": f"strong_edge_only_m{margin}",
                "strong_edge_only": True,
                "strong_min_margin": margin,
                "strong_min_expected_roi": 1.55,
            }
        )
    for threshold in [0.35, 0.45, 0.55, 0.65]:
        rows.append(
            {
                "name": f"age_positive_only_p{threshold}",
                "positive_context_only": True,
                "positive_threshold": threshold,
            }
        )
    for risk_th in [0.35, 0.45, 0.55, 0.65]:
        for margin in [1.5, 2.0, 2.5, 3.0]:
            rows.append(
                {
                    "name": f"age_risk_extra_edge_r{risk_th}_m{margin}",
                    "require_extra_edge_for_age_risk": True,
                    "risk_threshold": risk_th,
                    "risk_min_margin": margin,
                    "risk_min_expected_roi": 1.55,
                }
            )
    for threshold in [0.35, 0.45, 0.55]:
        for margin in [1.5, 2.0, 2.5, 3.0]:
            rows.append(
                {
                    "name": f"age_positive_or_edge_p{threshold}_m{margin}",
                    "keep_positive_or_strong_edge": True,
                    "positive_threshold": threshold,
                    "strong_min_margin": margin,
                    "strong_min_expected_roi": 1.55,
                }
            )
    for margin in [1.5, 2.0, 2.5, 3.0]:
        rows.append(
            {
                "name": f"skip_old_heavy_weak_edge_m{margin}",
                "skip_old_heavy_risk": True,
                "old_heavy_min_margin": margin,
                "old_heavy_min_expected_roi": 1.55,
            }
        )
    for margin in [1.5, 2.0, 2.5, 3.0]:
        rows.append(
            {
                "name": f"require_3yo_or_dirt_veteran_m{margin}",
                "require_3yo_or_veteran_context": True,
                "context_min_margin": margin,
            }
        )
    for risk_th in [0.35, 0.45, 0.55]:
        rows.append(
            {
                "name": f"halve_age_risk_r{risk_th}",
                "halve_age_risk": True,
                "halve_risk_threshold": risk_th,
            }
        )
    for threshold in [0.45, 0.55, 0.65]:
        rows.append(
            {
                "name": f"boost_age_positive_p{threshold}",
                "boost_positive": True,
                "boost_positive_threshold": threshold,
                "positive_multiplier": 1.20,
            }
        )
    for risk_th in [0.45, 0.55]:
        for pos_th in [0.45, 0.55]:
            rows.append(
                {
                    "name": f"hybrid_age_r{risk_th}_p{pos_th}",
                    "require_extra_edge_for_age_risk": True,
                    "risk_threshold": risk_th,
                    "risk_min_margin": 2.0,
                    "risk_min_expected_roi": 1.55,
                    "boost_positive": True,
                    "boost_positive_threshold": pos_th,
                    "positive_multiplier": 1.15,
                }
            )
    return rows


def segment_table(enriched: pd.DataFrame) -> pd.DataFrame:
    base = enriched[enriched["_base_stake"].gt(0)].copy()
    rows: list[dict] = []
    cuts = {
        "ticket_has_summer_3yo_allowance": "ticket_has_summer_3yo_allowance",
        "ticket_has_big_3yo_allowance": "ticket_has_big_3yo_allowance",
        "ticket_has_dirt_veteran_fit": "ticket_has_dirt_veteran_fit",
        "ticket_has_old_heavy_risk": "ticket_has_old_heavy_risk",
        "ticket_pair_age_shape": "ticket_pair_age_shape",
        "anchor_age_bucket": "anchor_age_bucket",
        "partner_age_bucket": "partner_age_bucket",
    }
    for label, col in cuts.items():
        if col not in base.columns:
            continue
        for value, g in base.groupby(col, dropna=False):
            tmp = g.copy()
            tmp["_eval_stake"] = tmp["_base_stake"]
            tmp["_eval_return"] = recalc_return(tmp, tmp["_eval_stake"])
            m = metrics(tmp, f"{label}={value}")
            m.update({"segment": label, "value": value})
            rows.append(m)
    bin_specs = {
        "ticket_age_weight_positive_score": [-0.01, 0.20, 0.40, 0.60, 1.01],
        "ticket_age_weight_risk_score": [-0.01, 0.20, 0.40, 0.60, 1.01],
        "ticket_pair_age_gap": [-0.01, 0.5, 1.5, 2.5, 10],
    }
    for col, bins in bin_specs.items():
        values = pd.cut(num(base, col, 0).fillna(0), bins=bins, include_lowest=True)
        for value, g in base.groupby(values, observed=False):
            tmp = g.copy()
            tmp["_eval_stake"] = tmp["_base_stake"]
            tmp["_eval_return"] = recalc_return(tmp, tmp["_eval_stake"])
            m = metrics(tmp, f"{col}={value}")
            m.update({"segment": f"{col}_bin", "value": str(value)})
            rows.append(m)
    return pd.DataFrame(rows)


def evaluate_policies(enriched: pd.DataFrame, train_max_year: int, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    policy_rows = []
    best_eval = None
    best_params = None
    best_score = -1e18
    for params in build_policy_grid():
        evaluated = apply_policy(enriched, params)
        all_m = metrics(evaluated, params["name"])
        train_m = metrics(evaluated[num(evaluated, "year", 0).le(train_max_year)], "train")
        test_m = metrics(evaluated[num(evaluated, "year", 0).eq(test_year)], "test")
        coverage_penalty = 0.0 if train_m["races"] >= 120 else -10.0
        score = train_m["roi"] + 0.08 * train_m["race_hit_rate"] + train_m["profit_yen"] / 250000.0 + coverage_penalty
        row = {**params, "score": score}
        row.update({f"all_{k}": v for k, v in all_m.items() if k != "policy"})
        row.update({f"train_{k}": v for k, v in train_m.items() if k != "policy"})
        row.update({f"test_{k}": v for k, v in test_m.items() if k != "policy"})
        policy_rows.append(row)
        if score > best_score:
            best_score = score
            best_params = params
            best_eval = evaluated
    return pd.DataFrame(policy_rows).sort_values("score", ascending=False), best_eval, best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate age x weight allowance x surface overlay.")
    parser.add_argument(
        "--tickets-csv",
        default="outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates_default/mcs_full_margin095_s0304_skip03119_selected_tickets.csv",
    )
    parser.add_argument(
        "--feature-csv",
        action="append",
        default=[
            "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features.csv",
            "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features.csv",
        ],
    )
    parser.add_argument("--output-dir", default="outputs/analysis/age_weight_surface_overlay_v1")
    parser.add_argument("--train-max-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    runners = load_runner_context(args.feature_csv)
    enriched = enrich_tickets(tickets, runners)
    out_dir = ensure_dir(args.output_dir)
    enriched.to_csv(out_dir / "age_weight_enriched_tickets.csv", index=False, encoding="utf-8-sig")
    segment_table(enriched).to_csv(out_dir / "segments.csv", index=False, encoding="utf-8-sig")

    policies, best_eval, best_params = evaluate_policies(enriched, args.train_max_year, args.test_year)
    policies.to_csv(out_dir / "policy_grid.csv", index=False, encoding="utf-8-sig")
    best_eval.to_csv(out_dir / "best_policy_tickets.csv", index=False, encoding="utf-8-sig")

    baseline = apply_policy(enriched, {"name": "baseline_current"})
    base_metrics = metrics(baseline, "baseline_current")
    best_metrics = metrics(best_eval, best_params["name"])
    pd.DataFrame([base_metrics, best_metrics]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    year_rows = []
    for year, _ in enriched.groupby(num(enriched, "year", 0), dropna=False):
        if pd.isna(year):
            continue
        y = int(year)
        year_rows.append(metrics(baseline[num(baseline, "year", 0).eq(y)], f"baseline_{y}"))
        year_rows.append(metrics(best_eval[num(best_eval, "year", 0).eq(y)], f"{best_params['name']}_{y}"))
    pd.DataFrame(year_rows).to_csv(out_dir / "yearly_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "tickets_csv": args.tickets_csv,
        "feature_csv": args.feature_csv,
        "output_dir": str(out_dir),
        "runner_rows": int(len(runners)),
        "ticket_rows": int(len(enriched)),
        "matched_anchor_rate": float(enriched["anchor_age_num"].notna().mean()) if "anchor_age_num" in enriched else 0.0,
        "matched_partner_rate": float(enriched["partner_age_num"].notna().mean()) if "partner_age_num" in enriched else 0.0,
        "baseline": base_metrics,
        "best_policy": best_params,
        "best": best_metrics,
        "top_policy_rows": policies.head(10).to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(summary), f, ensure_ascii=False, indent=2)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
