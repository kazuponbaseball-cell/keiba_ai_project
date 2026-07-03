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
    return pd.to_numeric(df[col], errors="coerce")


def norm01(series: pd.Series, lo: float, hi: float) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0)


def parse_ticket_dates(df: pd.DataFrame) -> pd.Series:
    if "date_key" in df.columns:
        parsed = pd.to_datetime(df["date_key"], errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=df.index)
    missing = parsed.isna()
    if missing.any() and "race_id" in df.columns:
        digits = df.loc[missing, "race_id"].astype(str).str.extract(r"(\d{8})", expand=False)
        parsed.loc[missing] = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    return parsed


def read_feature_file(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    # The feature cache uses Japanese source columns at stable positions. Keep this
    # script ASCII-safe by resolving those names from the header instead of writing
    # the literals directly.
    by_pos = {
        "date_raw": 0,
        "date_s": 1,
        "venue": 2,
        "race_no": 3,
        "horse_name": 7,
        "horse_no": 15,
        "popularity": 16,
        "win_odds": 17,
        "surface": 18,
        "going": 20,
        "career": 33,
        "race_id": 34,
        "finish": 38,
        "prev_popularity": 49,
        "prev_final3f_rank": 60,
        "target_top3": 73,
    }
    resolved = {k: header[v] for k, v in by_pos.items() if v < len(header)}
    ascii_cols = [
        "same_distance_category_starts",
        "same_venue_starts",
        "horse_turf_starts",
        "horse_dirt_starts",
        "prev_race_time_value",
        "prev_class_time_value_score",
        "horse_time_value_plus_margin",
    ]
    usecols = list(resolved.values()) + [c for c in ascii_cols if c in header]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    rename = {v: k for k, v in resolved.items()}
    df = df.rename(columns=rename)
    for col in ascii_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_runner_features(paths: list[str]) -> pd.DataFrame:
    frames = []
    for raw in paths:
        path = project_path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(read_feature_file(path))
    out = pd.concat(frames, ignore_index=True)
    out["race_id"] = out["race_id"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(16)
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out = out.drop_duplicates(["race_id", "horse_no"], keep="last")
    idx = out.index
    career = num(out, "career", 99).fillna(99)
    pop = num(out, "popularity", 99).fillna(99)
    odds = num(out, "win_odds", np.nan)
    same_dist = num(out, "same_distance_category_starts", 0).fillna(0)
    same_venue = num(out, "same_venue_starts", 0).fillna(0)
    turf_starts = num(out, "horse_turf_starts", 0).fillna(0)
    dirt_starts = num(out, "horse_dirt_starts", 0).fillna(0)
    surface_text = out.get("surface", pd.Series("", index=idx)).astype(str)
    is_turf = surface_text.str.contains("\u829d", regex=False)
    is_dirt = surface_text.str.contains("\u30c0", regex=False)
    current_surface_starts = pd.Series(np.nan, index=idx, dtype=float)
    current_surface_starts.loc[is_turf] = turf_starts.loc[is_turf]
    current_surface_starts.loc[is_dirt] = dirt_starts.loc[is_dirt]
    current_surface_starts = current_surface_starts.fillna(np.minimum(turf_starts, dirt_starts))

    out["fc_debutish_flag"] = career.le(1).astype(int)
    out["fc_low_career_flag"] = career.le(2).astype(int)
    out["fc_first_distance_category_flag"] = same_dist.le(0).astype(int)
    out["fc_low_distance_category_sample_flag"] = same_dist.le(1).astype(int)
    out["fc_first_venue_flag"] = same_venue.le(0).astype(int)
    out["fc_first_surface_flag"] = current_surface_starts.le(0).astype(int)
    out["fc_low_surface_sample_flag"] = current_surface_starts.le(1).astype(int)
    out["fc_market_supported_flag"] = (pop.le(2) | odds.le(4.0)).astype(int)
    out["fc_market_respected_flag"] = (pop.le(4) | odds.le(8.0)).astype(int)
    prev_time_value = np.maximum(
        norm01(num(out, "prev_class_time_value_score", 0).fillna(0), lo=-0.08, hi=0.22),
        norm01(num(out, "prev_race_time_value", 0).fillna(0), lo=-0.10, hi=0.25),
    )
    prev_time_margin = norm01(num(out, "horse_time_value_plus_margin", 0).fillna(0), lo=-0.55, hi=0.18)
    prev_final3f_rank = num(out, "prev_final3f_rank", 9).fillna(9)
    prev_finish_rank_score = (1.0 - norm01(prev_final3f_rank, lo=1.0, hi=9.0)).clip(0.0, 1.0)
    prev_popularity = num(out, "prev_popularity", 99).fillna(99)
    prev_market_score = (1.0 - norm01(prev_popularity, lo=1.0, hi=9.0)).clip(0.0, 1.0)
    out["fc_any_first_condition_flag"] = (
        out[
            [
                "fc_debutish_flag",
                "fc_first_distance_category_flag",
                "fc_first_venue_flag",
                "fc_first_surface_flag",
            ]
        ]
        .max(axis=1)
        .astype(int)
    )
    risk = (
        0.35 * out["fc_debutish_flag"]
        + 0.20 * out["fc_low_career_flag"]
        + 0.25 * out["fc_first_distance_category_flag"]
        + 0.15 * out["fc_first_venue_flag"]
        + 0.25 * out["fc_first_surface_flag"]
        + 0.10 * out["fc_low_surface_sample_flag"]
    )
    out["fc_runner_uncertainty_score"] = risk.clip(0, 1)
    out["fc_prev_impressive_score"] = (
        out["fc_low_career_flag"]
        * out["fc_market_respected_flag"]
        * (
            0.46 * prev_time_value
            + 0.28 * prev_time_margin
            + 0.16 * prev_finish_rank_score
            + 0.10 * prev_market_score
        )
    ).clip(0.0, 1.0)
    out["fc_impressive_supported_flag"] = (
        out["fc_prev_impressive_score"].ge(0.60) & out["fc_market_supported_flag"].eq(1)
    ).astype(int)
    out["fc_net_uncertainty_score"] = (
        out["fc_runner_uncertainty_score"] * (1.0 - 0.42 * out["fc_prev_impressive_score"])
    ).clip(0.0, 1.0)
    out["fc_supported_uncertainty_score"] = out["fc_runner_uncertainty_score"] * out["fc_market_supported_flag"]
    return out


def race_aggregates(runners: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for race_id, g in runners.groupby("race_id", sort=False):
        supported = g[g["fc_market_supported_flag"].eq(1)]
        first_pop = g[g["fc_market_respected_flag"].eq(1) & g["fc_any_first_condition_flag"].eq(1)]
        rows.append(
            {
                "race_id": race_id,
                "race_fc_supported_uncertain_count": int((supported["fc_runner_uncertainty_score"] >= 0.35).sum()),
                "race_fc_supported_uncertain_max": float(supported["fc_runner_uncertainty_score"].max()) if len(supported) else 0.0,
                "race_fc_first_condition_respected_count": int(len(first_pop)),
                "race_fc_any_top3": int(
                    (
                        g["fc_any_first_condition_flag"].eq(1)
                        & pd.to_numeric(g.get("target_top3"), errors="coerce").fillna(0).eq(1)
                    ).any()
                ),
                "race_fc_supported_top3": int(
                    (
                        g["fc_market_supported_flag"].eq(1)
                        & g["fc_runner_uncertainty_score"].ge(0.35)
                        & pd.to_numeric(g.get("target_top3"), errors="coerce").fillna(0).eq(1)
                    ).any()
                ),
            }
        )
    return pd.DataFrame(rows)


def merge_runner(tickets: pd.DataFrame, runners: pd.DataFrame, prefix: str, no_col: str) -> pd.DataFrame:
    cols = [
        "race_id",
        "horse_no",
        "horse_name",
        "popularity",
        "win_odds",
        "career",
        "same_distance_category_starts",
        "same_venue_starts",
        "horse_turf_starts",
        "horse_dirt_starts",
        "fc_debutish_flag",
        "fc_low_career_flag",
        "fc_first_distance_category_flag",
        "fc_low_distance_category_sample_flag",
        "fc_first_venue_flag",
        "fc_first_surface_flag",
        "fc_low_surface_sample_flag",
        "fc_market_supported_flag",
        "fc_market_respected_flag",
        "fc_any_first_condition_flag",
        "fc_runner_uncertainty_score",
        "fc_prev_impressive_score",
        "fc_impressive_supported_flag",
        "fc_net_uncertainty_score",
        "fc_supported_uncertainty_score",
    ]
    use = runners[[c for c in cols if c in runners.columns]].copy()
    use = use.rename(columns={c: f"{prefix}_{c}" for c in use.columns if c not in {"race_id", "horse_no"}})
    out = tickets.copy()
    out[no_col] = pd.to_numeric(out[no_col], errors="coerce").astype("Int64")
    return out.merge(use, left_on=["race_id", no_col], right_on=["race_id", "horse_no"], how="left").drop(columns=["horse_no"], errors="ignore")


def enrich_tickets(tickets: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = out["race_id"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(16)
    out["_date"] = parse_ticket_dates(out)
    if "year" not in out.columns:
        out["year"] = out["_date"].dt.year
    out = merge_runner(out, runners, "anchor", "anchor_no")
    out = merge_runner(out, runners, "partner", "partner_no")
    out = out.merge(race_aggregates(runners), on="race_id", how="left")
    idx = out.index
    a_unc = num(out, "anchor_fc_runner_uncertainty_score", 0).fillna(0)
    p_unc = num(out, "partner_fc_runner_uncertainty_score", 0).fillna(0)
    a_net_unc = num(out, "anchor_fc_net_uncertainty_score", 0).fillna(a_unc)
    p_net_unc = num(out, "partner_fc_net_uncertainty_score", 0).fillna(p_unc)
    a_impact = num(out, "anchor_fc_prev_impressive_score", 0).fillna(0)
    p_impact = num(out, "partner_fc_prev_impressive_score", 0).fillna(0)
    a_support = num(out, "anchor_fc_market_supported_flag", 0).fillna(0)
    p_support = num(out, "partner_fc_market_supported_flag", 0).fillna(0)
    a_supported_uncertain = ((a_unc >= 0.35) & a_support.eq(1)).astype(int)
    p_supported_uncertain = ((p_unc >= 0.35) & p_support.eq(1)).astype(int)
    out["ticket_fc_pair_uncertainty_score"] = np.maximum(a_unc, p_unc)
    out["ticket_fc_pair_raw_uncertainty_score"] = out["ticket_fc_pair_uncertainty_score"]
    out["ticket_fc_pair_net_uncertainty_score"] = np.maximum(a_net_unc, p_net_unc)
    out["ticket_fc_pair_impressive_prev_score"] = np.maximum(a_impact, p_impact)
    out["ticket_fc_pair_mean_uncertainty_score"] = (a_unc + p_unc) / 2.0
    out["ticket_fc_has_supported_uncertain_runner"] = np.maximum(a_supported_uncertain, p_supported_uncertain)
    out["ticket_fc_supported_uncertain_in_ticket_count"] = a_supported_uncertain + p_supported_uncertain
    race_count = num(out, "race_fc_supported_uncertain_count", 0).fillna(0)
    out["race_fc_supported_uncertain_not_in_ticket_flag"] = (
        race_count.sub(out["ticket_fc_supported_uncertain_in_ticket_count"]).gt(0)
    ).astype(int)
    race_first_count = num(out, "race_fc_first_condition_respected_count", 0).fillna(0)
    a_first_respected = (
        num(out, "anchor_fc_any_first_condition_flag", 0).fillna(0).eq(1)
        & num(out, "anchor_fc_market_respected_flag", 0).fillna(0).eq(1)
    ).astype(int)
    p_first_respected = (
        num(out, "partner_fc_any_first_condition_flag", 0).fillna(0).eq(1)
        & num(out, "partner_fc_market_respected_flag", 0).fillna(0).eq(1)
    ).astype(int)
    out["race_fc_first_condition_respected_not_in_ticket_flag"] = (
        race_first_count.sub(a_first_respected + p_first_respected).gt(0)
    ).astype(int)
    out["ticket_fc_race_uncertainty_score"] = (
        0.45 * out["race_fc_supported_uncertain_not_in_ticket_flag"]
        + 0.25 * out["race_fc_first_condition_respected_not_in_ticket_flag"]
        + 0.30 * out["ticket_fc_pair_uncertainty_score"]
        - 0.15 * out["ticket_fc_has_supported_uncertain_runner"]
    ).clip(0, 1)
    out["ticket_fc_low_data_alert_flag"] = out["ticket_fc_race_uncertainty_score"].ge(0.55).astype(int)
    for col in ["runtime_stake_yen", "runtime_return_yen", "runtime_backtest_pay_per100"]:
        if col not in out.columns:
            out[col] = np.nan
    out["_base_stake"] = (
        num(out, "runtime_stake_yen", np.nan)
        .fillna(num(out, "eval_stake_yen", np.nan))
        .fillna(num(out, "scaled_stake_yen", np.nan))
        .fillna(num(out, "stake_yen", 0))
        .fillna(0)
    )
    out["_base_return"] = (
        num(out, "runtime_return_yen", np.nan)
        .fillna(num(out, "eval_return_yen", np.nan))
        .fillna(num(out, "scaled_return_yen", np.nan))
        .fillna(num(out, "return_yen", 0))
        .fillna(0)
    )
    return out


def metrics(df: pd.DataFrame, label: str, stake_col: str = "_eval_stake", return_col: str = "_eval_return") -> dict:
    selected = df[num(df, stake_col, 0).fillna(0).gt(0)].copy()
    stake = float(num(selected, stake_col, 0).fillna(0).sum())
    returns = num(selected, return_col, 0).fillna(0)
    ret = float(returns.sum())
    races = int(selected["race_id"].nunique()) if len(selected) else 0
    hit_races = int(selected.loc[num(selected, return_col, 0).fillna(0).gt(0), "race_id"].nunique()) if len(selected) else 0
    if len(selected):
        by_race = (
            selected.sort_values(["_date", "race_id"])
            .groupby("race_id", sort=False)
            .agg(stake=(stake_col, "sum"), ret=(return_col, "sum"))
        )
        pnl = by_race["ret"] - by_race["stake"]
        eq = pnl.cumsum()
        dd = eq - eq.cummax()
        max_dd = float(dd.min()) if len(dd) else 0.0
    else:
        max_dd = 0.0
    sorted_returns = returns.sort_values(ascending=False).reset_index(drop=True)
    top1_removed_return = float(sorted_returns.iloc[1:].sum()) if len(sorted_returns) > 1 else 0.0
    top3_removed_return = float(sorted_returns.iloc[3:].sum()) if len(sorted_returns) > 3 else 0.0
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
        "top1_removed_roi": top1_removed_return / stake if stake else 0.0,
        "top3_removed_roi": top3_removed_return / stake if stake else 0.0,
    }


def recalc_return(df: pd.DataFrame, stake: pd.Series) -> pd.Series:
    pay = num(df, "runtime_backtest_pay_per100", np.nan)
    pay = pay.fillna(num(df, "runtime_pay_per100", np.nan)).fillna(num(df, "quote_pay_proxy_per100", 0)).fillna(0)
    hit = df.get("hit", pd.Series(False, index=df.index)).astype(bool)
    if "runtime_return_yen" in df.columns:
        hit = hit | num(df, "runtime_return_yen", 0).fillna(0).gt(0)
    direct_base_stake = num(df, "_base_stake", np.nan)
    direct_base_return = num(df, "_base_return", np.nan)
    has_direct_return = direct_base_stake.gt(0) & direct_base_return.notna()
    proportional = np.where(
        has_direct_return,
        direct_base_return * stake / direct_base_stake.replace(0, np.nan),
        np.nan,
    )
    pay_based = np.where(hit, pay * stake / 100.0, 0.0)
    return pd.Series(proportional, index=df.index).fillna(pd.Series(pay_based, index=df.index)).fillna(0.0)


def apply_policy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    base_stake = num(out, "_base_stake", 0).fillna(0)
    keep = base_stake.gt(0)
    risk = num(out, "ticket_fc_race_uncertainty_score", 0).fillna(0)
    pair_unc_col = "ticket_fc_pair_net_uncertainty_score" if params.get("use_impressive_rescue", False) else "ticket_fc_pair_uncertainty_score"
    pair_unc = num(out, pair_unc_col, 0).fillna(0)
    race_missing = num(out, "race_fc_supported_uncertain_not_in_ticket_flag", 0).fillna(0).eq(1)
    margin = num(out, "min_odds_margin_ratio", 0).fillna(0)
    expected_roi = num(out, "runtime_expected_roi", 0).fillna(0)

    if params.get("skip_unmodeled_supported", False):
        keep &= ~(race_missing & risk.ge(params.get("race_risk_threshold", 0.55)))
    if params.get("require_extra_edge_for_pair_uncertainty", False):
        weak_edge = margin.lt(params.get("uncertain_min_margin", 2.5)) | expected_roi.lt(params.get("uncertain_min_expected_roi", 1.6))
        keep &= ~(pair_unc.ge(params.get("pair_uncertainty_threshold", 0.65)) & weak_edge)

    stake = base_stake.where(keep, 0.0)
    if params.get("soft_halve_high_risk", False):
        stake = np.where(risk.ge(params.get("soft_risk_threshold", 0.65)), np.floor(stake * 0.5 / 100.0) * 100.0, stake)
    out["_eval_stake"] = pd.Series(stake, index=out.index).clip(lower=0)
    out["_eval_return"] = recalc_return(out, out["_eval_stake"])
    return out


def build_policy_grid() -> list[dict]:
    rows: list[dict] = [{"name": "baseline_current", "skip_unmodeled_supported": False}]
    for risk_th in [0.45, 0.55, 0.65, 0.75]:
        rows.append(
            {
                "name": f"skip_unmodeled_supported_risk{risk_th}",
                "skip_unmodeled_supported": True,
                "race_risk_threshold": risk_th,
            }
        )
    for pair_th in [0.45, 0.60, 0.75]:
        for min_margin in [1.5, 2.0, 2.5, 3.0]:
            rows.append(
                {
                    "name": f"pair_uncertainty_edge_u{pair_th}_m{min_margin}",
                    "require_extra_edge_for_pair_uncertainty": True,
                    "pair_uncertainty_threshold": pair_th,
                    "uncertain_min_margin": min_margin,
                    "uncertain_min_expected_roi": 1.6,
                }
            )
            rows.append(
                {
                    "name": f"impressive_rescue_pair_uncertainty_edge_u{pair_th}_m{min_margin}",
                    "require_extra_edge_for_pair_uncertainty": True,
                    "use_impressive_rescue": True,
                    "pair_uncertainty_threshold": pair_th,
                    "uncertain_min_margin": min_margin,
                    "uncertain_min_expected_roi": 1.6,
                }
            )
    for risk_th in [0.55, 0.65, 0.75]:
        rows.append(
            {
                "name": f"soft_halve_high_risk{risk_th}",
                "soft_halve_high_risk": True,
                "soft_risk_threshold": risk_th,
            }
        )
    for risk_th in [0.45, 0.55, 0.65]:
        for pair_th in [0.60, 0.75]:
            rows.append(
                {
                    "name": f"hybrid_r{risk_th}_u{pair_th}",
                    "skip_unmodeled_supported": True,
                    "race_risk_threshold": risk_th,
                    "require_extra_edge_for_pair_uncertainty": True,
                    "pair_uncertainty_threshold": pair_th,
                    "uncertain_min_margin": 2.5,
                    "uncertain_min_expected_roi": 1.6,
                }
            )
    return rows


def segment_table(enriched: pd.DataFrame) -> pd.DataFrame:
    base = enriched[enriched["_base_stake"].gt(0)].copy()
    rows = []
    cuts = {
        "race_supported_uncertain_not_in_ticket": "race_fc_supported_uncertain_not_in_ticket_flag",
        "race_first_condition_respected_not_in_ticket": "race_fc_first_condition_respected_not_in_ticket_flag",
        "ticket_low_data_alert": "ticket_fc_low_data_alert_flag",
        "ticket_has_supported_uncertain_runner": "ticket_fc_has_supported_uncertain_runner",
    }
    for label, col in cuts.items():
        if col not in base.columns:
            continue
        for value, g in base.groupby(col, dropna=False):
            tmp = g.copy()
            tmp["_eval_stake"] = tmp["_base_stake"]
            tmp["_eval_return"] = tmp["_base_return"]
            m = metrics(tmp, f"{label}={int(value) if pd.notna(value) else value}")
            m.update({"segment": label, "value": value})
            rows.append(m)
    bins = pd.cut(
        num(base, "ticket_fc_race_uncertainty_score", 0).fillna(0),
        bins=[-0.01, 0.25, 0.45, 0.65, 1.01],
        labels=["0-0.25", "0.25-0.45", "0.45-0.65", "0.65-1.0"],
    )
    for value, g in base.groupby(bins, observed=False):
        tmp = g.copy()
        tmp["_eval_stake"] = tmp["_base_stake"]
        tmp["_eval_return"] = tmp["_base_return"]
        m = metrics(tmp, f"risk_bin={value}")
        m.update({"segment": "ticket_fc_race_uncertainty_score_bin", "value": str(value)})
        rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate first-condition / sparse-history uncertainty overlay.")
    parser.add_argument(
        "--tickets-csv",
        default="outputs/analysis/race_day_runtime_operation_skip03119_smoke_v1/mcs_pbo_overlay/selected_after_live_safety.csv",
    )
    parser.add_argument("--feature-csv", action="append", default=None)
    parser.add_argument("--output-dir", default="outputs/analysis/first_condition_uncertainty_overlay_v1")
    parser.add_argument("--train-max-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    args = parser.parse_args()

    feature_csv = args.feature_csv or [
        "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features.csv",
        "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features.csv",
    ]

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    runners = load_runner_features(feature_csv)
    enriched = enrich_tickets(tickets, runners)
    out_dir = ensure_dir(args.output_dir)
    enriched.to_csv(out_dir / "first_condition_enriched_tickets.csv", index=False, encoding="utf-8-sig")
    segment_table(enriched).to_csv(out_dir / "segments.csv", index=False, encoding="utf-8-sig")

    policy_rows = []
    best = None
    best_score = -1e18
    for params in build_policy_grid():
        evaluated = apply_policy(enriched, params)
        all_m = metrics(evaluated, params["name"])
        train_m = metrics(evaluated[num(evaluated, "year", 0).le(args.train_max_year)], "train")
        test_m = metrics(evaluated[num(evaluated, "year", 0).eq(args.test_year)], "test")
        # Prefer robust train improvement without destroying coverage; test 2026 is tiny, so
        # it is treated as a guardrail instead of the optimizer target.
        train_race_penalty = 0.0 if train_m["races"] >= 120 else -10.0
        score = train_m["roi"] + 0.10 * train_m["race_hit_rate"] + train_m["profit_yen"] / 250000.0 + train_race_penalty
        row = {**params, "score": score}
        row.update({f"all_{k}": v for k, v in all_m.items() if k != "policy"})
        row.update({f"train_{k}": v for k, v in train_m.items() if k != "policy"})
        row.update({f"test_{k}": v for k, v in test_m.items() if k != "policy"})
        policy_rows.append(row)
        if score > best_score:
            best_score = score
            best = params
            best_eval = evaluated

    policies = pd.DataFrame(policy_rows).sort_values("score", ascending=False)
    policies.to_csv(out_dir / "policy_grid.csv", index=False, encoding="utf-8-sig")
    best_eval.to_csv(out_dir / "best_policy_tickets.csv", index=False, encoding="utf-8-sig")

    baseline = apply_policy(enriched, {"name": "baseline_current"})
    base_metrics = metrics(baseline, "baseline_current")
    best_metrics = metrics(best_eval, best["name"])
    year_rows = []
    for year, _ in enriched.groupby(num(enriched, "year", 0), dropna=False):
        if pd.isna(year):
            continue
        year = int(year)
        year_rows.append(metrics(baseline[num(baseline, "year", 0).eq(year)], f"baseline_{year}"))
        year_rows.append(metrics(best_eval[num(best_eval, "year", 0).eq(year)], f"{best['name']}_{year}"))
    yearly = pd.DataFrame(year_rows)
    yearly.to_csv(out_dir / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([base_metrics, best_metrics]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "tickets_csv": args.tickets_csv,
        "feature_csv": feature_csv,
        "output_dir": str(out_dir),
        "runner_rows": int(len(runners)),
        "ticket_rows": int(len(enriched)),
        "matched_anchor_rate": float(enriched["anchor_career"].notna().mean()) if "anchor_career" in enriched else 0.0,
        "matched_partner_rate": float(enriched["partner_career"].notna().mean()) if "partner_career" in enriched else 0.0,
        "best_policy": best,
        "baseline": base_metrics,
        "best": best_metrics,
        "top_policy_rows": policies.head(10).to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(summary), f, ensure_ascii=False, indent=2)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
