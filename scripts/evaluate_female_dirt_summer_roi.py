from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_FEATURES = [
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
    "body_weight_backfilled/owner_breeder_enriched/"
    "train_features_with_same_day_bias_v3_retro_body_breeder.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
    "body_weight_backfilled/owner_breeder_enriched/"
    "test_features_with_same_day_bias_v3_retro_body_breeder.csv",
]

DEFAULT_RUNNER_EVAL_FEATURES = [
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
    "body_weight_backfilled/owner_breeder_enriched/"
    "test_features_with_same_day_bias_v3_retro_body_breeder.csv",
]

DEFAULT_TICKETS = [
    "outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates/recommended_runtime_tickets.csv",
    "outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates/recommended_all_tickets.csv",
]

FACTOR_COLS = [
    "prev_female_only_win_current_dirt",
    "prev_female_only_win_prev_dirt_current_dirt",
    "prev_female_only_win_class_up_current_dirt",
    "prev_female_only_win_to_mixed_dirt",
    "prev_female_only_top3_current_dirt",
    "summer_female",
    "summer_female_dirt",
    "summer_female_turf",
    "summer_female_local",
    "summer_female_lightweight",
    "summer_female_large_loss",
    "summer_female_young",
]

LOCAL_VENUES = {
    "札幌",
    "函館",
    "福島",
    "新潟",
    "小倉",
    "\u672d\u5e4c",
    "\u51fd\u9928",
    "\u798f\u5cf6",
    "\u65b0\u6f5f",
    "\u5c0f\u5009",
}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def ensure_dir(value: str | Path) -> Path:
    path = project_path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_ready(value: Any) -> Any:
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


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    raw = frame[col]
    if raw.dtype == object:
        raw = raw.astype(str).str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(raw, errors="coerce").fillna(default)


def text(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=str)
    return frame[col].astype("string").fillna(default).astype(str).str.strip()


def clean_race_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(16)


def clean_horse_id(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def read_feature_file(path: Path) -> pd.DataFrame:
    header = read_csv(path, nrows=0).columns.tolist()
    by_pos = {
        "date_raw": 0,
        "date_s": 1,
        "venue": 2,
        "race_no": 3,
        "race_name": 5,
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
        "win_pay": 31,
        "place_pay": 32,
        "career": 33,
        "race_id": 34,
        "horse_id": 35,
        "finish": 38,
        "prev_class_name": 44,
        "prev_surface": 50,
        "prev_distance": 51,
        "prev_going": 52,
        "prev_race_id": 68,
    }
    resolved = {key: header[pos] for key, pos in by_pos.items() if pos < len(header)}
    optional_cols = [
        "rotation_class_up_flag",
        "rotation_class_down_flag",
        "body_prev_weight",
        "body_prev_delta",
        "body_prev_large_loss_flag",
        "body_female_large_loss_flag",
        "body_large_horse_flag",
        "body_very_large_horse_flag",
        "race_weight_light_rank_score",
        "target_win",
        "target_top3",
    ]
    usecols = list(dict.fromkeys(list(resolved.values()) + [c for c in optional_cols if c in header]))
    df = read_csv(path, usecols=usecols)
    df = df.rename(columns={v: k for k, v in resolved.items()})
    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_runner_features(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw in paths:
        path = project_path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(read_feature_file(path))
    df = pd.concat(frames, ignore_index=True)
    df["race_id"] = clean_race_id(df["race_id"])
    df["prev_race_id"] = clean_race_id(df["prev_race_id"])
    df["horse_id"] = clean_horse_id(df["horse_id"])
    df["horse_no"] = num(df, "horse_no").astype("Int64")
    df = df[df["race_id"].str.len().ge(8) & df["horse_no"].notna()].copy()
    df = df.drop_duplicates(["race_id", "horse_no"], keep="last")
    return add_female_dirt_summer_features(df)


def surface_simple(series: pd.Series) -> pd.Series:
    raw = series.astype("string").fillna("").astype(str)
    return pd.Series(
        np.select(
            [
                raw.str.contains("\u30c0", regex=False, na=False),
                raw.str.contains("\u829d", regex=False, na=False),
            ],
            ["dirt", "turf"],
            default="other",
        ),
        index=series.index,
    )


def build_race_context(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["is_female"] = text(work, "sex").eq("\u725d")
    grouped = work.groupby("race_id", sort=False)
    race = grouped.agg(
        runners=("horse_no", "count"),
        female_count=("is_female", "sum"),
        race_name=("race_name", "first"),
        surface=("surface", "first"),
    ).reset_index()
    name = race["race_name"].astype("string").fillna("").astype(str)
    race["race_name_has_female"] = name.str.contains("\u725d", regex=False)
    race["female_only_race"] = race["female_count"].eq(race["runners"]) | race["race_name_has_female"]
    race["race_surface_simple"] = surface_simple(race["surface"])
    return race[["race_id", "female_only_race", "race_surface_simple"]]


def add_female_dirt_summer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = pd.to_numeric(out["race_id"].str.slice(0, 4), errors="coerce")
    out["month"] = pd.to_numeric(out["race_id"].str.slice(4, 6), errors="coerce")
    out["current_surface_simple"] = surface_simple(out["surface"])
    out["prev_surface_simple"] = surface_simple(out["prev_surface"])
    out["is_female"] = text(out, "sex").eq("\u725d")
    out["is_summer"] = out["month"].between(6, 9, inclusive="both")
    out["age_num"] = num(out, "age")
    out["carried_weight_num"] = num(out, "carried_weight")
    out["finish_num"] = num(out, "finish")
    out["popularity_num"] = num(out, "popularity")
    out["win_odds_num"] = num(out, "win_odds")
    out["win_pay_num"] = num(out, "win_pay", 0.0)
    out["place_pay_num"] = num(out, "place_pay", 0.0)
    out["rotation_class_up_num"] = num(out, "rotation_class_up_flag", 0.0)
    out["body_female_large_loss_num"] = np.maximum(
        num(out, "body_female_large_loss_flag", 0.0),
        out["is_female"].astype(float) * num(out, "body_prev_large_loss_flag", 0.0),
    )

    race_context = build_race_context(out)
    out = out.merge(
        race_context.rename(
            columns={
                "female_only_race": "current_female_only_race",
                "race_surface_simple": "current_race_surface_simple",
            }
        ),
        on="race_id",
        how="left",
    )

    # The raw previous-race ID is sparse in some cache files. Reconstruct the
    # previous start from each horse's chronological rows and use it as the
    # primary fallback. This mirrors the safer female-switch model.
    ordered = out.sort_values(["horse_id", "race_id", "horse_no"]).copy()
    ordered["_prev_seq_female_only_race"] = ordered.groupby("horse_id", sort=False)[
        "current_female_only_race"
    ].shift(1)
    ordered["_prev_seq_finish_num"] = ordered.groupby("horse_id", sort=False)["finish_num"].shift(1)
    ordered["_prev_seq_surface_simple"] = ordered.groupby("horse_id", sort=False)[
        "current_surface_simple"
    ].shift(1)
    ordered["_prev_seq_race_id"] = ordered.groupby("horse_id", sort=False)["race_id"].shift(1)
    out = out.merge(
        ordered[
            [
                "race_id",
                "horse_no",
                "_prev_seq_female_only_race",
                "_prev_seq_finish_num",
                "_prev_seq_surface_simple",
                "_prev_seq_race_id",
            ]
        ],
        on=["race_id", "horse_no"],
        how="left",
    )

    prev_context = race_context.rename(
        columns={
            "race_id": "prev_race_id",
            "female_only_race": "prev_female_only_race",
            "race_surface_simple": "prev_race_surface_simple",
        }
    )
    out = out.merge(prev_context, on="prev_race_id", how="left")

    prev_lookup = out[["race_id", "horse_id", "finish_num", "current_surface_simple"]].drop_duplicates(
        ["race_id", "horse_id"], keep="last"
    )
    prev_lookup = prev_lookup.rename(
        columns={
            "race_id": "prev_race_id",
            "finish_num": "prev_finish_num",
            "current_surface_simple": "prev_runner_surface_simple",
        }
    )
    out = out.merge(prev_lookup, on=["prev_race_id", "horse_id"], how="left")
    out["prev_runner_surface_simple"] = (
        out["prev_runner_surface_simple"]
        .fillna(out["_prev_seq_surface_simple"])
        .fillna(out["prev_surface_simple"])
    )
    out["prev_female_only_race"] = out["prev_female_only_race"].combine_first(
        out["_prev_seq_female_only_race"]
    )
    out["prev_female_only_race"] = out["prev_female_only_race"].fillna(False).astype(bool)
    out["current_female_only_race"] = out["current_female_only_race"].fillna(False).astype(bool)
    out["prev_finish_for_angle"] = num(out, "prev_finish_num", np.nan).fillna(
        num(out, "_prev_seq_finish_num", 99)
    )
    out["prev_win"] = num(out, "prev_finish_for_angle", 99).eq(1)
    out["prev_top3"] = num(out, "prev_finish_for_angle", 99).le(3)
    out["current_dirt"] = out["current_surface_simple"].eq("dirt")
    out["current_turf"] = out["current_surface_simple"].eq("turf")
    out["prev_dirt"] = out["prev_runner_surface_simple"].eq("dirt")
    out["current_mixed_sex_race"] = ~out["current_female_only_race"]

    out["prev_female_only_win_current_dirt"] = (
        out["is_female"] & out["prev_female_only_race"] & out["prev_win"] & out["current_dirt"]
    ).astype(int)
    out["prev_female_only_win_prev_dirt_current_dirt"] = (
        out["is_female"]
        & out["prev_female_only_race"]
        & out["prev_win"]
        & out["prev_dirt"]
        & out["current_dirt"]
    ).astype(int)
    out["prev_female_only_win_class_up_current_dirt"] = (
        out["prev_female_only_win_current_dirt"].eq(1) & out["rotation_class_up_num"].ge(0.5)
    ).astype(int)
    out["prev_female_only_win_to_mixed_dirt"] = (
        out["prev_female_only_win_current_dirt"].eq(1) & out["current_mixed_sex_race"]
    ).astype(int)
    out["prev_female_only_top3_current_dirt"] = (
        out["is_female"] & out["prev_female_only_race"] & out["prev_top3"] & out["current_dirt"]
    ).astype(int)

    venue = text(out, "venue")
    out["summer_female"] = (out["is_female"] & out["is_summer"]).astype(int)
    out["summer_female_dirt"] = (out["summer_female"].eq(1) & out["current_dirt"]).astype(int)
    out["summer_female_turf"] = (out["summer_female"].eq(1) & out["current_turf"]).astype(int)
    out["summer_female_local"] = (out["summer_female"].eq(1) & venue.isin(LOCAL_VENUES)).astype(int)
    race_weight_median = out.groupby("race_id")["carried_weight_num"].transform("median")
    out["runner_weight_advantage_kg"] = (race_weight_median - out["carried_weight_num"]).fillna(0.0)
    out["summer_female_lightweight"] = (
        out["summer_female"].eq(1) & out["runner_weight_advantage_kg"].ge(1.0)
    ).astype(int)
    out["summer_female_large_loss"] = (
        out["summer_female"].eq(1) & out["body_female_large_loss_num"].ge(0.5)
    ).astype(int)
    out["summer_female_young"] = (out["summer_female"].eq(1) & out["age_num"].le(3)).astype(int)

    for col in FACTOR_COLS:
        out[col] = num(out, col, 0).fillna(0).astype(int)
    return out


def runner_metrics(rows: pd.DataFrame, label: str, factor: str) -> dict[str, Any]:
    if rows.empty:
        return {
            "scope": label,
            "factor": factor,
            "starts": 0,
            "races": 0,
            "win_rate_pct": None,
            "top3_rate_pct": None,
            "win_roi_pct": None,
            "place_roi_pct": None,
            "avg_popularity": None,
            "avg_win_odds": None,
        }
    finish = num(rows, "finish_num", 99)
    win_pay = num(rows, "win_pay_num", 0.0).where(finish.eq(1), 0.0)
    place_pay = num(rows, "place_pay_num", 0.0).where(finish.le(3), 0.0)
    stake = len(rows) * 100.0
    return {
        "scope": label,
        "factor": factor,
        "starts": int(len(rows)),
        "races": int(rows["race_id"].nunique()),
        "win_rate_pct": float(finish.eq(1).mean() * 100.0),
        "top3_rate_pct": float(finish.le(3).mean() * 100.0),
        "win_roi_pct": float(win_pay.sum() / stake * 100.0) if stake > 0 else None,
        "place_roi_pct": float(place_pay.sum() / stake * 100.0) if stake > 0 else None,
        "avg_popularity": float(num(rows, "popularity_num", np.nan).mean()),
        "avg_win_odds": float(num(rows, "win_odds_num", np.nan).mean()),
    }


def ticket_stake_return(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    stake_candidates = [
        "eval_stake_yen",
        "scaled_stake_yen",
        "runtime_stake_yen",
        "stake_yen",
        "proxy_stake_yen",
    ]
    return_candidates = [
        "eval_return_yen",
        "scaled_return_yen",
        "runtime_return_yen",
        "return_yen",
        "proxy_return_yen",
    ]
    stake = None
    ret = None
    for col in stake_candidates:
        if col in df.columns:
            stake = num(df, col, 0.0)
            break
    for col in return_candidates:
        if col in df.columns:
            ret = num(df, col, 0.0)
            break
    if stake is None:
        stake = pd.Series(100.0, index=df.index, dtype=float)
    if ret is None:
        ret = pd.Series(0.0, index=df.index, dtype=float)
    return stake.fillna(0.0), ret.fillna(0.0)


def ticket_metrics(rows: pd.DataFrame, label: str, pool_rows: int = 0) -> dict[str, Any]:
    if rows.empty:
        return {
            "label": label,
            "pool_rows": int(pool_rows),
            "tickets": 0,
            "races": 0,
            "days": 0,
            "hits": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": None,
            "hit_rate_pct": None,
        }
    stake, ret = ticket_stake_return(rows)
    hit = ret.gt(0)
    days = rows["race_id"].astype(str).str.slice(0, 8)
    return {
        "label": label,
        "pool_rows": int(pool_rows),
        "tickets": int(len(rows)),
        "races": int(rows["race_id"].nunique()),
        "days": int(days.nunique()),
        "hits": int(hit.sum()),
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(ret.sum() - stake.sum()),
        "roi_pct": float(ret.sum() / stake.sum() * 100.0) if stake.sum() > 0 else None,
        "hit_rate_pct": float(hit.mean() * 100.0),
        "avg_return_per_hit_yen": float(ret[hit].mean()) if hit.any() else 0.0,
        "max_return_yen": float(ret.max()) if len(ret) else 0.0,
        "year_min": int(pd.to_numeric(rows["race_id"].astype(str).str.slice(0, 4), errors="coerce").min()),
        "year_max": int(pd.to_numeric(rows["race_id"].astype(str).str.slice(0, 4), errors="coerce").max()),
    }


def enrich_tickets(tickets: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = clean_race_id(out["race_id"])
    for col in ["anchor_no", "partner_no"]:
        if col not in out.columns:
            raise ValueError(f"Ticket file needs {col}.")
        out[col] = num(out, col).astype("Int64")

    keep = [
        "race_id",
        "horse_no",
        "horse_name",
        "sex",
        "age_num",
        "current_surface_simple",
        "popularity_num",
        "win_odds_num",
        *FACTOR_COLS,
    ]
    lookup = runners[[c for c in keep if c in runners.columns]].drop_duplicates(["race_id", "horse_no"], keep="last")
    anchor = lookup.rename(columns={c: f"anchor_{c}" for c in lookup.columns if c not in {"race_id", "horse_no"}})
    partner = lookup.rename(columns={c: f"partner_{c}" for c in lookup.columns if c not in {"race_id", "horse_no"}})
    out = out.merge(anchor, left_on=["race_id", "anchor_no"], right_on=["race_id", "horse_no"], how="left")
    out = out.drop(columns=["horse_no"], errors="ignore")
    out = out.merge(partner, left_on=["race_id", "partner_no"], right_on=["race_id", "horse_no"], how="left")
    out = out.drop(columns=["horse_no"], errors="ignore")
    for factor in FACTOR_COLS:
        a = num(out, f"anchor_{factor}", 0.0).fillna(0).astype(int)
        p = num(out, f"partner_{factor}", 0.0).fillna(0).astype(int)
        out[f"ticket_any_{factor}"] = (a.eq(1) | p.eq(1)).astype(int)
        out[f"ticket_both_{factor}"] = (a.eq(1) & p.eq(1)).astype(int)
    return out


def make_ticket_segment_summary(tickets: pd.DataFrame, source_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"source": source_name, "factor": "baseline", "segment": "all", **ticket_metrics(tickets, "all", len(tickets))})
    for factor in FACTOR_COLS:
        any_col = f"ticket_any_{factor}"
        flagged = tickets[tickets[any_col].eq(1)].copy()
        unflagged = tickets[tickets[any_col].ne(1)].copy()
        rows.append({"source": source_name, "factor": factor, "segment": "has_factor", **ticket_metrics(flagged, f"has_{factor}", len(tickets))})
        rows.append({"source": source_name, "factor": factor, "segment": "without_factor", **ticket_metrics(unflagged, f"without_{factor}", len(tickets))})
        if len(flagged):
            for year, group in flagged.groupby(flagged["race_id"].astype(str).str.slice(0, 4), dropna=False):
                rows.append(
                    {
                        "source": source_name,
                        "factor": factor,
                        "segment": f"has_factor_year_{year}",
                        **ticket_metrics(group, f"has_{factor}_{year}", len(flagged)),
                    }
                )
    return pd.DataFrame(rows)


def make_policy_summary(tickets: pd.DataFrame, source_name: str) -> pd.DataFrame:
    policies: dict[str, pd.Series] = {"baseline_all": pd.Series(True, index=tickets.index)}
    for factor in FACTOR_COLS:
        policies[f"require_{factor}"] = tickets[f"ticket_any_{factor}"].eq(1)
        policies[f"exclude_{factor}"] = tickets[f"ticket_any_{factor}"].ne(1)

    policies["require_summer_female_without_large_loss"] = (
        tickets["ticket_any_summer_female"].eq(1) & tickets["ticket_any_summer_female_large_loss"].ne(1)
    )
    policies["exclude_summer_female_large_loss"] = tickets["ticket_any_summer_female_large_loss"].ne(1)
    policies["require_dirt_female_limited_win_angle"] = (
        tickets["ticket_any_prev_female_only_win_current_dirt"].eq(1)
        | tickets["ticket_any_prev_female_only_win_to_mixed_dirt"].eq(1)
    )
    policies["exclude_dirt_female_limited_win_angle"] = (
        tickets["ticket_any_prev_female_only_win_current_dirt"].ne(1)
        & tickets["ticket_any_prev_female_only_win_to_mixed_dirt"].ne(1)
    )

    rows = []
    for name, mask in policies.items():
        rows.append({"source": source_name, "policy": name, **ticket_metrics(tickets[mask].copy(), name, int(mask.sum()))})
    return pd.DataFrame(rows)


def make_policy_stress_summary(tickets: pd.DataFrame, source_name: str) -> pd.DataFrame:
    policies = {
        "baseline_all": pd.Series(True, index=tickets.index),
        "require_summer_female": tickets["ticket_any_summer_female"].eq(1),
        "require_summer_female_turf": tickets["ticket_any_summer_female_turf"].eq(1),
        "require_summer_female_lightweight": tickets["ticket_any_summer_female_lightweight"].eq(1),
        "exclude_summer_female_dirt": tickets["ticket_any_summer_female_dirt"].ne(1),
        "exclude_prev_female_only_win_current_dirt": tickets[
            "ticket_any_prev_female_only_win_current_dirt"
        ].ne(1),
    }
    rows: list[dict[str, Any]] = []
    for name, mask in policies.items():
        base = tickets[mask].copy()
        for remove_top_n in [0, 1, 3]:
            work = base.copy()
            stake, ret = ticket_stake_return(work)
            work["_stress_stake"] = stake
            work["_stress_return"] = ret
            if remove_top_n:
                top_idx = work["_stress_return"].sort_values(ascending=False).index[:remove_top_n]
                work.loc[top_idx, "_stress_return"] = 0.0
            s = float(work["_stress_stake"].sum())
            r = float(work["_stress_return"].sum())
            rows.append(
                {
                    "source": source_name,
                    "policy": name,
                    "remove_top_returns": remove_top_n,
                    "tickets": int(len(work)),
                    "races": int(work["race_id"].nunique()) if "race_id" in work.columns else 0,
                    "hits": int(work["_stress_return"].gt(0).sum()),
                    "stake_yen": s,
                    "return_yen": r,
                    "profit_yen": r - s,
                    "roi_pct": (r / s * 100.0) if s > 0 else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument("--runner-eval-features", nargs="*", default=DEFAULT_RUNNER_EVAL_FEATURES)
    parser.add_argument("--tickets", nargs="*", default=DEFAULT_TICKETS)
    parser.add_argument("--output-dir", default="outputs/analysis/female_dirt_summer_roi_v1")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    all_runners = load_runner_features(args.features)
    eval_runners = load_runner_features(args.runner_eval_features)

    runner_rows: list[dict[str, Any]] = []
    runner_rows.append(runner_metrics(eval_runners, "runner_test_all", "baseline_all"))
    for factor in FACTOR_COLS:
        runner_rows.append(runner_metrics(eval_runners[eval_runners[factor].eq(1)], "runner_test_has_factor", factor))
        runner_rows.append(runner_metrics(eval_runners[eval_runners[factor].ne(1)], "runner_test_without_factor", factor))
    runner_summary = pd.DataFrame(runner_rows)
    runner_summary.to_csv(out_dir / "runner_factor_summary.csv", index=False, encoding="utf-8-sig")

    ticket_segment_frames: list[pd.DataFrame] = []
    ticket_policy_frames: list[pd.DataFrame] = []
    ticket_stress_frames: list[pd.DataFrame] = []
    enriched_paths: dict[str, str] = {}
    for raw in args.tickets:
        path = project_path(raw)
        if not path.exists():
            continue
        tickets = read_csv(path)
        source_name = path.stem
        enriched = enrich_tickets(tickets, all_runners)
        enriched_path = out_dir / f"{source_name}_enriched.csv"
        enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
        enriched_paths[source_name] = str(enriched_path)
        ticket_segment_frames.append(make_ticket_segment_summary(enriched, source_name))
        ticket_policy_frames.append(make_policy_summary(enriched, source_name))
        ticket_stress_frames.append(make_policy_stress_summary(enriched, source_name))

    ticket_segments = pd.concat(ticket_segment_frames, ignore_index=True) if ticket_segment_frames else pd.DataFrame()
    ticket_policies = pd.concat(ticket_policy_frames, ignore_index=True) if ticket_policy_frames else pd.DataFrame()
    ticket_stress = pd.concat(ticket_stress_frames, ignore_index=True) if ticket_stress_frames else pd.DataFrame()
    ticket_segments.to_csv(out_dir / "ticket_factor_segments.csv", index=False, encoding="utf-8-sig")
    ticket_policies.to_csv(out_dir / "ticket_policy_summary.csv", index=False, encoding="utf-8-sig")
    ticket_stress.to_csv(out_dir / "ticket_policy_stress_summary.csv", index=False, encoding="utf-8-sig")

    def _top_policy(frame: pd.DataFrame, source: str) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        use = frame[(frame["source"].eq(source)) & (pd.to_numeric(frame["tickets"], errors="coerce").fillna(0).ge(20))].copy()
        if use.empty:
            return []
        use["roi_pct_num"] = pd.to_numeric(use["roi_pct"], errors="coerce")
        return use.sort_values(["roi_pct_num", "tickets"], ascending=[False, False]).head(12).drop(columns=["roi_pct_num"]).to_dict("records")

    summary = {
        "output_dir": str(out_dir),
        "features": args.features,
        "runner_eval_features": args.runner_eval_features,
        "tickets": [str(project_path(p)) for p in args.tickets if project_path(p).exists()],
        "runner_eval_rows": int(len(eval_runners)),
        "runner_eval_races": int(eval_runners["race_id"].nunique()),
        "runner_eval_year_min": int(eval_runners["year"].min()),
        "runner_eval_year_max": int(eval_runners["year"].max()),
        "factor_columns": FACTOR_COLS,
        "enriched_ticket_files": enriched_paths,
        "runner_factor_top_by_win_roi": runner_summary[
            (runner_summary["scope"].eq("runner_test_has_factor"))
            & (pd.to_numeric(runner_summary["starts"], errors="coerce").fillna(0).ge(30))
        ]
        .sort_values("win_roi_pct", ascending=False)
        .head(12)
        .to_dict("records"),
        "runner_factor_top_by_place_roi": runner_summary[
            (runner_summary["scope"].eq("runner_test_has_factor"))
            & (pd.to_numeric(runner_summary["starts"], errors="coerce").fillna(0).ge(30))
        ]
        .sort_values("place_roi_pct", ascending=False)
        .head(12)
        .to_dict("records"),
        "ticket_policy_top_recommended_runtime": _top_policy(ticket_policies, "recommended_runtime_tickets"),
        "ticket_policy_top_recommended_all": _top_policy(ticket_policies, "recommended_all_tickets"),
        "ticket_policy_stress_recommended_runtime": ticket_stress[
            ticket_stress["source"].eq("recommended_runtime_tickets")
        ].to_dict("records")
        if not ticket_stress.empty
        else [],
        "notes": [
            "This is a diagnostic overlay only. It does not change Champion tickets or dashboard logic.",
            "Previous female-only status is reconstructed from prior race entries and race names, then merged by prior race ID and horse ID.",
            "Ticket ROI uses the existing evaluation stake/return columns when present.",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
