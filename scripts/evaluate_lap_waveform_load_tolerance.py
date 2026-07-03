from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TICKETS = ROOT / "outputs/analysis/lap_positive_expansion_v1/lap_positive_expansion_selected_tickets.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/lap_waveform_load_tolerance_v1"
DEFAULT_RUNNER_FEATURES = [
    ROOT / "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features.csv",
    ROOT / "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features.csv",
]

COL_RACE_ID = "レースID(新/馬番無)"
COL_HORSE_NO = "馬番"
COL_HORSE_NAME = "馬名"

LAP_AXIS = ["fast", "slow", "instant", "sustain", "long_spurt"]
HORSE_LAP_COLS = {
    "fast": "horse_fast_lap_score_past5",
    "slow": "horse_slow_lap_score_past5",
    "instant": "horse_instant_lap_score_past5",
    "sustain": "horse_sustain_lap_score_past5",
    "long_spurt": "horse_long_spurt_lap_score_past5",
}


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp932", low_memory=False, **kwargs)


def ncol(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = df[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def first_existing(df: pd.DataFrame, cols: list[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in cols:
        if col in df.columns:
            out = out.fillna(ncol(df, col))
    return out.fillna(default)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def clip01(series: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return pd.Series(series).fillna(0.0).clip(0.0, 1.0)


def available_usecols(path: Path, wanted: list[str]) -> list[str]:
    header = read_csv(path, nrows=0)
    return [c for c in wanted if c in header.columns]


def load_runner_features(paths: list[Path]) -> pd.DataFrame:
    wanted = [
        COL_RACE_ID,
        COL_HORSE_NO,
        COL_HORSE_NAME,
        "target_score",
        "front_running_tendency",
        "closing_tendency",
        "horse_front_run_rate_past5",
        "horse_closer_rate_past5",
        "race_early_pressure_score",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "pace_fit_score",
        "front_advantage_score",
        "closer_advantage_score",
        "positioning_advantage_score",
        "front_pressure_rank_score",
        "solo_lead_potential",
        "lap_pace_versatility_score",
        "lap_aptitude_fit_score",
        "lap_aptitude_reliability_score",
        *HORSE_LAP_COLS.values(),
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        usecols = available_usecols(path, wanted)
        if COL_RACE_ID not in usecols or COL_HORSE_NO not in usecols:
            continue
        frame = read_csv(path, usecols=usecols)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("runner feature files were not found")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df["race_id"] = normalize_id(df[COL_RACE_ID])
    df["horse_no"] = pd.to_numeric(df[COL_HORSE_NO], errors="coerce").astype("Int64")
    for col in wanted:
        if col in {COL_RACE_ID, COL_HORSE_NO, COL_HORSE_NAME}:
            continue
        if col in df.columns:
            df[col] = ncol(df, col)
    keep = ["race_id", "horse_no", COL_HORSE_NAME] + [c for c in wanted if c not in {COL_RACE_ID, COL_HORSE_NO, COL_HORSE_NAME}]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["race_id", "horse_no"]).drop_duplicates(["race_id", "horse_no"], keep="last")
    return df


def add_side_features(tickets: pd.DataFrame, runner: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = normalize_id(out["race_id"])
    for side, no_col in [("anchor", "anchor_no"), ("partner", "partner_no")]:
        if no_col not in out.columns:
            fallback = "horse_a" if side == "anchor" else "horse_b"
            out[no_col] = ncol(out, fallback)
        out[no_col] = pd.to_numeric(out[no_col], errors="coerce").astype("Int64")
        rename_map: dict[str, str] = {"horse_no": no_col}
        if f"{side}_runner_name" not in out.columns:
            rename_map[COL_HORSE_NAME] = f"{side}_runner_name"
        for col in runner.columns:
            if col in {"race_id", "horse_no", COL_HORSE_NAME}:
                continue
            prefixed = f"{side}_{col}"
            if prefixed not in out.columns:
                rename_map[col] = prefixed
        side_runner = runner.rename(
            columns=rename_map
        )
        keep_cols = ["race_id", *[c for c in rename_map.values() if c != "race_id"]]
        keep_cols = list(dict.fromkeys([c for c in keep_cols if c in side_runner.columns]))
        side_runner = side_runner[keep_cols]
        out = out.merge(side_runner, on=["race_id", no_col], how="left")
    return out


def lap_mode_vector(df: pd.DataFrame) -> pd.DataFrame:
    vector = pd.DataFrame(index=df.index)

    vector["fast"] = first_existing(df, ["v2_prob_fast", "shape_fast_signal"], 0.0)
    vector["slow"] = first_existing(df, ["v2_prob_slow", "shape_slow_signal"], 0.0)
    vector["instant"] = first_existing(df, ["v2_prob_instant", "shape_instant_signal"], 0.0)
    vector["sustain"] = first_existing(df, ["v2_prob_sustain", "shape_sustain_signal"], 0.0)
    vector["long_spurt"] = 0.45 * vector["sustain"].fillna(0.0)

    pressure = first_existing(
        df,
        [
            "race_early_pressure_score",
            "anchor_race_early_pressure_score",
            "partner_race_early_pressure_score",
            "race_front_pressure",
            "race_front_count",
        ],
        0.0,
    ).fillna(0.0)
    collapse = first_existing(
        df,
        [
            "race_pace_collapse",
            "race_pace_collapse_risk",
            "anchor_race_pace_collapse_risk",
            "partner_race_pace_collapse_risk",
            "collapse_fit",
            "queue_front_load_score",
        ],
        0.0,
    ).fillna(0.0)
    slow = first_existing(
        df,
        ["race_slow_risk", "race_slow_pace_risk", "anchor_race_slow_pace_risk", "partner_race_slow_pace_risk"],
        0.0,
    ).fillna(0.0)

    pressure01 = (pressure / max(float(pressure.quantile(0.95)) if pressure.notna().any() else 1.0, 1.0)).clip(0.0, 1.0)
    collapse01 = (collapse / max(float(collapse.quantile(0.95)) if collapse.notna().any() else 1.0, 1.0)).clip(0.0, 1.0)
    slow01 = (slow / max(float(slow.quantile(0.95)) if slow.notna().any() else 1.0, 1.0)).clip(0.0, 1.0)

    vector["fast"] = vector["fast"].fillna(0.0) + 0.45 * pressure01 + 0.25 * collapse01
    vector["slow"] = vector["slow"].fillna(0.0) + 0.70 * slow01
    vector["instant"] = vector["instant"].fillna(0.0) + 0.25 * slow01
    vector["sustain"] = vector["sustain"].fillna(0.0) + 0.45 * collapse01 + 0.25 * pressure01
    vector["long_spurt"] = vector["long_spurt"].fillna(0.0) + 0.25 * collapse01

    for mode_col in ["v2_predicted_lap_mode", "cont_predicted_lap_mode", "expected_pace"]:
        if mode_col not in df.columns:
            continue
        mode = df[mode_col].astype(str).str.lower()
        vector.loc[mode.eq("fast"), "fast"] += 0.8
        vector.loc[mode.eq("slow"), "slow"] += 0.8
        vector.loc[mode.eq("instant"), "instant"] += 0.8
        vector.loc[mode.eq("sustain"), "sustain"] += 0.8
        vector.loc[mode.isin(["long_spurt", "longspurt"]), "long_spurt"] += 0.8

    return normalize_rows(vector[LAP_AXIS])


def normalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    row_sum = values.sum(axis=1)
    neutral = row_sum.le(0)
    values.loc[neutral, :] = 1.0
    row_sum = values.sum(axis=1).replace(0.0, np.nan)
    return values.div(row_sum, axis=0).fillna(1.0 / len(values.columns))


def cosine_similarity(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    av = a[LAP_AXIS].to_numpy(dtype=float)
    bv = b[LAP_AXIS].to_numpy(dtype=float)
    denom = np.linalg.norm(av, axis=1) * np.linalg.norm(bv, axis=1)
    dot = (av * bv).sum(axis=1)
    out = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    return pd.Series(out, index=a.index).clip(0.0, 1.0)


def side_lap_vector(df: pd.DataFrame, side: str) -> pd.DataFrame:
    values = pd.DataFrame(index=df.index)
    for axis, base_col in HORSE_LAP_COLS.items():
        values[axis] = ncol(df, f"{side}_{base_col}", 0.0)
    fit = ncol(df, f"{side}_lap_aptitude_fit_score", np.nan)
    reliability = ncol(df, f"{side}_lap_aptitude_reliability_score", np.nan)
    if fit.notna().any():
        values = values.mul(0.80 + 0.20 * fit.fillna(fit.median()).clip(0.0, 1.0), axis=0)
    if reliability.notna().any():
        values = values.mul(0.80 + 0.20 * reliability.fillna(reliability.median()).clip(0.0, 1.0), axis=0)
    return normalize_rows(values[LAP_AXIS])


def add_waveform_and_load_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    race_vec = lap_mode_vector(out)
    anchor_vec = side_lap_vector(out, "anchor")
    partner_vec = side_lap_vector(out, "partner")

    out["anchor_waveform_similarity"] = cosine_similarity(race_vec, anchor_vec)
    out["partner_waveform_similarity"] = cosine_similarity(race_vec, partner_vec)
    out["pair_waveform_min_similarity"] = np.minimum(
        out["anchor_waveform_similarity"], out["partner_waveform_similarity"]
    )
    out["pair_waveform_avg_similarity"] = (
        out["anchor_waveform_similarity"] + out["partner_waveform_similarity"]
    ) / 2.0
    out["pair_waveform_gap_similarity"] = (
        out["anchor_waveform_similarity"] - out["partner_waveform_similarity"]
    ).abs()

    existing_fit = first_existing(
        out,
        [
            "pair_lap_numeric_shadow_score",
            "pair_lap_numeric_fit_min",
            "pair_lap_profile_fit_min",
            "pair_min_lap_profile_fit_score",
            "lap_positive_score",
        ],
        np.nan,
    )
    existing_fit = existing_fit.fillna(out["pair_waveform_min_similarity"]).clip(0.0, 1.0)
    out["pair_waveform_combo_score"] = (
        0.52 * out["pair_waveform_min_similarity"]
        + 0.28 * out["pair_waveform_avg_similarity"]
        + 0.20 * existing_fit
    ).clip(0.0, 1.0)

    pressure = first_existing(
        out,
        [
            "race_early_pressure_score",
            "anchor_race_early_pressure_score",
            "partner_race_early_pressure_score",
            "race_front_pressure",
            "race_front_count",
        ],
        0.0,
    ).fillna(0.0)
    collapse = first_existing(
        out,
        [
            "race_pace_collapse",
            "race_pace_collapse_risk",
            "anchor_race_pace_collapse_risk",
            "partner_race_pace_collapse_risk",
            "collapse_fit",
            "queue_front_load_score",
        ],
        0.0,
    ).fillna(0.0)
    front_load = first_existing(
        out,
        ["queue_front_load_score", "same_day_projected_front_load_score", "race_front_count"],
        0.0,
    ).fillna(0.0)
    pressure01 = (pressure / max(float(pressure.quantile(0.95)) if pressure.notna().any() else 1.0, 1.0)).clip(0.0, 1.0)
    collapse01 = (collapse / max(float(collapse.quantile(0.95)) if collapse.notna().any() else 1.0, 1.0)).clip(0.0, 1.0)
    front_load01 = (front_load / max(float(front_load.quantile(0.95)) if front_load.notna().any() else 1.0, 1.0)).clip(0.0, 1.0)
    out["race_lap_load_score"] = (0.38 * pressure01 + 0.42 * collapse01 + 0.20 * front_load01).clip(0.0, 1.0)

    for side in ["anchor", "partner"]:
        front = first_existing(
            out,
            [
                f"{side}_projected_front5_prob",
                f"{side}_front5_model_prob",
                f"{side}_horse_front_run_rate_past5_feature",
                f"{side}_horse_front_run_rate_past5",
                "projected_front5_prob" if side == "anchor" else "__missing__",
            ],
            0.0,
        ).fillna(0.0)
        closer = first_existing(
            out,
            [
                f"{side}_closer_tendency_v2",
                f"{side}_horse_closer_rate_past5_feature",
                f"{side}_horse_closer_rate_past5",
                f"{side}_closing_tendency",
                f"{side}_closer_advantage_score",
            ],
            0.0,
        ).fillna(0.0)
        sustain = ncol(out, f"{side}_horse_sustain_lap_score_past5", 0.0).fillna(0.0).clip(0.0, 1.0)
        long_spurt = ncol(out, f"{side}_horse_long_spurt_lap_score_past5", 0.0).fillna(0.0).clip(0.0, 1.0)
        fast = ncol(out, f"{side}_horse_fast_lap_score_past5", 0.0).fillna(0.0).clip(0.0, 1.0)
        instant = ncol(out, f"{side}_horse_instant_lap_score_past5", 0.0).fillna(0.0).clip(0.0, 1.0)
        versatility = ncol(out, f"{side}_lap_pace_versatility_score", 0.0).fillna(0.0).clip(0.0, 1.0)
        reliability = ncol(out, f"{side}_lap_aptitude_reliability_score", 0.0).fillna(0.0).clip(0.0, 1.0)
        out[f"{side}_front_role_score"] = front.clip(0.0, 1.0)
        out[f"{side}_closer_role_score"] = closer.clip(0.0, 1.0)
        out[f"{side}_lap_load_tolerance_score"] = (
            0.30 * sustain + 0.25 * long_spurt + 0.18 * fast + 0.15 * versatility + 0.12 * reliability
        ).clip(0.0, 1.0)
        out[f"{side}_collapse_receiver_score"] = (
            0.32 * closer.clip(0.0, 1.0) + 0.25 * long_spurt + 0.23 * instant + 0.20 * sustain
        ).clip(0.0, 1.0)

    anchor_survive = out["anchor_front_role_score"] * out["anchor_lap_load_tolerance_score"]
    partner_survive = out["partner_front_role_score"] * out["partner_lap_load_tolerance_score"]
    out["pair_front_load_survival_score"] = out["race_lap_load_score"] * np.maximum(anchor_survive, partner_survive)

    role_a = anchor_survive * out["partner_collapse_receiver_score"]
    role_b = partner_survive * out["anchor_collapse_receiver_score"]
    out["pair_load_role_balance_score"] = out["race_lap_load_score"] * np.maximum(role_a, role_b)

    anchor_risk = out["anchor_front_role_score"] * (1.0 - out["anchor_lap_load_tolerance_score"])
    partner_risk = out["partner_front_role_score"] * (1.0 - out["partner_lap_load_tolerance_score"])
    out["pair_load_collapse_risk_score"] = out["race_lap_load_score"] * np.maximum(anchor_risk, partner_risk)

    out["pair_wave_load_combo_score"] = (
        0.45 * out["pair_waveform_combo_score"]
        + 0.35 * out["pair_load_role_balance_score"]
        + 0.20 * (1.0 - out["pair_load_collapse_risk_score"])
    ).clip(0.0, 1.0)
    return out


def normalize_ticket_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = normalize_id(out["race_id"])
    if "ticket_type" not in out.columns:
        out["ticket_type"] = "unknown"
    out["ticket_type"] = out["ticket_type"].astype(str).str.lower()

    if "stake_yen" in out.columns:
        out["stake_yen_eval"] = ncol(out, "stake_yen", 100.0).fillna(100.0)
    elif "runtime_stake_yen" in out.columns:
        out["stake_yen_eval"] = ncol(out, "runtime_stake_yen", 100.0).fillna(100.0)
    else:
        out["stake_yen_eval"] = 100.0

    if "return_yen" in out.columns:
        out["return_yen_eval"] = ncol(out, "return_yen", 0.0).fillna(0.0)
    elif "runtime_return_yen" in out.columns:
        out["return_yen_eval"] = ncol(out, "runtime_return_yen", 0.0).fillna(0.0)
    else:
        ret = pd.Series(0.0, index=out.index)
        if {"wide_hit", "wide_return_100"}.issubset(out.columns):
            ret = ret.mask(out["ticket_type"].eq("wide"), ncol(out, "wide_hit", 0).fillna(0) * ncol(out, "wide_return_100", 0).fillna(0))
        if {"umaren_hit", "umaren_return_100"}.issubset(out.columns):
            ret = ret.mask(out["ticket_type"].eq("umaren"), ncol(out, "umaren_hit", 0).fillna(0) * ncol(out, "umaren_return_100", 0).fillna(0))
        out["return_yen_eval"] = ret

    if "hit" in out.columns:
        out["hit_eval"] = ncol(out, "hit", 0).fillna(0).gt(0)
    else:
        hit = pd.Series(False, index=out.index)
        if "wide_hit" in out.columns:
            hit = hit.mask(out["ticket_type"].eq("wide"), ncol(out, "wide_hit", 0).fillna(0).gt(0))
        if "umaren_hit" in out.columns:
            hit = hit.mask(out["ticket_type"].eq("umaren"), ncol(out, "umaren_hit", 0).fillna(0).gt(0))
        out["hit_eval"] = hit

    if "year" not in out.columns:
        out["year"] = out["race_id"].str.slice(0, 4)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    return out


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = pd.to_numeric(profit, errors="coerce").fillna(0.0).cumsum()
    drawdown = curve - curve.cummax()
    return float(drawdown.min())


def roi_without_top(frame: pd.DataFrame, n: int) -> float:
    if frame.empty:
        return 0.0
    kept = frame.sort_values("return_yen_eval", ascending=False).iloc[n:]
    stake = float(kept["stake_yen_eval"].sum())
    ret = float(kept["return_yen_eval"].sum())
    return ret / stake * 100.0 if stake > 0 else 0.0


def metrics(frame: pd.DataFrame, segment: str, base_roi: float | None = None) -> dict[str, Any]:
    if frame.empty:
        return {
            "segment": segment,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": 0.0,
            "roi_lift_vs_base_pt": np.nan,
            "hit_rate_pct": 0.0,
            "race_hit_rate_pct": 0.0,
            "max_drawdown_yen": 0.0,
            "top1_removed_roi_pct": 0.0,
            "top3_removed_roi_pct": 0.0,
            "top5_removed_roi_pct": 0.0,
            "min_year_roi_pct": np.nan,
            "avg_wave_combo": np.nan,
            "avg_load_balance": np.nan,
            "avg_load_risk": np.nan,
        }
    stake = pd.to_numeric(frame["stake_yen_eval"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(frame["return_yen_eval"], errors="coerce").fillna(0.0)
    profit = ret - stake
    roi = float(ret.sum() / stake.sum() * 100.0) if float(stake.sum()) > 0 else 0.0
    by_year = frame.groupby("year", dropna=True).agg(stake=("stake_yen_eval", "sum"), ret=("return_yen_eval", "sum"))
    year_roi = (by_year["ret"] / by_year["stake"] * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
    race_hit = frame.groupby("race_id")["hit_eval"].max().mean() if "race_id" in frame.columns else np.nan
    ordered = frame.assign(_profit=profit).sort_values(["race_id", "ticket_type", "anchor_no", "partner_no"], kind="mergesort")
    return {
        "segment": segment,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0,
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(profit.sum()),
        "roi_pct": roi,
        "roi_lift_vs_base_pt": roi - base_roi if base_roi is not None else np.nan,
        "hit_rate_pct": float(frame["hit_eval"].mean() * 100.0),
        "race_hit_rate_pct": float(race_hit * 100.0) if pd.notna(race_hit) else np.nan,
        "max_drawdown_yen": max_drawdown(ordered["_profit"]),
        "top1_removed_roi_pct": roi_without_top(frame, 1),
        "top3_removed_roi_pct": roi_without_top(frame, 3),
        "top5_removed_roi_pct": roi_without_top(frame, 5),
        "min_year_roi_pct": float(year_roi.min()) if not year_roi.empty else np.nan,
        "avg_wave_combo": float(ncol(frame, "pair_waveform_combo_score", np.nan).mean()),
        "avg_load_balance": float(ncol(frame, "pair_load_role_balance_score", np.nan).mean()),
        "avg_load_risk": float(ncol(frame, "pair_load_collapse_risk_score", np.nan).mean()),
    }


def add_segment_metrics(df: pd.DataFrame, min_tickets: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []

    scopes: list[tuple[str, pd.DataFrame]] = [("all", df)]
    for ticket_type, sub in df.groupby("ticket_type", dropna=False):
        scopes.append((str(ticket_type), sub))

    score_cols = [
        "pair_waveform_min_similarity",
        "pair_waveform_avg_similarity",
        "pair_waveform_combo_score",
        "pair_front_load_survival_score",
        "pair_load_role_balance_score",
        "pair_wave_load_combo_score",
    ]
    risk_cols = ["pair_load_collapse_risk_score", "pair_waveform_gap_similarity"]

    for scope_name, frame in scopes:
        if frame.empty:
            continue
        base = metrics(frame, f"{scope_name}:base")
        base_roi = float(base["roi_pct"])
        base_rows.append(base)

        for col in score_cols:
            values = pd.to_numeric(frame[col], errors="coerce")
            for q in [0.50, 0.60, 0.70, 0.80, 0.90]:
                threshold = float(values.quantile(q)) if values.notna().any() else np.nan
                if not np.isfinite(threshold):
                    continue
                seg = frame[values.ge(threshold)]
                if len(seg) >= min_tickets:
                    rows.append(metrics(seg, f"{scope_name}:{col}_ge_q{int(q * 100)}({threshold:.3f})", base_roi))

        for col in risk_cols:
            values = pd.to_numeric(frame[col], errors="coerce")
            for q in [0.20, 0.30, 0.40]:
                threshold = float(values.quantile(q)) if values.notna().any() else np.nan
                if not np.isfinite(threshold):
                    continue
                seg = frame[values.le(threshold)]
                if len(seg) >= min_tickets:
                    rows.append(metrics(seg, f"{scope_name}:{col}_le_q{int(q * 100)}({threshold:.3f})", base_roi))

        combos = {
            "wave_combo_q70_and_load_balance_q60": (
                ncol(frame, "pair_waveform_combo_score").ge(ncol(frame, "pair_waveform_combo_score").quantile(0.70))
                & ncol(frame, "pair_load_role_balance_score").ge(ncol(frame, "pair_load_role_balance_score").quantile(0.60))
            ),
            "wave_combo_q70_and_low_load_risk_q40": (
                ncol(frame, "pair_waveform_combo_score").ge(ncol(frame, "pair_waveform_combo_score").quantile(0.70))
                & ncol(frame, "pair_load_collapse_risk_score").le(ncol(frame, "pair_load_collapse_risk_score").quantile(0.40))
            ),
            "wave_load_combo_q80_and_low_gap_q40": (
                ncol(frame, "pair_wave_load_combo_score").ge(ncol(frame, "pair_wave_load_combo_score").quantile(0.80))
                & ncol(frame, "pair_waveform_gap_similarity").le(ncol(frame, "pair_waveform_gap_similarity").quantile(0.40))
            ),
            "front_load_survival_q80_low_risk_q40": (
                ncol(frame, "pair_front_load_survival_score").ge(ncol(frame, "pair_front_load_survival_score").quantile(0.80))
                & ncol(frame, "pair_load_collapse_risk_score").le(ncol(frame, "pair_load_collapse_risk_score").quantile(0.40))
            ),
        }
        for label, mask in combos.items():
            seg = frame[mask.fillna(False)]
            if len(seg) >= min_tickets:
                rows.append(metrics(seg, f"{scope_name}:{label}", base_roi))

    base_df = pd.DataFrame(base_rows)
    seg_df = pd.DataFrame(rows)
    if not seg_df.empty:
        seg_df = seg_df.sort_values(
            ["roi_pct", "top3_removed_roi_pct", "tickets"],
            ascending=[False, False, False],
            kind="mergesort",
        )
    return base_df, seg_df


def compact_enriched(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "race_id",
        "year",
        "ticket_type",
        "anchor_no",
        "anchor_name",
        "anchor_pop",
        "anchor_odds",
        "partner_no",
        "partner_name",
        "partner_pop",
        "partner_odds",
        "stake_yen_eval",
        "return_yen_eval",
        "hit_eval",
        "pair_waveform_min_similarity",
        "pair_waveform_avg_similarity",
        "pair_waveform_gap_similarity",
        "pair_waveform_combo_score",
        "race_lap_load_score",
        "pair_front_load_survival_score",
        "pair_load_role_balance_score",
        "pair_load_collapse_risk_score",
        "pair_wave_load_combo_score",
        "anchor_waveform_similarity",
        "partner_waveform_similarity",
        "anchor_front_role_score",
        "partner_front_role_score",
        "anchor_lap_load_tolerance_score",
        "partner_lap_load_tolerance_score",
        "anchor_collapse_receiver_score",
        "partner_collapse_receiver_score",
        "v2_predicted_lap_mode",
        "expected_pace",
        "policy",
        "operation_profile",
        "runtime_action",
    ]
    return df[[c for c in wanted if c in df.columns]].copy()


def run(args: argparse.Namespace) -> dict[str, Any]:
    tickets = read_csv(args.tickets)
    tickets = normalize_ticket_frame(tickets)
    runner = load_runner_features([Path(p) for p in args.runner_features])
    joined = add_side_features(tickets, runner)
    enriched = add_waveform_and_load_features(joined)
    base_df, seg_df = add_segment_metrics(enriched, args.min_tickets)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    compact_enriched(enriched).to_csv(out_dir / "lap_waveform_load_enriched_tickets.csv", index=False, encoding="utf-8-sig")
    base_df.to_csv(out_dir / "base_metrics.csv", index=False, encoding="utf-8-sig")
    seg_df.to_csv(out_dir / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    top_segments = seg_df.head(20).to_dict(orient="records") if not seg_df.empty else []
    summary = {
        "tickets": str(args.tickets),
        "runner_features": [str(p) for p in args.runner_features],
        "output_dir": str(out_dir),
        "base": base_df.to_dict(orient="records"),
        "top_segments": top_segments,
        "notes": [
            "waveform_similarity compares the expected race lap-shape vector with each horse's past5 lap aptitude vector.",
            "load_tolerance estimates whether a front/stalking role can survive a high-load race and whether the partner can receive a collapse.",
            "This is an evaluation-only script; it does not change runtime BUY gates.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = [
        "# Lap Waveform / Load Tolerance Evaluation",
        "",
        "未検証だったラップ波形適性と消耗戦耐性を、買い目単位で検証した出力です。",
        "",
        "- `pair_waveform_*`: 今回想定されるラップ型と、2頭の過去5走ラップ適性の噛み合い",
        "- `pair_front_load_survival_score`: 速い流れ・先行負荷でも前目馬が残れるか",
        "- `pair_load_role_balance_score`: 前で耐える馬と、崩れを受ける馬の役割バランス",
        "- `pair_load_collapse_risk_score`: 前目に行くが耐性不足になりやすいリスク",
        "- `pair_wave_load_combo_score`: 波形適性と負荷耐性を合わせた総合シャドースコア",
        "",
        "正式BUY条件は変更していません。採用候補は `segment_metrics.csv` のROI、上位的中除外ROI、年度別最低ROIで判断してください。",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lap waveform similarity and load tolerance as shadow ROI filters.")
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    parser.add_argument("--runner-features", type=Path, nargs="+", default=DEFAULT_RUNNER_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-tickets", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
