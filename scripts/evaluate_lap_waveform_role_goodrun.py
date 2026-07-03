from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TICKETS = ROOT / "outputs/analysis/lap_positive_expansion_v1/lap_positive_expansion_selected_tickets.csv"
DEFAULT_RUNNER_LAP = ROOT / "outputs/analysis/lap_pair_refinement_candidates_v1/runner_lap_pair_refinement_features.csv"
DEFAULT_RUNNER_FEATURES = [
    ROOT / "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features.csv",
    ROOT / "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features.csv",
]
DEFAULT_OUT = ROOT / "outputs/analysis/lap_waveform_role_goodrun_v1"

COL_DATE = "日付"
COL_RACE_ID = "レースID(新/馬番無)"
COL_HORSE_NO = "馬番"
COL_HORSE_ID = "血統登録番号"
COL_HORSE_NAME = "馬名"
COL_FINISH = "確定着順"

AXES = ["fast", "slow", "instant", "sustain", "long_spurt"]


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


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def clip01(series: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return pd.Series(series).fillna(0.0).clip(0.0, 1.0)


def normalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    row_sum = values.sum(axis=1)
    values.loc[row_sum.le(0), :] = 1.0
    row_sum = values.sum(axis=1).replace(0.0, np.nan)
    return values.div(row_sum, axis=0).fillna(1.0 / max(len(values.columns), 1))


def cosine(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    cols = [c for c in a.columns if c in b.columns]
    if not cols:
        cols = list(a.columns)
    av = a[cols].to_numpy(dtype=float)
    bv = b[cols].to_numpy(dtype=float)
    denom = np.linalg.norm(av, axis=1) * np.linalg.norm(bv, axis=1)
    dot = (av * bv).sum(axis=1)
    out = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    return pd.Series(out, index=a.index).clip(0.0, 1.0)


def parse_date(series: pd.Series) -> pd.Series:
    raw = series.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip()
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    yymmdd = raw.str.fullmatch(r"\d{6}")
    if yymmdd.any():
        out.loc[yymmdd] = pd.to_datetime(raw.loc[yymmdd], format="%y%m%d", errors="coerce")
    yyyymmdd = raw.str.fullmatch(r"\d{8}")
    if yyyymmdd.any():
        out.loc[yyyymmdd] = pd.to_datetime(raw.loc[yyyymmdd], format="%Y%m%d", errors="coerce")
    missing = out.isna() & raw.ne("")
    if missing.any():
        out.loc[missing] = pd.to_datetime(raw.loc[missing], errors="coerce")
    return out


def available_usecols(path: Path, wanted: list[str]) -> list[str]:
    header = read_csv(path, nrows=0)
    return [c for c in wanted if c in header.columns]


def normalize_tickets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = normalize_id(out["race_id"])
    if "ticket_type" not in out.columns:
        out["ticket_type"] = "unknown"
    out["ticket_type"] = out["ticket_type"].astype(str).str.lower()
    for col, fallback in [("anchor_no", "horse_a"), ("partner_no", "horse_b")]:
        if col not in out.columns:
            out[col] = ncol(out, fallback)
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

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
        if "wide_hit" in out.columns and "wide_return_100" in out.columns:
            ret = ret.mask(out["ticket_type"].eq("wide"), ncol(out, "wide_hit", 0).fillna(0) * ncol(out, "wide_return_100", 0).fillna(0))
        if "umaren_hit" in out.columns and "umaren_return_100" in out.columns:
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


def load_runner_lap(path: Path) -> pd.DataFrame:
    wanted = [
        "race_id",
        "horse_no",
        "馬名",
        "popularity",
        "odds",
        "finish",
        "is_top3",
        "predicted_lap_mode",
        "actual_lap_mode_diagnostic",
        "race_need_fast",
        "race_need_slow",
        "race_need_instant",
        "race_need_sustain",
        "race_need_long_spurt",
        "horse_lap_fast",
        "horse_lap_slow",
        "horse_lap_instant",
        "horse_lap_sustain",
        "horse_lap_long_spurt",
        "lap_profile_fit_score",
        "lap_fit_confident_score",
        "lap_axis_candidate_score",
        "lap_partner_specialist_score",
        "lap_mismatch_popular_risk",
        "race_lap_prediction_confidence",
        "race_lap_profile_concentration",
        "horse_lap_profile_sharpness",
    ]
    df = read_csv(path, usecols=available_usecols(path, wanted))
    df["race_id"] = normalize_id(df["race_id"])
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    for col in wanted:
        if col not in {"race_id", "horse_no", "馬名", "predicted_lap_mode", "actual_lap_mode_diagnostic"} and col in df.columns:
            df[col] = ncol(df, col)
    return df.drop_duplicates(["race_id", "horse_no"], keep="last")


def add_runner_lap_sides(tickets: pd.DataFrame, runner_lap: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    for side, no_col in [("anchor", "anchor_no"), ("partner", "partner_no")]:
        rename: dict[str, str] = {"horse_no": no_col}
        for col in runner_lap.columns:
            if col in {"race_id", "horse_no"}:
                continue
            pref = f"{side}_{col}"
            if pref not in out.columns:
                rename[col] = pref
        side_frame = runner_lap.rename(columns=rename)
        keep = ["race_id", no_col, *[c for c in rename.values() if c not in {"race_id", no_col}]]
        keep = list(dict.fromkeys([c for c in keep if c in side_frame.columns]))
        out = out.merge(side_frame[keep], on=["race_id", no_col], how="left")
    return out


def build_goodrun_profiles(paths: list[Path]) -> pd.DataFrame:
    wanted = [
        COL_DATE,
        COL_RACE_ID,
        COL_HORSE_NO,
        COL_HORSE_ID,
        COL_HORSE_NAME,
        COL_FINISH,
        "target_score",
        "PCI",
        "PCI3",
        "RPCI",
        "Ave-3F",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        usecols = available_usecols(path, wanted)
        if not {COL_DATE, COL_RACE_ID, COL_HORSE_NO, COL_HORSE_ID}.issubset(usecols):
            continue
        frames.append(read_csv(path, usecols=usecols))
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw["race_id"] = normalize_id(raw[COL_RACE_ID])
    raw["horse_no"] = pd.to_numeric(raw[COL_HORSE_NO], errors="coerce").astype("Int64")
    raw["horse_id"] = normalize_id(raw[COL_HORSE_ID])
    raw["race_date"] = parse_date(raw[COL_DATE])
    for col in [COL_FINISH, "target_score", "PCI", "PCI3", "RPCI", "Ave-3F"]:
        if col in raw.columns:
            raw[col] = ncol(raw, col)
    raw = raw.dropna(subset=["race_id", "horse_no", "horse_id", "race_date"]).copy()
    raw = raw.sort_values(["horse_id", "race_date", "race_id"], kind="mergesort")
    raw = raw.drop_duplicates(["race_id", "horse_no"], keep="last")

    rpci = ncol(raw, "RPCI")
    pci = ncol(raw, "PCI")
    pci3 = ncol(raw, "PCI3")
    score = ncol(raw, "target_score", 0.0).fillna(0.0).clip(0.0, 1.0)
    finish = ncol(raw, COL_FINISH)
    good = score.ge(0.58) | finish.le(3)
    perf_weight = (0.40 + 0.60 * score).where(good, np.nan)

    raw["regime_fast"] = ((50.0 - rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)
    raw["regime_slow"] = ((rpci - 50.0) / 5.0).clip(0.0, 1.0).fillna(0.0)
    raw["regime_instant"] = ((pci - rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)
    raw["regime_sustain"] = (1.0 - (pci - rpci).abs() / 4.0).clip(0.0, 1.0).fillna(0.0)
    raw["regime_long_spurt"] = ((pci3 - rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)

    for axis in AXES:
        raw[f"goodrun_{axis}_value"] = raw[f"regime_{axis}"] * perf_weight
        raw[f"goodrun_{axis}_obs"] = raw[f"goodrun_{axis}_value"].notna().astype(float)

    grouped = raw.groupby("horse_id", sort=False)
    out = raw[["race_id", "horse_no"]].copy()
    count_total = pd.Series(0.0, index=raw.index)
    for axis in AXES:
        val = raw[f"goodrun_{axis}_value"].fillna(0.0)
        obs = raw[f"goodrun_{axis}_obs"].fillna(0.0)
        prior_sum = val.groupby(raw["horse_id"], sort=False).cumsum() - val
        prior_count = obs.groupby(raw["horse_id"], sort=False).cumsum() - obs
        out[f"goodrun_lap_{axis}_profile"] = (prior_sum / prior_count.replace(0.0, np.nan)).fillna(0.0)
        count_total = count_total + prior_count.fillna(0.0)
    out["goodrun_lap_evidence_count"] = count_total / len(AXES)
    out["goodrun_lap_profile_ready"] = out["goodrun_lap_evidence_count"].ge(1).astype(float)
    return out.drop_duplicates(["race_id", "horse_no"], keep="last")


def add_goodrun_sides(tickets: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    if profiles.empty:
        for side in ["anchor", "partner"]:
            for axis in AXES:
                out[f"{side}_goodrun_lap_{axis}_profile"] = 0.0
            out[f"{side}_goodrun_lap_evidence_count"] = 0.0
            out[f"{side}_goodrun_lap_profile_ready"] = 0.0
        return out
    for side, no_col in [("anchor", "anchor_no"), ("partner", "partner_no")]:
        rename = {"horse_no": no_col}
        for col in profiles.columns:
            if col in {"race_id", "horse_no"}:
                continue
            rename[col] = f"{side}_{col}"
        side_profiles = profiles.rename(columns=rename)
        out = out.merge(side_profiles, on=["race_id", no_col], how="left")
    return out


def ticket_race_need_vector(df: pd.DataFrame) -> pd.DataFrame:
    race_need = pd.DataFrame(index=df.index)
    for axis in AXES:
        a = ncol(df, f"anchor_race_need_{axis}", np.nan)
        p = ncol(df, f"partner_race_need_{axis}", np.nan)
        if a.notna().any() or p.notna().any():
            race_need[axis] = pd.concat([a, p], axis=1).mean(axis=1)
        else:
            race_need[axis] = 0.0
    return normalize_rows(race_need[AXES])


def side_vector(df: pd.DataFrame, side: str, prefix: str) -> pd.DataFrame:
    values = pd.DataFrame(index=df.index)
    for axis in AXES:
        values[axis] = ncol(df, f"{side}_{prefix}_{axis}", 0.0)
    return normalize_rows(values[AXES])


def pseudo_200m_wave_transform(vec: pd.DataFrame) -> pd.DataFrame:
    # Six synthetic 200m-ish phases: start load, early settle, mid relax,
    # acceleration, sustained drive, late deceleration. This is a proxy until
    # full race-lap strings are available for every historical race.
    fast = vec["fast"]
    slow = vec["slow"]
    instant = vec["instant"]
    sustain = vec["sustain"]
    long_spurt = vec["long_spurt"]
    phases = pd.DataFrame(index=vec.index)
    phases["p1_start"] = 0.68 * fast + 0.18 * sustain + 0.14 * long_spurt
    phases["p2_early"] = 0.46 * fast + 0.34 * sustain + 0.20 * slow
    phases["p3_mid"] = 0.52 * slow + 0.28 * sustain + 0.20 * instant
    phases["p4_accel"] = 0.58 * instant + 0.24 * sustain + 0.18 * slow
    phases["p5_drive"] = 0.50 * sustain + 0.34 * long_spurt + 0.16 * instant
    phases["p6_late"] = 0.46 * long_spurt + 0.38 * sustain + 0.16 * fast
    return normalize_rows(phases)


def add_lap_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    need = ticket_race_need_vector(out)

    for side in ["anchor", "partner"]:
        horse = side_vector(out, side, "horse_lap")
        goodrun = pd.DataFrame(index=out.index)
        for axis in AXES:
            goodrun[axis] = ncol(out, f"{side}_goodrun_lap_{axis}_profile", 0.0)
        goodrun = normalize_rows(goodrun[AXES])

        out[f"{side}_strict_waveform_similarity"] = cosine(pseudo_200m_wave_transform(need), pseudo_200m_wave_transform(horse))
        out[f"{side}_goodrun_lap_fit_score"] = cosine(need, goodrun)
        ready = ncol(out, f"{side}_goodrun_lap_profile_ready", 0.0).fillna(0.0).clip(0.0, 1.0)
        out[f"{side}_goodrun_lap_fit_score"] = (
            out[f"{side}_goodrun_lap_fit_score"] * ready + 0.50 * (1.0 - ready)
        ).clip(0.0, 1.0)

    out["strict_waveform_pair_min_score"] = np.minimum(
        out["anchor_strict_waveform_similarity"], out["partner_strict_waveform_similarity"]
    )
    out["strict_waveform_pair_avg_score"] = (
        out["anchor_strict_waveform_similarity"] + out["partner_strict_waveform_similarity"]
    ) / 2.0
    out["strict_waveform_pair_gap_score"] = (
        out["anchor_strict_waveform_similarity"] - out["partner_strict_waveform_similarity"]
    ).abs()
    out["goodrun_lap_pair_min_score"] = np.minimum(
        out["anchor_goodrun_lap_fit_score"], out["partner_goodrun_lap_fit_score"]
    )
    out["goodrun_lap_pair_avg_score"] = (
        out["anchor_goodrun_lap_fit_score"] + out["partner_goodrun_lap_fit_score"]
    ) / 2.0
    out["goodrun_lap_pair_ready_score"] = np.minimum(
        ncol(out, "anchor_goodrun_lap_profile_ready", 0.0).fillna(0.0),
        ncol(out, "partner_goodrun_lap_profile_ready", 0.0).fillna(0.0),
    )

    pressure = pd.concat(
        [
            ncol(out, "race_front_pressure", np.nan),
            ncol(out, "race_early_pressure_score", np.nan),
            ncol(out, "anchor_race_early_pressure_score", np.nan),
            ncol(out, "partner_race_early_pressure_score", np.nan),
            ncol(out, "queue_front_load_score", np.nan),
        ],
        axis=1,
    ).mean(axis=1).fillna(0.0)
    collapse = pd.concat(
        [
            ncol(out, "race_pace_collapse", np.nan),
            ncol(out, "race_pace_collapse_risk", np.nan),
            ncol(out, "anchor_race_pace_collapse_risk", np.nan),
            ncol(out, "partner_race_pace_collapse_risk", np.nan),
            ncol(out, "collapse_fit", np.nan),
        ],
        axis=1,
    ).mean(axis=1).fillna(0.0)
    pressure01 = (pressure / max(float(pressure.quantile(0.95)) if pressure.notna().any() else 1.0, 1.0)).clip(0.0, 1.0)
    collapse01 = (collapse / max(float(collapse.quantile(0.95)) if collapse.notna().any() else 1.0, 1.0)).clip(0.0, 1.0)
    out["race_role_load_score"] = (0.55 * pressure01 + 0.45 * collapse01).clip(0.0, 1.0)

    for side in ["anchor", "partner"]:
        front = pd.concat(
            [
                ncol(out, f"{side}_projected_front5_prob", np.nan),
                ncol(out, f"{side}_front5_model_prob", np.nan),
                ncol(out, f"{side}_horse_front_run_rate_past5_feature", np.nan),
                ncol(out, f"{side}_horse_front_run_rate_past5", np.nan),
                ncol(out, f"{side}_front_running_tendency", np.nan),
                ncol(out, "projected_front5_prob" if side == "anchor" else "__missing__", np.nan),
            ],
            axis=1,
        ).mean(axis=1).fillna(0.0).clip(0.0, 1.0)
        closer = pd.concat(
            [
                ncol(out, f"{side}_closer_tendency_v2", np.nan),
                ncol(out, f"{side}_horse_closer_rate_past5_feature", np.nan),
                ncol(out, f"{side}_horse_closer_rate_past5", np.nan),
                ncol(out, f"{side}_closing_tendency", np.nan),
                ncol(out, f"{side}_closer_advantage_score", np.nan),
            ],
            axis=1,
        ).mean(axis=1).fillna(0.0).clip(0.0, 1.0)
        sustain = ncol(out, f"{side}_horse_lap_sustain", 0.0).fillna(0.0).clip(0.0, 1.0)
        long_spurt = ncol(out, f"{side}_horse_lap_long_spurt", 0.0).fillna(0.0).clip(0.0, 1.0)
        instant = ncol(out, f"{side}_horse_lap_instant", 0.0).fillna(0.0).clip(0.0, 1.0)
        fast = ncol(out, f"{side}_horse_lap_fast", 0.0).fillna(0.0).clip(0.0, 1.0)
        out[f"{side}_role_front_score"] = front
        out[f"{side}_role_closer_score"] = closer
        out[f"{side}_role_front_survival_score"] = (front * (0.36 * sustain + 0.30 * long_spurt + 0.20 * fast + 0.14 * out[f"{side}_strict_waveform_similarity"])).clip(0.0, 1.0)
        out[f"{side}_role_collapse_receiver_score"] = (closer * (0.34 * instant + 0.32 * sustain + 0.24 * long_spurt + 0.10 * out[f"{side}_strict_waveform_similarity"])).clip(0.0, 1.0)

    role_a = out["anchor_role_front_survival_score"] * out["partner_role_collapse_receiver_score"]
    role_b = out["partner_role_front_survival_score"] * out["anchor_role_collapse_receiver_score"]
    front_front = out["anchor_role_front_score"] * out["partner_role_front_score"] * out["race_role_load_score"]
    out["lap_role_front_receiver_pair_score"] = np.maximum(role_a, role_b).clip(0.0, 1.0)
    out["lap_role_front_front_collision_risk"] = front_front.clip(0.0, 1.0)
    out["lap_role_pair_probability_proxy"] = (
        0.36 * out["strict_waveform_pair_avg_score"]
        + 0.26 * out["goodrun_lap_pair_avg_score"]
        + 0.26 * out["lap_role_front_receiver_pair_score"]
        + 0.12 * (1.0 - out["lap_role_front_front_collision_risk"])
    ).clip(0.0, 1.0)
    out["lap_advanced_combo_score"] = (
        0.34 * out["strict_waveform_pair_avg_score"]
        + 0.22 * out["strict_waveform_pair_min_score"]
        + 0.24 * out["goodrun_lap_pair_avg_score"]
        + 0.20 * out["lap_role_pair_probability_proxy"]
    ).clip(0.0, 1.0)
    return out


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = pd.to_numeric(profit, errors="coerce").fillna(0.0).cumsum()
    return float((curve - curve.cummax()).min())


def roi_without_top(frame: pd.DataFrame, n: int) -> float:
    kept = frame.sort_values("return_yen_eval", ascending=False).iloc[n:]
    stake = float(kept["stake_yen_eval"].sum())
    ret = float(kept["return_yen_eval"].sum())
    return ret / stake * 100.0 if stake > 0 else 0.0


def metrics(frame: pd.DataFrame, segment: str, base_roi: float | None = None) -> dict[str, Any]:
    if frame.empty:
        return {"segment": segment, "tickets": 0}
    stake = pd.to_numeric(frame["stake_yen_eval"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(frame["return_yen_eval"], errors="coerce").fillna(0.0)
    profit = ret - stake
    roi = float(ret.sum() / stake.sum() * 100.0) if float(stake.sum()) > 0 else 0.0
    by_year = frame.groupby("year", dropna=True).agg(stake=("stake_yen_eval", "sum"), ret=("return_yen_eval", "sum"))
    year_roi = (by_year["ret"] / by_year["stake"] * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "segment": segment,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(profit.sum()),
        "roi_pct": roi,
        "roi_lift_vs_base_pt": roi - base_roi if base_roi is not None else np.nan,
        "hit_rate_pct": float(frame["hit_eval"].mean() * 100.0),
        "race_hit_rate_pct": float(frame.groupby("race_id")["hit_eval"].max().mean() * 100.0),
        "max_drawdown_yen": max_drawdown(frame.assign(_profit=profit).sort_values(["race_id", "ticket_type"])["_profit"]),
        "top1_removed_roi_pct": roi_without_top(frame, 1),
        "top3_removed_roi_pct": roi_without_top(frame, 3),
        "top5_removed_roi_pct": roi_without_top(frame, 5),
        "min_year_roi_pct": float(year_roi.min()) if not year_roi.empty else np.nan,
        "avg_strict_wave": float(ncol(frame, "strict_waveform_pair_avg_score", np.nan).mean()),
        "avg_goodrun_fit": float(ncol(frame, "goodrun_lap_pair_avg_score", np.nan).mean()),
        "avg_role_proxy": float(ncol(frame, "lap_role_pair_probability_proxy", np.nan).mean()),
        "avg_advanced_combo": float(ncol(frame, "lap_advanced_combo_score", np.nan).mean()),
    }


def segment_metrics(df: pd.DataFrame, min_tickets: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    score_cols = [
        "strict_waveform_pair_avg_score",
        "strict_waveform_pair_min_score",
        "goodrun_lap_pair_avg_score",
        "goodrun_lap_pair_min_score",
        "lap_role_front_receiver_pair_score",
        "lap_role_pair_probability_proxy",
        "lap_advanced_combo_score",
    ]
    low_cols = ["strict_waveform_pair_gap_score", "lap_role_front_front_collision_risk"]
    scopes = [("all", df)]
    for ticket_type, sub in df.groupby("ticket_type", dropna=False):
        scopes.append((str(ticket_type), sub))

    for scope, frame in scopes:
        if frame.empty:
            continue
        base = metrics(frame, f"{scope}:base")
        base_roi = float(base["roi_pct"])
        base_rows.append(base)
        for col in score_cols:
            values = ncol(frame, col)
            for q in [0.50, 0.60, 0.70, 0.80, 0.90]:
                th = float(values.quantile(q)) if values.notna().any() else np.nan
                if not np.isfinite(th):
                    continue
                sub = frame[values.ge(th)]
                if len(sub) >= min_tickets:
                    rows.append(metrics(sub, f"{scope}:{col}_ge_q{int(q*100)}({th:.3f})", base_roi))
        for col in low_cols:
            values = ncol(frame, col)
            for q in [0.20, 0.30, 0.40]:
                th = float(values.quantile(q)) if values.notna().any() else np.nan
                if not np.isfinite(th):
                    continue
                sub = frame[values.le(th)]
                if len(sub) >= min_tickets:
                    rows.append(metrics(sub, f"{scope}:{col}_le_q{int(q*100)}({th:.3f})", base_roi))

        combos = {
            "strict_wave_q70_goodrun_q60": ncol(frame, "strict_waveform_pair_avg_score").ge(ncol(frame, "strict_waveform_pair_avg_score").quantile(.70))
            & ncol(frame, "goodrun_lap_pair_avg_score").ge(ncol(frame, "goodrun_lap_pair_avg_score").quantile(.60)),
            "role_proxy_q70_low_collision_q40": ncol(frame, "lap_role_pair_probability_proxy").ge(ncol(frame, "lap_role_pair_probability_proxy").quantile(.70))
            & ncol(frame, "lap_role_front_front_collision_risk").le(ncol(frame, "lap_role_front_front_collision_risk").quantile(.40)),
            "advanced_combo_q70_low_gap_q40": ncol(frame, "lap_advanced_combo_score").ge(ncol(frame, "lap_advanced_combo_score").quantile(.70))
            & ncol(frame, "strict_waveform_pair_gap_score").le(ncol(frame, "strict_waveform_pair_gap_score").quantile(.40)),
            "strict_wave_q70_goodrun_q60_role_q60": ncol(frame, "strict_waveform_pair_avg_score").ge(ncol(frame, "strict_waveform_pair_avg_score").quantile(.70))
            & ncol(frame, "goodrun_lap_pair_avg_score").ge(ncol(frame, "goodrun_lap_pair_avg_score").quantile(.60))
            & ncol(frame, "lap_role_pair_probability_proxy").ge(ncol(frame, "lap_role_pair_probability_proxy").quantile(.60)),
        }
        for label, mask in combos.items():
            sub = frame[mask.fillna(False)]
            if len(sub) >= min_tickets:
                rows.append(metrics(sub, f"{scope}:{label}", base_roi))

    base_df = pd.DataFrame(base_rows)
    seg_df = pd.DataFrame(rows)
    if not seg_df.empty:
        seg_df = seg_df.sort_values(["roi_pct", "top3_removed_roi_pct", "tickets"], ascending=[False, False, False])
    return base_df, seg_df


def compact_output(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "race_id",
        "year",
        "ticket_type",
        "anchor_no",
        "anchor_name",
        "anchor_pop",
        "partner_no",
        "partner_name",
        "partner_pop",
        "stake_yen_eval",
        "return_yen_eval",
        "hit_eval",
        "strict_waveform_pair_avg_score",
        "strict_waveform_pair_min_score",
        "strict_waveform_pair_gap_score",
        "goodrun_lap_pair_avg_score",
        "goodrun_lap_pair_min_score",
        "goodrun_lap_pair_ready_score",
        "lap_role_front_receiver_pair_score",
        "lap_role_front_front_collision_risk",
        "lap_role_pair_probability_proxy",
        "lap_advanced_combo_score",
        "anchor_strict_waveform_similarity",
        "partner_strict_waveform_similarity",
        "anchor_goodrun_lap_fit_score",
        "partner_goodrun_lap_fit_score",
        "anchor_role_front_survival_score",
        "partner_role_front_survival_score",
        "anchor_role_collapse_receiver_score",
        "partner_role_collapse_receiver_score",
        "operation_profile",
        "runtime_action",
        "policy",
    ]
    return df[[c for c in wanted if c in df.columns]].copy()


def run(args: argparse.Namespace) -> dict[str, Any]:
    tickets = normalize_tickets(read_csv(args.tickets))
    runner_lap = load_runner_lap(args.runner_lap)
    goodrun = build_goodrun_profiles([Path(p) for p in args.runner_features])
    enriched = add_runner_lap_sides(tickets, runner_lap)
    enriched = add_goodrun_sides(enriched, goodrun)
    enriched = add_lap_scores(enriched)
    base_df, seg_df = segment_metrics(enriched, args.min_tickets)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    compact_output(enriched).to_csv(args.out_dir / "lap_waveform_role_goodrun_enriched_tickets.csv", index=False, encoding="utf-8-sig")
    base_df.to_csv(args.out_dir / "base_metrics.csv", index=False, encoding="utf-8-sig")
    seg_df.to_csv(args.out_dir / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "tickets": str(args.tickets),
        "runner_lap": str(args.runner_lap),
        "runner_features": [str(p) for p in args.runner_features],
        "output_dir": str(args.out_dir),
        "base": base_df.to_dict(orient="records"),
        "top_segments": seg_df.head(20).to_dict(orient="records") if not seg_df.empty else [],
        "notes": [
            "strict_waveform is a pseudo-200m six-phase waveform built from lap need/profile axes because full 200m lap strings are not available for all historical races.",
            "role_pair_probability_proxy evaluates front-survivor plus collapse-receiver compatibility, with a front-front collision penalty.",
            "goodrun_lap_fit uses only prior good runs, shifted by horse, to avoid same-race leakage.",
            "This script is evaluation-only and does not change runtime BUY gates.",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = [
        "# Lap Waveform / Role / Good-Run Evaluation",
        "",
        "検証対象:",
        "",
        "1. 疑似200m波形マッチング",
        "2. ラップ×脚質ロールのペア確率プロキシ",
        "3. 過去好走時だけのラップ適性",
        "",
        "全期間の生200mラップ系列は常設CSVに不足しているため、今回は `fast/slow/instant/sustain/long_spurt` から6相の疑似200m波形を構成しています。実200mラップがTARGET等から全件取得できたら、この層だけ差し替え可能です。",
    ]
    (args.out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate advanced lap waveform, role compatibility, and good-run-only lap aptitude.")
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    parser.add_argument("--runner-lap", type=Path, default=DEFAULT_RUNNER_LAP)
    parser.add_argument("--runner-features", type=Path, nargs="+", default=DEFAULT_RUNNER_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-tickets", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
