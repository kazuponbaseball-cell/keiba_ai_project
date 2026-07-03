from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TICKETS = ROOT / "outputs/analysis/lap_positive_expansion_v1/lap_positive_expansion_selected_tickets.csv"
DEFAULT_TRAIN = ROOT / "data/datasets/cache/lap_pedigree_interactions_confirmed_opponent/train_features.csv"
DEFAULT_TEST = ROOT / "data/datasets/cache/lap_pedigree_interactions_confirmed_opponent/test_features.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/lap_numeric_gap_pair_overlap_v1"

COL_DATE = "日付"
COL_RACE_NO = "Ｒ"
COL_RACE_ID = "レースID(新/馬番無)"
COL_HORSE_NO = "馬番"
COL_HORSE_ID = "血統登録番号"
COL_PLACE = "場所"
COL_CLASS = "クラス名"
COL_SURFACE = "芝・ダ"
COL_DISTANCE = "距離"
COL_GOING = "馬場状態"
COL_SCORE = "target_score"
COL_FINISH = "確定着順"
COL_POPULARITY = "人気"

LAP_COLS = ["PCI", "PCI3", "RPCI", "Ave-3F"]


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def ncol(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = df[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def parse_date(series: pd.Series) -> pd.Series:
    raw = series.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip()
    out = pd.to_datetime(raw, errors="coerce")
    yymmdd = raw.str.fullmatch(r"\d{6}")
    if yymmdd.any():
        out.loc[yymmdd] = pd.to_datetime(raw.loc[yymmdd], format="%y%m%d", errors="coerce")
    yyyymmdd = raw.str.fullmatch(r"\d{8}")
    if yyyymmdd.any():
        out.loc[yyyymmdd] = pd.to_datetime(raw.loc[yyyymmdd], format="%Y%m%d", errors="coerce")
    return out


def available_usecols(path: Path, wanted: list[str]) -> list[str]:
    header = read_csv(path, nrows=0)
    return [c for c in wanted if c in header.columns]


def load_runner_rows(paths: list[Path]) -> pd.DataFrame:
    wanted = [
        COL_DATE,
        COL_RACE_NO,
        COL_RACE_ID,
        COL_HORSE_NO,
        COL_HORSE_ID,
        COL_PLACE,
        COL_CLASS,
        COL_SURFACE,
        COL_DISTANCE,
        COL_GOING,
        COL_SCORE,
        COL_FINISH,
        COL_POPULARITY,
        *LAP_COLS,
        "前PCI",
        "前走PCI3",
        "前走RPCI",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        usecols = available_usecols(path, wanted)
        if not {COL_DATE, COL_RACE_ID, COL_HORSE_NO, COL_HORSE_ID}.issubset(usecols):
            continue
        frames.append(read_csv(path, usecols=usecols, dtype=str, low_memory=False))
    if not frames:
        raise FileNotFoundError("runner feature files were not found or did not contain required columns")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df["race_id"] = df[COL_RACE_ID].astype(str).str.replace(r"\.0$", "", regex=True)
    df["horse_no"] = pd.to_numeric(df[COL_HORSE_NO], errors="coerce").astype("Int64")
    df["horse_id"] = df[COL_HORSE_ID].astype(str).str.replace(r"\.0$", "", regex=True)
    df["race_date"] = parse_date(df[COL_DATE])
    df["race_no"] = ncol(df, COL_RACE_NO)
    for col in [COL_DISTANCE, COL_SCORE, COL_FINISH, COL_POPULARITY, *LAP_COLS, "前PCI", "前走PCI3", "前走RPCI"]:
        if col in df.columns:
            df[col] = ncol(df, col)
    df = df.dropna(subset=["race_id", "horse_no", "horse_id", "race_date"]).copy()
    df = df.sort_values(["race_date", "race_no", "race_id", "horse_no"], kind="mergesort")
    df = df.drop_duplicates(["race_id", "horse_no"], keep="last")
    return df


def _date_prior_stats(races: pd.DataFrame, keys: list[str], value: str, min_count: int, prefix: str) -> pd.DataFrame:
    if not all(k in races.columns for k in keys):
        return races[["race_id"]].assign(**{f"{prefix}_{value.lower()}_prior": np.nan, f"{prefix}_{value.lower()}_count": 0.0})

    base = races.dropna(subset=["race_date", value]).copy()
    daily = (
        base.groupby([*keys, "race_date"], dropna=False, sort=False)[value]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values([*keys, "race_date"], kind="mergesort")
    )
    grouped = daily.groupby(keys, dropna=False, sort=False)
    prior_sum = grouped["sum"].cumsum() - daily["sum"]
    prior_count = grouped["count"].cumsum() - daily["count"]
    daily[f"{prefix}_{value.lower()}_count"] = prior_count.astype(float)
    daily[f"{prefix}_{value.lower()}_prior"] = (prior_sum / prior_count.replace(0, np.nan)).where(prior_count >= min_count)

    return races[["race_id", *keys, "race_date"]].merge(
        daily[[*keys, "race_date", f"{prefix}_{value.lower()}_prior", f"{prefix}_{value.lower()}_count"]],
        on=[*keys, "race_date"],
        how="left",
    )[["race_id", f"{prefix}_{value.lower()}_prior", f"{prefix}_{value.lower()}_count"]]


def _global_date_prior(races: pd.DataFrame, value: str, min_count: int) -> pd.DataFrame:
    base = races.dropna(subset=["race_date", value]).copy()
    daily = base.groupby("race_date", sort=True)[value].agg(["sum", "count"]).reset_index()
    daily["global_count"] = daily["count"].cumsum() - daily["count"]
    daily[f"global_{value.lower()}_prior"] = (
        (daily["sum"].cumsum() - daily["sum"]) / daily["global_count"].replace(0, np.nan)
    ).where(daily["global_count"] >= min_count)
    return races[["race_id", "race_date"]].merge(
        daily[["race_date", f"global_{value.lower()}_prior", "global_count"]],
        on="race_date",
        how="left",
    )[["race_id", f"global_{value.lower()}_prior", "global_count"]]


def add_course_lap_priors(rows: pd.DataFrame) -> pd.DataFrame:
    races = (
        rows[
            [
                "race_id",
                "race_date",
                "race_no",
                COL_PLACE,
                COL_SURFACE,
                COL_DISTANCE,
                COL_GOING,
                COL_CLASS,
                "RPCI",
                "PCI3",
            ]
        ]
        .drop_duplicates("race_id", keep="first")
        .copy()
        .sort_values(["race_date", "race_no", "race_id"], kind="mergesort")
    )

    feature = races[["race_id"]].copy()
    for value in ["RPCI", "PCI3"]:
        priors = [
            _date_prior_stats(races, [COL_PLACE, COL_SURFACE, COL_DISTANCE, COL_GOING, COL_CLASS], value, 3, "full"),
            _date_prior_stats(races, [COL_SURFACE, COL_DISTANCE, COL_GOING, COL_CLASS], value, 5, "class"),
            _date_prior_stats(races, [COL_SURFACE, COL_DISTANCE, COL_GOING], value, 8, "going"),
            _date_prior_stats(races, [COL_SURFACE, COL_DISTANCE], value, 12, "course"),
            _global_date_prior(races, value, 30),
        ]
        tmp = races[["race_id"]].copy()
        for prior in priors:
            tmp = tmp.merge(prior, on="race_id", how="left")

        lower = value.lower()
        tmp[f"course_base_{lower}_prior"] = tmp[f"full_{lower}_prior"]
        tmp[f"course_base_{lower}_count"] = tmp[f"full_{lower}_count"]
        for name in ["class", "going", "course", "global"]:
            pcol = f"{name}_{lower}_prior"
            ccol = "global_count" if name == "global" else f"{name}_{lower}_count"
            missing = tmp[f"course_base_{lower}_prior"].isna()
            tmp.loc[missing, f"course_base_{lower}_prior"] = tmp.loc[missing, pcol]
            tmp.loc[missing, f"course_base_{lower}_count"] = tmp.loc[missing, ccol]

        fallback = 50.0 if value in {"RPCI", "PCI3"} else np.nan
        tmp[f"course_base_{lower}_prior"] = tmp[f"course_base_{lower}_prior"].fillna(fallback)
        tmp[f"course_base_{lower}_count"] = tmp[f"course_base_{lower}_count"].fillna(0.0)
        feature = feature.merge(tmp[["race_id", f"course_base_{lower}_prior", f"course_base_{lower}_count"]], on="race_id", how="left")

    return rows.merge(feature, on="race_id", how="left")


def _group_shifted_cumsum(group_key: pd.Series, values: pd.Series) -> pd.Series:
    csum = values.groupby(group_key, sort=False).cumsum()
    return csum.groupby(group_key, sort=False).shift()


def add_horse_lap_priors(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy().sort_values(["horse_id", "race_date", "race_id"], kind="mergesort")
    score = ncol(out, COL_SCORE, 0.0).fillna(0.0).clip(0.0, 1.0)
    finish = ncol(out, COL_FINISH)
    good = ((score >= 0.62) | finish.le(3)).astype(float)
    good_weight = (0.35 + 0.65 * score).where(good.eq(1.0), 0.0)

    horse = out["horse_id"]
    history_one = pd.Series(1.0, index=out.index)
    out["horse_lap_history_count_prior"] = _group_shifted_cumsum(horse, history_one).fillna(0.0)
    out["horse_lap_good_count_prior"] = _group_shifted_cumsum(horse, good).fillna(0.0)
    out["horse_lap_good_weight_prior"] = _group_shifted_cumsum(horse, good_weight).fillna(0.0)

    for value in ["RPCI", "PCI3"]:
        values = ncol(out, value)
        valid = values.notna().astype(float)
        out[f"horse_{value.lower()}_past5_mean"] = values.groupby(horse, sort=False).transform(
            lambda s: s.shift().rolling(5, min_periods=1).mean()
        )
        out[f"horse_{value.lower()}_past5_std"] = values.groupby(horse, sort=False).transform(
            lambda s: s.shift().rolling(5, min_periods=2).std(ddof=0)
        )
        out[f"horse_{value.lower()}_past5_count"] = valid.groupby(horse, sort=False).transform(
            lambda s: s.shift().rolling(5, min_periods=1).sum()
        )

        weighted_sum = _group_shifted_cumsum(horse, values.fillna(0.0) * good_weight)
        weight_sum = _group_shifted_cumsum(horse, good_weight.where(values.notna(), 0.0))
        out[f"horse_good_{value.lower()}_prior"] = weighted_sum / weight_sum.replace(0, np.nan)

    out["horse_rpci_prior"] = (
        out["horse_good_rpci_prior"]
        .fillna(out["horse_rpci_past5_mean"])
        .fillna(ncol(out, "前走RPCI"))
        .fillna(out["course_base_rpci_prior"])
    )
    out["horse_pci3_prior"] = (
        out["horse_good_pci3_prior"]
        .fillna(out["horse_pci3_past5_mean"])
        .fillna(ncol(out, "前走PCI3"))
        .fillna(out["course_base_pci3_prior"])
    )
    out["horse_lap_prior_confidence"] = (
        0.50 * (out["horse_lap_good_count_prior"].clip(upper=3.0) / 3.0)
        + 0.35 * (out["horse_lap_history_count_prior"].clip(upper=5.0) / 5.0)
        + 0.15 * ncol(out, "前走RPCI").notna().astype(float)
    ).clip(0.0, 1.0)
    out["horse_lap_prior_volatility"] = (
        0.5 * out["horse_rpci_past5_std"].fillna(3.0).clip(0.0, 8.0) / 8.0
        + 0.5 * out["horse_pci3_past5_std"].fillna(3.0).clip(0.0, 8.0) / 8.0
    ).clip(0.0, 1.0)

    keep = [
        "race_id",
        "horse_no",
        "horse_id",
        COL_PLACE,
        COL_CLASS,
        COL_SURFACE,
        COL_DISTANCE,
        COL_GOING,
        "race_date",
        "race_no",
        "course_base_rpci_prior",
        "course_base_pci3_prior",
        "course_base_rpci_count",
        "course_base_pci3_count",
        "horse_rpci_prior",
        "horse_pci3_prior",
        "horse_lap_prior_confidence",
        "horse_lap_prior_volatility",
        "horse_lap_history_count_prior",
        "horse_lap_good_count_prior",
    ]
    return out[[c for c in keep if c in out.columns]].drop_duplicates(["race_id", "horse_no"], keep="last")


def add_side_features(tickets: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = out["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    out["horse_a"] = pd.to_numeric(out["horse_a"], errors="coerce").astype("Int64")
    out["horse_b"] = pd.to_numeric(out["horse_b"], errors="coerce").astype("Int64")

    base_cols = [
        "race_id",
        "horse_no",
        "horse_id",
        COL_PLACE,
        COL_CLASS,
        COL_SURFACE,
        COL_DISTANCE,
        COL_GOING,
        "race_date",
        "course_base_rpci_prior",
        "course_base_pci3_prior",
        "course_base_rpci_count",
        "course_base_pci3_count",
        "horse_rpci_prior",
        "horse_pci3_prior",
        "horse_lap_prior_confidence",
        "horse_lap_prior_volatility",
        "horse_lap_history_count_prior",
        "horse_lap_good_count_prior",
    ]
    side_base = runners[[c for c in base_cols if c in runners.columns]].copy()

    for side, horse_col in [("a", "horse_a"), ("b", "horse_b")]:
        renamed = side_base.rename(
            columns={c: f"{side}_{c}" for c in side_base.columns if c not in {"race_id", "horse_no"}}
        )
        out = out.merge(
            renamed,
            left_on=["race_id", horse_col],
            right_on=["race_id", "horse_no"],
            how="left",
        ).drop(columns=["horse_no"], errors="ignore")

    return out


def add_numeric_gap_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    base_rpci = pd.concat(
        [ncol(out, "a_course_base_rpci_prior", 50.0), ncol(out, "b_course_base_rpci_prior", 50.0)],
        axis=1,
    ).mean(axis=1).fillna(50.0)
    base_pci3 = pd.concat(
        [ncol(out, "a_course_base_pci3_prior", 50.0), ncol(out, "b_course_base_pci3_prior", 50.0)],
        axis=1,
    ).mean(axis=1).fillna(50.0)

    pressure = pd.concat(
        [
            ncol(out, "anchor_race_early_pressure_score", 0.5),
            ncol(out, "partner_race_early_pressure_score", 0.5),
        ],
        axis=1,
    ).mean(axis=1).fillna(0.5).clip(0.0, 1.0)
    collapse = pd.concat(
        [
            ncol(out, "anchor_race_pace_collapse_risk", 0.5),
            ncol(out, "partner_race_pace_collapse_risk", 0.5),
        ],
        axis=1,
    ).mean(axis=1).fillna(0.5).clip(0.0, 1.0)
    slow = pd.concat(
        [
            ncol(out, "anchor_race_slow_pace_risk", 0.5),
            ncol(out, "partner_race_slow_pace_risk", 0.5),
        ],
        axis=1,
    ).mean(axis=1).fillna(0.5).clip(0.0, 1.0)
    v2_fast = ncol(out, "v2_prob_fast", 0.0).fillna(0.0).clip(0.0, 1.0)
    v2_slow = ncol(out, "v2_prob_slow", 0.0).fillna(0.0).clip(0.0, 1.0)
    v2_instant = ncol(out, "v2_prob_instant", 0.0).fillna(0.0).clip(0.0, 1.0)
    v2_sustain = ncol(out, "v2_prob_sustain", 0.0).fillna(0.0).clip(0.0, 1.0)

    out["expected_numeric_rpci"] = (
        base_rpci
        + 2.7 * slow
        + 1.2 * v2_slow
        - 2.9 * collapse
        - 1.2 * pressure
        - 1.1 * v2_fast
        + 0.4 * v2_sustain
    ).clip(42.0, 58.0)
    out["expected_numeric_pci3"] = (
        base_pci3
        + 1.1 * slow
        + 0.8 * v2_slow
        + 0.7 * v2_instant
        + 0.4 * v2_sustain
        - 0.8 * collapse
        - 0.6 * v2_fast
        - 0.2 * pressure
    ).clip(42.0, 60.0)
    out["expected_numeric_lap_confidence"] = (
        0.45 * ncol(out, "v2_confidence", 0.5).fillna(0.5).clip(0.0, 1.0)
        + 0.25 * (1.0 - (pressure - slow).abs()).clip(0.0, 1.0)
        + 0.20 * (pd.concat([ncol(out, "a_course_base_rpci_count", 0.0), ncol(out, "b_course_base_rpci_count", 0.0)], axis=1).mean(axis=1).clip(0.0, 25.0) / 25.0)
        + 0.10 * (pd.concat([ncol(out, "a_course_base_pci3_count", 0.0), ncol(out, "b_course_base_pci3_count", 0.0)], axis=1).mean(axis=1).clip(0.0, 25.0) / 25.0)
    ).clip(0.0, 1.0)

    for side in ["a", "b"]:
        rpci_prior = ncol(out, f"{side}_horse_rpci_prior", np.nan).fillna(out["expected_numeric_rpci"])
        pci3_prior = ncol(out, f"{side}_horse_pci3_prior", np.nan).fillna(out["expected_numeric_pci3"])
        conf = ncol(out, f"{side}_horse_lap_prior_confidence", 0.0).fillna(0.0).clip(0.0, 1.0)
        vol = ncol(out, f"{side}_horse_lap_prior_volatility", 0.35).fillna(0.35).clip(0.0, 1.0)

        out[f"{side}_numeric_rpci_gap"] = (out["expected_numeric_rpci"] - rpci_prior).abs()
        out[f"{side}_numeric_pci3_gap"] = (out["expected_numeric_pci3"] - pci3_prior).abs()
        out[f"{side}_numeric_rpci_fit"] = (1.0 - out[f"{side}_numeric_rpci_gap"] / 8.0).clip(0.0, 1.0)
        out[f"{side}_numeric_pci3_fit"] = (1.0 - out[f"{side}_numeric_pci3_gap"] / 8.0).clip(0.0, 1.0)
        out[f"{side}_lap_numeric_fit_score"] = (
            0.52 * out[f"{side}_numeric_rpci_fit"] + 0.48 * out[f"{side}_numeric_pci3_fit"]
        ).clip(0.0, 1.0)
        out[f"{side}_lap_numeric_confident_fit_score"] = (
            out[f"{side}_lap_numeric_fit_score"]
            * (0.55 + 0.45 * conf)
            * (0.65 + 0.35 * out["expected_numeric_lap_confidence"])
            * (1.0 - 0.20 * vol)
        ).clip(0.0, 1.0)

    out["pair_lap_numeric_fit_min"] = pd.concat(
        [out["a_lap_numeric_fit_score"], out["b_lap_numeric_fit_score"]], axis=1
    ).min(axis=1)
    out["pair_lap_numeric_fit_avg"] = pd.concat(
        [out["a_lap_numeric_fit_score"], out["b_lap_numeric_fit_score"]], axis=1
    ).mean(axis=1)
    out["pair_lap_numeric_confident_min"] = pd.concat(
        [out["a_lap_numeric_confident_fit_score"], out["b_lap_numeric_confident_fit_score"]], axis=1
    ).min(axis=1)
    out["pair_lap_numeric_confident_avg"] = pd.concat(
        [out["a_lap_numeric_confident_fit_score"], out["b_lap_numeric_confident_fit_score"]], axis=1
    ).mean(axis=1)
    out["pair_lap_numeric_confidence_min"] = pd.concat(
        [
            ncol(out, "a_horse_lap_prior_confidence", 0.0).fillna(0.0),
            ncol(out, "b_horse_lap_prior_confidence", 0.0).fillna(0.0),
        ],
        axis=1,
    ).min(axis=1)
    out["pair_lap_numeric_rpci_overlap"] = (
        1.0
        - (
            ncol(out, "a_horse_rpci_prior", np.nan).fillna(out["expected_numeric_rpci"])
            - ncol(out, "b_horse_rpci_prior", np.nan).fillna(out["expected_numeric_rpci"])
        ).abs()
        / 8.0
    ).clip(0.0, 1.0)
    out["pair_lap_numeric_pci3_overlap"] = (
        1.0
        - (
            ncol(out, "a_horse_pci3_prior", np.nan).fillna(out["expected_numeric_pci3"])
            - ncol(out, "b_horse_pci3_prior", np.nan).fillna(out["expected_numeric_pci3"])
        ).abs()
        / 8.0
    ).clip(0.0, 1.0)
    out["pair_lap_numeric_pair_balance"] = (
        1.0 - (out["a_lap_numeric_fit_score"] - out["b_lap_numeric_fit_score"]).abs() / 0.45
    ).clip(0.0, 1.0)
    out["pair_lap_numeric_conflict_score"] = (
        1.0
        - (
            0.34 * out["pair_lap_numeric_fit_min"]
            + 0.22 * out["pair_lap_numeric_confident_min"]
            + 0.18 * out["pair_lap_numeric_rpci_overlap"]
            + 0.16 * out["pair_lap_numeric_pci3_overlap"]
            + 0.10 * out["pair_lap_numeric_pair_balance"]
        )
    ).clip(0.0, 1.0)
    out["pair_lap_numeric_shadow_score"] = (
        0.34 * out["pair_lap_numeric_fit_min"]
        + 0.24 * out["pair_lap_numeric_confident_min"]
        + 0.18 * out["pair_lap_numeric_rpci_overlap"]
        + 0.14 * out["pair_lap_numeric_pci3_overlap"]
        + 0.10 * (1.0 - out["pair_lap_numeric_conflict_score"])
    ).clip(0.0, 1.0)

    return out


def max_drawdown(net: pd.Series) -> float:
    curve = net.cumsum()
    drawdown = curve.cummax() - curve
    return float(drawdown.max()) if len(drawdown) else 0.0


def roi_without_top_returns(frame: pd.DataFrame, n: int) -> float:
    if frame.empty:
        return float("nan")
    stake = ncol(frame, "stake_yen", 0.0).fillna(0.0)
    ret = ncol(frame, "return_yen", 0.0).fillna(0.0)
    if n > 0:
        drop_idx = ret.sort_values(ascending=False).head(n).index
        stake = stake.drop(index=drop_idx)
        ret = ret.drop(index=drop_idx)
    return float(ret.sum() / stake.sum() * 100.0) if stake.sum() > 0 else float("nan")


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    stake = ncol(frame, "stake_yen", 0.0).fillna(0.0)
    ret = ncol(frame, "return_yen", 0.0).fillna(0.0)
    hit = frame.get("hit", pd.Series(False, index=frame.index)).astype(bool)
    years = pd.to_numeric(frame.get("year", pd.Series(np.nan, index=frame.index)), errors="coerce")
    by_year = {}
    for year, grp in frame.assign(_year=years).dropna(subset=["_year"]).groupby("_year"):
        y_stake = ncol(grp, "stake_yen", 0.0).fillna(0.0).sum()
        y_ret = ncol(grp, "return_yen", 0.0).fillna(0.0).sum()
        if y_stake > 0:
            by_year[str(int(year))] = float(y_ret / y_stake * 100.0)
    return {
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame.columns else int(len(frame)),
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(ret.sum() - stake.sum()),
        "roi": float(ret.sum() / stake.sum() * 100.0) if stake.sum() > 0 else float("nan"),
        "hit_rate": float(hit.mean() * 100.0) if len(hit) else float("nan"),
        "max_drawdown_yen": max_drawdown(ret - stake),
        "top1_removed_roi": roi_without_top_returns(frame, 1),
        "top3_removed_roi": roi_without_top_returns(frame, 3),
        "top5_removed_roi": roi_without_top_returns(frame, 5),
        "min_year_roi": float(min(by_year.values())) if by_year else float("nan"),
        "year_roi": by_year,
    }


def qcut_label(series: pd.Series, q: int = 5) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(values.rank(method="first"), q=q, labels=[f"q{i + 1}" for i in range(q)]).astype(str)
    except ValueError:
        return pd.Series("all", index=series.index)


def collect_segments(df: pd.DataFrame, min_tickets: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(name: str, mask: pd.Series, extra: dict[str, Any] | None = None) -> None:
        subset = df.loc[mask.fillna(False)].copy()
        if len(subset) < min_tickets:
            return
        item = {"segment": name, **metrics(subset)}
        if extra:
            item.update(extra)
        rows.append(item)

    for (policy, ticket_type), group in df.groupby(["policy", "ticket_type"], dropna=False):
        base_mask = df.index.isin(group.index)
        add(f"base::{policy}::{ticket_type}", pd.Series(base_mask, index=df.index), {"policy": policy, "ticket_type": ticket_type, "rule": "base"})

        local = df.loc[base_mask]
        for col, direction in [
            ("pair_lap_numeric_fit_min", "high"),
            ("pair_lap_numeric_confident_min", "high"),
            ("pair_lap_numeric_shadow_score", "high"),
            ("pair_lap_numeric_conflict_score", "low"),
            ("pair_lap_numeric_rpci_overlap", "high"),
            ("pair_lap_numeric_pci3_overlap", "high"),
        ]:
            if col not in local.columns:
                continue
            values = pd.to_numeric(local[col], errors="coerce")
            for pct in [0.50, 0.60, 0.70, 0.80]:
                threshold = values.quantile(pct)
                if not np.isfinite(threshold):
                    continue
                if direction == "high":
                    mask = base_mask & (pd.to_numeric(df[col], errors="coerce") >= threshold)
                    rule = f"{col}_ge_q{int(pct * 100)}"
                else:
                    threshold = values.quantile(1 - pct)
                    mask = base_mask & (pd.to_numeric(df[col], errors="coerce") <= threshold)
                    rule = f"{col}_le_q{int((1 - pct) * 100)}"
                add(f"{rule}::{policy}::{ticket_type}", pd.Series(mask, index=df.index), {"policy": policy, "ticket_type": ticket_type, "rule": rule})

        fit = pd.to_numeric(df["pair_lap_numeric_fit_min"], errors="coerce")
        shadow = pd.to_numeric(df["pair_lap_numeric_shadow_score"], errors="coerce")
        conflict = pd.to_numeric(df["pair_lap_numeric_conflict_score"], errors="coerce")
        conf = pd.to_numeric(df["pair_lap_numeric_confidence_min"], errors="coerce")
        local_fit = pd.to_numeric(local["pair_lap_numeric_fit_min"], errors="coerce")
        local_shadow = pd.to_numeric(local["pair_lap_numeric_shadow_score"], errors="coerce")
        local_conflict = pd.to_numeric(local["pair_lap_numeric_conflict_score"], errors="coerce")
        local_conf = pd.to_numeric(local["pair_lap_numeric_confidence_min"], errors="coerce")

        combo_specs = [
            (
                "numeric_fit_top30_conf_ok_conflict_low",
                (fit >= local_fit.quantile(0.70))
                & (conf >= local_conf.quantile(0.50))
                & (conflict <= local_conflict.quantile(0.40)),
            ),
            (
                "numeric_shadow_top25_conflict_low",
                (shadow >= local_shadow.quantile(0.75)) & (conflict <= local_conflict.quantile(0.45)),
            ),
            (
                "numeric_fit_top40_overlap_good",
                (fit >= local_fit.quantile(0.60))
                & (pd.to_numeric(df["pair_lap_numeric_rpci_overlap"], errors="coerce") >= pd.to_numeric(local["pair_lap_numeric_rpci_overlap"], errors="coerce").quantile(0.60))
                & (pd.to_numeric(df["pair_lap_numeric_pci3_overlap"], errors="coerce") >= pd.to_numeric(local["pair_lap_numeric_pci3_overlap"], errors="coerce").quantile(0.60)),
            ),
        ]
        for rule, rule_mask in combo_specs:
            add(f"{rule}::{policy}::{ticket_type}", pd.Series(base_mask, index=df.index) & rule_mask, {"policy": policy, "ticket_type": ticket_type, "rule": rule})

    return pd.DataFrame(rows).sort_values(["roi", "tickets"], ascending=[False, False]) if rows else pd.DataFrame()


def summarize_quantile_bands(df: pd.DataFrame, min_tickets: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cols = [
        "pair_lap_numeric_fit_min",
        "pair_lap_numeric_confident_min",
        "pair_lap_numeric_shadow_score",
        "pair_lap_numeric_conflict_score",
        "pair_lap_numeric_rpci_overlap",
        "pair_lap_numeric_pci3_overlap",
    ]
    for col in cols:
        if col not in df.columns:
            continue
        band = qcut_label(df[col], 5)
        for (ticket_type, label), group in df.assign(_band=band).groupby(["ticket_type", "_band"], dropna=False):
            if len(group) < min_tickets:
                continue
            row = {"feature": col, "band": label, "ticket_type": ticket_type, **metrics(group)}
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-tickets", type=int, default=20)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tickets = read_csv(args.tickets, low_memory=False)
    if "policy" not in tickets.columns:
        if "mcs_pbo_policy" in tickets.columns:
            tickets["policy"] = tickets["mcs_pbo_policy"].fillna(args.tickets.stem).astype(str)
        elif "operation_profile" in tickets.columns:
            tickets["policy"] = tickets["operation_profile"].fillna(args.tickets.stem).astype(str)
        else:
            tickets["policy"] = args.tickets.stem
    if "ticket_type" not in tickets.columns:
        tickets["ticket_type"] = "unknown"
    rows = load_runner_rows([args.train, args.test])
    rows = add_course_lap_priors(rows)
    runner_features = add_horse_lap_priors(rows)

    enriched = add_side_features(tickets, runner_features)
    enriched = add_numeric_gap_pair_features(enriched)
    enriched["numeric_lap_feature_coverage"] = (
        enriched[["a_horse_id", "b_horse_id", "a_horse_rpci_prior", "b_horse_rpci_prior"]].notna().mean(axis=1)
    )

    enriched_path = out_dir / "lap_numeric_gap_pair_overlap_enriched_tickets.csv"
    enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")

    segment_table = collect_segments(enriched, args.min_tickets)
    segment_path = out_dir / "lap_numeric_gap_pair_overlap_segment_metrics.csv"
    segment_table.to_csv(segment_path, index=False, encoding="utf-8-sig")

    band_table = summarize_quantile_bands(enriched, args.min_tickets)
    band_path = out_dir / "lap_numeric_gap_pair_overlap_quantile_bands.csv"
    band_table.to_csv(band_path, index=False, encoding="utf-8-sig")

    base_metrics = (
        enriched.groupby(["policy", "ticket_type"], dropna=False)
        .apply(lambda g: pd.Series(metrics(g)), include_groups=False)
        .reset_index()
    )
    base_path = out_dir / "lap_numeric_gap_pair_overlap_base_metrics.csv"
    base_metrics.to_csv(base_path, index=False, encoding="utf-8-sig")

    top_segments = []
    if not segment_table.empty:
        keep_cols = [
            "segment",
            "policy",
            "ticket_type",
            "rule",
            "tickets",
            "races",
            "roi",
            "hit_rate",
            "top3_removed_roi",
            "top5_removed_roi",
            "min_year_roi",
        ]
        top_segments = segment_table[[c for c in keep_cols if c in segment_table.columns]].head(30).to_dict("records")

    summary = {
        "input_tickets": str(args.tickets),
        "input_rows": int(len(rows)),
        "input_ticket_rows": int(len(tickets)),
        "enriched_rows": int(len(enriched)),
        "coverage_mean": float(enriched["numeric_lap_feature_coverage"].mean()),
        "output_files": {
            "enriched": str(enriched_path),
            "segments": str(segment_path),
            "bands": str(band_path),
            "base": str(base_path),
        },
        "top_segments": top_segments,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
