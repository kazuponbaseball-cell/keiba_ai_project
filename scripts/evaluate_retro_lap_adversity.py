from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_historical_condition_lap_context import DEFAULT_FEATURES  # noqa: E402
from scripts.evaluate_shape_adjusted_pair_selection import (  # noqa: E402
    DEFAULT_RACE_SHAPE,
    DEFAULT_UNIVERSE,
    add_shape_scores,
    bool_col,
    gate_mask,
    hit_flag,
    load_universe,
    ncol,
    ticket_return,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "analysis" / "retro_lap_adversity_v1"

RACE_COL = "レースID(新/馬番無)"
DATE_COL = "日付"
DATE_STR_COL = "日付S"
HORSE_COL = "血統登録番号"
HORSE_NO_COL = "馬番"

RUN_COLS = [
    "retro_lap_front_load_resistance",
    "retro_lap_front_load_excuse",
    "retro_lap_slow_closer_excuse",
    "retro_lap_long_spurt_resistance",
    "retro_lap_positive_score",
    "retro_lap_overhelped_score",
    "retro_lap_negative_score",
    "retro_lap_front_load_adversity",
    "retro_lap_slow_rear_adversity",
]

PRIOR_EXPORT_COLS = [
    "race_id",
    "horse_no",
    "horse_id",
    "retro_lap_prior_count",
    "prev_retro_lap_positive_score",
    "past3_avg_retro_lap_positive_score",
    "past3_max_retro_lap_positive_score",
    "past5_avg_retro_lap_positive_score",
    "past5_max_retro_lap_positive_score",
    "prev_retro_lap_overhelped_score",
    "past3_avg_retro_lap_overhelped_score",
    "past3_max_retro_lap_overhelped_score",
    "prev_retro_lap_negative_score",
    "past3_avg_retro_lap_negative_score",
    "past3_max_retro_lap_negative_score",
    "past3_avg_retro_lap_front_load_resistance",
    "past3_avg_retro_lap_slow_closer_excuse",
    "past3_avg_retro_lap_long_spurt_resistance",
]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv_any(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def num_col(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def clip01(x: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return pd.Series(x).clip(0.0, 1.0)


def parse_jra_date(raw: pd.Series, raw_str: pd.Series | None = None) -> pd.Series:
    if raw_str is not None:
        parsed = pd.to_datetime(raw_str.astype(str), format="%Y.%m.%d", errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    missing = parsed.isna()
    if missing.any():
        x = pd.to_numeric(raw, errors="coerce")
        y = (x // 10000).astype("float")
        m = ((x % 10000) // 100).astype("float")
        d = (x % 100).astype("float")
        year = np.where(y < 70, 2000 + y, 1900 + y)
        ymd = pd.DataFrame({"year": year, "month": m, "day": d}, index=raw.index)
        parsed2 = pd.to_datetime(ymd, errors="coerce")
        parsed = parsed.where(~missing, parsed2)
    return parsed


def stable_norm(series: pd.Series, lo: float, hi: float, default: float = 0.5) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(default).clip(0.0, 1.0)


def load_runner_rows(paths: list[Path]) -> pd.DataFrame:
    wanted = [
        DATE_COL,
        DATE_STR_COL,
        "場所",
        "Ｒ",
        "レース名",
        "クラス名",
        "馬名",
        "頭数",
        "出走頭数",
        HORSE_NO_COL,
        "人気",
        "単勝オッズ",
        "芝・ダ",
        "距離",
        "馬場状態",
        "着差",
        "3角",
        "4角",
        "4角.1",
        "上り3F順",
        "確定着順",
        RACE_COL,
        HORSE_COL,
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
        header = read_csv_any(path, nrows=0)
        usecols = [c for c in wanted if c in header.columns]
        if RACE_COL not in usecols or HORSE_NO_COL not in usecols or HORSE_COL not in usecols:
            continue
        frames.append(read_csv_any(path, usecols=usecols))
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw["race_id"] = raw[RACE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    raw["horse_no"] = pd.to_numeric(raw[HORSE_NO_COL], errors="coerce").astype("Int64")
    raw["horse_id"] = raw[HORSE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    raw["race_date"] = parse_jra_date(raw[DATE_COL], raw.get(DATE_STR_COL))
    raw = raw.dropna(subset=["race_id", "horse_no", "horse_id", "race_date"]).copy()
    raw = raw.drop_duplicates(["race_id", "horse_no", "horse_id"], keep="last")
    return raw.sort_values(["horse_id", "race_date", "race_id"]).reset_index(drop=True)


def add_retro_lap_run_scores(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    field = num_col(out, "出走頭数").fillna(num_col(out, "頭数")).fillna(0.0)
    race_size = out.groupby("race_id")["race_id"].transform("size").astype(float)
    field = field.where(field.gt(1), race_size).clip(lower=2.0)

    corner4 = num_col(out, "4角.1").fillna(num_col(out, "4角"))
    pos_rate = ((corner4 - 1.0) / (field - 1.0)).clip(0.0, 1.0).fillna(0.5)
    front_pos = (1.0 - pos_rate).clip(0.0, 1.0)
    rear_pos = pos_rate.clip(0.0, 1.0)
    mid_or_rear = (0.55 * rear_pos + 0.45 * (1.0 - (front_pos - 0.35).abs() / 0.65).clip(0.0, 1.0)).clip(0.0, 1.0)

    finish = num_col(out, "確定着順")
    finish_score = (1.0 - ((finish - 1.0) / (field - 1.0))).clip(0.0, 1.0).fillna(0.5)
    final3_rank = num_col(out, "上り3F順")
    final3_score = (1.0 - ((final3_rank - 1.0) / (field - 1.0))).clip(0.0, 1.0).fillna(0.5)
    target = num_col(out, "target_score").fillna(finish_score).clip(0.0, 1.0)
    margin = num_col(out, "着差").fillna(0.8)
    margin_good = (1.0 - ((margin + 0.15).clip(lower=-0.3, upper=2.2) + 0.3) / 2.5).clip(0.0, 1.0)

    content = (0.46 * target + 0.24 * finish_score + 0.18 * final3_score + 0.12 * margin_good).clip(0.0, 1.0)
    close_but_lost = ((1.0 - target).clip(0.0, 1.0) * (0.65 * margin_good + 0.35 * final3_score)).clip(0.0, 1.0)

    rpci = num_col(out, "RPCI").fillna(50.0)
    pci = num_col(out, "PCI").fillna(50.0)
    pci3 = num_col(out, "PCI3").fillna(pci)
    ave3f = num_col(out, "Ave-3F")

    front_loaded = clip01((50.5 - rpci) / 8.0)
    slow_finish = clip01((rpci - 51.5) / 8.0)
    instant_finish = clip01((pci - rpci) / 7.0)
    long_spurt = clip01((pci3 - rpci) / 7.0)
    fast_clock = (1.0 - stable_norm(ave3f, lo=34.2, hi=38.8)).fillna(0.5)

    front_load_adversity = (front_loaded * front_pos * (0.70 + 0.30 * fast_clock)).clip(0.0, 1.0)
    slow_rear_adversity = (slow_finish * rear_pos * (0.65 + 0.35 * instant_finish)).clip(0.0, 1.0)
    long_spurt_adversity = (long_spurt * mid_or_rear * (0.55 + 0.45 * final3_score)).clip(0.0, 1.0)

    out["retro_lap_front_load_adversity"] = front_load_adversity
    out["retro_lap_slow_rear_adversity"] = slow_rear_adversity
    out["retro_lap_front_load_resistance"] = (front_load_adversity * content).clip(0.0, 1.0)
    out["retro_lap_front_load_excuse"] = (front_load_adversity * close_but_lost).clip(0.0, 1.0)
    out["retro_lap_slow_closer_excuse"] = (slow_rear_adversity * (0.55 * final3_score + 0.25 * margin_good + 0.20 * target)).clip(0.0, 1.0)
    out["retro_lap_long_spurt_resistance"] = (long_spurt_adversity * (0.60 * final3_score + 0.25 * target + 0.15 * margin_good)).clip(0.0, 1.0)

    positive_parts = [
        out["retro_lap_front_load_resistance"],
        out["retro_lap_front_load_excuse"],
        out["retro_lap_slow_closer_excuse"],
        out["retro_lap_long_spurt_resistance"],
    ]
    out["retro_lap_positive_score"] = np.maximum.reduce([s.to_numpy() for s in positive_parts]).clip(0.0, 1.0)
    out["retro_lap_overhelped_score"] = np.maximum(
        (slow_finish * front_pos * content).to_numpy(),
        (front_loaded * rear_pos * content).to_numpy(),
    ).clip(0.0, 1.0)
    out["retro_lap_negative_score"] = np.maximum(
        (slow_finish * front_pos * (1.0 - content)).to_numpy(),
        (front_loaded * rear_pos * (1.0 - content)).to_numpy(),
    ).clip(0.0, 1.0)
    out["retro_lap_regime"] = np.select(
        [
            front_loaded.ge(0.35),
            slow_finish.ge(0.35),
            long_spurt.ge(0.30),
            instant_finish.ge(0.30),
        ],
        ["front_loaded", "slow_finish", "long_spurt", "instant_finish"],
        default="neutral",
    )
    return out


def add_prior_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.sort_values(["horse_id", "race_date", "race_id"]).copy()
    grouped = out.groupby("horse_id", sort=False)
    out["retro_lap_prior_count"] = grouped.cumcount()
    for col in RUN_COLS:
        shifted = grouped[col].shift(1)
        by_horse = shifted.groupby(out["horse_id"], sort=False)
        out[f"prev_{col}"] = shifted
        out[f"past3_avg_{col}"] = by_horse.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"past3_max_{col}"] = by_horse.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
        out[f"past5_avg_{col}"] = by_horse.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"past5_max_{col}"] = by_horse.rolling(5, min_periods=1).max().reset_index(level=0, drop=True)
    keep = [c for c in PRIOR_EXPORT_COLS if c in out.columns]
    return out[keep].drop_duplicates(["race_id", "horse_no"], keep="last").copy()


def merge_pair_features(universe: pd.DataFrame, runner: pd.DataFrame) -> pd.DataFrame:
    out = universe.copy()
    runner = runner.copy()
    runner["race_id"] = runner["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    runner["horse_no"] = pd.to_numeric(runner["horse_no"], errors="coerce").astype("Int64")
    value_cols = [c for c in runner.columns if c not in {"race_id", "horse_no", "horse_id"}]

    anchor = runner[["race_id", "horse_no"] + value_cols].rename(
        columns={"horse_no": "anchor_no", **{c: f"anchor_{c}" for c in value_cols}}
    )
    partner = runner[["race_id", "horse_no"] + value_cols].rename(
        columns={"horse_no": "partner_no", **{c: f"partner_{c}" for c in value_cols}}
    )
    out["anchor_no"] = pd.to_numeric(out["anchor_no"], errors="coerce").astype("Int64")
    out["partner_no"] = pd.to_numeric(out["partner_no"], errors="coerce").astype("Int64")
    out = out.merge(anchor, on=["race_id", "anchor_no"], how="left")
    out = out.merge(partner, on=["race_id", "partner_no"], how="left")

    a_pos = ncol(out, "anchor_past3_max_retro_lap_positive_score", 0.0).clip(0.0, 1.0)
    p_pos = ncol(out, "partner_past3_max_retro_lap_positive_score", 0.0).clip(0.0, 1.0)
    a_avg = ncol(out, "anchor_past3_avg_retro_lap_positive_score", 0.0).clip(0.0, 1.0)
    p_avg = ncol(out, "partner_past3_avg_retro_lap_positive_score", 0.0).clip(0.0, 1.0)
    a_neg = ncol(out, "anchor_past3_max_retro_lap_negative_score", 0.0).clip(0.0, 1.0)
    p_neg = ncol(out, "partner_past3_max_retro_lap_negative_score", 0.0).clip(0.0, 1.0)
    a_help = ncol(out, "anchor_past3_max_retro_lap_overhelped_score", 0.0).clip(0.0, 1.0)
    p_help = ncol(out, "partner_past3_max_retro_lap_overhelped_score", 0.0).clip(0.0, 1.0)
    evidence_a = ncol(out, "anchor_retro_lap_prior_count", 0.0)
    evidence_p = ncol(out, "partner_retro_lap_prior_count", 0.0)

    out["retro_lap_pair_pos_max"] = np.maximum(a_pos, p_pos)
    out["retro_lap_pair_pos_avg"] = ((a_avg + p_avg) / 2.0).clip(0.0, 1.0)
    out["retro_lap_pair_pos_min"] = np.minimum(a_avg, p_avg)
    out["retro_lap_pair_negative_max"] = np.maximum(a_neg, p_neg)
    out["retro_lap_pair_overhelped_max"] = np.maximum(a_help, p_help)
    out["retro_lap_pair_evidence_min"] = np.minimum(evidence_a, evidence_p)
    out["retro_lap_pair_evidence_ready"] = out["retro_lap_pair_evidence_min"].ge(1).astype(float)
    out["retro_lap_pair_fit_score"] = (
        0.38 * out["retro_lap_pair_pos_max"]
        + 0.22 * out["retro_lap_pair_pos_avg"]
        + 0.22 * (1.0 - out["retro_lap_pair_negative_max"])
        + 0.12 * (1.0 - out["retro_lap_pair_overhelped_max"])
        + 0.06 * out["retro_lap_pair_evidence_ready"]
    ).clip(0.0, 1.0)
    out["retro_lap_pair_risk_score"] = (
        0.52 * out["retro_lap_pair_negative_max"]
        + 0.34 * out["retro_lap_pair_overhelped_max"]
        + 0.14 * (1.0 - out["retro_lap_pair_evidence_ready"])
    ).clip(0.0, 1.0)
    return out


def select_top_per_race(frame: pd.DataFrame, score_col: str, ticket_type: str, policy: str, gate: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    sort_cols = [c for c in ["race_id", score_col, "market_overlay_score", "pair_quinella_score"] if c in frame.columns]
    selected = (
        frame.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1))
        .drop_duplicates("race_id", keep="first")
        .copy()
    )
    selected["policy"] = policy
    selected["gate"] = gate
    selected["ticket_type"] = ticket_type
    selected["stake_yen"] = 100.0
    selected["return_yen"] = ticket_return(selected, ticket_type)
    selected["hit"] = hit_flag(selected, ticket_type)
    selected["ticket_key"] = selected.apply(
        lambda r: f"{ticket_type}:{r['race_id']}:{int(r['horse_a'])}-{int(r['horse_b'])}", axis=1
    )
    return selected


def metric_row(frame: pd.DataFrame, policy: str) -> dict[str, Any]:
    stake = float(frame["stake_yen"].sum()) if not frame.empty else 0.0
    ret = float(frame["return_yen"].sum()) if not frame.empty else 0.0
    max_ret = float(frame["return_yen"].max()) if not frame.empty else 0.0
    ex_top = frame.drop(frame["return_yen"].idxmax()) if not frame.empty and max_ret > 0 else frame.iloc[0:0]
    ex_stake = float(ex_top["stake_yen"].sum()) if not ex_top.empty else 0.0
    ex_ret = float(ex_top["return_yen"].sum()) if not ex_top.empty else 0.0
    return {
        "policy": policy,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if not frame.empty else 0,
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi_pct": round(ret / stake * 100, 1) if stake > 0 else 0.0,
        "hit_rate_pct": round(float(frame["hit"].mean() * 100), 1) if not frame.empty else 0.0,
        "top_return_share_pct": round(max_ret / ret * 100, 1) if ret > 0 else 0.0,
        "roi_ex_top1_pct": round(ex_ret / ex_stake * 100, 1) if ex_stake > 0 else 0.0,
        "avg_retro_lap_fit": round(float(ncol(frame, "retro_lap_pair_fit_score", 0.0).mean()), 3) if not frame.empty else np.nan,
        "avg_retro_lap_risk": round(float(ncol(frame, "retro_lap_pair_risk_score", 0.0).mean()), 3) if not frame.empty else np.nan,
    }


def run_overlay_backtest(
    df: pd.DataFrame,
    gates: list[str],
    shape_weights: list[float],
    retro_weights: list[float],
    ticket_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []

    for gate in gates:
        gated = df[gate_mask(df, gate)].copy()
        if gated.empty:
            continue
        for ticket_type in ticket_types:
            for shape_weight in shape_weights:
                base_col = f"base_shape_score_w{shape_weight:.2f}"
                risk_weight = min(0.12, shape_weight * 0.75)
                gated[base_col] = (
                    (1.0 - shape_weight) * gated["shape_base_rank_score"]
                    + shape_weight * gated["shape_pair_fit_score"]
                    + 0.06 * gated["shape_value_score"]
                    - risk_weight * gated["shape_pair_risk_score"]
                )
                baseline = select_top_per_race(
                    gated,
                    base_col,
                    ticket_type,
                    f"{ticket_type}_{gate}_shape_w{shape_weight:.2f}_retro_w0.00",
                    gate,
                )
                baseline_keys = baseline[["race_id", "pair_key_norm"]].rename(columns={"pair_key_norm": "baseline_pair_key_norm"})

                for retro_weight in retro_weights:
                    if retro_weight == 0:
                        selected = baseline.copy()
                    else:
                        score_col = f"retro_lap_score_s{shape_weight:.2f}_r{retro_weight:.2f}"
                        gated[score_col] = (
                            gated[base_col]
                            + retro_weight * (ncol(gated, "retro_lap_pair_fit_score", 0.5) - 0.5)
                            - 0.55 * retro_weight * ncol(gated, "retro_lap_pair_risk_score", 0.0)
                        )
                        selected = select_top_per_race(
                            gated,
                            score_col,
                            ticket_type,
                            f"{ticket_type}_{gate}_shape_w{shape_weight:.2f}_retro_w{retro_weight:.2f}",
                            gate,
                        )
                    selected = selected.merge(baseline_keys, on="race_id", how="left")
                    selected["changed_from_baseline"] = selected["pair_key_norm"].ne(selected["baseline_pair_key_norm"])
                    selected["shape_weight"] = shape_weight
                    selected["retro_lap_weight"] = retro_weight
                    selections.append(selected)

                    row = metric_row(selected, selected["policy"].iloc[0] if not selected.empty else f"{ticket_type}_{gate}")
                    row["gate"] = gate
                    row["ticket_type"] = ticket_type
                    row["shape_weight"] = shape_weight
                    row["retro_lap_weight"] = retro_weight
                    changed = selected[selected["changed_from_baseline"]].copy()
                    ch = metric_row(changed, "changed_only")
                    row["changed_tickets"] = ch["tickets"]
                    row["changed_roi_pct"] = ch["roi_pct"]
                    row["changed_hit_rate_pct"] = ch["hit_rate_pct"]
                    summary_rows.append(row)

                    for year, gy in selected.groupby("year"):
                        yr = metric_row(gy, selected["policy"].iloc[0])
                        yr["year"] = int(year)
                        yr["gate"] = gate
                        yr["ticket_type"] = ticket_type
                        yr["shape_weight"] = shape_weight
                        yr["retro_lap_weight"] = retro_weight
                        yearly_rows.append(yr)

    detail = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    if not summary.empty:
        baseline = summary[summary["retro_lap_weight"].eq(0.0)][
            ["ticket_type", "gate", "shape_weight", "roi_pct", "hit_rate_pct", "tickets"]
        ].rename(
            columns={
                "roi_pct": "baseline_roi_pct",
                "hit_rate_pct": "baseline_hit_rate_pct",
                "tickets": "baseline_tickets",
            }
        )
        summary = summary.merge(baseline, on=["ticket_type", "gate", "shape_weight"], how="left")
        summary["roi_delta_vs_baseline_pct"] = summary["roi_pct"] - summary["baseline_roi_pct"]
        summary["hit_delta_vs_baseline_pct"] = summary["hit_rate_pct"] - summary["baseline_hit_rate_pct"]
        summary = summary.sort_values(["roi_pct", "tickets"], ascending=[False, False])
    return summary, yearly, detail


def run_segment_backtest(df: pd.DataFrame, gates: list[str], ticket_types: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = df[ncol(df, "retro_lap_pair_evidence_ready", 0.0).gt(0)].copy()
    if valid.empty:
        return pd.DataFrame()

    quantiles = {}
    for col in ["retro_lap_pair_fit_score", "retro_lap_pair_pos_max", "retro_lap_pair_risk_score"]:
        x = pd.to_numeric(valid[col], errors="coerce").dropna()
        if x.empty:
            continue
        quantiles[col] = {
            "q60": float(x.quantile(0.60)),
            "q70": float(x.quantile(0.70)),
            "q80": float(x.quantile(0.80)),
            "q90": float(x.quantile(0.90)),
        }

    segments: dict[str, pd.Series] = {
        "all": pd.Series(True, index=df.index),
        "retro_fit_top30": ncol(df, "retro_lap_pair_fit_score", 0.0).ge(quantiles.get("retro_lap_pair_fit_score", {}).get("q70", 1.0)),
        "retro_fit_top20": ncol(df, "retro_lap_pair_fit_score", 0.0).ge(quantiles.get("retro_lap_pair_fit_score", {}).get("q80", 1.0)),
        "retro_fit_top10": ncol(df, "retro_lap_pair_fit_score", 0.0).ge(quantiles.get("retro_lap_pair_fit_score", {}).get("q90", 1.0)),
        "retro_positive_top20": ncol(df, "retro_lap_pair_pos_max", 0.0).ge(quantiles.get("retro_lap_pair_pos_max", {}).get("q80", 1.0)),
        "retro_low_risk_bottom60": ncol(df, "retro_lap_pair_risk_score", 0.0).le(quantiles.get("retro_lap_pair_risk_score", {}).get("q60", 0.0)),
        "retro_positive_and_low_risk": (
            ncol(df, "retro_lap_pair_pos_max", 0.0).ge(quantiles.get("retro_lap_pair_pos_max", {}).get("q80", 1.0))
            & ncol(df, "retro_lap_pair_risk_score", 0.0).le(quantiles.get("retro_lap_pair_risk_score", {}).get("q60", 0.0))
        ),
    }

    for gate in gates:
        gate_base = df[gate_mask(df, gate)].copy()
        if gate_base.empty:
            continue
        base_col = "segment_base_score"
        gate_base[base_col] = (
            0.15 * gate_base["shape_base_rank_score"]
            + 0.85 * gate_base["shape_pair_fit_score"]
            + 0.06 * gate_base["shape_value_score"]
            - 0.12 * gate_base["shape_pair_risk_score"]
        )
        for seg_name, mask in segments.items():
            seg = gate_base[mask.reindex(gate_base.index).fillna(False)].copy()
            if seg.empty:
                continue
            for ticket_type in ticket_types:
                selected = select_top_per_race(seg, base_col, ticket_type, f"{ticket_type}_{gate}_{seg_name}", gate)
                row = metric_row(selected, selected["policy"].iloc[0] if not selected.empty else f"{ticket_type}_{gate}_{seg_name}")
                row["gate"] = gate
                row["ticket_type"] = ticket_type
                row["segment"] = seg_name
                row["candidate_pairs"] = int(len(seg))
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["roi_pct", "tickets"], ascending=[False, False])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest leak-safe retrospective lap adversity/benefit signals.")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--race-shape", default=str(DEFAULT_RACE_SHAPE))
    parser.add_argument("--feature-csv", action="append", default=[])
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--shape-weights", default="0.70,0.85,1.00")
    parser.add_argument("--retro-lap-weights", default="0,0.04,0.08,0.12,0.18,0.24")
    parser.add_argument("--gates", default="all,value_loose,value_mid,price_sane_strong")
    parser.add_argument("--ticket-types", default="umaren,wide")
    args = parser.parse_args()

    universe_path = project_path(args.universe)
    race_shape_path = project_path(args.race_shape)
    feature_paths = [project_path(p) for p in (args.feature_csv or DEFAULT_FEATURES)]
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_weights = [float(x.strip()) for x in args.shape_weights.split(",") if x.strip()]
    retro_weights = [float(x.strip()) for x in args.retro_lap_weights.split(",") if x.strip()]
    gates = [x.strip() for x in args.gates.split(",") if x.strip()]
    ticket_types = [x.strip() for x in args.ticket_types.split(",") if x.strip()]

    rows = load_runner_rows(feature_paths)
    if rows.empty:
        raise SystemExit("No runner rows loaded.")
    run_scores = add_retro_lap_run_scores(rows)
    prior = add_prior_features(run_scores)

    universe = add_shape_scores(load_universe(universe_path, race_shape_path))
    scored = merge_pair_features(universe, prior)

    summary, yearly, detail = run_overlay_backtest(scored, gates, shape_weights, retro_weights, ticket_types)
    segments = run_segment_backtest(scored, gates, ticket_types)

    runner_cols = [
        "race_id",
        "horse_no",
        "horse_id",
        "race_date",
        "場所",
        "クラス名",
        "芝・ダ",
        "距離",
        "馬場状態",
        "4角.1",
        "着差",
        "上り3F順",
        "target_score",
        "PCI",
        "PCI3",
        "RPCI",
        "Ave-3F",
        "retro_lap_regime",
    ] + RUN_COLS
    run_scores[[c for c in runner_cols if c in run_scores.columns]].to_csv(
        out_dir / "runner_retro_lap_run_scores.csv", index=False, encoding="utf-8-sig"
    )
    prior.to_csv(out_dir / "runner_retro_lap_prior_features.csv", index=False, encoding="utf-8-sig")

    score_cols = [
        "race_id",
        "year",
        "pair_key_norm",
        "anchor_no",
        "anchor_name",
        "partner_no",
        "partner_name",
        "retro_lap_pair_fit_score",
        "retro_lap_pair_risk_score",
        "retro_lap_pair_pos_max",
        "retro_lap_pair_pos_avg",
        "retro_lap_pair_negative_max",
        "retro_lap_pair_overhelped_max",
        "retro_lap_pair_evidence_min",
        "wide_hit",
        "umaren_hit",
        "wide_pay",
        "umaren_pay",
    ]
    scored[[c for c in score_cols if c in scored.columns]].to_csv(
        out_dir / "tickets_with_retro_lap_adversity.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(out_dir / "retro_lap_adversity_overlay_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "retro_lap_adversity_overlay_yearly.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(out_dir / "retro_lap_adversity_overlay_detail.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(out_dir / "retro_lap_adversity_segments.csv", index=False, encoding="utf-8-sig")

    best = summary.head(25).replace({np.nan: None}).to_dict(orient="records") if not summary.empty else []
    best_segments = segments.head(25).replace({np.nan: None}).to_dict(orient="records") if not segments.empty else []
    report = {
        "universe": str(universe_path.relative_to(ROOT)),
        "feature_rows": int(len(rows)),
        "runner_prior_rows": int(len(prior)),
        "ticket_rows": int(len(scored)),
        "races": int(scored["race_id"].nunique()) if not scored.empty else 0,
        "pair_evidence_ready_rate_pct": round(float(scored["retro_lap_pair_evidence_ready"].mean() * 100), 1)
        if "retro_lap_pair_evidence_ready" in scored
        else 0.0,
        "top_policies": best,
        "top_segments": best_segments,
        "note": "Shadow validation only. Each race uses only prior starts per horse; current-race result/lap is not used for that horse's prediction row.",
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# Retro Lap Adversity v1

目的: 過去走で受けたラップ不利/利得を、馬ごとに前走以前へシフトして、馬連・ワイドの相手選びに効くか検証する。

主な信号:
- `retro_lap_front_load_resistance`: 前傾/消耗ラップを前目で受けて内容を残した
- `retro_lap_front_load_excuse`: 前傾/消耗ラップを前目で受け、負けても言い訳がある
- `retro_lap_slow_closer_excuse`: スロー/瞬発寄りで後方から届かなかったが脚は使った
- `retro_lap_long_spurt_resistance`: 長く脚を使うラップで中後方から内容を出した
- `retro_lap_overhelped_score`: 流れに恵まれて好走した可能性
- `retro_lap_negative_score`: 恵まれた条件でも内容が弱かった可能性

リーク対策:
- 各馬のランスコアを日付順に並べ、`shift(1)` してから前走/過去3走/過去5走を作成。
- 検証対象レース自身の結果やラップは、そのレースの予測行には入らない。

出力:
- `runner_retro_lap_run_scores.csv`: 各出走の回顧ラップ不利/利得スコア
- `runner_retro_lap_prior_features.csv`: 予測時点で使える過去走集約
- `tickets_with_retro_lap_adversity.csv`: ペア単位の回顧ラップ特徴
- `retro_lap_adversity_overlay_summary.csv`: 既存ペア選定に加味したROI
- `retro_lap_adversity_segments.csv`: 高スコア層のセグメントROI

注意: 正式BUYは変更していない。採用する場合はシャドー運用で直前オッズ/馬場変化込みの再検証が必要。
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
