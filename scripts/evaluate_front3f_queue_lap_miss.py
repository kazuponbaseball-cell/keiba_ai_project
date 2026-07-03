from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FRONT_FEATURES = ROOT / "outputs/analysis/estimated_front3f_race_quality_v1/test_front3f_feature_light.csv"
DEFAULT_ACTUAL_RUNNERS = ROOT / "data/datasets/cache/lap_pedigree_interactions_confirmed_opponent/test_features.csv"
DEFAULT_CONTINUOUS_PACE = ROOT / "outputs/analysis/continuous_race_pace_prediction_v1/continuous_pace_predictions.csv"
DEFAULT_TICKETS = ROOT / "outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates/recommended_runtime_tickets.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/front3f_queue_lap_miss_v1"

COL_RACE_ID = "レースID(新/馬番無)"
COL_HORSE_NO = "馬番"
COL_HORSE_ID = "血統登録番号"
COL_HORSE_NAME = "馬名"
COL_POPULARITY = "人気"
COL_ODDS = "単勝オッズ"
COL_FINISH = "確定着順"


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def ncol(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = df[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def normalize_race_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def load_front_features(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    df = df.rename(columns={COL_RACE_ID: "race_id", COL_HORSE_NO: "horse_no", COL_HORSE_ID: "horse_id"})
    df["race_id"] = normalize_race_id(df["race_id"])
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    df["horse_id"] = df["horse_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    for col in [
        "horse_est_ten_speed_z_mean_past5",
        "horse_est_ten_speed_z_best_past5",
        "horse_est_ten_speed_goodrun_past5",
        "horse_est_fast_start_rate_past5",
        "horse_est_gap600_mean_past5",
        "horse_est_front3f_confidence_mean_past5",
        "race_est_ten_pressure_score",
        "race_est_fast_start_count",
        "race_est_ten_speed_std",
        "race_est_ten_speed_gap_top2",
        "race_est_queue_clarity_score",
        "ten_speed_pressure_fit_score",
        "ten_speed_solo_front_fit_score",
        "race_quality_front_load_score",
        "race_quality_unstable_ten_score",
        "front_load_retrospective_fit_score",
        "front_load_forward_resilience_score",
        "front_load_fade_risk_current_score",
        COL_POPULARITY,
        COL_ODDS,
    ]:
        if col in df.columns:
            df[col] = ncol(df, col)
    return df


def load_actual_runner_context(path: Path) -> pd.DataFrame:
    wanted = [
        COL_RACE_ID,
        COL_HORSE_NO,
        COL_HORSE_ID,
        COL_HORSE_NAME,
        COL_POPULARITY,
        COL_ODDS,
        COL_FINISH,
        "1角",
        "2角",
        "3角",
        "4角",
        "4角.1",
        "target_score",
        "target_win",
        "target_top3",
        "芝・ダ",
        "距離",
        "クラス名",
        "馬場状態",
    ]
    header = read_csv(path, nrows=0)
    usecols = [c for c in wanted if c in header.columns]
    df = read_csv(path, usecols=usecols)
    df = df.rename(columns={COL_RACE_ID: "race_id", COL_HORSE_NO: "horse_no", COL_HORSE_ID: "horse_id"})
    df["race_id"] = normalize_race_id(df["race_id"])
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    df["horse_id"] = df["horse_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    for col in [COL_POPULARITY, COL_ODDS, COL_FINISH, "1角", "2角", "3角", "4角", "4角.1", "target_score", "target_win", "target_top3", "距離"]:
        if col in df.columns:
            df[col] = ncol(df, col)

    corner_cols = [c for c in ["1角", "2角", "3角", "4角", "4角.1"] if c in df.columns]
    first_call = pd.Series(np.nan, index=df.index, dtype=float)
    for col in corner_cols:
        v = ncol(df, col)
        first_call = first_call.fillna(v.where(v.gt(0)))
    df["actual_first_call_pos"] = first_call
    df["actual_corner4_pos"] = ncol(df, "4角").where(ncol(df, "4角").gt(0)).fillna(ncol(df, "4角.1").where(ncol(df, "4角.1").gt(0)))
    df["actual_front5"] = df["actual_corner4_pos"].le(5)
    df["actual_early_top3"] = df["actual_first_call_pos"].le(3)
    df["actual_early_leader"] = False
    valid_first = df["actual_first_call_pos"].notna()
    if valid_first.any():
        min_pos = df.loc[valid_first].groupby("race_id")["actual_first_call_pos"].transform("min")
        df.loc[valid_first, "actual_early_leader"] = df.loc[valid_first, "actual_first_call_pos"].eq(min_pos)
    return df


def add_queue_prediction_features(front: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    keep_actual = [
        "race_id",
        "horse_no",
        "horse_id",
        COL_HORSE_NAME,
        COL_FINISH,
        COL_POPULARITY,
        COL_ODDS,
        "target_score",
        "target_win",
        "target_top3",
        "actual_first_call_pos",
        "actual_corner4_pos",
        "actual_front5",
        "actual_early_top3",
        "actual_early_leader",
        "芝・ダ",
        "距離",
        "クラス名",
        "馬場状態",
    ]
    out = front.merge(actual[[c for c in keep_actual if c in actual.columns]], on=["race_id", "horse_no", "horse_id"], how="left", suffixes=("", "_actual"))
    if COL_HORSE_NAME not in out.columns and f"{COL_HORSE_NAME}_actual" in out.columns:
        out[COL_HORSE_NAME] = out[f"{COL_HORSE_NAME}_actual"]

    speed = ncol(out, "horse_est_ten_speed_z_mean_past5", 0.0).fillna(0.0)
    best = ncol(out, "horse_est_ten_speed_z_best_past5", 0.0).fillna(0.0)
    good = ncol(out, "horse_est_ten_speed_goodrun_past5", 0.0).fillna(0.0)
    rate = ncol(out, "horse_est_fast_start_rate_past5", 0.0).fillna(0.0)
    conf = ncol(out, "horse_est_front3f_confidence_mean_past5", 0.0).fillna(0.0).clip(0.0, 1.0)
    out["pred_ten_queue_score"] = (0.54 * speed + 0.22 * best + 0.14 * good + 0.10 * rate) * (0.65 + 0.35 * conf)
    out["pred_ten_rank"] = out.groupby("race_id")["pred_ten_queue_score"].rank(ascending=False, method="first")
    out["pred_ten_pct_rank"] = out.groupby("race_id")["pred_ten_queue_score"].rank(pct=True, ascending=True)
    out["pred_front1"] = out["pred_ten_rank"].eq(1)
    out["pred_front3"] = out["pred_ten_rank"].le(3)
    out["pred_front5"] = out["pred_ten_rank"].le(5)

    clarity = ncol(out, "race_est_queue_clarity_score", 0.0).fillna(0.0).clip(0.0, 1.0)
    fast_count = ncol(out, "race_est_fast_start_count", 0.0).fillna(0.0)
    pressure = ncol(out, "race_est_ten_pressure_score", 0.0).fillna(0.0)
    gap = ncol(out, "race_est_ten_speed_gap_top2", 0.0).fillna(0.0)
    out["queue_type"] = np.select(
        [
            clarity.ge(0.55) & gap.ge(0.35) & fast_count.le(6),
            clarity.le(0.15) | fast_count.ge(10) | pressure.ge(1.0),
        ],
        ["single_leader_clear", "front_duel_dense"],
        default="mixed_queue",
    )
    return out


def load_continuous_pace(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["race_id"])
    df = read_csv(path)
    if "race_id" not in df.columns:
        return pd.DataFrame(columns=["race_id"])
    df["race_id"] = normalize_race_id(df["race_id"])
    for col in [
        "pred_front3f_sec",
        "front3f_sec",
        "pred_rpci",
        "rpci",
        "pred_pci3",
        "pci3",
        "cont_confidence",
        "cont_margin",
        "v2_confidence",
        "v2_margin",
    ]:
        if col in df.columns:
            df[col] = ncol(df, col)
    if {"pred_front3f_sec", "front3f_sec"}.issubset(df.columns):
        df["front3f_error_sec"] = df["pred_front3f_sec"] - df["front3f_sec"]
        df["front3f_abs_error_sec"] = df["front3f_error_sec"].abs()
    if {"pred_rpci", "rpci"}.issubset(df.columns):
        df["rpci_abs_error"] = (df["pred_rpci"] - df["rpci"]).abs()
    if {"pred_pci3", "pci3"}.issubset(df.columns):
        df["pci3_abs_error"] = (df["pred_pci3"] - df["pci3"]).abs()
    if {"cont_predicted_lap_mode", "actual_lap_mode"}.issubset(df.columns):
        df["cont_lap_mode_hit"] = df["cont_predicted_lap_mode"].astype(str).eq(df["actual_lap_mode"].astype(str))
    elif {"v2_predicted_lap_mode", "actual_lap_mode"}.issubset(df.columns):
        df["cont_lap_mode_hit"] = df["v2_predicted_lap_mode"].astype(str).eq(df["actual_lap_mode"].astype(str))
    else:
        df["cont_lap_mode_hit"] = np.nan
    keep = [
        "race_id",
        "actual_lap_mode",
        "cont_predicted_lap_mode",
        "v2_predicted_lap_mode",
        "cont_lap_mode_hit",
        "pred_front3f_sec",
        "front3f_sec",
        "front3f_error_sec",
        "front3f_abs_error_sec",
        "pred_rpci",
        "rpci",
        "rpci_abs_error",
        "pred_pci3",
        "pci3",
        "pci3_abs_error",
        "cont_confidence",
        "cont_margin",
        "v2_confidence",
        "v2_margin",
        "source",
        "class_tier",
        "surface",
        "distance_m",
    ]
    return df[[c for c in keep if c in df.columns]].drop_duplicates("race_id", keep="last")


def max_drawdown(net: pd.Series) -> float:
    curve = net.cumsum()
    dd = curve.cummax() - curve
    return float(dd.max()) if len(dd) else 0.0


def roi_without_top_returns(frame: pd.DataFrame, n: int) -> float:
    if frame.empty:
        return float("nan")
    stake = ncol(frame, "stake_yen", 0.0).fillna(0.0)
    ret = ncol(frame, "return_yen", 0.0).fillna(0.0)
    if n > 0 and len(ret):
        drop_idx = ret.sort_values(ascending=False).head(n).index
        stake = stake.drop(index=drop_idx)
        ret = ret.drop(index=drop_idx)
    return float(ret.sum() / stake.sum() * 100.0) if stake.sum() > 0 else float("nan")


def ticket_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    stake = ncol(frame, "stake_yen", 0.0).fillna(0.0)
    ret = ncol(frame, "return_yen", 0.0).fillna(0.0)
    hit = frame.get("hit", pd.Series(False, index=frame.index)).astype(bool)
    return {
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame.columns else int(len(frame)),
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(ret.sum() - stake.sum()),
        "roi": float(ret.sum() / stake.sum() * 100.0) if stake.sum() > 0 else float("nan"),
        "hit_rate": float(hit.mean() * 100.0) if len(hit) else float("nan"),
        "top1_removed_roi": roi_without_top_returns(frame, 1),
        "top3_removed_roi": roi_without_top_returns(frame, 3),
        "top5_removed_roi": roi_without_top_returns(frame, 5),
        "max_drawdown_yen": max_drawdown(ret - stake),
    }


def load_tickets(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if "policy" not in df.columns:
        if "mcs_pbo_policy" in df.columns:
            df["policy"] = df["mcs_pbo_policy"].fillna(path.stem).astype(str)
        elif "operation_profile" in df.columns:
            df["policy"] = df["operation_profile"].fillna(path.stem).astype(str)
        else:
            df["policy"] = path.stem
    if "ticket_type" not in df.columns:
        df["ticket_type"] = "unknown"
    for col in ["race_id", "policy", "ticket_type"]:
        df[col] = df[col].astype(str)
    df["race_id"] = normalize_race_id(df["race_id"])
    if "horse_a" not in df.columns and {"anchor_no", "partner_no"}.issubset(df.columns):
        a = ncol(df, "anchor_no")
        b = ncol(df, "partner_no")
        df["horse_a"] = np.minimum(a, b)
        df["horse_b"] = np.maximum(a, b)
    df["horse_a"] = pd.to_numeric(df["horse_a"], errors="coerce").astype("Int64")
    df["horse_b"] = pd.to_numeric(df["horse_b"], errors="coerce").astype("Int64")
    return df


def enrich_tickets_with_queue(tickets: pd.DataFrame, runners: pd.DataFrame, pace: pd.DataFrame) -> pd.DataFrame:
    side_cols = [
        "race_id",
        "horse_no",
        "horse_id",
        COL_HORSE_NAME,
        "pred_ten_queue_score",
        "pred_ten_rank",
        "pred_ten_pct_rank",
        "pred_front1",
        "pred_front3",
        "pred_front5",
        "actual_first_call_pos",
        "actual_corner4_pos",
        "actual_front5",
        "actual_early_leader",
        "target_score",
        "target_top3",
    ]
    base = runners[[c for c in side_cols if c in runners.columns]].copy()
    out = tickets.copy()
    for side, horse_col in [("a", "horse_a"), ("b", "horse_b")]:
        renamed = base.rename(columns={c: f"{side}_{c}" for c in base.columns if c not in {"race_id", "horse_no"}})
        out = out.merge(renamed, left_on=["race_id", horse_col], right_on=["race_id", "horse_no"], how="left")
        out = out.drop(columns=["horse_no"], errors="ignore")

    race_cols = [
        "race_id",
        "race_est_ten_pressure_score",
        "race_est_fast_start_count",
        "race_est_ten_speed_gap_top2",
        "race_est_queue_clarity_score",
        "queue_type",
    ]
    race_queue = runners[[c for c in race_cols if c in runners.columns]].drop_duplicates("race_id")
    out = out.merge(race_queue, on="race_id", how="left")
    out = out.merge(pace, on="race_id", how="left", suffixes=("", "_pace"))

    out["pair_pred_front5_any"] = out.get("a_pred_front5", False).fillna(False).astype(bool) | out.get("b_pred_front5", False).fillna(False).astype(bool)
    out["pair_pred_front5_both"] = out.get("a_pred_front5", False).fillna(False).astype(bool) & out.get("b_pred_front5", False).fillna(False).astype(bool)
    out["pair_pred_leader_any"] = out.get("a_pred_front1", False).fillna(False).astype(bool) | out.get("b_pred_front1", False).fillna(False).astype(bool)
    out["pair_actual_front5_any"] = out.get("a_actual_front5", False).fillna(False).astype(bool) | out.get("b_actual_front5", False).fillna(False).astype(bool)
    out["pair_actual_leader_any"] = out.get("a_actual_early_leader", False).fillna(False).astype(bool) | out.get("b_actual_early_leader", False).fillna(False).astype(bool)
    out["pair_pred_front_complement"] = out["pair_pred_front5_any"] & ~out["pair_pred_front5_both"]
    out["pair_front_prediction_hit"] = out["pair_pred_front5_any"].eq(out["pair_actual_front5_any"])
    out["race_front_read_good"] = ncol(out, "front3f_abs_error_sec", np.nan).le(0.55)
    out["race_rpci_read_good"] = ncol(out, "rpci_abs_error", np.nan).le(2.0)
    out["race_lap_read_good"] = out.get("cont_lap_mode_hit", pd.Series(False, index=out.index)).fillna(False).astype(bool) | (
        out["race_front_read_good"] & out["race_rpci_read_good"]
    )
    out["miss_decomposition"] = np.select(
        [
            out.get("hit", pd.Series(False, index=out.index)).astype(bool),
            out["race_lap_read_good"] & out["pair_actual_front5_any"],
            out["race_lap_read_good"] & ~out["pair_actual_front5_any"],
            ~out["race_lap_read_good"] & out["pair_actual_front5_any"],
        ],
        [
            "hit",
            "race_read_ok_pair_position_ok_but_missed",
            "race_read_ok_pair_position_wrong",
            "race_read_wrong_but_position_ok",
        ],
        default="race_read_wrong_and_position_wrong",
    )
    return out


def runner_queue_metrics(runners: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    race_rows: list[dict[str, Any]] = []
    for race_id, group in runners.groupby("race_id", sort=False):
        valid = group.dropna(subset=["actual_first_call_pos"]).copy()
        if valid.empty:
            continue
        pred_top1 = valid.loc[valid["pred_ten_rank"].idxmin()]
        actual_leaders = valid[valid["actual_early_leader"].fillna(False)]
        actual_front5 = valid[valid["actual_front5"].fillna(False)]
        pred_front3 = set(valid.loc[valid["pred_front3"], "horse_no"].astype(int).tolist())
        pred_front5 = set(valid.loc[valid["pred_front5"], "horse_no"].astype(int).tolist())
        leader_set = set(actual_leaders["horse_no"].astype(int).tolist())
        front5_set = set(actual_front5["horse_no"].astype(int).tolist())
        race_rows.append(
            {
                "race_id": race_id,
                "queue_type": str(valid["queue_type"].iloc[0]),
                "race_est_queue_clarity_score": float(ncol(valid, "race_est_queue_clarity_score").iloc[0]),
                "race_est_fast_start_count": float(ncol(valid, "race_est_fast_start_count").iloc[0]),
                "race_est_ten_pressure_score": float(ncol(valid, "race_est_ten_pressure_score").iloc[0]),
                "actual_leader_count": int(len(leader_set)),
                "actual_front5_count": int(len(front5_set)),
                "pred_top1_is_actual_leader": int(int(pred_top1["horse_no"]) in leader_set) if leader_set else np.nan,
                "pred_top3_contains_leader": int(bool(pred_front3 & leader_set)) if leader_set else np.nan,
                "pred_top5_contains_leader": int(bool(pred_front5 & leader_set)) if leader_set else np.nan,
                "pred_top5_front5_precision": len(pred_front5 & front5_set) / len(pred_front5) if pred_front5 else np.nan,
                "pred_top5_front5_recall": len(pred_front5 & front5_set) / len(front5_set) if front5_set else np.nan,
            }
        )
    race_metrics = pd.DataFrame(race_rows)
    by_queue = (
        race_metrics.groupby("queue_type", dropna=False)
        .agg(
            races=("race_id", "count"),
            top1_leader_hit=("pred_top1_is_actual_leader", "mean"),
            top3_contains_leader=("pred_top3_contains_leader", "mean"),
            top5_contains_leader=("pred_top5_contains_leader", "mean"),
            top5_front5_precision=("pred_top5_front5_precision", "mean"),
            top5_front5_recall=("pred_top5_front5_recall", "mean"),
            avg_clarity=("race_est_queue_clarity_score", "mean"),
            avg_fast_count=("race_est_fast_start_count", "mean"),
            avg_pressure=("race_est_ten_pressure_score", "mean"),
        )
        .reset_index()
    )
    return race_metrics, by_queue


def collect_ticket_segments(tickets: pd.DataFrame, min_tickets: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(name: str, mask: pd.Series, extra: dict[str, Any] | None = None) -> None:
        sub = tickets.loc[mask.fillna(False)].copy()
        if len(sub) < min_tickets:
            return
        row = {"segment": name, **ticket_metrics(sub)}
        if extra:
            row.update(extra)
        rows.append(row)

    add("base_all", pd.Series(True, index=tickets.index))
    for ticket_type, group in tickets.groupby("ticket_type", dropna=False):
        base = tickets.index.isin(group.index)
        add(f"base::{ticket_type}", pd.Series(base, index=tickets.index), {"ticket_type": ticket_type})
        for queue_type in sorted(tickets["queue_type"].dropna().astype(str).unique()):
            add(
                f"queue_type={queue_type}::{ticket_type}",
                pd.Series(base, index=tickets.index) & tickets["queue_type"].astype(str).eq(queue_type),
                {"ticket_type": ticket_type, "queue_type": queue_type},
            )
        checks = {
            "pair_pred_front5_any": tickets["pair_pred_front5_any"],
            "pair_pred_front5_both": tickets["pair_pred_front5_both"],
            "pair_pred_leader_any": tickets["pair_pred_leader_any"],
            "pair_front_complement": tickets["pair_pred_front_complement"],
            "race_lap_read_good": tickets["race_lap_read_good"],
            "race_lap_read_bad": ~tickets["race_lap_read_good"],
            "race_read_good_pair_front_any": tickets["race_lap_read_good"] & tickets["pair_pred_front5_any"],
            "race_read_good_pair_complement": tickets["race_lap_read_good"] & tickets["pair_pred_front_complement"],
            "dense_pair_front_any": tickets["queue_type"].astype(str).eq("front_duel_dense") & tickets["pair_pred_front5_any"],
            "clear_pair_leader_any": tickets["queue_type"].astype(str).eq("single_leader_clear") & tickets["pair_pred_leader_any"],
        }
        for name, mask in checks.items():
            add(f"{name}::{ticket_type}", pd.Series(base, index=tickets.index) & mask, {"ticket_type": ticket_type, "rule": name})
    for reason, group in tickets.groupby("miss_decomposition", dropna=False):
        add(f"miss::{reason}", tickets["miss_decomposition"].astype(str).eq(str(reason)), {"miss_decomposition": str(reason)})
    return pd.DataFrame(rows).sort_values(["roi", "tickets"], ascending=[False, False]) if rows else pd.DataFrame()


def write_readme(out_dir: Path, queue_by_type: pd.DataFrame, ticket_segments: pd.DataFrame) -> None:
    lines = [
        "# Front3F Queue And Lap Miss Diagnostics",
        "",
        "Purpose: test whether estimated front-3F queue order improves race-read and pair-ticket ROI.",
        "",
        "## Queue Prediction By Type",
        "| queue type | races | top1 leader hit | top3 contains leader | top5 precision | top5 recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in queue_by_type.iterrows():
        lines.append(
            f"| {r['queue_type']} | {int(r['races'])} | {r['top1_leader_hit']*100:.1f}% | "
            f"{r['top3_contains_leader']*100:.1f}% | {r['top5_front5_precision']*100:.1f}% | {r['top5_front5_recall']*100:.1f}% |"
        )
    lines += [
        "",
        "## Top Ticket Segments",
        "| segment | tickets | races | ROI | hit | top3 removed ROI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in ticket_segments.head(20).iterrows():
        lines.append(
            f"| {r['segment']} | {int(r['tickets'])} | {int(r['races'])} | {r['roi']:.1f}% | "
            f"{r['hit_rate']:.1f}% | {r['top3_removed_roi']:.1f}% |"
        )
    lines += [
        "",
        "## Interpretation Guide",
        "- If race_lap_read_good loses but pair position was wrong, pair selection is the bottleneck.",
        "- If race_lap_read_bad loses, race-quality prediction is the bottleneck.",
        "- Queue features should remain shadow unless they improve OOS ROI without relying on a few top payouts.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate front-3F queue prediction and lap miss decomposition.")
    parser.add_argument("--front-features", type=Path, default=DEFAULT_FRONT_FEATURES)
    parser.add_argument("--actual-runners", type=Path, default=DEFAULT_ACTUAL_RUNNERS)
    parser.add_argument("--continuous-pace", type=Path, default=DEFAULT_CONTINUOUS_PACE)
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-tickets", type=int, default=10)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    front = load_front_features(args.front_features)
    actual = load_actual_runner_context(args.actual_runners)
    runners = add_queue_prediction_features(front, actual)
    pace = load_continuous_pace(args.continuous_pace)
    tickets = load_tickets(args.tickets)
    enriched_tickets = enrich_tickets_with_queue(tickets, runners, pace)

    race_queue_metrics, queue_by_type = runner_queue_metrics(runners)
    ticket_segments = collect_ticket_segments(enriched_tickets, args.min_tickets)

    runner_path = out_dir / "front3f_queue_runner_features.csv"
    race_path = out_dir / "front3f_queue_race_prediction_metrics.csv"
    queue_path = out_dir / "front3f_queue_by_type.csv"
    ticket_path = out_dir / "front3f_queue_ticket_enriched.csv"
    segment_path = out_dir / "front3f_queue_ticket_segments.csv"

    runners.to_csv(runner_path, index=False, encoding="utf-8-sig")
    race_queue_metrics.to_csv(race_path, index=False, encoding="utf-8-sig")
    queue_by_type.to_csv(queue_path, index=False, encoding="utf-8-sig")
    enriched_tickets.to_csv(ticket_path, index=False, encoding="utf-8-sig")
    ticket_segments.to_csv(segment_path, index=False, encoding="utf-8-sig")
    write_readme(out_dir, queue_by_type, ticket_segments)

    summary = {
        "front_features": str(args.front_features),
        "tickets": str(args.tickets),
        "runners": int(len(runners)),
        "races": int(runners["race_id"].nunique()),
        "tickets_rows": int(len(enriched_tickets)),
        "ticket_races": int(enriched_tickets["race_id"].nunique()),
        "output_files": {
            "runners": str(runner_path),
            "race_queue_metrics": str(race_path),
            "queue_by_type": str(queue_path),
            "tickets": str(ticket_path),
            "segments": str(segment_path),
        },
        "queue_by_type": queue_by_type.to_dict("records"),
        "top_ticket_segments": ticket_segments.head(20).to_dict("records") if not ticket_segments.empty else [],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
