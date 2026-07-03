from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNERS = ROOT / "outputs/analysis/front3f_queue_lap_miss_lap_expansion_v1/front3f_queue_runner_features.csv"
DEFAULT_TICKETS = ROOT / "outputs/analysis/front3f_queue_lap_miss_lap_expansion_v1/front3f_queue_ticket_enriched.csv"
DEFAULT_PACE = ROOT / "outputs/analysis/continuous_race_pace_prediction_v1/continuous_pace_predictions.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/high_pressure_front_survival_context_v1"


def read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, usecols=usecols)


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def text(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[col].fillna(default).astype(str)


def norm_race_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(16)


def distance_bin(distance: pd.Series) -> pd.Series:
    d = pd.to_numeric(distance, errors="coerce")
    return pd.Series(
        np.select(
            [d <= 1200, d <= 1400, d <= 1600, d <= 1800, d <= 2000, d <= 2400, d > 2400],
            ["<=1200", "1201-1400", "1401-1600", "1601-1800", "1801-2000", "2001-2400", "2401+"],
            default="unknown",
        ),
        index=distance.index,
    )


def truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    s = series.fillna(False)
    if s.dtype == object:
        return s.astype(str).str.lower().isin(["true", "1", "yes"])
    return pd.to_numeric(s, errors="coerce").fillna(0).ne(0)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def metrics(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": np.nan,
            "hit_rate_pct": np.nan,
            "top_return_share_pct": np.nan,
            "roi_ex_top1_pct": np.nan,
            "roi_ex_top5_pct": np.nan,
        }
    stake = pd.to_numeric(frame["stake_yen"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(frame["return_yen"], errors="coerce").fillna(0.0)
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    pos = ret.sort_values(ascending=False)
    top_share = float(pos.iloc[0] / ret_sum) if ret_sum > 0 and len(pos) else np.nan
    top1_stake = float(stake.loc[pos.index[:1]].sum()) if len(pos) else 0.0
    top1_ret = float(pos.iloc[:1].sum()) if len(pos) else 0.0
    top5_stake = float(stake.loc[pos.index[:5]].sum()) if len(pos) else 0.0
    top5_ret = float(pos.iloc[:5].sum()) if len(pos) else 0.0
    return {
        "policy": label,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": stake_sum,
        "return_yen": ret_sum,
        "profit_yen": ret_sum - stake_sum,
        "roi_pct": safe_div(ret_sum, stake_sum) * 100,
        "hit_rate_pct": float(ret.gt(0).mean() * 100),
        "top_return_share_pct": top_share * 100 if np.isfinite(top_share) else np.nan,
        "roi_ex_top1_pct": safe_div(ret_sum - top1_ret, stake_sum - top1_stake) * 100,
        "roi_ex_top5_pct": safe_div(ret_sum - top5_ret, stake_sum - top5_stake) * 100,
    }


def build_race_table(runners: pd.DataFrame, pace: pd.DataFrame) -> pd.DataFrame:
    r = runners.copy()
    r["race_id"] = norm_race_id(r["race_id"])
    r["year"] = r["race_id"].str.slice(0, 4).astype(int)
    r["race_date"] = r["race_id"].str.slice(0, 8).astype(int)
    r["actual_front5_bool"] = truthy(r.get("actual_front5", pd.Series(False, index=r.index)))
    r["target_top3_bool"] = truthy(r.get("target_top3", pd.Series(False, index=r.index)))
    r["target_win_bool"] = truthy(r.get("target_win", pd.Series(False, index=r.index)))

    rows: list[dict[str, Any]] = []
    for race_id, group in r.groupby("race_id", sort=False):
        first = group.iloc[0]
        top3_front = int((group["actual_front5_bool"] & group["target_top3_bool"]).sum())
        winner_front = bool((group["actual_front5_bool"] & group["target_win_bool"]).any())
        rows.append(
            {
                "race_id": race_id,
                "year": int(first["year"]),
                "race_date": int(first["race_date"]),
                "surface": str(first.get("芝・ダ", "")),
                "distance_m": float(pd.to_numeric(first.get("距離"), errors="coerce")),
                "class_tier": str(first.get("クラス名", "")),
                "going": str(first.get("馬場状態", "")),
                "queue_type": str(first.get("queue_type", "")),
                "race_est_ten_pressure_score": float(pd.to_numeric(first.get("race_est_ten_pressure_score"), errors="coerce")),
                "race_est_fast_start_count": float(pd.to_numeric(first.get("race_est_fast_start_count"), errors="coerce")),
                "race_est_ten_speed_gap_top2": float(pd.to_numeric(first.get("race_est_ten_speed_gap_top2"), errors="coerce")),
                "race_est_queue_clarity_score": float(pd.to_numeric(first.get("race_est_queue_clarity_score"), errors="coerce")),
                "top3_front5_count": top3_front,
                "winner_front5": int(winner_front),
                "actual_front_survival": int(winner_front or top3_front >= 2),
                "actual_front_collapse": int(top3_front == 0),
                "actual_top3_front5_share": top3_front / 3.0,
            }
        )
    races = pd.DataFrame(rows)

    p = pace.copy()
    p["race_id"] = norm_race_id(p["race_id"])
    keep = [
        "race_id",
        "venue_code",
        "race_no",
        "actual_lap_mode",
        "cont_predicted_lap_mode",
        "cont_confidence",
        "cont_margin",
        "front3f_sec",
        "last3f_sec",
        "rpci",
        "pci3",
        "course_front3f_prior_sec",
        "course_front3f_prior_std",
        "course_front3f_prior_count",
        "pred_front3f_sec",
        "pred_rpci",
        "pred_pci3",
        "cont_front_delta_z",
        "cont_rpci_delta",
    ]
    races = races.merge(p[[c for c in keep if c in p.columns]], on="race_id", how="left")
    races["distance_bin"] = distance_bin(races["distance_m"])
    races["venue_code"] = races.get("venue_code", pd.Series("", index=races.index)).fillna("").astype(str)

    pressure = pd.to_numeric(races["race_est_ten_pressure_score"], errors="coerce").fillna(0.0).clip(0, 1)
    fast_count = pd.to_numeric(races["race_est_fast_start_count"], errors="coerce").fillna(0.0)
    duel = 1.0 - pd.to_numeric(races["race_est_queue_clarity_score"], errors="coerce").fillna(0.0).clip(0, 1)
    pred_fast = races["cont_predicted_lap_mode"].fillna("").astype(str).eq("fast").astype(float)
    races["pre_front_load_signal"] = (0.48 * pressure + 0.20 * (fast_count / 5.0).clip(0, 1) + 0.20 * duel + 0.12 * pred_fast).clip(0, 1)
    races["actual_fast_or_frontload"] = (
        races["actual_lap_mode"].fillna("").astype(str).eq("fast")
        | pd.to_numeric(races["rpci"], errors="coerce").lt(48.0)
        | pd.to_numeric(races["front3f_sec"], errors="coerce").lt(pd.to_numeric(races["course_front3f_prior_sec"], errors="coerce") - 0.25)
    )
    return races.sort_values(["race_date", "venue_code", "race_no", "race_id"], kind="mergesort").reset_index(drop=True)


def rolling_context_priors(races: pd.DataFrame) -> pd.DataFrame:
    out = races.copy()
    out["global_survival_prior"] = out["actual_front_survival"].shift().expanding(min_periods=30).mean()
    out["global_collapse_prior"] = out["actual_front_collapse"].shift().expanding(min_periods=30).mean()
    specs = [
        (["venue_code", "surface", "distance_bin", "class_tier", "going"], "full", 8),
        (["venue_code", "surface", "distance_bin", "class_tier"], "course_class", 10),
        (["venue_code", "surface", "distance_bin"], "course", 12),
        (["venue_code", "surface"], "venue_surface", 20),
        (["surface", "distance_bin"], "surface_distance", 25),
    ]
    for keys, prefix, min_periods in specs:
        group = out.groupby(keys, dropna=False, sort=False)
        out[f"{prefix}_survival_prior"] = group["actual_front_survival"].transform(
            lambda s: s.shift().expanding(min_periods=min_periods).mean()
        )
        out[f"{prefix}_collapse_prior"] = group["actual_front_collapse"].transform(
            lambda s: s.shift().expanding(min_periods=min_periods).mean()
        )
        out[f"{prefix}_support_count"] = group.cumcount()

    for target in ["survival", "collapse"]:
        out[f"context_{target}_prior"] = out[f"full_{target}_prior"]
        out["context_support_count"] = out["full_support_count"]
        for prefix in ["course_class", "course", "venue_surface", "surface_distance"]:
            missing = out[f"context_{target}_prior"].isna()
            out.loc[missing, f"context_{target}_prior"] = out.loc[missing, f"{prefix}_{target}_prior"]
            out.loc[missing, "context_support_count"] = out.loc[missing, f"{prefix}_support_count"]
        out[f"context_{target}_prior"] = out[f"context_{target}_prior"].fillna(out[f"global_{target}_prior"])
    out["context_survival_prior"] = out["context_survival_prior"].fillna(out["actual_front_survival"].expanding().mean().shift().fillna(out["actual_front_survival"].mean()))
    out["context_collapse_prior"] = out["context_collapse_prior"].fillna(out["actual_front_collapse"].expanding().mean().shift().fillna(out["actual_front_collapse"].mean()))
    out["front_survival_despite_pressure_score"] = (
        out["pre_front_load_signal"] * (0.78 * out["context_survival_prior"] + 0.22 * (1 - out["context_collapse_prior"]))
    ).clip(0, 1)
    out["front_collapse_warning_score"] = (
        out["pre_front_load_signal"] * (0.72 * out["context_collapse_prior"] + 0.28 * (1 - out["context_survival_prior"]))
    ).clip(0, 1)
    out["front_context_readability_score"] = (
        (out["context_survival_prior"] - out["context_collapse_prior"]).abs()
        * (pd.to_numeric(out["context_support_count"], errors="coerce").fillna(0).clip(upper=50) / 50.0)
    ).clip(0, 1)
    return out


def segment_summary(races: pd.DataFrame) -> pd.DataFrame:
    rows = []
    segment_sets = [
        ["venue_code"],
        ["surface"],
        ["distance_bin"],
        ["class_tier"],
        ["going"],
        ["queue_type"],
        ["venue_code", "surface"],
        ["venue_code", "surface", "distance_bin"],
        ["venue_code", "surface", "distance_bin", "class_tier"],
        ["surface", "distance_bin", "class_tier"],
        ["queue_type", "surface", "distance_bin"],
    ]
    high = races[races["pre_front_load_signal"].ge(races["pre_front_load_signal"].quantile(0.60))].copy()
    for keys in segment_sets:
        for values, group in high.groupby(keys, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            if len(group) < 20:
                continue
            row = {
                "segment": " x ".join(keys),
                "value": " x ".join(str(v) for v in values),
                "races": int(len(group)),
                "front_survival_rate_pct": float(group["actual_front_survival"].mean() * 100),
                "front_collapse_rate_pct": float(group["actual_front_collapse"].mean() * 100),
                "winner_front5_rate_pct": float(group["winner_front5"].mean() * 100),
                "avg_top3_front5_share_pct": float(group["actual_top3_front5_share"].mean() * 100),
                "avg_pre_front_load_signal": float(group["pre_front_load_signal"].mean()),
                "actual_fast_or_frontload_rate_pct": float(group["actual_fast_or_frontload"].mean() * 100),
                "avg_context_survival_prior": float(group["context_survival_prior"].mean()),
                "avg_context_collapse_prior": float(group["context_collapse_prior"].mean()),
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["front_survival_rate_pct", "races"], ascending=[False, False])


def build_ticket_context(tickets: pd.DataFrame, races: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    t = tickets.copy()
    t["race_id"] = norm_race_id(t["race_id"])
    merge_cols = [
        "race_id",
        "context_survival_prior",
        "context_collapse_prior",
        "front_survival_despite_pressure_score",
        "front_collapse_warning_score",
        "front_context_readability_score",
        "pre_front_load_signal",
        "actual_front_survival",
        "actual_front_collapse",
    ]
    t = t.merge(races[merge_cols], on="race_id", how="left")
    t["ticket_type"] = text(t, "ticket_type")
    t["year"] = pd.to_numeric(t.get("year"), errors="coerce").fillna(t["race_id"].str.slice(0, 4).astype(int)).astype(int)
    t["stake_yen"] = pd.to_numeric(t.get("stake_yen"), errors="coerce").fillna(0.0)
    t["return_yen"] = pd.to_numeric(t.get("return_yen"), errors="coerce").fillna(0.0)

    train = t[t["year"] < 2026].copy()
    thresholds = {
        "pressure_q60": float(pd.to_numeric(train["pre_front_load_signal"], errors="coerce").quantile(0.60)),
        "pressure_q70": float(pd.to_numeric(train["pre_front_load_signal"], errors="coerce").quantile(0.70)),
        "survival_q60": float(pd.to_numeric(train["context_survival_prior"], errors="coerce").quantile(0.60)),
        "survival_q70": float(pd.to_numeric(train["context_survival_prior"], errors="coerce").quantile(0.70)),
        "collapse_q70": float(pd.to_numeric(train["context_collapse_prior"], errors="coerce").quantile(0.70)),
        "collapse_q80": float(pd.to_numeric(train["context_collapse_prior"], errors="coerce").quantile(0.80)),
        "readability_q50": float(pd.to_numeric(train["front_context_readability_score"], errors="coerce").quantile(0.50)),
    }
    return t, thresholds


def make_policy_masks(t: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, pd.Series]:
    front_any = truthy(t.get("pair_pred_front5_any", pd.Series(False, index=t.index)))
    front_both = truthy(t.get("pair_pred_front5_both", pd.Series(False, index=t.index)))
    leader_any = truthy(t.get("pair_pred_leader_any", pd.Series(False, index=t.index)))
    complement = truthy(t.get("pair_pred_front_complement", pd.Series(False, index=t.index)))
    pressure60 = t["pre_front_load_signal"].ge(thresholds["pressure_q60"])
    pressure70 = t["pre_front_load_signal"].ge(thresholds["pressure_q70"])
    survival60 = t["context_survival_prior"].ge(thresholds["survival_q60"])
    survival70 = t["context_survival_prior"].ge(thresholds["survival_q70"])
    collapse70 = t["context_collapse_prior"].ge(thresholds["collapse_q70"])
    collapse80 = t["context_collapse_prior"].ge(thresholds["collapse_q80"])
    readable = t["front_context_readability_score"].ge(thresholds["readability_q50"])
    queue = text(t, "queue_type").fillna(text(t, "queue_shape_label"))

    policies = {
        "base_all": pd.Series(True, index=t.index),
        "pressure60_pair_front_any": pressure60 & front_any,
        "pressure60_pair_front_both": pressure60 & front_both,
        "pressure70_pair_front_any": pressure70 & front_any,
        "pressure60_survival70_pair_front_any": pressure60 & survival70 & front_any,
        "pressure60_survival70_pair_front_both": pressure60 & survival70 & front_both,
        "pressure60_readable_survival_pair_front_any": pressure60 & readable & survival60 & front_any,
        "pressure60_avoid_collapse80_front_any": pressure60 & ~collapse80 & front_any,
        "pressure60_front_complement": pressure60 & complement,
        "pressure60_leader_any": pressure60 & leader_any,
        "front_duel_dense_front_any": queue.eq("front_duel_dense") & front_any,
        "mixed_queue_front_any": queue.eq("mixed_queue") & front_any,
        "avoid_bad_pressure_context": ~((pressure60 & collapse70) & ~survival60),
    }
    return policies


def ticket_policy_metrics(tickets: pd.DataFrame, races: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    t, thresholds = build_ticket_context(tickets, races)
    policies = make_policy_masks(t, thresholds)
    rows = []
    by_year = []
    for ticket_type in ["wide", "umaren"]:
        type_mask = t["ticket_type"].eq(ticket_type)
        for name, mask in policies.items():
            sub = t[type_mask & mask.fillna(False)].copy()
            row = metrics(sub, f"{ticket_type}::{name}")
            rows.append(row)
            for year, group in sub.groupby("year", dropna=False):
                yr = metrics(group, f"{ticket_type}::{name}")
                yr["year"] = int(year)
                by_year.append(yr)
    return pd.DataFrame(rows), pd.DataFrame(by_year), thresholds


def robust_notes(metrics_by_year: pd.DataFrame) -> pd.DataFrame:
    if metrics_by_year.empty:
        return pd.DataFrame()
    rows = []
    for policy, group in metrics_by_year.groupby("policy"):
        y2026 = group[group["year"].eq(2026)]
        train = group[group["year"].lt(2026)]
        row = {
            "policy": policy,
            "train_years": ",".join(str(int(y)) for y in sorted(train["year"].unique())),
            "train_tickets": int(train["tickets"].sum()) if not train.empty else 0,
            "train_roi_pct": safe_div(float(train["return_yen"].sum()), float(train["stake_yen"].sum())) * 100 if not train.empty else np.nan,
            "oos_2026_tickets": int(y2026["tickets"].sum()) if not y2026.empty else 0,
            "oos_2026_roi_pct": safe_div(float(y2026["return_yen"].sum()), float(y2026["stake_yen"].sum())) * 100 if not y2026.empty else np.nan,
            "oos_2026_roi_ex_top1_pct": float(y2026["roi_ex_top1_pct"].iloc[0]) if len(y2026) == 1 else np.nan,
            "status": "shadow_only",
        }
        if row["oos_2026_tickets"] >= 30 and row["oos_2026_roi_pct"] >= 100 and row["oos_2026_roi_ex_top1_pct"] >= 80:
            row["status"] = "watchlist_candidate"
        if row["oos_2026_tickets"] >= 50 and row["oos_2026_roi_pct"] >= 130 and row["oos_2026_roi_ex_top1_pct"] >= 100:
            row["status"] = "gate_candidate_needs_mcs"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["status", "oos_2026_roi_pct", "oos_2026_tickets"], ascending=[True, False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runners", type=Path, default=DEFAULT_RUNNERS)
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    parser.add_argument("--pace", type=Path, default=DEFAULT_PACE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner_cols = [
        "race_id",
        "芝・ダ",
        "距離",
        "クラス名",
        "馬場状態",
        "queue_type",
        "race_est_ten_pressure_score",
        "race_est_fast_start_count",
        "race_est_ten_speed_gap_top2",
        "race_est_queue_clarity_score",
        "actual_front5",
        "target_top3",
        "target_win",
    ]
    pace_cols = [
        "race_id",
        "venue_code",
        "race_no",
        "actual_lap_mode",
        "cont_predicted_lap_mode",
        "cont_confidence",
        "cont_margin",
        "front3f_sec",
        "last3f_sec",
        "rpci",
        "pci3",
        "course_front3f_prior_sec",
        "course_front3f_prior_std",
        "course_front3f_prior_count",
        "pred_front3f_sec",
        "pred_rpci",
        "pred_pci3",
        "cont_front_delta_z",
        "cont_rpci_delta",
    ]
    ticket_cols = [
        "race_id",
        "year",
        "ticket_type",
        "stake_yen",
        "return_yen",
        "pair_pred_front5_any",
        "pair_pred_front5_both",
        "pair_pred_leader_any",
        "pair_pred_front_complement",
        "queue_type",
        "queue_shape_label",
    ]
    runners = read_csv(args.runners, usecols=lambda c: c in set(runner_cols))
    pace = read_csv(args.pace, usecols=lambda c: c in set(pace_cols))
    tickets = read_csv(args.tickets, usecols=lambda c: c in set(ticket_cols))

    races = rolling_context_priors(build_race_table(runners, pace))
    segments = segment_summary(races)
    policy, by_year, thresholds = ticket_policy_metrics(tickets, races)
    notes = robust_notes(by_year)

    races.to_csv(args.out_dir / "race_high_pressure_front_survival_context.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(args.out_dir / "high_pressure_front_survival_segments.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(args.out_dir / "ticket_policy_metrics.csv", index=False, encoding="utf-8-sig")
    by_year.to_csv(args.out_dir / "ticket_policy_metrics_by_year.csv", index=False, encoding="utf-8-sig")
    notes.to_csv(args.out_dir / "policy_robustness_notes.csv", index=False, encoding="utf-8-sig")

    top_segments = segments.head(20).to_dict(orient="records")
    top_policy = policy.sort_values(["roi_ex_top5_pct", "tickets"], ascending=[False, False]).head(20)
    summary = {
        "output_dir": str(args.out_dir.relative_to(ROOT)),
        "races": int(len(races)),
        "race_years": [int(x) for x in sorted(races["year"].dropna().unique())],
        "tickets": int(len(tickets)),
        "overall_front_survival_rate_pct": float(races["actual_front_survival"].mean() * 100),
        "overall_front_collapse_rate_pct": float(races["actual_front_collapse"].mean() * 100),
        "actual_fast_or_frontload_front_survival_rate_pct": float(
            races.loc[races["actual_fast_or_frontload"], "actual_front_survival"].mean() * 100
        ),
        "thresholds_train_pre2026": thresholds,
        "top_high_pressure_front_survival_segments": top_segments,
        "top_ticket_policies": top_policy.to_dict(orient="records"),
        "robust_status_counts": notes["status"].value_counts().to_dict() if not notes.empty else {},
        "note": (
            "Uses actual 4th-corner labels only for validation. Ticket policies use pre-race queue/front-load "
            "features and rolling historical context priors. Policies remain shadow unless 2026 OOS support is robust."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# High Pressure Front Survival Context v1",
                "",
                "Purpose: split high/front-load race setups into contexts where the front still survives vs collapses.",
                "Actual positions are validation labels only. Operational cuts use predicted front5/leader flags,",
                "pre-race ten-pressure signals, and rolling historical context priors.",
                "",
                "Key outputs:",
                "- race_high_pressure_front_survival_context.csv",
                "- high_pressure_front_survival_segments.csv",
                "- ticket_policy_metrics.csv",
                "- ticket_policy_metrics_by_year.csv",
                "- policy_robustness_notes.csv",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
