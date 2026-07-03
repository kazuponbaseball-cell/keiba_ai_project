from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACK = ROOT / "outputs/analysis/track_regime_course_change_v1/race_track_regime_features.csv"
DEFAULT_FRONT = ROOT / "outputs/analysis/high_pressure_front_survival_context_v1/race_high_pressure_front_survival_context.csv"
DEFAULT_TICKETS = ROOT / "outputs/analysis/front3f_queue_lap_miss_lap_expansion_v1/front3f_queue_ticket_enriched.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/track_regime_lap_context_v1"


def read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, usecols=usecols)


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def text(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=object)
    return df[col].fillna(default).astype(str)


def norm_race_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(16)


def truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if series.dtype == object:
        return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)


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
            "roi_ex_top1_pct": np.nan,
            "roi_ex_top5_pct": np.nan,
            "top_return_share_pct": np.nan,
        }
    stake = pd.to_numeric(frame["stake_yen"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(frame["return_yen"], errors="coerce").fillna(0.0)
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    ordered = ret.sort_values(ascending=False)
    top1 = ordered.index[:1]
    top5 = ordered.index[:5]
    return {
        "policy": label,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": stake_sum,
        "return_yen": ret_sum,
        "profit_yen": ret_sum - stake_sum,
        "roi_pct": safe_div(ret_sum, stake_sum) * 100,
        "hit_rate_pct": float(ret.gt(0).mean() * 100),
        "roi_ex_top1_pct": safe_div(ret_sum - float(ret.loc[top1].sum()), stake_sum - float(stake.loc[top1].sum())) * 100,
        "roi_ex_top5_pct": safe_div(ret_sum - float(ret.loc[top5].sum()), stake_sum - float(stake.loc[top5].sum())) * 100,
        "top_return_share_pct": safe_div(float(ordered.iloc[0]) if len(ordered) else 0.0, ret_sum) * 100,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def load_races(track_path: Path, front_path: Path) -> pd.DataFrame:
    track_cols = [
        "race_id",
        "date_key",
        "venue",
        "surface",
        "distance",
        "class_name",
        "going",
        "surface_is_turf",
        "turf_course_code",
        "opening_week_flag",
        "late_meet_flag",
        "course_change_flag",
        "race_days_since_course_change",
        "turf_races_since_meet_start",
        "turf_races_since_course_change",
        "inner_wear_proxy",
        "course_freshness_score",
        "opening_speed_bias_prior",
        "inside_front_bias_prior",
        "outer_late_bias_prior",
        "track_regime_confidence",
        "track_regime_direction",
    ]
    front_cols = [
        "race_id",
        "year",
        "venue_code",
        "race_no",
        "surface",
        "distance_m",
        "class_tier",
        "going",
        "queue_type",
        "pre_front_load_signal",
        "context_survival_prior",
        "context_collapse_prior",
        "front_survival_despite_pressure_score",
        "front_collapse_warning_score",
        "actual_front_survival",
        "actual_front_collapse",
        "actual_top3_front5_share",
        "winner_front5",
        "actual_fast_or_frontload",
        "actual_lap_mode",
        "cont_predicted_lap_mode",
        "cont_confidence",
        "cont_margin",
        "front3f_sec",
        "course_front3f_prior_sec",
        "rpci",
        "pci3",
    ]
    tr = read_csv(track_path, usecols=lambda c: c in set(track_cols))
    fr = read_csv(front_path, usecols=lambda c: c in set(front_cols))
    tr["race_id"] = norm_race_id(tr["race_id"])
    fr["race_id"] = norm_race_id(fr["race_id"])
    races = tr.merge(fr, on="race_id", how="inner", suffixes=("_track", ""))
    races["year"] = pd.to_numeric(races.get("year"), errors="coerce").fillna(races["race_id"].str.slice(0, 4).astype(int)).astype(int)
    races["race_date"] = pd.to_datetime(races["race_id"].str.slice(0, 8), format="%Y%m%d", errors="coerce")
    races["month"] = races["race_date"].dt.to_period("M").astype(str)
    races["is_turf"] = num(races, "surface_is_turf", 0).fillna(0).eq(1)
    races = races[races["is_turf"]].copy()
    for col in [
        "opening_week_flag",
        "late_meet_flag",
        "course_change_flag",
        "race_days_since_course_change",
        "inner_wear_proxy",
        "course_freshness_score",
        "opening_speed_bias_prior",
        "inside_front_bias_prior",
        "outer_late_bias_prior",
        "pre_front_load_signal",
        "context_survival_prior",
        "context_collapse_prior",
        "actual_front_survival",
        "actual_front_collapse",
        "actual_top3_front5_share",
        "winner_front5",
    ]:
        races[col] = pd.to_numeric(races.get(col), errors="coerce")
    races["pred_fastish"] = text(races, "cont_predicted_lap_mode").isin(["fast", "sustain"])
    races["pred_slowish"] = text(races, "cont_predicted_lap_mode").eq("slow")
    races["fresh_or_change"] = (
        races["opening_week_flag"].fillna(0).eq(1)
        | races["course_change_flag"].fillna(0).eq(1)
        | races["race_days_since_course_change"].fillna(99).le(1)
        | races["course_freshness_score"].fillna(0).ge(0.60)
    )
    races["worn_or_outer"] = (
        races["late_meet_flag"].fillna(0).eq(1)
        | races["inner_wear_proxy"].fillna(0).ge(0.55)
        | races["outer_late_bias_prior"].fillna(0).ge(0.45)
    )
    return races.sort_values(["race_date", "venue_code", "race_no"], kind="mergesort")


def race_segments(races: pd.DataFrame) -> pd.DataFrame:
    q = {
        "fresh_q80": float(races["course_freshness_score"].quantile(0.80)),
        "inside_q80": float(races["inside_front_bias_prior"].quantile(0.80)),
        "outer_q80": float(races["outer_late_bias_prior"].quantile(0.80)),
        "wear_q80": float(races["inner_wear_proxy"].quantile(0.80)),
        "pressure_q60": float(races["pre_front_load_signal"].quantile(0.60)),
    }
    masks = {
        "all_turf_track_regime": pd.Series(True, index=races.index),
        "opening_week": races["opening_week_flag"].eq(1),
        "course_change_first2_racedays": races["race_days_since_course_change"].le(1),
        "fresh_or_change": races["fresh_or_change"],
        "fresh_q80": races["course_freshness_score"].ge(q["fresh_q80"]),
        "inside_front_q80": races["inside_front_bias_prior"].ge(q["inside_q80"]),
        "outer_late_q80": races["outer_late_bias_prior"].ge(q["outer_q80"]),
        "inner_wear_q80": races["inner_wear_proxy"].ge(q["wear_q80"]),
        "fresh_pred_fastish": races["fresh_or_change"] & races["pred_fastish"],
        "fresh_high_pressure": races["fresh_or_change"] & races["pre_front_load_signal"].ge(q["pressure_q60"]),
        "outer_pred_fastish": races["outer_late_bias_prior"].ge(q["outer_q80"]) & races["pred_fastish"],
    }
    rows = []
    for name, mask in masks.items():
        sub = races.loc[mask.fillna(False)].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "segment": name,
                "races": int(len(sub)),
                "front_survival_rate_pct": float(sub["actual_front_survival"].mean() * 100),
                "front_collapse_rate_pct": float(sub["actual_front_collapse"].mean() * 100),
                "winner_front5_rate_pct": float(sub["winner_front5"].mean() * 100),
                "actual_fast_or_frontload_rate_pct": float(truthy(sub["actual_fast_or_frontload"]).mean() * 100),
                "avg_top3_front5_share_pct": float(sub["actual_top3_front5_share"].mean() * 100),
                "avg_front3f_vs_course_prior": float((num(sub, "front3f_sec") - num(sub, "course_front3f_prior_sec")).mean()),
                "avg_rpci": float(num(sub, "rpci").mean()),
                "avg_pre_front_load_signal": float(sub["pre_front_load_signal"].mean()),
                "avg_course_freshness": float(sub["course_freshness_score"].mean()),
                "avg_inside_front_prior": float(sub["inside_front_bias_prior"].mean()),
                "avg_outer_late_prior": float(sub["outer_late_bias_prior"].mean()),
            }
        )
    return pd.DataFrame(rows), q


def load_ticket_context(tickets_path: Path, races: pd.DataFrame) -> pd.DataFrame:
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
        "cont_predicted_lap_mode",
    ]
    t = read_csv(tickets_path, usecols=lambda c: c in set(ticket_cols))
    t["race_id"] = norm_race_id(t["race_id"])
    keep_cols = [
        "race_id",
        "year",
        "month",
        "fresh_or_change",
        "worn_or_outer",
        "opening_week_flag",
        "course_change_flag",
        "race_days_since_course_change",
        "course_freshness_score",
        "opening_speed_bias_prior",
        "inside_front_bias_prior",
        "outer_late_bias_prior",
        "inner_wear_proxy",
        "track_regime_direction",
        "pre_front_load_signal",
        "context_survival_prior",
        "context_collapse_prior",
        "pred_fastish",
        "pred_slowish",
    ]
    t = t.merge(races[keep_cols], on="race_id", how="inner", suffixes=("", "_race"))
    t["year"] = pd.to_numeric(t.get("year"), errors="coerce").fillna(t["race_id"].str.slice(0, 4).astype(int)).astype(int)
    t["stake_yen"] = pd.to_numeric(t["stake_yen"], errors="coerce").fillna(0.0)
    t["return_yen"] = pd.to_numeric(t["return_yen"], errors="coerce").fillna(0.0)
    return t


def ticket_segments(t: pd.DataFrame, race_thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    front_any = truthy(t.get("pair_pred_front5_any", pd.Series(False, index=t.index)))
    front_both = truthy(t.get("pair_pred_front5_both", pd.Series(False, index=t.index)))
    leader_any = truthy(t.get("pair_pred_leader_any", pd.Series(False, index=t.index)))
    complement = truthy(t.get("pair_pred_front_complement", pd.Series(False, index=t.index)))
    pressure60 = t["pre_front_load_signal"].ge(race_thresholds["pressure_q60"])
    survival70 = t["context_survival_prior"].ge(t.loc[t["year"].lt(2026), "context_survival_prior"].quantile(0.70))
    inside80 = t["inside_front_bias_prior"].ge(race_thresholds["inside_q80"])
    outer80 = t["outer_late_bias_prior"].ge(race_thresholds["outer_q80"])
    fresh80 = t["course_freshness_score"].ge(race_thresholds["fresh_q80"])
    wear80 = t["inner_wear_proxy"].ge(race_thresholds["wear_q80"])
    fresh = t["fresh_or_change"].fillna(False).astype(bool)
    fastish = t["pred_fastish"].fillna(False).astype(bool)

    policies = {
        "base_turf_track": pd.Series(True, index=t.index),
        "fresh_front_any": fresh & front_any,
        "fresh_fastish_front_any": fresh & fastish & front_any,
        "fresh_pressure_front_any": fresh & pressure60 & front_any,
        "fresh_pressure_survival_front_any": fresh & pressure60 & survival70 & front_any,
        "fresh_q80_front_any": fresh80 & front_any,
        "inside_q80_front_any": inside80 & front_any,
        "inside_q80_front_both": inside80 & front_both,
        "inside_q80_leader_any": inside80 & leader_any,
        "outer_q80_front_any": outer80 & front_any,
        "outer_q80_front_complement": outer80 & complement,
        "outer_q80_avoid_front_both": outer80 & ~front_both,
        "wear_q80_front_any": wear80 & front_any,
        "track_lap_front_survival_watch": fresh & pressure60 & survival70 & front_any,
        "track_lap_caution_front_both": outer80 & wear80 & front_both,
    }
    overall_rows = []
    year_rows = []
    for ticket_type in ["wide", "umaren"]:
        tm = text(t, "ticket_type").eq(ticket_type)
        for name, mask in policies.items():
            sub = t.loc[tm & mask.fillna(False)].copy()
            label = f"{ticket_type}::{name}"
            overall_rows.append(metrics(sub, label))
            for year, group in sub.groupby("year", dropna=False):
                row = metrics(group, label)
                row["year"] = int(year)
                year_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    by_year = pd.DataFrame(year_rows)
    boot = bootstrap_by_month(t, policies)
    return overall, by_year, boot


def bootstrap_by_month(t: pd.DataFrame, policies: dict[str, pd.Series], iterations: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for ticket_type in ["wide", "umaren"]:
        tm = text(t, "ticket_type").eq(ticket_type)
        for name, mask in policies.items():
            sub = t.loc[tm & mask.fillna(False)].copy()
            if sub.empty:
                continue
            monthly = sub.groupby("month", dropna=False).agg(stake_yen=("stake_yen", "sum"), return_yen=("return_yen", "sum"))
            months = monthly.index.to_numpy()
            stakes = monthly["stake_yen"].to_numpy(dtype=float)
            returns = monthly["return_yen"].to_numpy(dtype=float)
            rois = []
            for _ in range(iterations):
                idx = rng.integers(0, len(months), len(months))
                st = float(stakes[idx].sum())
                rt = float(returns[idx].sum())
                if st > 0:
                    rois.append(rt / st * 100)
            arr = np.array(rois, dtype=float)
            rows.append(
                {
                    "policy": f"{ticket_type}::{name}",
                    "months": int(len(months)),
                    "tickets": int(len(sub)),
                    "bootstrap_roi_p05": float(np.percentile(arr, 5)) if len(arr) else np.nan,
                    "bootstrap_roi_p50": float(np.percentile(arr, 50)) if len(arr) else np.nan,
                    "bootstrap_roi_p95": float(np.percentile(arr, 95)) if len(arr) else np.nan,
                    "bootstrap_prob_roi_gt_100": float((arr > 100).mean()) if len(arr) else np.nan,
                    "bootstrap_prob_roi_gt_120": float((arr > 120).mean()) if len(arr) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["bootstrap_prob_roi_gt_100", "bootstrap_roi_p50"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-races", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--front-races", type=Path, default=DEFAULT_FRONT)
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    races = load_races(args.track_races, args.front_races)
    race_seg, race_thresholds = race_segments(races)
    tickets = load_ticket_context(args.tickets, races)
    ticket_overall, ticket_year, ticket_boot = ticket_segments(tickets, race_thresholds)

    races.to_csv(args.out_dir / "race_track_regime_lap_context.csv", index=False, encoding="utf-8-sig")
    race_seg.to_csv(args.out_dir / "race_track_regime_lap_segments.csv", index=False, encoding="utf-8-sig")
    ticket_overall.to_csv(args.out_dir / "ticket_track_regime_lap_policy_metrics.csv", index=False, encoding="utf-8-sig")
    ticket_year.to_csv(args.out_dir / "ticket_track_regime_lap_policy_by_year.csv", index=False, encoding="utf-8-sig")
    ticket_boot.to_csv(args.out_dir / "ticket_track_regime_lap_bootstrap.csv", index=False, encoding="utf-8-sig")

    key_policies = [
        "wide::fresh_pressure_survival_front_any",
        "wide::fresh_fastish_front_any",
        "wide::inside_q80_front_any",
        "wide::outer_q80_front_complement",
        "wide::outer_q80_avoid_front_both",
    ]
    summary = {
        "output_dir": str(args.out_dir.relative_to(ROOT)),
        "race_count": int(len(races)),
        "ticket_count": int(len(tickets)),
        "coverage": {
            "date_min": str(races["race_date"].min().date()) if not races.empty else None,
            "date_max": str(races["race_date"].max().date()) if not races.empty else None,
            "years": [int(x) for x in sorted(races["year"].dropna().unique())],
        },
        "race_thresholds": race_thresholds,
        "race_segments": race_seg.to_dict(orient="records"),
        "key_ticket_policies": ticket_overall[ticket_overall["policy"].isin(key_policies)].to_dict(orient="records"),
        "key_ticket_by_year": ticket_year[ticket_year["policy"].isin(key_policies)].to_dict(orient="records"),
        "best_bootstrap": ticket_boot.head(10).to_dict(orient="records"),
        "note": (
            "Course A/B/C/D and opening-week features are only available for the track-condition coverage window. "
            "This validation uses predicted lap/queue fields for ticket policies; actual lap and position fields are diagnostic labels only."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Track Regime Lap Context v1",
                "",
                "Checks opening week / course-change / inner-wear priors together with predicted lap and front-position pair flags.",
                "Actual race shape is used only for validation.",
                "",
                "Key outputs:",
                "- race_track_regime_lap_context.csv",
                "- race_track_regime_lap_segments.csv",
                "- ticket_track_regime_lap_policy_metrics.csv",
                "- ticket_track_regime_lap_policy_by_year.csv",
                "- ticket_track_regime_lap_bootstrap.csv",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
