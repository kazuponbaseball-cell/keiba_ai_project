from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAX_FINAL_BUY_UMAREN_ODDS = 120.0
POSITION_FRONT_VALUE_MIN = 0.51


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={"race_id": str}, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False)


def append_history(path: Path, rows: pd.DataFrame, key_col: str) -> int:
    if rows.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        existing = read_csv(path)
        combined = pd.concat([existing.astype(str), rows.astype(str)], ignore_index=True, sort=False)
    else:
        combined = rows.astype(str)
    if key_col in combined.columns:
        combined = combined.drop_duplicates(subset=[key_col], keep="last")
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return int(len(combined))


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def text(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=str)
    return frame[col].astype("string").fillna(default).astype(str)


def first_num(frame: pd.DataFrame, names: list[str], default: float = np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def make_pair_key(frame: pd.DataFrame) -> pd.Series:
    a = first_num(frame, ["anchor_horse_no", "anchor_no", "horse_a", "a_no"])
    b = first_num(frame, ["partner_horse_no", "partner_no", "horse_b", "b_no"])
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return frame["race_id"].astype(str) + ":" + lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)


def make_ledger_ids(frame: pd.DataFrame) -> pd.Series:
    def one(row: pd.Series) -> str:
        raw = "|".join(
            [
                str(row.get("decision_label", "")),
                str(row.get("snapshot_time", "")),
                str(row.get("race_id", "")),
                str(row.get("pair_key", "")),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    return frame.apply(one, axis=1)


def norm01(series: pd.Series, lo: float = 0.0, hi: float = 1.0) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if hi <= lo:
        return pd.Series(0.0, index=series.index)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0)


def reason_flags(df: pd.DataFrame, *, include_post_time_lock: bool = True) -> pd.DataFrame:
    out = df.copy()
    live_odds = num(out, "live_odds")
    margin = num(out, "min_odds_margin_ratio", 0.0)
    expected_roi = num(out, "runtime_expected_roi", 0.0)
    first_unc = num(out, "first_condition_pair_uncertainty_score", 0.0)
    danger = num(out, "ticket_danger_popular_in_pair_score", 0.0)
    difficulty = num(out, "race_difficulty_score", 0.0)
    first_extra = first_unc.lt(0.60) | (margin.ge(3.00) & expected_roi.ge(1.60))
    danger_extra = danger.lt(0.42) | (margin.ge(3.00) & expected_roi.ge(1.60))
    readable_extra = difficulty.le(0.58) | (margin.ge(3.20) & expected_roi.ge(1.75))

    checks = {
        "LIVE_ODDS_MISSING": live_odds.isna() | live_odds.le(0),
        "SCORE_FAIL": num(out, "strongest_current_score", 0.0).lt(0.86),
        "MARGIN_FAIL": margin.lt(2.50),
        "SKIP_RISK_FAIL": num(out, "skip_risk_score", 0.0).gt(0.45),
        "FRONT_PROBABILITY_FAIL": num(out, "projected_front5_prob", 0.0).lt(0.60),
        "PACE_FIT_FAIL": num(out, "pace_fit_pair_score", 0.0).lt(0.35),
        "POSITION_FRONT_VALUE_FAIL": num(out, "position_front_value_score", 0.0).lt(POSITION_FRONT_VALUE_MIN),
        "WORKOUT_FAIL": num(out, "workout_pair_score", 0.0).lt(0.20),
        "ODDS_TOO_HIGH": live_odds.gt(MAX_FINAL_BUY_UMAREN_ODDS),
        "FIRST_CONDITION_FAIL": ~first_extra,
        "DANGER_FAVORITE_FAIL": ~danger_extra,
        "RACE_DIFFICULTY_FAIL": ~readable_extra,
    }
    if include_post_time_lock:
        checks = {
            "LIVE_ODDS_MISSING": checks["LIVE_ODDS_MISSING"],
            "POST_TIME_LOCK": num(out, "race_started_flag", 0.0).ge(1.0),
            **{k: v for k, v in checks.items() if k != "LIVE_ODDS_MISSING"},
        }
    reason_cols = []
    for reason, mask in checks.items():
        col = f"reason_{reason}"
        out[col] = mask.fillna(False)
        reason_cols.append(col)
    out["rejection_reason_count"] = out[reason_cols].sum(axis=1).astype(int)

    def join_reasons(row: pd.Series) -> str:
        reasons = [c.replace("reason_", "") for c in reason_cols if bool(row.get(c))]
        if len(reasons) > 1:
            reasons.append("MULTIPLE_FAIL")
        return "|".join(reasons) if reasons else "PASS"

    out["rejection_reasons"] = out.apply(join_reasons, axis=1)
    out["single_rejection_reason"] = np.where(out["rejection_reason_count"].eq(1), out["rejection_reasons"], "")
    out["shadow_challenger_bucket"] = np.select(
        [
            out["rejection_reasons"].eq("FRONT_PROBABILITY_FAIL"),
            out["rejection_reasons"].eq("MARGIN_FAIL"),
            out["rejection_reasons"].eq("POSITION_FRONT_VALUE_FAIL"),
            out["rejection_reasons"].eq("DANGER_FAVORITE_FAIL"),
            out["rejection_reasons"].eq("RACE_DIFFICULTY_FAIL"),
            out["rejection_reasons"].eq("ODDS_TOO_HIGH"),
            out["rejection_reasons"].eq("PASS"),
        ],
        [
            "challenger_a_front_only_fail",
            "challenger_c_margin_only_fail",
            "challenger_b_position_value_only_fail",
            "challenger_d_pair_danger_only_fail",
            "challenger_e_race_difficulty_only_fail",
            "odds_cap_fail_do_not_promote",
            "champion_buy_eligible",
        ],
        default=np.where(out["rejection_reason_count"].ge(2), "multiple_fail_do_not_promote", "other_single_fail"),
    )
    return out


def add_external_ai_audit_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    front_unc = (
        num(out, "anchor_front5_model_disagreement_score", 0.0)
        + num(out, "partner_front5_model_disagreement_score", 0.0)
    ) / 2.0
    scenario_fragility = (
        0.30 * norm01(num(out, "race_pace_collapse", 0.0))
        + 0.25 * norm01(num(out, "race_bias_volatility", 0.0))
        + 0.25 * norm01(num(out, "first_condition_pair_uncertainty_score", 0.0))
        + 0.20 * norm01(front_unc)
    )
    out["pair_danger_score_external_ai"] = (
        0.45 * norm01(num(out, "ticket_danger_popular_in_pair_score", 0.0))
        + 0.25 * scenario_fragility
        + 0.15 * norm01(num(out, "late_odds_drop_rate", 0.0))
        + 0.15 * norm01(num(out, "skip_risk_score", 0.0))
    ).clip(0.0, 1.0)
    out["prediction_uncertainty_score"] = (
        0.35 * norm01(num(out, "race_difficulty_score", 0.0))
        + 0.25 * norm01(num(out, "first_condition_pair_uncertainty_score", 0.0))
        + 0.20 * norm01(front_unc)
        + 0.20 * norm01(num(out, "race_pace_collapse", 0.0))
    ).clip(0.0, 1.0)
    out["market_efficiency_score"] = (
        0.35 * (1.0 - norm01(num(out, "market_overlay_score", 0.0)))
        + 0.25 * norm01(num(out, "race_market_top3_prob_sum", 0.0))
        + 0.20 * norm01(num(out, "late_odds_drop_rate", 0.0))
        + 0.20 * norm01(num(out, "ticket_danger_popular_in_pair_score", 0.0))
    ).clip(0.0, 1.0)
    out["race_readability_quadrant"] = np.select(
        [
            out["prediction_uncertainty_score"].lt(0.45) & out["market_efficiency_score"].lt(0.45),
            out["prediction_uncertainty_score"].lt(0.45) & out["market_efficiency_score"].ge(0.45),
            out["prediction_uncertainty_score"].ge(0.45) & out["market_efficiency_score"].lt(0.45),
            out["prediction_uncertainty_score"].ge(0.45) & out["market_efficiency_score"].ge(0.45),
        ],
        [
            "low_uncertainty_low_market_efficiency_buy_priority",
            "low_uncertainty_high_market_efficiency_value_check",
            "high_uncertainty_low_market_efficiency_shadow_only",
            "high_uncertainty_high_market_efficiency_skip",
        ],
        default="unknown",
    )
    return out


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_bucket = (
        df.groupby("shadow_challenger_bucket", dropna=False)
        .agg(
            candidates=("pair_key", "count"),
            races=("race_id", "nunique"),
            avg_live_odds=("live_odds", "mean"),
            avg_margin=("min_odds_margin_ratio", "mean"),
            avg_front5=("projected_front5_prob", "mean"),
            avg_pair_danger=("pair_danger_score_external_ai", "mean"),
            avg_prediction_uncertainty=("prediction_uncertainty_score", "mean"),
            avg_market_efficiency=("market_efficiency_score", "mean"),
        )
        .reset_index()
        .sort_values(["candidates"], ascending=False)
    )
    reason_cols = [c for c in df.columns if c.startswith("reason_")]
    reason_rows = []
    for col in reason_cols:
        subset = df[df[col]].copy()
        reason_rows.append(
            {
                "reason": col.replace("reason_", ""),
                "candidates": int(len(subset)),
                "races": int(subset["race_id"].nunique()) if not subset.empty else 0,
                "avg_live_odds": float(subset["live_odds"].mean()) if not subset.empty else None,
                "single_reason_candidates": int(subset["rejection_reason_count"].eq(1).sum()) if not subset.empty else 0,
            }
        )
    by_reason = pd.DataFrame(reason_rows).sort_values(["candidates"], ascending=False)
    return by_bucket, by_reason


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external-AI requested candidate rejection ledger.")
    parser.add_argument("--candidates-csv", default="outputs/analysis/current_strongest_runtime_v1/current_strongest_all_candidates.csv")
    parser.add_argument("--selected-csv", default="outputs/analysis/current_strongest_runtime_v1/selected_after_live_safety.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/candidate_rejection_ledger_v1")
    parser.add_argument("--decision-label", default="manual")
    parser.add_argument(
        "--history-csv",
        default="data/processed/live_decision_snapshots/current_strongest_candidate_rejection_ledger_history.csv",
    )
    parser.add_argument(
        "--ignore-post-time-lock",
        action="store_true",
        help="For post-race analysis only: ignore race_started_flag so rejection reasons remain analyzable.",
    )
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")
    candidates = read_csv(project_path(args.candidates_csv))
    selected = read_csv(project_path(args.selected_csv))
    if candidates.empty:
        payload = {"generated_at": generated_at, "rows": 0, "reason": "missing candidates"}
        (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    ledger = candidates.copy()
    ledger["pair_key"] = make_pair_key(ledger)
    if not selected.empty and "race_id" in selected.columns:
        selected = selected.copy()
        selected["pair_key"] = make_pair_key(selected)
        buy_keys = set(selected["pair_key"].astype(str))
    else:
        buy_keys = set()
    ledger["champion_selected_buy"] = ledger["pair_key"].astype(str).isin(buy_keys)
    ledger = reason_flags(ledger, include_post_time_lock=not args.ignore_post_time_lock)
    ledger = add_external_ai_audit_scores(ledger)
    ledger["decision"] = np.where(ledger["champion_selected_buy"], "BUY", "SHADOW_OR_SKIP")
    ledger["model_version"] = "current_strongest_mcs_pbo_strict_v1"
    ledger["snapshot_time"] = text(ledger, "runtime_decision_generated_at")
    ledger.loc[ledger["snapshot_time"].astype(str).str.strip().eq(""), "snapshot_time"] = generated_at
    ledger["decision_label"] = args.decision_label
    ledger["candidate_ledger_id"] = make_ledger_ids(ledger)
    ledger["track_state"] = text(ledger, "anchor_runtime_going").where(text(ledger, "anchor_runtime_going").ne(""), text(ledger, "partner_runtime_going"))
    ledger["track_state_freshness"] = np.where(
        text(ledger, "anchor_runtime_track_condition_available").str.lower().isin({"true", "1", "1.0"}),
        "available",
        "unknown_or_missing",
    )

    keep_cols = [
        "candidate_ledger_id",
        "race_id",
        "pair_key",
        "snapshot_time",
        "decision_label",
        "model_version",
        "anchor_horse_no",
        "partner_horse_no",
        "anchor_馬名",
        "partner_馬名",
        "ticket_hit_prob",
        "projected_front5_prob",
        "pace_fit_pair_score",
        "position_front_value_score",
        "position_closer_value_score",
        "closer_logic_watch_score",
        "closer_logic_watch_flag",
        "closer_pair_max",
        "collapse_fit",
        "front_front_slow_fit",
        "front_front_clash",
        "anchor_projected_front5_prob",
        "partner_projected_front5_prob",
        "live_odds",
        "min_odds_margin_ratio",
        "runtime_expected_roi",
        "strongest_current_score",
        "ticket_danger_popular_in_pair_score",
        "pair_danger_score_external_ai",
        "race_difficulty_score",
        "prediction_uncertainty_score",
        "market_efficiency_score",
        "race_readability_quadrant",
        "skip_risk_score",
        "track_state",
        "track_state_freshness",
        "decision",
        "champion_selected_buy",
        "rejection_reasons",
        "single_rejection_reason",
        "rejection_reason_count",
        "shadow_challenger_bucket",
    ]
    keep_cols = [c for c in keep_cols if c in ledger.columns]
    out = ledger[keep_cols + [c for c in ledger.columns if c.startswith("reason_")]].copy()
    by_bucket, by_reason = summarize(ledger)
    out.to_csv(out_dir / "candidate_rejection_ledger.csv", index=False, encoding="utf-8-sig")
    history_rows = 0
    if args.history_csv:
        history_rows = append_history(project_path(args.history_csv), out, "candidate_ledger_id")
    by_bucket.to_csv(out_dir / "shadow_challenger_bucket_summary.csv", index=False, encoding="utf-8-sig")
    by_reason.to_csv(out_dir / "rejection_reason_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_rows": int(len(ledger)),
        "candidate_races": int(ledger["race_id"].nunique()),
        "champion_selected_buys": int(ledger["champion_selected_buy"].sum()),
        "decision_label": args.decision_label,
        "history_csv": str(project_path(args.history_csv)) if args.history_csv else "",
        "history_rows": history_rows,
        "post_time_lock_included": not args.ignore_post_time_lock,
        "bucket_summary": by_bucket.to_dict(orient="records"),
        "reason_summary": by_reason.to_dict(orient="records"),
        "note": "Promote no Challenger from this ledger alone; require immutable snapshots and multi-week OOS results.",
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
