from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


GATE_ORDER = [
    "DATA_NORMAL",
    "LIVE_ODDS",
    "ODDS_CAP",
    "MARGIN",
    "SCORE",
    "PAIR_VALUE",
    "SKIP_RISK",
    "FRONT_PROBABILITY",
    "PACE_FIT",
    "WORKOUT",
    "FIRST_CONDITION",
    "DANGER_FAVORITE",
    "RACE_DIFFICULTY",
    "GOING_INFO",
]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={"race_id": str, "raceId": str}, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype={"race_id": str, "raceId": str}, low_memory=False)


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def text(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=str)
    return frame[col].astype("string").fillna(default).astype(str)


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


def reason_fail(frame: pd.DataFrame, reason: str) -> pd.Series:
    col = f"reason_{reason}"
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[col].astype(str).str.lower().isin({"true", "1", "1.0"})


def parse_pair_key(pair_key: pd.Series) -> tuple[pd.Series, pd.Series]:
    pair = pair_key.astype(str).str.extract(r":(?P<a>\d+)-(?P<b>\d+)$")
    return pd.to_numeric(pair["a"], errors="coerce"), pd.to_numeric(pair["b"], errors="coerce")


def finish_top2_map(pnl: pd.DataFrame) -> pd.DataFrame:
    if pnl.empty or "raceId" not in pnl.columns or "finishTop3" not in pnl.columns:
        return pd.DataFrame(columns=["race_id", "top1", "top2"])
    out = pnl[["raceId", "finishTop3"]].dropna().drop_duplicates("raceId").copy()
    parts = out["finishTop3"].astype(str).str.extract(r"^\s*(\d+)[^\d]+(\d+)")
    out["top1"] = pd.to_numeric(parts[0], errors="coerce")
    out["top2"] = pd.to_numeric(parts[1], errors="coerce")
    out = out.rename(columns={"raceId": "race_id"})
    return out[["race_id", "top1", "top2"]].dropna()


def add_gate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    live_odds = num(out, "live_odds")
    ticket_hit_prob = num(out, "ticket_hit_prob", 0.0)
    score = num(out, "strongest_current_score", 0.0)
    track_freshness = text(out, "track_state_freshness")
    track_state = text(out, "track_state")

    out["gate_DATA_NORMAL_pass"] = ticket_hit_prob.gt(0) & score.gt(0) & text(out, "race_id").ne("")
    out["gate_LIVE_ODDS_pass"] = ~reason_fail(out, "LIVE_ODDS_MISSING")
    out["gate_ODDS_CAP_pass"] = ~reason_fail(out, "ODDS_TOO_HIGH")
    out["gate_MARGIN_pass"] = ~reason_fail(out, "MARGIN_FAIL")
    out["gate_SCORE_pass"] = ~reason_fail(out, "SCORE_FAIL")
    out["gate_PAIR_VALUE_pass"] = live_odds.gt(0) & ticket_hit_prob.mul(live_odds).ge(1.0)
    out["gate_SKIP_RISK_pass"] = ~reason_fail(out, "SKIP_RISK_FAIL")
    out["gate_FRONT_PROBABILITY_pass"] = ~reason_fail(out, "FRONT_PROBABILITY_FAIL")
    out["gate_PACE_FIT_pass"] = ~reason_fail(out, "PACE_FIT_FAIL")
    out["gate_WORKOUT_pass"] = ~reason_fail(out, "WORKOUT_FAIL")
    out["gate_FIRST_CONDITION_pass"] = ~reason_fail(out, "FIRST_CONDITION_FAIL")
    out["gate_DANGER_FAVORITE_pass"] = ~reason_fail(out, "DANGER_FAVORITE_FAIL")
    out["gate_RACE_DIFFICULTY_pass"] = ~reason_fail(out, "RACE_DIFFICULTY_FAIL")
    out["gate_GOING_INFO_pass"] = track_freshness.ne("unknown_or_missing") | track_state.ne("")

    gate_cols = [f"gate_{gate}_pass" for gate in GATE_ORDER]
    out["strict_gate_pass_count"] = out[gate_cols].sum(axis=1).astype(int)
    out["strict_gate_fail_count"] = len(gate_cols) - out["strict_gate_pass_count"]

    first_fail = []
    for _, row in out[gate_cols].iterrows():
        fail = "PASS_ALL_AUDIT_GATES"
        for gate, col in zip(GATE_ORDER, gate_cols):
            if not bool(row[col]):
                fail = gate
                break
        first_fail.append(fail)
    out["first_fail_gate"] = first_fail
    out["passes_all_audit_gates"] = out["first_fail_gate"].eq("PASS_ALL_AUDIT_GATES")

    leave_one = []
    for _, row in out[gate_cols].iterrows():
        failed = [gate for gate, col in zip(GATE_ORDER, gate_cols) if not bool(row[col])]
        leave_one.append(failed[0] if len(failed) == 1 else "")
    out["leave_one_gate"] = leave_one
    out["leave_one_gate_candidate"] = out["leave_one_gate"].ne("")
    return out


def add_proxy_returns(df: pd.DataFrame, pnl: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    a, b = parse_pair_key(text(out, "pair_key"))
    out["pair_a_no"] = a
    out["pair_b_no"] = b
    top2 = finish_top2_map(pnl)
    if top2.empty:
        out["proxy_hit_umaren_top2"] = False
        out["proxy_return_yen"] = 0.0
        out["proxy_profit_yen"] = -100.0
        return out
    out = out.merge(top2, on="race_id", how="left")
    lo = np.minimum(out["top1"], out["top2"])
    hi = np.maximum(out["top1"], out["top2"])
    pair_lo = np.minimum(out["pair_a_no"], out["pair_b_no"])
    pair_hi = np.maximum(out["pair_a_no"], out["pair_b_no"])
    hit = pair_lo.eq(lo) & pair_hi.eq(hi)
    out["proxy_hit_umaren_top2"] = hit.fillna(False)
    out["proxy_stake_yen"] = 100.0
    out["proxy_return_yen"] = np.where(out["proxy_hit_umaren_top2"], num(out, "live_odds", 0.0) * 100.0, 0.0)
    out["proxy_profit_yen"] = out["proxy_return_yen"] - out["proxy_stake_yen"]
    return out


def max_drawdown(profits: pd.Series) -> float:
    if profits.empty:
        return 0.0
    curve = pd.to_numeric(profits, errors="coerce").fillna(0.0).cumsum()
    peak = curve.cummax()
    dd = curve - peak
    return float(dd.min())


def roi_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "candidates": 0,
            "races": 0,
            "hits": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": None,
            "max_drawdown_yen": 0.0,
        }
    stake = float(num(df, "proxy_stake_yen", 100.0).sum())
    ret = float(num(df, "proxy_return_yen", 0.0).sum())
    profit = ret - stake
    return {
        "candidates": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "hits": int(df["proxy_hit_umaren_top2"].sum()) if "proxy_hit_umaren_top2" in df.columns else 0,
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": profit,
        "roi_pct": (ret / stake * 100.0) if stake > 0 else None,
        "max_drawdown_yen": max_drawdown(df.sort_values(["race_id", "pair_key"])["proxy_profit_yen"]),
    }


def summarize_gates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    any_rows = []
    loo_rows = []
    waterfall_rows = []
    gate_cols = [f"gate_{gate}_pass" for gate in GATE_ORDER]
    passed_so_far = pd.Series(True, index=df.index)
    for gate, col in zip(GATE_ORDER, gate_cols):
        failed = ~df[col].astype(bool)
        subset = df[failed].copy()
        any_rows.append(
            {
                "gate": gate,
                "any_fail_candidates": int(len(subset)),
                "any_fail_races": int(subset["race_id"].nunique()) if not subset.empty else 0,
                "single_gate_only_candidates": int(df["leave_one_gate"].eq(gate).sum()),
                "single_gate_only_races": int(df.loc[df["leave_one_gate"].eq(gate), "race_id"].nunique()),
            }
        )
        fail_now = passed_so_far & failed
        waterfall_rows.append(
            {
                "gate": gate,
                "pass_before_gate": int(passed_so_far.sum()),
                "fail_at_gate": int(fail_now.sum()),
                "pass_after_gate": int((passed_so_far & ~failed).sum()),
                "races_fail_at_gate": int(df.loc[fail_now, "race_id"].nunique()),
            }
        )
        passed_so_far = passed_so_far & ~failed

        loo = df[df["leave_one_gate"].eq(gate)].copy()
        row = {"gate": gate, **roi_summary(loo)}
        loo_rows.append(row)

    first_fail = (
        df.groupby("first_fail_gate", dropna=False)
        .agg(candidates=("pair_key", "count"), races=("race_id", "nunique"))
        .reset_index()
        .sort_values(["candidates"], ascending=False)
    )
    return (
        pd.DataFrame(waterfall_rows),
        pd.DataFrame(any_rows),
        first_fail,
        pd.DataFrame(loo_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gate dropout audit requested by external AI.")
    parser.add_argument("--ledger-csv", default="outputs/analysis/candidate_rejection_ledger_prepost_sim_v1/candidate_rejection_ledger.csv")
    parser.add_argument("--pnl-detail-csv", default="outputs/analysis/current_live_pnl/current_live_pnl_detail.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/gate_dropout_audit_v1")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = read_csv(project_path(args.ledger_csv))
    pnl = read_csv(project_path(args.pnl_detail_csv))
    if ledger.empty:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "rows": 0,
            "reason": "missing ledger",
        }
        (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    audit = add_gate_columns(ledger)
    audit = add_proxy_returns(audit, pnl)
    waterfall, any_fail, first_fail, leave_one = summarize_gates(audit)

    audit.to_csv(out_dir / "gate_candidate_audit.csv", index=False, encoding="utf-8-sig")
    waterfall.to_csv(out_dir / "gate_waterfall.csv", index=False, encoding="utf-8-sig")
    any_fail.to_csv(out_dir / "gate_any_fail_summary.csv", index=False, encoding="utf-8-sig")
    first_fail.to_csv(out_dir / "gate_first_fail_summary.csv", index=False, encoding="utf-8-sig")
    leave_one.to_csv(out_dir / "gate_leave_one_out_summary.csv", index=False, encoding="utf-8-sig")

    pass_all = audit[audit["passes_all_audit_gates"]].copy()
    leave_one_candidates = audit[audit["leave_one_gate_candidate"]].copy()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(audit)),
        "races": int(audit["race_id"].nunique()),
        "passes_all_audit_gates": roi_summary(pass_all),
        "single_gate_only_total": roi_summary(leave_one_candidates),
        "waterfall": waterfall.to_dict(orient="records"),
        "any_fail": any_fail.to_dict(orient="records"),
        "first_fail": first_fail.to_dict(orient="records"),
        "leave_one_out": leave_one.to_dict(orient="records"),
        "return_method": "proxy umaren return uses current live_odds * 100 for top-2 unordered hits; this is diagnostic only.",
        "policy": {
            "champion_changed": False,
            "promotion_allowed": False,
            "purpose": "diagnose strict-gate dropout before any threshold relaxation",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
