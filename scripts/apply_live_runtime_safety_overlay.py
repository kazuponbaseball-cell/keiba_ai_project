from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_operational_win_addon import _json_default
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _read_optional(path: str, dtype: dict | None = None) -> pd.DataFrame | None:
    p = project_path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    return pd.read_csv(p, dtype=dtype or {"race_id": str}, low_memory=False)


def _metric(df: pd.DataFrame, label: str) -> dict:
    selected = df[_num(df.get("runtime_stake_yen"), df.index, 0.0).fillna(0.0).gt(0)].copy()
    stake = float(_num(selected.get("runtime_stake_yen"), selected.index, 0.0).fillna(0.0).sum())
    ret = float(_num(selected.get("runtime_return_yen"), selected.index, 0.0).fillna(0.0).sum())
    races = int(selected["race_id"].nunique()) if not selected.empty and "race_id" in selected.columns else 0
    if selected.empty:
        hit_tickets = selected
        hit_flag = pd.Series(dtype=bool)
    else:
        hit_raw = selected.get("hit")
        hit_flag = hit_raw.astype(bool) if hit_raw is not None else pd.Series(False, index=selected.index)
        hit_tickets = selected[hit_flag]
    hit_races = int(hit_tickets["race_id"].nunique()) if not hit_tickets.empty and "race_id" in hit_tickets.columns else 0
    curve = (
        selected.sort_values(["date_key", "race_id"]).groupby("race_id", sort=False)[["runtime_stake_yen", "runtime_return_yen"]].sum()
        if not selected.empty and "date_key" in selected.columns
        else pd.DataFrame(columns=["runtime_stake_yen", "runtime_return_yen"])
    )
    pnl = curve["runtime_return_yen"] - curve["runtime_stake_yen"] if not curve.empty else pd.Series(dtype=float)
    equity = pnl.cumsum()
    dd = equity - equity.cummax() if not equity.empty else pd.Series(dtype=float)
    return {
        "policy": label,
        "tickets": int(len(selected)),
        "races": races,
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(hit_flag.mean()) if len(selected) else 0.0,
        "race_hit_rate": hit_races / races if races else 0.0,
        "max_drawdown_yen": float(dd.min()) if not dd.empty else 0.0,
    }


def _resolve_horse_no_col(tickets: pd.DataFrame, prefix: str, preferred: str) -> str:
    candidates = [
        preferred,
        f"{prefix}_horse_no",
        "a_no" if prefix == "anchor" else "b_no",
    ]
    for col in candidates:
        if col in tickets.columns:
            return col
    return preferred


def _merge_body_weight(tickets: pd.DataFrame, body: pd.DataFrame | None, prefix: str, no_col: str) -> pd.DataFrame:
    out = tickets.copy()
    for col in (f"{prefix}_live_body_weight", f"{prefix}_live_body_weight_diff", f"{prefix}_body_weight_snapshot_at"):
        if col not in out.columns:
            out[col] = np.nan if "snapshot" not in col else ""
    no_col = _resolve_horse_no_col(out, prefix, no_col)
    if body is None or body.empty or no_col not in out.columns:
        return out
    b = body.copy()
    b["race_id"] = b["race_id"].astype(str)
    rename = {
        "馬番": "horse_no",
        "馬体重": "body_weight",
        "増減": "body_weight_diff",
        "snapshot_at": "body_weight_snapshot_at",
    }
    b = b.rename(columns={k: v for k, v in rename.items() if k in b.columns})
    required = {"race_id", "horse_no", "body_weight", "body_weight_diff"}
    if not required.issubset(b.columns):
        return out
    b["horse_no"] = pd.to_numeric(b["horse_no"], errors="coerce").astype("Int64")
    for col in ("body_weight", "body_weight_diff"):
        b[col] = pd.to_numeric(b[col], errors="coerce")
    if "body_weight_snapshot_at" not in b.columns:
        b["body_weight_snapshot_at"] = ""
    b = b[["race_id", "horse_no", "body_weight", "body_weight_diff", "body_weight_snapshot_at"]].drop_duplicates(
        ["race_id", "horse_no"], keep="last"
    )
    b = b.rename(
        columns={
            "body_weight": f"{prefix}_live_body_weight",
            "body_weight_diff": f"{prefix}_live_body_weight_diff",
            "body_weight_snapshot_at": f"{prefix}_body_weight_snapshot_at",
        }
    )
    out[no_col] = pd.to_numeric(out[no_col], errors="coerce").astype("Int64")
    return out.drop(
        columns=[f"{prefix}_live_body_weight", f"{prefix}_live_body_weight_diff", f"{prefix}_body_weight_snapshot_at"],
        errors="ignore",
    ).merge(b, left_on=["race_id", no_col], right_on=["race_id", "horse_no"], how="left").drop(columns=["horse_no"], errors="ignore")


def apply_overlay(tickets: pd.DataFrame, body_weight: pd.DataFrame | None = None) -> pd.DataFrame:
    out = tickets.copy()
    if out.empty:
        for col, default in {
            "live_odds_movement_risk": 0.0,
            "live_body_weight_risk": 0.0,
            "live_same_day_bias_risk": 0.0,
            "live_alert_risk_score": 0.0,
            "pre_live_safety_stake_yen": 0.0,
            "live_safety_action": "",
            "live_safety_stake_multiplier": 0.0,
            "live_safety_reason": "no_tickets",
            "live_safety_status": "",
            "runtime_return_yen": 0.0,
        }.items():
            if col not in out.columns:
                out[col] = default
        return out
    out["race_id"] = out["race_id"].astype(str)
    if "date_key" not in out.columns:
        out["date_key"] = out.get("日付S", out["race_id"].str[:8]).astype(str)
    out = _merge_body_weight(out, body_weight, "anchor", "anchor_no")
    out = _merge_body_weight(out, body_weight, "partner", "partner_no")
    idx = out.index

    odds_margin = _num(out.get("runtime_odds_margin_ratio"), idx, np.nan)
    late_drop = _num(out.get("late_odds_drop_rate"), idx, 0.0).fillna(0.0)
    late_drift = _num(out.get("late_odds_drift_rate"), idx, 0.0).fillna(0.0)
    expected_slip_roi = _num(out.get("expected_roi_after_slippage"), idx, np.nan)
    out["live_odds_movement_risk"] = _clip01(
        0.38 * late_drift
        + 0.22 * (1.0 - odds_margin.fillna(1.0)).clip(lower=0.0, upper=1.0)
        + 0.20 * (1.0 - expected_slip_roi.fillna(1.0)).clip(lower=0.0, upper=1.0)
        + 0.20 * late_drop.where(_num(out.get("market_overlay_score"), idx, 0.5).lt(0.45), 0.0)
    )

    anchor_diff = _num(out.get("anchor_live_body_weight_diff"), idx, np.nan)
    partner_diff = _num(out.get("partner_live_body_weight_diff"), idx, np.nan)
    body_score = _num(out.get("ticket_body_age_layoff_score"), idx, 0.5).fillna(0.5)
    live_body_available = anchor_diff.notna() | partner_diff.notna()
    extreme_anchor = (anchor_diff.abs() / 24.0).clip(0.0, 1.0)
    extreme_partner = (partner_diff.abs() / 24.0).clip(0.0, 1.0)
    pair_extreme = np.maximum(extreme_anchor.fillna(0.0), extreme_partner.fillna(0.0))
    out["live_body_weight_risk"] = _clip01((0.60 * pair_extreme + 0.40 * (1.0 - body_score)).where(live_body_available, 0.0))

    bias_ready = _num(out.get("same_day_bias_ready"), idx, _num(out.get("race_same_day_ready"), idx, 0.0)).fillna(0.0)
    bias_fit = _num(out.get("same_day_bias_fit_score"), idx, 0.5).fillna(0.5)
    bias_vol = _num(out.get("same_day_bias_volatility"), idx, _num(out.get("race_bias_volatility"), idx, 0.0)).fillna(0.0)
    front_load = _num(out.get("same_day_projected_front_load_score"), idx, 0.5).fillna(0.5)
    closer_block = _num(out.get("same_day_closer_blocked_index"), idx, 0.5).fillna(0.5)
    out["live_same_day_bias_risk"] = _clip01(
        bias_ready.clip(0, 1)
        * (0.46 * (1.0 - bias_fit).clip(0, 1) + 0.24 * bias_vol.clip(0, 1) + 0.15 * front_load.clip(0, 1) + 0.15 * closer_block.clip(0, 1))
    )

    danger = _num(out.get("ticket_danger_popular_score"), idx, 0.0).fillna(0.0)
    difficulty = _num(out.get("race_difficulty_score"), idx, _num(out.get("difficulty"), idx, 0.5)).fillna(0.5)
    pace_collapse = _num(out.get("race_pace_collapse"), idx, 0.0).fillna(0.0)
    out["live_alert_risk_score"] = _clip01(
        0.26 * out["live_odds_movement_risk"]
        + 0.18 * out["live_body_weight_risk"]
        + 0.18 * out["live_same_day_bias_risk"]
        + 0.22 * danger
        + 0.16 * np.maximum(difficulty, pace_collapse).clip(0, 1)
    )

    actions: list[str] = []
    reasons: list[str] = []
    status: list[str] = []
    mults: list[float] = []
    for _, row in out.iterrows():
        base_action = str(row.get("runtime_action") or "")
        risk = float(row.get("live_alert_risk_score") or 0.0)
        odds_risk = float(row.get("live_odds_movement_risk") or 0.0)
        body_risk = float(row.get("live_body_weight_risk") or 0.0)
        bias_risk = float(row.get("live_same_day_bias_risk") or 0.0)
        danger_risk = float(row.get("ticket_danger_popular_score") or 0.0)
        reason_bits = []
        if odds_risk >= 0.62:
            reason_bits.append("odds_movement_risk")
        if body_risk >= 0.62:
            reason_bits.append("body_weight_delta_risk")
        if bias_risk >= 0.62:
            reason_bits.append("same_day_bias_mismatch")
        if danger_risk >= 0.72:
            reason_bits.append("danger_popular")
        if risk >= 0.78:
            action, mult, st = "SKIP_ALERT", 0.0, "警戒見送り"
        elif risk >= 0.64 or odds_risk >= 0.70 or body_risk >= 0.72:
            action, mult, st = "REDUCE_ALERT", 0.50, "警戒減額"
        elif risk >= 0.54:
            action, mult, st = "WATCH_ALERT", 1.00, "警戒"
        else:
            action, mult, st = base_action, 1.00, str(row.get("runtime_ticket_status") or base_action)
        if base_action not in {"BUY", "BUY_CONTEXT_BOOST", "REDUCE"}:
            action, mult, st = base_action, 0.0, str(row.get("runtime_ticket_status") or base_action)
        actions.append(action)
        mults.append(mult)
        status.append(st)
        reasons.append("|".join(reason_bits) if reason_bits else "live_safety_ok")

    base_stake = _num(out.get("runtime_stake_yen"), idx, 0.0).fillna(0.0)
    out["pre_live_safety_stake_yen"] = base_stake
    out["live_safety_action"] = actions
    out["live_safety_stake_multiplier"] = mults
    out["live_safety_reason"] = reasons
    out["live_safety_status"] = status
    out["runtime_action"] = out["live_safety_action"]
    out["runtime_ticket_status"] = out["live_safety_status"]
    runtime_reason = out["runtime_reason"].astype(str) if "runtime_reason" in out.columns else pd.Series("", index=idx, dtype=str)
    out["runtime_reason"] = runtime_reason + "|live_safety:" + out["live_safety_reason"]
    out["runtime_stake_yen"] = (np.floor(base_stake * out["live_safety_stake_multiplier"] / 100.0) * 100.0).clip(lower=0.0)
    out.loc[out["runtime_action"].eq("REDUCE_ALERT") & out["runtime_stake_yen"].lt(100) & base_stake.gt(0), "runtime_stake_yen"] = 100.0
    pay = _num(out.get("runtime_backtest_pay_per100"), idx, _num(out.get("quote_pay_proxy_per100"), idx, 0.0)).fillna(0.0)
    hit_raw = out.get("hit")
    if hit_raw is None:
        hit_flag = pd.Series(False, index=idx)
    else:
        hit_flag = hit_raw.astype(bool)
    out["runtime_return_yen"] = np.where(hit_flag, pay * out["runtime_stake_yen"] / 100.0, 0.0)
    return out


def _summary_by_action(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action, g in df.groupby("live_safety_action", dropna=False):
        m = _metric(g, str(action))
        m["action"] = str(action)
        m["avg_alert_risk"] = float(_num(g.get("live_alert_risk_score"), g.index, 0.0).mean())
        rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply live runtime safety overlay for odds movement, body weight, same-day bias, and danger alerts.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/standard_staking_plan_v1/standard_staked_tickets.csv")
    parser.add_argument("--body-weight-csv", default="data/processed/live_body_weight/body_weight_latest.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/live_runtime_safety_overlay_v1")
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    body = _read_optional(args.body_weight_csv)
    overlaid = apply_overlay(tickets, body)

    out_dir = ensure_dir(project_path(args.output_dir))
    overlaid.to_csv(out_dir / "live_safety_overlaid_tickets.csv", index=False, encoding="utf-8-sig")
    _summary_by_action(overlaid).to_csv(out_dir / "live_safety_action_summary.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(
        [
            _metric(tickets, "before_live_safety"),
            _metric(overlaid, "after_live_safety"),
        ]
    )
    metrics.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    payload = {
        "tickets_csv": args.tickets_csv,
        "body_weight_loaded": body is not None and not body.empty,
        "output_tickets": str(out_dir / "live_safety_overlaid_tickets.csv"),
        "metrics": metrics.to_dict(orient="records"),
        "action_counts": overlaid["live_safety_action"].value_counts().to_dict(),
        "note": "This overlay is intentionally conservative. It is designed to reduce or warn on fragile live conditions, not to add aggressive buys.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
