from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_operational_win_addon import _json_default, _metrics
from src.utils.paths import ensure_dir, project_path


TYPE_RULES = {
    "wide": {
        "buy_margin": 1.30,
        "reduce_margin": 1.00,
        "wait_margin": 0.92,
        "min_prob": 0.055,
        "high_prob": 0.13,
        "stake_mult_buy": 1.00,
        "stake_mult_reduce": 0.50,
    },
    "umaren": {
        "buy_margin": 1.45,
        "reduce_margin": 1.08,
        "wait_margin": 0.98,
        "min_prob": 0.035,
        "high_prob": 0.11,
        "stake_mult_buy": 1.00,
        "stake_mult_reduce": 0.50,
    },
    "win": {
        "buy_margin": 1.25,
        "reduce_margin": 1.05,
        "wait_margin": 0.96,
        "min_prob": 0.020,
        "high_prob": 0.09,
        "stake_mult_buy": 1.00,
        "stake_mult_reduce": 0.50,
    },
}


DISPLAY_PAYOUT_RATES = {
    "win": 0.80,
    "place": 0.80,
    "wide": 0.775,
    "umaren": 0.775,
    "umatan": 0.75,
    "trio": 0.75,
    "trifecta": 0.725,
}


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _odds(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    values = _num(series, index, default)
    return values.where(values.ge(1.0) & values.lt(999.0))


def _read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False)


def _pair_key_from_ticket(df: pd.DataFrame) -> pd.Series:
    a = _num(df.get("anchor_no"), df.index, np.nan)
    b = _num(df.get("partner_no"), df.index, np.nan)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return df["race_id"].astype(str) + ":" + lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)


def _normalize_pair_live(live: pd.DataFrame | None) -> pd.DataFrame | None:
    if live is None or live.empty:
        return None
    out = live.copy()
    rename = {
        "bet": "ticket_type",
        "type": "ticket_type",
        "horse_a": "a_no",
        "horse_b": "b_no",
        "umaban_a": "a_no",
        "umaban_b": "b_no",
        "pay": "live_pay_per100",
        "odds": "live_odds",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    required = {"race_id", "ticket_type", "a_no", "b_no"}
    if not required.issubset(out.columns):
        return None
    if "live_pay_per100" not in out.columns:
        if "live_odds" not in out.columns:
            return None
        live_odds = _num(out["live_odds"], out.index, np.nan)
        out["live_pay_per100"] = live_odds.where(live_odds.gt(0)) * 100.0
    else:
        pay = _num(out["live_pay_per100"], out.index, np.nan)
        out["live_pay_per100"] = pay.where(pay.gt(0))
    out["race_id"] = out["race_id"].astype(str)
    out["ticket_type"] = out["ticket_type"].astype(str)
    a = _num(out["a_no"], out.index, np.nan)
    b = _num(out["b_no"], out.index, np.nan)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    out["runtime_pair_key"] = out["race_id"] + ":" + lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)
    if "snapshot_at" not in out.columns:
        out["snapshot_at"] = ""
    keep = ["race_id", "ticket_type", "runtime_pair_key", "live_pay_per100", "snapshot_at"]
    return out[keep].drop_duplicates(["race_id", "ticket_type", "runtime_pair_key"], keep="last")


def _normalize_single_live(live: pd.DataFrame | None) -> pd.DataFrame | None:
    if live is None or live.empty:
        return None
    out = live.copy()
    rename = {
        "horse_number": "horse_no",
        "win_odds": "live_win_odds",
        "odds": "live_win_odds",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    if not {"race_id", "horse_no"}.issubset(out.columns):
        return None
    out["race_id"] = out["race_id"].astype(str)
    out["horse_no"] = _num(out["horse_no"], out.index, np.nan)
    if "live_win_odds" not in out.columns:
        return None
    out["live_pay_per100"] = _odds(out["live_win_odds"], out.index, np.nan) * 100.0
    if "snapshot_at" not in out.columns:
        out["snapshot_at"] = ""
    return out[["race_id", "horse_no", "live_pay_per100", "snapshot_at"]].drop_duplicates(["race_id", "horse_no"], keep="last")


def _historical_pay_per100(df: pd.DataFrame) -> pd.Series:
    return np.select(
        [
            df["ticket_type"].eq("win"),
            df["ticket_type"].eq("wide"),
            df["ticket_type"].eq("umaren"),
        ],
        [
            _num(df.get("win_pay"), df.index, 0.0),
            _num(df.get("wide_pay"), df.index, 0.0),
            _num(df.get("umaren_pay"), df.index, 0.0),
        ],
        default=_num(df.get("return_yen"), df.index, 0.0),
    )


def _add_live_pay(tickets: pd.DataFrame, pair_live: pd.DataFrame | None, single_live: pd.DataFrame | None) -> pd.DataFrame:
    out = tickets.copy()
    # Runtime tickets are often passed through multiple review/debug stages.
    # Drop stale live columns before joining a fresh snapshot; otherwise pandas
    # suffixes them as *_x/*_y and the decision gate cannot see the new odds.
    out = out.drop(columns=["live_pay_per100", "snapshot_at"], errors="ignore")
    out["runtime_pair_key"] = _pair_key_from_ticket(out)
    if pair_live is not None:
        out = out.merge(pair_live, on=["race_id", "ticket_type", "runtime_pair_key"], how="left")
    else:
        out["live_pay_per100"] = np.nan
        out["snapshot_at"] = ""

    if single_live is not None:
        single = single_live.rename(columns={"live_pay_per100": "single_live_pay_per100", "snapshot_at": "single_snapshot_at"})
        out["anchor_no_num"] = _num(out.get("anchor_no"), out.index, np.nan)
        out = out.merge(single, left_on=["race_id", "anchor_no_num"], right_on=["race_id", "horse_no"], how="left")
        win_mask = out["ticket_type"].eq("win") & _num(out.get("single_live_pay_per100"), out.index, np.nan).gt(0)
        out.loc[win_mask, "live_pay_per100"] = out.loc[win_mask, "single_live_pay_per100"]
        out.loc[win_mask, "snapshot_at"] = out.loc[win_mask, "single_snapshot_at"].fillna("")
        out = out.drop(columns=["horse_no", "anchor_no_num", "single_live_pay_per100", "single_snapshot_at"], errors="ignore")

    return out


def apply_decisions(
    tickets: pd.DataFrame,
    pair_live: pd.DataFrame | None = None,
    single_live: pd.DataFrame | None = None,
    use_proxy_when_missing: bool = True,
) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["ticket_type"] = out.get("ticket_type", "").astype(str)
    out["stake_yen"] = _num(out.get("stake_yen"), out.index, 100.0).fillna(100.0)
    out["ticket_hit_prob"] = _num(out.get("ticket_hit_prob"), out.index, np.nan).fillna(0.0)
    out["required_pay_per100"] = _num(out.get("required_pay_per100"), out.index, np.nan)
    out["quote_pay_proxy_per100"] = _num(out.get("quote_pay_proxy_per100"), out.index, np.nan)
    out["min_acceptable_odds"] = _num(out.get("min_acceptable_odds"), out.index, np.nan)

    pair_live_norm = _normalize_pair_live(pair_live)
    single_live_norm = _normalize_single_live(single_live)
    out = _add_live_pay(out, pair_live_norm, single_live_norm)

    live_pay = _num(out.get("live_pay_per100"), out.index, np.nan)
    proxy_pay = out["quote_pay_proxy_per100"]
    out["runtime_pay_per100"] = live_pay.where(live_pay.gt(0), proxy_pay if use_proxy_when_missing else np.nan)
    out["runtime_odds"] = out["runtime_pay_per100"] / 100.0
    out["runtime_pay_source"] = np.where(live_pay.gt(0), "live", np.where(use_proxy_when_missing, "proxy", "missing"))
    out["runtime_odds_margin_ratio"] = out["runtime_pay_per100"] / out["required_pay_per100"].replace(0, np.nan)

    actions: list[str] = []
    multipliers: list[float] = []
    reasons: list[str] = []
    for _, row in out.iterrows():
        ticket_type = str(row.get("ticket_type", ""))
        rule = TYPE_RULES.get(ticket_type, TYPE_RULES["wide"])
        margin = float(row.get("runtime_odds_margin_ratio") or 0.0)
        prob = float(row.get("ticket_hit_prob") or 0.0)
        source = str(row.get("runtime_pay_source") or "")
        if source == "missing" or not np.isfinite(margin):
            action, mult, reason = "WAIT", 0.0, "live_odds_missing"
        elif prob < rule["min_prob"]:
            action, mult, reason = "SKIP", 0.0, "ticket_probability_too_low"
        elif margin >= rule["buy_margin"] or (margin >= rule["reduce_margin"] and prob >= rule["high_prob"]):
            action, mult, reason = "BUY", rule["stake_mult_buy"], "odds_value_cushion_ok"
        elif margin >= rule["reduce_margin"]:
            action, mult, reason = "REDUCE", rule["stake_mult_reduce"], "thin_but_playable"
        elif margin >= rule["wait_margin"]:
            action, mult, reason = "WAIT", 0.0, "near_threshold_watch_odds"
        else:
            action, mult, reason = "SKIP", 0.0, "odds_value_cushion_lost"
        actions.append(action)
        multipliers.append(mult)
        reasons.append(reason)

    out["runtime_action"] = actions
    out["runtime_stake_multiplier"] = multipliers
    out["runtime_reason"] = reasons
    out["runtime_stake_yen"] = (out["stake_yen"] * out["runtime_stake_multiplier"] / 100.0).round().mul(100.0)
    out.loc[out["runtime_action"].eq("REDUCE") & out["runtime_stake_yen"].lt(100), "runtime_stake_yen"] = 100.0
    out.loc[~out["runtime_action"].isin(["BUY", "REDUCE"]), "runtime_stake_yen"] = 0.0

    historical_pay = _historical_pay_per100(out)
    out["runtime_backtest_pay_per100"] = np.where(live_pay.gt(0), live_pay, historical_pay)
    out["runtime_return_yen"] = np.where(
        out.get("hit", False).astype(bool),
        out["runtime_backtest_pay_per100"] * out["runtime_stake_yen"] / 100.0,
        0.0,
    )
    out["runtime_expected_roi"] = out["ticket_hit_prob"] * out["runtime_pay_per100"] / 100.0
    payout_rate = out["ticket_type"].map(DISPLAY_PAYOUT_RATES).fillna(0.775).astype(float)
    runtime_odds = _num(out.get("runtime_odds"), out.index, np.nan).replace(0, np.nan)
    out["runtime_market_implied_prob_takeout_adj"] = (payout_rate / runtime_odds).replace([np.inf, -np.inf], np.nan)
    out["runtime_break_even_prob"] = (1.0 / runtime_odds).replace([np.inf, -np.inf], np.nan)
    out["runtime_model_market_prob_ratio"] = (
        out["ticket_hit_prob"] / out["runtime_market_implied_prob_takeout_adj"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    out["runtime_model_market_log_edge"] = (
        np.log(out["ticket_hit_prob"].clip(0.0001, 0.95))
        - np.log(out["runtime_market_implied_prob_takeout_adj"].clip(0.0001, 0.95))
    ).replace([np.inf, -np.inf], np.nan)
    out["runtime_break_even_prob_diff"] = out["ticket_hit_prob"] - out["runtime_break_even_prob"]
    out["runtime_ticket_status"] = np.select(
        [out["runtime_action"].eq("BUY"), out["runtime_action"].eq("REDUCE"), out["runtime_action"].eq("WAIT")],
        ["買い", "減額", "待ち"],
        default="見送り",
    )
    return out


def _runtime_metrics(df: pd.DataFrame, label: str) -> dict:
    selected = df[df["runtime_stake_yen"].gt(0)].copy()
    if selected.empty:
        return {"policy": label, "tickets": 0, "races": 0, "roi": 0.0}
    eval_df = selected.copy()
    eval_df["stake_yen"] = eval_df["runtime_stake_yen"]
    eval_df["return_yen"] = eval_df["runtime_return_yen"]
    return _metrics(eval_df, label)


def _decision_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ticket_type, action), g in df.groupby(["ticket_type", "runtime_action"], dropna=False):
        stake = float(_num(g.get("runtime_stake_yen"), g.index, 0.0).sum())
        ret = float(_num(g.get("runtime_return_yen"), g.index, 0.0).sum())
        rows.append(
            {
                "ticket_type": ticket_type,
                "runtime_action": action,
                "tickets": int(len(g)),
                "races": int(g["race_id"].nunique()),
                "avg_margin": float(_num(g.get("runtime_odds_margin_ratio"), g.index, np.nan).mean()),
                "avg_hit_prob": float(_num(g.get("ticket_hit_prob"), g.index, np.nan).mean()),
                "stake_yen": stake,
                "return_yen": ret,
                "roi": ret / stake if stake else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply runtime odds decision rules: buy, reduce, wait, or skip.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/min_odds_ticket_prob_gate_v1/min_odds_annotated_ticket_profiles.csv")
    parser.add_argument("--pair-live-csv", default="data/processed/live_odds/realtime_pair_odds_latest.csv")
    parser.add_argument("--single-live-csv", default="data/processed/live_odds/realtime_single_odds_latest.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/runtime_odds_decision_rules_v1")
    parser.add_argument("--no-proxy-when-missing", action="store_true", help="If live odds are missing, mark tickets as WAIT instead of using proxy odds.")
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    pair_live = _read_optional_csv(project_path(args.pair_live_csv))
    single_live = _read_optional_csv(project_path(args.single_live_csv))
    decisions = apply_decisions(tickets, pair_live, single_live, use_proxy_when_missing=not args.no_proxy_when_missing)

    out_dir = ensure_dir(project_path(args.output_dir))
    decisions.to_csv(out_dir / "runtime_ticket_decisions.csv", index=False, encoding="utf-8-sig")
    _decision_summary(decisions).to_csv(out_dir / "runtime_decision_summary.csv", index=False, encoding="utf-8-sig")

    selected = decisions[decisions["runtime_stake_yen"].gt(0)].copy()
    selected.to_csv(out_dir / "runtime_selected_tickets.csv", index=False, encoding="utf-8-sig")
    metrics = [_runtime_metrics(decisions, "runtime_all")]
    if "year" in decisions.columns:
        for year, g in decisions.groupby("year"):
            metrics.append(_runtime_metrics(g, f"runtime_year_{int(year)}"))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "runtime_metrics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "tickets_csv": args.tickets_csv,
        "pair_live_loaded": pair_live is not None and not pair_live.empty,
        "single_live_loaded": single_live is not None and not single_live.empty,
        "use_proxy_when_missing": not args.no_proxy_when_missing,
        "summary": metrics_df.to_dict(orient="records"),
        "action_counts": decisions["runtime_action"].value_counts().to_dict(),
        "note": "Use proxy mode for historical review/dashboard annotation. Use --no-proxy-when-missing in real operation if live odds are required before purchase.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
