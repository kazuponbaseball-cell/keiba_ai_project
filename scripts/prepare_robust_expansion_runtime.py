from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def fill_addon_runtime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    addon = out.get("source_label", "").astype(str).eq("expanded")
    if not addon.any():
        return out

    idx = out[addon].index
    pair_score = num(out.loc[idx].get("pair_score"), idx, 0.72).fillna(0.72)
    pair_q = num(out.loc[idx].get("pair_quinella_score"), idx, 0.50).fillna(0.50)
    overlay = num(out.loc[idx].get("market_overlay_score"), idx, 0.70).fillna(0.70)
    front5 = num(out.loc[idx].get("projected_front5_prob"), idx, 0.50).fillna(0.50)
    anchor_odds = num(out.loc[idx].get("anchor_odds"), idx, np.nan).clip(lower=1.0)
    partner_odds = num(out.loc[idx].get("partner_odds"), idx, np.nan).clip(lower=1.0)

    # Conservative addon probability: centered near the observed addon hit rate
    # and only gently adjusted by pre-race quality signals.
    quality = (
        0.34 * ((pair_score - 0.72) / 0.18).clip(0.0, 1.0)
        + 0.26 * ((pair_q - 0.50) / 0.22).clip(0.0, 1.0)
        + 0.22 * ((overlay - 0.70) / 0.25).clip(0.0, 1.0)
        + 0.18 * ((front5 - 0.50) / 0.30).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    hit_prob = (0.060 + 0.050 * quality).clip(0.045, 0.115)
    quote_pay = (100.0 * (anchor_odds * partner_odds * 0.32).clip(1.3, 260.0)).fillna(0.0)
    required_pay = (100.0 * 1.25 / hit_prob.replace(0, np.nan)).clip(800.0, 6000.0)

    out.loc[idx, "ticket_hit_prob"] = num(out.get("ticket_hit_prob"), out.index, np.nan).loc[idx].fillna(hit_prob)
    out.loc[idx, "quote_pay_proxy_per100"] = num(out.get("quote_pay_proxy_per100"), out.index, np.nan).loc[idx].fillna(quote_pay)
    out.loc[idx, "required_pay_per100"] = num(out.get("required_pay_per100"), out.index, np.nan).loc[idx].fillna(required_pay)
    out.loc[idx, "min_acceptable_odds"] = out.loc[idx, "required_pay_per100"] / 100.0
    out.loc[idx, "quote_odds_proxy"] = out.loc[idx, "quote_pay_proxy_per100"] / 100.0
    out.loc[idx, "min_odds_margin_ratio"] = out.loc[idx, "quote_pay_proxy_per100"] / out.loc[idx, "required_pay_per100"].replace(0, np.nan)
    out.loc[idx, "runtime_expected_roi"] = out.loc[idx, "ticket_hit_prob"] * out.loc[idx, "quote_pay_proxy_per100"] / 100.0
    out.loc[idx, "operation_profile"] = "selective_expand_runtime"
    out.loc[idx, "operation_profile_label"] = "robust_expansion"
    out.loc[idx, "operation_strength_rank"] = 1
    out.loc[idx, "operational_mode"] = "robust_expansion"
    out.loc[idx, "runtime_action"] = "BUY"
    out.loc[idx, "runtime_stake_yen"] = num(out.get("eval_stake_yen"), out.index, 0.0).loc[idx].fillna(num(out.get("stake_yen"), out.index, 0.0).loc[idx])
    out.loc[idx, "runtime_return_yen"] = num(out.get("eval_return_yen"), out.index, 0.0).loc[idx]
    out.loc[idx, "runtime_ticket_status"] = "BUY"
    out.loc[idx, "runtime_reason"] = "robust_expansion_pre_live"
    out.loc[idx, "runtime_pay_source"] = "proxy"
    out.loc[idx, "buy_reason_summary"] = "馬連追加候補: pair/overlay/front条件一致"
    out.loc[idx, "risk_reason_summary"] = "拡張枠のため実オッズ必須"
    out.loc[idx, "stake_adjustment_summary"] = "小額addon"
    out.loc[idx, "dashboard_decision_label"] = "BUY"
    return out


def metrics(df: pd.DataFrame, label: str) -> dict:
    stake = float(num(df.get("runtime_stake_yen"), df.index, 0.0).fillna(0.0).sum())
    ret = float(num(df.get("runtime_return_yen"), df.index, 0.0).fillna(0.0).sum())
    return {
        "label": label,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()) if "race_id" in df.columns else 0,
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "runtime_buy": int(df.get("runtime_action", "").astype(str).isin(["BUY", "REDUCE"]).sum()) if "runtime_action" in df.columns else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare standard plus robust expansion tickets for runtime live-odds decisions.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/selective_expansion_robust_v1/standard_plus_robust_expansion_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/robust_expansion_runtime_ready_v1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tickets = pd.read_csv(args.tickets_csv, dtype={"race_id": str}, low_memory=False)
    prepared = fill_addon_runtime_columns(tickets)
    prepared.to_csv(out_dir / "standard_plus_robust_runtime_ready_tickets.csv", index=False, encoding="utf-8-sig")
    addon = prepared[prepared.get("source_label", "").astype(str).eq("expanded")].copy()
    summary = pd.DataFrame([metrics(prepared, "standard_plus_robust_runtime_ready"), metrics(addon, "robust_addon_runtime_ready")])
    summary.to_csv(out_dir / "runtime_ready_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(out_dir),
        "tickets_csv": args.tickets_csv,
        "summary": summary.to_dict(orient="records"),
        "addon_runtime_columns_filled": int(len(addon)),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
