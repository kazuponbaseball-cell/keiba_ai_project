from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_expansion_roi_dilution import max_drawdown, normalize, num


def metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "label": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
        }
    race = (
        df.groupby("race_id", sort=False)
        .agg(stake_yen=("eval_stake_yen", "sum"), return_yen=("eval_return_yen", "sum"), hit=("hit_eval", "max"))
        .reset_index()
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    stake = float(df["eval_stake_yen"].sum())
    ret = float(df["eval_return_yen"].sum())
    return {
        "label": label,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(df["hit_eval"].mean()),
        "race_hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "max_drawdown_yen": max_drawdown(race["profit_yen"]),
    }


def top_removed(df: pd.DataFrame, top_n: int) -> dict:
    race = (
        df.groupby("race_id", sort=False)
        .agg(stake_yen=("eval_stake_yen", "sum"), return_yen=("eval_return_yen", "sum"))
        .reset_index()
    )
    if race.empty:
        return metrics(df, f"minus_top{top_n}")
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    kept = set(race.sort_values("profit_yen", ascending=False).iloc[top_n:]["race_id"])
    return metrics(df[df["race_id"].isin(kept)].copy(), f"minus_top{top_n}")


def add_policy_scores(extra: pd.DataFrame) -> pd.DataFrame:
    out = extra.copy()
    out["selective_policy"] = "skip"
    out["selective_reason"] = "extra_range_not_supported"
    out["selective_stake_yen"] = 0.0

    umaren = out["ticket_type"].eq("umaren")
    anchor_pop = num(out.get("anchor_pop"), out.index, np.nan)
    partner_pop = num(out.get("partner_pop"), out.index, np.nan)
    partner_odds = num(out.get("partner_odds"), out.index, np.nan)
    overlay = num(out.get("market_overlay_score"), out.index, 0.0).fillna(0.0)
    front5 = num(out.get("projected_front5_prob"), out.index, 0.0).fillna(0.0)
    pair_score = num(out.get("pair_score"), out.index, 0.0).fillna(0.0)
    pair_q = num(out.get("pair_quinella_score"), out.index, 0.0).fillna(0.0)
    anchor_danger = num(out.get("anchor_danger"), out.index, 0.0).fillna(0.0)
    partner_danger = num(out.get("partner_danger"), out.index, 0.0).fillna(0.0)
    total_danger = anchor_danger + partner_danger

    umaren_a = (
        umaren
        & anchor_pop.eq(3)
        & pair_score.ge(0.80)
        & front5.ge(0.60)
        & overlay.ge(0.70)
        & partner_odds.between(5.0, 80.0)
        & total_danger.lt(0.70)
    )
    umaren_b = (
        umaren
        & partner_pop.ge(10)
        & overlay.ge(0.85)
        & front5.between(0.45, 0.75)
        & pair_q.ge(0.50)
        & partner_odds.between(20.0, 160.0)
        & total_danger.lt(0.70)
    )
    umaren_c = (
        umaren
        & out["venue_eval"].eq("Niigata")
        & pair_score.ge(0.70)
        & overlay.ge(0.55)
        & partner_odds.between(5.0, 80.0)
        & total_danger.lt(0.70)
    )

    keep = umaren_a | umaren_b | umaren_c
    out.loc[umaren_a, "selective_policy"] = "umaren_anchor3_front_value"
    out.loc[umaren_a, "selective_reason"] = "anchor_pop3 + high pair/front/overlay"
    out.loc[umaren_b, "selective_policy"] = "umaren_extreme_partner10_value"
    out.loc[umaren_b, "selective_reason"] = "partner10plus + extreme overlay + usable front"
    out.loc[umaren_c, "selective_policy"] = "umaren_niigata_value"
    out.loc[umaren_c, "selective_reason"] = "Niigata umaren value segment"

    raw_stake = np.select(
        [umaren_a, umaren_b, umaren_c],
        [400.0, 300.0, 300.0],
        default=0.0,
    )
    out["selective_stake_yen"] = raw_stake
    out = out[keep].copy()
    out["eval_stake_yen"] = out["selective_stake_yen"]
    out["eval_return_yen"] = np.where(out["hit_eval"], num(out.get("umaren_pay"), out.index, 0.0).fillna(0.0) * out["eval_stake_yen"] / 100.0, 0.0)
    out["eval_profit_yen"] = out["eval_return_yen"] - out["eval_stake_yen"]
    out["stake_yen"] = out["eval_stake_yen"]
    out["return_yen"] = out["eval_return_yen"]
    out["operation_profile"] = "selective_expand"
    out["operation_profile_label"] = "selective_expand"
    out["operation_strength_rank"] = 1
    out["operational_mode"] = "selective_expansion"
    out["runtime_action"] = "BUY"
    out["runtime_stake_yen"] = out["eval_stake_yen"]
    out["runtime_return_yen"] = out["eval_return_yen"]
    out["runtime_ticket_status"] = "BUY"
    out["buy_reason_summary"] = out["selective_reason"]
    out["risk_reason_summary"] = "expanded range, use as addon only"
    out["stake_adjustment_summary"] = "small addon stake"
    out["dashboard_decision_label"] = "BUY"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a selective expansion addon from expanded tickets not in the final standard policy.")
    parser.add_argument("--expanded-csv", default="outputs/analysis/extended_period_validation_v1/fixed_proxy_selected_tickets_2024_2026.csv")
    parser.add_argument("--standard-csv", default="outputs/analysis/final_operational_quality_v1/standard_explained_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/selective_expansion_policy_v1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expanded = normalize(pd.read_csv(args.expanded_csv, dtype={"race_id": str}, low_memory=False), "expanded")
    standard = normalize(pd.read_csv(args.standard_csv, dtype={"race_id": str}, low_memory=False), "standard")
    extra = expanded[~expanded["ticket_key_eval"].isin(set(standard["ticket_key_eval"]))].copy()
    addon = add_policy_scores(extra)

    combined = pd.concat([standard, addon], ignore_index=True, sort=False)
    addon.to_csv(out_dir / "selective_expansion_addon_tickets.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(out_dir / "standard_plus_selective_expansion_tickets.csv", index=False, encoding="utf-8-sig")

    rows = [
        metrics(standard, "standard"),
        metrics(addon, "selective_expansion_addon"),
        metrics(combined, "standard_plus_selective_expansion"),
        top_removed(standard, 10) | {"label": "standard_minus_top10"},
        top_removed(addon, 10) | {"label": "addon_minus_top10"},
        top_removed(combined, 10) | {"label": "combined_minus_top10"},
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "selective_expansion_summary.csv", index=False, encoding="utf-8-sig")

    by_policy = pd.DataFrame([metrics(g, str(policy)) | {"selective_policy": policy} for policy, g in addon.groupby("selective_policy")])
    by_policy.to_csv(out_dir / "addon_by_policy.csv", index=False, encoding="utf-8-sig")
    by_year = pd.DataFrame([metrics(g, f"year_{year}") | {"year": year} for year, g in addon.groupby("year")])
    by_year.to_csv(out_dir / "addon_by_year.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "summary": summary.to_dict(orient="records"),
        "addon_by_policy": by_policy.to_dict(orient="records"),
        "addon_by_year": by_year.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
