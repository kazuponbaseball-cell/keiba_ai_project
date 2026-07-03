from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pay_per100(df: pd.DataFrame) -> pd.Series:
    return np.select(
        [
            df["ticket_type"].eq("wide"),
            df["ticket_type"].eq("umaren"),
            df["ticket_type"].eq("umatan_anchor_to_partner"),
            df["ticket_type"].eq("umatan_partner_to_anchor"),
        ],
        [
            _num(df["wide_pay"]).fillna(0.0),
            _num(df["umaren_pay"]).fillna(0.0),
            _num(df["umatan_pay"]).fillna(0.0),
            _num(df["umatan_pay"]).fillna(0.0),
        ],
        default=0.0,
    )


def _max_drawdown_by_race(tickets: pd.DataFrame) -> float:
    by_race = (
        tickets.groupby("race_id", as_index=False)
        .agg(stake_yen=("policy_stake_yen", "sum"), return_yen=("policy_return_yen", "sum"))
        .sort_values("race_id")
    )
    equity = (by_race["return_yen"] - by_race["stake_yen"]).cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _apply_budget_cap(df: pd.DataFrame, budget_yen: int) -> pd.DataFrame:
    rows = []
    for _, g in df.groupby("race_id", sort=False):
        g = g.copy()
        total = int(g["policy_stake_yen"].sum())
        if total > budget_yen:
            scale = budget_yen / total
            g["policy_stake_yen"] = (np.floor(g["policy_stake_yen"] * scale / 100.0) * 100.0).clip(lower=100.0)
            while g["policy_stake_yen"].sum() > budget_yen and len(g) > 0:
                idx = g.sort_values(["policy_stake_yen", "pair_score"], ascending=[False, True]).index[0]
                g.loc[idx, "policy_stake_yen"] = max(0.0, g.loc[idx, "policy_stake_yen"] - 100.0)
            g = g[g["policy_stake_yen"].gt(0)].copy()
        rows.append(g)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _stake_policy(df: pd.DataFrame, policy: str, budget_yen: int) -> pd.DataFrame:
    out = df.copy()
    pair = _num(out["pair_score"]).fillna(0.0)
    front = _num(out["projected_front5_prob"]).fillna(0.0)
    umaren_pay = _num(out["umaren_pay"]).fillna(0.0)

    if policy == "original":
        stake = _num(out["stake_yen"]).fillna(100.0)
    elif policy == "flat100":
        stake = pd.Series(100.0, index=out.index)
    elif policy == "hit_balanced":
        stake = pd.Series(100.0, index=out.index)
        stake = stake.where(~out["ticket_type"].eq("wide"), np.where((pair >= 0.70) & (front >= 0.55), 300.0, 200.0))
        stake = stake.where(~out["ticket_type"].eq("umaren"), 100.0)
    elif policy == "roi_guarded":
        stake = pd.Series(100.0, index=out.index)
        stake = stake.where(~out["ticket_type"].eq("wide"), np.where((pair >= 0.78) & (front >= 0.60), 300.0, 100.0))
        stake = stake.where(
            ~out["ticket_type"].eq("umaren"),
            np.select([(pair >= 0.82) & (umaren_pay >= 2500.0), umaren_pay >= 1200.0], [200.0, 100.0], default=0.0),
        )
    elif policy == "strong_only":
        stake = pd.Series(0.0, index=out.index)
        stake = stake.where(~(out["ticket_type"].eq("wide") & (pair >= 0.70) & (front >= 0.50)), 200.0)
        stake = stake.where(~(out["ticket_type"].eq("umaren") & (pair >= 0.78) & (umaren_pay >= 1800.0)), 100.0)
    else:
        raise ValueError(policy)

    out["policy_stake_yen"] = stake
    out = out[out["policy_stake_yen"].gt(0)].copy()
    out = _apply_budget_cap(out, budget_yen)
    out["pay_per100"] = _pay_per100(out)
    out["policy_return_yen"] = np.where(out["hit"].astype(bool), out["pay_per100"], 0.0) * out["policy_stake_yen"] / 100.0
    out["stake_policy"] = policy
    out["budget_yen"] = budget_yen
    return out


def _metrics(tickets: pd.DataFrame, policy: str, budget_yen: int) -> dict:
    if tickets.empty:
        return {"stake_policy": policy, "budget_yen": budget_yen, "tickets": 0, "races": 0, "roi": 0.0}
    stake = float(tickets["policy_stake_yen"].sum())
    ret = float(tickets["policy_return_yen"].sum())
    by_race = tickets.groupby("race_id").agg(
        stake_yen=("policy_stake_yen", "sum"),
        return_yen=("policy_return_yen", "sum"),
        hit=("hit", "max"),
    )
    return {
        "stake_policy": policy,
        "budget_yen": budget_yen,
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "avg_stake_per_race": float(by_race["stake_yen"].mean()),
        "ticket_hit_rate": float(tickets["hit"].mean()),
        "race_hit_rate": float(by_race["hit"].mean()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "max_drawdown_yen": _max_drawdown_by_race(tickets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stake sizing policies for dynamic pair tickets.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/dynamic_pair_ticket_allocation_strict_addon_v1/walkforward_selected_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/dynamic_pair_stake_sizing_v1")
    parser.add_argument("--budgets", nargs="*", type=int, default=[5000, 10000])
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), low_memory=False)
    policies = ["original", "flat100", "hit_balanced", "roi_guarded", "strong_only"]
    metric_rows = []
    yearly_rows = []
    frames = []
    for budget in args.budgets:
        for policy in policies:
            sized = _stake_policy(tickets, policy, budget)
            metric_rows.append(_metrics(sized, policy, budget))
            for year, g in sized.groupby("test_year"):
                row = _metrics(g, policy, budget)
                row["test_year"] = int(year)
                yearly_rows.append(row)
            frames.append(sized)

    out_dir = ensure_dir(project_path(args.output_dir))
    summary = pd.DataFrame(metric_rows).sort_values(["roi", "race_hit_rate"], ascending=[False, False])
    yearly = pd.DataFrame(yearly_rows).sort_values(["stake_policy", "budget_yen", "test_year"])
    all_tickets = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    summary.to_csv(out_dir / "stake_sizing_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "stake_sizing_yearly.csv", index=False, encoding="utf-8-sig")
    all_tickets.to_csv(out_dir / "stake_sizing_tickets.csv", index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(out_dir),
        "summary": summary.to_dict(orient="records"),
        "yearly": yearly.to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
