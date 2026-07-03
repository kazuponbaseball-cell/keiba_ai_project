from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={"race_id": str}, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False)


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def bool_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    raw = frame[col].astype(str).str.lower()
    numeric = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return raw.isin(["true", "1", "1.0", "yes", "y"]) | numeric.gt(0)


def race_date_from_id(race_id: pd.Series) -> pd.Series:
    digits = race_id.astype(str).str.extract(r"(\d{8})", expand=False)
    return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")


def prepare_tickets(path: Path) -> pd.DataFrame:
    raw = read_csv_safe(path)
    if raw.empty:
        raise SystemExit(f"No ticket rows found: {path}")
    df = raw.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["ticket_type"] = df.get("ticket_type", "").astype(str)
    stake = num(df, "runtime_stake_yen", 0.0)
    if stake.le(0).all():
        stake = num(df, "stake_yen", 0.0)
    df = df[stake.gt(0)].copy()
    df["_current_stake"] = stake.loc[df.index].astype(float)
    df["_date"] = race_date_from_id(df["race_id"])
    df["_year"] = df["_date"].dt.year
    df["_hit"] = bool_series(df, "hit")

    pay = pd.Series(np.nan, index=df.index, dtype=float)
    if "runtime_backtest_pay_per100" in df.columns:
        pay = pay.fillna(num(df, "runtime_backtest_pay_per100"))
    if "runtime_pay_per100" in df.columns:
        pay = pay.fillna(num(df, "runtime_pay_per100"))
    pay = pay.where(pay.gt(0))
    pay_fallback = pd.Series(
        np.select(
            [
                df["ticket_type"].eq("umaren"),
                df["ticket_type"].eq("wide"),
                df["ticket_type"].isin(["umatan_anchor_to_partner", "umatan_partner_to_anchor", "umatan"]),
            ],
            [
                num(df, "umaren_pay", 0.0),
                num(df, "wide_pay", 0.0),
                num(df, "umatan_pay", 0.0),
            ],
            default=0.0,
        ),
        index=df.index,
    )
    pay = pay.fillna(pay_fallback)
    odds = num(df, "runtime_odds")
    pay = pay.where(pay.gt(0), odds * 100.0)
    df["_pay_per100"] = pay.fillna(0.0)
    df["_decimal_odds"] = df["_pay_per100"] / 100.0

    prob = pd.Series(np.nan, index=df.index, dtype=float)
    if "ticket_type" in df.columns:
        prob = prob.where(~df["ticket_type"].eq("umaren"), num(df, "umaren_hit_prob_cal"))
        prob = prob.where(~df["ticket_type"].eq("wide"), num(df, "wide_hit_prob_cal"))
    prob = prob.fillna(num(df, "pair_calibrated_hit_prob"))
    prob = prob.fillna(num(df, "ticket_hit_prob"))
    df["_model_prob"] = prob.clip(0.0001, 0.95)
    df["_model_ev"] = df["_model_prob"] * df["_decimal_odds"]
    df["_break_even_prob"] = 1.0 / df["_decimal_odds"].replace(0, np.nan)
    df["_edge_prob"] = df["_model_prob"] - df["_break_even_prob"]
    df["_kelly_fraction"] = (
        (df["_decimal_odds"] * df["_model_prob"] - 1.0)
        / (df["_decimal_odds"] - 1.0).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=0.25)
    df["_actual_return_yen_current"] = np.where(df["_hit"], df["_pay_per100"] * df["_current_stake"] / 100.0, 0.0)
    return df[df["_decimal_odds"].gt(1.0)].copy()


def max_drawdown(race: pd.DataFrame) -> float:
    if race.empty:
        return 0.0
    equity = race["profit_yen"].cumsum()
    peak = equity.cummax().clip(lower=0.0)
    return float((equity - peak).min())


def apply_stake_policy(df: pd.DataFrame, policy: str, budget_per_race: int) -> pd.DataFrame:
    out = df.copy()
    if policy == "current_runtime":
        stake = out["_current_stake"]
    elif policy == "flat100":
        stake = pd.Series(100.0, index=out.index)
    elif policy == "flat500":
        stake = pd.Series(500.0, index=out.index)
    elif policy == "quarter_kelly":
        stake = np.floor((budget_per_race * out["_kelly_fraction"] * 0.25) / 100.0) * 100.0
    elif policy == "half_kelly":
        stake = np.floor((budget_per_race * out["_kelly_fraction"] * 0.50) / 100.0) * 100.0
    elif policy == "edge_scaled":
        edge = (out["_model_ev"] - 1.0).clip(lower=0.0, upper=4.0)
        stake = 100.0 + np.floor(edge * 2.0) * 100.0
    else:
        raise ValueError(f"unknown stake policy: {policy}")

    out["policy_stake_yen"] = pd.Series(stake, index=out.index).fillna(0.0).clip(lower=0.0, upper=budget_per_race)
    if policy in {"quarter_kelly", "half_kelly"}:
        out.loc[out["policy_stake_yen"].lt(100.0) & out["_model_ev"].gt(1.0), "policy_stake_yen"] = 100.0
    rows: list[pd.DataFrame] = []
    for _, race in out.groupby("race_id", sort=False):
        part = race.copy()
        total = float(part["policy_stake_yen"].sum())
        if total > budget_per_race:
            scale = budget_per_race / total
            part["policy_stake_yen"] = np.floor(part["policy_stake_yen"] * scale / 100.0) * 100.0
            while part["policy_stake_yen"].sum() > budget_per_race and len(part):
                idx = part.sort_values(["policy_stake_yen", "_model_ev"], ascending=[False, True]).index[0]
                part.loc[idx, "policy_stake_yen"] = max(0.0, part.loc[idx, "policy_stake_yen"] - 100.0)
        rows.append(part)
    out = pd.concat(rows, ignore_index=False, sort=False) if rows else out
    out = out[out["policy_stake_yen"].gt(0)].copy()
    out["policy_return_yen"] = np.where(out["_hit"], out["_pay_per100"] * out["policy_stake_yen"] / 100.0, 0.0)
    out["policy_profit_yen"] = out["policy_return_yen"] - out["policy_stake_yen"]
    out["stake_policy"] = policy
    out["budget_per_race"] = budget_per_race
    return out


def metrics(rows: pd.DataFrame, label: str) -> dict[str, Any]:
    if rows.empty:
        return {
            "label": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
        }
    race = (
        rows.groupby("race_id", sort=False)
        .agg(
            date=("_date", "min"),
            stake_yen=("policy_stake_yen", "sum"),
            return_yen=("policy_return_yen", "sum"),
            hit=("policy_return_yen", lambda s: bool((s > 0).any())),
        )
        .reset_index()
        .sort_values(["date", "race_id"])
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    stake = float(race["stake_yen"].sum())
    ret = float(race["return_yen"].sum())
    top_returns = race["return_yen"].sort_values(ascending=False)
    ex_top5 = race.copy()
    ex_top10 = race.copy()
    ex_top5.loc[top_returns.index[:5], "return_yen"] = 0.0
    ex_top10.loc[top_returns.index[:10], "return_yen"] = 0.0
    return {
        "label": label,
        "tickets": int(len(rows)),
        "races": int(rows["race_id"].nunique()),
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(rows["_hit"].mean()),
        "race_hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "avg_stake_per_race": float(race["stake_yen"].mean()) if len(race) else 0.0,
        "max_drawdown_yen": max_drawdown(race),
        "top5_removed_roi": float(ex_top5["return_yen"].sum() / stake) if stake else 0.0,
        "top10_removed_roi": float(ex_top10["return_yen"].sum() / stake) if stake else 0.0,
    }


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prob_bin"] = pd.cut(
        out["_model_prob"],
        bins=[0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 1.0],
        include_lowest=True,
    ).astype(str)
    rows: list[dict[str, Any]] = []
    for key, group in out.groupby(["ticket_type", "prob_bin"], dropna=False):
        ticket_type, prob_bin = key
        if group.empty:
            continue
        rows.append(
            {
                "ticket_type": ticket_type,
                "prob_bin": prob_bin,
                "tickets": int(len(group)),
                "races": int(group["race_id"].nunique()),
                "avg_model_prob": float(group["_model_prob"].mean()),
                "actual_hit_rate": float(group["_hit"].mean()),
                "brier": float(((group["_model_prob"] - group["_hit"].astype(float)) ** 2).mean()),
                "avg_decimal_odds": float(group["_decimal_odds"].mean()),
                "avg_model_ev": float(group["_model_ev"].mean()),
                "flat100_roi": float((np.where(group["_hit"], group["_pay_per100"], 0.0).sum()) / (len(group) * 100.0)),
            }
        )
    return pd.DataFrame(rows)


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
    parser = argparse.ArgumentParser(description="Analyze probability calibration and fractional Kelly stake sizing.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates_default/recommended_runtime_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/calibration_fractional_kelly_v1")
    parser.add_argument("--budgets", nargs="*", type=int, default=[3000, 5000, 10000])
    args = parser.parse_args()

    tickets = prepare_tickets(project_path(args.tickets_csv))
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration_df = calibration(tickets)
    policy_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    policies = ["current_runtime", "flat100", "flat500", "edge_scaled", "quarter_kelly", "half_kelly"]
    for budget in args.budgets:
        for policy in policies:
            sized = apply_stake_policy(tickets, policy, budget)
            metric = metrics(sized, f"{policy}_budget{budget}")
            metric["stake_policy"] = policy
            metric["budget_per_race"] = budget
            metric_rows.append(metric)
            for year, part in sized.groupby("_year", dropna=False):
                y = metrics(part, f"{policy}_budget{budget}_{year}")
                y["stake_policy"] = policy
                y["budget_per_race"] = budget
                y["year"] = int(year) if pd.notna(year) else None
                yearly_rows.append(y)
            policy_frames.append(sized)

    policy_summary = pd.DataFrame(metric_rows).sort_values(["top10_removed_roi", "roi"], ascending=[False, False])
    yearly = pd.DataFrame(yearly_rows).sort_values(["stake_policy", "budget_per_race", "year"])
    all_sized = pd.concat(policy_frames, ignore_index=True, sort=False) if policy_frames else pd.DataFrame()

    tickets.to_csv(out_dir / "prepared_tickets.csv", index=False, encoding="utf-8-sig")
    calibration_df.to_csv(out_dir / "probability_calibration.csv", index=False, encoding="utf-8-sig")
    policy_summary.to_csv(out_dir / "stake_policy_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "stake_policy_yearly.csv", index=False, encoding="utf-8-sig")
    all_sized.to_csv(out_dir / "stake_policy_tickets.csv", index=False, encoding="utf-8-sig")

    payload = {
        "tickets_csv": str(project_path(args.tickets_csv)),
        "output_dir": str(out_dir),
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "date_min": tickets["_date"].min().strftime("%Y-%m-%d") if tickets["_date"].notna().any() else None,
        "date_max": tickets["_date"].max().strftime("%Y-%m-%d") if tickets["_date"].notna().any() else None,
        "overall_brier": float(((tickets["_model_prob"] - tickets["_hit"].astype(float)) ** 2).mean()),
        "overall_avg_model_prob": float(tickets["_model_prob"].mean()),
        "overall_actual_hit_rate": float(tickets["_hit"].mean()),
        "top_stake_policies": policy_summary.head(12).to_dict(orient="records"),
        "note": "Stake policies do not create edge; use this only to compare drawdown and capital efficiency after selection is fixed.",
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))
    print(policy_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
