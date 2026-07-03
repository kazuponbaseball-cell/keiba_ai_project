from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def parse_dates(df: pd.DataFrame) -> pd.Series:
    if "date_key" in df.columns:
        parsed = pd.to_datetime(df["date_key"], errors="coerce")
    elif "日付S" in df.columns:
        parsed = pd.to_datetime(df["日付S"], errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=df.index)
    missing = parsed.isna()
    if missing.any() and "race_id" in df.columns:
        race_digits = df.loc[missing, "race_id"].astype(str).str.extract(r"(\d{8})", expand=False)
        parsed.loc[missing] = pd.to_datetime(race_digits, format="%Y%m%d", errors="coerce")
    return parsed


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["_date"] = parse_dates(out)
    if "year" not in out.columns:
        out["year"] = out["_date"].dt.year
    out["year"] = num(out, "year", np.nan).astype("Int64")
    out["_base_stake"] = num(out, "runtime_stake_yen", 0.0).fillna(0.0)
    out["_base_return"] = num(out, "runtime_return_yen", 0.0).fillna(0.0)
    stake_positive = out["_base_stake"].gt(0)
    derived_pay = np.where(stake_positive, out["_base_return"] / out["_base_stake"] * 100.0, np.nan)
    # Prefer realized runtime return so the no-overlay baseline exactly matches
    # the existing backtest. Fall back to stored/proxy pay only when the row is
    # marked as hit but the realized return is unavailable.
    out["_base_pay_per100"] = pd.Series(derived_pay, index=out.index)
    out["_base_pay_per100"] = out["_base_pay_per100"].fillna(num(out, "runtime_backtest_pay_per100", np.nan))
    out["_base_pay_per100"] = out["_base_pay_per100"].fillna(num(out, "quote_pay_proxy_per100", np.nan)).fillna(0.0)
    out["_hit_bool"] = out["_base_return"].gt(0) | num(out, "hit", 0.0).fillna(0.0).astype(bool)
    out["_margin"] = num(out, "min_odds_margin_ratio", 0.0).fillna(0.0)
    out["_expected_roi"] = num(out, "runtime_expected_roi", np.nan)
    out["_expected_roi"] = out["_expected_roi"].fillna(num(out, "expected_roi_after_slippage", 0.0)).fillna(0.0)
    out["_hit_prob"] = num(out, "ticket_hit_prob", 0.0).fillna(0.0)
    out["_overlay"] = num(out, "market_overlay_score", 0.0).fillna(0.0)
    out["_front5"] = num(out, "projected_front5_prob", 0.0).fillna(0.0)
    out["_pair_score"] = num(out, "pair_score", 0.0).fillna(0.0)
    out["_pair_q"] = num(out, "pair_quinella_score", 0.0).fillna(0.0)
    out["_late_value"] = num(out, "late_value_survives_score", 0.0).fillna(0.0)
    out["_priority_net"] = num(out, "priority_context_net_score", 0.0).fillna(0.0)
    out["_ticket_quality"] = num(out, "b_ticket_quality_score", 0.0).fillna(0.0)
    danger = num(out, "danger_sum", np.nan)
    danger = danger.fillna(num(out, "anchor_danger", 0.0).fillna(0.0) + num(out, "partner_danger", 0.0).fillna(0.0))
    out["_danger_sum"] = danger.fillna(0.0)
    out["_danger_gate_score"] = num(out, "ticket_danger_popular_score", np.nan)
    out["_danger_gate_score"] = out["_danger_gate_score"].fillna(out["_danger_sum"]).fillna(0.0)
    out["_difficulty"] = num(out, "race_difficulty_score", np.nan)
    out["_difficulty"] = out["_difficulty"].fillna(num(out, "race_difficulty_model_score", np.nan)).fillna(num(out, "difficulty", 0.0)).fillna(0.0)
    out["_skip_risk"] = num(out, "skip_risk_score", np.nan)
    out["_skip_risk"] = out["_skip_risk"].fillna(num(out, "skip_risk", np.nan)).fillna(out["_difficulty"]).fillna(0.0)
    if "ticket_type" not in out.columns:
        out["ticket_type"] = "unknown"
    if "source_label" not in out.columns:
        out["source_label"] = "unknown"
    return out[out["_base_stake"].gt(0)].copy()


def round_stake(series: pd.Series) -> pd.Series:
    return (np.floor(series.clip(lower=0.0) / 100.0) * 100.0).where(series.gt(0), 0.0)


def reprice(df: pd.DataFrame, stake: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["pre_mcs_pbo_stake_yen"] = out["_base_stake"]
    out["runtime_stake_yen"] = round_stake(stake).astype(float)
    out["runtime_return_yen"] = np.where(out["_hit_bool"], out["_base_pay_per100"] * out["runtime_stake_yen"] / 100.0, 0.0)
    out["runtime_action"] = np.where(out["runtime_stake_yen"].gt(0), "BUY", "SKIP")
    out["runtime_ticket_status"] = np.where(out["runtime_stake_yen"].gt(0), "買い", "見送り")
    out["runtime_reason"] = out.get("runtime_reason", "").astype(str) + "|mcs_pbo_survivor_overlay"
    return out


def race_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["race_id", "date", "stake_yen", "return_yen", "profit_yen", "hit"])
    race = (
        df.groupby("race_id", sort=False)
        .agg(date=("_date", "min"), stake_yen=("runtime_stake_yen", "sum"), return_yen=("runtime_return_yen", "sum"))
        .reset_index()
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    race["hit"] = race["return_yen"].gt(0)
    return race.sort_values(["date", "race_id"])


def max_drawdown(race: pd.DataFrame) -> float:
    if race.empty:
        return 0.0
    equity = race["profit_yen"].cumsum()
    peak = equity.cummax().clip(lower=0.0)
    return float((equity - peak).min())


def metrics(df: pd.DataFrame, label: str) -> dict:
    selected = df[df["runtime_stake_yen"].gt(0)].copy()
    if selected.empty:
        return {
            "policy": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
            "top5_removed_roi": 0.0,
            "top10_removed_roi": 0.0,
            "umaren_tickets": 0,
            "wide_tickets": 0,
            "win_tickets": 0,
        }
    race = race_table(selected)
    stake = float(selected["runtime_stake_yen"].sum())
    ret = float(selected["runtime_return_yen"].sum())

    def removed_roi(n: int) -> float:
        if len(race) <= n:
            return 0.0
        kept = race.sort_values("profit_yen", ascending=False).iloc[n:]
        kept_stake = float(kept["stake_yen"].sum())
        return float(kept["return_yen"].sum() / kept_stake) if kept_stake else 0.0

    counts = selected["ticket_type"].value_counts()
    return {
        "policy": label,
        "tickets": int(len(selected)),
        "races": int(selected["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(selected["runtime_return_yen"].gt(0).mean()),
        "race_hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "max_drawdown_yen": max_drawdown(race),
        "top5_removed_roi": removed_roi(5),
        "top10_removed_roi": removed_roi(10),
        "umaren_tickets": int(counts.get("umaren", 0)),
        "wide_tickets": int(counts.get("wide", 0)),
        "win_tickets": int(counts.get("win", 0)),
    }


@dataclass(frozen=True)
class Policy:
    name: str
    description: str
    stake_fn: Callable[[pd.DataFrame], pd.Series]


def policies() -> list[Policy]:
    def base(df: pd.DataFrame) -> pd.Series:
        return df["_base_stake"]

    def reduce_wide_win_50(df: pd.DataFrame) -> pd.Series:
        stake = df["_base_stake"].copy()
        stake.loc[df["ticket_type"].isin(["wide", "win"])] *= 0.5
        return stake

    def skip_wide_win(df: pd.DataFrame) -> pd.Series:
        return df["_base_stake"].where(df["ticket_type"].eq("umaren"), 0.0)

    def s0304(df: pd.DataFrame) -> pd.Series:
        mask = df["ticket_type"].eq("umaren") & df["_margin"].ge(0.95)
        return df["_base_stake"].where(mask, 0.0)

    def s0304_danger020(df: pd.DataFrame) -> pd.Series:
        mask = df["ticket_type"].eq("umaren") & df["_margin"].ge(0.95) & df["_danger_gate_score"].le(0.20)
        return df["_base_stake"].where(mask, 0.0)

    def s0304_skip03119(df: pd.DataFrame) -> pd.Series:
        mask = df["ticket_type"].eq("umaren") & df["_margin"].ge(0.95) & df["_skip_risk"].le(0.3119)
        return df["_base_stake"].where(mask, 0.0)

    def s0304_danger020_skip03119(df: pd.DataFrame) -> pd.Series:
        mask = (
            df["ticket_type"].eq("umaren")
            & df["_margin"].ge(0.95)
            & df["_danger_gate_score"].le(0.20)
            & df["_skip_risk"].le(0.3119)
        )
        return df["_base_stake"].where(mask, 0.0)

    def s0305(df: pd.DataFrame) -> pd.Series:
        mask = df["ticket_type"].eq("umaren") & df["_margin"].ge(0.95) & df["_hit_prob"].ge(0.08)
        return df["_base_stake"].where(mask, 0.0)

    def s0313(df: pd.DataFrame) -> pd.Series:
        mask = (
            df["ticket_type"].eq("umaren")
            & df["_margin"].ge(0.95)
            & df["_expected_roi"].ge(1.35)
            & df["_hit_prob"].ge(0.08)
        )
        return df["_base_stake"].where(mask, 0.0)

    def s0078(df: pd.DataFrame) -> pd.Series:
        mask = df["ticket_type"].eq("umaren") & df["_margin"].ge(1.0)
        return df["_base_stake"].where(mask, 0.0)

    def s0082(df: pd.DataFrame) -> pd.Series:
        mask = df["ticket_type"].eq("umaren") & df["_margin"].ge(1.0) & df["_expected_roi"].ge(1.35)
        return df["_base_stake"].where(mask, 0.0)

    def pbo_pair_quality(df: pd.DataFrame) -> pd.Series:
        mask = (
            df["ticket_type"].eq("umaren")
            & df["_margin"].ge(1.5)
            & df["_expected_roi"].ge(1.10)
            & df["_pair_score"].ge(0.72)
            & df["_pair_q"].ge(0.62)
        )
        return df["_base_stake"].where(mask, 0.0)

    def pbo_front_overlay(df: pd.DataFrame) -> pd.Series:
        mask = (
            df["ticket_type"].eq("umaren")
            & df["_margin"].ge(1.5)
            & df["_expected_roi"].ge(1.10)
            & df["_overlay"].ge(0.82)
            & df["_front5"].ge(0.65)
            & df["_danger_sum"].le(0.65)
        )
        return df["_base_stake"].where(mask, 0.0)

    def pbo_quality_union(df: pd.DataFrame) -> pd.Series:
        pair = pbo_pair_quality(df).gt(0)
        front = pbo_front_overlay(df).gt(0)
        quality = (
            df["ticket_type"].eq("umaren")
            & df["_margin"].ge(1.5)
            & df["_expected_roi"].ge(1.10)
            & df["_priority_net"].ge(0.0)
            & df["_ticket_quality"].ge(0.52)
            & df["_difficulty"].le(0.65)
        )
        return df["_base_stake"].where(pair | front | quality, 0.0)

    def hybrid_mcs_pbo_boost(df: pd.DataFrame) -> pd.Series:
        core = df["ticket_type"].eq("umaren") & df["_margin"].ge(1.0)
        strict = pbo_quality_union(df).gt(0)
        stake = df["_base_stake"].where(core, 0.0)
        stake.loc[strict] = (stake.loc[strict] * 1.20).clip(upper=3000.0)
        return stake

    return [
        Policy("mcs_full_margin095_s0304", "Full MCS: umaren + margin>=0.95", s0304),
        Policy("mcs_full_margin095_s0304_danger020", "Full MCS + danger popular gate <=0.20", s0304_danger020),
        Policy("mcs_full_margin095_s0304_skip03119", "Full MCS + skip risk <=0.3119", s0304_skip03119),
        Policy(
            "mcs_full_margin095_s0304_danger020_skip03119",
            "Full MCS + danger popular gate <=0.20 + skip risk <=0.3119",
            s0304_danger020_skip03119,
        ),
        Policy("mcs_full_margin095_hit08_s0305", "Full MCS: umaren + margin>=0.95 + hit_prob>=0.08", s0305),
        Policy("mcs_full_margin095_ev135_hit08_s0313", "Full MCS: umaren + margin>=0.95 + expected_roi>=1.35 + hit_prob>=0.08", s0313),
        Policy("baseline_all", "現行チケットをそのまま", base),
        Policy("reduce_wide_win_50", "ワイド・単勝を50%減額", reduce_wide_win_50),
        Policy("skip_wide_win_s0072", "MCS代表: 馬連のみ", skip_wide_win),
        Policy("mcs_margin_s0078", "MCS代表: 馬連 + margin>=1.0", s0078),
        Policy("mcs_margin_ev_s0082", "MCS代表: 馬連 + margin>=1.0 + 期待ROI>=1.35", s0082),
        Policy("pbo_pair_quality_s0402", "PBO型: 馬連 + margin>=1.5 + pair quality", pbo_pair_quality),
        Policy("pbo_front_overlay_s0334", "PBO型: 馬連 + 前目/妙味/低危険", pbo_front_overlay),
        Policy("pbo_quality_union", "PBO型3条件のunion", pbo_quality_union),
        Policy("hybrid_mcs_core_pbo_boost", "MCS馬連core + PBO強条件20%増額", hybrid_mcs_pbo_boost),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply and compare MCS/PBO survivor overlays on runtime tickets.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/pair_probability_runtime_v1/pair_calibrated_runtime_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/mcs_pbo_runtime_overlay_v1")
    parser.add_argument("--selected-policy", default="mcs_full_margin095_s0304_skip03119")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tickets = prepare(pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False))

    rows = []
    all_frames: dict[str, pd.DataFrame] = {}
    selected_frames: dict[str, pd.DataFrame] = {}
    by_year_rows = []
    for policy in policies():
        overlaid = reprice(tickets, policy.stake_fn(tickets))
        overlaid["mcs_pbo_policy"] = policy.name
        overlaid["mcs_pbo_policy_description"] = policy.description
        selected = overlaid[overlaid["runtime_stake_yen"].gt(0)].copy()
        all_frames[policy.name] = overlaid
        selected_frames[policy.name] = selected
        rows.append(metrics(overlaid, policy.name) | {"description": policy.description})
        for year, group in overlaid.groupby("year", dropna=True):
            by_year_rows.append(metrics(group, f"{policy.name}_{int(year)}") | {"policy_name": policy.name, "year": int(year)})

    comparison = pd.DataFrame(rows).sort_values(["top10_removed_roi", "top5_removed_roi", "roi"], ascending=[False, False, False])
    comparison.to_csv(out_dir / "policy_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(by_year_rows).to_csv(out_dir / "policy_by_year.csv", index=False, encoding="utf-8-sig")

    for name, selected in selected_frames.items():
        all_frames[name].to_csv(out_dir / f"{name}_all_tickets.csv", index=False, encoding="utf-8-sig")
        selected.to_csv(out_dir / f"{name}_selected_tickets.csv", index=False, encoding="utf-8-sig")

    selected_name = args.selected_policy if args.selected_policy in selected_frames else str(comparison.iloc[0]["policy"])
    all_frames[selected_name].to_csv(out_dir / "recommended_all_tickets.csv", index=False, encoding="utf-8-sig")
    selected_frames[selected_name].to_csv(out_dir / "recommended_runtime_tickets.csv", index=False, encoding="utf-8-sig")
    summary = {
        "tickets_csv": args.tickets_csv,
        "output_dir": str(out_dir),
        "selected_policy": selected_name,
        "recommended_all_tickets": str(out_dir / "recommended_all_tickets.csv"),
        "recommended_runtime_tickets": str(out_dir / "recommended_runtime_tickets.csv"),
        "best_by_top10_removed_roi": comparison.head(1).to_dict(orient="records")[0],
        "comparison": comparison.to_dict(orient="records"),
        "notes": [
            "MCS/PBO diagnostics supported umaren-centered policies; wide/win suppression is compared explicitly.",
            "top5/top10 removed ROI is used as the main robustness check because current ROI remains high-concentration.",
            "recommended policy defaults to mcs_full_margin095_s0304_skip03119: the strongest full MCS survivor family with skip-risk gate and enough race count.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
