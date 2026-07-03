from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def race_date_from_id(race_id: pd.Series) -> pd.Series:
    digits = race_id.astype(str).str.extract(r"(\d{8})", expand=False)
    return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")


def pair_numbers(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if {"horse_a", "horse_b"}.issubset(df.columns):
        a = num(df, "horse_a")
        b = num(df, "horse_b")
    else:
        a = num(df, "anchor_no")
        b = num(df, "partner_no")
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return pd.Series(lo, index=df.index), pd.Series(hi, index=df.index)


def add_pair_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    lo, hi = pair_numbers(out)
    out["_horse_a_norm"] = lo.astype("Int64")
    out["_horse_b_norm"] = hi.astype("Int64")
    out["_ticket_key_norm"] = (
        out["race_id"].astype(str)
        + ":"
        + out["_horse_a_norm"].astype(str)
        + "-"
        + out["_horse_b_norm"].astype(str)
    )
    return out


def merge_race_sim(base: pd.DataFrame, race_sim_csv: Path | None) -> pd.DataFrame:
    if race_sim_csv is None or not race_sim_csv.exists():
        return base
    sim = pd.read_csv(race_sim_csv, dtype={"race_id": str}, low_memory=False)
    sim = add_pair_key(sim)
    sim_cols = [
        c
        for c in sim.columns
        if c.startswith("race_sim_") or c in {"_ticket_key_norm", "race_id", "ticket_type"}
    ]
    sim = sim[sim_cols].drop_duplicates(["race_id", "_ticket_key_norm", "ticket_type"], keep="last")
    add_cols = [
        c
        for c in sim.columns
        if c not in {"race_id", "_ticket_key_norm", "ticket_type"} and c not in base.columns
    ]
    if not add_cols:
        return base
    return base.merge(
        sim[["race_id", "_ticket_key_norm", "ticket_type", *add_cols]],
        on=["race_id", "_ticket_key_norm", "ticket_type"],
        how="left",
    )


def available_quote_odds(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    sources: list[tuple[str, pd.Series]] = []
    if "runtime_pay_per100" in df.columns:
        sources.append(("runtime_pay_per100", num(df, "runtime_pay_per100") / 100.0))
    if "quote_odds_proxy" in df.columns:
        sources.append(("quote_odds_proxy", num(df, "quote_odds_proxy")))
    if "quote_pay_proxy_per100" in df.columns:
        sources.append(("quote_pay_proxy_per100", num(df, "quote_pay_proxy_per100") / 100.0))
    if "runtime_odds" in df.columns:
        sources.append(("runtime_odds", num(df, "runtime_odds")))

    odds = pd.Series(np.nan, index=df.index, dtype=float)
    source = pd.Series("", index=df.index, dtype=object)
    for name, series in sources:
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = odds.isna() & s.gt(1.0)
        odds.loc[mask] = s.loc[mask]
        source.loc[mask] = name
    return odds, source


def add_market_pair_edge(df: pd.DataFrame, payout_rate: float) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    odds, source = available_quote_odds(out)
    out["umaren_quote_odds_for_edge"] = odds
    out["umaren_quote_odds_source"] = source
    out["umaren_market_prob_takeout_adj"] = (float(payout_rate) / odds).replace([np.inf, -np.inf], np.nan)
    out["umaren_break_even_prob"] = (1.0 / odds).replace([np.inf, -np.inf], np.nan)

    prob_cols = [
        "ticket_hit_prob",
        "pair_calibrated_hit_prob",
        "race_sim_umaren_prob_cal",
    ]
    use_cols: list[str] = []
    for col in prob_cols:
        if col in out.columns:
            p = num(out, col)
            if p.notna().any():
                out[col] = p.clip(0.0001, 0.95)
                use_cols.append(col)

    if {"ticket_hit_prob", "race_sim_umaren_prob_cal"}.issubset(out.columns):
        ticket = num(out, "ticket_hit_prob")
        sim = num(out, "race_sim_umaren_prob_cal")
        out["blend_ticket75_racesim25_prob"] = (0.75 * ticket + 0.25 * sim).where(sim.notna(), ticket)
        out["blend_ticket50_racesim50_prob"] = (0.50 * ticket + 0.50 * sim).where(sim.notna(), ticket)
        use_cols.extend(["blend_ticket75_racesim25_prob", "blend_ticket50_racesim50_prob"])

    market_prob = out["umaren_market_prob_takeout_adj"].clip(0.0001, 0.95)
    for col in use_cols:
        p = num(out, col).clip(0.0001, 0.95)
        prefix = col.replace("_prob", "").replace("_hit", "")
        out[f"{prefix}_market_ratio"] = (p / market_prob).replace([np.inf, -np.inf], np.nan)
        out[f"{prefix}_market_log_edge"] = np.log(p) - np.log(market_prob)
        out[f"{prefix}_break_even_diff"] = p - out["umaren_break_even_prob"]
        out[f"{prefix}_model_expected_roi"] = p * odds
    return out, use_cols


def hit_pay_per100(df: pd.DataFrame) -> pd.Series:
    pay = num(df, "runtime_backtest_pay_per100")
    pay = pay.fillna(num(df, "umaren_pay"))
    stake = num(df, "runtime_stake_yen").replace(0, np.nan)
    derived = num(df, "runtime_return_yen") / stake * 100.0
    return pay.fillna(derived).fillna(0.0)


def max_drawdown(race: pd.DataFrame) -> float:
    if race.empty:
        return 0.0
    equity = race["profit_yen"].cumsum()
    peak = equity.cummax().clip(lower=0.0)
    return float((equity - peak).min())


def metric_row(df: pd.DataFrame, label: str, mask: pd.Series) -> dict:
    selected = df[mask.fillna(False)].copy()
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
            "avg_quote_odds": 0.0,
            "avg_market_prob": 0.0,
        }
    stake = num(selected, "_eval_stake_yen").fillna(0.0)
    hit = selected["_umaren_hit_bool"].astype(bool)
    ret = selected["_hit_pay_per100"].where(hit, 0.0) * stake / 100.0
    selected = selected.assign(_stake_eval=stake, _return_eval=ret)
    race = (
        selected.groupby("race_id", sort=False)
        .agg(
            date=("_date", "min"),
            stake_yen=("_stake_eval", "sum"),
            return_yen=("_return_eval", "sum"),
            hit=("_return_eval", lambda x: bool((x > 0).any())),
        )
        .reset_index()
        .sort_values(["date", "race_id"])
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    total_stake = float(race["stake_yen"].sum())
    total_return = float(race["return_yen"].sum())

    def removed_roi(n: int) -> float:
        if len(race) <= n:
            return 0.0
        kept = race.sort_values("profit_yen", ascending=False).iloc[n:]
        kept_stake = float(kept["stake_yen"].sum())
        return float(kept["return_yen"].sum() / kept_stake) if kept_stake else 0.0

    return {
        "policy": label,
        "tickets": int(len(selected)),
        "races": int(selected["race_id"].nunique()),
        "stake_yen": total_stake,
        "return_yen": total_return,
        "profit_yen": total_return - total_stake,
        "roi": total_return / total_stake if total_stake else 0.0,
        "ticket_hit_rate": float(hit.mean()),
        "race_hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "max_drawdown_yen": max_drawdown(race),
        "top5_removed_roi": removed_roi(5),
        "top10_removed_roi": removed_roi(10),
        "avg_quote_odds": float(selected["umaren_quote_odds_for_edge"].mean()),
        "avg_market_prob": float(selected["umaren_market_prob_takeout_adj"].mean()),
    }


def year_rows(df: pd.DataFrame, policy: str, mask: pd.Series) -> pd.DataFrame:
    rows = []
    selected = df[mask.fillna(False)].copy()
    if selected.empty:
        return pd.DataFrame()
    for year, g in selected.groupby("_year", dropna=False):
        rows.append(metric_row(g, policy, pd.Series(True, index=g.index)) | {"year": int(year) if pd.notna(year) else None})
    return pd.DataFrame(rows)


def evaluate_policies(df: pd.DataFrame, prob_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    yearly: list[pd.DataFrame] = []
    all_mask = pd.Series(True, index=df.index)
    rows.append(metric_row(df, "base_all_selected_umaren", all_mask))
    yearly.append(year_rows(df, "base_all_selected_umaren", all_mask))

    ev_thresholds = [1.00, 1.15, 1.25, 1.35, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00]
    ratio_thresholds = [1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00]
    quantiles = [0.25, 0.40, 0.50, 0.60, 0.75]

    for col in prob_cols:
        prefix = col.replace("_prob", "").replace("_hit", "")
        ev_col = f"{prefix}_model_expected_roi"
        ratio_col = f"{prefix}_market_ratio"
        log_col = f"{prefix}_market_log_edge"
        if ev_col not in df.columns:
            continue
        for th in ev_thresholds:
            mask = num(df, ev_col).ge(th)
            label = f"{prefix}:model_ev_ge_{th:.2f}"
            rows.append(metric_row(df, label, mask))
            yearly.append(year_rows(df, label, mask))
        for th in ratio_thresholds:
            mask = num(df, ratio_col).ge(th)
            label = f"{prefix}:market_ratio_ge_{th:.2f}"
            rows.append(metric_row(df, label, mask))
            yearly.append(year_rows(df, label, mask))
        score = num(df, log_col)
        if score.notna().sum() >= 20:
            for q in quantiles:
                cutoff = float(score.quantile(q))
                mask = score.ge(cutoff)
                label = f"{prefix}:log_edge_top_{int((1-q)*100)}pct"
                row = metric_row(df, label, mask)
                row["score_cutoff"] = cutoff
                rows.append(row)
                yearly.append(year_rows(df, label, mask))

    summary = pd.DataFrame(rows).sort_values(
        ["top10_removed_roi", "roi", "races"],
        ascending=[False, False, False],
        na_position="last",
    )
    yearly_df = pd.concat([y for y in yearly if y is not None and not y.empty], ignore_index=True, sort=False) if yearly else pd.DataFrame()
    return summary, yearly_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate umaren pair-edge gates using public implied pair probability.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/mcs_pbo_runtime_overlay_v3/mcs_full_margin095_s0304_selected_tickets.csv")
    parser.add_argument("--race-sim-csv", default="outputs/analysis/race_sim_umaren_probability_v2/race_sim_umaren_runtime_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/umaren_pair_edge_overlay_v1")
    parser.add_argument("--payout-rate", type=float, default=0.775)
    parser.add_argument(
        "--stake-col",
        default="runtime_stake_yen",
        help="Stake column used for historical ROI. Use pre_mcs_pbo_stake_yen to re-test all pre-overlay umaren candidates.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(project_path(args.output_dir))
    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    tickets = add_pair_key(tickets)
    tickets = merge_race_sim(tickets, project_path(args.race_sim_csv) if args.race_sim_csv else None)
    tickets = tickets[tickets.get("ticket_type", "").astype(str).eq("umaren")].copy()
    tickets["_eval_stake_yen"] = num(tickets, args.stake_col)
    tickets["_eval_stake_yen"] = tickets["_eval_stake_yen"].fillna(num(tickets, "runtime_stake_yen")).fillna(0.0)
    tickets = tickets[tickets["_eval_stake_yen"].gt(0)].copy()
    tickets["_date"] = race_date_from_id(tickets["race_id"])
    tickets["_year"] = tickets["_date"].dt.year.fillna(num(tickets, "year")).astype("Int64")
    tickets["_hit_pay_per100"] = hit_pay_per100(tickets)
    if "umaren_hit" in tickets.columns:
        tickets["_umaren_hit_bool"] = tickets["umaren_hit"].astype(str).str.lower().isin(["true", "1", "yes"]) | num(tickets, "umaren_hit", 0.0).eq(1)
    else:
        tickets["_umaren_hit_bool"] = num(tickets, "runtime_return_yen").gt(0)

    scored, prob_cols = add_market_pair_edge(tickets, args.payout_rate)
    scored.to_csv(out_dir / "umaren_pair_edge_ticket_scores.csv", index=False, encoding="utf-8-sig")
    summary, yearly = evaluate_policies(scored, prob_cols)
    summary.to_csv(out_dir / "umaren_pair_edge_policy_comparison.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "umaren_pair_edge_policy_by_year.csv", index=False, encoding="utf-8-sig")

    robust = summary[(summary["races"] >= 80) & (summary["top10_removed_roi"] > 0)]
    recommended = robust.head(10) if not robust.empty else summary.head(10)
    payload = {
        "output_dir": str(out_dir),
        "input_tickets_csv": str(project_path(args.tickets_csv)),
        "race_sim_csv": str(project_path(args.race_sim_csv)) if args.race_sim_csv else None,
        "payout_rate": args.payout_rate,
        "tickets": int(len(scored)),
        "races": int(scored["race_id"].nunique()),
        "probability_columns_evaluated": prob_cols,
        "recommended_top10": recommended.to_dict(orient="records"),
        "base": summary[summary["policy"].eq("base_all_selected_umaren")].to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, ensure_ascii=False, indent=2)
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
