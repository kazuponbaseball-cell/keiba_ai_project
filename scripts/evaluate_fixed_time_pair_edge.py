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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False)


def race_date_from_id(race_id: pd.Series) -> pd.Series:
    digits = race_id.astype(str).str.extract(r"(\d{8})", expand=False)
    return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")


def pair_key(df: pd.DataFrame, a_candidates: list[str], b_candidates: list[str]) -> pd.Series:
    a = pd.Series(np.nan, index=df.index, dtype=float)
    b = pd.Series(np.nan, index=df.index, dtype=float)
    for col in a_candidates:
        if col in df.columns:
            a = a.fillna(num(df, col))
    for col in b_candidates:
        if col in df.columns:
            b = b.fillna(num(df, col))
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return (
        df["race_id"].astype(str)
        + ":"
        + lo.astype("Int64").astype(str)
        + "-"
        + hi.astype("Int64").astype(str)
    )


def prepare_tickets(tickets: pd.DataFrame, stake_col: str) -> pd.DataFrame:
    if tickets.empty:
        return tickets
    out = tickets.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["ticket_type"] = out.get("ticket_type", "").astype(str)
    out = out[out["ticket_type"].eq("umaren")].copy()
    out["_ticket_key_norm"] = pair_key(out, ["anchor_no", "horse_a"], ["partner_no", "horse_b"])
    out["_date"] = race_date_from_id(out["race_id"])
    out["_year"] = out["_date"].dt.year.fillna(num(out, "year")).astype("Int64")
    out["_eval_stake_yen"] = num(out, stake_col).fillna(num(out, "runtime_stake_yen")).fillna(num(out, "stake_yen")).fillna(0.0)
    out = out[out["_eval_stake_yen"].gt(0)].copy()
    if "umaren_hit" in out.columns:
        hit_text = out["umaren_hit"].astype(str).str.lower()
        out["_umaren_hit_bool"] = hit_text.isin(["true", "1", "yes"]) | num(out, "umaren_hit", 0.0).eq(1)
    else:
        out["_umaren_hit_bool"] = num(out, "runtime_return_yen").gt(0)
    out["_final_pay_per100"] = num(out, "umaren_pay").fillna(num(out, "runtime_backtest_pay_per100")).fillna(0.0)
    return out


def prepare_timeline(timeline: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    if timeline.empty:
        return timeline
    out = timeline.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["ticket_type"] = out.get("ticket_type", "").astype(str)
    out = out[out["ticket_type"].eq("umaren")].copy()
    if labels:
        out = out[out.get("decision_label", "").astype(str).isin(labels)].copy()
    if out.empty:
        return out
    out["_ticket_key_norm"] = pair_key(out, ["a_no", "horse_a", "umaban_a"], ["b_no", "horse_b", "umaban_b"])
    out["fixed_pay_per100"] = num(out, "live_pay_per100").fillna(num(out, "live_odds") * 100.0)
    out["fixed_odds"] = out["fixed_pay_per100"] / 100.0
    out["_sort_stamp"] = pd.to_datetime(out.get("captured_at", ""), format="%Y%m%d_%H%M%S", errors="coerce")
    fallback_stamp = pd.to_datetime(out.get("snapshot_at", ""), errors="coerce")
    out["_sort_stamp"] = out["_sort_stamp"].fillna(fallback_stamp)
    out = out[out["fixed_odds"].gt(1.0)].copy()
    out = out.sort_values(["race_id", "_ticket_key_norm", "decision_label", "_sort_stamp"])
    return out.drop_duplicates(["race_id", "_ticket_key_norm", "decision_label"], keep="last")


def add_pair_edge(merged: pd.DataFrame, payout_rate: float) -> tuple[pd.DataFrame, list[str]]:
    out = merged.copy()
    out["fixed_market_implied_prob_takeout_adj"] = (payout_rate / out["fixed_odds"]).replace([np.inf, -np.inf], np.nan)
    out["fixed_break_even_prob"] = (1.0 / out["fixed_odds"]).replace([np.inf, -np.inf], np.nan)
    prob_cols = []
    for col in ["ticket_hit_prob", "pair_calibrated_hit_prob", "race_sim_umaren_prob_cal"]:
        if col in out.columns and num(out, col).notna().any():
            out[col] = num(out, col).clip(0.0001, 0.95)
            prob_cols.append(col)
    if {"ticket_hit_prob", "race_sim_umaren_prob_cal"}.issubset(out.columns):
        ticket = num(out, "ticket_hit_prob")
        sim = num(out, "race_sim_umaren_prob_cal")
        out["blend_ticket75_racesim25_prob"] = (0.75 * ticket + 0.25 * sim).where(sim.notna(), ticket)
        out["blend_ticket50_racesim50_prob"] = (0.50 * ticket + 0.50 * sim).where(sim.notna(), ticket)
        prob_cols.extend(["blend_ticket75_racesim25_prob", "blend_ticket50_racesim50_prob"])
    market = out["fixed_market_implied_prob_takeout_adj"].clip(0.0001, 0.95)
    for col in prob_cols:
        p = num(out, col).clip(0.0001, 0.95)
        prefix = col.replace("_prob", "").replace("_hit", "")
        out[f"{prefix}_fixed_model_ev"] = p * out["fixed_odds"]
        out[f"{prefix}_fixed_market_ratio"] = (p / market).replace([np.inf, -np.inf], np.nan)
        out[f"{prefix}_fixed_log_edge"] = np.log(p) - np.log(market)
        out[f"{prefix}_fixed_break_even_diff"] = p - out["fixed_break_even_prob"]
    out["final_vs_fixed_ratio"] = out["_final_pay_per100"] / out["fixed_pay_per100"].replace(0, np.nan)
    return out, prob_cols


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
            "avg_fixed_odds": 0.0,
            "avg_fixed_market_prob": 0.0,
            "hit_avg_final_vs_fixed_ratio": np.nan,
        }
    stake = selected["_eval_stake_yen"].fillna(0.0)
    ret = selected["_final_pay_per100"].where(selected["_umaren_hit_bool"], 0.0) * stake / 100.0
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

    hit_rows = selected[selected["_umaren_hit_bool"]].copy()
    return {
        "policy": label,
        "tickets": int(len(selected)),
        "races": int(selected["race_id"].nunique()),
        "stake_yen": total_stake,
        "return_yen": total_return,
        "profit_yen": total_return - total_stake,
        "roi": total_return / total_stake if total_stake else 0.0,
        "ticket_hit_rate": float(selected["_umaren_hit_bool"].mean()),
        "race_hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "max_drawdown_yen": max_drawdown(race),
        "top5_removed_roi": removed_roi(5),
        "top10_removed_roi": removed_roi(10),
        "avg_fixed_odds": float(selected["fixed_odds"].mean()),
        "avg_fixed_market_prob": float(selected["fixed_market_implied_prob_takeout_adj"].mean()),
        "hit_avg_final_vs_fixed_ratio": float(hit_rows["final_vs_fixed_ratio"].mean()) if not hit_rows.empty else np.nan,
    }


def evaluate(merged: pd.DataFrame, prob_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    yearly: list[dict] = []
    ev_thresholds = [1.00, 1.15, 1.25, 1.35, 1.50, 1.75, 2.00, 2.50, 3.00]
    ratio_thresholds = [1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00]
    for label, label_df in merged.groupby("decision_label", dropna=False):
        label_name = str(label)
        base_mask = pd.Series(True, index=label_df.index)
        base_policy = f"{label_name}:base_matched"
        rows.append(metric_row(label_df, base_policy, base_mask))
        for year, g in label_df.groupby("_year", dropna=False):
            yearly.append(metric_row(g, base_policy, pd.Series(True, index=g.index)) | {"year": int(year) if pd.notna(year) else None})
        for col in prob_cols:
            prefix = col.replace("_prob", "").replace("_hit", "")
            ev_col = f"{prefix}_fixed_model_ev"
            ratio_col = f"{prefix}_fixed_market_ratio"
            for th in ev_thresholds:
                policy = f"{label_name}:{prefix}:fixed_ev_ge_{th:.2f}"
                mask = num(label_df, ev_col).ge(th)
                rows.append(metric_row(label_df, policy, mask))
            for th in ratio_thresholds:
                policy = f"{label_name}:{prefix}:fixed_market_ratio_ge_{th:.2f}"
                mask = num(label_df, ratio_col).ge(th)
                rows.append(metric_row(label_df, policy, mask))
    summary = pd.DataFrame(rows).sort_values(["top10_removed_roi", "roi", "races"], ascending=[False, False, False])
    yearly_df = pd.DataFrame(yearly)
    return summary, yearly_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pair-edge rules at fixed live-odds decision labels such as T-5/T-3.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/mcs_pbo_runtime_overlay_v3/mcs_full_margin095_s0304_selected_tickets.csv")
    parser.add_argument("--pair-timeline-csv", default="data/processed/live_odds/realtime_pair_odds_timeline.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/fixed_time_pair_edge_v1")
    parser.add_argument("--labels", nargs="*", default=["T-5", "T-3", "final_check", "manual"])
    parser.add_argument("--stake-col", default="runtime_stake_yen")
    parser.add_argument("--payout-rate", type=float, default=0.775)
    args = parser.parse_args()

    out_dir = ensure_dir(project_path(args.output_dir))
    tickets = prepare_tickets(read_csv(project_path(args.tickets_csv)), args.stake_col)
    timeline = prepare_timeline(read_csv(project_path(args.pair_timeline_csv)), args.labels)

    if tickets.empty or timeline.empty:
        matched = pd.DataFrame()
        summary = pd.DataFrame()
        yearly = pd.DataFrame()
        prob_cols: list[str] = []
    else:
        matched = tickets.merge(
            timeline,
            on=["race_id", "_ticket_key_norm", "ticket_type"],
            how="inner",
            suffixes=("", "_timeline"),
        )
        matched, prob_cols = add_pair_edge(matched, args.payout_rate) if not matched.empty else (matched, [])
        summary, yearly = evaluate(matched, prob_cols) if not matched.empty else (pd.DataFrame(), pd.DataFrame())

    matched.to_csv(out_dir / "fixed_time_pair_edge_rows.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "fixed_time_pair_edge_policy_comparison.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "fixed_time_pair_edge_policy_by_year.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "tickets_csv": str(project_path(args.tickets_csv)),
        "pair_timeline_csv": str(project_path(args.pair_timeline_csv)),
        "labels": args.labels,
        "ticket_rows_after_filter": int(len(tickets)),
        "timeline_rows_after_filter": int(len(timeline)),
        "matched_rows": int(len(matched)),
        "matched_races": int(matched["race_id"].nunique()) if not matched.empty else 0,
        "probability_columns_evaluated": prob_cols,
        "top_policies": summary.head(20).to_dict(orient="records") if not summary.empty else [],
        "note": "No matched rows means T-5/T-3/final_check timeline snapshots have not been accumulated for the selected ticket set yet.",
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
