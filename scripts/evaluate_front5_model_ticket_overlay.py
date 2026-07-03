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


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def parse_dates(df: pd.DataFrame) -> pd.Series:
    if "date_key" in df.columns:
        parsed = pd.to_datetime(df["date_key"], errors="coerce")
    elif "日付S" in df.columns:
        parsed = pd.to_datetime(df["日付S"].astype(str), errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=df.index)
    missing = parsed.isna()
    if missing.any() and "race_id" in df.columns:
        digits = df.loc[missing, "race_id"].astype(str).str.extract(r"(\d{8})", expand=False)
        parsed.loc[missing] = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    return parsed


def prepare(
    tickets_csv: Path,
    front5_predictions_csv: Path,
) -> pd.DataFrame:
    tickets = pd.read_csv(tickets_csv, dtype={"race_id": str}, low_memory=False)
    tickets["race_id"] = tickets["race_id"].astype(str)
    tickets["_date"] = parse_dates(tickets)
    tickets["_stake"] = num(tickets, "runtime_stake_yen", 0.0).fillna(num(tickets, "stake_yen", 0.0)).fillna(0.0)
    tickets["_return"] = num(tickets, "runtime_return_yen", 0.0).fillna(num(tickets, "return_yen", 0.0)).fillna(0.0)
    tickets["_margin"] = num(tickets, "min_odds_margin_ratio", 0.0).fillna(0.0)
    tickets["_expected_roi"] = num(tickets, "runtime_expected_roi", np.nan).fillna(num(tickets, "expected_roi_after_slippage", 0.0)).fillna(0.0)
    tickets["_hit_prob"] = num(tickets, "ticket_hit_prob", 0.0).fillna(0.0)
    tickets["_heuristic_front5"] = num(tickets, "projected_front5_prob", 0.5).fillna(0.5)
    tickets["_partner_odds"] = num(tickets, "partner_odds", np.nan).fillna(num(tickets, "odds", np.nan))
    tickets["_partner_pop"] = num(tickets, "partner_pop", np.nan).fillna(num(tickets, "popularity", np.nan))
    tickets["_partner_no"] = num(tickets, "partner_no", np.nan)

    pred = pd.read_csv(front5_predictions_csv, dtype={"race_id": str}, low_memory=False)
    pred = pred[["race_id", "horse_no", "front5_model_prob", "front5_model_prob_raw"]].copy()
    pred["race_id"] = pred["race_id"].astype(str)
    pred["horse_no"] = num(pred, "horse_no", np.nan)
    pred = pred.drop_duplicates(["race_id", "horse_no"], keep="last")
    merged = tickets.merge(
        pred.rename(
            columns={
                "horse_no": "_partner_no",
                "front5_model_prob": "_model_front5",
                "front5_model_prob_raw": "_model_front5_raw",
            }
        ),
        on=["race_id", "_partner_no"],
        how="left",
    )
    merged["_model_front5"] = num(merged, "_model_front5", np.nan).fillna(merged["_heuristic_front5"])
    merged["_front5_model_available"] = num(merged, "_model_front5_raw", np.nan).notna()
    return merged[merged["_stake"].gt(0)].copy()


def race_table(df: pd.DataFrame, stake: pd.Series) -> pd.DataFrame:
    selected = df[stake.gt(0)].copy()
    selected["_eval_stake"] = stake.loc[selected.index]
    base_pay = pd.Series(np.where(df["_stake"].gt(0), df["_return"] / df["_stake"], 0.0), index=df.index)
    selected["_eval_return"] = np.where(selected["_return"].gt(0), base_pay.loc[selected.index] * selected["_eval_stake"], 0.0)
    if selected.empty:
        return pd.DataFrame(columns=["race_id", "date", "stake_yen", "return_yen", "profit_yen", "hit"])
    race = (
        selected.groupby("race_id", sort=False)
        .agg(date=("_date", "min"), stake_yen=("_eval_stake", "sum"), return_yen=("_eval_return", "sum"))
        .reset_index()
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    race["hit"] = race["return_yen"].gt(0)
    return race.sort_values(["date", "race_id"])


def metrics(df: pd.DataFrame, stake: pd.Series, label: str) -> dict:
    selected = df[stake.gt(0)].copy()
    if selected.empty:
        return {
            "policy": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "roi": 0.0,
            "race_hit_rate": 0.0,
            "top10_removed_roi": 0.0,
            "model_front5_available_rate": 0.0,
        }
    base_pay = pd.Series(np.where(df["_stake"].gt(0), df["_return"] / df["_stake"], 0.0), index=df.index)
    eval_return = np.where(selected["_return"].gt(0), base_pay.loc[selected.index] * stake.loc[selected.index], 0.0)
    race = race_table(df, stake)
    stake_sum = float(stake.loc[selected.index].sum())
    return_sum = float(eval_return.sum())

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
        "stake_yen": stake_sum,
        "return_yen": return_sum,
        "profit_yen": return_sum - stake_sum,
        "roi": return_sum / stake_sum if stake_sum else 0.0,
        "ticket_hit_rate": float((eval_return > 0).mean()),
        "race_hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "top5_removed_roi": removed_roi(5),
        "top10_removed_roi": removed_roi(10),
        "avg_model_front5": float(selected["_model_front5"].mean()),
        "avg_heuristic_front5": float(selected["_heuristic_front5"].mean()),
        "avg_partner_odds": float(selected["_partner_odds"].mean()),
        "avg_partner_pop": float(selected["_partner_pop"].mean()),
        "model_front5_available_rate": float(selected["_front5_model_available"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether the OOS front5 model improves ticket overlays.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/pair_probability_runtime_v1/pair_calibrated_runtime_tickets.csv")
    parser.add_argument("--front5-predictions-csv", default="outputs/analysis/front5_position_model_v1/front5_oos_predictions.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/front5_ticket_overlay_v1")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare(project_path(args.tickets_csv), project_path(args.front5_predictions_csv))

    base = df["ticket_type"].eq("umaren") & df["_margin"].ge(0.95)
    longshot = df["_partner_odds"].ge(10.0) | df["_partner_pop"].ge(5.0)

    policies = {
        "mcs_s0304_existing": base,
        "model_front_ge050": base & df["_model_front5"].ge(0.50),
        "model_front_ge060": base & df["_model_front5"].ge(0.60),
        "model_front_ge065": base & df["_model_front5"].ge(0.65),
        "model_longshot_front_ge050": base & longshot & df["_model_front5"].ge(0.50),
        "model_longshot_front_ge060": base & longshot & df["_model_front5"].ge(0.60),
        "heuristic_longshot_front_ge060": base & longshot & df["_heuristic_front5"].ge(0.60),
        "model_front_ge050_hit08": base & df["_model_front5"].ge(0.50) & df["_hit_prob"].ge(0.08),
        "model_front_ge060_hit08": base & df["_model_front5"].ge(0.60) & df["_hit_prob"].ge(0.08),
    }

    rows = []
    for label, mask in policies.items():
        stake = df["_stake"].where(mask, 0.0)
        rows.append(metrics(df, stake, label))

    comparison = pd.DataFrame(rows).sort_values(["top10_removed_roi", "roi"], ascending=[False, False])
    comparison.to_csv(out_dir / "front5_ticket_overlay_comparison.csv", index=False, encoding="utf-8-sig")

    best = comparison.head(1).to_dict(orient="records")[0] if not comparison.empty else {}
    summary = {
        "tickets_csv": args.tickets_csv,
        "front5_predictions_csv": args.front5_predictions_csv,
        "output_dir": str(out_dir),
        "rows": int(len(df)),
        "model_front5_available_rate": float(df["_front5_model_available"].mean()) if len(df) else 0.0,
        "best_by_top10_removed_roi": best,
        "comparison": comparison.to_dict(orient="records"),
        "notes": [
            "This does not use post-race corner position. Front5 model probabilities are OOS monthly predictions.",
            "The base policy is the full MCS survivor family: umaren + margin>=0.95.",
            "Use this result to decide whether to feed the front5 model into the next ticket-generation pass.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
