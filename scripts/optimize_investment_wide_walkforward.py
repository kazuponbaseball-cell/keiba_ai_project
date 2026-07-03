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
    if series.dtype == object:
        cleaned = series.astype(str).str.replace(r"[(),]", "", regex=True).replace({"nan": np.nan, "": np.nan})
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _load_wide_payoffs(path: Path) -> pd.DataFrame:
    wide = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
    for col in ["horse_a", "horse_b", "wide_pay"]:
        wide[col] = _num(wide[col])
    return wide


def _candidate_tickets(scored: pd.DataFrame, wide_payoffs: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    df["year"] = df["race_id"].astype(str).str[:4].astype(int)
    anchors = df[_num(df["ai_rank_num"]).eq(1)].copy()
    partners = df[_num(df["ai_rank_num"]).between(2, 8)].copy()
    rows = []
    partner_cols = [
        "race_id",
        "horse_no",
        "horse_name",
        "ai_rank_num",
        "pop_rank_num",
        "market_odds_live_or_final",
        "wide_partner_score",
        "projected_front5_prob",
        "market_overlay_score",
        "late_value_survives_score",
        "danger_favorite_score",
    ]
    anchor_cols = [
        "race_id",
        "horse_no",
        "horse_name",
        "pop_rank_num",
        "market_odds_live_or_final",
        "wide_axis_score",
        "danger_favorite_score",
        "skip_risk_score",
        "year",
    ]
    for race_id, a in anchors.groupby("race_id"):
        p = partners[partners["race_id"].eq(race_id)]
        if p.empty:
            continue
        aa = a[anchor_cols].head(1)
        pp = p[partner_cols].copy()
        merged = aa.merge(pp, on="race_id", suffixes=("_anchor", "_partner"))
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    tickets = pd.concat(rows, ignore_index=True, sort=False)
    tickets = tickets[tickets["horse_no_anchor"] != tickets["horse_no_partner"]].copy()
    tickets["horse_a"] = np.minimum(_num(tickets["horse_no_anchor"]), _num(tickets["horse_no_partner"]))
    tickets["horse_b"] = np.maximum(_num(tickets["horse_no_anchor"]), _num(tickets["horse_no_partner"]))
    tickets = tickets.merge(wide_payoffs, on=["race_id", "horse_a", "horse_b"], how="left")
    tickets["wide_hit"] = tickets["wide_pay"].notna()
    tickets["wide_return"] = _num(tickets["wide_pay"]).fillna(0.0)
    return tickets


def _filter_candidates(tickets: pd.DataFrame, params: dict) -> pd.DataFrame:
    t = tickets.copy()
    mask = (
        _num(t["wide_axis_score"]).ge(params["axis_min"])
        & _num(t["wide_partner_score"]).ge(params["partner_min"])
        & _num(t["market_odds_live_or_final_partner"]).ge(params["partner_odds_min"])
        & _num(t["market_odds_live_or_final_partner"]).le(params["partner_odds_max"])
        & _num(t["danger_favorite_score_anchor"]).le(params["anchor_danger_max"])
        & _num(t["danger_favorite_score_partner"]).le(params["partner_danger_max"])
        & _num(t["projected_front5_prob"]).ge(params["partner_front_min"])
    )
    selected = t[mask].copy()
    if selected.empty:
        return selected
    selected = (
        selected.sort_values(["race_id", "wide_partner_score", "market_overlay_score"], ascending=[True, False, False])
        .groupby("race_id", as_index=False)
        .head(params["partners_per_race"])
    )
    return selected


def _metrics(tickets: pd.DataFrame, label: str) -> dict:
    if tickets.empty:
        return {
            "policy": label,
            "tickets": 0,
            "races": 0,
            "hit_rate": 0.0,
            "roi": 0.0,
            "profit_yen_flat100": 0.0,
            "avg_partner_odds": None,
            "avg_partner_front5_prob": None,
        }
    stake = len(tickets) * 100.0
    ret = _num(tickets["wide_return"]).sum()
    return {
        "policy": label,
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "hit_rate": float(tickets["wide_hit"].mean()),
        "roi": float(ret / stake) if stake else 0.0,
        "profit_yen_flat100": float(ret - stake),
        "avg_partner_odds": float(_num(tickets["market_odds_live_or_final_partner"]).mean()),
        "avg_partner_front5_prob": float(_num(tickets["projected_front5_prob"]).mean()),
    }


def _param_grid() -> list[dict]:
    rows = []
    for axis_min in [0.62, 0.68, 0.72, 0.76]:
        for partner_min in [0.56, 0.60, 0.64, 0.68]:
            for partner_odds_min in [5.0, 8.0, 12.0]:
                for partner_odds_max in [40.0, 80.0, 999.0]:
                    for partner_front_min in [0.45, 0.55, 0.65]:
                        for partners_per_race in [1, 2]:
                            rows.append(
                                {
                                    "axis_min": axis_min,
                                    "partner_min": partner_min,
                                    "partner_odds_min": partner_odds_min,
                                    "partner_odds_max": partner_odds_max,
                                    "partner_front_min": partner_front_min,
                                    "anchor_danger_max": 0.55,
                                    "partner_danger_max": 0.35,
                                    "partners_per_race": partners_per_race,
                                }
                            )
    return rows


def _score_for_selection(metric: dict) -> float:
    if metric["tickets"] < 120:
        return -999.0
    return metric["roi"] * np.sqrt(max(metric["hit_rate"], 0.001)) * np.log1p(metric["tickets"])


def walkforward(tickets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = sorted(tickets["year"].dropna().astype(int).unique())
    params = _param_grid()
    train_rows = []
    wf_rows = []
    detail_frames = []
    for test_year in years[1:]:
        train = tickets[tickets["year"] < test_year].copy()
        test = tickets[tickets["year"] == test_year].copy()
        if train.empty or test.empty:
            continue
        scored = []
        for i, p in enumerate(params):
            selected = _filter_candidates(train, p)
            m = _metrics(selected, f"grid_{i}")
            m.update(p)
            m["selection_score"] = _score_for_selection(m)
            scored.append(m)
        train_summary = pd.DataFrame(scored).sort_values(
            ["selection_score", "roi", "tickets"], ascending=[False, False, False]
        )
        best = train_summary.iloc[0].to_dict()
        best_params = {k: best[k] for k in params[0].keys()}
        train_summary["test_year"] = test_year
        train_rows.append(train_summary.head(30))

        selected_test = _filter_candidates(test, best_params)
        test_metric = _metrics(selected_test, f"wf_test_{test_year}")
        test_metric.update(best_params)
        test_metric["test_year"] = test_year
        test_metric["train_roi"] = float(best["roi"])
        test_metric["train_tickets"] = int(best["tickets"])
        wf_rows.append(test_metric)
        if not selected_test.empty:
            tmp = selected_test.copy()
            tmp["test_year"] = test_year
            for k, v in best_params.items():
                tmp[k] = v
            detail_frames.append(tmp)
    return (
        pd.concat(train_rows, ignore_index=True, sort=False) if train_rows else pd.DataFrame(),
        pd.DataFrame(wf_rows),
        pd.concat(detail_frames, ignore_index=True, sort=False) if detail_frames else pd.DataFrame(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward optimize investment wide axis/partner policy.")
    parser.add_argument("--scored-csv", default="outputs/analysis/investment_decision_features_v1/investment_features_scored.csv")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/investment_wide_walkforward_v1")
    args = parser.parse_args()

    scored = pd.read_csv(project_path(args.scored_csv), low_memory=False)
    scored["race_id"] = scored["race_id"].astype(str)
    wide = _load_wide_payoffs(project_path(args.wide_payoff_csv))
    tickets = _candidate_tickets(scored, wide)
    out_dir = ensure_dir(project_path(args.output_dir))
    train_grid, wf_summary, wf_tickets = walkforward(tickets)

    all_default = _filter_candidates(
        tickets,
        {
            "axis_min": 0.70,
            "partner_min": 0.62,
            "partner_odds_min": 6.0,
            "partner_odds_max": 999.0,
            "partner_front_min": 0.0,
            "anchor_danger_max": 0.50,
            "partner_danger_max": 0.35,
            "partners_per_race": 2,
        },
    )
    default_metric = _metrics(all_default, "default_previous_like")

    tickets.to_csv(out_dir / "wide_candidate_universe.csv", index=False, encoding="utf-8-sig")
    train_grid.to_csv(out_dir / "walkforward_train_grid_top30.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(out_dir / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    wf_tickets.to_csv(out_dir / "walkforward_tickets.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "candidate_tickets": int(len(tickets)),
        "default_metric": default_metric,
        "walkforward_summary": wf_summary.to_dict(orient="records"),
        "walkforward_total": _metrics(wf_tickets, "walkforward_total"),
        "note": "Each test year uses thresholds selected only from prior years.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
