from __future__ import annotations

import argparse
import json
import sys
from itertools import product
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
    for c in ["horse_a", "horse_b", "wide_pay"]:
        wide[c] = _num(wide[c])
    return wide


def _single_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_cols = [
        "race_id",
        "horse_no",
        "horse_name",
        "ai_rank_num",
        "pop_rank_num",
        "market_odds_live_or_final",
        "win_suitability_score",
        "place_suitability_score",
        "wide_axis_score",
        "danger_favorite_score",
        "skip_risk_score",
        "projected_front5_prob",
        "win_return",
        "place_return",
        "is_win",
        "is_place",
    ]
    src = df[base_cols].copy()
    for ticket_type, ret_col, hit_col, score_col in [
        ("win", "win_return", "is_win", "win_suitability_score"),
        ("place", "place_return", "is_place", "place_suitability_score"),
    ]:
        t = src.copy()
        t["ticket_type"] = ticket_type
        t["ticket_key"] = ticket_type + ":" + t["race_id"].astype(str) + ":" + t["horse_no"].astype(str)
        t["primary_score"] = _num(t[score_col])
        t["return_yen"] = _num(t[ret_col]).fillna(0.0)
        t["hit"] = t[hit_col].astype(bool)
        rows.append(t)
    return pd.concat(rows, ignore_index=True, sort=False)


def _wide_candidates(df: pd.DataFrame, wide_payoffs: pd.DataFrame) -> pd.DataFrame:
    anchors = df[_num(df["ai_rank_num"]).eq(1)].copy()
    partners = df[_num(df["ai_rank_num"]).between(2, 8)].copy()
    frames = []
    for race_id, a in anchors.groupby("race_id"):
        p = partners[partners["race_id"].eq(race_id)]
        if p.empty:
            continue
        a = a.head(1)
        left = a[
            [
                "race_id",
                "horse_no",
                "horse_name",
                "pop_rank_num",
                "market_odds_live_or_final",
                "wide_axis_score",
                "danger_favorite_score",
                "skip_risk_score",
            ]
        ].rename(
            columns={
                "horse_no": "anchor_no",
                "horse_name": "anchor_name",
                "pop_rank_num": "anchor_pop",
                "market_odds_live_or_final": "anchor_odds",
                "danger_favorite_score": "anchor_danger",
                "skip_risk_score": "anchor_skip",
            }
        )
        right = p[
            [
                "race_id",
                "horse_no",
                "horse_name",
                "ai_rank_num",
                "pop_rank_num",
                "market_odds_live_or_final",
                "wide_partner_score",
                "market_overlay_score",
                "projected_front5_prob",
                "danger_favorite_score",
            ]
        ].rename(
            columns={
                "horse_no": "partner_no",
                "horse_name": "partner_name",
                "ai_rank_num": "partner_ai_rank",
                "pop_rank_num": "partner_pop",
                "market_odds_live_or_final": "partner_odds",
                "danger_favorite_score": "partner_danger",
            }
        )
        frames.append(left.merge(right, on="race_id", how="inner"))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out[out["anchor_no"] != out["partner_no"]].copy()
    out["horse_a"] = np.minimum(_num(out["anchor_no"]), _num(out["partner_no"]))
    out["horse_b"] = np.maximum(_num(out["anchor_no"]), _num(out["partner_no"]))
    out = out.merge(wide_payoffs, on=["race_id", "horse_a", "horse_b"], how="left")
    out["ticket_type"] = "wide"
    out["ticket_key"] = "wide:" + out["race_id"].astype(str) + ":" + out["horse_a"].astype(str) + "-" + out["horse_b"].astype(str)
    out["primary_score"] = 0.55 * _num(out["wide_partner_score"]) + 0.25 * _num(out["wide_axis_score"]) + 0.20 * _num(out["projected_front5_prob"])
    out["return_yen"] = _num(out["wide_pay"]).fillna(0.0)
    out["hit"] = out["wide_pay"].notna()
    return out


def _candidate_universe(scored: pd.DataFrame, wide_payoffs: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["year"] = scored["race_id"].astype(str).str[:4].astype(int)
    singles = _single_candidates(scored)
    wide = _wide_candidates(scored, wide_payoffs)
    singles["year"] = singles["race_id"].astype(str).str[:4].astype(int)
    if not wide.empty:
        wide["year"] = wide["race_id"].astype(str).str[:4].astype(int)
    return pd.concat([singles, wide], ignore_index=True, sort=False)


def _apply_policy(candidates: pd.DataFrame, params: dict) -> pd.DataFrame:
    frames = []
    if params["use_win"]:
        win = candidates[candidates["ticket_type"].eq("win")].copy()
        mask = (
            _num(win["win_suitability_score"]).ge(params["win_score_min"])
            & _num(win["market_odds_live_or_final"]).between(params["win_odds_min"], params["win_odds_max"])
            & _num(win["danger_favorite_score"]).le(params["danger_max"])
        )
        frames.append(win[mask])
    if params["use_place"]:
        place = candidates[candidates["ticket_type"].eq("place")].copy()
        mask = (
            _num(place["ai_rank_num"]).eq(1)
            & _num(place["place_suitability_score"]).ge(params["place_score_min"])
            & _num(place["danger_favorite_score"]).le(params["place_danger_max"])
            & _num(place["market_odds_live_or_final"]).ge(params["place_odds_min"])
        )
        frames.append(place[mask])
    if params["use_wide"]:
        wide = candidates[candidates["ticket_type"].eq("wide")].copy()
        mask = (
            _num(wide["wide_axis_score"]).ge(params["axis_min"])
            & _num(wide["wide_partner_score"]).ge(params["partner_min"])
            & _num(wide["partner_odds"]).between(params["partner_odds_min"], params["partner_odds_max"])
            & _num(wide["projected_front5_prob"]).ge(params["partner_front_min"])
            & _num(wide["anchor_danger"]).le(params["anchor_danger_max"])
            & _num(wide["partner_danger"]).le(params["partner_danger_max"])
        )
        wide = wide[mask].sort_values(["race_id", "primary_score", "market_overlay_score"], ascending=[True, False, False])
        wide = wide.groupby("race_id", as_index=False).head(params["wide_partners_per_race"])
        frames.append(wide)
    selected = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if selected.empty:
        return selected
    selected = selected.sort_values(["race_id", "ticket_type", "primary_score"], ascending=[True, True, False])
    selected = selected.drop_duplicates("ticket_key")
    if params["max_tickets_per_race"] > 0:
        selected = (
            selected.sort_values(["race_id", "primary_score"], ascending=[True, False])
            .groupby("race_id", as_index=False)
            .head(params["max_tickets_per_race"])
        )
    return selected


def _metrics(tickets: pd.DataFrame, label: str) -> dict:
    if tickets.empty:
        return {
            "policy": label,
            "tickets": 0,
            "races": 0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "roi": 0.0,
            "profit_yen_flat100": 0.0,
        }
    stake = len(tickets) * 100.0
    ret = _num(tickets["return_yen"]).fillna(0.0).sum()
    by_race = tickets.groupby("race_id").agg(hit=("hit", "max"), ret=("return_yen", "sum"), tickets=("ticket_key", "count"))
    type_counts = tickets["ticket_type"].value_counts().to_dict()
    return {
        "policy": label,
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "avg_tickets_per_race": float(len(tickets) / tickets["race_id"].nunique()),
        "ticket_hit_rate": float(tickets["hit"].mean()),
        "race_hit_rate": float(by_race["hit"].mean()),
        "roi": float(ret / stake) if stake else 0.0,
        "profit_yen_flat100": float(ret - stake),
        "win_tickets": int(type_counts.get("win", 0)),
        "place_tickets": int(type_counts.get("place", 0)),
        "wide_tickets": int(type_counts.get("wide", 0)),
        "avg_return_hit_ticket": float(_num(tickets.loc[tickets["hit"], "return_yen"]).mean()) if tickets["hit"].any() else 0.0,
    }


def _grid(allowed_modes: set[str] | None = None) -> list[dict]:
    rows = []
    strategy_modes = [
        {"name": "wide_only", "use_win": False, "use_place": False, "use_wide": True},
        {"name": "wide_plus_place_cover", "use_win": False, "use_place": True, "use_wide": True},
        {"name": "all_balanced", "use_win": True, "use_place": True, "use_wide": True},
        {"name": "win_wide_roi", "use_win": True, "use_place": False, "use_wide": True},
    ]
    if allowed_modes:
        strategy_modes = [m for m in strategy_modes if m["name"] in allowed_modes]
    for mode in strategy_modes:
        for axis_min, partner_min, pmin, pmax, fmin, topn in product(
            [0.62, 0.70],
            [0.60, 0.66],
            [6.0, 10.0],
            [40.0, 999.0],
            [0.45, 0.60],
            [1, 2],
        ):
            for place_min in ([0.70] if mode["use_place"] else [0.0]):
                for win_min in ([0.74] if mode["use_win"] else [0.0]):
                    row = {
                        **mode,
                        "axis_min": axis_min,
                        "partner_min": partner_min,
                        "partner_odds_min": pmin,
                        "partner_odds_max": pmax,
                        "partner_front_min": fmin,
                        "wide_partners_per_race": topn,
                        "anchor_danger_max": 0.55,
                        "partner_danger_max": 0.35,
                        "place_score_min": place_min,
                        "place_danger_max": 0.40,
                        "place_odds_min": 2.0,
                        "win_score_min": win_min,
                        "win_odds_min": 10.0,
                        "win_odds_max": 999.0,
                        "danger_max": 0.25,
                        "max_tickets_per_race": 4,
                    }
                    rows.append(row)
    return rows


def _selection_score(metric: dict, min_races: int, min_race_hit: float) -> float:
    if metric["races"] < min_races or metric["race_hit_rate"] < min_race_hit:
        return -999.0
    # ROI first, but do not ignore hit coverage.
    return metric["roi"] * np.sqrt(max(metric["race_hit_rate"], 0.001)) * np.log1p(metric["races"])


def _walkforward(
    candidates: pd.DataFrame,
    min_races: int,
    min_race_hit: float,
    allowed_modes: set[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = sorted(candidates["year"].dropna().astype(int).unique())
    grid = _grid(allowed_modes)
    train_rows = []
    wf_rows = []
    ticket_frames = []
    for test_year in years[1:]:
        train = candidates[candidates["year"] < test_year].copy()
        test = candidates[candidates["year"] == test_year].copy()
        scored = []
        for i, params in enumerate(grid):
            selected = _apply_policy(train, params)
            metric = _metrics(selected, f"grid_{i}_{params['name']}")
            metric.update(params)
            metric["selection_score"] = _selection_score(metric, min_races, min_race_hit)
            scored.append(metric)
        train_grid = pd.DataFrame(scored).sort_values(["selection_score", "roi", "race_hit_rate"], ascending=[False, False, False])
        best = train_grid.iloc[0].to_dict()
        params = {k: best[k] for k in grid[0].keys()}
        train_grid["test_year"] = test_year
        train_rows.append(train_grid.head(50))

        selected_test = _apply_policy(test, params)
        metric = _metrics(selected_test, f"wf_test_{test_year}_{params['name']}")
        metric.update(params)
        metric["test_year"] = test_year
        metric["train_roi"] = float(best["roi"])
        metric["train_race_hit_rate"] = float(best["race_hit_rate"])
        metric["train_races"] = int(best["races"])
        wf_rows.append(metric)
        if not selected_test.empty:
            tmp = selected_test.copy()
            tmp["test_year"] = test_year
            tmp["selected_policy"] = metric["policy"]
            ticket_frames.append(tmp)
    return (
        pd.concat(train_rows, ignore_index=True, sort=False) if train_rows else pd.DataFrame(),
        pd.DataFrame(wf_rows),
        pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize individual ticket purchases for ROI with hit-rate coverage.")
    parser.add_argument("--scored-csv", default="outputs/analysis/investment_decision_features_v1/investment_features_scored.csv")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/individual_ticket_purchase_optimizer_v1")
    parser.add_argument("--min-train-races", type=int, default=250)
    parser.add_argument("--min-race-hit", type=float, default=0.16)
    parser.add_argument("--modes", nargs="*", default=None, help="Optional strategy modes: wide_only wide_plus_place_cover all_balanced win_wide_roi")
    args = parser.parse_args()

    scored = pd.read_csv(project_path(args.scored_csv), low_memory=False)
    scored["race_id"] = scored["race_id"].astype(str)
    wide = _load_wide_payoffs(project_path(args.wide_payoff_csv))
    candidates = _candidate_universe(scored, wide)
    allowed_modes = set(args.modes) if args.modes else None
    train_grid, wf_summary, wf_tickets = _walkforward(candidates, args.min_train_races, args.min_race_hit, allowed_modes)
    out_dir = ensure_dir(project_path(args.output_dir))

    candidates.to_csv(out_dir / "ticket_candidate_universe.csv", index=False, encoding="utf-8-sig")
    train_grid.to_csv(out_dir / "walkforward_train_grid_top50.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(out_dir / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    wf_tickets.to_csv(out_dir / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "candidate_tickets": int(len(candidates)),
        "min_train_races": args.min_train_races,
        "min_race_hit": args.min_race_hit,
        "walkforward_summary": wf_summary.to_dict(orient="records"),
        "walkforward_total": _metrics(wf_tickets, "walkforward_total"),
        "note": "Flat 100 yen per selected ticket. Each test year uses conditions selected only from prior years.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
