from __future__ import annotations

import argparse
import itertools
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_operational_win_addon import _json_default, _load_base_tickets, _metrics, _num
from src.utils.paths import ensure_dir, project_path


def _load_trio_payoffs(raw_csv: Path, race_col: str, encoding: str) -> pd.DataFrame:
    usecols = [race_col, "確定着順", "３連複"]
    raw = pd.read_csv(raw_csv, encoding=encoding, usecols=lambda c: c in usecols, low_memory=False)
    raw[race_col] = raw[race_col].astype(str)
    raw["rank_num"] = _num(raw.get("確定着順"), raw.index, np.nan)
    raw["trio_pay"] = _num(raw.get("３連複"), raw.index, 0.0).fillna(0.0)
    return (
        raw[raw["rank_num"].isin([1, 2, 3])]
        .groupby(race_col, as_index=False)
        .agg(trio_pay=("trio_pay", "max"))
        .rename(columns={race_col: "race_id"})
    )


def _prepare_scored(scored: pd.DataFrame, trio_payoffs: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = pd.to_numeric(df.get("year", df["race_id"].str.slice(0, 4)), errors="coerce")
    df["horse_no"] = _num(df.get("horse_no"), df.index, 0).fillna(0).astype(int)
    df["ai_rank_num"] = _num(df.get("ai_rank_num"), df.index, 999.0).fillna(999.0)
    df["pop_rank_num"] = _num(df.get("pop_rank_num"), df.index, 999.0).fillna(999.0)
    df["odds"] = _num(df.get("market_odds_live_or_final"), df.index, np.nan).fillna(_num(df.get("odds_num"), df.index, np.nan))
    df["place_score"] = _num(df.get("place_suitability_score"), df.index, 0.0).fillna(0.0)
    df["wide_axis_score"] = _num(df.get("wide_axis_score"), df.index, df["place_score"]).fillna(df["place_score"])
    df["wide_partner_score"] = _num(df.get("wide_partner_score"), df.index, 0.0).fillna(0.0)
    df["quinella_score"] = _num(df.get("quinella_model_score_norm"), df.index, 0.0).fillna(0.0)
    df["overlay"] = _num(df.get("market_overlay_score"), df.index, 0.0).fillna(0.0)
    df["front5"] = _num(df.get("projected_front5_prob"), df.index, 0.0).fillna(0.0)
    df["danger"] = _num(df.get("danger_popular_hybrid_score"), df.index, np.nan).fillna(
        _num(df.get("danger_favorite_score"), df.index, 0.0)
    )
    df["skip_risk"] = _num(df.get("skip_risk_score"), df.index, 0.0).fillna(0.0)
    df["difficulty"] = _num(df.get("race_difficulty_model_score"), df.index, np.nan).fillna(
        _num(df.get("target_race_difficulty"), df.index, 0.5)
    )
    df["is_place"] = _num(df.get("is_place"), df.index, 0.0).fillna(0.0).gt(0)
    df["partner_score"] = (
        0.32 * df["wide_partner_score"]
        + 0.26 * df["quinella_score"]
        + 0.22 * df["overlay"]
        + 0.12 * df["front5"]
        + 0.08 * (1.0 - df["danger"].clip(0.0, 1.0))
    )
    return df.merge(trio_payoffs, on="race_id", how="left")


def _grid() -> list[dict]:
    rows: list[dict] = []
    for anchor_place_min, anchor_danger_max, diff_max, partner_topn, partner_score_min, partner_ai_rank_max, partner_odds_max, partner_danger_max in product(
        [0.68, 0.78],
        [0.55],
        [0.75, 1.01],
        [3, 4],
        [0.52, 0.62],
        [5, 7],
        [80.0, 300.0],
        [0.80],
    ):
        rows.append(
            {
                "anchor_place_min": anchor_place_min,
                "anchor_danger_max": anchor_danger_max,
                "diff_max": diff_max,
                "partner_topn": partner_topn,
                "partner_score_min": partner_score_min,
                "partner_ai_rank_max": partner_ai_rank_max,
                "partner_odds_max": partner_odds_max,
                "partner_danger_max": partner_danger_max,
            }
        )
    return rows


def _candidate_universe(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for race_id, race in df.groupby("race_id", sort=False):
        anchor_rows = race[
            race["ai_rank_num"].eq(1)
            & race["place_score"].ge(0.58)
            & race["danger"].le(0.65)
            & race["difficulty"].le(1.01)
            & race["skip_risk"].le(0.85)
        ]
        if anchor_rows.empty:
            continue
        anchor = anchor_rows.sort_values(["place_score", "wide_axis_score"], ascending=[False, False]).iloc[0]
        partners = race[
            race["horse_no"].ne(anchor["horse_no"])
            & race["ai_rank_num"].le(10)
            & race["partner_score"].ge(0.42)
            & race["odds"].le(300.0)
            & race["danger"].le(0.80)
        ].copy()
        partners = partners.sort_values(["partner_score", "overlay", "place_score"], ascending=[False, False, False]).head(5)
        partners = partners.reset_index(drop=True)
        if len(partners) < 2:
            continue
        for b_idx, c_idx in itertools.combinations(partners.index, 2):
            b = partners.loc[b_idx]
            c = partners.loc[c_idx]
            hit = bool(anchor["is_place"] and b["is_place"] and c["is_place"])
            rows.append(
                {
                    "race_id": race_id,
                    "year": int(anchor["year"]) if pd.notna(anchor["year"]) else None,
                    "ticket_type": "trio",
                    "hit": hit,
                    "trio_pay": float(anchor.get("trio_pay", 0.0) or 0.0),
                    "operation_profile": "trio_addon",
                    "operation_profile_label": "3連複追加",
                    "operation_strength_rank": 2,
                    "anchor_no": int(anchor["horse_no"]),
                    "anchor_name": str(anchor.get("horse_name", "")),
                    "partner_no": int(b["horse_no"]),
                    "partner_name": str(b.get("horse_name", "")),
                    "third_no": int(c["horse_no"]),
                    "third_name": str(c.get("horse_name", "")),
                    "anchor_danger": float(anchor["danger"]),
                    "partner_max_danger": float(max(b["danger"], c["danger"])),
                    "partner_max_rank": float(max(b["ai_rank_num"], c["ai_rank_num"])),
                    "partner_max_order": int(max(b_idx, c_idx) + 1),
                    "partner_max_odds": float(max(b["odds"], c["odds"])),
                    "partner_min_score": float(min(b["partner_score"], c["partner_score"])),
                    "pair_quinella_score": float((anchor["wide_axis_score"] + b["partner_score"] + c["partner_score"]) / 3.0),
                    "market_overlay_score": float(max(b["overlay"], c["overlay"])),
                    "race_difficulty_score": float(anchor["difficulty"]),
                    "anchor_place_score": float(anchor["place_score"]),
                    "b_partner_score": float(b["partner_score"]),
                    "c_partner_score": float(c["partner_score"]),
                }
            )
    return pd.DataFrame(rows)


def _select_trios(universe: pd.DataFrame, params: dict, stake_yen: int) -> pd.DataFrame:
    if universe.empty:
        return universe.copy()
    selected = universe[
        universe["anchor_place_score"].ge(params["anchor_place_min"])
        & universe["anchor_danger"].le(params["anchor_danger_max"])
        & universe["race_difficulty_score"].le(params["diff_max"])
        & universe["partner_max_order"].le(params["partner_topn"])
        & universe["partner_min_score"].ge(params["partner_score_min"])
        & universe["partner_max_rank"].le(params["partner_ai_rank_max"])
        & universe["partner_max_odds"].le(params["partner_odds_max"])
        & universe["partner_max_danger"].le(params["partner_danger_max"])
    ].copy()
    if not selected.empty:
        selected["stake_yen"] = float(stake_yen)
        selected["return_yen"] = selected["trio_pay"].where(selected["hit"], 0.0) * stake_yen / 100.0
    return selected


def _choose_policy(train: pd.DataFrame, stake_yen: int, min_train_races: int, min_race_hit_rate: float) -> tuple[dict | None, dict | None]:
    best_params: dict | None = None
    best_metrics: dict | None = None
    best_score = -np.inf
    for params in _grid():
        selected = _select_trios(train, params, stake_yen)
        metrics = _metrics(selected, "train_trio_addon")
        if metrics["races"] < min_train_races or metrics["race_hit_rate"] < min_race_hit_rate:
            continue
        score = (
            (metrics["roi"] - 1.0) * 100.0
            + metrics["race_hit_rate"] * 10.0
            - max(0.0, abs(metrics["max_drawdown_yen"]) / 10000.0) * 0.20
            - max(0.0, metrics["avg_stake_per_race"] - 600.0) / 1000.0
        )
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics
    return best_params, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize 3-renpuku addon tickets from existing scored data.")
    parser.add_argument("--scored-csv", default="outputs/analysis/risk_models_v1/investment_features_with_risk_models.csv")
    parser.add_argument("--base-tickets-csv", default="outputs/analysis/operational_win_addon_1pt_v1/combined_ticket_profiles.csv")
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--race-col", default="レースID(新/馬番無)")
    parser.add_argument("--encoding", default="cp932")
    parser.add_argument("--output-dir", default="outputs/analysis/operational_trio_addon_v1")
    parser.add_argument("--stake-yen", type=int, default=100)
    parser.add_argument("--min-train-races", type=int, default=60)
    parser.add_argument("--min-race-hit-rate", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = ensure_dir(project_path(args.output_dir))
    trio_payoffs = _load_trio_payoffs(project_path(args.raw_csv), args.race_col, args.encoding)
    scored = pd.read_csv(project_path(args.scored_csv), dtype={"race_id": str}, low_memory=False)
    candidates = _candidate_universe(_prepare_scored(scored, trio_payoffs))
    base = _load_base_tickets(project_path(args.base_tickets_csv))

    years = sorted(int(y) for y in candidates["year"].dropna().unique())
    wf_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    for year in years[1:]:
        train = candidates[candidates["year"].lt(year)].copy()
        test = candidates[candidates["year"].eq(year)].copy()
        params, train_metrics = _choose_policy(train, args.stake_yen, args.min_train_races, args.min_race_hit_rate)
        if params is None:
            wf_rows.append({"year": year, "selected": False})
            continue
        test_selected = _select_trios(test, params, args.stake_yen)
        test_metrics = _metrics(test_selected, f"test_{year}_trio_addon")
        ticket_frames.append(test_selected.assign(test_year=year))
        wf_rows.append(
            {
                "year": year,
                "selected": True,
                **{f"param_{k}": v for k, v in params.items()},
                **{f"train_{k}": v for k, v in (train_metrics or {}).items() if k != "policy"},
                **{f"test_{k}": v for k, v in test_metrics.items() if k != "policy"},
            }
        )

    trio_tickets = pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame()
    test_years = sorted(trio_tickets["year"].dropna().astype(int).unique()) if not trio_tickets.empty else years[1:]
    base_test = base[base["year"].isin(test_years)].copy() if test_years else base.iloc[0:0].copy()
    combined = pd.concat([base_test, trio_tickets], ignore_index=True, sort=False)
    summary = {
        "input": {
            "scored_csv": args.scored_csv,
            "base_tickets_csv": args.base_tickets_csv,
            "raw_csv": args.raw_csv,
            "stake_yen": args.stake_yen,
            "test_years": test_years,
        },
        "base_operational": _metrics(base_test, "base_operational"),
        "trio_addon": _metrics(trio_tickets, "trio_addon"),
        "combined": _metrics(combined, "base_plus_trio_addon"),
    }
    summary["delta_roi"] = summary["combined"]["roi"] - summary["base_operational"]["roi"]
    summary["delta_profit_yen"] = summary["combined"]["profit_yen"] - summary["base_operational"]["profit_yen"]

    pd.DataFrame(wf_rows).to_csv(out_dir / "walkforward_trio_summary.csv", index=False, encoding="utf-8-sig")
    trio_tickets.to_csv(out_dir / "trio_addon_tickets.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(out_dir / "combined_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary["base_operational"], summary["trio_addon"], summary["combined"]]).to_csv(
        out_dir / "metrics.csv", index=False, encoding="utf-8-sig"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
