from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_operational_win_addon import _json_default, _metrics
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _prepare(tickets: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = pd.to_numeric(df.get("year", df["race_id"].str.slice(0, 4)), errors="coerce")
    df["ticket_type"] = df.get("ticket_type", "").astype(str)
    df["stake_yen"] = _num(df.get("stake_yen"), df.index, 100.0).fillna(100.0)
    df["return_yen"] = _num(df.get("return_yen"), df.index, 0.0).fillna(0.0)
    df["hit"] = df.get("hit", False).astype(bool)
    for col, default in [
        ("pair_quinella_score", 0.65),
        ("market_overlay_score", 0.65),
        ("late_value_survives_score", 0.70),
        ("anchor_win_score", 0.35),
        ("anchor_danger", 0.25),
        ("partner_danger", 0.05),
        ("race_chaos_score", 0.50),
        ("race_solidness_score", 0.60),
        ("race_difficulty_score", 0.50),
        ("race_front_pressure", 0.25),
        ("race_pace_collapse", 0.20),
        ("operation_strength_rank", 2.0),
    ]:
        df[col] = _num(df.get(col), df.index, default).fillna(default)

    df["stake_quality_score"] = (
        0.30 * df["pair_quinella_score"]
        + 0.25 * df["late_value_survives_score"]
        + 0.18 * df["market_overlay_score"]
        + 0.12 * (1.0 - df["anchor_danger"])
        + 0.08 * (1.0 - df["partner_danger"].clip(0.0, 1.0))
        + 0.07 * (df["operation_strength_rank"].clip(1.0, 3.0) / 3.0)
    ).clip(0.0, 1.0)

    # Ticket-specific nudges. The purpose is not to create a new predictor, but
    # to size existing adopted tickets according to the risks that matter most
    # for that ticket type.
    df["ticket_sizing_score"] = df["stake_quality_score"]
    wide = df["ticket_type"].eq("wide")
    umaren = df["ticket_type"].eq("umaren")
    win = df["ticket_type"].eq("win")
    df.loc[wide, "ticket_sizing_score"] = (
        df.loc[wide, "stake_quality_score"]
        + 0.08 * df.loc[wide, "race_front_pressure"]
        + 0.06 * df.loc[wide, "race_chaos_score"]
        - 0.06 * df.loc[wide, "race_solidness_score"]
    ).clip(0.0, 1.0)
    df.loc[umaren, "ticket_sizing_score"] = (
        df.loc[umaren, "stake_quality_score"]
        + 0.10 * df.loc[umaren, "pair_quinella_score"]
        - 0.08 * df.loc[umaren, "race_pace_collapse"]
        - 0.05 * df.loc[umaren, "anchor_danger"]
    ).clip(0.0, 1.0)
    df.loc[win, "ticket_sizing_score"] = (
        df.loc[win, "stake_quality_score"]
        + 0.12 * df.loc[win, "anchor_win_score"]
        + 0.04 * df.loc[win, "race_chaos_score"]
    ).clip(0.0, 1.0)
    return df


def _policy_grid() -> list[dict]:
    policies: list[dict] = []
    for (
        low_units,
        mid_units,
        high_units,
        top_units,
        mid_threshold,
        high_threshold,
        top_threshold,
        wide_mult,
        umaren_mult,
        win_mult,
        max_units_per_race,
    ) in product(
        [0, 1],
        [1, 2],
        [2, 4],
        [3, 5],
        [0.68],
        [0.74, 0.78],
        [0.84],
        [1.0],
        [1.0, 1.5],
        [0.5, 1.0],
        [6, 10],
    ):
        if not (mid_threshold < high_threshold < top_threshold):
            continue
        if not (low_units <= mid_units <= high_units <= top_units):
            continue
        policies.append(
            {
                "low_units": low_units,
                "mid_units": mid_units,
                "high_units": high_units,
                "top_units": top_units,
                "mid_threshold": mid_threshold,
                "high_threshold": high_threshold,
                "top_threshold": top_threshold,
                "wide_mult": wide_mult,
                "umaren_mult": umaren_mult,
                "win_mult": win_mult,
                "max_units_per_race": max_units_per_race,
            }
        )
    return policies


def _apply_policy(tickets: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = tickets.copy()
    score = df["ticket_sizing_score"]
    units = np.select(
        [
            score.ge(params["top_threshold"]),
            score.ge(params["high_threshold"]),
            score.ge(params["mid_threshold"]),
        ],
        [params["top_units"], params["high_units"], params["mid_units"]],
        default=params["low_units"],
    ).astype(float)
    type_mult = np.select(
        [
            df["ticket_type"].eq("wide"),
            df["ticket_type"].eq("umaren"),
            df["ticket_type"].eq("win"),
        ],
        [params["wide_mult"], params["umaren_mult"], params["win_mult"]],
        default=1.0,
    )
    df["stake_units"] = np.floor(units * type_mult + 1e-9)
    df = df[df["stake_units"].gt(0)].copy()
    if df.empty:
        return df

    df["_priority"] = (
        df["ticket_sizing_score"].rank(method="first", ascending=False)
        + df["late_value_survives_score"].rank(method="first", ascending=False) / 10000.0
        + df["pair_quinella_score"].rank(method="first", ascending=False) / 100000000.0
    )
    kept = []
    for _, group in df.sort_values(["race_id", "_priority"], ascending=[True, True]).groupby("race_id", sort=False):
        running = 0.0
        rows = []
        for idx, row in group.iterrows():
            units_i = float(row["stake_units"])
            if running + units_i > params["max_units_per_race"]:
                continue
            running += units_i
            rows.append(idx)
        kept.extend(rows)
    df = df.loc[kept].copy()
    if df.empty:
        return df

    original_stake = df["stake_yen"].replace(0, np.nan)
    payout_per_100 = df["return_yen"] / original_stake * 100.0
    df["stake_yen"] = df["stake_units"] * 100.0
    df["return_yen"] = np.where(df["hit"], payout_per_100 * df["stake_yen"] / 100.0, 0.0)
    df["operation_profile"] = df.get("operation_profile", "").astype(str) + "_stake_sized"
    df["operation_profile_label"] = df.get("operation_profile_label", "").astype(str) + "+金額調整"
    return df.drop(columns=["_priority"], errors="ignore")


def _score_policy(metrics: dict) -> float:
    if metrics["races"] <= 0:
        return -np.inf
    return (
        metrics["profit_yen"]
        + 20000.0 * (metrics["roi"] - 1.0)
        + 7000.0 * metrics["race_hit_rate"]
        + metrics["max_drawdown_yen"] * 0.45
    )


def _choose_policy(train: pd.DataFrame, *, min_races: int, min_hit_rate: float) -> tuple[dict | None, pd.DataFrame]:
    rows: list[dict] = []
    best_params: dict | None = None
    best_score = -np.inf
    for i, params in enumerate(_policy_grid()):
        selected = _apply_policy(train, params)
        metrics = _metrics(selected, f"policy_{i}")
        if metrics["races"] < min_races or metrics["race_hit_rate"] < min_hit_rate:
            continue
        if metrics["roi"] < 1.1:
            continue
        score = _score_policy(metrics)
        rows.append({"policy_id": i, "score": score, **params, **metrics})
        if score > best_score:
            best_score = score
            best_params = params
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("score", ascending=False)
    return best_params, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize simple stake sizing for ROI-mode tickets.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/race_ticket_suitability_gate_v1/suitability_gated_ticket_profiles.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/roi_mode_stake_sizing_v1")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--min-train-races", type=int, default=220)
    parser.add_argument("--min-train-hit-rate", type=float, default=0.12)
    args = parser.parse_args()

    tickets = _prepare(pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False))
    train = tickets[tickets["year"].eq(args.train_year)].copy()
    test = tickets[tickets["year"].eq(args.test_year)].copy()
    params, candidates = _choose_policy(train, min_races=args.min_train_races, min_hit_rate=args.min_train_hit_rate)
    selected = _apply_policy(tickets, params) if params else tickets.iloc[0:0].copy()

    out_dir = ensure_dir(project_path(args.output_dir))
    selected.to_csv(out_dir / "stake_sized_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    if not candidates.empty:
        candidates.head(200).to_csv(out_dir / "candidate_policies.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input": {
            "tickets_csv": args.tickets_csv,
            "train_year": args.train_year,
            "test_year": args.test_year,
            "note": "Policy is selected on train year only. Stakes are 100-yen units and capped per race.",
        },
        "selected_params": params,
        "flat_all": _metrics(tickets, "flat_all"),
        "sized_all": _metrics(selected, "stake_sized_all"),
        "flat_train": _metrics(train, "flat_train"),
        "sized_train": _metrics(selected[selected["year"].eq(args.train_year)], "stake_sized_train"),
        "flat_test": _metrics(test, "flat_test"),
        "sized_test": _metrics(selected[selected["year"].eq(args.test_year)], "stake_sized_test"),
    }
    summary["adoption_check"] = {
        "improves_all_roi": summary["sized_all"]["roi"] > summary["flat_all"]["roi"],
        "improves_all_profit": summary["sized_all"]["profit_yen"] > summary["flat_all"]["profit_yen"],
        "improves_test_roi": summary["sized_test"]["roi"] > summary["flat_test"]["roi"],
        "improves_test_profit": summary["sized_test"]["profit_yen"] > summary["flat_test"]["profit_yen"],
    }
    pd.DataFrame(
        [
            summary["flat_all"],
            summary["sized_all"],
            summary["flat_train"],
            summary["sized_train"],
            summary["flat_test"],
            summary["sized_test"],
        ]
    ).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
