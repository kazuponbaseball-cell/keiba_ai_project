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
    for col, default in [
        ("race_difficulty_score", 0.5),
        ("race_market_top3_prob_sum", 0.65),
        ("race_favorite_danger", 0.45),
        ("race_field_size", 14.0),
        ("race_pace_collapse", 0.2),
        ("race_front_pressure", 0.25),
        ("race_young_or_maiden_risk", 0.0),
        ("pair_quinella_score", 0.65),
        ("anchor_win_score", 0.35),
        ("anchor_danger", 0.25),
        ("partner_danger", 0.05),
        ("late_value_survives_score", 0.7),
    ]:
        df[col] = _num(df.get(col), df.index, default).fillna(default)
    df["race_solidness_score"] = (
        0.45 * df["race_market_top3_prob_sum"]
        + 0.25 * (1.0 - df["race_difficulty_score"])
        + 0.20 * (1.0 - df["race_favorite_danger"])
        + 0.10 * (1.0 - df["race_pace_collapse"])
    ).clip(0.0, 1.0)
    df["race_chaos_score"] = (
        0.35 * df["race_difficulty_score"]
        + 0.20 * df["race_pace_collapse"]
        + 0.15 * df["race_favorite_danger"]
        + 0.15 * (df["race_field_size"] / 18.0)
        + 0.15 * df["race_young_or_maiden_risk"]
    ).clip(0.0, 1.0)
    return df


def _policy_grid() -> list[dict]:
    policies: list[dict] = []
    for (
        wide_chaos_min,
        wide_solid_max,
        wide_pair_min,
        wide_front_min,
        umaren_chaos_min,
        umaren_chaos_max,
        umaren_solid_max,
        umaren_pace_max,
        umaren_anchor_danger_max,
        win_chaos_min,
        win_pair_min,
    ) in product(
        [0.0, 0.40],
        [0.70, 1.01],
        [0.58, 0.68],
        [0.0, 0.24],
        [0.0, 0.36],
        [0.70, 1.01],
        [0.78, 1.01],
        [0.30, 1.01],
        [0.34, 0.50],
        [0.0, 0.54],
        [0.70, 0.80],
    ):
        if umaren_chaos_min > umaren_chaos_max:
            continue
        policies.append(
            {
                "wide_chaos_min": wide_chaos_min,
                "wide_solid_max": wide_solid_max,
                "wide_pair_min": wide_pair_min,
                "wide_front_min": wide_front_min,
                "umaren_chaos_min": umaren_chaos_min,
                "umaren_chaos_max": umaren_chaos_max,
                "umaren_solid_max": umaren_solid_max,
                "umaren_pace_max": umaren_pace_max,
                "umaren_anchor_danger_max": umaren_anchor_danger_max,
                "win_chaos_min": win_chaos_min,
                "win_pair_min": win_pair_min,
            }
        )
    return policies


def _select(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    wide = df["ticket_type"].eq("wide") & df["race_chaos_score"].ge(params["wide_chaos_min"])
    wide &= df["race_solidness_score"].le(params["wide_solid_max"])
    wide &= df["pair_quinella_score"].ge(params["wide_pair_min"])
    wide &= df["race_front_pressure"].ge(params["wide_front_min"])

    umaren = df["ticket_type"].eq("umaren") & df["race_chaos_score"].between(
        params["umaren_chaos_min"], params["umaren_chaos_max"]
    )
    umaren &= df["race_solidness_score"].le(params["umaren_solid_max"])
    umaren &= df["race_pace_collapse"].le(params["umaren_pace_max"])
    umaren &= df["anchor_danger"].le(params["umaren_anchor_danger_max"])

    win = df["ticket_type"].eq("win") & df["race_chaos_score"].ge(params["win_chaos_min"])
    win &= df["pair_quinella_score"].ge(params["win_pair_min"])
    return df[wide | umaren | win].copy()


def _score_policy(metrics: dict) -> float:
    if metrics["races"] <= 0:
        return -np.inf
    return (
        metrics["profit_yen"]
        + 24000.0 * (metrics["roi"] - 1.0)
        + 6000.0 * metrics["race_hit_rate"]
        + metrics["max_drawdown_yen"] * 0.35
    )


def _choose_policy(train: pd.DataFrame, *, min_races: int, min_tickets: int) -> tuple[dict | None, pd.DataFrame]:
    rows: list[dict] = []
    best_params: dict | None = None
    best_score = -np.inf
    for i, params in enumerate(_policy_grid()):
        selected = _select(train, params)
        metrics = _metrics(selected, f"policy_{i}")
        if metrics["races"] < min_races or metrics["tickets"] < min_tickets:
            continue
        if metrics["race_hit_rate"] < 0.12 or metrics["roi"] < 1.1:
            continue
        score = _score_policy(metrics)
        row = {"policy_id": i, "score": score, **params, **metrics}
        rows.append(row)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame()


def _segments(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    features = [
        "race_solidness_score",
        "race_chaos_score",
        "race_difficulty_score",
        "race_market_top3_prob_sum",
        "race_pace_collapse",
        "race_front_pressure",
        "pair_quinella_score",
    ]
    for ticket_type, part in df.groupby("ticket_type"):
        for feature in features:
            values = pd.to_numeric(part[feature], errors="coerce")
            try:
                bins = pd.qcut(values, 4, duplicates="drop")
            except ValueError:
                continue
            tmp = part.assign(_bin=bins)
            for bin_label, g in tmp.groupby("_bin", observed=True):
                metrics = _metrics(g, f"{ticket_type}_{feature}_{bin_label}")
                rows.append(
                    {
                        "ticket_type": ticket_type,
                        "feature": feature,
                        "bin": str(bin_label),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply race-ticket suitability gates to operational tickets.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/late_value_survival_gate_v1/gated_ticket_profiles.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/race_ticket_suitability_gate_v1")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--min-train-races", type=int, default=220)
    parser.add_argument("--min-train-tickets", type=int, default=320)
    args = parser.parse_args()

    tickets = _prepare(pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False))
    train = tickets[tickets["year"].eq(args.train_year)].copy()
    test = tickets[tickets["year"].eq(args.test_year)].copy()
    params, candidates = _choose_policy(train, min_races=args.min_train_races, min_tickets=args.min_train_tickets)
    if params is None:
        selected = tickets.iloc[0:0].copy()
    else:
        selected = _select(tickets, params)

    out_dir = ensure_dir(project_path(args.output_dir))
    selected.to_csv(out_dir / "suitability_gated_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    if not candidates.empty:
        candidates.head(200).to_csv(out_dir / "candidate_policies.csv", index=False, encoding="utf-8-sig")
    _segments(tickets).to_csv(out_dir / "suitability_segments.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input": {
            "tickets_csv": args.tickets_csv,
            "train_year": args.train_year,
            "test_year": args.test_year,
            "note": "Policy is chosen on the train year only. It gates ticket types by race solidity/chaos and ticket-specific suitability.",
        },
        "selected_params": params,
        "ungated_all": _metrics(tickets, "ungated_all"),
        "gated_all": _metrics(selected, "race_ticket_suitability_gated_all"),
        "ungated_train": _metrics(train, "ungated_train"),
        "gated_train": _metrics(selected[selected["year"].eq(args.train_year)], "gated_train"),
        "ungated_test": _metrics(test, "ungated_test"),
        "gated_test": _metrics(selected[selected["year"].eq(args.test_year)], "gated_test"),
    }
    summary["adoption_check"] = {
        "improves_all_roi": summary["gated_all"]["roi"] > summary["ungated_all"]["roi"],
        "improves_all_profit": summary["gated_all"]["profit_yen"] > summary["ungated_all"]["profit_yen"],
        "improves_test_roi": summary["gated_test"]["roi"] > summary["ungated_test"]["roi"],
        "improves_test_profit": summary["gated_test"]["profit_yen"] > summary["ungated_test"]["profit_yen"],
    }
    pd.DataFrame(
        [
            summary["ungated_all"],
            summary["gated_all"],
            summary["ungated_train"],
            summary["gated_train"],
            summary["ungated_test"],
            summary["gated_test"],
        ]
    ).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
