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


SCENARIOS = {
    "no_haircut": {"wide": 1.00, "umaren": 1.00, "win": 1.00},
    "mild_t5": {"wide": 0.92, "umaren": 0.86, "win": 0.95},
    "normal_t3": {"wide": 0.85, "umaren": 0.75, "win": 0.90},
    "severe_late": {"wide": 0.75, "umaren": 0.62, "win": 0.84},
    "shock": {"wide": 0.65, "umaren": 0.50, "win": 0.78},
}


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _prepare(tickets: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = pd.to_numeric(df.get("year", df["race_id"].str.slice(0, 4)), errors="coerce")
    df["ticket_type"] = df.get("ticket_type", "").astype(str)
    df["stake_yen"] = _num(df.get("stake_yen"), df.index, 0.0).fillna(0.0)
    df["return_yen"] = _num(df.get("return_yen"), df.index, 0.0).fillna(0.0)
    df["hit"] = df.get("hit", False).astype(bool)
    for col, default in [
        ("ticket_sizing_score", 0.75),
        ("stake_quality_score", 0.75),
        ("late_value_survives_score", 0.70),
        ("market_overlay_score", 0.65),
        ("pair_quinella_score", 0.65),
        ("race_chaos_score", 0.50),
        ("race_solidness_score", 0.60),
        ("race_difficulty_score", 0.50),
        ("anchor_danger", 0.25),
        ("partner_danger", 0.05),
        ("stake_units", 1.0),
    ]:
        df[col] = _num(df.get(col), df.index, default).fillna(default)
    return df


def _apply_haircut(tickets: pd.DataFrame, scenario: dict[str, float], label: str) -> pd.DataFrame:
    df = tickets.copy()
    factor = df["ticket_type"].map(scenario).fillna(1.0).astype(float)
    df["return_yen"] = np.where(df["hit"], df["return_yen"] * factor, 0.0)
    df["operation_profile"] = df.get("operation_profile", "").astype(str) + f"_{label}"
    return df


def _scenario_metrics(tickets: pd.DataFrame, prefix: str) -> list[dict]:
    rows: list[dict] = []
    for name, scenario in SCENARIOS.items():
        rows.append({"scenario": name, **_metrics(_apply_haircut(tickets, scenario, name), f"{prefix}_{name}")})
    return rows


def _grid() -> list[dict]:
    policies: list[dict] = []
    for (
        sizing_min,
        late_min,
        overlay_min,
        pair_min,
        max_units_per_race,
        max_units_per_ticket,
        allow_win,
        allow_umaren,
    ) in product(
        [0.68, 0.76, 0.80],
        [0.65, 0.76],
        [0.60, 0.75],
        [0.58, 0.70],
        [7, 10],
        [4, 5],
        [False, True],
        [True],
    ):
        policies.append(
            {
                "sizing_min": sizing_min,
                "late_min": late_min,
                "overlay_min": overlay_min,
                "pair_min": pair_min,
                "max_units_per_race": max_units_per_race,
                "max_units_per_ticket": max_units_per_ticket,
                "allow_win": allow_win,
                "allow_umaren": allow_umaren,
            }
        )
    return policies


def _select(tickets: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = tickets.copy()
    mask = (
        df["ticket_sizing_score"].ge(params["sizing_min"])
        & df["late_value_survives_score"].ge(params["late_min"])
        & df["market_overlay_score"].ge(params["overlay_min"])
        & df["pair_quinella_score"].ge(params["pair_min"])
    )
    if not params["allow_win"]:
        mask &= ~df["ticket_type"].eq("win")
    if not params["allow_umaren"]:
        mask &= ~df["ticket_type"].eq("umaren")
    df = df[mask].copy()
    if df.empty:
        return df
    df["stake_units"] = df["stake_units"].clip(upper=params["max_units_per_ticket"])
    df["stake_yen"] = df["stake_units"] * 100.0

    # Recompute returns from the original payout-per-100 implied by each row.
    original_stake = _num(tickets.loc[df.index, "stake_yen"], df.index, 100.0).replace(0, np.nan)
    original_return = _num(tickets.loc[df.index, "return_yen"], df.index, 0.0)
    payout_per_100 = original_return / original_stake * 100.0
    df["return_yen"] = np.where(df["hit"], payout_per_100 * df["stake_yen"] / 100.0, 0.0)

    df["_priority"] = (
        df["ticket_sizing_score"].rank(method="first", ascending=False)
        + df["late_value_survives_score"].rank(method="first", ascending=False) / 10000.0
        + df["market_overlay_score"].rank(method="first", ascending=False) / 100000000.0
    )
    keep: list[int] = []
    for _, group in df.sort_values(["race_id", "_priority"], ascending=[True, True]).groupby("race_id", sort=False):
        units = 0.0
        for idx, row in group.iterrows():
            next_units = float(row["stake_units"])
            if units + next_units > params["max_units_per_race"]:
                continue
            units += next_units
            keep.append(idx)
    return df.loc[keep].drop(columns=["_priority"], errors="ignore").copy()


def _score(metrics_normal: dict, metrics_severe: dict) -> float:
    if metrics_normal["races"] < 1:
        return -np.inf
    return (
        metrics_normal["profit_yen"]
        + 18000.0 * (metrics_normal["roi"] - 1.0)
        + 12000.0 * max(0.0, metrics_severe["roi"] - 1.0)
        + 5000.0 * metrics_normal["race_hit_rate"]
        + metrics_normal["max_drawdown_yen"] * 0.40
        + metrics_severe["max_drawdown_yen"] * 0.25
    )


def _choose(train: pd.DataFrame, min_races: int) -> tuple[dict | None, pd.DataFrame]:
    rows: list[dict] = []
    best_params: dict | None = None
    best_score = -np.inf
    for i, params in enumerate(_grid()):
        selected = _select(train, params)
        normal = _metrics(_apply_haircut(selected, SCENARIOS["normal_t3"], "normal_t3"), f"policy_{i}_normal")
        severe = _metrics(_apply_haircut(selected, SCENARIOS["severe_late"], "severe_late"), f"policy_{i}_severe")
        if normal["races"] < min_races or normal["race_hit_rate"] < 0.12:
            continue
        if normal["roi"] < 1.15 or severe["roi"] < 0.95:
            continue
        score = _score(normal, severe)
        rows.append(
            {
                "policy_id": i,
                "score": score,
                **params,
                "normal_roi": normal["roi"],
                "normal_profit_yen": normal["profit_yen"],
                "normal_races": normal["races"],
                "normal_race_hit_rate": normal["race_hit_rate"],
                "normal_max_drawdown_yen": normal["max_drawdown_yen"],
                "severe_roi": severe["roi"],
                "severe_profit_yen": severe["profit_yen"],
                "severe_max_drawdown_yen": severe["max_drawdown_yen"],
            }
        )
        if score > best_score:
            best_score = score
            best_params = params
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("score", ascending=False)
    return best_params, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate realistic operational ROI under late-odds degradation.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/roi_mode_stake_sizing_v1/stake_sized_ticket_profiles.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/realistic_operational_roi_v1")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--min-train-races", type=int, default=180)
    args = parser.parse_args()

    tickets = _prepare(pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False))
    train = tickets[tickets["year"].eq(args.train_year)].copy()
    test = tickets[tickets["year"].eq(args.test_year)].copy()
    params, candidates = _choose(train, args.min_train_races)
    robust = _select(tickets, params) if params else tickets.iloc[0:0].copy()

    out_dir = ensure_dir(project_path(args.output_dir))
    robust.to_csv(out_dir / "realistic_robust_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    if not candidates.empty:
        candidates.head(200).to_csv(out_dir / "candidate_policies.csv", index=False, encoding="utf-8-sig")

    stress_rows = []
    for scope, frame in [("current_all", tickets), ("robust_all", robust), ("current_train", train), ("robust_train", robust[robust["year"].eq(args.train_year)]), ("current_test", test), ("robust_test", robust[robust["year"].eq(args.test_year)])]:
        stress_rows.extend({"scope": scope, **row} for row in _scenario_metrics(frame, scope))
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(out_dir / "stress_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input": {
            "tickets_csv": args.tickets_csv,
            "train_year": args.train_year,
            "test_year": args.test_year,
            "note": "Late-odds operational risk is approximated by ticket-type payout haircuts. This is a conservative proxy until timestamped odds snapshots are populated.",
            "scenarios": SCENARIOS,
        },
        "selected_params": params,
        "current_no_haircut": _metrics(tickets, "current_no_haircut"),
        "robust_no_haircut": _metrics(robust, "robust_no_haircut"),
        "current_normal_t3": _metrics(_apply_haircut(tickets, SCENARIOS["normal_t3"], "normal_t3"), "current_normal_t3"),
        "robust_normal_t3": _metrics(_apply_haircut(robust, SCENARIOS["normal_t3"], "normal_t3"), "robust_normal_t3"),
        "current_test_normal_t3": _metrics(_apply_haircut(test, SCENARIOS["normal_t3"], "normal_t3"), "current_test_normal_t3"),
        "robust_test_normal_t3": _metrics(_apply_haircut(robust[robust["year"].eq(args.test_year)], SCENARIOS["normal_t3"], "normal_t3"), "robust_test_normal_t3"),
        "current_test_severe_late": _metrics(_apply_haircut(test, SCENARIOS["severe_late"], "severe_late"), "current_test_severe_late"),
        "robust_test_severe_late": _metrics(_apply_haircut(robust[robust["year"].eq(args.test_year)], SCENARIOS["severe_late"], "severe_late"), "robust_test_severe_late"),
    }
    summary["adoption_check"] = {
        "normal_t3_all_roi_improves": summary["robust_normal_t3"]["roi"] > summary["current_normal_t3"]["roi"],
        "normal_t3_all_profit_improves": summary["robust_normal_t3"]["profit_yen"] > summary["current_normal_t3"]["profit_yen"],
        "normal_t3_test_roi_improves": summary["robust_test_normal_t3"]["roi"] > summary["current_test_normal_t3"]["roi"],
        "normal_t3_test_profit_improves": summary["robust_test_normal_t3"]["profit_yen"] > summary["current_test_normal_t3"]["profit_yen"],
        "severe_test_roi_improves": summary["robust_test_severe_late"]["roi"] > summary["current_test_severe_late"]["roi"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
