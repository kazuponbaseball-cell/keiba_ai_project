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


HAIRCUTS = {
    "none": {"wide": 1.00, "umaren": 1.00, "win": 1.00},
    "mild_t5": {"wide": 0.92, "umaren": 0.86, "win": 0.95},
    "normal_t3": {"wide": 0.85, "umaren": 0.75, "win": 0.90},
    "severe_late": {"wide": 0.75, "umaren": 0.62, "win": 0.84},
}


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _prepare(tickets: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = _num(df.get("year"), df.index, np.nan).fillna(df["race_id"].str[:4].astype(float)).astype(int)
    df["ticket_type"] = df.get("ticket_type", "").astype(str)
    df["stake_yen"] = _num(df.get("stake_yen"), df.index, 100.0).fillna(100.0)
    df["return_yen"] = _num(df.get("return_yen"), df.index, 0.0).fillna(0.0)
    df["hit"] = df.get("hit", False).astype(bool)

    anchor_odds = _num(df.get("anchor_odds"), df.index, np.nan).fillna(_num(df.get("odds"), df.index, np.nan))
    anchor_odds = anchor_odds.fillna(_num(df.get("market_odds_live_or_final"), df.index, np.nan)).clip(lower=1.0)
    partner_odds = _num(df.get("partner_odds"), df.index, np.nan).fillna(anchor_odds).clip(lower=1.0)
    df["minodds_anchor_odds"] = anchor_odds.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["minodds_partner_odds"] = partner_odds.replace([np.inf, -np.inf], np.nan).fillna(df["minodds_anchor_odds"])

    win_raw = _num(df.get("anchor_ai_win_prob"), df.index, np.nan).fillna(_num(df.get("ai_win_prob_proxy"), df.index, np.nan))
    place_a = _num(df.get("anchor_place_score"), df.index, 0.0).fillna(0.0)
    place_b = _num(df.get("partner_place_score"), df.index, 0.0).fillna(0.0)
    pair_q = _num(df.get("pair_quinella_score"), df.index, 0.0).fillna(0.0)
    pair_score = _num(df.get("pair_score"), df.index, 0.0).fillna(pair_q)
    sizing = _num(df.get("ticket_sizing_score"), df.index, 0.0).fillna(0.0)
    late = _num(df.get("late_value_survives_score"), df.index, 0.0).fillna(0.0)

    df["prob_raw_score"] = np.select(
        [
            df["ticket_type"].eq("win"),
            df["ticket_type"].eq("wide"),
            df["ticket_type"].eq("umaren"),
        ],
        [
            win_raw,
            0.36 * pair_score + 0.30 * np.sqrt((place_a.clip(0, 1) * place_b.clip(0, 1)).clip(0, 1)) + 0.20 * late + 0.14 * sizing,
            0.42 * pair_q + 0.22 * win_raw.fillna(0.0) + 0.18 * late + 0.18 * sizing,
        ],
        default=pair_q,
    )
    df["prob_raw_score"] = _num(df["prob_raw_score"], df.index, 0.0).fillna(0.0).clip(0.0, 1.0)

    # Pre-race available payout proxy. For historical rows we do not know every
    # losing pair's final odds, so this intentionally uses only pre-race odds.
    df["quote_pay_proxy_per100"] = np.select(
        [
            df["ticket_type"].eq("win"),
            df["ticket_type"].eq("wide"),
            df["ticket_type"].eq("umaren"),
        ],
        [
            df["minodds_anchor_odds"] * 100.0,
            100.0 * (np.sqrt(df["minodds_anchor_odds"] * df["minodds_partner_odds"]) * 0.45).clip(1.1, 120.0),
            100.0 * (df["minodds_anchor_odds"] * df["minodds_partner_odds"] * 0.32).clip(1.3, 260.0),
        ],
        default=0.0,
    )
    return df


def _calibrate_by_type(train: pd.DataFrame, apply_to: pd.DataFrame, bins: int) -> pd.DataFrame:
    out = apply_to.copy()
    out["ticket_hit_prob"] = np.nan
    out["prob_calibration_bin"] = ""
    global_rate = float(train["hit"].mean()) if len(train) else 0.05
    for ticket_type, apply_g in out.groupby("ticket_type"):
        train_g = train[train["ticket_type"].eq(ticket_type)].copy()
        idx = apply_g.index
        if len(train_g) < max(20, bins * 5):
            out.loc[idx, "ticket_hit_prob"] = global_rate
            out.loc[idx, "prob_calibration_bin"] = "global"
            continue
        raw_train = train_g["prob_raw_score"].rank(method="first")
        try:
            train_g["_bin"] = pd.qcut(raw_train, bins, labels=False, duplicates="drop")
        except ValueError:
            out.loc[idx, "ticket_hit_prob"] = float(train_g["hit"].mean())
            out.loc[idx, "prob_calibration_bin"] = "type_mean"
            continue
        grouped = (
            train_g.groupby("_bin", observed=True)
            .agg(n=("hit", "size"), hit_rate=("hit", "mean"), raw_min=("prob_raw_score", "min"), raw_max=("prob_raw_score", "max"))
            .reset_index()
            .sort_values("raw_min")
        )
        # Smoothing keeps small bins from becoming all-or-nothing.
        type_rate = float(train_g["hit"].mean())
        grouped["prob"] = (grouped["hit_rate"] * grouped["n"] + type_rate * 12.0) / (grouped["n"] + 12.0)
        # Higher raw score should not imply a materially lower probability.
        grouped["prob"] = grouped["prob"].cummax().clip(0.001, 0.85)
        raw_apply = apply_g["prob_raw_score"]
        assigned = pd.Series(type_rate, index=idx, dtype=float)
        labels = pd.Series("fallback", index=idx, dtype=object)
        for _, row in grouped.iterrows():
            mask = raw_apply.between(row["raw_min"], row["raw_max"], inclusive="both")
            assigned.loc[mask[mask].index] = float(row["prob"])
            labels.loc[mask[mask].index] = f"{ticket_type}_bin{int(row['_bin'])}"
        # Outside train range gets nearest edge probability.
        if not grouped.empty:
            low = raw_apply.lt(float(grouped["raw_min"].min()))
            high = raw_apply.gt(float(grouped["raw_max"].max()))
            assigned.loc[low[low].index] = float(grouped.iloc[0]["prob"])
            assigned.loc[high[high].index] = float(grouped.iloc[-1]["prob"])
            labels.loc[low[low].index] = f"{ticket_type}_below"
            labels.loc[high[high].index] = f"{ticket_type}_above"
        out.loc[idx, "ticket_hit_prob"] = assigned
        out.loc[idx, "prob_calibration_bin"] = labels
    out["ticket_hit_prob"] = _num(out["ticket_hit_prob"], out.index, global_rate).fillna(global_rate).clip(0.001, 0.85)
    return out


def _apply_haircut(tickets: pd.DataFrame, scenario: str) -> pd.DataFrame:
    df = tickets.copy()
    factor = df["ticket_type"].map(HAIRCUTS[scenario]).fillna(1.0).astype(float)
    df["return_yen"] = np.where(df["hit"].astype(bool), _num(df.get("return_yen"), df.index, 0.0).fillna(0.0) * factor, 0.0)
    return df


def _add_requirements(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    haircut = out["ticket_type"].map(HAIRCUTS[params["scenario"]]).fillna(1.0).astype(float)
    margin = np.select(
        [out["ticket_type"].eq("win"), out["ticket_type"].eq("wide"), out["ticket_type"].eq("umaren")],
        [params["win_margin"], params["wide_margin"], params["umaren_margin"]],
        default=params["wide_margin"],
    )
    min_floor = np.select(
        [out["ticket_type"].eq("win"), out["ticket_type"].eq("wide"), out["ticket_type"].eq("umaren")],
        [params["win_floor"], params["wide_floor"], params["umaren_floor"]],
        default=0.0,
    )
    max_floor = np.select(
        [out["ticket_type"].eq("win"), out["ticket_type"].eq("wide"), out["ticket_type"].eq("umaren")],
        [params["win_cap"], params["wide_cap"], params["umaren_cap"]],
        default=1e9,
    )
    out["required_pay_per100"] = (100.0 * margin / (out["ticket_hit_prob"] * haircut).replace(0, np.nan)).clip(lower=min_floor)
    out["required_pay_per100"] = np.minimum(out["required_pay_per100"], max_floor)
    out["expected_roi_after_slippage"] = out["ticket_hit_prob"] * out["quote_pay_proxy_per100"] * haircut / 100.0
    out["min_acceptable_odds"] = out["required_pay_per100"] / 100.0
    out["quote_odds_proxy"] = out["quote_pay_proxy_per100"] / 100.0
    out["min_odds_margin_ratio"] = out["quote_pay_proxy_per100"] / out["required_pay_per100"].replace(0, np.nan)
    return out


def _apply_gate(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = _add_requirements(df, params)
    mask = out["quote_pay_proxy_per100"].ge(out["required_pay_per100"])
    mask &= out["expected_roi_after_slippage"].ge(params["min_expected_roi"])
    mask &= out["ticket_hit_prob"].ge(params["min_prob"])
    if not params["allow_win"]:
        mask &= ~out["ticket_type"].eq("win")
    selected = out[mask].copy()
    if selected.empty:
        return selected

    # Keep the already optimized stake plan, but optionally cap exposure after
    # applying the new odds/probability gate.
    selected["stake_units"] = _num(selected.get("stake_units"), selected.index, np.nan).fillna(
        _num(selected.get("stake_yen"), selected.index, 100.0) / 100.0
    )
    selected["stake_units"] = selected["stake_units"].clip(upper=params["max_units_per_ticket"])
    selected["stake_yen"] = selected["stake_units"] * 100.0

    original_stake = _num(out.loc[selected.index, "stake_yen"], selected.index, 100.0).replace(0, np.nan)
    original_return = _num(out.loc[selected.index, "return_yen"], selected.index, 0.0)
    pay_per100 = original_return / original_stake * 100.0
    selected["return_yen"] = np.where(selected["hit"].astype(bool), pay_per100 * selected["stake_yen"] / 100.0, 0.0)

    selected["_priority"] = (
        selected["expected_roi_after_slippage"].rank(method="first", ascending=False)
        + selected["ticket_hit_prob"].rank(method="first", ascending=False) / 10000.0
        + selected["ticket_sizing_score"].rank(method="first", ascending=False) / 100000000.0
    )
    keep: list[int] = []
    for _, group in selected.sort_values(["race_id", "_priority"], ascending=[True, True]).groupby("race_id", sort=False):
        units = 0.0
        for idx, row in group.iterrows():
            next_units = float(row["stake_units"])
            if units + next_units > params["max_units_per_race"]:
                continue
            units += next_units
            keep.append(idx)
    selected = selected.loc[keep].drop(columns=["_priority"], errors="ignore").copy()
    selected["operation_profile"] = selected.get("operation_profile", "").astype(str) + "_minodds_prob"
    selected["operation_profile_label"] = selected.get("operation_profile_label", "").astype(str) + "+最低オッズ"
    return selected


def _grid() -> list[dict]:
    rows: list[dict] = []
    for (
        scenario,
        win_margin,
        wide_margin,
        umaren_margin,
        min_expected_roi,
        min_prob,
        allow_win,
        max_units_per_ticket,
    ) in product(
        ["mild_t5", "normal_t3", "severe_late"],
        [0.90, 1.05],
        [0.90, 1.05],
        [0.90, 1.05, 1.20],
        [0.95, 1.05],
        [0.005, 0.04],
        [False, True],
        [5],
    ):
        rows.append(
            {
                "scenario": scenario,
                "win_margin": win_margin,
                "wide_margin": wide_margin,
                "umaren_margin": umaren_margin,
                "min_expected_roi": min_expected_roi,
                "min_prob": min_prob,
                "allow_win": allow_win,
                "max_units_per_ticket": max_units_per_ticket,
                "max_units_per_race": 10,
                "win_floor": 180.0,
                "wide_floor": 220.0,
                "umaren_floor": 700.0,
                "win_cap": 12000.0,
                "wide_cap": 12000.0,
                "umaren_cap": 30000.0,
            }
        )
    return rows


def _score(metrics_normal: dict, metrics_severe: dict) -> float:
    return (
        metrics_normal["profit_yen"]
        + 18000.0 * (metrics_normal["roi"] - 1.0)
        + 10000.0 * max(0.0, metrics_severe["roi"] - 1.0)
        + 5000.0 * metrics_normal["race_hit_rate"]
        + 0.35 * metrics_normal["max_drawdown_yen"]
        + 0.20 * metrics_severe["max_drawdown_yen"]
    )


def _choose(train: pd.DataFrame, min_train_races: int) -> tuple[dict | None, pd.DataFrame]:
    rows: list[dict] = []
    best_params = None
    best_score = -np.inf
    for i, params in enumerate(_grid()):
        selected = _apply_gate(train, params)
        normal = _metrics(_apply_haircut(selected, "normal_t3"), f"policy_{i}_normal")
        severe = _metrics(_apply_haircut(selected, "severe_late"), f"policy_{i}_severe")
        if normal["races"] < min_train_races or normal["race_hit_rate"] < 0.10:
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
                "normal_hit": normal["race_hit_rate"],
                "normal_dd": normal["max_drawdown_yen"],
                "severe_roi": severe["roi"],
                "severe_profit_yen": severe["profit_yen"],
                "severe_dd": severe["max_drawdown_yen"],
            }
        )
        if score > best_score:
            best_score = score
            best_params = params
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("score", ascending=False)
    return best_params, table


def _prob_segments(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for ticket_type, group in df.groupby("ticket_type"):
        try:
            bins = pd.qcut(group["ticket_hit_prob"], 4, duplicates="drop")
        except ValueError:
            continue
        for b, g in group.assign(_bin=bins).groupby("_bin", observed=True):
            row = _metrics(g, str(b))
            row.update(
                {
                    "ticket_type": ticket_type,
                    "prob_bin": str(b),
                    "avg_prob": float(g["ticket_hit_prob"].mean()),
                    "actual_ticket_hit": float(g["hit"].mean()),
                    "avg_required_pay": float(g["required_pay_per100"].mean()) if "required_pay_per100" in g else np.nan,
                    "avg_quote_pay_proxy": float(g["quote_pay_proxy_per100"].mean()),
                    "avg_expected_roi_after_slippage": float(g["expected_roi_after_slippage"].mean())
                    if "expected_roi_after_slippage" in g
                    else np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize minimum acceptable odds using ticket-type hit probability and late-odds slippage.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/roi_mode_stake_sizing_v1/stake_sized_ticket_profiles.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/min_odds_ticket_prob_gate_v1")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--min-train-races", type=int, default=180)
    args = parser.parse_args()

    tickets = _prepare(pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False))
    train_raw = tickets[tickets["year"].eq(args.train_year)].copy()
    test_raw = tickets[tickets["year"].eq(args.test_year)].copy()
    train = _calibrate_by_type(train_raw, train_raw, args.bins)
    test = _calibrate_by_type(train_raw, test_raw, args.bins)
    calibrated = pd.concat([train, test], ignore_index=False, sort=False).sort_index()

    params, candidates = _choose(train, args.min_train_races)
    annotated = _add_requirements(calibrated, params) if params else calibrated.copy()
    selected = _apply_gate(calibrated, params) if params else calibrated.iloc[0:0].copy()

    out_dir = ensure_dir(project_path(args.output_dir))
    calibrated.to_csv(out_dir / "ticket_prob_calibrated_profiles.csv", index=False, encoding="utf-8-sig")
    annotated.to_csv(out_dir / "min_odds_annotated_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out_dir / "min_odds_gated_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    if not candidates.empty:
        candidates.head(200).to_csv(out_dir / "candidate_policies.csv", index=False, encoding="utf-8-sig")
    _prob_segments(selected if not selected.empty else calibrated).to_csv(out_dir / "probability_segments.csv", index=False, encoding="utf-8-sig")

    base_test = test.copy()
    selected_test = selected[selected["year"].eq(args.test_year)].copy()
    summary = {
        "input": {
            "tickets_csv": args.tickets_csv,
            "train_year": args.train_year,
            "test_year": args.test_year,
            "bins": args.bins,
            "note": "Ticket probabilities are calibrated by ticket type on train year only. Minimum acceptable odds use live-available payout proxies and late-odds haircut assumptions.",
        },
        "selected_params": params,
        "base_all": _metrics(calibrated, "base_all"),
        "minodds_all": _metrics(selected, "minodds_all"),
        "base_test": _metrics(base_test, "base_test"),
        "minodds_test": _metrics(selected_test, "minodds_test"),
        "base_test_normal_t3": _metrics(_apply_haircut(base_test, "normal_t3"), "base_test_normal_t3"),
        "minodds_test_normal_t3": _metrics(_apply_haircut(selected_test, "normal_t3"), "minodds_test_normal_t3"),
        "base_test_severe_late": _metrics(_apply_haircut(base_test, "severe_late"), "base_test_severe_late"),
        "minodds_test_severe_late": _metrics(_apply_haircut(selected_test, "severe_late"), "minodds_test_severe_late"),
    }
    summary["adoption_check"] = {
        "all_roi_improves": summary["minodds_all"]["roi"] > summary["base_all"]["roi"],
        "all_profit_improves": summary["minodds_all"]["profit_yen"] > summary["base_all"]["profit_yen"],
        "test_normal_t3_roi_improves": summary["minodds_test_normal_t3"]["roi"] > summary["base_test_normal_t3"]["roi"],
        "test_normal_t3_profit_improves": summary["minodds_test_normal_t3"]["profit_yen"] > summary["base_test_normal_t3"]["profit_yen"],
        "test_severe_roi_improves": summary["minodds_test_severe_late"]["roi"] > summary["base_test_severe_late"]["roi"],
    }
    pd.DataFrame(
        [
            summary["base_all"],
            summary["minodds_all"],
            summary["base_test"],
            summary["minodds_test"],
            summary["base_test_normal_t3"],
            summary["minodds_test_normal_t3"],
            summary["base_test_severe_late"],
            summary["minodds_test_severe_late"],
        ]
    ).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
