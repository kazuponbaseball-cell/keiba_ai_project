from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_operational_win_addon import _json_default, _metrics, _num
from src.utils.paths import ensure_dir, project_path


def _series_num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    return _num(df.get(col), df.index, default)


def _prepare(tickets: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["ticket_type"] = df["ticket_type"].astype(str)
    df["year"] = pd.to_numeric(df.get("year", df["race_id"].str.slice(0, 4)), errors="coerce")
    if "stake_yen" not in df.columns:
        df["stake_yen"] = 100.0
    if "return_yen" not in df.columns:
        df["return_yen"] = 0.0
    df["hit"] = df.get("hit", False).astype(bool)

    anchor_odds = _series_num(df, "anchor_odds", np.nan).fillna(_series_num(df, "odds", np.nan))
    anchor_odds = anchor_odds.fillna(_series_num(df, "market_odds_live_or_final", np.nan))
    partner_odds = _series_num(df, "partner_odds", np.nan)
    partner_odds = partner_odds.fillna(anchor_odds)
    anchor_odds = anchor_odds.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=1.0)
    partner_odds = partner_odds.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=1.0)
    df["ev_anchor_odds"] = anchor_odds
    df["ev_partner_odds"] = partner_odds

    raw_win = _series_num(df, "win_prob", np.nan).fillna(_series_num(df, "ai_win_prob_proxy", np.nan))
    raw_wide = _series_num(df, "pair_score", np.nan).fillna(_series_num(df, "pair_quinella_score", np.nan))
    raw_umaren = _series_num(df, "pair_quinella_score", np.nan)
    raw_trio = _series_num(df, "trio_model_prob", np.nan).fillna(_series_num(df, "pair_quinella_score", np.nan))
    df["ev_raw_prob_score"] = np.select(
        [
            df["ticket_type"].eq("win"),
            df["ticket_type"].eq("wide"),
            df["ticket_type"].eq("umaren"),
            df["ticket_type"].eq("trio"),
        ],
        [raw_win, raw_wide, raw_umaren, raw_trio],
        default=np.nan,
    )
    df["ev_raw_prob_score"] = pd.to_numeric(df["ev_raw_prob_score"], errors="coerce").fillna(0.0).clip(lower=0.0)

    wide_pay_proxy = 100.0 * (np.sqrt(anchor_odds * partner_odds) * 0.45).clip(lower=1.1, upper=120.0)
    umaren_pay_proxy = 100.0 * (anchor_odds * partner_odds * 0.42).clip(lower=1.5, upper=250.0)
    trio_pay_proxy = 100.0 * (anchor_odds * partner_odds * 1.8).clip(lower=3.0, upper=650.0)
    win_pay_proxy = anchor_odds * 100.0
    df["ev_pay_proxy_per100"] = np.select(
        [
            df["ticket_type"].eq("win"),
            df["ticket_type"].eq("wide"),
            df["ticket_type"].eq("umaren"),
            df["ticket_type"].eq("trio"),
        ],
        [win_pay_proxy, wide_pay_proxy, umaren_pay_proxy, trio_pay_proxy],
        default=0.0,
    )
    return df


def _calibrate(train: pd.DataFrame, apply_to: pd.DataFrame) -> pd.DataFrame:
    out = apply_to.copy()
    out["ev_prob_est"] = 0.0
    for ticket_type, test_g in out.groupby("ticket_type"):
        train_g = train[train["ticket_type"].eq(ticket_type)]
        if train_g.empty:
            continue
        raw_mean = float(_series_num(train_g, "ev_raw_prob_score", 0.0).mean())
        hit_rate = float(train_g["hit"].astype(bool).mean())
        if not np.isfinite(raw_mean) or raw_mean <= 1e-9:
            scale = 0.0
        else:
            scale = hit_rate / raw_mean
        idx = test_g.index
        out.loc[idx, "ev_prob_est"] = (_series_num(test_g, "ev_raw_prob_score", 0.0) * scale).clip(0.001, 0.85)
    out["ticket_ev_proxy"] = out["ev_prob_est"] * out["ev_pay_proxy_per100"] / 100.0
    return out


def _grid() -> list[dict]:
    rows: list[dict] = []
    for win_min, win_max, wide_min, wide_max, umaren_min, umaren_max, trio_min, trio_max, max_diff in product(
        [0.00, 0.75],
        [1.80, 999.0],
        [0.00, 0.40, 0.55],
        [1.30, 999.0],
        [0.00, 0.80, 1.10],
        [1.60, 2.30, 999.0],
        [0.00, 0.80],
        [3.00, 999.0],
        [0.75, 1.01],
    ):
        if win_min > win_max or wide_min > wide_max or umaren_min > umaren_max or trio_min > trio_max:
            continue
        rows.append(
            {
                "win_ev_min": win_min,
                "win_ev_max": win_max,
                "wide_ev_min": wide_min,
                "wide_ev_max": wide_max,
                "umaren_ev_min": umaren_min,
                "umaren_ev_max": umaren_max,
                "trio_ev_min": trio_min,
                "trio_ev_max": trio_max,
                "difficulty_max": max_diff,
            }
        )
    return rows


def _apply_gate(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    t = df.copy()
    thresholds = np.select(
        [
            t["ticket_type"].eq("win"),
            t["ticket_type"].eq("wide"),
            t["ticket_type"].eq("umaren"),
            t["ticket_type"].eq("trio"),
        ],
        [
            params["win_ev_min"],
            params["wide_ev_min"],
            params["umaren_ev_min"],
            params["trio_ev_min"],
        ],
        default=999.0,
    )
    max_thresholds = np.select(
        [
            t["ticket_type"].eq("win"),
            t["ticket_type"].eq("wide"),
            t["ticket_type"].eq("umaren"),
            t["ticket_type"].eq("trio"),
        ],
        [
            params["win_ev_max"],
            params["wide_ev_max"],
            params["umaren_ev_max"],
            params["trio_ev_max"],
        ],
        default=999.0,
    )
    difficulty = _series_num(t, "race_difficulty_score", np.nan).fillna(_series_num(t, "difficulty", 0.0))
    selected = t[
        (t["ticket_ev_proxy"].ge(thresholds))
        & (t["ticket_ev_proxy"].le(max_thresholds))
        & difficulty.le(params["difficulty_max"])
    ].copy()
    return selected


def _choose_policy(train_calibrated: pd.DataFrame, min_train_tickets: int, min_race_hit: float) -> tuple[dict | None, dict | None]:
    best_params = None
    best_metrics = None
    best_score = -np.inf
    for params in _grid():
        selected = _apply_gate(train_calibrated, params)
        metrics = _metrics(selected, "train_ev_gate")
        if metrics["tickets"] < min_train_tickets or metrics["race_hit_rate"] < min_race_hit:
            continue
        score = (
            (metrics["roi"] - 1.0) * 100.0
            + metrics["race_hit_rate"] * 10.0
            + np.log1p(metrics["tickets"]) * 0.18
            - max(0.0, abs(metrics["max_drawdown_yen"]) / 10000.0) * 0.12
        )
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics
    return best_params, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize ticket-type-specific expected-value gates using pre-race odds proxies.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/operational_win_addon_1pt_v1/combined_ticket_profiles.csv")
    parser.add_argument("--extra-tickets-csv", default="", help="Optional additional tickets, e.g. 3-renpuku value model tickets.")
    parser.add_argument("--output-dir", default="outputs/analysis/ticket_type_ev_gate_v1")
    parser.add_argument("--min-train-tickets", type=int, default=100)
    parser.add_argument("--min-race-hit", type=float, default=0.08)
    args = parser.parse_args()

    base = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    frames = [base]
    if args.extra_tickets_csv:
        extra_path = project_path(args.extra_tickets_csv)
        if extra_path.exists():
            frames.append(pd.read_csv(extra_path, dtype={"race_id": str}, low_memory=False))
    tickets = _prepare(pd.concat(frames, ignore_index=True, sort=False))
    years = sorted(int(y) for y in tickets["year"].dropna().unique())
    wf_rows: list[dict] = []
    selected_frames: list[pd.DataFrame] = []

    for year in years[1:]:
        train = tickets[tickets["year"].lt(year)].copy()
        test = tickets[tickets["year"].eq(year)].copy()
        if train.empty or test.empty:
            wf_rows.append({"year": year, "selected": False})
            continue
        train_cal = _calibrate(train, train)
        test_cal = _calibrate(train, test)
        params, train_metrics = _choose_policy(train_cal, args.min_train_tickets, args.min_race_hit)
        if params is None:
            wf_rows.append({"year": year, "selected": False})
            continue
        test_selected = _apply_gate(test_cal, params)
        test_metrics = _metrics(test_selected, f"test_{year}_ev_gate")
        selected_frames.append(test_selected.assign(test_year=year))
        wf_rows.append(
            {
                "year": year,
                "selected": True,
                **{f"param_{k}": v for k, v in params.items()},
                **{f"train_{k}": v for k, v in (train_metrics or {}).items() if k != "policy"},
                **{f"test_{k}": v for k, v in test_metrics.items() if k != "policy"},
            }
        )

    selected = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    test_years = sorted(selected["year"].dropna().astype(int).unique()) if not selected.empty else years[1:]
    ungated = tickets[tickets["year"].isin(test_years)].copy()
    summary = {
        "input": {
            "tickets_csv": args.tickets_csv,
            "extra_tickets_csv": args.extra_tickets_csv,
            "test_years": test_years,
            "note": "EV gate uses model probability calibrated on prior years and pre-race odds proxies. Actual payoff is used only for backtest returns.",
        },
        "ungated": _metrics(ungated, "ungated"),
        "ev_gated": _metrics(selected, "ev_gated"),
    }
    summary["delta_roi"] = summary["ev_gated"]["roi"] - summary["ungated"]["roi"]
    summary["delta_profit_yen"] = summary["ev_gated"]["profit_yen"] - summary["ungated"]["profit_yen"]

    out_dir = ensure_dir(project_path(args.output_dir))
    pd.DataFrame(wf_rows).to_csv(out_dir / "walkforward_ev_gate_summary.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out_dir / "ev_gated_tickets.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary["ungated"], summary["ev_gated"]]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
