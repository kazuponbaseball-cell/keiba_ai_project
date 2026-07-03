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


def _prepare(tickets: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["ticket_type"] = df["ticket_type"].astype(str)
    df["year"] = pd.to_numeric(df.get("year", df["race_id"].str.slice(0, 4)), errors="coerce")
    df["anchor_danger_norm"] = _num(df.get("anchor_danger"), df.index, np.nan).fillna(_num(df.get("danger"), df.index, 0.0))
    df["partner_danger_norm"] = _num(df.get("partner_danger"), df.index, 0.0).fillna(0.0)
    df["anchor_pop_norm"] = _num(df.get("anchor_pop"), df.index, np.nan).fillna(_num(df.get("pop_rank_num"), df.index, 99.0))
    df["partner_pop_norm"] = _num(df.get("partner_pop"), df.index, 99.0).fillna(99.0)
    return df


def _apply_gate(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    t = df.copy()
    keep = pd.Series(True, index=t.index)

    win = t["ticket_type"].eq("win")
    keep &= ~(win & t["anchor_danger_norm"].gt(params["win_anchor_danger_max"]))

    umaren = t["ticket_type"].eq("umaren")
    keep &= ~(umaren & t["anchor_danger_norm"].gt(params["umaren_anchor_danger_max"]))
    keep &= ~(umaren & t["partner_danger_norm"].gt(params["umaren_partner_danger_max"]))

    wide = t["ticket_type"].eq("wide")
    # For wide, dangerous popular anchors can still be usable if the partner is very low-risk.
    wide_bad_anchor = wide & t["anchor_danger_norm"].gt(params["wide_anchor_danger_hard_max"])
    wide_soft_anchor = (
        wide
        & t["anchor_danger_norm"].gt(params["wide_anchor_danger_soft_max"])
        & t["partner_danger_norm"].gt(params["wide_partner_danger_for_soft_anchor_max"])
    )
    keep &= ~(wide_bad_anchor | wide_soft_anchor)

    trio = t["ticket_type"].eq("trio")
    if trio.any():
        keep &= ~(trio & t["anchor_danger_norm"].gt(params["trio_anchor_danger_max"]))

    return t[keep].copy()


def _grid() -> list[dict]:
    rows: list[dict] = []
    for win_max, umaren_anchor_max, umaren_partner_max, wide_soft, wide_partner_soft, wide_hard, trio_max in product(
        [0.09, 0.11, 0.13, 0.20],
        [0.30, 0.38, 0.50],
        [0.05, 0.08, 0.16, 0.32],
        [0.30, 0.38, 0.50],
        [0.04, 0.08, 0.16, 0.32],
        [0.42, 0.50],
        [0.25, 0.40, 0.65],
    ):
        if wide_soft > wide_hard:
            continue
        rows.append(
            {
                "win_anchor_danger_max": win_max,
                "umaren_anchor_danger_max": umaren_anchor_max,
                "umaren_partner_danger_max": umaren_partner_max,
                "wide_anchor_danger_soft_max": wide_soft,
                "wide_partner_danger_for_soft_anchor_max": wide_partner_soft,
                "wide_anchor_danger_hard_max": wide_hard,
                "trio_anchor_danger_max": trio_max,
            }
        )
    return rows


def _choose_policy(train: pd.DataFrame, min_train_tickets: int, min_race_hit: float) -> tuple[dict | None, dict | None]:
    best_params = None
    best_metrics = None
    best_score = -np.inf
    for params in _grid():
        selected = _apply_gate(train, params)
        metrics = _metrics(selected, "train_danger_gate")
        if metrics["tickets"] < min_train_tickets or metrics["race_hit_rate"] < min_race_hit:
            continue
        score = (
            (metrics["roi"] - 1.0) * 100.0
            + metrics["race_hit_rate"] * 9.0
            + np.log1p(metrics["tickets"]) * 0.20
            - max(0.0, abs(metrics["max_drawdown_yen"]) / 10000.0) * 0.12
        )
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics
    return best_params, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply ticket-type-specific dangerous-popular gates.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/late_value_survival_gate_v1/gated_ticket_profiles.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/ticket_type_danger_gate_v1")
    parser.add_argument("--min-train-tickets", type=int, default=120)
    parser.add_argument("--min-race-hit", type=float, default=0.10)
    args = parser.parse_args()

    tickets = _prepare(pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False))
    years = sorted(int(y) for y in tickets["year"].dropna().unique())
    wf_rows: list[dict] = []
    selected_frames: list[pd.DataFrame] = []
    for year in years[1:]:
        train = tickets[tickets["year"].lt(year)].copy()
        test = tickets[tickets["year"].eq(year)].copy()
        params, train_metrics = _choose_policy(train, args.min_train_tickets, args.min_race_hit)
        if params is None:
            wf_rows.append({"year": year, "selected": False})
            continue
        selected = _apply_gate(test, params)
        test_metrics = _metrics(selected, f"test_{year}_danger_gate")
        selected_frames.append(selected.assign(test_year=year))
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
    out_dir = ensure_dir(project_path(args.output_dir))
    selected.to_csv(out_dir / "danger_gated_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(wf_rows).to_csv(out_dir / "walkforward_danger_gate_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input": {
            "tickets_csv": args.tickets_csv,
            "test_years": test_years,
            "note": "Ticket-type-specific dangerous-popular gate. Win and umaren are stricter; wide allows risky anchors when partner risk is low.",
        },
        "ungated": _metrics(ungated, "ungated"),
        "danger_gated": _metrics(selected, "danger_gated"),
    }
    summary["delta_roi"] = summary["danger_gated"]["roi"] - summary["ungated"]["roi"]
    summary["delta_profit_yen"] = summary["danger_gated"]["profit_yen"] - summary["ungated"]["profit_yen"]
    pd.DataFrame([summary["ungated"], summary["danger_gated"]]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
