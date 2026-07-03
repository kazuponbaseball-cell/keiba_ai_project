from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from validate_final_kernel_race_level import (
    OUT as KERNEL_OUT,
    add_kernel_features,
    load_universe,
    policy_grid,
    prefilter,
    race_representatives,
    select_pairs,
)


FINAL = Path("outputs/analysis/final_operational_quality_v1/standard_explained_tickets.csv")
OUT = Path("outputs/analysis/final_kernel_missed_hit_diagnostics_v1")


MATCH_COLS = [
    "coverage",
    "venue_policy",
    "going_policy",
    "axis_min",
    "partner_min",
    "partner_odds_min",
    "partner_odds_max",
    "front_min",
    "anchor_danger_max",
    "partner_danger_max",
    "partner_quinella_min",
    "umaren_pair_score_min",
    "umaren_partner_odds_max",
    "umaren_quote_min",
    "stake_profile",
]


def _n(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_selected_params(test_year: int) -> tuple[dict, float]:
    summary = pd.read_csv(KERNEL_OUT / "walkforward_summary.csv")
    row = summary[summary["test_year"].eq(test_year)].iloc[0]
    grids = policy_grid()
    for params in grids:
        ok = True
        for col in MATCH_COLS:
            left = params[col]
            right = row[col]
            if isinstance(left, str):
                ok &= left == str(right)
            else:
                ok &= abs(float(left) - float(right)) < 1e-9
        if ok:
            return params, float(row["score_threshold"])
    raise RuntimeError(f"No matching params for {test_year}")


def final_hit_rows() -> pd.DataFrame:
    final = pd.read_csv(FINAL, dtype={"race_id": str}, low_memory=False)
    final = final[pd.to_numeric(final["runtime_stake_yen"], errors="coerce").fillna(0.0).gt(0)].copy()
    final["runtime_return_yen"] = pd.to_numeric(final["runtime_return_yen"], errors="coerce").fillna(0.0)
    final = final[final["runtime_return_yen"].gt(0)].copy()
    final["year"] = final["race_id"].str[:4].astype(int)
    return final


def failure_reasons(row: pd.Series, params: dict, threshold: float) -> list[str]:
    reasons: list[str] = []
    checks = [
        ("venue_allowed", row.get("venue") in params["venue_allowed"]),
        ("going_allowed", row.get("going") in params["going_allowed"]),
        ("axis_min", _n(row.get("wide_axis_score")) >= params["axis_min"]),
        ("partner_min", _n(row.get("wide_partner_score")) >= params["partner_min"]),
        ("partner_odds_min", _n(row.get("partner_odds")) >= params["partner_odds_min"]),
        ("partner_odds_max", _n(row.get("partner_odds")) <= params["partner_odds_max"]),
        ("front_min", _n(row.get("projected_front5_prob")) >= params["front_min"]),
        ("anchor_danger_max", _n(row.get("anchor_danger")) <= params["anchor_danger_max"]),
        ("partner_danger_max", _n(row.get("partner_danger")) <= params["partner_danger_max"]),
        ("score_threshold", _n(row.get("final_kernel_score")) >= threshold),
    ]
    for name, ok in checks:
        if not ok:
            reasons.append(name)

    if str(row.get("ticket_type")) == "umaren":
        umaren_checks = [
            ("umaren_pair_score_min", _n(row.get("pair_score")) >= params["umaren_pair_score_min"]),
            ("partner_quinella_min", _n(row.get("partner_quinella_score")) >= params["partner_quinella_min"]),
            ("umaren_partner_odds_max", _n(row.get("partner_odds")) <= params["umaren_partner_odds_max"]),
            ("umaren_quote_min", _n(row.get("umaren_quote_proxy")) >= params["umaren_quote_min"]),
        ]
        for name, ok in umaren_checks:
            if not ok:
                reasons.append(name)
    return reasons


def factor_notes(row: pd.Series, final_row: pd.Series) -> str:
    notes: list[str] = []
    if _n(row.get("market_overlay_score")) >= 0.75:
        notes.append("市場残差/妙味が強い")
    if _n(row.get("late_value_survives_score")) >= 0.75:
        notes.append("直前妙味が残りやすい")
    if _n(row.get("projected_front5_prob")) >= 0.65 and _n(row.get("partner_odds")) >= 7:
        notes.append("前に行ける人気薄候補")
    if _n(row.get("pair_quinella_score")) >= 0.70:
        notes.append("同時好走スコア高")
    if _n(row.get("partner_danger")) <= 0.18:
        notes.append("相手危険度は低い")
    if _n(final_row.get("ticket_danger_popular_score")) <= 0.30:
        notes.append("危険人気馬リスク低")
    if _n(final_row.get("race_difficulty_score")) <= 0.50:
        notes.append("レース難易度低め")
    if _n(final_row.get("b_priority_net_score")) >= 0.57:
        notes.append("B文脈良好")
    if _n(final_row.get("a_priority_net_score")) >= 0.30 and str(final_row.get("ticket_type")) == "umaren":
        notes.append("馬連A文脈良好")
    return " / ".join(notes)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    universe = add_kernel_features(load_universe())
    final_hits = final_hit_rows()
    selected = pd.read_csv(KERNEL_OUT / "walkforward_selected_tickets.csv", dtype={"race_id": str}, low_memory=False)

    diag_rows: list[dict] = []
    for _, frow in final_hits.iterrows():
        year = int(frow["year"])
        if year not in {2025, 2026}:
            continue
        params, threshold = load_selected_params(year)
        race_id = str(frow["race_id"])
        anchor_no = int(float(frow["anchor_no"]))
        partner_no = int(float(frow["partner_no"])) if not pd.isna(frow.get("partner_no")) else -1
        cand = universe[
            universe["race_id"].eq(race_id)
            & universe["anchor_no"].astype(float).eq(float(anchor_no))
            & universe["partner_no"].astype(float).eq(float(partner_no))
        ].copy()
        race_reps = race_representatives(universe[universe["race_id"].eq(race_id)], params)
        selected_pairs = select_pairs(universe[universe["race_id"].eq(race_id)], params, threshold)
        selected_same_race = selected[selected["race_id"].eq(race_id)].copy()

        if cand.empty:
            diag_rows.append(
                {
                    "race_id": race_id,
                    "year": year,
                    "ticket_type": frow.get("ticket_type"),
                    "anchor_name": frow.get("anchor_name"),
                    "partner_name": frow.get("partner_name"),
                    "candidate_found": False,
                    "failure_reasons": "candidate_not_found",
                }
            )
            continue
        row = cand.iloc[0]
        combined = row.copy()
        combined["ticket_type"] = frow.get("ticket_type")
        reasons = failure_reasons(combined, params, threshold)
        pre = prefilter(cand, params)
        above = _n(row.get("final_kernel_score")) >= threshold
        selected_pair_key = ""
        if not selected_pairs.empty:
            srow = selected_pairs.iloc[0]
            selected_pair_key = f"{int(srow['anchor_no'])}-{int(srow['partner_no'])}:{srow['anchor_name']}-{srow['partner_name']}"
        rep_pair_key = ""
        if not race_reps.empty:
            rrow = race_reps.iloc[0]
            rep_pair_key = f"{int(rrow['anchor_no'])}-{int(rrow['partner_no'])}:{rrow['anchor_name']}-{rrow['partner_name']}"

        diag_rows.append(
            {
                "race_id": race_id,
                "year": year,
                "ticket_type": frow.get("ticket_type"),
                "anchor_no": anchor_no,
                "anchor_name": frow.get("anchor_name"),
                "partner_no": partner_no,
                "partner_name": frow.get("partner_name"),
                "candidate_found": True,
                "prefilter_pass": not pre.empty,
                "above_score_threshold": above,
                "score_threshold": threshold,
                "final_kernel_score": _n(row.get("final_kernel_score")),
                "race_representative_pair": rep_pair_key,
                "selected_pair_under_kernel": selected_pair_key,
                "kernel_selected_same_race_tickets": int(len(selected_same_race)),
                "failure_reasons": " / ".join(reasons) if reasons else "passed_pair_filters",
                "wide_axis_score": _n(row.get("wide_axis_score")),
                "wide_partner_score": _n(row.get("wide_partner_score")),
                "market_overlay_score": _n(row.get("market_overlay_score")),
                "late_value_survives_score": _n(row.get("late_value_survives_score")),
                "projected_front5_prob": _n(row.get("projected_front5_prob")),
                "pair_score": _n(row.get("pair_score")),
                "pair_quinella_score": _n(row.get("pair_quinella_score")),
                "partner_quinella_score": _n(row.get("partner_quinella_score")),
                "anchor_danger": _n(row.get("anchor_danger")),
                "partner_danger": _n(row.get("partner_danger")),
                "partner_odds": _n(row.get("partner_odds")),
                "wide_quote_proxy": _n(row.get("wide_quote_proxy")),
                "umaren_quote_proxy": _n(row.get("umaren_quote_proxy")),
                "runtime_expected_roi": _n(frow.get("runtime_expected_roi")),
                "runtime_backtest_pay_per100": _n(frow.get("runtime_backtest_pay_per100")),
                "runtime_return_yen": _n(frow.get("runtime_return_yen")),
                "ticket_front_position_reliability_score": _n(frow.get("ticket_front_position_reliability_score")),
                "ticket_danger_popular_score": _n(frow.get("ticket_danger_popular_score")),
                "race_difficulty_score": _n(frow.get("race_difficulty_score")),
                "a_priority_net_score": _n(frow.get("a_priority_net_score")),
                "b_priority_net_score": _n(frow.get("b_priority_net_score")),
                "priority_context_net_score": _n(frow.get("priority_context_net_score")),
                "live_alert_risk_score": _n(frow.get("live_alert_risk_score")),
                "buy_reason_summary": frow.get("buy_reason_summary"),
                "risk_reason_summary": frow.get("risk_reason_summary"),
                "ai_factor_notes": factor_notes(row, frow),
            }
        )

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT / "missed_and_hit_diagnostics.csv", index=False, encoding="utf-8-sig")

    selected_2026 = selected[selected["race_id"].str[:4].eq("2026")].copy()
    selected_2026.to_csv(OUT / "kernel_selected_2026_tickets.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Final Kernel Missed-Hit Diagnostics",
        "",
        "## AI Factor Checklist",
        "",
        "- Race selection / skip model: check whether the final hit race was below the race-level score threshold.",
        "- Front-running underdog: `projected_front5_prob` plus partner odds.",
        "- Market residual / overlay: `market_overlay_score` and `runtime_expected_roi`.",
        "- Late odds value: `late_value_survives_score` and live-safety fields from final tickets.",
        "- Dangerous favorite / false popular: `ticket_danger_popular_score`, `anchor_danger`, `partner_danger`.",
        "- Race difficulty: `race_difficulty_score`.",
        "- Pair joint probability: `pair_score`, `pair_quinella_score`, `partner_quinella_score`.",
        "- Ticket suitability: A/B/context net scores and ticket type.",
        "",
        "## Missed 2026 Final Hits",
        "",
    ]
    if not diag.empty:
        target = diag[diag["year"].eq(2026)].copy()
        for _, row in target.iterrows():
            report.extend(
                [
                    f"### {row['race_id']} {row['ticket_type']} {row['anchor_name']} - {row['partner_name']}",
                    "",
                    f"- Failure: {row['failure_reasons']}",
                    f"- Kernel score: {row['final_kernel_score']:.3f} / threshold {row['score_threshold']:.3f}",
                    f"- Race representative: {row['race_representative_pair']}",
                    f"- Kernel selected pair: {row['selected_pair_under_kernel'] or 'none'}",
                    f"- AI factor notes: {row['ai_factor_notes']}",
                    f"- Market overlay: {row['market_overlay_score']:.3f}, late value: {row['late_value_survives_score']:.3f}, front5: {row['projected_front5_prob']:.3f}",
                    f"- Pair score: {row['pair_score']:.3f}, pair quinella: {row['pair_quinella_score']:.3f}, partner quinella: {row['partner_quinella_score']:.3f}",
                    f"- Danger popular: {row['ticket_danger_popular_score']:.3f}, race difficulty: {row['race_difficulty_score']:.3f}, B net: {row['b_priority_net_score']:.3f}",
                    "",
                ]
            )
    (OUT / "diagnostic_report.md").write_text("\n".join(report), encoding="utf-8")

    print(diag[diag["year"].eq(2026)].to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
