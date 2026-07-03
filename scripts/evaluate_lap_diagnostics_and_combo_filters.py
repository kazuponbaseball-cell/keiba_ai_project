from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/analysis/lap_diagnostics_combo_filters_v1"
DEFAULT_RUNNERS = ROOT / "outputs/analysis/lap_pair_refinement_candidates_v1/runner_lap_pair_refinement_features.csv"
DEFAULT_LAP_TICKETS = ROOT / "outputs/analysis/lap_pair_refinement_candidates_v1/s_priority_tickets_with_lap_pair_refinement.csv"
DEFAULT_TIME_TICKETS = ROOT / "outputs/analysis/time_value_refinement_candidates_v1/s_priority_tickets_with_time_refinement.csv"
DEFAULT_ABILITY_TICKETS = ROOT / "outputs/analysis/basic_ability_overlay_strongest_v1_verify/s_priority_tickets_with_basic_ability_overlay.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def ticket_key(frame: pd.DataFrame) -> pd.Series:
    race_id = frame["race_id"].astype(str)
    a = num(frame.get("anchor_no"), frame.index).fillna(-1).astype(int).astype(str)
    b = num(frame.get("partner_no"), frame.index).fillna(-1).astype(int).astype(str)
    ticket_type = frame.get("ticket_type", pd.Series("", index=frame.index)).fillna("").astype(str)
    return race_id + ":" + a + "-" + b + ":" + ticket_type


def metrics(frame: pd.DataFrame, policy: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": policy,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": np.nan,
            "hit_rate": np.nan,
            "top_return_share": np.nan,
            "roi_ex_top1": np.nan,
        }
    stake = num(frame.get("runtime_stake_yen"), frame.index).fillna(num(frame.get("stake_yen"), frame.index)).fillna(0.0)
    ret = num(frame.get("runtime_return_yen"), frame.index).fillna(num(frame.get("return_yen"), frame.index)).fillna(0.0)
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    profit = ret_sum - stake_sum
    if ret_sum > 0 and len(frame) > 1:
        top_i = int(ret.to_numpy().argmax())
        roi_ex_top1 = safe_div(ret_sum - float(ret.iloc[top_i]), stake_sum - float(stake.iloc[top_i]))
        top_share = float(ret.max() / ret_sum)
    else:
        roi_ex_top1 = np.nan
        top_share = np.nan
    return {
        "policy": policy,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame else 0,
        "stake_yen": stake_sum,
        "return_yen": ret_sum,
        "profit_yen": profit,
        "roi": safe_div(ret_sum, stake_sum),
        "hit_rate": float(ret.gt(0).mean()) if len(frame) else np.nan,
        "top_return_share": top_share,
        "roi_ex_top1": roi_ex_top1,
    }


def mode_or_first(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    values = values[values.ne("")]
    if values.empty:
        return "unknown"
    mode = values.mode()
    return str(mode.iloc[0]) if not mode.empty else str(values.iloc[0])


def build_race_diagnostics(runners: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    use = runners[runners["source"].astype(str).eq("test")].copy()
    need_cols = [c for c in use.columns if c.startswith("race_need_")]
    rows: list[dict[str, Any]] = []
    for race_id, group in use.groupby("race_id", sort=False):
        need_mean = group[need_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0) if need_cols else pd.Series(dtype=float)
        predicted = str(need_mean.idxmax()).replace("race_need_", "") if not need_mean.empty else mode_or_first(group["predicted_lap_mode"])
        actual = mode_or_first(group["actual_lap_mode_diagnostic"])
        confidence = float(num(group.get("race_lap_prediction_confidence"), group.index).mean())
        concentration = float(num(group.get("race_lap_profile_concentration"), group.index).mean())
        rows.append(
            {
                "race_id": race_id,
                "predicted_lap_mode_race": predicted,
                "actual_lap_mode": actual,
                "lap_prediction_hit": predicted == actual,
                "race_lap_prediction_confidence": confidence,
                "race_lap_profile_concentration": concentration,
                **{f"avg_{c}": float(need_mean.get(c, np.nan)) for c in need_cols},
            }
        )
    race_diag = pd.DataFrame(rows)
    confusion = (
        race_diag.pivot_table(
            index="actual_lap_mode",
            columns="predicted_lap_mode_race",
            values="race_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    race_diag["confidence_bin"] = pd.qcut(
        race_diag["race_lap_prediction_confidence"].rank(method="first"), q=5, labels=["q1_low", "q2", "q3", "q4", "q5_high"]
    )
    by_conf = (
        race_diag.groupby("confidence_bin", observed=False)
        .agg(
            races=("race_id", "count"),
            hit_rate=("lap_prediction_hit", "mean"),
            avg_confidence=("race_lap_prediction_confidence", "mean"),
        )
        .reset_index()
    )
    return race_diag, confusion, by_conf


def load_combined_tickets(lap_path: Path, time_path: Path, ability_path: Path, race_diag: pd.DataFrame) -> pd.DataFrame:
    lap = read_csv(lap_path)
    time = read_csv(time_path)
    ability = read_csv(ability_path)
    for frame in [lap, time, ability]:
        frame["_ticket_key"] = ticket_key(frame)
    time_cols = [
        "_ticket_key",
        "pair_min_time_value_relative_rank_score",
        "pair_avg_time_refinement_composite",
        "pair_min_time_refinement_composite",
        "pair_avg_time_value_relative_rank_score",
    ]
    ability_cols = [
        "_ticket_key",
        "pair_min_ability_floor_score_5",
        "pair_avg_ability_floor_score_5",
        "pair_min_ability_stability_score_3",
    ]
    out = lap.merge(time[[c for c in time_cols if c in time.columns]], on="_ticket_key", how="left")
    out = out.merge(ability[[c for c in ability_cols if c in ability.columns]], on="_ticket_key", how="left")
    out = out.merge(
        race_diag[["race_id", "predicted_lap_mode_race", "actual_lap_mode", "lap_prediction_hit"]],
        on="race_id",
        how="left",
    )
    ret = num(out.get("runtime_return_yen"), out.index).fillna(num(out.get("return_yen"), out.index)).fillna(0.0)
    out["ticket_hit"] = ret.gt(0)
    out["miss_decomposition"] = np.select(
        [
            out["ticket_hit"],
            out["lap_prediction_hit"].fillna(False),
            out["actual_lap_mode"].fillna("unknown").eq("unknown"),
        ],
        [
            "hit",
            "race_read_ok_horse_or_pair_wrong",
            "race_read_unknown",
        ],
        default="race_read_wrong_or_noisy",
    )
    return out


def evaluate_policies(tickets: pd.DataFrame) -> pd.DataFrame:
    floor = num(tickets.get("pair_min_ability_floor_score_5"), tickets.index)
    time_rel = num(tickets.get("pair_min_time_value_relative_rank_score"), tickets.index)
    time_top = num(tickets.get("pair_avg_time_refinement_composite"), tickets.index)
    lap_fit = num(tickets.get("pair_lap_same_race_fit_score"), tickets.index)
    lap_conf = num(tickets.get("pair_lap_race_confidence"), tickets.index)
    lap_contra = num(tickets.get("pair_lap_contradiction_score"), tickets.index)
    lap_mismatch = num(tickets.get("pair_lap_mismatch_popular_max"), tickets.index)

    thresholds = {
        "floor_q20": 0.20,
        "time_relative_q20": float(time_rel.quantile(0.20)),
        "time_top20": float(time_top.quantile(0.80)),
        "lap_fit_q30": float(lap_fit.quantile(0.30)),
        "lap_conf_q40": float(lap_conf.quantile(0.40)),
        "lap_conf_q50": float(lap_conf.quantile(0.50)),
        "lap_contra_q80": float(lap_contra.quantile(0.80)),
        "lap_mismatch_q80": float(lap_mismatch.quantile(0.80)),
    }
    masks = {
        "base_all": pd.Series(True, index=tickets.index),
        "ability_floor_q20": floor.ge(thresholds["floor_q20"]),
        "time_relative_q20": time_rel.ge(thresholds["time_relative_q20"]),
        "lap_fit_q30": lap_fit.ge(thresholds["lap_fit_q30"]),
        "lap_combo_fit_conf_no_contra": lap_fit.ge(thresholds["lap_fit_q30"])
        & lap_conf.ge(thresholds["lap_conf_q40"])
        & lap_contra.le(thresholds["lap_contra_q80"]),
        "ability_floor_and_lap_fit_q30": floor.ge(thresholds["floor_q20"]) & lap_fit.ge(thresholds["lap_fit_q30"]),
        "ability_floor_and_lap_combo": floor.ge(thresholds["floor_q20"])
        & lap_fit.ge(thresholds["lap_fit_q30"])
        & lap_conf.ge(thresholds["lap_conf_q40"])
        & lap_contra.le(thresholds["lap_contra_q80"]),
        "ability_time_relative": floor.ge(thresholds["floor_q20"]) & time_rel.ge(thresholds["time_relative_q20"]),
        "ability_time_relative_lap_fit": floor.ge(thresholds["floor_q20"])
        & time_rel.ge(thresholds["time_relative_q20"])
        & lap_fit.ge(thresholds["lap_fit_q30"]),
        "ability_time_relative_lap_combo": floor.ge(thresholds["floor_q20"])
        & time_rel.ge(thresholds["time_relative_q20"])
        & lap_fit.ge(thresholds["lap_fit_q30"])
        & lap_conf.ge(thresholds["lap_conf_q40"])
        & lap_contra.le(thresholds["lap_contra_q80"]),
        "ability_time_top20_lap_combo": floor.ge(thresholds["floor_q20"])
        & time_top.ge(thresholds["time_top20"])
        & lap_fit.ge(thresholds["lap_fit_q30"])
        & lap_conf.ge(thresholds["lap_conf_q40"])
        & lap_contra.le(thresholds["lap_contra_q80"]),
        "ability_lap_skip_popular_mismatch": floor.ge(thresholds["floor_q20"])
        & lap_mismatch.le(thresholds["lap_mismatch_q80"]),
    }
    rows = []
    by_year_rows = []
    for name, mask in masks.items():
        sub = tickets.loc[mask.fillna(False)].copy()
        rows.append(metrics(sub, name))
        for year, group in sub.groupby("year"):
            row = metrics(group, name)
            row["year"] = str(year)
            by_year_rows.append(row)
    out = pd.DataFrame(rows)
    out["year"] = "ALL"
    by_year = pd.DataFrame(by_year_rows)
    if not by_year.empty:
        by_year = by_year[["policy", "year", *[c for c in by_year.columns if c not in {"policy", "year"}]]]
    out.attrs["thresholds"] = thresholds
    return out, by_year


def evaluate_miss_decomposition(tickets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reason, group in tickets.groupby("miss_decomposition", dropna=False):
        rows.append(metrics(group, str(reason)))
    return pd.DataFrame(rows).sort_values("tickets", ascending=False)


def write_readme(
    out_dir: Path,
    race_diag: pd.DataFrame,
    confusion: pd.DataFrame,
    by_conf: pd.DataFrame,
    miss: pd.DataFrame,
    policies: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    def pct(x: float) -> str:
        return "" if pd.isna(x) else f"{x * 100:.1f}%"

    lines = [
        "# Lap Diagnostics And Combo Filters",
        "",
        "This report checks three pending items: race-quality prediction error, miss decomposition, and ability/time/lap combined filters.",
        "",
        "## Race Quality Prediction",
        f"- races: {len(race_diag):,}",
        f"- diagnostic hit rate: {pct(float(race_diag['lap_prediction_hit'].mean()))}",
        "",
        "### Confidence Bins",
        "| bin | races | hit | avg confidence |",
        "|---|---:|---:|---:|",
    ]
    for _, r in by_conf.iterrows():
        lines.append(f"| {r['confidence_bin']} | {int(r['races'])} | {pct(r['hit_rate'])} | {r['avg_confidence']:.3f} |")
    lines += [
        "",
        "## Miss Decomposition",
        "| group | tickets | races | ROI | hit | stake | return |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in miss.iterrows():
        lines.append(
            f"| {r['policy']} | {int(r['tickets'])} | {int(r['races'])} | {pct(r['roi'])} | "
            f"{pct(r['hit_rate'])} | {r['stake_yen']:.0f} | {r['return_yen']:.0f} |"
        )
    lines += [
        "",
        "## Combined Policies",
        "| policy | tickets | races | ROI | hit | ROI ex top1 | top return share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in policies.iterrows():
        lines.append(
            f"| {r['policy']} | {int(r['tickets'])} | {int(r['races'])} | {pct(r['roi'])} | "
            f"{pct(r['hit_rate'])} | {pct(r['roi_ex_top1'])} | {pct(r['top_return_share'])} |"
        )
    lines += [
        "",
        "## Thresholds",
        "```json",
        json.dumps(thresholds, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Interpretation",
        "- Race-quality prediction is still rough; use confidence and pair-fit as shadow gates before formal promotion.",
        "- If race read is correct but ticket misses, the issue is horse/pair selection rather than broad pace diagnosis.",
        "- Ability floor + time relative + lap fit is the most balanced combination to keep ticket count while improving ROI.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose lap prediction misses and combined ability/time/lap filters.")
    parser.add_argument("--runners-csv", default=str(DEFAULT_RUNNERS))
    parser.add_argument("--lap-tickets-csv", default=str(DEFAULT_LAP_TICKETS))
    parser.add_argument("--time-tickets-csv", default=str(DEFAULT_TIME_TICKETS))
    parser.add_argument("--ability-tickets-csv", default=str(DEFAULT_ABILITY_TICKETS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runners = read_csv(Path(args.runners_csv))
    race_diag, confusion, by_conf = build_race_diagnostics(runners)
    tickets = load_combined_tickets(Path(args.lap_tickets_csv), Path(args.time_tickets_csv), Path(args.ability_tickets_csv), race_diag)
    policies, policies_by_year = evaluate_policies(tickets)
    miss = evaluate_miss_decomposition(tickets)
    thresholds = policies.attrs.get("thresholds", {})

    race_diag.to_csv(out_dir / "race_quality_prediction_diagnostics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(out_dir / "race_quality_confusion_matrix.csv", index=False, encoding="utf-8-sig")
    by_conf.to_csv(out_dir / "race_quality_by_confidence_bin.csv", index=False, encoding="utf-8-sig")
    tickets.to_csv(out_dir / "combined_ticket_features.csv", index=False, encoding="utf-8-sig")
    policies.to_csv(out_dir / "combined_policy_metrics.csv", index=False, encoding="utf-8-sig")
    policies_by_year.to_csv(out_dir / "combined_policy_metrics_by_year.csv", index=False, encoding="utf-8-sig")
    miss.to_csv(out_dir / "miss_decomposition_metrics.csv", index=False, encoding="utf-8-sig")
    write_readme(out_dir, race_diag, confusion, by_conf, miss, policies, thresholds)

    summary = {
        "output_dir": str(out_dir),
        "race_quality": {
            "races": int(len(race_diag)),
            "hit_rate": float(race_diag["lap_prediction_hit"].mean()) if not race_diag.empty else None,
        },
        "thresholds": thresholds,
        "miss_decomposition": miss.to_dict(orient="records"),
        "combined_policy_metrics": policies.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
