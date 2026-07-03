from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_shape_adjusted_pair_selection import (  # noqa: E402
    DEFAULT_RACE_SHAPE,
    DEFAULT_UNIVERSE,
    OUT_DIR as BASE_OUT_DIR,
    add_shape_scores,
    bool_col,
    gate_mask,
    hit_flag,
    load_universe,
    metrics,
    ncol,
    select_top_per_race,
    ticket_return,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COURSE_DIAG = ROOT / "outputs" / "analysis" / "course_adjusted_front3f_signal_v1" / "course_adjusted_front3f_race_diagnostics.csv"
OUT_DIR = ROOT / "outputs" / "analysis" / "course_adjusted_shape_pair_selection_v1"


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def sigmoid(values: pd.Series | np.ndarray) -> pd.Series:
    arr = np.asarray(values, dtype=float)
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(arr, -30, 30))))


def load_course_diag(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    diag = read_csv(path)
    race_col = "race_id" if "race_id" in diag.columns else "レースID(新/馬番無)"
    if race_col not in diag.columns:
        return pd.DataFrame()
    diag = diag.rename(columns={race_col: "race_id"}).copy()
    diag["race_id"] = diag["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    keep = [
        "race_id",
        "course_front3f_prior_sec",
        "course_front3f_prior_std",
        "course_front3f_prior_count",
        "race_course_adj_ten_pressure_score",
        "race_course_adj_fast_start_count",
        "race_course_adj_ten_speed_gap_top2",
        "race_course_adj_queue_clarity_score",
    ]
    return diag[[c for c in keep if c in diag.columns]].drop_duplicates("race_id")


def add_course_adjusted_shape_scores(df: pd.DataFrame, course_diag: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not course_diag.empty:
        out = out.merge(course_diag, on="race_id", how="left")

    pressure = ncol(out, "race_course_adj_ten_pressure_score", ncol(out, "queue_front_load_score", 0.0)).fillna(0.0)
    fast_count = ncol(out, "race_course_adj_fast_start_count", ncol(out, "queue_candidate_count", 0.0)).fillna(0.0)
    gap = ncol(out, "race_course_adj_ten_speed_gap_top2", ncol(out, "queue_top_gap", 0.0)).fillna(0.0)
    clarity = ncol(out, "race_course_adj_queue_clarity_score", ncol(out, "queue_clarity_score", 0.5)).fillna(0.5).clip(0.0, 1.0)
    course_count = ncol(out, "course_front3f_prior_count", 0.0).fillna(0.0)
    course_sample_weight = (course_count / 30.0).clip(0.0, 1.0)

    pressure_q = pressure.groupby(out["race_id"]).transform("first")
    fast_q = fast_count.groupby(out["race_id"]).transform("first")
    gap_q = gap.groupby(out["race_id"]).transform("first")
    clarity_q = clarity.groupby(out["race_id"]).transform("first")

    course_duel = sigmoid(
        0.95 * pressure_q
        + 0.22 * fast_q
        - 2.00 * clarity_q
        - 0.55 * gap_q
        - 0.10
    ).set_axis(out.index).clip(0.0, 1.0)
    course_solo = sigmoid(
        2.15 * clarity_q
        + 0.92 * gap_q
        - 0.24 * fast_q
        - 0.20 * pressure_q
        - 0.45
    ).set_axis(out.index).clip(0.0, 1.0)
    course_no_clear = sigmoid(
        0.55
        - 0.65 * pressure_q
        - 1.30 * clarity_q
        - 0.25 * fast_q
    ).set_axis(out.index).clip(0.0, 1.0)
    mixed_weight = (1.0 - np.maximum.reduce([course_duel.to_numpy(), course_solo.to_numpy(), course_no_clear.to_numpy()])).clip(0.0, 1.0)
    mixed_weight = pd.Series(mixed_weight, index=out.index)

    base = ncol(out, "shape_base_score", ncol(out, "pair_quinella_score", 0.5)).clip(0.0, 1.0)
    base_rank = ncol(out, "shape_base_rank_score", base.groupby(out["race_id"]).rank(pct=True).fillna(0.5)).clip(0.0, 1.0)
    overlay = ncol(out, "market_overlay_score", 0.0).clip(0.0, 1.0)
    late = ncol(out, "late_value_survives_score", 0.0).clip(0.0, 1.0)
    front_max = ncol(out, "front_pair_max", ncol(out, "projected_front5_prob", 0.0)).clip(0.0, 1.0)
    front_min = ncol(out, "front_pair_min", 0.0).clip(0.0, 1.0)
    closer_max = ncol(out, "closer_pair_max", 0.0).clip(0.0, 1.0)
    diversity = ncol(out, "style_diversity", 0.0).clip(0.0, 1.0)
    clash = ncol(out, "front_front_clash", 0.0).clip(0.0, 1.0)
    front_slow = ncol(out, "front_front_slow_fit", 0.0).clip(0.0, 1.0)
    collapse = ncol(out, "collapse_fit", ncol(out, "race_pace_collapse", 0.0)).clip(0.0, 1.0)

    solo_fit = (0.48 * front_max + 0.24 * front_slow + 0.16 * front_min + 0.12 * base).clip(0.0, 1.0)
    duel_fit = (0.34 * closer_max + 0.26 * collapse + 0.23 * diversity + 0.10 * front_max + 0.07 * base).clip(0.0, 1.0)
    no_clear_fit = (0.42 * front_max + 0.24 * front_slow + 0.20 * base + 0.14 * overlay).clip(0.0, 1.0)
    mixed_fit = (0.32 * base + 0.24 * front_max + 0.24 * closer_max + 0.20 * diversity).clip(0.0, 1.0)

    course_fit_raw = (
        course_solo * solo_fit
        + course_duel * duel_fit
        + course_no_clear * no_clear_fit
        + mixed_weight * mixed_fit
    ) / (course_solo + course_duel + course_no_clear + mixed_weight).replace(0, np.nan)
    course_fit_raw = course_fit_raw.fillna(ncol(out, "shape_pair_fit_score", 0.5)).clip(0.0, 1.0)
    existing_fit = ncol(out, "shape_pair_fit_score", 0.5).clip(0.0, 1.0)
    course_fit = ((1.0 - 0.45 * course_sample_weight) * existing_fit + (0.45 * course_sample_weight) * course_fit_raw).clip(0.0, 1.0)

    front_burn = (course_duel * clash * (0.35 + 0.65 * front_min)).clip(0.0, 1.0)
    dead_slow_closer = (course_solo * closer_max * (1.0 - front_max)).clip(0.0, 1.0)
    no_clear_uncertainty = (course_no_clear * (1.0 - front_max) * 0.35).clip(0.0, 1.0)
    raw_risk = pd.Series(np.maximum.reduce([front_burn.to_numpy(), dead_slow_closer.to_numpy(), no_clear_uncertainty.to_numpy()]), index=out.index)
    existing_risk = ncol(out, "shape_pair_risk_score", 0.0).clip(0.0, 1.0)
    course_risk = ((1.0 - 0.45 * course_sample_weight) * existing_risk + (0.45 * course_sample_weight) * raw_risk).clip(0.0, 1.0)

    out["course_shape_duel_score"] = course_duel
    out["course_shape_solo_score"] = course_solo
    out["course_shape_no_clear_score"] = course_no_clear
    out["course_shape_sample_weight"] = course_sample_weight
    out["course_shape_pair_fit_score"] = course_fit
    out["course_shape_pair_risk_score"] = course_risk
    out["course_shape_value_score"] = (0.55 * overlay + 0.45 * late).clip(0.0, 1.0)
    out["course_shape_base_rank_score"] = base_rank
    out["course_shape_fit_delta"] = out["course_shape_pair_fit_score"] - existing_fit
    out["course_shape_risk_delta"] = out["course_shape_pair_risk_score"] - existing_risk
    return out


def select_course_top_per_race(frame: pd.DataFrame, score_col: str, ticket_type: str, policy: str, gate: str, weight: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    sort_cols = [c for c in ["race_id", score_col, "course_shape_base_rank_score", "market_overlay_score"] if c in frame.columns]
    selected = (
        frame.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1))
        .drop_duplicates("race_id", keep="first")
        .copy()
    )
    selected["policy"] = policy
    selected["gate"] = gate
    selected["shape_weight"] = weight
    selected["ticket_type"] = ticket_type
    selected["stake_yen"] = 100.0
    selected["return_yen"] = ticket_return(selected, ticket_type)
    selected["hit"] = hit_flag(selected, ticket_type)
    selected["ticket_key"] = selected.apply(
        lambda r: f"{ticket_type}:{r['race_id']}:{int(r['horse_a'])}-{int(r['horse_b'])}", axis=1
    )
    return selected


def evaluate_course(df: pd.DataFrame, gates: list[str], weights: list[float], ticket_types: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []

    for gate in gates:
        gated = df[gate_mask(df, gate)].copy()
        if gated.empty:
            continue
        for ticket_type in ticket_types:
            existing_base = select_top_per_race(gated, "shape_base_rank_score", ticket_type, f"{ticket_type}_{gate}_base", gate, 0.0)
            existing_shape = None
            for weight in weights:
                if weight == 0:
                    selected = existing_base.copy()
                    selected["policy"] = f"{ticket_type}_{gate}_base"
                    selected["shape_weight"] = 0.0
                else:
                    score_col = f"course_shape_adjusted_score_w{weight:.2f}"
                    risk_weight = min(0.12, weight * 0.75)
                    gated[score_col] = (
                        (1.0 - weight) * gated["course_shape_base_rank_score"]
                        + weight * gated["course_shape_pair_fit_score"]
                        + 0.06 * gated["course_shape_value_score"]
                        - risk_weight * gated["course_shape_pair_risk_score"]
                    )
                    selected = select_course_top_per_race(
                        gated,
                        score_col,
                        ticket_type,
                        f"{ticket_type}_{gate}_course_shape_w{weight:.2f}",
                        gate,
                        weight,
                    )
                    existing_score_col = f"existing_shape_adjusted_score_w{weight:.2f}"
                    gated[existing_score_col] = (
                        (1.0 - weight) * gated["shape_base_rank_score"]
                        + weight * gated["shape_pair_fit_score"]
                        + 0.06 * ncol(gated, "shape_value_score", gated["course_shape_value_score"])
                        - risk_weight * gated["shape_pair_risk_score"]
                    )
                    existing_shape = select_top_per_race(
                        gated,
                        existing_score_col,
                        ticket_type,
                        f"{ticket_type}_{gate}_existing_shape_w{weight:.2f}",
                        gate,
                        weight,
                    )
                    existing_keys = existing_shape[["race_id", "pair_key_norm"]].rename(columns={"pair_key_norm": "existing_shape_pair_key_norm"})
                    selected = selected.merge(existing_keys, on="race_id", how="left")
                    selected["changed_from_existing_shape"] = selected["pair_key_norm"].ne(selected["existing_shape_pair_key_norm"])

                base_keys = existing_base[["race_id", "pair_key_norm"]].rename(columns={"pair_key_norm": "base_pair_key_norm"})
                selected = selected.merge(base_keys, on="race_id", how="left")
                selected["changed_from_base"] = selected["pair_key_norm"].ne(selected["base_pair_key_norm"])
                if "changed_from_existing_shape" not in selected.columns:
                    selected["changed_from_existing_shape"] = False
                selections.append(selected)
                m = metrics(selected, selected["policy"].iloc[0] if not selected.empty else f"{ticket_type}_{gate}_{weight}")
                m["gate"] = gate
                m["ticket_type"] = ticket_type
                m["shape_weight"] = weight
                changed_existing = selected[selected["changed_from_existing_shape"]].copy()
                cm_existing = metrics(changed_existing, "course_changed_existing_only")
                changed_base = selected[selected["changed_from_base"]].copy()
                cm_base = metrics(changed_base, "course_changed_base_only")
                m["changed_from_existing_tickets"] = cm_existing["tickets"]
                m["changed_from_existing_roi_pct"] = cm_existing["roi_pct"]
                m["changed_from_existing_hit_rate_pct"] = cm_existing["hit_rate_pct"]
                m["changed_from_base_tickets"] = cm_base["tickets"]
                m["changed_from_base_roi_pct"] = cm_base["roi_pct"]
                m["changed_from_base_hit_rate_pct"] = cm_base["hit_rate_pct"]
                summary_rows.append(m)
                for year, gy in selected.groupby("year"):
                    ym = metrics(gy, selected["policy"].iloc[0])
                    ym["year"] = int(year)
                    ym["gate"] = gate
                    ym["ticket_type"] = ticket_type
                    ym["shape_weight"] = weight
                    yearly_rows.append(ym)

    detail = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    return summary, yearly, detail


def add_comparison(summary: pd.DataFrame, existing_summary_path: Path) -> pd.DataFrame:
    if summary.empty or not existing_summary_path.exists():
        return summary
    existing = read_csv(existing_summary_path)
    existing = existing[existing["policy"].astype(str).str.contains("_shape_w|_base", regex=True)].copy()
    existing["policy_key"] = (
        existing["ticket_type"].astype(str)
        + "|"
        + existing["gate"].astype(str)
        + "|"
        + existing["shape_weight"].astype(str)
    )
    out = summary.copy()
    out["policy_key"] = out["ticket_type"].astype(str) + "|" + out["gate"].astype(str) + "|" + out["shape_weight"].astype(str)
    keep = existing[["policy_key", "roi_pct", "hit_rate_pct", "tickets"]].rename(
        columns={
            "roi_pct": "existing_shape_roi_pct",
            "hit_rate_pct": "existing_shape_hit_rate_pct",
            "tickets": "existing_shape_tickets",
        }
    )
    out = out.merge(keep, on="policy_key", how="left")
    out["roi_delta_vs_existing_shape_pct"] = out["roi_pct"] - out["existing_shape_roi_pct"]
    out["hit_delta_vs_existing_shape_pct"] = out["hit_rate_pct"] - out["existing_shape_hit_rate_pct"]
    return out.drop(columns=["policy_key"], errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest course-adjusted front3F race-shape refinement on pair pickup ROI.")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--race-shape", default=str(DEFAULT_RACE_SHAPE))
    parser.add_argument("--course-diag", default=str(DEFAULT_COURSE_DIAG))
    parser.add_argument("--existing-summary", default=str(BASE_OUT_DIR.parent / "shape_adjusted_pair_selection_v1_strong_weights" / "shape_adjusted_pair_summary.csv"))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--weights", default="0,0.55,0.70,0.85,1.00")
    parser.add_argument("--gates", default="all,value_loose,value_mid,price_sane_strong")
    parser.add_argument("--ticket-types", default="umaren,wide")
    args = parser.parse_args()

    universe_path = Path(args.universe)
    universe_path = universe_path if universe_path.is_absolute() else ROOT / universe_path
    race_shape_path = Path(args.race_shape)
    race_shape_path = race_shape_path if race_shape_path.is_absolute() else ROOT / race_shape_path
    course_diag_path = Path(args.course_diag)
    course_diag_path = course_diag_path if course_diag_path.is_absolute() else ROOT / course_diag_path
    existing_summary_path = Path(args.existing_summary)
    existing_summary_path = existing_summary_path if existing_summary_path.is_absolute() else ROOT / existing_summary_path
    out_dir = Path(args.output_dir)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    gates = [x.strip() for x in args.gates.split(",") if x.strip()]
    ticket_types = [x.strip() for x in args.ticket_types.split(",") if x.strip()]

    df = load_universe(universe_path, race_shape_path)
    df = add_shape_scores(df)
    course_diag = load_course_diag(course_diag_path)
    df = add_course_adjusted_shape_scores(df, course_diag)
    summary, yearly, detail = evaluate_course(df, gates, weights, ticket_types)
    summary = add_comparison(summary, existing_summary_path)
    if not summary.empty:
        summary = summary.sort_values(["roi_pct", "tickets"], ascending=[False, False])
    if not yearly.empty:
        yearly = yearly.sort_values(["policy", "year"])

    score_cols = [
        "race_id",
        "year",
        "pair_key_norm",
        "anchor_no",
        "partner_no",
        "queue_shape_label",
        "race_course_adj_ten_pressure_score",
        "race_course_adj_fast_start_count",
        "race_course_adj_ten_speed_gap_top2",
        "race_course_adj_queue_clarity_score",
        "course_shape_duel_score",
        "course_shape_solo_score",
        "course_shape_no_clear_score",
        "course_shape_pair_fit_score",
        "course_shape_pair_risk_score",
        "course_shape_fit_delta",
        "course_shape_risk_delta",
    ]
    df[[c for c in score_cols if c in df.columns]].to_csv(out_dir / "course_shape_pair_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "course_shape_pair_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "course_shape_pair_yearly.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(out_dir / "course_shape_pair_detail.csv", index=False, encoding="utf-8-sig")

    top = summary.head(30).replace({np.nan: None}).to_dict(orient="records") if not summary.empty else []
    report = {
        "universe": str(universe_path.relative_to(ROOT)),
        "course_diag": str(course_diag_path.relative_to(ROOT)),
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "course_diag_races": int(course_diag["race_id"].nunique()) if not course_diag.empty else 0,
        "top_policies": top,
        "note": "Course-adjusted front3F is tested as a race-shape refinement. This is still a shadow backtest; production BUY should require yearly stability and live snapshot validation.",
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
