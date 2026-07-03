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
    add_shape_scores,
    gate_mask,
    hit_flag,
    load_universe,
    ncol,
    ticket_return,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER_LAP = ROOT / "outputs/analysis/lap_pair_refinement_candidates_v1/runner_lap_pair_refinement_features.csv"
DEFAULT_RACE_QUALITY = ROOT / "outputs/analysis/race_quality_prediction_v2/race_quality_v2_diagnostics.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/lap_positive_expansion_v1"

LAP_RUNNER_COLS = [
    "lap_profile_fit_score",
    "lap_profile_fit_rank_in_race",
    "lap_fit_confident_score",
    "lap_axis_candidate_score",
    "lap_partner_specialist_score",
    "lap_mismatch_popular_risk",
    "race_lap_prediction_confidence",
    "race_lap_profile_concentration",
    "horse_lap_profile_sharpness",
    "horse_lap_profile_top_mode",
    "predicted_lap_mode",
]

RACE_QUALITY_COLS = [
    "race_id",
    "v2_predicted_lap_mode",
    "v2_confidence",
    "v2_margin",
    "v2_v1_agree",
    "v2_prob_fast",
    "v2_prob_slow",
    "v2_prob_instant",
    "v2_prob_sustain",
    "shape_fast_signal",
    "shape_slow_signal",
    "shape_sustain_signal",
    "shape_instant_signal",
]


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def clip01(series: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return pd.Series(series).fillna(0.0).clip(0.0, 1.0)


def norm_mode(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"long_spurt", "longspurt"}:
        return "sustain"
    if text in {"fast", "slow", "instant", "sustain"}:
        return text
    return "unknown"


def pair_key_from_row(row: pd.Series) -> str:
    a = int(row["horse_a"]) if pd.notna(row.get("horse_a")) else int(row["anchor_no"])
    b = int(row["horse_b"]) if pd.notna(row.get("horse_b")) else int(row["partner_no"])
    lo, hi = sorted([a, b])
    return f"{row['race_id']}:{lo}-{hi}"


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = pd.to_numeric(profit, errors="coerce").fillna(0.0).cumsum()
    drawdown = curve - curve.cummax()
    return float(drawdown.min())


def metrics(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
            "top5_removed_roi": 0.0,
            "top10_removed_roi": 0.0,
            "avg_lap_positive_score": np.nan,
            "avg_v2_confidence": np.nan,
            "avg_lap_mismatch_max": np.nan,
        }
    stake = pd.to_numeric(frame["stake_yen"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(frame["return_yen"], errors="coerce").fillna(0.0)
    profit = ret - stake
    ordered = frame.assign(_profit=profit).sort_values(["race_id", "ticket_type", "pair_key_norm"], kind="mergesort")
    hit = frame["hit"].astype(bool)
    race_hit = frame.groupby("race_id")["hit"].max().mean()
    top_removed = frame.sort_values("return_yen", ascending=False)
    def roi_without(n: int) -> float:
        kept = top_removed.iloc[n:]
        st = float(pd.to_numeric(kept["stake_yen"], errors="coerce").fillna(0.0).sum())
        rt = float(pd.to_numeric(kept["return_yen"], errors="coerce").fillna(0.0).sum())
        return rt / st if st > 0 else 0.0

    return {
        "policy": label,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(profit.sum()),
        "roi": float(ret.sum() / stake.sum()) if float(stake.sum()) > 0 else 0.0,
        "hit_rate": float(hit.mean()),
        "race_hit_rate": float(race_hit),
        "max_drawdown_yen": max_drawdown(ordered["_profit"]),
        "top5_removed_roi": roi_without(5),
        "top10_removed_roi": roi_without(10),
        "avg_lap_positive_score": float(ncol(frame, "lap_positive_score", np.nan).mean()),
        "avg_v2_confidence": float(ncol(frame, "v2_confidence", np.nan).mean()),
        "avg_lap_mismatch_max": float(ncol(frame, "pair_lap_mismatch_popular_max", np.nan).mean()),
    }


def load_runner_lap(path: Path) -> pd.DataFrame:
    usecols = ["race_id", "horse_no"] + LAP_RUNNER_COLS
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    usecols = [c for c in usecols if c in header.columns]
    df = read_csv(path, usecols=usecols)
    df["race_id"] = df["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    for col in LAP_RUNNER_COLS:
        if col in df.columns and col not in {"horse_lap_profile_top_mode", "predicted_lap_mode"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(["race_id", "horse_no"], keep="last")


def add_lap_features(universe: pd.DataFrame, runner_lap: pd.DataFrame, race_quality_path: Path) -> pd.DataFrame:
    out = universe.copy()
    out["race_id"] = out["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    for side, no_col in [("anchor", "anchor_no"), ("partner", "partner_no")]:
        side_df = runner_lap.rename(
            columns={
                "horse_no": no_col,
                **{c: f"{side}_{c}" for c in runner_lap.columns if c not in {"race_id", "horse_no"}},
            }
        )
        side_df[no_col] = pd.to_numeric(side_df[no_col], errors="coerce").astype("Int64")
        out[no_col] = pd.to_numeric(out[no_col], errors="coerce").astype("Int64")
        out = out.merge(side_df, on=["race_id", no_col], how="left")

    if race_quality_path.exists():
        header = pd.read_csv(race_quality_path, nrows=0, encoding="utf-8-sig")
        usecols = [c for c in RACE_QUALITY_COLS if c in header.columns]
        rq = read_csv(race_quality_path, usecols=usecols)
        rq["race_id"] = rq["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        rq = rq.drop_duplicates("race_id", keep="last")
        out = out.merge(rq, on="race_id", how="left")

    idx = out.index
    for side in ["anchor", "partner"]:
        for col in LAP_RUNNER_COLS:
            full = f"{side}_{col}"
            if full not in out.columns:
                out[full] = np.nan

    out["pair_lap_profile_fit_min"] = np.minimum(
        ncol(out, "anchor_lap_profile_fit_score", 0.0), ncol(out, "partner_lap_profile_fit_score", 0.0)
    )
    out["pair_lap_profile_fit_avg"] = (
        ncol(out, "anchor_lap_profile_fit_score", 0.0) + ncol(out, "partner_lap_profile_fit_score", 0.0)
    ) / 2.0
    out["pair_lap_confident_min"] = np.minimum(
        ncol(out, "anchor_lap_fit_confident_score", 0.0), ncol(out, "partner_lap_fit_confident_score", 0.0)
    )
    out["pair_lap_confident_avg"] = (
        ncol(out, "anchor_lap_fit_confident_score", 0.0) + ncol(out, "partner_lap_fit_confident_score", 0.0)
    ) / 2.0
    out["pair_lap_axis_avg"] = (
        ncol(out, "anchor_lap_axis_candidate_score", 0.0) + ncol(out, "partner_lap_axis_candidate_score", 0.0)
    ) / 2.0
    out["pair_lap_partner_specialist_max"] = np.maximum(
        ncol(out, "anchor_lap_partner_specialist_score", 0.0),
        ncol(out, "partner_lap_partner_specialist_score", 0.0),
    )
    out["pair_lap_mismatch_popular_max"] = np.maximum(
        ncol(out, "anchor_lap_mismatch_popular_risk", 0.0),
        ncol(out, "partner_lap_mismatch_popular_risk", 0.0),
    )
    anchor_axis = ncol(out, "anchor_lap_axis_candidate_score", 0.0).clip(0.0, 1.0)
    partner_axis = ncol(out, "partner_lap_axis_candidate_score", 0.0).clip(0.0, 1.0)
    anchor_specialist = ncol(out, "anchor_lap_partner_specialist_score", 0.0).clip(0.0, 1.0)
    partner_specialist = ncol(out, "partner_lap_partner_specialist_score", 0.0).clip(0.0, 1.0)
    role_a = (0.56 * anchor_axis + 0.44 * partner_specialist).clip(0.0, 1.0)
    role_b = (0.56 * partner_axis + 0.44 * anchor_specialist).clip(0.0, 1.0)
    out["lap_axis_specialist_role_score"] = np.maximum(role_a, role_b)
    out["lap_role_anchor_side"] = np.where(role_a >= role_b, "anchor_axis_partner_specialist", "partner_axis_anchor_specialist")

    v2_mode = out.get("v2_predicted_lap_mode", pd.Series("unknown", index=idx)).map(norm_mode)
    anchor_mode = out.get("anchor_horse_lap_profile_top_mode", pd.Series("unknown", index=idx)).map(norm_mode)
    partner_mode = out.get("partner_horse_lap_profile_top_mode", pd.Series("unknown", index=idx)).map(norm_mode)
    out["pair_both_match_v2_lap_mode"] = (anchor_mode.eq(v2_mode) & partner_mode.eq(v2_mode) & v2_mode.ne("unknown")).astype(float)
    out["pair_one_match_v2_lap_mode"] = ((anchor_mode.eq(v2_mode) | partner_mode.eq(v2_mode)) & v2_mode.ne("unknown")).astype(float)

    v2_conf = ncol(out, "v2_confidence", 0.0).clip(0.0, 1.0)
    v2_margin = ncol(out, "v2_margin", 0.0).clip(0.0, 1.0)
    quality_rank = ncol(out, "pair_quinella_score", 0.0).groupby(out["race_id"]).rank(pct=True).fillna(0.5)
    value = np.maximum(ncol(out, "market_overlay_score", 0.0), ncol(out, "late_value_survives_score", 0.0)).clip(0.0, 1.0)

    out["lap_positive_score"] = (
        0.24 * out["pair_lap_profile_fit_avg"].fillna(0.0)
        + 0.18 * out["pair_lap_confident_min"].fillna(0.0)
        + 0.16 * out["pair_lap_axis_avg"].fillna(0.0)
        + 0.12 * out["pair_lap_partner_specialist_max"].fillna(0.0)
        + 0.12 * v2_conf
        + 0.08 * v2_margin
        + 0.06 * out["pair_one_match_v2_lap_mode"]
        + 0.04 * out["pair_both_match_v2_lap_mode"]
        - 0.16 * out["pair_lap_mismatch_popular_max"].fillna(0.0)
    ).clip(0.0, 1.0)
    out["lap_expansion_select_score"] = (
        0.42 * quality_rank
        + 0.26 * out["lap_positive_score"]
        + 0.18 * value
        + 0.08 * ncol(out, "shape_pair_fit_score", 0.0).clip(0.0, 1.0)
        - 0.10 * ncol(out, "shape_pair_risk_score", 0.0).clip(0.0, 1.0)
        - 0.10 * ncol(out, "danger_sum", 0.0).clip(0.0, 1.0)
    )
    out["lap_expansion_candidate_label"] = np.select(
        [
            out["lap_positive_score"].ge(0.50) & out["pair_lap_mismatch_popular_max"].le(0.28) & v2_conf.ge(0.30),
            out["lap_positive_score"].ge(0.44) & out["pair_lap_mismatch_popular_max"].le(0.34) & v2_conf.ge(0.24),
            out["lap_axis_specialist_role_score"].ge(0.50) & out["pair_lap_mismatch_popular_max"].le(0.30) & v2_conf.ge(0.24),
        ],
        ["lap_promote_strong", "lap_promote_watch", "lap_role_watch"],
        default="lap_neutral",
    )
    out["lap_role_expansion_select_score"] = (
        0.38 * quality_rank
        + 0.28 * out["lap_axis_specialist_role_score"].fillna(0.0)
        + 0.16 * value
        + 0.10 * v2_conf
        - 0.12 * out["pair_lap_mismatch_popular_max"].fillna(0.0)
        - 0.08 * ncol(out, "danger_sum", 0.0).clip(0.0, 1.0)
    )
    return out


def select_top_per_race(frame: pd.DataFrame, score_col: str, ticket_type: str, policy: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    selected = (
        frame.sort_values(["race_id", score_col, "pair_quinella_score"], ascending=[True, False, False], kind="mergesort")
        .drop_duplicates("race_id", keep="first")
        .copy()
    )
    selected["policy"] = policy
    selected["ticket_type"] = ticket_type
    selected["stake_yen"] = 100.0
    selected["return_yen"] = ticket_return(selected, ticket_type)
    selected["hit"] = hit_flag(selected, ticket_type)
    selected["pair_key_norm"] = selected.apply(pair_key_from_row, axis=1)
    return selected


def evaluate_expansion(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    selection_frames: list[pd.DataFrame] = []

    base_gate = "price_sane_strong"
    promotion_gates = ["value_mid", "value_loose"]
    thresholds = [
        ("lap_q60", 0.60, 0.32, 0.38),
        ("lap_q70", 0.70, 0.30, 0.42),
        ("lap_q80", 0.80, 0.28, 0.46),
        ("lap_q90", 0.90, 0.24, 0.50),
    ]
    role_thresholds = [
        ("role_q60", 0.60, 0.32, 0.40),
        ("role_q70", 0.70, 0.30, 0.44),
        ("role_q80", 0.80, 0.28, 0.48),
        ("role_q90", 0.90, 0.24, 0.52),
    ]

    base_pool = df[gate_mask(df, base_gate)].copy()
    for ticket_type in ["wide", "umaren"]:
        base = select_top_per_race(base_pool, "shape_base_rank_score", ticket_type, f"{ticket_type}_{base_gate}_base")
        summary_rows.append(metrics(base, f"{ticket_type}_{base_gate}_base"))
        for year, group in base.groupby("year"):
            row = metrics(group, f"{ticket_type}_{base_gate}_base")
            row["year"] = int(year)
            yearly_rows.append(row)
        selection_frames.append(base)

        base_races = set(base["race_id"].astype(str))
        for promote_gate in promotion_gates:
            loose_pool = df[gate_mask(df, promote_gate)].copy()
            extra_pool = loose_pool[~loose_pool["race_id"].astype(str).isin(base_races)].copy()
            if extra_pool.empty:
                continue
            for label, q, mismatch_max, min_score in thresholds:
                threshold = float(extra_pool["lap_positive_score"].quantile(q))
                mask = (
                    extra_pool["lap_positive_score"].ge(max(threshold, min_score))
                    & extra_pool["pair_lap_mismatch_popular_max"].le(mismatch_max)
                    & ncol(extra_pool, "v2_confidence", 0.0).ge(0.22)
                    & ncol(extra_pool, "pair_quinella_score", 0.0).ge(0.58)
                )
                promoted = select_top_per_race(
                    extra_pool[mask],
                    "lap_expansion_select_score",
                    ticket_type,
                    f"{ticket_type}_{promote_gate}_{label}_extra",
                )
                if promoted.empty:
                    continue
                combined = pd.concat([base, promoted], ignore_index=True, sort=False)
                policy = f"{ticket_type}_{base_gate}_plus_{promote_gate}_{label}"
                summary_rows.append(metrics(promoted, f"{policy}_extra_only"))
                summary_rows.append(metrics(combined, policy))
                promoted = promoted.copy()
                promoted["policy"] = f"{policy}_extra_only"
                combined = combined.copy()
                combined["policy"] = policy
                selection_frames.extend([promoted, combined])
                for year, group in promoted.groupby("year"):
                    row = metrics(group, f"{policy}_extra_only")
                    row["year"] = int(year)
                    yearly_rows.append(row)
                for year, group in combined.groupby("year"):
                    row = metrics(group, policy)
                    row["year"] = int(year)
                    yearly_rows.append(row)

            for label, q, mismatch_max, min_score in role_thresholds:
                threshold = float(extra_pool["lap_axis_specialist_role_score"].quantile(q))
                mask = (
                    extra_pool["lap_axis_specialist_role_score"].ge(max(threshold, min_score))
                    & extra_pool["pair_lap_mismatch_popular_max"].le(mismatch_max)
                    & ncol(extra_pool, "v2_confidence", 0.0).ge(0.22)
                    & ncol(extra_pool, "pair_quinella_score", 0.0).ge(0.58)
                )
                promoted = select_top_per_race(
                    extra_pool[mask],
                    "lap_role_expansion_select_score",
                    ticket_type,
                    f"{ticket_type}_{promote_gate}_{label}_extra",
                )
                if promoted.empty:
                    continue
                combined = pd.concat([base, promoted], ignore_index=True, sort=False)
                policy = f"{ticket_type}_{base_gate}_plus_{promote_gate}_{label}"
                summary_rows.append(metrics(promoted, f"{policy}_extra_only"))
                summary_rows.append(metrics(combined, policy))
                promoted = promoted.copy()
                promoted["policy"] = f"{policy}_extra_only"
                combined = combined.copy()
                combined["policy"] = policy
                selection_frames.extend([promoted, combined])
                for year, group in promoted.groupby("year"):
                    row = metrics(group, f"{policy}_extra_only")
                    row["year"] = int(year)
                    yearly_rows.append(row)
                for year, group in combined.groupby("year"):
                    row = metrics(group, policy)
                    row["year"] = int(year)
                    yearly_rows.append(row)

        # Replacement within the same strict gate, for comparison.
        for weight in [0.10, 0.18, 0.26, 0.34]:
            score_col = f"lap_blend_score_w{weight:.2f}"
            base_pool[score_col] = (
                (1.0 - weight) * ncol(base_pool, "shape_base_rank_score", 0.5)
                + weight * ncol(base_pool, "lap_expansion_select_score", 0.0)
            )
            repl = select_top_per_race(base_pool, score_col, ticket_type, f"{ticket_type}_{base_gate}_lap_replace_w{weight:.2f}")
            base_keys = set(base["pair_key_norm"].astype(str))
            repl["changed_from_base"] = ~repl["pair_key_norm"].astype(str).isin(base_keys)
            row = metrics(repl, f"{ticket_type}_{base_gate}_lap_replace_w{weight:.2f}")
            row["changed_tickets"] = int(repl["changed_from_base"].sum())
            if repl["changed_from_base"].any():
                changed = repl[repl["changed_from_base"]].copy()
                row["changed_roi"] = metrics(changed, "changed")["roi"]
                row["changed_hit_rate"] = metrics(changed, "changed")["hit_rate"]
            else:
                row["changed_roi"] = 0.0
                row["changed_hit_rate"] = 0.0
            summary_rows.append(row)
            repl["policy"] = f"{ticket_type}_{base_gate}_lap_replace_w{weight:.2f}"
            selection_frames.append(repl)

    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    selections = pd.concat(selection_frames, ignore_index=True, sort=False) if selection_frames else pd.DataFrame()
    return summary, yearly, selections


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate positive lap-based promotion of non-BUY pair candidates.")
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--race-shape-csv", type=Path, default=DEFAULT_RACE_SHAPE)
    parser.add_argument("--runner-lap-csv", type=Path, default=DEFAULT_RUNNER_LAP)
    parser.add_argument("--race-quality-csv", type=Path, default=DEFAULT_RACE_QUALITY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    universe = load_universe(args.universe_csv, args.race_shape_csv)
    universe = add_shape_scores(universe)
    runner_lap = load_runner_lap(args.runner_lap_csv)
    enriched = add_lap_features(universe, runner_lap, args.race_quality_csv)
    summary, yearly, selections = evaluate_expansion(enriched)

    summary_sorted = summary.sort_values(["roi", "tickets"], ascending=[False, False], kind="mergesort")
    practical = summary[
        summary["policy"].astype(str).str.contains("_plus_")
        & ~summary["policy"].astype(str).str.endswith("_extra_only")
        & summary["tickets"].ge(100)
    ].sort_values(["roi", "tickets"], ascending=[False, False], kind="mergesort")

    enriched_out_cols = [
        "race_id",
        "year",
        "anchor_no",
        "anchor_name",
        "partner_no",
        "partner_name",
        "horse_a",
        "horse_b",
        "pair_key_norm",
        "anchor_pop",
        "partner_pop",
        "anchor_odds",
        "partner_odds",
        "pair_quinella_score",
        "market_overlay_score",
        "late_value_survives_score",
        "shape_pair_fit_score",
        "shape_pair_risk_score",
        "lap_positive_score",
        "lap_expansion_select_score",
        "lap_axis_specialist_role_score",
        "lap_role_expansion_select_score",
        "lap_role_anchor_side",
        "lap_expansion_candidate_label",
        "pair_lap_profile_fit_min",
        "pair_lap_profile_fit_avg",
        "pair_lap_confident_min",
        "pair_lap_axis_avg",
        "pair_lap_partner_specialist_max",
        "pair_lap_mismatch_popular_max",
        "v2_predicted_lap_mode",
        "v2_confidence",
        "v2_margin",
        "wide_pay",
        "wide_hit",
        "umaren_pay",
        "umaren_hit",
    ]
    enriched_sample = (
        enriched.sort_values("lap_expansion_select_score", ascending=False)
        [[c for c in enriched_out_cols if c in enriched.columns]]
        .head(5000)
    )

    summary_sorted.to_csv(args.out_dir / "lap_positive_expansion_policy_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(args.out_dir / "lap_positive_expansion_by_year.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(args.out_dir / "lap_positive_expansion_selected_tickets.csv", index=False, encoding="utf-8-sig")
    enriched_sample.to_csv(args.out_dir / "lap_positive_expansion_top_candidates_sample.csv", index=False, encoding="utf-8-sig")

    best = summary_sorted.head(12).to_dict(orient="records")
    best_practical = practical.head(12).to_dict(orient="records")
    summary_json = {
        "universe_csv": str(args.universe_csv),
        "runner_lap_csv": str(args.runner_lap_csv),
        "race_quality_csv": str(args.race_quality_csv),
        "rows": {
            "universe": int(len(universe)),
            "races": int(universe["race_id"].nunique()),
            "runner_lap": int(len(runner_lap)),
            "enriched": int(len(enriched)),
        },
        "base_gate": "price_sane_strong",
        "promotion_gates": ["value_mid", "value_loose"],
        "top_overall": best,
        "top_practical_expansions": best_practical,
        "note": (
            "This is a positive promotion test. It adds lap-qualified extra races outside the strict base gate, "
            "then compares combined ROI/hit rate against the base strict policy. No production BUY gate is changed."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = [
        "# Lap Positive Expansion",
        "",
        "目的: ラップ系特徴を、買い控えだけでなく準候補の昇格に使えるか検証する。",
        "",
        "## Top Practical Expansions",
        "",
    ]
    for row in best_practical[:8]:
        readme.append(
            f"- {row['policy']}: tickets={row['tickets']}, races={row['races']}, "
            f"ROI={row['roi']:.3f}, hit={row['hit_rate']:.3f}, "
            f"top10_removed_roi={row['top10_removed_roi']:.3f}"
        )
    readme.append("")
    readme.append("正式BUYは変更していない。実運用では shadow / 昇格候補として扱う。")
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary_json, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
