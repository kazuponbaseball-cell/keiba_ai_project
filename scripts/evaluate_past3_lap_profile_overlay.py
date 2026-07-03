from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_historical_condition_lap_context import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_FRONT3F,
)
from scripts.evaluate_historical_race_quality_overlay import (  # noqa: E402
    add_race_quality_scores,
    build_prior_context_by_race,
    load_race_base,
    load_runner_features,
    project_path,
    read_csv_any,
)
from scripts.evaluate_shape_adjusted_pair_selection import (  # noqa: E402
    DEFAULT_RACE_SHAPE,
    DEFAULT_UNIVERSE,
    add_shape_scores,
    bool_col,
    gate_mask,
    hit_flag,
    load_universe,
    metrics,
    ncol,
    ticket_return,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "analysis" / "past3_lap_profile_overlay_v1"

RACE_COL = "レースID(新/馬番無)"
DATE_COL = "日付"
HORSE_COL = "血統登録番号"
HORSE_NO_COL = "馬番"

LAP_KEEP_COLS = [
    DATE_COL,
    RACE_COL,
    HORSE_COL,
    HORSE_NO_COL,
    "PCI",
    "PCI3",
    "RPCI",
    "Ave-3F",
    "target_score",
    "確定着順",
    "人気",
    "芝・ダ",
    "距離",
    "馬場状態",
    "クラス名",
]


def num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        if index is None:
            raise ValueError("index is required")
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def clip01(series: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return pd.Series(series).fillna(0.0).clip(0.0, 1.0)


def parse_date_series(series: pd.Series) -> pd.Series:
    raw = series.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip()
    out = pd.to_datetime(raw, errors="coerce")
    yymmdd = raw.str.fullmatch(r"\d{6}")
    if yymmdd.any():
        out.loc[yymmdd] = pd.to_datetime(raw.loc[yymmdd], format="%y%m%d", errors="coerce")
    yyyymmdd = raw.str.fullmatch(r"\d{8}")
    if yyyymmdd.any():
        out.loc[yyyymmdd] = pd.to_datetime(raw.loc[yyyymmdd], format="%Y%m%d", errors="coerce")
    return out


def load_lap_rows(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        header = read_csv_any(path, nrows=0)
        usecols = [c for c in LAP_KEEP_COLS if c in header.columns]
        if not {DATE_COL, RACE_COL, HORSE_COL, HORSE_NO_COL}.issubset(usecols):
            continue
        frames.append(read_csv_any(path, usecols=usecols))
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw["race_id"] = raw[RACE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    raw["horse_no"] = pd.to_numeric(raw[HORSE_NO_COL], errors="coerce").astype("Int64")
    raw["horse_id"] = raw[HORSE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    raw["race_date"] = parse_date_series(raw[DATE_COL])
    for col in ["PCI", "PCI3", "RPCI", "Ave-3F", "target_score", "確定着順", "人気", "距離"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["race_id", "horse_no", "horse_id", "race_date"]).copy()
    raw = raw.sort_values(["horse_id", "race_date", "race_id"], kind="mergesort")
    raw = raw.drop_duplicates(["race_id", "horse_no"], keep="last")
    return raw


def add_lag_lap_profiles(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    rpci = num(out.get("RPCI"), out.index)
    pci = num(out.get("PCI"), out.index)
    pci3 = num(out.get("PCI3"), out.index)
    score = num(out.get("target_score"), out.index, 0.0).fillna(0.0).clip(0.0, 1.0)

    out["_lap_fast_regime"] = ((50.0 - rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)
    out["_lap_slow_regime"] = ((rpci - 50.0) / 5.0).clip(0.0, 1.0).fillna(0.0)
    out["_lap_instant_regime"] = ((pci - rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)
    out["_lap_sustain_regime"] = (1.0 - (pci - rpci).abs() / 4.0).clip(0.0, 1.0).fillna(0.0)
    out["_lap_long_spurt_regime"] = ((pci3 - rpci) / 5.0).clip(0.0, 1.0).fillna(0.0)

    for kind in ["fast", "slow", "instant", "sustain", "long_spurt"]:
        out[f"_lap_{kind}_success"] = (out[f"_lap_{kind}_regime"] * score).clip(0.0, 1.0)

    lag_sources = [
        "PCI",
        "PCI3",
        "RPCI",
        "Ave-3F",
        "target_score",
        "_lap_fast_regime",
        "_lap_slow_regime",
        "_lap_instant_regime",
        "_lap_sustain_regime",
        "_lap_long_spurt_regime",
        "_lap_fast_success",
        "_lap_slow_success",
        "_lap_instant_success",
        "_lap_sustain_success",
        "_lap_long_spurt_success",
    ]
    grouped = out.groupby("horse_id", sort=False)
    for lag in (1, 2, 3):
        for col in lag_sources:
            out[f"past{lag}_{col.removeprefix('_')}"] = grouped[col].shift(lag)
        out[f"past{lag}_race_id"] = grouped["race_id"].shift(lag)
        out[f"past{lag}_race_date"] = grouped["race_date"].shift(lag)
        if "芝・ダ" in out.columns:
            out[f"past{lag}_surface"] = grouped["芝・ダ"].shift(lag)
        if "距離" in out.columns:
            out[f"past{lag}_distance"] = grouped["距離"].shift(lag)
        if "馬場状態" in out.columns:
            out[f"past{lag}_going"] = grouped["馬場状態"].shift(lag)
        if "クラス名" in out.columns:
            out[f"past{lag}_class_name"] = grouped["クラス名"].shift(lag)

    keep = ["race_id", "horse_no"]
    for lag in (1, 2, 3):
        keep.extend(
            [
                f"past{lag}_race_id",
                f"past{lag}_race_date",
                f"past{lag}_lap_fast_success",
                f"past{lag}_lap_slow_success",
                f"past{lag}_lap_instant_success",
                f"past{lag}_lap_sustain_success",
                f"past{lag}_lap_long_spurt_success",
                f"past{lag}_lap_fast_regime",
                f"past{lag}_lap_slow_regime",
                f"past{lag}_lap_instant_regime",
                f"past{lag}_lap_sustain_regime",
                f"past{lag}_lap_long_spurt_regime",
                f"past{lag}_RPCI",
                f"past{lag}_PCI",
                f"past{lag}_PCI3",
                f"past{lag}_target_score",
            ]
        )
    return out[[c for c in keep if c in out.columns]].drop_duplicates(["race_id", "horse_no"], keep="last")


def side_lag_fit(frame: pd.DataFrame, side: str) -> pd.Series:
    fast_need = ncol(frame, "race_quality_fast_need_score", 0.0).clip(0.0, 1.0)
    slow_need = ncol(frame, "race_quality_slow_need_score", 0.0).clip(0.0, 1.0)
    sustain_need = ncol(frame, "race_quality_sustain_need_score", 1.0).clip(0.0, 1.0)
    weights = {1: 1.00, 2: 0.86, 3: 0.72}
    lag_scores: list[pd.Series] = []
    evidence: list[pd.Series] = []

    for lag, weight in weights.items():
        prefix = f"past3_{side}_past{lag}"
        fast = ncol(frame, f"{prefix}_lap_fast_success", 0.0).clip(0.0, 1.0)
        slow = ncol(frame, f"{prefix}_lap_slow_success", 0.0).clip(0.0, 1.0)
        instant = ncol(frame, f"{prefix}_lap_instant_success", 0.0).clip(0.0, 1.0)
        sustain = ncol(frame, f"{prefix}_lap_sustain_success", 0.0).clip(0.0, 1.0)
        long_spurt = ncol(frame, f"{prefix}_lap_long_spurt_success", 0.0).clip(0.0, 1.0)
        lag_fit = (
            fast_need * (0.72 * fast + 0.18 * sustain + 0.10 * long_spurt)
            + slow_need * (0.58 * slow + 0.24 * instant + 0.18 * sustain)
            + sustain_need * (0.42 * sustain + 0.34 * long_spurt + 0.24 * instant)
        ).clip(0.0, 1.0)
        has_lag = (
            frame.get(f"{prefix}_race_id", pd.Series("", index=frame.index))
            .astype("string")
            .fillna("")
            .ne("")
            .astype(float)
        )
        lag_scores.append(weight * lag_fit * has_lag)
        evidence.append(weight * has_lag)

    denom = sum(evidence).replace(0.0, np.nan)
    weighted = (sum(lag_scores) / denom).replace([np.inf, -np.inf], np.nan)
    best = pd.concat([s / w for s, w in zip(lag_scores, weights.values())], axis=1).max(axis=1)
    ready = (sum(evidence) > 0).astype(float)
    return (0.70 * weighted.fillna(0.5) + 0.30 * best.fillna(0.5)).where(ready.gt(0), 0.5).clip(0.0, 1.0)


def add_past3_pair_lap_scores(df: pd.DataFrame, lag_profiles: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if lag_profiles.empty:
        out["past3_lap_pair_fit_score"] = 0.5
        out["past3_lap_pair_min_score"] = 0.5
        out["past3_lap_evidence_ready"] = 0.0
        return out

    profiles = lag_profiles.copy()
    profiles["horse_no"] = pd.to_numeric(profiles["horse_no"], errors="coerce").astype("Int64")
    for side, no_col in [("anchor", "anchor_no"), ("partner", "partner_no")]:
        rename = {c: f"past3_{side}_{c}" for c in profiles.columns if c not in ["race_id", "horse_no"]}
        side_features = profiles.rename(columns={"horse_no": no_col, **rename})
        side_features[no_col] = pd.to_numeric(side_features[no_col], errors="coerce").astype("Int64")
        out[no_col] = pd.to_numeric(out[no_col], errors="coerce").astype("Int64")
        out = out.merge(side_features, on=["race_id", no_col], how="left")

    out["anchor_past3_lap_fit_score"] = side_lag_fit(out, "anchor")
    out["partner_past3_lap_fit_score"] = side_lag_fit(out, "partner")
    avg = (out["anchor_past3_lap_fit_score"] + out["partner_past3_lap_fit_score"]) / 2.0
    max_fit = np.maximum(out["anchor_past3_lap_fit_score"], out["partner_past3_lap_fit_score"])
    out["past3_lap_pair_fit_score"] = (0.56 * max_fit + 0.44 * avg).clip(0.0, 1.0)
    out["past3_lap_pair_min_score"] = np.minimum(out["anchor_past3_lap_fit_score"], out["partner_past3_lap_fit_score"]).clip(0.0, 1.0)

    evidence_cols = [
        c
        for c in out.columns
        if c.startswith("past3_anchor_past") and c.endswith("_race_id")
    ] + [
        c
        for c in out.columns
        if c.startswith("past3_partner_past") and c.endswith("_race_id")
    ]
    if evidence_cols:
        evidence = out[evidence_cols].astype("string").fillna("").ne("").sum(axis=1)
        out["past3_lap_evidence_ready"] = evidence.gt(0).astype(float)
        out["past3_lap_evidence_count"] = evidence
    else:
        out["past3_lap_evidence_ready"] = 0.0
        out["past3_lap_evidence_count"] = 0
    return out


def select_top_per_race(
    frame: pd.DataFrame,
    score_col: str,
    ticket_type: str,
    policy: str,
    gate: str,
    shape_weight: float,
    hist_weight: float,
    lap_weight: float,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    sort_cols = [c for c in ["race_id", score_col, "market_overlay_score", "pair_quinella_score"] if c in frame.columns]
    selected = (
        frame.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1))
        .drop_duplicates("race_id", keep="first")
        .copy()
    )
    selected["policy"] = policy
    selected["gate"] = gate
    selected["shape_weight"] = shape_weight
    selected["hist_weight"] = hist_weight
    selected["past3_lap_weight"] = lap_weight
    selected["ticket_type"] = ticket_type
    selected["stake_yen"] = 100.0
    selected["return_yen"] = ticket_return(selected, ticket_type)
    selected["hit"] = hit_flag(selected, ticket_type)
    selected["ticket_key"] = selected.apply(
        lambda r: f"{ticket_type}:{r['race_id']}:{int(r['horse_a'])}-{int(r['horse_b'])}", axis=1
    )
    return selected


def run_backtest(
    df: pd.DataFrame,
    gates: list[str],
    shape_weights: list[float],
    hist_weights: list[float],
    lap_weights: list[float],
    ticket_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []

    for gate in gates:
        gated = df[gate_mask(df, gate)].copy()
        if gated.empty:
            continue
        for ticket_type in ticket_types:
            for shape_weight in shape_weights:
                risk_weight = min(0.12, shape_weight * 0.75)
                base_col = f"base_shape_score_w{shape_weight:.2f}"
                gated[base_col] = (
                    (1.0 - shape_weight) * gated["shape_base_rank_score"]
                    + shape_weight * gated["shape_pair_fit_score"]
                    + 0.06 * gated["shape_value_score"]
                    - risk_weight * gated["shape_pair_risk_score"]
                )
                baseline = select_top_per_race(
                    gated,
                    base_col,
                    ticket_type,
                    f"{ticket_type}_{gate}_shape_w{shape_weight:.2f}_hist_w0.00_past3lap_w0.00",
                    gate,
                    shape_weight,
                    0.0,
                    0.0,
                )
                baseline_keys = baseline[["race_id", "pair_key_norm"]].rename(columns={"pair_key_norm": "baseline_pair_key_norm"})

                for hist_weight in hist_weights:
                    hist_col = f"hist_score_s{shape_weight:.2f}_h{hist_weight:.2f}"
                    if hist_weight == 0:
                        gated[hist_col] = gated[base_col]
                    else:
                        gated[hist_col] = (
                            gated[base_col]
                            + hist_weight * (ncol(gated, "race_quality_pair_fit_score", 0.5) - 0.5)
                            + 0.25 * hist_weight * (ncol(gated, "race_quality_pair_min_score", 0.5) - 0.5)
                        )
                    for lap_weight in lap_weights:
                        if hist_weight == 0 and lap_weight == 0:
                            selected = baseline.copy()
                            selected["hist_weight"] = 0.0
                            selected["past3_lap_weight"] = 0.0
                        else:
                            score_col = f"past3_lap_score_s{shape_weight:.2f}_h{hist_weight:.2f}_l{lap_weight:.2f}"
                            gated[score_col] = (
                                gated[hist_col]
                                + lap_weight * (ncol(gated, "past3_lap_pair_fit_score", 0.5) - 0.5)
                                + 0.25 * lap_weight * (ncol(gated, "past3_lap_pair_min_score", 0.5) - 0.5)
                            )
                            selected = select_top_per_race(
                                gated,
                                score_col,
                                ticket_type,
                                f"{ticket_type}_{gate}_shape_w{shape_weight:.2f}_hist_w{hist_weight:.2f}_past3lap_w{lap_weight:.2f}",
                                gate,
                                shape_weight,
                                hist_weight,
                                lap_weight,
                            )
                        selected = selected.merge(baseline_keys, on="race_id", how="left")
                        selected["changed_from_baseline"] = selected["pair_key_norm"].ne(selected["baseline_pair_key_norm"])
                        selections.append(selected)

                        row = metrics(selected, selected["policy"].iloc[0] if not selected.empty else f"{ticket_type}_{gate}")
                        row["gate"] = gate
                        row["ticket_type"] = ticket_type
                        row["shape_weight"] = shape_weight
                        row["hist_weight"] = hist_weight
                        row["past3_lap_weight"] = lap_weight
                        changed = selected[selected["changed_from_baseline"]].copy()
                        changed_metrics = metrics(changed, "changed_only")
                        row["changed_tickets"] = changed_metrics["tickets"]
                        row["changed_roi_pct"] = changed_metrics["roi_pct"]
                        row["changed_hit_rate_pct"] = changed_metrics["hit_rate_pct"]
                        row["avg_past3_lap_fit"] = round(float(ncol(selected, "past3_lap_pair_fit_score", 0.5).mean()), 3) if not selected.empty else np.nan
                        row["past3_lap_ready_rate_pct"] = (
                            round(float(ncol(selected, "past3_lap_evidence_ready", 0.0).mean() * 100), 1) if not selected.empty else 0.0
                        )
                        summary_rows.append(row)

                        for year, gy in selected.groupby("year"):
                            yr = metrics(gy, selected["policy"].iloc[0])
                            yr["year"] = int(year)
                            yr["gate"] = gate
                            yr["ticket_type"] = ticket_type
                            yr["shape_weight"] = shape_weight
                            yr["hist_weight"] = hist_weight
                            yr["past3_lap_weight"] = lap_weight
                            yearly_rows.append(yr)

    detail = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    if not summary.empty:
        baseline = summary[summary["hist_weight"].eq(0.0) & summary["past3_lap_weight"].eq(0.0)][
            ["ticket_type", "gate", "shape_weight", "roi_pct", "hit_rate_pct", "tickets"]
        ].rename(
            columns={
                "roi_pct": "baseline_roi_pct",
                "hit_rate_pct": "baseline_hit_rate_pct",
                "tickets": "baseline_tickets",
            }
        )
        summary = summary.merge(baseline, on=["ticket_type", "gate", "shape_weight"], how="left")
        summary["roi_delta_vs_baseline_pct"] = summary["roi_pct"] - summary["baseline_roi_pct"]
        summary["hit_delta_vs_baseline_pct"] = summary["hit_rate_pct"] - summary["baseline_hit_rate_pct"]
        summary = summary.sort_values(["roi_pct", "tickets"], ascending=[False, False])
    return summary, yearly, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest explicit past1/2/3 lap-profile overlay for pair selection.")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--race-shape", default=str(DEFAULT_RACE_SHAPE))
    parser.add_argument("--feature-csv", action="append", default=[])
    parser.add_argument("--front3f-csv", default=DEFAULT_FRONT3F)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--shape-weights", default="0.70,0.85,1.00")
    parser.add_argument("--hist-weights", default="0,0.08")
    parser.add_argument("--past3-lap-weights", default="0,0.04,0.08,0.12,0.16,0.24")
    parser.add_argument("--gates", default="all,value_loose,value_mid,price_sane_strong")
    parser.add_argument("--ticket-types", default="umaren,wide")
    args = parser.parse_args()

    universe_path = project_path(args.universe)
    race_shape_path = project_path(args.race_shape)
    feature_paths = [project_path(p) for p in (args.feature_csv or DEFAULT_FEATURES)]
    front3f_path = project_path(args.front3f_csv)
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_weights = [float(x.strip()) for x in args.shape_weights.split(",") if x.strip()]
    hist_weights = [float(x.strip()) for x in args.hist_weights.split(",") if x.strip()]
    lap_weights = [float(x.strip()) for x in args.past3_lap_weights.split(",") if x.strip()]
    gates = [x.strip() for x in args.gates.split(",") if x.strip()]
    ticket_types = [x.strip() for x in args.ticket_types.split(",") if x.strip()]

    universe = add_shape_scores(load_universe(universe_path, race_shape_path))
    race_ids = set(universe["race_id"].astype(str))
    race_source = load_race_base(feature_paths, front3f_path)
    context = build_prior_context_by_race(race_source, race_ids, years=args.years)
    runner_features = load_runner_features(feature_paths)
    scored = add_race_quality_scores(universe, context, runner_features)

    lap_rows = load_lap_rows(feature_paths)
    lag_profiles = add_lag_lap_profiles(lap_rows) if not lap_rows.empty else pd.DataFrame()
    scored = add_past3_pair_lap_scores(scored, lag_profiles)

    summary, yearly, detail = run_backtest(scored, gates, shape_weights, hist_weights, lap_weights, ticket_types)

    score_cols = [
        "race_id",
        "year",
        "pair_key_norm",
        "anchor_no",
        "partner_no",
        "race_quality_label",
        "race_quality_fast_need_score",
        "race_quality_slow_need_score",
        "race_quality_sustain_need_score",
        "anchor_past3_lap_fit_score",
        "partner_past3_lap_fit_score",
        "past3_lap_pair_fit_score",
        "past3_lap_pair_min_score",
        "past3_lap_evidence_count",
        "wide_hit",
        "umaren_hit",
        "wide_pay",
        "umaren_pay",
    ]
    scored[[c for c in score_cols if c in scored.columns]].to_csv(out_dir / "past3_lap_pair_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "past3_lap_overlay_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "past3_lap_overlay_yearly.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(out_dir / "past3_lap_overlay_detail.csv", index=False, encoding="utf-8-sig")

    best = summary.head(30).replace({np.nan: None}).to_dict(orient="records") if not summary.empty else []
    baseline = (
        summary[summary["hist_weight"].eq(0.0) & summary["past3_lap_weight"].eq(0.0)]
        .sort_values(["gate", "ticket_type", "shape_weight"])
        .replace({np.nan: None})
        .to_dict(orient="records")
        if not summary.empty
        else []
    )
    report = {
        "universe": str(universe_path.relative_to(ROOT)),
        "rows": int(len(scored)),
        "races": int(scored["race_id"].nunique()) if not scored.empty else 0,
        "lag_profile_rows": int(len(lag_profiles)),
        "past3_lap_ready_rate_pct": round(float(scored["past3_lap_evidence_ready"].mean() * 100), 1)
        if "past3_lap_evidence_ready" in scored
        else 0.0,
        "years": args.years,
        "top_policies": best,
        "baseline_policies": baseline,
        "note": "Shadow backtest. Past1/2/3 lap profiles are kept separate and matched to expected race quality; no production BUY gate is changed by this script.",
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
