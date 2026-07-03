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
    estimate_pass_1000m_sec,
    load_feature_races,
    load_front3f_races,
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
OUT_DIR = ROOT / "outputs" / "analysis" / "historical_race_quality_overlay_v1"

FEATURE_COLS = [
    "レースID(新/馬番無)",
    "馬番",
    "past3_avg_time_value",
    "past3_best_time_value",
    "prev_race_time_value",
    "prev_class_time_value_score",
    "horse_time_value_plus_margin",
    "horse_fast_lap_score_past5",
    "horse_slow_lap_score_past5",
    "horse_sustain_lap_score_past5",
    "horse_long_spurt_lap_score_past5",
    "lap_aptitude_fit_score",
    "pace_fit_score",
    "horse_front_run_rate_past5",
    "front_running_tendency",
]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv_any(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        if index is None:
            raise ValueError("index is required")
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def norm01(series: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if lo is None:
        lo = float(x.quantile(0.05)) if x.notna().any() else 0.0
    if hi is None:
        hi = float(x.quantile(0.95)) if x.notna().any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=x.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5).clip(0.0, 1.0)


def load_runner_features(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        header = read_csv_any(path, nrows=0)
        usecols = [c for c in FEATURE_COLS if c in header.columns]
        if "レースID(新/馬番無)" not in usecols or "馬番" not in usecols:
            continue
        part = read_csv_any(path, usecols=usecols)
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["race_id", "horse_no"])

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw["race_id"] = raw["レースID(新/馬番無)"].astype(str).str.replace(r"\.0$", "", regex=True)
    raw["horse_no"] = pd.to_numeric(raw["馬番"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["race_id", "horse_no"]).drop_duplicates(["race_id", "horse_no"], keep="last")

    fast_clock = (
        0.28 * norm01(num(raw.get("past3_avg_time_value"), raw.index, 0.0).fillna(0.0), lo=-0.12, hi=0.12)
        + 0.18
        * norm01(
            num(raw.get("past3_best_time_value"), raw.index, 0.0)
            .fillna(num(raw.get("prev_race_time_value"), raw.index, 0.0))
            .fillna(0.0),
            lo=-0.10,
            hi=0.18,
        )
        + 0.18 * norm01(num(raw.get("prev_class_time_value_score"), raw.index, 0.0).fillna(0.0), lo=-0.10, hi=0.18)
        + 0.16 * norm01(num(raw.get("horse_time_value_plus_margin"), raw.index, 0.0).fillna(0.0), lo=-0.18, hi=0.18)
        + 0.12 * norm01(num(raw.get("horse_fast_lap_score_past5"), raw.index, 0.0).fillna(0.0), lo=0.0, hi=0.75)
        + 0.08 * norm01(num(raw.get("lap_aptitude_fit_score"), raw.index, 0.0).fillna(0.0), lo=0.0, hi=0.40)
    ).clip(0.0, 1.0)
    raw["fast_clock_runtime_score"] = fast_clock

    keep = [
        "race_id",
        "horse_no",
        "fast_clock_runtime_score",
        "horse_fast_lap_score_past5",
        "horse_slow_lap_score_past5",
        "horse_sustain_lap_score_past5",
        "horse_long_spurt_lap_score_past5",
        "pace_fit_score",
        "horse_front_run_rate_past5",
        "front_running_tendency",
    ]
    for col in keep:
        if col not in raw.columns:
            raw[col] = np.nan
    for col in keep[2:]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw[keep]


def load_race_base(feature_paths: list[Path], front3f_path: Path) -> pd.DataFrame:
    races = load_feature_races(feature_paths)
    front3f = load_front3f_races(front3f_path)
    if not front3f.empty and not races.empty:
        races = races.merge(front3f, on="race_id", how="left")
        races["winning_time_sec"] = races["race_total_time_sec"].fillna(races["winning_time_sec"])
    races["pass_1000m_sec"] = estimate_pass_1000m_sec(races)
    races = races.dropna(subset=["race_date", "surface", "distance"]).copy()
    races["race_id"] = races["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    races["distance"] = pd.to_numeric(races["distance"], errors="coerce").astype("Int64")
    return races.sort_values(["race_date", "race_id"]).drop_duplicates("race_id", keep="last")


def best_prior_context(source: pd.DataFrame, race: pd.Series, years: int) -> dict[str, Any] | None:
    race_date = race["race_date"]
    if pd.isna(race_date):
        return None
    start = race_date - pd.DateOffset(years=years)
    prior = source[(source["race_date"] >= start) & (source["race_date"] < race_date)]
    if prior.empty:
        return None

    scope_defs = [
        ("same_venue_distance_class_going", ["venue", "surface", "distance", "class_group", "going"]),
        ("same_venue_distance_class", ["venue", "surface", "distance", "class_group"]),
        ("same_venue_distance_going", ["venue", "surface", "distance", "going"]),
        ("same_venue_distance", ["venue", "surface", "distance"]),
        ("same_surface_distance_class", ["surface", "distance", "class_group"]),
        ("same_surface_distance", ["surface", "distance"]),
    ]
    metrics_cols = [
        "winning_time_sec",
        "race_first3f_sec",
        "race_last3f_sec",
        "pass_1000m_sec",
        "rpci",
        "pci3",
    ]
    for scope, keys in scope_defs:
        part = prior
        ok = True
        for key in keys:
            if key not in prior.columns or key not in race.index:
                ok = False
                break
            part = part[part[key].eq(race[key])]
        if not ok or part.empty:
            continue
        out: dict[str, Any] = {"scope": scope, "sample_count": int(part["race_id"].nunique())}
        for col in metrics_cols:
            out[f"avg_{col}"] = float(pd.to_numeric(part.get(col), errors="coerce").mean())
        return out
    return None


def build_prior_context_by_race(source: pd.DataFrame, race_ids: set[str], years: int) -> pd.DataFrame:
    targets = source[source["race_id"].isin(race_ids)].copy()
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[Any, ...], dict[str, Any] | None] = {}
    for _, race in targets.iterrows():
        key = (
            race["race_date"],
            race.get("venue"),
            race.get("surface"),
            int(race.get("distance")) if pd.notna(race.get("distance")) else None,
            race.get("class_group"),
            race.get("going"),
        )
        ctx = cache.get(key)
        if key not in cache:
            ctx = best_prior_context(source, race, years)
            cache[key] = ctx
        row = {"race_id": race["race_id"], "race_date": race["race_date"], "venue": race.get("venue"), "surface": race.get("surface")}
        row["distance"] = int(race.get("distance")) if pd.notna(race.get("distance")) else np.nan
        row["class_group"] = race.get("class_group")
        row["going"] = race.get("going")
        if ctx is None:
            row.update(
                {
                    "hist_scope": "",
                    "hist_sample_count": 0,
                    "hist_avg_front3f_sec": np.nan,
                    "hist_avg_rpci": np.nan,
                    "hist_avg_winning_time_sec": np.nan,
                    "hist_avg_last3f_sec": np.nan,
                    "hist_avg_1000m_sec": np.nan,
                }
            )
        else:
            row.update(
                {
                    "hist_scope": ctx["scope"],
                    "hist_sample_count": ctx["sample_count"],
                    "hist_avg_front3f_sec": ctx["avg_race_first3f_sec"],
                    "hist_avg_rpci": ctx["avg_rpci"],
                    "hist_avg_winning_time_sec": ctx["avg_winning_time_sec"],
                    "hist_avg_last3f_sec": ctx["avg_race_last3f_sec"],
                    "hist_avg_1000m_sec": ctx["avg_pass_1000m_sec"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_race_quality_scores(universe: pd.DataFrame, context: pd.DataFrame, runner_features: pd.DataFrame) -> pd.DataFrame:
    out = universe.copy()
    if context.empty:
        out["race_quality_pair_fit_score"] = 0.5
        out["race_quality_context_ready"] = 0.0
        return out

    pressure_cols = [c for c in ["anchor_race_early_pressure_score", "partner_race_early_pressure_score"] if c in out.columns]
    collapse_cols = [c for c in ["anchor_race_pace_collapse_risk", "partner_race_pace_collapse_risk"] if c in out.columns]
    slow_cols = [c for c in ["anchor_race_slow_pace_risk", "partner_race_slow_pace_risk"] if c in out.columns]
    race_regime = out[["race_id"]].drop_duplicates().copy()
    race_regime["race_pressure"] = (
        out.groupby("race_id")[pressure_cols].max().max(axis=1).reindex(race_regime["race_id"]).fillna(0.0).to_numpy()
        if pressure_cols
        else 0.0
    )
    race_regime["race_collapse"] = (
        out.groupby("race_id")[collapse_cols].max().max(axis=1).reindex(race_regime["race_id"]).fillna(0.0).to_numpy()
        if collapse_cols
        else 0.0
    )
    race_regime["race_slow"] = (
        out.groupby("race_id")[slow_cols].max().max(axis=1).reindex(race_regime["race_id"]).fillna(0.0).to_numpy()
        if slow_cols
        else 0.0
    )

    race_regime = race_regime.merge(context, on="race_id", how="left")
    race_regime["front_adj"] = np.clip(-0.80 * race_regime["race_pressure"] - 0.55 * race_regime["race_collapse"] + 0.70 * race_regime["race_slow"], -1.8, 1.8)
    race_regime["rpci_adj"] = np.clip(-2.8 * race_regime["race_collapse"] - 1.0 * race_regime["race_pressure"] + 2.4 * race_regime["race_slow"], -5.0, 5.0)
    race_regime["expected_front3f_sec"] = race_regime["hist_avg_front3f_sec"] + race_regime["front_adj"]
    race_regime["expected_rpci"] = race_regime["hist_avg_rpci"] + race_regime["rpci_adj"]
    front_delta = race_regime["expected_front3f_sec"] - race_regime["hist_avg_front3f_sec"]
    rpci_delta = race_regime["expected_rpci"] - race_regime["hist_avg_rpci"]
    race_regime["race_quality_fast_need_score"] = np.clip((((-front_delta) / 1.4 + (-rpci_delta) / 5.0) / 2.0), 0.0, 1.0).fillna(0.0)
    race_regime["race_quality_slow_need_score"] = np.clip(((front_delta / 1.4 + rpci_delta / 5.0) / 2.0), 0.0, 1.0).fillna(0.0)
    race_regime["race_quality_sustain_need_score"] = (1.0 - np.maximum(race_regime["race_quality_fast_need_score"], race_regime["race_quality_slow_need_score"])).clip(0.0, 1.0)
    race_regime["race_quality_label"] = np.select(
        [race_regime["race_quality_fast_need_score"].ge(0.38), race_regime["race_quality_slow_need_score"].ge(0.38)],
        ["fast", "slow"],
        default="standard",
    )
    race_regime["race_quality_context_ready"] = race_regime["hist_sample_count"].fillna(0).gt(0).astype(float)

    features = runner_features.copy()
    features["horse_no"] = pd.to_numeric(features["horse_no"], errors="coerce").astype("Int64")
    for side, no_col in [("anchor", "anchor_no"), ("partner", "partner_no")]:
        rename = {c: f"rq_{side}_{c}" for c in features.columns if c not in ["race_id", "horse_no"]}
        side_features = features.rename(columns={"horse_no": no_col, **rename})
        side_features[no_col] = pd.to_numeric(side_features[no_col], errors="coerce").astype("Int64")
        out[no_col] = pd.to_numeric(out[no_col], errors="coerce").astype("Int64")
        out = out.merge(side_features, on=["race_id", no_col], how="left")

    out = out.merge(
        race_regime[
            [
                "race_id",
                "hist_scope",
                "hist_sample_count",
                "hist_avg_front3f_sec",
                "hist_avg_rpci",
                "expected_front3f_sec",
                "expected_rpci",
                "race_quality_fast_need_score",
                "race_quality_slow_need_score",
                "race_quality_sustain_need_score",
                "race_quality_label",
                "race_quality_context_ready",
            ]
        ],
        on="race_id",
        how="left",
    )

    def side_fit(side: str) -> pd.Series:
        fast_clock = num(out.get(f"rq_{side}_fast_clock_runtime_score"), out.index, 0.5).fillna(0.5).clip(0.0, 1.0)
        fast_lap = num(out.get(f"rq_{side}_horse_fast_lap_score_past5"), out.index, 0.0).fillna(0.0).clip(0.0, 1.0)
        slow_lap = num(out.get(f"rq_{side}_horse_slow_lap_score_past5"), out.index, 0.0).fillna(0.0).clip(0.0, 1.0)
        sustain_lap = num(out.get(f"rq_{side}_horse_sustain_lap_score_past5"), out.index, 0.0).fillna(0.0).clip(0.0, 1.0)
        long_lap = num(out.get(f"rq_{side}_horse_long_spurt_lap_score_past5"), out.index, 0.0).fillna(0.0).clip(0.0, 1.0)
        pace_fit = num(out.get(f"rq_{side}_pace_fit_score"), out.index, 0.0).fillna(0.0).clip(0.0, 1.0)
        front_rate = (
            num(out.get(f"rq_{side}_horse_front_run_rate_past5"), out.index, np.nan)
            .fillna(num(out.get(f"rq_{side}_front_running_tendency"), out.index, np.nan))
            .fillna(num(out.get(f"{side}_front_running_tendency"), out.index, 0.0))
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        fast_fit = (0.52 * fast_clock + 0.30 * fast_lap + 0.18 * pace_fit).clip(0.0, 1.0)
        slow_fit = (0.42 * slow_lap + 0.25 * front_rate + 0.20 * pace_fit + 0.13 * fast_clock).clip(0.0, 1.0)
        sustain_fit = (0.34 * sustain_lap + 0.24 * long_lap + 0.24 * pace_fit + 0.18 * fast_clock).clip(0.0, 1.0)
        q = (
            ncol(out, "race_quality_fast_need_score", 0.0) * fast_fit
            + ncol(out, "race_quality_slow_need_score", 0.0) * slow_fit
            + ncol(out, "race_quality_sustain_need_score", 1.0) * sustain_fit
        ).clip(0.0, 1.0)
        low_sample = ncol(out, "hist_sample_count", 0.0).lt(3)
        q = q.where(~low_sample, (0.70 * q + 0.30 * 0.5).clip(0.0, 1.0))
        return q.fillna(0.5).clip(0.0, 1.0)

    out["anchor_race_quality_fit_score"] = side_fit("anchor")
    out["partner_race_quality_fit_score"] = side_fit("partner")
    avg = (out["anchor_race_quality_fit_score"] + out["partner_race_quality_fit_score"]) / 2.0
    max_fit = np.maximum(out["anchor_race_quality_fit_score"], out["partner_race_quality_fit_score"])
    out["race_quality_pair_fit_score"] = (0.58 * max_fit + 0.42 * avg).clip(0.0, 1.0)
    out["race_quality_pair_min_score"] = np.minimum(out["anchor_race_quality_fit_score"], out["partner_race_quality_fit_score"]).clip(0.0, 1.0)
    out["race_quality_context_ready"] = ncol(out, "race_quality_context_ready", 0.0).fillna(0.0)
    return out


def select_top_per_race(frame: pd.DataFrame, score_col: str, ticket_type: str, policy: str, gate: str, shape_weight: float, hist_weight: float) -> pd.DataFrame:
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
    selected["ticket_type"] = ticket_type
    selected["stake_yen"] = 100.0
    selected["return_yen"] = ticket_return(selected, ticket_type)
    selected["hit"] = hit_flag(selected, ticket_type)
    selected["ticket_key"] = selected.apply(
        lambda r: f"{ticket_type}:{r['race_id']}:{int(r['horse_a'])}-{int(r['horse_b'])}", axis=1
    )
    return selected


def run_backtest(df: pd.DataFrame, gates: list[str], shape_weights: list[float], hist_weights: list[float], ticket_types: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []

    for gate in gates:
        gated_base = df[gate_mask(df, gate)].copy()
        if gated_base.empty:
            continue
        for ticket_type in ticket_types:
            for shape_weight in shape_weights:
                risk_weight = min(0.12, shape_weight * 0.75)
                base_score_col = f"base_shape_score_w{shape_weight:.2f}"
                gated_base[base_score_col] = (
                    (1.0 - shape_weight) * gated_base["shape_base_rank_score"]
                    + shape_weight * gated_base["shape_pair_fit_score"]
                    + 0.06 * gated_base["shape_value_score"]
                    - risk_weight * gated_base["shape_pair_risk_score"]
                )
                baseline = select_top_per_race(
                    gated_base,
                    base_score_col,
                    ticket_type,
                    f"{ticket_type}_{gate}_shape_w{shape_weight:.2f}_hist_w0.00",
                    gate,
                    shape_weight,
                    0.0,
                )
                baseline_keys = baseline[["race_id", "pair_key_norm"]].rename(columns={"pair_key_norm": "baseline_pair_key_norm"})
                for hist_weight in hist_weights:
                    if hist_weight == 0:
                        selected = baseline.copy()
                        selected["hist_weight"] = 0.0
                    else:
                        score_col = f"historical_quality_score_s{shape_weight:.2f}_h{hist_weight:.2f}"
                        gated_base[score_col] = (
                            gated_base[base_score_col]
                            + hist_weight * (ncol(gated_base, "race_quality_pair_fit_score", 0.5) - 0.5)
                            + 0.25 * hist_weight * (ncol(gated_base, "race_quality_pair_min_score", 0.5) - 0.5)
                        )
                        selected = select_top_per_race(
                            gated_base,
                            score_col,
                            ticket_type,
                            f"{ticket_type}_{gate}_shape_w{shape_weight:.2f}_hist_w{hist_weight:.2f}",
                            gate,
                            shape_weight,
                            hist_weight,
                        )
                    selected = selected.merge(baseline_keys, on="race_id", how="left")
                    selected["changed_from_baseline"] = selected["pair_key_norm"].ne(selected["baseline_pair_key_norm"])
                    selections.append(selected)

                    row = metrics(selected, selected["policy"].iloc[0] if not selected.empty else f"{ticket_type}_{gate}")
                    row["gate"] = gate
                    row["ticket_type"] = ticket_type
                    row["shape_weight"] = shape_weight
                    row["hist_weight"] = hist_weight
                    changed = selected[selected["changed_from_baseline"]].copy()
                    changed_metrics = metrics(changed, "changed_only")
                    row["changed_tickets"] = changed_metrics["tickets"]
                    row["changed_roi_pct"] = changed_metrics["roi_pct"]
                    row["changed_hit_rate_pct"] = changed_metrics["hit_rate_pct"]
                    row["avg_race_quality_fit"] = round(float(ncol(selected, "race_quality_pair_fit_score", 0.5).mean()), 3) if not selected.empty else np.nan
                    row["context_ready_rate_pct"] = round(float(ncol(selected, "race_quality_context_ready", 0.0).mean() * 100), 1) if not selected.empty else 0.0
                    summary_rows.append(row)

                    for year, gy in selected.groupby("year"):
                        yr = metrics(gy, selected["policy"].iloc[0])
                        yr["year"] = int(year)
                        yr["gate"] = gate
                        yr["ticket_type"] = ticket_type
                        yr["shape_weight"] = shape_weight
                        yr["hist_weight"] = hist_weight
                        yearly_rows.append(yr)

    detail = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    if not summary.empty:
        baseline = summary[summary["hist_weight"].eq(0.0)][
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
    parser = argparse.ArgumentParser(description="Backtest historical same-condition race-quality overlay on pair selection ROI.")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--race-shape", default=str(DEFAULT_RACE_SHAPE))
    parser.add_argument("--feature-csv", action="append", default=[])
    parser.add_argument("--front3f-csv", default=DEFAULT_FRONT3F)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--shape-weights", default="0.70,0.85,1.00")
    parser.add_argument("--hist-weights", default="0,0.04,0.08,0.12,0.16,0.24")
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
    gates = [x.strip() for x in args.gates.split(",") if x.strip()]
    ticket_types = [x.strip() for x in args.ticket_types.split(",") if x.strip()]

    universe = load_universe(universe_path, race_shape_path)
    universe = add_shape_scores(universe)
    race_ids = set(universe["race_id"].astype(str))
    race_source = load_race_base(feature_paths, front3f_path)
    context = build_prior_context_by_race(race_source, race_ids, years=args.years)
    runner_features = load_runner_features(feature_paths)
    scored = add_race_quality_scores(universe, context, runner_features)

    summary, yearly, detail = run_backtest(scored, gates, shape_weights, hist_weights, ticket_types)

    score_cols = [
        "race_id",
        "year",
        "pair_key_norm",
        "anchor_no",
        "partner_no",
        "hist_scope",
        "hist_sample_count",
        "hist_avg_front3f_sec",
        "hist_avg_rpci",
        "expected_front3f_sec",
        "expected_rpci",
        "race_quality_label",
        "race_quality_fast_need_score",
        "race_quality_slow_need_score",
        "race_quality_pair_fit_score",
        "race_quality_pair_min_score",
        "wide_hit",
        "umaren_hit",
        "wide_pay",
        "umaren_pay",
    ]
    scored[[c for c in score_cols if c in scored.columns]].to_csv(out_dir / "historical_quality_pair_scores.csv", index=False, encoding="utf-8-sig")
    context.to_csv(out_dir / "leak_safe_race_context.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "historical_quality_pair_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "historical_quality_pair_yearly.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(out_dir / "historical_quality_pair_detail.csv", index=False, encoding="utf-8-sig")

    best = summary.head(30).replace({np.nan: None}).to_dict(orient="records") if not summary.empty else []
    report = {
        "universe": str(universe_path.relative_to(ROOT)),
        "rows": int(len(scored)),
        "races": int(scored["race_id"].nunique()),
        "context_races": int(context["race_id"].nunique()) if not context.empty else 0,
        "context_ready_rate_pct": round(float(scored["race_quality_context_ready"].mean() * 100), 1) if "race_quality_context_ready" in scored else 0.0,
        "years": args.years,
        "top_policies": best,
        "note": "Leak-safe historical same-condition race-quality overlay. Each race uses only prior races for baselines. This is a shadow backtest; production adoption should prefer stable yearly deltas, not only top ROI.",
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
