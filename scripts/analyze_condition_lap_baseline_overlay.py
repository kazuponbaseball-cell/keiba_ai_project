from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_historical_condition_lap_context import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_FRONT3F,
    estimate_pass_1000m_sec,
    load_feature_races,
    load_front3f_races,
    project_path,
)


DEFAULT_ALL_TICKETS = (
    ROOT / "outputs" / "analysis" / "mcs_pbo_runtime_overlay_v4_operational_gates" / "recommended_all_tickets.csv"
)
DEFAULT_RUNTIME_TICKETS = (
    ROOT / "outputs" / "analysis" / "mcs_pbo_runtime_overlay_v4_operational_gates" / "recommended_runtime_tickets.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "analysis" / "condition_lap_baseline_overlay_v1"


METRICS = {
    "winning_time_sec": "prior_winning_time_sec",
    "race_first3f_sec": "prior_front3f_sec",
    "race_last3f_sec": "prior_last3f_sec",
    "pass_1000m_sec": "prior_1000m_sec",
    "rpci": "prior_rpci",
    "pci3": "prior_pci3",
}


@dataclass(frozen=True)
class ScopeDef:
    name: str
    keys: tuple[str, ...]
    min_count: int


SCOPES = [
    ScopeDef("venue_surface_distance_class_going", ("venue", "surface", "distance", "class_group", "going"), 3),
    ScopeDef("venue_surface_distance_class", ("venue", "surface", "distance", "class_group"), 5),
    ScopeDef("venue_surface_distance_going", ("venue", "surface", "distance", "going"), 5),
    ScopeDef("venue_surface_distance", ("venue", "surface", "distance"), 8),
    ScopeDef("surface_distance_class", ("surface", "distance", "class_group"), 12),
    ScopeDef("surface_distance", ("surface", "distance"), 20),
]


def read_csv_any(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = df[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def clean_race_id(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(16)


def load_races(feature_paths: list[Path], front3f_path: Path) -> pd.DataFrame:
    races = load_feature_races(feature_paths)
    front3f = load_front3f_races(front3f_path)
    if not front3f.empty and not races.empty:
        races = races.merge(front3f, on="race_id", how="left")
        races["winning_time_sec"] = races["race_total_time_sec"].fillna(races["winning_time_sec"])
    races["pass_1000m_sec"] = estimate_pass_1000m_sec(races)
    races["race_id"] = clean_race_id(races["race_id"])
    races = races.sort_values(["race_date", "race_id"], kind="mergesort").drop_duplicates("race_id", keep="last")
    for col in ["distance", *METRICS.keys()]:
        if col in races.columns:
            races[col] = pd.to_numeric(races[col], errors="coerce")
    return races


def add_scope_priors(races: pd.DataFrame, scope: ScopeDef) -> pd.DataFrame:
    out = races[["race_id", *scope.keys]].copy()
    grouped = races.groupby(list(scope.keys), dropna=False, sort=False)
    out[f"{scope.name}_count"] = grouped.cumcount()
    for src, dst in METRICS.items():
        if src not in races.columns:
            continue
        out[f"{scope.name}_{dst}"] = grouped[src].transform(
            lambda s: s.shift().expanding(min_periods=scope.min_count).mean()
        )
    return out[["race_id", *[c for c in out.columns if c.startswith(f"{scope.name}_")]]]


def build_time_safe_condition_priors(races: pd.DataFrame) -> pd.DataFrame:
    priors = races[
        [
            "race_id",
            "race_date",
            "venue",
            "surface",
            "distance",
            "class_group",
            "going",
            *[c for c in METRICS if c in races.columns],
        ]
    ].copy()
    for scope in SCOPES:
        scoped = add_scope_priors(races, scope)
        priors = priors.merge(scoped, on="race_id", how="left")

    for dst in METRICS.values():
        priors[dst] = np.nan
    priors["prior_scope"] = ""
    priors["prior_sample_count"] = 0.0
    for scope in SCOPES:
        count_col = f"{scope.name}_count"
        fillable = priors["prior_scope"].eq("") & pd.to_numeric(priors[count_col], errors="coerce").ge(scope.min_count)
        metric_cols = [f"{scope.name}_{dst}" for dst in METRICS.values() if f"{scope.name}_{dst}" in priors.columns]
        if metric_cols:
            usable = fillable & priors[metric_cols].notna().any(axis=1)
        else:
            usable = fillable
        if not usable.any():
            continue
        priors.loc[usable, "prior_scope"] = scope.name
        priors.loc[usable, "prior_sample_count"] = pd.to_numeric(priors.loc[usable, count_col], errors="coerce")
        for dst in METRICS.values():
            src = f"{scope.name}_{dst}"
            if src in priors.columns:
                priors.loc[usable, dst] = priors.loc[usable, src]

    # Last-resort time-safe global priors. These are intentionally weak, but keep diagnostics available.
    for raw, dst in METRICS.items():
        if raw in priors.columns:
            global_prior = priors[raw].shift().expanding(min_periods=50).mean()
            missing = priors[dst].isna()
            priors.loc[missing, dst] = global_prior.loc[missing]
    priors.loc[priors["prior_scope"].eq(""), "prior_scope"] = "global_time_safe"
    keep = [
        "race_id",
        "race_date",
        "venue",
        "surface",
        "distance",
        "class_group",
        "going",
        "prior_scope",
        "prior_sample_count",
        *METRICS.values(),
        *[c for c in METRICS if c in priors.columns],
    ]
    return priors[keep]


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = pd.to_numeric(profit, errors="coerce").fillna(0.0).cumsum()
    dd = curve.cummax() - curve
    return float(dd.max())


def roi_without_top_returns(df: pd.DataFrame, n: int) -> float:
    if df.empty:
        return 0.0
    ordered = df.assign(_ret=pd.to_numeric(df["return_yen"], errors="coerce").fillna(0.0)).sort_values(
        "_ret", ascending=False
    )
    rest = ordered.iloc[n:].copy()
    stake = pd.to_numeric(rest["stake_yen"], errors="coerce").fillna(100.0)
    ret = pd.to_numeric(rest["return_yen"], errors="coerce").fillna(0.0)
    return float(ret.sum() / stake.sum() * 100.0) if stake.sum() else 0.0


def metrics(df: pd.DataFrame, label: str) -> dict[str, Any]:
    stake = pd.to_numeric(df.get("stake_yen"), errors="coerce").fillna(100.0)
    ret = pd.to_numeric(df.get("return_yen"), errors="coerce").fillna(0.0)
    hit = df.get("hit", ret.gt(0)).astype(str).str.lower().isin(["true", "1", "yes"]) | ret.gt(0)
    profit = ret - stake
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    return {
        "policy": label,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()) if "race_id" in df.columns else 0,
        "stake_yen": round(stake_sum, 1),
        "return_yen": round(ret_sum, 1),
        "profit_yen": round(float(profit.sum()), 1),
        "roi_pct": round(ret_sum / stake_sum * 100.0, 1) if stake_sum else 0.0,
        "hit_rate_pct": round(float(hit.mean() * 100.0), 1) if len(df) else 0.0,
        "roi_ex_top5_pct": round(roi_without_top_returns(df, 5), 1),
        "roi_ex_top10_pct": round(roi_without_top_returns(df, 10), 1),
        "max_drawdown_yen": round(max_drawdown(profit), 1),
    }


def enrich_tickets(tickets: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = clean_race_id(out["race_id"])
    pri = priors.copy()
    pri["race_id"] = clean_race_id(pri["race_id"])
    keep = [
        "race_id",
        "prior_scope",
        "prior_sample_count",
        "prior_winning_time_sec",
        "prior_front3f_sec",
        "prior_last3f_sec",
        "prior_1000m_sec",
        "prior_rpci",
        "prior_pci3",
        "winning_time_sec",
        "race_first3f_sec",
        "race_last3f_sec",
        "pass_1000m_sec",
        "rpci",
        "pci3",
    ]
    out = out.merge(pri[[c for c in keep if c in pri.columns]], on="race_id", how="left")

    pressure = num(out, "race_front_pressure", np.nan).fillna(num(out, "race_early_pressure_score", 0.0)).fillna(0.0)
    collapse = num(out, "race_pace_collapse", np.nan).fillna(num(out, "race_pace_collapse_risk", 0.0)).fillna(0.0)
    slow = num(out, "race_slow_risk", np.nan).fillna(num(out, "race_slow_pace_risk", 0.0)).fillna(0.0)

    # Deployable proxy: no post-race lap is used here. Positive = tougher/faster than the same-condition norm.
    out["condition_fast_pressure_proxy"] = (collapse + 0.55 * pressure - 0.65 * slow).astype(float)
    out["condition_slow_proxy"] = (slow - 0.65 * collapse - 0.30 * pressure).astype(float)
    out["expected_rpci_proxy"] = (
        num(out, "prior_rpci", 50.0).fillna(50.0)
        + 3.1 * slow
        - 3.2 * collapse
        - 0.9 * pressure
    ).clip(42.0, 58.0)
    out["expected_rpci_delta_from_condition"] = out["expected_rpci_proxy"] - num(out, "prior_rpci", 50.0).fillna(50.0)
    out["condition_prior_confident"] = pd.to_numeric(out["prior_sample_count"], errors="coerce").fillna(0.0).ge(8)

    # Diagnostic only: these are not deployable before the race.
    actual_rpci = num(out, "RPCI", np.nan).fillna(num(out, "rpci", np.nan))
    actual_pci3 = num(out, "PCI3", np.nan).fillna(num(out, "pci3", np.nan))
    out["actual_rpci_minus_condition"] = actual_rpci - num(out, "prior_rpci", np.nan)
    out["actual_pci3_minus_condition"] = actual_pci3 - num(out, "prior_pci3", np.nan)
    if "race_first3f_sec" in out.columns:
        out["actual_front3f_minus_condition"] = num(out, "race_first3f_sec", np.nan) - num(out, "prior_front3f_sec", np.nan)
    return out


def train_quantiles(df: pd.DataFrame) -> dict[str, float]:
    train = df[pd.to_numeric(df.get("year"), errors="coerce").fillna(pd.to_numeric(df.get("test_year"), errors="coerce")).lt(2026)]
    if train.empty:
        train = df
    out: dict[str, float] = {}
    for col in [
        "condition_fast_pressure_proxy",
        "condition_slow_proxy",
        "expected_rpci_delta_from_condition",
        "prior_rpci",
        "prior_pci3",
        "prior_sample_count",
    ]:
        s = pd.to_numeric(train[col], errors="coerce").dropna()
        if s.empty:
            continue
        for q in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            out[f"{col}_q{int(q * 100)}"] = float(s.quantile(q))
    fast_pos = pd.to_numeric(train["condition_fast_pressure_proxy"], errors="coerce")
    fast_pos = fast_pos[fast_pos.gt(0)]
    if not fast_pos.empty:
        for q in (0.4, 0.5, 0.6, 0.7, 0.8):
            out[f"condition_fast_pressure_positive_q{int(q * 100)}"] = float(fast_pos.quantile(q))
    return out


def policy_masks(df: pd.DataFrame, q: dict[str, float]) -> list[tuple[str, pd.Series]]:
    idx = df.index
    fast = pd.to_numeric(df["condition_fast_pressure_proxy"], errors="coerce").fillna(0.0)
    slow = pd.to_numeric(df["condition_slow_proxy"], errors="coerce").fillna(0.0)
    delta = pd.to_numeric(df["expected_rpci_delta_from_condition"], errors="coerce").fillna(0.0)
    prior_rpci = pd.to_numeric(df["prior_rpci"], errors="coerce").fillna(50.0)
    confident = df["condition_prior_confident"].fillna(False).astype(bool)
    front_pair = num(df, "front_pair_max", np.nan).fillna(num(df, "projected_front5_prob", 0.0)).fillna(0.0)
    front_both = num(df, "front_pair_min", 0.0).fillna(0.0)
    return [
        ("base", pd.Series(True, index=idx)),
        ("condition_prior_confident", confident),
        ("avoid_extreme_fast_q80", confident & fast.lt(q.get("condition_fast_pressure_proxy_q80", 999.0))),
        ("avoid_extreme_slow_q80", confident & slow.lt(q.get("condition_slow_proxy_q80", 999.0))),
        ("condition_middle_delta", confident & delta.between(q.get("expected_rpci_delta_from_condition_q30", -999.0), q.get("expected_rpci_delta_from_condition_q70", 999.0))),
        ("condition_fast_q70", confident & fast.ge(q.get("condition_fast_pressure_proxy_q70", 999.0))),
        ("condition_slow_q70", confident & slow.ge(q.get("condition_slow_proxy_q70", 999.0))),
        ("front_any_fast_q70", confident & front_pair.ge(0.55) & fast.ge(q.get("condition_fast_pressure_proxy_q70", 999.0))),
        ("front_both_fast_q70", confident & front_both.ge(0.45) & fast.ge(q.get("condition_fast_pressure_proxy_q70", 999.0))),
        ("condition_fast_positive_q60", confident & fast.ge(q.get("condition_fast_pressure_positive_q60", 999.0))),
        (
            "front_any_fast_positive_q60",
            confident & front_pair.ge(0.55) & fast.ge(q.get("condition_fast_pressure_positive_q60", 999.0)),
        ),
        ("front_any_slow_or_middle", confident & front_pair.ge(0.55) & fast.lt(q.get("condition_fast_pressure_proxy_q70", 999.0))),
        ("front_any_slow_condition", confident & front_pair.ge(0.55) & slow.ge(q.get("condition_slow_proxy_q60", 999.0))),
        ("fast_on_slow_prior", confident & prior_rpci.ge(q.get("prior_rpci_q60", 999.0)) & fast.ge(q.get("condition_fast_pressure_proxy_q70", 999.0))),
        ("fast_on_fast_prior", confident & prior_rpci.le(q.get("prior_rpci_q40", -999.0)) & fast.ge(q.get("condition_fast_pressure_proxy_q70", 999.0))),
        (
            "fast_positive_on_slow_prior",
            confident & prior_rpci.ge(q.get("prior_rpci_q60", 999.0)) & fast.ge(q.get("condition_fast_pressure_positive_q60", 999.0)),
        ),
        (
            "fast_positive_on_fast_prior",
            confident & prior_rpci.le(q.get("prior_rpci_q40", -999.0)) & fast.ge(q.get("condition_fast_pressure_positive_q60", 999.0)),
        ),
    ]


def evaluate_policies(df: pd.DataFrame, q: dict[str, float], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    for name, mask in policy_masks(df, q):
        part = df[mask].copy()
        rows.append({"dataset": prefix, **metrics(part, name)})
        year_col = pd.to_numeric(part.get("year"), errors="coerce").fillna(pd.to_numeric(part.get("test_year"), errors="coerce"))
        for year, yp in part.groupby(year_col, dropna=False):
            yearly_rows.append({"dataset": prefix, "year": int(year) if pd.notna(year) else None, **metrics(yp, name)})
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows)


def diagnostic_segments(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = df.copy()
    for col in ["actual_rpci_minus_condition", "actual_pci3_minus_condition", "actual_front3f_minus_condition"]:
        if col not in work.columns:
            continue
        valid = pd.to_numeric(work[col], errors="coerce").dropna()
        if len(valid) < 20:
            continue
        try:
            work[f"{col}_band"] = pd.qcut(valid.rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])
        except ValueError:
            continue
        band_col = f"{col}_band"
        for band, part in work.loc[valid.index].groupby(band_col, observed=False):
            rows.append({"diagnostic": col, "band": str(band), **metrics(part, "postrace_delta_band")})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate time-safe condition-lap baselines as deployable overlays.")
    parser.add_argument("--all-tickets", type=Path, default=DEFAULT_ALL_TICKETS)
    parser.add_argument("--runtime-tickets", type=Path, default=DEFAULT_RUNTIME_TICKETS)
    parser.add_argument("--front3f-csv", type=Path, default=project_path(DEFAULT_FRONT3F))
    parser.add_argument("--feature-csv", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_paths = [project_path(p) for p in (args.feature_csv or DEFAULT_FEATURES)]
    races = load_races(feature_paths, args.front3f_csv if args.front3f_csv.is_absolute() else ROOT / args.front3f_csv)
    priors = build_time_safe_condition_priors(races)
    priors.to_csv(out_dir / "time_safe_condition_lap_priors.csv", index=False, encoding="utf-8-sig")

    all_tickets = read_csv_any(args.all_tickets if args.all_tickets.is_absolute() else ROOT / args.all_tickets)
    runtime_tickets = read_csv_any(args.runtime_tickets if args.runtime_tickets.is_absolute() else ROOT / args.runtime_tickets)
    enriched_all = enrich_tickets(all_tickets, priors)
    enriched_runtime = enrich_tickets(runtime_tickets, priors)
    q = train_quantiles(enriched_all)

    all_metrics, all_yearly = evaluate_policies(enriched_all, q, "all_candidates")
    runtime_metrics, runtime_yearly = evaluate_policies(enriched_runtime, q, "runtime_selected")
    policy_metrics = pd.concat([all_metrics, runtime_metrics], ignore_index=True, sort=False)
    policy_yearly = pd.concat([all_yearly, runtime_yearly], ignore_index=True, sort=False)
    diagnostics = diagnostic_segments(enriched_all)

    enriched_all.to_csv(out_dir / "all_tickets_with_condition_lap_baseline.csv", index=False, encoding="utf-8-sig")
    enriched_runtime.to_csv(out_dir / "runtime_tickets_with_condition_lap_baseline.csv", index=False, encoding="utf-8-sig")
    policy_metrics.to_csv(out_dir / "policy_metrics.csv", index=False, encoding="utf-8-sig")
    policy_yearly.to_csv(out_dir / "policy_yearly.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(out_dir / "diagnostic_postrace_delta_segments.csv", index=False, encoding="utf-8-sig")

    runtime_view = policy_metrics[policy_metrics["dataset"].eq("runtime_selected")].copy()
    runtime_view = runtime_view.sort_values(["roi_ex_top10_pct", "roi_pct"], ascending=False)
    summary = {
        "output_dir": str(out_dir),
        "source_races": int(races["race_id"].nunique()),
        "all_ticket_rows": int(len(enriched_all)),
        "runtime_ticket_rows": int(len(enriched_runtime)),
        "prior_scope_counts": priors["prior_scope"].value_counts(dropna=False).to_dict(),
        "thresholds": q,
        "top_runtime_policies": runtime_view.head(8).to_dict("records"),
        "notes": [
            "Priors are time-safe: only races before each target race are used.",
            "Policy masks use deployable proxies based on pre-race pressure/slow/collapse signals plus same-condition priors.",
            "actual_*_minus_condition columns are diagnostic only and must not be used for live BUY decisions.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
