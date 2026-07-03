from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_retro_lap_adversity import (  # noqa: E402
    merge_pair_features,
    metric_row,
    ncol,
    select_top_per_race,
)
from scripts.evaluate_shape_adjusted_pair_selection import (  # noqa: E402
    DEFAULT_RACE_SHAPE,
    DEFAULT_UNIVERSE,
    add_shape_scores,
    gate_mask,
    load_universe,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs" / "analysis" / "retro_lap_adversity_v1"
OUT_DIR = SOURCE_DIR / "breakdown_v1"


def read_csv_any(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def bucket_distance(distance: pd.Series) -> pd.Series:
    d = pd.to_numeric(distance, errors="coerce")
    return pd.cut(
        d,
        bins=[0, 1199, 1399, 1599, 1799, 1999, 2199, 9999],
        labels=["<=1199", "1200-1399", "1400-1599", "1600-1799", "1800-1999", "2000-2199", "2200+"],
        include_lowest=True,
    ).astype("string").fillna("unknown")


def bucket_pop(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        x,
        bins=[0, 1, 3, 5, 8, 12, 99],
        labels=["1人気", "2-3人気", "4-5人気", "6-8人気", "9-12人気", "13人気+"],
        include_lowest=True,
    ).astype("string").fillna("unknown")


def bucket_odds(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        x,
        bins=[0, 2, 4, 7, 12, 25, 60, 999],
        labels=["<=2", "2-4", "4-7", "7-12", "12-25", "25-60", "60+"],
        include_lowest=True,
    ).astype("string").fillna("unknown")


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": 0.0,
            "hit_rate_pct": 0.0,
            "result_ready_rate_pct": 0.0,
            "roi_ex_top1_pct": 0.0,
            "top_return_share_pct": 0.0,
        }
    stake = float(frame["stake_yen"].sum())
    ret = float(frame["return_yen"].sum())
    top = float(frame["return_yen"].max())
    ex = frame.drop(frame["return_yen"].idxmax()) if top > 0 else frame.iloc[0:0]
    ex_stake = float(ex["stake_yen"].sum()) if not ex.empty else 0.0
    ex_ret = float(ex["return_yen"].sum()) if not ex.empty else 0.0
    return {
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi_pct": round(ret / stake * 100, 1) if stake else 0.0,
        "hit_rate_pct": round(float(frame["hit"].mean() * 100), 1),
        "result_ready_rate_pct": round(float(frame["result_ready"].mean() * 100), 1),
        "roi_ex_top1_pct": round(ex_ret / ex_stake * 100, 1) if ex_stake else 0.0,
        "top_return_share_pct": round(top / ret * 100, 1) if ret else 0.0,
        "avg_anchor_pop": round(float(pd.to_numeric(frame.get("anchor_pop"), errors="coerce").mean()), 2),
        "avg_partner_pop": round(float(pd.to_numeric(frame.get("partner_pop"), errors="coerce").mean()), 2),
        "avg_anchor_odds": round(float(pd.to_numeric(frame.get("anchor_odds"), errors="coerce").mean()), 2),
        "avg_partner_odds": round(float(pd.to_numeric(frame.get("partner_odds"), errors="coerce").mean()), 2),
        "avg_retro_fit": round(float(ncol(frame, "retro_lap_pair_fit_score", 0.0).mean()), 3),
        "avg_retro_risk": round(float(ncol(frame, "retro_lap_pair_risk_score", 0.0).mean()), 3),
    }


def grouped_metrics(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, dropna=False):
        row = metrics(group)
        row["dimension"] = column
        row["bucket"] = "missing" if pd.isna(value) else str(value)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["tickets", "roi_pct"], ascending=[False, False])
    return out


def add_race_metadata(selected: pd.DataFrame, source_dir: Path) -> pd.DataFrame:
    run = read_csv_any(source_dir / "runner_retro_lap_run_scores.csv", dtype={"race_id": str})
    run["race_id"] = run["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    keep = [
        "race_id",
        "race_date",
        "場所",
        "クラス名",
        "芝・ダ",
        "距離",
        "馬場状態",
        "PCI",
        "PCI3",
        "RPCI",
        "Ave-3F",
        "retro_lap_regime",
    ]
    race = (
        run[[c for c in keep if c in run.columns]]
        .drop_duplicates("race_id", keep="first")
        .rename(
            columns={
                "race_date": "race_date",
                "場所": "venue",
                "クラス名": "class_name",
                "芝・ダ": "surface",
                "距離": "distance",
                "馬場状態": "going",
                "retro_lap_regime": "actual_lap_regime",
            }
        )
    )
    out = selected.merge(race, on="race_id", how="left")
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")
    out["month"] = out["race_date"].dt.month.fillna(0).astype(int)
    out["distance_bucket"] = bucket_distance(out["distance"])
    out["anchor_pop_bucket"] = bucket_pop(out["anchor_pop"])
    out["partner_pop_bucket"] = bucket_pop(out["partner_pop"])
    out["anchor_odds_bucket"] = bucket_odds(out["anchor_odds"])
    out["partner_odds_bucket"] = bucket_odds(out["partner_odds"])
    out["odds_geom_bucket"] = bucket_odds(np.sqrt(pd.to_numeric(out["anchor_odds"], errors="coerce") * pd.to_numeric(out["partner_odds"], errors="coerce")))
    out["pair_style"] = out.get("anchor_style_bucket", "unknown").astype(str) + " x " + out.get("partner_style_bucket", "unknown").astype(str)
    out["race_quality"] = np.select(
        [
            pd.to_numeric(out.get("RPCI"), errors="coerce").lt(47.5),
            pd.to_numeric(out.get("RPCI"), errors="coerce").gt(53.0),
            (pd.to_numeric(out.get("PCI3"), errors="coerce") - pd.to_numeric(out.get("RPCI"), errors="coerce")).gt(3.0),
        ],
        ["前傾/消耗", "スロー/瞬発", "ロンスパ寄り"],
        default="中立",
    )
    out["result_ready"] = pd.to_numeric(out["umaren_pay"], errors="coerce").notna()
    return out


def build_selected(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = read_csv_any(source_dir / "runner_retro_lap_prior_features.csv", dtype={"race_id": str})
    universe = add_shape_scores(load_universe(DEFAULT_UNIVERSE, DEFAULT_RACE_SHAPE))
    scored = merge_pair_features(universe, prior)

    valid = scored[ncol(scored, "retro_lap_pair_evidence_ready", 0.0).gt(0)].copy()
    risk_q60 = float(valid["retro_lap_pair_risk_score"].quantile(0.60))

    mask = gate_mask(scored, "price_sane_strong") & ncol(scored, "retro_lap_pair_risk_score", 0.0).le(risk_q60)
    segment = scored[mask].copy()
    segment["segment_base_score"] = (
        0.15 * segment["shape_base_rank_score"]
        + 0.85 * segment["shape_pair_fit_score"]
        + 0.06 * segment["shape_value_score"]
        - 0.12 * segment["shape_pair_risk_score"]
    )
    selected = select_top_per_race(
        segment,
        "segment_base_score",
        "umaren",
        "umaren_price_sane_strong_retro_low_risk_bottom60",
        "price_sane_strong",
    )
    selected = add_race_metadata(selected, source_dir)

    baseline_pool = scored[gate_mask(scored, "price_sane_strong")].copy()
    baseline_pool["baseline_score"] = (
        0.15 * baseline_pool["shape_base_rank_score"]
        + 0.85 * baseline_pool["shape_pair_fit_score"]
        + 0.06 * baseline_pool["shape_value_score"]
        - 0.12 * baseline_pool["shape_pair_risk_score"]
    )
    baseline = select_top_per_race(
        baseline_pool,
        "baseline_score",
        "umaren",
        "umaren_price_sane_strong_all",
        "price_sane_strong",
    )
    baseline = add_race_metadata(baseline, source_dir)
    return selected, baseline


def write_breakdowns(selected: pd.DataFrame, baseline: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "selected_low_risk_umaren.csv", index=False, encoding="utf-8-sig")
    baseline.to_csv(out_dir / "selected_baseline_price_sane_umaren.csv", index=False, encoding="utf-8-sig")

    dimensions = [
        "year",
        "venue",
        "surface",
        "class_name",
        "going",
        "distance_bucket",
        "month",
        "queue_shape_label",
        "actual_shape",
        "actual_lap_regime",
        "race_quality",
        "anchor_style_bucket",
        "partner_style_bucket",
        "pair_style",
        "anchor_pop_bucket",
        "partner_pop_bucket",
        "anchor_odds_bucket",
        "partner_odds_bucket",
        "odds_geom_bucket",
    ]
    all_rows: list[pd.DataFrame] = []
    for dim in dimensions:
        if dim not in selected.columns:
            continue
        g = grouped_metrics(selected, dim)
        if dim == "year":
            g = g.rename(columns={"bucket": "year_bucket"})
        all_rows.append(g)
        g.to_csv(out_dir / f"breakdown_by_{dim}.csv", index=False, encoding="utf-8-sig")
    breakdown = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    breakdown.to_csv(out_dir / "breakdown_all_dimensions.csv", index=False, encoding="utf-8-sig")

    compare_rows: list[dict[str, Any]] = []
    for name, frame in [("baseline", baseline), ("retro_low_risk", selected)]:
        total = metrics(frame)
        total["set"] = name
        total["split"] = "all"
        compare_rows.append(total)
        for year, group in frame.groupby("year"):
            row = metrics(group)
            row["set"] = name
            row["split"] = f"year={int(year)}"
            compare_rows.append(row)
    compare = pd.DataFrame(compare_rows)
    compare.to_csv(out_dir / "baseline_vs_retro_low_risk.csv", index=False, encoding="utf-8-sig")

    selected_2026 = selected[selected["year"].eq(2026)].copy()
    selected_2026.to_csv(out_dir / "selected_low_risk_2026.csv", index=False, encoding="utf-8-sig")
    missing = selected_2026[~selected_2026["result_ready"]].copy()
    missing.to_csv(out_dir / "selected_low_risk_2026_result_missing.csv", index=False, encoding="utf-8-sig")

    worst_dims: dict[str, list[dict[str, Any]]] = {}
    for dim in dimensions:
        if dim not in selected.columns:
            continue
        g = grouped_metrics(selected[selected["year"].eq(2026)], dim)
        if g.empty:
            continue
        worst_dims[dim] = g.sort_values(["profit_yen", "tickets"], ascending=[True, False]).head(10).replace({np.nan: None}).to_dict(orient="records")

    report = {
        "selected_total": metrics(selected),
        "baseline_total": metrics(baseline),
        "selected_by_year": compare[compare["set"].eq("retro_low_risk") & compare["split"].str.startswith("year=")].replace({np.nan: None}).to_dict(orient="records"),
        "baseline_by_year": compare[compare["set"].eq("baseline") & compare["split"].str.startswith("year=")].replace({np.nan: None}).to_dict(orient="records"),
        "result_missing_2026": int(len(missing)),
        "selected_2026_tickets": int(len(selected_2026)),
        "worst_2026_dimensions": worst_dims,
        "note": "Breakdown for umaren price_sane_strong + retro_lap low-risk bottom60. This reconstructs the same shadow segment and compares it with the price_sane_strong baseline.",
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Break down retro lap low-risk segment by race/market/context dimensions.")
    parser.add_argument("--source-dir", default=str(SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    selected, baseline = build_selected(source_dir)
    report = write_breakdowns(selected, baseline, out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
