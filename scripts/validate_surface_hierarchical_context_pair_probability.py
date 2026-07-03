from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.strict_pair_probability_roi_protocol import (  # noqa: E402
    build_raw_probability_features,
    clip01,
    metrics,
    norm01,
    num,
    train_val_holdout,
    walkforward,
)


PAIR_UNIVERSE = Path("outputs/analysis/horse_lap_decomp_pair_probability_v1/pair_universe_with_lap_decomp_features.csv")
RACE_CONTEXT = Path("outputs/analysis/high_pressure_front_survival_context_v1/race_high_pressure_front_survival_context.csv")
OUT = Path("outputs/analysis/surface_hierarchical_context_pair_probability_v1")


VARIANTS = {
    "baseline": {"support": 0.000, "caution": 0.000, "market": 0.000},
    "hier_context_light": {"support": 0.012, "caution": 0.012, "market": 0.006},
    "hier_context_mid": {"support": 0.024, "caution": 0.024, "market": 0.010},
    "hier_context_strong": {"support": 0.040, "caution": 0.040, "market": 0.016},
    "hier_support_only": {"support": 0.018, "caution": 0.000, "market": 0.008},
    "hier_guard_light": {"support": 0.000, "caution": 0.018, "market": 0.000},
    "hier_guard_mid": {"support": 0.000, "caution": 0.032, "market": 0.000},
}


CONTEXT_SPECS = [
    ("surface", ["surface_clean"], 700.0, 0.10),
    ("surface_distance", ["surface_clean", "surface_distance_bin"], 450.0, 0.16),
    ("surface_venue", ["surface_clean", "venue"], 450.0, 0.16),
    ("surface_venue_distance", ["surface_clean", "venue", "surface_distance_bin"], 300.0, 0.24),
    ("surface_venue_distance_going", ["surface_clean", "venue", "surface_distance_bin", "going"], 260.0, 0.18),
    ("surface_distance_class", ["surface_clean", "surface_distance_bin", "surface_class_tier"], 300.0, 0.16),
]


class Stat:
    __slots__ = ("weight", "ret_sum", "hit_sum")

    def __init__(self) -> None:
        self.weight = 0.0
        self.ret_sum = 0.0
        self.hit_sum = 0.0

    def update(self, weight: float, ret: float, hit: float) -> None:
        if weight <= 0 or not np.isfinite(weight):
            return
        self.weight += float(weight)
        self.ret_sum += float(weight * ret)
        self.hit_sum += float(weight * hit)

    def mean_ret(self, fallback: float, smoothing: float) -> float:
        return float((self.ret_sum + fallback * smoothing) / (self.weight + smoothing)) if self.weight + smoothing else fallback

    def mean_hit(self, fallback: float, smoothing: float) -> float:
        return float((self.hit_sum + fallback * smoothing) / (self.weight + smoothing)) if self.weight + smoothing else fallback


def read_pair_universe() -> pd.DataFrame:
    if not PAIR_UNIVERSE.exists():
        raise FileNotFoundError(f"missing pair universe: {PAIR_UNIVERSE}")
    df = pd.read_csv(PAIR_UNIVERSE, dtype={"race_id": str}, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    if "year" not in df.columns:
        df["year"] = df["race_id"].str[:4].astype(int)
    return df


def attach_context(df: pd.DataFrame) -> pd.DataFrame:
    if not RACE_CONTEXT.exists():
        raise FileNotFoundError(f"missing race context: {RACE_CONTEXT}")
    ctx = pd.read_csv(
        RACE_CONTEXT,
        dtype={"race_id": str},
        usecols=lambda c: c in {"race_id", "surface", "venue_code", "distance_bin", "class_tier", "going"},
        low_memory=False,
    )
    ctx["race_id"] = ctx["race_id"].astype(str)
    ctx = ctx.drop_duplicates("race_id", keep="last").rename(
        columns={
            "surface": "surface_clean",
            "distance_bin": "surface_distance_bin",
            "class_tier": "surface_class_tier",
            "going": "surface_going_raw",
        }
    )
    out = df.merge(ctx, on="race_id", how="left")
    for col in ["surface_clean", "surface_distance_bin", "surface_class_tier", "going", "venue"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    out = out[out["surface_clean"].isin(["芝", "ダ"])].copy()
    out["race_date"] = pd.to_numeric(out["race_id"].str[:8], errors="coerce").fillna(0).astype(int)
    return out


def candidate_quality(df: pd.DataFrame) -> pd.Series:
    return (
        0.34 * clip01(df.get("pair_quinella_score", pd.Series(0.0, index=df.index)))
        + 0.20 * clip01(df.get("pair_score", pd.Series(0.0, index=df.index)))
        + 0.18 * clip01(df.get("market_overlay_score", pd.Series(0.0, index=df.index)))
        + 0.14 * clip01(df.get("late_value_survives_score", pd.Series(0.0, index=df.index)))
        + 0.14 * clip01(df.get("projected_front5_prob", pd.Series(0.0, index=df.index)))
    ).clip(0.0, 1.0)


def add_context_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    wide_hit = out["wide_hit"].astype(bool)
    umaren_hit = out["umaren_hit"].astype(bool)
    wide_ret = pd.to_numeric(out["wide_pay"], errors="coerce").where(wide_hit, 0.0).fillna(0.0) / 100.0
    umaren_ret = pd.to_numeric(out["umaren_pay"], errors="coerce").where(umaren_hit, 0.0).fillna(0.0) / 100.0
    # Capped target is used only to learn broad context priors. It intentionally avoids
    # letting one exceptional payout define an entire surface/venue bucket.
    out["_hier_context_ret"] = (0.67 * wide_ret + 0.33 * umaren_ret).clip(0.0, 8.0)
    out["_hier_context_hit"] = (wide_hit | umaren_hit).astype(float)
    out["_hier_context_weight"] = (0.25 + 0.75 * candidate_quality(out)).clip(0.10, 1.0)
    return out


def _key(row: pd.Series, cols: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(c, "")) for c in cols)


def add_rolling_hierarchical_context(df: pd.DataFrame) -> pd.DataFrame:
    work = add_context_target_columns(df).sort_values(["race_date", "race_id"], kind="mergesort").copy()
    stats: dict[str, dict[tuple[str, ...], Stat]] = {name: defaultdict(Stat) for name, _, _, _ in CONTEXT_SPECS}
    global_stat = Stat()
    default_ret = float(work["_hier_context_ret"].mean()) if len(work) else 0.75
    default_hit = float(work["_hier_context_hit"].mean()) if len(work) else 0.05

    score_rows: list[pd.DataFrame] = []
    for _, race_group in work.groupby("race_id", sort=False):
        g = race_group.copy()
        global_ret = global_stat.mean_ret(default_ret, 500.0)
        global_hit = global_stat.mean_hit(default_hit, 500.0)
        weighted_edges = []
        support_counts = []
        for name, cols, smoothing, layer_weight in CONTEXT_SPECS:
            layer_stats = stats[name]
            edges = []
            counts = []
            for _, row in g.iterrows():
                st = layer_stats[_key(row, cols)]
                prior_ret = st.mean_ret(global_ret, smoothing)
                prior_hit = st.mean_hit(global_hit, smoothing)
                ret_edge = np.tanh((prior_ret - global_ret) / 0.22)
                hit_edge = np.tanh((prior_hit - global_hit) / 0.035)
                confidence = min(st.weight / (smoothing * 1.6), 1.0)
                edges.append(float((0.72 * ret_edge + 0.28 * hit_edge) * confidence))
                counts.append(float(st.weight))
            edge_series = pd.Series(edges, index=g.index)
            weighted_edges.append(layer_weight * edge_series)
            support_counts.append(pd.Series(counts, index=g.index))
            g[f"hier_{name}_edge"] = edge_series
            g[f"hier_{name}_support_weight"] = support_counts[-1]

        total_edge = sum(weighted_edges)
        total_support = sum(support_counts)
        g["surface_hier_context_edge_score"] = total_edge.clip(-1.0, 1.0)
        g["surface_hier_context_support_score"] = total_edge.clip(lower=0.0, upper=1.0)
        g["surface_hier_context_caution_score"] = (-total_edge).clip(lower=0.0, upper=1.0)
        g["surface_hier_context_support_weight"] = total_support
        g["surface_hier_context_label"] = np.select(
            [
                g["surface_hier_context_edge_score"].ge(0.12),
                g["surface_hier_context_edge_score"].le(-0.12),
                g["surface_hier_context_edge_score"].ge(0.04),
                g["surface_hier_context_edge_score"].le(-0.04),
            ],
            [
                "surface_hier_support",
                "surface_hier_caution",
                "surface_hier_slight_support",
                "surface_hier_slight_caution",
            ],
            default="surface_hier_neutral",
        )
        score_rows.append(g)

        # Update after assigning all rows in this race, so current-race outcomes cannot leak.
        for _, row in race_group.iterrows():
            weight = float(row["_hier_context_weight"])
            ret = float(row["_hier_context_ret"])
            hit = float(row["_hier_context_hit"])
            global_stat.update(weight, ret, hit)
            for name, cols, _, _ in CONTEXT_SPECS:
                stats[name][_key(row, cols)].update(weight, ret, hit)

    out = pd.concat(score_rows, ignore_index=False).sort_index()
    return out.drop(columns=["_hier_context_ret", "_hier_context_hit", "_hier_context_weight"], errors="ignore")


def apply_variant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    params = VARIANTS[name]
    if name == "baseline":
        out["surface_hier_variant"] = name
        out["surface_hier_probability_adjustment"] = 0.0
        return out
    support = clip01(out["surface_hier_context_support_score"])
    caution = clip01(out["surface_hier_context_caution_score"])
    net = params["support"] * support - params["caution"] * caution
    out["pair_quinella_score"] = (num(out["pair_quinella_score"]).fillna(0.0) + net).clip(0.0, 1.0)
    out["pair_score"] = (num(out["pair_score"]).fillna(0.0) + 0.72 * net).clip(0.0, 1.0)
    out["market_overlay_score"] = (
        num(out["market_overlay_score"]).fillna(0.0)
        + params["market"] * (0.75 * support - 0.55 * caution)
    ).clip(0.0, 1.0)
    out["surface_hier_probability_adjustment"] = net
    out["surface_hier_variant"] = name
    return out


def aggregate_tickets(tickets: pd.DataFrame, label: str) -> dict:
    m = metrics(tickets, label)
    if not tickets.empty:
        year_rows = []
        for year, part in tickets.groupby("year", sort=True):
            y = metrics(part, f"{label}_{year}")
            year_rows.append({"year": int(year), "roi": y["roi"], "races": y["races"], "profit_yen": y["profit_yen"]})
        m["min_year_roi"] = min((x["roi"] for x in year_rows), default=0.0)
        m["year_metrics"] = year_rows
    else:
        m["min_year_roi"] = 0.0
        m["year_metrics"] = []
    return m


def run_variant(source: pd.DataFrame, name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = build_raw_probability_features(apply_variant(source, name))
    train_grid, wf_summary, wf_tickets = walkforward(scored)
    holdout = train_val_holdout(scored)
    total = aggregate_tickets(wf_tickets, f"{name}_walkforward_total")
    total["variant"] = name
    return total, train_grid, wf_summary, holdout, wf_tickets


def segment_metrics(tickets: pd.DataFrame, label: str) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for col in [
        "surface_hier_context_label",
        "surface_clean",
        "surface_distance_bin",
        "venue",
        "going",
        "ticket_type",
    ]:
        if col not in tickets.columns:
            continue
        for val, part in tickets.groupby(col, dropna=False):
            stake = float(pd.to_numeric(part.get("stake_yen"), errors="coerce").fillna(0.0).sum())
            ret = float(pd.to_numeric(part.get("return_yen"), errors="coerce").fillna(0.0).sum())
            rows.append(
                {
                    "variant": label,
                    "segment_col": col,
                    "segment_value": str(val),
                    "tickets": int(len(part)),
                    "races": int(part["race_id"].nunique()),
                    "stake_yen": stake,
                    "return_yen": ret,
                    "profit_yen": ret - stake,
                    "roi": ret / stake if stake else 0.0,
                }
            )
    for score_col in [
        "surface_hier_context_edge_score",
        "surface_hier_context_support_score",
        "surface_hier_context_caution_score",
    ]:
        if score_col not in tickets.columns:
            continue
        work = tickets.copy()
        score = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
        if score.nunique() < 4:
            continue
        try:
            work["_bin"] = pd.qcut(score.rank(method="first"), q=4, labels=["q1_low", "q2", "q3", "q4_high"])
        except ValueError:
            continue
        for val, part in work.groupby("_bin", observed=True):
            stake = float(pd.to_numeric(part.get("stake_yen"), errors="coerce").fillna(0.0).sum())
            ret = float(pd.to_numeric(part.get("return_yen"), errors="coerce").fillna(0.0).sum())
            rows.append(
                {
                    "variant": label,
                    "segment_col": f"{score_col}_quartile",
                    "segment_value": str(val),
                    "tickets": int(len(part)),
                    "races": int(part["race_id"].nunique()),
                    "stake_yen": stake,
                    "return_yen": ret,
                    "profit_yen": ret - stake,
                    "roi": ret / stake if stake else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = add_rolling_hierarchical_context(attach_context(read_pair_universe()))
    source.to_csv(OUT / "pair_universe_with_surface_hier_context.csv", index=False, encoding="utf-8-sig")

    summaries: list[dict] = []
    all_years: list[dict] = []
    all_segments: list[pd.DataFrame] = []
    holdouts: list[pd.DataFrame] = []
    best_variant = ""
    best_roi = -1.0

    for variant in VARIANTS:
        total, train_grid, wf_summary, holdout, tickets = run_variant(source, variant)
        summaries.append(total)
        for row in total.get("year_metrics", []):
            row["variant"] = variant
            all_years.append(row)
        train_grid.to_csv(OUT / f"{variant}_walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
        wf_summary.to_csv(OUT / f"{variant}_walkforward_summary.csv", index=False, encoding="utf-8-sig")
        holdout["variant"] = variant
        holdout.to_csv(OUT / f"{variant}_train2024_val2025_hold2026_grid.csv", index=False, encoding="utf-8-sig")
        tickets.to_csv(OUT / f"{variant}_walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
        seg = segment_metrics(tickets, variant)
        if not seg.empty:
            all_segments.append(seg)
        if total["roi"] > best_roi:
            best_roi = float(total["roi"])
            best_variant = variant
        holdouts.append(holdout)

    comparison = pd.DataFrame(summaries).drop(columns=["year_metrics"], errors="ignore")
    comparison = comparison.sort_values(["roi", "profit_yen"], ascending=False)
    comparison.to_csv(OUT / "variant_walkforward_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_years).to_csv(OUT / "variant_yearly_walkforward_summary.csv", index=False, encoding="utf-8-sig")
    if all_segments:
        pd.concat(all_segments, ignore_index=True, sort=False).to_csv(
            OUT / "variant_selected_ticket_segment_metrics.csv", index=False, encoding="utf-8-sig"
        )
    if holdouts:
        hold = pd.concat(holdouts, ignore_index=True, sort=False)
        hold.to_csv(OUT / "variant_holdout_grid_all.csv", index=False, encoding="utf-8-sig")
        hold.sort_values(["dev_score", "val_profit_yen"], ascending=False).groupby("variant", as_index=False).head(5).to_csv(
            OUT / "variant_holdout_top5_by_variant.csv", index=False, encoding="utf-8-sig"
        )

    summary = {
        "output_dir": str(OUT),
        "pair_universe_rows": int(len(source)),
        "race_count": int(source["race_id"].nunique()),
        "variants": list(VARIANTS),
        "best_variant": best_variant,
        "comparison": comparison.to_dict(orient="records"),
        "note": (
            "Time-safe rolling hierarchical context prior. It uses only previous races to shrink context edges across "
            "surface, distance, venue, going, and class. This is intended as a thin correction on the common model, "
            "not a separate turf/dirt model."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Surface Hierarchical Context Pair Probability v1",
                "",
                "Purpose: test a time-safe shrinkage correction for surface x distance x venue x going x class context.",
                "",
                "Key outputs:",
                "- pair_universe_with_surface_hier_context.csv",
                "- variant_walkforward_comparison.csv",
                "- variant_yearly_walkforward_summary.csv",
                "- variant_selected_ticket_segment_metrics.csv",
                "- summary.json",
            ]
        ),
        encoding="utf-8",
    )
    display_cols = [
        "variant",
        "races",
        "tickets",
        "stake_yen",
        "return_yen",
        "profit_yen",
        "roi",
        "race_hit_rate",
        "max_drawdown_yen",
        "min_year_roi",
    ]
    print(comparison[display_cols].to_string(index=False))
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
