from __future__ import annotations

import json
import sys
from pathlib import Path

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
OUT = Path("outputs/analysis/front_load_course_context_pair_probability_v1")


VARIANTS = {
    "baseline": {
        "survival": 0.000,
        "collapse": 0.000,
        "closer": 0.000,
        "market": 0.000,
        "front_prob": 0.000,
    },
    "front_survival_light": {
        "survival": 0.014,
        "collapse": 0.000,
        "closer": 0.000,
        "market": 0.006,
        "front_prob": 0.008,
    },
    "front_survival_mid": {
        "survival": 0.026,
        "collapse": 0.000,
        "closer": 0.000,
        "market": 0.010,
        "front_prob": 0.012,
    },
    "collapse_guard_light": {
        "survival": 0.000,
        "collapse": 0.018,
        "closer": 0.000,
        "market": 0.000,
        "front_prob": 0.010,
    },
    "collapse_guard_mid": {
        "survival": 0.000,
        "collapse": 0.032,
        "closer": 0.000,
        "market": 0.000,
        "front_prob": 0.014,
    },
    "context_switch_light": {
        "survival": 0.014,
        "collapse": 0.018,
        "closer": 0.008,
        "market": 0.006,
        "front_prob": 0.008,
    },
    "context_switch_mid": {
        "survival": 0.026,
        "collapse": 0.032,
        "closer": 0.014,
        "market": 0.010,
        "front_prob": 0.012,
    },
    "context_switch_strong": {
        "survival": 0.044,
        "collapse": 0.050,
        "closer": 0.024,
        "market": 0.016,
        "front_prob": 0.018,
    },
}


def read_pair_universe() -> pd.DataFrame:
    if not PAIR_UNIVERSE.exists():
        raise FileNotFoundError(f"missing pair universe: {PAIR_UNIVERSE}")
    df = pd.read_csv(PAIR_UNIVERSE, dtype={"race_id": str}, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    if "year" not in df.columns:
        df["year"] = df["race_id"].str[:4].astype(int)
    return df


def read_race_context() -> pd.DataFrame:
    if not RACE_CONTEXT.exists():
        raise FileNotFoundError(f"missing high-pressure race context: {RACE_CONTEXT}")
    keep = [
        "race_id",
        "venue_code",
        "surface",
        "distance_bin",
        "class_tier",
        "going",
        "queue_type",
        "pre_front_load_signal",
        "context_survival_prior",
        "context_collapse_prior",
        "front_survival_despite_pressure_score",
        "front_collapse_warning_score",
        "front_context_readability_score",
        "actual_front_survival",
        "actual_front_collapse",
        "actual_fast_or_frontload",
        "actual_top3_front5_share",
    ]
    ctx = pd.read_csv(RACE_CONTEXT, dtype={"race_id": str}, usecols=lambda c: c in keep, low_memory=False)
    ctx["race_id"] = ctx["race_id"].astype(str)
    return ctx.drop_duplicates("race_id", keep="last")


def attach_front_load_context(df: pd.DataFrame) -> pd.DataFrame:
    ctx = read_race_context()
    out = df.merge(ctx, on="race_id", how="left", suffixes=("", "_frontctx"))
    idx = out.index
    for col in [
        "pre_front_load_signal",
        "context_survival_prior",
        "context_collapse_prior",
        "front_survival_despite_pressure_score",
        "front_collapse_warning_score",
        "front_context_readability_score",
        "actual_top3_front5_share",
    ]:
        out[col] = num(out.get(col), idx, 0.0).fillna(0.0)
    for col in ["actual_front_survival", "actual_front_collapse", "actual_fast_or_frontload"]:
        out[col] = num(out.get(col), idx, 0.0).fillna(0.0).clip(0.0, 1.0)
    for col in ["surface", "distance_bin", "class_tier", "going_frontctx", "queue_type"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    front_load = clip01(out["pre_front_load_signal"])
    survival = clip01(out["front_survival_despite_pressure_score"])
    collapse = clip01(out["front_collapse_warning_score"])
    readability = clip01(out["front_context_readability_score"])
    survival_prior = clip01(out["context_survival_prior"])
    collapse_prior = clip01(out["context_collapse_prior"])
    pair_front = clip01(out.get("projected_front5_prob", pd.Series(0.0, index=idx)))
    lap_context = clip01(out.get("lap_decomp_context", pd.Series(0.0, index=idx)))
    lap_caution = clip01(out.get("lap_decomp_caution", pd.Series(0.0, index=idx)))

    anchor_type = out.get("anchor_lap_computed_horse_lap_type", pd.Series("", index=idx)).fillna("").astype(str)
    partner_type = out.get("partner_lap_computed_horse_lap_type", pd.Series("", index=idx)).fillna("").astype(str)
    closer_like = (
        anchor_type.isin(["instant", "long_spurt", "sustain"]) | partner_type.isin(["instant", "long_spurt", "sustain"])
    ).astype(float)
    fast_like = (anchor_type.eq("fast") | partner_type.eq("fast")).astype(float)

    # Pre-race, course-specific read: high front-load does not automatically mean collapse.
    # It is supportive only when historical context says front horses still survive there.
    out["front_load_survival_pair_fit"] = (
        front_load
        * (
            0.38 * survival
            + 0.22 * survival_prior
            + 0.16 * readability
            + 0.14 * pair_front
            + 0.10 * lap_context
        )
        * (0.78 + 0.22 * fast_like)
    ).clip(0.0, 1.0)
    out["front_load_collapse_front_risk"] = (
        front_load
        * (
            0.42 * collapse
            + 0.24 * collapse_prior
            + 0.15 * (1.0 - readability)
            + 0.12 * pair_front
            + 0.07 * lap_caution
        )
    ).clip(0.0, 1.0)
    out["front_load_collapse_closer_fit"] = (
        front_load
        * collapse
        * (
            0.38 * closer_like
            + 0.26 * lap_context
            + 0.20 * readability
            + 0.16 * (1.0 - pair_front)
        )
    ).clip(0.0, 1.0)
    out["front_load_course_net_score"] = (
        out["front_load_survival_pair_fit"]
        - out["front_load_collapse_front_risk"]
        + 0.45 * out["front_load_collapse_closer_fit"]
    ).clip(-1.0, 1.0)
    out["front_load_course_label"] = np.select(
        [
            out["front_load_collapse_front_risk"].ge(0.55),
            out["front_load_survival_pair_fit"].ge(0.62),
            out["front_load_collapse_closer_fit"].ge(0.35),
            out["front_load_course_net_score"].ge(0.10),
            out["front_load_course_net_score"].le(-0.10),
        ],
        [
            "front_load_collapse_risk",
            "front_load_front_survival",
            "front_load_closer_rescue",
            "front_load_slight_front_support",
            "front_load_slight_collapse_caution",
        ],
        default="front_load_neutral",
    )
    return out


def apply_variant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    params = VARIANTS[name]
    if name == "baseline":
        out["front_load_course_variant"] = name
        out["front_load_course_probability_adjustment"] = 0.0
        return out

    survival = clip01(out["front_load_survival_pair_fit"])
    collapse = clip01(out["front_load_collapse_front_risk"])
    closer = clip01(out["front_load_collapse_closer_fit"])
    net = (
        params["survival"] * survival
        - params["collapse"] * collapse
        + params["closer"] * closer
    )

    out["pair_quinella_score"] = (num(out["pair_quinella_score"]).fillna(0.0) + net).clip(0.0, 1.0)
    out["pair_score"] = (num(out["pair_score"]).fillna(0.0) + 0.72 * net).clip(0.0, 1.0)
    out["market_overlay_score"] = (
        num(out["market_overlay_score"]).fillna(0.0)
        + params["market"] * (0.76 * survival - 0.55 * collapse + 0.45 * closer)
    ).clip(0.0, 1.0)

    # Keep this deliberately small: it is a front-position confidence adjustment, not a result label.
    out["projected_front5_prob"] = (
        num(out["projected_front5_prob"]).fillna(0.0)
        + params["front_prob"] * (0.75 * survival - 0.85 * collapse)
    ).clip(0.0, 1.0)
    out["front_load_course_probability_adjustment"] = net
    out["front_load_course_variant"] = name
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
    rows: list[dict] = []
    for col in [
        "front_load_course_label",
        "ticket_type",
        "venue",
        "surface",
        "distance_bin",
        "class_tier",
        "queue_type",
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
        "front_load_survival_pair_fit",
        "front_load_collapse_front_risk",
        "front_load_collapse_closer_fit",
        "front_load_course_net_score",
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
    source = attach_front_load_context(read_pair_universe())
    source.to_csv(OUT / "pair_universe_with_front_load_course_context.csv", index=False, encoding="utf-8-sig")

    summaries: list[dict] = []
    holdouts: list[pd.DataFrame] = []
    all_segments: list[pd.DataFrame] = []
    all_years: list[dict] = []
    best_tickets: pd.DataFrame | None = None
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
            best_tickets = tickets
        holdouts.append(holdout)

    comparison = pd.DataFrame(summaries).drop(columns=["year_metrics"], errors="ignore")
    comparison = comparison.sort_values(["roi", "profit_yen"], ascending=False)
    comparison.to_csv(OUT / "variant_walkforward_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_years).to_csv(OUT / "variant_yearly_walkforward_summary.csv", index=False, encoding="utf-8-sig")
    if holdouts:
        hold = pd.concat(holdouts, ignore_index=True, sort=False)
        hold.to_csv(OUT / "variant_holdout_grid_all.csv", index=False, encoding="utf-8-sig")
        hold.sort_values(["dev_score", "val_profit_yen"], ascending=False).groupby("variant", as_index=False).head(5).to_csv(
            OUT / "variant_holdout_top5_by_variant.csv", index=False, encoding="utf-8-sig"
        )
    if all_segments:
        pd.concat(all_segments, ignore_index=True, sort=False).to_csv(
            OUT / "variant_selected_ticket_segment_metrics.csv", index=False, encoding="utf-8-sig"
        )

    summary = {
        "output_dir": str(OUT),
        "pair_universe_rows": int(len(source)),
        "race_context_rows": int(source["race_id"].nunique()),
        "variants": list(VARIANTS),
        "best_variant": best_variant,
        "comparison": comparison.to_dict(orient="records"),
        "note": (
            "Tests pre-race front-load x course-context signals as light pair-probability modifiers. "
            "Actual front survival/collapse labels are used only in the upstream rolling context validation, not as direct target-race inputs."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Front Load x Course Context Pair Probability v1",
                "",
                "Purpose: check whether high early-load races should support front-survival pairs or guard against front-collapse by course context.",
                "",
                "Key outputs:",
                "- pair_universe_with_front_load_course_context.csv",
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
