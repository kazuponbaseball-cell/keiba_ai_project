from __future__ import annotations

import json
from pathlib import Path
import sys

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
OPENING_CONTEXT = Path("outputs/analysis/opening_week_course_change_context_v1/race_opening_week_context.csv")
OUT = Path("outputs/analysis/opening_course_lap_pair_probability_v1")


VARIANTS = {
    "baseline": {"support": 0.0, "caution": 0.0, "market": 0.0, "d_penalty": 0.0, "opening_penalty": 0.0},
    "opening_course_lap_light": {"support": 0.012, "caution": 0.012, "market": 0.006, "d_penalty": 0.0, "opening_penalty": 0.0},
    "opening_course_lap_mid": {"support": 0.024, "caution": 0.024, "market": 0.010, "d_penalty": 0.0, "opening_penalty": 0.0},
    "course_caution_only": {"support": 0.000, "caution": 0.018, "market": 0.000, "d_penalty": 0.0, "opening_penalty": 0.0},
    "course_support_only": {"support": 0.018, "caution": 0.000, "market": 0.008, "d_penalty": 0.0, "opening_penalty": 0.0},
    "course_d_guard": {"support": 0.000, "caution": 0.000, "market": 0.000, "d_penalty": 0.050, "opening_penalty": 0.0},
    "course_d_opening_guard": {"support": 0.008, "caution": 0.010, "market": 0.004, "d_penalty": 0.050, "opening_penalty": 0.025},
    "opening_course_lap_strong": {"support": 0.038, "caution": 0.038, "market": 0.014, "d_penalty": 0.0, "opening_penalty": 0.0},
}


def read_pair_universe() -> pd.DataFrame:
    if not PAIR_UNIVERSE.exists():
        raise FileNotFoundError(f"missing pair universe with lap decomposition features: {PAIR_UNIVERSE}")
    df = pd.read_csv(PAIR_UNIVERSE, dtype={"race_id": str}, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    if "year" not in df.columns:
        df["year"] = df["race_id"].str[:4].astype(int)
    return df


def read_opening_context() -> pd.DataFrame:
    if not OPENING_CONTEXT.exists():
        raise FileNotFoundError(f"missing opening week/course-change context: {OPENING_CONTEXT}")
    keep = [
        "race_id",
        "meeting_stage",
        "is_opening_2days",
        "is_early_4days",
        "is_final_2days",
        "course_setting_available",
        "course_setting",
        "course_setting_changed",
        "course_setting_day_index",
        "course_change_stage",
        "front_survival_despite_pressure_score",
        "front_collapse_reinforced_score",
        "front_context_readability_score",
        "front_survival_edge_vs_global",
        "front_collapse_edge_vs_global",
        "pre_high_pressure_signal",
        "surface",
        "venue",
    ]
    ctx = pd.read_csv(OPENING_CONTEXT, dtype={"race_id": str}, usecols=lambda c: c in keep, low_memory=False)
    ctx["race_id"] = ctx["race_id"].astype(str)
    return ctx.drop_duplicates("race_id", keep="last")


def attach_opening_context(df: pd.DataFrame) -> pd.DataFrame:
    ctx = read_opening_context()
    out = df.merge(ctx, on="race_id", how="left", suffixes=("", "_opening"))
    idx = out.index
    for col in [
        "is_opening_2days",
        "is_early_4days",
        "is_final_2days",
        "course_setting_changed",
        "course_setting_day_index",
        "front_survival_despite_pressure_score",
        "front_collapse_reinforced_score",
        "front_context_readability_score",
        "front_survival_edge_vs_global",
        "front_collapse_edge_vs_global",
        "pre_high_pressure_signal",
    ]:
        out[col] = num(out.get(col), idx, 0.0).fillna(0.0)
    for col, default in [
        ("meeting_stage", "unknown"),
        ("course_setting", ""),
        ("course_change_stage", "unknown"),
        ("surface", ""),
        ("venue_opening", ""),
    ]:
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default).astype(str)

    lap_context = clip01(out.get("lap_decomp_context", pd.Series(0.0, index=idx)))
    lap_fit = clip01(out.get("lap_decomp_fit", pd.Series(0.0, index=idx)))
    lap_special = clip01(out.get("lap_decomp_special", pd.Series(0.0, index=idx)))
    lap_caution = clip01(out.get("lap_decomp_caution", pd.Series(0.0, index=idx)))

    readability = clip01(out["front_context_readability_score"])
    survival = clip01(out["front_survival_despite_pressure_score"])
    collapse = clip01(out["front_collapse_reinforced_score"])
    survival_edge = norm01(out["front_survival_edge_vs_global"], lo=-0.04, hi=0.05)
    collapse_edge = norm01(out["front_collapse_edge_vs_global"], lo=-0.03, hi=0.05)

    high_pressure_cut = float(out["pre_high_pressure_signal"].quantile(0.60)) if out["pre_high_pressure_signal"].notna().any() else 0.25
    high_pressure = out["pre_high_pressure_signal"].ge(high_pressure_cut).astype(float)
    opening = out["is_opening_2days"].ge(1.0).astype(float)
    final_2days = out["is_final_2days"].ge(1.0).astype(float)
    early_4days = out["is_early_4days"].ge(1.0).astype(float)
    course_setting = out["course_setting"].str.upper()
    course_c = course_setting.eq("C").astype(float)
    course_b = course_setting.eq("B").astype(float)
    course_d = course_setting.eq("D").astype(float)
    course_unknown = course_setting.eq("").astype(float)
    course_opening = out["course_change_stage"].isin(["change_day", "course_opening_2days", "course_early_4days"]).astype(float)

    out["opening_course_lap_support_score"] = (
        lap_context
        * (
            0.25 * readability
            + 0.22 * survival
            + 0.18 * survival_edge
            + 0.13 * final_2days
            + 0.10 * course_c
            + 0.07 * course_b
            + 0.05 * course_opening
        )
        + 0.10 * lap_fit * (course_c + final_2days).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["opening_course_lap_caution_score"] = (
        lap_context
        * (
            0.26 * collapse
            + 0.22 * collapse_edge
            + 0.20 * opening * high_pressure
            + 0.16 * course_d
            + 0.08 * course_unknown
            + 0.08 * lap_caution
        )
        + 0.08 * lap_special * opening * high_pressure
    ).clip(0.0, 1.0)
    out["opening_course_lap_context_net_score"] = (
        out["opening_course_lap_support_score"] - out["opening_course_lap_caution_score"]
    ).clip(-1.0, 1.0)
    out["opening_course_lap_label"] = np.select(
        [
            out["opening_course_lap_caution_score"].ge(0.55),
            out["opening_course_lap_support_score"].ge(0.55),
            out["opening_course_lap_context_net_score"].ge(0.12),
            out["opening_course_lap_context_net_score"].le(-0.12),
        ],
        [
            "opening_course_lap_caution",
            "opening_course_lap_support",
            "opening_course_lap_slight_support",
            "opening_course_lap_slight_caution",
        ],
        default="opening_course_lap_neutral",
    )
    return out


def apply_variant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    params = VARIANTS[name]
    if name == "baseline":
        out["opening_course_lap_variant"] = name
        return out

    support = clip01(out["opening_course_lap_support_score"])
    caution = clip01(out["opening_course_lap_caution_score"])
    course_d = out["course_setting"].fillna("").astype(str).str.upper().eq("D").astype(float)
    opening = num(out.get("is_opening_2days"), out.index, 0.0).fillna(0.0).ge(1.0).astype(float)
    net = (
        params["support"] * support
        - params["caution"] * caution
        - params.get("d_penalty", 0.0) * course_d
        - params.get("opening_penalty", 0.0) * opening
    )
    out["pair_quinella_score"] = (num(out["pair_quinella_score"]).fillna(0.0) + net).clip(0.0, 1.0)
    out["pair_score"] = (num(out["pair_score"]).fillna(0.0) + 0.72 * net).clip(0.0, 1.0)
    out["market_overlay_score"] = (
        num(out["market_overlay_score"]).fillna(0.0)
        + params["market"] * (0.70 * support - 0.50 * caution)
    ).clip(0.0, 1.0)
    out["opening_course_lap_probability_adjustment"] = net
    out["opening_course_lap_variant"] = name
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
    stake_col = pd.to_numeric(tickets.get("stake_yen"), errors="coerce").fillna(0.0)
    ret_col = pd.to_numeric(tickets.get("return_yen"), errors="coerce").fillna(0.0)
    for col in ["meeting_stage", "course_setting", "course_change_stage", "opening_course_lap_label", "ticket_type"]:
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
    # Support/caution quantile bins for selected tickets.
    for score_col in ["opening_course_lap_support_score", "opening_course_lap_caution_score"]:
        if score_col not in tickets.columns:
            continue
        work = tickets.copy()
        try:
            work["_bin"] = pd.qcut(pd.to_numeric(work[score_col], errors="coerce").rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])
        except ValueError:
            continue
        for val, part in work.groupby("_bin", observed=True):
            stake = float(pd.to_numeric(part.get("stake_yen"), errors="coerce").fillna(0.0).sum())
            ret = float(pd.to_numeric(part.get("return_yen"), errors="coerce").fillna(0.0).sum())
            rows.append(
                {
                    "variant": label,
                    "segment_col": score_col,
                    "segment_value": str(val),
                    "tickets": int(len(part)),
                    "races": int(part["race_id"].nunique()),
                    "stake_yen": stake,
                    "return_yen": ret,
                    "profit_yen": ret - stake,
                    "roi": ret / stake if stake else 0.0,
                }
            )
    _ = stake_col, ret_col
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    universe = attach_opening_context(read_pair_universe())
    universe.to_csv(OUT / "pair_universe_with_opening_course_lap_context.csv", index=False, encoding="utf-8-sig")

    comparison: list[dict] = []
    wf_frames: list[pd.DataFrame] = []
    hold_frames: list[pd.DataFrame] = []
    segment_frames: list[pd.DataFrame] = []

    for name in VARIANTS:
        total, train_grid, wf_summary, holdout, wf_tickets = run_variant(universe, name)
        comparison.append(total)
        train_grid.to_csv(OUT / f"{name}_walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
        wf_summary.insert(0, "variant", name)
        holdout.insert(0, "variant", name)
        wf_tickets.insert(0, "variant", name)
        wf_summary.to_csv(OUT / f"{name}_walkforward_summary.csv", index=False, encoding="utf-8-sig")
        holdout.to_csv(OUT / f"{name}_train2024_val2025_hold2026_grid.csv", index=False, encoding="utf-8-sig")
        wf_tickets.to_csv(OUT / f"{name}_walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
        wf_frames.append(wf_summary)
        hold_frames.append(holdout)
        segment_frames.append(segment_metrics(wf_tickets, name))

    comp = pd.DataFrame(comparison).sort_values(["roi", "profit_yen"], ascending=False)
    comp.to_csv(OUT / "variant_walkforward_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(wf_frames, ignore_index=True, sort=False).to_csv(OUT / "variant_yearly_walkforward_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(hold_frames, ignore_index=True, sort=False).to_csv(OUT / "variant_holdout_grid_all.csv", index=False, encoding="utf-8-sig")
    if segment_frames:
        pd.concat(segment_frames, ignore_index=True, sort=False).to_csv(OUT / "variant_selected_ticket_segment_metrics.csv", index=False, encoding="utf-8-sig")

    hold_top = (
        pd.concat(hold_frames, ignore_index=True, sort=False)
        .sort_values(["variant", "dev_score", "val_profit_yen"], ascending=[True, False, False])
        .groupby("variant", as_index=False)
        .head(5)
    )
    hold_top.to_csv(OUT / "variant_holdout_top5_by_variant.csv", index=False, encoding="utf-8-sig")

    summary = {
        "pair_universe": str((ROOT / PAIR_UNIVERSE).resolve()),
        "opening_context": str((ROOT / OPENING_CONTEXT).resolve()),
        "output_dir": str((ROOT / OUT).resolve()),
        "universe_rows": int(len(universe)),
        "universe_races": int(universe["race_id"].nunique()),
        "variants": VARIANTS,
        "best_variant": comp.iloc[0].replace({np.nan: None}).to_dict() if not comp.empty else {},
        "comparison": comp.replace({np.nan: None}).to_dict(orient="records"),
        "context_coverage": {
            "meeting_stage_known_rate": float(universe["meeting_stage"].ne("unknown").mean()),
            "course_setting_known_rate": float(universe["course_setting"].astype(str).ne("").mean()),
        },
        "decision_note": (
            "This validates opening-week/course-setting as a lap pair probability modifier. "
            "Use only if it improves walk-forward ROI without a large holdout or min-year deterioration."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
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
    print(comp[cols].to_string(index=False))
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
