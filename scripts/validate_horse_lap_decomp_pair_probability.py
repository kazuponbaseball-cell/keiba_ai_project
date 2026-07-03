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
    load_universe,
    metrics,
    norm01,
    num,
    train_val_holdout,
    walkforward,
)


LAP_COMPACT = Path("outputs/analysis/horse_lap_aptitude_decomposition_v1/all_ticket_lap_decomposition_compact.csv")
RUNNER_DECOMP = Path("outputs/analysis/horse_lap_aptitude_decomposition_v1/runner_lap_aptitude_decomposition.csv")
OUT = Path("outputs/analysis/horse_lap_decomp_pair_probability_v1")

HP = "\u524d\u534a\u8ca0\u8377\u578b"
INSTANT = "\u77ac\u767a\u578b"
SLOW_INSTANT = "\u30b9\u30ed\u30fc\u77ac\u767a\u5bc4\u308a"
SUSTAIN = "\u6301\u7d9a\u578b"
LONG_SPURT = "\u30ed\u30f3\u30b0\u30b9\u30d1\u30fc\u30c8\u578b"
MATCH_BOTH = "2\u982d\u3068\u3082\u60f3\u5b9a\u30e9\u30c3\u30d7\u4e00\u81f4"
MATCH_ONE = "\u7247\u65b9\u306e\u307f\u60f3\u5b9a\u30e9\u30c3\u30d7\u4e00\u81f4"


VARIANTS = {
    "baseline": {"context": 0.0, "special": 0.0, "caution": 0.0, "rank": 0.0},
    "lap_light": {"context": 0.012, "special": 0.012, "caution": 0.012, "rank": 0.010},
    "lap_mid": {"context": 0.024, "special": 0.026, "caution": 0.026, "rank": 0.020},
    "lap_strong": {"context": 0.040, "special": 0.045, "caution": 0.045, "rank": 0.032},
    "lap_context_gate": {"context": 0.018, "special": 0.034, "caution": 0.018, "rank": 0.034},
}


def pair_key(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    out = df.copy()
    a = num(out[left])
    b = num(out[right])
    out["_pair_lo"] = np.minimum(a, b).astype("Int64")
    out["_pair_hi"] = np.maximum(a, b).astype("Int64")
    return out


def load_lap_pair_features() -> pd.DataFrame:
    if not LAP_COMPACT.exists():
        raise FileNotFoundError(f"missing lap decomposition compact csv: {LAP_COMPACT}")

    lap = pd.read_csv(LAP_COMPACT, dtype={"race_id": str}, low_memory=False)
    lap = lap[lap["dataset"].eq("lap_positive_selected")].copy()
    if lap.empty:
        raise ValueError("lap decomposition compact csv has no lap_positive_selected rows")

    lap["anchor_no"] = num(lap["anchor_no"]).astype("Int64")
    lap["partner_no"] = num(lap["partner_no"]).astype("Int64")
    lap = pair_key(lap, "anchor_no", "partner_no")

    fit = num(lap.get("pair_lap_fit_min_eval"), lap.index, 0.0).fillna(0.0).clip(0.0, 1.0)
    cosine = num(lap.get("pair_lap_cosine_min_eval"), lap.index, 0.0).fillna(0.0).clip(0.0, 1.0)
    decomp = num(lap.get("horse_lap_decomp_score"), lap.index, 0.0).fillna(0.0).clip(0.0, 1.0)
    popular_mismatch = num(lap.get("pair_lap_mismatch_popular_max_eval"), lap.index, 0.0).fillna(0.0).clip(0.0, 1.0)

    race_mode = lap.get("race_lap_mode_label", pd.Series("", index=lap.index)).fillna("").astype(str)
    anchor_type = lap.get("anchor_lap_type_label", pd.Series("", index=lap.index)).fillna("").astype(str)
    partner_type = lap.get("partner_lap_type_label", pd.Series("", index=lap.index)).fillna("").astype(str)
    match_bucket = lap.get("lap_match_bucket_label", pd.Series("", index=lap.index)).fillna("").astype(str)

    has_hp = anchor_type.eq(HP) | partner_type.eq(HP)
    has_instant = anchor_type.eq(INSTANT) | partner_type.eq(INSTANT)
    has_sustain = anchor_type.eq(SUSTAIN) | partner_type.eq(SUSTAIN)
    has_long = anchor_type.eq(LONG_SPURT) | partner_type.eq(LONG_SPURT)

    lap["lap_decomp_available"] = 1.0
    lap["lap_decomp_fit"] = fit
    lap["lap_decomp_cosine"] = cosine
    lap["lap_decomp_score"] = decomp
    lap["lap_decomp_both_match"] = (match_bucket.eq(MATCH_BOTH) & fit.ge(0.55)).astype(float)
    lap["lap_decomp_one_match"] = (match_bucket.eq(MATCH_ONE) & fit.ge(0.60)).astype(float)
    lap["lap_decomp_hp_instant"] = (race_mode.eq(HP) & has_hp & has_instant).astype(float)
    lap["lap_decomp_slow_long"] = (race_mode.eq(SLOW_INSTANT) & has_long).astype(float)
    lap["lap_decomp_sustain_pair"] = (race_mode.eq(SUSTAIN) & has_sustain & fit.ge(0.65)).astype(float)
    lap["lap_decomp_special"] = (
        lap["lap_decomp_hp_instant"].mul(0.45)
        + lap["lap_decomp_slow_long"].mul(0.30)
        + lap["lap_decomp_sustain_pair"].mul(0.25)
    ).clip(0.0, 1.0)
    lap["lap_decomp_caution"] = (
        popular_mismatch.mul(0.65)
        + (fit.lt(0.25) & cosine.lt(0.35)).astype(float).mul(0.25)
        + (match_bucket.ne(MATCH_BOTH) & fit.lt(0.40)).astype(float).mul(0.10)
    ).clip(0.0, 1.0)
    lap["lap_decomp_context"] = (
        0.36 * fit
        + 0.26 * cosine
        + 0.18 * decomp
        + 0.12 * lap["lap_decomp_both_match"]
        + 0.08 * lap["lap_decomp_one_match"]
        + 0.12 * lap["lap_decomp_special"]
        - 0.16 * lap["lap_decomp_caution"]
    ).clip(0.0, 1.0)

    agg = (
        lap.groupby(["race_id", "_pair_lo", "_pair_hi"], as_index=False)
        .agg(
            lap_decomp_available=("lap_decomp_available", "max"),
            lap_decomp_fit=("lap_decomp_fit", "mean"),
            lap_decomp_cosine=("lap_decomp_cosine", "mean"),
            lap_decomp_score=("lap_decomp_score", "mean"),
            lap_decomp_both_match=("lap_decomp_both_match", "max"),
            lap_decomp_one_match=("lap_decomp_one_match", "max"),
            lap_decomp_special=("lap_decomp_special", "max"),
            lap_decomp_caution=("lap_decomp_caution", "max"),
            lap_decomp_context=("lap_decomp_context", "max"),
        )
    )
    return agg


def load_runner_lap_features() -> pd.DataFrame:
    if not RUNNER_DECOMP.exists():
        raise FileNotFoundError(f"missing runner lap decomposition csv: {RUNNER_DECOMP}")

    wanted = {
        "race_id",
        "horse_no",
        "computed_race_lap_mode",
        "computed_horse_lap_type",
        "horse_lap_type_label",
        "horse_lap_type_strength",
        "lap_profile_fit_score",
        "lap_fit_confident_score",
        "lap_mismatch_popular_risk",
        "race_lap_prediction_confidence",
        "race_lap_profile_concentration",
        "horse_lap_profile_sharpness",
        "computed_lap_fit_cosine",
    }
    df = pd.read_csv(RUNNER_DECOMP, dtype={"race_id": str}, usecols=lambda c: c in wanted, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df["horse_no"] = num(df["horse_no"]).astype("Int64")
    return df.dropna(subset=["race_id", "horse_no"]).drop_duplicates(["race_id", "horse_no"], keep="last")


def add_full_runner_lap_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    runner = load_runner_lap_features()
    base = df.copy()
    base["anchor_no"] = num(base["anchor_no"]).astype("Int64")
    base["partner_no"] = num(base["partner_no"]).astype("Int64")

    anchor = runner.add_prefix("anchor_lap_").rename(
        columns={"anchor_lap_race_id": "race_id", "anchor_lap_horse_no": "anchor_no"}
    )
    partner = runner.add_prefix("partner_lap_").rename(
        columns={"partner_lap_race_id": "race_id", "partner_lap_horse_no": "partner_no"}
    )
    out = base.merge(anchor, on=["race_id", "anchor_no"], how="left")
    out = out.merge(partner, on=["race_id", "partner_no"], how="left")

    def ncol(col: str) -> pd.Series:
        return num(out.get(col), out.index, 0.0).fillna(0.0).clip(0.0, 1.0)

    anchor_fit = ncol("anchor_lap_lap_profile_fit_score")
    partner_fit = ncol("partner_lap_lap_profile_fit_score")
    anchor_cosine = ncol("anchor_lap_computed_lap_fit_cosine")
    partner_cosine = ncol("partner_lap_computed_lap_fit_cosine")
    anchor_conf = ncol("anchor_lap_lap_fit_confident_score")
    partner_conf = ncol("partner_lap_lap_fit_confident_score")
    anchor_strength = ncol("anchor_lap_horse_lap_type_strength")
    partner_strength = ncol("partner_lap_horse_lap_type_strength")
    anchor_mismatch = ncol("anchor_lap_lap_mismatch_popular_risk")
    partner_mismatch = ncol("partner_lap_lap_mismatch_popular_risk")

    race_mode = (
        out.get("anchor_lap_computed_race_lap_mode", pd.Series("", index=out.index))
        .fillna(out.get("partner_lap_computed_race_lap_mode", pd.Series("", index=out.index)))
        .fillna("")
        .astype(str)
    )
    anchor_type = out.get("anchor_lap_computed_horse_lap_type", pd.Series("", index=out.index)).fillna("").astype(str)
    partner_type = out.get("partner_lap_computed_horse_lap_type", pd.Series("", index=out.index)).fillna("").astype(str)

    has_anchor = out.get("anchor_lap_computed_horse_lap_type", pd.Series(np.nan, index=out.index)).notna()
    has_partner = out.get("partner_lap_computed_horse_lap_type", pd.Series(np.nan, index=out.index)).notna()
    available = (has_anchor & has_partner).astype(float)

    anchor_match = anchor_type.eq(race_mode)
    partner_match = partner_type.eq(race_mode)
    both_match = (anchor_match & partner_match & anchor_fit.ge(0.45) & partner_fit.ge(0.45)).astype(float)
    one_match = ((anchor_match ^ partner_match) & np.minimum(anchor_fit, partner_fit).ge(0.50)).astype(float)

    has_fast = anchor_type.eq("fast") | partner_type.eq("fast")
    has_instant = anchor_type.eq("instant") | partner_type.eq("instant")
    has_sustain = anchor_type.eq("sustain") | partner_type.eq("sustain")
    has_long = anchor_type.eq("long_spurt") | partner_type.eq("long_spurt")
    race_fast = race_mode.eq("fast")
    race_slow_or_instant = race_mode.isin(["slow", "instant"])
    race_sustain = race_mode.eq("sustain")
    race_long = race_mode.eq("long_spurt")

    hp_instant = (race_fast & has_fast & has_instant).astype(float)
    slow_long = (race_slow_or_instant & has_long & (has_instant | has_sustain)).astype(float)
    sustain_pair = (race_sustain & has_sustain & np.minimum(anchor_fit, partner_fit).ge(0.55)).astype(float)
    long_pair = (race_long & has_long & np.maximum(anchor_strength, partner_strength).ge(0.08)).astype(float)

    out["lap_decomp_available"] = available
    out["lap_decomp_fit"] = (np.minimum(anchor_fit, partner_fit) * available).clip(0.0, 1.0)
    out["lap_decomp_cosine"] = (np.minimum(anchor_cosine, partner_cosine) * available).clip(0.0, 1.0)
    out["lap_decomp_score"] = (
        (
            0.38 * np.minimum(anchor_fit, partner_fit)
            + 0.30 * np.minimum(anchor_cosine, partner_cosine)
            + 0.18 * np.minimum(anchor_conf, partner_conf)
            + 0.14 * np.minimum(anchor_strength, partner_strength)
        )
        * available
    ).clip(0.0, 1.0)
    out["lap_decomp_both_match"] = both_match * available
    out["lap_decomp_one_match"] = one_match * available
    out["lap_decomp_special"] = (
        (0.35 * hp_instant + 0.25 * slow_long + 0.25 * sustain_pair + 0.15 * long_pair) * available
    ).clip(0.0, 1.0)
    out["lap_decomp_caution"] = (
        (
            0.50 * np.maximum(anchor_mismatch, partner_mismatch)
            + 0.24 * (np.minimum(anchor_fit, partner_fit).lt(0.24)).astype(float)
            + 0.18 * (np.minimum(anchor_conf, partner_conf).lt(0.08)).astype(float)
            + 0.08 * ((~anchor_match) & (~partner_match)).astype(float)
        )
        * available
    ).clip(0.0, 1.0)
    out["lap_decomp_context"] = (
        (
            0.32 * out["lap_decomp_fit"]
            + 0.24 * out["lap_decomp_cosine"]
            + 0.18 * out["lap_decomp_score"]
            + 0.10 * out["lap_decomp_both_match"]
            + 0.06 * out["lap_decomp_one_match"]
            + 0.12 * out["lap_decomp_special"]
            - 0.15 * out["lap_decomp_caution"]
        )
        * available
    ).clip(0.0, 1.0)
    return out


def attach_lap_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        out = add_full_runner_lap_pair_features(df)
        out["lap_decomp_feature_source"] = "runner_full"
    except Exception:
        base = pair_key(df, "anchor_no", "partner_no")
        lap = load_lap_pair_features()
        out = base.merge(lap, on=["race_id", "_pair_lo", "_pair_hi"], how="left")
        out["lap_decomp_feature_source"] = "compact_fallback"
        out = out.drop(columns=["_pair_lo", "_pair_hi"], errors="ignore")
    lap_cols = [c for c in out.columns if c.startswith("lap_decomp_") and c != "lap_decomp_feature_source"]
    for col in lap_cols:
        out[col] = num(out[col], out.index, 0.0).fillna(0.0)
    return out


def apply_variant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    params = VARIANTS[name]
    out = df.copy()
    if name == "baseline":
        out["lap_probability_variant"] = name
        return out

    available = clip01(out["lap_decomp_available"])
    context = clip01(out["lap_decomp_context"])
    special = clip01(out["lap_decomp_special"])
    caution = clip01(out["lap_decomp_caution"])
    both_match = clip01(out["lap_decomp_both_match"])
    fit_high = (num(out["lap_decomp_fit"]).fillna(0.0).ge(0.70) & num(out["lap_decomp_cosine"]).fillna(0.0).ge(0.60)).astype(float)

    additive = available * (
        params["context"] * context
        + params["special"] * special
        + 0.010 * both_match
        + 0.008 * fit_high
        - params["caution"] * caution
    )
    rank_additive = available * (
        params["rank"] * (0.55 * context + 0.25 * special + 0.20 * both_match)
        - (params["rank"] * 0.70) * caution
    )

    out["pair_quinella_score"] = (num(out["pair_quinella_score"]).fillna(0.0) + additive).clip(0.0, 1.0)
    out["pair_score"] = (num(out["pair_score"]).fillna(0.0) + 0.75 * additive).clip(0.0, 1.0)
    out["market_overlay_score"] = (num(out["market_overlay_score"]).fillna(0.0) + 0.35 * rank_additive).clip(0.0, 1.0)
    out["horse_lap_pair_probability_adjustment"] = additive
    out["horse_lap_rank_adjustment"] = rank_additive
    out["lap_probability_variant"] = name
    return out


def aggregate_tickets(tickets: pd.DataFrame, label: str) -> dict:
    m = metrics(tickets, label)
    if not tickets.empty:
        by_year = []
        for year, part in tickets.groupby("year", sort=True):
            y = metrics(part, f"{label}_{year}")
            by_year.append({"year": int(year), "roi": y["roi"], "races": y["races"], "profit_yen": y["profit_yen"]})
        m["min_year_roi"] = min((x["roi"] for x in by_year), default=0.0)
        m["year_metrics"] = by_year
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
    total["lap_adjustment_available_pairs"] = int(num(source.get("lap_decomp_available"), source.index, 0.0).fillna(0.0).gt(0).sum())
    total["lap_adjustment_available_rate"] = float(num(source.get("lap_decomp_available"), source.index, 0.0).fillna(0.0).gt(0).mean())
    return total, train_grid, wf_summary, holdout, wf_tickets


def compare_ticket_delta(baseline: pd.DataFrame, variant: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    if baseline.empty and variant.empty:
        return pd.DataFrame()
    base = baseline.copy()
    var = variant.copy()
    for frame in (base, var):
        if "ticket_key" not in frame.columns:
            frame["ticket_key"] = (
                frame["ticket_type"].astype(str)
                + ":"
                + frame["race_id"].astype(str)
                + ":"
                + frame["anchor_no"].astype(str)
                + "-"
                + frame["partner_no"].astype(str)
            )
    base_keys = set(base["ticket_key"].astype(str))
    var_keys = set(var["ticket_key"].astype(str))
    rows: list[pd.DataFrame] = []
    if var_keys - base_keys:
        added = var[var["ticket_key"].astype(str).isin(var_keys - base_keys)].copy()
        added["delta_type"] = "added_by_variant"
        rows.append(added)
    if base_keys - var_keys:
        removed = base[base["ticket_key"].astype(str).isin(base_keys - var_keys)].copy()
        removed["delta_type"] = "removed_from_baseline"
        rows.append(removed)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["variant"] = variant_name
    cols = ["variant"] + [c for c in out.columns if c != "variant"]
    out = out[cols]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    universe = attach_lap_features(load_universe())
    universe.to_csv(OUT / "pair_universe_with_lap_decomp_features.csv", index=False, encoding="utf-8-sig")

    comparison: list[dict] = []
    wf_frames: list[pd.DataFrame] = []
    holdout_frames: list[pd.DataFrame] = []
    ticket_by_variant: dict[str, pd.DataFrame] = {}

    for name in VARIANTS:
        total, train_grid, wf_summary, holdout, wf_tickets = run_variant(universe, name)
        comparison.append(total)
        train_grid.to_csv(OUT / f"{name}_walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
        wf_summary.insert(0, "variant", name)
        holdout.insert(0, "variant", name)
        wf_summary.to_csv(OUT / f"{name}_walkforward_summary.csv", index=False, encoding="utf-8-sig")
        holdout.to_csv(OUT / f"{name}_train2024_val2025_hold2026_grid.csv", index=False, encoding="utf-8-sig")
        wf_tickets.insert(0, "variant", name)
        wf_tickets.to_csv(OUT / f"{name}_walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
        wf_frames.append(wf_summary)
        holdout_frames.append(holdout)
        ticket_by_variant[name] = wf_tickets

    comp = pd.DataFrame(comparison).sort_values(["roi", "profit_yen"], ascending=False)
    comp.to_csv(OUT / "variant_walkforward_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(wf_frames, ignore_index=True, sort=False).to_csv(OUT / "variant_yearly_walkforward_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(holdout_frames, ignore_index=True, sort=False).to_csv(OUT / "variant_holdout_grid_all.csv", index=False, encoding="utf-8-sig")

    holdout_top = (
        pd.concat(holdout_frames, ignore_index=True, sort=False)
        .sort_values(["variant", "dev_score", "val_profit_yen"], ascending=[True, False, False])
        .groupby("variant", as_index=False)
        .head(5)
    )
    holdout_top.to_csv(OUT / "variant_holdout_top5_by_variant.csv", index=False, encoding="utf-8-sig")

    delta_frames = []
    baseline_tickets = ticket_by_variant.get("baseline", pd.DataFrame())
    for name, tickets in ticket_by_variant.items():
        if name == "baseline":
            continue
        delta = compare_ticket_delta(baseline_tickets, tickets, name)
        if not delta.empty:
            delta_frames.append(delta)
    if delta_frames:
        delta_all = pd.concat(delta_frames, ignore_index=True, sort=False)
        delta_all.to_csv(OUT / "variant_ticket_delta_vs_baseline.csv", index=False, encoding="utf-8-sig")
        delta_metrics = []
        for (variant, delta_type), part in delta_all.groupby(["variant", "delta_type"], sort=True):
            row = metrics(part, f"{variant}_{delta_type}")
            row["variant"] = variant
            row["delta_type"] = delta_type
            delta_metrics.append(row)
        pd.DataFrame(delta_metrics).to_csv(OUT / "variant_ticket_delta_metrics.csv", index=False, encoding="utf-8-sig")

    best = comp.iloc[0].to_dict() if not comp.empty else {}
    summary = {
        "input_lap_compact": str((ROOT / LAP_COMPACT).resolve()),
        "input_runner_decomp": str((ROOT / RUNNER_DECOMP).resolve()),
        "output_dir": str((ROOT / OUT).resolve()),
        "lap_decomp_feature_source": (
            str(universe["lap_decomp_feature_source"].dropna().iloc[0])
            if "lap_decomp_feature_source" in universe.columns and universe["lap_decomp_feature_source"].notna().any()
            else "unknown"
        ),
        "variants": VARIANTS,
        "universe_rows": int(len(universe)),
        "universe_races": int(universe["race_id"].nunique()),
        "lap_feature_pair_rows": int(num(universe.get("lap_decomp_available"), universe.index, 0.0).fillna(0.0).gt(0).sum()),
        "lap_feature_pair_rate": float(num(universe.get("lap_decomp_available"), universe.index, 0.0).fillna(0.0).gt(0).mean()),
        "best_variant": best,
        "comparison": comparison,
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
        "lap_adjustment_available_rate",
    ]
    print(comp[cols].to_string(index=False))
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
