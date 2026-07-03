from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_FEATURES = [
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/body_weight_backfilled/train_features_with_same_day_bias_v3_retro_body_context.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/body_weight_backfilled/test_features_with_same_day_bias_v3_retro_body_context.csv",
]
DEFAULT_TICKETS = "outputs/analysis/ticket_conversion_fixes_v1/walkforward_selected_tickets.csv"
DEFAULT_UNIVERSE = "outputs/analysis/ticket_conversion_fixes_v1/pair_strength_universe_with_conversion_features.csv"


FEATURE_COLS = [
    "レースID(新/馬番無)",
    "日付S",
    "場所",
    "Ｒ",
    "馬名",
    "馬番",
    "血統登録番号",
    "芝・ダ",
    "距離",
    "出走頭数",
    "頭数",
    "1角",
    "2角",
    "3角",
    "4角",
    "4角.1",
    "確定着順",
    "target_score",
    "target_top3",
    "prev_race_time_value",
    "prev_time_adjusted_by_day_bias",
    "prev_class_time_value_score",
    "past3_avg_time_value",
    "past3_best_time_value",
    "past3_avg_time_z",
    "past3_avg_time_adjusted_by_day_bias",
    "horse_time_value_plus_margin",
    "horse_fast_lap_count_past5",
    "horse_fast_lap_score_past5",
    "horse_slow_lap_count_past5",
    "horse_slow_lap_score_past5",
    "lap_pace_versatility_score",
    "lap_aptitude_fit_score",
    "lap_aptitude_reliability_score",
    "front_running_tendency",
    "horse_front_run_rate_past5",
    "horse_stalker_rate_past5",
    "horse_early_move_avg_past5",
    "prev_early_move",
    "prev_corner4_position_rate",
    "race_early_pressure_score",
    "race_pace_collapse_risk",
    "race_slow_pace_risk",
    "pace_fit_score",
    "front_advantage_score",
    "closer_advantage_score",
    "draw_pace_fit_score",
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


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    values = frame[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def norm_clip(series: pd.Series, lo: float, hi: float, default: float = 0.5) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = ((x - lo) / (hi - lo)).replace([np.inf, -np.inf], np.nan)
    return out.clip(0.0, 1.0).fillna(default)


def parse_date_key(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("")
    digits = text.str.replace(r"\D", "", regex=True)
    parsed = pd.to_datetime(digits.str[:8], format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(text, errors="coerce")
    return parsed.fillna(fallback)


def load_feature_rows(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = project_path(raw_path)
        if not path.exists():
            continue
        header = read_csv_any(path, nrows=0).columns.tolist()
        usecols = [col for col in FEATURE_COLS if col in header]
        frame = read_csv_any(path, usecols=usecols)
        frame["_source_csv"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No feature CSVs were loaded.")
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["race_id"] = out["レースID(新/馬番無)"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(16)
    out["horse_no"] = pd.to_numeric(out["馬番"], errors="coerce").astype("Int64")
    out = out[out["race_id"].str.len().eq(16) & out["horse_no"].notna()].copy()
    out["horse_no"] = out["horse_no"].astype(int)
    out["_date"] = parse_date_key(out["日付S"]) if "日付S" in out.columns else pd.NaT
    out["_race_no"] = pd.to_numeric(out.get("Ｒ"), errors="coerce").fillna(0.0)
    out["_horse_id"] = out.get("血統登録番号", out["馬名"]).astype("string").fillna("").astype(str)
    out = out.sort_values(["_horse_id", "_date", "race_id"], kind="mergesort").drop_duplicates(
        ["race_id", "horse_no"], keep="last"
    )
    return add_candidate_features(out)


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    field = num(out, "出走頭数")
    if field.isna().all():
        field = num(out, "頭数")
    field = field.replace(0, np.nan)

    c1 = num(out, "1角")
    c2 = num(out, "2角")
    c3 = num(out, "3角")
    c4 = num(out, "4角.1")
    if c4.isna().all():
        c4 = num(out, "4角")
    first_corner = pd.concat([c1, c2, c3, c4], axis=1).bfill(axis=1).iloc[:, 0]
    first_rate = (first_corner / field).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
    c4_rate = (c4 / field).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)

    out["_first_corner_rate_for_history"] = first_rate
    out["_first_corner_front_flag_for_history"] = first_rate.le(0.35).astype(float).where(first_rate.notna(), np.nan)
    out["_first_to_4c_gain_for_history"] = (first_corner - c4).replace([np.inf, -np.inf], np.nan)
    out["_corner4_rate_for_history"] = c4_rate

    out["course_corner_type"] = np.select(
        [c1.notna() | c2.notna(), c3.notna() | c4.notna()],
        ["four_corner", "two_corner"],
        default="straight_or_unknown",
    )
    out["course_corner_count_proxy"] = np.select(
        [out["course_corner_type"].eq("four_corner"), out["course_corner_type"].eq("two_corner")],
        [4.0, 2.0],
        default=0.0,
    )

    sort_cols = ["_horse_id", "_date", "race_id"]
    ordered = out.sort_values(sort_cols, kind="mergesort")
    grouped = ordered.groupby("_horse_id", sort=False)
    for source, dest, agg in [
        ("_first_corner_front_flag_for_history", "past5_first_corner_front_rate", "mean"),
        ("_first_corner_rate_for_history", "past5_first_corner_rate_avg", "mean"),
        ("_first_to_4c_gain_for_history", "past5_first_to_4c_gain_avg", "mean"),
        ("_corner4_rate_for_history", "past5_corner4_rate_avg", "mean"),
    ]:
        values = pd.to_numeric(ordered[source], errors="coerce")
        rolled = grouped[source].transform(lambda s: pd.to_numeric(s, errors="coerce").shift().rolling(5, min_periods=1).mean())
        out.loc[ordered.index, dest] = pd.to_numeric(rolled, errors="coerce").astype(float)

    out["early_start_score_candidate"] = (
        0.55 * num(out, "horse_front_run_rate_past5", 0.0).fillna(0.0).clip(0.0, 1.0)
        + 0.30 * out["past5_first_corner_front_rate"].fillna(0.0).clip(0.0, 1.0)
        + 0.15 * (1.0 - out["past5_first_corner_rate_avg"].fillna(0.75).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    out["second_leg_score_candidate"] = (
        0.55 * norm_clip(out["past5_first_to_4c_gain_avg"], -2.0, 3.0)
        + 0.25 * norm_clip(num(out, "horse_early_move_avg_past5"), -2.0, 3.0)
        + 0.20 * norm_clip(num(out, "prev_early_move"), -2.0, 3.0)
    ).clip(0.0, 1.0)
    out["fast_clock_tolerance_candidate"] = (
        0.28 * norm_clip(num(out, "past3_avg_time_value"), -0.12, 0.12)
        + 0.18 * norm_clip(num(out, "past3_best_time_value"), -0.10, 0.18)
        + 0.18 * norm_clip(num(out, "prev_class_time_value_score"), -0.10, 0.18)
        + 0.16 * norm_clip(num(out, "horse_time_value_plus_margin"), -0.18, 0.18)
        + 0.12 * norm_clip(num(out, "horse_fast_lap_score_past5"), 0.0, 0.75, default=0.0)
        + 0.08 * norm_clip(num(out, "lap_aptitude_fit_score"), 0.0, 0.40, default=0.0)
    ).clip(0.0, 1.0)

    ordered = out.sort_values(["_horse_id", "course_corner_type", "_date", "race_id"], kind="mergesort")
    grouped_corner = ordered.groupby(["_horse_id", "course_corner_type"], sort=False)
    starts = grouped_corner.cumcount()
    top3_csum = grouped_corner["target_top3"].cumsum().groupby(
        [ordered["_horse_id"], ordered["course_corner_type"]], sort=False
    ).shift(fill_value=0.0)
    score_csum = grouped_corner["target_score"].cumsum().groupby(
        [ordered["_horse_id"], ordered["course_corner_type"]], sort=False
    ).shift(fill_value=0.0)
    first_in_group = grouped_corner.cumcount().eq(0)
    top3_csum.loc[first_in_group] = 0.0
    score_csum.loc[first_in_group] = 0.0
    denom = starts.replace(0, np.nan)
    out.loc[ordered.index, "corner_type_starts_candidate"] = starts.astype(float)
    out.loc[ordered.index, "corner_type_top3_rate_candidate"] = (top3_csum / denom).fillna(0.0).astype(float)
    out.loc[ordered.index, "corner_type_avg_score_candidate"] = (score_csum / denom).fillna(0.0).astype(float)
    out["corner_type_fit_score_candidate"] = (
        0.60 * norm_clip(out["corner_type_avg_score_candidate"], 0.0, 0.65, default=0.0)
        + 0.25 * out["corner_type_top3_rate_candidate"].fillna(0.0).clip(0.0, 1.0)
        + 0.15 * (out["corner_type_starts_candidate"].fillna(0.0).clip(upper=5.0) / 5.0)
    ).clip(0.0, 1.0)
    out["corner_type_low_sample_flag_candidate"] = out["corner_type_starts_candidate"].fillna(0.0).lt(2).astype(float)

    keep = [
        "race_id",
        "horse_no",
        "course_corner_type",
        "course_corner_count_proxy",
        "fast_clock_tolerance_candidate",
        "early_start_score_candidate",
        "second_leg_score_candidate",
        "corner_type_fit_score_candidate",
        "corner_type_low_sample_flag_candidate",
        "past5_first_corner_front_rate",
        "past5_first_corner_rate_avg",
        "past5_first_to_4c_gain_avg",
    ]
    return out[keep].copy()


def enrich_pairs(pairs: pd.DataFrame, horse_features: pd.DataFrame) -> pd.DataFrame:
    out = pairs.copy()
    out["race_id"] = out["race_id"].astype(str).str.zfill(16)
    feature_cols = [c for c in horse_features.columns if c not in {"race_id", "horse_no"}]
    anchor = horse_features.rename(columns={c: f"anchor_{c}" for c in feature_cols})
    partner = horse_features.rename(columns={c: f"partner_{c}" for c in feature_cols})
    out = out.merge(anchor, left_on=["race_id", "anchor_no"], right_on=["race_id", "horse_no"], how="left")
    out = out.drop(columns=["horse_no"], errors="ignore")
    out = out.merge(partner, left_on=["race_id", "partner_no"], right_on=["race_id", "horse_no"], how="left")
    out = out.drop(columns=["horse_no"], errors="ignore")

    for base in [
        "fast_clock_tolerance_candidate",
        "early_start_score_candidate",
        "second_leg_score_candidate",
        "corner_type_fit_score_candidate",
    ]:
        a = pd.to_numeric(out.get(f"anchor_{base}"), errors="coerce")
        b = pd.to_numeric(out.get(f"partner_{base}"), errors="coerce")
        out[f"pair_{base}_mean"] = pd.concat([a, b], axis=1).mean(axis=1)
        out[f"pair_{base}_max"] = pd.concat([a, b], axis=1).max(axis=1)
        out[f"pair_{base}_min"] = pd.concat([a, b], axis=1).min(axis=1)
    out["pair_corner_type_low_sample_count"] = (
        pd.to_numeric(out.get("anchor_corner_type_low_sample_flag_candidate"), errors="coerce").fillna(1.0)
        + pd.to_numeric(out.get("partner_corner_type_low_sample_flag_candidate"), errors="coerce").fillna(1.0)
    )
    out["pair_course_corner_type"] = out.get("anchor_course_corner_type", "").astype("string").fillna("")
    return out


def max_drawdown_by_race(tickets: pd.DataFrame) -> float:
    if tickets.empty:
        return 0.0
    race = tickets.groupby("race_id", sort=False).agg(stake=("stake_yen", "sum"), ret=("return_yen", "sum"))
    profit = race["ret"].fillna(0.0) - race["stake"].fillna(0.0)
    equity = profit.cumsum()
    dd = equity - equity.cummax()
    return float(dd.min()) if len(dd) else 0.0


def metric(frame: pd.DataFrame, label: str, factor: str = "", condition: str = "") -> dict[str, Any]:
    if frame.empty:
        return {
            "label": label,
            "factor": factor,
            "condition": condition,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
        }
    stake = pd.to_numeric(frame["stake_yen"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(frame["return_yen"], errors="coerce").fillna(0.0)
    by_race = frame.assign(_ret=ret, _stake=stake).groupby("race_id", sort=False).agg(
        ret=("_ret", "sum"), stake=("_stake", "sum")
    )
    return {
        "label": label,
        "factor": factor,
        "condition": condition,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(ret.sum() - stake.sum()),
        "roi": float(ret.sum() / stake.sum()) if stake.sum() else 0.0,
        "ticket_hit_rate": float(ret.gt(0).mean()) if len(frame) else 0.0,
        "race_hit_rate": float(by_race["ret"].gt(0).mean()) if len(by_race) else 0.0,
        "max_drawdown_yen": max_drawdown_by_race(frame),
    }


def segment_by_quantiles(df: pd.DataFrame, value_col: str, label_prefix: str, variant: str) -> list[dict[str, Any]]:
    part = df[df["variant"].eq(variant)].copy() if "variant" in df.columns else df.copy()
    part = part[pd.to_numeric(part[value_col], errors="coerce").notna()].copy()
    if part.empty:
        return []
    values = pd.to_numeric(part[value_col], errors="coerce")
    q33, q66 = values.quantile([0.33, 0.66]).tolist()
    rows = [
        metric(part, f"{variant}_all", label_prefix, "all"),
        metric(part[values <= q33], f"{variant}_low", label_prefix, f"<=p33 {q33:.3f}"),
        metric(part[(values > q33) & (values < q66)], f"{variant}_mid", label_prefix, f"p33-p66 {q33:.3f}-{q66:.3f}"),
        metric(part[values >= q66], f"{variant}_high", label_prefix, f">=p66 {q66:.3f}"),
        metric(part[values >= values.quantile(0.80)], f"{variant}_top20", label_prefix, f">=p80 {values.quantile(0.80):.3f}"),
        metric(part[values <= values.quantile(0.20)], f"{variant}_bottom20", label_prefix, f"<=p20 {values.quantile(0.20):.3f}"),
    ]
    return rows


def compare_candidate_gates(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    part = df[df["variant"].eq(variant)].copy() if "variant" in df.columns else df.copy()
    if part.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    rows.append(metric(part, f"{variant}_base", "gate_combo", "base"))
    specs = [
        ("fast_clock", "pair_fast_clock_tolerance_candidate_mean", 0.33, "drop_bottom33"),
        ("fast_clock", "pair_fast_clock_tolerance_candidate_mean", 0.50, "keep_top50"),
        ("early_start", "pair_early_start_score_candidate_max", 0.33, "drop_bottom33"),
        ("early_start", "pair_early_start_score_candidate_max", 0.50, "keep_top50"),
        ("second_leg", "pair_second_leg_score_candidate_max", 0.33, "drop_bottom33"),
        ("corner_type", "pair_corner_type_fit_score_candidate_mean", 0.33, "drop_bottom33"),
        ("corner_low_sample", "pair_corner_type_low_sample_count", 0.0, "both_have_corner_sample"),
    ]
    for factor, col, threshold, mode in specs:
        v = pd.to_numeric(part[col], errors="coerce")
        if mode == "drop_bottom33":
            cut = v.quantile(threshold)
            keep = part[v.gt(cut)]
            condition = f"{col}>p{int(threshold*100)} {cut:.3f}"
        elif mode == "keep_top50":
            cut = v.quantile(threshold)
            keep = part[v.ge(cut)]
            condition = f"{col}>=p{int(threshold*100)} {cut:.3f}"
        elif mode == "both_have_corner_sample":
            keep = part[v.le(0.0)]
            condition = "pair_corner_type_low_sample_count==0"
        else:
            continue
        row = metric(keep, f"{variant}_{factor}_{mode}", factor, condition)
        row["kept_rate"] = float(len(keep) / len(part)) if len(part) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze fast-clock, early-start, second-leg, and corner-count aptitude candidates.")
    parser.add_argument("--feature-csv", action="append", default=[])
    parser.add_argument("--tickets-csv", default=DEFAULT_TICKETS)
    parser.add_argument("--universe-csv", default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-dir", default="outputs/analysis/speed_start_corner_factor_candidates_v1")
    parser.add_argument("--focus-variant", default="baseline")
    args = parser.parse_args()

    feature_paths = args.feature_csv or DEFAULT_FEATURES
    horse_features = load_feature_rows(feature_paths)
    tickets = read_csv_any(project_path(args.tickets_csv), dtype={"race_id": str})
    tickets["race_id"] = tickets["race_id"].astype(str).str.zfill(16)
    enriched = enrich_pairs(tickets, horse_features)

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_dir / "walkforward_selected_tickets_enriched.csv", index=False, encoding="utf-8-sig")

    factor_cols = [
        ("fast_clock", "pair_fast_clock_tolerance_candidate_mean"),
        ("early_start", "pair_early_start_score_candidate_max"),
        ("second_leg", "pair_second_leg_score_candidate_max"),
        ("corner_type", "pair_corner_type_fit_score_candidate_mean"),
    ]
    segment_rows: list[dict[str, Any]] = []
    variants = [args.focus_variant]
    if "variant" in enriched.columns:
        for variant in ["favorite_guard", "partner_floor", "dynamic_ticket_strict"]:
            if variant in set(enriched["variant"].dropna().astype(str)):
                variants.append(variant)
    variants = list(dict.fromkeys(variants))
    for variant in variants:
        for factor, col in factor_cols:
            if col in enriched.columns:
                segment_rows.extend(segment_by_quantiles(enriched, col, factor, variant))
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(out_dir / "factor_quantile_segments.csv", index=False, encoding="utf-8-sig")

    gates = pd.concat([compare_candidate_gates(enriched, variant) for variant in variants], ignore_index=True, sort=False)
    gates.to_csv(out_dir / "candidate_gate_comparison.csv", index=False, encoding="utf-8-sig")

    course_rows: list[dict[str, Any]] = []
    focus = enriched[enriched["variant"].eq(args.focus_variant)].copy() if "variant" in enriched.columns else enriched.copy()
    if "pair_course_corner_type" in focus.columns:
        for corner_type, part in focus.groupby("pair_course_corner_type", dropna=False):
            course_rows.append(metric(part, f"{args.focus_variant}_{corner_type}", "course_corner_type", str(corner_type)))
    course_summary = pd.DataFrame(course_rows)
    course_summary.to_csv(out_dir / "course_corner_type_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "horse_feature_rows": int(len(horse_features)),
        "ticket_rows": int(len(tickets)),
        "enriched_ticket_rows": int(len(enriched)),
        "focus_variant": args.focus_variant,
        "base_metric": metric(focus, f"{args.focus_variant}_base"),
        "best_gate_rows": (
            gates.sort_values(["roi", "races"], ascending=[False, False]).head(10).to_dict(orient="records")
            if not gates.empty
            else []
        ),
        "outputs": {
            "enriched_tickets": str(out_dir / "walkforward_selected_tickets_enriched.csv"),
            "factor_segments": str(out_dir / "factor_quantile_segments.csv"),
            "gate_comparison": str(out_dir / "candidate_gate_comparison.csv"),
            "course_summary": str(out_dir / "course_corner_type_summary.csv"),
        },
        "notes": [
            "Derived first-corner and corner-type features use shifted horse history only; current-race corner results are not used for the current row score.",
            "This is a small-sample post-selection diagnostic, not final adoption evidence.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
