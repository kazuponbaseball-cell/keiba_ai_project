from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "outputs/analysis/content_bridge_member_features_v1"
DEFAULT_TICKETS = ROOT / "outputs/analysis/optimized_stack_ab_s_gate_v1/s_priority_selected_tickets.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/time_value_refinement_candidates_v1"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def available_usecols(path: Path, wanted: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    return [c for c in wanted if c in header.columns]


def normalize_date(value: object) -> str:
    if pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 6:
        return "20" + digits
    if len(digits) >= 8:
        return digits[:8]
    return ""


def num(s: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if s is None:
        return pd.Series(default, index=index if index is not None else None)
    return pd.to_numeric(s, errors="coerce")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def pct_rank_train(train: pd.Series, values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ref = pd.to_numeric(train, errors="coerce").dropna().sort_values().to_numpy()
    vals = pd.to_numeric(values, errors="coerce")
    if len(ref) == 0:
        out = pd.Series(0.5, index=values.index)
    else:
        fill = float(np.nanmedian(ref))
        out = pd.Series(np.searchsorted(ref, vals.fillna(fill).to_numpy(), side="right") / len(ref), index=values.index)
    out = out.clip(0.0, 1.0)
    return out if higher_is_better else 1.0 - out


def rank_in_race(df: pd.DataFrame, col: str, higher_is_better: bool = True) -> pd.Series:
    values = num(df.get(col), df.index)
    counts = values.notna().groupby(df["race_id"]).transform("sum")
    ranks = values.groupby(df["race_id"]).rank(ascending=not higher_is_better, method="average")
    return ((counts - ranks) / (counts - 1)).replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0.0, 1.0)


def gap_to_best_in_race(df: pd.DataFrame, col: str) -> pd.Series:
    values = num(df.get(col), df.index)
    best = values.groupby(df["race_id"]).transform("max")
    spread = (best - values.groupby(df["race_id"]).transform("min")).replace(0, np.nan)
    return (1.0 - ((best - values) / spread)).replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0.0, 1.0)


def roc_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    mask = y.notna() & s.notna()
    y = y.loc[mask].astype(int)
    s = s.loc[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s.rank(method="average")
    pos_rank_sum = float(ranks.loc[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_runner_features() -> pd.DataFrame:
    wanted = [
        "レースID(新/馬番無)",
        "日付",
        "日付S",
        "場所",
        "Ｒ",
        "馬番",
        "人気",
        "単勝オッズ",
        "芝・ダ",
        "距離",
        "クラス名",
        "馬場状態",
        "頭数",
        "出走頭数",
        "確定着順",
        "target_score",
        "単勝配当",
        "複勝配当",
        "prev_race_time_value",
        "prev_time_z_course_distance",
        "prev_time_adjusted_by_day_bias",
        "prev_class_time_value_score",
        "past3_avg_time_value",
        "past3_best_time_value",
        "past3_avg_time_z",
        "past3_avg_time_adjusted_by_day_bias",
        "horse_time_value_plus_margin",
        "past3_avg_score",
        "same_distance_category_avg_score",
        "same_venue_avg_score",
        "horse_turf_avg_score",
        "horse_dirt_avg_score",
        "race_early_pressure_score",
        "front_advantage_score",
        "pace_fit_score",
        "prev_lap_pace_index",
        "prev_lap_finish_index",
        "prev_lap_sustain_index",
        "prev_lap_long_spurt_index",
        "horse_fast_lap_score_past5",
        "horse_slow_lap_score_past5",
        "horse_instant_lap_score_past5",
        "horse_sustain_lap_score_past5",
        "horse_long_spurt_lap_score_past5",
        "lap_aptitude_fit_score",
        "lap_aptitude_reliability_score",
    ]
    frames: list[pd.DataFrame] = []
    for source in ["train", "test"]:
        path = CONTENT_DIR / f"{source}_features_with_content_bridge.csv"
        usecols = available_usecols(path, wanted)
        df = read_csv(path, dtype=str, usecols=usecols)
        df["source"] = source
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"レースID(新/馬番無)": "race_id"})
    out["race_id"] = out["race_id"].astype(str)
    out["date_key"] = out.get("日付", pd.Series("", index=out.index)).map(normalize_date)
    missing = out["date_key"].eq("") & out.get("日付S", pd.Series("", index=out.index)).notna()
    out.loc[missing, "date_key"] = out.loc[missing, "日付S"].map(normalize_date)
    out["horse_no"] = num(out.get("馬番"), out.index)
    out["race_no"] = num(out.get("Ｒ"), out.index)
    out["popularity"] = num(out.get("人気"), out.index)
    out["odds"] = num(out.get("単勝オッズ"), out.index)
    out["finish"] = num(out.get("確定着順"), out.index)
    out["field_size"] = num(out.get("出走頭数"), out.index).fillna(num(out.get("頭数"), out.index))
    out["is_win"] = out["finish"].eq(1)
    out["is_top3"] = out["finish"].le(3)
    out["win_return_100"] = np.where(out["finish"].eq(1), num(out.get("単勝配当"), out.index, 0.0).fillna(0.0), 0.0)
    out["place_return_100"] = np.where(out["finish"].le(3), num(out.get("複勝配当"), out.index, 0.0).fillna(0.0), 0.0)
    return out


def add_time_refinement_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    train = out["source"].eq("train")
    idx = out.index

    prev_time = num(out.get("prev_race_time_value"), idx)
    prev_bias = num(out.get("prev_time_adjusted_by_day_bias"), idx)
    prev_class = num(out.get("prev_class_time_value_score"), idx)
    avg3 = num(out.get("past3_avg_time_value"), idx)
    best3 = num(out.get("past3_best_time_value"), idx)
    avg3_z = num(out.get("past3_avg_time_z"), idx)
    avg3_bias = num(out.get("past3_avg_time_adjusted_by_day_bias"), idx)
    time_plus_margin = num(out.get("horse_time_value_plus_margin"), idx)

    safe_parts = [
        pct_rank_train(out.loc[train, "prev_class_time_value_score"], prev_class, True),
        pct_rank_train(out.loc[train, "prev_time_adjusted_by_day_bias"], prev_bias, True),
        pct_rank_train(out.loc[train, "past3_avg_time_value"], avg3, True),
        pct_rank_train(out.loc[train, "past3_avg_time_adjusted_by_day_bias"], avg3_bias, True),
    ]
    out["time_value_safe_composite"] = pd.concat(safe_parts, axis=1).mean(axis=1)
    out["recency_weighted_time_score"] = (
        0.38 * pct_rank_train(out.loc[train, "prev_class_time_value_score"], prev_class, True)
        + 0.27 * pct_rank_train(out.loc[train, "prev_time_adjusted_by_day_bias"], prev_bias, True)
        + 0.23 * pct_rank_train(out.loc[train, "past3_avg_time_value"], avg3, True)
        + 0.12 * pct_rank_train(out.loc[train, "past3_avg_time_z"], avg3_z, True)
    ).clip(0.0, 1.0)

    best_gap = best3 - avg3
    out["best_time_gap_raw"] = best_gap
    best_rank = pct_rank_train(out.loc[train, "past3_best_time_value"], best3, True)
    gap_risk = pct_rank_train(best_gap.loc[train], best_gap, True)
    out["best_time_reproducibility"] = (0.62 * best_rank + 0.38 * (1.0 - gap_risk)).clip(0.0, 1.0)
    out["time_score_consistency"] = (
        0.62 * pct_rank_train(out.loc[train, "past3_avg_time_value"], avg3, True)
        + 0.38 * (1.0 - gap_risk)
    ).clip(0.0, 1.0)

    surface = out.get("芝・ダ", pd.Series("", index=idx)).fillna("").astype(str)
    surface_score = np.where(
        surface.eq("芝"),
        num(out.get("horse_turf_avg_score"), idx),
        np.where(surface.eq("ダ"), num(out.get("horse_dirt_avg_score"), idx), np.nan),
    )
    surface_score = pd.Series(surface_score, index=idx)
    condition_support = pd.concat(
        [
            num(out.get("same_distance_category_avg_score"), idx),
            num(out.get("same_venue_avg_score"), idx),
            surface_score,
            num(out.get("lap_aptitude_fit_score"), idx),
        ],
        axis=1,
    ).mean(axis=1)
    condition_support_rank = pct_rank_train(condition_support.loc[train], condition_support, True)
    out["condition_matched_time_score"] = (
        out["time_value_safe_composite"] * (0.58 + 0.42 * condition_support_rank)
    ).clip(0.0, 1.0)

    for col in [
        "prev_class_time_value_score",
        "prev_time_adjusted_by_day_bias",
        "past3_avg_time_value",
        "past3_best_time_value",
        "past3_avg_time_z",
        "horse_time_value_plus_margin",
    ]:
        if col in out.columns:
            out[f"{col}_race_rank"] = rank_in_race(out, col, True)
            out[f"{col}_gap_to_best"] = gap_to_best_in_race(out, col)

    out["time_value_relative_rank_score"] = pd.concat(
        [
            out.get("prev_class_time_value_score_race_rank"),
            out.get("past3_avg_time_value_race_rank"),
            out.get("past3_best_time_value_race_rank"),
            out.get("horse_time_value_plus_margin_race_rank"),
        ],
        axis=1,
    ).mean(axis=1)
    out["time_value_gap_to_best_score"] = pd.concat(
        [
            out.get("prev_class_time_value_score_gap_to_best"),
            out.get("past3_avg_time_value_gap_to_best"),
            out.get("past3_best_time_value_gap_to_best"),
        ],
        axis=1,
    ).mean(axis=1)

    race_mean_time = out["time_value_safe_composite"].groupby(out["race_id"]).transform("mean")
    race_top_time = out["time_value_safe_composite"].groupby(out["race_id"]).transform("max")
    pressure = pct_rank_train(
        num(out.loc[train, "race_early_pressure_score"], out.loc[train].index),
        num(out.get("race_early_pressure_score"), idx),
        True,
    )
    front_adv = pct_rank_train(
        num(out.loc[train, "front_advantage_score"], out.loc[train].index),
        num(out.get("front_advantage_score"), idx),
        True,
    )
    out["today_fast_clock_likelihood"] = (
        0.38 * race_mean_time.fillna(0.5)
        + 0.28 * race_top_time.fillna(0.5)
        + 0.22 * pressure.fillna(0.5)
        + 0.12 * front_adv.fillna(0.5)
    ).clip(0.0, 1.0)
    fast_lap = pct_rank_train(out.loc[train, "horse_fast_lap_score_past5"], num(out.get("horse_fast_lap_score_past5"), idx), True)
    out["fast_clock_x_today_likelihood"] = (fast_lap * out["today_fast_clock_likelihood"]).clip(0.0, 1.0)

    early_parts = [
        pct_rank_train(out.loc[train, "prev_lap_pace_index"], num(out.get("prev_lap_pace_index"), idx), True),
        pct_rank_train(out.loc[train, "horse_fast_lap_score_past5"], num(out.get("horse_fast_lap_score_past5"), idx), True),
    ]
    late_parts = [
        pct_rank_train(out.loc[train, "prev_lap_finish_index"], num(out.get("prev_lap_finish_index"), idx), True),
        pct_rank_train(out.loc[train, "horse_instant_lap_score_past5"], num(out.get("horse_instant_lap_score_past5"), idx), True),
    ]
    sustain_parts = [
        pct_rank_train(out.loc[train, "prev_lap_sustain_index"], num(out.get("prev_lap_sustain_index"), idx), True),
        pct_rank_train(out.loc[train, "prev_lap_long_spurt_index"], num(out.get("prev_lap_long_spurt_index"), idx), True),
        pct_rank_train(out.loc[train, "horse_sustain_lap_score_past5"], num(out.get("horse_sustain_lap_score_past5"), idx), True),
        pct_rank_train(out.loc[train, "horse_long_spurt_lap_score_past5"], num(out.get("horse_long_spurt_lap_score_past5"), idx), True),
    ]
    out["pace_tracking_score"] = pd.concat(early_parts, axis=1).mean(axis=1)
    out["late_speed_value"] = pd.concat(late_parts, axis=1).mean(axis=1)
    out["sustained_speed_score"] = pd.concat(sustain_parts, axis=1).mean(axis=1)
    out["time_refinement_composite"] = (
        0.30 * out["condition_matched_time_score"]
        + 0.24 * out["time_value_relative_rank_score"]
        + 0.18 * out["recency_weighted_time_score"]
        + 0.14 * out["best_time_reproducibility"]
        + 0.14 * out["fast_clock_x_today_likelihood"]
    ).clip(0.0, 1.0)
    return out


FEATURES = [
    "time_value_safe_composite",
    "recency_weighted_time_score",
    "best_time_reproducibility",
    "time_score_consistency",
    "condition_matched_time_score",
    "time_value_relative_rank_score",
    "time_value_gap_to_best_score",
    "today_fast_clock_likelihood",
    "fast_clock_x_today_likelihood",
    "pace_tracking_score",
    "late_speed_value",
    "sustained_speed_score",
    "time_refinement_composite",
    "prev_class_time_value_score_race_rank",
    "past3_avg_time_value_race_rank",
    "past3_best_time_value_race_rank",
]


def summarize_runner(df: pd.DataFrame, mask: pd.Series, segment: str) -> dict[str, object]:
    seg = df.loc[mask.fillna(False)].copy()
    if seg.empty:
        return {"segment": segment, "races": 0, "horses": 0, "win_rate": np.nan, "top3_rate": np.nan, "win_roi": np.nan, "place_roi": np.nan}
    stake = len(seg) * 100.0
    return {
        "segment": segment,
        "races": int(seg["race_id"].nunique()),
        "horses": int(len(seg)),
        "win_rate": float(seg["is_win"].mean()),
        "top3_rate": float(seg["is_top3"].mean()),
        "win_roi": safe_div(float(seg["win_return_100"].sum()), stake),
        "place_roi": safe_div(float(seg["place_return_100"].sum()), stake),
        "avg_popularity": float(seg["popularity"].mean()),
        "avg_odds": float(seg["odds"].mean()),
    }


def evaluate_runner_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, group in df.groupby("source", sort=False):
        group = group[group["finish"].notna()].copy()
        for feature in FEATURES:
            values = num(group.get(feature), group.index)
            valid = values.notna()
            if valid.sum() < 300:
                continue
            q90 = values.loc[valid].quantile(0.90)
            q80 = values.loc[valid].quantile(0.80)
            q20 = values.loc[valid].quantile(0.20)
            base = {
                "source": source,
                "feature": feature,
                "valid_rows": int(valid.sum()),
                "auc_top3": roc_auc(group.loc[valid, "is_top3"], values.loc[valid]),
                "auc_win": roc_auc(group.loc[valid, "is_win"], values.loc[valid]),
            }
            for label, mask in [
                ("top10pct", values.ge(q90)),
                ("top20pct", values.ge(q80)),
                ("bottom20pct", values.le(q20)),
                ("top20pct_pop4plus", values.ge(q80) & group["popularity"].ge(4)),
            ]:
                rows.append({**base, **summarize_runner(group, mask & valid, label)})
    return pd.DataFrame(rows)


def ticket_metrics(df: pd.DataFrame, name: str) -> dict[str, object]:
    if df.empty:
        return {"policy": name, "tickets": 0, "races": 0, "stake_yen": 0.0, "return_yen": 0.0, "roi": np.nan, "hit_rate": np.nan}
    stake = num(df.get("runtime_stake_yen"), df.index).fillna(num(df.get("stake_yen"), df.index, 0.0)).fillna(0.0)
    ret = num(df.get("runtime_return_yen"), df.index).fillna(num(df.get("return_yen"), df.index, 0.0)).fillna(0.0)
    return {
        "policy": name,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "roi": safe_div(float(ret.sum()), float(stake.sum())),
        "hit_rate": float(ret.gt(0).mean()),
    }


def load_tickets_with_runner_features(runners: pd.DataFrame, tickets_path: Path) -> pd.DataFrame:
    tickets = read_csv(tickets_path, dtype=str)
    tickets["race_id"] = tickets["race_id"].astype(str)
    tickets["anchor_no"] = num(tickets.get("anchor_no"), tickets.index)
    tickets["partner_no"] = num(tickets.get("partner_no"), tickets.index)
    cols = ["race_id", "horse_no", "source", *FEATURES]
    r = runners[cols].copy()
    r["_source_order"] = r["source"].map({"test": 0, "train": 1}).fillna(9)
    r = r.sort_values("_source_order").drop_duplicates(["race_id", "horse_no"], keep="first")
    anchor = r.add_prefix("anchor_").rename(columns={"anchor_race_id": "race_id", "anchor_horse_no": "anchor_no"})
    partner = r.add_prefix("partner_").rename(columns={"partner_race_id": "race_id", "partner_horse_no": "partner_no"})
    out = tickets.merge(anchor, on=["race_id", "anchor_no"], how="left").merge(
        partner, on=["race_id", "partner_no"], how="left"
    )
    for feature in FEATURES:
        a = num(out.get(f"anchor_{feature}"), out.index)
        p = num(out.get(f"partner_{feature}"), out.index)
        out[f"pair_min_{feature}"] = pd.concat([a, p], axis=1).min(axis=1)
        out[f"pair_avg_{feature}"] = pd.concat([a, p], axis=1).mean(axis=1)
        out[f"pair_max_{feature}"] = pd.concat([a, p], axis=1).max(axis=1)
        out[f"pair_gap_{feature}"] = (a - p).abs()
    return out


def evaluate_ticket_policies(tickets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = [ticket_metrics(tickets, "base_all")]
    candidates = [
        ("pair_min_condition_matched_time_score", "skip_low_condition_matched_time_q20", "low"),
        ("pair_min_time_value_relative_rank_score", "skip_low_time_relative_rank_q20", "low"),
        ("pair_avg_time_refinement_composite", "keep_high_time_refinement_top80", "high"),
        ("pair_min_fast_clock_x_today_likelihood", "skip_low_fast_clock_fit_q20", "low"),
        ("pair_max_best_time_reproducibility", "skip_low_best_repro_q20", "low"),
        ("pair_gap_recency_weighted_time_score", "skip_large_pair_time_gap_q80", "gap_high_bad"),
    ]
    for col, policy, mode in candidates:
        values = num(tickets.get(col), tickets.index)
        valid = values.notna()
        if valid.sum() < 20:
            continue
        if mode == "low":
            threshold = values.loc[valid].quantile(0.20)
            mask = values.ge(threshold) | values.isna()
        elif mode == "high":
            threshold = values.loc[valid].quantile(0.80)
            mask = values.ge(threshold)
        else:
            threshold = values.loc[valid].quantile(0.80)
            mask = values.le(threshold) | values.isna()
        rows.append({**ticket_metrics(tickets.loc[mask.fillna(False)], policy), "feature": col, "threshold": float(threshold)})
    return pd.DataFrame(rows)


def write_readme(out_dir: Path, runner_eval: pd.DataFrame, ticket_eval: pd.DataFrame, runners: pd.DataFrame) -> None:
    def fmt_pct(x: float) -> str:
        return "" if pd.isna(x) else f"{x * 100:.1f}%"

    top = runner_eval[(runner_eval["source"] == "test") & (runner_eval["segment"] == "top20pct")].copy()
    top = top.sort_values(["place_roi", "auc_top3"], ascending=False).head(10)
    lines = [
        "# Time Value Refinement Candidate Check",
        "",
        "External advice focused on turning raw time value into condition-fit, race-relative, reproducible, and pace-decomposed signals.",
        "",
        "## Coverage",
        f"- runner rows: {len(runners):,}",
        f"- test rows: {(runners['source'] == 'test').sum():,}",
        f"- races: {runners['race_id'].nunique():,}",
        "",
        "## Best Runner Segments (test top20%)",
        "| feature | horses | races | top3 | win ROI | place ROI | AUC top3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['feature']} | {int(r['horses'])} | {int(r['races'])} | {fmt_pct(r['top3_rate'])} | "
            f"{fmt_pct(r['win_roi'])} | {fmt_pct(r['place_roi'])} | {r['auc_top3']:.3f} |"
        )
    lines += [
        "",
        "## Strongest Ticket Overlay Policies",
        "| policy | tickets | races | ROI | hit | stake | return |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in ticket_eval.iterrows():
        lines.append(
            f"| {r['policy']} | {int(r['tickets'])} | {int(r['races'])} | {fmt_pct(r['roi'])} | "
            f"{fmt_pct(r['hit_rate'])} | {r['stake_yen']:.0f} | {r['return_yen']:.0f} |"
        )
    lines += [
        "",
        "## Recommendation",
        "- Race-relative time ranking and condition-matched time are implementable now.",
        "- Adopt only if ticket overlay improves ROI or gives a clear shadow label; do not blindly add all time derivatives.",
        "- Best-time-only boosts should be treated as risky unless reproducibility is also high.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate time-value refinement features for ROI-oriented horse racing AI.")
    parser.add_argument("--tickets-csv", default=str(DEFAULT_TICKETS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runners = add_time_refinement_features(load_runner_features())
    runner_eval = evaluate_runner_features(runners)
    tickets = load_tickets_with_runner_features(runners, Path(args.tickets_csv))
    ticket_eval = evaluate_ticket_policies(tickets)

    keep_cols = [
        "source",
        "race_id",
        "date_key",
        "場所",
        "Ｒ",
        "horse_no",
        "popularity",
        "odds",
        "finish",
        "is_win",
        "is_top3",
        *FEATURES,
    ]
    runners[keep_cols].to_csv(out_dir / "runner_time_refinement_features.csv", index=False, encoding="utf-8-sig")
    runner_eval.to_csv(out_dir / "runner_feature_segments.csv", index=False, encoding="utf-8-sig")
    tickets.to_csv(out_dir / "s_priority_tickets_with_time_refinement.csv", index=False, encoding="utf-8-sig")
    ticket_eval.to_csv(out_dir / "ticket_policy_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(out_dir),
        "coverage": {
            "runner_rows": int(len(runners)),
            "test_rows": int((runners["source"] == "test").sum()),
            "train_rows": int((runners["source"] == "train").sum()),
            "races": int(runners["race_id"].nunique()),
            "date_min": str(runners["date_key"].min()),
            "date_max": str(runners["date_key"].max()),
        },
        "runner_test_top20_by_place_roi": runner_eval[
            (runner_eval["source"] == "test") & (runner_eval["segment"] == "top20pct")
        ]
        .sort_values(["place_roi", "auc_top3"], ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "ticket_policy_metrics": ticket_eval.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(out_dir, runner_eval, ticket_eval, runners)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
