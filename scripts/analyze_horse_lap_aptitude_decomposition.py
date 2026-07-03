from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUNNER_LAP = (
    ROOT / "outputs" / "analysis" / "lap_pair_refinement_candidates_v1" / "runner_lap_pair_refinement_features.csv"
)
DEFAULT_TICKET_SETS = [
    (
        "lap_positive_selected",
        ROOT / "outputs" / "analysis" / "lap_positive_expansion_v1" / "lap_positive_expansion_selected_tickets.csv",
    ),
    (
        "mcs_runtime_recommended",
        ROOT
        / "outputs"
        / "analysis"
        / "mcs_pbo_runtime_overlay_v4_operational_gates"
        / "recommended_runtime_tickets.csv",
    ),
]
DEFAULT_OUT = ROOT / "outputs" / "analysis" / "horse_lap_aptitude_decomposition_v1"

AXES = ["fast", "slow", "instant", "sustain", "long_spurt"]
AXIS_LABELS = {
    "fast": "前半負荷型",
    "slow": "スロー瞬発寄り",
    "instant": "瞬発型",
    "sustain": "持続型",
    "long_spurt": "ロングスパート型",
}


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def ncol(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = df[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def text_col(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="string")
    return df[col].astype("string").fillna(default).str.replace(r"\.0$", "", regex=True).str.strip()


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip()


def available_usecols(path: Path, wanted: list[str]) -> list[str]:
    if not path.exists():
        return []
    header = read_csv(path, nrows=0)
    return [c for c in wanted if c in header.columns]


def normalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    row_sum = values.sum(axis=1)
    values.loc[row_sum.le(0), :] = 1.0
    row_sum = values.sum(axis=1).replace(0.0, np.nan)
    return values.div(row_sum, axis=0).fillna(1.0 / max(len(values.columns), 1))


def cosine(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    cols = [c for c in AXES if c in a.columns and c in b.columns]
    if not cols:
        return pd.Series(0.5, index=a.index)
    av = a[cols].to_numpy(dtype=float)
    bv = b[cols].to_numpy(dtype=float)
    denom = np.linalg.norm(av, axis=1) * np.linalg.norm(bv, axis=1)
    dot = (av * bv).sum(axis=1)
    out = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    return pd.Series(out, index=a.index).clip(0.0, 1.0)


def top_axis(values: pd.DataFrame) -> pd.Series:
    clean = values[[c for c in AXES if c in values.columns]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if clean.empty:
        return pd.Series("unknown", index=values.index)
    return clean.idxmax(axis=1).where(clean.max(axis=1).gt(0), "unknown")


def top_axis_strength(values: pd.DataFrame) -> pd.Series:
    clean = values[[c for c in AXES if c in values.columns]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if clean.empty:
        return pd.Series(0.0, index=values.index)
    arr = np.sort(clean.to_numpy(dtype=float), axis=1)
    if arr.shape[1] == 1:
        return pd.Series(arr[:, 0], index=values.index).clip(0.0, 1.0)
    return pd.Series(arr[:, -1] - arr[:, -2], index=values.index).clip(0.0, 1.0)


def load_runner_lap(path: Path) -> pd.DataFrame:
    wanted = [
        "source",
        "race_id",
        "horse_no",
        "馬名",
        "popularity",
        "odds",
        "finish",
        "is_win",
        "is_top3",
        "predicted_lap_mode",
        "actual_lap_mode_diagnostic",
        "lap_mode_prediction_hit_diagnostic",
        "horse_lap_profile_top_mode",
        "race_need_fast",
        "race_need_slow",
        "race_need_instant",
        "race_need_sustain",
        "race_need_long_spurt",
        "horse_lap_fast",
        "horse_lap_slow",
        "horse_lap_instant",
        "horse_lap_sustain",
        "horse_lap_long_spurt",
        "lap_profile_fit_score",
        "lap_profile_fit_rank_in_race",
        "lap_fit_confident_score",
        "lap_axis_candidate_score",
        "lap_partner_specialist_score",
        "lap_mismatch_popular_risk",
        "race_lap_prediction_confidence",
        "race_lap_profile_concentration",
        "horse_lap_profile_sharpness",
    ]
    usecols = available_usecols(path, wanted)
    if not {"race_id", "horse_no"}.issubset(usecols):
        raise FileNotFoundError(f"runner lap file is missing required columns: {path}")
    df = read_csv(path, usecols=usecols)
    df["race_id"] = normalize_id(df["race_id"])
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    for col in wanted:
        if col in {"source", "race_id", "horse_no", "馬名", "predicted_lap_mode", "actual_lap_mode_diagnostic", "horse_lap_profile_top_mode"}:
            continue
        if col in df.columns:
            df[col] = ncol(df, col)

    need = normalize_rows(
        pd.DataFrame({axis: ncol(df, f"race_need_{axis}", 0.0).fillna(0.0) for axis in AXES}, index=df.index)
    )
    horse = normalize_rows(
        pd.DataFrame({axis: ncol(df, f"horse_lap_{axis}", 0.0).fillna(0.0) for axis in AXES}, index=df.index)
    )
    df["computed_race_lap_mode"] = top_axis(need)
    df["computed_horse_lap_type"] = top_axis(horse)
    df["horse_lap_type_strength"] = top_axis_strength(horse)
    df["race_need_concentration"] = top_axis_strength(need)
    df["computed_lap_fit_cosine"] = cosine(need, horse)
    df["horse_lap_type_label"] = df["computed_horse_lap_type"].map(AXIS_LABELS).fillna("不明")
    return df.drop_duplicates(["race_id", "horse_no"], keep="last")


def normalize_tickets(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = dataset
    out["race_id"] = normalize_id(out["race_id"])
    if "ticket_type" not in out.columns:
        out["ticket_type"] = "unknown"
    out["ticket_type"] = out["ticket_type"].astype("string").fillna("unknown").str.lower()
    if "anchor_no" not in out.columns:
        out["anchor_no"] = ncol(out, "horse_a")
    if "partner_no" not in out.columns:
        out["partner_no"] = ncol(out, "horse_b")
    out["anchor_no"] = pd.to_numeric(out["anchor_no"], errors="coerce").astype("Int64")
    out["partner_no"] = pd.to_numeric(out["partner_no"], errors="coerce").astype("Int64")
    if "year" not in out.columns:
        out["year"] = out["race_id"].str.slice(0, 4)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")

    if "eval_stake_yen" in out.columns and "eval_return_yen" in out.columns:
        out["stake_yen_eval"] = ncol(out, "eval_stake_yen", 100.0).fillna(100.0)
        out["return_yen_eval"] = ncol(out, "eval_return_yen", 0.0).fillna(0.0)
    elif "stake_yen" in out.columns and "return_yen" in out.columns:
        out["stake_yen_eval"] = ncol(out, "stake_yen", 100.0).fillna(100.0)
        out["return_yen_eval"] = ncol(out, "return_yen", 0.0).fillna(0.0)
    elif "runtime_stake_yen" in out.columns and "runtime_return_yen" in out.columns:
        out["stake_yen_eval"] = ncol(out, "runtime_stake_yen", 100.0).fillna(100.0)
        out["return_yen_eval"] = ncol(out, "runtime_return_yen", 0.0).fillna(0.0)
    else:
        out["stake_yen_eval"] = 100.0
        out["return_yen_eval"] = 0.0

    if "hit_eval" in out.columns:
        out["hit_eval"] = ncol(out, "hit_eval", 0.0).fillna(0.0).gt(0)
    elif "hit" in out.columns:
        hit = out["hit"]
        if pd.api.types.is_bool_dtype(hit):
            out["hit_eval"] = hit.fillna(False)
        else:
            out["hit_eval"] = hit.astype("string").str.lower().isin(["true", "1", "yes"])
    else:
        out["hit_eval"] = out["return_yen_eval"].gt(0)
    return out


def add_runner_sides(tickets: pd.DataFrame, runner: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    keep_runner = runner.copy()
    for side, no_col in [("anchor", "anchor_no"), ("partner", "partner_no")]:
        rename = {"horse_no": no_col}
        for col in keep_runner.columns:
            if col in {"race_id", "horse_no"}:
                continue
            rename[col] = f"{side}_{col}"
        side_frame = keep_runner.rename(columns=rename)
        keep = ["race_id", no_col, *[c for c in rename.values() if c not in {"race_id", no_col}]]
        keep = list(dict.fromkeys([c for c in keep if c in side_frame.columns]))
        out = out.merge(side_frame[keep], on=["race_id", no_col], how="left")
    return out


def mode_series_from_ticket(df: pd.DataFrame) -> pd.Series:
    for col in ["v2_predicted_lap_mode", "anchor_predicted_lap_mode", "partner_predicted_lap_mode"]:
        if col in df.columns:
            s = text_col(df, col)
            if s.ne("").any():
                return s.where(s.ne(""), "unknown")
    needs = pd.DataFrame(index=df.index)
    for axis in AXES:
        needs[axis] = pd.concat(
            [
                ncol(df, f"anchor_race_need_{axis}", np.nan),
                ncol(df, f"partner_race_need_{axis}", np.nan),
            ],
            axis=1,
        ).mean(axis=1)
    return top_axis(normalize_rows(needs)).where(needs.notna().any(axis=1), "unknown")


def side_vector(df: pd.DataFrame, side: str, prefix: str = "horse_lap") -> pd.DataFrame:
    return normalize_rows(pd.DataFrame({axis: ncol(df, f"{side}_{prefix}_{axis}", 0.0).fillna(0.0) for axis in AXES}, index=df.index))


def add_lap_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_lap_mode"] = mode_series_from_ticket(out)
    out["race_lap_mode_label"] = out["race_lap_mode"].map(AXIS_LABELS).fillna("不明")
    for side in ["anchor", "partner"]:
        top_col = f"{side}_horse_lap_profile_top_mode"
        computed_col = f"{side}_computed_horse_lap_type"
        side_type = text_col(out, top_col) if top_col in out.columns else pd.Series("", index=out.index, dtype="string")
        if computed_col in out.columns:
            computed = text_col(out, computed_col)
            side_type = side_type.where(side_type.ne(""), computed)
        side_vec = side_vector(out, side)
        side_type = side_type.where(side_type.ne(""), top_axis(side_vec))
        out[f"{side}_lap_type"] = side_type.where(side_type.ne(""), "unknown")
        out[f"{side}_lap_type_label"] = out[f"{side}_lap_type"].map(AXIS_LABELS).fillna("不明")
        out[f"{side}_lap_type_strength"] = top_axis_strength(side_vec)
        out[f"{side}_lap_fit_score_eval"] = ncol(out, f"{side}_lap_profile_fit_score", np.nan).fillna(
            ncol(out, f"{side}_computed_lap_fit_cosine", np.nan)
        )
        out[f"{side}_lap_confident_score_eval"] = ncol(out, f"{side}_lap_fit_confident_score", np.nan).fillna(
            ncol(out, f"{side}_race_lap_prediction_confidence", np.nan)
        )

    out["anchor_lap_matches_race"] = out["anchor_lap_type"].eq(out["race_lap_mode"])
    out["partner_lap_matches_race"] = out["partner_lap_type"].eq(out["race_lap_mode"])
    match_count = out["anchor_lap_matches_race"].astype(int) + out["partner_lap_matches_race"].astype(int)
    out["lap_match_bucket"] = np.select(
        [match_count.eq(2), match_count.eq(1), match_count.eq(0)],
        ["both_match", "one_match", "no_match"],
        default="unknown",
    )
    out["lap_match_bucket_label"] = out["lap_match_bucket"].map(
        {"both_match": "2頭とも想定ラップ一致", "one_match": "片方一致", "no_match": "一致なし"}
    ).fillna("不明")
    pair_types = pd.concat([out["anchor_lap_type"], out["partner_lap_type"]], axis=1).astype(str)
    sorted_pair = np.sort(pair_types.to_numpy(), axis=1)
    out["lap_type_pair"] = [f"{a}+{b}" for a, b in sorted_pair]
    out["lap_type_pair_label"] = [
        f"{AXIS_LABELS.get(str(a), '不明')} + {AXIS_LABELS.get(str(b), '不明')}" for a, b in sorted_pair
    ]
    out["pair_lap_fit_min_eval"] = pd.concat(
        [ncol(out, "anchor_lap_fit_score_eval", np.nan), ncol(out, "partner_lap_fit_score_eval", np.nan)], axis=1
    ).min(axis=1)
    out["pair_lap_fit_avg_eval"] = pd.concat(
        [ncol(out, "anchor_lap_fit_score_eval", np.nan), ncol(out, "partner_lap_fit_score_eval", np.nan)], axis=1
    ).mean(axis=1)
    out["pair_lap_conf_min_eval"] = pd.concat(
        [ncol(out, "anchor_lap_confident_score_eval", np.nan), ncol(out, "partner_lap_confident_score_eval", np.nan)], axis=1
    ).min(axis=1)
    out["pair_lap_conf_avg_eval"] = pd.concat(
        [ncol(out, "anchor_lap_confident_score_eval", np.nan), ncol(out, "partner_lap_confident_score_eval", np.nan)], axis=1
    ).mean(axis=1)
    out["pair_lap_mismatch_popular_max_eval"] = pd.concat(
        [ncol(out, "anchor_lap_mismatch_popular_risk", 0.0), ncol(out, "partner_lap_mismatch_popular_risk", 0.0)],
        axis=1,
    ).max(axis=1)
    out["pair_lap_type_strength_min"] = pd.concat(
        [ncol(out, "anchor_lap_type_strength", 0.0), ncol(out, "partner_lap_type_strength", 0.0)], axis=1
    ).min(axis=1)

    race_vec = normalize_rows(
        pd.DataFrame(
            {
                axis: pd.concat(
                    [ncol(out, f"anchor_race_need_{axis}", np.nan), ncol(out, f"partner_race_need_{axis}", np.nan)],
                    axis=1,
                ).mean(axis=1)
                for axis in AXES
            },
            index=out.index,
        )
    )
    out["anchor_lap_cosine_eval"] = cosine(race_vec, side_vector(out, "anchor"))
    out["partner_lap_cosine_eval"] = cosine(race_vec, side_vector(out, "partner"))
    out["pair_lap_cosine_min_eval"] = np.minimum(out["anchor_lap_cosine_eval"], out["partner_lap_cosine_eval"])
    out["pair_lap_cosine_avg_eval"] = (out["anchor_lap_cosine_eval"] + out["partner_lap_cosine_eval"]) / 2.0
    out["pair_lap_gap_eval"] = (out["anchor_lap_cosine_eval"] - out["partner_lap_cosine_eval"]).abs()
    out["horse_lap_decomp_score"] = (
        0.34 * out["pair_lap_fit_min_eval"].fillna(0.5)
        + 0.24 * out["pair_lap_cosine_min_eval"].fillna(0.5)
        + 0.18 * out["pair_lap_conf_min_eval"].fillna(0.5)
        + 0.14 * (1.0 - out["pair_lap_mismatch_popular_max_eval"].fillna(0.0).clip(0.0, 1.0))
        + 0.10 * out["pair_lap_type_strength_min"].fillna(0.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    return out


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = pd.to_numeric(profit, errors="coerce").fillna(0.0).cumsum()
    return float((curve - curve.cummax()).min())


def roi_without_top(frame: pd.DataFrame, n: int) -> float:
    if frame.empty:
        return 0.0
    kept = frame.sort_values("return_yen_eval", ascending=False).iloc[n:]
    stake = float(pd.to_numeric(kept["stake_yen_eval"], errors="coerce").fillna(0.0).sum())
    ret = float(pd.to_numeric(kept["return_yen_eval"], errors="coerce").fillna(0.0).sum())
    return ret / stake * 100.0 if stake > 0 else 0.0


def metrics(frame: pd.DataFrame, segment: str, base_roi: float | None = None) -> dict[str, Any]:
    if frame.empty:
        return {
            "segment": segment,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": 0.0,
            "roi_lift_vs_base_pt": np.nan,
            "hit_rate_pct": 0.0,
            "race_hit_rate_pct": 0.0,
            "max_drawdown_yen": 0.0,
            "top1_removed_roi_pct": 0.0,
            "top3_removed_roi_pct": 0.0,
            "top5_removed_roi_pct": 0.0,
            "min_year_roi_pct": np.nan,
        }
    stake = pd.to_numeric(frame["stake_yen_eval"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(frame["return_yen_eval"], errors="coerce").fillna(0.0)
    profit = ret - stake
    roi = float(ret.sum() / stake.sum() * 100.0) if float(stake.sum()) > 0 else 0.0
    by_year = frame.groupby("year", dropna=True).agg(stake=("stake_yen_eval", "sum"), ret=("return_yen_eval", "sum"))
    year_roi = (by_year["ret"] / by_year["stake"] * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
    order_cols = [c for c in ["race_id", "ticket_type", "anchor_no", "partner_no"] if c in frame.columns]
    ordered_profit = frame[order_cols].copy() if order_cols else pd.DataFrame(index=frame.index)
    ordered_profit["_profit"] = profit.to_numpy()
    ordered_profit = ordered_profit.sort_values(order_cols, kind="mergesort") if order_cols else ordered_profit
    race_hit = frame.groupby("race_id")["hit_eval"].max().mean() if "race_id" in frame.columns else np.nan
    return {
        "segment": segment,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0,
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(profit.sum()),
        "roi_pct": roi,
        "roi_lift_vs_base_pt": roi - base_roi if base_roi is not None else np.nan,
        "hit_rate_pct": float(frame["hit_eval"].mean() * 100.0),
        "race_hit_rate_pct": float(race_hit * 100.0) if pd.notna(race_hit) else np.nan,
        "max_drawdown_yen": max_drawdown(ordered_profit["_profit"]),
        "top1_removed_roi_pct": roi_without_top(frame, 1),
        "top3_removed_roi_pct": roi_without_top(frame, 3),
        "top5_removed_roi_pct": roi_without_top(frame, 5),
        "min_year_roi_pct": float(year_roi.min()) if not year_roi.empty else np.nan,
        "avg_pair_lap_fit_min": float(ncol(frame, "pair_lap_fit_min_eval", np.nan).mean()),
        "avg_pair_lap_cosine_min": float(ncol(frame, "pair_lap_cosine_min_eval", np.nan).mean()),
        "avg_horse_lap_decomp": float(ncol(frame, "horse_lap_decomp_score", np.nan).mean()),
    }


def yearly_metrics(frame: pd.DataFrame, segment: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for year, sub in frame.groupby("year", dropna=True):
        row = metrics(sub, segment)
        row["year"] = int(year) if pd.notna(year) else None
        rows.append(row)
    return pd.DataFrame(rows)


def segment_breakdown(df: pd.DataFrame, min_tickets: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    yearly_rows: list[pd.DataFrame] = []
    groupers = [
        ("race_mode", ["race_lap_mode_label"]),
        ("match_bucket", ["lap_match_bucket_label"]),
        ("type_pair", ["lap_type_pair_label"]),
        ("race_mode_x_type_pair", ["race_lap_mode_label", "lap_type_pair_label"]),
    ]
    numeric_policies = [
        ("fit_min", "pair_lap_fit_min_eval", "ge", [0.60, 0.70, 0.80, 0.90]),
        ("cosine_min", "pair_lap_cosine_min_eval", "ge", [0.60, 0.70, 0.80, 0.90]),
        ("conf_min", "pair_lap_conf_min_eval", "ge", [0.60, 0.70, 0.80]),
        ("mismatch_popular", "pair_lap_mismatch_popular_max_eval", "le", [0.20, 0.30, 0.40]),
        ("decomp_score", "horse_lap_decomp_score", "ge", [0.60, 0.70, 0.80, 0.90]),
        ("gap", "pair_lap_gap_eval", "le", [0.20, 0.30, 0.40]),
    ]
    for dataset, ddf in df.groupby("dataset", dropna=False):
        for scope_name, scope_df in [("all", ddf), *[(str(t), sub) for t, sub in ddf.groupby("ticket_type", dropna=False)]]:
            if scope_df.empty:
                continue
            prefix = f"{dataset}:{scope_name}"
            base = metrics(scope_df, f"{prefix}:base")
            base_rows.append(base)
            base_roi = float(base["roi_pct"])
            yearly_rows.append(yearly_metrics(scope_df, f"{prefix}:base"))

            for label, cols in groupers:
                for keys, sub in scope_df.groupby(cols, dropna=False):
                    if len(sub) < min_tickets:
                        continue
                    if not isinstance(keys, tuple):
                        keys = (keys,)
                    key_text = " / ".join(str(k) for k in keys)
                    segment_rows.append(metrics(sub, f"{prefix}:{label}:{key_text}", base_roi))

            for name, col, op, quantiles in numeric_policies:
                values = ncol(scope_df, col, np.nan)
                if not values.notna().any():
                    continue
                for q in quantiles:
                    th = float(values.quantile(q))
                    mask = values.ge(th) if op == "ge" else values.le(th)
                    sub = scope_df[mask.fillna(False)]
                    if len(sub) < min_tickets:
                        continue
                    seg = f"{prefix}:policy:{name}_{op}_q{int(q*100)}({th:.3f})"
                    segment_rows.append(metrics(sub, seg, base_roi))
                    yearly_rows.append(yearly_metrics(sub, seg))

            combos = {
                "both_match_low_mismatch": scope_df["lap_match_bucket"].eq("both_match")
                & ncol(scope_df, "pair_lap_mismatch_popular_max_eval", 0.0).le(ncol(scope_df, "pair_lap_mismatch_popular_max_eval", 0.0).quantile(0.40)),
                "one_match_high_fit": scope_df["lap_match_bucket"].eq("one_match")
                & ncol(scope_df, "pair_lap_fit_min_eval", 0.0).ge(ncol(scope_df, "pair_lap_fit_min_eval", 0.0).quantile(0.70)),
                "high_decomp_low_gap": ncol(scope_df, "horse_lap_decomp_score", 0.0).ge(ncol(scope_df, "horse_lap_decomp_score", 0.0).quantile(0.70))
                & ncol(scope_df, "pair_lap_gap_eval", 1.0).le(ncol(scope_df, "pair_lap_gap_eval", 1.0).quantile(0.40)),
                "high_fit_conf_low_mismatch": ncol(scope_df, "pair_lap_fit_min_eval", 0.0).ge(ncol(scope_df, "pair_lap_fit_min_eval", 0.0).quantile(0.70))
                & ncol(scope_df, "pair_lap_conf_min_eval", 0.0).ge(ncol(scope_df, "pair_lap_conf_min_eval", 0.0).quantile(0.60))
                & ncol(scope_df, "pair_lap_mismatch_popular_max_eval", 1.0).le(ncol(scope_df, "pair_lap_mismatch_popular_max_eval", 1.0).quantile(0.50)),
            }
            for name, mask in combos.items():
                sub = scope_df[mask.fillna(False)]
                if len(sub) < min_tickets:
                    continue
                seg = f"{prefix}:combo:{name}"
                segment_rows.append(metrics(sub, seg, base_roi))
                yearly_rows.append(yearly_metrics(sub, seg))

    base_df = pd.DataFrame(base_rows)
    seg_df = pd.DataFrame(segment_rows)
    if not seg_df.empty:
        seg_df = seg_df.sort_values(["roi_pct", "top3_removed_roi_pct", "tickets"], ascending=[False, False, False])
    yearly_df = pd.concat([x for x in yearly_rows if x is not None and not x.empty], ignore_index=True) if yearly_rows else pd.DataFrame()
    return base_df, seg_df, yearly_df


def runner_type_metrics(runner: pd.DataFrame) -> pd.DataFrame:
    rows = []
    runner = runner.copy()
    runner["predicted_mode_label"] = runner["predicted_lap_mode"].map(AXIS_LABELS).fillna(runner["predicted_lap_mode"])
    runner["horse_type_label"] = runner["computed_horse_lap_type"].map(AXIS_LABELS).fillna("不明")
    runner["mode_type_match"] = runner["predicted_lap_mode"].astype(str).eq(runner["computed_horse_lap_type"].astype(str))
    for cols, prefix in [
        (["horse_type_label"], "horse_type"),
        (["predicted_mode_label"], "predicted_mode"),
        (["predicted_mode_label", "horse_type_label"], "mode_x_type"),
        (["mode_type_match"], "mode_match"),
    ]:
        for key, sub in runner.groupby(cols, dropna=False):
            if len(sub) < 30:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            rows.append(
                {
                    "segment": f"{prefix}:{' / '.join(map(str, key))}",
                    "rows": int(len(sub)),
                    "races": int(sub["race_id"].nunique()),
                    "win_rate_pct": float(ncol(sub, "is_win", 0.0).fillna(0.0).mean() * 100.0),
                    "top3_rate_pct": float(ncol(sub, "is_top3", 0.0).fillna(0.0).mean() * 100.0),
                    "avg_odds": float(ncol(sub, "odds", np.nan).mean()),
                    "avg_popularity": float(ncol(sub, "popularity", np.nan).mean()),
                    "avg_fit": float(ncol(sub, "lap_profile_fit_score", np.nan).mean()),
                    "avg_type_strength": float(ncol(sub, "horse_lap_type_strength", np.nan).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["top3_rate_pct", "rows"], ascending=[False, False])


def process_ticket_set(name: str, path: Path, runner: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    tickets = normalize_tickets(read_csv(path), name)
    return add_lap_decomposition(add_runner_sides(tickets, runner))


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose horse-specific lap aptitude and evaluate ticket ROI segments.")
    parser.add_argument("--runner-lap-csv", type=Path, default=DEFAULT_RUNNER_LAP)
    parser.add_argument("--ticket-csv", action="append", default=[], help="Optional NAME=PATH ticket CSV. Can be repeated.")
    parser.add_argument("--min-tickets", type=int, default=100)
    parser.add_argument("--write-full-overlays", action="store_true", help="Write full enriched ticket overlays. Off by default because they can be large.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = load_runner_lap(args.runner_lap_csv)
    runner.to_csv(args.output_dir / "runner_lap_aptitude_decomposition.csv", index=False, encoding="utf-8-sig")
    runner_metrics = runner_type_metrics(runner)
    runner_metrics.to_csv(args.output_dir / "runner_type_metrics.csv", index=False, encoding="utf-8-sig")

    ticket_sets: list[tuple[str, Path]] = list(DEFAULT_TICKET_SETS)
    for item in args.ticket_csv:
        if "=" not in item:
            raise ValueError("--ticket-csv must be NAME=PATH")
        name, raw_path = item.split("=", 1)
        ticket_sets.append((name, Path(raw_path)))

    frames = []
    missing = []
    for name, path in ticket_sets:
        frame = process_ticket_set(name, path, runner)
        if frame.empty:
            missing.append(str(path))
            continue
        if args.write_full_overlays or len(frame) <= 5000:
            frame.to_csv(args.output_dir / f"{name}_horse_lap_ticket_overlay.csv", index=False, encoding="utf-8-sig")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No ticket CSV could be processed.")
    all_tickets = pd.concat(frames, ignore_index=True, sort=False)
    compact_cols = [
        "dataset",
        "race_id",
        "year",
        "ticket_type",
        "anchor_no",
        "anchor_name",
        "partner_no",
        "partner_name",
        "race_lap_mode_label",
        "anchor_lap_type_label",
        "partner_lap_type_label",
        "lap_match_bucket_label",
        "lap_type_pair_label",
        "pair_lap_fit_min_eval",
        "pair_lap_cosine_min_eval",
        "pair_lap_conf_min_eval",
        "pair_lap_mismatch_popular_max_eval",
        "horse_lap_decomp_score",
        "stake_yen_eval",
        "return_yen_eval",
        "hit_eval",
    ]
    all_tickets[[c for c in compact_cols if c in all_tickets.columns]].to_csv(
        args.output_dir / "all_ticket_lap_decomposition_compact.csv", index=False, encoding="utf-8-sig"
    )

    base_df, segment_df, yearly_df = segment_breakdown(all_tickets, args.min_tickets)
    base_df.to_csv(args.output_dir / "base_metrics.csv", index=False, encoding="utf-8-sig")
    segment_df.to_csv(args.output_dir / "segment_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(args.output_dir / "yearly_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "runner_lap_csv": str(args.runner_lap_csv),
        "output_dir": str(args.output_dir),
        "runner_rows": int(len(runner)),
        "runner_races": int(runner["race_id"].nunique()),
        "ticket_rows": int(len(all_tickets)),
        "ticket_races": int(all_tickets["race_id"].nunique()),
        "datasets": sorted(all_tickets["dataset"].dropna().unique().tolist()),
        "missing_ticket_paths": missing,
        "base": base_df.to_dict(orient="records"),
        "top_segments": segment_df.head(30).to_dict(orient="records") if not segment_df.empty else [],
        "notes": [
            "Horse lap type is decomposed into fast/slow/instant/sustain/long_spurt using existing pre-race runner lap profile features.",
            "Ticket evaluation joins the runner lap profile to both anchor and partner, then evaluates race-mode match, pair-type combinations, fit, confidence, and mismatch risk.",
            "This is an analysis/shadow validation script; it does not change production BUY gates.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "ticket_rows": len(all_tickets), "top_segments": summary["top_segments"][:5]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
