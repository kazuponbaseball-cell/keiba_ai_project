from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "outputs" / "analysis" / "lap_positive_expansion_v1" / "lap_positive_expansion_selected_tickets.csv"
DEFAULT_COVARIANCE_IN = ROOT / "outputs" / "analysis" / "pair_covariance_synergy_v1" / "pair_covariance_detail.csv"
DEFAULT_OUT = ROOT / "outputs" / "analysis" / "lap_timing_profile_fit_v1"

BASE_POLICIES = [
    "wide_price_sane_strong_base",
    "umaren_price_sane_strong_base",
]
REPLACE_POLICIES = [
    "wide_price_sane_strong_lap_replace_w0.10",
    "wide_price_sane_strong_lap_replace_w0.18",
    "wide_price_sane_strong_lap_replace_w0.26",
    "wide_price_sane_strong_lap_replace_w0.34",
    "umaren_price_sane_strong_lap_replace_w0.10",
    "umaren_price_sane_strong_lap_replace_w0.18",
    "umaren_price_sane_strong_lap_replace_w0.26",
    "umaren_price_sane_strong_lap_replace_w0.34",
]


def num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def clean_race_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(16)


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = pd.to_numeric(profit, errors="coerce").fillna(0.0).cumsum()
    return float((curve.cummax() - curve).max())


def roi_without_top_returns(df: pd.DataFrame, n: int) -> float:
    if df.empty:
        return 0.0
    work = df.sort_values("return_yen", ascending=False).iloc[n:]
    stake = float(work["stake_yen"].sum())
    ret = float(work["return_yen"].sum())
    return ret / stake if stake else 0.0


def metrics(df: pd.DataFrame, label: str, bet_type: str, segment: str = "") -> dict[str, Any]:
    if df.empty:
        return {
            "label": label,
            "bet_type": bet_type,
            "segment": segment,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": 0.0,
            "hit_rate_pct": 0.0,
            "roi_ex_top1_pct": 0.0,
            "roi_ex_top3_pct": 0.0,
            "roi_ex_top5_pct": 0.0,
            "top_return_share_pct": 0.0,
            "max_drawdown_yen": 0.0,
        }
    stake = pd.to_numeric(df["stake_yen"], errors="coerce").fillna(100.0)
    ret = pd.to_numeric(df["return_yen"], errors="coerce").fillna(0.0)
    profit = ret - stake
    hit = as_bool(df["hit"]) | ret.gt(0)
    top_return = float(ret.max())
    return {
        "label": label,
        "bet_type": bet_type,
        "segment": segment,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": round(float(stake.sum()), 1),
        "return_yen": round(float(ret.sum()), 1),
        "profit_yen": round(float(profit.sum()), 1),
        "roi_pct": round(float(ret.sum() / stake.sum() * 100.0), 1) if float(stake.sum()) > 0 else 0.0,
        "hit_rate_pct": round(float(hit.mean() * 100.0), 1),
        "roi_ex_top1_pct": round(roi_without_top_returns(df, 1) * 100.0, 1),
        "roi_ex_top3_pct": round(roi_without_top_returns(df, 3) * 100.0, 1),
        "roi_ex_top5_pct": round(roi_without_top_returns(df, 5) * 100.0, 1),
        "top_return_share_pct": round(float(top_return / ret.sum() * 100.0), 1) if float(ret.sum()) > 0 else 0.0,
        "max_drawdown_yen": round(max_drawdown(profit), 1),
    }


def timing_bucket(row_mode: pd.Series) -> pd.Series:
    mode = row_mode.astype("string").fillna("")
    out = pd.Series("unknown", index=mode.index, dtype="string")
    out = out.mask(mode.isin(["instant", "slow"]), "L1_instant_finish")
    out = out.mask(mode.eq("sustain"), "L2_sustain")
    out = out.mask(mode.eq("fast"), "L3_front_load")
    return out


def add_timing_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = clean_race_id(out["race_id"])
    out["race_date"] = pd.to_datetime(out["race_id"].str[:8], format="%Y%m%d", errors="coerce")
    out["year"] = pd.to_numeric(out.get("year", out["race_id"].str[:4]), errors="coerce").fillna(
        out["race_date"].dt.year
    ).astype(int)

    out["lap_timing_bucket"] = timing_bucket(out.get("v2_predicted_lap_mode", pd.Series("", index=out.index)))
    anchor_mode = out.get("anchor_horse_lap_profile_top_mode", pd.Series("", index=out.index)).astype("string").fillna("")
    partner_mode = out.get("partner_horse_lap_profile_top_mode", pd.Series("", index=out.index)).astype("string").fillna("")

    l1_modes = {"instant", "slow"}
    l2_modes = {"sustain", "long_spurt"}
    l3_modes = {"fast", "long_spurt"}

    a_l1 = anchor_mode.isin(l1_modes).astype(float)
    p_l1 = partner_mode.isin(l1_modes).astype(float)
    a_l2 = anchor_mode.isin(l2_modes).astype(float)
    p_l2 = partner_mode.isin(l2_modes).astype(float)
    a_l3 = anchor_mode.isin(l3_modes).astype(float)
    p_l3 = partner_mode.isin(l3_modes).astype(float)

    bucket = out["lap_timing_bucket"].astype(str)
    l1_pair = (a_l1 + p_l1) / 2.0
    l2_pair = (a_l2 + p_l2) / 2.0
    l3_pair = (a_l3 + p_l3) / 2.0
    mode_match = pd.Series(0.0, index=out.index, dtype=float)
    mode_match = mode_match.mask(bucket.eq("L1_instant_finish"), l1_pair)
    mode_match = mode_match.mask(bucket.eq("L2_sustain"), l2_pair)
    mode_match = mode_match.mask(bucket.eq("L3_front_load"), l3_pair)

    fit_avg = num(out, "pair_lap_profile_fit_avg", 0.0).clip(0.0, 1.0)
    conf_avg = num(out, "pair_lap_confident_avg", 0.0).clip(0.0, 1.0)
    axis_avg = num(out, "pair_lap_axis_avg", 0.0).clip(0.0, 1.0)
    specialist = num(out, "pair_lap_partner_specialist_max", 0.0).clip(0.0, 1.0)
    mismatch = num(out, "pair_lap_mismatch_popular_max", 0.0).clip(0.0, 1.0)
    v2_conf = num(out, "v2_confidence", 0.0).clip(0.0, 1.0)
    v2_margin = num(out, "v2_margin", 0.0).clip(0.0, 1.0)

    out["lap_timing_mode_match_score"] = mode_match.clip(0.0, 1.0)
    out["lap_timing_fit_score_v1"] = (
        0.38 * fit_avg
        + 0.18 * conf_avg
        + 0.17 * axis_avg
        + 0.17 * out["lap_timing_mode_match_score"]
        + 0.10 * specialist
        - 0.14 * mismatch
    ).clip(0.0, 1.0)
    out["lap_timing_mismatch_risk_v1"] = (
        0.52 * mismatch
        + 0.24 * (1.0 - out["lap_timing_mode_match_score"])
        + 0.14 * (1.0 - fit_avg)
        + 0.10 * (1.0 - v2_conf)
    ).clip(0.0, 1.0)
    out["lap_timing_clarity_v1"] = (0.70 * v2_conf + 0.30 * (v2_margin * 3.0).clip(0.0, 1.0)).clip(0.0, 1.0)

    out["lap_l1_need"] = np.maximum(num(out, "shape_instant_signal", 0.0), num(out, "shape_slow_signal", 0.0))
    out["lap_l2_need"] = num(out, "shape_sustain_signal", 0.0)
    out["lap_l3_need"] = num(out, "shape_fast_signal", 0.0)
    return out


def thresholds(train: pd.DataFrame, base_policy: str) -> dict[str, float]:
    base = train[train["policy"].astype(str).eq(base_policy)]
    if base.empty:
        base = train
    cols = ["lap_timing_fit_score_v1", "lap_timing_mismatch_risk_v1", "lap_timing_clarity_v1", "lap_timing_mode_match_score"]
    out: dict[str, float] = {}
    for col in cols:
        s = pd.to_numeric(base[col], errors="coerce").dropna()
        if s.empty:
            continue
        for q in [0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            out[f"{col}_q{int(q * 100)}"] = float(s.quantile(q))
    return out


def policy_masks(df: pd.DataFrame, th: dict[str, float]) -> list[tuple[str, pd.Series]]:
    fit = df["lap_timing_fit_score_v1"]
    risk = df["lap_timing_mismatch_risk_v1"]
    clarity = df["lap_timing_clarity_v1"]
    match = df["lap_timing_mode_match_score"]
    return [
        ("base", pd.Series(True, index=df.index)),
        (
            "timing_fit_q50_risk_q75",
            fit.ge(th.get("lap_timing_fit_score_v1_q50", 0.0))
            & risk.le(th.get("lap_timing_mismatch_risk_v1_q75", 1.0)),
        ),
        (
            "timing_fit_q60_risk_q70",
            fit.ge(th.get("lap_timing_fit_score_v1_q60", 0.0))
            & risk.le(th.get("lap_timing_mismatch_risk_v1_q70", 1.0)),
        ),
        (
            "timing_fit_q65_risk_q65",
            fit.ge(th.get("lap_timing_fit_score_v1_q65", 0.0))
            & risk.le(th.get("lap_timing_mismatch_risk_v1_q65", 1.0)),
        ),
        (
            "timing_clear_fit_q55",
            fit.ge(th.get("lap_timing_fit_score_v1_q55", 0.0))
            & clarity.ge(th.get("lap_timing_clarity_v1_q50", 0.0)),
        ),
        (
            "mode_match_plus_fit_q50",
            match.ge(0.5)
            & fit.ge(th.get("lap_timing_fit_score_v1_q50", 0.0))
            & risk.le(th.get("lap_timing_mismatch_risk_v1_q75", 1.0)),
        ),
        (
            "mode_both_match",
            match.ge(1.0)
            & fit.ge(th.get("lap_timing_fit_score_v1_q45", 0.0)),
        ),
        (
            "no_popular_mismatch",
            risk.le(th.get("lap_timing_mismatch_risk_v1_q50", 1.0)),
        ),
    ]


def segment_metrics(base: pd.DataFrame, label: str, bet_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    segment_defs = [
        ("lap_timing_bucket", base["lap_timing_bucket"].astype(str)),
        ("mode_match_band", pd.cut(base["lap_timing_mode_match_score"], [-0.01, 0.49, 0.99, 1.01], labels=["none", "one", "both"])),
        ("fit_band", pd.qcut(base["lap_timing_fit_score_v1"].rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])),
        ("risk_band", pd.qcut(base["lap_timing_mismatch_risk_v1"].rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])),
    ]
    for seg_name, values in segment_defs:
        tmp = base.copy()
        tmp["_segment_value"] = values.astype("string").fillna("unknown")
        for val, group in tmp.groupby("_segment_value", dropna=False):
            if len(group) < 20:
                continue
            rows.append(metrics(group, label, bet_type, f"{seg_name}={val}"))
    return rows


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    for base_policy in BASE_POLICIES:
        bet_type = "wide" if base_policy.startswith("wide_") else "umaren"
        base_all = df[df["policy"].astype(str).eq(base_policy)].copy()
        if base_all.empty:
            continue
        train = base_all[base_all["year"] < 2026].copy()
        test = base_all[base_all["year"] >= 2026].copy()
        th = thresholds(train if not train.empty else base_all, base_policy)

        for gate_name, mask in policy_masks(base_all, th):
            subset = base_all[mask].copy()
            row = metrics(subset, gate_name, bet_type)
            row["source_policy"] = base_policy
            row["threshold_source"] = "pre_2026" if not train.empty else "all"
            all_rows.append(row)
            for year, gy in subset.groupby("year"):
                yr = metrics(gy, gate_name, bet_type)
                yr["source_policy"] = base_policy
                yr["year"] = int(year)
                year_rows.append(yr)

        if not test.empty:
            for gate_name, mask in policy_masks(test, th):
                subset = test[mask].copy()
                row = metrics(subset, f"oos_2026_{gate_name}", bet_type)
                row["source_policy"] = base_policy
                row["threshold_source"] = "pre_2026"
                oos_rows.append(row)

        segment_rows.extend(segment_metrics(base_all, base_policy, bet_type))

    for policy in REPLACE_POLICIES:
        current = df[df["policy"].astype(str).eq(policy)].copy()
        if current.empty:
            continue
        bet_type = "wide" if policy.startswith("wide_") else "umaren"
        row = metrics(current, "existing_lap_replace", bet_type)
        row["source_policy"] = policy
        all_rows.append(row)

    return pd.DataFrame(all_rows), pd.DataFrame(year_rows), pd.DataFrame(oos_rows), pd.DataFrame(segment_rows)


def evaluate_current_strongest_with_timing(timing_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not DEFAULT_COVARIANCE_IN.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cov_needed = {
        "race_id",
        "year",
        "policy",
        "horse_a",
        "horse_b",
        "ticket_type",
        "stake_yen",
        "return_yen",
        "hit",
        "race_date",
    }
    cov = pd.read_csv(DEFAULT_COVARIANCE_IN, usecols=lambda c: c in cov_needed, encoding="utf-8-sig", low_memory=False)
    cov["race_id"] = clean_race_id(cov["race_id"])
    cov["horse_a"] = pd.to_numeric(cov["horse_a"], errors="coerce").astype("Int64")
    cov["horse_b"] = pd.to_numeric(cov["horse_b"], errors="coerce").astype("Int64")
    cov["race_date"] = pd.to_datetime(cov.get("race_date", cov["race_id"].str[:8]), errors="coerce")
    missing_date = cov["race_date"].isna()
    if missing_date.any():
        cov.loc[missing_date, "race_date"] = pd.to_datetime(
            cov.loc[missing_date, "race_id"].str[:8], format="%Y%m%d", errors="coerce"
        )
    cov["year"] = pd.to_numeric(cov.get("year", cov["race_date"].dt.year), errors="coerce").fillna(
        cov["race_date"].dt.year
    ).astype(int)

    map_cols = [
        "race_id",
        "horse_a",
        "horse_b",
        "lap_timing_bucket",
        "lap_timing_mode_match_score",
        "lap_timing_fit_score_v1",
        "lap_timing_mismatch_risk_v1",
        "lap_timing_clarity_v1",
        "lap_l1_need",
        "lap_l2_need",
        "lap_l3_need",
    ]
    timing_map = timing_df[map_cols].drop_duplicates(["race_id", "horse_a", "horse_b"], keep="first").copy()
    timing_map["horse_a"] = pd.to_numeric(timing_map["horse_a"], errors="coerce").astype("Int64")
    timing_map["horse_b"] = pd.to_numeric(timing_map["horse_b"], errors="coerce").astype("Int64")
    merged = cov.merge(timing_map, on=["race_id", "horse_a", "horse_b"], how="left")
    # In case the stored pair order differs, try reversed order for missing rows.
    missing = merged["lap_timing_fit_score_v1"].isna()
    if missing.any():
        rev = timing_map.rename(columns={"horse_a": "horse_b", "horse_b": "horse_a"})
        fill = cov.loc[missing, cov.columns].merge(rev, on=["race_id", "horse_a", "horse_b"], how="left")
        for col in map_cols[3:]:
            merged.loc[missing, col] = fill[col].to_numpy()
    for col in map_cols[3:]:
        if col.startswith("lap_timing_bucket"):
            merged[col] = merged[col].astype("string").fillna("unknown")
        else:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    deployable = merged[merged["policy"].astype(str).eq("deployable_filter_base")].copy()
    if deployable.empty:
        return pd.DataFrame(), pd.DataFrame(), merged
    train = deployable[deployable["year"] < 2026].copy()
    th = thresholds(train if not train.empty else deployable, "deployable_filter_base")
    for gate_name, mask in policy_masks(deployable, th):
        subset = deployable[mask].copy()
        row = metrics(subset, gate_name, "umaren")
        row["source_policy"] = "deployable_filter_base_current_strongest"
        row["missing_timing_rows"] = int(deployable["lap_timing_fit_score_v1"].eq(0.0).sum())
        rows.append(row)
        for year, gy in subset.groupby("year"):
            yr = metrics(gy, gate_name, "umaren")
            yr["source_policy"] = "deployable_filter_base_current_strongest"
            yr["year"] = int(year)
            yearly_rows.append(yr)
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows), merged


def write_readme(out_dir: Path, overall: pd.DataFrame, oos: pd.DataFrame, segments: pd.DataFrame) -> None:
    lines = [
        "# Lap Timing Profile Fit v1",
        "",
        "Full 200m L3/L2/L1 splits are not available in the current horse-level base table, so this test uses a pre-race proxy:",
        "",
        "- L1: instant / slow finish profile",
        "- L2: sustain profile",
        "- L3: fast / front-load / long-spurt profile",
        "",
        "The test checks whether existing strongest wide/umaren candidates should be filtered by timing fit, mismatch risk, and predicted lap-mode match.",
        "",
        "## Overall Top",
        "",
    ]
    show = overall.sort_values(["bet_type", "roi_pct", "tickets"], ascending=[True, False, False]).head(16)
    for _, r in show.iterrows():
        lines.append(
            f"- {r['bet_type']} / {r['source_policy']} / {r['label']}: "
            f"tickets={int(r['tickets'])}, ROI={r['roi_pct']:.1f}%, hit={r['hit_rate_pct']:.1f}%, "
            f"exTop1={r['roi_ex_top1_pct']:.1f}%"
        )
    lines += ["", "## 2026 OOS Check", ""]
    if oos.empty:
        lines.append("- No 2026 OOS rows.")
    else:
        show = oos.sort_values(["bet_type", "roi_pct", "tickets"], ascending=[True, False, False]).head(16)
        for _, r in show.iterrows():
            lines.append(
                f"- {r['bet_type']} / {r['source_policy']} / {r['label']}: "
                f"tickets={int(r['tickets'])}, ROI={r['roi_pct']:.1f}%, hit={r['hit_rate_pct']:.1f}%, "
                f"exTop1={r['roi_ex_top1_pct']:.1f}%"
            )
    lines += ["", "## Interpretation", ""]
    lines.append("- If a gate raises ROI but sharply reduces tickets or loses 2026 stability, keep it as a confidence/shadow label.")
    lines.append("- If 2026 OOS and ex-top-return metrics both hold, it is a candidate for formal BUY filtering.")
    lines.append("- This does not replace the strongest model; it tests whether lap-timing suitability can refine the final gate.")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    needed = set(
        [
            "race_id",
            "year",
            "policy",
            "horse_a",
            "horse_b",
            "ticket_type",
            "stake_yen",
            "return_yen",
            "hit",
            "v2_predicted_lap_mode",
            "anchor_horse_lap_profile_top_mode",
            "partner_horse_lap_profile_top_mode",
            "shape_fast_signal",
            "shape_slow_signal",
            "shape_sustain_signal",
            "shape_instant_signal",
            "v2_confidence",
            "v2_margin",
            "pair_lap_profile_fit_avg",
            "pair_lap_confident_avg",
            "pair_lap_axis_avg",
            "pair_lap_partner_specialist_max",
            "pair_lap_mismatch_popular_max",
        ]
    )
    df = pd.read_csv(DEFAULT_IN, usecols=lambda c: c in needed, encoding="utf-8-sig", low_memory=False)
    df = add_timing_scores(df)
    overall, yearly, oos, segments = evaluate(df)
    strongest_overall, strongest_yearly, strongest_detail = evaluate_current_strongest_with_timing(df)

    overall = overall.sort_values(["bet_type", "source_policy", "label"], kind="mergesort")
    yearly = yearly.sort_values(["bet_type", "source_policy", "label", "year"], kind="mergesort")
    oos = oos.sort_values(["bet_type", "source_policy", "label"], kind="mergesort")
    segments = segments.sort_values(["bet_type", "label", "segment"], kind="mergesort")

    overall.to_csv(DEFAULT_OUT / "lap_timing_policy_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(DEFAULT_OUT / "lap_timing_policy_yearly.csv", index=False, encoding="utf-8-sig")
    oos.to_csv(DEFAULT_OUT / "lap_timing_policy_oos_2026.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(DEFAULT_OUT / "lap_timing_segment_metrics.csv", index=False, encoding="utf-8-sig")
    strongest_overall.to_csv(DEFAULT_OUT / "lap_timing_current_strongest_metrics.csv", index=False, encoding="utf-8-sig")
    strongest_yearly.to_csv(DEFAULT_OUT / "lap_timing_current_strongest_yearly.csv", index=False, encoding="utf-8-sig")
    if not strongest_detail.empty:
        strongest_detail.to_csv(DEFAULT_OUT / "lap_timing_current_strongest_detail.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input": str(DEFAULT_IN),
        "output_dir": str(DEFAULT_OUT),
        "overall_top": overall.sort_values("roi_pct", ascending=False).head(20).to_dict(orient="records"),
        "oos_top": oos.sort_values("roi_pct", ascending=False).head(20).to_dict(orient="records"),
        "current_strongest_top": strongest_overall.sort_values("roi_pct", ascending=False).head(20).to_dict(orient="records")
        if not strongest_overall.empty
        else [],
    }
    (DEFAULT_OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(DEFAULT_OUT, overall, oos, segments)

    print(f"Wrote {DEFAULT_OUT}")
    print("\nOverall:")
    print(
        overall[
            [
                "bet_type",
                "source_policy",
                "label",
                "tickets",
                "roi_pct",
                "hit_rate_pct",
                "roi_ex_top1_pct",
                "top_return_share_pct",
                "max_drawdown_yen",
            ]
        ]
        .sort_values(["bet_type", "roi_pct"], ascending=[True, False])
        .head(24)
        .to_string(index=False)
    )
    print("\n2026 OOS:")
    print(
        oos[
            [
                "bet_type",
                "source_policy",
                "label",
                "tickets",
                "roi_pct",
                "hit_rate_pct",
                "roi_ex_top1_pct",
                "top_return_share_pct",
            ]
        ]
        .sort_values(["bet_type", "roi_pct"], ascending=[True, False])
        .head(24)
        .to_string(index=False)
    )
    if not strongest_overall.empty:
        print("\nCurrent strongest deployable_filter_base:")
        print(
            strongest_overall[
                [
                    "source_policy",
                    "label",
                    "tickets",
                    "roi_pct",
                    "hit_rate_pct",
                    "roi_ex_top1_pct",
                    "top_return_share_pct",
                    "max_drawdown_yen",
                    "missing_timing_rows",
                ]
            ]
            .sort_values("roi_pct", ascending=False)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
