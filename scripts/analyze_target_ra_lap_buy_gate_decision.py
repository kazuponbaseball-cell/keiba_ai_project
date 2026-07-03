from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/analysis/target_ra_official_lap_history_overlay_v1/tickets_with_target_ra_official_lap_overlay.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/target_ra_lap_buy_gate_decision_v1"


USECOLS = [
    "race_id",
    "year",
    "policy",
    "ticket_type",
    "stake_yen",
    "return_yen",
    "hit",
    "official_pair_fit_max_avg",
    "official_pair_fit_max_min",
    "official_pair_fit_mean_avg",
    "official_pair_need_match_avg",
    "official_pair_strength_avg",
    "official_pair_ready_both",
    "official_pair_ready_any",
    "official_pair_mismatch_risk",
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


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = pd.to_numeric(profit, errors="coerce").fillna(0.0).cumsum()
    return float((curve.cummax() - curve).max())


def roi_without_top_returns(df: pd.DataFrame, n: int) -> float:
    if df.empty:
        return 0.0
    work = df.sort_values("return_yen", ascending=False).iloc[n:]
    stake = pd.to_numeric(work["stake_yen"], errors="coerce").fillna(100.0).sum()
    ret = pd.to_numeric(work["return_yen"], errors="coerce").fillna(0.0).sum()
    return float(ret / stake * 100.0) if stake else 0.0


def metrics(df: pd.DataFrame, policy: str, ticket_type: str, gate: str, split: str) -> dict[str, Any]:
    if df.empty:
        return {
            "policy": policy,
            "ticket_type": ticket_type,
            "gate": gate,
            "split": split,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": 0.0,
            "hit_rate_pct": 0.0,
            "roi_ex_top1_pct": 0.0,
            "roi_ex_top3_pct": 0.0,
            "max_drawdown_yen": 0.0,
        }
    stake = pd.to_numeric(df["stake_yen"], errors="coerce").fillna(100.0)
    ret = pd.to_numeric(df["return_yen"], errors="coerce").fillna(0.0)
    hit = df.get("hit", ret.gt(0)).astype(str).str.lower().isin(["true", "1", "1.0", "yes"]) | ret.gt(0)
    profit = ret - stake
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    return {
        "policy": policy,
        "ticket_type": ticket_type,
        "gate": gate,
        "split": split,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()) if "race_id" in df.columns else int(len(df)),
        "stake_yen": round(stake_sum, 1),
        "return_yen": round(ret_sum, 1),
        "profit_yen": round(float(profit.sum()), 1),
        "roi_pct": round(ret_sum / stake_sum * 100.0, 1) if stake_sum else 0.0,
        "hit_rate_pct": round(float(hit.mean() * 100.0), 1),
        "roi_ex_top1_pct": round(roi_without_top_returns(df, 1), 1),
        "roi_ex_top3_pct": round(roi_without_top_returns(df, 3), 1),
        "max_drawdown_yen": round(max_drawdown(profit), 1),
    }


def q(series: pd.Series, quantile: float, default: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.quantile(quantile)) if not clean.empty else default


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "1.0", "yes"])


def build_gate_masks(group: pd.DataFrame, train: pd.DataFrame) -> dict[str, pd.Series]:
    fit = pd.to_numeric(group["official_pair_fit_max_avg"], errors="coerce").fillna(0.0)
    fit_min = pd.to_numeric(group["official_pair_fit_max_min"], errors="coerce").fillna(0.0)
    fit_mean = pd.to_numeric(group["official_pair_fit_mean_avg"], errors="coerce").fillna(0.0)
    need = pd.to_numeric(group["official_pair_need_match_avg"], errors="coerce").fillna(0.0)
    strength = pd.to_numeric(group["official_pair_strength_avg"], errors="coerce").fillna(0.0)
    risk = pd.to_numeric(group["official_pair_mismatch_risk"], errors="coerce").fillna(1.0)
    ready_both = bool_series(group["official_pair_ready_both"])
    ready_any = bool_series(group["official_pair_ready_any"])

    fit_q60 = q(train["official_pair_fit_max_avg"], 0.60, 1.0)
    fit_q65 = q(train["official_pair_fit_max_avg"], 0.65, 1.0)
    fit_q70 = q(train["official_pair_fit_max_avg"], 0.70, 1.0)
    min_q55 = q(train["official_pair_fit_max_min"], 0.55, 1.0)
    mean_q60 = q(train["official_pair_fit_mean_avg"], 0.60, 1.0)
    need_q55 = q(train["official_pair_need_match_avg"], 0.55, 1.0)
    strength_q50 = q(train["official_pair_strength_avg"], 0.50, 1.0)
    risk_q55 = q(train["official_pair_mismatch_risk"], 0.55, 0.0)

    return {
        "base": pd.Series(True, index=group.index),
        "ready_both": ready_both,
        "ready_any": ready_any,
        "abs_fit55_risk55_both": ready_both & fit.ge(0.55) & risk.le(0.55),
        "abs_fit55_risk55_any": ready_any & fit.ge(0.55) & risk.le(0.55),
        "abs_fit45_risk65_any": ready_any & fit.ge(0.45) & risk.le(0.65),
        "train_fit_q60": fit.ge(fit_q60),
        "train_fit_q65": fit.ge(fit_q65),
        "train_fit_q70": fit.ge(fit_q70),
        "train_fit_q60_risk_q55": fit.ge(fit_q60) & risk.le(risk_q55),
        "train_minfit_q55": fit_min.ge(min_q55),
        "train_meanfit_q60": fit_mean.ge(mean_q60),
        "train_fit_q65_need_q55": fit.ge(fit_q65) & need.ge(need_q55),
        "train_fit_q60_strength_q50": fit.ge(fit_q60) & strength.ge(strength_q50),
    }


def make_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (policy, ticket_type), group in df.groupby(["policy", "ticket_type"], sort=False):
        train = group[pd.to_numeric(group["year"], errors="coerce").lt(2026)].copy()
        if train.empty:
            continue
        masks = build_gate_masks(group, train)
        splits = {
            "train_pre2026": group[pd.to_numeric(group["year"], errors="coerce").lt(2026)],
            "oos_2026": group[pd.to_numeric(group["year"], errors="coerce").eq(2026)],
        }
        for year in sorted(pd.to_numeric(group["year"], errors="coerce").dropna().unique()):
            splits[f"year_{int(year)}"] = group[pd.to_numeric(group["year"], errors="coerce").eq(year)]
        for gate, mask in masks.items():
            for split, split_df in splits.items():
                rows.append(metrics(split_df.loc[mask.reindex(split_df.index).fillna(False)], policy, ticket_type, gate, split))

    metrics_df = pd.DataFrame(rows)
    base = metrics_df[metrics_df["gate"].eq("base")][
        ["policy", "ticket_type", "split", "tickets", "roi_pct", "roi_ex_top1_pct"]
    ].rename(
        columns={
            "tickets": "base_tickets",
            "roi_pct": "base_roi_pct",
            "roi_ex_top1_pct": "base_roi_ex_top1_pct",
        }
    )
    joined = metrics_df.merge(base, on=["policy", "ticket_type", "split"], how="left")
    denom = pd.to_numeric(joined["base_tickets"], errors="coerce").replace(0, float("nan"))
    joined["ticket_keep_rate_pct"] = (
        pd.to_numeric(joined["tickets"], errors="coerce").div(denom).mul(100.0).round(1)
    )
    joined["roi_lift_pct"] = (joined["roi_pct"] - joined["base_roi_pct"]).round(1)
    joined["roi_ex_top1_lift_pct"] = (joined["roi_ex_top1_pct"] - joined["base_roi_ex_top1_pct"]).round(1)

    oos = joined[
        joined["split"].eq("oos_2026")
        & joined["gate"].ne("base")
        & joined["tickets"].ge(30)
    ].copy()
    train_ok = joined[
        joined["split"].eq("train_pre2026")
        & joined["gate"].ne("base")
        & joined["tickets"].ge(100)
    ][["policy", "ticket_type", "gate", "roi_pct", "roi_ex_top1_pct"]].rename(
        columns={"roi_pct": "train_roi_pct", "roi_ex_top1_pct": "train_roi_ex_top1_pct"}
    )
    rec = oos.merge(train_ok, on=["policy", "ticket_type", "gate"], how="left")
    rec["adoption_level"] = "shadow_only"
    rec.loc[
        rec["roi_pct"].ge(115)
        & rec["roi_ex_top1_pct"].ge(95)
        & rec["train_roi_pct"].ge(100)
        & rec["train_roi_ex_top1_pct"].ge(90),
        "adoption_level",
    ] = "buy_gate_candidate"
    rec.loc[
        rec["roi_pct"].ge(130)
        & rec["roi_ex_top1_pct"].lt(90),
        "adoption_level",
    ] = "fragile_high_roi"
    rec = rec.sort_values(
        ["adoption_level", "roi_pct", "roi_ex_top1_pct", "tickets"],
        ascending=[True, False, False, False],
    )
    return joined, rec


def main() -> int:
    parser = argparse.ArgumentParser(description="Assess whether TARGET RA official-lap history is safe enough for a BUY gate.")
    parser.add_argument("--tickets-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    input_path = project_path(args.tickets_csv)
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = read_csv_any(input_path, nrows=0)
    usecols = [col for col in USECOLS if col in header.columns]
    df = read_csv_any(input_path, usecols=usecols)
    for col in [
        "year",
        "stake_yen",
        "return_yen",
        "official_pair_fit_max_avg",
        "official_pair_fit_max_min",
        "official_pair_fit_mean_avg",
        "official_pair_need_match_avg",
        "official_pair_strength_avg",
        "official_pair_mismatch_risk",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "stake_yen" not in df.columns:
        df["stake_yen"] = 100.0
    if "return_yen" not in df.columns:
        df["return_yen"] = 0.0

    metrics_df, rec = make_report(df)
    metrics_path = out_dir / "gate_metrics_by_policy_split.csv"
    rec_path = out_dir / "gate_recommendations.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    rec.to_csv(rec_path, index=False, encoding="utf-8-sig")

    top = rec.head(20).to_dict(orient="records")
    summary = {
        "input_csv": str(input_path),
        "rows": int(len(df)),
        "policies": int(df[["policy", "ticket_type"]].drop_duplicates().shape[0]),
        "metrics_csv": str(metrics_path),
        "recommendations_csv": str(rec_path),
        "top_recommendations": top,
        "note": "buy_gate_candidate requires OOS ROI>=115, OOS ROI ex top1>=95, train ROI>=100, train ROI ex top1>=90. fragile_high_roi means headline ROI is high but depends too much on the largest hit.",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
