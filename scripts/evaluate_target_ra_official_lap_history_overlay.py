from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TICKETS = ROOT / "outputs" / "analysis" / "lap_positive_expansion_v1" / "lap_positive_expansion_selected_tickets.csv"
DEFAULT_HISTORY = ROOT / "data" / "processed" / "target_ra_race_laps" / "target_ra_lap_history_features.csv"
DEFAULT_OUT = ROOT / "outputs" / "analysis" / "target_ra_official_lap_history_overlay_v1"


LAP_AXES = [
    "front_load",
    "slow_finish",
    "l1_instant",
    "l2_sustain",
    "l3_long_spurt",
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


def clean_id(s: pd.Series, width: int = 16) -> pd.Series:
    return s.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(width)


def ncol(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    if df[col].dtype == object:
        values = df[col].astype(str).str.replace(",", "", regex=False)
    else:
        values = df[col]
    return pd.to_numeric(values, errors="coerce")


def clip01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = pd.to_numeric(profit, errors="coerce").fillna(0.0).cumsum()
    return float((curve.cummax() - curve).max())


def roi_without_top_returns(df: pd.DataFrame, n: int) -> float:
    if df.empty:
        return 0.0
    work = df.sort_values("return_yen", ascending=False).iloc[n:]
    stake = float(pd.to_numeric(work["stake_yen"], errors="coerce").fillna(100.0).sum())
    ret = float(pd.to_numeric(work["return_yen"], errors="coerce").fillna(0.0).sum())
    return ret / stake if stake else 0.0


def metrics(df: pd.DataFrame, policy: str, ticket_type: str, filter_name: str, split: str) -> dict[str, Any]:
    if df.empty:
        return {
            "policy": policy,
            "ticket_type": ticket_type,
            "filter_name": filter_name,
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
    hit = df.get("hit", ret.gt(0)).astype(str).str.lower().isin(["true", "1", "yes"]) | ret.gt(0)
    profit = ret - stake
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    return {
        "policy": policy,
        "ticket_type": ticket_type,
        "filter_name": filter_name,
        "split": split,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": round(stake_sum, 1),
        "return_yen": round(ret_sum, 1),
        "profit_yen": round(float(profit.sum()), 1),
        "roi_pct": round(ret_sum / stake_sum * 100.0, 1) if stake_sum else 0.0,
        "hit_rate_pct": round(float(hit.mean() * 100.0), 1),
        "roi_ex_top1_pct": round(roi_without_top_returns(df, 1) * 100.0, 1),
        "roi_ex_top3_pct": round(roi_without_top_returns(df, 3) * 100.0, 1),
        "max_drawdown_yen": round(max_drawdown(profit), 1),
    }


def normalized_need_vector(df: pd.DataFrame) -> pd.DataFrame:
    fast = np.maximum(clip01(ncol(df, "v2_prob_fast", 0.0)), clip01(ncol(df, "shape_fast_signal", 0.0)))
    slow = np.maximum(clip01(ncol(df, "v2_prob_slow", 0.0)), clip01(ncol(df, "shape_slow_signal", 0.0)))
    instant = np.maximum(clip01(ncol(df, "v2_prob_instant", 0.0)), clip01(ncol(df, "shape_instant_signal", 0.0)))
    sustain = np.maximum(clip01(ncol(df, "v2_prob_sustain", 0.0)), clip01(ncol(df, "shape_sustain_signal", 0.0)))
    queue_front = clip01(ncol(df, "queue_front_load_score", 0.0))
    long_spurt = (0.55 * sustain + 0.35 * fast + 0.10 * queue_front).clip(0.0, 1.0)
    out = pd.DataFrame(
        {
            "need_front_load": fast,
            "need_slow_finish": slow,
            "need_l1_instant": instant,
            "need_l2_sustain": sustain,
            "need_l3_long_spurt": long_spurt,
        },
        index=df.index,
    )
    row_sum = out.sum(axis=1).replace(0.0, np.nan)
    return out.div(row_sum, axis=0).fillna(1.0 / len(out.columns))


def load_history(path: Path) -> pd.DataFrame:
    wanted = [
        "race_id",
        "horse_no",
        "official_lap_history_count_past3",
        "official_lap_history_ready",
        "official_lap_profile_strength",
        "official_lap_profile_versatility",
        "official_lap_wave_volatility_past3_mean",
        *[f"official_{axis}_goodrun_score_past3_mean" for axis in LAP_AXES],
        *[f"official_{axis}_goodrun_score_past3_max" for axis in LAP_AXES],
        *[f"official_{axis}_need_past3_mean" for axis in LAP_AXES],
    ]
    header = read_csv_any(path, nrows=0)
    usecols = [c for c in wanted if c in header.columns]
    hist = read_csv_any(path, usecols=usecols)
    hist["race_id"] = clean_id(hist["race_id"])
    hist["horse_no"] = pd.to_numeric(hist["horse_no"], errors="coerce").astype("Int64")
    for col in hist.columns:
        if col not in {"race_id", "horse_no"}:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")
    return hist.drop_duplicates(["race_id", "horse_no"], keep="last")


def merge_side(df: pd.DataFrame, hist: pd.DataFrame, side: str, no_col: str) -> pd.DataFrame:
    out = df.copy()
    out[no_col] = pd.to_numeric(out[no_col], errors="coerce").astype("Int64")
    rename = {c: f"{side}_{c}" for c in hist.columns if c not in {"race_id", "horse_no"}}
    side_hist = hist.rename(columns={"horse_no": no_col, **rename})
    return out.merge(side_hist, on=["race_id", no_col], how="left")


def add_overlay_features(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = clean_id(out["race_id"])
    if "anchor_no" not in out.columns:
        out["anchor_no"] = out["horse_a"]
    if "partner_no" not in out.columns:
        out["partner_no"] = out["horse_b"]
    out = merge_side(out, hist, "anchor", "anchor_no")
    out = merge_side(out, hist, "partner", "partner_no")

    need = normalized_need_vector(out)
    for col in need.columns:
        out[f"official_{col}"] = need[col]

    for side in ["anchor", "partner"]:
        max_fit = pd.Series(0.0, index=out.index)
        mean_fit = pd.Series(0.0, index=out.index)
        need_match = pd.Series(0.0, index=out.index)
        for axis in LAP_AXES:
            weight = need[f"need_{axis}"]
            max_score = clip01(ncol(out, f"{side}_official_{axis}_goodrun_score_past3_max", 0.0))
            mean_score = clip01(ncol(out, f"{side}_official_{axis}_goodrun_score_past3_mean", 0.0))
            need_score = clip01(ncol(out, f"{side}_official_{axis}_need_past3_mean", 0.0))
            max_fit += weight * max_score
            mean_fit += weight * mean_score
            need_match += weight * need_score
        out[f"{side}_official_lap_fit_max"] = max_fit.clip(0.0, 1.0)
        out[f"{side}_official_lap_fit_mean"] = mean_fit.clip(0.0, 1.0)
        out[f"{side}_official_lap_need_match"] = need_match.clip(0.0, 1.0)
        out[f"{side}_official_lap_ready"] = ncol(out, f"{side}_official_lap_history_ready", 0.0).fillna(0.0).gt(0)
        out[f"{side}_official_lap_strength"] = clip01(ncol(out, f"{side}_official_lap_profile_strength", 0.0))

    out["official_pair_fit_max_avg"] = (
        out["anchor_official_lap_fit_max"] + out["partner_official_lap_fit_max"]
    ) / 2.0
    out["official_pair_fit_max_min"] = np.minimum(out["anchor_official_lap_fit_max"], out["partner_official_lap_fit_max"])
    out["official_pair_fit_mean_avg"] = (
        out["anchor_official_lap_fit_mean"] + out["partner_official_lap_fit_mean"]
    ) / 2.0
    out["official_pair_need_match_avg"] = (
        out["anchor_official_lap_need_match"] + out["partner_official_lap_need_match"]
    ) / 2.0
    out["official_pair_strength_avg"] = (
        out["anchor_official_lap_strength"] + out["partner_official_lap_strength"]
    ) / 2.0
    out["official_pair_ready_both"] = out["anchor_official_lap_ready"] & out["partner_official_lap_ready"]
    out["official_pair_ready_any"] = out["anchor_official_lap_ready"] | out["partner_official_lap_ready"]
    out["official_pair_mismatch_risk"] = (
        0.55 * (1.0 - out["official_pair_fit_max_avg"]) + 0.25 * (1.0 - out["official_pair_need_match_avg"]) + 0.20 * (1.0 - out["official_pair_strength_avg"])
    ).clip(0.0, 1.0)
    return out


def quantiles(train: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    cols = [
        "official_pair_fit_max_avg",
        "official_pair_fit_max_min",
        "official_pair_fit_mean_avg",
        "official_pair_need_match_avg",
        "official_pair_strength_avg",
        "official_pair_mismatch_risk",
    ]
    for col in cols:
        s = pd.to_numeric(train[col], errors="coerce").dropna()
        if s.empty:
            continue
        for q in [0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            out[f"{col}_q{int(q * 100)}"] = float(s.quantile(q))
    return out


def filter_masks(df: pd.DataFrame, q: dict[str, float]) -> list[tuple[str, pd.Series]]:
    idx = df.index
    fit = pd.to_numeric(df["official_pair_fit_max_avg"], errors="coerce").fillna(0.0)
    fit_min = pd.to_numeric(df["official_pair_fit_max_min"], errors="coerce").fillna(0.0)
    fit_mean = pd.to_numeric(df["official_pair_fit_mean_avg"], errors="coerce").fillna(0.0)
    need_match = pd.to_numeric(df["official_pair_need_match_avg"], errors="coerce").fillna(0.0)
    strength = pd.to_numeric(df["official_pair_strength_avg"], errors="coerce").fillna(0.0)
    risk = pd.to_numeric(df["official_pair_mismatch_risk"], errors="coerce").fillna(1.0)
    ready_both = df["official_pair_ready_both"].fillna(False).astype(bool)
    ready_any = df["official_pair_ready_any"].fillna(False).astype(bool)
    return [
        ("base", pd.Series(True, index=idx)),
        ("official_ready_both", ready_both),
        ("official_ready_any_fit_q60", ready_any & fit.ge(q.get("official_pair_fit_max_avg_q60", 1.0))),
        ("official_fit_q60", fit.ge(q.get("official_pair_fit_max_avg_q60", 1.0))),
        ("official_fit_q70", fit.ge(q.get("official_pair_fit_max_avg_q70", 1.0))),
        ("official_minfit_q55", fit_min.ge(q.get("official_pair_fit_max_min_q55", 1.0))),
        ("official_meanfit_q60", fit_mean.ge(q.get("official_pair_fit_mean_avg_q60", 1.0))),
        (
            "official_fit_q60_risk_q55",
            fit.ge(q.get("official_pair_fit_max_avg_q60", 1.0))
            & risk.le(q.get("official_pair_mismatch_risk_q55", 0.0)),
        ),
        (
            "official_fit_q65_need_q55",
            fit.ge(q.get("official_pair_fit_max_avg_q65", 1.0))
            & need_match.ge(q.get("official_pair_need_match_avg_q55", 1.0)),
        ),
        (
            "official_fit_q60_strength_q50",
            fit.ge(q.get("official_pair_fit_max_avg_q60", 1.0))
            & strength.ge(q.get("official_pair_strength_avg_q50", 1.0)),
        ),
    ]


def evaluate(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    enriched["year"] = pd.to_numeric(enriched.get("year"), errors="coerce").fillna(
        pd.to_numeric(enriched["race_id"].str[:4], errors="coerce")
    )
    for (policy, ticket_type), group in enriched.groupby(["policy", "ticket_type"], dropna=False):
        group = group.copy()
        train = group[group["year"].lt(2026)]
        oos = group[group["year"].ge(2026)]
        if train.empty:
            continue
        q = quantiles(train)
        for filter_name, mask in filter_masks(group, q):
            selected = group[mask.fillna(False)].copy()
            rows.append(metrics(selected[selected["year"].lt(2026)], str(policy), str(ticket_type), filter_name, "train_pre2026"))
            rows.append(metrics(selected[selected["year"].ge(2026)], str(policy), str(ticket_type), filter_name, "oos_2026"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TARGET RA official lap-history overlay on existing pair tickets.")
    parser.add_argument("--tickets-csv", type=Path, default=DEFAULT_TICKETS)
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tickets_csv = project_path(args.tickets_csv)
    history_csv = project_path(args.history_csv)
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tickets = read_csv_any(tickets_csv)
    tickets["race_id"] = clean_id(tickets["race_id"])
    hist = load_history(history_csv)
    enriched = add_overlay_features(tickets, hist)
    metrics_df = evaluate(enriched)

    enriched_out = out_dir / "tickets_with_target_ra_official_lap_overlay.csv"
    metrics_out = out_dir / "metrics_by_policy_filter.csv"
    leaderboard_out = out_dir / "oos_leaderboard_min50_tickets.csv"
    summary_out = out_dir / "summary.json"
    enriched.to_csv(enriched_out, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_out, index=False, encoding="utf-8-sig")

    oos = metrics_df[(metrics_df["split"].eq("oos_2026")) & metrics_df["tickets"].ge(50)].copy()
    if not oos.empty:
        oos = oos.sort_values(["roi_pct", "profit_yen", "tickets"], ascending=[False, False, False])
    oos.to_csv(leaderboard_out, index=False, encoding="utf-8-sig")

    base_oos = metrics_df[(metrics_df["split"].eq("oos_2026")) & metrics_df["filter_name"].eq("base")]
    summary = {
        "tickets_csv": str(tickets_csv),
        "history_csv": str(history_csv),
        "out_dir": str(out_dir),
        "input_tickets": int(len(tickets)),
        "enriched_tickets": int(len(enriched)),
        "history_ready_both_pct": round(float(enriched["official_pair_ready_both"].mean() * 100.0), 1)
        if len(enriched)
        else 0.0,
        "history_ready_any_pct": round(float(enriched["official_pair_ready_any"].mean() * 100.0), 1)
        if len(enriched)
        else 0.0,
        "base_oos_rows": base_oos.to_dict("records"),
        "best_oos_min50": oos.head(20).to_dict("records"),
        "enriched_csv": str(enriched_out),
        "metrics_csv": str(metrics_out),
        "leaderboard_csv": str(leaderboard_out),
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not oos.empty:
        print(oos.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
