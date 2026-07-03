from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = ROOT / "outputs" / "analysis" / "pair_joint_probability_v2_rebuilt_20260623" / "pair_universe_with_joint_v2_features.csv"
DEFAULT_RACE_SIM = ROOT / "outputs" / "analysis" / "race_sim_umaren_probability_v2" / "race_sim_pair_universe_calibrated.csv"
DEFAULT_OUT = ROOT / "outputs" / "analysis" / "umaren_pair_ranking_precision_v1"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = df[col]
    if values.dtype == object or str(values.dtype).startswith("string"):
        values = values.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def clip01(s: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(s, pd.Series):
        x = s
    else:
        x = pd.Series(s)
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def norm01(s: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if lo is None:
        lo = float(x.quantile(0.05)) if x.notna().any() else 0.0
    if hi is None:
        hi = float(x.quantile(0.95)) if x.notna().any() else 1.0
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=x.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5).clip(0.0, 1.0)


def sigmoid(x: pd.Series | np.ndarray) -> pd.Series:
    arr = np.asarray(x, dtype=float)
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(arr, -35.0, 35.0))))


def pair_key(df: pd.DataFrame, a_col: str = "anchor_no", b_col: str = "partner_no") -> pd.Series:
    a = num(df, a_col, np.nan)
    b = num(df, b_col, np.nan)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return df["race_id"].astype(str) + ":" + lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)


def add_race_sim(universe: pd.DataFrame, race_sim: pd.DataFrame) -> pd.DataFrame:
    out = universe.copy()
    if race_sim.empty:
        return out
    sim = race_sim.copy()
    sim["race_id"] = sim["race_id"].astype(str)
    sim["pair_key_norm"] = pair_key(sim)
    out["pair_key_norm"] = pair_key(out)
    keep = [
        "pair_key_norm",
        "race_sim_umaren_prob_raw",
        "race_sim_umaren_prob_cal",
        "race_sim_neutral_prob",
        "race_sim_front_prob",
        "race_sim_collapse_prob",
        "race_sim_front_share",
        "race_sim_collapse_share",
    ]
    sim = sim[[c for c in keep if c in sim.columns]].drop_duplicates("pair_key_norm")
    return out.merge(sim, on="pair_key_norm", how="left")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    if "year" not in out.columns:
        out["year"] = pd.to_numeric(out["race_id"].str[:4], errors="coerce").astype("Int64")
    else:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(pd.to_numeric(out["race_id"].str[:4], errors="coerce")).astype("Int64")
    out["pair_key_norm"] = pair_key(out)
    out["umaren_hit_bool"] = num(out, "umaren_hit", 0.0).gt(0) | num(out, "umaren_label", 0.0).gt(0)
    out["umaren_return_100_eval"] = num(out, "umaren_pay", 0.0).where(out["umaren_hit_bool"], 0.0)
    out["umaren_quote"] = num(out, "umaren_pay", np.nan).where(num(out, "umaren_pay", np.nan).gt(0), num(out, "umaren_quote_proxy", np.nan))
    out["anchor_rank_score"] = (1.0 - (num(out, "anchor_quinella_model_rank", 99.0) - 1.0) / 17.0).clip(0.0, 1.0)
    out["partner_rank_score"] = (1.0 - (num(out, "partner_quinella_model_rank", 99.0) - 1.0) / 17.0).clip(0.0, 1.0)
    out["rank_balance"] = np.sqrt((out["anchor_rank_score"] * out["partner_rank_score"]).clip(0, 1))
    out["pop_balance"] = np.sqrt(((1.0 / num(out, "anchor_pop", 99).clip(lower=1)) * (1.0 / num(out, "partner_pop", 99).clip(lower=1))).clip(0, 1))
    out["ai_market_blend"] = (
        0.24 * clip01(num(out, "pair_quinella_score", 0.0))
        + 0.16 * clip01(num(out, "joint_q_product", 0.0))
        + 0.15 * clip01(num(out, "joint_place_product", 0.0))
        + 0.14 * clip01(num(out, "joint_market_product", 0.0))
        + 0.10 * clip01(num(out, "rank_balance", 0.0))
        + 0.10 * clip01(num(out, "market_overlay_score", 0.0))
        + 0.07 * clip01(num(out, "late_value_survives_score", 0.0))
        + 0.04 * clip01(num(out, "projected_front5_prob", 0.0))
    ).clip(0.0, 1.0)

    front_min = clip01(num(out, "front_pair_min", num(out, "projected_front5_prob", 0.5)))
    front_max = clip01(num(out, "front_pair_max", num(out, "projected_front5_prob", 0.5)))
    closer_max = clip01(num(out, "closer_pair_max", 0.0))
    front_slow = clip01(num(out, "front_front_slow_fit", 0.0))
    collapse_fit = clip01(num(out, "collapse_fit", 0.0))
    diversity = clip01(num(out, "style_diversity", 0.0))
    danger = clip01(num(out, "danger_max", 0.0))
    skip = clip01(num(out, "skip_risk_score", 0.0))
    odds_geom = norm01(num(out, "odds_geom", np.nan).fillna(np.sqrt(num(out, "anchor_odds", 1.0) * num(out, "partner_odds", 1.0))), lo=1.5, hi=30.0)

    out["pair_exactness_score"] = (
        0.28 * out["ai_market_blend"]
        + 0.15 * clip01(num(out, "pair_score", 0.0))
        + 0.14 * front_max
        + 0.10 * front_min
        + 0.09 * closer_max
        + 0.08 * front_slow
        + 0.07 * collapse_fit
        + 0.05 * diversity
        + 0.04 * clip01(num(out, "partner_value_flag", 0.0))
        - 0.10 * danger
        - 0.07 * skip
    ).clip(0.0, 1.0)
    sim_prob = num(out, "race_sim_umaren_prob_cal", np.nan)
    out["race_sim_prob_score"] = norm01(sim_prob, lo=0.015, hi=0.14).where(sim_prob.notna(), 0.5)
    out["pair_exactness_sim_blend_score"] = (
        0.72 * out["pair_exactness_score"] + 0.28 * out["race_sim_prob_score"]
    ).clip(0.0, 1.0)
    out["pair_value_score_capped"] = (
        out["pair_exactness_sim_blend_score"]
        * (0.78 + 0.22 * odds_geom)
        * (0.86 + 0.14 * clip01(num(out, "market_overlay_score", 0.0)))
    ).clip(0.0, 1.25)
    out["pair_ev_like_score"] = (
        out["pair_exactness_sim_blend_score"]
        * num(out, "umaren_quote", np.nan).fillna(num(out, "umaren_quote_proxy", 0.0)).clip(0.0, 12000.0)
        / 100.0
    )
    return out


def selection_metrics(selected: pd.DataFrame, score_col: str) -> dict[str, Any]:
    if selected.empty:
        return {
            "score": score_col,
            "tickets": 0,
            "races": 0,
            "hit_rate_pct": 0.0,
            "roi_pct": 0.0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "top5_removed_roi_pct": 0.0,
            "top10_removed_roi_pct": 0.0,
            "avg_pay_hit": 0.0,
        }
    stake = pd.Series(100.0, index=selected.index)
    ret = num(selected, "umaren_return_100_eval", 0.0)
    hit = ret.gt(0)
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())

    def removed_roi(n: int) -> float:
        if len(ret) <= n:
            return 0.0
        kept = ret.drop(index=ret.sort_values(ascending=False).index[:n])
        kept_stake = stake.drop(index=ret.sort_values(ascending=False).index[:n])
        return float(100.0 * kept.sum() / kept_stake.sum()) if kept_stake.sum() else 0.0

    return {
        "score": score_col,
        "tickets": int(len(selected)),
        "races": int(selected["race_id"].nunique()),
        "hit_rate_pct": float(100.0 * hit.mean()),
        "roi_pct": float(100.0 * ret_sum / stake_sum) if stake_sum else 0.0,
        "stake_yen": stake_sum,
        "return_yen": ret_sum,
        "profit_yen": ret_sum - stake_sum,
        "top5_removed_roi_pct": removed_roi(5),
        "top10_removed_roi_pct": removed_roi(10),
        "avg_pay_hit": float(ret.loc[hit].mean()) if hit.any() else 0.0,
    }


def select_top(df: pd.DataFrame, score_col: str, top_n: int = 1, gate: pd.Series | None = None) -> pd.DataFrame:
    work = df.copy()
    if gate is not None:
        work = work[gate.reindex(work.index).fillna(False)].copy()
    if work.empty:
        return work
    work["_score"] = num(work, score_col, -999.0)
    work["_rank"] = work.groupby("race_id")["_score"].rank(method="first", ascending=False)
    return work[work["_rank"].le(top_n)].copy()


def rank_diagnostics(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    for score in score_cols:
        if score not in df.columns:
            continue
        work = df.copy()
        work["_score"] = num(work, score, -999.0)
        work["_rank"] = work.groupby("race_id")["_score"].rank(method="first", ascending=False)
        actual = work[work["umaren_hit_bool"]].copy()
        if actual.empty:
            continue
        rows.append(
            {
                "score": score,
                "actual_pairs": int(len(actual)),
                "median_actual_pair_rank": float(actual["_rank"].median()),
                "mean_actual_pair_rank": float(actual["_rank"].mean()),
                "actual_pair_rank_le1_pct": float(100.0 * actual["_rank"].le(1).mean()),
                "actual_pair_rank_le3_pct": float(100.0 * actual["_rank"].le(3).mean()),
                "actual_pair_rank_le5_pct": float(100.0 * actual["_rank"].le(5).mean()),
                "actual_pair_rank_le10_pct": float(100.0 * actual["_rank"].le(10).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["actual_pair_rank_le3_pct", "median_actual_pair_rank"], ascending=[False, True])


def evaluate(df: pd.DataFrame, score_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gate_any = pd.Series(True, index=df.index)
    # A practical gate: keep all realistic pairs, but avoid extreme lottery-only pairs.
    odds = num(df, "umaren_quote", np.nan).fillna(num(df, "umaren_quote_proxy", np.nan))
    practical_gate = (
        num(df, "pair_quinella_score", 0.0).ge(0.42)
        & num(df, "anchor_quinella_score", 0.0).ge(0.28)
        & num(df, "partner_quinella_score", 0.0).ge(0.24)
        & odds.between(250.0, 12000.0)
    )
    strict_gate = practical_gate & num(df, "pair_exactness_sim_blend_score", 0.0).ge(0.50) & num(df, "danger_max", 0.0).le(0.85)

    rows = []
    selections = []
    for score in score_cols:
        if score not in df.columns:
            continue
        for gate_name, gate in [("all", gate_any), ("practical", practical_gate), ("strict_exactness", strict_gate)]:
            for top_n in [1, 2, 3]:
                selected = select_top(df, score, top_n=top_n, gate=gate)
                selected["policy"] = f"{score}|{gate_name}|top{top_n}"
                selected["score_used"] = score
                selected["gate_used"] = gate_name
                selected["top_n"] = top_n
                m = selection_metrics(selected, f"{score}|{gate_name}|top{top_n}")
                rows.append(m)
                selections.append(selected)

    return pd.DataFrame(rows).sort_values(["top10_removed_roi_pct", "roi_pct"], ascending=[False, False]), rank_diagnostics(df, score_cols), pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate umaren pair ranking precision separately from ticket EV selection.")
    parser.add_argument("--universe-csv", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--race-sim-csv", default=str(DEFAULT_RACE_SIM))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--write-heavy-detail", action="store_true", help="Write full scored universe and all selected-pair rows.")
    args = parser.parse_args()

    universe_path = project_path(args.universe_csv)
    race_sim_path = project_path(args.race_sim_csv)
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv(universe_path)
    race_sim = read_csv(race_sim_path) if race_sim_path.exists() else pd.DataFrame()
    df = add_race_sim(df, race_sim)
    df = prepare(df)

    score_cols = [
        "pair_quinella_score",
        "pair_score",
        "ai_market_blend",
        "pair_exactness_score",
        "race_sim_prob_score",
        "pair_exactness_sim_blend_score",
        "pair_value_score_capped",
        "pair_ev_like_score",
    ]
    summary, rank_diag, selections = evaluate(df, score_cols)
    yearly_rows = []
    for policy, g in selections.groupby("policy", sort=False):
        for year, gy in g.groupby("year"):
            m = selection_metrics(gy, f"{policy}|{int(year)}")
            m["policy"] = policy
            m["year"] = int(year)
            yearly_rows.append(m)
    yearly = pd.DataFrame(yearly_rows).sort_values(["policy", "year"]) if yearly_rows else pd.DataFrame()

    summary_path = out_dir / "umaren_pair_ranking_policy_summary.csv"
    rank_path = out_dir / "umaren_actual_pair_rank_diagnostics.csv"
    yearly_path = out_dir / "umaren_pair_ranking_policy_by_year.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    rank_diag.to_csv(rank_path, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    if args.write_heavy_detail:
        enriched_path = out_dir / "umaren_pair_universe_with_precision_scores.csv"
        selections_path = out_dir / "umaren_pair_ranking_selected_pairs.csv"
        df.to_csv(enriched_path, index=False, encoding="utf-8-sig")
        selections.to_csv(selections_path, index=False, encoding="utf-8-sig")

    best = summary.head(20).to_dict(orient="records")
    report = {
        "universe_csv": str(universe_path),
        "race_sim_csv": str(race_sim_path),
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "actual_umaren_pairs": int(df["umaren_hit_bool"].sum()),
        "best_by_top10_removed_roi": best[0] if best else {},
        "top20_policies": best,
        "rank_diagnostics": rank_diag.to_dict(orient="records"),
        "heavy_detail_written": bool(args.write_heavy_detail),
        "notes": [
            "This test ranks pair candidates inside each race; it is separate from final BUY sizing.",
            "pair_exactness_* intentionally caps odds influence so that high-payout pairs do not dominate ranking by themselves.",
            "If top10_removed_roi is weak, the apparent ROI is likely driven by a few large payouts.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
