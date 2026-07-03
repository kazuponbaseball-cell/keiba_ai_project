from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def clip01(s: pd.Series | np.ndarray | float) -> pd.Series:
    return pd.Series(s).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def norm01(s: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if lo is None:
        lo = float(x.quantile(0.05)) if x.notna().any() else 0.0
    if hi is None:
        hi = float(x.quantile(0.95)) if x.notna().any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=x.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5).clip(0.0, 1.0)


def sigmoid(x: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -35.0, 35.0))))


def race_date_from_id(race_id: pd.Series) -> pd.Series:
    digits = race_id.astype(str).str.extract(r"(\d{8})", expand=False)
    return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")


def pair_key(df: pd.DataFrame, a_col: str = "anchor_no", b_col: str = "partner_no") -> pd.Series:
    a = pd.to_numeric(df[a_col], errors="coerce")
    b = pd.to_numeric(df[b_col], errors="coerce")
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return df["race_id"].astype(str) + ":" + lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)


def add_front5_model(runners: pd.DataFrame, front5_predictions_csv: Path | None) -> pd.DataFrame:
    out = runners.copy()
    if front5_predictions_csv is not None and front5_predictions_csv.exists():
        pred = pd.read_csv(front5_predictions_csv, dtype={"race_id": str}, low_memory=False)
        need = ["race_id", "horse_no", "front5_model_prob"]
        pred = pred[[c for c in need if c in pred.columns]].copy()
        if set(need).issubset(pred.columns):
            pred["race_id"] = pred["race_id"].astype(str)
            pred["horse_no"] = pd.to_numeric(pred["horse_no"], errors="coerce")
            pred = pred.drop_duplicates(["race_id", "horse_no"], keep="last")
            out = out.merge(pred, on=["race_id", "horse_no"], how="left")
    if "front5_model_prob" not in out.columns:
        out["front5_model_prob"] = np.nan
    out["front5_model_prob"] = pd.to_numeric(out["front5_model_prob"], errors="coerce")
    out["front5_model_or_heuristic_prob"] = out["front5_model_prob"].fillna(num(out, "projected_front5_prob", 0.5)).clip(0.001, 0.999)
    out["front5_model_available"] = out["front5_model_prob"].notna()
    return out


def prepare_runner_scores(runners: pd.DataFrame) -> pd.DataFrame:
    out = runners.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce")
    out["_date"] = race_date_from_id(out["race_id"])
    out["year"] = pd.to_numeric(out.get("year", out["race_id"].str[:4]), errors="coerce").astype("Int64")

    market = num(out, "market_win_prob_norm", np.nan)
    odds = num(out, "odds", np.nan).replace(0, np.nan)
    inv_odds = 1.0 / odds
    inv_norm = inv_odds / inv_odds.groupby(out["race_id"]).transform("sum")
    market = market.fillna(inv_norm).fillna(0.0)
    market = market / market.groupby(out["race_id"]).transform("sum").replace(0, np.nan)
    out["_market_prior"] = market.fillna(1.0 / num(out, "出走頭数", 14.0).replace(0, np.nan)).clip(0.0005, 0.85)

    front = out["front5_model_or_heuristic_prob"].astype(float).clip(0.001, 0.999)
    closer = (
        0.42 * clip01(num(out, "horse_closer_rate_past5", 0.0)).reset_index(drop=True)
        + 0.28 * clip01(num(out, "closing_tendency_y", 0.0)).reset_index(drop=True)
        + 0.18 * (1.0 - front.reset_index(drop=True))
        + 0.12 * norm01(num(out, "race_pace_collapse_risk", 0.0)).reset_index(drop=True)
    ).clip(0.0, 1.0)
    closer.index = out.index

    base = (
        0.28 * norm01(out["_market_prior"]).reset_index(drop=True)
        + 0.17 * clip01(num(out, "ai_win_prob_proxy", 0.0)).reset_index(drop=True)
        + 0.16 * clip01(num(out, "quinella_model_score_norm", 0.0)).reset_index(drop=True)
        + 0.12 * clip01(num(out, "place_suitability_score", 0.0)).reset_index(drop=True)
        + 0.10 * clip01(num(out, "win_suitability_score", 0.0)).reset_index(drop=True)
        + 0.08 * clip01(num(out, "market_overlay_score", 0.0)).reset_index(drop=True)
        + 0.06 * front.reset_index(drop=True)
        + 0.03 * clip01(num(out, "late_value_survives_score", 0.0)).reset_index(drop=True)
        - 0.08 * clip01(num(out, "danger_favorite_score", 0.0)).reset_index(drop=True)
        - 0.05 * clip01(num(out, "skip_risk_score", 0.0)).reset_index(drop=True)
    )
    base.index = out.index
    out["_base_score"] = base.clip(0.0, 1.0)
    out["_front_score"] = (out["_base_score"] + 0.35 * front + 0.14 * clip01(num(out, "front_advantage_score", 0.0))).clip(0.0, 1.5)
    out["_closer_score"] = (out["_base_score"] + 0.38 * closer + 0.08 * clip01(num(out, "race_pace_collapse_risk", 0.0))).clip(0.0, 1.5)
    return out


def make_weights(g: pd.DataFrame, score_col: str, scale: float) -> pd.Series:
    centered = g[score_col].astype(float) - float(g[score_col].astype(float).mean())
    market = g["_market_prior"].astype(float).clip(0.0005, 0.85)
    weights = np.power(market, 0.45) * np.exp(np.clip(scale * centered, -4.0, 4.0))
    return pd.Series(weights, index=g.index).clip(1e-6, None)


def unordered_top2_prob(weights: pd.Series, idx_a: int, idx_b: int) -> float:
    s = float(weights.sum())
    wa = float(weights.loc[idx_a])
    wb = float(weights.loc[idx_b])
    if s <= 0 or wa <= 0 or wb <= 0 or s <= max(wa, wb):
        return np.nan
    return float((wa / s) * (wb / max(s - wa, 1e-9)) + (wb / s) * (wa / max(s - wb, 1e-9)))


def scenario_mix(g: pd.DataFrame) -> tuple[float, float, float]:
    slow = float(num(g, "race_slow_pace_risk", 0.0).mean())
    collapse = float(num(g, "race_pace_collapse_risk", 0.0).mean())
    pressure = float(num(g, "race_early_pressure_score", 0.0).mean())
    front_adv = float(num(g, "front_advantage_score", 0.0).mean())
    expected_fast = float(g.get("expected_pace", pd.Series("", index=g.index)).astype(str).eq("fast").mean())
    front_share = np.clip(0.20 + 0.34 * slow + 0.20 * front_adv - 0.25 * collapse - 0.10 * expected_fast, 0.05, 0.65)
    collapse_share = np.clip(0.15 + 0.36 * collapse + 0.16 * expected_fast + 0.08 * pressure - 0.15 * slow, 0.05, 0.65)
    total = front_share + collapse_share
    if total > 0.88:
        front_share *= 0.88 / total
        collapse_share *= 0.88 / total
    neutral_share = 1.0 - front_share - collapse_share
    return float(neutral_share), float(front_share), float(collapse_share)


def score_pairs_by_race(runners: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    pair_rows = []
    runner_groups = {rid: g for rid, g in runners.groupby("race_id", sort=False)}
    for race_id, p in pairs.groupby("race_id", sort=False):
        g = runner_groups.get(race_id)
        if g is None or g.empty:
            continue
        g = g.dropna(subset=["horse_no"]).copy()
        horse_to_idx = {int(h): idx for h, idx in zip(g["horse_no"].astype(int), g.index)}
        neutral_w = make_weights(g, "_base_score", scale=3.4)
        front_w = make_weights(g, "_front_score", scale=3.1)
        collapse_w = make_weights(g, "_closer_score", scale=3.0)
        neutral_share, front_share, collapse_share = scenario_mix(g)
        for _, row in p.iterrows():
            a = row["horse_a"]
            b = row["horse_b"]
            if pd.isna(a) or pd.isna(b):
                continue
            ia = horse_to_idx.get(int(a))
            ib = horse_to_idx.get(int(b))
            if ia is None or ib is None or ia == ib:
                continue
            neutral = unordered_top2_prob(neutral_w, ia, ib)
            front = unordered_top2_prob(front_w, ia, ib)
            collapse = unordered_top2_prob(collapse_w, ia, ib)
            raw = neutral_share * neutral + front_share * front + collapse_share * collapse
            pair_rows.append(
                {
                    "race_id": race_id,
                    "horse_a": int(min(a, b)),
                    "horse_b": int(max(a, b)),
                    "race_sim_pair_key": f"{race_id}:{int(min(a, b))}-{int(max(a, b))}",
                    "race_sim_umaren_prob_raw": raw,
                    "race_sim_neutral_prob": neutral,
                    "race_sim_front_prob": front,
                    "race_sim_collapse_prob": collapse,
                    "race_sim_neutral_share": neutral_share,
                    "race_sim_front_share": front_share,
                    "race_sim_collapse_share": collapse_share,
                    "race_sim_field_size": int(len(g)),
                    "race_sim_front5_model_available_rate": float(g["front5_model_available"].mean()),
                }
            )
    return pd.DataFrame(pair_rows)


class BinCalibrator:
    def __init__(self, bins: pd.DataFrame, fallback: float):
        self.bins = bins
        self.fallback = float(fallback)

    def apply(self, raw: pd.Series) -> pd.Series:
        x = pd.to_numeric(raw, errors="coerce")
        out = pd.Series(self.fallback, index=x.index, dtype=float)
        if self.bins.empty:
            return out.clip(0.001, 0.95)
        ordered = self.bins.sort_values("raw_min")
        for _, row in ordered.iterrows():
            mask = x.between(float(row["raw_min"]), float(row["raw_max"]), inclusive="both")
            out.loc[mask] = float(row["prob"])
        out.loc[x.lt(float(ordered["raw_min"].min()))] = float(ordered.iloc[0]["prob"])
        out.loc[x.gt(float(ordered["raw_max"].max()))] = float(ordered.iloc[-1]["prob"])
        return out.clip(0.001, 0.95)


def fit_calibrator(train: pd.DataFrame, feature: str, label: str, bins: int = 12, smoothing: float = 40.0) -> BinCalibrator:
    use = train[[feature, label]].dropna().copy()
    use[feature] = pd.to_numeric(use[feature], errors="coerce")
    use[label] = pd.to_numeric(use[label], errors="coerce")
    use = use.dropna()
    fallback = float(use[label].mean()) if len(use) else 0.02
    if len(use) < max(80, bins * 12) or use[feature].nunique() < 4:
        return BinCalibrator(pd.DataFrame(), fallback)
    ranked = use[feature].rank(method="first")
    try:
        use["_bin"] = pd.qcut(ranked, bins, labels=False, duplicates="drop")
    except ValueError:
        return BinCalibrator(pd.DataFrame(), fallback)
    grouped = (
        use.groupby("_bin", observed=True)
        .agg(raw_min=(feature, "min"), raw_max=(feature, "max"), n=(label, "size"), hit_rate=(label, "mean"))
        .reset_index(drop=True)
        .sort_values("raw_min")
    )
    grouped["prob"] = (grouped["hit_rate"] * grouped["n"] + fallback * smoothing) / (grouped["n"] + smoothing)
    grouped["prob"] = grouped["prob"].cummax().clip(0.001, 0.95)
    return BinCalibrator(grouped, fallback)


def calibrate_monthly_oos(pairs: pd.DataFrame, min_train_rows: int, purge_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pairs.copy()
    out["race_sim_umaren_prob_cal"] = np.nan
    out["race_sim_calibration_train_rows"] = 0
    out["race_sim_calibration_fallback"] = np.nan
    out["_date"] = race_date_from_id(out["race_id"])
    out["year"] = out["_date"].dt.year
    rows = []
    months = sorted(out["_date"].dropna().dt.to_period("M").unique())
    for month in months:
        start = pd.Timestamp(month.start_time)
        end = start + pd.offsets.MonthBegin(1)
        train_end = start - pd.Timedelta(days=purge_days)
        train_mask = out["_date"].lt(train_end) & out["race_sim_umaren_prob_raw"].notna() & out["umaren_hit"].notna()
        test_mask = out["_date"].ge(start) & out["_date"].lt(end) & out["race_sim_umaren_prob_raw"].notna()
        if int(train_mask.sum()) < min_train_rows or int(test_mask.sum()) == 0:
            continue
        cal = fit_calibrator(out.loc[train_mask], "race_sim_umaren_prob_raw", "umaren_hit")
        out.loc[test_mask, "race_sim_umaren_prob_cal"] = cal.apply(out.loc[test_mask, "race_sim_umaren_prob_raw"])
        out.loc[test_mask, "race_sim_calibration_train_rows"] = int(train_mask.sum())
        out.loc[test_mask, "race_sim_calibration_fallback"] = cal.fallback
        rows.append(
            {
                "test_month": start.strftime("%Y-%m"),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "fallback": cal.fallback,
                "bins": int(len(cal.bins)),
                "raw_avg": float(out.loc[test_mask, "race_sim_umaren_prob_raw"].mean()),
                "cal_avg": float(out.loc[test_mask, "race_sim_umaren_prob_cal"].mean()),
                "actual_hit_rate": float(pd.to_numeric(out.loc[test_mask, "umaren_hit"], errors="coerce").fillna(0.0).mean()),
            }
        )
    return out, pd.DataFrame(rows)


def prepare_pairs_for_scoring(universe: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for df, label in [(universe, "universe"), (tickets, "tickets")]:
        tmp = df[["race_id", "anchor_no", "partner_no"]].copy()
        tmp["race_id"] = tmp["race_id"].astype(str)
        tmp["horse_a"] = np.minimum(pd.to_numeric(tmp["anchor_no"], errors="coerce"), pd.to_numeric(tmp["partner_no"], errors="coerce"))
        tmp["horse_b"] = np.maximum(pd.to_numeric(tmp["anchor_no"], errors="coerce"), pd.to_numeric(tmp["partner_no"], errors="coerce"))
        tmp["source"] = label
        frames.append(tmp[["race_id", "horse_a", "horse_b", "source"]])
    pairs = pd.concat(frames, ignore_index=True).dropna(subset=["race_id", "horse_a", "horse_b"])
    pairs["horse_a"] = pairs["horse_a"].astype(int)
    pairs["horse_b"] = pairs["horse_b"].astype(int)
    pairs = pairs.drop_duplicates(["race_id", "horse_a", "horse_b"])
    return pairs


def apply_to_tickets(tickets: pd.DataFrame, sim_pairs: pd.DataFrame, blend_weight: float) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["race_sim_pair_key"] = pair_key(out)
    keep = [
        "race_sim_pair_key",
        "race_sim_umaren_prob_raw",
        "race_sim_umaren_prob_cal",
        "race_sim_neutral_prob",
        "race_sim_front_prob",
        "race_sim_collapse_prob",
        "race_sim_neutral_share",
        "race_sim_front_share",
        "race_sim_collapse_share",
        "race_sim_calibration_train_rows",
        "race_sim_front5_model_available_rate",
    ]
    out = out.merge(sim_pairs[[c for c in keep if c in sim_pairs.columns]].drop_duplicates("race_sim_pair_key"), on="race_sim_pair_key", how="left")
    out["race_sim_probability_available"] = out["race_sim_umaren_prob_cal"].notna() & out.get("ticket_type", "").astype(str).eq("umaren")
    old_prob = num(out, "ticket_hit_prob", np.nan)
    sim_prob = num(out, "race_sim_umaren_prob_cal", np.nan)
    out["ticket_hit_prob_before_race_sim"] = old_prob
    out["min_odds_margin_ratio_before_race_sim"] = num(out, "min_odds_margin_ratio", np.nan)
    blend = float(np.clip(blend_weight, 0.0, 1.0))
    mask = out["race_sim_probability_available"] & old_prob.gt(0)
    out.loc[mask, "ticket_hit_prob"] = ((1.0 - blend) * old_prob.loc[mask] + blend * sim_prob.loc[mask]).clip(0.001, 0.95)
    out["race_sim_blend_weight"] = np.where(mask, blend, 0.0)
    quote = num(out, "quote_pay_proxy_per100", np.nan).fillna(num(out, "runtime_pay_per100", 0.0))
    out.loc[mask, "runtime_expected_roi"] = out.loc[mask, "ticket_hit_prob"] * quote.loc[mask] / 100.0
    old_required = num(out, "required_pay_per100", np.nan)
    old_target = old_prob * old_required / 100.0
    target = old_target.where(old_target.gt(0), 1.35)
    out.loc[mask, "required_pay_per100"] = (100.0 * target.loc[mask] / out.loc[mask, "ticket_hit_prob"].replace(0, np.nan)).clip(100.0, 20000.0)
    out.loc[mask, "min_acceptable_odds"] = out.loc[mask, "required_pay_per100"] / 100.0
    out.loc[mask, "min_odds_margin_ratio"] = (
        quote.loc[mask] / out.loc[mask, "required_pay_per100"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    note = pd.Series("not_umaren_kept_existing", index=out.index, dtype=object)
    note.loc[out.get("ticket_type", "").astype(str).eq("umaren")] = "race_sim_unavailable_kept_existing"
    note.loc[mask] = "race_sim_umaren_probability_blended"
    out["race_sim_runtime_note"] = note
    return out


def race_table(df: pd.DataFrame, stake: pd.Series, return_col: str) -> pd.DataFrame:
    selected = df[stake.gt(0)].copy()
    selected["_eval_stake"] = stake.loc[selected.index]
    base_stake = num(selected, "_base_stake", 0.0).replace(0, np.nan)
    pay_ratio = num(selected, return_col, 0.0) / base_stake
    selected["_eval_return"] = np.where(num(selected, return_col, 0.0).gt(0), pay_ratio.fillna(0.0) * selected["_eval_stake"], 0.0)
    if selected.empty:
        return pd.DataFrame(columns=["race_id", "date", "stake_yen", "return_yen", "profit_yen", "hit"])
    race = (
        selected.groupby("race_id", sort=False)
        .agg(date=("_date", "min"), stake_yen=("_eval_stake", "sum"), return_yen=("_eval_return", "sum"))
        .reset_index()
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    race["hit"] = race["return_yen"].gt(0)
    return race.sort_values(["date", "race_id"])


def metrics(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    stake = df["_base_stake"].where(mask, 0.0)
    selected = df[stake.gt(0)].copy()
    if selected.empty:
        return {"policy": label, "tickets": 0, "races": 0, "stake_yen": 0.0, "return_yen": 0.0, "roi": 0.0}
    race = race_table(df, stake, "_base_return")
    stake_sum = float(stake.loc[selected.index].sum())
    return_sum = float(race["return_yen"].sum())

    def removed_roi(n: int) -> float:
        if len(race) <= n:
            return 0.0
        kept = race.sort_values("profit_yen", ascending=False).iloc[n:]
        kept_stake = float(kept["stake_yen"].sum())
        return float(kept["return_yen"].sum() / kept_stake) if kept_stake else 0.0

    return {
        "policy": label,
        "tickets": int(len(selected)),
        "races": int(selected["race_id"].nunique()),
        "stake_yen": stake_sum,
        "return_yen": return_sum,
        "profit_yen": return_sum - stake_sum,
        "roi": return_sum / stake_sum if stake_sum else 0.0,
        "ticket_hit_rate": float(num(selected, "_base_return", 0.0).gt(0).mean()),
        "race_hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "top5_removed_roi": removed_roi(5),
        "top10_removed_roi": removed_roi(10),
        "avg_existing_prob": float(num(selected, "ticket_hit_prob_before_race_sim", np.nan).mean()),
        "avg_sim_prob": float(num(selected, "race_sim_umaren_prob_cal", np.nan).mean()),
        "sim_available_rate": float(selected["race_sim_probability_available"].mean()),
        "avg_front_share": float(num(selected, "race_sim_front_share", np.nan).mean()),
        "avg_collapse_share": float(num(selected, "race_sim_collapse_share", np.nan).mean()),
    }


def evaluate_policies(tickets: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    out["_date"] = race_date_from_id(out["race_id"])
    out["_base_stake"] = num(out, "runtime_stake_yen", 0.0).fillna(num(out, "stake_yen", 0.0)).fillna(0.0)
    out["_base_return"] = num(out, "runtime_return_yen", 0.0).fillna(num(out, "return_yen", 0.0)).fillna(0.0)
    margin = num(out, "min_odds_margin_ratio", 0.0).fillna(0.0)
    existing_prob = num(out, "ticket_hit_prob_before_race_sim", np.nan)
    sim_prob = num(out, "race_sim_umaren_prob_cal", np.nan)
    expected_roi = num(out, "runtime_expected_roi", np.nan).fillna(num(out, "expected_roi_after_slippage", 0.0)).fillna(0.0)
    base = out["ticket_type"].astype(str).eq("umaren") & margin.ge(0.95)
    available = out["race_sim_probability_available"].astype(bool)
    quote = num(out, "quote_pay_proxy_per100", np.nan).fillna(num(out, "runtime_pay_per100", np.nan))
    sim_ev = sim_prob * quote / 100.0

    policies = {
        "mcs_s0304_existing": base,
        "sim_available": base & available,
        "sim_prob_ge_006": base & available & sim_prob.ge(0.06),
        "sim_prob_ge_008": base & available & sim_prob.ge(0.08),
        "sim_prob_ge_010": base & available & sim_prob.ge(0.10),
        "sim_prob_ge_012": base & available & sim_prob.ge(0.12),
        "sim_prob_over_existing": base & available & sim_prob.gt(existing_prob),
        "sim_ev_ge_135": base & available & sim_ev.ge(1.35),
        "sim_ev_ge_160": base & available & sim_ev.ge(1.60),
        "sim_prob_ge_008_expected_roi_ge135": base & available & sim_prob.ge(0.08) & expected_roi.ge(1.35),
    }
    rows = [metrics(out, mask, name) for name, mask in policies.items()]
    return pd.DataFrame(rows).sort_values(["top10_removed_roi", "roi"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply race-level Plackett-Luce scenario simulation to umaren pair probabilities.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/pair_probability_runtime_v1/pair_calibrated_runtime_tickets.csv")
    parser.add_argument("--runner-csv", default="outputs/analysis/risk_models_v1/investment_features_with_risk_models.csv")
    parser.add_argument("--universe-csv", default="outputs/analysis/dynamic_pair_ticket_allocation_quinella_model_v1/pair_candidate_universe.csv")
    parser.add_argument("--front5-predictions-csv", default="outputs/analysis/front5_position_model_v1/front5_oos_predictions.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/race_sim_umaren_probability_v1")
    parser.add_argument("--blend-weight", type=float, default=0.35)
    parser.add_argument("--min-train-rows", type=int, default=5000)
    parser.add_argument("--purge-days", type=int, default=7)
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    universe = pd.read_csv(project_path(args.universe_csv), dtype={"race_id": str}, low_memory=False)
    runners = pd.read_csv(project_path(args.runner_csv), dtype={"race_id": str}, low_memory=False)
    runners = add_front5_model(runners, project_path(args.front5_predictions_csv))
    runners = prepare_runner_scores(runners)

    pairs = prepare_pairs_for_scoring(universe, tickets)
    sim_scores = score_pairs_by_race(runners, pairs)

    universe_cal = universe.copy()
    universe_cal["race_sim_pair_key"] = pair_key(universe_cal)
    universe_cal = universe_cal.merge(sim_scores, on="race_sim_pair_key", how="left", suffixes=("", "_sim"))
    universe_cal["umaren_hit"] = pd.to_numeric(universe_cal["umaren_hit"], errors="coerce").astype(float)
    universe_cal, fold_metrics = calibrate_monthly_oos(universe_cal, args.min_train_rows, args.purge_days)
    fold_metrics.to_csv(out_dir / "race_sim_calibration_folds.csv", index=False, encoding="utf-8-sig")

    sim_for_merge = universe_cal[
        [
            "race_sim_pair_key",
            "race_sim_umaren_prob_raw",
            "race_sim_umaren_prob_cal",
            "race_sim_neutral_prob",
            "race_sim_front_prob",
            "race_sim_collapse_prob",
            "race_sim_neutral_share",
            "race_sim_front_share",
            "race_sim_collapse_share",
            "race_sim_calibration_train_rows",
            "race_sim_front5_model_available_rate",
        ]
    ].drop_duplicates("race_sim_pair_key", keep="last")

    sim_scores.to_csv(out_dir / "race_sim_pair_raw_scores.csv", index=False, encoding="utf-8-sig")
    universe_cal.to_csv(out_dir / "race_sim_pair_universe_calibrated.csv", index=False, encoding="utf-8-sig")

    tickets_out = apply_to_tickets(tickets, sim_for_merge, args.blend_weight)
    tickets_out.to_csv(out_dir / "race_sim_umaren_runtime_tickets.csv", index=False, encoding="utf-8-sig")

    comparison = evaluate_policies(tickets_out)
    comparison.to_csv(out_dir / "race_sim_umaren_policy_comparison.csv", index=False, encoding="utf-8-sig")

    summary = {
        "tickets_csv": args.tickets_csv,
        "runner_csv": args.runner_csv,
        "universe_csv": args.universe_csv,
        "front5_predictions_csv": args.front5_predictions_csv,
        "output_dir": str(out_dir),
        "pairs_scored": int(len(sim_scores)),
        "universe_rows": int(len(universe_cal)),
        "tickets_rows": int(len(tickets_out)),
        "calibrated_universe_rows": int(universe_cal["race_sim_umaren_prob_cal"].notna().sum()),
        "calibrated_ticket_rows": int(tickets_out["race_sim_probability_available"].sum()),
        "blend_weight": args.blend_weight,
        "best_by_top10_removed_roi": comparison.head(1).to_dict(orient="records")[0] if not comparison.empty else {},
        "comparison": comparison.to_dict(orient="records"),
        "notes": [
            "Race simulation uses all runners in a race, then blends neutral/front/collapse Plackett-Luce top-2 probabilities.",
            "Monthly calibration uses only earlier candidate-universe rows with a purge gap.",
            "This is a first joint-probability layer; it should be adopted only if ticket ROI robustness improves.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
