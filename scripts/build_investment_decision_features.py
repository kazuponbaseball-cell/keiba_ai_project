from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series | float | int) -> pd.Series:
    if isinstance(series, (float, int)):
        return pd.Series(series)
    if series.dtype == object:
        cleaned = series.astype(str).str.replace(r"[(),]", "", regex=True).replace({"nan": np.nan, "": np.nan})
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _z_by_race(values: pd.Series, race_ids: pd.Series) -> pd.Series:
    v = _num(values)
    mean = v.groupby(race_ids).transform("mean")
    std = v.groupby(race_ids).transform("std").replace(0, np.nan)
    return ((v - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _norm01(series: pd.Series) -> pd.Series:
    s = _num(series).replace([np.inf, -np.inf], np.nan)
    lo = s.quantile(0.05)
    hi = s.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(0.5, index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5)


def _softmax_by_race(score: pd.Series, race_ids: pd.Series, temperature: float = 1.0) -> pd.Series:
    scaled = _z_by_race(score, race_ids) / max(temperature, 1e-6)
    out = pd.Series(np.nan, index=score.index, dtype=float)
    for _, idx in scaled.groupby(race_ids).groups.items():
        s = scaled.loc[idx].clip(-20, 20)
        e = np.exp(s - s.max())
        denom = e.sum()
        out.loc[idx] = e / denom if denom > 0 else np.nan
    return out.fillna(0.0)


def _sigmoid(x: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))


def _load_feature_lookup(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    race_col = _col(df, ["レースID(新/馬番無)", "race_id"])
    horse_col = _col(df, ["馬番", "horse_no", "horse_number"])
    if race_col is None or horse_col is None:
        raise ValueError("feature csv must have race id and horse number columns")
    keep = [
        race_col,
        horse_col,
        "4角.1",
        "4角",
        "出走頭数",
        "prev_corner4_position_rate",
        "front_running_tendency",
        "horse_front_run_rate_past5",
        "horse_stalker_rate_past5",
        "horse_closer_rate_past5",
        "front_pressure_rank_score",
        "solo_lead_potential",
        "race_slow_pace_risk",
        "race_pace_collapse_risk",
        "pace_fit_score",
        "front_advantage_score",
        "positioning_advantage_score",
        "draw_pace_fit_score",
        "単勝配当",
        "複勝配当",
    ]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy().rename(columns={race_col: "race_id", horse_col: "horse_no"})
    out["race_id"] = out["race_id"].astype(str)
    out["horse_no"] = _num(out["horse_no"])
    return out.drop_duplicates(["race_id", "horse_no"])


def _prepare(input_csv: Path, feature_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df["horse_no"] = _num(df["horse_no"])
    feature = _load_feature_lookup(feature_csv)
    suffix_cols = [c for c in feature.columns if c not in {"race_id", "horse_no"} and c in df.columns]
    feature = feature.rename(columns={c: f"{c}_feature" for c in suffix_cols})
    out = df.merge(feature, on=["race_id", "horse_no"], how="left")
    return out


def _first_existing(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return _num(df[name])
    return pd.Series(default, index=df.index, dtype=float)


def _clean_odds(series: pd.Series | float | int | None, index: pd.Index) -> pd.Series:
    if series is None:
        values = pd.Series(np.nan, index=index, dtype=float)
    elif isinstance(series, pd.Series):
        values = _num(series)
    else:
        values = _num(pd.Series(series, index=index))
    return values.where(values.ge(1.0) & values.lt(999.0))


def _first_valid_odds(df: pd.DataFrame, names: list[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(default, index=df.index, dtype=float)
    for name in names:
        if name in df.columns:
            out = out.fillna(_clean_odds(df[name], df.index))
    return out


def add_investment_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    race = out["race_id"]
    odds = _first_valid_odds(out, ["odds_latest_win", "odds_num"], np.nan)
    out["market_odds_live_or_final"] = odds

    raw_market = 1.0 / odds
    market_sum = raw_market.groupby(race).transform("sum").replace(0, np.nan)
    out["market_win_prob_norm"] = (raw_market / market_sum).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ai_win_prob_proxy"] = _softmax_by_race(out["ai_score"], race, temperature=1.15)
    out["ai_market_prob_diff"] = out["ai_win_prob_proxy"] - out["market_win_prob_norm"]
    out["ai_market_prob_ratio"] = (
        out["ai_win_prob_proxy"] / out["market_win_prob_norm"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out["win_ev_proxy"] = (out["ai_win_prob_proxy"] * odds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["market_overlay_score"] = (
        0.65 * _norm01(out["ai_market_prob_diff"])
        + 0.35 * _norm01(np.log(out["ai_market_prob_ratio"].clip(0.05, 20.0)))
    )

    latest = _first_valid_odds(out, ["odds_latest_win", "odds_num"], np.nan)
    prev = _first_valid_odds(out, ["odds_prev_win"], np.nan)
    first = _first_valid_odds(out, ["odds_first_win"], np.nan)
    out["late_odds_drop_rate"] = (prev / latest - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["session_odds_drop_rate"] = (first / latest - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["late_odds_drift_rate"] = (latest / prev - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["late_steam_flag"] = (
        (_first_existing(out, ["odds_steam_flag"], 0.0).fillna(0.0).gt(0))
        | out["late_odds_drop_rate"].ge(0.12)
    ).astype(float)
    out["late_drift_flag"] = (
        (_first_existing(out, ["odds_drift_flag"], 0.0).fillna(0.0).gt(0))
        | out["late_odds_drift_rate"].ge(0.20)
    ).astype(float)
    out["late_value_survives_score"] = (
        out["market_overlay_score"]
        + 0.15 * out["late_drift_flag"]
        - 0.20 * ((out["late_steam_flag"].eq(1)) & (out["win_ev_proxy"].lt(1.05))).astype(float)
    ).clip(0.0, 1.0)

    field = _first_existing(out, ["field_size_num", "出走頭数_feature", "出走頭数"], 12.0).replace(0, np.nan)
    frame = _first_existing(out, ["frame_num", "枠番"], np.nan)
    frame_inner = (1.0 - (frame - 1.0) / 7.0).clip(0.0, 1.0).fillna(0.5)
    front = _first_existing(out, ["horse_front_run_rate_past5", "horse_front_run_rate_past5_feature", "front_running_tendency_y", "front_running_tendency_feature", "front_running_tendency_x"], 0.0).fillna(0.0)
    stalker = _first_existing(out, ["horse_stalker_rate_past5", "horse_stalker_rate_past5_feature"], 0.0).fillna(0.0)
    prev_c4 = _first_existing(out, ["prev_corner4_position_rate", "prev_corner4_position_rate_feature"], 0.5).fillna(0.5)
    pressure = _first_existing(out, ["race_early_pressure_score", "race_front_runner_ratio"], 0.0).fillna(0.0)
    slow_risk = _first_existing(out, ["race_slow_pace_risk", "race_slow_pace_risk_feature"], 0.0).fillna(0.0)
    pace_fit = _first_existing(out, ["pace_fit_score", "pace_fit_score_feature"], 0.0).fillna(0.0)
    draw_fit = _first_existing(out, ["draw_pace_fit_score", "draw_pace_fit_score_feature"], 0.0).fillna(0.0)
    front_rank = _first_existing(out, ["front_pressure_rank_score", "front_pressure_rank_score_feature"], 0.0).fillna(0.0)
    expected_fast = out.get("expected_pace", "").astype(str).eq("fast").astype(float)

    front_logit = (
        -1.20
        + 2.40 * front
        + 0.85 * stalker
        + 1.20 * (1.0 - prev_c4.clip(0.0, 1.0))
        + 0.45 * frame_inner
        + 0.55 * front_rank
        + 0.35 * slow_risk
        + 0.20 * _norm01(pace_fit)
        + 0.15 * _norm01(draw_fit)
        - 0.65 * pressure
        - 0.35 * expected_fast
        - 0.15 * (field.ge(16)).astype(float)
    )
    out["projected_front5_prob"] = _sigmoid(front_logit).values

    actual_c4 = _first_existing(out, ["4角.1_feature", "4角_feature", "4角.1", "4角"], np.nan)
    out["actual_front5"] = actual_c4.le(5).where(actual_c4.notna(), np.nan)

    confidence = _norm01(_first_existing(out, ["ai_score_gap_to_second"], 0.0))
    top_market = out["market_win_prob_norm"].groupby(race).rank(ascending=False, method="first").le(2)
    bad_conditions = (
        out.get("馬場状態", "").astype(str).isin(["重", "不"]).astype(float)
        + out.get("expected_pace", "").astype(str).eq("fast").astype(float)
        + field.ge(16).astype(float)
        + out.get("class_group", "").astype(str).eq("open").astype(float)
    )
    out["danger_favorite_score"] = (
        top_market.astype(float)
        * (
            0.35 * (1.0 - out["market_overlay_score"])
            + 0.25 * (1.0 - confidence)
            + 0.20 * (bad_conditions / 4.0)
            + 0.15 * (1.0 - _norm01(pace_fit))
            + 0.05 * out["late_steam_flag"]
        )
    ).clip(0.0, 1.0)
    out["danger_favorite_flag"] = (
        (out["danger_favorite_score"].ge(0.45))
        & (out["market_win_prob_norm"].ge(out["market_win_prob_norm"].groupby(race).transform("quantile", 0.75)))
    ).astype(float)

    ai_rank = _num(out["ai_rank_num"]).fillna(_num(out["ai_rank"]))
    out["win_suitability_score"] = (
        0.40 * out["late_value_survives_score"]
        + 0.30 * _norm01(out["win_ev_proxy"])
        + 0.20 * confidence
        + 0.10 * _norm01(out["projected_front5_prob"])
        - 0.25 * out["danger_favorite_score"]
    ).clip(0.0, 1.0)
    out["place_suitability_score"] = (
        0.35 * (1.0 - (ai_rank - 1.0).clip(0, 8) / 8.0)
        + 0.25 * confidence
        + 0.20 * _norm01(pace_fit)
        + 0.20 * _norm01(out["projected_front5_prob"])
        - 0.20 * out["danger_favorite_score"]
    ).clip(0.0, 1.0)
    out["wide_axis_score"] = (
        0.50 * out["place_suitability_score"]
        + 0.25 * confidence
        + 0.25 * (1.0 - out["danger_favorite_score"])
    ).clip(0.0, 1.0)
    out["wide_partner_score"] = (
        0.42 * out["late_value_survives_score"]
        + 0.25 * _norm01(out["projected_front5_prob"])
        + 0.18 * (ai_rank.le(6)).astype(float)
        + 0.15 * odds.ge(8.0).astype(float)
        - 0.15 * out["danger_favorite_score"]
    ).clip(0.0, 1.0)
    out["skip_risk_score"] = (
        0.45 * out["danger_favorite_score"]
        + 0.25 * bad_conditions.clip(0, 4) / 4.0
        + 0.20 * (1.0 - confidence)
        + 0.10 * (out["late_steam_flag"].eq(1) & out["win_ev_proxy"].lt(1.0)).astype(float)
    ).clip(0.0, 1.0)
    return out


def _single_metrics(df: pd.DataFrame, mask: pd.Series, label: str, ret_col: str) -> dict:
    g = df[mask].copy()
    if g.empty:
        return {"policy": label, "bets": 0}
    stake = len(g) * 100.0
    ret = _num(g[ret_col]).sum()
    return {
        "policy": label,
        "bets": int(len(g)),
        "races": int(g["race_id"].nunique()),
        "hit_rate": float(g["is_win"].mean() if ret_col == "win_return" else g["is_place"].mean()),
        "roi": float(ret / stake),
        "profit_yen_flat100": float(ret - stake),
        "avg_odds": float(_num(g["market_odds_live_or_final"]).mean()),
        "avg_ai_rank": float(_num(g["ai_rank_num"]).mean()),
        "avg_market_prob": float(g["market_win_prob_norm"].mean()),
        "avg_ev": float(g["win_ev_proxy"].mean()),
        "avg_front5_prob": float(g["projected_front5_prob"].mean()),
        "avg_danger_fav": float(g["danger_favorite_score"].mean()),
    }


def _evaluate_singles(df: pd.DataFrame) -> pd.DataFrame:
    ai_rank = _num(df["ai_rank_num"])
    rows = [
        _single_metrics(df, ai_rank.eq(1), "baseline_ai1_win", "win_return"),
        _single_metrics(df, ai_rank.eq(1), "baseline_ai1_place", "place_return"),
        _single_metrics(df, ai_rank.le(3) & df["win_ev_proxy"].ge(1.15) & df["danger_favorite_score"].lt(0.45), "win_ev115_rank3_safe", "win_return"),
        _single_metrics(df, ai_rank.le(5) & df["win_suitability_score"].ge(0.72), "win_suitability_hi", "win_return"),
        _single_metrics(df, ai_rank.eq(1) & df["place_suitability_score"].ge(0.70) & df["skip_risk_score"].lt(0.50), "place_axis_suitable", "place_return"),
        _single_metrics(df, ai_rank.eq(1) & df["danger_favorite_flag"].eq(0), "place_ai1_not_danger_favorite", "place_return"),
    ]
    return pd.DataFrame(rows)


def _load_wide_payoffs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    wide = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
    for c in ["horse_a", "horse_b", "wide_pay"]:
        wide[c] = _num(wide[c])
    return wide


def _evaluate_wide(df: pd.DataFrame, wide_payoffs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if wide_payoffs.empty:
        return pd.DataFrame(), pd.DataFrame()
    anchors = df[
        (_num(df["ai_rank_num"]).eq(1))
        & df["wide_axis_score"].ge(0.70)
        & df["danger_favorite_score"].lt(0.50)
    ].copy()
    partners = df[
        (_num(df["ai_rank_num"]).between(2, 6))
        & df["wide_partner_score"].ge(0.62)
        & df["market_odds_live_or_final"].ge(6.0)
    ].copy()
    if anchors.empty or partners.empty:
        return pd.DataFrame(), pd.DataFrame()
    partner_top = (
        partners.sort_values(["race_id", "wide_partner_score", "market_overlay_score"], ascending=[True, False, False])
        .groupby("race_id", as_index=False)
        .head(2)
    )
    tickets = anchors[
        ["race_id", "horse_no", "horse_name", "ai_rank_num", "pop_rank_num", "market_odds_live_or_final", "wide_axis_score"]
    ].rename(
        columns={
            "horse_no": "anchor_no",
            "horse_name": "anchor_name",
            "ai_rank_num": "anchor_ai_rank",
            "pop_rank_num": "anchor_pop",
            "market_odds_live_or_final": "anchor_odds",
        }
    ).merge(
        partner_top[
            ["race_id", "horse_no", "horse_name", "ai_rank_num", "pop_rank_num", "market_odds_live_or_final", "wide_partner_score", "projected_front5_prob"]
        ].rename(
            columns={
                "horse_no": "partner_no",
                "horse_name": "partner_name",
                "ai_rank_num": "partner_ai_rank",
                "pop_rank_num": "partner_pop",
                "market_odds_live_or_final": "partner_odds",
            }
        ),
        on="race_id",
        how="inner",
    )
    tickets = tickets[tickets["anchor_no"] != tickets["partner_no"]].copy()
    tickets["horse_a"] = np.minimum(_num(tickets["anchor_no"]), _num(tickets["partner_no"]))
    tickets["horse_b"] = np.maximum(_num(tickets["anchor_no"]), _num(tickets["partner_no"]))
    tickets = tickets.merge(wide_payoffs, on=["race_id", "horse_a", "horse_b"], how="left")
    tickets["wide_hit"] = tickets["wide_pay"].notna()
    stake = len(tickets) * 100.0
    ret = _num(tickets["wide_pay"]).fillna(0.0).sum()
    summary = pd.DataFrame(
        [
            {
                "policy": "wide_axis_x_investment_partner_top2",
                "tickets": int(len(tickets)),
                "races": int(tickets["race_id"].nunique()),
                "hit_rate": float(tickets["wide_hit"].mean()) if len(tickets) else 0.0,
                "roi": float(ret / stake) if stake else 0.0,
                "profit_yen_flat100": float(ret - stake),
                "avg_anchor_pop": float(_num(tickets["anchor_pop"]).mean()),
                "avg_partner_pop": float(_num(tickets["partner_pop"]).mean()),
                "avg_partner_odds": float(_num(tickets["partner_odds"]).mean()),
                "avg_partner_front5_prob": float(_num(tickets["projected_front5_prob"]).mean()),
                "avg_pay_hit": float(_num(tickets.loc[tickets["wide_hit"], "wide_pay"]).mean()) if tickets["wide_hit"].any() else None,
            }
        ]
    )
    return summary, tickets


def _single_yearly(df: pd.DataFrame) -> pd.DataFrame:
    year = df["race_id"].astype(str).str[:4]
    ai_rank = _num(df["ai_rank_num"])
    specs = [
        ("baseline_ai1_win", ai_rank.eq(1), "win_return", "is_win"),
        ("win_ev115_rank3_safe", ai_rank.le(3) & df["win_ev_proxy"].ge(1.15) & df["danger_favorite_score"].lt(0.45), "win_return", "is_win"),
        ("win_suitability_hi", ai_rank.le(5) & df["win_suitability_score"].ge(0.72), "win_return", "is_win"),
        ("place_axis_suitable", ai_rank.eq(1) & df["place_suitability_score"].ge(0.70) & df["skip_risk_score"].lt(0.50), "place_return", "is_place"),
        ("place_ai1_not_danger_favorite", ai_rank.eq(1) & df["danger_favorite_flag"].eq(0), "place_return", "is_place"),
    ]
    rows = []
    for label, mask, ret_col, hit_col in specs:
        selected = df[mask].copy()
        selected["_year"] = year.loc[selected.index]
        for y, g in selected.groupby("_year"):
            if g.empty:
                continue
            stake = len(g) * 100.0
            ret = _num(g[ret_col]).fillna(0.0).sum()
            rows.append(
                {
                    "policy": label,
                    "year": int(y),
                    "bets": int(len(g)),
                    "races": int(g["race_id"].nunique()),
                    "hit_rate": float(g[hit_col].mean()),
                    "roi": float(ret / stake) if stake else 0.0,
                    "profit_yen_flat100": float(ret - stake),
                }
            )
    return pd.DataFrame(rows).sort_values(["policy", "year"])


def _wide_yearly(tickets: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame()
    out = tickets.copy()
    out["_year"] = out["race_id"].astype(str).str[:4]
    rows = []
    for y, g in out.groupby("_year"):
        stake = len(g) * 100.0
        ret = _num(g["wide_pay"]).fillna(0.0).sum()
        rows.append(
            {
                "policy": "wide_axis_x_investment_partner_top2",
                "year": int(y),
                "tickets": int(len(g)),
                "races": int(g["race_id"].nunique()),
                "hit_rate": float(g["wide_hit"].mean()),
                "roi": float(ret / stake) if stake else 0.0,
                "profit_yen_flat100": float(ret - stake),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def _front_model_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["actual_front5"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["front5_bin"] = pd.qcut(valid["projected_front5_prob"], q=5, labels=False, duplicates="drop")
    rows = []
    for b, g in valid.groupby("front5_bin"):
        rows.append(
            {
                "front5_bin": int(b),
                "rows": int(len(g)),
                "avg_projected_front5_prob": float(g["projected_front5_prob"].mean()),
                "actual_front5_rate": float(g["actual_front5"].astype(float).mean()),
                "win_rate": float(g["is_win"].mean()),
                "place_rate": float(g["is_place"].mean()),
                "avg_pop": float(_num(g["pop_rank_num"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expected-value, odds, front-position, dangerous-favorite and ticket-suitability features.")
    parser.add_argument("--prediction-csv", default="outputs/analysis/roi_stagnation_drivers_v1/prediction_detail_enriched.csv")
    parser.add_argument("--feature-csv", default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/investment_decision_features_v1")
    args = parser.parse_args()

    df = _prepare(project_path(args.prediction_csv), project_path(args.feature_csv))
    scored = add_investment_features(df)
    out_dir = ensure_dir(project_path(args.output_dir))
    wide_payoffs = _load_wide_payoffs(project_path(args.wide_payoff_csv))

    single_summary = _evaluate_singles(scored)
    wide_summary, wide_tickets = _evaluate_wide(scored, wide_payoffs)
    single_yearly = _single_yearly(scored)
    wide_yearly = _wide_yearly(wide_tickets)
    front_diag = _front_model_diagnostics(scored)

    scored.to_csv(out_dir / "investment_features_scored.csv", index=False, encoding="utf-8-sig")
    single_summary.to_csv(out_dir / "single_bet_policy_summary.csv", index=False, encoding="utf-8-sig")
    single_yearly.to_csv(out_dir / "single_bet_policy_yearly.csv", index=False, encoding="utf-8-sig")
    wide_summary.to_csv(out_dir / "wide_policy_summary.csv", index=False, encoding="utf-8-sig")
    wide_yearly.to_csv(out_dir / "wide_policy_yearly.csv", index=False, encoding="utf-8-sig")
    wide_tickets.to_csv(out_dir / "wide_policy_tickets.csv", index=False, encoding="utf-8-sig")
    front_diag.to_csv(out_dir / "projected_front5_diagnostics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "feature_groups_added": [
            "expected_value_and_market_probability",
            "late_odds_movement_if_available",
            "projected_front5_probability",
            "dangerous_favorite_score",
            "ticket_suitability_scores",
        ],
        "single_bet_summary": single_summary.to_dict(orient="records"),
        "single_bet_yearly": single_yearly.to_dict(orient="records"),
        "wide_summary": wide_summary.to_dict(orient="records"),
        "wide_yearly": wide_yearly.to_dict(orient="records"),
        "projected_front5_diagnostics": front_diag.to_dict(orient="records"),
        "limitations": [
            "Historical full-card late odds timeline is sparse; live columns are used only when present.",
            "AI probability is a proxy from within-race score softmax and should be calibrated with walk-forward probability calibration before production staking.",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
