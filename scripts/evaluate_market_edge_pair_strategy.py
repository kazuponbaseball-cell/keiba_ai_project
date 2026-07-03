from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_fixed_budget_ticket_portfolio import _existing_raw_csv, _load_payoffs
from scripts.evaluate_ticket_strategies import _add_model_columns, _col, _num
from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


def _safe_prob_from_odds(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce").replace(0, np.nan)
    prob = 1.0 / odds
    # Remove overround approximately race-by-race later; this is the raw market signal.
    return prob.replace([np.inf, -np.inf], np.nan)


def _softmax_by_race(score: pd.Series, race_ids: pd.Series) -> pd.Series:
    out = pd.Series(index=score.index, dtype=float)
    for _, idx in score.groupby(race_ids).groups.items():
        s = score.loc[idx].astype(float)
        z = s - s.max()
        e = np.exp(z)
        denom = e.sum()
        out.loc[idx] = e / denom if denom > 0 else np.nan
    return out


def _add_market_edge(scored: pd.DataFrame, race_col: str) -> pd.DataFrame:
    out = scored.copy()
    out["ai_win_prob_proxy"] = _softmax_by_race(out["ai_score"], out[race_col])
    raw_market = _safe_prob_from_odds(out["odds_decimal"])
    market_sum = raw_market.groupby(out[race_col]).transform("sum")
    out["market_win_prob_norm"] = (raw_market / market_sum).replace([np.inf, -np.inf], np.nan)
    out["ai_market_prob_diff"] = out["ai_win_prob_proxy"] - out["market_win_prob_norm"]
    out["ai_market_prob_ratio"] = out["ai_win_prob_proxy"] / out["market_win_prob_norm"].replace(0, np.nan)
    out["market_edge_score"] = (
        out["ai_market_prob_diff"].fillna(0.0) * 100.0
        + np.log(out["ai_market_prob_ratio"].clip(lower=0.01, upper=100.0)).fillna(0.0)
    )
    out["no_value_favorite_flag"] = (
        out["ai_rank"].eq(1)
        & out["pop_rank"].le(2)
        & out["ai_score_gap_to_second"].lt(0.10)
        & out["ai_market_prob_diff"].lt(0.02)
    )
    out["steam_flag_any"] = False
    out["drift_flag_any"] = False
    for col in ["odds_steam_flag", "odds_drift_flag"]:
        if col in out.columns:
            if "steam" in col:
                out["steam_flag_any"] = out["steam_flag_any"] | pd.to_numeric(out[col], errors="coerce").fillna(0).astype(bool)
            if "drift" in col:
                out["drift_flag_any"] = out["drift_flag_any"] | pd.to_numeric(out[col], errors="coerce").fillna(0).astype(bool)
    if "odds_win_change_from_prev_pct" in out.columns:
        chg = pd.to_numeric(out["odds_win_change_from_prev_pct"], errors="coerce")
        out["late_odds_value_flag"] = chg.ge(5.0) & out["market_edge_score"].gt(0)
        out["late_odds_overbet_flag"] = chg.le(-10.0) & out["ai_market_prob_diff"].lt(0.01)
    else:
        out["late_odds_value_flag"] = False
        out["late_odds_overbet_flag"] = False
    return out


def _select_one(df: pd.DataFrame, mask: pd.Series, race_col: str, sort_cols: list[str], ascending: list[bool]) -> pd.DataFrame:
    return df[mask].sort_values([race_col] + sort_cols, ascending=[True] + ascending).groupby(race_col, as_index=False).head(1)


def _select_topn(df: pd.DataFrame, mask: pd.Series, race_col: str, n: int, sort_cols: list[str], ascending: list[bool]) -> pd.DataFrame:
    return df[mask].sort_values([race_col] + sort_cols, ascending=[True] + ascending).groupby(race_col, as_index=False).head(n)


def _horse_cols(df: pd.DataFrame, race_col: str, prefix: str) -> pd.DataFrame:
    cols = [race_col, "horse_name_for_ticket", "ai_rank", "pop_rank", "popularity_num", "odds_decimal", "rank_num", "target_top3", "market_edge_score", "ai_market_prob_diff"]
    return df[cols].rename(
        columns={
            "horse_name_for_ticket": f"{prefix}_horse",
            "ai_rank": f"{prefix}_ai_rank",
            "pop_rank": f"{prefix}_pop_rank",
            "popularity_num": f"{prefix}_popularity",
            "odds_decimal": f"{prefix}_odds",
            "rank_num": f"{prefix}_finish",
            "target_top3": f"{prefix}_top3",
            "market_edge_score": f"{prefix}_edge_score",
            "ai_market_prob_diff": f"{prefix}_prob_diff",
        }
    )


def _build_pair_candidates(scored: pd.DataFrame, race_col: str) -> pd.DataFrame:
    anchor = (
        scored["ai_rank"].eq(1)
        & scored["ai_score_gap_to_second"].ge(0.05)
        & ~scored["no_value_favorite_flag"]
    )
    value_partner = (
        scored["ai_rank"].le(6)
        & scored["pop_rank"].ge(5)
        & scored["market_edge_score"].gt(0)
    )
    stronger_value_partner = (
        scored["ai_rank"].le(5)
        & scored["pop_rank"].ge(7)
        & scored["market_edge_score"].gt(0.75)
    )
    late_value_partner = (
        scored["ai_rank"].le(6)
        & scored["pop_rank"].ge(5)
        & (scored["market_edge_score"].gt(0) | scored["late_odds_value_flag"])
        & ~scored["late_odds_overbet_flag"]
    )
    specs = [
        ("anchor_x_value_top2", value_partner, 2),
        ("anchor_x_strong_value_top1", stronger_value_partner, 1),
        ("anchor_x_late_value_top2", late_value_partner, 2),
    ]
    frames = []
    anchors = _horse_cols(_select_one(scored, anchor, race_col, ["ai_score_gap_to_second", "ai_score"], [False, False]), race_col, "a")
    for strategy, mask, topn in specs:
        partners = _horse_cols(_select_topn(scored, mask, race_col, topn, ["market_edge_score", "ai_rank"], [False, True]), race_col, "b")
        tickets = anchors.merge(partners, on=race_col, how="inner")
        tickets = tickets[tickets["a_horse"] != tickets["b_horse"]].copy()
        if tickets.empty:
            continue
        tickets["strategy"] = strategy
        tickets["wide_hit"] = tickets["a_top3"].eq(1) & tickets["b_top3"].eq(1)
        tickets["umaren_hit"] = tickets["a_finish"].le(2) & tickets["b_finish"].le(2)
        frames.append(tickets)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _load_wide_payoffs(path_text: str, race_col: str) -> pd.DataFrame:
    path = project_path(path_text)
    if not path.exists():
        return pd.DataFrame(columns=[race_col, "horse_a", "horse_b", "wide_pay", "wide_popularity"])
    wide = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
    if race_col != "race_id" and "race_id" in wide.columns:
        wide = wide.rename(columns={"race_id": race_col})
    if race_col in wide.columns:
        wide[race_col] = wide[race_col].astype(str)
    for col in ["horse_a", "horse_b", "wide_pay", "wide_popularity"]:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")
    return wide


def _add_horse_numbers(tickets: pd.DataFrame, scored: pd.DataFrame, race_col: str) -> pd.DataFrame:
    if tickets.empty:
        return tickets
    horse_no_col = _col(scored, ["馬番", "鬥ｬ逡ｪ", "umaban", "horse_number"])
    if not horse_no_col:
        return tickets
    lookup = scored[[race_col, "horse_name_for_ticket", horse_no_col]].drop_duplicates().copy()
    lookup[horse_no_col] = pd.to_numeric(lookup[horse_no_col], errors="coerce")
    out = tickets.merge(
        lookup.rename(columns={"horse_name_for_ticket": "a_horse", horse_no_col: "a_horse_no"}),
        on=[race_col, "a_horse"],
        how="left",
    )
    out = out.merge(
        lookup.rename(columns={"horse_name_for_ticket": "b_horse", horse_no_col: "b_horse_no"}),
        on=[race_col, "b_horse"],
        how="left",
    )
    return out


def _attach_wide_payoffs(tickets: pd.DataFrame, wide_payoffs: pd.DataFrame, race_col: str) -> pd.DataFrame:
    out = tickets.copy()
    if race_col in out.columns:
        out[race_col] = out[race_col].astype(str)
    if "wide_pay" not in out.columns:
        out["wide_pay"] = np.nan
    if "wide_popularity" not in out.columns:
        out["wide_popularity"] = np.nan
    if out.empty or wide_payoffs.empty or "a_horse_no" not in out.columns or "b_horse_no" not in out.columns:
        return out
    out = out.drop(columns=["wide_pay", "wide_popularity"], errors="ignore")
    out["horse_a"] = np.minimum(pd.to_numeric(out["a_horse_no"], errors="coerce"), pd.to_numeric(out["b_horse_no"], errors="coerce"))
    out["horse_b"] = np.maximum(pd.to_numeric(out["a_horse_no"], errors="coerce"), pd.to_numeric(out["b_horse_no"], errors="coerce"))
    return out.merge(wide_payoffs, on=[race_col, "horse_a", "horse_b"], how="left")


def _metrics(tickets: pd.DataFrame, payoffs: pd.DataFrame, race_col: str) -> pd.DataFrame:
    rows = []
    if tickets.empty:
        return pd.DataFrame()
    payoff_cols = payoffs[[race_col, "umaren_pay"]].copy()
    payoff_cols[race_col] = payoff_cols[race_col].astype(str)
    tickets = tickets.copy()
    tickets[race_col] = tickets[race_col].astype(str)
    merged = tickets.merge(payoff_cols, on=race_col, how="left")
    for strategy, g in merged.groupby("strategy"):
        stake = len(g) * 100.0
        umaren_return = g["umaren_pay"].fillna(0.0).where(g["umaren_hit"], 0.0).sum()
        wide_coverage = float(g["wide_pay"].notna().mean()) if "wide_pay" in g.columns else 0.0
        wide_return = g["wide_pay"].fillna(0.0).where(g["wide_hit"], 0.0).sum() if wide_coverage > 0 else np.nan
        wide_hit = float(g["wide_hit"].mean())
        rows.append(
            {
                "strategy": strategy,
                "tickets": int(len(g)),
                "races": int(g[race_col].nunique()),
                "wide_hit_rate": wide_hit,
                "wide_break_even_avg_pay": float(100.0 / wide_hit) if wide_hit > 0 else None,
                "wide_payoff_coverage": wide_coverage,
                "wide_roi": float(wide_return / stake) if stake and not pd.isna(wide_return) else None,
                "wide_profit_flat100": float(wide_return - stake) if not pd.isna(wide_return) else None,
                "wide_avg_pay_hit": float(g.loc[g["wide_hit"], "wide_pay"].mean()) if "wide_pay" in g.columns and g["wide_hit"].any() else None,
                "umaren_hit_rate": float(g["umaren_hit"].mean()),
                "umaren_roi": float(umaren_return / stake) if stake else 0.0,
                "umaren_profit_flat100": float(umaren_return - stake),
                "avg_anchor_pop": float(g["a_popularity"].mean()),
                "avg_partner_pop": float(g["b_popularity"].mean()),
                "avg_partner_odds": float(g["b_odds"].mean()),
                "avg_partner_edge": float(g["b_edge_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["wide_roi", "umaren_roi", "wide_hit_rate"],
        ascending=[False, False, False],
        na_position="last",
    )


def _no_value_favorite_report(scored: pd.DataFrame, race_col: str) -> pd.DataFrame:
    top1 = scored[scored["ai_rank"].eq(1)].copy()
    top1["win"] = top1["rank_num"].eq(1)
    top1["top3"] = top1["rank_num"].le(3)
    win_pay_col = _col(top1, ["単勝配当", "蜊伜享驟榊ｽ・"])
    place_pay_col = _col(top1, ["複勝配当", "隍・享驟榊ｽ・"])
    top1["win_pay"] = _num(top1[win_pay_col]).where(top1["win"], 0.0)
    top1["place_pay"] = _num(top1[place_pay_col]).where(top1["top3"], 0.0)
    rows = []
    for name, part in [("top1_all", top1), ("top1_no_value_favorite", top1[top1["no_value_favorite_flag"]]), ("top1_keep", top1[~top1["no_value_favorite_flag"]])]:
        if part.empty:
            rows.append({"segment": name, "races": 0})
            continue
        rows.append(
            {
                "segment": name,
                "races": int(part[race_col].nunique()),
                "win_rate": float(part["win"].mean()),
                "top3_rate": float(part["top3"].mean()),
                "win_roi": float(part["win_pay"].sum() / (len(part) * 100.0)),
                "place_roi": float(part["place_pay"].sum() / (len(part) * 100.0)),
                "avg_pop": float(part["pop_rank"].mean()),
                "avg_odds": float(part["odds_decimal"].mean()),
                "avg_prob_diff": float(part["ai_market_prob_diff"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _wide_note(summary: pd.DataFrame) -> str:
    if summary.empty or "wide_payoff_coverage" not in summary.columns:
        return "Wide payoff CSV was not available; wide is evaluated by hit rate and break-even average payoff."
    coverage = float(summary["wide_payoff_coverage"].max())
    if coverage <= 0:
        return "Wide payoff CSV was not available or did not match ticket race/horse keys; wide ROI is not reliable yet."
    return "Actual wide payoff ROI is included from data/processed/target/wide_payoffs.csv."


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate market-edge pair strategy without trifecta as a default.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default="models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv",
    )
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/market_edge_pair_strategy")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    encoding = config["data"].get("encoding", "cp932")
    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = _add_model_columns(df, project_path(args.model), race_col)
    scored = _add_market_edge(scored, race_col)
    payoffs, _ = _load_payoffs(_existing_raw_csv(args.raw_csv), race_col, encoding)

    tickets = _build_pair_candidates(scored, race_col)
    tickets = _add_horse_numbers(tickets, scored, race_col)
    tickets = _attach_wide_payoffs(tickets, _load_wide_payoffs(args.wide_payoff_csv, race_col), race_col)
    summary = _metrics(tickets, payoffs, race_col)
    favorite_report = _no_value_favorite_report(scored, race_col)

    out_dir = ensure_dir(project_path(args.output_dir))
    scored_cols = [
        race_col,
        "horse_name_for_ticket",
        "ai_rank",
        "pop_rank",
        "odds_decimal",
        "ai_win_prob_proxy",
        "market_win_prob_norm",
        "ai_market_prob_diff",
        "market_edge_score",
        "no_value_favorite_flag",
        "late_odds_value_flag",
        "late_odds_overbet_flag",
    ]
    scored[scored_cols].to_csv(out_dir / "runner_market_edge_scores.csv", index=False, encoding="utf-8-sig")
    tickets.to_csv(out_dir / "market_edge_pair_tickets.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "market_edge_pair_summary.csv", index=False, encoding="utf-8-sig")
    favorite_report.to_csv(out_dir / "no_value_favorite_report.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "wide_payoff_csv": str(project_path(args.wide_payoff_csv)),
        "note": "Trifecta is intentionally excluded from default strategies. " + _wide_note(summary),
        "summary": summary.to_dict(orient="records"),
        "no_value_favorite_report": favorite_report.to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
