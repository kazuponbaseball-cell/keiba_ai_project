from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_TRAIN_FEATURES = (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/"
    "train_features_with_same_day_bias_v3_retro.csv"
)
DEFAULT_TEST_FEATURES = (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/"
    "test_features_with_same_day_bias_v3_retro.csv"
)
DEFAULT_MODEL = "models/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/baseline_ranker.pkl"
DEFAULT_PAIR_CANDIDATES = "outputs/analysis/dynamic_pair_ticket_allocation_rebuilt_20260623/pair_candidate_universe.csv"
DEFAULT_DYNAMIC_TICKETS = "outputs/analysis/dynamic_pair_ticket_allocation_rebuilt_20260623/walkforward_selected_tickets.csv"
DEFAULT_PURGED_TICKETS = "outputs/analysis/purged_walkforward_mcs_pbo_rebuilt_20260623/purged_walkforward_selected_tickets.csv"
DEFAULT_OUT = "outputs/analysis/local_yoshiba_bloodline_overlay_rebuilt_20260623"

LOCAL_TIGHT = {"\u672d\u5e4c", "\u51fd\u9928", "\u798f\u5cf6", "\u5c0f\u5009"}
HOKKAIDO = {"\u672d\u5e4c", "\u51fd\u9928"}
TURF = "\u829d"
DIRT = "\u30c0"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    values = frame[col]
    if values.dtype == object or str(values.dtype).startswith("string"):
        values = values.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


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


def top_removed_roi(ret: pd.Series, stake: pd.Series, top_n: int) -> float:
    if len(ret) <= top_n:
        return 0.0
    drop_idx = ret.sort_values(ascending=False).index[:top_n]
    ret2 = ret.drop(index=drop_idx)
    stake2 = stake.drop(index=drop_idx)
    return float(ret2.sum() / stake2.sum()) if stake2.sum() > 0 else 0.0


def max_drawdown_by_race(frame: pd.DataFrame, stake_col: str, return_col: str) -> float:
    if frame.empty:
        return 0.0
    tmp = frame.copy()
    tmp["_stake"] = num(tmp, stake_col)
    tmp["_return"] = num(tmp, return_col)
    tmp["_profit"] = tmp["_return"] - tmp["_stake"]
    sort_cols = [c for c in ["year", "race_id"] if c in tmp.columns]
    if sort_cols:
        tmp = tmp.sort_values(sort_cols, kind="mergesort")
    race_profit = tmp.groupby("race_id", sort=False)["_profit"].sum()
    equity = race_profit.cumsum()
    dd = equity - equity.cummax()
    return float(dd.min()) if len(dd) else 0.0


def metrics(frame: pd.DataFrame, stake_col: str, return_col: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
            "top5_removed_roi": 0.0,
            "top10_removed_roi": 0.0,
        }
    stake = num(frame, stake_col)
    ret = num(frame, return_col)
    hit = ret.gt(0)
    race_hit = frame.assign(_hit=hit).groupby("race_id")["_hit"].max()
    return {
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0,
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(ret.sum() - stake.sum()),
        "roi": float(ret.sum() / stake.sum()) if stake.sum() > 0 else 0.0,
        "ticket_hit_rate": float(hit.mean()) if len(hit) else 0.0,
        "race_hit_rate": float(race_hit.mean()) if len(race_hit) else 0.0,
        "max_drawdown_yen": max_drawdown_by_race(frame, stake_col, return_col),
        "top5_removed_roi": top_removed_roi(ret, stake, 5),
        "top10_removed_roi": top_removed_roi(ret, stake, 10),
    }


def load_runner_profile(feature_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(feature_csv, encoding="utf-8-sig", low_memory=False)
    race_col = frame.columns[34]
    horse_no_col = frame.columns[15]
    horse_name_col = frame.columns[7]
    venue_col = frame.columns[2]
    surface_col = frame.columns[18]
    out = pd.DataFrame(index=frame.index)
    out["race_id"] = frame[race_col].astype(str)
    out["horse_no"] = num(frame, horse_no_col).astype("Int64").astype(str)
    out["horse_name"] = frame[horse_name_col].astype(str)
    out["venue"] = frame[venue_col].astype(str)
    out["surface"] = frame[surface_col].astype(str)
    out["local_tight"] = out["venue"].isin(LOCAL_TIGHT)
    out["hokkaido"] = out["venue"].isin(HOKKAIDO)
    out["turf"] = out["surface"].str.contains(TURF, regex=False, na=False)
    out["dirt"] = out["surface"].str.contains(DIRT, regex=False, na=False)
    out["hokkaido_turf"] = out["hokkaido"] & out["turf"]
    out["local_turf"] = out["local_tight"] & out["turf"]
    out["local_dirt"] = out["local_tight"] & out["dirt"]
    out["venue_lift_combo"] = num(frame, "sire_venue_lift") + num(frame, "bms_venue_lift")
    out["sire_venue_lift"] = num(frame, "sire_venue_lift")
    out["bms_venue_lift"] = num(frame, "bms_venue_lift")
    out["bloodline_course_fit_score"] = num(frame, "bloodline_course_fit_score")
    out["bloodline_reliability_score"] = num(frame, "bloodline_reliability_score")
    out["horse_front_run_rate_past5"] = num(frame, "horse_front_run_rate_past5")
    out["front_running_tendency"] = num(frame, "front_running_tendency")
    out["front_advantage_score"] = num(frame, "front_advantage_score")
    return out.drop_duplicates(["race_id", "horse_no"], keep="last")


def thresholds_from_train(train_csv: Path) -> dict[str, float]:
    train = load_runner_profile(train_csv)
    return {
        "venue_lift_hi": float(train["venue_lift_combo"].quantile(0.75)),
        "venue_lift_top": float(train["venue_lift_combo"].quantile(0.90)),
        "course_fit_hi": float(train["bloodline_course_fit_score"].quantile(0.75)),
        "front_hi": float(train["horse_front_run_rate_past5"].quantile(0.65)),
    }


def add_side_profile(frame: pd.DataFrame, profile: pd.DataFrame, side: str, no_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = out["race_id"].astype(str)
    out[no_col] = num(out, no_col).astype("Int64").astype(str)
    side_profile = profile.add_prefix(f"{side}_").rename(
        columns={f"{side}_race_id": "race_id", f"{side}_horse_no": no_col}
    )
    return out.merge(side_profile, on=["race_id", no_col], how="left")


def add_flags(frame: pd.DataFrame, q: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    for side in ["anchor", "partner"]:
        out[f"{side}_venue_lift_hi"] = num(out, f"{side}_venue_lift_combo").ge(q["venue_lift_hi"])
        out[f"{side}_venue_lift_top"] = num(out, f"{side}_venue_lift_combo").ge(q["venue_lift_top"])
        out[f"{side}_course_fit_hi"] = num(out, f"{side}_bloodline_course_fit_score").ge(q["course_fit_hi"])
        out[f"{side}_front_profile_hi"] = num(out, f"{side}_horse_front_run_rate_past5").ge(q["front_hi"])
        out[f"{side}_local_venue_lift_hi"] = out[f"{side}_local_tight"].fillna(False).astype(bool) & out[f"{side}_venue_lift_hi"]
        out[f"{side}_local_course_fit_hi"] = out[f"{side}_local_tight"].fillna(False).astype(bool) & out[f"{side}_course_fit_hi"]
        out[f"{side}_hokkaido_turf_venue_lift_hi"] = out[f"{side}_hokkaido_turf"].fillna(False).astype(bool) & out[f"{side}_venue_lift_hi"]
        out[f"{side}_local_front_venue_lift_hi"] = (
            out[f"{side}_local_venue_lift_hi"] & out[f"{side}_front_profile_hi"]
        )
    out["partner_projected_front_ok"] = num(out, "projected_front5_prob").ge(0.45)
    out["partner_local_lift_projected_front"] = out["partner_local_venue_lift_hi"] & out["partner_projected_front_ok"]
    out["partner_hokkaido_turf_lift_projected_front"] = (
        out["partner_hokkaido_turf_venue_lift_hi"] & out["partner_projected_front_ok"]
    )
    out["either_local_venue_lift_hi"] = out["anchor_local_venue_lift_hi"] | out["partner_local_venue_lift_hi"]
    out["both_local_venue_lift_hi"] = out["anchor_local_venue_lift_hi"] & out["partner_local_venue_lift_hi"]
    return out


def quality_core_mask(frame: pd.DataFrame) -> pd.Series:
    pair_q = num(frame, "pair_quinella_score").quantile(0.75)
    overlay_q = num(frame, "market_overlay_score").quantile(0.75)
    late_q = num(frame, "late_value_survives_score").quantile(0.50)
    front_q = num(frame, "projected_front5_prob").quantile(0.50)
    return (
        num(frame, "pair_quinella_score").ge(pair_q)
        & num(frame, "market_overlay_score").ge(overlay_q)
        & num(frame, "late_value_survives_score").ge(late_q)
        & num(frame, "projected_front5_prob").ge(front_q)
        & num(frame, "anchor_danger").le(0.70)
        & num(frame, "partner_danger").le(0.70)
    )


def evaluate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["stake_100"] = 100.0
    out["umaren_return_100"] = np.where(out["umaren_hit"].astype(bool), num(out, "umaren_pay"), 0.0)
    out["wide_return_100"] = np.where(out["wide_hit"].astype(bool), num(out, "wide_pay"), 0.0)
    qcore = quality_core_mask(out)
    value_partner = (num(out, "partner_pop", 99).ge(4) | num(out, "partner_odds", 0).ge(8.0)) & num(
        out, "partner_ai_rank", 99
    ).le(8)
    anchor_ok = (num(out, "anchor_quinella_model_rank", 99).le(8) | num(out, "anchor_pop", 99).le(3)) & num(
        out, "skip_risk_score"
    ).le(0.75)
    masks = [
        ("all_pair_candidates", pd.Series(True, index=out.index)),
        ("quality_core", qcore),
        ("partner_local_venue_lift_hi", out["partner_local_venue_lift_hi"]),
        ("partner_local_venue_lift_hi_quality_core", out["partner_local_venue_lift_hi"] & qcore),
        ("partner_local_lift_projected_front_quality", out["partner_local_lift_projected_front"] & qcore),
        ("partner_local_lift_front_value_quality", out["partner_local_lift_projected_front"] & qcore & value_partner),
        (
            "partner_local_lift_front_anchor_ok_value_quality",
            out["partner_local_lift_projected_front"] & qcore & value_partner & anchor_ok,
        ),
        ("partner_hokkaido_turf_lift_projected_front_quality", out["partner_hokkaido_turf_lift_projected_front"] & qcore),
        ("partner_local_course_fit_hi_quality", out["partner_local_course_fit_hi"] & qcore),
        ("either_local_venue_lift_hi_quality", out["either_local_venue_lift_hi"] & qcore),
    ]
    rows = []
    for segment, mask in masks:
        part = out[mask.fillna(False)].copy()
        for bet_type, return_col in [("umaren", "umaren_return_100"), ("wide", "wide_return_100")]:
            row = {"source": "candidate_universe", "scope": "candidate_universe", "bet_type": bet_type, "segment": segment}
            row.update(metrics(part, "stake_100", return_col))
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_selected(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    action = frame.get("runtime_action", pd.Series("BUY", index=frame.index)).astype(str)
    buy = action.eq("BUY")
    masks = [
        ("all_buy", buy),
        ("partner_local_venue_lift_hi", buy & frame["partner_local_venue_lift_hi"]),
        ("partner_local_lift_projected_front", buy & frame["partner_local_lift_projected_front"]),
        ("partner_hokkaido_turf_lift_projected_front", buy & frame["partner_hokkaido_turf_lift_projected_front"]),
        ("anchor_local_venue_lift_hi", buy & frame["anchor_local_venue_lift_hi"]),
        ("either_local_venue_lift_hi", buy & frame["either_local_venue_lift_hi"]),
        ("both_local_venue_lift_hi", buy & frame["both_local_venue_lift_hi"]),
        ("partner_local_no_venue_lift_hi", buy & frame["partner_local_tight"].fillna(False).astype(bool) & ~frame["partner_venue_lift_hi"]),
        ("partner_not_local", buy & ~frame["partner_local_tight"].fillna(False).astype(bool)),
    ]
    rows = []
    for segment, mask in masks:
        part = frame[mask.fillna(False)].copy()
        row = {"source": label, "scope": "selected_tickets", "bet_type": "actual_stake", "segment": segment}
        row.update(metrics(part, "stake_yen", "return_yen"))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_runner_segments(test_features: Path, model_path: Path, q: dict[str, float], profile: pd.DataFrame) -> pd.DataFrame:
    test = pd.read_csv(test_features, encoding="utf-8-sig", low_memory=False)
    race_col = test.columns[34]
    pop_col = test.columns[16]
    odds_col = test.columns[17]
    winpay_col = test.columns[31]
    placepay_col = test.columns[32]
    horse_no_col = test.columns[15]
    with model_path.open("rb") as f:
        model = pickle.load(f)
    runner = profile.copy()
    runner["ai_score"] = model.predict(test)
    runner["ai_rank"] = runner.groupby("race_id")["ai_score"].rank(ascending=False, method="first").astype(int)
    runner["win"] = num(test, "target_win").astype(int)
    runner["top3"] = num(test, "target_top3").astype(int)
    runner["win_pay"] = num(test, winpay_col).where(runner["win"].eq(1), 0.0)
    runner["place_pay"] = num(test, placepay_col).where(runner["top3"].eq(1), 0.0)
    runner["pop"] = num(test, pop_col)
    runner["odds"] = num(test, odds_col)
    runner["horse_no"] = num(test, horse_no_col).astype("Int64").astype(str)
    runner["venue_lift_hi"] = runner["venue_lift_combo"].ge(q["venue_lift_hi"])
    runner["venue_lift_top"] = runner["venue_lift_combo"].ge(q["venue_lift_top"])
    runner["course_fit_hi"] = runner["bloodline_course_fit_score"].ge(q["course_fit_hi"])
    runner["front_hi"] = runner["horse_front_run_rate_past5"].ge(q["front_hi"])

    def runner_metrics(label: str, mask: pd.Series) -> dict[str, Any] | None:
        part = runner[mask.fillna(False)]
        if len(part) < 30:
            return None
        stake = len(part) * 100.0
        return {
            "segment": label,
            "bets": int(len(part)),
            "races": int(part["race_id"].nunique()),
            "win_rate": float(part["win"].mean()),
            "top3_rate": float(part["top3"].mean()),
            "win_roi": float(part["win_pay"].sum() / stake),
            "place_roi": float(part["place_pay"].sum() / stake),
            "avg_pop": float(part["pop"].mean()),
            "avg_odds": float(part["odds"].mean()),
        }

    rows = []
    for rank_label, rank_mask in [
        ("ai_top1", runner["ai_rank"].eq(1)),
        ("ai_top3", runner["ai_rank"].le(3)),
        ("ai_top5", runner["ai_rank"].le(5)),
    ]:
        flags = {
            "local_venue_lift_hi": runner["local_tight"] & runner["venue_lift_hi"],
            "local_venue_lift_top": runner["local_tight"] & runner["venue_lift_top"],
            "local_front_venue_lift_hi": runner["local_tight"] & runner["venue_lift_hi"] & runner["front_hi"],
            "hokkaido_turf_venue_lift_hi": runner["hokkaido_turf"] & runner["venue_lift_hi"],
            "local_course_fit_hi": runner["local_tight"] & runner["course_fit_hi"],
            "local_no_venue_lift_hi": runner["local_tight"] & ~runner["venue_lift_hi"],
        }
        for flag_label, flag_mask in flags.items():
            row = runner_metrics(f"{rank_label}&{flag_label}", rank_mask & flag_mask)
            if row:
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["win_roi", "place_roi"], ascending=False)
    return out


def yearly_breakdown(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    action = frame.get("runtime_action", pd.Series("BUY", index=frame.index)).astype(str)
    frame = frame[action.eq("BUY")].copy()
    masks = {
        "all_buy": pd.Series(True, index=frame.index),
        "partner_local_venue_lift_hi": frame["partner_local_venue_lift_hi"],
        "partner_local_lift_projected_front": frame["partner_local_lift_projected_front"],
        "partner_local_no_venue_lift_hi": frame["partner_local_tight"].fillna(False).astype(bool) & ~frame["partner_venue_lift_hi"],
        "partner_not_local": ~frame["partner_local_tight"].fillna(False).astype(bool),
    }
    rows = []
    for segment, mask in masks.items():
        part = frame[mask.fillna(False)].copy()
        for year, year_part in part.groupby("year"):
            row = {"source": label, "segment": segment, "year": int(year)}
            row.update(metrics(year_part, "stake_yen", "return_yen"))
            rows.append(row)
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.4f}"
            vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_review(out_dir: Path, segment_summary: pd.DataFrame, runner_summary: pd.DataFrame, verdict: str) -> None:
    selected = segment_summary[
        (segment_summary["scope"].eq("selected_tickets"))
        & segment_summary["segment"].isin(
            [
                "all_buy",
                "partner_local_venue_lift_hi",
                "partner_local_lift_projected_front",
                "partner_hokkaido_turf_lift_projected_front",
                "partner_local_no_venue_lift_hi",
                "partner_not_local",
            ]
        )
    ][["source", "segment", "tickets", "races", "roi", "ticket_hit_rate", "top5_removed_roi"]]
    candidates = segment_summary[
        (segment_summary["scope"].eq("candidate_universe"))
        & segment_summary["segment"].isin(
            [
                "quality_core",
                "partner_local_venue_lift_hi_quality_core",
                "partner_local_lift_projected_front_quality",
                "partner_local_lift_front_value_quality",
                "partner_local_lift_front_anchor_ok_value_quality",
                "partner_hokkaido_turf_lift_projected_front_quality",
            ]
        )
    ][["bet_type", "segment", "tickets", "races", "roi", "ticket_hit_rate", "top5_removed_roi"]]
    runner_key = runner_summary.head(20)[
        ["segment", "bets", "races", "win_roi", "place_roi", "win_rate", "top3_rate", "avg_pop"]
    ] if not runner_summary.empty else runner_summary
    body = [
        "# ローカル・小回り・洋芝 × 血統 venue lift 検証",
        "",
        "## 結論",
        "",
        verdict,
        "",
        "## 選抜済みチケット",
        "",
        md_table(selected),
        "",
        "## 候補宇宙",
        "",
        md_table(candidates),
        "",
        "## ランナー単体",
        "",
        md_table(runner_key),
        "",
        "## 解釈",
        "",
        "- ローカル/北海道芝の血統 venue lift は、ランナー単体では一部プラスに見える。",
        "- ただし現行BUYに重ねると、相手条件として安定した上乗せにはならない。",
        "- 候補宇宙でもquality_coreに入ると件数が薄くなり、馬連・ワイドとも不安定。",
        "- BUY拡張ではなく、函館/札幌/福島/小倉の準候補シャドー、理由表示、場別ガードの補助に留める。",
    ]
    (out_dir / "review.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local/tight/yoshiba x bloodline venue-lift overlays.")
    parser.add_argument("--train-features", default=DEFAULT_TRAIN_FEATURES)
    parser.add_argument("--test-features", default=DEFAULT_TEST_FEATURES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pair-candidates", default=DEFAULT_PAIR_CANDIDATES)
    parser.add_argument("--dynamic-tickets", default=DEFAULT_DYNAMIC_TICKETS)
    parser.add_argument("--purged-tickets", default=DEFAULT_PURGED_TICKETS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = thresholds_from_train(project_path(args.train_features))
    profile = load_runner_profile(project_path(args.test_features))

    pair_candidates = pd.read_csv(project_path(args.pair_candidates), low_memory=False)
    pair_candidates = add_side_profile(pair_candidates, profile, "anchor", "anchor_no")
    pair_candidates = add_side_profile(pair_candidates, profile, "partner", "partner_no")
    pair_candidates = add_flags(pair_candidates, q)
    candidate_summary = evaluate_candidates(pair_candidates)
    pair_candidates.to_csv(out_dir / "pair_candidate_local_bloodline_profile.csv", index=False, encoding="utf-8-sig")

    summaries = [candidate_summary]
    yearly = []
    for label, path in [
        ("dynamic", project_path(args.dynamic_tickets)),
        ("purged", project_path(args.purged_tickets)),
    ]:
        tickets = pd.read_csv(path, low_memory=False)
        tickets = add_side_profile(tickets, profile, "anchor", "anchor_no")
        tickets = add_side_profile(tickets, profile, "partner", "partner_no")
        tickets = add_flags(tickets, q)
        tickets.to_csv(out_dir / f"{label}_selected_ticket_local_bloodline_profile.csv", index=False, encoding="utf-8-sig")
        summaries.append(evaluate_selected(tickets, label))
        yearly.append(yearly_breakdown(tickets, label))

    segment_summary = pd.concat(summaries, ignore_index=True)
    yearly_summary = pd.concat(yearly, ignore_index=True) if yearly else pd.DataFrame()
    runner_summary = evaluate_runner_segments(project_path(args.test_features), project_path(args.model), q, profile)
    segment_summary.to_csv(out_dir / "segment_summary.csv", index=False, encoding="utf-8-sig")
    yearly_summary.to_csv(out_dir / "yearly_breakdown.csv", index=False, encoding="utf-8-sig")
    runner_summary.to_csv(out_dir / "runner_local_bloodline_segments.csv", index=False, encoding="utf-8-sig")

    dyn = segment_summary[
        (segment_summary["source"].eq("dynamic"))
        & (segment_summary["segment"].eq("partner_local_lift_projected_front"))
    ]
    pur = segment_summary[
        (segment_summary["source"].eq("purged"))
        & (segment_summary["segment"].eq("partner_local_lift_projected_front"))
    ]
    verdict = "最終BUY拡張としては不採用。ローカル/洋芝 × venue lift × 前目は、現行チケットではROI改善が安定しない。"
    if not dyn.empty and not pur.empty and float(dyn.iloc[0]["roi"]) > 1.05 and float(pur.iloc[0]["roi"]) > 1.05:
        verdict = "シャドー昇格候補。両検証でプラスだが、開催場別・年別安定性と上位払戻除外を追加確認してから採用。"

    summary = {
        "output_dir": str(out_dir),
        "thresholds": q,
        "merge_rates": {
            "pair_anchor_profile_rate": float(pair_candidates["anchor_venue_lift_combo"].notna().mean()),
            "pair_partner_profile_rate": float(pair_candidates["partner_venue_lift_combo"].notna().mean()),
        },
        "verdict": verdict,
        "key_metrics": segment_summary[
            segment_summary["segment"].isin(
                ["all_buy", "partner_local_venue_lift_hi", "partner_local_lift_projected_front"]
            )
        ].to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(out_dir, segment_summary, runner_summary, verdict)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
