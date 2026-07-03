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
    "body_weight_backfilled/train_features_with_same_day_bias_v3_retro_body_context.csv"
)
DEFAULT_TEST_FEATURES = (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/"
    "body_weight_backfilled/test_features_with_same_day_bias_v3_retro_body_context.csv"
)
DEFAULT_MODEL = "models/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/baseline_ranker.pkl"
DEFAULT_PAIR_CANDIDATES = "outputs/analysis/dynamic_pair_ticket_allocation_rebuilt_20260623/pair_candidate_universe.csv"
DEFAULT_DYNAMIC_TICKETS = "outputs/analysis/dynamic_pair_ticket_allocation_rebuilt_20260623/walkforward_selected_tickets.csv"
DEFAULT_PURGED_TICKETS = "outputs/analysis/purged_walkforward_mcs_pbo_rebuilt_20260623/purged_walkforward_selected_tickets.csv"
DEFAULT_OUT = "outputs/analysis/bloodline_only_buy_conditions_rebuilt_20260623"


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


def ticket_metrics(frame: pd.DataFrame, stake_col: str, return_col: str) -> dict[str, Any]:
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


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}" if math.isfinite(value) else ""
            vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def surface_bucket(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("")
    return pd.Series(
        np.select(
            [text.str.contains("芝", regex=False, na=False), text.str.contains("ダ", regex=False, na=False)],
            ["turf", "dirt"],
            default="other",
        ),
        index=values.index,
    )


def thresholds_from_train(train_csv: Path) -> dict[str, float]:
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    cols = [
        "bloodline_high_confidence_fit_score",
        "bloodline_lift_fit_score",
        "bloodline_course_fit_score",
        "bloodline_surface_distance_fit_score",
        "bloodline_pair_fit_score",
        "sire_surface_lift",
        "bms_surface_lift",
        "sire_distance_lift",
        "bms_distance_lift",
        "sire_venue_lift",
        "bms_venue_lift",
    ]
    q: dict[str, float] = {}
    for col in cols:
        s = num(train, col)
        q[f"{col}_q75"] = float(s.quantile(0.75))
        q[f"{col}_q90"] = float(s.quantile(0.90))
        q[f"{col}_q95"] = float(s.quantile(0.95))
    return q


def load_profile(feature_csv: Path, model_path: Path, q: dict[str, float]) -> pd.DataFrame:
    frame = pd.read_csv(feature_csv, encoding="utf-8-sig", low_memory=False)
    out = pd.DataFrame(index=frame.index)
    out["race_id"] = frame["レースID(新/馬番無)"].astype(str)
    out["horse_no"] = num(frame, "馬番").astype("Int64").astype(str)
    out["horse_name"] = frame["馬名"].astype(str)
    out["age"] = num(frame, "年齢", np.nan)
    out["surface_simple"] = surface_bucket(frame.get("芝・ダ", pd.Series("", index=frame.index)))
    out["finish"] = num(frame, "確定着順", np.nan)
    out["popularity"] = num(frame, "人気", np.nan)
    out["odds"] = num(frame, "単勝オッズ", np.nan)
    out["win_pay_100"] = num(frame, "単勝配当", 0.0)
    out["place_pay_100"] = num(frame, "複勝配当", 0.0)
    score_cols = [
        "bloodline_high_confidence_fit_score",
        "bloodline_lift_fit_score",
        "bloodline_course_fit_score",
        "bloodline_surface_distance_fit_score",
        "bloodline_pair_fit_score",
        "bloodline_reliability_score",
        "bloodline_low_sample_flag",
        "sire_surface_lift",
        "bms_surface_lift",
        "sire_distance_lift",
        "bms_distance_lift",
        "sire_venue_lift",
        "bms_venue_lift",
    ]
    for col in score_cols:
        out[col] = num(frame, col, 0.0)
    with model_path.open("rb") as f:
        model = pickle.load(f)
    pred = pd.Series(model.predict(frame), index=frame.index, dtype=float)
    out["ai_score"] = pred
    out["ai_rank"] = pred.groupby(out["race_id"]).rank(ascending=False, method="first")
    out["blood_reliable"] = num(out, "bloodline_reliability_score").ge(0.70) & num(out, "bloodline_low_sample_flag").le(0)
    out["blood_conf_hi"] = num(out, "bloodline_high_confidence_fit_score").ge(q["bloodline_high_confidence_fit_score_q75"])
    out["blood_conf_top"] = num(out, "bloodline_high_confidence_fit_score").ge(q["bloodline_high_confidence_fit_score_q90"])
    out["blood_conf_elite"] = num(out, "bloodline_high_confidence_fit_score").ge(q["bloodline_high_confidence_fit_score_q95"])
    out["blood_lift_hi"] = num(out, "bloodline_lift_fit_score").ge(q["bloodline_lift_fit_score_q75"])
    out["blood_lift_top"] = num(out, "bloodline_lift_fit_score").ge(q["bloodline_lift_fit_score_q90"])
    out["blood_course_hi"] = num(out, "bloodline_course_fit_score").ge(q["bloodline_course_fit_score_q75"])
    out["blood_course_top"] = num(out, "bloodline_course_fit_score").ge(q["bloodline_course_fit_score_q90"])
    out["blood_surface_distance_hi"] = num(out, "bloodline_surface_distance_fit_score").ge(q["bloodline_surface_distance_fit_score_q75"])
    out["blood_surface_distance_top"] = num(out, "bloodline_surface_distance_fit_score").ge(q["bloodline_surface_distance_fit_score_q90"])
    out["blood_pair_hi"] = num(out, "bloodline_pair_fit_score").ge(q["bloodline_pair_fit_score_q75"])
    out["bloodline_only_hi"] = out["blood_reliable"] & (
        out["blood_conf_top"]
        | (out["blood_lift_top"] & out["blood_course_hi"])
        | (out["blood_surface_distance_top"] & out["blood_lift_hi"])
        | (out["blood_pair_hi"] & out["blood_course_top"])
    )
    out["bloodline_only_elite"] = out["blood_reliable"] & (
        (out["blood_conf_elite"] & out["blood_lift_hi"])
        | (out["blood_course_top"] & out["blood_surface_distance_top"] & out["blood_lift_hi"])
    )
    out["bloodline_only_value"] = out["bloodline_only_hi"] & (num(out, "popularity", 99).ge(4) | num(out, "odds", 0).ge(8.0))
    out["bloodline_only_young_dirt"] = out["bloodline_only_hi"] & out["age"].le(2) & out["surface_simple"].eq("dirt")
    out["bloodline_only_young_dirt_value"] = out["bloodline_only_young_dirt"] & (
        num(out, "popularity", 99).ge(2) | num(out, "odds", 0).ge(3.0)
    )
    return out.drop_duplicates(["race_id", "horse_no"], keep="last")


def add_side_profile(frame: pd.DataFrame, profile: pd.DataFrame, side: str, no_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = out["race_id"].astype(str)
    out[no_col] = num(out, no_col).astype("Int64").astype(str)
    side_cols = [
        "race_id",
        "horse_no",
        "horse_name",
        "bloodline_only_hi",
        "bloodline_only_elite",
        "bloodline_only_value",
        "bloodline_only_young_dirt",
        "bloodline_only_young_dirt_value",
        "bloodline_high_confidence_fit_score",
        "bloodline_lift_fit_score",
        "bloodline_course_fit_score",
    ]
    side_profile = profile[side_cols].add_prefix(f"{side}_").rename(
        columns={f"{side}_race_id": "race_id", f"{side}_horse_no": no_col}
    )
    return out.merge(side_profile, on=["race_id", no_col], how="left")


def add_pair_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    flags = [
        "bloodline_only_hi",
        "bloodline_only_elite",
        "bloodline_only_value",
        "bloodline_only_young_dirt",
        "bloodline_only_young_dirt_value",
    ]
    for side in ["anchor", "partner"]:
        for flag in flags:
            out[f"{side}_{flag}"] = out.get(f"{side}_{flag}", False).fillna(False).astype(bool)
    for flag in flags:
        out[f"either_{flag}"] = out[f"anchor_{flag}"] | out[f"partner_{flag}"]
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


def evaluate_runner(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    frame["stake_100"] = 100.0
    frame["win_return_100"] = np.where(num(frame, "finish", 99).eq(1), num(frame, "win_pay_100"), 0.0)
    frame["place_return_100"] = np.where(num(frame, "finish", 99).le(3), num(frame, "place_pay_100"), 0.0)
    masks: list[tuple[str, pd.Series]] = [
        ("all_runners", pd.Series(True, index=frame.index)),
        ("bloodline_only_hi", frame["bloodline_only_hi"]),
        ("bloodline_only_elite", frame["bloodline_only_elite"]),
        ("bloodline_only_value", frame["bloodline_only_value"]),
        ("bloodline_only_young_dirt", frame["bloodline_only_young_dirt"]),
        ("bloodline_only_young_dirt_value", frame["bloodline_only_young_dirt_value"]),
        ("ai_top3", num(frame, "ai_rank", 99).le(3)),
        ("ai_top3_bloodline_only_hi", num(frame, "ai_rank", 99).le(3) & frame["bloodline_only_hi"]),
        ("ai_top1_young_dirt_bloodline_hi", num(frame, "ai_rank", 99).le(1) & frame["bloodline_only_young_dirt"]),
    ]
    rows = []
    for name, mask in masks:
        part = frame[mask.fillna(False)]
        for bet, ret_col in [("win", "win_return_100"), ("place", "place_return_100")]:
            rows.append({"source": "runner", "segment": name, "bet_type": bet, **ticket_metrics(part, "stake_100", ret_col)})
    return pd.DataFrame(rows)


def evaluate_selected(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    action = frame.get("runtime_action", pd.Series("BUY", index=frame.index)).astype(str)
    buy = action.eq("BUY")
    masks: dict[str, pd.Series] = {
        "all_buy_tickets": buy,
        "partner_bloodline_only_hi": buy & frame["partner_bloodline_only_hi"],
        "partner_bloodline_only_elite": buy & frame["partner_bloodline_only_elite"],
        "partner_bloodline_only_value": buy & frame["partner_bloodline_only_value"],
        "partner_bloodline_only_young_dirt": buy & frame["partner_bloodline_only_young_dirt"],
        "anchor_bloodline_only_hi": buy & frame["anchor_bloodline_only_hi"],
        "either_bloodline_only_hi": buy & frame["either_bloodline_only_hi"],
    }
    rows = []
    for name, mask in masks.items():
        part = frame[mask.fillna(False)]
        for ticket_type in sorted(part["ticket_type"].dropna().unique()):
            ticket_part = part[part["ticket_type"].eq(ticket_type)]
            rows.append({"source": label, "segment": name, "bet_type": ticket_type, **ticket_metrics(ticket_part, "stake_yen", "return_yen")})
        rows.append({"source": label, "segment": name, "bet_type": "all", **ticket_metrics(part, "stake_yen", "return_yen")})
    return pd.DataFrame(rows)


def evaluate_selected_by_year(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    masks: dict[str, pd.Series] = {
        "all_buy_tickets": pd.Series(True, index=frame.index),
        "partner_bloodline_only_hi": frame["partner_bloodline_only_hi"],
        "partner_bloodline_only_elite": frame["partner_bloodline_only_elite"],
        "anchor_bloodline_only_hi": frame["anchor_bloodline_only_hi"],
    }
    rows = []
    for name, mask in masks.items():
        for year, part in frame[mask.fillna(False)].groupby("year", sort=True):
            rows.append({"source": label, "segment": name, "year": int(year), **ticket_metrics(part, "stake_yen", "return_yen")})
    return pd.DataFrame(rows)


def evaluate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["stake_100"] = 100.0
    out["umaren_return_100"] = np.where(out["umaren_hit"].astype(bool), num(out, "umaren_pay"), 0.0)
    out["wide_return_100"] = np.where(out["wide_hit"].astype(bool), num(out, "wide_pay"), 0.0)
    qcore = quality_core_mask(out)
    partner_front = num(out, "projected_front5_prob").ge(0.45)
    partner_value = (num(out, "partner_pop", 99).ge(4) | num(out, "partner_odds", 0).ge(8.0))
    masks: dict[str, pd.Series] = {
        "all_pair_candidates": pd.Series(True, index=out.index),
        "quality_core": qcore,
        "partner_bloodline_only_hi": out["partner_bloodline_only_hi"],
        "partner_bloodline_only_value": out["partner_bloodline_only_hi"] & partner_value,
        "partner_bloodline_only_front_value": out["partner_bloodline_only_hi"] & partner_value & partner_front,
        "partner_bloodline_only_hi_quality": out["partner_bloodline_only_hi"] & qcore,
        "partner_bloodline_only_value_quality": out["partner_bloodline_only_hi"] & partner_value & qcore,
        "partner_bloodline_only_elite_quality": out["partner_bloodline_only_elite"] & qcore,
        "partner_bloodline_only_young_dirt_quality": out["partner_bloodline_only_young_dirt"] & qcore,
        "anchor_bloodline_only_hi_quality": out["anchor_bloodline_only_hi"] & qcore,
        "either_bloodline_only_hi_quality": out["either_bloodline_only_hi"] & qcore,
    }
    rows = []
    for name, mask in masks.items():
        part = out[mask.fillna(False)]
        for bet, ret_col in [("wide", "wide_return_100"), ("umaren", "umaren_return_100")]:
            rows.append({"source": "pair_candidate_universe", "segment": name, "bet_type": bet, **ticket_metrics(part, "stake_100", ret_col)})
    return pd.DataFrame(rows)


def render_review(summary: dict[str, Any], segment_summary: pd.DataFrame, runner_summary: pd.DataFrame, year_summary: pd.DataFrame) -> str:
    def pick(source: str, bet_type: str, contains: str, n: int = 12) -> list[dict[str, Any]]:
        part = segment_summary[
            segment_summary["source"].eq(source)
            & segment_summary["bet_type"].eq(bet_type)
            & segment_summary["segment"].astype(str).str.contains(contains, regex=False)
        ].copy()
        part = part.sort_values(["races", "roi"], ascending=[False, False]).head(n)
        cols = ["segment", "bet_type", "tickets", "races", "roi", "ticket_hit_rate", "top5_removed_roi", "profit_yen"]
        return part[cols].to_dict("records")

    runner_rows = runner_summary[runner_summary["segment"].astype(str).str.contains("bloodline", regex=False)].copy()
    runner_rows = runner_rows.sort_values(["races", "roi"], ascending=[False, False])
    runner_cols = ["segment", "bet_type", "tickets", "races", "roi", "ticket_hit_rate", "top5_removed_roi", "profit_yen"]
    year_rows = year_summary[year_summary["segment"].astype(str).str.contains("bloodline", regex=False)].copy()
    year_rows = year_rows.sort_values(["source", "segment", "year"])
    year_cols = ["source", "segment", "year", "tickets", "races", "roi", "profit_yen"]
    lines = [
        "# 血統単独BUY条件 検証",
        "",
        "## 目的",
        "",
        "AI順位・展開・前目確率を主条件にせず、血統スコアだけでBUY条件を追加できるかを確認した。",
        "参考として、既存のquality_coreに重ねた場合も見るが、採用判断は血統単独条件を重視する。",
        "",
        "## 結論",
        "",
        "- 血統単独BUY条件は採用しない。",
        "- 血統上位だけでは単勝/複勝とも100%に届かない。",
        "- 2歳ダートの血統上位は以前から小さく見えていたが、単独BUYにできるほどの厚みはない。",
        "- 既存quality_coreに重ねたワイドは一部100%超だが、血統が主因ではなく既存quality_core側の効果が大きい。",
        "- 血統はBUY数を増やす主条件ではなく、低キャリア/初条件の不確実性補正と説明タグとして使うのが安全。",
        "",
        "## 入力",
        "",
        f"- train_features: `{summary['inputs']['train_features']}`",
        f"- test_features: `{summary['inputs']['test_features']}`",
        f"- dynamic_tickets: `{summary['inputs']['dynamic_tickets']}`",
        f"- purged_tickets: `{summary['inputs']['purged_tickets']}`",
        "",
        "## ランナー単体",
        "",
        markdown_table(runner_rows[runner_cols].to_dict("records")),
        "",
        "## Dynamic BUY 後付け",
        "",
        markdown_table(pick("dynamic_selected_tickets", "all", "bloodline")),
        "",
        "## Purged BUY 後付け",
        "",
        markdown_table(pick("purged_selected_tickets", "all", "bloodline")),
        "",
        "## ペア候補 universe",
        "",
        markdown_table(pick("pair_candidate_universe", "wide", "bloodline")),
        "",
        "## 年度別",
        "",
        markdown_table(year_rows[year_cols].to_dict("records")),
        "",
        "## 採用判断",
        "",
        "`不採用`。これで外部AI優先度表の血統系 6-8 は一通り検証済み。",
        "残すなら、買い条件ではなく、画面の評価理由・低キャリア馬の救済候補・シャドー台帳に限定する。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", default=DEFAULT_TRAIN_FEATURES)
    parser.add_argument("--test-features", default=DEFAULT_TEST_FEATURES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pair-candidates", default=DEFAULT_PAIR_CANDIDATES)
    parser.add_argument("--dynamic-tickets", default=DEFAULT_DYNAMIC_TICKETS)
    parser.add_argument("--purged-tickets", default=DEFAULT_PURGED_TICKETS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_features = project_path(args.train_features)
    test_features = project_path(args.test_features)
    model_path = project_path(args.model)
    pair_candidates_path = project_path(args.pair_candidates)
    dynamic_tickets_path = project_path(args.dynamic_tickets)
    purged_tickets_path = project_path(args.purged_tickets)
    q = thresholds_from_train(train_features)

    profile = load_profile(test_features, model_path, q)
    profile.to_csv(out_dir / "runner_bloodline_only_profile.csv", index=False, encoding="utf-8-sig")

    dynamic = pd.read_csv(dynamic_tickets_path, encoding="utf-8-sig", low_memory=False)
    purged = pd.read_csv(purged_tickets_path, encoding="utf-8-sig", low_memory=False)
    candidates = pd.read_csv(pair_candidates_path, encoding="utf-8-sig", low_memory=False)
    for frame in [dynamic, purged, candidates]:
        frame["race_id"] = frame["race_id"].astype(str)
        frame["anchor_no"] = num(frame, "anchor_no").astype("Int64").astype(str)
        frame["partner_no"] = num(frame, "partner_no").astype("Int64").astype(str)

    dynamic = add_pair_flags(add_side_profile(add_side_profile(dynamic, profile, "anchor", "anchor_no"), profile, "partner", "partner_no"))
    purged = add_pair_flags(add_side_profile(add_side_profile(purged, profile, "anchor", "anchor_no"), profile, "partner", "partner_no"))
    candidates = add_pair_flags(add_side_profile(add_side_profile(candidates, profile, "anchor", "anchor_no"), profile, "partner", "partner_no"))
    dynamic.to_csv(out_dir / "dynamic_selected_with_bloodline_only.csv", index=False, encoding="utf-8-sig")
    purged.to_csv(out_dir / "purged_selected_with_bloodline_only.csv", index=False, encoding="utf-8-sig")

    runner_summary = evaluate_runner(profile)
    segment_summary = pd.concat(
        [
            evaluate_selected(dynamic, "dynamic_selected_tickets"),
            evaluate_selected(purged, "purged_selected_tickets"),
            evaluate_candidates(candidates),
        ],
        ignore_index=True,
    )
    year_summary = pd.concat(
        [
            evaluate_selected_by_year(dynamic, "dynamic_selected_tickets"),
            evaluate_selected_by_year(purged, "purged_selected_tickets"),
        ],
        ignore_index=True,
    )
    runner_summary.to_csv(out_dir / "runner_segments.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(out_dir / "segment_summary.csv", index=False, encoding="utf-8-sig")
    year_summary.to_csv(out_dir / "selected_year_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "inputs": {
            "train_features": str(train_features),
            "test_features": str(test_features),
            "model": str(model_path),
            "dynamic_tickets": str(dynamic_tickets_path),
            "purged_tickets": str(purged_tickets_path),
            "pair_candidates": str(pair_candidates_path),
        },
        "thresholds": q,
        "flag_counts": {
            "all_runners": int(len(profile)),
            "bloodline_only_hi": int(profile["bloodline_only_hi"].sum()),
            "bloodline_only_elite": int(profile["bloodline_only_elite"].sum()),
            "bloodline_only_value": int(profile["bloodline_only_value"].sum()),
            "bloodline_only_young_dirt": int(profile["bloodline_only_young_dirt"].sum()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review.md").write_text(render_review(summary, segment_summary, runner_summary, year_summary), encoding="utf-8")
    print(json.dumps(json_ready({"ok": True, "out_dir": str(out_dir), **summary["flag_counts"]}), ensure_ascii=False))


if __name__ == "__main__":
    main()
