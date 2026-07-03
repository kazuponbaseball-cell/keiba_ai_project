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
DEFAULT_OUT = "outputs/analysis/maternal_dirt_going_body_overlay_rebuilt_20260623"


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


def going_bucket(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("")
    return pd.Series(
        np.select(
            [
                text.str.contains("良", regex=False, na=False),
                text.str.contains("稍", regex=False, na=False),
                text.str.contains("重", regex=False, na=False),
                text.str.contains("不", regex=False, na=False),
            ],
            ["firm", "slightly_yielding", "yielding", "muddy"],
            default="unknown",
        ),
        index=values.index,
    )


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
    return {
        "bms_surface_lift_hi": float(num(train, "bms_surface_lift").quantile(0.75)),
        "bms_surface_lift_top": float(num(train, "bms_surface_lift").quantile(0.90)),
        "bms_going_lift_hi": float(num(train, "bms_going_lift").quantile(0.75)),
        "bms_going_lift_top": float(num(train, "bms_going_lift").quantile(0.90)),
        "bms_surface_top3_hi": float(num(train, "bms_surface_top3_rate").quantile(0.75)),
        "bms_going_top3_hi": float(num(train, "bms_going_top3_rate").quantile(0.75)),
        "body_weight_hi": float(num(train, "body_prev_weight").quantile(0.75)),
        "body_weight_top": float(num(train, "body_prev_weight").quantile(0.90)),
    }


def load_profile(feature_csv: Path, model_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(feature_csv, encoding="utf-8-sig", low_memory=False)
    out = pd.DataFrame(index=frame.index)
    out["race_id"] = frame["レースID(新/馬番無)"].astype(str)
    out["horse_no"] = num(frame, "馬番").astype("Int64").astype(str)
    out["horse_name"] = frame["馬名"].astype(str)
    out["surface_simple"] = surface_bucket(frame.get("芝・ダ", pd.Series("", index=frame.index)))
    out["going_bucket"] = going_bucket(frame.get("馬場状態", pd.Series("", index=frame.index)))
    out["finish"] = num(frame, "確定着順", np.nan)
    out["popularity"] = num(frame, "人気", np.nan)
    out["odds"] = num(frame, "単勝オッズ", np.nan)
    out["win_pay_100"] = num(frame, "単勝配当", 0.0)
    out["place_pay_100"] = num(frame, "複勝配当", 0.0)
    for col in [
        "bms_surface_lift",
        "bms_going_lift",
        "bms_surface_top3_rate",
        "bms_going_top3_rate",
        "bms_surface_avg_score",
        "bms_going_avg_score",
        "body_prev_weight",
        "body_large_horse_flag",
        "body_very_large_horse_flag",
        "body_race_heavy_top3_flag",
        "body_race_heavy_top5_flag",
        "body_weight_percentile_in_race",
        "body_weight_z_in_race",
    ]:
        out[col] = num(frame, col, 0.0)
    with model_path.open("rb") as f:
        model = pickle.load(f)
    pred = pd.Series(model.predict(frame), index=frame.index, dtype=float)
    out["ai_score"] = pred
    out["ai_rank"] = pred.groupby(out["race_id"]).rank(ascending=False, method="first")
    return out.drop_duplicates(["race_id", "horse_no"], keep="last")


def add_maternal_flags(frame: pd.DataFrame, q: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    out["is_dirt"] = out["surface_simple"].eq("dirt")
    out["is_wetish"] = out["going_bucket"].isin(["slightly_yielding", "yielding", "muddy"])
    out["is_bad"] = out["going_bucket"].isin(["yielding", "muddy"])
    out["bms_surface_lift_hi"] = num(out, "bms_surface_lift").ge(q["bms_surface_lift_hi"])
    out["bms_surface_lift_top"] = num(out, "bms_surface_lift").ge(q["bms_surface_lift_top"])
    out["bms_going_lift_hi"] = num(out, "bms_going_lift").ge(q["bms_going_lift_hi"])
    out["bms_going_lift_top"] = num(out, "bms_going_lift").ge(q["bms_going_lift_top"])
    out["bms_surface_top3_hi"] = num(out, "bms_surface_top3_rate").ge(q["bms_surface_top3_hi"])
    out["bms_going_top3_hi"] = num(out, "bms_going_top3_rate").ge(q["bms_going_top3_hi"])
    out["body_big500"] = num(out, "body_prev_weight").ge(500)
    out["body_big_train_hi"] = num(out, "body_prev_weight").ge(q["body_weight_hi"])
    out["body_big_train_top"] = num(out, "body_prev_weight").ge(q["body_weight_top"])
    out["body_power"] = (
        out["body_big500"]
        | num(out, "body_large_horse_flag").ge(1)
        | num(out, "body_race_heavy_top5_flag").ge(1)
        | num(out, "body_weight_percentile_in_race").ge(0.70)
    )
    out["body_power_strong"] = (
        num(out, "body_very_large_horse_flag").ge(1)
        | num(out, "body_race_heavy_top3_flag").ge(1)
        | num(out, "body_weight_percentile_in_race").ge(0.85)
    )
    out["maternal_dirt_fit"] = out["is_dirt"] & (out["bms_surface_lift_hi"] | out["bms_surface_top3_hi"])
    out["maternal_wet_fit"] = out["is_wetish"] & (out["bms_going_lift_hi"] | out["bms_going_top3_hi"])
    out["maternal_dirt_body_power"] = out["maternal_dirt_fit"] & out["body_power"]
    out["maternal_wet_body_power"] = out["maternal_wet_fit"] & out["body_power"]
    out["maternal_bad_body_power"] = out["is_bad"] & out["maternal_wet_fit"] & out["body_power"]
    out["maternal_dirt_wet_body_power"] = out["is_dirt"] & out["maternal_wet_fit"] & out["body_power"]
    out["maternal_dirt_wet_body_strong"] = out["is_dirt"] & out["maternal_wet_fit"] & out["body_power_strong"]
    out["maternal_power_combo"] = (out["maternal_dirt_body_power"] | out["maternal_wet_body_power"]) & (
        out["bms_surface_lift_hi"] | out["bms_going_lift_hi"]
    )
    return out


def add_side_profile(frame: pd.DataFrame, profile: pd.DataFrame, side: str, no_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = out["race_id"].astype(str)
    out[no_col] = num(out, no_col).astype("Int64").astype(str)
    side_cols = [
        "race_id",
        "horse_no",
        "horse_name",
        "surface_simple",
        "going_bucket",
        "body_prev_weight",
        "bms_surface_lift",
        "bms_going_lift",
        "maternal_dirt_fit",
        "maternal_wet_fit",
        "maternal_dirt_body_power",
        "maternal_wet_body_power",
        "maternal_bad_body_power",
        "maternal_dirt_wet_body_power",
        "maternal_dirt_wet_body_strong",
        "maternal_power_combo",
        "body_power",
        "body_power_strong",
    ]
    side_profile = profile[side_cols].add_prefix(f"{side}_").rename(
        columns={f"{side}_race_id": "race_id", f"{side}_horse_no": no_col}
    )
    return out.merge(side_profile, on=["race_id", no_col], how="left")


def add_pair_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    flag_cols = [
        "maternal_dirt_fit",
        "maternal_wet_fit",
        "maternal_dirt_body_power",
        "maternal_wet_body_power",
        "maternal_bad_body_power",
        "maternal_dirt_wet_body_power",
        "maternal_dirt_wet_body_strong",
        "maternal_power_combo",
        "body_power",
        "body_power_strong",
    ]
    for side in ["anchor", "partner"]:
        for col in flag_cols:
            full = f"{side}_{col}"
            out[full] = out.get(full, False).fillna(False).astype(bool)
    for col in flag_cols:
        out[f"either_{col}"] = out[f"anchor_{col}"] | out[f"partner_{col}"]
    out["partner_projected_front_ok"] = num(out, "projected_front5_prob").ge(0.45)
    out["partner_maternal_power_front"] = out["partner_maternal_power_combo"] & out["partner_projected_front_ok"]
    out["partner_maternal_power_value"] = out["partner_maternal_power_combo"] & (
        num(out, "partner_pop", 99).ge(4) | num(out, "partner_odds", 0).ge(8.0)
    )
    out["partner_maternal_power_front_value"] = out["partner_maternal_power_front"] & out["partner_maternal_power_value"]
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
        ("ai_top3", num(frame, "ai_rank", 99).le(3)),
        ("ai_top5", num(frame, "ai_rank", 99).le(5)),
        ("ai_top5_maternal_dirt_body_power", num(frame, "ai_rank", 99).le(5) & frame["maternal_dirt_body_power"]),
        ("ai_top5_maternal_wet_body_power", num(frame, "ai_rank", 99).le(5) & frame["maternal_wet_body_power"]),
        ("ai_top5_maternal_dirt_wet_body_power", num(frame, "ai_rank", 99).le(5) & frame["maternal_dirt_wet_body_power"]),
        ("ai_top5_maternal_power_value", num(frame, "ai_rank", 99).le(5) & frame["maternal_power_combo"] & num(frame, "popularity", 99).ge(4)),
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
    segments = [
        "all_buy_tickets",
        "partner_maternal_dirt_body_power",
        "partner_maternal_wet_body_power",
        "partner_maternal_bad_body_power",
        "partner_maternal_dirt_wet_body_power",
        "partner_maternal_dirt_wet_body_strong",
        "partner_maternal_power_combo",
        "partner_maternal_power_front",
        "partner_maternal_power_value",
        "partner_maternal_power_front_value",
        "anchor_maternal_power_combo",
        "either_maternal_power_combo",
    ]
    masks: dict[str, pd.Series] = {"all_buy_tickets": buy}
    for segment in segments[1:]:
        masks[segment] = buy & frame[segment]
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
        "partner_maternal_power_combo": frame["partner_maternal_power_combo"],
        "partner_maternal_dirt_body_power": frame["partner_maternal_dirt_body_power"],
        "partner_maternal_wet_body_power": frame["partner_maternal_wet_body_power"],
        "anchor_maternal_power_combo": frame["anchor_maternal_power_combo"],
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
    masks: dict[str, pd.Series] = {
        "all_pair_candidates": pd.Series(True, index=out.index),
        "quality_core": qcore,
        "partner_maternal_power_combo": out["partner_maternal_power_combo"],
        "partner_maternal_power_quality_core": out["partner_maternal_power_combo"] & qcore,
        "partner_maternal_power_front_quality": out["partner_maternal_power_front"] & qcore,
        "partner_maternal_power_front_value_quality": out["partner_maternal_power_front_value"] & qcore,
        "partner_maternal_dirt_body_quality": out["partner_maternal_dirt_body_power"] & qcore,
        "partner_maternal_wet_body_quality": out["partner_maternal_wet_body_power"] & qcore,
        "partner_maternal_dirt_wet_body_quality": out["partner_maternal_dirt_wet_body_power"] & qcore,
        "either_maternal_power_quality": out["either_maternal_power_combo"] & qcore,
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

    runner_rows = runner_summary[
        runner_summary["segment"].astype(str).str.contains("maternal", regex=False)
    ].copy()
    runner_rows = runner_rows.sort_values(["races", "roi"], ascending=[False, False])
    runner_cols = ["segment", "bet_type", "tickets", "races", "roi", "ticket_hit_rate", "top5_removed_roi", "profit_yen"]
    year_rows = year_summary[
        year_summary["segment"].astype(str).str.contains("maternal", regex=False)
    ].copy()
    year_rows = year_rows.sort_values(["source", "segment", "year"])
    year_cols = ["source", "segment", "year", "tickets", "races", "roi", "profit_yen"]
    lines = [
        "# 母系ダート・道悪・馬格 交互作用検証",
        "",
        "## 目的",
        "",
        "母系/BMSのダート・道悪適性が、馬格のある馬でより効くかを確認した。",
        "当日馬体重ではなく、既存データから作れる `前走馬体重/馬体重順位` を馬格 proxy として使った。",
        "",
        "## 結論",
        "",
        "- BUY昇格条件としては採用しない。",
        "- ランナー単体では `AI5位以内 × 母系ダート/道悪 × 馬格` は100%に届かない。",
        "- 既存BUYに後付けしても、相手馬がこの条件を持つケースは全体より悪化。",
        "- ペア候補 universe でも、ワイド/馬連とも安定した上振れは出ていない。",
        "- 使うなら、馬格が薄い母系ダート/道悪馬を過大評価しないための説明/警戒タグが妥当。",
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
        "## Dynamic BUY",
        "",
        markdown_table(pick("dynamic_selected_tickets", "all", "maternal")),
        "",
        "## Purged BUY",
        "",
        markdown_table(pick("purged_selected_tickets", "all", "maternal")),
        "",
        "## ペア候補 universe",
        "",
        markdown_table(pick("pair_candidate_universe", "wide", "maternal")),
        "",
        "## 年度別",
        "",
        markdown_table(year_rows[year_cols].to_dict("records")),
        "",
        "## 採用判断",
        "",
        "`不採用 / 説明・警戒タグ止まり`。条件としては筋が良いが、既存モデルの回収率改善には直結していない。",
        "次に検証するなら、母系だけでなく、当日馬体重の増減・当日馬場変化・T-5/T-3オッズ残存を加えたシャドー検証で扱う。",
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

    profile = add_maternal_flags(load_profile(test_features, model_path), q)
    profile.to_csv(out_dir / "runner_maternal_body_profile.csv", index=False, encoding="utf-8-sig")

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

    dynamic.to_csv(out_dir / "dynamic_selected_with_maternal_body.csv", index=False, encoding="utf-8-sig")
    purged.to_csv(out_dir / "purged_selected_with_maternal_body.csv", index=False, encoding="utf-8-sig")
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
            "maternal_dirt_body_power": int(profile["maternal_dirt_body_power"].sum()),
            "maternal_wet_body_power": int(profile["maternal_wet_body_power"].sum()),
            "maternal_dirt_wet_body_power": int(profile["maternal_dirt_wet_body_power"].sum()),
            "maternal_power_combo": int(profile["maternal_power_combo"].sum()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review.md").write_text(render_review(summary, segment_summary, runner_summary, year_summary), encoding="utf-8")
    print(json.dumps(json_ready({"ok": True, "out_dir": str(out_dir), **summary["flag_counts"]}), ensure_ascii=False))


if __name__ == "__main__":
    main()
