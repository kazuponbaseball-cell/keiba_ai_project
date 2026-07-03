from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


def _num(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _fmt_prob(value: float) -> float:
    return round(max(0.0, min(1.0, value)) * 100.0, 1)


def _display_probability_triplet(row: pd.Series) -> tuple[float, float, float]:
    """Return display-safe win/top2/top3 probabilities.

    Some upstream columns are model scores rather than calibrated event
    probabilities. The dashboard should still respect the event hierarchy:
    win <= quinella/top2 <= place/top3.
    """

    win = max(0.0, min(1.0, _num(row.get("ai_win_prob_proxy"), 0.0)))
    place_raw = max(0.0, min(1.0, _num(row.get("place_suitability_score"), 0.0)))
    # quinella_model_prob is a per-race softmax that sums to 1. Approximate a
    # top-2 event probability by scaling to two slots, then clamp for display.
    quinella_raw = max(0.0, min(1.0, _num(row.get("quinella_model_prob"), 0.0) * 2.0))
    place = max(place_raw, win)
    quinella = min(place, max(win, quinella_raw))
    return win, quinella, place


def _safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _safe_int_or_blank(value) -> int | str:
    try:
        if pd.isna(value):
            return ""
        return int(float(value))
    except Exception:
        return ""


def _round_or_blank(value, digits: int = 2) -> float | str:
    try:
        if pd.isna(value):
            return ""
        return round(float(value), digits)
    except Exception:
        return ""


def _normalize_race_date(value) -> str:
    text = _safe_text(value).strip()
    if not text:
        return ""
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return text


def _display_race_date(value: str) -> str:
    text = _normalize_race_date(value)
    try:
        dt = datetime.strptime(text, "%Y-%m-%d")
        return f"{dt.year}年{dt.month}月{dt.day}日"
    except Exception:
        return text


def _pace_label(value: str) -> str:
    text = str(value or "").lower()
    if "slow" in text or "スロー" in text:
        return "スロー"
    if "fast" in text or "high" in text or "ハイ" in text:
        return "ハイ"
    return "ミドル"


def _race_volatility(row: pd.Series) -> tuple[str, str]:
    difficulty = _num(row.get("race_difficulty_score"), 0.5)
    field = _num(row.get("race_field_size"), 0.0)
    top3_market = _num(row.get("race_market_top3_prob_sum"), 0.0)
    pace_collapse = _num(row.get("race_pace_collapse"), 0.0)
    score = 0.48 * difficulty + 0.18 * min(field / 18.0, 1.0) + 0.18 * pace_collapse + 0.16 * max(0.0, 1.0 - top3_market)
    if score >= 0.68:
        return "荒れる", "vol-high"
    if score >= 0.54:
        return "やや荒れ", "vol-mid"
    if score >= 0.40:
        return "やや固い", "vol-low"
    return "固い", "vol-solid"


def _race_action(profile: str, volatility: str, difficulty: float, pace_collapse: float) -> tuple[str, str, str]:
    if profile == "強気" and difficulty < 0.62 and pace_collapse < 0.65:
        return "買い", "action-buy", "条件は素直。オッズ低下だけ確認"
    if profile in ("強気", "標準") and volatility in ("荒れる", "やや荒れ"):
        return "条件付き", "action-caution", "荒れ要素あり。軸固定より相手分散"
    if profile in ("強気", "標準"):
        return "買い寄り", "action-lean", "買えるが直前オッズと馬体重を確認"
    if profile == "広め":
        return "小さく", "action-light", "妙味待ち。無理に厚くしない"
    return "見送り", "action-skip", "期待値か安定度が不足"


def _runner_points(row: pd.Series) -> tuple[list[str], list[str]]:
    points: list[str] = []
    concerns: list[str] = []
    if _num(row.get("ai_rank_num"), 99) <= 3:
        points.append("AI上位評価")
    if _num(row.get("ai_market_prob_diff"), 0.0) > 0.03 or _num(row.get("market_overlay_score"), 0.0) >= 0.60:
        points.append("市場評価より妙味")
    if _num(row.get("quinella_model_score_norm"), 0.0) >= 0.60:
        points.append("連対モデル高評価")
    if _num(row.get("projected_front5_prob"), 0.0) >= 0.60:
        points.append("前目に付ける確率高め")
    if _num(row.get("pace_fit_score"), _num(row.get("pace_fit_score_feature"), 0.0)) >= 0.60:
        points.append("想定展開に合う")
    if _num(row.get("vertical_underpopular_value_score"), 0.0) >= 0.62:
        points.append("過小評価の可能性")

    if _num(row.get("danger_popular_hybrid_score"), _num(row.get("danger_favorite_score"), 0.0)) >= 0.60:
        concerns.append("人気なら過信注意")
    if _num(row.get("vertical_overpopular_risk_score"), 0.0) >= 0.66:
        concerns.append("今回条件が過去好走型とズレる")
    if _num(row.get("skip_risk_score"), 0.0) >= 0.55:
        concerns.append("見送りリスク高め")
    if _num(row.get("late_odds_drift_rate"), 0.0) >= 0.55:
        concerns.append("直前オッズ悪化に注意")
    if _num(row.get("race_pace_collapse_risk"), _num(row.get("race_pace_collapse"), 0.0)) >= 0.75:
        concerns.append("展開崩れリスク")

    return points[:3] or ["強調材料は控えめ"], concerns[:3] or ["大きな減点は薄い"]


def _score_for_pace(row: pd.Series, pace: str) -> float:
    if pace == "slow":
        return _num(row.get("slow_ai_score"), _num(row.get("ai_score"), 0.0))
    if pace == "fast":
        return _num(row.get("fast_ai_score"), _num(row.get("ai_score"), 0.0))
    return _num(row.get("middle_ai_score"), _num(row.get("ai_score"), 0.0))


def _merge_live_pair_odds(tickets: pd.DataFrame, live_pair_odds: pd.DataFrame | None) -> pd.DataFrame:
    if live_pair_odds is None or live_pair_odds.empty:
        tickets["live_pay"] = pd.NA
        tickets["live_odds"] = pd.NA
        tickets["live_snapshot_at"] = ""
        return tickets

    live = live_pair_odds.copy()
    live["race_id"] = live["race_id"].astype(str)
    live["ticket_type"] = live["ticket_type"].astype(str)
    for col in ("a_no", "b_no"):
        live[col] = pd.to_numeric(live[col], errors="coerce").fillna(0).astype(int)
    live["horse_a"] = live[["a_no", "b_no"]].min(axis=1)
    live["horse_b"] = live[["a_no", "b_no"]].max(axis=1)
    live = live.rename(
        columns={
            "live_pay_per100": "live_pay",
            "snapshot_at": "live_snapshot_at",
        }
    )
    keep = ["race_id", "ticket_type", "horse_a", "horse_b", "live_pay", "live_odds", "live_snapshot_at"]
    live = live[[c for c in keep if c in live.columns]].drop_duplicates(["race_id", "ticket_type", "horse_a", "horse_b"], keep="last")

    out = tickets.copy()
    if "horse_a" not in out.columns or "horse_b" not in out.columns:
        anchor = pd.to_numeric(out.get("anchor_no"), errors="coerce").fillna(0).astype(int)
        partner = pd.to_numeric(out.get("partner_no"), errors="coerce").fillna(0).astype(int)
        out["horse_a"] = np.minimum(anchor, partner)
        out["horse_b"] = np.maximum(anchor, partner)
    return out.merge(live, on=["race_id", "ticket_type", "horse_a", "horse_b"], how="left")


def _merge_body_weight(scored: pd.DataFrame, body_weight: pd.DataFrame | None) -> pd.DataFrame:
    out = scored.copy()
    out["body_weight_live"] = pd.NA
    out["body_weight_diff_live"] = pd.NA
    out["body_weight_snapshot_at"] = ""
    if body_weight is None or body_weight.empty:
        return out

    body = body_weight.copy()
    body["race_id"] = body["race_id"].astype(str)
    if "horse_no" not in body.columns and "馬番" in body.columns:
        body = body.rename(columns={"馬番": "horse_no"})
    if "body_weight" not in body.columns and "馬体重" in body.columns:
        body = body.rename(columns={"馬体重": "body_weight"})
    if "body_weight_diff" not in body.columns and "増減" in body.columns:
        body = body.rename(columns={"増減": "body_weight_diff"})
    if "snapshot_at" in body.columns:
        body = body.rename(columns={"snapshot_at": "body_weight_snapshot_at"})
    for col in ("horse_no", "body_weight", "body_weight_diff"):
        if col in body.columns:
            body[col] = pd.to_numeric(body[col], errors="coerce")
    keep = ["race_id", "horse_no", "body_weight", "body_weight_diff", "body_weight_snapshot_at"]
    body = body[[c for c in keep if c in body.columns]].drop_duplicates(["race_id", "horse_no"], keep="last")
    body = body.rename(columns={"body_weight": "body_weight_live", "body_weight_diff": "body_weight_diff_live"})
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce")
    return out.drop(columns=["body_weight_live", "body_weight_diff_live", "body_weight_snapshot_at"]).merge(body, on=["race_id", "horse_no"], how="left")


def _merge_live_single_odds(scored: pd.DataFrame, live_single_odds: pd.DataFrame | None) -> pd.DataFrame:
    out = scored.copy()
    out["live_win_odds"] = pd.NA
    out["live_popularity"] = pd.NA
    out["live_place_odds_min"] = pd.NA
    out["live_place_odds_max"] = pd.NA
    out["live_single_snapshot_at"] = ""
    if live_single_odds is None or live_single_odds.empty:
        return out

    live = live_single_odds.copy()
    live["race_id"] = live["race_id"].astype(str)
    if "horse_no" not in live.columns and "馬番" in live.columns:
        live = live.rename(columns={"馬番": "horse_no"})
    live["horse_no"] = pd.to_numeric(live["horse_no"], errors="coerce")
    rename = {"snapshot_at": "live_single_snapshot_at"}
    live = live.rename(columns={k: v for k, v in rename.items() if k in live.columns})
    keep = ["race_id", "horse_no", "live_win_odds", "live_popularity", "live_place_odds_min", "live_place_odds_max", "live_single_snapshot_at"]
    for col in keep:
        if col not in live.columns:
            live[col] = pd.NA
    live = live[keep].drop_duplicates(["race_id", "horse_no"], keep="last")
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce")
    return out.drop(columns=["live_win_odds", "live_popularity", "live_place_odds_min", "live_place_odds_max", "live_single_snapshot_at"]).merge(
        live, on=["race_id", "horse_no"], how="left"
    )


def _build_payload(
    scored: pd.DataFrame,
    tickets: pd.DataFrame,
    max_races: int | None,
    live_pair_odds: pd.DataFrame | None = None,
    live_single_odds: pd.DataFrame | None = None,
    body_weight: pd.DataFrame | None = None,
) -> dict:
    scored = scored.copy()
    tickets = tickets.copy()
    scored["race_id"] = scored["race_id"].astype(str)
    tickets["race_id"] = tickets["race_id"].astype(str)
    tickets = _merge_live_pair_odds(tickets, live_pair_odds)
    scored = _merge_live_single_odds(scored, live_single_odds)
    scored = _merge_body_weight(scored, body_weight)

    target_races = (
        tickets[~tickets["operation_profile"].eq("skip")]
        .sort_values(["operation_strength_rank", "pair_quinella_score", "market_overlay_score"], ascending=[False, False, False])["race_id"]
        .drop_duplicates()
        .tolist()
    )
    if max_races:
        target_races = target_races[:max_races]

    scored = scored[scored["race_id"].isin(target_races)].copy()
    tickets = tickets[tickets["race_id"].isin(target_races)].copy()

    races = []
    for race_id in target_races:
        runners = scored[scored["race_id"].eq(race_id)].copy()
        race_tickets = tickets[tickets["race_id"].eq(race_id)].copy()
        if runners.empty:
            continue
        first = runners.iloc[0]
        race_date = _normalize_race_date(first.get("日付S"))
        profile_rank = race_tickets["operation_strength_rank"].max() if not race_tickets.empty else 0
        profile = "見送り"
        if profile_rank >= 3:
            profile = "強気"
        elif profile_rank == 2:
            profile = "標準"
        elif profile_rank == 1:
            profile = "広め"
        volatility, volatility_class = _race_volatility(race_tickets.iloc[0] if not race_tickets.empty else first)
        pace = _pace_label(first.get("expected_pace", "middle"))

        runner_rows = []
        for _, row in runners.iterrows():
            points, concerns = _runner_points(row)
            win_prob, quinella_prob, place_prob = _display_probability_triplet(row)
            runner_rows.append(
                {
                    "horseNo": int(_num(row.get("horse_no"), 0)),
                    "horseName": _safe_text(row.get("horse_name")),
                    "aiRank": int(_num(row.get("ai_rank_num"), 99)),
                    "popRank": int(_num(row.get("live_popularity"), _num(row.get("pop_rank_num"), 99))),
                    "odds": round(_num(row.get("live_win_odds"), _num(row.get("market_odds_live_or_final"), _num(row.get("odds_num"), 0.0))), 1),
                    "oddsSource": "当日" if not pd.isna(row.get("live_win_odds")) else "履歴",
                    "placeOddsRange": (
                        f"{_num(row.get('live_place_odds_min'), 0.0):.1f}-{_num(row.get('live_place_odds_max'), 0.0):.1f}"
                        if not pd.isna(row.get("live_place_odds_min")) and not pd.isna(row.get("live_place_odds_max"))
                        else ""
                    ),
                    "bodyWeight": _safe_int_or_blank(row.get("body_weight_live")),
                    "bodyWeightDiff": _safe_int_or_blank(row.get("body_weight_diff_live")),
                    "bodyWeightSource": "当日" if not pd.isna(row.get("body_weight_live")) else "未接続",
                    "winProb": _fmt_prob(win_prob),
                    "quinellaProb": _fmt_prob(quinella_prob),
                    "placeProb": _fmt_prob(place_prob),
                    "quinellaModelScore": _fmt_prob(_num(row.get("quinella_model_score_norm"), 0.0)),
                    "ev": round(_num(row.get("win_ev_proxy"), 0.0), 3),
                    "overlay": round(_num(row.get("market_overlay_score"), 0.0), 3),
                    "front5": _fmt_prob(_num(row.get("projected_front5_prob"), 0.0)),
                    "danger": _fmt_prob(_num(row.get("danger_popular_hybrid_score"), _num(row.get("danger_favorite_score"), 0.0))),
                    "oddsDrop": _fmt_prob(_num(row.get("late_odds_drop_rate"), 0.0)),
                    "oddsDrift": _fmt_prob(_num(row.get("late_odds_drift_rate"), 0.0)),
                    "slowScore": round(_score_for_pace(row, "slow"), 5),
                    "middleScore": round(_score_for_pace(row, "middle"), 5),
                    "fastScore": round(_score_for_pace(row, "fast"), 5),
                    "points": points,
                    "concerns": concerns,
                }
            )

        ticket_rows = []
        for _, row in race_tickets.sort_values(["operation_strength_rank", "pair_quinella_score"], ascending=[False, False]).iterrows():
            ticket_rows.append(
                {
                    "profile": _safe_text(row.get("operation_profile_label")),
                    "type": _safe_text(row.get("ticket_type")),
                    "anchor": _safe_text(row.get("anchor_name")),
                    "partner": _safe_text(row.get("partner_name")),
                    "third": _safe_text(row.get("third_name")),
                    "stake": int(_num(row.get("stake_yen"), 0)),
                    "odds": round(_num(row.get("odds"), _num(row.get("market_odds_live_or_final"), 0.0)), 1),
                    "pairScore": round(_num(row.get("pair_quinella_score"), 0.0), 3),
                    "overlay": round(_num(row.get("market_overlay_score"), 0.0), 3),
                    "pay": int(_num(row.get("wide_pay"), _num(row.get("umaren_pay"), _num(row.get("trio_pay"), _num(row.get("win_pay"), 0.0))))),
                    "livePay": _safe_int_or_blank(row.get("live_pay")),
                    "liveOdds": round(_num(row.get("live_odds"), 0.0), 2) if not pd.isna(row.get("live_odds")) else "",
                    "liveSnapshotAt": _safe_text(row.get("live_snapshot_at")),
                    "hitProb": _fmt_prob(_num(row.get("ticket_hit_prob"), 0.0)),
                    "minOdds": _round_or_blank(row.get("min_acceptable_odds"), 2),
                    "quoteOdds": _round_or_blank(row.get("quote_odds_proxy"), 2),
                    "oddsMargin": _round_or_blank(row.get("min_odds_margin_ratio"), 2),
                    "slipEv": _round_or_blank(row.get("expected_roi_after_slippage"), 2),
                    "probBin": _safe_text(row.get("prob_calibration_bin")),
                    "runtimeAction": _safe_text(row.get("runtime_action")),
                    "runtimeStatus": _safe_text(row.get("runtime_ticket_status")),
                    "runtimeReason": _safe_text(row.get("runtime_reason")),
                    "runtimeStake": int(_num(row.get("runtime_stake_yen"), _num(row.get("stake_yen"), 0))),
                    "runtimeOdds": _round_or_blank(row.get("runtime_odds"), 2),
                    "runtimeMargin": _round_or_blank(row.get("runtime_odds_margin_ratio"), 2),
                    "runtimePaySource": _safe_text(row.get("runtime_pay_source")),
                    "marketProb": _fmt_prob(
                        _num(
                            row.get("runtime_market_implied_prob_takeout_adj"),
                            _num(row.get("umaren_market_prob_takeout_adj"), 0.0),
                        )
                    ),
                    "breakEvenProb": _fmt_prob(_num(row.get("runtime_break_even_prob"), _num(row.get("umaren_break_even_prob"), 0.0))),
                    "marketRatio": _round_or_blank(
                        row.get("runtime_model_market_prob_ratio", row.get("pair_calibrated_market_ratio")),
                        2,
                    ),
                    "marketLogEdge": _round_or_blank(
                        row.get("runtime_model_market_log_edge", row.get("pair_calibrated_market_log_edge")),
                        2,
                    ),
                    "breakEvenDiff": _round_or_blank(row.get("runtime_break_even_prob_diff"), 3),
                    "ctxFront": _round_or_blank(row.get("ticket_front_position_reliability_score"), 2),
                    "ctxDanger": _round_or_blank(row.get("ticket_danger_popular_score"), 2),
                    "ctxBody": _round_or_blank(row.get("ticket_body_age_layoff_score"), 2),
                    "ctxStable": _round_or_blank(row.get("ticket_stable_jockey_buy_timing_score"), 2),
                    "ctxNet": _round_or_blank(row.get("priority_context_net_score"), 2),
                    "liveAlertRisk": _round_or_blank(row.get("live_alert_risk_score"), 2),
                    "liveOddsRisk": _round_or_blank(row.get("live_odds_movement_risk"), 2),
                    "liveBodyRisk": _round_or_blank(row.get("live_body_weight_risk"), 2),
                    "liveBiasRisk": _round_or_blank(row.get("live_same_day_bias_risk"), 2),
                    "liveSafetyReason": _safe_text(row.get("live_safety_reason")),
                    "preLiveSafetyStake": int(_num(row.get("pre_live_safety_stake_yen"), _num(row.get("runtime_stake_yen"), 0))),
                    "operationalMode": _safe_text(row.get("operational_mode")),
                    "buyReason": _safe_text(row.get("buy_reason_summary")),
                    "riskReason": _safe_text(row.get("risk_reason_summary")),
                    "stakeReason": _safe_text(row.get("stake_adjustment_summary")),
                    "decisionLabel": _safe_text(row.get("dashboard_decision_label")),
                }
            )
        runtime_counts = {"buy": 0, "reduce": 0, "wait": 0, "skip": 0, "alert": 0}
        runtime_stake_total = 0
        race_alert_max = 0.0
        race_alert_reasons: list[str] = []
        for t in ticket_rows:
            action_key = str(t.get("runtimeAction") or "")
            if action_key in ("BUY", "BUY_CONTEXT_BOOST"):
                runtime_counts["buy"] += 1
            elif action_key in ("REDUCE", "REDUCE_ALERT"):
                runtime_counts["reduce"] += 1
            elif action_key in ("WAIT", "WATCH_ALERT"):
                runtime_counts["wait"] += 1
            elif action_key in ("SKIP", "SKIP_ALERT"):
                runtime_counts["skip"] += 1
            if action_key in ("REDUCE_ALERT", "WATCH_ALERT", "SKIP_ALERT"):
                runtime_counts["alert"] += 1
            runtime_stake_total += int(_num(t.get("runtimeStake"), 0)) if action_key in ("BUY", "BUY_CONTEXT_BOOST", "REDUCE", "REDUCE_ALERT", "WATCH_ALERT") else 0
            race_alert_max = max(race_alert_max, _num(t.get("liveAlertRisk"), 0.0))
            if t.get("liveSafetyReason") and t.get("liveSafetyReason") != "live_safety_ok":
                race_alert_reasons.extend([p for p in str(t.get("liveSafetyReason")).split("|") if p])

        races.append(
            {
                "raceId": race_id,
                "title": f"{_safe_text(first.get('venue'))} {_safe_text(first.get('Ｒ'))}R {_safe_text(first.get('レース名'))}".strip(),
                "meta": {
                    "date": race_date,
                    "dateLabel": _display_race_date(race_date),
                    "venue": _safe_text(first.get("venue")),
                    "raceNo": int(_num(first.get("Ｒ"), 0)),
                    "surface": _safe_text(first.get("surface")),
                    "distance": int(_num(first.get("distance"), 0)),
                    "className": _safe_text(first.get("クラス名")),
                    "field": int(_num(first.get("出走頭数"), _num(first.get("頭数"), 0))),
                },
                "profile": profile,
                "pace": pace,
                "rpci": round(_num(first.get("RPCI"), 0.0), 1),
                "difficulty": round(_num(race_tickets.iloc[0].get("race_difficulty_score") if not race_tickets.empty else first.get("race_difficulty_model_score"), 0.0), 3),
                "paceCollapse": round(_num(race_tickets.iloc[0].get("race_pace_collapse") if not race_tickets.empty else first.get("race_pace_collapse_risk"), 0.0), 3),
                "volatility": volatility,
                "volatilityClass": volatility_class,
                "runtime": {
                    "buy": runtime_counts["buy"],
                    "reduce": runtime_counts["reduce"],
                    "wait": runtime_counts["wait"],
                    "skip": runtime_counts["skip"],
                    "alert": runtime_counts["alert"],
                    "stake": runtime_stake_total,
                },
                "liveAlert": {
                    "maxRisk": round(race_alert_max, 2),
                    "count": runtime_counts["alert"],
                    "reasons": sorted(set(race_alert_reasons))[:4],
                },
                "runners": sorted(runner_rows, key=lambda x: x["aiRank"]),
                "tickets": ticket_rows,
            }
        )
        action, action_class, action_note = _race_action(profile, volatility, races[-1]["difficulty"], races[-1]["paceCollapse"])
        races[-1]["action"] = action
        races[-1]["actionClass"] = action_class
        races[-1]["actionNote"] = action_note

    dates = [{"value": d, "label": _display_race_date(d)} for d in sorted({r["meta"]["date"] for r in races if r["meta"].get("date")})]
    venues = sorted({r["meta"]["venue"] for r in races if r["meta"].get("venue")})
    return {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataStatus": {
                "livePairOdds": bool(live_pair_odds is not None and not live_pair_odds.empty),
                "liveSingleOdds": bool(live_single_odds is not None and not live_single_odds.empty),
                "bodyWeight": bool(body_weight is not None and not body_weight.empty),
            },
        "dates": dates,
        "venues": venues,
        "races": races,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Keiba AI Dashboard</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #697586;
      --line: #d7dde6;
      --blue: #2563eb;
      --green: #17834f;
      --amber: #b76e00;
      --red: #b42318;
      --ink: #101828;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Segoe UI", "Yu Gothic UI", Meiryo, sans-serif; background: var(--bg); color: var(--text); }
    header { height: 56px; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 5; }
    header h1 { font-size: 18px; margin: 0; letter-spacing: 0; }
    .app { display: grid; grid-template-columns: 300px 1fr; min-height: calc(100vh - 56px); }
    aside { border-right: 1px solid var(--line); background: #fbfcfe; padding: 14px; overflow: auto; }
    main { padding: 18px; overflow: auto; }
    .control { margin-bottom: 12px; }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
    select, input { width: 100%; height: 36px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); padding: 0 10px; color: var(--text); }
    .status-panel { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: var(--panel); margin-bottom: 12px; }
    .status-row { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; padding: 3px 0; }
    .status-ok { color: var(--green); font-weight: 700; }
    .status-wait { color: var(--amber); font-weight: 700; }
    .refresh-row { display: grid; grid-template-columns: 1fr 92px; gap: 8px; align-items: center; margin-top: 8px; }
    .refresh-row label { display: flex; align-items: center; gap: 6px; margin: 0; }
    .refresh-row input { width: 16px; height: 16px; }
    .refresh-row select { height: 32px; padding: 0 6px; }
    .refresh-button { width: 100%; height: 34px; border: 1px solid var(--line); border-radius: 6px; background: #eef4ff; color: var(--blue); font-weight: 800; cursor: pointer; margin-top: 8px; }
    .race-list { display: grid; gap: 8px; margin-top: 12px; }
    .race-group { margin: 12px 0 2px; padding: 6px 2px 2px; font-size: 12px; color: var(--muted); font-weight: 800; border-top: 1px solid var(--line); }
    .race-button { border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 10px; text-align: left; cursor: pointer; }
    .race-button.active { border-color: var(--blue); box-shadow: inset 3px 0 0 var(--blue); }
    .race-button .name { font-weight: 700; font-size: 13px; }
    .race-button .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .toolbar { display: grid; grid-template-columns: 1fr 180px 180px; gap: 12px; margin-bottom: 14px; }
    .action-panel { display: grid; grid-template-columns: 160px 1fr; gap: 14px; align-items: center; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 12px 14px; margin-bottom: 14px; }
    .action-label { font-size: 24px; font-weight: 900; }
    .action-buy { color: var(--green); }
    .action-lean { color: var(--blue); }
    .action-caution { color: var(--amber); }
    .action-light { color: #475467; }
    .action-skip { color: var(--red); }
    .action-note { color: var(--muted); font-size: 13px; }
    .summary { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; margin-bottom: 14px; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; min-height: 74px; }
    .metric .k { font-size: 12px; color: var(--muted); }
    .metric .v { font-size: 22px; font-weight: 800; margin-top: 6px; color: var(--ink); }
    .vol-high { color: var(--red); }
    .vol-mid { color: var(--amber); }
    .vol-low { color: var(--green); }
    .vol-solid { color: var(--blue); }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 14px; overflow: hidden; }
    section h2 { margin: 0; padding: 12px 14px; font-size: 15px; border-bottom: 1px solid var(--line); background: #fbfcfe; }
    table { width: 100%; min-width: 1440px; border-collapse: collapse; table-layout: fixed; }
    th, td { border-bottom: 1px solid var(--line); padding: 10px 8px; vertical-align: middle; font-size: 13px; line-height: 1.35; }
    th { text-align: left; color: var(--muted); font-weight: 700; background: #fbfcfe; white-space: nowrap; }
    tr:last-child td { border-bottom: 0; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .rank { width: 46px; font-weight: 800; }
    .pin-rank, .pin-horse { position: sticky; background: #fff; z-index: 1; }
    th.pin-rank, th.pin-horse { background: #fbfcfe; z-index: 2; }
    .pin-rank { left: 0; }
    .pin-horse { left: 46px; box-shadow: 1px 0 0 var(--line); }
    .horse { font-weight: 700; }
    .chips { display: flex; flex-wrap: wrap; gap: 4px; }
    .chip { display: inline-flex; align-items: center; height: 22px; padding: 0 7px; border-radius: 999px; font-size: 12px; border: 1px solid var(--line); background: #f7f9fc; white-space: nowrap; }
    .chip.good { border-color: #badbcc; color: var(--green); background: #f0fff7; }
    .chip.bad { border-color: #f1b9b5; color: var(--red); background: #fff5f4; }
    .move-up { color: var(--green); font-weight: 800; }
    .move-down { color: var(--red); font-weight: 800; }
    .move-flat { color: var(--muted); }
    .runner-cards { display: none; }
    .runner-card { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fff; }
    .runner-card .runner-card-top { display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; }
    .runner-card .runner-card-name { font-weight: 900; color: var(--ink); }
    .runner-card .runner-card-rank { font-size: 12px; color: var(--muted); white-space: nowrap; }
    .runner-card .runner-card-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin: 10px 0; }
    .runner-card .runner-card-stat { border: 1px solid var(--line); border-radius: 6px; padding: 6px; min-width: 0; }
    .runner-card .runner-card-stat .k { color: var(--muted); font-size: 11px; }
    .runner-card .runner-card-stat .v { font-weight: 800; font-size: 14px; margin-top: 2px; }
    .ticket-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; padding: 12px; }
    .ticket { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fff; }
    .ticket .top { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
    .ticket .odds-check { margin-top: 6px; font-size: 12px; color: var(--muted); }
    .ticket .odds-ok { color: var(--green); font-weight: 800; }
    .ticket .odds-warn { color: var(--red); font-weight: 800; }
    .ticket .runtime-line { margin-top: 7px; font-size: 12px; font-weight: 800; }
    .ticket .runtime-buy { color: var(--green); }
    .ticket .runtime-reduce { color: var(--amber); }
    .ticket .runtime-wait { color: var(--blue); }
    .ticket .runtime-skip { color: var(--red); }
    .ticket .context-line { margin-top: 6px; font-size: 12px; color: var(--muted); }
    .ticket .alert-line { margin-top: 6px; font-size: 12px; font-weight: 800; color: var(--red); }
    .ticket .alert-line.watch { color: var(--amber); }
    .alert-pill { display:inline-flex; align-items:center; gap:4px; border-radius:999px; padding:3px 8px; font-size:12px; font-weight:800; background:#fff7ed; color:#c2410c; border:1px solid #fed7aa; }
    .alert-pill.danger { background:#fef2f2; color:#b91c1c; border-color:#fecaca; }
    .badge { font-size: 12px; font-weight: 700; padding: 3px 7px; border-radius: 999px; background: #eef4ff; color: var(--blue); }
    .badge.strong { background: #fff1f0; color: var(--red); }
    .badge.standard { background: #fff8e6; color: var(--amber); }
    .badge.broad { background: #eefbf4; color: var(--green); }
    .muted { color: var(--muted); }
    .empty { padding: 28px; color: var(--muted); text-align: center; }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); max-height: 300px; }
      .toolbar { grid-template-columns: 1fr; }
      .action-panel { grid-template-columns: 1fr; }
      .summary { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 700px) {
      header { height: auto; min-height: 50px; padding: 10px 12px; }
      header h1 { font-size: 16px; }
      aside { max-height: 42vh; padding: 10px; }
      main { padding: 10px; }
      .race-button { padding: 9px; }
      .race-button .sub { line-height: 1.35; }
      .toolbar { gap: 8px; margin-bottom: 10px; }
      #raceTitle { font-size: 18px !important; line-height: 1.25; }
      .action-panel { padding: 10px; margin-bottom: 10px; gap: 8px; }
      .action-label { font-size: 20px; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
      .metric { min-height: 58px; padding: 9px; }
      .metric .v { font-size: 17px; }
      section { margin-bottom: 10px; }
      section h2 { padding: 10px; font-size: 14px; }
      .runner-table-wrap { display: none; }
      .runner-cards { display: grid; gap: 8px; padding: 10px; }
      .runner-card .runner-card-stats { grid-template-columns: repeat(2, 1fr); }
      .ticket-grid { grid-template-columns: 1fr; gap: 8px; padding: 10px; }
      .ticket { padding: 10px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Keiba AI Dashboard</h1>
    <div class="muted" id="raceCount"></div>
  </header>
  <div class="app">
    <aside>
      <div class="status-panel">
        <div class="status-row"><span>最終生成</span><span id="generatedAt">-</span></div>
        <div class="status-row"><span>単勝オッズ</span><span id="singleOddsStatus">-</span></div>
        <div class="status-row"><span>ペアオッズ</span><span id="liveOddsStatus">-</span></div>
        <div class="status-row"><span>当日馬体重</span><span id="bodyWeightStatus">-</span></div>
        <div class="refresh-row">
          <label for="autoRefresh"><input id="autoRefresh" type="checkbox">自動更新</label>
          <select id="refreshSeconds">
            <option value="30">30秒</option>
            <option value="60" selected>60秒</option>
            <option value="180">3分</option>
          </select>
        </div>
        <button class="refresh-button" id="manualRefresh" type="button">画面を更新</button>
      </div>
      <div class="control">
        <label for="dateFilter">開催日</label>
        <select id="dateFilter"></select>
      </div>
      <div class="control">
        <label for="venueFilter">開催場</label>
        <select id="venueFilter"></select>
      </div>
      <div class="control">
        <label for="raceSearch">レース検索</label>
        <input id="raceSearch" placeholder="1R・東京・レース名・ID">
      </div>
      <div class="control">
        <label for="profileFilter">表示プロファイル</label>
        <select id="profileFilter">
          <option value="all">すべて</option>
          <option value="強気">強気</option>
          <option value="標準">標準</option>
          <option value="広め">広め</option>
        </select>
      </div>
      <div class="race-list" id="raceList"></div>
    </aside>
    <main>
      <div class="toolbar">
        <div>
          <div class="muted" id="raceMetaLine">選択レース</div>
          <h2 id="raceTitle" style="margin:4px 0 0;font-size:22px;"></h2>
        </div>
        <div class="control">
          <label for="paceSelect">展開シナリオ</label>
          <select id="paceSelect">
            <option value="middle">ミドル</option>
            <option value="slow">スロー</option>
            <option value="fast">ハイ</option>
          </select>
        </div>
        <div class="control">
          <label for="sortSelect">並び替え</label>
          <select id="sortSelect">
            <option value="scenario">展開別AI順位</option>
            <option value="ai">元AI順位</option>
            <option value="ev">期待値</option>
            <option value="odds">オッズ</option>
          </select>
        </div>
      </div>
      <div class="action-panel" id="actionPanel">
        <div>
          <div class="muted">レース判断</div>
          <div class="action-label" id="actionLabel">-</div>
        </div>
        <div>
          <div id="actionNote" class="action-note">-</div>
          <div class="muted" id="actionSub" style="margin-top:4px;"></div>
        </div>
      </div>
      <div class="summary" id="summary"></div>
      <section>
        <h2>出馬表・評価</h2>
        <div class="runner-table-wrap" style="overflow:auto;">
          <table>
            <thead>
              <tr>
                <th class="rank pin-rank">AI</th>
                <th class="pin-horse" style="width:190px;">馬</th>
                <th class="num">人気</th>
                <th class="num">オッズ</th>
                <th class="num">馬体重</th>
                <th class="num">勝率</th>
                <th class="num">連対</th>
                <th class="num">複勝</th>
                <th class="num">期待値</th>
                <th class="num">展開</th>
                <th class="num">直前</th>
                <th style="width:250px;">評価ポイント</th>
                <th style="width:250px;">懸念点</th>
              </tr>
            </thead>
            <tbody id="runnerBody"></tbody>
          </table>
        </div>
        <div class="runner-cards" id="runnerCards"></div>
      </section>
      <section>
        <h2>買い目候補</h2>
        <div class="ticket-grid" id="ticketGrid"></div>
      </section>
    </main>
  </div>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    let races = payload.races || [];
    let selectedRaceId = races[0]?.raceId || null;
    const raceList = document.getElementById('raceList');
    const dateFilter = document.getElementById('dateFilter');
    const venueFilter = document.getElementById('venueFilter');
    const raceSearch = document.getElementById('raceSearch');
    const profileFilter = document.getElementById('profileFilter');
    const paceSelect = document.getElementById('paceSelect');
    const sortSelect = document.getElementById('sortSelect');
    const autoRefresh = document.getElementById('autoRefresh');
    const refreshSeconds = document.getElementById('refreshSeconds');
    const manualRefresh = document.getElementById('manualRefresh');
    let refreshTimer = null;

    function initFilters() {
      const dates = payload.dates || [];
      dateFilter.innerHTML = '<option value="all">すべての日付</option>' + dates.map(d => `<option value="${d.value}">${d.label}</option>`).join('');
      venueFilter.innerHTML = '<option value="all">すべての開催場</option>' + (payload.venues || []).map(v => `<option value="${v}">${v}</option>`).join('');
      document.getElementById('generatedAt').textContent = payload.generatedAt || '-';
      const oddsReady = payload.dataStatus?.livePairOdds;
      const singleOddsReady = payload.dataStatus?.liveSingleOdds;
      const bodyReady = payload.dataStatus?.bodyWeight;
      document.getElementById('singleOddsStatus').innerHTML = singleOddsReady ? '<span class="status-ok">接続済み</span>' : '<span class="status-wait">履歴/未接続</span>';
      document.getElementById('liveOddsStatus').innerHTML = oddsReady ? '<span class="status-ok">接続済み</span>' : '<span class="status-wait">履歴/未接続</span>';
      document.getElementById('bodyWeightStatus').innerHTML = bodyReady ? '<span class="status-ok">接続済み</span>' : '<span class="status-wait">未接続</span>';
    }

    function setAutoRefresh() {
      if (refreshTimer) clearInterval(refreshTimer);
      refreshTimer = null;
      if (autoRefresh.checked) {
        refreshTimer = setInterval(() => window.location.reload(), Number(refreshSeconds.value) * 1000);
      }
    }

    function profileRank(profile) {
      return profile === '強気' ? 3 : profile === '標準' ? 2 : profile === '広め' ? 1 : 0;
    }
    function raceOrder(a, b) {
      const dateCompare = String(a.meta?.date || '').localeCompare(String(b.meta?.date || ''));
      if (dateCompare !== 0) return dateCompare;
      const venueCompare = String(a.meta?.venue || '').localeCompare(String(b.meta?.venue || ''), 'ja');
      if (venueCompare !== 0) return venueCompare;
      return Number(a.meta?.raceNo || 0) - Number(b.meta?.raceNo || 0);
    }
    function currentRaces() {
      const q = raceSearch.value.trim().toLowerCase();
      const pf = profileFilter.value;
      const date = dateFilter.value;
      const venue = venueFilter.value;
      return races.filter(r => {
        const hay = `${r.raceId} ${r.title} ${r.meta?.venue || ''} ${r.meta?.dateLabel || ''}`.toLowerCase();
        return (!q || hay.includes(q))
          && (pf === 'all' || r.profile === pf)
          && (date === 'all' || r.meta?.date === date)
          && (venue === 'all' || r.meta?.venue === venue);
      });
    }
    function renderRaceList() {
      const list = currentRaces().sort(raceOrder);
      document.getElementById('raceCount').textContent = `${list.length} races`;
      raceList.innerHTML = '';
      if (!list.find(r => r.raceId === selectedRaceId) && list[0]) selectedRaceId = list[0].raceId;
      let currentGroup = '';
      list.forEach(r => {
        const group = `${r.meta?.dateLabel || ''} ${r.meta?.venue || ''}`;
        if (group !== currentGroup) {
          currentGroup = group;
          const label = document.createElement('div');
          label.className = 'race-group';
          label.textContent = group;
          raceList.appendChild(label);
        }
        const btn = document.createElement('button');
        btn.className = `race-button ${r.raceId === selectedRaceId ? 'active' : ''}`;
        const runtime = r.runtime || {};
        const alertSub = r.liveAlert?.count ? ` / 警戒${r.liveAlert.count}` : '';
        const runtimeSub = runtime.buy || runtime.reduce || runtime.wait || runtime.alert
          ? ` / 買い${runtime.buy || 0} 減${runtime.reduce || 0} 待${runtime.wait || 0}${alertSub} / ${runtime.stake || 0}円`
          : '';
        btn.innerHTML = `<div class="name">${r.title}</div><div class="sub">${r.meta?.dateLabel || ''} / ${r.profile} / ${r.volatility} / RPCI ${r.rpci}${runtimeSub}</div>`;
        btn.onclick = () => { selectedRaceId = r.raceId; render(); };
        raceList.appendChild(btn);
      });
    }
    function scenarioScore(row, pace) {
      if (pace === 'slow') return row.slowScore;
      if (pace === 'fast') return row.fastScore;
      return row.middleScore;
    }
    function renderSummary(r) {
      const items = [
        ['勝負度', r.profile, ''],
        ['荒れ度', r.volatility, r.volatilityClass],
        ['想定展開', r.pace, ''],
        ['RPCI', r.rpci, ''],
        ['難易度', r.difficulty, ''],
        ['崩壊リスク', r.paceCollapse, '']
      ];
      items.push(['実運用', `買${r.runtime?.buy || 0}/減${r.runtime?.reduce || 0}/待${r.runtime?.wait || 0}`, '']);
      items.push(['予算', `${r.runtime?.stake || 0}円`, '']);
      if (r.liveAlert?.count) items.push(['警戒', `${r.liveAlert.count}件 / max ${r.liveAlert.maxRisk}`, Number(r.liveAlert.maxRisk || 0) >= 0.64 ? 'vol-high' : 'vol-mid']);
      document.getElementById('summary').innerHTML = items.map(([k,v,cls]) => `<div class="metric"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`).join('');
    }
    function renderAction(r) {
      const label = document.getElementById('actionLabel');
      label.className = `action-label ${r.actionClass || ''}`;
      label.textContent = r.action || '-';
      document.getElementById('actionNote').textContent = r.actionNote || '';
      document.getElementById('actionSub').textContent = `${r.profile} / ${r.volatility} / 難易度 ${r.difficulty} / 崩壊 ${r.paceCollapse}`;
    }
    function movementLabel(row) {
      const diff = Number(row.aiRank || 99) - Number(row.scenarioRank || 99);
      if (diff >= 2) return `<span class="move-up">↑${diff}</span>`;
      if (diff <= -2) return `<span class="move-down">↓${Math.abs(diff)}</span>`;
      return '<span class="move-flat">→</span>';
    }
    function oddsAlert(row) {
      if (Number(row.oddsDrop || 0) >= 25) return `<span class="move-up">急落 ${row.oddsDrop}%</span>`;
      if (Number(row.oddsDrift || 0) >= 25) return `<span class="move-down">悪化 ${row.oddsDrift}%</span>`;
      return '<span class="move-flat">-</span>';
    }
    function renderRunners(r) {
      const pace = paceSelect.value;
      const sort = sortSelect.value;
      const rows = r.runners.map(x => ({...x, scenarioScore: scenarioScore(x, pace)}));
      rows.sort((a,b) => {
        if (sort === 'ev') return b.ev - a.ev;
        if (sort === 'odds') return a.odds - b.odds;
        if (sort === 'ai') return a.aiRank - b.aiRank;
        return b.scenarioScore - a.scenarioScore;
      });
      const ranked = rows.map((x,i) => ({...x, scenarioRank: i + 1}));
      document.getElementById('runnerBody').innerHTML = ranked.map(x => `
        <tr>
          <td class="rank pin-rank">${sort === 'scenario' ? x.scenarioRank : x.aiRank}</td>
          <td class="pin-horse"><div class="horse">${x.horseNo}. ${x.horseName}</div><div class="muted">元AI ${x.aiRank}位 / 前目 ${x.front5}%</div></td>
          <td class="num">${x.popRank}</td>
          <td class="num">${x.odds}<div class="muted">${x.oddsSource || ''}${x.placeOddsRange ? ` / 複 ${x.placeOddsRange}` : ''}</div></td>
          <td class="num">${x.bodyWeight ? `${x.bodyWeight}${x.bodyWeightDiff !== '' ? `(${x.bodyWeightDiff > 0 ? '+' : ''}${x.bodyWeightDiff})` : ''}` : '<span class="muted">未</span>'}</td>
          <td class="num">${x.winProb}%</td>
          <td class="num">${x.quinellaProb}%</td>
          <td class="num">${x.placeProb}%</td>
          <td class="num">${x.ev}</td>
          <td class="num">${movementLabel(x)}</td>
          <td class="num">${oddsAlert(x)}</td>
          <td><div class="chips">${x.points.map(p => `<span class="chip good">${p}</span>`).join('')}</div></td>
          <td><div class="chips">${x.concerns.map(p => `<span class="chip bad">${p}</span>`).join('')}</div></td>
        </tr>
      `).join('');
      const cards = document.getElementById('runnerCards');
      if (cards) {
        cards.innerHTML = ranked.map(x => `
          <div class="runner-card">
            <div class="runner-card-top">
              <div>
                <div class="runner-card-name">${x.horseNo}. ${x.horseName}</div>
                <div class="muted">AI ${x.aiRank} / scenario ${x.scenarioRank} / 人気 ${x.popRank}</div>
              </div>
              <div class="runner-card-rank">odds ${x.odds}</div>
            </div>
            <div class="runner-card-stats">
              <div class="runner-card-stat"><div class="k">勝率</div><div class="v">${x.winProb}%</div></div>
              <div class="runner-card-stat"><div class="k">連対</div><div class="v">${x.quinellaProb}%</div></div>
              <div class="runner-card-stat"><div class="k">複勝</div><div class="v">${x.placeProb}%</div></div>
              <div class="runner-card-stat"><div class="k">期待値</div><div class="v">${x.ev}</div></div>
            </div>
            <div class="chips">${x.points.map(p => `<span class="chip good">${p}</span>`).join('')}</div>
            <div class="chips" style="margin-top:6px;">${x.concerns.map(p => `<span class="chip bad">${p}</span>`).join('')}</div>
          </div>
        `).join('');
      }
    }
    function renderTickets(r) {
      const tickets = r.tickets || [];
      const grid = document.getElementById('ticketGrid');
      if (!tickets.length) {
        grid.innerHTML = '<div class="empty">買い目候補なし</div>';
        return;
      }
      grid.innerHTML = tickets.map(t => {
        const cls = t.profile === '強気' ? 'strong' : t.profile === '標準' ? 'standard' : 'broad';
        const oddsText = t.type === 'win' && t.odds ? ` / odds ${t.odds}` : '';
        const hasMinOdds = t.minOdds !== '' && t.minOdds !== null && t.minOdds !== undefined;
        const marginClass = Number(t.oddsMargin || 0) >= 1 ? 'odds-ok' : 'odds-warn';
        const oddsCheck = hasMinOdds
          ? `<div class="odds-check">成立 ${t.hitProb}% / 最低 ${t.minOdds} / 代理 ${t.quoteOdds} / <span class="${marginClass}">余裕 ${t.oddsMargin}x</span>${t.slipEv ? ` / slipEV ${t.slipEv}` : ''}</div>`
          : '';
        const action = t.runtimeAction || '';
        const actionClass = ['BUY','BUY_CONTEXT_BOOST'].includes(action) ? 'runtime-buy' : ['REDUCE','REDUCE_ALERT'].includes(action) ? 'runtime-reduce' : ['WAIT','WATCH_ALERT'].includes(action) ? 'runtime-wait' : 'runtime-skip';
        const runtimeText = action
          ? `<div class="runtime-line ${actionClass}">${t.runtimeStatus || action} / runtime stake ${t.runtimeStake} / odds ${t.runtimeOdds || '-'} / margin ${t.runtimeMargin || '-'}x / ${t.runtimePaySource || '-'}</div>`
          : '';
        const hasPairEdge = t.marketRatio !== '' && t.marketRatio !== null && t.marketRatio !== undefined;
        const pairEdgeClass = Number(t.marketRatio || 0) >= 1.5 ? 'odds-ok' : Number(t.marketRatio || 0) >= 1 ? 'odds-warn' : 'runtime-skip';
        const pairEdgeText = hasPairEdge
          ? `<div class="odds-check">市場確率 ${t.marketProb}% / BE ${t.breakEvenProb}% / <span class="${pairEdgeClass}">市場比 ${t.marketRatio}x</span>${t.marketLogEdge ? ` / logEdge ${t.marketLogEdge}` : ''}${t.breakEvenDiff ? ` / BE差 ${t.breakEvenDiff}` : ''}</div>`
          : '';
        const hasContext = t.ctxFront !== '' || t.ctxDanger !== '' || t.ctxBody !== '' || t.ctxStable !== '';
        const contextText = hasContext
          ? `<div class="context-line">前位置 ${t.ctxFront || '-'} / 危険 ${t.ctxDanger || '-'} / 馬体休 ${t.ctxBody || '-'} / 厩騎 ${t.ctxStable || '-'} / 総合 ${t.ctxNet || '-'}</div>`
          : '';
        const explainText = (t.buyReason || t.riskReason || t.stakeReason)
          ? `<div class="context-line">理由 ${t.buyReason || '-'} / 警戒 ${t.riskReason || '-'} / 金額 ${t.stakeReason || '-'}</div>`
          : '';
        const hasLiveAlert = t.liveAlertRisk !== '' && Number(t.liveAlertRisk || 0) >= 0.40;
        const alertClass = Number(t.liveAlertRisk || 0) >= 0.64 ? '' : ' watch';
        const alertText = hasLiveAlert
          ? `<div class="alert-line${alertClass}">警戒 ${t.liveAlertRisk} / odds ${t.liveOddsRisk || '-'} / 馬体 ${t.liveBodyRisk || '-'} / 当日Bias ${t.liveBiasRisk || '-'}${t.liveSafetyReason && t.liveSafetyReason !== 'live_safety_ok' ? ` / ${t.liveSafetyReason}` : ''}</div>`
          : '';
        return `<div class="ticket">
          <div class="top"><span class="badge ${cls}">${t.operationalMode || t.profile}</span><span class="muted">${t.type}</span></div>
          <div><b>${t.anchor}</b>${t.partner ? ` - <b>${t.partner}</b>` : ''}${t.third ? ` - <b>${t.third}</b>` : ''}</div>
          <div class="muted">stake ${t.stake}${oddsText} / pair ${t.pairScore} / overlay ${t.overlay} / pay ${t.livePay || t.pay}${t.livePay ? ' live' : ''}</div>
          ${oddsCheck}
          ${runtimeText}
          ${pairEdgeText}
          ${contextText}
          ${explainText}
          ${alertText}
        </div>`;
      }).join('');
    }
    function render() {
      renderRaceList();
      const r = races.find(x => x.raceId === selectedRaceId) || currentRaces()[0];
      if (!r) return;
      document.getElementById('raceTitle').textContent = r.title;
      document.getElementById('raceMetaLine').textContent = `${r.meta?.dateLabel || ''} / ${r.meta?.surface || ''}${r.meta?.distance || ''}m / ${r.meta?.className || ''} / ${r.meta?.field || '-'}頭`;
      paceSelect.value = r.pace === 'スロー' ? 'slow' : r.pace === 'ハイ' ? 'fast' : paceSelect.value;
      renderAction(r);
      renderSummary(r);
      renderRunners(r);
      renderTickets(r);
    }
    raceSearch.oninput = render;
    dateFilter.onchange = render;
    venueFilter.onchange = render;
    profileFilter.onchange = render;
    autoRefresh.onchange = setAutoRefresh;
    refreshSeconds.onchange = setAutoRefresh;
    manualRefresh.onclick = () => window.location.reload();
    paceSelect.onchange = () => { const r = races.find(x => x.raceId === selectedRaceId); if (r) renderRunners(r); };
    sortSelect.onchange = () => { const r = races.find(x => x.raceId === selectedRaceId); if (r) renderRunners(r); };
    initFilters();
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static Keiba AI dashboard HTML from scored runners and operational ticket profiles.")
    parser.add_argument("--scored-csv", default="outputs/analysis/risk_models_v1/investment_features_with_risk_models.csv")
    parser.add_argument("--tickets-csv", default="outputs/analysis/operational_ticket_profiles_v1/ticket_profiles.csv")
    parser.add_argument("--live-pair-odds-csv", default="data/processed/live_odds/realtime_pair_odds_latest.csv")
    parser.add_argument("--live-single-odds-csv", default="data/processed/live_odds/realtime_single_odds_latest.csv")
    parser.add_argument("--body-weight-csv", default="")
    parser.add_argument("--output-html", default="outputs/ui/keiba_dashboard.html")
    parser.add_argument("--max-races", type=int, default=120)
    args = parser.parse_args()

    scored = pd.read_csv(project_path(args.scored_csv), dtype={"race_id": str}, low_memory=False)
    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    live_pair_odds_path = project_path(args.live_pair_odds_csv)
    live_pair_odds = pd.read_csv(live_pair_odds_path, dtype={"race_id": str}, low_memory=False) if live_pair_odds_path.exists() else None
    live_single_odds_path = project_path(args.live_single_odds_csv)
    live_single_odds = pd.read_csv(live_single_odds_path, dtype={"race_id": str}, low_memory=False) if live_single_odds_path.exists() else None
    body_weight = None
    if args.body_weight_csv:
        body_weight_path = project_path(args.body_weight_csv)
        body_weight = pd.read_csv(body_weight_path, dtype={"race_id": str}, low_memory=False) if body_weight_path.exists() else None

    payload = _build_payload(
        scored,
        tickets,
        args.max_races,
        live_pair_odds=live_pair_odds,
        live_single_odds=live_single_odds,
        body_weight=body_weight,
    )
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    output = project_path(args.output_html)
    ensure_dir(output.parent)
    output.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_html": str(output),
                "races": len(payload["races"]),
                "dates": payload["dates"],
                "venues": payload["venues"],
                "live_pair_odds_loaded": payload["dataStatus"]["livePairOdds"],
                "live_single_odds_loaded": payload["dataStatus"]["liveSingleOdds"],
                "body_weight_loaded": payload["dataStatus"]["bodyWeight"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
