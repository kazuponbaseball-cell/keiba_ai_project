from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LINE_PUSH_ENDPOINT = "https://api.line.me/v2/bot/message/push"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv_safe(path: str | Path) -> pd.DataFrame:
    p = project_path(path)
    if not p.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(p, encoding=enc, low_memory=False, dtype=str)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(p, low_memory=False, dtype=str)


def text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if s else default


def number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        s = str(value).replace(",", "").strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def normalize_ticket_type_for_lap(value: Any) -> str:
    raw = text(value).lower()
    if raw in {"quinella", "umaren"}:
        return "umaren"
    if raw == "wide":
        return "wide"
    return raw


def horse_lap_decomp_fields(row: pd.Series) -> dict[str, Any]:
    mode = text(row.get("race_lap_mode_label"))
    type_pair = text(row.get("lap_type_pair_label"))
    match_bucket = text(row.get("lap_match_bucket_label"))
    fit = number(row.get("pair_lap_fit_min_eval"), 0.0)
    cosine = number(row.get("pair_lap_cosine_min_eval"), 0.0)
    score = number(row.get("horse_lap_decomp_score"), 0.0)
    mismatch = number(row.get("pair_lap_mismatch_popular_max_eval"), 0.0)
    label = ""
    note = ""
    if mode == "\u524d\u534a\u8ca0\u8377\u578b" and type_pair in {
        "\u524d\u534a\u8ca0\u8377\u578b + \u77ac\u767a\u578b",
        "\u77ac\u767a\u578b + \u524d\u534a\u8ca0\u8377\u578b",
    }:
        label = "horse_lap_high_pressure_instant_pair"
        note = "\u524d\u534a\u8ca0\u8377\u60f3\u5b9a\u3067\u8ca0\u8377\u578b\u3068\u77ac\u767a\u578b\u306e\u7d44\u307f\u5408\u308f\u305b"
    elif match_bucket == "2\u982d\u3068\u3082\u60f3\u5b9a\u30e9\u30c3\u30d7\u4e00\u81f4" and fit >= 0.70:
        label = "horse_lap_both_match"
        note = "\u4e21\u99ac\u3068\u3082\u60f3\u5b9a\u30e9\u30c3\u30d7\u306b\u5408\u3046"
    elif fit >= 0.75 and cosine >= 0.70:
        label = "horse_lap_pair_fit_good"
        note = "\u30e9\u30c3\u30d7\u5206\u89e3\u3067\u30da\u30a2\u76f8\u6027\u304c\u826f\u3044"
    elif (
        (mode == "\u30b9\u30ed\u30fc\u77ac\u767a\u5bc4\u308a" and type_pair in {
            "\u30b9\u30ed\u30fc\u77ac\u767a\u5bc4\u308a + \u6301\u7d9a\u578b",
            "\u6301\u7d9a\u578b + \u30b9\u30ed\u30fc\u77ac\u767a\u5bc4\u308a",
        })
        or (mode == "\u524d\u534a\u8ca0\u8377\u578b" and type_pair in {
            "\u6301\u7d9a\u578b + \u6301\u7d9a\u578b",
            "\u30ed\u30f3\u30b0\u30b9\u30d1\u30fc\u30c8\u578b + \u6301\u7d9a\u578b",
            "\u6301\u7d9a\u578b + \u30ed\u30f3\u30b0\u30b9\u30d1\u30fc\u30c8\u578b",
        })
        or mismatch >= 0.35
    ):
        label = "horse_lap_pair_caution"
        note = "\u30ec\u30fc\u30b9\u8cea\u3068\u30da\u30a2\u306e\u30e9\u30c3\u30d7\u5f79\u5272\u304c\u5668\u7528\u3057\u306b\u304f\u3044"
    return {
        "horse_lap_decomp_runtime_label": label,
        "horse_lap_decomp_runtime_note": note,
        "horse_lap_decomp_runtime_score": max(fit, cosine, score),
    }


def enrich_horse_lap_decomp(frame: pd.DataFrame, overlay_csv: str) -> pd.DataFrame:
    if frame.empty or not overlay_csv:
        return frame
    overlay = read_csv_safe(overlay_csv)
    required = {"race_id", "ticket_type", "anchor_no", "partner_no"}
    if overlay.empty or not required.issubset(overlay.columns):
        return frame
    lookup: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for _, row in overlay.iterrows():
        race_id = text(row.get("race_id"))
        ticket_type = normalize_ticket_type_for_lap(row.get("ticket_type"))
        a_no = int(number(row.get("anchor_no"), 0))
        b_no = int(number(row.get("partner_no"), 0))
        if not race_id or not ticket_type or not a_no or not b_no:
            continue
        lo, hi = sorted([a_no, b_no])
        lookup[(race_id, ticket_type, lo, hi)] = horse_lap_decomp_fields(row)
    out = frame.copy()
    for idx, row in out.iterrows():
        race_id = text(row.get("race_id"))
        ticket_type = normalize_ticket_type_for_lap(row.get("ticket_type"))
        a_no = int(number(first_value(row, ["anchor_no", "horse_a", "a_no"]), 0))
        b_no = int(number(first_value(row, ["partner_no", "horse_b", "b_no"]), 0))
        if not race_id or not a_no or not b_no:
            continue
        lo, hi = sorted([a_no, b_no])
        for key, value in lookup.get((race_id, ticket_type, lo, hi), {}).items():
            out.at[idx, key] = value
    return out


def yen(value: Any) -> str:
    n = int(round(number(value, 0.0)))
    return f"{n:,}円" if n else "0円"


def odds_text(value: Any) -> str:
    n = number(value, 0.0)
    return f"{n:.1f}倍" if n else "-"


def ticket_type_label(value: Any) -> str:
    raw = text(value).lower()
    labels = {
        "wide": "ワイド",
        "umaren": "馬連",
        "quinella": "馬連",
        "win": "単勝",
        "place": "複勝",
        "umatan": "馬単",
        "exacta": "馬単",
        "trio": "3連複",
        "trifecta": "3連単",
    }
    return labels.get(raw, text(value, "馬券"))


def humanize_reason_summary(reason: Any) -> str:
    raw = text(reason)
    if not raw:
        return "最強版の購入条件を通過"
    lower = raw.lower()
    parts: list[str] = []
    if "strongest_current" in lower or "strongest" in lower:
        parts.append("最強版の厳選条件を通過")
    if "umaren_only" in lower:
        parts.append("馬連向き")
    if "wide_only" in lower:
        parts.append("ワイド向き")
    margin = re.search(r"margin=([0-9.]+)", lower)
    if margin:
        parts.append(f"オッズ余裕 {float(margin.group(1)):.2f}倍")
    skip = re.search(r"skip=([0-9.]+)", lower)
    if skip:
        parts.append(f"見送りリスク {float(skip.group(1)):.2f}")
    front5 = re.search(r"front5=([0-9.]+)", lower)
    if front5:
        parts.append(f"前目期待 {float(front5.group(1)) * 100:.0f}%")
    first_condition = re.search(r"first_condition=([a-z0-9_]+)", lower)
    if first_condition:
        label = {
            "low_sample_edge_ok": "\u521d\u6761\u4ef6\u30fb\u4f4e\u30b5\u30f3\u30d7\u30eb\u306f\u5fc5\u8981\u5999\u5473\u3092\u539a\u3081\u306b\u78ba\u8a8d",
            "clear": "\u521d\u6761\u4ef6\u30ea\u30b9\u30af\u306f\u8584\u3044",
        }.get(first_condition.group(1), "")
        if label:
            parts.append(label)
    danger_popular = re.search(r"danger_popular=([a-z0-9_]+)", lower)
    if danger_popular:
        label = {
            "included_popular_edge_ok": "\u5371\u967a\u4eba\u6c17\u3092\u542b\u3080\u305f\u3081\u5999\u5473\u3092\u539a\u3081\u306b\u78ba\u8a8d",
            "clear": "\u5371\u967a\u4eba\u6c17\u306e\u5dfb\u304d\u8fbc\u307f\u306f\u8584\u3044",
        }.get(danger_popular.group(1), "")
        if label:
            parts.append(label)
    readability = re.search(r"readability=([a-z0-9_]+)", lower)
    if readability:
        label = {
            "difficult_edge_ok": "\u96e3\u3057\u3044\u30ec\u30fc\u30b9\u306e\u305f\u3081\u5999\u5473\u3092\u539a\u3081\u306b\u78ba\u8a8d",
            "clear": "\u30ec\u30fc\u30b9\u306e\u8aad\u307f\u3084\u3059\u3055\u306f\u8a31\u5bb9\u7bc4\u56f2",
        }.get(readability.group(1), "")
        if label:
            parts.append(label)
    if parts:
        return " / ".join(parts)
    return raw.replace("|", " / ").replace("_", " ")


def pace_label(value: Any) -> str:
    raw = text(value).lower()
    if raw == "slow":
        return "スロー"
    if raw in {"middle", "mid"}:
        return "ミドル"
    if raw in {"fast", "high"}:
        return "ハイ"
    return text(value, "-")


def pace_shape_label(value: Any) -> str:
    labels = {
        "single_leader_clear": "単騎逃げ濃厚",
        "front_duel_dense": "先行密集",
        "matched_speed_duel": "テン互角",
        "no_clear_leader": "逃げ不在",
        "mixed_queue": "隊列混在",
    }
    return labels.get(text(value), text(value, ""))


def pace_sentence(metrics: dict[str, float | str]) -> str:
    pace = text(metrics.get("pace"))
    pressure = float(metrics.get("early_pressure") or 0.0)
    front_count = float(metrics.get("front_count") or 0.0)
    queue = float(metrics.get("queue_clarity") or 0.0)
    duel = float(metrics.get("front_duel_risk") or 0.0)
    front_load = float(metrics.get("front_load") or 0.0)
    shape = pace_shape_label(metrics.get("pace_shape"))
    load = "先行負荷は軽め"
    if pressure >= 0.50:
        load = "先行負荷は高め"
    elif pressure >= 0.30:
        load = "先行負荷は標準"
    if pace == "slow":
        tail = "好位を取れる馬の位置利を重視します。"
    elif pace == "fast":
        tail = "前の消耗と差し込みの両方を警戒します。"
    elif pace:
        tail = "極端な偏りより、AI評価とオッズ妙味の両立を重視します。"
    else:
        tail = "展開データが薄いため、直前オッズと気配を優先します。"
    shape_parts = []
    if shape:
        shape_parts.append(f"隊列={shape}")
    if queue > 0:
        shape_parts.append(f"明瞭度{queue:.2f}")
    if duel > 0:
        shape_parts.append(f"競り合い{duel:.2f}")
    if front_load > 0:
        shape_parts.append(f"前半負荷{front_load:.2f}")
    shape_text = " / ".join(shape_parts)
    return f"{pace_label(pace)}想定。逃げ候補{front_count:.0f}頭、{load}（{pressure:.2f}）。{shape_text + '。' if shape_text else ''}{tail}"


def natural_ticket_reason(row: pd.Series, metrics: dict[str, float | str], *, reference: bool = False) -> str:
    ticket_type = ticket_type_label(first_value(row, ["ticket_type"], "馬券"))
    odds = number(first_value(row, ["runtime_odds", "live_odds", "quote_odds_proxy"], 0), 0.0)
    roi = number(first_value(row, ["runtime_expected_roi", "expected_roi_after_slippage"], 0), 0.0)
    hit = number(first_value(row, ["ticket_hit_prob", "pair_calibrated_hit_prob"], 0), 0.0)
    pace_pair_label = text(first_value(row, ["pace_pair_gate_label"], ""))
    pace_pair_note = text(first_value(row, ["pace_pair_gate_note"], ""))
    pace_pair_score = number(first_value(row, ["continuous_pair_formal_score"], 0), 0.0)
    closer_label = text(first_value(row, ["closer_shadow_label"], ""))
    closer_note = text(first_value(row, ["closer_shadow_note"], ""))
    closer_score = number(first_value(row, ["closer_shadow_score"], 0), 0.0)
    front_context_label = text(first_value(row, ["front_context_gate_label"], ""))
    front_context_note = text(first_value(row, ["front_context_gate_note"], ""))
    front_context_collapse = number(first_value(row, ["front_context_collapse_risk_score"], 0), 0.0)
    front_context_survival = number(first_value(row, ["front_context_survival_support_score"], 0), 0.0)
    queue_lap_title = text(first_value(row, ["queueLapTitle", "queue_lap_title"], ""))
    queue_lap_label = text(first_value(row, ["queueLapLabel", "queue_lap_label"], ""))
    queue_lap_note = text(first_value(row, ["queueLapNote", "queue_lap_note"], ""))
    queue_lap_priority = number(first_value(row, ["queueLapPriority", "queue_lap_priority"], 0), 0.0)
    lap_advanced_label = text(first_value(row, ["lap_advanced_shadow_label"], ""))
    lap_advanced_note = text(first_value(row, ["lap_advanced_shadow_note"], ""))
    lap_advanced_score = number(first_value(row, ["lap_advanced_combo_score"], 0), 0.0)
    lap_track_label = text(first_value(row, ["lap_track_shadow_label"], ""))
    lap_track_note = text(first_value(row, ["lap_track_shadow_note"], ""))
    lap_track_score = number(first_value(row, ["lap_track_shadow_score"], 0), 0.0)
    lap_advanced_map = {
        "lap_role_goodrun_strong": "ラップロール強: 想定ラップと前受け/差し役割がかみ合う",
        "goodrun_lap_strong": "好走時ラップ適性強: 好走時のラップ型と今回想定が近い",
        "lap_advanced_combo_strong": "ラップ総合強: 波形・好走時ラップ・役割が揃う",
        "lap_advanced_combo_watch": "ラップ総合注視: ラップ面は準候補",
    }
    if not queue_lap_label and lap_advanced_label in lap_advanced_map:
        queue_lap_label = lap_advanced_label
        queue_lap_title = lap_advanced_map[lap_advanced_label].split(":", 1)[0]
        queue_lap_note = lap_advanced_note or lap_advanced_map[lap_advanced_label]
        queue_lap_priority = lap_advanced_score
    if not queue_lap_label:
        anchor_front = max(
            number(first_value(row, ["anchor_projected_front5_prob", "anchor_front5_model_prob"], 0), 0.0),
            0.0,
        )
        partner_front = max(
            number(first_value(row, ["partner_projected_front5_prob", "partner_front5_model_prob"], 0), 0.0),
            0.0,
        )
        pair_front = number(first_value(row, ["projected_front5_prob", "ticket_front_position_reliability_score"], 0), 0.0)
        queue_clarity = max(
            number(first_value(row, ["race_queue_clarity_score", "anchor_runtime_queue_clarity_score"], 0), 0.0),
            number(first_value(row, ["partner_runtime_queue_clarity_score"], 0), 0.0),
        )
        duel_risk = max(
            number(first_value(row, ["race_front_duel_risk_score", "anchor_runtime_front_duel_risk_score"], 0), 0.0),
            number(first_value(row, ["partner_runtime_front_duel_risk_score"], 0), 0.0),
        )
        pair_front_both = anchor_front >= 0.60 and partner_front >= 0.60
        pair_front_any = pair_front_both or max(anchor_front, partner_front, pair_front) >= 0.60
        if pair_front_both:
            queue_lap_label = "front_pair_strong"
            queue_lap_title = "\u524d\u76ee\u30da\u30a2\u5f37"
            queue_lap_note = "\u30da\u30a22\u982d\u3068\u3082\u524d\u76ee\u5019\u88dc"
            queue_lap_priority = 0.35
        elif pair_front_any and queue_clarity >= 0.62:
            queue_lap_label = "lap_good_front_any"
            queue_lap_title = "\u968a\u5217\u8aad\u307f\u826f\u597d"
            queue_lap_note = "\u524d\u76ee\u5019\u88dc\u304c\u3044\u3066\u968a\u5217\u3082\u8aad\u307f\u3084\u3059\u3044"
            queue_lap_priority = 0.30
        elif duel_risk >= 0.62:
            queue_lap_label = "mixed_queue_watch"
            queue_lap_title = "\u968a\u5217\u6df7\u6226\u6ce8\u8996"
            queue_lap_note = "\u524d\u306e\u7af6\u308a\u5408\u3044\u304c\u5f37\u304f\u30ef\u30a4\u30c9\u306f\u6ce8\u8996"
            queue_lap_priority = 0.12
    lap_promote_label = text(first_value(row, ["lap_positive_expansion_label"], ""))
    lap_promote_note = text(first_value(row, ["lap_positive_expansion_note"], ""))
    lap_promote_score = number(first_value(row, ["lap_positive_expansion_score"], 0), 0.0)
    lap_role_score = number(first_value(row, ["lap_axis_specialist_role_score"], 0), 0.0)
    raw_reason = text(first_value(row, ["runtime_reason", "buy_reason_summary"], ""))
    lower = raw_reason.lower()
    margin = None
    match = re.search(r"margin=([0-9.]+)", lower)
    if match:
        margin = float(match.group(1))
    front5 = None
    match = re.search(r"front5=([0-9.]+)", lower)
    if match:
        front5 = float(match.group(1))

    if reference:
        return "購入条件は未達ですが、AI上位同士の比較用として表示しています。直前オッズが大きく改善しない限り見送り寄りです。"

    parts: list[str] = []
    if "strongest" in lower:
        parts.append("最強版の厳選条件を通過")
    else:
        parts.append("購入条件を通過")
    if ticket_type == "馬連":
        parts.append("2頭の連対期待とオッズ妙味のバランスが良い")
    elif ticket_type == "ワイド":
        parts.append("的中率を残しながら妙味を拾う形")
    if margin is not None and margin >= 1.2:
        parts.append(f"必要オッズを約{margin:.1f}倍上回る")
    elif odds:
        parts.append(f"現在{odds:.1f}倍で妙味を確認")
    if front5 is not None and front5 >= 0.60:
        parts.append(f"前目に行ける見込みが高い（約{front5:.0%}）")
    if roi >= 1.5:
        parts.append(f"期待値が高い（EV {roi:.2f}）")
    elif hit >= 0.20:
        parts.append(f"的中率も一定水準（{hit:.1%}）")
    pace_pair_map = {
        "pace_pair_strong": "展開読みとペア適性が噛み合う",
        "pace_pair_watch": "展開面は注視",
        "pace_pair_caution": "展開とペア適性は弱め",
    }
    if pace_pair_label in pace_pair_map:
        note = pace_pair_note or pace_pair_map[pace_pair_label]
        score_text = f" {pace_pair_score:.2f}" if pace_pair_score > 0 else ""
        parts.append(f"{note}{score_text}")
    closer_map = {
        "closer_watch_strong": "差しが届く展開を強めに警戒",
        "closer_watch": "差し浮上に注意",
    }
    if closer_label in closer_map:
        note = closer_note or closer_map[closer_label]
        score_text = f" {closer_score:.2f}" if closer_score > 0 else ""
        parts.append(f"{note}{score_text}")
    front_context_map = {
        "front_context_collapse_alert": "前崩れ文脈が強く、前目ペアは慎重",
        "front_context_collapse_watch": "前崩れ文脈に注意",
        "front_context_survival_watch": "前が残る文脈は比較的読みやすい",
    }
    if front_context_label in front_context_map:
        note = front_context_note or front_context_map[front_context_label]
        score_value = front_context_survival if front_context_label == "front_context_survival_watch" else front_context_collapse
        score_text = f" {score_value:.2f}" if score_value > 0 else ""
        parts.append(f"{note}{score_text}")
    queue_lap_map = {
        "front_pair_lap_good": "\u524d\u76ee\u30da\u30a2\u5f37\u3067\u30e9\u30c3\u30d7\u8aad\u307f\u3082\u826f\u597d",
        "front_pair_strong": "\u524d\u76ee\u306b\u884c\u3051\u308b\u30da\u30a2\u3068\u3057\u3066\u6ce8\u76ee",
        "lap_good_front_any": "\u968a\u5217\u8aad\u307f\u3068\u30e9\u30c3\u30d7\u8aad\u307f\u306f\u826f\u597d",
        "mixed_queue_watch": "\u968a\u5217\u6df7\u6226\u3067\u30ef\u30a4\u30c9\u306f\u6ce8\u8996",
        "lap_read_weak": "\u30e9\u30c3\u30d7\u8aad\u307f\u306f\u5f31\u3081",
        "lap_role_goodrun_strong": "ラップロール強",
        "goodrun_lap_strong": "好走時ラップ適性強",
        "lap_advanced_combo_strong": "ラップ総合強",
        "lap_advanced_combo_watch": "ラップ総合注視",
    }
    queue_note = queue_lap_note or queue_lap_title or queue_lap_map.get(queue_lap_label, "")
    if queue_note:
        score_text = f" {queue_lap_priority:.2f}" if queue_lap_priority > 0 else ""
        parts.append(f"{queue_note}{score_text}")
    lap_track_map = {
        "positive": "馬場×ラップ強: 馬場状態とラップ適性の組み合わせが良い",
        "positive_soft": "馬場×ラップ注視: サンプルは小さいが条件は良い",
        "caution": "馬場×ラップ参考: 年別ブレがあり買い昇格ではなく注視",
    }
    if lap_track_label in lap_track_map:
        note = lap_track_note or lap_track_map[lap_track_label]
        score_text = f" {lap_track_score:.2f}" if lap_track_score > 0 else ""
        parts.append(f"{note}{score_text}")
    horse_lap_note = text(first_value(row, ["horse_lap_decomp_runtime_note"], ""))
    horse_lap_score = number(first_value(row, ["horse_lap_decomp_runtime_score"], 0), 0.0)
    if horse_lap_note:
        score_text = f" {horse_lap_score:.2f}" if horse_lap_score > 0 else ""
        parts.append(f"{horse_lap_note}{score_text}")
    lap_promote_map = {
        "lap_1win_fast_same_distance_shadow": "ラップ1勝シャドー強: fast想定と距離継続の条件が揃う",
        "lap_promote_strong": "ラップ適合から昇格候補",
        "lap_promote_watch": "ラップ面は準候補",
        "lap_role_watch": "軸と相手のラップ役割は合う",
    }
    if lap_promote_label in lap_promote_map:
        note = lap_promote_note or lap_promote_map[lap_promote_label]
        score_value = lap_role_score if lap_promote_label == "lap_role_watch" else lap_promote_score
        score_text = f" {score_value:.2f}" if score_value > 0 else ""
        parts.append(f"{note}{score_text}")
    return "。".join(parts) + "。"


def percent(value: Any) -> str:
    n = number(value, 0.0)
    return f"{n * 100:.1f}%"


def pick_col(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    return ""


def first_value(row: pd.Series, names: list[str], default: Any = "") -> Any:
    for name in names:
        if name in row.index:
            value = row.get(name)
            if text(value):
                return value
    return default


def date_key(value: Any) -> str:
    raw = text(value)
    if not raw:
        return ""
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) == 6:
        yy = int(digits[:2])
        yyyy = 2000 + yy if yy < 80 else 1900 + yy
        return f"{yyyy}{digits[2:]}"
    if len(digits) == 8 and digits.startswith(("19", "20")):
        return digits
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y%m%d")
    if len(digits) >= 8:
        return digits[:8]
    return ""


def parse_post_time(date: str, value: Any) -> datetime | None:
    raw = text(value)
    if not date or not raw:
        return None
    for fmt in ("%Y%m%d %H:%M", "%Y%m%d %H%M", "%Y%m%d %H:%M:%S"):
        try:
            return datetime.strptime(f"{date} {raw if ':' in raw else raw.zfill(4)}", fmt)
        except ValueError:
            pass
    parsed = pd.to_datetime(f"{date} {raw}", errors="coerce")
    return parsed.to_pydatetime() if pd.notna(parsed) else None


def race_id_keys(value: Any) -> set[str]:
    raw = text(value)
    if not raw:
        return set()
    keys = {raw}
    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits:
        keys.add(str(int(digits)))
    if len(digits) >= 16 and digits.startswith("20"):
        keys.add(digits[:16])
        tail = digits[8:16]
        if len(tail) == 8:
            keys.add(str(int(tail[:2])) + digits[2:4] + tail[2:])
        for n in (12, 10, 9, 8):
            keys.add(str(int(digits[-n:])))
    if len(digits) >= 12 and digits.startswith("20"):
        keys.add(digits[:12])
        keys.add(str(int(digits[4:6])) + digits[2:4] + digits[6:12])
    if len(digits) >= 9 and not digits.startswith("20"):
        keys.add(digits[-9:])
    return {k for k in keys if k}


def strict_race_ids_from_keys(keys: set[str]) -> set[str]:
    strict: set[str] = set()
    for key in keys:
        digits = "".join(ch for ch in text(key) if ch.isdigit())
        if len(digits) >= 16 and digits.startswith("20"):
            strict.add(digits[:16])
    return strict


def row_strict_race_ids(row: pd.Series) -> set[str]:
    strict: set[str] = set()
    for col in ["race_id", "繝ｬ繝ｼ繧ｹID(譁ｰ/鬥ｬ逡ｪ辟｡)", "target_race_id", "race_key"]:
        if col not in row.index:
            continue
        raw = text(row.get(col))
        if not raw:
            continue
        digits = "".join(ch for ch in raw if ch.isdigit())
        if len(digits) >= 16 and digits.startswith("20"):
            strict.add(digits[:16])
    return strict


def source_url_race_id(value: Any) -> str:
    raw = text(value)
    match = re.search(r"race_id=(\d+)", raw)
    return match.group(1) if match else ""


def entry_race_keys(row: pd.Series, dkey: str) -> set[str]:
    keys: set[str] = set()
    source_id = source_url_race_id(row.get("source_url", ""))
    if source_id:
        keys |= race_id_keys(source_id)
        if dkey and len(source_id) >= 12:
            keys.add(dkey + source_id[4:12])
    compact = first_value(row, ["レースID(新/馬番無)", "race_id", "race_key"], "")
    if compact:
        keys |= race_id_keys(compact)
    return keys


@dataclass
class RaceInfo:
    key: str
    keys: set[str]
    date_key: str
    venue: str
    race_no: int
    race_name: str
    surface: str
    distance: str
    field_size: str
    post_time: datetime

    @property
    def label(self) -> str:
        distance = f"{self.surface}{self.distance}m" if self.surface or self.distance else ""
        field = f"{self.field_size}頭" if self.field_size else ""
        return " ".join(
            x
            for x in [
                self.post_time.strftime("%H:%M"),
                f"{self.venue}{self.race_no}R",
                self.race_name,
                distance,
                field,
            ]
            if x
        )


def load_default_entry_csv() -> Path | None:
    summary_path = project_path("outputs/ui/live_odds_dashboard.summary.json")
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            entry = summary.get("entry_csv")
            if entry and Path(entry).exists():
                return Path(entry)
        except Exception:
            pass
    candidates = sorted(
        (ROOT / "data/datasets/inference/weekly").glob("entry_snapshot_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_dashboard_url(explicit: str) -> str:
    if text(explicit):
        return text(explicit)
    info_path = project_path("outputs/runtime/public_dashboard_tunnel.json")
    if not info_path.exists():
        return ""
    try:
        info = json.loads(info_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return ""
    public_url = text(info.get("public_url"))
    if public_url:
        return f"{public_url.rstrip('/')}/outputs/ui/live_odds_dashboard.html"
    dashboard_url = text(info.get("dashboard_url"))
    if dashboard_url:
        return dashboard_url
    return ""


def build_race_infos(entry: pd.DataFrame, target_date: str) -> list[RaceInfo]:
    if entry.empty:
        return []
    date_col = pick_col(entry, ["日付", "日付S", "date", "date_key"])
    venue_col = pick_col(entry, ["場所", "venue", "場名"])
    race_no_col = pick_col(entry, ["Ｒ", "R", "レース番号"])
    race_name_col = pick_col(entry, ["レース名", "race_name"])
    post_col = pick_col(entry, ["発走時刻", "post_time", "発走"])
    surface_col = pick_col(entry, ["芝・ダ", "surface"])
    distance_col = pick_col(entry, ["距離", "distance"])
    field_col = pick_col(entry, ["頭数", "出走頭数", "field_size"])
    if not (date_col and venue_col and race_no_col and post_col):
        return []

    tmp = entry.copy()
    tmp["_date_key"] = tmp[date_col].map(date_key)
    tmp = tmp[tmp["_date_key"].eq(target_date)].copy()
    if tmp.empty:
        return []
    tmp["_race_no_int"] = pd.to_numeric(tmp[race_no_col], errors="coerce").fillna(0).astype(int)
    group_cols = ["_date_key", venue_col, "_race_no_int"]
    if race_name_col:
        group_cols.append(race_name_col)

    races: list[RaceInfo] = []
    for _, race in tmp.groupby(group_cols, sort=False, dropna=False):
        row = race.iloc[0]
        dkey = text(row.get("_date_key"))
        post = parse_post_time(dkey, row.get(post_col))
        if not post:
            continue
        keys: set[str] = set()
        for _, r in race.iterrows():
            keys |= entry_race_keys(r, dkey)
        if not keys:
            # Last-resort key for message-only display.
            keys.add(f"{dkey}:{text(row.get(venue_col))}:{int(row.get('_race_no_int'))}")
        key = sorted(keys, key=lambda x: (not x.startswith(dkey), -len(x), x))[0]
        races.append(
            RaceInfo(
                key=key,
                keys=keys,
                date_key=dkey,
                venue=text(row.get(venue_col)),
                race_no=int(row.get("_race_no_int")),
                race_name=text(row.get(race_name_col)) if race_name_col else "",
                surface=text(row.get(surface_col)) if surface_col else "",
                distance=text(row.get(distance_col)) if distance_col else "",
                field_size=text(row.get(field_col), str(len(race))) if field_col else str(len(race)),
                post_time=post,
            )
        )
    return sorted(races, key=lambda r: (r.post_time, r.venue, r.race_no))


def row_race_keys(row: pd.Series) -> set[str]:
    keys = set()
    for col in ["race_id", "レースID(新/馬番無)", "target_race_id", "race_key"]:
        if col in row.index:
            keys |= race_id_keys(row.get(col))
    return keys


def add_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["_race_keys"] = [row_race_keys(row) for _, row in work.iterrows()]
    work["_strict_race_ids"] = [row_strict_race_ids(row) for _, row in work.iterrows()]
    return work


def read_results_for_date(path: str | Path, target_date: str) -> pd.DataFrame:
    p = project_path(path)
    if not p.exists():
        return pd.DataFrame()
    wanted_cols = {
        "レースID(新/馬番無)",
        "race_id",
        "target_race_id",
        "race_key",
        "馬名",
        "horse_name",
        "name",
        "確定着順",
        "finish_rank",
        "rank",
        "着順",
        "日付",
        "日付S",
        "date",
        "date_key",
    }
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            chunks: list[pd.DataFrame] = []
            reader = pd.read_csv(
                p,
                encoding=enc,
                low_memory=False,
                dtype=str,
                usecols=lambda col: col in wanted_cols,
                chunksize=200_000,
            )
            for chunk in reader:
                date_col = pick_col(chunk, ["日付", "日付S", "date", "date_key"])
                if date_col:
                    chunk = chunk[chunk[date_col].map(date_key).eq(target_date)].copy()
                if not chunk.empty:
                    chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        except UnicodeDecodeError:
            continue
        except ValueError:
            # Fall back to the generic reader if the source has an unexpected header.
            raw = read_csv_safe(p)
            date_col = pick_col(raw, ["日付", "日付S", "date", "date_key"])
            if date_col:
                raw = raw[raw[date_col].map(date_key).eq(target_date)].copy()
            return raw
    raw = read_csv_safe(p)
    date_col = pick_col(raw, ["日付", "日付S", "date", "date_key"])
    if date_col:
        raw = raw[raw[date_col].map(date_key).eq(target_date)].copy()
    return raw


def filter_by_race(df: pd.DataFrame, race: RaceInfo) -> pd.DataFrame:
    if df.empty or "_race_keys" not in df.columns:
        return pd.DataFrame()
    race_strict_ids = strict_race_ids_from_keys(race.keys)
    if race_strict_ids and "_strict_race_ids" in df.columns:
        strict_mask = df["_strict_race_ids"].map(lambda ids: bool(set(ids) & race_strict_ids))
        if strict_mask.any():
            return df[strict_mask].copy()
        # Rows that carry a full JRA race_id must not fall back to broad venue/date keys.
        has_strict_rows = df["_strict_race_ids"].map(lambda ids: bool(ids)).any()
        if has_strict_rows:
            loose_only = df[~df["_strict_race_ids"].map(lambda ids: bool(ids))].copy()
            if loose_only.empty:
                return pd.DataFrame(columns=df.columns)
            df = loose_only
    mask = df["_race_keys"].map(lambda keys: bool(set(keys) & race.keys))
    return df[mask].copy()


def best_candidates_for_race(candidates: pd.DataFrame, race: RaceInfo, limit: int = 2) -> pd.DataFrame:
    work = filter_by_race(candidates, race)
    if work.empty:
        return work
    work["_score"] = pd.to_numeric(work.get("strongest_current_score", 0), errors="coerce").fillna(0.0)
    return work.sort_values("_score", ascending=False).head(limit)


def reference_candidates_for_race(candidates: pd.DataFrame, race: RaceInfo, limit: int = 2) -> pd.DataFrame:
    work = filter_by_race(candidates, race)
    if work.empty:
        return work
    work = work.copy()
    work["_score"] = pd.to_numeric(work.get("strongest_current_score", 0), errors="coerce").fillna(0.0)
    work["_odds"] = pd.to_numeric(work.get("live_odds", work.get("runtime_odds", 0)), errors="coerce").fillna(0.0)
    work["_hit"] = pd.to_numeric(work.get("ticket_hit_prob", 0), errors="coerce").fillna(0.0)
    realistic = work[work["_odds"].between(1.0, 120.0) & work["_hit"].ge(0.05)].copy()
    if realistic.empty:
        realistic = work
    return realistic.sort_values(["_score", "_hit"], ascending=[False, False]).head(limit)


def buy_tickets_for_race(tickets: pd.DataFrame, race: RaceInfo) -> pd.DataFrame:
    work = filter_by_race(tickets, race)
    if work.empty:
        return work
    if "runtime_action" in work.columns:
        buy = work[work["runtime_action"].astype(str).str.upper().eq("BUY")].copy()
        if not buy.empty:
            work = buy
    work["_stake"] = pd.to_numeric(work.get("runtime_stake_yen", work.get("stake_yen", 0)), errors="coerce").fillna(0.0)
    work["_roi"] = pd.to_numeric(work.get("runtime_expected_roi", 0), errors="coerce").fillna(0.0)
    return work.sort_values(["_stake", "_roi"], ascending=[False, False])


def race_best_metrics(candidates: pd.DataFrame, race: RaceInfo) -> dict[str, float | str]:
    best = best_candidates_for_race(candidates, race, limit=1)
    if best.empty:
        return {
            "score": 0.0,
            "skip": 1.0,
            "pace": "",
            "front_count": 0.0,
            "early_pressure": 0.0,
            "queue_clarity": 0.0,
            "front_duel_risk": 0.0,
            "front_load": 0.0,
            "pace_shape": "",
        }
    row = best.iloc[0]
    return {
        "score": number(row.get("strongest_current_score"), 0.0),
        "skip": number(row.get("skip_risk_score"), 1.0),
        "pace": text(row.get("anchor_expected_pace")),
        "front_count": number(row.get("anchor_race_front_runner_count"), 0.0),
        "early_pressure": number(row.get("anchor_race_early_pressure_score"), 0.0),
        "queue_clarity": number(first_value(row, ["race_queue_clarity_score", "anchor_runtime_queue_clarity_score"]), 0.0),
        "front_duel_risk": number(first_value(row, ["race_front_duel_risk_score", "anchor_runtime_front_duel_risk_score"]), 0.0),
        "front_load": number(first_value(row, ["race_projected_front_load_score", "anchor_runtime_projected_front_load_score"]), 0.0),
        "pace_shape": text(first_value(row, ["race_pace_shape_label", "anchor_runtime_pace_shape_label"], "")),
    }


def race_policy_exclusion(race: RaceInfo, candidates: pd.DataFrame) -> str:
    if race.venue == "函館":
        return "最強版ゲート: 函館は見送り"
    work = filter_by_race(candidates, race)
    if work.empty:
        return ""
    soft_cols = [
        col
        for col in [
            "anchor_runtime_soft_heavy_flag",
            "partner_runtime_soft_heavy_flag",
            "runtime_soft_heavy_flag",
        ]
        if col in work.columns
    ]
    if soft_cols:
        flags = work[soft_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if bool(flags.gt(0).any().any()):
            return "最強版ゲート: 重・不良は見送り"
    return ""


def race_status(race: RaceInfo, tickets: pd.DataFrame, candidates: pd.DataFrame) -> tuple[str, str]:
    buy = buy_tickets_for_race(tickets, race)
    metrics = race_best_metrics(candidates, race)
    if not buy.empty:
        return "購入候補", "最強版の厳選買い目あり"
    excluded = race_policy_exclusion(race, candidates)
    if excluded:
        return "見送り濃厚", excluded
    score = float(metrics["score"])
    skip = float(metrics["skip"])
    if score >= 0.88 and skip <= 0.50:
        return "購入可能性あり", f"候補度{score:.2f} / リスク{skip:.2f}"
    if score >= 0.84 and skip <= 0.62:
        return "要監視", f"候補度{score:.2f} / 直前オッズ次第"
    return "見送り濃厚", f"候補度{score:.2f} / リスク{skip:.2f}"


def pre_race_notification_allowed(
    race: RaceInfo,
    tickets: pd.DataFrame,
    candidates: pd.DataFrame,
    policy: str,
) -> bool:
    if not buy_tickets_for_race(tickets, race).empty:
        return True
    if policy == "all":
        return True
    if policy == "buy_or_watch":
        status, _ = race_status(race, tickets, candidates)
        return status in {"購入可能性あり", "要監視"}
    return False


def ticket_line(row: pd.Series, reference: bool = False) -> str:
    ticket_type = ticket_type_label(first_value(row, ["ticket_type"], "馬券"))
    anchor_no = text(first_value(row, ["anchor_no", "anchor_horse_no", "a_no"], ""))
    partner_no = text(first_value(row, ["partner_no", "partner_horse_no", "b_no"], ""))
    anchor = text(first_value(row, ["anchor_name", "anchor_馬名", "anchor_horse_name"], ""))
    partner = text(first_value(row, ["partner_name", "partner_馬名", "partner_horse_name"], ""))
    stake = number(first_value(row, ["runtime_stake_yen", "stake_yen"], 0), 0.0)
    odds = number(first_value(row, ["runtime_odds", "live_odds", "quote_odds_proxy"], 0), 0.0)
    pay_per100 = number(first_value(row, ["runtime_pay_per100", "live_pay_per100", "quote_pay_proxy_per100"], 0), 0.0)
    if not pay_per100 and odds:
        pay_per100 = odds * 100
    payout = stake * pay_per100 / 100 if stake else pay_per100
    hit = number(first_value(row, ["ticket_hit_prob", "pair_calibrated_hit_prob"], 0), 0.0)
    roi = number(first_value(row, ["runtime_expected_roi", "expected_roi_after_slippage"], 0), 0.0)
    amount = "0円（参考のみ）" if reference else yen(stake)
    payout_label = f"100円時 {int(round(payout)):,}円" if reference else f"想定払戻 {int(round(payout)):,}円"
    return (
        f"{ticket_type} {anchor_no}-{partner_no} {anchor}-{partner}\n"
        f"  金額 {amount} / {odds_text(odds)} / {payout_label} / 的中 {hit:.1%} / EV {roi:.2f}"
    )


def normalize_name(value: Any) -> str:
    return re.sub(r"\s+", "", text(value))


def result_rows_for_ticket(results: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if results.empty or "_race_keys" not in results.columns:
        return pd.DataFrame()
    keys = row_race_keys(row)
    if not keys:
        return pd.DataFrame()
    mask = results["_race_keys"].map(lambda item: bool(set(item) & keys))
    return results[mask].copy()


def finish_for_name(result_rows: pd.DataFrame, horse_name: Any) -> float | None:
    if result_rows.empty:
        return None
    name = normalize_name(horse_name)
    if not name:
        return None
    horse_col = pick_col(result_rows, ["馬名", "horse_name", "name"])
    finish_col = pick_col(result_rows, ["確定着順", "finish_rank", "rank", "着順"])
    if not horse_col or not finish_col:
        return None
    work = result_rows.copy()
    work["_name_norm"] = work[horse_col].map(normalize_name)
    hit = work[work["_name_norm"].eq(name)]
    if hit.empty:
        return None
    return number(hit.iloc[0].get(finish_col), 0.0)


def evaluate_ticket_result(row: pd.Series, results: pd.DataFrame) -> dict[str, Any]:
    ticket_type_raw = text(first_value(row, ["ticket_type"], "")).lower()
    ticket_label = ticket_type_label(ticket_type_raw)
    stake = number(first_value(row, ["runtime_stake_yen", "stake_yen"], 0), 0.0)
    pay_per100 = number(first_value(row, ["runtime_pay_per100", "live_pay_per100", "quote_pay_proxy_per100"], 0), 0.0)
    odds = number(first_value(row, ["runtime_odds", "live_odds", "quote_odds_proxy"], 0), 0.0)
    if not pay_per100 and odds:
        pay_per100 = odds * 100.0

    anchor_no = text(first_value(row, ["anchor_no", "anchor_horse_no", "a_no"], ""))
    partner_no = text(first_value(row, ["partner_no", "partner_horse_no", "b_no"], ""))
    anchor_name = text(first_value(row, ["anchor_name", "anchor_馬名", "anchor_horse_name"], ""))
    partner_name = text(first_value(row, ["partner_name", "partner_馬名", "partner_horse_name"], ""))
    race_results = result_rows_for_ticket(results, row)
    anchor_finish = finish_for_name(race_results, anchor_name)
    partner_finish = finish_for_name(race_results, partner_name) if partner_name else None

    pending = False
    hit = False
    if anchor_finish is None:
        pending = True
    elif ticket_type_raw in {"win", "tansho"}:
        hit = anchor_finish == 1
    elif ticket_type_raw in {"place", "fukusho"}:
        hit = anchor_finish <= 3
    elif ticket_type_raw in {"wide"}:
        pending = partner_finish is None
        hit = (not pending) and anchor_finish <= 3 and partner_finish <= 3
    elif ticket_type_raw in {"umaren", "quinella"}:
        pending = partner_finish is None
        hit = (not pending) and anchor_finish <= 2 and partner_finish <= 2
    elif ticket_type_raw in {"umatan", "exacta"}:
        pending = partner_finish is None
        hit = (not pending) and anchor_finish == 1 and partner_finish == 2
    else:
        pending = partner_name != "" and partner_finish is None

    ret = pay_per100 * stake / 100.0 if hit and stake else 0.0
    combo = f"{anchor_no}-{partner_no} {anchor_name}-{partner_name}" if partner_name else f"{anchor_no} {anchor_name}"
    finish_text = "-"
    if anchor_finish is not None:
        finish_text = f"{int(anchor_finish)}着"
        if partner_name:
            finish_text += f"/{int(partner_finish)}着" if partner_finish is not None else "/未取込"
    return {
        "ticket_type": ticket_label,
        "combo": combo,
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake if not pending else 0.0,
        "hit": hit,
        "pending": pending,
        "finish_text": finish_text,
        "odds": odds,
    }


def build_daily_summary_message(
    races: list[RaceInfo],
    tickets: pd.DataFrame,
    results: pd.DataFrame,
    dashboard_url: str,
) -> str:
    rows = tickets.copy()
    if not rows.empty and races and "_race_keys" in rows.columns:
        race_keys: set[str] = set()
        for race in races:
            race_keys |= race.keys
        rows = rows[rows["_race_keys"].map(lambda keys: bool(set(keys) & race_keys))].copy()
    if not rows.empty and "runtime_action" in rows.columns:
        buy_rows = rows[rows["runtime_action"].astype(str).str.upper().eq("BUY")].copy()
        if not buy_rows.empty:
            rows = buy_rows
    if rows.empty:
        title_date = races[0].post_time.strftime("%Y/%m/%d") if races else datetime.now().strftime("%Y/%m/%d")
        lines = [f"Keiba AI 一日成績 {title_date}", "本日の購入対象はありませんでした。"]
        if dashboard_url:
            lines.append(f"画面: {dashboard_url}")
        return "\n".join(lines).strip()

    evaluated = [evaluate_ticket_result(row, results) for _, row in rows.iterrows()]
    settled = [r for r in evaluated if not r["pending"]]
    pending = [r for r in evaluated if r["pending"]]
    total_stake = sum(float(r["stake_yen"]) for r in evaluated)
    stake = sum(float(r["stake_yen"]) for r in settled)
    ret = sum(float(r["return_yen"]) for r in settled)
    profit = ret - stake
    roi = ret / stake if stake else 0.0
    hits = sum(1 for r in settled if r["hit"])
    race_count = rows["race_id"].astype(str).nunique() if "race_id" in rows.columns else len(rows)
    title_date = races[0].post_time.strftime("%Y/%m/%d") if races else datetime.now().strftime("%Y/%m/%d")
    lines = [
        f"Keiba AI 一日成績 {title_date}",
        f"購入 {len(rows)}点 / {race_count}R / 予定投資 {yen(total_stake)}",
    ]
    if settled:
        mark = "プラス" if profit > 0 else "マイナス" if profit < 0 else "トントン"
        lines.append(
            f"確定 {len(settled)}点: 的中 {hits}点 / 投資 {yen(stake)} / 払戻 {yen(ret)} / 収支 {yen(profit)} / 回収率 {roi:.1%}（{mark}）"
        )
    if pending:
        lines.append(f"未確定 {len(pending)}点: 結果CSVに当日結果がまだ入っていません。TARGET更新後に再集計します。")
    if dashboard_url:
        lines.append(f"画面: {dashboard_url}")
    lines.append("")

    for item in evaluated[:8]:
        status = "未確定" if item["pending"] else "的中" if item["hit"] else "不的中"
        result_part = f"{item['finish_text']} / {status}"
        money_part = f"投資{yen(item['stake_yen'])}"
        if not item["pending"]:
            money_part += f"→払戻{yen(item['return_yen'])}"
        lines.append(f"{item['ticket_type']} {item['combo']} : {result_part} / {money_part}")
    if len(evaluated) > 8:
        lines.append(f"...ほか {len(evaluated) - 8}点")
    return "\n".join(lines).strip()


def build_morning_message(
    races: list[RaceInfo],
    tickets: pd.DataFrame,
    candidates: pd.DataFrame,
    dashboard_url: str,
    max_lines: int,
) -> str:
    counts = {"購入候補": 0, "購入可能性あり": 0, "要監視": 0, "見送り濃厚": 0}
    rows: list[str] = []
    for race in races:
        status, reason = race_status(race, tickets, candidates)
        counts[status] = counts.get(status, 0) + 1
        metrics = race_best_metrics(candidates, race)
        pace = f" / {pace_label(metrics['pace'])}" if metrics.get("pace") else ""
        rows.append(f"{race.label} : {status}（{reason}{pace}）")
    title_date = races[0].post_time.strftime("%Y/%m/%d") if races else ""
    lines = [
        f"Keiba AI 朝チェック {title_date}",
        f"購入候補 {counts['購入候補']}R / 可能性あり {counts['購入可能性あり']}R / 要監視 {counts['要監視']}R / 見送り濃厚 {counts['見送り濃厚']}R",
    ]
    if dashboard_url:
        lines.append(f"画面: {dashboard_url}")
    lines.append("")
    lines.extend(rows[:max_lines])
    if len(rows) > max_lines:
        lines.append(f"...ほか {len(rows) - max_lines}R")
    return "\n".join(lines).strip()


def build_pre_race_message(
    race: RaceInfo,
    stage: str,
    tickets: pd.DataFrame,
    candidates: pd.DataFrame,
    dashboard_url: str,
    reference_limit: int,
) -> str:
    buy = buy_tickets_for_race(tickets, race)
    metrics = race_best_metrics(candidates, race)
    if not buy.empty:
        decision = "購入" if stage == "final3" else "購入候補"
        rows = buy
        reference = False
    else:
        decision = "参考・見送り"
        rows = reference_candidates_for_race(candidates, race, limit=reference_limit)
        reference = True

    stage_label = "3分前 確定版" if stage == "final3" else "5分前 チェック"
    lines = [
        f"Keiba AI {stage_label}",
        race.label,
        f"判定: {decision}",
        f"展開予想: {pace_sentence(metrics)}",
    ]
    if dashboard_url:
        lines.append(f"画面: {dashboard_url}")
    lines.append("")
    if rows.empty:
        lines.append("推奨買い目: なし")
        lines.append("理由: 候補度が低く、直前オッズ込みでも買い条件未達です。")
        return "\n".join(lines).strip()

    for _, row in rows.iterrows():
        lines.append(ticket_line(row, reference=reference))
        lines.append(f"  理由 {natural_ticket_reason(row, metrics, reference=reference)}")
    if reference:
        lines.append("")
        lines.append("※参考表示です。購入条件は未達なので、実買いは見送り扱いです。")
    return "\n".join(lines).strip()


def send_line_push(message: str, token: str, to: str) -> dict[str, Any]:
    payload = {"to": to, "messages": [{"type": "text", "text": message}]}
    request = urllib.request.Request(
        LINE_PUSH_ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {"ok": True, "status": response.status, "body": response.read().decode("utf-8", errors="replace")}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "body": exc.read().decode("utf-8", errors="replace")}
    except urllib.error.URLError as exc:
        return {"ok": False, "status": None, "body": str(exc.reason)}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"sent": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state.get("sent"), dict):
            state["sent"] = {}
        return state
    except Exception:
        return {"sent": {}}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def message_hash(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def event_due(now: datetime, race: RaceInfo, stage: str) -> bool:
    if stage == "pre5":
        return race.post_time - timedelta(minutes=5) <= now < race.post_time - timedelta(minutes=3)
    if stage == "final3":
        return race.post_time - timedelta(minutes=3) <= now < race.post_time
    return False


def parse_now(value: str) -> datetime:
    if value:
        parsed = pd.to_datetime(value, errors="raise")
        return parsed.to_pydatetime()
    return datetime.now()


def env_value(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                stored, _ = winreg.QueryValueEx(key, name)
                return text(stored)
        except Exception:
            return ""
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Send timed LINE notifications for race-day Keiba AI operation.")
    parser.add_argument("--entry-csv", default="")
    parser.add_argument("--tickets-csv", default="outputs/analysis/current_strongest_runtime_v1/selected_after_live_safety.csv")
    parser.add_argument("--candidates-csv", default="outputs/analysis/current_strongest_runtime_v1/current_strongest_all_candidates.csv")
    parser.add_argument(
        "--horse-lap-decomp-csv",
        default="outputs/analysis/horse_lap_aptitude_decomposition_v1/all_ticket_lap_decomposition_compact.csv",
    )
    parser.add_argument("--results-csv", default="data/processed/normalized/results.csv")
    parser.add_argument("--date", default="", help="YYYYMMDD. Defaults to today.")
    parser.add_argument("--now", default="")
    parser.add_argument("--event", choices=["auto", "morning", "pre5", "final3", "daily"], default="auto")
    parser.add_argument("--race-key", default="")
    parser.add_argument("--morning-time", default="08:00")
    parser.add_argument("--morning-window-minutes", type=int, default=180)
    parser.add_argument("--daily-summary-time", default="17:10")
    parser.add_argument("--disable-final3", action="store_true", help="Do not send 3-minute pre-race notices in auto mode.")
    parser.add_argument("--disable-daily-summary", action="store_true", help="Do not send the end-of-day daily summary in auto mode.")
    parser.add_argument("--dashboard-url", default="")
    parser.add_argument("--state-json", default="data/processed/notifications/race_day_timed_line_state.json")
    parser.add_argument("--output-dir", default="outputs/notifications/race_day_line")
    parser.add_argument("--max-morning-lines", type=int, default=60)
    parser.add_argument("--reference-limit", type=int, default=2)
    parser.add_argument(
        "--pre-race-policy",
        choices=["buy", "buy_or_watch", "all"],
        default="buy",
        help="Which races should get 5-minute/3-minute LINE messages. Default is BUY races only.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--send-if-configured", action="store_true")
    parser.add_argument("--line-token-env", default="LINE_CHANNEL_ACCESS_TOKEN")
    parser.add_argument("--line-to-env", default="LINE_USER_ID")
    args = parser.parse_args()

    now = parse_now(args.now)
    target_date = date_key(args.date) if args.date else now.strftime("%Y%m%d")
    entry_path = project_path(args.entry_csv) if args.entry_csv else load_default_entry_csv()
    if not entry_path:
        raise SystemExit("entry csv was not found")
    entry = read_csv_safe(entry_path)
    tickets = add_match_keys(enrich_horse_lap_decomp(read_csv_safe(args.tickets_csv), args.horse_lap_decomp_csv))
    candidates = add_match_keys(enrich_horse_lap_decomp(read_csv_safe(args.candidates_csv), args.horse_lap_decomp_csv))
    races = build_race_infos(entry, target_date)
    dashboard_url = resolve_dashboard_url(args.dashboard_url)

    if args.race_key:
        wanted = race_id_keys(args.race_key)
        wanted_strict = strict_race_ids_from_keys(wanted)
        if wanted_strict:
            races = [race for race in races if strict_race_ids_from_keys(race.keys) & wanted_strict]
        else:
            races = [race for race in races if race.keys & wanted]

    state_path = project_path(args.state_json)
    state = load_state(state_path)
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events: list[tuple[str, str, str]] = []
    if args.event in {"auto", "morning"}:
        hh, mm = [int(x) for x in args.morning_time.split(":", 1)]
        morning_at = datetime.combine(now.date(), time(hh, mm))
        morning_due = morning_at <= now < morning_at + timedelta(minutes=args.morning_window_minutes)
        if args.force or args.event == "morning" or morning_due:
            message = build_morning_message(races, tickets, candidates, dashboard_url, args.max_morning_lines)
            events.append((f"{target_date}:morning", "morning", message))

    if args.event in {"auto", "daily"} and not args.disable_daily_summary:
        hh, mm = [int(x) for x in args.daily_summary_time.split(":", 1)]
        try:
            target_day = datetime.strptime(target_date, "%Y%m%d").date()
        except ValueError:
            target_day = now.date()
        daily_at = datetime.combine(target_day, time(hh, mm))
        if args.force or args.event == "daily" or now >= daily_at:
            result_rows = add_match_keys(read_results_for_date(args.results_csv, target_date))
            message = build_daily_summary_message(races, tickets, result_rows, dashboard_url)
            events.append((f"{target_date}:daily", "daily", message))

    pre_race_stages = ["pre5"]
    if not args.disable_final3:
        pre_race_stages.append("final3")
    for race in races:
        for stage in pre_race_stages:
            if args.event not in {"auto", stage}:
                continue
            if args.force or args.event == stage or event_due(now, race, stage):
                if not args.force and not pre_race_notification_allowed(
                    race,
                    tickets,
                    candidates,
                    args.pre_race_policy,
                ):
                    continue
                message = build_pre_race_message(race, stage, tickets, candidates, dashboard_url, args.reference_limit)
                events.append((f"{race.key}:{stage}", stage, message))

    token = env_value(args.line_token_env)
    to = env_value(args.line_to_env)
    can_send = args.send or (args.send_if_configured and token and to)
    if args.send and (not token or not to):
        raise SystemExit(
            textwrap.dedent(
                f"""\
                LINE credentials are missing.
                Set environment variables:
                  {args.line_token_env}=<Messaging API channel access token>
                  {args.line_to_env}=<LINE user ID or group ID>
                """
            )
        )

    results: list[dict[str, Any]] = []
    for event_id, stage, message in events:
        digest = message_hash(message)
        # Race-day notifications are event based, not content based. Odds and
        # candidate wording can change on every refresh, but the morning check,
        # 5-minute notice, 3-minute notice, and daily summary should each be sent
        # only once unless the operator explicitly passes --force.
        already_sent = event_id in state["sent"]
        skipped = already_sent and not args.force
        output_path = out_dir / f"{event_id.replace(':', '_')}.txt"
        output_path.write_text(message + "\n", encoding="utf-8-sig")
        result: dict[str, Any] = {
            "event_id": event_id,
            "stage": stage,
            "skipped_duplicate": skipped,
            "duplicate_policy": "event_id_once",
            "sent": False,
            "output_text": str(output_path),
            "message_preview": message[:500],
        }
        if not skipped and can_send:
            line_result = send_line_push(message, token, to)
            result["line"] = line_result
            result["sent"] = bool(line_result.get("ok"))
            if result["sent"]:
                state["sent"][event_id] = {
                    "hash": digest,
                    "sent_at": now.isoformat(timespec="seconds"),
                    "stage": stage,
                }
        elif not skipped and not can_send:
            result["dry_run"] = True
        results.append(result)

    save_state(state_path, state)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "now": now.isoformat(timespec="seconds"),
        "date": target_date,
        "entry_csv": str(entry_path),
        "dashboard_url": dashboard_url,
        "races": len(races),
        "events": len(events),
        "results": results,
        "send_requested": bool(args.send or args.send_if_configured),
        "can_send": bool(can_send),
    }
    summary_path = out_dir / "latest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
