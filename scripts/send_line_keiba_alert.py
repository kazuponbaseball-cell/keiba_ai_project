from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_preday_dashboard_html import (  # noqa: E402
    choose_entry_source,
    choose_predictions_dir,
    date_key,
    fmt_date,
    has_value,
    latest_prediction,
    merge_prediction,
    number,
    pick_col,
    race_decision,
    race_id_col,
    race_value,
    read_csv_safe,
    text,
    top_runner,
)


LINE_PUSH_ENDPOINT = "https://api.line.me/v2/bot/message/push"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def display_horse(row: pd.Series) -> str:
    horse_no = text(row.get("馬番")).strip()
    if horse_no.endswith(".0"):
        horse_no = horse_no[:-2]
    name = text(row.get("馬名"), "不明")
    jockey = text(row.get("騎手"))
    score = number(row.get("ai_score"), 0.0)
    rank = text(row.get("ai_rank"), "-")
    if rank.endswith(".0"):
        rank = rank[:-2]
    bits = [f"AI{rank}", f"{horse_no} {name}".strip(), f"score {score:.3f}"]
    if jockey:
        bits.append(jockey)
    return " / ".join(bits)


def race_label(race: pd.DataFrame) -> str:
    date = fmt_date(race_value(race, ["日付S", "日付", "date"]))
    venue = text(race_value(race, ["場所"]))
    race_no = text(race_value(race, ["Ｒ", "R"]))
    race_name = text(race_value(race, ["レース名"]))
    surface = text(race_value(race, ["芝・ダ"]))
    distance = text(race_value(race, ["距離"]))
    if distance.endswith(".0"):
        distance = distance[:-2]
    field = text(race_value(race, ["頭数", "出走頭数"], len(race)))
    if field.endswith(".0"):
        field = field[:-2]
    return f"{date} {venue}{race_no}R {race_name} {surface}{distance}m {field}頭"


def body_race_keys(value: object) -> set[str]:
    raw = text(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    keys = {raw} if raw else set()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits:
        keys.add(str(int(digits)))
    if len(digits) >= 16 and digits.startswith("20"):
        # Current live IDs may be YYYYMMDD + JJKKDDR. Convert that to the
        # compact TARGET-style key J + YY + KKDDR, e.g. 2026062002010312
        # -> 226010312.
        tail = digits[8:16]
        if len(tail) == 8:
            venue = str(int(tail[:2]))
            keys.add(venue + digits[2:4] + tail[2:])
    if len(digits) >= 12 and digits.startswith("20"):
        # netkeiba-style 202605030501 -> project-style 526030501
        venue = str(int(digits[4:6]))
        keys.add(venue + digits[2:4] + digits[6:12])
    if len(digits) >= 16 and digits.startswith("20"):
        # Some TARGET/JV exports carry a longer race identifier. Keep useful suffixes
        # for loose matching when only the compact race id is available.
        for n in (12, 10, 9, 8):
            keys.add(str(int(digits[-n:])))
    return {k for k in keys if k}


def merge_body_weight(combined: pd.DataFrame, body_weight_csv: str) -> pd.DataFrame:
    if not body_weight_csv:
        return combined
    path = project_path(body_weight_csv)
    if not path.exists() or combined.empty:
        return combined
    body = read_csv_safe(path)
    if body.empty:
        return combined

    race_col = race_id_col(combined)
    horse_col = pick_col(combined, ["馬番", "horse_no"])
    if not race_col or not horse_col:
        return combined

    body = body.rename(
        columns={
            "馬番": "horse_no",
            "馬体重": "body_weight",
            "増減": "body_weight_diff",
            "snapshot_at": "body_weight_snapshot_at",
        }
    )
    if "race_id" not in body.columns or "horse_no" not in body.columns or "body_weight" not in body.columns:
        return combined
    if "body_weight_diff" not in body.columns:
        body["body_weight_diff"] = pd.NA
    if "body_weight_snapshot_at" not in body.columns:
        body["body_weight_snapshot_at"] = ""

    left = combined.copy()
    left["_horse_key"] = pd.to_numeric(left[horse_col], errors="coerce").astype("Int64").astype("string")
    left["_race_keys"] = left[race_col].map(body_race_keys)
    left = left.explode("_race_keys").rename(columns={"_race_keys": "_race_key"})

    right = body.copy()
    right["_horse_key"] = pd.to_numeric(right["horse_no"], errors="coerce").astype("Int64").astype("string")
    right["_race_keys"] = right["race_id"].map(body_race_keys)
    right = right.explode("_race_keys").rename(columns={"_race_keys": "_race_key"})
    right = right[["_race_key", "_horse_key", "body_weight", "body_weight_diff", "body_weight_snapshot_at"]].dropna(subset=["_race_key", "_horse_key"])
    right = right.drop_duplicates(["_race_key", "_horse_key"], keep="last")

    merged = left.merge(right, on=["_race_key", "_horse_key"], how="left")
    merged["_body_hit"] = merged["body_weight"].notna()
    merged = (
        merged.sort_values(["_body_hit"], ascending=False)
        .drop_duplicates([race_col, horse_col], keep="first")
        .drop(columns=["_horse_key", "_race_key", "_body_hit"], errors="ignore")
    )
    return merged


def should_include(decision: dict[str, Any], mode: str, require_body_weight: bool) -> bool:
    action_class = decision["class"]
    waits = set(decision.get("waits", []))
    if require_body_weight and "当日馬体重待ち" in waits:
        return False
    if mode == "all":
        return True
    if mode == "preday":
        return action_class in {"candidate", "buy"}
    if mode == "buyable":
        return action_class == "buy" or (action_class == "candidate" and "当日馬体重待ち" not in waits)
    if mode == "skip":
        return action_class == "skip"
    return action_class in {"candidate", "buy"}


def build_alert_message(
    *,
    combined: pd.DataFrame,
    source_kind: str,
    mode: str,
    require_body_weight: bool,
    dashboard_url: str,
    max_races: int,
) -> tuple[str, dict[str, int]]:
    race_col = race_id_col(combined)
    if combined.empty or not race_col:
        return "Keiba AI\n出馬表データがありません。", {"buy": 0, "candidate": 0, "wait": 0, "skip": 0, "sent": 0}

    rows: list[tuple[pd.DataFrame, dict[str, Any]]] = []
    counts = {"buy": 0, "candidate": 0, "wait": 0, "skip": 0, "sent": 0}
    for _, race in combined.groupby(race_col, sort=False):
        dec = race_decision(race, source_kind)
        counts[dec["class"]] += 1
        if should_include(dec, mode, require_body_weight):
            rows.append((race, dec))

    def order_key(item: tuple[pd.DataFrame, dict[str, Any]]) -> tuple[int, float]:
        priority = {"buy": 0, "candidate": 1, "wait": 2, "skip": 3}.get(item[1]["class"], 9)
        return priority, -float(item[1]["metrics"].get("confidence", 0.0))

    rows = sorted(rows, key=order_key)[:max_races]
    counts["sent"] = len(rows)

    source_label = "TARGET公式" if source_kind == "target" else "外部暫定" if source_kind == "external" else "未検出"
    title = {
        "preday": "前日候補",
        "buyable": "買えそう判定",
        "skip": "見送り候補",
        "all": "全体サマリ",
        "final": "最終買い目",
    }.get(mode, "候補")
    lines = [
        f"Keiba AI {title}",
        f"データ: {source_label}",
        f"買い候補 {counts['buy']} / 暫定 {counts['candidate']} / 保留 {counts['wait']} / 見送り {counts['skip']}",
    ]
    if require_body_weight:
        lines.append("条件: 馬体重確認済みのみ")
    if dashboard_url:
        lines.append(f"画面: {dashboard_url}")
    lines.append("")

    if not rows:
        lines.append("通知対象レースはありません。")
    for i, (race, dec) in enumerate(rows, 1):
        top = top_runner(race)
        metrics = dec["metrics"]
        waits = "、".join(dec.get("waits") or ["なし"])
        positives = "、".join(dec.get("positives") or [])
        reasons = "、".join(dec.get("reasons") or [])
        lines.extend(
            [
                f"{i}. 【{dec['action']}】{race_label(race)}",
                f"軸候補: {display_horse(top)}",
                f"信頼 {metrics['confidence']:.3f} / AI差 {metrics['gap']:.3f} / 履歴 {metrics['previous_rate']:.0%}",
                f"材料: {positives}",
                f"待ち: {waits}",
                f"注意: {reasons}",
                "",
            ]
        )

    message = "\n".join(lines).strip()
    # LINE text message limit is generous enough for the usual shortlist, but keep it tidy.
    if len(message) > 4500:
        message = message[:4450].rstrip() + "\n\n...（長いため省略）"
    return message, counts


def ticket_value(row: pd.Series, names: list[str], default: object = "") -> object:
    for name in names:
        if name in row.index and has_value(row.get(name)):
            return row.get(name)
    return default


def yen(value: object) -> str:
    amount = number(value, 0.0)
    return f"{int(round(amount)):,}円" if amount else "-"


def ticket_type_label(value: object) -> str:
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

def normalize_ticket_type_for_lap(value: object) -> str:
    raw = text(value).lower()
    if raw in {"quinella", "umaren"}:
        return "umaren"
    if raw == "wide":
        return "wide"
    return raw


def horse_lap_decomp_fields(row: pd.Series) -> dict[str, object]:
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
    overlay_path = project_path(overlay_csv)
    if not overlay_path.exists():
        return frame
    overlay = read_csv_safe(overlay_path)
    required = {"race_id", "ticket_type", "anchor_no", "partner_no"}
    if overlay.empty or not required.issubset(overlay.columns):
        return frame
    lookup: dict[tuple[str, str, int, int], dict[str, object]] = {}
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
        a_no = int(number(ticket_value(row, ["anchor_no", "horse_a", "a_no"]), 0))
        b_no = int(number(ticket_value(row, ["partner_no", "horse_b", "b_no"]), 0))
        if not race_id or not a_no or not b_no:
            continue
        lo, hi = sorted([a_no, b_no])
        for key, value in lookup.get((race_id, ticket_type, lo, hi), {}).items():
            out.at[idx, key] = value
    return out


def humanize_reason_summary(reason: object) -> str:
    raw = text(reason)
    if not raw:
        return "購入条件を通過"
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
    gelding = re.search(r"gelding=([a-z0-9_]+)", lower)
    if gelding:
        label = {
            "gelding_risk": "去勢明けリスク確認",
            "gelding_second_upside": "去勢2戦目の上積み候補",
            "gelding_third_underneath": "去勢3戦目で相手候補",
            "gelding_debut_watch": "去勢明け初戦は慎重評価",
            "gelding_unknown_timing": "セ馬の手術時期は不明",
        }.get(gelding.group(1), "")
        if label:
            parts.append(label)
    if parts:
        return " / ".join(parts)
    return raw.replace("|", " / ").replace("_", " ")


def queue_lap_reason_summary(row: pd.Series) -> str:
    title = text(ticket_value(row, ["queueLapTitle", "queue_lap_title"], ""))
    label = text(ticket_value(row, ["queueLapLabel", "queue_lap_label"], ""))
    note = text(ticket_value(row, ["queueLapNote", "queue_lap_note"], ""))
    priority = number(ticket_value(row, ["queueLapPriority", "queue_lap_priority"], 0), 0.0)
    horse_lap_note = text(ticket_value(row, ["horse_lap_decomp_runtime_note"], ""))
    horse_lap_score = number(ticket_value(row, ["horse_lap_decomp_runtime_score"], 0), 0.0)
    if horse_lap_note:
        score = f" {horse_lap_score:.2f}" if horse_lap_score > 0 else ""
        return f"{horse_lap_note}{score}"
    lap_advanced_label = text(ticket_value(row, ["lap_advanced_shadow_label"], ""))
    lap_advanced_note = text(ticket_value(row, ["lap_advanced_shadow_note"], ""))
    lap_advanced_score = number(ticket_value(row, ["lap_advanced_combo_score"], 0), 0.0)
    lap_track_label = text(ticket_value(row, ["lap_track_shadow_label"], ""))
    lap_track_note = text(ticket_value(row, ["lap_track_shadow_note"], ""))
    lap_track_score = number(ticket_value(row, ["lap_track_shadow_score"], 0), 0.0)
    lap_advanced_map = {
        "lap_role_goodrun_strong": "ラップロール強: 想定ラップと前受け/差し役割がかみ合う",
        "goodrun_lap_strong": "好走時ラップ適性強: 好走時のラップ型と今回想定が近い",
        "lap_advanced_combo_strong": "ラップ総合強: 波形・好走時ラップ・役割が揃う",
        "lap_advanced_combo_watch": "ラップ総合注視: ラップ面は準候補",
    }
    if not label and lap_advanced_label in lap_advanced_map:
        label = lap_advanced_label
        title = lap_advanced_map[lap_advanced_label].split(":", 1)[0]
        note = lap_advanced_note or lap_advanced_map[lap_advanced_label]
        priority = lap_advanced_score
    if not label:
        anchor_front = max(
            number(ticket_value(row, ["anchor_projected_front5_prob", "anchor_front5_model_prob"], 0), 0.0),
            0.0,
        )
        partner_front = max(
            number(ticket_value(row, ["partner_projected_front5_prob", "partner_front5_model_prob"], 0), 0.0),
            0.0,
        )
        pair_front = number(ticket_value(row, ["projected_front5_prob", "ticket_front_position_reliability_score"], 0), 0.0)
        queue_clarity = max(
            number(ticket_value(row, ["race_queue_clarity_score", "anchor_runtime_queue_clarity_score"], 0), 0.0),
            number(ticket_value(row, ["partner_runtime_queue_clarity_score"], 0), 0.0),
        )
        duel_risk = max(
            number(ticket_value(row, ["race_front_duel_risk_score", "anchor_runtime_front_duel_risk_score"], 0), 0.0),
            number(ticket_value(row, ["partner_runtime_front_duel_risk_score"], 0), 0.0),
        )
        pair_front_both = anchor_front >= 0.60 and partner_front >= 0.60
        pair_front_any = pair_front_both or max(anchor_front, partner_front, pair_front) >= 0.60
        if pair_front_both:
            title = "\u524d\u76ee\u30da\u30a2\u5f37"
            note = "\u30da\u30a22\u982d\u3068\u3082\u524d\u76ee\u5019\u88dc"
            priority = 0.35
        elif pair_front_any and queue_clarity >= 0.62:
            title = "\u968a\u5217\u8aad\u307f\u826f\u597d"
            note = "\u524d\u76ee\u5019\u88dc\u304c\u3044\u3066\u968a\u5217\u3082\u8aad\u307f\u3084\u3059\u3044"
            priority = 0.30
        elif duel_risk >= 0.62:
            title = "\u968a\u5217\u6df7\u6226\u6ce8\u8996"
            note = "\u524d\u306e\u7af6\u308a\u5408\u3044\u304c\u5f37\u304f\u30ef\u30a4\u30c9\u306f\u6ce8\u8996"
            priority = 0.12
    if not (title or note):
        lap_track_map = {
            "positive": "馬場×ラップ強: 馬場状態とラップ適性の組み合わせが良い",
            "positive_soft": "馬場×ラップ注視: サンプルは小さいが条件は良い",
            "caution": "馬場×ラップ参考: 年別ブレがあり買い昇格ではなく注視",
        }
        if lap_track_label in lap_track_map:
            score = f" {lap_track_score:.2f}" if lap_track_score > 0 else ""
            return f"{lap_track_note or lap_track_map[lap_track_label]}{score}"
        return ""
    score = f" {priority:.2f}" if priority > 0 else ""
    out = f"{title or note}{score}" + (f": {note}" if title and note and title not in note else "")
    lap_track_map = {
        "positive": "馬場×ラップ強",
        "positive_soft": "馬場×ラップ注視",
        "caution": "馬場×ラップ参考",
    }
    if lap_track_label in lap_track_map:
        track_score = f" {lap_track_score:.2f}" if lap_track_score > 0 else ""
        out = f"{out} / {lap_track_note or lap_track_map[lap_track_label]}{track_score}"
    return out


def odds_text(value: object) -> str:
    val = number(value, 0.0)
    return f"{val:.1f}倍" if val else "-"


def build_race_label_lookup(entry: pd.DataFrame) -> dict[str, str]:
    if entry.empty:
        return {}
    race_col = race_id_col(entry)
    if not race_col:
        return {}

    lookup: dict[str, str] = {}
    for race_id, race in entry.groupby(race_col, sort=False):
        label = race_label(race)
        for key in body_race_keys(race_id):
            lookup[key] = label
    return lookup


def ticket_race_label(row: pd.Series, race_lookup: dict[str, str]) -> str:
    raw_id = ticket_value(row, ["race_id", "レースID(新/馬番無)"], "")
    for key in body_race_keys(raw_id):
        if key in race_lookup:
            return race_lookup[key]

    date = text(ticket_value(row, ["日付", "date_key", "_date"]))
    venue = text(ticket_value(row, ["venue", "場所", "venue_eval"]))
    race_no = text(ticket_value(row, ["Ｒ", "R"]))
    race_name = text(ticket_value(row, ["レース名", "race_class_name"]))
    race = " ".join(v for v in [date, f"{venue}{race_no}R" if venue or race_no else "", race_name] if v)
    return race or text(raw_id)


def default_entry_csv_from_dashboard_summary() -> Path | None:
    summary_path = project_path("outputs/ui/live_odds_dashboard.summary.json")
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    entry_csv = summary.get("entry_csv")
    if not entry_csv:
        return None
    path = Path(entry_csv)
    if not path.is_absolute():
        path = project_path(path)
    return path if path.exists() else None


def read_live_dashboard_payload(path: Path | None = None) -> dict[str, Any]:
    dashboard_path = path or project_path("outputs/ui/live_odds_dashboard.html")
    if not dashboard_path.exists():
        return {}
    html = dashboard_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"const payload = (.*?);\n", html)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def dashboard_shadow_message(
    payload: dict[str, Any],
    dashboard_url: str,
    max_races: int,
) -> tuple[str, dict[str, int]]:
    counts = payload.get("counts") or {}
    shadow_rows = payload.get("shadowTicketRows") or []
    races = {str(r.get("raceId")): r for r in payload.get("races") or []}
    rows = sorted(
        shadow_rows,
        key=lambda r: (
            -float(r.get("jointEv") or 0),
            -float(r.get("expectedRoi") or 0),
            str(r.get("raceId") or ""),
        ),
    )[:max_races]

    lines = [
        "Keiba AI 現在の候補",
        f"正式BUY {int(counts.get('ticketRaces') or 0)}R / V2参考 {int(counts.get('shadowTicketRaces') or 0)}R",
        "正式BUYは0件のため、以下は購入対象外の参考候補です。",
    ]
    if dashboard_url:
        lines.append(f"画面: {dashboard_url}")
    lines.append("")

    if not rows:
        lines.append("V2参考候補もありません。現時点では見送り中心です。")
    for i, row in enumerate(rows, 1):
        race = races.get(str(row.get("raceId"))) or {}
        race_label_text = " ".join(
            x
            for x in [
                str(race.get("dateLabel") or "").strip(),
                f"{race.get('venue', '')}{race.get('raceNo', '')}R".strip(),
                str(race.get("raceName") or "").strip(),
            ]
            if x
        ) or str(row.get("raceId") or "")
        combo = f"{row.get('aNo', '')}-{row.get('bNo', '')} {row.get('aName', '')}-{row.get('bName', '')}".strip()
        odds = number(row.get("liveOdds"), 0.0)
        joint_prob = number(row.get("jointProb"), 0.0)
        expected_roi = number(row.get("expectedRoi"), 0.0)
        fail = text(row.get("failReasonSummary")) or text(row.get("reason"))
        lines.extend(
            [
                f"{i}. 【V2参考・非購入】{race_label_text}",
                f"ワイド: {combo}",
                f"目安 {odds_text(odds)} / 同時好走 {joint_prob:.1%} / ROI {expected_roi:.2f}",
                f"買えない理由: {fail}",
                "",
            ]
        )

    message = "\n".join(lines).strip()
    if len(message) > 4500:
        message = message[:4450].rstrip() + "\n\n...（長いため省略）"
    return message, {
        "tickets": int(counts.get("ticketRows") or 0),
        "races": int(counts.get("ticketRaces") or 0),
        "shadow_races": int(counts.get("shadowTicketRaces") or 0),
        "shadow_rows": int(counts.get("shadowTicketRows") or 0),
    }


def final_ticket_message(
    tickets: pd.DataFrame,
    dashboard_url: str,
    max_races: int,
    entry: pd.DataFrame | None = None,
) -> tuple[str, dict[str, int]]:
    if tickets.empty:
        payload = read_live_dashboard_payload()
        if payload:
            return dashboard_shadow_message(payload, dashboard_url, max_races)
        return "Keiba AI 最終買い目\n買い目CSVが空です。", {"tickets": 0, "races": 0, "stake_yen": 0}

    work = tickets.copy()
    race_lookup = build_race_label_lookup(entry if entry is not None else pd.DataFrame())
    if "runtime_action" in work.columns:
        buy_mask = work["runtime_action"].astype(str).str.upper().eq("BUY")
        if buy_mask.any():
            work = work[buy_mask].copy()
    elif "dashboard_decision_label" in work.columns:
        buy_mask = work["dashboard_decision_label"].astype(str).str.upper().eq("BUY")
        if buy_mask.any():
            work = work[buy_mask].copy()

    if "runtime_expected_roi" in work.columns:
        work["_sort_roi"] = pd.to_numeric(work["runtime_expected_roi"], errors="coerce").fillna(0.0)
    elif "expected_roi_after_slippage" in work.columns:
        work["_sort_roi"] = pd.to_numeric(work["expected_roi_after_slippage"], errors="coerce").fillna(0.0)
    else:
        work["_sort_roi"] = 0.0
    stake_col = pick_col(work, ["runtime_stake_yen", "scaled_stake_yen", "stake_yen", "eval_stake_yen"])
    if stake_col:
        work["_sort_stake"] = pd.to_numeric(work[stake_col], errors="coerce").fillna(0.0)
    else:
        work["_sort_stake"] = 0.0

    work = work.sort_values(["_sort_stake", "_sort_roi"], ascending=[False, False]).head(max_races)
    race_col = pick_col(work, ["race_id", "レースID(新/馬番無)"]) or "race_id"
    total_stake = int(round(work["_sort_stake"].sum())) if "_sort_stake" in work else 0
    race_count = int(work[race_col].nunique()) if race_col in work.columns else 0
    lines = [
        "Keiba AI 最終買い目",
        f"買い目 {len(work)}点 / レース {race_count}R / 予定金額 {total_stake:,}円",
    ]
    if dashboard_url:
        lines.append(f"画面: {dashboard_url}")
    lines.append("")

    for i, (_, row) in enumerate(work.iterrows(), 1):
        date = text(ticket_value(row, ["日付S", "date_key", "_date"]))
        venue = text(ticket_value(row, ["venue", "場所", "venue_eval"]))
        race_no = text(ticket_value(row, ["Ｒ", "R"]))
        race_name = text(ticket_value(row, ["レース名", "race_class_name"]))
        race = ticket_race_label(row, race_lookup)
        ticket_type = ticket_type_label(ticket_value(row, ["ticket_type"], "ticket"))
        anchor = text(ticket_value(row, ["anchor_name", "anchor_horse_name", "a_horse"], "軸"))
        partner = text(ticket_value(row, ["partner_name", "partner_horse_name", "b_horse"], "相手"))
        third = text(ticket_value(row, ["third_name", "third_horse"], ""))
        stake = ticket_value(row, ["runtime_stake_yen", "scaled_stake_yen", "stake_yen", "eval_stake_yen"], 0)
        live_odds = ticket_value(row, ["runtime_odds", "live_odds", "quote_odds_proxy"], 0)
        exp_roi = number(ticket_value(row, ["runtime_expected_roi", "expected_roi_after_slippage", "wide_ev_proxy", "umaren_ev_proxy"], 0), 0.0)
        hit_prob = number(ticket_value(row, ["pair_calibrated_hit_prob", "ticket_hit_prob", "wide_hit_prob_cal", "umaren_hit_prob_cal"], 0), 0.0)
        reason = text(ticket_value(row, ["runtime_reason", "buy_reason_summary", "risk_reason_summary"], ""))
        reason_text = humanize_reason_summary(reason)
        queue_lap_reason = queue_lap_reason_summary(row)
        if queue_lap_reason:
            reason_text = f"{reason_text} / {queue_lap_reason}"
        horses = f"{anchor} - {partner}" + (f" - {third}" if third else "")
        lines.extend(
            [
                f"{i}. {race}",
                f"{ticket_type}: {horses}",
                f"金額 {yen(stake)} / 目安 {odds_text(live_odds)} / hit {hit_prob:.1%} / ROI {exp_roi:.2f}",
                f"理由: {reason_text}",
                "",
            ]
        )

    message = "\n".join(lines).strip()
    if len(message) > 4500:
        message = message[:4450].rstrip() + "\n\n...（長いため省略）"
    return message, {"tickets": int(len(work)), "races": race_count, "stake_yen": total_stake}


def send_line_push(message: str, token: str, to: str) -> dict[str, Any]:
    payload = {
        "to": to,
        "messages": [{"type": "text", "text": message}],
    }
    request = urllib.request.Request(
        LINE_PUSH_ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {"ok": True, "status": response.status, "body": body}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "body": body}
    except urllib.error.URLError as exc:
        return {"ok": False, "status": None, "body": str(exc.reason)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Send Keiba AI race candidate alerts to LINE Messaging API.")
    parser.add_argument("--entry-csv", default=None)
    parser.add_argument("--target-entry-csv", default="data/datasets/inference/weekly/entry_snapshot.csv")
    parser.add_argument("--fallback-entry-glob", default="data/datasets/inference/weekly/entry_snapshot_netkeiba_*_enriched.csv")
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument("--today", default=None, help="YYYY-MM-DD. Defaults to local date.")
    parser.add_argument("--mode", choices=["preday", "buyable", "skip", "all", "final"], default="preday")
    parser.add_argument("--require-body-weight", action="store_true")
    parser.add_argument("--body-weight-csv", default="")
    parser.add_argument("--tickets-csv", default="")
    parser.add_argument(
        "--horse-lap-decomp-csv",
        default="outputs/analysis/horse_lap_aptitude_decomposition_v1/all_ticket_lap_decomposition_compact.csv",
    )
    parser.add_argument("--dashboard-url", default="")
    parser.add_argument("--max-races", type=int, default=8)
    parser.add_argument("--output-text", default="outputs/notifications/line_keiba_alert_latest.txt")
    parser.add_argument("--send", action="store_true", help="Actually send. Default is dry-run.")
    parser.add_argument("--line-token-env", default="LINE_CHANNEL_ACCESS_TOKEN")
    parser.add_argument("--line-to-env", default="LINE_USER_ID")
    args = parser.parse_args()

    if args.mode == "final":
        if not args.tickets_csv:
            raise SystemExit("--tickets-csv is required when --mode final")
        tickets_path = project_path(args.tickets_csv)
        tickets = read_csv_safe(tickets_path) if tickets_path.exists() else pd.DataFrame()
        tickets = enrich_horse_lap_decomp(tickets, args.horse_lap_decomp_csv)
        final_entry_path = project_path(args.entry_csv) if args.entry_csv else default_entry_csv_from_dashboard_summary()
        final_entry = read_csv_safe(final_entry_path) if final_entry_path and final_entry_path.exists() else pd.DataFrame()
        message, counts = final_ticket_message(tickets, args.dashboard_url, args.max_races, final_entry)
        output_path = project_path(args.output_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(message + "\n", encoding="utf-8")
        result: dict[str, Any] = {
            "dry_run": not args.send,
            "mode": args.mode,
            "tickets_csv": str(tickets_path),
            "entry_csv": str(final_entry_path) if final_entry_path else "",
            "counts": counts,
            "output_text": str(output_path),
        }
        if args.send:
            token = os.environ.get(args.line_token_env, "")
            to = os.environ.get(args.line_to_env, "")
            if not token or not to:
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
            result["line"] = send_line_push(message, token, to)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("\n--- message preview ---")
        print(message)
        return

    today = datetime.strptime(args.today, "%Y-%m-%d") if args.today else datetime.now()
    entry_path, snapshot, source_kind, source_reason = choose_entry_source(
        explicit_entry_csv=args.entry_csv,
        target_entry_csv=args.target_entry_csv,
        fallback_entry_glob=args.fallback_entry_glob,
        today=today,
    )
    predictions_dir = choose_predictions_dir(args.predictions_dir, source_kind)
    pred_path = latest_prediction(predictions_dir)
    prediction = read_csv_safe(pred_path) if pred_path else None
    combined = merge_prediction(snapshot, prediction)
    combined = merge_body_weight(combined, args.body_weight_csv)
    message, counts = build_alert_message(
        combined=combined,
        source_kind=source_kind,
        mode=args.mode,
        require_body_weight=args.require_body_weight,
        dashboard_url=args.dashboard_url,
        max_races=args.max_races,
    )

    output_path = project_path(args.output_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(message + "\n", encoding="utf-8")

    result: dict[str, Any] = {
        "dry_run": not args.send,
        "entry_csv": str(entry_path),
        "prediction_csv": str(pred_path) if pred_path else "",
        "source_kind": source_kind,
        "source_reason": source_reason,
        "mode": args.mode,
        "counts": counts,
        "output_text": str(output_path),
    }
    if args.send:
        token = os.environ.get(args.line_token_env, "")
        to = os.environ.get(args.line_to_env, "")
        if not token or not to:
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
        result["line"] = send_line_push(message, token, to)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n--- message preview ---")
    print(message)


if __name__ == "__main__":
    main()
