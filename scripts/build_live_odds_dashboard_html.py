from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.gelding_transition import (  # noqa: E402
    enrich_current_entries_with_gelding_context,
    read_csv_any,
)
from src.features.odds_timeline import clean_odds_value  # noqa: E402

VENUES = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def read_csv_safe(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def read_json_safe(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_gelding_history(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    history = read_csv_any(path)
    if "race_date" in history.columns:
        history["race_date"] = pd.to_datetime(history["race_date"], errors="coerce")
    return history


def latest_file(pattern: str) -> Path | None:
    files = [Path(p) for p in glob_paths(pattern)]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def glob_paths(pattern: str) -> list[str]:
    return [str(p) for p in ROOT.glob(pattern)]


def text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


def num(value: object) -> float | None:
    try:
        if value is None or pd.isna(value) or text(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def odds_num(value: object) -> float | None:
    return clean_odds_value(value)


def int_text(value: object) -> str:
    value_num = num(value)
    if value_num is None:
        return ""
    return str(int(value_num))


def signed_int_text(value: object) -> str:
    value_num = num(value)
    if value_num is None:
        return ""
    value_i = int(value_num)
    return f"{value_i:+d}" if value_i else "0"


def body_weight_label(weight: object, diff: object) -> str:
    weight_text = int_text(weight)
    if not weight_text:
        return ""
    diff_text = signed_int_text(diff)
    return f"{weight_text}({diff_text})" if diff_text else weight_text


def carried_weight_label(weight: object) -> str:
    weight_num = num(weight)
    if weight_num is None:
        return ""
    if abs(weight_num - round(weight_num)) < 0.05:
        return f"{int(round(weight_num))}kg"
    return f"{weight_num:.1f}kg"


def normalize_surface(value: object) -> str:
    raw = text(value).strip()
    if raw.startswith("芝") or raw.lower() in {"turf", "grass"}:
        return "芝"
    if raw.startswith("ダ") or raw.lower() in {"dirt", "sand"}:
        return "ダ"
    if raw.startswith("障"):
        return "障"
    return raw


def parse_date_key(value: object) -> str:
    raw = text(value).strip()
    if not raw:
        return ""
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y%m%d", "%y%m%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y%m%d")
        except Exception:
            pass
    if raw.isdigit() and len(raw) == 6:
        try:
            return datetime.strptime("20" + raw, "%Y%m%d").strftime("%Y%m%d")
        except Exception:
            return ""
    return ""


def read_dashboard_default_date() -> str:
    payload = read_json_safe(project_path("outputs/runtime/current_dashboard_inputs.json"))
    return parse_date_key(payload.get("date"))


def format_date(date_key: str) -> str:
    try:
        dt = datetime.strptime(date_key, "%Y%m%d")
        return f"{dt.year}年{dt.month}月{dt.day}日"
    except Exception:
        return date_key


def format_snapshot(value: str) -> str:
    raw = text(value).strip()
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y/%m/%d %H:%M:%S")
        except Exception:
            pass
    return raw


def load_track_condition_map(track_condition_csv: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not track_condition_csv or not track_condition_csv.exists():
        return {}
    track = read_csv_safe(track_condition_csv, dtype=str)
    required = {"effective_date", "venue", "turf_going", "dirt_going"}
    if track.empty or not required.issubset(track.columns):
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in track.iterrows():
        date_key = parse_date_key(row.get("effective_date"))
        venue = text(row.get("venue")).strip()
        if not date_key or not venue:
            continue
        out[(date_key, venue)] = {
            "turfGoing": text(row.get("turf_going")).strip(),
            "dirtGoing": text(row.get("dirt_going")).strip(),
            "trackWeather": text(row.get("weather")).strip(),
            "trackFetchedAt": format_snapshot(text(row.get("fetched_at")).strip()),
            "trackSource": text(row.get("source_url")).strip(),
        }
    return out


def load_result_track_condition_map(result_track_csv: Path | None) -> dict[str, dict[str, Any]]:
    if not result_track_csv or not result_track_csv.exists():
        return {}
    result_track = read_csv_safe(result_track_csv, dtype=str)
    if result_track.empty or "race_id" not in result_track.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in result_track.iterrows():
        race_id = text(row.get("race_id")).strip()
        if not race_id:
            continue
        out[race_id] = {
            "turfGoing": text(row.get("turf_going")).strip(),
            "dirtGoing": text(row.get("dirt_going")).strip(),
            "runtimeGoing": text(row.get("runtime_going")).strip(),
            "trackWeather": text(row.get("weather")).strip(),
            "trackFetchedAt": format_snapshot(text(row.get("fetched_at")).strip()),
            "trackSource": text(row.get("source")).strip() or "jra_result",
            "trackConditionSource": "result",
        }
    return out


def apply_track_condition_to_races(
    races: dict[str, dict[str, Any]],
    track_condition_map: dict[tuple[str, str], dict[str, Any]],
    result_track_condition_map: dict[str, dict[str, Any]] | None = None,
) -> None:
    result_track_condition_map = result_track_condition_map or {}
    for race in races.values():
        date_key = text(race.get("dateKey")).strip()
        venue = text(race.get("venue")).strip()
        result_condition = result_track_condition_map.get(text(race.get("raceId")).strip())
        condition = track_condition_map.get((date_key, venue), {})
        race.update(
            {
                "turfGoing": condition.get("turfGoing", ""),
                "dirtGoing": condition.get("dirtGoing", ""),
                "trackWeather": condition.get("trackWeather", ""),
                "trackFetchedAt": condition.get("trackFetchedAt", ""),
                "trackSource": condition.get("trackSource", ""),
                "trackConditionSource": "current" if condition else "",
            }
        )
        surface = normalize_surface(race.get("surface"))
        runtime_going = ""
        if surface == "芝":
            runtime_going = condition.get("turfGoing", "")
        elif surface == "ダ":
            runtime_going = condition.get("dirtGoing", "")
        race["runtimeGoing"] = runtime_going
        race["trackConditionAvailable"] = bool(runtime_going or condition.get("turfGoing") or condition.get("dirtGoing"))
        if result_condition:
            race.update(result_condition)
            race["trackConditionAvailable"] = bool(
                race.get("runtimeGoing") or race.get("turfGoing") or race.get("dirtGoing")
            )


def parse_official_race_id(race_id: object) -> dict[str, Any]:
    rid = text(race_id).strip()
    if rid.endswith(".0"):
        rid = rid[:-2]
    rid = rid.zfill(16) if rid.isdigit() and len(rid) <= 16 else rid
    if not (rid.isdigit() and len(rid) == 16):
        return {
            "raceId": rid,
            "dateKey": "",
            "dateLabel": "",
            "venueCode": "",
            "venue": "",
            "kaiji": "",
            "nichiji": "",
            "raceNo": 0,
            "raceLabel": rid,
        }
    date_key = rid[:8]
    venue_code = rid[8:10]
    race_no = int(rid[14:16])
    venue = VENUES.get(venue_code, venue_code)
    return {
        "raceId": rid,
        "dateKey": date_key,
        "dateLabel": format_date(date_key),
        "venueCode": venue_code,
        "venue": venue,
        "kaiji": rid[10:12],
        "nichiji": rid[12:14],
        "raceNo": race_no,
        "raceLabel": f"{format_date(date_key)} {venue}{race_no}R",
    }


def row_to_official_race_id(row: pd.Series) -> str:
    source_url = text(row.get("source_url"))
    match = re.search(r"race_id=(\d{12})", source_url)
    netkeiba_id = match.group(1) if match else ""
    if not netkeiba_id:
        raw_id = text(row.get("race_id") or row.get("レースID(新/馬番無)") or "")
        digits = re.sub(r"\D", "", raw_id)
        if len(digits) == 16:
            return digits
        if len(digits) >= 8 and len(digits) != 12:
            date_key = parse_date_key(row.get("日付S") or row.get("日付"))
            if not date_key:
                return ""
            tail = digits[-8:]
            venue = digits[:-8].zfill(2)
            return date_key + venue + tail[2:8]
        if len(digits) >= 12:
            netkeiba_id = digits[:12]
    date_key = parse_date_key(row.get("日付S") or row.get("日付"))
    if not date_key or not netkeiba_id or len(netkeiba_id) < 12:
        return ""
    return date_key + netkeiba_id[4:12]


def _put_if_present(target: dict[str, Any], key: str, value: object) -> None:
    if has_value := (value is not None and not pd.isna(value) and text(value).strip() != ""):
        target[key] = value


def _runner_points(info: dict[str, Any]) -> list[str]:
    points: list[str] = []
    ai_rank = num(info.get("aiRank"))
    ai_score = num(info.get("aiScore"))
    front = num(info.get("frontRunning"))
    closing = num(info.get("closing"))
    gap = num(info.get("scoreGap"))
    first_impact = num(info.get("firstConditionImpact"))
    if ai_rank is not None and ai_rank <= 3:
        points.append("AI上位評価")
    if first_impact is not None and first_impact >= 0.60:
        points.append("少キャリアでも前走内容強い")
    if ai_score is not None and ai_score >= 0.44:
        points.append("基礎スコア高め")
    if gap is not None and gap >= 0.006:
        points.append("AI差あり")
    if front is not None and front >= 0.55:
        points.append("前に行ける形")
    if closing is not None and closing >= 0.55:
        points.append("差し脚評価")
    if text(info.get("expectedPace")) == "slow" and front is not None and front >= 0.45:
        points.append("スロー想定で位置利")
    course_ten_note = text(info.get("courseTenRunnerNote"))
    course_ten_speed = num(info.get("courseTenSpeed"))
    if "テン速い" in course_ten_note or (course_ten_speed is not None and course_ten_speed >= 0.45):
        points.append("クラス×コース基準でテン速い")
    gelding_note = text(info.get("geldingNote"))
    if "2戦目" in gelding_note:
        points.append("去勢2戦目で上積み候補")
    elif "3戦目" in gelding_note:
        points.append("去勢3戦目で相手候補")
    return points[:4] or ["前日材料で暫定評価"]


def _runner_concerns(info: dict[str, Any]) -> list[str]:
    concerns: list[str] = []
    bucket = text(info.get("confidenceBucket"))
    pressure = num(info.get("pressureScore"))
    ai_rank = num(info.get("aiRank"))
    front = num(info.get("frontRunning"))
    first_uncertainty = num(info.get("firstConditionUncertainty"))
    if bucket == "low":
        concerns.append("AI信頼度は低め")
    if first_uncertainty is not None and first_uncertainty >= 0.45:
        concerns.append("少キャリア/初条件の不確実性")
    if pressure is not None and pressure >= 0.75 and front is not None and front >= 0.55:
        concerns.append("先行負荷に注意")
    course_ten_note = text(info.get("courseTenRunnerNote"))
    course_ten_speed = num(info.get("courseTenSpeed"))
    if "テン不足" in course_ten_note or (course_ten_speed is not None and course_ten_speed <= -0.45):
        concerns.append("クラス×コース基準ではテン不足")
    if ai_rank is not None and ai_rank >= 8:
        concerns.append("AI評価は下位")
    gelding_note = text(info.get("geldingNote"))
    if "強く割引" in gelding_note or "人気薄は割引" in gelding_note:
        concerns.append(gelding_note)
    elif "去勢明け初戦" in gelding_note:
        concerns.append("去勢明け初戦は慎重")
    return concerns[:3]


def build_entry_map(
    entry_csv: Path | None,
    prediction_csv: Path | None = None,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    horse_map: dict[tuple[str, int], dict[str, Any]] = {}
    race_meta: dict[str, dict[str, Any]] = {}

    sources: list[tuple[Path | None, bool]] = [(entry_csv, False), (prediction_csv, True)]
    for csv_path, is_prediction in sources:
        if csv_path is None or not csv_path.exists():
            continue
        frame = read_csv_safe(csv_path)
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            race_id = row_to_official_race_id(row)
            horse_no = num(row.get("馬番") if "馬番" in row.index else row.get("horse_no"))
            if not race_id or horse_no is None:
                continue
            horse_no_i = int(horse_no)
            info = horse_map.setdefault((race_id, horse_no_i), {})
            for key, value in {
                "horseName": row.get("馬名") if "馬名" in row.index else row.get("horse_name"),
                "jockey": row.get("騎手") if "騎手" in row.index else row.get("jockey"),
                "carriedWeight": num(row.get("斤量") if "斤量" in row.index else row.get("weight_carried")),
                "frameNo": int_text(row.get("枠番") if "枠番" in row.index else row.get("frame_no")),
                "popularitySnapshot": int_text(row.get("人気") if "人気" in row.index else ""),
                "predayWinOdds": odds_num(row.get("odds_latest_win") if "odds_latest_win" in row.index else row.get("単勝オッズ")),
            }.items():
                if value not in ("", None) and not pd.isna(value):
                    info[key] = value
            for key, value in {
                "geldingPhase": row.get("gelding_phase"),
                "geldingRisk": num(row.get("gelding_risk_score")),
                "geldingValue": num(row.get("gelding_value_score")),
                "geldingNote": row.get("gelding_context_note"),
            }.items():
                if value not in ("", None) and not pd.isna(value):
                    info[key] = value
            if is_prediction:
                for key, value in {
                    "aiRank": num(row.get("ai_rank")),
                    "aiScore": num(row.get("ai_score")),
                    "expectedPace": text(row.get("expected_pace")),
                    "frontRunning": num(row.get("front_running_tendency")),
                    "closing": num(row.get("closing_tendency")),
                    "scoreGap": num(row.get("ai_score_gap_to_second")),
                    "confidence": num(row.get("ai_confidence_score")),
                    "confidenceBucket": text(row.get("ai_confidence_bucket")),
                    "pressureScore": num(row.get("race_early_pressure_score")),
                }.items():
                    if value not in ("", None) and not pd.isna(value):
                        info[key] = value
                info["points"] = _runner_points(info)
                info["concerns"] = _runner_concerns(info)

            meta = race_meta.setdefault(race_id, {})
            for key, value in {
                "raceName": row.get("レース名") if "レース名" in row.index else row.get("race_name"),
                "className": row.get("クラス名") if "クラス名" in row.index else row.get("class_name"),
                "startTime": row.get("発走時刻") if "発走時刻" in row.index else row.get("start_time"),
                "surface": row.get("芝・ダ") if "芝・ダ" in row.index else row.get("surface"),
                "distance": int_text(row.get("距離") if "距離" in row.index else row.get("distance")),
                "fieldSize": int_text(row.get("頭数") if "頭数" in row.index else row.get("field_size")),
            }.items():
                if not meta.get(key) and value not in ("", None) and not pd.isna(value):
                    meta[key] = value
            if is_prediction:
                top_rank = num(row.get("ai_rank"))
                if top_rank == 1 or "expectedPace" not in meta:
                    for key, value in {
                        "expectedPace": text(row.get("expected_pace")),
                        "frontRunnerCount": num(row.get("race_front_runner_count")),
                        "pressureScore": num(row.get("race_early_pressure_score")),
                        "confidence": num(row.get("ai_confidence_score")),
                        "scoreGap": num(row.get("ai_score_gap_to_second")),
                    }.items():
                        if value not in ("", None) and not pd.isna(value):
                            meta[key] = value
    return horse_map, race_meta


def row_horse_info(horse_map: dict[tuple[str, int], dict[str, Any]], race_id: str, horse_no: int) -> dict[str, Any]:
    return horse_map.get((race_id, horse_no), {})


def first_value(row: pd.Series, names: list[str]) -> object:
    for name in names:
        if name in row.index:
            value = row.get(name)
            if value is not None and not pd.isna(value) and text(value).strip() != "":
                return value
    return pd.NA


def first_number(row: pd.Series, names: list[str]) -> float | None:
    for name in names:
        if name in row.index:
            value = num(row.get(name))
            if value is not None:
                return value
    return None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def race_upset_forecast(race: dict[str, Any], runners: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate race volatility from pre-race information only."""

    score_value = 0.45
    reasons: list[str] = []

    field_size = num(race.get("fieldSize"))
    if field_size is not None:
        if field_size >= 16:
            score_value += 0.12
            reasons.append("多頭数")
        elif field_size >= 14:
            score_value += 0.07
        elif field_size <= 10:
            score_value -= 0.08
            reasons.append("少頭数")

    odds_values = sorted(v for v in (num(r.get("winOdds")) for r in runners) if v is not None and v > 0)
    if odds_values:
        favorite = odds_values[0]
        if favorite <= 2.0:
            score_value -= 0.18
            reasons.append(f"1人気{favorite:.1f}倍")
        elif favorite <= 3.0:
            score_value -= 0.10
            reasons.append(f"1人気{favorite:.1f}倍")
        elif favorite >= 5.0:
            score_value += 0.12
            reasons.append("人気割れ")
        implied = [1.0 / v for v in odds_values]
        total = sum(implied)
        if total > 0:
            top3_share = sum(implied[:3]) / total
            if top3_share >= 0.62:
                score_value -= 0.08
                reasons.append("人気集中")
            elif top3_share <= 0.44:
                score_value += 0.08
                reasons.append("人気分散")

    confidence = num(race.get("confidence"))
    if confidence is not None:
        if confidence >= 0.65:
            score_value -= 0.08
            reasons.append("AI信頼高め")
        elif confidence <= 0.25:
            score_value += 0.08
            reasons.append("AI信頼低め")

    gap = num(race.get("scoreGap"))
    if gap is not None:
        if gap >= 0.12:
            score_value -= 0.07
            reasons.append("AI差あり")
        elif gap <= 0.025:
            score_value += 0.07
            reasons.append("AI差薄い")

    pressure = num(race.get("pressureScore"))
    if pressure is not None:
        if pressure >= 0.50:
            score_value += 0.08
            reasons.append("先行負荷高め")
        elif pressure <= 0.15:
            score_value -= 0.03

    front_count = num(race.get("frontRunnerCount"))
    if front_count is not None and front_count >= 5:
        score_value += 0.05
        reasons.append("逃げ候補多め")

    score_value = clamp(score_value)
    if score_value >= 0.66:
        label = "高め"
    elif score_value >= 0.45:
        label = "中"
    else:
        label = "堅め"

    note = " / ".join(dict.fromkeys(reasons[:4])) if reasons else "標準的"
    return {
        "upsetScore": round(score_value, 3),
        "upsetLabel": label,
        "upsetNote": note,
    }


def format_date_ja(date_key: str) -> str:
    try:
        dt = datetime.strptime(date_key, "%Y%m%d")
        return f"{dt.year}年{dt.month}月{dt.day}日"
    except Exception:
        return date_key


def parse_date_key_any(value: object) -> str:
    raw = re.sub(r"\D", "", text(value))
    if len(raw) == 8 and raw.startswith("20"):
        return raw
    if len(raw) == 6:
        return "20" + raw
    return ""


def official_race_id_for_display(row: pd.Series) -> str:
    raw = text(row.get("レースID(新/馬番無)") if "レースID(新/馬番無)" in row.index else row.get("race_id"))
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 16 and digits.startswith("20"):
        return digits
    source_url = text(row.get("source_url"))
    match = re.search(r"race_id=(\d{12})", source_url)
    date_key = parse_date_key_any(row.get("日付") if "日付" in row.index else row.get("date"))
    if date_key and match:
        return date_key + match.group(1)[4:12]
    venue = text(row.get("場所"))
    race_no = int(num(row.get("Ｒ")) or 0)
    venue_code = {
        "札幌": "01",
        "函館": "02",
        "福島": "03",
        "新潟": "04",
        "東京": "05",
        "中山": "06",
        "中京": "07",
        "京都": "08",
        "阪神": "09",
        "小倉": "10",
    }.get(venue)
    if date_key and venue_code and race_no:
        return f"{date_key}{venue_code}0000{race_no:02d}"
    return ""


def build_entry_display_rows(
    entry_csv: Path | None,
    prediction_csv: Path | None,
    desired_dates: set[str],
    body_map: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    for source_path, source_name in ((entry_csv, "entry"), (prediction_csv, "prediction")):
        if source_path is None or not source_path.exists():
            continue
        frame = read_csv_safe(source_path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["_display_source"] = source_name
        frame["_race_id_display"] = frame.apply(official_race_id_for_display, axis=1)
        frame = frame[frame["_race_id_display"].astype(str).str.len().eq(16)].copy()
        if frame.empty:
            continue
        frame["_date_key_display"] = frame["_race_id_display"].astype(str).str[:8]
        frames.append(frame)

    if not frames:
        return {}, []

    combined = pd.concat(frames, ignore_index=True, sort=False)
    available_dates = sorted(d for d in combined["_date_key_display"].dropna().astype(str).unique() if d)
    if not desired_dates and available_dates:
        desired_dates = {available_dates[-1]}
    if desired_dates:
        combined = combined[combined["_date_key_display"].isin(desired_dates)].copy()

    races: dict[str, dict[str, Any]] = {}
    runners_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in combined.iterrows():
        race_id = text(row.get("_race_id_display"))
        horse_no = num(row.get("馬番") if "馬番" in row.index else row.get("horse_no"))
        if not race_id or horse_no is None or horse_no <= 0:
            continue
        horse_no_i = int(horse_no)
        date_key = race_id[:8]
        venue = text(row.get("場所") if "場所" in row.index else row.get("venue"))
        race_no = int(num(row.get("Ｒ") if "Ｒ" in row.index else row.get("race_no")) or 0)
        race_name = text(row.get("レース名") if "レース名" in row.index else row.get("race_name"))
        surface = text(row.get("芝・ダ") if "芝・ダ" in row.index else row.get("surface"))
        distance = int_text(row.get("距離") if "距離" in row.index else row.get("distance"))
        field_size = int_text(row.get("頭数") if "頭数" in row.index else row.get("出走頭数"))
        start_time = text(row.get("発走時刻") if "発走時刻" in row.index else row.get("start_time"))
        venue_code = race_id[8:10] if len(race_id) == 16 else ""
        meta = races.setdefault(
            race_id,
            {
                "raceId": race_id,
                "dateKey": date_key,
                "dateLabel": format_date_ja(date_key),
                "venueCode": venue_code,
                "venue": venue or VENUES.get(venue_code, venue_code),
                "kaiji": race_id[10:12] if len(race_id) == 16 else "",
                "nichiji": race_id[12:14] if len(race_id) == 16 else "",
                "raceNo": race_no,
                "raceLabel": f"{format_date_ja(date_key)} {venue or VENUES.get(venue_code, venue_code)}{race_no}R",
            },
        )
        for key, value in {
            "raceName": race_name,
            "startTime": start_time,
            "surface": surface,
            "distance": distance,
            "fieldSize": field_size,
        }.items():
            if value and not meta.get(key):
                meta[key] = value

        key = (race_id, horse_no_i)
        info = runners_by_key.setdefault(key, {"raceId": race_id, "horseNo": horse_no_i})
        for out_key, value in {
            "horseName": text(row.get("馬名") if "馬名" in row.index else row.get("horse_name")),
            "jockey": text(row.get("騎手") if "騎手" in row.index else row.get("jockey")),
            "carriedWeight": num(row.get("斤量") if "斤量" in row.index else row.get("weight_carried")),
            "frameNo": int_text(row.get("枠番") if "枠番" in row.index else row.get("frame_no")),
            "aiRank": int_text(row.get("ai_rank")),
            "aiScore": num(row.get("ai_score")),
            "expectedPace": text(row.get("expected_pace")),
            "frontRunning": num(row.get("front_running_tendency")),
            "closing": num(row.get("closing_tendency")),
            "confidence": num(row.get("ai_confidence_score")),
            "confidenceBucket": text(row.get("ai_confidence_bucket")),
            "predayWinOdds": odds_num(row.get("odds_latest_win") if "odds_latest_win" in row.index else row.get("単勝オッズ")),
            "popularity": int_text(row.get("人気") if "人気" in row.index else row.get("popularity")),
        }.items():
            if value not in ("", None) and not pd.isna(value):
                info[out_key] = value
        if num(row.get("ai_rank")) == 1 or "expectedPace" not in meta:
            for key_name, value in {
                "expectedPace": text(row.get("expected_pace")),
                "frontRunnerCount": num(row.get("race_front_runner_count")),
                "pressureScore": num(row.get("race_early_pressure_score")),
                "confidence": num(row.get("ai_confidence_score")),
                "scoreGap": num(row.get("ai_score_gap_to_second")),
            }.items():
                if value not in ("", None) and not pd.isna(value):
                    meta[key_name] = value

    runners: list[dict[str, Any]] = []
    for (race_id, horse_no), info in runners_by_key.items():
        body_info = body_map.get((race_id, horse_no), {})
        row_info = {
            "raceId": race_id,
            "horseNo": horse_no,
            "frameNo": info.get("frameNo", ""),
            "horseName": info.get("horseName", ""),
            "jockey": info.get("jockey", ""),
            "carriedWeight": info.get("carriedWeight"),
            "carriedWeightText": carried_weight_label(info.get("carriedWeight")),
            "aiRank": info.get("aiRank", ""),
            "aiScore": info.get("aiScore"),
            "expectedPace": info.get("expectedPace", ""),
            "frontRunning": info.get("frontRunning"),
            "closing": info.get("closing"),
            "confidence": info.get("confidence"),
            "confidenceBucket": info.get("confidenceBucket", ""),
            "points": _runner_points(
                {
                    "aiRank": info.get("aiRank"),
                    "aiScore": info.get("aiScore"),
                    "frontRunning": info.get("frontRunning"),
                    "closing": info.get("closing"),
                    "expectedPace": info.get("expectedPace"),
                    "firstConditionImpact": info.get("firstConditionImpact"),
                }
            ),
            "concerns": _runner_concerns(
                {
                    "aiRank": info.get("aiRank"),
                    "frontRunning": info.get("frontRunning"),
                    "confidenceBucket": info.get("confidenceBucket"),
                    "firstConditionUncertainty": info.get("firstConditionUncertainty"),
                }
            )
            or ["オッズ未取得のため暫定評価"],
            "winOdds": info.get("predayWinOdds"),
            "popularity": info.get("popularity", ""),
            "bodyWeight": num(body_info.get("bodyWeight")),
            "bodyWeightDiff": num(body_info.get("bodyWeightDiff")),
            "bodyWeightText": body_weight_label(body_info.get("bodyWeight"), body_info.get("bodyWeightDiff")),
            "bodyWeightSnapshot": body_info.get("bodyWeightSnapshot", ""),
            "placeMin": None,
            "placeMax": None,
            "predayWinOdds": info.get("predayWinOdds"),
            "snapshotAt": "",
            "snapshotLabel": "",
        }
        runners.append(row_info)
    return races, runners


def parse_ticket_numbers(row: pd.Series) -> tuple[int, int | None]:
    a = first_number(row, ["horse_a", "a_no", "anchor_no", "horse_no", "馬番"])
    b = first_number(row, ["horse_b", "b_no", "partner_no"])
    if a is None:
        raw = text(first_value(row, ["numbers", "pair_key", "runtime_pair_key", "ticket_key"]))
        match = re.search(r"(\d{1,2})\D+(\d{1,2})", raw)
        if match:
            a = float(match.group(1))
            b = float(match.group(2))
    return int(a or 0), int(b) if b is not None else None


def ticket_type_label(ticket_type: str) -> str:
    normalized = text(ticket_type).strip().lower()
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
    return labels.get(normalized, text(ticket_type))


def truthy(value: object, default: bool = False) -> bool:
    raw = text(value).strip().lower()
    if raw == "":
        return default
    if raw in {"true", "1", "1.0", "yes", "y"}:
        return True
    if raw in {"false", "0", "0.0", "no", "n"}:
        return False
    return default


def humanize_reason_summary(reason: str) -> str:
    raw = text(reason).strip()
    if not raw:
        return "\u6700\u5f37\u7248\u306e\u8cfc\u5165\u6761\u4ef6\u3092\u901a\u904e"

    lower = raw.lower()
    parts: list[str] = []
    if "strongest_current" in lower or "strongest" in lower:
        parts.append("\u6700\u5f37\u7248\u306e\u53b3\u9078\u6761\u4ef6\u3092\u901a\u904e")
    if "umaren_only" in lower:
        parts.append("\u99ac\u9023\u5411\u304d")
    if "wide_only" in lower:
        parts.append("\u30ef\u30a4\u30c9\u5411\u304d")
    margin = re.search(r"margin=([0-9.]+)", lower)
    if margin:
        value = float(margin.group(1))
        label = "\u30aa\u30c3\u30ba\u5999\u5473\u306f\u304b\u306a\u308a\u5927\u304d\u3044" if value >= 3.0 else "\u30aa\u30c3\u30ba\u5999\u5473\u306f\u8cb7\u3044\u6c34\u6e96" if value >= 2.5 else "\u30aa\u30c3\u30ba\u5999\u5473\u306f\u5c11\u3057\u3042\u308a" if value >= 1.5 else "\u30aa\u30c3\u30ba\u5999\u5473\u306f\u8584\u3044"
        parts.append(label)
    skip = re.search(r"skip=([0-9.]+)", lower)
    if skip:
        value = float(skip.group(1))
        label = "\u898b\u9001\u308a\u30ea\u30b9\u30af\u306f\u9ad8\u3081" if value >= 0.60 else "\u898b\u9001\u308a\u30ea\u30b9\u30af\u306f\u3084\u3084\u9ad8\u3081" if value >= 0.45 else "\u898b\u9001\u308a\u30ea\u30b9\u30af\u306f\u6a19\u6e96" if value >= 0.25 else "\u898b\u9001\u308a\u30ea\u30b9\u30af\u306f\u4f4e\u3081"
        parts.append(label)
    front5 = re.search(r"front5=([0-9.]+)", lower)
    if front5:
        value = float(front5.group(1))
        label = "\u524d\u306b\u884c\u3051\u308b\u898b\u8fbc\u307f\u306f\u9ad8\u3044" if value >= 0.70 else "\u524d\u306b\u884c\u3051\u308b\u898b\u8fbc\u307f\u306f\u8cb7\u3044\u6c34\u6e96" if value >= 0.60 else "\u524d\u306b\u884c\u3051\u308b\u898b\u8fbc\u307f\u306f\u6a19\u6e96" if value >= 0.45 else "\u524d\u306b\u884c\u3051\u308b\u898b\u8fbc\u307f\u306f\u4f4e\u3081"
        parts.append(label)
    first_condition = re.search(r"first_condition=([a-z0-9_]+)", lower)
    if first_condition:
        label = {
            "low_sample_edge_ok": "\u521d\u6761\u4ef6\u30fb\u4f4e\u30b5\u30f3\u30d7\u30eb\u306f\u5fc5\u8981\u5999\u5473\u3092\u542b\u3081\u3066\u78ba\u8a8d",
            "clear": "\u521d\u6761\u4ef6\u30ea\u30b9\u30af\u306f\u8584\u3044",
        }.get(first_condition.group(1), "")
        if label:
            parts.append(label)
    danger_popular = re.search(r"danger_popular=([a-z0-9_]+)", lower)
    if danger_popular:
        label = {
            "included_popular_edge_ok": "\u5371\u967a\u4eba\u6c17\u3092\u542b\u3080\u305f\u3081\u5999\u5473\u3092\u542b\u3081\u3066\u78ba\u8a8d",
            "clear": "\u5371\u967a\u4eba\u6c17\u306e\u5dfb\u304d\u8fbc\u307f\u306f\u8584\u3044",
        }.get(danger_popular.group(1), "")
        if label:
            parts.append(label)
    readability = re.search(r"readability=([a-z0-9_]+)", lower)
    if readability:
        label = {"hard": "\u8aad\u307f\u306b\u304f\u3044\u30ec\u30fc\u30b9", "easy": "\u8aad\u307f\u3084\u3059\u3044\u30ec\u30fc\u30b9"}.get(readability.group(1), "")
        if label:
            parts.append(label)
    parts = list(dict.fromkeys(parts))
    return " / ".join(parts) if parts else raw


def _max_first_number(row: pd.Series, names: list[str], default: float = 0.0) -> float:
    values = [first_number(row, [name]) for name in names if name in row.index]
    values = [float(value) for value in values if value is not None and not pd.isna(value)]
    return max(values) if values else default


def queue_lap_shadow_context(row: pd.Series) -> dict[str, Any]:
    """Translate queue/front-3F/lap features into dashboard labels.

    These fields are explanatory shadow diagnostics only. They do not change the
    formal strongest BUY gates.
    """

    anchor_front = _max_first_number(
        row,
        [
            "anchor_projected_front5_prob",
            "anchor_front5_model_prob",
            "a_pred_front5",
        ],
    )
    partner_front = _max_first_number(
        row,
        [
            "partner_projected_front5_prob",
            "partner_front5_model_prob",
            "b_pred_front5",
        ],
    )
    pair_front = _max_first_number(
        row,
        [
            "projected_front5_prob",
            "ticket_front_position_reliability_score",
            "position_front_value_score",
        ],
    )
    pair_front_any = max(anchor_front, partner_front, pair_front) >= 0.60
    pair_front_both = anchor_front >= 0.60 and partner_front >= 0.60

    queue_clarity = _max_first_number(
        row,
        [
            "race_queue_clarity_score",
            "anchor_runtime_queue_clarity_score",
            "partner_runtime_queue_clarity_score",
        ],
    )
    duel_risk = _max_first_number(
        row,
        [
            "race_front_duel_risk_score",
            "anchor_runtime_front_duel_risk_score",
            "partner_runtime_front_duel_risk_score",
        ],
    )
    front_load = _max_first_number(
        row,
        [
            "race_projected_front_load_score",
            "anchor_runtime_projected_front_load_score",
            "partner_runtime_projected_front_load_score",
        ],
    )
    pace_shape = text(first_value(row, ["race_pace_shape_label", "anchor_runtime_pace_shape_label", "partner_runtime_pace_shape_label"]))

    pace_pair_label = text(row.get("pace_pair_gate_label"))
    lap_promote_label = text(row.get("lap_positive_expansion_label"))
    lap_advanced_label = text(row.get("lap_advanced_shadow_label"))
    lap_advanced_note = text(row.get("lap_advanced_shadow_note"))
    lap_advanced_score = _max_first_number(row, ["lap_advanced_combo_score"])
    lap_track_label = text(row.get("lap_track_shadow_label"))
    lap_track_note = text(row.get("lap_track_shadow_note"))
    lap_track_score = _max_first_number(row, ["lap_track_shadow_score"])
    target_ra_lap_label = text(row.get("target_ra_lap_shadow_label"))
    target_ra_lap_note = text(row.get("target_ra_lap_shadow_note"))
    target_ra_lap_fit = _max_first_number(row, ["target_ra_lap_pair_fit_score"])
    target_ra_lap_ready = _max_first_number(row, ["target_ra_lap_pair_ready_score"])
    target_ra_lap_mismatch = _max_first_number(row, ["target_ra_lap_mismatch_risk_score"])
    goodrun_lap_score = _max_first_number(row, ["goodrun_lap_pair_avg_score"])
    role_proxy_score = _max_first_number(row, ["lap_role_pair_probability_proxy"])
    horse_lap_decomp_label = text(row.get("horse_lap_decomp_runtime_label"))
    horse_lap_decomp_note = text(row.get("horse_lap_decomp_runtime_note"))
    horse_lap_decomp_score = _max_first_number(row, ["horse_lap_decomp_runtime_score", "horse_lap_decomp_score"])
    horse_lap_decomp_race_mode = text(row.get("horse_lap_decomp_race_mode_label"))
    horse_lap_decomp_type_pair = text(row.get("horse_lap_decomp_type_pair_label"))
    horse_lap_decomp_match = text(row.get("horse_lap_decomp_match_bucket_label"))
    race_quality_tag = text(first_value(row, ["race_quality_v2_runtime_tag", "anchor_race_quality_v2_runtime_tag", "partner_race_quality_v2_runtime_tag"]))
    past3_lap_tag = text(row.get("past3_lap_runtime_tag"))
    readability = _max_first_number(row, ["continuous_pair_readability_score", "pace_regime_readability_score"])
    pace_fit = _max_first_number(row, ["continuous_pair_pace_fit_score", "pace_fit_pair_score"])
    lap_advanced_good = lap_advanced_label in {
        "lap_role_goodrun_strong",
        "goodrun_lap_strong",
        "lap_advanced_combo_strong",
        "lap_advanced_combo_watch",
    }
    lap_good = (
        lap_promote_label in {"lap_promote_strong", "lap_promote_watch", "lap_role_watch", "lap_1win_fast_same_distance_shadow"}
        or lap_advanced_good
        or pace_pair_label in {"pace_pair_strong", "pace_pair_watch"}
        or target_ra_lap_label in {"target_ra_lap_fit_strong_both", "target_ra_lap_fit_strong", "target_ra_lap_fit_watch"}
        or (readability >= 0.58 and pace_fit >= 0.42)
        or "good" in race_quality_tag.lower()
        or "ok" in race_quality_tag.lower()
    )
    lap_weak = (
        pace_pair_label == "pace_pair_caution"
        or text(row.get("past3_lap_bad_band_flag")) in {"1", "1.0", "True", "true"}
        or target_ra_lap_label == "target_ra_lap_caution"
        or "bad" in past3_lap_tag.lower()
        or "caution" in race_quality_tag.lower()
    )

    notes: list[str] = []
    priority = 0.0
    if pair_front_both:
        notes.append("\u30da\u30a22\u982d\u3068\u3082\u524d\u76ee\u5019\u88dc")
        priority += 0.35
    elif pair_front_any:
        notes.append("\u30da\u30a2\u306e\u7247\u65b9\u306f\u524d\u76ee\u5019\u88dc")
        priority += 0.18

    if queue_clarity >= 0.62 and duel_risk < 0.55:
        notes.append("\u968a\u5217\u306f\u6bd4\u8f03\u7684\u8aad\u307f\u3084\u3059\u3044")
        priority += 0.22
        queue_label = "queue_read_good"
    elif duel_risk >= 0.62 or front_load >= 0.60:
        notes.append("\u524d\u306f\u6df7\u307f\u3084\u3059\u304f\u3001\u30ef\u30a4\u30c9\u306f\u6ce8\u8996")
        priority += 0.12
        queue_label = "front_duel_dense"
    else:
        queue_label = "queue_neutral"

    if lap_good:
        notes.append("\u30e9\u30c3\u30d7\u8aad\u307f\u306f\u826f\u597d")
        priority += 0.25
    advanced_title_map = {
        "lap_role_goodrun_strong": "\u30e9\u30c3\u30d7\u30ed\u30fc\u30eb\u5f37",
        "goodrun_lap_strong": "\u597d\u8d70\u6642\u30e9\u30c3\u30d7\u9069\u6027\u5f37",
        "lap_advanced_combo_strong": "\u30e9\u30c3\u30d7\u7dcf\u5408\u5f37",
        "lap_advanced_combo_watch": "\u30e9\u30c3\u30d7\u7dcf\u5408\u6ce8\u8996",
    }
    if lap_advanced_label in advanced_title_map:
        advanced_note = lap_advanced_note or advanced_title_map[lap_advanced_label]
        if advanced_note:
            notes.append(advanced_note)
        priority += 0.26 if lap_advanced_label != "lap_advanced_combo_watch" else 0.14
    if lap_track_label in {"positive", "positive_soft"}:
        notes.append(lap_track_note or "馬場状態とラップ適性の組み合わせは注目")
        priority += 0.22 if lap_track_label == "positive" else 0.14
    elif lap_track_label == "caution":
        notes.append(lap_track_note or "馬場×ラップは注視止まり")
        priority += 0.06
    if target_ra_lap_label in {"target_ra_lap_fit_strong_both", "target_ra_lap_fit_strong"}:
        notes.append(target_ra_lap_note or "公式RAラップ履歴の裏付けあり")
        priority += 0.18
    elif target_ra_lap_label == "target_ra_lap_fit_watch":
        notes.append(target_ra_lap_note or "公式RAラップ履歴は参考プラス")
        priority += 0.10
    elif target_ra_lap_label == "target_ra_lap_caution":
        notes.append(target_ra_lap_note or "公式RAラップ履歴の裏付け薄め")
        priority -= 0.05
    if horse_lap_decomp_label in {"horse_lap_pair_fit_good", "horse_lap_high_pressure_instant_pair", "horse_lap_both_match"}:
        notes.append(horse_lap_decomp_note or "\u30e9\u30c3\u30d7\u5206\u89e3\u3067\u30da\u30a2\u76f8\u6027\u304c\u826f\u3044")
        priority += 0.24 if horse_lap_decomp_label == "horse_lap_high_pressure_instant_pair" else 0.18
    elif horse_lap_decomp_label == "horse_lap_pair_caution":
        notes.append(horse_lap_decomp_note or "\u30e9\u30c3\u30d7\u5206\u89e3\u3067\u5668\u7528\u3057\u306b\u304f\u3044\u30da\u30a2")
        priority -= 0.10
    if lap_weak:
        notes.append("\u30e9\u30c3\u30d7\u8aad\u307f\u306f\u5f31\u3081")
        priority -= 0.18

    if target_ra_lap_label == "target_ra_lap_fit_strong_both":
        label = "target_ra_lap_fit_strong_both"
        title = "公式RAラップ強"
    elif target_ra_lap_label == "target_ra_lap_fit_strong":
        label = "target_ra_lap_fit_strong"
        title = "公式RAラップ裏付け"
    elif lap_track_label == "positive" and lap_good:
        label = "lap_track_positive"
        title = "馬場×ラップ強"
    elif pair_front_both and lap_good:
        label = "front_pair_lap_good"
        title = "\u524d\u76ee\u30da\u30a2\u5f37"
    elif pair_front_both:
        label = "front_pair_strong"
        title = "\u524d\u76ee\u30da\u30a2\u5f37"
    elif lap_good and pair_front_any:
        label = "lap_good_front_any"
        title = "\u968a\u5217\u8aad\u307f\u826f\u597d"
    elif lap_advanced_label in advanced_title_map:
        label = lap_advanced_label
        title = advanced_title_map[lap_advanced_label]
    elif lap_track_label in {"positive", "positive_soft"}:
        label = "lap_track_watch"
        title = "馬場×ラップ注視"
    elif lap_track_label == "caution":
        label = "lap_track_caution"
        title = "馬場×ラップ参考"
    elif queue_label == "front_duel_dense":
        label = "mixed_queue_watch"
        title = "\u968a\u5217\u6df7\u6226\u6ce8\u8996"
    elif lap_weak:
        label = "lap_read_weak"
        title = "\u30e9\u30c3\u30d7\u8aad\u307f\u5f31"
    else:
        label = "queue_lap_neutral"
        title = ""

    if horse_lap_decomp_label == "horse_lap_high_pressure_instant_pair":
        label = "horse_lap_high_pressure_instant_pair"
        title = "\u524d\u534a\u8ca0\u8377\u00d7\u77ac\u767a\u30da\u30a2"
    elif horse_lap_decomp_label == "horse_lap_pair_fit_good":
        label = "horse_lap_pair_fit_good"
        title = "\u30e9\u30c3\u30d7\u76f8\u6027\u826f"
    elif horse_lap_decomp_label == "horse_lap_both_match":
        label = "horse_lap_both_match"
        title = "\u4e21\u99ac\u30e9\u30c3\u30d7\u4e00\u81f4"
    elif horse_lap_decomp_label == "horse_lap_pair_caution":
        label = "horse_lap_pair_caution"
        title = "\u30e9\u30c3\u30d7\u76f8\u6027\u6ce8\u610f"

    return {
        "queueLapLabel": label,
        "queueLapTitle": title,
        "queueLapNote": "\u3001".join(dict.fromkeys(notes[:4])),
        "queueLapPriority": round(clamp(priority), 3),
        "pairPredFront5Any": pair_front_any,
        "pairPredFront5Both": pair_front_both,
        "raceLapReadGood": bool(lap_good),
        "raceLapReadWeak": bool(lap_weak),
        "queueTypeLabel": queue_label,
        "queueClarityForLabel": round(queue_clarity, 3),
        "frontDuelRiskForLabel": round(duel_risk, 3),
        "frontLoadForLabel": round(front_load, 3),
        "paceShapeForLabel": pace_shape,
        "lapAdvancedScore": round(lap_advanced_score, 3),
        "lapTrackShadowLabel": lap_track_label,
        "lapTrackShadowScore": round(lap_track_score, 3),
        "lapTrackShadowNote": lap_track_note,
        "targetRaLapLabel": target_ra_lap_label,
        "targetRaLapFitScore": round(target_ra_lap_fit, 3),
        "targetRaLapReadyScore": round(target_ra_lap_ready, 3),
        "targetRaLapMismatchScore": round(target_ra_lap_mismatch, 3),
        "targetRaLapNote": target_ra_lap_note,
        "goodrunLapScore": round(goodrun_lap_score, 3),
        "lapRoleProxyScore": round(role_proxy_score, 3),
        "horseLapDecompLabel": horse_lap_decomp_label,
        "horseLapDecompScore": round(horse_lap_decomp_score, 3),
        "horseLapDecompNote": horse_lap_decomp_note,
        "horseLapDecompRaceMode": horse_lap_decomp_race_mode,
        "horseLapDecompTypePair": horse_lap_decomp_type_pair,
        "horseLapDecompMatch": horse_lap_decomp_match,
    }


def _normalize_ticket_type_for_lap(value: object) -> str:
    raw = text(value).strip().lower()
    if raw in {"quinella", "umaren"}:
        return "umaren"
    if raw == "wide":
        return "wide"
    return raw


def _lap_decomp_runtime_fields(row: pd.Series) -> dict[str, Any]:
    mode = text(row.get("race_lap_mode_label"))
    type_pair = text(row.get("lap_type_pair_label"))
    match_bucket = text(row.get("lap_match_bucket_label"))
    fit = num(row.get("pair_lap_fit_min_eval")) or 0.0
    cosine = num(row.get("pair_lap_cosine_min_eval")) or 0.0
    score = num(row.get("horse_lap_decomp_score")) or 0.0
    mismatch = num(row.get("pair_lap_mismatch_popular_max_eval")) or 0.0
    label = ""
    note = ""
    if mode == "\u524d\u534a\u8ca0\u8377\u578b" and type_pair in {
        "\u524d\u534a\u8ca0\u8377\u578b + \u77ac\u767a\u578b",
        "\u77ac\u767a\u578b + \u524d\u534a\u8ca0\u8377\u578b",
    }:
        label = "horse_lap_high_pressure_instant_pair"
        note = "\u524d\u534a\u8ca0\u8377\u60f3\u5b9a\u3067\u3001\u8ca0\u8377\u578b\u3068\u77ac\u767a\u578b\u306e\u7d44\u307f\u5408\u308f\u305b"
    elif match_bucket == "2\u982d\u3068\u3082\u60f3\u5b9a\u30e9\u30c3\u30d7\u4e00\u81f4" and fit >= 0.70:
        label = "horse_lap_both_match"
        note = "\u4e21\u99ac\u3068\u3082\u60f3\u5b9a\u30e9\u30c3\u30d7\u306b\u5408\u3044\u3001\u30da\u30a2\u76f8\u6027\u306f\u826f\u3044"
    elif fit >= 0.75 and cosine >= 0.70:
        label = "horse_lap_pair_fit_good"
        note = "\u99ac\u3054\u3068\u306e\u30e9\u30c3\u30d7\u5206\u89e3\u3067\u30da\u30a2\u76f8\u6027\u304c\u826f\u3044"
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
        "horse_lap_decomp_race_mode_label": mode,
        "horse_lap_decomp_anchor_type_label": text(row.get("anchor_lap_type_label")),
        "horse_lap_decomp_partner_type_label": text(row.get("partner_lap_type_label")),
        "horse_lap_decomp_match_bucket_label": match_bucket,
        "horse_lap_decomp_type_pair_label": type_pair,
        "horse_lap_decomp_pair_fit_min": fit,
        "horse_lap_decomp_pair_cosine_min": cosine,
        "horse_lap_decomp_mismatch_popular": mismatch,
    }


def load_horse_lap_decomp_lookup(path: Path | None) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    frame = read_csv_safe(path, dtype={"race_id": str})
    required = {"race_id", "ticket_type", "anchor_no", "partner_no"}
    if frame.empty or not required.issubset(frame.columns):
        return {}
    dataset_priority = {"mcs_runtime_recommended": 3, "lap_positive_selected": 2}
    lookup: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    scores: dict[tuple[str, str, int, int], tuple[int, float]] = {}
    for _, row in frame.iterrows():
        race_id = text(row.get("race_id"))
        ticket_type = _normalize_ticket_type_for_lap(row.get("ticket_type"))
        a_no = num(row.get("anchor_no"))
        b_no = num(row.get("partner_no"))
        if not race_id or not ticket_type or a_no is None or b_no is None:
            continue
        lo, hi = sorted([int(a_no), int(b_no)])
        key = (race_id, ticket_type, lo, hi)
        fields = _lap_decomp_runtime_fields(row)
        dataset = text(row.get("dataset"))
        fields["horse_lap_decomp_dataset"] = dataset
        row_score = float(fields.get("horse_lap_decomp_runtime_score") or 0.0)
        choice = (dataset_priority.get(dataset, 1), row_score)
        if key not in lookup or choice > scores[key]:
            lookup[key] = fields
            scores[key] = choice
    return lookup


def row_with_horse_lap_decomp(
    row: pd.Series,
    lookup: dict[tuple[str, str, int, int], dict[str, Any]],
    race_id: str,
    ticket_type: str,
    a_no: int,
    b_no: int | None,
) -> pd.Series:
    if not lookup or not b_no:
        return row
    lo, hi = sorted([a_no, b_no])
    info = lookup.get((race_id, _normalize_ticket_type_for_lap(ticket_type), lo, hi))
    if not info:
        return row
    out = row.copy()
    for key, value in info.items():
        out[key] = value
    return out


def resolve_default_tickets_csv() -> Path | None:
    summary = project_path("outputs/analysis/race_day_runtime_operation_latest/summary.json")
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            for key in ("selected_csv", "final_tickets_csv"):
                value = text(data.get(key))
                if value:
                    path = project_path(value)
                    if path.exists():
                        return path
        except Exception:
            pass
    return latest_file("outputs/analysis/race_day_runtime_operation*/mcs_pbo_overlay/selected_after_live_safety.csv")


def resolve_default_candidates_csv() -> Path | None:
    path = project_path("outputs/analysis/current_strongest_runtime_v1/current_strongest_all_candidates.csv")
    return path if path.exists() else None


def load_candidate_runner_context(candidates_csv: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if candidates_csv is None or not candidates_csv.exists():
        return {}
    candidates = read_csv_safe(candidates_csv, dtype={"race_id": str})
    if candidates.empty or "race_id" not in candidates.columns:
        return {}
    context: dict[tuple[str, int], dict[str, Any]] = {}

    def update_side(row: pd.Series, side: str) -> None:
        race_id = text(row.get("race_id"))
        horse_no = num(row.get(f"{side}_horse_no"))
        if not race_id or horse_no is None or not np.isfinite(horse_no) or horse_no <= 0:
            return
        key = (race_id, int(horse_no))
        info = context.setdefault(key, {})
        for src, dst in {
            f"{side}_first_condition_prev_impressive_score": "firstConditionImpact",
            f"{side}_first_condition_uncertainty_score": "firstConditionRawUncertainty",
            f"{side}_first_condition_net_uncertainty_score": "firstConditionUncertainty",
        }.items():
            value = num(row.get(src))
            if value is None or not np.isfinite(value):
                continue
            info[dst] = max(float(info.get(dst) or 0.0), float(value))

    for _, row in candidates.iterrows():
        update_side(row, "anchor")
        update_side(row, "partner")
    return context


def _first_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def dashboard_class_group(value: object) -> str:
    raw = text(value).strip()
    if "新馬" in raw:
        return "新馬"
    if "未勝利" in raw:
        return "未勝利"
    if "1勝" in raw or "500万" in raw:
        return "1勝"
    if "2勝" in raw or "1000万" in raw:
        return "2勝"
    if "3勝" in raw or "1600万" in raw:
        return "3勝"
    if any(token in raw for token in ["Ｇ１", "G1", "GⅠ", "ＧⅠ"]):
        return "G1"
    if any(token in raw for token in ["Ｇ２", "G2", "GⅡ", "ＧⅡ"]):
        return "G2"
    if any(token in raw for token in ["Ｇ３", "G3", "GⅢ", "ＧⅢ"]):
        return "G3"
    if "ｵｰﾌﾟﾝ" in raw or "オープン" in raw or raw.upper() == "OP":
        return "OP"
    if "L" in raw or "リステッド" in raw:
        return "L"
    return raw or "不明"


def dashboard_surface_key(value: object) -> str:
    raw = text(value).strip()
    if raw.startswith("芝") or raw.lower() in {"turf", "grass"}:
        return "芝"
    if raw.startswith("ダ") or raw.lower() in {"dirt", "sand"}:
        return "ダ"
    if raw.startswith("障"):
        return "障"
    return raw


def load_historical_condition_context(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = read_csv_safe(path)
    required = {"years", "scope", "surface", "distance", "sample_count"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.copy()
    for col in [
        "years",
        "distance",
        "sample_count",
        "avg_winning_time_sec",
        "avg_front3f_sec",
        "avg_last3f_sec",
        "avg_1000m_sec",
        "avg_rpci",
        "avg_pci3",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["venue", "surface", "class_group", "going", "scope"]:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__ALL__").astype(str)
    out["surface"] = out["surface"].map(dashboard_surface_key)
    return out


def _best_historical_condition_row(context: pd.DataFrame, race: dict[str, Any], years: int) -> pd.Series | None:
    if context.empty:
        return None
    venue = text(race.get("venue")).strip()
    surface = dashboard_surface_key(race.get("surface"))
    distance = num(race.get("distance"))
    class_name = text(race.get("className") or race.get("raceName")).strip()
    class_key = dashboard_class_group(class_name)
    going = text(race.get("runtimeGoing")).strip()
    if not surface or distance is None:
        return None
    base = context[
        context["years"].eq(years)
        & context["surface"].eq(surface)
        & context["distance"].eq(int(distance))
    ].copy()
    if base.empty:
        return None
    candidate_defs = [
        ("同場同距離×クラス×馬場", venue, class_key, going, 1),
        ("同場同距離×クラス", venue, class_key, "__ALL__", 2),
        ("同場同距離×馬場", venue, "__ALL__", going, 3),
        ("同場同距離", venue, "__ALL__", "__ALL__", 4),
        ("同芝ダ同距離×クラス", "__ALL__", class_key, "__ALL__", 5),
        ("同芝ダ同距離", "__ALL__", "__ALL__", "__ALL__", 6),
    ]
    for scope, venue_key, class_group, going_key, priority in candidate_defs:
        part = base[base["scope"].eq(scope)].copy()
        if venue_key != "__ALL__":
            part = part[part["venue"].eq(venue_key)]
        if class_group != "__ALL__":
            part = part[part["class_group"].eq(class_group)]
        if going_key != "__ALL__":
            part = part[part["going"].eq(going_key)]
        if part.empty:
            continue
        part["_priority"] = priority
        part = part.sort_values(["_priority", "sample_count"], ascending=[True, False])
        return part.iloc[0]
    return None


def apply_historical_condition_context(races: dict[str, dict[str, Any]], context: pd.DataFrame) -> int:
    if context.empty:
        return 0
    attached = 0
    metric_map = {
        "avg_winning_time_sec": "AvgTimeSec",
        "avg_front3f_sec": "Front3fSec",
        "avg_last3f_sec": "Last3fSec",
        "avg_1000m_sec": "Pass1000mSec",
        "avg_rpci": "RPCI",
        "avg_pci3": "PCI3",
    }
    for race in races.values():
        has_any = False
        for years, prefix in [(5, "hist5"), (10, "hist10")]:
            row = _best_historical_condition_row(context, race, years)
            if row is None:
                continue
            race[f"{prefix}Scope"] = text(row.get("scope"))
            race[f"{prefix}SampleCount"] = num(row.get("sample_count"))
            race[f"{prefix}ClassGroup"] = text(row.get("class_group"))
            race[f"{prefix}Going"] = text(row.get("going"))
            for src, dst in metric_map.items():
                value = num(row.get(src))
                if value is not None:
                    race[f"{prefix}{dst}"] = value
            has_any = True
        if has_any:
            attached += 1
    return attached


def load_course_ten_context(
    course_ten_context_csv: Path | None,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    if course_ten_context_csv is None or not course_ten_context_csv.exists():
        return {}, {}
    frame = read_csv_safe(course_ten_context_csv)
    if frame.empty:
        return {}, {}
    race_col = _first_column(frame, ["レースID(新/馬番無)", "race_id"])
    horse_col = _first_column(frame, ["馬番", "horse_no"])
    if not race_col:
        return {}, {}
    runner_context: dict[tuple[str, int], dict[str, Any]] = {}
    race_context: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        race_id = text(row.get(race_col)).strip()
        if race_id.endswith(".0"):
            race_id = race_id[:-2]
        if not race_id:
            continue
        race_context.setdefault(
            race_id,
            {
                "courseTenRaceLabel": text(row.get("course_ten_race_label")),
                "courseTenPressureScore": num(row.get("race_course_adj_ten_pressure_score")),
                "courseTenFastStartCount": num(row.get("race_course_adj_fast_start_count")),
                "courseTenSpeedGapTop2": num(row.get("race_course_adj_ten_speed_gap_top2")),
                "courseTenQueueClarityScore": num(row.get("race_course_adj_queue_clarity_score")),
                "courseTenPriorCount": num(row.get("course_front3f_prior_count")),
                "courseTenHistoryCount": num(row.get("course_ten_history_count")),
                "expectedFront3fSec": num(row.get("race_expected_front3f_sec")),
                "expectedFront3fPriorSec": num(row.get("race_front3f_prior_sec_for_display")),
                "expectedFront3fAdjustmentSec": num(row.get("race_expected_front3f_adjustment_sec")),
                "expectedFront3fPriorCount": num(row.get("race_expected_front3f_prior_count")),
                "expectedFront3fNote": text(row.get("race_expected_front3f_note")),
            },
        )
        horse_no = num(row.get(horse_col)) if horse_col else None
        if horse_no is None or not np.isfinite(horse_no) or horse_no <= 0:
            continue
        runner_context[(race_id, int(horse_no))] = {
            "courseTenRunnerNote": text(row.get("course_ten_runner_note")),
            "courseTenSpeed": num(row.get("horse_course_adj_ten_speed_mean_past5")),
            "courseTenBest": num(row.get("horse_course_adj_ten_speed_best_past5")),
            "courseTenFastStartRate": num(row.get("horse_course_adj_fast_start_rate_past5")),
            "courseTenRaceZ": num(row.get("course_adj_ten_race_z")),
            "courseTenHistoryAvailable": num(row.get("course_ten_history_available")),
        }
    return runner_context, race_context


def build_race_decision_rows(candidates_csv: Path | None, current_race_ids: set[str]) -> list[dict[str, Any]]:
    if candidates_csv is None or not candidates_csv.exists():
        return []
    candidates = read_csv_safe(candidates_csv, dtype={"race_id": str})
    if candidates.empty or "race_id" not in candidates.columns:
        return []

    def f(row: pd.Series, col: str, default: float = 0.0) -> float:
        value = row.get(col)
        try:
            if value is None or pd.isna(value) or text(value) == "":
                return default
            return float(str(value).replace(",", ""))
        except Exception:
            return default

    def text_col(frame: pd.DataFrame, col: str) -> pd.Series:
        if col in frame.columns:
            return frame[col].astype("string").fillna("").astype(str)
        return pd.Series("", index=frame.index, dtype=str)

    rows: list[dict[str, Any]] = []
    for race_id, race_df in candidates.groupby(candidates["race_id"].astype(str)):
        if race_id not in current_race_ids:
            continue
        race_df = race_df.copy()
        venue_code = race_id[8:10] if len(race_id) >= 10 else ""
        score = pd.to_numeric(race_df.get("strongest_current_score"), errors="coerce").fillna(0.0)
        race_df["_decision_score"] = score
        race_df = race_df.sort_values(
            ["_decision_score", "min_odds_margin_ratio", "runtime_expected_roi"],
            ascending=[False, False, False],
        )
        top = race_df.iloc[0]

        anchor_rank = pd.to_numeric(race_df.get("anchor_ai_rank_num"), errors="coerce").fillna(99)
        partner_rank = pd.to_numeric(race_df.get("partner_ai_rank_num"), errors="coerce").fillna(99)
        margin = pd.to_numeric(race_df.get("min_odds_margin_ratio"), errors="coerce").fillna(0.0)
        expected_roi = pd.to_numeric(race_df.get("runtime_expected_roi"), errors="coerce").fillna(0.0)
        skip_risk = pd.to_numeric(race_df.get("skip_risk_score"), errors="coerce").fillna(1.0)
        danger = pd.to_numeric(race_df.get("ticket_danger_popular_score"), errors="coerce").fillna(1.0)
        gelding = pd.to_numeric(race_df.get("gelding_pair_risk_score"), errors="coerce").fillna(0.0)
        quinella = pd.to_numeric(race_df.get("pair_quinella_score"), errors="coerce").fillna(0.0)
        parity = (
            pd.to_numeric(race_df.get("anchor_strongest_feature_parity_ready"), errors="coerce").fillna(0.0).ge(1.0)
            & pd.to_numeric(race_df.get("partner_strongest_feature_parity_ready"), errors="coerce").fillna(0.0).ge(1.0)
        )
        base_mask = (
            anchor_rank.le(3)
            & partner_rank.le(8)
            & parity
            & margin.ge(0.95)
            & expected_roi.ge(1.35)
            & skip_risk.le(0.52)
            & danger.le(0.70)
            & gelding.le(0.35)
            & quinella.ge(0.52)
        )
        base_rows = race_df[base_mask].copy()
        closer_flag = pd.to_numeric(race_df.get("closer_logic_watch_flag"), errors="coerce").fillna(0.0).ge(1.0)
        closer_rows = race_df[closer_flag].copy()
        if not closer_rows.empty:
            closer_rows["_closer_watch_score"] = pd.to_numeric(
                closer_rows.get("closer_logic_watch_score"), errors="coerce"
            ).fillna(0.0)
            closer_rows = closer_rows.sort_values(
                ["_closer_watch_score", "min_odds_margin_ratio", "runtime_expected_roi"],
                ascending=[False, False, False],
            )
            top_closer = closer_rows.iloc[0]
        else:
            top_closer = None

        soft_heavy = (
            pd.to_numeric(race_df.get("anchor_runtime_soft_heavy_flag"), errors="coerce").fillna(0.0).gt(0.0)
            | pd.to_numeric(race_df.get("partner_runtime_soft_heavy_flag"), errors="coerce").fillna(0.0).gt(0.0)
        )
        anchor_going = text_col(race_df, "anchor_runtime_going_class")
        partner_going = text_col(race_df, "partner_runtime_going_class")
        tokyo_wet = venue_code == "05" and bool(
            anchor_going.isin(["Yielding", "Soft", "Heavy"]).any()
            or partner_going.isin(["Yielding", "Soft", "Heavy"]).any()
        )
        hakodate = venue_code == "02"
        op_allowed = (not hakodate) and (not bool(soft_heavy.any())) and (not tokyo_wet)

        if not base_rows.empty:
            b = base_rows.sort_values(
                ["_decision_score", "min_odds_margin_ratio", "runtime_expected_roi"],
                ascending=[False, False, False],
            )
            top_base = b.iloc[0]
        else:
            top_base = top

        b_margin = f(top_base, "min_odds_margin_ratio")
        b_expected = f(top_base, "runtime_expected_roi")
        b_score = f(top_base, "strongest_current_score")
        b_skip = f(top_base, "skip_risk_score", 1.0)
        b_front5 = f(top_base, "projected_front5_prob")
        b_pace = f(top_base, "pace_fit_pair_score")
        b_workout = f(top_base, "workout_pair_score")
        b_live = f(top_base, "live_odds", 9999.0)
        b_queue_clarity = f(top_base, "race_queue_clarity_score")
        b_duel_risk = f(top_base, "race_front_duel_risk_score")
        b_front_load = f(top_base, "race_projected_front_load_score")
        b_front_gap = num(first_value(top_base, ["race_lead_top_gap", "race_front5_top_gap"])) or 0.0
        b_front_candidates = num(first_value(top_base, ["race_lead_candidate_count", "race_front5_candidate_count"])) or 0.0
        b_pace_shape = text(first_value(top_base, ["race_pace_shape_label", "anchor_runtime_pace_shape_label"]))
        first_unc = f(top_base, "first_condition_pair_uncertainty_score")
        danger_in_pair = f(top_base, "ticket_danger_popular_in_pair_score")
        difficulty = f(top_base, "race_difficulty_score")
        first_ok = first_unc < 0.45 or (b_margin >= 3.0 and b_expected >= 1.6)
        danger_ok = danger_in_pair < 0.42 or (b_margin >= 3.0 and b_expected >= 1.6)
        difficulty_ok = difficulty <= 0.58 or (b_margin >= 3.2 and b_expected >= 1.75)
        top_output = top_base
        strict_no_guard = (
            not base_rows.empty
            and b_score >= 0.86
            and b_margin >= 2.50
            and b_skip <= 0.45
            and b_front5 >= 0.60
            and b_pace >= 0.35
            and b_workout >= 0.20
            and b_live <= 120.0
            and first_ok
            and danger_ok
            and difficulty_ok
        )

        guard_reasons = []
        if hakodate:
            guard_reasons.append("函館ガード")
        if bool(soft_heavy.any()):
            guard_reasons.append("道悪ガード")
        if tokyo_wet:
            guard_reasons.append("東京道悪ガード")

        if strict_no_guard and not op_allowed:
            key, label, class_name = "candidate", "強め見送り", "candidate"
            reason = "買い水準に近いが、" + "・".join(guard_reasons) + "で購入対象外"
            priority = 0.90
        elif strict_no_guard and op_allowed:
            key, label, class_name = "candidate", "準買い候補", "candidate"
            reason = "最終BUY候補に近いが、当日上限や最終選定で未採用"
            priority = 0.86
        elif top_closer is not None and f(top_closer, "closer_logic_watch_score") >= 0.58:
            top_output = top_closer
            b_margin = f(top_output, "min_odds_margin_ratio")
            b_expected = f(top_output, "runtime_expected_roi")
            b_score = f(top_output, "strongest_current_score")
            b_skip = f(top_output, "skip_risk_score", 1.0)
            b_front5 = f(top_output, "projected_front5_prob")
            b_live = f(top_output, "live_odds", 9999.0)
            b_queue_clarity = f(top_output, "race_queue_clarity_score")
            b_duel_risk = f(top_output, "race_front_duel_risk_score")
            b_front_load = f(top_output, "race_projected_front_load_score")
            b_front_gap = num(first_value(top_output, ["race_lead_top_gap", "race_front5_top_gap"])) or 0.0
            b_front_candidates = num(first_value(top_output, ["race_lead_candidate_count", "race_front5_candidate_count"])) or 0.0
            b_pace_shape = text(first_value(top_output, ["race_pace_shape_label", "anchor_runtime_pace_shape_label"]))
            key, label, class_name = "closer_watch", "差しハマり注視", "closer-watch"
            reason = "先行負荷と崩れ余地があり、差し・好位差しの相手妙味をシャドー検証"
            priority = 0.74
        elif not base_rows.empty and (b_score >= 0.78 or b_margin >= 2.0 or b_expected >= 2.0):
            key, label, class_name = "watch", "注視", "watch"
            reason = "AI値とオッズ妙味はあるが、最終BUY条件までは未達"
            priority = 0.70
        elif not base_rows.empty:
            key, label, class_name = "weak", "参考弱", "weak"
            reason = "候補はあるが、強度・安全条件が不足"
            priority = 0.52
        elif b_score >= 0.72:
            key, label, class_name = "weak", "参考弱", "weak"
            reason = "上位候補はいるが、基礎フィルター未達"
            priority = 0.42
        else:
            key, label, class_name = "skip", "見送り", "skip"
            reason = "買い水準の候補なし"
            priority = 0.20

        rows.append(
            {
                "raceId": race_id,
                "key": key,
                "label": label,
                "className": class_name,
                "priorityScore": round(priority, 3),
                "topScore": round(b_score, 3),
                "margin": round(b_margin, 2),
                "expectedRoi": round(b_expected, 2),
                "skipRisk": round(b_skip, 2),
                "front5": round(b_front5, 2),
                "queueClarityScore": round(b_queue_clarity, 3),
                "frontDuelRiskScore": round(b_duel_risk, 3),
                "frontLoadScore": round(b_front_load, 3),
                "front5TopGap": round(b_front_gap, 3),
                "front5CandidateCount": round(b_front_candidates, 1),
                "paceShapeLabel": b_pace_shape,
                "guardReasons": guard_reasons,
                "reason": reason,
                "topPair": {
                    "aNo": int(f(top_output, "anchor_horse_no") or f(top_output, "horse_a") or 0),
                    "bNo": int(f(top_output, "partner_horse_no") or f(top_output, "horse_b") or 0),
                    "odds": round(b_live, 1) if b_live < 9999 else None,
                },
            }
        )
    return rows


def build_ticket_rows(
    tickets_csv: Path | None,
    current_race_ids: set[str],
    horse_map: dict[tuple[str, int], dict[str, Any]],
    pair_lookup: dict[tuple[str, str, int, int], dict[str, Any]],
    horse_lap_decomp_lookup: dict[tuple[str, str, int, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if tickets_csv is None or not tickets_csv.exists():
        return []
    tickets = read_csv_safe(tickets_csv, dtype={"race_id": str})
    if tickets.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in tickets.iterrows():
        race_id = text(row.get("race_id"))
        if not (race_id.isdigit() and len(race_id) == 16):
            race_id = row_to_official_race_id(row)
        if race_id not in current_race_ids:
            continue
        ticket_type = text(first_value(row, ["ticket_type", "ticketType"]), "unknown")
        a_no, b_no = parse_ticket_numbers(row)
        if not a_no:
            continue
        row = row_with_horse_lap_decomp(row, horse_lap_decomp_lookup or {}, race_id, ticket_type, a_no, b_no)
        stake = first_number(row, ["runtime_stake_yen", "scaled_stake_yen", "eval_stake_yen", "stake_yen", "amount_yen"])
        if stake is None or stake <= 0:
            continue
        a_info = row_horse_info(horse_map, race_id, a_no)
        b_info = row_horse_info(horse_map, race_id, b_no or 0)
        pair_key = None
        if b_no is not None:
            lo, hi = sorted([a_no, b_no])
            pair_key = (race_id, ticket_type, lo, hi)
        live = pair_lookup.get(pair_key, {}) if pair_key else {}
        action = text(
            first_value(
                row,
                ["dashboard_decision_label", "runtime_ticket_status", "live_safety_status", "runtime_action", "operation_action"],
            )
        )
        reason = text(first_value(row, ["buy_reason_summary", "runtime_reason", "risk_reason_summary", "stake_adjustment_summary"]))
        queue_lap = queue_lap_shadow_context(row)
        rows.append(
            {
                "raceId": race_id,
                "ticketType": ticket_type,
                "ticketLabel": ticket_type_label(ticket_type),
                "aNo": a_no,
                "bNo": b_no,
                "aName": text(first_value(row, ["anchor_name", "anchor_horse_name", "horse_name"])) or a_info.get("horseName", ""),
                "bName": text(first_value(row, ["partner_name", "partner_horse_name"])) or b_info.get("horseName", ""),
                "stakeYen": stake,
                "action": action or "買い",
                "reason": humanize_reason_summary(reason),
                "liveOdds": live.get("odds") or first_number(row, ["runtime_odds", "live_odds", "quote_odds_proxy"]),
                "livePay": live.get("payPer100") or first_number(row, ["runtime_pay_per100", "live_pay_per100", "quote_pay_proxy_per100"]),
                "minOdds": first_number(row, ["min_acceptable_odds"]),
                "expectedRoi": first_number(row, ["runtime_expected_roi", "expected_roi_after_slippage"]),
                "pacePairLabel": text(row.get("pace_pair_gate_label")),
                "pacePairNote": text(row.get("pace_pair_gate_note")),
                "pacePairScore": first_number(row, ["continuous_pair_formal_score"]),
                "pacePairFitScore": first_number(row, ["continuous_pair_pace_fit_score"]),
                "pacePairReadabilityScore": first_number(row, ["continuous_pair_readability_score"]),
                "closerShadowLabel": text(row.get("closer_shadow_label")),
                "closerShadowNote": text(row.get("closer_shadow_note")),
                "closerShadowScore": first_number(row, ["closer_shadow_score"]),
                "frontContextLabel": text(row.get("front_context_gate_label")),
                "frontContextNote": text(row.get("front_context_gate_note")),
                "frontContextCollapseScore": first_number(row, ["front_context_collapse_risk_score"]),
                "frontContextSurvivalScore": first_number(row, ["front_context_survival_support_score"]),
                "lapPromoteLabel": text(row.get("lap_positive_expansion_label")),
                "lapPromoteNote": text(row.get("lap_positive_expansion_note")),
                "lapPromoteScore": first_number(row, ["lap_positive_expansion_score"]),
                "lapRoleScore": first_number(row, ["lap_axis_specialist_role_score"]),
                "productionContextLabel": text(row.get("production_context_label")),
                "productionContextAdjustment": first_number(row, ["production_context_probability_adjustment"]),
                "productionContextFrontAdjustment": first_number(row, ["prod_front_context_probability_adjustment"]),
                "productionContextLapAdjustment": first_number(row, ["prod_s_priority_probability_adjustment"]),
                "productionContextCloserAdjustment": first_number(row, ["prod_closer_course_probability_adjustment"]),
                "decisionGeneratedAt": text(row.get("runtime_decision_generated_at")),
                "generatedBeforePost": truthy(row.get("runtime_decision_generated_before_post"), default=False),
                "purchaseValid": truthy(row.get("runtime_purchase_valid"), default=True),
                "preservedAfterPost": truthy(row.get("runtime_preserved_after_post"), default=False),
                **queue_lap,
            }
        )
    rows.sort(key=lambda r: (r["raceId"], -float(r.get("stakeYen") or 0), r["ticketType"], r["aNo"], r.get("bNo") or 0))
    return rows


def build_shadow_ticket_rows(
    shadow_csv: Path | None,
    current_race_ids: set[str],
    horse_map: dict[tuple[str, int], dict[str, Any]],
    pair_lookup: dict[tuple[str, str, int, int], dict[str, Any]],
    horse_lap_decomp_lookup: dict[tuple[str, str, int, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if shadow_csv is None or not shadow_csv.exists():
        return []
    shadow = read_csv_safe(shadow_csv, dtype={"race_id": str})
    if shadow.empty:
        return []
    rows: list[dict[str, Any]] = []

    def shadow_fail_reasons(row: pd.Series) -> list[str]:
        margin = first_number(row, ["min_odds_margin_ratio"]) or 0.0
        expected = first_number(row, ["runtime_expected_roi", "expected_roi_after_slippage", "joint_v2_live_expected_roi"]) or 0.0
        score_value = first_number(row, ["strongest_current_score"]) or 0.0
        skip_risk = first_number(row, ["skip_risk_score"]) or 0.0
        front5 = first_number(row, ["projected_front5_prob"]) or 0.0
        pace = first_number(row, ["pace_fit_pair_score"]) or 0.0
        front_value = first_number(row, ["position_front_value_score"]) or 0.0
        workout = first_number(row, ["workout_pair_score"]) or 0.0
        live_odds = first_number(row, ["live_odds", "runtime_odds"]) or 0.0
        first_unc = first_number(row, ["first_condition_pair_uncertainty_score"]) or 0.0
        danger = first_number(row, ["ticket_danger_popular_in_pair_score"]) or 0.0
        difficulty = first_number(row, ["race_difficulty_score"]) or 0.0
        fast_clock = first_number(row, ["fast_clock_pair_score"]) or 0.0
        corner_shape = first_number(row, ["corner_shape_pair_score"]) or 0.0
        corner_low_sample = first_number(row, ["corner_shape_low_sample_count"]) or 0.0
        hakodate = max(first_number(row, ["anchor_runtime_hakodate_flag"]) or 0.0, first_number(row, ["partner_runtime_hakodate_flag"]) or 0.0)
        soft_heavy = max(first_number(row, ["anchor_runtime_soft_heavy_flag"]) or 0.0, first_number(row, ["partner_runtime_soft_heavy_flag"]) or 0.0)
        odds_ready = first_number(row, ["odds_timeline_ready"]) or 0.0
        edge_override = margin >= 3.0 and expected >= 1.6
        reasons: list[str] = []
        if score_value < 0.86:
            reasons.append("\u5019\u88dc\u5f37\u5ea6\u304c\u8cb7\u3044\u6c34\u6e96\u306b\u5c4a\u304b\u306a\u3044")
        if margin < 2.50:
            reasons.append("\u30aa\u30c3\u30ba\u5999\u5473\u304c\u8cb7\u3044\u6c34\u6e96\u306b\u5c4a\u304b\u306a\u3044")
        if skip_risk > 0.45:
            reasons.append("\u898b\u9001\u308a\u30ea\u30b9\u30af\u304c\u9ad8\u3081")
        if front5 < 0.60:
            reasons.append("\u524d\u306b\u884c\u3051\u308b\u898b\u8fbc\u307f\u304c\u4e0d\u8db3")
        if pace < 0.35:
            reasons.append("\u5c55\u958b\u9069\u6027\u304c\u4e0d\u8db3")
        if front_value < 0.51:
            reasons.append("\u524d\u76ee\u3067\u904b\u3076\u4fa1\u5024\u304c\u4e0d\u8db3")
        if workout < 0.20:
            reasons.append("\u8abf\u6559\u306e\u52dd\u8ca0\u30d1\u30bf\u30fc\u30f3\u52a0\u70b9\u304c\u4e0d\u8db3")
        if live_odds > 120.0:
            reasons.append("\u30aa\u30c3\u30ba\u904e\u5927\u3067\u30ce\u30a4\u30ba\u5927")
        if first_unc >= 0.45 and not edge_override:
            reasons.append("\u521d\u6761\u4ef6\u30fb\u7d4c\u9a13\u4e0d\u8db3\u306e\u4e0d\u78ba\u5b9f\u6027\u304c\u9ad8\u3081")
        if danger >= 0.42 and not edge_override:
            reasons.append("\u5371\u967a\u4eba\u6c17\u99ac\u30ea\u30b9\u30af\u304c\u9ad8\u3081")
        if difficulty > 0.58 and not edge_override:
            reasons.append("\u30ec\u30fc\u30b9\u96e3\u6613\u5ea6\u304c\u9ad8\u3081")
        if fast_clock < 0.24 and not edge_override:
            reasons.append("\u901f\u3044\u6642\u8a08\u3078\u306e\u8010\u6027\u304c\u4e0d\u8db3")
        if corner_low_sample >= 2 and corner_shape < 0.70:
            reasons.append("\u30b3\u30fc\u30ca\u30fc\u5f62\u72b6\u3078\u306e\u7d4c\u9a13\u304c\u4e0d\u8db3")
        if hakodate > 0:
            reasons.append("\u51fd\u9928\u30ac\u30fc\u30c9\u4e2d")
        if soft_heavy > 0:
            reasons.append("\u9053\u60aa\u30ac\u30fc\u30c9\u4e2d")
        if odds_ready < 1:
            reasons.append("T-5/T-3\u76f4\u524d\u30aa\u30c3\u30ba\u672a\u78ba\u5b9a")
        return list(dict.fromkeys(reasons))

    for _, row in shadow.iterrows():
        race_id = text(row.get("race_id"))
        if not (race_id.isdigit() and len(race_id) == 16):
            race_id = row_to_official_race_id(row)
        if race_id not in current_race_ids:
            continue
        ticket_type = text(first_value(row, ["ticket_type", "ticketType"])) or "wide"
        a_no, b_no = parse_ticket_numbers(row)
        if not a_no or not b_no:
            continue
        row = row_with_horse_lap_decomp(row, horse_lap_decomp_lookup or {}, race_id, ticket_type, a_no, b_no)
        a_info = row_horse_info(horse_map, race_id, a_no)
        b_info = row_horse_info(horse_map, race_id, b_no)
        lo, hi = sorted([a_no, b_no])
        live = pair_lookup.get((race_id, ticket_type, lo, hi), {})
        joint_prob = first_number(row, ["joint_v2_wide_prob"])
        joint_ev = first_number(row, ["joint_v2_wide_ev_proxy"])
        fail_reasons = shadow_fail_reasons(row)
        reason_parts = ["V2\u30da\u30a2\u78ba\u7387\u30ac\u30fc\u30c9\u306f\u901a\u904e"]
        if fail_reasons:
            reason_parts.append("BUY\u6607\u683c\u3067\u304d\u306a\u3044\u7406\u7531: " + " / ".join(fail_reasons[:4]))
        if joint_prob is not None:
            if joint_prob >= 0.12:
                reason_parts.append("\u540c\u6642\u597d\u8d70\u78ba\u7387\u306f\u9ad8\u3081")
            elif joint_prob >= 0.07:
                reason_parts.append("\u540c\u6642\u597d\u8d70\u78ba\u7387\u306f\u6a19\u6e96")
            else:
                reason_parts.append("\u540c\u6642\u597d\u8d70\u78ba\u7387\u306f\u4f4e\u3081")
        if joint_ev is not None:
            if joint_ev >= 2.0:
                reason_parts.append("\u5999\u5473\u306f\u304b\u306a\u308a\u5927\u304d\u3044")
            elif joint_ev >= 1.3:
                reason_parts.append("\u5999\u5473\u306f\u3042\u308b")
            elif joint_ev >= 1.0:
                reason_parts.append("\u5999\u5473\u306f\u30ae\u30ea\u30ae\u30ea")
            else:
                reason_parts.append("\u5999\u5473\u306f\u8584\u3044")
        queue_lap = queue_lap_shadow_context(row)
        if queue_lap.get("queueLapNote"):
            reason_parts.append(str(queue_lap["queueLapNote"]))
        rows.append({
            "raceId": race_id,
            "ticketType": ticket_type,
            "ticketLabel": ticket_type_label(ticket_type),
            "aNo": a_no,
            "bNo": b_no,
            "aName": text(first_value(row, ["anchor_name", "anchor_horse_name", "aName", "anchor_\u99ac\u540d"])) or a_info.get("horseName", ""),
            "bName": text(first_value(row, ["partner_name", "partner_horse_name", "bName", "partner_\u99ac\u540d"])) or b_info.get("horseName", ""),
            "stakeYen": 0,
            "action": "\u975e\u8cfc\u5165\u30fb\u5f37\u3081\u898b\u9001\u308a\uff08V2\u53c2\u8003\uff09",
            "reason": " / ".join(reason_parts),
            "liveOdds": live.get("odds") or first_number(row, ["live_odds", "runtime_odds", "quote_odds_proxy"]),
            "livePay": live.get("payPer100") or first_number(row, ["live_pay_per100", "runtime_pay_per100", "quote_pay_proxy_per100"]),
            "minOdds": first_number(row, ["min_acceptable_odds"]),
            "expectedRoi": first_number(row, ["runtime_expected_roi", "expected_roi_after_slippage"]),
            "pacePairLabel": text(row.get("pace_pair_gate_label")),
            "pacePairNote": text(row.get("pace_pair_gate_note")),
            "pacePairScore": first_number(row, ["continuous_pair_formal_score"]),
            "pacePairFitScore": first_number(row, ["continuous_pair_pace_fit_score"]),
            "pacePairReadabilityScore": first_number(row, ["continuous_pair_readability_score"]),
            "closerShadowLabel": text(row.get("closer_shadow_label")),
            "closerShadowNote": text(row.get("closer_shadow_note")),
            "closerShadowScore": first_number(row, ["closer_shadow_score"]),
            "frontContextLabel": text(row.get("front_context_gate_label")),
            "frontContextNote": text(row.get("front_context_gate_note")),
            "frontContextCollapseScore": first_number(row, ["front_context_collapse_risk_score"]),
            "frontContextSurvivalScore": first_number(row, ["front_context_survival_support_score"]),
            "lapPromoteLabel": text(row.get("lap_positive_expansion_label")),
            "lapPromoteNote": text(row.get("lap_positive_expansion_note")),
            "lapPromoteScore": first_number(row, ["lap_positive_expansion_score"]),
            "lapRoleScore": first_number(row, ["lap_axis_specialist_role_score"]),
            "jointProb": joint_prob,
            "jointEv": joint_ev,
            "strongestScore": first_number(row, ["strongest_current_score"]),
            "reference": True,
            "shadowType": "pair_v2",
            "referencePriority": 0.90 + 0.10 * (first_number(row, ["strongest_current_score"]) or 0.0),
            **queue_lap,
        })
    rows.sort(key=lambda r: (r["raceId"], -float(r.get("referencePriority") or 0), -float(r.get("expectedRoi") or 0), r["aNo"], r.get("bNo") or 0))
    return rows


def build_shape_shadow_ticket_rows(
    candidates_csv: Path | None,
    current_race_ids: set[str],
    horse_map: dict[tuple[str, int], dict[str, Any]],
    pair_lookup: dict[tuple[str, str, int, int], dict[str, Any]],
    course_ten_race_context: dict[str, dict[str, Any]] | None = None,
    horse_lap_decomp_lookup: dict[tuple[str, str, int, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if candidates_csv is None or not candidates_csv.exists():
        return []
    cand = read_csv_safe(candidates_csv, dtype={"race_id": str})
    if cand.empty or "race_id" not in cand.columns:
        return []

    cand = cand.copy()
    cand["race_id"] = cand["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    cand = cand[cand["race_id"].isin(current_race_ids)].copy()
    if cand.empty:
        return []

    def series_num(name: str, default: float = 0.0) -> pd.Series:
        if name in cand.columns:
            return pd.to_numeric(cand[name], errors="coerce").fillna(default)
        return pd.Series(default, index=cand.index, dtype=float)

    def clipped(name: str, default: float = 0.0) -> pd.Series:
        return series_num(name, default).clip(0.0, 1.0)

    if "horse_a" not in cand.columns:
        cand["horse_a"] = pd.to_numeric(cand.get("anchor_horse_no"), errors="coerce")
    if "horse_b" not in cand.columns:
        cand["horse_b"] = pd.to_numeric(cand.get("partner_horse_no"), errors="coerce")
    cand["ticket_type"] = cand.get("ticket_type", "umaren").fillna("umaren").astype(str)

    base = clipped("pair_quinella_score", 0.5)
    base_rank = base.groupby(cand["race_id"]).rank(pct=True).fillna(0.5)
    overlay = clipped("market_overlay_score", 0.0)
    late = clipped("late_value_survives_score", 0.0)
    value_score = (0.55 * overlay + 0.45 * late).clip(0.0, 1.0)
    front_max = clipped("projected_front5_prob", 0.0)
    anchor_front = clipped("anchor_projected_front5_prob", 0.0)
    partner_front = clipped("partner_projected_front5_prob", 0.0)
    front_min = np.minimum(anchor_front, partner_front).clip(0.0, 1.0)
    closer_max = clipped("closer_pair_max", 0.0)
    diversity = clipped("style_diversity_score", 0.0)
    clash = clipped("front_front_clash", 0.0)
    front_slow = clipped("front_front_slow_fit", 0.0)
    collapse = clipped("collapse_fit", 0.0)
    duel = clipped("race_front_duel_risk_score", 0.0)
    clarity = clipped("race_queue_clarity_score", 0.0)
    label = cand.get("race_pace_shape_label", pd.Series("mixed_queue", index=cand.index)).fillna("mixed_queue").astype(str)

    single_fit = (0.46 * front_max + 0.30 * front_slow + 0.14 * front_min + 0.10 * base).clip(0.0, 1.0)
    no_clear_fit = (0.50 * front_max + 0.22 * front_slow + 0.18 * base + 0.10 * overlay).clip(0.0, 1.0)
    duel_fit = (0.35 * closer_max + 0.24 * collapse + 0.21 * diversity + 0.12 * front_max + 0.08 * base).clip(0.0, 1.0)
    matched_fit = (0.30 * closer_max + 0.25 * collapse + 0.24 * diversity + 0.13 * front_max + 0.08 * base).clip(0.0, 1.0)
    mixed_fit = (0.30 * base + 0.25 * front_max + 0.25 * closer_max + 0.20 * diversity).clip(0.0, 1.0)
    fit = pd.Series(0.5, index=cand.index, dtype=float)
    fit = fit.where(~label.eq("single_leader_clear"), single_fit)
    fit = fit.where(~label.eq("no_clear_leader"), no_clear_fit)
    fit = fit.where(~label.eq("front_duel_dense"), duel_fit)
    fit = fit.where(~label.eq("matched_speed_duel"), matched_fit)
    fit = fit.where(~label.eq("mixed_queue"), mixed_fit)
    front_burn = (duel * clash * (0.45 + 0.55 * front_min)).clip(0.0, 1.0)
    dead_slow_closer = ((1.0 - duel) * clarity * closer_max * (1.0 - front_max)).clip(0.0, 1.0)
    no_clear_uncertainty = (label.eq("no_clear_leader").astype(float) * (1.0 - front_max) * 0.35).clip(0.0, 1.0)
    risk = pd.Series(
        np.maximum.reduce([front_burn.to_numpy(), dead_slow_closer.to_numpy(), no_clear_uncertainty.to_numpy()]),
        index=cand.index,
    ).clip(0.0, 1.0)

    cand["shape_shadow_fit_score"] = fit.clip(0.0, 1.0)
    cand["shape_shadow_risk_score"] = risk
    cand["shape_shadow_score"] = (
        0.30 * base_rank
        + 0.70 * cand["shape_shadow_fit_score"]
        + 0.06 * value_score
        - 0.12 * cand["shape_shadow_risk_score"]
    ).clip(0.0, 1.0)

    anchor_odds = series_num("anchor_live_win_odds", np.nan).fillna(series_num("anchor_odds", 999.0))
    partner_odds = series_num("partner_live_win_odds", np.nan).fillna(series_num("partner_odds", 999.0))
    danger_pair = clipped("ticket_danger_popular_in_pair_score", np.nan).fillna(clipped("ticket_danger_popular_score", 0.0))
    gate = (
        base.ge(0.64)
        & value_score.ge(0.68)
        & series_num("skip_risk_score", 1.0).le(0.42)
        & danger_pair.le(0.85)
        & partner_odds.between(5.0, 80.0)
        & anchor_odds.between(2.0, 60.0)
        & cand["ticket_type"].str.lower().eq("umaren")
    )
    pool = cand[gate].copy()
    if pool.empty:
        return []

    base_top = (
        pool.sort_values(["race_id", "pair_quinella_score", "market_overlay_score"], ascending=[True, False, False])
        .drop_duplicates("race_id", keep="first")
        [["race_id", "horse_a", "horse_b"]]
        .copy()
    )
    base_top["base_key"] = base_top.apply(
        lambda r: f"{int(min(r['horse_a'], r['horse_b']))}-{int(max(r['horse_a'], r['horse_b']))}", axis=1
    )
    top = (
        pool.sort_values(["race_id", "shape_shadow_score", "shape_shadow_fit_score", "market_overlay_score"], ascending=[True, False, False, False])
        .drop_duplicates("race_id", keep="first")
        .copy()
    )
    top = top.merge(base_top[["race_id", "base_key"]], on="race_id", how="left")
    top["shape_key"] = top.apply(
        lambda r: f"{int(min(r['horse_a'], r['horse_b']))}-{int(max(r['horse_a'], r['horse_b']))}", axis=1
    )
    top["shape_changed_from_base"] = top["shape_key"].ne(top["base_key"])

    def shape_label_jp(value: str) -> str:
        labels = {
            "single_leader_clear": "\u5358\u9a0e\u9003\u3052\u6fc3\u539a",
            "front_duel_dense": "\u5148\u884c\u5bc6\u96c6",
            "matched_speed_duel": "\u30c6\u30f3\u4e89\u3044",
            "no_clear_leader": "\u9003\u3052\u4e0d\u5728",
            "mixed_queue": "\u968a\u5217\u6df7\u5728",
        }
        return labels.get(text(value), text(value, "\u968a\u5217\u4e0d\u660e"))

    rows: list[dict[str, Any]] = []
    for _, row in top.iterrows():
        race_id = text(row.get("race_id"))
        a_no = int(num(row.get("horse_a")) or num(row.get("anchor_horse_no")) or 0)
        b_no = int(num(row.get("horse_b")) or num(row.get("partner_horse_no")) or 0)
        if not a_no or not b_no:
            continue
        row = row_with_horse_lap_decomp(row, horse_lap_decomp_lookup or {}, race_id, "umaren", a_no, b_no)
        lo, hi = sorted([a_no, b_no])
        live = pair_lookup.get((race_id, "umaren", lo, hi), {})
        a_info = row_horse_info(horse_map, race_id, a_no)
        b_info = row_horse_info(horse_map, race_id, b_no)
        label_jp = shape_label_jp(text(row.get("race_pace_shape_label")))
        changed_note = "\u5f93\u6765\u4e0a\u4f4d\u304b\u3089\u5165\u66ff" if truthy(row.get("shape_changed_from_base"), default=False) else "\u5f93\u6765\u4e0a\u4f4d\u3082\u7dad\u6301"
        course_ten = (course_ten_race_context or {}).get(race_id, {})
        course_ten_note = text(course_ten.get("courseTenRaceLabel"))
        if course_ten_note:
            pressure_value = float(course_ten.get("courseTenPressureScore") or 0)
            queue_value = float(course_ten.get("courseTenQueueClarityScore") or 0)
            pressure_label = "\u30c6\u30f3\u8ca0\u8377\u9ad8\u3081" if pressure_value >= 0.50 else "\u30c6\u30f3\u8ca0\u8377\u6a19\u6e96" if pressure_value >= 0.30 else "\u30c6\u30f3\u8ca0\u8377\u8efd\u3081"
            queue_label = "\u968a\u5217\u8aad\u307f\u3084\u3059\u3044" if queue_value >= 0.65 else "\u968a\u5217\u6a19\u6e96" if queue_value >= 0.35 else "\u968a\u5217\u8aad\u307f\u306b\u304f\u3044"
            course_ten_note = f" / {course_ten_note} / {pressure_label} / {queue_label}"
        fit_score = float(row.get("shape_shadow_fit_score") or 0)
        risk_score = float(row.get("shape_shadow_risk_score") or 0)
        fit_label = "\u9069\u6027\u9ad8\u3081" if fit_score >= 0.70 else "\u9069\u6027\u6a19\u6e96" if fit_score >= 0.45 else "\u9069\u6027\u4f4e\u3081"
        risk_label = "\u30ea\u30b9\u30af\u9ad8\u3081" if risk_score >= 0.55 else "\u30ea\u30b9\u30af\u6a19\u6e96" if risk_score >= 0.30 else "\u30ea\u30b9\u30af\u4f4e\u3081"
        reason = f"\u5c55\u958b\u88dc\u6b63\u30b7\u30e3\u30c9\u30fc: {label_jp} / {fit_label} / {risk_label} / {changed_note}{course_ten_note}"
        queue_lap = queue_lap_shadow_context(row)
        if queue_lap.get("queueLapNote"):
            reason = f"{reason} / {queue_lap['queueLapNote']}"
        rows.append(
            {
                "raceId": race_id,
                "ticketType": "umaren",
                "ticketLabel": ticket_type_label("umaren"),
                "aNo": a_no,
                "bNo": b_no,
                "aName": text(first_value(row, ["anchor_馬名", "anchor_name", "anchor_horse_name"])) or a_info.get("horseName", ""),
                "bName": text(first_value(row, ["partner_馬名", "partner_name", "partner_horse_name"])) or b_info.get("horseName", ""),
                "stakeYen": 0,
                "action": "非購入・展開補正シャドー",
                "reason": reason,
                "liveOdds": live.get("odds") or first_number(row, ["live_odds", "runtime_odds", "quote_odds_proxy"]),
                "livePay": live.get("payPer100") or first_number(row, ["live_pay_per100", "runtime_pay_per100", "quote_pay_proxy_per100"]),
                "minOdds": first_number(row, ["min_acceptable_odds"]),
                "expectedRoi": first_number(row, ["runtime_expected_roi", "expected_roi_after_slippage"]),
                "pacePairLabel": text(row.get("pace_pair_gate_label")),
                "pacePairNote": text(row.get("pace_pair_gate_note")),
                "pacePairScore": first_number(row, ["continuous_pair_formal_score"]),
                "pacePairFitScore": first_number(row, ["continuous_pair_pace_fit_score"]),
                "pacePairReadabilityScore": first_number(row, ["continuous_pair_readability_score"]),
                "closerShadowLabel": text(row.get("closer_shadow_label")),
                "closerShadowNote": text(row.get("closer_shadow_note")),
                "closerShadowScore": first_number(row, ["closer_shadow_score"]),
                "frontContextLabel": text(row.get("front_context_gate_label")),
                "frontContextNote": text(row.get("front_context_gate_note")),
                "frontContextCollapseScore": first_number(row, ["front_context_collapse_risk_score"]),
                "frontContextSurvivalScore": first_number(row, ["front_context_survival_support_score"]),
                "lapPromoteLabel": text(row.get("lap_positive_expansion_label")),
                "lapPromoteNote": text(row.get("lap_positive_expansion_note")),
                "lapPromoteScore": first_number(row, ["lap_positive_expansion_score"]),
                "lapRoleScore": first_number(row, ["lap_axis_specialist_role_score"]),
                "shapeAdjustedScore": float(row.get("shape_shadow_score") or 0.0),
                "shapeFit": float(row.get("shape_shadow_fit_score") or 0.0),
                "shapeRisk": float(row.get("shape_shadow_risk_score") or 0.0),
                "shapeLabel": label_jp,
                "changedFromBase": truthy(row.get("shape_changed_from_base"), default=False),
                "reference": True,
                "shadowType": "shape_umaren",
                "referencePriority": 0.92 + 0.08 * float(row.get("shape_shadow_score") or 0.0),
                **queue_lap,
            }
        )
    rows.sort(
        key=lambda r: (
            r["raceId"],
            -float(r.get("referencePriority") or 0),
            -float(r.get("shapeAdjustedScore") or 0),
            r["aNo"],
            r.get("bNo") or 0,
        )
    )
    return rows


def build_payload(
    single_csv: Path,
    pair_csv: Path,
    entry_csv: Path | None,
    prediction_csv: Path | None,
    tickets_csv: Path | None,
    candidates_csv: Path | None = None,
    wide_shadow_csv: Path | None = None,
    course_ten_context_csv: Path | None = None,
    historical_condition_context_csv: Path | None = None,
    horse_lap_decomp_csv: Path | None = None,
    body_weight_csv: Path | None = None,
    track_condition_csv: Path | None = None,
    result_track_csv: Path | None = None,
    gelding_history_csv: Path | None = None,
    default_date_key: str = "",
) -> dict[str, Any]:
    single = read_csv_safe(single_csv, dtype={"race_id": str}) if single_csv.exists() else pd.DataFrame()
    pair = read_csv_safe(pair_csv, dtype={"race_id": str}) if pair_csv.exists() else pd.DataFrame()
    body = read_csv_safe(body_weight_csv, dtype={"race_id": str}) if body_weight_csv and body_weight_csv.exists() else pd.DataFrame()
    gelding_history = read_gelding_history(gelding_history_csv)
    if entry_csv and entry_csv.exists() and not gelding_history.empty:
        entry_frame = read_csv_safe(entry_csv)
        entry_frame = enrich_current_entries_with_gelding_context(entry_frame, gelding_history)
        tmp_entry = ROOT / "outputs" / "analysis" / "current_strongest_runtime_v1" / "entry_with_gelding_context_for_dashboard.csv"
        tmp_entry.parent.mkdir(parents=True, exist_ok=True)
        entry_frame.to_csv(tmp_entry, index=False, encoding="utf-8-sig")
        entry_csv = tmp_entry
    track_condition_map = load_track_condition_map(track_condition_csv)
    result_track_condition_map = load_result_track_condition_map(result_track_csv)
    horse_map, race_meta = build_entry_map(entry_csv, prediction_csv)
    candidate_runner_context = load_candidate_runner_context(candidates_csv)
    course_ten_runner_context, course_ten_race_context = load_course_ten_context(course_ten_context_csv)
    historical_condition_context = load_historical_condition_context(historical_condition_context_csv)
    horse_lap_decomp_lookup = load_horse_lap_decomp_lookup(horse_lap_decomp_csv)
    for key, overlay_info in candidate_runner_context.items():
        info = horse_map.setdefault(key, {})
        info.update(overlay_info)
        info["points"] = _runner_points(info)
        info["concerns"] = _runner_concerns(info)
    for key, overlay_info in course_ten_runner_context.items():
        info = horse_map.setdefault(key, {})
        info.update(overlay_info)
        points = list(info.get("points") or _runner_points(info))
        concerns = list(info.get("concerns") or _runner_concerns(info))
        note = text(overlay_info.get("courseTenRunnerNote"))
        if "テン速い" in note:
            points.insert(0, "クラス×コース基準でテン速い")
        elif "テン不足" in note:
            concerns.append("クラス×コース基準ではテン不足")
        info["points"] = list(dict.fromkeys(points))[:4]
        info["concerns"] = list(dict.fromkeys(concerns))[:4]
    body_map: dict[tuple[str, int], dict[str, Any]] = {}
    if not body.empty:
        body = body.rename(
            columns={
                "馬番": "horse_no",
                "馬体重": "body_weight",
                "増減": "body_weight_diff",
                "馬体重増減": "body_weight_diff",
            }
        )
        if {"race_id", "horse_no", "body_weight"}.issubset(body.columns):
            body["race_id"] = body["race_id"].astype(str)
            body["horse_no"] = pd.to_numeric(body["horse_no"], errors="coerce").astype("Int64")
            body["body_weight"] = pd.to_numeric(body["body_weight"], errors="coerce")
            if "body_weight_diff" not in body.columns:
                body["body_weight_diff"] = pd.NA
            body["body_weight_diff"] = pd.to_numeric(body["body_weight_diff"], errors="coerce")
            body = body[body["race_id"].ne("") & body["horse_no"].notna() & body["body_weight"].notna()].copy()
            body = body.sort_values(["race_id", "horse_no", "snapshot_at" if "snapshot_at" in body.columns else "body_weight"])
            body = body.drop_duplicates(["race_id", "horse_no"], keep="last")
            for _, row in body.iterrows():
                body_map[(text(row.get("race_id")), int(row.get("horse_no")))] = {
                    "bodyWeight": num(row.get("body_weight")),
                    "bodyWeightDiff": num(row.get("body_weight_diff")),
                    "bodyWeightSnapshot": text(row.get("snapshot_at")),
                }

    races: dict[str, dict[str, Any]] = {}
    singles: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    pair_lookup: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    snapshot_values: list[str] = []

    if not single.empty:
        for _, row in single.iterrows():
            meta = parse_official_race_id(row.get("race_id"))
            race_id = meta["raceId"]
            horse_no = int(num(row.get("horse_no")) or 0)
            info = row_horse_info(horse_map, race_id, horse_no)
            body_info = body_map.get((race_id, horse_no), {})
            snapshot = text(row.get("snapshot_at"))
            if snapshot:
                snapshot_values.append(snapshot)
            races.setdefault(race_id, {**meta, **race_meta.get(race_id, {})})
            win_odds = odds_num(row.get("live_win_odds"))
            popularity = int_text(row.get("live_popularity")) or info.get("popularitySnapshot", "")
            concerns = list(info.get("concerns") or [])
            if win_odds is not None and win_odds <= 3.0:
                concerns.append("市場評価が高く妙味確認")
            if popularity and int_text(popularity) and int(int_text(popularity)) >= 8:
                concerns.append("人気薄で再現性確認")
            body_diff = num(body_info.get("bodyWeightDiff"))
            if body_diff is not None and abs(body_diff) >= 16:
                concerns.append("馬体重増減大")
            if not concerns:
                concerns.append("大きな減点は未検出")
            singles.append(
                {
                    "raceId": race_id,
                    "horseNo": horse_no,
                    "frameNo": info.get("frameNo", ""),
                    "horseName": info.get("horseName", ""),
                    "jockey": info.get("jockey", ""),
                    "carriedWeight": info.get("carriedWeight"),
                    "carriedWeightText": carried_weight_label(info.get("carriedWeight")),
                    "aiRank": int_text(info.get("aiRank")),
                    "aiScore": num(info.get("aiScore")),
                    "expectedPace": info.get("expectedPace", ""),
                    "frontRunning": num(info.get("frontRunning")),
                    "closing": num(info.get("closing")),
                    "courseTenSpeed": num(info.get("courseTenSpeed")),
                    "courseTenBest": num(info.get("courseTenBest")),
                    "courseTenFastStartRate": num(info.get("courseTenFastStartRate")),
                    "courseTenHistoryAvailable": num(info.get("courseTenHistoryAvailable")),
                    "courseTenRunnerNote": info.get("courseTenRunnerNote", ""),
                    "confidence": num(info.get("confidence")),
                    "confidenceBucket": info.get("confidenceBucket", ""),
                    "points": list(info.get("points") or ["前日材料で暫定評価"])[:4],
                    "concerns": concerns[:4],
                    "winOdds": win_odds,
                    "popularity": popularity,
                    "bodyWeight": num(body_info.get("bodyWeight")),
                    "bodyWeightDiff": body_diff,
                    "bodyWeightText": body_weight_label(body_info.get("bodyWeight"), body_info.get("bodyWeightDiff")),
                    "bodyWeightSnapshot": body_info.get("bodyWeightSnapshot", ""),
                    "placeMin": odds_num(row.get("live_place_odds_min")),
                    "placeMax": odds_num(row.get("live_place_odds_max")),
                    "predayWinOdds": info.get("predayWinOdds"),
                    "snapshotAt": snapshot,
                    "snapshotLabel": format_snapshot(snapshot),
                }
            )

    desired_dates = {text(r.get("dateKey")) for r in races.values() if text(r.get("dateKey"))}
    entry_races, entry_runners = build_entry_display_rows(entry_csv, prediction_csv, desired_dates, body_map)
    for race_id, meta in entry_races.items():
        existing = races.setdefault(race_id, meta)
        for key, value in meta.items():
            if key not in existing or existing.get(key) in ("", None):
                existing[key] = value

    existing_runner_keys = {(text(r.get("raceId")), int(num(r.get("horseNo")) or 0)) for r in singles}
    for runner in entry_runners:
        key = (text(runner.get("raceId")), int(num(runner.get("horseNo")) or 0))
        overlay_info = candidate_runner_context.get(key, {})
        if overlay_info:
            runner.update(overlay_info)
            points = list(runner.get("points") or [])
            concerns = list(runner.get("concerns") or [])
            if num(overlay_info.get("firstConditionImpact")) is not None and num(overlay_info.get("firstConditionImpact")) >= 0.60:
                points.insert(0, "少キャリアでも前走内容強い")
            if num(overlay_info.get("firstConditionUncertainty")) is not None and num(overlay_info.get("firstConditionUncertainty")) >= 0.45:
                concerns.append("少キャリア/初条件の不確実性")
            runner["points"] = list(dict.fromkeys(points))[:4]
            runner["concerns"] = list(dict.fromkeys(concerns))[:4]
        if key not in existing_runner_keys:
            singles.append(runner)
            existing_runner_keys.add(key)

    runners_by_race: dict[str, list[dict[str, Any]]] = {}
    for runner in singles:
        runners_by_race.setdefault(text(runner.get("raceId")), []).append(runner)
    for race_id, race in races.items():
        race.update(race_upset_forecast(race, runners_by_race.get(race_id, [])))

    if not pair.empty:
        for _, row in pair.iterrows():
            meta = parse_official_race_id(row.get("race_id"))
            race_id = meta["raceId"]
            a_no = int(num(row.get("a_no")) or 0)
            b_no = int(num(row.get("b_no")) or 0)
            a_info = row_horse_info(horse_map, race_id, a_no)
            b_info = row_horse_info(horse_map, race_id, b_no)
            snapshot = text(row.get("snapshot_at"))
            if snapshot:
                snapshot_values.append(snapshot)
            races.setdefault(race_id, {**meta, **race_meta.get(race_id, {})})
            lo, hi = sorted([a_no, b_no])
            pair_lookup[(race_id, text(row.get("ticket_type")), lo, hi)] = {
                "odds": num(row.get("live_odds")),
                "payPer100": num(row.get("live_pay_per100")),
            }
            pairs.append(
                {
                    "raceId": race_id,
                    "ticketType": text(row.get("ticket_type")),
                    "aNo": a_no,
                    "bNo": b_no,
                    "aName": a_info.get("horseName", ""),
                    "bName": b_info.get("horseName", ""),
                    "odds": num(row.get("live_odds")),
                    "payPer100": num(row.get("live_pay_per100")),
                    "popularity": int_text(row.get("popularity")),
                    "snapshotAt": snapshot,
                    "snapshotLabel": format_snapshot(snapshot),
                }
            )

    apply_track_condition_to_races(races, track_condition_map, result_track_condition_map)
    for race_id, context in course_ten_race_context.items():
        if race_id in races:
            races[race_id].update({k: v for k, v in context.items() if v not in ("", None)})
    historical_condition_races = apply_historical_condition_context(races, historical_condition_context)
    race_list = list(races.values())
    race_list.sort(key=lambda r: (r.get("dateKey", ""), r.get("venueCode", ""), int(r.get("raceNo") or 0)))
    current_race_ids = {r["raceId"] for r in race_list}
    ticket_rows = build_ticket_rows(tickets_csv, current_race_ids, horse_map, pair_lookup, horse_lap_decomp_lookup)
    v2_shadow_ticket_rows = build_shadow_ticket_rows(
        wide_shadow_csv, current_race_ids, horse_map, pair_lookup, horse_lap_decomp_lookup
    )
    shape_shadow_ticket_rows = build_shape_shadow_ticket_rows(
        candidates_csv, current_race_ids, horse_map, pair_lookup, course_ten_race_context, horse_lap_decomp_lookup
    )
    shadow_ticket_rows = v2_shadow_ticket_rows + shape_shadow_ticket_rows
    race_decision_rows = build_race_decision_rows(candidates_csv, current_race_ids)
    for decision in race_decision_rows:
        race = races.get(text(decision.get("raceId")))
        if not race:
            continue
        for src, dst in {
            "queueClarityScore": "queueClarityScore",
            "frontDuelRiskScore": "frontDuelRiskScore",
            "frontLoadScore": "frontLoadScore",
            "front5TopGap": "front5TopGap",
            "front5CandidateCount": "front5CandidateCount",
            "paceShapeLabel": "paceShapeLabel",
        }.items():
            value = decision.get(src)
            if value not in ("", None) and not pd.isna(value):
                race[dst] = value
    max_snapshot = max(snapshot_values) if snapshot_values else ""
    return {
        "generatedAt": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "defaultDate": default_date_key,
        "latestSnapshot": max_snapshot,
        "latestSnapshotLabel": format_snapshot(max_snapshot),
        "source": "JRA公式オッズ",
        "counts": {
            "races": len(race_list),
            "singleRows": len(singles),
            "pairRows": len(pairs),
            "umarenRows": sum(1 for r in pairs if r["ticketType"] == "umaren"),
            "wideRows": sum(1 for r in pairs if r["ticketType"] == "wide"),
            "ticketRows": len(ticket_rows),
            "ticketRaces": len({r["raceId"] for r in ticket_rows}),
            "shadowTicketRows": len(shadow_ticket_rows),
            "shadowTicketRaces": len({r["raceId"] for r in shadow_ticket_rows}),
            "v2ShadowTicketRows": len(v2_shadow_ticket_rows),
            "v2ShadowTicketRaces": len({r["raceId"] for r in v2_shadow_ticket_rows}),
            "shapeShadowTicketRows": len(shape_shadow_ticket_rows),
            "shapeShadowTicketRaces": len({r["raceId"] for r in shape_shadow_ticket_rows}),
            "courseTenContextRows": len(course_ten_runner_context),
            "courseTenContextRaces": len(course_ten_race_context),
            "historicalConditionRaces": historical_condition_races,
            "horseLapDecompPairs": len(horse_lap_decomp_lookup),
            "decisionRows": len(race_decision_rows),
        },
        "races": race_list,
        "singleRows": singles,
        "pairRows": pairs,
        "ticketRows": ticket_rows,
        "shadowTicketRows": shadow_ticket_rows,
        "raceDecisionRows": race_decision_rows,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0">
  <meta http-equiv="Pragma" content="no-cache">
  <meta http-equiv="Expires" content="0">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ライブオッズ</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d8dee7;
      --soft: #eef3f5;
      --accent: #0f766e;
      --accent-soft: #dff4ef;
      --warn: #b45309;
      --warn-soft: #fff4dc;
      --danger: #b42318;
      --danger-soft: #fee4e2;
      --radius: 8px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      position: static;
      background: rgba(245, 247, 248, 0.96);
      border-bottom: 1px solid var(--line);
    }
    .bar {
      max-width: 1180px;
      margin: 0 auto;
      padding: 10px 16px 9px;
      display: grid;
      gap: 8px;
    }
    .title-row {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
    }
    .status {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 26px;
      padding: 4px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      white-space: nowrap;
    }
    .pill.ok { background: var(--accent-soft); color: #075e56; border-color: #a6d8ce; }
    .controls {
      display: grid;
      grid-template-columns: minmax(150px, 1fr) minmax(170px, 1fr) minmax(236px, 0.9fr);
      gap: 8px;
      align-items: end;
    }
    .hidden-control {
      display: none;
    }
    label {
      display: grid;
      gap: 4px;
      font-size: 11px;
      color: var(--muted);
      font-weight: 650;
    }
    select, input {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      color: var(--ink);
      padding: 7px 9px;
      font: inherit;
      font-size: 14px;
    }
    .refresh-control {
      position: relative;
      display: grid;
      grid-template-columns: minmax(112px, 1fr) minmax(112px, 1fr);
      gap: 8px;
      min-width: 236px;
      padding-top: 17px;
      align-self: end;
    }
    .refresh-button {
      width: 100%;
      min-height: 38px;
      border: 1px solid #0f766e;
      border-radius: var(--radius);
      background: #0f766e;
      color: #fff;
      font: inherit;
      font-size: 14px;
      font-weight: 750;
      cursor: pointer;
    }
    .refresh-button.secondary {
      border-color: var(--line);
      background: var(--panel);
      color: var(--ink);
    }
    .refresh-button:disabled {
      opacity: 0.68;
      cursor: progress;
    }
    .refresh-status {
      position: absolute;
      left: 0;
      top: 58px;
      width: 100%;
      min-height: 14px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.2;
    }
    .button-filters {
      display: grid;
      grid-template-columns: minmax(0, 0.65fr) minmax(0, 1.35fr);
      gap: 8px;
    }
    .button-filters .button-filter-row:nth-child(3) {
      grid-column: 1 / -1;
    }
    .button-filter-row {
      display: grid;
      gap: 5px;
    }
    .button-filter-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }
    .button-strip {
      display: flex;
      gap: 7px;
      overflow-x: auto;
      padding-bottom: 2px;
      scrollbar-width: thin;
    }
    .filter-button {
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      color: var(--ink);
      padding: 7px 10px;
      font: inherit;
      font-size: 13px;
      font-weight: 750;
      white-space: nowrap;
      cursor: pointer;
    }
    .venue-button {
      display: grid;
      gap: 2px;
      text-align: left;
      min-width: 118px;
    }
    .venue-button .venue-name {
      font-size: 13px;
      font-weight: 850;
      line-height: 1.1;
    }
    .venue-button .venue-next {
      font-size: 10px;
      font-weight: 750;
      color: inherit;
      opacity: 0.72;
      line-height: 1.1;
    }
    .filter-button:hover {
      border-color: #9fb4c0;
      background: #f8fbfb;
    }
    .filter-button.active {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    .race-button-pill {
      display: grid;
      gap: 3px;
      width: 160px;
      min-width: 160px;
      max-width: 160px;
      text-align: left;
      border-width: 2px;
      overflow: hidden;
    }
    .race-button-pill .race-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      min-width: 0;
    }
    .race-button-pill .race-no {
      font-size: 13px;
      font-weight: 850;
    }
    .race-button-pill .decision-tag {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 0;
      max-width: 96px;
      min-height: 20px;
      padding: 2px 6px;
      border: 1px solid currentColor;
      border-radius: 999px;
      font-size: 10px;
      font-weight: 850;
      line-height: 1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .race-button-pill .race-name {
      max-width: 132px;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 11px;
      font-weight: 650;
      color: inherit;
      opacity: 0.86;
    }
    .race-button-pill .race-time {
      font-size: 10px;
      font-weight: 700;
      color: inherit;
      opacity: 0.72;
    }
    .race-button-pill .race-status-row {
      display: flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
      overflow: hidden;
    }
    .race-button-pill .post-tag {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 19px;
      padding: 2px 6px;
      border-radius: 999px;
      border: 1px solid #94a3b8;
      background: #e2e8f0;
      color: #334155;
      font-size: 10px;
      font-weight: 850;
      line-height: 1;
      white-space: nowrap;
    }
    .filter-button.buy { border-color: #0f766e; background: #e5f5f2; color: #075e56; }
    .filter-button.candidate { border-color: #d97706; background: #fff4df; color: #92400e; }
    .filter-button.closer-watch { border-color: #8b5cf6; background: #f3e8ff; color: #6d28d9; }
    .filter-button.watch { border-color: #2563eb; background: #eff6ff; color: #1d4ed8; }
    .filter-button.weak { border-color: #cbd5e1; background: #f8fafc; color: #475569; }
    .filter-button.skip { border-color: #efb0ac; background: #fff7f6; color: #9f2a21; }
    .filter-button.wait { border-color: #f0c37d; background: #fffbeb; color: #92400e; }
    .filter-button.active {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    .filter-button.active .decision-tag {
      border-color: rgba(255,255,255,0.7);
      background: rgba(255,255,255,0.16);
      color: #fff;
    }
    .filter-button.finished {
      border-color: #94a3b8;
      background: #eef2f7;
      color: #64748b;
      opacity: 0.74;
    }
    .filter-button.finished.active {
      border-color: #334155;
      background: #475569;
      color: #fff;
      opacity: 0.92;
    }
    .filter-button.finished.active .post-tag {
      border-color: rgba(255,255,255,0.7);
      background: rgba(255,255,255,0.16);
      color: #fff;
    }
    .filter-button.post-finished {
      opacity: 0.78;
    }
    .filter-button.post-finished .race-name,
    .filter-button.post-finished .race-time {
      opacity: 0.64;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 14px 16px 36px;
      display: grid;
      gap: 14px;
    }
    .race-head {
      display: grid;
      gap: 8px;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
    }
    .race-head-top {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 10px;
      flex-wrap: wrap;
    }
    h2 {
      margin: 0;
      font-size: 19px;
      line-height: 1.25;
    }
    .meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }
    .lap-context-panel {
      display: grid;
      gap: 8px;
      padding: 10px 11px;
      border: 1px solid #dbe5ee;
      border-radius: var(--radius);
      background: #f8fbfc;
    }
    .lap-context-panel.empty {
      display: none;
    }
    .lap-context-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 12px;
      color: var(--muted);
    }
    .lap-context-title {
      color: #17202b;
      font-weight: 750;
      font-size: 13px;
    }
    .lap-context-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(84px, 1fr));
      gap: 7px;
    }
    .lap-context-cell {
      min-width: 0;
      padding: 7px 8px;
      border: 1px solid #e1e8ef;
      border-radius: 8px;
      background: #fff;
    }
    .lap-context-label {
      display: block;
      color: var(--muted);
      font-size: 10px;
      line-height: 1.2;
      white-space: nowrap;
    }
    .lap-context-value {
      display: block;
      margin-top: 2px;
      color: #17202b;
      font-weight: 750;
      font-size: 13px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .lap-context-comment {
      color: #465363;
      font-size: 12px;
      line-height: 1.45;
    }
    .lap-context-badge {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 8px;
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      background: #fff;
      color: #334155;
      font-weight: 700;
    }
    .lap-context-badge.fast {
      border-color: #f7b7a3;
      background: #fff1ec;
      color: #b4411c;
    }
    .lap-context-badge.slow {
      border-color: #aac9f6;
      background: #eef6ff;
      color: #1d4ed8;
    }
    .lap-context-badge.neutral {
      border-color: #bed7cb;
      background: #effaf4;
      color: #166534;
    }
    .decision-pill {
      border-color: #a6d8ce;
      background: var(--accent-soft);
      color: #075e56;
      font-weight: 750;
    }
    .decision-pill.candidate {
      border-color: #f0c37d;
      background: var(--warn-soft);
      color: var(--warn);
    }
    .decision-pill.closer-watch {
      border-color: #c4b5fd;
      background: #f3e8ff;
      color: #6d28d9;
    }
    .decision-pill.watch {
      border-color: #93c5fd;
      background: #dbeafe;
      color: #1d4ed8;
    }
    .decision-pill.weak {
      border-color: #cbd5e1;
      background: #f1f5f9;
      color: #475569;
    }
    .decision-pill.skip {
      border-color: #f3b6b2;
      background: var(--danger-soft);
      color: #9f2a21;
    }
    .decision-pill.wait {
      border-color: #f0c37d;
      background: var(--warn-soft);
      color: var(--warn);
    }
    .decision-pill.finished {
      border-color: #94a3b8;
      background: #e2e8f0;
      color: #334155;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .pair-control,
    .pair-odds-section {
      display: none;
    }
    section {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fafbfc;
    }
    .section-head h3 {
      margin: 0;
      font-size: 15px;
    }
    .section-head .mini {
      font-size: 12px;
      color: var(--muted);
    }
    .ticket-note {
      display: grid;
      gap: 5px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      color: #465363;
      font-size: 12px;
      line-height: 1.55;
    }
    .ticket-note strong {
      color: #17202b;
      font-weight: 750;
    }
    .win5-section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .win5-body {
      display: grid;
      gap: 10px;
      padding: 12px;
    }
    .win5-summary {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }
    .win5-budget-row {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }
    .win5-budget-buttons {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .win5-leg-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
    }
    .win5-leg {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fff;
      padding: 9px;
      min-width: 0;
    }
    .win5-leg h4 {
      margin: 0 0 6px;
      font-size: 13px;
      line-height: 1.25;
    }
    .win5-horses {
      display: grid;
      gap: 4px;
      font-size: 12px;
      line-height: 1.35;
    }
    .win5-horse {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      border-top: 1px solid #edf0f3;
      padding-top: 4px;
    }
    .win5-horse strong {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .win5-horse.picked {
      color: #075e56;
      font-weight: 750;
    }
    .win5-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .table-wrap {
      overflow: auto;
      max-height: calc(100vh - 260px);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 8px 9px;
      border-bottom: 1px solid #edf0f3;
      vertical-align: middle;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f7f9fb;
      color: #526071;
      font-size: 12px;
      text-align: left;
    }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr:hover td { background: #f6faf9; }
    .frame-cell {
      text-align: center;
      min-width: 42px;
    }
    .frame-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 30px;
      height: 26px;
      padding: 0 6px;
      border-radius: 5px;
      border: 1px solid #aeb7c2;
      font-weight: 850;
      font-variant-numeric: tabular-nums;
    }
    .frame-1 { background: #ffffff; color: #111827; }
    .frame-2 { background: #111827; color: #ffffff; border-color: #111827; }
    .frame-3 { background: #dc2626; color: #ffffff; border-color: #dc2626; }
    .frame-4 { background: #2563eb; color: #ffffff; border-color: #2563eb; }
    .frame-5 { background: #facc15; color: #111827; border-color: #eab308; }
    .frame-6 { background: #16a34a; color: #ffffff; border-color: #16a34a; }
    .frame-7 { background: #f97316; color: #ffffff; border-color: #f97316; }
    .frame-8 { background: #ec4899; color: #ffffff; border-color: #ec4899; }
    .horse {
      white-space: normal;
      min-width: 120px;
      font-weight: 650;
    }
    .ticket-section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .ticket-section.reference {
      border-color: #f0c37d;
    }
    .ticket-section.reference .section-head {
      background: var(--warn-soft);
    }
    .ticket-section.reference #ticketTitle {
      color: var(--warn);
    }
    .sub {
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 500;
    }
    .chips {
      display: flex;
      gap: 4px;
      flex-wrap: wrap;
      min-width: 170px;
      white-space: normal;
    }
    .chips span {
      display: inline-flex;
      align-items: center;
      min-height: 23px;
      padding: 3px 7px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--soft);
      font-size: 11px;
      line-height: 1.2;
    }
    .chips.good span { color: #075e56; border-color: #a6d8ce; background: var(--accent-soft); }
    .chips.bad span { color: #9f2a21; border-color: #f3b6b2; background: var(--danger-soft); }
    .rank {
      font-weight: 800;
      color: var(--accent);
      font-variant-numeric: tabular-nums;
    }
    .odds {
      font-weight: 750;
      font-variant-numeric: tabular-nums;
    }
    .odds.fav { color: var(--danger); }
    .odds.mid { color: var(--accent); }
    .odds.long { color: var(--warn); }
    .blank {
      padding: 28px 14px;
      text-align: center;
      color: var(--muted);
    }
    @media (max-width: 860px) {
      header {
        position: static;
        backdrop-filter: none;
      }
      .bar {
        padding-top: 8px;
        padding-bottom: 8px;
        gap: 6px;
      }
      .title-row {
        display: none;
      }
      .status {
        gap: 5px;
        flex-wrap: nowrap;
        overflow-x: auto;
        padding-bottom: 1px;
      }
      .pill {
        min-height: 23px;
        padding: 3px 7px;
        font-size: 11px;
      }
      .controls { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
      .refresh-control {
        grid-column: 1 / -1;
        min-width: 0;
        padding-top: 0;
      }
      .refresh-status {
        position: static;
        grid-column: 1 / -1;
        min-height: 12px;
      }
      .button-filters {
        gap: 4px;
      }
      .button-filter-row {
        gap: 3px;
      }
      .button-filter-label {
        display: none;
      }
      .button-strip {
        gap: 5px;
        padding-bottom: 1px;
      }
      .filter-button {
        min-height: 28px;
        padding: 4px 8px;
        font-size: 12px;
      }
      .race-button-pill {
        width: 126px;
        min-width: 126px;
        max-width: 126px;
        gap: 1px;
        padding: 4px 7px;
      }
      .race-button-pill .race-no {
        font-size: 12px;
      }
      .race-button-pill .decision-tag {
        max-width: 72px;
        min-height: 18px;
        padding: 2px 5px;
        font-size: 9px;
      }
      .race-button-pill .race-name {
        max-width: 106px;
        font-size: 10px;
      }
      .race-button-pill .race-time {
        font-size: 9px;
      }
      main {
        padding-top: 8px;
        gap: 9px;
      }
      .grid { grid-template-columns: 1fr; }
      .win5-leg-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .table-wrap { max-height: none; }
      th, td { padding: 8px 7px; }
      .hide-sm { display: none; }
    }
    @media (max-width: 520px) {
      .controls { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
      .refresh-control {
        grid-template-columns: 1fr 1fr;
        gap: 6px;
      }
      .refresh-button {
        min-height: 32px;
        font-size: 12px;
        padding: 6px 7px;
      }
      .bar, main { padding-left: 10px; padding-right: 10px; }
      h1 {
        position: absolute;
        width: 1px;
        height: 1px;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
      }
      h2 { font-size: 17px; }
      select, input {
        min-height: 34px;
        font-size: 13px;
        padding: 6px 8px;
      }
      label {
        gap: 3px;
        font-size: 10px;
      }
      .race-button-pill {
        width: 112px;
        min-width: 112px;
        max-width: 112px;
      }
      .race-button-pill .decision-tag {
        max-width: 58px;
      }
      .race-button-pill .race-name {
        max-width: 92px;
      }
      .race-head {
        padding: 8px;
        gap: 5px;
      }
      .race-head-top {
        gap: 6px;
      }
      .race-head-top .status {
        width: auto;
        min-width: 0;
        flex-wrap: nowrap;
        overflow-x: auto;
      }
      .meta {
        gap: 5px;
        font-size: 11px;
        flex-wrap: nowrap;
        overflow-x: auto;
        padding-bottom: 1px;
      }
      .lap-context-panel {
        padding: 8px;
        gap: 6px;
      }
      .lap-context-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 5px;
      }
      .lap-context-cell {
        padding: 6px 7px;
      }
      .lap-context-value {
        font-size: 12px;
      }
      .lap-context-comment {
        font-size: 11px;
      }
      .section-head {
        padding: 8px 9px;
      }
      .ticket-note {
        padding: 8px 9px;
        gap: 3px;
        font-size: 11px;
        line-height: 1.42;
      }
      .ticket-section .table-wrap th,
      .ticket-section .table-wrap td {
        padding-top: 6px;
        padding-bottom: 6px;
      }
      table { font-size: 12px; }
      .horse { min-width: 104px; }
      .win5-leg-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div class="title-row">
        <h1>ライブオッズ</h1>
        <div class="status" id="status"></div>
      </div>
      <div class="controls">
        <label>開催日<select id="dateSelect"></select></label>
        <label>判定<select id="decisionFilter"></select></label>
        <label class="hidden-control">競馬場<select id="venueSelect"></select></label>
        <label class="hidden-control">レース<select id="raceSelect"></select></label>
        <label class="pair-control">ペア券種<select id="ticketType"><option value="wide">ワイド</option><option value="umaren">馬連</option></select></label>
        <div class="refresh-control">
          <button type="button" class="refresh-button" id="refreshRaceButton">このレース更新</button>
          <button type="button" class="refresh-button secondary" id="refreshAllButton">全レース更新</button>
          <span class="refresh-status" id="refreshStatus"></span>
        </div>
      </div>
      <div class="button-filters">
        <div class="button-filter-row">
          <div class="button-filter-label">表示</div>
          <div class="button-strip" id="viewModeButtons">
            <button type="button" class="filter-button active" data-view-mode="race">通常レース</button>
            <button type="button" class="filter-button" data-view-mode="win5">WIN5</button>
          </div>
        </div>
        <div class="button-filter-row">
          <div class="button-filter-label">競馬場</div>
          <div class="button-strip" id="venueButtons"></div>
        </div>
        <div class="button-filter-row">
          <div class="button-filter-label">レース</div>
          <div class="button-strip" id="raceButtons"></div>
        </div>
      </div>
    </div>
  </header>
  <main>
    <section class="win5-section" id="win5Section">
      <div class="section-head">
        <h3>WIN5専用モデル</h3>
        <span class="mini" id="win5Count"></span>
      </div>
      <div class="win5-body">
        <div class="win5-budget-row">
          <span>予算</span>
          <div class="win5-budget-buttons" id="win5BudgetButtons"></div>
        </div>
        <div class="win5-summary" id="win5Summary"></div>
        <div class="win5-leg-grid" id="win5Legs"></div>
        <div class="win5-note" id="win5Note"></div>
      </div>
    </section>
    <div class="race-head" id="raceHead">
      <div class="race-head-top">
        <h2 id="raceTitle">レースを選択</h2>
        <div class="status"><span class="pill ok" id="snapshotPill"></span><span class="pill decision-pill" id="decisionPill"></span></div>
      </div>
      <div class="meta" id="raceMeta"></div>
      <div class="lap-context-panel empty" id="lapContextPanel"></div>
    </div>
    <section class="ticket-section" id="ticketSection">
      <div class="section-head">
        <h3 id="ticketTitle">最終買い目</h3>
        <span class="mini" id="ticketCount"></span>
      </div>
      <div class="ticket-note" id="ticketNote"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>券種</th>
              <th>組み合わせ</th>
              <th class="num">購入額</th>
              <th class="num">現在オッズ</th>
              <th class="num hide-sm">最低オッズ</th>
              <th class="hide-sm">理由</th>
            </tr>
          </thead>
          <tbody id="ticketBody"></tbody>
        </table>
      </div>
    </section>
    <div class="grid" id="raceGrid">
      <section>
        <div class="section-head">
          <h3>単勝・複勝</h3>
          <span class="mini" id="singleCount"></span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th class="num">AI</th>
                <th class="num">馬番</th>
                <th>馬名</th>
                <th class="num">単勝</th>
                <th class="num">複勝</th>
                <th class="num hide-sm">人気</th>
                <th class="num hide-sm">斤量</th>
                <th class="num hide-sm">馬体重</th>
                <th>評価ポイント</th>
                <th>懸念点</th>
              </tr>
            </thead>
            <tbody id="singleBody"></tbody>
          </table>
        </div>
      </section>
      <section class="pair-odds-section" aria-hidden="true">
        <div class="section-head">
          <h3 id="pairTitle">ワイド</h3>
          <span class="mini" id="pairCount"></span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>組み合わせ</th>
                <th class="num">オッズ</th>
                <th class="num hide-sm">100円払戻</th>
                <th class="num hide-sm">人気</th>
              </tr>
            </thead>
            <tbody id="pairBody"></tbody>
          </table>
        </div>
      </section>
    </div>
  </main>
  <script>
    const payload = __PAYLOAD__;
    const races = payload.races || [];
    const singles = payload.singleRows || [];
    const pairs = payload.pairRows || [];
    const tickets = payload.ticketRows || [];
    const shadowTickets = payload.shadowTicketRows || [];
    const raceDecisions = payload.raceDecisionRows || [];
    const win5 = payload.win5 || {};
    const byId = new Map(races.map(r => [r.raceId, r]));
    const decisionByRace = new Map(raceDecisions.map(r => [r.raceId, r]));
    const ticketsByRace = new Map();
    tickets.forEach(t => {
      if (!ticketsByRace.has(t.raceId)) ticketsByRace.set(t.raceId, []);
      ticketsByRace.get(t.raceId).push(t);
    });
    const shadowTicketsByRace = new Map();
    shadowTickets.forEach(t => {
      if (!shadowTicketsByRace.has(t.raceId)) shadowTicketsByRace.set(t.raceId, []);
      shadowTicketsByRace.get(t.raceId).push(t);
    });
    const dateSelect = document.getElementById('dateSelect');
    const venueSelect = document.getElementById('venueSelect');
    const decisionFilter = document.getElementById('decisionFilter');
    const raceSelect = document.getElementById('raceSelect');
    const ticketType = document.getElementById('ticketType');
    const refreshRaceButton = document.getElementById('refreshRaceButton');
    const refreshAllButton = document.getElementById('refreshAllButton');
    const refreshStatus = document.getElementById('refreshStatus');
    const viewModeButtons = document.getElementById('viewModeButtons');
    const win5BudgetButtons = document.getElementById('win5BudgetButtons');
    const venueButtons = document.getElementById('venueButtons');
    const raceButtons = document.getElementById('raceButtons');
    let currentViewMode = 'race';
    let currentWin5Budget = 10000;

    function uniq(arr) { return [...new Set(arr.filter(Boolean))]; }
    function yen(v) { return Number.isFinite(v) ? `${Math.round(v).toLocaleString()}円` : ''; }
    function odds(v) { return Number.isFinite(v) ? v.toFixed(v >= 100 ? 0 : 1) : ''; }
    function score(v) { return Number.isFinite(v) ? v.toFixed(3) : ''; }
    function score2(v) { return Number.isFinite(v) ? v.toFixed(2) : ''; }
    function pct(v) { return Number.isFinite(v) ? `${(v * 100).toFixed(0)}%` : ''; }
    function metricBand(value, kind) {
      const v = Number(value);
      if (!Number.isFinite(v)) return '';
      if (kind === 'pressure') {
        if (v >= 0.50) return '高め';
        if (v >= 0.30) return '標準';
        return '軽め';
      }
      if (kind === 'duel') {
        if (v >= 0.70) return '高め';
        if (v >= 0.45) return 'やや高め';
        if (v >= 0.25) return '低め';
        return 'かなり低め';
      }
      if (kind === 'clarity') {
        if (v >= 0.65) return '読みやすい';
        if (v >= 0.35) return '標準';
        return '読みにくい';
      }
      if (kind === 'load') {
        if (v >= 0.65) return 'かなり重い';
        if (v >= 0.45) return '重め';
        if (v >= 0.25) return '標準';
        return '軽め';
      }
      if (kind === 'priority') {
        if (v >= 0.85) return 'A級';
        if (v >= 0.65) return '高め';
        if (v >= 0.45) return '注視';
        return '低め';
      }
      if (kind === 'strength') {
        if (v >= 0.86) return '買い水準';
        if (v >= 0.75) return 'かなり強い';
        if (v >= 0.60) return '強め';
        if (v >= 0.45) return '標準';
        return '弱め';
      }
      if (kind === 'confidence') {
        if (v >= 0.65) return '高め';
        if (v >= 0.35) return '標準';
        if (v >= 0.20) return '低め';
        return 'かなり低め';
      }
      if (kind === 'frontProb') {
        if (v >= 0.70) return '高い';
        if (v >= 0.60) return '買い水準';
        if (v >= 0.45) return '標準';
        return '低め';
      }
      if (kind === 'edge') {
        if (v >= 3.00) return 'かなり大きい';
        if (v >= 2.50) return '買い水準';
        if (v >= 1.50) return '少しあり';
        return '薄い';
      }
      if (kind === 'ev') {
        if (v >= 2.00) return '大きい';
        if (v >= 1.30) return 'あり';
        if (v >= 1.00) return 'ぎりぎり';
        return '不足';
      }
      if (kind === 'risk') {
        if (v >= 0.60) return '高い';
        if (v >= 0.45) return 'やや高い';
        if (v >= 0.25) return '標準';
        return '低い';
      }
      if (kind === 'aiScore') {
        if (v >= 0.70) return '最上位級';
        if (v >= 0.55) return '上位級';
        if (v >= 0.40) return '標準';
        return '低め';
      }
      if (v >= 0.66) return '高め';
      if (v >= 0.33) return '標準';
      return '低め';
    }
    function metricNote(kind, band) {
      if (kind === 'pressure') {
        if (band === '高め') return '前に行く馬への消耗が大きくなりやすい';
        if (band === '標準') return '極端な消耗までは見込みにくい';
        return '前が楽をしやすい';
      }
      if (kind === 'duel') {
        if (band === '高め') return '先行勢がぶつかりやすく、隊列が決まるまで脚を使いやすい';
        if (band === 'やや高め') return '前の並び次第で消耗が出やすい';
        return '先行争いは落ち着きやすい';
      }
      if (kind === 'clarity') {
        if (band === '読みやすい') return '逃げ・好位の並びを比較的読みやすい';
        if (band === '標準') return '隊列はある程度読めるが決め打ちはしにくい';
        return '逃げ馬や好位勢の並びが読みにくい';
      }
      if (kind === 'load') {
        if (band === 'かなり重い' || band === '重め') return '前半で脚を使う想定';
        if (band === '標準') return '平均的な前半負荷';
        return '前半は緩みやすい想定';
      }
      if (kind === 'risk') {
        if (band === '高い' || band === 'やや高い') return '買いに上げるには不安が残る';
        if (band === '標準') return '極端な不安まではない';
        return '見送り要因は小さい';
      }
      return '';
    }
    function metricReadable(label, value, kind) {
      const v = Number(value);
      if (!Number.isFinite(v)) return '';
      const band = metricBand(v, kind);
      const note = metricNote(kind, band);
      return `${label}: ${band}（${score2(v)}）${note ? ` = ${note}` : ''}`;
    }
    function metricChip(label, value, kind) {
      const v = Number(value);
      if (!Number.isFinite(v)) return '';
      return `${label} ${metricBand(v, kind)}`;
    }
    function metricCompact(label, value, kind) {
      const v = Number(value);
      if (!Number.isFinite(v)) return '';
      const band = metricBand(v, kind);
      return `${label}: ${band}`;
    }
    function aiScoreLabel(value) {
      const v = Number(value);
      if (!Number.isFinite(v)) return '';
      const band = v >= 0.70 ? '\u6700\u4e0a\u4f4d\u7d1a'
        : v >= 0.55 ? '\u4e0a\u4f4d\u7d1a'
        : v >= 0.40 ? '\u6a19\u6e96'
        : '\u4f4e\u3081';
      return `AI\u30b9\u30b3\u30a2 ${score(v)}\uff08${band}\uff09`;
    }
    function courseTenSpeedLabel(value) {
      const v = Number(value);
      if (!Number.isFinite(v)) return '';
      if (v >= 1.00) return '補正テン かなり速い';
      if (v >= 0.35) return '補正テン 速め';
      if (v > -0.35) return '補正テン 標準';
      if (v > -1.00) return '補正テン 遅め';
      return '補正テン かなり遅め';
    }
    function runningTendencyLabel(label, value) {
      const v = Number(value);
      if (!Number.isFinite(v)) return '';
      if (v >= 0.60) return `${label}高い`;
      if (v >= 0.40) return `${label}標準`;
      if (v >= 0.20) return `${label}少しあり`;
      return `${label}低め`;
    }
    const AI_SCORE_GAP_BENCHMARK = {
      median: 0.04977569591934494,
      top25: 0.094654,
      top10: 0.144283,
      winCandidate: 0.176303,
    };
    function aiScoreGapLabel(race) {
      const gap = Number(race && race.scoreGap);
      if (!Number.isFinite(gap)) return '';
      let band = '差は薄め';
      if (gap >= AI_SCORE_GAP_BENCHMARK.winCandidate) band = '単勝検討ライン';
      else if (gap >= AI_SCORE_GAP_BENCHMARK.top10) band = '上位10%相当';
      else if (gap >= AI_SCORE_GAP_BENCHMARK.top25) band = '上位25%相当';
      else if (gap >= AI_SCORE_GAP_BENCHMARK.median) band = '中央値以上';
      return `AI1位の抜け具合: ${band}`;
    }
    function paceLabel(v) {
      if (v === 'slow') return 'スロー';
      if (v === 'middle') return 'ミドル';
      if (v === 'fast') return 'ハイ';
      return v || '';
    }
    function paceShapeLabel(v) {
      const labels = {
        single_leader_clear: '単騎逃げ濃厚',
        front_duel_dense: '先行密集',
        matched_speed_duel: 'テン互角',
        no_clear_leader: '逃げ不在',
        mixed_queue: '隊列混在',
      };
      return labels[String(v || '')] || String(v || '');
    }
    function clsOdds(v) {
      if (!Number.isFinite(v)) return '';
      if (v <= 3.0) return 'fav';
      if (v <= 15.0) return 'mid';
      return 'long';
    }
    function pairTypeLabel(t) {
      const labels = { wide: 'ワイド', umaren: '馬連', quinella: '馬連', win: '単勝', place: '複勝', umatan: '馬単', exacta: '馬単', trio: '3連複', trifecta: '3連単' };
      return labels[String(t || '').toLowerCase()] || t || '';
    }
    function renderWin5() {
      const section = document.getElementById('win5Section');
      if (!section) return;
      const legs = win5.legs || [];
      const plans = win5.plans || [];
      if (!legs.length || !plans.length) {
        section.style.display = 'none';
        return;
      }
      section.style.display = '';
      const availableBudgets = plans.map(p => Number(p.budgetYen)).filter(Number.isFinite);
      if (!availableBudgets.includes(Number(currentWin5Budget))) {
        currentWin5Budget = availableBudgets.includes(10000) ? 10000 : availableBudgets[availableBudgets.length - 1];
      }
      if (win5BudgetButtons) {
        win5BudgetButtons.innerHTML = availableBudgets.map(budget => `
          <button type="button" class="filter-button ${Number(currentWin5Budget) === budget ? 'active' : ''}" data-win5-budget="${budget}">
            ${budget.toLocaleString()}円
          </button>`).join('');
      }
      const mainPlan = plans.find(p => Number(p.budgetYen) === Number(currentWin5Budget)) || plans[plans.length - 1] || {};
      const selectionByLeg = new Map((mainPlan.selections || []).map(x => [Number(x.legNo), x]));
      document.getElementById('win5Count').textContent = `${legs.length}レース / ${mainPlan.combos || 0}点`;
      document.getElementById('win5Summary').innerHTML = [
        `<span class="pill ok">推奨 ${mainPlan.stakeYen ? Math.round(mainPlan.stakeYen).toLocaleString() : 0}円</span>`,
        `<span class="pill">${mainPlan.combos || 0}点</span>`,
        `<span class="pill">予算 ${mainPlan.budgetYen ? Math.round(mainPlan.budgetYen).toLocaleString() : 0}円</span>`,
        `<span class="pill">推定的中率 ${pct(Number(mainPlan.estimatedHitProb || 0))}</span>`,
        `<span class="pill">${win5.targetSource === 'explicit' ? '公式指定' : '対象R自動推定'}</span>`,
      ].join('');
      document.getElementById('win5Legs').innerHTML = legs.map(leg => {
        const selected = selectionByLeg.get(Number(leg.legNo)) || {};
        const selectedNos = new Set((selected.horses || []).map(h => Number(h.horseNo)));
        const candidates = (leg.candidates || []).slice(0, Math.max(Number(selected.count || 0), 4));
        const horses = candidates.map(h => {
          const picked = selectedNos.has(Number(h.horseNo));
          const oddsText = Number.isFinite(Number(h.odds)) ? ` / ${odds(Number(h.odds))}倍` : '';
          const popText = h.popularity ? ` / ${h.popularity}人気` : '';
          return `<div class="win5-horse ${picked ? 'picked' : ''}">
            <strong>${picked ? '✓ ' : ''}${h.horseNo}. ${h.horseName || ''}</strong>
            <span>${pct(Number(h.prob || 0))}${oddsText}${popText}</span>
          </div>`;
        }).join('');
        return `<div class="win5-leg">
          <h4>${leg.legNo}. ${leg.venue}${leg.raceNo}R ${leg.raceName || ''}</h4>
          <div class="sub">${leg.difficulty_label || ''} / 採用${selected.count || leg.base_count || 0}頭 / カバー${pct(Number(selected.coverageProb || 0))}</div>
          <div class="win5-horses">${horses}</div>
        </div>`;
      }).join('');
      const budgetRows = plans.map(p => `${Math.round(p.budgetYen).toLocaleString()}円: ${p.combos}点/${Math.round(p.stakeYen).toLocaleString()}円`).join('　');
      document.getElementById('win5Note').textContent = `${budgetRows}。現段階はWIN5対象レースを自動推定しています。公式対象レースが取れたらrace_id指定で差し替えます。`;
    }
    function horseLabel(no, name) { return name ? `${no}. ${name}` : `${no}`; }
    function raceShortLabel(race) {
      const no = race && race.raceNo ? `${race.raceNo}R` : '';
      const name = (race && race.raceName ? String(race.raceName) : '').replace(/\\s+/g, ' ').trim();
      return { no, name };
    }
    function racePostDateTime(race) {
      if (!race || !race.startTime) return null;
      const dateDigits = String(race.dateKey || race.date || '').replace(/\\D/g, '');
      const timeMatch = String(race.startTime || '').match(/(\\d{1,2}):(\\d{2})/);
      if (dateDigits.length < 8 || !timeMatch) return null;
      const year = Number(dateDigits.slice(0, 4));
      const month = Number(dateDigits.slice(4, 6)) - 1;
      const day = Number(dateDigits.slice(6, 8));
      const hour = Number(timeMatch[1]);
      const minute = Number(timeMatch[2]);
      if (![year, month, day, hour, minute].every(Number.isFinite)) return null;
      return new Date(year, month, day, hour, minute, 0, 0);
    }
    function raceHasStarted(race) {
      const postTime = racePostDateTime(race);
      return postTime ? Date.now() >= postTime.getTime() : false;
    }
    function raceTimeValue(race) {
      const postTime = racePostDateTime(race);
      return postTime ? postTime.getTime() : Number.POSITIVE_INFINITY;
    }
    function raceSortCompare(a, b) {
      const aStarted = raceHasStarted(a);
      const bStarted = raceHasStarted(b);
      if (aStarted !== bStarted) return Number(aStarted) - Number(bStarted);
      const byTime = raceTimeValue(a) - raceTimeValue(b);
      if (Number.isFinite(byTime) && byTime !== 0) return byTime;
      return String(a.venueCode).localeCompare(String(b.venueCode)) || Number(a.raceNo) - Number(b.raceNo);
    }
    function nextRaceForVenue(dateKey, venueCode) {
      const rows = races
        .filter(r => (!dateKey || r.dateKey === dateKey) && (!venueCode || r.venueCode === venueCode) && !raceHasStarted(r))
        .sort(raceSortCompare);
      return rows[0] || null;
    }
    function frameBadge(frameNo, label) {
      const frame = Number(frameNo);
      const text = label !== undefined && label !== null && String(label).trim() ? String(label).trim() : (Number.isFinite(frame) ? String(frame) : '-');
      if (!Number.isFinite(frame) || frame < 1 || frame > 8) return `<span class="frame-badge">${text}</span>`;
      return `<span class="frame-badge frame-${frame}">${text}</span>`;
    }
    function buttonClassForDecision(raceId) {
      const decision = decisionForRace(raceId);
      return decision && decision.className ? decision.className : '';
    }
    function paceReady(race) {
      return !(Number(race.frontRunnerCount || 0) === 0 && Number(race.pressureScore || 0) === 0);
    }
    function effectiveTicketsForRace(raceId) {
      const race = byId.get(raceId) || {};
      const started = raceHasStarted(race);
      return (ticketsByRace.get(raceId) || []).filter(t => {
        if (t.purchaseValid === false) return false;
        return started ? t.generatedBeforePost === true : true;
      });
    }
    function decisionForRace(raceId) {
      const raceTickets = effectiveTicketsForRace(raceId);
      if (raceTickets.length) return { key: 'buy', label: '買い目あり', className: 'buy', reason: '最終BUY条件を通過' };
      const modelDecision = decisionByRace.get(raceId);
      const shadowRows = shadowTicketsByRace.get(raceId) || [];
      const candidateShadowRows = shadowRows.filter(t => t.shadowType === 'v2_wide');
      if (candidateShadowRows.length && (!modelDecision || modelDecision.key !== 'closer_watch')) {
        const topShadow = [...candidateShadowRows].sort((a, b) => Number(b.jointEv || 0) - Number(a.jointEv || 0))[0] || {};
        const failReason = topShadow.failReasonSummary || topShadow.reason || '';
        return {
          key: 'candidate',
          label: '強め見送り（V2参考・非購入）',
          className: 'candidate',
          reason: failReason
            ? `V2ではワイド参考候補。ただしBUY昇格は不可: ${failReason}`
            : 'V2ではワイド参考候補。ただしBUY昇格に必要な最強版ゲートが未達です。',
          priorityScore: 0.88,
          expectedRoi: Number(topShadow.expectedRoi),
          topPair: {
            aNo: topShadow.aNo,
            bNo: topShadow.bNo,
            odds: topShadow.liveOdds,
          },
        };
      }
      if (modelDecision) {
        return {
          key: modelDecision.key || 'skip',
          label: modelDecision.label || '見送り',
          className: modelDecision.className || modelDecision.key || 'skip',
          priorityScore: Number(modelDecision.priorityScore),
          topScore: Number(modelDecision.topScore),
          margin: Number(modelDecision.margin),
          expectedRoi: Number(modelDecision.expectedRoi),
          skipRisk: Number(modelDecision.skipRisk),
          front5: Number(modelDecision.front5),
          reason: modelDecision.reason || '',
          guardReasons: modelDecision.guardReasons || [],
          topPair: modelDecision.topPair || {},
        };
      }
      const rows = singles.filter(x => x.raceId === raceId);
      if (!rows.length) return { key: 'wait', label: '当日材料待ち', className: 'wait' };
      const top = [...rows].sort((a, b) => Number(a.aiRank || 999) - Number(b.aiRank || 999))[0] || {};
      const strongAi = Number(top.aiRank || 999) <= 3 && (
        (Number.isFinite(top.aiScore) && top.aiScore >= 0.44) ||
        (Number.isFinite(race.scoreGap) && race.scoreGap >= 0.006) ||
        (Number.isFinite(race.confidence) && race.confidence >= 0.04)
      );
      if (strongAi) return { key: 'watch', label: '注視', className: 'watch', reason: 'AI上位だが最強版候補CSVなし' };
      return { key: 'skip', label: '見送り', className: 'skip', reason: '買い水準の候補なし' };
    }
    function decisionDisplayLabel(decision) {
      const key = decision && decision.key ? decision.key : '';
      const labels = {
        buy: 'BUY 買い目あり',
        candidate: 'A 強め見送り',
        closer_watch: 'B 差しハマり注視',
        watch: 'C 注視',
        weak: 'D 参考弱',
        wait: '待機',
        skip: '見送り',
        finished: '出走済',
      };
      return labels[key] || (decision && decision.label ? String(decision.label) : '未判定');
    }
    function compactDecisionLabel(decision) {
      const key = decision && decision.key ? decision.key : '';
      const labels = {
        buy: 'BUY',
        candidate: 'A 強め',
        closer_watch: 'B 差し',
        watch: 'C 注視',
        weak: 'D 参考',
        wait: '待機',
        skip: '見送り',
      };
      if (labels[key]) return labels[key];
      if (decision && decision.key === 'closer_watch') return '差し注視';
      const label = decision && decision.label ? String(decision.label) : '未判定';
      if (label.includes('V2ワイド')) return 'V2参考';
      if (label.includes('強め見送り')) return '強め';
      if (label.includes('準買い')) return '準候補';
      if (label.includes('買い目')) return '買い';
      if (label.includes('参考弱')) return '参考';
      return label;
    }
    function filteredRaces() {
      const d = dateSelect.value;
      const venue = venueSelect.value;
      const decision = decisionFilter.value;
      return races.filter(r => {
        if (d && r.dateKey !== d) return false;
        if (venue && r.venueCode !== venue) return false;
        if (decision === 'finished') return raceHasStarted(r);
        if (decision && decisionForRace(r.raceId).key !== decision) return false;
        return true;
      });
    }
    function raceFilterLabel(key) {
      const priorityLabels = {
        buy: 'BUY 買い目あり',
        candidate: 'A 強め見送り',
        closer_watch: 'B 差しハマり注視',
        watch: 'C 注視',
        weak: 'D 参考弱',
        wait: '待機',
        skip: '見送り',
        finished: '出走済',
      };
      if (priorityLabels[key]) return priorityLabels[key];
      if (key === 'closer_watch') return '差しハマり注視';
      const labels = { buy: '買い目あり', candidate: '強め見送り', watch: '注視', weak: '参考弱', wait: '当日材料待ち', skip: '見送り', finished: '出走済' };
      return labels[key] || '全レース';
    }
    function findPair(raceId, type, aNo, bNo) {
      const lo = Math.min(Number(aNo), Number(bNo));
      const hi = Math.max(Number(aNo), Number(bNo));
      return pairs.find(x => x.raceId === raceId && x.ticketType === type && Math.min(Number(x.aNo), Number(x.bNo)) === lo && Math.max(Number(x.aNo), Number(x.bNo)) === hi);
    }
    function referenceTicketsForRace(raceId, decision) {
      const shadowRows = shadowTicketsByRace.get(raceId) || [];
      if (shadowRows.length) {
        const shapeRows = shadowRows
          .filter(x => x.shadowType === 'shape_umaren')
          .sort((a, b) => Number(b.shapeAdjustedScore || b.referencePriority || 0) - Number(a.shapeAdjustedScore || a.referencePriority || 0))
          .slice(0, 1);
        const v2Rows = shadowRows
          .filter(x => x.shadowType !== 'shape_umaren')
          .sort((a, b) => Number(b.jointEv || 0) - Number(a.jointEv || 0))
          .slice(0, 2);
        return [...shapeRows, ...v2Rows].slice(0, 3);
      }
      const rows = singles.filter(x => x.raceId === raceId)
        .filter(x => x.aiRank)
        .sort((a, b) => Number(a.aiRank || 999) - Number(b.aiRank || 999));
      if (rows.length < 2) return [];
      const anchor = rows[0];
      const partners = rows.slice(1, 4);
      const out = [];
      const referenceMeta = (() => {
        if (decision.key === 'closer_watch') {
          return {
            action: '非購入・差し注視',
            reason: '先行勢に負荷がかかる想定。差し・好位差しの妙味はあるが、正式BUYの前目価値ゲートとは別系統のためシャドー検証。',
          };
        }
        if (decision.key === 'candidate') {
          return {
            action: '非購入・準候補',
            reason: '買い水準に近い候補。ただし最強版の最終BUY条件は未通過。',
          };
        }
        if (decision.key === 'watch') {
          return {
            action: '非購入・注視',
            reason: 'AI値や妙味はあるが、安全条件または直前条件が不足。',
          };
        }
        if (decision.key === 'weak') {
          return {
            action: '非購入・参考弱',
            reason: '確認用。買い目としては弱く、購入対象外。',
          };
        }
        return {
          action: '非購入・見送り',
          reason: '買い水準未達。AI上位の確認用としてのみ表示。',
        };
      })();
      const addTicket = (type, partner, stakeYen) => {
        const pair = findPair(raceId, type, anchor.horseNo, partner.horseNo) || {};
        out.push({
          raceId,
          ticketType: type,
          ticketLabel: ticket_type_label_js(type),
          aNo: anchor.horseNo,
          bNo: partner.horseNo,
          aName: anchor.horseName,
          bName: partner.horseName,
          stakeYen,
          action: referenceMeta.action,
          reason: referenceMeta.reason,
          liveOdds: pair.odds,
          livePay: pair.payPer100,
          minOdds: null,
          reference: true,
        });
      };
      if (partners[0]) addTicket('wide', partners[0], 100);
      if (partners[1]) addTicket('wide', partners[1], 100);
      if (partners[0]) addTicket('umaren', partners[0], 100);
      return out;
    }
    function ticket_type_label_js(ticketType) {
      return pairTypeLabel(ticketType);
    }
    function pacePairLabelText(label) {
      const map = {
        pace_pair_strong: '\u5c55\u958b\u00d7\u30da\u30a2\u9069\u6027\u5f37',
        pace_pair_watch: '\u5c55\u958b\u00d7\u30da\u30a2\u6ce8\u8996',
        pace_pair_caution: '\u5c55\u958b\u00d7\u30da\u30a2\u5f31\u3081',
        pace_pair_neutral: '\u5c55\u958b\u00d7\u30da\u30a2\u4e2d\u7acb',
      };
      return map[label] || '';
    }
    function closerShadowLabelText(label) {
      const map = {
        closer_watch_strong: '\u5dee\u3057\u6d6e\u4e0a\u3092\u5f37\u3081\u306b\u8b66\u6212',
        closer_watch: '\u5dee\u3057\u6d6e\u4e0a\u306b\u6ce8\u610f',
        closer_neutral: '',
      };
      return map[label] || '';
    }
    function frontContextLabelText(label) {
      const map = {
        front_context_collapse_alert: '\u524d\u5d29\u308c\u6587\u8108\u304c\u5f37\u3044',
        front_context_collapse_watch: '\u524d\u5d29\u308c\u6587\u8108\u306b\u6ce8\u610f',
        front_context_survival_watch: '\u524d\u6b8b\u308a\u6587\u8108\u306f\u8aad\u307f\u3084\u3059\u3044',
        front_context_neutral: '',
      };
      return map[label] || '';
    }
    function lapPromoteLabelText(label) {
      const map = {
        lap_1win_fast_same_distance_shadow: '\u30e9\u30c3\u30d71\u52dd\u30b7\u30e3\u30c9\u30fc\u5f37',
        lap_promote_strong: '\u30e9\u30c3\u30d7\u9069\u5408\u304b\u3089\u6607\u683c\u5019\u88dc',
        lap_promote_watch: '\u30e9\u30c3\u30d7\u9762\u306f\u6e96\u5019\u88dc',
        lap_role_watch: '\u8ef8\u3068\u76f8\u624b\u306e\u30e9\u30c3\u30d7\u5f79\u5272\u306f\u5408\u3046',
        lap_neutral: '',
      };
      return map[label] || '';
    }
    function queueLapLabelText(label) {
      const map = {
        front_pair_lap_good: '\u524d\u76ee\u30da\u30a2\u5f37\u00d7\u30e9\u30c3\u30d7\u8aad\u307f\u826f\u597d',
        front_pair_strong: '\u524d\u76ee\u30da\u30a2\u5f37',
        lap_good_front_any: '\u968a\u5217\u8aad\u307f\u826f\u597d',
        mixed_queue_watch: '\u968a\u5217\u6df7\u6226\u30ef\u30a4\u30c9\u6ce8\u8996',
        lap_read_weak: '\u30e9\u30c3\u30d7\u8aad\u307f\u5f31\u3081',
        lap_role_goodrun_strong: '\u30e9\u30c3\u30d7\u30ed\u30fc\u30eb\u5f37',
        goodrun_lap_strong: '\u597d\u8d70\u6642\u30e9\u30c3\u30d7\u9069\u6027\u5f37',
        lap_advanced_combo_strong: '\u30e9\u30c3\u30d7\u7dcf\u5408\u5f37',
        lap_advanced_combo_watch: '\u30e9\u30c3\u30d7\u7dcf\u5408\u6ce8\u8996',
        lap_track_positive: '\u99ac\u5834\u00d7\u30e9\u30c3\u30d7\u5f37',
        lap_track_watch: '\u99ac\u5834\u00d7\u30e9\u30c3\u30d7\u6ce8\u8996',
        lap_track_caution: '\u99ac\u5834\u00d7\u30e9\u30c3\u30d7\u53c2\u8003',
        target_ra_lap_fit_strong_both: '\u516c\u5f0fRA\u30e9\u30c3\u30d7\u5f37',
        target_ra_lap_fit_strong: '\u516c\u5f0fRA\u30e9\u30c3\u30d7\u88cf\u4ed8\u3051',
        target_ra_lap_fit_watch: '\u516c\u5f0fRA\u30e9\u30c3\u30d7\u6ce8\u8996',
        target_ra_lap_caution: '\u516c\u5f0fRA\u30e9\u30c3\u30d7\u53c2\u8003',
        horse_lap_high_pressure_instant_pair: '\u524d\u534a\u8ca0\u8377\u00d7\u77ac\u767a\u30da\u30a2',
        horse_lap_pair_fit_good: '\u30e9\u30c3\u30d7\u76f8\u6027\u826f',
        horse_lap_both_match: '\u4e21\u99ac\u30e9\u30c3\u30d7\u4e00\u81f4',
        horse_lap_pair_caution: '\u30e9\u30c3\u30d7\u76f8\u6027\u6ce8\u610f',
        queue_lap_neutral: '',
      };
      return map[label] || '';
    }
    function scoreSuffix(value) {
      const n = Number(value);
      return Number.isFinite(n) && n > 0 ? ` ${n.toFixed(2)}` : '';
    }
    function contextAdjSuffix(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return '';
      const sign = n >= 0 ? '+' : '';
      return ` 補正${sign}${n.toFixed(3)}`;
    }
    function productionContextText(label, adjustment) {
      const n = Number(adjustment);
      const map = {
        prod_context_support: '最強文脈は強く支持',
        prod_context_caution: '最強文脈は警戒',
        prod_s_lap_support: 'ラップ文脈が支持',
        prod_front_survival_support: '前残り文脈が支持',
        prod_closer_context_support: '差し文脈が支持',
        prod_context_neutral: '',
      };
      if (map[label]) return map[label];
      if (Number.isFinite(n)) {
        if (n >= 0.030) return '最強文脈は強く支持';
        if (n >= 0.010) return '最強文脈はやや支持';
        if (n <= -0.030) return '最強文脈は警戒';
        if (n < 0) return '最強文脈はやや警戒';
      }
      return '';
    }
    function ticketExtraPlain(x) {
      const parts = [];
      const prodContext = productionContextText(x.productionContextLabel, x.productionContextAdjustment);
      if (prodContext) {
        parts.push(`${prodContext}${contextAdjSuffix(x.productionContextAdjustment)}`);
      }
      const paceLabel = x.pacePairNote || pacePairLabelText(x.pacePairLabel);
      if (paceLabel && x.pacePairLabel !== 'pace_pair_neutral') {
        parts.push(`${paceLabel}${scoreSuffix(x.pacePairScore)}`);
      }
      const closerLabel = x.closerShadowNote || closerShadowLabelText(x.closerShadowLabel);
      if (closerLabel && x.closerShadowLabel !== 'closer_neutral') {
        parts.push(`${closerLabel}${scoreSuffix(x.closerShadowScore)}`);
      }
      const frontContextLabel = x.frontContextNote || frontContextLabelText(x.frontContextLabel);
      if (frontContextLabel && x.frontContextLabel !== 'front_context_neutral') {
        const score = x.frontContextLabel === 'front_context_survival_watch'
          ? x.frontContextSurvivalScore
          : x.frontContextCollapseScore;
        parts.push(`${frontContextLabel}${scoreSuffix(score)}`);
      }
      const lapPromoteLabel = x.lapPromoteNote || lapPromoteLabelText(x.lapPromoteLabel);
      if (lapPromoteLabel && x.lapPromoteLabel !== 'lap_neutral') {
        const score = x.lapPromoteLabel === 'lap_role_watch' ? x.lapRoleScore : x.lapPromoteScore;
        parts.push(`${lapPromoteLabel}${scoreSuffix(score)}`);
      }
      const queueLapLabel = x.queueLapTitle || queueLapLabelText(x.queueLapLabel);
      if (queueLapLabel && x.queueLapLabel !== 'queue_lap_neutral') {
        const note = x.queueLapNote ? `: ${x.queueLapNote}` : '';
        parts.push(`${queueLapLabel}${scoreSuffix(x.queueLapPriority)}${note}`);
      }
      return parts.join(' / ');
    }
    function ticketExtraHtml(x) {
      const plain = ticketExtraPlain(x);
      return plain ? `<div class="sub">${plain}</div>` : '';
    }
    function ticketReasonSummary(ticketRows, finalTicketRows, decision) {
      if (!ticketRows.length) {
        return '当日運用パイプライン後に、オッズ・馬体重・直前条件を確認して表示します。';
      }
      if (!finalTicketRows.length) {
        const shapeReasons = [...new Set(ticketRows
          .filter(x => x.shadowType === 'shape_umaren')
          .map(x => x.reason)
          .filter(Boolean))]
          .slice(0, 2);
        if (shapeReasons.length) {
          return shapeReasons.join(' / ');
        }
        if (decision.reason) {
          return `${decisionDisplayLabel(decision)}: ${decision.reason}`;
        }
        if (decision.key === 'candidate') {
          return '買い水準に近い候補ですが、最強版の最終BUY条件は未通過です。参考表示で、購入額は0円扱いです。';
        }
        if (decision.key === 'watch') {
          return '注視レースです。AI値や妙味はありますが、安全条件または直前条件が不足しているため購入対象外です。';
        }
        if (decision.key === 'weak') {
          return '参考弱です。比較用には残しますが、買い目としては弱いため購入対象外です。';
        }
        return '見送りです。AI上位馬の確認用として参考組み合わせだけ表示していますが、購入対象外です。';
      }
      const reasons = [...new Set(ticketRows.map(x => x.reason).filter(Boolean))].slice(0, 2);
      const extras = [...new Set(ticketRows.map(ticketExtraPlain).filter(Boolean))].slice(0, 2);
      if (reasons.length && extras.length) return `${reasons.join(' / ')} / ${extras.join(' / ')}`;
      if (!reasons.length && extras.length) return extras.join(' / ');
      return reasons.length ? reasons.join(' / ') : '最強版の購入条件を通過し、現在オッズと最低基準のバランスが残っています。';
    }
    function racePaceSummary(race, hasPace) {
      if (!hasPace) return '展開データが不足しているため暫定評価です。新馬戦や材料不足のレースは直前気配を優先します。';
      const pace = paceLabel(race.expectedPace) || '不明';
      const pressure = Number(race.pressureScore);
      const front = Number(race.frontRunnerCount);
      const queue = Number(race.queueClarityScore);
      const duel = Number(race.frontDuelRiskScore);
      const loadScore = Number(race.frontLoadScore);
      const load = Number.isFinite(pressure) ? metricReadable('先行負荷', pressure, 'pressure') : '先行負荷は未算出';
      const frontText = Number.isFinite(front) ? `逃げ候補${front}頭` : '逃げ候補は未算出';
      const shape = race.paceShapeLabel ? `隊列は${paceShapeLabel(race.paceShapeLabel)}` : '';
      const clarity = Number.isFinite(queue) ? metricReadable('隊列の読みやすさ', queue, 'clarity') : '';
      const duelText = Number.isFinite(duel) ? metricReadable('競り合いリスク', duel, 'duel') : '';
      const loadProjection = Number.isFinite(loadScore) ? metricReadable('前半負荷予測', loadScore, 'load') : '';
      const courseTenParts = [];
      if (race.courseTenRaceLabel) courseTenParts.push(race.courseTenRaceLabel);
      if (Number.isFinite(Number(race.courseTenPressureScore))) courseTenParts.push(metricReadable('コース補正テン負荷', Number(race.courseTenPressureScore), 'pressure'));
      if (Number.isFinite(Number(race.courseTenQueueClarityScore))) courseTenParts.push(metricReadable('コース補正隊列', Number(race.courseTenQueueClarityScore), 'clarity'));
      const courseTenText = courseTenParts.length ? `クラス・コース別テン補正では${courseTenParts.join(' / ')}。` : '';
      const tail = race.expectedPace === 'fast'
        ? '前の消耗と差し込みを両方警戒します。'
        : race.expectedPace === 'slow'
          ? '好位を取れる馬の位置利を重視します。'
          : '極端な偏りより、AI評価とオッズ妙味の両立を重視します。';
      const shapeParts = [shape, clarity, duelText, loadProjection].filter(Boolean).join(' / ');
      return `${pace}想定。${frontText}、${load}。${shapeParts ? `${shapeParts}。` : ''}${courseTenText}${tail}`;
    }
    function raceUpsetSummary(race) {
      if (!race.upsetLabel) return '未算出';
      const value = Number.isFinite(race.upsetScore) ? ` ${pct(race.upsetScore)}` : '';
      const note = race.upsetNote ? `（${race.upsetNote}）` : '';
      return `${race.upsetLabel}${value}${note}`;
    }
    function raceGoingLabel(race) {
      if (!race || !race.trackConditionAvailable) return '';
      const parts = [];
      if (race.turfGoing) parts.push(`芝:${race.turfGoing}`);
      if (race.dirtGoing) parts.push(`ダ:${race.dirtGoing}`);
      return parts.join(' / ');
    }
    function runtimeGoingLabel(race) {
      if (!race || !race.trackConditionAvailable) return '未取得';
      const surface = race.surface || '';
      const going = race.runtimeGoing || '';
      if (going) return `${surface || '当該'} ${going}`;
      return raceGoingLabel(race) || '未取得';
    }
    function trackConditionSourceLabel(race) {
      if (!race || !race.trackConditionAvailable) return '馬場';
      return race.trackConditionSource === 'result' ? '結果馬場' : '現在馬場';
    }
    function raceGoingSummary(race) {
      if (!race || !race.trackConditionAvailable) {
        return '当日馬場は未取得です。更新ボタンでJRA公式馬場を再取得します。';
      }
      const fetched = race.trackFetchedAt ? `（取得 ${race.trackFetchedAt}）` : '';
      return `${trackConditionSourceLabel(race)}: このレースは ${runtimeGoingLabel(race)}。開催場全体は ${raceGoingLabel(race)} ${fetched}`;
    }
    function timeSec(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return '未取得';
      const minutes = Math.floor(n / 60);
      const seconds = n - minutes * 60;
      return minutes > 0 ? `${minutes}:${seconds.toFixed(1).padStart(4, '0')}` : `${seconds.toFixed(1)}秒`;
    }
    function lapSec(value) {
      const n = Number(value);
      return Number.isFinite(n) ? `${n.toFixed(1)}秒` : '未取得';
    }
    function metricNum(value, digits = 1) {
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(digits) : '未取得';
    }
    function historicalConditionLine(race, prefix, label) {
      const n = Number(race[`${prefix}SampleCount`]);
      if (!Number.isFinite(n) || n <= 0) return '';
      const scope = race[`${prefix}Scope`] || '同条件';
      return `${label} ${scope} n=${Math.round(n)}：平均時計 ${timeSec(race[`${prefix}AvgTimeSec`])} / 前3F ${lapSec(race[`${prefix}Front3fSec`])} / 後3F ${lapSec(race[`${prefix}Last3fSec`])} / 1000m推定 ${lapSec(race[`${prefix}Pass1000mSec`])} / RPCI ${metricNum(race[`${prefix}RPCI`], 1)} / PCI3 ${metricNum(race[`${prefix}PCI3`], 1)}`;
    }
    function historicalConditionSummary(race) {
      const lines = [
        historicalConditionLine(race, 'hist5', '過去5年'),
        historicalConditionLine(race, 'hist10', '過去10年'),
      ].filter(Boolean);
      return lines.length ? lines.join('<br>') : '同条件の履歴基準は未取得です。';
    }
    function historicalConditionChip(race) {
      const n = Number(race.hist5SampleCount);
      if (!Number.isFinite(n) || n <= 0) return '';
      return `同条件5年 n=${Math.round(n)} 平均時計 ${timeSec(race.hist5AvgTimeSec)} / 前3F ${lapSec(race.hist5Front3fSec)} / 後3F ${lapSec(race.hist5Last3fSec)}`;
    }
    function lapDeltaText(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return '未算出';
      if (Math.abs(n) < 0.05) return '±0.0秒';
      return `${n > 0 ? '+' : ''}${n.toFixed(1)}秒`;
    }
    function conditionConfidenceLabel(n) {
      const count = Number(n);
      if (!Number.isFinite(count) || count <= 0) return '未取得';
      if (count >= 25) return `高 n=${Math.round(count)}`;
      if (count >= 10) return `中 n=${Math.round(count)}`;
      return `低 n=${Math.round(count)}`;
    }
    function conditionFront3fDeviation(race) {
      const expected = Number(race.expectedFront3fSec);
      const base = Number(race.hist5Front3fSec);
      if (!Number.isFinite(expected) || !Number.isFinite(base)) {
        return { label: '比較未算出', className: '', delta: NaN, comment: '想定前半3Fまたは同条件基準が不足しています。' };
      }
      const delta = expected - base;
      if (delta <= -0.5) {
        return { label: '基準よりかなり速い想定', className: 'fast', delta, comment: '前半負荷が重くなりやすく、前で受ける馬の持続力と差し込み余地を確認します。' };
      }
      if (delta <= -0.2) {
        return { label: '基準より速め', className: 'fast', delta, comment: '同条件平均より締まった流れ寄りです。テンの速さだけでなく消耗耐性を見ます。' };
      }
      if (delta >= 0.5) {
        return { label: '基準よりかなり緩い想定', className: 'slow', delta, comment: '隊列が落ち着きやすく、好位・瞬発力・位置取りの価値が上がります。' };
      }
      if (delta >= 0.2) {
        return { label: '基準より緩め', className: 'slow', delta, comment: '同条件平均より前半は落ち着く想定です。前残りと上がり性能の両面を確認します。' };
      }
      return { label: '同条件標準に近い', className: 'neutral', delta, comment: 'ラップ基準からは極端なズレは小さく、能力・馬場・オッズの総合判断を優先します。' };
    }
    function lapContextCell(label, value) {
      return `<div class="lap-context-cell"><span class="lap-context-label">${label}</span><span class="lap-context-value">${value}</span></div>`;
    }
    function conditionLapPanelHtml(race) {
      const hasHist = Number.isFinite(Number(race.hist5SampleCount)) && Number(race.hist5SampleCount) > 0;
      const hasExpected = Number.isFinite(Number(race.expectedFront3fSec));
      if (!hasHist && !hasExpected) return '';
      const deviation = conditionFront3fDeviation(race);
      const cells = [
        lapContextCell('基準信頼度', conditionConfidenceLabel(race.hist5SampleCount)),
        lapContextCell('今回想定 前半3F', hasExpected ? lapSec(race.expectedFront3fSec) : '未算出'),
        lapContextCell('同条件5年 前半3F', hasHist ? lapSec(race.hist5Front3fSec) : '未取得'),
        lapContextCell('前3F差', lapDeltaText(deviation.delta)),
        lapContextCell('平均勝ち時計', hasHist ? timeSec(race.hist5AvgTimeSec) : '未取得'),
        lapContextCell('後半3F基準', hasHist ? lapSec(race.hist5Last3fSec) : '未取得'),
        lapContextCell('1000m基準', hasHist ? lapSec(race.hist5Pass1000mSec) : '未取得'),
        lapContextCell('RPCI / PCI3', hasHist ? `${metricNum(race.hist5RPCI, 1)} / ${metricNum(race.hist5PCI3, 1)}` : '未取得'),
      ].join('');
      const hist10 = Number.isFinite(Number(race.hist10SampleCount)) && Number(race.hist10SampleCount) > 0
        ? `10年 n=${Math.round(Number(race.hist10SampleCount))} / 前3F ${lapSec(race.hist10Front3fSec)} / RPCI ${metricNum(race.hist10RPCI, 1)}`
        : '10年基準は未取得';
      const note = race.expectedFront3fNote ? ` / ${race.expectedFront3fNote}` : '';
      return `
        <div class="lap-context-head">
          <span class="lap-context-title">同条件ラップ基準</span>
          <span class="lap-context-badge ${deviation.className}">${deviation.label}</span>
        </div>
        <div class="lap-context-grid">${cells}</div>
        <div class="lap-context-comment">${deviation.comment} ${hist10}${note}</div>`;
    }
    function setOptions(select, rows, valueFn, labelFn, keepValue = true) {
      const current = keepValue ? select.value : '';
      select.innerHTML = '';
      rows.forEach(row => {
        const opt = document.createElement('option');
        opt.value = valueFn(row);
        opt.textContent = labelFn(row);
        select.appendChild(opt);
      });
      if (current && [...select.options].some(o => o.value === current)) select.value = current;
    }
    function renderVenueButtons() {
      if (!venueButtons) return;
      const current = venueSelect.value;
      const rows = [...venueSelect.options].map(opt => ({ value: opt.value, label: opt.textContent || '' }));
      venueButtons.innerHTML = '';
      rows.forEach(row => {
        const nextRace = nextRaceForVenue(dateSelect.value, row.value);
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `filter-button venue-button ${row.value === current ? 'active' : ''}`;
        const name = document.createElement('span');
        name.className = 'venue-name';
        name.textContent = row.label;
        button.appendChild(name);
        if (nextRace) {
          const next = document.createElement('span');
          next.className = 'venue-next';
          next.textContent = `次 ${nextRace.raceNo}R ${nextRace.startTime || ''}`.trim();
          button.appendChild(next);
        }
        button.addEventListener('click', () => {
          venueSelect.value = row.value;
          updateDecisionOptions(false);
          updateRaceOptions(false);
          render();
        });
        venueButtons.appendChild(button);
      });
    }
    function renderRaceButtons() {
      if (!raceButtons) return;
      const current = raceSelect.value;
      const rows = filteredRaces()
        .sort(raceSortCompare);
      raceButtons.innerHTML = '';
      rows.forEach(race => {
        const label = raceShortLabel(race);
        const decision = decisionForRace(race.raceId);
        const started = raceHasStarted(race);
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `filter-button race-button-pill ${race.raceId === current ? 'active' : ''} ${decision.className || buttonClassForDecision(race.raceId)} ${started ? 'post-finished' : ''}`;
        const top = document.createElement('span');
        top.className = 'race-top';
        const no = document.createElement('span');
        no.className = 'race-no';
        no.textContent = label.no || race.raceLabel || race.raceId;
        const decisionTag = document.createElement('span');
        decisionTag.className = 'decision-tag';
        decisionTag.textContent = compactDecisionLabel(decision);
        decisionTag.title = decision.label || '未判定';
        const name = document.createElement('span');
        name.className = 'race-name';
        name.textContent = label.name || race.surface || '';
        const statusRow = document.createElement('span');
        statusRow.className = 'race-status-row';
        const time = document.createElement('span');
        time.className = 'race-time';
        time.textContent = race.startTime ? `${race.startTime} 発走` : '';
        const postTag = document.createElement('span');
        postTag.className = 'post-tag';
        postTag.textContent = '出走済';
        top.appendChild(no);
        top.appendChild(decisionTag);
        button.appendChild(top);
        button.appendChild(name);
        if (time.textContent) statusRow.appendChild(time);
        if (started) statusRow.appendChild(postTag);
        if (statusRow.childNodes.length) button.appendChild(statusRow);
        button.title = `${race.venue || ''}${label.no || ''} ${label.name || ''} / ${decision.label || ''}${started ? ' / 出走済' : ''}`.trim();
        button.addEventListener('click', () => {
          raceSelect.value = race.raceId;
          render();
        });
        raceButtons.appendChild(button);
      });
      if (!rows.length) {
        const blank = document.createElement('span');
        blank.className = 'pill';
        blank.textContent = '該当レースなし';
        raceButtons.appendChild(blank);
      }
    }
    function renderFilterButtons() {
      renderViewModeButtons();
      renderVenueButtons();
      renderRaceButtons();
    }
    function initStatus() {
      const c = payload.counts || {};
      document.getElementById('status').innerHTML = `
        <span class="pill ok">${payload.source || 'ライブ'} 接続</span>
        <span class="pill">${c.races || 0}R</span>
        <span class="pill">単複 ${c.singleRows || 0}頭</span>
        <span class="pill">馬連/ワイド ${c.pairRows || 0}点</span>
        <span class="pill">買い目 ${c.ticketRows || 0}点</span>
        <span class="pill">V2参考（非購入） ${c.v2ShadowTicketRaces || 0}R</span>
        <span class="pill">展開補正シャドー ${c.shapeShadowTicketRaces || 0}R</span>
        <span class="pill">補正テン文脈 ${c.courseTenContextRaces || 0}R</span>
        <span class="pill">同条件基準 ${c.historicalConditionRaces || 0}R</span>
        <span class="pill">生成 ${payload.generatedAt || ''}</span>`;
    }
    function updateDecisionOptions(keep = true) {
      const current = keep ? decisionFilter.value : '';
      const d = dateSelect.value;
      const venue = venueSelect.value;
      const base = races.filter(r => (!d || r.dateKey === d) && (!venue || r.venueCode === venue));
      const counts = { buy: 0, candidate: 0, closer_watch: 0, watch: 0, weak: 0, wait: 0, skip: 0, finished: 0 };
      base.forEach(r => {
        const key = decisionForRace(r.raceId).key;
        counts[key] = (counts[key] || 0) + 1;
        if (raceHasStarted(r)) counts.finished = (counts.finished || 0) + 1;
      });
      const opts = [
        { value: '', label: `全レース (${base.length})` },
        { value: 'buy', label: `${raceFilterLabel('buy')} (${counts.buy})` },
        { value: 'candidate', label: `${raceFilterLabel('candidate')} (${counts.candidate})` },
        { value: 'closer_watch', label: `${raceFilterLabel('closer_watch')} (${counts.closer_watch})` },
        { value: 'watch', label: `${raceFilterLabel('watch')} (${counts.watch})` },
        { value: 'weak', label: `${raceFilterLabel('weak')} (${counts.weak})` },
        { value: 'wait', label: `${raceFilterLabel('wait')} (${counts.wait})` },
        { value: 'skip', label: `${raceFilterLabel('skip')} (${counts.skip})` },
        { value: 'finished', label: `${raceFilterLabel('finished')} (${counts.finished})` },
      ];
      setOptions(decisionFilter, opts, x => x.value, x => x.label, keep);
      if (current && [...decisionFilter.options].some(o => o.value === current)) decisionFilter.value = current;
    }
    function renderViewModeButtons() {
      if (!viewModeButtons) return;
      [...viewModeButtons.querySelectorAll('[data-view-mode]')].forEach(button => {
        button.classList.toggle('active', button.dataset.viewMode === currentViewMode);
      });
    }
    function updateRefreshLabels() {
      if (refreshRaceButton) {
        refreshRaceButton.textContent = currentViewMode === 'win5' ? 'WIN5更新' : 'このレース更新';
      }
    }
    function setRefreshDisabled(disabled) {
      if (refreshRaceButton) refreshRaceButton.disabled = disabled;
      if (refreshAllButton) refreshAllButton.disabled = disabled;
    }
    function applyViewMode() {
      const win5Section = document.getElementById('win5Section');
      const raceHead = document.getElementById('raceHead');
      const ticketSection = document.getElementById('ticketSection');
      const raceGrid = document.getElementById('raceGrid');
      const showWin5 = currentViewMode === 'win5';
      if (win5Section) win5Section.style.display = showWin5 ? '' : 'none';
      if (raceHead) raceHead.style.display = showWin5 ? 'none' : '';
      if (ticketSection) ticketSection.style.display = showWin5 ? 'none' : '';
      if (raceGrid) raceGrid.style.display = showWin5 ? 'none' : '';
      renderViewModeButtons();
      updateRefreshLabels();
    }
    function initFilters() {
      const dateKeys = uniq(races.map(r => r.dateKey)).sort();
      setOptions(dateSelect, dateKeys, x => x, x => {
        const r = races.find(v => v.dateKey === x);
        return r ? r.dateLabel : x;
      }, false);
      const preferredDate = payload.defaultDate || '';
      if (dateKeys.includes(preferredDate)) {
        dateSelect.value = preferredDate;
      } else if (dateKeys.length) {
        dateSelect.value = dateKeys[dateKeys.length - 1];
      }
      updateVenueOptions(false);
      updateDecisionOptions(false);
      updateRaceOptions(false);
      renderFilterButtons();
    }
    function updateVenueOptions(keep = true) {
      const d = dateSelect.value;
      const venues = races.filter(r => r.dateKey === d)
        .sort((a, b) => String(a.venueCode).localeCompare(String(b.venueCode)))
        .map(r => ({ code: r.venueCode, venue: r.venue }));
      const uniqueVenues = [];
      const seen = new Set();
      venues.forEach(v => { if (!seen.has(v.code)) { seen.add(v.code); uniqueVenues.push(v); } });
      setOptions(venueSelect, [{ code: '', venue: `全場 (${uniqueVenues.length})` }, ...uniqueVenues], v => v.code, v => v.venue, keep);
      [...venueSelect.options].filter(o => !o.value).forEach(o => o.remove());
      const selectedVenueValid = [...venueSelect.options].some(o => o.value === venueSelect.value);
      if (!keep || !selectedVenueValid || !venueSelect.value) {
        const nextRace = nextRaceForVenue(d, '');
        const fallback = nextRace ? nextRace.venueCode : (uniqueVenues[0] ? uniqueVenues[0].code : '');
        if (fallback) venueSelect.value = fallback;
      }
    }
    function updateRaceOptions(keep = true) {
      const raceRows = filteredRaces()
        .sort(raceSortCompare);
      setOptions(raceSelect, raceRows, r => r.raceId, r => `${r.venue}${r.raceNo}R ${r.raceName || ''}`.trim(), keep);
    }
    function render() {
      const raceId = raceSelect.value;
      renderFilterButtons();
      const race = byId.get(raceId) || {};
      const decision = decisionForRace(raceId);
      const hasPace = paceReady(race);
      const singleRows = singles.filter(x => x.raceId === raceId).sort((a, b) => {
        const ar = Number(a.aiRank || 999);
        const br = Number(b.aiRank || 999);
        return ar - br || a.horseNo - b.horseNo;
      });
      const type = ticketType.value;
      const pairRows = pairs
        .filter(x => x.raceId === raceId && x.ticketType === type)
        .sort((a, b) => {
          const ao = Number.isFinite(a.odds) ? a.odds : 999999;
          const bo = Number.isFinite(b.odds) ? b.odds : 999999;
          return ao - bo || a.aNo - b.aNo || a.bNo - b.bNo;
        });

      document.getElementById('raceTitle').textContent = race.raceLabel ? `${race.raceLabel} ${race.raceName || ''}` : 'レースを選択';
      document.getElementById('snapshotPill').textContent = `最新 ${payload.latestSnapshotLabel || '-'}`;
      const decisionPill = document.getElementById('decisionPill');
      decisionPill.textContent = decisionDisplayLabel(decision);
      decisionPill.className = `pill decision-pill ${decision.className}`;
      document.getElementById('raceMeta').innerHTML = [
        Number.isFinite(decision.priorityScore) ? metricChip('優先度', decision.priorityScore, 'priority') : '',
        Number.isFinite(decision.topScore) ? metricChip('候補強度', decision.topScore, 'strength') : '',
        aiScoreGapLabel(race),
        Number.isFinite(decision.margin) ? metricChip('妙味', decision.margin, 'edge') : '',
        decision.reason ? `判断 ${decision.reason}` : '',
        decision.topPair && decision.topPair.aNo && decision.topPair.bNo ? `候補 ${decision.topPair.aNo}-${decision.topPair.bNo}${Number.isFinite(decision.topPair.odds) ? ` / ${odds(decision.topPair.odds)}` : ''}` : '',
        race.startTime ? `発走 ${race.startTime}` : '',
        race.surface && race.distance ? `${race.surface}${race.distance}m` : '',
        race.trackConditionAvailable ? `${trackConditionSourceLabel(race)} ${raceGoingLabel(race)}` : '馬場 未取得',
        race.trackConditionAvailable ? `このレース ${runtimeGoingLabel(race)}` : '',
        race.fieldSize ? `${race.fieldSize}頭` : '',
        race.expectedPace && hasPace ? `想定 ${paceLabel(race.expectedPace)}` : '',
        hasPace && Number.isFinite(race.pressureScore) ? metricChip('先行負荷', race.pressureScore, 'pressure') : '',
        hasPace && Number.isFinite(race.frontRunnerCount) ? `逃げ候補 ${race.frontRunnerCount}` : '',
        race.paceShapeLabel ? `隊列 ${paceShapeLabel(race.paceShapeLabel)}` : '',
        Number.isFinite(race.queueClarityScore) ? metricChip('隊列', race.queueClarityScore, 'clarity') : '',
        Number.isFinite(race.frontDuelRiskScore) ? metricChip('競り合い', race.frontDuelRiskScore, 'duel') : '',
        race.courseTenRaceLabel ? race.courseTenRaceLabel : '',
        Number.isFinite(race.expectedFront3fSec) ? `\u60f3\u5b9a\u524d\u534a3F ${race.expectedFront3fSec.toFixed(1)}\u79d2${race.expectedFront3fNote ? `\uff08${race.expectedFront3fNote}\uff09` : ''}` : '',
        historicalConditionChip(race),
        Number.isFinite(race.courseTenPressureScore) ? metricChip('コース補正テン', race.courseTenPressureScore, 'pressure') : '',
        Number.isFinite(race.courseTenQueueClarityScore) ? metricChip('コース補正隊列', race.courseTenQueueClarityScore, 'clarity') : '',
        race.upsetLabel ? `荒れ予想 ${raceUpsetSummary(race)}` : '',
        !hasPace ? '展開データ不足' : '',
        Number.isFinite(race.confidence) ? metricChip('AI信頼', race.confidence, 'confidence') : '',
        race.kaiji && race.nichiji ? `${Number(race.kaiji)}回${race.venue || ''}${Number(race.nichiji)}日` : '',
      ].filter(Boolean).map(x => `<span class="pill">${x}</span>`).join('');
      const lapContextPanel = document.getElementById('lapContextPanel');
      if (lapContextPanel) {
        const lapContextHtml = conditionLapPanelHtml(race);
        lapContextPanel.innerHTML = lapContextHtml;
        lapContextPanel.classList.toggle('empty', !lapContextHtml);
      }

      const raceStarted = raceHasStarted(race);
      const rawFinalTicketRows = (ticketsByRace.get(raceId) || []).sort((a, b) => Number(b.stakeYen || 0) - Number(a.stakeYen || 0));
      const finalTicketRows = rawFinalTicketRows.filter(x => {
        if (x.purchaseValid === false) return false;
        return raceStarted ? x.generatedBeforePost === true : true;
      });
      const ticketRows = finalTicketRows.length ? finalTicketRows : referenceTicketsForRace(raceId, decision);
      const totalStake = ticketRows.reduce((sum, x) => sum + (Number(x.stakeYen) || 0), 0);
      const hasFinalTickets = finalTicketRows.length > 0;
      const hasShadowReference = !hasFinalTickets && ticketRows.some(x => x.shadowType === 'v2_wide');
      const hasShapeReference = !hasFinalTickets && ticketRows.some(x => x.shadowType === 'shape_umaren');
      const ticketSection = document.getElementById('ticketSection');
      if (ticketSection) ticketSection.classList.toggle('reference', !hasFinalTickets);
      document.getElementById('ticketTitle').textContent = hasFinalTickets
        ? '最終買い目'
        : hasShapeReference
          ? '展開補正シャドー買い目（非購入）'
          : hasShadowReference
          ? 'V2参考買い目（購入対象外）'
          : '参考買い目（購入対象外）';
      document.getElementById('ticketCount').textContent = ticketRows.length
        ? (hasFinalTickets
          ? `最終 ${ticketRows.length}点 / ${Math.round(totalStake).toLocaleString()}円`
          : hasShapeReference
            ? `展開補正 ${ticketRows.length}点 / 購入額0円`
            : hasShadowReference
            ? `V2参考 ${ticketRows.length}点 / 購入額0円`
            : `参考 ${ticketRows.length}点 / 購入額0円`)
        : '未生成';
      document.getElementById('ticketNote').innerHTML = `
        <div><strong>選択理由:</strong> ${ticketReasonSummary(ticketRows, finalTicketRows, decision)}</div>
        <div><strong>馬場状態:</strong> ${raceGoingSummary(race)}</div>
        <div><strong>展開予想:</strong> ${racePaceSummary(race, hasPace)}</div>
        <div><strong>荒れ予想:</strong> ${raceUpsetSummary(race)}</div>`;
      document.getElementById('ticketBody').innerHTML = ticketRows.length ? ticketRows.map(x => {
        const combo = x.bNo ? `${horseLabel(x.aNo, x.aName)} - ${horseLabel(x.bNo, x.bName)}` : horseLabel(x.aNo, x.aName);
        const livePaySub = Number.isFinite(x.livePay) ? `<div class="sub">${yen(x.livePay)}</div>` : '';
        return `<tr>
          <td>${pairTypeLabel(x.ticketType || x.ticketLabel)}<div class="sub">${x.action || ''}</div></td>
          <td class="horse">${combo}</td>
          <td class="num">${x.reference ? '非購入' : `${Math.round(Number(x.stakeYen) || 0).toLocaleString()}円`}</td>
          <td class="num odds ${clsOdds(x.liveOdds)}">${odds(x.liveOdds)}${livePaySub}</td>
          <td class="num hide-sm">${odds(x.minOdds)}</td>
          <td class="hide-sm"><div class="chips ${x.reference ? 'bad' : 'good'}"><span>${x.reason || '最終買い目'}</span></div>${ticketExtraHtml(x)}</td>
        </tr>`;
      }).join('') : `<tr><td colspan="6" class="blank">このレースの最終買い目はまだ未生成です。当日運用パイプライン後、ここに券種・組み合わせ・購入額・現在オッズが表示されます。</td></tr>`;

      document.getElementById('singleCount').textContent = `${singleRows.length}頭`;
      document.getElementById('singleBody').innerHTML = singleRows.length ? singleRows.map(x => {
        const place = Number.isFinite(x.placeMin) && Number.isFinite(x.placeMax) ? `${x.placeMin.toFixed(1)}-${x.placeMax.toFixed(1)}` : '';
        const preday = Number.isFinite(x.predayWinOdds) ? `<div class="sub">前日 ${odds(x.predayWinOdds)}</div>` : '';
        const popularity = x.popularity ? ` / 現在${x.popularity}人気` : '';
        const carriedWeight = x.carriedWeightText ? ` / 斤 ${x.carriedWeightText}` : '';
        const bodyWeight = x.bodyWeightText ? ` / 馬体 ${x.bodyWeightText}` : '';
        const courseTen = Number(x.courseTenHistoryAvailable || 0) >= 1 && Number.isFinite(x.courseTenSpeed)
          ? ` / ${courseTenSpeedLabel(x.courseTenSpeed)}`
          : '';
        const points = (x.points || []).map(v => `<span>${v}</span>`).join('');
        const concerns = (x.concerns || []).map(v => `<span>${v}</span>`).join('');
        return `<tr>
          <td class="num rank">${x.aiRank || '-'}<div class="sub">${aiScoreLabel(x.aiScore)}</div></td>
          <td class="frame-cell">${frameBadge(x.frameNo, x.horseNo)}</td>
          <td class="horse">${x.horseName || ''}<div class="sub">${x.jockey || ''}${popularity}${carriedWeight}${bodyWeight}${courseTen}${x.expectedPace && hasPace ? ` / ${paceLabel(x.expectedPace)}` : ''}${hasPace && Number.isFinite(x.frontRunning) ? ` / ${runningTendencyLabel('前目傾向', x.frontRunning)}` : ''}${hasPace && Number.isFinite(x.closing) ? ` / ${runningTendencyLabel('差し傾向', x.closing)}` : ''}${!hasPace ? ' / 展開データ不足' : ''}</div></td>
          <td class="num odds ${clsOdds(x.winOdds)}">${odds(x.winOdds)}${preday}</td>
          <td class="num">${place}</td>
          <td class="num hide-sm">${x.popularity || ''}</td>
          <td class="num hide-sm">${x.carriedWeightText || ''}</td>
          <td class="num hide-sm">${x.bodyWeightText || ''}</td>
          <td><div class="chips good">${points}</div></td>
          <td><div class="chips bad">${concerns}</div></td>
        </tr>`;
      }).join('') : `<tr><td colspan="10" class="blank">単複オッズがありません</td></tr>`;

      document.getElementById('pairTitle').textContent = pairTypeLabel(type);
      document.getElementById('pairCount').textContent = `${pairRows.length}点`;
      document.getElementById('pairBody').innerHTML = pairRows.length ? pairRows.map(x => {
        return `<tr>
          <td class="horse">${horseLabel(x.aNo, x.aName)} - ${horseLabel(x.bNo, x.bName)}</td>
          <td class="num odds ${clsOdds(x.odds)}">${odds(x.odds)}</td>
          <td class="num hide-sm">${yen(x.payPer100)}</td>
          <td class="num hide-sm">${x.popularity || ''}</td>
        </tr>`;
      }).join('') : `<tr><td colspan="4" class="blank">${pairTypeLabel(type)}オッズがありません</td></tr>`;
      applyViewMode();
    }
    dateSelect.addEventListener('change', () => { updateVenueOptions(false); updateDecisionOptions(false); updateRaceOptions(false); render(); });
    venueSelect.addEventListener('change', () => { updateDecisionOptions(false); updateRaceOptions(false); render(); });
    decisionFilter.addEventListener('change', () => { updateRaceOptions(false); render(); });
    raceSelect.addEventListener('change', render);
    ticketType.addEventListener('change', render);
    if (viewModeButtons) {
      viewModeButtons.addEventListener('click', event => {
        const button = event.target.closest('[data-view-mode]');
        if (!button) return;
        currentViewMode = button.dataset.viewMode || 'race';
        applyViewMode();
      });
    }
    if (win5BudgetButtons) {
      win5BudgetButtons.addEventListener('click', event => {
        const button = event.target.closest('[data-win5-budget]');
        if (!button) return;
        currentWin5Budget = Number(button.dataset.win5Budget || currentWin5Budget);
        renderWin5();
      });
    }
    async function refreshLiveData(scope = 'race') {
      if ((!refreshRaceButton && !refreshAllButton) || !refreshStatus) return;
      if (window.location.protocol === 'file:') {
        refreshStatus.innerHTML = 'HTTP版で開くと更新できます: <a href="http://127.0.0.1:8766/outputs/ui/live_odds_dashboard.html">127.0.0.1:8766</a>';
        return;
      }
      const isFull = scope === 'all';
      setRefreshDisabled(true);
      refreshStatus.textContent = isFull
        ? '全レース取得中...'
        : (currentViewMode === 'win5' ? 'WIN5対象R取得中...' : 'このレース取得中...');
      try {
        const refreshUrl = new URL('/api/refresh', window.location.origin);
        refreshUrl.searchParams.set('mode', isFull ? 'full' : 'quick');
        if (dateSelect && dateSelect.value) refreshUrl.searchParams.set('date', dateSelect.value);
        if (!isFull && currentViewMode === 'win5') {
          const win5RaceIds = (win5.legs || []).map(leg => String(leg.raceId || '')).filter(v => /^\\d{16}$/.test(v));
          if (!win5RaceIds.length) {
            throw new Error('WIN5対象レースが未生成です');
          }
          refreshUrl.searchParams.set('race_ids', win5RaceIds.join(','));
        } else if (!isFull && raceSelect && raceSelect.value) {
          refreshUrl.searchParams.set('race_id', raceSelect.value);
        }
        const response = await fetch(refreshUrl.toString(), {
          method: 'POST',
          credentials: 'same-origin',
          cache: 'no-store',
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok || result.ok === false) {
          throw new Error(result.error || `HTTP ${response.status}`);
        }
        refreshStatus.textContent = '更新完了';
        const nextUrl = new URL(window.location.href);
        nextUrl.searchParams.set('v', String(Date.now()));
        window.location.replace(nextUrl.toString());
      } catch (error) {
        refreshStatus.textContent = `更新失敗: ${error.message || error}`;
        setRefreshDisabled(false);
      }
    }
    if (refreshRaceButton) {
      refreshRaceButton.addEventListener('click', () => refreshLiveData('race'));
      if (window.location.protocol === 'file:' && refreshStatus) {
        refreshStatus.innerHTML = 'HTTP版のみ: <a href="http://127.0.0.1:8766/outputs/ui/live_odds_dashboard.html">開く</a>';
      }
    }
    if (refreshAllButton) {
      refreshAllButton.addEventListener('click', () => refreshLiveData('all'));
      if (window.location.protocol === 'file:' && refreshStatus) {
        refreshStatus.innerHTML = 'HTTP版のみ: <a href="http://127.0.0.1:8766/outputs/ui/live_odds_dashboard.html">開く</a>';
      }
    }
    initStatus();
    initFilters();
    renderWin5();
    render();
    setInterval(renderFilterButtons, 60000);
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local dashboard for current JRA live odds.")
    parser.add_argument("--single-odds-csv", default="data/processed/live_odds/realtime_single_odds_latest.csv")
    parser.add_argument("--pair-odds-csv", default="data/processed/live_odds/realtime_pair_odds_latest.csv")
    parser.add_argument(
        "--entry-csv",
        default="",
        help="Optional entry snapshot CSV. Defaults to latest weekly enriched odds snapshot.",
    )
    parser.add_argument(
        "--prediction-csv",
        default="",
        help="Optional prediction CSV with ai_rank/ai_score/expected_pace. Defaults to latest preday enriched odds prediction.",
    )
    parser.add_argument(
        "--tickets-csv",
        default="",
        help="Optional final/selected ticket CSV. Defaults to latest race-day runtime selected tickets when available.",
    )
    parser.add_argument(
        "--candidates-csv",
        default="",
        help="Optional all-candidates CSV for non-buy race priority labels.",
    )
    parser.add_argument(
        "--wide-shadow-csv",
        default="outputs/analysis/current_strongest_runtime_v1/pair_joint_v2_runtime_guard/wide_shadow_guard_ok_candidates.csv",
        help="Optional V2 wide shadow candidate CSV. Displayed as non-purchase reference tickets.",
    )
    parser.add_argument(
        "--course-ten-context-csv",
        default="outputs/analysis/current_strongest_runtime_v1/current_course_adjusted_front3f_context.csv",
        help="Optional class/course-adjusted estimated front3F context CSV for dashboard notes.",
    )
    parser.add_argument(
        "--historical-condition-context-csv",
        default="outputs/analysis/historical_condition_lap_context_v1/condition_lap_baselines.csv",
        help="Optional historical same-condition time/lap baseline CSV for dashboard display.",
    )
    parser.add_argument(
        "--horse-lap-decomp-csv",
        default="outputs/analysis/horse_lap_aptitude_decomposition_v1/all_ticket_lap_decomposition_compact.csv",
        help="Optional horse lap aptitude decomposition ticket overlay CSV for explanatory labels.",
    )
    parser.add_argument("--body-weight-csv", default="data/processed/live_body_weight/body_weight_latest.csv")
    parser.add_argument("--track-condition-csv", default="data/processed/live_track_conditions/current_track_conditions.csv")
    parser.add_argument("--result-track-csv", default="outputs/analysis/current_live_pnl/current_result_track_conditions.csv")
    parser.add_argument("--gelding-history-csv", default="data/processed/gelding_transition/gelding_transition_history.csv")
    parser.add_argument("--win5-json", default="outputs/analysis/win5_runtime/win5_plan.json")
    parser.add_argument("--default-date", default="", help="YYYYMMDD date selected when the dashboard opens.")
    parser.add_argument("--output-html", default="outputs/ui/live_odds_dashboard.html")
    args = parser.parse_args()

    single_csv = project_path(args.single_odds_csv)
    pair_csv = project_path(args.pair_odds_csv)
    entry_csv = project_path(args.entry_csv) if args.entry_csv else latest_file(
        "data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched_workout_knowledge.csv"
    ) or latest_file(
        "data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched_workout.csv"
    ) or latest_file(
        "data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched.csv"
    ) or latest_file(
        "data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay.csv"
    ) or latest_file(
        "data/datasets/inference/weekly/entry_snapshot_*_enriched_odds_workout_knowledge.csv"
    ) or latest_file(
        "data/datasets/inference/weekly/entry_snapshot_*_enriched_odds.csv"
    )
    if entry_csv is not None and not entry_csv.exists():
        entry_csv = latest_file("data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched_workout_knowledge.csv") or latest_file(
            "data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched_workout.csv"
        ) or latest_file(
            "data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched.csv"
        ) or latest_file(
            "data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay.csv"
        ) or latest_file(
            "data/datasets/inference/weekly/entry_snapshot_*_enriched_odds_workout_knowledge.csv"
        ) or latest_file(
            "data/datasets/inference/weekly/entry_snapshot_*_enriched_odds.csv"
        )
    prediction_csv = project_path(args.prediction_csv) if args.prediction_csv else (
        latest_file("outputs/predictions/preday_target_de_overlay_*/baseline_predictions_*.csv")
        or latest_file("outputs/predictions/preday_strongest_feature_parity/baseline_predictions_*.csv")
        or latest_file("outputs/predictions/preday_netkeiba_enriched_odds_history_context/baseline_predictions_*.csv")
        or latest_file("outputs/predictions/preday_netkeiba_enriched_odds/baseline_predictions_*.csv")
    )
    tickets_csv = project_path(args.tickets_csv) if args.tickets_csv else resolve_default_tickets_csv()
    candidates_csv = project_path(args.candidates_csv) if args.candidates_csv else resolve_default_candidates_csv()
    wide_shadow_csv = project_path(args.wide_shadow_csv) if args.wide_shadow_csv else None
    course_ten_context_csv = project_path(args.course_ten_context_csv) if args.course_ten_context_csv else None
    historical_condition_context_csv = (
        project_path(args.historical_condition_context_csv) if args.historical_condition_context_csv else None
    )
    horse_lap_decomp_csv = project_path(args.horse_lap_decomp_csv) if args.horse_lap_decomp_csv else None
    body_weight_csv = project_path(args.body_weight_csv) if args.body_weight_csv else None
    track_condition_csv = project_path(args.track_condition_csv) if args.track_condition_csv else None
    result_track_csv = project_path(args.result_track_csv) if args.result_track_csv else None
    gelding_history_csv = project_path(args.gelding_history_csv) if args.gelding_history_csv else None
    win5_json = project_path(args.win5_json) if args.win5_json else None
    default_date_key = parse_date_key(args.default_date) or read_dashboard_default_date()

    payload = build_payload(
        single_csv,
        pair_csv,
        entry_csv,
        prediction_csv,
        tickets_csv,
        candidates_csv,
        wide_shadow_csv,
        course_ten_context_csv,
        historical_condition_context_csv,
        horse_lap_decomp_csv,
        body_weight_csv,
        track_condition_csv,
        result_track_csv,
        gelding_history_csv,
        default_date_key,
    )
    payload["win5"] = read_json_safe(win5_json)
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    output_path = project_path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "output_html": str(output_path),
                "single_odds_csv": str(single_csv),
                "pair_odds_csv": str(pair_csv),
                "body_weight_csv": str(body_weight_csv) if body_weight_csv else "",
                "entry_csv": str(entry_csv) if entry_csv else "",
                "prediction_csv": str(prediction_csv) if prediction_csv else "",
                "tickets_csv": str(tickets_csv) if tickets_csv else "",
                "candidates_csv": str(candidates_csv) if candidates_csv else "",
                "wide_shadow_csv": str(wide_shadow_csv) if wide_shadow_csv else "",
                "course_ten_context_csv": str(course_ten_context_csv) if course_ten_context_csv else "",
                "historical_condition_context_csv": str(historical_condition_context_csv)
                if historical_condition_context_csv
                else "",
                "horse_lap_decomp_csv": str(horse_lap_decomp_csv) if horse_lap_decomp_csv else "",
                "track_condition_csv": str(track_condition_csv) if track_condition_csv else "",
                "result_track_csv": str(result_track_csv) if result_track_csv else "",
                "win5_json": str(win5_json) if win5_json else "",
                "default_date": payload.get("defaultDate", ""),
                "generated_at": payload["generatedAt"],
                "latest_snapshot": payload["latestSnapshotLabel"],
                "counts": payload["counts"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_html": str(output_path), **payload["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
