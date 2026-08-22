#!/usr/bin/env python3
"""Build the Race Intelligence Lite+ Sunday observation view.

This module is deliberately independent from prediction, market, purchase and
notification code.  It consumes only explicitly named, read-only audit files
and a sanitized JRA current-entry snapshot.  All horse ordering is the official
horse-number order; nothing in this file produces a model score or a race rank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_module
import json
import math
import re
import statistics
import urllib.request
import urllib.parse
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


JST = ZoneInfo("Asia/Tokyo")
TARGET_DATE = date(2026, 8, 23)
STATUS_VALUES = {"observed", "derived", "proxy", "unobserved"}
ROUTE_MATCH_VALUES = {"exact", "partial", "similar", "unclassified", "unobserved"}
CONFIDENCE_VALUES = {"high", "medium", "low", "unobserved"}
ROLE_VALUES = ("lead", "front", "stalking", "midpack", "rear", "unobserved")
SCENARIO_VALUES = ("SLOW", "MIDDLE", "FAST")
SENSITIVITY_VALUES = {"supportive", "neutral", "adverse", "unobserved"}

SAFETY = {
    "observation_only": True,
    "read_only_view": True,
    "formal_prediction": False,
    "ranking_claims": False,
    "probability_claims": False,
    "market_data_used": False,
    "trading_hooks": False,
    "formal_buy": False,
    "send_order": False,
    "stake": 0,
    "training_executed": False,
    "outer_oos_executed": False,
    "exp_033_run_executed": False,
    "exp_034_real_data_execution": False,
}

# Fail closed on columns/keys associated with quarantined prediction or market
# paths.  Historical finish_position is explicitly allowed observation data.
FORBIDDEN_KEY_PATTERNS = (
    re.compile(r"(^|_)ai_(score|rank)($|_)"),
    re.compile(r"(^|_)(slow|middle|fast)_(ai_)?(score|rank)($|_)"),
    re.compile(r"(^|_)(win|place|top3)_prob(ability)?($|_)"),
    re.compile(r"(^|_)(odds|popularity|market|payoff|roi|ev|mispricing)($|_)"),
    re.compile(r"(^|_)(buy|ticket|purchase|champion|notification|order)_"),
    re.compile(r"^(buy|ticket|purchase|champion|notification|order)$"),
)

# Negative safety attestations in audited sources are allowed only so their
# false value can be checked; they never become horse or race evidence.
NEGATIVE_ATTESTATION_KEYS = {
    "historical_market_columns_included",
    "ability_proxy_added_to_ai_score",
    "market_fields_extracted",
    "market_data_used",
    "display_order_rule",
}

ENTRY_ALLOWED_COLUMNS = {
    "target_date", "manifest_leg", "win5_role", "race_id", "race_name", "venue",
    "race_no", "horse_id", "horse_name", "jockey", "carried_weight_kg", "trainer",
    "trainer_affiliation", "frame_number", "horse_number", "draw_status", "entry_stage",
    "current_entry_status", "source_row_order", "identity_join_key", "identity_join_method",
    "horse_name_join_used", "official_entry_url", "official_snapshot_path",
    "official_snapshot_sha256", "official_snapshot_asof_jst", "formal_buy", "send_order",
    "stake",
}

HISTORY_ALLOWED_COLUMNS = {
    "target_date", "target_leg", "target_role", "target_race_id", "target_horse_id",
    "target_horse_name", "draw_status", "entry_stage", "history_row_type",
    "history_run_present", "history_race_id", "race_date", "venue_code", "venue",
    "surface", "distance_m", "jv_track_code", "track_code", "inner_outer",
    "course_setting", "competition_domain", "finish_position", "corner1", "corner2",
    "corner3", "corner4", "final_3f_sec", "finish_gap_sec", "ra_joined",
    "ra_source_file", "ra_source_line_no", "route_base_key", "route_id",
    "route_match_level", "route_context_complete", "missing_components", "pre_target_flag",
    "future_or_post_target_flag", "history_scope_start", "target_asof_cutoff",
    "source_data_asof", "source_file", "source_row_key", "identity_join_key",
    "identity_join_method", "horse_name_join_used", "historical_market_columns_included",
    "draw_dependent_features_used", "formal_buy", "send_order", "stake",
}

TARGET_ALLOWED_COLUMNS = set("""target_date,manifest_leg,win5_role,manifest_race_name,manifest_aliases_pipe,selection_mode,resolution_match_count,resolution_status,race_id,race_name,venue,race_no,surface_manifest,distance_m,post_time_jst,declared_runner_count,active_runner_count,confirmed_draw_count,scheduled_pending_draw_count,scratched_count,race_draw_status,race_entry_stage,runner_identity_key,horse_name_join_used,official_entry_url,official_snapshot_path,official_snapshot_sha256,official_snapshot_asof_jst,manifest_path,manifest_sha256,formal_buy,send_order,stake""".split(","))

COVERAGE_ALLOWED_COLUMNS = set("""target_leg,target_role,target_race_id,target_race_name,declared_runner_count,race_draw_status,target_route_id,venue,surface,distance_m,route_variant,target_course_setting,history_population_definition,history_date_floor,history_observed_min_date,history_observed_max_date,broader_base_route_races,broader_base_route_runner_rows,partial_candidate_races,same_condition_runner_rows,condition_match_definition,certified_exact_races,exact_evaluation_status,certified_similar_races,similar_status,course_setting_direct_coverage_rate,renovation_version_known_races,ra_lap_joined_races,ra_lap_join_denominator,ra_lap_join_rate,ra_lap_full_valid_races,ra_lap_full_valid_rate,role_known_runner_rows,role_runner_denominator,role_determination_rate,role_complete_races,role_complete_race_denominator,role_complete_race_rate,any_corner_known_runner_rows,corner_runner_denominator,any_corner_coverage_rate,scenario_labeled_races,scenario_denominator,scenario_label_rate,scenario_taxonomy_status,ability_proxy_status,performance_residual_status,full_curve_geometry_status,future_information_screen,tier3_evidence_status,predraw_evidence_status,strict_other_course_transfer_status,draw_dependent_features_used,identity_join_key,horse_name_join_used,formal_buy,send_order,stake""".split(","))

READINESS_ALLOWED_COLUMNS = set("""target_leg,target_role,target_race_id,target_race_name,target_horse_id,target_horse_name,draw_status,entry_stage,history_run_count,unique_history_race_count,unique_route_base_key_count,ra_joined_history_race_count,ra_history_race_join_rate,route_connected_run_rate,any_corner_known_run_count,final_3f_known_run_count,finish_position_known_run_count,same_condition_partial_race_count,same_condition_ra_join_rate,same_condition_role_determination_rate,same_condition_scenario_label_rate,history_extraction_status,ra_lap_status,route_closure_status,same_condition_evidence_status,prestart_member_level_status,prestart_member_level_blocker,ability_distribution_status,ability_distribution_source_warning,position_role_profile_status,lap_terminal_environment_status,workout_condition_status,workout_source_effective_coverage,workout_source_warning,other_course_transfer_status,other_course_transfer_blocker,ability_proxy_added_to_ai_score,draw_features_status,draw_dependent_features_computed,excluded_draw_dependent_analysis_pipe,predraw_readiness_status,identity_join_method,horse_name_join_used,formal_buy,send_order,stake""".split(","))

ROUTE_COVERAGE_ALLOWED_COLUMNS = set("""target_leg,target_role,target_race_id,target_horse_id,target_horse_name,draw_status,history_run_count,unique_history_race_count,unique_route_base_key_count,ra_joined_history_race_count,ra_history_race_join_rate,route_connected_run_count,route_connected_run_rate,route_exact_run_count,route_partial_run_count,route_fallback_run_count,route_unmapped_run_count,route_context_complete_run_count,first_history_date,last_history_date,future_or_post_target_row_count,route_coverage_status,exact_route_blocker,identity_join_key,horse_name_join_used,draw_dependent_features_used,formal_buy,send_order,stake""".split(","))

TARGET_MANIFEST_KEYS = {"schema_version", "selection_mode", "date", "timezone", "source", "legs"}
TARGET_MANIFEST_LEG_KEYS = {"leg", "race_name", "race_name_aliases", "surface", "distance_m", "post_time", "role"}
ROUTE_CARD_TOP_KEYS = {"schema_version", "date", "principles", "common_traits", "cards", "sources"}
ROUTE_CARD_KEYS = {"leg", "race", "conditions", "post_time", "route_id", "venue", "surface", "distance_m", "course_setting", "current_context", "static_route", "segments", "scenario_role", "target_observable", "proxy_only", "dynamic_updates", "sources"}
ROUTE_CONTEXT_KEYS = {"meeting", "course_setting_day_index", "course_change", "rail_shift", "important_note"}
ROUTE_SEGMENT_KEYS = {"segment", "focus"}
ROUTE_SOURCE_KEYS = {"id", "title", "url"}
OFFICIAL_TOP_KEYS = {"schema_version", "target_date", "fetched_at_jst", "request_method", "current_field_only", "past_performance_cells_excluded", "market_fields_extracted", "identity_join", "audited_declared_differences", "race_count", "runner_count", "races"}
OFFICIAL_RACE_KEYS = {"leg", "race_id", "race_name", "official_entry_url", "http_status", "raw_response_sha256", "raw_response_bytes", "field_boundary", "runners"}
OFFICIAL_RUNNER_KEYS = {"race_id", "horse_id", "horse_name", "frame_no", "horse_no", "sex", "age", "coat_color", "assigned_weight", "jockey", "trainer", "trainer_affiliation", "entry_status"}
OFFICIAL_DIFFERENCE_KEYS = {"race_id", "horse_id", "field", "audited_snapshot_value", "current_official_value", "resolution"}

FREEZE_TOP_LEVEL_KEYS = {"schema_version", "target_date", "observation_sha256", "safety", "overrides"}
FREEZE_SAFETY_KEYS = {"direct_horse_score_edit", "ranking_claims", "probability_claims"}
FREEZE_OVERRIDE_KEYS = {"race_id", "post_time_jst", "main_scenario", "alternative_scenarios", "confidence", "reason", "race_day_notes", "main_danger_horse"}
SOURCE_MANIFEST_KEYS = {"schema_version", "generated_at_jst", "allowlisted_sources", "official_current_entry", "excluded_source_classes", "safety", "outputs"}
SOURCE_RECORD_KEYS = {"kind", "path", "sha256", "bytes"}
SOURCE_OUTPUT_KEYS = {"observation_data", "html", "human_freeze_template"}
SOURCE_OUTPUT_RECORD_KEYS = {"path", "sha256"}
OBSERVATION_TOP_KEYS = {"schema_version", "title", "target_date", "generated_at_jst", "display_order_rule", "safety", "data_contract", "race_count", "runner_count", "races"}
OBSERVATION_RACE_KEYS = {"leg", "win5_role", "race_id", "race_name", "track", "race_no", "surface", "distance_m", "post_time_jst", "runner_count", "official_entry_url", "draw_status", "meeting_day", "course_change", "track_condition", "day_bias", "wind", "course_structure", "geometry_status", "route_match_level", "same_condition_evidence", "ra_lap_join_coverage", "role_classification_coverage", "historical_pace_tendency", "queue_pace_points", "human_scenario", "favorite_collapse", "notes_for_post_race_review", "comparison_columns", "comparison", "horses"}
OBSERVATION_HORSE_KEYS = {"display_order", "horse_id", "basics", "role_position", "ability", "pace_role_traits", "course_shape_traits", "condition", "transfer_evidence", "scenario_sensitivity", "paths", "uncertainty"}
BASICS_KEYS = {"horse_name", "frame_no", "horse_no", "jockey", "trainer", "trainer_affiliation", "sex", "age", "coat_color", "assigned_weight", "entry_status"}
ROLE_POSITION_KEYS = {"base_role", "position_flexibility", "need_lead", "lead_frequency_proxy", "can_rate", "pressure_rivals", "inside_neighbor_context", "outside_neighbor_context", "likely_early_band", "queue_confidence", "queue_notes", "evidence_count", "status"}
ABILITY_KEYS = {"recent_standard", "demonstrated_upper", "lower_band", "repeatability", "evidence_count", "ability_band", "interpretation"}
CONDITION_KEYS = {"workout_relative", "condition_trend", "rest_pattern", "fatigue_rebound_proxy", "state_confidence"}
PATH_KEYS = {"win_path", "lose_path", "favorite_collapse_replacement_path", "fragility_note"}
UNCERTAINTY_KEYS = {"evidence_states", "missing_reason", "confidence", "history_coverage_confidence", "what_remains_unclear", "route_match_level", "history_evidence_count", "ra_joined_history_race_count", "route_coverage_status", "exact_route_blocker"}
TRANSFER_KEYS = {"date", "venue_distance", "observed_race_shape", "observed_role", "what_was_demanded", "how_the_horse_responded", "why_it_transfers", "evidence_strength", "history_race_id", "history_route_context_level"}
SCENARIO_SENSITIVITY_KEYS = {"assessment", "evidence_note", "status", "confidence"}
EVIDENCE_KEYS = {"value", "status", "route_match_level", "evidence_count", "confidence", "note", "missing_reason"}
PACE_TRAIT_KEYS = {"early_positioning", "tracking_speed", "sustained_pressure_tolerance", "front_pressure_tolerance", "fast_finish_fit", "slow_finish_fit", "late_retention", "role_flexibility"}
COURSE_TRAIT_KEYS = {"corner_speed_maintenance", "corner_progression", "straight_acceleration", "high_speed_duration", "slope_tolerance", "stamina", "outside_loss_tolerance", "traffic_flexibility"}

COMPARISON_COLUMNS = (
    "horse_no", "horse_name", "role", "ability_band", "demand_fit",
    "early_burden_fit", "corner_fit", "finish_fit", "condition", "best_scenario",
    "win_path_short", "lose_path_short", "confidence",
)

TRAIT_LABELS = {
    "early_positioning": "序盤位置",
    "tracking_speed": "追走位置維持",
    "sustained_pressure_tolerance": "持続圧耐性",
    "front_pressure_tolerance": "前圧耐性",
    "fast_finish_fit": "高速終盤適合",
    "slow_finish_fit": "消耗終盤適合",
    "late_retention": "終盤残存",
    "role_flexibility": "役割柔軟性",
    "corner_speed_maintenance": "コーナー位置維持",
    "corner_progression": "コーナー進出",
    "straight_acceleration": "直線進出",
    "high_speed_duration": "高速持続",
    "slope_tolerance": "坂・起伏耐性",
    "stamina": "距離持久",
    "outside_loss_tolerance": "外々ロス耐性",
    "traffic_flexibility": "交通柔軟性",
}


class LitePlusError(RuntimeError):
    """Fail-closed validation error."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_iso_datetime(raw: str) -> datetime:
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise LitePlusError(f"invalid ISO datetime: {raw}") from exc
    if value.tzinfo is None:
        raise LitePlusError(f"timezone is required: {raw}")
    return value.astimezone(JST)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def write_json(path: Path, value: Any) -> None:
    write_text(path, pretty_json(value))


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html_module.unescape(re.sub(r"<.*?>", "", value))).strip()


def as_int(raw: Any) -> int | None:
    text = str(raw or "").strip()
    if not text or not re.fullmatch(r"-?\d+", text):
        return None
    return int(text)


def as_float(raw: Any) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def valid_finish_position(raw: Any) -> int | None:
    """Return a JRA flat placing; non-finish sentinel codes stay missing."""
    value = as_int(raw)
    return value if value is not None and 1 <= value <= 18 else None


def valid_final_3f(raw: Any) -> float | None:
    """Exclude fixed-width missing/non-finish sentinels such as 99.9."""
    value = as_float(raw)
    return value if value is not None and 25.0 <= value <= 60.0 else None


def is_false(raw: Any) -> bool:
    return str(raw).strip().lower() in {"false", "0"}


def reject_forbidden_keys(keys: Iterable[str], source_name: str) -> None:
    found = sorted({
        key for key in keys
        if key not in NEGATIVE_ATTESTATION_KEYS
        and any(pattern.search(key.lower()) for pattern in FORBIDDEN_KEY_PATTERNS)
    })
    if found:
        raise LitePlusError(f"forbidden prediction/market keys in {source_name}: {found}")


def reject_forbidden_recursive(value: Any, source_name: str, path: str = "$") -> None:
    if isinstance(value, Mapping):
        reject_forbidden_keys((str(key) for key in value), f"{source_name}:{path}")
        for key, child in value.items():
            if key in {"stake", "formal_buy", "send_order", "market_data_used"}:
                expected = 0 if key == "stake" else False
                if path != "$.safety" or child != expected:
                    raise LitePlusError(f"safety attestation outside fixed safety block: {source_name}:{path}.{key}")
            if key == "market_fields_extracted" and (source_name != "official_snapshot" or path != "$" or child is not False):
                raise LitePlusError(f"official market attestation differs: {source_name}:{path}.{key}")
            reject_forbidden_recursive(child, source_name, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_recursive(child, source_name, f"{path}[{index}]")


def require_exact_keys(value: Mapping[str, Any], expected: set[str], source_name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise LitePlusError(
            f"schema keys differ in {source_name}; missing={sorted(expected - actual)} "
            f"unexpected={sorted(actual - expected)}"
        )


def validate_columns(rows: Sequence[Mapping[str, str]], allowed: set[str], source_name: str) -> None:
    if not rows:
        raise LitePlusError(f"empty source: {source_name}")
    keys = set(rows[0])
    reject_forbidden_keys(keys, source_name)
    unexpected = sorted(keys - allowed)
    if unexpected:
        raise LitePlusError(f"unexpected columns in {source_name}: {unexpected}")


def validate_safety_rows(rows: Sequence[Mapping[str, str]], source_name: str) -> None:
    for index, row in enumerate(rows, start=2):
        if "formal_buy" in row and not is_false(row["formal_buy"]):
            raise LitePlusError(f"formal_buy must be false: {source_name}:{index}")
        if "send_order" in row and not is_false(row["send_order"]):
            raise LitePlusError(f"send_order must be false: {source_name}:{index}")
        if "stake" in row and str(row["stake"]).strip() not in {"0", "0.0"}:
            raise LitePlusError(f"stake must be zero: {source_name}:{index}")
        if "horse_name_join_used" in row and not is_false(row["horse_name_join_used"]):
            raise LitePlusError(f"horse-name join is forbidden: {source_name}:{index}")
        if "ability_proxy_added_to_ai_score" in row and not is_false(row["ability_proxy_added_to_ai_score"]):
            raise LitePlusError(f"legacy AI overlap attestation must be false: {source_name}:{index}")


def validate_target_manifest_schema(value: Mapping[str, Any]) -> None:
    require_exact_keys(value, TARGET_MANIFEST_KEYS, "target_manifest")
    reject_forbidden_recursive(value, "target_manifest")
    legs = value.get("legs")
    if not isinstance(legs, list) or len(legs) != 5:
        raise LitePlusError("target manifest requires five legs")
    for index, leg in enumerate(legs):
        if not isinstance(leg, Mapping):
            raise LitePlusError(f"target manifest leg is not an object: {index}")
        require_exact_keys(leg, TARGET_MANIFEST_LEG_KEYS, f"target_manifest.legs[{index}]")


def validate_route_cards_schema(value: Mapping[str, Any]) -> None:
    require_exact_keys(value, ROUTE_CARD_TOP_KEYS, "route_cards")
    reject_forbidden_recursive(value, "route_cards")
    cards = value.get("cards")
    if not isinstance(cards, list) or len(cards) != 5:
        raise LitePlusError("route cards require five cards")
    for index, card in enumerate(cards):
        if not isinstance(card, Mapping):
            raise LitePlusError(f"route card is not an object: {index}")
        require_exact_keys(card, ROUTE_CARD_KEYS, f"route_cards.cards[{index}]")
        require_exact_keys(card["current_context"], ROUTE_CONTEXT_KEYS, f"route_cards.cards[{index}].current_context")
        for segment_index, segment in enumerate(card["segments"]):
            require_exact_keys(segment, ROUTE_SEGMENT_KEYS, f"route_cards.cards[{index}].segments[{segment_index}]")
        if set(card["scenario_role"]) != {"slow", "middle", "fast"}:
            raise LitePlusError(f"route scenario keys differ: {index}")
        for scenario, roles in card["scenario_role"].items():
            if set(roles) != {"front", "stalker", "closer"}:
                raise LitePlusError(f"route role keys differ: {index}/{scenario}")
    for index, source in enumerate(value["sources"]):
        require_exact_keys(source, ROUTE_SOURCE_KEYS, f"route_cards.sources[{index}]")


def validate_official_snapshot_schema(value: Mapping[str, Any]) -> None:
    require_exact_keys(value, OFFICIAL_TOP_KEYS, "official_snapshot")
    reject_forbidden_recursive(value, "official_snapshot")
    if value.get("schema_version") != "race_intelligence_lite_plus_official_entries_v1":
        raise LitePlusError("official snapshot schema version differs")
    if value.get("request_method") != "GET" or value.get("current_field_only") is not True:
        raise LitePlusError("official snapshot is not a current-field GET capture")
    if value.get("past_performance_cells_excluded") is not True or value.get("market_fields_extracted") is not False:
        raise LitePlusError("official snapshot inclusion boundary differs")
    races = value.get("races")
    if not isinstance(races, list) or len(races) != 5:
        raise LitePlusError("official snapshot requires five races")
    for index, race in enumerate(races):
        require_exact_keys(race, OFFICIAL_RACE_KEYS, f"official_snapshot.races[{index}]")
        for runner_index, runner in enumerate(race["runners"]):
            require_exact_keys(runner, OFFICIAL_RUNNER_KEYS, f"official_snapshot.races[{index}].runners[{runner_index}]")
    for index, difference in enumerate(value["audited_declared_differences"]):
        require_exact_keys(difference, OFFICIAL_DIFFERENCE_KEYS, f"official_snapshot.audited_declared_differences[{index}]")


def evidence(
    value: Any,
    *,
    status: str,
    evidence_count: int = 0,
    confidence: str = "unobserved",
    note: str = "",
    missing_reason: str = "",
    route_match_level: str = "unobserved",
) -> dict[str, Any]:
    if status not in STATUS_VALUES:
        raise LitePlusError(f"invalid evidence status: {status}")
    if confidence not in CONFIDENCE_VALUES:
        raise LitePlusError(f"invalid confidence: {confidence}")
    if route_match_level not in ROUTE_MATCH_VALUES:
        raise LitePlusError(f"invalid route match level: {route_match_level}")
    if status == "unobserved" and not missing_reason:
        raise LitePlusError("unobserved evidence requires missing_reason")
    return {
        "value": value,
        "status": status,
        "route_match_level": route_match_level,
        "evidence_count": int(evidence_count),
        "confidence": confidence,
        "note": note,
        "missing_reason": missing_reason,
    }


def unobserved(reason: str, note: str = "") -> dict[str, Any]:
    return evidence("未観測", status="unobserved", confidence="unobserved", note=note,
                    missing_reason=reason, route_match_level="unobserved")


def parse_official_field_html(raw: bytes, race_id: str, url: str) -> list[dict[str, Any]]:
    """Extract current-entry cells only; past-performance cells are excluded."""
    try:
        text = raw.decode("cp932")
    except UnicodeDecodeError as exc:
        raise LitePlusError(f"JRA page is not cp932: {url}") from exc
    boundary = text.find('<div class="record_unit')
    if boundary < 0:
        raise LitePlusError(f"official field boundary absent: {url}")
    field_html = text[:boundary]
    runners: list[dict[str, Any]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", field_html, flags=re.S):
        current = tr.split('<td class="past', 1)[0]
        horse = re.search(r'pw01dud00(\d{10})/[^\"]+\">(.*?)</a>', current, flags=re.S)
        if not horse:
            continue
        frame = re.search(r'alt="枠(\d+)', current)
        number = re.search(r'<td class="num"[^>]*>(.*?)</td>', current, flags=re.S)
        age = re.search(r'<p class="age">(.*?)</p>', current, flags=re.S)
        weight = re.search(r'<p class="weight">(.*?)<span>kg</span>', current, flags=re.S)
        jockey = re.search(r'<p class="jockey"><a[^>]*>(.*?)</a>', current, flags=re.S)
        trainer = re.search(r'<p class="trainer">(.*?)</p>', current, flags=re.S)
        division = re.search(r'<span class="division">\((.*?)\)</span>', current, flags=re.S)
        age_text = clean_text(age.group(1) if age else "")
        sex_age = re.fullmatch(r"([^0-9/]+)(\d+)/(.+)", age_text)
        row = {
            "race_id": race_id,
            "horse_id": horse.group(1),
            "horse_name": clean_text(horse.group(2)),
            "frame_no": as_int(frame.group(1) if frame else ""),
            "horse_no": as_int(clean_text(number.group(1) if number else "")),
            "sex": sex_age.group(1) if sex_age else "",
            "age": as_int(sex_age.group(2) if sex_age else ""),
            "coat_color": sex_age.group(3) if sex_age else "",
            "assigned_weight": as_float(clean_text(weight.group(1) if weight else "")),
            "jockey": clean_text(jockey.group(1) if jockey else ""),
            "trainer": clean_text(re.split(r'<span class="division">', trainer.group(1), maxsplit=1)[0]
                                  if trainer else ""),
            "trainer_affiliation": clean_text(division.group(1) if division else ""),
            "entry_status": "active",
        }
        required = ("frame_no", "horse_no", "sex", "age", "assigned_weight", "jockey", "trainer")
        if any(row[field] in {None, ""} for field in required):
            raise LitePlusError(f"incomplete official runner row: {race_id}/{row['horse_id']}")
        runners.append(row)
    if not runners:
        raise LitePlusError(f"no current runners parsed: {url}")
    return runners


def fetch_official_snapshot(
    targets_path: Path,
    entries_path: Path,
    fetched_at_jst: str,
) -> dict[str, Any]:
    targets = read_csv(targets_path)
    entries = read_csv(entries_path)
    validate_columns(targets, TARGET_ALLOWED_COLUMNS, targets_path.name)
    validate_columns(entries, ENTRY_ALLOWED_COLUMNS, entries_path.name)
    validate_safety_rows(targets, targets_path.name)
    validate_safety_rows(entries, entries_path.name)
    if len(targets) != 5:
        raise LitePlusError("official capture requires exactly five resolved targets")
    timestamp = parse_iso_datetime(fetched_at_jst).isoformat(timespec="seconds")
    expected = {(row["race_id"], row["horse_id"]): row for row in entries}
    if len(expected) != len(entries):
        raise LitePlusError("duplicate race_id+horse_id in declared entry snapshot")

    races: list[dict[str, Any]] = []
    observed_keys: set[tuple[str, str]] = set()
    audited_declared_differences: list[dict[str, str]] = []
    for target in sorted(targets, key=lambda row: int(row["manifest_leg"])):
        url = target["official_entry_url"]
        parsed_url = urllib.parse.urlsplit(url)
        if (parsed_url.scheme != "https" or parsed_url.hostname != "www.jra.go.jp"
                or parsed_url.path != "/JRADB/accessD.html"
                or not parsed_url.query.startswith("CNAME=pw01dde")):
            raise LitePlusError(f"official URL is outside the JRA entry allowlist: {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 RaceIntelligenceLitePlus/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:  # read-only GET
            raw = response.read()
            status = getattr(response, "status", 200)
            final_url = response.geturl()
        if status != 200:
            raise LitePlusError(f"JRA GET failed ({status}): {url}")
        final_parts = urllib.parse.urlsplit(final_url)
        if (final_parts.scheme != "https" or final_parts.hostname != "www.jra.go.jp"
                or final_parts.path != "/JRADB/accessD.html"
                or not final_parts.query.startswith("CNAME=pw01dde")):
            raise LitePlusError(f"official GET redirected outside the JRA entry allowlist: {final_url}")
        race_id = target["race_id"]
        runners = parse_official_field_html(raw, race_id, url)
        for runner in runners:
            key = (race_id, runner["horse_id"])
            if key not in expected:
                raise LitePlusError(f"official runner not in audited declared universe: {key}")
            declared = expected[key]
            for field, official_key in (
                ("horse_name", "horse_name"), ("jockey", "jockey"),
                ("trainer", "trainer"), ("carried_weight_kg", "assigned_weight"),
            ):
                left = str(declared[field]).replace(" ", "").strip()
                right = str(runner[official_key]).replace(" ", "").replace(".0", "").strip()
                if field == "carried_weight_kg":
                    left = left.replace(".0", "")
                if left != right:
                    # The 2026-08-21 audit snapshot has a known parsing limitation for
                    # non-JRA trainers rendered without an <a> element.  Identity is
                    # still horse-id exact; current official trainer text supersedes
                    # only that descriptive field and the difference is preserved.
                    if field == "trainer":
                        audited_declared_differences.append({
                            "race_id": race_id,
                            "horse_id": runner["horse_id"],
                            "field": field,
                            "audited_snapshot_value": declared[field],
                            "current_official_value": str(runner[official_key]),
                            "resolution": "current official field retained; identity join unchanged",
                        })
                    else:
                        raise LitePlusError(f"official/declaration mismatch {key} {field}: {left!r}!={right!r}")
            observed_keys.add(key)
        races.append({
            "leg": int(target["manifest_leg"]),
            "race_id": race_id,
            "race_name": target["race_name"],
            "official_entry_url": url,
            "http_status": status,
            "raw_response_sha256": sha256_bytes(raw),
            "raw_response_bytes": len(raw),
            "field_boundary": "before first div.record_unit; each row truncated before first td.past",
            "runners": sorted(runners, key=lambda row: row["horse_no"]),
        })
    if observed_keys != set(expected):
        missing = sorted(set(expected) - observed_keys)
        raise LitePlusError(f"official universe mismatch; missing={missing}")
    return {
        "schema_version": "race_intelligence_lite_plus_official_entries_v1",
        "target_date": TARGET_DATE.isoformat(),
        "fetched_at_jst": timestamp,
        "request_method": "GET",
        "current_field_only": True,
        "past_performance_cells_excluded": True,
        "market_fields_extracted": False,
        "identity_join": "race_id+horse_id exact",
        "audited_declared_differences": audited_declared_differences,
        "race_count": len(races),
        "runner_count": len(observed_keys),
        "races": races,
    }


def observed_runs(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    result = [dict(row) for row in rows if str(row.get("history_run_present", "")).lower() == "true"]
    result.sort(key=lambda row: (row["race_date"], row["history_race_id"]), reverse=True)
    return result


def corner_positions(row: Mapping[str, str]) -> list[int]:
    return [value for value in (as_int(row.get(f"corner{i}")) for i in range(1, 5)) if value and value > 0]


def role_band(position: int | None) -> str:
    if position is None:
        return "unobserved"
    if position == 1:
        return "lead"
    if position <= 4:
        return "front"
    if position <= 7:
        return "stalking"
    if position <= 11:
        return "midpack"
    return "rear"


def role_profile(runs: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    samples: list[tuple[str, int]] = []
    for run in runs[:5]:
        corners = corner_positions(run)
        if corners:
            samples.append((role_band(corners[0]), corners[0]))
    if not samples:
        return {
            "base_role": "unobserved", "position_flexibility": "unobserved",
            "need_lead": "unobserved", "lead_frequency_proxy": "unobserved",
            "can_rate": "unobserved", "evidence_count": 0,
            "confidence": "unobserved",
        }
    role_counts = Counter(role for role, _ in samples)
    role_order = {value: index for index, value in enumerate(ROLE_VALUES)}
    base = sorted(role_counts, key=lambda role: (-role_counts[role], role_order[role]))[0]
    unique = len(role_counts)
    flexibility = "high" if unique >= 3 else "medium" if unique == 2 else "low"
    lead_count = sum(position == 1 for _, position in samples)
    lead_frequency = "high" if lead_count * 2 >= len(samples) and lead_count else "some" if lead_count else "low"
    can_rate_hits = 0
    for run in runs:
        corners = corner_positions(run)
        finish = valid_finish_position(run.get("finish_position"))
        if corners and corners[0] >= 3 and finish is not None and finish <= 3:
            can_rate_hits += 1
    confidence = "high" if len(samples) >= 5 else "medium" if len(samples) >= 2 else "low"
    return {
        "base_role": base,
        "position_flexibility": flexibility,
        "need_lead": "unobserved",
        "lead_frequency_proxy": lead_frequency,
        "can_rate": "observed" if can_rate_hits else "not_confirmed",
        "can_rate_evidence_count": can_rate_hits,
        "evidence_count": len(samples),
        "confidence": confidence,
        "sample_roles": [role for role, _ in samples],
    }


def finish_band(position: float | int | None) -> str:
    if position is None:
        return "不明"
    if position <= 3:
        return "高（上位着順観測）"
    if position <= 7:
        return "中（中位着順観測）"
    return "低（後方着順観測）"


def ability_block(runs: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    finishes = [value for value in (valid_finish_position(run.get("finish_position")) for run in runs[:5]) if value]
    if not finishes:
        missing = "target前のJRA平地実走がない"
        return {
            "recent_standard": unobserved(missing),
            "demonstrated_upper": unobserved(missing),
            "lower_band": unobserved(missing),
            "repeatability": unobserved(missing),
            "evidence_count": 0,
            "ability_band": "不明",
            "interpretation": "過去着順の観測帯であり、このレースの能力順位ではない",
        }
    common = {
        "status": "derived", "evidence_count": len(finishes),
        "confidence": "high" if len(finishes) >= 5 else "medium" if len(finishes) >= 2 else "low",
        "note": "直近最大5走の実着順だけを帯化。相手・クラス補正なし。対象馬間の順位ではない",
        "route_match_level": "unclassified",
    }
    median = statistics.median(finishes)
    spread = max(finishes) - min(finishes)
    repeatability = "安定" if len(finishes) >= 3 and spread <= 3 else "変動" if len(finishes) >= 3 else "証拠少"
    return {
        "recent_standard": evidence(finish_band(median), **common),
        "demonstrated_upper": evidence(finish_band(min(finishes)), **common),
        "lower_band": evidence(finish_band(max(finishes)), **common),
        "repeatability": evidence(repeatability, **common),
        "evidence_count": len(finishes),
        "ability_band": finish_band(median),
        "interpretation": "過去着順の観測帯であり、このレースの能力順位ではない",
    }


def support_from_boolean(samples: Sequence[bool], note: str) -> dict[str, Any]:
    count = len(samples)
    if not count:
        return unobserved("必要な角位置または着順がない", note)
    hits = sum(samples)
    value = "supportive" if hits * 5 >= count * 3 else "adverse" if hits * 10 <= count * 3 else "neutral"
    return evidence(value, status="proxy", evidence_count=count,
                    confidence="high" if count >= 5 else "medium" if count >= 2 else "low",
                    note=f"{note}（該当 {hits}/{count}走）", route_match_level="unclassified")


def build_traits(runs: Sequence[Mapping[str, str]], profile: Mapping[str, Any],
                 target: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    corner_runs = [(run, corner_positions(run)) for run in runs[:8] if corner_positions(run)]
    early = [corners[0] <= 5 for _, corners in corner_runs]
    retained = []
    corner_held = []
    corner_progressed = []
    straight_progressed = []
    for run, corners in corner_runs:
        finish = valid_finish_position(run.get("finish_position"))
        if finish:
            retained.append(finish <= corners[-1] + 2)
            straight_progressed.append(finish < corners[-1])
        if len(corners) >= 2:
            corner_held.append(corners[-1] <= corners[0] + 2)
            corner_progressed.append(corners[-1] < corners[0])

    base_role = profile["base_role"]
    pace_traits = {
        "early_positioning": evidence(
            base_role, status="proxy", evidence_count=profile["evidence_count"],
            confidence=profile["confidence"], route_match_level="unclassified",
            note="直近最大5走の最初に観測された角位置を絶対位置帯へ変換。頭数補正なし",
        ) if base_role != "unobserved" else unobserved("角位置履歴がない"),
        "tracking_speed": support_from_boolean(early, "最初の観測角で5番手以内を追走できたかの位置proxy"),
        "sustained_pressure_tolerance": unobserved("個体別区間速度とpressure durationが未整備"),
        "front_pressure_tolerance": unobserved("先行競合中の個体負荷を直接観測していない"),
        "fast_finish_fit": unobserved("独立したFAST終盤taxonomyが凍結されていない"),
        "slow_finish_fit": unobserved("独立した消耗終盤taxonomyが凍結されていない"),
        "late_retention": support_from_boolean(retained, "最終角位置から着順を2つ超えて落とさないかの位置proxy"),
        "role_flexibility": evidence(
            profile["position_flexibility"], status="proxy", evidence_count=profile["evidence_count"],
            confidence=profile["confidence"], route_match_level="unclassified",
            note="直近角位置帯の種類数。自在性そのものの直接観測ではない",
        ) if base_role != "unobserved" else unobserved("角位置履歴がない"),
    }

    same_venue_surface = [
        run for run in runs
        if run.get("venue") == target["venue"] and normalize_surface(run.get("surface", "")) == target["surface"]
    ]
    distance_runs = [
        run for run in runs
        if normalize_surface(run.get("surface", "")) == target["surface"]
        and (as_int(run.get("distance_m")) or 0) >= int(target["distance_m"])
    ]
    stamina_finishes = [valid_finish_position(run.get("finish_position")) for run in distance_runs]
    venue_finishes = [valid_finish_position(run.get("finish_position")) for run in same_venue_surface]
    stamina_samples = [finish <= 6 for finish in stamina_finishes if finish is not None]
    venue_samples = [finish <= 6 for finish in venue_finishes if finish is not None]
    course_traits = {
        "corner_speed_maintenance": support_from_boolean(corner_held, "最初と最後の角の位置差によるproxy。実速度ではない"),
        "corner_progression": support_from_boolean(corner_progressed, "角区間で通過順位を上げたかのproxy"),
        "straight_acceleration": support_from_boolean(straight_progressed, "最終角から着順を上げたかのproxy"),
        "high_speed_duration": unobserved("個体別区間速度がない"),
        "slope_tolerance": support_from_boolean(venue_samples, "同競馬場・同馬場の6着以内観測を起伏対応の弱いproxyとして使用"),
        "stamina": support_from_boolean(stamina_samples, "同馬場かつ今回以上の距離で6着以内かのproxy"),
        "outside_loss_tolerance": unobserved("過去走の実走行距離・外々経路を使用していない"),
        "traffic_flexibility": unobserved("包まれ・進路・砂被りのイベントデータがない"),
    }
    return pace_traits, course_traits


def normalize_surface(value: str) -> str:
    lowered = value.strip().lower()
    if lowered == "turf" or lowered.startswith("芝"):
        return "turf"
    if lowered == "dirt" or lowered.startswith("ダート"):
        return "dirt"
    return lowered


def normalize_name(value: str) -> str:
    return re.sub(r"[\s\u3000]", "", value)


def unique_index(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    source_name: str,
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    result: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        if any(not part for part in key):
            raise LitePlusError(f"empty identity key in {source_name}: {key}")
        if key in result:
            raise LitePlusError(f"duplicate identity key in {source_name}: {key}")
        result[key] = row
    return result


def validate_cross_source_universe(
    targets_manifest: Mapping[str, Any],
    route_cards: Mapping[str, Any],
    targets: Sequence[Mapping[str, str]],
    entries: Sequence[Mapping[str, str]],
    history: Sequence[Mapping[str, str]],
    coverage: Sequence[Mapping[str, str]],
    readiness: Sequence[Mapping[str, str]],
    route_coverage: Sequence[Mapping[str, str]],
    official_snapshot: Mapping[str, Any],
) -> None:
    if str(targets_manifest.get("date")) != TARGET_DATE.strftime("%Y%m%d"):
        raise LitePlusError("target manifest date differs from Sunday target")
    manifest_legs = unique_index(targets_manifest["legs"], ("leg",), "target_manifest.legs")
    target_legs = unique_index(targets, ("manifest_leg",), "resolved_targets")
    card_legs = unique_index(route_cards["cards"], ("leg",), "route_cards.cards")
    expected_legs = {(str(index),) for index in range(1, 6)}
    if set(manifest_legs) != expected_legs or set(target_legs) != expected_legs or set(card_legs) != expected_legs:
        raise LitePlusError("leg universes differ across manifest, resolved targets and route cards")
    race_ids: set[str] = set()
    for leg_key in sorted(expected_legs):
        manifest_leg = manifest_legs[leg_key]
        target = target_legs[leg_key]
        card = card_legs[leg_key]
        race_id = target["race_id"]
        if race_id in race_ids:
            raise LitePlusError(f"duplicate resolved race_id: {race_id}")
        race_ids.add(race_id)
        aliases = [manifest_leg["race_name"], *manifest_leg.get("race_name_aliases", [])]
        if normalize_name(target["manifest_race_name"]) != normalize_name(manifest_leg["race_name"]):
            raise LitePlusError(f"manifest race name differs at leg {leg_key[0]}")
        if normalize_name(target["race_name"]) not in {normalize_name(name) for name in aliases}:
            raise LitePlusError(f"resolved official race name is not a manifest name/alias: {race_id}")
        if target["win5_role"] != manifest_leg["role"] or target["selection_mode"] != "race_name_only":
            raise LitePlusError(f"role/selection mode differs at leg {leg_key[0]}")
        if normalize_surface(target["surface_manifest"]) != normalize_surface(manifest_leg["surface"]):
            raise LitePlusError(f"manifest surface differs at leg {leg_key[0]}")
        if int(target["distance_m"]) != int(manifest_leg["distance_m"]):
            raise LitePlusError(f"manifest distance differs at leg {leg_key[0]}")
        if target["post_time_jst"] != manifest_leg["post_time"]:
            raise LitePlusError(f"manifest post time differs at leg {leg_key[0]}")
        if card["venue"] != target["venue"] or normalize_surface(card["surface"]) != normalize_surface(target["surface_manifest"]):
            raise LitePlusError(f"route card venue/surface differs at leg {leg_key[0]}")
        if int(card["distance_m"]) != int(target["distance_m"]) or card["post_time"] != target["post_time_jst"]:
            raise LitePlusError(f"route card distance/post time differs at leg {leg_key[0]}")
        if str(target["target_date"]).replace("-", "") != TARGET_DATE.strftime("%Y%m%d"):
            raise LitePlusError(f"resolved target date differs: {race_id}")

    declared = unique_index(entries, ("race_id", "horse_id"), "entries")
    readiness_index = unique_index(readiness, ("target_race_id", "target_horse_id"), "readiness")
    route_index = unique_index(route_coverage, ("target_race_id", "target_horse_id"), "route_coverage")
    history_horse_keys = {
        (row["target_race_id"], row["target_horse_id"])
        for row in history
    }
    coverage_index = unique_index(coverage, ("target_race_id",), "coverage")
    official_index = flatten_official(official_snapshot)
    if set(declared) != set(readiness_index) or set(declared) != set(route_index):
        raise LitePlusError("entry/readiness/route-coverage horse universes differ")
    if set(declared) != history_horse_keys:
        raise LitePlusError("entry/history horse universes differ")
    if set(declared) != set(official_index):
        raise LitePlusError("entry/current-official horse universes differ")
    if {(race_id,) for race_id in race_ids} != set(coverage_index):
        raise LitePlusError("resolved target/same-condition race universes differ")
    for target in targets:
        race_id = target["race_id"]
        count = sum(key[0] == race_id for key in declared)
        if count != int(target["declared_runner_count"]):
            raise LitePlusError(f"declared runner count differs: {race_id}")
        if int(coverage_index[(race_id,)]["declared_runner_count"]) != count:
            raise LitePlusError(f"coverage runner count differs: {race_id}")
    if any(row["target_race_id"] not in race_ids for row in history):
        raise LitePlusError("history includes a target outside the five-race universe")


def transfer_score(run: Mapping[str, str], target: Mapping[str, Any]) -> tuple[int, str]:
    surface_match = normalize_surface(run.get("surface", "")) == target["surface"]
    distance_delta = abs((as_int(run.get("distance_m")) or 0) - int(target["distance_m"]))
    venue_match = run.get("venue") == target["venue"]
    score = (6 if venue_match and surface_match and distance_delta == 0 else
             4 if surface_match and distance_delta == 0 else
             3 if surface_match and distance_delta <= 200 else
             1 if surface_match else 0)
    reason = ("同競馬場・同馬場・同距離" if score == 6 else
              "同馬場・同距離" if score == 4 else
              "同馬場・近接距離" if score == 3 else
              "同馬場のみ" if score == 1 else "条件接続が弱い")
    return score, reason


def build_transfer_evidence(runs: Sequence[Mapping[str, str]], target: Mapping[str, Any]) -> list[dict[str, Any]]:
    scored = [(transfer_score(run, target), run) for run in runs]
    scored.sort(key=lambda item: (item[0][0], item[1]["race_date"], item[1]["history_race_id"]), reverse=True)
    result: list[dict[str, Any]] = []
    for (score, reason), run in scored[:3]:
        corners = corner_positions(run)
        finish = valid_finish_position(run.get("finish_position"))
        f3 = valid_final_3f(run.get("final_3f_sec"))
        route_match = "partial" if score == 6 else "unclassified"
        strength = "partial-strong" if score == 6 else "qualitative-medium" if score >= 3 else "qualitative-limited"
        observed_role = role_band(corners[0]) if corners else "unobserved"
        response_parts = []
        if corners:
            response_parts.append(f"角位置 {'-'.join(map(str, corners))}")
        if finish:
            response_parts.append(f"{finish}着")
        elif str(run.get("finish_position", "")).strip():
            response_parts.append("非完走等（着順sentinelを数値評価から除外）")
        if f3 is not None:
            response_parts.append(f"上がり3F {f3:.1f}秒")
        ra_joined = str(run.get("ra_joined", "")).lower() == "true"
        result.append({
            "date": run["race_date"],
            "venue_distance": f"{run['venue']} {normalize_surface(run['surface'])} {run['distance_m']}m",
            "observed_race_shape": evidence(
                "未観測（RA race-level joinフラグのみ）", status="unobserved",
                evidence_count=0, confidence="unobserved", route_match_level=route_match,
                note=("RA結合フラグは確認済み" if ra_joined else "RA結合も未確認")
                     + "。lap配列や凍結済みshape labelをこの入力に含めない",
                missing_reason="独立S/M/F shapeとlap配列がmarket-excluded history scopeにない",
            ),
            "observed_role": evidence(
                observed_role, status="derived" if corners else "unobserved",
                evidence_count=1 if corners else 0, confidence="low" if corners else "unobserved",
                route_match_level=route_match,
                note="最初に観測された角位置の絶対位置帯" if corners else "",
                missing_reason="角位置なし" if not corners else "",
            ),
            "what_was_demanded": evidence(
                f"{normalize_surface(run['surface'])}{run['distance_m']}mでの追走位置と終盤残存",
                status="derived", evidence_count=1, confidence="low", route_match_level=route_match,
                note="詳細geometry/pressureは接続していない",
            ),
            "how_the_horse_responded": evidence(
                "、".join(response_parts) if response_parts else "観測不足",
                status="observed" if response_parts else "unobserved", evidence_count=1 if response_parts else 0,
                confidence="medium" if response_parts else "unobserved", route_match_level=route_match,
                missing_reason="角位置・着順・上がりがない" if not response_parts else "",
            ),
            "why_it_transfers": evidence(
                f"{reason}。今回への厳密転用ではなく観測参照",
                status="proxy", evidence_count=1, confidence="medium" if score >= 4 else "low",
                route_match_level=route_match,
                note="renovation/versionとstrict transfer metricは未整備",
            ),
            "evidence_strength": strength,
            "history_race_id": run["history_race_id"],
            "history_route_context_level": run.get("route_match_level", "unobserved"),
        })
    return result


def scenario_sensitivity() -> dict[str, dict[str, Any]]:
    reason = "独立したSLOW/MIDDLE/FAST taxonomyと個体別scenario responseが凍結されていない"
    return {
        scenario: {
            "assessment": "unobserved",
            "evidence_note": reason,
            "status": "unobserved",
            "confidence": "unobserved",
        }
        for scenario in SCENARIO_VALUES
    }


def requirement_role(base_role: str) -> str:
    if base_role in {"lead", "front"}:
        return "front"
    if base_role in {"stalking", "midpack"}:
        return "stalker"
    return "closer"


def short_requirements(card: Mapping[str, Any], base_role: str) -> list[str]:
    matrix_role = requirement_role(base_role)
    union: list[str] = []
    for scenario in ("slow", "middle", "fast"):
        for trait in card["scenario_role"][scenario][matrix_role]:
            if trait not in union:
                union.append(trait)
    return union[:5]


def build_paths(profile: Mapping[str, Any], card: Mapping[str, Any], pace_traits: Mapping[str, Any]) -> dict[str, str]:
    role = profile["base_role"]
    requirements = short_requirements(card, role)
    readable = [TRAIT_LABELS.get(value, value) for value in requirements]
    if role == "unobserved":
        win = "役割未観測。発馬後に位置と追走可否を確認し、要求能力との接続を再評価"
        lose = "役割証拠がなく、位置取りと負荷配分が想定から外れる"
    else:
        win = f"{role}帯で過大な位置確保コストを払わず、{'・'.join(readable)}を満たす"
        lose = "序盤の役割が崩れ、コーナーまたは終盤で観測済みの位置維持proxyが再現しない"
    if profile.get("lead_frequency_proxy") == "high":
        lose += "。先頭頻度は高いが、ハナ依存性そのものは未観測"
    elif pace_traits["late_retention"]["value"] == "adverse":
        lose += "。終盤残存proxyは adverse"
    return {
        "win_path": win,
        "lose_path": lose,
        "favorite_collapse_replacement_path": "市場人気は未観測。先行依存馬が崩れる場合は役割柔軟性が観測された追走帯が構造上の受益候補（馬の推薦・順位ではない）",
        "fragility_note": "定性説明のみ。相手関係補正、勝率、順位、オッズは含まない",
    }


def confidence_from_count(count: int) -> str:
    return "high" if count >= 8 else "medium" if count >= 3 else "low" if count else "unobserved"


def build_horse(
    runner: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, str]],
    readiness: Mapping[str, str],
    route_coverage: Mapping[str, str],
    target: Mapping[str, Any],
    card: Mapping[str, Any],
) -> dict[str, Any]:
    runs = observed_runs(history_rows)
    profile = role_profile(runs)
    ability = ability_block(runs)
    pace_traits, course_traits = build_traits(runs, profile, target)
    transfers = build_transfer_evidence(runs, target)
    last_date = date.fromisoformat(runs[0]["race_date"]) if runs else None
    rest_days = (TARGET_DATE - last_date).days if last_date else None
    condition = {
        "workout_relative": unobserved("対象馬as-of調教join未実行。全体source coverageを個体値へ転用しない"),
        "condition_trend": unobserved("状態を直接示す対象日付付き個体データがない"),
        "rest_pattern": evidence(
            f"最終出走から{rest_days}日", status="derived", evidence_count=1,
            confidence="high", route_match_level="unclassified", note=f"最終観測日 {last_date.isoformat()}",
        ) if last_date else unobserved("target前の実走がない"),
        "fatigue_rebound_proxy": unobserved("疲労・反動を定義する個体時系列proxyが未承認"),
        "state_confidence": "low" if last_date else "unobserved",
    }
    paths = build_paths(profile, card, pace_traits)
    history_coverage_confidence = confidence_from_count(len(runs))
    overall_confidence = "low" if runs else "unobserved"
    evidence_states = sorted({
        item["status"]
        for item in walk_evidence({
            "ability": ability,
            "pace_role_traits": pace_traits,
            "course_shape_traits": course_traits,
            "condition": condition,
            "transfer_evidence": transfers,
        })
    })
    return {
        "display_order": runner["horse_no"],
        "horse_id": runner["horse_id"],
        "basics": {
            "horse_name": runner["horse_name"],
            "frame_no": runner["frame_no"],
            "horse_no": runner["horse_no"],
            "jockey": runner["jockey"],
            "trainer": runner["trainer"],
            "trainer_affiliation": runner["trainer_affiliation"],
            "sex": runner["sex"],
            "age": runner["age"],
            "coat_color": runner["coat_color"],
            "assigned_weight": runner["assigned_weight"],
            "entry_status": runner["entry_status"],
        },
        "role_position": {
            "base_role": profile["base_role"],
            "position_flexibility": profile["position_flexibility"],
            "need_lead": profile["need_lead"],
            "lead_frequency_proxy": profile["lead_frequency_proxy"],
            "can_rate": profile["can_rate"],
            "pressure_rivals": [],
            "inside_neighbor_context": "後でrace queueから付与",
            "outside_neighbor_context": "後でrace queueから付与",
            "likely_early_band": profile["base_role"],
            "queue_confidence": profile["confidence"],
            "queue_notes": [],
            "evidence_count": profile["evidence_count"],
            "status": "proxy" if profile["base_role"] != "unobserved" else "unobserved",
        },
        "ability": ability,
        "pace_role_traits": pace_traits,
        "course_shape_traits": course_traits,
        "condition": condition,
        "transfer_evidence": transfers,
        "scenario_sensitivity": scenario_sensitivity(),
        "paths": paths,
        "uncertainty": {
            "evidence_states": evidence_states,
            "missing_reason": "target個体調教、個体別lap、traffic、厳密geometry、凍結済みS/M/F responseが未接続",
            "confidence": overall_confidence,
            "history_coverage_confidence": history_coverage_confidence,
            "what_remains_unclear": "当日状態、馬場、風、先行競合の実強度、枠からの実走経路、相手関係補正",
            "route_match_level": "partial" if runs else "unobserved",
            "history_evidence_count": len(runs),
            "ra_joined_history_race_count": as_int(readiness.get("ra_joined_history_race_count")) or 0,
            "route_coverage_status": route_coverage.get("route_coverage_status", "unobserved"),
            "exact_route_blocker": route_coverage.get("exact_route_blocker", "unobserved"),
        },
    }


def attach_queue_context(horses: list[dict[str, Any]]) -> None:
    ordered = sorted(horses, key=lambda horse: horse["basics"]["horse_no"])
    pressure = [
        horse for horse in ordered
        if horse["role_position"]["base_role"] in {"lead", "front"}
    ]
    for index, horse in enumerate(ordered):
        role = horse["role_position"]
        others = [item for item in pressure if item["horse_id"] != horse["horse_id"]]
        role["pressure_rivals"] = [
            {"horse_no": item["basics"]["horse_no"], "horse_name": item["basics"]["horse_name"],
             "role": item["role_position"]["base_role"]}
            for item in others
        ]
        inside = ordered[index - 1] if index > 0 else None
        outside = ordered[index + 1] if index + 1 < len(ordered) else None
        role["inside_neighbor_context"] = (
            f"{inside['basics']['horse_no']} {inside['basics']['horse_name']} / {inside['role_position']['base_role']}"
            if inside else "最内側（内隣なし）"
        )
        role["outside_neighbor_context"] = (
            f"{outside['basics']['horse_no']} {outside['basics']['horse_name']} / {outside['role_position']['base_role']}"
            if outside else "最外側（外隣なし）"
        )
        notes: list[str] = []
        if role["base_role"] in {"lead", "front"} and others:
            notes.append(f"先行争い候補が他に{len(others)}頭。位置確保コストを観測")
        if role["lead_frequency_proxy"] == "high":
            notes.append("先頭頻度proxyは高い。need-lead/ハナ失敗依存性は未観測")
        if horse["basics"]["horse_no"] == 1 and role["base_role"] not in {"lead", "front"}:
            notes.append("内で包まれる可能性は未観測。発馬後の進路を確認")
        if horse["basics"]["horse_no"] == ordered[-1]["basics"]["horse_no"]:
            notes.append("最外。外々の実距離ロスは未観測")
        if not notes:
            notes.append("枠順は観測済み。実際の包まれ・外々・位置確保コストは発走後イベント")
        role["queue_notes"] = notes


def route_summary(card: Mapping[str, Any]) -> dict[str, Any]:
    route_lines = list(card["static_route"])
    segments = list(card["segments"])

    def selected(words: Sequence[str], fallback: str) -> str:
        hits = [line for line in route_lines if any(word in line for word in words)]
        return " / ".join(hits) if hits else fallback

    first = segments[0]["focus"] if segments else "未観測"
    return {
        "summary": evidence(" / ".join(route_lines), status="observed", evidence_count=len(route_lines),
                            confidence="medium", route_match_level="partial",
                            note="JRA course/program由来の定性route card。数値geometry調整には不使用"),
        "first_corner_burden": evidence(first, status="derived", evidence_count=1,
                                        confidence="medium", route_match_level="partial",
                                        note="route cardの最初のsegment focus"),
        "slope_straight_corner": evidence(
            selected(("坂", "上り", "下り", "平坦", "直線", "コーナー", "角", "起伏"), "定性記述なし"),
            status="observed", evidence_count=len(route_lines), confidence="medium",
            route_match_level="partial", note="完全curve geometryは未整備",
        ),
        "segments": segments,
    }


def scenario_distribution(raw: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed = as_int(value)
        if parsed is not None:
            result[key] = parsed
    return result


def race_human_template(target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "main_scenario": "",
        "alternative_scenarios": [],
        "confidence": "",
        "reason": "",
        "race_day_notes": "",
        "main_danger_horse": "",
        "freeze_status": "UNFROZEN_HUMAN_INPUT_REQUIRED",
        "freeze_timestamp": "",
        "post_race_review": "",
        "post_time_jst": f"{TARGET_DATE.isoformat()}T{target['post_time_jst']}:00+09:00",
    }


def comparison_row(horse: Mapping[str, Any], card: Mapping[str, Any]) -> dict[str, str]:
    basics = horse["basics"]
    role = horse["role_position"]["base_role"]
    corner = horse["course_shape_traits"]["corner_speed_maintenance"]
    finish = horse["pace_role_traits"]["late_retention"]
    requirements = short_requirements(card, role)
    observed_requirements = [
        trait for trait in requirements
        if trait in horse["pace_role_traits"] and horse["pace_role_traits"][trait]["status"] != "unobserved"
        or trait in horse["course_shape_traits"] and horse["course_shape_traits"][trait]["status"] != "unobserved"
    ]
    demand_fit = "一部証拠" if observed_requirements else "未観測"
    if role in {"lead", "front"}:
        burden = "位置確保コスト要確認"
    elif role == "unobserved":
        burden = "未観測"
    else:
        burden = "追走遅れ/進路を確認"
    return {
        "horse_no": str(basics["horse_no"]),
        "horse_name": basics["horse_name"],
        "role": role,
        "ability_band": horse["ability"]["ability_band"],
        "demand_fit": demand_fit,
        "early_burden_fit": burden,
        "corner_fit": str(corner["value"]),
        "finish_fit": str(finish["value"]),
        "condition": "未観測",
        "best_scenario": "未観測",
        "win_path_short": horse["paths"]["win_path"],
        "lose_path_short": horse["paths"]["lose_path"],
        "confidence": horse["uncertainty"]["confidence"],
    }


def flatten_official(snapshot: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for race in snapshot["races"]:
        for runner in race["runners"]:
            key = (race["race_id"], runner["horse_id"])
            if key in result:
                raise LitePlusError(f"duplicate official runner: {key}")
            result[key] = dict(runner)
    return result


def build_observation(
    *,
    targets_manifest: Mapping[str, Any],
    route_cards: Mapping[str, Any],
    targets: Sequence[Mapping[str, str]],
    entries: Sequence[Mapping[str, str]],
    history: Sequence[Mapping[str, str]],
    coverage: Sequence[Mapping[str, str]],
    readiness: Sequence[Mapping[str, str]],
    route_coverage: Sequence[Mapping[str, str]],
    official_snapshot: Mapping[str, Any],
    generated_at_jst: str,
) -> dict[str, Any]:
    validate_target_manifest_schema(targets_manifest)
    validate_route_cards_schema(route_cards)
    if targets_manifest.get("selection_mode") != "race_name_only":
        raise LitePlusError("target manifest must use race_name_only")
    if len(targets) != 5 or len(route_cards.get("cards", [])) != 5:
        raise LitePlusError("exactly five targets and route cards are required")
    validate_columns(entries, ENTRY_ALLOWED_COLUMNS, "entries")
    validate_columns(history, HISTORY_ALLOWED_COLUMNS, "history")
    validate_columns(targets, TARGET_ALLOWED_COLUMNS, "targets")
    validate_columns(coverage, COVERAGE_ALLOWED_COLUMNS, "coverage")
    validate_columns(readiness, READINESS_ALLOWED_COLUMNS, "readiness")
    validate_columns(route_coverage, ROUTE_COVERAGE_ALLOWED_COLUMNS, "route_coverage")
    for name, rows in (("targets", targets), ("entries", entries), ("history", history),
                       ("coverage", coverage), ("readiness", readiness),
                       ("route_coverage", route_coverage)):
        validate_safety_rows(rows, name)
    if any(str(row.get("historical_market_columns_included", "")).lower() != "false" for row in history):
        raise LitePlusError("history source must certify historical market columns excluded")
    if any(str(row.get("draw_dependent_features_used", "")).lower() != "false" for row in history):
        raise LitePlusError("history source unexpectedly used draw-dependent features")
    if any(str(row.get("future_or_post_target_flag", "")).lower() != "false" for row in history):
        raise LitePlusError("future/post-target history detected")
    validate_official_snapshot_schema(official_snapshot)
    if official_snapshot.get("runner_count") != 70:
        raise LitePlusError("official snapshot must contain all 70 runners")
    validate_cross_source_universe(
        targets_manifest, route_cards, targets, entries, history, coverage,
        readiness, route_coverage, official_snapshot,
    )

    timestamp = parse_iso_datetime(generated_at_jst).isoformat(timespec="seconds")
    if parse_iso_datetime(timestamp) < parse_iso_datetime(str(official_snapshot["fetched_at_jst"])):
        raise LitePlusError("observation generation time precedes official entry capture")
    official = flatten_official(official_snapshot)
    declared = {(row["race_id"], row["horse_id"]): row for row in entries}
    if set(official) != set(declared):
        raise LitePlusError("official and declared runner universes differ")
    history_by_horse: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in history:
        history_by_horse[(row["target_race_id"], row["target_horse_id"])].append(dict(row))
    readiness_map = {(row["target_race_id"], row["target_horse_id"]): row for row in readiness}
    route_map = {(row["target_race_id"], row["target_horse_id"]): row for row in route_coverage}
    coverage_map = {row["target_race_id"]: row for row in coverage}
    card_map = {int(card["leg"]): card for card in route_cards["cards"]}

    races: list[dict[str, Any]] = []
    for target in sorted(targets, key=lambda row: int(row["manifest_leg"])):
        leg = int(target["manifest_leg"])
        race_id = target["race_id"]
        card = card_map[leg]
        cov = coverage_map[race_id]
        runner_rows = [official[key] for key in official if key[0] == race_id]
        runner_rows.sort(key=lambda row: row["horse_no"])
        horses = [
            build_horse(
                runner,
                history_by_horse[(race_id, runner["horse_id"])],
                readiness_map[(race_id, runner["horse_id"])],
                route_map[(race_id, runner["horse_id"])],
                {
                    "venue": target["venue"],
                    "surface": normalize_surface(target["surface_manifest"]),
                    "distance_m": int(target["distance_m"]),
                    "post_time_jst": target["post_time_jst"],
                },
                card,
            )
            for runner in runner_rows
        ]
        attach_queue_context(horses)
        role_counts = Counter(horse["role_position"]["base_role"] for horse in horses)
        lead_frequency_names = [
            horse["basics"]["horse_name"] for horse in horses
            if horse["role_position"]["lead_frequency_proxy"] == "high"
        ]
        flexible_names = [
            horse["basics"]["horse_name"] for horse in horses
            if horse["role_position"]["position_flexibility"] in {"high", "medium"}
        ]
        comparison = [comparison_row(horse, card) for horse in horses]
        races.append({
            "leg": leg,
            "win5_role": target["win5_role"],
            "race_id": race_id,
            "race_name": target["race_name"],
            "track": target["venue"],
            "race_no": int(target["race_no"]),
            "surface": normalize_surface(target["surface_manifest"]),
            "distance_m": int(target["distance_m"]),
            "post_time_jst": target["post_time_jst"],
            "runner_count": len(horses),
            "official_entry_url": target["official_entry_url"],
            "draw_status": "confirmed",
            "meeting_day": card["current_context"].get("meeting"),
            "course_change": card["current_context"].get("course_change"),
            "track_condition": unobserved("開催当日のJRA馬場状態をfreeze前に人が記録"),
            "day_bias": unobserved("日曜前半レースの観測前"),
            "wind": unobserved("当日風向・風速を接続していない"),
            "course_structure": route_summary(card),
            "geometry_status": cov["full_curve_geometry_status"],
            "route_match_level": "partial",
            "same_condition_evidence": {
                "exact_count": int(cov["certified_exact_races"]),
                "partial_count": int(cov["partial_candidate_races"]),
                "similar_count": int(cov["certified_similar_races"]),
                "status": "partial",
                "exact_caveat": cov["exact_evaluation_status"],
                "similar_caveat": cov["similar_status"],
                "population_definition": cov["history_population_definition"],
            },
            "ra_lap_join_coverage": {
                "joined": int(cov["ra_lap_joined_races"]),
                "denominator": int(cov["ra_lap_join_denominator"]),
                "rate": float(cov["ra_lap_join_rate"]),
                "status": "observed",
                "note": "race-level RA join。個体別sectionalではない",
            },
            "role_classification_coverage": {
                "known_runner_rows": int(cov["role_known_runner_rows"]),
                "denominator": int(cov["role_runner_denominator"]),
                "rate": float(cov["role_determination_rate"]),
                "status": "proxy",
                "note": "history route populationの位置role coverage。Lite+個体roleは角位置proxy",
            },
            "historical_pace_tendency": {
                "slow_middle_fast": unobserved("既存actual_lap_modeはpaceとshapeを混在し、独立S/M/F taxonomy未凍結"),
                "legacy_shape_distribution": unobserved(
                    "market-excluded recovery coverageには分布列がなく、件数だけから構成比を作らない",
                    "旧taxonomyのlabel coverageはあるが、slow/fast/sustain/instant分布はLite+入力に未収録",
                ),
                "front_pressure": unobserved("pressure duration/先行圧thresholdはNOT_ACTIVATED"),
                "finish_environment": evidence(
                    card["segments"][-1]["focus"], status="proxy", evidence_count=1,
                    confidence="medium", route_match_level="partial",
                    note="route requirement cardの最終segment。過去finish分類ではない",
                ),
            },
            "queue_pace_points": {
                "role_counts": {role: role_counts.get(role, 0) for role in ROLE_VALUES},
                "need_lead_horses": [],
                "lead_frequency_high_horses": lead_frequency_names,
                "summary": f"lead/front proxy {role_counts.get('lead', 0) + role_counts.get('front', 0)}頭、先頭頻度high {len(lead_frequency_names)}頭。need-leadは全頭未観測。公式馬番順、順位なし",
                "status": "proxy",
                "confidence": "medium" if sum(role_counts.values()) - role_counts.get("unobserved", 0) >= len(horses) * 0.8 else "low",
                "notes": "包まれ・外々・位置確保コスト・ハナ失敗は各馬cardに定性表示",
            },
            "human_scenario": race_human_template({"post_time_jst": target["post_time_jst"]}),
            "favorite_collapse": {
                "likely_favorite_or_main_danger_horse": "未入力（市場人気を自動取得しない）",
                "what_fails": "人が指定したmain danger horseの役割前提・位置確保・終盤残存のどれが崩れるかをfreeze理由に記入",
                "race_state_change": "先行依存なら前圧増加/ハナ失敗、差し依存なら仕掛け遅れ/trafficを確認",
                "who_benefits": "役割柔軟性proxyあり（公式馬番順・非順位）: " + ("、".join(flexible_names) if flexible_names else "未観測"),
                "confidence": "low",
                "caveat": "favorite/popularity/odds未使用。馬の推薦ではなく崩壊時の構造観測欄",
            },
            "notes_for_post_race_review": "HTMLのpost-race review欄へ別記。pre-race freezeは上書きしない",
            "comparison_columns": list(COMPARISON_COLUMNS),
            "comparison": comparison,
            "horses": horses,
        })

    result = {
        "schema_version": "race_intelligence_lite_plus_sunday_observation_v1",
        "title": "Race Intelligence Lite+ Sunday Observation View",
        "target_date": TARGET_DATE.isoformat(),
        "generated_at_jst": timestamp,
        "display_order_rule": "race leg then official horse_no; not a rank",
        "safety": dict(SAFETY),
        "data_contract": {
            "evidence_status": sorted(STATUS_VALUES),
            "route_match_level": sorted(ROUTE_MATCH_VALUES),
            "confidence": sorted(CONFIDENCE_VALUES),
            "scenario_sensitivity": sorted(SENSITIVITY_VALUES),
            "missingness_rule": "unobserved is never coerced to neutral or zero",
        },
        "race_count": len(races),
        "runner_count": sum(race["runner_count"] for race in races),
        "races": races,
    }
    validate_observation(result)
    return result


def walk_evidence(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"value", "status", "route_match_level", "evidence_count", "confidence", "note", "missing_reason"} <= set(value):
            yield value
        for child in value.values():
            yield from walk_evidence(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_evidence(child)


def validate_observation(data: Mapping[str, Any]) -> None:
    require_exact_keys(data, OBSERVATION_TOP_KEYS, "observation")
    reject_forbidden_recursive(data, "observation")
    if data.get("safety") != SAFETY:
        raise LitePlusError("safety constants differ")
    races = data.get("races", [])
    if len(races) != 5 or data.get("race_count") != 5:
        raise LitePlusError("observation must contain five races")
    if sum(len(race.get("horses", [])) for race in races) != 70 or data.get("runner_count") != 70:
        raise LitePlusError("observation must contain all 70 horse cards")
    race_ids: set[str] = set()
    horse_keys: set[tuple[str, str]] = set()
    for race_index, race in enumerate(races):
        require_exact_keys(race, OBSERVATION_RACE_KEYS, f"observation.races[{race_index}]")
        race_id = race["race_id"]
        if race_id in race_ids:
            raise LitePlusError(f"duplicate race: {race_id}")
        race_ids.add(race_id)
        horses = race["horses"]
        if len(horses) != len(race["comparison"]) or len(horses) != race["runner_count"]:
            raise LitePlusError(f"card/comparison parity failed: {race_id}")
        for row_index, row in enumerate(race["comparison"]):
            require_exact_keys(row, set(COMPARISON_COLUMNS), f"observation.races[{race_index}].comparison[{row_index}]")
        numbers = [horse["basics"]["horse_no"] for horse in horses]
        if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
            raise LitePlusError(f"official horse number order invalid: {race_id}")
        for horse_index, horse in enumerate(horses):
            context = f"observation.races[{race_index}].horses[{horse_index}]"
            require_exact_keys(horse, OBSERVATION_HORSE_KEYS, context)
            require_exact_keys(horse["basics"], BASICS_KEYS, context + ".basics")
            require_exact_keys(horse["role_position"], ROLE_POSITION_KEYS, context + ".role_position")
            require_exact_keys(horse["ability"], ABILITY_KEYS, context + ".ability")
            require_exact_keys(horse["pace_role_traits"], PACE_TRAIT_KEYS, context + ".pace_role_traits")
            require_exact_keys(horse["course_shape_traits"], COURSE_TRAIT_KEYS, context + ".course_shape_traits")
            require_exact_keys(horse["condition"], CONDITION_KEYS, context + ".condition")
            require_exact_keys(horse["paths"], PATH_KEYS, context + ".paths")
            require_exact_keys(horse["uncertainty"], UNCERTAINTY_KEYS, context + ".uncertainty")
            key = (race_id, horse["horse_id"])
            if key in horse_keys:
                raise LitePlusError(f"duplicate horse card: {key}")
            horse_keys.add(key)
            required_blocks = {"basics", "role_position", "ability", "pace_role_traits",
                               "course_shape_traits", "condition", "transfer_evidence",
                               "scenario_sensitivity", "paths", "uncertainty"}
            if not required_blocks <= set(horse):
                raise LitePlusError(f"missing horse blocks: {key}")
            if len(horse["transfer_evidence"]) > 3:
                raise LitePlusError(f"too many transfer races: {key}")
            for transfer_index, transfer in enumerate(horse["transfer_evidence"]):
                require_exact_keys(transfer, TRANSFER_KEYS, context + f".transfer_evidence[{transfer_index}]")
                if date.fromisoformat(transfer["date"]) >= TARGET_DATE:
                    raise LitePlusError(f"post-target transfer evidence: {key}")
            if set(horse["scenario_sensitivity"]) != set(SCENARIO_VALUES):
                raise LitePlusError(f"scenario sensitivity keys invalid: {key}")
            if any(item["assessment"] not in SENSITIVITY_VALUES for item in horse["scenario_sensitivity"].values()):
                raise LitePlusError(f"scenario sensitivity value invalid: {key}")
            for scenario, item in horse["scenario_sensitivity"].items():
                require_exact_keys(item, SCENARIO_SENSITIVITY_KEYS, context + f".scenario_sensitivity.{scenario}")
        if tuple(race["comparison_columns"]) != COMPARISON_COLUMNS:
            raise LitePlusError(f"comparison columns invalid: {race_id}")
    for item in walk_evidence(data):
        require_exact_keys(item, EVIDENCE_KEYS, "observation.evidence")
        if item["status"] not in STATUS_VALUES or item["confidence"] not in CONFIDENCE_VALUES:
            raise LitePlusError("invalid evidence vocabulary")
        if item["route_match_level"] not in ROUTE_MATCH_VALUES:
            raise LitePlusError("invalid route match vocabulary")
        if item["status"] == "unobserved" and not item["missing_reason"]:
            raise LitePlusError("unobserved evidence without missing reason")


def esc(value: Any) -> str:
    return html_module.escape(str(value), quote=True)


def render_evidence_item(label: str, item: Mapping[str, Any]) -> str:
    details = []
    if item.get("note"):
        details.append(str(item["note"]))
    if item.get("missing_reason"):
        details.append("missing: " + str(item["missing_reason"]))
    return (
        '<div class="evidence-item">'
        f'<div class="evidence-label">{esc(label)}</div>'
        f'<div class="evidence-value">{esc(item["value"])}</div>'
        '<div class="badges">'
        f'<span class="badge status-{esc(item["status"])}">{esc(item["status"])}</span>'
        f'<span class="badge">route {esc(item["route_match_level"])}</span>'
        f'<span class="badge">n={esc(item["evidence_count"])}</span>'
        f'<span class="badge">conf {esc(item["confidence"])}</span>'
        '</div>'
        f'<div class="evidence-note">{esc(" / ".join(details))}</div>'
        '</div>'
    )


def render_trait_grid(values: Mapping[str, Mapping[str, Any]]) -> str:
    ordered = [key for key in TRAIT_LABELS if key in values]
    ordered.extend(sorted(set(values) - set(ordered)))
    return '<div class="evidence-grid">' + ''.join(
        render_evidence_item(TRAIT_LABELS.get(key, key), values[key]) for key in ordered
    ) + '</div>'


def render_transfer(transfers: Sequence[Mapping[str, Any]]) -> str:
    if not transfers:
        return '<p class="missing">過去走なし。transfer evidenceは未観測。</p>'
    rows = []
    for transfer in transfers:
        rows.append(
            '<tr>'
            f'<td>{esc(transfer["date"])}<br><small>{esc(transfer["history_race_id"])}</small></td>'
            f'<td>{esc(transfer["venue_distance"])}</td>'
            f'<td>{esc(transfer["observed_race_shape"]["value"])}<br><small>{esc(transfer["observed_race_shape"]["missing_reason"])}</small></td>'
            f'<td>{esc(transfer["observed_role"]["value"])}<br><small>{esc(transfer["observed_role"]["note"] or transfer["observed_role"]["missing_reason"])}</small></td>'
            f'<td>{esc(transfer["what_was_demanded"]["value"])}</td>'
            f'<td>{esc(transfer["how_the_horse_responded"]["value"])}</td>'
            f'<td>{esc(transfer["why_it_transfers"]["value"])}</td>'
            f'<td>{esc(transfer["evidence_strength"])}<br><small>history route {esc(transfer["history_route_context_level"])}</small></td>'
            '</tr>'
        )
    return (
        '<div class="table-wrap"><table class="detail-table"><thead><tr>'
        '<th>date</th><th>venue / distance</th><th>observed race shape</th><th>observed role</th><th>what demanded</th>'
        '<th>response</th><th>why transfer</th><th>strength</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table></div>'
    )


def render_sensitivity(values: Mapping[str, Mapping[str, Any]]) -> str:
    return '<div class="scenario-grid">' + ''.join(
        f'<div class="scenario-card"><b>{esc(key)}</b><span>{esc(values[key]["assessment"])}</span>'
        f'<small>{esc(values[key]["evidence_note"])}</small></div>' for key in SCENARIO_VALUES
    ) + '</div>'


def render_horse(horse: Mapping[str, Any]) -> str:
    basics = horse["basics"]
    role = horse["role_position"]
    ability = horse["ability"]
    pressure = "、".join(
        f'{item["horse_no"]} {item["horse_name"]}({item["role"]})' for item in role["pressure_rivals"]
    ) or "なし（role proxy上）"
    ability_items = ''.join(
        render_evidence_item(label, ability[key])
        for key, label in (("recent_standard", "recent standard"), ("demonstrated_upper", "demonstrated upper"),
                           ("lower_band", "lower band"), ("repeatability", "repeatability"))
    )
    return f'''
<details class="horse-card" id="horse-{esc(horse['horse_id'])}">
  <summary>
    <span class="horse-number frame-{esc(basics['frame_no'])}">{esc(basics['horse_no'])}</span>
    <span class="horse-title"><b>{esc(basics['horse_name'])}</b><small>{esc(basics['sex'])}{esc(basics['age'])} / {esc(basics['assigned_weight'])}kg / {esc(basics['jockey'])}</small></span>
    <span class="summary-chip">role {esc(role['base_role'])}</span>
    <span class="summary-chip">ability {esc(ability['ability_band'])}</span>
    <span class="summary-chip">conf {esc(horse['uncertainty']['confidence'])}</span>
  </summary>
  <div class="horse-body">
    <section><h4>Basics</h4><div class="facts">
      <span>枠 {esc(basics['frame_no'])}</span><span>馬番 {esc(basics['horse_no'])}</span>
      <span>騎手 {esc(basics['jockey'])}</span><span>調教師 {esc(basics['trainer'])} ({esc(basics['trainer_affiliation'])})</span>
      <span>{esc(basics['sex'])}{esc(basics['age'])} / {esc(basics['coat_color'])}</span><span>斤量 {esc(basics['assigned_weight'])}kg</span>
    </div></section>
    <section><h4>Role / Position</h4><div class="facts">
      <span>base role: {esc(role['base_role'])}</span><span>flexibility: {esc(role['position_flexibility'])}</span>
      <span>need lead: {esc(role['need_lead'])}</span><span>lead frequency proxy: {esc(role['lead_frequency_proxy'])}</span><span>can rate: {esc(role['can_rate'])}</span>
      <span>likely early: {esc(role['likely_early_band'])}</span><span>queue confidence: {esc(role['queue_confidence'])}</span>
    </div><p><b>pressure rivals:</b> {esc(pressure)}</p>
      <p><b>内隣:</b> {esc(role['inside_neighbor_context'])} / <b>外隣:</b> {esc(role['outside_neighbor_context'])}</p>
      <ul>{''.join(f'<li>{esc(note)}</li>' for note in role['queue_notes'])}</ul>
    </section>
    <section><h4>Ability <small>過去着順帯。順位主張ではありません</small></h4><div class="evidence-grid">{ability_items}</div></section>
    <section><h4>Pace / Role Traits</h4>{render_trait_grid(horse['pace_role_traits'])}</section>
    <section><h4>Course / Shape Traits</h4>{render_trait_grid(horse['course_shape_traits'])}</section>
    <section><h4>Condition</h4><div class="evidence-grid">
      {render_evidence_item('workout relative', horse['condition']['workout_relative'])}
      {render_evidence_item('condition trend', horse['condition']['condition_trend'])}
      {render_evidence_item('rest pattern', horse['condition']['rest_pattern'])}
      {render_evidence_item('fatigue rebound proxy', horse['condition']['fatigue_rebound_proxy'])}
    </div><p>state confidence: <b>{esc(horse['condition']['state_confidence'])}</b></p></section>
    <section><h4>Transfer Evidence <small>最大3走</small></h4>{render_transfer(horse['transfer_evidence'])}</section>
    <section><h4>Scenario Sensitivity</h4>{render_sensitivity(horse['scenario_sensitivity'])}</section>
    <section class="paths"><h4>Win / Lose Path</h4>
      <div><b>win path</b><p>{esc(horse['paths']['win_path'])}</p></div>
      <div><b>lose path</b><p>{esc(horse['paths']['lose_path'])}</p></div>
      <div><b>favorite-collapse replacement</b><p>{esc(horse['paths']['favorite_collapse_replacement_path'])}</p></div>
      <div><b>fragility</b><p>{esc(horse['paths']['fragility_note'])}</p></div>
    </section>
    <section class="uncertainty"><h4>Uncertainty</h4>
      <p><b>states:</b> {esc(', '.join(horse['uncertainty']['evidence_states']))} / <b>route:</b> {esc(horse['uncertainty']['route_match_level'])} / <b>overall confidence:</b> {esc(horse['uncertainty']['confidence'])} / <b>history coverage:</b> {esc(horse['uncertainty']['history_coverage_confidence'])}</p>
      <p><b>missing:</b> {esc(horse['uncertainty']['missing_reason'])}</p>
      <p><b>remains unclear:</b> {esc(horse['uncertainty']['what_remains_unclear'])}</p>
    </section>
  </div>
</details>'''


def render_comparison(race: Mapping[str, Any]) -> str:
    headers = ''.join(f'<th>{esc(column)}</th>' for column in race["comparison_columns"])
    rows = ''.join(
        '<tr>' + ''.join(f'<td>{esc(row[column])}</td>' for column in race["comparison_columns"]) + '</tr>'
        for row in race["comparison"]
    )
    return f'<div class="table-wrap"><table class="compare-table"><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>'


def render_race(race: Mapping[str, Any]) -> str:
    course = race["course_structure"]
    same = race["same_condition_evidence"]
    ra = race["ra_lap_join_coverage"]
    role = race["role_classification_coverage"]
    legacy = race["historical_pace_tendency"]["legacy_shape_distribution"]
    scenario_options = '<option value="">選択してください</option><option>SLOW</option><option>MIDDLE</option><option>FAST</option>'
    confidence_options = '<option value="">選択してください</option><option>high</option><option>medium</option><option>low</option>'
    horses = ''.join(render_horse(horse) for horse in race["horses"])
    return f'''
<article class="race" id="race-{esc(race['leg'])}" data-race-id="{esc(race['race_id'])}" data-post-time="{esc(race['human_scenario']['post_time_jst'])}">
  <header class="race-header">
    <div><span class="leg">WIN{esc(race['leg'])}</span><h2>{esc(race['track'])}{esc(race['race_no'])}R {esc(race['race_name'])}</h2></div>
    <div class="race-meta">{esc(race['surface'])} {esc(race['distance_m'])}m · {esc(race['post_time_jst'])} · {esc(race['runner_count'])}頭</div>
  </header>
  <section class="race-board">
    <div class="panel span-2"><h3>Course Structure</h3>
      {render_evidence_item('course geometry summary', course['summary'])}
      {render_evidence_item('first-corner burden', course['first_corner_burden'])}
      {render_evidence_item('slope / straight / corner', course['slope_straight_corner'])}
      <p><b>meeting:</b> {esc(race['meeting_day'])} / <b>course change:</b> {esc(race['course_change'])}</p>
      <p><b>geometry status:</b> {esc(race['geometry_status'])}</p>
    </div>
    <div class="panel"><h3>Same-condition Evidence</h3>
      <div class="count-grid"><div><b>{esc(same['exact_count'])}</b><span>Exact*</span></div><div><b>{esc(same['partial_count'])}</b><span>Partial</span></div><div><b>{esc(same['similar_count'])}</b><span>Similar*</span></div></div>
      <p class="caveat">*Exact 0はversion未判定、Similar 0はmetric未凍結。該当なしの意味ではありません。</p>
      <p>RA lap join: <b>{esc(ra['joined'])}/{esc(ra['denominator'])}</b> ({esc(ra['status'])})</p>
      <p>role coverage: <b>{esc(role['known_runner_rows'])}/{esc(role['denominator'])}</b> ({esc(role['status'])})</p>
    </div>
    <div class="panel"><h3>Pace / Finish Environment</h3>
      {render_evidence_item('historical S/M/F', race['historical_pace_tendency']['slow_middle_fast'])}
      {render_evidence_item('legacy mixed shape', legacy)}
      {render_evidence_item('front pressure', race['historical_pace_tendency']['front_pressure'])}
      {render_evidence_item('finish environment', race['historical_pace_tendency']['finish_environment'])}
    </div>
    <div class="panel"><h3>Queue / Pace Points</h3>
      <p>{esc(race['queue_pace_points']['summary'])}</p>
      <p><b>confidence:</b> {esc(race['queue_pace_points']['confidence'])} · <b>status:</b> proxy</p>
      <p>{esc(race['queue_pace_points']['notes'])}</p>
    </div>
    <div class="panel"><h3>Race-day Missingness</h3>
      {render_evidence_item('track condition', race['track_condition'])}
      {render_evidence_item('day bias', race['day_bias'])}
      {render_evidence_item('wind', race['wind'])}
    </div>
    <div class="panel span-2 collapse"><h3>Favorite Collapse — Human designation only</h3>
      <p><b>likely favorite / main danger:</b> {esc(race['favorite_collapse']['likely_favorite_or_main_danger_horse'])}</p>
      <p><b>what fails:</b> {esc(race['favorite_collapse']['what_fails'])}</p>
      <p><b>race state change:</b> {esc(race['favorite_collapse']['race_state_change'])}</p>
      <p><b>who benefits:</b> {esc(race['favorite_collapse']['who_benefits'])}</p>
      <p><b>confidence:</b> {esc(race['favorite_collapse']['confidence'])} / {esc(race['favorite_collapse']['caveat'])}</p>
    </div>
  </section>

  <section class="freeze-block">
    <div class="freeze-title"><h3>Human Scenario / Pre-race Freeze</h3><span class="unfrozen">UNFROZEN</span></div>
    <p>ここはHuman入力欄です。Lite+はscenarioを自動決定しません。5レース記入後にJSONをexportし、CLIで発走前freezeしてください。</p>
    <div class="form-grid">
      <label>Main scenario<select data-freeze-field="main_scenario">{scenario_options}</select></label>
      <label>Confidence<select data-freeze-field="confidence">{confidence_options}</select></label>
      <label class="wide">Alternative scenario(s)<input data-freeze-field="alternative_scenarios" placeholder="例: FAST, SLOW"></label>
      <label class="wide">Reason<textarea data-freeze-field="reason" rows="3" placeholder="観測根拠と反証条件"></textarea></label>
      <label>Likely favorite / main danger (human)<input data-freeze-field="main_danger_horse" placeholder="任意。人気自動取得なし"></label>
      <label>Freeze timestamp<input value="CLIが付与" readonly></label>
      <label class="wide">Race-day notes<textarea data-freeze-field="race_day_notes" rows="3" placeholder="馬場・風・取消・当日bias"></textarea></label>
      <label class="wide post-review">Post-race review (別artifact)<textarea data-review-field="review" rows="3" placeholder="結果後に記入。pre-race freezeを変更しない"></textarea></label>
    </div>
  </section>

  <section><div class="section-heading"><h3>全頭比較テーブル</h3><span>公式馬番順 / non-ranking</span></div>{render_comparison(race)}</section>
  <section><div class="section-heading"><h3>Horse Cards — 全{esc(race['runner_count'])}頭</h3><span>クリックで展開</span></div>{horses}</section>
</article>'''


CSS = r'''
:root{--bg:#0b1213;--panel:#121d1e;--panel2:#172526;--ink:#edf5f0;--muted:#9eb1ab;--line:#2a3e3c;--mint:#86d6b1;--amber:#f2c66d;--red:#ef8d7e;--blue:#8ebed4}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 15% 0,#18302b 0,#0b1213 35%);color:var(--ink);font-family:Inter,"Noto Sans JP","Yu Gothic UI",sans-serif;line-height:1.65}
a{color:var(--mint)}.shell{max-width:1540px;margin:auto;padding:24px}.hero{border:1px solid #315349;border-radius:24px;padding:30px;background:linear-gradient(135deg,rgba(28,69,56,.95),rgba(11,18,19,.95));box-shadow:0 24px 70px #0008}.eyebrow{color:var(--mint);font-weight:800;letter-spacing:.12em;text-transform:uppercase}.hero h1{font-size:clamp(30px,5vw,64px);line-height:1.05;margin:.2em 0}.hero p{max-width:900px;color:#d3e3dc}.safety-strip{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}.safety-strip span{border:1px solid #87d6b177;color:#c9f1df;padding:6px 11px;border-radius:999px;font-size:13px;font-weight:700}.freeze-warning{margin:20px 0;padding:16px 20px;border:1px solid #d19f44;background:#3a2b11;border-radius:14px;color:#ffe2a6;font-weight:700}.nav{display:flex;gap:8px;overflow:auto;position:sticky;top:0;z-index:20;background:#0b1213ee;padding:12px 0}.nav a{white-space:nowrap;text-decoration:none;background:var(--panel);border:1px solid var(--line);padding:8px 13px;border-radius:10px}.race{margin:38px 0 70px}.race-header{display:flex;align-items:end;justify-content:space-between;gap:18px;border-bottom:2px solid var(--mint);padding:0 2px 14px}.race-header h2{margin:4px 0 0;font-size:clamp(25px,3vw,40px)}.leg{color:var(--mint);font-weight:900;letter-spacing:.14em}.race-meta{color:var(--muted);font-weight:700}.race-board{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:18px 0}.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px}.panel h3,.freeze-block h3{margin-top:0}.span-2{grid-column:span 2}.count-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.count-grid div{background:#0c1516;border-radius:12px;padding:13px;text-align:center}.count-grid b{display:block;font-size:28px}.count-grid span{color:var(--muted)}.caveat,.missing{color:var(--amber)}.evidence-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:9px}.evidence-item{padding:11px;background:#0d1718;border:1px solid #223435;border-radius:10px}.evidence-label{text-transform:uppercase;font-size:11px;letter-spacing:.08em;color:var(--muted)}.evidence-value{font-weight:750;margin:4px 0}.badges{display:flex;gap:4px;flex-wrap:wrap}.badge{font-size:10px;border:1px solid var(--line);border-radius:999px;padding:2px 6px;color:var(--muted)}.status-observed{color:var(--mint);border-color:#4e9d7a}.status-derived{color:var(--blue);border-color:#487f98}.status-proxy{color:var(--amber);border-color:#a67b31}.status-unobserved{color:var(--red);border-color:#9b4f45}.evidence-note{font-size:11px;color:var(--muted);margin-top:7px}.freeze-block{background:linear-gradient(135deg,#252210,#171a17);border:1px solid #9e7c38;border-radius:18px;padding:20px;margin:18px 0}.freeze-title,.section-heading{display:flex;align-items:center;justify-content:space-between;gap:12px}.unfrozen{background:#6a4818;color:#ffe0a0;border-radius:999px;padding:5px 10px;font-size:12px;font-weight:900}.form-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.form-grid label{display:flex;flex-direction:column;gap:5px;color:#d8e2de;font-size:13px;font-weight:700}.form-grid .wide{grid-column:span 2}.form-grid input,.form-grid select,.form-grid textarea{width:100%;border-radius:9px;border:1px solid #495c57;background:#0b1213;color:var(--ink);padding:10px;font:inherit}.post-review{border-top:1px dashed #5c6c68;padding-top:12px}.action-bar{position:sticky;bottom:12px;z-index:30;display:flex;gap:10px;justify-content:center;margin:28px auto;padding:10px;border-radius:16px;background:#0b1213e8;border:1px solid var(--line);width:max-content;max-width:100%}.action-bar button{border:0;border-radius:10px;padding:11px 16px;background:var(--mint);color:#082018;font-weight:900;cursor:pointer}.action-bar button.secondary{background:#263838;color:var(--ink)}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:14px}.compare-table,.detail-table{border-collapse:collapse;width:100%;min-width:1250px;background:var(--panel)}th,td{padding:10px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left;font-size:12px}th{position:sticky;top:0;background:#1c2c2b;color:var(--mint);z-index:2}.horse-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin:9px 0;overflow:hidden}.horse-card summary{display:flex;align-items:center;gap:12px;padding:14px;cursor:pointer;list-style:none}.horse-card summary::-webkit-details-marker{display:none}.horse-card[open]{border-color:#4f786b}.horse-number{display:grid;place-items:center;width:38px;height:38px;border-radius:8px;background:#dde5e2;color:#07100e;font-size:20px;font-weight:900}.horse-title{display:flex;flex-direction:column;min-width:230px;flex:1}.horse-title b{font-size:17px}.horse-title small{color:var(--muted)}.summary-chip{border:1px solid var(--line);padding:4px 8px;border-radius:999px;font-size:11px;color:var(--muted)}.horse-body{padding:0 16px 18px}.horse-body section{border-top:1px solid var(--line);padding-top:13px;margin-top:13px}.horse-body h4{margin:0 0 10px;color:var(--mint);font-size:17px}.horse-body h4 small{color:var(--muted);font-weight:400}.facts{display:flex;gap:8px;flex-wrap:wrap}.facts span{background:#0d1718;border-radius:8px;padding:6px 9px}.scenario-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.scenario-card{display:flex;flex-direction:column;background:#0d1718;border-radius:10px;padding:12px}.scenario-card span{color:var(--red);font-weight:800}.scenario-card small{color:var(--muted)}.paths{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.paths h4{grid-column:span 2}.paths>div{background:#0d1718;border-radius:10px;padding:12px}.uncertainty{background:#231c16;padding:14px!important;border-radius:12px}.method{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:40px}.footer{color:var(--muted);padding:30px 0;text-align:center}
@media(max-width:820px){.shell{padding:13px}.race-header{align-items:start;flex-direction:column}.race-board{grid-template-columns:1fr}.span-2{grid-column:span 1}.form-grid{grid-template-columns:1fr}.form-grid .wide{grid-column:span 1}.summary-chip{display:none}.horse-title{min-width:0}.paths{grid-template-columns:1fr}.paths h4{grid-column:span 1}.scenario-grid{grid-template-columns:1fr}.action-bar{width:100%;overflow:auto;justify-content:flex-start}}
'''


JS = r'''
function downloadJson(name, value) {
  const blob = new Blob([JSON.stringify(value, null, 2) + "\n"], {type: "application/json"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}
function collectFreeze() {
  const overrides = [];
  let invalid = false;
  document.querySelectorAll("article.race").forEach((race) => {
    const get = (field) => race.querySelector(`[data-freeze-field="${field}"]`).value.trim();
    const main = get("main_scenario");
    const confidence = get("confidence");
    const reason = get("reason");
    if (!main || !confidence || !reason) invalid = true;
    overrides.push({
      race_id: race.dataset.raceId,
      post_time_jst: race.dataset.postTime,
      main_scenario: main,
      alternative_scenarios: get("alternative_scenarios").split(",").map(x => x.trim()).filter(Boolean),
      confidence: confidence,
      reason: reason,
      race_day_notes: get("race_day_notes"),
      main_danger_horse: get("main_danger_horse")
    });
  });
  if (invalid) { alert("全5レースのMain scenario / Confidence / Reasonを記入してください。"); return; }
  downloadJson("human_scenario_freeze.input.json", {
    schema_version: "race_intelligence_lite_plus_human_freeze_input_v1",
    target_date: "2026-08-23",
    observation_sha256: document.body.dataset.observationSha256,
    safety: {direct_horse_score_edit: false, ranking_claims: false, probability_claims: false},
    overrides: overrides
  });
}
function collectReview() {
  const freezeSha = prompt("freeze_manifest.json のSHA-256を64桁で入力してください");
  if (!freezeSha || !/^[0-9a-f]{64}$/i.test(freezeSha)) {
    alert("post-race reviewにはfreeze manifest SHA-256が必要です。");
    return;
  }
  const reviews = [];
  document.querySelectorAll("article.race").forEach((race) => {
    reviews.push({race_id: race.dataset.raceId, review: race.querySelector("[data-review-field=review]").value.trim()});
  });
  downloadJson("post_race_review.input.json", {
    schema_version: "race_intelligence_lite_plus_post_race_review_v1",
    target_date: "2026-08-23",
    observation_sha256: document.body.dataset.observationSha256,
    freeze_manifest_sha256: freezeSha.toLowerCase(),
    recorded_at_browser_iso: new Date().toISOString(),
    reviews: reviews
  });
}
function expandAll(open) { document.querySelectorAll("details.horse-card").forEach(x => x.open = open); }
'''


def render_html(data: Mapping[str, Any], observation_sha256: str) -> str:
    validate_observation(data)
    nav = ''.join(
        f'<a href="#race-{race["leg"]}">WIN{race["leg"]} {esc(race["race_name"])}</a>' for race in data["races"]
    )
    races = ''.join(render_race(race) for race in data["races"])
    safety = ''.join(
        f'<span>{label}</span>' for label in (
            "Observation-only", "No AI rank", "No probability", "No odds / no stake",
            "No BUY / no order", "Pre-race freeze required",
        )
    )
    return f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(data['title'])}</title><style>{CSS}</style></head>
<body data-observation-sha256="{esc(observation_sha256)}"><main class="shell">
<header class="hero"><div class="eyebrow">Sunday · 2026.08.23 · all 70 runners</div>
<h1>Race Intelligence<br>Lite+</h1>
<p>レース構造・要求能力・各馬の観測証拠・浮上条件・失速条件を読むためのSunday Observation View。公式馬番順で表示し、順位・勝率・市場評価は生成しません。</p>
<div class="safety-strip">{safety}</div></header>
<div class="freeze-warning">Human Scenarioは未凍結です。全5レースを記入し、最初の発走前にCLI freezeを完了してください。</div>
<nav class="nav">{nav}</nav>
{races}
<section class="method"><h2>Evidence boundary</h2>
<p><b>observed</b> は入力に直接ある事実、<b>derived</b> は透明な変換、<b>proxy</b> は代替観測、<b>unobserved</b> は欠損です。unobservedをneutralや0へ埋めません。</p>
<p>Exact=0はroute version未解決、Similar=0は類似metric未凍結を示します。歴史paceの旧ラベルはpaceとshapeを混在するため、SLOW/MIDDLE/FASTへ変換していません。</p>
<p>生成: {esc(data['generated_at_jst'])} / observation SHA-256: <code>{esc(observation_sha256)}</code></p></section>
<div class="action-bar"><button type="button" onclick="collectFreeze()">Human Freeze入力JSONをexport</button><button type="button" class="secondary" onclick="collectReview()">Post-race reviewを別export</button><button type="button" class="secondary" onclick="expandAll(true)">全カード展開</button><button type="button" class="secondary" onclick="expandAll(false)">閉じる</button></div>
<footer class="footer">Race Intelligence Lite+ · read-only observation artifact · no trading hooks</footer>
</main><script>{JS}</script></body></html>'''


def provenance_path(path: Path) -> str:
    resolved = path.resolve()
    normalized = resolved.as_posix()
    for marker, prefix in (("/outputs/analysis/", "outputs/analysis/"), ("/docs/", "docs/")):
        if marker in normalized:
            return prefix + normalized.split(marker, 1)[1]
    return path.name


def source_record(path: Path, kind: str) -> dict[str, Any]:
    return {"kind": kind, "path": provenance_path(path), "sha256": sha256_path(path), "bytes": path.stat().st_size}


def build_source_manifest(paths: Mapping[str, Path], official: Mapping[str, Any], generated_at: str) -> dict[str, Any]:
    records = [source_record(path, kind) for kind, path in sorted(paths.items())]
    return {
        "schema_version": "race_intelligence_lite_plus_source_manifest_v1",
        "generated_at_jst": parse_iso_datetime(generated_at).isoformat(timespec="seconds"),
        "allowlisted_sources": records,
        "official_current_entry": {
            "path": provenance_path(paths["official_current_entry"]),
            "fetched_at_jst": official["fetched_at_jst"],
            "identity_join": official["identity_join"],
            "runner_count": official["runner_count"],
            "raw_pages": [
                {key: race[key] for key in ("leg", "race_id", "official_entry_url", "http_status",
                                             "raw_response_sha256", "raw_response_bytes", "field_boundary")}
                for race in official["races"]
            ],
        },
        "excluded_source_classes": [
            "legacy AI score/rank and derived scenario score/rank",
            "prediction/model output",
            "odds/popularity/market/payoff/ROI/EV/mispricing",
            "BUY/candidate/ticket/stake/Champion/notification/order",
            "training/backtest/outer OOS/EXP-033 run/EXP-034 real-data execution",
        ],
        "safety": dict(SAFETY),
    }


def human_template(data: Mapping[str, Any], observation_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "race_intelligence_lite_plus_human_freeze_input_v1",
        "target_date": data["target_date"],
        "observation_sha256": observation_sha256,
        "safety": {"direct_horse_score_edit": False, "ranking_claims": False, "probability_claims": False},
        "overrides": [
            {
                "race_id": race["race_id"],
                "post_time_jst": race["human_scenario"]["post_time_jst"],
                "main_scenario": "",
                "alternative_scenarios": [],
                "confidence": "",
                "reason": "",
                "race_day_notes": "",
                "main_danger_horse": "",
            }
            for race in data["races"]
        ],
    }


def validate_human_freeze(value: Mapping[str, Any], observation: Mapping[str, Any], now_jst: datetime) -> None:
    require_exact_keys(value, FREEZE_TOP_LEVEL_KEYS, "human_freeze")
    reject_forbidden_recursive(value, "human_freeze")
    if value.get("schema_version") != "race_intelligence_lite_plus_human_freeze_input_v1":
        raise LitePlusError("invalid human freeze schema")
    if value.get("target_date") != observation.get("target_date"):
        raise LitePlusError("human freeze target date differs")
    if now_jst < parse_iso_datetime(str(observation.get("generated_at_jst", ""))):
        raise LitePlusError("freeze clock precedes the observation snapshot")
    observation_bytes = canonical_json_bytes(observation)
    if value.get("observation_sha256") != sha256_bytes(observation_bytes):
        raise LitePlusError("human freeze references a different observation snapshot")
    safety = value.get("safety")
    if not isinstance(safety, Mapping):
        raise LitePlusError("human freeze safety must be an object")
    require_exact_keys(safety, FREEZE_SAFETY_KEYS, "human_freeze.safety")
    if safety != {"direct_horse_score_edit": False, "ranking_claims": False, "probability_claims": False}:
        raise LitePlusError("human freeze safety constants differ")
    race_map = {race["race_id"]: race for race in observation["races"]}
    overrides = value.get("overrides", [])
    if not isinstance(overrides, list):
        raise LitePlusError("human freeze overrides must be a list")
    for index, item in enumerate(overrides):
        if not isinstance(item, Mapping):
            raise LitePlusError(f"human override is not an object: {index}")
        require_exact_keys(item, FREEZE_OVERRIDE_KEYS, f"human_freeze.overrides[{index}]")
    if len(overrides) != 5 or {item.get("race_id") for item in overrides} != set(race_map):
        raise LitePlusError("human freeze must cover exactly all five races")
    for item in overrides:
        if item.get("main_scenario") not in SCENARIO_VALUES:
            raise LitePlusError(f"invalid/missing human scenario: {item.get('race_id')}")
        alternatives = item.get("alternative_scenarios")
        if not isinstance(alternatives, list) or any(value not in SCENARIO_VALUES for value in alternatives):
            raise LitePlusError(f"invalid alternative scenarios: {item.get('race_id')}")
        if item.get("confidence") not in {"high", "medium", "low"}:
            raise LitePlusError(f"invalid/missing human confidence: {item.get('race_id')}")
        if not str(item.get("reason", "")).strip():
            raise LitePlusError(f"human reason required: {item.get('race_id')}")
        expected_post = datetime.fromisoformat(race_map[item["race_id"]]["human_scenario"]["post_time_jst"])
        supplied_post = datetime.fromisoformat(item["post_time_jst"])
        if supplied_post != expected_post:
            raise LitePlusError(f"post time changed: {item.get('race_id')}")
        if now_jst >= expected_post:
            raise LitePlusError(f"freeze is not pre-race: {item.get('race_id')}")


def freeze_human_scenario(
    observation_path: Path,
    human_path: Path,
    output_dir: Path,
    now_jst: datetime | None = None,
) -> dict[str, Any]:
    observation = read_json(observation_path)
    validate_observation(observation)
    human = read_json(human_path)
    if now_jst is None:
        now_jst = datetime.now(JST)
    elif now_jst.tzinfo is None:
        raise LitePlusError("freeze clock must be timezone-aware")
    else:
        now_jst = now_jst.astimezone(JST)
    validate_human_freeze(human, observation, now_jst)
    if output_dir.exists():
        raise LitePlusError(f"freeze output already exists; refusing overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = {
        "schema_version": human["schema_version"],
        "target_date": human["target_date"],
        "observation_sha256": human["observation_sha256"],
        "safety": dict(human["safety"]),
        "overrides": [dict(item) for item in human["overrides"]],
    }
    frozen["freeze_timestamp_jst"] = now_jst.isoformat(timespec="seconds")
    frozen["freeze_status"] = "FROZEN_PRE_RACE"
    frozen["post_race_review_included"] = False
    frozen_path = output_dir / "human_scenario_freeze.json"
    write_json(frozen_path, frozen)
    manifest = {
        "schema_version": "race_intelligence_lite_plus_freeze_manifest_v1",
        "freeze_timestamp_jst": frozen["freeze_timestamp_jst"],
        "immutable_pre_race": True,
        "post_race_review_policy": "separate append-only artifact; never mutate this directory",
        "files": {
            "observation": {"path": observation_path.as_posix(), "sha256": sha256_path(observation_path)},
            "human_input": {"path": human_path.as_posix(), "sha256": sha256_path(human_path)},
            "frozen": {"path": frozen_path.as_posix(), "sha256": sha256_path(frozen_path)},
        },
        "safety": dict(SAFETY),
    }
    write_json(output_dir / "freeze_manifest.json", manifest)
    return manifest


def command_capture(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise LitePlusError(f"output exists; pass --overwrite explicitly: {output}")
    snapshot = fetch_official_snapshot(Path(args.targets), Path(args.entries), args.fetched_at_jst)
    write_json(output, snapshot)
    print(f"captured {snapshot['race_count']} races / {snapshot['runner_count']} runners -> {output}")


def command_build(args: argparse.Namespace) -> None:
    paths = {
        "target_race_name_manifest": Path(args.target_manifest),
        "route_requirement_cards": Path(args.route_cards),
        "resolved_official_targets": Path(args.targets),
        "declared_runner_audit": Path(args.entries),
        "market_excluded_history": Path(args.history),
        "same_condition_coverage": Path(args.coverage),
        "horse_readiness": Path(args.readiness),
        "horse_route_coverage": Path(args.route_coverage),
        "official_current_entry": Path(args.official_entries),
    }
    for path in paths.values():
        if not path.is_file():
            raise LitePlusError(f"input file missing: {path}")
    targets_manifest = read_json(paths["target_race_name_manifest"])
    route_cards = read_json(paths["route_requirement_cards"])
    targets = read_csv(paths["resolved_official_targets"])
    entries = read_csv(paths["declared_runner_audit"])
    history = read_csv(paths["market_excluded_history"])
    coverage = read_csv(paths["same_condition_coverage"])
    readiness = read_csv(paths["horse_readiness"])
    route_coverage = read_csv(paths["horse_route_coverage"])
    official = read_json(paths["official_current_entry"])
    manifest_sha = sha256_path(paths["target_race_name_manifest"])
    if any(row.get("manifest_sha256") != manifest_sha for row in targets):
        raise LitePlusError("resolved targets are not bound to the supplied race-name manifest SHA-256")
    data = build_observation(
        targets_manifest=targets_manifest, route_cards=route_cards, targets=targets,
        entries=entries, history=history, coverage=coverage, readiness=readiness,
        route_coverage=route_coverage, official_snapshot=official,
        generated_at_jst=args.generated_at_jst,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "race_intelligence_lite_plus_data.json"
    html_path = output_dir / "race_intelligence_lite_plus.html"
    template_path = output_dir / "human_scenario_freeze.template.json"
    manifest_path = output_dir / "source_manifest.json"
    for path in (data_path, html_path, template_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise LitePlusError(f"output exists; pass --overwrite explicitly: {path}")
    data_bytes = canonical_json_bytes(data)
    write_text(data_path, data_bytes.decode("utf-8"))
    data_sha = sha256_bytes(data_bytes)
    canonical_data = json.loads(data_bytes.decode("utf-8"))
    write_text(html_path, render_html(canonical_data, data_sha))
    write_json(template_path, human_template(canonical_data, data_sha))
    source_manifest = build_source_manifest(paths, official, args.generated_at_jst)
    source_manifest["outputs"] = {
        "observation_data": {"path": provenance_path(data_path), "sha256": sha256_path(data_path)},
        "html": {"path": provenance_path(html_path), "sha256": sha256_path(html_path)},
        "human_freeze_template": {"path": provenance_path(template_path), "sha256": sha256_path(template_path)},
    }
    write_json(manifest_path, source_manifest)
    print(f"built {data['race_count']} races / {data['runner_count']} horses -> {output_dir}")


def command_verify(args: argparse.Namespace) -> None:
    observation_path = Path(args.observation)
    html_path = Path(args.html)
    source_manifest_path = Path(args.source_manifest)
    official_path = Path(args.official_entries)
    template_path = Path(args.freeze_template)
    data = read_json(observation_path)
    validate_observation(data)
    data_sha = sha256_bytes(canonical_json_bytes(data))
    html_text = html_path.read_text(encoding="utf-8")
    expected_html = render_html(data, data_sha)
    if html_text != expected_html:
        raise LitePlusError("HTML is not the deterministic render of the observation JSON")
    manifest = read_json(source_manifest_path)
    require_exact_keys(manifest, SOURCE_MANIFEST_KEYS, "source_manifest")
    reject_forbidden_recursive(manifest, "source_manifest")
    if manifest.get("schema_version") != "race_intelligence_lite_plus_source_manifest_v1":
        raise LitePlusError("source manifest schema version differs")
    if manifest.get("safety") != SAFETY:
        raise LitePlusError("source manifest safety constants differ")
    records = manifest.get("allowlisted_sources")
    if not isinstance(records, list) or len(records) != 9:
        raise LitePlusError("source manifest must contain nine allowlisted source records")
    for index, record in enumerate(records):
        require_exact_keys(record, SOURCE_RECORD_KEYS, f"source_manifest.allowlisted_sources[{index}]")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])):
            raise LitePlusError(f"source SHA-256 invalid: {index}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise LitePlusError("source manifest outputs must be an object")
    require_exact_keys(outputs, SOURCE_OUTPUT_KEYS, "source_manifest.outputs")
    for name, record in outputs.items():
        require_exact_keys(record, SOURCE_OUTPUT_RECORD_KEYS, f"source_manifest.outputs.{name}")
    if outputs["observation_data"]["sha256"] != sha256_path(observation_path):
        raise LitePlusError("observation SHA-256 differs from source manifest")
    if outputs["html"]["sha256"] != sha256_path(html_path):
        raise LitePlusError("HTML SHA-256 differs from source manifest")
    template = read_json(template_path)
    if template != human_template(data, data_sha):
        raise LitePlusError("Human freeze template is not bound to the observation JSON")
    if outputs["human_freeze_template"]["sha256"] != sha256_path(template_path):
        raise LitePlusError("Human freeze template SHA-256 differs from source manifest")
    official = read_json(official_path)
    validate_official_snapshot_schema(official)
    official_records = [record for record in records if record["kind"] == "official_current_entry"]
    if len(official_records) != 1 or official_records[0]["sha256"] != sha256_path(official_path):
        raise LitePlusError("official current-entry snapshot differs from source manifest")
    print(f"verified {data['race_count']} races / {data['runner_count']} horses")


def command_freeze(args: argparse.Namespace) -> None:
    manifest = freeze_human_scenario(Path(args.observation), Path(args.human_input),
                                     Path(args.output_dir))
    print(f"frozen at {manifest['freeze_timestamp_jst']} -> {args.output_dir}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture-official", help="GET and sanitize current JRA entries")
    capture.add_argument("--targets", required=True)
    capture.add_argument("--entries", required=True)
    capture.add_argument("--fetched-at-jst", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--overwrite", action="store_true")
    capture.set_defaults(func=command_capture)

    build = sub.add_parser("build", help="build deterministic observation JSON/HTML")
    build.add_argument("--target-manifest", required=True)
    build.add_argument("--route-cards", required=True)
    build.add_argument("--targets", required=True)
    build.add_argument("--entries", required=True)
    build.add_argument("--history", required=True)
    build.add_argument("--coverage", required=True)
    build.add_argument("--readiness", required=True)
    build.add_argument("--route-coverage", required=True)
    build.add_argument("--official-entries", required=True)
    build.add_argument("--generated-at-jst", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(func=command_build)

    verify = sub.add_parser("verify", help="validate an existing observation artifact")
    verify.add_argument("--observation", required=True)
    verify.add_argument("--html", required=True)
    verify.add_argument("--source-manifest", required=True)
    verify.add_argument("--official-entries", required=True)
    verify.add_argument("--freeze-template", required=True)
    verify.set_defaults(func=command_verify)

    freeze = sub.add_parser("freeze", help="create an immutable pre-race Human Scenario freeze")
    freeze.add_argument("--observation", required=True)
    freeze.add_argument("--human-input", required=True)
    freeze.add_argument("--output-dir", required=True)
    freeze.set_defaults(func=command_freeze)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.func(args)
    except LitePlusError as exc:
        raise SystemExit(f"FAIL-CLOSED: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
