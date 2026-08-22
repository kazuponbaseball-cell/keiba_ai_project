#!/usr/bin/env python3
"""Build the synthetic-only Decision Summary Layer V0.

This module is deliberately independent from the real Race Intelligence Lite+
builder.  It accepts only a strict, synthetic Horsecard projection and creates a
read-only view model.  It does not train, predict, rank, materialize real data,
or connect to market/trading paths.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "decision_summary_layer_v0"
SOURCE_SCHEMA_VERSION = "decision_summary_layer_v0_synthetic_source"
LAYER_TYPE = "READ_ONLY_CONSUMER_VIEW_MODEL"
REVIEW_SEMANTICS = "REVIEW_PRIORITY_NOT_ABILITY_ORDERING"
RULE_VERSION = "ORDERED_BOOLEAN_PRECEDENCE_V0"

CLASSIFICATIONS = (
    "PRIMARY_REVIEW",
    "CONDITIONAL_REVIEW",
    "FRAGILE",
    "INSUFFICIENT",
)
ROLE_VALUES = ("LEAD", "FRONT", "STALKING", "MIDPACK", "REAR", "UNKNOWN")
ROLE_RANGE_VALUES = ("BROAD", "MULTIPLE", "NARROW", "UNKNOWN")
FLEXIBILITY_VALUES = ("FLEXIBLE", "LIMITED", "UNKNOWN")
FORWARD_VALUES = ("FORWARD", "MIXED", "LOW", "UNKNOWN")
DEPENDENCY_VALUES = ("NO", "POSSIBLE", "UNKNOWN")
BURDEN_VALUES = ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
CONDITION_VALUES = ("SUPPORTIVE", "NEUTRAL", "ADVERSE", "UNKNOWN")
EVIDENCE_STATUS_VALUES = ("OBSERVED", "DERIVED", "PROXY", "UNOBSERVED")
EVIDENCE_SECTIONS = ("ROUTE", "QUEUE", "ROLE", "CONDITION", "TRANSFER", "OTHER")

REASON_CODE_CATALOG = {
    "ABILITY_CLEAN_NOT_AVAILABLE": "Versioned clean ability is not available in Weekend V0.",
    "PRIMARY_ROLE_REACHABLE": "Target role and expected position band are structurally available.",
    "PRIMARY_QUEUE_SUPPORTIVE": "Queue burden is supportive under the fixed V0 rule.",
    "CURRENT_CONDITION_SUPPORTIVE": "Directly observed current-condition evidence is supportive.",
    "CONDITIONAL_MEDIUM_FIRST_TURN_COST": "A medium first-turn position cost must ease.",
    "CONDITIONAL_MEDIUM_PRESSURE_RIVAL": "A medium Queue conflict must ease.",
    "CONDITIONAL_CAN_RATE_PATH_AVAILABLE": "Observed rating flexibility provides a conditional path.",
    "FRAGILE_ROLE_RANGE_NARROW": "A narrow role range and limited flexibility create a failure path.",
    "FRAGILE_LEAD_DEPENDENCY_POSSIBLE": "The evidence indicates lead/position dependency.",
    "FRAGILE_QUEUE_HIGH_CONFLICT": "At least one high Queue conflict is present.",
    "FRAGILE_FIRST_TURN_COST_HIGH": "The first-turn position cost is high.",
    "FRAGILE_CURRENT_CONDITION_ADVERSE": "Direct current-condition evidence is adverse.",
    "INSUFFICIENT_ROLE_CORE": "Role or expected-position evidence is incomplete.",
    "INSUFFICIENT_QUEUE_CONTEXT": "Queue fit cannot be resolved from allowed evidence.",
    "CONDITION_CURRENT_NOT_AVAILABLE": "Current-condition fit is not directly observed.",
}
REASON_CODE_ORDER = tuple(REASON_CODE_CATALOG)

POSITIVE_WORLD_STATES = {
    "WORLD_TARGET_ROLE_REPRODUCED": "今回も対象roleと想定位置帯を再現できる",
    "WORLD_QUEUE_COST_STAYS_LOW": "序盤のQueue負荷が低いまま推移する",
    "WORLD_EARLY_PRESSURE_EASES": "序盤の位置取り圧が緩和される",
    "WORLD_CAN_RATE_PATH_AVAILABLE": "控える選択肢が機能して位置依存を弱める",
    "WORLD_CURRENT_CONDITION_SUPPORT_PERSISTS": "直接観測された状態面の支えが維持される",
}
FAILURE_WORLD_STATES = {
    "FAILURE_ROLE_BAND_MISSED": "必要な位置帯へ入れない",
    "FAILURE_NONLEAD_POSITION_FORCED": "主導権を取れず依存する位置を失う",
    "FAILURE_EARLY_CONFLICT_ESCALATES": "序盤のQueue競合が強まる",
    "FAILURE_FIRST_TURN_COST_REALIZED": "1角までの位置確保コストが顕在化する",
    "FAILURE_CURRENT_CONDITION_FAILS": "状態面の逆風がレース中の反応を制限する",
}
POSITIVE_WORLD_STATE_ORDER = tuple(POSITIVE_WORLD_STATES)
FAILURE_WORLD_STATE_ORDER = tuple(FAILURE_WORLD_STATES)

TRIGGER_EVIDENCE_DIMENSION = {
    "WORLD_TARGET_ROLE_REPRODUCED": "role",
    "WORLD_QUEUE_COST_STAYS_LOW": "queue",
    "WORLD_EARLY_PRESSURE_EASES": "queue",
    "WORLD_CAN_RATE_PATH_AVAILABLE": "role",
    "WORLD_CURRENT_CONDITION_SUPPORT_PERSISTS": "current_condition",
    "FAILURE_ROLE_BAND_MISSED": "role",
    "FAILURE_NONLEAD_POSITION_FORCED": "queue",
    "FAILURE_EARLY_CONFLICT_ESCALATES": "queue",
    "FAILURE_FIRST_TURN_COST_REALIZED": "queue",
    "FAILURE_CURRENT_CONDITION_FAILS": "current_condition",
}

SOURCE_TOP_KEYS = {"schema_version", "synthetic_fixture", "safety", "races"}
SOURCE_SAFETY_KEYS = {
    "real_data_materialized",
    "training_executed",
    "oos_evaluation_executed",
    "target_prediction_generated",
    "formal_buy",
    "send_order",
    "stake",
}
SOURCE_RACE_KEYS = {"display_order", "race_id", "race_name", "runners"}
SOURCE_RUNNER_KEYS = {
    "horse_id",
    "basics",
    "role_queue_evidence",
    "current_condition_evidence",
    "evidence_details",
}
BASICS_KEYS = {
    "horse_name",
    "frame_no",
    "horse_no",
    "jockey",
    "trainer",
    "sex",
    "age",
    "assigned_weight",
}
ROLE_QUEUE_KEYS = {
    "target_context_role",
    "role_range",
    "expected_position",
    "position_flexibility",
    "forward_propensity",
    "lead_dependency",
    "first_turn_position_cost",
    "can_rate",
    "pressure_rivals",
    "role_evidence_ids",
    "queue_evidence_ids",
}
PRESSURE_RIVAL_KEYS = {"horse_no", "conflict_level", "evidence_ids"}
CONDITION_KEYS = {"fit", "status", "evidence_ids"}
EVIDENCE_DETAIL_KEYS = {
    "evidence_id",
    "section",
    "label",
    "value",
    "status",
    "source_note",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
FORBIDDEN_EVIDENCE_TEXT = (
    re.compile(r"\bai[\s_-]*(?:score|rank)\b", re.IGNORECASE),
    re.compile(r"\b(?:rank|gap|score|confidence|odds?|market|popularity|payoff|payout|refund|probability)\b", re.IGNORECASE),
    re.compile(r"\b(?:win|place)[\s_-]*(?:probability|prob|rate)\b", re.IGNORECASE),
    re.compile(r"\bscenario[\s_-]*sensitivity\b", re.IGNORECASE),
    re.compile(r"\bweighted[\s_-]*total\b", re.IGNORECASE),
    re.compile(r"\b(?:buy|stake|champion|notification|order)\b", re.IGNORECASE),
    re.compile(r"\bev\b", re.IGNORECASE),
    re.compile(r"\b(?:SLOW|MIDDLE|FAST)\b"),
    re.compile(r"\bS\s*/\s*M\s*/\s*F\b", re.IGNORECASE),
    re.compile(r"(?:オッズ|人気|払戻|購入|買い|注文|通知|勝率|複勝率)"),
)
FORBIDDEN_EVIDENCE_TOKENS = {
    "rank",
    "gap",
    "score",
    "confidence",
    "odds",
    "odd",
    "market",
    "popularity",
    "payoff",
    "payout",
    "refund",
    "probability",
    "buy",
    "stake",
    "champion",
    "notification",
    "order",
    "ev",
    "slow",
    "middle",
    "fast",
}
FORBIDDEN_EVIDENCE_COMPOUNDS = {
    "aiscore",
    "airank",
    "winprobability",
    "winprob",
    "winrate",
    "placeprobability",
    "placeprob",
    "placerate",
    "scenariosensitivity",
    "weightedtotal",
    "marketvalue",
    "marketprice",
    "expectedvalue",
}
FORBIDDEN_JAPANESE_TEXT = (
    "旧漏洩",
    "AI順位",
    "ａｉ順位",
    "市場価格",
    "期待値",
    "オッズ",
    "人気",
    "払戻",
    "購入",
    "買い",
    "注文",
    "通知",
    "勝率",
    "複勝率",
)


class DecisionSummaryError(ValueError):
    """Fail-closed error for an invalid Decision Summary source or output."""


def _reject_constant(value: str) -> None:
    raise DecisionSummaryError(f"non-finite JSON number is forbidden: {value}")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DecisionSummaryError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_strict_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DecisionSummaryError(f"cannot read JSON source {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DecisionSummaryError("source root must be an object")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def detail_ref(race_id: str, horse_id: str) -> str:
    """Return an injective, fragment-only anchor for a race/horse identity pair."""

    return f"#detail-r{len(race_id)}-{race_id}-h{len(horse_id)}-{horse_id}"


def require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DecisionSummaryError(f"{context}: key mismatch; missing={missing}, extra={extra}")


def require_object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DecisionSummaryError(f"{context}: object required")
    return value


def require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise DecisionSummaryError(f"{context}: array required")
    return value


def require_string(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise DecisionSummaryError(f"{context}: non-empty string required")
    return value


def reject_forbidden_evidence_text(value: str, context: str) -> str:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9]+", camel_split)]
    joined = "".join(tokens)
    if (
        any(token in FORBIDDEN_EVIDENCE_TOKENS for token in tokens)
        or any(compound in joined for compound in FORBIDDEN_EVIDENCE_COMPOUNDS)
        or any(marker in value for marker in FORBIDDEN_JAPANESE_TEXT)
    ):
        raise DecisionSummaryError(f"{context}: prohibited legacy/scenario/market text")
    for pattern in FORBIDDEN_EVIDENCE_TEXT:
        if pattern.search(value):
            raise DecisionSummaryError(f"{context}: prohibited legacy/scenario/market text")
    return value


def require_identifier(value: Any, context: str) -> str:
    text = require_string(value, context)
    if not SAFE_ID.fullmatch(text):
        raise DecisionSummaryError(f"{context}: invalid identifier")
    return text


def require_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DecisionSummaryError(f"{context}: integer >= {minimum} required")
    return value


def require_enum(value: Any, allowed: Sequence[str], context: str) -> str:
    text = require_string(value, context)
    if text not in allowed:
        raise DecisionSummaryError(f"{context}: {text!r} not in {list(allowed)}")
    return text


def require_bool(value: Any, expected: bool, context: str) -> bool:
    if not isinstance(value, bool) or value is not expected:
        raise DecisionSummaryError(f"{context}: must be {expected}")
    return value


def require_zero(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise DecisionSummaryError(f"{context}: must be integer zero")
    return value


def normalize_evidence_ids(value: Any, context: str) -> list[str]:
    items = require_list(value, context)
    normalized = [require_identifier(item, f"{context}[{index}]") for index, item in enumerate(items)]
    if len(normalized) != len(set(normalized)):
        raise DecisionSummaryError(f"{context}: duplicate evidence id")
    return sorted(normalized)


def validate_safety(value: Any, context: str) -> dict[str, Any]:
    safety = require_object(value, context)
    require_exact_keys(safety, SOURCE_SAFETY_KEYS, context)
    return {
        "formal_buy": require_bool(safety["formal_buy"], False, f"{context}.formal_buy"),
        "oos_evaluation_executed": require_bool(
            safety["oos_evaluation_executed"], False, f"{context}.oos_evaluation_executed"
        ),
        "real_data_materialized": require_bool(
            safety["real_data_materialized"], False, f"{context}.real_data_materialized"
        ),
        "send_order": require_bool(safety["send_order"], False, f"{context}.send_order"),
        "stake": require_zero(safety["stake"], f"{context}.stake"),
        "target_prediction_generated": require_bool(
            safety["target_prediction_generated"], False, f"{context}.target_prediction_generated"
        ),
        "training_executed": require_bool(
            safety["training_executed"], False, f"{context}.training_executed"
        ),
    }


def normalize_basics(value: Any, context: str) -> dict[str, Any]:
    basics = require_object(value, context)
    require_exact_keys(basics, BASICS_KEYS, context)
    return {
        "age": require_int(basics["age"], f"{context}.age", minimum=1),
        "assigned_weight": require_string(basics["assigned_weight"], f"{context}.assigned_weight"),
        "frame_no": require_int(basics["frame_no"], f"{context}.frame_no", minimum=1),
        "horse_name": require_string(basics["horse_name"], f"{context}.horse_name"),
        "horse_no": require_int(basics["horse_no"], f"{context}.horse_no", minimum=1),
        "jockey": require_string(basics["jockey"], f"{context}.jockey"),
        "sex": require_string(basics["sex"], f"{context}.sex"),
        "trainer": require_string(basics["trainer"], f"{context}.trainer"),
    }


def normalize_role_queue(value: Any, context: str) -> dict[str, Any]:
    role_queue = require_object(value, context)
    require_exact_keys(role_queue, ROLE_QUEUE_KEYS, context)
    rivals: list[dict[str, Any]] = []
    for index, raw in enumerate(require_list(role_queue["pressure_rivals"], f"{context}.pressure_rivals")):
        rival_context = f"{context}.pressure_rivals[{index}]"
        rival = require_object(raw, rival_context)
        require_exact_keys(rival, PRESSURE_RIVAL_KEYS, rival_context)
        rivals.append(
            {
                "conflict_level": require_enum(
                    rival["conflict_level"], ("LOW", "MEDIUM", "HIGH"), f"{rival_context}.conflict_level"
                ),
                "evidence_ids": normalize_evidence_ids(rival["evidence_ids"], f"{rival_context}.evidence_ids"),
                "horse_no": require_int(rival["horse_no"], f"{rival_context}.horse_no", minimum=1),
            }
        )
        if not rivals[-1]["evidence_ids"]:
            raise DecisionSummaryError(f"{rival_context}: pressure rival requires evidence ids")
    rival_numbers = [item["horse_no"] for item in rivals]
    if len(rival_numbers) != len(set(rival_numbers)):
        raise DecisionSummaryError(f"{context}.pressure_rivals: duplicate horse number")
    rivals.sort(key=lambda item: item["horse_no"])
    normalized = {
        "can_rate": require_enum(role_queue["can_rate"], ("YES", "NO", "UNKNOWN"), f"{context}.can_rate"),
        "expected_position": require_enum(
            role_queue["expected_position"], ROLE_VALUES, f"{context}.expected_position"
        ),
        "first_turn_position_cost": require_enum(
            role_queue["first_turn_position_cost"], BURDEN_VALUES, f"{context}.first_turn_position_cost"
        ),
        "forward_propensity": require_enum(
            role_queue["forward_propensity"], FORWARD_VALUES, f"{context}.forward_propensity"
        ),
        "lead_dependency": require_enum(
            role_queue["lead_dependency"], DEPENDENCY_VALUES, f"{context}.lead_dependency"
        ),
        "position_flexibility": require_enum(
            role_queue["position_flexibility"], FLEXIBILITY_VALUES, f"{context}.position_flexibility"
        ),
        "pressure_rivals": rivals,
        "queue_evidence_ids": normalize_evidence_ids(
            role_queue["queue_evidence_ids"], f"{context}.queue_evidence_ids"
        ),
        "role_evidence_ids": normalize_evidence_ids(
            role_queue["role_evidence_ids"], f"{context}.role_evidence_ids"
        ),
        "role_range": require_enum(role_queue["role_range"], ROLE_RANGE_VALUES, f"{context}.role_range"),
        "target_context_role": require_enum(
            role_queue["target_context_role"], ROLE_VALUES, f"{context}.target_context_role"
        ),
    }
    known_role_value = any(
        (
            normalized["can_rate"] != "UNKNOWN",
            normalized["expected_position"] != "UNKNOWN",
            normalized["forward_propensity"] != "UNKNOWN",
            normalized["position_flexibility"] != "UNKNOWN",
            normalized["role_range"] != "UNKNOWN",
            normalized["target_context_role"] != "UNKNOWN",
        )
    )
    known_queue_value = any(
        (
            normalized["first_turn_position_cost"] != "UNKNOWN",
            normalized["lead_dependency"] != "UNKNOWN",
            bool(normalized["pressure_rivals"]),
        )
    )
    if known_role_value and not normalized["role_evidence_ids"]:
        raise DecisionSummaryError(f"{context}: known role values require role evidence ids")
    if known_queue_value and not normalized["queue_evidence_ids"]:
        raise DecisionSummaryError(f"{context}: known Queue values require Queue evidence ids")
    for rival in normalized["pressure_rivals"]:
        if not set(rival["evidence_ids"]) <= set(normalized["queue_evidence_ids"]):
            raise DecisionSummaryError(f"{context}: pressure-rival evidence must be Queue evidence")
    return normalized


def normalize_condition(value: Any, context: str) -> dict[str, Any]:
    condition = require_object(value, context)
    require_exact_keys(condition, CONDITION_KEYS, context)
    fit = require_enum(condition["fit"], CONDITION_VALUES, f"{context}.fit")
    status = require_enum(condition["status"], EVIDENCE_STATUS_VALUES, f"{context}.status")
    evidence_ids = normalize_evidence_ids(condition["evidence_ids"], f"{context}.evidence_ids")
    if status == "UNOBSERVED" and (fit != "UNKNOWN" or evidence_ids):
        raise DecisionSummaryError(f"{context}: unobserved condition must be UNKNOWN with no evidence ids")
    if status == "OBSERVED" and (fit == "UNKNOWN" or not evidence_ids):
        raise DecisionSummaryError(f"{context}: observed condition needs a known fit and evidence ids")
    return {"evidence_ids": evidence_ids, "fit": fit, "status": status}


def normalize_evidence_details(value: Any, context: str) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    for index, raw in enumerate(require_list(value, context)):
        item_context = f"{context}[{index}]"
        item = require_object(raw, item_context)
        require_exact_keys(item, EVIDENCE_DETAIL_KEYS, item_context)
        details.append(
            {
                "evidence_id": require_identifier(item["evidence_id"], f"{item_context}.evidence_id"),
                "label": reject_forbidden_evidence_text(
                    require_string(item["label"], f"{item_context}.label"), f"{item_context}.label"
                ),
                "section": require_enum(item["section"], EVIDENCE_SECTIONS, f"{item_context}.section"),
                "source_note": reject_forbidden_evidence_text(
                    require_string(item["source_note"], f"{item_context}.source_note", allow_empty=True),
                    f"{item_context}.source_note",
                ),
                "status": require_enum(item["status"], EVIDENCE_STATUS_VALUES, f"{item_context}.status"),
                "value": reject_forbidden_evidence_text(
                    require_string(item["value"], f"{item_context}.value", allow_empty=True),
                    f"{item_context}.value",
                ),
            }
        )
    ids = [item["evidence_id"] for item in details]
    if len(ids) != len(set(ids)):
        raise DecisionSummaryError(f"{context}: duplicate evidence id")
    details.sort(key=lambda item: item["evidence_id"])
    return details


def project_horsecard_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the strict synthetic projection.

    Unknown keys fail closed.  This is the only source boundary for Weekend V0,
    so legacy ability, model confidence, scenario sensitivity, and market fields
    cannot enter the consumer.
    """

    root = require_object(source, "source")
    require_exact_keys(root, SOURCE_TOP_KEYS, "source")
    if root["schema_version"] != SOURCE_SCHEMA_VERSION:
        raise DecisionSummaryError(f"source.schema_version must be {SOURCE_SCHEMA_VERSION}")
    require_bool(root["synthetic_fixture"], True, "source.synthetic_fixture")
    safety = validate_safety(root["safety"], "source.safety")

    races: list[dict[str, Any]] = []
    for race_index, raw_race in enumerate(require_list(root["races"], "source.races")):
        race_context = f"source.races[{race_index}]"
        race = require_object(raw_race, race_context)
        require_exact_keys(race, SOURCE_RACE_KEYS, race_context)
        normalized_runners: list[dict[str, Any]] = []
        for runner_index, raw_runner in enumerate(require_list(race["runners"], f"{race_context}.runners")):
            runner_context = f"{race_context}.runners[{runner_index}]"
            runner = require_object(raw_runner, runner_context)
            require_exact_keys(runner, SOURCE_RUNNER_KEYS, runner_context)
            normalized = {
                "basics": normalize_basics(runner["basics"], f"{runner_context}.basics"),
                "current_condition_evidence": normalize_condition(
                    runner["current_condition_evidence"], f"{runner_context}.current_condition_evidence"
                ),
                "evidence_details": normalize_evidence_details(
                    runner["evidence_details"], f"{runner_context}.evidence_details"
                ),
                "horse_id": require_identifier(runner["horse_id"], f"{runner_context}.horse_id"),
                "role_queue_evidence": normalize_role_queue(
                    runner["role_queue_evidence"], f"{runner_context}.role_queue_evidence"
                ),
            }
            normalized_runners.append(normalized)

        horse_ids = [runner["horse_id"] for runner in normalized_runners]
        horse_numbers = [runner["basics"]["horse_no"] for runner in normalized_runners]
        if not normalized_runners:
            raise DecisionSummaryError(f"{race_context}.runners: at least one runner required")
        if len(horse_ids) != len(set(horse_ids)):
            raise DecisionSummaryError(f"{race_context}.runners: duplicate horse_id")
        if len(horse_numbers) != len(set(horse_numbers)):
            raise DecisionSummaryError(f"{race_context}.runners: duplicate horse_no")
        universe = set(horse_numbers)
        for runner in normalized_runners:
            own_no = runner["basics"]["horse_no"]
            for rival in runner["role_queue_evidence"]["pressure_rivals"]:
                if rival["horse_no"] == own_no or rival["horse_no"] not in universe:
                    raise DecisionSummaryError(
                        f"{race_context}: pressure rival {rival['horse_no']} is outside the race or self"
                    )
            detail_index = {item["evidence_id"]: item for item in runner["evidence_details"]}
            detail_ids = set(detail_index)
            role_ids = set(runner["role_queue_evidence"]["role_evidence_ids"])
            queue_ids = set(runner["role_queue_evidence"]["queue_evidence_ids"])
            condition_ids = set(runner["current_condition_evidence"]["evidence_ids"])
            referenced = role_ids | queue_ids | condition_ids
            for rival in runner["role_queue_evidence"]["pressure_rivals"]:
                referenced.update(rival["evidence_ids"])
            if not referenced <= detail_ids:
                raise DecisionSummaryError(
                    f"{race_context}.{runner['horse_id']}: missing evidence details {sorted(referenced - detail_ids)}"
                )
            if any(detail_index[evidence_id]["status"] == "UNOBSERVED" for evidence_id in referenced):
                raise DecisionSummaryError(
                    f"{race_context}.{runner['horse_id']}: structured evidence cannot reference UNOBSERVED detail"
                )
            if any(detail_index[evidence_id]["section"] not in {"ROLE", "ROUTE"} for evidence_id in role_ids):
                raise DecisionSummaryError(
                    f"{race_context}.{runner['horse_id']}: role evidence must use ROLE or ROUTE detail"
                )
            if any(detail_index[evidence_id]["section"] != "QUEUE" for evidence_id in queue_ids):
                raise DecisionSummaryError(
                    f"{race_context}.{runner['horse_id']}: Queue evidence must use QUEUE detail"
                )
            if any(detail_index[evidence_id]["section"] != "CONDITION" for evidence_id in condition_ids):
                raise DecisionSummaryError(
                    f"{race_context}.{runner['horse_id']}: condition evidence must use CONDITION detail"
                )
            if condition_ids and any(
                detail_index[evidence_id]["status"] != runner["current_condition_evidence"]["status"]
                for evidence_id in condition_ids
            ):
                raise DecisionSummaryError(
                    f"{race_context}.{runner['horse_id']}: condition block/detail status mismatch"
                )
        normalized_runners.sort(key=lambda item: item["basics"]["horse_no"])
        races.append(
            {
                "display_order": require_int(race["display_order"], f"{race_context}.display_order", minimum=1),
                "race_id": require_identifier(race["race_id"], f"{race_context}.race_id"),
                "race_name": require_string(race["race_name"], f"{race_context}.race_name"),
                "runners": normalized_runners,
            }
        )

    if not races:
        raise DecisionSummaryError("source.races: at least one race required")
    race_ids = [race["race_id"] for race in races]
    display_orders = [race["display_order"] for race in races]
    if len(race_ids) != len(set(race_ids)):
        raise DecisionSummaryError("source.races: duplicate race_id")
    if len(display_orders) != len(set(display_orders)):
        raise DecisionSummaryError("source.races: duplicate display_order")
    races.sort(key=lambda race: race["display_order"])
    return {
        "races": races,
        "safety": safety,
        "schema_version": SOURCE_SCHEMA_VERSION,
        "synthetic_fixture": True,
    }


def role_core_available(role_queue: Mapping[str, Any]) -> bool:
    return all(
        (
            role_queue["target_context_role"] != "UNKNOWN",
            role_queue["role_range"] != "UNKNOWN",
            role_queue["expected_position"] != "UNKNOWN",
            bool(role_queue["role_evidence_ids"]),
        )
    )


def derive_current_condition(condition: Mapping[str, Any]) -> dict[str, Any]:
    directly_available = condition["status"] == "OBSERVED" and condition["fit"] != "UNKNOWN"
    return {
        "availability": "AVAILABLE" if directly_available else "NOT_AVAILABLE",
        "evidence_count": len(condition["evidence_ids"]) if directly_available else 0,
        "source_status": condition["status"],
        "value": condition["fit"] if directly_available else "UNKNOWN",
    }


def derive_queue_fit(role_queue: Mapping[str, Any]) -> str:
    levels = {rival["conflict_level"] for rival in role_queue["pressure_rivals"]}
    if (
        "HIGH" in levels
        or role_queue["first_turn_position_cost"] == "HIGH"
        or role_queue["lead_dependency"] == "POSSIBLE"
    ):
        return "ADVERSE"
    required_known = (
        role_core_available(role_queue)
        and role_queue["forward_propensity"] != "UNKNOWN"
        and role_queue["first_turn_position_cost"] != "UNKNOWN"
        and role_queue["lead_dependency"] != "UNKNOWN"
        and bool(role_queue["queue_evidence_ids"])
    )
    if not required_known:
        return "UNKNOWN"
    if "MEDIUM" in levels or role_queue["first_turn_position_cost"] == "MEDIUM":
        return "CONDITIONAL"
    return "SUPPORTIVE"


def make_trigger(code: str, evidence_ids: Sequence[str], kind: str) -> dict[str, Any]:
    catalog = POSITIVE_WORLD_STATES if kind == "UPSIDE" else FAILURE_WORLD_STATES
    normalized_ids = sorted(set(evidence_ids))
    return {
        "code": code,
        "description": catalog[code],
        "evidence_count": len(normalized_ids),
        "evidence_ids": normalized_ids,
        "kind": kind,
    }


def derive_triggers(runner: Mapping[str, Any], queue_fit: str, condition: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    role_queue = runner["role_queue_evidence"]
    role_ids = role_queue["role_evidence_ids"]
    queue_ids = role_queue["queue_evidence_ids"]
    upside_codes: dict[str, Sequence[str]] = {}
    failure_codes: dict[str, Sequence[str]] = {}
    if role_core_available(role_queue):
        upside_codes["WORLD_TARGET_ROLE_REPRODUCED"] = role_ids
    if queue_fit == "SUPPORTIVE":
        upside_codes["WORLD_QUEUE_COST_STAYS_LOW"] = queue_ids
    if queue_fit == "CONDITIONAL":
        upside_codes["WORLD_EARLY_PRESSURE_EASES"] = queue_ids
        if role_queue["can_rate"] == "YES":
            upside_codes["WORLD_CAN_RATE_PATH_AVAILABLE"] = role_ids
    if condition["value"] == "SUPPORTIVE":
        upside_codes["WORLD_CURRENT_CONDITION_SUPPORT_PERSISTS"] = runner[
            "current_condition_evidence"
        ]["evidence_ids"]

    high_rival_ids = sorted(
        {
            evidence_id
            for rival in role_queue["pressure_rivals"]
            if rival["conflict_level"] == "HIGH"
            for evidence_id in rival["evidence_ids"]
        }
    )
    if high_rival_ids:
        failure_codes["FAILURE_EARLY_CONFLICT_ESCALATES"] = high_rival_ids
    if role_queue["first_turn_position_cost"] == "HIGH":
        failure_codes["FAILURE_FIRST_TURN_COST_REALIZED"] = queue_ids
    if role_queue["lead_dependency"] == "POSSIBLE":
        failure_codes["FAILURE_NONLEAD_POSITION_FORCED"] = queue_ids
    if role_queue["role_range"] == "NARROW" and role_queue["position_flexibility"] == "LIMITED":
        failure_codes["FAILURE_ROLE_BAND_MISSED"] = role_ids
    if condition["value"] == "ADVERSE":
        failure_codes["FAILURE_CURRENT_CONDITION_FAILS"] = runner[
            "current_condition_evidence"
        ]["evidence_ids"]

    upside = [
        make_trigger(code, upside_codes[code], "UPSIDE")
        for code in POSITIVE_WORLD_STATE_ORDER
        if code in upside_codes
    ]
    fragility = [
        make_trigger(code, failure_codes[code], "FAILURE")
        for code in FAILURE_WORLD_STATE_ORDER
        if code in failure_codes
    ]
    return upside, fragility


def derive_reason_codes(
    role_queue: Mapping[str, Any],
    condition: Mapping[str, Any],
    queue_fit: str,
) -> list[str]:
    codes = {"ABILITY_CLEAN_NOT_AVAILABLE"}
    core = role_core_available(role_queue)
    if core:
        codes.add("PRIMARY_ROLE_REACHABLE")
    else:
        codes.add("INSUFFICIENT_ROLE_CORE")
    if queue_fit == "SUPPORTIVE":
        codes.add("PRIMARY_QUEUE_SUPPORTIVE")
    elif queue_fit == "CONDITIONAL":
        if role_queue["first_turn_position_cost"] == "MEDIUM":
            codes.add("CONDITIONAL_MEDIUM_FIRST_TURN_COST")
        if any(rival["conflict_level"] == "MEDIUM" for rival in role_queue["pressure_rivals"]):
            codes.add("CONDITIONAL_MEDIUM_PRESSURE_RIVAL")
        if role_queue["can_rate"] == "YES":
            codes.add("CONDITIONAL_CAN_RATE_PATH_AVAILABLE")
    elif queue_fit == "UNKNOWN":
        codes.add("INSUFFICIENT_QUEUE_CONTEXT")

    if role_queue["role_range"] == "NARROW" and role_queue["position_flexibility"] == "LIMITED":
        codes.add("FRAGILE_ROLE_RANGE_NARROW")
    if role_queue["lead_dependency"] == "POSSIBLE":
        codes.add("FRAGILE_LEAD_DEPENDENCY_POSSIBLE")
    if any(rival["conflict_level"] == "HIGH" for rival in role_queue["pressure_rivals"]):
        codes.add("FRAGILE_QUEUE_HIGH_CONFLICT")
    if role_queue["first_turn_position_cost"] == "HIGH":
        codes.add("FRAGILE_FIRST_TURN_COST_HIGH")
    if condition["value"] == "ADVERSE":
        codes.add("FRAGILE_CURRENT_CONDITION_ADVERSE")
    if condition["value"] == "SUPPORTIVE":
        codes.add("CURRENT_CONDITION_SUPPORTIVE")
    if condition["availability"] == "NOT_AVAILABLE":
        codes.add("CONDITION_CURRENT_NOT_AVAILABLE")
    return [code for code in REASON_CODE_ORDER if code in codes]


def classify_review_priority(
    role_queue: Mapping[str, Any],
    queue_fit: str,
    fragility: Sequence[Mapping[str, Any]],
    upside: Sequence[Mapping[str, Any]],
) -> str:
    if fragility:
        return "FRAGILE"
    core = role_core_available(role_queue)
    if core and queue_fit == "SUPPORTIVE":
        return "PRIMARY_REVIEW"
    if core and queue_fit == "CONDITIONAL" and upside:
        return "CONDITIONAL_REVIEW"
    return "INSUFFICIENT"


def derive_confidence(
    classification: str,
    role_available: bool,
    queue_fit: str,
    condition: Mapping[str, Any],
) -> dict[str, Any]:
    dimensions: list[str] = []
    if role_available:
        dimensions.append("ROLE")
    if queue_fit != "UNKNOWN":
        dimensions.append("QUEUE")
    if condition["availability"] == "AVAILABLE":
        dimensions.append("CONDITION")
    if classification == "INSUFFICIENT":
        value = "NOT_AVAILABLE"
    elif classification == "FRAGILE":
        value = "LOW"
    elif {"ROLE", "QUEUE", "CONDITION"} <= set(dimensions):
        value = "MEDIUM"
    else:
        value = "LOW"
    return {
        "available_dimensions": dimensions,
        "basis": "SUMMARY_EVIDENCE_AVAILABILITY_ONLY; UPSTREAM_CONFIDENCE_UNUSED; CAPPED_AT_MEDIUM",
        "upstream_confidence_used": False,
        "value": value,
    }


def build_world_state_text(
    classification: str,
    role_queue: Mapping[str, Any],
    upside: Sequence[Mapping[str, Any]],
    fragility: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    if classification == "INSUFFICIENT":
        return (
            "NOT_AVAILABLE: positive world-state evidence is insufficient.",
            "NOT_AVAILABLE: failure world-state evidence is insufficient.",
        )
    positive_text = " / ".join(trigger["description"] for trigger in upside)
    if not positive_text:
        positive_text = "NOT_AVAILABLE: positive world-state evidence is insufficient."
    failure_text = " / ".join(trigger["description"] for trigger in fragility)
    if not failure_text:
        failure_text = "想定位置帯に届かない、またはQueue・状態面の肯定条件が成立しない"
    return positive_text, failure_text


def derive_horse_summary(race_id: str, runner: Mapping[str, Any]) -> dict[str, Any]:
    role_queue = runner["role_queue_evidence"]
    condition = derive_current_condition(runner["current_condition_evidence"])
    queue_fit = derive_queue_fit(role_queue)
    upside, fragility = derive_triggers(runner, queue_fit, condition)
    classification = classify_review_priority(role_queue, queue_fit, fragility, upside)
    reason_codes = derive_reason_codes(role_queue, condition, queue_fit)
    winning_state, failure_state = build_world_state_text(
        classification, role_queue, upside, fragility
    )

    role_ids = set(role_queue["role_evidence_ids"])
    queue_ids = set(role_queue["queue_evidence_ids"])
    for rival in role_queue["pressure_rivals"]:
        queue_ids.update(rival["evidence_ids"])
    condition_ids = set(runner["current_condition_evidence"]["evidence_ids"])
    distinct_ids = role_ids | queue_ids | condition_ids
    basics = dict(runner["basics"])
    return {
        "ability_status": {
            "availability": "NOT_AVAILABLE",
            "evidence_count": 0,
            "reason_codes": ["ABILITY_CLEAN_NOT_AVAILABLE"],
            "value": "UNKNOWN",
        },
        "basics": basics,
        "classification": classification,
        "classification_semantics": "REVIEW_PRIORITY_ONLY",
        "confidence": derive_confidence(
            classification, role_core_available(role_queue), queue_fit, condition
        ),
        "current_condition_fit": condition,
        "detail_ref": detail_ref(race_id, runner["horse_id"]),
        "evidence_count": {
            "classification_uses_count": False,
            "current_condition": len(condition_ids),
            "detail_items": len(runner["evidence_details"]),
            "evidence_ids_by_dimension": {
                "current_condition": sorted(condition_ids),
                "queue": sorted(queue_ids),
                "role": sorted(role_ids),
            },
            "queue": len(queue_ids),
            "role": len(role_ids),
            "total_distinct": len(distinct_ids),
        },
        "failure_world_state": failure_state,
        "fragility_triggers": fragility,
        "horse_id": runner["horse_id"],
        "queue_fit": {
            "availability": "AVAILABLE" if queue_fit != "UNKNOWN" else "NOT_AVAILABLE",
            "evidence_count": len(queue_ids) if queue_fit != "UNKNOWN" else 0,
            "value": queue_fit,
        },
        "reason_codes": reason_codes,
        "role_expected_position": {
            "availability": "AVAILABLE" if role_core_available(role_queue) else "NOT_AVAILABLE",
            "expected_position": role_queue["expected_position"],
            "position_flexibility": role_queue["position_flexibility"],
            "role": role_queue["target_context_role"],
            "role_range": role_queue["role_range"],
        },
        "upside_triggers": upside,
        "winning_or_in_the_money_world_state": winning_state,
    }


def race_reference(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "failure_world_state_codes": [item["code"] for item in summary["fragility_triggers"]],
        "horse_id": summary["horse_id"],
        "horse_name": summary["basics"]["horse_name"],
        "horse_no": summary["basics"]["horse_no"],
        "positive_world_state_codes": [item["code"] for item in summary["upside_triggers"]],
        "reason_codes": summary["reason_codes"],
    }


def aggregate_key_world_states(horses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    beneficiaries: dict[tuple[str, str], set[int]] = {}
    for horse in horses:
        horse_no = horse["basics"]["horse_no"]
        for trigger in horse["upside_triggers"]:
            beneficiaries.setdefault(("UPSIDE", trigger["code"]), set()).add(horse_no)
        for trigger in horse["fragility_triggers"]:
            beneficiaries.setdefault(("FAILURE", trigger["code"]), set()).add(horse_no)
    result: list[dict[str, Any]] = []
    for code in POSITIVE_WORLD_STATE_ORDER:
        key = ("UPSIDE", code)
        if key in beneficiaries:
            result.append(
                {
                    "code": code,
                    "description": POSITIVE_WORLD_STATES[code],
                    "horse_nos": sorted(beneficiaries[key]),
                    "state_kind": "UPSIDE",
                }
            )
    for code in FAILURE_WORLD_STATE_ORDER:
        key = ("FAILURE", code)
        if key in beneficiaries:
            result.append(
                {
                    "code": code,
                    "description": FAILURE_WORLD_STATES[code],
                    "horse_nos": sorted(beneficiaries[key]),
                    "state_kind": "FAILURE",
                }
            )
    return result


def build_decision_summary(source: Mapping[str, Any]) -> dict[str, Any]:
    projected = project_horsecard_source(source)
    source_bytes = canonical_json_bytes(projected)
    races: list[dict[str, Any]] = []
    runner_count = 0
    for race in projected["races"]:
        horse_summaries = [
            derive_horse_summary(race["race_id"], runner) for runner in race["runners"]
        ]
        runner_count += len(horse_summaries)
        buckets = {
            classification: [
                race_reference(summary)
                for summary in horse_summaries
                if summary["classification"] == classification
            ]
            for classification in CLASSIFICATIONS
        }
        details = [
            {
                "detail_ref": detail_ref(race["race_id"], runner["horse_id"]),
                "horse_id": runner["horse_id"],
                "horse_name": runner["basics"]["horse_name"],
                "horse_no": runner["basics"]["horse_no"],
                "items": [dict(item) for item in runner["evidence_details"]],
            }
            for runner in race["runners"]
        ]
        races.append(
            {
                "decision_summary": {
                    "conditional_review": buckets["CONDITIONAL_REVIEW"],
                    "display_order_rule": "OFFICIAL_HORSE_NO; NOT_REVIEW_ORDER",
                    "fragile_or_downgrade": buckets["FRAGILE"],
                    "insufficient_review": buckets["INSUFFICIENT"],
                    "key_world_states": aggregate_key_world_states(horse_summaries),
                    "primary_review": buckets["PRIMARY_REVIEW"],
                },
                "display_order": race["display_order"],
                "evidence_details": details,
                "horse_summaries": horse_summaries,
                "race_id": race["race_id"],
                "race_name": race["race_name"],
                "runner_count": len(horse_summaries),
            }
        )
    result = {
        "aggregation_rule": RULE_VERSION,
        "layer_type": LAYER_TYPE,
        "race_count": len(races),
        "races": races,
        "reason_code_catalog": dict(REASON_CODE_CATALOG),
        "review_priority_semantics": REVIEW_SEMANTICS,
        "runner_count": runner_count,
        "safety": projected["safety"],
        "schema_version": SCHEMA_VERSION,
        "source_mode": "SYNTHETIC_FIXTURE_ONLY",
        "source_projection_sha256": sha256_hex(source_bytes),
        "synthetic_only": True,
        "weighted_total_generated": False,
    }
    validate_decision_summary(result, projected)
    return result


def validate_decision_summary(output: Mapping[str, Any], projected: Mapping[str, Any] | None = None) -> None:
    expected_top = {
        "aggregation_rule",
        "layer_type",
        "race_count",
        "races",
        "reason_code_catalog",
        "review_priority_semantics",
        "runner_count",
        "safety",
        "schema_version",
        "source_mode",
        "source_projection_sha256",
        "synthetic_only",
        "weighted_total_generated",
    }
    root = require_object(output, "output")
    require_exact_keys(root, expected_top, "output")
    if root["schema_version"] != SCHEMA_VERSION:
        raise DecisionSummaryError("output schema version mismatch")
    if root["layer_type"] != LAYER_TYPE or root["review_priority_semantics"] != REVIEW_SEMANTICS:
        raise DecisionSummaryError("output review-priority semantics mismatch")
    if root["aggregation_rule"] != RULE_VERSION or root["source_mode"] != "SYNTHETIC_FIXTURE_ONLY":
        raise DecisionSummaryError("output rule/source mode mismatch")
    source_hash = require_string(root["source_projection_sha256"], "output.source_projection_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise DecisionSummaryError("output source projection digest is invalid")
    require_bool(output["synthetic_only"], True, "output.synthetic_only")
    require_bool(output["weighted_total_generated"], False, "output.weighted_total_generated")
    validate_safety(output["safety"], "output.safety")
    if output["reason_code_catalog"] != REASON_CODE_CATALOG:
        raise DecisionSummaryError("reason-code catalog mismatch")
    races = require_list(root["races"], "output.races")
    if require_int(root["race_count"], "output.race_count", minimum=1) != len(races):
        raise DecisionSummaryError("output race_count mismatch")
    if not races:
        raise DecisionSummaryError("output.races: at least one race required")
    total = 0
    global_detail_refs: set[str] = set()
    for race_index, race in enumerate(races):
        context = f"output.races[{race_index}]"
        race = require_object(race, context)
        require_exact_keys(
            race,
            {
                "decision_summary",
                "display_order",
                "evidence_details",
                "horse_summaries",
                "race_id",
                "race_name",
                "runner_count",
            },
            context,
        )
        race_id = require_identifier(race["race_id"], f"{context}.race_id")
        require_string(race["race_name"], f"{context}.race_name")
        require_int(race["display_order"], f"{context}.display_order", minimum=1)
        horses = require_list(race["horse_summaries"], f"{context}.horse_summaries")
        details = require_list(race["evidence_details"], f"{context}.evidence_details")
        if not horses or require_int(race["runner_count"], f"{context}.runner_count", minimum=1) != len(horses) or len(horses) != len(details):
            raise DecisionSummaryError(f"{context}: runner/detail count mismatch")
        total += len(horses)
        horse_keys = {
            "ability_status",
            "basics",
            "classification",
            "classification_semantics",
            "confidence",
            "current_condition_fit",
            "detail_ref",
            "evidence_count",
            "failure_world_state",
            "fragility_triggers",
            "horse_id",
            "queue_fit",
            "reason_codes",
            "role_expected_position",
            "upside_triggers",
            "winning_or_in_the_money_world_state",
        }
        for horse_index, raw_horse in enumerate(horses):
            horse_context = f"{context}.horse_summaries[{horse_index}]"
            horse = require_object(raw_horse, horse_context)
            require_exact_keys(horse, horse_keys, horse_context)
            horse_id = require_identifier(horse["horse_id"], f"{horse_context}.horse_id")
            basics = normalize_basics(horse["basics"], f"{horse_context}.basics")
            if basics != horse["basics"]:
                raise DecisionSummaryError(f"{horse_context}: basics are not canonical")
            expected_ref = detail_ref(race_id, horse_id)
            if horse["detail_ref"] != expected_ref:
                raise DecisionSummaryError(f"{horse_context}: invalid detail ref")
            if expected_ref in global_detail_refs:
                raise DecisionSummaryError(f"{horse_context}: duplicate global detail ref")
            global_detail_refs.add(expected_ref)
            if horse["classification_semantics"] != "REVIEW_PRIORITY_ONLY":
                raise DecisionSummaryError(f"{horse_context}: classification semantics mismatch")
            classification = require_enum(
                horse["classification"], CLASSIFICATIONS, f"{horse_context}.classification"
            )
            if horse["ability_status"] != {
                "availability": "NOT_AVAILABLE",
                "evidence_count": 0,
                "reason_codes": ["ABILITY_CLEAN_NOT_AVAILABLE"],
                "value": "UNKNOWN",
            }:
                raise DecisionSummaryError(f"{horse_context}: clean ability must remain unavailable")

            condition = require_object(horse["current_condition_fit"], f"{horse_context}.current_condition_fit")
            require_exact_keys(
                condition,
                {"availability", "evidence_count", "source_status", "value"},
                f"{horse_context}.current_condition_fit",
            )
            require_enum(condition["availability"], ("AVAILABLE", "NOT_AVAILABLE"), f"{horse_context}.current_condition_fit.availability")
            require_enum(condition["source_status"], EVIDENCE_STATUS_VALUES, f"{horse_context}.current_condition_fit.source_status")
            require_enum(condition["value"], CONDITION_VALUES, f"{horse_context}.current_condition_fit.value")
            condition_count = require_int(condition["evidence_count"], f"{horse_context}.current_condition_fit.evidence_count")
            if condition["availability"] == "AVAILABLE":
                if condition["source_status"] != "OBSERVED" or condition["value"] == "UNKNOWN" or condition_count == 0:
                    raise DecisionSummaryError(f"{horse_context}: invalid available condition")
            elif condition["value"] != "UNKNOWN" or condition_count != 0:
                raise DecisionSummaryError(f"{horse_context}: unavailable condition must remain unknown")

            queue = require_object(horse["queue_fit"], f"{horse_context}.queue_fit")
            require_exact_keys(queue, {"availability", "evidence_count", "value"}, f"{horse_context}.queue_fit")
            require_enum(queue["availability"], ("AVAILABLE", "NOT_AVAILABLE"), f"{horse_context}.queue_fit.availability")
            require_enum(queue["value"], ("SUPPORTIVE", "CONDITIONAL", "ADVERSE", "UNKNOWN"), f"{horse_context}.queue_fit.value")
            queue_count = require_int(queue["evidence_count"], f"{horse_context}.queue_fit.evidence_count")
            if queue["availability"] == "AVAILABLE":
                if queue["value"] == "UNKNOWN" or queue_count == 0:
                    raise DecisionSummaryError(f"{horse_context}: available Queue needs evidence")
            elif queue["value"] != "UNKNOWN" or queue_count != 0:
                raise DecisionSummaryError(f"{horse_context}: unavailable Queue must remain unknown")

            role = require_object(horse["role_expected_position"], f"{horse_context}.role_expected_position")
            require_exact_keys(
                role,
                {"availability", "expected_position", "position_flexibility", "role", "role_range"},
                f"{horse_context}.role_expected_position",
            )
            require_enum(role["availability"], ("AVAILABLE", "NOT_AVAILABLE"), f"{horse_context}.role_expected_position.availability")
            require_enum(role["expected_position"], ROLE_VALUES, f"{horse_context}.role_expected_position.expected_position")
            require_enum(role["position_flexibility"], FLEXIBILITY_VALUES, f"{horse_context}.role_expected_position.position_flexibility")
            require_enum(role["role"], ROLE_VALUES, f"{horse_context}.role_expected_position.role")
            require_enum(role["role_range"], ROLE_RANGE_VALUES, f"{horse_context}.role_expected_position.role_range")
            if role["availability"] == "AVAILABLE" and (
                role["role"] == "UNKNOWN" or role["role_range"] == "UNKNOWN" or role["expected_position"] == "UNKNOWN"
            ):
                raise DecisionSummaryError(f"{horse_context}: available role core contains UNKNOWN")

            confidence = require_object(horse["confidence"], f"{horse_context}.confidence")
            require_exact_keys(
                confidence,
                {"available_dimensions", "basis", "upstream_confidence_used", "value"},
                f"{horse_context}.confidence",
            )
            dimensions = require_list(confidence["available_dimensions"], f"{horse_context}.confidence.available_dimensions")
            if dimensions != [item for item in ("ROLE", "QUEUE", "CONDITION") if item in dimensions] or len(dimensions) != len(set(dimensions)):
                raise DecisionSummaryError(f"{horse_context}: confidence dimensions are invalid")
            if confidence["basis"] != "SUMMARY_EVIDENCE_AVAILABILITY_ONLY; UPSTREAM_CONFIDENCE_UNUSED; CAPPED_AT_MEDIUM":
                raise DecisionSummaryError(f"{horse_context}: confidence basis mismatch")
            require_bool(confidence["upstream_confidence_used"], False, f"{horse_context}.confidence.upstream_confidence_used")
            confidence_value = require_enum(confidence["value"], ("MEDIUM", "LOW", "NOT_AVAILABLE"), f"{horse_context}.confidence.value")
            if classification == "INSUFFICIENT" and confidence_value != "NOT_AVAILABLE":
                raise DecisionSummaryError(f"{horse_context}: insufficient confidence mismatch")
            if classification == "FRAGILE" and confidence_value != "LOW":
                raise DecisionSummaryError(f"{horse_context}: fragile confidence mismatch")

            evidence = require_object(horse["evidence_count"], f"{horse_context}.evidence_count")
            require_exact_keys(
                evidence,
                {
                    "classification_uses_count",
                    "current_condition",
                    "detail_items",
                    "evidence_ids_by_dimension",
                    "queue",
                    "role",
                    "total_distinct",
                },
                f"{horse_context}.evidence_count",
            )
            require_bool(evidence["classification_uses_count"], False, f"{horse_context}.evidence_count.classification_uses_count")
            for key in ("current_condition", "detail_items", "queue", "role", "total_distinct"):
                require_int(evidence[key], f"{horse_context}.evidence_count.{key}")
            dimension_ids_raw = require_object(
                evidence["evidence_ids_by_dimension"],
                f"{horse_context}.evidence_count.evidence_ids_by_dimension",
            )
            require_exact_keys(
                dimension_ids_raw,
                {"current_condition", "queue", "role"},
                f"{horse_context}.evidence_count.evidence_ids_by_dimension",
            )
            dimension_ids: dict[str, list[str]] = {}
            for dimension in ("current_condition", "queue", "role"):
                ids = normalize_evidence_ids(
                    dimension_ids_raw[dimension],
                    f"{horse_context}.evidence_count.evidence_ids_by_dimension.{dimension}",
                )
                if ids != dimension_ids_raw[dimension]:
                    raise DecisionSummaryError(
                        f"{horse_context}: {dimension} evidence ids are not canonical"
                    )
                if evidence[dimension] != len(ids):
                    raise DecisionSummaryError(
                        f"{horse_context}: {dimension} evidence count/id mismatch"
                    )
                dimension_ids[dimension] = ids
            dimension_sets = {key: set(ids) for key, ids in dimension_ids.items()}
            if (
                dimension_sets["current_condition"] & dimension_sets["queue"]
                or dimension_sets["current_condition"] & dimension_sets["role"]
                or dimension_sets["queue"] & dimension_sets["role"]
            ):
                raise DecisionSummaryError(f"{horse_context}: evidence dimensions overlap")
            all_dimension_ids = set().union(*dimension_sets.values())
            if evidence["total_distinct"] != len(all_dimension_ids):
                raise DecisionSummaryError(f"{horse_context}: distinct evidence id count mismatch")
            if condition["availability"] == "AVAILABLE" and condition_count != evidence["current_condition"]:
                raise DecisionSummaryError(f"{horse_context}: condition summary/evidence count mismatch")
            if queue["availability"] == "AVAILABLE" and queue_count != evidence["queue"]:
                raise DecisionSummaryError(f"{horse_context}: Queue summary/evidence count mismatch")

            def validate_triggers(raw: Any, kind: str, trigger_context: str) -> list[Mapping[str, Any]]:
                triggers = require_list(raw, trigger_context)
                valid_catalog = POSITIVE_WORLD_STATES if kind == "UPSIDE" else FAILURE_WORLD_STATES
                valid_order = POSITIVE_WORLD_STATE_ORDER if kind == "UPSIDE" else FAILURE_WORLD_STATE_ORDER
                codes: list[str] = []
                for trigger_index, raw_trigger in enumerate(triggers):
                    item_context = f"{trigger_context}[{trigger_index}]"
                    trigger = require_object(raw_trigger, item_context)
                    require_exact_keys(
                        trigger,
                        {"code", "description", "evidence_count", "evidence_ids", "kind"},
                        item_context,
                    )
                    code = require_enum(trigger["code"], valid_order, f"{item_context}.code")
                    if trigger["kind"] != kind or trigger["description"] != valid_catalog[code]:
                        raise DecisionSummaryError(f"{item_context}: trigger catalog mismatch")
                    trigger_ids = normalize_evidence_ids(
                        trigger["evidence_ids"], f"{item_context}.evidence_ids"
                    )
                    if not trigger_ids or trigger_ids != trigger["evidence_ids"]:
                        raise DecisionSummaryError(
                            f"{item_context}: non-empty canonical evidence ids required"
                        )
                    trigger_count = require_int(
                        trigger["evidence_count"], f"{item_context}.evidence_count", minimum=1
                    )
                    if trigger_count != len(trigger_ids):
                        raise DecisionSummaryError(f"{item_context}: evidence count/id mismatch")
                    dimension = TRIGGER_EVIDENCE_DIMENSION[code]
                    if not set(trigger_ids) <= dimension_sets[dimension]:
                        raise DecisionSummaryError(
                            f"{item_context}: trigger evidence is outside {dimension} evidence"
                        )
                    codes.append(code)
                if codes != [code for code in valid_order if code in codes] or len(codes) != len(set(codes)):
                    raise DecisionSummaryError(f"{trigger_context}: trigger order/uniqueness mismatch")
                return triggers

            upside = validate_triggers(horse["upside_triggers"], "UPSIDE", f"{horse_context}.upside_triggers")
            fragility = validate_triggers(horse["fragility_triggers"], "FAILURE", f"{horse_context}.fragility_triggers")
            require_string(horse["winning_or_in_the_money_world_state"], f"{horse_context}.winning_or_in_the_money_world_state")
            require_string(horse["failure_world_state"], f"{horse_context}.failure_world_state")
            expected_winning, expected_failure = build_world_state_text(
                classification,
                {"expected_position": role["expected_position"]},
                upside,
                fragility,
            )
            if (
                horse["winning_or_in_the_money_world_state"] != expected_winning
                or horse["failure_world_state"] != expected_failure
            ):
                raise DecisionSummaryError(f"{horse_context}: world-state text/trigger mismatch")
            codes = require_list(horse["reason_codes"], f"{horse_context}.reason_codes")
            if (
                not codes
                or any(not isinstance(code, str) or code not in REASON_CODE_CATALOG for code in codes)
                or len(codes) != len(set(codes))
                or codes != [code for code in REASON_CODE_ORDER if code in codes]
            ):
                raise DecisionSummaryError(f"{horse_context}: invalid reason codes")
            if classification == "PRIMARY_REVIEW":
                required = {"PRIMARY_ROLE_REACHABLE", "PRIMARY_QUEUE_SUPPORTIVE"}
                if (
                    not required <= set(codes)
                    or fragility
                    or role["availability"] != "AVAILABLE"
                    or queue["value"] != "SUPPORTIVE"
                ):
                    raise DecisionSummaryError(f"{horse_context}: invalid primary contract")
            if classification == "CONDITIONAL_REVIEW" and (
                not upside
                or fragility
                or role["availability"] != "AVAILABLE"
                or queue["value"] != "CONDITIONAL"
            ):
                raise DecisionSummaryError(f"{horse_context}: invalid conditional contract")
            if classification == "FRAGILE" and not fragility:
                raise DecisionSummaryError(f"{horse_context}: fragile requires failure trigger")
            if fragility and classification != "FRAGILE":
                raise DecisionSummaryError(f"{horse_context}: fragility precedence mismatch")
            if classification == "INSUFFICIENT" and not fragility and (
                role["availability"] == "AVAILABLE"
                and queue["value"] in {"SUPPORTIVE", "CONDITIONAL"}
            ):
                raise DecisionSummaryError(f"{horse_context}: insufficient classification mismatch")

            code_set = set(codes)
            if (role["availability"] == "AVAILABLE") != ("PRIMARY_ROLE_REACHABLE" in code_set):
                raise DecisionSummaryError(f"{horse_context}: role reason mismatch")
            if (queue["value"] == "SUPPORTIVE") != ("PRIMARY_QUEUE_SUPPORTIVE" in code_set):
                raise DecisionSummaryError(f"{horse_context}: Queue reason mismatch")
            if (condition["value"] == "SUPPORTIVE") != ("CURRENT_CONDITION_SUPPORTIVE" in code_set):
                raise DecisionSummaryError(f"{horse_context}: condition support reason mismatch")
            if (condition["availability"] == "NOT_AVAILABLE") != ("CONDITION_CURRENT_NOT_AVAILABLE" in code_set):
                raise DecisionSummaryError(f"{horse_context}: condition availability reason mismatch")
            failure_reason_by_trigger = {
                "FAILURE_ROLE_BAND_MISSED": "FRAGILE_ROLE_RANGE_NARROW",
                "FAILURE_NONLEAD_POSITION_FORCED": "FRAGILE_LEAD_DEPENDENCY_POSSIBLE",
                "FAILURE_EARLY_CONFLICT_ESCALATES": "FRAGILE_QUEUE_HIGH_CONFLICT",
                "FAILURE_FIRST_TURN_COST_REALIZED": "FRAGILE_FIRST_TURN_COST_HIGH",
                "FAILURE_CURRENT_CONDITION_FAILS": "FRAGILE_CURRENT_CONDITION_ADVERSE",
            }
            failure_codes = {trigger["code"] for trigger in fragility}
            for trigger_code, reason_code in failure_reason_by_trigger.items():
                if (trigger_code in failure_codes) != (reason_code in code_set):
                    raise DecisionSummaryError(f"{horse_context}: failure trigger/reason mismatch")
            upside_codes = {trigger["code"] for trigger in upside}
            if role["availability"] == "AVAILABLE" and "WORLD_TARGET_ROLE_REPRODUCED" not in upside_codes:
                raise DecisionSummaryError(f"{horse_context}: available role needs role world state")
            if queue["value"] == "SUPPORTIVE" and "WORLD_QUEUE_COST_STAYS_LOW" not in upside_codes:
                raise DecisionSummaryError(f"{horse_context}: supportive Queue world state missing")
            if queue["value"] == "CONDITIONAL" and "WORLD_EARLY_PRESSURE_EASES" not in upside_codes:
                raise DecisionSummaryError(f"{horse_context}: conditional Queue world state missing")
            if condition["value"] == "SUPPORTIVE" and "WORLD_CURRENT_CONDITION_SUPPORT_PERSISTS" not in upside_codes:
                raise DecisionSummaryError(f"{horse_context}: condition world state missing")

            expected_confidence = derive_confidence(
                classification,
                role["availability"] == "AVAILABLE",
                queue["value"],
                condition,
            )
            if confidence != expected_confidence:
                raise DecisionSummaryError(f"{horse_context}: confidence derivation mismatch")
            if evidence["total_distinct"] != evidence["role"] + evidence["queue"] + evidence["current_condition"]:
                raise DecisionSummaryError(f"{horse_context}: distinct evidence count mismatch")
            if role["availability"] == "AVAILABLE" and evidence["role"] == 0:
                raise DecisionSummaryError(f"{horse_context}: available role needs typed evidence")
            if queue["availability"] == "AVAILABLE" and evidence["queue"] == 0:
                raise DecisionSummaryError(f"{horse_context}: available Queue needs typed evidence")

        horse_numbers = [horse["basics"]["horse_no"] for horse in horses]
        if horse_numbers != sorted(horse_numbers) or len(horse_numbers) != len(set(horse_numbers)):
            raise DecisionSummaryError(f"{context}: horse summaries not unique official-number order")
        horse_ids = [horse["horse_id"] for horse in horses]
        detail_ids = [detail["horse_id"] for detail in details]
        if horse_ids != detail_ids or len(horse_ids) != len(set(horse_ids)):
            raise DecisionSummaryError(f"{context}: detail identity mismatch")
        summary = require_object(race["decision_summary"], f"{context}.decision_summary")
        require_exact_keys(
            summary,
            {
                "conditional_review",
                "display_order_rule",
                "fragile_or_downgrade",
                "insufficient_review",
                "key_world_states",
                "primary_review",
            },
            f"{context}.decision_summary",
        )
        if summary["display_order_rule"] != "OFFICIAL_HORSE_NO; NOT_REVIEW_ORDER":
            raise DecisionSummaryError(f"{context}: display-order semantics mismatch")
        expected_buckets = {
            "primary_review": "PRIMARY_REVIEW",
            "conditional_review": "CONDITIONAL_REVIEW",
            "fragile_or_downgrade": "FRAGILE",
            "insufficient_review": "INSUFFICIENT",
        }
        seen: list[str] = []
        by_id = {horse["horse_id"]: horse for horse in horses}
        for bucket_name, classification in expected_buckets.items():
            refs = require_list(summary[bucket_name], f"{context}.decision_summary.{bucket_name}")
            ref_numbers = [ref["horse_no"] for ref in refs]
            if ref_numbers != sorted(ref_numbers):
                raise DecisionSummaryError(f"{context}.{bucket_name}: not official-number order")
            expected_ids = [
                horse["horse_id"] for horse in horses if horse["classification"] == classification
            ]
            actual_ids = [ref["horse_id"] for ref in refs]
            if actual_ids != expected_ids:
                raise DecisionSummaryError(f"{context}.{bucket_name}: classification mismatch")
            seen.extend(actual_ids)
            for ref in refs:
                require_exact_keys(
                    require_object(ref, f"{context}.{bucket_name}.reference"),
                    {"failure_world_state_codes", "horse_id", "horse_name", "horse_no", "positive_world_state_codes", "reason_codes"},
                    f"{context}.{bucket_name}.reference",
                )
                if ref != race_reference(by_id[ref["horse_id"]]):
                    raise DecisionSummaryError(f"{context}.{bucket_name}: horse reference mismatch")
        if sorted(seen) != sorted(horse_ids) or len(seen) != len(set(seen)):
            raise DecisionSummaryError(f"{context}: classification partition mismatch")
        expected_states = aggregate_key_world_states(horses)
        if summary["key_world_states"] != expected_states:
            raise DecisionSummaryError(f"{context}: key world-state mismatch")
        for detail_index, raw_detail in enumerate(details):
            detail_context = f"{context}.evidence_details[{detail_index}]"
            detail = require_object(raw_detail, detail_context)
            require_exact_keys(detail, {"detail_ref", "horse_id", "horse_name", "horse_no", "items"}, detail_context)
            horse = horses[detail_index]
            if (
                detail["horse_id"] != horse["horse_id"]
                or detail["horse_name"] != horse["basics"]["horse_name"]
                or detail["horse_no"] != horse["basics"]["horse_no"]
                or detail["detail_ref"] != horse["detail_ref"]
            ):
                raise DecisionSummaryError(f"{detail_context}: horse/detail identity mismatch")
            canonical_items = normalize_evidence_details(detail["items"], f"{detail_context}.items")
            if canonical_items != detail["items"]:
                raise DecisionSummaryError(f"{detail_context}: evidence detail order is not canonical")
            evidence = horse["evidence_count"]
            if evidence["detail_items"] != len(canonical_items):
                raise DecisionSummaryError(f"{detail_context}: detail-item count mismatch")
            detail_by_id = {item["evidence_id"]: item for item in canonical_items}
            dimension_sections = {
                "role": {"ROLE", "ROUTE"},
                "queue": {"QUEUE"},
                "current_condition": {"CONDITION"},
            }
            dimension_ids = evidence["evidence_ids_by_dimension"]
            for dimension, allowed_sections in dimension_sections.items():
                ids = dimension_ids[dimension]
                missing = set(ids) - set(detail_by_id)
                if missing:
                    raise DecisionSummaryError(
                        f"{detail_context}: missing {dimension} evidence details {sorted(missing)}"
                    )
                if any(detail_by_id[evidence_id]["status"] == "UNOBSERVED" for evidence_id in ids):
                    raise DecisionSummaryError(
                        f"{detail_context}: {dimension} evidence references UNOBSERVED detail"
                    )
                if any(
                    detail_by_id[evidence_id]["section"] not in allowed_sections
                    for evidence_id in ids
                ):
                    raise DecisionSummaryError(
                        f"{detail_context}: {dimension} evidence detail section mismatch"
                    )
            if any(
                detail_by_id[evidence_id]["status"] != horse["current_condition_fit"]["source_status"]
                for evidence_id in dimension_ids["current_condition"]
            ):
                raise DecisionSummaryError(
                    f"{detail_context}: condition summary/detail status mismatch"
                )

    race_ids = [race["race_id"] for race in races]
    display_orders = [race["display_order"] for race in races]
    if len(race_ids) != len(set(race_ids)) or display_orders != sorted(display_orders) or len(display_orders) != len(set(display_orders)):
        raise DecisionSummaryError("output races are not unique display order")
    if require_int(root["runner_count"], "output.runner_count", minimum=1) != total:
        raise DecisionSummaryError("output runner_count mismatch")
    if projected is not None:
        expected_hash = sha256_hex(canonical_json_bytes(projected))
        if root["source_projection_sha256"] != expected_hash:
            raise DecisionSummaryError("output source projection digest mismatch")
        source_pairs = [
            (race["race_id"], runner["horse_id"], runner["basics"]["horse_no"])
            for race in projected["races"]
            for runner in race["runners"]
        ]
        output_pairs = [
            (race["race_id"], horse["horse_id"], horse["basics"]["horse_no"])
            for race in races
            for horse in race["horse_summaries"]
        ]
        if source_pairs != output_pairs:
            raise DecisionSummaryError("runner identity universe was not preserved")
        for source_race, output_race in zip(projected["races"], races):
            expected_horses = [
                derive_horse_summary(source_race["race_id"], runner)
                for runner in source_race["runners"]
            ]
            if output_race["horse_summaries"] != expected_horses:
                raise DecisionSummaryError("horse summary differs from the safe source projection")


def _chips(values: Sequence[str], css_class: str = "chip") -> str:
    if not values:
        return '<span class="muted">—</span>'
    return "".join(f'<span class="{css_class}">{html.escape(str(value))}</span>' for value in values)


def _trigger_text(values: Sequence[Mapping[str, Any]]) -> str:
    if not values:
        return '<span class="muted">—</span>'
    return "<br>".join(
        f'<span class="trigger"><code>{html.escape(item["code"])}</code> {html.escape(item["description"])}</span>'
        for item in values
    )


def render_group(title: str, values: Sequence[Mapping[str, Any]], css_class: str) -> str:
    if not values:
        body = '<p class="muted">該当なし</p>'
    else:
        body = "".join(
            "<article class=\"review-horse\">"
            f'<h4>{item["horse_no"]} {html.escape(item["horse_name"])}</h4>'
            f'<div class="chips">{_chips(item["reason_codes"])}</div>'
            "</article>"
            for item in values
        )
    return f'<section class="review-group {css_class}"><h3>{html.escape(title)}</h3>{body}</section>'


def render_horse_row(horse: Mapping[str, Any]) -> str:
    basics = horse["basics"]
    role = horse["role_expected_position"]
    evidence = horse["evidence_count"]
    return f"""
      <tr data-horse-id="{html.escape(horse['horse_id'])}">
        <td><strong>{basics['horse_no']} {html.escape(basics['horse_name'])}</strong><br><a href="{html.escape(horse['detail_ref'])}">詳細証拠</a></td>
        <td><span class="classification {horse['classification'].lower()}">{horse['classification']}</span></td>
        <td>{horse['ability_status']['value']} / {horse['ability_status']['availability']}</td>
        <td>{horse['current_condition_fit']['value']}</td>
        <td>{horse['queue_fit']['value']}</td>
        <td>{role['role']} → {role['expected_position']}<br><small>{role['role_range']} / {role['position_flexibility']}</small></td>
        <td>{_trigger_text(horse['upside_triggers'])}</td>
        <td>{_trigger_text(horse['fragility_triggers'])}</td>
        <td>{html.escape(horse['winning_or_in_the_money_world_state'])}</td>
        <td>{html.escape(horse['failure_world_state'])}</td>
        <td>{horse['confidence']['value']}<br><small>upstream未使用</small></td>
        <td>{evidence['total_distinct']} distinct / {evidence['detail_items']} detail</td>
        <td><div class="chips">{_chips(horse['reason_codes'])}</div></td>
      </tr>"""


def render_evidence_detail(detail: Mapping[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f'<td><code>{html.escape(item["evidence_id"])}</code></td>'
        f'<td>{html.escape(item["section"])}</td>'
        f'<td>{html.escape(item["label"])}</td>'
        f'<td>{html.escape(item["value"])}</td>'
        f'<td>{html.escape(item["status"])}</td>'
        f'<td>{html.escape(item["source_note"])}</td>'
        "</tr>"
        for item in detail["items"]
    )
    if not rows:
        rows = '<tr><td colspan="6" class="muted">詳細証拠なし</td></tr>'
    detail_id = detail["detail_ref"].lstrip("#")
    return f"""
      <details class="horse-evidence" id="{html.escape(detail_id)}">
        <summary>{detail['horse_no']} {html.escape(detail['horse_name'])} — Horsecard / Queue / Route evidence</summary>
        <div class="table-wrap"><table><thead><tr><th>ID</th><th>section</th><th>label</th><th>value</th><th>status</th><th>source note</th></tr></thead><tbody>{rows}</tbody></table></div>
      </details>"""


def render_race(race: Mapping[str, Any]) -> str:
    decision = race["decision_summary"]
    world_states = "".join(
        f'<li><code>{html.escape(item["code"])}</code> {html.escape(item["description"])} <span class="muted">対象馬番: {", ".join(str(number) for number in item["horse_nos"])}</span></li>'
        for item in decision["key_world_states"]
    ) or '<li class="muted">観測可能なworld stateなし</li>'
    rows = "".join(render_horse_row(horse) for horse in race["horse_summaries"])
    details = "".join(render_evidence_detail(detail) for detail in race["evidence_details"]).lstrip()
    return f"""
    <article class="race" data-race-id="{html.escape(race['race_id'])}">
      <header class="race-header"><p>Race {race['display_order']} · {html.escape(race['race_id'])}</p><h2>{html.escape(race['race_name'])}</h2><p>{race['runner_count']} runners · official horse-number order</p></header>
      <section class="decision-summary" aria-label="Decision Summary">
        <div class="section-heading"><p class="eyebrow">Decision Summary</p><h2>最初に確認する観測ポイント</h2><p>review priorityのみ。能力順位・結果確率ではありません。</p></div>
        <div class="review-grid">
          {render_group('まず見る馬', decision['primary_review'], 'primary')}
          {render_group('展開次第で見る馬', decision['conditional_review'], 'conditional')}
          {render_group('Fragile / 割引・注意', decision['fragile_or_downgrade'], 'fragile')}
          <section class="review-group world"><h3>Key world states</h3><ul>{world_states}</ul></section>
        </div>
      </section>
      <section class="all-runners" aria-label="All runner summaries">
        <div class="section-heading"><p class="eyebrow">All runners preserved</p><h2>全馬 Decision Summary</h2><p>分類外の馬もINSUFFICIENTとして保持します。行順はreview priorityではなく公式馬番順です。</p></div>
        <div class="table-wrap"><table class="summary-table"><thead><tr><th>馬</th><th>review priority</th><th>ability</th><th>condition</th><th>Queue</th><th>role / expected</th><th>upside trigger</th><th>fragility trigger</th><th>winning / in-the-money world</th><th>failure world</th><th>confidence</th><th>evidence</th><th>reason codes</th></tr></thead><tbody>{rows}</tbody></table></div>
      </section>
      <section class="evidence-section" aria-label="Horsecard evidence details">
        <div class="section-heading"><p class="eyebrow">Expandable evidence</p><h2>Horsecard / Queue / Route 詳細証拠</h2><p>既存詳細を置換せず、各馬の折りたたみ内に保持します。</p></div>
        {details}
      </section>
    </article>"""


def render_decision_summary_html(output: Mapping[str, Any]) -> str:
    validate_decision_summary(output)
    races = "".join(render_race(race) for race in output["races"]).lstrip()
    source_hash = html.escape(output["source_projection_sha256"])
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Decision Summary Layer V0 — Synthetic Observation</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1020; --panel:#151c31; --line:#2b3857; --text:#eef3ff; --muted:#9eabc6; --cyan:#79d9ee; --amber:#ffd27a; --red:#ff8d96; --green:#92e4b8; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.55 system-ui,"Yu Gothic UI",sans-serif; }}
    main {{ width:min(1600px,100%); margin:auto; padding:24px; }} .hero {{ border:1px solid var(--line); border-radius:18px; padding:24px; background:linear-gradient(135deg,#17213b,#10172a); }}
    .hero h1,.race h2 {{ margin:.15rem 0; }} .guardrails {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }}
    .guardrails span,.chip,.classification {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:3px 8px; }}
    .race {{ margin-top:28px; border:1px solid var(--line); border-radius:18px; overflow:hidden; background:#10172a; }} .race-header,.decision-summary,.all-runners,.evidence-section {{ padding:22px; }}
    .race-header {{ background:#19233d; }} .decision-summary {{ border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
    .section-heading {{ margin-bottom:14px; }} .section-heading p {{ margin:.2rem 0; color:var(--muted); }} .eyebrow {{ text-transform:uppercase; letter-spacing:.14em; color:var(--cyan)!important; font-size:12px; }}
    .review-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }} .review-group {{ border:1px solid var(--line); border-radius:14px; padding:14px; background:var(--panel); min-height:150px; }}
    .review-group.primary {{ border-top:3px solid var(--green); }} .review-group.conditional {{ border-top:3px solid var(--amber); }} .review-group.fragile {{ border-top:3px solid var(--red); }} .review-group.world {{ border-top:3px solid var(--cyan); }}
    .review-group h3,.review-horse h4 {{ margin:.15rem 0 .55rem; }} .review-horse {{ border-top:1px solid var(--line); padding:.6rem 0; }} .review-horse:first-of-type {{ border-top:0; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:4px; }} .chip {{ font:11px/1.3 ui-monospace,Consolas,monospace; color:#c6d8ff; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; }} table {{ width:100%; border-collapse:collapse; min-width:1100px; }} th,td {{ text-align:left; vertical-align:top; padding:10px; border-bottom:1px solid var(--line); }} th {{ position:sticky; top:0; background:#1a2540; color:#cad8f5; }}
    .summary-table td {{ min-width:125px; }} .summary-table td:first-child {{ min-width:180px; }} code {{ color:var(--cyan); }} a {{ color:var(--cyan); }} small,.muted {{ color:var(--muted); }}
    .classification.primary_review {{ color:var(--green); }} .classification.conditional_review {{ color:var(--amber); }} .classification.fragile {{ color:var(--red); }}
    .horse-evidence {{ margin:10px 0; border:1px solid var(--line); border-radius:12px; background:var(--panel); }} .horse-evidence summary {{ cursor:pointer; padding:14px; font-weight:700; }} .horse-evidence .table-wrap {{ margin:0 14px 14px; }}
    @media(max-width:1050px) {{ .review-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }} @media(max-width:620px) {{ main {{ padding:10px; }} .review-grid {{ grid-template-columns:1fr; }} .race-header,.decision-summary,.all-runners,.evidence-section {{ padding:14px; }} }}
  </style>
</head>
<body><main>
  <header class="hero">
    <p class="eyebrow">Horse Intelligence / WIN5 · Weekend V0</p>
    <h1>Decision Summary Layer V0</h1>
    <p>全馬の詳細証拠を読む前に、まず見る馬・条件待ち・明示的fragilityを確認するread-only view modelです。</p>
    <div class="guardrails"><span>Review priority only</span><span>能力順位ではありません</span><span>勝率・複勝率ではありません</span><span>Synthetic fixture only</span><span>No trading hooks</span></div>
    <p class="muted">source projection sha256: <code>{source_hash}</code></p>
  </header>
  {races}
</main></body></html>
"""


def build_files(source: Mapping[str, Any]) -> tuple[bytes, bytes]:
    output = build_decision_summary(source)
    json_bytes = canonical_json_bytes(output)
    html_bytes = render_decision_summary_html(output).encode("utf-8")
    return json_bytes, html_bytes


def write_outputs(source: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    json_bytes, html_bytes = build_files(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "decision_summary_v0.json"
    html_path = output_dir / "decision_summary_v0.html"
    json_path.write_bytes(json_bytes)
    html_path.write_bytes(html_bytes)
    return json_path, html_path


def verify_outputs(source: Mapping[str, Any], output_dir: Path) -> None:
    expected_json, expected_html = build_files(source)
    paths = {
        output_dir / "decision_summary_v0.json": expected_json,
        output_dir / "decision_summary_v0.html": expected_html,
    }
    for path, expected in paths.items():
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise DecisionSummaryError(f"cannot read committed output {path}: {exc}") from exc
        if actual != expected:
            raise DecisionSummaryError(f"deterministic output mismatch: {path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--input", required=True, type=Path)
        subparser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = load_json(args.input)
    if args.command == "build":
        json_path, html_path = write_outputs(source, args.output_dir)
        print(json_path)
        print(html_path)
    else:
        verify_outputs(source, args.output_dir)
        print(f"verified {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
