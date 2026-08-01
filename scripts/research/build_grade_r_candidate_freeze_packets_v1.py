from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
HASH_FIELDS = {
    "candidate_freeze_record_hash",
    "packet_file_sha256",
}
TARGET_FORBIDDEN_FIELDS = {
    "current_odds",
    "current_popularity",
    "final_odds",
    "market_rank",
    "official_result",
    "payout",
    "popularity",
    "result",
    "roi",
    "t3_odds",
}


class CandidateContractError(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_time(value: Any, timezone_name: str) -> datetime:
    if value is None or not str(value).strip():
        raise CandidateContractError("FEATURE_SOURCE_TIME_VIOLATION", "timestamp missing")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    zone = ZoneInfo(timezone_name)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"ledger line {line_number} is not an object")
        records.append(value)
    return records


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _horse_sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value).strip()
    try:
        return (0, int(text), text)
    except ValueError:
        return (1, 0, text)


def canonical_pair(horse_a: Any, horse_b: Any) -> tuple[str, str]:
    values = sorted((str(horse_a).strip(), str(horse_b).strip()), key=_horse_sort_key)
    if not values[0] or values[0] == values[1]:
        raise CandidateContractError("PROBABILITY_CONTRACT_VIOLATION", "invalid pair")
    return values[0], values[1]


def canonical_triplet(values: Iterable[Any]) -> tuple[str, str, str]:
    horses = sorted((str(value).strip() for value in values), key=_horse_sort_key)
    if len(horses) != 3 or any(not horse for horse in horses) or len(set(horses)) != 3:
        raise CandidateContractError(
            "PROBABILITY_CONTRACT_VIOLATION", "triplet must contain three horses"
        )
    return horses[0], horses[1], horses[2]


def load_adapter_config(path: Path) -> dict[str, Any]:
    config = load_json_object(path)
    safety = config.get("safety", {})
    for field in (
        "formal_buy",
        "send_order",
        "production_dashboard_write",
        "notification",
        "credential_access",
        "order_module_import",
        "real_data_during_preparation",
        "roi_calculation",
    ):
        if safety.get(field) is not False:
            raise ValueError(f"{field} must remain false")
    if safety.get("stake") != 0:
        raise ValueError("stake must remain zero")
    policy = config.get("candidate_policy", {})
    if policy.get("name") != "WIDE_1_NON_ODDS_COHERENT_TOP3":
        raise ValueError("candidate policy mismatch")
    if policy.get("candidate_substitution_allowed") is not False:
        raise ValueError("candidate substitution must remain false")
    if policy.get("alternative_pair_search_allowed") is not False:
        raise ValueError("alternative pair search must remain false")
    card = config.get("target_card", {})
    expected = [int(value) for value in card.get("expected_race_numbers", [])]
    if expected != list(range(1, 13)):
        raise ValueError("target card must contain race numbers 1 through 12")
    if int(card.get("expected_race_count", 0)) != len(expected):
        raise ValueError("target card count mismatch")
    return config


def validate_target_manifest(
    manifest: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    if manifest.get("experiment_id") != config.get("experiment_id"):
        raise ValueError("target manifest experiment_id mismatch")
    if manifest.get("cohort_id") != config.get("cohort_id"):
        raise ValueError("target manifest cohort_id mismatch")
    if manifest.get("data_class") not in {"synthetic", "real-data"}:
        raise ValueError("target manifest data_class must be synthetic or real-data")
    expected_card = config["target_card"]
    observed_card = manifest.get("target_card", {})
    for field in ("race_date", "venue_code", "meeting_no", "day_no"):
        if str(observed_card.get(field)) != str(expected_card.get(field)):
            raise ValueError(f"target card {field} mismatch")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("target manifest records must be a list")
    expected_numbers = [int(value) for value in expected_card["expected_race_numbers"]]
    race_numbers = [int(record.get("race_no", 0)) for record in records]
    if sorted(race_numbers) != expected_numbers or len(set(race_numbers)) != len(records):
        raise ValueError("target manifest must contain exactly the registered 12 races")
    race_ids = [str(record.get("race_id", "")).strip() for record in records]
    if any(not race_id for race_id in race_ids) or len(set(race_ids)) != len(race_ids):
        raise ValueError("target race IDs must be present and unique")
    for record in records:
        forbidden = sorted(TARGET_FORBIDDEN_FIELDS.intersection(record))
        if forbidden:
            raise ValueError("target record contains forbidden fields: " + ", ".join(forbidden))
        if record.get("target_registered") is not True:
            raise ValueError("every target race must be pre-registered")
        for field in ("scheduled_post_time", "candidate_feature_cutoff_time"):
            if not str(record.get(field, "")).strip():
                raise ValueError(f"target record missing {field}")
    return sorted(records, key=lambda record: int(record["race_no"]))


def assert_real_data_authorized(root: Path, experiment_id: str) -> None:
    registry_path = root / "research" / "REGISTRY.jsonl"
    events = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("experiment_id") == experiment_id
    ]
    if not events:
        raise ValueError("real-data execution has no registry event")
    latest = events[-1]
    if latest.get("status") != "running":
        raise ValueError("real-data execution requires RUNNING status")
    if latest.get("real_data_execution_allowed") is not True:
        raise ValueError("real-data execution is not authorized")
    if not str(latest.get("run_scope_digest", "")).strip():
        raise ValueError("real-data execution requires a bound run scope")
    if latest.get("formal_buy") is not False or latest.get("send_order") is not False:
        raise ValueError("registry safety flags are not fail-closed")
    if latest.get("stake") != 0:
        raise ValueError("registry stake must remain zero")


def load_feature_source_records(path: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json_object(path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("feature source manifest records must be a list")
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("feature source record must be an object")
        race_id = str(record.get("race_id", "")).strip()
        if not race_id or race_id in output:
            raise ValueError("feature source race_id must be present and unique")
        claimed_hash = str(record.get("source_record_hash", "")).strip()
        hash_payload = {key: value for key, value in record.items() if key != "source_record_hash"}
        if not _is_sha256(claimed_hash) or canonical_digest(hash_payload) != claimed_hash:
            record = dict(record)
            record["_source_record_hash_valid"] = False
        else:
            record = dict(record)
            record["_source_record_hash_valid"] = True
        output[race_id] = record
    return output


def load_top3_feature_rows(path: Path) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            grouped[str(row.get("race_id", "")).strip()].append(dict(row))
    return headers, dict(grouped)


def _first_value(row: dict[str, str], candidates: Iterable[str]) -> str:
    for candidate in candidates:
        value = str(row.get(candidate, "")).strip()
        if value:
            return value
    return ""


def _optional_float(row: dict[str, str], candidates: Iterable[str]) -> float | None:
    raw = _first_value(row, candidates)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _required_float(
    row: dict[str, str], candidates: Iterable[str], field_name: str
) -> float:
    value = _optional_float(row, candidates)
    if value is None:
        raise CandidateContractError(
            "CANDIDATE_SOURCE_NOT_READY", f"missing/non-numeric runner feature {field_name}"
        )
    return value


def _percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], _horse_sort_key(item[0])))
    result: dict[str, float] = {}
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        percentile = average_rank / len(ordered)
        for index in range(position, end):
            result[ordered[index][0]] = percentile
        position = end
    return result


def _derive_expected_pace(rows: list[dict[str, str]]) -> str:
    first = rows[0]
    pressure = _optional_float(first, ["race_early_pressure_score"])
    collapse = _optional_float(first, ["race_pace_collapse_risk"])
    slow = _optional_float(first, ["race_slow_pace_risk"])
    if pressure is not None and collapse is not None and slow is not None:
        if pressure >= 0.60 or collapse >= 0.55:
            return "fast"
        if slow >= 0.55 and pressure < 0.45:
            return "slow"
        return "middle"
    fallback = _first_value(first, ["expected_pace"]).lower()
    if fallback:
        return fallback
    raise CandidateContractError(
        "CANDIDATE_SOURCE_NOT_READY", "pace inputs and frozen expected_pace are missing"
    )


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _pair_features(
    left: dict[str, float],
    right: dict[str, float],
    *,
    field_size: int,
    front_runner_count: float,
    race_pressure: float,
    expected_pace: str,
) -> dict[str, float]:
    front_density = _clip(front_runner_count / field_size, 0.0, 1.0)
    pressure = _clip(race_pressure, 0.0, 1.0)
    front_hold = max(0.01, 0.40 + (1.0 - pressure) * 0.35 + (0.20 if "slow" in expected_pace else 0.0))
    pace_collapse = max(0.01, 0.20 + pressure * 0.55 + (0.20 if ("fast" in expected_pace or "high" in expected_pace) else 0.0))
    slow_sprint = max(0.01, 0.25 + (1.0 - pressure) * 0.20 + (0.25 if "slow" in expected_pace else 0.0))
    neutral = 0.25
    denominator = front_hold + pace_collapse + slow_sprint + neutral
    scenario = [
        front_hold / denominator,
        pace_collapse / denominator,
        slow_sprint / denominator,
        neutral / denominator,
    ]
    front_left = _clip(left["front"], 0.0, 1.0)
    front_right = _clip(right["front"], 0.0, 1.0)
    closer_left = _clip(left["closer"], 0.0, 1.0)
    closer_right = _clip(right["closer"], 0.0, 1.0)
    mid_left = _clip(1.0 - front_left - closer_left, 0.0, 1.0)
    mid_right = _clip(1.0 - front_right - closer_right, 0.0, 1.0)
    escape_left = _clip(front_left * left["front_rank_pct"], 0.0, 1.0)
    escape_right = _clip(front_right * right["front_rank_pct"], 0.0, 1.0)
    pair_escape_clash = _clip(
        escape_left * escape_right * pressure * (1.0 + front_density), 0.0, 2.0
    )
    pair_front_clash = _clip(
        front_left * front_right * pressure * (1.0 + front_density), 0.0, 2.0
    )
    pair_clash = max(pair_escape_clash, pair_front_clash)
    fits_left = [
        front_left,
        _clip(0.75 * closer_left + 0.25 * mid_left, 0.0, 1.0),
        _clip(0.70 * closer_left + 0.30 * (1.0 - front_left), 0.0, 1.0),
        _clip(0.45 * front_left + 0.25 * mid_left + 0.30 * closer_left, 0.0, 1.0),
    ]
    fits_right = [
        front_right,
        _clip(0.75 * closer_right + 0.25 * mid_right, 0.0, 1.0),
        _clip(0.70 * closer_right + 0.30 * (1.0 - front_right), 0.0, 1.0),
        _clip(0.45 * front_right + 0.25 * mid_right + 0.30 * closer_right, 0.0, 1.0),
    ]
    joint = [left_fit * right_fit for left_fit, right_fit in zip(fits_left, fits_right)]
    joint_fit = sum(weight * value for weight, value in zip(scenario, joint))
    shared_failure = sum(
        weight * (1.0 - left_fit) * (1.0 - right_fit)
        for weight, left_fit, right_fit in zip(scenario, fits_left, fits_right)
    )
    mean_joint = sum(joint) / len(joint)
    scenario_variance = sum((value - mean_joint) ** 2 for value in joint) / len(joint)
    return {
        "pair_joint_fit": joint_fit,
        "pair_clash_score": pair_clash,
        "pair_shared_failure": shared_failure,
        "pair_scenario_variance": scenario_variance,
    }


def build_top3_features_from_runner_rows(
    rows: list[dict[str, str]], bundle: dict[str, Any]
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        race_id = str(row.get("race_id", "")).strip()
        if race_id:
            grouped[race_id].append(row)
    output: dict[str, list[dict[str, str]]] = {}
    for race_id, race_rows in grouped.items():
        by_horse: dict[str, dict[str, str]] = {}
        for row in race_rows:
            horse_id = _first_value(row, ["horse_no", "horse_id"])
            if not horse_id or horse_id in by_horse:
                raise CandidateContractError(
                    "STARTER_UNIVERSE_MISMATCH", f"duplicate/missing runner identity in {race_id}"
                )
            by_horse[horse_id] = row
        runners = sorted(by_horse, key=_horse_sort_key)
        if len(runners) < 3:
            continue
        ai_scores = {
            horse_id: _required_float(by_horse[horse_id], ["ai_score"], "ai_score")
            for horse_id in runners
        }
        ai_ranks = {
            horse_id: _required_float(by_horse[horse_id], ["ai_rank"], "ai_rank")
            for horse_id in runners
        }
        primary_strength = _percentile_ranks(ai_scores)
        denominator = max(1.0, len(runners) - 1.0)
        rank_strength = {
            horse_id: _clip(1.0 - (ai_ranks[horse_id] - 1.0) / denominator, 0.0, 1.0)
            for horse_id in runners
        }
        front_values = {
            horse_id: _optional_float(
                by_horse[horse_id],
                ["front_running_tendency_x", "front_running_tendency", "horse_front_run_rate_past5"],
            )
            for horse_id in runners
        }
        closer_values = {
            horse_id: _optional_float(
                by_horse[horse_id],
                ["closing_tendency_x", "closing_tendency", "horse_closer_rate_past5"],
            )
            for horse_id in runners
        }
        front_rank_pct = _percentile_ranks(
            {horse_id: value if value is not None else 0.0 for horse_id, value in front_values.items()}
        )
        expected_pace = _derive_expected_pace(race_rows)
        first = race_rows[0]
        race_pressure = _optional_float(first, ["race_early_pressure_score"])
        if race_pressure is None:
            race_pressure = 0.5
        front_runner_count = _optional_float(
            first, ["race_front_runner_count_x", "race_front_runner_count", "race_need_lead_count"]
        )
        if front_runner_count is None:
            front_runner_count = len(runners) * 0.25
        runner_context = {
            horse_id: {
                "front": front_values[horse_id] if front_values[horse_id] is not None else 0.0,
                "closer": closer_values[horse_id] if closer_values[horse_id] is not None else 0.0,
                "front_rank_pct": front_rank_pct[horse_id],
            }
            for horse_id in runners
        }
        pair_lookup = {
            canonical_pair(left, right): _pair_features(
                runner_context[left],
                runner_context[right],
                field_size=len(runners),
                front_runner_count=front_runner_count,
                race_pressure=race_pressure,
                expected_pace=expected_pace,
            )
            for left, right in itertools.combinations(runners, 2)
        }
        race_output: list[dict[str, str]] = []
        for triplet in itertools.combinations(runners, 3):
            horse_rows = [by_horse[horse_id] for horse_id in triplet]
            floors = [
                _optional_float(row, ["ability_floor_score_5"]) for row in horse_rows
            ]
            m1c_any_core_missing = float(any(value is None for value in floors))
            floor_values = [value if value is not None else 0.5 for value in floors]
            stability = [
                _optional_float(row, ["ability_stability_score_3"]) for row in horse_rows
            ]
            stability_values = [value if value is not None else 0.5 for value in stability]
            recent_values = [
                _optional_float(row, ["recent_weighted_score_3"]) for row in horse_rows
            ]
            recent_values = [value if value is not None else 0.5 for value in recent_values]
            condition_values = [
                _optional_float(row, ["condition_adjusted_recent_ability_score"])
                for row in horse_rows
            ]
            condition_values = [value if value is not None else 0.5 for value in condition_values]
            experience_values = [
                _optional_float(row, ["career_shallow_flag"]) for row in horse_rows
            ]
            experience_values = [value if value is not None else 0.5 for value in experience_values]
            growth_values = [
                _optional_float(row, ["career_growth_zone_flag"]) for row in horse_rows
            ]
            growth_values = [value if value is not None else 0.5 for value in growth_values]
            pair_values = [pair_lookup[canonical_pair(left, right)] for left, right in itertools.combinations(triplet, 2)]
            sorted_floor = sorted(floor_values)
            sorted_stability = sorted(stability_values)
            feature_row: dict[str, str] = {
                "race_id": race_id,
                "horse_id_1": triplet[0],
                "horse_id_2": triplet[1],
                "horse_id_3": triplet[2],
                "sum_primary_strength": str(sum(primary_strength[horse_id] for horse_id in triplet)),
                "triplet_min_ability_floor": str(sorted_floor[0]),
                "triplet_second_min_ability_floor": str(sorted_floor[1]),
                "triplet_mean_ability_floor": str(sum(floor_values) / 3.0),
                "triplet_min_recent_stability": str(sorted_stability[0]),
                "triplet_second_min_recent_stability": str(sorted_stability[1]),
                "triplet_mean_recent_weighted": str(sum(recent_values) / 3.0),
                "triplet_mean_condition_recent": str(sum(condition_values) / 3.0),
                "triplet_experience_risk_count": str(sum(experience_values)),
                "triplet_growth_zone_count": str(sum(growth_values)),
                "m1c_any_core_missing": str(m1c_any_core_missing),
                "triplet_min_pair_joint_fit": str(min(value["pair_joint_fit"] for value in pair_values)),
                "triplet_max_pair_clash": str(max(value["pair_clash_score"] for value in pair_values)),
                "triplet_max_shared_failure": str(max(value["pair_shared_failure"] for value in pair_values)),
                "triplet_max_pair_scenario_variance": str(max(value["pair_scenario_variance"] for value in pair_values)),
            }
            missing_bundle_features = [
                feature for feature in bundle["feature_cols"] if feature not in feature_row
            ]
            if missing_bundle_features:
                raise CandidateContractError(
                    "CANDIDATE_SOURCE_NOT_READY",
                    "runner builder cannot produce bundle feature(s): "
                    + ", ".join(missing_bundle_features),
                )
            race_output.append(feature_row)
        output[race_id] = race_output
    return output


def load_runner_feature_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def validate_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("model_kind") != "linear_top3_set_softmax":
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "unexpected model kind")
    feature_cols = bundle.get("feature_cols")
    means = bundle.get("mean")
    stds = bundle.get("std")
    weights = bundle.get("weights")
    if not all(isinstance(value, list) for value in (feature_cols, means, stds, weights)):
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "bundle arrays missing")
    if not feature_cols or len({str(value) for value in feature_cols}) != len(feature_cols):
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "invalid feature schema")
    if not (len(feature_cols) == len(means) == len(stds) == len(weights)):
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "bundle length mismatch")
    numbers = [float(value) for value in means + stds + weights]
    if not all(math.isfinite(value) for value in numbers):
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "non-finite bundle value")
    if any(float(value) <= 0.0 for value in stds):
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "standard deviation must be positive")
    temperature = float(bundle.get("temperature", 0.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "temperature must be positive")


def _row_utility(row: dict[str, str], bundle: dict[str, Any]) -> float:
    utility = 0.0
    for column, mean, std, weight in zip(
        bundle["feature_cols"], bundle["mean"], bundle["std"], bundle["weights"]
    ):
        raw = row.get(str(column))
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise CandidateContractError(
                "CANDIDATE_SOURCE_NOT_READY", f"missing/non-numeric feature {column}"
            ) from exc
        if not math.isfinite(value):
            raise CandidateContractError(
                "CANDIDATE_SOURCE_NOT_READY", f"non-finite feature {column}"
            )
        utility += ((value - float(mean)) / float(std)) * float(weight)
    return utility


def _softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exponents = [math.exp(value - maximum) for value in scaled]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]


def _logit_offset_probability(probability: float, intercept: float) -> float:
    clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
    logit = math.log(clipped / (1.0 - clipped)) + intercept
    return 1.0 / (1.0 + math.exp(-logit))


def _validate_source_record(
    source: dict[str, Any],
    *,
    target: dict[str, Any],
    bundle_sha256: str,
    feature_schema_hash: str,
    input_snapshot_hash: str,
    timezone_name: str,
) -> list[str]:
    if source.get("_source_record_hash_valid") is not True:
        raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", "source record hash invalid")
    if source.get("source_contract_ok") is not True:
        raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", "source contract failed")
    if str(source.get("inference_bundle_hash", "")) != bundle_sha256:
        raise CandidateContractError(
            "INFERENCE_BUNDLE_HASH_MISMATCH", "source record bundle hash mismatch"
        )
    if str(source.get("feature_schema_hash", "")) != feature_schema_hash:
        raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", "feature schema hash mismatch")
    for field in ("input_snapshot_hash", "starter_universe_hash_at_freeze"):
        if not _is_sha256(source.get(field)):
            raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", f"invalid {field}")
    if str(source.get("input_snapshot_hash", "")) != input_snapshot_hash:
        raise CandidateContractError(
            "CANDIDATE_SOURCE_NOT_READY", "input snapshot hash mismatch"
        )
    target_cutoff = parse_time(target["candidate_feature_cutoff_time"], timezone_name)
    source_cutoff = parse_time(source.get("candidate_feature_cutoff_time"), timezone_name)
    source_max = parse_time(source.get("feature_input_max_source_event_time"), timezone_name)
    source_received = parse_time(source.get("source_received_at"), timezone_name)
    if source_cutoff != target_cutoff or source_max > target_cutoff or source_received > target_cutoff:
        raise CandidateContractError(
            "FEATURE_SOURCE_TIME_VIOLATION", "feature source is later than the candidate cutoff"
        )
    runners_raw = source.get("runner_ids")
    if not isinstance(runners_raw, list):
        raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", "runner_ids missing")
    runners = sorted({str(value).strip() for value in runners_raw if str(value).strip()}, key=_horse_sort_key)
    if len(runners) < 3 or len(runners) != len(runners_raw):
        raise CandidateContractError("STARTER_UNIVERSE_MISMATCH", "invalid runner universe")
    return runners


def derive_candidate(
    rows: list[dict[str, str]],
    *,
    runners: list[str],
    bundle: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    forbidden = {str(value) for value in config["forbidden_candidate_columns"]}
    for row in rows:
        populated = [field for field in forbidden if str(row.get(field, "")).strip()]
        if populated:
            raise CandidateContractError(
                "FORBIDDEN_CANDIDATE_INPUT_COLUMN",
                "forbidden non-empty fields: " + ", ".join(sorted(populated)),
            )
    horse_columns = [str(value) for value in config["candidate_policy"]["horse_id_columns"]]
    observed: dict[tuple[str, str, str], dict[str, str]] = {}
    universe = set(runners)
    for row in rows:
        triplet = canonical_triplet(row.get(column, "") for column in horse_columns)
        if not set(triplet).issubset(universe) or triplet in observed:
            raise CandidateContractError(
                "STARTER_UNIVERSE_MISMATCH", "triplet duplicate or outside runner universe"
            )
        observed[triplet] = row
    expected = set(itertools.combinations(runners, 3))
    if set(observed) != expected:
        raise CandidateContractError(
            "PROBABILITY_CONTRACT_VIOLATION",
            f"expected {len(expected)} triplets, observed {len(observed)}",
        )
    ordered_triplets = sorted(observed, key=lambda values: tuple(_horse_sort_key(v) for v in values))
    utilities = [_row_utility(observed[triplet], bundle) for triplet in ordered_triplets]
    set_probabilities = _softmax(utilities, float(bundle["temperature"]))
    set_mass_error = abs(sum(set_probabilities) - 1.0)
    tolerance = float(config["probability_contract"]["tolerance"])
    if set_mass_error > tolerance:
        raise CandidateContractError("PROBABILITY_CONTRACT_VIOLATION", "Top3 set mass failed")
    wide: dict[tuple[str, str], float] = defaultdict(float)
    for triplet, probability in zip(ordered_triplets, set_probabilities):
        for horse_a, horse_b in itertools.combinations(triplet, 2):
            wide[canonical_pair(horse_a, horse_b)] += probability
    wide_mass_error = abs(sum(wide.values()) - 3.0)
    if wide_mass_error > tolerance or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0 + tolerance
        for value in wide.values()
    ):
        raise CandidateContractError("PROBABILITY_CONTRACT_VIOLATION", "WIDE mass failed")
    ranked = sorted(
        wide.items(),
        key=lambda item: (
            -item[1],
            _horse_sort_key(item[0][0]),
            _horse_sort_key(item[0][1]),
        ),
    )
    pair, top1_probability = ranked[0]
    top2_probability = ranked[1][1] if len(ranked) > 1 else 0.0
    offset = float(config["candidate_policy"]["action_calibrator_offset_intercept"])
    return {
        "candidate_horse_id_1": pair[0],
        "candidate_horse_id_2": pair[1],
        "candidate_pair_key": f"{pair[0]}-{pair[1]}",
        "p_wide_coherent_raw": top1_probability,
        "p_action_calibrated": _logit_offset_probability(top1_probability, offset),
        "top1_top2_margin": top1_probability - top2_probability,
        "confidence_gate_pass": top1_probability
        >= float(config["candidate_policy"]["primary_confidence_threshold"]),
        "set_probability_mass_error": set_mass_error,
        "wide_probability_mass_error": wide_mass_error,
        "probability_contract_ok": True,
    }


def _candidate_record_hash(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key not in HASH_FIELDS}
    return canonical_digest(payload)


def _base_record(
    *,
    config: dict[str, Any],
    target: dict[str, Any],
    source: dict[str, Any] | None,
    bundle_sha256: str,
    feature_schema_hash: str,
    start_time: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "cohort_id": config["cohort_id"],
        "card_id": f"{config['target_card']['race_date']}_{config['target_card']['venue_code']}",
        "race_id": str(target["race_id"]),
        "race_no": int(target["race_no"]),
        "target_registered": True,
        "scheduled_post_time_asof": target["scheduled_post_time"],
        "candidate_feature_cutoff_time": target["candidate_feature_cutoff_time"],
        "candidate_generation_started_at": start_time.isoformat(timespec="milliseconds"),
        "candidate_generation_completed_at": (start_time + timedelta(milliseconds=1)).isoformat(timespec="milliseconds"),
        "candidate_freeze_committed_at": (start_time + timedelta(milliseconds=2)).isoformat(timespec="milliseconds"),
        "inference_bundle_hash": bundle_sha256,
        "feature_schema_hash": feature_schema_hash,
        "candidate_policy_hash": canonical_digest(config["candidate_policy"]),
        "confidence_policy_hash": canonical_digest(
            {
                "signal": "p_wide_coherent_raw",
                "threshold": config["candidate_policy"]["primary_confidence_threshold"],
            }
        ),
        "input_snapshot_hash": str((source or {}).get("input_snapshot_hash", "")),
        "source_record_hash": str((source or {}).get("source_record_hash", "")),
        "feature_input_max_source_event_time": (source or {}).get("feature_input_max_source_event_time"),
        "starter_universe_hash_at_freeze": str((source or {}).get("starter_universe_hash_at_freeze", "")),
        "runner_count": len((source or {}).get("runner_ids", [])),
        "candidate_uses_odds": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def build_candidate_record(
    *,
    config: dict[str, Any],
    target: dict[str, Any],
    source: dict[str, Any] | None,
    rows: list[dict[str, str]],
    bundle: dict[str, Any],
    bundle_sha256: str,
    feature_schema_hash: str,
    input_snapshot_hash: str,
    start_time: datetime,
) -> dict[str, Any]:
    base = _base_record(
        config=config,
        target=target,
        source=source,
        bundle_sha256=bundle_sha256,
        feature_schema_hash=feature_schema_hash,
        start_time=start_time,
    )
    try:
        if source is None or not rows:
            raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", "source or feature rows missing")
        runners = _validate_source_record(
            source,
            target=target,
            bundle_sha256=bundle_sha256,
            feature_schema_hash=feature_schema_hash,
            input_snapshot_hash=input_snapshot_hash,
            timezone_name=config["timezone"],
        )
        candidate = derive_candidate(rows, runners=runners, bundle=bundle, config=config)
        base.update(candidate)
        base.update(
            {
                "record_status": "CANDIDATE_READY",
                "candidate_freeze_contract_ok": True,
                "failure_reason_codes": [],
            }
        )
    except CandidateContractError as exc:
        base.update(
            {
                "record_status": "FAILED",
                "candidate_freeze_contract_ok": False,
                "failure_reason_codes": [exc.reason],
                "failure_detail": exc.detail,
                "candidate_horse_id_1": "",
                "candidate_horse_id_2": "",
                "candidate_pair_key": "",
                "p_wide_coherent_raw": None,
                "p_action_calibrated": None,
                "top1_top2_margin": None,
                "confidence_gate_pass": False,
                "set_probability_mass_error": None,
                "wide_probability_mass_error": None,
                "probability_contract_ok": False,
            }
        )
    base["candidate_freeze_record_hash"] = _candidate_record_hash(base)
    return base


def _write_or_verify_packet(packet_path: Path, record: dict[str, Any]) -> str:
    if packet_path.exists():
        existing = load_json_object(packet_path)
        if existing != record:
            raise ValueError(f"immutable candidate packet differs: {packet_path}")
    else:
        write_json_atomic(packet_path, record)
    persisted = load_json_object(packet_path)
    claimed_hash = str(persisted.get("candidate_freeze_record_hash", ""))
    if claimed_hash != _candidate_record_hash(persisted):
        raise ValueError(f"candidate packet hash verification failed: {packet_path}")
    return file_sha256(packet_path)


def _idempotency_key(config: dict[str, Any], race_id: str) -> str:
    return canonical_digest(
        {
            "cohort_id": config["cohort_id"],
            "event_type": "candidate_freeze_persist_ack",
            "experiment_id": config["experiment_id"],
            "race_id": race_id,
        }
    )


def _verify_ledger_record(record: dict[str, Any], output_dir: Path) -> None:
    if record.get("formal_buy") is not False or record.get("send_order") is not False:
        raise ValueError("unsafe ledger row")
    if record.get("stake") != 0 or record.get("candidate_uses_odds") is not False:
        raise ValueError("unsafe stake or odds flag")
    packet_path = output_dir / str(record.get("packet_path", ""))
    if not packet_path.is_file() or file_sha256(packet_path) != record.get("packet_file_sha256"):
        raise ValueError("ledger packet hash verification failed")
    packet = load_json_object(packet_path)
    if packet.get("candidate_freeze_record_hash") != record.get("candidate_freeze_record_hash"):
        raise ValueError("ledger candidate record hash mismatch")


def _build_summary(
    *,
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    relevant = [
        record
        for record in ledger
        if record.get("experiment_id") == config["experiment_id"]
        and record.get("cohort_id") == config["cohort_id"]
    ]
    target_ids = {str(target["race_id"]) for target in targets}
    counts = Counter(str(record.get("race_id", "")) for record in relevant)
    duplicates = sum(max(0, value - 1) for value in counts.values())
    observed = target_ids.intersection(counts)
    ready = sum(record.get("record_status") == "CANDIDATE_READY" for record in relevant)
    failures = sum(record.get("record_status") == "FAILED" for record in relevant)
    unsafe = sum(
        record.get("formal_buy") is not False
        or record.get("send_order") is not False
        or record.get("stake") != 0
        or record.get("candidate_uses_odds") is not False
        for record in relevant
    )
    missing = sorted(target_ids.difference(observed))
    if duplicates or unsafe or missing:
        status = "INVALID"
    elif ready == len(targets):
        status = "PASS"
    else:
        status = "IN_PROGRESS"
    reasons = Counter(
        reason
        for record in relevant
        for reason in record.get("failure_reason_codes", [])
    )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "cohort_id": config["cohort_id"],
        "status": status,
        "expected_target_rows": len(targets),
        "recorded_target_rows": len(observed),
        "candidate_ready_rows": ready,
        "failed_rows": failures,
        "candidate_freeze_packet_ledger_completeness": (
            len(observed) / len(targets) if targets else 0.0
        ),
        "missing_target_race_ids": missing,
        "duplicate_packet_rows": duplicates,
        "unsafe_rows": unsafe,
        "failure_reason_counts": dict(sorted(reasons.items())),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "roi_calculated": False,
    }


def run_adapter(
    *,
    target_manifest_path: Path,
    feature_source_manifest_path: Path,
    top3_feature_csv_path: Path | None,
    runner_feature_csv_path: Path | None,
    inference_bundle_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    now: datetime,
    execution_mode: str,
) -> dict[str, Any]:
    target_manifest = load_json_object(target_manifest_path)
    if target_manifest.get("data_class") != execution_mode:
        raise ValueError("execution mode and target manifest data_class mismatch")
    targets = validate_target_manifest(target_manifest, config)
    source_records = load_feature_source_records(feature_source_manifest_path)
    bundle_sha256 = file_sha256(inference_bundle_path)
    bundle = load_json_object(inference_bundle_path)
    bundle_error: CandidateContractError | None = None
    try:
        validate_bundle(bundle)
    except CandidateContractError as exc:
        bundle_error = exc
    if (top3_feature_csv_path is None) == (runner_feature_csv_path is None):
        raise ValueError("provide exactly one of top3_feature_csv_path or runner_feature_csv_path")
    input_path = top3_feature_csv_path or runner_feature_csv_path
    assert input_path is not None
    input_snapshot_hash = file_sha256(input_path)
    if top3_feature_csv_path is not None:
        _headers, feature_rows = load_top3_feature_rows(top3_feature_csv_path)
    else:
        try:
            feature_rows = build_top3_features_from_runner_rows(
                load_runner_feature_rows(input_path), bundle
            )
        except CandidateContractError as exc:
            feature_rows = {}
            bundle_error = bundle_error or exc
    if execution_mode == "real-data":
        expected = str(config["bundle_contract"]["production_bundle_sha256"])
        if bundle_sha256 != expected:
            bundle_error = CandidateContractError(
                "INFERENCE_BUNDLE_HASH_MISMATCH", "production bundle hash mismatch"
            )
        if bundle.get("candidate_policy") != config["bundle_contract"]["candidate_policy"]:
            bundle_error = CandidateContractError(
                "INFERENCE_BUNDLE_INVALID", "production candidate policy mismatch"
            )
    feature_schema_hash = canonical_digest(bundle.get("feature_cols", []))
    packets_dir = output_dir / "packets"
    ledger_path = output_dir / "candidate_freeze_ledger.jsonl"
    summary_path = output_dir / "candidate_freeze_summary.json"
    existing = read_jsonl(ledger_path)
    existing_by_race: dict[str, dict[str, Any]] = {}
    for record in existing:
        if record.get("experiment_id") != config["experiment_id"] or record.get("cohort_id") != config["cohort_id"]:
            continue
        race_id = str(record.get("race_id", ""))
        if race_id in existing_by_race:
            raise ValueError("duplicate candidate-freeze ledger row")
        _verify_ledger_record(record, output_dir)
        existing_by_race[race_id] = record

    for index, target in enumerate(targets):
        race_id = str(target["race_id"])
        if race_id in existing_by_race:
            continue
        start_time = now + timedelta(milliseconds=index * 10)
        source = source_records.get(race_id)
        rows = feature_rows.get(race_id, [])
        if bundle_error is not None:
            source_for_failure = dict(source or {})
            source_for_failure["source_contract_ok"] = False
            record = _base_record(
                config=config,
                target=target,
                source=source_for_failure,
                bundle_sha256=bundle_sha256,
                feature_schema_hash=feature_schema_hash,
                input_snapshot_hash=input_snapshot_hash,
                start_time=start_time,
            )
            record.update(
                {
                    "record_status": "FAILED",
                    "candidate_freeze_contract_ok": False,
                    "failure_reason_codes": [bundle_error.reason],
                    "failure_detail": bundle_error.detail,
                    "candidate_horse_id_1": "",
                    "candidate_horse_id_2": "",
                    "candidate_pair_key": "",
                    "p_wide_coherent_raw": None,
                    "p_action_calibrated": None,
                    "top1_top2_margin": None,
                    "confidence_gate_pass": False,
                    "set_probability_mass_error": None,
                    "wide_probability_mass_error": None,
                    "probability_contract_ok": False,
                }
            )
            record["candidate_freeze_record_hash"] = _candidate_record_hash(record)
        else:
            record = build_candidate_record(
                config=config,
                target=target,
                source=source,
                rows=rows,
                bundle=bundle,
                bundle_sha256=bundle_sha256,
                feature_schema_hash=feature_schema_hash,
                input_snapshot_hash=input_snapshot_hash,
                start_time=start_time,
            )
        packet_relative = Path("packets") / f"{race_id}.candidate_freeze.json"
        packet_path = output_dir / packet_relative
        packet_sha = _write_or_verify_packet(packet_path, record)
        ack_time = start_time + timedelta(milliseconds=3)
        ledger_event = {
            "schema_version": 1,
            "event_type": "candidate_freeze_persist_ack",
            "experiment_id": config["experiment_id"],
            "cohort_id": config["cohort_id"],
            "race_id": race_id,
            "race_no": int(target["race_no"]),
            "record_status": record["record_status"],
            "candidate_freeze_contract_ok": record["candidate_freeze_contract_ok"],
            "failure_reason_codes": record["failure_reason_codes"],
            "candidate_pair_key": record["candidate_pair_key"],
            "candidate_freeze_record_hash": record["candidate_freeze_record_hash"],
            "candidate_freeze_persist_ack_at": ack_time.isoformat(timespec="milliseconds"),
            "packet_path": packet_relative.as_posix(),
            "packet_file_sha256": packet_sha,
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
            "idempotency_key": _idempotency_key(config, race_id),
        }
        append_jsonl(ledger_path, ledger_event)
        _verify_ledger_record(ledger_event, output_dir)

    ledger = read_jsonl(ledger_path)
    summary = _build_summary(config=config, targets=targets, ledger=ledger)
    write_json_atomic(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build research-only non-odds Grade-R candidate-freeze packets."
    )
    parser.add_argument("--target-manifest-json", type=Path, required=True)
    parser.add_argument("--feature-source-manifest-json", type=Path, required=True)
    feature_input = parser.add_mutually_exclusive_group(required=True)
    feature_input.add_argument("--top3-feature-csv", type=Path)
    feature_input.add_argument("--runner-feature-csv", type=Path)
    parser.add_argument("--inference-bundle-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "grade_r_candidate_freeze_adapter_v1.json",
    )
    parser.add_argument(
        "--execution-mode", choices=("synthetic", "real-data"), default="synthetic"
    )
    parser.add_argument("--now", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_adapter_config(args.config)
    if args.execution_mode == "real-data":
        assert_real_data_authorized(ROOT, str(config["experiment_id"]))
    now = (
        parse_time(args.now, config["timezone"])
        if args.now
        else datetime.now(ZoneInfo(config["timezone"]))
    )
    summary = run_adapter(
        target_manifest_path=args.target_manifest_json,
        feature_source_manifest_path=args.feature_source_manifest_json,
        top3_feature_csv_path=args.top3_feature_csv,
        runner_feature_csv_path=args.runner_feature_csv,
        inference_bundle_path=args.inference_bundle_json,
        output_dir=args.output_dir,
        config=config,
        now=now,
        execution_mode=args.execution_mode,
    )
    print(canonical_json(summary))
    return 0 if summary["status"] != "INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
