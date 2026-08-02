from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_strict_t3_capture_packet_v1 import (  # noqa: E402
    JraAccessRestrictionError,
    JraOfficialCaptureError,
    ScheduleContractError,
    assert_real_data_authorized,
    build_schedule_lock_record,
    build_schedule_lock_from_document,
    build_jra_official_capture_packet,
    canonical_digest,
    canonical_json,
    canonical_pair_key,
    candidate_record_digest,
    capture_packet_digest,
    file_sha256,
    load_config,
    load_json_object,
    parse_time,
    quote_attempt_times,
    schedule_record_digest,
    verify_candidate_record,
    verify_capture_packet,
    verify_schedule_lock_record,
    write_json_atomic_immutable,
)
from schedule_only_provider_v1 import (  # noqa: E402
    FileBackedScheduleProvider,
    ScheduleProviderContractError,
    ScheduleProviderUnavailable,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"ledger line {line_number} is not an object: {path}")
        records.append(value)
    return records


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value or "").strip())
    if not str(path):
        raise ValueError("path is missing")
    return path if path.is_absolute() else base_dir / path


def _decision_digest(record: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in record.items()
        if key != "t3_decision_record_hash"
    }
    return canonical_digest(payload)


def _idempotency_key(config: dict[str, Any], race_id: str) -> str:
    return canonical_digest(
        {
            "cohort_id": config["cohort_id"],
            "event_type": "strict_t3_shadow_decision",
            "experiment_id": config["experiment_id"],
            "race_id": race_id,
        }
    )


def verify_decision_record(record: dict[str, Any]) -> None:
    if record.get("formal_buy") is not False:
        raise ValueError("decision formal_buy violation")
    if record.get("send_order") is not False:
        raise ValueError("decision send_order violation")
    if record.get("stake") != 0:
        raise ValueError("decision stake violation")
    if record.get("candidate_uses_odds") is not False:
        raise ValueError("decision candidate odds-firewall violation")
    if record.get("candidate_changed_after_odds") is not False:
        raise ValueError("decision candidate changed after odds")
    if record.get("shadow_action") not in {"NO_BET", "PAPER_READY"}:
        raise ValueError("decision shadow_action is invalid")
    if record.get("t3_decision_record_hash") != _decision_digest(record):
        raise ValueError("decision record hash mismatch")
    if not str(record.get("idempotency_key", "")).strip():
        raise ValueError("decision idempotency key is missing")


def append_decision_jsonl(path: Path, record: dict[str, Any]) -> bool:
    verify_decision_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    marker_dir = path.parent / f".{path.name}.ids"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / str(record["idempotency_key"])
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(str(record["idempotency_key"]))
            handle.flush()
            os.fsync(handle.fileno())
        with path.open("a", encoding="utf-8", newline="\n") as ledger:
            ledger.write(canonical_json(record) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
    except Exception:
        marker.unlink(missing_ok=True)
        raise
    return True


def read_decision_jsonl(path: Path) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    for record in records:
        verify_decision_record(record)
    return records


def _candidate_packet_path(acknowledgement: dict[str, Any], ledger_path: Path) -> Path:
    return _resolve_path(acknowledgement.get("packet_path"), ledger_path.parent)


def load_candidate_population(
    source_manifest_path: Path,
    config: dict[str, Any],
    *,
    enforce_expected_counts: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json_object(source_manifest_path)
    if manifest.get("experiment_id") != config["experiment_id"]:
        raise ValueError("source manifest experiment mismatch")
    if manifest.get("source_experiment_id") != config["source_experiment_id"]:
        raise ValueError("source manifest source experiment mismatch")
    if manifest.get("data_class") not in {"synthetic", "real-data"}:
        raise ValueError("source manifest data_class is invalid")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source manifest sources are missing")

    population: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("source manifest entry must be an object")
        ledger_path = _resolve_path(
            source.get("candidate_ledger_jsonl"), source_manifest_path.parent
        )
        expected_hash = str(source.get("candidate_ledger_sha256", "")).strip()
        if not expected_hash or file_sha256(ledger_path) != expected_hash:
            raise ValueError("candidate source ledger hash mismatch")
        for acknowledgement in _read_jsonl(ledger_path):
            race_id = str(acknowledgement.get("race_id", "")).strip()
            if not race_id or race_id in observed_ids:
                raise ValueError("candidate source race_id is missing or duplicated")
            candidate_path = _candidate_packet_path(acknowledgement, ledger_path)
            candidate = load_json_object(candidate_path)
            packet_sha = file_sha256(candidate_path)
            verify_candidate_record(
                candidate,
                acknowledgement,
                candidate_packet_sha256=packet_sha,
                config=config,
            )
            if str(candidate.get("race_id", "")) != race_id:
                raise ValueError("candidate packet race mismatch")
            population.append(
                {
                    "card_id": str(source.get("card_id", candidate.get("card_id", ""))),
                    "candidate": candidate,
                    "acknowledgement": acknowledgement,
                    "candidate_packet_path": candidate_path,
                    "candidate_packet_sha256": packet_sha,
                }
            )
            observed_ids.add(race_id)

    population.sort(
        key=lambda row: (
            str(row["card_id"]),
            int(row["candidate"].get("race_no", 0)),
            str(row["candidate"].get("race_id", "")),
        )
    )
    if enforce_expected_counts:
        expected = config["population"]
        ready = sum(
            row["candidate"].get("record_status") == "CANDIDATE_READY"
            for row in population
        )
        failed = sum(
            row["candidate"].get("record_status") == "FAILED"
            for row in population
        )
        if len(population) != int(expected["expected_target_rows"]):
            raise ValueError("candidate population does not contain 36 rows")
        if ready != int(expected["expected_candidate_ready_rows"]):
            raise ValueError("candidate population does not contain 25 ready rows")
        if failed != int(expected["expected_source_failure_rows"]):
            raise ValueError("candidate population does not contain 11 source failures")
    return manifest, population


def _base_decision(
    *,
    candidate: dict[str, Any],
    acknowledgement: dict[str, Any],
    card_id: str,
    config: dict[str, Any],
    committed_at: datetime,
    action: str,
    reason: str,
) -> dict[str, Any]:
    pair_key = str(candidate.get("candidate_pair_key", ""))
    record = {
        "schema_version": 1,
        "event_type": "strict_t3_shadow_decision",
        "experiment_id": config["experiment_id"],
        "source_experiment_id": config["source_experiment_id"],
        "cohort_id": config["cohort_id"],
        "card_id": card_id,
        "race_id": str(candidate.get("race_id", "")),
        "race_no": int(candidate.get("race_no", 0)),
        "record_status": "T3_DECISION_COMMITTED",
        "source_candidate_status": str(candidate.get("record_status", "")),
        "ticket_type": "wide",
        "candidate_pair_key": pair_key,
        "candidate_horse_id_1": str(candidate.get("candidate_horse_id_1", "")),
        "candidate_horse_id_2": str(candidate.get("candidate_horse_id_2", "")),
        "candidate_freeze_record_hash": str(
            candidate.get("candidate_freeze_record_hash", "")
        ),
        "candidate_freeze_persist_ack_at": acknowledgement.get(
            "candidate_freeze_persist_ack_at"
        ),
        "candidate_freeze_contract_ok": candidate.get("candidate_freeze_contract_ok")
        is True,
        "candidate_uses_odds": False,
        "candidate_changed_after_odds": False,
        "confidence_gate_pass": candidate.get("confidence_gate_pass") is True,
        "p_action_calibrated": candidate.get("p_action_calibrated"),
        "quote_evaluation_entered": False,
        "measurement_only": False,
        "schedule_record_hash": None,
        "schedule_version": None,
        "scheduled_post_time_used": None,
        "t3_cutoff_time": None,
        "decision_deadline_at": None,
        "odds_join_started_at": None,
        "quote_request_started_at": None,
        "t3_quote_source_event_time": None,
        "t3_quote_received_at": None,
        "t3_quote_selected_asof_time": None,
        "feed_heartbeat_source_event_time": None,
        "feed_heartbeat_received_at": None,
        "feed_sequence_id": None,
        "market_status": None,
        "t3_wide_odds_low": None,
        "t3_wide_odds_high": None,
        "quote_age_seconds": None,
        "quote_ingestion_delay_seconds": None,
        "starter_universe_hash_at_freeze": candidate.get(
            "starter_universe_hash_at_freeze"
        ),
        "starter_universe_hash_at_t3": None,
        "scratch_known_by_t3": None,
        "candidate_pair_t3_quote_valid": False,
        "research_expected_return_low": None,
        "shadow_action": action,
        "no_bet_reason_codes": [] if action == "PAPER_READY" else [reason],
        "decision_reason": reason,
        "t3_decision_committed_at": committed_at.isoformat(timespec="milliseconds"),
        "target_registered": True,
        "post_cutoff_backfill": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "idempotency_key": _idempotency_key(
            config, str(candidate.get("race_id", ""))
        ),
    }
    return record


def _finalize(record: dict[str, Any]) -> dict[str, Any]:
    record["t3_decision_record_hash"] = _decision_digest(record)
    verify_decision_record(record)
    return record


def source_failure_decision(
    *,
    candidate: dict[str, Any],
    acknowledgement: dict[str, Any],
    card_id: str,
    config: dict[str, Any],
    committed_at: datetime,
) -> dict[str, Any]:
    if candidate.get("record_status") != "FAILED":
        raise ValueError("source failure decision requires a failed candidate")
    source_reason = str((candidate.get("failure_reason_codes") or [""])[0])
    decision_reason = {
        "UNSUPPORTED_RACE_TYPE": "NO_BET_UNSUPPORTED_RACE_TYPE",
        "SOURCE_READINESS_DEADLINE_MISSED": (
            "NO_BET_SOURCE_READINESS_DEADLINE_MISSED"
        ),
    }.get(source_reason, "NO_BET_SOURCE_NOT_READY")
    record = _base_decision(
        candidate=candidate,
        acknowledgement=acknowledgement,
        card_id=card_id,
        config=config,
        committed_at=committed_at,
        action="NO_BET",
        reason=decision_reason,
    )
    record["candidate_pair_key"] = ""
    record["candidate_horse_id_1"] = ""
    record["candidate_horse_id_2"] = ""
    record["p_action_calibrated"] = None
    record["confidence_gate_pass"] = False
    return _finalize(record)


def no_capture_decision(
    *,
    candidate: dict[str, Any],
    acknowledgement: dict[str, Any],
    card_id: str,
    config: dict[str, Any],
    committed_at: datetime,
    reason: str = "NO_BET_T3_QUOTE_NOT_AVAILABLE",
) -> dict[str, Any]:
    return _finalize(
        _base_decision(
            candidate=candidate,
            acknowledgement=acknowledgement,
            card_id=card_id,
            config=config,
            committed_at=committed_at,
            action="NO_BET",
            reason=reason,
        )
    )


def evaluate_ready_capture(
    *,
    candidate: dict[str, Any],
    acknowledgement: dict[str, Any],
    candidate_packet_sha256: str,
    capture_packet: dict[str, Any],
    card_id: str,
    config: dict[str, Any],
    committed_at: datetime,
) -> dict[str, Any]:
    record = _base_decision(
        candidate=candidate,
        acknowledgement=acknowledgement,
        card_id=card_id,
        config=config,
        committed_at=committed_at,
        action="NO_BET",
        reason="NO_BET_CAPTURE_PACKET_INVALID",
    )

    def finish(reason: str, *, action: str = "NO_BET") -> dict[str, Any]:
        record["shadow_action"] = action
        record["decision_reason"] = reason
        record["no_bet_reason_codes"] = [] if action == "PAPER_READY" else [reason]
        return _finalize(record)

    try:
        verify_capture_packet(capture_packet, config)
    except (TypeError, ValueError):
        return finish("NO_BET_CAPTURE_PACKET_INVALID")

    if capture_packet.get("data_class") not in {"synthetic", "real-data"}:
        return finish("NO_BET_CAPTURE_PACKET_INVALID")
    if capture_packet.get("packet_status") != "QUOTE_CAPTURED":
        return finish("NO_BET_CAPTURE_PACKET_INVALID")
    if capture_packet.get("quote_evaluation_allowed") is not True:
        return finish("NO_BET_CAPTURE_PACKET_INVALID")
    if str(capture_packet.get("race_id", "")) != str(candidate.get("race_id", "")):
        return finish("NO_BET_CAPTURE_PACKET_IDENTITY_MISMATCH")
    if capture_packet.get("candidate_freeze_record_hash") != candidate.get(
        "candidate_freeze_record_hash"
    ):
        return finish("NO_BET_CANDIDATE_HASH_MISMATCH")
    if capture_packet.get("candidate_packet_file_sha256") != candidate_packet_sha256:
        return finish("NO_BET_CANDIDATE_HASH_MISMATCH")
    try:
        frozen_pair = canonical_pair_key(candidate.get("candidate_pair_key"))
        captured_pair = canonical_pair_key(capture_packet.get("candidate_pair_key"))
    except ValueError:
        return finish("NO_BET_CAPTURE_PACKET_IDENTITY_MISMATCH")
    if frozen_pair != captured_pair or capture_packet.get("ticket_type") != "wide":
        return finish("NO_BET_CAPTURE_PACKET_IDENTITY_MISMATCH")

    schedule = capture_packet.get("schedule_record")
    universe = capture_packet.get("universe_observation")
    quote = capture_packet.get("quote_observation")
    if not all(isinstance(value, dict) for value in (schedule, universe, quote)):
        return finish("NO_BET_CAPTURE_PACKET_INVALID")
    assert isinstance(schedule, dict)
    assert isinstance(universe, dict)
    assert isinstance(quote, dict)
    record["quote_evaluation_entered"] = True

    if str(schedule.get("race_id", "")) != record["race_id"]:
        return finish("NO_BET_SCHEDULE_CONTRACT_FAILURE")
    if schedule.get("schedule_contract_ok") is not True:
        return finish("NO_BET_SCHEDULE_CONTRACT_FAILURE")
    schedule_hash = str(schedule.get("schedule_record_hash", ""))
    if not schedule_hash or schedule_hash != schedule_record_digest(schedule):
        return finish("NO_BET_SCHEDULE_CONTRACT_FAILURE")
    record["schedule_record_hash"] = schedule_hash
    record["schedule_version"] = schedule.get("schedule_version")
    record["scheduled_post_time_used"] = schedule.get("scheduled_post_time_used")

    timezone_name = config["timezone"]
    try:
        post_time = parse_time(schedule.get("scheduled_post_time_used"), timezone_name)
        schedule_source = parse_time(schedule.get("schedule_source_event_time"), timezone_name)
        schedule_received = parse_time(schedule.get("schedule_received_at"), timezone_name)
        schedule_lock = (
            parse_time(schedule.get("schedule_lock_time"), timezone_name)
            if config.get("schedule_contract", {}).get("require_schedule_only_lock")
            is True
            else schedule_received
        )
        freeze_ack = parse_time(
            acknowledgement.get("candidate_freeze_persist_ack_at"), timezone_name
        )
        odds_join = parse_time(quote.get("odds_join_started_at"), timezone_name)
        request = parse_time(quote.get("quote_request_started_at"), timezone_name)
        source_time = parse_time(quote.get("t3_quote_source_event_time"), timezone_name)
        received = parse_time(quote.get("t3_quote_received_at"), timezone_name)
        selected = parse_time(quote.get("t3_quote_selected_asof_time"), timezone_name)
        captured = parse_time(capture_packet.get("capture_packet_created_at"), timezone_name)
        heartbeat_source = parse_time(
            quote.get("feed_heartbeat_source_event_time"), timezone_name
        )
        heartbeat_received = parse_time(
            quote.get("feed_heartbeat_received_at"), timezone_name
        )
    except (TypeError, ValueError):
        return finish("NO_BET_SOURCE_TIME_CONTRACT_FAILURE")

    timing = config["timing"]
    cutoff = post_time - timedelta(
        seconds=int(timing["strict_t3_cutoff_seconds_before_post"])
    )
    poll_open = post_time - timedelta(
        seconds=int(timing["poll_window_opens_seconds_before_post"])
    )
    decision_deadline = post_time - timedelta(
        seconds=int(timing["decision_deadline_seconds_before_post"])
    )
    record["t3_cutoff_time"] = cutoff.isoformat(timespec="milliseconds")
    record["decision_deadline_at"] = decision_deadline.isoformat(timespec="milliseconds")
    record["odds_join_started_at"] = odds_join.isoformat(timespec="milliseconds")
    record["quote_request_started_at"] = request.isoformat(timespec="milliseconds")
    record["t3_quote_source_event_time"] = source_time.isoformat(timespec="milliseconds")
    record["t3_quote_received_at"] = received.isoformat(timespec="milliseconds")
    record["t3_quote_selected_asof_time"] = selected.isoformat(timespec="milliseconds")
    record["feed_heartbeat_source_event_time"] = heartbeat_source.isoformat(
        timespec="milliseconds"
    )
    record["feed_heartbeat_received_at"] = heartbeat_received.isoformat(
        timespec="milliseconds"
    )

    requires_schedule_lock = (
        config.get("schedule_contract", {}).get("require_schedule_only_lock") is True
    )
    if requires_schedule_lock:
        try:
            verify_schedule_lock_record(schedule, candidate=candidate, config=config)
        except (ScheduleContractError, TypeError, ValueError):
            return finish("NO_BET_SCHEDULE_CONTRACT_FAILURE")
        if not schedule_source <= schedule_received <= schedule_lock < odds_join:
            return finish("NO_BET_SCHEDULE_CONTRACT_FAILURE")
    elif not schedule_source <= schedule_received <= odds_join:
        return finish("NO_BET_SCHEDULE_CONTRACT_FAILURE")
    if any(value > cutoff for value in (source_time, received, selected)):
        return finish("NO_BET_T3_QUOTE_ASOF_VIOLATION")
    if requires_schedule_lock:
        if not freeze_ack < odds_join <= request <= received <= selected <= captured <= committed_at:
            return finish("NO_BET_SOURCE_TIME_CONTRACT_FAILURE")
    elif not freeze_ack < request <= received <= odds_join <= selected <= captured <= committed_at:
        return finish("NO_BET_SOURCE_TIME_CONTRACT_FAILURE")
    if request < poll_open:
        return finish("NO_BET_POLL_WINDOW_CONTRACT_FAILURE")
    if committed_at > decision_deadline:
        return finish("NO_BET_DECISION_DEADLINE_MISSED")
    if not heartbeat_source <= heartbeat_received <= selected:
        return finish("NO_BET_FEED_HEARTBEAT_FAILURE")

    quote_age = (selected - source_time).total_seconds()
    ingestion_delay = (received - source_time).total_seconds()
    heartbeat_age = (selected - heartbeat_received).total_seconds()
    record["quote_age_seconds"] = quote_age
    record["quote_ingestion_delay_seconds"] = ingestion_delay
    if quote_age < 0 or quote_age > float(timing["maximum_quote_age_seconds"]):
        return finish("NO_BET_T3_QUOTE_STALE")
    if ingestion_delay < 0 or ingestion_delay > float(
        timing["maximum_quote_ingestion_delay_seconds"]
    ):
        return finish("NO_BET_T3_QUOTE_INGESTION_DELAY")
    if heartbeat_age < 0 or heartbeat_age > float(
        timing["maximum_feed_heartbeat_age_seconds"]
    ):
        return finish("NO_BET_FEED_HEARTBEAT_FAILURE")

    if str(universe.get("race_id", "")) != record["race_id"]:
        return finish("NO_BET_STARTER_UNIVERSE_CHANGED")
    freeze_universe = str(candidate.get("starter_universe_hash_at_freeze", ""))
    t3_universe = str(universe.get("starter_universe_hash_at_t3", ""))
    record["starter_universe_hash_at_t3"] = t3_universe
    record["scratch_known_by_t3"] = universe.get("scratch_known_by_t3")
    if universe.get("scratch_known_by_t3") is True:
        return finish("NO_BET_SCRATCH_KNOWN_BY_T3")
    if (
        not freeze_universe
        or not t3_universe
        or freeze_universe != t3_universe
        or universe.get("starter_universe_unchanged") is not True
    ):
        return finish("NO_BET_STARTER_UNIVERSE_CHANGED")

    try:
        quote_pair = canonical_pair_key(quote.get("quote_pair_key"))
    except ValueError:
        return finish("NO_BET_EXACT_CANDIDATE_QUOTE_INVALID")
    if (
        str(quote.get("race_id", "")) != record["race_id"]
        or str(quote.get("ticket_type", "")).lower() != "wide"
        or quote_pair != frozen_pair
    ):
        return finish("NO_BET_EXACT_CANDIDATE_QUOTE_INVALID")
    if quote.get("quote_unique") is not True:
        return finish("NO_BET_EXACT_CANDIDATE_QUOTE_AMBIGUOUS")
    market_status = str(quote.get("market_status", "")).upper()
    record["market_status"] = market_status
    record["feed_sequence_id"] = quote.get("feed_sequence_id")
    if market_status not in set(config["quote_contract"]["allowed_market_statuses"]):
        return finish("NO_BET_MARKET_NOT_OPEN")
    try:
        odds_low = float(quote.get("t3_wide_odds_low"))
        odds_high = float(quote.get("t3_wide_odds_high"))
    except (TypeError, ValueError):
        return finish("NO_BET_EXACT_CANDIDATE_QUOTE_INVALID")
    if (
        not math.isfinite(odds_low)
        or not math.isfinite(odds_high)
        or odds_low <= 0
        or odds_high < odds_low
    ):
        return finish("NO_BET_EXACT_CANDIDATE_QUOTE_INVALID")
    record["t3_wide_odds_low"] = odds_low
    record["t3_wide_odds_high"] = odds_high
    record["candidate_pair_t3_quote_valid"] = True

    probability = float(candidate["p_action_calibrated"])
    expected_return = probability * odds_low
    record["research_expected_return_low"] = expected_return
    if candidate.get("confidence_gate_pass") is not True:
        record["measurement_only"] = True
        return finish("NO_BET_CONFIDENCE_GATE_FAILED")
    minimum_er = float(
        config["decision_policy"]["minimum_research_expected_return_low"]
    )
    if expected_return < minimum_er:
        return finish("NO_BET_VALUE_BELOW_THRESHOLD")
    return finish("STRICT_T3_CONTRACTS_PASSED", action="PAPER_READY")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _summary(
    *,
    config: dict[str, Any],
    population: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    pending: list[str],
) -> dict[str, Any]:
    target_ids = {str(row["candidate"]["race_id"]) for row in population}
    relevant = [row for row in decisions if str(row.get("race_id", "")) in target_ids]
    counts = Counter(str(row.get("race_id", "")) for row in relevant)
    duplicates = sum(max(0, count - 1) for count in counts.values())
    observed = target_ids.intersection(counts)
    missing = sorted(target_ids.difference(observed))
    failed_candidate_ids = {
        str(row["candidate"]["race_id"])
        for row in population
        if row["candidate"].get("record_status") == "FAILED"
    }
    source_failures = [
        row for row in relevant if str(row.get("race_id", "")) in failed_candidate_ids
    ]
    source_failures_entering_quote = sum(
        row.get("quote_evaluation_entered") is True for row in source_failures
    )
    candidate_changes = sum(
        row.get("candidate_changed_after_odds") is True for row in relevant
    )
    unsafe = sum(
        row.get("formal_buy") is not False
        or row.get("send_order") is not False
        or row.get("stake") != 0
        for row in relevant
    )
    action_counts = Counter(str(row.get("shadow_action", "")) for row in relevant)
    reason_counts = Counter(str(row.get("decision_reason", "")) for row in relevant)
    invalid = bool(
        duplicates or candidate_changes or unsafe or source_failures_entering_quote
    )
    if invalid:
        status = "INVALID"
    elif not missing:
        status = "PASS"
    else:
        status = "IN_PROGRESS"
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "cohort_id": config["cohort_id"],
        "status": status,
        "expected_target_rows": len(population),
        "recorded_target_rows": len(observed),
        "strict_t3_shadow_ledger_contract_completeness": (
            len(observed) / len(population) if population else 0.0
        ),
        "candidate_ready_rows": sum(
            row["candidate"].get("record_status") == "CANDIDATE_READY"
            for row in population
        ),
        "source_failure_rows": sum(
            row["candidate"].get("record_status") == "FAILED" for row in population
        ),
        "source_failure_rows_preserved": len(source_failures),
        "source_failure_rows_entering_quote_evaluation": source_failures_entering_quote,
        "quote_evaluation_rows": sum(
            row.get("quote_evaluation_entered") is True for row in relevant
        ),
        "measurement_only_rows": sum(
            row.get("measurement_only") is True for row in relevant
        ),
        "missing_target_race_ids": missing,
        "pending_target_race_ids": sorted(set(pending)),
        "duplicate_decision_rows": duplicates,
        "candidate_changes_after_odds": candidate_changes,
        "unsafe_output_rows": unsafe,
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "roi_calculated": False,
        "production_dashboard_writes": 0,
    }


def _schedule_lock_path(capture_packet_dir: Path, race_id: str) -> Path:
    return capture_packet_dir / f"{race_id}.schedule_lock.json"


def _capture_failure_path(capture_packet_dir: Path, race_id: str) -> Path:
    return capture_packet_dir / f"{race_id}.capture_failure.json"


def _failure_decision_reason(path: Path) -> str:
    if not path.is_file():
        return "NO_BET_T3_QUOTE_NOT_AVAILABLE"
    marker = load_json_object(path)
    reason = str(marker.get("capture_failure_reason", ""))
    if reason.startswith("NO_BET_"):
        return reason
    return {
        "PUBLIC_SOURCE_ACCESS_RESTRICTED": "NO_BET_T3_QUOTE_NOT_AVAILABLE",
        "PUBLIC_QUOTE_UNAVAILABLE": "NO_BET_T3_QUOTE_NOT_AVAILABLE",
    }.get(reason, "NO_BET_T3_QUOTE_NOT_AVAILABLE")


def run_shadow_decisions(
    *,
    source_manifest_path: Path,
    capture_packet_dir: Path,
    decision_ledger_path: Path,
    summary_path: Path,
    config: dict[str, Any],
    now: datetime,
    enforce_expected_counts: bool = True,
) -> dict[str, Any]:
    _manifest, population = load_candidate_population(
        source_manifest_path, config, enforce_expected_counts=enforce_expected_counts
    )
    existing = read_decision_jsonl(decision_ledger_path)
    existing_ids = {
        str(row.get("race_id", ""))
        for row in existing
        if row.get("experiment_id") == config["experiment_id"]
        and row.get("cohort_id") == config["cohort_id"]
    }
    if len(existing_ids) != len(
        [
            row
            for row in existing
            if row.get("experiment_id") == config["experiment_id"]
            and row.get("cohort_id") == config["cohort_id"]
        ]
    ):
        raise ValueError("duplicate decision race_id in existing ledger")
    pending: list[str] = []
    cutoff_seconds = int(config["timing"]["strict_t3_cutoff_seconds_before_post"])

    for row in population:
        candidate = row["candidate"]
        acknowledgement = row["acknowledgement"]
        race_id = str(candidate["race_id"])
        if race_id in existing_ids:
            continue
        if candidate.get("record_status") == "FAILED":
            decision = source_failure_decision(
                candidate=candidate,
                acknowledgement=acknowledgement,
                card_id=row["card_id"],
                config=config,
                committed_at=now,
            )
            append_decision_jsonl(decision_ledger_path, decision)
            existing_ids.add(race_id)
            continue

        capture_path = capture_packet_dir / f"{race_id}.strict_t3_capture.json"
        requires_schedule_lock = (
            config.get("schedule_contract", {}).get("require_schedule_only_lock")
            is True
        )
        schedule_path = _schedule_lock_path(capture_packet_dir, race_id)
        if requires_schedule_lock:
            if not schedule_path.is_file():
                failure_path = _capture_failure_path(capture_packet_dir, race_id)
                if failure_path.is_file():
                    decision = no_capture_decision(
                        candidate=candidate,
                        acknowledgement=acknowledgement,
                        card_id=row["card_id"],
                        config=config,
                        committed_at=now,
                        reason=_failure_decision_reason(failure_path),
                    )
                    append_decision_jsonl(decision_ledger_path, decision)
                    existing_ids.add(race_id)
                else:
                    pending.append(race_id)
                continue
            schedule_record = load_json_object(schedule_path)
            try:
                verify_schedule_lock_record(
                    schedule_record, candidate=candidate, config=config
                )
            except (ScheduleContractError, TypeError, ValueError):
                decision = no_capture_decision(
                    candidate=candidate,
                    acknowledgement=acknowledgement,
                    card_id=row["card_id"],
                    config=config,
                    committed_at=now,
                    reason="NO_BET_SCHEDULE_CONTRACT_FAILURE",
                )
                append_decision_jsonl(decision_ledger_path, decision)
                existing_ids.add(race_id)
                continue
            if schedule_record.get("schedule_contract_ok") is not True:
                decision = no_capture_decision(
                    candidate=candidate,
                    acknowledgement=acknowledgement,
                    card_id=row["card_id"],
                    config=config,
                    committed_at=now,
                    reason="NO_BET_SCHEDULE_CONTRACT_FAILURE",
                )
                append_decision_jsonl(decision_ledger_path, decision)
                existing_ids.add(race_id)
                continue
            post_time = parse_time(
                schedule_record.get("scheduled_post_time_used"), config["timezone"]
            )
        else:
            post_time = parse_time(
                candidate.get("scheduled_post_time_asof"), config["timezone"]
            )
        cutoff = post_time - timedelta(seconds=cutoff_seconds)
        if capture_path.is_file() and now >= cutoff:
            capture = load_json_object(capture_path)
            decision = evaluate_ready_capture(
                candidate=candidate,
                acknowledgement=acknowledgement,
                candidate_packet_sha256=row["candidate_packet_sha256"],
                capture_packet=capture,
                card_id=row["card_id"],
                config=config,
                committed_at=now,
            )
            append_decision_jsonl(decision_ledger_path, decision)
            existing_ids.add(race_id)
            continue
        if now < cutoff:
            pending.append(race_id)
            continue
        decision = no_capture_decision(
            candidate=candidate,
            acknowledgement=acknowledgement,
            card_id=row["card_id"],
            config=config,
            committed_at=now,
            reason=_failure_decision_reason(
                _capture_failure_path(capture_packet_dir, race_id)
            ),
        )
        append_decision_jsonl(decision_ledger_path, decision)
        existing_ids.add(race_id)

    decisions = read_decision_jsonl(decision_ledger_path)
    summary = _summary(
        config=config, population=population, decisions=decisions, pending=pending
    )
    _write_json_atomic(summary_path, summary)
    return summary


def _capture_failure_marker(
    *,
    path: Path,
    candidate: dict[str, Any],
    reason: str,
    observed_at: datetime,
    config: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "cohort_id": config["cohort_id"],
        "race_id": str(candidate["race_id"]),
        "candidate_freeze_record_hash": candidate["candidate_freeze_record_hash"],
        "capture_failure_reason": reason,
        "observed_at": observed_at.isoformat(timespec="milliseconds"),
        "candidate_uses_odds": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    payload["capture_failure_record_hash"] = canonical_digest(payload)
    write_json_atomic_immutable(path, payload)


def _quote_attempt_ledger_path(capture_packet_dir: Path, race_id: str) -> Path:
    return capture_packet_dir / f"{race_id}.quote_attempts.jsonl"


def _quote_attempt_digest(record: dict[str, Any]) -> str:
    return canonical_digest(
        {
            key: value
            for key, value in record.items()
            if key != "quote_attempt_record_hash"
        }
    )


def append_quote_attempt_jsonl(path: Path, record: dict[str, Any]) -> bool:
    if record.get("formal_buy") is not False or record.get("send_order") is not False:
        raise ValueError("quote attempt safety violation")
    if record.get("stake") != 0 or record.get("candidate_uses_odds") is not False:
        raise ValueError("quote attempt odds or stake violation")
    if record.get("quote_attempt_record_hash") != _quote_attempt_digest(record):
        raise ValueError("quote attempt hash mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    marker_dir = path.parent / f".{path.name}.ids"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / str(record["idempotency_key"])
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(str(record["idempotency_key"]))
            handle.flush()
            os.fsync(handle.fileno())
        with path.open("a", encoding="utf-8", newline="\n") as ledger:
            ledger.write(canonical_json(record) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
    except Exception:
        marker.unlink(missing_ok=True)
        raise
    return True


def _record_quote_attempt(
    *,
    path: Path,
    candidate: dict[str, Any],
    schedule_record: dict[str, Any],
    attempt_index: int,
    attempted_at: datetime,
    decision_reason: str,
    accepted: bool,
    capture_packet: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    quote = (capture_packet or {}).get("quote_observation") or {}
    record = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "cohort_id": config["cohort_id"],
        "race_id": str(candidate["race_id"]),
        "candidate_freeze_record_hash": candidate["candidate_freeze_record_hash"],
        "schedule_record_hash": schedule_record["schedule_record_hash"],
        "quote_attempt_index": attempt_index,
        "attempted_at": attempted_at.isoformat(timespec="milliseconds"),
        "odds_join_started_at": quote.get("odds_join_started_at"),
        "quote_request_started_at": quote.get("quote_request_started_at"),
        "quote_source_event_time": quote.get("t3_quote_source_event_time"),
        "quote_received_at": quote.get("t3_quote_received_at"),
        "quote_selected_asof_time": quote.get("t3_quote_selected_asof_time"),
        "decision_reason": decision_reason,
        "accepted": accepted,
        "capture_packet_hash": (capture_packet or {}).get("capture_packet_hash"),
        "candidate_uses_odds": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "idempotency_key": canonical_digest(
            {
                "cohort_id": config["cohort_id"],
                "event_type": "strict_t3_quote_attempt",
                "race_id": str(candidate["race_id"]),
                "quote_attempt_index": attempt_index,
            }
        ),
    }
    record["quote_attempt_record_hash"] = _quote_attempt_digest(record)
    append_quote_attempt_jsonl(path, record)
    return record


def _schedule_observation(
    *,
    candidate: dict[str, Any],
    fetch_schedule_document,
    clock,
    config: dict[str, Any],
) -> dict[str, Any]:
    observed = fetch_schedule_document(candidate)
    if isinstance(observed, str):
        received_at = clock()
        document = observed
        source_event_time = received_at
        locked_at = clock()
        source_reference = "schedule_only_callback"
        return build_schedule_lock_from_document(
            candidate=candidate,
            document=document,
            schedule_source_event_time=source_event_time,
            schedule_received_at=received_at,
            schedule_lock_time=locked_at,
            source_reference=source_reference,
            config=config,
        )
    if isinstance(observed, dict):
        document_value = observed.get("document", "")
        if isinstance(document_value, bytes):
            document = document_value.decode("utf-8")
        else:
            document = str(document_value)
        source_event_time = observed.get("source_event_time")
        received_at = observed.get("received_at")
        locked_at = observed.get("locked_at") or clock()
        source_reference = str(
            observed.get("source_reference", "schedule_only_callback")
        )
        if str(observed.get("scheduled_post_time", "")).strip():
            record = build_schedule_lock_record(
                candidate=candidate,
                scheduled_post_time=observed["scheduled_post_time"],
                schedule_source_event_time=source_event_time,
                schedule_received_at=received_at,
                schedule_lock_time=locked_at,
                source_reference=source_reference,
                source_payload_sha256=str(observed.get("source_payload_sha256", "")),
                config=config,
            )
            for field in (
                "schedule_provider_id",
                "schedule_observation_id",
                "provider_status",
            ):
                if str(observed.get(field, "")).strip():
                    record[field] = observed[field]
            record["source_payload_sha256"] = record[
                "schedule_source_payload_sha256"
            ]
            record["source_reference"] = record["schedule_source_reference"]
            record["schedule_record_hash"] = schedule_record_digest(record)
            return record
        if not document.strip():
            raise ScheduleContractError(
                "schedule-only callback returned neither a time nor a document"
            )
        return build_schedule_lock_from_document(
            candidate=candidate,
            document=document,
            schedule_source_event_time=source_event_time,
            schedule_received_at=received_at,
            schedule_lock_time=locked_at,
            source_reference=source_reference,
            config=config,
        )
    raise ScheduleContractError("schedule-only callback returned an invalid payload")


def _quote_attempt_is_eligible(decision: dict[str, Any]) -> bool:
    return (
        decision.get("quote_evaluation_entered") is True
        and decision.get("candidate_pair_t3_quote_valid") is True
        and decision.get("decision_reason")
        not in {
            "NO_BET_CAPTURE_PACKET_INVALID",
            "NO_BET_CAPTURE_PACKET_IDENTITY_MISMATCH",
            "NO_BET_CANDIDATE_HASH_MISMATCH",
            "NO_BET_SCHEDULE_CONTRACT_FAILURE",
            "NO_BET_SOURCE_TIME_CONTRACT_FAILURE",
            "NO_BET_POLL_WINDOW_CONTRACT_FAILURE",
            "NO_BET_T3_QUOTE_ASOF_VIOLATION",
            "NO_BET_T3_QUOTE_STALE",
            "NO_BET_T3_QUOTE_INGESTION_DELAY",
            "NO_BET_FEED_HEARTBEAT_FAILURE",
            "NO_BET_STARTER_UNIVERSE_CHANGED",
            "NO_BET_SCRATCH_KNOWN_BY_T3",
            "NO_BET_EXACT_CANDIDATE_QUOTE_INVALID",
            "NO_BET_EXACT_CANDIDATE_QUOTE_AMBIGUOUS",
            "NO_BET_MARKET_NOT_OPEN",
        }
    )


def _last_attempt_reason(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "NO_BET_T3_QUOTE_NOT_AVAILABLE"
    reason = str(attempts[-1].get("decision_reason", ""))
    return reason if reason.startswith("NO_BET_") else "NO_BET_T3_QUOTE_NOT_AVAILABLE"


def _run_live_jra_card_hardened(
    *,
    manifest: dict[str, Any],
    population: list[dict[str, Any]],
    source_manifest_path: Path,
    capture_packet_dir: Path,
    raw_html_dir: Path,
    decision_ledger_path: Path,
    summary_path: Path,
    config: dict[str, Any],
    fetch_cname,
    fetch_schedule_document,
    clock,
    sleeper,
    max_consecutive_unavailable: int,
    enforce_expected_counts: bool,
) -> dict[str, Any]:
    if fetch_schedule_document is None:
        raise ValueError("EXP012 live worker requires a schedule-only provider")
    data_class = str(manifest.get("data_class", ""))
    consecutive_unavailable = 0

    while True:
        current = clock()
        summary = run_shadow_decisions(
            source_manifest_path=source_manifest_path,
            capture_packet_dir=capture_packet_dir,
            decision_ledger_path=decision_ledger_path,
            summary_path=summary_path,
            config=config,
            now=current,
            enforce_expected_counts=enforce_expected_counts,
        )
        if summary["status"] in {"PASS", "INVALID"}:
            return summary

        decided_ids = {
            str(row.get("race_id", ""))
            for row in read_decision_jsonl(decision_ledger_path)
            if row.get("experiment_id") == config["experiment_id"]
            and row.get("cohort_id") == config["cohort_id"]
        }
        next_event_times: list[datetime] = []
        progressed = False
        for row in population:
            candidate = row["candidate"]
            race_id = str(candidate["race_id"])
            if race_id in decided_ids or candidate.get("record_status") != "CANDIDATE_READY":
                continue
            capture_path = capture_packet_dir / f"{race_id}.strict_t3_capture.json"
            failure_path = _capture_failure_path(capture_packet_dir, race_id)
            schedule_path = _schedule_lock_path(capture_packet_dir, race_id)
            attempt_path = _quote_attempt_ledger_path(capture_packet_dir, race_id)
            if capture_path.is_file() or failure_path.is_file():
                continue

            if not schedule_path.is_file():
                candidate_post = parse_time(
                    candidate.get("scheduled_post_time_asof"), config["timezone"]
                )
                preflight_at = candidate_post - timedelta(
                    seconds=int(
                        config["timing"]["poll_window_opens_seconds_before_post"]
                    )
                )
                now = clock()
                if now < preflight_at:
                    next_event_times.append(preflight_at)
                    continue
                try:
                    schedule_record = _schedule_observation(
                        candidate=candidate,
                        fetch_schedule_document=fetch_schedule_document,
                        clock=clock,
                        config=config,
                    )
                    write_json_atomic_immutable(schedule_path, schedule_record)
                    progressed = True
                except ScheduleProviderUnavailable as exc:
                    _capture_failure_marker(
                        path=failure_path,
                        candidate=candidate,
                        reason="NO_BET_SCHEDULE_SOURCE_UNAVAILABLE",
                        observed_at=clock(),
                        config=config,
                    )
                    print(
                        canonical_json(
                            {
                                "event": "SCHEDULE_SOURCE_UNAVAILABLE",
                                "race_id": race_id,
                                "error": str(exc),
                            }
                        ),
                        flush=True,
                    )
                    progressed = True
                    continue
                except JraAccessRestrictionError as exc:
                    _capture_failure_marker(
                        path=failure_path,
                        candidate=candidate,
                        reason="PUBLIC_SOURCE_ACCESS_RESTRICTED",
                        observed_at=clock(),
                        config=config,
                    )
                    stopped = dict(summary)
                    stopped["status"] = "STOPPED_ACCESS_RESTRICTION"
                    stopped["stop_race_id"] = race_id
                    stopped["stop_reason"] = str(exc)
                    _write_json_atomic(summary_path, stopped)
                    return stopped
                except (
                    ScheduleProviderContractError,
                    ScheduleContractError,
                    TypeError,
                    ValueError,
                ) as exc:
                    _capture_failure_marker(
                        path=failure_path,
                        candidate=candidate,
                        reason="NO_BET_SCHEDULE_CONTRACT_FAILURE",
                        observed_at=clock(),
                        config=config,
                    )
                    print(
                        canonical_json(
                            {
                                "event": "SCHEDULE_PREFLIGHT_FAILED",
                                "race_id": race_id,
                                "error": str(exc),
                            }
                        ),
                        flush=True,
                    )
                    progressed = True
                    continue

            schedule_record = load_json_object(schedule_path)
            try:
                verify_schedule_lock_record(
                    schedule_record, candidate=candidate, config=config
                )
            except (ScheduleContractError, TypeError, ValueError):
                _capture_failure_marker(
                    path=failure_path,
                    candidate=candidate,
                    reason="NO_BET_SCHEDULE_CONTRACT_FAILURE",
                    observed_at=clock(),
                    config=config,
                )
                progressed = True
                continue
            if schedule_record.get("schedule_contract_ok") is not True:
                _capture_failure_marker(
                    path=failure_path,
                    candidate=candidate,
                    reason="NO_BET_SCHEDULE_CONTRACT_FAILURE",
                    observed_at=clock(),
                    config=config,
                )
                progressed = True
                continue

            attempts = _read_jsonl(attempt_path)
            attempt_times = quote_attempt_times(schedule_record, config)
            cutoff = parse_time(
                schedule_record["t3_cutoff_time"], config["timezone"]
            )
            now = clock()
            if len(attempts) >= len(attempt_times) or now >= cutoff:
                _capture_failure_marker(
                    path=failure_path,
                    candidate=candidate,
                    reason=_last_attempt_reason(attempts),
                    observed_at=now,
                    config=config,
                )
                progressed = True
                continue
            attempt_index = len(attempts) + 1
            attempt_at = attempt_times[attempt_index - 1]
            if now < attempt_at:
                next_event_times.append(attempt_at)
                continue

            packet: dict[str, Any] | None = None
            attempt_reason = "NO_BET_T3_QUOTE_NOT_AVAILABLE"
            accepted = False
            try:
                packet = build_jra_official_capture_packet(
                    candidate=candidate,
                    acknowledgement=row["acknowledgement"],
                    candidate_packet_sha256=row["candidate_packet_sha256"],
                    output_raw_html=(
                        raw_html_dir / f"{race_id}.attempt-{attempt_index:02d}.wide.html"
                    ),
                    fetch_cname=fetch_cname,
                    clock=clock,
                    data_class=data_class,
                    config=config,
                    schedule_record=schedule_record,
                )
                packet["quote_attempt_index"] = attempt_index
                packet["capture_packet_hash"] = capture_packet_digest(packet)
                verify_capture_packet(packet, config)
                provisional = evaluate_ready_capture(
                    candidate=candidate,
                    acknowledgement=row["acknowledgement"],
                    candidate_packet_sha256=row["candidate_packet_sha256"],
                    capture_packet=packet,
                    card_id=row["card_id"],
                    config=config,
                    committed_at=clock(),
                )
                attempt_reason = str(provisional["decision_reason"])
                accepted = _quote_attempt_is_eligible(provisional)
                attempt_packet_path = (
                    capture_packet_dir
                    / f"{race_id}.attempt-{attempt_index:02d}.capture.json"
                )
                write_json_atomic_immutable(attempt_packet_path, packet)
                if accepted:
                    write_json_atomic_immutable(capture_path, packet)
                    consecutive_unavailable = 0
            except JraAccessRestrictionError as exc:
                _capture_failure_marker(
                    path=failure_path,
                    candidate=candidate,
                    reason="PUBLIC_SOURCE_ACCESS_RESTRICTED",
                    observed_at=clock(),
                    config=config,
                )
                stopped = dict(summary)
                stopped["status"] = "STOPPED_ACCESS_RESTRICTION"
                stopped["stop_race_id"] = race_id
                stopped["stop_reason"] = str(exc)
                _write_json_atomic(summary_path, stopped)
                return stopped
            except (JraOfficialCaptureError, RuntimeError, TimeoutError) as exc:
                consecutive_unavailable += 1
                attempt_reason = "NO_BET_T3_QUOTE_NOT_AVAILABLE"
                if consecutive_unavailable >= max_consecutive_unavailable:
                    _capture_failure_marker(
                        path=failure_path,
                        candidate=candidate,
                        reason="PUBLIC_QUOTE_UNAVAILABLE",
                        observed_at=clock(),
                        config=config,
                    )
                    stopped = dict(summary)
                    stopped["status"] = "STOPPED_CONSECUTIVE_UNAVAILABLE"
                    stopped["stop_race_id"] = race_id
                    stopped["stop_reason"] = str(exc)
                    _write_json_atomic(summary_path, stopped)
                    return stopped
            _record_quote_attempt(
                path=attempt_path,
                candidate=candidate,
                schedule_record=schedule_record,
                attempt_index=attempt_index,
                attempted_at=now,
                decision_reason=attempt_reason,
                accepted=accepted,
                capture_packet=packet,
                config=config,
            )
            progressed = True
            if not accepted and attempt_index < len(attempt_times):
                next_event_times.append(attempt_times[attempt_index])
            else:
                next_event_times.append(cutoff)

        current = clock()
        summary = run_shadow_decisions(
            source_manifest_path=source_manifest_path,
            capture_packet_dir=capture_packet_dir,
            decision_ledger_path=decision_ledger_path,
            summary_path=summary_path,
            config=config,
            now=current,
            enforce_expected_counts=enforce_expected_counts,
        )
        if summary["status"] in {"PASS", "INVALID"}:
            return summary
        future_events = [value for value in next_event_times if value > current]
        if not future_events:
            if progressed:
                continue
            sleeper(0.25)
            continue
        sleeper(max(0.0, (min(future_events) - current).total_seconds()))


def run_live_jra_card(
    *,
    source_manifest_path: Path,
    capture_packet_dir: Path,
    raw_html_dir: Path,
    decision_ledger_path: Path,
    summary_path: Path,
    config: dict[str, Any],
    fetch_cname,
    clock,
    fetch_schedule_document=None,
    sleeper=time.sleep,
    max_consecutive_unavailable: int = 3,
    enforce_expected_counts: bool = True,
) -> dict[str, Any]:
    manifest, population = load_candidate_population(
        source_manifest_path, config, enforce_expected_counts=enforce_expected_counts
    )
    data_class = str(manifest.get("data_class", ""))
    if data_class not in {"synthetic", "real-data"}:
        raise ValueError("live-card source manifest data class is invalid")
    capture_packet_dir.mkdir(parents=True, exist_ok=True)
    raw_html_dir.mkdir(parents=True, exist_ok=True)
    if config.get("schedule_contract", {}).get("require_schedule_only_lock") is True:
        return _run_live_jra_card_hardened(
            manifest=manifest,
            population=population,
            source_manifest_path=source_manifest_path,
            capture_packet_dir=capture_packet_dir,
            raw_html_dir=raw_html_dir,
            decision_ledger_path=decision_ledger_path,
            summary_path=summary_path,
            config=config,
            fetch_cname=fetch_cname,
            fetch_schedule_document=fetch_schedule_document,
            clock=clock,
            sleeper=sleeper,
            max_consecutive_unavailable=max_consecutive_unavailable,
            enforce_expected_counts=enforce_expected_counts,
        )
    consecutive_unavailable = 0

    while True:
        now = clock()
        summary = run_shadow_decisions(
            source_manifest_path=source_manifest_path,
            capture_packet_dir=capture_packet_dir,
            decision_ledger_path=decision_ledger_path,
            summary_path=summary_path,
            config=config,
            now=now,
            enforce_expected_counts=enforce_expected_counts,
        )
        if summary["status"] in {"PASS", "INVALID"}:
            return summary

        decided_ids = {
            str(row.get("race_id", ""))
            for row in read_decision_jsonl(decision_ledger_path)
            if row.get("experiment_id") == config["experiment_id"]
            and row.get("cohort_id") == config["cohort_id"]
        }
        next_event_times: list[datetime] = []
        captured_in_cycle = False
        for row in population:
            candidate = row["candidate"]
            race_id = str(candidate["race_id"])
            if race_id in decided_ids or candidate.get("record_status") != "CANDIDATE_READY":
                continue
            post_time = parse_time(
                candidate.get("scheduled_post_time_asof"), config["timezone"]
            )
            poll_open = post_time - timedelta(
                seconds=int(config["timing"]["poll_window_opens_seconds_before_post"])
            )
            cutoff = post_time - timedelta(
                seconds=int(config["timing"]["strict_t3_cutoff_seconds_before_post"])
            )
            capture_path = capture_packet_dir / f"{race_id}.strict_t3_capture.json"
            failure_path = capture_packet_dir / f"{race_id}.capture_failure.json"
            if capture_path.is_file() or failure_path.is_file():
                next_event_times.append(cutoff)
                continue
            current = clock()
            if current < poll_open:
                next_event_times.append(poll_open)
                continue
            if current >= cutoff:
                next_event_times.append(current)
                continue
            try:
                packet = build_jra_official_capture_packet(
                    candidate=candidate,
                    acknowledgement=row["acknowledgement"],
                    candidate_packet_sha256=row["candidate_packet_sha256"],
                    output_raw_html=raw_html_dir / f"{race_id}.wide.html",
                    fetch_cname=fetch_cname,
                    clock=clock,
                    data_class=data_class,
                    config=config,
                )
                verify_capture_packet(packet, config)
                write_json_atomic_immutable(capture_path, packet)
                consecutive_unavailable = 0
                captured_in_cycle = True
                print(
                    canonical_json(
                        {
                            "event": "T3_QUOTE_CAPTURED",
                            "race_id": race_id,
                            "candidate_pair_key": candidate["candidate_pair_key"],
                            "formal_buy": False,
                            "send_order": False,
                            "stake": 0,
                        }
                    ),
                    flush=True,
                )
            except JraAccessRestrictionError as exc:
                _capture_failure_marker(
                    path=failure_path,
                    candidate=candidate,
                    reason="PUBLIC_SOURCE_ACCESS_RESTRICTED",
                    observed_at=clock(),
                    config=config,
                )
                stopped = dict(summary)
                stopped["status"] = "STOPPED_ACCESS_RESTRICTION"
                stopped["stop_race_id"] = race_id
                stopped["stop_reason"] = str(exc)
                _write_json_atomic(summary_path, stopped)
                return stopped
            except (JraOfficialCaptureError, RuntimeError, TimeoutError) as exc:
                consecutive_unavailable += 1
                _capture_failure_marker(
                    path=failure_path,
                    candidate=candidate,
                    reason="PUBLIC_QUOTE_UNAVAILABLE",
                    observed_at=clock(),
                    config=config,
                )
                print(
                    canonical_json(
                        {
                            "event": "T3_QUOTE_FETCH_FAILED",
                            "race_id": race_id,
                            "error": str(exc),
                            "consecutive_unavailable": consecutive_unavailable,
                        }
                    ),
                    flush=True,
                )
                if consecutive_unavailable >= max_consecutive_unavailable:
                    stopped = dict(summary)
                    stopped["status"] = "STOPPED_CONSECUTIVE_UNAVAILABLE"
                    stopped["stop_race_id"] = race_id
                    stopped["stop_reason"] = str(exc)
                    _write_json_atomic(summary_path, stopped)
                    return stopped
            next_event_times.append(cutoff)

        current = clock()
        summary = run_shadow_decisions(
            source_manifest_path=source_manifest_path,
            capture_packet_dir=capture_packet_dir,
            decision_ledger_path=decision_ledger_path,
            summary_path=summary_path,
            config=config,
            now=current,
            enforce_expected_counts=enforce_expected_counts,
        )
        if summary["status"] in {"PASS", "INVALID"}:
            return summary
        future_events = [value for value in next_event_times if value > current]
        if not future_events:
            if captured_in_cycle:
                continue
            sleeper(0.25)
            continue
        sleep_seconds = max(0.0, (min(future_events) - current).total_seconds())
        if sleep_seconds:
            print(
                canonical_json(
                    {
                        "event": "WAITING_FOR_NEXT_T3_EVENT",
                        "seconds": round(sleep_seconds, 3),
                        "pending": summary["pending_target_race_ids"],
                    }
                ),
                flush=True,
            )
            sleeper(sleep_seconds)


def build_file_schedule_provider(
    path: Path,
    *,
    config: dict[str, Any],
    clock,
) -> FileBackedScheduleProvider:
    provider_config = config.get("schedule_provider", {})
    if provider_config.get("kind") != "file_backed_jsonl":
        raise ValueError("file-backed schedule provider is not configured")
    return FileBackedScheduleProvider(
        path,
        timezone_name=config["timezone"],
        clock=clock,
        provider_id=str(
            provider_config.get("provider_id", "file_backed_schedule_v1")
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Commit strict-T3 research-only shadow decisions."
    )
    parser.add_argument("--source-manifest-json", type=Path, required=True)
    parser.add_argument("--capture-packet-dir", type=Path, required=True)
    parser.add_argument("--decision-ledger-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "strict_t3_shadow_decision_exp011.json",
    )
    parser.add_argument(
        "--execution-mode", choices=("synthetic", "real-data"), default="synthetic"
    )
    parser.add_argument("--now", default="")
    parser.add_argument("--live-jra-official", action="store_true")
    parser.add_argument("--raw-html-dir", type=Path)
    parser.add_argument(
        "--schedule-observations-jsonl",
        type=Path,
        help="Local append-only schedule observations; required by EXP013 live research mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    manifest = load_json_object(args.source_manifest_json)
    if manifest.get("data_class") != args.execution_mode:
        raise ValueError("execution mode and source manifest data_class mismatch")
    if args.execution_mode == "real-data":
        assert_real_data_authorized(ROOT, config["experiment_id"])
    if args.live_jra_official:
        if args.execution_mode != "real-data":
            raise ValueError("live JRA card mode requires real-data execution")
        if args.now:
            raise ValueError("live JRA card mode does not accept --now")
        if args.raw_html_dir is None:
            raise ValueError("live JRA card mode requires --raw-html-dir")
        import fetch_jra_official_odds as jra_odds

        live_clock = lambda: datetime.now(ZoneInfo(config["timezone"]))
        schedule_provider = None
        if config.get("schedule_contract", {}).get("require_schedule_only_lock") is True:
            if args.schedule_observations_jsonl is None:
                raise ValueError(
                    "hardened live research mode requires --schedule-observations-jsonl"
                )
            schedule_provider = build_file_schedule_provider(
                args.schedule_observations_jsonl,
                config=config,
                clock=live_clock,
            )
        summary = run_live_jra_card(
            source_manifest_path=args.source_manifest_json,
            capture_packet_dir=args.capture_packet_dir,
            raw_html_dir=args.raw_html_dir,
            decision_ledger_path=args.decision_ledger_jsonl,
            summary_path=args.summary_json,
            config=config,
            fetch_cname=lambda cname: jra_odds.post_cname(
                cname, timeout=5.0, retries=0
            ),
            fetch_schedule_document=schedule_provider,
            clock=live_clock,
        )
    else:
        now = (
            parse_time(args.now, config["timezone"])
            if args.now
            else datetime.now(ZoneInfo(config["timezone"]))
        )
        summary = run_shadow_decisions(
            source_manifest_path=args.source_manifest_json,
            capture_packet_dir=args.capture_packet_dir,
            decision_ledger_path=args.decision_ledger_jsonl,
            summary_path=args.summary_json,
            config=config,
            now=now,
        )
    print(canonical_json(summary))
    return 0 if summary["status"] != "INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
