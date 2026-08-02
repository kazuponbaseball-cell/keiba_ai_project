from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
HASH_FIELDS = {"candidate_freeze_record_hash", "packet_file_sha256"}
FORBIDDEN_INPUT_FIELDS = {
    "actual_order_acknowledgement",
    "alternative_pair",
    "credential",
    "current_popularity",
    "final_odds",
    "formal_stake_yen",
    "market_rank",
    "official_result",
    "order_payload",
    "payout",
    "popularity",
    "result",
    "roi",
}


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
        raise ValueError("timestamp is missing")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    zone = ZoneInfo(timezone_name)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


def iso_time(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def canonical_pair_key(value: Any) -> str:
    numbers = re.findall(r"\d+", str(value or ""))
    if len(numbers) != 2:
        raise ValueError("pair key must contain exactly two horse numbers")
    first, second = (int(number) for number in numbers)
    if first < 1 or second < 1 or first == second:
        raise ValueError("pair key must contain two distinct positive horse numbers")
    low, high = sorted((first, second))
    return f"{low}-{high}"


def _recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _recursive_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _recursive_keys(nested)


def assert_no_forbidden_fields(*records: dict[str, Any]) -> None:
    observed = set()
    for record in records:
        observed.update(_recursive_keys(record))
    forbidden = sorted(observed.intersection(FORBIDDEN_INPUT_FIELDS))
    if forbidden:
        raise ValueError("forbidden input field(s): " + ", ".join(forbidden))


def load_config(path: Path) -> dict[str, Any]:
    config = load_json_object(path)
    if config.get("experiment_id") != "EXP-20260802-011":
        raise ValueError("unexpected experiment_id")
    if config.get("source_experiment_id") != "EXP-20260802-010":
        raise ValueError("unexpected source_experiment_id")
    safety = config.get("safety", {})
    for field in (
        "formal_buy",
        "send_order",
        "target_handoff",
        "production_dashboard_write",
        "notification_send",
        "order_path_execution",
        "roi_calculation",
        "real_data_during_preparation",
    ):
        if safety.get(field) is not False:
            raise ValueError(f"{field} must remain false")
    if safety.get("stake") != 0:
        raise ValueError("stake must remain zero")
    contract = config.get("quote_contract", {})
    if contract.get("ticket_type") != "wide":
        raise ValueError("only the frozen WIDE candidate is allowed")
    if contract.get("candidate_substitution_allowed") is not False:
        raise ValueError("candidate substitution must remain false")
    if contract.get("alternative_pair_search_allowed") is not False:
        raise ValueError("alternative pair search must remain false")
    policy = config.get("decision_policy", {})
    if policy.get("threshold_search_allowed") is not False:
        raise ValueError("threshold search must remain false")
    if float(policy.get("minimum_research_expected_return_low", 0)) != 1.5:
        raise ValueError("registered minimum research ER must remain 1.5")
    timing = config.get("timing", {})
    poll = int(timing.get("poll_window_opens_seconds_before_post", 0))
    cutoff = int(timing.get("strict_t3_cutoff_seconds_before_post", 0))
    deadline = int(timing.get("decision_deadline_seconds_before_post", 0))
    if not poll > cutoff > deadline > 0:
        raise ValueError("timing contract must satisfy poll > cutoff > deadline")
    return config


def candidate_record_digest(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key not in HASH_FIELDS}
    return canonical_digest(payload)


def schedule_record_digest(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "schedule_record_hash"}
    return canonical_digest(payload)


def capture_packet_digest(packet: dict[str, Any]) -> str:
    payload = {key: value for key, value in packet.items() if key != "capture_packet_hash"}
    return canonical_digest(payload)


def verify_candidate_record(
    candidate: dict[str, Any],
    acknowledgement: dict[str, Any],
    *,
    candidate_packet_sha256: str,
    config: dict[str, Any],
) -> None:
    assert_no_forbidden_fields(candidate, acknowledgement)
    if candidate.get("experiment_id") != config["source_experiment_id"]:
        raise ValueError("candidate source experiment mismatch")
    if str(acknowledgement.get("race_id", "")) != str(candidate.get("race_id", "")):
        raise ValueError("candidate acknowledgement race mismatch")
    claimed_hash = str(candidate.get("candidate_freeze_record_hash", ""))
    if not claimed_hash or claimed_hash != candidate_record_digest(candidate):
        raise ValueError("candidate record hash mismatch")
    if acknowledgement.get("candidate_freeze_record_hash") != claimed_hash:
        raise ValueError("candidate acknowledgement hash mismatch")
    if acknowledgement.get("packet_file_sha256") != candidate_packet_sha256:
        raise ValueError("candidate packet file hash mismatch")
    if acknowledgement.get("formal_buy") is not False:
        raise ValueError("candidate acknowledgement formal_buy violation")
    if acknowledgement.get("send_order") is not False:
        raise ValueError("candidate acknowledgement send_order violation")
    if acknowledgement.get("stake") != 0:
        raise ValueError("candidate acknowledgement stake violation")
    if acknowledgement.get("candidate_uses_odds") is not False:
        raise ValueError("candidate acknowledgement odds firewall violation")
    parse_time(
        acknowledgement.get("candidate_freeze_persist_ack_at"), config["timezone"]
    )

    status = str(candidate.get("record_status", ""))
    if status == "CANDIDATE_READY":
        if candidate.get("candidate_freeze_contract_ok") is not True:
            raise ValueError("ready candidate freeze contract is false")
        if candidate.get("candidate_uses_odds") is not False:
            raise ValueError("ready candidate odds firewall violation")
        pair_key = canonical_pair_key(candidate.get("candidate_pair_key"))
        horse_pair = canonical_pair_key(
            f"{candidate.get('candidate_horse_id_1', '')}-{candidate.get('candidate_horse_id_2', '')}"
        )
        if pair_key != horse_pair:
            raise ValueError("candidate horse IDs do not match pair key")
        probability = float(candidate.get("p_action_calibrated"))
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("candidate action probability is invalid")
        if candidate.get("failure_reason_codes") not in ([], None):
            raise ValueError("ready candidate has failure reasons")
    elif status == "FAILED":
        expected_reason = config["population"]["source_failure_reason"]
        reasons = candidate.get("failure_reason_codes")
        if not isinstance(reasons, list) or expected_reason not in reasons:
            raise ValueError("failed candidate is not the registered source failure")
        if candidate.get("candidate_freeze_contract_ok") is not False:
            raise ValueError("failed candidate freeze contract must be false")
        if any(
            str(candidate.get(field, "")).strip()
            for field in (
                "candidate_pair_key",
                "candidate_horse_id_1",
                "candidate_horse_id_2",
            )
        ):
            raise ValueError("failed candidate must not contain a candidate pair")
        if candidate.get("p_action_calibrated") is not None:
            raise ValueError("failed candidate must not contain action probability")
    else:
        raise ValueError("candidate record_status is not registered")


def _copy_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return json.loads(canonical_json(record))


def build_capture_packet(
    *,
    candidate: dict[str, Any],
    acknowledgement: dict[str, Any],
    candidate_packet_sha256: str,
    schedule_record: dict[str, Any] | None,
    universe_observation: dict[str, Any] | None,
    quote_observation: dict[str, Any] | None,
    captured_at: Any,
    data_class: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if data_class not in {"synthetic", "real-data"}:
        raise ValueError("data_class must be synthetic or real-data")
    verify_candidate_record(
        candidate,
        acknowledgement,
        candidate_packet_sha256=candidate_packet_sha256,
        config=config,
    )
    captured = parse_time(captured_at, config["timezone"])
    race_id = str(candidate["race_id"])
    ready = candidate["record_status"] == "CANDIDATE_READY"

    if not ready:
        if any(value is not None for value in (schedule_record, universe_observation, quote_observation)):
            raise ValueError("source-failure row must not enter quote evaluation")
        packet = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "source_experiment_id": config["source_experiment_id"],
            "cohort_id": config["cohort_id"],
            "data_class": data_class,
            "packet_status": "SOURCE_NOT_READY",
            "race_id": race_id,
            "race_no": int(candidate.get("race_no", 0)),
            "candidate_freeze_record_hash": candidate["candidate_freeze_record_hash"],
            "candidate_packet_file_sha256": candidate_packet_sha256,
            "candidate_pair_key": "",
            "ticket_type": "wide",
            "confidence_gate_pass": False,
            "p_action_calibrated": None,
            "quote_evaluation_allowed": False,
            "source_failure_reason": config["population"]["source_failure_reason"],
            "candidate_freeze_persist_ack_at": acknowledgement[
                "candidate_freeze_persist_ack_at"
            ],
            "schedule_record": None,
            "universe_observation": None,
            "quote_observation": None,
            "capture_packet_created_at": iso_time(captured),
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
    else:
        if not all(
            isinstance(value, dict)
            for value in (schedule_record, universe_observation, quote_observation)
        ):
            raise ValueError("ready candidate requires schedule, universe, and quote records")
        assert schedule_record is not None
        assert universe_observation is not None
        assert quote_observation is not None
        assert_no_forbidden_fields(schedule_record, universe_observation, quote_observation)
        packet = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "source_experiment_id": config["source_experiment_id"],
            "cohort_id": config["cohort_id"],
            "data_class": data_class,
            "packet_status": "QUOTE_CAPTURED",
            "race_id": race_id,
            "race_no": int(candidate.get("race_no", 0)),
            "candidate_freeze_record_hash": candidate["candidate_freeze_record_hash"],
            "candidate_packet_file_sha256": candidate_packet_sha256,
            "candidate_pair_key": canonical_pair_key(candidate["candidate_pair_key"]),
            "ticket_type": "wide",
            "confidence_gate_pass": candidate.get("confidence_gate_pass") is True,
            "p_action_calibrated": float(candidate["p_action_calibrated"]),
            "quote_evaluation_allowed": True,
            "source_failure_reason": None,
            "candidate_freeze_persist_ack_at": acknowledgement[
                "candidate_freeze_persist_ack_at"
            ],
            "schedule_record": _copy_record(schedule_record),
            "universe_observation": _copy_record(universe_observation),
            "quote_observation": _copy_record(quote_observation),
            "capture_packet_created_at": iso_time(captured),
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
    packet["capture_packet_hash"] = capture_packet_digest(packet)
    return packet


def verify_capture_packet(packet: dict[str, Any], config: dict[str, Any]) -> None:
    assert_no_forbidden_fields(packet)
    if packet.get("experiment_id") != config["experiment_id"]:
        raise ValueError("capture packet experiment mismatch")
    if packet.get("source_experiment_id") != config["source_experiment_id"]:
        raise ValueError("capture packet source experiment mismatch")
    if packet.get("capture_packet_hash") != capture_packet_digest(packet):
        raise ValueError("capture packet hash mismatch")
    if packet.get("candidate_uses_odds") is not False:
        raise ValueError("capture packet odds firewall violation")
    if packet.get("formal_buy") is not False or packet.get("send_order") is not False:
        raise ValueError("capture packet safety violation")
    if packet.get("stake") != 0:
        raise ValueError("capture packet stake violation")
    if packet.get("packet_status") == "SOURCE_NOT_READY":
        if packet.get("quote_evaluation_allowed") is not False:
            raise ValueError("source failure entered quote evaluation")
        if any(
            packet.get(field) is not None
            for field in ("schedule_record", "universe_observation", "quote_observation")
        ):
            raise ValueError("source failure packet contains quote inputs")
    elif packet.get("packet_status") == "QUOTE_CAPTURED":
        if packet.get("quote_evaluation_allowed") is not True:
            raise ValueError("ready capture packet disables quote evaluation")
        canonical_pair_key(packet.get("candidate_pair_key"))
    else:
        raise ValueError("capture packet status is invalid")


def write_json_atomic_immutable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_json_object(path)
        if existing != payload:
            raise ValueError(f"immutable capture packet differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    if load_json_object(path) != payload:
        raise ValueError("capture packet read-back verification failed")


def assert_real_data_authorized(root: Path, experiment_id: str) -> None:
    registry_path = root / "research" / "REGISTRY.jsonl"
    events = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("experiment_id") == experiment_id:
            events.append(event)
    if not events or events[-1].get("status") != "running":
        raise ValueError("real-data execution requires RUNNING registry status")
    latest = events[-1]
    if latest.get("real_data_execution_allowed") is not True:
        raise ValueError("real-data execution is not authorized")
    if not str(latest.get("run_scope_digest", "")).strip():
        raise ValueError("real-data execution requires a bound run scope")
    if latest.get("formal_buy") is not False or latest.get("send_order") is not False:
        raise ValueError("registry safety flags are not fail-closed")
    if latest.get("stake") != 0:
        raise ValueError("registry stake must remain zero")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one immutable strict-T3 research capture packet."
    )
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--candidate-ack-json", type=Path, required=True)
    parser.add_argument("--schedule-json", type=Path)
    parser.add_argument("--universe-json", type=Path)
    parser.add_argument("--quote-json", type=Path)
    parser.add_argument("--captured-at", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "strict_t3_shadow_decision_exp011.json",
    )
    parser.add_argument(
        "--execution-mode", choices=("synthetic", "real-data"), default="synthetic"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.execution_mode == "real-data":
        assert_real_data_authorized(ROOT, config["experiment_id"])
    candidate = load_json_object(args.candidate_json)
    acknowledgement = load_json_object(args.candidate_ack_json)
    packet = build_capture_packet(
        candidate=candidate,
        acknowledgement=acknowledgement,
        candidate_packet_sha256=file_sha256(args.candidate_json),
        schedule_record=load_json_object(args.schedule_json) if args.schedule_json else None,
        universe_observation=(
            load_json_object(args.universe_json) if args.universe_json else None
        ),
        quote_observation=load_json_object(args.quote_json) if args.quote_json else None,
        captured_at=args.captured_at,
        data_class=args.execution_mode,
        config=config,
    )
    verify_capture_packet(packet, config)
    write_json_atomic_immutable(args.output_json, packet)
    print(canonical_json({"output": str(args.output_json), "packet": packet}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
