from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from race_day_shadow_sidecar_v1 import (  # noqa: E402
    append_event_jsonl,
    candidate_digest,
    coordinator_config_for,
    evaluate_sidecar_decision,
    load_sidecar_config,
    read_event_jsonl,
    verify_sidecar_output_safety,
)


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_TARGET_FIELDS = {
    "alternative_pair",
    "final_odds",
    "official_result",
    "payout",
    "popularity",
    "result",
}
REASON_MAP = {
    "CANDIDATE_HASH_MISMATCH": "NO_BET_CANDIDATE_HASH_MISMATCH",
    "EXACT_CANDIDATE_QUOTE_MISSING_OR_AMBIGUOUS": "NO_BET_T3_QUOTE_NOT_AVAILABLE",
    "QUOTE_SOURCE_TIME_MISSING_OR_AMBIGUOUS": "NO_BET_T3_QUOTE_NOT_AVAILABLE",
    "ROBUST_EXPECTED_RETURN_BELOW_THRESHOLD": "NO_BET_VALUE_BELOW_THRESHOLD",
    "SCHEDULE_CONTRACT_FAILED": "NO_BET_SCHEDULE_CONTRACT_FAILURE",
    "STARTER_UNIVERSE_CHANGED": "NO_BET_STARTER_UNIVERSE_CHANGED",
    "STRICT_QUOTE_CUTOFF_FAILED": "NO_BET_T3_QUOTE_ASOF_VIOLATION",
}


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _parse_time(value: Any, timezone_name: str) -> datetime:
    if value is None or not str(value).strip():
        raise ValueError("timestamp is missing")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    zone = ZoneInfo(timezone_name)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value or "").strip())
    if not str(path):
        raise ValueError("capture packet path is missing")
    return path if path.is_absolute() else base_dir / path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_pilot_config(path: Path) -> dict[str, Any]:
    config = _load_json_object(path)
    safety = config.get("safety", {})
    for field in (
        "formal_buy",
        "send_order",
        "target_handoff",
        "production_dashboard_write",
        "real_data_during_preparation",
        "roi_calculation",
    ):
        if safety.get(field) is not False:
            raise ValueError(f"{field} must remain false")
    if safety.get("stake") != 0:
        raise ValueError("stake must remain zero")

    policy = config.get("policy", {})
    if policy.get("candidate_policy") != "WIDE_1_NON_ODDS_FROZEN":
        raise ValueError("candidate policy must remain frozen WIDE_1")
    if policy.get("ticket_type") != "wide":
        raise ValueError("pilot supports the registered wide ticket only")
    if policy.get("candidate_substitution_allowed") is not False:
        raise ValueError("candidate substitution must remain false")
    if policy.get("alternative_pair_search_allowed") is not False:
        raise ValueError("alternative pair search must remain false")
    if policy.get("wait_at_cutoff_becomes_no_bet") is not True:
        raise ValueError("WAIT must become NO_BET at the strict cutoff")

    card = config.get("target_card", {})
    expected = [int(value) for value in card.get("expected_race_numbers", [])]
    if expected != list(range(1, 13)):
        raise ValueError("target card must contain race numbers 1 through 12")
    if int(card.get("expected_race_count", 0)) != len(expected):
        raise ValueError("target race count does not match race numbers")
    return config


def validate_target_manifest(
    manifest: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    if manifest.get("experiment_id") != config.get("experiment_id"):
        raise ValueError("manifest experiment_id mismatch")
    if manifest.get("cohort_id") != config.get("cohort_id"):
        raise ValueError("manifest cohort_id mismatch")
    if manifest.get("data_class") not in {"synthetic", "real-data"}:
        raise ValueError("manifest data_class must be synthetic or real-data")

    expected_card = config["target_card"]
    observed_card = manifest.get("target_card", {})
    for field in ("race_date", "venue_code", "meeting_no", "day_no"):
        if str(observed_card.get(field)) != str(expected_card.get(field)):
            raise ValueError(f"target card {field} mismatch")

    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("target manifest records must be a list")
    race_numbers = [int(record.get("race_no", 0)) for record in records]
    expected_numbers = [
        int(value) for value in expected_card["expected_race_numbers"]
    ]
    if sorted(race_numbers) != expected_numbers or len(set(race_numbers)) != len(
        race_numbers
    ):
        raise ValueError("target manifest race numbers mismatch")
    race_ids = [str(record.get("race_id", "")).strip() for record in records]
    if any(not race_id for race_id in race_ids) or len(set(race_ids)) != len(race_ids):
        raise ValueError("target manifest race_id values must be present and unique")
    for record in records:
        forbidden = sorted(FORBIDDEN_TARGET_FIELDS.intersection(record))
        if forbidden:
            raise ValueError(
                "target record contains forbidden field(s): " + ", ".join(forbidden)
            )
        if record.get("target_registered") is not True:
            raise ValueError("every target record must be pre-registered")
        if not str(record.get("scheduled_post_time", "")).strip():
            raise ValueError("every target record needs scheduled_post_time")
    return sorted(records, key=lambda row: int(row["race_no"]))


def assert_real_data_authorized(root: Path, experiment_id: str) -> None:
    registry = root / "research" / "REGISTRY.jsonl"
    events = [
        json.loads(line)
        for line in registry.read_text(encoding="utf-8").splitlines()
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


def _pilot_idempotency_key(
    experiment_id: str, cohort_id: str, race_id: str
) -> str:
    raw = "|".join((experiment_id, cohort_id, race_id))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _missing_packet_event(
    *,
    config: dict[str, Any],
    target: dict[str, Any],
    decision_time: datetime,
) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "cohort_id": config["cohort_id"],
        "card_id": target.get("card_id", ""),
        "race_id": str(target["race_id"]),
        "race_no": int(target["race_no"]),
        "ticket_type": "wide",
        "candidate_pair_key": "",
        "candidate_freeze_record_hash": "",
        "status": "NO_BET",
        "reason": "NO_BET_T3_CAPTURE_PACKET_MISSING",
        "source_reason": "CAPTURE_PACKET_MISSING_AT_CUTOFF",
        "trigger": "",
        "robust_expected_return": None,
        "quote_request_started_at": None,
        "quote_received_at": None,
        "quote_source_event_time": None,
        "decision_generated_at": decision_time.isoformat(timespec="seconds"),
        "market_status": "",
        "target_registered": True,
        "candidate_changed_after_odds": False,
        "idempotency_key": _pilot_idempotency_key(
            str(config["experiment_id"]),
            str(config["cohort_id"]),
            str(target["race_id"]),
        ),
        "formal_buy": False,
        "send_order": False,
        "paper_stake_yen": 0,
    }


def _normalize_final_event(
    event: dict[str, Any],
    *,
    config: dict[str, Any],
    target: dict[str, Any],
    candidate_row: dict[str, Any],
) -> dict[str, Any]:
    source_reason = str(event.get("reason", ""))
    if event.get("status") == "WAIT":
        event["status"] = "NO_BET"
        event["reason"] = REASON_MAP.get(
            source_reason, f"NO_BET_{source_reason or 'UNRESOLVED_WAIT'}"
        )
    elif event.get("status") == "NO_BET":
        event["reason"] = REASON_MAP.get(
            source_reason, f"NO_BET_{source_reason or 'CONTRACT_FAILURE'}"
        )
    event["source_reason"] = source_reason
    event["experiment_id"] = config["experiment_id"]
    event["cohort_id"] = config["cohort_id"]
    event["card_id"] = target.get("card_id", "")
    event["race_no"] = int(target["race_no"])
    event["target_registered"] = True
    event["candidate_changed_after_odds"] = (
        str(event.get("candidate_pair_key", ""))
        != str(candidate_row.get("candidate_pair_key", ""))
    )
    event["idempotency_key"] = _pilot_idempotency_key(
        str(config["experiment_id"]),
        str(config["cohort_id"]),
        str(target["race_id"]),
    )
    event["formal_buy"] = False
    event["send_order"] = False
    event["paper_stake_yen"] = 0
    return event


def _evaluate_packet(
    *,
    packet_path: Path,
    target: dict[str, Any],
    config: dict[str, Any],
    sidecar_config: dict[str, Any],
    coordinator_config: dict[str, Any],
) -> dict[str, Any]:
    packet = _load_json_object(packet_path)
    if str(packet.get("race_id", "")) != str(target["race_id"]):
        raise ValueError("capture packet race_id mismatch")
    candidate_row = packet.get("candidate_row")
    schedule_record = packet.get("schedule_record")
    universe_observation = packet.get("universe_observation")
    if not all(
        isinstance(value, dict)
        for value in (candidate_row, schedule_record, universe_observation)
    ):
        raise ValueError("capture packet records must be JSON objects")
    odds_html_path = _resolve_path(packet.get("odds_html_path"), packet_path.parent)
    odds_html = odds_html_path.read_text(encoding="utf-8")
    event = evaluate_sidecar_decision(
        candidate_row=candidate_row,
        schedule_record=schedule_record,
        universe_observation=universe_observation,
        odds_html=odds_html,
        odds_join_started_at=packet.get("odds_join_started_at"),
        quote_request_started_at=packet.get("quote_request_started_at"),
        quote_received_at=packet.get("quote_received_at"),
        decision_time=packet.get("decision_time"),
        robust_expected_return=packet.get("robust_expected_return"),
        sidecar_config=sidecar_config,
        coordinator_config=coordinator_config,
    )
    return _normalize_final_event(
        event,
        config=config,
        target=target,
        candidate_row=candidate_row,
    )


def _build_summary(
    *,
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    ledger_records: list[dict[str, Any]],
    pending_race_ids: list[str],
) -> dict[str, Any]:
    cohort_records = [
        record
        for record in ledger_records
        if record.get("experiment_id") == config["experiment_id"]
        and record.get("cohort_id") == config["cohort_id"]
    ]
    target_ids = {str(target["race_id"]) for target in targets}
    observed_ids = [
        str(record.get("race_id", ""))
        for record in cohort_records
        if str(record.get("race_id", "")) in target_ids
    ]
    counts = Counter(observed_ids)
    duplicate_rows = sum(max(0, count - 1) for count in counts.values())
    missing_ids = sorted(target_ids.difference(observed_ids))
    candidate_changes = sum(
        record.get("candidate_changed_after_odds") is True for record in cohort_records
    )
    unsafe_rows = sum(
        record.get("formal_buy") is not False
        or record.get("send_order") is not False
        or record.get("paper_stake_yen") != 0
        for record in cohort_records
    )
    invalid = bool(duplicate_rows or candidate_changes or unsafe_rows)
    if invalid:
        status = "INVALID"
    elif not missing_ids:
        status = "PASS"
    else:
        status = "IN_PROGRESS"
    status_counts = Counter(str(record.get("status", "")) for record in cohort_records)
    reason_counts = Counter(str(record.get("reason", "")) for record in cohort_records)
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "cohort_id": config["cohort_id"],
        "status": status,
        "expected_target_rows": len(targets),
        "recorded_target_rows": len(set(observed_ids)),
        "one_card_target_ledger_completeness": (
            len(set(observed_ids)) / len(targets) if targets else 0.0
        ),
        "missing_target_race_ids": missing_ids,
        "pending_target_race_ids": sorted(set(pending_race_ids)),
        "duplicate_decision_rows": duplicate_rows,
        "candidate_changes_after_odds": candidate_changes,
        "unsafe_output_rows": unsafe_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "roi_calculated": False,
    }


def run_one_card(
    *,
    manifest_path: Path,
    ledger_path: Path,
    summary_path: Path,
    config: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    manifest = _load_json_object(manifest_path)
    targets = validate_target_manifest(manifest, config)
    sidecar_path = _resolve_path(config["sidecar_config_path"], ROOT)
    sidecar_config = dict(load_sidecar_config(sidecar_path))
    sidecar_config["experiment_id"] = config["experiment_id"]
    coordinator_config = coordinator_config_for(sidecar_config, root=ROOT)
    existing_records = read_event_jsonl(ledger_path)
    existing_ids = {
        str(record.get("race_id", ""))
        for record in existing_records
        if record.get("experiment_id") == config["experiment_id"]
        and record.get("cohort_id") == config["cohort_id"]
    }
    pending: list[str] = []
    manifest_dir = manifest_path.parent
    cutoff_seconds = int(config["policy"]["strict_cutoff_seconds_before_post"])

    for target in targets:
        race_id = str(target["race_id"])
        if race_id in existing_ids:
            continue
        packet_raw = str(target.get("capture_packet_path", "")).strip()
        packet_path = _resolve_path(packet_raw, manifest_dir) if packet_raw else None
        if packet_path is not None and packet_path.is_file():
            event = _evaluate_packet(
                packet_path=packet_path,
                target=target,
                config=config,
                sidecar_config=sidecar_config,
                coordinator_config=coordinator_config,
            )
            append_event_jsonl(ledger_path, event)
            existing_ids.add(race_id)
            continue

        post_time = _parse_time(target["scheduled_post_time"], config["timezone"])
        cutoff = post_time - timedelta(seconds=cutoff_seconds)
        if now < cutoff:
            pending.append(race_id)
            continue
        event = _missing_packet_event(
            config=config,
            target=target,
            decision_time=now,
        )
        append_event_jsonl(ledger_path, event)
        existing_ids.add(race_id)

    records = read_event_jsonl(ledger_path)
    verify_sidecar_output_safety(records)
    summary = _build_summary(
        config=config,
        targets=targets,
        ledger_records=records,
        pending_race_ids=pending,
    )
    _write_json_atomic(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the research-only one-card Grade-R shadow aggregator."
    )
    parser.add_argument("--target-manifest-json", type=Path, required=True)
    parser.add_argument("--ledger-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "grade_r_one_card_pilot_v1.json",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("synthetic", "real-data"),
        default="synthetic",
    )
    parser.add_argument("--now", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_pilot_config(args.config)
    manifest = _load_json_object(args.target_manifest_json)
    if manifest.get("data_class") != args.execution_mode:
        raise ValueError("execution mode and manifest data_class mismatch")
    if args.execution_mode == "real-data":
        assert_real_data_authorized(ROOT, str(config["experiment_id"]))
    now = (
        _parse_time(args.now, config["timezone"])
        if args.now
        else datetime.now(ZoneInfo(config["timezone"]))
    )
    summary = run_one_card(
        manifest_path=args.target_manifest_json,
        ledger_path=args.ledger_jsonl,
        summary_path=args.summary_json,
        config=config,
        now=now,
    )
    print(_canonical_json(summary))
    return 0 if summary["status"] != "INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
