from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


FORBIDDEN_CANDIDATE_FIELDS = {
    "final_odds",
    "live_odds",
    "odds",
    "payout",
    "popularity",
    "result",
    "robust_expected_return",
    "runtime_odds",
}


@dataclass(frozen=True)
class PaperDecision:
    race_id: str
    ticket_type: str
    candidate_pair_key: str
    candidate_freeze_record_hash: str
    status: str
    reason: str
    trigger: str
    robust_expected_return: float | None
    quote_received_at: str | None
    decision_generated_at: str
    idempotency_key: str
    formal_buy: bool = False
    send_order: bool = False
    paper_stake_yen: int = 0


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    safety = config.get("safety", {})
    if safety.get("formal_buy") is not False:
        raise ValueError("formal_buy must remain false")
    if safety.get("send_order") is not False:
        raise ValueError("send_order must remain false")
    if safety.get("stake") != 0:
        raise ValueError("stake must remain zero")
    if safety.get("target_handoff") is not False:
        raise ValueError("target_handoff must remain false")
    if safety.get("real_data_during_preparation") is not False:
        raise ValueError("real_data_during_preparation must remain false")
    return config


def parse_time(value: Any, timezone_name: str) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
    return parsed.astimezone(ZoneInfo(timezone_name))


def canonical_candidate_payload(candidate: dict[str, Any]) -> dict[str, str]:
    forbidden = sorted(FORBIDDEN_CANDIDATE_FIELDS.intersection(candidate))
    if forbidden:
        raise ValueError(f"candidate contains market or result field(s): {', '.join(forbidden)}")
    required = ("race_id", "ticket_type", "candidate_pair_key")
    payload: dict[str, str] = {}
    for field in required:
        value = str(candidate.get(field, "")).strip()
        if not value:
            raise ValueError(f"candidate is missing {field}")
        payload[field] = value
    return payload


def candidate_digest(candidate: dict[str, Any]) -> str:
    payload = canonical_candidate_payload(candidate)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _idempotency_key(
    experiment_id: str,
    candidate_hash: str,
) -> str:
    payload = "|".join((experiment_id, candidate_hash))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decision(
    *,
    config: dict[str, Any],
    candidate: dict[str, Any],
    candidate_hash: str,
    decision_time: datetime,
    status: str,
    reason: str,
    trigger: str = "",
    robust_er: float | None = None,
    quote_received_at: datetime | None = None,
) -> PaperDecision:
    return PaperDecision(
        race_id=str(candidate.get("race_id", "")),
        ticket_type=str(candidate.get("ticket_type", "")),
        candidate_pair_key=str(candidate.get("candidate_pair_key", "")),
        candidate_freeze_record_hash=candidate_hash,
        status=status,
        reason=reason,
        trigger=trigger,
        robust_expected_return=robust_er,
        quote_received_at=(
            quote_received_at.isoformat(timespec="seconds")
            if quote_received_at is not None
            else None
        ),
        decision_generated_at=decision_time.isoformat(timespec="seconds"),
        idempotency_key=_idempotency_key(
            str(config["experiment_id"]), candidate_hash
        ),
    )


def evaluate_paper_decision(
    candidate: dict[str, Any],
    quote: dict[str, Any],
    value_record: dict[str, Any],
    schedule: dict[str, Any],
    decision_time: datetime,
    config: dict[str, Any],
) -> PaperDecision:
    timezone_name = str(config["timezone"])
    candidate_hash = candidate_digest(candidate)
    registered_hash = str(candidate.get("candidate_freeze_record_hash", "")).strip()
    if not registered_hash or registered_hash != candidate_hash:
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="CANDIDATE_HASH_MISMATCH",
        )

    contract_pairs = (
        ("race_id", "QUOTE_RACE_MISMATCH"),
        ("ticket_type", "QUOTE_TICKET_TYPE_MISMATCH"),
        ("candidate_pair_key", "QUOTE_CANDIDATE_PAIR_MISMATCH"),
    )
    for field, reason in contract_pairs:
        if str(quote.get(field, "")) != str(candidate.get(field, "")):
            return _decision(
                config=config,
                candidate=candidate,
                candidate_hash=candidate_hash,
                decision_time=decision_time,
                status="NO_BET",
                reason=reason,
            )

    observed_hash = str(value_record.get("candidate_freeze_record_hash", "")).strip()
    if observed_hash != candidate_hash:
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="VALUE_CANDIDATE_HASH_MISMATCH",
        )

    quote_source_time = parse_time(quote.get("quote_source_event_time"), timezone_name)
    quote_received_at = parse_time(quote.get("quote_received_at"), timezone_name)
    value_generated_at = parse_time(value_record.get("decision_generated_at"), timezone_name)
    post_time = parse_time(schedule.get("post_time"), timezone_name)
    if quote_source_time is None or quote_received_at is None:
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="QUOTE_TIME_MISSING",
        )
    if value_generated_at is None or post_time is None:
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="DECISION_OR_SCHEDULE_TIME_MISSING",
            quote_received_at=quote_received_at,
        )
    if not (quote_source_time <= quote_received_at <= value_generated_at <= decision_time):
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="SOURCE_TIME_ORDER_VIOLATION",
            quote_received_at=quote_received_at,
        )

    timing = config["timing"]
    quote_age = (decision_time - quote_received_at).total_seconds()
    decision_age = (decision_time - value_generated_at).total_seconds()
    if quote_age > float(timing["maximum_quote_age_seconds"]):
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="QUOTE_STALE",
            quote_received_at=quote_received_at,
        )
    if decision_age > float(timing["maximum_decision_age_seconds"]):
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="VALUE_DECISION_STALE",
            quote_received_at=quote_received_at,
        )

    if quote.get("quote_contract_ok") is not True:
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="QUOTE_CONTRACT_FAILED",
            quote_received_at=quote_received_at,
        )
    if value_record.get("starter_universe_unchanged") is not True:
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="STARTER_UNIVERSE_CHANGED",
            quote_received_at=quote_received_at,
        )
    allowed_market_statuses = set(config["contracts"]["allowed_market_statuses"])
    if str(quote.get("market_status", "")) not in allowed_market_statuses:
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="MARKET_NOT_OPEN",
            quote_received_at=quote_received_at,
        )

    seconds_to_post = (post_time - decision_time).total_seconds()
    if seconds_to_post < float(timing["hard_stop_seconds_before_post"]):
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="PAPER_DECISION_DEADLINE_MISSED",
            quote_received_at=quote_received_at,
        )
    if seconds_to_post > float(timing["monitor_window_opens_seconds_before_post"]):
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="WAIT",
            reason="MONITOR_WINDOW_NOT_OPEN",
            quote_received_at=quote_received_at,
        )

    robust_er_raw = value_record.get("robust_expected_return")
    try:
        robust_er = float(robust_er_raw)
    except (TypeError, ValueError):
        robust_er = None
    if robust_er is None:
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="NO_BET",
            reason="ROBUST_EXPECTED_RETURN_MISSING",
            quote_received_at=quote_received_at,
        )

    normal_window = seconds_to_post <= float(
        timing["normal_window_opens_seconds_before_post"]
    )
    minimum_er = float(config["value"]["minimum_robust_expected_return"])
    early_minimum_er = float(
        config["value"]["early_window_minimum_robust_expected_return"]
    )
    if normal_window and robust_er >= minimum_er:
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="PAPER_READY",
            reason="CONTRACTS_PASSED",
            trigger="NORMAL_WINDOW",
            robust_er=robust_er,
            quote_received_at=quote_received_at,
        )
    if robust_er >= early_minimum_er:
        return _decision(
            config=config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_time,
            status="PAPER_READY",
            reason="CONTRACTS_PASSED",
            trigger="EARLY_VALUE",
            robust_er=robust_er,
            quote_received_at=quote_received_at,
        )
    return _decision(
        config=config,
        candidate=candidate,
        candidate_hash=candidate_hash,
        decision_time=decision_time,
        status="WAIT",
        reason="ROBUST_EXPECTED_RETURN_BELOW_THRESHOLD",
        robust_er=robust_er,
        quote_received_at=quote_received_at,
    )


def append_idempotent(
    records: list[dict[str, Any]], decision: PaperDecision
) -> bool:
    if decision.status != "PAPER_READY":
        return False
    if any(row.get("idempotency_key") == decision.idempotency_key for row in records):
        return False
    records.append(asdict(decision))
    return True


def verify_output_safety(records: Iterable[dict[str, Any]]) -> None:
    for row in records:
        if row.get("formal_buy") is not False:
            raise ValueError("formal_buy output violation")
        if row.get("send_order") is not False:
            raise ValueError("send_order output violation")
        if row.get("paper_stake_yen") != 0:
            raise ValueError("paper stake output violation")
