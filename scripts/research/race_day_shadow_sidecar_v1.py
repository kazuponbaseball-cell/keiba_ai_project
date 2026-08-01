from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from dataclasses import asdict
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from realtime_paper_decision_wiring_v1 import (  # noqa: E402
    candidate_digest,
    evaluate_paper_decision,
    load_config as load_coordinator_config,
)


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_MARKET_STATUSES = {"OPEN", "TRADING"}
FORBIDDEN_INPUT_FIELDS = {
    "actual_order_acknowledgement",
    "credential",
    "final_odds",
    "official_result",
    "payout",
    "popularity",
}


def _strict_bool(value: Any) -> bool | None:
    if value is True or value is False:
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def _parse_time(value: Any, timezone_name: str) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    zone = ZoneInfo(timezone_name)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value is not None else None


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_pair_key(value: Any) -> str:
    numbers = re.findall(r"\d+", unicodedata.normalize("NFKC", str(value or "")))
    if len(numbers) != 2:
        raise ValueError("candidate pair must contain exactly two horse numbers")
    first, second = (int(number) for number in numbers)
    if first == second or first < 1 or second < 1:
        raise ValueError("candidate pair must contain two distinct positive horse numbers")
    low, high = sorted((first, second))
    return f"{low}-{high}"


def load_sidecar_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    safety = config.get("safety", {})
    required_false = (
        "formal_buy",
        "send_order",
        "target_handoff",
        "production_dashboard_write",
        "real_data_during_preparation",
    )
    for field in required_false:
        if safety.get(field) is not False:
            raise ValueError(f"{field} must remain false")
    if safety.get("stake") != 0:
        raise ValueError("stake must remain zero")
    quote_contract = config.get("quote_contract", {})
    if quote_contract.get("ticket_type") != "wide":
        raise ValueError("sidecar supports the registered wide contract only")
    if int(quote_contract.get("source_timestamp_resolution_seconds", 0)) != 60:
        raise ValueError("JRA displayed source time must declare 60-second resolution")
    return config


def coordinator_config_for(
    sidecar_config: dict[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    path = Path(str(sidecar_config["coordinator_config_path"]))
    if not path.is_absolute():
        path = root / path
    return load_coordinator_config(path)


def _visible_text(document: str) -> str:
    without_script = re.sub(
        r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>",
        " ",
        document,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<[^>]+>", " ", without_script)
    normalized = unicodedata.normalize("NFKC", unescape(text))
    return re.sub(r"\s+", " ", normalized).strip()


def extract_quote_source_time(
    document: str,
    *,
    race_id: str,
    timezone_name: str,
    timestamp_pattern: str,
) -> datetime:
    race_digits = re.sub(r"\D", "", race_id)
    if len(race_digits) < 8:
        raise ValueError("race_id does not contain a race date")
    text = _visible_text(document)
    matches = list(re.finditer(timestamp_pattern, text))
    observed = {
        (int(match.group("hour")), int(match.group("minute")))
        for match in matches
    }
    if len(observed) != 1:
        raise ValueError("quote source timestamp is missing or ambiguous")
    hour, minute = next(iter(observed))
    if hour > 23 or minute > 59:
        raise ValueError("quote source timestamp is outside the clock range")
    date = datetime.strptime(race_digits[:8], "%Y%m%d").date()
    return datetime(
        date.year,
        date.month,
        date.day,
        hour,
        minute,
        tzinfo=ZoneInfo(timezone_name),
    )


def _attribute(attrs: str, name: str) -> str:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*(['\"])(.*?)\1",
        attrs,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return unescape(match.group(2)) if match else ""


def _node_text(fragment: str) -> str:
    return _visible_text(fragment)


def _first_number(value: str) -> float | None:
    normalized = unicodedata.normalize("NFKC", value).replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", normalized)
    return float(match.group(0)) if match else None


def _span_value(fragment: str, class_name: str) -> float | None:
    for match in re.finditer(
        r"<span\b(?P<attrs>[^>]*)>(?P<body>.*?)</span>",
        fragment,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        classes = set(_attribute(match.group("attrs"), "class").split())
        if class_name in classes:
            return _first_number(_node_text(match.group("body")))
    return None


def parse_exact_wide_quote(document: str, candidate_pair_key: str) -> dict[str, float]:
    expected = canonical_pair_key(candidate_pair_key)
    matches: list[dict[str, float]] = []
    for table in re.finditer(
        r"<table\b(?P<attrs>[^>]*)>(?P<body>.*?)</table>",
        document,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        classes = set(_attribute(table.group("attrs"), "class").split())
        if "wide" not in classes:
            continue
        body = table.group("body")
        caption = re.search(
            r"<caption\b[^>]*>(.*?)</caption>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        first_number = _first_number(_node_text(caption.group(1))) if caption else None
        if first_number is None:
            continue
        for row in re.finditer(
            r"<tr\b[^>]*>(?P<body>.*?)</tr>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            row_body = row.group("body")
            header = re.search(
                r"<th\b[^>]*>(.*?)</th>",
                row_body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            cell = re.search(
                r"<td\b[^>]*>(.*?)</td>",
                row_body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            second_number = _first_number(_node_text(header.group(1))) if header else None
            if second_number is None or cell is None:
                continue
            pair = canonical_pair_key(f"{int(first_number)}-{int(second_number)}")
            if pair != expected:
                continue
            cell_body = cell.group(1)
            odds_low = _span_value(cell_body, "min")
            odds_high = _span_value(cell_body, "max")
            fallback = _first_number(_node_text(cell_body))
            if odds_low is None:
                odds_low = fallback
            if odds_high is None:
                odds_high = odds_low
            if odds_low is None or odds_high is None:
                continue
            if odds_low <= 0 or odds_high < odds_low:
                continue
            matches.append({"odds_low": float(odds_low), "odds_high": float(odds_high)})
    unique = {(row["odds_low"], row["odds_high"]) for row in matches}
    if len(unique) != 1:
        raise ValueError("exact candidate wide quote is missing or ambiguous")
    odds_low, odds_high = next(iter(unique))
    return {"odds_low": odds_low, "odds_high": odds_high}


def _sidecar_idempotency_key(
    experiment_id: str,
    race_id: str,
    candidate_hash: str,
) -> str:
    raw = "|".join((experiment_id, race_id, candidate_hash))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _base_event(
    *,
    config: dict[str, Any],
    candidate: dict[str, Any],
    candidate_hash: str,
    decision_time: datetime,
    status: str,
    reason: str,
    robust_expected_return: float | None,
    quote_request_started_at: datetime | None,
    quote_received_at: datetime | None,
    quote_source_event_time: datetime | None,
    market_status: str,
    schedule_record_hash: str,
    schedule_contract_ok: bool,
    starter_universe_hash_at_freeze: str,
    starter_universe_hash_at_quote: str,
) -> dict[str, Any]:
    return {
        "race_id": str(candidate.get("race_id", "")),
        "ticket_type": str(candidate.get("ticket_type", "wide")),
        "candidate_pair_key": str(candidate.get("candidate_pair_key", "")),
        "candidate_freeze_record_hash": candidate_hash,
        "status": status,
        "reason": reason,
        "trigger": "",
        "robust_expected_return": robust_expected_return,
        "quote_request_started_at": _iso(quote_request_started_at),
        "quote_received_at": _iso(quote_received_at),
        "quote_source_event_time": _iso(quote_source_event_time),
        "decision_generated_at": _iso(decision_time),
        "market_status": market_status,
        "schedule_record_hash": schedule_record_hash,
        "schedule_contract_ok": schedule_contract_ok,
        "starter_universe_hash_at_freeze": starter_universe_hash_at_freeze,
        "starter_universe_hash_at_quote": starter_universe_hash_at_quote,
        "starter_universe_unchanged": bool(
            starter_universe_hash_at_freeze
            and starter_universe_hash_at_freeze == starter_universe_hash_at_quote
        ),
        "idempotency_key": _sidecar_idempotency_key(
            str(config["experiment_id"]),
            str(candidate.get("race_id", "")),
            candidate_hash,
        ),
        "formal_buy": False,
        "send_order": False,
        "paper_stake_yen": 0,
    }


def _input_has_forbidden_fields(*records: dict[str, Any]) -> bool:
    return any(FORBIDDEN_INPUT_FIELDS.intersection(record) for record in records)


def evaluate_sidecar_decision(
    *,
    candidate_row: dict[str, Any],
    schedule_record: dict[str, Any],
    universe_observation: dict[str, Any],
    odds_html: str,
    odds_join_started_at: Any,
    quote_request_started_at: Any,
    quote_received_at: Any,
    decision_time: Any,
    robust_expected_return: float | None,
    sidecar_config: dict[str, Any],
    coordinator_config: dict[str, Any],
) -> dict[str, Any]:
    timezone_name = str(sidecar_config["timezone"])
    parsed_decision_at = _parse_time(decision_time, timezone_name)
    decision_at = parsed_decision_at or datetime.now(ZoneInfo(timezone_name))
    request_at = _parse_time(quote_request_started_at, timezone_name)
    received_at = _parse_time(quote_received_at, timezone_name)
    odds_join_at = _parse_time(odds_join_started_at, timezone_name)
    race_id = str(candidate_row.get("race_id", "")).strip()
    ticket_type = str(candidate_row.get("ticket_type", "wide")).strip().lower()
    pair_valid = True
    try:
        pair_key = canonical_pair_key(candidate_row.get("candidate_pair_key"))
    except ValueError:
        pair_valid = False
        pair_key = str(candidate_row.get("candidate_pair_key", "")).strip()
    candidate = {
        "race_id": race_id,
        "ticket_type": ticket_type,
        "candidate_pair_key": pair_key,
    }
    candidate_hash = str(candidate_row.get("candidate_freeze_record_hash", "")).strip()
    market_status = str(universe_observation.get("market_status", "")).strip().upper()
    schedule_hash = str(schedule_record.get("schedule_record_hash", "")).strip()
    schedule_ok = _strict_bool(schedule_record.get("schedule_contract_ok")) is True
    freeze_universe_hash = str(
        candidate_row.get("starter_universe_hash_at_freeze", "")
    ).strip()
    quote_universe_hash = str(
        universe_observation.get("starter_universe_hash_at_quote", "")
    ).strip()
    source_time: datetime | None = None

    def fail(reason: str) -> dict[str, Any]:
        return _base_event(
            config=sidecar_config,
            candidate=candidate,
            candidate_hash=candidate_hash,
            decision_time=decision_at,
            status="NO_BET",
            reason=reason,
            robust_expected_return=robust_expected_return,
            quote_request_started_at=request_at,
            quote_received_at=received_at,
            quote_source_event_time=source_time,
            market_status=market_status,
            schedule_record_hash=schedule_hash,
            schedule_contract_ok=schedule_ok,
            starter_universe_hash_at_freeze=freeze_universe_hash,
            starter_universe_hash_at_quote=quote_universe_hash,
        )

    if _input_has_forbidden_fields(candidate_row, schedule_record, universe_observation):
        return fail("FORBIDDEN_INPUT_FIELD")
    if parsed_decision_at is None:
        return fail("DECISION_TIME_MISSING")
    if not race_id or ticket_type != "wide" or not pair_valid:
        return fail("CANDIDATE_IDENTITY_INVALID")
    if _strict_bool(candidate_row.get("candidate_freeze_contract_ok")) is not True:
        return fail("CANDIDATE_FREEZE_CONTRACT_FAILED")
    if _strict_bool(candidate_row.get("candidate_uses_odds")) is not False:
        return fail("CANDIDATE_ODDS_FIREWALL_FAILED")
    try:
        expected_hash = candidate_digest(candidate)
    except ValueError:
        return fail("CANDIDATE_IDENTITY_INVALID")
    if not candidate_hash or candidate_hash != expected_hash:
        return fail("CANDIDATE_HASH_MISMATCH")
    candidate["candidate_freeze_record_hash"] = candidate_hash

    freeze_ack = _parse_time(
        candidate_row.get("candidate_freeze_persist_ack_at"), timezone_name
    )
    if (
        freeze_ack is None
        or odds_join_at is None
        or request_at is None
        or received_at is None
        or not (freeze_ack < odds_join_at <= request_at <= received_at <= decision_at)
    ):
        return fail("SOURCE_TIME_ORDER_VIOLATION")
    if not schedule_hash or not schedule_ok:
        return fail("SCHEDULE_CONTRACT_FAILED")
    post_time = _parse_time(schedule_record.get("scheduled_post_time"), timezone_name)
    if post_time is None:
        return fail("SCHEDULE_TIME_MISSING")
    if (
        not freeze_universe_hash
        or freeze_universe_hash != quote_universe_hash
        or _strict_bool(universe_observation.get("starter_universe_unchanged"))
        is not True
    ):
        return fail("STARTER_UNIVERSE_CHANGED")
    if market_status not in ALLOWED_MARKET_STATUSES:
        return fail("MARKET_NOT_OPEN")
    requested_race_id = str(universe_observation.get("race_id", "")).strip()
    requested_ticket_type = str(
        universe_observation.get("ticket_type", "")
    ).strip().lower()
    try:
        requested_pair_key = canonical_pair_key(
            universe_observation.get("quote_pair_key")
        )
    except ValueError:
        return fail("QUOTE_REQUEST_IDENTITY_INVALID")
    if (
        requested_race_id != race_id
        or requested_ticket_type != "wide"
        or requested_pair_key != pair_key
    ):
        return fail("QUOTE_REQUEST_IDENTITY_MISMATCH")

    quote_config = sidecar_config["quote_contract"]
    try:
        source_time = extract_quote_source_time(
            odds_html,
            race_id=race_id,
            timezone_name=timezone_name,
            timestamp_pattern=str(quote_config["source_timestamp_pattern"]),
        )
    except ValueError:
        return fail("QUOTE_SOURCE_TIME_MISSING_OR_AMBIGUOUS")
    if source_time > received_at:
        return fail("QUOTE_SOURCE_AFTER_RECEIPT")
    cutoff = post_time - timedelta(
        seconds=int(quote_config["strict_cutoff_seconds_before_post"])
    )
    source_upper_bound = source_time + timedelta(
        seconds=int(quote_config["source_timestamp_resolution_seconds"])
    )
    if source_upper_bound > cutoff or received_at > cutoff:
        return fail("STRICT_QUOTE_CUTOFF_FAILED")
    try:
        parsed_quote = parse_exact_wide_quote(odds_html, pair_key)
    except ValueError:
        return fail("EXACT_CANDIDATE_QUOTE_MISSING_OR_AMBIGUOUS")
    if robust_expected_return is not None:
        try:
            robust_er_number = float(robust_expected_return)
        except (TypeError, ValueError):
            return fail("ROBUST_EXPECTED_RETURN_INVALID")
        if not math.isfinite(robust_er_number):
            return fail("ROBUST_EXPECTED_RETURN_INVALID")

    quote = {
        "race_id": race_id,
        "ticket_type": "wide",
        "candidate_pair_key": pair_key,
        "quote_source_event_time": _iso(source_time),
        "quote_received_at": _iso(received_at),
        "quote_contract_ok": True,
        "market_status": market_status,
        "odds_low": parsed_quote["odds_low"],
        "odds_high": parsed_quote["odds_high"],
    }
    value_record = {
        "candidate_freeze_record_hash": candidate_hash,
        "decision_generated_at": _iso(decision_at),
        "robust_expected_return": robust_expected_return,
        "starter_universe_unchanged": True,
    }
    coordinator_decision = evaluate_paper_decision(
        candidate,
        quote,
        value_record,
        {"post_time": _iso(post_time)},
        decision_at,
        coordinator_config,
    )
    event = asdict(coordinator_decision)
    event.update(
        {
            "quote_request_started_at": _iso(request_at),
            "quote_source_event_time": _iso(source_time),
            "market_status": market_status,
            "schedule_record_hash": schedule_hash,
            "schedule_contract_ok": schedule_ok,
            "starter_universe_hash_at_freeze": freeze_universe_hash,
            "starter_universe_hash_at_quote": quote_universe_hash,
            "starter_universe_unchanged": True,
            "idempotency_key": _sidecar_idempotency_key(
                str(sidecar_config["experiment_id"]), race_id, candidate_hash
            ),
            "formal_buy": False,
            "send_order": False,
            "paper_stake_yen": 0,
        }
    )
    return event


def verify_sidecar_output_safety(records: Iterable[dict[str, Any]]) -> None:
    for record in records:
        if record.get("formal_buy") is not False:
            raise ValueError("formal_buy output violation")
        if record.get("send_order") is not False:
            raise ValueError("send_order output violation")
        if record.get("paper_stake_yen") != 0:
            raise ValueError("paper stake output violation")


def append_event_jsonl(path: Path, event: dict[str, Any]) -> bool:
    verify_sidecar_output_safety([event])
    idempotency_key = str(event.get("idempotency_key", "")).strip()
    if not idempotency_key:
        raise ValueError("event is missing idempotency_key")
    path.parent.mkdir(parents=True, exist_ok=True)
    marker_dir = path.parent / f".{path.name}.ids"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / idempotency_key
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as marker_handle:
            marker_handle.write(idempotency_key)
            marker_handle.flush()
            os.fsync(marker_handle.fileno())
        with path.open("a", encoding="utf-8", newline="\n") as ledger:
            ledger.write(_canonical_json(event) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
    except Exception:
        marker.unlink(missing_ok=True)
        raise
    return True


def read_event_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    verify_sidecar_output_safety(records)
    return records


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create one research-only exact-wide shadow event from frozen inputs."
    )
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--schedule-json", type=Path, required=True)
    parser.add_argument("--universe-json", type=Path, required=True)
    parser.add_argument("--odds-html", type=Path, required=True)
    parser.add_argument("--odds-join-started-at", required=True)
    parser.add_argument("--quote-request-started-at", required=True)
    parser.add_argument("--quote-received-at", required=True)
    parser.add_argument("--decision-time", required=True)
    parser.add_argument("--robust-expected-return", type=float, required=True)
    parser.add_argument("--ledger-jsonl", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "race_day_shadow_sidecar_v1.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sidecar_config = load_sidecar_config(args.config)
    coordinator_config = coordinator_config_for(sidecar_config)
    event = evaluate_sidecar_decision(
        candidate_row=_load_json(args.candidate_json),
        schedule_record=_load_json(args.schedule_json),
        universe_observation=_load_json(args.universe_json),
        odds_html=args.odds_html.read_text(encoding="utf-8"),
        odds_join_started_at=args.odds_join_started_at,
        quote_request_started_at=args.quote_request_started_at,
        quote_received_at=args.quote_received_at,
        decision_time=args.decision_time,
        robust_expected_return=args.robust_expected_return,
        sidecar_config=sidecar_config,
        coordinator_config=coordinator_config,
    )
    appended = append_event_jsonl(args.ledger_jsonl, event)
    print(_canonical_json({"appended": appended, "event": event}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
