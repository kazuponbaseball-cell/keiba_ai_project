from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo


FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_FIELDS = {
    "schema_version",
    "schedule_provider_id",
    "schedule_observation_id",
    "race_id",
    "scheduled_post_time",
    "source_event_time",
    "received_at",
    "source_reference",
    "source_payload_sha256",
    "schedule_version",
}
FORBIDDEN_FIELD_MARKERS = {
    "authorization",
    "cookie",
    "credential",
    "final_odds",
    "market",
    "market_rank",
    "odds",
    "order",
    "payout",
    "popularity",
    "result",
    "roi",
    "send_order",
    "stake",
}


class ScheduleProviderError(ValueError):
    pass


class ScheduleProviderUnavailable(ScheduleProviderError):
    pass


class ScheduleProviderContractError(ScheduleProviderError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_time(value: Any, timezone_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ScheduleProviderContractError("schedule timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScheduleProviderContractError("schedule timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ScheduleProviderContractError("schedule timestamp must be timezone-aware")
    return parsed.astimezone(ZoneInfo(timezone_name))


def _recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key).strip().lower()
            yield from _recursive_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _recursive_keys(nested)


def _assert_no_forbidden_fields(record: dict[str, Any]) -> None:
    observed = set(_recursive_keys(record))
    forbidden = sorted(
        key
        for key in observed
        if key in FORBIDDEN_FIELD_MARKERS
        or key.endswith("_odds")
        or key.endswith("_payout")
        or key.endswith("_result")
    )
    if forbidden:
        raise ScheduleProviderContractError(
            "schedule provider payload contains forbidden field(s): "
            + ", ".join(forbidden)
        )


def _validate_record(record: Any, timezone_name: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ScheduleProviderContractError("schedule observation must be an object")
    _assert_no_forbidden_fields(record)
    unexpected = sorted(set(record) - ALLOWED_FIELDS)
    if unexpected:
        raise ScheduleProviderContractError(
            "schedule observation contains unexpected field(s): "
            + ", ".join(unexpected)
        )
    required = ALLOWED_FIELDS - {"schema_version"}
    missing = sorted(
        field for field in required if not str(record.get(field, "")).strip()
    )
    if missing:
        raise ScheduleProviderContractError(
            "schedule observation is missing field(s): " + ", ".join(missing)
        )
    if record.get("schema_version", 1) != 1:
        raise ScheduleProviderContractError("schedule observation schema is unsupported")
    if not re.fullmatch(r"\d{12}", str(record["race_id"])):
        raise ScheduleProviderContractError("schedule race_id must be a 12-digit key")
    if not FULL_SHA256.fullmatch(str(record["source_payload_sha256"]).lower()):
        raise ScheduleProviderContractError("schedule source payload hash is invalid")
    source = parse_time(record["source_event_time"], timezone_name)
    received = parse_time(record["received_at"], timezone_name)
    post = parse_time(record["scheduled_post_time"], timezone_name)
    if source > received:
        raise ScheduleProviderContractError(
            "schedule source event time is after received time"
        )
    normalized = dict(record)
    normalized["source_payload_sha256"] = str(
        record["source_payload_sha256"]
    ).lower()
    normalized["source_event_time"] = source.isoformat(timespec="milliseconds")
    normalized["received_at"] = received.isoformat(timespec="milliseconds")
    normalized["scheduled_post_time"] = post.isoformat(timespec="milliseconds")
    normalized["observation_digest"] = canonical_digest(normalized)
    return normalized


def _deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_observation: dict[tuple[str, str], dict[str, Any]] = {}
    by_version: dict[tuple[str, str, str], dict[str, Any]] = {}
    unique: list[dict[str, Any]] = []
    for record in records:
        provider_id = str(record["schedule_provider_id"])
        race_id = str(record["race_id"])
        observation_key = (provider_id, str(record["schedule_observation_id"]))
        version_key = (provider_id, race_id, str(record["schedule_version"]))
        prior_observation = by_observation.get(observation_key)
        if prior_observation is not None:
            if prior_observation != record:
                raise ScheduleProviderContractError(
                    "conflicting duplicate schedule_observation_id"
                )
            continue
        prior_version = by_version.get(version_key)
        if prior_version is not None:
            comparable_prior = {
                key: value
                for key, value in prior_version.items()
                if key not in {"schedule_observation_id", "observation_digest"}
            }
            comparable_current = {
                key: value
                for key, value in record.items()
                if key not in {"schedule_observation_id", "observation_digest"}
            }
            if comparable_prior != comparable_current:
                raise ScheduleProviderContractError(
                    "conflicting duplicate schedule_version"
                )
            continue
        by_observation[observation_key] = record
        by_version[version_key] = record
        unique.append(record)
    return unique


class FileBackedScheduleProvider:
    """Read schedule-only observations from a local append-only JSONL snapshot."""

    def __init__(
        self,
        path: Path,
        *,
        timezone_name: str,
        clock: Callable[[], datetime],
        provider_id: str = "file_backed_schedule_v1",
    ) -> None:
        self.path = Path(path)
        self.timezone_name = timezone_name
        self.clock = clock
        self.provider_id = provider_id

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            raise ScheduleProviderUnavailable("schedule observation file is unavailable")
        records: list[dict[str, Any]] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ScheduleProviderContractError(
                            f"invalid schedule JSONL at line {line_number}"
                        ) from exc
                    record = _validate_record(raw, self.timezone_name)
                    if record["schedule_provider_id"] != self.provider_id:
                        raise ScheduleProviderContractError(
                            "schedule observation provider_id mismatch"
                        )
                    records.append(record)
        except OSError as exc:
            raise ScheduleProviderUnavailable(
                "schedule observation file cannot be read"
            ) from exc
        if not records:
            raise ScheduleProviderUnavailable("schedule observation file is empty")
        return _deduplicate(records)

    def __call__(self, candidate: dict[str, Any]) -> dict[str, Any]:
        race_id = str(candidate.get("race_id", ""))
        now = self.clock().astimezone(ZoneInfo(self.timezone_name))
        candidates = [
            record
            for record in self._load()
            if str(record["race_id"]) == race_id
            and parse_time(record["received_at"], self.timezone_name) <= now
        ]
        if not candidates:
            raise ScheduleProviderUnavailable(
                "no as-of schedule observation is available for race"
            )
        latest_received = max(
            parse_time(record["received_at"], self.timezone_name)
            for record in candidates
        )
        latest = [
            record
            for record in candidates
            if parse_time(record["received_at"], self.timezone_name)
            == latest_received
        ]
        if len({str(record["schedule_version"]) for record in latest}) != 1:
            raise ScheduleProviderContractError(
                "ambiguous schedule versions share the latest received time"
            )
        candidates.sort(
            key=lambda record: (
                parse_time(record["received_at"], self.timezone_name),
                parse_time(record["source_event_time"], self.timezone_name),
                str(record["schedule_observation_id"]),
            )
        )
        selected = candidates[-1]
        bound_reference = (
            f"{selected['source_reference']}"
            f"#provider={selected['schedule_provider_id']}"
            f";observation={selected['schedule_observation_id']}"
            f";version={selected['schedule_version']}"
        )
        return {
            "scheduled_post_time": selected["scheduled_post_time"],
            "source_event_time": selected["source_event_time"],
            "received_at": selected["received_at"],
            "locked_at": now.isoformat(timespec="milliseconds"),
            "source_reference": bound_reference,
            "source_payload_sha256": selected["source_payload_sha256"],
            "schedule_provider_id": selected["schedule_provider_id"],
            "schedule_observation_id": selected["schedule_observation_id"],
            "provider_schedule_version": selected["schedule_version"],
            "provider_status": "OBSERVATION_SELECTED",
        }
