from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol


CATALOG_SCHEMA_VERSION = 1
CANDIDATE_ROLE = "candidate_only_projection"
SETTLEMENT_ROLE = "settlement_projection"
EXPECTED_ROLES = (CANDIDATE_ROLE, SETTLEMENT_ROLE)
EXPECTED_SETTLEMENT_COLUMNS = (
    "candidate_hit",
    "candidate_key",
    "official_outcome_completeness",
    "official_wide_pay",
    "race_id",
)
EXPECTED_KEY_COLUMNS = ("race_id", "candidate_key")

FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")
SAFE_COLUMN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")

_RELEASE_FIELDS = {
    "schema_version",
    "release_id",
    "status",
    "revoked",
    "manifest_sha256",
    "authentication_receipt_sha256",
    "status_receipt_sha256",
    "entries",
}
_ENTRY_FIELDS = {
    "entry_id",
    "role",
    "object_id",
    "content_sha256",
    "schema_sha256",
    "provenance_sha256",
    "role_attestations",
    "columns",
    "key_columns",
    "row_count",
    "race_count",
    "ordered_race_id_set_sha256",
}
_CANDIDATE_ATTESTATION_FIELDS = {
    "p_action_cross_source_equality_attestation_sha256",
    "candidate_materializer_usecols_sha256",
    "decision_base_lineage_sha256",
}
_SETTLEMENT_ATTESTATION_FIELDS = {
    "official_settlement_provenance_sha256",
}
_CATALOG_REQUIREMENT_FIELDS = {
    "exact_active_release_count",
    "exact_entry_count",
    "allowed_roles",
    "candidate_role",
    "settlement_role",
    "candidate_and_settlement_must_be_distinct_content_objects",
}
_CANDIDATE_REQUIREMENT_FIELDS = {"required_columns", "forbidden_columns", "key"}
_SETTLEMENT_REQUIREMENT_FIELDS = {"required_columns", "key"}

# These tokens are forbidden in the candidate projection even if a recipe tries
# to place such a field in its declared allowlist.  The allowlist is exact as a
# second, independent control.
_CANDIDATE_FORBIDDEN_TOKENS = {
    "hit",
    "market",
    "odds",
    "outcome",
    "pay",
    "payoff",
    "payout",
    "popularity",
    "price",
    "profit",
    "result",
    "return",
    "roi",
    "stake",
}


class CatalogValidationError(ValueError):
    """Fail-closed catalog metadata validation failure."""


class PregrantCatalogMetadataProvider(Protocol):
    """Metadata-only provider used before the irreversible access receipt.

    The provider method must return authenticated manifest metadata.  This
    interface intentionally exposes no blob, row, path, URL, or content reader.
    """

    def list_authenticated_release_metadata(self) -> Sequence[Mapping[str, Any]]:
        ...


def _exact_mapping(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CatalogValidationError(f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        raise CatalogValidationError(
            f"{label} fields differ from the exact contract; "
            f"missing={missing}, extra={extra}"
        )
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SAFE_IDENTIFIER.fullmatch(value):
        raise CatalogValidationError(f"{label} is not a safe identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not FULL_SHA256.fullmatch(value):
        raise CatalogValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise CatalogValidationError(f"{label} must be a positive integer")
    return value


def _columns(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise CatalogValidationError(f"{label} must be a non-empty list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not SAFE_COLUMN.fullmatch(item):
            raise CatalogValidationError(f"{label} contains an invalid column name")
        normalized.append(item)
    if len(normalized) != len(set(normalized)):
        raise CatalogValidationError(f"{label} contains duplicate columns")
    return tuple(normalized)


def _candidate_column_is_forbidden(column: str) -> bool:
    tokens = {token for token in column.lower().split("_") if token}
    return bool(tokens & _CANDIDATE_FORBIDDEN_TOKENS)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CatalogValidationError("catalog metadata is not canonical JSON data") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def release_manifest_sha256(
    *, release_id: str, normalized_entries: Sequence[Mapping[str, Any]]
) -> str:
    """Digest the immutable, content-addressed portion of a release manifest."""

    projection = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "release_id": release_id,
        "entries": list(normalized_entries),
    }
    return canonical_sha256(projection)


def _normalize_requirements(value: Any) -> dict[str, Any]:
    payload = _exact_mapping(value, _CATALOG_REQUIREMENT_FIELDS, "catalog requirements")
    if payload["exact_active_release_count"] != 1 or type(
        payload["exact_active_release_count"]
    ) is not int:
        raise CatalogValidationError("catalog requires exactly one ACTIVE release")
    if payload["exact_entry_count"] != 2 or type(payload["exact_entry_count"]) is not int:
        raise CatalogValidationError("catalog release requires exactly two entries")
    if payload["candidate_and_settlement_must_be_distinct_content_objects"] is not True:
        raise CatalogValidationError("candidate and settlement objects must be distinct")

    roles = _columns(payload["allowed_roles"], "catalog allowed_roles")
    if set(roles) != set(EXPECTED_ROLES) or len(roles) != 2:
        raise CatalogValidationError("catalog roles must be exactly candidate and settlement")

    candidate = _exact_mapping(
        payload["candidate_role"],
        _CANDIDATE_REQUIREMENT_FIELDS,
        "candidate catalog requirements",
    )
    candidate_columns = _columns(
        candidate["required_columns"], "candidate required_columns"
    )
    candidate_forbidden = _columns(
        candidate["forbidden_columns"], "candidate forbidden_columns"
    )
    candidate_key = _columns(candidate["key"], "candidate key")
    if candidate_key != EXPECTED_KEY_COLUMNS:
        raise CatalogValidationError("candidate key must be race_id then candidate_key")
    if not set(EXPECTED_KEY_COLUMNS).issubset(candidate_columns):
        raise CatalogValidationError("candidate allowlist omits its identity key")
    if set(candidate_columns) & set(candidate_forbidden):
        raise CatalogValidationError("candidate allowlist intersects its forbidden columns")
    forbidden_allowed = sorted(
        column for column in candidate_columns if _candidate_column_is_forbidden(column)
    )
    if forbidden_allowed:
        raise CatalogValidationError(
            "candidate allowlist contains result/payoff/odds/market data: "
            f"{forbidden_allowed}"
        )

    settlement = _exact_mapping(
        payload["settlement_role"],
        _SETTLEMENT_REQUIREMENT_FIELDS,
        "settlement catalog requirements",
    )
    settlement_columns = _columns(
        settlement["required_columns"], "settlement required_columns"
    )
    settlement_key = _columns(settlement["key"], "settlement key")
    if set(settlement_columns) != set(EXPECTED_SETTLEMENT_COLUMNS) or len(
        settlement_columns
    ) != len(EXPECTED_SETTLEMENT_COLUMNS):
        raise CatalogValidationError("settlement allowlist differs from the exact contract")
    if settlement_key != EXPECTED_KEY_COLUMNS:
        raise CatalogValidationError("settlement key must be race_id then candidate_key")

    return {
        "candidate_columns": tuple(sorted(candidate_columns)),
        "candidate_forbidden_columns": tuple(sorted(candidate_forbidden)),
        "candidate_key": candidate_key,
        "settlement_columns": tuple(sorted(settlement_columns)),
        "settlement_key": settlement_key,
    }


def _normalize_entry(
    value: Any,
    *,
    requirements: Mapping[str, Any],
    expected_race_count: int | None,
) -> dict[str, Any]:
    payload = _exact_mapping(value, _ENTRY_FIELDS, "catalog entry")
    role = payload["role"]
    if role not in EXPECTED_ROLES:
        raise CatalogValidationError("catalog entry has an unknown role")

    content_sha256 = _sha256(payload["content_sha256"], "entry content_sha256")
    object_id = payload["object_id"]
    if object_id != f"sha256:{content_sha256}":
        raise CatalogValidationError("catalog object_id is not its content address")

    columns = _columns(payload["columns"], f"{role} columns")
    key_columns = _columns(payload["key_columns"], f"{role} key_columns")
    if role == CANDIDATE_ROLE:
        attestation_payload = _exact_mapping(
            payload["role_attestations"],
            _CANDIDATE_ATTESTATION_FIELDS,
            "candidate role_attestations",
        )
        if tuple(sorted(columns)) != requirements["candidate_columns"]:
            raise CatalogValidationError(
                "candidate entry columns differ from the exact recipe allowlist"
            )
        forbidden = sorted(
            column
            for column in columns
            if column in requirements["candidate_forbidden_columns"]
            or _candidate_column_is_forbidden(column)
        )
        if forbidden:
            raise CatalogValidationError(
                "candidate entry contains result/payoff/odds/market data: "
                f"{forbidden}"
            )
        if key_columns != requirements["candidate_key"]:
            raise CatalogValidationError("candidate entry key differs from the recipe")
    else:
        attestation_payload = _exact_mapping(
            payload["role_attestations"],
            _SETTLEMENT_ATTESTATION_FIELDS,
            "settlement role_attestations",
        )
        if tuple(sorted(columns)) != requirements["settlement_columns"]:
            raise CatalogValidationError(
                "settlement entry columns differ from the exact allowlist"
            )
        if key_columns != requirements["settlement_key"]:
            raise CatalogValidationError("settlement entry key differs from the contract")

    row_count = _positive_integer(payload["row_count"], f"{role} row_count")
    race_count = _positive_integer(payload["race_count"], f"{role} race_count")
    if row_count != race_count:
        raise CatalogValidationError("catalog projection must contain one row per race")
    if expected_race_count is not None and race_count != expected_race_count:
        raise CatalogValidationError("catalog race count differs from the frozen cohort")

    return {
        "entry_id": _identifier(payload["entry_id"], "catalog entry_id"),
        "role": role,
        "object_id": object_id,
        "content_sha256": content_sha256,
        "schema_sha256": _sha256(payload["schema_sha256"], "entry schema_sha256"),
        "provenance_sha256": _sha256(
            payload["provenance_sha256"], "entry provenance_sha256"
        ),
        "role_attestations": {
            field: _sha256(
                attestation_payload[field],
                f"{role} role_attestations.{field}",
            )
            for field in sorted(attestation_payload)
        },
        "columns": list(sorted(columns)),
        "key_columns": list(key_columns),
        "row_count": row_count,
        "race_count": race_count,
        "ordered_race_id_set_sha256": _sha256(
            payload["ordered_race_id_set_sha256"],
            "entry ordered_race_id_set_sha256",
        ),
    }


def _normalize_release(
    value: Any,
    *,
    requirements: Mapping[str, Any],
    expected_race_count: int | None,
) -> dict[str, Any]:
    payload = _exact_mapping(value, _RELEASE_FIELDS, "catalog release metadata")
    if payload["schema_version"] != CATALOG_SCHEMA_VERSION or type(
        payload["schema_version"]
    ) is not int:
        raise CatalogValidationError("catalog release schema_version must be integer 1")
    release_id = _identifier(payload["release_id"], "catalog release_id")
    status = payload["status"]
    if status not in {"ACTIVE", "INACTIVE", "REVOKED"}:
        raise CatalogValidationError("catalog release status is unknown")
    if type(payload["revoked"]) is not bool:
        raise CatalogValidationError("catalog release revoked must be boolean")
    if (status == "REVOKED") != payload["revoked"]:
        raise CatalogValidationError("catalog release status and revoked flag disagree")

    entries_raw = payload["entries"]
    if not isinstance(entries_raw, list) or len(entries_raw) != 2:
        raise CatalogValidationError("catalog release must contain exactly two entries")
    entries = [
        _normalize_entry(
            item,
            requirements=requirements,
            expected_race_count=expected_race_count,
        )
        for item in entries_raw
    ]
    entries.sort(key=lambda item: item["role"])
    roles = [entry["role"] for entry in entries]
    if roles != sorted(EXPECTED_ROLES):
        raise CatalogValidationError("catalog release roles are missing or duplicated")
    if len({entry["entry_id"] for entry in entries}) != 2:
        raise CatalogValidationError("catalog entry IDs must be distinct")
    if len({entry["object_id"] for entry in entries}) != 2 or len(
        {entry["content_sha256"] for entry in entries}
    ) != 2:
        raise CatalogValidationError(
            "candidate and settlement must be distinct content-addressed objects"
        )
    race_set_digests = {entry["ordered_race_id_set_sha256"] for entry in entries}
    if len(race_set_digests) != 1:
        raise CatalogValidationError("candidate and settlement race sets differ")

    manifest_sha256 = _sha256(payload["manifest_sha256"], "release manifest_sha256")
    expected_manifest_sha256 = release_manifest_sha256(
        release_id=release_id,
        normalized_entries=entries,
    )
    if manifest_sha256 != expected_manifest_sha256:
        raise CatalogValidationError("catalog release manifest digest mismatch")

    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "release_id": release_id,
        "status": status,
        "revoked": payload["revoked"],
        "manifest_sha256": manifest_sha256,
        "authentication_receipt_sha256": _sha256(
            payload["authentication_receipt_sha256"],
            "release authentication_receipt_sha256",
        ),
        "status_receipt_sha256": _sha256(
            payload["status_receipt_sha256"], "release status_receipt_sha256"
        ),
        "entries": entries,
    }


def validate_pregrant_catalog_metadata(
    provider: PregrantCatalogMetadataProvider,
    *,
    catalog_requirements: Mapping[str, Any],
    expected_race_count: int | None = None,
) -> dict[str, Any]:
    """Resolve exactly one ACTIVE candidate/settlement release from metadata.

    This pregrant operation deliberately has no blob/row/content reader.  It
    returns only a normalized metadata binding suitable for a frozen run scope.
    """

    requirements = _normalize_requirements(catalog_requirements)
    if expected_race_count is not None:
        expected_race_count = _positive_integer(
            expected_race_count, "expected_race_count"
        )
    try:
        raw_releases = provider.list_authenticated_release_metadata()
    except Exception as exc:
        raise CatalogValidationError(
            "authenticated catalog metadata is unavailable; fail closed"
        ) from exc
    if not isinstance(raw_releases, Sequence) or isinstance(
        raw_releases, (str, bytes, bytearray)
    ):
        raise CatalogValidationError("catalog provider must return a release sequence")

    releases = [
        _normalize_release(
            release,
            requirements=requirements,
            expected_race_count=expected_race_count,
        )
        for release in raw_releases
    ]
    if len({release["release_id"] for release in releases}) != len(releases):
        raise CatalogValidationError("catalog release IDs are duplicated")
    if len({release["manifest_sha256"] for release in releases}) != len(releases):
        raise CatalogValidationError("catalog release manifests are duplicated")

    active = [
        release
        for release in releases
        if release["status"] == "ACTIVE" and release["revoked"] is False
    ]
    if len(active) != 1:
        raise CatalogValidationError("catalog must have exactly one ACTIVE release")

    selected = active[0]
    entries_by_role = {entry["role"]: entry for entry in selected["entries"]}
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "release_id": selected["release_id"],
        "release_manifest_sha256": selected["manifest_sha256"],
        "authentication_receipt_sha256": selected[
            "authentication_receipt_sha256"
        ],
        "status_receipt_sha256": selected["status_receipt_sha256"],
        "candidate_entry": entries_by_role[CANDIDATE_ROLE],
        "settlement_entry": entries_by_role[SETTLEMENT_ROLE],
        "ordered_race_id_set_sha256": selected["entries"][0][
            "ordered_race_id_set_sha256"
        ],
        "race_count": selected["entries"][0]["race_count"],
        "metadata_only": True,
    }


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CANDIDATE_ROLE",
    "CatalogValidationError",
    "EXPECTED_KEY_COLUMNS",
    "EXPECTED_ROLES",
    "EXPECTED_SETTLEMENT_COLUMNS",
    "PregrantCatalogMetadataProvider",
    "SETTLEMENT_ROLE",
    "canonical_json_bytes",
    "canonical_sha256",
    "release_manifest_sha256",
    "validate_pregrant_catalog_metadata",
]
