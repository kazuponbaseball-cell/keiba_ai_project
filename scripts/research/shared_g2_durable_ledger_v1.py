from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


SCHEMA_VERSION = 1
ZERO_SHA256 = "0" * 64

FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
SAFE_OPAQUE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,255}$")
BASE64URL_VALUE = re.compile(r"^[A-Za-z0-9_-]{43,4096}$")

AUTHENTICATED_ENVELOPE_KIND = "SHARED_G2_AUTHENTICATED_REMOTE_ENVELOPE_V1"
GLOBAL_HEAD_KIND = "SHARED_G2_GLOBAL_HEAD_V1"
WITNESS_CHECKPOINT_KIND = "SHARED_G2_MONOTONIC_WITNESS_CHECKPOINT_V1"
SUBJECT_HEAD_SNAPSHOT_KIND = "SHARED_G2_SUBJECT_HEAD_SNAPSHOT_V1"
CUTOVER_RECEIPT_KIND = "SHARED_G2_CUTOVER_ACTIVATED_V1"
TRANSACTION_REQUEST_KIND = "SHARED_G2_TRANSACTION_REQUEST_V1"
TRANSACTION_RECEIPT_KIND = "SHARED_G2_TRANSACTION_COMMITTED_V1"

AUTHENTICATION_FIELDS = frozenset(
    {
        "scheme",
        "key_id",
        "signature",
        "attestation_digest",
        "issued_at",
    }
)
ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "envelope_kind",
        "payload_type",
        "payload_digest",
        "payload",
        "authentication",
    }
)
SAFETY_FIELDS = frozenset({"formal_buy", "send_order", "stake"})

SUBJECT_KINDS = frozenset(
    {
        "CATALOG_RELEASE",
        "COMMENT_ID",
        "EXACT_SUBJECT",
        "LANE",
        "LEASE",
        "LEASE_SET",
        "QUESTION_FAMILY",
        "RECIPE",
        "RUN",
        "PHASE_OUTPUT",
        "SEMANTIC_SUBJECT",
    }
)

OPERATION_REQUIRED_SUBJECT_KINDS: dict[str, frozenset[str]] = {
    "RND_SCOPE_SEAL": frozenset({"RUN"}),
    "RND_APPROVAL_RESERVE": frozenset(
        {"COMMENT_ID", "RUN", "SEMANTIC_SUBJECT", "EXACT_SUBJECT"}
    ),
    "RND_PREACCESS_ABORT": frozenset(
        {"RUN", "SEMANTIC_SUBJECT", "EXACT_SUBJECT", "LEASE_SET"}
    ),
    "RND_DECISION_LEASE_BATCH_ISSUE": frozenset({"LEASE", "RUN"}),
    "RND_PHASE_LEASE_ISSUE": frozenset({"LEASE"}),
    "RND_DECISION_IRREVERSIBLE_START": frozenset(
        {"RUN", "LEASE", "SEMANTIC_SUBJECT", "EXACT_SUBJECT", "QUESTION_FAMILY"}
    ),
    "RND_PHASE_LEASE_CONSUME": frozenset({"LEASE"}),
    "RND_PHASE_OUTPUT_SEAL": frozenset({"PHASE_OUTPUT"}),
    "RND_RESULT_SEAL": frozenset({"PHASE_OUTPUT", "RUN"}),
    "RND_COMPLETE": frozenset({"RUN"}),
    "RND_FAIL_CLOSE": frozenset({"RUN"}),
}
ALLOWED_OPERATION_KINDS = frozenset(OPERATION_REQUIRED_SUBJECT_KINDS)
OPERATION_MUTATION_ACTIONS: Mapping[str, str] = MappingProxyType(
    {
        "RND_SCOPE_SEAL": "SEAL_RUN_SCOPE",
        "RND_APPROVAL_RESERVE": "RESERVE_APPROVED_RUN",
        "RND_PREACCESS_ABORT": "ABORT_PREACCESS_AND_TOMBSTONE",
        "RND_DECISION_LEASE_BATCH_ISSUE": "ISSUE_DECISION_LEASE_BATCH",
        "RND_PHASE_LEASE_ISSUE": "ISSUE_ONE_SHOT_PHASE_LEASE",
        "RND_DECISION_IRREVERSIBLE_START": (
            "CONSUME_DECISION_LEASE_BATCH_IRREVERSIBLY"
        ),
        "RND_PHASE_LEASE_CONSUME": "CONSUME_ONE_SHOT_PHASE_LEASE",
        "RND_PHASE_OUTPUT_SEAL": "SEAL_TRUSTED_PHASE_OUTPUT",
        "RND_RESULT_SEAL": "SEAL_TRUSTED_PHASE_OUTPUT",
        "RND_COMPLETE": "COMPLETE_SEALED_RUN",
        "RND_FAIL_CLOSE": "INVALIDATE_RUN_FAIL_CLOSED",
    }
)
OPERATION_REQUIRED_SUBJECT_KIND_COUNTS: Mapping[str, Mapping[str, int]] = (
    MappingProxyType(
        {
            operation: MappingProxyType(
                {
                    kind: (
                        2
                        if kind == "LEASE"
                        and operation
                        in {
                            "RND_DECISION_LEASE_BATCH_ISSUE",
                            "RND_DECISION_IRREVERSIBLE_START",
                        }
                        else 1
                    )
                    for kind in kinds
                }
            )
            for operation, kinds in OPERATION_REQUIRED_SUBJECT_KINDS.items()
        }
    )
)

RUN_LIFECYCLE_STATE_KIND = "SHARED_G2_REGISTERED_DIAGNOSTIC_RUN_STATE_V1"
RUN_LIFECYCLE_STATES = frozenset(
    {
        "RND_RUN_SCOPE_FROZEN",
        "RND_RUN_APPROVAL_REQUIRED",
        "RND_APPROVED",
        "RND_LEASED",
        "RND_RUNNING",
        "RND_RESULT_SEALED",
        "RND_COMPLETED",
        "INVALID",
    }
)
RUN_LIFECYCLE_TRANSITIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "RND_RUN_SCOPE_FROZEN": frozenset(
            {"RND_RUN_APPROVAL_REQUIRED", "INVALID"}
        ),
        "RND_RUN_APPROVAL_REQUIRED": frozenset({"RND_APPROVED", "INVALID"}),
        "RND_APPROVED": frozenset({"RND_LEASED", "INVALID"}),
        "RND_LEASED": frozenset({"RND_RUNNING", "INVALID"}),
        "RND_RUNNING": frozenset({"RND_RESULT_SEALED", "INVALID"}),
        "RND_RESULT_SEALED": frozenset({"RND_COMPLETED", "INVALID"}),
        "RND_COMPLETED": frozenset(),
        "INVALID": frozenset(),
    }
)
RUN_LIFECYCLE_CANONICAL_SEQUENCES: Mapping[str, int] = MappingProxyType(
    {
        "RND_RUN_SCOPE_FROZEN": 0,
        "RND_RUN_APPROVAL_REQUIRED": 1,
        "RND_APPROVED": 2,
        "RND_LEASED": 3,
        "RND_RUNNING": 4,
        "RND_RESULT_SEALED": 5,
        "RND_COMPLETED": 6,
    }
)
SINGLE_USE_SUBJECT_STATE_KIND = (
    "SHARED_G2_REGISTERED_DIAGNOSTIC_SINGLE_USE_SUBJECT_STATE_V1"
)
QUESTION_FAMILY_STATE_KIND = (
    "SHARED_G2_REGISTERED_DIAGNOSTIC_QUESTION_FAMILY_STATE_V1"
)
SINGLE_USE_SUBJECT_STATES = frozenset(
    {
        "PROVISIONALLY_RESERVED",
        "RELEASED_PREACCESS_ABORT",
        "IRREVERSIBLY_CONSUMED",
    }
)
RUN_FAIL_CLOSE_CODES = frozenset(
    {
        "AUTHORITY_UNAVAILABLE",
        "AUTHENTICATION_FAILURE",
        "CAPABILITY_VIOLATION",
        "CONTRACT_VIOLATION",
        "DIGEST_DRIFT",
        "EXPIRED_LEASE",
        "HEAD_OR_WITNESS_STALE",
        "REPLICA_MISMATCH",
    }
)

FORBIDDEN_AUTHENTICATION_SCHEMES = frozenset(
    {
        "DISABLED",
        "FIXTURE",
        "IN_MEMORY",
        "LOCAL",
        "NONE",
        "PLAINTEXT",
        "SELF_ASSERTED",
        "TEST",
        "UNCONFIGURED",
    }
)


class SharedG2Error(RuntimeError):
    """Base class for shared-G2 fail-closed errors."""


class SharedG2Unavailable(SharedG2Error):
    """The authenticated remote authority or witness cannot be used."""


class SharedG2ValidationError(SharedG2Error):
    """A remote object does not satisfy the frozen shared-G2 contract."""


class SharedG2AuthenticationError(SharedG2ValidationError):
    """An authenticated envelope cannot be verified by its external trust root."""


class SharedG2StaleAuthority(SharedG2ValidationError):
    """A head, epoch, receipt, or independent witness is stale or forked."""


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SharedG2ValidationError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def strict_json_loads(text: str, *, label: str = "JSON") -> Any:
    if not isinstance(text, str):
        raise SharedG2ValidationError(f"{label} must be UTF-8 JSON text")

    def reject_constant(value: str) -> None:
        raise SharedG2ValidationError(f"non-standard JSON constant is forbidden: {value}")

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=_strict_object_pairs,
        )
    except (json.JSONDecodeError, SharedG2ValidationError) as exc:
        raise SharedG2ValidationError(f"cannot parse strict {label}: {exc}") from exc


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SharedG2ValidationError("NaN and Infinity are forbidden")
        return 0.0 if value == 0.0 else value
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise SharedG2ValidationError("JSON object keys must be strings")
        return {key: _canonical_value(item) for key, item in value.items()}
    raise SharedG2ValidationError(
        f"unsupported canonical JSON value type: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    normalized = _canonical_value(value)
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SharedG2ValidationError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def parse_utc_timestamp(value: Any, label: str) -> datetime:
    text = _string(value, label)
    if not text.endswith("Z"):
        raise SharedG2ValidationError(f"{label} must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise SharedG2ValidationError(f"{label} is not a valid RFC3339 timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise SharedG2ValidationError(f"{label} must be UTC")
    return parsed


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SharedG2ValidationError(f"{label} must be an object")
    payload = dict(value)
    canonical_json_bytes(payload)
    return payload


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    payload = _object(value, label)
    missing = sorted(fields - set(payload))
    extra = sorted(set(payload) - fields)
    if missing:
        raise SharedG2ValidationError(
            f"{label} is missing required field(s): {', '.join(missing)}"
        )
    if extra:
        raise SharedG2ValidationError(
            f"{label} contains unexpected field(s): {', '.join(extra)}"
        )
    return payload


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SharedG2ValidationError(
            f"{label} must be a non-empty string without outer whitespace"
        )
    return value


def _identifier(value: Any, label: str) -> str:
    text = _string(value, label)
    if not SAFE_IDENTIFIER.fullmatch(text):
        raise SharedG2ValidationError(f"{label} must be a safe identifier")
    return text


def _opaque_identifier(value: Any, label: str) -> str:
    text = _string(value, label)
    if not SAFE_OPAQUE_IDENTIFIER.fullmatch(text):
        raise SharedG2ValidationError(f"{label} must be a safe opaque identifier")
    return text


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if not FULL_SHA256.fullmatch(text):
        raise SharedG2ValidationError(f"{label} must be a lowercase full SHA-256")
    return text


def _nonzero_sha256(value: Any, label: str) -> str:
    digest = _sha256(value, label)
    if digest == ZERO_SHA256:
        raise SharedG2ValidationError(f"{label} must not be the all-zero digest")
    return digest


def _git_sha(value: Any, label: str) -> str:
    text = _string(value, label)
    if not FULL_GIT_SHA.fullmatch(text):
        raise SharedG2ValidationError(f"{label} must be a lowercase full Git SHA")
    return text


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SharedG2ValidationError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    number = _nonnegative_int(value, label)
    if number == 0:
        raise SharedG2ValidationError(f"{label} must be positive")
    return number


def _safety(value: Any, label: str) -> dict[str, Any]:
    payload = _exact(value, SAFETY_FIELDS, label)
    if payload != {"formal_buy": False, "send_order": False, "stake": 0}:
        raise SharedG2ValidationError(
            f"{label} must keep formal_buy=false, send_order=false, stake=0"
        )
    return payload


def _digest_map(value: Any, label: str) -> dict[str, str]:
    payload = _object(value, label)
    if not payload:
        raise SharedG2ValidationError(f"{label} must not be empty")
    result: dict[str, str] = {}
    for key, item in payload.items():
        normalized_key = _identifier(key, f"{label} key")
        result[normalized_key] = _nonzero_sha256(
            item, f"{label}.{normalized_key}"
        )
    if list(result) != sorted(result):
        raise SharedG2ValidationError(f"{label} keys must be Unicode-sorted")
    return result


@runtime_checkable
class AuthenticatedEnvelopeVerifier(Protocol):
    """External trust-root verifier; implementations must not trust the workload."""

    def verify_authenticated_payload(
        self,
        *,
        domain_separator: str,
        payload: bytes,
        authentication: Mapping[str, Any],
    ) -> bool:
        ...


@runtime_checkable
class RemoteLedgerTransport(Protocol):
    """Authenticated transactional G2 transport implemented outside this repository."""

    def fetch_current_head(self) -> Mapping[str, Any]:
        ...

    def commit_transaction(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def fetch_transaction_receipt(self, receipt_payload_digest: str) -> Mapping[str, Any]:
        ...

    def fetch_subject_head_snapshot(
        self,
        *,
        subject_kind: str,
        subject_digest: str,
        generation: int,
        global_sequence: int,
        global_head_digest: str,
    ) -> Mapping[str, Any]:
        ...


@runtime_checkable
class RemoteMonotonicWitnessTransport(Protocol):
    """Independent witness transport; it must not share the ledger authority identity."""

    def fetch_checkpoint(
        self,
        *,
        authority_id: str,
        activation_epoch: str,
        global_sequence: int,
        global_head_digest: str,
    ) -> Mapping[str, Any]:
        ...


class UnconfiguredSharedG2Adapter:
    """Runtime default that deliberately provides no authority and no fallback."""

    _MESSAGE = (
        "shared external G2 authority is unconfigured; local, in-memory, SQLite, "
        "worktree, branch, and process-memory authority fallbacks are forbidden"
    )

    @staticmethod
    def _fail() -> None:
        raise SharedG2Unavailable(UnconfiguredSharedG2Adapter._MESSAGE)

    def fetch_current_head(self) -> Mapping[str, Any]:
        self._fail()

    def commit_transaction(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        self._fail()

    def fetch_transaction_receipt(self, receipt_payload_digest: str) -> Mapping[str, Any]:
        del receipt_payload_digest
        self._fail()

    def fetch_subject_head_snapshot(
        self,
        *,
        subject_kind: str,
        subject_digest: str,
        generation: int,
        global_sequence: int,
        global_head_digest: str,
    ) -> Mapping[str, Any]:
        del (
            subject_kind,
            subject_digest,
            generation,
            global_sequence,
            global_head_digest,
        )
        self._fail()

    def fetch_checkpoint(
        self,
        *,
        authority_id: str,
        activation_epoch: str,
        global_sequence: int,
        global_head_digest: str,
    ) -> Mapping[str, Any]:
        del authority_id, activation_epoch, global_sequence, global_head_digest
        self._fail()

    def verify_authenticated_payload(
        self,
        *,
        domain_separator: str,
        payload: bytes,
        authentication: Mapping[str, Any],
    ) -> bool:
        del domain_separator, payload, authentication
        self._fail()


@dataclass(frozen=True)
class AuthenticatedPayload:
    payload_type: str
    payload_digest: str
    payload: dict[str, Any]
    authentication: dict[str, Any]
    envelope_digest: str


def validate_authenticated_identity_binding(
    envelope: AuthenticatedPayload,
    *,
    expected_identity_digest: str,
    label: str,
) -> str:
    """Bind a verified envelope to the configured signer/attester identity."""
    expected = _nonzero_sha256(
        expected_identity_digest, f"{label}.expected_identity_digest"
    )
    observed = _nonzero_sha256(
        envelope.authentication["attestation_digest"],
        f"{label}.authentication.attestation_digest",
    )
    if observed != expected:
        raise SharedG2AuthenticationError(
            f"{label} was authenticated by an unexpected identity"
        )
    return observed


def validate_authenticated_envelope(
    value: Any,
    *,
    expected_payload_type: str,
    verifier: AuthenticatedEnvelopeVerifier,
) -> AuthenticatedPayload:
    envelope = _exact(value, ENVELOPE_FIELDS, "authenticated remote envelope")
    if envelope["schema_version"] != SCHEMA_VERSION or isinstance(
        envelope["schema_version"], bool
    ):
        raise SharedG2ValidationError("authenticated envelope schema_version must be 1")
    if envelope["envelope_kind"] != AUTHENTICATED_ENVELOPE_KIND:
        raise SharedG2ValidationError("authenticated envelope kind is invalid")
    payload_type = _identifier(envelope["payload_type"], "authenticated payload type")
    if payload_type != expected_payload_type:
        raise SharedG2ValidationError(
            f"authenticated payload type mismatch: {payload_type!r} != {expected_payload_type!r}"
        )
    raw_payload = _object(envelope["payload"], "authenticated payload")
    payload_bytes = canonical_json_bytes(raw_payload)
    payload = _object(
        strict_json_loads(
            payload_bytes.decode("utf-8"), label="authenticated payload"
        ),
        "authenticated payload",
    )
    payload_digest = _sha256(envelope["payload_digest"], "authenticated payload digest")
    observed_digest = hashlib.sha256(payload_bytes).hexdigest()
    if payload_digest != observed_digest:
        raise SharedG2ValidationError("authenticated payload digest mismatch")

    authentication = _exact(
        envelope["authentication"], AUTHENTICATION_FIELDS, "authentication evidence"
    )
    scheme = _identifier(authentication["scheme"], "authentication scheme")
    if scheme.upper() in FORBIDDEN_AUTHENTICATION_SCHEMES:
        raise SharedG2AuthenticationError(
            f"authentication scheme {scheme!r} cannot establish runtime authority"
        )
    _opaque_identifier(authentication["key_id"], "authentication key_id")
    signature = _string(authentication["signature"], "authentication signature")
    if not BASE64URL_VALUE.fullmatch(signature):
        raise SharedG2AuthenticationError(
            "authentication signature must be a non-trivial base64url value"
        )
    _nonzero_sha256(
        authentication["attestation_digest"], "authentication attestation digest"
    )
    parse_utc_timestamp(authentication["issued_at"], "authentication issued_at")

    try:
        verified = verifier.verify_authenticated_payload(
            domain_separator=f"keiba-ai/shared-g2/v1/{payload_type}",
            payload=payload_bytes,
            authentication=authentication,
        )
    except SharedG2Error:
        raise
    except Exception as exc:
        raise SharedG2AuthenticationError(
            f"external authentication verifier failed closed: {exc}"
        ) from exc
    if verified is not True:
        raise SharedG2AuthenticationError("external authentication verifier rejected payload")

    return AuthenticatedPayload(
        payload_type=payload_type,
        payload_digest=payload_digest,
        payload=payload,
        authentication=authentication,
        envelope_digest=canonical_digest(envelope),
    )


GLOBAL_HEAD_FIELDS = frozenset(
    {
        "schema_version",
        "object_kind",
        "authority_id",
        "activation_epoch",
        "backend_identity_digest",
        "cutover_receipt_digest",
        "sequence",
        "head_digest",
        "observed_at",
        "safety",
    }
)


@dataclass(frozen=True)
class GlobalHead:
    authority_id: str
    activation_epoch: str
    backend_identity_digest: str
    cutover_receipt_digest: str
    sequence: int
    head_digest: str
    observed_at: str

    def position(self) -> tuple[int, str]:
        return self.sequence, self.head_digest

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "object_kind": GLOBAL_HEAD_KIND,
            "authority_id": self.authority_id,
            "activation_epoch": self.activation_epoch,
            "backend_identity_digest": self.backend_identity_digest,
            "cutover_receipt_digest": self.cutover_receipt_digest,
            "sequence": self.sequence,
            "head_digest": self.head_digest,
            "observed_at": self.observed_at,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }


def validate_global_head(value: Any, *, label: str = "global head") -> GlobalHead:
    payload = _exact(value, GLOBAL_HEAD_FIELDS, label)
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError(f"{label}.schema_version must be 1")
    if payload["object_kind"] != GLOBAL_HEAD_KIND:
        raise SharedG2ValidationError(f"{label}.object_kind is invalid")
    _safety(payload["safety"], f"{label}.safety")
    observed_at = _string(payload["observed_at"], f"{label}.observed_at")
    parse_utc_timestamp(observed_at, f"{label}.observed_at")
    return GlobalHead(
        authority_id=_identifier(payload["authority_id"], f"{label}.authority_id"),
        activation_epoch=_identifier(
            payload["activation_epoch"], f"{label}.activation_epoch"
        ),
        backend_identity_digest=_nonzero_sha256(
            payload["backend_identity_digest"], f"{label}.backend_identity_digest"
        ),
        cutover_receipt_digest=_nonzero_sha256(
            payload["cutover_receipt_digest"], f"{label}.cutover_receipt_digest"
        ),
        sequence=_nonnegative_int(payload["sequence"], f"{label}.sequence"),
        head_digest=_nonzero_sha256(payload["head_digest"], f"{label}.head_digest"),
        observed_at=observed_at,
    )


SUBJECT_HEAD_FIELDS = frozenset(
    {
        "subject_kind",
        "subject_digest",
        "generation",
        "sequence",
        "head_digest",
        "state_digest",
    }
)


@dataclass(frozen=True)
class SubjectHead:
    subject_kind: str
    subject_digest: str
    generation: int
    sequence: int
    head_digest: str
    state_digest: str

    def key(self) -> tuple[str, str, int]:
        return self.subject_kind, self.subject_digest, self.generation

    def to_wire(self) -> dict[str, Any]:
        return {
            "subject_kind": self.subject_kind,
            "subject_digest": self.subject_digest,
            "generation": self.generation,
            "sequence": self.sequence,
            "head_digest": self.head_digest,
            "state_digest": self.state_digest,
        }


def validate_subject_head(value: Any, *, label: str = "subject head") -> SubjectHead:
    payload = _exact(value, SUBJECT_HEAD_FIELDS, label)
    subject_kind = _identifier(payload["subject_kind"], f"{label}.subject_kind")
    if subject_kind not in SUBJECT_KINDS:
        raise SharedG2ValidationError(f"{label}.subject_kind is not registered")
    return SubjectHead(
        subject_kind=subject_kind,
        subject_digest=_nonzero_sha256(
            payload["subject_digest"], f"{label}.subject_digest"
        ),
        generation=_nonnegative_int(payload["generation"], f"{label}.generation"),
        sequence=_nonnegative_int(payload["sequence"], f"{label}.sequence"),
        head_digest=_sha256(payload["head_digest"], f"{label}.head_digest"),
        state_digest=_nonzero_sha256(
            payload["state_digest"], f"{label}.state_digest"
        ),
    )


SUBJECT_HEAD_SNAPSHOT_FIELDS = frozenset(
    {
        "schema_version",
        "snapshot_kind",
        "authority_id",
        "activation_epoch",
        "backend_identity_digest",
        "cutover_receipt_digest",
        "global_head",
        "subject_head",
        "read_at",
        "safety",
    }
)


@dataclass(frozen=True)
class SubjectHeadSnapshot:
    global_head: GlobalHead
    subject_head: SubjectHead
    read_at: str
    payload_digest: str
    envelope_digest: str


def validate_subject_head_snapshot(
    value: Any,
    *,
    context: "CutoverContext",
    expected_global_head: GlobalHead,
    expected_subject_kind: str,
    expected_subject_digest: str,
    expected_generation: int,
    verifier: AuthenticatedEnvelopeVerifier,
) -> SubjectHeadSnapshot:
    envelope = validate_authenticated_envelope(
        value, expected_payload_type=SUBJECT_HEAD_SNAPSHOT_KIND, verifier=verifier
    )
    validate_authenticated_identity_binding(
        envelope,
        expected_identity_digest=context.expectations.backend_identity_digest,
        label="subject head snapshot",
    )
    payload = _exact(
        envelope.payload, SUBJECT_HEAD_SNAPSHOT_FIELDS, "subject head snapshot"
    )
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError("subject head snapshot schema_version must be 1")
    if payload["snapshot_kind"] != SUBJECT_HEAD_SNAPSHOT_KIND:
        raise SharedG2ValidationError("subject head snapshot kind is invalid")
    _safety(payload["safety"], "subject head snapshot.safety")
    expected = context.expectations
    if (
        _identifier(payload["authority_id"], "subject head snapshot.authority_id")
        != expected.authority_id
        or _identifier(
            payload["activation_epoch"], "subject head snapshot.activation_epoch"
        )
        != expected.activation_epoch
        or _nonzero_sha256(
            payload["backend_identity_digest"],
            "subject head snapshot.backend_identity_digest",
        )
        != expected.backend_identity_digest
        or _nonzero_sha256(
            payload["cutover_receipt_digest"],
            "subject head snapshot.cutover_receipt_digest",
        )
        != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority("subject head snapshot authority mismatch")
    global_head = validate_global_head(
        payload["global_head"], label="subject head snapshot.global_head"
    )
    if global_head != expected_global_head:
        raise SharedG2StaleAuthority(
            "subject head snapshot does not bind the requested global head"
        )
    subject_head = validate_subject_head(
        payload["subject_head"], label="subject head snapshot.subject_head"
    )
    if (
        subject_head.subject_kind != expected_subject_kind
        or subject_head.subject_digest != expected_subject_digest
        or subject_head.generation != expected_generation
    ):
        raise SharedG2StaleAuthority(
            "subject head snapshot returned a different subject identity"
        )
    read_at = _string(payload["read_at"], "subject head snapshot.read_at")
    if parse_utc_timestamp(
        read_at, "subject head snapshot.read_at"
    ) < parse_utc_timestamp(
        global_head.observed_at, "subject head snapshot.global_head.observed_at"
    ):
        raise SharedG2ValidationError(
            "subject head snapshot read time predates its global head"
        )
    return SubjectHeadSnapshot(
        global_head=global_head,
        subject_head=subject_head,
        read_at=read_at,
        payload_digest=envelope.payload_digest,
        envelope_digest=envelope.envelope_digest,
    )


def _subject_head_list(value: Any, label: str) -> tuple[SubjectHead, ...]:
    if not isinstance(value, list):
        raise SharedG2ValidationError(f"{label} must be a list")
    heads = tuple(
        validate_subject_head(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    keys = [head.key() for head in heads]
    if keys != sorted(keys):
        raise SharedG2ValidationError(f"{label} must be sorted by canonical subject key")
    if len(keys) != len(set(keys)):
        raise SharedG2ValidationError(f"{label} contains duplicate subject keys")
    return heads


SUBJECT_MUTATION_FIELDS = frozenset(
    {
        "subject_kind",
        "subject_digest",
        "generation",
        "expected_sequence",
        "expected_head_digest",
        "new_state_digest",
    }
)


@dataclass(frozen=True)
class SubjectMutation:
    subject_kind: str
    subject_digest: str
    generation: int
    expected_sequence: int
    expected_head_digest: str
    new_state_digest: str

    def __post_init__(self) -> None:
        subject_kind = _identifier(self.subject_kind, "subject mutation.subject_kind")
        if subject_kind not in SUBJECT_KINDS:
            raise SharedG2ValidationError(
                "subject mutation.subject_kind is not registered"
            )
        object.__setattr__(self, "subject_kind", subject_kind)
        object.__setattr__(
            self,
            "subject_digest",
            _nonzero_sha256(self.subject_digest, "subject mutation.subject_digest"),
        )
        object.__setattr__(
            self,
            "generation",
            _nonnegative_int(self.generation, "subject mutation.generation"),
        )
        object.__setattr__(
            self,
            "expected_sequence",
            _nonnegative_int(
                self.expected_sequence, "subject mutation.expected_sequence"
            ),
        )
        object.__setattr__(
            self,
            "expected_head_digest",
            _sha256(
                self.expected_head_digest,
                "subject mutation.expected_head_digest",
            ),
        )
        object.__setattr__(
            self,
            "new_state_digest",
            _nonzero_sha256(
                self.new_state_digest, "subject mutation.new_state_digest"
            ),
        )

    def key(self) -> tuple[str, str, int]:
        return self.subject_kind, self.subject_digest, self.generation

    def to_wire(self) -> dict[str, Any]:
        return {
            "subject_kind": self.subject_kind,
            "subject_digest": self.subject_digest,
            "generation": self.generation,
            "expected_sequence": self.expected_sequence,
            "expected_head_digest": self.expected_head_digest,
            "new_state_digest": self.new_state_digest,
        }


def validate_subject_mutation(
    value: Any, *, label: str = "subject mutation"
) -> SubjectMutation:
    payload = _exact(value, SUBJECT_MUTATION_FIELDS, label)
    subject_kind = _identifier(payload["subject_kind"], f"{label}.subject_kind")
    if subject_kind not in SUBJECT_KINDS:
        raise SharedG2ValidationError(f"{label}.subject_kind is not registered")
    return SubjectMutation(
        subject_kind=subject_kind,
        subject_digest=_nonzero_sha256(
            payload["subject_digest"], f"{label}.subject_digest"
        ),
        generation=_nonnegative_int(payload["generation"], f"{label}.generation"),
        expected_sequence=_nonnegative_int(
            payload["expected_sequence"], f"{label}.expected_sequence"
        ),
        expected_head_digest=_sha256(
            payload["expected_head_digest"], f"{label}.expected_head_digest"
        ),
        new_state_digest=_nonzero_sha256(
            payload["new_state_digest"], f"{label}.new_state_digest"
        ),
    )


@dataclass(frozen=True)
class RunLifecycleState:
    """Canonical run lifecycle state; arbitrary state payloads are not accepted."""

    run_scope_digest: str
    generation: int
    lifecycle_state: str
    lifecycle_sequence: int
    predecessor_state_digest: str
    transition_evidence_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_scope_digest",
            _nonzero_sha256(
                self.run_scope_digest, "run lifecycle state.run_scope_digest"
            ),
        )
        object.__setattr__(
            self,
            "generation",
            _nonnegative_int(self.generation, "run lifecycle state.generation"),
        )
        state = _identifier(
            self.lifecycle_state, "run lifecycle state.lifecycle_state"
        )
        if state not in RUN_LIFECYCLE_STATES:
            raise SharedG2ValidationError(
                f"run lifecycle state is not registered: {state!r}"
            )
        object.__setattr__(self, "lifecycle_state", state)
        sequence = _nonnegative_int(
            self.lifecycle_sequence, "run lifecycle state.lifecycle_sequence"
        )
        object.__setattr__(self, "lifecycle_sequence", sequence)
        if state != "INVALID" and sequence != RUN_LIFECYCLE_CANONICAL_SEQUENCES[state]:
            raise SharedG2ValidationError(
                "run lifecycle state has a non-canonical lifecycle sequence"
            )
        if state == "INVALID" and sequence not in range(1, 7):
            raise SharedG2ValidationError(
                "INVALID run lifecycle state sequence must identify one bounded predecessor"
            )
        predecessor = _sha256(
            self.predecessor_state_digest,
            "run lifecycle state.predecessor_state_digest",
        )
        evidence = _sha256(
            self.transition_evidence_digest,
            "run lifecycle state.transition_evidence_digest",
        )
        if sequence == 0:
            if predecessor != ZERO_SHA256 or evidence != ZERO_SHA256:
                raise SharedG2ValidationError(
                    "initial run lifecycle state must use zero predecessor/evidence digests"
                )
            if state != "RND_RUN_SCOPE_FROZEN":
                raise SharedG2ValidationError(
                    "initial run lifecycle state must be RND_RUN_SCOPE_FROZEN"
                )
        elif predecessor == ZERO_SHA256 or evidence == ZERO_SHA256:
            raise SharedG2ValidationError(
                "non-initial run lifecycle state requires predecessor and evidence digests"
            )
        object.__setattr__(self, "predecessor_state_digest", predecessor)
        object.__setattr__(self, "transition_evidence_digest", evidence)

    @classmethod
    def initial(cls, *, run_scope_digest: str, generation: int) -> "RunLifecycleState":
        return cls(
            run_scope_digest=run_scope_digest,
            generation=generation,
            lifecycle_state="RND_RUN_SCOPE_FROZEN",
            lifecycle_sequence=0,
            predecessor_state_digest=ZERO_SHA256,
            transition_evidence_digest=ZERO_SHA256,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "state_kind": RUN_LIFECYCLE_STATE_KIND,
            "run_scope_digest": self.run_scope_digest,
            "generation": self.generation,
            "lifecycle_state": self.lifecycle_state,
            "lifecycle_sequence": self.lifecycle_sequence,
            "predecessor_state_digest": self.predecessor_state_digest,
            "transition_evidence_digest": self.transition_evidence_digest,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())


@dataclass(frozen=True)
class RunLifecycleTransition:
    previous: RunLifecycleState
    current: RunLifecycleState
    mutation: SubjectMutation


def build_run_lifecycle_transition(
    *,
    current_head: SubjectHead,
    current_state: RunLifecycleState,
    new_state: str,
    transition_evidence_digest: str,
) -> RunLifecycleTransition:
    """Build a finite state transition and its exact RUN CAS mutation."""
    if (
        current_head.subject_kind != "RUN"
        or current_head.subject_digest != current_state.run_scope_digest
        or current_head.generation != current_state.generation
        or current_head.state_digest != current_state.digest
    ):
        raise SharedG2StaleAuthority(
            "RUN subject head does not bind the supplied canonical lifecycle state"
        )
    destination = _identifier(new_state, "run lifecycle transition destination")
    allowed = RUN_LIFECYCLE_TRANSITIONS[current_state.lifecycle_state]
    if destination not in allowed:
        raise SharedG2ValidationError(
            f"illegal run lifecycle transition: "
            f"{current_state.lifecycle_state} -> {destination}"
        )
    evidence = _nonzero_sha256(
        transition_evidence_digest,
        "run lifecycle transition evidence digest",
    )
    next_state = RunLifecycleState(
        run_scope_digest=current_state.run_scope_digest,
        generation=current_state.generation,
        lifecycle_state=destination,
        lifecycle_sequence=current_state.lifecycle_sequence + 1,
        predecessor_state_digest=current_state.digest,
        transition_evidence_digest=evidence,
    )
    mutation = SubjectMutation(
        subject_kind="RUN",
        subject_digest=current_head.subject_digest,
        generation=current_head.generation,
        expected_sequence=current_head.sequence,
        expected_head_digest=current_head.head_digest,
        new_state_digest=next_state.digest,
    )
    return RunLifecycleTransition(
        previous=current_state,
        current=next_state,
        mutation=mutation,
    )


def build_run_completion_transition(
    *,
    current_head: SubjectHead,
    current_state: RunLifecycleState,
    result_seal_receipt_digest: str,
) -> RunLifecycleTransition:
    evidence = canonical_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "transition": "RND_RESULT_SEALED_TO_RND_COMPLETED",
            "run_scope_digest": current_state.run_scope_digest,
            "result_seal_receipt_digest": _nonzero_sha256(
                result_seal_receipt_digest, "result seal receipt digest"
            ),
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
    )
    return build_run_lifecycle_transition(
        current_head=current_head,
        current_state=current_state,
        new_state="RND_COMPLETED",
        transition_evidence_digest=evidence,
    )


def build_run_invalid_transition(
    *,
    current_head: SubjectHead,
    current_state: RunLifecycleState,
    failure_code: str,
    authenticated_failure_evidence_digest: str,
) -> RunLifecycleTransition:
    code = _identifier(failure_code, "run fail-close code")
    if code not in RUN_FAIL_CLOSE_CODES:
        raise SharedG2ValidationError(
            f"run fail-close code is not registered: {code!r}"
        )
    evidence = canonical_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "transition": f"{current_state.lifecycle_state}_TO_INVALID",
            "run_scope_digest": current_state.run_scope_digest,
            "failure_code": code,
            "authenticated_failure_evidence_digest": _nonzero_sha256(
                authenticated_failure_evidence_digest,
                "authenticated failure evidence digest",
            ),
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
    )
    return build_run_lifecycle_transition(
        current_head=current_head,
        current_state=current_state,
        new_state="INVALID",
        transition_evidence_digest=evidence,
    )


@dataclass(frozen=True)
class SingleUseSubjectState:
    subject_kind: str
    subject_digest: str
    generation: int
    run_scope_digest: str
    reservation_state: str
    state_sequence: int
    predecessor_state_digest: str
    transition_evidence_digest: str

    def __post_init__(self) -> None:
        kind = _identifier(
            self.subject_kind, "single-use subject state.subject_kind"
        )
        if kind not in {"SEMANTIC_SUBJECT", "EXACT_SUBJECT"}:
            raise SharedG2ValidationError(
                "single-use state supports only semantic and exact subjects"
            )
        state = _identifier(
            self.reservation_state, "single-use subject state.reservation_state"
        )
        if state not in SINGLE_USE_SUBJECT_STATES:
            raise SharedG2ValidationError(
                f"single-use reservation state is not registered: {state!r}"
            )
        sequence = _nonnegative_int(
            self.state_sequence, "single-use subject state.state_sequence"
        )
        predecessor = _sha256(
            self.predecessor_state_digest,
            "single-use subject state.predecessor_state_digest",
        )
        evidence = _nonzero_sha256(
            self.transition_evidence_digest,
            "single-use subject state.transition_evidence_digest",
        )
        if sequence == 0:
            if state != "PROVISIONALLY_RESERVED" or predecessor != ZERO_SHA256:
                raise SharedG2ValidationError(
                    "initial single-use state must be a zero-predecessor provisional reservation"
                )
        elif sequence != 1 or state == "PROVISIONALLY_RESERVED" or predecessor == ZERO_SHA256:
            raise SharedG2ValidationError(
                "single-use terminal state must be the one bounded transition from its reservation"
            )
        object.__setattr__(self, "subject_kind", kind)
        object.__setattr__(
            self,
            "subject_digest",
            _nonzero_sha256(
                self.subject_digest, "single-use subject state.subject_digest"
            ),
        )
        object.__setattr__(
            self,
            "generation",
            _nonnegative_int(
                self.generation, "single-use subject state.generation"
            ),
        )
        object.__setattr__(
            self,
            "run_scope_digest",
            _nonzero_sha256(
                self.run_scope_digest, "single-use subject state.run_scope_digest"
            ),
        )
        object.__setattr__(self, "reservation_state", state)
        object.__setattr__(self, "state_sequence", sequence)
        object.__setattr__(self, "predecessor_state_digest", predecessor)
        object.__setattr__(self, "transition_evidence_digest", evidence)

    @classmethod
    def provisional(
        cls,
        *,
        subject_kind: str,
        subject_digest: str,
        generation: int,
        run_scope_digest: str,
        approval_receipt_digest: str,
    ) -> "SingleUseSubjectState":
        return cls(
            subject_kind=subject_kind,
            subject_digest=subject_digest,
            generation=generation,
            run_scope_digest=run_scope_digest,
            reservation_state="PROVISIONALLY_RESERVED",
            state_sequence=0,
            predecessor_state_digest=ZERO_SHA256,
            transition_evidence_digest=approval_receipt_digest,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "state_kind": SINGLE_USE_SUBJECT_STATE_KIND,
            "subject_kind": self.subject_kind,
            "subject_digest": self.subject_digest,
            "generation": self.generation,
            "run_scope_digest": self.run_scope_digest,
            "reservation_state": self.reservation_state,
            "state_sequence": self.state_sequence,
            "predecessor_state_digest": self.predecessor_state_digest,
            "transition_evidence_digest": self.transition_evidence_digest,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())


@dataclass(frozen=True)
class SingleUseSubjectTransition:
    previous: SingleUseSubjectState
    current: SingleUseSubjectState
    mutation: SubjectMutation


def build_single_use_subject_transition(
    *,
    current_head: SubjectHead,
    current_state: SingleUseSubjectState,
    new_state: str,
    transition_evidence_digest: str,
) -> SingleUseSubjectTransition:
    if (
        current_head.subject_kind != current_state.subject_kind
        or current_head.subject_digest != current_state.subject_digest
        or current_head.generation != current_state.generation
        or current_head.state_digest != current_state.digest
    ):
        raise SharedG2StaleAuthority(
            "single-use subject head does not bind the canonical reservation state"
        )
    destination = _identifier(new_state, "single-use transition destination")
    if (
        current_state.reservation_state != "PROVISIONALLY_RESERVED"
        or destination
        not in {"RELEASED_PREACCESS_ABORT", "IRREVERSIBLY_CONSUMED"}
    ):
        raise SharedG2ValidationError(
            f"illegal single-use transition: "
            f"{current_state.reservation_state} -> {destination}"
        )
    next_state = SingleUseSubjectState(
        subject_kind=current_state.subject_kind,
        subject_digest=current_state.subject_digest,
        generation=current_state.generation,
        run_scope_digest=current_state.run_scope_digest,
        reservation_state=destination,
        state_sequence=current_state.state_sequence + 1,
        predecessor_state_digest=current_state.digest,
        transition_evidence_digest=_nonzero_sha256(
            transition_evidence_digest,
            "single-use transition evidence digest",
        ),
    )
    mutation = SubjectMutation(
        subject_kind=current_head.subject_kind,
        subject_digest=current_head.subject_digest,
        generation=current_head.generation,
        expected_sequence=current_head.sequence,
        expected_head_digest=current_head.head_digest,
        new_state_digest=next_state.digest,
    )
    return SingleUseSubjectTransition(
        previous=current_state,
        current=next_state,
        mutation=mutation,
    )


@dataclass(frozen=True)
class QuestionFamilyState:
    question_family_digest: str
    execution_count: int
    state_sequence: int
    predecessor_state_digest: str
    transition_evidence_digest: str

    def __post_init__(self) -> None:
        count = _nonnegative_int(
            self.execution_count, "question family state.execution_count"
        )
        if count not in {0, 1}:
            raise SharedG2ValidationError(
                "registered diagnostic question-family execution count is bounded to 0 or 1"
            )
        sequence = _nonnegative_int(
            self.state_sequence, "question family state.state_sequence"
        )
        predecessor = _sha256(
            self.predecessor_state_digest,
            "question family state.predecessor_state_digest",
        )
        evidence = _sha256(
            self.transition_evidence_digest,
            "question family state.transition_evidence_digest",
        )
        if count == 0:
            if sequence != 0 or predecessor != ZERO_SHA256 or evidence != ZERO_SHA256:
                raise SharedG2ValidationError(
                    "unused question family must be the zero-predecessor genesis state"
                )
        elif sequence != 1 or predecessor == ZERO_SHA256 or evidence == ZERO_SHA256:
            raise SharedG2ValidationError(
                "irreversibly used question family must be the single bounded increment"
            )
        object.__setattr__(
            self,
            "question_family_digest",
            _nonzero_sha256(
                self.question_family_digest,
                "question family state.question_family_digest",
            ),
        )
        object.__setattr__(self, "execution_count", count)
        object.__setattr__(self, "state_sequence", sequence)
        object.__setattr__(self, "predecessor_state_digest", predecessor)
        object.__setattr__(self, "transition_evidence_digest", evidence)

    @classmethod
    def unused(cls, *, question_family_digest: str) -> "QuestionFamilyState":
        return cls(
            question_family_digest=question_family_digest,
            execution_count=0,
            state_sequence=0,
            predecessor_state_digest=ZERO_SHA256,
            transition_evidence_digest=ZERO_SHA256,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "state_kind": QUESTION_FAMILY_STATE_KIND,
            "question_family_digest": self.question_family_digest,
            "execution_count": self.execution_count,
            "state_sequence": self.state_sequence,
            "predecessor_state_digest": self.predecessor_state_digest,
            "transition_evidence_digest": self.transition_evidence_digest,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())


@dataclass(frozen=True)
class QuestionFamilyTransition:
    previous: QuestionFamilyState
    current: QuestionFamilyState
    mutation: SubjectMutation


def build_question_family_irreversible_increment(
    *,
    current_head: SubjectHead,
    current_state: QuestionFamilyState,
    transition_evidence_digest: str,
) -> QuestionFamilyTransition:
    if (
        current_head.subject_kind != "QUESTION_FAMILY"
        or current_head.subject_digest != current_state.question_family_digest
        or current_head.generation != 0
        or current_head.state_digest != current_state.digest
        or current_state.execution_count != 0
    ):
        raise SharedG2StaleAuthority(
            "question-family head is not the authenticated unused canonical state"
        )
    next_state = QuestionFamilyState(
        question_family_digest=current_state.question_family_digest,
        execution_count=1,
        state_sequence=1,
        predecessor_state_digest=current_state.digest,
        transition_evidence_digest=_nonzero_sha256(
            transition_evidence_digest,
            "question family increment evidence digest",
        ),
    )
    mutation = SubjectMutation(
        subject_kind="QUESTION_FAMILY",
        subject_digest=current_head.subject_digest,
        generation=0,
        expected_sequence=current_head.sequence,
        expected_head_digest=current_head.head_digest,
        new_state_digest=next_state.digest,
    )
    return QuestionFamilyTransition(
        previous=current_state,
        current=next_state,
        mutation=mutation,
    )


def normalize_subject_mutations(
    values: Sequence[SubjectMutation | Mapping[str, Any]],
    *,
    operation_kind: str,
) -> tuple[SubjectMutation, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise SharedG2ValidationError("subject mutations must be a non-empty sequence")
    mutations: list[SubjectMutation] = []
    for index, value in enumerate(values):
        if isinstance(value, SubjectMutation):
            mutation = validate_subject_mutation(
                value.to_wire(), label=f"subject mutations[{index}]"
            )
        else:
            mutation = validate_subject_mutation(value, label=f"subject mutations[{index}]")
        mutations.append(mutation)
    keys = [mutation.key() for mutation in mutations]
    if keys != sorted(keys):
        raise SharedG2ValidationError("subject mutations must be sorted by canonical key")
    if len(keys) != len(set(keys)):
        raise SharedG2ValidationError("subject mutations contain duplicate subject keys")
    required = OPERATION_REQUIRED_SUBJECT_KINDS.get(operation_kind)
    if required is None:
        raise SharedG2ValidationError(f"unknown shared-G2 operation: {operation_kind!r}")
    observed = {mutation.subject_kind for mutation in mutations}
    if observed != required:
        missing = sorted(required - observed)
        extra = sorted(observed - required)
        raise SharedG2ValidationError(
            f"{operation_kind} subject mutation kinds must be exact; "
            f"missing={missing}, extra={extra}"
        )
    observed_counts = {
        kind: sum(1 for mutation in mutations if mutation.subject_kind == kind)
        for kind in observed
    }
    expected_counts = dict(OPERATION_REQUIRED_SUBJECT_KIND_COUNTS[operation_kind])
    if observed_counts != expected_counts:
        raise SharedG2ValidationError(
            f"{operation_kind} subject mutation multiplicity must be exact; "
            f"expected={expected_counts}, observed={observed_counts}"
        )
    return tuple(mutations)


CUTOVER_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_kind",
        "authority_id",
        "activation_epoch",
        "repository",
        "base_branch",
        "current_main_sha",
        "backend_identity_digest",
        "witness_identity_digest",
        "runtime_blob_digests",
        "migration_digests",
        "old_writer_fence_digest",
        "second_remote_compare_digest",
        "initial_global_head",
        "initial_witness_checkpoint_digest",
        "writer_identity_digest",
        "activated_at",
        "safety",
    }
)
MIGRATION_DIGEST_FIELDS = frozenset(
    {
        "legacy_event_chain_digest",
        "global_comment_id_set_digest",
        "terminal_and_nonterminal_subject_head_set_digest",
    }
)


@dataclass(frozen=True)
class CutoverExpectations:
    repository: str
    base_branch: str
    current_main_sha: str
    authority_id: str
    activation_epoch: str
    backend_identity_digest: str
    witness_identity_digest: str
    runtime_blob_digests: Mapping[str, str]
    migration_digests: Mapping[str, str]
    old_writer_fence_digest: str
    second_remote_compare_digest: str

    def __post_init__(self) -> None:
        _identifier(self.authority_id, "cutover expectation authority_id")
        _identifier(self.activation_epoch, "cutover expectation activation_epoch")
        _string(self.repository, "cutover expectation repository")
        _string(self.base_branch, "cutover expectation base_branch")
        _git_sha(self.current_main_sha, "cutover expectation current_main_sha")
        backend = _nonzero_sha256(
            self.backend_identity_digest, "cutover expectation backend_identity_digest"
        )
        witness = _nonzero_sha256(
            self.witness_identity_digest, "cutover expectation witness_identity_digest"
        )
        if backend == witness:
            raise SharedG2ValidationError(
                "ledger backend and external witness identities must be distinct"
            )
        runtime_digests = _digest_map(
            self.runtime_blob_digests,
            "cutover expectation runtime_blob_digests",
        )
        migration = _exact(
            self.migration_digests,
            MIGRATION_DIGEST_FIELDS,
            "cutover expectation migration_digests",
        )
        normalized_migration = {
            key: _nonzero_sha256(
                value, f"cutover expectation migration_digests.{key}"
            )
            for key, value in migration.items()
        }
        _nonzero_sha256(
            self.old_writer_fence_digest,
            "cutover expectation old_writer_fence_digest",
        )
        _nonzero_sha256(
            self.second_remote_compare_digest,
            "cutover expectation second_remote_compare_digest",
        )
        object.__setattr__(
            self, "runtime_blob_digests", MappingProxyType(runtime_digests)
        )
        object.__setattr__(
            self, "migration_digests", MappingProxyType(normalized_migration)
        )


@dataclass(frozen=True)
class CutoverContext:
    expectations: CutoverExpectations
    cutover_receipt_digest: str
    authenticated_payload_digest: str
    initial_global_head: GlobalHead
    initial_witness_checkpoint_digest: str
    activated_at: str


def _cutover_binding_digest(payload: Mapping[str, Any]) -> str:
    """Hash a bootstrap-safe projection without a self-referential head field."""
    projection = dict(_object(payload, "cutover receipt digest projection"))
    initial_head = dict(
        _object(
            projection["initial_global_head"],
            "cutover receipt digest projection.initial_global_head",
        )
    )
    initial_head["cutover_receipt_digest"] = ZERO_SHA256
    projection["initial_global_head"] = initial_head
    return canonical_digest(projection)


def validate_cutover_receipt(
    value: Any,
    *,
    expectations: CutoverExpectations,
    verifier: AuthenticatedEnvelopeVerifier,
) -> CutoverContext:
    envelope = validate_authenticated_envelope(
        value, expected_payload_type=CUTOVER_RECEIPT_KIND, verifier=verifier
    )
    validate_authenticated_identity_binding(
        envelope,
        expected_identity_digest=expectations.backend_identity_digest,
        label="cutover receipt",
    )
    payload = _exact(envelope.payload, CUTOVER_RECEIPT_FIELDS, "cutover receipt")
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError("cutover receipt schema_version must be 1")
    if payload["receipt_kind"] != CUTOVER_RECEIPT_KIND:
        raise SharedG2ValidationError("cutover receipt kind is invalid")
    _safety(payload["safety"], "cutover receipt.safety")
    observed = {
        "repository": _string(payload["repository"], "cutover receipt.repository"),
        "base_branch": _string(payload["base_branch"], "cutover receipt.base_branch"),
        "current_main_sha": _git_sha(
            payload["current_main_sha"], "cutover receipt.current_main_sha"
        ),
        "authority_id": _identifier(payload["authority_id"], "cutover receipt.authority_id"),
        "activation_epoch": _identifier(
            payload["activation_epoch"], "cutover receipt.activation_epoch"
        ),
        "backend_identity_digest": _sha256(
            payload["backend_identity_digest"], "cutover receipt.backend_identity_digest"
        ),
        "witness_identity_digest": _sha256(
            payload["witness_identity_digest"], "cutover receipt.witness_identity_digest"
        ),
        "old_writer_fence_digest": _sha256(
            payload["old_writer_fence_digest"], "cutover receipt.old_writer_fence_digest"
        ),
        "second_remote_compare_digest": _sha256(
            payload["second_remote_compare_digest"],
            "cutover receipt.second_remote_compare_digest",
        ),
    }
    for field, expected in (
        ("repository", expectations.repository),
        ("base_branch", expectations.base_branch),
        ("current_main_sha", expectations.current_main_sha),
        ("authority_id", expectations.authority_id),
        ("activation_epoch", expectations.activation_epoch),
        ("backend_identity_digest", expectations.backend_identity_digest),
        ("witness_identity_digest", expectations.witness_identity_digest),
        ("old_writer_fence_digest", expectations.old_writer_fence_digest),
        ("second_remote_compare_digest", expectations.second_remote_compare_digest),
    ):
        if observed[field] != expected:
            raise SharedG2StaleAuthority(f"cutover receipt {field} mismatch")
    if observed["backend_identity_digest"] == observed["witness_identity_digest"]:
        raise SharedG2ValidationError(
            "cutover receipt ledger and witness identities must be distinct"
        )
    runtime_digests = _digest_map(
        payload["runtime_blob_digests"], "cutover receipt.runtime_blob_digests"
    )
    if runtime_digests != dict(expectations.runtime_blob_digests):
        raise SharedG2StaleAuthority("cutover receipt runtime blob digest set mismatch")
    migration_digests = _exact(
        payload["migration_digests"],
        MIGRATION_DIGEST_FIELDS,
        "cutover receipt.migration_digests",
    )
    normalized_migration = {
        key: _sha256(value, f"cutover receipt.migration_digests.{key}")
        for key, value in migration_digests.items()
    }
    if normalized_migration != dict(expectations.migration_digests):
        raise SharedG2StaleAuthority("cutover receipt migration digest set mismatch")

    initial_head = validate_global_head(
        payload["initial_global_head"], label="cutover receipt.initial_global_head"
    )
    cutover_binding_digest = _cutover_binding_digest(payload)
    if (
        initial_head.authority_id != expectations.authority_id
        or initial_head.activation_epoch != expectations.activation_epoch
        or initial_head.backend_identity_digest != expectations.backend_identity_digest
        or initial_head.cutover_receipt_digest != cutover_binding_digest
    ):
        raise SharedG2StaleAuthority("cutover receipt initial global head binding mismatch")
    if initial_head.sequence != 0 or initial_head.head_digest == ZERO_SHA256:
        raise SharedG2ValidationError(
            "cutover receipt initial global head must be non-zero sequence-0 genesis"
        )
    initial_witness_digest = _nonzero_sha256(
        payload["initial_witness_checkpoint_digest"],
        "cutover receipt.initial_witness_checkpoint_digest",
    )
    _nonzero_sha256(
        payload["writer_identity_digest"], "cutover receipt.writer_identity_digest"
    )
    activated_at = _string(payload["activated_at"], "cutover receipt.activated_at")
    activated_time = parse_utc_timestamp(activated_at, "cutover receipt.activated_at")
    if parse_utc_timestamp(
        initial_head.observed_at, "cutover receipt.initial_global_head.observed_at"
    ) > activated_time:
        raise SharedG2ValidationError(
            "cutover initial global head cannot postdate cutover activation"
        )
    return CutoverContext(
        expectations=expectations,
        cutover_receipt_digest=cutover_binding_digest,
        authenticated_payload_digest=envelope.payload_digest,
        initial_global_head=initial_head,
        initial_witness_checkpoint_digest=initial_witness_digest,
        activated_at=activated_at,
    )


WITNESS_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_kind",
        "authority_id",
        "activation_epoch",
        "backend_identity_digest",
        "witness_identity_digest",
        "cutover_receipt_digest",
        "checkpoint_sequence",
        "observed_global_head",
        "previous_checkpoint_digest",
        "checkpoint_digest",
        "witnessed_at",
        "safety",
    }
)


@dataclass(frozen=True)
class WitnessCheckpoint:
    authority_id: str
    activation_epoch: str
    backend_identity_digest: str
    witness_identity_digest: str
    cutover_receipt_digest: str
    checkpoint_sequence: int
    observed_global_head: GlobalHead
    previous_checkpoint_digest: str
    checkpoint_digest: str
    witnessed_at: str


def validate_witness_checkpoint(
    value: Any,
    *,
    context: CutoverContext,
    expected_head: GlobalHead,
    verifier: AuthenticatedEnvelopeVerifier,
) -> WitnessCheckpoint:
    envelope = validate_authenticated_envelope(
        value, expected_payload_type=WITNESS_CHECKPOINT_KIND, verifier=verifier
    )
    validate_authenticated_identity_binding(
        envelope,
        expected_identity_digest=context.expectations.witness_identity_digest,
        label="monotonic witness checkpoint",
    )
    payload = _exact(
        envelope.payload, WITNESS_CHECKPOINT_FIELDS, "monotonic witness checkpoint"
    )
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError("witness checkpoint schema_version must be 1")
    if payload["checkpoint_kind"] != WITNESS_CHECKPOINT_KIND:
        raise SharedG2ValidationError("witness checkpoint kind is invalid")
    _safety(payload["safety"], "witness checkpoint.safety")
    authority_id = _identifier(payload["authority_id"], "witness checkpoint.authority_id")
    activation_epoch = _identifier(
        payload["activation_epoch"], "witness checkpoint.activation_epoch"
    )
    backend_digest = _sha256(
        payload["backend_identity_digest"], "witness checkpoint.backend_identity_digest"
    )
    witness_digest = _sha256(
        payload["witness_identity_digest"], "witness checkpoint.witness_identity_digest"
    )
    cutover_digest = _sha256(
        payload["cutover_receipt_digest"], "witness checkpoint.cutover_receipt_digest"
    )
    expected = context.expectations
    if (
        authority_id != expected.authority_id
        or activation_epoch != expected.activation_epoch
        or backend_digest != expected.backend_identity_digest
        or witness_digest != expected.witness_identity_digest
        or cutover_digest != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority("witness checkpoint authority binding mismatch")
    if backend_digest == witness_digest:
        raise SharedG2ValidationError("external witness identity equals ledger backend identity")
    observed_head = validate_global_head(
        payload["observed_global_head"], label="witness checkpoint.observed_global_head"
    )
    if observed_head.position() != expected_head.position():
        raise SharedG2StaleAuthority("witness checkpoint does not bind the requested global head")
    _require_same_authority(observed_head, expected_head, "witness global head")
    checkpoint_sequence = _positive_int(
        payload["checkpoint_sequence"], "witness checkpoint.checkpoint_sequence"
    )
    if checkpoint_sequence != observed_head.sequence + 1:
        raise SharedG2StaleAuthority(
            "witness checkpoint sequence must equal global head sequence plus one"
        )
    previous_digest = _sha256(
        payload["previous_checkpoint_digest"],
        "witness checkpoint.previous_checkpoint_digest",
    )
    checkpoint_digest = _sha256(
        payload["checkpoint_digest"], "witness checkpoint.checkpoint_digest"
    )
    digest_projection = dict(payload)
    digest_projection.pop("checkpoint_digest")
    if canonical_digest(digest_projection) != checkpoint_digest:
        raise SharedG2ValidationError("witness checkpoint self-digest mismatch")
    if observed_head.sequence == 0 and (
        checkpoint_digest != context.initial_witness_checkpoint_digest
        or previous_digest != ZERO_SHA256
    ):
        raise SharedG2StaleAuthority(
            "genesis witness checkpoint does not match the authenticated cutover"
        )
    witnessed_at = _string(payload["witnessed_at"], "witness checkpoint.witnessed_at")
    witnessed_time = parse_utc_timestamp(
        witnessed_at, "witness checkpoint.witnessed_at"
    )
    if witnessed_time < parse_utc_timestamp(
        observed_head.observed_at,
        "witness checkpoint.observed_global_head.observed_at",
    ):
        raise SharedG2ValidationError(
            "witness checkpoint cannot predate the observed global head"
        )
    return WitnessCheckpoint(
        authority_id=authority_id,
        activation_epoch=activation_epoch,
        backend_identity_digest=backend_digest,
        witness_identity_digest=witness_digest,
        cutover_receipt_digest=cutover_digest,
        checkpoint_sequence=checkpoint_sequence,
        observed_global_head=observed_head,
        previous_checkpoint_digest=previous_digest,
        checkpoint_digest=checkpoint_digest,
        witnessed_at=witnessed_at,
    )


def _require_same_authority(left: GlobalHead, right: GlobalHead, label: str) -> None:
    if (
        left.authority_id != right.authority_id
        or left.activation_epoch != right.activation_epoch
        or left.backend_identity_digest != right.backend_identity_digest
        or left.cutover_receipt_digest != right.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority(f"{label} authority or cutover binding mismatch")


MUTATION_PAYLOAD_REQUIRED_FIELDS = frozenset(
    {"schema_version", "action", "operation_kind", "safety"}
)


def _validate_mutation_payload(
    value: Any, *, operation_kind: str
) -> dict[str, Any]:
    payload = _object(value, "transaction mutation payload")
    missing = sorted(MUTATION_PAYLOAD_REQUIRED_FIELDS - set(payload))
    if missing:
        raise SharedG2ValidationError(
            "transaction mutation payload is missing inspectable control field(s): "
            + ", ".join(missing)
        )
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError(
            "transaction mutation payload schema_version must be 1"
        )
    action = _identifier(
        payload["action"], "transaction mutation payload.action"
    )
    if action != OPERATION_MUTATION_ACTIONS[operation_kind]:
        raise SharedG2ValidationError(
            "transaction mutation payload action is not registered for its operation"
        )
    if (
        _identifier(
            payload["operation_kind"],
            "transaction mutation payload.operation_kind",
        )
        != operation_kind
    ):
        raise SharedG2ValidationError(
            "transaction mutation payload operation does not match request"
        )
    _safety(payload["safety"], "transaction mutation payload.safety")
    # Canonical round-trip detaches the authority request from caller-owned mappings.
    return _object(
        strict_json_loads(
            canonical_json_bytes(payload).decode("utf-8"),
            label="transaction mutation payload",
        ),
        "transaction mutation payload",
    )


TRANSACTION_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "request_kind",
        "operation_kind",
        "operation_id",
        "authority_id",
        "activation_epoch",
        "backend_identity_digest",
        "cutover_receipt_digest",
        "idempotency_key",
        "run_scope_digest",
        "mutation_digest",
        "mutation_payload",
        "expected_output_type",
        "expected_global_head",
        "subject_mutations",
        "requested_at",
        "safety",
    }
)


@dataclass(frozen=True)
class TransactionRequest:
    operation_kind: str
    operation_id: str
    authority_id: str
    activation_epoch: str
    backend_identity_digest: str
    cutover_receipt_digest: str
    idempotency_key: str
    run_scope_digest: str
    mutation_digest: str
    mutation_payload: dict[str, Any]
    expected_output_type: str
    expected_global_head: GlobalHead
    subject_mutations: tuple[SubjectMutation, ...]
    requested_at: str

    @classmethod
    def build(
        cls,
        *,
        operation_kind: str,
        operation_id: str,
        context: CutoverContext,
        idempotency_key: str,
        run_scope_digest: str,
        mutation_payload: Mapping[str, Any],
        expected_output_type: str,
        expected_global_head: GlobalHead,
        subject_mutations: Sequence[SubjectMutation | Mapping[str, Any]],
        requested_at: str,
    ) -> "TransactionRequest":
        operation = _identifier(operation_kind, "transaction operation_kind")
        if operation not in ALLOWED_OPERATION_KINDS:
            raise SharedG2ValidationError(f"unsupported transaction operation: {operation!r}")
        _require_context_head(context, expected_global_head, "transaction expected head")
        mutations = normalize_subject_mutations(subject_mutations, operation_kind=operation)
        timestamp = _string(requested_at, "transaction requested_at")
        parse_utc_timestamp(timestamp, "transaction requested_at")
        normalized_mutation_payload = _validate_mutation_payload(
            mutation_payload, operation_kind=operation
        )
        return cls(
            operation_kind=operation,
            operation_id=_identifier(operation_id, "transaction operation_id"),
            authority_id=context.expectations.authority_id,
            activation_epoch=context.expectations.activation_epoch,
            backend_identity_digest=context.expectations.backend_identity_digest,
            cutover_receipt_digest=context.cutover_receipt_digest,
            idempotency_key=_opaque_identifier(idempotency_key, "transaction idempotency_key"),
            run_scope_digest=_nonzero_sha256(
                run_scope_digest, "transaction run_scope_digest"
            ),
            mutation_digest=canonical_digest(normalized_mutation_payload),
            mutation_payload=normalized_mutation_payload,
            expected_output_type=_identifier(
                expected_output_type, "transaction expected_output_type"
            ),
            expected_global_head=expected_global_head,
            subject_mutations=mutations,
            requested_at=timestamp,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "request_kind": TRANSACTION_REQUEST_KIND,
            "operation_kind": self.operation_kind,
            "operation_id": self.operation_id,
            "authority_id": self.authority_id,
            "activation_epoch": self.activation_epoch,
            "backend_identity_digest": self.backend_identity_digest,
            "cutover_receipt_digest": self.cutover_receipt_digest,
            "idempotency_key": self.idempotency_key,
            "run_scope_digest": self.run_scope_digest,
            "mutation_digest": self.mutation_digest,
            "mutation_payload": self.mutation_payload,
            "expected_output_type": self.expected_output_type,
            "expected_global_head": self.expected_global_head.to_wire(),
            "subject_mutations": [mutation.to_wire() for mutation in self.subject_mutations],
            "requested_at": self.requested_at,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())


def validate_transaction_request(value: Any) -> TransactionRequest:
    payload = _exact(value, TRANSACTION_REQUEST_FIELDS, "transaction request")
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError("transaction request schema_version must be 1")
    if payload["request_kind"] != TRANSACTION_REQUEST_KIND:
        raise SharedG2ValidationError("transaction request kind is invalid")
    _safety(payload["safety"], "transaction request.safety")
    operation = _identifier(payload["operation_kind"], "transaction request.operation_kind")
    if operation not in ALLOWED_OPERATION_KINDS:
        raise SharedG2ValidationError("transaction request operation is not registered")
    expected_head = validate_global_head(
        payload["expected_global_head"], label="transaction request.expected_global_head"
    )
    mutations_raw = payload["subject_mutations"]
    if not isinstance(mutations_raw, list):
        raise SharedG2ValidationError("transaction request.subject_mutations must be a list")
    mutations = normalize_subject_mutations(mutations_raw, operation_kind=operation)
    requested_at = _string(payload["requested_at"], "transaction request.requested_at")
    parse_utc_timestamp(requested_at, "transaction request.requested_at")
    mutation_payload = _validate_mutation_payload(
        payload["mutation_payload"], operation_kind=operation
    )
    mutation_digest = _nonzero_sha256(
        payload["mutation_digest"], "transaction request.mutation_digest"
    )
    if mutation_digest != canonical_digest(mutation_payload):
        raise SharedG2ValidationError(
            "transaction mutation digest does not bind its inspectable payload"
        )
    return TransactionRequest(
        operation_kind=operation,
        operation_id=_identifier(payload["operation_id"], "transaction request.operation_id"),
        authority_id=_identifier(payload["authority_id"], "transaction request.authority_id"),
        activation_epoch=_identifier(
            payload["activation_epoch"], "transaction request.activation_epoch"
        ),
        backend_identity_digest=_sha256(
            payload["backend_identity_digest"],
            "transaction request.backend_identity_digest",
        ),
        cutover_receipt_digest=_sha256(
            payload["cutover_receipt_digest"],
            "transaction request.cutover_receipt_digest",
        ),
        idempotency_key=_opaque_identifier(
            payload["idempotency_key"], "transaction request.idempotency_key"
        ),
        run_scope_digest=_nonzero_sha256(
            payload["run_scope_digest"], "transaction request.run_scope_digest"
        ),
        mutation_digest=mutation_digest,
        mutation_payload=mutation_payload,
        expected_output_type=_identifier(
            payload["expected_output_type"], "transaction request.expected_output_type"
        ),
        expected_global_head=expected_head,
        subject_mutations=mutations,
        requested_at=requested_at,
    )


TRANSACTION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_kind",
        "authority_id",
        "activation_epoch",
        "backend_identity_digest",
        "cutover_receipt_digest",
        "transaction_id",
        "idempotency_key",
        "operation_kind",
        "operation_id",
        "request_digest",
        "run_scope_digest",
        "mutation_digest",
        "previous_global_head",
        "new_global_head",
        "previous_subject_heads",
        "new_subject_heads",
        "operation_output_type",
        "operation_output_digest",
        "writer_identity_digest",
        "committed_at",
        "safety",
    }
)


@dataclass(frozen=True)
class TransactionReceipt:
    payload_digest: str
    envelope_digest: str
    transaction_id: str
    operation_kind: str
    operation_id: str
    request_digest: str
    run_scope_digest: str
    mutation_digest: str
    previous_global_head: GlobalHead
    new_global_head: GlobalHead
    previous_subject_heads: tuple[SubjectHead, ...]
    new_subject_heads: tuple[SubjectHead, ...]
    operation_output_type: str
    operation_output_digest: str
    committed_at: str


def validate_transaction_receipt(
    value: Any,
    *,
    request: TransactionRequest,
    context: CutoverContext,
    verifier: AuthenticatedEnvelopeVerifier,
) -> TransactionReceipt:
    envelope = validate_authenticated_envelope(
        value, expected_payload_type=TRANSACTION_RECEIPT_KIND, verifier=verifier
    )
    validate_authenticated_identity_binding(
        envelope,
        expected_identity_digest=context.expectations.backend_identity_digest,
        label="transaction receipt",
    )
    payload = _exact(
        envelope.payload, TRANSACTION_RECEIPT_FIELDS, "transaction receipt"
    )
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError("transaction receipt schema_version must be 1")
    if payload["receipt_kind"] != TRANSACTION_RECEIPT_KIND:
        raise SharedG2ValidationError("transaction receipt kind is invalid")
    _safety(payload["safety"], "transaction receipt.safety")
    if (
        _identifier(payload["authority_id"], "transaction receipt.authority_id")
        != context.expectations.authority_id
        or _identifier(
            payload["activation_epoch"], "transaction receipt.activation_epoch"
        )
        != context.expectations.activation_epoch
        or _sha256(
            payload["backend_identity_digest"],
            "transaction receipt.backend_identity_digest",
        )
        != context.expectations.backend_identity_digest
        or _sha256(
            payload["cutover_receipt_digest"],
            "transaction receipt.cutover_receipt_digest",
        )
        != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority("transaction receipt authority binding mismatch")

    checks = {
        "idempotency_key": _opaque_identifier(
            payload["idempotency_key"], "transaction receipt.idempotency_key"
        ),
        "operation_kind": _identifier(
            payload["operation_kind"], "transaction receipt.operation_kind"
        ),
        "operation_id": _identifier(
            payload["operation_id"], "transaction receipt.operation_id"
        ),
        "request_digest": _sha256(
            payload["request_digest"], "transaction receipt.request_digest"
        ),
        "run_scope_digest": _sha256(
            payload["run_scope_digest"], "transaction receipt.run_scope_digest"
        ),
        "mutation_digest": _sha256(
            payload["mutation_digest"], "transaction receipt.mutation_digest"
        ),
        "operation_output_type": _identifier(
            payload["operation_output_type"], "transaction receipt.operation_output_type"
        ),
    }
    expected_checks = {
        "idempotency_key": request.idempotency_key,
        "operation_kind": request.operation_kind,
        "operation_id": request.operation_id,
        "request_digest": request.digest,
        "run_scope_digest": request.run_scope_digest,
        "mutation_digest": request.mutation_digest,
        "operation_output_type": request.expected_output_type,
    }
    if checks != expected_checks:
        raise SharedG2ValidationError("transaction receipt does not bind the exact request")

    previous_global = validate_global_head(
        payload["previous_global_head"], label="transaction receipt.previous_global_head"
    )
    new_global = validate_global_head(
        payload["new_global_head"], label="transaction receipt.new_global_head"
    )
    _require_context_head(context, previous_global, "transaction previous head")
    _require_context_head(context, new_global, "transaction new head")
    if previous_global.position() != request.expected_global_head.position():
        raise SharedG2StaleAuthority("transaction receipt previous global head CAS mismatch")
    if new_global.sequence != previous_global.sequence + 1:
        raise SharedG2ValidationError("transaction must advance global sequence exactly once")
    if new_global.head_digest == previous_global.head_digest:
        raise SharedG2ValidationError("transaction must change the global head digest")

    previous_subjects = _subject_head_list(
        payload["previous_subject_heads"], "transaction receipt.previous_subject_heads"
    )
    new_subjects = _subject_head_list(
        payload["new_subject_heads"], "transaction receipt.new_subject_heads"
    )
    mutation_by_key = {mutation.key(): mutation for mutation in request.subject_mutations}
    previous_by_key = {head.key(): head for head in previous_subjects}
    new_by_key = {head.key(): head for head in new_subjects}
    if set(previous_by_key) != set(mutation_by_key) or set(new_by_key) != set(mutation_by_key):
        raise SharedG2ValidationError(
            "transaction subject head sets do not exactly match requested mutations"
        )
    for key, mutation in mutation_by_key.items():
        previous = previous_by_key[key]
        new = new_by_key[key]
        if (
            previous.sequence != mutation.expected_sequence
            or previous.head_digest != mutation.expected_head_digest
        ):
            raise SharedG2StaleAuthority(
                f"transaction subject CAS mismatch for {mutation.subject_kind}"
            )
        if new.sequence != previous.sequence + 1:
            raise SharedG2ValidationError(
                f"transaction subject sequence did not advance once for {mutation.subject_kind}"
            )
        if new.head_digest == previous.head_digest:
            raise SharedG2ValidationError(
                f"transaction subject head did not change for {mutation.subject_kind}"
            )
        if new.state_digest != mutation.new_state_digest:
            raise SharedG2ValidationError(
                f"transaction subject state mismatch for {mutation.subject_kind}"
            )

    output_digest = _nonzero_sha256(
        payload["operation_output_digest"],
        "transaction receipt.operation_output_digest",
    )
    _nonzero_sha256(
        payload["writer_identity_digest"], "transaction receipt.writer_identity_digest"
    )
    committed_at = _string(payload["committed_at"], "transaction receipt.committed_at")
    committed_time = parse_utc_timestamp(
        committed_at, "transaction receipt.committed_at"
    )
    requested_time = parse_utc_timestamp(
        request.requested_at, "transaction request.requested_at"
    )
    previous_observed_time = parse_utc_timestamp(
        previous_global.observed_at,
        "transaction receipt.previous_global_head.observed_at",
    )
    new_observed_time = parse_utc_timestamp(
        new_global.observed_at, "transaction receipt.new_global_head.observed_at"
    )
    if committed_time < requested_time or committed_time < previous_observed_time:
        raise SharedG2ValidationError(
            "transaction commit time predates its request or previous global head"
        )
    if new_observed_time < previous_observed_time:
        raise SharedG2ValidationError(
            "transaction new global head observation time is not monotonic"
        )
    return TransactionReceipt(
        payload_digest=envelope.payload_digest,
        envelope_digest=envelope.envelope_digest,
        transaction_id=_opaque_identifier(
            payload["transaction_id"], "transaction receipt.transaction_id"
        ),
        operation_kind=checks["operation_kind"],
        operation_id=checks["operation_id"],
        request_digest=checks["request_digest"],
        run_scope_digest=checks["run_scope_digest"],
        mutation_digest=checks["mutation_digest"],
        previous_global_head=previous_global,
        new_global_head=new_global,
        previous_subject_heads=previous_subjects,
        new_subject_heads=new_subjects,
        operation_output_type=checks["operation_output_type"],
        operation_output_digest=output_digest,
        committed_at=committed_at,
    )


@dataclass(frozen=True)
class AuthoritySnapshot:
    global_head: GlobalHead
    witness: WitnessCheckpoint


@dataclass(frozen=True)
class CommittedTransaction:
    receipt: TransactionReceipt
    witness: WitnessCheckpoint


def _require_context_head(context: CutoverContext, head: GlobalHead, label: str) -> None:
    expected = context.expectations
    if (
        head.authority_id != expected.authority_id
        or head.activation_epoch != expected.activation_epoch
        or head.backend_identity_digest != expected.backend_identity_digest
        or head.cutover_receipt_digest != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority(f"{label} does not bind the active cutover context")


class SharedG2AuthorityClient:
    """Fail-closed client for an injected remote ledger and independent witness."""

    def __init__(
        self,
        *,
        context: CutoverContext,
        ledger_transport: RemoteLedgerTransport,
        witness_transport: RemoteMonotonicWitnessTransport,
        ledger_envelope_verifier: AuthenticatedEnvelopeVerifier,
        witness_envelope_verifier: AuthenticatedEnvelopeVerifier,
    ) -> None:
        if isinstance(ledger_transport, UnconfiguredSharedG2Adapter):
            raise SharedG2Unavailable(UnconfiguredSharedG2Adapter._MESSAGE)
        if isinstance(witness_transport, UnconfiguredSharedG2Adapter):
            raise SharedG2Unavailable(UnconfiguredSharedG2Adapter._MESSAGE)
        self.context = context
        self._ledger = ledger_transport
        self._witness = witness_transport
        self._ledger_verifier = ledger_envelope_verifier
        self._witness_verifier = witness_envelope_verifier

    def _witness_for(self, head: GlobalHead) -> WitnessCheckpoint:
        try:
            raw = self._witness.fetch_checkpoint(
                authority_id=head.authority_id,
                activation_epoch=head.activation_epoch,
                global_sequence=head.sequence,
                global_head_digest=head.head_digest,
            )
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(
                f"external monotonic witness unavailable; halt lane: {exc}"
            ) from exc
        return validate_witness_checkpoint(
            raw,
            context=self.context,
            expected_head=head,
            verifier=self._witness_verifier,
        )

    def read_snapshot(self) -> AuthoritySnapshot:
        try:
            raw = self._ledger.fetch_current_head()
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(f"remote shared-G2 head unavailable: {exc}") from exc
        authenticated = validate_authenticated_envelope(
            raw, expected_payload_type=GLOBAL_HEAD_KIND, verifier=self._ledger_verifier
        )
        validate_authenticated_identity_binding(
            authenticated,
            expected_identity_digest=self.context.expectations.backend_identity_digest,
            label="current shared-G2 head",
        )
        head = validate_global_head(authenticated.payload)
        _require_context_head(self.context, head, "current shared-G2 head")
        witness = self._witness_for(head)
        return AuthoritySnapshot(global_head=head, witness=witness)

    def read_subject_head_snapshot(
        self,
        *,
        subject_kind: str,
        subject_digest: str,
        generation: int,
    ) -> tuple[SubjectHeadSnapshot, AuthoritySnapshot]:
        kind = _identifier(subject_kind, "subject snapshot subject_kind")
        if kind not in SUBJECT_KINDS:
            raise SharedG2ValidationError("subject snapshot kind is not registered")
        digest = _nonzero_sha256(
            subject_digest, "subject snapshot subject_digest"
        )
        normalized_generation = _nonnegative_int(
            generation, "subject snapshot generation"
        )
        authority_snapshot = self.read_snapshot()
        head = authority_snapshot.global_head
        try:
            raw = self._ledger.fetch_subject_head_snapshot(
                subject_kind=kind,
                subject_digest=digest,
                generation=normalized_generation,
                global_sequence=head.sequence,
                global_head_digest=head.head_digest,
            )
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(
                f"remote shared-G2 subject head unavailable: {exc}"
            ) from exc
        subject_snapshot = validate_subject_head_snapshot(
            raw,
            context=self.context,
            expected_global_head=head,
            expected_subject_kind=kind,
            expected_subject_digest=digest,
            expected_generation=normalized_generation,
            verifier=self._ledger_verifier,
        )
        return subject_snapshot, authority_snapshot

    def fetch_receipt_envelope(
        self, receipt_payload_digest: str
    ) -> AuthenticatedPayload:
        """Fetch authenticated bytes; full receipt validation needs the request."""
        digest = _sha256(receipt_payload_digest, "transaction receipt payload digest")
        try:
            raw = self._ledger.fetch_transaction_receipt(digest)
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(f"remote transaction receipt unavailable: {exc}") from exc
        envelope = validate_authenticated_envelope(
            raw, expected_payload_type=TRANSACTION_RECEIPT_KIND, verifier=self._ledger_verifier
        )
        validate_authenticated_identity_binding(
            envelope,
            expected_identity_digest=self.context.expectations.backend_identity_digest,
            label="fetched transaction receipt",
        )
        if envelope.payload_digest != digest:
            raise SharedG2ValidationError("fetched transaction receipt digest mismatch")
        return envelope

    def commit(self, request: TransactionRequest) -> CommittedTransaction:
        normalized_request = validate_transaction_request(request.to_wire())
        if normalized_request != request:
            raise SharedG2ValidationError("transaction request is not canonical")
        _require_context_head(
            self.context, request.expected_global_head, "transaction expected global head"
        )
        snapshot = self.read_snapshot()
        if snapshot.global_head.position() != request.expected_global_head.position():
            raise SharedG2StaleAuthority("transaction expected global head is stale")
        try:
            raw_receipt = self._ledger.commit_transaction(request.to_wire())
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(
                f"remote shared-G2 transaction failed closed: {exc}"
            ) from exc
        receipt = validate_transaction_receipt(
            raw_receipt,
            request=request,
            context=self.context,
            verifier=self._ledger_verifier,
        )
        try:
            stored_raw = self._ledger.fetch_transaction_receipt(receipt.payload_digest)
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(
                f"committed transaction receipt could not be read back: {exc}"
            ) from exc
        stored = validate_transaction_receipt(
            stored_raw,
            request=request,
            context=self.context,
            verifier=self._ledger_verifier,
        )
        if (
            stored.payload_digest != receipt.payload_digest
            or stored.envelope_digest != receipt.envelope_digest
        ):
            raise SharedG2ValidationError(
                "transaction receipt changed between commit and read-back"
            )
        witness = self._witness_for(receipt.new_global_head)
        if (
            witness.checkpoint_sequence != snapshot.witness.checkpoint_sequence + 1
            or witness.previous_checkpoint_digest
            != snapshot.witness.checkpoint_digest
        ):
            raise SharedG2StaleAuthority(
                "independent witness chain did not advance exactly from the pre-CAS checkpoint"
            )
        return CommittedTransaction(receipt=receipt, witness=witness)


def self_check_shared_g2_contract() -> None:
    if set(RUN_LIFECYCLE_CANONICAL_SEQUENCES) != (
        set(RUN_LIFECYCLE_STATES) - {"INVALID"}
    ):
        raise AssertionError("run lifecycle canonical sequences are incomplete")
    if set(OPERATION_REQUIRED_SUBJECT_KINDS) != set(ALLOWED_OPERATION_KINDS):
        raise AssertionError("operation allowlist is inconsistent")
    if set(OPERATION_REQUIRED_SUBJECT_KIND_COUNTS) != set(
        ALLOWED_OPERATION_KINDS
    ):
        raise AssertionError("operation mutation-count allowlist is inconsistent")
    if set(OPERATION_MUTATION_ACTIONS) != set(ALLOWED_OPERATION_KINDS):
        raise AssertionError("operation mutation-action allowlist is inconsistent")
    if any(not required for required in OPERATION_REQUIRED_SUBJECT_KINDS.values()):
        raise AssertionError("every operation must mutate at least one typed subject")
    for operation, required in OPERATION_REQUIRED_SUBJECT_KINDS.items():
        counts = OPERATION_REQUIRED_SUBJECT_KIND_COUNTS[operation]
        if set(counts) != set(required) or any(count < 1 for count in counts.values()):
            raise AssertionError(
                "every operation must have exact positive subject-kind multiplicities"
            )
    if OPERATION_REQUIRED_SUBJECT_KIND_COUNTS[
        "RND_DECISION_LEASE_BATCH_ISSUE"
    ]["LEASE"] != 2:
        raise AssertionError("decision lease issue must atomically create two leases")
    if OPERATION_REQUIRED_SUBJECT_KIND_COUNTS[
        "RND_DECISION_IRREVERSIBLE_START"
    ]["LEASE"] != 2:
        raise AssertionError("decision irreversible start must atomically consume two leases")
    sample = {
        "b": [1, True, None],
        "a": {"unicode": "馬"},
    }
    if canonical_json_bytes(sample) != b'{"a":{"unicode":"\xe9\xa6\xac"},"b":[1,true,null]}':
        raise AssertionError("canonical JSON contract drifted")
    adapter = UnconfiguredSharedG2Adapter()
    try:
        adapter.fetch_current_head()
    except SharedG2Unavailable:
        pass
    else:
        raise AssertionError("unconfigured adapter did not fail closed")


__all__ = [
    "ALLOWED_OPERATION_KINDS",
    "AUTHENTICATED_ENVELOPE_KIND",
    "AuthenticatedEnvelopeVerifier",
    "AuthenticatedPayload",
    "AuthoritySnapshot",
    "CommittedTransaction",
    "CutoverContext",
    "CutoverExpectations",
    "GLOBAL_HEAD_KIND",
    "GlobalHead",
    "OPERATION_REQUIRED_SUBJECT_KIND_COUNTS",
    "OPERATION_REQUIRED_SUBJECT_KINDS",
    "OPERATION_MUTATION_ACTIONS",
    "RemoteLedgerTransport",
    "RemoteMonotonicWitnessTransport",
    "RUN_FAIL_CLOSE_CODES",
    "RUN_LIFECYCLE_STATES",
    "RUN_LIFECYCLE_CANONICAL_SEQUENCES",
    "RUN_LIFECYCLE_TRANSITIONS",
    "RunLifecycleState",
    "RunLifecycleTransition",
    "SCHEMA_VERSION",
    "SUBJECT_KINDS",
    "SUBJECT_HEAD_SNAPSHOT_KIND",
    "SharedG2AuthenticationError",
    "SharedG2AuthorityClient",
    "SharedG2Error",
    "SharedG2StaleAuthority",
    "SharedG2Unavailable",
    "SharedG2ValidationError",
    "SingleUseSubjectState",
    "SingleUseSubjectTransition",
    "SubjectHead",
    "SubjectHeadSnapshot",
    "SubjectMutation",
    "TRANSACTION_RECEIPT_KIND",
    "TransactionReceipt",
    "TransactionRequest",
    "UnconfiguredSharedG2Adapter",
    "WITNESS_CHECKPOINT_KIND",
    "WitnessCheckpoint",
    "ZERO_SHA256",
    "build_question_family_irreversible_increment",
    "build_run_completion_transition",
    "build_run_invalid_transition",
    "build_run_lifecycle_transition",
    "build_single_use_subject_transition",
    "canonical_digest",
    "canonical_json_bytes",
    "normalize_subject_mutations",
    "parse_utc_timestamp",
    "self_check_shared_g2_contract",
    "strict_json_loads",
    "validate_authenticated_envelope",
    "validate_authenticated_identity_binding",
    "validate_cutover_receipt",
    "validate_global_head",
    "validate_subject_head",
    "validate_subject_head_snapshot",
    "validate_subject_mutation",
    "validate_transaction_receipt",
    "validate_transaction_request",
    "validate_witness_checkpoint",
    "QuestionFamilyState",
    "QuestionFamilyTransition",
]
