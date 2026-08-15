from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import re
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

try:
    from .shared_g2_durable_ledger_v1 import (
        AuthenticatedEnvelopeVerifier,
        AuthoritySnapshot,
        CommittedTransaction,
        CutoverContext,
        GlobalHead,
        QuestionFamilyState,
        RunLifecycleState,
        SCHEMA_VERSION,
        SingleUseSubjectState,
        SharedG2AuthorityClient,
        SharedG2Error,
        SharedG2StaleAuthority,
        SharedG2Unavailable,
        SharedG2ValidationError,
        SubjectHead,
        SubjectHeadSnapshot,
        SubjectMutation,
        TransactionRequest,
        UnconfiguredSharedG2Adapter,
        ZERO_SHA256,
        build_question_family_irreversible_increment,
        build_run_lifecycle_transition,
        build_single_use_subject_transition,
        canonical_digest,
        parse_utc_timestamp,
        validate_authenticated_envelope,
        validate_authenticated_identity_binding,
        validate_global_head,
    )
except ImportError:  # pragma: no cover - direct script/module loading
    from shared_g2_durable_ledger_v1 import (
        AuthenticatedEnvelopeVerifier,
        AuthoritySnapshot,
        CommittedTransaction,
        CutoverContext,
        GlobalHead,
        QuestionFamilyState,
        RunLifecycleState,
        SCHEMA_VERSION,
        SingleUseSubjectState,
        SharedG2AuthorityClient,
        SharedG2Error,
        SharedG2StaleAuthority,
        SharedG2Unavailable,
        SharedG2ValidationError,
        SubjectHead,
        SubjectHeadSnapshot,
        SubjectMutation,
        TransactionRequest,
        UnconfiguredSharedG2Adapter,
        ZERO_SHA256,
        build_question_family_irreversible_increment,
        build_run_lifecycle_transition,
        build_single_use_subject_transition,
        canonical_digest,
        parse_utc_timestamp,
        validate_authenticated_envelope,
        validate_authenticated_identity_binding,
        validate_global_head,
    )


PHASE_LEASE_KIND = "SHARED_G2_REGISTERED_DIAGNOSTIC_PHASE_LEASE_V1"
LEASE_CONSUMPTION_RECEIPT_KIND = (
    "SHARED_G2_REGISTERED_DIAGNOSTIC_LEASE_CONSUMPTION_V1"
)
LEASE_STATE_KIND = "SHARED_G2_REGISTERED_DIAGNOSTIC_LEASE_STATE_V1"
PHASE_OUTPUT_ATTESTATION_KIND = (
    "SHARED_G2_TRUSTED_REGISTERED_DIAGNOSTIC_PHASE_OUTPUT_ATTESTATION_V1"
)
PHASE_OUTPUT_SEAL_RECEIPT_KIND = (
    "SHARED_G2_REGISTERED_DIAGNOSTIC_PHASE_OUTPUT_SEAL_V1"
)
PHASE_OUTPUT_STATE_KIND = (
    "SHARED_G2_REGISTERED_DIAGNOSTIC_PHASE_OUTPUT_STATE_V1"
)
DECISION_LEASE_BATCH_KIND = (
    "SHARED_G2_REGISTERED_DIAGNOSTIC_DECISION_LEASE_BATCH_V1"
)
DECISION_CONSUMPTION_BATCH_KIND = (
    "SHARED_G2_REGISTERED_DIAGNOSTIC_DECISION_CONSUMPTION_BATCH_V1"
)
MAX_LEASE_TTL_SECONDS = 3_600
QUESTION_FAMILY_AGGREGATE_GENERATION = 0

LEASED_PHASES = (
    "DECISION_FREEZE",
    "SETTLEMENT_DIAGNOSTIC",
    "REPLICA_COMPARE",
    "RESULT_SEAL",
)
DECISION_REPLICA_IDS = ("clean_a", "clean_b")
REPLICA_COMPARE_ACTOR = "lane_coordinator"
RESULT_SEAL_ACTOR = "canonical_sealer"
ALLOWED_ACTORS_BY_PHASE = MappingProxyType(
    {
        "DECISION_FREEZE": frozenset(DECISION_REPLICA_IDS),
        "SETTLEMENT_DIAGNOSTIC": frozenset(DECISION_REPLICA_IDS),
        "REPLICA_COMPARE": frozenset({REPLICA_COMPARE_ACTOR}),
        "RESULT_SEAL": frozenset({RESULT_SEAL_ACTOR}),
    }
)

ISSUE_OPERATION_BY_PHASE = MappingProxyType(
    {
        "DECISION_FREEZE": "RND_DECISION_LEASE_BATCH_ISSUE",
        "SETTLEMENT_DIAGNOSTIC": "RND_PHASE_LEASE_ISSUE",
        "REPLICA_COMPARE": "RND_PHASE_LEASE_ISSUE",
        "RESULT_SEAL": "RND_PHASE_LEASE_ISSUE",
    }
)
CONSUME_OPERATION_BY_PHASE = MappingProxyType(
    {
        "DECISION_FREEZE": "RND_DECISION_IRREVERSIBLE_START",
        "SETTLEMENT_DIAGNOSTIC": "RND_PHASE_LEASE_CONSUME",
        "REPLICA_COMPARE": "RND_PHASE_LEASE_CONSUME",
        "RESULT_SEAL": "RND_PHASE_LEASE_CONSUME",
    }
)
OUTPUT_SEAL_OPERATION_BY_PHASE = MappingProxyType(
    {
        "DECISION_FREEZE": "RND_PHASE_OUTPUT_SEAL",
        "SETTLEMENT_DIAGNOSTIC": "RND_PHASE_OUTPUT_SEAL",
        "REPLICA_COMPARE": "RND_PHASE_OUTPUT_SEAL",
        "RESULT_SEAL": "RND_RESULT_SEAL",
    }
)
PHASE_CAPABILITY_PROJECTIONS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "DECISION_FREEZE": MappingProxyType(
            {
                "phase": "DECISION_FREEZE",
                "allowed_read_roles": ["candidate_only_projection"],
                "allowed_write_roles": ["decision_freeze_projection"],
                "candidate_content_access": True,
                "settlement_content_access": False,
                "result_projection_access": False,
                "odds_price_popularity_or_market_access": False,
                "payoff_access": False,
                "roi_calculation": False,
                "network_access": False,
                "credential_access": False,
                "formal_buy": False,
                "send_order": False,
                "stake": 0,
            }
        ),
        "SETTLEMENT_DIAGNOSTIC": MappingProxyType(
            {
                "phase": "SETTLEMENT_DIAGNOSTIC",
                "allowed_read_roles": [
                    "decision_freeze_projection",
                    "settlement_projection",
                ],
                "allowed_write_roles": ["replica_scientific_projection"],
                "candidate_content_access": False,
                "settlement_content_access": True,
                "result_projection_access": False,
                "odds_price_popularity_or_market_access": False,
                "payoff_access": True,
                "roi_calculation": True,
                "network_access": False,
                "credential_access": False,
                "formal_buy": False,
                "send_order": False,
                "stake": 0,
            }
        ),
        "REPLICA_COMPARE": MappingProxyType(
            {
                "phase": "REPLICA_COMPARE",
                "allowed_read_roles": ["replica_scientific_projection"],
                "allowed_write_roles": ["replica_comparison_projection"],
                "candidate_content_access": False,
                "settlement_content_access": False,
                "result_projection_access": True,
                "odds_price_popularity_or_market_access": False,
                "payoff_access": False,
                "roi_calculation": False,
                "network_access": False,
                "credential_access": False,
                "formal_buy": False,
                "send_order": False,
                "stake": 0,
            }
        ),
        "RESULT_SEAL": MappingProxyType(
            {
                "phase": "RESULT_SEAL",
                "allowed_read_roles": [
                    "authenticated_phase_receipts",
                    "replica_comparison_projection",
                ],
                "allowed_write_roles": ["authenticated_result_seal"],
                "candidate_content_access": False,
                "settlement_content_access": False,
                "result_projection_access": True,
                "odds_price_popularity_or_market_access": False,
                "payoff_access": False,
                "roi_calculation": False,
                "network_access": False,
                "credential_access": False,
                "formal_buy": False,
                "send_order": False,
                "stake": 0,
            }
        ),
    }
)
PHASE_CAPABILITY_DIGESTS = MappingProxyType(
    {
        phase: canonical_digest(profile)
        for phase, profile in PHASE_CAPABILITY_PROJECTIONS.items()
    }
)

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
SAFE_OPAQUE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,255}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFETY_FIELDS = frozenset({"formal_buy", "send_order", "stake"})


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SharedG2ValidationError(f"{label} must be an object")
    payload = dict(value)
    canonical_digest(payload)
    return payload


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    payload = _object(value, label)
    missing = sorted(fields - set(payload))
    extra = sorted(set(payload) - fields)
    if missing or extra:
        raise SharedG2ValidationError(
            f"{label} fields must be exact; missing={missing}, extra={extra}"
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


def _sha256(value: Any, label: str, *, allow_zero: bool = False) -> str:
    text = _string(value, label)
    if not FULL_SHA256.fullmatch(text):
        raise SharedG2ValidationError(f"{label} must be a lowercase full SHA-256")
    if not allow_zero and text == ZERO_SHA256:
        raise SharedG2ValidationError(f"{label} must not be the all-zero digest")
    return text


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SharedG2ValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SharedG2ValidationError(f"{label} must be a non-negative integer")
    return value


def _safety(value: Any, label: str) -> None:
    payload = _exact(value, SAFETY_FIELDS, label)
    if payload != {"formal_buy": False, "send_order": False, "stake": 0}:
        raise SharedG2ValidationError(
            f"{label} must keep formal_buy=false, send_order=false, stake=0"
        )


def _same_global_head(left: GlobalHead, right: GlobalHead, label: str) -> None:
    if left != right:
        raise SharedG2StaleAuthority(f"{label} global head mismatch")


def _require_context_head(
    context: CutoverContext, head: GlobalHead, label: str
) -> None:
    expected = context.expectations
    if (
        head.authority_id != expected.authority_id
        or head.activation_epoch != expected.activation_epoch
        or head.backend_identity_digest != expected.backend_identity_digest
        or head.cutover_receipt_digest != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority(
            f"{label} does not bind the active shared-G2 cutover"
        )


@runtime_checkable
class RemoteLeaseMaterialTransport(Protocol):
    """Read-only access to immutable lease material stored by shared G2."""

    def fetch_phase_lease(self, lease_payload_digest: str) -> Mapping[str, Any]:
        ...

    def fetch_lease_consumption_receipt(
        self, receipt_payload_digest: str
    ) -> Mapping[str, Any]:
        ...

    def fetch_phase_output_attestation(
        self, attestation_payload_digest: str
    ) -> Mapping[str, Any]:
        ...

    def fetch_phase_output_seal_receipt(
        self, receipt_payload_digest: str
    ) -> Mapping[str, Any]:
        ...

    def fetch_decision_lease_batch(
        self, batch_payload_digest: str
    ) -> Mapping[str, Any]:
        ...

    def fetch_decision_consumption_batch(
        self, batch_payload_digest: str
    ) -> Mapping[str, Any]:
        ...


class UnconfiguredSharedG2LeaseAdapter:
    """Default runtime adapter: deliberately no local or synthetic authority."""

    _MESSAGE = (
        "shared external G2 lease authority is unconfigured; local, in-memory, "
        "SQLite, worktree, branch, and process-memory fallbacks are forbidden"
    )

    @staticmethod
    def _fail() -> None:
        raise SharedG2Unavailable(UnconfiguredSharedG2LeaseAdapter._MESSAGE)

    def fetch_phase_lease(self, lease_payload_digest: str) -> Mapping[str, Any]:
        del lease_payload_digest
        self._fail()

    def fetch_lease_consumption_receipt(
        self, receipt_payload_digest: str
    ) -> Mapping[str, Any]:
        del receipt_payload_digest
        self._fail()

    def fetch_phase_output_attestation(
        self, attestation_payload_digest: str
    ) -> Mapping[str, Any]:
        del attestation_payload_digest
        self._fail()

    def fetch_phase_output_seal_receipt(
        self, receipt_payload_digest: str
    ) -> Mapping[str, Any]:
        del receipt_payload_digest
        self._fail()

    def fetch_decision_lease_batch(
        self, batch_payload_digest: str
    ) -> Mapping[str, Any]:
        del batch_payload_digest
        self._fail()

    def fetch_decision_consumption_batch(
        self, batch_payload_digest: str
    ) -> Mapping[str, Any]:
        del batch_payload_digest
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


LEASE_BINDING_FIELDS = frozenset(
    {
        "run_scope_digest",
        "run_generation",
        "recipe_digest",
        "semantic_subject_digest",
        "exact_run_subject_digest",
        "question_family_digest",
        "replica_id",
        "phase",
        "attempt",
        "predecessor_receipt_digest",
        "issue_revalidation_receipt_digest",
        "policy_digest",
        "schema_digest",
        "verifier_digest",
        "executor_digest",
        "runner_digest",
        "bound_capability_profile_digest",
        "phase_capability_digest",
        "ttl_seconds",
    }
)


@dataclass(frozen=True)
class LeaseBinding:
    run_scope_digest: str
    run_generation: int
    recipe_digest: str
    semantic_subject_digest: str
    exact_run_subject_digest: str
    question_family_digest: str
    replica_id: str
    phase: str
    attempt: int
    predecessor_receipt_digest: str
    issue_revalidation_receipt_digest: str
    policy_digest: str
    schema_digest: str
    verifier_digest: str
    executor_digest: str
    runner_digest: str
    bound_capability_profile_digest: str
    phase_capability_digest: str
    ttl_seconds: int

    def __post_init__(self) -> None:
        phase = _identifier(self.phase, "lease binding.phase")
        if phase not in LEASED_PHASES:
            raise SharedG2ValidationError(
                f"lease binding phase is not registered: {phase!r}"
            )
        attempt = _positive_int(self.attempt, "lease binding.attempt")
        if attempt != 1:
            raise SharedG2ValidationError(
                "registered diagnostic lease attempt must be exactly 1; retries are forbidden"
            )
        ttl = _positive_int(self.ttl_seconds, "lease binding.ttl_seconds")
        if ttl > MAX_LEASE_TTL_SECONDS:
            raise SharedG2ValidationError(
                f"lease TTL exceeds {MAX_LEASE_TTL_SECONDS} seconds"
            )
        object.__setattr__(
            self,
            "run_scope_digest",
            _sha256(self.run_scope_digest, "lease binding.run_scope_digest"),
        )
        object.__setattr__(
            self,
            "run_generation",
            _nonnegative_int(self.run_generation, "lease binding.run_generation"),
        )
        object.__setattr__(
            self,
            "recipe_digest",
            _sha256(self.recipe_digest, "lease binding.recipe_digest"),
        )
        for field in (
            "semantic_subject_digest",
            "exact_run_subject_digest",
            "question_family_digest",
        ):
            object.__setattr__(
                self, field, _sha256(getattr(self, field), f"lease binding.{field}")
            )
        replica_id = _identifier(self.replica_id, "lease binding.replica_id")
        if replica_id not in ALLOWED_ACTORS_BY_PHASE[phase]:
            raise SharedG2ValidationError(
                f"lease actor {replica_id!r} is not registered for phase {phase!r}"
            )
        object.__setattr__(self, "replica_id", replica_id)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "attempt", attempt)
        for field in (
            "predecessor_receipt_digest",
            "issue_revalidation_receipt_digest",
            "policy_digest",
            "schema_digest",
            "verifier_digest",
            "executor_digest",
            "runner_digest",
            "bound_capability_profile_digest",
        ):
            object.__setattr__(
                self, field, _sha256(getattr(self, field), f"lease binding.{field}")
            )
        phase_capability_digest = _sha256(
            self.phase_capability_digest,
            "lease binding.phase_capability_digest",
        )
        expected_phase_digest = PHASE_CAPABILITY_DIGESTS[phase]
        if phase_capability_digest != expected_phase_digest:
            raise SharedG2ValidationError(
                "lease phase capability digest does not match the finite phase profile"
            )
        object.__setattr__(
            self, "phase_capability_digest", phase_capability_digest
        )
        object.__setattr__(self, "ttl_seconds", ttl)

    @classmethod
    def from_wire(cls, value: Any, *, label: str = "lease binding") -> "LeaseBinding":
        payload = _exact(value, LEASE_BINDING_FIELDS, label)
        return cls(**payload)

    def to_wire(self) -> dict[str, Any]:
        return {
            "run_scope_digest": self.run_scope_digest,
            "run_generation": self.run_generation,
            "recipe_digest": self.recipe_digest,
            "semantic_subject_digest": self.semantic_subject_digest,
            "exact_run_subject_digest": self.exact_run_subject_digest,
            "question_family_digest": self.question_family_digest,
            "replica_id": self.replica_id,
            "phase": self.phase,
            "attempt": self.attempt,
            "predecessor_receipt_digest": self.predecessor_receipt_digest,
            "issue_revalidation_receipt_digest": (
                self.issue_revalidation_receipt_digest
            ),
            "policy_digest": self.policy_digest,
            "schema_digest": self.schema_digest,
            "verifier_digest": self.verifier_digest,
            "executor_digest": self.executor_digest,
            "runner_digest": self.runner_digest,
            "bound_capability_profile_digest": (
                self.bound_capability_profile_digest
            ),
            "phase_capability_digest": self.phase_capability_digest,
            "ttl_seconds": self.ttl_seconds,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_wire())

    @property
    def domain_digest(self) -> str:
        return canonical_digest(
            {
                "domain": "keiba-ai/shared-g2/registered-diagnostic/lease/v1",
                "run_scope_digest": self.run_scope_digest,
                "recipe_digest": self.recipe_digest,
                "replica_id": self.replica_id,
                "phase": self.phase,
                "attempt": self.attempt,
            }
        )


def _lease_state_digest(
    *,
    binding: LeaseBinding,
    status: str,
    lease_payload_digest: str,
    dispatch_digest: str,
) -> str:
    if status not in {"ISSUED", "CONSUMED"}:
        raise SharedG2ValidationError("lease state status is not registered")
    lease_digest = _sha256(
        lease_payload_digest,
        "lease state.lease_payload_digest",
        allow_zero=status == "ISSUED",
    )
    dispatch = _sha256(
        dispatch_digest,
        "lease state.dispatch_digest",
        allow_zero=status == "ISSUED",
    )
    if status == "ISSUED" and (
        lease_digest != ZERO_SHA256 or dispatch != ZERO_SHA256
    ):
        raise SharedG2ValidationError(
            "ISSUED lease state must use zero output placeholders"
        )
    if status == "CONSUMED" and (
        lease_digest == ZERO_SHA256 or dispatch == ZERO_SHA256
    ):
        raise SharedG2ValidationError(
            "CONSUMED lease state must bind lease and dispatch digests"
        )
    return canonical_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "state_kind": LEASE_STATE_KIND,
            "status": status,
            "lease_domain_digest": binding.domain_digest,
            "binding_digest": binding.digest,
            "lease_payload_digest": lease_digest,
            "dispatch_digest": dispatch,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
    )


def _issue_lease_mutation(binding: LeaseBinding) -> SubjectMutation:
    return SubjectMutation(
        subject_kind="LEASE",
        subject_digest=binding.domain_digest,
        generation=0,
        expected_sequence=0,
        expected_head_digest=ZERO_SHA256,
        new_state_digest=_lease_state_digest(
            binding=binding,
            status="ISSUED",
            lease_payload_digest=ZERO_SHA256,
            dispatch_digest=ZERO_SHA256,
        ),
    )


PHASE_LEASE_FIELDS = frozenset(
    {
        "schema_version",
        "lease_kind",
        "authority_id",
        "activation_epoch",
        "backend_identity_digest",
        "cutover_receipt_digest",
        "lease_id",
        "binding",
        "binding_digest",
        "status",
        "retry_budget",
        "issued_from_global_head",
        "issued_global_head",
        "issue_transaction_id",
        "lease_authority_identity_digest",
        "issued_at",
        "expires_at",
        "safety",
    }
)


@dataclass(frozen=True)
class PhaseLease:
    lease_digest: str
    envelope_digest: str
    lease_id: str
    binding: LeaseBinding
    issued_from_global_head: GlobalHead
    issued_global_head: GlobalHead
    issue_transaction_id: str
    lease_authority_identity_digest: str
    issued_at: str
    expires_at: str


def validate_phase_lease(
    value: Any,
    *,
    expected_binding: LeaseBinding,
    issue_transaction: CommittedTransaction,
    context: CutoverContext,
    expected_lease_authority_identity_digest: str,
    verifier: AuthenticatedEnvelopeVerifier,
) -> PhaseLease:
    envelope = validate_authenticated_envelope(
        value, expected_payload_type=PHASE_LEASE_KIND, verifier=verifier
    )
    validate_authenticated_identity_binding(
        envelope,
        expected_identity_digest=expected_lease_authority_identity_digest,
        label="phase lease",
    )
    payload = _exact(envelope.payload, PHASE_LEASE_FIELDS, "phase lease")
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError("phase lease schema_version must be 1")
    if payload["lease_kind"] != PHASE_LEASE_KIND:
        raise SharedG2ValidationError("phase lease kind is invalid")
    if payload["status"] != "ISSUED" or payload["retry_budget"] != 0:
        raise SharedG2ValidationError(
            "phase lease must be immutable ISSUED with retry_budget=0"
        )
    _safety(payload["safety"], "phase lease.safety")

    expected = context.expectations
    if (
        _identifier(payload["authority_id"], "phase lease.authority_id")
        != expected.authority_id
        or _identifier(payload["activation_epoch"], "phase lease.activation_epoch")
        != expected.activation_epoch
        or _sha256(
            payload["backend_identity_digest"],
            "phase lease.backend_identity_digest",
        )
        != expected.backend_identity_digest
        or _sha256(
            payload["cutover_receipt_digest"],
            "phase lease.cutover_receipt_digest",
        )
        != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority("phase lease authority binding mismatch")

    binding = LeaseBinding.from_wire(payload["binding"])
    binding_digest = _sha256(payload["binding_digest"], "phase lease.binding_digest")
    if (
        binding != expected_binding
        or binding_digest != binding.digest
        or binding.run_scope_digest != issue_transaction.receipt.run_scope_digest
    ):
        raise SharedG2ValidationError("phase lease binding mismatch")

    receipt = issue_transaction.receipt
    expected_operation = ISSUE_OPERATION_BY_PHASE[binding.phase]
    if (
        receipt.operation_kind != expected_operation
        or receipt.operation_output_type != PHASE_LEASE_KIND
        or receipt.operation_output_digest != envelope.payload_digest
    ):
        raise SharedG2ValidationError(
            "phase lease does not match its atomic issue transaction"
        )
    issued_from = validate_global_head(
        payload["issued_from_global_head"],
        label="phase lease.issued_from_global_head",
    )
    issued_head = validate_global_head(
        payload["issued_global_head"], label="phase lease.issued_global_head"
    )
    _require_context_head(context, issued_from, "phase lease issue predecessor")
    _require_context_head(context, issued_head, "phase lease issue result")
    _same_global_head(
        issued_from, receipt.previous_global_head, "phase lease issue predecessor"
    )
    _same_global_head(issued_head, receipt.new_global_head, "phase lease issue result")

    transaction_id = _opaque_identifier(
        payload["issue_transaction_id"], "phase lease.issue_transaction_id"
    )
    if transaction_id != receipt.transaction_id:
        raise SharedG2ValidationError("phase lease transaction ID mismatch")
    authority_identity = _sha256(
        payload["lease_authority_identity_digest"],
        "phase lease.lease_authority_identity_digest",
    )
    expected_authority_identity = _sha256(
        expected_lease_authority_identity_digest,
        "expected lease authority identity digest",
    )
    if authority_identity != expected_authority_identity:
        raise SharedG2ValidationError("phase lease issuer identity mismatch")
    issued_at = _string(payload["issued_at"], "phase lease.issued_at")
    expires_at = _string(payload["expires_at"], "phase lease.expires_at")
    issued_time = parse_utc_timestamp(issued_at, "phase lease.issued_at")
    expires_time = parse_utc_timestamp(expires_at, "phase lease.expires_at")
    if issued_at != receipt.committed_at:
        raise SharedG2ValidationError(
            "phase lease issued_at must equal the atomic transaction commit time"
        )
    if expires_time != issued_time + timedelta(seconds=binding.ttl_seconds):
        raise SharedG2ValidationError(
            "phase lease expiry does not equal its hash-bound TTL"
        )
    return PhaseLease(
        lease_digest=envelope.payload_digest,
        envelope_digest=envelope.envelope_digest,
        lease_id=_opaque_identifier(payload["lease_id"], "phase lease.lease_id"),
        binding=binding,
        issued_from_global_head=issued_from,
        issued_global_head=issued_head,
        issue_transaction_id=transaction_id,
        lease_authority_identity_digest=authority_identity,
        issued_at=issued_at,
        expires_at=expires_at,
    )


def _validate_decision_binding_pair(
    values: Sequence[LeaseBinding],
) -> tuple[LeaseBinding, LeaseBinding]:
    if len(values) != 2:
        raise SharedG2ValidationError(
            "decision lease batch requires exactly two replica bindings"
        )
    bindings = tuple(sorted(values, key=lambda item: item.replica_id))
    if tuple(item.replica_id for item in bindings) != DECISION_REPLICA_IDS:
        raise SharedG2ValidationError(
            "decision lease batch replica set must be exactly clean_a and clean_b"
        )
    if any(item.phase != "DECISION_FREEZE" for item in bindings):
        raise SharedG2ValidationError(
            "decision lease batch may contain only DECISION_FREEZE bindings"
        )
    shared_fields = (
        "run_scope_digest",
        "run_generation",
        "recipe_digest",
        "semantic_subject_digest",
        "exact_run_subject_digest",
        "question_family_digest",
        "predecessor_receipt_digest",
        "issue_revalidation_receipt_digest",
        "policy_digest",
        "schema_digest",
        "verifier_digest",
        "executor_digest",
        "runner_digest",
        "bound_capability_profile_digest",
        "phase_capability_digest",
        "ttl_seconds",
    )
    for field in shared_fields:
        if getattr(bindings[0], field) != getattr(bindings[1], field):
            raise SharedG2ValidationError(
                f"decision lease batch shared binding field differs: {field}"
            )
    return bindings[0], bindings[1]


def decision_lease_batch_domain_digest(
    bindings: Sequence[LeaseBinding],
) -> str:
    normalized = _validate_decision_binding_pair(bindings)
    return canonical_digest(
        {
            "domain": "keiba-ai/shared-g2/registered-diagnostic/decision-lease-batch/v1",
            "run_scope_digest": normalized[0].run_scope_digest,
            "recipe_digest": normalized[0].recipe_digest,
            "replicas": [
                {
                    "replica_id": item.replica_id,
                    "lease_domain_digest": item.domain_digest,
                    "binding_digest": item.digest,
                }
                for item in normalized
            ],
        }
    )


DECISION_LEASE_ENTRY_FIELDS = frozenset(
    {
        "lease_id",
        "binding",
        "binding_digest",
        "status",
        "retry_budget",
        "issued_at",
        "expires_at",
    }
)
DECISION_LEASE_BATCH_FIELDS = frozenset(
    {
        "schema_version",
        "batch_kind",
        "authority_id",
        "activation_epoch",
        "backend_identity_digest",
        "cutover_receipt_digest",
        "batch_domain_digest",
        "run_scope_digest",
        "recipe_digest",
        "leases",
        "issued_from_global_head",
        "issued_global_head",
        "issue_transaction_id",
        "lease_authority_identity_digest",
        "issued_at",
        "safety",
    }
)


@dataclass(frozen=True)
class DecisionLeaseBatchReceipt:
    payload_digest: str
    envelope_digest: str
    batch_domain_digest: str
    leases: tuple[PhaseLease, PhaseLease]
    issued_from_global_head: GlobalHead
    issued_global_head: GlobalHead
    issue_transaction_id: str
    issued_at: str


def validate_decision_lease_batch_receipt(
    value: Any,
    *,
    expected_bindings: Sequence[LeaseBinding],
    issue_transaction: CommittedTransaction,
    context: CutoverContext,
    expected_lease_authority_identity_digest: str,
    verifier: AuthenticatedEnvelopeVerifier,
) -> DecisionLeaseBatchReceipt:
    envelope = validate_authenticated_envelope(
        value, expected_payload_type=DECISION_LEASE_BATCH_KIND, verifier=verifier
    )
    validate_authenticated_identity_binding(
        envelope,
        expected_identity_digest=expected_lease_authority_identity_digest,
        label="decision lease batch",
    )
    payload = _exact(
        envelope.payload,
        DECISION_LEASE_BATCH_FIELDS,
        "decision lease batch receipt",
    )
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError(
            "decision lease batch schema_version must be 1"
        )
    if payload["batch_kind"] != DECISION_LEASE_BATCH_KIND:
        raise SharedG2ValidationError("decision lease batch kind is invalid")
    _safety(payload["safety"], "decision lease batch.safety")
    expected = context.expectations
    if (
        _identifier(
            payload["authority_id"], "decision lease batch.authority_id"
        )
        != expected.authority_id
        or _identifier(
            payload["activation_epoch"],
            "decision lease batch.activation_epoch",
        )
        != expected.activation_epoch
        or _sha256(
            payload["backend_identity_digest"],
            "decision lease batch.backend_identity_digest",
        )
        != expected.backend_identity_digest
        or _sha256(
            payload["cutover_receipt_digest"],
            "decision lease batch.cutover_receipt_digest",
        )
        != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority(
            "decision lease batch authority binding mismatch"
        )
    bindings = _validate_decision_binding_pair(expected_bindings)
    batch_domain = decision_lease_batch_domain_digest(bindings)
    if _sha256(
        payload["batch_domain_digest"],
        "decision lease batch.batch_domain_digest",
    ) != batch_domain:
        raise SharedG2ValidationError("decision lease batch domain mismatch")
    if (
        _sha256(
            payload["run_scope_digest"],
            "decision lease batch.run_scope_digest",
        )
        != bindings[0].run_scope_digest
        or _sha256(
            payload["recipe_digest"], "decision lease batch.recipe_digest"
        )
        != bindings[0].recipe_digest
    ):
        raise SharedG2ValidationError(
            "decision lease batch run or recipe binding mismatch"
        )
    raw_entries = payload["leases"]
    if not isinstance(raw_entries, list) or len(raw_entries) != 2:
        raise SharedG2ValidationError(
            "decision lease batch must contain exactly two lease entries"
        )
    entries = [
        _exact(item, DECISION_LEASE_ENTRY_FIELDS, f"decision lease batch.leases[{i}]")
        for i, item in enumerate(raw_entries)
    ]
    observed_bindings = tuple(
        LeaseBinding.from_wire(item["binding"], label="decision lease entry.binding")
        for item in entries
    )
    if observed_bindings != bindings:
        raise SharedG2ValidationError(
            "decision lease batch entries are not canonical clean_a/clean_b bindings"
        )
    receipt = issue_transaction.receipt
    if (
        receipt.operation_kind != "RND_DECISION_LEASE_BATCH_ISSUE"
        or receipt.operation_output_type != DECISION_LEASE_BATCH_KIND
        or receipt.operation_output_digest != envelope.payload_digest
        or receipt.run_scope_digest != bindings[0].run_scope_digest
    ):
        raise SharedG2ValidationError(
            "decision lease batch does not bind its atomic issue transaction"
        )
    issued_from = validate_global_head(
        payload["issued_from_global_head"],
        label="decision lease batch.issued_from_global_head",
    )
    issued_head = validate_global_head(
        payload["issued_global_head"],
        label="decision lease batch.issued_global_head",
    )
    _same_global_head(
        issued_from, receipt.previous_global_head, "decision lease batch predecessor"
    )
    _same_global_head(
        issued_head, receipt.new_global_head, "decision lease batch result"
    )
    transaction_id = _opaque_identifier(
        payload["issue_transaction_id"],
        "decision lease batch.issue_transaction_id",
    )
    if transaction_id != receipt.transaction_id:
        raise SharedG2ValidationError(
            "decision lease batch transaction ID mismatch"
        )
    if _sha256(
        payload["lease_authority_identity_digest"],
        "decision lease batch.lease_authority_identity_digest",
    ) != _sha256(
        expected_lease_authority_identity_digest,
        "expected lease authority identity digest",
    ):
        raise SharedG2ValidationError(
            "decision lease batch authority identity mismatch"
        )
    batch_issued_at = _string(
        payload["issued_at"], "decision lease batch.issued_at"
    )
    if batch_issued_at != receipt.committed_at:
        raise SharedG2ValidationError(
            "decision lease batch issued_at must equal transaction commit time"
        )
    leases: list[PhaseLease] = []
    for entry, binding in zip(entries, bindings):
        if entry["status"] != "ISSUED" or entry["retry_budget"] != 0:
            raise SharedG2ValidationError(
                "decision lease entry must be ISSUED with retry_budget=0"
            )
        binding_digest = _sha256(
            entry["binding_digest"], "decision lease entry.binding_digest"
        )
        if binding_digest != binding.digest:
            raise SharedG2ValidationError(
                "decision lease entry binding digest mismatch"
            )
        issued_at = _string(entry["issued_at"], "decision lease entry.issued_at")
        expires_at = _string(
            entry["expires_at"], "decision lease entry.expires_at"
        )
        issued_time = parse_utc_timestamp(
            issued_at, "decision lease entry.issued_at"
        )
        expires_time = parse_utc_timestamp(
            expires_at, "decision lease entry.expires_at"
        )
        if issued_at != batch_issued_at or expires_time != issued_time + timedelta(
            seconds=binding.ttl_seconds
        ):
            raise SharedG2ValidationError(
                "decision lease entry time or hash-bound TTL mismatch"
            )
        lease_digest = canonical_digest(entry)
        leases.append(
            PhaseLease(
                lease_digest=lease_digest,
                envelope_digest=envelope.envelope_digest,
                lease_id=_opaque_identifier(
                    entry["lease_id"], "decision lease entry.lease_id"
                ),
                binding=binding,
                issued_from_global_head=issued_from,
                issued_global_head=issued_head,
                issue_transaction_id=transaction_id,
                lease_authority_identity_digest=(
                    expected_lease_authority_identity_digest
                ),
                issued_at=issued_at,
                expires_at=expires_at,
            )
        )
    if leases[0].lease_id == leases[1].lease_id:
        raise SharedG2ValidationError("decision lease IDs must be distinct")
    return DecisionLeaseBatchReceipt(
        payload_digest=envelope.payload_digest,
        envelope_digest=envelope.envelope_digest,
        batch_domain_digest=batch_domain,
        leases=(leases[0], leases[1]),
        issued_from_global_head=issued_from,
        issued_global_head=issued_head,
        issue_transaction_id=transaction_id,
        issued_at=batch_issued_at,
    )


LEASE_CONSUMPTION_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_kind",
        "authority_id",
        "activation_epoch",
        "backend_identity_digest",
        "cutover_receipt_digest",
        "status",
        "lease_id",
        "lease_payload_digest",
        "binding_digest",
        "operation_kind",
        "dispatch_digest",
        "consume_revalidation_receipt_digest",
        "consumed_from_global_head",
        "consumed_global_head",
        "consume_transaction_id",
        "lease_authority_identity_digest",
        "consumed_at",
        "retry_budget",
        "safety",
    }
)


@dataclass(frozen=True)
class LeaseConsumptionReceipt:
    payload_digest: str
    envelope_digest: str
    lease_id: str
    lease_payload_digest: str
    binding_digest: str
    operation_kind: str
    dispatch_digest: str
    consume_revalidation_receipt_digest: str
    consumed_from_global_head: GlobalHead
    consumed_global_head: GlobalHead
    consume_transaction_id: str
    consumed_at: str


def validate_lease_consumption_receipt(
    value: Any,
    *,
    lease: PhaseLease,
    consume_transaction: CommittedTransaction,
    context: CutoverContext,
    expected_dispatch_digest: str,
    expected_consume_revalidation_receipt_digest: str,
    expected_lease_authority_identity_digest: str,
    verifier: AuthenticatedEnvelopeVerifier,
) -> LeaseConsumptionReceipt:
    envelope = validate_authenticated_envelope(
        value,
        expected_payload_type=LEASE_CONSUMPTION_RECEIPT_KIND,
        verifier=verifier,
    )
    validate_authenticated_identity_binding(
        envelope,
        expected_identity_digest=expected_lease_authority_identity_digest,
        label="lease consumption receipt",
    )
    payload = _exact(
        envelope.payload,
        LEASE_CONSUMPTION_FIELDS,
        "lease consumption receipt",
    )
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError(
            "lease consumption receipt schema_version must be 1"
        )
    if payload["receipt_kind"] != LEASE_CONSUMPTION_RECEIPT_KIND:
        raise SharedG2ValidationError("lease consumption receipt kind is invalid")
    if payload["status"] != "CONSUMED" or payload["retry_budget"] != 0:
        raise SharedG2ValidationError(
            "lease consumption receipt must be CONSUMED with retry_budget=0"
        )
    _safety(payload["safety"], "lease consumption receipt.safety")

    expected = context.expectations
    if (
        _identifier(
            payload["authority_id"], "lease consumption receipt.authority_id"
        )
        != expected.authority_id
        or _identifier(
            payload["activation_epoch"],
            "lease consumption receipt.activation_epoch",
        )
        != expected.activation_epoch
        or _sha256(
            payload["backend_identity_digest"],
            "lease consumption receipt.backend_identity_digest",
        )
        != expected.backend_identity_digest
        or _sha256(
            payload["cutover_receipt_digest"],
            "lease consumption receipt.cutover_receipt_digest",
        )
        != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority(
            "lease consumption receipt authority binding mismatch"
        )

    receipt = consume_transaction.receipt
    expected_operation = CONSUME_OPERATION_BY_PHASE[lease.binding.phase]
    operation = _identifier(
        payload["operation_kind"], "lease consumption receipt.operation_kind"
    )
    if (
        operation != expected_operation
        or receipt.operation_kind != expected_operation
        or receipt.operation_output_type != LEASE_CONSUMPTION_RECEIPT_KIND
        or receipt.operation_output_digest != envelope.payload_digest
        or receipt.run_scope_digest != lease.binding.run_scope_digest
    ):
        raise SharedG2ValidationError(
            "lease consumption receipt does not bind its atomic transaction"
        )

    lease_id = _opaque_identifier(
        payload["lease_id"], "lease consumption receipt.lease_id"
    )
    lease_digest = _sha256(
        payload["lease_payload_digest"],
        "lease consumption receipt.lease_payload_digest",
    )
    binding_digest = _sha256(
        payload["binding_digest"], "lease consumption receipt.binding_digest"
    )
    dispatch_digest = _sha256(
        payload["dispatch_digest"], "lease consumption receipt.dispatch_digest"
    )
    revalidation_digest = _sha256(
        payload["consume_revalidation_receipt_digest"],
        "lease consumption receipt.consume_revalidation_receipt_digest",
    )
    if (
        lease_id != lease.lease_id
        or lease_digest != lease.lease_digest
        or binding_digest != lease.binding.digest
        or dispatch_digest
        != _sha256(expected_dispatch_digest, "expected dispatch digest")
        or revalidation_digest
        != _sha256(
            expected_consume_revalidation_receipt_digest,
            "expected consume revalidation receipt digest",
        )
    ):
        raise SharedG2ValidationError(
            "lease consumption receipt exact lease/domain/dispatch binding mismatch"
        )

    consumed_from = validate_global_head(
        payload["consumed_from_global_head"],
        label="lease consumption receipt.consumed_from_global_head",
    )
    consumed_head = validate_global_head(
        payload["consumed_global_head"],
        label="lease consumption receipt.consumed_global_head",
    )
    _require_context_head(context, consumed_from, "lease consumption predecessor")
    _require_context_head(context, consumed_head, "lease consumption result")
    _same_global_head(
        consumed_from,
        receipt.previous_global_head,
        "lease consumption predecessor",
    )
    _same_global_head(
        consumed_head, receipt.new_global_head, "lease consumption result"
    )
    transaction_id = _opaque_identifier(
        payload["consume_transaction_id"],
        "lease consumption receipt.consume_transaction_id",
    )
    if transaction_id != receipt.transaction_id:
        raise SharedG2ValidationError(
            "lease consumption receipt transaction ID mismatch"
        )
    authority_identity = _sha256(
        payload["lease_authority_identity_digest"],
        "lease consumption receipt.lease_authority_identity_digest",
    )
    if authority_identity != _sha256(
        expected_lease_authority_identity_digest,
        "expected lease authority identity digest",
    ):
        raise SharedG2ValidationError(
            "lease consumption authority identity mismatch"
        )
    consumed_at = _string(
        payload["consumed_at"], "lease consumption receipt.consumed_at"
    )
    consumed_time = parse_utc_timestamp(
        consumed_at, "lease consumption receipt.consumed_at"
    )
    if consumed_at != receipt.committed_at:
        raise SharedG2ValidationError(
            "lease consumed_at must equal the atomic transaction commit time"
        )
    if consumed_time < parse_utc_timestamp(lease.issued_at, "phase lease.issued_at"):
        raise SharedG2ValidationError("lease consumption predates lease issue")
    if consumed_time >= parse_utc_timestamp(lease.expires_at, "phase lease.expires_at"):
        raise SharedG2ValidationError("expired phase lease cannot be consumed")

    return LeaseConsumptionReceipt(
        payload_digest=envelope.payload_digest,
        envelope_digest=envelope.envelope_digest,
        lease_id=lease_id,
        lease_payload_digest=lease_digest,
        binding_digest=binding_digest,
        operation_kind=operation,
        dispatch_digest=dispatch_digest,
        consume_revalidation_receipt_digest=revalidation_digest,
        consumed_from_global_head=consumed_from,
        consumed_global_head=consumed_head,
        consume_transaction_id=transaction_id,
        consumed_at=consumed_at,
    )


DECISION_CONSUMPTION_ENTRY_FIELDS = frozenset(
    {
        "replica_id",
        "lease_id",
        "lease_payload_digest",
        "binding_digest",
        "dispatch_digest",
    }
)
DECISION_CONSUMPTION_BATCH_FIELDS = frozenset(
    {
        "schema_version",
        "batch_kind",
        "authority_id",
        "activation_epoch",
        "backend_identity_digest",
        "cutover_receipt_digest",
        "batch_domain_digest",
        "run_scope_digest",
        "recipe_digest",
        "status",
        "retry_budget",
        "leases",
        "consume_revalidation_receipt_digest",
        "consumed_from_global_head",
        "consumed_global_head",
        "consume_transaction_id",
        "lease_authority_identity_digest",
        "consumed_at",
        "safety",
    }
)


@dataclass(frozen=True)
class DecisionConsumptionBatchReceipt:
    payload_digest: str
    envelope_digest: str
    batch_domain_digest: str
    consumptions: tuple[LeaseConsumptionReceipt, LeaseConsumptionReceipt]
    consume_revalidation_receipt_digest: str
    consumed_from_global_head: GlobalHead
    consumed_global_head: GlobalHead
    consume_transaction_id: str
    consumed_at: str


def validate_decision_consumption_batch_receipt(
    value: Any,
    *,
    issued: "IssuedDecisionLeaseBatch",
    consume_transaction: CommittedTransaction,
    context: CutoverContext,
    expected_dispatch_digests: Mapping[str, str],
    expected_consume_revalidation_receipt_digest: str,
    expected_lease_authority_identity_digest: str,
    verifier: AuthenticatedEnvelopeVerifier,
) -> DecisionConsumptionBatchReceipt:
    envelope = validate_authenticated_envelope(
        value,
        expected_payload_type=DECISION_CONSUMPTION_BATCH_KIND,
        verifier=verifier,
    )
    validate_authenticated_identity_binding(
        envelope,
        expected_identity_digest=expected_lease_authority_identity_digest,
        label="decision consumption batch",
    )
    payload = _exact(
        envelope.payload,
        DECISION_CONSUMPTION_BATCH_FIELDS,
        "decision consumption batch receipt",
    )
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError(
            "decision consumption batch schema_version must be 1"
        )
    if payload["batch_kind"] != DECISION_CONSUMPTION_BATCH_KIND:
        raise SharedG2ValidationError(
            "decision consumption batch kind is invalid"
        )
    if payload["status"] != "CONSUMED" or payload["retry_budget"] != 0:
        raise SharedG2ValidationError(
            "decision consumption batch must be CONSUMED with retry_budget=0"
        )
    _safety(payload["safety"], "decision consumption batch.safety")
    expected = context.expectations
    if (
        _identifier(
            payload["authority_id"],
            "decision consumption batch.authority_id",
        )
        != expected.authority_id
        or _identifier(
            payload["activation_epoch"],
            "decision consumption batch.activation_epoch",
        )
        != expected.activation_epoch
        or _sha256(
            payload["backend_identity_digest"],
            "decision consumption batch.backend_identity_digest",
        )
        != expected.backend_identity_digest
        or _sha256(
            payload["cutover_receipt_digest"],
            "decision consumption batch.cutover_receipt_digest",
        )
        != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority(
            "decision consumption batch authority binding mismatch"
        )
    batch = issued.receipt
    bindings = tuple(item.binding for item in batch.leases)
    batch_domain = decision_lease_batch_domain_digest(bindings)
    if (
        _sha256(
            payload["batch_domain_digest"],
            "decision consumption batch.batch_domain_digest",
        )
        != batch_domain
        or _sha256(
            payload["run_scope_digest"],
            "decision consumption batch.run_scope_digest",
        )
        != bindings[0].run_scope_digest
        or _sha256(
            payload["recipe_digest"],
            "decision consumption batch.recipe_digest",
        )
        != bindings[0].recipe_digest
    ):
        raise SharedG2ValidationError(
            "decision consumption batch run/recipe/domain mismatch"
        )
    expected_dispatch = {
        _identifier(key, "decision dispatch replica_id"): _sha256(
            item, f"decision dispatch digest.{key}"
        )
        for key, item in expected_dispatch_digests.items()
    }
    if tuple(sorted(expected_dispatch)) != DECISION_REPLICA_IDS:
        raise SharedG2ValidationError(
            "decision batch dispatch set must be exactly clean_a and clean_b"
        )
    raw_entries = payload["leases"]
    if not isinstance(raw_entries, list) or len(raw_entries) != 2:
        raise SharedG2ValidationError(
            "decision consumption batch requires exactly two lease entries"
        )
    entries = [
        _exact(
            item,
            DECISION_CONSUMPTION_ENTRY_FIELDS,
            f"decision consumption batch.leases[{index}]",
        )
        for index, item in enumerate(raw_entries)
    ]
    observed_replicas = tuple(
        _identifier(
            item["replica_id"], "decision consumption entry.replica_id"
        )
        for item in entries
    )
    if observed_replicas != DECISION_REPLICA_IDS:
        raise SharedG2ValidationError(
            "decision consumption entries must be ordered clean_a then clean_b"
        )
    receipt = consume_transaction.receipt
    if (
        receipt.operation_kind != "RND_DECISION_IRREVERSIBLE_START"
        or receipt.operation_output_type != DECISION_CONSUMPTION_BATCH_KIND
        or receipt.operation_output_digest != envelope.payload_digest
        or receipt.run_scope_digest != bindings[0].run_scope_digest
    ):
        raise SharedG2ValidationError(
            "decision consumption batch does not bind the irreversible transaction"
        )
    consumed_from = validate_global_head(
        payload["consumed_from_global_head"],
        label="decision consumption batch.consumed_from_global_head",
    )
    consumed_head = validate_global_head(
        payload["consumed_global_head"],
        label="decision consumption batch.consumed_global_head",
    )
    _same_global_head(
        consumed_from,
        receipt.previous_global_head,
        "decision consumption batch predecessor",
    )
    _same_global_head(
        consumed_head, receipt.new_global_head, "decision consumption batch result"
    )
    transaction_id = _opaque_identifier(
        payload["consume_transaction_id"],
        "decision consumption batch.consume_transaction_id",
    )
    if transaction_id != receipt.transaction_id:
        raise SharedG2ValidationError(
            "decision consumption batch transaction ID mismatch"
        )
    authority_identity = _sha256(
        payload["lease_authority_identity_digest"],
        "decision consumption batch.lease_authority_identity_digest",
    )
    if authority_identity != _sha256(
        expected_lease_authority_identity_digest,
        "expected lease authority identity digest",
    ):
        raise SharedG2ValidationError(
            "decision consumption batch authority identity mismatch"
        )
    revalidation_digest = _sha256(
        payload["consume_revalidation_receipt_digest"],
        "decision consumption batch.consume_revalidation_receipt_digest",
    )
    if revalidation_digest != _sha256(
        expected_consume_revalidation_receipt_digest,
        "expected decision consume revalidation receipt digest",
    ):
        raise SharedG2ValidationError(
            "decision consumption batch revalidation receipt mismatch"
        )
    consumed_at = _string(
        payload["consumed_at"], "decision consumption batch.consumed_at"
    )
    consumed_time = parse_utc_timestamp(
        consumed_at, "decision consumption batch.consumed_at"
    )
    if consumed_at != receipt.committed_at:
        raise SharedG2ValidationError(
            "decision consumption batch time must equal transaction commit time"
        )
    consumptions: list[LeaseConsumptionReceipt] = []
    for entry, phase_lease in zip(entries, batch.leases):
        dispatch = _sha256(
            entry["dispatch_digest"], "decision consumption entry.dispatch_digest"
        )
        if (
            _opaque_identifier(
                entry["lease_id"], "decision consumption entry.lease_id"
            )
            != phase_lease.lease_id
            or _sha256(
                entry["lease_payload_digest"],
                "decision consumption entry.lease_payload_digest",
            )
            != phase_lease.lease_digest
            or _sha256(
                entry["binding_digest"],
                "decision consumption entry.binding_digest",
            )
            != phase_lease.binding.digest
            or dispatch != expected_dispatch[phase_lease.binding.replica_id]
        ):
            raise SharedG2ValidationError(
                "decision consumption entry exact lease/dispatch mismatch"
            )
        if consumed_time >= parse_utc_timestamp(
            phase_lease.expires_at, "decision phase lease.expires_at"
        ):
            raise SharedG2ValidationError(
                "expired decision lease cannot participate in irreversible batch"
            )
        consumptions.append(
            LeaseConsumptionReceipt(
                payload_digest=envelope.payload_digest,
                envelope_digest=envelope.envelope_digest,
                lease_id=phase_lease.lease_id,
                lease_payload_digest=phase_lease.lease_digest,
                binding_digest=phase_lease.binding.digest,
                operation_kind="RND_DECISION_IRREVERSIBLE_START",
                dispatch_digest=dispatch,
                consume_revalidation_receipt_digest=revalidation_digest,
                consumed_from_global_head=consumed_from,
                consumed_global_head=consumed_head,
                consume_transaction_id=transaction_id,
                consumed_at=consumed_at,
            )
        )
    return DecisionConsumptionBatchReceipt(
        payload_digest=envelope.payload_digest,
        envelope_digest=envelope.envelope_digest,
        batch_domain_digest=batch_domain,
        consumptions=(consumptions[0], consumptions[1]),
        consume_revalidation_receipt_digest=revalidation_digest,
        consumed_from_global_head=consumed_from,
        consumed_global_head=consumed_head,
        consume_transaction_id=transaction_id,
        consumed_at=consumed_at,
    )


PHASE_OUTPUT_ATTESTATION_FIELDS = frozenset(
    {
        "schema_version",
        "attestation_kind",
        "authority_id",
        "activation_epoch",
        "cutover_receipt_digest",
        "run_scope_digest",
        "recipe_digest",
        "replica_id",
        "phase",
        "attempt",
        "binding_digest",
        "lease_payload_digest",
        "lease_consumption_receipt_digest",
        "consumed_global_head",
        "output_digest",
        "policy_digest",
        "schema_digest",
        "verifier_digest",
        "executor_digest",
        "runner_digest",
        "bound_capability_profile_digest",
        "phase_capability_digest",
        "attester_identity_digest",
        "attested_at",
        "safety",
    }
)


@dataclass(frozen=True)
class TrustedPhaseOutputAttestation:
    payload_digest: str
    envelope_digest: str
    binding_digest: str
    lease_payload_digest: str
    lease_consumption_receipt_digest: str
    consumed_global_head: GlobalHead
    output_digest: str
    attester_identity_digest: str
    attested_at: str


def validate_trusted_phase_output_attestation(
    value: Any,
    *,
    consumed: "ConsumedPhaseLease",
    context: CutoverContext,
    expected_attester_identity_digest: str,
    verifier: AuthenticatedEnvelopeVerifier,
) -> TrustedPhaseOutputAttestation:
    envelope = validate_authenticated_envelope(
        value,
        expected_payload_type=PHASE_OUTPUT_ATTESTATION_KIND,
        verifier=verifier,
    )
    validate_authenticated_identity_binding(
        envelope,
        expected_identity_digest=expected_attester_identity_digest,
        label="trusted phase-output attestation",
    )
    payload = _exact(
        envelope.payload,
        PHASE_OUTPUT_ATTESTATION_FIELDS,
        "trusted phase-output attestation",
    )
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError(
            "phase-output attestation schema_version must be 1"
        )
    if payload["attestation_kind"] != PHASE_OUTPUT_ATTESTATION_KIND:
        raise SharedG2ValidationError("phase-output attestation kind is invalid")
    _safety(payload["safety"], "trusted phase-output attestation.safety")
    expected = context.expectations
    if (
        _identifier(
            payload["authority_id"],
            "trusted phase-output attestation.authority_id",
        )
        != expected.authority_id
        or _identifier(
            payload["activation_epoch"],
            "trusted phase-output attestation.activation_epoch",
        )
        != expected.activation_epoch
        or _sha256(
            payload["cutover_receipt_digest"],
            "trusted phase-output attestation.cutover_receipt_digest",
        )
        != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority(
            "trusted phase-output attestation cutover binding mismatch"
        )
    binding = consumed.lease.binding
    scalar_checks = {
        "run_scope_digest": binding.run_scope_digest,
        "recipe_digest": binding.recipe_digest,
        "replica_id": binding.replica_id,
        "phase": binding.phase,
        "attempt": binding.attempt,
        "binding_digest": binding.digest,
        "lease_payload_digest": consumed.lease.lease_digest,
        "lease_consumption_receipt_digest": consumed.receipt.payload_digest,
        "policy_digest": binding.policy_digest,
        "schema_digest": binding.schema_digest,
        "verifier_digest": binding.verifier_digest,
        "executor_digest": binding.executor_digest,
        "runner_digest": binding.runner_digest,
        "bound_capability_profile_digest": (
            binding.bound_capability_profile_digest
        ),
        "phase_capability_digest": binding.phase_capability_digest,
    }
    observed_checks = {
        "run_scope_digest": _sha256(
            payload["run_scope_digest"],
            "trusted phase-output attestation.run_scope_digest",
        ),
        "recipe_digest": _sha256(
            payload["recipe_digest"],
            "trusted phase-output attestation.recipe_digest",
        ),
        "replica_id": _identifier(
            payload["replica_id"],
            "trusted phase-output attestation.replica_id",
        ),
        "phase": _identifier(
            payload["phase"], "trusted phase-output attestation.phase"
        ),
        "attempt": _positive_int(
            payload["attempt"], "trusted phase-output attestation.attempt"
        ),
        "binding_digest": _sha256(
            payload["binding_digest"],
            "trusted phase-output attestation.binding_digest",
        ),
        "lease_payload_digest": _sha256(
            payload["lease_payload_digest"],
            "trusted phase-output attestation.lease_payload_digest",
        ),
        "lease_consumption_receipt_digest": _sha256(
            payload["lease_consumption_receipt_digest"],
            "trusted phase-output attestation.lease_consumption_receipt_digest",
        ),
        "policy_digest": _sha256(
            payload["policy_digest"],
            "trusted phase-output attestation.policy_digest",
        ),
        "schema_digest": _sha256(
            payload["schema_digest"],
            "trusted phase-output attestation.schema_digest",
        ),
        "verifier_digest": _sha256(
            payload["verifier_digest"],
            "trusted phase-output attestation.verifier_digest",
        ),
        "executor_digest": _sha256(
            payload["executor_digest"],
            "trusted phase-output attestation.executor_digest",
        ),
        "runner_digest": _sha256(
            payload["runner_digest"],
            "trusted phase-output attestation.runner_digest",
        ),
        "bound_capability_profile_digest": _sha256(
            payload["bound_capability_profile_digest"],
            "trusted phase-output attestation.bound_capability_profile_digest",
        ),
        "phase_capability_digest": _sha256(
            payload["phase_capability_digest"],
            "trusted phase-output attestation.phase_capability_digest",
        ),
    }
    if observed_checks != scalar_checks:
        raise SharedG2ValidationError(
            "trusted phase-output attestation exact execution binding mismatch"
        )
    consumed_head = validate_global_head(
        payload["consumed_global_head"],
        label="trusted phase-output attestation.consumed_global_head",
    )
    _same_global_head(
        consumed_head,
        consumed.transaction.receipt.new_global_head,
        "trusted phase-output attestation consumed",
    )
    output_digest = _sha256(
        payload["output_digest"],
        "trusted phase-output attestation.output_digest",
    )
    attester_identity = _sha256(
        payload["attester_identity_digest"],
        "trusted phase-output attestation.attester_identity_digest",
    )
    if attester_identity != _sha256(
        expected_attester_identity_digest,
        "expected phase-output attester identity digest",
    ):
        raise SharedG2ValidationError(
            "trusted phase-output attester identity mismatch"
        )
    attested_at = _string(
        payload["attested_at"], "trusted phase-output attestation.attested_at"
    )
    if parse_utc_timestamp(
        attested_at, "trusted phase-output attestation.attested_at"
    ) < parse_utc_timestamp(
        consumed.receipt.consumed_at, "lease consumption receipt.consumed_at"
    ):
        raise SharedG2ValidationError(
            "phase-output attestation predates lease consumption"
        )
    return TrustedPhaseOutputAttestation(
        payload_digest=envelope.payload_digest,
        envelope_digest=envelope.envelope_digest,
        binding_digest=binding.digest,
        lease_payload_digest=consumed.lease.lease_digest,
        lease_consumption_receipt_digest=consumed.receipt.payload_digest,
        consumed_global_head=consumed_head,
        output_digest=output_digest,
        attester_identity_digest=attester_identity,
        attested_at=attested_at,
    )


def _phase_output_subject_digest(binding: LeaseBinding) -> str:
    return canonical_digest(
        {
            "domain": "keiba-ai/shared-g2/registered-diagnostic/phase-output/v1",
            "lease_domain_digest": binding.domain_digest,
            "binding_digest": binding.digest,
        }
    )


def _phase_output_state_digest(
    *,
    consumed: "ConsumedPhaseLease",
    attestation: TrustedPhaseOutputAttestation,
) -> str:
    return canonical_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "state_kind": PHASE_OUTPUT_STATE_KIND,
            "status": "SEALED",
            "phase_output_subject_digest": _phase_output_subject_digest(
                consumed.lease.binding
            ),
            "binding_digest": consumed.lease.binding.digest,
            "lease_payload_digest": consumed.lease.lease_digest,
            "lease_consumption_receipt_digest": consumed.receipt.payload_digest,
            "output_attestation_digest": attestation.payload_digest,
            "output_digest": attestation.output_digest,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
    )


PHASE_OUTPUT_SEAL_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_kind",
        "authority_id",
        "activation_epoch",
        "backend_identity_digest",
        "cutover_receipt_digest",
        "run_scope_digest",
        "recipe_digest",
        "replica_id",
        "phase",
        "attempt",
        "binding_digest",
        "phase_output_subject_digest",
        "lease_payload_digest",
        "lease_consumption_receipt_digest",
        "output_attestation_digest",
        "output_digest",
        "operation_kind",
        "sealed_from_global_head",
        "sealed_global_head",
        "seal_transaction_id",
        "lease_authority_identity_digest",
        "sealed_at",
        "safety",
    }
)


@dataclass(frozen=True)
class PhaseOutputSealReceipt:
    payload_digest: str
    envelope_digest: str
    run_scope_digest: str
    recipe_digest: str
    replica_id: str
    phase: str
    attempt: int
    binding_digest: str
    phase_output_subject_digest: str
    lease_payload_digest: str
    lease_consumption_receipt_digest: str
    output_attestation_digest: str
    output_digest: str
    operation_kind: str
    sealed_from_global_head: GlobalHead
    sealed_global_head: GlobalHead
    seal_transaction_id: str
    sealed_at: str


def validate_phase_output_seal_receipt(
    value: Any,
    *,
    consumed: "ConsumedPhaseLease",
    attestation: TrustedPhaseOutputAttestation,
    seal_transaction: CommittedTransaction,
    context: CutoverContext,
    expected_lease_authority_identity_digest: str,
    verifier: AuthenticatedEnvelopeVerifier,
) -> PhaseOutputSealReceipt:
    envelope = validate_authenticated_envelope(
        value,
        expected_payload_type=PHASE_OUTPUT_SEAL_RECEIPT_KIND,
        verifier=verifier,
    )
    validate_authenticated_identity_binding(
        envelope,
        expected_identity_digest=expected_lease_authority_identity_digest,
        label="phase-output seal receipt",
    )
    payload = _exact(
        envelope.payload, PHASE_OUTPUT_SEAL_FIELDS, "phase-output seal receipt"
    )
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError(
            "phase-output seal receipt schema_version must be 1"
        )
    if payload["receipt_kind"] != PHASE_OUTPUT_SEAL_RECEIPT_KIND:
        raise SharedG2ValidationError("phase-output seal receipt kind is invalid")
    _safety(payload["safety"], "phase-output seal receipt.safety")
    expected = context.expectations
    if (
        _identifier(payload["authority_id"], "phase-output seal.authority_id")
        != expected.authority_id
        or _identifier(
            payload["activation_epoch"], "phase-output seal.activation_epoch"
        )
        != expected.activation_epoch
        or _sha256(
            payload["backend_identity_digest"],
            "phase-output seal.backend_identity_digest",
        )
        != expected.backend_identity_digest
        or _sha256(
            payload["cutover_receipt_digest"],
            "phase-output seal.cutover_receipt_digest",
        )
        != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority(
            "phase-output seal receipt authority binding mismatch"
        )
    binding = consumed.lease.binding
    observed = {
        "run_scope_digest": _sha256(
            payload["run_scope_digest"], "phase-output seal.run_scope_digest"
        ),
        "recipe_digest": _sha256(
            payload["recipe_digest"], "phase-output seal.recipe_digest"
        ),
        "replica_id": _identifier(
            payload["replica_id"], "phase-output seal.replica_id"
        ),
        "phase": _identifier(payload["phase"], "phase-output seal.phase"),
        "attempt": _positive_int(
            payload["attempt"], "phase-output seal.attempt"
        ),
        "binding_digest": _sha256(
            payload["binding_digest"], "phase-output seal.binding_digest"
        ),
        "phase_output_subject_digest": _sha256(
            payload["phase_output_subject_digest"],
            "phase-output seal.phase_output_subject_digest",
        ),
        "lease_payload_digest": _sha256(
            payload["lease_payload_digest"],
            "phase-output seal.lease_payload_digest",
        ),
        "lease_consumption_receipt_digest": _sha256(
            payload["lease_consumption_receipt_digest"],
            "phase-output seal.lease_consumption_receipt_digest",
        ),
        "output_attestation_digest": _sha256(
            payload["output_attestation_digest"],
            "phase-output seal.output_attestation_digest",
        ),
        "output_digest": _sha256(
            payload["output_digest"], "phase-output seal.output_digest"
        ),
    }
    required = {
        "run_scope_digest": binding.run_scope_digest,
        "recipe_digest": binding.recipe_digest,
        "replica_id": binding.replica_id,
        "phase": binding.phase,
        "attempt": binding.attempt,
        "binding_digest": binding.digest,
        "phase_output_subject_digest": _phase_output_subject_digest(binding),
        "lease_payload_digest": consumed.lease.lease_digest,
        "lease_consumption_receipt_digest": consumed.receipt.payload_digest,
        "output_attestation_digest": attestation.payload_digest,
        "output_digest": attestation.output_digest,
    }
    if observed != required:
        raise SharedG2ValidationError(
            "phase-output seal receipt exact output binding mismatch"
        )
    transaction_receipt = seal_transaction.receipt
    operation = _identifier(
        payload["operation_kind"], "phase-output seal.operation_kind"
    )
    if (
        operation != OUTPUT_SEAL_OPERATION_BY_PHASE[binding.phase]
        or transaction_receipt.operation_kind != operation
        or transaction_receipt.operation_output_type
        != PHASE_OUTPUT_SEAL_RECEIPT_KIND
        or transaction_receipt.operation_output_digest != envelope.payload_digest
        or transaction_receipt.run_scope_digest != binding.run_scope_digest
    ):
        raise SharedG2ValidationError(
            "phase-output seal receipt does not bind its atomic transaction"
        )
    sealed_from = validate_global_head(
        payload["sealed_from_global_head"],
        label="phase-output seal.sealed_from_global_head",
    )
    sealed_head = validate_global_head(
        payload["sealed_global_head"],
        label="phase-output seal.sealed_global_head",
    )
    _same_global_head(
        sealed_from,
        transaction_receipt.previous_global_head,
        "phase-output seal predecessor",
    )
    _same_global_head(
        sealed_head, transaction_receipt.new_global_head, "phase-output seal result"
    )
    transaction_id = _opaque_identifier(
        payload["seal_transaction_id"], "phase-output seal.seal_transaction_id"
    )
    if transaction_id != transaction_receipt.transaction_id:
        raise SharedG2ValidationError("phase-output seal transaction ID mismatch")
    if _sha256(
        payload["lease_authority_identity_digest"],
        "phase-output seal.lease_authority_identity_digest",
    ) != _sha256(
        expected_lease_authority_identity_digest,
        "expected lease authority identity digest",
    ):
        raise SharedG2ValidationError("phase-output seal authority identity mismatch")
    sealed_at = _string(payload["sealed_at"], "phase-output seal.sealed_at")
    if sealed_at != transaction_receipt.committed_at:
        raise SharedG2ValidationError(
            "phase-output sealed_at must equal the atomic transaction commit time"
        )
    if parse_utc_timestamp(
        sealed_at, "phase-output seal.sealed_at"
    ) < parse_utc_timestamp(
        attestation.attested_at, "trusted phase-output attestation.attested_at"
    ):
        raise SharedG2ValidationError(
            "phase-output seal predates its trusted output attestation"
        )
    return PhaseOutputSealReceipt(
        payload_digest=envelope.payload_digest,
        envelope_digest=envelope.envelope_digest,
        run_scope_digest=binding.run_scope_digest,
        recipe_digest=binding.recipe_digest,
        replica_id=binding.replica_id,
        phase=binding.phase,
        attempt=binding.attempt,
        binding_digest=binding.digest,
        phase_output_subject_digest=_phase_output_subject_digest(binding),
        lease_payload_digest=consumed.lease.lease_digest,
        lease_consumption_receipt_digest=consumed.receipt.payload_digest,
        output_attestation_digest=attestation.payload_digest,
        output_digest=attestation.output_digest,
        operation_kind=operation,
        sealed_from_global_head=sealed_from,
        sealed_global_head=sealed_head,
        seal_transaction_id=transaction_id,
        sealed_at=sealed_at,
    )


def validate_authenticated_phase_output_seal_projection(
    value: Any,
    *,
    context: CutoverContext,
    expected_lease_authority_identity_digest: str,
    verifier: AuthenticatedEnvelopeVerifier,
) -> PhaseOutputSealReceipt:
    """Validate the self-contained G2-signed seal without workload artifacts."""
    envelope = validate_authenticated_envelope(
        value,
        expected_payload_type=PHASE_OUTPUT_SEAL_RECEIPT_KIND,
        verifier=verifier,
    )
    validate_authenticated_identity_binding(
        envelope,
        expected_identity_digest=expected_lease_authority_identity_digest,
        label="phase-output seal receipt projection",
    )
    payload = _exact(
        envelope.payload, PHASE_OUTPUT_SEAL_FIELDS, "phase-output seal receipt"
    )
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise SharedG2ValidationError(
            "phase-output seal receipt schema_version must be 1"
        )
    if payload["receipt_kind"] != PHASE_OUTPUT_SEAL_RECEIPT_KIND:
        raise SharedG2ValidationError("phase-output seal receipt kind is invalid")
    _safety(payload["safety"], "phase-output seal receipt.safety")
    expected = context.expectations
    if (
        _identifier(payload["authority_id"], "phase-output seal.authority_id")
        != expected.authority_id
        or _identifier(
            payload["activation_epoch"], "phase-output seal.activation_epoch"
        )
        != expected.activation_epoch
        or _sha256(
            payload["backend_identity_digest"],
            "phase-output seal.backend_identity_digest",
        )
        != expected.backend_identity_digest
        or _sha256(
            payload["cutover_receipt_digest"],
            "phase-output seal.cutover_receipt_digest",
        )
        != context.cutover_receipt_digest
    ):
        raise SharedG2StaleAuthority(
            "phase-output seal receipt authority binding mismatch"
        )
    run_scope_digest = _sha256(
        payload["run_scope_digest"], "phase-output seal.run_scope_digest"
    )
    recipe_digest = _sha256(
        payload["recipe_digest"], "phase-output seal.recipe_digest"
    )
    replica_id = _identifier(
        payload["replica_id"], "phase-output seal.replica_id"
    )
    phase = _identifier(payload["phase"], "phase-output seal.phase")
    attempt = _positive_int(payload["attempt"], "phase-output seal.attempt")
    if (
        phase not in LEASED_PHASES
        or replica_id not in ALLOWED_ACTORS_BY_PHASE[phase]
        or attempt != 1
    ):
        raise SharedG2ValidationError(
            "phase-output seal phase/actor/attempt domain is invalid"
        )
    binding_digest = _sha256(
        payload["binding_digest"], "phase-output seal.binding_digest"
    )
    phase_output_subject_digest = _sha256(
        payload["phase_output_subject_digest"],
        "phase-output seal.phase_output_subject_digest",
    )
    lease_payload_digest = _sha256(
        payload["lease_payload_digest"],
        "phase-output seal.lease_payload_digest",
    )
    consumption_digest = _sha256(
        payload["lease_consumption_receipt_digest"],
        "phase-output seal.lease_consumption_receipt_digest",
    )
    attestation_digest = _sha256(
        payload["output_attestation_digest"],
        "phase-output seal.output_attestation_digest",
    )
    output_digest = _sha256(
        payload["output_digest"], "phase-output seal.output_digest"
    )
    operation = _identifier(
        payload["operation_kind"], "phase-output seal.operation_kind"
    )
    if operation != OUTPUT_SEAL_OPERATION_BY_PHASE[phase]:
        raise SharedG2ValidationError(
            "phase-output seal operation does not match its phase"
        )
    sealed_from = validate_global_head(
        payload["sealed_from_global_head"],
        label="phase-output seal.sealed_from_global_head",
    )
    sealed_head = validate_global_head(
        payload["sealed_global_head"],
        label="phase-output seal.sealed_global_head",
    )
    _require_context_head(context, sealed_from, "phase-output seal predecessor")
    _require_context_head(context, sealed_head, "phase-output seal result")
    if (
        sealed_head.sequence != sealed_from.sequence + 1
        or sealed_head.head_digest == sealed_from.head_digest
    ):
        raise SharedG2ValidationError(
            "phase-output seal global head transition is invalid"
        )
    transaction_id = _opaque_identifier(
        payload["seal_transaction_id"], "phase-output seal.seal_transaction_id"
    )
    if _sha256(
        payload["lease_authority_identity_digest"],
        "phase-output seal.lease_authority_identity_digest",
    ) != _sha256(
        expected_lease_authority_identity_digest,
        "expected lease authority identity digest",
    ):
        raise SharedG2ValidationError("phase-output seal authority identity mismatch")
    sealed_at = _string(payload["sealed_at"], "phase-output seal.sealed_at")
    sealed_time = parse_utc_timestamp(sealed_at, "phase-output seal.sealed_at")
    if sealed_time < parse_utc_timestamp(
        sealed_from.observed_at, "phase-output seal predecessor observed_at"
    ):
        raise SharedG2ValidationError(
            "phase-output seal time predates its predecessor global head"
        )
    return PhaseOutputSealReceipt(
        payload_digest=envelope.payload_digest,
        envelope_digest=envelope.envelope_digest,
        run_scope_digest=run_scope_digest,
        recipe_digest=recipe_digest,
        replica_id=replica_id,
        phase=phase,
        attempt=attempt,
        binding_digest=binding_digest,
        phase_output_subject_digest=phase_output_subject_digest,
        lease_payload_digest=lease_payload_digest,
        lease_consumption_receipt_digest=consumption_digest,
        output_attestation_digest=attestation_digest,
        output_digest=output_digest,
        operation_kind=operation,
        sealed_from_global_head=sealed_from,
        sealed_global_head=sealed_head,
        seal_transaction_id=transaction_id,
        sealed_at=sealed_at,
    )


def phase_output_seal_state_digest(receipt: PhaseOutputSealReceipt) -> str:
    return canonical_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "state_kind": PHASE_OUTPUT_STATE_KIND,
            "status": "SEALED",
            "phase_output_subject_digest": receipt.phase_output_subject_digest,
            "binding_digest": receipt.binding_digest,
            "lease_payload_digest": receipt.lease_payload_digest,
            "lease_consumption_receipt_digest": (
                receipt.lease_consumption_receipt_digest
            ),
            "output_attestation_digest": receipt.output_attestation_digest,
            "output_digest": receipt.output_digest,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
    )


@dataclass(frozen=True)
class RevalidatedPhaseOutputSeal:
    receipt: PhaseOutputSealReceipt
    subject_head: SubjectHead
    subject_snapshot: SubjectHeadSnapshot
    authority_snapshot: AuthoritySnapshot


@dataclass(frozen=True)
class IssuedPhaseLease:
    lease: PhaseLease
    transaction: CommittedTransaction
    resulting_run_state: RunLifecycleState | None


@dataclass(frozen=True)
class ConsumedPhaseLease:
    lease: PhaseLease
    receipt: LeaseConsumptionReceipt
    issue_transaction: CommittedTransaction
    transaction: CommittedTransaction
    resulting_run_state: RunLifecycleState | None
    decision_issued_batch: IssuedDecisionLeaseBatch | None = None
    decision_consumption_batch: DecisionConsumptionBatchReceipt | None = None


@dataclass(frozen=True)
class IssuedDecisionLeaseBatch:
    receipt: DecisionLeaseBatchReceipt
    transaction: CommittedTransaction
    resulting_run_state: RunLifecycleState


@dataclass(frozen=True)
class ConsumedDecisionLeaseBatch:
    issued: IssuedDecisionLeaseBatch
    receipt: DecisionConsumptionBatchReceipt
    transaction: CommittedTransaction
    resulting_run_state: RunLifecycleState
    replica_consumptions: tuple[ConsumedPhaseLease, ConsumedPhaseLease]


@dataclass(frozen=True)
class SealedPhaseOutput:
    consumed: ConsumedPhaseLease
    attestation: TrustedPhaseOutputAttestation
    receipt: PhaseOutputSealReceipt
    transaction: CommittedTransaction
    resulting_run_state: RunLifecycleState | None


@dataclass(frozen=True)
class RunLifecycleView:
    head: SubjectHead
    state: RunLifecycleState


@dataclass(frozen=True)
class IrreversibleLifecycleView:
    run: RunLifecycleView
    semantic_head: SubjectHead
    semantic_state: SingleUseSubjectState
    exact_head: SubjectHead
    exact_state: SingleUseSubjectState
    question_family_head: SubjectHead
    question_family_state: QuestionFamilyState


def canonical_predecessor_output_digest(
    *,
    successor_binding: LeaseBinding,
    predecessor_output_seals: Sequence[SealedPhaseOutput],
) -> str:
    seals = tuple(
        sorted(predecessor_output_seals, key=lambda item: item.receipt.replica_id)
    )
    if successor_binding.phase == "SETTLEMENT_DIAGNOSTIC":
        if len(seals) != 1:
            raise SharedG2ValidationError(
                "settlement lease requires exactly one decision output seal"
            )
        receipt = seals[0].receipt
        if (
            receipt.phase != "DECISION_FREEZE"
            or receipt.replica_id != successor_binding.replica_id
        ):
            raise SharedG2ValidationError(
                "settlement predecessor must be the same-replica decision output seal"
            )
        digest = receipt.payload_digest
    elif successor_binding.phase == "REPLICA_COMPARE":
        if successor_binding.replica_id != REPLICA_COMPARE_ACTOR or len(seals) != 2:
            raise SharedG2ValidationError(
                "replica compare requires lane_coordinator and exactly two settlement seals"
            )
        receipts = tuple(item.receipt for item in seals)
        if (
            tuple(item.replica_id for item in receipts) != DECISION_REPLICA_IDS
            or any(item.phase != "SETTLEMENT_DIAGNOSTIC" for item in receipts)
            or receipts[0].payload_digest == receipts[1].payload_digest
        ):
            raise SharedG2ValidationError(
                "replica compare predecessors must be distinct clean_a/clean_b settlement seals"
            )
        digest = canonical_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "projection_kind": (
                    "REGISTERED_DIAGNOSTIC_ORDERED_SETTLEMENT_SEAL_PAIR_V1"
                ),
                "run_scope_digest": successor_binding.run_scope_digest,
                "recipe_digest": successor_binding.recipe_digest,
                "receipts": [
                    {
                        "replica_id": item.replica_id,
                        "receipt_payload_digest": item.payload_digest,
                        "binding_digest": item.binding_digest,
                        "output_digest": item.output_digest,
                    }
                    for item in receipts
                ],
                "safety": {"formal_buy": False, "send_order": False, "stake": 0},
            }
        )
    elif successor_binding.phase == "RESULT_SEAL":
        if successor_binding.replica_id != RESULT_SEAL_ACTOR or len(seals) != 1:
            raise SharedG2ValidationError(
                "result seal requires canonical_sealer and one compare seal"
            )
        receipt = seals[0].receipt
        if (
            receipt.phase != "REPLICA_COMPARE"
            or receipt.replica_id != REPLICA_COMPARE_ACTOR
        ):
            raise SharedG2ValidationError(
                "result seal predecessor must be lane_coordinator replica comparison"
            )
        digest = receipt.payload_digest
    else:
        raise SharedG2ValidationError(
            "decision batch predecessor is its approval receipt, not a phase output"
        )
    for sealed in seals:
        receipt = sealed.receipt
        if (
            receipt.run_scope_digest != successor_binding.run_scope_digest
            or receipt.recipe_digest != successor_binding.recipe_digest
            or receipt.attempt != 1
        ):
            raise SharedG2ValidationError(
                "phase predecessor run/recipe/attempt binding mismatch"
            )
    return digest


def _subject_head_by_kind(
    heads: Sequence[SubjectHead], subject_kind: str
) -> SubjectHead:
    matches = [item for item in heads if item.subject_kind == subject_kind]
    if len(matches) != 1:
        raise SharedG2ValidationError(
            f"exactly one {subject_kind} subject mutation is required"
        )
    return matches[0]


class SharedG2LeaseAuthorityClient:
    """Typed one-shot lease client; all accepted material is remote-authenticated."""

    def __init__(
        self,
        *,
        authority: SharedG2AuthorityClient,
        lease_material_transport: RemoteLeaseMaterialTransport,
        lease_envelope_verifier: AuthenticatedEnvelopeVerifier,
        lease_authority_identity_digest: str,
        phase_output_attester_identity_digest: str,
    ) -> None:
        if isinstance(lease_material_transport, UnconfiguredSharedG2LeaseAdapter):
            raise SharedG2Unavailable(UnconfiguredSharedG2LeaseAdapter._MESSAGE)
        if isinstance(lease_envelope_verifier, UnconfiguredSharedG2LeaseAdapter):
            raise SharedG2Unavailable(UnconfiguredSharedG2LeaseAdapter._MESSAGE)
        if isinstance(authority, UnconfiguredSharedG2Adapter):
            raise SharedG2Unavailable(
                "shared-G2 ledger authority is unconfigured; lease issuance is disabled"
            )
        self._authority = authority
        self._materials = lease_material_transport
        self._verifier = lease_envelope_verifier
        self._lease_authority_identity_digest = _sha256(
            lease_authority_identity_digest,
            "lease authority identity digest",
        )
        self._phase_output_attester_identity_digest = _sha256(
            phase_output_attester_identity_digest,
            "phase-output attester identity digest",
        )

    @property
    def context(self) -> CutoverContext:
        return self._authority.context

    def _fetch_phase_lease_envelope(self, digest: str) -> Mapping[str, Any]:
        expected_digest = _sha256(digest, "phase lease payload digest")
        try:
            raw = self._materials.fetch_phase_lease(expected_digest)
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(
                f"remote phase lease material unavailable: {exc}"
            ) from exc
        authenticated = validate_authenticated_envelope(
            raw, expected_payload_type=PHASE_LEASE_KIND, verifier=self._verifier
        )
        if authenticated.payload_digest != expected_digest:
            raise SharedG2ValidationError("remote phase lease digest mismatch")
        return raw

    def _fetch_consumption_envelope(self, digest: str) -> Mapping[str, Any]:
        expected_digest = _sha256(
            digest, "lease consumption receipt payload digest"
        )
        try:
            raw = self._materials.fetch_lease_consumption_receipt(expected_digest)
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(
                f"remote lease consumption receipt unavailable: {exc}"
            ) from exc
        authenticated = validate_authenticated_envelope(
            raw,
            expected_payload_type=LEASE_CONSUMPTION_RECEIPT_KIND,
            verifier=self._verifier,
        )
        if authenticated.payload_digest != expected_digest:
            raise SharedG2ValidationError(
                "remote lease consumption receipt digest mismatch"
            )
        return raw

    def _fetch_output_attestation_envelope(
        self, digest: str
    ) -> Mapping[str, Any]:
        expected_digest = _sha256(
            digest, "phase-output attestation payload digest"
        )
        try:
            raw = self._materials.fetch_phase_output_attestation(expected_digest)
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(
                f"remote phase-output attestation unavailable: {exc}"
            ) from exc
        authenticated = validate_authenticated_envelope(
            raw,
            expected_payload_type=PHASE_OUTPUT_ATTESTATION_KIND,
            verifier=self._verifier,
        )
        if authenticated.payload_digest != expected_digest:
            raise SharedG2ValidationError(
                "remote phase-output attestation digest mismatch"
            )
        return raw

    def _fetch_output_seal_envelope(self, digest: str) -> Mapping[str, Any]:
        expected_digest = _sha256(
            digest, "phase-output seal receipt payload digest"
        )
        try:
            raw = self._materials.fetch_phase_output_seal_receipt(expected_digest)
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(
                f"remote phase-output seal receipt unavailable: {exc}"
            ) from exc
        authenticated = validate_authenticated_envelope(
            raw,
            expected_payload_type=PHASE_OUTPUT_SEAL_RECEIPT_KIND,
            verifier=self._verifier,
        )
        if authenticated.payload_digest != expected_digest:
            raise SharedG2ValidationError(
                "remote phase-output seal receipt digest mismatch"
            )
        return raw

    def _fetch_decision_lease_batch_envelope(
        self, digest: str
    ) -> Mapping[str, Any]:
        expected_digest = _sha256(digest, "decision lease batch payload digest")
        try:
            raw = self._materials.fetch_decision_lease_batch(expected_digest)
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(
                f"remote decision lease batch unavailable: {exc}"
            ) from exc
        authenticated = validate_authenticated_envelope(
            raw,
            expected_payload_type=DECISION_LEASE_BATCH_KIND,
            verifier=self._verifier,
        )
        if authenticated.payload_digest != expected_digest:
            raise SharedG2ValidationError(
                "remote decision lease batch digest mismatch"
            )
        return raw

    def _fetch_decision_consumption_batch_envelope(
        self, digest: str
    ) -> Mapping[str, Any]:
        expected_digest = _sha256(
            digest, "decision consumption batch payload digest"
        )
        try:
            raw = self._materials.fetch_decision_consumption_batch(expected_digest)
        except SharedG2Error:
            raise
        except Exception as exc:
            raise SharedG2Unavailable(
                f"remote decision consumption batch unavailable: {exc}"
            ) from exc
        authenticated = validate_authenticated_envelope(
            raw,
            expected_payload_type=DECISION_CONSUMPTION_BATCH_KIND,
            verifier=self._verifier,
        )
        if authenticated.payload_digest != expected_digest:
            raise SharedG2ValidationError(
                "remote decision consumption batch digest mismatch"
            )
        return raw

    def fetch_and_revalidate_unrevoked_phase_output_seal(
        self, receipt_payload_digest: str
    ) -> RevalidatedPhaseOutputSeal:
        """Read-only exact replay path for result sealing and audit retrieval."""
        raw = self._fetch_output_seal_envelope(receipt_payload_digest)
        receipt = validate_authenticated_phase_output_seal_projection(
            raw,
            context=self.context,
            expected_lease_authority_identity_digest=(
                self._lease_authority_identity_digest
            ),
            verifier=self._verifier,
        )
        subject_snapshot, authority_snapshot = (
            self._authority.read_subject_head_snapshot(
                subject_kind="PHASE_OUTPUT",
                subject_digest=receipt.phase_output_subject_digest,
                generation=0,
            )
        )
        subject_head = subject_snapshot.subject_head
        if (
            subject_head.sequence != 1
            or subject_head.state_digest != phase_output_seal_state_digest(receipt)
            or subject_snapshot.global_head.sequence
            < receipt.sealed_global_head.sequence
        ):
            raise SharedG2StaleAuthority(
                "phase-output seal is absent, superseded, revoked, or not current"
            )
        return RevalidatedPhaseOutputSeal(
            receipt=receipt,
            subject_head=subject_head,
            subject_snapshot=subject_snapshot,
            authority_snapshot=authority_snapshot,
        )

    def issue_decision_lease_batch(
        self,
        *,
        bindings: Sequence[LeaseBinding],
        expected_global_head: GlobalHead,
        operation_id: str,
        idempotency_key: str,
        requested_at: str,
        run_lifecycle: RunLifecycleView,
    ) -> IssuedDecisionLeaseBatch:
        clean_a, clean_b = _validate_decision_binding_pair(bindings)
        normalized_bindings = (clean_a, clean_b)
        if (
            run_lifecycle.state.run_scope_digest != clean_a.run_scope_digest
            or run_lifecycle.state.generation != clean_a.run_generation
            or run_lifecycle.state.lifecycle_state != "RND_APPROVED"
        ):
            raise SharedG2ValidationError(
                "decision lease batch requires the matching canonical RND_APPROVED state"
            )
        batch_domain = decision_lease_batch_domain_digest(normalized_bindings)
        transition_evidence = canonical_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "transition": "RND_APPROVED_TO_RND_LEASED",
                "batch_domain_digest": batch_domain,
                "bindings": [item.to_wire() for item in normalized_bindings],
                "predecessor_receipt_digest": clean_a.predecessor_receipt_digest,
                "issue_revalidation_receipt_digest": (
                    clean_a.issue_revalidation_receipt_digest
                ),
                "safety": {"formal_buy": False, "send_order": False, "stake": 0},
            }
        )
        run_transition = build_run_lifecycle_transition(
            current_head=run_lifecycle.head,
            current_state=run_lifecycle.state,
            new_state="RND_LEASED",
            transition_evidence_digest=transition_evidence,
        )
        lease_mutations = [_issue_lease_mutation(item) for item in normalized_bindings]
        mutations = [*lease_mutations, run_transition.mutation]
        mutations.sort(key=lambda item: item.key())
        mutation_payload = {
            "schema_version": SCHEMA_VERSION,
            "action": "ISSUE_DECISION_LEASE_BATCH",
            "operation_kind": "RND_DECISION_LEASE_BATCH_ISSUE",
            "batch_domain_digest": batch_domain,
            "bindings": [item.to_wire() for item in normalized_bindings],
            "binding_digests": [item.digest for item in normalized_bindings],
            "lease_domain_digests": [
                item.domain_digest for item in normalized_bindings
            ],
            "expected_global_head": expected_global_head.to_wire(),
            "transition_evidence_digest": transition_evidence,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
        request = TransactionRequest.build(
            operation_kind="RND_DECISION_LEASE_BATCH_ISSUE",
            operation_id=operation_id,
            context=self.context,
            idempotency_key=idempotency_key,
            run_scope_digest=clean_a.run_scope_digest,
            mutation_payload=mutation_payload,
            expected_output_type=DECISION_LEASE_BATCH_KIND,
            expected_global_head=expected_global_head,
            subject_mutations=mutations,
            requested_at=requested_at,
        )
        committed = self._authority.commit(request)
        raw_batch = self._fetch_decision_lease_batch_envelope(
            committed.receipt.operation_output_digest
        )
        receipt = validate_decision_lease_batch_receipt(
            raw_batch,
            expected_bindings=normalized_bindings,
            issue_transaction=committed,
            context=self.context,
            expected_lease_authority_identity_digest=(
                self._lease_authority_identity_digest
            ),
            verifier=self._verifier,
        )
        return IssuedDecisionLeaseBatch(
            receipt=receipt,
            transaction=committed,
            resulting_run_state=run_transition.current,
        )

    def consume_decision_lease_batch(
        self,
        *,
        issued: IssuedDecisionLeaseBatch,
        expected_global_head: GlobalHead,
        dispatch_digests: Mapping[str, str],
        consume_revalidation_receipt_digest: str,
        operation_id: str,
        idempotency_key: str,
        requested_at: str,
        irreversible_lifecycle: IrreversibleLifecycleView,
    ) -> ConsumedDecisionLeaseBatch:
        raw_issue = self._fetch_decision_lease_batch_envelope(
            issued.receipt.payload_digest
        )
        revalidated_issue = validate_decision_lease_batch_receipt(
            raw_issue,
            expected_bindings=tuple(item.binding for item in issued.receipt.leases),
            issue_transaction=issued.transaction,
            context=self.context,
            expected_lease_authority_identity_digest=(
                self._lease_authority_identity_digest
            ),
            verifier=self._verifier,
        )
        if revalidated_issue != issued.receipt:
            raise SharedG2ValidationError(
                "decision lease batch changed before irreversible consume"
            )
        normalized_dispatch = {
            _identifier(key, "decision dispatch replica_id"): _sha256(
                value, f"decision dispatch digest.{key}"
            )
            for key, value in dispatch_digests.items()
        }
        if tuple(sorted(normalized_dispatch)) != DECISION_REPLICA_IDS:
            raise SharedG2ValidationError(
                "decision batch dispatch set must be exactly clean_a and clean_b"
            )
        consume_revalidation = _sha256(
            consume_revalidation_receipt_digest,
            "decision consume revalidation receipt digest",
        )
        requested_time = parse_utc_timestamp(
            requested_at, "decision batch consume requested_at"
        )
        if any(
            requested_time
            >= parse_utc_timestamp(item.expires_at, "decision phase lease.expires_at")
            for item in issued.receipt.leases
        ):
            raise SharedG2ValidationError(
                "decision lease batch consume request is at or after an immutable expiry"
            )
        lease_heads = {
            head.subject_digest: head
            for head in issued.transaction.receipt.new_subject_heads
            if head.subject_kind == "LEASE"
        }
        if set(lease_heads) != {
            item.binding.domain_digest for item in issued.receipt.leases
        }:
            raise SharedG2ValidationError(
                "decision issue receipt does not contain both exact LEASE subject heads"
            )
        lease_mutations: list[SubjectMutation] = []
        for item in issued.receipt.leases:
            head = lease_heads[item.binding.domain_digest]
            lease_mutations.append(
                SubjectMutation(
                    subject_kind="LEASE",
                    subject_digest=item.binding.domain_digest,
                    generation=0,
                    expected_sequence=head.sequence,
                    expected_head_digest=head.head_digest,
                    new_state_digest=_lease_state_digest(
                        binding=item.binding,
                        status="CONSUMED",
                        lease_payload_digest=item.lease_digest,
                        dispatch_digest=normalized_dispatch[item.binding.replica_id],
                    ),
                )
            )
        binding = issued.receipt.leases[0].binding
        view = irreversible_lifecycle
        if (
            view.run.state.run_scope_digest != binding.run_scope_digest
            or view.run.state.generation != binding.run_generation
            or view.run.state.lifecycle_state != "RND_LEASED"
            or view.semantic_state.subject_kind != "SEMANTIC_SUBJECT"
            or view.semantic_state.subject_digest != binding.semantic_subject_digest
            or view.semantic_state.generation != binding.run_generation
            or view.semantic_state.run_scope_digest != binding.run_scope_digest
            or view.exact_state.subject_kind != "EXACT_SUBJECT"
            or view.exact_state.subject_digest != binding.exact_run_subject_digest
            or view.exact_state.generation != binding.run_generation
            or view.exact_state.run_scope_digest != binding.run_scope_digest
            or view.question_family_head.subject_kind != "QUESTION_FAMILY"
            or view.question_family_head.subject_digest
            != binding.question_family_digest
            # QUESTION_FAMILY generation zero is the cross-run aggregate CAS.
            # A fresh per-run generation would permit a second irreversible
            # execution after an abort/re-reservation cycle.
            or view.question_family_head.generation
            != QUESTION_FAMILY_AGGREGATE_GENERATION
            or view.question_family_state.question_family_digest
            != binding.question_family_digest
        ):
            raise SharedG2ValidationError(
                "irreversible lifecycle bundle does not match decision batch binding"
            )
        transition_evidence = canonical_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "transition": "RND_LEASED_TO_RND_RUNNING_IRREVERSIBLE_BATCH",
                "decision_lease_batch_receipt_digest": issued.receipt.payload_digest,
                "batch_domain_digest": issued.receipt.batch_domain_digest,
                "leases": [
                    {
                        "replica_id": item.binding.replica_id,
                        "lease_payload_digest": item.lease_digest,
                        "dispatch_digest": normalized_dispatch[item.binding.replica_id],
                    }
                    for item in issued.receipt.leases
                ],
                "consume_revalidation_receipt_digest": consume_revalidation,
                "semantic_subject_digest": binding.semantic_subject_digest,
                "exact_run_subject_digest": binding.exact_run_subject_digest,
                "question_family_digest": binding.question_family_digest,
                "safety": {"formal_buy": False, "send_order": False, "stake": 0},
            }
        )
        run_transition = build_run_lifecycle_transition(
            current_head=view.run.head,
            current_state=view.run.state,
            new_state="RND_RUNNING",
            transition_evidence_digest=transition_evidence,
        )
        semantic_transition = build_single_use_subject_transition(
            current_head=view.semantic_head,
            current_state=view.semantic_state,
            new_state="IRREVERSIBLY_CONSUMED",
            transition_evidence_digest=transition_evidence,
        )
        exact_transition = build_single_use_subject_transition(
            current_head=view.exact_head,
            current_state=view.exact_state,
            new_state="IRREVERSIBLY_CONSUMED",
            transition_evidence_digest=transition_evidence,
        )
        family_transition = build_question_family_irreversible_increment(
            current_head=view.question_family_head,
            current_state=view.question_family_state,
            transition_evidence_digest=transition_evidence,
        )
        mutations = [
            *lease_mutations,
            run_transition.mutation,
            semantic_transition.mutation,
            exact_transition.mutation,
            family_transition.mutation,
        ]
        mutations.sort(key=lambda item: item.key())
        mutation_payload = {
            "schema_version": SCHEMA_VERSION,
            "action": "CONSUME_DECISION_LEASE_BATCH_IRREVERSIBLY",
            "operation_kind": "RND_DECISION_IRREVERSIBLE_START",
            "decision_lease_batch_receipt_digest": issued.receipt.payload_digest,
            "batch_domain_digest": issued.receipt.batch_domain_digest,
            "bindings": [item.binding.to_wire() for item in issued.receipt.leases],
            "leases": [
                {
                    "replica_id": item.binding.replica_id,
                    "lease_id": item.lease_id,
                    "lease_payload_digest": item.lease_digest,
                    "dispatch_digest": normalized_dispatch[item.binding.replica_id],
                }
                for item in issued.receipt.leases
            ],
            "consume_revalidation_receipt_digest": consume_revalidation,
            "expected_global_head": expected_global_head.to_wire(),
            "transition_evidence_digest": transition_evidence,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
        request = TransactionRequest.build(
            operation_kind="RND_DECISION_IRREVERSIBLE_START",
            operation_id=operation_id,
            context=self.context,
            idempotency_key=idempotency_key,
            run_scope_digest=binding.run_scope_digest,
            mutation_payload=mutation_payload,
            expected_output_type=DECISION_CONSUMPTION_BATCH_KIND,
            expected_global_head=expected_global_head,
            subject_mutations=mutations,
            requested_at=requested_at,
        )
        committed = self._authority.commit(request)
        raw_consumption = self._fetch_decision_consumption_batch_envelope(
            committed.receipt.operation_output_digest
        )
        receipt = validate_decision_consumption_batch_receipt(
            raw_consumption,
            issued=issued,
            consume_transaction=committed,
            context=self.context,
            expected_dispatch_digests=normalized_dispatch,
            expected_consume_revalidation_receipt_digest=consume_revalidation,
            expected_lease_authority_identity_digest=(
                self._lease_authority_identity_digest
            ),
            verifier=self._verifier,
        )
        phase_consumptions = tuple(
            ConsumedPhaseLease(
                lease=phase_lease,
                receipt=lease_receipt,
                issue_transaction=issued.transaction,
                transaction=committed,
                resulting_run_state=run_transition.current,
                decision_issued_batch=issued,
                decision_consumption_batch=receipt,
            )
            for phase_lease, lease_receipt in zip(
                issued.receipt.leases, receipt.consumptions
            )
        )
        return ConsumedDecisionLeaseBatch(
            issued=issued,
            receipt=receipt,
            transaction=committed,
            resulting_run_state=run_transition.current,
            replica_consumptions=(phase_consumptions[0], phase_consumptions[1]),
        )

    def issue_phase_lease(
        self,
        *,
        binding: LeaseBinding,
        expected_global_head: GlobalHead,
        operation_id: str,
        idempotency_key: str,
        requested_at: str,
        predecessor_output_seals: Sequence[SealedPhaseOutput],
    ) -> IssuedPhaseLease:
        if binding.phase == "DECISION_FREEZE":
            raise SharedG2ValidationError(
                "decision leases must be issued together by issue_decision_lease_batch()"
        )
        operation_kind = ISSUE_OPERATION_BY_PHASE[binding.phase]
        predecessor_digest = canonical_predecessor_output_digest(
            successor_binding=binding,
            predecessor_output_seals=predecessor_output_seals,
        )
        if predecessor_digest != binding.predecessor_receipt_digest:
            raise SharedG2ValidationError(
                "successor binding does not bind the canonical predecessor output projection"
            )
        for predecessor_output_seal in predecessor_output_seals:
            predecessor_receipt = predecessor_output_seal.receipt
            current_predecessor = (
                self.fetch_and_revalidate_unrevoked_phase_output_seal(
                    predecessor_receipt.payload_digest
                )
            )
            if current_predecessor.receipt != predecessor_receipt:
                raise SharedG2ValidationError(
                    "predecessor phase-output seal is no longer the current authenticated subject head"
                )
            _same_global_head(
                current_predecessor.authority_snapshot.global_head,
                expected_global_head,
                "successor lease predecessor proof",
            )
            raw_predecessor_attestation = self._fetch_output_attestation_envelope(
                predecessor_output_seal.attestation.payload_digest
            )
            revalidated_attestation = validate_trusted_phase_output_attestation(
                raw_predecessor_attestation,
                consumed=predecessor_output_seal.consumed,
                context=self.context,
                expected_attester_identity_digest=(
                    self._phase_output_attester_identity_digest
                ),
                verifier=self._verifier,
            )
            raw_predecessor_seal = self._fetch_output_seal_envelope(
                predecessor_receipt.payload_digest
            )
            revalidated_seal = validate_phase_output_seal_receipt(
                raw_predecessor_seal,
                consumed=predecessor_output_seal.consumed,
                attestation=revalidated_attestation,
                seal_transaction=predecessor_output_seal.transaction,
                context=self.context,
                expected_lease_authority_identity_digest=(
                    self._lease_authority_identity_digest
                ),
                verifier=self._verifier,
            )
            if (
                revalidated_attestation != predecessor_output_seal.attestation
                or revalidated_seal != predecessor_receipt
            ):
                raise SharedG2ValidationError(
                    "predecessor phase-output evidence changed during remote revalidation"
                )
        lease_mutation = _issue_lease_mutation(binding)
        mutations: list[SubjectMutation] = [lease_mutation]
        mutations.sort(key=lambda item: item.key())
        mutation_payload = {
            "schema_version": SCHEMA_VERSION,
            "action": "ISSUE_ONE_SHOT_PHASE_LEASE",
            "operation_kind": operation_kind,
            "binding": binding.to_wire(),
            "binding_digest": binding.digest,
            "lease_domain_digest": binding.domain_digest,
            "expected_global_head": expected_global_head.to_wire(),
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
        request = TransactionRequest.build(
            operation_kind=operation_kind,
            operation_id=operation_id,
            context=self.context,
            idempotency_key=idempotency_key,
            run_scope_digest=binding.run_scope_digest,
            mutation_payload=mutation_payload,
            expected_output_type=PHASE_LEASE_KIND,
            expected_global_head=expected_global_head,
            subject_mutations=mutations,
            requested_at=requested_at,
        )
        committed = self._authority.commit(request)
        raw = self._fetch_phase_lease_envelope(
            committed.receipt.operation_output_digest
        )
        lease = validate_phase_lease(
            raw,
            expected_binding=binding,
            issue_transaction=committed,
            context=self.context,
            expected_lease_authority_identity_digest=(
                self._lease_authority_identity_digest
            ),
            verifier=self._verifier,
        )
        return IssuedPhaseLease(
            lease=lease,
            transaction=committed,
            resulting_run_state=None,
        )

    def consume_phase_lease(
        self,
        *,
        issued: IssuedPhaseLease,
        expected_global_head: GlobalHead,
        dispatch_digest: str,
        consume_revalidation_receipt_digest: str,
        operation_id: str,
        idempotency_key: str,
        requested_at: str,
    ) -> ConsumedPhaseLease:
        if issued.lease.binding.phase == "DECISION_FREEZE":
            raise SharedG2ValidationError(
                "decision leases must be consumed together by consume_decision_lease_batch()"
            )
        raw_lease = self._fetch_phase_lease_envelope(issued.lease.lease_digest)
        lease = validate_phase_lease(
            raw_lease,
            expected_binding=issued.lease.binding,
            issue_transaction=issued.transaction,
            context=self.context,
            expected_lease_authority_identity_digest=(
                self._lease_authority_identity_digest
            ),
            verifier=self._verifier,
        )
        if lease != issued.lease:
            raise SharedG2ValidationError(
                "phase lease changed between issue and pre-consume read-back"
            )
        dispatch = _sha256(dispatch_digest, "phase dispatch digest")
        consume_revalidation = _sha256(
            consume_revalidation_receipt_digest,
            "consume revalidation receipt digest",
        )
        requested_time = parse_utc_timestamp(requested_at, "lease consume requested_at")
        if requested_time >= parse_utc_timestamp(lease.expires_at, "phase lease.expires_at"):
            raise SharedG2ValidationError(
                "lease consume request is at or after the immutable expiry"
            )

        lease_issue_head = _subject_head_by_kind(
            issued.transaction.receipt.new_subject_heads, "LEASE"
        )
        lease_mutation = SubjectMutation(
            subject_kind="LEASE",
            subject_digest=lease.binding.domain_digest,
            generation=0,
            expected_sequence=lease_issue_head.sequence,
            expected_head_digest=lease_issue_head.head_digest,
            new_state_digest=_lease_state_digest(
                binding=lease.binding,
                status="CONSUMED",
                lease_payload_digest=lease.lease_digest,
                dispatch_digest=dispatch,
            ),
        )
        operation_kind = CONSUME_OPERATION_BY_PHASE[lease.binding.phase]
        mutations = [lease_mutation]
        mutations.sort(key=lambda item: item.key())
        mutation_payload = {
            "schema_version": SCHEMA_VERSION,
            "action": "CONSUME_ONE_SHOT_PHASE_LEASE",
            "operation_kind": operation_kind,
            "binding": lease.binding.to_wire(),
            "binding_digest": lease.binding.digest,
            "lease_id": lease.lease_id,
            "lease_payload_digest": lease.lease_digest,
            "dispatch_digest": dispatch,
            "consume_revalidation_receipt_digest": consume_revalidation,
            "expected_global_head": expected_global_head.to_wire(),
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
        request = TransactionRequest.build(
            operation_kind=operation_kind,
            operation_id=operation_id,
            context=self.context,
            idempotency_key=idempotency_key,
            run_scope_digest=lease.binding.run_scope_digest,
            mutation_payload=mutation_payload,
            expected_output_type=LEASE_CONSUMPTION_RECEIPT_KIND,
            expected_global_head=expected_global_head,
            subject_mutations=mutations,
            requested_at=requested_at,
        )
        committed = self._authority.commit(request)
        raw_receipt = self._fetch_consumption_envelope(
            committed.receipt.operation_output_digest
        )
        receipt = validate_lease_consumption_receipt(
            raw_receipt,
            lease=lease,
            consume_transaction=committed,
            context=self.context,
            expected_dispatch_digest=dispatch,
            expected_consume_revalidation_receipt_digest=consume_revalidation,
            expected_lease_authority_identity_digest=(
                self._lease_authority_identity_digest
            ),
            verifier=self._verifier,
        )
        return ConsumedPhaseLease(
            lease=lease,
            receipt=receipt,
            issue_transaction=issued.transaction,
            transaction=committed,
            resulting_run_state=None,
        )

    def seal_phase_output(
        self,
        *,
        consumed: ConsumedPhaseLease,
        output_attestation_payload_digest: str,
        expected_global_head: GlobalHead,
        operation_id: str,
        idempotency_key: str,
        requested_at: str,
        run_lifecycle: RunLifecycleView | None = None,
    ) -> SealedPhaseOutput:
        lease = consumed.lease
        if lease.binding.phase == "DECISION_FREEZE":
            issued_batch = consumed.decision_issued_batch
            consumption_batch = consumed.decision_consumption_batch
            if issued_batch is None or consumption_batch is None:
                raise SharedG2ValidationError(
                    "decision phase output requires the authenticated two-lease batch chain"
                )
            raw_issue_batch = self._fetch_decision_lease_batch_envelope(
                issued_batch.receipt.payload_digest
            )
            revalidated_issue = validate_decision_lease_batch_receipt(
                raw_issue_batch,
                expected_bindings=tuple(
                    item.binding for item in issued_batch.receipt.leases
                ),
                issue_transaction=issued_batch.transaction,
                context=self.context,
                expected_lease_authority_identity_digest=(
                    self._lease_authority_identity_digest
                ),
                verifier=self._verifier,
            )
            raw_consumption_batch = (
                self._fetch_decision_consumption_batch_envelope(
                    consumption_batch.payload_digest
                )
            )
            dispatches = {
                phase_lease.binding.replica_id: lease_receipt.dispatch_digest
                for phase_lease, lease_receipt in zip(
                    issued_batch.receipt.leases,
                    consumption_batch.consumptions,
                )
            }
            revalidated_batch = validate_decision_consumption_batch_receipt(
                raw_consumption_batch,
                issued=issued_batch,
                consume_transaction=consumed.transaction,
                context=self.context,
                expected_dispatch_digests=dispatches,
                expected_consume_revalidation_receipt_digest=(
                    consumption_batch.consume_revalidation_receipt_digest
                ),
                expected_lease_authority_identity_digest=(
                    self._lease_authority_identity_digest
                ),
                verifier=self._verifier,
            )
            matching = [
                item
                for item in revalidated_batch.consumptions
                if item.lease_payload_digest == lease.lease_digest
            ]
            if (
                revalidated_issue != issued_batch.receipt
                or revalidated_batch != consumption_batch
                or len(matching) != 1
                or matching[0] != consumed.receipt
            ):
                raise SharedG2ValidationError(
                    "decision batch evidence changed before phase-output seal"
                )
        else:
            raw_lease = self._fetch_phase_lease_envelope(lease.lease_digest)
            revalidated_lease = validate_phase_lease(
                raw_lease,
                expected_binding=lease.binding,
                issue_transaction=consumed.issue_transaction,
                context=self.context,
                expected_lease_authority_identity_digest=(
                    self._lease_authority_identity_digest
                ),
                verifier=self._verifier,
            )
            if revalidated_lease != lease:
                raise SharedG2ValidationError(
                    "phase lease changed before phase-output seal"
                )
            raw_consumption = self._fetch_consumption_envelope(
                consumed.receipt.payload_digest
            )
            revalidated_consumption = validate_lease_consumption_receipt(
                raw_consumption,
                lease=lease,
                consume_transaction=consumed.transaction,
                context=self.context,
                expected_dispatch_digest=consumed.receipt.dispatch_digest,
                expected_consume_revalidation_receipt_digest=(
                    consumed.receipt.consume_revalidation_receipt_digest
                ),
                expected_lease_authority_identity_digest=(
                    self._lease_authority_identity_digest
                ),
                verifier=self._verifier,
            )
            if revalidated_consumption != consumed.receipt:
                raise SharedG2ValidationError(
                    "lease consumption receipt changed before phase-output seal"
                )
        raw_attestation = self._fetch_output_attestation_envelope(
            output_attestation_payload_digest
        )
        attestation = validate_trusted_phase_output_attestation(
            raw_attestation,
            consumed=consumed,
            context=self.context,
            expected_attester_identity_digest=(
                self._phase_output_attester_identity_digest
            ),
            verifier=self._verifier,
        )
        requested_time = parse_utc_timestamp(
            requested_at, "phase-output seal requested_at"
        )
        if requested_time < parse_utc_timestamp(
            attestation.attested_at, "trusted phase-output attestation.attested_at"
        ):
            raise SharedG2ValidationError(
                "phase-output seal request predates the trusted attestation"
            )
        output_mutation = SubjectMutation(
            subject_kind="PHASE_OUTPUT",
            subject_digest=_phase_output_subject_digest(consumed.lease.binding),
            generation=0,
            expected_sequence=0,
            expected_head_digest=ZERO_SHA256,
            new_state_digest=_phase_output_state_digest(
                consumed=consumed, attestation=attestation
            ),
        )
        operation_kind = OUTPUT_SEAL_OPERATION_BY_PHASE[
            consumed.lease.binding.phase
        ]
        mutations: list[SubjectMutation] = [output_mutation]
        resulting_run_state: RunLifecycleState | None = None
        transition_evidence = canonical_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "action": "SEAL_TRUSTED_PHASE_OUTPUT",
                "operation_kind": operation_kind,
                "binding_digest": consumed.lease.binding.digest,
                "lease_consumption_receipt_digest": consumed.receipt.payload_digest,
                "output_attestation_digest": attestation.payload_digest,
                "output_digest": attestation.output_digest,
                "safety": {"formal_buy": False, "send_order": False, "stake": 0},
            }
        )
        if consumed.lease.binding.phase == "RESULT_SEAL":
            if run_lifecycle is None:
                raise SharedG2ValidationError(
                    "RESULT_SEAL output requires the typed RUNNING-to-RESULT_SEALED transition"
                )
            if (
                run_lifecycle.state.run_scope_digest
                != consumed.lease.binding.run_scope_digest
                or run_lifecycle.state.generation
                != consumed.lease.binding.run_generation
                or run_lifecycle.state.lifecycle_state != "RND_RUNNING"
            ):
                raise SharedG2ValidationError(
                    "RESULT_SEAL RUN lifecycle does not match the frozen binding"
                )
            run_transition = build_run_lifecycle_transition(
                current_head=run_lifecycle.head,
                current_state=run_lifecycle.state,
                new_state="RND_RESULT_SEALED",
                transition_evidence_digest=transition_evidence,
            )
            mutations.append(run_transition.mutation)
            resulting_run_state = run_transition.current
        elif run_lifecycle is not None:
            raise SharedG2ValidationError(
                "non-result phase-output seal cannot change the RUN lifecycle"
            )
        mutations.sort(key=lambda item: item.key())
        mutation_payload = {
            "schema_version": SCHEMA_VERSION,
            "action": "SEAL_TRUSTED_PHASE_OUTPUT",
            "operation_kind": operation_kind,
            "binding": consumed.lease.binding.to_wire(),
            "binding_digest": consumed.lease.binding.digest,
            "lease_payload_digest": consumed.lease.lease_digest,
            "lease_consumption_receipt_digest": consumed.receipt.payload_digest,
            "output_attestation_digest": attestation.payload_digest,
            "output_digest": attestation.output_digest,
            "expected_global_head": expected_global_head.to_wire(),
            "transition_evidence_digest": transition_evidence,
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
        request = TransactionRequest.build(
            operation_kind=operation_kind,
            operation_id=operation_id,
            context=self.context,
            idempotency_key=idempotency_key,
            run_scope_digest=consumed.lease.binding.run_scope_digest,
            mutation_payload=mutation_payload,
            expected_output_type=PHASE_OUTPUT_SEAL_RECEIPT_KIND,
            expected_global_head=expected_global_head,
            subject_mutations=mutations,
            requested_at=requested_at,
        )
        committed = self._authority.commit(request)
        raw_seal = self._fetch_output_seal_envelope(
            committed.receipt.operation_output_digest
        )
        receipt = validate_phase_output_seal_receipt(
            raw_seal,
            consumed=consumed,
            attestation=attestation,
            seal_transaction=committed,
            context=self.context,
            expected_lease_authority_identity_digest=(
                self._lease_authority_identity_digest
            ),
            verifier=self._verifier,
        )
        return SealedPhaseOutput(
            consumed=consumed,
            attestation=attestation,
            receipt=receipt,
            transaction=committed,
            resulting_run_state=resulting_run_state,
        )


def self_check_shared_g2_lease_contract() -> None:
    if QUESTION_FAMILY_AGGREGATE_GENERATION != 0:
        raise AssertionError("question-family cross-generation aggregate key drifted")
    if tuple(ISSUE_OPERATION_BY_PHASE) != LEASED_PHASES:
        raise AssertionError("lease issue phase allowlist drifted")
    if tuple(CONSUME_OPERATION_BY_PHASE) != LEASED_PHASES:
        raise AssertionError("lease consume phase allowlist drifted")
    if tuple(OUTPUT_SEAL_OPERATION_BY_PHASE) != LEASED_PHASES:
        raise AssertionError("phase-output seal allowlist drifted")
    if len(set(PHASE_CAPABILITY_DIGESTS.values())) != len(LEASED_PHASES):
        raise AssertionError("phase capability profiles must remain domain-separated")
    for phase, profile in PHASE_CAPABILITY_PROJECTIONS.items():
        if profile["phase"] != phase:
            raise AssertionError("phase capability projection is mislabeled")
        if (
            profile["odds_price_popularity_or_market_access"] is not False
            or profile["network_access"] is not False
            or profile["credential_access"] is not False
            or profile["formal_buy"] is not False
            or profile["send_order"] is not False
            or profile["stake"] != 0
        ):
            raise AssertionError("phase capability safety boundary drifted")
    adapter = UnconfiguredSharedG2LeaseAdapter()
    try:
        adapter.fetch_phase_lease("1" * 64)
    except SharedG2Unavailable:
        pass
    else:
        raise AssertionError("unconfigured lease adapter did not fail closed")


__all__ = [
    "ALLOWED_ACTORS_BY_PHASE",
    "DECISION_CONSUMPTION_BATCH_FIELDS",
    "DECISION_CONSUMPTION_BATCH_KIND",
    "DECISION_CONSUMPTION_ENTRY_FIELDS",
    "DECISION_LEASE_BATCH_FIELDS",
    "DECISION_LEASE_BATCH_KIND",
    "DECISION_LEASE_ENTRY_FIELDS",
    "DECISION_REPLICA_IDS",
    "CONSUME_OPERATION_BY_PHASE",
    "ConsumedDecisionLeaseBatch",
    "ConsumedPhaseLease",
    "DecisionConsumptionBatchReceipt",
    "DecisionLeaseBatchReceipt",
    "ISSUE_OPERATION_BY_PHASE",
    "IrreversibleLifecycleView",
    "IssuedDecisionLeaseBatch",
    "IssuedPhaseLease",
    "LEASED_PHASES",
    "LEASE_CONSUMPTION_FIELDS",
    "LEASE_CONSUMPTION_RECEIPT_KIND",
    "LeaseBinding",
    "LeaseConsumptionReceipt",
    "MAX_LEASE_TTL_SECONDS",
    "PHASE_CAPABILITY_DIGESTS",
    "PHASE_CAPABILITY_PROJECTIONS",
    "PHASE_LEASE_KIND",
    "PHASE_OUTPUT_ATTESTATION_FIELDS",
    "PHASE_OUTPUT_ATTESTATION_KIND",
    "PHASE_OUTPUT_SEAL_FIELDS",
    "PHASE_OUTPUT_SEAL_RECEIPT_KIND",
    "QUESTION_FAMILY_AGGREGATE_GENERATION",
    "OUTPUT_SEAL_OPERATION_BY_PHASE",
    "PhaseLease",
    "PhaseOutputSealReceipt",
    "REPLICA_COMPARE_ACTOR",
    "RESULT_SEAL_ACTOR",
    "RevalidatedPhaseOutputSeal",
    "RemoteLeaseMaterialTransport",
    "SharedG2LeaseAuthorityClient",
    "SealedPhaseOutput",
    "RunLifecycleView",
    "TrustedPhaseOutputAttestation",
    "UnconfiguredSharedG2LeaseAdapter",
    "canonical_predecessor_output_digest",
    "decision_lease_batch_domain_digest",
    "phase_output_seal_state_digest",
    "self_check_shared_g2_lease_contract",
    "validate_authenticated_phase_output_seal_projection",
    "validate_decision_consumption_batch_receipt",
    "validate_decision_lease_batch_receipt",
    "validate_lease_consumption_receipt",
    "validate_phase_lease",
    "validate_phase_output_seal_receipt",
    "validate_trusted_phase_output_attestation",
]
