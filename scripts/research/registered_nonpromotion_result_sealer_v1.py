from __future__ import annotations

from dataclasses import fields, is_dataclass
import re
from typing import Any, Mapping, Protocol, runtime_checkable

from registered_nonpromotion_contract_v1 import (
    ContractError,
    RegisteredRecipe,
    canonical_digest,
    verify_canonical_run_scope,
    verify_run_scope_digest,
)

try:
    from shared_g2_lease_authority_v1 import (
        PhaseOutputSealReceipt,
        RevalidatedPhaseOutputSeal,
        phase_output_seal_state_digest,
    )
except ImportError:  # pragma: no cover - package import
    from .shared_g2_lease_authority_v1 import (
        PhaseOutputSealReceipt,
        RevalidatedPhaseOutputSeal,
        phase_output_seal_state_digest,
    )


FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHA_FIELD = re.compile(r"(?:digest|sha256)$")

COMPARISON_PROJECTION_FIELDS = frozenset(
    {
        "schema_version",
        "receipt_kind",
        "run_scope_digest",
        "recipe_digest",
        "replica_ids",
        "replica_result_digests",
        "scientific_projection_digest",
        "computed_outcome",
        "both_contract_status_valid",
        "bitwise_semantic_equality",
        "authority",
        "authenticated_phase_output_seal_required",
        "receipt_digest",
    }
)

RESULT_PROJECTION_FIELDS = frozenset(
    {
        "schema_version",
        "projection_kind",
        "gate_kind",
        "run_scope_digest",
        "recipe_id",
        "recipe_version",
        "recipe_digest",
        "question_family_digest",
        "semantic_subject_digest",
        "exact_subject_digest",
        "comparison_projection_digest",
        "policy_schema_verifier_digest",
        "scientific_projection_digest",
        "computed_outcome",
        "evidence_purpose_class",
        "source_authority_class",
        "confirmatory",
        "reused_development_oos",
        "strict_t3_rows",
        "promotion_eligible",
        "score_credit",
        "shadow_transition_supported",
        "production_transition_supported",
        "formal_buy",
        "send_order",
        "stake",
        "result_projection_digest",
    }
)

SEALED_RESULT_FIELDS = RESULT_PROJECTION_FIELDS | frozenset(
    {
        "comparison_phase_output_seal_receipt_digest",
        "result_phase_output_seal_receipt_digest",
        "result_digest",
    }
)


@runtime_checkable
class AuthenticatedPhaseOutputSealVerifier(Protocol):
    """Read-only shared-G2 verifier; it must fetch remote current authority.

    Implementations must authenticate the immutable seal, check the current
    PHASE_OUTPUT subject head/state, and verify a current independent monotonic
    witness.  A local cache, caller assertion, or boolean is not sufficient.
    """

    def fetch_and_revalidate_unrevoked_phase_output_seal(
        self, receipt_payload_digest: str
    ) -> RevalidatedPhaseOutputSeal:
        ...


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not FULL_SHA256.fullmatch(value)
        or value == "0" * 64
    ):
        raise ContractError(f"{label} must be a non-zero full lowercase SHA-256")
    return value


def _exact_mapping(
    value: Mapping[str, Any], *, fields_set: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    payload = dict(value)
    missing = sorted(fields_set - set(payload))
    extra = sorted(set(payload) - fields_set)
    if missing or extra:
        raise ContractError(
            f"{label} fields must be exact; missing={missing}, extra={extra}"
        )
    return payload


def _validate_all_sha_fields(value: Any, *, label: str) -> None:
    """Validate every digest/SHA-256 field in an accepted evidence object."""

    def walk(item: Any, path: str, field_name: str | None = None) -> None:
        if field_name is not None and SHA_FIELD.search(field_name):
            _sha256(item, label=path)
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ContractError(f"{path} contains a non-string field name")
                walk(nested, f"{path}.{key}", key)
            return
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                walk(getattr(item, field.name), f"{path}.{field.name}", field.name)
            return
        if isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                walk(nested, f"{path}[{index}]")

    walk(value, label)


def _verify_embedded_digest(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    stored = _sha256(value.get(field), label=f"{label}.{field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if canonical_digest(unsigned) != stored:
        raise ContractError(f"{label} digest mismatch")
    return stored


def _verify_comparison_projection(
    *, run_scope: Mapping[str, Any], comparison_projection: Mapping[str, Any]
) -> tuple[str, str, str]:
    payload = _exact_mapping(
        comparison_projection,
        fields_set=COMPARISON_PROJECTION_FIELDS,
        label="replica comparison projection",
    )
    _validate_all_sha_fields(payload, label="replica comparison projection")
    comparison_digest = _verify_embedded_digest(
        payload,
        field="receipt_digest",
        label="replica comparison projection",
    )
    run_digest = verify_run_scope_digest(run_scope)
    if payload["schema_version"] != 1 or isinstance(
        payload["schema_version"], bool
    ):
        raise ContractError("replica comparison schema_version must be 1")
    if payload["receipt_kind"] != "REPLICA_COMPARISON":
        raise ContractError("unexpected comparison projection kind")
    if payload["run_scope_digest"] != run_digest:
        raise ContractError("comparison projection belongs to another run")
    if payload["recipe_digest"] != run_scope.get("recipe_digest"):
        raise ContractError("comparison projection recipe mismatch")
    if payload["replica_ids"] != ["clean_a", "clean_b"]:
        raise ContractError("comparison projection requires the two frozen replicas")
    result_digests = payload["replica_result_digests"]
    if (
        not isinstance(result_digests, list)
        or len(result_digests) != 2
        or result_digests != sorted(result_digests)
        or len(set(result_digests)) != 2
    ):
        raise ContractError("comparison replica result digests are not canonical")
    for index, digest in enumerate(result_digests):
        _sha256(digest, label=f"replica result digest {index}")
    if payload["both_contract_status_valid"] is not True:
        raise ContractError("both replicas must be contract-valid")
    if payload["bitwise_semantic_equality"] is not True:
        raise ContractError("replica semantic projections must be identical")
    if payload["authority"] is not False:
        raise ContractError("comparison projection must not self-assert authority")
    if payload["authenticated_phase_output_seal_required"] is not True:
        raise ContractError("comparison projection must require shared-G2 sealing")
    outcome = payload["computed_outcome"]
    if outcome not in {"NO_DECISION_EFFECT", "DIRECTIONAL_EFFECT", "INVALID"}:
        raise ContractError("computed outcome is outside the registered result enum")
    scientific_digest = _sha256(
        payload["scientific_projection_digest"],
        label="comparison scientific projection digest",
    )
    return comparison_digest, scientific_digest, outcome


def project_registered_nonpromotion_result(
    *,
    registered: RegisteredRecipe,
    run_scope: Mapping[str, Any],
    comparison_projection: Mapping[str, Any],
    policy_schema_verifier_digest: str,
) -> dict[str, Any]:
    """Build the deterministic result projection without claiming authority."""

    run_digest = verify_canonical_run_scope(registered, run_scope)
    _validate_all_sha_fields(run_scope, label="run scope")
    comparison_digest, scientific_digest, outcome = _verify_comparison_projection(
        run_scope=run_scope,
        comparison_projection=comparison_projection,
    )
    aggregate_digest = _sha256(
        policy_schema_verifier_digest,
        label="policy/schema/verifier digest",
    )
    projection = {
        "schema_version": 1,
        "projection_kind": "REGISTERED_NONPROMOTION_RESULT_PROJECTION_V1",
        "gate_kind": run_scope["gate_kind"],
        "run_scope_digest": run_digest,
        "recipe_id": run_scope["recipe_id"],
        "recipe_version": run_scope["recipe_version"],
        "recipe_digest": _sha256(
            run_scope["recipe_digest"], label="run scope recipe digest"
        ),
        "question_family_digest": _sha256(
            run_scope["question_family_digest"],
            label="run scope question family digest",
        ),
        "semantic_subject_digest": _sha256(
            run_scope["semantic_subject_digest"],
            label="run scope semantic subject digest",
        ),
        "exact_subject_digest": _sha256(
            run_scope["exact_subject_digest"],
            label="run scope exact subject digest",
        ),
        "comparison_projection_digest": comparison_digest,
        "policy_schema_verifier_digest": aggregate_digest,
        "scientific_projection_digest": scientific_digest,
        "computed_outcome": outcome,
        "evidence_purpose_class": "DIAGNOSTIC_NONPROMOTION",
        "source_authority_class": "B_LOCAL_HASHED",
        "confirmatory": False,
        "reused_development_oos": True,
        "strict_t3_rows": 0,
        "promotion_eligible": False,
        "score_credit": 0,
        "shadow_transition_supported": False,
        "production_transition_supported": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    projection["result_projection_digest"] = canonical_digest(projection)
    return projection


def _verify_result_projection(
    *,
    registered: RegisteredRecipe,
    run_scope: Mapping[str, Any],
    result_projection: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    projection = _exact_mapping(
        result_projection,
        fields_set=RESULT_PROJECTION_FIELDS,
        label="registered result projection",
    )
    _validate_all_sha_fields(projection, label="registered result projection")
    projection_digest = _verify_embedded_digest(
        projection,
        field="result_projection_digest",
        label="registered result projection",
    )
    if projection["projection_kind"] != "REGISTERED_NONPROMOTION_RESULT_PROJECTION_V1":
        raise ContractError("unexpected result projection kind")
    if projection["schema_version"] != 1 or isinstance(
        projection["schema_version"], bool
    ):
        raise ContractError("result projection schema_version must be 1")
    run_digest = verify_canonical_run_scope(registered, run_scope)
    if projection["run_scope_digest"] != run_digest:
        raise ContractError("result projection belongs to another run")
    for field in (
        "gate_kind",
        "recipe_id",
        "recipe_version",
        "recipe_digest",
        "question_family_digest",
        "semantic_subject_digest",
        "exact_subject_digest",
    ):
        if projection[field] != run_scope.get(field):
            raise ContractError(f"result projection run binding mismatch in {field}")
    safety = {
        "confirmatory": False,
        "promotion_eligible": False,
        "score_credit": 0,
        "shadow_transition_supported": False,
        "production_transition_supported": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    for field, expected in safety.items():
        if projection[field] != expected:
            raise ContractError(f"result projection safety field drifted: {field}")
    exact_contract_fields = {
        "evidence_purpose_class": "DIAGNOSTIC_NONPROMOTION",
        "source_authority_class": "B_LOCAL_HASHED",
        "reused_development_oos": True,
        "strict_t3_rows": 0,
    }
    for field, expected in exact_contract_fields.items():
        if projection[field] != expected:
            raise ContractError(f"result projection contract field drifted: {field}")
    if projection["computed_outcome"] not in {
        "NO_DECISION_EFFECT",
        "DIRECTIONAL_EFFECT",
        "INVALID",
    }:
        raise ContractError("computed outcome is outside the registered result enum")
    return projection, projection_digest


def _remote_revalidate_seal(
    *,
    seal_verifier: AuthenticatedPhaseOutputSealVerifier,
    receipt_payload_digest: str,
    expected_run_scope_digest: str,
    expected_recipe_digest: str,
    expected_phase: str,
    expected_actor: str,
    expected_output_digest: str,
) -> Any:
    receipt_digest = _sha256(
        receipt_payload_digest, label=f"{expected_phase} seal receipt digest"
    )
    evidence = seal_verifier.fetch_and_revalidate_unrevoked_phase_output_seal(
        receipt_digest
    )
    receipt = getattr(evidence, "receipt", None)
    if not isinstance(evidence, RevalidatedPhaseOutputSeal) or not isinstance(
        receipt, PhaseOutputSealReceipt
    ):
        raise ContractError(
            "shared-G2 verifier did not return typed phase-output seal evidence"
        )
    _validate_all_sha_fields(evidence, label=f"authenticated {expected_phase} seal")
    scalar_checks = {
        "payload_digest": receipt_digest,
        "run_scope_digest": expected_run_scope_digest,
        "recipe_digest": expected_recipe_digest,
        "phase": expected_phase,
        "replica_id": expected_actor,
        "attempt": 1,
        "output_digest": expected_output_digest,
    }
    for field, expected in scalar_checks.items():
        if getattr(receipt, field) != expected:
            raise ContractError(
                f"authenticated {expected_phase} seal mismatch in {field}"
            )
    expected_operation = (
        "RND_RESULT_SEAL"
        if expected_phase == "RESULT_SEAL"
        else "RND_PHASE_OUTPUT_SEAL"
    )
    if receipt.operation_kind != expected_operation:
        raise ContractError(
            f"authenticated {expected_phase} seal operation kind mismatch"
        )
    subject_head = getattr(evidence, "subject_head", None)
    if (
        subject_head is None
        or getattr(subject_head, "subject_kind", None) != "PHASE_OUTPUT"
    ):
        raise ContractError(
            "shared-G2 verifier did not return the current PHASE_OUTPUT head"
        )
    if subject_head.subject_digest != receipt.phase_output_subject_digest:
        raise ContractError("authenticated phase-output subject digest mismatch")
    for field in ("subject_digest", "head_digest", "state_digest"):
        _sha256(
            getattr(subject_head, field, None),
            label=f"authenticated {expected_phase} subject head {field}",
        )
    if (
        not isinstance(getattr(subject_head, "sequence", None), int)
        or isinstance(subject_head.sequence, bool)
        or subject_head.sequence != 1
    ):
        raise ContractError(
            "authenticated phase-output subject head is not the one-shot commit"
        )
    if subject_head.state_digest != phase_output_seal_state_digest(receipt):
        raise ContractError("authenticated phase-output state does not bind the seal")
    subject_snapshot = getattr(evidence, "subject_snapshot", None)
    if subject_snapshot is None or subject_snapshot.subject_head != subject_head:
        raise ContractError("shared-G2 current subject snapshot mismatch")
    authority_snapshot = getattr(evidence, "authority_snapshot", None)
    if authority_snapshot is None:
        raise ContractError(
            "shared-G2 verifier did not return a witnessed authority snapshot"
        )
    if subject_snapshot.global_head != authority_snapshot.global_head:
        raise ContractError("shared-G2 subject and witness snapshots disagree")
    if (
        authority_snapshot.witness.observed_global_head
        != authority_snapshot.global_head
    ):
        raise ContractError(
            "shared-G2 monotonic witness does not bind the current head"
        )
    if authority_snapshot.global_head.sequence < receipt.sealed_global_head.sequence:
        raise ContractError("shared-G2 current head predates the authenticated seal")
    return evidence


def _verify_comparison_to_result_seal_order(
    comparison_evidence: RevalidatedPhaseOutputSeal,
    result_evidence: RevalidatedPhaseOutputSeal,
) -> None:
    comparison_head = comparison_evidence.receipt.sealed_global_head
    result_from_head = result_evidence.receipt.sealed_from_global_head
    authority_fields = (
        "authority_id",
        "activation_epoch",
        "backend_identity_digest",
        "cutover_receipt_digest",
    )
    if any(
        getattr(comparison_head, field) != getattr(result_from_head, field)
        for field in authority_fields
    ):
        raise ContractError("comparison and result seals use different G2 authorities")
    if result_from_head.sequence < comparison_head.sequence:
        raise ContractError("result seal authority head predates the comparison seal")


def seal_registered_nonpromotion_result(
    *,
    registered: RegisteredRecipe,
    run_scope: Mapping[str, Any],
    result_projection: Mapping[str, Any],
    comparison_phase_output_seal_receipt_digest: str,
    result_phase_output_seal_receipt_digest: str,
    seal_verifier: AuthenticatedPhaseOutputSealVerifier,
) -> dict[str, Any]:
    """Attach only remotely revalidated, unrevoked shared-G2 authority evidence."""

    projection, projection_digest = _verify_result_projection(
        registered=registered,
        run_scope=run_scope,
        result_projection=result_projection,
    )
    run_digest = projection["run_scope_digest"]
    recipe_digest = projection["recipe_digest"]
    comparison_evidence = _remote_revalidate_seal(
        seal_verifier=seal_verifier,
        receipt_payload_digest=comparison_phase_output_seal_receipt_digest,
        expected_run_scope_digest=run_digest,
        expected_recipe_digest=recipe_digest,
        expected_phase="REPLICA_COMPARE",
        expected_actor="lane_coordinator",
        expected_output_digest=projection["comparison_projection_digest"],
    )
    result_evidence = _remote_revalidate_seal(
        seal_verifier=seal_verifier,
        receipt_payload_digest=result_phase_output_seal_receipt_digest,
        expected_run_scope_digest=run_digest,
        expected_recipe_digest=recipe_digest,
        expected_phase="RESULT_SEAL",
        expected_actor="canonical_sealer",
        expected_output_digest=projection_digest,
    )
    _verify_comparison_to_result_seal_order(
        comparison_evidence, result_evidence
    )
    comparison_receipt = comparison_evidence.receipt
    result_receipt = result_evidence.receipt

    sealed = dict(projection)
    sealed.update(
        {
            "comparison_phase_output_seal_receipt_digest": (
                comparison_receipt.payload_digest
            ),
            "result_phase_output_seal_receipt_digest": result_receipt.payload_digest,
        }
    )
    sealed["result_digest"] = canonical_digest(sealed)
    return sealed


def validate_exact_replay(
    *,
    registered: RegisteredRecipe,
    run_scope: Mapping[str, Any],
    sealed_result: Mapping[str, Any],
    seal_verifier: AuthenticatedPhaseOutputSealVerifier,
) -> Mapping[str, Any]:
    """Revalidate both remote seals for read-only replay; never rerun on a miss."""

    sealed = _exact_mapping(
        sealed_result,
        fields_set=SEALED_RESULT_FIELDS,
        label="sealed registered result",
    )
    _validate_all_sha_fields(sealed, label="sealed registered result")
    _verify_embedded_digest(
        sealed, field="result_digest", label="sealed registered result"
    )
    projection = {field: sealed[field] for field in RESULT_PROJECTION_FIELDS}
    verified_projection, projection_digest = _verify_result_projection(
        registered=registered,
        run_scope=run_scope,
        result_projection=projection,
    )
    comparison_evidence = _remote_revalidate_seal(
        seal_verifier=seal_verifier,
        receipt_payload_digest=sealed[
            "comparison_phase_output_seal_receipt_digest"
        ],
        expected_run_scope_digest=verified_projection["run_scope_digest"],
        expected_recipe_digest=verified_projection["recipe_digest"],
        expected_phase="REPLICA_COMPARE",
        expected_actor="lane_coordinator",
        expected_output_digest=verified_projection["comparison_projection_digest"],
    )
    result_evidence = _remote_revalidate_seal(
        seal_verifier=seal_verifier,
        receipt_payload_digest=sealed["result_phase_output_seal_receipt_digest"],
        expected_run_scope_digest=verified_projection["run_scope_digest"],
        expected_recipe_digest=verified_projection["recipe_digest"],
        expected_phase="RESULT_SEAL",
        expected_actor="canonical_sealer",
        expected_output_digest=projection_digest,
    )
    _verify_comparison_to_result_seal_order(
        comparison_evidence, result_evidence
    )
    return sealed_result


__all__ = [
    "AuthenticatedPhaseOutputSealVerifier",
    "project_registered_nonpromotion_result",
    "seal_registered_nonpromotion_result",
    "validate_exact_replay",
]
