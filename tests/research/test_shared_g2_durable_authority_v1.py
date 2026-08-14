from __future__ import annotations

import inspect
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import shared_g2_durable_ledger_v1 as durable
import shared_g2_lease_authority_v1 as lease


H = {str(index): f"{index:x}" * 64 for index in range(1, 10)}
H.update({"a": "a" * 64, "b": "b" * 64, "c": "c" * 64})


class AcceptingSyntheticVerifier:
    """Test-only verifier; this class is never imported by runtime modules."""

    def verify_authenticated_payload(
        self,
        *,
        domain_separator: str,
        payload: bytes,
        authentication: dict[str, object],
    ) -> bool:
        return bool(domain_separator and payload and authentication)


def envelope(
    payload_type: str,
    payload: dict[str, object],
    *,
    scheme: str = "ED25519",
    attestation_digest: str = H["9"],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "envelope_kind": durable.AUTHENTICATED_ENVELOPE_KIND,
        "payload_type": payload_type,
        "payload_digest": durable.canonical_digest(payload),
        "payload": payload,
        "authentication": {
            "scheme": scheme,
            "key_id": "remote-authority-key-v1",
            "signature": "A" * 43,
            "attestation_digest": attestation_digest,
            "issued_at": "2026-08-15T00:00:03Z",
        },
    }


def expectations() -> durable.CutoverExpectations:
    return durable.CutoverExpectations(
        repository="kazuponbaseball-cell/keiba_ai_project",
        base_branch="main",
        current_main_sha="a" * 40,
        authority_id="shared-g2-authority-v1",
        activation_epoch="epoch-20260815-v1",
        backend_identity_digest=H["1"],
        witness_identity_digest=H["2"],
        runtime_blob_digests={"runner": H["3"]},
        migration_digests={
            "legacy_event_chain_digest": H["4"],
            "global_comment_id_set_digest": H["5"],
            "terminal_and_nonterminal_subject_head_set_digest": H["6"],
        },
        old_writer_fence_digest=H["7"],
        second_remote_compare_digest=H["8"],
    )


def context() -> durable.CutoverContext:
    expected = expectations()
    genesis = durable.GlobalHead(
        authority_id=expected.authority_id,
        activation_epoch=expected.activation_epoch,
        backend_identity_digest=expected.backend_identity_digest,
        cutover_receipt_digest=H["a"],
        sequence=0,
        head_digest=H["b"],
        observed_at="2026-08-15T00:00:00Z",
    )
    return durable.CutoverContext(
        expectations=expected,
        cutover_receipt_digest=H["a"],
        authenticated_payload_digest=H["c"],
        initial_global_head=genesis,
        initial_witness_checkpoint_digest=H["8"],
        activated_at="2026-08-15T00:00:00Z",
    )


def global_head(
    ctx: durable.CutoverContext,
    *,
    sequence: int,
    digest: str,
    observed_at: str,
) -> durable.GlobalHead:
    return durable.GlobalHead(
        authority_id=ctx.expectations.authority_id,
        activation_epoch=ctx.expectations.activation_epoch,
        backend_identity_digest=ctx.expectations.backend_identity_digest,
        cutover_receipt_digest=ctx.cutover_receipt_digest,
        sequence=sequence,
        head_digest=digest,
        observed_at=observed_at,
    )


def binding(
    *,
    phase: str = "SETTLEMENT_DIAGNOSTIC",
    replica_id: str = "clean_a",
    attempt: int = 1,
    phase_capability_digest: str | None = None,
    predecessor_receipt_digest: str = H["6"],
) -> lease.LeaseBinding:
    return lease.LeaseBinding(
        run_scope_digest=H["1"],
        run_generation=0,
        recipe_digest=H["2"],
        semantic_subject_digest=H["3"],
        exact_run_subject_digest=H["4"],
        question_family_digest=H["5"],
        replica_id=replica_id,
        phase=phase,
        attempt=attempt,
        predecessor_receipt_digest=predecessor_receipt_digest,
        issue_revalidation_receipt_digest=H["7"],
        policy_digest=H["8"],
        schema_digest=H["9"],
        verifier_digest=H["a"],
        executor_digest=H["b"],
        runner_digest=H["c"],
        bound_capability_profile_digest=H["1"],
        phase_capability_digest=(
            phase_capability_digest
            if phase_capability_digest is not None
            else lease.PHASE_CAPABILITY_DIGESTS[phase]
        ),
        ttl_seconds=1200,
    )


def synthetic_consumed_phase(
    ctx: durable.CutoverContext,
) -> lease.ConsumedPhaseLease:
    bound = binding()
    issue_previous = global_head(
        ctx,
        sequence=1,
        digest=H["1"],
        observed_at="2026-08-15T00:00:01Z",
    )
    issue_new = global_head(
        ctx,
        sequence=2,
        digest=H["2"],
        observed_at="2026-08-15T00:00:02Z",
    )
    consume_new = global_head(
        ctx,
        sequence=3,
        digest=H["3"],
        observed_at="2026-08-15T00:00:03Z",
    )
    witness = durable.WitnessCheckpoint(
        authority_id=ctx.expectations.authority_id,
        activation_epoch=ctx.expectations.activation_epoch,
        backend_identity_digest=ctx.expectations.backend_identity_digest,
        witness_identity_digest=ctx.expectations.witness_identity_digest,
        cutover_receipt_digest=ctx.cutover_receipt_digest,
        checkpoint_sequence=3,
        observed_global_head=issue_new,
        previous_checkpoint_digest=H["4"],
        checkpoint_digest=H["5"],
        witnessed_at="2026-08-15T00:00:02Z",
    )
    issue_receipt = durable.TransactionReceipt(
        payload_digest=H["4"],
        envelope_digest=H["5"],
        transaction_id="transaction-issue-001",
        operation_kind="RND_PHASE_LEASE_ISSUE",
        operation_id="operation-issue-001",
        request_digest=H["6"],
        run_scope_digest=bound.run_scope_digest,
        mutation_digest=H["7"],
        previous_global_head=issue_previous,
        new_global_head=issue_new,
        previous_subject_heads=(),
        new_subject_heads=(),
        operation_output_type=lease.PHASE_LEASE_KIND,
        operation_output_digest=H["8"],
        committed_at="2026-08-15T00:00:02Z",
    )
    issue_transaction = durable.CommittedTransaction(
        receipt=issue_receipt,
        witness=witness,
    )
    phase_lease = lease.PhaseLease(
        lease_digest=H["8"],
        envelope_digest=H["9"],
        lease_id="lease-clean-a-settlement-001",
        binding=bound,
        issued_from_global_head=issue_previous,
        issued_global_head=issue_new,
        issue_transaction_id=issue_receipt.transaction_id,
        lease_authority_identity_digest=H["a"],
        issued_at="2026-08-15T00:00:02Z",
        expires_at="2026-08-15T00:20:02Z",
    )
    consume_receipt = durable.TransactionReceipt(
        payload_digest=H["9"],
        envelope_digest=H["a"],
        transaction_id="transaction-consume-001",
        operation_kind="RND_PHASE_LEASE_CONSUME",
        operation_id="operation-consume-001",
        request_digest=H["b"],
        run_scope_digest=bound.run_scope_digest,
        mutation_digest=H["c"],
        previous_global_head=issue_new,
        new_global_head=consume_new,
        previous_subject_heads=(),
        new_subject_heads=(),
        operation_output_type=lease.LEASE_CONSUMPTION_RECEIPT_KIND,
        operation_output_digest=H["1"],
        committed_at="2026-08-15T00:00:03Z",
    )
    consume_witness = durable.WitnessCheckpoint(
        authority_id=ctx.expectations.authority_id,
        activation_epoch=ctx.expectations.activation_epoch,
        backend_identity_digest=ctx.expectations.backend_identity_digest,
        witness_identity_digest=ctx.expectations.witness_identity_digest,
        cutover_receipt_digest=ctx.cutover_receipt_digest,
        checkpoint_sequence=4,
        observed_global_head=consume_new,
        previous_checkpoint_digest=H["5"],
        checkpoint_digest=H["6"],
        witnessed_at="2026-08-15T00:00:03Z",
    )
    consume_transaction = durable.CommittedTransaction(
        receipt=consume_receipt,
        witness=consume_witness,
    )
    consumption = lease.LeaseConsumptionReceipt(
        payload_digest=H["1"],
        envelope_digest=H["2"],
        lease_id=phase_lease.lease_id,
        lease_payload_digest=phase_lease.lease_digest,
        binding_digest=bound.digest,
        operation_kind="RND_PHASE_LEASE_CONSUME",
        dispatch_digest=H["2"],
        consume_revalidation_receipt_digest=H["3"],
        consumed_from_global_head=issue_new,
        consumed_global_head=consume_new,
        consume_transaction_id=consume_receipt.transaction_id,
        consumed_at="2026-08-15T00:00:03Z",
    )
    return lease.ConsumedPhaseLease(
        lease=phase_lease,
        receipt=consumption,
        issue_transaction=issue_transaction,
        transaction=consume_transaction,
        resulting_run_state=None,
    )


def synthetic_sealed_output(
    ctx: durable.CutoverContext,
    *,
    phase: str,
    replica_id: str,
    receipt_payload_digest: str,
    output_digest: str,
) -> lease.SealedPhaseOutput:
    consumed = synthetic_consumed_phase(ctx)
    attestation = lease.TrustedPhaseOutputAttestation(
        payload_digest=H["4"],
        envelope_digest=H["5"],
        binding_digest=H["3"],
        lease_payload_digest=H["8"],
        lease_consumption_receipt_digest=H["1"],
        consumed_global_head=consumed.transaction.receipt.new_global_head,
        output_digest=output_digest,
        attester_identity_digest=H["5"],
        attested_at="2026-08-15T00:00:04Z",
    )
    receipt = lease.PhaseOutputSealReceipt(
        payload_digest=receipt_payload_digest,
        envelope_digest=H["9"],
        run_scope_digest=H["1"],
        recipe_digest=H["2"],
        replica_id=replica_id,
        phase=phase,
        attempt=1,
        binding_digest=durable.canonical_digest(
            {"phase": phase, "replica_id": replica_id}
        ),
        phase_output_subject_digest=durable.canonical_digest(
            {
                "phase": phase,
                "replica_id": replica_id,
                "receipt_payload_digest": receipt_payload_digest,
            }
        ),
        lease_payload_digest=H["8"],
        lease_consumption_receipt_digest=H["1"],
        output_attestation_digest=attestation.payload_digest,
        output_digest=output_digest,
        operation_kind=lease.OUTPUT_SEAL_OPERATION_BY_PHASE[phase],
        sealed_from_global_head=global_head(
            ctx,
            sequence=3,
            digest=H["3"],
            observed_at="2026-08-15T00:00:03Z",
        ),
        sealed_global_head=global_head(
            ctx,
            sequence=4,
            digest=H["4"],
            observed_at="2026-08-15T00:00:04Z",
        ),
        seal_transaction_id="transaction-phase-output-seal-001",
        sealed_at="2026-08-15T00:00:04Z",
    )
    return lease.SealedPhaseOutput(
        consumed=consumed,
        attestation=attestation,
        receipt=receipt,
        transaction=consumed.transaction,
        resulting_run_state=None,
    )


def subject_mutation(kind: str, digest: str) -> durable.SubjectMutation:
    return durable.SubjectMutation(
        subject_kind=kind,
        subject_digest=digest,
        generation=0,
        expected_sequence=0,
        expected_head_digest=durable.ZERO_SHA256,
        new_state_digest=durable.canonical_digest(
            {"kind": kind, "subject_digest": digest}
        ),
    )


class CommitProbeStop(RuntimeError):
    pass


class CapturingAuthority:
    def __init__(self, ctx: durable.CutoverContext) -> None:
        self.context = ctx
        self.request: durable.TransactionRequest | None = None

    def commit(self, request: durable.TransactionRequest) -> None:
        self.request = request
        raise CommitProbeStop("synthetic commit boundary reached")


class SharedG2NegativeContractTests(unittest.TestCase):
    def test_unconfigured_adapters_always_fail_closed(self) -> None:
        ledger_adapter = durable.UnconfiguredSharedG2Adapter()
        lease_adapter = lease.UnconfiguredSharedG2LeaseAdapter()
        with self.assertRaises(durable.SharedG2Unavailable):
            ledger_adapter.fetch_current_head()
        with self.assertRaises(durable.SharedG2Unavailable):
            ledger_adapter.commit_transaction({})
        with self.assertRaises(durable.SharedG2Unavailable):
            lease_adapter.fetch_phase_lease(H["1"])
        with self.assertRaises(durable.SharedG2Unavailable):
            lease_adapter.fetch_phase_output_attestation(H["1"])
        with self.assertRaises(durable.SharedG2Unavailable):
            lease_adapter.fetch_phase_output_seal_receipt(H["1"])

    def test_local_or_test_authentication_scheme_is_rejected(self) -> None:
        verifier = AcceptingSyntheticVerifier()
        for scheme in ("LOCAL", "TEST", "IN_MEMORY", "SELF_ASSERTED"):
            with self.subTest(scheme=scheme):
                with self.assertRaises(durable.SharedG2AuthenticationError):
                    durable.validate_authenticated_envelope(
                        envelope("SYNTHETIC_PAYLOAD_V1", {"value": 1}, scheme=scheme),
                        expected_payload_type="SYNTHETIC_PAYLOAD_V1",
                        verifier=verifier,
                    )

    def test_authenticated_envelope_cannot_self_assert_another_identity(
        self,
    ) -> None:
        authenticated = durable.validate_authenticated_envelope(
            envelope("SYNTHETIC_PAYLOAD_V1", {"value": 1}),
            expected_payload_type="SYNTHETIC_PAYLOAD_V1",
            verifier=AcceptingSyntheticVerifier(),
        )
        with self.assertRaisesRegex(
            durable.SharedG2AuthenticationError, "unexpected identity"
        ):
            durable.validate_authenticated_identity_binding(
                authenticated,
                expected_identity_digest=H["8"],
                label="synthetic authority",
            )

    def test_head_with_wrong_cutover_is_stale(self) -> None:
        ctx = context()
        stale = durable.GlobalHead(
            authority_id=ctx.expectations.authority_id,
            activation_epoch=ctx.expectations.activation_epoch,
            backend_identity_digest=ctx.expectations.backend_identity_digest,
            cutover_receipt_digest=H["9"],
            sequence=1,
            head_digest=H["1"],
            observed_at="2026-08-15T00:00:01Z",
        )
        mutation = durable.SubjectMutation(
            subject_kind="RUN",
            subject_digest=H["1"],
            generation=0,
            expected_sequence=0,
            expected_head_digest=durable.ZERO_SHA256,
            new_state_digest=H["2"],
        )
        with self.assertRaises(durable.SharedG2StaleAuthority):
            durable.TransactionRequest.build(
                operation_kind="RND_SCOPE_SEAL",
                operation_id="scope-seal-001",
                context=ctx,
                idempotency_key="scope-seal-001",
                run_scope_digest=H["1"],
                mutation_payload={
                    "schema_version": 1,
                    "action": "SEAL_RUN_SCOPE",
                    "operation_kind": "RND_SCOPE_SEAL",
                    "safety": {
                        "formal_buy": False,
                        "send_order": False,
                        "stake": 0,
                    },
                },
                expected_output_type="SCOPE_SEAL_RECEIPT_V1",
                expected_global_head=stale,
                subject_mutations=[mutation],
                requested_at="2026-08-15T00:00:01Z",
            )

    def test_witness_for_another_head_is_rejected(self) -> None:
        ctx = context()
        expected_head = global_head(
            ctx,
            sequence=4,
            digest=H["1"],
            observed_at="2026-08-15T00:00:01Z",
        )
        wrong_head = global_head(
            ctx,
            sequence=4,
            digest=H["2"],
            observed_at="2026-08-15T00:00:01Z",
        )
        payload: dict[str, object] = {
            "schema_version": 1,
            "checkpoint_kind": durable.WITNESS_CHECKPOINT_KIND,
            "authority_id": ctx.expectations.authority_id,
            "activation_epoch": ctx.expectations.activation_epoch,
            "backend_identity_digest": ctx.expectations.backend_identity_digest,
            "witness_identity_digest": ctx.expectations.witness_identity_digest,
            "cutover_receipt_digest": ctx.cutover_receipt_digest,
            "checkpoint_sequence": 5,
            "observed_global_head": wrong_head.to_wire(),
            "previous_checkpoint_digest": H["3"],
            "checkpoint_digest": durable.ZERO_SHA256,
            "witnessed_at": "2026-08-15T00:00:02Z",
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
        projection = dict(payload)
        projection.pop("checkpoint_digest")
        payload["checkpoint_digest"] = durable.canonical_digest(projection)
        with self.assertRaises(durable.SharedG2StaleAuthority):
            durable.validate_witness_checkpoint(
                envelope(
                    durable.WITNESS_CHECKPOINT_KIND,
                    payload,
                    attestation_digest=ctx.expectations.witness_identity_digest,
                ),
                context=ctx,
                expected_head=expected_head,
                verifier=AcceptingSyntheticVerifier(),
            )

    def test_lease_domain_is_exact_and_retry_attempt_is_rejected(self) -> None:
        clean_a = binding(replica_id="clean_a")
        clean_b = binding(replica_id="clean_b")
        decision = binding(phase="DECISION_FREEZE")
        self.assertNotEqual(clean_a.domain_digest, clean_b.domain_digest)
        self.assertNotEqual(clean_a.domain_digest, decision.domain_digest)
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "attempt must be exactly 1"
        ):
            binding(attempt=2)

    def test_decision_batch_requires_exact_replicas_and_mutation_counts(self) -> None:
        clean_a = binding(phase="DECISION_FREEZE", replica_id="clean_a")
        clean_b = binding(phase="DECISION_FREEZE", replica_id="clean_b")
        self.assertEqual(
            tuple(
                item.replica_id
                for item in lease._validate_decision_binding_pair(
                    (clean_b, clean_a)
                )
            ),
            lease.DECISION_REPLICA_IDS,
        )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "exactly clean_a and clean_b"
        ):
            lease._validate_decision_binding_pair((clean_a, clean_a))
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "exactly two replica bindings"
        ):
            lease._validate_decision_binding_pair((clean_a,))

        issue_mutations = sorted(
            (
                subject_mutation("LEASE", H["1"]),
                subject_mutation("LEASE", H["2"]),
                subject_mutation("RUN", H["3"]),
            ),
            key=lambda item: item.key(),
        )
        normalized_issue = durable.normalize_subject_mutations(
            issue_mutations,
            operation_kind="RND_DECISION_LEASE_BATCH_ISSUE",
        )
        self.assertEqual(
            Counter(item.subject_kind for item in normalized_issue),
            Counter({"LEASE": 2, "RUN": 1}),
        )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "multiplicity must be exact"
        ):
            durable.normalize_subject_mutations(
                [item for item in issue_mutations if item.subject_digest != H["2"]],
                operation_kind="RND_DECISION_LEASE_BATCH_ISSUE",
            )

        irreversible_mutations = sorted(
            (
                subject_mutation("LEASE", H["1"]),
                subject_mutation("LEASE", H["2"]),
                subject_mutation("RUN", H["3"]),
                subject_mutation("SEMANTIC_SUBJECT", H["4"]),
                subject_mutation("EXACT_SUBJECT", H["5"]),
                subject_mutation("QUESTION_FAMILY", H["6"]),
            ),
            key=lambda item: item.key(),
        )
        normalized_consume = durable.normalize_subject_mutations(
            irreversible_mutations,
            operation_kind="RND_DECISION_IRREVERSIBLE_START",
        )
        self.assertEqual(
            Counter(item.subject_kind for item in normalized_consume),
            Counter(
                {
                    "LEASE": 2,
                    "RUN": 1,
                    "SEMANTIC_SUBJECT": 1,
                    "EXACT_SUBJECT": 1,
                    "QUESTION_FAMILY": 1,
                }
            ),
        )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "multiplicity must be exact"
        ):
            durable.normalize_subject_mutations(
                [
                    item
                    for item in irreversible_mutations
                    if not (
                        item.subject_kind == "LEASE"
                        and item.subject_digest == H["2"]
                    )
                ],
                operation_kind="RND_DECISION_IRREVERSIBLE_START",
            )

        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "action is not registered"
        ):
            durable.TransactionRequest.build(
                operation_kind="RND_DECISION_LEASE_BATCH_ISSUE",
                operation_id="wrong-batch-action-001",
                context=context(),
                idempotency_key="wrong-batch-action-001",
                run_scope_digest=H["1"],
                mutation_payload={
                    "schema_version": 1,
                    "action": "ISSUE_ONE_SHOT_PHASE_LEASE",
                    "operation_kind": "RND_DECISION_LEASE_BATCH_ISSUE",
                    "safety": {
                        "formal_buy": False,
                        "send_order": False,
                        "stake": 0,
                    },
                },
                expected_output_type=lease.DECISION_LEASE_BATCH_KIND,
                expected_global_head=global_head(
                    context(),
                    sequence=1,
                    digest=H["1"],
                    observed_at="2026-08-15T00:00:01Z",
                ),
                subject_mutations=issue_mutations,
                requested_at="2026-08-15T00:00:01Z",
            )

    def test_decision_batch_client_builds_two_lease_issue_request(self) -> None:
        ctx = context()
        authority = CapturingAuthority(ctx)
        client = lease.SharedG2LeaseAuthorityClient(
            authority=authority,
            lease_material_transport=object(),
            lease_envelope_verifier=AcceptingSyntheticVerifier(),
            lease_authority_identity_digest=H["a"],
            phase_output_attester_identity_digest=H["b"],
        )
        approved = durable.RunLifecycleState(
            run_scope_digest=H["1"],
            generation=0,
            lifecycle_state="RND_APPROVED",
            lifecycle_sequence=2,
            predecessor_state_digest=H["7"],
            transition_evidence_digest=H["8"],
        )
        approved_head = durable.SubjectHead(
            subject_kind="RUN",
            subject_digest=H["1"],
            generation=0,
            sequence=2,
            head_digest=H["9"],
            state_digest=approved.digest,
        )
        clean_a = binding(phase="DECISION_FREEZE", replica_id="clean_a")
        clean_b = binding(phase="DECISION_FREEZE", replica_id="clean_b")
        with self.assertRaises(CommitProbeStop):
            client.issue_decision_lease_batch(
                bindings=(clean_b, clean_a),
                expected_global_head=global_head(
                    ctx,
                    sequence=2,
                    digest=H["2"],
                    observed_at="2026-08-15T00:00:02Z",
                ),
                operation_id="decision-batch-issue-001",
                idempotency_key="decision-batch-issue-001",
                requested_at="2026-08-15T00:00:03Z",
                run_lifecycle=lease.RunLifecycleView(
                    head=approved_head, state=approved
                ),
            )
        self.assertIsNotNone(authority.request)
        assert authority.request is not None
        self.assertEqual(
            Counter(
                item.subject_kind for item in authority.request.subject_mutations
            ),
            Counter({"LEASE": 2, "RUN": 1}),
        )
        self.assertEqual(
            [
                item["replica_id"]
                for item in authority.request.mutation_payload["bindings"]
            ],
            ["clean_a", "clean_b"],
        )

    def test_decision_batch_client_builds_one_atomic_irreversible_request(
        self,
    ) -> None:
        ctx = context()
        authority = CapturingAuthority(ctx)
        client = lease.SharedG2LeaseAuthorityClient(
            authority=authority,
            lease_material_transport=object(),
            lease_envelope_verifier=AcceptingSyntheticVerifier(),
            lease_authority_identity_digest=H["a"],
            phase_output_attester_identity_digest=H["b"],
        )
        clean_a = binding(phase="DECISION_FREEZE", replica_id="clean_a")
        clean_b = binding(phase="DECISION_FREEZE", replica_id="clean_b")
        issue_previous = global_head(
            ctx,
            sequence=1,
            digest=H["1"],
            observed_at="2026-08-15T00:00:01Z",
        )
        issue_new = global_head(
            ctx,
            sequence=2,
            digest=H["2"],
            observed_at="2026-08-15T00:00:02Z",
        )
        phase_leases = (
            lease.PhaseLease(
                lease_digest=H["7"],
                envelope_digest=H["9"],
                lease_id="decision-lease-clean-a-001",
                binding=clean_a,
                issued_from_global_head=issue_previous,
                issued_global_head=issue_new,
                issue_transaction_id="decision-batch-issue-transaction-001",
                lease_authority_identity_digest=H["a"],
                issued_at="2026-08-15T00:00:02Z",
                expires_at="2026-08-15T00:20:02Z",
            ),
            lease.PhaseLease(
                lease_digest=H["8"],
                envelope_digest=H["9"],
                lease_id="decision-lease-clean-b-001",
                binding=clean_b,
                issued_from_global_head=issue_previous,
                issued_global_head=issue_new,
                issue_transaction_id="decision-batch-issue-transaction-001",
                lease_authority_identity_digest=H["a"],
                issued_at="2026-08-15T00:00:02Z",
                expires_at="2026-08-15T00:20:02Z",
            ),
        )
        leased_state = durable.RunLifecycleState(
            run_scope_digest=H["1"],
            generation=0,
            lifecycle_state="RND_LEASED",
            lifecycle_sequence=3,
            predecessor_state_digest=H["7"],
            transition_evidence_digest=H["8"],
        )
        lease_heads = tuple(
            durable.SubjectHead(
                subject_kind="LEASE",
                subject_digest=item.binding.domain_digest,
                generation=0,
                sequence=1,
                head_digest=durable.canonical_digest(
                    {"lease_head": item.binding.replica_id}
                ),
                state_digest=durable.canonical_digest(
                    {"lease_state": item.binding.replica_id}
                ),
            )
            for item in phase_leases
        )
        issued_batch = lease.IssuedDecisionLeaseBatch(
            receipt=lease.DecisionLeaseBatchReceipt(
                payload_digest=H["9"],
                envelope_digest=H["a"],
                batch_domain_digest=lease.decision_lease_batch_domain_digest(
                    (clean_a, clean_b)
                ),
                leases=phase_leases,
                issued_from_global_head=issue_previous,
                issued_global_head=issue_new,
                issue_transaction_id="decision-batch-issue-transaction-001",
                issued_at="2026-08-15T00:00:02Z",
            ),
            transaction=SimpleNamespace(
                receipt=SimpleNamespace(new_subject_heads=lease_heads)
            ),
            resulting_run_state=leased_state,
        )
        run_head = durable.SubjectHead(
            subject_kind="RUN",
            subject_digest=H["1"],
            generation=0,
            sequence=3,
            head_digest=H["3"],
            state_digest=leased_state.digest,
        )
        semantic_state = durable.SingleUseSubjectState.provisional(
            subject_kind="SEMANTIC_SUBJECT",
            subject_digest=H["3"],
            generation=0,
            run_scope_digest=H["1"],
            approval_receipt_digest=H["7"],
        )
        exact_state = durable.SingleUseSubjectState.provisional(
            subject_kind="EXACT_SUBJECT",
            subject_digest=H["4"],
            generation=0,
            run_scope_digest=H["1"],
            approval_receipt_digest=H["7"],
        )
        family_state = durable.QuestionFamilyState.unused(
            question_family_digest=H["5"]
        )
        irreversible = lease.IrreversibleLifecycleView(
            run=lease.RunLifecycleView(head=run_head, state=leased_state),
            semantic_head=durable.SubjectHead(
                subject_kind="SEMANTIC_SUBJECT",
                subject_digest=H["3"],
                generation=0,
                sequence=1,
                head_digest=H["4"],
                state_digest=semantic_state.digest,
            ),
            semantic_state=semantic_state,
            exact_head=durable.SubjectHead(
                subject_kind="EXACT_SUBJECT",
                subject_digest=H["4"],
                generation=0,
                sequence=1,
                head_digest=H["5"],
                state_digest=exact_state.digest,
            ),
            exact_state=exact_state,
            question_family_head=durable.SubjectHead(
                subject_kind="QUESTION_FAMILY",
                subject_digest=H["5"],
                generation=0,
                sequence=0,
                head_digest=durable.ZERO_SHA256,
                state_digest=family_state.digest,
            ),
            question_family_state=family_state,
        )
        with patch.object(
            client,
            "_fetch_decision_lease_batch_envelope",
            return_value={},
        ), patch.object(
            lease,
            "validate_decision_lease_batch_receipt",
            return_value=issued_batch.receipt,
        ), self.assertRaises(CommitProbeStop):
            client.consume_decision_lease_batch(
                issued=issued_batch,
                expected_global_head=issue_new,
                dispatch_digests={"clean_a": H["1"], "clean_b": H["2"]},
                consume_revalidation_receipt_digest=H["6"],
                operation_id="decision-batch-consume-001",
                idempotency_key="decision-batch-consume-001",
                requested_at="2026-08-15T00:00:03Z",
                irreversible_lifecycle=irreversible,
            )
        self.assertIsNotNone(authority.request)
        assert authority.request is not None
        self.assertEqual(
            authority.request.operation_kind,
            "RND_DECISION_IRREVERSIBLE_START",
        )
        self.assertEqual(
            Counter(
                item.subject_kind for item in authority.request.subject_mutations
            ),
            Counter(
                {
                    "LEASE": 2,
                    "RUN": 1,
                    "SEMANTIC_SUBJECT": 1,
                    "EXACT_SUBJECT": 1,
                    "QUESTION_FAMILY": 1,
                }
            ),
        )
        self.assertEqual(
            [
                item["replica_id"]
                for item in authority.request.mutation_payload["leases"]
            ],
            ["clean_a", "clean_b"],
        )
        wrong_generation = lease.IrreversibleLifecycleView(
            run=irreversible.run,
            semantic_head=irreversible.semantic_head,
            semantic_state=irreversible.semantic_state,
            exact_head=irreversible.exact_head,
            exact_state=irreversible.exact_state,
            question_family_head=durable.SubjectHead(
                subject_kind="QUESTION_FAMILY",
                subject_digest=H["5"],
                generation=1,
                sequence=0,
                head_digest=durable.ZERO_SHA256,
                state_digest=family_state.digest,
            ),
            question_family_state=family_state,
        )
        with patch.object(
            client,
            "_fetch_decision_lease_batch_envelope",
            return_value={},
        ), patch.object(
            lease,
            "validate_decision_lease_batch_receipt",
            return_value=issued_batch.receipt,
        ), self.assertRaisesRegex(
            durable.SharedG2ValidationError,
            "irreversible lifecycle bundle",
        ):
            client.consume_decision_lease_batch(
                issued=issued_batch,
                expected_global_head=issue_new,
                dispatch_digests={"clean_a": H["1"], "clean_b": H["2"]},
                consume_revalidation_receipt_digest=H["6"],
                operation_id="decision-batch-wrong-generation-001",
                idempotency_key="decision-batch-wrong-generation-001",
                requested_at="2026-08-15T00:00:03Z",
                irreversible_lifecycle=wrong_generation,
            )

    def test_singular_decision_lease_paths_are_rejected(self) -> None:
        ctx = context()
        decision = binding(phase="DECISION_FREEZE", replica_id="clean_a")
        expected_head = global_head(
            ctx,
            sequence=1,
            digest=H["1"],
            observed_at="2026-08-15T00:00:01Z",
        )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "issued together"
        ):
            lease.SharedG2LeaseAuthorityClient.issue_phase_lease(
                object(),
                binding=decision,
                expected_global_head=expected_head,
                operation_id="forbidden-single-decision-issue",
                idempotency_key="forbidden-single-decision-issue",
                requested_at="2026-08-15T00:00:01Z",
                predecessor_output_seals=(),
            )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "consumed together"
        ):
            lease.SharedG2LeaseAuthorityClient.consume_phase_lease(
                object(),
                issued=SimpleNamespace(
                    lease=SimpleNamespace(binding=decision)
                ),
                expected_global_head=expected_head,
                dispatch_digest=H["2"],
                consume_revalidation_receipt_digest=H["3"],
                operation_id="forbidden-single-decision-consume",
                idempotency_key="forbidden-single-decision-consume",
                requested_at="2026-08-15T00:00:01Z",
            )

    def test_phase_specific_predecessor_topology_and_replay_rejection(self) -> None:
        ctx = context()
        decision_a = synthetic_sealed_output(
            ctx,
            phase="DECISION_FREEZE",
            replica_id="clean_a",
            receipt_payload_digest=H["7"],
            output_digest=H["1"],
        )
        decision_b = synthetic_sealed_output(
            ctx,
            phase="DECISION_FREEZE",
            replica_id="clean_b",
            receipt_payload_digest=H["8"],
            output_digest=H["2"],
        )
        settlement_successor = binding(
            phase="SETTLEMENT_DIAGNOSTIC",
            replica_id="clean_a",
            predecessor_receipt_digest=decision_a.receipt.payload_digest,
        )
        self.assertEqual(
            lease.canonical_predecessor_output_digest(
                successor_binding=settlement_successor,
                predecessor_output_seals=(decision_a,),
            ),
            decision_a.receipt.payload_digest,
        )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "same-replica"
        ):
            lease.canonical_predecessor_output_digest(
                successor_binding=settlement_successor,
                predecessor_output_seals=(decision_b,),
            )

        settlement_a = synthetic_sealed_output(
            ctx,
            phase="SETTLEMENT_DIAGNOSTIC",
            replica_id="clean_a",
            receipt_payload_digest=H["7"],
            output_digest=H["3"],
        )
        settlement_b = synthetic_sealed_output(
            ctx,
            phase="SETTLEMENT_DIAGNOSTIC",
            replica_id="clean_b",
            receipt_payload_digest=H["8"],
            output_digest=H["4"],
        )
        compare_successor = binding(
            phase="REPLICA_COMPARE",
            replica_id=lease.REPLICA_COMPARE_ACTOR,
        )
        compare_projection = lease.canonical_predecessor_output_digest(
            successor_binding=compare_successor,
            predecessor_output_seals=(settlement_b, settlement_a),
        )
        self.assertEqual(
            compare_projection,
            lease.canonical_predecessor_output_digest(
                successor_binding=binding(
                    phase="REPLICA_COMPARE",
                    replica_id=lease.REPLICA_COMPARE_ACTOR,
                    predecessor_receipt_digest=compare_projection,
                ),
                predecessor_output_seals=(settlement_a, settlement_b),
            ),
        )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "exactly two settlement seals"
        ):
            lease.canonical_predecessor_output_digest(
                successor_binding=compare_successor,
                predecessor_output_seals=(settlement_a,),
            )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "distinct clean_a/clean_b"
        ):
            replayed_b = synthetic_sealed_output(
                ctx,
                phase="SETTLEMENT_DIAGNOSTIC",
                replica_id="clean_b",
                receipt_payload_digest=settlement_a.receipt.payload_digest,
                output_digest=H["4"],
            )
            lease.canonical_predecessor_output_digest(
                successor_binding=compare_successor,
                predecessor_output_seals=(settlement_a, replayed_b),
            )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "distinct clean_a/clean_b"
        ):
            lease.canonical_predecessor_output_digest(
                successor_binding=compare_successor,
                predecessor_output_seals=(settlement_a, decision_b),
            )

        compare_seal = synthetic_sealed_output(
            ctx,
            phase="REPLICA_COMPARE",
            replica_id=lease.REPLICA_COMPARE_ACTOR,
            receipt_payload_digest=H["9"],
            output_digest=compare_projection,
        )
        result_successor = binding(
            phase="RESULT_SEAL",
            replica_id=lease.RESULT_SEAL_ACTOR,
            predecessor_receipt_digest=compare_seal.receipt.payload_digest,
        )
        self.assertEqual(
            lease.canonical_predecessor_output_digest(
                successor_binding=result_successor,
                predecessor_output_seals=(compare_seal,),
            ),
            compare_seal.receipt.payload_digest,
        )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "lane_coordinator"
        ):
            lease.canonical_predecessor_output_digest(
                successor_binding=result_successor,
                predecessor_output_seals=(settlement_a,),
            )

    def test_phase_capability_union_or_market_drift_is_rejected(self) -> None:
        wrong_digest = lease.PHASE_CAPABILITY_DIGESTS["DECISION_FREEZE"]
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "finite phase profile"
        ):
            binding(
                phase="SETTLEMENT_DIAGNOSTIC",
                phase_capability_digest=wrong_digest,
            )
        for profile in lease.PHASE_CAPABILITY_PROJECTIONS.values():
            self.assertFalse(profile["odds_price_popularity_or_market_access"])
            self.assertFalse(profile["network_access"])
            self.assertFalse(profile["credential_access"])
            self.assertFalse(profile["formal_buy"])
            self.assertFalse(profile["send_order"])
            self.assertEqual(profile["stake"], 0)

    def test_trusted_output_attestation_rejects_cross_replica_domain(self) -> None:
        ctx = context()
        consumed = synthetic_consumed_phase(ctx)
        bound = consumed.lease.binding
        payload: dict[str, object] = {
            "schema_version": 1,
            "attestation_kind": lease.PHASE_OUTPUT_ATTESTATION_KIND,
            "authority_id": ctx.expectations.authority_id,
            "activation_epoch": ctx.expectations.activation_epoch,
            "cutover_receipt_digest": ctx.cutover_receipt_digest,
            "run_scope_digest": bound.run_scope_digest,
            "recipe_digest": bound.recipe_digest,
            "replica_id": bound.replica_id,
            "phase": bound.phase,
            "attempt": bound.attempt,
            "binding_digest": bound.digest,
            "lease_payload_digest": consumed.lease.lease_digest,
            "lease_consumption_receipt_digest": consumed.receipt.payload_digest,
            "consumed_global_head": (
                consumed.transaction.receipt.new_global_head.to_wire()
            ),
            "output_digest": H["4"],
            "policy_digest": bound.policy_digest,
            "schema_digest": bound.schema_digest,
            "verifier_digest": bound.verifier_digest,
            "executor_digest": bound.executor_digest,
            "runner_digest": bound.runner_digest,
            "bound_capability_profile_digest": (
                bound.bound_capability_profile_digest
            ),
            "phase_capability_digest": bound.phase_capability_digest,
            "attester_identity_digest": H["5"],
            "attested_at": "2026-08-15T00:00:04Z",
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
        validated = lease.validate_trusted_phase_output_attestation(
            envelope(
                lease.PHASE_OUTPUT_ATTESTATION_KIND,
                payload,
                attestation_digest=H["5"],
            ),
            consumed=consumed,
            context=ctx,
            expected_attester_identity_digest=H["5"],
            verifier=AcceptingSyntheticVerifier(),
        )
        self.assertEqual(validated.output_digest, H["4"])

        cross_replica = dict(payload)
        cross_replica["replica_id"] = "clean_b"
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "exact execution binding mismatch"
        ):
            lease.validate_trusted_phase_output_attestation(
                envelope(
                    lease.PHASE_OUTPUT_ATTESTATION_KIND,
                    cross_replica,
                    attestation_digest=H["5"],
                ),
                consumed=consumed,
                context=ctx,
                expected_attester_identity_digest=H["5"],
                verifier=AcceptingSyntheticVerifier(),
            )

    def test_replayed_lease_issue_receipt_fails_subject_cas(self) -> None:
        ctx = context()
        bound = binding()
        mutation = lease._issue_lease_mutation(bound)
        previous_global = global_head(
            ctx,
            sequence=1,
            digest=H["1"],
            observed_at="2026-08-15T00:00:01Z",
        )
        request = durable.TransactionRequest.build(
            operation_kind="RND_PHASE_LEASE_ISSUE",
            operation_id="lease-issue-001",
            context=ctx,
            idempotency_key="lease-issue-001",
            run_scope_digest=bound.run_scope_digest,
            mutation_payload={
                "schema_version": 1,
                "action": "ISSUE_ONE_SHOT_PHASE_LEASE",
                "operation_kind": "RND_PHASE_LEASE_ISSUE",
                "safety": {"formal_buy": False, "send_order": False, "stake": 0},
            },
            expected_output_type=lease.PHASE_LEASE_KIND,
            expected_global_head=previous_global,
            subject_mutations=[mutation],
            requested_at="2026-08-15T00:00:01Z",
        )
        next_global = global_head(
            ctx,
            sequence=2,
            digest=H["2"],
            observed_at="2026-08-15T00:00:02Z",
        )
        replayed_previous_subject = durable.SubjectHead(
            subject_kind="LEASE",
            subject_digest=bound.domain_digest,
            generation=0,
            sequence=1,
            head_digest=H["3"],
            state_digest=mutation.new_state_digest,
        )
        replayed_new_subject = durable.SubjectHead(
            subject_kind="LEASE",
            subject_digest=bound.domain_digest,
            generation=0,
            sequence=2,
            head_digest=H["4"],
            state_digest=H["5"],
        )
        receipt_payload = {
            "schema_version": 1,
            "receipt_kind": durable.TRANSACTION_RECEIPT_KIND,
            "authority_id": ctx.expectations.authority_id,
            "activation_epoch": ctx.expectations.activation_epoch,
            "backend_identity_digest": ctx.expectations.backend_identity_digest,
            "cutover_receipt_digest": ctx.cutover_receipt_digest,
            "transaction_id": "transaction-lease-replay-001",
            "idempotency_key": request.idempotency_key,
            "operation_kind": request.operation_kind,
            "operation_id": request.operation_id,
            "request_digest": request.digest,
            "run_scope_digest": request.run_scope_digest,
            "mutation_digest": request.mutation_digest,
            "previous_global_head": previous_global.to_wire(),
            "new_global_head": next_global.to_wire(),
            "previous_subject_heads": [replayed_previous_subject.to_wire()],
            "new_subject_heads": [replayed_new_subject.to_wire()],
            "operation_output_type": lease.PHASE_LEASE_KIND,
            "operation_output_digest": H["6"],
            "writer_identity_digest": H["7"],
            "committed_at": "2026-08-15T00:00:02Z",
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
        with self.assertRaisesRegex(
            durable.SharedG2StaleAuthority, "subject CAS mismatch"
        ):
            durable.validate_transaction_receipt(
                envelope(
                    durable.TRANSACTION_RECEIPT_KIND,
                    receipt_payload,
                    attestation_digest=ctx.expectations.backend_identity_digest,
                ),
                request=request,
                context=ctx,
                verifier=AcceptingSyntheticVerifier(),
            )

    def test_replayed_phase_output_seal_fails_subject_cas(self) -> None:
        ctx = context()
        previous_global = global_head(
            ctx,
            sequence=6,
            digest=H["1"],
            observed_at="2026-08-15T00:00:06Z",
        )
        output_subject_digest = durable.canonical_digest(
            {"phase": "REPLICA_COMPARE", "replica": "clean_a"}
        )
        mutation = durable.SubjectMutation(
            subject_kind="PHASE_OUTPUT",
            subject_digest=output_subject_digest,
            generation=0,
            expected_sequence=0,
            expected_head_digest=durable.ZERO_SHA256,
            new_state_digest=H["2"],
        )
        request = durable.TransactionRequest.build(
            operation_kind="RND_PHASE_OUTPUT_SEAL",
            operation_id="phase-output-seal-001",
            context=ctx,
            idempotency_key="phase-output-seal-001",
            run_scope_digest=H["3"],
            mutation_payload={
                "schema_version": 1,
                "action": "SEAL_TRUSTED_PHASE_OUTPUT",
                "operation_kind": "RND_PHASE_OUTPUT_SEAL",
                "safety": {"formal_buy": False, "send_order": False, "stake": 0},
            },
            expected_output_type=lease.PHASE_OUTPUT_SEAL_RECEIPT_KIND,
            expected_global_head=previous_global,
            subject_mutations=[mutation],
            requested_at="2026-08-15T00:00:06Z",
        )
        next_global = global_head(
            ctx,
            sequence=7,
            digest=H["5"],
            observed_at="2026-08-15T00:00:07Z",
        )
        replayed_previous = durable.SubjectHead(
            subject_kind="PHASE_OUTPUT",
            subject_digest=output_subject_digest,
            generation=0,
            sequence=1,
            head_digest=H["6"],
            state_digest=H["2"],
        )
        replayed_new = durable.SubjectHead(
            subject_kind="PHASE_OUTPUT",
            subject_digest=output_subject_digest,
            generation=0,
            sequence=2,
            head_digest=H["7"],
            state_digest=H["8"],
        )
        payload = {
            "schema_version": 1,
            "receipt_kind": durable.TRANSACTION_RECEIPT_KIND,
            "authority_id": ctx.expectations.authority_id,
            "activation_epoch": ctx.expectations.activation_epoch,
            "backend_identity_digest": ctx.expectations.backend_identity_digest,
            "cutover_receipt_digest": ctx.cutover_receipt_digest,
            "transaction_id": "transaction-output-replay-001",
            "idempotency_key": request.idempotency_key,
            "operation_kind": request.operation_kind,
            "operation_id": request.operation_id,
            "request_digest": request.digest,
            "run_scope_digest": request.run_scope_digest,
            "mutation_digest": request.mutation_digest,
            "previous_global_head": previous_global.to_wire(),
            "new_global_head": next_global.to_wire(),
            "previous_subject_heads": [replayed_previous.to_wire()],
            "new_subject_heads": [replayed_new.to_wire()],
            "operation_output_type": lease.PHASE_OUTPUT_SEAL_RECEIPT_KIND,
            "operation_output_digest": H["9"],
            "writer_identity_digest": H["a"],
            "committed_at": "2026-08-15T00:00:07Z",
            "safety": {"formal_buy": False, "send_order": False, "stake": 0},
        }
        with self.assertRaisesRegex(
            durable.SharedG2StaleAuthority, "subject CAS mismatch"
        ):
            durable.validate_transaction_receipt(
                envelope(
                    durable.TRANSACTION_RECEIPT_KIND,
                    payload,
                    attestation_digest=ctx.expectations.backend_identity_digest,
                ),
                request=request,
                context=ctx,
                verifier=AcceptingSyntheticVerifier(),
            )

    def test_bounded_facade_has_no_free_form_lifecycle_or_output_digest(self) -> None:
        issue_parameters = inspect.signature(
            lease.SharedG2LeaseAuthorityClient.issue_phase_lease
        ).parameters
        consume_parameters = inspect.signature(
            lease.SharedG2LeaseAuthorityClient.consume_phase_lease
        ).parameters
        decision_consume_parameters = inspect.signature(
            lease.SharedG2LeaseAuthorityClient.consume_decision_lease_batch
        ).parameters
        seal_parameters = inspect.signature(
            lease.SharedG2LeaseAuthorityClient.seal_phase_output
        ).parameters
        self.assertNotIn("run_transition_mutation", issue_parameters)
        self.assertNotIn("lifecycle_mutations", consume_parameters)
        self.assertNotIn("irreversible_lifecycle", consume_parameters)
        self.assertIn(
            "irreversible_lifecycle", decision_consume_parameters
        )
        self.assertNotIn("output_digest", seal_parameters)
        self.assertIn("output_attestation_payload_digest", seal_parameters)
        self.assertEqual(
            lease.CONSUME_OPERATION_BY_PHASE["RESULT_SEAL"],
            "RND_PHASE_LEASE_CONSUME",
        )
        self.assertEqual(
            lease.OUTPUT_SEAL_OPERATION_BY_PHASE["RESULT_SEAL"],
            "RND_RESULT_SEAL",
        )
        self.assertEqual(
            durable.OPERATION_REQUIRED_SUBJECT_KINDS["RND_RESULT_SEAL"],
            frozenset({"PHASE_OUTPUT", "RUN"}),
        )

    def test_lifecycle_and_single_use_replay_are_terminal(self) -> None:
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "non-canonical lifecycle sequence"
        ):
            durable.RunLifecycleState(
                run_scope_digest=H["1"],
                generation=0,
                lifecycle_state="RND_APPROVED",
                lifecycle_sequence=4,
                predecessor_state_digest=H["2"],
                transition_evidence_digest=H["3"],
            )
        initial = durable.RunLifecycleState.initial(
            run_scope_digest=H["1"], generation=0
        )
        head = durable.SubjectHead(
            subject_kind="RUN",
            subject_digest=H["1"],
            generation=0,
            sequence=0,
            head_digest=durable.ZERO_SHA256,
            state_digest=initial.digest,
        )
        transition = durable.build_run_lifecycle_transition(
            current_head=head,
            current_state=initial,
            new_state="RND_RUN_APPROVAL_REQUIRED",
            transition_evidence_digest=H["2"],
        )
        next_head = durable.SubjectHead(
            subject_kind="RUN",
            subject_digest=H["1"],
            generation=0,
            sequence=1,
            head_digest=H["3"],
            state_digest=transition.current.digest,
        )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "illegal run lifecycle transition"
        ):
            durable.build_run_lifecycle_transition(
                current_head=next_head,
                current_state=transition.current,
                new_state="RND_RUNNING",
                transition_evidence_digest=H["4"],
            )

        provisional = durable.SingleUseSubjectState.provisional(
            subject_kind="SEMANTIC_SUBJECT",
            subject_digest=H["4"],
            generation=0,
            run_scope_digest=H["1"],
            approval_receipt_digest=H["5"],
        )
        subject_head = durable.SubjectHead(
            subject_kind="SEMANTIC_SUBJECT",
            subject_digest=H["4"],
            generation=0,
            sequence=1,
            head_digest=H["6"],
            state_digest=provisional.digest,
        )
        consumed = durable.build_single_use_subject_transition(
            current_head=subject_head,
            current_state=provisional,
            new_state="IRREVERSIBLY_CONSUMED",
            transition_evidence_digest=H["7"],
        )
        consumed_head = durable.SubjectHead(
            subject_kind="SEMANTIC_SUBJECT",
            subject_digest=H["4"],
            generation=0,
            sequence=2,
            head_digest=H["8"],
            state_digest=consumed.current.digest,
        )
        with self.assertRaisesRegex(
            durable.SharedG2ValidationError, "illegal single-use transition"
        ):
            durable.build_single_use_subject_transition(
                current_head=consumed_head,
                current_state=consumed.current,
                new_state="IRREVERSIBLY_CONSUMED",
                transition_evidence_digest=H["9"],
            )


if __name__ == "__main__":
    unittest.main()
