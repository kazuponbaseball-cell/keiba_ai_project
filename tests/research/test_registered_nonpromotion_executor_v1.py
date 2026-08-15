from __future__ import annotations

import copy
import sys
import unittest
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import registered_nonpromotion_contract_v1 as contract
import registered_nonpromotion_supervised_executor_v1 as executor
import shared_g2_durable_ledger_v1 as durable
import shared_g2_lease_authority_v1 as g2


class RegisteredExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registered = contract.resolve_registered_recipe(
            ROOT,
            recipe_id="historical_ai_duplicate_gate_impact_v1",
            recipe_version=1,
        )

    def candidate_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        fold_sequence = ["fold2"] * 1661 + ["fold3"] * 1653 + ["fold4"] * 432
        for index, fold in enumerate(fold_sequence):
            p = 0.34 if index == 0 else 0.25
            a = 0.37 if index == 0 else 0.27
            rows.append(
                {
                    "candidate_generated": True,
                    "candidate_key": f"{index % 18 + 1}-{(index + 1) % 18 + 1}",
                    "eligible_race": True,
                    "fold": fold,
                    "horse_a": index % 18 + 1,
                    "horse_b": (index + 1) % 18 + 1,
                    "p_action_C0_offset": a,
                    "race_date": f"2026-01-{index % 28 + 1:02d}",
                    "race_id": f"{2025000000000000 + index:016d}",
                    "top1_wide_prob": p,
                    "venue_code": f"{index % 10 + 1:02d}",
                }
            )
        return rows

    def settlement_rows(
        self, candidates: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        return [
            {
                "race_id": row["race_id"],
                "candidate_key": row["candidate_key"],
                "candidate_hit": False,
                "official_outcome_completeness": True,
                "official_wide_pay": None,
            }
            for row in candidates
        ]

    def run_scope(self) -> dict[str, object]:
        bindings = {
            "repository": contract.DEFAULT_REPOSITORY,
            "base_branch": contract.DEFAULT_BASE_BRANCH,
            "run_scope_base_commit": "a" * 40,
            "verified_current_main_sha": "b" * 40,
            "approvers_blob_sha": "c" * 40,
            "approvers_content_sha256": "d" * 64,
            "activation_receipt_sha256": "1" * 64,
            "cutover_receipt_sha256": "1" * 64,
            "schema_bundle_sha256": self.registered.schema_bundle_digest,
            "approval_evidence_schema_sha256": "3" * 64,
            "capability_profile_sha256": "4" * 64,
            "policy_blob_sha256": self.registered.policy_file_sha256,
            "recipe_blob_sha256": self.registered.recipe_file_sha256,
            "compiler_blob_sha256": "5" * 64,
            "authority_verifier_blob_sha256": "6" * 64,
            "catalog_validator_blob_sha256": "7" * 64,
            "executor_blob_sha256": "8" * 64,
            "runner_blob_sha256": "9" * 64,
            "result_sealer_blob_sha256": "a" * 64,
            "g2_authority_service_blob_sha256": "b" * 64,
            "phase_lease_schema_sha256": "c" * 64,
            "phase_operation_receipt_schema_sha256": "d" * 64,
            "environment_manifest_sha256": "e" * 64,
            "catalog_release_id": "catalog_release_v1",
            "catalog_release_sha256": "7" * 64,
            "catalog_release_status": "ACTIVE",
            "catalog_release_revoked": False,
            "catalog_status_receipt_sha256": "8" * 64,
            "candidate_entry_sha256": "8" * 64,
            "candidate_schema_sha256": "9" * 64,
            "candidate_provenance_sha256": "a" * 64,
            "p_action_cross_source_equality_attestation_sha256": "b" * 64,
            "candidate_materializer_usecols_sha256": "c" * 64,
            "decision_base_lineage_sha256": "d" * 64,
            "settlement_entry_sha256": "9" * 64,
            "settlement_schema_sha256": "e" * 64,
            "settlement_provenance_sha256": "f" * 64,
            "official_settlement_provenance_sha256": "1" * 64,
            "cohort_manifest_sha256": "a" * 64,
            "ordered_race_set_sha256": "b" * 64,
            "output_root": "outputs/research/RND-001",
            "sealed_at": "2026-08-15T00:00:00Z",
            "expected_pregrant_global_head": "c" * 64,
            "expected_pregrant_subject_head": "d" * 64,
            "cutover_epoch": 1,
            "external_witness_checkpoint_sha256": "f" * 64,
        }
        bindings.update(self.registered.runtime_material_digests)
        return contract.compile_run_scope(self.registered, bindings)

    def settlement_result_pair(
        self,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        candidates = self.candidate_rows()
        run = self.run_scope()
        results: dict[str, dict[str, object]] = {}
        for replica_id, operation_digest in (("clean_a", "e" * 64), ("clean_b", "f" * 64)):
            decisions, receipt = executor._freeze_decisions_after_authenticated_mount(
                self.registered,
                candidates,
                run_scope_digest=str(run["run_scope_digest"]),
                replica_id=replica_id,
                irreversible_receipt_digest="d" * 64,
            )
            results[replica_id] = executor._settle_diagnostic_after_authenticated_mount(
                self.registered,
                decisions,
                self.settlement_rows(candidates),
                run_scope_digest=str(run["run_scope_digest"]),
                replica_id=replica_id,
                decision_freeze_receipt=receipt,
                settlement_operation_receipt_digest=operation_digest,
                bootstrap_replicates_override_for_synthetic_test=1,
            )
        return run, results["clean_a"], results["clean_b"]

    def head(self) -> durable.GlobalHead:
        return durable.GlobalHead(
            authority_id="shared-g2",
            activation_epoch="epoch-v1",
            backend_identity_digest="1" * 64,
            cutover_receipt_digest="2" * 64,
            sequence=10,
            head_digest="3" * 64,
            observed_at="2026-08-15T00:00:00Z",
        )

    def binding(
        self,
        *,
        run: dict[str, object],
        phase: str,
        replica_id: str,
        predecessor_digest: str,
    ) -> g2.LeaseBinding:
        return g2.LeaseBinding(
            run_scope_digest=str(run["run_scope_digest"]),
            run_generation=0,
            recipe_digest=self.registered.recipe_digest,
            semantic_subject_digest="4" * 64,
            exact_run_subject_digest="5" * 64,
            question_family_digest="6" * 64,
            replica_id=replica_id,
            phase=phase,
            attempt=1,
            predecessor_receipt_digest=predecessor_digest,
            issue_revalidation_receipt_digest="7" * 64,
            policy_digest="8" * 64,
            schema_digest="9" * 64,
            verifier_digest="a" * 64,
            executor_digest="b" * 64,
            runner_digest="c" * 64,
            bound_capability_profile_digest="d" * 64,
            phase_capability_digest=g2.PHASE_CAPABILITY_DIGESTS[phase],
            ttl_seconds=60,
        )

    def phase_lease(
        self, binding: g2.LeaseBinding, *, marker: str
    ) -> g2.PhaseLease:
        head = self.head()
        return g2.PhaseLease(
            lease_digest=marker * 64,
            envelope_digest="e" * 64,
            lease_id=f"lease-{marker}",
            binding=binding,
            issued_from_global_head=head,
            issued_global_head=head,
            issue_transaction_id=f"tx-{marker}",
            lease_authority_identity_digest="f" * 64,
            issued_at="2026-08-15T00:00:00Z",
            expires_at="2026-08-15T00:01:00Z",
        )

    def consumed_phase(
        self, lease: g2.PhaseLease, *, marker: str
    ) -> g2.ConsumedPhaseLease:
        head = self.head()
        receipt = g2.LeaseConsumptionReceipt(
            payload_digest=marker * 64,
            envelope_digest="e" * 64,
            lease_id=lease.lease_id,
            lease_payload_digest=lease.lease_digest,
            binding_digest=lease.binding.digest,
            operation_kind="RND_PHASE_LEASE_CONSUME",
            dispatch_digest="a" * 64,
            consume_revalidation_receipt_digest="b" * 64,
            consumed_from_global_head=head,
            consumed_global_head=head,
            consume_transaction_id=f"consume-{marker}",
            consumed_at="2026-08-15T00:00:01Z",
        )
        return g2.ConsumedPhaseLease(
            lease=lease,
            receipt=receipt,
            issue_transaction=None,  # type: ignore[arg-type]
            transaction=None,  # type: ignore[arg-type]
            resulting_run_state=None,
        )

    def settlement_seal(
        self,
        *,
        run: dict[str, object],
        result: dict[str, object],
        replica_id: str,
        marker: str,
    ) -> g2.SealedPhaseOutput:
        binding = self.binding(
            run=run,
            phase="SETTLEMENT_DIAGNOSTIC",
            replica_id=replica_id,
            predecessor_digest="1" * 64,
        )
        consumed = self.consumed_phase(self.phase_lease(binding, marker=marker), marker=marker)
        # Bind the synthetic result to the exact settlement consumption.
        result["settlement_operation_receipt_digest"] = consumed.receipt.payload_digest
        unsigned = dict(result)
        unsigned.pop("result_digest", None)
        result["result_digest"] = contract.canonical_digest(unsigned)
        output_digest = str(result["result_digest"])
        attestation = g2.TrustedPhaseOutputAttestation(
            payload_digest=("a" if marker != "a" else "b") * 64,
            envelope_digest="c" * 64,
            binding_digest=binding.digest,
            lease_payload_digest=consumed.lease.lease_digest,
            lease_consumption_receipt_digest=consumed.receipt.payload_digest,
            consumed_global_head=self.head(),
            output_digest=output_digest,
            attester_identity_digest="d" * 64,
            attested_at="2026-08-15T00:00:02Z",
        )
        receipt = g2.PhaseOutputSealReceipt(
            payload_digest=("b" if marker != "b" else "c") * 64,
            envelope_digest="d" * 64,
            run_scope_digest=str(run["run_scope_digest"]),
            recipe_digest=self.registered.recipe_digest,
            replica_id=replica_id,
            phase="SETTLEMENT_DIAGNOSTIC",
            attempt=1,
            binding_digest=binding.digest,
            phase_output_subject_digest="e" * 64,
            lease_payload_digest=consumed.lease.lease_digest,
            lease_consumption_receipt_digest=consumed.receipt.payload_digest,
            output_attestation_digest=attestation.payload_digest,
            output_digest=output_digest,
            operation_kind="RND_PHASE_OUTPUT_SEAL",
            sealed_from_global_head=self.head(),
            sealed_global_head=self.head(),
            seal_transaction_id=f"seal-{marker}",
            sealed_at="2026-08-15T00:00:03Z",
        )
        return g2.SealedPhaseOutput(
            consumed=consumed,
            attestation=attestation,
            receipt=receipt,
            transaction=None,  # type: ignore[arg-type]
            resulting_run_state=None,
        )

    def compare_chain(
        self,
        *,
        run: dict[str, object],
        seals: tuple[g2.SealedPhaseOutput, g2.SealedPhaseOutput],
    ) -> tuple[
        g2.IssuedPhaseLease,
        g2.ConsumedPhaseLease,
        tuple[g2.RevalidatedPhaseOutputSeal, g2.RevalidatedPhaseOutputSeal],
    ]:
        provisional = self.binding(
            run=run,
            phase="REPLICA_COMPARE",
            replica_id=g2.REPLICA_COMPARE_ACTOR,
            predecessor_digest="f" * 64,
        )
        predecessor = g2.canonical_predecessor_output_digest(
            successor_binding=provisional,
            predecessor_output_seals=seals,
        )
        binding = replace(provisional, predecessor_receipt_digest=predecessor)
        lease = self.phase_lease(binding, marker="d")
        issued = g2.IssuedPhaseLease(
            lease=lease,
            transaction=None,  # type: ignore[arg-type]
            resulting_run_state=None,
        )
        consumed = self.consumed_phase(lease, marker="e")
        evidence = tuple(
            g2.RevalidatedPhaseOutputSeal(
                receipt=seal.receipt,
                subject_head=None,  # type: ignore[arg-type]
                subject_snapshot=None,  # type: ignore[arg-type]
                authority_snapshot=None,  # type: ignore[arg-type]
            )
            for seal in seals
        )
        return issued, consumed, evidence  # type: ignore[return-value]

    def compare_executor(
        self,
        *,
        run: dict[str, object],
        authority: g2.SharedG2LeaseAuthorityClient,
    ) -> executor.SupervisedDiagnosticExecutor:
        instance = object.__new__(executor.SupervisedDiagnosticExecutor)
        object.__setattr__(instance, "registered", self.registered)
        object.__setattr__(instance, "run_scope", run)
        object.__setattr__(instance, "authority", authority)
        object.__setattr__(instance, "content", None)
        return instance

    def consumed_decision_batch(
        self, *, run: dict[str, object]
    ) -> g2.ConsumedDecisionLeaseBatch:
        transaction = object()
        issued = g2.IssuedDecisionLeaseBatch(
            receipt=None,  # type: ignore[arg-type]
            transaction=transaction,  # type: ignore[arg-type]
            resulting_run_state=None,  # type: ignore[arg-type]
        )
        batch_receipt = SimpleNamespace(payload_digest="c" * 64)
        replica_consumptions: list[g2.ConsumedPhaseLease] = []
        for replica_id, marker in (("clean_a", "a"), ("clean_b", "b")):
            binding = self.binding(
                run=run,
                phase="DECISION_FREEZE",
                replica_id=replica_id,
                predecessor_digest="1" * 64,
            )
            consumed = self.consumed_phase(
                self.phase_lease(binding, marker=marker), marker=marker
            )
            replica_consumptions.append(
                replace(
                    consumed,
                    transaction=transaction,  # type: ignore[arg-type]
                    decision_issued_batch=issued,
                    decision_consumption_batch=batch_receipt,  # type: ignore[arg-type]
                )
            )
        return g2.ConsumedDecisionLeaseBatch(
            issued=issued,
            receipt=batch_receipt,  # type: ignore[arg-type]
            transaction=transaction,  # type: ignore[arg-type]
            resulting_run_state=None,  # type: ignore[arg-type]
            replica_consumptions=(replica_consumptions[0], replica_consumptions[1]),
        )

    def test_bounded_d0_d1_diagnostic(self) -> None:
        candidates = self.candidate_rows()
        run = self.run_scope()
        decisions, receipt = executor._freeze_decisions_after_authenticated_mount(
            self.registered,
            candidates,
            run_scope_digest=run["run_scope_digest"],
            replica_id="clean_a",
            irreversible_receipt_digest="e" * 64,
        )
        self.assertEqual(len(decisions), 3746)
        self.assertEqual(
            receipt["decision_rows_digest"],
            contract.canonical_digest(decisions),
        )
        self.assertEqual(
            len(executor.decision_output_projection_digest(decisions, receipt)),
            64,
        )
        self.assertFalse(decisions[0]["d0_eligible"])
        self.assertTrue(decisions[0]["d1_eligible"])
        result = executor._settle_diagnostic_after_authenticated_mount(
            self.registered,
            decisions,
            self.settlement_rows(candidates),
            run_scope_digest=run["run_scope_digest"],
            replica_id="clean_a",
            decision_freeze_receipt=receipt,
            settlement_operation_receipt_digest="f" * 64,
            bootstrap_replicates_override_for_synthetic_test=8,
        )
        primary = result["scientific_projection"]["primary"]
        self.assertEqual(primary["decision_disagreement_count"], 1)
        self.assertEqual(primary["sum_delta_profit_yen"], -100.0)
        self.assertEqual(result["computed_outcome"], "DIRECTIONAL_EFFECT")
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(result["stake"], 0)

    def test_decision_batch_consume_and_single_replica_execution_are_separate(self) -> None:
        class CountingContent:
            def __init__(self, rows: list[dict[str, object]]) -> None:
                self.rows = rows
                self.replica_reads: list[str] = []

            def read_candidate_rows(self, *, replica_id: str, **_: object) -> list[dict[str, object]]:
                self.replica_reads.append(replica_id)
                return self.rows

        run = self.run_scope()
        batch = self.consumed_decision_batch(run=run)
        authority = object.__new__(g2.SharedG2LeaseAuthorityClient)
        subject = self.compare_executor(run=run, authority=authority)
        content = CountingContent(self.candidate_rows())
        object.__setattr__(subject, "content", content)
        with mock.patch.object(
            g2.SharedG2LeaseAuthorityClient,
            "consume_decision_lease_batch",
            return_value=batch,
        ):
            observed = subject.consume_decision_freeze_batch(
                issued_batch=batch.issued,
                expected_global_head=self.head(),
                dispatch_digests={"clean_a": "1" * 64, "clean_b": "2" * 64},
                consume_revalidation_receipt_digest="3" * 64,
                operation_id="decision-batch",
                idempotency_key="decision-batch-idempotency",
                requested_at="2026-08-15T00:00:00Z",
                irreversible_lifecycle=None,  # type: ignore[arg-type]
            )
        self.assertIs(observed, batch)
        self.assertEqual(content.replica_reads, [])
        decisions, receipt = subject.decision_freeze_replica(
            replica_id="clean_a", consumed_batch=batch
        )
        self.assertEqual(len(decisions), 3746)
        self.assertEqual(receipt["replica_id"], "clean_a")
        self.assertEqual(content.replica_reads, ["clean_a"])
        duplicated = replace(
            batch,
            replica_consumptions=(
                batch.replica_consumptions[0],
                batch.replica_consumptions[0],
            ),
        )
        with self.assertRaisesRegex(contract.ContractError, "exactly once"):
            subject.decision_freeze_replica(
                replica_id="clean_a", consumed_batch=duplicated
            )
        self.assertEqual(content.replica_reads, ["clean_a"])
        self.assertFalse(hasattr(subject, "decision_freeze_batch"))

    def test_market_column_is_rejected_in_every_phase(self) -> None:
        candidates = self.candidate_rows()
        candidates[0]["wide_odds_low"] = 3.2
        with self.assertRaisesRegex(contract.ContractError, "market|schema"):
            executor.validate_candidate_rows(self.registered.recipe, candidates)
        settlements = self.settlement_rows(self.candidate_rows())
        settlements[0]["market_price"] = 2.1
        with self.assertRaisesRegex(contract.ContractError, "market"):
            executor.validate_settlement_rows(self.registered.recipe, settlements)

    def test_missing_settlement_never_drops_a_race(self) -> None:
        candidates = self.candidate_rows()
        decisions, receipt = executor._freeze_decisions_after_authenticated_mount(
            self.registered,
            candidates,
            run_scope_digest="a" * 64,
            replica_id="clean_a",
            irreversible_receipt_digest="b" * 64,
        )
        settlements = self.settlement_rows(candidates)[:-1]
        with self.assertRaisesRegex(contract.ContractError, "row count"):
            executor._settle_diagnostic_after_authenticated_mount(
                self.registered,
                decisions,
                settlements,
                run_scope_digest="a" * 64,
                replica_id="clean_a",
                decision_freeze_receipt=receipt,
                settlement_operation_receipt_digest="c" * 64,
                bootstrap_replicates_override_for_synthetic_test=1,
            )

    def test_private_replica_projection_requires_semantic_equality(self) -> None:
        left = {
            "replica_id": "clean_a",
            "run_scope_digest": "a" * 64,
            "recipe_digest": "b" * 64,
            "scientific_projection_digest": "c" * 64,
            "computed_outcome": "NO_DECISION_EFFECT",
            "result_digest": "d" * 64,
        }
        right = dict(left, replica_id="clean_b", result_digest="e" * 64)
        receipt = executor._compare_replica_results(left, right)
        self.assertTrue(receipt["bitwise_semantic_equality"])
        with self.assertRaisesRegex(contract.ContractError, "distinct"):
            executor._compare_replica_results(left, left)
        changed = dict(right, scientific_projection_digest="f" * 64)
        with self.assertRaisesRegex(contract.ContractError, "mismatch"):
            executor._compare_replica_results(left, changed)
        self.assertFalse(hasattr(executor, "compare_replica_results"))
        self.assertNotIn("_compare_replica_results", executor.__all__)

    def test_typed_replica_compare_binds_current_settlement_seals(self) -> None:
        run, clean_a, clean_b = self.settlement_result_pair()
        seals = (
            self.settlement_seal(
                run=run, result=clean_a, replica_id="clean_a", marker="a"
            ),
            self.settlement_seal(
                run=run, result=clean_b, replica_id="clean_b", marker="b"
            ),
        )
        issued, consumed, evidence = self.compare_chain(run=run, seals=seals)
        authority = object.__new__(g2.SharedG2LeaseAuthorityClient)
        subject = self.compare_executor(run=run, authority=authority)
        with mock.patch.object(
            g2.SharedG2LeaseAuthorityClient,
            "fetch_and_revalidate_unrevoked_phase_output_seal",
            side_effect=evidence,
        ) as revalidate, mock.patch.object(
            g2.SharedG2LeaseAuthorityClient,
            "consume_phase_lease",
            return_value=consumed,
        ) as consume:
            comparison, observed_consumed = subject.replica_compare(
                issued_lease=issued,
                expected_global_head=self.head(),
                dispatch_digest="1" * 64,
                consume_revalidation_receipt_digest="2" * 64,
                operation_id="compare-operation",
                idempotency_key="compare-idempotency",
                requested_at="2026-08-15T00:00:04Z",
                clean_a_result=clean_a,
                clean_b_result=clean_b,
                settlement_output_seals=seals,
            )
        self.assertEqual(observed_consumed, consumed)
        self.assertEqual(comparison["replica_ids"], ["clean_a", "clean_b"])
        self.assertEqual(
            comparison["scientific_projection_digest"],
            clean_a["scientific_projection_digest"],
        )
        self.assertEqual(
            executor.settlement_result_output_digest(clean_a),
            seals[0].receipt.output_digest,
        )
        self.assertEqual(
            revalidate.call_args_list,
            [mock.call(seals[0].receipt.payload_digest), mock.call(seals[1].receipt.payload_digest)],
        )
        consume.assert_called_once()

    def test_replica_compare_never_reads_results_before_remote_consume(self) -> None:
        class CountingMapping(Mapping[str, object]):
            def __init__(self) -> None:
                self.read_count = 0

            def __getitem__(self, key: str) -> object:
                self.read_count += 1
                raise KeyError(key)

            def __iter__(self) -> Iterator[str]:
                self.read_count += 1
                return iter(())

            def __len__(self) -> int:
                self.read_count += 1
                return 0

        run, clean_a, clean_b = self.settlement_result_pair()
        seals = (
            self.settlement_seal(
                run=run, result=clean_a, replica_id="clean_a", marker="a"
            ),
            self.settlement_seal(
                run=run, result=clean_b, replica_id="clean_b", marker="b"
            ),
        )
        issued, _, evidence = self.compare_chain(run=run, seals=seals)
        authority = object.__new__(g2.SharedG2LeaseAuthorityClient)
        subject = self.compare_executor(run=run, authority=authority)
        left = CountingMapping()
        right = CountingMapping()
        with mock.patch.object(
            g2.SharedG2LeaseAuthorityClient,
            "fetch_and_revalidate_unrevoked_phase_output_seal",
            side_effect=evidence,
        ), mock.patch.object(
            g2.SharedG2LeaseAuthorityClient,
            "consume_phase_lease",
            side_effect=contract.ContractError("remote consume failed"),
        ):
            with self.assertRaisesRegex(contract.ContractError, "consume failed"):
                subject.replica_compare(
                    issued_lease=issued,
                    expected_global_head=self.head(),
                    dispatch_digest="1" * 64,
                    consume_revalidation_receipt_digest="2" * 64,
                    operation_id="compare-operation",
                    idempotency_key="compare-idempotency",
                    requested_at="2026-08-15T00:00:04Z",
                    clean_a_result=left,
                    clean_b_result=right,
                    settlement_output_seals=seals,
                )
        self.assertEqual((left.read_count, right.read_count), (0, 0))

    def test_settlement_never_reads_decision_output_before_lease_consume(self) -> None:
        run, clean_a, _ = self.settlement_result_pair()
        synthetic = self.settlement_seal(
            run=run, result=clean_a, replica_id="clean_a", marker="a"
        )
        decision_seal = replace(
            synthetic,
            receipt=replace(synthetic.receipt, phase="DECISION_FREEZE"),
        )
        provisional = self.binding(
            run=run,
            phase="SETTLEMENT_DIAGNOSTIC",
            replica_id="clean_a",
            predecessor_digest="f" * 64,
        )
        predecessor = g2.canonical_predecessor_output_digest(
            successor_binding=provisional,
            predecessor_output_seals=(decision_seal,),
        )
        issued = g2.IssuedPhaseLease(
            lease=self.phase_lease(
                replace(provisional, predecessor_receipt_digest=predecessor),
                marker="d",
            ),
            transaction=None,  # type: ignore[arg-type]
            resulting_run_state=None,
        )
        evidence = g2.RevalidatedPhaseOutputSeal(
            receipt=decision_seal.receipt,
            subject_head=None,  # type: ignore[arg-type]
            subject_snapshot=None,  # type: ignore[arg-type]
            authority_snapshot=None,  # type: ignore[arg-type]
        )
        authority = object.__new__(g2.SharedG2LeaseAuthorityClient)
        subject = self.compare_executor(run=run, authority=authority)
        with mock.patch.object(
            g2.SharedG2LeaseAuthorityClient,
            "fetch_and_revalidate_unrevoked_phase_output_seal",
            return_value=evidence,
        ), mock.patch.object(
            g2.SharedG2LeaseAuthorityClient,
            "consume_phase_lease",
            side_effect=contract.ContractError("settlement consume failed"),
        ), mock.patch.object(
            executor, "decision_output_projection_digest"
        ) as decision_digest:
            with self.assertRaisesRegex(contract.ContractError, "consume failed"):
                subject.settlement_diagnostic(
                    issued_lease=issued,
                    expected_global_head=self.head(),
                    dispatch_digest="1" * 64,
                    consume_revalidation_receipt_digest="2" * 64,
                    operation_id="settlement-operation",
                    idempotency_key="settlement-idempotency",
                    requested_at="2026-08-15T00:00:04Z",
                    decision_rows=[],
                    decision_freeze_receipt={},
                    decision_output_seal=decision_seal,
                )
        decision_digest.assert_not_called()

    def test_replica_compare_rejects_fabricated_result_and_wrong_seal(self) -> None:
        run, clean_a, clean_b = self.settlement_result_pair()
        seals = (
            self.settlement_seal(
                run=run, result=clean_a, replica_id="clean_a", marker="a"
            ),
            self.settlement_seal(
                run=run, result=clean_b, replica_id="clean_b", marker="b"
            ),
        )
        issued, consumed, evidence = self.compare_chain(run=run, seals=seals)
        authority = object.__new__(g2.SharedG2LeaseAuthorityClient)
        subject = self.compare_executor(run=run, authority=authority)
        fabricated = copy.deepcopy(clean_a)
        fabricated["computed_outcome"] = "NO_DECISION_EFFECT"
        with mock.patch.object(
            g2.SharedG2LeaseAuthorityClient,
            "fetch_and_revalidate_unrevoked_phase_output_seal",
            side_effect=evidence,
        ), mock.patch.object(
            g2.SharedG2LeaseAuthorityClient,
            "consume_phase_lease",
            return_value=consumed,
        ):
            with self.assertRaisesRegex(contract.ContractError, "self-digest"):
                subject.replica_compare(
                    issued_lease=issued,
                    expected_global_head=self.head(),
                    dispatch_digest="1" * 64,
                    consume_revalidation_receipt_digest="2" * 64,
                    operation_id="compare-operation",
                    idempotency_key="compare-idempotency",
                    requested_at="2026-08-15T00:00:04Z",
                    clean_a_result=fabricated,
                    clean_b_result=clean_b,
                    settlement_output_seals=seals,
                )

        wrong_receipt = replace(seals[0].receipt, output_digest="f" * 64)
        wrong_seal = replace(seals[0], receipt=wrong_receipt)
        wrong_seals = (wrong_seal, seals[1])
        wrong_issued, _, wrong_evidence = self.compare_chain(
            run=run, seals=wrong_seals
        )
        with mock.patch.object(
            g2.SharedG2LeaseAuthorityClient,
            "fetch_and_revalidate_unrevoked_phase_output_seal",
            side_effect=wrong_evidence,
        ), mock.patch.object(
            g2.SharedG2LeaseAuthorityClient,
            "consume_phase_lease",
        ) as consume:
            with self.assertRaisesRegex(contract.ContractError, "authenticated replica"):
                subject.replica_compare(
                    issued_lease=wrong_issued,
                    expected_global_head=self.head(),
                    dispatch_digest="1" * 64,
                    consume_revalidation_receipt_digest="2" * 64,
                    operation_id="compare-operation-2",
                    idempotency_key="compare-idempotency-2",
                    requested_at="2026-08-15T00:00:04Z",
                    clean_a_result=clean_a,
                    clean_b_result=clean_b,
                    settlement_output_seals=wrong_seals,
                )
        consume.assert_not_called()

    def test_duck_typed_authority_is_rejected_before_content_access(self) -> None:
        class FakeAuthority:
            def consume_decision_lease_batch(self, **_: object) -> object:
                raise AssertionError("fake authority must never be called")

        class CountingContent:
            read_count = 0

            def fetch_candidate_rows(self, **_: object) -> list[object]:
                self.read_count += 1
                return []

            def fetch_settlement_rows(self, **_: object) -> list[object]:
                self.read_count += 1
                return []

        content = CountingContent()
        with self.assertRaisesRegex(contract.ContractError, "duck-typed authority"):
            executor.RUN_SCOPE_BOUND_PHASE_PLAN(
                registered=self.registered,
                run_scope=self.run_scope(),
                authority=FakeAuthority(),
                content_transport=content,
            )
        self.assertEqual(content.read_count, 0)

    def test_settlement_content_rejects_untyped_chain_before_read(self) -> None:
        class CountingContent:
            read_count = 0

            def fetch_candidate_rows(self, **_: object) -> list[object]:
                self.read_count += 1
                return []

            def fetch_settlement_rows(self, **_: object) -> list[object]:
                self.read_count += 1
                return []

        transport = CountingContent()
        provider = executor.AuthenticatedProtectedContentProvider(
            transport=transport,
            run_scope_digest="a" * 64,
            catalog_release_id="catalog_release_v1",
            candidate_entry_sha256="b" * 64,
            settlement_entry_sha256="c" * 64,
        )
        with self.assertRaisesRegex(contract.ContractError, "typed consumption"):
            provider.read_settlement_rows(
                consumed={},  # type: ignore[arg-type]
                decision_output_seal={},  # type: ignore[arg-type]
            )
        self.assertEqual(transport.read_count, 0)

    def test_structured_argv_resolves_to_exact_callable(self) -> None:
        run = self.run_scope()
        self.assertTrue(callable(executor.RUN_SCOPE_BOUND_PHASE_PLAN))
        self.assertEqual(
            executor.resolve_run_scope_bound_phase_plan(self.registered, run),
            (
                "VERIFY_CUTOVER_AND_RUN_SCOPE",
                "DECISION_FREEZE",
                "SETTLEMENT_DIAGNOSTIC",
                "REPLICA_COMPARE",
                "RESULT_SEAL",
            ),
        )
        tampered = copy.deepcopy(run)
        tampered["resolved_contracts"]["execution_contract"][
            "structured_argv"
        ].append("unexpected")
        with self.assertRaises(contract.ContractError):
            executor.resolve_run_scope_bound_phase_plan(self.registered, tampered)

    def test_legacy_single_replica_live_entrypoint_is_absent(self) -> None:
        self.assertFalse(hasattr(executor.SupervisedDiagnosticExecutor, "decision_freeze"))
        self.assertFalse(hasattr(executor, "SharedG2Authority"))
        self.assertFalse(hasattr(executor, "freeze_decisions"))
        self.assertFalse(hasattr(executor, "settle_diagnostic"))
        self.assertNotIn(
            "_freeze_decisions_after_authenticated_mount", executor.__all__
        )
        self.assertNotIn(
            "_settle_diagnostic_after_authenticated_mount", executor.__all__
        )

if __name__ == "__main__":
    unittest.main()
