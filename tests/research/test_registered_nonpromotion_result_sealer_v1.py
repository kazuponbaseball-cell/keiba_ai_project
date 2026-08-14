from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import registered_nonpromotion_contract_v1 as contract
import registered_nonpromotion_result_sealer_v1 as sealer
from shared_g2_durable_ledger_v1 import (
    AuthoritySnapshot,
    GlobalHead,
    SubjectHead,
    SubjectHeadSnapshot,
    WitnessCheckpoint,
)
from shared_g2_lease_authority_v1 import (
    PhaseOutputSealReceipt,
    RevalidatedPhaseOutputSeal,
    phase_output_seal_state_digest,
)


H = {character: character * 64 for character in "0123456789abcdef"}


class FakeRemoteSealVerifier:
    def __init__(self, evidence: list[RevalidatedPhaseOutputSeal]) -> None:
        self._evidence = {
            item.receipt.payload_digest: item for item in evidence
        }
        self.revoked: set[str] = set()
        self.calls: list[str] = []

    def fetch_and_revalidate_unrevoked_phase_output_seal(
        self, receipt_payload_digest: str
    ) -> RevalidatedPhaseOutputSeal:
        self.calls.append(receipt_payload_digest)
        if receipt_payload_digest in self.revoked:
            raise contract.ContractError("remote shared-G2 seal is revoked")
        try:
            return self._evidence[receipt_payload_digest]
        except KeyError as exc:
            raise contract.ContractError(
                "remote shared-G2 seal is unavailable"
            ) from exc


class RegisteredNonpromotionResultSealerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registered = contract.resolve_registered_recipe(
            ROOT,
            recipe_id="historical_ai_duplicate_gate_impact_v1",
            recipe_version=1,
        )

    def run_scope(self) -> dict[str, Any]:
        bindings = {
            "repository": contract.DEFAULT_REPOSITORY,
            "base_branch": contract.DEFAULT_BASE_BRANCH,
            "run_scope_base_commit": "a" * 40,
            "verified_current_main_sha": "b" * 40,
            "approvers_blob_sha": "c" * 40,
            "approvers_content_sha256": H["d"],
            "activation_receipt_sha256": H["1"],
            "cutover_receipt_sha256": H["1"],
            "schema_bundle_sha256": self.registered.schema_bundle_digest,
            "approval_evidence_schema_sha256": H["3"],
            "capability_profile_sha256": H["4"],
            "policy_blob_sha256": self.registered.policy_file_sha256,
            "recipe_blob_sha256": self.registered.recipe_file_sha256,
            "compiler_blob_sha256": H["5"],
            "authority_verifier_blob_sha256": H["6"],
            "catalog_validator_blob_sha256": H["7"],
            "executor_blob_sha256": H["8"],
            "runner_blob_sha256": H["9"],
            "result_sealer_blob_sha256": H["a"],
            "g2_authority_service_blob_sha256": H["b"],
            "phase_lease_schema_sha256": H["c"],
            "phase_operation_receipt_schema_sha256": H["d"],
            "environment_manifest_sha256": H["e"],
            "catalog_release_id": "catalog_release_v1",
            "catalog_release_sha256": H["7"],
            "catalog_release_status": "ACTIVE",
            "catalog_release_revoked": False,
            "catalog_status_receipt_sha256": H["8"],
            "candidate_entry_sha256": H["8"],
            "candidate_schema_sha256": H["9"],
            "candidate_provenance_sha256": H["a"],
            "p_action_cross_source_equality_attestation_sha256": H["b"],
            "candidate_materializer_usecols_sha256": H["c"],
            "decision_base_lineage_sha256": H["d"],
            "settlement_entry_sha256": H["9"],
            "settlement_schema_sha256": H["e"],
            "settlement_provenance_sha256": H["f"],
            "official_settlement_provenance_sha256": H["1"],
            "cohort_manifest_sha256": H["a"],
            "ordered_race_set_sha256": H["b"],
            "output_root": "outputs/research/RND-001",
            "sealed_at": "2026-08-15T00:00:00Z",
            "expected_pregrant_global_head": H["c"],
            "expected_pregrant_subject_head": H["d"],
            "cutover_epoch": 1,
            "external_witness_checkpoint_sha256": H["f"],
        }
        bindings.update(self.registered.runtime_material_digests)
        return contract.compile_run_scope(self.registered, bindings)

    def comparison_projection(self, run_scope: dict[str, Any]) -> dict[str, Any]:
        projection = {
            "schema_version": 1,
            "receipt_kind": "REPLICA_COMPARISON",
            "run_scope_digest": run_scope["run_scope_digest"],
            "recipe_digest": run_scope["recipe_digest"],
            "replica_ids": ["clean_a", "clean_b"],
            "replica_result_digests": [H["2"], H["3"]],
            "scientific_projection_digest": H["4"],
            "computed_outcome": "NO_DECISION_EFFECT",
            "both_contract_status_valid": True,
            "bitwise_semantic_equality": True,
            "authority": False,
            "authenticated_phase_output_seal_required": True,
        }
        projection["receipt_digest"] = contract.canonical_digest(projection)
        return projection

    @staticmethod
    def head(sequence: int, digest: str) -> GlobalHead:
        return GlobalHead(
            authority_id="shared-g2",
            activation_epoch="epoch-1",
            backend_identity_digest=H["5"],
            cutover_receipt_digest=H["6"],
            sequence=sequence,
            head_digest=digest,
            observed_at=f"2026-08-15T00:00:{sequence:02d}Z",
        )

    def evidence(
        self,
        *,
        run_scope: dict[str, Any],
        phase: str,
        actor: str,
        output_digest: str,
        receipt_digest: str,
        from_sequence: int,
        sealed_sequence: int,
    ) -> RevalidatedPhaseOutputSeal:
        from_head = self.head(from_sequence, H["7"])
        sealed_head = self.head(sealed_sequence, H["8"])
        receipt = PhaseOutputSealReceipt(
            payload_digest=receipt_digest,
            envelope_digest=H["9"],
            run_scope_digest=run_scope["run_scope_digest"],
            recipe_digest=run_scope["recipe_digest"],
            replica_id=actor,
            phase=phase,
            attempt=1,
            binding_digest=H["a"],
            phase_output_subject_digest=H["f"],
            lease_payload_digest=H["b"],
            lease_consumption_receipt_digest=H["c"],
            output_attestation_digest=H["d"],
            output_digest=output_digest,
            operation_kind=(
                "RND_RESULT_SEAL"
                if phase == "RESULT_SEAL"
                else "RND_PHASE_OUTPUT_SEAL"
            ),
            sealed_from_global_head=from_head,
            sealed_global_head=sealed_head,
            seal_transaction_id=f"txn-{phase.lower()}",
            sealed_at=f"2026-08-15T00:00:{sealed_sequence:02d}Z",
        )
        subject = SubjectHead(
            subject_kind="PHASE_OUTPUT",
            subject_digest=receipt.phase_output_subject_digest,
            generation=0,
            sequence=1,
            head_digest=H["b"],
            state_digest=phase_output_seal_state_digest(receipt),
        )
        witness = WitnessCheckpoint(
            authority_id="shared-g2",
            activation_epoch="epoch-1",
            backend_identity_digest=H["5"],
            witness_identity_digest=H["d"],
            cutover_receipt_digest=H["6"],
            checkpoint_sequence=sealed_sequence + 1,
            observed_global_head=sealed_head,
            previous_checkpoint_digest=H["e"],
            checkpoint_digest=H["f"],
            witnessed_at=f"2026-08-15T00:00:{sealed_sequence:02d}Z",
        )
        subject_snapshot = SubjectHeadSnapshot(
            global_head=sealed_head,
            subject_head=subject,
            read_at=f"2026-08-15T00:00:{sealed_sequence:02d}Z",
            payload_digest=H["3"],
            envelope_digest=H["4"],
        )
        return RevalidatedPhaseOutputSeal(
            receipt=receipt,
            subject_head=subject,
            subject_snapshot=subject_snapshot,
            authority_snapshot=AuthoritySnapshot(
                global_head=sealed_head,
                witness=witness,
            ),
        )

    def build_case(self) -> tuple[
        dict[str, Any],
        dict[str, Any],
        FakeRemoteSealVerifier,
        str,
        str,
    ]:
        run_scope = self.run_scope()
        comparison = self.comparison_projection(run_scope)
        projection = sealer.project_registered_nonpromotion_result(
            registered=self.registered,
            run_scope=run_scope,
            comparison_projection=comparison,
            policy_schema_verifier_digest=H["e"],
        )
        comparison_seal_digest = H["1"]
        result_seal_digest = H["2"]
        verifier = FakeRemoteSealVerifier(
            [
                self.evidence(
                    run_scope=run_scope,
                    phase="REPLICA_COMPARE",
                    actor="lane_coordinator",
                    output_digest=comparison["receipt_digest"],
                    receipt_digest=comparison_seal_digest,
                    from_sequence=10,
                    sealed_sequence=11,
                ),
                self.evidence(
                    run_scope=run_scope,
                    phase="RESULT_SEAL",
                    actor="canonical_sealer",
                    output_digest=projection["result_projection_digest"],
                    receipt_digest=result_seal_digest,
                    from_sequence=12,
                    sealed_sequence=13,
                ),
            ]
        )
        return (
            run_scope,
            projection,
            verifier,
            comparison_seal_digest,
            result_seal_digest,
        )

    def test_two_step_projection_and_authenticated_sealing(self) -> None:
        run, projection, verifier, comparison_seal, result_seal = self.build_case()
        sealed = sealer.seal_registered_nonpromotion_result(
            registered=self.registered,
            run_scope=run,
            result_projection=projection,
            comparison_phase_output_seal_receipt_digest=comparison_seal,
            result_phase_output_seal_receipt_digest=result_seal,
            seal_verifier=verifier,
        )
        self.assertEqual(sealed["computed_outcome"], "NO_DECISION_EFFECT")
        self.assertEqual(
            sealed["result_phase_output_seal_receipt_digest"], result_seal
        )
        self.assertEqual(verifier.calls, [comparison_seal, result_seal])

        replay = sealer.validate_exact_replay(
            registered=self.registered,
            run_scope=run,
            sealed_result=sealed,
            seal_verifier=verifier,
        )
        self.assertEqual(replay["result_digest"], sealed["result_digest"])
        self.assertEqual(
            verifier.calls,
            [comparison_seal, result_seal, comparison_seal, result_seal],
        )

    def test_comparison_projection_must_match_authenticated_compare_seal(self) -> None:
        run, projection, verifier, comparison_seal, result_seal = self.build_case()
        evidence = verifier._evidence[comparison_seal]
        verifier._evidence[comparison_seal] = RevalidatedPhaseOutputSeal(
            receipt=copy.copy(evidence.receipt),
            subject_head=evidence.subject_head,
            subject_snapshot=evidence.subject_snapshot,
            authority_snapshot=evidence.authority_snapshot,
        )
        object.__setattr__(
            verifier._evidence[comparison_seal].receipt,
            "output_digest",
            H["0"],
        )
        with self.assertRaisesRegex(contract.ContractError, "output_digest"):
            sealer.seal_registered_nonpromotion_result(
                registered=self.registered,
                run_scope=run,
                result_projection=projection,
                comparison_phase_output_seal_receipt_digest=comparison_seal,
                result_phase_output_seal_receipt_digest=result_seal,
                seal_verifier=verifier,
            )

    def test_result_projection_must_match_authenticated_result_seal(self) -> None:
        run, projection, verifier, comparison_seal, result_seal = self.build_case()
        changed = copy.deepcopy(projection)
        changed["computed_outcome"] = "DIRECTIONAL_EFFECT"
        changed["result_projection_digest"] = contract.canonical_digest(
            {
                key: value
                for key, value in changed.items()
                if key != "result_projection_digest"
            }
        )
        with self.assertRaisesRegex(contract.ContractError, "output_digest"):
            sealer.seal_registered_nonpromotion_result(
                registered=self.registered,
                run_scope=run,
                result_projection=changed,
                comparison_phase_output_seal_receipt_digest=comparison_seal,
                result_phase_output_seal_receipt_digest=result_seal,
                seal_verifier=verifier,
            )

    def test_phase_actor_is_exact(self) -> None:
        run, projection, verifier, comparison_seal, result_seal = self.build_case()
        evidence = verifier._evidence[comparison_seal]
        object.__setattr__(evidence.receipt, "replica_id", "clean_a")
        with self.assertRaisesRegex(contract.ContractError, "replica_id"):
            sealer.seal_registered_nonpromotion_result(
                registered=self.registered,
                run_scope=run,
                result_projection=projection,
                comparison_phase_output_seal_receipt_digest=comparison_seal,
                result_phase_output_seal_receipt_digest=result_seal,
                seal_verifier=verifier,
            )

    def test_exact_replay_fails_when_remote_seal_is_revoked(self) -> None:
        run, projection, verifier, comparison_seal, result_seal = self.build_case()
        sealed = sealer.seal_registered_nonpromotion_result(
            registered=self.registered,
            run_scope=run,
            result_projection=projection,
            comparison_phase_output_seal_receipt_digest=comparison_seal,
            result_phase_output_seal_receipt_digest=result_seal,
            seal_verifier=verifier,
        )
        verifier.revoked.add(result_seal)
        with self.assertRaisesRegex(contract.ContractError, "revoked"):
            sealer.validate_exact_replay(
                registered=self.registered,
                run_scope=run,
                sealed_result=sealed,
                seal_verifier=verifier,
            )

    def test_malformed_sha_in_authenticated_evidence_fails_closed(self) -> None:
        run, projection, verifier, comparison_seal, result_seal = self.build_case()
        evidence = verifier._evidence[comparison_seal]
        object.__setattr__(evidence.receipt, "envelope_digest", "NOT-A-SHA")
        with self.assertRaisesRegex(contract.ContractError, "lowercase SHA-256"):
            sealer.seal_registered_nonpromotion_result(
                registered=self.registered,
                run_scope=run,
                result_projection=projection,
                comparison_phase_output_seal_receipt_digest=comparison_seal,
                result_phase_output_seal_receipt_digest=result_seal,
                seal_verifier=verifier,
            )

    def test_current_phase_output_subject_must_bind_exact_receipt(self) -> None:
        run, projection, verifier, comparison_seal, result_seal = self.build_case()
        evidence = verifier._evidence[comparison_seal]
        object.__setattr__(evidence.subject_head, "subject_digest", H["0"])
        with self.assertRaisesRegex(
            contract.ContractError, "non-zero|subject digest mismatch"
        ):
            sealer.seal_registered_nonpromotion_result(
                registered=self.registered,
                run_scope=run,
                result_projection=projection,
                comparison_phase_output_seal_receipt_digest=comparison_seal,
                result_phase_output_seal_receipt_digest=result_seal,
                seal_verifier=verifier,
            )

    def test_flat_self_asserted_operation_receipt_is_not_an_api(self) -> None:
        run_scope = self.run_scope()
        comparison = self.comparison_projection(run_scope)
        with self.assertRaises(TypeError):
            sealer.seal_registered_nonpromotion_result(  # type: ignore[call-arg]
                registered=self.registered,
                run_scope=run_scope,
                comparison_receipt=comparison,
                authenticated_seal_operation_receipt={
                    "authenticated_remote_g2": True,
                    "one_shot_consumed": True,
                },
                policy_schema_verifier_digest=H["f"],
            )


if __name__ == "__main__":
    unittest.main()
