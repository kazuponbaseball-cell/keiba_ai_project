from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import registered_nonpromotion_contract_v1 as contract


class RegisteredNonpromotionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registered = contract.resolve_registered_recipe(
            ROOT,
            recipe_id="historical_ai_duplicate_gate_impact_v1",
            recipe_version=1,
        )

    def bindings(self) -> dict[str, object]:
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
            "environment_manifest_sha256": "f" * 64,
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
            "external_witness_checkpoint_sha256": "e" * 64,
        }
        bindings.update(self.registered.runtime_material_digests)
        return bindings

    def test_registered_source_fingerprints_are_exact(self) -> None:
        self.assertEqual(
            self.registered.recipe["source_design"]["formula_fingerprint_sha256"],
            "67b0d3e5b92166a10b3077bff03e107c9db071b310f60f43b8758ce316eda878",
        )
        self.assertEqual(
            self.registered.question_family_digest,
            "a9555c0fe36f57b924822ed8d718ba6d2b150140feb4607d7749e300151318d7",
        )
        self.assertEqual(
            self.registered.eligibility_ast_digest,
            "09383c64cca20fa5a7a8fcfb9f76f1bf7feeb5e8b76744a90e4aa954b9ba0c5f",
        )

    def test_d0_d1_boundaries_and_subset(self) -> None:
        recipe = self.registered.recipe
        rows = [
            {
                "race_id": "2026010101010101",
                "candidate_key": "1-2",
                "eligible_race": True,
                "candidate_generated": True,
                "top1_wide_prob": 0.324999999999,
                "p_action_C0_offset": 0.3543,
            },
            {
                "race_id": "2026010101010102",
                "candidate_key": "2-3",
                "eligible_race": True,
                "candidate_generated": True,
                "top1_wide_prob": 0.325,
                "p_action_C0_offset": 0.3543,
            },
            {
                "race_id": "2026010101010103",
                "candidate_key": "3-4",
                "eligible_race": True,
                "candidate_generated": True,
                "top1_wide_prob": 0.369,
                "p_action_C0_offset": 0.399999,
            },
            {
                "race_id": "2026010101010104",
                "candidate_key": "4-5",
                "eligible_race": True,
                "candidate_generated": True,
                "top1_wide_prob": 0.3691,
                "p_action_C0_offset": 0.4,
            },
        ]
        decisions = contract.evaluate_registered_decisions(recipe, rows)
        self.assertEqual(
            [(row["d0_eligible"], row["d1_eligible"]) for row in decisions],
            [(True, True), (False, True), (False, True), (False, False)],
        )

    def test_recipe_score_cannot_be_laundered(self) -> None:
        altered = copy.deepcopy(self.registered.recipe)
        altered["ordinary_strategy_score"]["threshold_met"] = True
        entry = copy.deepcopy(self.registered.policy["recipe_registry"]["entries"][0])
        entry["canonical_recipe_sha256"] = contract.canonical_digest(altered)
        with self.assertRaisesRegex(contract.ContractError, "score"):
            contract.validate_recipe(altered, entry)

    def test_unknown_ast_operator_is_rejected(self) -> None:
        altered = copy.deepcopy(self.registered.recipe)
        altered["recipe_contract"]["clause_registry"]["a_ge_0_25"]["op"] = "python"
        entry = copy.deepcopy(self.registered.policy["recipe_registry"]["entries"][0])
        altered["source_design"]["formula_fingerprint_sha256"] = contract.canonical_digest(
            altered["recipe_contract"]
        )
        entry["source_formula_fingerprint_sha256"] = altered["source_design"][
            "formula_fingerprint_sha256"
        ]
        entry["eligibility_ast_sha256"] = contract.canonical_digest(
            contract.eligibility_ast_projection(altered)
        )
        entry["canonical_recipe_sha256"] = contract.canonical_digest(altered)
        with self.assertRaisesRegex(contract.ContractError, "unsupported clause"):
            contract.validate_recipe(altered, entry)

    def test_run_scope_is_registry_resolved_and_digest_bound(self) -> None:
        run = contract.compile_run_scope(self.registered, self.bindings())
        self.assertEqual(run["approval_keyword"], contract.APPROVAL_KEYWORD)
        self.assertEqual(run["ordinary_strategy_score"]["recorded_total"], 46)
        self.assertFalse(run["ordinary_strategy_score"]["threshold_met"])
        self.assertEqual(contract.verify_run_scope_digest(run), run["run_scope_digest"])
        self.assertEqual(
            contract.verify_canonical_run_scope(self.registered, run),
            run["run_scope_digest"],
        )
        changed = copy.deepcopy(run)
        changed["resolved_contracts"]["metric"]["offline_evaluation_notional_yen"] = 200
        with self.assertRaisesRegex(contract.ContractError, "digest"):
            contract.verify_run_scope_digest(changed)

    def test_self_rehashed_subject_alias_is_not_a_canonical_run_scope(self) -> None:
        run = contract.compile_run_scope(self.registered, self.bindings())
        changed = copy.deepcopy(run)
        changed["semantic_subject"]["cohort_id"] = "alias_cohort"
        changed["semantic_subject_digest"] = contract.canonical_digest(
            changed["semantic_subject"]
        )
        unsigned = dict(changed)
        unsigned.pop("run_scope_digest")
        changed["run_scope_digest"] = contract.canonical_digest(unsigned)
        self.assertEqual(
            contract.verify_run_scope_digest(changed), changed["run_scope_digest"]
        )
        with self.assertRaisesRegex(contract.ContractError, "canonical"):
            contract.verify_canonical_run_scope(self.registered, changed)

    def test_run_scope_rejects_free_form_or_missing_binding(self) -> None:
        bindings = self.bindings()
        bindings["threshold"] = 0.99
        with self.assertRaisesRegex(contract.ContractError, "keys mismatch"):
            contract.compile_run_scope(self.registered, bindings)
        bindings = self.bindings()
        del bindings["catalog_release_sha256"]
        with self.assertRaisesRegex(contract.ContractError, "keys mismatch"):
            contract.compile_run_scope(self.registered, bindings)

    def test_run_scope_is_bound_to_role_separated_catalog_metadata(self) -> None:
        run = contract.compile_run_scope(self.registered, self.bindings())
        metadata = {
            "release_id": "catalog_release_v1",
            "release_manifest_sha256": "7" * 64,
            "status_receipt_sha256": "8" * 64,
            "ordered_race_id_set_sha256": "b" * 64,
            "race_count": 3746,
            "candidate_entry": {
                "content_sha256": "8" * 64,
                "schema_sha256": "9" * 64,
                "provenance_sha256": "a" * 64,
                "role_attestations": {
                    "p_action_cross_source_equality_attestation_sha256": "b" * 64,
                    "candidate_materializer_usecols_sha256": "c" * 64,
                    "decision_base_lineage_sha256": "d" * 64,
                },
            },
            "settlement_entry": {
                "content_sha256": "9" * 64,
                "schema_sha256": "e" * 64,
                "provenance_sha256": "f" * 64,
                "role_attestations": {
                    "official_settlement_provenance_sha256": "1" * 64,
                },
            },
        }
        contract.verify_run_scope_catalog_binding(self.registered, run, metadata)
        metadata["candidate_entry"]["content_sha256"] = "f" * 64
        with self.assertRaisesRegex(contract.ContractError, "candidate_entry_sha256"):
            contract.verify_run_scope_catalog_binding(self.registered, run, metadata)
    def test_strict_json_rejects_duplicates_and_nonfinite(self) -> None:
        with self.assertRaises(contract.ContractError):
            contract.strict_json_loads('{"a":1,"a":2}')
        with self.assertRaises(contract.ContractError):
            contract.strict_json_loads('{"a":NaN}')

    def test_policy_cannot_enable_local_authority(self) -> None:
        altered = copy.deepcopy(self.registered.policy)
        altered["shared_g2"]["local_fallback"] = True
        with self.assertRaisesRegex(contract.ContractError, "fallback"):
            contract.validate_policy(altered)


if __name__ == "__main__":
    unittest.main()
