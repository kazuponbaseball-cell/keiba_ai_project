from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import registered_nonpromotion_contract_v1 as strict_contract
import registered_nonpromotion_offline_contract_v1 as contract


class RegisteredNonpromotionOfflineContractV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registered = contract.resolve_offline_registered_recipe(ROOT)

    def bindings(self) -> dict[str, object]:
        source_bindings: dict[str, object] = {}
        for index, (role, expected) in enumerate(contract.SOURCE_INPUTS.items(), 1):
            source_bindings[role] = {
                "path": expected["path"],
                "expected_sha256": expected["expected_sha256"],
                "observed_sha256": expected["expected_sha256"],
                "observed_byte_size": index * 100,
            }
        return {
            "repository": contract.DEFAULT_REPOSITORY,
            "base_branch": contract.DEFAULT_BASE_BRANCH,
            "run_scope_base_commit": "a" * 40,
            "verified_current_main_sha": "a" * 40,
            "approvers_blob_sha": "b" * 40,
            "approvers_content_sha256": "c" * 64,
            "runtime_material_sha256": dict(
                self.registered.runtime_material_digests
            ),
            "source_bindings": source_bindings,
            "projection_bindings": {
                "candidate_projection": {
                    "path": contract.PROJECTION_INPUTS["candidate_projection"]["path"],
                    "sha256": "d" * 64,
                    "byte_size": 1000,
                },
                "settlement_projection": {
                    "path": contract.PROJECTION_INPUTS["settlement_projection"]["path"],
                    "sha256": "e" * 64,
                    "byte_size": 2000,
                },
            },
            "materialization_manifest": {
                "path": contract.MATERIALIZATION_MANIFEST_PATH,
                "sha256": "1" * 64,
                "byte_size": 3000,
            },
            "python_minor_version": "3.11",
            "numpy_version": "2.4.3",
            "environment_manifest_sha256": "f" * 64,
            "output_root": contract.FIXED_OUTPUT_ROOT,
            "sealed_at": "2026-08-15T01:00:00Z",
        }

    def test_resolves_exact_strict_recipe_without_changing_strict_gate(self) -> None:
        registered = self.registered
        self.assertEqual(registered.policy["gate_kind"], contract.GATE_KIND)
        self.assertEqual(
            registered.source_registered.policy["gate_kind"],
            contract.SOURCE_GATE_KIND,
        )
        self.assertEqual(registered.recipe_digest, contract.EXPECTED_RECIPE_DIGEST)
        self.assertEqual(
            registered.question_family_digest,
            contract.EXPECTED_QUESTION_FAMILY_DIGEST,
        )
        self.assertEqual(
            registered.eligibility_ast_digest,
            contract.EXPECTED_ELIGIBILITY_AST_DIGEST,
        )
        self.assertEqual(
            registered.recipe["ordinary_strategy_score"], contract.EXPECTED_SCORE
        )

    def test_existing_strict_material_bytes_remain_at_pr43_hashes(self) -> None:
        expected = {
            "research/REGISTERED_NONPROMOTION_DIAGNOSTIC_V1.json": (
                "3a8282ae0ad858c9fe091cd5714cc37fcb70430f63c89f32fc395f72407804c4"
            ),
            "research/diagnostic_recipes/historical_ai_duplicate_gate_impact_v1.json": (
                "e2bc778b1a252429ae7d14291ef9d39207c54417cef7fca0fef81079fb03cb70"
            ),
            "scripts/research/registered_nonpromotion_contract_v1.py": (
                "e227565ffc20814842ed09880117cfb4b1f8c18ee14299994dfc96f2c9380c56"
            ),
            "scripts/research/registered_nonpromotion_supervised_executor_v1.py": (
                "10a27908af16f16b65575778d05bf2e1bc323f5e2ba253ac27e27228df1c4163"
            ),
        }
        for relative, digest in expected.items():
            self.assertEqual(
                strict_contract.normalized_utf8_lf_sha256(ROOT / relative), digest
            )

    def test_compiles_one_exact_nonpromotion_scope(self) -> None:
        scope = contract.compile_offline_run_scope(self.registered, self.bindings())
        self.assertEqual(
            contract.verify_canonical_offline_run_scope(self.registered, scope),
            scope["run_scope_digest"],
        )
        self.assertEqual(scope["recipe_id"], contract.RECIPE_ID)
        self.assertEqual(
            scope["implementation_commit"], scope["verified_current_main_sha"]
        )
        self.assertEqual(scope["recipe_version"], 1)
        self.assertEqual(scope["ordinary_strategy_score"], contract.EXPECTED_SCORE)
        self.assertEqual(scope["semantic_subject"]["race_count"], 3746)
        self.assertEqual(
            scope["resolved_contract_digests"]["lifecycle_sha256"],
            strict_contract.canonical_digest(
                self.registered.policy["lifecycle"]
            ),
        )
        self.assertEqual(
            scope["resolved_contract_digests"]["cross_route_sha256"],
            strict_contract.canonical_digest(
                self.registered.policy["cross_route_contract"]
            ),
        )
        self.assertEqual(
            scope["semantic_subject"]["comparison_arm_ids"],
            ["D0_REFERENCE", "D1_REMOVE_RAW_GATES"],
        )
        self.assertEqual(scope["single_use_policy"], "ONE_ACCEPTED_EXECUTION")
        self.assertEqual(
            scope["single_use_enforcement"],
            "BEST_EFFORT_LOCAL_EXCLUSIVE_RECEIPT",
        )
        self.assertFalse(scope["global_replay_proof"])
        self.assertFalse(scope["rollback_resistant"])
        self.assertFalse(scope["durable_remote_ledger"])
        self.assertEqual(
            scope["network_isolation"], "APPLICATION_LEVEL_NOT_OS_SANDBOX"
        )
        for flag in (
            "authority",
            "confirmatory",
            "promotion_eligible",
            "formal_buy",
            "send_order",
        ):
            self.assertFalse(scope[flag])
        self.assertEqual(scope["score_credit"], 0)
        self.assertEqual(scope["stake"], 0)
        self.assertEqual(
            contract.offline_run_scope_artifact_path(scope["run_scope_digest"]),
            (
                f"{contract.RUN_SCOPE_ARTIFACT_DIRECTORY}/"
                f"{scope['run_scope_digest']}.run.json"
            ),
        )

    def test_scope_digest_and_canonical_recompile_both_fail_closed(self) -> None:
        scope = contract.compile_offline_run_scope(self.registered, self.bindings())
        changed = copy.deepcopy(scope)
        changed["semantic_subject"]["race_count"] = 3745
        with self.assertRaisesRegex(strict_contract.ContractError, "digest mismatch"):
            contract.verify_offline_run_scope_digest(changed)
        changed["run_scope_digest"] = strict_contract.canonical_digest(
            {key: value for key, value in changed.items() if key != "run_scope_digest"}
        )
        contract.verify_offline_run_scope_digest(changed)
        with self.assertRaisesRegex(strict_contract.ContractError, "canonical compilation"):
            contract.verify_canonical_offline_run_scope(self.registered, changed)

    def test_requires_exact_current_main_and_environment(self) -> None:
        for key, value, message in (
            ("verified_current_main_sha", "b" * 40, "exact current main"),
            ("python_minor_version", "3.13", "Python minor"),
            ("numpy_version", "2.4.2", "NumPy"),
            ("output_root", "outputs/research/other", "single fixed root"),
        ):
            bindings = self.bindings()
            bindings[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                strict_contract.ContractError, message
            ):
                contract.compile_offline_run_scope(self.registered, bindings)

        bindings = self.bindings()
        bindings["sealed_at"] = "2026-08-15 01:00:00Z"
        with self.assertRaisesRegex(strict_contract.ContractError, "ISO-8601"):
            contract.compile_offline_run_scope(self.registered, bindings)

    def test_lifecycle_is_exact_and_fail_closed(self) -> None:
        policy = copy.deepcopy(self.registered.policy)
        policy["lifecycle"]["allowed_transitions"]["RNOD_APPROVED"] = [
            "RNOD_RUNNING"
        ]
        with self.assertRaisesRegex(strict_contract.ContractError, "lifecycle"):
            contract.validate_offline_policy(policy)

        policy = copy.deepcopy(self.registered.policy)
        policy["cross_route_contract"][
            "cross_route_single_use_guaranteed"
        ] = True
        with self.assertRaisesRegex(strict_contract.ContractError, "cross-route"):
            contract.validate_offline_policy(policy)

    def test_requires_exact_three_raw_sources(self) -> None:
        mutations = []
        wrong_path = self.bindings()
        wrong_path["source_bindings"]["diagnostic_master"]["path"] = "data/other.csv"
        mutations.append((wrong_path, "path differs"))
        wrong_hash = self.bindings()
        wrong_hash["source_bindings"]["p_action_artifact"]["observed_sha256"] = "1" * 64
        mutations.append((wrong_hash, "hash differs"))
        zero_size = self.bindings()
        zero_size["source_bindings"]["official_payoff_source"]["observed_byte_size"] = 0
        mutations.append((zero_size, "byte size"))
        missing = self.bindings()
        del missing["source_bindings"]["official_payoff_source"]
        mutations.append((missing, "exact three"))
        for bindings, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(
                strict_contract.ContractError, message
            ):
                contract.compile_offline_run_scope(self.registered, bindings)

    def test_requires_two_distinct_fixed_projections(self) -> None:
        bindings = self.bindings()
        bindings["projection_bindings"]["candidate_projection"]["path"] = (
            "outputs/other.jsonl"
        )
        with self.assertRaisesRegex(strict_contract.ContractError, "path differs"):
            contract.compile_offline_run_scope(self.registered, bindings)

        bindings = self.bindings()
        bindings["projection_bindings"]["settlement_projection"]["sha256"] = "d" * 64
        with self.assertRaisesRegex(strict_contract.ContractError, "must be distinct"):
            contract.compile_offline_run_scope(self.registered, bindings)

    def test_materialization_manifest_path_hash_and_size_are_bound(self) -> None:
        bindings = self.bindings()
        bindings["materialization_manifest"]["path"] = "outputs/manifest.json"
        with self.assertRaisesRegex(strict_contract.ContractError, "path differs"):
            contract.compile_offline_run_scope(self.registered, bindings)
        bindings = self.bindings()
        bindings["materialization_manifest"]["sha256"] = "not-a-sha"
        with self.assertRaisesRegex(strict_contract.ContractError, "SHA"):
            contract.compile_offline_run_scope(self.registered, bindings)
        bindings = self.bindings()
        bindings["materialization_manifest"]["byte_size"] = 0
        with self.assertRaisesRegex(strict_contract.ContractError, "byte size"):
            contract.compile_offline_run_scope(self.registered, bindings)

    def test_runtime_material_bundle_is_exact_and_complete(self) -> None:
        bindings = self.bindings()
        del bindings["runtime_material_sha256"]["runner_blob_sha256"]
        with self.assertRaisesRegex(strict_contract.ContractError, "material bundle"):
            contract.compile_offline_run_scope(self.registered, bindings)
        bindings = self.bindings()
        bindings["runtime_material_sha256"]["runner_blob_sha256"] = "0" * 64
        with self.assertRaisesRegex(strict_contract.ContractError, "material bundle"):
            contract.compile_offline_run_scope(self.registered, bindings)

    def test_bindings_reject_extra_keys_and_nonfinite_values(self) -> None:
        bindings = self.bindings()
        bindings["unregistered_threshold"] = 0.3
        with self.assertRaisesRegex(strict_contract.ContractError, "extra"):
            contract.compile_offline_run_scope(self.registered, bindings)
        bindings = self.bindings()
        bindings["source_bindings"]["diagnostic_master"]["observed_byte_size"] = float(
            "nan"
        )
        with self.assertRaisesRegex(strict_contract.ContractError, "non-finite"):
            contract.compile_offline_run_scope(self.registered, bindings)


if __name__ == "__main__":
    unittest.main()
