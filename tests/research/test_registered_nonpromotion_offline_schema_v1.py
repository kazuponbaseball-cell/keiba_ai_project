from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
TEST_DIR = ROOT / "tests" / "research"
for directory in (SCRIPT_DIR, TEST_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import registered_nonpromotion_contract_v1 as strict_contract
import registered_nonpromotion_offline_contract_v1 as contract
from test_registered_nonpromotion_schema_v1 import (
    SchemaValidationError,
    _strict_json,
    _validate,
)


SCHEMA_PATHS = {
    key: ROOT / value for key, value in contract.EXPECTED_SCHEMA_PATHS.items()
}


class RegisteredNonpromotionOfflineSchemaV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schemas = {key: _strict_json(path) for key, path in SCHEMA_PATHS.items()}
        cls.registered = contract.resolve_offline_registered_recipe(ROOT)

    def bindings(self) -> dict[str, object]:
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
            "source_bindings": {
                role: {
                    "path": value["path"],
                    "expected_sha256": value["expected_sha256"],
                    "observed_sha256": value["expected_sha256"],
                    "observed_byte_size": 100 + index,
                }
                for index, (role, value) in enumerate(contract.SOURCE_INPUTS.items())
            },
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
            "python_minor_version": "3.12",
            "numpy_version": "2.4.3",
            "environment_manifest_sha256": "f" * 64,
            "output_root": contract.FIXED_OUTPUT_ROOT,
            "sealed_at": "2026-08-15T01:00:00Z",
        }

    def approval_evidence(self, run_scope_digest: str) -> dict[str, object]:
        base = "a" * 40
        body = f"{contract.APPROVAL_KEYWORD} {run_scope_digest}"
        materials = [
            {
                "path": path,
                "ref": base,
                "blob_sha": f"{index + 1:040x}",
                "content_sha256": f"{index + 1:064x}",
            }
            for index, path in enumerate(contract.RUNTIME_MATERIAL_PATHS.values())
        ]
        return {
            "schema_version": 1,
            "gate_kind": contract.GATE_KIND,
            "run_scope_digest": run_scope_digest,
            "verification_checkpoint": "INITIAL_APPROVAL",
            "original_evidence_digest": None,
            "github_trust": {
                "repository_full_name": contract.DEFAULT_REPOSITORY,
                "base_branch": "main",
                "verified_current_main_sha": base,
                "verified_base_commit": base,
                "compare_url": (
                    "https://api.github.com/repos/kazuponbaseball-cell/"
                    f"keiba_ai_project/compare/{base}...{base}"
                ),
                "compare_status": "identical",
                "merge_base_sha": base,
                "approvers_blob_sha": "b" * 40,
                "approvers_content_sha256": "c" * 64,
                "verification_time": "2026-08-15T01:01:00Z",
            },
            "runtime_materials": {
                "run_scope_base_commit": base,
                "materials": materials,
                "schema_bundle_sha256": "d" * 64,
                "evidence_digest": "e" * 64,
            },
            "ordinary_registry": {
                "path": "research/REGISTRY.jsonl",
                "ref": base,
                "blob_sha": "c" * 40,
                "content_sha256": "f" * 64,
                "scanned_nonblank_line_count": 10,
                "used_grant_comment_id_count": 6,
                "target_comment_id": 123,
                "target_comment_id_unused": True,
            },
            "comment": {
                "approval_type": contract.APPROVAL_KEYWORD,
                "approval_digest": run_scope_digest,
                "repository": contract.DEFAULT_REPOSITORY,
                "issue_number": 44,
                "comment_id": 123,
                "url": (
                    "https://github.com/kazuponbaseball-cell/keiba_ai_project/"
                    "issues/44#issuecomment-123"
                ),
                "author": "human-reviewer",
                "author_type": "User",
                "body": body,
                "body_sha256": strict_contract.canonical_digest(body),
                "created_at": "2026-08-15T01:01:00Z",
                "updated_at": "2026-08-15T01:01:00Z",
            },
            "limitations": {
                "single_use_policy": "ONE_ACCEPTED_EXECUTION",
                "single_use_enforcement": "BEST_EFFORT_LOCAL_EXCLUSIVE_RECEIPT",
                "global_replay_proof": False,
                "rollback_resistant": False,
                "durable_remote_ledger": False,
                "network_isolation": "APPLICATION_LEVEL_NOT_OS_SANDBOX",
            },
            "implementation_current_main_ancestry_verified": True,
            "authority": False,
            "local_offline_permission": True,
            "global_uniqueness_guaranteed": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
            "evidence_digest": "9" * 64,
        }

    @staticmethod
    def metric_projection() -> dict[str, object]:
        def arm(name: str) -> dict[str, object]:
            return {
                "arm": name,
                "bet_count": 1,
                "hit_count": 0,
                "stake_denominator_yen": 100.0,
                "return_yen": 0.0,
                "profit_yen": -100.0,
                "roi_percent": 0.0,
            }

        return {
            "enrolled_race_count": 3746,
            "d0": arm("D0"),
            "d1": arm("D1"),
            "sum_delta_profit_yen": 0.0,
            "mean_delta_profit_yen_per_enrolled_race": 0.0,
            "decision_disagreement_count": 0,
        }

    def result(self, scope: dict[str, object]) -> dict[str, object]:
        primary = self.metric_projection()
        scientific = {
            "primary": primary,
            "sensitivity": {
                "common_top1_return_zeroed": copy.deepcopy(primary),
                "common_top3_return_zeroed": copy.deepcopy(primary),
                "common_2000_yen_winsor": copy.deepcopy(primary),
                "top1_race_ids": ["2025000000000001"],
                "top3_race_ids": [
                    "2025000000000001",
                    "2025000000000002",
                    "2025000000000003",
                ],
            },
            "bootstrap": {
                "cluster_count": 10,
                "replicates": 100000,
                "seed": 20260814,
                "rng": "numpy.random.Generator(PCG64)",
                "mean": 0.0,
                "one_sided_95_lower_bound": 0.0,
                "distribution_digest": "1" * 64,
            },
            "candidate_projection_digest": "2" * 64,
            "decision_vector_digest": "3" * 64,
            "settlement_projection_digest": "4" * 64,
            "paired_rows_digest": "5" * 64,
            "contract_status": "VALID",
        }
        scientific_digest = strict_contract.canonical_digest(scientific)
        return {
            "schema_version": 1,
            "gate_kind": contract.GATE_KIND,
            "lifecycle_state": "RNOD_COMPLETED",
            "run_scope_digest": scope["run_scope_digest"],
            "semantic_subject_digest": scope["semantic_subject_digest"],
            "exact_subject_digest": scope["exact_subject_digest"],
            "recipe_id": contract.RECIPE_ID,
            "recipe_version": 1,
            "recipe_digest": contract.EXPECTED_RECIPE_DIGEST,
            "approval_evidence_digest": "6" * 64,
            "source_bindings_digest": "7" * 64,
            "projection_bindings_digest": "8" * 64,
            "decision_freeze_receipt_digest": "9" * 64,
            "scientific_projection": scientific,
            "scientific_projection_digest": scientific_digest,
            "computed_outcome": "NO_DECISION_EFFECT",
            "replica_mode": "LOGICAL_SAME_PROCESS_SHARED_SEALED_INPUT_BYTES",
            "replica_attempt_count": 1,
            "replica_scientific_projection_digests": {
                "clean_a": scientific_digest,
                "clean_b": scientific_digest,
            },
            "replica_semantic_equality": True,
            "authority": False,
            "single_use_policy": "ONE_ACCEPTED_EXECUTION",
            "single_use_enforcement": "BEST_EFFORT_LOCAL_EXCLUSIVE_RECEIPT",
            "global_replay_proof": False,
            "rollback_resistant": False,
            "durable_remote_ledger": False,
            "network_isolation": "APPLICATION_LEVEL_NOT_OS_SANDBOX",
            "evidence_purpose_class": "DIAGNOSTIC_NONPROMOTION",
            "source_authority_class": "B_LOCAL_HASHED",
            "reused_development_oos": True,
            "strict_t3_rows": 0,
            "confirmatory": False,
            "promotion_eligible": False,
            "score_credit": 0,
            "shadow_transition_supported": False,
            "production_transition_supported": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
            "result_digest": "a" * 64,
        }

    def test_schemas_are_strict_draft7_documents(self) -> None:
        for schema_id, schema in self.schemas.items():
            with self.subTest(schema=schema_id):
                self.assertEqual(
                    schema["$schema"], "http://json-schema.org/draft-07/schema#"
                )
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(
                    set(schema["required"]), set(schema["properties"])
                )
                for name, definition in schema.get("definitions", {}).items():
                    if definition.get("type") == "object":
                        self.assertFalse(
                            definition.get("additionalProperties", True),
                            f"{schema_id}.{name} is not strict",
                        )
                        self.assertEqual(
                            set(definition.get("required", [])),
                            set(definition.get("properties", {})),
                            f"{schema_id}.{name} required/properties differ",
                        )

    def test_policy_and_compiled_scope_validate(self) -> None:
        policy = _strict_json(contract.POLICY_RELATIVE_PATH if contract.POLICY_RELATIVE_PATH.is_absolute() else ROOT / contract.POLICY_RELATIVE_PATH)
        _validate(policy, self.schemas["policy"], self.schemas["policy"])
        scope = contract.compile_offline_run_scope(self.registered, self.bindings())
        _validate(scope, self.schemas["run_scope"], self.schemas["run_scope"])

    def test_approval_and_result_examples_validate(self) -> None:
        scope = contract.compile_offline_run_scope(self.registered, self.bindings())
        approval = self.approval_evidence(str(scope["run_scope_digest"]))
        _validate(
            approval,
            self.schemas["approval_evidence"],
            self.schemas["approval_evidence"],
        )
        result = self.result(scope)
        _validate(result, self.schemas["result"], self.schemas["result"])
        invalid_completed = copy.deepcopy(result)
        invalid_completed["computed_outcome"] = "INVALID"
        with self.assertRaises(SchemaValidationError):
            _validate(
                invalid_completed,
                self.schemas["result"],
                self.schemas["result"],
            )

    def test_all_schemas_reject_extra_fields(self) -> None:
        scope = contract.compile_offline_run_scope(self.registered, self.bindings())
        values = {
            "policy": _strict_json(ROOT / contract.POLICY_RELATIVE_PATH),
            "run_scope": scope,
            "approval_evidence": self.approval_evidence(
                str(scope["run_scope_digest"])
            ),
            "result": self.result(scope),
        }
        for schema_id, value in values.items():
            changed = copy.deepcopy(value)
            changed["unauthorized"] = True
            with self.subTest(schema=schema_id), self.assertRaises(
                SchemaValidationError
            ):
                _validate(changed, self.schemas[schema_id], self.schemas[schema_id])

    def test_nested_authority_and_market_escalations_are_rejected(self) -> None:
        scope = contract.compile_offline_run_scope(self.registered, self.bindings())
        approval = self.approval_evidence(str(scope["run_scope_digest"]))
        approval["limitations"]["durable_remote_ledger"] = True
        with self.assertRaises(SchemaValidationError):
            _validate(
                approval,
                self.schemas["approval_evidence"],
                self.schemas["approval_evidence"],
            )
        result = self.result(scope)
        result["promotion_eligible"] = True
        with self.assertRaises(SchemaValidationError):
            _validate(result, self.schemas["result"], self.schemas["result"])


if __name__ == "__main__":
    unittest.main()
