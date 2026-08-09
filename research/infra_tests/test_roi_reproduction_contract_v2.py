from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import infrastructure_safety_contract as infrastructure_contract
import roi_reproduction_contract_v2 as contract
import scope_contract as legacy_contract


POLICY_PATH = ROOT / contract.POLICY_PATH
HASHES = {
    name: character * 64
    for name, character in {
        "policy": "1",
        "schema": "2",
        "prompt": "3",
        "context": "4",
        "response": "5",
        "catalog_release": "6",
        "catalog_entry": "7",
        "manifest": "8",
        "artifact": "9",
    }.items()
}


def ref(path: str, digest: str = HASHES["manifest"]) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def sample_proposal() -> dict[str, Any]:
    catalog_refs = {
        name: ref(f"research/synthetic/roi_reproduction/{name}.json")
        for name in (
            "column_policy",
            "feature_lineage",
            "target_label",
            "fold_design",
            "runner_universe_policy",
            "model_recipe",
            "canonicalization",
            "environment_template",
            "reference_artifact_contract",
        )
    }
    model_recipe_ref = catalog_refs["model_recipe"]
    return {
        "schema_version": 2,
        "experiment_id": "M0-REPRO-FIXTURE",
        "gate_kind": contract.GATE_KIND,
        "gate_contract_version": contract.GATE_CONTRACT_VERSION,
        "execution_kind": contract.EXECUTION_KIND,
        "policy_ref": ref(contract.POLICY_PATH, HASHES["policy"]),
        "schema_ref": ref(contract.SCHEMA_PATHS["proposal"], HASHES["schema"]),
        "title": "Frozen M0 historical reproduction",
        "hypothesis": "The frozen M0 recipe can be reconstructed deterministically.",
        "null_hypothesis": "The frozen M0 recipe cannot be reconstructed deterministically.",
        "racing_mechanism": "Frozen runner-strength summaries reproduce the historical output.",
        "non_promotion_purpose": "Reconstruction audit only; no model adoption authority.",
        "brain_provenance": {
            "source_kind": "external_ai",
            "provider": "OpenAI",
            "model_id": "manual-review-model",
            "transfer_mode": "manual",
            "prompt_sha256": HASHES["prompt"],
            "sanitized_context_manifest_ref": ref(
                "research/synthetic/roi_reproduction/context_manifest.json",
                HASHES["context"],
            ),
            "response_sha256": HASHES["response"],
        },
        "repository_contract": {
            "repository": "kazuponbaseball-cell/keiba_ai_project",
            "base_branch": "main",
            "base_commit": "a" * 40,
        },
        "target_population": "Frozen historical M0 runner universe",
        "in_scope": ["deterministic reconstruction", "probability contract"],
        "out_of_scope": ["model adoption", "production changes"],
        "expected_changed_paths": [
            "research/synthetic/roi_reproduction/test_fixture.py",
            "scripts/research/roi_reproduction_experiments/fixture.py",
        ],
        "raw_data_sources": ["legacy_m0_snapshot"],
        "data_as_of": "2026-08-09T00:00:00Z",
        "allowed_columns": [
            "critical_missing_count",
            "min_primary_strength",
            "min_rank_strength",
            "sum_primary_strength",
            "sum_rank_strength",
        ],
        "forbidden_columns": ["odds", "payoff", "popularity", "price", "roi"],
        "lineage_hash_requirements": ["sha256", "source_as_of"],
        "chronological_fold_design": {
            "partition_unit": "race_id",
            "ordered_partitions": [
                "train",
                "validation",
                "calibration",
                "reused_historical_test",
            ],
            "reused_development_test": True,
            "race_overlap_allowed": False,
            "prospective_outer_oos": False,
        },
        "fold_manifest_ref": ref(
            "research/synthetic/roi_reproduction/fold_manifest.json"
        ),
        "purge_embargo": {"purge_days": 1, "embargo_days": 1},
        "reference_catalog_contract_refs": catalog_refs,
        "reference_catalog_entry_ref": {
            "catalog_release_id": "M0-CATALOG-RELEASE",
            "catalog_release_digest": HASHES["catalog_release"],
            "entry_id": "M0-CATALOG-ENTRY",
            "entry_digest": HASHES["catalog_entry"],
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        },
        "model_identity_contract": {
            "legacy_reference_status": "CATALOG_RESOLVED_EXACTLY_ONE",
            "identity_domain_resolution_count": 1,
            "legacy_run_mode": "core",
            "canonical_model_name": "M0_frozen_recipe",
            "canonical_probability_stage": "M0_raw",
            "comparison_artifact_ids": ["full_probability", "model_state"],
            "model_recipe_manifest_ref": model_recipe_ref,
            "post_reference_changes_allowed": False,
            "challenger_present": False,
        },
        "primary_metric": {
            "name": "m0_reproduction_equivalence",
            "comparison": "canonical_digest_or_preregistered_numeric_tolerance",
            "uses_roi": False,
            "promotion_metric": False,
        },
        "secondary_metrics": ["canonical_digest", "numeric_tolerance"],
        "required_effect": {
            "two_clean_replicas_required": True,
            "probability_contract_required": True,
            "canonical_reference_required_for_reproduced": True,
            "strategy_score_credit": 0,
            "promotion_authority": False,
            "roi_calculations": 0,
        },
        "rejection_gate": ["digest_mismatch", "probability_contract_failure"],
        "stop_conditions": ["catalog_binding_failure", "contract_failure"],
        "compute_budget": {
            "maximum_runtime_minutes": 60,
            "replica_count": 2,
            "network_calls": 0,
            "external_api_calls": 0,
            "roi_calculations": 0,
        },
        "allowed_variant_count": 1,
        "allowed_threshold_search_count": 0,
        "score_components": dict(contract.M0_SCORE_COMPONENTS),
        "eligibility_contract": {
            "eligibility_class": "nonpromotion_reproduction_audit",
            "strategy_score_applicable": False,
            "recorded_strategy_score": 23,
            "strategy_score_credit": 0,
            "score_threshold_override_allowed": False,
            "non_promotion_only": True,
            "success_grants_shadow": False,
            "success_grants_model_change": False,
            "success_grants_buy": False,
        },
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "safety": dict(contract.G1_CAPABILITIES),
    }


def sample_run(proposal: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    manifest = lambda name: ref(  # noqa: E731 - compact fixture constructor
        f"research/synthetic/roi_reproduction/{name}.json"
    )
    return {
        "schema_version": 2,
        "experiment_id": proposal["experiment_id"],
        "gate_kind": contract.GATE_KIND,
        "gate_contract_version": contract.GATE_CONTRACT_VERSION,
        "execution_kind": contract.EXECUTION_KIND,
        "policy_ref": ref(contract.POLICY_PATH, HASHES["policy"]),
        "schema_ref": ref(contract.SCHEMA_PATHS["run"], HASHES["schema"]),
        "proposal_scope": proposal,
        "proposal_scope_digest": contract.canonical_digest_v2(proposal),
        "reference_catalog_entry_ref": proposal["reference_catalog_entry_ref"],
        "execution_commit_sha": "b" * 40,
        "config_refs": [manifest("config")],
        "data_input_manifest_refs": [manifest("data_input_manifest")],
        "source_lineage_manifest_ref": manifest("source_lineage_manifest"),
        "fold_manifest_ref": proposal["fold_manifest_ref"],
        "runner_universe_manifest_ref": manifest("runner_universe_manifest"),
        "feature_lineage_manifest_ref": manifest("feature_lineage_manifest"),
        "target_label_manifest_ref": manifest("target_label_manifest"),
        "model_recipe_manifest_ref": proposal["model_identity_contract"][
            "model_recipe_manifest_ref"
        ],
        "reference_artifact_manifest_ref": manifest("reference_artifact_manifest"),
        "canonicalization_manifest_ref": manifest("canonicalization_manifest"),
        "dependency_environment_manifest_ref": manifest("environment_manifest"),
        "seed": 20260809,
        "exact_execution_commands": [
            {
                "template_id": "m0_reproduction",
                "arguments": [
                    {"name": "replicas", "value_type": "integer", "value": 2},
                    {"name": "fixture_only", "value_type": "boolean", "value": False},
                ],
            }
        ],
        "replicate_contract": {
            "replicate_count": 2,
            "replica_ids": ["clean_a", "clean_b"],
            "same_execution_commit": True,
            "same_run_scope_digest": True,
            "same_input_hashes": True,
            "same_environment_hash": True,
            "isolated_checkouts": True,
            "shared_mutable_cache": False,
            "network_calls": 0,
            "digest_algorithm": "sha256",
            "crash_automatic_retry": False,
        },
        "conditional_policy_refs": {},
        "capabilities": copy.deepcopy(
            policy["phase_capabilities"]["historical_reproduction"]["flags"]
        ),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def schema_object_node(schema: dict[str, Any]) -> dict[str, Any]:
    candidates = [schema, *schema.get("allOf", []), *schema.get("oneOf", [])]
    matches = [
        item
        for item in candidates
        if isinstance(item, dict) and "required" in item and "properties" in item
    ]
    signatures = {
        (frozenset(item["required"]), frozenset(item["properties"])) for item in matches
    }
    if not matches or len(signatures) != 1:
        raise AssertionError("schema does not have one unambiguous top-level object contract")
    return matches[0]


class RoiReproductionContractV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy, cls.policy_raw_sha256 = contract.load_policy_v2(POLICY_PATH)

    def test_strict_json_and_canonical_float_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            contract.strict_json_loads_v2('{"a":1,"a":2}')
        for raw in ('{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "non-standard"):
                contract.strict_json_loads_v2(raw)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                contract.canonical_json_bytes_v2({"value": value})

        self.assertEqual(
            contract.canonical_json_text_v2({"z": "日本語", "a": -0.0}),
            '{"a":0.0,"z":"日本語"}',
        )
        self.assertEqual(contract.canonical_float64_hex(-0.0), "0000000000000000")
        self.assertEqual(contract.canonical_float64_hex(1.0), "3ff0000000000000")
        with self.assertRaises(ValueError):
            contract.canonical_float64_hex(True)

    def test_policy_load_is_non_authoritative_and_hashes_exact_bytes(self) -> None:
        expected_hash = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()
        self.assertEqual(self.policy_raw_sha256, expected_hash)
        self.assertFalse(self.policy["authority"])
        self.assertEqual(self.policy["implementation_layer"], "G1_CONTRACT_COMPILER_ONLY")
        self.assertEqual(self.policy["implementation_status"], "EXECUTION_FORBIDDEN")
        self.assertEqual(self.policy["safety"], contract.G1_CAPABILITIES)
        self.assertEqual(tuple(self.policy["capability_field_set"]), contract.CAPABILITY_FIELDS)

        for field in ("automatic_execution", "price_access", "roi_calculation", "credential_access"):
            mutated = copy.deepcopy(self.policy)
            mutated["phase_capabilities"]["historical_reproduction"]["flags"][field] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "frozen"):
                contract.normalize_policy_v2(mutated)
        mutated_mode = copy.deepcopy(self.policy)
        mutated_mode["phase_capabilities"]["historical_reproduction"]["mode"] = "anything"
        with self.assertRaisesRegex(ValueError, "frozen"):
            contract.normalize_policy_v2(mutated_mode)
        integer_authority = copy.deepcopy(self.policy)
        integer_authority["authority"] = 0
        with self.assertRaises(ValueError):
            contract.normalize_policy_v2(integer_authority)

    def test_proposal_normalizes_but_remains_blocked_catalog(self) -> None:
        proposal = contract.normalize_proposal_v2(sample_proposal(), policy=self.policy)
        evaluation = contract.evaluate_proposal_v2(proposal, policy=self.policy)
        self.assertEqual(set(proposal), contract.PROPOSAL_FIELDS)
        self.assertEqual(sum(proposal["score_components"].values()), 23)
        self.assertEqual(evaluation["status"], "BLOCKED_CATALOG")
        self.assertFalse(evaluation["proposal_or_queue_creation_allowed"])
        self.assertFalse(evaluation["execution_authority"])
        self.assertEqual(evaluation["capabilities"], contract.G1_CAPABILITIES)

    def test_proposal_fails_closed_on_score_catalog_market_path_and_capability_drift(self) -> None:
        cases: list[tuple[str, Any]] = []

        changed_score = sample_proposal()
        changed_score["score_components"]["independent_information"] = 1
        cases.append(("score", changed_score))

        missing_catalog_binding = sample_proposal()
        del missing_catalog_binding["reference_catalog_entry_ref"]["entry_id"]
        cases.append(("catalog", missing_catalog_binding))

        market_column = sample_proposal()
        market_column["allowed_columns"].append("odds")
        cases.append(("market", market_column))

        market_scope = sample_proposal()
        market_scope["in_scope"] = ["odds_or_popularity_or_market_or_payoff_or_roi"]
        cases.append(("market_scope", market_scope))

        forbidden_path = sample_proposal()
        forbidden_path["expected_changed_paths"] = ["config/roi_reproduction.json"]
        cases.append(("path", forbidden_path))

        for marker_path in (
            "research/synthetic/roi_reproduction/odds_reader.py",
            "research/synthetic/roi_reproduction/new_feature.py",
            "research/synthetic/roi_reproduction/recipe_update.py",
            "research/synthetic/roi_reproduction/tolerance_change.py",
            "research/synthetic/roi_reproduction/value_policy_change.py",
        ):
            marked_path = sample_proposal()
            marked_path["expected_changed_paths"] = [marker_path]
            cases.append((f"marked_path:{marker_path}", marked_path))

        for frozen_root in (
            "research/ROI_REPRODUCTION_GATE_V2.json",
            "research/schemas/roi_reproduction_proposal_v2.schema.json",
            "scripts/research/roi_reproduction_contract_v2.py",
            "research/drafts/ROI_REPRODUCTION_GATE_V2_CONTRACT_MAP.design.json",
        ):
            self_amendment = sample_proposal()
            self_amendment["expected_changed_paths"] = [frozen_root]
            cases.append((f"self_amendment:{frozen_root}", self_amendment))

        raw_absolute_path = sample_proposal()
        raw_absolute_path["raw_data_sources"] = ["C:/private/data.csv"]
        cases.append(("raw_source_path", raw_absolute_path))

        raw_market_source = sample_proposal()
        raw_market_source["raw_data_sources"] = ["odds"]
        cases.append(("raw_market_source", raw_market_source))

        roi_metric = sample_proposal()
        roi_metric["secondary_metrics"] = ["roi"]
        cases.append(("roi_metric", roi_metric))

        payoff_rejection = sample_proposal()
        payoff_rejection["rejection_gate"] = ["payoff_drop"]
        cases.append(("payoff_rejection", payoff_rejection))

        production_scope = sample_proposal()
        production_scope["in_scope"] = ["production"]
        cases.append(("production_scope", production_scope))

        unsafe_purpose = sample_proposal()
        unsafe_purpose["non_promotion_purpose"] = "production buy ROI"
        cases.append(("unsafe_purpose", unsafe_purpose))

        japanese_market_source = sample_proposal()
        japanese_market_source["raw_data_sources"] = ["オッズ"]
        cases.append(("japanese_market_source", japanese_market_source))

        unsafe_context_ref = sample_proposal()
        unsafe_context_ref["brain_provenance"]["sanitized_context_manifest_ref"] = ref(
            "config/roi_reproduction_context.json"
        )
        cases.append(("unsafe_context_ref", unsafe_context_ref))

        unsafe_fold_ref = sample_proposal()
        unsafe_fold_ref["fold_manifest_ref"] = ref(
            "research/synthetic/roi_reproduction/odds_fold.json"
        )
        cases.append(("unsafe_fold_ref", unsafe_fold_ref))

        recipe_mismatch = sample_proposal()
        recipe_mismatch["model_identity_contract"]["model_recipe_manifest_ref"] = ref(
            "research/synthetic/roi_reproduction/other_model_recipe.json",
            "f" * 64,
        )
        cases.append(("recipe_mismatch", recipe_mismatch))

        capability_flip = sample_proposal()
        capability_flip["safety"]["automatic_execution"] = True
        cases.append(("capability", capability_flip))

        score_laundering = sample_proposal()
        score_laundering["eligibility_contract"]["strategy_score_credit"] = 1
        cases.append(("score_laundering", score_laundering))

        integer_boolean_metric = sample_proposal()
        integer_boolean_metric["primary_metric"]["uses_roi"] = 0
        cases.append(("integer_boolean_metric", integer_boolean_metric))

        integer_boolean_fold = sample_proposal()
        integer_boolean_fold["chronological_fold_design"]["reused_development_test"] = 1
        cases.append(("integer_boolean_fold", integer_boolean_fold))

        for name, proposal in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                contract.normalize_proposal_v2(proposal, policy=self.policy)

    def test_queue_fixture_candidate_has_no_authority(self) -> None:
        proposal = contract.normalize_proposal_v2(sample_proposal(), policy=self.policy)
        queue = contract.build_non_authoritative_queue_v4_candidate(
            proposal=proposal,
            proposal_scope_path=(
                "research/synthetic/roi_reproduction/M0-REPRO-FIXTURE.proposal.json"
            ),
            created_at="2026-08-09T00:00:00Z",
            policy=self.policy,
            policy_ref=ref(contract.POLICY_PATH, HASHES["policy"]),
            schema_ref=ref(contract.SCHEMA_PATHS["queue"], HASHES["schema"]),
            fixture_only=True,
        )
        self.assertEqual(queue["status"], "BLOCKED_CATALOG")
        self.assertEqual(queue["capabilities"], contract.G1_CAPABILITIES)
        self.assertEqual(
            queue["capability_digest"], contract.canonical_digest_v2(contract.G1_CAPABILITIES)
        )
        self.assertFalse(queue["formal_buy"])
        self.assertFalse(queue["send_order"])
        self.assertEqual(queue["stake"], 0)

        with self.assertRaisesRegex(ValueError, "synthetic contract fixtures"):
            contract.build_non_authoritative_queue_v4_candidate(
                proposal=proposal,
                proposal_scope_path=(
                    "research/synthetic/roi_reproduction/M0-REPRO-FIXTURE.proposal.json"
                ),
                created_at="2026-08-09T00:00:00Z",
                policy=self.policy,
                policy_ref=ref(contract.POLICY_PATH, HASHES["policy"]),
                schema_ref=ref(contract.SCHEMA_PATHS["queue"], HASHES["schema"]),
                fixture_only=False,
            )

        unsafe = copy.deepcopy(queue)
        unsafe["capabilities"]["automatic_execution"] = True
        unsafe["capability_digest"] = contract.canonical_digest_v2(unsafe["capabilities"])
        with self.assertRaises(ValueError):
            contract.normalize_queue_v4(unsafe, policy=self.policy)

    def test_run_can_be_normalized_but_is_always_blocked_capability(self) -> None:
        proposal = contract.normalize_proposal_v2(sample_proposal(), policy=self.policy)
        run = sample_run(proposal, self.policy)
        normalized = contract.normalize_run_v2(run, policy=self.policy, proposal=proposal)
        evaluation = contract.evaluate_run_v2(
            normalized,
            policy=self.policy,
            proposal=proposal,
        )
        self.assertEqual(set(normalized), contract.RUN_FIELDS)
        self.assertEqual(evaluation["status"], "BLOCKED_CAPABILITY")
        self.assertEqual(evaluation["implementation_status"], "EXECUTION_FORBIDDEN")
        self.assertFalse(evaluation["execution_allowed"])
        self.assertFalse(evaluation["one_shot_lease_verified"])

        unsafe = copy.deepcopy(run)
        unsafe["capabilities"]["automatic_execution"] = True
        with self.assertRaises(ValueError):
            contract.normalize_run_v2(unsafe, policy=self.policy, proposal=proposal)

        market_command = copy.deepcopy(run)
        market_command["exact_execution_commands"][0]["arguments"].append(
            {"name": "source", "value_type": "string", "value": "odds"}
        )
        with self.assertRaisesRegex(ValueError, "run command"):
            contract.normalize_run_v2(
                market_command, policy=self.policy, proposal=proposal
            )

        unknown_template = copy.deepcopy(run)
        unknown_template["exact_execution_commands"][0]["template_id"] = "caller_selected"
        with self.assertRaisesRegex(ValueError, "code-owned"):
            contract.normalize_run_v2(
                unknown_template, policy=self.policy, proposal=proposal
            )

        multiple_commands = copy.deepcopy(run)
        multiple_commands["exact_execution_commands"].append(
            copy.deepcopy(multiple_commands["exact_execution_commands"][0])
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            contract.normalize_run_v2(
                multiple_commands, policy=self.policy, proposal=proposal
            )

        market_manifest = copy.deepcopy(run)
        market_manifest["data_input_manifest_refs"][0] = ref(
            "research/synthetic/roi_reproduction/odds_manifest.json"
        )
        with self.assertRaisesRegex(ValueError, "manifest reference"):
            contract.normalize_run_v2(
                market_manifest, policy=self.policy, proposal=proposal
            )

        integer_boolean_replica = copy.deepcopy(run)
        integer_boolean_replica["replicate_contract"]["same_execution_commit"] = 1
        with self.assertRaises(ValueError):
            contract.normalize_run_v2(
                integer_boolean_replica, policy=self.policy, proposal=proposal
            )

    def test_catalog_release_rejects_ambiguous_logical_objects(self) -> None:
        def entry(entry_id: str, logical_id: str, digest_character: str) -> dict[str, Any]:
            return {
                "entry_id": entry_id,
                "logical_object_id": logical_id,
                "content_sha256": digest_character * 64,
                "byte_size": 1,
                "schema_sha256": "a" * 64,
                "row_count": 0,
                "source_time": "2026-08-09T00:00:00Z",
                "event_time": "2026-08-09T00:00:00Z",
                "received_time": "2026-08-09T00:00:00Z",
                "data_as_of": "2026-08-09T00:00:00Z",
                "upstream_lineage_sha256": "b" * 64,
                "resolved_model_identity": "M0_frozen_recipe",
                "resolved_input_universe_identity": "frozen_runner_universe",
                "snapshot_id": f"SNAPSHOT-{entry_id}",
                "formal_buy": False,
                "send_order": False,
                "stake": 0,
            }

        def release(records: list[dict[str, Any]]) -> dict[str, Any]:
            normalized_records = []
            for item in records:
                normalized_entry = contract.normalize_catalog_entry_v1(item, policy=self.policy)
                normalized_records.append(
                    {
                        "entry": item,
                        "entry_digest": contract.canonical_digest_v2(normalized_entry),
                    }
                )
            entry_index = sorted(
                (
                    {"entry_id": item["entry"]["entry_id"], "entry_digest": item["entry_digest"]}
                    for item in normalized_records
                ),
                key=lambda item: item["entry_id"],
            )
            return {
                "schema_version": 1,
                "catalog_kind": "roi_reproduction_reference_catalog_v1",
                "catalog_contract_version": 1,
                "release_id": "CATALOG-RELEASE-001",
                "gate_kind": contract.GATE_KIND,
                "gate_contract_version": contract.GATE_CONTRACT_VERSION,
                "policy_ref": ref(contract.POLICY_PATH),
                "schema_ref": ref(contract.SCHEMA_PATHS["catalog_release"]),
                "catalog_publication_scope_digest": "1" * 64,
                "catalog_publication_approval_evidence_digest": "2" * 64,
                "repository_contract": {
                    "repository": "kazuponbaseball-cell/keiba_ai_project",
                    "base_branch": "main",
                    "base_commit": "a" * 40,
                },
                "github_trust_evidence": {"evidence_kind": "fixture", "digest": "3" * 64},
                "provider_contract": {
                    "provider_kind": "reference_metadata_provider",
                    "provider_version": "v1",
                    "execution_commit": "b" * 40,
                    "code_sha256": "4" * 64,
                    "environment_sha256": "5" * 64,
                    "command_template_id": "reference_metadata",
                    "network_required": False,
                    "raw_rows_returned": False,
                },
                "provider_identity": "reference_metadata_provider_v1",
                "provider_execution_commit": "b" * 40,
                "provider_code_sha256": "4" * 64,
                "provider_environment_sha256": "5" * 64,
                "publication_receipt_digest": "6" * 64,
                "catalog_entry_set_digest": contract.canonical_digest_v2(entry_index),
                "logical_entries": normalized_records,
                "lineage_edges": [],
                "created_at": "2026-08-09T00:00:00Z",
                "capabilities": dict(contract.G1_CAPABILITIES),
                "formal_buy": False,
                "send_order": False,
                "stake": 0,
            }

        first = entry("ENTRY-001", "legacy_m0_snapshot", "7")
        contract.normalize_catalog_release_v1(release([first]), policy=self.policy)
        alias = entry("ENTRY-002", "LEGACY_M0_SNAPSHOT", "8")
        with self.assertRaisesRegex(ValueError, "ambiguous duplicate logical object"):
            contract.normalize_catalog_release_v1(
                release([first, alias]), policy=self.policy
            )

    def test_schema_top_level_exact_fields_match_normalizers(self) -> None:
        contracts = {
            "proposal": contract.PROPOSAL_FIELDS,
            "run": contract.RUN_FIELDS,
            "result": contract.RESULT_FIELDS,
            "review": contract.REVIEW_FIELDS,
            "queue": contract.QUEUE_FIELDS,
            "catalog_publication_scope": contract.CATALOG_PUBLICATION_SCOPE_FIELDS,
            "catalog_publication_provider_lease": contract.CATALOG_PROVIDER_LEASE_FIELDS,
            "catalog_release": contract.CATALOG_RELEASE_FIELDS,
            "catalog_entry": contract.CATALOG_ENTRY_FIELDS,
            "catalog_publication_receipt": contract.CATALOG_PUBLICATION_RECEIPT_FIELDS,
            "lease_operation_record": contract.LEASE_OPERATION_RECORD_COMMON_FIELDS,
            "catalog_publication_event": contract.CATALOG_EVENT_FIELDS,
            "catalog_release_status_event": contract.RELEASE_STATUS_EVENT_FIELDS,
            "registry_event": contract.REGISTRY_EVENT_FIELDS,
        }
        for schema_kind, expected_fields in contracts.items():
            with self.subTest(schema_kind=schema_kind):
                path = ROOT / contract.SCHEMA_PATHS[schema_kind]
                schema = contract.strict_json_load_v2(path)
                node = schema_object_node(schema)
                self.assertIs(node.get("additionalProperties"), False)
                self.assertEqual(set(node["required"]), set(expected_fields))
                self.assertEqual(set(node["properties"]), set(expected_fields))

    def test_schema_keywords_and_internal_refs_are_resolvable(self) -> None:
        allowed_keywords = {
            "$schema", "$id", "$ref", "$defs", "title", "description",
            "type", "const", "enum", "minimum", "maximum", "exclusiveMinimum",
            "exclusiveMaximum", "minLength", "maxLength", "pattern", "format",
            "minItems", "maxItems", "uniqueItems", "items", "prefixItems",
            "contains", "minContains", "maxContains", "properties", "patternProperties", "required",
            "additionalProperties", "propertyNames", "minProperties", "maxProperties",
            "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
            "dependentRequired", "dependentSchemas",
        }

        def visit(node: Any, *, root: dict[str, Any], label: str) -> None:
            if isinstance(node, bool):
                return
            self.assertIsInstance(node, dict, label)
            unknown = set(node) - allowed_keywords
            self.assertFalse(unknown, f"{label} has unknown schema keywords: {sorted(unknown)}")
            reference = node.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.removeprefix("#/$defs/")
                self.assertIn(name, root.get("$defs", {}), f"{label} has unresolved ref {reference}")
            for container_key in ("properties", "patternProperties", "$defs", "dependentSchemas"):
                for name, child in node.get(container_key, {}).items():
                    visit(child, root=root, label=f"{label}.{container_key}.{name}")
            for child_key in (
                "items", "contains", "not", "if", "then", "else",
                "additionalProperties", "propertyNames",
            ):
                child = node.get(child_key)
                if isinstance(child, (dict, bool)):
                    visit(child, root=root, label=f"{label}.{child_key}")
            for list_key in ("prefixItems", "allOf", "anyOf", "oneOf"):
                for index, child in enumerate(node.get(list_key, [])):
                    visit(child, root=root, label=f"{label}.{list_key}[{index}]")

        for kind, path in contract.SCHEMA_PATHS.items():
            schema = contract.strict_json_load_v2(ROOT / path)
            with self.subTest(kind=kind):
                visit(schema, root=schema, label=kind)

    def test_schema_union_branch_fields_match_phase_and_receipt_normalizers(self) -> None:
        entry_ref_schema = contract.strict_json_load_v2(
            ROOT / contract.SCHEMA_PATHS["catalog_entry_ref"]
        )
        entry_ref_fields = {
            "catalog_release_id",
            "catalog_release_digest",
            "entry_id",
            "entry_digest",
            "formal_buy",
            "send_order",
            "stake",
        }
        self.assertEqual(set(entry_ref_schema["required"]), entry_ref_fields)
        self.assertEqual(set(entry_ref_schema["properties"]), entry_ref_fields)
        self.assertIs(entry_ref_schema["additionalProperties"], False)

        lease_schema = contract.strict_json_load_v2(
            ROOT / contract.SCHEMA_PATHS["execution_lease"]
        )
        observed_lease_fields = {
            frozenset(branch["required"]) for branch in lease_schema["oneOf"]
        }
        expected_lease_fields = {
            frozenset(contract.EXECUTION_LEASE_COMMON_FIELDS | phase_fields)
            for phase_fields in contract.EXECUTION_LEASE_PHASE_FIELDS.values()
        }
        self.assertEqual(observed_lease_fields, expected_lease_fields)
        for branch in lease_schema["oneOf"]:
            self.assertIs(branch["additionalProperties"], False)
            self.assertEqual(set(branch["properties"]), set(branch["required"]))

        receipt_schema = contract.strict_json_load_v2(
            ROOT / contract.SCHEMA_PATHS["durable_ledger_receipt"]
        )
        observed_receipt_fields = {
            frozenset(branch["required"]) for branch in receipt_schema["oneOf"]
        }
        expected_receipt_fields = {
            frozenset(fields)
            for fields in (
                contract.ACTIVATION_RECEIPT_FIELDS,
                contract.EXECUTION_ISSUE_RECEIPT_FIELDS,
                contract.EXECUTION_CONSUMPTION_RECEIPT_COMMON_FIELDS,
                contract.CATALOG_PROVIDER_ISSUE_RECEIPT_FIELDS,
                contract.CATALOG_PROVIDER_CONSUMPTION_RECEIPT_FIELDS,
            )
        }
        self.assertEqual(observed_receipt_fields, expected_receipt_fields)
        for branch in receipt_schema["oneOf"]:
            self.assertIs(branch["additionalProperties"], False)
            self.assertEqual(set(branch["properties"]), set(branch["required"]))

    def test_legacy_and_infrastructure_raw_bytes_and_digests_are_unchanged(self) -> None:
        raw_hashes = {
            "research/REGISTRY.jsonl": (
                "58ce1b54bf41c375b6df493bffb10ae5612875d2f47dd55ecd9c93ff6d29016c"
            ),
            "research/INFRASTRUCTURE_GATE.json": (
                "22126a1fb46aead49fc56465e567d009e16d85001824ea0a90a0bc5a2fe7da22"
            ),
            "research/scopes/EXP-20260808-030.proposal.json": (
                "8ae4ebd6329b4df84b0ffa70c5cd1117e7e4aa0dd0493f66ed7bf1b1faa30f04"
            ),
            "research/scopes/EXP-20260808-030.run.json": (
                "4304eee094d02a6fbce09966972b36683a5fda0fb1534b637460a91b62e8882b"
            ),
            "research/scopes/EXP-20260808-031.proposal.json": (
                "7fc629acf0d50edc7accfed93a894d5e841a1ebb4ea3a4d8fa534c39bb58f97e"
            ),
            "research/scopes/EXP-20260808-031.run.json": (
                "913dff2ab94c779306a89e9cff3b14831d0ceacc966daa9a6b015da846591beb"
            ),
        }
        for relative, expected in raw_hashes.items():
            with self.subTest(relative=relative):
                self.assertEqual(
                    hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected
                )

        canonical_digests = {
            "EXP-20260808-030": (
                "890a242b6a14485e233473c96342cfbe66ff3f09178f546a50f2d37f93ab3610",
                "8ef37a63b165c7d1b41b65a3a331c311bfe5957b659921317ca1128d2322bd31",
            ),
            "EXP-20260808-031": (
                "3f97b3d9c57a79ebbcc91746d6e5f27a37253395095019658d49aa5389672410",
                "5aac4066bd3839509990369998568bb8d30df77b2391e11cbf141f980c837804",
            ),
        }
        for experiment_id, (proposal_digest, run_digest) in canonical_digests.items():
            with self.subTest(experiment_id=experiment_id):
                proposal_path = ROOT / "research" / "scopes" / f"{experiment_id}.proposal.json"
                run_path = ROOT / "research" / "scopes" / f"{experiment_id}.run.json"
                proposal = legacy_contract.normalize_proposal_scope(
                    legacy_contract.strict_json_load(proposal_path),
                    expected_experiment_id=experiment_id,
                )
                run = legacy_contract.normalize_run_scope(
                    legacy_contract.strict_json_load(run_path),
                    proposal_scope=proposal,
                )
                self.assertEqual(legacy_contract.canonical_digest(proposal), proposal_digest)
                self.assertEqual(legacy_contract.canonical_digest(run), run_digest)

        self.assertEqual(infrastructure_contract.QUEUE_SCHEMA_VERSION, 3)
        self.assertEqual(infrastructure_contract.EVENT_SCHEMA_VERSION, 3)
        self.assertEqual(infrastructure_contract.GATE_KIND, "infrastructure_safety_v1")


if __name__ == "__main__":
    unittest.main()
