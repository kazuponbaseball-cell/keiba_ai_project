from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import roi_reproduction_contract_v2 as contract


POLICY_PATH = ROOT / contract.POLICY_PATH
H = {character: character * 64 for character in "123456789abcdef"}
EXPECTED_ARTIFACT_IDS = ["full_probability", "model_state"]
EXPECTED_NUMERIC_CONTRACT = [
    {
        "metric_id": "nll",
        "fold_id": "reused_test",
        "model_name": "M0_frozen_recipe",
        "probability_stage": "M0_raw",
        "unit": "per_runner",
        "reference_value_hex": "3ff0000000000000",
        "absolute_tolerance_hex": "3d719799812dea11",
    }
]


def ref(path: str, digest: str = H["1"]) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def event_safety() -> dict[str, Any]:
    return {
        "event_is_execution_authority": False,
        "automatic_execution_allowed": False,
        "production_approved": False,
        "merge_approved": False,
        "buy_approved": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def actor() -> dict[str, str]:
    return {"actor_kind": "synthetic_fixture", "actor_id": "unit-test"}


def digest_evidence(kind: str, digest: str = H["2"]) -> dict[str, str]:
    return {"evidence_kind": kind, "digest": digest}


def sample_result(policy: dict[str, Any]) -> dict[str, Any]:
    proposal_digest = H["3"]
    run_digest = H["4"]

    def replica(replica_id: str) -> dict[str, Any]:
        return {
            "replica_id": replica_id,
            "commit_sha": "a" * 40,
            "worktree_clean": True,
            "environment_manifest_ref": ref(
                f"research/synthetic/roi_reproduction/{replica_id}_environment.json"
            ),
            "command_digest": H["5"],
            "input_digest_set": [H["6"]],
            "artifact_digests": {
                "full_probability": H["7"],
                "model_state": H["8"],
            },
        }

    return {
        "schema_version": 1,
        "result_kind": "roi_reproduction_result_v1",
        "experiment_id": "M0-REPRO-FIXTURE",
        "gate_kind": contract.GATE_KIND,
        "gate_contract_version": contract.GATE_CONTRACT_VERSION,
        "execution_kind": contract.EXECUTION_KIND,
        "policy_ref": ref(contract.POLICY_PATH),
        "schema_ref": ref(contract.SCHEMA_PATHS["result"]),
        "proposal_scope_digest": proposal_digest,
        "run_scope_digest": run_digest,
        "execution_commit_sha": "a" * 40,
        "computed_outcome": "REPRODUCTION_FAILED",
        "replicas": [replica("clean_a"), replica("clean_b")],
        "determinism_check": {
            "all_equal": True,
            "mismatch_count": 0,
            "artifact_digest_pairs": [
                {
                    "artifact_id": "full_probability",
                    "clean_a_digest": H["7"],
                    "clean_b_digest": H["7"],
                },
                {
                    "artifact_id": "model_state",
                    "clean_a_digest": H["8"],
                    "clean_b_digest": H["8"],
                }
            ],
        },
        "runner_label_contract_check": {
            "passed": True,
            "evidence_ref": ref(
                "research/synthetic/roi_reproduction/runner_label_evidence.json"
            ),
        },
        "probability_contract_check": {
            "passed": True,
            "evidence_ref": ref(
                "research/synthetic/roi_reproduction/probability_evidence.json"
            ),
        },
        "numeric_equivalence": {
            "reference_available": True,
            "all_within_tolerance": False,
            "rows": [
                {
                    "metric_id": "nll",
                    "fold_id": "reused_test",
                    "model_name": "M0_frozen_recipe",
                    "probability_stage": "M0_raw",
                    "unit": "per_runner",
                    "reference_value_hex": "3ff0000000000000",
                    "observed_value_hex": "4000000000000000",
                    "absolute_tolerance_hex": "3d719799812dea11",
                    "pass": False,
                }
            ],
        },
        "artifact_refs": [
            ref("research/synthetic/roi_reproduction/result_manifest.json")
        ],
        "safety": dict(contract.G1_CAPABILITIES),
        "limitations": ["synthetic fixture only"],
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def normalize_sample_result(
    result: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    return contract.normalize_result_v1(
        result,
        policy=policy,
        proposal_scope_digest=result["proposal_scope_digest"],
        run_scope_digest=result["run_scope_digest"],
        expected_experiment_id=result["experiment_id"],
        expected_execution_commit_sha="a" * 40,
        expected_model_name="M0_frozen_recipe",
        expected_probability_stage="M0_raw",
        expected_artifact_ids=EXPECTED_ARTIFACT_IDS,
        expected_numeric_contract=EXPECTED_NUMERIC_CONTRACT,
        expected_reference_available=True,
    )


def initial_registry_event() -> dict[str, Any]:
    capabilities = dict(contract.G1_CAPABILITIES)
    return {
        "schema_version": 4,
        "event_id": "REGISTRY-EVENT-001",
        "global_sequence": 1,
        "experiment_sequence": 1,
        "experiment_id": "M0-REPRO-FIXTURE",
        "gate_kind": contract.GATE_KIND,
        "gate_contract_version": contract.GATE_CONTRACT_VERSION,
        "policy_ref": ref(contract.POLICY_PATH),
        "schema_ref": ref(contract.SCHEMA_PATHS["registry_event"]),
        "status": "PROPOSED",
        "previous_status": None,
        "previous_experiment_event_id": None,
        "previous_experiment_event_digest": None,
        "occurred_at": "2026-08-09T00:00:00Z",
        "observer_actor": actor(),
        "proposal_scope_digest": H["3"],
        "reference_catalog_release_digest": H["4"],
        "reference_catalog_entry_digest": H["5"],
        "run_scope_digest": None,
        "review_digest": None,
        "root_of_trust_evidence": None,
        "github_trust_evidence": None,
        "durable_ledger_evidence": None,
        "legacy_registry_import_evidence": None,
        "approval_grant_evidence": [],
        "revalidated_approval_evidence": [],
        "capabilities": capabilities,
        "capability_digest": contract.canonical_digest_v2(capabilities),
        "execution_kind": contract.EXECUTION_KIND,
        "execution_lease_receipt": None,
        "result_evidence": None,
        "paths": {
            "proposal_scope": (
                "research/synthetic/roi_reproduction/M0-REPRO-FIXTURE.proposal.json"
            ),
            "queue": "research/synthetic/roi_reproduction/M0-REPRO-FIXTURE.queue.json",
            "run_scope": None,
            "result": None,
            "review": None,
        },
        "artifacts": [],
        "notes": ["synthetic fixture only"],
        "safety": event_safety(),
    }


def initial_catalog_event() -> dict[str, Any]:
    capabilities = dict(contract.G1_CAPABILITIES)
    return {
        "schema_version": 1,
        "event_kind": contract.CATALOG_EVENT_KIND,
        "event_id": "CATALOG-EVENT-001",
        "global_sequence": 1,
        "subject_sequence": 1,
        "catalog_publication_scope_id": "CATALOG-SCOPE-001",
        "catalog_publication_scope_digest": H["3"],
        "catalog_gate_kind": contract.CATALOG_GATE_KIND,
        "catalog_contract_version": contract.CATALOG_CONTRACT_VERSION,
        "policy_ref": ref(contract.POLICY_PATH),
        "schema_ref": ref(contract.SCHEMA_PATHS["catalog_publication_event"]),
        "status": "CATALOG_PUBLICATION_SCOPE_PROPOSED",
        "previous_status": None,
        "previous_catalog_event_id": None,
        "previous_catalog_event_digest": None,
        "occurred_at": "2026-08-09T00:00:00Z",
        "observer_actor": actor(),
        "root_of_trust_evidence": None,
        "github_trust_evidence": None,
        "durable_ledger_evidence": None,
        "approval_grant_evidence": [],
        "revalidated_approval_evidence": [],
        "provider_lease_issue_receipt_digest": None,
        "provider_lease_consumption_receipt_digest": None,
        "catalog_publication_receipt_digest": None,
        "reference_catalog_release_id": None,
        "reference_catalog_release_digest": None,
        "capabilities": capabilities,
        "capability_digest": contract.canonical_digest_v2(capabilities),
        "result_evidence": None,
        "artifacts": [],
        "notes": ["synthetic fixture only"],
        "safety": event_safety(),
    }


def initial_release_status_event() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_kind": contract.CATALOG_RELEASE_STATUS_EVENT_KIND,
        "event_id": "RELEASE-STATUS-001",
        "global_sequence": 1,
        "release_status_sequence": 1,
        "reference_catalog_release_id": "CATALOG-RELEASE-001",
        "reference_catalog_release_digest": H["4"],
        "policy_ref": ref(contract.POLICY_PATH),
        "schema_ref": ref(contract.SCHEMA_PATHS["catalog_release_status_event"]),
        "status": "ACTIVE",
        "previous_status": "INITIAL",
        "previous_release_status_event_id": None,
        "previous_release_status_event_digest": None,
        "reason_code": "PUBLICATION_COMPLETED",
        "status_change_evidence_digest": H["5"],
        "effective_at": "2026-08-09T00:00:00Z",
        "observer_actor": actor(),
        "durable_ledger_evidence": digest_evidence("ledger_head", H["6"]),
        "signer_or_attestation_digest": H["7"],
        "notes": ["synthetic fixture only"],
        "safety": event_safety(),
    }


class RoiReproductionLifecycleV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy, _ = contract.load_policy_v2(POLICY_PATH)

    def test_experiment_lifecycle_accepts_only_declared_edges(self) -> None:
        contract.validate_experiment_transition(None, "PROPOSED")
        for previous, targets in contract.EXPERIMENT_TRANSITIONS.items():
            for target in targets:
                with self.subTest(previous=previous, target=target):
                    contract.validate_experiment_transition(previous, target)

        nonterminal = set(contract.EXPERIMENT_STATES) - contract.EXPERIMENT_TERMINAL
        for status in nonterminal:
            self.assertIn("INVALID", contract.EXPERIMENT_TRANSITIONS[status])

    def test_experiment_lifecycle_rejects_skip_self_and_terminal_resurrection(self) -> None:
        cases = (
            (None, "PREPARING"),
            ("PROPOSED", "PREPARING"),
            ("RUNNING", "RUNNING"),
            ("REPRODUCED", "INVALID"),
            ("REJECTED", "PROPOSED"),
            ("INVALID", "PROPOSED"),
        )
        for previous, status in cases:
            with self.subTest(previous=previous, status=status), self.assertRaises(ValueError):
                contract.validate_experiment_transition(previous, status)

    def test_catalog_and_release_lifecycles_reject_skips_and_resurrection(self) -> None:
        contract.validate_catalog_transition(None, "CATALOG_PUBLICATION_SCOPE_PROPOSED")
        for previous, targets in contract.CATALOG_TRANSITIONS.items():
            for target in targets:
                with self.subTest(previous=previous, target=target):
                    contract.validate_catalog_transition(previous, target)
        for previous, status in (
            (None, "CATALOG_PUBLISHING"),
            ("CATALOG_PUBLICATION_SCOPE_PROPOSED", "CATALOG_PUBLISHED"),
            ("CATALOG_PUBLISHING", "CATALOG_PUBLISHING"),
            ("CATALOG_PUBLISHED", "INVALID"),
        ):
            with self.subTest(previous=previous, status=status), self.assertRaises(ValueError):
                contract.validate_catalog_transition(previous, status)

        contract.validate_release_status_transition("INITIAL", "ACTIVE")
        contract.validate_release_status_transition("ACTIVE", "REVOKED")
        for previous, status in (
            ("INITIAL", "REVOKED"),
            ("ACTIVE", "ACTIVE"),
            ("REVOKED", "ACTIVE"),
        ):
            with self.subTest(previous=previous, status=status), self.assertRaises(ValueError):
                contract.validate_release_status_transition(previous, status)

    def test_review_cannot_upgrade_or_rewrite_computed_outcome(self) -> None:
        result = sample_result(self.policy)
        normalized_result = normalize_sample_result(result, self.policy)
        review = {
            "schema_version": 1,
            "experiment_id": normalized_result["experiment_id"],
            "gate_kind": contract.GATE_KIND,
            "gate_contract_version": contract.GATE_CONTRACT_VERSION,
            "execution_kind": contract.EXECUTION_KIND,
            "policy_ref": ref(contract.POLICY_PATH),
            "schema_ref": ref(contract.SCHEMA_PATHS["review"]),
            "proposal_scope_digest": normalized_result["proposal_scope_digest"],
            "run_scope_digest": normalized_result["run_scope_digest"],
            "validated_result_digest": contract.canonical_digest_v2(normalized_result),
            "computed_outcome": "REPRODUCTION_FAILED",
            "proposed_terminal_status": "REPRODUCTION_FAILED",
            "review_limitations": ["synthetic fixture only"],
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        normalized_review = contract.normalize_review_v1(
            review,
            result=normalized_result,
            policy=self.policy,
            proposal_scope_digest=normalized_result["proposal_scope_digest"],
            run_scope_digest=normalized_result["run_scope_digest"],
            expected_experiment_id=normalized_result["experiment_id"],
            expected_execution_commit_sha=normalized_result["execution_commit_sha"],
            expected_model_name="M0_frozen_recipe",
            expected_probability_stage="M0_raw",
            expected_artifact_ids=EXPECTED_ARTIFACT_IDS,
            expected_numeric_contract=EXPECTED_NUMERIC_CONTRACT,
            expected_reference_available=True,
        )
        self.assertEqual(normalized_review["proposed_terminal_status"], "REPRODUCTION_FAILED")

        upgraded = copy.deepcopy(review)
        upgraded["proposed_terminal_status"] = "REPRODUCED"
        with self.assertRaisesRegex(ValueError, "cannot upgrade"):
            contract.normalize_review_v1(
                upgraded,
                result=normalized_result,
                policy=self.policy,
                proposal_scope_digest=normalized_result["proposal_scope_digest"],
                run_scope_digest=normalized_result["run_scope_digest"],
                expected_experiment_id=normalized_result["experiment_id"],
                expected_execution_commit_sha=normalized_result["execution_commit_sha"],
                expected_model_name="M0_frozen_recipe",
                expected_probability_stage="M0_raw",
                expected_artifact_ids=EXPECTED_ARTIFACT_IDS,
                expected_numeric_contract=EXPECTED_NUMERIC_CONTRACT,
                expected_reference_available=True,
            )

        rewritten = copy.deepcopy(review)
        rewritten["computed_outcome"] = "REPRODUCED"
        with self.assertRaisesRegex(ValueError, "cannot rewrite"):
            contract.normalize_review_v1(
                rewritten,
                result=normalized_result,
                policy=self.policy,
                proposal_scope_digest=normalized_result["proposal_scope_digest"],
                run_scope_digest=normalized_result["run_scope_digest"],
                expected_experiment_id=normalized_result["experiment_id"],
                expected_execution_commit_sha=normalized_result["execution_commit_sha"],
                expected_model_name="M0_frozen_recipe",
                expected_probability_stage="M0_raw",
                expected_artifact_ids=EXPECTED_ARTIFACT_IDS,
                expected_numeric_contract=EXPECTED_NUMERIC_CONTRACT,
                expected_reference_available=True,
            )

    def test_result_outcome_is_derived_from_bound_non_roi_evidence(self) -> None:
        reproduced = sample_result(self.policy)
        reproduced["computed_outcome"] = "REPRODUCED"
        row = reproduced["numeric_equivalence"]["rows"][0]
        row["observed_value_hex"] = row["reference_value_hex"]
        row["pass"] = True
        reproduced["numeric_equivalence"]["all_within_tolerance"] = True
        normalize_sample_result(reproduced, self.policy)

        false_contract = copy.deepcopy(reproduced)
        false_contract["probability_contract_check"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "require INVALID"):
            normalize_sample_result(false_contract, self.policy)

        forged_reference_availability = copy.deepcopy(reproduced)
        forged_reference_availability["numeric_equivalence"]["reference_available"] = False
        with self.assertRaisesRegex(ValueError, "trusted catalog evidence"):
            normalize_sample_result(forged_reference_availability, self.policy)

        empty_pairs = copy.deepcopy(reproduced)
        empty_pairs["determinism_check"]["artifact_digest_pairs"] = []
        with self.assertRaisesRegex(ValueError, "non-empty"):
            normalize_sample_result(empty_pairs, self.policy)

        trivial_artifact = copy.deepcopy(reproduced)
        for replica in trivial_artifact["replicas"]:
            del replica["artifact_digests"]["model_state"]
        trivial_artifact["determinism_check"]["artifact_digest_pairs"] = [
            trivial_artifact["determinism_check"]["artifact_digest_pairs"][0]
        ]
        with self.assertRaisesRegex(ValueError, "trusted proposal contract"):
            normalize_sample_result(trivial_artifact, self.policy)

        wrong_commit = copy.deepcopy(reproduced)
        wrong_commit["replicas"][1]["commit_sha"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "identical inputs"):
            normalize_sample_result(wrong_commit, self.policy)

        forged_pass = copy.deepcopy(reproduced)
        forged_pass["numeric_equivalence"]["rows"][0][
            "observed_value_hex"
        ] = "4000000000000000"
        with self.assertRaisesRegex(ValueError, "does not match its numeric values"):
            normalize_sample_result(forged_pass, self.policy)

        rewritten_reference = copy.deepcopy(reproduced)
        rewritten_reference["numeric_equivalence"]["rows"][0][
            "reference_value_hex"
        ] = "4000000000000000"
        rewritten_reference["numeric_equivalence"]["rows"][0]["pass"] = False
        rewritten_reference["numeric_equivalence"]["all_within_tolerance"] = False
        with self.assertRaisesRegex(ValueError, "trusted contract"):
            normalize_sample_result(rewritten_reference, self.policy)

        for name, tolerance in (
            ("infinite", "7ff0000000000000"),
            ("negative", "bff0000000000000"),
        ):
            invalid_tolerance = copy.deepcopy(reproduced)
            invalid_tolerance["numeric_equivalence"]["rows"][0][
                "absolute_tolerance_hex"
            ] = tolerance
            with self.subTest(name=name), self.assertRaises(ValueError):
                normalize_sample_result(invalid_tolerance, self.policy)

        for field, value in (
            ("metric_id", "roi"),
            ("model_name", "challenger_model"),
            ("probability_stage", "M0_temperature_scaled"),
        ):
            wrong_identity = copy.deepcopy(reproduced)
            wrong_identity["numeric_equivalence"]["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize_sample_result(wrong_identity, self.policy)

    def test_registry_event_is_non_authoritative_and_safety_is_exact(self) -> None:
        event = initial_registry_event()
        normalized = contract.normalize_registry_event_v4(event, policy=self.policy)
        self.assertEqual(normalized["safety"], event_safety())
        self.assertEqual(normalized["capabilities"], contract.G1_CAPABILITIES)

        unsafe = copy.deepcopy(event)
        unsafe["safety"]["event_is_execution_authority"] = True
        with self.assertRaises(ValueError):
            contract.normalize_registry_event_v4(unsafe, policy=self.policy)

        integer_false = copy.deepcopy(event)
        integer_false["safety"]["automatic_execution_allowed"] = 0
        with self.assertRaises(ValueError):
            contract.normalize_registry_event_v4(integer_false, policy=self.policy)

        capability_flip = copy.deepcopy(event)
        capability_flip["capabilities"]["automatic_execution"] = True
        capability_flip["capability_digest"] = contract.canonical_digest_v2(
            capability_flip["capabilities"]
        )
        with self.assertRaises(ValueError):
            contract.normalize_registry_event_v4(capability_flip, policy=self.policy)

    def test_approval_statuses_require_exact_grant_evidence_kinds(self) -> None:
        experiment = initial_registry_event()
        experiment.update(
            {
                "event_id": "REGISTRY-EVENT-002",
                "global_sequence": 2,
                "experiment_sequence": 2,
                "status": "APPROVED_TO_PREPARE",
                "previous_status": "PROPOSED",
                "previous_experiment_event_id": "REGISTRY-EVENT-001",
                "previous_experiment_event_digest": H["8"],
            }
        )
        with self.assertRaisesRegex(ValueError, "approval evidence"):
            contract.normalize_registry_event_v4(experiment, policy=self.policy)
        experiment["approval_grant_evidence"] = [
            digest_evidence("APPROVED_TO_PREPARE", H["9"])
        ]
        contract.normalize_registry_event_v4(experiment, policy=self.policy)

        wrong_kind = copy.deepcopy(experiment)
        wrong_kind["approval_grant_evidence"] = [
            digest_evidence("APPROVED_TO_RUN", H["a"])
        ]
        with self.assertRaisesRegex(ValueError, "approval evidence"):
            contract.normalize_registry_event_v4(wrong_kind, policy=self.policy)

        catalog = initial_catalog_event()
        catalog.update(
            {
                "event_id": "CATALOG-EVENT-002",
                "global_sequence": 2,
                "subject_sequence": 2,
                "status": "APPROVED_TO_PUBLISH_REFERENCE_CATALOG",
                "previous_status": "CATALOG_PUBLICATION_SCOPE_PROPOSED",
                "previous_catalog_event_id": "CATALOG-EVENT-001",
                "previous_catalog_event_digest": H["b"],
            }
        )
        with self.assertRaisesRegex(ValueError, "approval evidence"):
            contract.normalize_catalog_publication_event_v1(catalog, policy=self.policy)
        catalog["approval_grant_evidence"] = [
            digest_evidence("APPROVED_TO_PUBLISH_REFERENCE_CATALOG", H["c"])
        ]
        contract.normalize_catalog_publication_event_v1(catalog, policy=self.policy)

    def test_run_and_result_statuses_require_bound_digests_and_evidence(self) -> None:
        preparing = initial_registry_event()
        preparing.update(
            {
                "event_id": "REGISTRY-EVENT-PREPARING",
                "global_sequence": 2,
                "experiment_sequence": 2,
                "status": "PREPARING",
                "previous_status": "APPROVED_TO_PREPARE",
                "previous_experiment_event_id": "REGISTRY-EVENT-PREPARE-APPROVED",
                "previous_experiment_event_digest": H["7"],
                "revalidated_approval_evidence": [
                    digest_evidence("APPROVED_TO_PREPARE", H["8"])
                ],
                "capabilities": dict(
                    self.policy["phase_capabilities"]["preparation_a0"]["flags"]
                ),
            }
        )
        preparing["capability_digest"] = contract.canonical_digest_v2(
            preparing["capabilities"]
        )
        with self.assertRaisesRegex(ValueError, "consumed lease evidence"):
            contract.normalize_registry_event_v4(preparing, policy=self.policy)
        preparing["execution_lease_receipt"] = digest_evidence(
            "PREPARATION_LEASE_CONSUMED", H["9"]
        )
        contract.normalize_registry_event_v4(preparing, policy=self.policy)

        run_required = initial_registry_event()
        run_required.update(
            {
                "event_id": "REGISTRY-EVENT-RUN-REQUIRED",
                "global_sequence": 2,
                "experiment_sequence": 2,
                "status": "RUN_APPROVAL_REQUIRED",
                "previous_status": "CATALOG_BOUND",
                "previous_experiment_event_id": "REGISTRY-EVENT-CATALOG-BOUND",
                "previous_experiment_event_digest": H["8"],
                "revalidated_approval_evidence": [
                    digest_evidence("APPROVED_TO_PREPARE", H["9"])
                ],
                "capabilities": dict(
                    self.policy["phase_capabilities"]["catalog_binding"]["flags"]
                ),
            }
        )
        run_required["capability_digest"] = contract.canonical_digest_v2(
            run_required["capabilities"]
        )
        with self.assertRaisesRegex(ValueError, "run_scope_digest"):
            contract.normalize_registry_event_v4(run_required, policy=self.policy)
        run_required["run_scope_digest"] = H["a"]
        run_required["paths"]["run_scope"] = (
            "research/synthetic/roi_reproduction/M0-REPRO-FIXTURE.run.json"
        )
        contract.normalize_registry_event_v4(run_required, policy=self.policy)

        review_required = copy.deepcopy(run_required)
        review_required.update(
            {
                "event_id": "REGISTRY-EVENT-REVIEW-REQUIRED",
                "status": "REVIEW_REQUIRED",
                "previous_status": "RUNNING",
                "previous_experiment_event_id": "REGISTRY-EVENT-RUNNING",
                "previous_experiment_event_digest": H["b"],
                "revalidated_approval_evidence": [
                    digest_evidence("APPROVED_TO_PREPARE", H["9"]),
                    digest_evidence("APPROVED_TO_RUN", H["c"]),
                ],
                "capabilities": dict(self.policy["phase_capabilities"]["review"]["flags"]),
                "review_digest": None,
                "result_evidence": None,
            }
        )
        review_required["paths"]["result"] = (
            "research/synthetic/roi_reproduction/M0-REPRO-FIXTURE.result.json"
        )
        review_required["paths"]["review"] = (
            "research/synthetic/roi_reproduction/M0-REPRO-FIXTURE.review.json"
        )
        review_required["capability_digest"] = contract.canonical_digest_v2(
            review_required["capabilities"]
        )
        with self.assertRaisesRegex(ValueError, "review_digest"):
            contract.normalize_registry_event_v4(review_required, policy=self.policy)
        review_required["review_digest"] = H["d"]
        with self.assertRaisesRegex(ValueError, "result_evidence"):
            contract.normalize_registry_event_v4(review_required, policy=self.policy)
        review_required["result_evidence"] = digest_evidence(
            "VALIDATED_REPRODUCTION_RESULT", H["e"]
        )
        contract.normalize_registry_event_v4(review_required, policy=self.policy)

        preloaded = initial_registry_event()
        preloaded["run_scope_digest"] = H["f"]
        preloaded["paths"]["run_scope"] = (
            "research/synthetic/roi_reproduction/preloaded.run.json"
        )
        with self.assertRaisesRegex(ValueError, "preloaded run_scope_digest"):
            contract.normalize_registry_event_v4(preloaded, policy=self.policy)

    def test_catalog_and_release_events_are_non_authoritative(self) -> None:
        catalog_event = initial_catalog_event()
        normalized_catalog = contract.normalize_catalog_publication_event_v1(
            catalog_event, policy=self.policy
        )
        self.assertEqual(normalized_catalog["safety"], event_safety())

        preloaded_catalog = copy.deepcopy(catalog_event)
        preloaded_catalog["provider_lease_issue_receipt_digest"] = H["7"]
        with self.assertRaisesRegex(ValueError, "preloaded"):
            contract.normalize_catalog_publication_event_v1(
                preloaded_catalog, policy=self.policy
            )

        catalog_unsafe = copy.deepcopy(catalog_event)
        catalog_unsafe["safety"]["production_approved"] = True
        with self.assertRaises(ValueError):
            contract.normalize_catalog_publication_event_v1(
                catalog_unsafe, policy=self.policy
            )

        catalog_capability_flip = copy.deepcopy(catalog_event)
        catalog_capability_flip["capabilities"]["automatic_execution"] = True
        catalog_capability_flip["capability_digest"] = contract.canonical_digest_v2(
            catalog_capability_flip["capabilities"]
        )
        with self.assertRaises(ValueError):
            contract.normalize_catalog_publication_event_v1(
                catalog_capability_flip, policy=self.policy
            )

        release_event = initial_release_status_event()
        normalized_release = contract.normalize_catalog_release_status_event_v1(release_event)
        self.assertEqual(normalized_release["safety"], event_safety())

        release_unsafe = copy.deepcopy(release_event)
        release_unsafe["safety"]["merge_approved"] = True
        with self.assertRaises(ValueError):
            contract.normalize_catalog_release_status_event_v1(release_unsafe)

    def test_catalog_phase_evidence_and_release_sequence_fail_closed(self) -> None:
        publishing = initial_catalog_event()
        publishing.update(
            {
                "event_id": "CATALOG-EVENT-003",
                "global_sequence": 3,
                "subject_sequence": 3,
                "status": "CATALOG_PUBLISHING",
                "previous_status": "APPROVED_TO_PUBLISH_REFERENCE_CATALOG",
                "previous_catalog_event_id": "CATALOG-EVENT-002",
                "previous_catalog_event_digest": H["8"],
                "provider_lease_issue_receipt_digest": H["9"],
                "provider_lease_consumption_receipt_digest": H["a"],
                "revalidated_approval_evidence": [
                    digest_evidence("APPROVED_TO_PUBLISH_REFERENCE_CATALOG", H["f"])
                ],
            }
        )
        publishing["capabilities"] = copy.deepcopy(
            self.policy["phase_capabilities"]["catalog_release_maintenance"]["flags"]
        )
        publishing["capability_digest"] = contract.canonical_digest_v2(
            publishing["capabilities"]
        )
        contract.normalize_catalog_publication_event_v1(publishing, policy=self.policy)

        missing_issue = copy.deepcopy(publishing)
        missing_issue["provider_lease_issue_receipt_digest"] = None
        with self.assertRaisesRegex(ValueError, "requires provider issue"):
            contract.normalize_catalog_publication_event_v1(
                missing_issue, policy=self.policy
            )

        published = copy.deepcopy(publishing)
        published.update(
            {
                "event_id": "CATALOG-EVENT-004",
                "global_sequence": 4,
                "subject_sequence": 4,
                "status": "CATALOG_PUBLISHED",
                "previous_status": "CATALOG_PUBLISHING",
                "previous_catalog_event_id": "CATALOG-EVENT-003",
                "previous_catalog_event_digest": H["b"],
                "catalog_publication_receipt_digest": H["c"],
                "reference_catalog_release_id": "CATALOG-RELEASE-001",
                "reference_catalog_release_digest": H["d"],
                "result_evidence": digest_evidence("CATALOG_PUBLICATION_COMPLETED", H["e"]),
            }
        )
        published["capabilities"] = dict(contract.G1_CAPABILITIES)
        published["capability_digest"] = contract.canonical_digest_v2(
            published["capabilities"]
        )
        contract.normalize_catalog_publication_event_v1(published, policy=self.policy)

        incomplete_publication = copy.deepcopy(published)
        incomplete_publication["reference_catalog_release_digest"] = None
        with self.assertRaisesRegex(ValueError, "complete publication evidence"):
            contract.normalize_catalog_publication_event_v1(
                incomplete_publication, policy=self.policy
            )

        failed = copy.deepcopy(published)
        failed.update(
            {
                "event_id": "CATALOG-EVENT-005",
                "status": "CATALOG_PUBLICATION_FAILED",
                "catalog_publication_receipt_digest": None,
                "reference_catalog_release_id": None,
                "reference_catalog_release_digest": None,
                "result_evidence": digest_evidence("CATALOG_PUBLICATION_FAILED", H["e"]),
            }
        )
        contract.normalize_catalog_publication_event_v1(failed, policy=self.policy)
        missing_failure_evidence = copy.deepcopy(failed)
        missing_failure_evidence["result_evidence"] = None
        with self.assertRaisesRegex(ValueError, "issue, consumption, and result evidence"):
            contract.normalize_catalog_publication_event_v1(
                missing_failure_evidence, policy=self.policy
            )

        zero_global_sequence = initial_release_status_event()
        zero_global_sequence["global_sequence"] = 0
        with self.assertRaisesRegex(ValueError, "global_sequence must start at 1"):
            contract.normalize_catalog_release_status_event_v1(zero_global_sequence)

    def test_event_identity_constants_match_their_schemas(self) -> None:
        cases = (
            (
                "catalog_publication_event",
                initial_catalog_event(),
                ("schema_version", "event_kind", "catalog_gate_kind", "catalog_contract_version"),
            ),
            (
                "catalog_release_status_event",
                initial_release_status_event(),
                ("schema_version", "event_kind"),
            ),
            (
                "registry_event",
                initial_registry_event(),
                (
                    "schema_version",
                    "gate_kind",
                    "gate_contract_version",
                    "execution_kind",
                ),
            ),
        )
        for schema_kind, fixture, fields in cases:
            schema = contract.strict_json_load_v2(ROOT / contract.SCHEMA_PATHS[schema_kind])
            node = schema["allOf"][0] if "allOf" in schema else schema
            for field in fields:
                with self.subTest(schema_kind=schema_kind, field=field):
                    self.assertEqual(
                        node["properties"][field]["const"],
                        fixture[field],
                        f"{schema_kind}.{field} differs between schema and normalizer fixture",
                    )

    def test_mixed_registry_history_preserves_legacy_bytes_and_requires_canonical_v4(self) -> None:
        legacy_v2 = b'{"schema_version":2,"legacy":"v2 bytes stay unchanged"}\n'
        infrastructure_v3 = b'{"schema_version":3,"infra":"v3 bytes stay unchanged"}\n'
        event = initial_registry_event()
        v4 = contract.compile_non_authoritative_registry_line_v4_fixture(
            event, policy=self.policy, fixture_only=True
        )
        with self.assertRaisesRegex(ValueError, "non-authoritative fixtures"):
            contract.compile_non_authoritative_registry_line_v4_fixture(
                event, policy=self.policy, fixture_only=False
            )
        raw = legacy_v2 + infrastructure_v3 + v4

        preserved = contract.validate_mixed_registry_history(raw, policy=self.policy)
        self.assertEqual(preserved, [legacy_v2, infrastructure_v3, v4])
        self.assertEqual(b"".join(preserved), raw)

        restarted = initial_registry_event()
        restarted["event_id"] = "REGISTRY-EVENT-RESTART"
        restarted["global_sequence"] = 2
        restarted_v4 = contract.compile_non_authoritative_registry_line_v4_fixture(
            restarted, policy=self.policy, fixture_only=True
        )
        with self.assertRaisesRegex(ValueError, "experiment sequences must be contiguous"):
            contract.validate_mixed_registry_history(
                legacy_v2 + infrastructure_v3 + v4 + restarted_v4,
                policy=self.policy,
            )

        noncanonical_v4 = (
            json.dumps(event, ensure_ascii=False, sort_keys=False).encode("utf-8") + b"\n"
        )
        with self.assertRaisesRegex(ValueError, "not canonical"):
            contract.validate_mixed_registry_history(
                legacy_v2 + infrastructure_v3 + noncanonical_v4,
                policy=self.policy,
            )

        with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
            contract.validate_mixed_registry_history(
                b'{"schema_version":5}\n', policy=self.policy
            )


if __name__ == "__main__":
    unittest.main()
