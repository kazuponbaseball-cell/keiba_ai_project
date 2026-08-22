from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from registered_nonpromotion_contract_v1 import (
    ContractError,
    RegisteredRecipe,
    canonical_digest,
    load_strict_json,
    normalized_utf8_lf_sha256,
    resolve_registered_recipe,
)


GATE_KIND = "registered_nonpromotion_offline_diagnostic_v1"
APPROVAL_KEYWORD = "APPROVED_OFFLINE_NONPROMOTION_DIAGNOSTIC_RUN"
DEFAULT_REPOSITORY = "kazuponbaseball-cell/keiba_ai_project"
DEFAULT_BASE_BRANCH = "main"

RECIPE_ID = "historical_ai_duplicate_gate_impact_v1"
RECIPE_VERSION = 1
SOURCE_GATE_KIND = "registered_nonpromotion_diagnostic_v1"
SOURCE_POLICY_RELATIVE_PATH = Path("research/REGISTERED_NONPROMOTION_DIAGNOSTIC_V1.json")
SOURCE_RECIPE_RELATIVE_PATH = Path(
    "research/diagnostic_recipes/historical_ai_duplicate_gate_impact_v1.json"
)
POLICY_RELATIVE_PATH = Path("research/REGISTERED_NONPROMOTION_OFFLINE_DIAGNOSTIC_V1.json")
FIXED_OUTPUT_ROOT = (
    "outputs/research/registered_nonpromotion_offline/"
    "historical_ai_duplicate_gate_impact_v1"
)
MATERIALIZATION_MANIFEST_PATH = (
    "outputs/research/registered_nonpromotion_offline_materialized/"
    "historical_ai_duplicate_gate_impact_v1/materialization_manifest.json"
)
RUN_SCOPE_ARTIFACT_DIRECTORY = (
    "outputs/research/registered_nonpromotion_offline_scopes"
)

EXPECTED_RECIPE_DIGEST = (
    "3754b796ba6dcb3f3c0f084641411e5e3decce0c8d934a0a294976f66caab292"
)
EXPECTED_QUESTION_FAMILY_DIGEST = (
    "a9555c0fe36f57b924822ed8d718ba6d2b150140feb4607d7749e300151318d7"
)
EXPECTED_ELIGIBILITY_AST_DIGEST = (
    "09383c64cca20fa5a7a8fcfb9f76f1bf7feeb5e8b76744a90e4aa954b9ba0c5f"
)
EXPECTED_FORMULA_FINGERPRINT = (
    "67b0d3e5b92166a10b3077bff03e107c9db071b310f60f43b8758ce316eda878"
)

EXPECTED_SCHEMA_PATHS = {
    "policy": "research/schemas/registered_nonpromotion_offline_policy_v1.schema.json",
    "run_scope": "research/schemas/registered_nonpromotion_offline_run_scope_v1.schema.json",
    "approval_evidence": (
        "research/schemas/registered_nonpromotion_offline_approval_evidence_v1.schema.json"
    ),
    "result": "research/schemas/registered_nonpromotion_offline_result_v1.schema.json",
}

RUNTIME_MATERIAL_PATHS = {
    "policy_blob_sha256": POLICY_RELATIVE_PATH.as_posix(),
    "policy_schema_blob_sha256": EXPECTED_SCHEMA_PATHS["policy"],
    "run_scope_schema_blob_sha256": EXPECTED_SCHEMA_PATHS["run_scope"],
    "approval_evidence_schema_blob_sha256": EXPECTED_SCHEMA_PATHS[
        "approval_evidence"
    ],
    "result_schema_blob_sha256": EXPECTED_SCHEMA_PATHS["result"],
    "contract_blob_sha256": (
        "scripts/research/registered_nonpromotion_offline_contract_v1.py"
    ),
    "approval_verifier_blob_sha256": (
        "scripts/research/registered_nonpromotion_offline_approval_v1.py"
    ),
    "runner_blob_sha256": (
        "scripts/research/registered_nonpromotion_offline_runner_v1.py"
    ),
    "github_approval_blob_sha256": "scripts/research/github_approval.py",
    "source_policy_blob_sha256": SOURCE_POLICY_RELATIVE_PATH.as_posix(),
    "source_recipe_blob_sha256": SOURCE_RECIPE_RELATIVE_PATH.as_posix(),
    "source_recipe_schema_blob_sha256": (
        "research/schemas/registered_nonpromotion_recipe_v1.schema.json"
    ),
    "source_contract_blob_sha256": (
        "scripts/research/registered_nonpromotion_contract_v1.py"
    ),
    "source_science_blob_sha256": (
        "scripts/research/registered_nonpromotion_supervised_executor_v1.py"
    ),
}

SOURCE_INPUTS = {
    "diagnostic_master": {
        "path": (
            "outputs/analysis/umaren_wide_rebuild_v1/"
            "wide_diagnostic_master_v1/wide_diagnostic_master_v1.csv"
        ),
        "expected_sha256": (
            "697142b64e8052b212731dc0319ccafb7f61ac29dbc46f67385f9ae050129de9"
        ),
    },
    "p_action_artifact": {
        "path": (
            "outputs/analysis/umaren_wide_rebuild_v1/"
            "m1c_action_calibration_offset_v1/"
            "m1c_action_calibration_offset_oos_predictions.csv"
        ),
        "expected_sha256": (
            "34f56b5a61261bd9b6cfd38797b65bd88415d0778d98cef29eebfbe2f09e513c"
        ),
    },
    "official_payoff_source": {
        "path": "data/processed/target/wide_payoffs.csv",
        "expected_sha256": (
            "b94b0c0ea2ce4424d70432f7d070a9083d01850876a710432ec5b98538070d83"
        ),
    },
}

PROJECTION_INPUTS = {
    "candidate_projection": {
        "path": (
            "outputs/research/registered_nonpromotion_offline_materialized/"
            "historical_ai_duplicate_gate_impact_v1/candidate_projection.jsonl"
        )
    },
    "settlement_projection": {
        "path": (
            "outputs/research/registered_nonpromotion_offline_materialized/"
            "historical_ai_duplicate_gate_impact_v1/settlement_projection.jsonl"
        )
    },
}

EXPECTED_SCORE = {
    "recorded_total": 46,
    "threshold": 75,
    "threshold_met": False,
    "status": "BLOCKED_SCORE",
    "score_credit": 0,
    "threshold_override_allowed": False,
}

FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_RELATIVE_PATH = re.compile(r"^[A-Za-z0-9_.\-/]+$")
UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)

RUN_BINDING_KEYS = {
    "repository",
    "base_branch",
    "run_scope_base_commit",
    "verified_current_main_sha",
    "approvers_blob_sha",
    "approvers_content_sha256",
    "runtime_material_sha256",
    "source_bindings",
    "projection_bindings",
    "materialization_manifest",
    "python_minor_version",
    "numpy_version",
    "environment_manifest_sha256",
    "output_root",
    "sealed_at",
}


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ContractError(f"{label} keys differ; missing={missing}, extra={extra}")


def _require_sha(value: Any, *, label: str, git: bool = False) -> str:
    pattern = FULL_GIT_SHA if git else FULL_SHA256
    if (
        not isinstance(value, str)
        or not pattern.fullmatch(value)
        or set(value) == {"0"}
    ):
        raise ContractError(f"{label} must be a lowercase full {'Git ' if git else ''}SHA")
    return value


def _require_safe_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SAFE_RELATIVE_PATH.fullmatch(value):
        raise ContractError(f"{label} must be a safe repository-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise ContractError(f"{label} must be a safe repository-relative path")
    return value


def _require_utc(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not UTC_TIMESTAMP.fullmatch(value):
        raise ContractError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError(f"{label} must be UTC")
    return value


def offline_run_scope_artifact_path(run_scope_digest: str) -> str:
    """Derive the only permitted scope artifact path without self-reference."""

    digest = _require_sha(run_scope_digest, label="run_scope_digest")
    return f"{RUN_SCOPE_ARTIFACT_DIRECTORY}/{digest}.run.json"


def _validate_finite_json(value: Any, *, path: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ContractError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ContractError(f"{path} contains a non-string key")
        for key, item in value.items():
            _validate_finite_json(item, path=f"{path}.{key}")
        return
    raise ContractError(f"{path} contains a non-JSON value")


def _expected_source_recipe_policy() -> dict[str, Any]:
    return {
        "source_gate_kind": SOURCE_GATE_KIND,
        "source_policy_path": SOURCE_POLICY_RELATIVE_PATH.as_posix(),
        "recipe_id": RECIPE_ID,
        "recipe_version": RECIPE_VERSION,
        "recipe_path": SOURCE_RECIPE_RELATIVE_PATH.as_posix(),
        "canonical_recipe_sha256": EXPECTED_RECIPE_DIGEST,
        "source_formula_fingerprint_sha256": EXPECTED_FORMULA_FINGERPRINT,
        "question_family_sha256": EXPECTED_QUESTION_FAMILY_DIGEST,
        "eligibility_ast_sha256": EXPECTED_ELIGIBILITY_AST_DIGEST,
        "tier": "B_REGISTERED_HISTORICAL_IMPACT",
        "source_authority_class": "B_LOCAL_HASHED",
        "ordinary_strategy_score": dict(EXPECTED_SCORE),
    }


def _expected_activation_contract() -> dict[str, Any]:
    return {
        "containing_commit_must_be_human_merged_to_main": True,
        "github_read_only_containing_commit_verification_required": True,
        "run_scope_implementation_commit_must_equal_verified_current_main": True,
        "human_merge_metadata_machine_verified": False,
        "human_merge_remains_procedural_prerequisite": True,
        "human_main_merge_activates_gate_availability": True,
        "human_main_merge_is_run_approval": False,
        "branch_ci_ready_review_or_chat_is_gate_availability": False,
        "external_g2_required": False,
        "repository_or_local_artifact_is_authority": False,
        "single_use_policy": "ONE_ACCEPTED_EXECUTION",
        "single_use_enforcement": "BEST_EFFORT_LOCAL_EXCLUSIVE_RECEIPT",
        "global_replay_proof": False,
        "rollback_resistant": False,
        "durable_remote_ledger": False,
        "network_isolation": "APPLICATION_LEVEL_NOT_OS_SANDBOX",
    }


def _expected_run_approval() -> dict[str, Any]:
    return {
        "keyword": APPROVAL_KEYWORD,
        "comment_count": 1,
        "run_scope_must_be_sealed_before_comment": True,
        "allowlisted_github_user_required": True,
        "github_actor_type": "User",
        "automation_actor_allowed": False,
        "edited_comment_allowed": False,
        "exact_body_required": True,
        "comment_global_consumption_guaranteed": False,
    }


def _expected_lifecycle_contract() -> dict[str, Any]:
    return {
        "states": [
            "RNOD_RUN_SCOPE_FROZEN",
            "RNOD_RUN_APPROVAL_REQUIRED",
            "RNOD_APPROVED",
            "RNOD_RUNNING",
            "RNOD_RESULT_SEALED",
            "RNOD_COMPLETED",
            "INVALID",
        ],
        "allowed_transitions": {
            "RNOD_RUN_SCOPE_FROZEN": ["INVALID", "RNOD_RUN_APPROVAL_REQUIRED"],
            "RNOD_RUN_APPROVAL_REQUIRED": ["INVALID", "RNOD_APPROVED"],
            "RNOD_APPROVED": ["INVALID", "RNOD_RUNNING"],
            "RNOD_RUNNING": ["INVALID", "RNOD_RESULT_SEALED"],
            "RNOD_RESULT_SEALED": ["INVALID", "RNOD_COMPLETED"],
            "RNOD_COMPLETED": [],
            "INVALID": [],
        },
        "protected_content_access_failure_before_start": "BLOCKED_PREACCESS",
        "blocked_preaccess_consumes_run": False,
        "exclusive_start_receipt_required_before_candidate_open": True,
        "failure_after_exclusive_start_receipt": "INVALID_AFTER_START_NO_RETRY",
        "preparing_state_exists": False,
        "shadow_or_ack_state_exists": False,
        "skip_self_transition_terminal_resurrection_or_retry_allowed": False,
    }


def _expected_cross_route_contract() -> dict[str, Any]:
    return {
        "strict_route_independent_and_unmodified": True,
        "cross_route_duplicate_execution_technically_prevented": False,
        "cross_route_single_use_guaranteed": False,
        "offline_result_is_global_question_family_consumption": False,
        "later_strict_result_may_not_claim_independent_evidence_without_binding_prior_offline_exposure": True,
    }


def _expected_execution_contract() -> dict[str, Any]:
    return {
        "runner_template_id": "REGISTERED_NONPROMOTION_OFFLINE_EXECUTOR_V1",
        "structured_argv": [
            "registered_nonpromotion_offline_runner_v1",
            "execute_offline_registered_diagnostic",
        ],
        "shell_interpretation": False,
        "free_form_argv_allowed": False,
        "exact_recipe_count": 1,
        "exact_arm_count": 2,
        "exact_registered_transform_count": 1,
        "registered_transform_id": "REMOVE_RAW_P_GATE_FAMILY",
        "race_count": 3746,
        "fold_counts": {"fold2": 1661, "fold3": 1653, "fold4": 432},
        "decision_freeze_before_settlement_access": True,
        "master_candidate_usecols_only_before_freeze": True,
        "p_action_cross_check_before_freeze": True,
        "materialization_authority": "HUMAN_MERGED_EXACT_FIXED_PROJECTION_ONLY",
        "materialization_may_not_compute_decisions_or_metrics": True,
        "materialization_may_not_compute_roi": True,
        "materialization_may_not_compute_thresholds_or_variants": True,
        "fixed_output_root": FIXED_OUTPUT_ROOT,
        "run_scope_artifact_directory": RUN_SCOPE_ARTIFACT_DIRECTORY,
        "run_scope_artifact_filename_rule": "<run_scope_digest>.run.json",
        "run_scope_serialization": "UTF8_LF_CANONICAL_JSON",
        "run_scope_overwrite_allowed": False,
        "fresh_output_root_required": True,
        "overwrite_allowed": False,
        "retry_count": 0,
        "threshold_search_count": 0,
        "refit_count": 0,
        "recalibration_count": 0,
        "python_minor_versions": ["3.11", "3.12"],
        "numpy_version": "2.4.3",
        "replica_mode": "LOGICAL_SAME_PROCESS_SHARED_SEALED_INPUT_BYTES",
        "replica_ids": ["clean_a", "clean_b"],
        "attempt_count_per_replica": 1,
        "semantic_equality_required": True,
        "preferred_replica_selection_allowed": False,
        "result_publish_requires_both_projection_digests_match": True,
        "clean_process_or_os_isolation_claimed": False,
    }


def _expected_projection_contract() -> dict[str, Any]:
    return {
        "materialization_authority": "HUMAN_MERGED_EXACT_FIXED_PROJECTION_ONLY",
        "materialization_may_not_compute_decisions_or_metrics": True,
        "materialization_may_not_compute_roi": True,
        "materialization_may_not_compute_thresholds_or_variants": True,
        "raw_source_bytes_may_contain_forbidden_columns": True,
        "raw_source_open_requires_exact_hash_match": True,
        "preaccess_raw_provenance_revalidation_required": True,
        "preaccess_raw_provenance_revalidation_may_compute_decisions_or_metrics": False,
        "runner_code_opens_raw_sources_after_exclusive_start_receipt": False,
        "raw_source_filesystem_capability_isolation_claimed": False,
        "official_payoff_allowed_projection_columns": [
            "race_id",
            "horse_a",
            "horse_b",
            "wide_pay",
        ],
        "raw_source_allowed_projection_columns": {
            "diagnostic_master": [
                "candidate_generated",
                "eligible_race",
                "fold",
                "horse_a",
                "horse_b",
                "p_action_C0_offset",
                "race_date",
                "race_id",
                "top1_pair_key",
                "top1_wide_prob",
                "venue_code",
            ],
            "p_action_artifact": [
                "fold",
                "p_action_C0_offset",
                "race_id",
                "top1_pair_key",
                "top1_wide_prob",
            ],
            "official_payoff_source": ["horse_a", "horse_b", "race_id", "wide_pay"],
        },
        "candidate_cross_source_equality_columns": [
            "fold",
            "p_action_C0_offset",
            "race_id",
            "top1_pair_key",
            "top1_wide_prob",
        ],
        "candidate_cross_source_equality_required_for_all_3746_rows": True,
        "p_action_lineage_formula": "sigmoid(logit(p)+0.130654047367905)",
        "p_action_numeric_semantics": "IEEE754_BINARY64_FROM_UTF8_DECIMAL",
        "p_action_absolute_tolerance": "1e-12",
        "p_action_lineage_check_required_for_all_3746_rows": True,
        "candidate_key_formula": (
            "f'{min(int(horse_a),int(horse_b))}-"
            "{max(int(horse_a),int(horse_b))}'"
        ),
        "candidate_horse_numbers_distinct_positive_integers": True,
        "candidate_identity_check_required_for_all_3746_rows": True,
        "forbidden_fields_may_not_be_selected_or_converted_to_typed_semantic_values_used_persisted_exposed_or_influence_output": True,
        "csv_header_and_record_tokenization_for_allowlisted_extraction_allowed": True,
        "decision_and_run_phases_mount_raw_sources": False,
        "materialized_projections_contain_market_fields": False,
        "candidate_projection_path": PROJECTION_INPUTS["candidate_projection"]["path"],
        "settlement_projection_path": PROJECTION_INPUTS["settlement_projection"]["path"],
        "candidate_projection_must_be_sealed_before_settlement_projection_open": True,
        "candidate_and_settlement_projection_must_be_distinct": True,
        "projection_format": "UTF8_LF_STRICT_JSONL",
        "race_count": 3746,
        "materialization_manifest_path": MATERIALIZATION_MANIFEST_PATH,
        "materialization_manifest_required": True,
        "materializer_usecols_attestation_required": True,
    }


def _expected_evidence_contract() -> dict[str, Any]:
    return {
        "evidence_purpose_class": "DIAGNOSTIC_NONPROMOTION",
        "source_authority_class": "B_LOCAL_HASHED",
        "reused_development_oos": True,
        "strict_t3_rows": 0,
        "confirmatory": False,
        "promotion_eligible": False,
        "score_credit": 0,
        "shadow_transition_supported": False,
        "production_transition_supported": False,
    }


def _expected_safety() -> dict[str, Any]:
    return {
        "automatic_execution": False,
        "automatic_github_approval": False,
        "odds_price_popularity_or_market_access": False,
        "training": False,
        "model_inference": False,
        "recalibration": False,
        "threshold_search": False,
        "workload_network_access": False,
        "workload_external_api_calls": False,
        "workload_credential_access": False,
        "fixed_read_only_git_control_plane_subprocess_allowed": True,
        "free_form_or_workload_subprocess_allowed": False,
        "purchase_path_access": False,
        "production_change": False,
        "shadow_transition": False,
        "notification_side_effects": False,
        "order_side_effects": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def validate_offline_policy(policy: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "gate_kind",
        "gate_contract_version",
        "implementation_layer",
        "implementation_status",
        "authority",
        "execution_status",
        "repository",
        "base_branch",
        "activation_contract",
        "schema_paths",
        "source_recipe",
        "source_inputs",
        "projection_contract",
        "run_approval",
        "lifecycle",
        "cross_route_contract",
        "execution_contract",
        "evidence_contract",
        "result_contract",
        "safety",
    }
    _require_exact_keys(policy, expected_keys, label="offline policy")
    expected_scalars = {
        "schema_version": 1,
        "gate_kind": GATE_KIND,
        "gate_contract_version": 1,
        "implementation_layer": "LOCAL_HASH_BOUND_OFFLINE_EXECUTOR",
        "implementation_status": (
            "IMPLEMENTED_CONDITIONAL_ON_CONTAINING_COMMIT_HUMAN_MERGE"
        ),
        "authority": False,
        "execution_status": "HUMAN_MERGE_AND_RUN_APPROVAL_REQUIRED",
        "repository": DEFAULT_REPOSITORY,
        "base_branch": DEFAULT_BASE_BRANCH,
    }
    for key, expected in expected_scalars.items():
        if policy.get(key) != expected:
            raise ContractError(f"offline policy {key} mismatch")
    if policy.get("activation_contract") != _expected_activation_contract():
        raise ContractError("offline activation contract mismatch")
    if policy.get("schema_paths") != EXPECTED_SCHEMA_PATHS:
        raise ContractError("offline schema path registry mismatch")
    if policy.get("source_recipe") != _expected_source_recipe_policy():
        raise ContractError("offline source recipe contract mismatch")
    expected_sources = [
        {"role": role, **SOURCE_INPUTS[role]}
        for role in ("diagnostic_master", "p_action_artifact", "official_payoff_source")
    ]
    if policy.get("source_inputs") != expected_sources:
        raise ContractError("offline source input registry mismatch")
    if policy.get("projection_contract") != _expected_projection_contract():
        raise ContractError("offline projection contract mismatch")
    if policy.get("run_approval") != _expected_run_approval():
        raise ContractError("offline run approval contract mismatch")
    if policy.get("lifecycle") != _expected_lifecycle_contract():
        raise ContractError("offline lifecycle contract mismatch")
    if policy.get("cross_route_contract") != _expected_cross_route_contract():
        raise ContractError("offline cross-route contract mismatch")
    if policy.get("execution_contract") != _expected_execution_contract():
        raise ContractError("offline execution contract mismatch")
    if policy.get("evidence_contract") != _expected_evidence_contract():
        raise ContractError("offline evidence contract mismatch")
    expected_result = {
        **_expected_evidence_contract(),
        "allowed_outcomes": ["NO_DECISION_EFFECT", "DIRECTIONAL_EFFECT", "INVALID"],
        "completed_result_schema_outcomes": [
            "NO_DECISION_EFFECT",
            "DIRECTIONAL_EFFECT",
        ],
        "invalid_failure_artifact_contains_performance": False,
        "single_use_policy": "ONE_ACCEPTED_EXECUTION",
        "single_use_enforcement": "BEST_EFFORT_LOCAL_EXCLUSIVE_RECEIPT",
        "global_replay_proof": False,
        "rollback_resistant": False,
        "durable_remote_ledger": False,
        "network_isolation": "APPLICATION_LEVEL_NOT_OS_SANDBOX",
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    if policy.get("result_contract") != expected_result:
        raise ContractError("offline result contract mismatch")
    if policy.get("safety") != _expected_safety():
        raise ContractError("offline safety contract mismatch")


@dataclass(frozen=True)
class OfflineRegisteredRecipe:
    policy: dict[str, Any]
    recipe: dict[str, Any]
    source_registered: RegisteredRecipe
    policy_digest: str
    policy_file_sha256: str
    recipe_digest: str
    recipe_file_sha256: str
    schema_bundle_digest: str
    question_family_digest: str
    eligibility_ast_digest: str
    runtime_material_digests: Mapping[str, str]


def resolve_offline_registered_recipe(root: Path) -> OfflineRegisteredRecipe:
    root = Path(root)
    policy_path = root / POLICY_RELATIVE_PATH
    policy = load_strict_json(policy_path)
    if not isinstance(policy, dict):
        raise ContractError("offline policy must be an object")
    validate_offline_policy(policy)

    source = resolve_registered_recipe(
        root,
        recipe_id=RECIPE_ID,
        recipe_version=RECIPE_VERSION,
    )
    if (
        source.policy.get("gate_kind") != SOURCE_GATE_KIND
        or source.recipe_digest != EXPECTED_RECIPE_DIGEST
        or source.question_family_digest != EXPECTED_QUESTION_FAMILY_DIGEST
        or source.eligibility_ast_digest != EXPECTED_ELIGIBILITY_AST_DIGEST
        or source.recipe.get("source_design", {}).get("formula_fingerprint_sha256")
        != EXPECTED_FORMULA_FINGERPRINT
        or source.recipe.get("ordinary_strategy_score") != EXPECTED_SCORE
        or source.recipe.get("cohort", {}).get("race_count") != 3746
        or source.recipe.get("cohort", {}).get("fold_counts")
        != {"fold2": 1661, "fold3": 1653, "fold4": 432}
    ):
        raise ContractError("strict source recipe differs from the offline fixed recipe")

    schema_materials: list[dict[str, str]] = []
    for schema_id, raw_path in sorted(EXPECTED_SCHEMA_PATHS.items()):
        path = root / Path(*PurePosixPath(raw_path).parts)
        schema = load_strict_json(path)
        if (
            not isinstance(schema, dict)
            or schema.get("$schema") != "http://json-schema.org/draft-07/schema#"
            or schema.get("additionalProperties") is not False
        ):
            raise ContractError(f"offline schema {schema_id} is not strict Draft-07")
        schema_materials.append(
            {
                "schema_id": schema_id,
                "path": raw_path,
                "content_sha256": normalized_utf8_lf_sha256(path),
            }
        )

    runtime_material_digests: dict[str, str] = {}
    for field, raw_path in RUNTIME_MATERIAL_PATHS.items():
        pure = PurePosixPath(raw_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ContractError(f"runtime material path escapes repository: {raw_path}")
        runtime_material_digests[field] = normalized_utf8_lf_sha256(
            root / Path(*pure.parts)
        )
    schema_bundle_digest = canonical_digest(schema_materials)
    runtime_material_digests["schema_bundle_sha256"] = schema_bundle_digest

    return OfflineRegisteredRecipe(
        policy=dict(policy),
        recipe=dict(source.recipe),
        source_registered=source,
        policy_digest=canonical_digest(policy),
        policy_file_sha256=runtime_material_digests["policy_blob_sha256"],
        recipe_digest=source.recipe_digest,
        recipe_file_sha256=runtime_material_digests["source_recipe_blob_sha256"],
        schema_bundle_digest=schema_bundle_digest,
        question_family_digest=source.question_family_digest,
        eligibility_ast_digest=source.eligibility_ast_digest,
        runtime_material_digests=runtime_material_digests,
    )


def _validate_source_bindings(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ContractError("source_bindings must be an object")
    if set(value) != set(SOURCE_INPUTS):
        raise ContractError("source_bindings must contain the exact three source roles")
    output: dict[str, dict[str, Any]] = {}
    for role in ("diagnostic_master", "p_action_artifact", "official_payoff_source"):
        binding = value.get(role)
        if not isinstance(binding, Mapping):
            raise ContractError(f"source binding {role} must be an object")
        _require_exact_keys(
            binding,
            {"path", "expected_sha256", "observed_sha256", "observed_byte_size"},
            label=f"source binding {role}",
        )
        expected = SOURCE_INPUTS[role]
        path = _require_safe_relative_path(binding.get("path"), label=f"{role} path")
        if path != expected["path"]:
            raise ContractError(f"source binding {role} path differs from policy")
        expected_sha = _require_sha(
            binding.get("expected_sha256"), label=f"{role} expected_sha256"
        )
        observed_sha = _require_sha(
            binding.get("observed_sha256"), label=f"{role} observed_sha256"
        )
        if expected_sha != expected["expected_sha256"] or observed_sha != expected_sha:
            raise ContractError(f"source binding {role} hash differs from policy")
        byte_size = binding.get("observed_byte_size")
        if type(byte_size) is not int or byte_size <= 0:
            raise ContractError(f"source binding {role} byte size must be positive")
        output[role] = dict(binding)
    paths = [item["path"] for item in output.values()]
    if len(paths) != len(set(paths)):
        raise ContractError("source bindings must resolve to three distinct paths")
    return output


def _validate_projection_bindings(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(PROJECTION_INPUTS):
        raise ContractError(
            "projection_bindings must contain candidate and settlement projections"
        )
    output: dict[str, dict[str, Any]] = {}
    for role in ("candidate_projection", "settlement_projection"):
        binding = value.get(role)
        if not isinstance(binding, Mapping):
            raise ContractError(f"projection binding {role} must be an object")
        _require_exact_keys(
            binding,
            {"path", "sha256", "byte_size"},
            label=f"projection binding {role}",
        )
        path = _require_safe_relative_path(binding.get("path"), label=f"{role} path")
        if path != PROJECTION_INPUTS[role]["path"]:
            raise ContractError(f"projection binding {role} path differs from policy")
        _require_sha(binding.get("sha256"), label=f"{role} sha256")
        byte_size = binding.get("byte_size")
        if type(byte_size) is not int or byte_size <= 0:
            raise ContractError(f"projection binding {role} byte size must be positive")
        output[role] = dict(binding)
    if (
        output["candidate_projection"]["path"]
        == output["settlement_projection"]["path"]
        or output["candidate_projection"]["sha256"]
        == output["settlement_projection"]["sha256"]
    ):
        raise ContractError("candidate and settlement projections must be distinct")
    return output


def _validate_materialization_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("materialization_manifest must be an object")
    _require_exact_keys(
        value,
        {"path", "sha256", "byte_size"},
        label="materialization_manifest",
    )
    path = _require_safe_relative_path(
        value.get("path"), label="materialization_manifest path"
    )
    if path != MATERIALIZATION_MANIFEST_PATH:
        raise ContractError("materialization manifest path differs from policy")
    _require_sha(value.get("sha256"), label="materialization_manifest sha256")
    byte_size = value.get("byte_size")
    if type(byte_size) is not int or byte_size <= 0:
        raise ContractError("materialization manifest byte size must be positive")
    return dict(value)


def compile_offline_run_scope(
    registered: OfflineRegisteredRecipe,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(registered, OfflineRegisteredRecipe):
        raise ContractError("offline compiler requires an OfflineRegisteredRecipe")
    if not isinstance(bindings, Mapping):
        raise ContractError("offline run bindings must be an object")
    _require_exact_keys(bindings, RUN_BINDING_KEYS, label="offline run bindings")
    _validate_finite_json(dict(bindings))
    if bindings.get("repository") != DEFAULT_REPOSITORY:
        raise ContractError("offline run repository mismatch")
    if bindings.get("base_branch") != DEFAULT_BASE_BRANCH:
        raise ContractError("offline run base branch mismatch")
    base_commit = _require_sha(
        bindings.get("run_scope_base_commit"), label="run_scope_base_commit", git=True
    )
    current_main = _require_sha(
        bindings.get("verified_current_main_sha"),
        label="verified_current_main_sha",
        git=True,
    )
    if base_commit != current_main:
        raise ContractError("offline run scope must bind the exact current main commit")
    _require_sha(bindings.get("approvers_blob_sha"), label="approvers_blob_sha", git=True)
    _require_sha(
        bindings.get("approvers_content_sha256"), label="approvers_content_sha256"
    )
    runtime_materials = bindings.get("runtime_material_sha256")
    if not isinstance(runtime_materials, Mapping) or dict(runtime_materials) != dict(
        registered.runtime_material_digests
    ):
        raise ContractError("offline runtime material bundle differs from repository bytes")
    source_bindings = _validate_source_bindings(bindings.get("source_bindings"))
    projection_bindings = _validate_projection_bindings(
        bindings.get("projection_bindings")
    )
    materialization_manifest = _validate_materialization_manifest(
        bindings.get("materialization_manifest")
    )
    python_minor = bindings.get("python_minor_version")
    if python_minor not in {"3.11", "3.12"}:
        raise ContractError("offline Python minor version must be 3.11 or 3.12")
    if bindings.get("numpy_version") != "2.4.3":
        raise ContractError("offline NumPy version must be exactly 2.4.3")
    _require_sha(
        bindings.get("environment_manifest_sha256"),
        label="environment_manifest_sha256",
    )
    output_root = _require_safe_relative_path(
        bindings.get("output_root"), label="output_root"
    )
    if output_root != FIXED_OUTPUT_ROOT:
        raise ContractError("offline output root differs from the single fixed root")
    sealed_at = _require_utc(bindings.get("sealed_at"), label="sealed_at")

    recipe = registered.recipe
    semantic_subject = {
        "gate_kind": GATE_KIND,
        "recipe_id": RECIPE_ID,
        "recipe_version": RECIPE_VERSION,
        "recipe_digest": registered.recipe_digest,
        "question_family_digest": registered.question_family_digest,
        "eligibility_ast_digest": registered.eligibility_ast_digest,
        "tier": "B_REGISTERED_HISTORICAL_IMPACT",
        "registered_transform_id": "REMOVE_RAW_P_GATE_FAMILY",
        "comparison_arm_ids": ["D0_REFERENCE", "D1_REMOVE_RAW_GATES"],
        "cohort_id": "CITED_FOLDS_2_TO_4",
        "race_count": 3746,
        "fold_counts": {"fold2": 1661, "fold3": 1653, "fold4": 432},
        "primary_metric": recipe["metric"]["primary_metric"],
        "ordinary_strategy_score": dict(EXPECTED_SCORE),
        "evidence_purpose_class": "DIAGNOSTIC_NONPROMOTION",
        "source_authority_class": "B_LOCAL_HASHED",
        "confirmatory": False,
        "promotion_eligible": False,
        "score_credit": 0,
    }
    exact_subject = {
        "run_scope_base_commit": base_commit,
        "implementation_commit": base_commit,
        "runtime_material_bundle_sha256": canonical_digest(
            dict(runtime_materials)
        ),
        "source_bindings": source_bindings,
        "projection_bindings": projection_bindings,
        "materialization_manifest": materialization_manifest,
        "environment": {
            "python_minor_version": python_minor,
            "numpy_version": "2.4.3",
            "environment_manifest_sha256": bindings[
                "environment_manifest_sha256"
            ],
        },
        "output_root": output_root,
    }
    resolved_contract_digests = {
        "comparison_sha256": canonical_digest(recipe["comparison"]),
        "cohort_sha256": canonical_digest(recipe["cohort"]),
        "metric_sha256": canonical_digest(recipe["metric"]),
        "sensitivity_sha256": canonical_digest(recipe["sensitivity"]),
        "bootstrap_sha256": canonical_digest(recipe["bootstrap"]),
        "lifecycle_sha256": canonical_digest(_expected_lifecycle_contract()),
        "cross_route_sha256": canonical_digest(_expected_cross_route_contract()),
        "execution_sha256": canonical_digest(_expected_execution_contract()),
        "projection_sha256": canonical_digest(_expected_projection_contract()),
        "evidence_sha256": canonical_digest(_expected_evidence_contract()),
        "safety_sha256": canonical_digest(_expected_safety()),
    }
    replica_topology = {
        "replica_mode": "LOGICAL_SAME_PROCESS_SHARED_SEALED_INPUT_BYTES",
        "replica_ids": ["clean_a", "clean_b"],
        "attempt_count_per_replica": 1,
        "semantic_equality_required": True,
        "preferred_replica_selection_allowed": False,
        "retry_count": 0,
        "result_publish_requires_both_projection_digests_match": True,
        "clean_process_or_os_isolation_claimed": False,
    }
    run_scope = {
        "schema_version": 1,
        "gate_kind": GATE_KIND,
        "status": "RNOD_RUN_SCOPE_FROZEN",
        "authority": False,
        "repository": DEFAULT_REPOSITORY,
        "base_branch": DEFAULT_BASE_BRANCH,
        "run_scope_base_commit": base_commit,
        "implementation_commit": base_commit,
        "verified_current_main_sha": current_main,
        "approvers_blob_sha": bindings["approvers_blob_sha"],
        "approvers_content_sha256": bindings["approvers_content_sha256"],
        "policy_digest": registered.policy_digest,
        "recipe_id": RECIPE_ID,
        "recipe_version": RECIPE_VERSION,
        "recipe_digest": registered.recipe_digest,
        "question_family_digest": registered.question_family_digest,
        "eligibility_ast_digest": registered.eligibility_ast_digest,
        "tier": "B_REGISTERED_HISTORICAL_IMPACT",
        "ordinary_strategy_score": dict(EXPECTED_SCORE),
        "resolved_contract_digests": resolved_contract_digests,
        "replica_topology": replica_topology,
        "runtime_bindings": dict(bindings),
        "semantic_subject": semantic_subject,
        "semantic_subject_digest": canonical_digest(semantic_subject),
        "exact_subject": exact_subject,
        "exact_subject_digest": canonical_digest(exact_subject),
        "approval_keyword": APPROVAL_KEYWORD,
        "sealed_at": sealed_at,
        "output_root": output_root,
        "single_use_policy": "ONE_ACCEPTED_EXECUTION",
        "single_use_enforcement": "BEST_EFFORT_LOCAL_EXCLUSIVE_RECEIPT",
        "global_replay_proof": False,
        "rollback_resistant": False,
        "durable_remote_ledger": False,
        "network_isolation": "APPLICATION_LEVEL_NOT_OS_SANDBOX",
        "evidence_purpose_class": "DIAGNOSTIC_NONPROMOTION",
        "source_authority_class": "B_LOCAL_HASHED",
        "confirmatory": False,
        "promotion_eligible": False,
        "score_credit": 0,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    run_scope["run_scope_digest"] = canonical_digest(run_scope)
    return run_scope


def verify_offline_run_scope_digest(scope: Mapping[str, Any]) -> str:
    if not isinstance(scope, Mapping):
        raise ContractError("offline run scope must be an object")
    stored = _require_sha(scope.get("run_scope_digest"), label="run_scope_digest")
    unsigned = dict(scope)
    unsigned.pop("run_scope_digest", None)
    _validate_finite_json(unsigned)
    if canonical_digest(unsigned) != stored:
        raise ContractError("offline run_scope_digest mismatch")
    required = {
        "schema_version": 1,
        "gate_kind": GATE_KIND,
        "status": "RNOD_RUN_SCOPE_FROZEN",
        "authority": False,
        "approval_keyword": APPROVAL_KEYWORD,
        "output_root": FIXED_OUTPUT_ROOT,
        "single_use_policy": "ONE_ACCEPTED_EXECUTION",
        "single_use_enforcement": "BEST_EFFORT_LOCAL_EXCLUSIVE_RECEIPT",
        "global_replay_proof": False,
        "rollback_resistant": False,
        "durable_remote_ledger": False,
        "network_isolation": "APPLICATION_LEVEL_NOT_OS_SANDBOX",
        "evidence_purpose_class": "DIAGNOSTIC_NONPROMOTION",
        "source_authority_class": "B_LOCAL_HASHED",
        "confirmatory": False,
        "promotion_eligible": False,
        "score_credit": 0,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    for key, expected in required.items():
        if scope.get(key) != expected:
            raise ContractError(f"offline run scope {key} safety binding mismatch")
    return stored


def verify_canonical_offline_run_scope(
    registered: OfflineRegisteredRecipe,
    scope: Mapping[str, Any],
) -> str:
    stored = verify_offline_run_scope_digest(scope)
    bindings = scope.get("runtime_bindings")
    if not isinstance(bindings, Mapping):
        raise ContractError("offline run scope runtime_bindings are missing")
    expected = compile_offline_run_scope(registered, bindings)
    if dict(scope) != expected:
        raise ContractError("offline run scope differs from canonical compilation")
    return stored
