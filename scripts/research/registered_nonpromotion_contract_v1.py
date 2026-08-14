from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


GATE_KIND = "registered_nonpromotion_diagnostic_v1"
APPROVAL_KEYWORD = "APPROVED_NONPROMOTION_DIAGNOSTIC_RUN"
DEFAULT_REPOSITORY = "kazuponbaseball-cell/keiba_ai_project"
DEFAULT_BASE_BRANCH = "main"

FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_OUTPUT = re.compile(r"^[A-Za-z0-9_.\-/]+$")

POLICY_RELATIVE_PATH = Path("research/REGISTERED_NONPROMOTION_DIAGNOSTIC_V1.json")
EXPECTED_SCHEMA_PATHS = {
    "policy": "research/schemas/registered_nonpromotion_policy_v1.schema.json",
    "recipe": "research/schemas/registered_nonpromotion_recipe_v1.schema.json",
    "catalog_release": "research/schemas/registered_nonpromotion_catalog_release_v1.schema.json",
    "run_scope": "research/schemas/registered_nonpromotion_run_scope_v1.schema.json",
    "authority_receipt": "research/schemas/registered_nonpromotion_authority_receipt_v1.schema.json",
    "subject_head_snapshot": "research/schemas/registered_nonpromotion_subject_head_snapshot_v1.schema.json",
    "phase_lease": "research/schemas/registered_nonpromotion_phase_lease_v1.schema.json",
    "decision_lease_batch": "research/schemas/registered_nonpromotion_decision_lease_batch_v1.schema.json",
    "decision_consumption_batch": "research/schemas/registered_nonpromotion_decision_consumption_batch_v1.schema.json",
    "lease_consumption_receipt": "research/schemas/registered_nonpromotion_lease_consumption_receipt_v1.schema.json",
    "phase_output_attestation": "research/schemas/registered_nonpromotion_phase_output_attestation_v1.schema.json",
    "phase_output_seal_receipt": "research/schemas/registered_nonpromotion_phase_output_seal_receipt_v1.schema.json",
    "result": "research/schemas/registered_nonpromotion_result_v1.schema.json",
    "cutover_receipt": "research/schemas/registered_nonpromotion_cutover_receipt_v1.schema.json",
}
RUNTIME_MATERIAL_PATHS = {
    "compiler_blob_sha256": "scripts/research/registered_nonpromotion_contract_v1.py",
    "authority_verifier_blob_sha256": "scripts/research/registered_nonpromotion_authority_verifier_v1.py",
    "catalog_validator_blob_sha256": "scripts/research/registered_nonpromotion_catalog_v1.py",
    "executor_blob_sha256": "scripts/research/registered_nonpromotion_supervised_executor_v1.py",
    "runner_blob_sha256": "scripts/research/registered_nonpromotion_supervised_executor_v1.py",
    "result_sealer_blob_sha256": "scripts/research/registered_nonpromotion_result_sealer_v1.py",
    "subject_head_snapshot_schema_sha256": EXPECTED_SCHEMA_PATHS[
        "subject_head_snapshot"
    ],
    "phase_lease_schema_sha256": EXPECTED_SCHEMA_PATHS["phase_lease"],
    "decision_lease_batch_schema_sha256": EXPECTED_SCHEMA_PATHS[
        "decision_lease_batch"
    ],
    "decision_consumption_batch_schema_sha256": EXPECTED_SCHEMA_PATHS[
        "decision_consumption_batch"
    ],
    "lease_consumption_receipt_schema_sha256": EXPECTED_SCHEMA_PATHS[
        "lease_consumption_receipt"
    ],
    "phase_output_attestation_schema_sha256": EXPECTED_SCHEMA_PATHS[
        "phase_output_attestation"
    ],
    "phase_operation_receipt_schema_sha256": EXPECTED_SCHEMA_PATHS[
        "authority_receipt"
    ],
    "phase_output_seal_receipt_schema_sha256": EXPECTED_SCHEMA_PATHS[
        "phase_output_seal_receipt"
    ],
    "approval_evidence_schema_sha256": EXPECTED_SCHEMA_PATHS["authority_receipt"],
}
G2_MATERIAL_PATHS = (
    "scripts/research/shared_g2_durable_ledger_v1.py",
    "scripts/research/shared_g2_lease_authority_v1.py",
)


class ContractError(ValueError):
    """A fail-closed registered diagnostic contract violation."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ContractError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def strict_json_loads(text: str, *, label: str = "JSON") -> Any:
    if not isinstance(text, str):
        raise ContractError(f"{label} must be UTF-8 text")

    def reject_constant(value: str) -> None:
        raise ContractError(f"{label} contains forbidden non-finite value {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ContractError(f"invalid {label}: {exc}") from exc


def _validate_json_value(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"non-finite number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"non-string object key at {path}")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ContractError(f"non-JSON value at {path}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_utf8_lf_sha256(path: Path) -> str:
    """Hash Git-normalized text bytes independent of checkout CRLF settings."""

    try:
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ContractError(f"UTF-8 BOM is forbidden: {path}")
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"unable to hash UTF-8 text {path}: {exc}") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def load_strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"unable to read {path}: {exc}") from exc
    value = strict_json_loads(raw, label=str(path))
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain one JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], required: Iterable[str], *, label: str
) -> None:
    expected = set(required)
    observed = set(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ContractError(f"{label} keys mismatch; missing={missing}, extra={extra}")


def _require_sha(value: Any, *, label: str, git: bool = False) -> str:
    pattern = FULL_GIT_SHA if git else FULL_SHA256
    if (
        not isinstance(value, str)
        or not pattern.fullmatch(value)
        or (not git and value == "0" * 64)
    ):
        kind = "Git SHA" if git else "SHA-256"
        raise ContractError(f"{label} must be a non-zero full lowercase {kind}")
    return value


def _require_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ContractError(f"{label} is not a safe identifier")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{label} must be a boolean")
    return value


def _require_positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _require_iso8601_utc(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContractError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError(f"{label} must be UTC")
    return value


def eligibility_ast_projection(recipe: Mapping[str, Any]) -> dict[str, Any]:
    contract = recipe.get("recipe_contract")
    if not isinstance(contract, dict):
        raise ContractError("recipe_contract is missing")
    return {
        "clause_registry": contract.get("clause_registry"),
        "typed_gate_nodes": contract.get("typed_gate_nodes"),
        "arm_asts": contract.get("arm_asts"),
        "allowed_ast_diff": contract.get("allowed_ast_diff"),
    }


def question_family_projection(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Derive family identity from trusted source/AST registries, not display text."""

    source = recipe.get("question_family_source")
    contract = recipe.get("recipe_contract")
    if not isinstance(source, dict) or not isinstance(contract, dict):
        raise ContractError("question-family source or recipe contract is missing")
    reference_id = source.get("reference_ast_node_id")
    nodes = contract.get("typed_gate_nodes")
    if not isinstance(reference_id, str) or not isinstance(nodes, dict):
        raise ContractError("question-family reference node is invalid")
    if reference_id not in nodes:
        raise ContractError("question-family reference node is not registered")
    return {
        "source_lineage_id": source.get("immutable_source_lineage_id"),
        "reference_ast_node_id": reference_id,
        "reference_ast_node": nodes[reference_id],
        "target_decision_registry_id": source.get("target_decision_registry_id"),
        "population_registry_id": source.get("population_registry_id"),
        "metric_registry_id": source.get("metric_registry_id"),
    }


def validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("schema_version") != 1:
        raise ContractError("unsupported policy schema_version")
    if policy.get("gate_kind") != GATE_KIND:
        raise ContractError("policy gate_kind mismatch")
    if policy.get("authority") is not False:
        raise ContractError("repository policy cannot self-assert runtime authority")
    if policy.get("execution_status") != "EXECUTION_FORBIDDEN":
        raise ContractError("uncut policy must remain EXECUTION_FORBIDDEN")
    if policy.get("repository") != DEFAULT_REPOSITORY:
        raise ContractError("policy repository mismatch")
    if policy.get("base_branch") != DEFAULT_BASE_BRANCH:
        raise ContractError("policy base branch mismatch")
    _require_sha(policy.get("root_declaration_commit"), label="root declaration commit", git=True)
    if policy.get("schema_paths") != EXPECTED_SCHEMA_PATHS:
        raise ContractError("policy schema path registry mismatch")
    activation = policy.get("activation_contract")
    if not isinstance(activation, dict):
        raise ContractError("activation contract missing")
    for key in ("lane_activated", "external_g2_configured", "cutover_receipt_present"):
        if activation.get(key) is not False:
            raise ContractError(f"activation_contract.{key} must remain false in repository")
    for key in (
        "containing_commit_must_be_human_merged_to_main",
        "external_g2_must_be_configured",
        "authenticated_cutover_receipt_required",
    ):
        if activation.get(key) is not True:
            raise ContractError(f"activation_contract.{key} must be true")

    safety = policy.get("safety")
    if not isinstance(safety, dict):
        raise ContractError("policy safety object missing")
    for key in (
        "automatic_execution",
        "automatic_github_approval",
        "workload_network_access",
        "workload_external_api_calls",
        "workload_credential_access",
        "purchase_path_access",
        "production_change",
        "shadow_approval",
        "merge",
        "notification_side_effects",
        "order_side_effects",
        "formal_buy",
        "send_order",
    ):
        if safety.get(key) is not False:
            raise ContractError(f"policy safety.{key} must be false")
    if safety.get("stake") != 0:
        raise ContractError("policy stake must be zero")

    shared = policy.get("shared_g2")
    if not isinstance(shared, dict):
        raise ContractError("shared_g2 contract missing")
    hard_true = (
        "sole_live_authority",
        "global_and_subject_head_atomic_cas",
        "complete_legacy_event_comment_id_and_subject_head_migration_required",
        "old_writer_fence_required",
        "second_remote_compare_required",
        "external_monotonic_witness_required",
        "all_terminal_and_nonterminal_subject_heads_import_required",
        "backup_restore_rollback_and_fork_detection_required",
        "phase_lease_one_shot",
    )
    for key in hard_true:
        if shared.get(key) is not True:
            raise ContractError(f"shared_g2.{key} must be true")
    if shared.get("local_fallback") is not False:
        raise ContractError("local G2 fallback is forbidden")
    if shared.get("dual_writer_allowed") is not False:
        raise ContractError("dual-writer authority is forbidden")

    registry = policy.get("recipe_registry")
    if not isinstance(registry, dict) or registry.get("append_only") is not True:
        raise ContractError("append-only recipe registry missing")
    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ContractError("recipe registry must be non-empty")
    identities: set[tuple[str, int]] = set()
    family_digests: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ContractError(f"recipe registry entry {index} must be an object")
        recipe_id = _require_id(entry.get("recipe_id"), label="recipe_id")
        version = _require_positive_int(entry.get("recipe_version"), label="recipe_version")
        identity = (recipe_id, version)
        if identity in identities:
            raise ContractError("duplicate recipe identity")
        identities.add(identity)
        for key in (
            "canonical_recipe_sha256",
            "source_formula_fingerprint_sha256",
            "question_family_sha256",
            "eligibility_ast_sha256",
        ):
            _require_sha(entry.get(key), label=f"recipe registry {key}")
        family = entry["question_family_sha256"]
        if family in family_digests:
            raise ContractError("more than one active recipe in a question family")
        family_digests.add(family)


def validate_recipe(recipe: Mapping[str, Any], registry_entry: Mapping[str, Any]) -> None:
    if recipe.get("schema_version") != 1:
        raise ContractError("unsupported recipe schema_version")
    if recipe.get("gate_kind") != GATE_KIND:
        raise ContractError("recipe gate_kind mismatch")
    if recipe.get("recipe_id") != registry_entry.get("recipe_id"):
        raise ContractError("recipe_id differs from trusted registry")
    if recipe.get("recipe_version") != registry_entry.get("recipe_version"):
        raise ContractError("recipe_version differs from trusted registry")
    if recipe.get("tier") != registry_entry.get("tier"):
        raise ContractError("recipe tier differs from trusted registry")

    if canonical_digest(recipe) != registry_entry.get("canonical_recipe_sha256"):
        raise ContractError("recipe canonical digest mismatch")
    source = recipe.get("source_design")
    if not isinstance(source, dict) or source.get("authority") is not False:
        raise ContractError("source design must remain non-authoritative")
    if source.get("formula_fingerprint_sha256") != registry_entry.get(
        "source_formula_fingerprint_sha256"
    ):
        raise ContractError("source formula fingerprint mismatch")
    if canonical_digest(recipe.get("recipe_contract")) != source.get(
        "formula_fingerprint_sha256"
    ):
        raise ContractError("recipe_contract formula fingerprint mismatch")
    if canonical_digest(question_family_projection(recipe)) != registry_entry.get(
        "question_family_sha256"
    ):
        raise ContractError("question-family digest mismatch")
    if canonical_digest(eligibility_ast_projection(recipe)) != registry_entry.get(
        "eligibility_ast_sha256"
    ):
        raise ContractError("eligibility AST digest mismatch")

    score = recipe.get("ordinary_strategy_score")
    expected_score = {
        "recorded_total": 46,
        "threshold": 75,
        "threshold_met": False,
        "status": "BLOCKED_SCORE",
        "score_credit": 0,
        "threshold_override_allowed": False,
    }
    if score != expected_score:
        raise ContractError("ordinary score record was omitted or altered")

    comparison = recipe.get("comparison")
    if not isinstance(comparison, dict):
        raise ContractError("comparison contract missing")
    if comparison.get("arm_ids") != ["D0_REFERENCE", "D1_REMOVE_RAW_GATES"]:
        raise ContractError("exact two-arm comparison was altered")
    for key in ("threshold_search_count", "refit_count", "recalibration_count"):
        if comparison.get(key) != 0:
            raise ContractError(f"comparison {key} must remain zero")
    for key in (
        "candidate_identity_change",
        "candidate_rank_or_tier_change",
        "live_policy_or_config_mutation",
    ):
        if comparison.get(key) is not False:
            raise ContractError(f"comparison {key} must remain false")

    contract = recipe.get("recipe_contract")
    if not isinstance(contract, dict):
        raise ContractError("recipe_contract missing")
    clauses = contract.get("clause_registry")
    nodes = contract.get("typed_gate_nodes")
    arms = contract.get("arm_asts")
    if not all(isinstance(item, dict) for item in (clauses, nodes, arms)):
        raise ContractError("typed AST maps are missing")
    allowed_clause_ops = {"boolean_eq", "ge", "lt"}
    for clause_id, clause in clauses.items():
        _require_id(clause_id, label="clause id")
        if not isinstance(clause, dict) or clause.get("op") not in allowed_clause_ops:
            raise ContractError(f"unsupported clause {clause_id}")
        if set(clause) != {"op", "column", "value"}:
            raise ContractError(f"clause {clause_id} has extra or missing fields")
        _require_id(clause.get("column"), label=f"clause {clause_id} column")
    for node_id, node in nodes.items():
        _require_id(node_id, label="node id")
        if not isinstance(node, dict) or set(node) != {"op", "clause_refs"}:
            raise ContractError(f"node {node_id} shape mismatch")
        if node.get("op") != "and":
            raise ContractError("only declarative AND nodes are permitted")
        refs = node.get("clause_refs")
        if not isinstance(refs, list) or not refs or len(refs) != len(set(refs)):
            raise ContractError(f"node {node_id} clause_refs invalid")
        if any(ref not in clauses for ref in refs):
            raise ContractError(f"node {node_id} references an unknown clause")
    for arm_id, arm in arms.items():
        _require_id(arm_id, label="arm id")
        if not isinstance(arm, dict) or set(arm) != {"op", "node_refs"}:
            raise ContractError(f"arm {arm_id} shape mismatch")
        if arm.get("op") != "and":
            raise ContractError("only declarative AND arms are permitted")
        refs = arm.get("node_refs")
        if not isinstance(refs, list) or not refs or len(refs) != len(set(refs)):
            raise ContractError(f"arm {arm_id} node_refs invalid")
        if any(ref not in nodes for ref in refs):
            raise ContractError(f"arm {arm_id} references an unknown node")
    diff = contract.get("allowed_ast_diff")
    if diff != {
        "removed_node_refs": ["RAW_P_GATE_FAMILY"],
        "added_node_refs": [],
        "changed_clause_ids": [],
        "changed_threshold_ids": [],
        "changed_non_ai_clause_ids": [],
    }:
        raise ContractError("the single registered AST deletion was altered")
    if contract.get("expected_relation") != "D0_SUBSET_OF_D1":
        raise ContractError("D0 subset relation was altered")

    safety = recipe.get("safety")
    if not isinstance(safety, dict):
        raise ContractError("recipe safety contract missing")
    for key, value in safety.items():
        if key == "stake":
            if value != 0:
                raise ContractError("recipe stake must be zero")
        elif value is not False:
            raise ContractError(f"recipe safety.{key} must be false")


@dataclass(frozen=True)
class RegisteredRecipe:
    policy: dict[str, Any]
    recipe: dict[str, Any]
    policy_digest: str
    policy_file_sha256: str
    recipe_digest: str
    recipe_file_sha256: str
    schema_bundle_digest: str
    question_family_digest: str
    eligibility_ast_digest: str
    runtime_material_digests: Mapping[str, str]


def resolve_registered_recipe(
    root: Path,
    *,
    recipe_id: str,
    recipe_version: int,
) -> RegisteredRecipe:
    policy_path = root / POLICY_RELATIVE_PATH
    policy = load_strict_json(policy_path)
    validate_policy(policy)
    entries = policy["recipe_registry"]["entries"]
    matches = [
        entry
        for entry in entries
        if entry.get("recipe_id") == recipe_id
        and entry.get("recipe_version") == recipe_version
    ]
    if len(matches) != 1:
        raise ContractError("registered recipe identity must resolve exactly once")
    entry = matches[0]
    raw_path = entry.get("path")
    if not isinstance(raw_path, str):
        raise ContractError("recipe path missing")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ContractError("recipe path escapes repository")
    recipe_path = root / Path(*pure.parts)
    recipe = load_strict_json(recipe_path)
    validate_recipe(recipe, entry)
    schema_materials: list[dict[str, str]] = []
    for schema_id, raw_schema_path in sorted(EXPECTED_SCHEMA_PATHS.items()):
        pure_schema = PurePosixPath(raw_schema_path)
        if pure_schema.is_absolute() or ".." in pure_schema.parts:
            raise ContractError("schema path escapes repository")
        schema_path = root / Path(*pure_schema.parts)
        schema = load_strict_json(schema_path)
        if schema.get("$schema") != "http://json-schema.org/draft-07/schema#":
            raise ContractError(f"schema {schema_id} is not Draft-07")
        if schema.get("additionalProperties") is not False:
            raise ContractError(f"schema {schema_id} root must reject additional properties")
        schema_materials.append(
            {
                "schema_id": schema_id,
                "path": raw_schema_path,
                "content_sha256": normalized_utf8_lf_sha256(schema_path),
            }
        )
    runtime_material_digests = {
        field: normalized_utf8_lf_sha256(root / Path(*PurePosixPath(path).parts))
        for field, path in RUNTIME_MATERIAL_PATHS.items()
    }
    g2_materials = [
        {
            "path": path,
            "content_sha256": normalized_utf8_lf_sha256(
                root / Path(*PurePosixPath(path).parts)
            ),
        }
        for path in G2_MATERIAL_PATHS
    ]
    runtime_material_digests["g2_authority_service_blob_sha256"] = canonical_digest(
        g2_materials
    )
    runtime_material_digests["capability_profile_sha256"] = canonical_digest(
        policy["capability_profiles"]
    )
    runtime_material_digests["schema_bundle_sha256"] = canonical_digest(
        schema_materials
    )
    runtime_material_digests["policy_blob_sha256"] = normalized_utf8_lf_sha256(
        policy_path
    )
    runtime_material_digests["recipe_blob_sha256"] = normalized_utf8_lf_sha256(
        recipe_path
    )
    return RegisteredRecipe(
        policy=dict(policy),
        recipe=dict(recipe),
        policy_digest=canonical_digest(policy),
        policy_file_sha256=runtime_material_digests["policy_blob_sha256"],
        recipe_digest=canonical_digest(recipe),
        recipe_file_sha256=runtime_material_digests["recipe_blob_sha256"],
        schema_bundle_digest=runtime_material_digests["schema_bundle_sha256"],
        question_family_digest=canonical_digest(question_family_projection(recipe)),
        eligibility_ast_digest=canonical_digest(eligibility_ast_projection(recipe)),
        runtime_material_digests=runtime_material_digests,
    )


def _to_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{label} must be numeric")
    if isinstance(value, (str, int, float)):
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{label} must be numeric") from exc
        if math.isfinite(parsed):
            return parsed
    raise ContractError(f"{label} must be finite")


def evaluate_clause(clause: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    column = clause["column"]
    if column not in row:
        raise ContractError(f"candidate row lacks required column {column}")
    op = clause["op"]
    expected = clause["value"]
    observed = row[column]
    if op == "boolean_eq":
        if type(observed) is not bool or type(expected) is not bool:
            raise ContractError(f"boolean clause {column} requires booleans")
        return observed is expected
    left = _to_float(observed, label=column)
    right = _to_float(expected, label=f"threshold for {column}")
    if op == "ge":
        return left >= right
    if op == "lt":
        return left < right
    raise ContractError(f"unsupported clause operator {op}")


def evaluate_arm(
    recipe_contract: Mapping[str, Any], arm_id: str, row: Mapping[str, Any]
) -> bool:
    clauses = recipe_contract["clause_registry"]
    nodes = recipe_contract["typed_gate_nodes"]
    arms = recipe_contract["arm_asts"]
    if arm_id not in arms:
        raise ContractError(f"unknown arm {arm_id}")
    arm = arms[arm_id]
    return all(
        all(evaluate_clause(clauses[ref], row) for ref in nodes[node_id]["clause_refs"])
        for node_id in arm["node_refs"]
    )


def evaluate_registered_decisions(
    recipe: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    contract = recipe["recipe_contract"]
    output: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in rows:
        race_id = row.get("race_id")
        candidate_key = row.get("candidate_key")
        if not isinstance(race_id, str) or not isinstance(candidate_key, str):
            raise ContractError("candidate key fields must be strings")
        key = (race_id, candidate_key)
        if key in seen_keys:
            raise ContractError(f"duplicate candidate key {key}")
        seen_keys.add(key)
        d0 = evaluate_arm(contract, "D0_REFERENCE", row)
        d1 = evaluate_arm(contract, "D1_REMOVE_RAW_GATES", row)
        if d0 and not d1:
            raise ContractError("D0 true / D1 false violates registered subset relation")
        output.append(
            {
                "race_id": race_id,
                "candidate_key": candidate_key,
                "d0_eligible": d0,
                "d1_eligible": d1,
            }
        )
    return output


RUN_BINDING_KEYS = {
    "repository",
    "base_branch",
    "run_scope_base_commit",
    "verified_current_main_sha",
    "approvers_blob_sha",
    "approvers_content_sha256",
    "activation_receipt_sha256",
    "cutover_receipt_sha256",
    "schema_bundle_sha256",
    "approval_evidence_schema_sha256",
    "capability_profile_sha256",
    "policy_blob_sha256",
    "recipe_blob_sha256",
    "compiler_blob_sha256",
    "authority_verifier_blob_sha256",
    "catalog_validator_blob_sha256",
    "executor_blob_sha256",
    "runner_blob_sha256",
    "result_sealer_blob_sha256",
    "g2_authority_service_blob_sha256",
    "subject_head_snapshot_schema_sha256",
    "phase_lease_schema_sha256",
    "decision_lease_batch_schema_sha256",
    "decision_consumption_batch_schema_sha256",
    "lease_consumption_receipt_schema_sha256",
    "phase_output_attestation_schema_sha256",
    "phase_operation_receipt_schema_sha256",
    "phase_output_seal_receipt_schema_sha256",
    "environment_manifest_sha256",
    "catalog_release_id",
    "catalog_release_sha256",
    "catalog_release_status",
    "catalog_release_revoked",
    "catalog_status_receipt_sha256",
    "candidate_entry_sha256",
    "candidate_schema_sha256",
    "candidate_provenance_sha256",
    "p_action_cross_source_equality_attestation_sha256",
    "candidate_materializer_usecols_sha256",
    "decision_base_lineage_sha256",
    "settlement_entry_sha256",
    "settlement_schema_sha256",
    "settlement_provenance_sha256",
    "official_settlement_provenance_sha256",
    "cohort_manifest_sha256",
    "ordered_race_set_sha256",
    "output_root",
    "sealed_at",
    "expected_pregrant_global_head",
    "expected_pregrant_subject_head",
    "cutover_epoch",
    "external_witness_checkpoint_sha256",
}


def compile_run_scope(
    registered: RegisteredRecipe,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(bindings, RUN_BINDING_KEYS, label="run bindings")
    if bindings["repository"] != DEFAULT_REPOSITORY:
        raise ContractError("run repository mismatch")
    if bindings["base_branch"] != DEFAULT_BASE_BRANCH:
        raise ContractError("run base branch mismatch")
    for key in ("run_scope_base_commit", "verified_current_main_sha", "approvers_blob_sha"):
        _require_sha(bindings[key], label=key, git=True)
    for key in (
        "approvers_content_sha256",
        "activation_receipt_sha256",
        "cutover_receipt_sha256",
        "schema_bundle_sha256",
        "approval_evidence_schema_sha256",
        "capability_profile_sha256",
        "policy_blob_sha256",
        "recipe_blob_sha256",
        "compiler_blob_sha256",
        "authority_verifier_blob_sha256",
        "catalog_validator_blob_sha256",
        "executor_blob_sha256",
        "runner_blob_sha256",
        "result_sealer_blob_sha256",
        "g2_authority_service_blob_sha256",
        "subject_head_snapshot_schema_sha256",
        "phase_lease_schema_sha256",
        "decision_lease_batch_schema_sha256",
        "decision_consumption_batch_schema_sha256",
        "lease_consumption_receipt_schema_sha256",
        "phase_output_attestation_schema_sha256",
        "phase_operation_receipt_schema_sha256",
        "phase_output_seal_receipt_schema_sha256",
        "environment_manifest_sha256",
        "catalog_release_sha256",
        "catalog_status_receipt_sha256",
        "candidate_entry_sha256",
        "candidate_schema_sha256",
        "candidate_provenance_sha256",
        "p_action_cross_source_equality_attestation_sha256",
        "candidate_materializer_usecols_sha256",
        "decision_base_lineage_sha256",
        "settlement_entry_sha256",
        "settlement_schema_sha256",
        "settlement_provenance_sha256",
        "official_settlement_provenance_sha256",
        "cohort_manifest_sha256",
        "ordered_race_set_sha256",
        "expected_pregrant_global_head",
        "expected_pregrant_subject_head",
        "external_witness_checkpoint_sha256",
    ):
        _require_sha(bindings[key], label=key)
    if bindings["catalog_release_status"] != "ACTIVE":
        raise ContractError("catalog release must be ACTIVE")
    if bindings["catalog_release_revoked"] is not False:
        raise ContractError("revoked catalog releases cannot be bound")
    _require_positive_int(bindings["cutover_epoch"], label="cutover_epoch")
    catalog_release_id = _require_id(bindings["catalog_release_id"], label="catalog_release_id")
    sealed_at = _require_iso8601_utc(bindings["sealed_at"], label="sealed_at")
    output_root = bindings["output_root"]
    if (
        not isinstance(output_root, str)
        or not SAFE_OUTPUT.fullmatch(output_root)
        or output_root.startswith(("/", "\\"))
        or ".." in PurePosixPath(output_root).parts
    ):
        raise ContractError("output_root must be a safe repository-relative path")
    for field, expected_digest in registered.runtime_material_digests.items():
        if bindings[field] != expected_digest:
            raise ContractError(f"{field} differs from the resolved repository material")

    recipe = registered.recipe
    exact_subject = {
        "recipe_id": recipe["recipe_id"],
        "recipe_version": recipe["recipe_version"],
        "recipe_digest": registered.recipe_digest,
        "recipe_blob_sha256": registered.recipe_file_sha256,
        "catalog_release_id": catalog_release_id,
        "catalog_release_sha256": bindings["catalog_release_sha256"],
        "candidate_entry_sha256": bindings["candidate_entry_sha256"],
        "candidate_schema_sha256": bindings["candidate_schema_sha256"],
        "candidate_provenance_sha256": bindings["candidate_provenance_sha256"],
        "p_action_cross_source_equality_attestation_sha256": bindings[
            "p_action_cross_source_equality_attestation_sha256"
        ],
        "settlement_entry_sha256": bindings["settlement_entry_sha256"],
        "settlement_schema_sha256": bindings["settlement_schema_sha256"],
        "settlement_provenance_sha256": bindings["settlement_provenance_sha256"],
        "cohort_manifest_sha256": bindings["cohort_manifest_sha256"],
        "ordered_race_set_sha256": bindings["ordered_race_set_sha256"],
        "environment_manifest_sha256": bindings["environment_manifest_sha256"],
        "output_root": output_root,
    }
    semantic_subject = {
        "recipe_id": recipe["recipe_id"],
        "recipe_version": recipe["recipe_version"],
        "recipe_digest": registered.recipe_digest,
        "question_family_digest": registered.question_family_digest,
        "gate_kind": GATE_KIND,
        "tier": recipe["tier"],
        "registered_transform_id": recipe["comparison"]["registered_transform_id"],
        "source_lineage_id": recipe["question_family_source"][
            "immutable_source_lineage_id"
        ],
        "reference_ast_node_id": recipe["question_family_source"][
            "reference_ast_node_id"
        ],
        "target_decision_registry_id": recipe["question_family_source"][
            "target_decision_registry_id"
        ],
        "population_registry_id": recipe["question_family_source"]["population_registry_id"],
        "metric_registry_id": recipe["question_family_source"]["metric_registry_id"],
        "cohort_id": recipe["cohort"]["cohort_id"],
        "cohort_rule": recipe["cohort"],
        "candidate_identity_contract": recipe["catalog_requirements"]["candidate_role"],
        "candidate_identity_change": recipe["comparison"]["candidate_identity_change"],
        "candidate_rank_or_tier_change": recipe["comparison"][
            "candidate_rank_or_tier_change"
        ],
        "ordinary_strategy_score": recipe["ordinary_strategy_score"],
        "metric": recipe["metric"],
        "sensitivity": recipe["sensitivity"],
        "source_formula_fingerprint_sha256": recipe["source_design"][
            "formula_fingerprint_sha256"
        ],
        "lineage_sources": recipe["source_inventory"]["lineage_sources"],
    }
    run_scope = {
        "schema_version": 1,
        "gate_kind": GATE_KIND,
        "status": "RND_RUN_SCOPE_FROZEN",
        "authority": False,
        "repository": DEFAULT_REPOSITORY,
        "base_branch": DEFAULT_BASE_BRANCH,
        "run_scope_base_commit": bindings["run_scope_base_commit"],
        "verified_current_main_sha": bindings["verified_current_main_sha"],
        "approvers_blob_sha": bindings["approvers_blob_sha"],
        "approvers_content_sha256": bindings["approvers_content_sha256"],
        "policy_digest": registered.policy_digest,
        "recipe_id": recipe["recipe_id"],
        "recipe_version": recipe["recipe_version"],
        "recipe_digest": registered.recipe_digest,
        "question_family_digest": registered.question_family_digest,
        "eligibility_ast_digest": registered.eligibility_ast_digest,
        "tier": recipe["tier"],
        "ordinary_strategy_score": recipe["ordinary_strategy_score"],
        "resolved_contracts": {
            "comparison": recipe["comparison"],
            "cohort": recipe["cohort"],
            "metric": recipe["metric"],
            "sensitivity": recipe["sensitivity"],
            "bootstrap": recipe["bootstrap"],
            "execution_contract": recipe["execution_contract"],
            "phase_plan": recipe["phase_plan"],
            "replica_topology": recipe["replica_topology"],
            "output_contract": recipe["output_contract"],
            "safety": recipe["safety"],
        },
        "runtime_bindings": dict(bindings),
        "semantic_subject": semantic_subject,
        "semantic_subject_digest": canonical_digest(semantic_subject),
        "exact_subject": exact_subject,
        "exact_subject_digest": canonical_digest(exact_subject),
        "approval_keyword": APPROVAL_KEYWORD,
        "sealed_at": sealed_at,
        "output_root": output_root,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    run_scope["run_scope_digest"] = canonical_digest(run_scope)
    return run_scope


def verify_run_scope_digest(run_scope: Mapping[str, Any]) -> str:
    if not isinstance(run_scope, dict):
        raise ContractError("run scope must be an object")
    stored = run_scope.get("run_scope_digest")
    _require_sha(stored, label="run_scope_digest")
    unsigned = dict(run_scope)
    unsigned.pop("run_scope_digest", None)
    expected = canonical_digest(unsigned)
    if stored != expected:
        raise ContractError("run_scope_digest mismatch")
    if run_scope.get("gate_kind") != GATE_KIND:
        raise ContractError("run scope gate_kind mismatch")
    if run_scope.get("status") != "RND_RUN_SCOPE_FROZEN":
        raise ContractError("run scope must be frozen before approval")
    if run_scope.get("authority") is not False:
        raise ContractError("run scope cannot self-assert authority")
    if run_scope.get("formal_buy") is not False or run_scope.get("send_order") is not False:
        raise ContractError("run scope BUY/order flags must be false")
    if run_scope.get("stake") != 0:
        raise ContractError("run scope stake must be zero")
    return stored


def verify_canonical_run_scope(
    registered: RegisteredRecipe, run_scope: Mapping[str, Any]
) -> str:
    """Recompile a frozen scope from the registered recipe and require byte semantics.

    A self-consistent caller hash is not sufficient: the semantic/exact subjects,
    resolved contracts, and every runtime binding must be the deterministic output
    of the current-main registered recipe compiler.
    """

    stored = verify_run_scope_digest(run_scope)
    bindings = run_scope.get("runtime_bindings")
    if not isinstance(bindings, Mapping):
        raise ContractError("run scope runtime_bindings are missing")
    expected = compile_run_scope(registered, bindings)
    if dict(run_scope) != expected:
        raise ContractError("run scope differs from canonical registered compilation")
    return stored


def verify_run_scope_catalog_binding(
    registered: RegisteredRecipe,
    run_scope: Mapping[str, Any],
    catalog_metadata: Mapping[str, Any],
) -> None:
    """Bind the frozen scope to metadata returned by the strict catalog resolver."""

    verify_canonical_run_scope(registered, run_scope)
    if not isinstance(catalog_metadata, Mapping):
        raise ContractError("catalog metadata binding must be an object")
    candidate = catalog_metadata.get("candidate_entry")
    settlement = catalog_metadata.get("settlement_entry")
    if not isinstance(candidate, Mapping) or not isinstance(settlement, Mapping):
        raise ContractError("catalog metadata lacks its two role entries")
    candidate_attestations = candidate.get("role_attestations")
    settlement_attestations = settlement.get("role_attestations")
    if not isinstance(candidate_attestations, Mapping) or not isinstance(
        settlement_attestations, Mapping
    ):
        raise ContractError("catalog role attestations are missing")
    bindings = run_scope.get("runtime_bindings")
    if not isinstance(bindings, Mapping):
        raise ContractError("run runtime_bindings are missing")
    expected = {
        "catalog_release_id": catalog_metadata.get("release_id"),
        "catalog_release_sha256": catalog_metadata.get("release_manifest_sha256"),
        "catalog_status_receipt_sha256": catalog_metadata.get(
            "status_receipt_sha256"
        ),
        "candidate_entry_sha256": candidate.get("content_sha256"),
        "candidate_schema_sha256": candidate.get("schema_sha256"),
        "candidate_provenance_sha256": candidate.get("provenance_sha256"),
        "p_action_cross_source_equality_attestation_sha256": candidate_attestations.get(
            "p_action_cross_source_equality_attestation_sha256"
        ),
        "candidate_materializer_usecols_sha256": candidate_attestations.get(
            "candidate_materializer_usecols_sha256"
        ),
        "decision_base_lineage_sha256": candidate_attestations.get(
            "decision_base_lineage_sha256"
        ),
        "settlement_entry_sha256": settlement.get("content_sha256"),
        "settlement_schema_sha256": settlement.get("schema_sha256"),
        "settlement_provenance_sha256": settlement.get("provenance_sha256"),
        "official_settlement_provenance_sha256": settlement_attestations.get(
            "official_settlement_provenance_sha256"
        ),
        "ordered_race_set_sha256": catalog_metadata.get(
            "ordered_race_id_set_sha256"
        ),
    }
    for key, value in expected.items():
        if bindings.get(key) != value:
            raise ContractError(f"run scope/catalog metadata mismatch at {key}")
    if bindings.get("catalog_release_status") != "ACTIVE":
        raise ContractError("run scope does not bind an ACTIVE catalog release")
    if bindings.get("catalog_release_revoked") is not False:
        raise ContractError("run scope binds a revoked catalog release")
    if catalog_metadata.get("race_count") != run_scope.get("resolved_contracts", {}).get(
        "cohort", {}
    ).get("race_count"):
        raise ContractError("catalog race count differs from registered cohort")


__all__ = [
    "APPROVAL_KEYWORD",
    "ContractError",
    "GATE_KIND",
    "G2_MATERIAL_PATHS",
    "EXPECTED_SCHEMA_PATHS",
    "RUNTIME_MATERIAL_PATHS",
    "RegisteredRecipe",
    "canonical_digest",
    "canonical_json_bytes",
    "compile_run_scope",
    "eligibility_ast_projection",
    "evaluate_registered_decisions",
    "load_strict_json",
    "normalized_utf8_lf_sha256",
    "question_family_projection",
    "resolve_registered_recipe",
    "sha256_file",
    "strict_json_loads",
    "validate_policy",
    "validate_recipe",
    "verify_canonical_run_scope",
    "verify_run_scope_digest",
    "verify_run_scope_catalog_binding",
]
