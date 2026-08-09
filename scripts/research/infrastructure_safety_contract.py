from __future__ import annotations

import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable


GATE_KIND = "infrastructure_safety_v1"
PROPOSAL_CONTRACT = {
    "name": "infrastructure_safety_proposal",
    "version": 1,
}
RUN_CONTRACT = {
    "name": "infrastructure_safety_run",
    "version": 1,
}
QUEUE_SCHEMA_VERSION = 3
EVENT_SCHEMA_VERSION = 3
CHANGED_PATH_MANIFEST_VERSION = 1

SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SAFE_TEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
SAFE_UNITTEST_MODULE = re.compile(r"^research\.infra_tests\.test_[A-Za-z0-9_]+$")
SAFE_UNITTEST_PATH = re.compile(r"^research/infra_tests/test_[A-Za-z0-9_]+\.py$")
SAFE_REPOSITORY_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")
DUNDER_IDENTIFIER = re.compile(r"^__[A-Za-z0-9_]+__$")
DUNDER_STRING = re.compile(r"__[A-Za-z0-9_]+__")
SAFE_DUNDER_IDENTIFIERS = {"__name__"}
SAFE_DUNDER_STRINGS = {"__main__"}

HARD_CHECKS = {
    "bounded_compute",
    "compatibility_locked",
    "deterministic_lineage",
    "expected_paths_firewalled",
    "gate_identity_bound",
    "non_race_scope",
    "policy_hash_bound",
    "single_variant_no_threshold_search",
    "structured_test_matrix",
    "synthetic_inputs_only",
    "zero_side_effect_safety",
}

POLICY_FIELDS = {
    "schema_version",
    "gate_kind",
    "mode",
    "proposal_contract",
    "run_contract",
    "queue_schema_version",
    "event_schema_version",
    "allowed_changed_path_prefixes",
    "allowed_config_path_prefixes",
    "allowed_synthetic_path_prefixes",
    "forbidden_changed_path_prefixes",
    "root_of_trust_paths",
    "forbidden_path_tokens",
    "allowed_command_template_ids",
    "allowed_import_modules",
    "forbidden_import_roots",
    "forbidden_dynamic_call_names",
    "maximum_synthetic_fixture_bytes",
    "required_hard_checks",
    "safety",
}
PROPOSAL_FIELDS = {
    "contract",
    "gate_kind",
    "experiment_id",
    "title",
    "change_hypothesis",
    "null_hypothesis",
    "safety_objective",
    "failure_modes",
    "in_scope",
    "out_of_scope",
    "expected_changed_paths",
    "input_classes",
    "source_as_of",
    "lineage_hash_requirements",
    "test_matrix",
    "primary_metric",
    "required_effect",
    "rejection_gate",
    "stop_conditions",
    "compute_budget",
    "allowed_variant_count",
    "allowed_threshold_search_count",
    "base_commit",
    "gate_policy",
    "compatibility_contract",
    "formal_buy",
    "send_order",
    "stake",
}
RUN_FIELDS = {
    "contract",
    "gate_kind",
    "proposal_scope",
    "proposal_scope_digest",
    "execution_commit_sha",
    "config_hashes",
    "synthetic_input_hashes",
    "changed_path_manifest",
    "changed_path_manifest_digest",
    "dependency_environment_manifest",
    "seed",
    "command_plan",
    "exact_execution_argv",
    "execution_context",
    "execution_kind",
    "external_api_calls",
    "network_calls",
    "actual_codex_dispatch",
    "real_data_execution",
    "roi_calculation",
    "production_change",
    "credential_access",
    "notification_side_effects",
    "order_side_effects",
    "formal_buy",
    "send_order",
    "stake",
}
QUEUE_FIELDS = {
    "schema_version",
    "gate_kind",
    "contract_version",
    "experiment_id",
    "title",
    "owner",
    "created_at",
    "status",
    "gate_evaluation",
    "proposal_scope",
    "proposal_scope_file",
    "proposal_scope_digest",
    "human_approved_to_prepare",
    "human_approved_to_run",
    "automatic_execution_allowed",
    "execution_authorized",
    "production_approved",
    "merge_approved",
    "buy_approved",
    "production_change_allowed",
    "merge_allowed",
    "buy_logic_change_allowed",
    "formal_buy",
    "send_order",
    "stake",
    "experiment_markdown",
    "notes",
}
EVENT_FIELDS = {
    "schema_version",
    "event_id",
    "sequence",
    "experiment_id",
    "gate_kind",
    "gate_contract_version",
    "status",
    "previous_status",
    "previous_event_id",
    "occurred_at",
    "actor",
    "gate_evaluation",
    "proposal_scope_digest",
    "run_scope_digest",
    "github_trust_evidence",
    "gate_policy_evidence",
    "main_registry_evidence",
    "approval_evidence",
    "revalidated_approval_evidence",
    "human_approved",
    "human_prepare_approval_recorded",
    "human_run_approval_recorded",
    "preparation_authorized",
    "synthetic_fixture_tests_allowed",
    "real_data_execution_allowed",
    "automatic_execution_allowed",
    "execution_authorized",
    "production_approved",
    "merge_approved",
    "buy_approved",
    "production_change_allowed",
    "merge_allowed",
    "buy_logic_change_allowed",
    "formal_buy",
    "send_order",
    "stake",
    "execution_kind",
    "artifacts",
    "notes",
    "queue_file",
    "run_scope_file",
}

POLICY_SAFETY_FIELDS = {
    "external_api_calls",
    "network_calls",
    "actual_codex_dispatch",
    "real_data_execution",
    "roi_calculation",
    "production_change",
    "credential_access",
    "notification_side_effects",
    "order_side_effects",
    "formal_buy",
    "send_order",
    "stake",
}
ALLOWED_INPUT_CLASSES = {"git_tracked_contract", "synthetic_fixture"}
REQUIRED_TEST_KINDS = {"positive", "negative", "backward_compatibility"}
INFRASTRUCTURE_STATUSES = {
    "blocked_gate",
    "proposed",
    "approved_to_prepare",
    "preparing",
    "run_approval_required",
    "approved_to_run",
    "running",
    "review_required",
    "rejected",
    "invalid",
}
def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str, *, label: str = "JSON") -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=_strict_object_pairs,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON: {exc}") from exc


def strict_json_load(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read strict JSON from {path}: {exc}") from exc
    return strict_json_loads(text, label=str(path))


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _require_exact_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(value))
    unexpected = sorted(set(value) - fields)
    if missing:
        raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{label} contains unexpected field(s): {', '.join(unexpected)}")


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    canonical_json_bytes(value)
    return value


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain a NUL byte")
    return value.strip()


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_false(value: Any, field: str) -> None:
    if value is not False:
        raise ValueError(f"{field} must be false")


def _require_zero(value: Any, field: str) -> None:
    if value != 0 or isinstance(value, bool):
        raise ValueError(f"{field} must be 0")


def _require_positive_int(value: Any, field: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return value


def _require_sha256(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    text = _require_nonempty_string(value, field).lower()
    if not FULL_SHA256.fullmatch(text):
        raise ValueError(f"{field} must be a full lowercase SHA-256 digest")
    return text


def _require_git_sha(value: Any, field: str) -> str:
    text = _require_nonempty_string(value, field).lower()
    if not FULL_GIT_SHA.fullmatch(text):
        raise ValueError(f"{field} must be a full lowercase 40-character Git SHA")
    return text


def _require_git_blob_sha(
    value: Any,
    field: str,
    *,
    nullable: bool = False,
) -> str | None:
    if value is None and nullable:
        return None
    text = _require_nonempty_string(value, field).lower()
    if not FULL_GIT_SHA.fullmatch(text):
        raise ValueError(f"{field} must be a full lowercase 40-character Git blob SHA")
    return text


def _require_repository_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    if value != value.strip():
        raise ValueError(f"{field} must not have leading or trailing whitespace")
    text = value
    if not SAFE_REPOSITORY_PATH.fullmatch(text):
        raise ValueError(
            f"{field} must use only ASCII letters, digits, dot, underscore, hyphen, and slash"
        )
    if "\\" in text:
        raise ValueError(f"{field} must use forward slashes")
    if ":" in text:
        raise ValueError(f"{field} must not contain ':' or a Windows ADS path")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a normalized repository-relative path")
    if path.as_posix() != text:
        raise ValueError(f"{field} must be a canonical repository-relative path")
    if any(part.endswith((".", " ")) for part in path.parts):
        raise ValueError(f"{field} path components must not end with dot or space")
    return path.as_posix()


def _require_string_list(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
    sort_as_set: bool = True,
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{field} must be {qualifier} of strings")
    normalized = [_require_nonempty_string(item, f"{field}[]") for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must not contain duplicates")
    return sorted(normalized) if sort_as_set else normalized


def _require_path_list(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{field} must be {qualifier} of repository paths")
    normalized = [_require_repository_path(item, f"{field}[]") for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must not contain duplicates")
    casefolded = [item.casefold() for item in normalized]
    if len(casefolded) != len(set(casefolded)):
        raise ValueError(f"{field} must not contain case-fold-colliding paths")
    return sorted(normalized)


def _normalize_contract(value: Any, expected: dict[str, Any], field: str) -> dict[str, Any]:
    payload = _require_object(value, field)
    _require_exact_fields(payload, {"name", "version"}, field)
    if payload != expected:
        raise ValueError(f"{field} must equal {canonical_json_text(expected)}")
    return dict(expected)


def _normalize_manifest_ref(value: Any, field: str) -> dict[str, str]:
    payload = _require_object(value, field)
    _require_exact_fields(payload, {"path", "sha256"}, field)
    digest = _require_sha256(payload["sha256"], f"{field}.sha256")
    assert isinstance(digest, str)
    return {
        "path": _require_repository_path(payload["path"], f"{field}.path"),
        "sha256": digest,
    }


def _normalize_manifest_refs(value: Any, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list of manifest references")
    normalized = [_normalize_manifest_ref(item, f"{field}[]") for item in value]
    paths = [item["path"] for item in normalized]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{field} must not contain duplicate paths")
    if len(paths) != len({path.casefold() for path in paths}):
        raise ValueError(f"{field} must not contain case-fold-colliding paths")
    return sorted(normalized, key=lambda item: item["path"])


def _normalize_policy_safety(value: Any) -> dict[str, Any]:
    payload = _require_object(value, "policy.safety")
    _require_exact_fields(payload, POLICY_SAFETY_FIELDS, "policy.safety")
    for field in POLICY_SAFETY_FIELDS - {"network_calls", "stake"}:
        _require_false(payload[field], f"policy.safety.{field}")
    _require_zero(payload["network_calls"], "policy.safety.network_calls")
    _require_zero(payload["stake"], "policy.safety.stake")
    return {field: (0 if field in {"network_calls", "stake"} else False) for field in sorted(POLICY_SAFETY_FIELDS)}


def _normalize_policy_prefixes(value: Any, field: str) -> list[str]:
    prefixes = _require_string_list(value, field)
    for prefix in prefixes:
        if not prefix.endswith("/"):
            raise ValueError(f"{field} entries must end with '/'")
        normalized = _require_repository_path(prefix[:-1], f"{field}[]") + "/"
        if normalized != prefix:
            raise ValueError(f"{field} entries must be canonical repository prefixes")
    if len(prefixes) != len({prefix.casefold() for prefix in prefixes}):
        raise ValueError(f"{field} must not contain case-fold-colliding prefixes")
    return prefixes


def normalize_gate_policy(value: Any) -> dict[str, Any]:
    payload = _require_object(value, "infrastructure gate policy")
    _require_exact_fields(payload, POLICY_FIELDS, "infrastructure gate policy")
    if payload["schema_version"] != 1:
        raise ValueError("policy.schema_version must be 1")
    if payload["gate_kind"] != GATE_KIND:
        raise ValueError(f"policy.gate_kind must be {GATE_KIND}")
    if payload["mode"] != "all_hard_checks_must_pass":
        raise ValueError("policy.mode must be all_hard_checks_must_pass")
    if payload["queue_schema_version"] != QUEUE_SCHEMA_VERSION:
        raise ValueError(f"policy.queue_schema_version must be {QUEUE_SCHEMA_VERSION}")
    if payload["event_schema_version"] != EVENT_SCHEMA_VERSION:
        raise ValueError(f"policy.event_schema_version must be {EVENT_SCHEMA_VERSION}")

    allowed_prefixes = _normalize_policy_prefixes(
        payload["allowed_changed_path_prefixes"],
        "policy.allowed_changed_path_prefixes",
    )
    config_prefixes = _normalize_policy_prefixes(
        payload["allowed_config_path_prefixes"],
        "policy.allowed_config_path_prefixes",
    )
    synthetic_prefixes = _normalize_policy_prefixes(
        payload["allowed_synthetic_path_prefixes"],
        "policy.allowed_synthetic_path_prefixes",
    )
    forbidden_prefixes = _normalize_policy_prefixes(
        payload["forbidden_changed_path_prefixes"],
        "policy.forbidden_changed_path_prefixes",
    )

    root_paths = _require_path_list(payload["root_of_trust_paths"], "policy.root_of_trust_paths")
    forbidden_tokens = [
        item.lower()
        for item in _require_string_list(
            payload["forbidden_path_tokens"],
            "policy.forbidden_path_tokens",
        )
    ]
    template_ids = _require_string_list(
        payload["allowed_command_template_ids"],
        "policy.allowed_command_template_ids",
    )
    unknown_templates = sorted(set(template_ids) - set(command_template_ids()))
    if unknown_templates:
        raise ValueError(
            "policy allows unknown command template(s): " + ", ".join(unknown_templates)
        )
    allowed_import_modules = [
        item.casefold()
        for item in _require_string_list(
            payload["allowed_import_modules"],
            "policy.allowed_import_modules",
        )
    ]
    if any(
        not re.fullmatch(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*", item)
        for item in allowed_import_modules
    ):
        raise ValueError("policy.allowed_import_modules contains an invalid module name")
    forbidden_import_roots = [
        item.casefold()
        for item in _require_string_list(
            payload["forbidden_import_roots"],
            "policy.forbidden_import_roots",
        )
    ]
    forbidden_dynamic_call_names = _require_string_list(
        payload["forbidden_dynamic_call_names"],
        "policy.forbidden_dynamic_call_names",
    )
    maximum_fixture_bytes = _require_positive_int(
        payload["maximum_synthetic_fixture_bytes"],
        "policy.maximum_synthetic_fixture_bytes",
        maximum=16 * 1024 * 1024,
    )
    required_checks = set(
        _require_string_list(payload["required_hard_checks"], "policy.required_hard_checks")
    )
    if required_checks != HARD_CHECKS:
        raise ValueError("policy.required_hard_checks must equal the code-owned hard-check set")

    return {
        "schema_version": 1,
        "gate_kind": GATE_KIND,
        "mode": "all_hard_checks_must_pass",
        "proposal_contract": _normalize_contract(
            payload["proposal_contract"], PROPOSAL_CONTRACT, "policy.proposal_contract"
        ),
        "run_contract": _normalize_contract(
            payload["run_contract"], RUN_CONTRACT, "policy.run_contract"
        ),
        "queue_schema_version": QUEUE_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "allowed_changed_path_prefixes": allowed_prefixes,
        "allowed_config_path_prefixes": config_prefixes,
        "allowed_synthetic_path_prefixes": synthetic_prefixes,
        "forbidden_changed_path_prefixes": forbidden_prefixes,
        "root_of_trust_paths": root_paths,
        "forbidden_path_tokens": sorted(forbidden_tokens),
        "allowed_command_template_ids": template_ids,
        "allowed_import_modules": sorted(allowed_import_modules),
        "forbidden_import_roots": sorted(forbidden_import_roots),
        "forbidden_dynamic_call_names": forbidden_dynamic_call_names,
        "maximum_synthetic_fixture_bytes": maximum_fixture_bytes,
        "required_hard_checks": sorted(HARD_CHECKS),
        "safety": _normalize_policy_safety(payload["safety"]),
    }


def load_gate_policy(path: Path) -> tuple[dict[str, Any], str]:
    policy = normalize_gate_policy(strict_json_load(path))
    return policy, sha256_file(path)


def _lexical_tokens(value: str) -> set[str]:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return {
        token
        for token in re.split(r"[^a-z0-9]+", separated.casefold())
        if token
    }


def _path_tokens(path: str) -> set[str]:
    return {
        token
        for part in PurePosixPath(path).parts
        for token in _lexical_tokens(part)
    }


def validate_changed_path(path: str, policy: dict[str, Any]) -> str:
    normalized = _require_repository_path(path, "changed path")
    lowered = normalized.casefold()
    root_paths = {item.casefold() for item in policy["root_of_trust_paths"]}
    if lowered in root_paths:
        raise ValueError(f"changed path reaches a root-of-trust file: {normalized}")
    if any(lowered.startswith(prefix.casefold()) for prefix in policy["forbidden_changed_path_prefixes"]):
        raise ValueError(f"changed path reaches a forbidden path prefix: {normalized}")
    if not any(lowered.startswith(prefix.casefold()) for prefix in policy["allowed_changed_path_prefixes"]):
        raise ValueError(f"changed path is outside the infrastructure allowlist: {normalized}")
    if lowered.startswith("config/") and not normalized.endswith(".example.json"):
        raise ValueError(
            "changed config paths must use the exact case-sensitive .example.json suffix: "
            f"{normalized}"
        )
    if any(
        lowered.startswith(prefix.casefold())
        for prefix in policy["allowed_synthetic_path_prefixes"]
    ) and not normalized.endswith(".json"):
        raise ValueError(
            "changed synthetic material paths must use the exact .json suffix: "
            f"{normalized}"
        )
    risky_tokens = _path_tokens(normalized) & set(policy["forbidden_path_tokens"])
    if risky_tokens:
        raise ValueError(
            f"changed path contains forbidden capability token(s) {sorted(risky_tokens)}: {normalized}"
        )
    return normalized


def validate_config_reference_path(path: str, policy: dict[str, Any]) -> str:
    normalized = _require_repository_path(path, "config reference path")
    if normalized == "research/INFRASTRUCTURE_GATE.json":
        return normalized
    lowered = normalized.casefold()
    if not any(
        lowered.startswith(prefix.casefold())
        for prefix in policy["allowed_config_path_prefixes"]
    ):
        raise ValueError(
            "config reference is outside the infrastructure config allowlist: "
            f"{normalized}"
        )
    if not normalized.endswith(".example.json"):
        raise ValueError(
            "infrastructure config references must use the exact .example.json suffix: "
            f"{normalized}"
        )
    return normalized


def validate_synthetic_reference_path(
    path: str,
    policy: dict[str, Any],
    *,
    field: str,
) -> str:
    normalized = _require_repository_path(path, field)
    lowered = normalized.casefold()
    if not any(
        lowered.startswith(prefix.casefold())
        for prefix in policy["allowed_synthetic_path_prefixes"]
    ):
        raise ValueError(
            f"{field} is outside the synthetic material allowlist: {normalized}"
        )
    if not normalized.endswith(".json"):
        raise ValueError(f"{field} must use the exact .json suffix: {normalized}")
    risky_tokens = _path_tokens(normalized) & set(policy["forbidden_path_tokens"])
    if risky_tokens:
        raise ValueError(
            f"{field} contains forbidden capability token(s) "
            f"{sorted(risky_tokens)}: {normalized}"
        )
    return normalized


def infrastructure_lifecycle_paths(experiment_id: str) -> tuple[str, ...]:
    """Return the only lifecycle artifacts admitted in addition to implementation paths."""

    normalized_id = _require_nonempty_string(experiment_id, "experiment_id")
    if not SAFE_EXPERIMENT_ID.fullmatch(normalized_id):
        raise ValueError("experiment_id is not a safe identifier")
    return (
        "research/REGISTRY.jsonl",
        f"research/experiments/{normalized_id}.md",
        f"research/queue/{normalized_id}.json",
        f"research/scopes/{normalized_id}.proposal.json",
    )


def _validate_manifest_changed_path(
    path: str,
    *,
    policy: dict[str, Any],
    lifecycle_paths: set[str],
) -> str:
    normalized = _require_repository_path(path, "changed path")
    if normalized in lifecycle_paths:
        return normalized
    return validate_changed_path(normalized, policy)


def _normalize_test_matrix(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("test_matrix must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(value):
        payload = _require_object(item, f"test_matrix[{index}]")
        _require_exact_fields(payload, {"test_id", "kind", "assertion"}, f"test_matrix[{index}]")
        test_id = _require_nonempty_string(payload["test_id"], f"test_matrix[{index}].test_id")
        if not SAFE_TEST_ID.fullmatch(test_id):
            raise ValueError(f"test_matrix[{index}].test_id is not a safe identifier")
        kind = _require_nonempty_string(payload["kind"], f"test_matrix[{index}].kind")
        if kind not in REQUIRED_TEST_KINDS:
            raise ValueError(f"test_matrix[{index}].kind is unsupported: {kind}")
        normalized.append(
            {
                "test_id": test_id,
                "kind": kind,
                "assertion": _require_nonempty_string(
                    payload["assertion"], f"test_matrix[{index}].assertion"
                ),
            }
        )
    ids = [item["test_id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("test_matrix test_id values must be unique")
    method_names = [_test_method_name(test_id) for test_id in ids]
    if len(method_names) != len(set(method_names)):
        raise ValueError(
            "test_matrix test_id values must map to unique unittest method names"
        )
    observed_kinds = {item["kind"] for item in normalized}
    if observed_kinds != REQUIRED_TEST_KINDS:
        raise ValueError(
            "test_matrix must include positive, negative, and backward_compatibility cases"
        )
    return sorted(normalized, key=lambda item: item["test_id"])


def _test_method_name(test_id: str) -> str:
    """Return the sole unittest method name bound to a pre-registered test id."""

    normalized = re.sub(r"[-_]+", "_", test_id.casefold()).strip("_")
    return f"test_{normalized}"


def _normalize_primary_metric(value: Any) -> dict[str, Any]:
    payload = _require_object(value, "primary_metric")
    _require_exact_fields(payload, {"name", "direction", "required_value"}, "primary_metric")
    expected = {
        "name": "pre_registered_infrastructure_contract_pass_fraction",
        "direction": "higher_is_better",
        "required_value": 1.0,
    }
    if payload != expected:
        raise ValueError(f"primary_metric must equal {canonical_json_text(expected)}")
    return expected


def _normalize_required_effect(value: Any) -> dict[str, Any]:
    fields = {
        "all_pre_registered_tests_pass",
        "external_api_calls",
        "network_calls",
        "real_data_rows",
        "roi_calculations",
        "production_changes",
        "buy_order_notification_side_effects",
    }
    payload = _require_object(value, "required_effect")
    _require_exact_fields(payload, fields, "required_effect")
    if payload["all_pre_registered_tests_pass"] is not True:
        raise ValueError("required_effect.all_pre_registered_tests_pass must be true")
    for field in fields - {"all_pre_registered_tests_pass"}:
        _require_zero(payload[field], f"required_effect.{field}")
    return {
        "all_pre_registered_tests_pass": True,
        **{field: 0 for field in sorted(fields - {"all_pre_registered_tests_pass"})},
    }


def _normalize_compute_budget(value: Any) -> dict[str, Any]:
    fields = {
        "maximum_runtime_minutes",
        "network_calls",
        "external_api_calls",
        "real_data_rows",
        "random_seed",
    }
    payload = _require_object(value, "compute_budget")
    _require_exact_fields(payload, fields, "compute_budget")
    maximum_runtime = _require_positive_int(
        payload["maximum_runtime_minutes"],
        "compute_budget.maximum_runtime_minutes",
        maximum=60,
    )
    for field in ("network_calls", "external_api_calls", "real_data_rows"):
        _require_zero(payload[field], f"compute_budget.{field}")
    seed = payload["random_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("compute_budget.random_seed must be an integer")
    return {
        "maximum_runtime_minutes": maximum_runtime,
        "network_calls": 0,
        "external_api_calls": 0,
        "real_data_rows": 0,
        "random_seed": seed,
    }


def _normalize_compatibility_contract(value: Any) -> dict[str, bool]:
    fields = {
        "legacy_roi_proposal_digest_unchanged",
        "legacy_queue_schema_v2_readable",
        "legacy_event_schema_v2_readable",
    }
    payload = _require_object(value, "compatibility_contract")
    _require_exact_fields(payload, fields, "compatibility_contract")
    for field in fields:
        if payload[field] is not True:
            raise ValueError(f"compatibility_contract.{field} must be true")
    return {field: True for field in sorted(fields)}


def _require_disabled_scope_safety(payload: dict[str, Any], label: str) -> None:
    _require_false(payload["formal_buy"], f"{label}.formal_buy")
    _require_false(payload["send_order"], f"{label}.send_order")
    _require_zero(payload["stake"], f"{label}.stake")


def normalize_infrastructure_proposal(
    value: Any,
    *,
    policy: dict[str, Any],
    policy_sha256: str | None = None,
    expected_experiment_id: str | None = None,
) -> dict[str, Any]:
    policy = normalize_gate_policy(policy)
    payload = _require_object(value, "infrastructure proposal")
    _require_exact_fields(payload, PROPOSAL_FIELDS, "infrastructure proposal")
    _require_disabled_scope_safety(payload, "infrastructure proposal")
    _normalize_contract(payload["contract"], PROPOSAL_CONTRACT, "proposal.contract")
    if payload["gate_kind"] != GATE_KIND:
        raise ValueError(f"proposal.gate_kind must be {GATE_KIND}")

    experiment_id = _require_nonempty_string(payload["experiment_id"], "experiment_id")
    if not SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError("experiment_id is not a safe 3-64 character identifier")
    if expected_experiment_id is not None and experiment_id != expected_experiment_id:
        raise ValueError(
            f"proposal experiment_id mismatch: expected {expected_experiment_id!r}, found {experiment_id!r}"
        )

    expected_paths = _require_path_list(payload["expected_changed_paths"], "expected_changed_paths")
    expected_paths = sorted(validate_changed_path(path, policy) for path in expected_paths)
    input_classes = _require_string_list(payload["input_classes"], "input_classes")
    unsupported_inputs = sorted(set(input_classes) - ALLOWED_INPUT_CLASSES)
    if unsupported_inputs:
        raise ValueError("input_classes contains non-synthetic input class(es): " + ", ".join(unsupported_inputs))
    if "synthetic_fixture" not in input_classes:
        raise ValueError("input_classes must include synthetic_fixture")

    allowed_variant_count = payload["allowed_variant_count"]
    if allowed_variant_count != 1 or isinstance(allowed_variant_count, bool):
        raise ValueError("allowed_variant_count must be exactly 1")
    threshold_count = payload["allowed_threshold_search_count"]
    if threshold_count != 0 or isinstance(threshold_count, bool):
        raise ValueError("allowed_threshold_search_count must be 0")

    gate_policy = _normalize_manifest_ref(payload["gate_policy"], "gate_policy")
    if gate_policy["path"] != "research/INFRASTRUCTURE_GATE.json":
        raise ValueError("gate_policy.path must be research/INFRASTRUCTURE_GATE.json")
    if policy_sha256 is not None:
        expected_policy_hash = _require_sha256(policy_sha256, "policy_sha256")
        if gate_policy["sha256"] != expected_policy_hash:
            raise ValueError("proposal gate_policy hash differs from the loaded gate policy")

    normalized = {
        "contract": dict(PROPOSAL_CONTRACT),
        "gate_kind": GATE_KIND,
        "experiment_id": experiment_id,
        "title": _require_nonempty_string(payload["title"], "title"),
        "change_hypothesis": _require_nonempty_string(
            payload["change_hypothesis"], "change_hypothesis"
        ),
        "null_hypothesis": _require_nonempty_string(payload["null_hypothesis"], "null_hypothesis"),
        "safety_objective": _require_nonempty_string(payload["safety_objective"], "safety_objective"),
        "failure_modes": _require_string_list(payload["failure_modes"], "failure_modes"),
        "in_scope": _require_string_list(payload["in_scope"], "in_scope"),
        "out_of_scope": _require_string_list(payload["out_of_scope"], "out_of_scope"),
        "expected_changed_paths": expected_paths,
        "input_classes": input_classes,
        "source_as_of": _require_nonempty_string(payload["source_as_of"], "source_as_of"),
        "lineage_hash_requirements": _require_string_list(
            payload["lineage_hash_requirements"], "lineage_hash_requirements"
        ),
        "test_matrix": _normalize_test_matrix(payload["test_matrix"]),
        "primary_metric": _normalize_primary_metric(payload["primary_metric"]),
        "required_effect": _normalize_required_effect(payload["required_effect"]),
        "rejection_gate": _require_string_list(payload["rejection_gate"], "rejection_gate"),
        "stop_conditions": _require_string_list(payload["stop_conditions"], "stop_conditions"),
        "compute_budget": _normalize_compute_budget(payload["compute_budget"]),
        "allowed_variant_count": 1,
        "allowed_threshold_search_count": 0,
        "base_commit": _require_git_sha(payload["base_commit"], "base_commit"),
        "gate_policy": gate_policy,
        "compatibility_contract": _normalize_compatibility_contract(
            payload["compatibility_contract"]
        ),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    canonical_json_bytes(normalized)
    return normalized


def evaluate_infrastructure_gate(
    proposal: dict[str, Any],
    *,
    policy: dict[str, Any],
) -> dict[str, Any]:
    normalized_policy = normalize_gate_policy(policy)
    normalized_proposal = normalize_infrastructure_proposal(
        proposal,
        policy=normalized_policy,
    )
    expected_paths = normalized_proposal["expected_changed_paths"]
    checks = {
        "bounded_compute": (
            normalized_proposal["compute_budget"]["maximum_runtime_minutes"] <= 60
            and normalized_proposal["compute_budget"]["network_calls"] == 0
            and normalized_proposal["compute_budget"]["external_api_calls"] == 0
            and normalized_proposal["compute_budget"]["real_data_rows"] == 0
        ),
        "compatibility_locked": all(
            normalized_proposal["compatibility_contract"].values()
        ),
        "deterministic_lineage": bool(
            normalized_proposal["lineage_hash_requirements"]
        ),
        "expected_paths_firewalled": all(
            validate_changed_path(path, normalized_policy) == path
            for path in expected_paths
        ),
        "gate_identity_bound": (
            normalized_proposal["contract"] == PROPOSAL_CONTRACT
            and normalized_proposal["gate_kind"] == GATE_KIND
        ),
        "non_race_scope": not any(
            _path_tokens(path) & set(normalized_policy["forbidden_path_tokens"])
            for path in expected_paths
        ),
        "policy_hash_bound": (
            normalized_proposal["gate_policy"]["path"]
            == "research/INFRASTRUCTURE_GATE.json"
            and FULL_SHA256.fullmatch(
                normalized_proposal["gate_policy"]["sha256"]
            )
            is not None
        ),
        "single_variant_no_threshold_search": (
            normalized_proposal["allowed_variant_count"] == 1
            and normalized_proposal["allowed_threshold_search_count"] == 0
        ),
        "structured_test_matrix": (
            {item["kind"] for item in normalized_proposal["test_matrix"]}
            == REQUIRED_TEST_KINDS
        ),
        "synthetic_inputs_only": (
            set(normalized_proposal["input_classes"]) <= ALLOWED_INPUT_CLASSES
            and "synthetic_fixture" in normalized_proposal["input_classes"]
        ),
        "zero_side_effect_safety": (
            normalized_proposal["formal_buy"] is False
            and normalized_proposal["send_order"] is False
            and normalized_proposal["stake"] == 0
            and all(
                normalized_proposal["required_effect"][field] == 0
                for field in (
                    "external_api_calls",
                    "network_calls",
                    "real_data_rows",
                    "roi_calculations",
                    "production_changes",
                    "buy_order_notification_side_effects",
                )
            )
        ),
    }
    if set(checks) != set(normalized_policy["required_hard_checks"]):
        raise ValueError("gate evaluation differs from the policy hard-check set")
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError("infrastructure hard check failed: " + ", ".join(failed))
    return {
        "gate_kind": GATE_KIND,
        "gate_contract_version": 1,
        "mode": "all_hard_checks_must_pass",
        "checks": checks,
        "passed": True,
        "proposal_scope_digest": canonical_digest(normalized_proposal),
    }


def _normalize_gate_evaluation(
    value: Any,
    *,
    proposal_digest: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    payload = _require_object(value, "gate_evaluation")
    fields = {
        "gate_kind",
        "gate_contract_version",
        "mode",
        "checks",
        "passed",
        "proposal_scope_digest",
    }
    _require_exact_fields(payload, fields, "gate_evaluation")
    if payload["gate_kind"] != GATE_KIND or payload["gate_contract_version"] != 1:
        raise ValueError("gate_evaluation identity is invalid")
    if payload["mode"] != "all_hard_checks_must_pass":
        raise ValueError("gate_evaluation mode is invalid")
    checks = _require_object(payload["checks"], "gate_evaluation.checks")
    if set(checks) != set(policy["required_hard_checks"]):
        raise ValueError("gate_evaluation checks differ from policy")
    if any(value is not True for value in checks.values()):
        raise ValueError("every infrastructure hard check must be true")
    if payload["passed"] is not True:
        raise ValueError("gate_evaluation.passed must be true")
    if payload["proposal_scope_digest"] != proposal_digest:
        raise ValueError("gate_evaluation proposal digest mismatch")
    return {
        "gate_kind": GATE_KIND,
        "gate_contract_version": 1,
        "mode": "all_hard_checks_must_pass",
        "checks": {name: True for name in sorted(checks)},
        "passed": True,
        "proposal_scope_digest": proposal_digest,
    }


def _normalize_changed_path_entry(
    value: Any,
    *,
    policy: dict[str, Any],
    lifecycle_paths: set[str],
    index: int,
) -> dict[str, Any]:
    payload = _require_object(value, f"changed_path_manifest.entries[{index}]")
    fields = {"path", "change_type", "base_blob_sha", "execution_blob_sha"}
    _require_exact_fields(payload, fields, f"changed_path_manifest.entries[{index}]")
    path = _validate_manifest_changed_path(
        payload["path"],
        policy=policy,
        lifecycle_paths=lifecycle_paths,
    )
    change_type = _require_nonempty_string(
        payload["change_type"], f"changed_path_manifest.entries[{index}].change_type"
    )
    if change_type not in {"added", "modified", "deleted"}:
        raise ValueError(f"unsupported changed-path type: {change_type}")
    base_hash = _require_git_blob_sha(
        payload["base_blob_sha"],
        f"changed_path_manifest.entries[{index}].base_blob_sha",
        nullable=True,
    )
    execution_hash = _require_git_blob_sha(
        payload["execution_blob_sha"],
        f"changed_path_manifest.entries[{index}].execution_blob_sha",
        nullable=True,
    )
    if change_type == "added" and (base_hash is not None or execution_hash is None):
        raise ValueError("added changed-path entries require null base hash and an execution hash")
    if change_type == "modified" and (base_hash is None or execution_hash is None):
        raise ValueError("modified changed-path entries require base and execution hashes")
    if change_type == "deleted" and (base_hash is None or execution_hash is not None):
        raise ValueError("deleted changed-path entries require a base hash and null execution hash")
    return {
        "path": path,
        "change_type": change_type,
        "base_blob_sha": base_hash,
        "execution_blob_sha": execution_hash,
    }


def normalize_changed_path_manifest(
    value: Any,
    *,
    proposal_scope: dict[str, Any],
    execution_commit_sha: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    payload = _require_object(value, "changed_path_manifest")
    fields = {"schema_version", "base_commit", "execution_commit", "entries"}
    _require_exact_fields(payload, fields, "changed_path_manifest")
    if payload["schema_version"] != CHANGED_PATH_MANIFEST_VERSION:
        raise ValueError(f"changed_path_manifest.schema_version must be {CHANGED_PATH_MANIFEST_VERSION}")
    base_commit = _require_git_sha(payload["base_commit"], "changed_path_manifest.base_commit")
    if base_commit != proposal_scope["base_commit"]:
        raise ValueError("changed-path manifest base commit differs from proposal base commit")
    execution_commit = _require_git_sha(
        payload["execution_commit"], "changed_path_manifest.execution_commit"
    )
    if execution_commit != execution_commit_sha:
        raise ValueError("changed-path manifest execution commit differs from run scope")
    entries_value = payload["entries"]
    if not isinstance(entries_value, list) or not entries_value:
        raise ValueError("changed_path_manifest.entries must be a non-empty list")
    lifecycle_paths = set(infrastructure_lifecycle_paths(proposal_scope["experiment_id"]))
    entries = [
        _normalize_changed_path_entry(
            item,
            policy=policy,
            lifecycle_paths=lifecycle_paths,
            index=index,
        )
        for index, item in enumerate(entries_value)
    ]
    paths = [item["path"] for item in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("changed_path_manifest entries must not repeat a path")
    if len(paths) != len({path.casefold() for path in paths}):
        raise ValueError("changed_path_manifest entries must not case-fold collide")
    required_paths = set(proposal_scope["expected_changed_paths"]) | lifecycle_paths
    if set(paths) != required_paths:
        raise ValueError(
            "changed-path manifest paths differ from expected implementation and lifecycle paths"
        )
    return {
        "schema_version": CHANGED_PATH_MANIFEST_VERSION,
        "base_commit": base_commit,
        "execution_commit": execution_commit,
        "entries": sorted(entries, key=lambda item: item["path"]),
    }


def command_template_ids() -> tuple[str, ...]:
    return (
        "python_unittest_discover_infrastructure_v1",
        "python_unittest_module_v1",
    )


def bound_python_executable() -> str:
    executable = Path(sys.executable).resolve()
    if not executable.is_file():
        raise ValueError("cannot bind the current Python executable")
    return str(executable)


def _unittest_module_path(module: str) -> str:
    return module.replace(".", "/") + ".py"


def _normalize_command_step(
    value: Any,
    *,
    expected_changed_paths: set[str],
    readable_paths: set[str],
    policy: dict[str, Any],
    index: int,
) -> tuple[dict[str, Any], list[str]]:
    payload = _require_object(value, f"command_plan[{index}]")
    _require_exact_fields(payload, {"template_id", "parameters"}, f"command_plan[{index}]")
    template_id = _require_nonempty_string(payload["template_id"], f"command_plan[{index}].template_id")
    if template_id not in policy["allowed_command_template_ids"]:
        raise ValueError(f"command template is not allowlisted: {template_id}")
    parameters = _require_object(payload["parameters"], f"command_plan[{index}].parameters")

    if template_id == "python_unittest_module_v1":
        _require_exact_fields(parameters, {"module"}, f"command_plan[{index}].parameters")
        module = _require_nonempty_string(parameters["module"], f"command_plan[{index}].parameters.module")
        if not SAFE_UNITTEST_MODULE.fullmatch(module):
            raise ValueError(
                "unittest module must be a research.infra_tests.test_* module"
            )
        module_path = _unittest_module_path(module)
        if module_path not in expected_changed_paths:
            raise ValueError(
                "unittest module must name an actual retained changed "
                f"research/infra_tests/test_*.py path: {module_path}"
            )
        normalized_parameters: dict[str, Any] = {"module": module}
        pattern = module.rsplit(".", 1)[-1] + ".py"
        argv = [
            bound_python_executable(),
            "-B",
            "-I",
            "-S",
            "-m",
            "unittest",
            "discover",
            "-s",
            "research/infra_tests",
            "-p",
            pattern,
            "-v",
        ]
    elif template_id == "python_unittest_discover_infrastructure_v1":
        _require_exact_fields(parameters, set(), f"command_plan[{index}].parameters")
        if not any(
            SAFE_UNITTEST_PATH.fullmatch(path)
            for path in expected_changed_paths
        ):
            raise ValueError(
                "infrastructure unittest discovery requires at least one retained "
                "changed research/infra_tests/test_*.py path"
            )
        normalized_parameters = {}
        argv = [
            bound_python_executable(),
            "-B",
            "-I",
            "-S",
            "-m",
            "unittest",
            "discover",
            "-s",
            "research/infra_tests",
            "-p",
            "test_*.py",
        ]
    else:
        raise ValueError(f"unknown command template: {template_id}")

    return {"template_id": template_id, "parameters": normalized_parameters}, argv


def render_command_plan(
    value: Any,
    *,
    expected_changed_paths: set[str],
    readable_paths: set[str],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    if not isinstance(value, list) or not value:
        raise ValueError("command_plan must be a non-empty list")
    plan: list[dict[str, Any]] = []
    exact_argv: list[list[str]] = []
    for index, item in enumerate(value):
        step, argv = _normalize_command_step(
            item,
            expected_changed_paths=expected_changed_paths,
            readable_paths=readable_paths,
            policy=policy,
            index=index,
        )
        plan.append(step)
        exact_argv.append(argv)
    template_ids = {step["template_id"] for step in plan}
    if not template_ids & {
        "python_unittest_module_v1",
        "python_unittest_discover_infrastructure_v1",
    }:
        raise ValueError("command_plan must include an infrastructure unittest template")
    return plan, exact_argv


def _normalize_execution_context(
    value: Any,
    *,
    maximum_runtime_minutes: int,
) -> dict[str, Any]:
    payload = _require_object(value, "execution_context")
    fields = {
        "working_directory",
        "inherit_environment",
        "environment",
        "timeout_seconds",
        "network_access",
        "credential_environment_access",
        "filesystem_write_paths",
    }
    _require_exact_fields(payload, fields, "execution_context")
    if payload["working_directory"] != "repository_root":
        raise ValueError("execution_context.working_directory must be repository_root")
    _require_false(payload["inherit_environment"], "execution_context.inherit_environment")
    environment = _require_object(payload["environment"], "execution_context.environment")
    if environment:
        raise ValueError("execution_context.environment must be empty")
    expected_timeout = maximum_runtime_minutes * 60
    if payload["timeout_seconds"] != expected_timeout:
        raise ValueError("execution_context.timeout_seconds must match proposal compute budget")
    _require_false(payload["network_access"], "execution_context.network_access")
    _require_false(
        payload["credential_environment_access"],
        "execution_context.credential_environment_access",
    )
    write_paths = _require_path_list(
        payload["filesystem_write_paths"],
        "execution_context.filesystem_write_paths",
        allow_empty=True,
    )
    if write_paths:
        raise ValueError("execution_context.filesystem_write_paths must be empty")
    return {
        "working_directory": "repository_root",
        "inherit_environment": False,
        "environment": {},
        "timeout_seconds": expected_timeout,
        "network_access": False,
        "credential_environment_access": False,
        "filesystem_write_paths": [],
    }


def _normalize_exact_argv(value: Any) -> list[list[str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("exact_execution_argv must be a non-empty list")
    normalized: list[list[str]] = []
    for index, argv in enumerate(value):
        if not isinstance(argv, list) or not argv:
            raise ValueError(f"exact_execution_argv[{index}] must be a non-empty argv list")
        tokens = []
        for token_index, token in enumerate(argv):
            text = _require_nonempty_string(token, f"exact_execution_argv[{index}][{token_index}]")
            if "\r" in text or "\n" in text:
                raise ValueError("argv tokens must not contain line breaks")
            tokens.append(text)
        normalized.append(tokens)
    return normalized


def _require_disabled_run_safety(payload: dict[str, Any]) -> None:
    for field in (
        "external_api_calls",
        "actual_codex_dispatch",
        "real_data_execution",
        "roi_calculation",
        "production_change",
        "credential_access",
        "notification_side_effects",
        "order_side_effects",
        "formal_buy",
        "send_order",
    ):
        _require_false(payload[field], f"run scope.{field}")
    _require_zero(payload["network_calls"], "run scope.network_calls")
    _require_zero(payload["stake"], "run scope.stake")


def normalize_infrastructure_run_scope(
    value: Any,
    *,
    proposal_scope: dict[str, Any],
    policy: dict[str, Any],
    policy_sha256: str | None = None,
) -> dict[str, Any]:
    policy = normalize_gate_policy(policy)
    proposal_scope = normalize_infrastructure_proposal(
        proposal_scope,
        policy=policy,
        policy_sha256=policy_sha256,
    )
    payload = _require_object(value, "infrastructure run scope")
    _require_exact_fields(payload, RUN_FIELDS, "infrastructure run scope")
    _require_disabled_run_safety(payload)
    _normalize_contract(payload["contract"], RUN_CONTRACT, "run.contract")
    if payload["gate_kind"] != GATE_KIND:
        raise ValueError(f"run.gate_kind must be {GATE_KIND}")
    if payload["execution_kind"] != "synthetic":
        raise ValueError("infrastructure run execution_kind must be synthetic")

    embedded_proposal = normalize_infrastructure_proposal(
        payload["proposal_scope"],
        policy=policy,
        policy_sha256=policy_sha256,
        expected_experiment_id=proposal_scope["experiment_id"],
    )
    if embedded_proposal != proposal_scope:
        raise ValueError("run proposal_scope differs from the frozen proposal")
    proposal_digest = canonical_digest(proposal_scope)
    if payload["proposal_scope_digest"] != proposal_digest:
        raise ValueError("run proposal_scope_digest differs from the frozen proposal")
    execution_commit = _require_git_sha(payload["execution_commit_sha"], "execution_commit_sha")

    config_hashes = _normalize_manifest_refs(payload["config_hashes"], "config_hashes")
    config_hashes = [
        {
            **reference,
            "path": validate_config_reference_path(reference["path"], policy),
        }
        for reference in config_hashes
    ]
    if proposal_scope["gate_policy"] not in config_hashes:
        raise ValueError("config_hashes must include the exact gate_policy reference")
    synthetic_inputs = _normalize_manifest_refs(
        payload["synthetic_input_hashes"], "synthetic_input_hashes"
    )
    synthetic_inputs = [
        {
            **reference,
            "path": validate_synthetic_reference_path(
                reference["path"],
                policy,
                field="synthetic_input_hashes[].path",
            ),
        }
        for reference in synthetic_inputs
    ]
    environment = _normalize_manifest_ref(
        payload["dependency_environment_manifest"], "dependency_environment_manifest"
    )
    environment = {
        **environment,
        "path": validate_synthetic_reference_path(
            environment["path"],
            policy,
            field="dependency_environment_manifest.path",
        ),
    }
    changed_manifest = normalize_changed_path_manifest(
        payload["changed_path_manifest"],
        proposal_scope=proposal_scope,
        execution_commit_sha=execution_commit,
        policy=policy,
    )
    changed_digest = canonical_digest(changed_manifest)
    if payload["changed_path_manifest_digest"] != changed_digest:
        raise ValueError("changed_path_manifest_digest mismatch")

    readable_paths = {
        item["path"] for item in [*config_hashes, *synthetic_inputs, environment]
    } | set(proposal_scope["expected_changed_paths"])
    retained_changed_paths = {
        entry["path"]
        for entry in changed_manifest["entries"]
        if entry["execution_blob_sha"] is not None
    }
    plan, derived_argv = render_command_plan(
        payload["command_plan"],
        expected_changed_paths=retained_changed_paths,
        readable_paths=readable_paths,
        policy=policy,
    )
    exact_argv = _normalize_exact_argv(payload["exact_execution_argv"])
    if exact_argv != derived_argv:
        raise ValueError("exact_execution_argv does not match the structured command plan")
    execution_context = _normalize_execution_context(
        payload["execution_context"],
        maximum_runtime_minutes=proposal_scope["compute_budget"][
            "maximum_runtime_minutes"
        ],
    )
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    return {
        "contract": dict(RUN_CONTRACT),
        "gate_kind": GATE_KIND,
        "proposal_scope": proposal_scope,
        "proposal_scope_digest": proposal_digest,
        "execution_commit_sha": execution_commit,
        "config_hashes": config_hashes,
        "synthetic_input_hashes": synthetic_inputs,
        "changed_path_manifest": changed_manifest,
        "changed_path_manifest_digest": changed_digest,
        "dependency_environment_manifest": environment,
        "seed": seed,
        "command_plan": plan,
        "exact_execution_argv": exact_argv,
        "execution_context": execution_context,
        "execution_kind": "synthetic",
        "external_api_calls": False,
        "network_calls": 0,
        "actual_codex_dispatch": False,
        "real_data_execution": False,
        "roi_calculation": False,
        "production_change": False,
        "credential_access": False,
        "notification_side_effects": False,
        "order_side_effects": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def resolve_repository_path(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    normalized = _require_repository_path(relative_path, "repository path")
    path = root.joinpath(*PurePosixPath(normalized).parts)
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"path escapes repository root: {relative_path}") from exc
    return path


def _reject_linked_path(root: Path, path: Path, field: str) -> None:
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes repository root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            is_junction = getattr(current, "is_junction", lambda: False)()
            if current.is_symlink() or is_junction:
                raise ValueError(f"{field} must not traverse a symlink or junction: {path}")
        except OSError as exc:
            raise ValueError(f"cannot inspect {field} path: {path}") from exc
    try:
        if path.resolve(strict=False) != path:
            raise ValueError(f"{field} must not traverse a symlink or junction: {path}")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve {field} path: {path}") from exc


def reject_linked_repository_path(root: Path, path: Path, field: str) -> None:
    """Public fail-closed wrapper for code-owned CLI input paths."""

    _reject_linked_path(root, path, field)


def verify_manifest_ref(root: Path, reference: dict[str, str], field: str) -> None:
    path = resolve_repository_path(root, reference["path"])
    _reject_linked_path(root, path, field)
    if not path.is_file():
        raise ValueError(f"{field} file not found: {reference['path']}")
    observed = sha256_file(path)
    if observed != reference["sha256"]:
        raise ValueError(
            f"{field} hash changed for {reference['path']}: expected {reference['sha256']}, observed {observed}"
        )


def _scan_sensitive_json_values(
    value: Any,
    *,
    path: str = "payload",
    material_label: str = "synthetic fixture",
) -> None:
    sensitive_key_tokens = {
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private",
        "secret",
        "secrets",
        "token",
    }
    sensitive_key_names = {
        "api_key",
        "apikey",
        "client_secret",
        "private_key",
    }
    real_data_keys = {
        "finish_position",
        "horse_id",
        "odds",
        "payout",
        "race_date",
        "race_id",
        "result",
        "runner_id",
    }
    secret_markers = (
        "-----begin private key",
        "-----begin rsa private key",
        "ghp_",
        "github_pat_",
        "sk-proj-",
    )
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).casefold()
            key_tokens = {
                token
                for token in re.split(r"[^a-z0-9]+", str(key).casefold())
                if token
            }
            if (
                key_tokens & sensitive_key_tokens
                or normalized_key in sensitive_key_names
            ) and item not in (None, False, 0, ""):
                raise ValueError(
                    f"{material_label} contains a populated sensitive key: {path}.{key}"
                )
            if normalized_key in real_data_keys and item not in (None, False, 0, "", [], {}):
                raise ValueError(
                    f"{material_label} contains row-level real-data shape: {path}.{key}"
                )
            _scan_sensitive_json_values(
                item,
                path=f"{path}.{key}",
                material_label=material_label,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_sensitive_json_values(
                item,
                path=f"{path}[{index}]",
                material_label=material_label,
            )
    elif isinstance(value, str):
        lowered = value.casefold()
        if any(marker in lowered for marker in secret_markers):
            raise ValueError(f"{material_label} contains secret-like material at {path}")


def _load_synthetic_fixture(
    root: Path,
    reference: dict[str, str],
    *,
    field: str,
    fixture_kind: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    verify_manifest_ref(root, reference, field)
    path = resolve_repository_path(root, reference["path"])
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot inspect {field}: {reference['path']}") from exc
    if size > policy["maximum_synthetic_fixture_bytes"]:
        raise ValueError(
            f"{field} exceeds the synthetic fixture size limit: {reference['path']}"
        )
    payload = strict_json_load(path)
    envelope = _require_object(payload, field)
    fields = {
        "schema_version",
        "fixture_kind",
        "synthetic",
        "provenance",
        "contains_real_data",
        "contains_credentials",
        "payload",
    }
    _require_exact_fields(envelope, fields, field)
    expected = {
        "schema_version": 1,
        "fixture_kind": fixture_kind,
        "synthetic": True,
        "provenance": "code_owned_synthetic_fixture_v1",
        "contains_real_data": False,
        "contains_credentials": False,
    }
    for key, expected_value in expected.items():
        if envelope[key] != expected_value:
            raise ValueError(f"{field}.{key} must be {expected_value!r}")
    _scan_sensitive_json_values(envelope["payload"])
    return envelope


def expected_environment_payload() -> dict[str, str]:
    executable = Path(bound_python_executable())
    return {
        "python_executable": str(executable),
        "python_executable_sha256": sha256_file(executable),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }


def verify_infrastructure_run_materials(root: Path, run_scope: dict[str, Any]) -> None:
    policy = normalize_gate_policy(
        strict_json_load(root / "research" / "INFRASTRUCTURE_GATE.json")
    )
    for index, reference in enumerate(run_scope["config_hashes"]):
        field = f"config_hashes[{index}]"
        verify_manifest_ref(root, reference, field)
        path = resolve_repository_path(root, reference["path"])
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"cannot inspect {field}: {reference['path']}") from exc
        if size > policy["maximum_synthetic_fixture_bytes"]:
            raise ValueError(f"{field} exceeds the JSON material size limit")
        _scan_sensitive_json_values(
            strict_json_load(path),
            path="config",
            material_label="infrastructure config",
        )
    for index, reference in enumerate(run_scope["synthetic_input_hashes"]):
        _load_synthetic_fixture(
            root,
            reference,
            field=f"synthetic_input_hashes[{index}]",
            fixture_kind="synthetic_input",
            policy=policy,
        )
    environment = _load_synthetic_fixture(
        root,
        run_scope["dependency_environment_manifest"],
        field="dependency_environment_manifest",
        fixture_kind="dependency_environment",
        policy=policy,
    )
    if environment["payload"] != expected_environment_payload():
        raise ValueError(
            "dependency_environment_manifest payload differs from the current isolated "
            "Python interpreter"
        )
    for entry in run_scope["changed_path_manifest"]["entries"]:
        path = resolve_repository_path(root, entry["path"])
        _reject_linked_path(root, path, f"changed path {entry['path']}")
        if entry["change_type"] == "deleted":
            if path.exists():
                raise ValueError(f"deleted changed path still exists: {entry['path']}")
            continue
        if not path.is_file():
            raise ValueError(f"changed path file not found: {entry['path']}")
        # Exact Git blob identity is revalidated by verify_infrastructure_commit_diff.
        # This local file check only establishes that the execution-side path exists
        # and is not a symlink before a transition verifier consults committed trees.


GitRunner = Callable[[Path, list[str]], bytes]


def _default_git_runner(root: Path, arguments: list[str]) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"local Git verification failed: {stderr or arguments[0]}")
    return completed.stdout


def _decode_git_path(value: bytes, field: str) -> str:
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} is not valid UTF-8") from exc


def _read_tree_entry(
    *,
    root: Path,
    commit: str,
    path: str,
    runner: GitRunner,
) -> dict[str, str] | None:
    output = runner(root, ["ls-tree", "-z", commit, "--", path])
    if not output:
        return None
    records = [item for item in output.split(b"\0") if item]
    if len(records) != 1:
        raise ValueError(f"Git tree lookup is ambiguous for path: {path}")
    record = records[0]
    try:
        metadata, raw_path = record.split(b"\t", 1)
        mode_bytes, type_bytes, object_bytes = metadata.split(b" ", 2)
    except ValueError as exc:
        raise ValueError(f"malformed Git tree entry for path: {path}") from exc
    observed_path = _decode_git_path(raw_path, "Git tree path")
    if observed_path != path:
        raise ValueError(f"Git tree returned an unexpected path: {observed_path} != {path}")
    mode = mode_bytes.decode("ascii", errors="strict")
    object_type = type_bytes.decode("ascii", errors="strict")
    object_sha = object_bytes.decode("ascii", errors="strict").lower()
    if mode == "120000":
        raise ValueError(f"symbolic links are forbidden in infrastructure diffs: {path}")
    if mode == "160000" or object_type == "commit":
        raise ValueError(f"submodules are forbidden in infrastructure diffs: {path}")
    if object_type != "blob" or mode not in {"100644", "100755"}:
        raise ValueError(
            f"unsupported Git tree entry mode/type for {path}: {mode} {object_type}"
        )
    blob_sha = _require_git_blob_sha(object_sha, f"Git blob SHA for {path}")
    assert isinstance(blob_sha, str)
    return {"mode": mode, "blob_sha": blob_sha}


def _read_git_blob(
    *,
    root: Path,
    blob_sha: str,
    runner: GitRunner,
) -> bytes:
    return runner(root, ["cat-file", "blob", blob_sha])


def _validate_changed_python_source(
    *,
    path: str,
    content: bytes,
    policy: dict[str, Any],
    changed_modules: set[str],
) -> None:
    try:
        source = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"changed Python source is not UTF-8: {path}") from exc
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise ValueError(f"changed Python source is not parseable: {path}: {exc}") from exc

    forbidden_imports = {item.casefold() for item in policy["forbidden_import_roots"]}
    allowed_imports = {item.casefold() for item in policy["allowed_import_modules"]}
    forbidden_calls = {
        item.casefold() for item in policy["forbidden_dynamic_call_names"]
    }
    forbidden_tokens = {item.casefold() for item in policy["forbidden_path_tokens"]}
    # Capability/path tokens intentionally match camelCase and snake_case symbol
    # components. Import roots and dynamic-call names are matched exactly below;
    # splitting ``load_tests`` into ``tests`` would reject ordinary TestCase names.
    forbidden_symbol_tokens = forbidden_tokens
    for node in ast.walk(tree):
        modules: list[tuple[str, list[str] | None]] = []
        if isinstance(node, ast.Import):
            modules = [(alias.name, None) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError(f"relative imports are forbidden in changed Python source: {path}")
            modules = [(node.module or "", [alias.name for alias in node.names])]
        for module, imported_names in modules:
            root_name = module.split(".", 1)[0].casefold()
            module_name = module.casefold()
            module_tokens = _lexical_tokens(module)
            if root_name in forbidden_imports or module_tokens & forbidden_tokens:
                raise ValueError(
                    f"changed Python source imports a forbidden capability module "
                    f"{module!r}: {path}"
                )
            changed_import = module_name in changed_modules
            if imported_names is not None and not changed_import:
                changed_import = bool(imported_names) and all(
                    name != "*"
                    and f"{module_name}.{name.casefold()}" in changed_modules
                    for name in imported_names
                )
            if module_name not in allowed_imports and not changed_import:
                raise ValueError(
                    f"changed Python source imports a module outside the pure allowlist "
                    f"{module!r}: {path}"
                )
            if module_name == "unittest" and imported_names and "mock" in imported_names:
                raise ValueError(f"unittest.mock is forbidden in changed Python source: {path}")
            if imported_names and "*" in imported_names:
                raise ValueError(f"wildcard imports are forbidden in changed Python source: {path}")
        if isinstance(node, ast.Call):
            call_name: str | None = None
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            else:
                raise ValueError(
                    "changed Python source uses a non-static dynamic call target "
                    f"{type(node.func).__name__}: {path}"
                )
            if call_name.casefold() in forbidden_calls:
                raise ValueError(
                    f"changed Python source uses forbidden dynamic call "
                    f"{call_name!r}: {path}"
                )
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            unsafe_dunders = {
                match.casefold()
                for match in DUNDER_STRING.findall(node.value)
                if match.casefold() not in SAFE_DUNDER_STRINGS
            }
            if unsafe_dunders:
                raise ValueError(
                    "changed Python source contains forbidden dunder string(s) "
                    f"{sorted(unsafe_dunders)}: {path}"
                )
        symbol: str | None = None
        if isinstance(node, ast.Name):
            symbol = node.id
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol = node.name
        elif isinstance(node, ast.Attribute):
            symbol = node.attr
        elif isinstance(node, ast.alias):
            symbol = node.asname or node.name.rsplit(".", 1)[-1]
        if symbol is not None:
            folded_symbol = symbol.casefold()
            is_test_method_symbol = (
                isinstance(node, ast.FunctionDef)
                and folded_symbol.startswith("test_")
            )
            if (
                DUNDER_IDENTIFIER.fullmatch(folded_symbol)
                and folded_symbol not in SAFE_DUNDER_IDENTIFIERS
            ):
                raise ValueError(
                    "changed Python source contains a forbidden dunder symbol "
                    f"{symbol!r}: {path}"
                )
            if (
                not is_test_method_symbol
                and folded_symbol in forbidden_tokens | forbidden_imports | forbidden_calls
            ):
                raise ValueError(
                    "changed Python source contains a forbidden capability symbol "
                    f"{symbol!r}: {path}"
                )
            risky_symbols = (
                set()
                if is_test_method_symbol
                else _lexical_tokens(symbol) & forbidden_symbol_tokens
            )
            if risky_symbols:
                raise ValueError(
                    "changed Python source contains forbidden capability symbol(s) "
                    f"{sorted(risky_symbols)}: {path}"
                )


def _git_canonical_registry_bytes(content: bytes, *, field: str) -> bytes:
    normalized = content.replace(b"\r\n", b"\n")
    if b"\r" in normalized:
        raise ValueError(f"{field} contains a bare carriage return")
    return normalized


def _verify_registry_append_only(
    *,
    root: Path,
    manifest: dict[str, Any],
    runner: GitRunner,
) -> None:
    registry_path = "research/REGISTRY.jsonl"
    entry = next(
        (item for item in manifest["entries"] if item["path"] == registry_path),
        None,
    )
    if entry is None or entry["execution_blob_sha"] is None:
        raise ValueError("infrastructure diff must retain research/REGISTRY.jsonl")
    base_content = (
        b""
        if entry["base_blob_sha"] is None
        else _read_git_blob(
            root=root,
            blob_sha=entry["base_blob_sha"],
            runner=runner,
        )
    )
    execution_content = _read_git_blob(
        root=root,
        blob_sha=entry["execution_blob_sha"],
        runner=runner,
    )
    base_content = _git_canonical_registry_bytes(
        base_content,
        field="base registry blob",
    )
    execution_content = _git_canonical_registry_bytes(
        execution_content,
        field="execution registry blob",
    )
    if not execution_content.startswith(base_content) or execution_content == base_content:
        raise ValueError(
            "execution commit must preserve the base registry as an exact append-only prefix"
        )
    worktree_path = resolve_repository_path(root, registry_path)
    _reject_linked_path(root, worktree_path, "worktree registry")
    try:
        worktree_content = worktree_path.read_bytes()
    except OSError as exc:
        raise ValueError("cannot read the worktree registry") from exc
    worktree_content = _git_canonical_registry_bytes(
        worktree_content,
        field="worktree registry",
    )
    if not worktree_content.startswith(execution_content):
        raise ValueError(
            "worktree registry rewrites or deletes execution-commit history; fail-close"
        )


def _verify_changed_python_sources(
    *,
    root: Path,
    manifest: dict[str, Any],
    policy: dict[str, Any],
    runner: GitRunner,
) -> None:
    changed_modules = {
        entry["path"][:-3].replace("/", ".").casefold()
        for entry in manifest["entries"]
        if entry["execution_blob_sha"] is not None
        and entry["path"].endswith(".py")
        and not entry["path"].endswith("/__init__.py")
    }
    for entry in manifest["entries"]:
        path = entry["path"]
        blob_sha = entry["execution_blob_sha"]
        if blob_sha is None or not path.endswith(".py"):
            continue
        _validate_changed_python_source(
            path=path,
            content=_read_git_blob(root=root, blob_sha=blob_sha, runner=runner),
            policy=policy,
            changed_modules=changed_modules,
        )


def _is_unittest_testcase_base(
    value: ast.expr,
    *,
    unittest_aliases: set[str],
    testcase_aliases: set[str],
    testcase_class_names: set[str],
) -> bool:
    if isinstance(value, ast.Name):
        return value.id in testcase_aliases or value.id in testcase_class_names
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "TestCase"
        and isinstance(value.value, ast.Name)
        and value.value.id in unittest_aliases
    )


def _discoverable_unittest_methods(*, path: str, content: bytes) -> set[str]:
    try:
        source = content.decode("utf-8", errors="strict")
        tree = ast.parse(source, filename=path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError(f"cannot inspect committed unittest evidence: {path}") from exc

    unittest_aliases: set[str] = set()
    testcase_aliases: set[str] = set()
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "unittest":
                    unittest_aliases.add(alias.asname or "unittest")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "unittest":
            for alias in node.names:
                if alias.name == "TestCase":
                    testcase_aliases.add(alias.asname or "TestCase")

    testcase_class_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for class_node in classes:
            if class_node.name in testcase_class_names:
                continue
            if any(
                _is_unittest_testcase_base(
                    base,
                    unittest_aliases=unittest_aliases,
                    testcase_aliases=testcase_aliases,
                    testcase_class_names=testcase_class_names,
                )
                for base in class_node.bases
            ):
                testcase_class_names.add(class_node.name)
                changed = True

    methods: set[str] = set()
    for class_node in classes:
        if class_node.name not in testcase_class_names:
            continue
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                if item.name in methods:
                    raise ValueError(
                        "committed infrastructure unittest method name is ambiguous: "
                        f"{item.name} in {path}"
                    )
                methods.add(item.name)
    return methods


def _verify_committed_test_evidence(
    *,
    root: Path,
    run_scope: dict[str, Any],
    manifest: dict[str, Any],
    runner: GitRunner,
) -> None:
    test_entries = {
        entry["path"]: entry
        for entry in manifest["entries"]
        if entry["execution_blob_sha"] is not None
        and SAFE_UNITTEST_PATH.fullmatch(entry["path"])
    }
    if not test_entries:
        raise ValueError(
            "infrastructure run must retain a committed changed "
            "research/infra_tests/test_*.py module"
        )

    scheduled_paths: set[str] = set()
    for step in run_scope["command_plan"]:
        if step["template_id"] == "python_unittest_discover_infrastructure_v1":
            scheduled_paths.update(test_entries)
        elif step["template_id"] == "python_unittest_module_v1":
            scheduled_paths.add(_unittest_module_path(step["parameters"]["module"]))
    if not scheduled_paths or not scheduled_paths <= set(test_entries):
        raise ValueError(
            "command_plan is not bound to retained committed changed infrastructure tests"
        )

    method_locations: dict[str, list[str]] = {}
    methods_by_path: dict[str, set[str]] = {}
    for path, entry in test_entries.items():
        blob_sha = entry["execution_blob_sha"]
        assert isinstance(blob_sha, str)
        methods = _discoverable_unittest_methods(
            path=path,
            content=_read_git_blob(root=root, blob_sha=blob_sha, runner=runner),
        )
        methods_by_path[path] = methods
        for method in methods:
            method_locations.setdefault(method, []).append(path)

    for path in scheduled_paths:
        if not methods_by_path[path]:
            raise ValueError(
                "scheduled committed infrastructure unittest module contains zero "
                f"discoverable TestCase methods: {path}"
            )

    for matrix_entry in run_scope["proposal_scope"]["test_matrix"]:
        method_name = _test_method_name(matrix_entry["test_id"])
        locations = method_locations.get(method_name, [])
        if len(locations) != 1:
            raise ValueError(
                "pre-registered test_matrix entry must map to exactly one committed "
                f"changed unittest.TestCase method {method_name!r}"
            )
        if locations[0] not in scheduled_paths:
            raise ValueError(
                "pre-registered unittest method is not covered by command_plan: "
                f"{method_name} in {locations[0]}"
            )


def _verify_committed_run_materials(
    *,
    root: Path,
    run_scope: dict[str, Any],
    runner: GitRunner,
) -> None:
    references = [
        *run_scope["config_hashes"],
        *run_scope["synthetic_input_hashes"],
        run_scope["dependency_environment_manifest"],
    ]
    execution_commit = run_scope["execution_commit_sha"]
    for reference in references:
        path = reference["path"]
        entry = _read_tree_entry(
            root=root,
            commit=execution_commit,
            path=path,
            runner=runner,
        )
        if entry is None:
            raise ValueError(f"run material is not committed at execution commit: {path}")
        content = _read_git_blob(
            root=root,
            blob_sha=entry["blob_sha"],
            runner=runner,
        )
        observed = hashlib.sha256(content).hexdigest()
        if observed != reference["sha256"]:
            raise ValueError(
                f"execution-commit run material hash differs from scope for {path}"
            )


def _read_actual_changed_path_manifest(
    *,
    root: Path,
    base_commit: str,
    execution_commit: str,
    lifecycle_paths: set[str],
    policy: dict[str, Any],
    runner: GitRunner,
) -> dict[str, Any]:
    output = runner(
        root,
        ["diff", "--name-status", "--no-renames", "-z", base_commit, execution_commit, "--"],
    )
    tokens = output.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    if len(tokens) % 2 != 0:
        raise ValueError("malformed NUL-delimited Git name-status output")
    entries: list[dict[str, Any]] = []
    for index in range(0, len(tokens), 2):
        try:
            status = tokens[index].decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("Git diff status is not ASCII") from exc
        path = _validate_manifest_changed_path(
            _decode_git_path(tokens[index + 1], "Git diff path"),
            policy=policy,
            lifecycle_paths=lifecycle_paths,
        )
        if status not in {"A", "M", "D"}:
            raise ValueError(f"unsupported Git diff status {status!r} for {path}")
        base_entry = _read_tree_entry(
            root=root,
            commit=base_commit,
            path=path,
            runner=runner,
        )
        execution_entry = _read_tree_entry(
            root=root,
            commit=execution_commit,
            path=path,
            runner=runner,
        )
        if status == "A" and (base_entry is not None or execution_entry is None):
            raise ValueError(f"Git added-path tree evidence is inconsistent: {path}")
        if status == "M" and (base_entry is None or execution_entry is None):
            raise ValueError(f"Git modified-path tree evidence is inconsistent: {path}")
        if status == "D" and (base_entry is None or execution_entry is not None):
            raise ValueError(f"Git deleted-path tree evidence is inconsistent: {path}")
        entries.append(
            {
                "path": path,
                "change_type": {"A": "added", "M": "modified", "D": "deleted"}[status],
                "base_blob_sha": None if base_entry is None else base_entry["blob_sha"],
                "execution_blob_sha": (
                    None if execution_entry is None else execution_entry["blob_sha"]
                ),
            }
        )
    paths = [item["path"] for item in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("Git diff repeats a changed path")
    if len(paths) != len({path.casefold() for path in paths}):
        raise ValueError("Git diff contains case-fold-colliding paths")
    return {
        "schema_version": CHANGED_PATH_MANIFEST_VERSION,
        "base_commit": base_commit,
        "execution_commit": execution_commit,
        "entries": sorted(entries, key=lambda item: item["path"]),
    }


def verify_infrastructure_commit_diff(
    root: Path,
    run_scope: dict[str, Any],
    *,
    runner: GitRunner | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-read committed trees and require exact equality with the frozen diff manifest.

    ``runner`` is injectable for synthetic tests.  Its signature is
    ``runner(repository_root, git_arguments_without_git_prefix) -> stdout_bytes``.
    The default invokes only read-only local ``git diff`` and ``git ls-tree``.
    """

    root = root.resolve()
    if policy is None:
        policy_path = root / "research" / "INFRASTRUCTURE_GATE.json"
        policy, policy_sha256 = load_gate_policy(policy_path)
    else:
        policy = normalize_gate_policy(policy)
        policy_path = root / "research" / "INFRASTRUCTURE_GATE.json"
        policy_sha256 = sha256_file(policy_path) if policy_path.is_file() else None
    normalized_run = normalize_infrastructure_run_scope(
        run_scope,
        proposal_scope=run_scope["proposal_scope"],
        policy=policy,
        policy_sha256=policy_sha256,
    )
    active_runner = runner or _default_git_runner
    try:
        active_runner(
            root,
            [
                "merge-base",
                "--is-ancestor",
                normalized_run["proposal_scope"]["base_commit"],
                normalized_run["execution_commit_sha"],
            ],
        )
    except ValueError as exc:
        raise ValueError(
            "execution commit is not a descendant of the proposal base commit"
        ) from exc
    actual = _read_actual_changed_path_manifest(
        root=root,
        base_commit=normalized_run["proposal_scope"]["base_commit"],
        execution_commit=normalized_run["execution_commit_sha"],
        lifecycle_paths=set(
            infrastructure_lifecycle_paths(normalized_run["proposal_scope"]["experiment_id"])
        ),
        policy=policy,
        runner=active_runner,
    )
    frozen = normalized_run["changed_path_manifest"]
    if actual != frozen:
        raise ValueError(
            "committed base-to-execution Git diff differs from changed_path_manifest"
        )
    if canonical_digest(actual) != normalized_run["changed_path_manifest_digest"]:
        raise ValueError("committed Git diff digest differs from frozen run scope")
    _verify_committed_run_materials(
        root=root,
        run_scope=normalized_run,
        runner=active_runner,
    )
    _verify_registry_append_only(
        root=root,
        manifest=actual,
        runner=active_runner,
    )
    _verify_changed_python_sources(
        root=root,
        manifest=actual,
        policy=policy,
        runner=active_runner,
    )
    _verify_committed_test_evidence(
        root=root,
        run_scope=normalized_run,
        manifest=actual,
        runner=active_runner,
    )
    return actual


def build_infrastructure_queue(
    *,
    proposal_scope: dict[str, Any],
    proposal_scope_file: str,
    experiment_markdown: str,
    owner: str,
    created_at: str,
    notes: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    proposal_digest = canonical_digest(proposal_scope)
    gate_evaluation = evaluate_infrastructure_gate(proposal_scope, policy=policy)
    return {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "gate_kind": GATE_KIND,
        "contract_version": 1,
        "experiment_id": proposal_scope["experiment_id"],
        "title": proposal_scope["title"],
        "owner": _require_nonempty_string(owner, "owner"),
        "created_at": _require_nonempty_string(created_at, "created_at"),
        "status": "proposed",
        "gate_evaluation": gate_evaluation,
        "proposal_scope": proposal_scope,
        "proposal_scope_file": _require_repository_path(
            proposal_scope_file, "proposal_scope_file"
        ),
        "proposal_scope_digest": proposal_digest,
        "human_approved_to_prepare": False,
        "human_approved_to_run": False,
        "automatic_execution_allowed": False,
        "execution_authorized": False,
        "production_approved": False,
        "merge_approved": False,
        "buy_approved": False,
        "production_change_allowed": False,
        "merge_allowed": False,
        "buy_logic_change_allowed": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "experiment_markdown": _require_repository_path(
            experiment_markdown, "experiment_markdown"
        ),
        "notes": notes.strip(),
    }


def normalize_infrastructure_queue(
    value: Any,
    *,
    policy: dict[str, Any],
    policy_sha256: str | None = None,
) -> dict[str, Any]:
    policy = normalize_gate_policy(policy)
    payload = _require_object(value, "infrastructure queue")
    _require_exact_fields(payload, QUEUE_FIELDS, "infrastructure queue")
    if payload["schema_version"] != QUEUE_SCHEMA_VERSION:
        raise ValueError(f"queue.schema_version must be {QUEUE_SCHEMA_VERSION}")
    if payload["gate_kind"] != GATE_KIND or payload["contract_version"] != 1:
        raise ValueError("queue gate identity is invalid")
    if payload["status"] != "proposed":
        raise ValueError("new infrastructure queue status must be proposed")
    proposal = normalize_infrastructure_proposal(
        payload["proposal_scope"],
        policy=policy,
        policy_sha256=policy_sha256,
        expected_experiment_id=payload["experiment_id"],
    )
    digest = canonical_digest(proposal)
    if payload["proposal_scope_digest"] != digest:
        raise ValueError("queue proposal_scope_digest mismatch")
    gate_evaluation = _normalize_gate_evaluation(
        payload["gate_evaluation"], proposal_digest=digest, policy=policy
    )
    expected_proposal_file = (
        f"research/scopes/{proposal['experiment_id']}.proposal.json"
    )
    expected_markdown = f"research/experiments/{proposal['experiment_id']}.md"
    proposal_scope_file = _require_repository_path(
        payload["proposal_scope_file"], "proposal_scope_file"
    )
    experiment_markdown = _require_repository_path(
        payload["experiment_markdown"], "experiment_markdown"
    )
    if proposal_scope_file != expected_proposal_file:
        raise ValueError(
            "queue proposal_scope_file must be the code-owned lifecycle path: "
            f"{expected_proposal_file}"
        )
    if experiment_markdown != expected_markdown:
        raise ValueError(
            "queue experiment_markdown must be the code-owned lifecycle path: "
            f"{expected_markdown}"
        )
    for field in (
        "human_approved_to_prepare",
        "human_approved_to_run",
        "automatic_execution_allowed",
        "execution_authorized",
        "production_approved",
        "merge_approved",
        "buy_approved",
        "production_change_allowed",
        "merge_allowed",
        "buy_logic_change_allowed",
        "formal_buy",
        "send_order",
    ):
        _require_false(payload[field], f"queue.{field}")
    _require_zero(payload["stake"], "queue.stake")
    normalized = dict(payload)
    normalized.update(
        {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "gate_kind": GATE_KIND,
            "contract_version": 1,
            "proposal_scope": proposal,
            "proposal_scope_digest": digest,
            "gate_evaluation": gate_evaluation,
            "proposal_scope_file": proposal_scope_file,
            "experiment_markdown": experiment_markdown,
            "owner": _require_nonempty_string(payload["owner"], "owner"),
            "created_at": _require_nonempty_string(payload["created_at"], "created_at"),
            "notes": str(payload["notes"]).strip(),
        }
    )
    return normalized


def normalize_gate_policy_evidence(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    payload = _require_object(value, "gate_policy_evidence")
    fields = {"path", "ref", "blob_sha", "content_sha256"}
    _require_exact_fields(payload, fields, "gate_policy_evidence")
    path = _require_repository_path(payload["path"], "gate_policy_evidence.path")
    if path != "research/INFRASTRUCTURE_GATE.json":
        raise ValueError(
            "gate_policy_evidence.path must be research/INFRASTRUCTURE_GATE.json"
        )
    ref = _require_git_sha(payload["ref"], "gate_policy_evidence.ref")
    blob_sha = _require_git_blob_sha(
        payload["blob_sha"], "gate_policy_evidence.blob_sha"
    )
    content_sha256 = _require_sha256(
        payload["content_sha256"], "gate_policy_evidence.content_sha256"
    )
    assert isinstance(blob_sha, str)
    assert isinstance(content_sha256, str)
    return {
        "path": path,
        "ref": ref,
        "blob_sha": blob_sha,
        "content_sha256": content_sha256,
    }


def normalize_main_registry_evidence(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    payload = _require_object(value, "main_registry_evidence")
    fields = {"path", "ref", "blob_sha", "content_sha256"}
    _require_exact_fields(payload, fields, "main_registry_evidence")
    path = _require_repository_path(payload["path"], "main_registry_evidence.path")
    if path != "research/REGISTRY.jsonl":
        raise ValueError(
            "main_registry_evidence.path must be research/REGISTRY.jsonl"
        )
    ref = _require_git_sha(payload["ref"], "main_registry_evidence.ref")
    blob_sha = _require_git_blob_sha(
        payload["blob_sha"], "main_registry_evidence.blob_sha"
    )
    content_sha256 = _require_sha256(
        payload["content_sha256"], "main_registry_evidence.content_sha256"
    )
    assert isinstance(blob_sha, str)
    assert isinstance(content_sha256, str)
    return {
        "path": path,
        "ref": ref,
        "blob_sha": blob_sha,
        "content_sha256": content_sha256,
    }


def normalize_infrastructure_event(
    value: Any,
    *,
    policy: dict[str, Any],
) -> dict[str, Any]:
    policy = normalize_gate_policy(policy)
    payload = _require_object(value, "infrastructure registry event")
    _require_exact_fields(payload, EVENT_FIELDS, "infrastructure registry event")
    if payload["schema_version"] != EVENT_SCHEMA_VERSION:
        raise ValueError(f"event.schema_version must be {EVENT_SCHEMA_VERSION}")
    if payload["gate_kind"] != GATE_KIND or payload["gate_contract_version"] != 1:
        raise ValueError("event gate identity is invalid")
    status = _require_nonempty_string(payload["status"], "event.status").lower()
    if status not in INFRASTRUCTURE_STATUSES:
        raise ValueError(f"unsupported infrastructure status: {status}")
    if status == "running" and payload["execution_kind"] != "synthetic":
        raise ValueError("infrastructure RUNNING requires execution_kind=synthetic")
    if status != "running" and payload["execution_kind"] not in {"none", "synthetic"}:
        raise ValueError("infrastructure execution_kind must be none or synthetic")
    if payload["execution_kind"] == "real-data":
        raise ValueError("infrastructure events never permit real-data execution")
    digest = _require_sha256(payload["proposal_scope_digest"], "event.proposal_scope_digest")
    assert isinstance(digest, str)
    gate_evaluation = _normalize_gate_evaluation(
        payload["gate_evaluation"], proposal_digest=digest, policy=policy
    )
    # A schema-v3 event is a pending governance candidate until its exact bytes
    # are human-merged into GitHub current main.  The v1 evidence compiler never
    # emits an executable preparation/run authority token.
    expected_flags = {
        "preparation_authorized": False,
        "synthetic_fixture_tests_allowed": False,
        "real_data_execution_allowed": False,
        "automatic_execution_allowed": False,
        "execution_authorized": False,
    }
    for field, expected in expected_flags.items():
        if payload[field] is not expected:
            raise ValueError(f"event.{field} must be {str(expected).lower()} for {status}")
    for field in (
        "production_approved",
        "merge_approved",
        "buy_approved",
        "production_change_allowed",
        "merge_allowed",
        "buy_logic_change_allowed",
        "formal_buy",
        "send_order",
    ):
        _require_false(payload[field], f"event.{field}")
    _require_zero(payload["stake"], "event.stake")
    sequence = _require_positive_int(payload["sequence"], "event.sequence")
    artifacts = _require_path_list(payload["artifacts"], "event.artifacts", allow_empty=True)
    revalidated = payload["revalidated_approval_evidence"]
    if not isinstance(revalidated, list):
        raise ValueError("event.revalidated_approval_evidence must be a list")
    canonical_json_bytes(revalidated)
    gate_policy_evidence = normalize_gate_policy_evidence(
        payload["gate_policy_evidence"]
    )
    main_registry_evidence = normalize_main_registry_evidence(
        payload["main_registry_evidence"]
    )
    if status in {"blocked_gate", "proposed"}:
        if gate_policy_evidence is not None:
            raise ValueError(f"event.gate_policy_evidence must be null for {status}")
        if main_registry_evidence is not None:
            raise ValueError(f"event.main_registry_evidence must be null for {status}")
    elif status != "invalid" and gate_policy_evidence is None:
        raise ValueError(f"event.gate_policy_evidence is required for {status}")
    if status not in {"blocked_gate", "proposed", "invalid"} and main_registry_evidence is None:
        raise ValueError(f"event.main_registry_evidence is required for {status}")
    for field in (
        "human_approved",
        "human_prepare_approval_recorded",
        "human_run_approval_recorded",
    ):
        _require_bool(payload[field], f"event.{field}")
    run_digest = payload["run_scope_digest"]
    if run_digest is not None:
        run_digest = _require_sha256(run_digest, "event.run_scope_digest")
    event_experiment_id = _require_nonempty_string(
        payload["experiment_id"], "event.experiment_id"
    )
    if not SAFE_EXPERIMENT_ID.fullmatch(event_experiment_id):
        raise ValueError("event.experiment_id is not a safe identifier")
    expected_queue_file = f"research/queue/{event_experiment_id}.json"
    queue_file = _require_repository_path(payload["queue_file"], "event.queue_file")
    if queue_file != expected_queue_file:
        raise ValueError(f"event.queue_file must be {expected_queue_file}")
    expected_run_scope_file = f"research/scopes/{event_experiment_id}.run.json"
    run_scope_file = (
        None
        if payload["run_scope_file"] is None
        else _require_repository_path(payload["run_scope_file"], "event.run_scope_file")
    )
    if run_scope_file is not None and run_scope_file != expected_run_scope_file:
        raise ValueError(f"event.run_scope_file must be {expected_run_scope_file}")

    normalized = dict(payload)
    normalized.update(
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "gate_kind": GATE_KIND,
            "gate_contract_version": 1,
            "sequence": sequence,
            "status": status,
            "gate_evaluation": gate_evaluation,
            "proposal_scope_digest": digest,
            "run_scope_digest": run_digest,
            "gate_policy_evidence": gate_policy_evidence,
            "main_registry_evidence": main_registry_evidence,
            "artifacts": artifacts,
            "event_id": _require_nonempty_string(payload["event_id"], "event.event_id"),
            "experiment_id": event_experiment_id,
            "occurred_at": _require_nonempty_string(payload["occurred_at"], "event.occurred_at"),
            "actor": _require_nonempty_string(payload["actor"], "event.actor"),
            "notes": str(payload["notes"]).strip(),
            "queue_file": queue_file,
            "run_scope_file": run_scope_file,
        }
    )
    canonical_json_bytes(normalized)
    return normalized


def build_initial_infrastructure_event(
    *,
    queue: dict[str, Any],
    policy: dict[str, Any],
    event_id: str,
    occurred_at: str,
    actor: str,
) -> dict[str, Any]:
    queue = normalize_infrastructure_queue(queue, policy=policy)
    event = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "sequence": 1,
        "experiment_id": queue["experiment_id"],
        "gate_kind": GATE_KIND,
        "gate_contract_version": 1,
        "status": "proposed",
        "previous_status": None,
        "previous_event_id": None,
        "occurred_at": occurred_at,
        "actor": actor,
        "gate_evaluation": queue["gate_evaluation"],
        "proposal_scope_digest": queue["proposal_scope_digest"],
        "run_scope_digest": None,
        "github_trust_evidence": None,
        "gate_policy_evidence": None,
        "main_registry_evidence": None,
        "approval_evidence": None,
        "revalidated_approval_evidence": [],
        "human_approved": False,
        "human_prepare_approval_recorded": False,
        "human_run_approval_recorded": False,
        "preparation_authorized": False,
        "synthetic_fixture_tests_allowed": False,
        "real_data_execution_allowed": False,
        "automatic_execution_allowed": False,
        "execution_authorized": False,
        "production_approved": False,
        "merge_approved": False,
        "buy_approved": False,
        "production_change_allowed": False,
        "merge_allowed": False,
        "buy_logic_change_allowed": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "execution_kind": "none",
        "artifacts": [],
        "notes": "Infrastructure safety proposal registered; no execution authorized.",
        "queue_file": f"research/queue/{queue['experiment_id']}.json",
        "run_scope_file": None,
    }
    return normalize_infrastructure_event(event, policy=policy)
