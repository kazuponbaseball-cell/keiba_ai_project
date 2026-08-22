#!/usr/bin/env python3
"""Versioned ordinary real-data run contract.

This module is deliberately separate from ``scope_contract.py``.  The latter is
the frozen legacy-v2 contract and its bytes, field set, canonical digests and
writer semantics must not be reinterpreted.

The v3 functions in this file are control-plane validators only.  They never
mount or read a real-data row/blob and they do not execute a command.  A real
row access boundary must additionally present a human-merged RUNNING event and
an exact execution receipt produced after the metadata-only preflight.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import locale as runtime_locale
import math
import os
import platform
import re
import subprocess
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from scope_contract import (
    canonical_digest,
    canonical_json_bytes,
    normalize_proposal_scope,
    normalize_run_scope,
    strict_json_load,
)
from github_approval import (
    COMMENT_EVIDENCE_FIELDS,
    DEFAULT_BASE_BRANCH,
    DEFAULT_REPOSITORY,
    GITHUB_API_URL,
    GitHubApprovalProvider,
    GitHubRestApprovalProvider,
    fetch_github_file_at_commit,
    verify_approval_comment,
    verify_github_main_head_unchanged,
    verify_github_trust,
)


RUN_SCOPE_SCHEMA_VERSION = "ordinary_real_data_run_v3"
RESULT_MANIFEST_SCHEMA_VERSION = "ordinary_real_data_result_v1"
EXECUTION_RECEIPT_SCHEMA_VERSION = "ordinary_real_data_execution_receipt_v1"
METADATA_PREFLIGHT_RECEIPT_VERSION = "ordinary_real_data_metadata_preflight_v1"
ENVIRONMENT_LOCK_SCHEMA_VERSION = "ordinary_real_data_environment_lock_v1"
INPUT_MANIFEST_SCHEMA_VERSION = "ordinary_real_data_input_manifest_v1"
OUTPUT_ATTESTATION_SCHEMA_VERSION = "ordinary_real_data_output_attestation_v1"

FULL_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}\Z")
PYTHON_VERSION = re.compile(r"3\.(?:11|12)\.[0-9]+\Z")
DATE_TEXT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
MUTABLE_PATH_PARTS = {"latest", "current", "active", "champion"}
AUTHORITY_MODULE_PATH = "scripts/research/ordinary_real_data_run_contract_v3.py"
EXECUTABLE_INPUT_SUFFIXES = {
    ".bat",
    ".cmd",
    ".com",
    ".dll",
    ".egg",
    ".exe",
    ".pyd",
    ".py",
    ".pyc",
    ".pyo",
    ".pth",
    ".ps1",
    ".sh",
    ".so",
    ".whl",
    ".zip",
}

CAPABILITY_FIELDS = (
    "read_real_input_manifests",
    "read_real_runner_rows",
    "canonicalize_input_release",
    "read_historical_training_rows",
    "read_sealed_canonical_input_release",
    "train_research_model",
    "validate_research_model",
    "calibrate_research_model",
    "evaluate_outer_oos_once",
    "infer_target_runners",
    "write_research_outputs",
    "seal_research_outputs",
    "production_model_write",
    "champion_change",
    "candidate_policy_change",
    "value_policy_change",
    "formal_buy",
    "notification",
    "order",
    "nonzero_stake",
    "merge",
    "production_promotion",
)

ALWAYS_FALSE_CAPABILITIES = (
    "production_model_write",
    "champion_change",
    "candidate_policy_change",
    "value_policy_change",
    "formal_buy",
    "notification",
    "order",
    "nonzero_stake",
    "merge",
    "production_promotion",
)


def _capabilities(*enabled: str) -> dict[str, bool]:
    unknown = set(enabled) - set(CAPABILITY_FIELDS)
    if unknown:
        raise AssertionError(f"unknown internal capability: {sorted(unknown)}")
    return {name: name in enabled for name in CAPABILITY_FIELDS}


CAPABILITY_PROFILES: dict[str, dict[str, Any]] = {
    "synthetic_governance_v1": {
        "execution_kind": "synthetic",
        "experiment_id": None,
        "capabilities": _capabilities(),
        "phase_plan": (
            ("metadata_preflight", ()),
        ),
    },
    "exp034_input_canonicalization_v1": {
        "execution_kind": "real_data",
        "experiment_id": "EXP-20260821-034",
        "capabilities": _capabilities(
            "read_real_input_manifests",
            "read_real_runner_rows",
            "canonicalize_input_release",
            "read_historical_training_rows",
            "write_research_outputs",
            "seal_research_outputs",
        ),
        "phase_plan": (
            ("metadata_preflight", ("read_real_input_manifests",)),
            (
                "canonicalize_input_release",
                (
                    "read_real_runner_rows",
                    "canonicalize_input_release",
                    "read_historical_training_rows",
                    "write_research_outputs",
                ),
            ),
            (
                "seal_research_outputs",
                ("write_research_outputs", "seal_research_outputs"),
            ),
        ),
    },
    "exp033_leakfree_research_v1": {
        "execution_kind": "real_data",
        "experiment_id": "EXP-20260821-033",
        "capabilities": _capabilities(
            "read_real_input_manifests",
            "read_real_runner_rows",
            "read_historical_training_rows",
            "read_sealed_canonical_input_release",
            "train_research_model",
            "validate_research_model",
            "calibrate_research_model",
            "evaluate_outer_oos_once",
            "infer_target_runners",
            "write_research_outputs",
            "seal_research_outputs",
        ),
        "phase_plan": (
            ("metadata_preflight", ("read_real_input_manifests",)),
            (
                "execute_research_plan",
                (
                    "read_real_runner_rows",
                    "read_historical_training_rows",
                    "read_sealed_canonical_input_release",
                    "train_research_model",
                    "validate_research_model",
                    "calibrate_research_model",
                    "evaluate_outer_oos_once",
                    "infer_target_runners",
                    "write_research_outputs",
                ),
            ),
            (
                "seal_research_outputs",
                ("write_research_outputs", "seal_research_outputs"),
            ),
        ),
    },
}

RUN_FIELDS = (
    "run_scope_schema_version",
    "proposal_scope",
    "proposal_scope_digest",
    "execution_kind",
    "capability_profile",
    "execution_commit_sha",
    "code_hashes",
    "config_hashes",
    "data_input_manifest_hashes",
    "input_catalog",
    "runner_universe_manifest_hash",
    "feature_input_release_hash",
    "training_source_manifest_hash",
    "target_source_manifest_hash",
    "feature_lineage_manifest_hash",
    "label_eligibility_contract_hash",
    "fold_manifest_hash",
    "dependency_environment_lock_hash",
    "environment",
    "random_seed",
    "repository_working_directory",
    "exact_commands",
    "network_policy",
    "read_allowlist",
    "write_allowlist",
    "output_root",
    "compute_budget",
    "source_cutoff",
    "as_of",
    "phase_plan",
    "output_sealing_contract",
    "formal_buy",
    "send_order",
    "stake",
)

REF_FIELDS = ("path", "sha256")
PROFILE_FIELDS = ("profile_id", "capabilities")
CATALOG_FIELDS = (
    "catalog_id",
    "source_release_id",
    "manifest",
    "attestation",
    "row_count",
    "race_count",
    "runner_count",
    "source_event_time_coverage",
    "received_at_coverage",
    "available_as_of_coverage",
    "max_source_event_time",
    "max_received_at",
    "max_available_as_of",
    "revoked",
    "revocation_status",
    "runner_universe_digest",
    "runner_identity_digest",
    "target_date",
    "race_ids",
    "phase_read_capabilities",
    "metadata_manifest_refs",
    "row_blob_refs",
)
CATALOG_PAYLOAD_FIELDS = tuple(
    field for field in CATALOG_FIELDS if field not in {"manifest", "attestation"}
)
ATTESTATION_FIELDS = (
    "kind",
    "content_sha256",
    "signature_sha256",
    "signer_identity",
)
COVERAGE_FIELDS = ("covered_count", "total_count")
ARTIFACT_BINDING_FIELDS = (
    "binding_role",
    "artifact_role",
    "contract",
    "producer_run_scope",
    "producer_execution_receipt",
    "producer_output_attestation",
    "artifact_manifest",
    "artifact_sha256",
)
ENVIRONMENT_FIELDS = (
    "interpreter_path",
    "interpreter_version",
    "dependency_versions",
    "locale",
    "timezone",
)
DEPENDENCY_FIELDS = ("name", "version")
ENVIRONMENT_LOCK_FIELDS = ("schema_version", "environment")
INPUT_MANIFEST_FIELDS = (
    "schema_version",
    "manifest_id",
    "manifest_kind",
    "source_release_id",
    "row_count",
    "race_count",
    "runner_count",
    "source_event_time_coverage",
    "received_at_coverage",
    "available_as_of_coverage",
    "max_source_event_time",
    "max_received_at",
    "max_available_as_of",
    "revoked",
    "revocation_status",
    "row_blob_refs",
    "identity_set_sha256",
)
INPUT_MANIFEST_KINDS = {
    "runner_universe",
    "training_source",
    "target_source",
    "supporting_input",
    "synthetic_fixture",
}
COMMAND_FIELDS = (
    "command_id",
    "phase_id",
    "executable",
    "argv",
    "working_directory",
    "timeout_seconds",
)
READ_ALLOW_FIELDS = (
    "path",
    "sha256",
    "access_class",
    "required_capability",
    "phases",
)
WRITE_ALLOW_FIELDS = ("path", "required_capability", "phases")
ROW_ACCESS_CAPABILITY = {
    "runner_row_blob": "read_real_runner_rows",
    "canonicalization_source_row_blob": "canonicalize_input_release",
    "historical_training_row_blob": "read_historical_training_rows",
    "sealed_input_row_blob": "read_sealed_canonical_input_release",
}
PHASE_FIELDS = (
    "phase_id",
    "required_capabilities",
    "command_id",
    "read_paths",
    "write_paths",
)
COMPUTE_FIELDS = (
    "timeout_seconds",
    "cpu_cores",
    "memory_mib",
    "disk_mib",
    "max_model_fits",
    "max_outer_oos_evaluations",
    "max_target_inference_calls",
)
OUTPUT_CONTRACT_FIELDS = (
    "schema_version",
    "mode",
    "execution_receipt_path",
    "result_manifest_path",
    "failure_manifest_path",
    "overwrite_allowed",
    "partial_output_consumer_eligible",
    "failure_manifest_required",
    "artifact_roles",
    "artifact_paths",
    "required_result_fields",
)
ARTIFACT_PATH_FIELDS = ("role", "path")
RESULT_FIELDS = (
    "result_manifest_schema_version",
    "experiment_id",
    "capability_profile_id",
    "run_scope_digest",
    "execution_receipt_digest",
    "status",
    "generated_at",
    "as_of",
    "output_root",
    "artifacts",
    "partial_outputs",
    "code_hashes_digest",
    "config_hashes_digest",
    "input_manifest_hashes_digest",
    "environment_lock_sha256",
    "consumer_eligible",
    "formal_buy",
    "send_order",
    "stake",
)
RESULT_ARTIFACT_FIELDS = (
    "role",
    "path",
    "sha256",
    "row_count",
    "race_count",
    "runner_count",
    "complete",
)
OUTPUT_ATTESTATION_FIELDS = (
    "attestation_schema_version",
    "producer_experiment_id",
    "producer_run_scope_path",
    "producer_run_scope_digest",
    "running_event_id",
    "execution_receipt_path",
    "execution_receipt_sha256",
    "execution_receipt_digest",
    "result_manifest_path",
    "result_manifest_sha256",
    "result_manifest_digest",
    "artifacts_digest",
    "output_root",
    "status",
    "consumer_eligible",
    "attested_at",
    "formal_buy",
    "send_order",
    "stake",
)
EXECUTION_RECEIPT_FIELDS = (
    "receipt_schema_version",
    "experiment_id",
    "running_event_id",
    "run_scope_digest",
    "execution_kind",
    "capability_profile_id",
    "execution_commit_sha",
    "verified_current_main_sha",
    "execution_commit_compare_status",
    "execution_commit_compare_url",
    "execution_commit_merge_base_sha",
    "current_main_registry_sha256",
    "prepare_approval_comment_id",
    "run_approval_comment_id",
    "metadata_preflight_digest",
    "capability_profile_digest",
    "input_manifest_hashes_digest",
    "environment_digest",
    "exact_commands_digest",
    "read_allowlist_digest",
    "write_allowlist_digest",
    "output_root",
    "output_root_reservation_digest",
    "output_root_was_fresh",
    "real_data_execution_allowed",
    "execution_authorized",
    "issued_at",
    "formal_buy",
    "send_order",
    "stake",
)
AUTHORITY_CONTEXT_FIELDS = (
    "root",
    "status",
    "cli_execution_kind",
    "prepare_evidence",
    "run_evidence",
    "execution_commit",
    "current_main_sha",
    "merged_running_event",
    "current_main_registry_bytes",
    "execution_receipt",
    "metadata_preflight_receipt",
    "observed_environment",
)
ANCESTRY_EVIDENCE_FIELDS = ("status", "url", "merge_base_sha")
REQUIRED_RESULT_FIELDS = RESULT_FIELDS

ARTIFACT_ROLES_BY_PROFILE = {
    "synthetic_governance_v1": (),
    "exp034_input_canonicalization_v1": (
        "canonical_runner_universe",
        "canonical_training_input_release",
        "canonical_target_input_release",
        "canonical_feature_lineage",
        "label_eligibility_reconciliation",
    ),
    "exp033_leakfree_research_v1": (
        "research_model",
        "validation_report",
        "calibration_report",
        "outer_oos_report",
        "target_research_prediction",
    ),
}
ARTIFACT_FORMAT_BY_ROLE = {
    "synthetic_fixture_report": "canonical_json",
    "canonical_runner_universe": "identity_jsonl",
    "canonical_training_input_release": "identity_jsonl",
    "canonical_target_input_release": "identity_jsonl",
    "canonical_feature_lineage": "identity_jsonl",
    "label_eligibility_reconciliation": "identity_jsonl",
    "research_model": "opaque_binary",
    "validation_report": "canonical_json",
    "calibration_report": "canonical_json",
    "outer_oos_report": "identity_jsonl",
    "target_research_prediction": "identity_jsonl",
}
REQUIRED_RUNNER_BY_PROFILE = {
    "exp034_input_canonicalization_v1": (
        "scripts/research/run_exp033_input_canonicalization_v0.py"
    ),
    "exp033_leakfree_research_v1": (
        "scripts/research/run_leakfree_predraw_baseline_v0.py"
    ),
}
REQUIRED_CONFIG_BY_PROFILE = {
    "exp034_input_canonicalization_v1": (
        "research/configs/EXP-20260821-034.input_canonicalization_v0.json"
    ),
    "exp033_leakfree_research_v1": (
        "research/configs/EXP-20260821-033.leakfree_predraw_baseline_v0.json"
    ),
}
REQUIRED_DEPENDENCIES_BY_PROFILE = {
    "synthetic_governance_v1": {"python"},
    "exp034_input_canonicalization_v1": {"python"},
    "exp033_leakfree_research_v1": {"numpy", "pandas", "python"},
}


class ContractError(ValueError):
    """Fail-closed v3 contract violation."""


def _exact_object(value: Any, fields: Iterable[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    expected = set(fields)
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ContractError(f"{label} fields differ; missing={missing}, extra={extra}")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"{label} must be a canonical nonblank string")
    return value


def _identifier(value: Any, label: str) -> str:
    text = _string(value, label)
    if not SAFE_ID.fullmatch(text):
        raise ContractError(f"{label} is not a safe identifier")
    return text


def _hash(value: Any, label: str) -> str:
    text = _string(value, label)
    if not FULL_SHA256.fullmatch(text):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return text


def _git_sha(value: Any, label: str) -> str:
    text = _string(value, label)
    if not FULL_GIT_SHA.fullmatch(text):
        raise ContractError(f"{label} must be a lowercase Git commit SHA")
    return text


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{label} must be a boolean")
    return value


def _canonical_timestamp(value: Any, label: str) -> str:
    text = _string(value, label)
    if not text.endswith("Z"):
        raise ContractError(f"{label} must be canonical UTC RFC3339 ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is not a valid RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError(f"{label} must be UTC")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if canonical != text:
        raise ContractError(f"{label} is not in canonical UTC form")
    return text


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _repository_path(value: Any, label: str) -> str:
    text = _string(value, label)
    if "\\" in text or any(token in text for token in ("*", "?", "[", "]")):
        raise ContractError(f"{label} must be an exact POSIX repository path")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError(f"{label} must be normalized and repository-relative")
    if any(part.casefold() in MUTABLE_PATH_PARTS for part in path.parts):
        raise ContractError(f"{label} contains a mutable alias")
    return path.as_posix()


def _absolute_working_directory(value: Any, label: str) -> str:
    text = _string(value, label)
    if not (text.startswith("/") or re.match(r"^[A-Za-z]:/", text)):
        raise ContractError(f"{label} must be an exact absolute path")
    if "\\" in text or text.endswith("/") or "/../" in f"/{text}/":
        raise ContractError(f"{label} must be canonical")
    return text


def _sorted_unique_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractError(f"{label} must be a string array")
    normalized = [_string(item, f"{label} item") for item in value]
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise ContractError(f"{label} must be sorted and duplicate-free")
    return normalized


def _ref(value: Any, label: str) -> dict[str, str]:
    obj = _exact_object(value, REF_FIELDS, label)
    return {
        "path": _repository_path(obj["path"], f"{label}.path"),
        "sha256": _hash(obj["sha256"], f"{label}.sha256"),
    }


def _ref_list(value: Any, label: str, *, nonempty: bool = True) -> list[dict[str, str]]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ContractError(f"{label} must be a{' nonempty' if nonempty else ''} array")
    refs = [_ref(item, f"{label}[{index}]") for index, item in enumerate(value)]
    paths = [item["path"] for item in refs]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ContractError(f"{label} must be sorted by path and duplicate-free")
    return refs


def _coverage(value: Any, label: str, expected_total: int) -> dict[str, int]:
    obj = _exact_object(value, COVERAGE_FIELDS, label)
    covered = _integer(obj["covered_count"], f"{label}.covered_count")
    total = _integer(obj["total_count"], f"{label}.total_count")
    if total != expected_total or covered != total:
        raise ContractError(f"{label} must cover every catalog row")
    return {"covered_count": covered, "total_count": total}


def _catalog_access_refs(
    value: Any,
    label: str,
    *,
    metadata: bool,
) -> list[dict[str, Any]]:
    """Normalize the catalog's exact metadata/row access inventory.

    The catalog payload is hash-bound before any row mount.  Carrying the same
    typed path/hash/capability/phase tuples here prevents an unrelated row blob
    from being substituted into an otherwise valid source manifest.
    """

    if not isinstance(value, list) or (metadata and not value):
        raise ContractError(f"{label} must be a{' nonempty' if metadata else ''} array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        obj = _exact_object(item, READ_ALLOW_FIELDS, f"{label}[{index}]")
        capability = obj["required_capability"]
        if capability is not None:
            capability = _string(capability, f"{label}[{index}].required_capability")
        access_class = _string(obj["access_class"], f"{label}[{index}].access_class")
        if metadata:
            if access_class != "metadata_manifest":
                raise ContractError(f"{label} may contain only metadata manifests")
        else:
            expected_capability = ROW_ACCESS_CAPABILITY.get(access_class)
            if expected_capability is None or capability != expected_capability:
                raise ContractError(
                    f"{label} row class differs from its finite capability"
                )
        path = _repository_path(obj["path"], f"{label}[{index}].path")
        if not metadata:
            lexical = PurePosixPath(path)
            if (
                lexical.suffix.casefold() in EXECUTABLE_INPUT_SUFFIXES
                or any(part.casefold() == "__pycache__" for part in lexical.parts)
            ):
                raise ContractError("row input paths cannot contain executable code")
        result.append(
            {
                "path": path,
                "sha256": _hash(obj["sha256"], f"{label}[{index}].sha256"),
                "access_class": access_class,
                "required_capability": capability,
                "phases": _sorted_unique_strings(
                    obj["phases"], f"{label}[{index}].phases"
                ),
            }
        )
    paths = [item["path"] for item in result]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ContractError(f"{label} must be sorted and path-unique")
    return result


def _normalize_input_catalog(value: Any) -> dict[str, Any]:
    obj = _exact_object(value, CATALOG_FIELDS, "input_catalog")
    row_count = _integer(obj["row_count"], "input_catalog.row_count")
    race_count = _integer(obj["race_count"], "input_catalog.race_count")
    runner_count = _integer(obj["runner_count"], "input_catalog.runner_count")
    manifest = _ref(obj["manifest"], "input_catalog.manifest")
    attestation_obj = _exact_object(
        obj["attestation"], ATTESTATION_FIELDS, "input_catalog.attestation"
    )
    kind = _string(attestation_obj["kind"], "input_catalog.attestation.kind")
    if kind != "sha256_bound":
        raise ContractError("input_catalog must use the implemented sha256-bound attestation")
    content_sha = _hash(
        attestation_obj["content_sha256"],
        "input_catalog.attestation.content_sha256",
    )
    signature = attestation_obj["signature_sha256"]
    signer = attestation_obj["signer_identity"]
    if signature is not None or signer is not None:
        raise ContractError("sha256-bound catalog must not claim an unverified signature")
    if content_sha != manifest["sha256"]:
        raise ContractError("catalog attestation digest differs from manifest digest")
    if _bool(obj["revoked"], "input_catalog.revoked") is not False:
        raise ContractError("revoked input catalog is forbidden")
    if obj["revocation_status"] != "active":
        raise ContractError("input catalog revocation_status must be active")
    target_date = _string(obj["target_date"], "input_catalog.target_date")
    if not DATE_TEXT.fullmatch(target_date):
        raise ContractError("input_catalog.target_date must be YYYY-MM-DD")
    race_ids = _sorted_unique_strings(obj["race_ids"], "input_catalog.race_ids")
    if len(race_ids) != race_count:
        raise ContractError("input catalog race count differs from race_ids")
    phase_caps = _sorted_unique_strings(
        obj["phase_read_capabilities"], "input_catalog.phase_read_capabilities"
    )
    if any(item not in CAPABILITY_FIELDS for item in phase_caps):
        raise ContractError("input catalog cites an unknown phase capability")
    return {
        "catalog_id": _identifier(obj["catalog_id"], "input_catalog.catalog_id"),
        "source_release_id": _identifier(
            obj["source_release_id"], "input_catalog.source_release_id"
        ),
        "manifest": manifest,
        "attestation": {
            "kind": kind,
            "content_sha256": content_sha,
            "signature_sha256": signature,
            "signer_identity": signer,
        },
        "row_count": row_count,
        "race_count": race_count,
        "runner_count": runner_count,
        "source_event_time_coverage": _coverage(
            obj["source_event_time_coverage"],
            "input_catalog.source_event_time_coverage",
            row_count,
        ),
        "received_at_coverage": _coverage(
            obj["received_at_coverage"],
            "input_catalog.received_at_coverage",
            row_count,
        ),
        "available_as_of_coverage": _coverage(
            obj["available_as_of_coverage"],
            "input_catalog.available_as_of_coverage",
            row_count,
        ),
        "max_source_event_time": _canonical_timestamp(
            obj["max_source_event_time"], "input_catalog.max_source_event_time"
        ),
        "max_received_at": _canonical_timestamp(
            obj["max_received_at"], "input_catalog.max_received_at"
        ),
        "max_available_as_of": _canonical_timestamp(
            obj["max_available_as_of"], "input_catalog.max_available_as_of"
        ),
        "revoked": False,
        "revocation_status": "active",
        "runner_universe_digest": _hash(
            obj["runner_universe_digest"], "input_catalog.runner_universe_digest"
        ),
        "runner_identity_digest": _hash(
            obj["runner_identity_digest"], "input_catalog.runner_identity_digest"
        ),
        "target_date": target_date,
        "race_ids": race_ids,
        "phase_read_capabilities": phase_caps,
        "metadata_manifest_refs": _catalog_access_refs(
            obj["metadata_manifest_refs"],
            "input_catalog.metadata_manifest_refs",
            metadata=True,
        ),
        "row_blob_refs": _catalog_access_refs(
            obj["row_blob_refs"],
            "input_catalog.row_blob_refs",
            metadata=False,
        ),
    }


def _catalog_payload(catalog: dict[str, Any]) -> dict[str, Any]:
    return {field: catalog[field] for field in CATALOG_PAYLOAD_FIELDS}


def _normalize_catalog_payload(value: Any) -> dict[str, Any]:
    """Normalize the separately hashed catalog payload (no self-reference)."""

    obj = _exact_object(value, CATALOG_PAYLOAD_FIELDS, "input catalog payload")
    synthetic_wrapper = {
        **obj,
        "manifest": {
            "path": "research/catalogs/payload-placeholder.json",
            "sha256": "0" * 64,
        },
        "attestation": {
            "kind": "sha256_bound",
            "content_sha256": "0" * 64,
            "signature_sha256": None,
            "signer_identity": None,
        },
    }
    normalized = _normalize_input_catalog(synthetic_wrapper)
    return _catalog_payload(normalized)


def _normalize_input_manifest(value: Any, label: str) -> dict[str, Any]:
    obj = _exact_object(value, INPUT_MANIFEST_FIELDS, label)
    if obj["schema_version"] != INPUT_MANIFEST_SCHEMA_VERSION:
        raise ContractError(f"{label} has an unknown schema version")
    row_count = _integer(obj["row_count"], f"{label}.row_count")
    race_count = _integer(obj["race_count"], f"{label}.race_count")
    runner_count = _integer(obj["runner_count"], f"{label}.runner_count")
    manifest_kind = _string(obj["manifest_kind"], f"{label}.manifest_kind")
    if manifest_kind not in INPUT_MANIFEST_KINDS:
        raise ContractError(f"{label} has an unsupported manifest kind")
    row_refs = _catalog_access_refs(
        obj["row_blob_refs"], f"{label}.row_blob_refs", metadata=False
    )
    if manifest_kind == "synthetic_fixture":
        if row_refs or any((row_count, race_count, runner_count)):
            raise ContractError(f"{label} synthetic fixture manifest must be row-empty")
    elif not row_refs or min(row_count, race_count, runner_count) < 1:
        raise ContractError(f"{label} must enumerate nonempty exact row metadata")
    maxima = {
        field: _canonical_timestamp(obj[field], f"{label}.{field}")
        for field in (
            "max_source_event_time",
            "max_received_at",
            "max_available_as_of",
        )
    }
    parsed_maxima = [
        datetime.fromisoformat(maxima[field][:-1] + "+00:00")
        for field in (
            "max_source_event_time",
            "max_received_at",
            "max_available_as_of",
        )
    ]
    if parsed_maxima != sorted(parsed_maxima):
        raise ContractError(f"{label} source/receipt/availability maxima are inconsistent")
    if _bool(obj["revoked"], f"{label}.revoked") is not False:
        raise ContractError(f"{label} is revoked")
    if obj["revocation_status"] != "active":
        raise ContractError(f"{label} revocation status is not active")
    return {
        "schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
        "manifest_id": _identifier(obj["manifest_id"], f"{label}.manifest_id"),
        "manifest_kind": manifest_kind,
        "source_release_id": _identifier(
            obj["source_release_id"], f"{label}.source_release_id"
        ),
        "row_count": row_count,
        "race_count": race_count,
        "runner_count": runner_count,
        "source_event_time_coverage": _coverage(
            obj["source_event_time_coverage"],
            f"{label}.source_event_time_coverage",
            row_count,
        ),
        "received_at_coverage": _coverage(
            obj["received_at_coverage"],
            f"{label}.received_at_coverage",
            row_count,
        ),
        "available_as_of_coverage": _coverage(
            obj["available_as_of_coverage"],
            f"{label}.available_as_of_coverage",
            row_count,
        ),
        **maxima,
        "revoked": False,
        "revocation_status": "active",
        "row_blob_refs": row_refs,
        "identity_set_sha256": _hash(
            obj["identity_set_sha256"], f"{label}.identity_set_sha256"
        ),
    }


def _artifact_binding(value: Any, label: str, *, expected_role: str) -> dict[str, Any]:
    obj = _exact_object(value, ARTIFACT_BINDING_FIELDS, label)
    role = _string(obj["binding_role"], f"{label}.binding_role")
    artifact_role = _identifier(obj["artifact_role"], f"{label}.artifact_role")
    if artifact_role != expected_role:
        raise ContractError(f"{label} artifact role differs from the finite connector")
    contract = _ref(obj["contract"], f"{label}.contract")
    artifact = obj["artifact_manifest"]
    if role == "planned_output_contract":
        if (
            artifact is not None
            or obj["artifact_sha256"] is not None
            or obj["producer_run_scope"] is not None
            or obj["producer_execution_receipt"] is not None
            or obj["producer_output_attestation"] is not None
        ):
            raise ContractError(f"{label} planned output cannot predeclare a future content hash")
        normalized_artifact = None
        artifact_sha256 = None
        producer_run_scope = None
        producer_execution_receipt = None
        producer_output_attestation = None
    elif role == "sealed_input_artifact":
        normalized_artifact = _ref(artifact, f"{label}.artifact_manifest")
        artifact_sha256 = _hash(obj["artifact_sha256"], f"{label}.artifact_sha256")
        producer_run_scope = _ref(
            obj["producer_run_scope"], f"{label}.producer_run_scope"
        )
        producer_execution_receipt = _ref(
            obj["producer_execution_receipt"],
            f"{label}.producer_execution_receipt",
        )
        producer_output_attestation = _ref(
            obj["producer_output_attestation"],
            f"{label}.producer_output_attestation",
        )
    else:
        raise ContractError(f"{label} has an unsupported binding role")
    return {
        "binding_role": role,
        "artifact_role": artifact_role,
        "contract": contract,
        "producer_run_scope": producer_run_scope,
        "producer_execution_receipt": producer_execution_receipt,
        "producer_output_attestation": producer_output_attestation,
        "artifact_manifest": normalized_artifact,
        "artifact_sha256": artifact_sha256,
    }


def validate_capability_profile(
    value: Any,
    *,
    execution_kind: str,
    experiment_id: str,
) -> dict[str, Any]:
    obj = _exact_object(value, PROFILE_FIELDS, "capability_profile")
    profile_id = _identifier(obj["profile_id"], "capability_profile.profile_id")
    policy = CAPABILITY_PROFILES.get(profile_id)
    if policy is None:
        raise ContractError("unknown capability profile")
    if policy["execution_kind"] != execution_kind:
        raise ContractError("capability profile execution kind mismatch")
    expected_experiment = policy["experiment_id"]
    if expected_experiment is not None and experiment_id != expected_experiment:
        raise ContractError("capability profile is bound to a different experiment")
    capabilities = _exact_object(
        obj["capabilities"], CAPABILITY_FIELDS, "capability_profile.capabilities"
    )
    normalized = {
        field: _bool(capabilities[field], f"capability_profile.capabilities.{field}")
        for field in CAPABILITY_FIELDS
    }
    if normalized != policy["capabilities"]:
        raise ContractError("capability profile differs from its finite trusted definition")
    if any(normalized[field] for field in ALWAYS_FALSE_CAPABILITIES):
        raise ContractError("prohibited production/BUY capability must remain false")
    if execution_kind == "synthetic" and any(
        normalized[field]
        for field in (
            "read_real_input_manifests",
            "read_real_runner_rows",
            "canonicalize_input_release",
            "read_historical_training_rows",
            "read_sealed_canonical_input_release",
            "train_research_model",
            "validate_research_model",
            "calibrate_research_model",
            "evaluate_outer_oos_once",
            "infer_target_runners",
        )
    ):
        raise ContractError("synthetic profile cannot acquire real-data capability")
    return {"profile_id": profile_id, "capabilities": normalized}


def _environment(value: Any) -> dict[str, Any]:
    obj = _exact_object(value, ENVIRONMENT_FIELDS, "environment")
    interpreter_path = _absolute_working_directory(
        obj["interpreter_path"], "environment.interpreter_path"
    )
    version = _string(obj["interpreter_version"], "environment.interpreter_version")
    if not PYTHON_VERSION.fullmatch(version):
        raise ContractError("environment interpreter must be exact Python 3.11 or 3.12")
    dependencies_raw = obj["dependency_versions"]
    if not isinstance(dependencies_raw, list) or not dependencies_raw:
        raise ContractError("environment dependency_versions must be nonempty")
    dependencies: list[dict[str, str]] = []
    for index, item in enumerate(dependencies_raw):
        dep = _exact_object(item, DEPENDENCY_FIELDS, f"dependency_versions[{index}]")
        dependencies.append(
            {
                "name": _identifier(dep["name"], f"dependency_versions[{index}].name"),
                "version": _string(dep["version"], f"dependency_versions[{index}].version"),
            }
        )
    names = [item["name"] for item in dependencies]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ContractError("dependency_versions must be sorted and duplicate-free")
    return {
        "interpreter_path": interpreter_path,
        "interpreter_version": version,
        "dependency_versions": dependencies,
        "locale": _string(obj["locale"], "environment.locale"),
        "timezone": _string(obj["timezone"], "environment.timezone"),
    }


def _normalize_environment_lock(value: Any) -> dict[str, Any]:
    obj = _exact_object(value, ENVIRONMENT_LOCK_FIELDS, "dependency environment lock")
    if obj["schema_version"] != ENVIRONMENT_LOCK_SCHEMA_VERSION:
        raise ContractError("unknown dependency environment lock schema version")
    return {
        "schema_version": ENVIRONMENT_LOCK_SCHEMA_VERSION,
        "environment": _environment(obj["environment"]),
    }


def _read_allowlist(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractError("read_allowlist must be an array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        obj = _exact_object(item, READ_ALLOW_FIELDS, f"read_allowlist[{index}]")
        capability = obj["required_capability"]
        if capability is not None:
            capability = _string(
                capability, f"read_allowlist[{index}].required_capability"
            )
            if capability not in CAPABILITY_FIELDS:
                raise ContractError("read allowlist cites an unknown capability")
        result.append(
            {
                "path": _repository_path(obj["path"], f"read_allowlist[{index}].path"),
                "sha256": _hash(obj["sha256"], f"read_allowlist[{index}].sha256"),
                "access_class": _string(
                    obj["access_class"], f"read_allowlist[{index}].access_class"
                ),
                "required_capability": capability,
                "phases": _sorted_unique_strings(
                    obj["phases"], f"read_allowlist[{index}].phases"
                ),
            }
        )
        if result[-1]["access_class"] not in {
            "metadata_manifest",
            *ROW_ACCESS_CAPABILITY,
        }:
            raise ContractError("read allowlist access_class is unsupported")
        expected_capability = ROW_ACCESS_CAPABILITY.get(result[-1]["access_class"])
        if expected_capability is not None and capability != expected_capability:
            raise ContractError("row access class differs from its required capability")
        if expected_capability is not None:
            lexical = PurePosixPath(result[-1]["path"])
            if (
                lexical.suffix.casefold() in EXECUTABLE_INPUT_SUFFIXES
                or any(part.casefold() == "__pycache__" for part in lexical.parts)
            ):
                raise ContractError("row input paths cannot contain executable code")
    paths = [item["path"] for item in result]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ContractError("read_allowlist must be sorted and path-unique")
    return result


def _write_allowlist(
    value: Any, output_root: str, *, allow_empty: bool = False
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ContractError("write_allowlist must be an array allowed by its profile")
    result: list[dict[str, Any]] = []
    prefix = output_root + "/"
    for index, item in enumerate(value):
        obj = _exact_object(item, WRITE_ALLOW_FIELDS, f"write_allowlist[{index}]")
        path = _repository_path(obj["path"], f"write_allowlist[{index}].path")
        if not path.startswith(prefix):
            raise ContractError("write allowlist path is outside the frozen output root")
        if PurePosixPath(path).suffix.casefold() in EXECUTABLE_INPUT_SUFFIXES:
            raise ContractError("write allowlist cannot create executable/importable code")
        result.append(
            {
                "path": path,
                "required_capability": _string(
                    obj["required_capability"],
                    f"write_allowlist[{index}].required_capability",
                ),
                "phases": _sorted_unique_strings(
                    obj["phases"], f"write_allowlist[{index}].phases"
                ),
            }
        )
        if result[-1]["required_capability"] != "write_research_outputs":
            raise ContractError("write allowlist requires write_research_outputs")
    paths = [item["path"] for item in result]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ContractError("write_allowlist must be sorted and path-unique")
    return result


def _commands(
    value: Any,
    *,
    environment: dict[str, Any],
    working_directory: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ContractError("exact_commands must be a nonempty ordered array")
    commands: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(value):
        obj = _exact_object(item, COMMAND_FIELDS, f"exact_commands[{index}]")
        command_id = _identifier(obj["command_id"], f"exact_commands[{index}].command_id")
        if command_id in ids:
            raise ContractError("exact command IDs must be unique")
        ids.add(command_id)
        executable = _absolute_working_directory(
            obj["executable"], f"exact_commands[{index}].executable"
        )
        if executable != environment["interpreter_path"]:
            raise ContractError("exact command executable differs from frozen interpreter")
        argv = obj["argv"]
        if not isinstance(argv, list) or len(argv) < 2:
            raise ContractError("exact command argv must be a structured nonempty array")
        normalized_argv = [
            _string(arg, f"exact_commands[{index}].argv[{arg_index}]")
            for arg_index, arg in enumerate(argv)
        ]
        if normalized_argv[0] != executable:
            raise ContractError("argv[0] must equal the frozen executable")
        if any(arg in {"-c", "--command"} for arg in normalized_argv[1:]):
            raise ContractError("free-form interpreter command text is forbidden")
        if any(
            PurePosixPath(arg.casefold()).name
            in {"sh", "bash", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh"}
            for arg in normalized_argv
        ):
            raise ContractError("shell commands are forbidden; use exact structured argv")
        command_cwd = _absolute_working_directory(
            obj["working_directory"], f"exact_commands[{index}].working_directory"
        )
        if command_cwd != working_directory:
            raise ContractError("exact command working directory differs from repository cwd")
        commands.append(
            {
                "command_id": command_id,
                "phase_id": _identifier(
                    obj["phase_id"], f"exact_commands[{index}].phase_id"
                ),
                "executable": executable,
                "argv": normalized_argv,
                "working_directory": command_cwd,
                "timeout_seconds": _integer(
                    obj["timeout_seconds"],
                    f"exact_commands[{index}].timeout_seconds",
                    minimum=1,
                ),
            }
        )
    return commands


def _phases(
    value: Any,
    *,
    profile_id: str,
    commands: list[dict[str, Any]],
    read_allowlist: list[dict[str, Any]],
    write_allowlist: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractError("phase_plan must be an array")
    expected = CAPABILITY_PROFILES[profile_id]["phase_plan"]
    if len(value) != len(expected):
        raise ContractError("phase plan length differs from the finite profile")
    command_map = {item["command_id"]: item for item in commands}
    read_map = {item["path"]: item for item in read_allowlist}
    write_map = {item["path"]: item for item in write_allowlist}
    normalized: list[dict[str, Any]] = []
    used_commands: set[str] = set()
    for index, ((expected_id, expected_caps), item) in enumerate(zip(expected, value)):
        obj = _exact_object(item, PHASE_FIELDS, f"phase_plan[{index}]")
        phase_id = _identifier(obj["phase_id"], f"phase_plan[{index}].phase_id")
        if phase_id != expected_id:
            raise ContractError("phase order/name differs from the finite capability profile")
        required = _sorted_unique_strings(
            obj["required_capabilities"],
            f"phase_plan[{index}].required_capabilities",
        )
        if required != sorted(expected_caps):
            raise ContractError("phase required capabilities differ from the finite profile")
        command_id = _identifier(obj["command_id"], f"phase_plan[{index}].command_id")
        command = command_map.get(command_id)
        if command is None or command["phase_id"] != phase_id:
            raise ContractError("phase command is absent or belongs to another phase")
        if command_id in used_commands:
            raise ContractError("one exact command cannot be reused across phases")
        used_commands.add(command_id)
        read_paths = _sorted_unique_strings(obj["read_paths"], f"phase_plan[{index}].read_paths")
        write_paths = _sorted_unique_strings(
            obj["write_paths"], f"phase_plan[{index}].write_paths"
        )
        for path in read_paths:
            if path not in read_map or phase_id not in read_map[path]["phases"]:
                raise ContractError("phase read path is not authorized by the read allowlist")
            entry = read_map[path]
            capability = entry["required_capability"]
            if capability is not None and capability not in required:
                raise ContractError("phase lacks the read path's exact required capability")
            if entry["access_class"] == "metadata_manifest":
                expected_metadata_capability = (
                    None
                    if profile_id == "synthetic_governance_v1"
                    else "read_real_input_manifests"
                )
                if (
                    phase_id != "metadata_preflight"
                    or capability != expected_metadata_capability
                ):
                    raise ContractError(
                        "metadata manifest reads are limited to the metadata preflight capability"
                    )
        for path in write_paths:
            if path not in write_map or phase_id not in write_map[path]["phases"]:
                raise ContractError("phase write path is not authorized by the write allowlist")
            if write_map[path]["required_capability"] not in required:
                raise ContractError("phase lacks write_research_outputs for an output path")
        normalized.append(
            {
                "phase_id": phase_id,
                "required_capabilities": required,
                "command_id": command_id,
                "read_paths": read_paths,
                "write_paths": write_paths,
            }
        )
    if used_commands != set(command_map):
        raise ContractError("every exact command must be bound to exactly one phase")
    return normalized


def _compute_budget(value: Any, profile_id: str) -> dict[str, int]:
    obj = _exact_object(value, COMPUTE_FIELDS, "compute_budget")
    result = {
        "timeout_seconds": _integer(obj["timeout_seconds"], "compute_budget.timeout_seconds", minimum=1),
        "cpu_cores": _integer(obj["cpu_cores"], "compute_budget.cpu_cores", minimum=1),
        "memory_mib": _integer(obj["memory_mib"], "compute_budget.memory_mib", minimum=1),
        "disk_mib": _integer(obj["disk_mib"], "compute_budget.disk_mib", minimum=1),
        "max_model_fits": _integer(obj["max_model_fits"], "compute_budget.max_model_fits"),
        "max_outer_oos_evaluations": _integer(
            obj["max_outer_oos_evaluations"], "compute_budget.max_outer_oos_evaluations"
        ),
        "max_target_inference_calls": _integer(
            obj["max_target_inference_calls"], "compute_budget.max_target_inference_calls"
        ),
    }
    if profile_id != "exp033_leakfree_research_v1" and any(
        result[field]
        for field in (
            "max_model_fits",
            "max_outer_oos_evaluations",
            "max_target_inference_calls",
        )
    ):
        raise ContractError("non-EXP033 profile cannot receive model/OOS/inference budget")
    if profile_id == "exp033_leakfree_research_v1" and (
        result["max_model_fits"] != 1
        or result["max_outer_oos_evaluations"] != 1
        or result["max_target_inference_calls"] != 1
    ):
        raise ContractError("EXP033 budget must freeze one fit, one outer OOS, and one inference")
    return result


def _output_contract(value: Any, output_root: str, profile_id: str) -> dict[str, Any]:
    obj = _exact_object(value, OUTPUT_CONTRACT_FIELDS, "output_sealing_contract")
    if obj["schema_version"] != RESULT_MANIFEST_SCHEMA_VERSION:
        raise ContractError("unknown result manifest schema version")
    if obj["mode"] != "append_only_immutable":
        raise ContractError("output sealing must be append-only and immutable")
    receipt_path = _repository_path(
        obj["execution_receipt_path"],
        "output_sealing_contract.execution_receipt_path",
    )
    result_path = _repository_path(
        obj["result_manifest_path"], "output_sealing_contract.result_manifest_path"
    )
    failure_path = _repository_path(
        obj["failure_manifest_path"], "output_sealing_contract.failure_manifest_path"
    )
    prefix = output_root + "/"
    if not all(
        path.startswith(prefix) for path in (receipt_path, result_path, failure_path)
    ):
        raise ContractError("receipt/result/failure manifest must be under the output root")
    if receipt_path != f"{output_root}/execution_receipt.json":
        raise ContractError("execution receipt path must use its deterministic name")
    if result_path != f"{output_root}/result.manifest.json":
        raise ContractError("success result manifest path must use its deterministic name")
    if failure_path != f"{output_root}/failure.manifest.json":
        raise ContractError("failure manifest path must use its deterministic name")
    if _bool(obj["overwrite_allowed"], "output_sealing_contract.overwrite_allowed") is not False:
        raise ContractError("output overwrite is forbidden")
    if _bool(
        obj["partial_output_consumer_eligible"],
        "output_sealing_contract.partial_output_consumer_eligible",
    ) is not False:
        raise ContractError("partial output cannot be consumer-eligible")
    if _bool(
        obj["failure_manifest_required"],
        "output_sealing_contract.failure_manifest_required",
    ) is not True:
        raise ContractError("crash/failure must produce an immutable failure manifest")
    roles = _sorted_unique_strings(
        obj["artifact_roles"], "output_sealing_contract.artifact_roles"
    )
    if tuple(roles) != tuple(sorted(ARTIFACT_ROLES_BY_PROFILE[profile_id])):
        raise ContractError("output artifact roles differ from the finite profile")
    artifact_paths_raw = obj["artifact_paths"]
    if not isinstance(artifact_paths_raw, list):
        raise ContractError("output artifact paths must be an array")
    artifact_paths: list[dict[str, str]] = []
    for index, item in enumerate(artifact_paths_raw):
        entry = _exact_object(
            item,
            ARTIFACT_PATH_FIELDS,
            f"output_sealing_contract.artifact_paths[{index}]",
        )
        artifact_paths.append(
            {
                "role": _identifier(
                    entry["role"],
                    f"output_sealing_contract.artifact_paths[{index}].role",
                ),
                "path": _repository_path(
                    entry["path"],
                    f"output_sealing_contract.artifact_paths[{index}].path",
                ),
            }
        )
    if [item["role"] for item in artifact_paths] != roles:
        raise ContractError("output artifact-path roles must equal the finite role set")
    paths = [item["path"] for item in artifact_paths]
    if (
        len(paths) != len(set(paths))
        or not all(path.startswith(prefix) for path in paths)
        or set(paths) & {receipt_path, result_path, failure_path}
    ):
        raise ContractError("output artifact paths must be unique exact files under the root")
    if any(PurePosixPath(path).suffix.casefold() in EXECUTABLE_INPUT_SUFFIXES for path in paths):
        raise ContractError("output artifact paths cannot create executable code")
    for item in artifact_paths:
        expected_suffix = {
            "canonical_json": ".json",
            "identity_jsonl": ".jsonl",
            "opaque_binary": ".bin",
        }[ARTIFACT_FORMAT_BY_ROLE[item["role"]]]
        if PurePosixPath(item["path"]).suffix.casefold() != expected_suffix:
            raise ContractError(
                "output artifact extension differs from its finite role schema"
            )
    fields = obj["required_result_fields"]
    if not isinstance(fields, list) or tuple(fields) != REQUIRED_RESULT_FIELDS:
        raise ContractError("required result manifest fields differ from the v3 contract")
    return {
        "schema_version": RESULT_MANIFEST_SCHEMA_VERSION,
        "mode": "append_only_immutable",
        "execution_receipt_path": receipt_path,
        "result_manifest_path": result_path,
        "failure_manifest_path": failure_path,
        "overwrite_allowed": False,
        "partial_output_consumer_eligible": False,
        "failure_manifest_required": True,
        "artifact_roles": roles,
        "artifact_paths": artifact_paths,
        "required_result_fields": list(REQUIRED_RESULT_FIELDS),
    }


def normalize_ordinary_real_data_run_scope(
    value: Any,
    *,
    proposal_scope: dict[str, Any],
) -> dict[str, Any]:
    """Normalize an exact v3 scope; unknown and extra fields fail closed."""

    obj = _exact_object(value, RUN_FIELDS, "ordinary v3 run scope")
    if obj["run_scope_schema_version"] != RUN_SCOPE_SCHEMA_VERSION:
        raise ContractError("unknown ordinary run_scope_schema_version")
    proposal = normalize_proposal_scope(
        proposal_scope,
        expected_experiment_id=proposal_scope.get("experiment_id"),
    )
    embedded = normalize_proposal_scope(
        obj["proposal_scope"], expected_experiment_id=proposal["experiment_id"]
    )
    if embedded != proposal:
        raise ContractError("run scope proposal_scope differs from the frozen proposal")
    proposal_digest = _hash(obj["proposal_scope_digest"], "proposal_scope_digest")
    if proposal_digest != canonical_digest(proposal):
        raise ContractError("proposal scope digest mismatch")
    execution_kind = _string(obj["execution_kind"], "execution_kind")
    if execution_kind not in {"synthetic", "real_data"}:
        raise ContractError("execution_kind must be synthetic or real_data")
    profile = validate_capability_profile(
        obj["capability_profile"],
        execution_kind=execution_kind,
        experiment_id=proposal["experiment_id"],
    )
    profile_id = profile["profile_id"]
    code_hashes = _ref_list(obj["code_hashes"], "code_hashes")
    config_hashes = _ref_list(obj["config_hashes"], "config_hashes")
    data_refs = _ref_list(obj["data_input_manifest_hashes"], "data_input_manifest_hashes")
    catalog = _normalize_input_catalog(obj["input_catalog"])
    expected_catalog_capabilities = sorted(
        name
        for name, enabled in profile["capabilities"].items()
        if enabled
        and name
        in {
            "read_real_input_manifests",
            "read_real_runner_rows",
            "canonicalize_input_release",
            "read_historical_training_rows",
            "read_sealed_canonical_input_release",
        }
    )
    if catalog["phase_read_capabilities"] != expected_catalog_capabilities:
        raise ContractError("input catalog phase capabilities differ from the finite profile")
    catalog_times = [
        datetime.fromisoformat(catalog[field][:-1] + "+00:00")
        for field in (
            "max_source_event_time",
            "max_received_at",
            "max_available_as_of",
        )
    ]
    if catalog_times != sorted(catalog_times):
        raise ContractError("input catalog source/receipt/availability maxima are inconsistent")
    if profile_id in {"exp034_input_canonicalization_v1", "exp033_leakfree_research_v1"}:
        if (
            catalog["race_count"] != 5
            or catalog["runner_count"] != 70
            or catalog["target_date"] != "2026-08-23"
        ):
            raise ContractError("EXP033/034 catalog must bind the exact 5-race/70-runner target")
    if profile_id == "synthetic_governance_v1" and (
        catalog["row_count"] != 0
        or catalog["race_count"] != 0
        or catalog["runner_count"] != 0
        or catalog["race_ids"]
        or catalog["row_blob_refs"]
    ):
        raise ContractError("synthetic governance catalog must remain row-empty")
    if catalog["manifest"] in data_refs:
        raise ContractError(
            "catalog payload is a separate metadata envelope, not an input manifest"
        )
    runner_ref = _ref(obj["runner_universe_manifest_hash"], "runner_universe_manifest_hash")
    training_ref = _ref(obj["training_source_manifest_hash"], "training_source_manifest_hash")
    target_ref = _ref(obj["target_source_manifest_hash"], "target_source_manifest_hash")
    for required_ref in (runner_ref, training_ref, target_ref):
        if required_ref not in data_refs:
            raise ContractError("required input manifest is absent from data_input_manifest_hashes")
    if runner_ref["sha256"] != catalog["runner_universe_digest"]:
        raise ContractError("catalog runner-universe digest differs from the bound manifest")
    feature_binding = _artifact_binding(
        obj["feature_input_release_hash"],
        "feature_input_release_hash",
        expected_role="canonical_target_input_release",
    )
    lineage_binding = _artifact_binding(
        obj["feature_lineage_manifest_hash"],
        "feature_lineage_manifest_hash",
        expected_role="canonical_feature_lineage",
    )
    if profile_id == "exp034_input_canonicalization_v1":
        if (
            feature_binding["binding_role"] != "planned_output_contract"
            or lineage_binding["binding_role"] != "planned_output_contract"
        ):
            raise ContractError("EXP034 must bind planned output contracts, not future hashes")
    if profile_id == "synthetic_governance_v1" and (
        feature_binding["binding_role"] != "planned_output_contract"
        or lineage_binding["binding_role"] != "planned_output_contract"
    ):
        raise ContractError("synthetic scope cannot consume sealed real-data artifacts")
    if profile_id == "exp033_leakfree_research_v1":
        if (
            feature_binding["binding_role"] != "sealed_input_artifact"
            or lineage_binding["binding_role"] != "sealed_input_artifact"
        ):
            raise ContractError("EXP033 must bind sealed EXP034 input artifacts")
        # Producer scope/receipt/result refs are metadata envelopes, not row
        # manifests.  Their exact hashes still participate in the run digest and
        # metadata catalog, while the producer result's artifact bytes are listed
        # separately as typed row blobs.
    environment = _environment(obj["environment"])
    dependency_names = {item["name"].casefold() for item in environment["dependency_versions"]}
    if not REQUIRED_DEPENDENCIES_BY_PROFILE[profile_id].issubset(dependency_names):
        raise ContractError("runtime environment omits a profile-required dependency")
    working_directory = _absolute_working_directory(
        obj["repository_working_directory"], "repository_working_directory"
    )
    commands = _commands(
        obj["exact_commands"],
        environment=environment,
        working_directory=working_directory,
    )
    code_paths = {item["path"] for item in code_hashes}
    if AUTHORITY_MODULE_PATH not in code_paths:
        raise ContractError("v3 authority/broker module is absent from exact code hashes")
    required_runner = REQUIRED_RUNNER_BY_PROFILE.get(profile_id)
    if required_runner is not None and required_runner not in code_paths:
        raise ContractError("profile runner is absent from exact code hashes")
    config_paths = {item["path"] for item in config_hashes}
    required_config = REQUIRED_CONFIG_BY_PROFILE.get(profile_id)
    if required_config is not None and required_config not in config_paths:
        raise ContractError("profile config is absent from exact config hashes")
    for command in commands:
        expected_runner = (
            (required_runner or command["argv"][3])
            if len(command["argv"]) > 3
            else ""
        )
        expected_config = (
            (required_config or command["argv"][7])
            if len(command["argv"]) > 7
            else ""
        )
        expected_argv = [
            command["executable"],
            "-I",
            "-B",
            expected_runner,
            "--phase",
            command["phase_id"],
            "--config",
            expected_config,
        ]
        if command["argv"] != expected_argv:
            raise ContractError(
                "exact argv must directly invoke the hash-bound runner with its "
                "finite phase and config; module/shell/unbound dispatch is forbidden"
            )
        if expected_runner not in code_paths or expected_config not in config_paths:
            raise ContractError("exact argv runner/config is not hash-bound")
    output_root = _repository_path(obj["output_root"], "output_root")
    expected_prefix = f"outputs/research/{proposal['experiment_id']}/"
    if not output_root.startswith(expected_prefix):
        raise ContractError("output root must be experiment-specific")
    read_allowlist = _read_allowlist(obj["read_allowlist"])
    write_allowlist = _write_allowlist(
        obj["write_allowlist"],
        output_root,
        allow_empty=profile_id == "synthetic_governance_v1",
    )
    row_paths = {
        item["path"]
        for item in read_allowlist
        if item["access_class"] != "metadata_manifest"
    }
    if row_paths & (code_paths | config_paths):
        raise ContractError("row inputs cannot overlap executable code or config paths")
    if {item["path"] for item in write_allowlist} & {
        *(item["path"] for item in read_allowlist),
        *code_paths,
        *config_paths,
    }:
        raise ContractError("output paths cannot overlap any frozen input/code/config path")
    phase_plan = _phases(
        obj["phase_plan"],
        profile_id=profile_id,
        commands=commands,
        read_allowlist=read_allowlist,
        write_allowlist=write_allowlist,
    )
    allowed_phase_ids = {item["phase_id"] for item in phase_plan}
    for entry in [*read_allowlist, *write_allowlist]:
        if not set(entry["phases"]).issubset(allowed_phase_ids):
            raise ContractError("access allowlist cites an unknown phase")
    metadata_refs = [
        *data_refs,
        catalog["manifest"],
        runner_ref,
        training_ref,
        target_ref,
        _ref(obj["label_eligibility_contract_hash"], "label_eligibility_contract_hash"),
        _ref(obj["fold_manifest_hash"], "fold_manifest_hash"),
        _ref(
            obj["dependency_environment_lock_hash"],
            "dependency_environment_lock_hash",
        ),
        feature_binding["contract"],
        lineage_binding["contract"],
    ]
    committed_contract_refs = [
        metadata_refs[-5],
        metadata_refs[-4],
        metadata_refs[-3],
        feature_binding["contract"],
        lineage_binding["contract"],
    ]
    for ref in committed_contract_refs:
        if (
            not ref["path"].startswith("research/")
            or PurePosixPath(ref["path"]).suffix.casefold() != ".json"
        ):
            raise ContractError(
                "committed control-plane contracts must be exact research JSON files"
            )
    for binding in (feature_binding, lineage_binding):
        for ref_field in (
            "producer_run_scope",
            "producer_execution_receipt",
            "producer_output_attestation",
            "artifact_manifest",
        ):
            if binding[ref_field] is not None:
                metadata_refs.append(binding[ref_field])
    metadata_by_path: dict[str, str] = {}
    for ref in metadata_refs:
        prior = metadata_by_path.setdefault(ref["path"], ref["sha256"])
        if prior != ref["sha256"]:
            raise ContractError("one metadata path is bound to conflicting hashes")
    read_by_path = {item["path"]: item for item in read_allowlist}
    for path, digest in metadata_by_path.items():
        entry = read_by_path.get(path)
        expected_metadata_capability = (
            None
            if profile_id == "synthetic_governance_v1"
            else "read_real_input_manifests"
        )
        if (
            entry is None
            or entry["access_class"] != "metadata_manifest"
            or entry["required_capability"] != expected_metadata_capability
            or entry["sha256"] != digest
        ):
            raise ContractError(
                "every hash-bound metadata ref must be an exact metadata read allowlist entry"
            )
    catalog_metadata_expected = [
        read_by_path[path]
        for path in sorted(metadata_by_path)
        if path != catalog["manifest"]["path"]
    ]
    catalog_rows_expected = [
        entry
        for entry in read_allowlist
        if entry["access_class"] != "metadata_manifest"
    ]
    if catalog["metadata_manifest_refs"] != catalog_metadata_expected:
        raise ContractError(
            "input catalog metadata inventory differs from the exact bound manifests"
        )
    if catalog["row_blob_refs"] != catalog_rows_expected:
        raise ContractError(
            "input catalog row inventory differs from the exact typed read allowlist"
        )
    if execution_kind == "real_data" and not catalog_rows_expected:
        raise ContractError("real-data scope must bind at least one exact row blob")
    metadata_phase = next(
        (item for item in phase_plan if item["phase_id"] == "metadata_preflight"),
        None,
    )
    if metadata_phase is None or metadata_phase["read_paths"] != sorted(metadata_by_path):
        raise ContractError("metadata_preflight must read the exact metadata-ref set")
    used_read_paths = {
        path for phase in phase_plan for path in phase["read_paths"]
    }
    used_write_paths = {
        path for phase in phase_plan for path in phase["write_paths"]
    }
    if used_read_paths != set(read_by_path):
        raise ContractError("every read allowlist path must belong to an exact phase")
    if used_write_paths != {item["path"] for item in write_allowlist}:
        raise ContractError("every write allowlist path must belong to an exact phase")
    for entry in read_allowlist:
        actual_phases = sorted(
            phase["phase_id"]
            for phase in phase_plan
            if entry["path"] in phase["read_paths"]
        )
        if entry["phases"] != actual_phases:
            raise ContractError("read allowlist phases differ from actual phase use")
    for entry in write_allowlist:
        actual_phases = sorted(
            phase["phase_id"]
            for phase in phase_plan
            if entry["path"] in phase["write_paths"]
        )
        if entry["phases"] != actual_phases:
            raise ContractError("write allowlist phases differ from actual phase use")
    network = _exact_object(obj["network_policy"], ("mode", "allowed_hosts"), "network_policy")
    if network["mode"] != "disabled" or network["allowed_hosts"] != []:
        raise ContractError("ordinary v3 network access must remain disabled")
    source_cutoff = _canonical_timestamp(obj["source_cutoff"], "source_cutoff")
    as_of = _canonical_timestamp(obj["as_of"], "as_of")
    if datetime.fromisoformat(source_cutoff[:-1] + "+00:00") > datetime.fromisoformat(
        as_of[:-1] + "+00:00"
    ):
        raise ContractError("source_cutoff must not be later than as_of")
    cutoff_time = datetime.fromisoformat(source_cutoff[:-1] + "+00:00")
    for catalog_field in (
        "max_source_event_time",
        "max_received_at",
        "max_available_as_of",
    ):
        observed_time = datetime.fromisoformat(catalog[catalog_field][:-1] + "+00:00")
        if observed_time > cutoff_time:
            raise ContractError("input catalog source-time metadata exceeds source_cutoff")
    budget = _compute_budget(obj["compute_budget"], profile_id)
    if any(command["timeout_seconds"] > budget["timeout_seconds"] for command in commands):
        raise ContractError("exact command timeout exceeds the frozen compute budget")
    if sum(command["timeout_seconds"] for command in commands) > budget["timeout_seconds"]:
        raise ContractError("ordered command timeouts exceed the total compute budget")
    output_contract = _output_contract(
        obj["output_sealing_contract"], output_root, profile_id
    )
    write_by_path = {item["path"]: item for item in write_allowlist}
    manifest_paths = {
        output_contract["result_manifest_path"],
        output_contract["failure_manifest_path"],
    }
    artifact_paths = {item["path"] for item in output_contract["artifact_paths"]}
    if profile_id == "synthetic_governance_v1":
        if write_by_path or artifact_paths:
            raise ContractError("synthetic governance scopes cannot declare execution outputs")
    else:
        expected_write_paths = manifest_paths | artifact_paths
        if set(write_by_path) != expected_write_paths:
            raise ContractError(
                "real-data write allowlist must equal the exact result/failure/artifact paths"
            )
        execution_phase_ids = [
            phase["phase_id"]
            for phase in phase_plan
            if phase["phase_id"] not in {"metadata_preflight", "seal_research_outputs"}
        ]
        if len(execution_phase_ids) != 1:
            raise ContractError("real-data profile must have one finite execution phase")
        for path in manifest_paths:
            if write_by_path[path]["phases"] != ["seal_research_outputs"]:
                raise ContractError("result/failure manifest must be seal-phase-only")
        for path in artifact_paths:
            if write_by_path[path]["phases"] != execution_phase_ids:
                raise ContractError("artifact output must be written only by the finite execution phase")
    normalized = {
        "run_scope_schema_version": RUN_SCOPE_SCHEMA_VERSION,
        "proposal_scope": proposal,
        "proposal_scope_digest": proposal_digest,
        "execution_kind": execution_kind,
        "capability_profile": profile,
        "execution_commit_sha": _git_sha(obj["execution_commit_sha"], "execution_commit_sha"),
        "code_hashes": code_hashes,
        "config_hashes": config_hashes,
        "data_input_manifest_hashes": data_refs,
        "input_catalog": catalog,
        "runner_universe_manifest_hash": runner_ref,
        "feature_input_release_hash": feature_binding,
        "training_source_manifest_hash": training_ref,
        "target_source_manifest_hash": target_ref,
        "feature_lineage_manifest_hash": lineage_binding,
        "label_eligibility_contract_hash": _ref(
            obj["label_eligibility_contract_hash"], "label_eligibility_contract_hash"
        ),
        "fold_manifest_hash": _ref(obj["fold_manifest_hash"], "fold_manifest_hash"),
        "dependency_environment_lock_hash": _ref(
            obj["dependency_environment_lock_hash"], "dependency_environment_lock_hash"
        ),
        "environment": environment,
        "random_seed": _integer(obj["random_seed"], "random_seed"),
        "repository_working_directory": working_directory,
        "exact_commands": commands,
        "network_policy": {"mode": "disabled", "allowed_hosts": []},
        "read_allowlist": read_allowlist,
        "write_allowlist": write_allowlist,
        "output_root": output_root,
        "compute_budget": budget,
        "source_cutoff": source_cutoff,
        "as_of": as_of,
        "phase_plan": phase_plan,
        "output_sealing_contract": output_contract,
        "formal_buy": _bool(obj["formal_buy"], "formal_buy"),
        "send_order": _bool(obj["send_order"], "send_order"),
        "stake": _integer(obj["stake"], "stake"),
    }
    if normalized["formal_buy"] or normalized["send_order"] or normalized["stake"] != 0:
        raise ContractError("formal_buy/send_order/stake must remain false/false/0")
    if tuple(normalized) != RUN_FIELDS:
        raise AssertionError("internal v3 run field order mismatch")
    return normalized


def dispatch_ordinary_run_scope(
    value: Any,
    *,
    proposal_scope: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Explicitly dispatch absent marker to v2 and exact marker to v3."""

    if not isinstance(value, dict):
        raise ContractError("ordinary run scope must be an object")
    if "run_scope_schema_version" not in value:
        return "legacy_v2", normalize_run_scope(value, proposal_scope=proposal_scope)
    if value.get("run_scope_schema_version") != RUN_SCOPE_SCHEMA_VERSION:
        raise ContractError("unknown ordinary run_scope_schema_version; fail-close")
    return RUN_SCOPE_SCHEMA_VERSION, normalize_ordinary_real_data_run_scope(
        value, proposal_scope=proposal_scope
    )


def load_frozen_ordinary_run_scope(
    root: Path,
    path: Path,
    proposal_scope: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    raw = strict_json_load(path)
    version, scope = dispatch_ordinary_run_scope(raw, proposal_scope=proposal_scope)
    expected_bytes = canonical_json_bytes(scope) + b"\n"
    if version == RUN_SCOPE_SCHEMA_VERSION and path.read_bytes() != expected_bytes:
        raise ContractError("v3 run scope bytes are not canonical UTF-8/LF JSON")
    return version, scope, canonical_digest(scope)


def _resolve_repository_path(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    lexical = PurePosixPath(relative)
    current = resolved_root
    for part in lexical.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ContractError(f"linked repository path is forbidden: {relative}")
    path = root / relative
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"path escapes repository root: {relative}") from exc
    return resolved


def _sha256_file(path: Path, label: str) -> str:
    if not path.is_file():
        raise ContractError(f"{label} metadata file is missing")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_bytes(root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{path}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ContractError(f"execution commit does not contain hash-bound material: {path}")
    return completed.stdout


def _current_git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    observed = completed.stdout.strip().lower()
    if completed.returncode != 0 or not FULL_GIT_SHA.fullmatch(observed):
        raise ContractError("cannot observe the runtime execution commit")
    return observed


def _git_worktree_changes(root: Path) -> set[str]:
    """Return every tracked/untracked path reported by Git, without shell parsing."""

    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ContractError("cannot inspect the runtime worktree state")
    records = completed.stdout.split(b"\0")
    paths: set[str] = set()
    for record in records:
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise ContractError("runtime worktree status is malformed or contains a rename")
        status = record[:2].decode("ascii", errors="strict")
        if "R" in status or "C" in status:
            raise ContractError("runtime worktree rename/copy state is forbidden")
        try:
            path = record[3:].decode("utf-8", errors="strict").replace("\\", "/")
        except UnicodeDecodeError as exc:
            raise ContractError("runtime worktree contains a non-UTF-8 path") from exc
        paths.add(_repository_path(path, "runtime worktree path"))
    return paths


def _git_ignored_executable_paths(root: Path) -> set[str]:
    patterns = [f"*{suffix}" for suffix in sorted(EXECUTABLE_INPUT_SUFFIXES)]
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            *patterns,
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ContractError("cannot inspect ignored executable/importable paths")
    paths: set[str] = set()
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            path = raw.decode("utf-8", errors="strict").replace("\\", "/")
        except UnicodeDecodeError as exc:
            raise ContractError("ignored executable path is not UTF-8") from exc
        paths.add(_repository_path(path, "ignored executable path"))
    return paths


def verify_execution_worktree_state(root: Path, scope: dict[str, Any]) -> None:
    """Reject executable/import drift while allowing only exact bound I/O paths."""

    allowed = {
        *(item["path"] for item in scope["read_allowlist"]),
        *(item["path"] for item in scope["write_allowlist"]),
        scope["output_sealing_contract"]["execution_receipt_path"],
    }
    unexpected = sorted(_git_worktree_changes(root.resolve()) - allowed)
    if unexpected:
        raise ContractError(
            "runtime worktree has dirty/untracked paths outside the exact bound I/O set: "
            + ", ".join(unexpected)
        )
    ignored_executables = sorted(_git_ignored_executable_paths(root.resolve()))
    if ignored_executables:
        raise ContractError(
            "runtime worktree contains ignored executable/importable paths: "
            + ", ".join(ignored_executables)
        )


def observe_runtime_environment(scope: dict[str, Any]) -> dict[str, Any]:
    """Measure the interpreter/dependency/locale/timezone binding in this process."""

    installed: dict[str, str] = {"python": platform.python_version()}
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name:
            raise ContractError("installed distribution lacks a canonical name")
        version = distribution.version
        if not isinstance(version, str) or not version:
            raise ContractError(f"installed distribution lacks a version: {name}")
        prior = installed.setdefault(name, version)
        if prior != version:
            raise ContractError(f"duplicate installed distribution versions: {name}")
    dependencies = [
        {"name": name, "version": installed[name]}
        for name in sorted(installed)
    ]
    timezone_name = os.environ.get("TZ") or str(datetime.now().astimezone().tzinfo)
    observed = {
        "interpreter_path": Path(sys.executable).resolve().as_posix(),
        "interpreter_version": platform.python_version(),
        "dependency_versions": dependencies,
        "locale": runtime_locale.setlocale(runtime_locale.LC_ALL, None),
        "timezone": timezone_name,
    }
    return _environment(observed)


def observe_process_argv() -> list[str]:
    """Observe the actual interpreter argv, including interpreter flags."""

    observed = list(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
    if not observed:
        raise ContractError("cannot observe runtime argv")
    observed[0] = Path(sys.executable).resolve().as_posix()
    if not all(isinstance(item, str) and item for item in observed):
        raise ContractError("runtime argv contains a noncanonical value")
    return observed


def verify_runtime_interpreter_isolation() -> None:
    if (
        getattr(sys.flags, "isolated", 0) != 1
        or getattr(sys.flags, "safe_path", False) is not True
        or os.environ.get("PYTHONPATH")
        or os.environ.get("PYTHONHOME")
    ):
        raise ContractError(
            "runtime interpreter must use -I with no PYTHONPATH/PYTHONHOME injection"
        )


def _all_metadata_refs(scope: dict[str, Any]) -> list[dict[str, str]]:
    refs = [
        *scope["data_input_manifest_hashes"],
        scope["input_catalog"]["manifest"],
        scope["runner_universe_manifest_hash"],
        scope["training_source_manifest_hash"],
        scope["target_source_manifest_hash"],
        scope["label_eligibility_contract_hash"],
        scope["fold_manifest_hash"],
        scope["dependency_environment_lock_hash"],
        scope["feature_input_release_hash"]["contract"],
        scope["feature_lineage_manifest_hash"]["contract"],
    ]
    for field in ("feature_input_release_hash", "feature_lineage_manifest_hash"):
        for ref_field in (
            "producer_run_scope",
            "producer_execution_receipt",
            "producer_output_attestation",
            "artifact_manifest",
        ):
            ref = scope[field][ref_field]
            if ref is not None:
                refs.append(ref)
    unique = {(item["path"], item["sha256"]): item for item in refs}
    return [unique[key] for key in sorted(unique)]


def _strict_canonical_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError(f"non-standard JSON constant is forbidden: {token}")
            ),
            object_pairs_hook=_strict_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or payload != canonical_json_bytes(value) + b"\n":
        raise ContractError(f"{label} bytes are not canonical UTF-8/LF JSON")
    return value


def normalize_output_attestation(
    value: Any,
    *,
    producer_scope: dict[str, Any],
    producer_scope_path: str,
    producer_scope_digest: str,
    receipt: dict[str, Any],
    receipt_path: str,
    receipt_sha256: str,
    result: dict[str, Any],
    result_path: str,
    result_sha256: str,
) -> dict[str, Any]:
    """Normalize the human-merged post-run anchor for an EXP034 output seal."""

    obj = _exact_object(value, OUTPUT_ATTESTATION_FIELDS, "output attestation")
    normalized = {
        "attestation_schema_version": _string(
            obj["attestation_schema_version"],
            "output_attestation.attestation_schema_version",
        ),
        "producer_experiment_id": _string(
            obj["producer_experiment_id"],
            "output_attestation.producer_experiment_id",
        ),
        "producer_run_scope_path": _repository_path(
            obj["producer_run_scope_path"],
            "output_attestation.producer_run_scope_path",
        ),
        "producer_run_scope_digest": _hash(
            obj["producer_run_scope_digest"],
            "output_attestation.producer_run_scope_digest",
        ),
        "running_event_id": _identifier(
            obj["running_event_id"], "output_attestation.running_event_id"
        ),
        "execution_receipt_path": _repository_path(
            obj["execution_receipt_path"],
            "output_attestation.execution_receipt_path",
        ),
        "execution_receipt_sha256": _hash(
            obj["execution_receipt_sha256"],
            "output_attestation.execution_receipt_sha256",
        ),
        "execution_receipt_digest": _hash(
            obj["execution_receipt_digest"],
            "output_attestation.execution_receipt_digest",
        ),
        "result_manifest_path": _repository_path(
            obj["result_manifest_path"],
            "output_attestation.result_manifest_path",
        ),
        "result_manifest_sha256": _hash(
            obj["result_manifest_sha256"],
            "output_attestation.result_manifest_sha256",
        ),
        "result_manifest_digest": _hash(
            obj["result_manifest_digest"],
            "output_attestation.result_manifest_digest",
        ),
        "artifacts_digest": _hash(
            obj["artifacts_digest"], "output_attestation.artifacts_digest"
        ),
        "output_root": _repository_path(
            obj["output_root"], "output_attestation.output_root"
        ),
        "status": _string(obj["status"], "output_attestation.status"),
        "consumer_eligible": _bool(
            obj["consumer_eligible"], "output_attestation.consumer_eligible"
        ),
        "attested_at": _canonical_timestamp(
            obj["attested_at"], "output_attestation.attested_at"
        ),
        "formal_buy": _bool(obj["formal_buy"], "output_attestation.formal_buy"),
        "send_order": _bool(obj["send_order"], "output_attestation.send_order"),
        "stake": _integer(obj["stake"], "output_attestation.stake"),
    }
    expected = {
        "attestation_schema_version": OUTPUT_ATTESTATION_SCHEMA_VERSION,
        "producer_experiment_id": producer_scope["proposal_scope"]["experiment_id"],
        "producer_run_scope_path": producer_scope_path,
        "producer_run_scope_digest": producer_scope_digest,
        "running_event_id": receipt["running_event_id"],
        "execution_receipt_path": receipt_path,
        "execution_receipt_sha256": receipt_sha256,
        "execution_receipt_digest": canonical_digest(receipt),
        "result_manifest_path": result_path,
        "result_manifest_sha256": result_sha256,
        "result_manifest_digest": canonical_digest(result),
        "artifacts_digest": canonical_digest(result["artifacts"]),
        "output_root": producer_scope["output_root"],
        "status": "success",
        "consumer_eligible": True,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    for field, expected_value in expected.items():
        if normalized[field] != expected_value:
            raise ContractError(f"output attestation differs from sealed producer: {field}")
    if datetime.fromisoformat(
        normalized["attested_at"][:-1] + "+00:00"
    ) < datetime.fromisoformat(result["generated_at"][:-1] + "+00:00"):
        raise ContractError("output attestation predates the sealed result")
    return normalized


def _verify_exp034_sealed_connectors(root: Path, scope: dict[str, Any]) -> None:
    if scope["capability_profile"]["profile_id"] != "exp033_leakfree_research_v1":
        return
    feature = scope["feature_input_release_hash"]
    lineage = scope["feature_lineage_manifest_hash"]
    for ref_field in (
        "producer_run_scope",
        "producer_execution_receipt",
        "producer_output_attestation",
        "artifact_manifest",
    ):
        if feature[ref_field] != lineage[ref_field]:
            raise ContractError("EXP033 feature/lineage inputs must share one sealed EXP034 run")
    producer_scope_ref = feature["producer_run_scope"]
    receipt_ref = feature["producer_execution_receipt"]
    attestation_ref = feature["producer_output_attestation"]
    result_ref = feature["artifact_manifest"]
    producer_scope_value = _strict_canonical_json_bytes(
        _resolve_repository_path(root, producer_scope_ref["path"]).read_bytes(),
        "EXP034 producer run scope",
    )
    producer_version, producer_scope = dispatch_ordinary_run_scope(
        producer_scope_value,
        proposal_scope=producer_scope_value.get("proposal_scope"),
    )
    if (
        producer_version != RUN_SCOPE_SCHEMA_VERSION
        or producer_scope["proposal_scope"]["experiment_id"] != "EXP-20260821-034"
        or producer_scope["capability_profile"]["profile_id"]
        != "exp034_input_canonicalization_v1"
        or producer_scope["execution_kind"] != "real_data"
    ):
        raise ContractError("sealed connector producer is not the exact EXP034 v3 profile")
    producer_digest = canonical_digest(producer_scope)
    receipt_value = _strict_canonical_json_bytes(
        _resolve_repository_path(root, receipt_ref["path"]).read_bytes(),
        "EXP034 producer execution receipt",
    )
    receipt = normalize_execution_receipt(
        receipt_value,
        run_scope=producer_scope,
        run_scope_digest=producer_digest,
    )
    result_value = _strict_canonical_json_bytes(
        _resolve_repository_path(root, result_ref["path"]).read_bytes(),
        "EXP034 result manifest",
    )
    result = normalize_result_manifest(
        result_value,
        run_scope=producer_scope,
        run_scope_digest=producer_digest,
    )
    if (
        result_ref["path"]
        != producer_scope["output_sealing_contract"]["result_manifest_path"]
        or receipt_ref["path"]
        != producer_scope["output_sealing_contract"]["execution_receipt_path"]
        or result["status"] != "success"
        or result["consumer_eligible"] is not True
        or result["execution_receipt_digest"] != canonical_digest(receipt)
    ):
        raise ContractError("EXP034 producer seal/receipt chain is incomplete")
    provider: GitHubApprovalProvider = GitHubRestApprovalProvider()
    _, producer_trust = verify_github_trust(
        provider=provider,
        repository=DEFAULT_REPOSITORY,
        base_branch=DEFAULT_BASE_BRANCH,
        base_commit=producer_scope["proposal_scope"]["base_commit"],
    )
    producer_live_main = producer_trust["verified_current_main_sha"]
    expected_attestation_path = (
        "research/attestations/EXP-20260821-034/"
        f"{producer_digest}.output_attestation.json"
    )
    if attestation_ref["path"] != expected_attestation_path:
        raise ContractError("EXP034 output attestation path is not digest-addressed")
    remote_attestation_bytes, _ = fetch_github_file_at_commit(
        provider=provider,
        repository=DEFAULT_REPOSITORY,
        path=attestation_ref["path"],
        ref=producer_live_main,
    )
    if hashlib.sha256(remote_attestation_bytes).hexdigest() != attestation_ref["sha256"]:
        raise ContractError("EXP034 output attestation is not exact on GitHub main")
    normalize_output_attestation(
        _strict_canonical_json_bytes(
            remote_attestation_bytes, "EXP034 GitHub output attestation"
        ),
        producer_scope=producer_scope,
        producer_scope_path=producer_scope_ref["path"],
        producer_scope_digest=producer_digest,
        receipt=receipt,
        receipt_path=receipt_ref["path"],
        receipt_sha256=receipt_ref["sha256"],
        result=result,
        result_path=result_ref["path"],
        result_sha256=result_ref["sha256"],
    )
    producer_registry_bytes, _ = fetch_github_file_at_commit(
        provider=provider,
        repository=DEFAULT_REPOSITORY,
        path="research/REGISTRY.jsonl",
        ref=producer_live_main,
    )
    try:
        producer_events = [
            json.loads(
                line.decode("utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ContractError(f"non-standard JSON constant is forbidden: {token}")
                ),
                object_pairs_hook=_strict_pairs,
            )
            for line in producer_registry_bytes.splitlines()
            if line.strip()
        ]
        producer_chains, _ = _validate_remote_registry_history(producer_events)
    except (UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
        raise ContractError("live producer Registry is invalid") from exc
    producer_running_matches = [
        event
        for event in producer_events
        if event.get("event_id") == receipt["running_event_id"]
    ]
    if len(producer_running_matches) != 1:
        raise ContractError("producer receipt event is absent/duplicated on GitHub main")
    producer_running = producer_running_matches[0]
    producer_chain = producer_chains.get("EXP-20260821-034", [])
    if (
        producer_running.get("status") != "running"
        or producer_running.get("run_scope_digest") != producer_digest
        or producer_running.get("execution_kind") != "real-data"
        or not producer_chain
        or producer_chain[-1].get("status") in {"invalid", "rejected"}
    ):
        raise ContractError("producer lifecycle is not an accepted real-data run")
    producer_revalidations = producer_running.get("revalidated_approval_evidence")
    if not isinstance(producer_revalidations, list):
        raise ContractError("producer RUNNING event lacks approval revalidations")
    producer_prepare = next(
        (
            evidence
            for evidence in producer_revalidations
            if evidence.get("approval_type") == "APPROVED_TO_PREPARE"
        ),
        None,
    )
    producer_run = next(
        (
            evidence
            for evidence in producer_revalidations
            if evidence.get("approval_type") == "APPROVED_TO_RUN"
        ),
        None,
    )
    if not isinstance(producer_prepare, dict) or not isinstance(producer_run, dict):
        raise ContractError("producer Prepare/Run evidence is incomplete")
    _verify_live_github_execution_authority(
        provider=provider,
        run_scope=producer_scope,
        execution_commit=producer_scope["execution_commit_sha"],
        current_main_sha=producer_live_main,
        prepare_evidence=producer_prepare,
        run_evidence=producer_run,
        current_main_registry_bytes=producer_registry_bytes,
        merged_running_event=producer_running,
    )
    receipt_main = receipt["verified_current_main_sha"]
    receipt_registry_bytes, _ = fetch_github_file_at_commit(
        provider=provider,
        repository=DEFAULT_REPOSITORY,
        path="research/REGISTRY.jsonl",
        ref=receipt_main,
    )
    if hashlib.sha256(receipt_registry_bytes).hexdigest() != receipt[
        "current_main_registry_sha256"
    ]:
        raise ContractError("producer receipt Registry anchor is not on GitHub")
    try:
        receipt_events = [
            json.loads(
                line.decode("utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ContractError(f"non-standard JSON constant is forbidden: {token}")
                ),
                object_pairs_hook=_strict_pairs,
            )
            for line in receipt_registry_bytes.splitlines()
            if line.strip()
        ]
        receipt_chains, _ = _validate_remote_registry_history(receipt_events)
    except (UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
        raise ContractError("producer receipt Registry anchor is invalid") from exc
    if receipt_chains.get("EXP-20260821-034", [])[-1:] != [producer_running]:
        raise ContractError("producer RUNNING event was not the receipt-time chain head")
    try:
        receipt_comparison = provider.compare_commits(
            DEFAULT_REPOSITORY, receipt_main, producer_live_main
        )
    except Exception as exc:
        raise ContractError("producer receipt-main ancestry is unavailable") from exc
    receipt_compare_url = (
        f"{GITHUB_API_URL}/repos/{DEFAULT_REPOSITORY}/compare/"
        f"{receipt_main}...{producer_live_main}"
    )
    if (
        not isinstance(receipt_comparison, dict)
        or receipt_comparison.get("status") not in {"ahead", "identical"}
        or not isinstance(receipt_comparison.get("base_commit"), dict)
        or receipt_comparison["base_commit"].get("sha") != receipt_main
        or not isinstance(receipt_comparison.get("merge_base_commit"), dict)
        or receipt_comparison["merge_base_commit"].get("sha") != receipt_main
        or receipt_comparison.get("url") != receipt_compare_url
    ):
        raise ContractError("producer receipt main is not an ancestor of live main")
    artifacts = {item["role"]: item for item in result["artifacts"]}
    _verify_exact_output_root_contents(
        root,
        producer_scope,
        expected_file_paths={
            receipt_ref["path"],
            result_ref["path"],
            *(item["path"] for item in artifacts.values()),
        },
    )
    read_entries = {item["path"]: item for item in scope["read_allowlist"]}
    required_access_classes = {
        "canonical_runner_universe": "runner_row_blob",
        "canonical_training_input_release": "historical_training_row_blob",
        "canonical_target_input_release": "sealed_input_row_blob",
        "canonical_feature_lineage": "sealed_input_row_blob",
        "label_eligibility_reconciliation": "sealed_input_row_blob",
    }
    if set(artifacts) != set(required_access_classes):
        raise ContractError("EXP034 sealed result has an incomplete artifact-role set")
    for role, expected_access_class in required_access_classes.items():
        artifact = artifacts[role]
        entry = read_entries.get(artifact["path"])
        if (
            entry is None
            or entry["access_class"] != expected_access_class
            or entry["sha256"] != artifact["sha256"]
        ):
            raise ContractError(
                "every EXP034 sealed artifact must have an exact typed EXP033 row binding"
            )
    for binding in (feature, lineage):
        artifact = artifacts.get(binding["artifact_role"])
        if artifact is None or artifact["sha256"] != binding["artifact_sha256"]:
            raise ContractError("sealed connector artifact hash differs from EXP034 result")
        entry = read_entries.get(artifact["path"])
        if (
            entry is None
            or entry["access_class"] != "sealed_input_row_blob"
            or entry["sha256"] != artifact["sha256"]
        ):
            raise ContractError("sealed EXP034 artifact is absent from EXP033 row access")


def verify_ordinary_real_data_run_materials(
    root: Path,
    scope: dict[str, Any],
    *,
    catalog_bytes: bytes | None = None,
    output_root_exists: Callable[[Path], bool] | None = None,
) -> dict[str, Any]:
    """Verify code and metadata only; never open an allowlisted row/blob."""

    root = root.resolve()
    if scope["repository_working_directory"] != root.as_posix():
        raise ContractError("frozen repository working directory differs from verifier root")
    commit = scope["execution_commit_sha"]
    for field in ("code_hashes", "config_hashes"):
        for ref in scope[field]:
            observed = hashlib.sha256(_git_blob_bytes(root, commit, ref["path"])).hexdigest()
            if observed != ref["sha256"]:
                raise ContractError(f"{field} differs from the execution-commit blob")
            current_path = _resolve_repository_path(root, ref["path"])
            if _sha256_file(current_path, ref["path"]) != ref["sha256"]:
                raise ContractError(f"{field} worktree bytes differ from execution commit")
    static_refs = [
        scope["label_eligibility_contract_hash"],
        scope["fold_manifest_hash"],
        scope["dependency_environment_lock_hash"],
        scope["feature_input_release_hash"]["contract"],
        scope["feature_lineage_manifest_hash"]["contract"],
    ]
    for ref in static_refs:
        observed = hashlib.sha256(_git_blob_bytes(root, commit, ref["path"])).hexdigest()
        if observed != ref["sha256"]:
            raise ContractError("static v3 contract differs from the execution-commit blob")
        _strict_canonical_json_bytes(
            _resolve_repository_path(root, ref["path"]).read_bytes(),
            f"static control-plane contract {ref['path']}",
        )
    for ref in _all_metadata_refs(scope):
        path = _resolve_repository_path(root, ref["path"])
        if _sha256_file(path, ref["path"]) != ref["sha256"]:
            raise ContractError(f"hash-bound metadata changed: {ref['path']}")
    input_manifests: dict[str, dict[str, Any]] = {}
    row_inventory: dict[str, dict[str, Any]] = {}
    for index, ref in enumerate(scope["data_input_manifest_hashes"]):
        path = _resolve_repository_path(root, ref["path"])
        manifest = _normalize_input_manifest(
            _strict_canonical_json_bytes(
                path.read_bytes(), f"data input manifest[{index}]"
            ),
            f"data input manifest[{index}]",
        )
        input_manifests[ref["path"]] = manifest
        for row_ref in manifest["row_blob_refs"]:
            prior = row_inventory.setdefault(row_ref["path"], row_ref)
            if prior != row_ref:
                raise ContractError(
                    "one row blob has conflicting manifest hash/capability bindings"
                )
            if prior is not row_ref:
                raise ContractError("one row blob is assigned to multiple input manifests")
    if sorted(row_inventory.values(), key=lambda item: item["path"]) != scope[
        "input_catalog"
    ]["row_blob_refs"]:
        raise ContractError(
            "input manifests do not enumerate the catalog's exact typed row inventory"
        )
    if sum(item["row_count"] for item in input_manifests.values()) != scope[
        "input_catalog"
    ]["row_count"]:
        raise ContractError("catalog row count differs from its exact input manifests")
    for field in (
        "max_source_event_time",
        "max_received_at",
        "max_available_as_of",
    ):
        observed_max = max(
            (item[field] for item in input_manifests.values()),
            key=lambda value: datetime.fromisoformat(value[:-1] + "+00:00"),
        )
        if observed_max != scope["input_catalog"][field]:
            raise ContractError("catalog source-time maximum differs from its manifests")
    synthetic_profile = (
        scope["capability_profile"]["profile_id"] == "synthetic_governance_v1"
    )
    named_manifest_kinds = (
        (
            scope["runner_universe_manifest_hash"],
            "synthetic_fixture" if synthetic_profile else "runner_universe",
        ),
        (
            scope["training_source_manifest_hash"],
            "synthetic_fixture" if synthetic_profile else "training_source",
        ),
        (
            scope["target_source_manifest_hash"],
            "synthetic_fixture" if synthetic_profile else "target_source",
        ),
    )
    for ref, expected_kind in named_manifest_kinds:
        manifest = input_manifests.get(ref["path"])
        if manifest is None or manifest["manifest_kind"] != expected_kind:
            raise ContractError("named input manifest has the wrong finite manifest kind")
    if synthetic_profile:
        empty_identity_digest = canonical_digest([])
        if (
            scope["input_catalog"]["runner_identity_digest"]
            != empty_identity_digest
            or any(
                item["identity_set_sha256"] != empty_identity_digest
                for item in input_manifests.values()
            )
        ):
            raise ContractError("synthetic input manifests must bind the empty identity set")
    for field_name in (() if synthetic_profile else (
        "runner_universe_manifest_hash",
        "target_source_manifest_hash",
    )):
        manifest = input_manifests[scope[field_name]["path"]]
        if (
            manifest["race_count"] != scope["input_catalog"]["race_count"]
            or manifest["runner_count"] != scope["input_catalog"]["runner_count"]
            or manifest["identity_set_sha256"]
            != scope["input_catalog"]["runner_identity_digest"]
        ):
            raise ContractError(
                "target/runner input manifest identity/count differs from catalog"
            )
    environment_lock_path = _resolve_repository_path(
        root, scope["dependency_environment_lock_hash"]["path"]
    )
    environment_lock = _normalize_environment_lock(
        _strict_canonical_json_bytes(
            environment_lock_path.read_bytes(), "dependency environment lock"
        )
    )
    if environment_lock["environment"] != scope["environment"]:
        raise ContractError(
            "dependency environment lock differs from the frozen runtime environment"
        )
    _verify_exp034_sealed_connectors(root, scope)
    catalog_ref = scope["input_catalog"]["manifest"]
    if catalog_bytes is None:
        catalog_path = _resolve_repository_path(root, catalog_ref["path"])
        catalog_bytes = catalog_path.read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != catalog_ref["sha256"]:
        raise ContractError("input catalog bytes differ from the frozen digest")
    try:
        parsed_catalog = json.loads(
            catalog_bytes.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError(f"non-standard JSON constant is forbidden: {token}")
            ),
            object_pairs_hook=_strict_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("input catalog is not strict UTF-8 JSON") from exc
    if catalog_bytes != canonical_json_bytes(parsed_catalog) + b"\n":
        raise ContractError("input catalog payload bytes are not canonical UTF-8/LF JSON")
    observed_catalog = _normalize_catalog_payload(parsed_catalog)
    if observed_catalog != _catalog_payload(scope["input_catalog"]):
        raise ContractError("input catalog metadata differs from the canonical run scope")
    validate_output_root_fresh(root, scope, exists=output_root_exists)
    return {
        "receipt_schema_version": METADATA_PREFLIGHT_RECEIPT_VERSION,
        "run_scope_digest": canonical_digest(scope),
        "catalog_digest": catalog_ref["sha256"],
        "source_release_id": scope["input_catalog"]["source_release_id"],
        "row_count_metadata": scope["input_catalog"]["row_count"],
        "race_count_metadata": scope["input_catalog"]["race_count"],
        "runner_count_metadata": scope["input_catalog"]["runner_count"],
        "source_time_coverage_complete": True,
        "revoked": False,
        "real_data_rows_opened": 0,
    }


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def hash_bound_repository_paths(scope: dict[str, Any]) -> set[str]:
    paths = {
        *(item["path"] for item in _all_metadata_refs(scope)),
    }
    committed_paths = {
        *(item["path"] for item in scope["code_hashes"]),
        *(item["path"] for item in scope["config_hashes"]),
        scope["label_eligibility_contract_hash"]["path"],
        scope["fold_manifest_hash"]["path"],
        scope["dependency_environment_lock_hash"]["path"],
        scope["feature_input_release_hash"]["contract"]["path"],
        scope["feature_lineage_manifest_hash"]["contract"]["path"],
    }
    paths.difference_update(committed_paths)
    return set(paths)


def validate_output_root_fresh(
    root: Path,
    scope: dict[str, Any],
    *,
    exists: Callable[[Path], bool] | None = None,
) -> None:
    output = _resolve_repository_path(root.resolve(), scope["output_root"])
    checker = exists or Path.exists
    if checker(output):
        raise ContractError("frozen output root already exists; overwrite/reuse is forbidden")


def verify_metadata_preflight(
    scope: dict[str, Any],
    observed_catalog_metadata: dict[str, Any],
    *,
    row_loader: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Pure metadata comparison; ``row_loader`` is intentionally never called."""

    observed = _normalize_catalog_payload(observed_catalog_metadata)
    if observed != _catalog_payload(scope["input_catalog"]):
        raise ContractError("observed catalog metadata differs from the approved run scope")
    receipt = {
        "receipt_schema_version": METADATA_PREFLIGHT_RECEIPT_VERSION,
        "run_scope_digest": canonical_digest(scope),
        "catalog_digest": scope["input_catalog"]["manifest"]["sha256"],
        "source_release_id": observed["source_release_id"],
        "row_count_metadata": observed["row_count"],
        "race_count_metadata": observed["race_count"],
        "runner_count_metadata": observed["runner_count"],
        "source_time_coverage_complete": True,
        "revoked": False,
        "real_data_rows_opened": 0,
    }
    return receipt


def _approval_evidence_ok(evidence: Any, keyword: str, digest: str) -> bool:
    return bool(
        isinstance(evidence, dict)
        and set(evidence) == set(COMMENT_EVIDENCE_FIELDS)
        and type(evidence.get("comment_id")) is int
        and evidence["comment_id"] > 0
        and evidence.get("approval_type") == keyword
        and evidence.get("approval_digest") == digest
        and evidence.get("body") == f"{keyword} {digest}"
        and evidence.get("author_type") == "User"
        and evidence.get("created_at") == evidence.get("updated_at")
    )


REGISTRY_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"blocked_score", "proposed", "invalid"},
    "blocked_score": {"invalid"},
    "proposed": {"approved_to_prepare", "invalid"},
    "approved_to_prepare": {"preparing", "invalid"},
    "preparing": {"run_approval_required", "invalid"},
    "run_approval_required": {"approved_to_run", "invalid"},
    "approved_to_run": {"running", "invalid"},
    "running": {"review_required", "invalid"},
    "review_required": {"rejected", "approved_for_shadow", "invalid"},
    "rejected": {"invalid"},
    "approved_for_shadow": {"invalid"},
    "invalid": set(),
}
REGISTRY_APPROVAL_KEYWORDS = {
    "approved_to_prepare": "APPROVED_TO_PREPARE",
    "approved_to_run": "APPROVED_TO_RUN",
    "approved_for_shadow": "APPROVED_FOR_SHADOW",
}


def _normalized_registry_status(value: Any) -> str:
    if not isinstance(value, str):
        raise ContractError("Registry status must be a string")
    return value.strip().lower().replace("-", "_")


def _validate_remote_registry_history(
    events: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    """Strictly validate all chains and globally unique approval grants."""

    chains: dict[str, list[dict[str, Any]]] = {}
    grants: dict[int, dict[str, Any]] = {}
    event_ids: set[str] = set()
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise ContractError(f"Registry event {index} is not an object")
        event_id = event.get("event_id")
        experiment_id = event.get("experiment_id")
        if (
            not isinstance(event_id, str)
            or not event_id
            or event_id in event_ids
            or not isinstance(experiment_id, str)
            or not experiment_id
        ):
            raise ContractError("Registry contains an invalid/duplicate event identity")
        event_ids.add(event_id)
        status = _normalized_registry_status(event.get("status"))
        if status not in REGISTRY_TRANSITIONS:
            raise ContractError("Registry contains an unknown status")
        chain = chains.setdefault(experiment_id, [])
        sequence = event.get("sequence")
        if type(sequence) is not int or sequence != len(chain) + 1:
            raise ContractError("Registry sequence is not positive and contiguous")
        previous = chain[-1] if chain else None
        if previous is None:
            if (
                event.get("previous_event_id") is not None
                or event.get("previous_status") is not None
                or status not in REGISTRY_TRANSITIONS[None]
            ):
                raise ContractError("Registry first-event transition is invalid")
        else:
            previous_status = _normalized_registry_status(previous.get("status"))
            if (
                event.get("previous_event_id") != previous.get("event_id")
                or _normalized_registry_status(event.get("previous_status"))
                != previous_status
                or status not in REGISTRY_TRANSITIONS[previous_status]
            ):
                raise ContractError("Registry historical transition is invalid")
        revalidations = event.get("revalidated_approval_evidence", [])
        if not isinstance(revalidations, list):
            raise ContractError("Registry approval revalidations are malformed")
        for evidence in revalidations:
            if (
                not isinstance(evidence, dict)
                or type(evidence.get("comment_id")) is not int
                or grants.get(evidence["comment_id"]) != evidence
            ):
                raise ContractError("Registry revalidation is not an exact prior grant")
        grant = event.get("approval_evidence")
        expected_keyword = REGISTRY_APPROVAL_KEYWORDS.get(status)
        if expected_keyword is None:
            if grant is not None:
                raise ContractError("non-grant Registry event contains approval evidence")
        else:
            digest_field = {
                "approved_to_prepare": "proposal_scope_digest",
                "approved_to_run": "run_scope_digest",
                "approved_for_shadow": "review_digest",
            }[status]
            digest = event.get(digest_field)
            if (
                not isinstance(grant, dict)
                or set(grant) != set(COMMENT_EVIDENCE_FIELDS)
                or type(grant.get("comment_id")) is not int
                or grant["comment_id"] <= 0
                or grant.get("approval_type") != expected_keyword
                or grant.get("approval_digest") != digest
                or not isinstance(digest, str)
                or not FULL_SHA256.fullmatch(digest)
                or grant.get("body") != f"{expected_keyword} {digest}"
                or grant.get("body_sha256")
                != hashlib.sha256(grant["body"].encode("utf-8")).hexdigest()
                or grant["comment_id"] in grants
            ):
                raise ContractError("Registry grant history is malformed or reused")
            grants[grant["comment_id"]] = grant
        chain.append(event)
    return chains, grants


def _verify_live_github_execution_authority(
    *,
    provider: GitHubApprovalProvider,
    run_scope: dict[str, Any],
    execution_commit: str,
    current_main_sha: str,
    prepare_evidence: dict[str, Any],
    run_evidence: dict[str, Any],
    current_main_registry_bytes: bytes,
    merged_running_event: dict[str, Any],
) -> dict[str, Any]:
    """Re-fetch every external authority fact from the code-owned GitHub remote."""

    proposal = run_scope["proposal_scope"]
    allowlist, trust = verify_github_trust(
        provider=provider,
        repository=DEFAULT_REPOSITORY,
        base_branch=DEFAULT_BASE_BRANCH,
        base_commit=proposal["base_commit"],
    )
    observed_main = trust["verified_current_main_sha"]
    if current_main_sha != observed_main:
        raise ContractError("caller current-main SHA differs from live GitHub main")

    try:
        comparison = provider.compare_commits(
            DEFAULT_REPOSITORY,
            execution_commit,
            observed_main,
        )
    except Exception as exc:
        raise ContractError(
            f"GitHub execution-commit ancestry unavailable; fail-close: {exc}"
        ) from exc
    if not isinstance(comparison, dict):
        raise ContractError("GitHub execution-commit comparison is not an object")
    status = comparison.get("status")
    base = comparison.get("base_commit")
    merge_base = comparison.get("merge_base_commit")
    expected_url = (
        f"{GITHUB_API_URL}/repos/{DEFAULT_REPOSITORY}/compare/"
        f"{execution_commit}...{observed_main}"
    )
    if (
        status not in {"ahead", "identical"}
        or not isinstance(base, dict)
        or base.get("sha") != execution_commit
        or not isinstance(merge_base, dict)
        or merge_base.get("sha") != execution_commit
        or comparison.get("url") != expected_url
        or (status == "identical" and execution_commit != observed_main)
        or (status == "ahead" and execution_commit == observed_main)
    ):
        raise ContractError("execution commit is not the exact live GitHub-main ancestor")

    remote_registry, registry_evidence = fetch_github_file_at_commit(
        provider=provider,
        repository=DEFAULT_REPOSITORY,
        path="research/REGISTRY.jsonl",
        ref=observed_main,
    )
    if remote_registry != current_main_registry_bytes:
        raise ContractError("caller Registry bytes differ from live GitHub current main")
    experiment_id = proposal["experiment_id"]
    remote_run_scope, _ = fetch_github_file_at_commit(
        provider=provider,
        repository=DEFAULT_REPOSITORY,
        path=f"research/scopes/{experiment_id}.run.json",
        ref=observed_main,
    )
    if remote_run_scope != canonical_json_bytes(run_scope) + b"\n":
        raise ContractError("live GitHub run-scope bytes differ from the approved scope")
    remote_proposal, _ = fetch_github_file_at_commit(
        provider=provider,
        repository=DEFAULT_REPOSITORY,
        path=f"research/scopes/{experiment_id}.proposal.json",
        ref=observed_main,
    )
    if remote_proposal != canonical_json_bytes(proposal) + b"\n":
        raise ContractError("live GitHub proposal bytes differ from the embedded scope")

    event_trust = merged_running_event.get("github_trust_evidence")
    event_main = (
        event_trust.get("verified_current_main_sha")
        if isinstance(event_trust, dict)
        else None
    )
    if not isinstance(event_main, str) or not FULL_GIT_SHA.fullmatch(event_main):
        raise ContractError("merged RUNNING event lacks a valid pre-merge main anchor")
    try:
        event_comparison = provider.compare_commits(
            DEFAULT_REPOSITORY,
            event_main,
            observed_main,
        )
    except Exception as exc:
        raise ContractError(
            f"RUNNING event merge ancestry unavailable; fail-close: {exc}"
        ) from exc
    event_expected_url = (
        f"{GITHUB_API_URL}/repos/{DEFAULT_REPOSITORY}/compare/"
        f"{event_main}...{observed_main}"
    )
    if (
        not isinstance(event_comparison, dict)
        or event_comparison.get("status") not in {"ahead", "identical"}
        or not isinstance(event_comparison.get("base_commit"), dict)
        or event_comparison["base_commit"].get("sha") != event_main
        or not isinstance(event_comparison.get("merge_base_commit"), dict)
        or event_comparison["merge_base_commit"].get("sha") != event_main
        or event_comparison.get("url") != event_expected_url
        or (
            event_comparison.get("status") == "identical"
            and event_main != observed_main
        )
        or (
            event_comparison.get("status") == "ahead"
            and event_main == observed_main
        )
    ):
        raise ContractError("RUNNING event was not human-merged into live GitHub main")

    live_approvals: list[dict[str, Any]] = []
    for evidence, keyword, digest in (
        (
            prepare_evidence,
            "APPROVED_TO_PREPARE",
            run_scope["proposal_scope_digest"],
        ),
        (run_evidence, "APPROVED_TO_RUN", canonical_digest(run_scope)),
    ):
        if not _approval_evidence_ok(evidence, keyword, digest):
            raise ContractError("stored v3 approval evidence is incomplete or edited")
        try:
            live = verify_approval_comment(
                provider=provider,
                allowlist=allowlist,
                repository=DEFAULT_REPOSITORY,
                issue_number=evidence["issue_number"],
                comment_id=evidence["comment_id"],
                approval_keyword=keyword,
                approval_digest=digest,
            )
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        if live.get("created_at") != live.get("updated_at") or live != evidence:
            raise ContractError("live GitHub approval differs from stored unedited evidence")
        live_approvals.append(live)

    verify_github_main_head_unchanged(
        provider=provider,
        repository=DEFAULT_REPOSITORY,
        base_branch=DEFAULT_BASE_BRANCH,
        expected_sha=observed_main,
    )
    return {
        "current_main_sha": observed_main,
        "registry_sha256": registry_evidence["content_sha256"],
        "execution_compare_status": status,
        "execution_compare_url": expected_url,
        "execution_merge_base_sha": execution_commit,
        "prepare_evidence": live_approvals[0],
        "run_evidence": live_approvals[1],
    }


def normalize_execution_receipt(
    value: Any,
    *,
    run_scope: dict[str, Any],
    run_scope_digest: str,
) -> dict[str, Any]:
    obj = _exact_object(value, EXECUTION_RECEIPT_FIELDS, "execution_receipt")
    proposal = run_scope["proposal_scope"]
    normalized = {
        "receipt_schema_version": _string(
            obj["receipt_schema_version"], "execution_receipt.receipt_schema_version"
        ),
        "experiment_id": _string(obj["experiment_id"], "execution_receipt.experiment_id"),
        "running_event_id": _identifier(
            obj["running_event_id"], "execution_receipt.running_event_id"
        ),
        "run_scope_digest": _hash(
            obj["run_scope_digest"], "execution_receipt.run_scope_digest"
        ),
        "execution_kind": _string(
            obj["execution_kind"], "execution_receipt.execution_kind"
        ),
        "capability_profile_id": _identifier(
            obj["capability_profile_id"], "execution_receipt.capability_profile_id"
        ),
        "execution_commit_sha": _git_sha(
            obj["execution_commit_sha"], "execution_receipt.execution_commit_sha"
        ),
        "verified_current_main_sha": _git_sha(
            obj["verified_current_main_sha"], "execution_receipt.verified_current_main_sha"
        ),
        "execution_commit_compare_status": _string(
            obj["execution_commit_compare_status"],
            "execution_receipt.execution_commit_compare_status",
        ),
        "execution_commit_compare_url": _string(
            obj["execution_commit_compare_url"],
            "execution_receipt.execution_commit_compare_url",
        ),
        "execution_commit_merge_base_sha": _git_sha(
            obj["execution_commit_merge_base_sha"],
            "execution_receipt.execution_commit_merge_base_sha",
        ),
        "current_main_registry_sha256": _hash(
            obj["current_main_registry_sha256"],
            "execution_receipt.current_main_registry_sha256",
        ),
        "prepare_approval_comment_id": _integer(
            obj["prepare_approval_comment_id"],
            "execution_receipt.prepare_approval_comment_id",
            minimum=1,
        ),
        "run_approval_comment_id": _integer(
            obj["run_approval_comment_id"],
            "execution_receipt.run_approval_comment_id",
            minimum=1,
        ),
        "metadata_preflight_digest": _hash(
            obj["metadata_preflight_digest"],
            "execution_receipt.metadata_preflight_digest",
        ),
        "capability_profile_digest": _hash(
            obj["capability_profile_digest"],
            "execution_receipt.capability_profile_digest",
        ),
        "input_manifest_hashes_digest": _hash(
            obj["input_manifest_hashes_digest"],
            "execution_receipt.input_manifest_hashes_digest",
        ),
        "environment_digest": _hash(
            obj["environment_digest"], "execution_receipt.environment_digest"
        ),
        "exact_commands_digest": _hash(
            obj["exact_commands_digest"], "execution_receipt.exact_commands_digest"
        ),
        "read_allowlist_digest": _hash(
            obj["read_allowlist_digest"], "execution_receipt.read_allowlist_digest"
        ),
        "write_allowlist_digest": _hash(
            obj["write_allowlist_digest"], "execution_receipt.write_allowlist_digest"
        ),
        "output_root": _repository_path(obj["output_root"], "execution_receipt.output_root"),
        "output_root_reservation_digest": _hash(
            obj["output_root_reservation_digest"],
            "execution_receipt.output_root_reservation_digest",
        ),
        "output_root_was_fresh": _bool(
            obj["output_root_was_fresh"], "execution_receipt.output_root_was_fresh"
        ),
        "real_data_execution_allowed": _bool(
            obj["real_data_execution_allowed"],
            "execution_receipt.real_data_execution_allowed",
        ),
        "execution_authorized": _bool(
            obj["execution_authorized"],
            "execution_receipt.execution_authorized",
        ),
        "issued_at": _canonical_timestamp(obj["issued_at"], "execution_receipt.issued_at"),
        "formal_buy": _bool(obj["formal_buy"], "execution_receipt.formal_buy"),
        "send_order": _bool(obj["send_order"], "execution_receipt.send_order"),
        "stake": _integer(obj["stake"], "execution_receipt.stake"),
    }
    if normalized["receipt_schema_version"] != EXECUTION_RECEIPT_SCHEMA_VERSION:
        raise ContractError("unknown execution receipt schema")
    if normalized["experiment_id"] != proposal["experiment_id"]:
        raise ContractError("execution receipt experiment mismatch")
    if normalized["run_scope_digest"] != run_scope_digest:
        raise ContractError("execution receipt run scope digest mismatch")
    if normalized["execution_kind"] != run_scope["execution_kind"]:
        raise ContractError("execution receipt kind mismatch")
    if normalized["capability_profile_id"] != run_scope["capability_profile"]["profile_id"]:
        raise ContractError("execution receipt capability mismatch")
    if normalized["execution_commit_sha"] != run_scope["execution_commit_sha"]:
        raise ContractError("execution receipt commit mismatch")
    if normalized["execution_commit_compare_status"] not in {"ahead", "identical"}:
        raise ContractError("execution receipt ancestry status is invalid")
    if normalized["execution_commit_merge_base_sha"] != run_scope["execution_commit_sha"]:
        raise ContractError("execution receipt merge base differs from execution commit")
    parsed_compare_url = urllib.parse.urlparse(normalized["execution_commit_compare_url"])
    if (
        parsed_compare_url.scheme != "https"
        or parsed_compare_url.netloc != "api.github.com"
        or not parsed_compare_url.path.endswith(
            f"/compare/{run_scope['execution_commit_sha']}..."
            f"{normalized['verified_current_main_sha']}"
        )
    ):
        raise ContractError("execution receipt compare URL is not exact GitHub ancestry evidence")
    if (
        normalized["execution_commit_compare_status"] == "identical"
        and normalized["verified_current_main_sha"] != run_scope["execution_commit_sha"]
    ):
        raise ContractError("identical ancestry receipt has mismatched current main")
    if (
        normalized["execution_commit_compare_status"] == "ahead"
        and normalized["verified_current_main_sha"] == run_scope["execution_commit_sha"]
    ):
        raise ContractError("ahead ancestry receipt has identical SHAs")
    if normalized["capability_profile_digest"] != canonical_digest(
        run_scope["capability_profile"]
    ):
        raise ContractError("execution receipt capability profile digest mismatch")
    if normalized["input_manifest_hashes_digest"] != canonical_digest(
        run_scope["data_input_manifest_hashes"]
    ):
        raise ContractError("execution receipt input manifest digest mismatch")
    if normalized["environment_digest"] != canonical_digest(run_scope["environment"]):
        raise ContractError("execution receipt environment digest mismatch")
    if normalized["exact_commands_digest"] != canonical_digest(run_scope["exact_commands"]):
        raise ContractError("execution receipt command digest mismatch")
    if normalized["read_allowlist_digest"] != canonical_digest(run_scope["read_allowlist"]):
        raise ContractError("execution receipt read allowlist digest mismatch")
    if normalized["write_allowlist_digest"] != canonical_digest(run_scope["write_allowlist"]):
        raise ContractError("execution receipt write allowlist digest mismatch")
    if normalized["output_root"] != run_scope["output_root"]:
        raise ContractError("execution receipt output root mismatch")
    if normalized["prepare_approval_comment_id"] == normalized["run_approval_comment_id"]:
        raise ContractError("prepare and run approvals must use different comment IDs")
    if normalized["output_root_was_fresh"] is not True:
        raise ContractError("execution receipt lacks a fresh output-root reservation")
    if (
        normalized["real_data_execution_allowed"] is not False
        or normalized["execution_authorized"] is not False
        or run_scope["execution_kind"] != "real_data"
    ):
        raise ContractError(
            "the durable receipt must remain non-executing; effective authority is phase-bound"
        )
    if datetime.fromisoformat(normalized["issued_at"][:-1] + "+00:00") < datetime.fromisoformat(
        run_scope["as_of"][:-1] + "+00:00"
    ):
        raise ContractError("execution receipt predates the frozen as_of")
    expected_reservation_digest = canonical_digest(
        {
            "run_scope_digest": run_scope_digest,
            "verified_current_main_sha": normalized["verified_current_main_sha"],
            "output_root": run_scope["output_root"],
            "fresh": True,
        }
    )
    if normalized["output_root_reservation_digest"] != expected_reservation_digest:
        raise ContractError("output-root reservation digest mismatch")
    if normalized["formal_buy"] or normalized["send_order"] or normalized["stake"] != 0:
        raise ContractError("execution receipt safety flags changed")
    return normalized


def verify_real_data_authorization(
    *,
    root: Path | None = None,
    status: str,
    run_scope: dict[str, Any],
    cli_execution_kind: str,
    prepare_evidence: dict[str, Any] | None,
    run_evidence: dict[str, Any] | None,
    execution_commit: str,
    current_main_sha: str,
    merged_running_event: dict[str, Any] | None = None,
    current_main_registry_bytes: bytes | None = None,
    execution_receipt: dict[str, Any] | None = None,
    metadata_preflight_receipt: dict[str, Any] | None = None,
    observed_environment: dict[str, Any] | None = None,
    phase_id: str | None = None,
    observed_argv: list[str] | None = None,
) -> bool:
    """Return true only for a fully bound, merged, receipt-backed v3 real run."""

    try:
        canonical_scope = normalize_ordinary_real_data_run_scope(
            run_scope,
            proposal_scope=run_scope["proposal_scope"],
        )
    except (KeyError, ContractError, TypeError):
        return False
    if canonical_scope != run_scope:
        return False
    if run_scope.get("run_scope_schema_version") != RUN_SCOPE_SCHEMA_VERSION:
        return False
    if status != "running" or run_scope.get("execution_kind") != "real_data":
        return False
    if cli_execution_kind not in {"real_data", "real-data"}:
        return False
    phase = next(
        (item for item in run_scope.get("phase_plan", []) if item.get("phase_id") == phase_id),
        None,
    )
    if not isinstance(phase, dict):
        return False
    command = next(
        (
            item
            for item in run_scope.get("exact_commands", [])
            if item.get("command_id") == phase.get("command_id")
        ),
        None,
    )
    if not isinstance(command, dict) or observed_argv != command.get("argv"):
        return False
    run_digest = canonical_digest(run_scope)
    proposal_digest = run_scope["proposal_scope_digest"]
    if not _approval_evidence_ok(prepare_evidence, "APPROVED_TO_PREPARE", proposal_digest):
        return False
    if not _approval_evidence_ok(run_evidence, "APPROVED_TO_RUN", run_digest):
        return False
    if prepare_evidence["comment_id"] == run_evidence["comment_id"]:
        return False
    if execution_commit != run_scope["execution_commit_sha"]:
        return False
    if not FULL_GIT_SHA.fullmatch(current_main_sha):
        return False
    if (
        not isinstance(merged_running_event, dict)
        or not isinstance(current_main_registry_bytes, bytes)
        or not isinstance(execution_receipt, dict)
        or not isinstance(metadata_preflight_receipt, dict)
        or observed_environment != run_scope.get("environment")
    ):
        return False
    approval_provider: GitHubApprovalProvider = GitHubRestApprovalProvider()
    try:
        live_authority = _verify_live_github_execution_authority(
            provider=approval_provider,
            run_scope=run_scope,
            execution_commit=execution_commit,
            current_main_sha=current_main_sha,
            prepare_evidence=prepare_evidence,
            run_evidence=run_evidence,
            current_main_registry_bytes=current_main_registry_bytes,
            merged_running_event=merged_running_event,
        )
    except (ContractError, ValueError, KeyError, TypeError):
        return False
    trust = merged_running_event.get("github_trust_evidence")
    revalidations = merged_running_event.get("revalidated_approval_evidence")
    if (
        merged_running_event.get("status") != "running"
        or merged_running_event.get("run_scope_digest") != run_digest
        or merged_running_event.get("execution_kind") != "real-data"
        or merged_running_event.get("real_data_execution_allowed") is not False
        or not isinstance(trust, dict)
        or not isinstance(trust.get("verified_current_main_sha"), str)
        or not FULL_GIT_SHA.fullmatch(trust["verified_current_main_sha"])
        or merged_running_event.get("human_prepare_approval_recorded") is not True
        or merged_running_event.get("human_run_approval_recorded") is not True
        or merged_running_event.get("execution_authorized") is not False
        or merged_running_event.get("automatic_execution_allowed") is not False
        or merged_running_event.get("formal_buy") is not False
        or merged_running_event.get("send_order") is not False
        or merged_running_event.get("stake") != 0
        or not isinstance(revalidations, list)
        or prepare_evidence not in revalidations
        or run_evidence not in revalidations
    ):
        return False
    if hashlib.sha256(current_main_registry_bytes).hexdigest() != execution_receipt.get(
        "current_main_registry_sha256"
    ):
        return False
    try:
        registry_events = [
            json.loads(
                line.decode("utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ContractError(f"non-standard JSON constant is forbidden: {token}")
                ),
                object_pairs_hook=_strict_pairs,
            )
            for line in current_main_registry_bytes.splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError, ContractError):
        return False
    try:
        chains, _ = _validate_remote_registry_history(registry_events)
    except ContractError:
        return False
    matching_running = [
        event
        for event in registry_events
        if isinstance(event, dict)
        and event.get("event_id") == merged_running_event.get("event_id")
    ]
    if matching_running != [merged_running_event]:
        return False
    experiment_events = chains.get(
        run_scope["proposal_scope"]["experiment_id"], []
    )
    if not experiment_events or experiment_events[-1] != merged_running_event:
        return False
    for index, event in enumerate(experiment_events, start=1):
        if event.get("sequence") != index:
            return False
        if index == 1:
            if event.get("previous_event_id") is not None or event.get("previous_status") is not None:
                return False
        else:
            previous = experiment_events[index - 2]
            if (
                event.get("previous_event_id") != previous.get("event_id")
                or event.get("previous_status") != previous.get("status")
            ):
                return False
    expected_run_path = (
        f"research/scopes/{run_scope['proposal_scope']['experiment_id']}.run.json"
    )
    if merged_running_event.get("run_scope_file") != expected_run_path:
        return False
    for evidence in (prepare_evidence, run_evidence):
        grant_count = sum(
            isinstance(event, dict)
            and isinstance(event.get("approval_evidence"), dict)
            and event["approval_evidence"].get("comment_id") == evidence["comment_id"]
            for event in registry_events
        )
        if grant_count != 1:
            return False
    expected_preflight = {
        "receipt_schema_version": METADATA_PREFLIGHT_RECEIPT_VERSION,
        "run_scope_digest": run_digest,
        "catalog_digest": run_scope["input_catalog"]["manifest"]["sha256"],
        "source_release_id": run_scope["input_catalog"]["source_release_id"],
        "row_count_metadata": run_scope["input_catalog"]["row_count"],
        "race_count_metadata": run_scope["input_catalog"]["race_count"],
        "runner_count_metadata": run_scope["input_catalog"]["runner_count"],
        "source_time_coverage_complete": True,
        "revoked": False,
        "real_data_rows_opened": 0,
    }
    if metadata_preflight_receipt != expected_preflight:
        return False
    try:
        receipt = normalize_execution_receipt(
            execution_receipt,
            run_scope=run_scope,
            run_scope_digest=run_digest,
        )
    except ContractError:
        return False
    if root is None:
        return False
    try:
        resolved_root = root.resolve()
        output_root_path = _resolve_repository_path(
            resolved_root, run_scope["output_root"]
        )
        receipt_path = _resolve_repository_path(
            resolved_root,
            run_scope["output_sealing_contract"]["execution_receipt_path"],
        )
        if (
            not output_root_path.is_dir()
            or output_root_path.is_symlink()
            or not receipt_path.is_file()
            or receipt_path.is_symlink()
            or receipt_path.read_bytes() != canonical_json_bytes(receipt) + b"\n"
        ):
            return False
    except (OSError, ContractError):
        return False
    return bool(
        receipt["running_event_id"] == merged_running_event.get("event_id")
        and receipt["verified_current_main_sha"] == current_main_sha
        and receipt["current_main_registry_sha256"]
        == live_authority["registry_sha256"]
        and receipt["execution_commit_compare_status"]
        == live_authority["execution_compare_status"]
        and receipt["execution_commit_compare_url"]
        == live_authority["execution_compare_url"]
        and receipt["execution_commit_merge_base_sha"]
        == live_authority["execution_merge_base_sha"]
        and receipt["prepare_approval_comment_id"] == prepare_evidence["comment_id"]
        and receipt["run_approval_comment_id"] == run_evidence["comment_id"]
        and receipt["metadata_preflight_digest"]
        == canonical_digest(metadata_preflight_receipt)
    )


def _atomic_reserve_output_root(path: Path) -> bool:
    """Atomically reserve one fresh root; a crash leaves it permanently unusable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        return False
    return True


def issue_execution_receipt(
    *,
    root: Path,
    status: str,
    run_scope: dict[str, Any],
    cli_execution_kind: str,
    prepare_evidence: dict[str, Any],
    run_evidence: dict[str, Any],
    execution_commit: str,
    current_main_sha: str,
    ancestry_evidence: dict[str, Any],
    merged_running_event: dict[str, Any],
    current_main_registry_bytes: bytes,
    metadata_preflight_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Issue a receipt only after merged authority checks and atomic root reserve.

    The function executes no experiment command and opens no data row.  If it crashes
    after reserving the directory but before the caller persists the receipt, that root
    remains unusable and a new scope/root is required.
    """

    root = root.resolve()
    approval_provider: GitHubApprovalProvider = GitHubRestApprovalProvider()
    if Path.cwd().resolve() != root:
        raise ContractError("receipt issuer cwd differs from the frozen repository root")
    verify_runtime_interpreter_isolation()
    observed_commit = _current_git_commit(root)
    if observed_commit != run_scope["execution_commit_sha"] or execution_commit != observed_commit:
        raise ContractError("receipt issuer is not running the frozen execution commit")
    verify_execution_worktree_state(root, run_scope)
    observed_environment = observe_runtime_environment(run_scope)
    if observed_environment != run_scope["environment"]:
        raise ContractError("receipt issuer environment differs from the frozen environment")
    observed_preflight = verify_ordinary_real_data_run_materials(root, run_scope)
    if observed_preflight != metadata_preflight_receipt:
        raise ContractError("metadata preflight receipt was not reproduced at issuance")

    ancestry = _exact_object(
        ancestry_evidence,
        ANCESTRY_EVIDENCE_FIELDS,
        "execution ancestry evidence",
    )
    run_scope_digest = canonical_digest(run_scope)
    reservation_digest = canonical_digest(
        {
            "run_scope_digest": run_scope_digest,
            "verified_current_main_sha": current_main_sha,
            "output_root": run_scope["output_root"],
            "fresh": True,
        }
    )
    receipt = {
        "receipt_schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
        "experiment_id": run_scope["proposal_scope"]["experiment_id"],
        "running_event_id": merged_running_event.get("event_id"),
        "run_scope_digest": run_scope_digest,
        "execution_kind": run_scope["execution_kind"],
        "capability_profile_id": run_scope["capability_profile"]["profile_id"],
        "execution_commit_sha": execution_commit,
        "verified_current_main_sha": current_main_sha,
        "execution_commit_compare_status": ancestry.get("status"),
        "execution_commit_compare_url": ancestry.get("url"),
        "execution_commit_merge_base_sha": ancestry.get("merge_base_sha"),
        "current_main_registry_sha256": hashlib.sha256(
            current_main_registry_bytes
        ).hexdigest(),
        "prepare_approval_comment_id": prepare_evidence.get("comment_id"),
        "run_approval_comment_id": run_evidence.get("comment_id"),
        "metadata_preflight_digest": canonical_digest(metadata_preflight_receipt),
        "capability_profile_digest": canonical_digest(run_scope["capability_profile"]),
        "input_manifest_hashes_digest": canonical_digest(
            run_scope["data_input_manifest_hashes"]
        ),
        "environment_digest": canonical_digest(run_scope["environment"]),
        "exact_commands_digest": canonical_digest(run_scope["exact_commands"]),
        "read_allowlist_digest": canonical_digest(run_scope["read_allowlist"]),
        "write_allowlist_digest": canonical_digest(run_scope["write_allowlist"]),
        "output_root": run_scope["output_root"],
        "output_root_reservation_digest": reservation_digest,
        "output_root_was_fresh": True,
        "real_data_execution_allowed": False,
        "execution_authorized": False,
        "issued_at": _utc_now_text(),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    _verify_live_github_execution_authority(
        provider=approval_provider,
        run_scope=run_scope,
        execution_commit=execution_commit,
        current_main_sha=current_main_sha,
        prepare_evidence=prepare_evidence,
        run_evidence=run_evidence,
        current_main_registry_bytes=current_main_registry_bytes,
        merged_running_event=merged_running_event,
    )
    normalize_execution_receipt(
        receipt,
        run_scope=run_scope,
        run_scope_digest=run_scope_digest,
    )
    output_path = _resolve_repository_path(root, run_scope["output_root"])
    if not _atomic_reserve_output_root(output_path):
        raise ContractError("output root was already present or could not be reserved")
    if not output_path.is_dir() or output_path.is_symlink():
        raise ContractError("output-root reservation is not an exact fresh directory")
    normalized_receipt = normalize_execution_receipt(
        receipt,
        run_scope=run_scope,
        run_scope_digest=run_scope_digest,
    )
    receipt_path = _resolve_repository_path(
        root,
        run_scope["output_sealing_contract"]["execution_receipt_path"],
    )
    receipt_payload = canonical_json_bytes(normalized_receipt) + b"\n"
    try:
        descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(receipt_payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ContractError(
            "cannot persist the immutable execution receipt; output root remains unusable"
        ) from exc
    return normalized_receipt


def verify_access_request(
    scope: dict[str, Any],
    *,
    phase_id: str,
    mode: str,
    path: str,
    authority_context: dict[str, Any] | None = None,
) -> None:
    """Validate one access request; row/write access requires a receipt."""

    canonical_path = _repository_path(path, "access path")
    phase = next((item for item in scope["phase_plan"] if item["phase_id"] == phase_id), None)
    if phase is None:
        raise ContractError("access request cites an unknown phase")
    read_entry = next(
        (item for item in scope["read_allowlist"] if item["path"] == canonical_path),
        None,
    )
    if mode == "metadata_read":
        allowed = (
            canonical_path in phase["read_paths"]
            and isinstance(read_entry, dict)
            and read_entry["access_class"] == "metadata_manifest"
        )
    elif mode == "row_read":
        allowed = (
            canonical_path in phase["read_paths"]
            and isinstance(read_entry, dict)
            and read_entry["access_class"].endswith("_row_blob")
            and read_entry["required_capability"]
            in phase["required_capabilities"]
        )
    elif mode == "write":
        allowed = canonical_path in phase["write_paths"]
    else:
        raise ContractError("unknown access mode")
    if not allowed:
        raise ContractError("access request is outside the phase-specific allowlist")
    if mode in {"row_read", "write"}:
        if not isinstance(authority_context, dict):
            raise ContractError("row/write access requires a complete v3 authority context")
        context = _exact_object(
            authority_context,
            AUTHORITY_CONTEXT_FIELDS,
            "authority_context",
        ).copy()
        authority_root = context["root"]
        if not isinstance(authority_root, Path):
            raise ContractError("authority context root must be an exact Path")
        if _current_git_commit(authority_root.resolve()) != scope["execution_commit_sha"]:
            raise ContractError("access broker is not running the frozen execution commit")
        if Path.cwd().resolve() != authority_root.resolve():
            raise ContractError("access broker cwd differs from the frozen repository root")
        verify_runtime_interpreter_isolation()
        verify_execution_worktree_state(authority_root.resolve(), scope)
        measured_environment = observe_runtime_environment(scope)
        if context["observed_environment"] != measured_environment:
            raise ContractError("access broker environment changed after receipt issuance")
        command = next(
            item
            for item in scope["exact_commands"]
            if item["command_id"] == phase["command_id"]
        )
        observed_argv = observe_process_argv()
        if observed_argv != command["argv"]:
            raise ContractError("runtime argv differs from the exact approved phase command")
        context["run_scope"] = scope
        context["phase_id"] = phase_id
        context["observed_argv"] = observed_argv
        if not verify_real_data_authorization(**context):
            raise ContractError("row/write access lacks verified merged v3 authority")


def read_authorized_bytes(
    root: Path,
    scope: dict[str, Any],
    *,
    phase_id: str,
    path: str,
    authority_context: dict[str, Any] | None = None,
    metadata_only: bool = False,
) -> bytes:
    """Authorize first, then open and hash one exact metadata/row path."""

    mode = "metadata_read" if metadata_only else "row_read"
    resolved_root = root.resolve()
    if not metadata_only:
        if (
            not isinstance(authority_context, dict)
            or not isinstance(authority_context.get("root"), Path)
            or authority_context["root"].resolve() != resolved_root
        ):
            raise ContractError("authorized read root differs from the receipt root")
    if scope["repository_working_directory"] != resolved_root.as_posix():
        raise ContractError("authorized read root differs from the frozen repository cwd")
    verify_access_request(
        scope,
        phase_id=phase_id,
        mode=mode,
        path=path,
        authority_context=authority_context,
    )
    canonical_path = _repository_path(path, "authorized read path")
    resolved = _resolve_repository_path(resolved_root, canonical_path)
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError("authorized read target is not an exact regular file")
    payload = resolved.read_bytes()
    entry = next(item for item in scope["read_allowlist"] if item["path"] == canonical_path)
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ContractError("authorized input bytes changed after run approval")
    return payload


def write_authorized_bytes(
    root: Path,
    scope: dict[str, Any],
    *,
    phase_id: str,
    path: str,
    payload: bytes,
    authority_context: dict[str, Any],
) -> None:
    """Write one exact allowlisted output with O_EXCL after live authority checks."""

    if not isinstance(payload, bytes):
        raise ContractError("authorized output payload must be bytes")
    resolved_root = root.resolve()
    if (
        not isinstance(authority_context, dict)
        or not isinstance(authority_context.get("root"), Path)
        or authority_context["root"].resolve() != resolved_root
        or scope["repository_working_directory"] != resolved_root.as_posix()
    ):
        raise ContractError("authorized write root differs from receipt/frozen cwd")
    verify_access_request(
        scope,
        phase_id=phase_id,
        mode="write",
        path=path,
        authority_context=authority_context,
    )
    canonical_path = _repository_path(path, "authorized write path")
    resolved = _resolve_repository_path(resolved_root, canonical_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_repository_path(resolved_root, canonical_path)
    try:
        descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ContractError("authorized output path already exists or cannot be sealed") from exc


def _result_artifacts(value: Any, output_root: str, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be an array")
    result: list[dict[str, Any]] = []
    prefix = output_root + "/"
    for index, item in enumerate(value):
        obj = _exact_object(item, RESULT_ARTIFACT_FIELDS, f"{label}[{index}]")
        path = _repository_path(obj["path"], f"{label}[{index}].path")
        if not path.startswith(prefix):
            raise ContractError("result artifact is outside the frozen output root")
        result.append(
            {
                "role": _identifier(obj["role"], f"{label}[{index}].role"),
                "path": path,
                "sha256": _hash(obj["sha256"], f"{label}[{index}].sha256"),
                "row_count": _integer(obj["row_count"], f"{label}[{index}].row_count"),
                "race_count": _integer(obj["race_count"], f"{label}[{index}].race_count"),
                "runner_count": _integer(obj["runner_count"], f"{label}[{index}].runner_count"),
                "complete": _bool(obj["complete"], f"{label}[{index}].complete"),
            }
        )
    paths = [item["path"] for item in result]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ContractError(f"{label} must be sorted and path-unique")
    roles = [item["role"] for item in result]
    if len(roles) != len(set(roles)):
        raise ContractError(f"{label} roles must be unique")
    return result


def normalize_result_manifest(
    value: Any,
    *,
    run_scope: dict[str, Any],
    run_scope_digest: str,
) -> dict[str, Any]:
    obj = _exact_object(value, RESULT_FIELDS, "result_manifest")
    status = _string(obj["status"], "result_manifest.status")
    if status not in {"success", "failure"}:
        raise ContractError("result status must be success or failure")
    artifacts = _result_artifacts(obj["artifacts"], run_scope["output_root"], "artifacts")
    partial = _result_artifacts(
        obj["partial_outputs"], run_scope["output_root"], "partial_outputs"
    )
    consumer = _bool(obj["consumer_eligible"], "result_manifest.consumer_eligible")
    expected_roles = set(run_scope["output_sealing_contract"]["artifact_roles"])
    expected_role_paths = {
        item["role"]: item["path"]
        for item in run_scope["output_sealing_contract"]["artifact_paths"]
    }
    artifact_roles = {item["role"] for item in artifacts}
    partial_roles = {item["role"] for item in partial}
    if artifact_roles & partial_roles or not (artifact_roles | partial_roles).issubset(
        expected_roles
    ):
        raise ContractError("result artifact roles are duplicated or outside the frozen set")
    for item in [*artifacts, *partial]:
        if item["path"] != expected_role_paths.get(item["role"]):
            raise ContractError("result artifact path differs from the frozen role mapping")
    if status == "success":
        if partial or not consumer or artifact_roles != expected_roles or not all(
            item["complete"] for item in artifacts
        ):
            raise ContractError("successful output must be complete, role-exact and consumer-eligible")
    elif consumer:
        raise ContractError("failure must be consumer-ineligible")
    if any(item["complete"] for item in partial):
        raise ContractError("partial outputs must be explicitly incomplete")
    if {item["path"] for item in artifacts} & {item["path"] for item in partial}:
        raise ContractError("complete and partial output paths must be disjoint")
    normalized = {
        "result_manifest_schema_version": _string(
            obj["result_manifest_schema_version"], "result_manifest_schema_version"
        ),
        "experiment_id": _string(obj["experiment_id"], "result_manifest.experiment_id"),
        "capability_profile_id": _identifier(
            obj["capability_profile_id"], "result_manifest.capability_profile_id"
        ),
        "run_scope_digest": _hash(obj["run_scope_digest"], "result_manifest.run_scope_digest"),
        "execution_receipt_digest": _hash(
            obj["execution_receipt_digest"], "result_manifest.execution_receipt_digest"
        ),
        "status": status,
        "generated_at": _canonical_timestamp(obj["generated_at"], "result_manifest.generated_at"),
        "as_of": _canonical_timestamp(obj["as_of"], "result_manifest.as_of"),
        "output_root": _repository_path(obj["output_root"], "result_manifest.output_root"),
        "artifacts": artifacts,
        "partial_outputs": partial,
        "code_hashes_digest": _hash(obj["code_hashes_digest"], "result_manifest.code_hashes_digest"),
        "config_hashes_digest": _hash(
            obj["config_hashes_digest"], "result_manifest.config_hashes_digest"
        ),
        "input_manifest_hashes_digest": _hash(
            obj["input_manifest_hashes_digest"],
            "result_manifest.input_manifest_hashes_digest",
        ),
        "environment_lock_sha256": _hash(
            obj["environment_lock_sha256"], "result_manifest.environment_lock_sha256"
        ),
        "consumer_eligible": consumer,
        "formal_buy": _bool(obj["formal_buy"], "result_manifest.formal_buy"),
        "send_order": _bool(obj["send_order"], "result_manifest.send_order"),
        "stake": _integer(obj["stake"], "result_manifest.stake"),
    }
    if normalized["result_manifest_schema_version"] != RESULT_MANIFEST_SCHEMA_VERSION:
        raise ContractError("unknown result manifest schema version")
    if normalized["experiment_id"] != run_scope["proposal_scope"]["experiment_id"]:
        raise ContractError("result experiment mismatch")
    if normalized["capability_profile_id"] != run_scope["capability_profile"]["profile_id"]:
        raise ContractError("result capability profile mismatch")
    if normalized["run_scope_digest"] != run_scope_digest:
        raise ContractError("result run scope digest mismatch")
    if normalized["as_of"] != run_scope["as_of"] or normalized["output_root"] != run_scope["output_root"]:
        raise ContractError("result as_of/output root differs from the run scope")
    if datetime.fromisoformat(normalized["generated_at"][:-1] + "+00:00") < datetime.fromisoformat(
        normalized["as_of"][:-1] + "+00:00"
    ):
        raise ContractError("result generated_at precedes the frozen as_of")
    if normalized["environment_lock_sha256"] != run_scope["dependency_environment_lock_hash"]["sha256"]:
        raise ContractError("result environment hash differs from the run scope")
    if normalized["code_hashes_digest"] != canonical_digest(run_scope["code_hashes"]):
        raise ContractError("result code digest differs from the run scope")
    if normalized["config_hashes_digest"] != canonical_digest(run_scope["config_hashes"]):
        raise ContractError("result config digest differs from the run scope")
    if normalized["input_manifest_hashes_digest"] != canonical_digest(
        run_scope["data_input_manifest_hashes"]
    ):
        raise ContractError("result input digest differs from the run scope")
    if normalized["formal_buy"] or normalized["send_order"] or normalized["stake"] != 0:
        raise ContractError("result safety flags changed")
    return normalized


def canonical_result_manifest_bytes(
    value: Any,
    *,
    run_scope: dict[str, Any],
    run_scope_digest: str,
) -> bytes:
    return canonical_json_bytes(
        normalize_result_manifest(
            value,
            run_scope=run_scope,
            run_scope_digest=run_scope_digest,
        )
    ) + b"\n"


def _derive_artifact_counts(
    role: str,
    payload: bytes,
    *,
    complete: bool,
    run_scope: dict[str, Any],
) -> tuple[int, int, int]:
    artifact_format = ARTIFACT_FORMAT_BY_ROLE[role]
    if artifact_format == "opaque_binary":
        if not payload:
            raise ContractError(f"sealed binary artifact is empty: {role}")
        return 0, 0, 0
    if artifact_format == "canonical_json":
        _strict_canonical_json_bytes(payload, f"output artifact {role}")
        return 0, 0, 0
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ContractError(f"output JSONL is not canonical LF data: {role}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(
                line.decode("utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ContractError(f"non-standard JSON constant is forbidden: {token}")
                ),
                object_pairs_hook=_strict_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"output JSONL row is invalid: {role}[{index}]") from exc
        if not isinstance(value, dict) or line != canonical_json_bytes(value):
            raise ContractError(f"output JSONL row is not canonical: {role}[{index}]")
        for field in ("race_id", "horse_id"):
            raw = value.get(field)
            if (
                not isinstance(raw, str)
                or not raw
                or raw != raw.strip()
            ):
                raise ContractError(
                    f"output JSONL lacks canonical identity: {role}[{index}].{field}"
                )
        rows.append(value)
    races = {row["race_id"] for row in rows}
    identities = {(row["race_id"], row["horse_id"]) for row in rows}
    row_count, race_count, runner_count = len(rows), len(races), len(identities)
    if complete and role in {
        "canonical_runner_universe",
        "canonical_target_input_release",
        "target_research_prediction",
    } and (row_count, race_count, runner_count) != (70, 5, 70):
        raise ContractError(f"{role} must contain exact 5-race/70-runner rows")
    if complete and role in {
        "canonical_runner_universe",
        "canonical_target_input_release",
        "canonical_feature_lineage",
        "target_research_prediction",
    }:
        identity_digest = canonical_digest(
            [
                {"race_id": race_id, "horse_id": horse_id}
                for race_id, horse_id in sorted(identities)
            ]
        )
        if (
            sorted(races) != run_scope["input_catalog"]["race_ids"]
            or identity_digest
            != run_scope["input_catalog"]["runner_identity_digest"]
        ):
            raise ContractError(f"{role} identity universe differs from the run scope")
    if role in {
        "canonical_training_input_release",
        "canonical_feature_lineage",
        "label_eligibility_reconciliation",
        "outer_oos_report",
    } and (row_count == 0 or race_count == 0 or runner_count == 0):
        raise ContractError(f"{role} cannot be an empty successful artifact")
    return row_count, race_count, runner_count


def _verify_exact_output_root_contents(
    root: Path,
    run_scope: dict[str, Any],
    *,
    expected_file_paths: set[str],
) -> None:
    """Reject symlinks, transient files, and unmanifested directories in the root."""

    resolved_root = root.resolve()
    output_root = _resolve_repository_path(resolved_root, run_scope["output_root"])
    if not output_root.is_dir() or output_root.is_symlink():
        raise ContractError("sealed output root is absent or not an exact directory")
    expected_absolute = {
        _resolve_repository_path(resolved_root, path) for path in expected_file_paths
    }
    allowed_directories = {output_root}
    for path in expected_absolute:
        parent = path.parent
        while parent != output_root:
            if output_root not in parent.parents:
                raise ContractError("expected output file escapes the frozen output root")
            allowed_directories.add(parent)
            parent = parent.parent
    observed_files: set[Path] = set()
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ContractError("sealed output root contains a symlink")
        if path.is_dir():
            if path not in allowed_directories:
                raise ContractError("sealed output root contains an unapproved directory")
        elif path.is_file():
            observed_files.add(path)
        else:
            raise ContractError("sealed output root contains a non-regular filesystem entry")
    if observed_files != expected_absolute:
        raise ContractError("sealed output root contains missing or unmanifested files")


def verify_output_seal(
    *,
    root: Path,
    run_scope: dict[str, Any],
    run_scope_digest: str,
    execution_receipt: dict[str, Any],
    result_manifest: dict[str, Any],
    result_manifest_path: str,
    result_manifest_bytes: bytes,
    artifact_bytes_by_path: dict[str, bytes],
    authority_context: dict[str, Any],
) -> dict[str, Any]:
    """Verify a result/failure seal against receipt and exact output bytes."""

    receipt = normalize_execution_receipt(
        execution_receipt,
        run_scope=run_scope,
        run_scope_digest=run_scope_digest,
    )
    if (
        not isinstance(authority_context, dict)
        or authority_context.get("execution_receipt") != receipt
        or not isinstance(authority_context.get("root"), Path)
        or authority_context["root"].resolve() != root.resolve()
    ):
        raise ContractError("output seal authority differs from its durable receipt/root")
    normalized = normalize_result_manifest(
        result_manifest,
        run_scope=run_scope,
        run_scope_digest=run_scope_digest,
    )
    generated_at = datetime.fromisoformat(
        normalized["generated_at"][:-1] + "+00:00"
    )
    observed_now = datetime.fromisoformat(_utc_now_text()[:-1] + "+00:00")
    if generated_at > observed_now + timedelta(seconds=5):
        raise ContractError("result generated_at is in the future")
    if generated_at < datetime.fromisoformat(receipt["issued_at"][:-1] + "+00:00"):
        raise ContractError("result generated_at predates the execution receipt")
    if normalized["execution_receipt_digest"] != canonical_digest(receipt):
        raise ContractError("result manifest execution receipt digest mismatch")
    expected_manifest_path = run_scope["output_sealing_contract"][
        "result_manifest_path"
        if normalized["status"] == "success"
        else "failure_manifest_path"
    ]
    if result_manifest_path != expected_manifest_path:
        raise ContractError("result/failure manifest path differs from the frozen contract")
    expected_manifest_bytes = canonical_json_bytes(normalized) + b"\n"
    if result_manifest_bytes != expected_manifest_bytes:
        raise ContractError("result/failure manifest bytes are not exact canonical UTF-8/LF")
    verify_access_request(
        run_scope,
        phase_id="seal_research_outputs",
        mode="write",
        path=result_manifest_path,
        authority_context=authority_context,
    )
    expected_paths = {
        item["path"]: item
        for item in [*normalized["artifacts"], *normalized["partial_outputs"]]
    }
    if set(artifact_bytes_by_path) != set(expected_paths):
        raise ContractError("output seal bytes differ from the exact declared artifact set")
    for path, artifact in expected_paths.items():
        payload = artifact_bytes_by_path[path]
        if not isinstance(payload, bytes):
            raise ContractError("output seal artifact evidence must be bytes")
        if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
            raise ContractError(f"output artifact hash mismatch: {path}")
        derived_counts = _derive_artifact_counts(
            artifact["role"],
            payload,
            complete=artifact["complete"],
            run_scope=run_scope,
        )
        declared_counts = (
            artifact["row_count"],
            artifact["race_count"],
            artifact["runner_count"],
        )
        if declared_counts != derived_counts:
            raise ContractError(f"output artifact counts differ from exact bytes: {path}")
        resolved = _resolve_repository_path(root.resolve(), path)
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or resolved.read_bytes() != payload
        ):
            raise ContractError(f"output artifact bytes are not durably present: {path}")
    _verify_exact_output_root_contents(
        root,
        run_scope,
        expected_file_paths={
            run_scope["output_sealing_contract"]["execution_receipt_path"],
            *expected_paths,
        },
    )
    return normalized


def seal_output_manifest(
    *,
    root: Path,
    run_scope: dict[str, Any],
    run_scope_digest: str,
    execution_receipt: dict[str, Any],
    result_manifest: dict[str, Any],
    result_manifest_path: str,
    result_manifest_bytes: bytes,
    artifact_bytes_by_path: dict[str, bytes],
    authority_context: dict[str, Any],
) -> dict[str, Any]:
    """Verify the whole seal, then persist its exact manifest with O_EXCL."""

    normalized = verify_output_seal(
        root=root,
        run_scope=run_scope,
        run_scope_digest=run_scope_digest,
        execution_receipt=execution_receipt,
        result_manifest=result_manifest,
        result_manifest_path=result_manifest_path,
        result_manifest_bytes=result_manifest_bytes,
        artifact_bytes_by_path=artifact_bytes_by_path,
        authority_context=authority_context,
    )
    write_authorized_bytes(
        root,
        run_scope,
        phase_id="seal_research_outputs",
        path=result_manifest_path,
        payload=result_manifest_bytes,
        authority_context=authority_context,
    )
    _verify_exact_output_root_contents(
        root,
        run_scope,
        expected_file_paths={
            run_scope["output_sealing_contract"]["execution_receipt_path"],
            result_manifest_path,
            *artifact_bytes_by_path,
        },
    )
    return normalized


__all__ = [
    "ALWAYS_FALSE_CAPABILITIES",
    "CAPABILITY_FIELDS",
    "CAPABILITY_PROFILES",
    "ContractError",
    "ENVIRONMENT_LOCK_SCHEMA_VERSION",
    "EXECUTION_RECEIPT_SCHEMA_VERSION",
    "INPUT_MANIFEST_SCHEMA_VERSION",
    "OUTPUT_ATTESTATION_SCHEMA_VERSION",
    "RESULT_MANIFEST_SCHEMA_VERSION",
    "RUN_FIELDS",
    "RUN_SCOPE_SCHEMA_VERSION",
    "canonical_result_manifest_bytes",
    "dispatch_ordinary_run_scope",
    "hash_bound_repository_paths",
    "issue_execution_receipt",
    "load_frozen_ordinary_run_scope",
    "normalize_execution_receipt",
    "normalize_output_attestation",
    "normalize_ordinary_real_data_run_scope",
    "normalize_result_manifest",
    "observe_process_argv",
    "observe_runtime_environment",
    "read_authorized_bytes",
    "seal_output_manifest",
    "validate_capability_profile",
    "validate_output_root_fresh",
    "verify_access_request",
    "verify_execution_worktree_state",
    "verify_metadata_preflight",
    "verify_ordinary_real_data_run_materials",
    "verify_output_seal",
    "verify_real_data_authorization",
    "verify_runtime_interpreter_isolation",
    "write_authorized_bytes",
]
