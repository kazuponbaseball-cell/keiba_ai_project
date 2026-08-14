from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path, PurePosixPath
from typing import Any, Callable


GATE_KIND = "roi_reproduction_audit_v2"
GATE_CONTRACT_VERSION = 2
EXECUTION_KIND = "historical_reproduction_v2"
IMPLEMENTATION_LAYER = "G1_CONTRACT_COMPILER_ONLY"
IMPLEMENTATION_STATUS = "EXECUTION_FORBIDDEN"
CATALOG_EVENT_KIND = "ROI_REPRODUCTION_CATALOG_PUBLICATION_EVENT"
CATALOG_GATE_KIND = "roi_reproduction_reference_catalog"
CATALOG_CONTRACT_VERSION = 1
CATALOG_RELEASE_STATUS_EVENT_KIND = (
    "ROI_REPRODUCTION_CATALOG_RELEASE_STATUS_EVENT"
)

POLICY_SCHEMA_VERSION = 1
PROPOSAL_SCHEMA_VERSION = 2
RUN_SCHEMA_VERSION = 2
RESULT_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = 1
QUEUE_SCHEMA_VERSION = 4
EVENT_SCHEMA_VERSION = 4

POLICY_PATH = "research/ROI_REPRODUCTION_GATE_V2.json"
POLICY_CANONICAL_SHA256 = (
    "dfd2ffe479bdda81409b9952f69875345294ed344e9830fcec3ab30962d2b724"
)
SCHEMA_PATHS = {
    "proposal": "research/schemas/roi_reproduction_proposal_v2.schema.json",
    "catalog_publication_scope": "research/schemas/roi_reproduction_reference_catalog_publication_scope_v1.schema.json",
    "catalog_publication_provider_lease": "research/schemas/roi_reproduction_catalog_publication_provider_lease_v1.schema.json",
    "catalog_release": "research/schemas/roi_reproduction_reference_catalog_release_v1.schema.json",
    "catalog_entry": "research/schemas/roi_reproduction_reference_catalog_entry_v1.schema.json",
    "catalog_entry_ref": "research/schemas/roi_reproduction_reference_catalog_entry_ref_v1.schema.json",
    "run": "research/schemas/roi_reproduction_run_v2.schema.json",
    "result": "research/schemas/roi_reproduction_result_v1.schema.json",
    "review": "research/schemas/roi_reproduction_review_v1.schema.json",
    "queue": "research/schemas/roi_reproduction_queue_v4.schema.json",
    "execution_lease": "research/schemas/roi_reproduction_execution_lease_v1.schema.json",
    "lease_operation_record": "research/schemas/roi_reproduction_lease_operation_record_v1.schema.json",
    "durable_ledger_receipt": "research/schemas/roi_reproduction_durable_ledger_receipt_v1.schema.json",
    "catalog_publication_receipt": "research/schemas/roi_reproduction_catalog_publication_receipt_v1.schema.json",
    "catalog_publication_event": "research/schemas/roi_reproduction_catalog_publication_event_v1.schema.json",
    "catalog_release_status_event": "research/schemas/roi_reproduction_catalog_release_status_event_v1.schema.json",
    "registry_event": "research/schemas/roi_reproduction_registry_event_v4.schema.json",
}

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,127}$")
SAFE_REPOSITORY_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")

CAPABILITY_FIELDS = (
    "actual_codex_dispatch",
    "automatic_execution",
    "automatic_github_approval",
    "synthetic_fixture_tests",
    "real_data_execution",
    "supervised_manifest_hash_read",
    "raw_row_output",
    "model_real_data_access",
    "historical_result_label_access",
    "historical_training",
    "historical_replay",
    "prospective_outer_oos",
    "backtest_interpretation",
    "price_access",
    "payoff_access",
    "roi_calculation",
    "offline_unit_notional_evaluation",
    "research_candidate_policy_change",
    "production_candidate_policy_change",
    "workload_network_access",
    "workload_external_api_calls",
    "credential_access",
    "purchase_path_access",
    "production_change",
    "shadow_approval",
    "merge",
    "notification_side_effects",
    "order_side_effects",
    "formal_buy",
    "send_order",
    "stake",
)

G1_CAPABILITIES: dict[str, bool | int] = {
    name: (0 if name == "stake" else False) for name in CAPABILITY_FIELDS
}


def _frozen_phase_flags(**enabled: bool) -> dict[str, bool | int]:
    flags = dict(G1_CAPABILITIES)
    for name, value in enabled.items():
        if name not in flags or name == "stake" or value is not True:
            raise ValueError("invalid frozen phase capability declaration")
        flags[name] = True
    return flags


EXPECTED_PHASE_CAPABILITIES = {
    "design_g1": {
        "mode": "design_only",
        "effective_after": "never_in_this_design",
        "flags": dict(G1_CAPABILITIES),
    },
    "preparation_a0": {
        "mode": "preparation_synthetic_only",
        "effective_after": (
            "durable_ledger_prepare_grant_reservation_and_fresh_revalidation"
        ),
        "flags": _frozen_phase_flags(synthetic_fixture_tests=True),
    },
    "catalog_release_maintenance": {
        "mode": "reference_catalog_release_metadata_only",
        "effective_after": (
            "separate_catalog_release_grant_and_one_shot_publication_receipt"
        ),
        "flags": _frozen_phase_flags(
            real_data_execution=True,
            supervised_manifest_hash_read=True,
        ),
    },
    "catalog_binding": {
        "mode": "prebound_catalog_entry_revalidation_only",
        "effective_after": (
            "proposal_bound_catalog_release_and_entry_digest_revalidated"
        ),
        "flags": dict(G1_CAPABILITIES),
    },
    "historical_reproduction": {
        "mode": "historical_reproduction_only",
        "effective_after": (
            "durable_ledger_run_grant_reservation_and_all_required_grants_"
            "catalog_and_replica_lease_revalidated"
        ),
        "flags": _frozen_phase_flags(
            real_data_execution=True,
            model_real_data_access=True,
            historical_result_label_access=True,
            historical_training=True,
            historical_replay=True,
        ),
    },
    "review": {
        "mode": "review_only",
        "effective_after": "validated_result_manifest_and_fresh_acknowledgement",
        "flags": dict(G1_CAPABILITIES),
    },
}

SCORE_FIELDS = {
    "independent_information",
    "racing_mechanism",
    "outer_oos_failure_evidence",
    "leakage_safety",
    "minimal_falsifiability",
    "acquisition_implementation_cost",
}
SCORE_MAXIMA = {
    "independent_information": 25,
    "racing_mechanism": 20,
    "outer_oos_failure_evidence": 20,
    "leakage_safety": 15,
    "minimal_falsifiability": 10,
    "acquisition_implementation_cost": 10,
}
M0_SCORE_COMPONENTS = {
    "independent_information": 0,
    "racing_mechanism": 0,
    "outer_oos_failure_evidence": 0,
    "leakage_safety": 8,
    "minimal_falsifiability": 10,
    "acquisition_implementation_cost": 5,
}

POLICY_FIELDS = {
    "schema_version",
    "gate_kind",
    "gate_contract_version",
    "execution_kind",
    "implementation_layer",
    "implementation_status",
    "authority",
    "policy_update_mode",
    "queue_schema_version",
    "event_schema_version",
    "schema_paths",
    "g1_bootstrap_exact_changed_paths",
    "g1_forbidden_changed_path_prefixes",
    "required_hard_checks",
    "route_to_strategy_gate_markers",
    "allowed_legacy_run_modes",
    "allowed_probability_stages",
    "allowed_result_outcomes",
    "capability_field_set",
    "phase_capabilities",
    "strategy_score_record",
    "safety",
}

PROPOSAL_FIELDS = {
    "schema_version",
    "experiment_id",
    "gate_kind",
    "gate_contract_version",
    "execution_kind",
    "policy_ref",
    "schema_ref",
    "title",
    "hypothesis",
    "null_hypothesis",
    "racing_mechanism",
    "non_promotion_purpose",
    "brain_provenance",
    "repository_contract",
    "target_population",
    "in_scope",
    "out_of_scope",
    "expected_changed_paths",
    "raw_data_sources",
    "data_as_of",
    "allowed_columns",
    "forbidden_columns",
    "lineage_hash_requirements",
    "chronological_fold_design",
    "fold_manifest_ref",
    "purge_embargo",
    "reference_catalog_contract_refs",
    "reference_catalog_entry_ref",
    "model_identity_contract",
    "primary_metric",
    "secondary_metrics",
    "required_effect",
    "rejection_gate",
    "stop_conditions",
    "compute_budget",
    "allowed_variant_count",
    "allowed_threshold_search_count",
    "score_components",
    "eligibility_contract",
    "formal_buy",
    "send_order",
    "stake",
    "safety",
}

RUN_FIELDS = {
    "schema_version",
    "experiment_id",
    "gate_kind",
    "gate_contract_version",
    "execution_kind",
    "policy_ref",
    "schema_ref",
    "proposal_scope",
    "proposal_scope_digest",
    "reference_catalog_entry_ref",
    "execution_commit_sha",
    "config_refs",
    "data_input_manifest_refs",
    "source_lineage_manifest_ref",
    "fold_manifest_ref",
    "runner_universe_manifest_ref",
    "feature_lineage_manifest_ref",
    "target_label_manifest_ref",
    "model_recipe_manifest_ref",
    "reference_artifact_manifest_ref",
    "canonicalization_manifest_ref",
    "dependency_environment_manifest_ref",
    "seed",
    "exact_execution_commands",
    "replicate_contract",
    "conditional_policy_refs",
    "capabilities",
    "formal_buy",
    "send_order",
    "stake",
}

RESULT_FIELDS = {
    "schema_version",
    "result_kind",
    "experiment_id",
    "gate_kind",
    "gate_contract_version",
    "execution_kind",
    "policy_ref",
    "schema_ref",
    "proposal_scope_digest",
    "run_scope_digest",
    "execution_commit_sha",
    "computed_outcome",
    "replicas",
    "determinism_check",
    "runner_label_contract_check",
    "probability_contract_check",
    "numeric_equivalence",
    "artifact_refs",
    "safety",
    "limitations",
    "formal_buy",
    "send_order",
    "stake",
}

REVIEW_FIELDS = {
    "schema_version",
    "experiment_id",
    "gate_kind",
    "gate_contract_version",
    "execution_kind",
    "policy_ref",
    "schema_ref",
    "proposal_scope_digest",
    "run_scope_digest",
    "validated_result_digest",
    "computed_outcome",
    "proposed_terminal_status",
    "review_limitations",
    "formal_buy",
    "send_order",
    "stake",
}

QUEUE_FIELDS = {
    "schema_version",
    "experiment_id",
    "gate_kind",
    "gate_contract_version",
    "execution_kind",
    "policy_ref",
    "schema_ref",
    "proposal_scope_path",
    "proposal_scope_digest",
    "status",
    "capabilities",
    "capability_digest",
    "created_at",
    "formal_buy",
    "send_order",
    "stake",
}

REGISTRY_EVENT_FIELDS = {
    "schema_version",
    "event_id",
    "global_sequence",
    "experiment_sequence",
    "experiment_id",
    "gate_kind",
    "gate_contract_version",
    "policy_ref",
    "schema_ref",
    "status",
    "previous_status",
    "previous_experiment_event_id",
    "previous_experiment_event_digest",
    "occurred_at",
    "observer_actor",
    "proposal_scope_digest",
    "reference_catalog_release_digest",
    "reference_catalog_entry_digest",
    "run_scope_digest",
    "review_digest",
    "root_of_trust_evidence",
    "github_trust_evidence",
    "durable_ledger_evidence",
    "legacy_registry_import_evidence",
    "approval_grant_evidence",
    "revalidated_approval_evidence",
    "capabilities",
    "capability_digest",
    "execution_kind",
    "execution_lease_receipt",
    "result_evidence",
    "paths",
    "artifacts",
    "notes",
    "safety",
}

EXPERIMENT_STATES = (
    "PROPOSED",
    "APPROVED_TO_PREPARE",
    "PREPARING",
    "CATALOG_BOUND",
    "RUN_APPROVAL_REQUIRED",
    "APPROVED_TO_RUN",
    "RUNNING",
    "REVIEW_REQUIRED",
    "ACKNOWLEDGED_REPRODUCTION_RESULT",
    "REPRODUCED",
    "RECONSTRUCTED_NOT_REPRODUCED",
    "REPRODUCTION_FAILED",
    "REJECTED",
    "INVALID",
)
EXPERIMENT_TERMINAL = {
    "REPRODUCED",
    "RECONSTRUCTED_NOT_REPRODUCED",
    "REPRODUCTION_FAILED",
    "REJECTED",
    "INVALID",
}
EXPERIMENT_TRANSITIONS = {
    "PROPOSED": {"APPROVED_TO_PREPARE", "INVALID"},
    "APPROVED_TO_PREPARE": {"PREPARING", "INVALID"},
    "PREPARING": {"CATALOG_BOUND", "INVALID"},
    "CATALOG_BOUND": {"RUN_APPROVAL_REQUIRED", "INVALID"},
    "RUN_APPROVAL_REQUIRED": {"APPROVED_TO_RUN", "INVALID"},
    "APPROVED_TO_RUN": {"RUNNING", "INVALID"},
    "RUNNING": {"REVIEW_REQUIRED", "INVALID"},
    "REVIEW_REQUIRED": {"ACKNOWLEDGED_REPRODUCTION_RESULT", "INVALID"},
    "ACKNOWLEDGED_REPRODUCTION_RESULT": {
        "REPRODUCED",
        "RECONSTRUCTED_NOT_REPRODUCED",
        "REPRODUCTION_FAILED",
        "REJECTED",
        "INVALID",
    },
    **{status: set() for status in EXPERIMENT_TERMINAL},
}

CATALOG_STATES = (
    "CATALOG_PUBLICATION_SCOPE_PROPOSED",
    "APPROVED_TO_PUBLISH_REFERENCE_CATALOG",
    "CATALOG_PUBLISHING",
    "CATALOG_PUBLISHED",
    "CATALOG_PUBLICATION_FAILED",
    "INVALID",
)
CATALOG_TRANSITIONS = {
    "CATALOG_PUBLICATION_SCOPE_PROPOSED": {
        "APPROVED_TO_PUBLISH_REFERENCE_CATALOG",
        "INVALID",
    },
    "APPROVED_TO_PUBLISH_REFERENCE_CATALOG": {"CATALOG_PUBLISHING", "INVALID"},
    "CATALOG_PUBLISHING": {
        "CATALOG_PUBLISHED",
        "CATALOG_PUBLICATION_FAILED",
        "INVALID",
    },
    "CATALOG_PUBLISHED": set(),
    "CATALOG_PUBLICATION_FAILED": set(),
    "INVALID": set(),
}
RELEASE_STATUS_TRANSITIONS = {
    "INITIAL": {"ACTIVE"},
    "ACTIVE": {"REVOKED"},
    "REVOKED": set(),
}

EXPERIMENT_EVENT_EVIDENCE_REQUIREMENTS = {
    "PROPOSED": ((), ()),
    "APPROVED_TO_PREPARE": (("APPROVED_TO_PREPARE",), ()),
    "PREPARING": ((), ("APPROVED_TO_PREPARE",)),
    "CATALOG_BOUND": ((), ("APPROVED_TO_PREPARE",)),
    "RUN_APPROVAL_REQUIRED": ((), ("APPROVED_TO_PREPARE",)),
    "APPROVED_TO_RUN": (
        ("APPROVED_TO_RUN",),
        ("APPROVED_TO_PREPARE",),
    ),
    "RUNNING": ((), ("APPROVED_TO_PREPARE", "APPROVED_TO_RUN")),
    "REVIEW_REQUIRED": ((), ("APPROVED_TO_PREPARE", "APPROVED_TO_RUN")),
    "ACKNOWLEDGED_REPRODUCTION_RESULT": (
        ("ACKNOWLEDGED_REPRODUCTION_RESULT",),
        ("APPROVED_TO_PREPARE", "APPROVED_TO_RUN"),
    ),
    "REPRODUCED": (
        (),
        (
            "ACKNOWLEDGED_REPRODUCTION_RESULT",
            "APPROVED_TO_PREPARE",
            "APPROVED_TO_RUN",
        ),
    ),
    "RECONSTRUCTED_NOT_REPRODUCED": (
        (),
        (
            "ACKNOWLEDGED_REPRODUCTION_RESULT",
            "APPROVED_TO_PREPARE",
            "APPROVED_TO_RUN",
        ),
    ),
    "REPRODUCTION_FAILED": (
        (),
        (
            "ACKNOWLEDGED_REPRODUCTION_RESULT",
            "APPROVED_TO_PREPARE",
            "APPROVED_TO_RUN",
        ),
    ),
    "REJECTED": (
        (),
        (
            "ACKNOWLEDGED_REPRODUCTION_RESULT",
            "APPROVED_TO_PREPARE",
            "APPROVED_TO_RUN",
        ),
    ),
}

CATALOG_EVENT_EVIDENCE_REQUIREMENTS = {
    "CATALOG_PUBLICATION_SCOPE_PROPOSED": ((), ()),
    "APPROVED_TO_PUBLISH_REFERENCE_CATALOG": (
        ("APPROVED_TO_PUBLISH_REFERENCE_CATALOG",),
        (),
    ),
    "CATALOG_PUBLISHING": (
        (),
        ("APPROVED_TO_PUBLISH_REFERENCE_CATALOG",),
    ),
    "CATALOG_PUBLISHED": (
        (),
        ("APPROVED_TO_PUBLISH_REFERENCE_CATALOG",),
    ),
    "CATALOG_PUBLICATION_FAILED": (
        (),
        ("APPROVED_TO_PUBLISH_REFERENCE_CATALOG",),
    ),
}


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def strict_json_loads_v2(text: str, *, label: str = "JSON") -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=_strict_object_pairs,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read strict JSON from {label}: {exc}") from exc


def strict_json_load_v2(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read strict JSON from {path}: {exc}") from exc
    return strict_json_loads_v2(text, label=str(path))


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN and Infinity are forbidden")
        return 0.0 if value == 0.0 else value
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _canonical_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported canonical JSON value type: {type(value).__name__}")


def canonical_json_bytes_v2(value: Any) -> bytes:
    normalized = _canonical_value(value)
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_text_v2(value: Any) -> str:
    return canonical_json_bytes_v2(value).decode("utf-8")


def canonical_digest_v2(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_v2(value)).hexdigest()


def canonical_float64_hex(value: float) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("canonical float must be numeric and not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("canonical float must be finite")
    if number == 0.0:
        number = 0.0
    return struct.pack(">d", number).hex()


def sha256_file_v2(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    canonical_json_bytes_v2(value)
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    payload = _object(value, label)
    missing = sorted(fields - set(payload))
    extra = sorted(set(payload) - fields)
    if missing:
        raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"{label} contains unexpected field(s): {', '.join(extra)}")
    return payload


def _same_json_type_and_value(observed: Any, expected: Any) -> bool:
    """Compare frozen JSON values without Python's bool/int aliasing."""
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _same_json_type_and_value(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _same_json_type_and_value(left, right)
            for left, right in zip(observed, expected)
        )
    return observed == expected


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty string without outer whitespace")
    return value


def _identifier(value: Any, label: str) -> str:
    text = _string(value, label)
    if not SAFE_IDENTIFIER.fullmatch(text):
        raise ValueError(f"{label} must be a safe 3-128 character identifier")
    return text


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if not FULL_SHA256.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase full SHA-256 digest")
    return text


def _git_sha(value: Any, label: str) -> str:
    text = _string(value, label)
    if not FULL_GIT_SHA.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase full Git SHA")
    return text


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _repository_path(value: Any, label: str) -> str:
    text = _string(value, label)
    if "\\" in text or ":" in text or not text.isascii() or not SAFE_REPOSITORY_PATH.fullmatch(text):
        raise ValueError(f"{label} must be an ASCII POSIX repository-relative path")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a canonical repository-relative path")
    if any(part.endswith((".", " ")) for part in path.parts):
        raise ValueError(f"{label} contains a forbidden trailing dot or space")
    return path.as_posix()


def _semantic_strings(
    value: Any,
    label: str,
    *,
    allow_empty: bool = False,
    paths: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label} must be {'a' if allow_empty else 'a non-empty'} list")
    normalize: Callable[[Any, str], str] = _repository_path if paths else _string
    normalized = [normalize(item, f"{label}[]") for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} contains duplicate values")
    folded = [item.casefold() for item in normalized]
    if len(folded) != len(set(folded)):
        raise ValueError(f"{label} contains case-fold aliases")
    return sorted(normalized)


def _ordered_strings(value: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label} must be {'a' if allow_empty else 'a non-empty'} list")
    normalized = [_string(item, f"{label}[]") for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} contains duplicate values")
    return normalized


def normalize_ref(value: Any, label: str = "reference") -> dict[str, str]:
    payload = _exact(value, {"path", "sha256"}, label)
    return {
        "path": _repository_path(payload["path"], f"{label}.path"),
        "sha256": _sha256(payload["sha256"], f"{label}.sha256"),
    }


def _normalize_refs(value: Any, label: str, *, allow_empty: bool = False) -> list[dict[str, str]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label} must be {'a' if allow_empty else 'a non-empty'} list of references")
    refs = [normalize_ref(item, f"{label}[]") for item in value]
    paths = [item["path"] for item in refs]
    if len(paths) != len(set(paths)) or len(paths) != len({path.casefold() for path in paths}):
        raise ValueError(f"{label} contains duplicate or case-fold aliased paths")
    return sorted(refs, key=lambda item: item["path"])


def _normalize_proposal_contract_ref(
    value: Any,
    label: str,
    *,
    policy: dict[str, Any],
) -> dict[str, str]:
    reference = normalize_ref(value, label)
    path = reference["path"]
    allowed_prefixes = (
        "research/contracts/roi_reproduction/",
        "research/manifests/roi_reproduction/",
        "research/synthetic/roi_reproduction/",
    )
    if not any(path.startswith(prefix) for prefix in allowed_prefixes):
        raise ValueError(f"{label}.path is outside the isolated ROI reproduction contract roots")
    if _path_has_forbidden_prefix(path, policy) or _contains_market_or_promotion_marker(path, policy):
        raise ValueError(f"{label}.path contains a forbidden route marker")
    return reference


def _require_safety_constants(payload: dict[str, Any], label: str) -> None:
    if payload.get("formal_buy") is not False:
        raise ValueError(f"{label}.formal_buy must be false")
    if payload.get("send_order") is not False:
        raise ValueError(f"{label}.send_order must be false")
    if isinstance(payload.get("stake"), bool) or payload.get("stake") != 0:
        raise ValueError(f"{label}.stake must be integer 0")


def normalize_capabilities(value: Any, *, label: str = "capabilities") -> dict[str, bool | int]:
    payload = _exact(value, set(CAPABILITY_FIELDS), label)
    normalized: dict[str, bool | int] = {}
    for field in CAPABILITY_FIELDS:
        item = payload[field]
        if field == "stake":
            if isinstance(item, bool) or item != 0:
                raise ValueError(f"{label}.stake must be integer 0")
            normalized[field] = 0
        else:
            if not isinstance(item, bool):
                raise ValueError(f"{label}.{field} must be boolean")
            normalized[field] = item
    if normalized["formal_buy"] or normalized["send_order"]:
        raise ValueError(f"{label} cannot enable formal BUY or send order")
    return normalized


def _normalize_phase_capability(value: Any, label: str) -> dict[str, Any]:
    payload = _exact(value, {"mode", "effective_after", "flags"}, label)
    return {
        "mode": _string(payload["mode"], f"{label}.mode"),
        "effective_after": _string(payload["effective_after"], f"{label}.effective_after"),
        "flags": normalize_capabilities(payload["flags"], label=f"{label}.flags"),
    }


def _normalize_strategy_score_record(value: Any) -> dict[str, Any]:
    fields = {
        "applicable",
        "recorded_total",
        "credit",
        "threshold_override_allowed",
        "source_ref",
    }
    payload = _exact(value, fields, "policy.strategy_score_record")
    if payload["applicable"] is not False:
        raise ValueError("strategy score must be non-applicable")
    if isinstance(payload["recorded_total"], bool) or payload["recorded_total"] != 23:
        raise ValueError("strategy score recorded_total must be 23")
    if isinstance(payload["credit"], bool) or payload["credit"] != 0:
        raise ValueError("strategy score credit must be 0")
    if payload["threshold_override_allowed"] is not False:
        raise ValueError("strategy score threshold override must be false")
    return {
        "applicable": False,
        "recorded_total": 23,
        "credit": 0,
        "threshold_override_allowed": False,
        "source_ref": normalize_ref(payload["source_ref"], "policy.strategy_score_record.source_ref"),
    }


def normalize_policy_v2(value: Any) -> dict[str, Any]:
    payload = _exact(value, POLICY_FIELDS, "ROI reproduction policy")
    constants = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "execution_kind": EXECUTION_KIND,
        "implementation_layer": IMPLEMENTATION_LAYER,
        "implementation_status": IMPLEMENTATION_STATUS,
        "authority": False,
        "policy_update_mode": "NEW_KIND_VERSION_AND_PATH_ONLY",
        "queue_schema_version": QUEUE_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
    }
    for field, expected in constants.items():
        if not _same_json_type_and_value(payload[field], expected):
            raise ValueError(f"policy.{field} must equal {expected!r}")
    schema_paths = _exact(payload["schema_paths"], set(SCHEMA_PATHS), "policy.schema_paths")
    normalized_paths = {
        key: _repository_path(schema_paths[key], f"policy.schema_paths.{key}")
        for key in sorted(SCHEMA_PATHS)
    }
    if normalized_paths != {key: SCHEMA_PATHS[key] for key in sorted(SCHEMA_PATHS)}:
        raise ValueError("policy.schema_paths must equal the code-owned G1 schema paths")
    capability_fields = _ordered_strings(payload["capability_field_set"], "policy.capability_field_set")
    if capability_fields != list(CAPABILITY_FIELDS):
        raise ValueError("policy.capability_field_set must equal the exact code-owned field order")
    phase_payload = _object(payload["phase_capabilities"], "policy.phase_capabilities")
    expected_phases = set(EXPECTED_PHASE_CAPABILITIES)
    _exact(phase_payload, expected_phases, "policy.phase_capabilities")
    phases = {
        name: _normalize_phase_capability(phase_payload[name], f"policy.phase_capabilities.{name}")
        for name in sorted(expected_phases)
    }
    if phases != EXPECTED_PHASE_CAPABILITIES:
        raise ValueError("policy phase capability maps differ from the frozen G1/G2 design")
    safety = normalize_capabilities(payload["safety"], label="policy.safety")
    if safety != G1_CAPABILITIES:
        raise ValueError("policy safety must keep every G1 capability false/0")
    normalized = {
        **constants,
        "schema_paths": normalized_paths,
        "g1_bootstrap_exact_changed_paths": _semantic_strings(
            payload["g1_bootstrap_exact_changed_paths"],
            "policy.g1_bootstrap_exact_changed_paths",
            paths=True,
        ),
        "g1_forbidden_changed_path_prefixes": _semantic_strings(
            payload["g1_forbidden_changed_path_prefixes"],
            "policy.g1_forbidden_changed_path_prefixes",
            paths=True,
        ),
        "required_hard_checks": _semantic_strings(
            payload["required_hard_checks"], "policy.required_hard_checks"
        ),
        "route_to_strategy_gate_markers": _semantic_strings(
            payload["route_to_strategy_gate_markers"],
            "policy.route_to_strategy_gate_markers",
        ),
        "allowed_legacy_run_modes": _semantic_strings(
            payload["allowed_legacy_run_modes"], "policy.allowed_legacy_run_modes"
        ),
        "allowed_probability_stages": _semantic_strings(
            payload["allowed_probability_stages"], "policy.allowed_probability_stages"
        ),
        "allowed_result_outcomes": _semantic_strings(
            payload["allowed_result_outcomes"], "policy.allowed_result_outcomes"
        ),
        "capability_field_set": capability_fields,
        "phase_capabilities": phases,
        "strategy_score_record": _normalize_strategy_score_record(payload["strategy_score_record"]),
        "safety": safety,
    }
    if canonical_digest_v2(normalized) != POLICY_CANONICAL_SHA256:
        raise ValueError("policy content differs from the frozen G1 root of trust")
    return normalized


def load_policy_v2(path: Path) -> tuple[dict[str, Any], str]:
    policy = normalize_policy_v2(strict_json_load_v2(path))
    return policy, sha256_file_v2(path)


def _bind_artifact_identity(
    payload: dict[str, Any],
    *,
    schema_version: int,
    schema_kind: str,
    policy_sha256: str | None,
    schema_sha256: str | None,
    label: str,
) -> dict[str, Any]:
    if isinstance(payload["schema_version"], bool) or payload["schema_version"] != schema_version:
        raise ValueError(f"{label}.schema_version must be integer {schema_version}")
    if payload["gate_kind"] != GATE_KIND:
        raise ValueError(f"{label}.gate_kind must be {GATE_KIND}")
    if (
        isinstance(payload["gate_contract_version"], bool)
        or payload["gate_contract_version"] != GATE_CONTRACT_VERSION
    ):
        raise ValueError(f"{label}.gate_contract_version must be integer {GATE_CONTRACT_VERSION}")
    if "execution_kind" in payload and payload["execution_kind"] != EXECUTION_KIND:
        raise ValueError(f"{label}.execution_kind must be {EXECUTION_KIND}")
    policy_ref = normalize_ref(payload["policy_ref"], f"{label}.policy_ref")
    schema_ref = normalize_ref(payload["schema_ref"], f"{label}.schema_ref")
    if policy_ref["path"] != POLICY_PATH:
        raise ValueError(f"{label}.policy_ref.path must be {POLICY_PATH}")
    if schema_ref["path"] != SCHEMA_PATHS[schema_kind]:
        raise ValueError(
            f"{label}.schema_ref.path must be {SCHEMA_PATHS[schema_kind]}"
        )
    if policy_sha256 is not None and policy_ref["sha256"] != _sha256(
        policy_sha256, "policy_sha256"
    ):
        raise ValueError(f"{label}.policy_ref.sha256 does not match the loaded policy")
    if schema_sha256 is not None and schema_ref["sha256"] != _sha256(
        schema_sha256, "schema_sha256"
    ):
        raise ValueError(f"{label}.schema_ref.sha256 does not match the code-owned schema")
    return {"policy_ref": policy_ref, "schema_ref": schema_ref}


def _normalize_brain_provenance(value: Any, policy: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "source_kind",
        "provider",
        "model_id",
        "transfer_mode",
        "prompt_sha256",
        "sanitized_context_manifest_ref",
        "response_sha256",
    }
    payload = _exact(value, fields, "proposal.brain_provenance")
    constants = {
        "source_kind": "external_ai",
        "provider": "OpenAI",
        "transfer_mode": "manual",
    }
    for field, expected in constants.items():
        if payload[field] != expected:
            raise ValueError(f"proposal.brain_provenance.{field} must be {expected!r}")
    return {
        **constants,
        "model_id": _string(payload["model_id"], "proposal.brain_provenance.model_id"),
        "prompt_sha256": _sha256(
            payload["prompt_sha256"], "proposal.brain_provenance.prompt_sha256"
        ),
        "sanitized_context_manifest_ref": _normalize_proposal_contract_ref(
            payload["sanitized_context_manifest_ref"],
            "proposal.brain_provenance.sanitized_context_manifest_ref",
            policy=policy,
        ),
        "response_sha256": _sha256(
            payload["response_sha256"], "proposal.brain_provenance.response_sha256"
        ),
    }


def _normalize_repository_contract(value: Any, label: str) -> dict[str, str]:
    payload = _exact(value, {"repository", "base_branch", "base_commit"}, label)
    if payload["repository"] != "kazuponbaseball-cell/keiba_ai_project":
        raise ValueError(f"{label}.repository must name the code-owned repository")
    if payload["base_branch"] != "main":
        raise ValueError(f"{label}.base_branch must be main")
    return {
        "repository": "kazuponbaseball-cell/keiba_ai_project",
        "base_branch": "main",
        "base_commit": _git_sha(payload["base_commit"], f"{label}.base_commit"),
    }


def _normalize_catalog_entry_ref(value: Any, label: str) -> dict[str, Any]:
    fields = {
        "catalog_release_id",
        "catalog_release_digest",
        "entry_id",
        "entry_digest",
        "formal_buy",
        "send_order",
        "stake",
    }
    payload = _exact(value, fields, label)
    _require_safety_constants(payload, label)
    return {
        "catalog_release_id": _identifier(
            payload["catalog_release_id"], f"{label}.catalog_release_id"
        ),
        "catalog_release_digest": _sha256(
            payload["catalog_release_digest"], f"{label}.catalog_release_digest"
        ),
        "entry_id": _identifier(payload["entry_id"], f"{label}.entry_id"),
        "entry_digest": _sha256(payload["entry_digest"], f"{label}.entry_digest"),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def _normalize_catalog_contract_refs(
    value: Any, policy: dict[str, Any]
) -> dict[str, dict[str, str]]:
    fields = {
        "column_policy",
        "feature_lineage",
        "target_label",
        "fold_design",
        "runner_universe_policy",
        "model_recipe",
        "canonicalization",
        "environment_template",
        "reference_artifact_contract",
    }
    payload = _exact(value, fields, "proposal.reference_catalog_contract_refs")
    return {
        name: _normalize_proposal_contract_ref(
            payload[name],
            f"proposal.reference_catalog_contract_refs.{name}",
            policy=policy,
        )
        for name in sorted(fields)
    }


def _normalize_model_identity(value: Any, policy: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "legacy_reference_status",
        "identity_domain_resolution_count",
        "legacy_run_mode",
        "canonical_model_name",
        "canonical_probability_stage",
        "comparison_artifact_ids",
        "model_recipe_manifest_ref",
        "post_reference_changes_allowed",
        "challenger_present",
    }
    payload = _exact(value, fields, "proposal.model_identity_contract")
    if payload["legacy_reference_status"] != "CATALOG_RESOLVED_EXACTLY_ONE":
        raise ValueError("model identity must be resolved by the future catalog verifier")
    if (
        isinstance(payload["identity_domain_resolution_count"], bool)
        or payload["identity_domain_resolution_count"] != 1
    ):
        raise ValueError("model identity resolution count must be integer 1")
    run_mode = _string(payload["legacy_run_mode"], "model_identity.legacy_run_mode")
    if run_mode not in policy["allowed_legacy_run_modes"]:
        raise ValueError("legacy_run_mode is outside the frozen finite domain")
    probability_stage = _string(
        payload["canonical_probability_stage"],
        "model_identity.canonical_probability_stage",
    )
    if probability_stage not in policy["allowed_probability_stages"]:
        raise ValueError("canonical_probability_stage is outside the frozen finite domain")
    if payload["post_reference_changes_allowed"] is not False:
        raise ValueError("post-reference recipe or identity changes are forbidden")
    if payload["challenger_present"] is not False:
        raise ValueError("challenger comparison is outside the reproduction gate")
    canonical_model_name = _identifier(
        payload["canonical_model_name"], "model_identity.canonical_model_name"
    )
    if _contains_market_or_promotion_marker(canonical_model_name, policy):
        raise ValueError("canonical model name contains a forbidden strategy marker")
    comparison_artifact_ids = _semantic_strings(
        payload["comparison_artifact_ids"], "model_identity.comparison_artifact_ids"
    )
    for artifact_id in comparison_artifact_ids:
        _identifier(artifact_id, "model_identity.comparison_artifact_ids[]")
        if _contains_market_or_promotion_marker(artifact_id, policy):
            raise ValueError("comparison artifact ID contains a forbidden strategy marker")
    return {
        "legacy_reference_status": "CATALOG_RESOLVED_EXACTLY_ONE",
        "identity_domain_resolution_count": 1,
        "legacy_run_mode": run_mode,
        "canonical_model_name": canonical_model_name,
        "canonical_probability_stage": probability_stage,
        "comparison_artifact_ids": comparison_artifact_ids,
        "model_recipe_manifest_ref": _normalize_proposal_contract_ref(
            payload["model_recipe_manifest_ref"],
            "model_identity.model_recipe_manifest_ref",
            policy=policy,
        ),
        "post_reference_changes_allowed": False,
        "challenger_present": False,
    }


def _normalize_score_components(value: Any) -> dict[str, int]:
    payload = _exact(value, SCORE_FIELDS, "proposal.score_components")
    normalized: dict[str, int] = {}
    for name in sorted(SCORE_FIELDS):
        item = payload[name]
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"proposal.score_components.{name} must be an integer")
        if not 0 <= item <= SCORE_MAXIMA[name]:
            raise ValueError(f"proposal.score_components.{name} is out of range")
        normalized[name] = item
    if normalized != {name: M0_SCORE_COMPONENTS[name] for name in sorted(SCORE_FIELDS)}:
        raise ValueError("M0 reproduction score components are immutable and must total 23")
    return normalized


def _normalize_eligibility_contract(value: Any) -> dict[str, Any]:
    fields = {
        "eligibility_class",
        "strategy_score_applicable",
        "recorded_strategy_score",
        "strategy_score_credit",
        "score_threshold_override_allowed",
        "non_promotion_only",
        "success_grants_shadow",
        "success_grants_model_change",
        "success_grants_buy",
    }
    payload = _exact(value, fields, "proposal.eligibility_contract")
    expected = {
        "eligibility_class": "nonpromotion_reproduction_audit",
        "strategy_score_applicable": False,
        "recorded_strategy_score": 23,
        "strategy_score_credit": 0,
        "score_threshold_override_allowed": False,
        "non_promotion_only": True,
        "success_grants_shadow": False,
        "success_grants_model_change": False,
        "success_grants_buy": False,
    }
    for name, wanted in expected.items():
        observed = payload[name]
        if not _same_json_type_and_value(observed, wanted):
            raise ValueError(f"proposal.eligibility_contract.{name} must equal {wanted!r}")
    return expected


def _normalize_fold_design(value: Any) -> dict[str, Any]:
    fields = {
        "partition_unit",
        "ordered_partitions",
        "reused_development_test",
        "race_overlap_allowed",
        "prospective_outer_oos",
    }
    payload = _exact(value, fields, "proposal.chronological_fold_design")
    expected = {
        "partition_unit": "race_id",
        "ordered_partitions": ["train", "validation", "calibration", "reused_historical_test"],
        "reused_development_test": True,
        "race_overlap_allowed": False,
        "prospective_outer_oos": False,
    }
    if not _same_json_type_and_value(payload, expected):
        raise ValueError("chronological fold design must equal the frozen reproduction design")
    return expected


def _normalize_purge_embargo(value: Any) -> dict[str, int]:
    payload = _exact(value, {"purge_days", "embargo_days"}, "proposal.purge_embargo")
    return {
        "purge_days": _nonnegative_int(payload["purge_days"], "purge_days"),
        "embargo_days": _nonnegative_int(payload["embargo_days"], "embargo_days"),
    }


def _normalize_primary_metric(value: Any) -> dict[str, Any]:
    payload = _exact(
        value,
        {"name", "comparison", "uses_roi", "promotion_metric"},
        "proposal.primary_metric",
    )
    expected = {
        "name": "m0_reproduction_equivalence",
        "comparison": "canonical_digest_or_preregistered_numeric_tolerance",
        "uses_roi": False,
        "promotion_metric": False,
    }
    if not _same_json_type_and_value(payload, expected):
        raise ValueError("primary_metric must be reproduction-only and non-promotion")
    return expected


def _normalize_required_effect(value: Any) -> dict[str, Any]:
    fields = {
        "two_clean_replicas_required",
        "probability_contract_required",
        "canonical_reference_required_for_reproduced",
        "strategy_score_credit",
        "promotion_authority",
        "roi_calculations",
    }
    payload = _exact(value, fields, "proposal.required_effect")
    expected = {
        "two_clean_replicas_required": True,
        "probability_contract_required": True,
        "canonical_reference_required_for_reproduced": True,
        "strategy_score_credit": 0,
        "promotion_authority": False,
        "roi_calculations": 0,
    }
    for name, wanted in expected.items():
        observed = payload[name]
        if not _same_json_type_and_value(observed, wanted):
            raise ValueError(f"proposal.required_effect.{name} must equal {wanted!r}")
    return expected


def _normalize_compute_budget(value: Any) -> dict[str, int]:
    fields = {
        "maximum_runtime_minutes",
        "replica_count",
        "network_calls",
        "external_api_calls",
        "roi_calculations",
    }
    payload = _exact(value, fields, "proposal.compute_budget")
    runtime = _nonnegative_int(payload["maximum_runtime_minutes"], "maximum_runtime_minutes")
    if not 1 <= runtime <= 1440:
        raise ValueError("maximum_runtime_minutes must be between 1 and 1440")
    for field in ("network_calls", "external_api_calls", "roi_calculations"):
        if isinstance(payload[field], bool) or payload[field] != 0:
            raise ValueError(f"proposal.compute_budget.{field} must be integer 0")
    if isinstance(payload["replica_count"], bool) or payload["replica_count"] != 2:
        raise ValueError("proposal.compute_budget.replica_count must be integer 2")
    return {
        "maximum_runtime_minutes": runtime,
        "replica_count": 2,
        "network_calls": 0,
        "external_api_calls": 0,
        "roi_calculations": 0,
    }


def _normalize_proposal_safety(value: Any) -> dict[str, bool | int]:
    normalized = normalize_capabilities(value, label="proposal.safety")
    if normalized != G1_CAPABILITIES:
        raise ValueError("proposal artifacts grant no current capabilities in G1")
    return normalized


def _path_has_forbidden_prefix(path: str, policy: dict[str, Any]) -> bool:
    folded = path.casefold()
    return any(
        folded == prefix.casefold().rstrip("/")
        or folded.startswith(prefix.casefold().rstrip("/") + "/")
        for prefix in policy["g1_forbidden_changed_path_prefixes"]
    )


def _validate_expected_reproduction_paths(paths: list[str], policy: dict[str, Any]) -> None:
    allowed_prefixes = (
        "research/synthetic/roi_reproduction/",
        "scripts/research/roi_reproduction_experiments/",
    )
    for path in paths:
        if _path_has_forbidden_prefix(path, policy):
            raise ValueError(f"expected_changed_paths contains a forbidden prefix: {path}")
        if not any(path.startswith(prefix) for prefix in allowed_prefixes):
            raise ValueError(
                "future reproduction preparation paths must stay inside an isolated experiment namespace"
            )
        if _contains_market_or_promotion_marker(path, policy):
            raise ValueError(
                f"expected_changed_paths contains a market, strategy, or promotion route marker: {path}"
            )
        tokens = {token for token in re.split(r"[^a-z0-9]+", path.casefold()) if token}
        if "roi" not in tokens or "reproduction" not in tokens:
            raise ValueError(
                "future reproduction preparation paths must be isolated under an explicit roi_reproduction name"
            )


def _contains_market_or_promotion_marker(value: str, policy: dict[str, Any]) -> bool:
    folded = value.casefold()
    if any(marker in folded for marker in ("オッズ", "人気", "回収率", "購入", "賭け", "期待値", "収益")):
        return True
    for namespace in ("roi_reproduction", "roi-reproduction", "roi reproduction"):
        folded = folded.replace(namespace, "")
    tokens = {token for token in re.split(r"[^a-z0-9]+", folded) if token}
    if tokens & {
        "odds",
        "popularity",
        "market",
        "payoff",
        "payout",
        "price",
        "roi",
        "buy",
        "ticket",
        "candidate",
        "challenger",
        "promotion",
        "production",
        "prospective",
        "shadow",
        "stake",
        "order",
        "notification",
        "credential",
        "secret",
        "bet",
        "wager",
        "pnl",
    }:
        return True
    compact = re.sub(r"[^a-z0-9]+", "", folded)
    if any(
        marker in compact
        for marker in (
            "challengercomparison",
            "expectedvalue",
            "newcalibrator",
            "newestimator",
            "newfeature",
            "newloss",
            "newtarget",
            "postreferencerecipechange",
            "postreferencetolerancechange",
            "recipeupdate",
            "returnrate",
            "tolerancechange",
            "valuepolicychange",
        )
    ):
        return True
    return any(
        re.sub(r"[^a-z0-9]+", "", marker.casefold()) in compact
        for marker in policy["route_to_strategy_gate_markers"]
    )


def normalize_proposal_v2(
    value: Any,
    *,
    policy: dict[str, Any],
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
    expected_experiment_id: str | None = None,
) -> dict[str, Any]:
    policy = normalize_policy_v2(policy)
    payload = _exact(value, PROPOSAL_FIELDS, "ROI reproduction proposal")
    identity = _bind_artifact_identity(
        payload,
        schema_version=PROPOSAL_SCHEMA_VERSION,
        schema_kind="proposal",
        policy_sha256=policy_sha256,
        schema_sha256=schema_sha256,
        label="proposal",
    )
    _require_safety_constants(payload, "proposal")
    experiment_id = _identifier(payload["experiment_id"], "proposal.experiment_id")
    if expected_experiment_id is not None and experiment_id != expected_experiment_id:
        raise ValueError("proposal experiment_id does not match the expected ID")
    expected_paths = _semantic_strings(
        payload["expected_changed_paths"], "proposal.expected_changed_paths", paths=True
    )
    _validate_expected_reproduction_paths(expected_paths, policy)
    allowed_columns = _semantic_strings(payload["allowed_columns"], "proposal.allowed_columns")
    expected_columns = sorted(
        [
            "critical_missing_count",
            "min_primary_strength",
            "min_rank_strength",
            "sum_primary_strength",
            "sum_rank_strength",
        ]
    )
    if allowed_columns != expected_columns:
        raise ValueError("proposal.allowed_columns must equal the frozen five-feature M0 allowlist")
    for column in allowed_columns:
        if _contains_market_or_promotion_marker(column, policy):
            raise ValueError(f"allowed model column belongs to a forbidden family: {column}")
    in_scope = _semantic_strings(payload["in_scope"], "proposal.in_scope")
    raw_sources = _semantic_strings(payload["raw_data_sources"], "proposal.raw_data_sources")
    secondary_metrics = _semantic_strings(
        payload["secondary_metrics"], "proposal.secondary_metrics"
    )
    lineage_requirements = _semantic_strings(
        payload["lineage_hash_requirements"], "proposal.lineage_hash_requirements"
    )
    rejection_gate = _semantic_strings(payload["rejection_gate"], "proposal.rejection_gate")
    stop_conditions = _semantic_strings(
        payload["stop_conditions"], "proposal.stop_conditions"
    )
    for text in [
        payload["title"],
        payload["hypothesis"],
        payload["null_hypothesis"],
        payload["racing_mechanism"],
        payload["non_promotion_purpose"],
        payload["target_population"],
        *in_scope,
        *raw_sources,
        *secondary_metrics,
        *lineage_requirements,
        *rejection_gate,
        *stop_conditions,
    ]:
        if _contains_market_or_promotion_marker(_string(text, "proposal scope text"), policy):
            raise ValueError("reproduction scope contains a strategy/market/promotion marker")
    for source in raw_sources:
        if any(character in source for character in ("/", "\\", ":")) or ".." in source:
            raise ValueError("raw_data_sources must be opaque logical IDs, not paths or URLs")
    if raw_sources != ["legacy_m0_snapshot"]:
        raise ValueError("raw_data_sources must equal the single code-owned M0 logical source ID")
    normalized = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "execution_kind": EXECUTION_KIND,
        **identity,
        "title": _string(payload["title"], "proposal.title"),
        "hypothesis": _string(payload["hypothesis"], "proposal.hypothesis"),
        "null_hypothesis": _string(payload["null_hypothesis"], "proposal.null_hypothesis"),
        "racing_mechanism": _string(payload["racing_mechanism"], "proposal.racing_mechanism"),
        "non_promotion_purpose": _string(
            payload["non_promotion_purpose"], "proposal.non_promotion_purpose"
        ),
        "brain_provenance": _normalize_brain_provenance(payload["brain_provenance"], policy),
        "repository_contract": _normalize_repository_contract(
            payload["repository_contract"], "proposal.repository_contract"
        ),
        "target_population": _string(payload["target_population"], "proposal.target_population"),
        "in_scope": in_scope,
        "out_of_scope": _semantic_strings(payload["out_of_scope"], "proposal.out_of_scope"),
        "expected_changed_paths": expected_paths,
        "raw_data_sources": raw_sources,
        "data_as_of": _string(payload["data_as_of"], "proposal.data_as_of"),
        "allowed_columns": allowed_columns,
        "forbidden_columns": _semantic_strings(
            payload["forbidden_columns"], "proposal.forbidden_columns"
        ),
        "lineage_hash_requirements": lineage_requirements,
        "chronological_fold_design": _normalize_fold_design(
            payload["chronological_fold_design"]
        ),
        "fold_manifest_ref": _normalize_proposal_contract_ref(
            payload["fold_manifest_ref"], "proposal.fold_manifest_ref", policy=policy
        ),
        "purge_embargo": _normalize_purge_embargo(payload["purge_embargo"]),
        "reference_catalog_contract_refs": _normalize_catalog_contract_refs(
            payload["reference_catalog_contract_refs"], policy
        ),
        "reference_catalog_entry_ref": _normalize_catalog_entry_ref(
            payload["reference_catalog_entry_ref"], "proposal.reference_catalog_entry_ref"
        ),
        "model_identity_contract": _normalize_model_identity(
            payload["model_identity_contract"], policy
        ),
        "primary_metric": _normalize_primary_metric(payload["primary_metric"]),
        "secondary_metrics": secondary_metrics,
        "required_effect": _normalize_required_effect(payload["required_effect"]),
        "rejection_gate": rejection_gate,
        "stop_conditions": stop_conditions,
        "compute_budget": _normalize_compute_budget(payload["compute_budget"]),
        "allowed_variant_count": payload["allowed_variant_count"],
        "allowed_threshold_search_count": payload["allowed_threshold_search_count"],
        "score_components": _normalize_score_components(payload["score_components"]),
        "eligibility_contract": _normalize_eligibility_contract(payload["eligibility_contract"]),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "safety": _normalize_proposal_safety(payload["safety"]),
    }
    if isinstance(normalized["allowed_variant_count"], bool) or normalized["allowed_variant_count"] != 1:
        raise ValueError("allowed_variant_count must be integer 1")
    if (
        isinstance(normalized["allowed_threshold_search_count"], bool)
        or normalized["allowed_threshold_search_count"] != 0
    ):
        raise ValueError("allowed_threshold_search_count must be integer 0")
    if (
        normalized["reference_catalog_contract_refs"]["model_recipe"]
        != normalized["model_identity_contract"]["model_recipe_manifest_ref"]
    ):
        raise ValueError("proposal model recipe references do not identify the same artifact")
    canonical_json_bytes_v2(normalized)
    return normalized


def evaluate_proposal_v2(
    proposal: dict[str, Any],
    *,
    policy: dict[str, Any],
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_proposal_v2(
        proposal,
        policy=policy,
        policy_sha256=policy_sha256,
        schema_sha256=schema_sha256,
    )
    structural_checks = {
        "gate_identity_bound": True,
        "nonpromotion_reproduction_only": True,
        "strategy_score_non_applicable_recorded_23_credit_0": True,
        "single_variant_no_threshold_search": True,
        "price_payoff_roi_shadow_buy_firewall": True,
        "exactly_one_catalog_identity_declared": True,
        "all_current_capabilities_disabled": True,
    }
    return {
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "proposal_scope_digest": canonical_digest_v2(normalized),
        "structural_checks": structural_checks,
        "structural_contract_passed": all(structural_checks.values()),
        "catalog_verified": False,
        "durable_ledger_verified": False,
        "g2_authority_verifier_available": False,
        "status": "BLOCKED_CATALOG",
        "implementation_status": IMPLEMENTATION_STATUS,
        "proposal_or_queue_creation_allowed": False,
        "execution_authority": False,
        "capabilities": dict(G1_CAPABILITIES),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


CATALOG_ENTRY_FIELDS = {
    "entry_id",
    "logical_object_id",
    "content_sha256",
    "byte_size",
    "schema_sha256",
    "row_count",
    "source_time",
    "event_time",
    "received_time",
    "data_as_of",
    "upstream_lineage_sha256",
    "resolved_model_identity",
    "resolved_input_universe_identity",
    "snapshot_id",
    "formal_buy",
    "send_order",
    "stake",
}

CATALOG_PUBLICATION_SCOPE_FIELDS = {
    "schema_version",
    "catalog_kind",
    "catalog_contract_version",
    "scope_id",
    "gate_kind",
    "gate_contract_version",
    "policy_ref",
    "schema_ref",
    "repository_contract",
    "provider_contract",
    "source_requests",
    "expected_metadata_schema",
    "resource_budget",
    "capabilities",
    "formal_buy",
    "send_order",
    "stake",
}

CATALOG_PROVIDER_LEASE_FIELDS = {
    "lease_id",
    "subject_kind",
    "catalog_publication_scope_id",
    "catalog_publication_scope_digest",
    "catalog_kind",
    "catalog_contract_version",
    "gate_kind",
    "gate_contract_version",
    "phase",
    "capability_digest",
    "command_digest",
    "provider_identity_digest",
    "provider_code_digest",
    "provider_environment_digest",
    "verified_current_main_sha",
    "durable_ledger_backend_identity_digest",
    "durable_ledger_head_sequence",
    "durable_ledger_head_digest",
    "grant_reservation_receipt_digest",
    "policy_digest",
    "schema_digest",
    "github_evidence_digest",
    "issued_at",
    "expires_at",
    "human_supervisor_identity",
    "retry_budget",
    "formal_buy",
    "send_order",
    "stake",
}

CATALOG_RELEASE_FIELDS = {
    "schema_version",
    "catalog_kind",
    "catalog_contract_version",
    "release_id",
    "gate_kind",
    "gate_contract_version",
    "schema_ref",
    "policy_ref",
    "catalog_publication_scope_digest",
    "catalog_publication_approval_evidence_digest",
    "repository_contract",
    "github_trust_evidence",
    "provider_contract",
    "provider_identity",
    "provider_execution_commit",
    "provider_code_sha256",
    "provider_environment_sha256",
    "publication_receipt_digest",
    "catalog_entry_set_digest",
    "logical_entries",
    "lineage_edges",
    "created_at",
    "capabilities",
    "formal_buy",
    "send_order",
    "stake",
}


def _normalize_digest_evidence(value: Any, label: str) -> dict[str, str]:
    payload = _exact(value, {"evidence_kind", "digest"}, label)
    return {
        "evidence_kind": _string(payload["evidence_kind"], f"{label}.evidence_kind"),
        "digest": _sha256(payload["digest"], f"{label}.digest"),
    }


def _normalize_optional_digest_evidence(value: Any, label: str) -> dict[str, str] | None:
    if value is None:
        return None
    return _normalize_digest_evidence(value, label)


def normalize_catalog_entry_v1(value: Any, *, policy: dict[str, Any]) -> dict[str, Any]:
    policy = normalize_policy_v2(policy)
    payload = _exact(value, CATALOG_ENTRY_FIELDS, "catalog entry")
    _require_safety_constants(payload, "catalog entry")
    if isinstance(payload["byte_size"], bool) or not isinstance(payload["byte_size"], int) or payload["byte_size"] < 0:
        raise ValueError("catalog entry byte_size must be a non-negative integer")
    if isinstance(payload["row_count"], bool) or not isinstance(payload["row_count"], int) or payload["row_count"] < 0:
        raise ValueError("catalog entry row_count must be a non-negative integer")
    for field in (
        "logical_object_id",
        "source_time",
        "event_time",
        "received_time",
        "data_as_of",
        "resolved_model_identity",
        "resolved_input_universe_identity",
        "snapshot_id",
    ):
        text = _string(payload[field], f"catalog entry.{field}")
        if field == "logical_object_id" and any(mark in text for mark in ("/", "\\", ":", "..")):
            raise ValueError("catalog logical_object_id must be opaque and path-free")
        if field in {
            "logical_object_id",
            "resolved_model_identity",
            "resolved_input_universe_identity",
            "snapshot_id",
        } and _contains_market_or_promotion_marker(text, policy):
            raise ValueError("catalog entry contains a strategy/market/promotion marker")
    return {
        "entry_id": _identifier(payload["entry_id"], "catalog entry.entry_id"),
        "logical_object_id": payload["logical_object_id"],
        "content_sha256": _sha256(payload["content_sha256"], "catalog entry.content_sha256"),
        "byte_size": payload["byte_size"],
        "schema_sha256": _sha256(payload["schema_sha256"], "catalog entry.schema_sha256"),
        "row_count": payload["row_count"],
        "source_time": payload["source_time"],
        "event_time": payload["event_time"],
        "received_time": payload["received_time"],
        "data_as_of": payload["data_as_of"],
        "upstream_lineage_sha256": _sha256(
            payload["upstream_lineage_sha256"], "catalog entry.upstream_lineage_sha256"
        ),
        "resolved_model_identity": payload["resolved_model_identity"],
        "resolved_input_universe_identity": payload["resolved_input_universe_identity"],
        "snapshot_id": payload["snapshot_id"],
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def normalize_catalog_entry_ref_v1(value: Any) -> dict[str, Any]:
    return _normalize_catalog_entry_ref(value, "catalog entry ref")


def _normalize_provider_contract(
    value: Any, label: str, *, policy: dict[str, Any]
) -> dict[str, Any]:
    fields = {
        "provider_kind",
        "provider_version",
        "execution_commit",
        "code_sha256",
        "environment_sha256",
        "command_template_id",
        "network_required",
        "raw_rows_returned",
    }
    payload = _exact(value, fields, label)
    if payload["network_required"] is not False or payload["raw_rows_returned"] is not False:
        raise ValueError(f"{label} cannot require network or return raw rows")
    for field in ("provider_kind", "provider_version", "command_template_id"):
        if _contains_market_or_promotion_marker(
            _string(payload[field], f"{label}.{field}"), policy
        ):
            raise ValueError(f"{label} contains a strategy/market/promotion marker")
    return {
        "provider_kind": _string(payload["provider_kind"], f"{label}.provider_kind"),
        "provider_version": _string(payload["provider_version"], f"{label}.provider_version"),
        "execution_commit": _git_sha(payload["execution_commit"], f"{label}.execution_commit"),
        "code_sha256": _sha256(payload["code_sha256"], f"{label}.code_sha256"),
        "environment_sha256": _sha256(
            payload["environment_sha256"], f"{label}.environment_sha256"
        ),
        "command_template_id": _identifier(
            payload["command_template_id"], f"{label}.command_template_id"
        ),
        "network_required": False,
        "raw_rows_returned": False,
    }


def normalize_catalog_publication_scope_v1(
    value: Any,
    *,
    policy: dict[str, Any],
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
    policy = normalize_policy_v2(policy)
    payload = _exact(value, CATALOG_PUBLICATION_SCOPE_FIELDS, "catalog publication scope")
    identity = _bind_artifact_identity(
        payload,
        schema_version=1,
        schema_kind="catalog_publication_scope",
        policy_sha256=policy_sha256,
        schema_sha256=schema_sha256,
        label="catalog publication scope",
    )
    _require_safety_constants(payload, "catalog publication scope")
    if payload["catalog_kind"] != "roi_reproduction_reference_catalog_v1":
        raise ValueError("catalog publication scope catalog_kind is invalid")
    if isinstance(payload["catalog_contract_version"], bool) or payload["catalog_contract_version"] != 1:
        raise ValueError("catalog publication scope contract version must be integer 1")
    source_requests = payload["source_requests"]
    if not isinstance(source_requests, list) or not source_requests:
        raise ValueError("catalog source_requests must be non-empty")
    normalized_requests: list[dict[str, str]] = []
    for item in source_requests:
        request = _exact(item, {"logical_source_id", "role"}, "catalog source request")
        logical_id = _string(request["logical_source_id"], "catalog source logical ID")
        if any(mark in logical_id for mark in ("/", "\\", ":", "..")):
            raise ValueError("catalog logical source IDs must be opaque and path-free")
        normalized_requests.append(
            {"logical_source_id": logical_id, "role": _string(request["role"], "catalog source role")}
        )
    if len({item["logical_source_id"].casefold() for item in normalized_requests}) != len(normalized_requests):
        raise ValueError("catalog source_requests contain duplicate logical source IDs")
    for request in normalized_requests:
        if any(
            _contains_market_or_promotion_marker(request[field], policy)
            for field in ("logical_source_id", "role")
        ):
            raise ValueError("catalog source request contains a strategy/market/promotion marker")
    metadata_schema = _exact(
        payload["expected_metadata_schema"],
        {"allowed_fields", "raw_values_allowed", "absolute_paths_allowed", "secrets_allowed"},
        "catalog expected_metadata_schema",
    )
    if any(metadata_schema[name] is not False for name in ("raw_values_allowed", "absolute_paths_allowed", "secrets_allowed")):
        raise ValueError("catalog metadata schema cannot allow raw values, paths, or secrets")
    allowed_metadata_fields = _semantic_strings(
        metadata_schema["allowed_fields"], "catalog metadata allowed_fields"
    )
    if any(
        _contains_market_or_promotion_marker(field, policy)
        for field in allowed_metadata_fields
    ):
        raise ValueError("catalog metadata schema contains a strategy/market/promotion marker")
    resource_budget = _exact(
        payload["resource_budget"],
        {"maximum_bytes", "maximum_runtime_seconds", "network_calls", "retry_budget"},
        "catalog resource_budget",
    )
    for field in ("maximum_bytes", "maximum_runtime_seconds"):
        if isinstance(resource_budget[field], bool) or not isinstance(resource_budget[field], int) or resource_budget[field] <= 0:
            raise ValueError(f"catalog resource_budget.{field} must be a positive integer")
    for field in ("network_calls", "retry_budget"):
        if isinstance(resource_budget[field], bool) or resource_budget[field] != 0:
            raise ValueError(f"catalog resource_budget.{field} must be integer 0")
    capabilities = normalize_capabilities(payload["capabilities"], label="catalog scope capabilities")
    if capabilities != policy["phase_capabilities"]["catalog_release_maintenance"]["flags"]:
        raise ValueError("catalog scope capabilities differ from the frozen future phase contract")
    return {
        "schema_version": 1,
        "catalog_kind": "roi_reproduction_reference_catalog_v1",
        "catalog_contract_version": 1,
        "scope_id": _identifier(payload["scope_id"], "catalog publication scope.scope_id"),
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        **identity,
        "repository_contract": _normalize_repository_contract(
            payload["repository_contract"], "catalog publication scope.repository_contract"
        ),
        "provider_contract": _normalize_provider_contract(
            payload["provider_contract"],
            "catalog publication scope.provider_contract",
            policy=policy,
        ),
        "source_requests": sorted(normalized_requests, key=lambda item: item["logical_source_id"]),
        "expected_metadata_schema": {
            "allowed_fields": allowed_metadata_fields,
            "raw_values_allowed": False,
            "absolute_paths_allowed": False,
            "secrets_allowed": False,
        },
        "resource_budget": resource_budget,
        "capabilities": capabilities,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def normalize_catalog_publication_provider_lease_v1(value: Any) -> dict[str, Any]:
    payload = _exact(value, CATALOG_PROVIDER_LEASE_FIELDS, "catalog provider lease")
    _require_safety_constants(payload, "catalog provider lease")
    constants = {
        "subject_kind": "CATALOG_PUBLICATION_SCOPE",
        "catalog_kind": "roi_reproduction_reference_catalog_v1",
        "catalog_contract_version": 1,
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "phase": "CATALOG_PUBLISHING",
        "retry_budget": 0,
    }
    for field, expected in constants.items():
        observed = payload[field]
        if not _same_json_type_and_value(observed, expected):
            raise ValueError(f"catalog provider lease.{field} must equal {expected!r}")
    normalized = dict(payload)
    normalized.update(constants)
    for field in (
        "capability_digest",
        "command_digest",
        "provider_identity_digest",
        "provider_code_digest",
        "provider_environment_digest",
        "durable_ledger_backend_identity_digest",
        "durable_ledger_head_digest",
        "grant_reservation_receipt_digest",
        "policy_digest",
        "schema_digest",
        "github_evidence_digest",
    ):
        normalized[field] = _sha256(payload[field], f"catalog provider lease.{field}")
    expected_capability_digest = canonical_digest_v2(
        EXPECTED_PHASE_CAPABILITIES["catalog_release_maintenance"]["flags"]
    )
    if normalized["capability_digest"] != expected_capability_digest:
        raise ValueError("catalog provider lease capability digest differs from its frozen phase")
    normalized["verified_current_main_sha"] = _git_sha(
        payload["verified_current_main_sha"], "catalog provider lease.verified_current_main_sha"
    )
    normalized["durable_ledger_head_sequence"] = _nonnegative_int(
        payload["durable_ledger_head_sequence"], "catalog provider lease.durable_ledger_head_sequence"
    )
    for field in (
        "lease_id",
        "catalog_publication_scope_id",
        "issued_at",
        "expires_at",
        "human_supervisor_identity",
    ):
        normalized[field] = _string(payload[field], f"catalog provider lease.{field}")
    normalized["catalog_publication_scope_digest"] = _sha256(
        payload["catalog_publication_scope_digest"],
        "catalog provider lease.catalog_publication_scope_digest",
    )
    normalized["formal_buy"] = False
    normalized["send_order"] = False
    normalized["stake"] = 0
    return normalized


def normalize_catalog_release_v1(
    value: Any,
    *,
    policy: dict[str, Any],
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
    policy = normalize_policy_v2(policy)
    payload = _exact(value, CATALOG_RELEASE_FIELDS, "catalog release")
    identity = _bind_artifact_identity(
        payload,
        schema_version=1,
        schema_kind="catalog_release",
        policy_sha256=policy_sha256,
        schema_sha256=schema_sha256,
        label="catalog release",
    )
    _require_safety_constants(payload, "catalog release")
    if (
        payload["catalog_kind"] != "roi_reproduction_reference_catalog_v1"
        or isinstance(payload["catalog_contract_version"], bool)
        or payload["catalog_contract_version"] != 1
    ):
        raise ValueError("catalog release identity is invalid")
    capabilities = normalize_capabilities(payload["capabilities"], label="catalog release capabilities")
    if capabilities != G1_CAPABILITIES:
        raise ValueError("published catalog artifact carries no active capabilities")
    entries = payload["logical_entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("catalog release logical_entries must be non-empty")
    normalized_entries: list[dict[str, Any]] = []
    for item in entries:
        record = _exact(item, {"entry", "entry_digest"}, "catalog release entry record")
        entry = normalize_catalog_entry_v1(record["entry"], policy=policy)
        digest = _sha256(record["entry_digest"], "catalog release entry_digest")
        if digest != canonical_digest_v2(entry):
            raise ValueError("catalog release entry digest does not match its strict entry payload")
        normalized_entries.append({"entry": entry, "entry_digest": digest})
    entry_ids = [item["entry"]["entry_id"] for item in normalized_entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("catalog release contains duplicate entry IDs")
    logical_object_ids = [
        item["entry"]["logical_object_id"].casefold() for item in normalized_entries
    ]
    if len(logical_object_ids) != len(set(logical_object_ids)):
        raise ValueError("catalog release contains ambiguous duplicate logical object IDs")
    edges = payload["lineage_edges"]
    if not isinstance(edges, list):
        raise ValueError("catalog release lineage_edges must be a list")
    normalized_edges: list[dict[str, str]] = []
    for item in edges:
        edge = _exact(
            item,
            {"upstream_entry_id", "downstream_entry_id", "relation"},
            "catalog lineage edge",
        )
        normalized_edges.append(
            {
                "upstream_entry_id": _identifier(edge["upstream_entry_id"], "lineage upstream"),
                "downstream_entry_id": _identifier(edge["downstream_entry_id"], "lineage downstream"),
                "relation": _string(edge["relation"], "lineage relation"),
            }
        )
    known_entry_ids = set(entry_ids)
    edge_keys = [
        (
            edge["upstream_entry_id"],
            edge["downstream_entry_id"],
            edge["relation"],
        )
        for edge in normalized_edges
    ]
    if len(edge_keys) != len(set(edge_keys)):
        raise ValueError("catalog release contains duplicate lineage edges")
    for upstream, downstream, _ in edge_keys:
        if upstream not in known_entry_ids or downstream not in known_entry_ids:
            raise ValueError("catalog lineage edge references an unknown entry")
        if upstream == downstream:
            raise ValueError("catalog lineage edges cannot be self-referential")
    adjacency = {entry_id: set() for entry_id in known_entry_ids}
    indegree = {entry_id: 0 for entry_id in known_entry_ids}
    for upstream, downstream, _ in edge_keys:
        if downstream not in adjacency[upstream]:
            adjacency[upstream].add(downstream)
            indegree[downstream] += 1
    frontier = sorted(entry_id for entry_id, count in indegree.items() if count == 0)
    visited = 0
    while frontier:
        entry_id = frontier.pop(0)
        visited += 1
        for downstream in sorted(adjacency[entry_id]):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                frontier.append(downstream)
                frontier.sort()
    if visited != len(known_entry_ids):
        raise ValueError("catalog lineage graph must be acyclic")
    provider_contract = _normalize_provider_contract(
        payload["provider_contract"], "catalog release.provider_contract", policy=policy
    )
    provider_execution_commit = _git_sha(
        payload["provider_execution_commit"], "catalog release.provider_execution_commit"
    )
    provider_code_sha256 = _sha256(
        payload["provider_code_sha256"], "catalog release.provider_code_sha256"
    )
    provider_environment_sha256 = _sha256(
        payload["provider_environment_sha256"],
        "catalog release.provider_environment_sha256",
    )
    if (
        provider_execution_commit != provider_contract["execution_commit"]
        or provider_code_sha256 != provider_contract["code_sha256"]
        or provider_environment_sha256 != provider_contract["environment_sha256"]
    ):
        raise ValueError("catalog release provider identity fields are not cross-bound")
    entry_set_digest = canonical_digest_v2(
        [
            {"entry_id": item["entry"]["entry_id"], "entry_digest": item["entry_digest"]}
            for item in sorted(normalized_entries, key=lambda item: item["entry"]["entry_id"])
        ]
    )
    if payload["catalog_entry_set_digest"] != entry_set_digest:
        raise ValueError("catalog_entry_set_digest does not match the strict entry index")
    return {
        "schema_version": 1,
        "catalog_kind": "roi_reproduction_reference_catalog_v1",
        "catalog_contract_version": 1,
        "release_id": _identifier(payload["release_id"], "catalog release.release_id"),
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        **identity,
        "catalog_publication_scope_digest": _sha256(
            payload["catalog_publication_scope_digest"], "catalog release.publication scope digest"
        ),
        "catalog_publication_approval_evidence_digest": _sha256(
            payload["catalog_publication_approval_evidence_digest"],
            "catalog release.approval evidence digest",
        ),
        "repository_contract": _normalize_repository_contract(
            payload["repository_contract"], "catalog release.repository_contract"
        ),
        "github_trust_evidence": _normalize_digest_evidence(
            payload["github_trust_evidence"], "catalog release.github_trust_evidence"
        ),
        "provider_contract": provider_contract,
        "provider_identity": _string(payload["provider_identity"], "catalog release.provider_identity"),
        "provider_execution_commit": provider_execution_commit,
        "provider_code_sha256": provider_code_sha256,
        "provider_environment_sha256": provider_environment_sha256,
        "publication_receipt_digest": _sha256(
            payload["publication_receipt_digest"], "catalog release.publication_receipt_digest"
        ),
        "catalog_entry_set_digest": entry_set_digest,
        "logical_entries": sorted(normalized_entries, key=lambda item: item["entry"]["entry_id"]),
        "lineage_edges": sorted(
            normalized_edges,
            key=lambda item: (
                item["upstream_entry_id"],
                item["downstream_entry_id"],
                item["relation"],
            ),
        ),
        "created_at": _string(payload["created_at"], "catalog release.created_at"),
        "capabilities": capabilities,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def _normalize_command(value: Any, label: str) -> dict[str, Any]:
    payload = _exact(value, {"template_id", "arguments"}, label)
    template_id = _identifier(payload["template_id"], f"{label}.template_id")
    arguments = payload["arguments"]
    if not isinstance(arguments, list):
        raise ValueError(f"{label}.arguments must be a list")
    normalized_arguments: list[dict[str, Any]] = []
    for item in arguments:
        argument = _exact(item, {"name", "value_type", "value"}, f"{label}.arguments[]")
        name = _identifier(argument["name"], f"{label}.arguments[].name")
        value_type = _string(argument["value_type"], f"{label}.arguments[].value_type")
        if value_type not in {"string", "integer", "boolean"}:
            raise ValueError(f"{label}.arguments[].value_type is unsupported")
        observed = argument["value"]
        if value_type == "string":
            observed = _string(observed, f"{label}.arguments[].value")
            compact = observed.casefold()
            if any(token in observed for token in ("|", ">", "<", "&&", ";")):
                raise ValueError("structured command arguments cannot contain shell operators")
            if "://" in compact:
                raise ValueError("structured command arguments cannot contain URLs")
        elif value_type == "integer":
            if isinstance(observed, bool) or not isinstance(observed, int):
                raise ValueError(f"{label}.arguments[].value must be an integer")
        elif not isinstance(observed, bool):
            raise ValueError(f"{label}.arguments[].value must be boolean")
        normalized_arguments.append({"name": name, "value_type": value_type, "value": observed})
    names = [item["name"] for item in normalized_arguments]
    if len(names) != len(set(names)):
        raise ValueError(f"{label}.arguments contains duplicate names")
    normalized = {"template_id": template_id, "arguments": normalized_arguments}
    expected = {
        "template_id": "m0_reproduction",
        "arguments": [
            {"name": "replicas", "value_type": "integer", "value": 2},
            {"name": "fixture_only", "value_type": "boolean", "value": False},
        ],
    }
    if not _same_json_type_and_value(normalized, expected):
        raise ValueError("run command must equal the single code-owned M0 reproduction template")
    return expected


def _normalize_replicate_contract(value: Any) -> dict[str, Any]:
    fields = {
        "replicate_count",
        "replica_ids",
        "same_execution_commit",
        "same_run_scope_digest",
        "same_input_hashes",
        "same_environment_hash",
        "isolated_checkouts",
        "shared_mutable_cache",
        "network_calls",
        "digest_algorithm",
        "crash_automatic_retry",
    }
    payload = _exact(value, fields, "run.replicate_contract")
    expected = {
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
    }
    if not _same_json_type_and_value(payload, expected):
        raise ValueError("run.replicate_contract must equal the frozen two-clean-replica contract")
    return expected


def normalize_run_v2(
    value: Any,
    *,
    policy: dict[str, Any],
    proposal: dict[str, Any],
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
    policy = normalize_policy_v2(policy)
    normalized_proposal = normalize_proposal_v2(
        proposal,
        policy=policy,
        policy_sha256=policy_sha256,
    )
    payload = _exact(value, RUN_FIELDS, "ROI reproduction run")
    identity = _bind_artifact_identity(
        payload,
        schema_version=RUN_SCHEMA_VERSION,
        schema_kind="run",
        policy_sha256=policy_sha256,
        schema_sha256=schema_sha256,
        label="run",
    )
    _require_safety_constants(payload, "run")
    embedded = normalize_proposal_v2(
        payload["proposal_scope"],
        policy=policy,
        policy_sha256=policy_sha256,
        expected_experiment_id=normalized_proposal["experiment_id"],
    )
    if embedded != normalized_proposal:
        raise ValueError("run proposal_scope differs from the frozen proposal")
    proposal_digest = canonical_digest_v2(normalized_proposal)
    if payload["proposal_scope_digest"] != proposal_digest:
        raise ValueError("run proposal_scope_digest differs from the frozen proposal")
    entry_ref = _normalize_catalog_entry_ref(
        payload["reference_catalog_entry_ref"], "run.reference_catalog_entry_ref"
    )
    if entry_ref != normalized_proposal["reference_catalog_entry_ref"]:
        raise ValueError("run catalog entry ref differs from proposal")
    conditional_policy_refs = _exact(
        payload["conditional_policy_refs"], set(), "run.conditional_policy_refs"
    )
    commands = payload["exact_execution_commands"]
    if not isinstance(commands, list) or len(commands) != 1:
        raise ValueError("run.exact_execution_commands must contain exactly one code-owned command")
    capabilities = normalize_capabilities(payload["capabilities"], label="run.capabilities")
    if capabilities != policy["phase_capabilities"]["historical_reproduction"]["flags"]:
        raise ValueError("run capabilities differ from the frozen future historical phase")
    refs = {
        "config_refs": _normalize_refs(payload["config_refs"], "run.config_refs"),
        "data_input_manifest_refs": _normalize_refs(
            payload["data_input_manifest_refs"], "run.data_input_manifest_refs"
        ),
    }
    for field in (
        "source_lineage_manifest_ref",
        "fold_manifest_ref",
        "runner_universe_manifest_ref",
        "feature_lineage_manifest_ref",
        "target_label_manifest_ref",
        "model_recipe_manifest_ref",
        "reference_artifact_manifest_ref",
        "canonicalization_manifest_ref",
        "dependency_environment_manifest_ref",
    ):
        refs[field] = normalize_ref(payload[field], f"run.{field}")
    if refs["fold_manifest_ref"] != normalized_proposal["fold_manifest_ref"]:
        raise ValueError("run fold manifest differs from proposal")
    if (
        refs["model_recipe_manifest_ref"]
        != normalized_proposal["model_identity_contract"]["model_recipe_manifest_ref"]
    ):
        raise ValueError("run model recipe manifest differs from proposal identity")
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("run.seed must be an integer")
    normalized_commands = [
        _normalize_command(item, "run.exact_execution_commands[]") for item in commands
    ]
    sensitive_command_values: list[str] = []
    for command in normalized_commands:
        sensitive_command_values.append(command["template_id"])
        for argument in command["arguments"]:
            sensitive_command_values.append(argument["name"])
            if isinstance(argument["value"], str):
                sensitive_command_values.append(argument["value"])
    for text in sensitive_command_values:
        if _contains_market_or_promotion_marker(text, policy):
            raise ValueError("run command contains a strategy/market/promotion marker")
    for reference in (
        *refs["config_refs"],
        *refs["data_input_manifest_refs"],
        *(
            refs[field]
            for field in refs
            if field not in {"config_refs", "data_input_manifest_refs"}
        ),
    ):
        path = reference["path"]
        if _path_has_forbidden_prefix(path, policy):
            raise ValueError(f"run manifest reference uses a forbidden path: {path}")
        if _contains_market_or_promotion_marker(path, policy):
            raise ValueError("run manifest reference contains a strategy/market/promotion marker")
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "experiment_id": normalized_proposal["experiment_id"],
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "execution_kind": EXECUTION_KIND,
        **identity,
        "proposal_scope": normalized_proposal,
        "proposal_scope_digest": proposal_digest,
        "reference_catalog_entry_ref": entry_ref,
        "execution_commit_sha": _git_sha(payload["execution_commit_sha"], "run.execution_commit_sha"),
        **refs,
        "seed": seed,
        "exact_execution_commands": normalized_commands,
        "replicate_contract": _normalize_replicate_contract(payload["replicate_contract"]),
        "conditional_policy_refs": conditional_policy_refs,
        "capabilities": capabilities,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def evaluate_run_v2(
    run: dict[str, Any],
    *,
    policy: dict[str, Any],
    proposal: dict[str, Any],
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_run_v2(
        run,
        policy=policy,
        proposal=proposal,
        policy_sha256=policy_sha256,
        schema_sha256=schema_sha256,
    )
    return {
        "run_scope_digest": canonical_digest_v2(normalized),
        "g1_shape_validation_passed": True,
        "structural_contract_passed": False,
        "trusted_policy_and_schema_digest_context_verified": False,
        "catalog_manifest_conformance_verified": False,
        "execution_context_and_expected_outputs_verified": False,
        "catalog_verified": False,
        "durable_ledger_verified": False,
        "one_shot_lease_verified": False,
        "execution_allowed": False,
        "status": "BLOCKED_CAPABILITY",
        "implementation_status": IMPLEMENTATION_STATUS,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def _normalize_check(value: Any, label: str) -> dict[str, Any]:
    payload = _exact(value, {"passed", "evidence_ref"}, label)
    if not isinstance(payload["passed"], bool):
        raise ValueError(f"{label}.passed must be boolean")
    return {
        "passed": payload["passed"],
        "evidence_ref": normalize_ref(payload["evidence_ref"], f"{label}.evidence_ref"),
    }


def _normalize_replica_result(value: Any, label: str) -> dict[str, Any]:
    fields = {
        "replica_id",
        "commit_sha",
        "worktree_clean",
        "environment_manifest_ref",
        "command_digest",
        "input_digest_set",
        "artifact_digests",
    }
    payload = _exact(value, fields, label)
    if payload["replica_id"] not in {"clean_a", "clean_b"}:
        raise ValueError(f"{label}.replica_id must be clean_a or clean_b")
    if payload["worktree_clean"] is not True:
        raise ValueError(f"{label}.worktree_clean must be true")
    input_digests = _semantic_strings(payload["input_digest_set"], f"{label}.input_digest_set")
    artifact_digests = _object(payload["artifact_digests"], f"{label}.artifact_digests")
    normalized_artifacts: dict[str, str] = {}
    for name, digest in sorted(artifact_digests.items()):
        normalized_artifacts[_identifier(name, f"{label}.artifact_digests key")] = _sha256(
            digest, f"{label}.artifact_digests.{name}"
        )
    if not normalized_artifacts:
        raise ValueError(f"{label}.artifact_digests must be non-empty")
    return {
        "replica_id": payload["replica_id"],
        "commit_sha": _git_sha(payload["commit_sha"], f"{label}.commit_sha"),
        "worktree_clean": True,
        "environment_manifest_ref": normalize_ref(
            payload["environment_manifest_ref"], f"{label}.environment_manifest_ref"
        ),
        "command_digest": _sha256(payload["command_digest"], f"{label}.command_digest"),
        "input_digest_set": [_sha256(item, f"{label}.input_digest_set[]") for item in input_digests],
        "artifact_digests": normalized_artifacts,
    }


def _normalize_determinism_check(value: Any) -> dict[str, Any]:
    fields = {"all_equal", "mismatch_count", "artifact_digest_pairs"}
    payload = _exact(value, fields, "result.determinism_check")
    if not isinstance(payload["all_equal"], bool):
        raise ValueError("result.determinism_check.all_equal must be boolean")
    mismatch_count = _nonnegative_int(
        payload["mismatch_count"], "result.determinism_check.mismatch_count"
    )
    pairs = payload["artifact_digest_pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("artifact_digest_pairs must be a non-empty list")
    normalized_pairs: list[dict[str, str]] = []
    for item in pairs:
        pair = _exact(item, {"artifact_id", "clean_a_digest", "clean_b_digest"}, "artifact digest pair")
        normalized_pairs.append(
            {
                "artifact_id": _identifier(pair["artifact_id"], "artifact digest pair.artifact_id"),
                "clean_a_digest": _sha256(pair["clean_a_digest"], "artifact digest pair.clean_a"),
                "clean_b_digest": _sha256(pair["clean_b_digest"], "artifact digest pair.clean_b"),
            }
        )
    artifact_ids = [item["artifact_id"] for item in normalized_pairs]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("artifact_digest_pairs contains duplicate artifact IDs")
    observed_mismatches = sum(
        item["clean_a_digest"] != item["clean_b_digest"] for item in normalized_pairs
    )
    if mismatch_count != observed_mismatches or payload["all_equal"] != (observed_mismatches == 0):
        raise ValueError("determinism summary does not match artifact digest pairs")
    return {
        "all_equal": payload["all_equal"],
        "mismatch_count": mismatch_count,
        "artifact_digest_pairs": sorted(normalized_pairs, key=lambda item: item["artifact_id"]),
    }


def _decode_finite_binary64_hex(value: str, label: str) -> float:
    try:
        decoded = struct.unpack(">d", bytes.fromhex(value))[0]
    except (ValueError, struct.error) as exc:
        raise ValueError(f"{label} must be binary64 hex") from exc
    if not math.isfinite(decoded):
        raise ValueError(f"{label} must decode to a finite number")
    return decoded


def _normalize_numeric_equivalence(
    value: Any,
    *,
    policy: dict[str, Any],
    expected_model_name: str,
    expected_probability_stage: str,
    expected_numeric_contract: list[dict[str, str]],
    expected_reference_available: bool,
) -> dict[str, Any]:
    fields = {"reference_available", "all_within_tolerance", "rows"}
    payload = _exact(value, fields, "result.numeric_equivalence")
    if not isinstance(payload["reference_available"], bool) or not isinstance(
        payload["all_within_tolerance"], bool
    ):
        raise ValueError("numeric equivalence flags must be boolean")
    if not isinstance(expected_reference_available, bool):
        raise ValueError("expected_reference_available must be boolean")
    if payload["reference_available"] is not expected_reference_available:
        raise ValueError("numeric reference availability differs from trusted catalog evidence")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("numeric equivalence rows must be a non-empty list")
    fields_row = {
        "metric_id",
        "fold_id",
        "model_name",
        "probability_stage",
        "unit",
        "reference_value_hex",
        "observed_value_hex",
        "absolute_tolerance_hex",
        "pass",
    }
    normalized_rows: list[dict[str, Any]] = []
    for item in rows:
        row = _exact(item, fields_row, "numeric equivalence row")
        for field in ("reference_value_hex", "observed_value_hex", "absolute_tolerance_hex"):
            text = _string(row[field], f"numeric equivalence row.{field}")
            if not re.fullmatch(r"[0-9a-f]{16}", text):
                raise ValueError(f"numeric equivalence row.{field} must be binary64 hex")
        if not isinstance(row["pass"], bool):
            raise ValueError("numeric equivalence row.pass must be boolean")
        if row["model_name"] != expected_model_name:
            raise ValueError("numeric equivalence row uses an unexpected model name")
        if row["probability_stage"] != expected_probability_stage:
            raise ValueError("numeric equivalence row uses an unexpected probability stage")
        for field in ("metric_id", "fold_id", "model_name", "probability_stage", "unit"):
            if _contains_market_or_promotion_marker(
                _string(row[field], f"numeric equivalence row.{field}"), policy
            ):
                raise ValueError("numeric equivalence row contains a forbidden strategy marker")
        reference_value = _decode_finite_binary64_hex(
            row["reference_value_hex"], "numeric equivalence reference value"
        )
        observed_value = _decode_finite_binary64_hex(
            row["observed_value_hex"], "numeric equivalence observed value"
        )
        absolute_tolerance = _decode_finite_binary64_hex(
            row["absolute_tolerance_hex"], "numeric equivalence absolute tolerance"
        )
        if absolute_tolerance < 0:
            raise ValueError("numeric equivalence absolute tolerance must be non-negative")
        expected_pass = abs(reference_value - observed_value) <= absolute_tolerance
        if row["pass"] != expected_pass:
            raise ValueError("numeric equivalence row.pass does not match its numeric values")
        normalized_rows.append(
            {
                **{name: _string(row[name], f"numeric row.{name}") for name in (
                    "metric_id", "fold_id", "model_name", "probability_stage", "unit"
                )},
                "reference_value_hex": row["reference_value_hex"],
                "observed_value_hex": row["observed_value_hex"],
                "absolute_tolerance_hex": row["absolute_tolerance_hex"],
                "pass": row["pass"],
            }
        )
    semantic_keys = [
        (
            row["metric_id"],
            row["fold_id"],
            row["model_name"],
            row["probability_stage"],
            row["unit"],
        )
        for row in normalized_rows
    ]
    if len(semantic_keys) != len(set(semantic_keys)):
        raise ValueError("numeric equivalence contains duplicate semantic rows")
    expected_fields = {
        "metric_id",
        "fold_id",
        "model_name",
        "probability_stage",
        "unit",
        "reference_value_hex",
        "absolute_tolerance_hex",
    }
    if not isinstance(expected_numeric_contract, list) or not expected_numeric_contract:
        raise ValueError("trusted numeric equivalence contract must be a non-empty list")
    normalized_expected: list[dict[str, str]] = []
    for item in expected_numeric_contract:
        expected_row = _exact(item, expected_fields, "trusted numeric equivalence contract row")
        normalized_expected_row = {
            name: _string(expected_row[name], f"trusted numeric contract.{name}")
            for name in expected_fields
        }
        for field in ("metric_id", "fold_id", "model_name", "probability_stage", "unit"):
            if _contains_market_or_promotion_marker(normalized_expected_row[field], policy):
                raise ValueError("trusted numeric equivalence contract contains a strategy marker")
        if normalized_expected_row["model_name"] != expected_model_name:
            raise ValueError("trusted numeric contract uses an unexpected model name")
        if normalized_expected_row["probability_stage"] != expected_probability_stage:
            raise ValueError("trusted numeric contract uses an unexpected probability stage")
        _decode_finite_binary64_hex(
            normalized_expected_row["reference_value_hex"],
            "trusted numeric contract reference value",
        )
        trusted_tolerance = _decode_finite_binary64_hex(
            normalized_expected_row["absolute_tolerance_hex"],
            "trusted numeric contract absolute tolerance",
        )
        if trusted_tolerance < 0:
            raise ValueError("trusted numeric contract tolerance must be non-negative")
        normalized_expected.append(normalized_expected_row)
    expected_keys = [
        (
            row["metric_id"],
            row["fold_id"],
            row["model_name"],
            row["probability_stage"],
            row["unit"],
        )
        for row in normalized_expected
    ]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("trusted numeric equivalence contract contains duplicate rows")
    expected_by_key = dict(zip(expected_keys, normalized_expected))
    if set(semantic_keys) != set(expected_by_key):
        raise ValueError("numeric equivalence rows differ from the trusted contract")
    for row, key in zip(normalized_rows, semantic_keys):
        expected_row = expected_by_key[key]
        if (
            row["reference_value_hex"] != expected_row["reference_value_hex"]
            or row["absolute_tolerance_hex"] != expected_row["absolute_tolerance_hex"]
        ):
            raise ValueError("numeric equivalence reference or tolerance differs from the trusted contract")
    if payload["all_within_tolerance"] != all(row["pass"] for row in normalized_rows):
        raise ValueError("numeric equivalence summary does not match rows")
    return {
        "reference_available": payload["reference_available"],
        "all_within_tolerance": payload["all_within_tolerance"],
        "rows": sorted(
            normalized_rows,
            key=lambda row: (
                row["metric_id"],
                row["fold_id"],
                row["model_name"],
                row["probability_stage"],
                row["unit"],
            ),
        ),
    }


def normalize_result_v1(
    value: Any,
    *,
    policy: dict[str, Any],
    proposal_scope_digest: str,
    run_scope_digest: str,
    expected_experiment_id: str,
    expected_execution_commit_sha: str,
    expected_model_name: str,
    expected_probability_stage: str,
    expected_artifact_ids: list[str],
    expected_numeric_contract: list[dict[str, str]],
    expected_reference_available: bool,
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
    policy = normalize_policy_v2(policy)
    payload = _exact(value, RESULT_FIELDS, "ROI reproduction result")
    identity = _bind_artifact_identity(
        payload,
        schema_version=RESULT_SCHEMA_VERSION,
        schema_kind="result",
        policy_sha256=policy_sha256,
        schema_sha256=schema_sha256,
        label="result",
    )
    _require_safety_constants(payload, "result")
    if payload["result_kind"] != "roi_reproduction_result_v1":
        raise ValueError("result.result_kind is invalid")
    proposal_digest = _sha256(payload["proposal_scope_digest"], "result.proposal_scope_digest")
    run_digest = _sha256(payload["run_scope_digest"], "result.run_scope_digest")
    if proposal_digest != _sha256(proposal_scope_digest, "proposal_scope_digest"):
        raise ValueError("result proposal digest binding failed")
    if run_digest != _sha256(run_scope_digest, "run_scope_digest"):
        raise ValueError("result run digest binding failed")
    experiment_id = _identifier(payload["experiment_id"], "result.experiment_id")
    trusted_experiment_id = _identifier(expected_experiment_id, "expected_experiment_id")
    if experiment_id != trusted_experiment_id:
        raise ValueError("result experiment_id differs from the trusted experiment")
    execution_commit = _git_sha(payload["execution_commit_sha"], "result.execution_commit_sha")
    trusted_execution_commit = _git_sha(
        expected_execution_commit_sha, "expected_execution_commit_sha"
    )
    if execution_commit != trusted_execution_commit:
        raise ValueError("result execution commit differs from the trusted run")
    trusted_model_name = _identifier(expected_model_name, "expected_model_name")
    if expected_probability_stage not in policy["allowed_probability_stages"]:
        raise ValueError("expected_probability_stage is not allowed by the frozen policy")
    outcome = _string(payload["computed_outcome"], "result.computed_outcome")
    if outcome not in policy["allowed_result_outcomes"]:
        raise ValueError("result computed_outcome is not allowed")
    replicas = payload["replicas"]
    if not isinstance(replicas, list) or len(replicas) != 2:
        raise ValueError("result must contain exactly two replicas")
    normalized_replicas = [_normalize_replica_result(item, "result.replicas[]") for item in replicas]
    if {item["replica_id"] for item in normalized_replicas} != {"clean_a", "clean_b"}:
        raise ValueError("result replicas must be clean_a and clean_b")
    normalized_replicas = sorted(normalized_replicas, key=lambda item: item["replica_id"])
    replica_a, replica_b = normalized_replicas
    same_bound_inputs = (
        replica_a["commit_sha"] == replica_b["commit_sha"] == trusted_execution_commit
        and replica_a["environment_manifest_ref"]["sha256"]
        == replica_b["environment_manifest_ref"]["sha256"]
        and replica_a["command_digest"] == replica_b["command_digest"]
        and replica_a["input_digest_set"] == replica_b["input_digest_set"]
    )
    determinism = _normalize_determinism_check(payload["determinism_check"])
    artifact_ids_a = set(replica_a["artifact_digests"])
    artifact_ids_b = set(replica_b["artifact_digests"])
    pair_by_id = {
        item["artifact_id"]: item for item in determinism["artifact_digest_pairs"]
    }
    if artifact_ids_a != artifact_ids_b or set(pair_by_id) != artifact_ids_a:
        raise ValueError("determinism pairs must exactly cover both replica artifact sets")
    trusted_artifact_ids = _semantic_strings(
        expected_artifact_ids, "expected_artifact_ids"
    )
    for artifact_id in trusted_artifact_ids:
        _identifier(artifact_id, "expected_artifact_ids[]")
        if _contains_market_or_promotion_marker(artifact_id, policy):
            raise ValueError("expected artifact ID contains a forbidden strategy marker")
    if artifact_ids_a != set(trusted_artifact_ids):
        raise ValueError("result artifact IDs differ from the trusted proposal contract")
    for artifact_id, pair in pair_by_id.items():
        if (
            pair["clean_a_digest"] != replica_a["artifact_digests"][artifact_id]
            or pair["clean_b_digest"] != replica_b["artifact_digests"][artifact_id]
        ):
            raise ValueError("determinism pair does not match replica artifact digests")
    numeric = _normalize_numeric_equivalence(
        payload["numeric_equivalence"],
        policy=policy,
        expected_model_name=trusted_model_name,
        expected_probability_stage=expected_probability_stage,
        expected_numeric_contract=expected_numeric_contract,
        expected_reference_available=expected_reference_available,
    )
    runner_check = _normalize_check(
        payload["runner_label_contract_check"], "result.runner_label_contract_check"
    )
    probability_check = _normalize_check(
        payload["probability_contract_check"], "result.probability_contract_check"
    )
    if outcome != "INVALID":
        if not same_bound_inputs:
            raise ValueError("non-INVALID result replicas must bind identical inputs")
        if not runner_check["passed"] or not probability_check["passed"]:
            raise ValueError("failed runner or probability contracts require INVALID")
        if determinism["all_equal"] and numeric["all_within_tolerance"]:
            derived_outcome = (
                "REPRODUCED"
                if numeric["reference_available"]
                else "RECONSTRUCTED_NOT_REPRODUCED"
            )
        else:
            derived_outcome = "REPRODUCTION_FAILED"
        if outcome != derived_outcome:
            raise ValueError("computed_outcome does not match validated result evidence")
    for evidence_check in (runner_check, probability_check):
        path = evidence_check["evidence_ref"]["path"]
        if _path_has_forbidden_prefix(path, policy) or _contains_market_or_promotion_marker(
            path, policy
        ):
            raise ValueError("result contract evidence uses a forbidden path")
    artifact_refs = _normalize_refs(payload["artifact_refs"], "result.artifact_refs")
    for artifact_ref in artifact_refs:
        path = artifact_ref["path"]
        if _path_has_forbidden_prefix(path, policy) or _contains_market_or_promotion_marker(
            path, policy
        ):
            raise ValueError("result artifact uses a forbidden path")
    safety = normalize_capabilities(payload["safety"], label="result.safety")
    if safety != G1_CAPABILITIES:
        raise ValueError("result artifact carries no active capabilities")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_kind": "roi_reproduction_result_v1",
        "experiment_id": experiment_id,
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "execution_kind": EXECUTION_KIND,
        **identity,
        "proposal_scope_digest": proposal_digest,
        "run_scope_digest": run_digest,
        "execution_commit_sha": execution_commit,
        "computed_outcome": outcome,
        "replicas": normalized_replicas,
        "determinism_check": determinism,
        "runner_label_contract_check": runner_check,
        "probability_contract_check": probability_check,
        "numeric_equivalence": numeric,
        "artifact_refs": artifact_refs,
        "safety": safety,
        "limitations": _semantic_strings(payload["limitations"], "result.limitations", allow_empty=True),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


OUTCOME_REVIEW_MAPPING = {
    "REPRODUCED": {"REPRODUCED", "REJECTED"},
    "RECONSTRUCTED_NOT_REPRODUCED": {"RECONSTRUCTED_NOT_REPRODUCED", "REJECTED"},
    "REPRODUCTION_FAILED": {"REPRODUCTION_FAILED", "REJECTED"},
    "INVALID": {"INVALID"},
}


def normalize_review_v1(
    value: Any,
    *,
    result: dict[str, Any],
    policy: dict[str, Any],
    proposal_scope_digest: str,
    run_scope_digest: str,
    expected_experiment_id: str,
    expected_execution_commit_sha: str,
    expected_model_name: str,
    expected_probability_stage: str,
    expected_artifact_ids: list[str],
    expected_numeric_contract: list[dict[str, str]],
    expected_reference_available: bool,
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
    policy = normalize_policy_v2(policy)
    normalized_result = normalize_result_v1(
        result,
        policy=policy,
        proposal_scope_digest=proposal_scope_digest,
        run_scope_digest=run_scope_digest,
        expected_experiment_id=expected_experiment_id,
        expected_execution_commit_sha=expected_execution_commit_sha,
        expected_model_name=expected_model_name,
        expected_probability_stage=expected_probability_stage,
        expected_artifact_ids=expected_artifact_ids,
        expected_numeric_contract=expected_numeric_contract,
        expected_reference_available=expected_reference_available,
        policy_sha256=policy_sha256,
    )
    payload = _exact(value, REVIEW_FIELDS, "ROI reproduction review")
    identity = _bind_artifact_identity(
        payload,
        schema_version=REVIEW_SCHEMA_VERSION,
        schema_kind="review",
        policy_sha256=policy_sha256,
        schema_sha256=schema_sha256,
        label="review",
    )
    _require_safety_constants(payload, "review")
    expected_id = _identifier(expected_experiment_id, "review expected_experiment_id")
    if normalized_result["experiment_id"] != expected_id or payload["experiment_id"] != expected_id:
        raise ValueError("review experiment_id differs from the trusted experiment ID")
    for field in ("proposal_scope_digest", "run_scope_digest"):
        if payload[field] != normalized_result[field]:
            raise ValueError(f"review {field} differs from validated result")
    result_digest = canonical_digest_v2(normalized_result)
    if payload["validated_result_digest"] != result_digest:
        raise ValueError("review validated_result_digest differs from validated result")
    if payload["computed_outcome"] != normalized_result["computed_outcome"]:
        raise ValueError("review cannot rewrite the computed outcome")
    proposed = _string(payload["proposed_terminal_status"], "review.proposed_terminal_status")
    if proposed not in OUTCOME_REVIEW_MAPPING[normalized_result["computed_outcome"]]:
        raise ValueError("human review cannot upgrade or rewrite the computed outcome")
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "experiment_id": normalized_result["experiment_id"],
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "execution_kind": EXECUTION_KIND,
        **identity,
        "proposal_scope_digest": normalized_result["proposal_scope_digest"],
        "run_scope_digest": normalized_result["run_scope_digest"],
        "validated_result_digest": result_digest,
        "computed_outcome": normalized_result["computed_outcome"],
        "proposed_terminal_status": proposed,
        "review_limitations": _semantic_strings(
            payload["review_limitations"], "review.review_limitations", allow_empty=True
        ),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def normalize_queue_v4(
    value: Any,
    *,
    policy: dict[str, Any],
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
    policy = normalize_policy_v2(policy)
    payload = _exact(value, QUEUE_FIELDS, "ROI reproduction queue candidate")
    identity = _bind_artifact_identity(
        payload,
        schema_version=QUEUE_SCHEMA_VERSION,
        schema_kind="queue",
        policy_sha256=policy_sha256,
        schema_sha256=schema_sha256,
        label="queue",
    )
    _require_safety_constants(payload, "queue")
    if payload["status"] != "BLOCKED_CATALOG":
        raise ValueError("G1 queue candidates must remain BLOCKED_CATALOG")
    capabilities = normalize_capabilities(payload["capabilities"], label="queue.capabilities")
    if capabilities != G1_CAPABILITIES:
        raise ValueError("G1 queue candidates carry no capabilities")
    if payload["capability_digest"] != canonical_digest_v2(capabilities):
        raise ValueError("queue capability_digest does not match capabilities")
    path = _repository_path(payload["proposal_scope_path"], "queue.proposal_scope_path")
    if not path.startswith("research/synthetic/roi_reproduction/"):
        raise ValueError("G1 queue candidate may bind only a code-owned synthetic fixture path")
    return {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "experiment_id": _identifier(payload["experiment_id"], "queue.experiment_id"),
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "execution_kind": EXECUTION_KIND,
        **identity,
        "proposal_scope_path": path,
        "proposal_scope_digest": _sha256(
            payload["proposal_scope_digest"], "queue.proposal_scope_digest"
        ),
        "status": "BLOCKED_CATALOG",
        "capabilities": capabilities,
        "capability_digest": canonical_digest_v2(capabilities),
        "created_at": _string(payload["created_at"], "queue.created_at"),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def build_non_authoritative_queue_v4_candidate(
    *,
    proposal: dict[str, Any],
    proposal_scope_path: str,
    created_at: str,
    policy: dict[str, Any],
    policy_ref: dict[str, str],
    schema_ref: dict[str, str],
    fixture_only: bool,
) -> dict[str, Any]:
    if fixture_only is not True:
        raise ValueError("G1 queue compilation is available only for synthetic contract fixtures")
    normalized_proposal = normalize_proposal_v2(proposal, policy=policy)
    evaluation = evaluate_proposal_v2(normalized_proposal, policy=policy)
    if evaluation["status"] != "BLOCKED_CATALOG" or evaluation["proposal_or_queue_creation_allowed"]:
        raise ValueError("G1 must not convert an unverified catalog assertion into a live proposal")
    candidate = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "experiment_id": normalized_proposal["experiment_id"],
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "execution_kind": EXECUTION_KIND,
        "policy_ref": policy_ref,
        "schema_ref": schema_ref,
        "proposal_scope_path": proposal_scope_path,
        "proposal_scope_digest": canonical_digest_v2(normalized_proposal),
        "status": "BLOCKED_CATALOG",
        "capabilities": dict(G1_CAPABILITIES),
        "capability_digest": canonical_digest_v2(G1_CAPABILITIES),
        "created_at": created_at,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    return normalize_queue_v4(candidate, policy=policy)


EXECUTION_LEASE_COMMON_FIELDS = {
    "lease_id",
    "experiment_id",
    "gate_kind",
    "gate_contract_version",
    "execution_kind",
    "phase",
    "capability_digest",
    "command_digest",
    "execution_commit",
    "verified_current_main_sha",
    "durable_ledger_backend_identity_digest",
    "durable_ledger_head_sequence",
    "durable_ledger_head_digest",
    "grant_reservation_receipt_digest",
    "verifier_code_digest",
    "executor_code_digest",
    "policy_digest",
    "schema_digest",
    "github_evidence_digest",
    "issued_at",
    "expires_at",
    "human_supervisor_identity",
    "retry_budget",
    "formal_buy",
    "send_order",
    "stake",
}
EXECUTION_LEASE_PHASE_FIELDS = {
    "PREPARATION": {"proposal_scope_digest", "reference_catalog_entry_digest"},
    "HISTORICAL_RUN_REPLICA": {
        "proposal_scope_digest",
        "reference_catalog_release_digest",
        "reference_catalog_entry_digest",
        "run_scope_digest",
        "replica_id",
    },
}

LEASE_OPERATION_RECORD_COMMON_FIELDS = {
    "schema_version",
    "record_kind",
    "operation_id",
    "subject_kind",
    "subject_id",
    "operation_kind",
    "operation_sequence",
    "previous_operation_digest",
    "lifecycle_authorizer_event_digest",
    "lease_id",
    "lease_digest",
    "capability_digest",
    "operation_binding",
    "occurred_at",
    "writer_identity_digest",
    "policy_digest",
    "schema_digest",
    "formal_buy",
    "send_order",
    "stake",
}

OPERATION_BINDING_FIELDS = {
    "EXPERIMENT_LEASE_ISSUED": {
        "experiment_id",
        "phase",
        "proposal_scope_digest",
        "grant_reservation_digest",
    },
    "EXPERIMENT_RUN_REPLICA_LEASE_ISSUED": {
        "experiment_id",
        "phase",
        "proposal_scope_digest",
        "run_scope_digest",
        "replica_id",
        "grant_reservation_digest",
    },
    "EXPERIMENT_PREPARATION_LEASE_CONSUMED": {
        "experiment_id",
        "phase",
        "proposal_scope_digest",
        "dispatch_reservation_digest",
    },
    "EXPERIMENT_RUN_REPLICA_LEASE_CONSUMED": {
        "experiment_id",
        "phase",
        "run_scope_digest",
        "replica_id",
        "dispatch_reservation_digest",
    },
    "CATALOG_PROVIDER_LEASE_ISSUED": {
        "catalog_publication_scope_id",
        "catalog_publication_scope_digest",
        "provider_identity_digest",
        "grant_reservation_digest",
    },
    "CATALOG_PROVIDER_LEASE_CONSUMED": {
        "catalog_publication_scope_id",
        "catalog_publication_scope_digest",
        "provider_identity_digest",
        "dispatch_reservation_digest",
    },
}

OPERATION_BINDING_CONTRACTS = {
    "EXPERIMENT_LEASE_ISSUED": {
        "subject_kind": "EXPERIMENT",
        "subject_id_field": "experiment_id",
        "phase": "PREPARATION",
        "capability_phase": "preparation_a0",
    },
    "EXPERIMENT_RUN_REPLICA_LEASE_ISSUED": {
        "subject_kind": "EXPERIMENT",
        "subject_id_field": "experiment_id",
        "phase": "HISTORICAL_RUN_REPLICA",
        "capability_phase": "historical_reproduction",
    },
    "EXPERIMENT_PREPARATION_LEASE_CONSUMED": {
        "subject_kind": "EXPERIMENT",
        "subject_id_field": "experiment_id",
        "phase": "PREPARATION",
        "capability_phase": "preparation_a0",
    },
    "EXPERIMENT_RUN_REPLICA_LEASE_CONSUMED": {
        "subject_kind": "EXPERIMENT",
        "subject_id_field": "experiment_id",
        "phase": "HISTORICAL_RUN_REPLICA",
        "capability_phase": "historical_reproduction",
    },
    "CATALOG_PROVIDER_LEASE_ISSUED": {
        "subject_kind": "CATALOG_PUBLICATION_SCOPE",
        "subject_id_field": "catalog_publication_scope_id",
        "phase": None,
        "capability_phase": "catalog_release_maintenance",
    },
    "CATALOG_PROVIDER_LEASE_CONSUMED": {
        "subject_kind": "CATALOG_PUBLICATION_SCOPE",
        "subject_id_field": "catalog_publication_scope_id",
        "phase": None,
        "capability_phase": "catalog_release_maintenance",
    },
}

ACTIVATION_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_kind",
    "ledger_contract_version",
    "policy_digest",
    "schema_digest",
    "activation_epoch",
    "backend_identity_digest",
    "transaction_id",
    "idempotency_key",
    "source_main_sha",
    "source_registry_blob_sha",
    "source_registry_content_sha256",
    "imported_event_chain_digest",
    "imported_experiment_head_set_digest",
    "imported_comment_id_set_digest",
    "previous_active_backend_digest",
    "new_active_backend_digest",
    "new_global_head_sequence",
    "new_global_head_digest",
    "activated_at",
    "writer_identity_digest",
    "signer_or_attestation_digest",
    "formal_buy",
    "send_order",
    "stake",
}

EXECUTION_ISSUE_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_kind",
    "ledger_contract_version",
    "policy_digest",
    "schema_digest",
    "backend_identity_digest",
    "transaction_id",
    "idempotency_key",
    "previous_global_head_sequence",
    "previous_global_head_digest",
    "new_global_head_sequence",
    "new_global_head_digest",
    "previous_experiment_operation_sequence",
    "previous_experiment_operation_digest",
    "new_experiment_operation_sequence",
    "new_experiment_operation_digest",
    "event_digest",
    "grant_reservation_digest",
    "lease_id",
    "lease_digest",
    "issued_at",
    "writer_identity_digest",
    "signer_or_attestation_digest",
    "formal_buy",
    "send_order",
    "stake",
}

EXECUTION_CONSUMPTION_RECEIPT_COMMON_FIELDS = {
    "schema_version",
    "receipt_kind",
    "ledger_contract_version",
    "policy_digest",
    "schema_digest",
    "backend_identity_digest",
    "transaction_id",
    "idempotency_key",
    "previous_global_head_sequence",
    "previous_global_head_digest",
    "new_global_head_sequence",
    "new_global_head_digest",
    "previous_experiment_operation_sequence",
    "previous_experiment_operation_digest",
    "new_experiment_operation_sequence",
    "new_experiment_operation_digest",
    "lease_id",
    "lease_digest",
    "phase_binding",
    "consumed_at",
    "writer_identity_digest",
    "signer_or_attestation_digest",
    "formal_buy",
    "send_order",
    "stake",
}

CATALOG_PROVIDER_ISSUE_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_kind",
    "ledger_contract_version",
    "policy_digest",
    "schema_digest",
    "backend_identity_digest",
    "transaction_id",
    "idempotency_key",
    "previous_global_head_sequence",
    "previous_global_head_digest",
    "new_global_head_sequence",
    "new_global_head_digest",
    "previous_catalog_operation_sequence",
    "previous_catalog_operation_digest",
    "new_catalog_operation_sequence",
    "new_catalog_operation_digest",
    "event_digest",
    "grant_reservation_digest",
    "catalog_publication_scope_digest",
    "provider_lease_id",
    "provider_lease_digest",
    "provider_identity_digest",
    "issued_at",
    "writer_identity_digest",
    "signer_or_attestation_digest",
    "formal_buy",
    "send_order",
    "stake",
}

CATALOG_PROVIDER_CONSUMPTION_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_kind",
    "ledger_contract_version",
    "policy_digest",
    "schema_digest",
    "backend_identity_digest",
    "transaction_id",
    "idempotency_key",
    "previous_global_head_sequence",
    "previous_global_head_digest",
    "new_global_head_sequence",
    "new_global_head_digest",
    "previous_catalog_operation_sequence",
    "previous_catalog_operation_digest",
    "new_catalog_operation_sequence",
    "new_catalog_operation_digest",
    "catalog_publication_scope_digest",
    "provider_lease_id",
    "provider_lease_digest",
    "provider_identity_digest",
    "dispatch_reservation_id",
    "dispatch_reservation_digest",
    "consumed_at",
    "writer_identity_digest",
    "signer_or_attestation_digest",
    "formal_buy",
    "send_order",
    "stake",
}

CATALOG_PUBLICATION_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_kind",
    "policy_digest",
    "schema_digest",
    "catalog_publication_scope_digest",
    "catalog_publication_approval_evidence_digest",
    "provider_lease_id",
    "provider_lease_digest",
    "provider_identity_digest",
    "provider_code_digest",
    "provider_environment_digest",
    "command_digest",
    "input_logical_source_set_digest",
    "output_catalog_entry_set_digest",
    "snapshot_set_digest",
    "started_at",
    "completed_at",
    "catalog_provider_lease_consumption_receipt_digest",
    "signer_or_attestation_digest",
    "formal_buy",
    "send_order",
    "stake",
}


def normalize_execution_lease_v1(
    value: Any,
    *,
    policy: dict[str, Any],
) -> dict[str, Any]:
    policy = normalize_policy_v2(policy)
    payload = _object(value, "execution lease")
    phase = payload.get("phase")
    if phase not in EXECUTION_LEASE_PHASE_FIELDS:
        raise ValueError("execution lease phase is unsupported")
    _exact(payload, EXECUTION_LEASE_COMMON_FIELDS | EXECUTION_LEASE_PHASE_FIELDS[phase], "execution lease")
    _require_safety_constants(payload, "execution lease")
    constants = {
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "execution_kind": EXECUTION_KIND,
        "retry_budget": 0,
    }
    for field, expected in constants.items():
        observed = payload[field]
        if not _same_json_type_and_value(observed, expected):
            raise ValueError(f"execution lease.{field} must equal {expected!r}")
    normalized = dict(payload)
    for field in (
        "capability_digest",
        "command_digest",
        "durable_ledger_backend_identity_digest",
        "durable_ledger_head_digest",
        "grant_reservation_receipt_digest",
        "verifier_code_digest",
        "executor_code_digest",
        "policy_digest",
        "schema_digest",
        "github_evidence_digest",
        "proposal_scope_digest",
        "reference_catalog_entry_digest",
    ):
        normalized[field] = _sha256(payload[field], f"execution lease.{field}")
    if phase == "HISTORICAL_RUN_REPLICA":
        for field in ("reference_catalog_release_digest", "run_scope_digest"):
            normalized[field] = _sha256(payload[field], f"execution lease.{field}")
        if payload["replica_id"] not in {"clean_a", "clean_b"}:
            raise ValueError("execution lease replica_id must be clean_a or clean_b")
    normalized["lease_id"] = _identifier(payload["lease_id"], "execution lease.lease_id")
    normalized["experiment_id"] = _identifier(payload["experiment_id"], "execution lease.experiment_id")
    normalized["execution_commit"] = _git_sha(payload["execution_commit"], "execution lease.execution_commit")
    normalized["verified_current_main_sha"] = _git_sha(
        payload["verified_current_main_sha"], "execution lease.verified_current_main_sha"
    )
    normalized["durable_ledger_head_sequence"] = _nonnegative_int(
        payload["durable_ledger_head_sequence"], "execution lease.durable_ledger_head_sequence"
    )
    for field in ("issued_at", "expires_at", "human_supervisor_identity"):
        normalized[field] = _string(payload[field], f"execution lease.{field}")
    normalized.update(constants)
    normalized["formal_buy"] = False
    normalized["send_order"] = False
    normalized["stake"] = 0
    expected_capabilities = (
        policy["phase_capabilities"]["preparation_a0"]["flags"]
        if phase == "PREPARATION"
        else policy["phase_capabilities"]["historical_reproduction"]["flags"]
    )
    if normalized["capability_digest"] != canonical_digest_v2(expected_capabilities):
        raise ValueError("execution lease capability digest differs from the frozen phase")
    return normalized


def normalize_lease_operation_record_v1(value: Any) -> dict[str, Any]:
    payload = _exact(value, LEASE_OPERATION_RECORD_COMMON_FIELDS, "lease operation record")
    _require_safety_constants(payload, "lease operation record")
    if (
        isinstance(payload["schema_version"], bool)
        or payload["schema_version"] != 1
        or payload["record_kind"] != "roi_reproduction_lease_operation_v1"
    ):
        raise ValueError("lease operation record identity is invalid")
    operation_kind = _string(payload["operation_kind"], "lease operation record.operation_kind")
    if operation_kind not in OPERATION_BINDING_FIELDS:
        raise ValueError("lease operation kind is unsupported")
    binding = _exact(
        payload["operation_binding"],
        OPERATION_BINDING_FIELDS[operation_kind],
        "lease operation record.operation_binding",
    )
    normalized_binding: dict[str, Any] = {}
    for field, item in binding.items():
        if field.endswith("_digest"):
            normalized_binding[field] = _sha256(item, f"operation_binding.{field}")
        elif field in {"experiment_id", "catalog_publication_scope_id"}:
            normalized_binding[field] = _identifier(item, f"operation_binding.{field}")
        elif field == "replica_id":
            if item not in {"clean_a", "clean_b"}:
                raise ValueError("operation_binding.replica_id must be clean_a or clean_b")
            normalized_binding[field] = item
        elif field == "phase":
            normalized_binding[field] = _string(item, "operation_binding.phase")
        else:
            normalized_binding[field] = item
    operation_contract = OPERATION_BINDING_CONTRACTS[operation_kind]
    expected_phase = operation_contract["phase"]
    if expected_phase is not None and normalized_binding.get("phase") != expected_phase:
        raise ValueError("lease operation phase does not match operation_kind")
    subject_kind = _string(payload["subject_kind"], "lease operation record.subject_kind")
    subject_id = _identifier(payload["subject_id"], "lease operation record.subject_id")
    expected_subject_id = normalized_binding[operation_contract["subject_id_field"]]
    if subject_kind != operation_contract["subject_kind"] or subject_id != expected_subject_id:
        raise ValueError("lease operation subject is not bound to operation_binding")
    capability_digest = _sha256(
        payload["capability_digest"], "lease operation record.capability_digest"
    )
    expected_capability_digest = canonical_digest_v2(
        EXPECTED_PHASE_CAPABILITIES[operation_contract["capability_phase"]]["flags"]
    )
    if capability_digest != expected_capability_digest:
        raise ValueError("lease operation capability digest differs from its frozen phase")
    return {
        "schema_version": 1,
        "record_kind": "roi_reproduction_lease_operation_v1",
        "operation_id": _identifier(payload["operation_id"], "lease operation record.operation_id"),
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "operation_kind": operation_kind,
        "operation_sequence": _nonnegative_int(
            payload["operation_sequence"], "lease operation record.operation_sequence"
        ),
        "previous_operation_digest": _sha256(
            payload["previous_operation_digest"], "lease operation record.previous_operation_digest"
        ),
        "lifecycle_authorizer_event_digest": _sha256(
            payload["lifecycle_authorizer_event_digest"],
            "lease operation record.lifecycle_authorizer_event_digest",
        ),
        "lease_id": _identifier(payload["lease_id"], "lease operation record.lease_id"),
        "lease_digest": _sha256(payload["lease_digest"], "lease operation record.lease_digest"),
        "capability_digest": capability_digest,
        "operation_binding": normalized_binding,
        "occurred_at": _string(payload["occurred_at"], "lease operation record.occurred_at"),
        "writer_identity_digest": _sha256(
            payload["writer_identity_digest"], "lease operation record.writer_identity_digest"
        ),
        "policy_digest": _sha256(payload["policy_digest"], "lease operation record.policy_digest"),
        "schema_digest": _sha256(payload["schema_digest"], "lease operation record.schema_digest"),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def _normalize_receipt_common(
    payload: dict[str, Any],
    *,
    expected_kind: str,
    sequence_fields: tuple[str, ...],
    digest_fields: tuple[str, ...],
    string_fields: tuple[str, ...],
) -> dict[str, Any]:
    _require_safety_constants(payload, "durable ledger receipt")
    if (
        isinstance(payload["schema_version"], bool)
        or payload["schema_version"] != 1
        or payload["receipt_kind"] != expected_kind
    ):
        raise ValueError("durable receipt identity is invalid")
    if isinstance(payload["ledger_contract_version"], bool) or payload["ledger_contract_version"] != 1:
        raise ValueError("durable receipt ledger_contract_version must be integer 1")
    normalized = dict(payload)
    for field in sequence_fields:
        normalized[field] = _nonnegative_int(payload[field], f"durable receipt.{field}")
    for field in digest_fields:
        normalized[field] = _sha256(payload[field], f"durable receipt.{field}")
    for field in string_fields:
        normalized[field] = (
            _identifier(payload[field], f"durable receipt.{field}")
            if field.endswith("_id") or field in {"transaction_id", "idempotency_key"}
            else _string(payload[field], f"durable receipt.{field}")
        )
    for field in sequence_fields:
        if field.startswith("previous_"):
            new_field = "new_" + field.removeprefix("previous_")
            if new_field in normalized and normalized[new_field] != normalized[field] + 1:
                raise ValueError(
                    f"durable receipt.{new_field} must increment {field} by one"
                )
    for field in digest_fields:
        if field.startswith("previous_"):
            new_field = "new_" + field.removeprefix("previous_")
            if new_field in normalized and normalized[new_field] == normalized[field]:
                raise ValueError(f"durable receipt.{new_field} must advance the digest head")
    normalized["schema_version"] = 1
    normalized["receipt_kind"] = expected_kind
    normalized["ledger_contract_version"] = 1
    normalized["formal_buy"] = False
    normalized["send_order"] = False
    normalized["stake"] = 0
    return normalized


def normalize_durable_ledger_receipt_v1(value: Any) -> dict[str, Any]:
    payload = _object(value, "durable ledger receipt")
    kind = payload.get("receipt_kind")
    if kind == "DURABLE_LEDGER_ACTIVATION":
        _exact(payload, ACTIVATION_RECEIPT_FIELDS, "durable ledger activation receipt")
        normalized = _normalize_receipt_common(
            payload,
            expected_kind=kind,
            sequence_fields=("new_global_head_sequence",),
            digest_fields=(
                "policy_digest", "schema_digest", "backend_identity_digest",
                "source_registry_content_sha256", "imported_event_chain_digest",
                "imported_experiment_head_set_digest", "imported_comment_id_set_digest",
                "previous_active_backend_digest", "new_active_backend_digest",
                "new_global_head_digest", "writer_identity_digest", "signer_or_attestation_digest",
            ),
            string_fields=("activation_epoch", "transaction_id", "idempotency_key", "activated_at"),
        )
        normalized["source_main_sha"] = _git_sha(payload["source_main_sha"], "activation source_main_sha")
        normalized["source_registry_blob_sha"] = _string(
            payload["source_registry_blob_sha"], "activation source_registry_blob_sha"
        )
        return normalized
    if kind == "EXECUTION_LEASE_ISSUED":
        _exact(payload, EXECUTION_ISSUE_RECEIPT_FIELDS, "execution lease issue receipt")
        return _normalize_receipt_common(
            payload,
            expected_kind=kind,
            sequence_fields=(
                "previous_global_head_sequence", "new_global_head_sequence",
                "previous_experiment_operation_sequence", "new_experiment_operation_sequence",
            ),
            digest_fields=(
                "policy_digest", "schema_digest", "backend_identity_digest",
                "previous_global_head_digest", "new_global_head_digest",
                "previous_experiment_operation_digest", "new_experiment_operation_digest",
                "event_digest", "grant_reservation_digest", "lease_digest",
                "writer_identity_digest", "signer_or_attestation_digest",
            ),
            string_fields=("transaction_id", "idempotency_key", "lease_id", "issued_at"),
        )
    if kind == "EXECUTION_LEASE_CONSUMED":
        _exact(payload, EXECUTION_CONSUMPTION_RECEIPT_COMMON_FIELDS, "execution lease consumption receipt")
        normalized = _normalize_receipt_common(
            payload,
            expected_kind=kind,
            sequence_fields=(
                "previous_global_head_sequence", "new_global_head_sequence",
                "previous_experiment_operation_sequence", "new_experiment_operation_sequence",
            ),
            digest_fields=(
                "policy_digest", "schema_digest", "backend_identity_digest",
                "previous_global_head_digest", "new_global_head_digest",
                "previous_experiment_operation_digest", "new_experiment_operation_digest",
                "lease_digest", "writer_identity_digest", "signer_or_attestation_digest",
            ),
            string_fields=("transaction_id", "idempotency_key", "lease_id", "consumed_at"),
        )
        phase_binding = _object(payload["phase_binding"], "consumption receipt.phase_binding")
        phase = phase_binding.get("phase")
        if phase == "PREPARATION":
            _exact(phase_binding, {"phase", "proposal_scope_digest"}, "preparation phase_binding")
            normalized["phase_binding"] = {
                "phase": phase,
                "proposal_scope_digest": _sha256(
                    phase_binding["proposal_scope_digest"], "phase_binding.proposal_scope_digest"
                ),
            }
        elif phase == "HISTORICAL_RUN_REPLICA":
            _exact(phase_binding, {"phase", "run_scope_digest", "replica_id"}, "run phase_binding")
            if phase_binding["replica_id"] not in {"clean_a", "clean_b"}:
                raise ValueError("run consumption replica_id is invalid")
            normalized["phase_binding"] = {
                "phase": phase,
                "run_scope_digest": _sha256(
                    phase_binding["run_scope_digest"], "phase_binding.run_scope_digest"
                ),
                "replica_id": phase_binding["replica_id"],
            }
        else:
            raise ValueError("execution consumption phase_binding is invalid")
        return normalized
    if kind == "CATALOG_PROVIDER_LEASE_ISSUED":
        _exact(payload, CATALOG_PROVIDER_ISSUE_RECEIPT_FIELDS, "catalog provider issue receipt")
        return _normalize_receipt_common(
            payload,
            expected_kind=kind,
            sequence_fields=(
                "previous_global_head_sequence", "new_global_head_sequence",
                "previous_catalog_operation_sequence", "new_catalog_operation_sequence",
            ),
            digest_fields=tuple(
                field for field in CATALOG_PROVIDER_ISSUE_RECEIPT_FIELDS
                if field.endswith("_digest")
            ),
            string_fields=(
                "transaction_id", "idempotency_key",
                "provider_lease_id", "issued_at",
            ),
        )
    if kind == "CATALOG_PROVIDER_LEASE_CONSUMED":
        _exact(
            payload,
            CATALOG_PROVIDER_CONSUMPTION_RECEIPT_FIELDS,
            "catalog provider consumption receipt",
        )
        return _normalize_receipt_common(
            payload,
            expected_kind=kind,
            sequence_fields=(
                "previous_global_head_sequence", "new_global_head_sequence",
                "previous_catalog_operation_sequence", "new_catalog_operation_sequence",
            ),
            digest_fields=tuple(
                field for field in CATALOG_PROVIDER_CONSUMPTION_RECEIPT_FIELDS
                if field.endswith("_digest")
            ),
            string_fields=(
                "transaction_id", "idempotency_key",
                "provider_lease_id", "dispatch_reservation_id", "consumed_at",
            ),
        )
    raise ValueError("unsupported durable ledger receipt_kind")


def normalize_catalog_publication_receipt_v1(value: Any) -> dict[str, Any]:
    payload = _exact(value, CATALOG_PUBLICATION_RECEIPT_FIELDS, "catalog publication receipt")
    _require_safety_constants(payload, "catalog publication receipt")
    if (
        isinstance(payload["schema_version"], bool)
        or payload["schema_version"] != 1
        or payload["receipt_kind"] != "CATALOG_PUBLICATION_COMPLETED"
    ):
        raise ValueError("catalog publication receipt identity is invalid")
    normalized = dict(payload)
    for field in CATALOG_PUBLICATION_RECEIPT_FIELDS:
        if field.endswith("_digest"):
            normalized[field] = _sha256(payload[field], f"catalog publication receipt.{field}")
    for field in ("provider_lease_id", "started_at", "completed_at"):
        normalized[field] = _string(payload[field], f"catalog publication receipt.{field}")
    normalized["formal_buy"] = False
    normalized["send_order"] = False
    normalized["stake"] = 0
    return normalized


CATALOG_EVENT_FIELDS = {
    "schema_version",
    "event_kind",
    "event_id",
    "global_sequence",
    "subject_sequence",
    "catalog_publication_scope_id",
    "catalog_publication_scope_digest",
    "catalog_gate_kind",
    "catalog_contract_version",
    "policy_ref",
    "schema_ref",
    "status",
    "previous_status",
    "previous_catalog_event_id",
    "previous_catalog_event_digest",
    "occurred_at",
    "observer_actor",
    "root_of_trust_evidence",
    "github_trust_evidence",
    "durable_ledger_evidence",
    "approval_grant_evidence",
    "revalidated_approval_evidence",
    "provider_lease_issue_receipt_digest",
    "provider_lease_consumption_receipt_digest",
    "catalog_publication_receipt_digest",
    "reference_catalog_release_id",
    "reference_catalog_release_digest",
    "capabilities",
    "capability_digest",
    "result_evidence",
    "artifacts",
    "notes",
    "safety",
}

RELEASE_STATUS_EVENT_FIELDS = {
    "schema_version",
    "event_kind",
    "event_id",
    "global_sequence",
    "release_status_sequence",
    "reference_catalog_release_id",
    "reference_catalog_release_digest",
    "policy_ref",
    "schema_ref",
    "status",
    "previous_status",
    "previous_release_status_event_id",
    "previous_release_status_event_digest",
    "reason_code",
    "status_change_evidence_digest",
    "effective_at",
    "observer_actor",
    "durable_ledger_evidence",
    "signer_or_attestation_digest",
    "notes",
    "safety",
}

EVENT_SAFETY_FIELDS = {
    "event_is_execution_authority",
    "automatic_execution_allowed",
    "production_approved",
    "merge_approved",
    "buy_approved",
    "formal_buy",
    "send_order",
    "stake",
}


def _normalize_event_safety(value: Any, label: str) -> dict[str, Any]:
    payload = _exact(value, EVENT_SAFETY_FIELDS, label)
    expected = {
        "event_is_execution_authority": False,
        "automatic_execution_allowed": False,
        "production_approved": False,
        "merge_approved": False,
        "buy_approved": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    for field, wanted in expected.items():
        observed = payload[field]
        if not _same_json_type_and_value(observed, wanted):
            raise ValueError(f"{label}.{field} must equal {wanted!r}")
    return expected


def _normalize_actor(value: Any, label: str) -> dict[str, str]:
    payload = _exact(value, {"actor_kind", "actor_id"}, label)
    return {
        "actor_kind": _string(payload["actor_kind"], f"{label}.actor_kind"),
        "actor_id": _string(payload["actor_id"], f"{label}.actor_id"),
    }


def _optional_sha(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, label)


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _normalize_evidence_list(value: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    normalized = [_normalize_digest_evidence(item, f"{label}[]") for item in value]
    digests = [item["digest"] for item in normalized]
    if len(digests) != len(set(digests)):
        raise ValueError(f"{label} contains duplicate evidence digests")
    kinds = [item["evidence_kind"] for item in normalized]
    if len(kinds) != len(set(kinds)):
        raise ValueError(f"{label} contains duplicate evidence kinds")
    return sorted(normalized, key=lambda item: (item["evidence_kind"], item["digest"]))


def _require_exact_event_evidence(
    *,
    status: str,
    approvals: list[dict[str, str]],
    revalidated: list[dict[str, str]],
    requirements: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    label: str,
) -> None:
    expected = requirements.get(status)
    if expected is None:
        return
    approval_kinds = tuple(sorted(item["evidence_kind"] for item in approvals))
    revalidated_kinds = tuple(sorted(item["evidence_kind"] for item in revalidated))
    expected_approvals = tuple(sorted(expected[0]))
    expected_revalidated = tuple(sorted(expected[1]))
    if approval_kinds != expected_approvals:
        raise ValueError(
            f"{label} approval evidence must be exactly {expected_approvals!r}"
        )
    if revalidated_kinds != expected_revalidated:
        raise ValueError(
            f"{label} revalidated evidence must be exactly {expected_revalidated!r}"
        )


def _normalize_event_paths(value: Any) -> dict[str, str | None]:
    fields = {"proposal_scope", "queue", "run_scope", "result", "review"}
    payload = _exact(value, fields, "registry event.paths")
    normalized: dict[str, str | None] = {}
    for field in sorted(fields):
        normalized[field] = (
            None
            if payload[field] is None
            else _repository_path(payload[field], f"registry event.paths.{field}")
        )
    return normalized


def validate_experiment_transition(previous_status: str | None, status: str) -> None:
    if status not in EXPERIMENT_STATES:
        raise ValueError(f"unknown experiment status: {status!r}")
    if previous_status is None:
        if status != "PROPOSED":
            raise ValueError("the first experiment event must be PROPOSED")
        return
    if previous_status not in EXPERIMENT_TRANSITIONS:
        raise ValueError(f"unknown previous experiment status: {previous_status!r}")
    if status not in EXPERIMENT_TRANSITIONS[previous_status]:
        raise ValueError(f"illegal experiment transition: {previous_status} -> {status}")


def validate_catalog_transition(previous_status: str | None, status: str) -> None:
    if status not in CATALOG_STATES:
        raise ValueError(f"unknown catalog status: {status!r}")
    if previous_status is None:
        if status != "CATALOG_PUBLICATION_SCOPE_PROPOSED":
            raise ValueError("the first catalog event must be scope proposed")
        return
    if previous_status not in CATALOG_TRANSITIONS or status not in CATALOG_TRANSITIONS[previous_status]:
        raise ValueError(f"illegal catalog transition: {previous_status} -> {status}")


def validate_release_status_transition(previous_status: str, status: str) -> None:
    if previous_status not in RELEASE_STATUS_TRANSITIONS:
        raise ValueError(f"unknown previous release status: {previous_status!r}")
    if status not in RELEASE_STATUS_TRANSITIONS[previous_status]:
        raise ValueError(f"illegal release status transition: {previous_status} -> {status}")


def _experiment_status_capabilities(status: str, policy: dict[str, Any]) -> dict[str, bool | int]:
    if status == "PREPARING":
        return policy["phase_capabilities"]["preparation_a0"]["flags"]
    if status in {"CATALOG_BOUND", "RUN_APPROVAL_REQUIRED", "APPROVED_TO_RUN"}:
        return policy["phase_capabilities"]["catalog_binding"]["flags"]
    if status == "RUNNING":
        return policy["phase_capabilities"]["historical_reproduction"]["flags"]
    if status in {
        "REVIEW_REQUIRED",
        "ACKNOWLEDGED_REPRODUCTION_RESULT",
        "REPRODUCED",
        "RECONSTRUCTED_NOT_REPRODUCED",
        "REPRODUCTION_FAILED",
        "REJECTED",
    }:
        return policy["phase_capabilities"]["review"]["flags"]
    return policy["phase_capabilities"]["design_g1"]["flags"]


def _catalog_status_capabilities(status: str, policy: dict[str, Any]) -> dict[str, bool | int]:
    if status == "CATALOG_PUBLISHING":
        return policy["phase_capabilities"]["catalog_release_maintenance"]["flags"]
    return policy["phase_capabilities"]["design_g1"]["flags"]


def normalize_registry_event_v4(
    value: Any,
    *,
    policy: dict[str, Any],
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
    policy = normalize_policy_v2(policy)
    payload = _exact(value, REGISTRY_EVENT_FIELDS, "ROI reproduction registry event")
    identity = _bind_artifact_identity(
        payload,
        schema_version=EVENT_SCHEMA_VERSION,
        schema_kind="registry_event",
        policy_sha256=policy_sha256,
        schema_sha256=schema_sha256,
        label="registry event",
    )
    status = _string(payload["status"], "registry event.status")
    previous_status = _optional_string(payload["previous_status"], "registry event.previous_status")
    validate_experiment_transition(previous_status, status)
    global_sequence = _nonnegative_int(payload["global_sequence"], "registry event.global_sequence")
    experiment_sequence = _nonnegative_int(
        payload["experiment_sequence"], "registry event.experiment_sequence"
    )
    if global_sequence < 1 or experiment_sequence < 1:
        raise ValueError("registry event sequences must start at 1")
    previous_event_id = _optional_string(
        payload["previous_experiment_event_id"], "registry event.previous_experiment_event_id"
    )
    previous_event_digest = _optional_sha(
        payload["previous_experiment_event_digest"],
        "registry event.previous_experiment_event_digest",
    )
    if previous_status is None:
        if experiment_sequence != 1 or previous_event_id is not None or previous_event_digest is not None:
            raise ValueError("initial registry event must have sequence 1 and no previous event")
    elif experiment_sequence <= 1 or previous_event_id is None or previous_event_digest is None:
        raise ValueError("non-initial registry event must bind its previous experiment event")
    capabilities = normalize_capabilities(payload["capabilities"], label="registry event.capabilities")
    if capabilities != _experiment_status_capabilities(status, policy):
        raise ValueError("registry event capabilities differ from the frozen status phase")
    if payload["capability_digest"] != canonical_digest_v2(capabilities):
        raise ValueError("registry event capability_digest does not match capabilities")
    safety = _normalize_event_safety(payload["safety"], "registry event.safety")
    approvals = _normalize_evidence_list(
        payload["approval_grant_evidence"], "registry event.approval_grant_evidence"
    )
    revalidated = _normalize_evidence_list(
        payload["revalidated_approval_evidence"],
        "registry event.revalidated_approval_evidence",
    )
    _require_exact_event_evidence(
        status=status,
        approvals=approvals,
        revalidated=revalidated,
        requirements=EXPERIMENT_EVENT_EVIDENCE_REQUIREMENTS,
        label=f"registry event status {status}",
    )
    proposal_digest = _sha256(
        payload["proposal_scope_digest"], "registry event.proposal_scope_digest"
    )
    run_digest = _optional_sha(payload["run_scope_digest"], "registry event.run_scope_digest")
    review_digest = _optional_sha(payload["review_digest"], "registry event.review_digest")
    result_evidence = _normalize_optional_digest_evidence(
        payload["result_evidence"], "registry event.result_evidence"
    )
    execution_lease_receipt = _normalize_optional_digest_evidence(
        payload["execution_lease_receipt"], "registry event.execution_lease_receipt"
    )
    paths = _normalize_event_paths(payload["paths"])
    statuses_requiring_run = {
        "RUN_APPROVAL_REQUIRED",
        "APPROVED_TO_RUN",
        "RUNNING",
        "REVIEW_REQUIRED",
        "ACKNOWLEDGED_REPRODUCTION_RESULT",
        "REPRODUCED",
        "RECONSTRUCTED_NOT_REPRODUCED",
        "REPRODUCTION_FAILED",
        "REJECTED",
    }
    statuses_requiring_review = {
        "REVIEW_REQUIRED",
        "ACKNOWLEDGED_REPRODUCTION_RESULT",
        "REPRODUCED",
        "RECONSTRUCTED_NOT_REPRODUCED",
        "REPRODUCTION_FAILED",
        "REJECTED",
    }
    if status in statuses_requiring_run and run_digest is None:
        raise ValueError(f"registry status {status} requires run_scope_digest")
    if status not in statuses_requiring_run and run_digest is not None:
        raise ValueError(f"registry status {status} forbids preloaded run_scope_digest")
    if status in statuses_requiring_review and review_digest is None:
        raise ValueError(f"registry status {status} requires review_digest")
    if status in statuses_requiring_review and result_evidence is None:
        raise ValueError(f"registry status {status} requires result_evidence")
    if status not in statuses_requiring_review and (
        review_digest is not None or result_evidence is not None
    ):
        raise ValueError(f"registry status {status} forbids preloaded review or result evidence")
    statuses_requiring_lease = {"PREPARING", "RUNNING"}
    if status in statuses_requiring_lease and execution_lease_receipt is None:
        raise ValueError(f"registry status {status} requires consumed lease evidence")
    if status not in statuses_requiring_lease and execution_lease_receipt is not None:
        raise ValueError(f"registry status {status} forbids preloaded execution lease evidence")
    expected_lease_kind = {
        "PREPARING": "PREPARATION_LEASE_CONSUMED",
        "RUNNING": "HISTORICAL_REPRODUCTION_LEASE_SET_CONSUMED",
    }.get(status)
    if (
        execution_lease_receipt is not None
        and execution_lease_receipt["evidence_kind"] != expected_lease_kind
    ):
        raise ValueError(f"registry status {status} has the wrong lease evidence kind")
    if result_evidence is not None and result_evidence["evidence_kind"] != "VALIDATED_REPRODUCTION_RESULT":
        raise ValueError("registry result evidence must be a validated reproduction result")
    if status in statuses_requiring_run and paths["run_scope"] is None:
        raise ValueError(f"registry status {status} requires a run-scope path")
    if status not in statuses_requiring_run and paths["run_scope"] is not None:
        raise ValueError(f"registry status {status} forbids a preloaded run-scope path")
    if status in statuses_requiring_review:
        if paths["result"] is None or paths["review"] is None:
            raise ValueError(f"registry status {status} requires result and review paths")
    elif paths["result"] is not None or paths["review"] is not None:
        raise ValueError(f"registry status {status} forbids preloaded result or review paths")
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": _identifier(payload["event_id"], "registry event.event_id"),
        "global_sequence": global_sequence,
        "experiment_sequence": experiment_sequence,
        "experiment_id": _identifier(payload["experiment_id"], "registry event.experiment_id"),
        "gate_kind": GATE_KIND,
        "gate_contract_version": GATE_CONTRACT_VERSION,
        **identity,
        "status": status,
        "previous_status": previous_status,
        "previous_experiment_event_id": previous_event_id,
        "previous_experiment_event_digest": previous_event_digest,
        "occurred_at": _string(payload["occurred_at"], "registry event.occurred_at"),
        "observer_actor": _normalize_actor(payload["observer_actor"], "registry event.observer_actor"),
        "proposal_scope_digest": proposal_digest,
        "reference_catalog_release_digest": _sha256(
            payload["reference_catalog_release_digest"],
            "registry event.reference_catalog_release_digest",
        ),
        "reference_catalog_entry_digest": _sha256(
            payload["reference_catalog_entry_digest"],
            "registry event.reference_catalog_entry_digest",
        ),
        "run_scope_digest": run_digest,
        "review_digest": review_digest,
        "root_of_trust_evidence": _normalize_optional_digest_evidence(
            payload["root_of_trust_evidence"], "registry event.root_of_trust_evidence"
        ),
        "github_trust_evidence": _normalize_optional_digest_evidence(
            payload["github_trust_evidence"], "registry event.github_trust_evidence"
        ),
        "durable_ledger_evidence": _normalize_optional_digest_evidence(
            payload["durable_ledger_evidence"], "registry event.durable_ledger_evidence"
        ),
        "legacy_registry_import_evidence": _normalize_optional_digest_evidence(
            payload["legacy_registry_import_evidence"],
            "registry event.legacy_registry_import_evidence",
        ),
        "approval_grant_evidence": approvals,
        "revalidated_approval_evidence": revalidated,
        "capabilities": capabilities,
        "capability_digest": canonical_digest_v2(capabilities),
        "execution_kind": EXECUTION_KIND,
        "execution_lease_receipt": execution_lease_receipt,
        "result_evidence": result_evidence,
        "paths": paths,
        "artifacts": _normalize_refs(
            payload["artifacts"], "registry event.artifacts", allow_empty=True
        ),
        "notes": _semantic_strings(payload["notes"], "registry event.notes", allow_empty=True),
        "safety": safety,
    }


def normalize_catalog_publication_event_v1(
    value: Any,
    *,
    policy: dict[str, Any],
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
    policy = normalize_policy_v2(policy)
    payload = _exact(value, CATALOG_EVENT_FIELDS, "catalog publication event")
    if (
        isinstance(payload["schema_version"], bool)
        or payload["schema_version"] != 1
        or payload["event_kind"] != CATALOG_EVENT_KIND
        or payload["catalog_gate_kind"] != CATALOG_GATE_KIND
        or isinstance(payload["catalog_contract_version"], bool)
        or payload["catalog_contract_version"] != CATALOG_CONTRACT_VERSION
    ):
        raise ValueError("catalog publication event kind is invalid")
    policy_ref = normalize_ref(payload["policy_ref"], "catalog publication event.policy_ref")
    schema_ref = normalize_ref(payload["schema_ref"], "catalog publication event.schema_ref")
    if (
        policy_ref["path"] != POLICY_PATH
        or schema_ref["path"] != SCHEMA_PATHS["catalog_publication_event"]
    ):
        raise ValueError("catalog publication event uses a non-code-owned policy/schema")
    if policy_sha256 is not None and policy_ref["sha256"] != policy_sha256:
        raise ValueError("catalog publication event policy digest mismatch")
    if schema_sha256 is not None and schema_ref["sha256"] != schema_sha256:
        raise ValueError("catalog publication event schema digest mismatch")
    status = _string(payload["status"], "catalog publication event.status")
    previous_status = _optional_string(
        payload["previous_status"], "catalog publication event.previous_status"
    )
    validate_catalog_transition(previous_status, status)
    global_sequence = _nonnegative_int(payload["global_sequence"], "catalog event.global_sequence")
    subject_sequence = _nonnegative_int(payload["subject_sequence"], "catalog event.subject_sequence")
    if global_sequence < 1 or subject_sequence < 1:
        raise ValueError("catalog event sequences must start at 1")
    previous_id = _optional_string(
        payload["previous_catalog_event_id"], "catalog event.previous_catalog_event_id"
    )
    previous_digest = _optional_sha(
        payload["previous_catalog_event_digest"], "catalog event.previous_catalog_event_digest"
    )
    if previous_status is None:
        if subject_sequence != 1 or previous_id is not None or previous_digest is not None:
            raise ValueError("initial catalog event must have sequence 1 and no previous event")
    elif subject_sequence <= 1 or previous_id is None or previous_digest is None:
        raise ValueError("non-initial catalog event must bind its previous event")
    capabilities = normalize_capabilities(payload["capabilities"], label="catalog event.capabilities")
    if capabilities != _catalog_status_capabilities(status, policy):
        raise ValueError("catalog event capabilities differ from the frozen status phase")
    if payload["capability_digest"] != canonical_digest_v2(capabilities):
        raise ValueError("catalog event capability digest mismatch")
    issue_receipt_digest = _optional_sha(
        payload["provider_lease_issue_receipt_digest"], "catalog event.issue receipt"
    )
    consumption_receipt_digest = _optional_sha(
        payload["provider_lease_consumption_receipt_digest"],
        "catalog event.consume receipt",
    )
    publication_receipt_digest = _optional_sha(
        payload["catalog_publication_receipt_digest"],
        "catalog event.publication receipt",
    )
    release_id = _optional_string(
        payload["reference_catalog_release_id"], "catalog event.release_id"
    )
    release_digest = _optional_sha(
        payload["reference_catalog_release_digest"], "catalog event.release_digest"
    )
    result_evidence = _normalize_optional_digest_evidence(
        payload["result_evidence"], "catalog event.result_evidence"
    )
    if status == "CATALOG_PUBLISHING" and (
        issue_receipt_digest is None or consumption_receipt_digest is None
    ):
        raise ValueError("CATALOG_PUBLISHING requires provider issue and consumption receipts")
    if status == "CATALOG_PUBLISHED" and any(
        item is None
        for item in (
            issue_receipt_digest,
            consumption_receipt_digest,
            publication_receipt_digest,
            release_id,
            release_digest,
            result_evidence,
        )
    ):
        raise ValueError("CATALOG_PUBLISHED requires complete publication evidence")
    if status == "CATALOG_PUBLICATION_FAILED" and (
        issue_receipt_digest is None
        or consumption_receipt_digest is None
        or result_evidence is None
    ):
        raise ValueError("CATALOG_PUBLICATION_FAILED requires issue, consumption, and result evidence")
    if status in {
        "CATALOG_PUBLICATION_SCOPE_PROPOSED",
        "APPROVED_TO_PUBLISH_REFERENCE_CATALOG",
    } and any(
        item is not None
        for item in (
            issue_receipt_digest,
            consumption_receipt_digest,
            publication_receipt_digest,
            release_id,
            release_digest,
            result_evidence,
        )
    ):
        raise ValueError(f"{status} forbids preloaded provider or publication evidence")
    if status == "CATALOG_PUBLISHING" and any(
        item is not None
        for item in (publication_receipt_digest, release_id, release_digest, result_evidence)
    ):
        raise ValueError("CATALOG_PUBLISHING forbids preloaded publication output evidence")
    if status == "CATALOG_PUBLICATION_FAILED" and any(
        item is not None
        for item in (publication_receipt_digest, release_id, release_digest)
    ):
        raise ValueError("CATALOG_PUBLICATION_FAILED forbids successful publication evidence")
    expected_result_kind = {
        "CATALOG_PUBLISHED": "CATALOG_PUBLICATION_COMPLETED",
        "CATALOG_PUBLICATION_FAILED": "CATALOG_PUBLICATION_FAILED",
    }.get(status)
    if result_evidence is not None and result_evidence["evidence_kind"] != expected_result_kind:
        raise ValueError(f"catalog status {status} has the wrong result evidence kind")
    approvals = _normalize_evidence_list(
        payload["approval_grant_evidence"], "catalog event.approval_grant_evidence"
    )
    revalidated = _normalize_evidence_list(
        payload["revalidated_approval_evidence"],
        "catalog event.revalidated_approval_evidence",
    )
    _require_exact_event_evidence(
        status=status,
        approvals=approvals,
        revalidated=revalidated,
        requirements=CATALOG_EVENT_EVIDENCE_REQUIREMENTS,
        label=f"catalog event status {status}",
    )
    return {
        "schema_version": 1,
        "event_kind": CATALOG_EVENT_KIND,
        "event_id": _identifier(payload["event_id"], "catalog event.event_id"),
        "global_sequence": global_sequence,
        "subject_sequence": subject_sequence,
        "catalog_publication_scope_id": _identifier(
            payload["catalog_publication_scope_id"], "catalog event.scope_id"
        ),
        "catalog_publication_scope_digest": _sha256(
            payload["catalog_publication_scope_digest"], "catalog event.scope_digest"
        ),
        "catalog_gate_kind": CATALOG_GATE_KIND,
        "catalog_contract_version": CATALOG_CONTRACT_VERSION,
        "policy_ref": policy_ref,
        "schema_ref": schema_ref,
        "status": status,
        "previous_status": previous_status,
        "previous_catalog_event_id": previous_id,
        "previous_catalog_event_digest": previous_digest,
        "occurred_at": _string(payload["occurred_at"], "catalog event.occurred_at"),
        "observer_actor": _normalize_actor(payload["observer_actor"], "catalog event.observer_actor"),
        "root_of_trust_evidence": _normalize_optional_digest_evidence(
            payload["root_of_trust_evidence"], "catalog event.root_of_trust_evidence"
        ),
        "github_trust_evidence": _normalize_optional_digest_evidence(
            payload["github_trust_evidence"], "catalog event.github_trust_evidence"
        ),
        "durable_ledger_evidence": _normalize_optional_digest_evidence(
            payload["durable_ledger_evidence"], "catalog event.durable_ledger_evidence"
        ),
        "approval_grant_evidence": approvals,
        "revalidated_approval_evidence": revalidated,
        "provider_lease_issue_receipt_digest": issue_receipt_digest,
        "provider_lease_consumption_receipt_digest": consumption_receipt_digest,
        "catalog_publication_receipt_digest": publication_receipt_digest,
        "reference_catalog_release_id": release_id,
        "reference_catalog_release_digest": release_digest,
        "capabilities": capabilities,
        "capability_digest": canonical_digest_v2(capabilities),
        "result_evidence": result_evidence,
        "artifacts": _normalize_refs(payload["artifacts"], "catalog event.artifacts", allow_empty=True),
        "notes": _semantic_strings(payload["notes"], "catalog event.notes", allow_empty=True),
        "safety": _normalize_event_safety(payload["safety"], "catalog event.safety"),
    }


def normalize_catalog_release_status_event_v1(
    value: Any,
    *,
    policy_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
    payload = _exact(value, RELEASE_STATUS_EVENT_FIELDS, "catalog release status event")
    if (
        isinstance(payload["schema_version"], bool)
        or payload["schema_version"] != 1
        or payload["event_kind"] != CATALOG_RELEASE_STATUS_EVENT_KIND
    ):
        raise ValueError("catalog release status event identity is invalid")
    policy_ref = normalize_ref(payload["policy_ref"], "release status event.policy_ref")
    schema_ref = normalize_ref(payload["schema_ref"], "release status event.schema_ref")
    if policy_ref["path"] != POLICY_PATH or schema_ref["path"] != SCHEMA_PATHS["catalog_release_status_event"]:
        raise ValueError("catalog release status event uses a non-code-owned policy/schema")
    if policy_sha256 is not None and policy_ref["sha256"] != policy_sha256:
        raise ValueError("catalog release status policy digest mismatch")
    if schema_sha256 is not None and schema_ref["sha256"] != schema_sha256:
        raise ValueError("catalog release status schema digest mismatch")
    status = _string(payload["status"], "release status event.status")
    previous_status = _string(payload["previous_status"], "release status event.previous_status")
    validate_release_status_transition(previous_status, status)
    previous_id = _optional_string(
        payload["previous_release_status_event_id"], "release status event.previous_event_id"
    )
    previous_digest = _optional_sha(
        payload["previous_release_status_event_digest"], "release status event.previous_event_digest"
    )
    sequence = _nonnegative_int(
        payload["release_status_sequence"], "release status event.release_status_sequence"
    )
    if previous_status == "INITIAL":
        if sequence != 1 or previous_id is not None or previous_digest is not None:
            raise ValueError("ACTIVE must be the first release-status event")
    elif sequence <= 1 or previous_id is None or previous_digest is None:
        raise ValueError("REVOKED must bind the previous ACTIVE event")
    global_sequence = _nonnegative_int(
        payload["global_sequence"], "release status event.global_sequence"
    )
    if global_sequence < 1:
        raise ValueError("release status global_sequence must start at 1")
    return {
        "schema_version": 1,
        "event_kind": CATALOG_RELEASE_STATUS_EVENT_KIND,
        "event_id": _identifier(payload["event_id"], "release status event.event_id"),
        "global_sequence": global_sequence,
        "release_status_sequence": sequence,
        "reference_catalog_release_id": _identifier(
            payload["reference_catalog_release_id"], "release status event.release_id"
        ),
        "reference_catalog_release_digest": _sha256(
            payload["reference_catalog_release_digest"], "release status event.release_digest"
        ),
        "policy_ref": policy_ref,
        "schema_ref": schema_ref,
        "status": status,
        "previous_status": previous_status,
        "previous_release_status_event_id": previous_id,
        "previous_release_status_event_digest": previous_digest,
        "reason_code": _string(payload["reason_code"], "release status event.reason_code"),
        "status_change_evidence_digest": _sha256(
            payload["status_change_evidence_digest"], "release status event.change evidence"
        ),
        "effective_at": _string(payload["effective_at"], "release status event.effective_at"),
        "observer_actor": _normalize_actor(payload["observer_actor"], "release status event.observer_actor"),
        "durable_ledger_evidence": _normalize_digest_evidence(
            payload["durable_ledger_evidence"], "release status event.ledger evidence"
        ),
        "signer_or_attestation_digest": _sha256(
            payload["signer_or_attestation_digest"], "release status event.signer digest"
        ),
        "notes": _semantic_strings(payload["notes"], "release status event.notes", allow_empty=True),
        "safety": _normalize_event_safety(payload["safety"], "release status event.safety"),
    }


def compile_non_authoritative_registry_line_v4_fixture(
    event: dict[str, Any],
    *,
    policy: dict[str, Any],
    fixture_only: bool,
) -> bytes:
    if fixture_only is not True:
        raise ValueError("G1 can compile registry lines only as non-authoritative fixtures")
    normalized = normalize_registry_event_v4(event, policy=policy)
    return canonical_json_bytes_v2(normalized) + b"\n"


def validate_mixed_registry_history(
    raw: bytes,
    *,
    policy: dict[str, Any],
) -> list[bytes]:
    lines = raw.splitlines(keepends=True)
    if raw and not raw.endswith(b"\n"):
        raise ValueError("registry history must end with LF")
    preserved: list[bytes] = []
    seen_v4_event_ids: set[str] = set()
    v4_heads: dict[str, dict[str, Any]] = {}
    previous_v4_global_sequence: int | None = None
    for index, line in enumerate(lines, start=1):
        if not line.endswith(b"\n") or line.endswith(b"\r\n"):
            raise ValueError(f"registry line {index} must use LF")
        try:
            text = line[:-1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"registry line {index} is not UTF-8") from exc
        payload = strict_json_loads_v2(text, label=f"registry line {index}")
        if not isinstance(payload, dict):
            raise ValueError(f"registry line {index} must be an object")
        version = payload.get("schema_version")
        if isinstance(version, bool) or version not in {2, 3, 4}:
            raise ValueError(f"registry line {index} has an unsupported schema_version")
        if version == 4:
            normalized_event = normalize_registry_event_v4(payload, policy=policy)
            if line != canonical_json_bytes_v2(normalized_event) + b"\n":
                raise ValueError(f"registry v4 line {index} is not canonical")
            event_id = normalized_event["event_id"]
            if event_id in seen_v4_event_ids:
                raise ValueError(f"registry v4 line {index} reuses an event_id")
            seen_v4_event_ids.add(event_id)
            global_sequence = normalized_event["global_sequence"]
            if (
                previous_v4_global_sequence is not None
                and global_sequence != previous_v4_global_sequence + 1
            ):
                raise ValueError("registry v4 global sequences must be contiguous")
            previous_v4_global_sequence = global_sequence
            experiment_id = normalized_event["experiment_id"]
            previous_head = v4_heads.get(experiment_id)
            if previous_head is None:
                if normalized_event["experiment_sequence"] != 1:
                    if normalized_event["legacy_registry_import_evidence"] is None:
                        raise ValueError(
                            "a non-initial v4 experiment head requires legacy import evidence"
                        )
                elif (
                    normalized_event["previous_experiment_event_id"] is not None
                    or normalized_event["previous_experiment_event_digest"] is not None
                ):
                    raise ValueError("initial v4 experiment event cannot name a predecessor")
            else:
                if normalized_event["experiment_sequence"] != previous_head["experiment_sequence"] + 1:
                    raise ValueError("registry v4 experiment sequences must be contiguous")
                if normalized_event["previous_status"] != previous_head["status"]:
                    raise ValueError("registry v4 previous_status does not match its head")
                if normalized_event["previous_experiment_event_id"] != previous_head["event_id"]:
                    raise ValueError("registry v4 predecessor event_id does not match its head")
                if (
                    normalized_event["previous_experiment_event_digest"]
                    != canonical_digest_v2(previous_head)
                ):
                    raise ValueError("registry v4 predecessor digest does not match its head")
            v4_heads[experiment_id] = normalized_event
        preserved.append(line)
    return preserved
