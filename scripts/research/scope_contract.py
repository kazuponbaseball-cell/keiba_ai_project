from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


SCORE_WEIGHTS = {
    "independent_information": 25,
    "racing_mechanism": 20,
    "outer_oos_failure_evidence": 20,
    "leakage_safety": 15,
    "minimal_falsifiability": 10,
    "acquisition_implementation_cost": 10,
}
RUN_SCORE_THRESHOLD = 75
SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")

PROPOSAL_FIELDS = {
    "experiment_id",
    "title",
    "hypothesis",
    "null_hypothesis",
    "racing_mechanism",
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
    "fold_manifest",
    "purge_embargo",
    "primary_metric",
    "required_effect",
    "rejection_gate",
    "stop_conditions",
    "compute_budget",
    "allowed_variant_count",
    "allowed_threshold_search_count",
    "base_commit",
    "score_components",
    "formal_buy",
    "send_order",
    "stake",
}

RUN_FIELDS = {
    "proposal_scope",
    "proposal_scope_digest",
    "execution_commit_sha",
    "config_hashes",
    "data_input_manifest_hashes",
    "fold_manifest_hash",
    "runner_universe_manifest_hash",
    "dependency_environment_manifest",
    "seed",
    "exact_execution_commands",
    "formal_buy",
    "send_order",
    "stake",
}

# These arrays are semantic sets.  Canonicalization sorts them by Unicode code
# point after requiring unique strings.  Other arrays (especially commands)
# preserve order because order is part of their meaning.
SET_LIKE_PROPOSAL_LISTS = {
    "in_scope",
    "out_of_scope",
    "expected_changed_paths",
    "raw_data_sources",
    "allowed_columns",
    "forbidden_columns",
    "lineage_hash_requirements",
    "rejection_gate",
    "stop_conditions",
}


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read strict JSON from {path}: {exc}") from exc


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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_fields(payload: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(payload))
    unexpected = sorted(set(payload) - required)
    if missing:
        raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{label} contains unexpected field(s): {', '.join(unexpected)}")


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_repository_path(value: Any, field: str) -> str:
    text = _require_nonempty_string(value, field).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a normalized repository-relative path")
    return path.as_posix()


def _require_string_list(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
    repository_paths: bool = False,
    sort_as_set: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{field} must be {qualifier} of strings")
    normalized = [
        (
            _require_repository_path(item, f"{field}[]")
            if repository_paths
            else _require_nonempty_string(item, f"{field}[]")
        )
        for item in value
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must not contain duplicates")
    return sorted(normalized) if sort_as_set else normalized


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field} must be a non-empty JSON object")
    canonical_json_bytes(value)
    return value


def _require_sha256(value: Any, field: str) -> str:
    text = _require_nonempty_string(value, field).lower()
    if not FULL_SHA256.fullmatch(text):
        raise ValueError(f"{field} must be a full lowercase SHA-256 digest")
    return text


def _require_git_sha(value: Any, field: str) -> str:
    text = _require_nonempty_string(value, field).lower()
    if not FULL_GIT_SHA.fullmatch(text):
        raise ValueError(f"{field} must be a full 40-character lowercase Git commit SHA")
    return text


def _normalize_manifest_ref(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object with path and sha256")
    _require_exact_fields(value, {"path", "sha256"}, field)
    return {
        "path": _require_repository_path(value["path"], f"{field}.path"),
        "sha256": _require_sha256(value["sha256"], f"{field}.sha256"),
    }


def _normalize_manifest_refs(value: Any, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list of path/hash objects")
    refs = [_normalize_manifest_ref(item, f"{field}[]") for item in value]
    paths = [item["path"] for item in refs]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{field} must not contain duplicate paths")
    return sorted(refs, key=lambda item: item["path"])


def _normalize_scores(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("score_components must be an object")
    _require_exact_fields(value, set(SCORE_WEIGHTS), "score_components")
    scores: dict[str, int] = {}
    for name, maximum in SCORE_WEIGHTS.items():
        score = value[name]
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(f"score_components.{name} must be an integer")
        if not 0 <= score <= maximum:
            raise ValueError(f"score_components.{name} must be between 0 and {maximum}")
        scores[name] = score
    return scores


def _require_disabled_safety_flags(payload: dict[str, Any], label: str) -> None:
    if payload.get("formal_buy") is not False:
        raise ValueError(f"{label}.formal_buy must be false")
    if payload.get("send_order") is not False:
        raise ValueError(f"{label}.send_order must be false")
    if payload.get("stake") != 0 or isinstance(payload.get("stake"), bool):
        raise ValueError(f"{label}.stake must be 0")


def normalize_proposal_scope(
    value: Any,
    *,
    expected_experiment_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("proposal scope must be a JSON object")
    _require_exact_fields(value, PROPOSAL_FIELDS, "proposal scope")
    _require_disabled_safety_flags(value, "proposal scope")

    experiment_id = _require_nonempty_string(value["experiment_id"], "experiment_id")
    if not SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError("experiment_id is not a safe 3-64 character identifier")
    if expected_experiment_id is not None and experiment_id != expected_experiment_id:
        raise ValueError(
            f"proposal experiment_id mismatch: expected {expected_experiment_id!r}, "
            f"found {experiment_id!r}"
        )

    normalized: dict[str, Any] = {
        "experiment_id": experiment_id,
        "title": _require_nonempty_string(value["title"], "title"),
        "hypothesis": _require_nonempty_string(value["hypothesis"], "hypothesis"),
        "null_hypothesis": _require_nonempty_string(
            value["null_hypothesis"], "null_hypothesis"
        ),
        "racing_mechanism": _require_nonempty_string(
            value["racing_mechanism"], "racing_mechanism"
        ),
        "target_population": _require_nonempty_string(
            value["target_population"], "target_population"
        ),
    }
    for field in SET_LIKE_PROPOSAL_LISTS:
        normalized[field] = _require_string_list(
            value[field],
            field,
            repository_paths=field == "expected_changed_paths",
            sort_as_set=True,
        )
    normalized.update(
        {
            "data_as_of": _require_nonempty_string(value["data_as_of"], "data_as_of"),
            "chronological_fold_design": _require_object(
                value["chronological_fold_design"], "chronological_fold_design"
            ),
            "fold_manifest": _normalize_manifest_ref(value["fold_manifest"], "fold_manifest"),
            "purge_embargo": _require_object(value["purge_embargo"], "purge_embargo"),
            "primary_metric": _require_object(value["primary_metric"], "primary_metric"),
            "required_effect": _require_object(value["required_effect"], "required_effect"),
            "compute_budget": _require_object(value["compute_budget"], "compute_budget"),
            "base_commit": _require_git_sha(value["base_commit"], "base_commit"),
            "score_components": _normalize_scores(value["score_components"]),
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
    )
    for field in ("allowed_variant_count", "allowed_threshold_search_count"):
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        normalized[field] = number
    return normalized


def proposal_score_total(scope: dict[str, Any]) -> int:
    return sum(scope["score_components"].values())


def normalize_run_scope(
    value: Any,
    *,
    proposal_scope: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("run scope must be a JSON object")
    _require_exact_fields(value, RUN_FIELDS, "run scope")
    _require_disabled_safety_flags(value, "run scope")

    normalized_proposal = normalize_proposal_scope(
        value["proposal_scope"],
        expected_experiment_id=proposal_scope["experiment_id"],
    )
    if normalized_proposal != proposal_scope:
        raise ValueError("run scope proposal_scope does not match the frozen proposal scope")
    proposal_digest = canonical_digest(proposal_scope)
    if value["proposal_scope_digest"] != proposal_digest:
        raise ValueError("run scope proposal_scope_digest does not match proposal scope")

    seed = value["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    normalized = {
        "proposal_scope": proposal_scope,
        "proposal_scope_digest": proposal_digest,
        "execution_commit_sha": _require_git_sha(
            value["execution_commit_sha"], "execution_commit_sha"
        ),
        "config_hashes": _normalize_manifest_refs(value["config_hashes"], "config_hashes"),
        "data_input_manifest_hashes": _normalize_manifest_refs(
            value["data_input_manifest_hashes"], "data_input_manifest_hashes"
        ),
        "fold_manifest_hash": _normalize_manifest_ref(
            value["fold_manifest_hash"], "fold_manifest_hash"
        ),
        "runner_universe_manifest_hash": _normalize_manifest_ref(
            value["runner_universe_manifest_hash"], "runner_universe_manifest_hash"
        ),
        "dependency_environment_manifest": _normalize_manifest_ref(
            value["dependency_environment_manifest"], "dependency_environment_manifest"
        ),
        "seed": seed,
        "exact_execution_commands": _require_string_list(
            value["exact_execution_commands"],
            "exact_execution_commands",
        ),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    if normalized["fold_manifest_hash"] != proposal_scope["fold_manifest"]:
        raise ValueError("run scope fold manifest path/hash differs from proposal scope")
    return normalized


def resolve_repository_path(root: Path, relative_path: str) -> Path:
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {relative_path}") from exc
    return resolved


def verify_manifest_ref(root: Path, reference: dict[str, str], field: str) -> None:
    path = resolve_repository_path(root, reference["path"])
    if not path.is_file():
        raise ValueError(f"{field} file not found: {reference['path']}")
    observed = sha256_file(path)
    if observed != reference["sha256"]:
        raise ValueError(
            f"{field} hash changed for {reference['path']}: "
            f"expected {reference['sha256']}, observed {observed}"
        )


def verify_run_materials(root: Path, run_scope: dict[str, Any]) -> None:
    for index, reference in enumerate(run_scope["config_hashes"]):
        verify_manifest_ref(root, reference, f"config_hashes[{index}]")
    for index, reference in enumerate(run_scope["data_input_manifest_hashes"]):
        verify_manifest_ref(root, reference, f"data_input_manifest_hashes[{index}]")
    for field in (
        "fold_manifest_hash",
        "runner_universe_manifest_hash",
        "dependency_environment_manifest",
    ):
        verify_manifest_ref(root, run_scope[field], field)


def load_frozen_proposal(
    root: Path,
    queue: dict[str, Any],
    experiment_id: str,
) -> tuple[dict[str, Any], str, Path]:
    path_value = queue.get("proposal_scope_file")
    if not isinstance(path_value, str):
        raise ValueError("queue is missing proposal_scope_file")
    scope_path = resolve_repository_path(root, _require_repository_path(path_value, "proposal_scope_file"))
    scope = normalize_proposal_scope(
        strict_json_load(scope_path),
        expected_experiment_id=experiment_id,
    )
    digest = canonical_digest(scope)
    if queue.get("proposal_scope_digest") != digest:
        raise ValueError("proposal scope changed after queue creation")
    if queue.get("proposal_scope") != scope:
        raise ValueError("queue proposal_scope differs from canonical proposal scope file")
    return scope, digest, scope_path


def load_frozen_run_scope(
    root: Path,
    path: Path,
    proposal_scope: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    run_scope = normalize_run_scope(strict_json_load(path), proposal_scope=proposal_scope)
    return run_scope, canonical_digest(run_scope)
