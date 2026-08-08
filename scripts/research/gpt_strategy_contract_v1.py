from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scope_contract import (  # noqa: E402
    RUN_SCORE_THRESHOLD,
    SAFE_EXPERIMENT_ID,
    canonical_digest,
    canonical_json_text,
    normalize_proposal_scope,
    proposal_score_total,
)


STRATEGY_SCHEMA_VERSION = 1
STRATEGY_FIELDS = {
    "schema_version",
    "strategy_id",
    "brain_model_id",
    "brain_prompt_hash",
    "context_manifest_hash",
    "proposal_scope",
    "safety",
}
SAFETY_FIELDS = {
    "actual_codex_dispatch",
    "automatic_github_approval",
    "candidate_policy_change",
    "credential_access",
    "external_api_calls",
    "formal_buy",
    "merge",
    "notification_side_effects",
    "production_change",
    "purchase_path_access",
    "real_data_execution",
    "send_order",
    "stake",
}
REQUIRED_FORBIDDEN_COLUMNS = {
    "api_key",
    "credential",
    "current_odds",
    "final_odds",
    "formal_stake_yen",
    "market_rank",
    "official_result",
    "order_payload",
    "payout",
    "popularity",
    "production_path",
    "roi_as_training_label",
    "secret",
    "send_order_true",
}
RISKY_ALLOWED_COLUMNS = REQUIRED_FORBIDDEN_COLUMNS | {
    "final_popularity",
    "market_odds",
    "official_payoff",
    "race_result",
}
BANNED_PATH_PARTS = {
    "buy",
    "credentials",
    "notifications",
    "orders",
    "production",
    "purchase",
    "secrets",
}
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FREE_FORM_COMMAND = re.compile(
    r"(?:^|[\s`])(?:bash|cmd(?:\.exe)?|curl|gh\s+api|git\s+push|"
    r"invoke-webrequest|powershell|pwsh|python\s+-c|sh|start-process|wget)"
    r"(?:\s|$)",
    re.IGNORECASE,
)


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=_strict_object_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read strict JSON from {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(value))
    unexpected = sorted(set(value) - fields)
    if missing:
        raise ValueError(f"{label} is missing required field(s): {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{label} contains unexpected field(s): {', '.join(unexpected)}")


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_nonempty_string(value, field).lower()
    if not FULL_SHA256.fullmatch(text):
        raise ValueError(f"{field} must be a full lowercase SHA-256 digest")
    return text


def _validate_expected_paths(paths: list[str]) -> None:
    for raw_path in paths:
        path = PurePosixPath(raw_path)
        path_tokens = {
            token
            for part in path.parts
            for token in re.split(r"[^a-z0-9]+", part.lower())
            if token
        }
        risky_parts = path_tokens & BANNED_PATH_PARTS
        if risky_parts:
            raise ValueError(
                "expected_changed_paths reaches a prohibited path component: "
                + ", ".join(sorted(risky_parts))
            )


def _normalize_safety(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("safety must be an object")
    _require_exact_fields(value, SAFETY_FIELDS, "safety")
    normalized: dict[str, Any] = {}
    for field in sorted(SAFETY_FIELDS - {"stake"}):
        if value[field] is not False:
            raise ValueError(f"safety.{field} must be false")
        normalized[field] = False
    if value["stake"] != 0 or isinstance(value["stake"], bool):
        raise ValueError("safety.stake must be 0")
    normalized["stake"] = 0
    return normalized


def normalize_strategy(
    value: Any,
    *,
    expected_experiment_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("strategy payload must be a JSON object")
    _require_exact_fields(value, STRATEGY_FIELDS, "strategy payload")
    if value["schema_version"] != STRATEGY_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {STRATEGY_SCHEMA_VERSION}")

    strategy_id = _require_nonempty_string(value["strategy_id"], "strategy_id")
    if not SAFE_EXPERIMENT_ID.fullmatch(strategy_id):
        raise ValueError("strategy_id is not a safe 3-64 character identifier")

    proposal_scope = normalize_proposal_scope(
        value["proposal_scope"],
        expected_experiment_id=expected_experiment_id,
    )
    if proposal_scope["allowed_variant_count"] != 1:
        raise ValueError("allowed_variant_count must be exactly 1")
    if proposal_scope["allowed_threshold_search_count"] != 0:
        raise ValueError("allowed_threshold_search_count must be 0")
    if proposal_score_total(proposal_scope) < RUN_SCORE_THRESHOLD:
        raise ValueError(
            f"proposal score must be at least {RUN_SCORE_THRESHOLD} before dispatch"
        )

    allowed_lower = {name.lower() for name in proposal_scope["allowed_columns"]}
    risky_allowed = sorted(allowed_lower & RISKY_ALLOWED_COLUMNS)
    if risky_allowed:
        raise ValueError(
            "allowed_columns contains market, result, payout, secret, or order data: "
            + ", ".join(risky_allowed)
        )
    forbidden_lower = {name.lower() for name in proposal_scope["forbidden_columns"]}
    missing_forbidden = sorted(REQUIRED_FORBIDDEN_COLUMNS - forbidden_lower)
    if missing_forbidden:
        raise ValueError(
            "forbidden_columns is missing required firewall fields: "
            + ", ".join(missing_forbidden)
        )
    _validate_expected_paths(proposal_scope["expected_changed_paths"])
    for task in proposal_scope["in_scope"]:
        if FREE_FORM_COMMAND.search(task):
            raise ValueError("in_scope must not contain a free-form shell command")

    normalized = {
        "schema_version": STRATEGY_SCHEMA_VERSION,
        "strategy_id": strategy_id,
        "brain_model_id": _require_nonempty_string(
            value["brain_model_id"], "brain_model_id"
        ),
        "brain_prompt_hash": _require_sha256(
            value["brain_prompt_hash"], "brain_prompt_hash"
        ),
        "context_manifest_hash": _require_sha256(
            value["context_manifest_hash"], "context_manifest_hash"
        ),
        "proposal_scope": proposal_scope,
        "safety": _normalize_safety(value["safety"]),
    }
    canonical_json_text(normalized)
    return normalized


def strategy_digest(strategy: dict[str, Any]) -> str:
    return canonical_digest(normalize_strategy(strategy))


def compile_proposal_scope(strategy: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_strategy(strategy)
    return json.loads(canonical_json_text(normalized["proposal_scope"]))
