from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "EXP-20260821-034"
VARIANT = "exp033_input_canonicalization_v0"
PROPOSAL_DIGEST = "da76a639a01a179f3061adaa14236741c226b223772953bb5551b2d277351021"
PROPOSAL_BYTE_SHA256 = "c6c02b486f1e253f606f4438e4e0a7ea4eb567cd283f00ca37fb25a760d0937e"
DEFAULT_CONFIG = "research/configs/EXP-20260821-034.input_canonicalization_v0.json"
CONFIG_BYTE_SHA256 = "d8af440e91565137f69ce5ed4f5408cd82003215f916d1d7ad3a85a6e29f249b"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
IDENTITY_COLUMNS = ("race_id", "horse_id")
SYNTHETIC_PREFIX = "SYN-"
LINEAGE_SAFE_VERDICT = "certified_asof_safe"
MISSING_REASONS = {
    "not_applicable",
    "not_declared_by_source",
    "not_observed_before_as_of",
    "structural_no_prior_history",
    "source_value_missing",
}
RUNNER_STATUSES = {"declared_active", "scratched"}
ENTRY_STAGES = {"declared_without_draw", "declared_with_draw"}
DRAW_STATUSES = {"confirmed", "scheduled_pending_draw", "scratched"}
PRE_CUTOFF_NONSTARTER_STATUSES = {"scratched", "nonstarter", "excluded"}
WHOLE_RACE_INELIGIBLE_STATUSES = {
    "did_not_finish",
    "race_stopped",
    "disqualified",
    "abnormal",
    "void",
    "dead_heat",
    "result_missing",
}
RESULT_REASON_PRIORITY = (
    "result_missing",
    "did_not_finish",
    "race_stopped",
    "disqualified",
    "abnormal",
    "void",
    "dead_heat",
)
LABEL_INELIGIBILITY_REASONS = {
    "eligible",
    "precutoff_nonstarter_retained",
    "target_labels_absent",
    "no_effective_starters",
    "fewer_than_two_effective_starters",
    "nonstarter_status_not_bound_before_prediction",
    "dead_heat_or_rank_gap",
    *WHOLE_RACE_INELIGIBLE_STATUSES,
}
REAL_DATA_BLOCKER_CODE = "BLOCKED_VERSIONED_REAL_DATA_EXECUTION_CONTRACT"
MANIFEST_KINDS = {
    "runner_universe_manifest",
    "training_source_manifest",
    "target_source_manifest",
    "feature_release_manifest",
    "lineage_manifest",
    "label_eligibility_manifest",
    "environment_manifest",
    "dependency_lock_manifest",
    "release_diff_manifest",
    "canonical_root_manifest",
}
ARTIFACT_NAME_BY_KIND = {
    "runner_universe_manifest": "runner_universe_release",
    "training_source_manifest": "training_source_release",
    "target_source_manifest": "target_source_release",
    "feature_release_manifest": "feature_input_release",
    "lineage_manifest": "lineage_release",
    "label_eligibility_manifest": "label_eligibility_release",
    "environment_manifest": "environment_release",
    "dependency_lock_manifest": "dependency_lock_release",
    "release_diff_manifest": "release_diff",
    "canonical_root_manifest": "canonical_root",
}
MANIFEST_REQUIRED_FIELDS = {
    "experiment_id",
    "manifest_kind",
    "release_family_id",
    "release_version",
    "parent_manifest_digest",
    "as_of",
    "source_cutoff",
    "generator_execution_commit",
    "generator_script_sha256",
    "config_sha256",
    "dependency_environment_sha256",
    "schema_sha256",
    "input_source_paths_and_sha256",
    "output_artifact_paths_and_sha256",
    "row_count",
    "race_count",
    "runner_count",
    "duplicate_count",
    "row_counts",
    "race_counts",
    "identity_counts",
    "duplicate_and_missing_counts",
    "missing_reason_distribution",
    "as_of_verdict_counts",
    "label_eligibility_counts",
    "artifacts",
    "source_time_completeness",
    "source_hash_completeness",
    "certification_status",
    "formal_buy",
    "send_order",
    "stake",
}
MANIFEST_ARTIFACT_FIELDS = {"name", "path", "sha256", "row_count"}
ENVIRONMENT_ARTIFACT_FIELDS = {
    "schema_version",
    "experiment_id",
    "variant",
    "environment_contract_version",
    "python_implementation",
    "python_version",
    "platform",
    "executable_sha256",
    "encoding",
    "line_endings",
    "timezone",
    "locale",
    "pythonhashseed",
    "network_access",
    "filesystem_mtime_as_received_at",
    "formal_buy",
    "send_order",
    "stake",
}
DEPENDENCY_LOCK_ARTIFACT_FIELDS = {
    "schema_version",
    "experiment_id",
    "variant",
    "lock_version",
    "dependency_policy",
    "packages",
    "interpreter_sha256",
    "config_sha256",
    "formal_buy",
    "send_order",
    "stake",
}
RACE_AGGREGATE_FEATURES = {
    "出走頭数",
    "race_front_runner_count",
    "race_front_runner_ratio",
    "race_closer_count",
    "race_closer_ratio",
    "race_early_pressure_score",
    "front_pressure_rank_score",
    "race_surface_top3_rank_score",
    "race_distance_top3_rank_score",
    "race_weight_light_rank_score",
    "race_need_lead_count",
    "race_need_lead_ratio",
    "race_stalker_count_deep",
    "race_midpack_count_deep",
    "race_deep_closer_count",
    "race_pace_collapse_risk",
    "race_slow_pace_risk",
    "solo_lead_potential",
    "pace_fit_score",
    "front_advantage_score",
    "closer_advantage_score",
    "positioning_advantage_score",
}

DECLARED_LABEL_BASE_FIELDS = {
    "race_id",
    "horse_id",
    "runner_status",
    "prediction_event_time",
    "source_event_time",
    "received_at",
    "available_as_of",
    "source_version",
    "source_content_sha256",
    "missing_reason",
}
DECLARED_LABEL_PRE_CUTOFF_FIELDS = {
    "precutoff_starter_status",
    "status_available_pre_cutoff",
}
JSON_SCALAR_TYPES = {type(None), bool, int, float, str}


class ContractError(ValueError):
    """A deterministic fail-closed contract violation."""


class AuthorizationError(ContractError):
    """The requested operation lacks a hash-bound Research OS authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def strict_json_loads(raw: str, *, source: str = "JSON") -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ContractError(f"invalid {source}: {exc}") from exc


def strict_json_load(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read contract JSON {path}: {exc}") from exc
    value = strict_json_loads(raw, source=str(path))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical-JSON serializable: {exc}") from exc


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]], *, sort_key: Sequence[str]) -> bytes:
    materialized = [dict(row) for row in rows]
    materialized.sort(key=lambda row: tuple(str(row[column]) for column in sort_key))
    return b"".join(canonical_json_bytes(row) + b"\n" for row in materialized)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def feature_value_binding_hash(
    *,
    race_id: str,
    horse_id: str,
    feature_name: str,
    feature_value: Any,
    feature_dtype: str,
    prediction_event_time: str,
    transformation_name: str,
    transformation_version: str,
    transformation_code_sha256: str,
    source_content_sha256_set: Sequence[str],
) -> str:
    """Bind a wide value to its immutable lineage without changing its frozen columns."""
    return canonical_digest(
        {
            "binding_version": "exp034_feature_value_binding_v1",
            "race_id": race_id,
            "horse_id": horse_id,
            "feature_name": feature_name,
            "feature_value": feature_value,
            "feature_dtype": feature_dtype,
            "prediction_event_time": prediction_event_time,
            "transformation_name": transformation_name,
            "transformation_version": transformation_version,
            "transformation_code_sha256": transformation_code_sha256,
            "source_content_sha256_set": sorted(set(source_content_sha256_set)),
        }
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _repo_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ContractError(f"path escapes repository: {relative_path}") from exc
    return path


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "<na>"}
    return False


def _require_nonblank_string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and not _is_blank(value), f"{label} must be a nonblank string")
    _require(value == value.strip(), f"{label} must already be canonical without surrounding whitespace")
    return value


def _require_hash(value: Any, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a string")
    text = value
    _require(bool(HASH_RE.fullmatch(text)), f"{label} must be lowercase SHA-256")
    return text


def _require_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    _require(isinstance(row.get("race_id"), str), "race_id must be a string")
    _require(isinstance(row.get("horse_id"), str), "horse_id must be a string")
    race_id = row["race_id"].strip()
    horse_id = row["horse_id"].strip()
    _require(race_id and horse_id, "race_id and horse_id must be nonblank")
    _require(
        row["race_id"] == race_id and row["horse_id"] == horse_id,
        "race_id and horse_id must already be canonical without surrounding whitespace",
    )
    _require(race_id != "__MISSING__" and horse_id != "__MISSING__", "reserved missing identity")
    return race_id, horse_id


def _parse_timestamp(value: Any, label: str) -> datetime:
    _require(isinstance(value, str) and value and value == value.strip(), f"{label} is required in canonical form")
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ContractError(f"{label} must be RFC3339: {value}") from exc
    _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{label} must be timezone-aware")
    normalized = parsed.astimezone(timezone.utc)
    _require(value == utc_text(normalized), f"{label} must be canonical UTC RFC3339 with Z")
    return normalized


def utc_text(value: datetime) -> str:
    _require(value.tzinfo is not None and value.utcoffset() is not None, "timezone-aware datetime required")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_safety_flags(value: Mapping[str, Any]) -> None:
    _require(type(value.get("formal_buy")) is bool and value["formal_buy"] is False, "formal_buy must be boolean false")
    _require(type(value.get("send_order")) is bool and value["send_order"] is False, "send_order must be boolean false")
    _require(type(value.get("stake")) is int and value["stake"] == 0, "stake must be integer zero")


def _verify_hash_ref(
    reference: Mapping[str, Any],
    *,
    allow_crlf_checkout_normalization: bool = False,
) -> Path:
    path = _repo_path(str(reference.get("path", "")))
    _require(path.is_file(), f"frozen contract is missing: {reference.get('path')}")
    expected = _require_hash(reference.get("sha256"), f"hash for {reference.get('path')}")
    raw = path.read_bytes()
    observed = sha256_bytes(raw)
    if observed != expected and allow_crlf_checkout_normalization:
        normalized = raw.replace(b"\r\n", b"\n")
        _require(b"\r" not in normalized, f"bare CR in frozen contract: {reference.get('path')}")
        observed = sha256_bytes(normalized)
    _require(observed == expected, f"frozen contract hash mismatch: {reference.get('path')}")
    return path


@dataclass(frozen=True)
class ContractBundle:
    config_path: Path
    config: dict[str, Any]
    input_contract: dict[str, Any]
    allowlist: dict[str, Any]
    denylist: dict[str, Any]
    exp033_fold: dict[str, Any]
    environment: dict[str, Any]

    @property
    def numeric_features(self) -> tuple[str, ...]:
        return tuple(self.allowlist["numeric_features"])

    @property
    def categorical_features(self) -> tuple[str, ...]:
        return tuple(self.allowlist["categorical_features"])

    @property
    def ordered_features(self) -> tuple[str, ...]:
        return (*self.numeric_features, *self.categorical_features)

    @property
    def denied_features(self) -> frozenset[str]:
        return frozenset(self.denylist["numeric_features"] + self.denylist["categorical_features"])

    @property
    def runner_columns(self) -> tuple[str, ...]:
        return tuple(self.input_contract["canonical_runner_universe_schema"]["ordered_columns"])

    @property
    def lineage_columns(self) -> tuple[str, ...]:
        return tuple(self.input_contract["feature_lineage_schema"]["ordered_columns"])


def load_and_verify_contract(config_path: str | Path = DEFAULT_CONFIG) -> ContractBundle:
    path = Path(config_path)
    if not path.is_absolute():
        path = _repo_path(str(path))
    _require(
        path.resolve() == _repo_path(DEFAULT_CONFIG),
        "only the approved canonical config path may be opened",
    )
    _require(sha256_file(path) == CONFIG_BYTE_SHA256, "canonical config byte hash mismatch")
    config = strict_json_load(path)
    _require(config.get("experiment_id") == EXPERIMENT_ID, "wrong experiment id")
    _require(config.get("variant") == VARIANT, "wrong or unapproved variant")
    _require(config.get("research_only") is True, "research_only must be true")
    _validate_safety_flags(config)
    for key in (
        "production_change_allowed",
        "champion_change_allowed",
        "notification_allowed",
        "order_allowed",
        "network_allowed",
        "netkeiba_allowed",
    ):
        _require(config.get(key) is False, f"{key} must remain false")

    proposal_ref = config["proposal_scope"]
    proposal_path = _repo_path(str(proposal_ref["path"]))
    _require(sha256_file(proposal_path) == PROPOSAL_BYTE_SHA256, "proposal byte hash mismatch")
    proposal = strict_json_load(proposal_path)
    _require(canonical_digest(proposal) == PROPOSAL_DIGEST, "proposal digest mismatch")
    _require(proposal_ref["digest"] == PROPOSAL_DIGEST, "config proposal digest mismatch")
    _require(proposal_ref["byte_sha256"] == PROPOSAL_BYTE_SHA256, "config proposal byte hash mismatch")

    frozen = config["frozen_contracts"]
    input_path = _verify_hash_ref(frozen["input_canonicalization_contract"])
    _verify_hash_ref(frozen["fold_manifest"])
    _verify_hash_ref(frozen["synthetic_fixture_plan"])
    env_path = _verify_hash_ref(frozen["dependency_environment_manifest"])
    # These two files predate EXP-034's LF-only contract.  Git stores their
    # approved bytes with LF, while core.autocrlf may expose CRLF in a Windows
    # checkout.  Only these inherited tracked references receive this narrow
    # clean-filter equivalent; every EXP-034 dependency is verified byte-exact.
    allow_path = _verify_hash_ref(
        frozen["exp033_feature_allowlist"],
        allow_crlf_checkout_normalization=True,
    )
    deny_path = _verify_hash_ref(
        frozen["exp033_feature_denylist"],
        allow_crlf_checkout_normalization=True,
    )
    fold_path = _verify_hash_ref(
        frozen["exp033_fold_manifest"],
        allow_crlf_checkout_normalization=True,
    )
    input_contract = strict_json_load(input_path)
    allowlist = strict_json_load(allow_path)
    denylist = strict_json_load(deny_path)
    exp033_fold = strict_json_load(fold_path)
    environment = strict_json_load(env_path)

    _require(input_contract.get("experiment_id") == EXPERIMENT_ID, "input contract experiment mismatch")
    _require(input_contract.get("variant") == VARIANT, "input contract variant mismatch")
    _require(len(allowlist.get("numeric_features", [])) == 77, "EXP-033 numeric count changed")
    _require(len(allowlist.get("categorical_features", [])) == 11, "EXP-033 categorical count changed")
    _require(len(set(allowlist["numeric_features"] + allowlist["categorical_features"])) == 88, "EXP-033 allowlist is not unique")
    _require(allowlist.get("total_count") == 88, "EXP-033 total feature count changed")
    _require(len(denylist.get("numeric_features", [])) == 279, "EXP-033 denied numeric count changed")
    _require(len(denylist.get("categorical_features", [])) == 7, "EXP-033 denied categorical count changed")
    _require(denylist.get("total_count") == 286, "EXP-033 denied total feature count changed")
    _require(
        not set(allowlist["numeric_features"] + allowlist["categorical_features"])
        & set(denylist["numeric_features"] + denylist["categorical_features"]),
        "EXP-033 allowlist and denylist overlap",
    )
    _require(environment.get("supported_python") == ["3.11", "3.12"], "supported Python contract changed")
    _require(environment.get("runtime", {}).get("dependencies") == [], "Prepare implementation must remain stdlib-only")
    _validate_safety_flags(environment)

    runtime = config["runtime_authorization"]
    _require(runtime.get("real_data_cli_status") == "fail_closed", "real-data CLI must remain fail-closed")
    _require(runtime.get("versioned_real_data_run_contract_required") is True, "versioned real-data contract must be required")
    _require(runtime.get("run_scope_generation_during_prepare") is False, "Prepare must not generate a run scope")
    _require(runtime.get("real_source_path_resolution_during_prepare") is False, "Prepare must not resolve real source paths")
    _require(runtime.get("real_data_rows_during_prepare") == 0, "Prepare real-data row budget must be zero")
    _require(
        config["exp033_handoff"].get("exp033_denylist_sha256")
        == frozen["exp033_feature_denylist"]["sha256"],
        "EXP-033 denylist handoff hash mismatch",
    )
    _require(
        config["manifest_composition_contract"].get("artifact_name_by_kind") == ARTIFACT_NAME_BY_KIND,
        "manifest artifact-role contract changed",
    )
    _require(
        config["runner_contract"].get("feature_safe_dependency_fields") == list(RUNNER_FEATURE_SAFE_FIELDS),
        "feature-safe runner dependency contract changed",
    )
    lineage_contract = config["source_lineage_contract"]
    _require(
        set(lineage_contract.get("accepted_canonical_source_payloads", []))
        == {"runner_feature_safe", "declared_card_event", "completed_result_event"},
        "accepted canonical source payload contract changed",
    )
    _require(
        set(lineage_contract.get("direct_source_evidence_required_fields", []))
        == DEPENDENCY_EVIDENCE_FIELDS,
        "direct source evidence contract changed",
    )
    return ContractBundle(path, config, input_contract, allowlist, denylist, exp033_fold, environment)


def source_payload_hash(payload: Mapping[str, Any]) -> str:
    cleaned = {
        key: value
        for key, value in payload.items()
        if key not in {"source_content_sha256", "source_content_hash", "row_payload_sha256"}
    }
    return canonical_digest(cleaned)


def row_payload_hash(row: Mapping[str, Any]) -> str:
    return canonical_digest({key: value for key, value in row.items() if key != "row_payload_sha256"})


def validate_source_envelope(
    row: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    *,
    later_prediction_event_time: str | None = None,
) -> None:
    _require_identity(row)
    record_kind = str(row.get("record_kind", ""))
    _require(record_kind in {"declared_card", "completed_result"}, "unsupported record_kind")
    prediction = _parse_timestamp(row.get("prediction_event_time"), "prediction_event_time")
    source = _parse_timestamp(row.get("source_event_time"), "source_event_time")
    received = _parse_timestamp(row.get("received_at"), "received_at")
    available = _parse_timestamp(row.get("available_as_of"), "available_as_of")
    _require_nonblank_string(row.get("source_version"), "source_version")
    _require(row.get("missing_reason") in MISSING_REASONS, "unknown missing_reason")
    source_hash = _require_hash(row.get("source_content_sha256"), "source_content_sha256")
    _require(source_hash == source_payload_hash(source_payload), "source payload hash mismatch")
    _require(source <= received <= available, "source_event_time <= received_at <= available_as_of is required")
    if record_kind == "declared_card":
        _require(available < prediction, "declared card must be available before prediction")
    else:
        _require(prediction < source, "completed result must occur after its race prediction time")
        if later_prediction_event_time is not None:
            later = _parse_timestamp(later_prediction_event_time, "later_prediction_event_time")
            _require(available < later, "completed result is not available before later prediction")


def _validate_card_core_values(row: Mapping[str, Any]) -> None:
    for field in ("年齢", "距離"):
        _require(type(row[field]) is int and row[field] > 0, f"{field} must be a positive integer")
    _require(
        type(row["斤量"]) in {int, float}
        and not isinstance(row["斤量"], bool)
        and math.isfinite(float(row["斤量"]))
        and float(row["斤量"]) > 0.0,
        "斤量 must be positive finite numeric",
    )
    for field in ("場所", "性別", "騎手コード", "調教師コード", "芝・ダ", "クラス名", "トラックコード"):
        _require_nonblank_string(row[field], f"declared-card field {field}")


def _canonical_result_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    result_status = _require_nonblank_string(row["result_status"], "result_status")
    _require(
        result_status == "finished" or result_status in WHOLE_RACE_INELIGIBLE_STATUSES,
        f"unknown result_status: {result_status}",
    )
    rank = row["official_finish_rank_raw"]
    finish = row["確定着順"]
    _require(rank is None or (type(rank) is int and rank > 0), "official_finish_rank_raw must be positive int or null")
    _require(finish is None or (type(finish) is int and finish > 0), "確定着順 must be positive int or null")
    _require(rank == finish, "確定着順 and official_finish_rank_raw differ")
    if result_status == "finished":
        _require(type(rank) is int and rank > 0, "finished result requires an official finish rank")
    for field in ("1角", "2角", "4角"):
        value = row[field]
        _require(value is None or (type(value) is int and value > 0), f"{field} must be positive int or null")
    return {
        "starter_status": "starter",
        "status_available_pre_cutoff": False,
        "official_result_status": result_status,
        "official_finish_rank_raw": rank,
    }


def canonical_event_source_payload(
    row: Mapping[str, Any],
    source_path: str,
    bundle: ContractBundle,
) -> dict[str, Any]:
    record_kind = row.get("record_kind")
    contract = bundle.input_contract["event_release_schema"]
    envelope_columns = tuple(bundle.input_contract["source_envelope_schema"]["ordered_columns"])
    canonical_envelope = {
        field: row[field]
        for field in envelope_columns
        if field != "source_content_sha256"
    }
    if record_kind == "declared_card":
        columns = tuple(contract["historical_and_target_card_core_columns"])
        payload: dict[str, Any] = {
            "canonical_event_fields": {field: row[field] for field in columns},
            "canonical_source_envelope": canonical_envelope,
            "source_path": source_path,
            "source_raw_columns": sorted(columns),
        }
    elif record_kind == "completed_result":
        columns = tuple(contract["historical_result_columns"])
        payload = {
            "canonical_event_fields": {field: row[field] for field in columns},
            "canonical_result_fields": _canonical_result_fields(row),
            "canonical_source_envelope": canonical_envelope,
            "source_path": source_path,
            "source_raw_columns": sorted(columns),
        }
    else:
        raise ContractError(f"unsupported event record_kind: {record_kind}")
    return payload


def validate_event_release(
    rows: Iterable[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    source_payloads: Mapping[tuple[str, str, str], Mapping[str, Any]],
    target_partition: bool,
) -> tuple[dict[str, Any], ...]:
    materialized = [dict(row) for row in rows]
    _require(materialized, "event release must not be empty")
    contract = bundle.input_contract["event_release_schema"]
    envelope_columns = set(bundle.input_contract["source_envelope_schema"]["ordered_columns"])
    card_columns = set(contract["historical_and_target_card_core_columns"])
    result_columns = set(contract["historical_result_columns"])
    seen: set[tuple[str, str, str]] = set()
    cards: set[tuple[str, str]] = set()
    results: set[tuple[str, str]] = set()
    for row in materialized:
        record_kind = str(row.get("record_kind", ""))
        race_id, horse_id = _require_identity(row)
        key = (record_kind, race_id, horse_id)
        _require(key not in seen, f"duplicate event row: {key}")
        seen.add(key)
        _require(envelope_columns.issubset(row), f"source envelope fields missing: {key}")
        _require(key in source_payloads, f"source payload missing: {key}")
        expected_payload_fields = {
            "canonical_event_fields",
            "canonical_source_envelope",
            "source_path",
            "source_raw_columns",
        }
        if record_kind == "completed_result":
            expected_payload_fields.add("canonical_result_fields")
        _require(set(source_payloads[key]) == expected_payload_fields, f"event source payload differs from the default-deny schema: {key}")
        validate_source_envelope(row, source_payloads[key])
        if record_kind == "declared_card":
            _require(set(row) == envelope_columns | card_columns, f"declared-card fields differ from the frozen default-deny schema: {key}")
            _validate_card_core_values(row)
            expected_projection = {field: row[field] for field in card_columns}
            cards.add((race_id, horse_id))
        else:
            _require(not target_partition, "target partition must contain zero result rows")
            _require(set(row) == envelope_columns | result_columns, f"completed-result fields differ from the frozen default-deny schema: {key}")
            result_projection = _canonical_result_fields(row)
            _require(source_payloads[key].get("canonical_result_fields") == result_projection, f"completed-result eligibility projection mismatch: {key}")
            expected_projection = {field: row[field] for field in result_columns}
            results.add((race_id, horse_id))
        _require(
            source_payloads[key]["source_raw_columns"] == sorted(expected_projection),
            f"event raw-column projection mismatch: {key}",
        )
        _validate_source_path(source_payloads[key]["source_path"], bundle, "event source_path")
        _require(
            source_payloads[key].get("canonical_event_fields") == expected_projection,
            f"event source projection mismatch: {key}",
        )
        expected_envelope = {
            field: row[field]
            for field in envelope_columns
            if field != "source_content_sha256"
        }
        _require(
            source_payloads[key].get("canonical_source_envelope") == expected_envelope,
            f"event source envelope projection mismatch: {key}",
        )
    _require(not results or results.issubset(cards), "completed result lacks its declared-card identity")
    return tuple(sorted(materialized, key=lambda row: (row["prediction_event_time"], row["race_id"], row["horse_id"], row["record_kind"])))


REQUESTED_RUNNER_DTO_FIELDS = (
    "release_id",
    "release_version",
    "parent_release_id",
    "race_id",
    "horse_id",
    "horse_name",
    "race_name",
    "event_date",
    "post_time",
    "jockey_id",
    "trainer_id",
    "age",
    "sex",
    "assigned_weight",
    "frame_no",
    "horse_no",
    "draw_status",
    "entry_stage",
    "runner_status",
    "source_event_time",
    "received_at",
    "available_as_of",
    "source_version",
    "source_content_hash",
    "missing_reason",
)

RUNNER_SOURCE_BOUND_FIELDS = (
    "race_id",
    "horse_id",
    "horse_name",
    "race_name",
    "draw_status",
    "entry_stage",
    "runner_status",
    "frame_number",
    "horse_number",
    "jockey_id",
    "jockey_name",
    "carried_weight",
    "trainer_id",
    "trainer_name",
    "active_for_feature_materialization",
)
RUNNER_FEATURE_SAFE_FIELDS = (
    "race_id",
    "horse_id",
    "runner_status",
    "jockey_id",
    "carried_weight",
    "trainer_id",
    "active_for_feature_materialization",
)


def canonical_runner_source_payload(row: Mapping[str, Any], source_path: str) -> dict[str, Any]:
    return {
        "canonical_source_envelope": {
            field: row[field]
            for field in (
                "race_id",
                "horse_id",
                "as_of",
                "source_event_time",
                "received_at",
                "available_as_of",
                "source_version",
                "missing_reason",
            )
        },
        "canonical_runner_fields": {
            field: row[field] for field in RUNNER_SOURCE_BOUND_FIELDS
        },
        "feature_safe_runner_fields": {
            field: row[field] for field in RUNNER_FEATURE_SAFE_FIELDS
        },
        "feature_safe_source_raw_columns": sorted(RUNNER_FEATURE_SAFE_FIELDS),
        "source_path": source_path,
        "source_raw_columns": sorted(RUNNER_SOURCE_BOUND_FIELDS),
    }


def validate_runner_source_binding(row: Mapping[str, Any], source_payload: Mapping[str, Any]) -> None:
    _require(
        set(source_payload)
        == {
            "canonical_runner_fields",
            "canonical_source_envelope",
            "feature_safe_runner_fields",
            "feature_safe_source_raw_columns",
            "source_path",
            "source_raw_columns",
        },
        "runner source payload differs from the default-deny schema",
    )
    projection = source_payload.get("canonical_runner_fields")
    _require(isinstance(projection, dict), "runner source payload lacks canonical_runner_fields")
    _require(set(projection) == set(RUNNER_SOURCE_BOUND_FIELDS), "runner source projection fields mismatch")
    for field in RUNNER_SOURCE_BOUND_FIELDS:
        _require(projection[field] == row[field], f"runner source projection mismatch: {field}")
    expected_envelope = {
        field: row[field]
        for field in (
            "race_id",
            "horse_id",
            "as_of",
            "source_event_time",
            "received_at",
            "available_as_of",
            "source_version",
            "missing_reason",
        )
    }
    _require(
        source_payload.get("canonical_source_envelope") == expected_envelope,
        "runner source envelope projection mismatch",
    )
    _require(
        source_payload["source_raw_columns"] == sorted(RUNNER_SOURCE_BOUND_FIELDS),
        "runner source raw-column projection is incomplete",
    )
    safe_projection = source_payload.get("feature_safe_runner_fields")
    _require(isinstance(safe_projection, dict), "runner source payload lacks feature_safe_runner_fields")
    _require(set(safe_projection) == set(RUNNER_FEATURE_SAFE_FIELDS), "feature-safe runner projection fields mismatch")
    for field in RUNNER_FEATURE_SAFE_FIELDS:
        _require(safe_projection[field] == row[field], f"feature-safe runner projection mismatch: {field}")
    _require(
        source_payload["feature_safe_source_raw_columns"] == sorted(RUNNER_FEATURE_SAFE_FIELDS),
        "feature-safe runner raw-column projection is incomplete",
    )


def runner_feature_dependency_payload(source_payload: Mapping[str, Any]) -> dict[str, Any]:
    safe_envelope_fields = (
        "race_id",
        "horse_id",
        "as_of",
        "source_event_time",
        "received_at",
        "available_as_of",
        "source_version",
    )
    return {
        "canonical_source_envelope": {
            field: copy.deepcopy(source_payload["canonical_source_envelope"][field])
            for field in safe_envelope_fields
        },
        "raw_values": copy.deepcopy(source_payload["feature_safe_runner_fields"]),
        "source_path": source_payload["source_path"],
        "source_raw_columns": list(source_payload["feature_safe_source_raw_columns"]),
    }


def runner_feature_dependency_hash(source_payload: Mapping[str, Any]) -> str:
    return source_payload_hash(runner_feature_dependency_payload(source_payload))


def requested_runner_dto_to_approved_fragments(
    dto: Mapping[str, Any],
    source_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = [field for field in REQUESTED_RUNNER_DTO_FIELDS if field not in dto]
    _require(not missing, f"requested runner DTO fields missing: {missing}")
    race_id, horse_id = _require_identity(dto)
    prediction = _parse_timestamp(dto["post_time"], "post_time")
    post_time_text = str(dto["post_time"]).replace("Z", "+00:00")
    post_time_local = datetime.fromisoformat(post_time_text)
    _require(date.fromisoformat(str(dto["event_date"])) == post_time_local.date(), "event_date and post_time date differ")
    parent = dto["parent_release_id"]
    if parent is not None:
        _require_hash(parent, "parent_release_id/parent_manifest_digest")
    envelope = {
        "record_kind": "declared_card",
        "race_id": race_id,
        "horse_id": horse_id,
        "prediction_event_time": utc_text(prediction),
        "source_event_time": dto["source_event_time"],
        "received_at": dto["received_at"],
        "available_as_of": dto["available_as_of"],
        "source_version": dto["source_version"],
        "source_content_sha256": dto["source_content_hash"],
        "missing_reason": dto["missing_reason"],
    }
    validate_source_envelope(envelope, source_payload)
    runner = {
        "release_family_id": dto["release_id"],
        "release_version": dto["release_version"],
        "parent_manifest_digest": parent,
        "as_of": utc_text(prediction),
        "race_id": race_id,
        "horse_id": horse_id,
        "horse_name": dto.get("horse_name"),
        "race_name": dto["race_name"],
        "draw_status": dto["draw_status"],
        "entry_stage": dto["entry_stage"],
        "runner_status": dto["runner_status"],
        "frame_number": dto["frame_no"],
        "horse_number": dto["horse_no"],
        "jockey_id": dto["jockey_id"],
        "jockey_name": dto.get("jockey_name"),
        "carried_weight": dto["assigned_weight"],
        "trainer_id": dto["trainer_id"],
        "trainer_name": dto.get("trainer_name"),
        "active_for_feature_materialization": dto["runner_status"] != "scratched",
        "change_reason": dto.get("change_reason", "initial_declaration"),
        "source_event_time": dto["source_event_time"],
        "received_at": dto["received_at"],
        "available_as_of": dto["available_as_of"],
        "source_version": dto["source_version"],
        "source_content_sha256": dto["source_content_hash"],
        "missing_reason": dto["missing_reason"],
        "row_payload_sha256": "",
    }
    runner["row_payload_sha256"] = row_payload_hash(runner)
    validate_runner_source_binding(runner, source_payload)
    declared_card_fragment = {
        **envelope,
        "event_date": dto["event_date"],
        "年齢": dto["age"],
        "性別": dto["sex"],
        "斤量": dto["assigned_weight"],
        "fragment_only": True,
        "missing_required_card_fields": ["距離", "場所", "騎手コード", "調教師コード", "芝・ダ", "クラス名", "トラックコード"],
    }
    return runner, declared_card_fragment


def seal_runner_row(row: Mapping[str, Any], source_payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(dict(row))
    validate_runner_source_binding(sealed, source_payload)
    sealed["source_content_sha256"] = source_payload_hash(source_payload)
    sealed["row_payload_sha256"] = ""
    sealed["row_payload_sha256"] = row_payload_hash(sealed)
    return sealed


def _validate_runner_times(row: Mapping[str, Any]) -> None:
    source = _parse_timestamp(row.get("source_event_time"), "source_event_time")
    received = _parse_timestamp(row.get("received_at"), "received_at")
    available = _parse_timestamp(row.get("available_as_of"), "available_as_of")
    as_of = _parse_timestamp(row.get("as_of"), "as_of")
    _require(source <= received <= available < as_of, "runner source times must precede as_of")


def validate_runner_universe(
    rows: Iterable[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    source_payloads: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    materialized = [dict(row) for row in rows]
    _require(materialized, "runner universe must not be empty")
    expected_columns = set(bundle.runner_columns)
    identities: set[tuple[str, str]] = set()
    release_meta: set[tuple[Any, Any, Any, Any]] = set()
    published_numbers: dict[str, set[int]] = {}
    rows_by_race: dict[str, list[dict[str, Any]]] = {}
    for row in materialized:
        _require(tuple(row) == bundle.runner_columns, "runner row names/order do not match the frozen canonical schema")
        identity = _require_identity(row)
        _require(identity not in identities, f"duplicate runner identity: {identity}")
        identities.add(identity)
        _require_nonblank_string(row["release_family_id"], "release_family_id")
        _require(type(row["release_version"]) is int and row["release_version"] >= 1, "release_version must be a positive integer")
        if row["release_version"] == 1:
            _require(row["parent_manifest_digest"] is None, "v1 parent manifest must be null")
        else:
            _require_hash(row["parent_manifest_digest"], "parent_manifest_digest")
        _validate_runner_times(row)
        _require_nonblank_string(row["source_version"], "source_version")
        _require(row["missing_reason"] in MISSING_REASONS, "unknown missing_reason")
        for field in (
            "horse_name",
            "race_name",
            "jockey_id",
            "jockey_name",
            "trainer_id",
            "trainer_name",
            "change_reason",
        ):
            _require_nonblank_string(row[field], f"runner core field {field}")
        _require(
            type(row["carried_weight"]) in {int, float}
            and not isinstance(row["carried_weight"], bool)
            and math.isfinite(float(row["carried_weight"]))
            and float(row["carried_weight"]) > 0.0,
            "carried_weight must be positive finite numeric",
        )
        _require_hash(row["source_content_sha256"], "source_content_sha256")
        _require(row_payload_hash(row) == row["row_payload_sha256"], "runner row payload hash mismatch")
        if source_payloads is not None:
            _require(identity in source_payloads, f"source payload missing for {identity}")
            _require(source_payload_hash(source_payloads[identity]) == row["source_content_sha256"], "runner source payload hash mismatch")
            validate_runner_source_binding(row, source_payloads[identity])
            _validate_source_path(source_payloads[identity]["source_path"], bundle, "runner source_path")
            _validate_raw_columns(
                source_payloads[identity]["feature_safe_source_raw_columns"],
                bundle,
                "feature-safe runner source_raw_columns",
            )
        _require(row["draw_status"] in DRAW_STATUSES, "unknown draw_status")
        _require(row["entry_stage"] in ENTRY_STAGES, "unknown entry_stage")
        _require(row["runner_status"] in RUNNER_STATUSES, "unknown runner_status")
        _require(
            (row["draw_status"] == "scratched") is (row["runner_status"] == "scratched"),
            "draw_status and runner_status scratch state differ",
        )
        frame, horse = row["frame_number"], row["horse_number"]
        if row["draw_status"] == "scheduled_pending_draw":
            _require(frame is None and horse is None, "pending draw numbers must remain null")
            _require(row["entry_stage"] == "declared_without_draw", "pending draw entry stage mismatch")
            _require(row["missing_reason"] == "not_declared_by_source", "pending draw missing reason mismatch")
        elif row["draw_status"] == "confirmed":
            _require(type(frame) is int and 1 <= frame <= 8, "confirmed frame number must be in 1..8")
            _require(type(horse) is int and horse > 0, "confirmed horse number must be a positive integer")
            _require(row["entry_stage"] == "declared_with_draw", "confirmed draw entry stage mismatch")
            _require(row["missing_reason"] == "not_applicable", "confirmed draw must not claim a missing source value")
            used = published_numbers.setdefault(identity[0], set())
            _require(horse not in used, f"duplicate horse number in race {identity[0]}")
            used.add(horse)
        else:
            _require(
                (frame is None and horse is None)
                or (type(frame) is int and 1 <= frame <= 8 and type(horse) is int and horse > 0),
                "scratched draw numbers must be null or within the official domains",
            )
            expected_reason = "not_declared_by_source" if frame is None else "not_applicable"
            _require(row["missing_reason"] == expected_reason, "scratched draw missing reason mismatch")
            if horse is not None:
                used = published_numbers.setdefault(identity[0], set())
                _require(horse not in used, f"duplicate horse number in race {identity[0]}")
                used.add(horse)
        active = row["active_for_feature_materialization"]
        _require(type(active) is bool, "active_for_feature_materialization must be boolean")
        _require(active is (row["runner_status"] != "scratched"), "scratch activity flag mismatch")
        release_meta.add((row["release_family_id"], row["release_version"], row["parent_manifest_digest"], row["as_of"]))
        rows_by_race.setdefault(identity[0], []).append(row)
    _require(len(release_meta) == 1, "runner universe mixes release identities")
    for race_id, race_rows in rows_by_race.items():
        field_size = len(race_rows)
        _require(
            all(row["horse_number"] is None or row["horse_number"] <= field_size for row in race_rows),
            f"horse number exceeds declared field size: {race_id}",
        )
        pending = [row for row in race_rows if row["draw_status"] == "scheduled_pending_draw"]
        if pending:
            _require(
                all(row["draw_status"] != "confirmed" for row in race_rows),
                f"race mixes pending and confirmed draw states: {race_id}",
            )
            _require(
                all(row["frame_number"] is None and row["horse_number"] is None for row in race_rows),
                f"pre-draw race contains a published number: {race_id}",
            )
        else:
            numbers = [row["horse_number"] for row in race_rows]
            _require(
                all(type(number) is int for number in numbers)
                and sorted(numbers) == list(range(1, field_size + 1)),
                f"draw-confirmed race horse numbers are not an exact 1..n permutation: {race_id}",
            )
    return tuple(sorted(materialized, key=lambda row: (str(row["race_id"]), str(row["horse_id"]))))


def validate_predraw_baseline_runner_universe(
    rows: Iterable[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    source_payloads: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    validated = validate_runner_universe(rows, bundle, source_payloads=source_payloads)
    identity_contract = bundle.input_contract["canonical_runner_universe_schema"]["identity_contract"]
    draw_contract = bundle.input_contract["canonical_runner_universe_schema"]["draw_contract"]
    counts: dict[str, int] = {}
    for row in validated:
        counts[row["race_id"]] = counts.get(row["race_id"], 0) + 1
    _require(len(counts) == identity_contract["baseline_race_count"], "predraw race count differs from the frozen baseline")
    _require(len(validated) == identity_contract["baseline_runner_count"], "predraw runner count differs from the frozen baseline")
    _require(
        sorted(counts.values()) == sorted(identity_contract["baseline_per_race_counts"]),
        "predraw per-race runner counts differ from the frozen baseline",
    )
    pending = sum(row["draw_status"] == "scheduled_pending_draw" for row in validated)
    confirmed = sum(row["draw_status"] == "confirmed" for row in validated)
    scratched = sum(row["draw_status"] == "scratched" for row in validated)
    _require(pending == draw_contract["scheduled_pending_draw_baseline_count"], "predraw pending-draw count mismatch")
    _require(confirmed == draw_contract["confirmed_baseline_count"], "predraw confirmed-draw count mismatch")
    _require(scratched == 0, "baseline predraw release unexpectedly contains a scratch")
    return validated


def certified_runner_release_bytes(
    rows: Iterable[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    source_payloads: Mapping[tuple[str, str], Mapping[str, Any]],
) -> bytes:
    validated = validate_runner_universe(rows, bundle, source_payloads=source_payloads)
    return canonical_jsonl_bytes(validated, sort_key=IDENTITY_COLUMNS)


def runner_release_bytes(rows: Iterable[Mapping[str, Any]], bundle: ContractBundle) -> bytes:
    validated = validate_runner_universe(rows, bundle)
    return canonical_jsonl_bytes(validated, sort_key=IDENTITY_COLUMNS)


def runner_release_digest(rows: Iterable[Mapping[str, Any]], bundle: ContractBundle) -> str:
    return sha256_bytes(runner_release_bytes(rows, bundle))


def join_runner_updates(
    base_rows: Iterable[Mapping[str, Any]],
    updates: Iterable[Mapping[str, Any]],
    *,
    join_keys: Sequence[str] = IDENTITY_COLUMNS,
) -> dict[tuple[str, str], dict[str, Any]]:
    _require(tuple(join_keys) == IDENTITY_COLUMNS, "runner updates may join only on race_id x horse_id; horse-name join is forbidden")
    base = {_require_identity(row): dict(row) for row in base_rows}
    for update in updates:
        identity = _require_identity(update)
        _require(identity in base, f"unexpected runner identity in update: {identity}")
        base[identity].update(dict(update))
    return base


def _runner_diff_audited_fields(bundle: ContractBundle) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                "release_version",
                "parent_manifest_digest",
                "as_of",
                *bundle.input_contract["saturday_postdraw_update_contract"]["versioned_fields"],
                "runner_status",
                "source_event_time",
                "received_at",
                "available_as_of",
                "source_version",
                "source_content_sha256",
                "missing_reason",
            )
        )
    )


def build_child_runner_release(
    parent_rows: Iterable[Mapping[str, Any]],
    updates: Iterable[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    child_version: int,
    child_as_of: str,
    parent_manifest: Mapping[str, Any],
    parent_source_payloads: Mapping[tuple[str, str], Mapping[str, Any]],
    source_payloads: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    parent = validate_runner_universe(parent_rows, bundle, source_payloads=parent_source_payloads)
    parent_bytes = certified_runner_release_bytes(parent, bundle, source_payloads=parent_source_payloads)
    parent_artifacts_input = parent_manifest.get("artifacts")
    _require(isinstance(parent_artifacts_input, list) and len(parent_artifacts_input) == 1, "parent runner manifest must bind one artifact")
    parent_artifact_path = _require_nonblank_string(parent_artifacts_input[0].get("path"), "parent runner artifact path")
    validate_manifest(parent_manifest, bundle=bundle, artifact_bytes_by_path={parent_artifact_path: parent_bytes})
    _require(parent_manifest.get("manifest_kind") == "runner_universe_manifest", "wrong parent manifest kind")
    _require(parent_manifest.get("release_version") == parent[0]["release_version"], "parent manifest version mismatch")
    _require(parent_manifest.get("release_family_id") == parent[0]["release_family_id"], "parent manifest family mismatch")
    _require(parent_manifest.get("row_count") == len(parent), "parent manifest row count mismatch")
    _require(parent_manifest.get("runner_count") == len(parent), "parent manifest runner count mismatch")
    _require(
        parent_manifest.get("race_count") == len({row["race_id"] for row in parent}),
        "parent manifest race count mismatch",
    )
    _require(
        type(parent_manifest.get("source_time_completeness")) is float
        and parent_manifest["source_time_completeness"] == 1.0,
        "parent source-time certification is incomplete",
    )
    _require(
        type(parent_manifest.get("source_hash_completeness")) is float
        and parent_manifest["source_hash_completeness"] == 1.0,
        "parent source-hash certification is incomplete",
    )
    _require(parent_manifest.get("certification_status") == "certified", "parent runner release is not certified")
    parent_artifacts = {
        item["name"]: item for item in parent_manifest.get("artifacts", [])
    }
    _require("runner_universe_release" in parent_artifacts, "parent runner artifact reference missing")
    _require(
        parent_artifacts["runner_universe_release"]["sha256"] == sha256_bytes(parent_bytes),
        "parent manifest does not bind the runner release",
    )
    parent_digest = str(parent_manifest["content_hash"])
    parent_version = int(parent[0]["release_version"])
    _require(type(child_version) is int and child_version == parent_version + 1, "child release version must increment by one")
    child_time = _parse_timestamp(child_as_of, "child_as_of")
    parent_as_of = _parse_timestamp(parent[0]["as_of"], "parent as_of")
    _require(
        child_time == parent_as_of,
        "runner as_of is the immutable prediction cutoff across child versions",
    )
    updates_by_id: dict[tuple[str, str], dict[str, Any]] = {}
    allowed = {
        "race_id",
        "horse_id",
        "draw_status",
        "entry_stage",
        "runner_status",
        "frame_number",
        "horse_number",
        "jockey_id",
        "jockey_name",
        "carried_weight",
        "active_for_feature_materialization",
        "change_reason",
        "source_event_time",
        "received_at",
        "available_as_of",
        "source_version",
        "missing_reason",
    }
    for raw in updates:
        update = dict(raw)
        identity = _require_identity(update)
        _require(identity not in updates_by_id, f"duplicate child update: {identity}")
        _require(set(update).issubset(allowed), f"unapproved child fields: {sorted(set(update) - allowed)}")
        _require(identity in {_require_identity(row) for row in parent}, f"child adds an unexpected identity: {identity}")
        for field in ("source_event_time", "received_at", "available_as_of", "source_version", "missing_reason"):
            _require(field in update, f"child update lacks source evidence field {field}")
        _require(identity in source_payloads, f"child update lacks source payload: {identity}")
        original = next(row for row in parent if _require_identity(row) == identity)
        old_draw = original["draw_status"]
        new_draw = update.get("draw_status", old_draw)
        _require(old_draw != "scratched", "a scratched runner cannot be reactivated or revised in place")
        _require(
            (old_draw == "scheduled_pending_draw" and new_draw in {"confirmed", "scratched"})
            or (old_draw == "confirmed" and new_draw in {"confirmed", "scratched"})
            or new_draw == old_draw,
            f"unapproved draw transition: {old_draw} -> {new_draw}",
        )
        if new_draw == old_draw:
            _require(
                any(field in update for field in ("jockey_id", "jockey_name", "carried_weight")),
                "same-stage child update must be a jockey or carried-weight correction",
            )
        if old_draw == "confirmed" and new_draw == "scratched":
            _require(update.get("frame_number", original["frame_number"]) == original["frame_number"], "a postdraw scratch must retain its official frame number")
            _require(update.get("horse_number", original["horse_number"]) == original["horse_number"], "a postdraw scratch must retain its official horse number")
        previous_available = _parse_timestamp(original["available_as_of"], "parent available_as_of")
        new_available = _parse_timestamp(update["available_as_of"], "child available_as_of")
        _require(previous_available < new_available < child_time, "child evidence must be newer and available before child as_of")
        updates_by_id[identity] = update
    _require(updates_by_id, "child release requires at least one evidenced update")

    child_rows: list[dict[str, Any]] = []
    diffs: list[dict[str, Any]] = []
    audited_fields = _runner_diff_audited_fields(bundle)
    for original in parent:
        identity = _require_identity(original)
        child = copy.deepcopy(original)
        child["release_version"] = child_version
        child["parent_manifest_digest"] = parent_digest
        child["as_of"] = utc_text(child_time)
        update = updates_by_id.get(identity)
        old_payload_hash = original["row_payload_sha256"]
        if update is not None:
            for field, value in update.items():
                if field not in IDENTITY_COLUMNS:
                    child[field] = value
            validate_runner_source_binding(child, source_payloads[identity])
            child["source_content_sha256"] = source_payload_hash(source_payloads[identity])
            child["row_payload_sha256"] = ""
            child["row_payload_sha256"] = row_payload_hash(child)
        else:
            child["row_payload_sha256"] = ""
            child["row_payload_sha256"] = row_payload_hash(child)
        changes = []
        for field in audited_fields:
            if original.get(field) != child.get(field):
                changes.append({"field": field, "old": original.get(field), "new": child.get(field)})
        _require(changes, f"child row lacks an auditable version delta: {identity}")
        diffs.append(
            {
                "race_id": identity[0],
                "horse_id": identity[1],
                "old_row_payload_sha256": old_payload_hash,
                "new_row_payload_sha256": child["row_payload_sha256"],
                "changes": changes,
                "reason": child["change_reason"] if update is not None else "release_version_successor",
                "source_content_sha256": child["source_content_sha256"],
            }
        )
        child_rows.append(child)
    _require(runner_release_bytes(parent, bundle) == parent_bytes, "parent release was mutated")
    child_source_payloads = dict(parent_source_payloads)
    child_source_payloads.update(source_payloads)
    validated_child = validate_runner_universe(child_rows, bundle, source_payloads=child_source_payloads)
    _require({_require_identity(row) for row in validated_child} == {_require_identity(row) for row in parent}, "child identity set changed")
    return validated_child, tuple(sorted(diffs, key=lambda row: (row["race_id"], row["horse_id"])))


FEATURE_CELL_REQUIRED_FIELDS = {
    "release_id",
    "release_version",
    "race_id",
    "horse_id",
    "prediction_event_time",
    "feature_name",
    "feature_value",
    "dtype",
    "source_path",
    "source_event_time",
    "received_at",
    "available_as_of",
    "source_version",
    "source_content_hash",
    "transform_name",
    "transform_version",
    "transform_code_hash",
    "dependency_feature_names",
    "dependency_content_hashes",
    "missing_reason",
    "asof_safe",
    "lineage_status",
}


def _feature_schema_digest(bundle: ContractBundle) -> str:
    return canonical_digest(
        {
            "numeric_features": list(bundle.numeric_features),
            "categorical_features": list(bundle.categorical_features),
            "numeric_dtype": "float64",
            "categorical_dtype": "string",
        }
    )


def _forbidden_dependency_name(value: str, bundle: ContractBundle | None = None) -> bool:
    if bundle is not None and value in bundle.denied_features:
        return True
    lowered = value.lower()
    forbidden = (
        "race_winner_prior_strength",
        "race_top3_prior_strength_mean",
        "odds",
        "popularity",
        "market",
        "payout",
        "roi",
        "buy",
        "stake",
        "candidate",
        "value_score",
        "target_result",
        "current_result",
        "future",
        "future_opponent",
        "frame_number",
        "horse_number",
        "frame_no",
        "horse_no",
        "draw_",
        "枠番",
        "馬番",
        "オッズ",
        "人気",
        "払戻",
    )
    direct_result_fields = {
        "確定着順",
        "着差",
        "走破時計",
        "走破タイム",
        "上り",
        "上がり",
        "1角",
        "2角",
        "3角",
        "4角",
        "result_status",
        "official_result_status",
        "official_finish_rank_raw",
        "starter_status",
        "status_available_pre_cutoff",
    }
    return any(token in lowered for token in forbidden) or value.strip() in direct_result_fields


def _validate_raw_columns(
    columns: Any,
    bundle: ContractBundle,
    label: str,
    *,
    historical_completed_result: bool = False,
) -> tuple[str, ...]:
    _require(isinstance(columns, list) and columns, f"{label} must be a nonempty list")
    _require(len(columns) == len(set(columns)), f"{label} contains duplicates")
    approved_non_features = set(bundle.config["source_lineage_contract"]["approved_non_feature_raw_columns"])
    historical_result_columns = set(bundle.config["source_lineage_contract"]["historical_completed_result_raw_columns"])
    approved = set(bundle.ordered_features) | approved_non_features
    if historical_completed_result:
        approved |= historical_result_columns
    normalized: list[str] = []
    for item in columns:
        name = _require_nonblank_string(item, f"{label} item")
        _require(name in approved, f"raw column is outside the default-deny source allowlist: {name}")
        historical_result_field = historical_completed_result and name in historical_result_columns
        _require(historical_result_field or name not in bundle.denied_features, f"EXP-033 denied raw column: {name}")
        _require(historical_result_field or not _forbidden_dependency_name(name, bundle), f"forbidden raw column: {name}")
        normalized.append(name)
    return tuple(normalized)


def _validate_raw_scalar(value: Any, label: str) -> None:
    _require(value is None or type(value) in {bool, int, float, str}, f"{label} must be a JSON scalar")
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} must be finite")


def _validate_source_path(value: Any, bundle: ContractBundle, label: str = "source_path") -> str:
    path = _require_nonblank_string(value, label)
    _require(not Path(path).is_absolute() and ".." not in Path(path).parts, f"{label} is unsafe")
    _require(not _forbidden_dependency_name(path, bundle), f"forbidden {label}")
    return path


DEPENDENCY_EVIDENCE_FIELDS = {
    "canonical_source_payload",
    "content_sha256",
    "source_path",
    "source_event_time",
    "received_at",
    "available_as_of",
    "source_version",
}


def _validate_dependency_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_hash: str,
    prediction_event_time: datetime,
    bundle: ContractBundle,
    consuming_race_id: str,
    consuming_horse_id: str,
    allowed_historical_horse_ids: set[str],
) -> dict[str, Any]:
    _require(
        consuming_horse_id in allowed_historical_horse_ids,
        "consuming horse is outside its approved dependency context",
    )
    item = dict(evidence)
    _require(set(item) == DEPENDENCY_EVIDENCE_FIELDS, "dependency evidence schema mismatch")
    content_hash = _require_hash(item["content_sha256"], "dependency content_sha256")
    _require(content_hash == expected_hash, "dependency evidence key/hash mismatch")
    payload = item["canonical_source_payload"]
    _require(isinstance(payload, dict), "dependency canonical_source_payload must be an object")
    payload_fields = set(payload)
    runner_safe_fields = {"canonical_source_envelope", "raw_values", "source_path", "source_raw_columns"}
    declared_event_fields = {
        "canonical_event_fields",
        "canonical_source_envelope",
        "source_path",
        "source_raw_columns",
    }
    completed_event_fields = {*declared_event_fields, "canonical_result_fields"}
    _require(
        payload_fields in (runner_safe_fields, declared_event_fields, completed_event_fields),
        "dependency canonical_source_payload differs from every approved default-deny schema",
    )
    _require(source_payload_hash(payload) == content_hash, "dependency payload hash mismatch")
    source_path = _validate_source_path(item["source_path"], bundle, "dependency source_path")
    _require(payload["source_path"] == source_path, "dependency source_path differs from its evidence envelope")
    record_kind: str
    source_race_prediction: datetime
    if payload_fields == runner_safe_fields:
        envelope = payload["canonical_source_envelope"]
        expected_envelope_fields = {
            "race_id",
            "horse_id",
            "as_of",
            "source_event_time",
            "received_at",
            "available_as_of",
            "source_version",
        }
        _require(isinstance(envelope, dict) and set(envelope) == expected_envelope_fields, "runner dependency source envelope schema mismatch")
        source_race_id, source_horse_id = _require_identity(envelope)
        _require(source_race_id == consuming_race_id, "runner dependency belongs to another race")
        _require(source_horse_id in allowed_historical_horse_ids, "runner dependency belongs to an unrelated horse")
        raw_columns = _validate_raw_columns(
            payload["source_raw_columns"], bundle, "runner dependency source_raw_columns"
        )
        raw_values = payload["raw_values"]
        _require(isinstance(raw_values, dict), "runner dependency raw_values must be an object")
        _require(set(raw_values) == set(raw_columns), "runner dependency raw_values/raw-column projection mismatch")
        for raw_name, raw_value in raw_values.items():
            _validate_raw_scalar(raw_value, f"runner dependency raw value {raw_name}")
        _require(
            envelope["source_event_time"] == item["source_event_time"]
            and envelope["received_at"] == item["received_at"]
            and envelope["available_as_of"] == item["available_as_of"]
            and envelope["source_version"] == item["source_version"],
            "runner dependency source envelope differs from its evidence",
        )
        _require(
            utc_text(_parse_timestamp(envelope["as_of"], "runner dependency as_of"))
            == utc_text(prediction_event_time),
            "runner dependency cutoff differs from consuming prediction",
        )
        source_race_prediction = prediction_event_time
        record_kind = "runner_feature_safe"
    else:
        envelope = payload["canonical_source_envelope"]
        envelope_fields = set(bundle.input_contract["source_envelope_schema"]["ordered_columns"]) - {
            "source_content_sha256"
        }
        _require(
            isinstance(envelope, dict) and set(envelope) == envelope_fields,
            "event dependency source envelope schema mismatch",
        )
        source_race_id, source_horse_id = _require_identity(envelope)
        record_kind = _require_nonblank_string(envelope["record_kind"], "event dependency record_kind")
        expected_record_kind = "completed_result" if payload_fields == completed_event_fields else "declared_card"
        _require(record_kind == expected_record_kind, "event dependency record kind/payload schema mismatch")
        _require(source_horse_id in allowed_historical_horse_ids, "event dependency belongs to an unrelated horse")
        _require(
            envelope["source_event_time"] == item["source_event_time"]
            and envelope["received_at"] == item["received_at"]
            and envelope["available_as_of"] == item["available_as_of"]
            and envelope["source_version"] == item["source_version"],
            "event dependency source envelope differs from its evidence",
        )
        source_race_prediction = _parse_timestamp(
            envelope["prediction_event_time"], "event dependency prediction_event_time"
        )
        event_projection = payload["canonical_event_fields"]
        _require(isinstance(event_projection, dict), "event dependency projection must be an object")
        if record_kind == "declared_card":
            expected_columns = tuple(
                bundle.input_contract["event_release_schema"]["historical_and_target_card_core_columns"]
            )
            _require(set(event_projection) == set(expected_columns), "declared-card dependency projection schema mismatch")
            _validate_card_core_values(event_projection)
            if source_race_id == consuming_race_id:
                _require(
                    source_race_prediction == prediction_event_time,
                    "current declared-card dependency cutoff differs from consuming prediction",
                )
            else:
                _require(
                    source_race_prediction < prediction_event_time,
                    "historical declared-card dependency is not strictly prior",
                )
        else:
            expected_columns = tuple(bundle.input_contract["event_release_schema"]["historical_result_columns"])
            _require(set(event_projection) == set(expected_columns), "completed-result dependency projection schema mismatch")
            _require(source_race_id != consuming_race_id, "current-race result dependency is forbidden")
            canonical_result = _canonical_result_fields(event_projection)
            _require(payload["canonical_result_fields"] == canonical_result, "completed-result eligibility projection mismatch")
        _require(
            payload["source_raw_columns"] == sorted(expected_columns),
            "event dependency raw-column projection mismatch",
        )
        _validate_raw_columns(
            payload["source_raw_columns"],
            bundle,
            "event dependency source_raw_columns",
            historical_completed_result=record_kind == "completed_result",
        )
    source = _parse_timestamp(item["source_event_time"], "dependency source_event_time")
    received = _parse_timestamp(item["received_at"], "dependency received_at")
    available = _parse_timestamp(item["available_as_of"], "dependency available_as_of")
    _require(source <= received <= available < prediction_event_time, "dependency is not strictly available before prediction")
    if record_kind == "completed_result":
        _require(source_race_prediction < source, "historical result source time must follow its own prediction cutoff")
    source_version = _require_nonblank_string(item["source_version"], "dependency source_version")
    return {
        "content_sha256": content_hash,
        "source_path": source_path,
        "source_event_time": source,
        "received_at": received,
        "available_as_of": available,
        "source_version": source_version,
        "record_kind": record_kind,
        "race_id": source_race_id,
        "horse_id": source_horse_id,
        "prediction_event_time": source_race_prediction,
    }


def build_feature_release(
    cells: Iterable[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    source_payloads: Mapping[tuple[str, str, str], Mapping[str, Any]],
    runner_rows: Iterable[Mapping[str, Any]],
    runner_source_payloads: Mapping[tuple[str, str], Mapping[str, Any]],
    dependency_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    materialized = [dict(cell) for cell in cells]
    _require(materialized, "feature cells must not be empty")
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    lineage_rows: list[dict[str, Any]] = []
    allowed = set(bundle.ordered_features)
    numeric = set(bundle.numeric_features)
    categorical = set(bundle.categorical_features)
    seen_triples: set[tuple[str, str, str]] = set()
    release_meta: set[tuple[Any, Any]] = set()
    runner_universe = validate_runner_universe(
        runner_rows,
        bundle,
        source_payloads=runner_source_payloads,
    )
    runner_identities = {_require_identity(row) for row in runner_universe if row["active_for_feature_materialization"]}
    runner_by_identity = {_require_identity(row): row for row in runner_universe}
    runner_evidence_by_hash: dict[str, dict[str, Any]] = {}
    runner_identity_by_hash: dict[str, tuple[str, str]] = {}
    effective_runner_horse_ids_by_race: dict[str, set[str]] = {}
    for row in runner_universe:
        identity = _require_identity(row)
        payload = runner_source_payloads[identity]
        feature_safe_payload = runner_feature_dependency_payload(payload)
        feature_safe_hash = runner_feature_dependency_hash(payload)
        if row["active_for_feature_materialization"]:
            effective_runner_horse_ids_by_race.setdefault(row["race_id"], set()).add(row["horse_id"])
        runner_evidence_by_hash[feature_safe_hash] = {
            "canonical_source_payload": feature_safe_payload,
            "content_sha256": feature_safe_hash,
            "source_path": payload["source_path"],
            "source_event_time": row["source_event_time"],
            "received_at": row["received_at"],
            "available_as_of": row["available_as_of"],
            "source_version": row["source_version"],
        }
        runner_identity_by_hash[feature_safe_hash] = identity
    supplied_dependency_evidence = dict(dependency_evidence or {})
    _require(
        all(isinstance(key, str) and HASH_RE.fullmatch(key) for key in supplied_dependency_evidence),
        "dependency evidence keys must be lowercase SHA-256",
    )
    for cell in materialized:
        _require(set(cell) == FEATURE_CELL_REQUIRED_FIELDS, "feature cell schema mismatch")
        race_id, horse_id = _require_identity(cell)
        _require((race_id, horse_id) in runner_identities, "feature cell identity is not an effective certified runner")
        feature = str(cell["feature_name"])
        triple = (race_id, horse_id, feature)
        _require(triple not in seen_triples, f"duplicate feature lineage: {triple}")
        seen_triples.add(triple)
        _require(feature in allowed, f"feature is outside the EXP-033 allowlist: {feature}")
        expected_dtype = "float64" if feature in numeric else "string"
        _require(cell["dtype"] == expected_dtype, f"feature dtype mismatch: {feature}")
        value = cell["feature_value"]
        if feature in numeric:
            _require(type(value) in {int, float} and not isinstance(value, bool), f"numeric feature value required: {feature}")
            _require(math.isfinite(float(value)), f"non-finite feature value: {feature}")
            value = float(value)
        else:
            _require(isinstance(value, str), f"string feature value required: {feature}")
            _require(not _is_blank(value), f"blank categorical feature value: {feature}")
        _require(cell["asof_safe"] is True, f"as-of proof failed: {triple}")
        _require(cell["lineage_status"] == LINEAGE_SAFE_VERDICT, f"lineage is not certified: {triple}")
        _require(cell["missing_reason"] in MISSING_REASONS, "unknown feature missing_reason")
        source_path = _validate_source_path(cell["source_path"], bundle)
        _require_nonblank_string(cell["source_version"], "source_version")
        transform_name = _require_nonblank_string(cell["transform_name"], "transform_name")
        transform_version = _require_nonblank_string(cell["transform_version"], "transform_version")
        _require(not _forbidden_dependency_name(transform_name, bundle), "forbidden transform name")
        _require(not _forbidden_dependency_name(transform_version, bundle), "forbidden transform version")
        source_hash = _require_hash(cell["source_content_hash"], "source_content_hash")
        transform_hash = _require_hash(cell["transform_code_hash"], "transform_code_hash")
        approved_transform = bundle.config["feature_contract"]["approved_transform"]
        _require(
            transform_name == approved_transform["name"]
            and transform_version == approved_transform["version"]
            and transform_hash == approved_transform["code_sha256"],
            f"feature transform is outside the frozen default-deny registry: {triple}",
        )
        _require(isinstance(cell["dependency_feature_names"], list), "dependency_feature_names must be a list")
        _require(isinstance(cell["dependency_content_hashes"], list), "dependency_content_hashes must be a list")
        dependency_names = list(cell["dependency_feature_names"])
        dependency_hashes = sorted(set(cell["dependency_content_hashes"]))
        _require(len(dependency_names) == len(set(dependency_names)), "duplicate dependency feature name")
        _require(len(cell["dependency_content_hashes"]) == len(dependency_hashes), "duplicate dependency content hash")
        _require(all(isinstance(name, str) and not _forbidden_dependency_name(name, bundle) for name in dependency_names), "forbidden feature dependency")
        _require(dependency_hashes and all(HASH_RE.fullmatch(str(item)) for item in dependency_hashes), "dependency hashes are required")
        _require(source_hash in dependency_hashes, f"feature dependency set omits its source payload: {triple}")
        source = _parse_timestamp(cell["source_event_time"], "source_event_time")
        received = _parse_timestamp(cell["received_at"], "received_at")
        available = _parse_timestamp(cell["available_as_of"], "available_as_of")
        prediction = _parse_timestamp(cell["prediction_event_time"], "prediction_event_time")
        _require(
            utc_text(prediction) == utc_text(_parse_timestamp(runner_by_identity[(race_id, horse_id)]["as_of"], "runner as_of")),
            f"feature prediction cutoff differs from the certified runner cutoff: {triple}",
        )
        _require(source <= received <= available < prediction, f"feature source is not available pre-race: {triple}")
        _require(triple in source_payloads, f"direct feature source evidence missing: {triple}")
        direct_evidence = dict(source_payloads[triple])
        _require(
            set(direct_evidence) == DEPENDENCY_EVIDENCE_FIELDS,
            f"direct feature source evidence schema mismatch: {triple}",
        )
        _require(
            direct_evidence["content_sha256"] == source_hash,
            f"direct feature source evidence hash mismatch: {triple}",
        )
        dependency_records: list[dict[str, Any]] = []
        allowed_historical_horse_ids = (
            effective_runner_horse_ids_by_race[race_id]
            if feature in RACE_AGGREGATE_FEATURES
            else {horse_id}
        )
        for dependency_hash in dependency_hashes:
            if dependency_hash == source_hash:
                raw_evidence = direct_evidence
            elif dependency_hash in runner_evidence_by_hash:
                dependency_identity = runner_identity_by_hash[dependency_hash]
                if feature in RACE_AGGREGATE_FEATURES:
                    _require(
                        dependency_identity[0] == race_id,
                        f"race aggregate cites a runner from another race: {triple}/{dependency_identity}",
                    )
                else:
                    _require(
                        dependency_identity == (race_id, horse_id),
                        f"horse feature cites another current runner: {triple}/{dependency_identity}",
                    )
                raw_evidence = runner_evidence_by_hash[dependency_hash]
            else:
                _require(
                    dependency_hash in supplied_dependency_evidence,
                    f"feature dependency lacks hash-bound source/time evidence: {dependency_hash}",
                )
                raw_evidence = supplied_dependency_evidence[dependency_hash]
            dependency_records.append(
                _validate_dependency_evidence(
                    raw_evidence,
                    expected_hash=dependency_hash,
                    prediction_event_time=prediction,
                    bundle=bundle,
                    consuming_race_id=race_id,
                    consuming_horse_id=horse_id,
                    allowed_historical_horse_ids=allowed_historical_horse_ids,
                )
            )
        direct_record = next(
            (item for item in dependency_records if item["content_sha256"] == source_hash),
            None,
        )
        _require(direct_record is not None, f"direct feature source was not validated: {triple}")
        _require(direct_record["source_path"] == source_path, f"feature source path/evidence mismatch: {triple}")
        _require(direct_record["source_version"] == cell["source_version"], f"feature source version/evidence mismatch: {triple}")
        _require(direct_record["source_event_time"] == source, f"feature source_event_time/evidence mismatch: {triple}")
        _require(direct_record["received_at"] == received, f"feature received_at/evidence mismatch: {triple}")
        _require(direct_record["available_as_of"] == available, f"feature available_as_of/evidence mismatch: {triple}")
        if cell["missing_reason"] != "not_applicable":
            if feature in categorical:
                _require(value == "__MISSING__", f"categorical missing value policy mismatch: {feature}")
            elif feature == "prev_corner4_position_rate":
                _require(value == 0.5, "missing prior corner4 must use the frozen neutral 0.5")
            else:
                _require(value == 0.0, f"numeric missing value policy mismatch: {feature}")
        if feature in RACE_AGGREGATE_FEATURES:
            expected_runner_identities = {
                identity
                for identity in runner_identities
                if identity[0] == race_id
            }
            observed_current_identities = {
                (item["race_id"], item["horse_id"])
                for item in dependency_records
                if item["race_id"] == race_id
            }
            _require(expected_runner_identities, f"race aggregate runner dependencies missing: {race_id}")
            _require(
                expected_runner_identities.issubset(observed_current_identities),
                f"race aggregate lineage omits an effective runner: {triple}",
            )
        grouped.setdefault((race_id, horse_id), {})[feature] = {**cell, "feature_value": value}
        release_meta.add((cell["release_id"], cell["release_version"]))
        source_content_hashes = sorted({item["content_sha256"] for item in dependency_records})
        value_binding_hash = feature_value_binding_hash(
            race_id=race_id,
            horse_id=horse_id,
            feature_name=feature,
            feature_value=value,
            feature_dtype=expected_dtype,
            prediction_event_time=utc_text(prediction),
            transformation_name=cell["transform_name"],
            transformation_version=cell["transform_version"],
            transformation_code_sha256=transform_hash,
            source_content_sha256_set=source_content_hashes,
        )
        lineage = {
            "race_id": race_id,
            "horse_id": horse_id,
            "feature_name": feature,
            "feature_dtype": expected_dtype,
            "source_paths": sorted({item["source_path"] for item in dependency_records}),
            "source_versions": sorted({item["source_version"] for item in dependency_records}),
            "source_content_sha256_set": source_content_hashes,
            "dependency_feature_names": sorted(set(dependency_names)),
            "dependency_content_sha256_set": sorted({*dependency_hashes, value_binding_hash}),
            "transformation_name": cell["transform_name"],
            "transformation_version": cell["transform_version"],
            "transformation_code_sha256": transform_hash,
            "max_source_event_time": utc_text(max(item["source_event_time"] for item in dependency_records)),
            "max_received_at": utc_text(max(item["received_at"] for item in dependency_records)),
            "max_available_as_of": utc_text(max(item["available_as_of"] for item in dependency_records)),
            "prediction_event_time": utc_text(prediction),
            "missing_reason": cell["missing_reason"],
            "lineage_verdict": LINEAGE_SAFE_VERDICT,
            "row_payload_sha256": "",
        }
        lineage["row_payload_sha256"] = row_payload_hash(lineage)
        _require(tuple(lineage) == bundle.lineage_columns, "lineage row order differs from frozen schema")
        lineage_rows.append(lineage)
    _require(len(release_meta) == 1, "feature release mixes release identities")
    _require(set(grouped) == runner_identities, "feature release identity set differs from the effective runner universe")
    expected_features = set(bundle.ordered_features)
    for identity, features in grouped.items():
        _require(set(features) == expected_features, f"feature release is incomplete for {identity}")
        metadata = {
            (
                item["prediction_event_time"],
                item["release_id"],
                item["release_version"],
            )
            for item in features.values()
        }
        _require(len(metadata) == 1, f"feature cells mix prediction/release metadata for {identity}")
    race_prediction_times: dict[str, set[str]] = {}
    for (race_id, _horse_id), features in grouped.items():
        first_feature = features[bundle.ordered_features[0]]
        race_prediction_times.setdefault(race_id, set()).add(first_feature["prediction_event_time"])
    _require(all(len(values) == 1 for values in race_prediction_times.values()), "race feature rows mix prediction event times")
    lineage_rows.sort(key=lambda row: (row["race_id"], row["horse_id"], row["feature_name"]))
    lineage_bytes = canonical_jsonl_bytes(lineage_rows, sort_key=("race_id", "horse_id", "feature_name"))
    lineage_hash = sha256_bytes(lineage_bytes)
    schema_hash = _feature_schema_digest(bundle)
    wide_rows: list[dict[str, Any]] = []
    for identity in sorted(grouped):
        feature_map = grouped[identity]
        first = feature_map[bundle.ordered_features[0]]
        row: dict[str, Any] = {
            "race_id": identity[0],
            "horse_id": identity[1],
            "prediction_event_time": first["prediction_event_time"],
            "release_family_id": first["release_id"],
            "release_version": first["release_version"],
            "feature_schema_sha256": schema_hash,
            "lineage_manifest_sha256": lineage_hash,
        }
        for feature in bundle.ordered_features:
            row[feature] = feature_map[feature]["feature_value"]
        wide_rows.append(row)
    validate_wide_feature_release(wide_rows, bundle, lineage_manifest_sha256=lineage_hash)
    wide_bytes = canonical_jsonl_bytes(wide_rows, sort_key=("prediction_event_time", "race_id", "horse_id"))
    return {
        "wide_rows": tuple(wide_rows),
        "lineage_rows": tuple(lineage_rows),
        "feature_schema_sha256": schema_hash,
        "lineage_manifest_sha256": lineage_hash,
        "feature_release_sha256": sha256_bytes(wide_bytes),
        "wide_bytes": wide_bytes,
        "lineage_bytes": lineage_bytes,
    }


def validate_wide_feature_release(
    rows: Iterable[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    lineage_manifest_sha256: str,
) -> None:
    expected_prefix = tuple(bundle.input_contract["feature_input_release_schema"]["identity_and_audit_prefix"])
    expected_columns = (*expected_prefix, *bundle.ordered_features)
    identities: set[tuple[str, str]] = set()
    for row in rows:
        _require(tuple(row) == expected_columns, "wide feature row names/order mismatch")
        identity = _require_identity(row)
        _require(identity not in identities, f"duplicate wide feature identity: {identity}")
        identities.add(identity)
        _require(row["lineage_manifest_sha256"] == lineage_manifest_sha256, "lineage manifest hash mismatch")
        _require(row["feature_schema_sha256"] == _feature_schema_digest(bundle), "feature schema hash mismatch")
        for feature in bundle.numeric_features:
            value = row[feature]
            _require(type(value) is float and math.isfinite(value), f"wide numeric dtype/value mismatch: {feature}")
        for feature in bundle.categorical_features:
            _require_nonblank_string(row[feature], f"wide categorical value {feature}")


def _derive_race_ineligibility_reason(
    effective_results: Sequence[Mapping[str, Any] | None],
    *,
    target_labels_absent: bool = False,
    unbound_nonstarter: bool = False,
) -> str | None:
    """Derive the one frozen whole-race reason without inspecting model outcomes."""

    if target_labels_absent:
        return "target_labels_absent"
    if not effective_results:
        return "no_effective_starters"
    if len(effective_results) < 2:
        return "fewer_than_two_effective_starters"
    if unbound_nonstarter:
        return "nonstarter_status_not_bound_before_prediction"

    statuses: list[str] = []
    ranks: list[Any] = []
    for result in effective_results:
        if result is None:
            statuses.append("result_missing")
            ranks.append(None)
            continue
        status = result.get("official_result_status", result.get("result_status"))
        if status is None:
            statuses.append("result_missing")
            ranks.append(None)
            continue
        _require(
            status == "finished" or status in WHOLE_RACE_INELIGIBLE_STATUSES,
            f"unknown official result status in label policy: {status}",
        )
        statuses.append(str(status))
        ranks.append(result.get("official_finish_rank_raw"))
    for reason in RESULT_REASON_PRIORITY:
        if reason in statuses:
            return reason
    if any(type(rank) is not int or rank < 1 for rank in ranks):
        return "result_missing"
    if sorted(ranks) != list(range(1, len(effective_results) + 1)):
        return "dead_heat_or_rank_gap"
    return None


def classify_label_eligibility(
    declared_rows: Iterable[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    target_labels_absent: bool = False,
    declared_source_payloads: Mapping[tuple[str, str], Mapping[str, Any]],
    result_source_payloads: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    declared = [dict(row) for row in declared_rows]
    _require(declared, "declared rows are required")
    declared_by_race: dict[str, list[dict[str, Any]]] = {}
    race_prediction_cutoffs: dict[str, str] = {}
    identities: set[tuple[str, str]] = set()
    for row in declared:
        _require(
            set(row) in (DECLARED_LABEL_BASE_FIELDS, DECLARED_LABEL_BASE_FIELDS | DECLARED_LABEL_PRE_CUTOFF_FIELDS),
            "declared label row differs from the default-deny schema",
        )
        identity = _require_identity(row)
        _require(identity not in identities, f"duplicate declared identity: {identity}")
        _require(identity in declared_source_payloads, f"declared source payload missing: {identity}")
        prediction_text = utc_text(_parse_timestamp(row["prediction_event_time"], "declared prediction_event_time"))
        prior_cutoff = race_prediction_cutoffs.setdefault(identity[0], prediction_text)
        _require(prior_cutoff == prediction_text, f"declared race batch mixes prediction cutoffs: {identity[0]}")
        declared_envelope = {
            "record_kind": "declared_card",
            "race_id": identity[0],
            "horse_id": identity[1],
            "prediction_event_time": row.get("prediction_event_time", row.get("as_of")),
            "source_event_time": row.get("source_event_time"),
            "received_at": row.get("received_at"),
            "available_as_of": row.get("available_as_of"),
            "source_version": row.get("source_version"),
            "source_content_sha256": row.get("source_content_sha256"),
            "missing_reason": row.get("missing_reason"),
        }
        validate_source_envelope(declared_envelope, declared_source_payloads[identity])
        _require(
            set(declared_source_payloads[identity])
            == {"canonical_declared_status", "source_path", "source_raw_columns"},
            f"declared status payload differs from the default-deny schema: {identity}",
        )
        expected_declared_status = {"runner_status": row.get("runner_status", "declared_active")}
        for optional_field in ("precutoff_starter_status", "status_available_pre_cutoff"):
            if optional_field in row:
                expected_declared_status[optional_field] = row[optional_field]
        _require(
            declared_source_payloads[identity].get("canonical_declared_status")
            == expected_declared_status,
            f"declared status source projection mismatch: {identity}",
        )
        _validate_source_path(declared_source_payloads[identity]["source_path"], bundle, "declared source_path")
        _require(
            declared_source_payloads[identity]["source_raw_columns"] == sorted(expected_declared_status),
            f"declared status raw-column projection mismatch: {identity}",
        )
        runner_status = _require_nonblank_string(row["runner_status"], "declared runner_status")
        _require(runner_status in RUNNER_STATUSES, f"unknown declared runner status: {runner_status}")
        if set(row) == DECLARED_LABEL_BASE_FIELDS | DECLARED_LABEL_PRE_CUTOFF_FIELDS:
            _require(
                row["precutoff_starter_status"] in {"starter", *PRE_CUTOFF_NONSTARTER_STATUSES},
                "unknown declared precutoff starter status",
            )
            _require(type(row["status_available_pre_cutoff"]) is bool, "status_available_pre_cutoff must be boolean")
        identities.add(identity)
        declared_by_race.setdefault(identity[0], []).append(row)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    result_rows_materialized = [dict(row) for row in result_rows]
    if target_labels_absent:
        _require(not result_rows_materialized, "target label partition must contain zero result rows")
    elif result_rows_materialized:
        _require(result_source_payloads is not None, "result source payload evidence is required")
    for row in result_rows_materialized:
        envelope_columns = set(bundle.input_contract["source_envelope_schema"]["ordered_columns"])
        result_columns = set(bundle.input_contract["event_release_schema"]["historical_result_columns"])
        _require(set(row) == envelope_columns | result_columns, "label result row is not a canonical completed-result event")
        _require(row.get("record_kind") == "completed_result", "label result row must be completed_result")
        identity = _require_identity(row)
        _require(identity in identities, f"result identity is not declared: {identity}")
        _require(identity not in results, f"duplicate result identity: {identity}")
        _require(result_source_payloads is not None and identity in result_source_payloads, f"result payload missing: {identity}")
        validate_source_envelope(row, result_source_payloads[identity])
        _require(
            set(result_source_payloads[identity])
            == {
                "canonical_event_fields",
                "canonical_result_fields",
                "canonical_source_envelope",
                "source_path",
                "source_raw_columns",
            },
            f"result payload differs from the default-deny schema: {identity}",
        )
        expected_event_fields = {field: row[field] for field in result_columns}
        _require(
            result_source_payloads[identity].get("canonical_event_fields") == expected_event_fields,
            f"result event source projection mismatch: {identity}",
        )
        expected_envelope = {
            field: row[field]
            for field in envelope_columns
            if field != "source_content_sha256"
        }
        _require(
            result_source_payloads[identity].get("canonical_source_envelope") == expected_envelope,
            f"result source envelope projection mismatch: {identity}",
        )
        canonical_result = _canonical_result_fields(row)
        _require(
            result_source_payloads[identity].get("canonical_result_fields") == canonical_result,
            f"result source projection mismatch: {identity}",
        )
        _validate_source_path(result_source_payloads[identity]["source_path"], bundle, "result source_path")
        _require(
            result_source_payloads[identity]["source_raw_columns"] == sorted(expected_event_fields),
            f"result raw-column projection mismatch: {identity}",
        )
        result_prediction = utc_text(_parse_timestamp(row["prediction_event_time"], "result prediction_event_time"))
        _require(
            result_prediction == race_prediction_cutoffs[identity[0]],
            f"declared/result prediction cutoff mismatch: {identity}",
        )
        results[identity] = {**row, **canonical_result}

    ledger: list[dict[str, Any]] = []
    for race_id, race_declared in sorted(declared_by_race.items()):
        effective: list[tuple[str, str]] = []
        preexcluded: set[tuple[str, str]] = set()
        unbound_nonstarter = False
        for row in race_declared:
            identity = _require_identity(row)
            runner_status = str(row.get("runner_status", "declared_active"))
            _require(runner_status in RUNNER_STATUSES, f"unknown declared runner status: {runner_status}")
            result = results.get(identity)
            starter_status = str((result or {}).get("starter_status", "starter"))
            _require(starter_status in {"starter", *PRE_CUTOFF_NONSTARTER_STATUSES}, f"unknown starter_status: {starter_status}")
            declared_starter_status = str(row.get("precutoff_starter_status", "starter"))
            _require(
                declared_starter_status in {"starter", *PRE_CUTOFF_NONSTARTER_STATUSES},
                f"unknown declared precutoff starter status: {declared_starter_status}",
            )
            declared_preexcluded = runner_status == "scratched" or declared_starter_status in PRE_CUTOFF_NONSTARTER_STATUSES
            if declared_preexcluded:
                proof = row.get("status_available_pre_cutoff")
                _require(type(proof) is bool and proof is True, "precutoff nonstarter requires exact certified boolean proof")
                if runner_status == "scratched":
                    _require(declared_starter_status in {"starter", "scratched"}, "scratch status evidence contradicts runner ledger")
                _require(result is None, "a certified precutoff nonstarter must not have a completed-result event")
                preexcluded.add(identity)
                continue
            elif starter_status in PRE_CUTOFF_NONSTARTER_STATUSES:
                proof = result.get("status_available_pre_cutoff") if result else None
                _require(type(proof) is bool, "nonstarter cutoff flag must be an exact boolean")
                unbound_nonstarter = True
            effective.append(identity)
        for identity in effective:
            result = results.get(identity)
            if result is not None:
                _require(
                    not (
                        result.get("official_result_status") == "finished"
                        and result.get("starter_status") != "starter"
                    ),
                    "finished result contradicts nonstarter status",
                )
        race_reason = _derive_race_ineligibility_reason(
            [results.get(identity) for identity in effective],
            target_labels_absent=target_labels_absent,
            unbound_nonstarter=unbound_nonstarter,
        )
        race_eligible = race_reason is None and not target_labels_absent
        for row in sorted(race_declared, key=lambda item: str(item["horse_id"])):
            identity = _require_identity(row)
            result = results.get(identity, {})
            excluded = identity in preexcluded
            item_reason = race_reason or ("precutoff_nonstarter_retained" if excluded else "eligible")
            ledger.append(
                {
                    "race_id": identity[0],
                    "horse_id": identity[1],
                    "declared_status": row.get("runner_status", "declared_active"),
                    "starter_status": result.get(
                        "starter_status",
                        row.get("precutoff_starter_status", "not_applicable" if target_labels_absent else "starter"),
                    ),
                    "official_result_status": result.get("official_result_status"),
                    "official_finish_rank_raw": result.get("official_finish_rank_raw"),
                    "row_label_eligible": bool(race_eligible and not excluded),
                    "race_label_eligible": bool(race_eligible),
                    "ineligibility_reason": item_reason,
                    "source_content_sha256": result.get("source_content_sha256", row.get("source_content_sha256")),
                    "row_payload_sha256": "",
                }
            )
            ledger[-1]["row_payload_sha256"] = row_payload_hash(ledger[-1])
    _require(len(ledger) == len(declared), "label reconciliation deleted a declared runner")
    return tuple(sorted(ledger, key=lambda row: (row["race_id"], row["horse_id"])))


def _validate_manifest_path(path_value: Any, bundle: ContractBundle, *, output: bool) -> str:
    path = _require_nonblank_string(path_value, "manifest path")
    parsed = Path(path)
    _require(not parsed.is_absolute() and ".." not in parsed.parts, "manifest path is unsafe")
    _require(
        all(
            part.casefold() != "latest" and not part.casefold().startswith("latest.")
            for part in parsed.parts
        ),
        "mutable latest manifest aliases are forbidden",
    )
    _require(not _forbidden_dependency_name(path, bundle), "manifest path contains a forbidden source or sink")
    contract = bundle.config["manifest_path_contract"]
    if output:
        _require(
            path.startswith(contract["synthetic_output_prefix"])
            or path.startswith(contract["canonical_output_prefix"]),
            "manifest output path is outside the approved research roots",
        )
    else:
        _require(
            path.startswith(contract["synthetic_input_prefix"])
            or path.startswith(contract["canonical_output_prefix"])
            or path in set(contract["approved_real_input_paths"]),
            "manifest input path is outside the approved source contract",
        )
    return path


def _observed_artifact_row_count(path: str, raw: bytes) -> int:
    _require(isinstance(raw, bytes), f"artifact byte evidence must be bytes: {path}")
    _require(raw and not raw.startswith(b"\xef\xbb\xbf"), f"artifact must be nonempty UTF-8 without BOM: {path}")
    _require(b"\r" not in raw and raw.endswith(b"\n"), f"artifact must be canonical LF text: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"artifact is not UTF-8: {path}") from exc
    if path.endswith(".jsonl"):
        lines = [line for line in text.splitlines() if line]
        _require(lines, f"JSONL artifact is empty: {path}")
        for ordinal, line in enumerate(lines, start=1):
            strict_json_loads(line, source=f"{path}:{ordinal}")
        return len(lines)
    parsed = strict_json_loads(text, source=path)
    _require(isinstance(parsed, dict), f"JSON artifact must be an object: {path}")
    if "row_count" in parsed:
        _require(type(parsed["row_count"]) is int and parsed["row_count"] >= 0, f"artifact row_count is invalid: {path}")
        return parsed["row_count"]
    return 1


def artifact_schema_digest(kind: str, bundle: ContractBundle) -> str:
    envelope = list(bundle.input_contract["source_envelope_schema"]["ordered_columns"])
    event_contract = bundle.input_contract["event_release_schema"]
    schemas: dict[str, Any] = {
        "runner_universe_manifest": list(bundle.runner_columns),
        "training_source_manifest": {
            "declared_card": envelope + list(event_contract["historical_and_target_card_core_columns"]),
            "completed_result": envelope + list(event_contract["historical_result_columns"]),
        },
        "target_source_manifest": {
            "declared_card": envelope + list(event_contract["historical_and_target_card_core_columns"]),
        },
        "feature_release_manifest": {
            "prefix": list(bundle.input_contract["feature_input_release_schema"]["identity_and_audit_prefix"]),
            "features": list(bundle.ordered_features),
        },
        "lineage_manifest": list(bundle.lineage_columns),
        "label_eligibility_manifest": list(bundle.input_contract["label_eligibility_contract"]["eligibility_output_columns"]),
        "release_diff_manifest": [
            "race_id",
            "horse_id",
            "old_row_payload_sha256",
            "new_row_payload_sha256",
            "changes",
            "reason",
            "source_content_sha256",
        ],
        "environment_manifest": sorted(ENVIRONMENT_ARTIFACT_FIELDS),
        "dependency_lock_manifest": sorted(DEPENDENCY_LOCK_ARTIFACT_FIELDS),
        "canonical_root_manifest": [
            "experiment_id",
            "release_family_id",
            "release_version",
            "parent_manifest_digest",
            "as_of",
            "source_cutoff",
            "dependency_manifest_digests",
            "row_count",
            "race_count",
            "runner_count",
            "formal_buy",
            "send_order",
            "stake",
        ],
    }
    _require(kind in schemas, f"unknown artifact schema kind: {kind}")
    return canonical_digest({"manifest_kind": kind, "schema": schemas[kind]})


def _parse_canonical_jsonl(path: str, raw: bytes) -> list[dict[str, Any]]:
    _observed_artifact_row_count(path, raw)
    _require(path.endswith(".jsonl"), f"role-specific row artifact must use .jsonl: {path}")
    rows: list[dict[str, Any]] = []
    for ordinal, line_bytes in enumerate(raw.splitlines(keepends=True), start=1):
        _require(line_bytes.endswith(b"\n") and b"\r" not in line_bytes, f"noncanonical JSONL line ending: {path}:{ordinal}")
        line = line_bytes[:-1].decode("utf-8")
        value = strict_json_loads(line, source=f"{path}:{ordinal}")
        _require(isinstance(value, dict), f"JSONL row must be an object: {path}:{ordinal}")
        _require(canonical_json_bytes(value) + b"\n" == line_bytes, f"JSONL row is not canonical: {path}:{ordinal}")
        rows.append(value)
    return rows


def _parse_canonical_json_object(path: str, raw: bytes) -> dict[str, Any]:
    _observed_artifact_row_count(path, raw)
    _require(path.endswith(".json"), f"role-specific object artifact must use .json: {path}")
    value = strict_json_loads(raw.decode("utf-8"), source=path)
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    _require(canonical_json_bytes(value) + b"\n" == raw, f"JSON artifact is not canonical: {path}")
    return value


def _validate_structural_event_rows(
    rows: Sequence[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    target_partition: bool,
) -> tuple[dict[str, Any], ...]:
    envelope_columns = set(bundle.input_contract["source_envelope_schema"]["ordered_columns"])
    event_contract = bundle.input_contract["event_release_schema"]
    card_columns = set(event_contract["historical_and_target_card_core_columns"])
    result_columns = set(event_contract["historical_result_columns"])
    seen: set[tuple[str, str, str]] = set()
    cards: set[tuple[str, str]] = set()
    results: set[tuple[str, str]] = set()
    race_cutoffs: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        record_kind = _require_nonblank_string(row.get("record_kind"), "event record_kind")
        race_id, horse_id = _require_identity(row)
        key = (record_kind, race_id, horse_id)
        _require(key not in seen, f"duplicate event row: {key}")
        seen.add(key)
        prediction = _parse_timestamp(row.get("prediction_event_time"), "prediction_event_time")
        source = _parse_timestamp(row.get("source_event_time"), "source_event_time")
        received = _parse_timestamp(row.get("received_at"), "received_at")
        available = _parse_timestamp(row.get("available_as_of"), "available_as_of")
        cutoff_text = utc_text(prediction)
        _require(race_cutoffs.setdefault(race_id, cutoff_text) == cutoff_text, f"event race batch mixes prediction cutoffs: {race_id}")
        _require_nonblank_string(row.get("source_version"), "source_version")
        _require_hash(row.get("source_content_sha256"), "source_content_sha256")
        _require(row.get("missing_reason") in MISSING_REASONS, "unknown event missing_reason")
        _require(source <= received <= available, "event source time order is invalid")
        if record_kind == "declared_card":
            _require(set(row) == envelope_columns | card_columns, f"declared-card artifact schema mismatch: {key}")
            _require(available < prediction, "declared card is not available pre-race")
            _validate_card_core_values(row)
            cards.add((race_id, horse_id))
        elif record_kind == "completed_result":
            _require(not target_partition, "target source artifact contains a result row")
            _require(set(row) == envelope_columns | result_columns, f"completed-result artifact schema mismatch: {key}")
            _require(prediction < source, "completed result does not follow its prediction cutoff")
            _canonical_result_fields(row)
            results.add((race_id, horse_id))
        else:
            raise ContractError(f"unsupported event record_kind: {record_kind}")
        normalized.append(row)
    _require(not results or results.issubset(cards), "completed result lacks a declared-card event")
    if target_partition:
        _require(cards and not results, "target source artifact must contain cards and zero results")
    else:
        _require(cards and results, "training source artifact must contain both cards and completed results")
    return tuple(sorted(normalized, key=lambda row: (row["prediction_event_time"], row["race_id"], row["horse_id"], row["record_kind"])))


def _validate_lineage_artifact_rows(rows: Sequence[Mapping[str, Any]], bundle: ContractBundle) -> tuple[dict[str, Any], ...]:
    seen: set[tuple[str, str, str]] = set()
    by_identity: dict[tuple[str, str], set[str]] = {}
    normalized: list[dict[str, Any]] = []
    for raw_row in rows:
        _require(set(raw_row) == set(bundle.lineage_columns), "lineage artifact row schema mismatch")
        row = {field: raw_row[field] for field in bundle.lineage_columns}
        race_id, horse_id = _require_identity(row)
        feature = _require_nonblank_string(row["feature_name"], "lineage feature_name")
        triple = (race_id, horse_id, feature)
        _require(triple not in seen, f"duplicate lineage identity: {triple}")
        seen.add(triple)
        _require(feature in set(bundle.ordered_features), "lineage feature is outside the allowlist")
        expected_dtype = "float64" if feature in set(bundle.numeric_features) else "string"
        _require(row["feature_dtype"] == expected_dtype, "lineage feature dtype mismatch")
        for field in ("source_paths", "source_versions", "source_content_sha256_set", "dependency_feature_names", "dependency_content_sha256_set"):
            _require(isinstance(row[field], list) and row[field], f"lineage list is empty: {field}")
            _require(row[field] == sorted(set(row[field])), f"lineage list is not sorted unique: {field}")
        for path in row["source_paths"]:
            _validate_source_path(path, bundle, "lineage source_path")
        for version in row["source_versions"]:
            _require_nonblank_string(version, "lineage source_version")
        for digest in (*row["source_content_sha256_set"], *row["dependency_content_sha256_set"]):
            _require_hash(digest, "lineage dependency hash")
        _require(
            set(row["source_content_sha256_set"]).issubset(set(row["dependency_content_sha256_set"])),
            "lineage source hashes are not included in the dependency hash set",
        )
        _require(not any(_forbidden_dependency_name(name, bundle) for name in row["dependency_feature_names"]), "lineage contains a forbidden feature dependency")
        _require_hash(row["transformation_code_sha256"], "lineage transformation hash")
        _require_nonblank_string(row["transformation_name"], "lineage transformation_name")
        _require_nonblank_string(row["transformation_version"], "lineage transformation_version")
        approved_transform = bundle.config["feature_contract"]["approved_transform"]
        _require(
            row["transformation_name"] == approved_transform["name"]
            and row["transformation_version"] == approved_transform["version"]
            and row["transformation_code_sha256"] == approved_transform["code_sha256"],
            "lineage transform is outside the frozen default-deny registry",
        )
        max_source = _parse_timestamp(row["max_source_event_time"], "lineage max_source_event_time")
        max_received = _parse_timestamp(row["max_received_at"], "lineage max_received_at")
        max_available = _parse_timestamp(row["max_available_as_of"], "lineage max_available_as_of")
        prediction = _parse_timestamp(row["prediction_event_time"], "lineage prediction_event_time")
        _require(
            max_source <= max_received <= max_available < prediction,
            "lineage requires source <= received <= available < prediction",
        )
        _require(row["missing_reason"] in MISSING_REASONS, "lineage missing reason is invalid")
        _require(row["lineage_verdict"] == LINEAGE_SAFE_VERDICT, "lineage verdict is not certified")
        _require(row_payload_hash(row) == row["row_payload_sha256"], "lineage row hash mismatch")
        by_identity.setdefault((race_id, horse_id), set()).add(feature)
        normalized.append(row)
    expected = set(bundle.ordered_features)
    _require(by_identity and all(features == expected for features in by_identity.values()), "lineage artifact lacks exact 88-feature coverage")
    return tuple(sorted(normalized, key=lambda row: (row["race_id"], row["horse_id"], row["feature_name"])))


def _validate_label_artifact_rows(rows: Sequence[Mapping[str, Any]], bundle: ContractBundle) -> tuple[dict[str, Any], ...]:
    expected_columns = tuple(bundle.input_contract["label_eligibility_contract"]["eligibility_output_columns"])
    seen: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    by_race: dict[str, list[dict[str, Any]]] = {}
    for raw_row in rows:
        _require(set(raw_row) == set(expected_columns), "label eligibility artifact schema mismatch")
        row = {field: raw_row[field] for field in expected_columns}
        identity = _require_identity(row)
        _require(identity not in seen, f"duplicate label eligibility identity: {identity}")
        seen.add(identity)
        for field in ("row_label_eligible", "race_label_eligible"):
            _require(type(row[field]) is bool, f"label eligibility field must be boolean: {field}")
        declared_status = _require_nonblank_string(row["declared_status"], "declared_status")
        _require(declared_status in RUNNER_STATUSES, "label declared_status is invalid")
        starter_status = _require_nonblank_string(row["starter_status"], "starter_status")
        _require(starter_status in {"starter", *PRE_CUTOFF_NONSTARTER_STATUSES}, "label starter_status is invalid")
        reason = _require_nonblank_string(row["ineligibility_reason"], "ineligibility_reason")
        _require(reason in LABEL_INELIGIBILITY_REASONS, "label ineligibility_reason is invalid")
        if starter_status == "starter":
            _require(declared_status == "declared_active", "effective starter must be declared_active")
        elif declared_status == "scratched":
            _require(starter_status == "scratched", "scratched declared status contradicts starter status")
        _require_hash(row["source_content_sha256"], "label source_content_sha256")
        _require(row_payload_hash(row) == row["row_payload_sha256"], "label eligibility row hash mismatch")
        if row["row_label_eligible"]:
            _require(row["race_label_eligible"], "row cannot be eligible when its race is ineligible")
            _require(starter_status == "starter", "eligible label row must be an effective starter")
            _require(row["official_result_status"] == "finished", "eligible label row must have a finished result")
            _require(type(row["official_finish_rank_raw"]) is int and row["official_finish_rank_raw"] >= 1, "eligible rank is invalid")
            _require(row["ineligibility_reason"] == "eligible", "eligible label row reason mismatch")
        elif row["race_label_eligible"]:
            _require(starter_status in PRE_CUTOFF_NONSTARTER_STATUSES, "race-eligible excluded row is not a certified nonstarter")
            _require(row["official_result_status"] is None and row["official_finish_rank_raw"] is None, "precutoff nonstarter must not have a result")
            _require(row["ineligibility_reason"] == "precutoff_nonstarter_retained", "precutoff nonstarter reason mismatch")
        by_race.setdefault(identity[0], []).append(row)
        normalized.append(row)
    for race_id, race_rows in by_race.items():
        race_values = {row["race_label_eligible"] for row in race_rows}
        _require(len(race_values) == 1, f"label race mixes eligibility verdicts: {race_id}")
        effective = [row for row in race_rows if row["starter_status"] == "starter"]
        expected_reason = _derive_race_ineligibility_reason(effective)
        expected_eligible = expected_reason is None
        _require(next(iter(race_values)) is expected_eligible, f"label race verdict differs from frozen policy: {race_id}")
        for row in race_rows:
            expected_row_reason = expected_reason or (
                "eligible" if row["starter_status"] == "starter" else "precutoff_nonstarter_retained"
            )
            _require(row["ineligibility_reason"] == expected_row_reason, f"label reason differs from frozen policy: {race_id}/{row['horse_id']}")
            _require(
                row["row_label_eligible"] is (expected_eligible and row["starter_status"] == "starter"),
                f"label row verdict differs from frozen policy: {race_id}/{row['horse_id']}",
            )
    return tuple(sorted(normalized, key=lambda row: (row["race_id"], row["horse_id"])))


def _validate_artifact_semantics(
    manifest: Mapping[str, Any],
    raw: bytes,
    bundle: ContractBundle,
) -> dict[str, Any]:
    kind = str(manifest["manifest_kind"])
    path = str(manifest["artifacts"][0]["path"])
    _require(manifest["schema_sha256"] == artifact_schema_digest(kind, bundle), f"artifact schema hash mismatch: {kind}")
    result: dict[str, Any] = {"kind": kind, "identities": set(), "race_ids": set(), "rows": ()}
    if kind in {
        "runner_universe_manifest",
        "training_source_manifest",
        "target_source_manifest",
        "feature_release_manifest",
        "lineage_manifest",
        "label_eligibility_manifest",
        "release_diff_manifest",
    }:
        rows = _parse_canonical_jsonl(path, raw)
        if kind == "runner_universe_manifest":
            _require(all(set(row) == set(bundle.runner_columns) for row in rows), "runner artifact row schema mismatch")
            ordered_rows = [{field: row[field] for field in bundle.runner_columns} for row in rows]
            validated = validate_runner_universe(ordered_rows, bundle)
            for row in validated:
                _require(row["release_family_id"] == manifest["release_family_id"], "runner artifact release family mismatch")
                _require(row["release_version"] == manifest["release_version"], "runner artifact release version mismatch")
                _require(row["parent_manifest_digest"] == manifest["parent_manifest_digest"], "runner artifact parent mismatch")
                _require(utc_text(_parse_timestamp(row["as_of"], "runner as_of")) == utc_text(_parse_timestamp(manifest["as_of"], "manifest as_of")), "runner artifact as_of mismatch")
            canonical = canonical_jsonl_bytes(validated, sort_key=IDENTITY_COLUMNS)
        elif kind in {"training_source_manifest", "target_source_manifest"}:
            validated = _validate_structural_event_rows(rows, bundle, target_partition=kind == "target_source_manifest")
            canonical = canonical_jsonl_bytes(validated, sort_key=("prediction_event_time", "race_id", "horse_id", "record_kind"))
        elif kind == "feature_release_manifest":
            _require(rows, "feature release artifact is empty")
            feature_columns = (
                *tuple(bundle.input_contract["feature_input_release_schema"]["identity_and_audit_prefix"]),
                *bundle.ordered_features,
            )
            _require(all(set(row) == set(feature_columns) for row in rows), "feature artifact row schema mismatch")
            rows = [{field: row[field] for field in feature_columns} for row in rows]
            lineage_hashes = {row.get("lineage_manifest_sha256") for row in rows}
            _require(len(lineage_hashes) == 1, "feature release mixes lineage manifest hashes")
            lineage_hash = _require_hash(next(iter(lineage_hashes)), "feature release lineage_manifest_sha256")
            validate_wide_feature_release(rows, bundle, lineage_manifest_sha256=lineage_hash)
            validated = tuple(rows)
            for row in validated:
                _require(row["release_family_id"] == manifest["release_family_id"], "feature artifact release family mismatch")
                _require(row["release_version"] == manifest["release_version"], "feature artifact release version mismatch")
            canonical = canonical_jsonl_bytes(validated, sort_key=("prediction_event_time", "race_id", "horse_id"))
            result["lineage_artifact_sha256"] = lineage_hash
        elif kind == "lineage_manifest":
            validated = _validate_lineage_artifact_rows(rows, bundle)
            canonical = canonical_jsonl_bytes(validated, sort_key=("race_id", "horse_id", "feature_name"))
        elif kind == "label_eligibility_manifest":
            validated = _validate_label_artifact_rows(rows, bundle)
            observed_label_counts = {
                "eligible": sum(row["row_label_eligible"] is True for row in validated),
                "ineligible": sum(row["row_label_eligible"] is False for row in validated),
            }
            _require(
                observed_label_counts == manifest["label_eligibility_counts"],
                "label eligibility count map differs from artifact bytes",
            )
            canonical = canonical_jsonl_bytes(validated, sort_key=("race_id", "horse_id"))
        else:
            validated_rows: list[dict[str, Any]] = []
            for row in rows:
                diff_columns = (
                        "race_id",
                        "horse_id",
                        "old_row_payload_sha256",
                        "new_row_payload_sha256",
                        "changes",
                        "reason",
                        "source_content_sha256",
                    )
                _require(set(row) == set(diff_columns), "release diff artifact schema mismatch")
                row = {field: row[field] for field in diff_columns}
                _require_identity(row)
                for field in ("old_row_payload_sha256", "new_row_payload_sha256", "source_content_sha256"):
                    _require_hash(row[field], f"release diff {field}")
                _require(isinstance(row["changes"], list) and row["changes"], "release diff changes are required")
                for change in row["changes"]:
                    _require(set(change) == {"field", "old", "new"}, "release diff change schema mismatch")
                    _require_nonblank_string(change["field"], "release diff field")
                _require_nonblank_string(row["reason"], "release diff reason")
                validated_rows.append(row)
            validated = tuple(sorted(validated_rows, key=lambda row: (row["race_id"], row["horse_id"])))
            canonical = canonical_jsonl_bytes(validated, sort_key=("race_id", "horse_id"))
        _require(canonical == raw, f"artifact row ordering or serialization is not canonical: {kind}")
        identities = {
            (str(row["race_id"]), str(row["horse_id"]))
            for row in validated
            if "race_id" in row and "horse_id" in row
        }
        race_ids = {race_id for race_id, _horse_id in identities}
        prediction_times: dict[tuple[str, str], set[str]] = {}
        available_times: list[datetime] = []
        missing_distribution: dict[str, int] = {}
        for row in validated:
            if "prediction_event_time" in row:
                identity = _require_identity(row)
                prediction_text = utc_text(_parse_timestamp(row["prediction_event_time"], "artifact prediction_event_time"))
                prediction_times.setdefault(identity, set()).add(prediction_text)
                _require(
                    _parse_timestamp(row["prediction_event_time"], "artifact prediction_event_time")
                    <= _parse_timestamp(manifest["as_of"], "manifest as_of"),
                    f"artifact prediction cutoff exceeds manifest as_of: {kind}",
                )
            available_field = "max_available_as_of" if "max_available_as_of" in row else "available_as_of" if "available_as_of" in row else None
            if available_field is not None:
                available_times.append(_parse_timestamp(row[available_field], f"artifact {available_field}"))
            if "missing_reason" in row:
                reason = str(row["missing_reason"])
                missing_distribution[reason] = missing_distribution.get(reason, 0) + 1
        if available_times:
            max_available = max(available_times)
            _require(
                max_available == _parse_timestamp(manifest["source_cutoff"], "manifest source_cutoff"),
                f"manifest source_cutoff is not derived from artifact availability: {kind}",
            )
            result["max_available_as_of"] = utc_text(max_available)
        if missing_distribution:
            _require(
                missing_distribution == manifest["missing_reason_distribution"],
                f"manifest missing-reason distribution differs from artifact bytes: {kind}",
            )
        result.update(
            {
                "rows": validated,
                "identities": identities,
                "race_ids": race_ids,
                "prediction_times": prediction_times,
            }
        )
    else:
        value = _parse_canonical_json_object(path, raw)
        if kind == "canonical_root_manifest":
            expected = {
                "experiment_id",
                "release_family_id",
                "release_version",
                "parent_manifest_digest",
                "as_of",
                "source_cutoff",
                "dependency_manifest_digests",
                "row_count",
                "race_count",
                "runner_count",
                "formal_buy",
                "send_order",
                "stake",
            }
            _require(set(value) == expected, "canonical root artifact schema mismatch")
            for field in expected - {"dependency_manifest_digests"}:
                _require(value[field] == manifest[field], f"canonical root artifact field mismatch: {field}")
            _require(value["dependency_manifest_digests"] == manifest["dependency_manifest_digests"], "canonical root artifact dependency map mismatch")
        elif kind == "environment_manifest":
            _require(set(value) == ENVIRONMENT_ARTIFACT_FIELDS, "environment artifact schema mismatch")
            _require(value["schema_version"] == 1, "environment artifact schema version mismatch")
            _require(value["experiment_id"] == EXPERIMENT_ID and value["variant"] == VARIANT, "environment artifact identity mismatch")
            for field in (
                "environment_contract_version",
                "python_implementation",
                "python_version",
                "platform",
                "encoding",
                "line_endings",
                "timezone",
                "locale",
                "pythonhashseed",
            ):
                _require_nonblank_string(value[field], f"environment {field}")
            _require_hash(value["executable_sha256"], "environment executable_sha256")
            _require(value["encoding"] == "UTF-8" and value["line_endings"] == "LF" and value["timezone"] == "UTC", "environment encoding/time contract mismatch")
            _require(value["network_access"] is False, "environment network access must be false")
            _require(value["filesystem_mtime_as_received_at"] is False, "filesystem mtime cannot be receipt evidence")
            _validate_safety_flags(value)
        elif kind == "dependency_lock_manifest":
            _require(set(value) == DEPENDENCY_LOCK_ARTIFACT_FIELDS, "dependency-lock artifact schema mismatch")
            _require(value["schema_version"] == 1, "dependency-lock schema version mismatch")
            _require(value["experiment_id"] == EXPERIMENT_ID and value["variant"] == VARIANT, "dependency-lock artifact identity mismatch")
            _require_nonblank_string(value["lock_version"], "dependency lock_version")
            _require(value["dependency_policy"] == "Python standard library only", "dependency-lock policy mismatch")
            _require(value["packages"] == [], "Prepare dependency lock must remain stdlib-only")
            _require_hash(value["interpreter_sha256"], "dependency interpreter_sha256")
            _require_hash(value["config_sha256"], "dependency config_sha256")
            _validate_safety_flags(value)
        else:
            raise ContractError(f"unsupported object artifact kind: {kind}")
        result["object"] = value
    observed_semantic_rows = (
        len(result["rows"])
        if result["rows"]
        else int(result.get("object", {}).get("row_count", 1))
    )
    _require(observed_semantic_rows == manifest["row_count"], f"semantic artifact row count mismatch: {kind}")
    if result["race_ids"]:
        _require(len(result["race_ids"]) == manifest["race_count"], f"semantic artifact race count mismatch: {kind}")
        _require(len(result["identities"]) == manifest["runner_count"], f"semantic artifact runner count mismatch: {kind}")
    return result


def seal_manifest(
    payload: Mapping[str, Any],
    bundle: ContractBundle | None = None,
    *,
    artifact_bytes_by_path: Mapping[str, bytes],
) -> dict[str, Any]:
    bundle = bundle or load_and_verify_contract()
    manifest = copy.deepcopy(dict(payload))
    supplied_fields = set(manifest) - {"content_hash"}
    expected_fields = set(MANIFEST_REQUIRED_FIELDS)
    if manifest.get("manifest_kind") == "canonical_root_manifest":
        expected_fields.add("dependency_manifest_digests")
    _require(supplied_fields == expected_fields, "manifest fields differ from the default-deny schema")
    _validate_safety_flags(manifest)
    _require(manifest.get("experiment_id") == EXPERIMENT_ID, "manifest experiment id mismatch")
    _require(manifest.get("manifest_kind") in MANIFEST_KINDS, "manifest_kind is not approved")
    _require_nonblank_string(manifest.get("release_family_id"), "release_family_id")
    _require(type(manifest.get("release_version")) is int and manifest["release_version"] >= 1, "release_version is invalid")
    if manifest["release_version"] == 1:
        _require(manifest.get("parent_manifest_digest") is None, "v1 parent manifest must be null")
    else:
        _require_hash(manifest.get("parent_manifest_digest"), "parent_manifest_digest")
    as_of = _parse_timestamp(manifest.get("as_of"), "as_of")
    source_cutoff = _parse_timestamp(manifest.get("source_cutoff"), "source_cutoff")
    _require(source_cutoff <= as_of, "source_cutoff must not exceed as_of")
    for field in ("row_count", "race_count", "runner_count", "duplicate_count"):
        _require(type(manifest.get(field)) is int and manifest[field] >= 0, f"{field} must be a nonnegative integer")
    _require(manifest["duplicate_count"] == 0, "duplicate_count must be zero")
    _require(
        isinstance(manifest["generator_execution_commit"], str)
        and bool(COMMIT_RE.fullmatch(manifest["generator_execution_commit"])),
        "generator execution commit is invalid",
    )
    for field in ("generator_script_sha256", "config_sha256", "dependency_environment_sha256", "schema_sha256"):
        _require_hash(manifest[field], field)
    for field in ("row_counts", "race_counts", "identity_counts", "duplicate_and_missing_counts", "missing_reason_distribution", "as_of_verdict_counts", "label_eligibility_counts"):
        _require(isinstance(manifest[field], dict) and manifest[field], f"manifest count map is required: {field}")
        _require(
            all(type(value) is int and value >= 0 for value in manifest[field].values()),
            f"manifest count map contains an invalid value: {field}",
        )
    _require(set(manifest["row_counts"]) == {"total"}, "row count map schema mismatch")
    _require(set(manifest["race_counts"]) == {"total"}, "race count map schema mismatch")
    _require(set(manifest["identity_counts"]) == {"runner_count"}, "identity count map schema mismatch")
    _require(
        set(manifest["duplicate_and_missing_counts"]) == {"duplicate_key_count", "missing_key_count"},
        "duplicate/missing count map schema mismatch",
    )
    _require(set(manifest["missing_reason_distribution"]).issubset(MISSING_REASONS), "missing reason count map schema mismatch")
    _require(
        set(manifest["as_of_verdict_counts"]) == {LINEAGE_SAFE_VERDICT},
        "as-of verdict count map schema mismatch",
    )
    expected_label_keys = (
        {"eligible", "ineligible"}
        if manifest["manifest_kind"] == "label_eligibility_manifest"
        else {"not_applicable"}
    )
    _require(set(manifest["label_eligibility_counts"]) == expected_label_keys, "label eligibility count map schema mismatch")
    _require(manifest["row_counts"]["total"] == manifest["row_count"], "row count map mismatch")
    _require(manifest["race_counts"]["total"] == manifest["race_count"], "race count map mismatch")
    _require(manifest["identity_counts"]["runner_count"] == manifest["runner_count"], "runner count map mismatch")
    _require(manifest["duplicate_and_missing_counts"]["duplicate_key_count"] == 0, "duplicate key count must be zero")
    _require(manifest["duplicate_and_missing_counts"]["missing_key_count"] == 0, "missing key count must be zero")
    _require(sum(manifest["missing_reason_distribution"].values()) == manifest["row_count"], "missing reason distribution mismatch")
    _require(sum(manifest["as_of_verdict_counts"].values()) == manifest["row_count"], "as-of verdict distribution mismatch")
    _require(sum(manifest["label_eligibility_counts"].values()) == manifest["row_count"], "label eligibility distribution mismatch")
    _require(
        type(manifest["source_time_completeness"]) is float and manifest["source_time_completeness"] == 1.0,
        "source-time completeness must be exactly 1.0",
    )
    _require(
        type(manifest["source_hash_completeness"]) is float and manifest["source_hash_completeness"] == 1.0,
        "source-hash completeness must be exactly 1.0",
    )
    _require(manifest["certification_status"] == "certified", "manifest is not certified")
    input_sources = manifest["input_source_paths_and_sha256"]
    _require(isinstance(input_sources, list) and input_sources, "input source references are required")
    _require(
        input_sources
        == sorted(input_sources, key=lambda item: (str(item.get("path", "")), str(item.get("sha256", "")))),
        "input source references must be sorted deterministically",
    )
    source_paths: set[str] = set()
    for source in input_sources:
        _require(set(source) == {"path", "sha256"}, "input source reference schema mismatch")
        source_path = _validate_manifest_path(source["path"], bundle, output=False)
        _require(source_path not in source_paths, "duplicate input source path")
        source_paths.add(source_path)
        _require(not Path(source_path).is_absolute() and ".." not in Path(source_path).parts, "input source path is unsafe")
        _require_hash(source["sha256"], f"input source hash {source_path}")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, "manifest artifacts are required")
    artifact_names: set[str] = set()
    artifact_paths: set[str] = set()
    _require(len(artifacts) == 1, "each canonical manifest must bind exactly one role-specific artifact")
    for artifact in artifacts:
        _require(set(artifact) == MANIFEST_ARTIFACT_FIELDS, "artifact reference schema mismatch")
        _require_nonblank_string(artifact["name"], "artifact name")
        _require(
            artifact["name"] == ARTIFACT_NAME_BY_KIND[manifest["manifest_kind"]],
            "artifact name does not match the manifest role",
        )
        _require(artifact["name"] not in artifact_names, "duplicate artifact name")
        artifact_names.add(artifact["name"])
        artifact_path = _validate_manifest_path(artifact["path"], bundle, output=True)
        _require(artifact_path not in artifact_paths, "duplicate artifact path")
        artifact_paths.add(artifact_path)
        _require_hash(artifact["sha256"], f"artifact hash {artifact.get('name')}")
        _require(type(artifact["row_count"]) is int and artifact["row_count"] >= 0, "artifact row_count is invalid")
        _require(artifact["row_count"] == manifest["row_count"], "artifact row_count differs from manifest row_count")
    expected_outputs = [
        {"path": artifact["path"], "sha256": artifact["sha256"]}
        for artifact in artifacts
    ]
    _require(manifest["output_artifact_paths_and_sha256"] == expected_outputs, "output artifact binding list mismatch")
    expected_artifact_paths = {str(item["path"]) for item in artifacts}
    _require(set(artifact_bytes_by_path) == expected_artifact_paths, "certified manifest requires exact artifact byte evidence")
    for artifact in artifacts:
        artifact_path = str(artifact["path"])
        raw = artifact_bytes_by_path[artifact_path]
        _require(sha256_bytes(raw) == artifact["sha256"], f"artifact byte hash mismatch: {artifact_path}")
        _require(
            _observed_artifact_row_count(artifact_path, raw) == artifact["row_count"],
            f"artifact observed row count mismatch: {artifact_path}",
        )
        _validate_artifact_semantics(manifest, raw, bundle)
    manifest.pop("content_hash", None)
    manifest["content_hash"] = canonical_digest(manifest)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    bundle: ContractBundle | None = None,
    artifact_bytes_by_path: Mapping[str, bytes],
) -> None:
    bundle = bundle or load_and_verify_contract()
    observed = copy.deepcopy(dict(manifest))
    _require_hash(observed.get("content_hash"), "manifest content_hash")
    resealed = seal_manifest(
        {key: value for key, value in observed.items() if key != "content_hash"},
        bundle,
        artifact_bytes_by_path=artifact_bytes_by_path,
    )
    _require(observed == resealed, "manifest content or contract mismatch")


def _certified_source_contexts(
    by_kind: Mapping[str, Mapping[str, Any]],
    semantic_by_kind: Mapping[str, Mapping[str, Any]],
    bundle: ContractBundle,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str], set[str]],
]:
    """Recompute every row source hash and retain its immutable provenance.

    Manifest metadata is not accepted as source proof by itself.  A row digest
    is certified only when exactly one manifest input path reproduces the
    canonical source payload.  The resulting digest-to-context map is then the
    sole authority used by the feature-lineage connector.
    """

    contexts: dict[str, dict[str, Any]] = {}
    current_hashes_by_identity: dict[tuple[str, str], set[str]] = {}

    def input_paths(kind: str) -> tuple[str, ...]:
        refs = by_kind[kind]["input_source_paths_and_sha256"]
        paths = tuple(str(item["path"]) for item in refs)
        _require(paths, f"source manifest has no input paths: {kind}")
        return paths

    def register(
        digest: str,
        payload: Mapping[str, Any],
        *,
        partition: str,
        record_kind: str,
        row: Mapping[str, Any],
        source_path: str,
    ) -> None:
        identity = _require_identity(row)
        context = {
            "partition": partition,
            "record_kind": record_kind,
            "race_id": identity[0],
            "horse_id": identity[1],
            "source_path": source_path,
            "source_version": _require_nonblank_string(row["source_version"], "certified source_version"),
            "source_event_time": utc_text(_parse_timestamp(row["source_event_time"], "certified source_event_time")),
            "received_at": utc_text(_parse_timestamp(row["received_at"], "certified received_at")),
            "available_as_of": utc_text(_parse_timestamp(row["available_as_of"], "certified available_as_of")),
            "prediction_event_time": utc_text(
                _parse_timestamp(
                    row["as_of"] if partition == "runner" else row["prediction_event_time"],
                    "certified prediction_event_time",
                )
            ),
            "canonical_payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
        }
        existing = contexts.get(digest)
        _require(
            existing is None or existing == context,
            f"one source hash is bound to different canonical payload/provenance: {digest}",
        )
        contexts[digest] = context

    runner_paths = input_paths("runner_universe_manifest")
    for row in semantic_by_kind["runner_universe_manifest"]["rows"]:
        digest = _require_hash(row["source_content_sha256"], "runner source_content_sha256")
        matches = [
            (path, canonical_runner_source_payload(row, path))
            for path in runner_paths
            if source_payload_hash(canonical_runner_source_payload(row, path)) == digest
        ]
        _require(len(matches) == 1, f"runner source hash does not bind exactly one input path: {_require_identity(row)}")
        source_path, payload = matches[0]
        register(
            digest,
            payload,
            partition="runner",
            record_kind="runner_universe",
            row=row,
            source_path=source_path,
        )
        safe_payload = runner_feature_dependency_payload(payload)
        safe_digest = source_payload_hash(safe_payload)
        register(
            safe_digest,
            safe_payload,
            partition="runner",
            record_kind="runner_feature_safe",
            row=row,
            source_path=source_path,
        )
        # The full runner payload is retained above for universe/version audit,
        # but it contains draw_status/frame_number/horse_number.  Only the
        # explicitly projected draw-free digest may satisfy a model feature's
        # current-input lineage requirement.
        current_hashes_by_identity.setdefault(_require_identity(row), set()).add(safe_digest)

    for manifest_kind, partition in (
        ("target_source_manifest", "target"),
        ("training_source_manifest", "training"),
    ):
        paths = input_paths(manifest_kind)
        for row in semantic_by_kind[manifest_kind]["rows"]:
            digest = _require_hash(row["source_content_sha256"], "event source_content_sha256")
            matches = [
                (path, canonical_event_source_payload(row, path, bundle))
                for path in paths
                if source_payload_hash(canonical_event_source_payload(row, path, bundle)) == digest
            ]
            _require(
                len(matches) == 1,
                f"event source hash does not bind exactly one input path: {partition}/{row['record_kind']}/{_require_identity(row)}",
            )
            source_path, payload = matches[0]
            register(
                digest,
                payload,
                partition=partition,
                record_kind=str(row["record_kind"]),
                row=row,
                source_path=source_path,
            )
            if partition == "target" and row["record_kind"] == "declared_card":
                current_hashes_by_identity.setdefault(_require_identity(row), set()).add(digest)

    return contexts, current_hashes_by_identity


def _validated_manifest_map(
    manifests: Iterable[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    artifact_bytes_by_path: Mapping[str, bytes],
) -> dict[str, dict[str, Any]]:
    all_manifests = [dict(raw) for raw in manifests]
    expected_artifact_paths = {
        str(artifact["path"])
        for manifest in all_manifests
        for artifact in manifest.get("artifacts", [])
        if isinstance(artifact, dict) and "path" in artifact
    }
    _require(set(artifact_bytes_by_path) == expected_artifact_paths, "manifest-set artifact byte evidence is incomplete or contains extras")
    by_kind: dict[str, dict[str, Any]] = {}
    semantic_by_kind: dict[str, dict[str, Any]] = {}
    for manifest in all_manifests:
        manifest_paths = {str(item["path"]) for item in manifest.get("artifacts", [])}
        validate_manifest(
            manifest,
            bundle=bundle,
            artifact_bytes_by_path={path: artifact_bytes_by_path[path] for path in manifest_paths},
        )
        kind = str(manifest["manifest_kind"])
        _require(kind not in by_kind, f"duplicate manifest kind: {kind}")
        by_kind[kind] = manifest
        artifact = manifest["artifacts"][0]
        semantic_by_kind[kind] = _validate_artifact_semantics(
            manifest,
            artifact_bytes_by_path[str(artifact["path"])],
            bundle,
        )
    expected = set(bundle.config["required_manifest_kinds"])
    _require(set(by_kind) == expected, "manifest set does not contain exactly the approved kinds")
    root = by_kind["canonical_root_manifest"]
    dependencies = root.get("dependency_manifest_digests")
    _require(isinstance(dependencies, dict), "root dependency manifest map is required")
    leaf_kinds = expected - {"canonical_root_manifest"}
    _require(set(dependencies) == leaf_kinds, "root manifest does not bind every approved leaf manifest")
    for kind in leaf_kinds:
        _require(dependencies[kind] == by_kind[kind]["content_hash"], f"root manifest leaf hash mismatch: {kind}")
    synchronized = set(bundle.config["manifest_composition_contract"]["root_synchronized_kinds"])
    reusable = set(bundle.config["manifest_composition_contract"]["immutable_reusable_kinds"])
    _require(synchronized | reusable == leaf_kinds, "manifest composition kind policy is incomplete")
    execution_fields = (
        "generator_execution_commit",
        "generator_script_sha256",
        "config_sha256",
        "dependency_environment_sha256",
    )
    for kind in leaf_kinds:
        leaf = by_kind[kind]
        for field in execution_fields:
            _require(leaf[field] == root[field], f"manifest execution binding differs from root: {kind}/{field}")
        if kind in synchronized:
            for field in ("release_family_id", "release_version", "parent_manifest_digest", "as_of"):
                _require(leaf[field] == root[field], f"root-synchronized release binding differs: {kind}/{field}")
        else:
            _require(leaf["release_version"] <= root["release_version"], f"reusable manifest is newer than root: {kind}")
            _require(
                _parse_timestamp(leaf["as_of"], f"{kind} as_of") <= _parse_timestamp(root["as_of"], "root as_of"),
                f"reusable manifest cutoff exceeds root: {kind}",
            )
    max_cutoff = max(_parse_timestamp(by_kind[kind]["source_cutoff"], f"{kind} source_cutoff") for kind in leaf_kinds)
    _require(max_cutoff == _parse_timestamp(root["source_cutoff"], "root source_cutoff"), "root source_cutoff is not the leaf maximum")
    _require(root["row_count"] == sum(by_kind[kind]["row_count"] for kind in leaf_kinds), "root row count does not equal leaf total")
    runner_manifest = by_kind["runner_universe_manifest"]
    _require(root["race_count"] == runner_manifest["race_count"], "root race count differs from runner manifest")
    _require(root["runner_count"] == runner_manifest["runner_count"], "root runner count differs from runner manifest")
    for kind in synchronized:
        leaf = by_kind[kind]
        _require(leaf["race_count"] == runner_manifest["race_count"], f"manifest race universe differs: {kind}")
        _require(leaf["runner_count"] == runner_manifest["runner_count"], f"manifest runner universe differs: {kind}")
    runner_rows = semantic_by_kind["runner_universe_manifest"]["rows"]
    training_rows = semantic_by_kind["training_source_manifest"]["rows"]
    target_rows = semantic_by_kind["target_source_manifest"]["rows"]
    feature_rows = semantic_by_kind["feature_release_manifest"]["rows"]
    lineage_rows = semantic_by_kind["lineage_manifest"]["rows"]
    diff_rows = semantic_by_kind["release_diff_manifest"]["rows"]
    _require(
        semantic_by_kind["training_source_manifest"]["race_ids"].isdisjoint(
            semantic_by_kind["target_source_manifest"]["race_ids"]
        ),
        "training source contains a target-race identity/result",
    )
    certified_source_contexts, current_hashes_by_identity = _certified_source_contexts(
        by_kind,
        semantic_by_kind,
        bundle,
    )
    runner_by_identity = {_require_identity(row): row for row in runner_rows}
    target_by_identity = {_require_identity(row): row for row in target_rows}
    feature_by_identity = {_require_identity(row): row for row in feature_rows}
    diff_by_identity = {_require_identity(row): row for row in diff_rows}
    runner_identities = set(runner_by_identity)
    active_runner_identities = {
        identity for identity, row in runner_by_identity.items() if row["active_for_feature_materialization"]
    }
    active_runner_count_by_race: dict[str, int] = {}
    for race_id, _horse_id in active_runner_identities:
        active_runner_count_by_race[race_id] = active_runner_count_by_race.get(race_id, 0) + 1
    target_identities = set(target_by_identity)
    feature_identities = set(feature_by_identity)
    lineage_identities = semantic_by_kind["lineage_manifest"]["identities"]
    diff_identities = set(diff_by_identity)
    _require(runner_identities == target_identities, "target source identity set differs from runner universe")
    _require(active_runner_identities == feature_identities, "feature identity set differs from the effective runner universe")
    _require(active_runner_identities == lineage_identities, "lineage identity set differs from the effective runner universe")
    _require(runner_identities == diff_identities, "release diff identity set differs from runner universe")
    for identity in sorted(runner_identities):
        runner_row = runner_by_identity[identity]
        target_row = target_by_identity[identity]
        _require(target_row["record_kind"] == "declared_card", "target artifact contains a non-card row")
        _require(target_row["騎手コード"] == runner_row["jockey_id"], f"runner/target jockey mismatch: {identity}")
        _require(target_row["調教師コード"] == runner_row["trainer_id"], f"runner/target trainer mismatch: {identity}")
        _require(float(target_row["斤量"]) == float(runner_row["carried_weight"]), f"runner/target assigned weight mismatch: {identity}")
        _require(
            utc_text(_parse_timestamp(target_row["prediction_event_time"], "target prediction_event_time"))
            == utc_text(_parse_timestamp(runner_row["as_of"], "runner as_of")),
            f"runner/target cutoff mismatch: {identity}",
        )
        if identity in feature_by_identity:
            wide = feature_by_identity[identity]
            _require(
                wide["出走頭数"] == float(active_runner_count_by_race[identity[0]]),
                f"declared-field runner count/feature mismatch: {identity}/出走頭数",
            )
            for field in ("年齢", "斤量", "距離"):
                _require(
                    float(wide[field]) == float(target_row[field]),
                    f"target card/feature numeric core mismatch: {identity}/{field}",
                )
            for field in ("場所", "性別", "騎手コード", "調教師コード", "芝・ダ", "クラス名", "トラックコード"):
                _require(
                    wide[field] == target_row[field],
                    f"target card/feature categorical core mismatch: {identity}/{field}",
                )
    lineage_by_triple = {
        (row["race_id"], row["horse_id"], row["feature_name"]): row for row in lineage_rows
    }
    feature_safe_record_kinds = {
        ("runner", "runner_feature_safe"),
        ("target", "declared_card"),
        ("training", "declared_card"),
        ("training", "completed_result"),
    }
    certified_source_hashes = {
        digest
        for digest, context in certified_source_contexts.items()
        if (context["partition"], context["record_kind"]) in feature_safe_record_kinds
    }
    for identity, wide in feature_by_identity.items():
        runner_row = runner_by_identity[identity]
        prediction_text = utc_text(_parse_timestamp(wide["prediction_event_time"], "feature prediction_event_time"))
        _require(
            prediction_text == utc_text(_parse_timestamp(runner_row["as_of"], "runner as_of")),
            f"runner/feature cutoff mismatch: {identity}",
        )
        for feature_name in bundle.ordered_features:
            lineage_row = lineage_by_triple[(identity[0], identity[1], feature_name)]
            _require(
                set(lineage_row["source_content_sha256_set"]).issubset(certified_source_hashes),
                f"feature lineage cites a source hash absent from sealed source artifacts: {identity}/{feature_name}",
            )
            source_hashes = set(lineage_row["source_content_sha256_set"])
            contexts = [certified_source_contexts[digest] for digest in lineage_row["source_content_sha256_set"]]
            expected_paths = sorted({context["source_path"] for context in contexts})
            expected_versions = sorted({context["source_version"] for context in contexts})
            _require(lineage_row["source_paths"] == expected_paths, f"lineage source paths are not derived from sealed sources: {identity}/{feature_name}")
            _require(lineage_row["source_versions"] == expected_versions, f"lineage source versions are not derived from sealed sources: {identity}/{feature_name}")
            for field, lineage_field in (
                ("source_event_time", "max_source_event_time"),
                ("received_at", "max_received_at"),
                ("available_as_of", "max_available_as_of"),
            ):
                expected_max = utc_text(max(_parse_timestamp(context[field], field) for context in contexts))
                _require(
                    lineage_row[lineage_field] == expected_max,
                    f"lineage {lineage_field} is not derived from sealed sources: {identity}/{feature_name}",
                )
            active_same_race = {
                candidate for candidate in active_runner_identities if candidate[0] == identity[0]
            }
            allowed_historical_horses = (
                {candidate[1] for candidate in active_same_race}
                if feature_name in RACE_AGGREGATE_FEATURES
                else {identity[1]}
            )
            for context in contexts:
                context_identity = (context["race_id"], context["horse_id"])
                if context["partition"] in {"runner", "target"}:
                    expected_current_kind = (
                        "runner_feature_safe" if context["partition"] == "runner" else "declared_card"
                    )
                    _require(
                        context["record_kind"] == expected_current_kind,
                        f"unapproved target/current source entered predraw lineage: {identity}/{feature_name}",
                    )
                    if feature_name in RACE_AGGREGATE_FEATURES:
                        _require(
                            context_identity in active_same_race,
                            f"race aggregate cites another target race/runner: {identity}/{feature_name}/{context_identity}",
                        )
                    else:
                        _require(
                            context_identity == identity,
                            f"horse feature cites another current runner: {identity}/{feature_name}/{context_identity}",
                        )
                else:
                    _require(context["partition"] == "training", "unapproved lineage source partition")
                    _require(context["race_id"] != identity[0], "target race appears in historical lineage")
                    _require(
                        _parse_timestamp(context["prediction_event_time"], "historical prediction_event_time")
                        < _parse_timestamp(lineage_row["prediction_event_time"], "lineage prediction_event_time"),
                        f"historical lineage event is not strictly earlier: {identity}/{feature_name}/{context_identity}",
                    )
                    _require(
                        context["horse_id"] in allowed_historical_horses,
                        f"historical lineage cites an unrelated horse: {identity}/{feature_name}/{context_identity}",
                    )
                    _require(
                        _parse_timestamp(context["available_as_of"], "historical available_as_of")
                        < _parse_timestamp(lineage_row["prediction_event_time"], "lineage prediction_event_time"),
                        f"historical lineage is not strictly prior: {identity}/{feature_name}/{context_identity}",
                    )
            required_current_identities = active_same_race if feature_name in RACE_AGGREGATE_FEATURES else {identity}
            for required_identity in required_current_identities:
                _require(
                    source_hashes & current_hashes_by_identity.get(required_identity, set()),
                    f"lineage omits current certified input for {required_identity}: {identity}/{feature_name}",
                )
            _require(lineage_row["prediction_event_time"] == prediction_text, f"feature/lineage cutoff mismatch: {identity}/{feature_name}")
            _require(
                _parse_timestamp(lineage_row["max_available_as_of"], "lineage max_available_as_of")
                < _parse_timestamp(prediction_text, "feature prediction_event_time"),
                f"feature lineage is not available pre-race: {identity}/{feature_name}",
            )
            if lineage_row["missing_reason"] != "not_applicable":
                if feature_name in set(bundle.categorical_features):
                    expected_missing_value: Any = "__MISSING__"
                elif feature_name == "prev_corner4_position_rate":
                    expected_missing_value = 0.5
                else:
                    expected_missing_value = 0.0
                _require(
                    wide[feature_name] == expected_missing_value,
                    f"wide feature value contradicts lineage missing semantics: {identity}/{feature_name}",
                )
            expected_value_binding = feature_value_binding_hash(
                race_id=identity[0],
                horse_id=identity[1],
                feature_name=feature_name,
                feature_value=wide[feature_name],
                feature_dtype=lineage_row["feature_dtype"],
                prediction_event_time=prediction_text,
                transformation_name=lineage_row["transformation_name"],
                transformation_version=lineage_row["transformation_version"],
                transformation_code_sha256=lineage_row["transformation_code_sha256"],
                source_content_sha256_set=lineage_row["source_content_sha256_set"],
            )
            _require(
                set(lineage_row["dependency_content_sha256_set"])
                == source_hashes | {expected_value_binding},
                f"feature dependencies are not the exact sealed sources plus value binding: {identity}/{feature_name}",
            )
    _require(
        by_kind["feature_release_manifest"]["source_cutoff"]
        == by_kind["lineage_manifest"]["source_cutoff"],
        "feature and lineage source cutoffs differ",
    )
    _require(
        by_kind["release_diff_manifest"]["source_cutoff"]
        == by_kind["runner_universe_manifest"]["source_cutoff"],
        "release diff and runner source cutoffs differ",
    )
    absent_hash = canonical_digest({"state": "absent"})
    allowed_diff_fields = set(bundle.input_contract["saturday_postdraw_update_contract"]["versioned_fields"]) | {
        "runner_status",
        "release_version",
        "parent_manifest_digest",
        "as_of",
        "source_event_time",
        "received_at",
        "available_as_of",
        "source_version",
        "source_content_sha256",
        "missing_reason",
    }
    for identity, diff_row in diff_by_identity.items():
        runner_row = runner_by_identity[identity]
        _require(diff_row["new_row_payload_sha256"] == runner_row["row_payload_sha256"], f"diff does not bind current runner row: {identity}")
        _require(diff_row["source_content_sha256"] == runner_row["source_content_sha256"], f"diff source hash differs from runner row: {identity}")
        change_fields = [change["field"] for change in diff_row["changes"]]
        _require(len(change_fields) == len(set(change_fields)), f"diff contains duplicate changed fields: {identity}")
        _require(set(change_fields).issubset(allowed_diff_fields), f"diff contains an unapproved field: {identity}")
        if runner_manifest["release_version"] == 1:
            _require(diff_row["old_row_payload_sha256"] == absent_hash, f"v1 diff old-row sentinel mismatch: {identity}")
            _require(diff_row["reason"] == "initial_declaration", f"v1 diff reason mismatch: {identity}")
            _require(
                diff_row["changes"] == [{"field": "release_version", "old": None, "new": 1}],
                f"v1 diff is not the exact initial declaration: {identity}",
            )
        else:
            _require(diff_row["reason"] != "initial_declaration", f"child diff claims an initial declaration: {identity}")

    training_cards = {
        _require_identity(row): row for row in training_rows if row["record_kind"] == "declared_card"
    }
    training_results = {
        _require_identity(row): row for row in training_rows if row["record_kind"] == "completed_result"
    }
    label_rows = semantic_by_kind["label_eligibility_manifest"]["rows"]
    label_by_identity = {_require_identity(row): row for row in label_rows}
    _require(set(training_cards) == set(label_by_identity), "label ledger identity set differs from training declared cards")
    for identity, label_row in label_by_identity.items():
        result_row = training_results.get(identity)
        if result_row is None:
            _require(
                label_row["starter_status"] == "starter",
                f"pre-cutoff nonstarter status lacks a separately sealed status-evidence ledger: {identity}",
            )
            _require(
                label_row["official_result_status"] is None
                and label_row["official_finish_rank_raw"] is None,
                f"missing-result label carries result fields: {identity}",
            )
            _require(label_row["source_content_sha256"] == training_cards[identity]["source_content_sha256"], f"nonstarter label source differs from card: {identity}")
        else:
            _require(label_row["starter_status"] == "starter", f"training result is attached to a nonstarter label: {identity}")
            _require(label_row["declared_status"] == "declared_active", f"training result is attached to a non-active declaration: {identity}")
            _require(label_row["official_result_status"] == result_row["result_status"], f"label/result status mismatch: {identity}")
            _require(label_row["official_finish_rank_raw"] == result_row["official_finish_rank_raw"], f"label/result rank mismatch: {identity}")
            _require(label_row["source_content_sha256"] == result_row["source_content_sha256"], f"label/result source hash mismatch: {identity}")
    label_races: dict[str, list[dict[str, Any]]] = {}
    for row in label_rows:
        label_races.setdefault(str(row["race_id"]), []).append(row)
    for race_id, race_rows in label_races.items():
        effective = [row for row in race_rows if row["starter_status"] == "starter"]
        observed_race_verdicts = {row["race_label_eligible"] for row in race_rows}
        _require(len(observed_race_verdicts) == 1, f"label race verdict is not atomic: {race_id}")
        expected_reason = _derive_race_ineligibility_reason(effective)
        expected_race_eligible = expected_reason is None
        _require(
            next(iter(observed_race_verdicts)) is expected_race_eligible,
            f"label race verdict is not derived from the frozen result policy: {race_id}",
        )
        for row in race_rows:
            expected_row_reason = expected_reason or (
                "eligible" if row["starter_status"] == "starter" else "precutoff_nonstarter_retained"
            )
            _require(row["ineligibility_reason"] == expected_row_reason, f"label reason differs from training results: {race_id}/{row['horse_id']}")
            _require(
                row["row_label_eligible"] is (expected_race_eligible and row["starter_status"] == "starter"),
                f"label row verdict differs from race policy: {race_id}/{row['horse_id']}",
            )
    _require(
        by_kind["label_eligibility_manifest"]["source_cutoff"]
        == by_kind["training_source_manifest"]["source_cutoff"],
        "label and training source cutoffs differ",
    )
    environment_object = semantic_by_kind["environment_manifest"]["object"]
    dependency_object = semantic_by_kind["dependency_lock_manifest"]["object"]
    _require(
        dependency_object["interpreter_sha256"] == environment_object["executable_sha256"],
        "dependency lock is not bound to the environment interpreter",
    )
    _require(
        dependency_object["config_sha256"] == by_kind["canonical_root_manifest"]["config_sha256"],
        "dependency lock is not bound to the canonicalizer config",
    )
    lineage_artifact_hash = str(_manifest_artifact(by_kind["lineage_manifest"])["sha256"])
    _require(
        semantic_by_kind["feature_release_manifest"].get("lineage_artifact_sha256") == lineage_artifact_hash,
        "feature release does not bind the exact lineage artifact bytes",
    )
    return by_kind


def validate_manifest_set(
    manifests: Iterable[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    artifact_bytes_by_path: Mapping[str, bytes],
) -> str:
    by_kind = _validated_manifest_map(manifests, bundle, artifact_bytes_by_path=artifact_bytes_by_path)
    return str(by_kind["canonical_root_manifest"]["content_hash"])


def _manifest_artifact(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    artifacts = manifest["artifacts"]
    _require(len(artifacts) == 1, "manifest must bind exactly one artifact")
    return artifacts[0]


def _validate_predraw_handoff_runner_rows(
    runner_rows: Iterable[Mapping[str, Any]],
) -> None:
    rows = tuple(dict(row) for row in runner_rows)
    _require(len(rows) == 70, "predraw runner universe must contain exactly 70 rows")
    _require(
        all(row["release_version"] == 1 and row["parent_manifest_digest"] is None for row in rows),
        "predraw runner rows must be immutable v1 rows without a parent",
    )
    race_groups = {
        race_id: [row for row in rows if row["race_id"] == race_id]
        for race_id in sorted({row["race_id"] for row in rows})
    }
    per_race_counts = [len(group) for group in race_groups.values()]
    draw_counts = {
        status: sum(row["draw_status"] == status for row in rows)
        for status in DRAW_STATUSES
    }
    _require(
        sorted(per_race_counts) == sorted([14, 14, 10, 16, 16]),
        "predraw race-count multiset differs from the frozen contract",
    )
    _require(
        draw_counts == {"confirmed": 26, "scheduled_pending_draw": 44, "scratched": 0},
        "predraw draw-status counts differ from the frozen 26/44 contract",
    )
    _require(
        all(len({row["draw_status"] for row in group}) == 1 for group in race_groups.values()),
        "predraw draw publication is not race-atomic",
    )
    confirmed_sizes = sorted(
        len(group) for group in race_groups.values() if group[0]["draw_status"] == "confirmed"
    )
    pending_sizes = sorted(
        len(group)
        for group in race_groups.values()
        if group[0]["draw_status"] == "scheduled_pending_draw"
    )
    _require(
        confirmed_sizes == [10, 16],
        "predraw confirmed races do not match the frozen 10+16 publication",
    )
    _require(
        pending_sizes == [14, 14, 16],
        "predraw pending races do not match the frozen 14+14+16 publication",
    )


def _validate_handoff_execution_context(
    by_kind: Mapping[str, Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    synthetic_fixture: bool,
    expected_execution_bindings: Mapping[str, str] | None,
) -> None:
    binding_fields = {
        "generator_execution_commit",
        "generator_script_sha256",
        "config_sha256",
        "dependency_environment_sha256",
    }
    artifact_paths = {
        str(_manifest_artifact(manifest)["path"])
        for manifest in by_kind.values()
    }
    runner_manifest = by_kind["runner_universe_manifest"]
    runner_artifact = _manifest_artifact(runner_manifest)
    if synthetic_fixture:
        _require(expected_execution_bindings is None, "synthetic handoff must not claim real execution bindings")
        _require(
            all(path.startswith(bundle.config["manifest_path_contract"]["synthetic_output_prefix"]) for path in artifact_paths),
            "synthetic handoff contains a non-synthetic artifact path",
        )
        _require(str(runner_manifest["release_family_id"]).startswith(SYNTHETIC_PREFIX), "synthetic handoff release is not namespaced")
        _require(str(runner_artifact["path"]).startswith("outputs/research/SYN/"), "synthetic runner artifact path mismatch")
        return
    _require(expected_execution_bindings is not None, "canonical handoff requires exact run-scope execution bindings")
    _require(set(expected_execution_bindings) == binding_fields, "canonical handoff execution binding schema mismatch")
    _require(
        all(path.startswith(bundle.config["manifest_path_contract"]["canonical_output_prefix"]) for path in artifact_paths),
        "canonical handoff contains a synthetic or out-of-scope artifact path",
    )
    _require(not str(runner_manifest["release_family_id"]).startswith(SYNTHETIC_PREFIX), "canonical handoff uses a synthetic release family")
    for field in binding_fields:
        expected = expected_execution_bindings[field]
        if field == "generator_execution_commit":
            _require(isinstance(expected, str) and COMMIT_RE.fullmatch(expected), "expected execution commit is invalid")
        else:
            _require_hash(expected, f"expected {field}")
        _require(all(manifest[field] == expected for manifest in by_kind.values()), f"canonical handoff execution binding mismatch: {field}")


def _validate_draw_confirmed_parent_chain(
    child_by_kind: Mapping[str, Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    child_artifact_bytes_by_path: Mapping[str, bytes],
    parent_manifests: Iterable[Mapping[str, Any]],
    parent_artifact_bytes_by_path: Mapping[str, bytes],
    expected_parent_manifest_digest: str,
    expected_parent_root_manifest_digest: str,
) -> None:
    expected_parent = _require_hash(expected_parent_manifest_digest, "expected parent manifest digest")
    expected_parent_root = _require_hash(
        expected_parent_root_manifest_digest,
        "expected parent root manifest digest",
    )
    parent_by_kind = _validated_manifest_map(
        parent_manifests,
        bundle,
        artifact_bytes_by_path=parent_artifact_bytes_by_path,
    )
    child_runner_manifest = child_by_kind["runner_universe_manifest"]
    parent_runner_manifest = parent_by_kind["runner_universe_manifest"]
    parent_root_manifest = parent_by_kind["canonical_root_manifest"]
    _require(
        parent_root_manifest["content_hash"] == expected_parent_root,
        "parent manifest set is not anchored to the expected canonical root",
    )
    _require(parent_runner_manifest["content_hash"] == expected_parent, "parent runner manifest is not the expected sealed parent")
    _require(
        child_runner_manifest["parent_manifest_digest"] == parent_runner_manifest["content_hash"],
        "child release does not bind the exact parent runner manifest",
    )
    _require(
        child_runner_manifest["release_family_id"] == parent_runner_manifest["release_family_id"],
        "child release family differs from its parent",
    )
    _require(
        child_runner_manifest["release_version"] == parent_runner_manifest["release_version"] + 1,
        "child release version does not increment its parent by one",
    )
    synchronized_parent_kinds = {
        *bundle.config["manifest_composition_contract"]["root_synchronized_kinds"],
        "canonical_root_manifest",
    }
    _require(
        all(
            parent_by_kind[kind]["release_version"] == 1
            and parent_by_kind[kind]["parent_manifest_digest"] is None
            for kind in synchronized_parent_kinds
        ),
        "draw-confirmed parent is not the frozen predraw v1/root release",
    )
    for kind in bundle.config["manifest_composition_contract"]["immutable_reusable_kinds"]:
        _require(
            child_by_kind[kind]["content_hash"] == parent_by_kind[kind]["content_hash"],
            f"draw-confirmed child changed an immutable reusable manifest: {kind}",
        )

    child_runner_artifact = _manifest_artifact(child_runner_manifest)
    parent_runner_artifact = _manifest_artifact(parent_runner_manifest)
    child_diff_manifest = child_by_kind["release_diff_manifest"]
    child_diff_artifact = _manifest_artifact(child_diff_manifest)
    child_runner_rows = _validate_artifact_semantics(
        child_runner_manifest,
        child_artifact_bytes_by_path[str(child_runner_artifact["path"])],
        bundle,
    )["rows"]
    parent_runner_rows = _validate_artifact_semantics(
        parent_runner_manifest,
        parent_artifact_bytes_by_path[str(parent_runner_artifact["path"])],
        bundle,
    )["rows"]
    _validate_predraw_handoff_runner_rows(parent_runner_rows)
    diff_rows = _validate_artifact_semantics(
        child_diff_manifest,
        child_artifact_bytes_by_path[str(child_diff_artifact["path"])],
        bundle,
    )["rows"]
    child_by_identity = {_require_identity(row): row for row in child_runner_rows}
    parent_by_identity = {_require_identity(row): row for row in parent_runner_rows}
    diff_by_identity = {_require_identity(row): row for row in diff_rows}
    _require(set(child_by_identity) == set(parent_by_identity) == set(diff_by_identity), "parent/child/diff identity sets differ")

    audited_fields = _runner_diff_audited_fields(bundle)
    draw_only_update = True
    for identity in sorted(child_by_identity):
        parent = parent_by_identity[identity]
        child = child_by_identity[identity]
        diff = diff_by_identity[identity]
        _require(diff["old_row_payload_sha256"] == parent["row_payload_sha256"], f"diff does not bind parent row: {identity}")
        _require(diff["new_row_payload_sha256"] == child["row_payload_sha256"], f"diff does not bind child row: {identity}")
        expected_changes = [
            {"field": field, "old": parent.get(field), "new": child.get(field)}
            for field in audited_fields
            if parent.get(field) != child.get(field)
        ]
        feature_material_change_fields = {
            "runner_status",
            "jockey_id",
            "carried_weight",
            "trainer_id",
            "active_for_feature_materialization",
        }
        if any(change["field"] in feature_material_change_fields for change in expected_changes):
            draw_only_update = False
        _require(diff["changes"] == expected_changes, f"diff is not the exact parent-to-child field projection: {identity}")
        old_draw, new_draw = parent["draw_status"], child["draw_status"]
        _require(old_draw != "scratched", f"scratched parent runner was revised: {identity}")
        _require(
            (old_draw == "scheduled_pending_draw" and new_draw in {"confirmed", "scratched"})
            or (old_draw == "confirmed" and new_draw in {"confirmed", "scratched"}),
            f"unapproved parent-to-child draw transition: {identity}",
        )
        if old_draw == "confirmed" and new_draw == "scratched":
            _require(
                child["frame_number"] == parent["frame_number"]
                and child["horse_number"] == parent["horse_number"],
                f"postdraw scratch changed its published number: {identity}",
            )
        source_changed = any(
            parent.get(field) != child.get(field)
            for field in ("source_event_time", "received_at", "available_as_of", "source_version", "source_content_sha256")
        )
        if source_changed:
            _require(
                _parse_timestamp(parent["available_as_of"], "parent available_as_of")
                < _parse_timestamp(child["available_as_of"], "child available_as_of")
                < _parse_timestamp(child["as_of"], "child as_of"),
                f"child source evidence is not a newer pre-race receipt: {identity}",
            )
        expected_reason = child["change_reason"] if source_changed else "release_version_successor"
        _require(diff["reason"] == expected_reason, f"diff reason is not derived from the child row: {identity}")

    if draw_only_update:
        def artifact_rows(
            manifests_by_kind: Mapping[str, Mapping[str, Any]],
            evidence: Mapping[str, bytes],
            kind: str,
        ) -> tuple[dict[str, Any], ...]:
            manifest = manifests_by_kind[kind]
            artifact = _manifest_artifact(manifest)
            return _validate_artifact_semantics(
                manifest,
                evidence[str(artifact["path"])],
                bundle,
            )["rows"]

        parent_feature_rows = artifact_rows(
            parent_by_kind, parent_artifact_bytes_by_path, "feature_release_manifest"
        )
        child_feature_rows = artifact_rows(
            child_by_kind, child_artifact_bytes_by_path, "feature_release_manifest"
        )
        feature_ignored_fields = {"release_family_id", "release_version", "lineage_manifest_sha256"}
        parent_feature_projection = [
            {key: value for key, value in row.items() if key not in feature_ignored_fields}
            for row in parent_feature_rows
        ]
        child_feature_projection = [
            {key: value for key, value in row.items() if key not in feature_ignored_fields}
            for row in child_feature_rows
        ]
        _require(
            parent_feature_projection == child_feature_projection,
            "draw-only child changed a predraw-safe feature value or identity",
        )
        _require(
            artifact_rows(parent_by_kind, parent_artifact_bytes_by_path, "lineage_manifest")
            == artifact_rows(child_by_kind, child_artifact_bytes_by_path, "lineage_manifest"),
            "draw-only child changed predraw-safe feature lineage",
        )


def derive_exp033_handoff_bindings(
    manifests: Iterable[Mapping[str, Any]],
    bundle: ContractBundle,
    *,
    stage: str,
    artifact_bytes_by_path: Mapping[str, bytes],
    synthetic_fixture: bool = False,
    expected_execution_bindings: Mapping[str, str] | None = None,
    parent_manifests: Iterable[Mapping[str, Any]] | None = None,
    parent_artifact_bytes_by_path: Mapping[str, bytes] | None = None,
    expected_parent_manifest_digest: str | None = None,
    expected_parent_root_manifest_digest: str | None = None,
) -> dict[str, Any]:
    _require(stage in bundle.config["exp033_handoff"]["allowed_stages"], "handoff stage is invalid")
    if not synthetic_fixture:
        raise AuthorizationError(
            f"{REAL_DATA_BLOCKER_CODE}: canonical EXP-033 handoff is not available until the "
            "training-feature, status-evidence, per-race cutoff, and versioned real-data contracts are approved"
        )
    by_kind = _validated_manifest_map(manifests, bundle, artifact_bytes_by_path=artifact_bytes_by_path)
    _validate_handoff_execution_context(
        by_kind,
        bundle,
        synthetic_fixture=synthetic_fixture,
        expected_execution_bindings=expected_execution_bindings,
    )
    root = by_kind["canonical_root_manifest"]
    runner = by_kind["runner_universe_manifest"]
    feature = by_kind["feature_release_manifest"]
    training = by_kind["training_source_manifest"]
    target = by_kind["target_source_manifest"]
    lineage = by_kind["lineage_manifest"]
    label_eligibility = by_kind["label_eligibility_manifest"]
    environment = by_kind["environment_manifest"]
    dependency = by_kind["dependency_lock_manifest"]
    _require(runner["race_count"] == 5 and runner["runner_count"] == 70, "EXP-033 handoff requires the frozen 5-race/70-runner universe")
    for kind, manifest in (("target", target), ("feature", feature), ("lineage", lineage)):
        _require(
            manifest["race_count"] == 5 and manifest["runner_count"] == 70,
            f"EXP-033 {kind} manifest is incomplete for the frozen runner universe",
        )
    _require(target["row_count"] == 70, "target declared-card release must contain exactly 70 rows")
    _require(feature["row_count"] == 70, "feature input release must contain exactly 70 wide rows")
    _require(lineage["row_count"] == 70 * len(bundle.ordered_features), "lineage release must contain exactly 70 x 88 rows")
    runner_artifact = _manifest_artifact(runner)
    runner_semantic = _validate_artifact_semantics(
        runner,
        artifact_bytes_by_path[str(runner_artifact["path"])],
        bundle,
    )
    runner_rows = runner_semantic["rows"]
    if stage == "predraw":
        _require(
            parent_manifests is None
            and parent_artifact_bytes_by_path is None
            and expected_parent_manifest_digest is None
            and expected_parent_root_manifest_digest is None,
            "predraw handoff must not claim a parent release",
        )
        _validate_predraw_handoff_runner_rows(runner_rows)
    else:
        _require(
            parent_manifests is not None
            and parent_artifact_bytes_by_path is not None
            and expected_parent_manifest_digest is not None
            and expected_parent_root_manifest_digest is not None,
            "draw-confirmed handoff requires exact parent manifests, artifact bytes, and digest",
        )
        _validate_draw_confirmed_parent_chain(
            by_kind,
            bundle,
            child_artifact_bytes_by_path=artifact_bytes_by_path,
            parent_manifests=parent_manifests,
            parent_artifact_bytes_by_path=parent_artifact_bytes_by_path,
            expected_parent_manifest_digest=expected_parent_manifest_digest,
            expected_parent_root_manifest_digest=expected_parent_root_manifest_digest,
        )
        _require(runner["release_version"] >= 2 and runner["parent_manifest_digest"] is not None, "draw-confirmed handoff must be an immutable child release")
        _require(all(row["draw_status"] in {"confirmed", "scratched"} for row in runner_rows), "draw-confirmed handoff still contains pending draw rows")
        _require(all(row["active_for_feature_materialization"] for row in runner_rows), "a scratched child cannot silently satisfy EXP-033's frozen 70-runner gate")
    completeness_manifests = (root, runner, target, feature, lineage)
    _require(
        all(item["source_time_completeness"] == 1.0 for item in completeness_manifests),
        "handoff source-time completeness is not certified",
    )

    def release_id(manifest: Mapping[str, Any]) -> str:
        return f"{manifest['release_family_id']}:v{manifest['release_version']}"

    return {
        "canonical_root_manifest_sha256": root["content_hash"],
        "runner_universe_release_sha256": _manifest_artifact(runner)["sha256"],
        "feature_input_release_sha256": _manifest_artifact(feature)["sha256"],
        "training_release_sha256": _manifest_artifact(training)["sha256"],
        "target_release_sha256": _manifest_artifact(target)["sha256"],
        "lineage_manifest_sha256": lineage["content_hash"],
        "label_eligibility_manifest_sha256": label_eligibility["content_hash"],
        "fold_manifest_sha256": bundle.config["exp033_handoff"]["exp033_fold_manifest_sha256"],
        "environment_manifest_sha256": environment["content_hash"],
        "dependency_lock_sha256": dependency["content_hash"],
        "runner_universe_release_id": release_id(runner),
        "feature_input_release_id": release_id(feature),
        "training_release_id": release_id(training),
        "target_release_id": release_id(target),
        "label_eligibility_version": bundle.config["label_eligibility"]["version"],
        "stage": stage,
        "target_runner_completeness": 1.0,
        "source_time_completeness": 1.0,
    }


def validate_exp033_handoff(
    handoff: Mapping[str, Any],
    bundle: ContractBundle,
    *,
    manifests: Iterable[Mapping[str, Any]],
    stage: str,
    artifact_bytes_by_path: Mapping[str, bytes],
    synthetic_fixture: bool = False,
    expected_execution_bindings: Mapping[str, str] | None = None,
    parent_manifests: Iterable[Mapping[str, Any]] | None = None,
    parent_artifact_bytes_by_path: Mapping[str, bytes] | None = None,
    expected_parent_manifest_digest: str | None = None,
    expected_parent_root_manifest_digest: str | None = None,
) -> None:
    contract = bundle.config["exp033_handoff"]
    expected_bindings = derive_exp033_handoff_bindings(
        manifests,
        bundle,
        stage=stage,
        artifact_bytes_by_path=artifact_bytes_by_path,
        synthetic_fixture=synthetic_fixture,
        expected_execution_bindings=expected_execution_bindings,
        parent_manifests=parent_manifests,
        parent_artifact_bytes_by_path=parent_artifact_bytes_by_path,
        expected_parent_manifest_digest=expected_parent_manifest_digest,
        expected_parent_root_manifest_digest=expected_parent_root_manifest_digest,
    )
    expected_fields = {
        "contract_version",
        "consumer_experiment_id",
        *contract["required_hash_fields"],
        *contract["required_identity_fields"],
        "exp033_allowlist_sha256",
        "exp033_denylist_sha256",
        "target_runner_completeness",
        "source_time_completeness",
        "formal_buy",
        "send_order",
        "stake",
    }
    _require(set(handoff) == expected_fields, "handoff fields differ from the default-deny schema")
    _require(handoff.get("contract_version") == contract["contract_version"], "handoff contract version mismatch")
    _require(handoff.get("consumer_experiment_id") == "EXP-20260821-033", "handoff consumer mismatch")
    for field in contract["required_hash_fields"]:
        _require_hash(handoff.get(field), field)
        _require(handoff.get(field) == expected_bindings.get(field), f"handoff hash is not bound to sealed input: {field}")
    for field in contract["required_identity_fields"]:
        _require(not _is_blank(handoff.get(field)), f"handoff field is required: {field}")
        _require(handoff.get(field) == expected_bindings.get(field), f"handoff identity is not bound to sealed input: {field}")
    _require(handoff["stage"] in contract["allowed_stages"], "handoff stage is invalid")
    _require(handoff["label_eligibility_version"] == bundle.config["label_eligibility"]["version"], "label eligibility version mismatch")
    _require(handoff.get("exp033_allowlist_sha256") == contract["exp033_allowlist_sha256"], "EXP-033 allowlist hash mismatch")
    _require(handoff.get("exp033_denylist_sha256") == contract["exp033_denylist_sha256"], "EXP-033 denylist hash mismatch")
    _require(handoff.get("fold_manifest_sha256") == contract["exp033_fold_manifest_sha256"], "EXP-033 fold hash mismatch")
    _require(
        type(handoff.get("target_runner_completeness")) is float
        and handoff["target_runner_completeness"] == expected_bindings["target_runner_completeness"],
        "target runner completeness gate failed",
    )
    _require(
        type(handoff.get("source_time_completeness")) is float
        and handoff["source_time_completeness"] == expected_bindings["source_time_completeness"],
        "source-time completeness gate failed",
    )
    _validate_safety_flags(handoff)


def real_data_blocker_report(bundle: ContractBundle) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": REAL_DATA_BLOCKER_CODE,
        "real_data_rows_opened": 0,
        "materialization_executed": False,
        "run_scope_generated": False,
        "requirements": list(bundle.config["real_data_execution_blockers"]),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def refuse_real_materialization(*, source_loader: Callable[..., Any] | None = None) -> None:
    del source_loader
    raise AuthorizationError(
        f"{REAL_DATA_BLOCKER_CODE}: real-data paths are not resolved or opened; "
        "a versioned execution_kind-bound run contract and distinct APPROVED_TO_RUN are required"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP-034 canonical input contract preparation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-contract", "blocker-report"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default=DEFAULT_CONFIG)
    real = subparsers.add_parser("materialize-real-data")
    real.add_argument("--config", default=DEFAULT_CONFIG)
    real.add_argument("--source-manifest", required=True)
    real.add_argument("--output-root", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    real_source_loader: Callable[..., Any] | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "materialize-real-data":
        try:
            refuse_real_materialization(source_loader=real_source_loader)
        except AuthorizationError as exc:
            print(str(exc), file=sys.stderr)
            return 3
        raise AssertionError("unreachable")
    try:
        bundle = load_and_verify_contract(args.config)
        if args.command == "validate-contract":
            print(
                json.dumps(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "variant": VARIANT,
                        "proposal_digest": PROPOSAL_DIGEST,
                        "runner_schema_columns": len(bundle.runner_columns),
                        "feature_count": len(bundle.ordered_features),
                        "numeric_count": len(bundle.numeric_features),
                        "categorical_count": len(bundle.categorical_features),
                        "real_data_cli": "fail_closed",
                        "formal_buy": False,
                        "send_order": False,
                        "stake": 0,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(real_data_blocker_report(bundle), ensure_ascii=False, sort_keys=True))
    except ContractError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
