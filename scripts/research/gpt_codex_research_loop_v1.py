from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpt_strategy_contract_v1 import (  # noqa: E402
    canonical_digest,
    canonical_json_text,
    compile_proposal_scope,
    normalize_strategy,
    sha256_file,
    strict_json_load,
)


DISPATCH_SCHEMA_VERSION = 1
DISPATCH_FIELDS = {
    "schema_version",
    "packet_type",
    "experiment_id",
    "strategy_id",
    "strategy_digest",
    "brain_model_id",
    "brain_prompt_hash",
    "context_manifest_hash",
    "proposal_scope_digest",
    "approval_status",
    "approval_event_id",
    "approval_evidence_hash",
    "dispatch_mode",
    "tasks",
    "expected_changed_paths",
    "exact_execution_commands",
    "exact_execution_commands_hash",
    "allowed_operations",
    "forbidden_operations",
    "formal_buy",
    "send_order",
    "stake",
    "external_api_calls",
    "actual_codex_dispatch",
    "real_data_execution",
    "production_change",
}
PREPARATION_STATUSES = {
    "approved_to_prepare",
    "preparing",
    "run_approval_required",
}
FORBIDDEN_FEEDBACK_KEYS = {
    "api_key",
    "credential",
    "order_payload",
    "password",
    "secret",
    "token",
}


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _parse_json_object(text: str, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    try:
        payload = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=_strict_object_pairs,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _require_false(value: Any, field: str) -> None:
    if value is not False:
        raise ValueError(f"{field} must be false")


def _require_zero(value: Any, field: str) -> None:
    if value != 0 or isinstance(value, bool):
        raise ValueError(f"{field} must be 0")


def _reject_forbidden_feedback_keys(value: Any, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_FEEDBACK_KEYS:
                raise ValueError(f"{path} contains forbidden key: {key}")
            _reject_forbidden_feedback_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_feedback_keys(child, f"{path}[{index}]")


def read_registry_history(registry_path: Path, experiment_id: str) -> list[dict[str, Any]]:
    if not registry_path.is_file():
        raise ValueError(f"registry not found: {registry_path}")
    history: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        registry_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        event = _parse_json_object(raw_line, f"registry line {line_number}")
        if event.get("experiment_id") != experiment_id:
            continue
        expected_sequence = len(history) + 1
        if event.get("sequence") != expected_sequence:
            raise ValueError("registry sequence is not contiguous")
        expected_previous = history[-1].get("event_id") if history else None
        if event.get("previous_event_id") != expected_previous:
            raise ValueError("registry previous_event_id chain is invalid")
        _require_false(event.get("formal_buy"), "registry formal_buy")
        _require_false(event.get("send_order"), "registry send_order")
        _require_zero(event.get("stake"), "registry stake")
        history.append(event)
    if not history:
        raise ValueError(f"no registry history for {experiment_id}")
    return history


def _prepare_evidence(latest_event: dict[str, Any], proposal_digest: str) -> dict[str, Any]:
    evidence_candidates: list[dict[str, Any]] = []
    direct = latest_event.get("approval_evidence")
    if isinstance(direct, dict):
        evidence_candidates.append(direct)
    revalidated = latest_event.get("revalidated_approval_evidence")
    if isinstance(revalidated, list):
        evidence_candidates.extend(item for item in revalidated if isinstance(item, dict))
    for evidence in evidence_candidates:
        if (
            evidence.get("approval_type") == "APPROVED_TO_PREPARE"
            and evidence.get("approval_digest") == proposal_digest
        ):
            return evidence
    raise ValueError("latest registry event lacks matching preparation approval evidence")


def build_preparation_dispatch_packet(
    strategy_payload: dict[str, Any],
    *,
    registry_path: Path,
) -> dict[str, Any]:
    strategy = normalize_strategy(strategy_payload)
    proposal_scope = compile_proposal_scope(strategy)
    proposal_digest = canonical_digest(proposal_scope)
    history = read_registry_history(registry_path, proposal_scope["experiment_id"])
    latest = history[-1]
    if latest.get("status") not in PREPARATION_STATUSES:
        raise ValueError("Research OS preparation approval is not active")
    if latest.get("proposal_scope_digest") != proposal_digest:
        raise ValueError("strategy proposal digest differs from the approved proposal")
    if latest.get("preparation_authorized") is not True:
        raise ValueError("latest registry event does not authorize preparation")
    if latest.get("synthetic_fixture_tests_allowed") is not True:
        raise ValueError("latest registry event does not authorize synthetic fixtures")
    _require_false(
        latest.get("real_data_execution_allowed"),
        "registry real_data_execution_allowed",
    )
    _require_false(latest.get("automatic_execution_allowed"), "automatic execution")
    evidence = _prepare_evidence(latest, proposal_digest)

    exact_commands: list[str] = []
    packet = {
        "schema_version": DISPATCH_SCHEMA_VERSION,
        "packet_type": "codex_preparation_dispatch",
        "experiment_id": proposal_scope["experiment_id"],
        "strategy_id": strategy["strategy_id"],
        "strategy_digest": canonical_digest(strategy),
        "brain_model_id": strategy["brain_model_id"],
        "brain_prompt_hash": strategy["brain_prompt_hash"],
        "context_manifest_hash": strategy["context_manifest_hash"],
        "proposal_scope_digest": proposal_digest,
        "approval_status": latest["status"],
        "approval_event_id": latest["event_id"],
        "approval_evidence_hash": canonical_digest(evidence),
        "dispatch_mode": "synthetic_preparation_packet_only",
        "tasks": proposal_scope["in_scope"],
        "expected_changed_paths": proposal_scope["expected_changed_paths"],
        "exact_execution_commands": exact_commands,
        "exact_execution_commands_hash": canonical_digest(exact_commands),
        "allowed_operations": [
            "edit_expected_changed_paths",
            "run_pre_registered_synthetic_fixture_tests",
            "write_hash_bound_preparation_artifacts",
        ],
        "forbidden_operations": [
            "actual_codex_dispatch",
            "automatic_github_approval",
            "external_api_call",
            "merge",
            "production_change",
            "purchase_or_order",
            "read_real_race_odds_result_or_payout_data",
            "real_data_backtest_oos_or_roi_execution",
        ],
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "external_api_calls": False,
        "actual_codex_dispatch": False,
        "real_data_execution": False,
        "production_change": False,
    }
    canonical_json_text(packet)
    return packet


def normalize_dispatch_packet(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("dispatch packet must be an object")
    missing = sorted(DISPATCH_FIELDS - set(value))
    unexpected = sorted(set(value) - DISPATCH_FIELDS)
    if missing:
        raise ValueError(f"dispatch packet is missing field(s): {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"dispatch packet has unexpected field(s): {', '.join(unexpected)}")
    if value["schema_version"] != DISPATCH_SCHEMA_VERSION:
        raise ValueError("unsupported dispatch schema_version")
    if value["packet_type"] != "codex_preparation_dispatch":
        raise ValueError("unexpected dispatch packet_type")
    if value["dispatch_mode"] != "synthetic_preparation_packet_only":
        raise ValueError("dispatch mode is not preparation-only")
    if value["exact_execution_commands"] != []:
        raise ValueError("preparation packet must not contain execution commands")
    if value["exact_execution_commands_hash"] != canonical_digest([]):
        raise ValueError("execution command hash is invalid")
    for field in (
        "formal_buy",
        "send_order",
        "external_api_calls",
        "actual_codex_dispatch",
        "real_data_execution",
        "production_change",
    ):
        _require_false(value[field], f"dispatch {field}")
    _require_zero(value["stake"], "dispatch stake")
    canonical_json_text(value)
    return value


def build_result_feedback_packet(
    dispatch_payload: dict[str, Any],
    *,
    result_manifest: dict[str, Any],
    result_summary: dict[str, Any],
    review_prompt_path: Path,
) -> dict[str, Any]:
    dispatch = normalize_dispatch_packet(dispatch_payload)
    if not isinstance(result_manifest, dict) or not result_manifest:
        raise ValueError("result_manifest must be a non-empty object")
    if not isinstance(result_summary, dict) or not result_summary:
        raise ValueError("result_summary must be a non-empty object")
    _reject_forbidden_feedback_keys(result_manifest, "result_manifest")
    _reject_forbidden_feedback_keys(result_summary, "result_summary")
    if not review_prompt_path.is_file():
        raise ValueError(f"review prompt not found: {review_prompt_path}")
    feedback = {
        "schema_version": 1,
        "packet_type": "gpt_result_feedback",
        "experiment_id": dispatch["experiment_id"],
        "strategy_id": dispatch["strategy_id"],
        "strategy_digest": dispatch["strategy_digest"],
        "brain_model_id": dispatch["brain_model_id"],
        "brain_prompt_hash": dispatch["brain_prompt_hash"],
        "context_manifest_hash": dispatch["context_manifest_hash"],
        "proposal_scope_digest": dispatch["proposal_scope_digest"],
        "dispatch_packet_hash": canonical_digest(dispatch),
        "exact_execution_commands_hash": dispatch["exact_execution_commands_hash"],
        "result_manifest": result_manifest,
        "result_manifest_hash": canonical_digest(result_manifest),
        "result_summary": result_summary,
        "result_summary_hash": canonical_digest(result_summary),
        "review_prompt_hash": sha256_file(review_prompt_path),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    canonical_json_text(feedback)
    return feedback


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(canonical_json_text(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        _atomic_write_json(output.resolve(), payload)
    print(canonical_json_text(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic GPT-to-Codex research packets. This tool never calls "
            "an external API, launches Codex, reads race data, or reaches BUY paths."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-strategy")
    validate.add_argument("--strategy", type=Path, required=True)

    compile_parser = subparsers.add_parser("compile-proposal")
    compile_parser.add_argument("--strategy", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path)

    dispatch = subparsers.add_parser("prepare-dispatch")
    dispatch.add_argument("--strategy", type=Path, required=True)
    dispatch.add_argument("--registry", type=Path, required=True)
    dispatch.add_argument("--output", type=Path)

    feedback = subparsers.add_parser("build-feedback")
    feedback.add_argument("--dispatch", type=Path, required=True)
    feedback.add_argument("--result-manifest", type=Path, required=True)
    feedback.add_argument("--result-summary", type=Path, required=True)
    feedback.add_argument("--review-prompt", type=Path, required=True)
    feedback.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-strategy":
            strategy = normalize_strategy(strict_json_load(args.strategy.resolve()))
            _emit(
                {
                    "valid": True,
                    "strategy_id": strategy["strategy_id"],
                    "strategy_digest": canonical_digest(strategy),
                    "proposal_scope_digest": canonical_digest(strategy["proposal_scope"]),
                },
                None,
            )
        elif args.command == "compile-proposal":
            strategy = normalize_strategy(strict_json_load(args.strategy.resolve()))
            _emit(compile_proposal_scope(strategy), args.output)
        elif args.command == "prepare-dispatch":
            strategy = strict_json_load(args.strategy.resolve())
            packet = build_preparation_dispatch_packet(
                strategy,
                registry_path=args.registry.resolve(),
            )
            _emit(packet, args.output)
        elif args.command == "build-feedback":
            dispatch_payload = strict_json_load(args.dispatch.resolve())
            result_manifest = strict_json_load(args.result_manifest.resolve())
            result_summary = strict_json_load(args.result_summary.resolve())
            packet = build_result_feedback_packet(
                dispatch_payload,
                result_manifest=result_manifest,
                result_summary=result_summary,
                review_prompt_path=args.review_prompt.resolve(),
            )
            _emit(packet, args.output)
        else:
            parser.error("unsupported command")
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
