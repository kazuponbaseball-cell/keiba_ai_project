from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scope_contract import (  # noqa: E402
    FULL_SHA256,
    RUN_SCORE_THRESHOLD,
    SAFE_EXPERIMENT_ID,
    SCORE_WEIGHTS,
    canonical_digest,
    load_frozen_proposal,
    load_frozen_run_scope,
    proposal_score_total,
    strict_json_load,
    strict_json_loads,
    verify_run_materials,
)
from infrastructure_safety_contract import (  # noqa: E402
    EVENT_SCHEMA_VERSION as INFRASTRUCTURE_EVENT_SCHEMA_VERSION,
    GATE_KIND as INFRASTRUCTURE_GATE_KIND,
    load_gate_policy,
    normalize_gate_policy,
    normalize_infrastructure_event,
    normalize_infrastructure_proposal,
    normalize_infrastructure_queue,
    normalize_infrastructure_run_scope,
    reject_linked_repository_path,
    strict_json_load as strict_infrastructure_json_load,
    strict_json_loads as strict_infrastructure_json_loads,
    verify_infrastructure_commit_diff,
    verify_infrastructure_run_materials,
)
from github_approval import (  # noqa: E402
    COMMENT_EVIDENCE_FIELDS,
    DEFAULT_BASE_BRANCH,
    DEFAULT_REPOSITORY,
    GitHubApprovalProvider,
    GitHubRestApprovalProvider,
    fetch_github_file_at_commit,
    reverify_same_approval as reverify_github_approval,
    utc_now,
    verify_approval_comment,
    verify_github_file_at_commit,
    verify_github_main_head_unchanged,
    verify_github_trust,
)


ALLOWED_STATUSES = {
    "blocked_score",
    "proposed",
    "approved_to_prepare",
    "preparing",
    "run_approval_required",
    "approved_to_run",
    "running",
    "review_required",
    "rejected",
    "approved_for_shadow",
    "invalid",
}
TRANSITIONS: dict[str | None, set[str]] = {
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
APPROVAL_STATUS_TO_KEYWORD = {
    "approved_to_prepare": "APPROVED_TO_PREPARE",
    "approved_to_run": "APPROVED_TO_RUN",
    "approved_for_shadow": "APPROVED_FOR_SHADOW",
}
APPROVAL_GRANT_STATUSES = frozenset(APPROVAL_STATUS_TO_KEYWORD)
PRIOR_APPROVAL_GRANTS_BY_STATUS = {
    "preparing": ("approved_to_prepare",),
    "approved_to_run": ("approved_to_prepare",),
    "running": ("approved_to_prepare", "approved_to_run"),
    "approved_for_shadow": ("approved_to_prepare", "approved_to_run"),
}
GITHUB_APPROVAL_CHECK_STATUSES = APPROVAL_GRANT_STATUSES | frozenset(
    PRIOR_APPROVAL_GRANTS_BY_STATUS
)
MAIN_REGISTRY_CHECK_STATUSES = frozenset(ALLOWED_STATUSES)
PROHIBITED_STATUS_HINT = "production/merge/BUY approval is outside this registry's authority"


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_status(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def resolve_from_root(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


@contextmanager
def exclusive_registry_handle(path: Path):
    """Hold an exclusive process lock for a compare-and-append operation."""

    handle = path.open("a+b")
    lock_length = max(1, os.fstat(handle.fileno()).st_size)
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, lock_length)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise ValueError("research registry is locked by another writer; retry") from exc
    try:
        yield handle
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, lock_length)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def current_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot resolve execution commit: {completed.stderr.strip()}")
    return completed.stdout.strip().lower()


def verify_execution_worktree(root: Path, allowed_dirty_paths: set[str]) -> None:
    """Reject uncommitted code or undeclared files outside hash-bound artifacts."""

    completed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot inspect execution worktree; fail-close: {completed.stderr.strip()}")
    unexpected: list[str] = []
    for line in completed.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        normalized = path.replace("\\", "/")
        if normalized not in allowed_dirty_paths:
            unexpected.append(normalized)
    if unexpected:
        raise ValueError(
            "execution worktree contains uncommitted or untracked paths outside "
            "hash-bound lifecycle artifacts: " + ", ".join(sorted(unexpected))
        )


def load_queue(
    path: Path,
    root: Path,
    experiment_id: str,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, int], int]:
    if not path.is_file():
        raise ValueError(f"queue record not found: {path}")
    payload = strict_json_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"queue record must be a JSON object: {path}")
    if payload.get("schema_version") != 2:
        raise ValueError("queue schema_version must be 2")
    if payload.get("experiment_id") != experiment_id:
        raise ValueError(
            f"queue experiment_id mismatch: expected {experiment_id!r}, "
            f"found {payload.get('experiment_id')!r}"
        )
    for flag in (
        "human_approved_to_prepare",
        "human_approved_to_run",
        "human_approved_for_shadow",
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
        if payload.get(flag) is not False:
            raise ValueError(f"queue safety flag {flag!r} must remain false")
    if payload.get("stake") != 0 or isinstance(payload.get("stake"), bool):
        raise ValueError("queue safety field 'stake' must remain 0")

    proposal_scope, proposal_digest, _ = load_frozen_proposal(root, payload, experiment_id)
    components = proposal_scope["score_components"]
    total = proposal_score_total(proposal_scope)
    score = payload.get("score")
    if not isinstance(score, dict):
        raise ValueError("queue is missing score")
    if score.get("components") != components or score.get("total") != total:
        raise ValueError("queue score differs from canonical proposal scope")
    return payload, proposal_scope, proposal_digest, components, total


def load_queue_context(path: Path, root: Path, experiment_id: str) -> dict[str, Any]:
    """Load legacy ROI queue v2 or explicit infrastructure queue v3."""

    if not path.is_file():
        raise ValueError(f"queue record not found: {path}")
    payload = strict_json_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"queue record must be a JSON object: {path}")
    schema_version = payload.get("schema_version")
    if schema_version == 2:
        queue, proposal, digest, components, total = load_queue(
            path, root, experiment_id
        )
        return {
            "gate_kind": "roi_research_v1",
            "queue": queue,
            "proposal_scope": proposal,
            "proposal_scope_digest": digest,
            "score_components": components,
            "score_total": total,
            "score_threshold": RUN_SCORE_THRESHOLD,
            "policy": None,
            "policy_sha256": None,
        }
    if schema_version != 3:
        raise ValueError("queue schema_version must be 2 or 3")
    if payload.get("gate_kind") != INFRASTRUCTURE_GATE_KIND:
        raise ValueError("queue schema v3 is reserved for infrastructure_safety_v1")
    expected_queue_path = root / "research" / "queue" / f"{experiment_id}.json"
    if path.resolve() != expected_queue_path.resolve():
        raise ValueError(
            "infrastructure queue must use its code-owned lifecycle path: "
            f"{expected_queue_path.relative_to(root).as_posix()}"
        )

    policy_path = root / "research" / "INFRASTRUCTURE_GATE.json"
    policy, policy_sha256 = load_gate_policy(policy_path)
    queue = normalize_infrastructure_queue(
        payload,
        policy=policy,
        policy_sha256=policy_sha256,
    )
    if queue["experiment_id"] != experiment_id:
        raise ValueError(
            f"queue experiment_id mismatch: expected {experiment_id!r}, "
            f"found {queue['experiment_id']!r}"
        )
    proposal_path = resolve_from_root(root, Path(queue["proposal_scope_file"]))
    proposal = normalize_infrastructure_proposal(
        strict_infrastructure_json_load(proposal_path),
        policy=policy,
        policy_sha256=policy_sha256,
        expected_experiment_id=experiment_id,
    )
    if proposal != queue["proposal_scope"]:
        raise ValueError("queue proposal_scope differs from canonical infrastructure scope")
    digest = canonical_digest(proposal)
    if digest != queue["proposal_scope_digest"]:
        raise ValueError("infrastructure proposal scope changed after queue creation")
    return {
        "gate_kind": INFRASTRUCTURE_GATE_KIND,
        "queue": queue,
        "proposal_scope": proposal,
        "proposal_scope_digest": digest,
        "score_components": None,
        "score_total": None,
        "score_threshold": None,
        "policy": policy,
        "policy_sha256": policy_sha256,
    }


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = strict_json_loads(raw, source=f"{path}:{line_number}")
            except ValueError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"registry event at {path}:{line_number} is not an object")
            if not isinstance(event.get("experiment_id"), str) or not isinstance(
                event.get("status"), str
            ):
                raise ValueError(
                    f"registry event at {path}:{line_number} lacks experiment_id/status"
                )
            events.append(event)
    return events


def _canonical_registry_bytes(content: bytes, *, label: str) -> bytes:
    normalized = content.replace(b"\r\n", b"\n")
    if b"\r" in normalized:
        raise ValueError(f"{label} contains a bare carriage return; fail-close")
    if normalized and not normalized.endswith(b"\n"):
        raise ValueError(f"{label} is not newline-terminated JSONL; fail-close")
    return normalized


def verify_current_main_registry_exact(
    *,
    remote_content: bytes,
    local_snapshot: bytes,
) -> None:
    remote = _canonical_registry_bytes(
        remote_content,
        label="GitHub current-main registry",
    )
    local = _canonical_registry_bytes(
        local_snapshot,
        label="local research registry",
    )
    if local != remote:
        raise ValueError(
            "local research registry must exactly equal the GitHub current-main "
            "registry before creating one pending transition; refresh main and retry"
        )


def validate_registry_event_chains(events: list[dict[str, Any]]) -> None:
    """Reject duplicate identities and broken per-experiment append-only chains."""

    seen_event_ids: set[str] = set()
    latest_by_experiment: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events, start=1):
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError(f"registry event {index} has an invalid event_id")
        if event_id in seen_event_ids:
            raise ValueError(f"registry contains duplicate event_id {event_id!r}")
        seen_event_ids.add(event_id)

        experiment_id = event["experiment_id"]
        sequence = event.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise ValueError(f"registry event {event_id} has an invalid sequence")
        previous = latest_by_experiment.get(experiment_id)
        status = normalize_status(str(event.get("status", "")))
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"registry event {event_id} has an unknown status")
        if previous is None:
            if sequence != 1:
                raise ValueError(f"registry history for {experiment_id} must start at sequence 1")
            if event.get("previous_event_id") is not None or event.get("previous_status") is not None:
                raise ValueError(f"first registry event for {experiment_id} must not have a predecessor")
            if status not in TRANSITIONS[None]:
                raise ValueError(
                    f"registry history for {experiment_id} starts with an invalid status"
                )
        else:
            if sequence != previous["sequence"] + 1:
                raise ValueError(f"registry sequence is not contiguous for {experiment_id}")
            if event.get("previous_event_id") != previous["event_id"]:
                raise ValueError(f"registry previous_event_id chain is broken for {experiment_id}")
            if normalize_status(str(event.get("previous_status", ""))) != normalize_status(
                str(previous["status"])
            ):
                raise ValueError(f"registry previous_status chain is broken for {experiment_id}")
            previous_status = normalize_status(str(previous["status"]))
            if previous_status not in TRANSITIONS or status not in TRANSITIONS[previous_status]:
                raise ValueError(
                    f"registry contains an invalid historical transition for {experiment_id}: "
                    f"{previous_status} -> {status}"
                )
        latest_by_experiment[experiment_id] = event


def contains_unbound_legacy_real_data_running(events: list[dict[str, Any]]) -> bool:
    """Detect v2 real-data RUNNING history that predates execution-kind binding."""

    return any(
        event.get("schema_version") == 2
        and normalize_status(str(event.get("status", ""))) == "running"
        and event.get("execution_kind") == "real-data"
        for event in events
    )


def validate_approval_grant_history(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return globally consumed approval comment IDs, rejecting ambiguous history."""

    consumed: dict[int, dict[str, Any]] = {}
    for event in events:
        status = normalize_status(str(event.get("status", "")))
        if status not in APPROVAL_GRANT_STATUSES:
            continue
        evidence = event.get("approval_evidence")
        if not isinstance(evidence, dict) or set(evidence) != set(COMMENT_EVIDENCE_FIELDS):
            raise ValueError(
                f"registry {status} grant is missing or has malformed approval_evidence; "
                "fail-close"
            )
        comment_id = evidence.get("comment_id")
        if isinstance(comment_id, bool) or not isinstance(comment_id, int) or comment_id <= 0:
            raise ValueError(
                f"registry {status} grant has an invalid approval comment ID; fail-close"
            )
        expected_keyword = APPROVAL_STATUS_TO_KEYWORD[status]
        if evidence.get("approval_type") != expected_keyword:
            raise ValueError(
                f"registry {status} grant has mismatched approval evidence; fail-close"
            )
        approval_digest = evidence.get("approval_digest")
        event_digest_field = {
            "approved_to_prepare": "proposal_scope_digest",
            "approved_to_run": "run_scope_digest",
            "approved_for_shadow": "review_digest",
        }[status]
        if approval_digest != event.get(event_digest_field):
            raise ValueError(
                f"registry {status} grant digest is not bound to its event; fail-close"
            )
        body = evidence.get("body")
        body_sha256 = evidence.get("body_sha256")
        if (
            not isinstance(approval_digest, str)
            or not FULL_SHA256.fullmatch(approval_digest)
            or body != f"{expected_keyword} {approval_digest}"
            or body_sha256 != hashlib.sha256(body.encode("utf-8")).hexdigest()
        ):
            raise ValueError(
                f"registry {status} grant has malformed approval evidence; fail-close"
            )
        prior = consumed.get(comment_id)
        if prior is not None:
            raise ValueError(
                "registry contains a reused approval comment ID "
                f"{comment_id} in {prior.get('experiment_id')!r} and "
                f"{event.get('experiment_id')!r}; fail-close"
            )
        consumed[comment_id] = event
    return consumed


def ensure_approval_comment_id_unused(
    events: list[dict[str, Any]],
    comment_id: int,
) -> None:
    if isinstance(comment_id, bool) or not isinstance(comment_id, int) or comment_id <= 0:
        raise ValueError("approval comment ID must be a positive integer")
    consumed = validate_approval_grant_history(events)
    if comment_id in consumed:
        prior = consumed[comment_id]
        raise ValueError(
            "approval comment ID "
            f"{comment_id} was already used by an approval grant for "
            f"{prior.get('experiment_id')!r}; fail-close"
        )


def find_unique_approval_grant(
    history: list[dict[str, Any]],
    grant_status: str,
) -> dict[str, Any]:
    matches = [
        event
        for event in history
        if normalize_status(str(event.get("status", ""))) == grant_status
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {grant_status.upper()} grant; found {len(matches)}; fail-close"
        )
    return matches[0]


def reverify_prior_approvals(
    *,
    transition_status: str,
    history: list[dict[str, Any]],
    provider: GitHubApprovalProvider,
    allowlist: dict[str, Any],
    proposal_digest: str,
    current_run_scope_digest: str | None,
) -> list[dict[str, Any]]:
    reverified: list[dict[str, Any]] = []
    for grant_status in PRIOR_APPROVAL_GRANTS_BY_STATUS.get(transition_status, ()):
        grant = find_unique_approval_grant(history, grant_status)
        if grant_status == "approved_to_prepare":
            expected_digest = proposal_digest
        else:
            expected_digest = grant.get("run_scope_digest")
            if not isinstance(expected_digest, str) or not FULL_SHA256.fullmatch(
                expected_digest
            ):
                raise ValueError(
                    "stored APPROVED_TO_RUN grant has an invalid run scope digest; fail-close"
                )
            if (
                current_run_scope_digest is not None
                and expected_digest != current_run_scope_digest
            ):
                raise ValueError("run scope digest differs from APPROVED_TO_RUN")
        evidence = grant.get("approval_evidence")
        if not isinstance(evidence, dict) or set(evidence) != set(
            COMMENT_EVIDENCE_FIELDS
        ):
            raise ValueError("stored approval evidence is missing or invalid; fail-close")
        if evidence.get("approval_type") != APPROVAL_STATUS_TO_KEYWORD[grant_status]:
            raise ValueError("stored approval keyword differs from the required grant; fail-close")
        if evidence.get("approval_digest") != expected_digest:
            raise ValueError("stored approval digest differs from the required scope; fail-close")
        reverified.append(
            reverify_github_approval(
                evidence=evidence,
                provider=provider,
                allowlist=allowlist,
                expected_approval_type=APPROVAL_STATUS_TO_KEYWORD[grant_status],
                expected_approval_digest=expected_digest,
            )
        )
    return reverified


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append one fail-closed Research OS lifecycle event. Human approval "
            "comes only from a verified GitHub Issue comment; caller flags and "
            "actor strings cannot grant approval. Infrastructure schema-v3 output "
            "is pending evidence only and never grants preparation or execution authority."
        )
    )
    parser.add_argument("experiment_id", help="Existing experiment queue identifier.")
    parser.add_argument("status", help="New lifecycle state.")
    parser.add_argument(
        "--human-approved",
        action="store_true",
        help="Deprecated and rejected; GitHub approval evidence is mandatory.",
    )
    parser.add_argument(
        "--actor",
        default=getpass.getuser(),
        help="Audit caller only; never proof of human identity.",
    )
    parser.add_argument("--github-repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--github-base-branch", default=DEFAULT_BASE_BRANCH)
    parser.add_argument("--issue-number", type=int, default=None)
    parser.add_argument("--approval-comment-id", type=int, default=None)
    parser.add_argument(
        "--review-digest",
        default=None,
        help="Immutable review/result digest required for APPROVED_FOR_SHADOW.",
    )
    parser.add_argument(
        "--run-scope-file",
        type=Path,
        default=None,
        help="Canonical run scope; defaults to research/scopes/<id>.run.json.",
    )
    parser.add_argument(
        "--execution-kind",
        choices=("none", "synthetic", "real-data"),
        default="none",
        help="Declared work kind. Real data is accepted only at RUNNING.",
    )
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--notes", default="")
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("research/REGISTRY.jsonl"),
    )
    parser.add_argument("--queue-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    approval_provider: GitHubApprovalProvider | None = None,
    execution_commit_provider: Callable[[Path], str] = current_commit,
    execution_worktree_verifier: Callable[[Path, set[str]], None] = verify_execution_worktree,
    infrastructure_diff_verifier: Callable[[Path, dict[str, Any]], Any] = (
        verify_infrastructure_commit_diff
    ),
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    experiment_id = args.experiment_id.strip()
    if not SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
        parser.error("experiment_id is not a safe 3-64 character identifier")
    raw_status = args.status.strip()
    status = normalize_status(raw_status)
    if status not in ALLOWED_STATUSES:
        parser.error(
            f"unsupported status {raw_status!r}; allowed: {', '.join(sorted(ALLOWED_STATUSES))}. "
            f"{PROHIBITED_STATUS_HINT}"
        )
    if args.human_approved:
        parser.error(
            "--human-approved cannot grant approval; verified GitHub Issue evidence is required"
        )
    if args.execution_kind == "real-data" and status != "running":
        parser.error("real-data execution is forbidden before RUNNING")
    actor = args.actor.strip()
    if not actor:
        parser.error("--actor must not be blank")
    artifacts = [item.strip() for item in args.artifact]
    if any(not item for item in artifacts):
        parser.error("--artifact values must not be blank")

    root = args.root.resolve()
    registry_path = resolve_from_root(root, args.registry)
    expected_registry_path = root / "research" / "REGISTRY.jsonl"
    try:
        reject_linked_repository_path(root, registry_path, "research registry")
    except ValueError as exc:
        parser.error(str(exc))
    if registry_path.absolute() != expected_registry_path.absolute():
        parser.error("--registry must be the code-owned research/REGISTRY.jsonl ledger")
    try:
        registry_snapshot = registry_path.read_bytes() if registry_path.exists() else b""
    except OSError as exc:
        parser.error(f"cannot snapshot research registry: {exc}")
    queue_path = (
        resolve_from_root(root, args.queue_file)
        if args.queue_file is not None
        else root / "research" / "queue" / f"{experiment_id}.json"
    )
    try:
        queue_context = load_queue_context(queue_path, root, experiment_id)
        queue = queue_context["queue"]
        proposal_scope = queue_context["proposal_scope"]
        proposal_digest = queue_context["proposal_scope_digest"]
        components = queue_context["score_components"]
        total = queue_context["score_total"]
        gate_kind = queue_context["gate_kind"]
        infrastructure_policy = queue_context["policy"]
        infrastructure_policy_sha256 = queue_context["policy_sha256"]
        events = load_events(registry_path)
        validate_registry_event_chains(events)
        validate_approval_grant_history(events)
    except ValueError as exc:
        parser.error(str(exc))

    history = [event for event in events if event["experiment_id"] == experiment_id]
    infrastructure_gate = gate_kind == INFRASTRUCTURE_GATE_KIND
    if infrastructure_gate:
        assert infrastructure_policy is not None
        try:
            history = [
                normalize_infrastructure_event(event, policy=infrastructure_policy)
                for event in history
            ]
        except ValueError as exc:
            parser.error(str(exc))
        historical_policy_evidence = [
            event["gate_policy_evidence"]
            for event in history
            if event["gate_policy_evidence"] is not None
        ]
        if any(
            evidence["ref"] != proposal_scope["base_commit"]
            or evidence["content_sha256"]
            != proposal_scope["gate_policy"]["sha256"]
            for evidence in historical_policy_evidence
        ):
            parser.error(
                "infrastructure gate policy evidence is not bound to the proposal base/hash"
            )
        if historical_policy_evidence and any(
            evidence != historical_policy_evidence[0]
            for evidence in historical_policy_evidence[1:]
        ):
            parser.error(
                "infrastructure gate policy evidence changed within the registry chain"
            )
        inherited_gate_policy_evidence = (
            historical_policy_evidence[-1] if historical_policy_evidence else None
        )
    elif any(event.get("gate_kind") == INFRASTRUCTURE_GATE_KIND for event in history):
        parser.error("registry gate profile differs from the queue profile")
    else:
        inherited_gate_policy_evidence = None
    inherited_main_registry_evidence = next(
        (
            event.get("main_registry_evidence")
            for event in reversed(history)
            if isinstance(event.get("main_registry_evidence"), dict)
        ),
        None,
    )
    if (
        not infrastructure_gate
        and contains_unbound_legacy_real_data_running(history)
        and status != "invalid"
    ):
        parser.error(
            "legacy ROI history contains unbound real-data RUNNING; "
            "only INVALID is permitted"
        )
    previous = history[-1] if history else None
    previous_status = normalize_status(str(previous["status"])) if previous else None
    if previous_status not in TRANSITIONS:
        parser.error(f"registry has unknown previous state: {previous_status!r}")
    if status not in TRANSITIONS[previous_status]:
        parser.error(
            f"invalid transition for {experiment_id}: {previous_status or '<none>'} -> {status}"
        )

    if infrastructure_gate:
        if status == "blocked_score":
            parser.error("infrastructure safety proposals use hard gates, not BLOCKED_SCORE")
        if status == "approved_for_shadow":
            parser.error("infrastructure safety lifecycle does not support shadow approval")
        if args.execution_kind == "real-data":
            parser.error("infrastructure safety lifecycle is synthetic-only")
        if status != "running" and args.execution_kind != "none":
            parser.error(
                "infrastructure execution kind is none outside the synthetic RUNNING state"
            )
        threshold_met = True
    else:
        assert isinstance(total, int)
        if status == "running" and args.execution_kind == "real-data":
            parser.error(
                "legacy ROI run scope does not bind execution_kind; real-data "
                "execution is fail-closed until a versioned run contract binds it"
            )
        threshold_met = total >= RUN_SCORE_THRESHOLD
        if status == "blocked_score" and threshold_met:
            parser.error("BLOCKED_SCORE is invalid because score meets the threshold")
        if status == "proposed" and not threshold_met:
            parser.error("PROPOSED is invalid because score is below the threshold")
        if status not in {"blocked_score", "invalid"} and not threshold_met:
            parser.error(f"{status.upper()} requires score >= {RUN_SCORE_THRESHOLD}")

    prior_scope_digests = {
        event.get("proposal_scope_digest")
        for event in history
        if event.get("proposal_scope_digest") is not None
    }
    if status != "invalid" and prior_scope_digests and prior_scope_digests != {proposal_digest}:
        parser.error("proposal scope changed after approval; use a new experiment ID or reapproval")

    if status in APPROVAL_GRANT_STATUSES:
        missing_evidence_messages = {
            "approved_to_prepare": "APPROVED_TO_PREPARE requires GitHub Issue comment evidence",
            "approved_to_run": "APPROVED_TO_RUN requires GitHub Issue comment evidence",
            "approved_for_shadow": (
                "APPROVED_FOR_SHADOW requires separate GitHub Issue comment evidence"
            ),
        }
        if args.issue_number is None or args.approval_comment_id is None:
            parser.error(missing_evidence_messages[status])
        try:
            ensure_approval_comment_id_unused(events, args.approval_comment_id)
        except ValueError as exc:
            parser.error(str(exc))

    provider = approval_provider or GitHubRestApprovalProvider()
    base_commit = proposal_scope["base_commit"]
    allowlist: dict[str, Any] | None = None
    github_trust_evidence: dict[str, Any] | None = None
    gate_policy_evidence: dict[str, Any] | None = inherited_gate_policy_evidence
    main_registry_evidence: dict[str, Any] | None = (
        inherited_main_registry_evidence
    )
    if status in MAIN_REGISTRY_CHECK_STATUSES:
        try:
            allowlist, github_trust_evidence = verify_github_trust(
                provider=provider,
                repository=args.github_repository,
                base_branch=args.github_base_branch,
                base_commit=base_commit,
            )
            remote_registry, main_registry_evidence = fetch_github_file_at_commit(
                provider=provider,
                repository=args.github_repository,
                path="research/REGISTRY.jsonl",
                ref=github_trust_evidence["verified_current_main_sha"],
            )
            verify_current_main_registry_exact(
                remote_content=remote_registry,
                local_snapshot=registry_snapshot,
            )
            if infrastructure_gate and status in GITHUB_APPROVAL_CHECK_STATUSES:
                assert infrastructure_policy is not None
                policy_content, refreshed_gate_policy_evidence = (
                    verify_github_file_at_commit(
                        provider=provider,
                        repository=args.github_repository,
                        path="research/INFRASTRUCTURE_GATE.json",
                        ref=base_commit,
                        expected_content_sha256=proposal_scope["gate_policy"]["sha256"],
                    )
                )
                if (
                    gate_policy_evidence is not None
                    and refreshed_gate_policy_evidence != gate_policy_evidence
                ):
                    raise ValueError(
                        "GitHub base-commit infrastructure gate policy evidence changed; "
                        "fail-close"
                    )
                gate_policy_evidence = refreshed_gate_policy_evidence
                try:
                    remote_policy = normalize_gate_policy(
                        strict_infrastructure_json_loads(
                            policy_content.decode("utf-8"),
                            label="GitHub base-commit infrastructure gate policy",
                        )
                    )
                except (UnicodeDecodeError, ValueError) as exc:
                    raise ValueError(
                        "GitHub base-commit infrastructure gate policy is invalid; fail-close"
                    ) from exc
                if remote_policy != infrastructure_policy:
                    raise ValueError(
                        "GitHub base-commit infrastructure gate policy differs from local "
                        "hash-bound policy; fail-close"
                    )
        except ValueError as exc:
            parser.error(str(exc))
    approval_evidence: dict[str, Any] | None = None
    revalidated_approval_evidence: list[dict[str, Any]] = []
    run_scope: dict[str, Any] | None = None
    run_scope_digest: str | None = None
    run_scope_path: Path | None = None
    review_digest: str | None = None

    run_scope_required = status in {
        "run_approval_required",
        "approved_to_run",
        "running",
    } or (
        infrastructure_gate and status in {"review_required", "rejected"}
    )
    if run_scope_required:
        path_from_history = next(
            (
                event.get("run_scope_file")
                for event in reversed(history)
                if isinstance(event.get("run_scope_file"), str)
            ),
            None,
        )
        run_scope_path = (
            resolve_from_root(root, args.run_scope_file)
            if args.run_scope_file is not None
            else (
                resolve_from_root(root, Path(path_from_history))
                if path_from_history
                else root / "research" / "scopes" / f"{experiment_id}.run.json"
            )
        )
        if infrastructure_gate:
            expected_run_scope_path = (
                root / "research" / "scopes" / f"{experiment_id}.run.json"
            )
            if run_scope_path.resolve() != expected_run_scope_path.resolve():
                parser.error(
                    "infrastructure run scope must use its code-owned lifecycle path: "
                    f"research/scopes/{experiment_id}.run.json"
                )
        try:
            if infrastructure_gate:
                assert infrastructure_policy is not None
                run_scope = normalize_infrastructure_run_scope(
                    strict_infrastructure_json_load(run_scope_path),
                    proposal_scope=proposal_scope,
                    policy=infrastructure_policy,
                    policy_sha256=infrastructure_policy_sha256,
                )
                run_scope_digest = canonical_digest(run_scope)
            else:
                run_scope, run_scope_digest = load_frozen_run_scope(
                    root, run_scope_path, proposal_scope
                )
            prior_run_digests = {
                event.get("run_scope_digest")
                for event in history
                if event.get("run_scope_digest") is not None
            }
            if prior_run_digests and prior_run_digests != {run_scope_digest}:
                raise ValueError(
                    "run scope changed after RUN_APPROVAL_REQUIRED; create a new ID or reapprove"
                )
            observed_commit = execution_commit_provider(root)
            if run_scope["execution_commit_sha"] != observed_commit:
                raise ValueError(
                    "execution commit SHA changed after run scope freeze: "
                    f"{run_scope['execution_commit_sha']} != {observed_commit}"
                )
            if infrastructure_gate:
                verify_infrastructure_run_materials(root, run_scope)
                infrastructure_diff_verifier(root, run_scope)
            else:
                verify_run_materials(root, run_scope)
            allowed_dirty_paths = {
                "research/REGISTRY.jsonl",
                queue.get("proposal_scope_file", ""),
                queue.get("experiment_markdown", ""),
                str(queue_path.relative_to(root).as_posix()),
                str(run_scope_path.relative_to(root).as_posix()),
            }
            if not infrastructure_gate:
                for field in ("config_hashes", "data_input_manifest_hashes"):
                    allowed_dirty_paths.update(item["path"] for item in run_scope[field])
                for field in (
                    "fold_manifest_hash",
                    "runner_universe_manifest_hash",
                    "dependency_environment_manifest",
                ):
                    allowed_dirty_paths.add(run_scope[field]["path"])
            execution_worktree_verifier(root, allowed_dirty_paths)
        except ValueError as exc:
            parser.error(str(exc))

    if status == "approved_for_shadow":
        review_digest = (args.review_digest or "").strip().lower()
        if not FULL_SHA256.fullmatch(review_digest):
            parser.error("APPROVED_FOR_SHADOW requires a full --review-digest")

    if status in PRIOR_APPROVAL_GRANTS_BY_STATUS:
        if allowlist is None:
            parser.error("GitHub approval allowlist is unavailable; fail-close")
        try:
            revalidated_approval_evidence = reverify_prior_approvals(
                transition_status=status,
                history=history,
                provider=provider,
                allowlist=allowlist,
                proposal_digest=proposal_digest,
                current_run_scope_digest=run_scope_digest if status == "running" else None,
            )
        except ValueError as exc:
            parser.error(str(exc))

    if status == "approved_to_prepare":
        try:
            approval_evidence = verify_approval_comment(
                provider=provider,
                allowlist=allowlist,
                repository=args.github_repository,
                issue_number=args.issue_number,
                comment_id=args.approval_comment_id,
                approval_keyword="APPROVED_TO_PREPARE",
                approval_digest=proposal_digest,
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif status == "approved_to_run":
        if run_scope_digest is None:
            parser.error("APPROVED_TO_RUN requires a frozen run scope")
        try:
            approval_evidence = verify_approval_comment(
                provider=provider,
                allowlist=allowlist,
                repository=args.github_repository,
                issue_number=args.issue_number,
                comment_id=args.approval_comment_id,
                approval_keyword="APPROVED_TO_RUN",
                approval_digest=run_scope_digest,
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif status == "approved_for_shadow":
        try:
            approval_evidence = verify_approval_comment(
                provider=provider,
                allowlist=allowlist,
                repository=args.github_repository,
                issue_number=args.issue_number,
                comment_id=args.approval_comment_id,
                approval_keyword="APPROVED_FOR_SHADOW",
                approval_digest=review_digest,
            )
        except ValueError as exc:
            parser.error(str(exc))

    if status == "running" and args.execution_kind == "none":
        parser.error("RUNNING requires --execution-kind synthetic")
    if (
        infrastructure_gate
        and status == "running"
        and (
            run_scope is None
            or args.execution_kind != run_scope.get("execution_kind")
            or args.execution_kind != "synthetic"
        )
    ):
        parser.error(
            "infrastructure RUNNING execution kind must match the frozen synthetic run scope"
        )

    prior_prepare = any(event["status"] == "approved_to_prepare" for event in history)
    prior_run = any(event["status"] == "approved_to_run" for event in history)
    prior_shadow = any(event["status"] == "approved_for_shadow" for event in history)
    prepare_recorded = prior_prepare or status == "approved_to_prepare"
    run_recorded = prior_run or status == "approved_to_run"
    shadow_recorded = prior_shadow or status == "approved_for_shadow"
    preparing_allowed = status in {
        "approved_to_prepare",
        "preparing",
        "run_approval_required",
    }
    running = status == "running"
    synthetic_running = running and args.execution_kind == "synthetic"
    real_data_running = running and args.execution_kind == "real-data"
    if infrastructure_gate:
        assert infrastructure_policy is not None
        event = normalize_infrastructure_event(
            {
                "schema_version": INFRASTRUCTURE_EVENT_SCHEMA_VERSION,
                "event_id": str(uuid.uuid4()),
                "sequence": len(history) + 1,
                "experiment_id": experiment_id,
                "gate_kind": INFRASTRUCTURE_GATE_KIND,
                "gate_contract_version": 1,
                "status": status,
                "previous_status": previous_status,
                "previous_event_id": previous.get("event_id") if previous else None,
                "occurred_at": utc_now(),
                "actor": actor,
                "gate_evaluation": queue["gate_evaluation"],
                "proposal_scope_digest": proposal_digest,
                "run_scope_digest": run_scope_digest,
                "github_trust_evidence": github_trust_evidence,
                "gate_policy_evidence": gate_policy_evidence,
                "main_registry_evidence": (
                    None if status == "proposed" else main_registry_evidence
                ),
                "approval_evidence": approval_evidence,
                "revalidated_approval_evidence": revalidated_approval_evidence,
                "human_approved": approval_evidence is not None
                or bool(revalidated_approval_evidence),
                "human_prepare_approval_recorded": prepare_recorded,
                "human_run_approval_recorded": run_recorded,
                # Schema-v3 events are local pending candidates until the exact
                # bytes are human-merged into GitHub current main.  No worktree
                # event grants preparation or execution authority by itself.
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
                "execution_kind": args.execution_kind,
                "artifacts": artifacts,
                "notes": args.notes.strip(),
                "queue_file": queue_path.relative_to(root).as_posix(),
                "run_scope_file": (
                    run_scope_path.relative_to(root).as_posix()
                    if run_scope_path is not None
                    else None
                ),
            },
            policy=infrastructure_policy,
        )
    else:
        event = {
            "schema_version": 2,
            "event_id": str(uuid.uuid4()),
            "sequence": len(history) + 1,
            "experiment_id": experiment_id,
            "status": status,
            "previous_status": previous_status,
            "previous_event_id": previous.get("event_id") if previous else None,
            "occurred_at": utc_now(),
            "actor": actor,
            "score_components": components,
            "score_total": total,
            "score_threshold": RUN_SCORE_THRESHOLD,
            "score_threshold_met": threshold_met,
            "proposal_scope_digest": proposal_digest,
            "github_trust_evidence": github_trust_evidence,
            "run_scope_digest": run_scope_digest,
            "review_digest": review_digest,
            "approval_evidence": approval_evidence,
            "revalidated_approval_evidence": revalidated_approval_evidence,
            "human_approved": approval_evidence is not None
            or bool(revalidated_approval_evidence),
            "human_prepare_approval_recorded": prepare_recorded,
            "human_run_approval_recorded": run_recorded,
            "human_shadow_approval_recorded": shadow_recorded,
            "preparation_authorized": preparing_allowed,
            "synthetic_fixture_tests_allowed": preparing_allowed or synthetic_running,
            "real_data_execution_allowed": real_data_running,
            "automatic_execution_allowed": running,
            "execution_authorized": running,
            "production_approved": False,
            "merge_approved": False,
            "buy_approved": False,
            "production_change_allowed": False,
            "merge_allowed": False,
            "buy_logic_change_allowed": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
            "execution_kind": args.execution_kind,
            "artifacts": artifacts,
            "notes": args.notes.strip(),
            "queue_file": str(queue_path),
            "run_scope_file": str(run_scope_path) if run_scope_path is not None else None,
        }
    line = (
        json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if github_trust_evidence is None:
        parser.error("GitHub current-main evidence is required for every transition")
    verified_current_main_sha = github_trust_evidence["verified_current_main_sha"]
    if args.dry_run:
        try:
            verify_github_main_head_unchanged(
                provider=provider,
                repository=args.github_repository,
                base_branch=args.github_base_branch,
                expected_sha=verified_current_main_sha,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {"dry_run": True, "registry": str(registry_path), "event": event},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with exclusive_registry_handle(registry_path) as handle:
            reject_linked_repository_path(root, registry_path, "research registry")
            handle.seek(0)
            if handle.read() != registry_snapshot:
                raise ValueError(
                    "research registry changed during verification; fail-close and retry"
                )
            verify_github_main_head_unchanged(
                provider=provider,
                repository=args.github_repository,
                base_branch=args.github_base_branch,
                expected_sha=verified_current_main_sha,
            )
            reject_linked_repository_path(root, registry_path, "research registry")
            handle.seek(0)
            if handle.read() != registry_snapshot:
                raise ValueError(
                    "research registry changed during GitHub head recheck; "
                    "fail-close and retry"
                )
            handle.seek(0, os.SEEK_END)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"appended": True, "registry": str(registry_path), "event": event},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.stderr.close()
        raise SystemExit(1)
