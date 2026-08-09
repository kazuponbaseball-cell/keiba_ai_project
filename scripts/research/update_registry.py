from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import subprocess
import sys
import uuid
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
    verify_run_materials,
)
from github_approval import (  # noqa: E402
    COMMENT_EVIDENCE_FIELDS,
    DEFAULT_BASE_BRANCH,
    DEFAULT_REPOSITORY,
    GitHubApprovalProvider,
    GitHubRestApprovalProvider,
    reverify_same_approval as reverify_github_approval,
    utc_now,
    verify_approval_comment,
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
PROHIBITED_STATUS_HINT = "production/merge/BUY approval is outside this registry's authority"


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_status(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def resolve_from_root(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


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


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
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
            "actor strings cannot grant approval."
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
    queue_path = (
        resolve_from_root(root, args.queue_file)
        if args.queue_file is not None
        else root / "research" / "queue" / f"{experiment_id}.json"
    )
    try:
        queue, proposal_scope, proposal_digest, components, total = load_queue(
            queue_path, root, experiment_id
        )
        events = load_events(registry_path)
        validate_approval_grant_history(events)
    except ValueError as exc:
        parser.error(str(exc))

    history = [event for event in events if event["experiment_id"] == experiment_id]
    previous = history[-1] if history else None
    previous_status = normalize_status(str(previous["status"])) if previous else None
    if previous_status not in TRANSITIONS:
        parser.error(f"registry has unknown previous state: {previous_status!r}")
    if status not in TRANSITIONS[previous_status]:
        parser.error(
            f"invalid transition for {experiment_id}: {previous_status or '<none>'} -> {status}"
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

    provider = approval_provider or GitHubRestApprovalProvider()
    base_commit = proposal_scope["base_commit"]
    allowlist: dict[str, Any] | None = None
    github_trust_evidence: dict[str, Any] | None = None
    if status in GITHUB_APPROVAL_CHECK_STATUSES:
        try:
            allowlist, github_trust_evidence = verify_github_trust(
                provider=provider,
                repository=args.github_repository,
                base_branch=args.github_base_branch,
                base_commit=base_commit,
            )
        except ValueError as exc:
            parser.error(str(exc))
    approval_evidence: dict[str, Any] | None = None
    revalidated_approval_evidence: list[dict[str, Any]] = []
    run_scope: dict[str, Any] | None = None
    run_scope_digest: str | None = None
    run_scope_path: Path | None = None
    review_digest: str | None = None

    if status in {"run_approval_required", "approved_to_run", "running"}:
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
        try:
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
            verify_run_materials(root, run_scope)
            allowed_dirty_paths = {
                "research/REGISTRY.jsonl",
                queue.get("proposal_scope_file", ""),
                queue.get("experiment_markdown", ""),
                str(queue_path.relative_to(root).as_posix()),
                str(run_scope_path.relative_to(root).as_posix()),
            }
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
        parser.error("RUNNING requires --execution-kind synthetic or real-data")

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
    event: dict[str, Any] = {
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
        "human_approved": approval_evidence is not None or bool(revalidated_approval_evidence),
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
    if args.dry_run:
        print(
            json.dumps(
                {"dry_run": True, "registry": str(registry_path), "event": event},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        event,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    with registry_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
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
