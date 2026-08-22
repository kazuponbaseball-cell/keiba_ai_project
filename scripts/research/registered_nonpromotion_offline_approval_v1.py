from __future__ import annotations

import hashlib
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import github_approval
from registered_nonpromotion_contract_v1 import (
    ContractError,
    canonical_digest,
    strict_json_loads,
)
from registered_nonpromotion_offline_contract_v1 import (
    APPROVAL_KEYWORD,
    DEFAULT_BASE_BRANCH,
    DEFAULT_REPOSITORY,
    EXPECTED_SCHEMA_PATHS,
    GATE_KIND,
    POLICY_RELATIVE_PATH,
    RUNTIME_MATERIAL_PATHS,
    SOURCE_RECIPE_RELATIVE_PATH,
    OfflineRegisteredRecipe,
    resolve_offline_registered_recipe,
    verify_canonical_offline_run_scope,
)


REGISTRY_PATH = "research/REGISTRY.jsonl"
INITIAL_CHECKPOINT = "INITIAL_APPROVAL"
REVERIFY_CHECKPOINTS = {
    "BEFORE_CANDIDATE_OPEN",
    "BEFORE_RESULT_PUBLISH",
}
ORDINARY_APPROVAL_TYPES = {
    "APPROVED_TO_PREPARE",
    "APPROVED_TO_RUN",
    "APPROVED_FOR_SHADOW",
}
AUTOMATION_LOGIN_PATTERNS = {
    "automation",
    "bot",
    "codex",
    "dependabot",
    "github-actions",
}
LIMITATIONS = {
    "single_use_policy": "ONE_ACCEPTED_EXECUTION",
    "single_use_enforcement": "BEST_EFFORT_LOCAL_EXCLUSIVE_RECEIPT",
    "global_replay_proof": False,
    "rollback_resistant": False,
    "durable_remote_ledger": False,
    "network_isolation": "APPLICATION_LEVEL_NOT_OS_SANDBOX",
}
EVIDENCE_FIELDS = {
    "schema_version",
    "gate_kind",
    "run_scope_digest",
    "verification_checkpoint",
    "original_evidence_digest",
    "github_trust",
    "runtime_materials",
    "ordinary_registry",
    "comment",
    "limitations",
    "implementation_current_main_ancestry_verified",
    "authority",
    "local_offline_permission",
    "global_uniqueness_guaranteed",
    "formal_buy",
    "send_order",
    "stake",
    "evidence_digest",
}
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"
)


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not UTC_TIMESTAMP.fullmatch(value):
        raise ContractError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError(f"{label} must be UTC")
    return parsed


def _clock_from(
    now: str | Callable[[], str] | None,
) -> Callable[[], str]:
    if now is None:
        return github_approval.utc_now
    if isinstance(now, str):
        _parse_utc(now, label="verification time")
        return lambda: now
    if callable(now):
        return now
    raise ContractError("now must be an ISO-8601 UTC string, callable, or None")


def _current_main_sha(
    *,
    provider: github_approval.GitHubApprovalProvider,
) -> str:
    try:
        payload = provider.get_branch_ref(DEFAULT_REPOSITORY, DEFAULT_BASE_BRANCH)
    except Exception as exc:
        raise ContractError(f"GitHub main verification unavailable; fail-close: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("ref") != "refs/heads/main":
        raise ContractError("GitHub main ref mismatch; fail-close")
    ref_object = payload.get("object")
    if not isinstance(ref_object, dict) or ref_object.get("type") != "commit":
        raise ContractError("GitHub main ref does not point to a commit; fail-close")
    sha = ref_object.get("sha")
    if not isinstance(sha, str) or not FULL_GIT_SHA.fullmatch(sha):
        raise ContractError("GitHub current main SHA is invalid; fail-close")
    return sha


def _fetch_comment(
    *,
    provider: github_approval.GitHubApprovalProvider,
    repository: str,
    comment_id: int,
) -> dict[str, Any]:
    try:
        value = provider.get_issue_comment(repository, comment_id)
    except Exception as exc:
        raise ContractError(f"GitHub approval unavailable; fail-close: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("GitHub approval response must be an object")
    return value


def _comment_issue_number(comment: Mapping[str, Any], repository: str) -> int:
    raw = comment.get("issue_url")
    if not isinstance(raw, str):
        raise ContractError("GitHub approval comment issue_url is missing")
    parsed = urllib.parse.urlparse(raw)
    expected_path = f"/repos/{repository}/issues/"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.github.com"
        or parsed.params
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith(expected_path)
    ):
        raise ContractError("GitHub approval issue_url repository mismatch")
    suffix = parsed.path[len(expected_path) :]
    if not suffix.isdigit() or int(suffix) <= 0:
        raise ContractError("GitHub approval issue number is invalid")
    return int(suffix)


def _validate_html_url(
    value: Any,
    *,
    repository: str,
    issue_number: int,
    comment_id: int,
) -> str:
    if not isinstance(value, str):
        raise ContractError("GitHub approval html_url is missing")
    parsed = urllib.parse.urlparse(value)
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    owner, repo = repository.split("/", 1)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.params
        or parsed.query
        or len(parts) != 4
        or [part.lower() for part in parts[:2]] != [owner.lower(), repo.lower()]
        or parts[2] not in {"issues", "pull"}
        or parts[3] != str(issue_number)
        or parsed.fragment != f"issuecomment-{comment_id}"
    ):
        raise ContractError("GitHub approval html_url mismatch")
    return value


def _verify_comment(
    *,
    provider: github_approval.GitHubApprovalProvider,
    allowlist: Mapping[str, Any],
    repository: str,
    issue_number: int,
    comment_id: int,
    run_digest: str,
    sealed_at: str,
) -> dict[str, Any]:
    if type(issue_number) is not int or issue_number <= 0:
        raise ContractError("issue_number must be a positive integer")
    if type(comment_id) is not int or comment_id <= 0:
        raise ContractError("comment_id must be a positive integer")
    comment = _fetch_comment(
        provider=provider,
        repository=repository,
        comment_id=comment_id,
    )
    if type(comment.get("id")) is not int or comment.get("id") != comment_id:
        raise ContractError("GitHub approval comment ID mismatch")
    if _comment_issue_number(comment, repository) != issue_number:
        raise ContractError("GitHub approval comment belongs to another Issue")

    user = comment.get("user")
    if not isinstance(user, dict):
        raise ContractError("GitHub approval author is missing")
    login = user.get("login")
    actor_type = user.get("type")
    if not isinstance(login, str) or not login.strip() or actor_type != "User":
        raise ContractError("only an authenticated GitHub User can approve")
    normalized_login = login.strip().lower()
    approvers = allowlist.get("approvers")
    denied = allowlist.get("denied_login_patterns")
    if (
        not isinstance(approvers, set)
        or not all(isinstance(item, str) for item in approvers)
        or not isinstance(denied, list)
        or not all(isinstance(item, str) and item for item in denied)
    ):
        raise ContractError("GitHub allowlist evidence is invalid")
    denied_patterns = AUTOMATION_LOGIN_PATTERNS | {
        str(item).strip().lower() for item in denied
    }
    if any(pattern in normalized_login for pattern in denied_patterns):
        raise ContractError("automation actors cannot approve offline diagnostics")
    if normalized_login not in {str(item).strip().lower() for item in approvers}:
        raise ContractError("GitHub approval author is not allowlisted")

    body = comment.get("body")
    expected_body = f"{APPROVAL_KEYWORD} {run_digest}"
    if body != expected_body:
        raise ContractError(f"approval body must exactly equal {expected_body!r}")
    created_at = comment.get("created_at")
    updated_at = comment.get("updated_at")
    created = _parse_utc(created_at, label="comment created_at")
    updated = _parse_utc(updated_at, label="comment updated_at")
    sealed = _parse_utc(sealed_at, label="run scope sealed_at")
    if created <= sealed:
        raise ContractError("approval comment must be created after run-scope sealing")
    if updated != created or updated_at != created_at:
        raise ContractError("edited approval comments are rejected")
    html_url = _validate_html_url(
        comment.get("html_url"),
        repository=repository,
        issue_number=issue_number,
        comment_id=comment_id,
    )
    return {
        "approval_type": APPROVAL_KEYWORD,
        "approval_digest": run_digest,
        "repository": repository,
        "issue_number": issue_number,
        "comment_id": comment_id,
        "url": html_url,
        "author": login.strip(),
        "author_type": actor_type,
        "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _verify_remote_runtime_materials(
    *,
    provider: github_approval.GitHubApprovalProvider,
    registered: OfflineRegisteredRecipe,
    run_scope: Mapping[str, Any],
) -> dict[str, Any]:
    repository = run_scope["repository"]
    ref = run_scope["run_scope_base_commit"]
    bindings = run_scope.get("runtime_bindings")
    if not isinstance(bindings, Mapping):
        raise ContractError("offline run scope runtime bindings are missing")
    expected = bindings.get("runtime_material_sha256")
    if not isinstance(expected, Mapping) or dict(expected) != dict(
        registered.runtime_material_digests
    ):
        raise ContractError("offline runtime material bindings are not canonical")

    cache: dict[str, tuple[bytes, dict[str, str]]] = {}
    for field, path in sorted(RUNTIME_MATERIAL_PATHS.items()):
        expected_digest = expected.get(field)
        if not isinstance(expected_digest, str):
            raise ContractError(f"offline runtime material digest is missing: {field}")
        try:
            content, evidence = github_approval.verify_github_file_at_commit(
                provider=provider,
                repository=repository,
                path=path,
                ref=ref,
                expected_content_sha256=expected_digest,
            )
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        if path in cache:
            raise ContractError(f"duplicate offline runtime material path: {path}")
        cache[path] = (content, evidence)

    policy_path = POLICY_RELATIVE_PATH.as_posix()
    try:
        remote_policy = strict_json_loads(
            cache[policy_path][0].decode("utf-8"),
            label="remote offline policy",
        )
    except UnicodeError as exc:
        raise ContractError("remote offline policy is not UTF-8") from exc
    if (
        not isinstance(remote_policy, dict)
        or canonical_digest(remote_policy) != run_scope.get("policy_digest")
    ):
        raise ContractError("remote offline policy canonical digest mismatch")

    recipe_path = SOURCE_RECIPE_RELATIVE_PATH.as_posix()
    try:
        remote_recipe = strict_json_loads(
            cache[recipe_path][0].decode("utf-8"),
            label="remote source recipe",
        )
    except UnicodeError as exc:
        raise ContractError("remote source recipe is not UTF-8") from exc
    if (
        not isinstance(remote_recipe, dict)
        or canonical_digest(remote_recipe) != run_scope.get("recipe_digest")
    ):
        raise ContractError("remote source recipe canonical digest mismatch")

    schema_materials: list[dict[str, str]] = []
    for schema_id, path in sorted(EXPECTED_SCHEMA_PATHS.items()):
        evidence = cache[path][1]
        schema_materials.append(
            {
                "schema_id": schema_id,
                "path": path,
                "content_sha256": evidence["content_sha256"],
            }
        )
    if canonical_digest(schema_materials) != expected.get("schema_bundle_sha256"):
        raise ContractError("remote offline schema bundle digest mismatch")

    output = {
        "run_scope_base_commit": ref,
        "materials": [
            evidence
            for _, evidence in sorted(cache.values(), key=lambda item: item[1]["path"])
        ],
        "schema_bundle_sha256": expected["schema_bundle_sha256"],
    }
    output["evidence_digest"] = canonical_digest(output)
    return output


def _collect_ordinary_grant_comment_ids(value: Any, output: set[int]) -> None:
    if isinstance(value, Mapping):
        approval_type = value.get("approval_type")
        if approval_type in ORDINARY_APPROVAL_TYPES:
            comment_id = value.get("comment_id")
            if type(comment_id) is not int or comment_id <= 0:
                raise ContractError("ordinary registry grant comment_id is invalid")
            output.add(comment_id)
        for child in value.values():
            _collect_ordinary_grant_comment_ids(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect_ordinary_grant_comment_ids(child, output)


def _verify_ordinary_registry_comment_unused(
    *,
    provider: github_approval.GitHubApprovalProvider,
    repository: str,
    ref: str,
    comment_id: int,
) -> dict[str, Any]:
    try:
        content, file_evidence = github_approval.fetch_github_file_at_commit(
            provider=provider,
            repository=repository,
            path=REGISTRY_PATH,
            ref=ref,
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    if content.startswith(b"\xef\xbb\xbf"):
        raise ContractError("remote ordinary registry must not contain a UTF-8 BOM")
    try:
        raw = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("remote ordinary registry is not UTF-8") from exc
    used: set[int] = set()
    nonblank_lines = 0
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        nonblank_lines += 1
        event = strict_json_loads(
            line,
            label=f"remote ordinary registry line {line_number}",
        )
        if not isinstance(event, dict):
            raise ContractError(
                f"remote ordinary registry line {line_number} must be an object"
            )
        _collect_ordinary_grant_comment_ids(event, used)
    if comment_id in used:
        raise ContractError(
            "GitHub approval comment ID is already used by an ordinary registry grant"
        )
    return {
        **file_evidence,
        "scanned_nonblank_line_count": nonblank_lines,
        "used_grant_comment_id_count": len(used),
        "target_comment_id": comment_id,
        "target_comment_id_unused": True,
    }


def _build_evidence(
    *,
    run_scope: Mapping[str, Any],
    run_digest: str,
    checkpoint: str,
    original_evidence_digest: str | None,
    trust: Mapping[str, Any],
    runtime_materials: Mapping[str, Any],
    ordinary_registry: Mapping[str, Any],
    comment: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = {
        "schema_version": 1,
        "gate_kind": GATE_KIND,
        "run_scope_digest": run_digest,
        "verification_checkpoint": checkpoint,
        "original_evidence_digest": original_evidence_digest,
        "github_trust": dict(trust),
        "runtime_materials": dict(runtime_materials),
        "ordinary_registry": dict(ordinary_registry),
        "comment": dict(comment),
        "limitations": dict(LIMITATIONS),
        "implementation_current_main_ancestry_verified": True,
        "authority": False,
        "local_offline_permission": True,
        "global_uniqueness_guaranteed": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    evidence["evidence_digest"] = canonical_digest(evidence)
    return evidence


def _verify_current(
    *,
    root: Path,
    run_scope: Mapping[str, Any],
    issue_number: int,
    comment_id: int,
    provider: github_approval.GitHubApprovalProvider,
    now: str | Callable[[], str] | None,
    checkpoint: str,
    original_evidence_digest: str | None,
) -> dict[str, Any]:
    registered = resolve_offline_registered_recipe(Path(root))
    run_digest = verify_canonical_offline_run_scope(registered, run_scope)
    repository = run_scope.get("repository")
    base_branch = run_scope.get("base_branch")
    base_commit = run_scope.get("run_scope_base_commit")
    try:
        allowlist, trust = github_approval.verify_github_trust(
            provider=provider,
            repository=repository,
            base_branch=base_branch,
            base_commit=base_commit,
            clock=_clock_from(now),
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    _parse_utc(trust.get("verification_time"), label="GitHub verification time")
    if trust.get("verified_current_main_sha") != run_scope.get(
        "verified_current_main_sha"
    ):
        raise ContractError("GitHub main moved or differs from frozen offline run scope")
    if trust.get("verified_base_commit") != base_commit:
        raise ContractError("GitHub base commit differs from frozen offline run scope")
    if trust.get("approvers_blob_sha") != run_scope.get("approvers_blob_sha"):
        raise ContractError("APPROVERS blob differs from frozen offline run scope")
    if trust.get("approvers_content_sha256") != run_scope.get(
        "approvers_content_sha256"
    ):
        raise ContractError("APPROVERS content differs from frozen offline run scope")

    runtime_materials = _verify_remote_runtime_materials(
        provider=provider,
        registered=registered,
        run_scope=run_scope,
    )
    ordinary_registry = _verify_ordinary_registry_comment_unused(
        provider=provider,
        repository=repository,
        ref=base_commit,
        comment_id=comment_id,
    )
    comment = _verify_comment(
        provider=provider,
        allowlist=allowlist,
        repository=repository,
        issue_number=issue_number,
        comment_id=comment_id,
        run_digest=run_digest,
        sealed_at=run_scope.get("sealed_at"),
    )
    try:
        github_approval.verify_github_main_head_unchanged(
            provider=provider,
            repository=repository,
            base_branch=base_branch,
            expected_sha=trust["verified_current_main_sha"],
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return _build_evidence(
        run_scope=run_scope,
        run_digest=run_digest,
        checkpoint=checkpoint,
        original_evidence_digest=original_evidence_digest,
        trust=trust,
        runtime_materials=runtime_materials,
        ordinary_registry=ordinary_registry,
        comment=comment,
    )


def _verify_stored_initial_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_run_digest: str,
) -> str:
    if not isinstance(evidence, Mapping) or set(evidence) != EVIDENCE_FIELDS:
        raise ContractError("stored offline approval evidence keys differ")
    stored_digest = evidence.get("evidence_digest")
    if (
        not isinstance(stored_digest, str)
        or len(stored_digest) != 64
        or any(character not in "0123456789abcdef" for character in stored_digest)
    ):
        raise ContractError("stored offline approval evidence digest is invalid")
    unsigned = dict(evidence)
    unsigned.pop("evidence_digest", None)
    if canonical_digest(unsigned) != stored_digest:
        raise ContractError("stored offline approval evidence changed")
    required = {
        "schema_version": 1,
        "gate_kind": GATE_KIND,
        "run_scope_digest": expected_run_digest,
        "verification_checkpoint": INITIAL_CHECKPOINT,
        "original_evidence_digest": None,
        "limitations": LIMITATIONS,
        "implementation_current_main_ancestry_verified": True,
        "authority": False,
        "local_offline_permission": True,
        "global_uniqueness_guaranteed": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    for key, expected in required.items():
        if evidence.get(key) != expected:
            raise ContractError(f"stored offline approval evidence {key} mismatch")
    return stored_digest


def verify_offline_run_approval(
    *,
    root: Path,
    run_scope: Mapping[str, Any],
    issue_number: int,
    comment_id: int,
    provider: github_approval.GitHubApprovalProvider,
    now: str | Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Verify one exact post-seal GitHub comment for the fixed local run.

    The result grants only the local, non-promotional diagnostic permission
    represented by the frozen scope.  It is deliberately not a global or
    rollback-resistant authority receipt.
    """

    return _verify_current(
        root=Path(root),
        run_scope=run_scope,
        issue_number=issue_number,
        comment_id=comment_id,
        provider=provider,
        now=now,
        checkpoint=INITIAL_CHECKPOINT,
        original_evidence_digest=None,
    )


def reverify_offline_run_approval(
    *,
    root: Path,
    run_scope: Mapping[str, Any],
    approval_evidence: Mapping[str, Any],
    provider: github_approval.GitHubApprovalProvider,
    now: str | Callable[[], str] | None = None,
    checkpoint: str,
) -> dict[str, Any]:
    """Re-fetch all remote evidence at one of the two protected boundaries."""

    if checkpoint not in REVERIFY_CHECKPOINTS:
        raise ContractError("unsupported offline approval revalidation checkpoint")
    registered = resolve_offline_registered_recipe(Path(root))
    run_digest = verify_canonical_offline_run_scope(registered, run_scope)
    original_digest = _verify_stored_initial_evidence(
        approval_evidence,
        expected_run_digest=run_digest,
    )
    stored_comment = approval_evidence.get("comment")
    if not isinstance(stored_comment, Mapping):
        raise ContractError("stored offline approval comment evidence is missing")
    current = _verify_current(
        root=Path(root),
        run_scope=run_scope,
        issue_number=stored_comment.get("issue_number"),
        comment_id=stored_comment.get("comment_id"),
        provider=provider,
        now=now,
        checkpoint=checkpoint,
        original_evidence_digest=original_digest,
    )
    if current["comment"] != stored_comment:
        raise ContractError("offline approval comment changed or was replaced")
    if current["runtime_materials"] != approval_evidence.get("runtime_materials"):
        raise ContractError("offline runtime material evidence changed")
    if current["ordinary_registry"] != approval_evidence.get("ordinary_registry"):
        raise ContractError("ordinary registry evidence changed")
    stored_trust = approval_evidence.get("github_trust")
    current_trust = current.get("github_trust")
    if not isinstance(stored_trust, Mapping) or not isinstance(current_trust, Mapping):
        raise ContractError("stored GitHub trust evidence is missing")
    stable_stored = dict(stored_trust)
    stable_current = dict(current_trust)
    stable_stored.pop("verification_time", None)
    stable_current.pop("verification_time", None)
    if stable_current != stable_stored:
        raise ContractError("GitHub trust evidence changed")
    return current


def verify_offline_gate_availability(
    *,
    root: Path,
    provider: github_approval.GitHubApprovalProvider,
    expected_base_commit: str | None = None,
    now: str | Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Verify that the fixed implementation bytes exist on GitHub main ancestry.

    This metadata-only check is suitable immediately before projection
    materialization.  It does not inspect an approval comment and does not
    grant permission to execute the ROI diagnostic.
    """

    registered = resolve_offline_registered_recipe(Path(root))
    if expected_base_commit is None:
        base_commit = _current_main_sha(provider=provider)
    elif not isinstance(expected_base_commit, str) or not FULL_GIT_SHA.fullmatch(
        expected_base_commit
    ):
        raise ContractError("expected_base_commit must be a full lowercase Git SHA")
    else:
        base_commit = expected_base_commit
    try:
        _, trust = github_approval.verify_github_trust(
            provider=provider,
            repository=DEFAULT_REPOSITORY,
            base_branch=DEFAULT_BASE_BRANCH,
            base_commit=base_commit,
            clock=_clock_from(now),
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    _parse_utc(trust.get("verification_time"), label="GitHub verification time")
    runtime_materials = _verify_remote_runtime_materials(
        provider=provider,
        registered=registered,
        run_scope={
            "repository": DEFAULT_REPOSITORY,
            "run_scope_base_commit": base_commit,
            "runtime_bindings": {
                "runtime_material_sha256": dict(
                    registered.runtime_material_digests
                )
            },
            "policy_digest": registered.policy_digest,
            "recipe_digest": registered.recipe_digest,
        },
    )
    try:
        github_approval.verify_github_main_head_unchanged(
            provider=provider,
            repository=DEFAULT_REPOSITORY,
            base_branch=DEFAULT_BASE_BRANCH,
            expected_sha=trust["verified_current_main_sha"],
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    evidence = {
        "schema_version": 1,
        "gate_kind": GATE_KIND,
        "repository": DEFAULT_REPOSITORY,
        "base_branch": DEFAULT_BASE_BRANCH,
        "implementation_commit": base_commit,
        "verified_current_main_sha": trust["verified_current_main_sha"],
        "github_trust": trust,
        "runtime_materials": runtime_materials,
        "implementation_current_main_ancestry_verified": True,
        "materialization_permission": True,
        "run_approval_verified": False,
        "local_offline_run_permission": False,
        "authority": False,
        "limitations": dict(LIMITATIONS),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    evidence["evidence_digest"] = canonical_digest(evidence)
    return evidence


__all__ = [
    "INITIAL_CHECKPOINT",
    "LIMITATIONS",
    "REGISTRY_PATH",
    "REVERIFY_CHECKPOINTS",
    "reverify_offline_run_approval",
    "verify_offline_gate_availability",
    "verify_offline_run_approval",
]
