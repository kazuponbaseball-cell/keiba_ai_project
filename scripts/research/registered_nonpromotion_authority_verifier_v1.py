from __future__ import annotations

import hashlib
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import github_approval
from registered_nonpromotion_contract_v1 import (
    APPROVAL_KEYWORD,
    ContractError,
    EXPECTED_SCHEMA_PATHS,
    G2_MATERIAL_PATHS,
    POLICY_RELATIVE_PATH,
    RUNTIME_MATERIAL_PATHS,
    RegisteredRecipe,
    canonical_digest,
    strict_json_loads,
    verify_canonical_run_scope,
)


AUTOMATION_LOGIN_PATTERNS = {
    "automation",
    "bot",
    "codex",
    "dependabot",
    "github-actions",
}
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _parse_time(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContractError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError(f"{label} must be UTC")
    return parsed


def _comment_issue_number(comment: Mapping[str, Any], repository: str) -> int:
    raw = comment.get("issue_url")
    if not isinstance(raw, str):
        raise ContractError("GitHub approval comment issue_url is missing")
    parsed = urllib.parse.urlparse(raw)
    expected_prefix = f"/repos/{repository}/issues/"
    if parsed.scheme != "https" or parsed.netloc != "api.github.com":
        raise ContractError("GitHub approval issue_url host is invalid")
    if not parsed.path.startswith(expected_prefix):
        raise ContractError("GitHub approval issue_url repository mismatch")
    suffix = parsed.path[len(expected_prefix) :]
    if not suffix.isdigit() or int(suffix) <= 0:
        raise ContractError("GitHub approval issue number is invalid")
    return int(suffix)


def _validate_html_url(
    value: Any, *, repository: str, issue_number: int, comment_id: int
) -> str:
    if not isinstance(value, str):
        raise ContractError("GitHub approval html_url is missing")
    parsed = urllib.parse.urlparse(value)
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    owner, repo = repository.split("/", 1)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or len(parts) != 4
        or [item.lower() for item in parts[:2]] != [owner.lower(), repo.lower()]
        or parts[2] not in {"issues", "pull"}
        or parts[3] != str(issue_number)
        or parsed.fragment != f"issuecomment-{comment_id}"
    ):
        raise ContractError("GitHub approval html_url mismatch")
    return value


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


def verify_registered_runtime_materials(
    *,
    provider: github_approval.GitHubApprovalProvider,
    registered: RegisteredRecipe,
    run_scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify every repository-owned run material from immutable GitHub bytes."""

    verify_canonical_run_scope(registered, run_scope)
    repository = run_scope["repository"]
    ref = run_scope["run_scope_base_commit"]
    bindings = run_scope.get("runtime_bindings")
    if not isinstance(bindings, Mapping):
        raise ContractError("run scope runtime bindings are missing")
    cache: dict[str, tuple[bytes, dict[str, str]]] = {}

    def fetch(path: str, expected: str) -> tuple[bytes, dict[str, str]]:
        if path in cache:
            content, evidence = cache[path]
            if evidence["content_sha256"] != expected:
                raise ContractError(f"conflicting expected digest for {path}")
            return content, evidence
        try:
            value = github_approval.verify_github_file_at_commit(
                provider=provider,
                repository=repository,
                path=path,
                ref=ref,
                expected_content_sha256=expected,
            )
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        cache[path] = value
        return value

    policy_path = POLICY_RELATIVE_PATH.as_posix()
    policy_bytes, _ = fetch(policy_path, bindings["policy_blob_sha256"])
    try:
        policy = strict_json_loads(policy_bytes.decode("utf-8"), label="remote policy")
    except UnicodeError as exc:
        raise ContractError("remote policy is not UTF-8") from exc
    if not isinstance(policy, dict) or canonical_digest(policy) != run_scope["policy_digest"]:
        raise ContractError("remote policy canonical digest mismatch")
    entries = policy.get("recipe_registry", {}).get("entries", [])
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("recipe_id") == run_scope["recipe_id"]
        and entry.get("recipe_version") == run_scope["recipe_version"]
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("path"), str):
        raise ContractError("remote policy does not resolve the exact recipe once")
    recipe_path = matches[0]["path"]
    recipe_bytes, _ = fetch(recipe_path, bindings["recipe_blob_sha256"])
    try:
        recipe = strict_json_loads(recipe_bytes.decode("utf-8"), label="remote recipe")
    except UnicodeError as exc:
        raise ContractError("remote recipe is not UTF-8") from exc
    if not isinstance(recipe, dict) or canonical_digest(recipe) != run_scope["recipe_digest"]:
        raise ContractError("remote recipe canonical digest mismatch")

    for field, path in RUNTIME_MATERIAL_PATHS.items():
        fetch(path, bindings[field])
    g2_materials: list[dict[str, str]] = []
    for path in G2_MATERIAL_PATHS:
        if path in cache:
            content, evidence = cache[path]
        else:
            try:
                content, evidence = github_approval.fetch_github_file_at_commit(
                    provider=provider,
                    repository=repository,
                    path=path,
                    ref=ref,
                )
            except ValueError as exc:
                raise ContractError(str(exc)) from exc
            cache[path] = (content, evidence)
        g2_materials.append(
            {"path": path, "content_sha256": evidence["content_sha256"]}
        )
    if canonical_digest(g2_materials) != bindings["g2_authority_service_blob_sha256"]:
        raise ContractError("remote shared-G2 material bundle digest mismatch")

    schema_materials: list[dict[str, str]] = []
    for schema_id, path in sorted(EXPECTED_SCHEMA_PATHS.items()):
        if path in cache:
            _, evidence = cache[path]
        else:
            try:
                content, evidence = github_approval.fetch_github_file_at_commit(
                    provider=provider,
                    repository=repository,
                    path=path,
                    ref=ref,
                )
            except ValueError as exc:
                raise ContractError(str(exc)) from exc
            cache[path] = (content, evidence)
        schema_materials.append(
            {
                "schema_id": schema_id,
                "path": path,
                "content_sha256": evidence["content_sha256"],
            }
        )
    if canonical_digest(schema_materials) != bindings["schema_bundle_sha256"]:
        raise ContractError("remote schema bundle digest mismatch")
    if canonical_digest(policy.get("capability_profiles")) != bindings[
        "capability_profile_sha256"
    ]:
        raise ContractError("remote capability profile digest mismatch")
    materials = [
        evidence
        for _, evidence in sorted(cache.values(), key=lambda item: item[1]["path"])
    ]
    output = {
        "run_scope_base_commit": ref,
        "materials": materials,
        "schema_bundle_sha256": bindings["schema_bundle_sha256"],
        "g2_authority_service_blob_sha256": bindings[
            "g2_authority_service_blob_sha256"
        ],
        "capability_profile_sha256": bindings["capability_profile_sha256"],
    }
    output["evidence_digest"] = canonical_digest(output)
    return output


def verify_registered_run_approval(
    *,
    provider: github_approval.GitHubApprovalProvider,
    registered: RegisteredRecipe,
    run_scope: Mapping[str, Any],
    issue_number: int,
    comment_id: int,
    clock: Callable[[], str] = github_approval.utc_now,
) -> dict[str, Any]:
    """Verify the lane's one exact comment without changing ordinary keywords.

    This returns evidence only.  It is not authority until an authenticated
    shared-G2 transaction globally reserves the comment and both subjects.
    """

    run_digest = verify_canonical_run_scope(registered, run_scope)
    if type(issue_number) is not int or issue_number <= 0:
        raise ContractError("issue_number must be a positive integer")
    if type(comment_id) is not int or comment_id <= 0:
        raise ContractError("comment_id must be a positive integer")
    repository = run_scope.get("repository")
    base_branch = run_scope.get("base_branch")
    base_commit = run_scope.get("run_scope_base_commit")
    try:
        allowlist, trust = github_approval.verify_github_trust(
            provider=provider,
            repository=repository,
            base_branch=base_branch,
            base_commit=base_commit,
            clock=clock,
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    if trust["verified_current_main_sha"] != run_scope.get("verified_current_main_sha"):
        raise ContractError("GitHub main moved or differs from frozen run scope")
    if trust["approvers_blob_sha"] != run_scope.get("approvers_blob_sha"):
        raise ContractError("APPROVERS blob differs from frozen run scope")
    if trust["approvers_content_sha256"] != run_scope.get("approvers_content_sha256"):
        raise ContractError("APPROVERS content differs from frozen run scope")
    runtime_materials = verify_registered_runtime_materials(
        provider=provider,
        registered=registered,
        run_scope=run_scope,
    )

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
    if not isinstance(approvers, set) or not isinstance(denied, list):
        raise ContractError("GitHub allowlist evidence is invalid")
    patterns = AUTOMATION_LOGIN_PATTERNS | {str(item).lower() for item in denied}
    if any(pattern in normalized_login for pattern in patterns):
        raise ContractError("automation actors cannot approve diagnostics")
    if normalized_login not in {str(item).lower() for item in approvers}:
        raise ContractError("GitHub approval author is not allowlisted")
    body = comment.get("body")
    expected_body = f"{APPROVAL_KEYWORD} {run_digest}"
    if body != expected_body:
        raise ContractError(f"approval body must exactly equal {expected_body!r}")
    created_at = comment.get("created_at")
    updated_at = comment.get("updated_at")
    created = _parse_time(created_at, label="comment created_at")
    updated = _parse_time(updated_at, label="comment updated_at")
    sealed = _parse_time(run_scope.get("sealed_at"), label="run scope sealed_at")
    if created <= sealed:
        raise ContractError("approval comment must be created after run-scope sealing")
    if updated != created:
        raise ContractError("edited approval comments are rejected")
    html_url = _validate_html_url(
        comment.get("html_url"),
        repository=repository,
        issue_number=issue_number,
        comment_id=comment_id,
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

    comment_evidence = {
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
    evidence = {
        "schema_version": 1,
        "gate_kind": run_scope["gate_kind"],
        "run_scope_digest": run_digest,
        "github_trust": trust,
        "runtime_materials": runtime_materials,
        "comment": comment_evidence,
        "global_g2_reservation_required": True,
        "authority": False,
    }
    evidence["evidence_digest"] = canonical_digest(evidence)
    return evidence


def reverify_registered_run_approval(
    *,
    provider: github_approval.GitHubApprovalProvider,
    registered: RegisteredRecipe,
    run_scope: Mapping[str, Any],
    stored_evidence: Mapping[str, Any],
    clock: Callable[[], str] = github_approval.utc_now,
) -> dict[str, Any]:
    if not isinstance(stored_evidence, dict):
        raise ContractError("stored approval evidence is invalid")
    stored_digest = stored_evidence.get("evidence_digest")
    if not isinstance(stored_digest, str) or not FULL_SHA256.fullmatch(stored_digest):
        raise ContractError("stored approval evidence digest is invalid")
    unsigned = dict(stored_evidence)
    unsigned.pop("evidence_digest", None)
    if canonical_digest(unsigned) != stored_digest:
        raise ContractError("stored approval evidence changed")
    comment = stored_evidence.get("comment")
    if not isinstance(comment, dict):
        raise ContractError("stored comment evidence missing")
    current = verify_registered_run_approval(
        provider=provider,
        registered=registered,
        run_scope=run_scope,
        issue_number=comment.get("issue_number"),
        comment_id=comment.get("comment_id"),
        clock=clock,
    )
    if current.get("comment") != comment:
        raise ContractError("approval comment changed or was replaced")
    if current.get("github_trust", {}).get("verified_base_commit") != stored_evidence.get(
        "github_trust", {}
    ).get("verified_base_commit"):
        raise ContractError("approval base evidence changed")
    return current


__all__ = [
    "reverify_registered_run_approval",
    "verify_registered_run_approval",
    "verify_registered_runtime_materials",
]
