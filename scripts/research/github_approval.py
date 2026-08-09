from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Callable, Protocol


GITHUB_API_URL = "https://api.github.com"
GITHUB_WEB_URL = "https://github.com"
GITHUB_API_VERSION = "2022-11-28"
DEFAULT_REPOSITORY = "kazuponbaseball-cell/keiba_ai_project"
DEFAULT_BASE_BRANCH = "main"
APPROVERS_PATH = "research/APPROVERS.json"

FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
APPROVAL_KEYWORDS = {
    "APPROVED_TO_PREPARE",
    "APPROVED_TO_RUN",
    "APPROVED_FOR_SHADOW",
}
AUTOMATION_LOGIN_PATTERNS = {
    "bot",
    "codex",
    "automation",
    "github-actions",
    "dependabot",
}

GITHUB_TRUST_EVIDENCE_FIELDS = (
    "repository_full_name",
    "base_branch",
    "verified_current_main_sha",
    "verified_base_commit",
    "compare_url",
    "compare_status",
    "merge_base_sha",
    "approvers_blob_sha",
    "approvers_content_sha256",
    "verification_time",
)

COMMENT_EVIDENCE_FIELDS = (
    "approval_type",
    "approval_digest",
    "repository",
    "issue_number",
    "comment_id",
    "url",
    "author",
    "author_type",
    "body",
    "body_sha256",
    "created_at",
    "updated_at",
)

REMOTE_FILE_EVIDENCE_FIELDS = (
    "path",
    "ref",
    "blob_sha",
    "content_sha256",
)


class GitHubApprovalProvider(Protocol):
    """Injectable, read-only GitHub data source used by approval verification."""

    def get_repository(self, repository: str) -> dict[str, Any]:
        ...

    def get_branch_ref(self, repository: str, branch: str) -> dict[str, Any]:
        ...

    def compare_commits(
        self,
        repository: str,
        base_commit: str,
        head_commit: str,
    ) -> dict[str, Any]:
        ...

    def get_file_contents(
        self,
        repository: str,
        path: str,
        ref: str,
    ) -> dict[str, Any]:
        ...

    def get_issue_comment(self, repository: str, comment_id: int) -> dict[str, Any]:
        ...


class GitHubRestApprovalProvider:
    """GET-only provider pinned to the public GitHub REST API."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")

    @staticmethod
    def _repository_path(repository: str) -> str:
        if not isinstance(repository, str) or not SAFE_REPOSITORY.fullmatch(repository):
            raise ValueError("repository must be an owner/name GitHub repository")
        owner, name = repository.split("/", 1)
        return f"{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(name, safe='')}"

    def _get_json(self, url: str, label: str) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "api.github.com":
            raise RuntimeError(f"refusing non-GitHub API URL for {label}")
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "keiba-ai-research-os",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                raw = response.read()
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise RuntimeError(f"GitHub {label} GET failed: {exc}") from exc
        def reject_constant(value: str) -> None:
            raise ValueError(f"non-standard JSON constant is forbidden: {value}")

        try:
            payload = json.loads(
                raw.decode("utf-8"),
                parse_constant=reject_constant,
                object_pairs_hook=_strict_json_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"GitHub {label} response is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"GitHub {label} response is not an object")
        return payload

    def get_repository(self, repository: str) -> dict[str, Any]:
        repository_path = self._repository_path(repository)
        return self._get_json(
            f"{GITHUB_API_URL}/repos/{repository_path}",
            "repository",
        )

    def get_branch_ref(self, repository: str, branch: str) -> dict[str, Any]:
        repository_path = self._repository_path(repository)
        if not isinstance(branch, str) or not branch:
            raise ValueError("branch must be a non-empty string")
        encoded_branch = urllib.parse.quote(branch, safe="")
        return self._get_json(
            f"{GITHUB_API_URL}/repos/{repository_path}/git/ref/heads/{encoded_branch}",
            "branch ref",
        )

    def compare_commits(
        self,
        repository: str,
        base_commit: str,
        head_commit: str,
    ) -> dict[str, Any]:
        repository_path = self._repository_path(repository)
        encoded_base = urllib.parse.quote(base_commit, safe="")
        encoded_head = urllib.parse.quote(head_commit, safe="")
        return self._get_json(
            f"{GITHUB_API_URL}/repos/{repository_path}/compare/{encoded_base}...{encoded_head}",
            "commit comparison",
        )

    def get_file_contents(
        self,
        repository: str,
        path: str,
        ref: str,
    ) -> dict[str, Any]:
        repository_path = self._repository_path(repository)
        if not isinstance(path, str) or not path:
            raise ValueError("repository file path must be a non-empty string")
        normalized_path = PurePosixPath(path)
        if normalized_path.is_absolute() or any(part in {"", ".", ".."} for part in normalized_path.parts):
            raise ValueError("repository file path must be normalized and relative")
        encoded_path = urllib.parse.quote(normalized_path.as_posix(), safe="/")
        query = urllib.parse.urlencode({"ref": ref})
        return self._get_json(
            f"{GITHUB_API_URL}/repos/{repository_path}/contents/{encoded_path}?{query}",
            "repository contents",
        )

    def get_issue_comment(self, repository: str, comment_id: int) -> dict[str, Any]:
        repository_path = self._repository_path(repository)
        if type(comment_id) is not int or comment_id <= 0:
            raise ValueError("comment_id must be a positive integer")
        return self._get_json(
            f"{GITHUB_API_URL}/repos/{repository_path}/issues/comments/{comment_id}",
            "issue comment",
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object; fail-close")
    return value


def _require_git_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not FULL_GIT_SHA.fullmatch(value):
        raise ValueError(f"{label} is not a full lowercase Git SHA; fail-close")
    return value


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _parse_approvers_content(content: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    try:
        raw = content.decode("utf-8")
        payload = json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("GitHub base-commit APPROVERS.json is invalid JSON; fail-close") from exc
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
        raise ValueError("GitHub base-commit APPROVERS.json schema_version must be 1")
    if payload["schema_version"] != 1:
        raise ValueError("GitHub base-commit APPROVERS.json schema_version must be 1")

    approvers = payload.get("approvers")
    denied_patterns = payload.get("denied_login_patterns", [])
    if not isinstance(approvers, list) or not approvers:
        raise ValueError("GitHub base-commit APPROVERS.json must contain approvers")
    logins: set[str] = set()
    for entry in approvers:
        if not isinstance(entry, dict) or not isinstance(entry.get("login"), str):
            raise ValueError("GitHub base-commit APPROVERS.json contains an invalid approver")
        login = entry["login"].strip().lower()
        if not login:
            raise ValueError("GitHub base-commit APPROVERS.json contains a blank login")
        if login in logins:
            raise ValueError("GitHub base-commit APPROVERS.json contains a duplicate login")
        logins.add(login)
    if not isinstance(denied_patterns, list) or any(
        not isinstance(item, str) or not item.strip() for item in denied_patterns
    ):
        raise ValueError("GitHub base-commit denied_login_patterns must be strings")
    normalized_patterns = [item.strip().lower() for item in denied_patterns]
    if len(normalized_patterns) != len(set(normalized_patterns)):
        raise ValueError("GitHub base-commit denied_login_patterns must be unique")
    return {
        "approvers": logins,
        "denied_login_patterns": normalized_patterns,
    }


def _provider_call(label: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        payload = operation()
    except Exception as exc:
        raise ValueError(f"GitHub {label} verification unavailable; fail-close: {exc}") from exc
    return _require_object(payload, f"GitHub {label} response")


def verify_github_trust(
    *,
    provider: GitHubApprovalProvider,
    repository: str,
    base_branch: str,
    base_commit: str,
    clock: Callable[[], str] = utc_now,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify GitHub main ancestry and load the immutable base-commit allowlist."""

    if not isinstance(repository, str) or repository.lower() != DEFAULT_REPOSITORY.lower():
        raise ValueError(
            f"GitHub repository mismatch; expected {DEFAULT_REPOSITORY!r}; fail-close"
        )
    if base_branch != DEFAULT_BASE_BRANCH:
        raise ValueError(
            f"GitHub base branch mismatch; expected {DEFAULT_BASE_BRANCH!r}; fail-close"
        )
    verified_base = _require_git_sha(base_commit, "proposal base commit")

    repository_payload = _provider_call(
        "repository",
        lambda: provider.get_repository(repository),
    )
    full_name = repository_payload.get("full_name")
    if not isinstance(full_name, str) or full_name.lower() != DEFAULT_REPOSITORY.lower():
        raise ValueError("GitHub repository full_name mismatch; fail-close")
    if repository_payload.get("default_branch") != DEFAULT_BASE_BRANCH:
        raise ValueError("GitHub repository default branch mismatch; fail-close")

    ref_payload = _provider_call(
        "main ref",
        lambda: provider.get_branch_ref(repository, base_branch),
    )
    if ref_payload.get("ref") != f"refs/heads/{DEFAULT_BASE_BRANCH}":
        raise ValueError("GitHub main ref mismatch; fail-close")
    ref_object = _require_object(ref_payload.get("object"), "GitHub main ref object")
    if ref_object.get("type") != "commit":
        raise ValueError("GitHub main ref does not point to a commit; fail-close")
    current_main = _require_git_sha(ref_object.get("sha"), "GitHub current main SHA")

    compare_payload = _provider_call(
        "compare",
        lambda: provider.compare_commits(repository, verified_base, current_main),
    )
    compare_status = compare_payload.get("status")
    if compare_status not in {"ahead", "identical"}:
        raise ValueError(
            f"proposal base commit is not an ancestor of GitHub main "
            f"(compare status {compare_status!r}); fail-close"
        )
    compare_base = _require_object(compare_payload.get("base_commit"), "GitHub compare base")
    if compare_base.get("sha") != verified_base:
        raise ValueError("GitHub compare base commit mismatch; fail-close")
    merge_base = _require_object(
        compare_payload.get("merge_base_commit"),
        "GitHub compare merge base",
    )
    merge_base_sha = _require_git_sha(merge_base.get("sha"), "GitHub merge-base SHA")
    if merge_base_sha != verified_base:
        raise ValueError("proposal base commit is not the GitHub main merge base; fail-close")
    if compare_status == "identical" and current_main != verified_base:
        raise ValueError("GitHub identical comparison SHA mismatch; fail-close")
    if compare_status == "ahead" and current_main == verified_base:
        raise ValueError("GitHub ahead comparison SHA mismatch; fail-close")

    compare_url = compare_payload.get("url")
    expected_compare_url = (
        f"{GITHUB_API_URL}/repos/{DEFAULT_REPOSITORY}/compare/"
        f"{verified_base}...{current_main}"
    )
    if compare_url != expected_compare_url:
        raise ValueError("GitHub compare URL mismatch; fail-close")

    contents_payload = _provider_call(
        "APPROVERS.json",
        lambda: provider.get_file_contents(
            repository,
            APPROVERS_PATH,
            verified_base,
        ),
    )
    if contents_payload.get("type") != "file" or contents_payload.get("path") != APPROVERS_PATH:
        raise ValueError("GitHub APPROVERS.json path or type mismatch; fail-close")
    if contents_payload.get("encoding") != "base64":
        raise ValueError("GitHub APPROVERS.json must use base64 content encoding; fail-close")
    blob_sha = _require_git_sha(
        contents_payload.get("sha"),
        "GitHub APPROVERS.json blob SHA",
    )
    encoded_content = contents_payload.get("content")
    if not isinstance(encoded_content, str) or not encoded_content:
        raise ValueError("GitHub APPROVERS.json content is missing; fail-close")
    try:
        compact_content = "".join(encoded_content.split())
        content = base64.b64decode(compact_content, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("GitHub APPROVERS.json content is invalid base64; fail-close") from exc
    allowlist = _parse_approvers_content(content)
    content_sha256 = hashlib.sha256(content).hexdigest()

    verification_time = clock()
    if not isinstance(verification_time, str) or not verification_time:
        raise ValueError("GitHub verification time is invalid; fail-close")
    evidence = {
        "repository_full_name": full_name,
        "base_branch": base_branch,
        "verified_current_main_sha": current_main,
        "verified_base_commit": verified_base,
        "compare_url": compare_url,
        "compare_status": compare_status,
        "merge_base_sha": merge_base_sha,
        "approvers_blob_sha": blob_sha,
        "approvers_content_sha256": content_sha256,
        "verification_time": verification_time,
    }
    if tuple(evidence) != GITHUB_TRUST_EVIDENCE_FIELDS:
        raise AssertionError("internal GitHub trust evidence schema mismatch")
    return allowlist, evidence


def verify_github_main_head_unchanged(
    *,
    provider: GitHubApprovalProvider,
    repository: str,
    base_branch: str,
    expected_sha: str,
) -> None:
    """Re-read the code-owned main ref and fail if it moved after verification."""

    if not isinstance(repository, str) or repository.lower() != DEFAULT_REPOSITORY.lower():
        raise ValueError(
            f"GitHub repository mismatch; expected {DEFAULT_REPOSITORY!r}; fail-close"
        )
    if base_branch != DEFAULT_BASE_BRANCH:
        raise ValueError(
            f"GitHub base branch mismatch; expected {DEFAULT_BASE_BRANCH!r}; fail-close"
        )
    verified_expected = _require_git_sha(expected_sha, "expected GitHub main SHA")
    ref_payload = _provider_call(
        "main ref recheck",
        lambda: provider.get_branch_ref(repository, base_branch),
    )
    if ref_payload.get("ref") != f"refs/heads/{DEFAULT_BASE_BRANCH}":
        raise ValueError("GitHub main ref mismatch during append recheck; fail-close")
    ref_object = _require_object(
        ref_payload.get("object"),
        "GitHub main ref recheck object",
    )
    if ref_object.get("type") != "commit":
        raise ValueError("GitHub main ref recheck does not point to a commit; fail-close")
    observed_sha = _require_git_sha(
        ref_object.get("sha"),
        "GitHub current main SHA recheck",
    )
    if observed_sha != verified_expected:
        raise ValueError(
            "GitHub current main moved during registry verification; fail-close and retry"
        )


def fetch_github_file_at_commit(
    *,
    provider: GitHubApprovalProvider,
    repository: str,
    path: str,
    ref: str,
) -> tuple[bytes, dict[str, str]]:
    """Fetch one immutable commit file and return its raw hash evidence."""

    if not isinstance(repository, str) or repository.lower() != DEFAULT_REPOSITORY.lower():
        raise ValueError(
            f"GitHub repository mismatch; expected {DEFAULT_REPOSITORY!r}; fail-close"
        )
    normalized_path = PurePosixPath(path)
    if (
        not isinstance(path, str)
        or normalized_path.is_absolute()
        or any(part in {"", ".", ".."} for part in normalized_path.parts)
        or normalized_path.as_posix() != path
    ):
        raise ValueError("GitHub policy path must be normalized and repository-relative")
    verified_ref = _require_git_sha(ref, "GitHub file ref")

    payload = _provider_call(
        path,
        lambda: provider.get_file_contents(repository, path, verified_ref),
    )
    if payload.get("type") != "file" or payload.get("path") != path:
        raise ValueError("GitHub file path or type mismatch; fail-close")
    if payload.get("encoding") != "base64":
        raise ValueError("GitHub file must use base64 content encoding; fail-close")
    blob_sha = _require_git_sha(payload.get("sha"), "GitHub file blob SHA")
    encoded_content = payload.get("content")
    if not isinstance(encoded_content, str):
        raise ValueError("GitHub file content is missing; fail-close")
    try:
        content = base64.b64decode("".join(encoded_content.split()), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("GitHub file content is invalid base64; fail-close") from exc
    observed_sha256 = hashlib.sha256(content).hexdigest()
    evidence = {
        "path": path,
        "ref": verified_ref,
        "blob_sha": blob_sha,
        "content_sha256": observed_sha256,
    }
    if tuple(evidence) != REMOTE_FILE_EVIDENCE_FIELDS:
        raise AssertionError("internal GitHub remote-file evidence schema mismatch")
    return content, evidence


def verify_github_file_at_commit(
    *,
    provider: GitHubApprovalProvider,
    repository: str,
    path: str,
    ref: str,
    expected_content_sha256: str,
) -> tuple[bytes, dict[str, str]]:
    """Fetch one immutable commit file and require its raw content hash."""

    if not isinstance(expected_content_sha256, str) or not FULL_SHA256.fullmatch(
        expected_content_sha256
    ):
        raise ValueError("expected GitHub file content SHA-256 is invalid; fail-close")
    content, evidence = fetch_github_file_at_commit(
        provider=provider,
        repository=repository,
        path=path,
        ref=ref,
    )
    observed_sha256 = evidence["content_sha256"]
    if observed_sha256 != expected_content_sha256:
        raise ValueError(
            "GitHub base-commit file content hash mismatch; fail-close: "
            f"expected {expected_content_sha256}, observed {observed_sha256}"
        )
    return content, evidence


def _comment_issue_number(comment: dict[str, Any], repository: str) -> int:
    issue_url = comment.get("issue_url")
    if not isinstance(issue_url, str):
        raise ValueError("GitHub approval comment is missing issue_url")
    expected_prefix = f"{GITHUB_API_URL}/repos/{repository}/issues/"
    if not issue_url.startswith(expected_prefix):
        raise ValueError("GitHub approval comment repository mismatch")
    suffix = issue_url[len(expected_prefix) :]
    if not suffix.isdigit() or int(suffix) <= 0:
        raise ValueError("GitHub approval comment issue_url is invalid")
    return int(suffix)


def _validate_comment_html_url(
    html_url: Any,
    repository: str,
    issue_number: int,
    comment_id: int,
) -> str:
    if not isinstance(html_url, str) or not html_url:
        raise ValueError("GitHub approval comment URL is invalid")
    parsed = urllib.parse.urlparse(html_url)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        raise ValueError("GitHub approval comment URL host is invalid")
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    expected_repository = repository.split("/", 1)
    if len(parts) != 4 or [part.lower() for part in parts[:2]] != [
        part.lower() for part in expected_repository
    ]:
        raise ValueError("GitHub approval comment URL repository mismatch")
    if parts[2] not in {"issues", "pull"} or parts[3] != str(issue_number):
        raise ValueError("GitHub approval comment URL Issue mismatch")
    if parsed.fragment != f"issuecomment-{comment_id}":
        raise ValueError("GitHub approval comment URL comment ID mismatch")
    return html_url


def verify_approval_comment(
    *,
    provider: GitHubApprovalProvider,
    allowlist: dict[str, Any],
    repository: str,
    issue_number: int,
    comment_id: int,
    approval_keyword: str,
    approval_digest: str,
) -> dict[str, Any]:
    """Verify one exact approval comment against a GitHub-derived allowlist."""

    if not isinstance(repository, str) or repository.lower() != DEFAULT_REPOSITORY.lower():
        raise ValueError("GitHub approval repository mismatch; fail-close")
    if type(issue_number) is not int or issue_number <= 0:
        raise ValueError("GitHub approval Issue number must be a positive integer")
    if type(comment_id) is not int or comment_id <= 0:
        raise ValueError("GitHub approval comment ID must be a positive integer")
    if approval_keyword not in APPROVAL_KEYWORDS:
        raise ValueError("unsupported GitHub approval keyword")
    if not isinstance(approval_digest, str) or not FULL_SHA256.fullmatch(approval_digest):
        raise ValueError("GitHub approval digest must be a full lowercase SHA-256")

    try:
        comment = provider.get_issue_comment(repository, comment_id)
    except Exception as exc:
        raise ValueError(
            f"GitHub approval verification unavailable; fail-close: {exc}"
        ) from exc
    comment = _require_object(comment, "GitHub approval comment")
    if comment.get("id") != comment_id or type(comment.get("id")) is not int:
        raise ValueError("GitHub approval comment ID mismatch")
    if _comment_issue_number(comment, repository) != issue_number:
        raise ValueError("GitHub approval comment belongs to a different Issue")

    user = _require_object(comment.get("user"), "GitHub approval comment author")
    login_raw = user.get("login")
    author_type = user.get("type")
    if not isinstance(login_raw, str) or not login_raw.strip():
        raise ValueError("GitHub approval comment author login is invalid")
    login = login_raw.strip()
    normalized_login = login.lower()
    approvers = allowlist.get("approvers") if isinstance(allowlist, dict) else None
    denied_patterns = (
        allowlist.get("denied_login_patterns", []) if isinstance(allowlist, dict) else None
    )
    if not isinstance(approvers, set) or not all(isinstance(item, str) for item in approvers):
        raise ValueError("GitHub approval allowlist is invalid; fail-close")
    if not isinstance(denied_patterns, list) or not all(
        isinstance(item, str) and item for item in denied_patterns
    ):
        raise ValueError("GitHub approval denied patterns are invalid; fail-close")
    automation_patterns = set(AUTOMATION_LOGIN_PATTERNS)
    automation_patterns.update(item.lower() for item in denied_patterns)
    if author_type != "User" or any(pattern in normalized_login for pattern in automation_patterns):
        raise ValueError("Codex or automation actor cannot approve research")
    if normalized_login not in {item.lower() for item in approvers}:
        raise ValueError(f"GitHub approval author {login!r} is not in the allowlist")

    body = comment.get("body")
    if not isinstance(body, str):
        raise ValueError("GitHub approval comment body is invalid")
    expected_body = f"{approval_keyword} {approval_digest}"
    if body != expected_body:
        raise ValueError(
            f"GitHub approval comment digest or format mismatch; expected {expected_body!r}"
        )
    created_at = comment.get("created_at")
    updated_at = comment.get("updated_at")
    if not all(isinstance(value, str) and value for value in (created_at, updated_at)):
        raise ValueError("GitHub approval comment metadata is incomplete")
    html_url = _validate_comment_html_url(
        comment.get("html_url"),
        repository,
        issue_number,
        comment_id,
    )
    evidence = {
        "approval_type": approval_keyword,
        "approval_digest": approval_digest,
        "repository": repository,
        "issue_number": issue_number,
        "comment_id": comment_id,
        "url": html_url,
        "author": login,
        "author_type": author_type,
        "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "created_at": created_at,
        "updated_at": updated_at,
    }
    if tuple(evidence) != COMMENT_EVIDENCE_FIELDS:
        raise AssertionError("internal GitHub comment evidence schema mismatch")
    return evidence


def reverify_same_approval(
    *,
    evidence: dict[str, Any],
    provider: GitHubApprovalProvider,
    allowlist: dict[str, Any],
    expected_approval_type: str,
    expected_approval_digest: str,
) -> dict[str, Any]:
    """Re-fetch one prior grant and reject every immutable-field change or deletion."""

    if not isinstance(evidence, dict) or set(evidence) != set(COMMENT_EVIDENCE_FIELDS):
        raise ValueError("stored approval evidence is missing or invalid; fail-close")
    if evidence.get("approval_type") != expected_approval_type:
        raise ValueError("stored approval keyword differs from the required grant; fail-close")
    if evidence.get("approval_digest") != expected_approval_digest:
        raise ValueError("stored approval digest differs from the required scope; fail-close")
    current = verify_approval_comment(
        provider=provider,
        allowlist=allowlist,
        repository=evidence["repository"],
        issue_number=evidence["issue_number"],
        comment_id=evidence["comment_id"],
        approval_keyword=expected_approval_type,
        approval_digest=expected_approval_digest,
    )
    for field in COMMENT_EVIDENCE_FIELDS:
        if current[field] != evidence[field]:
            raise ValueError(
                f"GitHub approval comment changed after approval ({field}); fail-close"
            )
    return current


__all__ = [
    "APPROVERS_PATH",
    "APPROVAL_KEYWORDS",
    "COMMENT_EVIDENCE_FIELDS",
    "DEFAULT_BASE_BRANCH",
    "DEFAULT_REPOSITORY",
    "GITHUB_API_URL",
    "GITHUB_TRUST_EVIDENCE_FIELDS",
    "REMOTE_FILE_EVIDENCE_FIELDS",
    "GitHubApprovalProvider",
    "GitHubRestApprovalProvider",
    "fetch_github_file_at_commit",
    "reverify_same_approval",
    "utc_now",
    "verify_approval_comment",
    "verify_github_file_at_commit",
    "verify_github_main_head_unchanged",
    "verify_github_trust",
]
