from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import github_approval


REPOSITORY = github_approval.DEFAULT_REPOSITORY
BASE_BRANCH = github_approval.DEFAULT_BASE_BRANCH
BASE_COMMIT = "a" * 40
MAIN_COMMIT = "b" * 40
APPROVERS_BLOB_SHA = "c" * 40
APPROVAL_DIGEST = "d" * 64
ISSUE_NUMBER = 17
COMMENT_ID = 1001
APPROVER = "kazuponbaseball-cell"
VERIFIED_AT = "2026-07-31T01:02:03Z"


def approvers_bytes(*logins: str) -> bytes:
    payload = {
        "schema_version": 1,
        "approvers": [
            {"login": login, "role": "repository_owner"} for login in logins
        ],
        "denied_login_patterns": [
            "bot",
            "codex",
            "automation",
            "github-actions",
            "dependabot",
        ],
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def contents_payload(content: bytes, *, blob_sha: str = APPROVERS_BLOB_SHA) -> dict[str, Any]:
    return {
        "type": "file",
        "path": github_approval.APPROVERS_PATH,
        "sha": blob_sha,
        "encoding": "base64",
        "content": base64.b64encode(content).decode("ascii"),
    }


class FakeGitHubApprovalProvider:
    def __init__(self) -> None:
        self.repository_payload: dict[str, Any] = {
            "full_name": REPOSITORY,
            "default_branch": BASE_BRANCH,
        }
        self.ref_payload: dict[str, Any] = {
            "ref": "refs/heads/main",
            "object": {"type": "commit", "sha": MAIN_COMMIT},
        }
        self.compare_payload: dict[str, Any] = self._compare_payload(
            BASE_COMMIT,
            MAIN_COMMIT,
            status="ahead",
            merge_base=BASE_COMMIT,
        )
        self.file_payloads: dict[str, dict[str, Any]] = {
            BASE_COMMIT: contents_payload(approvers_bytes(APPROVER)),
            MAIN_COMMIT: contents_payload(
                approvers_bytes("malicious-main-user"),
                blob_sha="e" * 40,
            ),
        }
        self.comments: dict[int, dict[str, Any]] = {}
        self.failures: dict[str, Exception] = {}
        self.calls: list[tuple[Any, ...]] = []

    @staticmethod
    def _compare_payload(
        base: str,
        head: str,
        *,
        status: str,
        merge_base: str,
    ) -> dict[str, Any]:
        return {
            "url": (
                f"{github_approval.GITHUB_API_URL}/repos/{REPOSITORY}/compare/"
                f"{base}...{head}"
            ),
            "status": status,
            "base_commit": {"sha": base},
            "merge_base_commit": {"sha": merge_base},
        }

    def set_comparison(
        self,
        *,
        head: str,
        status: str,
        merge_base: str,
    ) -> None:
        self.ref_payload["object"]["sha"] = head
        self.compare_payload = self._compare_payload(
            BASE_COMMIT,
            head,
            status=status,
            merge_base=merge_base,
        )

    def _maybe_fail(self, operation: str) -> None:
        if operation in self.failures:
            raise self.failures[operation]

    def get_repository(self, repository: str) -> dict[str, Any]:
        self.calls.append(("repository", repository))
        self._maybe_fail("repository")
        return copy.deepcopy(self.repository_payload)

    def get_branch_ref(self, repository: str, branch: str) -> dict[str, Any]:
        self.calls.append(("branch_ref", repository, branch))
        self._maybe_fail("branch_ref")
        return copy.deepcopy(self.ref_payload)

    def compare_commits(
        self,
        repository: str,
        base_commit: str,
        head_commit: str,
    ) -> dict[str, Any]:
        self.calls.append(("compare", repository, base_commit, head_commit))
        self._maybe_fail("compare")
        return copy.deepcopy(self.compare_payload)

    def get_file_contents(
        self,
        repository: str,
        path: str,
        ref: str,
    ) -> dict[str, Any]:
        self.calls.append(("contents", repository, path, ref))
        self._maybe_fail("contents")
        if ref not in self.file_payloads:
            raise KeyError(f"file unavailable at {ref}")
        return copy.deepcopy(self.file_payloads[ref])

    def get_issue_comment(self, repository: str, comment_id: int) -> dict[str, Any]:
        self.calls.append(("comment", repository, comment_id))
        self._maybe_fail("comment")
        if comment_id not in self.comments:
            raise KeyError(f"comment {comment_id} not found")
        return copy.deepcopy(self.comments[comment_id])

    def add_comment(
        self,
        *,
        comment_id: int = COMMENT_ID,
        issue_number: int = ISSUE_NUMBER,
        author: str = APPROVER,
        author_type: str = "User",
        keyword: str = "APPROVED_TO_PREPARE",
        digest: str = APPROVAL_DIGEST,
    ) -> None:
        body = f"{keyword} {digest}"
        self.comments[comment_id] = {
            "id": comment_id,
            "html_url": (
                f"https://github.com/{REPOSITORY}/issues/{issue_number}"
                f"#issuecomment-{comment_id}"
            ),
            "issue_url": (
                f"https://api.github.com/repos/{REPOSITORY}/issues/{issue_number}"
            ),
            "user": {"login": author, "type": author_type},
            "created_at": "2026-07-31T00:00:00Z",
            "updated_at": "2026-07-31T00:00:00Z",
            "body": body,
        }


class GitHubTrustTests(unittest.TestCase):
    def verify(
        self,
        provider: FakeGitHubApprovalProvider,
        *,
        repository: str = REPOSITORY,
        branch: str = BASE_BRANCH,
        base: str = BASE_COMMIT,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return github_approval.verify_github_trust(
            provider=provider,
            repository=repository,
            base_branch=branch,
            base_commit=base,
            clock=lambda: VERIFIED_AT,
        )

    def test_remote_main_ancestor_loads_only_base_commit_allowlist_and_evidence(self) -> None:
        provider = FakeGitHubApprovalProvider()
        allowlist, evidence = self.verify(provider)

        self.assertEqual(allowlist["approvers"], {APPROVER.lower()})
        self.assertNotIn("malicious-main-user", allowlist["approvers"])

        self.assertEqual(
            provider.calls,
            [
                ("repository", REPOSITORY),
                ("branch_ref", REPOSITORY, BASE_BRANCH),
                ("compare", REPOSITORY, BASE_COMMIT, MAIN_COMMIT),
                ("contents", REPOSITORY, github_approval.APPROVERS_PATH, BASE_COMMIT),
            ],
        )
        expected_content = approvers_bytes(APPROVER)
        self.assertEqual(
            evidence,
            {
                "repository_full_name": REPOSITORY,
                "base_branch": BASE_BRANCH,
                "verified_current_main_sha": MAIN_COMMIT,
                "verified_base_commit": BASE_COMMIT,
                "compare_url": (
                    f"https://api.github.com/repos/{REPOSITORY}/compare/"
                    f"{BASE_COMMIT}...{MAIN_COMMIT}"
                ),
                "compare_status": "ahead",
                "merge_base_sha": BASE_COMMIT,
                "approvers_blob_sha": APPROVERS_BLOB_SHA,
                "approvers_content_sha256": hashlib.sha256(expected_content).hexdigest(),
                "verification_time": VERIFIED_AT,
            },
        )

    def test_base_commit_policy_file_is_bound_to_raw_content_hash(self) -> None:
        provider = FakeGitHubApprovalProvider()
        path = "research/INFRASTRUCTURE_GATE.json"
        content = b'{"gate_kind":"infrastructure_safety_v1"}\n'
        payload = contents_payload(content, blob_sha="f" * 40)
        payload["path"] = path
        provider.file_payloads[BASE_COMMIT] = payload
        expected_hash = hashlib.sha256(content).hexdigest()

        observed, evidence = github_approval.verify_github_file_at_commit(
            provider=provider,
            repository=REPOSITORY,
            path=path,
            ref=BASE_COMMIT,
            expected_content_sha256=expected_hash,
        )
        self.assertEqual(observed, content)
        self.assertEqual(
            evidence,
            {
                "path": path,
                "ref": BASE_COMMIT,
                "blob_sha": "f" * 40,
                "content_sha256": expected_hash,
            },
        )

        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            github_approval.verify_github_file_at_commit(
                provider=provider,
                repository=REPOSITORY,
                path=path,
                ref=BASE_COMMIT,
                expected_content_sha256="0" * 64,
            )

    def test_identical_base_and_current_main_is_allowed(self) -> None:
        provider = FakeGitHubApprovalProvider()
        provider.set_comparison(
            head=BASE_COMMIT,
            status="identical",
            merge_base=BASE_COMMIT,
        )
        allowlist, evidence = self.verify(provider)
        self.assertEqual(allowlist["approvers"], {APPROVER.lower()})
        self.assertEqual(evidence["compare_status"], "identical")
        self.assertEqual(evidence["verified_current_main_sha"], BASE_COMMIT)

    def test_local_origin_and_local_approvers_cannot_affect_remote_decision(self) -> None:
        provider = FakeGitHubApprovalProvider()
        with tempfile.TemporaryDirectory() as directory:
            local_approvers = Path(directory) / "research" / "APPROVERS.json"
            local_approvers.parent.mkdir(parents=True)
            local_approvers.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "approvers": [{"login": "malicious-local-user"}],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                subprocess,
                "run",
                side_effect=AssertionError("local git must not be consulted"),
            ):
                allowlist, _evidence = self.verify(provider)
        self.assertEqual(allowlist["approvers"], {APPROVER.lower()})
        self.assertNotIn("malicious-local-user", allowlist["approvers"])

    def test_non_ancestor_and_diverged_comparisons_fail_closed(self) -> None:
        cases = (
            ("behind", MAIN_COMMIT, "not an ancestor"),
            ("diverged", "f" * 40, "not an ancestor"),
            ("ahead", "f" * 40, "not the GitHub main merge base"),
        )
        for status, merge_base, expected in cases:
            with self.subTest(status=status, merge_base=merge_base):
                provider = FakeGitHubApprovalProvider()
                provider.set_comparison(
                    head=MAIN_COMMIT,
                    status=status,
                    merge_base=merge_base,
                )
                with self.assertRaisesRegex(ValueError, expected):
                    self.verify(provider)

    def test_every_github_trust_api_failure_fails_closed(self) -> None:
        for operation in ("repository", "branch_ref", "compare", "contents"):
            with self.subTest(operation=operation):
                provider = FakeGitHubApprovalProvider()
                provider.failures[operation] = OSError(f"{operation} unavailable")
                with self.assertRaisesRegex(ValueError, "unavailable; fail-close"):
                    self.verify(provider)

    def test_repository_branch_ref_and_compare_mismatches_fail_closed(self) -> None:
        mutations = (
            (
                lambda provider: provider.repository_payload.__setitem__(
                    "full_name", "attacker/repository"
                ),
                "full_name mismatch",
            ),
            (
                lambda provider: provider.repository_payload.__setitem__(
                    "default_branch", "attacker-main"
                ),
                "default branch mismatch",
            ),
            (
                lambda provider: provider.ref_payload.__setitem__(
                    "ref", "refs/heads/attacker"
                ),
                "main ref mismatch",
            ),
            (
                lambda provider: provider.compare_payload["base_commit"].__setitem__(
                    "sha", "f" * 40
                ),
                "base commit mismatch",
            ),
            (
                lambda provider: provider.compare_payload.__setitem__(
                    "url", "https://api.github.com/repos/attacker/repo/compare/a...b"
                ),
                "compare URL mismatch",
            ),
        )
        for mutate, expected in mutations:
            with self.subTest(expected=expected):
                provider = FakeGitHubApprovalProvider()
                mutate(provider)
                with self.assertRaisesRegex(ValueError, expected):
                    self.verify(provider)

        with self.assertRaisesRegex(ValueError, "repository mismatch"):
            self.verify(FakeGitHubApprovalProvider(), repository="attacker/repository")
        with self.assertRaisesRegex(ValueError, "base branch mismatch"):
            self.verify(FakeGitHubApprovalProvider(), branch="attacker-main")
        with self.assertRaisesRegex(ValueError, "proposal base commit"):
            self.verify(FakeGitHubApprovalProvider(), base="not-a-sha")

    def test_missing_or_invalid_remote_approvers_content_fails_closed(self) -> None:
        cases: tuple[tuple[str, Any, str], ...] = (
            (
                "wrong_path",
                lambda payload: payload.__setitem__("path", "research/OTHER.json"),
                "path or type mismatch",
            ),
            (
                "wrong_encoding",
                lambda payload: payload.__setitem__("encoding", "none"),
                "base64 content encoding",
            ),
            (
                "invalid_blob",
                lambda payload: payload.__setitem__("sha", "not-a-sha"),
                "blob SHA",
            ),
            (
                "invalid_base64",
                lambda payload: payload.__setitem__("content", "%%%"),
                "invalid base64",
            ),
            (
                "invalid_json",
                lambda payload: payload.__setitem__(
                    "content", base64.b64encode(b"{invalid").decode("ascii")
                ),
                "invalid JSON",
            ),
            (
                "invalid_schema",
                lambda payload: payload.__setitem__(
                    "content",
                    base64.b64encode(b'{"schema_version":2,"approvers":[]}').decode(
                        "ascii"
                    ),
                ),
                "schema_version must be 1",
            ),
        )
        for name, mutate, expected in cases:
            with self.subTest(name=name):
                provider = FakeGitHubApprovalProvider()
                mutate(provider.file_payloads[BASE_COMMIT])
                with self.assertRaisesRegex(ValueError, expected):
                    self.verify(provider)

        provider = FakeGitHubApprovalProvider()
        provider.file_payloads.pop(BASE_COMMIT)
        with self.assertRaisesRegex(ValueError, "APPROVERS.json.*unavailable"):
            self.verify(provider)


class ApprovalCommentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = FakeGitHubApprovalProvider()
        self.provider.add_comment()
        self.allowlist = {
            "approvers": {APPROVER.lower()},
            "denied_login_patterns": ["bot", "codex", "automation"],
        }

    def verify_comment(self) -> dict[str, Any]:
        return github_approval.verify_approval_comment(
            provider=self.provider,
            allowlist=self.allowlist,
            repository=REPOSITORY,
            issue_number=ISSUE_NUMBER,
            comment_id=COMMENT_ID,
            approval_keyword="APPROVED_TO_PREPARE",
            approval_digest=APPROVAL_DIGEST,
        )

    def test_comment_evidence_includes_exact_body_and_all_immutable_fields(self) -> None:
        evidence = self.verify_comment()
        body = f"APPROVED_TO_PREPARE {APPROVAL_DIGEST}"
        self.assertEqual(set(evidence), set(github_approval.COMMENT_EVIDENCE_FIELDS))
        self.assertEqual(evidence["body"], body)
        self.assertEqual(
            evidence["body_sha256"],
            hashlib.sha256(body.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(evidence["comment_id"], COMMENT_ID)
        self.assertEqual(evidence["issue_number"], ISSUE_NUMBER)

    def test_comment_deletion_and_api_failure_fail_closed(self) -> None:
        evidence = self.verify_comment()
        self.provider.comments.pop(COMMENT_ID)
        with self.assertRaisesRegex(ValueError, "unavailable; fail-close"):
            github_approval.reverify_same_approval(
                evidence=evidence,
                provider=self.provider,
                allowlist=self.allowlist,
                expected_approval_type="APPROVED_TO_PREPARE",
                expected_approval_digest=APPROVAL_DIGEST,
            )

    def test_reverification_rejects_every_mutable_github_comment_field(self) -> None:
        mutations = (
            (
                "comment_id",
                lambda comment: comment.__setitem__("id", COMMENT_ID + 1),
            ),
            (
                "issue_number",
                lambda comment: comment.__setitem__(
                    "issue_url",
                    f"https://api.github.com/repos/{REPOSITORY}/issues/{ISSUE_NUMBER + 1}",
                ),
            ),
            (
                "url",
                lambda comment: comment.__setitem__(
                    "html_url",
                    f"https://github.com/{REPOSITORY}/issues/{ISSUE_NUMBER}"
                    f"#issuecomment-{COMMENT_ID + 1}",
                ),
            ),
            (
                "author",
                lambda comment: comment["user"].__setitem__("login", "second-owner"),
            ),
            (
                "author_type",
                lambda comment: comment["user"].__setitem__("type", "Bot"),
            ),
            (
                "body",
                lambda comment: comment.__setitem__(
                    "body", f"APPROVED_TO_RUN {'e' * 64}"
                ),
            ),
            (
                "created_at",
                lambda comment: comment.__setitem__(
                    "created_at", "2026-07-31T00:01:00Z"
                ),
            ),
            (
                "updated_at",
                lambda comment: comment.__setitem__(
                    "updated_at", "2026-07-31T00:01:00Z"
                ),
            ),
        )
        for field, mutate in mutations:
            with self.subTest(field=field):
                provider = FakeGitHubApprovalProvider()
                provider.add_comment()
                allowlist = copy.deepcopy(self.allowlist)
                allowlist["approvers"].add("second-owner")
                evidence = github_approval.verify_approval_comment(
                    provider=provider,
                    allowlist=allowlist,
                    repository=REPOSITORY,
                    issue_number=ISSUE_NUMBER,
                    comment_id=COMMENT_ID,
                    approval_keyword="APPROVED_TO_PREPARE",
                    approval_digest=APPROVAL_DIGEST,
                )
                mutate(provider.comments[COMMENT_ID])
                with self.assertRaises(ValueError):
                    github_approval.reverify_same_approval(
                        evidence=evidence,
                        provider=provider,
                        allowlist=allowlist,
                        expected_approval_type="APPROVED_TO_PREPARE",
                        expected_approval_digest=APPROVAL_DIGEST,
                    )

    def test_stored_body_or_body_hash_tampering_is_rejected(self) -> None:
        for field, value in (
            ("body", "tampered stored body"),
            ("body_sha256", "0" * 64),
            ("updated_at", "2026-07-31T00:02:00Z"),
        ):
            with self.subTest(field=field):
                evidence = self.verify_comment()
                evidence[field] = value
                with self.assertRaisesRegex(ValueError, "changed after approval"):
                    github_approval.reverify_same_approval(
                        evidence=evidence,
                        provider=self.provider,
                        allowlist=self.allowlist,
                        expected_approval_type="APPROVED_TO_PREPARE",
                        expected_approval_digest=APPROVAL_DIGEST,
                    )

    def test_stored_keyword_or_digest_tampering_is_rejected(self) -> None:
        for field, value, expected in (
            ("approval_type", "APPROVED_TO_RUN", "stored approval keyword"),
            ("approval_digest", "e" * 64, "stored approval digest"),
        ):
            with self.subTest(field=field):
                evidence = self.verify_comment()
                evidence[field] = value
                with self.assertRaisesRegex(ValueError, expected):
                    github_approval.reverify_same_approval(
                        evidence=evidence,
                        provider=self.provider,
                        allowlist=self.allowlist,
                        expected_approval_type="APPROVED_TO_PREPARE",
                        expected_approval_digest=APPROVAL_DIGEST,
                    )


if __name__ == "__main__":
    unittest.main()
