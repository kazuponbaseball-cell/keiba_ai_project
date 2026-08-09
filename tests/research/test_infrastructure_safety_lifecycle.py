from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SCRIPTS = REPO_ROOT / "scripts" / "research"
if str(RESEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS))

import infrastructure_safety_contract as contract
import update_registry


REPOSITORY = "kazuponbaseball-cell/keiba_ai_project"
APPROVER = "human-infrastructure-approver"
ISSUE_NUMBER = 73
EXPERIMENT_ID = "GOV-SYNTHETIC-LIFECYCLE-001"
BASE_COMMIT = "a" * 40
EXECUTION_COMMIT = "b" * 40
EXPECTED_PATHS = [
    "scripts/research/example_infrastructure.py",
    "research/infra_tests/test_example_infrastructure.py",
]
LIFECYCLE_PATHS = list(contract.infrastructure_lifecycle_paths(EXPERIMENT_ID))
ALL_CHANGED_PATHS = [*EXPECTED_PATHS, *LIFECYCLE_PATHS]
EXECUTION_BLOBS = {
    EXPECTED_PATHS[0]: "1" * 40,
    EXPECTED_PATHS[1]: "2" * 40,
    **{
        path: str(index) * 40
        for index, path in enumerate(LIFECYCLE_PATHS, start=3)
    },
}


def file_ref(root: Path, path: str) -> dict[str, str]:
    return {
        "path": path,
        "sha256": contract.sha256_file(root / path),
    }


def sample_proposal(policy_sha256: str) -> dict[str, Any]:
    return {
        "contract": dict(contract.PROPOSAL_CONTRACT),
        "gate_kind": contract.GATE_KIND,
        "experiment_id": EXPERIMENT_ID,
        "title": "Exercise the infrastructure safety lifecycle",
        "change_hypothesis": (
            "The v3 gate preserves two-stage approval while allowing only one "
            "bounded synthetic lifecycle."
        ),
        "null_hypothesis": (
            "The lifecycle permits real data, shadow, reused approvals, or "
            "scope-kind drift."
        ),
        "safety_objective": (
            "Verify fail-closed infrastructure transitions with no network or "
            "real-data access."
        ),
        "failure_modes": [
            "Approval comment identifiers are reused",
            "Run execution kind differs from the frozen scope",
        ],
        "in_scope": [
            "Injected GitHub approval fixtures",
            "Synthetic registry lifecycle transitions",
        ],
        "out_of_scope": [
            "External API and real-data execution",
            "Production, purchase, shadow, order, and notification behavior",
        ],
        "expected_changed_paths": list(EXPECTED_PATHS),
        "input_classes": ["git_tracked_contract", "synthetic_fixture"],
        "source_as_of": "2026-08-09T00:00:00+09:00",
        "lineage_hash_requirements": [
            "Base-to-execution Git blob manifest SHA-256",
            "Gate policy raw file SHA-256",
        ],
        "test_matrix": [
            {
                "test_id": "positive-synthetic-lifecycle",
                "kind": "positive",
                "assertion": "The approved synthetic lifecycle reaches review required.",
            },
            {
                "test_id": "negative-real-data-shadow",
                "kind": "negative",
                "assertion": "Real-data and shadow transitions fail closed.",
            },
            {
                "test_id": "compat-v2-comment-namespace",
                "kind": "backward_compatibility",
                "assertion": "Approval comment IDs remain single-use across v2 and v3.",
            },
        ],
        "primary_metric": {
            "name": "pre_registered_infrastructure_contract_pass_fraction",
            "direction": "higher_is_better",
            "required_value": 1.0,
        },
        "required_effect": {
            "all_pre_registered_tests_pass": True,
            "external_api_calls": 0,
            "network_calls": 0,
            "real_data_rows": 0,
            "roi_calculations": 0,
            "production_changes": 0,
            "buy_order_notification_side_effects": 0,
        },
        "rejection_gate": [
            "Reject any failed hard check",
            "Reject any run-kind or approval mismatch",
        ],
        "stop_conditions": [
            "Stop on any non-synthetic execution kind",
            "Stop on any safety capability change",
        ],
        "compute_budget": {
            "maximum_runtime_minutes": 5,
            "network_calls": 0,
            "external_api_calls": 0,
            "real_data_rows": 0,
            "random_seed": 20260809,
        },
        "allowed_variant_count": 1,
        "allowed_threshold_search_count": 0,
        "base_commit": BASE_COMMIT,
        "gate_policy": {
            "path": "research/INFRASTRUCTURE_GATE.json",
            "sha256": policy_sha256,
        },
        "compatibility_contract": {
            "legacy_roi_proposal_digest_unchanged": True,
            "legacy_queue_schema_v2_readable": True,
            "legacy_event_schema_v2_readable": True,
        },
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def sample_run_scope(root: Path, proposal: dict[str, Any]) -> dict[str, Any]:
    changed_path_manifest = {
        "schema_version": 1,
        "base_commit": BASE_COMMIT,
        "execution_commit": EXECUTION_COMMIT,
        "entries": sorted(
            [
                {
                    "path": path,
                    "change_type": "added",
                    "base_blob_sha": None,
                    "execution_blob_sha": EXECUTION_BLOBS[path],
                }
                for path in ALL_CHANGED_PATHS
            ],
            key=lambda entry: entry["path"],
        ),
    }
    command_plan = [
        {
            "template_id": "python_unittest_module_v1",
            "parameters": {
                "module": "research.infra_tests.test_example_infrastructure"
            },
        },
    ]
    python = contract.bound_python_executable()
    exact_argv = [
        [
            python,
            "-B",
            "-I",
            "-S",
            "-m",
            "unittest",
            "discover",
            "-s",
            "research/infra_tests",
            "-p",
            "test_example_infrastructure.py",
            "-v",
        ],
    ]
    return {
        "contract": dict(contract.RUN_CONTRACT),
        "gate_kind": contract.GATE_KIND,
        "proposal_scope": proposal,
        "proposal_scope_digest": contract.canonical_digest(proposal),
        "execution_commit_sha": EXECUTION_COMMIT,
        "config_hashes": [file_ref(root, "research/INFRASTRUCTURE_GATE.json")],
        "synthetic_input_hashes": [
            file_ref(root, "research/synthetic/input.json")
        ],
        "changed_path_manifest": changed_path_manifest,
        "changed_path_manifest_digest": contract.canonical_digest(
            changed_path_manifest
        ),
        "dependency_environment_manifest": file_ref(
            root, "research/synthetic/environment.json"
        ),
        "seed": 20260809,
        "command_plan": command_plan,
        "exact_execution_argv": exact_argv,
        "execution_context": {
            "working_directory": "repository_root",
            "inherit_environment": False,
            "environment": {},
            "timeout_seconds": proposal["compute_budget"]["maximum_runtime_minutes"] * 60,
            "network_access": False,
            "credential_environment_access": False,
            "filesystem_write_paths": [],
        },
        "execution_kind": "synthetic",
        "external_api_calls": False,
        "network_calls": 0,
        "actual_codex_dispatch": False,
        "real_data_execution": False,
        "roi_calculation": False,
        "production_change": False,
        "credential_access": False,
        "notification_side_effects": False,
        "order_side_effects": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def approval_evidence(
    *,
    comment_id: int,
    approval_type: str,
    approval_digest: str,
    experiment_id: str,
) -> dict[str, Any]:
    del experiment_id
    body = f"{approval_type} {approval_digest}"
    return {
        "approval_type": approval_type,
        "approval_digest": approval_digest,
        "repository": REPOSITORY,
        "issue_number": ISSUE_NUMBER,
        "comment_id": comment_id,
        "url": (
            f"https://github.com/{REPOSITORY}/issues/{ISSUE_NUMBER}"
            f"#issuecomment-{comment_id}"
        ),
        "author": APPROVER,
        "author_type": "User",
        "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "created_at": "2026-08-09T00:00:00Z",
        "updated_at": "2026-08-09T00:00:00Z",
    }


class FakeGitHubProvider:
    """In-memory read-only provider; no method performs network I/O."""

    def __init__(self, policy_content: bytes) -> None:
        self.policy_content = policy_content
        self.registry_content = b""
        self.current_main = BASE_COMMIT
        self.branch_calls = 0
        self.move_main_on_recheck = False
        self.approvers_content = json.dumps(
            {
                "schema_version": 1,
                "approvers": [{"login": APPROVER}],
                "denied_login_patterns": ["bot", "codex", "automation"],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self.comments: dict[int, dict[str, Any]] = {}
        self.file_calls: list[tuple[str, str]] = []
        self.comment_calls: list[int] = []

    def get_repository(self, repository: str) -> dict[str, Any]:
        if repository != REPOSITORY:
            raise AssertionError(f"unexpected repository: {repository}")
        return {"full_name": REPOSITORY, "default_branch": "main"}

    def get_branch_ref(self, repository: str, branch: str) -> dict[str, Any]:
        if repository != REPOSITORY or branch != "main":
            raise AssertionError(f"unexpected branch lookup: {repository} {branch}")
        self.branch_calls += 1
        if self.move_main_on_recheck and self.branch_calls % 2 == 0:
            self.current_main = "f" * 40
        return {
            "ref": "refs/heads/main",
            "object": {"type": "commit", "sha": self.current_main},
        }

    def compare_commits(
        self,
        repository: str,
        base_commit: str,
        head_commit: str,
    ) -> dict[str, Any]:
        if (repository, base_commit, head_commit) != (
            REPOSITORY,
            BASE_COMMIT,
            self.current_main,
        ):
            raise AssertionError("unexpected comparison")
        return {
            "status": "identical" if head_commit == BASE_COMMIT else "ahead",
            "url": (
                f"https://api.github.com/repos/{REPOSITORY}/compare/"
                f"{BASE_COMMIT}...{head_commit}"
            ),
            "base_commit": {"sha": BASE_COMMIT},
            "merge_base_commit": {"sha": BASE_COMMIT},
        }

    def get_file_contents(
        self,
        repository: str,
        path: str,
        ref: str,
    ) -> dict[str, Any]:
        if repository != REPOSITORY:
            raise AssertionError(f"unexpected contents lookup: {repository} {ref}")
        self.file_calls.append((path, ref))
        if path == "research/APPROVERS.json":
            if ref != BASE_COMMIT:
                raise AssertionError(f"unexpected approvers ref: {ref}")
            content = self.approvers_content
            blob_sha = "c" * 40
        elif path == "research/INFRASTRUCTURE_GATE.json":
            if ref != BASE_COMMIT:
                raise AssertionError(f"unexpected policy ref: {ref}")
            content = self.policy_content
            blob_sha = "d" * 40
        elif path == "research/REGISTRY.jsonl":
            if ref != self.current_main:
                raise AssertionError(f"unexpected registry ref: {ref}")
            content = self.registry_content
            blob_sha = "e" * 40
        else:
            raise AssertionError(f"unexpected contents path: {path}")
        return {
            "type": "file",
            "path": path,
            "encoding": "base64",
            "sha": blob_sha,
            "content": base64.b64encode(content).decode("ascii"),
        }

    def get_issue_comment(self, repository: str, comment_id: int) -> dict[str, Any]:
        if repository != REPOSITORY:
            raise AssertionError(f"unexpected comment repository: {repository}")
        self.comment_calls.append(comment_id)
        if comment_id not in self.comments:
            raise KeyError(f"comment {comment_id} not found")
        return copy.deepcopy(self.comments[comment_id])

    def add_comment(self, comment_id: int, keyword: str, digest: str) -> None:
        self.comments[comment_id] = {
            "id": comment_id,
            "html_url": (
                f"https://github.com/{REPOSITORY}/issues/{ISSUE_NUMBER}"
                f"#issuecomment-{comment_id}"
            ),
            "issue_url": (
                f"https://api.github.com/repos/{REPOSITORY}/issues/{ISSUE_NUMBER}"
            ),
            "user": {"login": APPROVER, "type": "User"},
            "created_at": "2026-08-09T00:00:00Z",
            "updated_at": "2026-08-09T00:00:00Z",
            "body": f"{keyword} {digest}",
        }


class InfrastructureSafetyLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.policy_path = self.root / "research/INFRASTRUCTURE_GATE.json"
        self.policy_path.parent.mkdir(parents=True)
        source_policy = REPO_ROOT / "research/INFRASTRUCTURE_GATE.json"
        self.policy_path.write_bytes(source_policy.read_bytes())
        self.policy, self.policy_sha256 = contract.load_gate_policy(self.policy_path)
        self.proposal = contract.normalize_infrastructure_proposal(
            sample_proposal(self.policy_sha256),
            policy=self.policy,
            policy_sha256=self.policy_sha256,
        )

        for index, path in enumerate(EXPECTED_PATHS):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"# synthetic lifecycle file {index}\n", encoding="utf-8")
        fixtures = {
            "research/synthetic/input.json": {
                "schema_version": 1,
                "fixture_kind": "synthetic_input",
                "synthetic": True,
                "provenance": "code_owned_synthetic_fixture_v1",
                "contains_real_data": False,
                "contains_credentials": False,
                "payload": {"synthetic": True},
            },
            "research/synthetic/environment.json": {
                "schema_version": 1,
                "fixture_kind": "dependency_environment",
                "synthetic": True,
                "provenance": "code_owned_synthetic_fixture_v1",
                "contains_real_data": False,
                "contains_credentials": False,
                "payload": contract.expected_environment_payload(),
            },
        }
        for path, payload in fixtures.items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        proposal_path = (
            self.root / f"research/scopes/{EXPERIMENT_ID}.proposal.json"
        )
        proposal_path.parent.mkdir(parents=True, exist_ok=True)
        proposal_path.write_text(
            contract.canonical_json_text(self.proposal) + "\n",
            encoding="utf-8",
        )
        queue = contract.build_infrastructure_queue(
            proposal_scope=self.proposal,
            proposal_scope_file=(
                f"research/scopes/{EXPERIMENT_ID}.proposal.json"
            ),
            experiment_markdown=(
                f"research/experiments/{EXPERIMENT_ID}.md"
            ),
            owner="human-owner",
            created_at="2026-08-09T00:00:00Z",
            notes="synthetic lifecycle fixture",
            policy=self.policy,
        )
        queue_path = self.root / f"research/queue/{EXPERIMENT_ID}.json"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text(
            contract.canonical_json_text(queue) + "\n",
            encoding="utf-8",
        )
        experiment_path = (
            self.root / f"research/experiments/{EXPERIMENT_ID}.md"
        )
        experiment_path.parent.mkdir(parents=True, exist_ok=True)
        experiment_path.write_text(
            "# Synthetic infrastructure lifecycle fixture\n",
            encoding="utf-8",
        )

        self.registry_path = self.root / "research/REGISTRY.jsonl"
        self.provider = FakeGitHubProvider(self.policy_path.read_bytes())
        self.commit_calls: list[Path] = []
        self.worktree_calls: list[tuple[Path, set[str]]] = []
        self.diff_calls: list[tuple[Path, str]] = []

    def _execution_commit_provider(self, root: Path) -> str:
        self.commit_calls.append(root)
        return EXECUTION_COMMIT

    def _worktree_verifier(self, root: Path, allowed_paths: set[str]) -> None:
        self.worktree_calls.append((root, set(allowed_paths)))

    def _diff_verifier(self, root: Path, run_scope: dict[str, Any]) -> None:
        self.diff_calls.append((root, contract.canonical_digest(run_scope)))

    def _argv(self, status: str) -> list[str]:
        return [
            EXPERIMENT_ID,
            status,
            "--root",
            str(self.root),
            "--actor",
            "lifecycle-test-caller",
        ]

    def _approval_args(self, comment_id: int) -> list[str]:
        return [
            "--github-repository",
            REPOSITORY,
            "--issue-number",
            str(ISSUE_NUMBER),
            "--approval-comment-id",
            str(comment_id),
        ]

    def _merge_registry_to_fake_main(self) -> None:
        self.provider.registry_content = self.registry_path.read_bytes()
        self.provider.current_main = hashlib.sha256(
            self.provider.registry_content
        ).hexdigest()[:40]

    def _append(
        self,
        status: str,
        *,
        extra: list[str] | None = None,
        merge_to_main: bool = True,
    ) -> dict[str, Any]:
        argv = self._argv(status)
        if extra:
            argv.extend(extra)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = update_registry.main(
                argv,
                approval_provider=self.provider,
                execution_commit_provider=self._execution_commit_provider,
                execution_worktree_verifier=self._worktree_verifier,
                infrastructure_diff_verifier=self._diff_verifier,
            )
        self.assertEqual(result, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["appended"])
        if merge_to_main:
            # Every durable transition is serialized by a human merge.  The
            # next candidate must start from these exact current-main bytes.
            self._merge_registry_to_fake_main()
        return payload["event"]

    def _append_error(
        self,
        status: str,
        expected_message: str,
        *,
        extra: list[str] | None = None,
    ) -> str:
        argv = self._argv(status)
        if extra:
            argv.extend(extra)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                update_registry.main(
                    argv,
                    approval_provider=self.provider,
                    execution_commit_provider=self._execution_commit_provider,
                    execution_worktree_verifier=self._worktree_verifier,
                    infrastructure_diff_verifier=self._diff_verifier,
                )
        self.assertEqual(caught.exception.code, 2)
        error = stderr.getvalue()
        self.assertIn(expected_message, error)
        self.assertEqual(stdout.getvalue(), "")
        return error

    def _write_run_scope(self) -> tuple[dict[str, Any], str]:
        run_scope = contract.normalize_infrastructure_run_scope(
            sample_run_scope(self.root, self.proposal),
            proposal_scope=self.proposal,
            policy=self.policy,
            policy_sha256=self.policy_sha256,
        )
        run_path = self.root / f"research/scopes/{EXPERIMENT_ID}.run.json"
        run_path.write_text(
            contract.canonical_json_text(run_scope) + "\n",
            encoding="utf-8",
        )
        return run_scope, contract.canonical_digest(run_scope)

    def _events(self) -> list[dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.registry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_complete_synthetic_lifecycle_is_approved_and_then_shut_down(self) -> None:
        proposed = self._append("PROPOSED")
        proposal_digest = proposed["proposal_scope_digest"]

        prepare_comment_id = 73001
        self.provider.add_comment(
            prepare_comment_id,
            "APPROVED_TO_PREPARE",
            proposal_digest,
        )
        approved_prepare = self._append(
            "APPROVED_TO_PREPARE",
            extra=self._approval_args(prepare_comment_id),
        )
        preparing = self._append("PREPARING")

        run_scope, run_digest = self._write_run_scope()
        run_required = self._append("RUN_APPROVAL_REQUIRED")
        run_comment_id = 73002
        self.provider.add_comment(
            run_comment_id,
            "APPROVED_TO_RUN",
            run_digest,
        )
        approved_run = self._append(
            "APPROVED_TO_RUN",
            extra=self._approval_args(run_comment_id),
        )

        self._append_error(
            "RUNNING",
            "infrastructure safety lifecycle is synthetic-only",
            extra=["--execution-kind", "real-data"],
        )
        running = self._append(
            "RUNNING",
            extra=["--execution-kind", "synthetic"],
        )
        review = self._append(
            "REVIEW_REQUIRED",
            extra=["--artifact", "research/results/synthetic-result.json"],
        )
        self._append_error(
            "APPROVED_FOR_SHADOW",
            "infrastructure safety lifecycle does not support shadow approval",
            extra=["--review-digest", "f" * 64],
        )

        self.assertEqual(
            [event["status"] for event in self._events()],
            [
                "proposed",
                "approved_to_prepare",
                "preparing",
                "run_approval_required",
                "approved_to_run",
                "running",
                "review_required",
            ],
        )
        self.assertNotEqual(prepare_comment_id, run_comment_id)
        self.assertEqual(
            approved_prepare["approval_evidence"]["comment_id"],
            prepare_comment_id,
        )
        self.assertEqual(
            approved_run["approval_evidence"]["comment_id"],
            run_comment_id,
        )
        self.assertEqual(
            [item["comment_id"] for item in preparing["revalidated_approval_evidence"]],
            [prepare_comment_id],
        )
        self.assertEqual(
            [item["comment_id"] for item in approved_run["revalidated_approval_evidence"]],
            [prepare_comment_id],
        )
        self.assertEqual(
            [item["comment_id"] for item in running["revalidated_approval_evidence"]],
            [prepare_comment_id, run_comment_id],
        )

        for event in (
            approved_prepare,
            preparing,
            run_required,
            approved_run,
            running,
            review,
        ):
            evidence = event["gate_policy_evidence"]
            self.assertEqual(evidence["ref"], BASE_COMMIT)
            self.assertEqual(evidence["content_sha256"], self.policy_sha256)
            registry_evidence = event["main_registry_evidence"]
            self.assertEqual(registry_evidence["path"], "research/REGISTRY.jsonl")
            self.assertRegex(registry_evidence["ref"], r"^[0-9a-f]{40}$")
        policy_calls = [
            call
            for call in self.provider.file_calls
            if call[0] == "research/INFRASTRUCTURE_GATE.json"
        ]
        self.assertEqual(
            policy_calls,
            [("research/INFRASTRUCTURE_GATE.json", BASE_COMMIT)] * 4,
        )

        for event in (run_required, approved_run, running):
            self.assertEqual(event["run_scope_digest"], run_digest)
        self.assertEqual(running["execution_kind"], run_scope["execution_kind"])
        self.assertEqual(running["execution_kind"], "synthetic")
        self.assertFalse(running["synthetic_fixture_tests_allowed"])
        self.assertFalse(running["automatic_execution_allowed"])
        self.assertFalse(running["execution_authorized"])
        self.assertFalse(running["real_data_execution_allowed"])

        self.assertEqual(review["execution_kind"], "none")
        self.assertFalse(review["preparation_authorized"])
        self.assertFalse(review["synthetic_fixture_tests_allowed"])
        self.assertFalse(review["real_data_execution_allowed"])
        self.assertFalse(review["automatic_execution_allowed"])
        self.assertFalse(review["execution_authorized"])
        self.assertFalse(review["formal_buy"])
        self.assertFalse(review["send_order"])
        self.assertEqual(review["stake"], 0)

        self.assertEqual(len(self.commit_calls), 4)
        self.assertEqual(len(self.worktree_calls), 4)
        self.assertEqual(len(self.diff_calls), 4)
        self.assertTrue(all(root == self.root for root in self.commit_calls))
        self.assertTrue(all(root == self.root for root, _ in self.worktree_calls))
        self.assertTrue(all(root == self.root for root, _ in self.diff_calls))
        committed_material_paths = {
            "research/INFRASTRUCTURE_GATE.json",
            "research/synthetic/input.json",
            "research/synthetic/environment.json",
        }
        for _root, allowed_dirty_paths in self.worktree_calls:
            self.assertTrue(committed_material_paths.isdisjoint(allowed_dirty_paths))

    def test_prepare_comment_id_cannot_be_reused_for_run_approval(self) -> None:
        proposed = self._append("PROPOSED")
        prepare_comment_id = 73005
        self.provider.add_comment(
            prepare_comment_id,
            "APPROVED_TO_PREPARE",
            proposed["proposal_scope_digest"],
        )
        self._append(
            "APPROVED_TO_PREPARE",
            extra=self._approval_args(prepare_comment_id),
        )
        self._append("PREPARING")
        self._write_run_scope()
        self._append("RUN_APPROVAL_REQUIRED")

        self._append_error(
            "APPROVED_TO_RUN",
            f"approval comment ID {prepare_comment_id} was already used",
            extra=self._approval_args(prepare_comment_id),
        )
        self.assertEqual(
            [event["status"] for event in self._events()],
            [
                "proposed",
                "approved_to_prepare",
                "preparing",
                "run_approval_required",
            ],
        )

    def test_remote_policy_hash_mismatch_blocks_prepare_approval(self) -> None:
        proposed = self._append("PROPOSED")
        comment_id = 73003
        self.provider.add_comment(
            comment_id,
            "APPROVED_TO_PREPARE",
            proposed["proposal_scope_digest"],
        )
        self.provider.policy_content = b'{"tampered":true}\n'
        self._append_error(
            "APPROVED_TO_PREPARE",
            "GitHub base-commit file content hash mismatch",
            extra=self._approval_args(comment_id),
        )
        self.assertEqual([event["status"] for event in self._events()], ["proposed"])

    def test_stale_local_registry_cannot_ignore_current_main_history(self) -> None:
        proposed = self._append("PROPOSED")
        comment_id = 73007
        self.provider.add_comment(
            comment_id,
            "APPROVED_TO_PREPARE",
            proposed["proposal_scope_digest"],
        )
        self.provider.registry_content = b'{"remote":"newer-history"}\n'
        self._append_error(
            "APPROVED_TO_PREPARE",
            "must exactly equal the GitHub current-main registry",
            extra=self._approval_args(comment_id),
        )
        self.assertEqual([event["status"] for event in self._events()], ["proposed"])

    def test_unmerged_candidates_are_non_authoritative_and_cannot_advance(self) -> None:
        first = self._append("PROPOSED", merge_to_main=False)
        first_bytes = self.registry_path.read_bytes()
        for field in (
            "preparation_authorized",
            "synthetic_fixture_tests_allowed",
            "real_data_execution_allowed",
            "automatic_execution_allowed",
            "execution_authorized",
        ):
            self.assertFalse(first[field])

        comment_id = 73008
        self.provider.add_comment(
            comment_id,
            "APPROVED_TO_PREPARE",
            first["proposal_scope_digest"],
        )
        self._append_error(
            "APPROVED_TO_PREPARE",
            "must exactly equal the GitHub current-main registry",
            extra=self._approval_args(comment_id),
        )

        # Deleting and replaying the unmerged suffix can create only another
        # pending candidate.  It cannot activate or advance either history.
        self.registry_path.unlink()
        replay = self._append("PROPOSED", merge_to_main=False)
        self.assertNotEqual(replay["event_id"], first["event_id"])
        self.assertFalse(replay["preparation_authorized"])
        self.assertFalse(replay["execution_authorized"])

        self.provider.registry_content = first_bytes
        self.provider.current_main = hashlib.sha256(first_bytes).hexdigest()[:40]
        self._append_error(
            "APPROVED_TO_PREPARE",
            "must exactly equal the GitHub current-main registry",
            extra=self._approval_args(comment_id),
        )

    def test_main_head_move_immediately_before_append_fails_closed(self) -> None:
        self.provider.move_main_on_recheck = True
        self._append_error(
            "PROPOSED",
            "GitHub current main moved during registry verification",
        )
        self.assertEqual(self._events(), [])

    def test_registry_link_is_rejected_before_any_transition(self) -> None:
        target_directory = tempfile.TemporaryDirectory()
        self.addCleanup(target_directory.cleanup)
        target = Path(target_directory.name) / "outside-registry.jsonl"
        target.write_bytes(b"")
        try:
            self.registry_path.symlink_to(target)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")
        self._append_error(
            "PROPOSED",
            "must not traverse a symlink or junction",
        )
        self.assertEqual(target.read_bytes(), b"")

    def test_stored_policy_evidence_must_remain_bound_to_proposal(self) -> None:
        proposed = self._append("PROPOSED")
        comment_id = 73006
        self.provider.add_comment(
            comment_id,
            "APPROVED_TO_PREPARE",
            proposed["proposal_scope_digest"],
        )
        self._append(
            "APPROVED_TO_PREPARE",
            extra=self._approval_args(comment_id),
        )
        events = self._events()
        events[-1]["gate_policy_evidence"]["content_sha256"] = "f" * 64
        self.registry_path.write_text(
            "".join(contract.canonical_json_text(event) + "\n" for event in events),
            encoding="utf-8",
        )

        self._append_error(
            "PREPARING",
            "infrastructure gate policy evidence is not bound to the proposal base/hash",
        )
        self.assertEqual(len(self._events()), 2)

    def test_v2_approval_comment_id_cannot_be_reused_by_v3(self) -> None:
        reused_comment_id = 73004
        legacy_digest = "e" * 64
        legacy_experiment_id = "EXP-LEGACY-V2-001"
        legacy_events = [
            {
                "schema_version": 2,
                "event_id": "legacy-v2-proposed",
                "sequence": 1,
                "experiment_id": legacy_experiment_id,
                "status": "proposed",
                "previous_status": None,
                "previous_event_id": None,
            },
            {
                "schema_version": 2,
                "event_id": "legacy-v2-approved-prepare",
                "sequence": 2,
                "experiment_id": legacy_experiment_id,
                "status": "approved_to_prepare",
                "previous_status": "proposed",
                "previous_event_id": "legacy-v2-proposed",
                "proposal_scope_digest": legacy_digest,
                "approval_evidence": approval_evidence(
                    comment_id=reused_comment_id,
                    approval_type="APPROVED_TO_PREPARE",
                    approval_digest=legacy_digest,
                    experiment_id=legacy_experiment_id,
                ),
            },
        ]
        self.registry_path.write_text(
            "".join(
                json.dumps(event, separators=(",", ":")) + "\n"
                for event in legacy_events
            ),
            encoding="utf-8",
        )
        self._merge_registry_to_fake_main()

        proposed = self._append("PROPOSED")
        self.provider.add_comment(
            reused_comment_id,
            "APPROVED_TO_PREPARE",
            proposed["proposal_scope_digest"],
        )
        self._append_error(
            "APPROVED_TO_PREPARE",
            "approval comment ID 73004 was already used",
            extra=self._approval_args(reused_comment_id),
        )
        events = self._events()
        self.assertEqual(len(events), 3)
        self.assertEqual(events[-1]["schema_version"], 3)
        self.assertEqual(events[-1]["status"], "proposed")

    def test_invalid_requires_exact_main_but_not_remote_policy_evidence(self) -> None:
        self._append("PROPOSED")
        invalid = self._append("INVALID")

        self.assertEqual(invalid["status"], "invalid")
        self.assertIsNone(invalid["gate_policy_evidence"])
        self.assertFalse(invalid["preparation_authorized"])
        self.assertFalse(invalid["synthetic_fixture_tests_allowed"])
        self.assertFalse(invalid["real_data_execution_allowed"])
        self.assertFalse(invalid["automatic_execution_allowed"])
        self.assertFalse(invalid["execution_authorized"])
        self.assertFalse(
            any(
                path == "research/INFRASTRUCTURE_GATE.json"
                for path, _ref in self.provider.file_calls
            )
        )
        self.assertTrue(
            any(
                path == "research/REGISTRY.jsonl"
                for path, _ref in self.provider.file_calls
            )
        )


if __name__ == "__main__":
    unittest.main()
