from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Callable
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SCRIPTS = REPO_ROOT / "scripts" / "research"
if str(RESEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS))

import scope_contract


def load_cli_module(name: str, relative_path: str) -> ModuleType:
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


create_experiment = load_cli_module(
    "create_experiment_lifecycle_test",
    "scripts/research/create_experiment.py",
)
update_registry = load_cli_module(
    "update_registry_lifecycle_test",
    "scripts/research/update_registry.py",
)


BASE_COMMIT = "a" * 40
EXECUTION_COMMIT = "b" * 40
REPOSITORY = "kazuponbaseball-cell/keiba_ai_project"
ISSUE_NUMBER = 17
APPROVER = "kazuponbaseball-cell"

SCORE_74 = {
    "independent_information": 25,
    "racing_mechanism": 20,
    "outer_oos_failure_evidence": 20,
    "leakage_safety": 9,
    "minimal_falsifiability": 0,
    "acquisition_implementation_cost": 0,
}
SCORE_75 = {
    "independent_information": 25,
    "racing_mechanism": 20,
    "outer_oos_failure_evidence": 20,
    "leakage_safety": 10,
    "minimal_falsifiability": 0,
    "acquisition_implementation_cost": 0,
}


class FakeApprovalProvider:
    def __init__(self) -> None:
        self.comments: dict[int, dict[str, Any]] = {}
        self.error: Exception | None = None
        self.calls: list[tuple[str, int]] = []
        self.current_main = "c" * 40
        self.registry_content = b""
        self.branch_calls = 0
        self.approvers_content = json.dumps(
            {
                "schema_version": 1,
                "approvers": [{"login": APPROVER}],
                "denied_login_patterns": ["bot", "codex", "automation"],
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def get_repository(self, repository: str) -> dict[str, Any]:
        return {"full_name": REPOSITORY, "default_branch": "main"}

    def get_branch_ref(self, repository: str, branch: str) -> dict[str, Any]:
        self.branch_calls += 1
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
        return {
            "status": "identical" if base_commit == head_commit else "ahead",
            "url": (
                f"https://api.github.com/repos/{REPOSITORY}/compare/"
                f"{base_commit}...{head_commit}"
            ),
            "base_commit": {"sha": base_commit},
            "merge_base_commit": {"sha": base_commit},
        }

    def get_file_contents(
        self,
        repository: str,
        path: str,
        ref: str,
    ) -> dict[str, Any]:
        if path == "research/REGISTRY.jsonl":
            if ref != self.current_main:
                raise AssertionError(f"unexpected registry ref: {ref}")
            return {
                "type": "file",
                "path": path,
                "encoding": "base64",
                "sha": "e" * 40,
                "content": base64.b64encode(self.registry_content).decode("ascii"),
            }
        return {
            "type": "file",
            "path": path,
            "encoding": "base64",
            "sha": "d" * 40,
            "content": base64.b64encode(self.approvers_content).decode("ascii"),
        }

    def get_issue_comment(self, repository: str, comment_id: int) -> dict[str, Any]:
        self.calls.append((repository, comment_id))
        if self.error is not None:
            raise self.error
        if comment_id not in self.comments:
            raise KeyError(f"comment {comment_id} not found")
        return copy.deepcopy(self.comments[comment_id])


class ExperimentLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        research_dir = self.root / "research"
        research_dir.mkdir(parents=True)
        shutil.copyfile(
            REPO_ROOT / "research" / "EXPERIMENT_TEMPLATE.md",
            research_dir / "EXPERIMENT_TEMPLATE.md",
        )
        self.provider = FakeApprovalProvider()
        self.execution_commit = EXECUTION_COMMIT
        self.next_comment_id = 1000

        self.materials = {
            "config": self._write_material("config/experiment.json", b'{"alpha":1}\n'),
            "data": self._write_material("research/manifests/data.json", b'{"rows":10}\n'),
            "fold": self._write_material("research/manifests/folds.json", b'{"folds":1}\n'),
            "runner": self._write_material(
                "research/manifests/runners.json", b'{"runners":12}\n'
            ),
            "environment": self._write_material(
                "research/manifests/environment.json", b'{"python":"3.11"}\n'
            ),
        }

    def _write_material(self, relative: str, content: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _manifest_ref(self, key: str) -> dict[str, str]:
        path = self.materials[key]
        return {
            "path": path.relative_to(self.root).as_posix(),
            "sha256": scope_contract.sha256_file(path),
        }

    def _proposal_scope(
        self,
        experiment_id: str,
        scores: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        return {
            "experiment_id": experiment_id,
            "title": "Lifecycle contract test",
            "hypothesis": "A new as-of signal improves set NLL.",
            "null_hypothesis": "The challenger does not improve set NLL.",
            "racing_mechanism": "The signal measures pre-race readiness not present in history.",
            "target_population": "JRA races with at least eight starters.",
            "in_scope": ["research-only challenger feature"],
            "out_of_scope": ["production", "BUY", "orders", "notifications"],
            "expected_changed_paths": [
                "config/experiment.json",
                "scripts/research/challenger.py",
            ],
            "raw_data_sources": ["approved as-of source"],
            "data_as_of": "T-3 minutes",
            "allowed_columns": ["race_id", "runner_id", "asof_signal"],
            "forbidden_columns": ["odds", "popularity", "payoff", "roi"],
            "lineage_hash_requirements": [
                "source content SHA-256",
                "received-at manifest SHA-256",
            ],
            "chronological_fold_design": {
                "order": ["train", "validation", "calibration", "outer_test"],
                "race_overlap": 0,
            },
            "fold_manifest": self._manifest_ref("fold"),
            "purge_embargo": {"purge_days": 1, "embargo_days": 1},
            "primary_metric": {"name": "set_nll", "direction": "lower"},
            "required_effect": {"delta_lte": -0.001},
            "rejection_gate": ["delta_set_nll >= 0"],
            "stop_conditions": ["probability contract failure", "lineage failure"],
            "compute_budget": {"max_cpu_minutes": 30, "max_memory_gb": 4},
            "allowed_variant_count": 1,
            "allowed_threshold_search_count": 0,
            "base_commit": BASE_COMMIT,
            "score_components": dict(SCORE_75 if scores is None else scores),
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }

    def _run_scope(self, experiment_id: str) -> dict[str, Any]:
        queue = self._queue(experiment_id)
        proposal = queue["proposal_scope"]
        return {
            "proposal_scope": proposal,
            "proposal_scope_digest": queue["proposal_scope_digest"],
            "execution_commit_sha": EXECUTION_COMMIT,
            "config_hashes": [self._manifest_ref("config")],
            "data_input_manifest_hashes": [self._manifest_ref("data")],
            "fold_manifest_hash": self._manifest_ref("fold"),
            "runner_universe_manifest_hash": self._manifest_ref("runner"),
            "dependency_environment_manifest": self._manifest_ref("environment"),
            "seed": 20260731,
            "exact_execution_commands": [
                "python scripts/research/challenger.py --config config/experiment.json"
            ],
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }

    def _execution_commit_provider(self, _root: Path) -> str:
        return self.execution_commit

    def _invoke(self, main: Callable[..., int], argv: list[str], **kwargs: Any) -> dict[str, Any]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(argv, **kwargs)
        self.assertEqual(exit_code, 0, stderr.getvalue())
        return json.loads(stdout.getvalue())

    def _assert_cli_error(
        self,
        main: Callable[..., int],
        argv: list[str],
        message: str,
        **kwargs: Any,
    ) -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main(argv, **kwargs)
        self.assertEqual(caught.exception.code, 2)
        error_text = stderr.getvalue()
        self.assertIn(message, error_text)
        return error_text

    def _create(
        self,
        experiment_id: str,
        scores: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        input_path = self.root / "inputs" / f"{experiment_id}.proposal.json"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(
            json.dumps(
                self._proposal_scope(experiment_id, scores),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self._invoke(
            create_experiment.main,
            [
                experiment_id,
                "--proposal-scope",
                str(input_path),
                "--owner",
                "test-researcher",
                "--root",
                str(self.root),
            ],
        )

    def _append_argv(self, experiment_id: str, status: str) -> list[str]:
        return [
            experiment_id,
            status,
            "--actor",
            "test-caller",
            "--root",
            str(self.root),
        ]

    def _append(
        self,
        experiment_id: str,
        status: str,
        *,
        extra: list[str] | None = None,
        provider: FakeApprovalProvider | None = None,
    ) -> dict[str, Any]:
        argv = self._append_argv(experiment_id, status)
        if extra:
            argv.extend(extra)
        active_provider = provider or self.provider
        result = self._invoke(
            update_registry.main,
            argv,
            approval_provider=active_provider,
            execution_commit_provider=self._execution_commit_provider,
            execution_worktree_verifier=lambda _root, _allowed: None,
        )
        registry_path = self.root / "research" / "REGISTRY.jsonl"
        active_provider.registry_content = registry_path.read_bytes()
        active_provider.current_main = hashlib.sha256(
            active_provider.registry_content
        ).hexdigest()[:40]
        return result

    def _append_error(
        self,
        experiment_id: str,
        status: str,
        message: str,
        *,
        extra: list[str] | None = None,
        provider: FakeApprovalProvider | None = None,
    ) -> str:
        argv = self._append_argv(experiment_id, status)
        if extra:
            argv.extend(extra)
        return self._assert_cli_error(
            update_registry.main,
            argv,
            message,
            approval_provider=provider or self.provider,
            execution_commit_provider=self._execution_commit_provider,
            execution_worktree_verifier=lambda _root, _allowed: None,
        )

    def _queue(self, experiment_id: str) -> dict[str, Any]:
        return json.loads(
            (self.root / "research" / "queue" / f"{experiment_id}.json").read_text(
                encoding="utf-8"
            )
        )

    def _events(self) -> list[dict[str, Any]]:
        path = self.root / "research" / "REGISTRY.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def _grant_event(self, experiment_id: str, status: str) -> dict[str, Any]:
        matches = [
            event
            for event in self._events()
            if event["experiment_id"] == experiment_id and event["status"] == status
        ]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def _grant_comment_id(self, experiment_id: str, status: str) -> int:
        evidence = self._grant_event(experiment_id, status)["approval_evidence"]
        self.assertIsInstance(evidence, dict)
        comment_id = evidence["comment_id"]
        self.assertIs(type(comment_id), int)
        return comment_id

    def _add_comment(
        self,
        keyword: str,
        digest: str,
        *,
        author: str = APPROVER,
        author_type: str = "User",
        body: str | None = None,
        issue_number: int = ISSUE_NUMBER,
    ) -> int:
        comment_id = self.next_comment_id
        self.next_comment_id += 1
        self.provider.comments[comment_id] = {
            "id": comment_id,
            "html_url": f"https://github.com/{REPOSITORY}/issues/{issue_number}#issuecomment-{comment_id}",
            "issue_url": f"https://api.github.com/repos/{REPOSITORY}/issues/{issue_number}",
            "user": {"login": author, "type": author_type},
            "created_at": "2026-07-31T00:00:00Z",
            "updated_at": "2026-07-31T00:00:00Z",
            "body": body if body is not None else f"{keyword} {digest}",
        }
        return comment_id

    def _approval_args(self, comment_id: int) -> list[str]:
        return [
            "--github-repository",
            REPOSITORY,
            "--issue-number",
            str(ISSUE_NUMBER),
            "--approval-comment-id",
            str(comment_id),
        ]

    def _start_proposal(self, experiment_id: str) -> None:
        self._create(experiment_id)
        self._append(experiment_id, "PROPOSED")

    def _approve_prepare(self, experiment_id: str) -> int:
        digest = self._queue(experiment_id)["proposal_scope_digest"]
        comment_id = self._add_comment("APPROVED_TO_PREPARE", digest)
        self._append(
            experiment_id,
            "APPROVED_TO_PREPARE",
            extra=self._approval_args(comment_id),
        )
        return comment_id

    def _start_preparing(self, experiment_id: str) -> None:
        self._start_proposal(experiment_id)
        self._approve_prepare(experiment_id)
        self._append(experiment_id, "PREPARING", extra=["--execution-kind", "synthetic"])

    def _freeze_run_scope(self, experiment_id: str) -> tuple[Path, str]:
        run_scope = scope_contract.normalize_run_scope(
            self._run_scope(experiment_id),
            proposal_scope=self._queue(experiment_id)["proposal_scope"],
        )
        path = self.root / "research" / "scopes" / f"{experiment_id}.run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            scope_contract.canonical_json_text(run_scope) + "\n",
            encoding="utf-8",
        )
        digest = scope_contract.canonical_digest(run_scope)
        self._append(experiment_id, "RUN_APPROVAL_REQUIRED")
        return path, digest

    def _approve_run(self, experiment_id: str, digest: str) -> int:
        comment_id = self._add_comment("APPROVED_TO_RUN", digest)
        self._append(
            experiment_id,
            "APPROVED_TO_RUN",
            extra=self._approval_args(comment_id),
        )
        return comment_id

    def _approved_run(self, experiment_id: str) -> tuple[Path, str, int]:
        self._start_preparing(experiment_id)
        path, digest = self._freeze_run_scope(experiment_id)
        comment_id = self._approve_run(experiment_id, digest)
        return path, digest, comment_id

    def _running(self, experiment_id: str) -> tuple[Path, str, int]:
        path, digest, comment_id = self._approved_run(experiment_id)
        self._append(experiment_id, "RUNNING", extra=["--execution-kind", "synthetic"])
        return path, digest, comment_id

    def _review_required(self, experiment_id: str) -> tuple[int, int]:
        self._running(experiment_id)
        self._append(experiment_id, "REVIEW_REQUIRED", extra=["--artifact", "result.json"])
        return (
            self._grant_comment_id(experiment_id, "approved_to_prepare"),
            self._grant_comment_id(experiment_id, "approved_to_run"),
        )

    def test_score_74_is_blocked_and_75_is_proposed(self) -> None:
        blocked = self._create("exp-score-74", SCORE_74)
        proposed = self._create("exp-score-75", SCORE_75)
        self.assertEqual(blocked["status"], "blocked_score")
        self.assertEqual(blocked["score_total"], 74)
        self.assertEqual(proposed["status"], "proposed")
        self.assertEqual(proposed["score_total"], 75)
        self.assertRegex(proposed["proposal_scope_digest"], r"^[0-9a-f]{64}$")

    def test_human_approved_flag_alone_cannot_approve(self) -> None:
        experiment_id = "exp-flag-rejected"
        self._start_proposal(experiment_id)
        self._append_error(
            experiment_id,
            "APPROVED_TO_PREPARE",
            "--human-approved cannot grant approval",
            extra=["--human-approved"],
        )

    def test_human_approved_flag_alone_cannot_grant_run_approval(self) -> None:
        experiment_id = "exp-run-flag-rejected"
        self._start_preparing(experiment_id)
        self._freeze_run_scope(experiment_id)
        self._append_error(
            experiment_id,
            "APPROVED_TO_RUN",
            "--human-approved cannot grant approval",
            extra=["--human-approved"],
        )

    def test_caller_actor_alone_cannot_approve(self) -> None:
        experiment_id = "exp-actor-rejected"
        self._start_proposal(experiment_id)
        self._append_error(
            experiment_id,
            "APPROVED_TO_PREPARE",
            "requires GitHub Issue comment evidence",
            extra=["--actor", APPROVER],
        )

    def test_allowlist_outsider_is_rejected(self) -> None:
        experiment_id = "exp-outsider"
        self._start_proposal(experiment_id)
        digest = self._queue(experiment_id)["proposal_scope_digest"]
        comment_id = self._add_comment(
            "APPROVED_TO_PREPARE", digest, author="not-allowed-user"
        )
        self._append_error(
            experiment_id,
            "APPROVED_TO_PREPARE",
            "not in the allowlist",
            extra=self._approval_args(comment_id),
        )

    def test_allowlist_is_read_from_proposal_base_commit_not_worktree(self) -> None:
        worktree_allowlist = self.root / "research" / "APPROVERS.json"
        worktree_allowlist.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "approvers": [{"login": "malicious-branch-user"}],
                }
            ),
            encoding="utf-8",
        )
        experiment_id = "exp-worktree-allowlist-ignored"
        self._start_proposal(experiment_id)
        digest = self._queue(experiment_id)["proposal_scope_digest"]
        comment_id = self._add_comment("APPROVED_TO_PREPARE", digest)
        event = self._append(
            experiment_id,
            "APPROVED_TO_PREPARE",
            extra=self._approval_args(comment_id),
        )["event"]
        self.assertEqual(event["approval_evidence"]["author"], APPROVER)
        self.assertEqual(
            event["github_trust_evidence"]["verified_base_commit"], BASE_COMMIT
        )

    def test_codex_and_automation_authors_cannot_self_approve(self) -> None:
        experiment_id = "exp-automation-rejected"
        self._start_proposal(experiment_id)
        digest = self._queue(experiment_id)["proposal_scope_digest"]
        cases = (
            ("codex-research", "User"),
            ("approved-bot", "Bot"),
            ("github-actions", "Bot"),
        )
        for author, author_type in cases:
            with self.subTest(author=author):
                comment_id = self._add_comment(
                    "APPROVED_TO_PREPARE",
                    digest,
                    author=author,
                    author_type=author_type,
                )

                self._assert_cli_error(
                    update_registry.main,
                    self._append_argv(experiment_id, "APPROVED_TO_PREPARE")
                    + self._approval_args(comment_id),
                    "Codex or automation actor cannot approve research",
                    approval_provider=self.provider,
                    execution_commit_provider=self._execution_commit_provider,
                )

    def test_missing_comment_and_digest_mismatch_are_rejected(self) -> None:
        for suffix, comment_setup, expected in (
            (
                "missing",
                lambda digest: 999999,
                "GitHub approval verification unavailable; fail-close",
            ),
            (
                "digest",
                lambda digest: self._add_comment(
                    "APPROVED_TO_PREPARE", "0" * 64
                ),
                "digest or format mismatch",
            ),
        ):
            with self.subTest(case=suffix):
                experiment_id = f"exp-comment-{suffix}"
                self._start_proposal(experiment_id)
                digest = self._queue(experiment_id)["proposal_scope_digest"]
                comment_id = comment_setup(digest)
                self._append_error(
                    experiment_id,
                    "APPROVED_TO_PREPARE",
                    expected,
                    extra=self._approval_args(comment_id),
                )

    def test_github_unavailable_fails_closed(self) -> None:
        experiment_id = "exp-github-unavailable"
        self._start_proposal(experiment_id)
        digest = self._queue(experiment_id)["proposal_scope_digest"]
        comment_id = self._add_comment("APPROVED_TO_PREPARE", digest)
        self.provider.error = OSError("network unavailable")
        self._append_error(
            experiment_id,
            "APPROVED_TO_PREPARE",
            "GitHub approval verification unavailable; fail-close",
            extra=self._approval_args(comment_id),
        )

    def test_github_unavailable_immediately_before_running_fails_closed(self) -> None:
        experiment_id = "exp-running-github-unavailable"
        self._approved_run(experiment_id)
        self.provider.error = OSError("network unavailable")
        self._append_error(
            experiment_id,
            "RUNNING",
            "GitHub approval verification unavailable; fail-close",
        )

    def test_preparing_requires_approved_to_prepare(self) -> None:
        experiment_id = "exp-no-prepare-approval"
        self._start_proposal(experiment_id)
        self._append_error(experiment_id, "PREPARING", "invalid transition")

    def test_running_requires_approved_to_run(self) -> None:
        experiment_id = "exp-no-run-approval"
        self._start_preparing(experiment_id)
        self._freeze_run_scope(experiment_id)
        self._append_error(experiment_id, "RUNNING", "invalid transition")

    def test_prepare_stage_forbids_real_data_execution(self) -> None:
        experiment_id = "exp-prepare-real-data"
        self._start_proposal(experiment_id)
        self._approve_prepare(experiment_id)
        self._append_error(
            experiment_id,
            "PREPARING",
            "real-data execution is forbidden before RUNNING",
            extra=["--execution-kind", "real-data"],
        )

    def test_synthetic_running_allows_synthetic_and_denies_real_data(self) -> None:
        experiment_id = "exp-running-synthetic-capabilities"
        self._approved_run(experiment_id)
        event = self._append(
            experiment_id,
            "RUNNING",
            extra=["--execution-kind", "synthetic"],
        )["event"]
        self.assertTrue(event["synthetic_fixture_tests_allowed"])
        self.assertFalse(event["real_data_execution_allowed"])
        self.assertTrue(event["automatic_execution_allowed"])
        self.assertTrue(event["execution_authorized"])

    def test_legacy_real_data_running_fails_closed_until_kind_is_scope_bound(self) -> None:
        experiment_id = "exp-running-real-data-unbound"
        self._approved_run(experiment_id)
        self._append_error(
            experiment_id,
            "RUNNING",
            "legacy ROI run scope does not bind execution_kind",
            extra=["--execution-kind", "real-data"],
        )

    def test_historical_unbound_real_data_running_can_only_be_invalidated(self) -> None:
        experiment_id = "exp-stored-real-data-unbound"
        self._running(experiment_id)
        registry_path = self.root / "research" / "REGISTRY.jsonl"
        events = self._events()
        events[-1]["execution_kind"] = "real-data"
        events[-1]["real_data_execution_allowed"] = True
        registry_path.write_text(
            "".join(scope_contract.canonical_json_text(event) + "\n" for event in events),
            encoding="utf-8",
        )
        # Model a historical bad event that was already merged to main.  The
        # INVALID candidate must start from the exact durable ledger bytes.
        self.provider.registry_content = registry_path.read_bytes()
        self.provider.current_main = hashlib.sha256(
            self.provider.registry_content
        ).hexdigest()[:40]

        self._append_error(
            experiment_id,
            "REVIEW_REQUIRED",
            "legacy ROI history contains unbound real-data RUNNING; only INVALID",
        )
        event = self._append(experiment_id, "INVALID")["event"]
        self.assertEqual(event["status"], "invalid")
        self.assertFalse(event["execution_authorized"])

    def test_running_rejects_execution_kind_none(self) -> None:
        experiment_id = "exp-running-kind-none"
        self._approved_run(experiment_id)
        self._append_error(
            experiment_id,
            "RUNNING",
            "RUNNING requires --execution-kind synthetic",
        )

    def test_post_run_review_disables_all_execution_capabilities(self) -> None:
        experiment_id = "exp-post-run-capabilities"
        self._running(experiment_id)
        event = self._append(
            experiment_id,
            "REVIEW_REQUIRED",
            extra=["--artifact", "result.json"],
        )["event"]
        self.assertFalse(event["synthetic_fixture_tests_allowed"])
        self.assertFalse(event["real_data_execution_allowed"])
        self.assertFalse(event["automatic_execution_allowed"])
        self.assertFalse(event["execution_authorized"])

    def test_each_major_proposal_scope_change_is_detected(self) -> None:
        mutations: dict[str, Callable[[dict[str, Any]], None]] = {
            "hypothesis": lambda scope: scope.__setitem__("hypothesis", "changed"),
            "target_population": lambda scope: scope.__setitem__(
                "target_population", "changed population"
            ),
            "data_as_of": lambda scope: scope.__setitem__("data_as_of", "T-1"),
            "feature_scope": lambda scope: scope["in_scope"].append("new feature"),
            "expected_paths": lambda scope: scope["expected_changed_paths"].append(
                "scripts/research/extra.py"
            ),
            "fold": lambda scope: scope["fold_manifest"].__setitem__(
                "sha256", "1" * 64
            ),
            "primary_metric": lambda scope: scope["primary_metric"].__setitem__(
                "name", "brier"
            ),
            "gates": lambda scope: scope["rejection_gate"].append("new gate"),
            "stop_conditions": lambda scope: scope["stop_conditions"].append(
                "new stop"
            ),
            "compute_budget": lambda scope: scope["compute_budget"].__setitem__(
                "max_cpu_minutes", 60
            ),
            "base_commit": lambda scope: scope.__setitem__("base_commit", "c" * 40),
            "scores": lambda scope: scope["score_components"].__setitem__(
                "acquisition_implementation_cost", 1
            ),
            "safety_flags": lambda scope: scope.__setitem__("formal_buy", True),
        }
        for index, (field, mutate) in enumerate(mutations.items()):
            with self.subTest(field=field):
                experiment_id = f"exp-scope-{index:02d}"
                self._start_proposal(experiment_id)
                self._approve_prepare(experiment_id)
                scope_path = (
                    self.root
                    / "research"
                    / "scopes"
                    / f"{experiment_id}.proposal.json"
                )
                scope = json.loads(scope_path.read_text(encoding="utf-8"))
                mutate(scope)
                scope_path.write_text(
                    json.dumps(scope, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self._append_error(
                    experiment_id,
                    "PREPARING",
                    "proposal scope",
                )

    def test_execution_commit_change_is_detected(self) -> None:
        experiment_id = "exp-commit-change"
        self._approved_run(experiment_id)
        self.execution_commit = "c" * 40
        self._append_error(
            experiment_id,
            "RUNNING",
            "execution commit SHA changed",
        )

    def test_uncommitted_code_outside_hash_bound_artifacts_is_rejected(self) -> None:
        completed = mock.Mock(
            returncode=0,
            stdout=(
                " M scripts/research/challenger.py\n"
                "?? research/scopes/exp.run.json\n"
            ),
            stderr="",
        )
        with mock.patch.object(
            update_registry.subprocess,
            "run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(ValueError, "uncommitted or untracked paths"):
                update_registry.verify_execution_worktree(
                    self.root,
                    {"research/scopes/exp.run.json"},
                )

    def test_config_data_and_fold_hash_changes_are_detected(self) -> None:
        for key, expected in (
            ("config", "config_hashes"),
            ("data", "data_input_manifest_hashes"),
            ("fold", "fold_manifest_hash"),
        ):
            with self.subTest(material=key):
                experiment_id = f"exp-hash-{key}"
                self._approved_run(experiment_id)
                with self.materials[key].open("ab") as handle:
                    handle.write(b"changed\n")
                self._append_error(experiment_id, "RUNNING", expected)
                self.materials[key].write_bytes(
                    {
                        "config": b'{"alpha":1}\n',
                        "data": b'{"rows":10}\n',
                        "fold": b'{"folds":1}\n',
                    }[key]
                )

    def test_execution_command_change_is_detected(self) -> None:
        experiment_id = "exp-command-change"
        path, _digest, _comment = self._approved_run(experiment_id)
        run_scope = json.loads(path.read_text(encoding="utf-8"))
        run_scope["exact_execution_commands"].append("python changed.py")
        path.write_text(
            scope_contract.canonical_json_text(run_scope) + "\n",
            encoding="utf-8",
        )
        self._append_error(
            experiment_id,
            "RUNNING",
            "run scope changed",
        )

    def test_prepare_comment_id_cannot_be_reused_for_run_grant(self) -> None:
        experiment_id = "exp-reuse-prepare-as-run"
        self._start_preparing(experiment_id)
        _path, run_digest = self._freeze_run_scope(experiment_id)
        prepare_comment_id = self._grant_comment_id(
            experiment_id, "approved_to_prepare"
        )
        self.provider.comments[prepare_comment_id]["body"] = (
            f"APPROVED_TO_RUN {run_digest}"
        )
        self.provider.comments[prepare_comment_id]["updated_at"] = (
            "2026-07-31T00:01:00Z"
        )
        self._append_error(
            experiment_id,
            "APPROVED_TO_RUN",
            "already used by an approval grant",
            extra=self._approval_args(prepare_comment_id),
        )

    def test_run_comment_id_cannot_be_reused_for_shadow_grant(self) -> None:
        experiment_id = "exp-reuse-run-as-shadow"
        _prepare_comment_id, run_comment_id = self._review_required(experiment_id)
        review_digest = "d" * 64
        self.provider.comments[run_comment_id]["body"] = (
            f"APPROVED_FOR_SHADOW {review_digest}"
        )
        self.provider.comments[run_comment_id]["updated_at"] = (
            "2026-07-31T00:01:00Z"
        )
        self._append_error(
            experiment_id,
            "APPROVED_FOR_SHADOW",
            "already used by an approval grant",
            extra=self._approval_args(run_comment_id)
            + ["--review-digest", review_digest],
        )

    def test_comment_id_cannot_be_reused_across_experiments(self) -> None:
        first_experiment = "exp-global-comment-first"
        self._start_proposal(first_experiment)
        comment_id = self._approve_prepare(first_experiment)

        second_experiment = "exp-global-comment-second"
        self._start_proposal(second_experiment)
        second_digest = self._queue(second_experiment)["proposal_scope_digest"]
        self.provider.comments[comment_id]["body"] = (
            f"APPROVED_TO_PREPARE {second_digest}"
        )
        self.provider.comments[comment_id]["updated_at"] = (
            "2026-07-31T00:01:00Z"
        )
        self._append_error(
            second_experiment,
            "APPROVED_TO_PREPARE",
            "already used by an approval grant",
            extra=self._approval_args(comment_id),
        )

    def test_revalidation_events_do_not_consume_comment_ids_as_new_grants(self) -> None:
        experiment_id = "exp-revalidation-not-grant"
        self._running(experiment_id)
        events = self._events()
        consumed = update_registry.validate_approval_grant_history(events)
        self.assertEqual(len(consumed), 2)
        for status in ("preparing", "running"):
            event = next(item for item in events if item["status"] == status)
            self.assertIsNone(event["approval_evidence"])
            self.assertTrue(event["revalidated_approval_evidence"])

    def test_deleted_prepare_comment_blocks_run_approval(self) -> None:
        experiment_id = "exp-prepare-deleted-before-run-approval"
        self._start_preparing(experiment_id)
        _path, run_digest = self._freeze_run_scope(experiment_id)
        prepare_comment_id = self._grant_comment_id(
            experiment_id, "approved_to_prepare"
        )
        del self.provider.comments[prepare_comment_id]
        run_comment_id = self._add_comment("APPROVED_TO_RUN", run_digest)
        self._append_error(
            experiment_id,
            "APPROVED_TO_RUN",
            "GitHub approval verification unavailable; fail-close",
            extra=self._approval_args(run_comment_id),
        )

    def test_deleted_prepare_or_run_comment_blocks_running(self) -> None:
        for index, grant_status in enumerate(
            ("approved_to_prepare", "approved_to_run")
        ):
            with self.subTest(grant_status=grant_status):
                experiment_id = f"exp-deleted-before-running-{index}"
                self._approved_run(experiment_id)
                comment_id = self._grant_comment_id(experiment_id, grant_status)
                del self.provider.comments[comment_id]
                self._append_error(
                    experiment_id,
                    "RUNNING",
                    "GitHub approval verification unavailable; fail-close",
                    extra=["--execution-kind", "synthetic"],
                )

    def test_deleted_prepare_or_run_comment_blocks_shadow_approval(self) -> None:
        for index, grant_status in enumerate(
            ("approved_to_prepare", "approved_to_run")
        ):
            with self.subTest(grant_status=grant_status):
                experiment_id = f"exp-deleted-before-shadow-{index}"
                self._review_required(experiment_id)
                comment_id = self._grant_comment_id(experiment_id, grant_status)
                del self.provider.comments[comment_id]
                review_digest = "d" * 64
                shadow_comment_id = self._add_comment(
                    "APPROVED_FOR_SHADOW", review_digest
                )
                self._append_error(
                    experiment_id,
                    "APPROVED_FOR_SHADOW",
                    "GitHub approval verification unavailable; fail-close",
                    extra=self._approval_args(shadow_comment_id)
                    + ["--review-digest", review_digest],
                )

    def test_prepare_comment_edit_blocks_run_approval_even_with_fresh_run_comment(self) -> None:
        experiment_id = "exp-prepare-edited-before-run-approval"
        self._start_preparing(experiment_id)
        _path, run_digest = self._freeze_run_scope(experiment_id)
        prepare_comment_id = self._grant_comment_id(
            experiment_id, "approved_to_prepare"
        )
        self.provider.comments[prepare_comment_id]["body"] = (
            f"APPROVED_TO_RUN {run_digest}"
        )
        self.provider.comments[prepare_comment_id]["updated_at"] = (
            "2026-07-31T00:01:00Z"
        )
        run_comment_id = self._add_comment("APPROVED_TO_RUN", run_digest)
        self._append_error(
            experiment_id,
            "APPROVED_TO_RUN",
            "digest or format mismatch",
            extra=self._approval_args(run_comment_id),
        )

    def test_run_comment_edit_blocks_shadow_even_with_fresh_shadow_comment(self) -> None:
        experiment_id = "exp-run-edited-before-shadow"
        _prepare_comment_id, run_comment_id = self._review_required(experiment_id)
        review_digest = "d" * 64
        self.provider.comments[run_comment_id]["body"] = (
            f"APPROVED_FOR_SHADOW {review_digest}"
        )
        self.provider.comments[run_comment_id]["updated_at"] = (
            "2026-07-31T00:01:00Z"
        )
        shadow_comment_id = self._add_comment("APPROVED_FOR_SHADOW", review_digest)
        self._append_error(
            experiment_id,
            "APPROVED_FOR_SHADOW",
            "digest or format mismatch",
            extra=self._approval_args(shadow_comment_id)
            + ["--review-digest", review_digest],
        )

    def test_updated_at_change_blocks_each_later_approval_gate(self) -> None:
        experiment_id = "exp-updated-before-run-approval"
        self._start_preparing(experiment_id)
        _path, run_digest = self._freeze_run_scope(experiment_id)
        prepare_comment_id = self._grant_comment_id(
            experiment_id, "approved_to_prepare"
        )
        self.provider.comments[prepare_comment_id]["updated_at"] = (
            "2026-07-31T00:01:00Z"
        )
        run_comment_id = self._add_comment("APPROVED_TO_RUN", run_digest)
        self._append_error(
            experiment_id,
            "APPROVED_TO_RUN",
            "changed after approval (updated_at)",
            extra=self._approval_args(run_comment_id),
        )

        for index, grant_status in enumerate(
            ("approved_to_prepare", "approved_to_run")
        ):
            with self.subTest(gate="running", grant_status=grant_status):
                running_experiment = f"exp-updated-before-running-{index}"
                self._approved_run(running_experiment)
                comment_id = self._grant_comment_id(
                    running_experiment, grant_status
                )
                self.provider.comments[comment_id]["updated_at"] = (
                    "2026-07-31T00:01:00Z"
                )
                self._append_error(
                    running_experiment,
                    "RUNNING",
                    "changed after approval (updated_at)",
                    extra=["--execution-kind", "synthetic"],
                )

        for index, grant_status in enumerate(
            ("approved_to_prepare", "approved_to_run")
        ):
            with self.subTest(gate="shadow", grant_status=grant_status):
                shadow_experiment = f"exp-updated-before-shadow-{index}"
                self._review_required(shadow_experiment)
                comment_id = self._grant_comment_id(
                    shadow_experiment, grant_status
                )
                self.provider.comments[comment_id]["updated_at"] = (
                    "2026-07-31T00:01:00Z"
                )
                review_digest = "d" * 64
                shadow_comment_id = self._add_comment(
                    "APPROVED_FOR_SHADOW", review_digest
                )
                self._append_error(
                    shadow_experiment,
                    "APPROVED_FOR_SHADOW",
                    "changed after approval (updated_at)",
                    extra=self._approval_args(shadow_comment_id)
                    + ["--review-digest", review_digest],
                )

    def test_distinct_prepare_run_and_shadow_comment_ids_succeed(self) -> None:
        experiment_id = "exp-distinct-approval-comments"
        prepare_comment_id, run_comment_id = self._review_required(experiment_id)
        review_digest = "d" * 64
        shadow_comment_id = self._add_comment("APPROVED_FOR_SHADOW", review_digest)
        event = self._append(
            experiment_id,
            "APPROVED_FOR_SHADOW",
            extra=self._approval_args(shadow_comment_id)
            + ["--review-digest", review_digest],
        )["event"]
        self.assertEqual(
            len({prepare_comment_id, run_comment_id, shadow_comment_id}), 3
        )
        self.assertEqual(event["approval_evidence"]["comment_id"], shadow_comment_id)
        self.assertEqual(
            [item["comment_id"] for item in event["revalidated_approval_evidence"]],
            [prepare_comment_id, run_comment_id],
        )

    def test_approval_comment_edit_before_running_is_rejected(self) -> None:
        for field, value in (
            ("updated_at", "2026-07-31T00:01:00Z"),
            ("body", f"APPROVED_TO_RUN {'f' * 64}"),
            ("user", {"login": "changed-user", "type": "User"}),
        ):
            with self.subTest(field=field):
                experiment_id = f"exp-edited-{field}"
                _path, _digest, comment_id = self._approved_run(experiment_id)
                self.provider.comments[comment_id][field] = value
                self._append_error(
                    experiment_id,
                    "RUNNING",
                    (
                        "GitHub approval comment changed after approval"
                        if field == "updated_at"
                        else "digest or format mismatch"
                        if field == "body"
                        else "not in the allowlist"
                    ),
                )

    def test_run_approval_event_persists_complete_comment_evidence(self) -> None:
        experiment_id = "exp-evidence"
        _path, digest, comment_id = self._approved_run(experiment_id)
        event = self._events()[-1]
        evidence = event["approval_evidence"]
        self.assertEqual(tuple(evidence), update_registry.COMMENT_EVIDENCE_FIELDS)
        self.assertEqual(evidence["comment_id"], comment_id)
        self.assertEqual(evidence["approval_digest"], digest)
        self.assertEqual(evidence["approval_type"], "APPROVED_TO_RUN")
        self.assertEqual(evidence["author"], APPROVER)
        self.assertEqual(evidence["author_type"], "User")
        self.assertEqual(evidence["issue_number"], ISSUE_NUMBER)
        self.assertEqual(evidence["body"], f"APPROVED_TO_RUN {digest}")
        for field in (
            "url",
            "created_at",
            "updated_at",
            "body_sha256",
        ):
            self.assertTrue(evidence[field])
        self.assertRegex(evidence["body_sha256"], r"^[0-9a-f]{64}$")

    def test_shadow_requires_separate_approval(self) -> None:
        experiment_id = "exp-shadow-separate"
        _path, _digest, run_comment_id = self._running(experiment_id)
        self._append(experiment_id, "REVIEW_REQUIRED", extra=["--artifact", "result.json"])
        review_digest = "d" * 64
        self._append_error(
            experiment_id,
            "APPROVED_FOR_SHADOW",
            "already used by an approval grant",
            extra=self._approval_args(run_comment_id)
            + ["--review-digest", review_digest],
        )
        shadow_comment = self._add_comment("APPROVED_FOR_SHADOW", review_digest)
        event = self._append(
            experiment_id,
            "APPROVED_FOR_SHADOW",
            extra=self._approval_args(shadow_comment)
            + ["--review-digest", review_digest],
        )["event"]
        self.assertEqual(event["review_digest"], review_digest)
        self.assertTrue(event["human_shadow_approval_recorded"])

    def test_invalid_is_terminal(self) -> None:
        experiment_id = "exp-invalid-terminal"
        self._start_proposal(experiment_id)
        self._append(experiment_id, "INVALID")
        for status in (
            "PROPOSED",
            "APPROVED_TO_PREPARE",
            "PREPARING",
            "RUNNING",
            "REVIEW_REQUIRED",
        ):
            with self.subTest(status=status):
                self._append_error(experiment_id, status, "invalid transition")

    def test_safety_flags_remain_locked_through_running(self) -> None:
        experiment_id = "exp-safety"
        self._running(experiment_id)
        records = [self._queue(experiment_id), *self._events()]
        for record in records:
            with self.subTest(status=record["status"]):
                self.assertIs(record["formal_buy"], False)
                self.assertIs(record["send_order"], False)
                self.assertEqual(record["stake"], 0)
                for field in (
                    "production_approved",
                    "merge_approved",
                    "buy_approved",
                    "production_change_allowed",
                    "merge_allowed",
                    "buy_logic_change_allowed",
                ):
                    self.assertIs(record[field], False)

    def test_registry_cannot_grant_production_merge_or_buy_approval(self) -> None:
        experiment_id = "exp-authority"
        self._create(experiment_id)
        for status in (
            "APPROVED_FOR_PRODUCTION",
            "APPROVED_FOR_MERGE",
            "APPROVED_FOR_BUY",
            "MERGED",
            "BUY",
        ):
            with self.subTest(status=status):
                self._append_error(
                    experiment_id,
                    status,
                    update_registry.PROHIBITED_STATUS_HINT,
                )

    def test_proposal_markdown_and_queue_record_same_digest(self) -> None:
        experiment_id = "exp-digest-records"
        result = self._create(experiment_id)
        queue = self._queue(experiment_id)
        markdown = (
            self.root / "research" / "experiments" / f"{experiment_id}.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(result["proposal_scope_digest"], queue["proposal_scope_digest"])
        self.assertIn(queue["proposal_scope_digest"], markdown)

    def test_canonical_serialization_sorts_sets_but_preserves_command_order(self) -> None:
        scope = self._proposal_scope("exp-canonical-order")
        scope["in_scope"] = ["zeta", "alpha"]
        normalized = scope_contract.normalize_proposal_scope(scope)
        self.assertEqual(normalized["in_scope"], ["alpha", "zeta"])
        self.assertIn("作用機序", scope_contract.canonical_json_text({"値": "作用機序"}))

        queue_like = {
            "proposal_scope": normalized,
            "proposal_scope_digest": scope_contract.canonical_digest(normalized),
            "execution_commit_sha": EXECUTION_COMMIT,
            "config_hashes": [self._manifest_ref("config")],
            "data_input_manifest_hashes": [self._manifest_ref("data")],
            "fold_manifest_hash": self._manifest_ref("fold"),
            "runner_universe_manifest_hash": self._manifest_ref("runner"),
            "dependency_environment_manifest": self._manifest_ref("environment"),
            "seed": 1,
            "exact_execution_commands": ["python first.py", "python second.py"],
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        run_scope = scope_contract.normalize_run_scope(
            queue_like,
            proposal_scope=normalized,
        )
        self.assertEqual(
            run_scope["exact_execution_commands"],
            ["python first.py", "python second.py"],
        )

    def test_create_refuses_to_overwrite_scope_queue_or_markdown(self) -> None:
        experiment_id = "exp-no-overwrite"
        self._create(experiment_id)
        input_path = self.root / "inputs" / f"{experiment_id}.proposal.json"
        self._assert_cli_error(
            create_experiment.main,
            [
                experiment_id,
                "--proposal-scope",
                str(input_path),
                "--root",
                str(self.root),
            ],
            "refusing to overwrite existing file",
        )


class LegacyDigestCompatibilityTests(unittest.TestCase):
    def test_committed_roi_scope_digests_remain_unchanged(self) -> None:
        expected = {
            "EXP-20260808-030": (
                "890a242b6a14485e233473c96342cfbe66ff3f09178f546a50f2d37f93ab3610",
                "8ef37a63b165c7d1b41b65a3a331c311bfe5957b659921317ca1128d2322bd31",
            ),
            "EXP-20260808-031": (
                "3f97b3d9c57a79ebbcc91746d6e5f27a37253395095019658d49aa5389672410",
                "5aac4066bd3839509990369998568bb8d30df77b2391e11cbf141f980c837804",
            ),
        }
        for experiment_id, (proposal_digest, run_digest) in expected.items():
            with self.subTest(experiment_id=experiment_id):
                proposal = scope_contract.normalize_proposal_scope(
                    scope_contract.strict_json_load(
                        REPO_ROOT / "research" / "scopes" / f"{experiment_id}.proposal.json"
                    ),
                    expected_experiment_id=experiment_id,
                )
                self.assertEqual(scope_contract.canonical_digest(proposal), proposal_digest)
                run_scope = scope_contract.normalize_run_scope(
                    scope_contract.strict_json_load(
                        REPO_ROOT / "research" / "scopes" / f"{experiment_id}.run.json"
                    ),
                    proposal_scope=proposal,
                )
                self.assertEqual(scope_contract.canonical_digest(run_scope), run_digest)


if __name__ == "__main__":
    unittest.main()
