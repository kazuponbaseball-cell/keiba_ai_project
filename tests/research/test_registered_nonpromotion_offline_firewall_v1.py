from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import registered_nonpromotion_offline_runner_v1 as runner
from registered_nonpromotion_contract_v1 import ContractError, canonical_digest


def _approval_evidence(checkpoint: str, *, marker: str) -> dict[str, object]:
    evidence: dict[str, object] = {
        "verification_checkpoint": checkpoint,
        "comment": {"issue_number": 7, "comment_id": 11},
        "marker": marker,
    }
    evidence["evidence_digest"] = canonical_digest(evidence)
    return evidence


class RegisteredNonpromotionOfflineFirewallV1Tests(unittest.TestCase):
    def test_gate_unavailable_opens_no_raw_source_for_materialize_or_scope_seal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            repo = workspace / "repo"
            source = workspace / "source"
            repo.mkdir()
            source.mkdir()
            approval = SimpleNamespace(
                verify_offline_gate_availability=mock.Mock(
                    side_effect=ContractError("not merged current main")
                )
            )
            raw_reader = mock.Mock(
                side_effect=AssertionError("raw source was opened before gate availability")
            )
            with (
                mock.patch.object(runner, "_approval_module", return_value=approval),
                mock.patch.object(runner, "_read_verified_source_bytes", raw_reader),
            ):
                with self.assertRaisesRegex(ContractError, "not merged current main"):
                    runner._materialize_fixed_projections_worker(
                        root=repo, source_root=source, provider=object()
                    )
                with self.assertRaisesRegex(ContractError, "not merged current main"):
                    runner._compile_fixed_run_scope_artifact_worker(
                        root=repo, source_root=source, provider=object()
                    )
            raw_reader.assert_not_called()

    def test_workload_firewall_denies_network_subprocess_and_environment_view(self) -> None:
        original_environment = os.environ
        with runner._application_workload_firewall():
            self.assertEqual(os.environ, {})
            denied_calls = (
                lambda: socket.socket(),
                lambda: socket.create_connection(("127.0.0.1", 1)),
                lambda: socket.getaddrinfo("localhost", 80),
                lambda: subprocess.run(["python", "-V"]),
                lambda: subprocess.Popen(["python", "-V"]),
                lambda: os.system("python -V"),
                lambda: os.popen("python -V"),
                lambda: urllib.request.urlopen("https://example.invalid"),
            )
            for invoke in denied_calls:
                with self.assertRaises(runner.OfflineFirewallViolation):
                    invoke()
        self.assertIs(os.environ, original_environment)

    def test_control_plane_subprocess_is_two_fixed_read_only_git_commands(self) -> None:
        head = "a" * 40
        fixed_root = Path("C:/fixed/repo")
        completed = [
            SimpleNamespace(stdout=head + "\n"),
            SimpleNamespace(stdout=""),
        ]
        with mock.patch.object(runner.subprocess, "run", side_effect=completed) as run:
            runner._verify_clean_git_head(fixed_root, head)
        self.assertEqual(run.call_count, 2)
        first, second = run.call_args_list
        self.assertEqual(
            first.args[0],
            ["git", "-C", str(fixed_root), "rev-parse", "HEAD"],
        )
        self.assertEqual(
            second.args[0],
            [
                "git",
                "-C",
                str(fixed_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
        )
        for invocation in (first, second):
            self.assertEqual(
                invocation.kwargs,
                {
                    "check": True,
                    "capture_output": True,
                    "text": True,
                    "timeout": 15,
                },
            )

    def test_path_traversal_ads_unc_and_symlink_are_rejected(self) -> None:
        for value in (
            "../secret.csv",
            "safe/../../secret.csv",
            "/absolute.csv",
            "safe\\windows.csv",
            "safe/file.csv:stream",
            "safe//file.csv",
            "./safe/file.csv",
            "safe/",
        ):
            with self.subTest(value=value), self.assertRaises(ContractError):
                runner._safe_relative(value, label="test path")
        with self.assertRaisesRegex(ContractError, "UNC"):
            runner._safe_root(Path(r"\\server\share"), label="test root")

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "target.csv"
            target.write_text("value\n", encoding="utf-8")
            link = root / "link.csv"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlink creation is unavailable on this Windows host")
            with self.assertRaisesRegex(ContractError, "symlink|reparse"):
                runner._safe_bound_path(
                    root, "link.csv", label="linked source", must_exist=True
                )
            target_root = root / "target-root"
            target_root.mkdir()
            linked_root = root / "linked-root"
            linked_root.symlink_to(target_root, target_is_directory=True)
            with self.assertRaisesRegex(ContractError, "symlink|reparse"):
                runner._safe_root(linked_root, label="linked root")

    def test_persisted_approval_evidence_tamper_and_absence_fail_closed(self) -> None:
        evidence = _approval_evidence(
            "BEFORE_CANDIDATE_OPEN", marker="candidate"
        )
        snapshot = runner._snapshot_approval_evidence(
            evidence,
            checkpoint="BEFORE_CANDIDATE_OPEN",
            issue_number=7,
            comment_id=11,
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "approval.json"
            runner._write_json_exclusive(path, snapshot)
            runner._verify_persisted_mapping(
                path, snapshot, label="approval evidence"
            )
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ContractError, "changed"):
                runner._verify_persisted_mapping(
                    path, snapshot, label="approval evidence"
                )
            with self.assertRaisesRegex(ContractError, "missing"):
                runner._verify_persisted_mapping(
                    root / "absent.json", snapshot, label="approval evidence"
                )
        forged = dict(evidence)
        forged["marker"] = "tampered"
        with self.assertRaisesRegex(ContractError, "digest mismatch"):
            runner._snapshot_approval_evidence(
                forged,
                checkpoint="BEFORE_CANDIDATE_OPEN",
                issue_number=7,
                comment_id=11,
            )

    def test_crash_tombstone_recovers_invalid_before_raw_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            output = repo / runner.FIXED_OUTPUT_ROOT
            output.mkdir(parents=True)
            scope = {
                "output_root": runner.FIXED_OUTPUT_ROOT,
                "run_scope_digest": "d" * 64,
                "semantic_subject_digest": "e" * 64,
                "exact_subject_digest": "f" * 64,
            }
            registered = SimpleNamespace()
            approval_loader = mock.Mock(
                side_effect=AssertionError("recovery may not reach GitHub")
            )
            raw_reader = mock.Mock(
                side_effect=AssertionError("recovery may not open raw sources")
            )
            with (
                mock.patch.object(
                    runner, "resolve_offline_registered_recipe", return_value=registered
                ),
                mock.patch.object(runner, "verify_canonical_offline_run_scope"),
                mock.patch.object(runner, "_approval_module", approval_loader),
                mock.patch.object(runner, "_read_verified_source_bytes", raw_reader),
                self.assertRaises(runner.OfflineInvalidAfterStart),
            ):
                runner._execute_offline_registered_diagnostic_worker(
                    root=repo,
                    source_root=repo / "does-not-exist",
                    run_scope=scope,
                    issue_number=7,
                    comment_id=11,
                    provider=object(),
                )
            approval_loader.assert_not_called()
            raw_reader.assert_not_called()
            self.assertEqual(
                json.loads((output / "INVALID.json").read_text(encoding="utf-8")),
                {
                    "reason_code": "INVALID_AFTER_START_NO_RETRY",
                    "status": "INVALID",
                },
            )
            with self.assertRaises(runner.OfflineInvalidAfterStart):
                runner._recover_or_reject_existing_output(output, scope)

    def test_process_lock_rejects_concurrent_runner_without_touching_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "fixed-output"
            with runner._exclusive_run_lock(output):
                with self.assertRaises(runner.OfflineRunAlreadyRunning):
                    with runner._exclusive_run_lock(output):
                        self.fail("a concurrent runner acquired the same local lock")
                self.assertFalse(output.exists())

    def test_recovery_lock_cleans_only_fixed_orphan_result_staging(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "fixed-output"
            orphan = output.parent / (
                f".{output.name}.result." + "a" * 32 + ".tmp"
            )
            orphan.write_text("ROI=9999", encoding="utf-8")
            unrelated = output.parent / ".unrelated.tmp"
            unrelated.write_text("preserve", encoding="utf-8")
            with runner._exclusive_run_lock(output):
                runner._cleanup_orphan_result_temps(output)
            self.assertFalse(orphan.exists())
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "preserve")

    def test_partial_invalid_is_repaired_to_number_free_canonical_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            (output / "INVALID.json").write_bytes(b'{"status":"INVALID"')
            runner._write_invalid_after_start(output)
            self.assertEqual(
                (output / "INVALID.json").read_bytes(),
                b'{"reason_code":"INVALID_AFTER_START_NO_RETRY","status":"INVALID"}\n',
            )

    def test_live_token_requirement_fails_before_raw_or_github(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            scope = {
                "output_root": runner.FIXED_OUTPUT_ROOT,
                "run_scope_digest": "d" * 64,
                "semantic_subject_digest": "e" * 64,
                "exact_subject_digest": "f" * 64,
            }
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(
                    runner,
                    "resolve_offline_registered_recipe",
                    return_value=SimpleNamespace(),
                ),
                mock.patch.object(runner, "verify_canonical_offline_run_scope"),
                mock.patch.object(
                    runner,
                    "_approval_module",
                    side_effect=AssertionError("token gate must precede GitHub"),
                ),
                self.assertRaisesRegex(ContractError, "GH_TOKEN|GITHUB_TOKEN"),
            ):
                runner._execute_offline_registered_diagnostic_worker(
                    root=repo,
                    source_root=repo / "does-not-exist",
                    run_scope=scope,
                    issue_number=7,
                    comment_id=11,
                    provider=object(),
                    _require_live_token=True,
                )

    def test_candidate_open_occurs_after_start_and_failure_publishes_only_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            projection_bindings = {
                "candidate_projection": {
                    "path": "candidate.jsonl",
                    "sha256": "1" * 64,
                    "byte_size": 3,
                },
                "settlement_projection": {
                    "path": "settlement.jsonl",
                    "sha256": "2" * 64,
                    "byte_size": 3,
                },
            }
            source_bindings = {"fixed": "lineage"}
            registered = SimpleNamespace(
                runtime_material_digests={"runner": "b" * 64},
                recipe={},
            )
            scope = {
                "verified_current_main_sha": "a" * 40,
                "run_scope_base_commit": "a" * 40,
                "run_scope_digest": "d" * 64,
                "semantic_subject_digest": "e" * 64,
                "exact_subject_digest": "f" * 64,
                "output_root": runner.FIXED_OUTPUT_ROOT,
                "runtime_bindings": {
                    "environment_manifest_sha256": "3" * 64,
                    "materialization_manifest": {
                        "path": runner.MATERIALIZATION_MANIFEST_PATH,
                        "sha256": "4" * 64,
                        "byte_size": 10,
                    },
                    "source_bindings": source_bindings,
                    "projection_bindings": projection_bindings,
                },
            }
            manifest = {
                "implementation_binding": {
                    "implementation_commit": "a" * 40,
                    "runtime_material_bundle_sha256": canonical_digest(
                        dict(registered.runtime_material_digests)
                    ),
                }
            }
            approval = SimpleNamespace(
                verify_offline_run_approval=mock.Mock(
                    return_value=_approval_evidence(
                        "INITIAL_APPROVAL", marker="initial"
                    )
                ),
                reverify_offline_run_approval=mock.Mock(
                    return_value=_approval_evidence(
                        "BEFORE_CANDIDATE_OPEN", marker="candidate"
                    )
                ),
            )
            events: list[tuple[str, str]] = []
            original_write = runner._write_json_exclusive

            def observed_write(path: Path, value: object) -> None:
                events.append(("write", path.name))
                original_write(path, value)  # type: ignore[arg-type]

            def fail_candidate_read(
                _root: Path, _binding: object, *, label: str
            ) -> tuple[Path, bytes]:
                events.append(("read", label))
                raise RuntimeError("ROI=9999 profit=8888 race_payload=SECRET")

            raw_reader = mock.Mock(
                side_effect=AssertionError("run path must never open a raw source")
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    runner, "resolve_offline_registered_recipe", return_value=registered
                ),
                mock.patch.object(runner, "verify_canonical_offline_run_scope"),
                mock.patch.object(runner, "_verify_clean_git_head"),
                mock.patch.object(runner, "_verify_runtime_environment"),
                mock.patch.object(
                    runner,
                    "_verify_materialization_manifest",
                    return_value=(manifest, source_bindings, projection_bindings),
                ),
                mock.patch.object(runner, "_approval_module", return_value=approval),
                mock.patch.object(
                    runner, "_verify_deterministic_materialization_against_raw"
                ),
                mock.patch.object(runner, "_read_bound_bytes", side_effect=fail_candidate_read),
                mock.patch.object(runner, "_read_verified_source_bytes", raw_reader),
                mock.patch.object(runner, "_write_json_exclusive", side_effect=observed_write),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
                self.assertRaisesRegex(
                    ContractError, "INVALID_AFTER_START_NO_RETRY"
                ),
            ):
                runner._execute_offline_registered_diagnostic_worker(
                    root=repo,
                    source_root=repo,
                    run_scope=scope,
                    issue_number=7,
                    comment_id=11,
                    provider=object(),
                )
            self.assertEqual(
                events,
                [
                    ("write", "approval_evidence_initial.json"),
                    ("write", "approval_evidence_before_candidate.json"),
                    ("write", "start_receipt.json"),
                    ("read", "candidate projection"),
                ],
            )
            raw_reader.assert_not_called()
            approval.reverify_offline_run_approval.assert_called_once()
            invalid_path = repo / runner.FIXED_OUTPUT_ROOT / "INVALID.json"
            start_receipt = json.loads(
                (repo / runner.FIXED_OUTPUT_ROOT / "start_receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                start_receipt["approval_evidence_digest"],
                _approval_evidence(
                    "BEFORE_CANDIDATE_OPEN", marker="candidate"
                )["evidence_digest"],
            )
            self.assertEqual(
                json.loads(invalid_path.read_text(encoding="utf-8")),
                {
                    "reason_code": "INVALID_AFTER_START_NO_RETRY",
                    "status": "INVALID",
                },
            )
            leaked = stdout.getvalue() + stderr.getvalue() + invalid_path.read_text(
                encoding="utf-8"
            )
            for forbidden in ("9999", "8888", "SECRET", "profit", "race_payload"):
                self.assertNotIn(forbidden, leaked)

    def test_cli_post_start_failure_prints_only_stable_invalid_status(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            invalid = repo / runner.FIXED_OUTPUT_ROOT / "INVALID.json"
            invalid.parent.mkdir(parents=True)
            invalid.write_text(
                '{"reason_code":"INVALID_AFTER_START_NO_RETRY","status":"INVALID"}\n',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(runner, "_github_provider", return_value=object()),
                mock.patch.object(runner, "_load_fixed_scope_artifact", return_value={}),
                mock.patch.object(
                    runner,
                    "execute_offline_registered_diagnostic",
                    side_effect=runner.OfflineInvalidAfterStart(
                        "ROI=9999 secret row payload"
                    ),
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = runner.main(
                    [
                        "--root",
                        str(repo),
                        "run",
                        "--run-scope-digest",
                        "d" * 64,
                        "--source-root",
                        str(repo),
                        "--issue-number",
                        "7",
                        "--comment-id",
                        "11",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                stderr.getvalue(),
                '{"reason_code":"INVALID_AFTER_START_NO_RETRY","status":"INVALID"}\n',
            )
            self.assertNotIn("9999", stderr.getvalue())
            self.assertNotIn("secret", stderr.getvalue())

    def test_cli_lock_held_status_is_stable_and_number_free(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(runner, "_load_fixed_scope_artifact", return_value={}),
                mock.patch.object(
                    runner,
                    "execute_offline_registered_diagnostic",
                    side_effect=runner.OfflineRunAlreadyRunning("ROI=9999 secret"),
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = runner.main(
                    [
                        "--root",
                        str(repo),
                        "run",
                        "--run-scope-digest",
                        "d" * 64,
                        "--source-root",
                        str(repo),
                        "--issue-number",
                        "7",
                        "--comment-id",
                        "11",
                    ]
                )
            self.assertEqual(status, 3)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                stderr.getvalue(),
                '{"reason_code":"LOCAL_RUN_LOCK_HELD","status":"RNOD_RUNNING"}\n',
            )
            self.assertNotIn("9999", stderr.getvalue())

    def test_invalid_persistence_io_failure_is_no_throw_and_not_filesystem_classified(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            with mock.patch.object(
                runner,
                "_write_bytes_exclusive",
                side_effect=OSError("disk failure with ROI=9999"),
            ):
                runner._write_invalid_after_start(output)
            self.assertFalse((output / "INVALID.json").exists())

            repo = Path(raw) / "repo"
            repo.mkdir()
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(runner, "_load_fixed_scope_artifact", return_value={}),
                mock.patch.object(
                    runner,
                    "execute_offline_registered_diagnostic",
                    side_effect=runner.OfflineInvalidAfterStart("ROI=9999"),
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = runner.main(
                    [
                        "--root",
                        str(repo),
                        "run",
                        "--run-scope-digest",
                        "d" * 64,
                        "--source-root",
                        str(repo),
                        "--issue-number",
                        "7",
                        "--comment-id",
                        "11",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                stderr.getvalue(),
                '{"reason_code":"INVALID_AFTER_START_NO_RETRY","status":"INVALID"}\n',
            )
            self.assertNotIn("9999", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
