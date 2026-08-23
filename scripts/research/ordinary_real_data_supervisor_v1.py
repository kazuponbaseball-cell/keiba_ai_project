#!/usr/bin/env python3
"""Trusted-supervisor boundary for ordinary real-data execution v1.

The real-data entrypoint is intentionally non-operational in v1.  It validates
the frozen command shape and then requires three real OS enforcement planes;
the code-owned factory currently reports all three as unsupported and stops
before starting a child or opening a row.

The executable state machine in this module is synthetic-only.  It is useful
for deterministic timeout/crash/seal tests but its marker is never accepted as
real authority and it is not a replacement for an OS sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import ordinary_real_data_run_contract_v3 as v3_contract
from ordinary_real_data_execution_broker_v1 import (
    SYNTHETIC_ASSURANCE,
    ExecutionBoundaryError,
    OrdinaryRealDataExecutionBrokerV1,
    SyntheticFixtureEnforcement,
    assert_no_link_or_reparse,
    build_real_data_execution_broker,
    create_exclusive_file,
)


SUPERVISOR_SCHEMA_VERSION = "ordinary_real_data_execution_supervisor_v1"
SYNTHETIC_SEAL_SCHEMA_VERSION = "synthetic_supervisor_seal_v1"
FORBIDDEN_DISPATCH_FLAGS = frozenset({"-c", "--command", "-m"})
SHELL_EXECUTABLE_NAMES = frozenset(
    {
        "bash",
        "bash.exe",
        "cmd",
        "cmd.exe",
        "dash",
        "fish",
        "ksh",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "sh",
        "sh.exe",
        "zsh",
    }
)


@dataclass(frozen=True)
class FrozenPhaseCommand:
    phase_id: str
    executable: str
    argv: tuple[str, ...]
    working_directory: str
    timeout_seconds: int
    runner_path: str
    config_path: str


@dataclass(frozen=True)
class SyntheticCommandPlan:
    """Exact synthetic process binding; callers do not pass free-form shell text."""

    phase_id: str
    executable: str
    argv: tuple[str, ...]
    working_directory: str
    environment: tuple[tuple[str, str], ...]
    timeout_seconds: int
    runner_sha256: str
    config_sha256: str
    declared_output_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SupervisedOutcome:
    status: str
    reason_code: str
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    seal_path: str
    child_processes_started: int
    real_data_rows_opened: int
    automatic_execution_allowed: bool = False
    formal_buy: bool = False
    send_order: bool = False
    stake: int = 0


def _canonical_relative_path(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\0" in value
    ):
        raise ExecutionBoundaryError("PATH_NOT_ALLOWLISTED", f"{label} is not canonical")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ExecutionBoundaryError("PATH_NOT_ALLOWLISTED", f"{label} escapes its root")
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_full_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_argv_shape(
    *,
    executable: str,
    argv: Sequence[str],
    phase_id: str,
) -> tuple[str, str]:
    if not isinstance(argv, (tuple, list)) or not all(
        isinstance(item, str) and item and "\0" not in item for item in argv
    ):
        raise ExecutionBoundaryError("EXACT_ARGV_MISMATCH", "argv must be a structured array")
    if len(argv) != 8:
        raise ExecutionBoundaryError(
            "EXACT_ARGV_MISMATCH",
            "command must be python -I -B <runner> --phase <id> --config <path>",
        )
    if argv[0] != executable or tuple(argv[1:3]) != ("-I", "-B"):
        raise ExecutionBoundaryError(
            "EXACT_ARGV_MISMATCH", "interpreter and -I -B flags are exact and ordered"
        )
    if any(item in FORBIDDEN_DISPATCH_FLAGS for item in argv):
        raise ExecutionBoundaryError(
            "SHELL_OR_FREE_FORM_COMMAND_FORBIDDEN",
            "-c/--command/-m dispatch is forbidden",
        )
    if (
        Path(executable).name.casefold() in SHELL_EXECUTABLE_NAMES
        or Path(argv[3]).suffix.casefold() != ".py"
    ):
        raise ExecutionBoundaryError(
            "SHELL_OR_FREE_FORM_COMMAND_FORBIDDEN",
            "a Python script runner is required without a shell",
        )
    if argv[4] != "--phase" or argv[5] != phase_id or argv[6] != "--config":
        raise ExecutionBoundaryError(
            "EXACT_ARGV_MISMATCH", "phase/config argument positions differ"
        )
    runner = _canonical_relative_path(argv[3], "runner path")
    config = _canonical_relative_path(argv[7], "config path")
    return runner, config


def frozen_phase_command(scope: dict[str, Any], phase_id: str) -> FrozenPhaseCommand:
    """Validate and project one exact v3 command without executing it."""

    if not isinstance(scope, dict) or scope.get("run_scope_schema_version") != v3_contract.RUN_SCOPE_SCHEMA_VERSION:
        raise ExecutionBoundaryError(
            "RUN_SCOPE_NOT_EXACT_V3", "supervisor accepts only ordinary_real_data_run_v3"
        )
    if scope.get("execution_kind") != "real_data":
        raise ExecutionBoundaryError(
            "EXECUTION_KIND_MISMATCH", "real supervisor requires execution_kind=real_data"
        )
    try:
        normalized = v3_contract.normalize_ordinary_real_data_run_scope(
            scope, proposal_scope=scope["proposal_scope"]
        )
    except (KeyError, TypeError, v3_contract.ContractError) as exc:
        raise ExecutionBoundaryError(
            "RUN_SCOPE_NOT_EXACT_V3", "the frozen v3 normalizer rejected the scope"
        ) from exc
    if normalized != scope:
        raise ExecutionBoundaryError(
            "RUN_SCOPE_NOT_EXACT_V3", "scope differs from canonical v3"
        )
    phase = next((item for item in scope["phase_plan"] if item["phase_id"] == phase_id), None)
    if phase is None:
        raise ExecutionBoundaryError("PHASE_MISMATCH", "phase is not frozen")
    command = next(
        (item for item in scope["exact_commands"] if item["command_id"] == phase["command_id"]),
        None,
    )
    if command is None or command["phase_id"] != phase_id:
        raise ExecutionBoundaryError("PHASE_MISMATCH", "phase command binding differs")
    runner, config = _require_exact_argv_shape(
        executable=command["executable"],
        argv=command["argv"],
        phase_id=phase_id,
    )
    if (
        command["executable"] != scope["environment"]["interpreter_path"]
        or command["working_directory"] != scope["repository_working_directory"]
    ):
        raise ExecutionBoundaryError(
            "EXACT_PROCESS_BINDING_MISMATCH",
            "interpreter or cwd differs from the frozen environment",
        )
    code_paths = {item["path"] for item in scope["code_hashes"]}
    config_paths = {item["path"] for item in scope["config_hashes"]}
    if runner not in code_paths or config not in config_paths:
        raise ExecutionBoundaryError(
            "EXACT_PROCESS_BINDING_MISMATCH", "runner/config is not hash-bound by the scope"
        )
    if command["timeout_seconds"] > scope["compute_budget"]["timeout_seconds"]:
        raise ExecutionBoundaryError(
            "RESOURCE_BUDGET_MISMATCH", "phase timeout exceeds the frozen run budget"
        )
    return FrozenPhaseCommand(
        phase_id=phase_id,
        executable=command["executable"],
        argv=tuple(command["argv"]),
        working_directory=command["working_directory"],
        timeout_seconds=command["timeout_seconds"],
        runner_path=runner,
        config_path=config,
    )


_REAL_SUPERVISOR_FACTORY_TOKEN = object()


class OrdinaryRealDataExecutionSupervisorV1:
    """Non-bypass claim is withheld until a real OS backend and runner exist."""

    def __init__(
        self,
        *,
        broker: OrdinaryRealDataExecutionBrokerV1,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _REAL_SUPERVISOR_FACTORY_TOKEN:
            raise ExecutionBoundaryError(
                "REAL_SUPERVISOR_FACTORY_REQUIRED", "real supervisor construction is code-owned"
            )
        if not isinstance(broker, OrdinaryRealDataExecutionBrokerV1):
            raise ExecutionBoundaryError(
                "SYNTHETIC_BACKEND_NOT_REAL_AUTHORITY",
                "synthetic supervisor cannot be promoted to real mode",
            )
        self._broker = broker
        self._child_processes_started = 0

    @property
    def child_processes_started(self) -> int:
        return self._child_processes_started

    @property
    def broker(self) -> OrdinaryRealDataExecutionBrokerV1:
        return self._broker

    def run_phase(self, *, scope: dict[str, Any], phase_id: str) -> SupervisedOutcome:
        frozen_phase_command(scope, phase_id)
        self._broker.assert_real_execution_ready()
        # A future backend must enter its OS sandbox before this point and bind
        # the child handle to that proof.  v1 intentionally has no fallback.
        raise ExecutionBoundaryError(
            "REAL_EXECUTION_BACKEND_NOT_IMPLEMENTED",
            "no v1 backend may launch a real-data child",
        )


def build_real_data_execution_supervisor(
    *, backend: object | None = None
) -> OrdinaryRealDataExecutionSupervisorV1:
    if backend is not None:
        raise ExecutionBoundaryError(
            "SYNTHETIC_BACKEND_NOT_REAL_AUTHORITY",
            "callers cannot inject a backend into the real supervisor factory",
        )
    return OrdinaryRealDataExecutionSupervisorV1(
        broker=build_real_data_execution_broker(),
        _factory_token=_REAL_SUPERVISOR_FACTORY_TOKEN,
    )


class SyntheticImmutableSealPort:
    """O_EXCL fixture seal; this is deliberately not a v3 result manifest."""

    def __init__(self, *, root: Path, fixture_id: str) -> None:
        lexical_root = assert_no_link_or_reparse(root, label="synthetic seal root")
        self._root = lexical_root.resolve()
        if not self._root.is_dir():
            raise ExecutionBoundaryError(
                "SYNTHETIC_FIXTURE_ROOT_INVALID", "synthetic seal root is invalid"
            )
        if not isinstance(fixture_id, str) or not fixture_id:
            raise ExecutionBoundaryError(
                "SYNTHETIC_FIXTURE_ID_INVALID", "synthetic seal fixture_id is required"
            )
        if (self._root / "fixture-seal").exists():
            raise ExecutionBoundaryError(
                "OUTPUT_ROOT_NOT_FRESH", "synthetic seal namespace must be absent"
            )
        self._fixture_id = fixture_id
        self._sealed = False

    @property
    def root(self) -> Path:
        return self._root

    @property
    def fixture_id(self) -> str:
        return self._fixture_id

    def seal(
        self,
        *,
        status: str,
        reason_code: str,
        exit_code: int | None,
        stdout: bytes,
        stderr: bytes,
    ) -> str:
        if status not in {"success", "failure"}:
            raise ExecutionBoundaryError(
                "OUTPUT_SEAL_STATUS_MISMATCH", "synthetic seal status is invalid"
            )
        if self._sealed:
            raise ExecutionBoundaryError(
                "OUTPUT_ROOT_NOT_FRESH", "synthetic execution was already sealed"
            )
        name = "synthetic.success.manifest.json" if status == "success" else "synthetic.failure.manifest.json"
        relative = f"fixture-seal/{name}"
        target = self._root / relative
        value = {
            "schema_version": SYNTHETIC_SEAL_SCHEMA_VERSION,
            "fixture_id": self._fixture_id,
            "assurance": SYNTHETIC_ASSURANCE,
            "execution_kind": "synthetic",
            "status": status,
            "reason_code": reason_code,
            "exit_code": exit_code,
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "consumer_eligible": False,
            "real_data_execution_allowed": False,
            "execution_authorized": False,
            "automatic_execution_allowed": False,
            "real_data_rows_opened": 0,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        create_exclusive_file(target, payload)
        self._sealed = True
        return relative


class SyntheticExecutionSupervisorV1:
    """A separate synthetic subprocess state machine, never a real backend."""

    def __init__(
        self,
        *,
        root: Path,
        enforcement: SyntheticFixtureEnforcement,
        stdout_limit_bytes: int = 65536,
        stderr_limit_bytes: int = 65536,
    ) -> None:
        lexical_root = assert_no_link_or_reparse(root, label="synthetic supervisor root")
        resolved = lexical_root.resolve()
        if not resolved.is_dir():
            raise ExecutionBoundaryError(
                "SYNTHETIC_FIXTURE_ROOT_INVALID", "synthetic supervisor root is invalid"
            )
        if not isinstance(enforcement, SyntheticFixtureEnforcement):
            raise ExecutionBoundaryError(
                "SYNTHETIC_FIXTURE_MARKER_REQUIRED", "synthetic assurance marker is required"
            )
        for label, value in (
            ("stdout_limit_bytes", stdout_limit_bytes),
            ("stderr_limit_bytes", stderr_limit_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self._root = resolved
        self._enforcement = enforcement
        self._stdout_limit = stdout_limit_bytes
        self._stderr_limit = stderr_limit_bytes
        self._child_processes_started = 0
        self._run_attempted = False

    @property
    def assurance(self) -> str:
        return self._enforcement.assurance

    @property
    def child_processes_started(self) -> int:
        return self._child_processes_started

    def _resolve_regular(self, relative: str, label: str) -> Path:
        canonical = _canonical_relative_path(relative, label)
        lexical_target = assert_no_link_or_reparse(
            self._root / canonical, label=label
        )
        target = lexical_target.resolve()
        if self._root not in target.parents or not target.is_file():
            raise ExecutionBoundaryError(
                "PATH_NOT_ALLOWLISTED", f"{label} is not an exact fixture file"
            )
        return target

    def _validate_plan(self, plan: SyntheticCommandPlan) -> tuple[Path, Path, dict[str, str]]:
        if not isinstance(plan, SyntheticCommandPlan):
            raise ExecutionBoundaryError("EXACT_ARGV_MISMATCH", "typed synthetic plan is required")
        if not isinstance(plan.working_directory, str):
            raise ExecutionBoundaryError("CWD_MISMATCH", "synthetic cwd must be an exact string")
        working_directory = assert_no_link_or_reparse(
            Path(plan.working_directory), label="synthetic working directory"
        )
        if working_directory.resolve() != self._root:
            raise ExecutionBoundaryError("CWD_MISMATCH", "synthetic cwd differs from fixture root")
        if not isinstance(plan.executable, str) or not Path(plan.executable).is_absolute():
            raise ExecutionBoundaryError(
                "INTERPRETER_MISMATCH", "synthetic interpreter path differs"
            )
        runner_rel, config_rel = _require_exact_argv_shape(
            executable=plan.executable,
            argv=plan.argv,
            phase_id=plan.phase_id,
        )
        lexical_executable = assert_no_link_or_reparse(
            Path(plan.executable), label="synthetic interpreter"
        )
        executable = lexical_executable.resolve()
        trusted_interpreter = Path(sys.executable).resolve()
        if not executable.is_file() or executable != trusted_interpreter:
            raise ExecutionBoundaryError(
                "INTERPRETER_MISMATCH", "synthetic interpreter path differs"
            )
        runner = self._resolve_regular(runner_rel, "synthetic runner")
        config = self._resolve_regular(config_rel, "synthetic config")
        try:
            hashes_match = (
                _is_full_sha256(plan.runner_sha256)
                and _sha256_file(runner) == plan.runner_sha256
                and _is_full_sha256(plan.config_sha256)
                and _sha256_file(config) == plan.config_sha256
            )
        except OSError as exc:
            raise ExecutionBoundaryError(
                "HASH_BOUND_EXECUTABLE_MISMATCH", "cannot hash runner or config"
            ) from exc
        if not hashes_match:
            raise ExecutionBoundaryError(
                "HASH_BOUND_EXECUTABLE_MISMATCH", "runner or config hash differs"
            )
        if isinstance(plan.timeout_seconds, bool) or not isinstance(plan.timeout_seconds, int) or plan.timeout_seconds < 1:
            raise ExecutionBoundaryError(
                "RESOURCE_BUDGET_MISMATCH", "synthetic timeout must be a positive integer"
            )
        if (
            not isinstance(plan.environment, tuple)
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or "=" in item[0]
                or "\0" in item[0]
                or not isinstance(item[1], str)
                or "\0" in item[1]
                for item in plan.environment
            )
        ):
            raise ExecutionBoundaryError(
                "EXACT_ENVIRONMENT_MISMATCH", "synthetic environment is not exact and isolated"
            )
        environment = dict(plan.environment)
        if (
            len(environment) != len(plan.environment)
            or tuple(sorted(plan.environment)) != plan.environment
            or any(
                key.casefold() in {"pythonpath", "pythonhome"}
                for key in environment
            )
        ):
            raise ExecutionBoundaryError(
                "EXACT_ENVIRONMENT_MISMATCH", "synthetic environment is not exact and isolated"
            )
        if not isinstance(plan.declared_output_paths, tuple) or not all(
            isinstance(item, str) for item in plan.declared_output_paths
        ):
            raise ExecutionBoundaryError(
                "OUTPUT_PATH_NOT_MANIFESTED", "declared outputs must be an exact tuple"
            )
        outputs = tuple(
            _canonical_relative_path(item, "declared output")
            for item in plan.declared_output_paths
        )
        if outputs != tuple(sorted(set(outputs))):
            raise ExecutionBoundaryError(
                "OUTPUT_PATH_NOT_MANIFESTED", "declared outputs must be sorted and unique"
            )
        for relative in outputs:
            if relative == "fixture-seal" or relative.startswith("fixture-seal/"):
                raise ExecutionBoundaryError(
                    "OUTPUT_PATH_NOT_MANIFESTED", "fixture-seal is a reserved namespace"
                )
            lexical_target = assert_no_link_or_reparse(
                self._root / relative, label="declared synthetic output"
            )
            target = lexical_target.resolve()
            if self._root not in target.parents or target.exists():
                raise ExecutionBoundaryError(
                    "OUTPUT_ROOT_NOT_FRESH", "declared synthetic output must be absent"
                )
        if os.path.lexists(self._root / "fixture-seal"):
            raise ExecutionBoundaryError(
                "OUTPUT_ROOT_NOT_FRESH", "fixture-seal namespace was pre-created"
            )
        return runner, config, environment

    def _observed_tree(self) -> tuple[set[str], set[str]]:
        files: set[str] = set()
        directories: set[str] = set()
        try:
            paths = list(self._root.rglob("*"))
        except OSError as exc:
            raise ExecutionBoundaryError(
                "FILESYSTEM_INSPECTION_FAILED", "cannot enumerate synthetic fixture root"
            ) from exc
        for path in paths:
            assert_no_link_or_reparse(path, label="synthetic fixture tree")
            relative = path.relative_to(self._root).as_posix()
            try:
                if path.is_file():
                    files.add(relative)
                elif path.is_dir():
                    directories.add(relative)
                else:
                    raise ExecutionBoundaryError(
                        "UNMANIFESTED_OUTPUT", "fixture contains a non-regular entry"
                    )
            except OSError as exc:
                raise ExecutionBoundaryError(
                    "FILESYSTEM_INSPECTION_FAILED", "cannot inspect synthetic fixture tree"
                ) from exc
        return files, directories

    def _bounded(self, payload: bytes, limit: int) -> tuple[bytes, bool]:
        return payload[:limit], len(payload) > limit

    @staticmethod
    def _seal_or_fail_closed(
        sealer: SyntheticImmutableSealPort,
        *,
        status: str,
        reason_code: str,
        exit_code: int | None,
        stdout: bytes,
        stderr: bytes,
    ) -> str:
        try:
            return sealer.seal(
                status=status,
                reason_code=reason_code,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
            )
        except ExecutionBoundaryError as exc:
            raise ExecutionBoundaryError(
                "SYNTHETIC_SEAL_UNAVAILABLE",
                "synthetic output remains non-consumer-eligible and no seal is claimed",
            ) from exc

    def run(
        self,
        *,
        plan: SyntheticCommandPlan,
        sealer: SyntheticImmutableSealPort,
    ) -> SupervisedOutcome:
        self._validate_plan(plan)
        if not isinstance(sealer, SyntheticImmutableSealPort):
            raise ExecutionBoundaryError(
                "SYNTHETIC_SEAL_PORT_REQUIRED", "typed synthetic immutable sealer required"
            )
        if sealer.root != self._root:
            raise ExecutionBoundaryError(
                "SYNTHETIC_SEAL_PORT_REQUIRED", "synthetic sealer root differs from fixture root"
            )
        if sealer.fixture_id != self._enforcement.fixture_id:
            raise ExecutionBoundaryError(
                "SYNTHETIC_FIXTURE_ID_MISMATCH", "supervisor and sealer fixture IDs differ"
            )
        if self._run_attempted:
            raise ExecutionBoundaryError(
                "ONE_SHOT_BUDGET_EXHAUSTED", "synthetic supervisor never retries"
            )
        self._run_attempted = True
        before_files, before_directories = self._observed_tree()
        environment = dict(plan.environment)
        try:
            process = subprocess.Popen(
                list(plan.argv),
                cwd=plan.working_directory,
                env=environment,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            seal_path = self._seal_or_fail_closed(
                sealer,
                status="failure",
                reason_code="PROCESS_START_FAILED",
                exit_code=None,
                stdout=b"",
                stderr=b"",
            )
            return SupervisedOutcome(
                status="failure",
                reason_code="PROCESS_START_FAILED",
                exit_code=None,
                stdout=b"",
                stderr=b"",
                stdout_truncated=False,
                stderr_truncated=False,
                seal_path=seal_path,
                child_processes_started=0,
                real_data_rows_opened=0,
            )
        self._child_processes_started += 1
        timed_out = False
        process_io_failed = False
        try:
            stdout, stderr = process.communicate(timeout=plan.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                process.kill()
                stdout, stderr = process.communicate()
            except OSError:
                stdout, stderr = b"", b""
        except OSError:
            process_io_failed = True
            try:
                process.kill()
                stdout, stderr = process.communicate()
            except OSError:
                stdout, stderr = b"", b""
        stdout_bounded, stdout_truncated = self._bounded(stdout, self._stdout_limit)
        stderr_bounded, stderr_truncated = self._bounded(stderr, self._stderr_limit)
        declared = set(plan.declared_output_paths)
        allowed_new_directories: set[str] = set()
        for relative in declared:
            parent = PurePosixPath(relative).parent
            while parent.as_posix() not in {"", "."}:
                allowed_new_directories.add(parent.as_posix())
                parent = parent.parent
        try:
            after_files, after_directories = self._observed_tree()
            new_files = after_files - before_files
            new_directories = after_directories - before_directories
            reserved_namespace_created = any(
                path == "fixture-seal" or path.startswith("fixture-seal/")
                for path in new_files | new_directories
            )
            if reserved_namespace_created:
                raise ExecutionBoundaryError(
                    "SYNTHETIC_SEAL_UNAVAILABLE",
                    "the child modified the reserved synthetic seal namespace",
                )
            unexpected = sorted(
                (new_files - declared)
                | (new_directories - allowed_new_directories)
            )
            tree_reason: str | None = None
        except ExecutionBoundaryError as exc:
            if exc.reason_code == "SYNTHETIC_SEAL_UNAVAILABLE":
                raise
            unexpected = []
            tree_reason = exc.reason_code
        if timed_out:
            status, reason_code, exit_code = "failure", "CHILD_TIMEOUT", None
        elif process_io_failed:
            status, reason_code, exit_code = "failure", "CHILD_IO_FAILURE", process.returncode
        elif tree_reason is not None:
            status, reason_code, exit_code = "failure", tree_reason, process.returncode
        elif process.returncode != 0:
            status, reason_code, exit_code = "failure", "CHILD_CRASH", process.returncode
        elif unexpected:
            status, reason_code, exit_code = "failure", "UNMANIFESTED_OUTPUT", process.returncode
        else:
            status, reason_code, exit_code = "success", "SYNTHETIC_SUPERVISED_RUN_OK", process.returncode
        seal_path = self._seal_or_fail_closed(
            sealer,
            status=status,
            reason_code=reason_code,
            exit_code=exit_code,
            stdout=stdout_bounded,
            stderr=stderr_bounded,
        )
        return SupervisedOutcome(
            status=status,
            reason_code=reason_code,
            exit_code=exit_code,
            stdout=stdout_bounded,
            stderr=stderr_bounded,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            seal_path=seal_path,
            child_processes_started=self._child_processes_started,
            real_data_rows_opened=0,
        )


__all__ = [
    "FrozenPhaseCommand",
    "OrdinaryRealDataExecutionSupervisorV1",
    "SUPERVISOR_SCHEMA_VERSION",
    "SYNTHETIC_SEAL_SCHEMA_VERSION",
    "SupervisedOutcome",
    "SyntheticCommandPlan",
    "SyntheticExecutionSupervisorV1",
    "SyntheticImmutableSealPort",
    "build_real_data_execution_supervisor",
    "frozen_phase_command",
]
