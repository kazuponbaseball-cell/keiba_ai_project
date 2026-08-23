from __future__ import annotations

import copy
import hashlib
import inspect
import json
import shutil
import stat
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
for import_path in (ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import scope_contract
import ordinary_real_data_execution_broker_v1 as broker
import ordinary_real_data_run_contract_v3 as v3_contract
import ordinary_real_data_supervisor_v1 as supervisor
import tests.research.test_ordinary_real_data_run_v3 as baseline


REGISTRY_SHA256 = "5f61f1d02ba2deebce1ca9db57de9048d22628977128b4e231888d8766937e0b"
PROTECTED_SHA256 = {
    "scripts/research/scope_contract.py": (
        "7244ef769feff2f9a7ec0e0328c58bfa3facf4aadb2e2b0868c6d5917fb3daf8"
    ),
    "scripts/research/prepare_run_scope.py": (
        "65680c1e8df4cc0c5c5242280cd6c41339095cad0e522c84ba0ae9e9a2771c5a"
    ),
    "research/schemas/roi_reproduction_run_v2.schema.json": (
        "4054da62b3b0163e0725364a654d4504550b227547a88160e6952c0a6a6755bc"
    ),
    "research/schemas/roi_reproduction_proposal_v2.schema.json": (
        "23a8e4f047b9a95debd295fceb347841124ccee0f8c5fb33c5224c9ed217b965"
    ),
    "scripts/research/ordinary_real_data_run_contract_v3.py": (
        "ca3280e97f310b71a302b85f2ccc5f657cabf78c85e974e1f43d12abc3175cf9"
    ),
    "scripts/research/prepare_ordinary_real_data_run_scope_v3.py": (
        "a87b250cda3b21e0ab55a8e8f63f6ec6e81c68c89f7037c5ac401e27c96c496d"
    ),
    "research/schemas/ordinary_real_data_run_v3.schema.json": (
        "205ae3af0de2a59e8b472f74c6a842eaef535d26fef5801350a1e488d66e3650"
    ),
}
LEGACY_GOLDEN_DIGESTS = {
    "EXP-20260808-030": (
        "890a242b6a14485e233473c96342cfbe66ff3f09178f546a50f2d37f93ab3610",
        "8ef37a63b165c7d1b41b65a3a331c311bfe5957b659921317ca1128d2322bd31",
    ),
    "EXP-20260808-031": (
        "3f97b3d9c57a79ebbcc91746d6e5f27a37253395095019658d49aa5389672410",
        "5aac4066bd3839509990369998568bb8d30df77b2391e11cbf141f980c837804",
    ),
}


class RealRowReadBomb:
    """A real-row backend substitute that must remain untouched in every test."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *_args: Any, **_kwargs: Any) -> bytes:
        self.count += 1
        raise AssertionError("a focused synthetic test attempted a real-data row read")


class OrdinaryRealDataExecutionBrokerV1Tests(unittest.TestCase):
    """Adversarial synthetic coverage for user acceptance items 1 through 43."""

    maxDiff = None

    def setUp(self) -> None:
        fixture = baseline.V3Fixture(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.v3_fixture = fixture
        self.scope = fixture.scope()
        self.real_meter = broker.AccessMeter()
        self.real_row_bomb = RealRowReadBomb()

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.synthetic_root = Path(temporary.name).resolve()
        self._write_synthetic_file("config.json", b"{}\n")
        self._write_synthetic_file("success.py", b"print('synthetic success')\n")
        self._write_synthetic_file(
            "crash.py",
            b"import sys\nprint('synthetic crash', file=sys.stderr)\nraise SystemExit(7)\n",
        )
        self._write_synthetic_file(
            "timeout.py", b"import time\ntime.sleep(5)\n"
        )
        self._write_synthetic_file(
            "unmanifested.py",
            b"from pathlib import Path\nPath('rogue.tmp').write_text('x', encoding='utf-8')\n",
        )
        self.synthetic_enforcement = broker.SyntheticFixtureEnforcement(
            fixture_id="BROKER-V1-SYNTHETIC"
        )

    def _write_synthetic_file(self, relative: str, payload: bytes) -> Path:
        target = self.synthetic_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    @staticmethod
    def _digest(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _file_digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _assert_reason(self, expected: str, function: Any, *args: Any, **kwargs: Any) -> None:
        with self.assertRaises(broker.ExecutionBoundaryError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.reason_code, expected)

    def _phase_id(self) -> str:
        return "canonicalize_input_release"

    def _row_entry(self) -> dict[str, Any]:
        return next(
            item
            for item in self.scope["read_allowlist"]
            if item["access_class"].endswith("_row_blob")
        )

    def _metadata_entry(self) -> dict[str, Any]:
        return next(
            item
            for item in self.scope["read_allowlist"]
            if item["access_class"] == "metadata_manifest"
        )

    def _read_request(
        self, entry: dict[str, Any] | None = None, **changes: Any
    ) -> broker.BrokerReadRequest:
        selected = entry or self._row_entry()
        values = {
            "scope_version": self.scope["run_scope_schema_version"],
            "execution_kind": self.scope["execution_kind"],
            "capability_profile_id": self.scope["capability_profile"]["profile_id"],
            "phase_id": selected["phases"][0],
            "path": selected["path"],
            "access_class": selected["access_class"],
            "required_capability": selected["required_capability"],
            "expected_sha256": selected["sha256"],
        }
        values.update(changes)
        return broker.BrokerReadRequest(**values)

    def _real_broker(self) -> broker.OrdinaryRealDataExecutionBrokerV1:
        return broker.build_real_data_execution_broker(meter=self.real_meter)

    def _assert_real_row_prevented(
        self,
        *,
        scope: dict[str, Any] | None = None,
        request: broker.BrokerReadRequest | None = None,
        expected_reason: str = "NETWORK_ISOLATION_UNAVAILABLE",
    ) -> None:
        self._assert_reason(
            expected_reason,
            self._real_broker().open_row_blob,
            root=self.v3_fixture.root,
            scope=scope or self.scope,
            request=request or self._read_request(),
            authority_context={},
        )
        self.assertEqual(self.real_meter.real_data_rows_opened, 0)
        self.assertEqual(self.real_meter.bytes_delivered, 0)
        self.assertEqual(self.real_row_bomb.count, 0)

    def _authority_case(self) -> dict[str, Any]:
        return self.v3_fixture.authority_case()

    def _synthetic_plan(
        self,
        runner: str,
        *,
        timeout_seconds: int = 2,
        declared_output_paths: tuple[str, ...] = (),
    ) -> supervisor.SyntheticCommandPlan:
        runner_path = self.synthetic_root / runner
        config_path = self.synthetic_root / "config.json"
        executable = str(Path(sys.executable).resolve())
        phase_id = "synthetic_phase"
        return supervisor.SyntheticCommandPlan(
            phase_id=phase_id,
            executable=executable,
            argv=(
                executable,
                "-I",
                "-B",
                runner,
                "--phase",
                phase_id,
                "--config",
                "config.json",
            ),
            working_directory=str(self.synthetic_root),
            environment=(),
            timeout_seconds=timeout_seconds,
            runner_sha256=self._file_digest(runner_path),
            config_sha256=self._file_digest(config_path),
            declared_output_paths=declared_output_paths,
        )

    def _synthetic_supervisor(self) -> supervisor.SyntheticExecutionSupervisorV1:
        return supervisor.SyntheticExecutionSupervisorV1(
            root=self.synthetic_root,
            enforcement=self.synthetic_enforcement,
        )

    def _synthetic_sealer(self) -> supervisor.SyntheticImmutableSealPort:
        return supervisor.SyntheticImmutableSealPort(
            root=self.synthetic_root,
            fixture_id=self.synthetic_enforcement.fixture_id,
        )

    def _read_seal(self, relative: str) -> dict[str, Any]:
        return json.loads((self.synthetic_root / relative).read_text(encoding="utf-8"))

    def _synthetic_blob_broker(
        self,
        *,
        payload: bytes = b"synthetic row\n",
        access_class: str = "runner_row_blob",
        capability: str | None = "read_real_runner_rows",
    ) -> tuple[broker.SyntheticFixtureBrokerV1, str, str, broker.AccessMeter]:
        relative = "fixture/blobs/row.jsonl"
        target = self._write_synthetic_file(relative, payload)
        digest = self._file_digest(target)
        meter = broker.AccessMeter()
        fixture_broker = broker.SyntheticFixtureBrokerV1(
            root=self.synthetic_root,
            enforcement=self.synthetic_enforcement,
            entries={
                relative: broker.SyntheticFixtureEntry(
                    path=relative,
                    access_class=access_class,
                    required_capability=capability,
                    sha256=digest,
                )
            },
            meter=meter,
        )
        return fixture_broker, relative, digest, meter

    def _isolated_synthetic_case(
        self,
        root: Path,
        *,
        runner_name: str,
        runner_payload: bytes,
        fixture_id: str,
    ) -> tuple[
        supervisor.SyntheticExecutionSupervisorV1,
        supervisor.SyntheticImmutableSealPort,
        supervisor.SyntheticCommandPlan,
    ]:
        config_path = root / "config.json"
        runner_path = root / runner_name
        config_path.write_bytes(b"{}\n")
        runner_path.write_bytes(runner_payload)
        executable = str(Path(sys.executable).resolve())
        phase_id = "synthetic_phase"
        enforcement = broker.SyntheticFixtureEnforcement(fixture_id=fixture_id)
        synthetic = supervisor.SyntheticExecutionSupervisorV1(
            root=root,
            enforcement=enforcement,
        )
        sealer = supervisor.SyntheticImmutableSealPort(
            root=root,
            fixture_id=enforcement.fixture_id,
        )
        plan = supervisor.SyntheticCommandPlan(
            phase_id=phase_id,
            executable=executable,
            argv=(
                executable,
                "-I",
                "-B",
                runner_path.name,
                "--phase",
                phase_id,
                "--config",
                config_path.name,
            ),
            working_directory=str(root),
            environment=(),
            timeout_seconds=2,
            runner_sha256=self._file_digest(runner_path),
            config_sha256=self._file_digest(config_path),
        )
        return synthetic, sealer, plan

    def test_01_exact_v3_scope_required(self) -> None:
        """§12 #1: reject every value that is not an exact canonical v3 scope."""

        self._assert_reason(
            "RUN_SCOPE_NOT_EXACT_V3", supervisor.frozen_phase_command, {}, self._phase_id()
        )
        command = supervisor.frozen_phase_command(self.scope, self._phase_id())
        self.assertEqual(command.phase_id, self._phase_id())
        self.assertEqual(self.real_meter.real_data_rows_opened, 0)

    def test_02_legacy_v2_real_data_rejected(self) -> None:
        """§12 #2: versionless legacy v2 cannot enter the new real-data broker."""

        experiment_id = "EXP-20260808-030"
        proposal = scope_contract.normalize_proposal_scope(
            scope_contract.strict_json_load(
                ROOT / "research/scopes" / f"{experiment_id}.proposal.json"
            ),
            expected_experiment_id=experiment_id,
        )
        legacy = scope_contract.normalize_run_scope(
            scope_contract.strict_json_load(
                ROOT / "research/scopes" / f"{experiment_id}.run.json"
            ),
            proposal_scope=proposal,
        )
        version, dispatched = v3_contract.dispatch_ordinary_run_scope(
            legacy, proposal_scope=proposal
        )
        self.assertEqual(version, "legacy_v2")
        self.assertEqual(dispatched, legacy)
        request = self._read_request()
        self._assert_reason(
            "LEGACY_V2_REAL_DATA_FORBIDDEN",
            self._real_broker().open_metadata,
            root=self.v3_fixture.root,
            scope=legacy,
            request=request,
        )
        self.assertEqual(self.real_meter.real_data_rows_opened, 0)

    def test_03_unknown_version_rejected(self) -> None:
        """§12 #3: null, case-changed and unknown scope versions fail closed."""

        for value in (None, "ORDINARY_REAL_DATA_RUN_V3", "ordinary_real_data_run_v4"):
            with self.subTest(version=value):
                changed = copy.deepcopy(self.scope)
                changed["run_scope_schema_version"] = value
                self._assert_reason(
                    "RUN_SCOPE_VERSION_UNKNOWN",
                    self._real_broker().open_metadata,
                    root=self.v3_fixture.root,
                    scope=changed,
                    request=self._read_request(),
                )
        self.assertEqual(self.real_meter.real_data_rows_opened, 0)

    def test_04_branch_local_running_not_authority(self) -> None:
        """§12 #4: a branch-local RUNNING object is not merged authority."""

        case = self._authority_case()
        self.assertTrue(self.v3_fixture.authorize(case))
        pending = copy.deepcopy(case["event"])
        pending["event_id"] = "branch-local-running"
        self.assertFalse(
            self.v3_fixture.authorize(case, merged_running_event=pending)
        )
        self._assert_real_row_prevented()

    def test_05_merged_running_required(self) -> None:
        """§12 #5: absence of the current-main RUNNING event denies authority."""

        case = self._authority_case()
        self.assertFalse(self.v3_fixture.authorize(case, merged_running_event=None))
        self._assert_real_row_prevented()

    def test_06_prepare_approval_required(self) -> None:
        """§12 #6: missing Prepare evidence is denied before row access."""

        case = self._authority_case()
        self.assertFalse(self.v3_fixture.authorize(case, prepare_evidence=None))
        self._assert_real_row_prevented()

    def test_07_run_approval_required(self) -> None:
        """§12 #7: missing Run evidence is denied before row access."""

        case = self._authority_case()
        self.assertFalse(self.v3_fixture.authorize(case, run_evidence=None))
        self._assert_real_row_prevented()

    def test_08_edited_approval_rejected(self) -> None:
        """§12 #8: edited immutable approval evidence is denied."""

        case = self._authority_case()
        edited = copy.deepcopy(case["prepare"])
        edited["updated_at"] = "2026-08-22T15:00:01Z"
        self.assertFalse(self.v3_fixture.authorize(case, prepare_evidence=edited))
        self._assert_real_row_prevented()

    def test_09_reused_approval_rejected(self) -> None:
        """§12 #9: Prepare and Run IDs cannot be reused as one grant."""

        case = self._authority_case()
        reused = copy.deepcopy(case["run"])
        reused["comment_id"] = case["prepare"]["comment_id"]
        self.assertFalse(self.v3_fixture.authorize(case, run_evidence=reused))
        self._assert_real_row_prevented()

    def test_10_execution_commit_mismatch_rejected(self) -> None:
        """§12 #10: execution-commit mismatch is denied by frozen v3 authority."""

        case = self._authority_case()
        self.assertFalse(
            self.v3_fixture.authorize(case, execution_commit="d" * 40)
        )
        self._assert_real_row_prevented()

    def test_11_current_main_drift_rejected(self) -> None:
        """§12 #11: current-main drift invalidates the synthetic authority case."""

        case = self._authority_case()
        self.assertFalse(self.v3_fixture.authorize(case, current_main_sha="d" * 40))
        self._assert_real_row_prevented()

    def test_12_dirty_or_unverifiable_worktree_rejected(self) -> None:
        """§12 #12: worktree verification is mandatory in the broker call chain."""

        access_source = inspect.getsource(v3_contract.verify_access_request)
        self.assertIn("verify_execution_worktree_state", access_source)
        with self.assertRaises(v3_contract.ContractError):
            v3_contract.verify_execution_worktree_state(
                self.v3_fixture.root, self.scope
            )
        self._assert_real_row_prevented()

    def test_13_cwd_mismatch_rejected(self) -> None:
        """§12 #13: supervisor rejects a working directory other than fixture root."""

        plan = replace(
            self._synthetic_plan("success.py"),
            working_directory=str(self.synthetic_root / "other"),
        )
        self._assert_reason(
            "CWD_MISMATCH",
            self._synthetic_supervisor().run,
            plan=plan,
            sealer=self._synthetic_sealer(),
        )

    def test_14_interpreter_mismatch_rejected(self) -> None:
        """§12 #14: executable and argv[0] must name the same exact interpreter."""

        base = self._synthetic_plan("success.py")
        fake_interpreter = str(self.synthetic_root / "fake-python")
        non_python_executable = (
            shutil.which("where") or shutil.which("true") or shutil.which("echo")
        )
        if non_python_executable is None:
            self.fail("a portable non-Python system executable is required")
        cases = {
            "missing_interpreter": fake_interpreter,
            "existing_non_python_executable": str(
                Path(non_python_executable).resolve()
            ),
        }
        for label, executable in cases.items():
            with self.subTest(label=label, executable=executable):
                argv = list(base.argv)
                argv[0] = executable
                plan = replace(base, executable=executable, argv=tuple(argv))
                synthetic = self._synthetic_supervisor()
                self._assert_reason(
                    "INTERPRETER_MISMATCH",
                    synthetic.run,
                    plan=plan,
                    sealer=self._synthetic_sealer(),
                )
                self.assertEqual(synthetic.child_processes_started, 0)

    def test_15_dependency_environment_mismatch_rejected(self) -> None:
        """§12 #15: complete dependency/environment drift denies v3 authority."""

        case = self._authority_case()
        changed = copy.deepcopy(case["scope"]["environment"])
        changed["dependency_versions"][0]["version"] = "0.0.0"
        self.assertFalse(
            self.v3_fixture.authorize(case, observed_environment=changed)
        )
        self._assert_real_row_prevented()

        invalid_environments = (
            (("BAD=KEY", "value"),),
            (("BAD\0KEY", "value"),),
            (("SAFE_KEY", "bad\0value"),),
        )
        for environment in invalid_environments:
            with self.subTest(environment=repr(environment)):
                synthetic = self._synthetic_supervisor()
                self._assert_reason(
                    "EXACT_ENVIRONMENT_MISMATCH",
                    synthetic.run,
                    plan=replace(
                        self._synthetic_plan("success.py"),
                        environment=environment,
                    ),
                    sealer=self._synthetic_sealer(),
                )
                self.assertEqual(synthetic.child_processes_started, 0)

    def test_16_argv_mismatch_rejected(self) -> None:
        """§12 #16: phase and argv position drift is rejected before child start."""

        plan = self._synthetic_plan("success.py")
        argv = list(plan.argv)
        argv[5] = "other_phase"
        plan = replace(plan, argv=tuple(argv))
        synthetic = self._synthetic_supervisor()
        self._assert_reason(
            "EXACT_ARGV_MISMATCH",
            synthetic.run,
            plan=plan,
            sealer=self._synthetic_sealer(),
        )
        self.assertEqual(synthetic.child_processes_started, 0)

        nul_argv = list(self._synthetic_plan("success.py").argv)
        nul_argv[5] = "synthetic_phase\0forged"
        synthetic = self._synthetic_supervisor()
        self._assert_reason(
            "EXACT_ARGV_MISMATCH",
            synthetic.run,
            plan=replace(
                self._synthetic_plan("success.py"),
                argv=tuple(nul_argv),
            ),
            sealer=self._synthetic_sealer(),
        )
        self.assertEqual(synthetic.child_processes_started, 0)

    def test_17_shell_and_free_form_command_rejected(self) -> None:
        """§12 #17: shell executables, -m and free-form dispatch are forbidden."""

        base = self._synthetic_plan("success.py")
        module_argv = list(base.argv)
        module_argv[3] = "-m"
        shell = str(self.synthetic_root / "cmd.exe")
        shell_argv = list(base.argv)
        shell_argv[0] = shell
        cases = (
            (base.executable, tuple(module_argv)),
            (shell, tuple(shell_argv)),
        )
        for executable, argv in cases:
            with self.subTest(argv=argv):
                self._assert_reason(
                    "SHELL_OR_FREE_FORM_COMMAND_FORBIDDEN",
                    supervisor._require_exact_argv_shape,
                    executable=executable,
                    argv=argv,
                    phase_id=base.phase_id,
                )

    def test_18_phase_mismatch_rejected(self) -> None:
        """§12 #18: an unknown phase is rejected before enforcement or file access."""

        self._assert_real_row_prevented(
            request=self._read_request(phase_id="unknown_phase"),
            expected_reason="PHASE_MISMATCH",
        )

    def test_19_capability_mismatch_rejected(self) -> None:
        """§12 #19: caller capability must equal entry, phase and finite profile."""

        entry = self._row_entry()
        wrong = (
            "read_real_runner_rows"
            if entry["required_capability"] != "read_real_runner_rows"
            else "canonicalize_input_release"
        )
        self._assert_real_row_prevented(
            request=self._read_request(required_capability=wrong),
            expected_reason="CAPABILITY_MISMATCH",
        )

    def test_20_access_class_mismatch_rejected(self) -> None:
        """§12 #20: a different finite row class cannot substitute for the frozen class."""

        entry = self._row_entry()
        wrong = (
            "sealed_input_row_blob"
            if entry["access_class"] != "sealed_input_row_blob"
            else "runner_row_blob"
        )
        self._assert_real_row_prevented(
            request=self._read_request(access_class=wrong),
            expected_reason="ACCESS_CLASS_MISMATCH",
        )

    def test_21_unallowlisted_path_rejected(self) -> None:
        """§12 #21: exact phase allowlisting is required for every path."""

        self._assert_real_row_prevented(
            request=self._read_request(path="research/synthetic/not-approved.jsonl"),
            expected_reason="PATH_NOT_ALLOWLISTED",
        )

    def test_22_hash_mismatch_rejected_before_delivery(self) -> None:
        """§12 #22: opened synthetic bytes are rehashed before caller delivery."""

        fixture_broker, relative, digest, meter = self._synthetic_blob_broker()
        (self.synthetic_root / relative).write_bytes(b"tampered synthetic row\n")
        self._assert_reason(
            "INPUT_HASH_MISMATCH",
            fixture_broker.open_blob,
            path=relative,
            access_class="runner_row_blob",
            required_capability="read_real_runner_rows",
            expected_sha256=digest,
        )
        self.assertEqual(meter.real_data_rows_opened, 0)
        self.assertEqual(meter.synthetic_blobs_opened, 1)
        self.assertEqual(meter.bytes_delivered, 0)

    def test_23_metadata_to_row_relabel_rejected(self) -> None:
        """§12 #23: metadata_manifest cannot be relabeled as a row blob."""

        fixture_broker, relative, digest, meter = self._synthetic_blob_broker(
            access_class="metadata_manifest", capability=None
        )
        self._assert_reason(
            "METADATA_AS_ROW_RELABEL_FORBIDDEN",
            fixture_broker.open_blob,
            path=relative,
            access_class="runner_row_blob",
            required_capability=None,
            expected_sha256=digest,
        )
        self.assertEqual(meter.synthetic_blobs_opened, 0)
        self.assertEqual(meter.real_data_rows_opened, 0)

    def test_24_row_to_metadata_relabel_rejected(self) -> None:
        """§12 #24: a row blob cannot be relabeled as metadata."""

        fixture_broker, relative, digest, meter = self._synthetic_blob_broker()
        self._assert_reason(
            "ROW_AS_METADATA_RELABEL_FORBIDDEN",
            fixture_broker.open_blob,
            path=relative,
            access_class="metadata_manifest",
            required_capability="read_real_runner_rows",
            expected_sha256=digest,
        )
        self.assertEqual(meter.synthetic_blobs_opened, 0)
        self.assertEqual(meter.real_data_rows_opened, 0)

    def test_25_receipt_alone_is_not_authority(self) -> None:
        """§12 #25: a structurally valid durable receipt cannot bypass OS enforcement."""

        case = self._authority_case()
        normalized = v3_contract.normalize_execution_receipt(
            case["receipt"],
            run_scope=case["scope"],
            run_scope_digest=v3_contract.canonical_digest(case["scope"]),
        )
        self.assertFalse(normalized["real_data_execution_allowed"])
        self.assertFalse(normalized["execution_authorized"])
        self._assert_real_row_prevented()

    def test_26_receipt_and_supervisor_authority_flags_are_false(self) -> None:
        """§12 #26: v3 receipt flags and supervisor automatic authority stay false."""

        case = self._authority_case()
        receipt = v3_contract.normalize_execution_receipt(
            case["receipt"],
            run_scope=case["scope"],
            run_scope_digest=v3_contract.canonical_digest(case["scope"]),
        )
        self.assertFalse(receipt["real_data_execution_allowed"])
        self.assertFalse(receipt["execution_authorized"])
        report = broker.observe_real_host_enforcement().as_dict()
        self.assertFalse(report["automatic_execution_allowed"])
        self.assertFalse(report["real_data_execution_allowed"])
        self.assertFalse(report["execution_authorized"])

    def test_27_existing_output_root_rejected(self) -> None:
        """§12 #27: create-exclusive synthetic seal refuses an existing seal path."""

        sealer = self._synthetic_sealer()
        sealer.seal(
            status="failure",
            reason_code="FIRST_SEAL",
            exit_code=None,
            stdout=b"",
            stderr=b"",
        )
        self._assert_reason(
            "OUTPUT_ROOT_NOT_FRESH",
            sealer.seal,
            status="failure",
            reason_code="SECOND_SEAL",
            exit_code=None,
            stdout=b"",
            stderr=b"",
        )

    def test_28_symlink_or_reparse_escape_rejected(self) -> None:
        """§12 #28: linked/reparse-like fixture entries fail before process start."""

        synthetic = self._synthetic_supervisor()
        linked_stat = SimpleNamespace(
            st_mode=stat.S_IFLNK,
            st_file_attributes=0,
        )
        with mock.patch.object(
            broker.os.path, "lexists", return_value=True
        ), mock.patch.object(broker.os, "lstat", return_value=linked_stat):
            self._assert_reason(
                "SYMLINK_OR_REPARSE_ESCAPE",
                broker.assert_no_link_or_reparse,
                self.synthetic_root / "linked",
                label="synthetic linked fixture",
            )
        self.assertEqual(synthetic.child_processes_started, 0)
        self.assertEqual(self.real_meter.real_data_rows_opened, 0)

    def test_29_timeout_seals_failure(self) -> None:
        """§12 #29: timeout kills the synthetic child and seals immutable failure."""

        synthetic = self._synthetic_supervisor()
        outcome = synthetic.run(
            plan=self._synthetic_plan("timeout.py", timeout_seconds=1),
            sealer=self._synthetic_sealer(),
        )
        self.assertEqual((outcome.status, outcome.reason_code), ("failure", "CHILD_TIMEOUT"))
        self.assertIsNone(outcome.exit_code)
        seal = self._read_seal(outcome.seal_path)
        self.assertEqual(seal["status"], "failure")
        self.assertEqual(seal["reason_code"], "CHILD_TIMEOUT")
        self.assertFalse(seal["consumer_eligible"])
        self.assertEqual(seal["real_data_rows_opened"], 0)

    def test_30_child_crash_seals_failure(self) -> None:
        """§12 #30: nonzero child exit seals consumer-ineligible failure."""

        outcome = self._synthetic_supervisor().run(
            plan=self._synthetic_plan("crash.py"),
            sealer=self._synthetic_sealer(),
        )
        self.assertEqual((outcome.status, outcome.reason_code), ("failure", "CHILD_CRASH"))
        self.assertEqual(outcome.exit_code, 7)
        seal = self._read_seal(outcome.seal_path)
        self.assertEqual(seal["reason_code"], "CHILD_CRASH")
        self.assertFalse(seal["consumer_eligible"])
        self.assertEqual(outcome.real_data_rows_opened, 0)

        with tempfile.TemporaryDirectory() as temporary:
            start_failure_root = Path(temporary).resolve()
            synthetic, sealer, plan = self._isolated_synthetic_case(
                start_failure_root,
                runner_name="never_started.py",
                runner_payload=b"raise AssertionError('must not start')\n",
                fixture_id="BROKER-V1-POPEN-VALUE-ERROR",
            )
            with mock.patch.object(
                supervisor.subprocess,
                "Popen",
                side_effect=ValueError("synthetic process creation rejected"),
            ) as popen:
                start_failure = synthetic.run(plan=plan, sealer=sealer)
            popen.assert_called_once()
            self.assertEqual(
                (start_failure.status, start_failure.reason_code),
                ("failure", "PROCESS_START_FAILED"),
            )
            self.assertEqual(start_failure.child_processes_started, 0)
            self.assertEqual(synthetic.child_processes_started, 0)
            failure_seal = json.loads(
                (start_failure_root / start_failure.seal_path).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(failure_seal["status"], "failure")
            self.assertEqual(failure_seal["reason_code"], "PROCESS_START_FAILED")
            self.assertFalse(failure_seal["consumer_eligible"])
            self.assertEqual(failure_seal["real_data_rows_opened"], 0)

    def test_31_unmanifested_output_becomes_failure(self) -> None:
        """§12 #31: an undeclared child output produces an immutable failure seal."""

        outcome = self._synthetic_supervisor().run(
            plan=self._synthetic_plan("unmanifested.py"),
            sealer=self._synthetic_sealer(),
        )
        self.assertEqual(
            (outcome.status, outcome.reason_code),
            ("failure", "UNMANIFESTED_OUTPUT"),
        )
        self.assertTrue((self.synthetic_root / "rogue.tmp").is_file())
        seal = self._read_seal(outcome.seal_path)
        self.assertEqual(seal["reason_code"], "UNMANIFESTED_OUTPUT")
        self.assertFalse(seal["consumer_eligible"])

        with tempfile.TemporaryDirectory() as temporary:
            forged_root = Path(temporary).resolve()
            config_path = forged_root / "config.json"
            runner_path = forged_root / "forge_failure_seal.py"
            config_path.write_bytes(b"{}\n")
            runner_path.write_bytes(
                b"from pathlib import Path\n"
                b"target = Path('fixture-seal/synthetic.failure.manifest.json')\n"
                b"target.parent.mkdir(parents=True)\n"
                b"target.write_bytes(b'forged child seal\\n')\n"
            )
            executable = str(Path(sys.executable).resolve())
            phase_id = "synthetic_phase"
            enforcement = broker.SyntheticFixtureEnforcement(
                fixture_id="BROKER-V1-FORGED-SEAL"
            )
            synthetic = supervisor.SyntheticExecutionSupervisorV1(
                root=forged_root,
                enforcement=enforcement,
            )
            sealer = supervisor.SyntheticImmutableSealPort(
                root=forged_root,
                fixture_id=enforcement.fixture_id,
            )
            plan = supervisor.SyntheticCommandPlan(
                phase_id=phase_id,
                executable=executable,
                argv=(
                    executable,
                    "-I",
                    "-B",
                    runner_path.name,
                    "--phase",
                    phase_id,
                    "--config",
                    config_path.name,
                ),
                working_directory=str(forged_root),
                environment=(),
                timeout_seconds=2,
                runner_sha256=self._file_digest(runner_path),
                config_sha256=self._file_digest(config_path),
            )
            self._assert_reason(
                "SYNTHETIC_SEAL_UNAVAILABLE",
                synthetic.run,
                plan=plan,
                sealer=sealer,
            )
            forged_path = (
                forged_root / "fixture-seal" / "synthetic.failure.manifest.json"
            )
            self.assertEqual(forged_path.read_bytes(), b"forged child seal\n")
            self.assertEqual(synthetic.child_processes_started, 1)
            self.assertEqual(self.real_meter.real_data_rows_opened, 0)

        reserved_cases = {
            "directory_only": (
                b"from pathlib import Path\nPath('fixture-seal').mkdir()\n",
                None,
            ),
            "success_manifest_only": (
                b"from pathlib import Path\n"
                b"target = Path('fixture-seal/synthetic.success.manifest.json')\n"
                b"target.parent.mkdir(parents=True)\n"
                b"target.write_bytes(b'forged child success seal\\n')\n",
                b"forged child success seal\n",
            ),
        }
        for label, (runner_payload, expected_success_payload) in reserved_cases.items():
            with self.subTest(reserved_namespace=label):
                with tempfile.TemporaryDirectory() as temporary:
                    reserved_root = Path(temporary).resolve()
                    synthetic, sealer, plan = self._isolated_synthetic_case(
                        reserved_root,
                        runner_name=f"reserve_{label}.py",
                        runner_payload=runner_payload,
                        fixture_id=f"BROKER-V1-{label.upper()}",
                    )
                    with mock.patch.object(
                        sealer, "seal", wraps=sealer.seal
                    ) as formal_seal:
                        self._assert_reason(
                            "SYNTHETIC_SEAL_UNAVAILABLE",
                            synthetic.run,
                            plan=plan,
                            sealer=sealer,
                        )
                    formal_seal.assert_not_called()
                    failure_path = (
                        reserved_root
                        / "fixture-seal"
                        / "synthetic.failure.manifest.json"
                    )
                    self.assertFalse(failure_path.exists())
                    success_path = (
                        reserved_root
                        / "fixture-seal"
                        / "synthetic.success.manifest.json"
                    )
                    if expected_success_payload is None:
                        self.assertFalse(success_path.exists())
                    else:
                        self.assertEqual(
                            success_path.read_bytes(), expected_success_payload
                        )
                    self.assertEqual(synthetic.child_processes_started, 1)
                    self.assertEqual(self.real_meter.real_data_rows_opened, 0)

    def test_32_unsupported_network_isolation_rejected_pre_row(self) -> None:
        """§12 #32: unavailable OS network isolation stops before row access."""

        report = broker.RealHostEnforcementReport(
            backend_id="test-real-host",
            host_system="SyntheticHost",
            network_isolation=broker.EnforcementStatus.UNSUPPORTED_FAIL_CLOSED,
            filesystem_broker_isolation=broker.EnforcementStatus.ENFORCED,
            resource_supervision=broker.EnforcementStatus.ENFORCED,
        )
        self._assert_reason(
            "NETWORK_ISOLATION_UNAVAILABLE", broker._require_real_enforcement, report
        )
        self.assertEqual(self.real_meter.real_data_rows_opened, 0)

    def test_33_unsupported_broker_filesystem_isolation_rejected_pre_row(self) -> None:
        """§12 #33: unavailable broker-only filesystem confinement stops pre-row."""

        report = broker.RealHostEnforcementReport(
            backend_id="test-real-host",
            host_system="SyntheticHost",
            network_isolation=broker.EnforcementStatus.ENFORCED,
            filesystem_broker_isolation=broker.EnforcementStatus.UNSUPPORTED_FAIL_CLOSED,
            resource_supervision=broker.EnforcementStatus.ENFORCED,
        )
        self._assert_reason(
            "BROKER_ONLY_FILESYSTEM_ISOLATION_UNAVAILABLE",
            broker._require_real_enforcement,
            report,
        )
        self.assertEqual(self.real_meter.real_data_rows_opened, 0)

    def test_34_unsupported_resource_enforcement_rejected_pre_row(self) -> None:
        """§12 #34: unavailable resource enforcement stops before row access."""

        report = broker.RealHostEnforcementReport(
            backend_id="test-real-host",
            host_system="SyntheticHost",
            network_isolation=broker.EnforcementStatus.ENFORCED,
            filesystem_broker_isolation=broker.EnforcementStatus.ENFORCED,
            resource_supervision=broker.EnforcementStatus.UNSUPPORTED_FAIL_CLOSED,
        )
        self._assert_reason(
            "RESOURCE_ENFORCEMENT_UNAVAILABLE",
            broker._require_real_enforcement,
            report,
        )
        self.assertEqual(self.real_meter.real_data_rows_opened, 0)

        enforced_probe = broker.RealHostEnforcementReport(
            backend_id="forged-all-enforced-probe",
            host_system="SyntheticHost",
            network_isolation=broker.EnforcementStatus.ENFORCED,
            filesystem_broker_isolation=broker.EnforcementStatus.ENFORCED,
            resource_supervision=broker.EnforcementStatus.ENFORCED,
        )
        write_entry = self.scope["write_allowlist"][0]
        write_request = broker.BrokerWriteRequest(
            scope_version=self.scope["run_scope_schema_version"],
            execution_kind=self.scope["execution_kind"],
            capability_profile_id=self.scope["capability_profile"]["profile_id"],
            phase_id=write_entry["phases"][0],
            path=write_entry["path"],
            required_capability=write_entry["required_capability"],
        )
        with (
            mock.patch.object(
                broker,
                "observe_real_host_enforcement",
                return_value=enforced_probe,
            ),
            mock.patch.object(
                broker.v3_contract, "issue_execution_receipt"
            ) as issuer,
            mock.patch.object(broker.v3_contract, "read_authorized_bytes") as reader,
            mock.patch.object(broker.v3_contract, "write_authorized_bytes") as writer,
            mock.patch.object(broker.v3_contract, "seal_output_manifest") as sealer,
        ):
            real_broker = broker.build_real_data_execution_broker(
                meter=self.real_meter
            )
            calls = (
                (real_broker.assert_real_execution_ready, (), {}),
                (real_broker.issue_execution_receipt, (), {}),
                (
                    real_broker.open_row_blob,
                    (),
                    {
                        "root": self.v3_fixture.root,
                        "scope": self.scope,
                        "request": self._read_request(),
                        "authority_context": {},
                    },
                ),
                (
                    real_broker.open_output,
                    (),
                    {
                        "root": self.v3_fixture.root,
                        "scope": self.scope,
                        "request": write_request,
                        "payload": b"synthetic-only",
                        "authority_context": {},
                    },
                ),
                (real_broker.seal_failure, (), {}),
            )
            for function, args, kwargs in calls:
                with self.subTest(boundary=function.__name__):
                    self._assert_reason(
                        "REAL_EXECUTION_BACKEND_NOT_IMPLEMENTED",
                        function,
                        *args,
                        **kwargs,
                    )
            issuer.assert_not_called()
            reader.assert_not_called()
            writer.assert_not_called()
            sealer.assert_not_called()
        self.assertEqual(self.real_meter.real_data_rows_opened, 0)

    def test_35_synthetic_fixture_happy_path_passes(self) -> None:
        """§12 #35: isolated synthetic broker and supervisor happy paths pass."""

        fixture_broker, relative, digest, meter = self._synthetic_blob_broker()
        payload = fixture_broker.open_blob(
            path=relative,
            access_class="runner_row_blob",
            required_capability="read_real_runner_rows",
            expected_sha256=digest,
        )
        self.assertEqual(payload, b"synthetic row\n")
        outcome = self._synthetic_supervisor().run(
            plan=self._synthetic_plan("success.py"),
            sealer=self._synthetic_sealer(),
        )
        self.assertEqual(
            (outcome.status, outcome.reason_code, outcome.exit_code),
            ("success", "SYNTHETIC_SUPERVISED_RUN_OK", 0),
        )
        self.assertEqual(meter.real_data_rows_opened, 0)
        self.assertEqual(meter.synthetic_blobs_opened, 1)
        self.assertEqual(meter.bytes_delivered, len(payload))
        self.assertEqual(outcome.real_data_rows_opened, 0)

    def test_36_synthetic_backend_cannot_claim_real_status(self) -> None:
        """§12 #36: synthetic assurance cannot enter either real-data factory."""

        for factory in (
            broker.build_real_data_execution_broker,
            supervisor.build_real_data_execution_supervisor,
        ):
            with self.subTest(factory=factory.__name__):
                self._assert_reason(
                    "SYNTHETIC_BACKEND_NOT_REAL_AUTHORITY",
                    factory,
                    backend=self.synthetic_enforcement,
                )

        synthetic = self._synthetic_supervisor()
        mismatched_sealer = supervisor.SyntheticImmutableSealPort(
            root=self.synthetic_root,
            fixture_id="DIFFERENT-SYNTHETIC-FIXTURE",
        )
        self._assert_reason(
            "SYNTHETIC_FIXTURE_ID_MISMATCH",
            synthetic.run,
            plan=self._synthetic_plan("success.py"),
            sealer=mismatched_sealer,
        )
        self.assertEqual(synthetic.child_processes_started, 0)
        self.assertFalse((self.synthetic_root / "fixture-seal").exists())
        self.assertEqual(self.real_meter.real_data_rows_opened, 0)

    def test_37_real_data_rows_opened_remains_zero(self) -> None:
        """§12 #37: preflight and every unsupported real boundary keep the counter zero."""

        receipt = v3_contract.verify_metadata_preflight(
            self.scope,
            self.v3_fixture.catalog_payload(self.scope),
            row_loader=self.real_row_bomb,
        )
        self.assertEqual(receipt["real_data_rows_opened"], 0)
        self._assert_real_row_prevented()
        self.assertEqual(self.real_row_bomb.count, 0)
        self.assertEqual(self.real_meter.real_data_rows_opened, 0)

    def test_38_formal_buy_remains_false(self) -> None:
        """§12 #38: scope, receipt, status and synthetic seal keep formal_buy false."""

        changed = copy.deepcopy(self.scope)
        changed["formal_buy"] = True
        self._assert_reason(
            "RUN_SCOPE_NOT_EXACT_V3",
            supervisor.frozen_phase_command,
            changed,
            self._phase_id(),
        )
        report = broker.observe_real_host_enforcement().as_dict()
        self.assertFalse(report["formal_buy"])

    def test_39_send_order_remains_false(self) -> None:
        """§12 #39: scope and all boundary reports keep send_order false."""

        changed = copy.deepcopy(self.scope)
        changed["send_order"] = True
        self._assert_reason(
            "RUN_SCOPE_NOT_EXACT_V3",
            supervisor.frozen_phase_command,
            changed,
            self._phase_id(),
        )
        report = broker.observe_real_host_enforcement().as_dict()
        self.assertFalse(report["send_order"])

    def test_40_stake_remains_zero(self) -> None:
        """§12 #40: nonzero stake is rejected and reports retain exact integer zero."""

        for value in (1, True):
            with self.subTest(stake=value):
                changed = copy.deepcopy(self.scope)
                changed["stake"] = value
                self._assert_reason(
                    "RUN_SCOPE_NOT_EXACT_V3",
                    supervisor.frozen_phase_command,
                    changed,
                    self._phase_id(),
                )
        report = broker.observe_real_host_enforcement().as_dict()
        self.assertEqual(report["stake"], 0)
        self.assertIs(type(report["stake"]), int)

    def test_41_registry_bytes_are_unchanged(self) -> None:
        """§12 #41: focused tests do not alter the protected Registry byte stream."""

        path = ROOT / "research/REGISTRY.jsonl"
        before = path.read_bytes()
        self.assertEqual(self._digest(before), REGISTRY_SHA256)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self._digest(path.read_bytes()), REGISTRY_SHA256)

    def test_42_v2_golden_compatibility_is_unchanged(self) -> None:
        """§12 #42: protected v2 bytes, fields and four golden digests remain fixed."""

        for relative in (
            "scripts/research/scope_contract.py",
            "scripts/research/prepare_run_scope.py",
            "research/schemas/roi_reproduction_run_v2.schema.json",
            "research/schemas/roi_reproduction_proposal_v2.schema.json",
        ):
            with self.subTest(path=relative):
                self.assertEqual(
                    self._file_digest(ROOT / relative), PROTECTED_SHA256[relative]
                )
        self.assertEqual(
            scope_contract.RUN_FIELDS,
            {
                "proposal_scope",
                "proposal_scope_digest",
                "execution_commit_sha",
                "config_hashes",
                "data_input_manifest_hashes",
                "fold_manifest_hash",
                "runner_universe_manifest_hash",
                "dependency_environment_manifest",
                "seed",
                "exact_execution_commands",
                "formal_buy",
                "send_order",
                "stake",
            },
        )
        for experiment_id, expected in LEGACY_GOLDEN_DIGESTS.items():
            with self.subTest(experiment_id=experiment_id):
                proposal = scope_contract.normalize_proposal_scope(
                    scope_contract.strict_json_load(
                        ROOT / "research/scopes" / f"{experiment_id}.proposal.json"
                    ),
                    expected_experiment_id=experiment_id,
                )
                run_scope = scope_contract.normalize_run_scope(
                    scope_contract.strict_json_load(
                        ROOT / "research/scopes" / f"{experiment_id}.run.json"
                    ),
                    proposal_scope=proposal,
                )
                self.assertEqual(
                    (
                        scope_contract.canonical_digest(proposal),
                        scope_contract.canonical_digest(run_scope),
                    ),
                    expected,
                )

    def test_43_v3_canonical_digest_semantics_are_unchanged(self) -> None:
        """§12 #43: protected v3 bytes and canonical JSON/digest semantics remain fixed."""

        for relative in (
            "scripts/research/ordinary_real_data_run_contract_v3.py",
            "scripts/research/prepare_ordinary_real_data_run_scope_v3.py",
            "research/schemas/ordinary_real_data_run_v3.schema.json",
        ):
            with self.subTest(path=relative):
                self.assertEqual(
                    self._file_digest(ROOT / relative), PROTECTED_SHA256[relative]
                )
        value = {"z": [2, 1], "a": "日本語", "flag": False}
        expected_bytes = '{"a":"日本語","flag":false,"z":[2,1]}'.encode("utf-8")
        self.assertEqual(v3_contract.canonical_json_bytes(value), expected_bytes)
        self.assertEqual(
            v3_contract.canonical_digest(value),
            "cf55b0003f86c827b1f0a73d8a34781bd323b90f6262ea151344421e12ebd599",
        )
        normalized = v3_contract.normalize_ordinary_real_data_run_scope(
            self.scope, proposal_scope=self.scope["proposal_scope"]
        )
        self.assertEqual(normalized, self.scope)
        command_mutation = copy.deepcopy(self.scope)
        command_mutation["exact_commands"] = list(
            reversed(command_mutation["exact_commands"])
        )
        self.assertNotEqual(
            v3_contract.canonical_digest(command_mutation),
            v3_contract.canonical_digest(self.scope),
        )


if __name__ == "__main__":
    unittest.main()
