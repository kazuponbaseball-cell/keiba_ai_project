from __future__ import annotations

import contextlib
import copy
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

import create_infrastructure_experiment as create_cli
import infrastructure_safety_contract as contract
import prepare_infrastructure_run_scope as prepare_cli


EXPERIMENT_ID = "GOV-SYNTHETIC-001"
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
    **{path: str(index) * 40 for index, path in enumerate(LIFECYCLE_PATHS, start=3)},
}
MATERIAL_BLOBS = {
    "research/INFRASTRUCTURE_GATE.json": "7" * 40,
    "research/synthetic/input.json": "8" * 40,
    "research/synthetic/environment.json": "9" * 40,
}


def file_ref(root: Path, path: str) -> dict[str, str]:
    return {
        "path": path,
        "sha256": contract.sha256_file(root / path),
    }


def sample_proposal(
    policy_sha256: str,
    *,
    experiment_id: str = EXPERIMENT_ID,
    expected_paths: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "contract": dict(contract.PROPOSAL_CONTRACT),
        "gate_kind": contract.GATE_KIND,
        "experiment_id": experiment_id,
        "title": "Add one bounded infrastructure contract",
        "change_hypothesis": (
            "A strict synthetic-only gate can validate one bounded infrastructure change."
        ),
        "null_hypothesis": (
            "The gate accepts unsafe paths, commands, inputs, or capability flags."
        ),
        "safety_objective": (
            "Preserve Research OS approvals and ROI gates without any external side effect."
        ),
        "failure_modes": [
            "Changed-path manifest differs from committed Git trees",
            "Structured command rendering accepts a shell string",
        ],
        "in_scope": [
            "Canonical infrastructure proposal and run contracts",
            "Synthetic contract tests",
        ],
        "out_of_scope": [
            "External API and real-data execution",
            "Production, purchase, order, and notification behavior",
        ],
        "expected_changed_paths": list(expected_paths or EXPECTED_PATHS),
        "input_classes": ["git_tracked_contract", "synthetic_fixture"],
        "source_as_of": "2026-08-09T00:00:00+09:00",
        "lineage_hash_requirements": [
            "Base-to-execution Git blob manifest SHA-256",
            "Gate policy raw file SHA-256",
        ],
        "test_matrix": [
            {
                "test_id": "positive-valid-contract",
                "kind": "positive",
                "assertion": "A valid synthetic contract normalizes deterministically.",
            },
            {
                "test_id": "negative-unsafe-path",
                "kind": "negative",
                "assertion": "An unsafe changed path fails closed.",
            },
            {
                "test_id": "compat-roi-v1",
                "kind": "backward_compatibility",
                "assertion": "Legacy ROI proposal digests are not reinterpreted.",
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
            "Reject any committed diff mismatch",
        ],
        "stop_conditions": [
            "Stop on any safety flag change",
            "Stop on any non-synthetic execution kind",
        ],
        "compute_budget": {
            "maximum_runtime_minutes": 30,
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


def setup_synthetic_root(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    policy_source = REPO_ROOT / "research" / "INFRASTRUCTURE_GATE.json"
    policy_path = root / "research" / "INFRASTRUCTURE_GATE.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_bytes(policy_source.read_bytes())
    policy, policy_sha256 = contract.load_gate_policy(policy_path)
    proposal = contract.normalize_infrastructure_proposal(
        sample_proposal(policy_sha256),
        policy=policy,
        policy_sha256=policy_sha256,
    )
    for index, path in enumerate(ALL_CHANGED_PATHS):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if path == "research/infra_tests/test_example_infrastructure.py":
            target.write_text(
                "import unittest\n\n"
                "class SyntheticInfrastructureTests(unittest.TestCase):\n"
                "    def test_positive_valid_contract(self):\n"
                "        self.assertTrue(True)\n\n"
                "    def test_negative_unsafe_path(self):\n"
                "        self.assertTrue(True)\n\n"
                "    def test_compat_roi_v1(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
        else:
            target.write_text(f"# synthetic changed file {index}\n", encoding="utf-8")
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
    for path, value in fixtures.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return proposal, policy, policy_sha256


def sample_run_scope(
    root: Path,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
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
    plan = [
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
        "synthetic_input_hashes": [file_ref(root, "research/synthetic/input.json")],
        "changed_path_manifest": manifest,
        "changed_path_manifest_digest": contract.canonical_digest(manifest),
        "dependency_environment_manifest": file_ref(
            root, "research/synthetic/environment.json"
        ),
        "seed": 20260809,
        "command_plan": plan,
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


def fake_git_runner(
    blobs: dict[str, str] | None = None,
    *,
    mode: str = "100644",
    object_type: str = "blob",
    registry_execution_content: bytes | None = None,
    ancestor: bool = True,
):
    execution_blobs = dict(EXECUTION_BLOBS if blobs is None else blobs)
    tree_blobs = {**execution_blobs, **MATERIAL_BLOBS}

    def runner(root: Path, arguments: list[str]) -> bytes:
        if arguments[0] == "merge-base":
            if not ancestor:
                raise ValueError("synthetic commits are unrelated")
            return b""
        if arguments[0] == "diff":
            return b"".join(
                b"A\0" + path.encode("utf-8") + b"\0" for path in ALL_CHANGED_PATHS
            )
        if arguments[0] == "ls-tree":
            commit = arguments[2]
            path = arguments[-1]
            if commit == BASE_COMMIT:
                return b""
            if commit != EXECUTION_COMMIT or path not in tree_blobs:
                raise AssertionError(f"unexpected synthetic git query: {arguments}")
            return (
                f"{mode} {object_type} {tree_blobs[path]}\t{path}\0".encode("utf-8")
            )
        if arguments[0] == "cat-file":
            blob_sha = arguments[2]
            path = next(
                (
                    candidate
                    for candidate, candidate_sha in tree_blobs.items()
                    if candidate_sha == blob_sha
                ),
                None,
            )
            if path is None:
                raise AssertionError(f"unexpected synthetic blob query: {arguments}")
            if path == "research/REGISTRY.jsonl" and registry_execution_content is not None:
                return registry_execution_content
            return (root / path).read_bytes()
        raise AssertionError(f"unexpected synthetic git command: {arguments}")

    return runner


class InfrastructureSafetyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy_path = REPO_ROOT / "research" / "INFRASTRUCTURE_GATE.json"
        cls.policy, cls.policy_sha256 = contract.load_gate_policy(cls.policy_path)

    def test_01_policy_is_non_compensating_and_synthetic_only(self) -> None:
        self.assertEqual(self.policy["gate_kind"], contract.GATE_KIND)
        self.assertEqual(self.policy["mode"], "all_hard_checks_must_pass")
        self.assertEqual(
            set(self.policy["required_hard_checks"]),
            contract.HARD_CHECKS,
        )
        self.assertFalse(self.policy["safety"]["real_data_execution"])
        self.assertEqual(self.policy["safety"]["network_calls"], 0)

    def test_02_strict_json_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            contract.strict_json_loads('{"a":1,"a":2}')
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-standard JSON constant"):
                    contract.strict_json_loads(f'{{"value":{value}}}')

    def test_03_valid_proposal_has_all_hard_checks_and_no_numeric_score(self) -> None:
        proposal = contract.normalize_infrastructure_proposal(
            sample_proposal(self.policy_sha256),
            policy=self.policy,
            policy_sha256=self.policy_sha256,
        )
        evaluation = contract.evaluate_infrastructure_gate(proposal, policy=self.policy)
        self.assertTrue(evaluation["passed"])
        self.assertTrue(all(evaluation["checks"].values()))
        self.assertNotIn("score_components", proposal)
        self.assertNotIn("score_total", evaluation)

    def test_04_path_firewall_rejects_roots_tokens_ads_and_casefold_collisions(self) -> None:
        cases = (
            "scripts/research/update_registry.py",
            "scripts/research/infrastructure_safety_contract.py",
            "scripts/research/create_infrastructure_experiment.py",
            "scripts/research/prepare_infrastructure_run_scope.py",
            "research/schemas/infrastructure_safety_proposal_v1.schema.json",
            "research/schemas/infrastructure_safety_run_v1.schema.json",
            "tests/research/test_future_infrastructure.py",
            "src/example.py",
            "scripts/research/model_adapter.py",
            "scripts/research/predictorBuilder.py",
            "config/baseline_features.json",
            "config/safe.EXAMPLE.JSON",
            "scripts/research/example.py:stream",
            "scripts/research/example./file.py",
            "scripts/research/example.py ",
            "scripts//research/example.py",
            "scripts/research/mоdel_adapter.py",
            "scripts/research/example\n.py",
        )
        for path in cases:
            with self.subTest(path=path):
                proposal = sample_proposal(self.policy_sha256, expected_paths=[path])
                with self.assertRaises(ValueError):
                    contract.normalize_infrastructure_proposal(
                        proposal,
                        policy=self.policy,
                        policy_sha256=self.policy_sha256,
                    )
        allowed_config = sample_proposal(
            self.policy_sha256,
            expected_paths=["config/safe.example.json"],
        )
        self.assertEqual(
            contract.normalize_infrastructure_proposal(
                allowed_config,
                policy=self.policy,
                policy_sha256=self.policy_sha256,
            )["expected_changed_paths"],
            ["config/safe.example.json"],
        )
        collision = sample_proposal(
            self.policy_sha256,
            expected_paths=["scripts/research/Foo.py", "scripts/research/foo.py"],
        )
        with self.assertRaisesRegex(ValueError, "case-fold"):
            contract.normalize_infrastructure_proposal(
                collision,
                policy=self.policy,
                policy_sha256=self.policy_sha256,
            )

    def test_05_test_matrix_must_cover_all_three_kinds(self) -> None:
        proposal = sample_proposal(self.policy_sha256)
        proposal["test_matrix"] = proposal["test_matrix"][:2]
        with self.assertRaisesRegex(ValueError, "must include positive"):
            contract.normalize_infrastructure_proposal(
                proposal,
                policy=self.policy,
                policy_sha256=self.policy_sha256,
            )
        collision = sample_proposal(self.policy_sha256)
        collision["test_matrix"][1]["test_id"] = "positive_valid_contract"
        with self.assertRaisesRegex(ValueError, "unique unittest method names"):
            contract.normalize_infrastructure_proposal(
                collision,
                policy=self.policy,
                policy_sha256=self.policy_sha256,
            )

    def test_06_safety_variant_threshold_and_policy_hash_fail_closed(self) -> None:
        mutations = (
            ("formal_buy", True),
            ("send_order", True),
            ("stake", 1),
            ("allowed_variant_count", 2),
            ("allowed_threshold_search_count", 1),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                proposal = sample_proposal(self.policy_sha256)
                proposal[field] = value
                with self.assertRaises(ValueError):
                    contract.normalize_infrastructure_proposal(
                        proposal,
                        policy=self.policy,
                        policy_sha256=self.policy_sha256,
                    )
        proposal = sample_proposal("f" * 64)
        with self.assertRaisesRegex(ValueError, "policy hash differs"):
            contract.normalize_infrastructure_proposal(
                proposal,
                policy=self.policy,
                policy_sha256=self.policy_sha256,
            )

    def test_07_queue_and_initial_event_use_schema_v3_without_score(self) -> None:
        proposal = contract.normalize_infrastructure_proposal(
            sample_proposal(self.policy_sha256),
            policy=self.policy,
            policy_sha256=self.policy_sha256,
        )
        queue = contract.build_infrastructure_queue(
            proposal_scope=proposal,
            proposal_scope_file=f"research/scopes/{EXPERIMENT_ID}.proposal.json",
            experiment_markdown=f"research/experiments/{EXPERIMENT_ID}.md",
            owner="human-owner",
            created_at="2026-08-09T00:00:00Z",
            notes="synthetic",
            policy=self.policy,
        )
        queue = contract.normalize_infrastructure_queue(
            queue,
            policy=self.policy,
            policy_sha256=self.policy_sha256,
        )
        event = contract.build_initial_infrastructure_event(
            queue=queue,
            policy=self.policy,
            event_id="event-infra-1",
            occurred_at="2026-08-09T00:00:01Z",
            actor="codex",
        )
        self.assertEqual(queue["schema_version"], 3)
        self.assertEqual(event["schema_version"], 3)
        self.assertNotIn("score", queue)
        self.assertNotIn("score_total", event)
        self.assertIsNone(event["gate_policy_evidence"])
        self.assertFalse(event["execution_authorized"])
        wrong_scope_path = copy.deepcopy(queue)
        wrong_scope_path["proposal_scope_file"] = "research/scopes/alias.proposal.json"
        with self.assertRaisesRegex(ValueError, "code-owned lifecycle path"):
            contract.normalize_infrastructure_queue(
                wrong_scope_path,
                policy=self.policy,
                policy_sha256=self.policy_sha256,
            )

    def test_08_event_gate_policy_evidence_is_strict_and_running_is_synthetic(self) -> None:
        proposal = contract.normalize_infrastructure_proposal(
            sample_proposal(self.policy_sha256),
            policy=self.policy,
            policy_sha256=self.policy_sha256,
        )
        queue = contract.build_infrastructure_queue(
            proposal_scope=proposal,
            proposal_scope_file=f"research/scopes/{EXPERIMENT_ID}.proposal.json",
            experiment_markdown=f"research/experiments/{EXPERIMENT_ID}.md",
            owner="human-owner",
            created_at="2026-08-09T00:00:00Z",
            notes="",
            policy=self.policy,
        )
        event = contract.build_initial_infrastructure_event(
            queue=queue,
            policy=self.policy,
            event_id="event-infra-1",
            occurred_at="2026-08-09T00:00:01Z",
            actor="codex",
        )
        event.update(
            {
                "sequence": 2,
                "status": "running",
                "previous_status": "approved_to_run",
                "previous_event_id": "event-approved-run",
                "run_scope_digest": "c" * 64,
                "gate_policy_evidence": {
                    "path": "research/INFRASTRUCTURE_GATE.json",
                    "ref": BASE_COMMIT,
                    "blob_sha": "d" * 40,
                    "content_sha256": self.policy_sha256,
                },
                "main_registry_evidence": {
                    "path": "research/REGISTRY.jsonl",
                    "ref": BASE_COMMIT,
                    "blob_sha": "e" * 40,
                    "content_sha256": "e" * 64,
                },
                "human_approved": True,
                "human_prepare_approval_recorded": True,
                "human_run_approval_recorded": True,
                "synthetic_fixture_tests_allowed": False,
                "automatic_execution_allowed": False,
                "execution_authorized": False,
                "execution_kind": "synthetic",
            }
        )
        normalized = contract.normalize_infrastructure_event(event, policy=self.policy)
        self.assertEqual(normalized["gate_policy_evidence"]["ref"], BASE_COMMIT)
        self.assertFalse(normalized["preparation_authorized"])
        self.assertFalse(normalized["execution_authorized"])
        unsafe = copy.deepcopy(event)
        unsafe["execution_kind"] = "real-data"
        unsafe["real_data_execution_allowed"] = True
        with self.assertRaises(ValueError):
            contract.normalize_infrastructure_event(unsafe, policy=self.policy)
        shadow = copy.deepcopy(event)
        shadow["status"] = "approved_for_shadow"
        with self.assertRaisesRegex(ValueError, "unsupported infrastructure status"):
            contract.normalize_infrastructure_event(shadow, policy=self.policy)
        malformed = copy.deepcopy(event)
        malformed["gate_policy_evidence"]["extra"] = True
        with self.assertRaisesRegex(ValueError, "unexpected field"):
            contract.normalize_infrastructure_event(malformed, policy=self.policy)
        missing = copy.deepcopy(event)
        missing["gate_policy_evidence"] = None
        with self.assertRaisesRegex(ValueError, "is required"):
            contract.normalize_infrastructure_event(missing, policy=self.policy)
        proposed_with_evidence = contract.build_initial_infrastructure_event(
            queue=queue,
            policy=self.policy,
            event_id="event-infra-proposed",
            occurred_at="2026-08-09T00:00:02Z",
            actor="codex",
        )
        proposed_with_evidence["gate_policy_evidence"] = event["gate_policy_evidence"]
        with self.assertRaisesRegex(ValueError, "must be null"):
            contract.normalize_infrastructure_event(proposed_with_evidence, policy=self.policy)
        wrong_queue_path = copy.deepcopy(event)
        wrong_queue_path["queue_file"] = "research/queue/alias.json"
        with self.assertRaisesRegex(ValueError, "event.queue_file must be"):
            contract.normalize_infrastructure_event(wrong_queue_path, policy=self.policy)

    def test_09_valid_run_scope_uses_structured_templates_and_synthetic_locks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal, policy, policy_sha256 = setup_synthetic_root(root)
            run = sample_run_scope(root, proposal)
            normalized = contract.normalize_infrastructure_run_scope(
                run,
                proposal_scope=proposal,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            contract.verify_infrastructure_run_materials(root, normalized)
        self.assertEqual(
            {entry["path"] for entry in normalized["changed_path_manifest"]["entries"]},
            set(EXPECTED_PATHS) | set(LIFECYCLE_PATHS),
        )
        self.assertEqual(normalized["execution_kind"], "synthetic")
        self.assertFalse(normalized["real_data_execution"])
        self.assertEqual(normalized["network_calls"], 0)
        self.assertEqual(
            normalized["exact_execution_argv"][0][0:6],
            [
                contract.bound_python_executable(),
                "-B",
                "-I",
                "-S",
                "-m",
                "unittest",
            ],
        )
        self.assertEqual(
            normalized["execution_context"],
            {
                "working_directory": "repository_root",
                "inherit_environment": False,
                "environment": {},
                "timeout_seconds": 1800,
                "network_access": False,
                "credential_environment_access": False,
                "filesystem_write_paths": [],
            },
        )

    def test_10_unknown_template_free_form_and_exact_argv_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal, policy, policy_sha256 = setup_synthetic_root(root)
            for case in ("unknown", "free-form", "argv", "context"):
                with self.subTest(case=case):
                    run = sample_run_scope(root, proposal)
                    if case == "unknown":
                        run["command_plan"][0]["template_id"] = "shell_v1"
                    elif case == "free-form":
                        run["command_plan"][0]["parameters"]["command"] = "pwsh -Command whoami"
                    elif case == "argv":
                        run["exact_execution_argv"][0] = ["pwsh", "-Command", "whoami"]
                    else:
                        run["execution_context"]["timeout_seconds"] += 1
                    with self.assertRaises(ValueError):
                        contract.normalize_infrastructure_run_scope(
                            run,
                            proposal_scope=proposal,
                            policy=policy,
                            policy_sha256=policy_sha256,
                        )

    def test_10_command_plan_is_bound_to_retained_changed_test_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal, policy, policy_sha256 = setup_synthetic_root(root)
            missing = sample_run_scope(root, proposal)
            missing["command_plan"][0]["parameters"]["module"] = (
                "research.infra_tests.test_missing"
            )
            with self.assertRaisesRegex(ValueError, "actual retained changed"):
                contract.normalize_infrastructure_run_scope(
                    missing,
                    proposal_scope=proposal,
                    policy=policy,
                    policy_sha256=policy_sha256,
                )

            discovery = sample_run_scope(root, proposal)
            discovery["command_plan"] = [
                {
                    "template_id": "python_unittest_discover_infrastructure_v1",
                    "parameters": {},
                }
            ]
            discovery["exact_execution_argv"] = [
                [
                    contract.bound_python_executable(),
                    "-B",
                    "-I",
                    "-S",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "research/infra_tests",
                    "-p",
                    "test_*.py",
                ]
            ]
            normalized = contract.normalize_infrastructure_run_scope(
                discovery,
                proposal_scope=proposal,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            contract.verify_infrastructure_commit_diff(
                root,
                normalized,
                runner=fake_git_runner(),
                policy=policy,
            )

    def test_11_changed_path_manifest_requires_exact_paths_types_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal, policy, policy_sha256 = setup_synthetic_root(root)
            cases = ("missing", "type", "wrong-lifecycle", "digest")
            for case in cases:
                with self.subTest(case=case):
                    run = sample_run_scope(root, proposal)
                    if case == "missing":
                        run["changed_path_manifest"]["entries"].pop()
                    elif case == "type":
                        run["changed_path_manifest"]["entries"][0]["change_type"] = "modified"
                    elif case == "wrong-lifecycle":
                        lifecycle_entry = next(
                            entry
                            for entry in run["changed_path_manifest"]["entries"]
                            if entry["path"] == f"research/queue/{EXPERIMENT_ID}.json"
                        )
                        lifecycle_entry["path"] = "research/queue/GOV-SYNTHETIC-OTHER.json"
                    else:
                        run["changed_path_manifest_digest"] = "f" * 64
                    if case != "digest":
                        run["changed_path_manifest_digest"] = contract.canonical_digest(
                            run["changed_path_manifest"]
                        )
                    with self.assertRaises(ValueError):
                        contract.normalize_infrastructure_run_scope(
                            run,
                            proposal_scope=proposal,
                            policy=policy,
                            policy_sha256=policy_sha256,
                        )

    def test_12_commit_diff_is_re_read_and_must_exactly_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal, policy, policy_sha256 = setup_synthetic_root(root)
            normalized = contract.normalize_infrastructure_run_scope(
                sample_run_scope(root, proposal),
                proposal_scope=proposal,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            actual = contract.verify_infrastructure_commit_diff(
                root,
                normalized,
                runner=fake_git_runner(),
                policy=policy,
            )
            self.assertEqual(actual, normalized["changed_path_manifest"])
            bad_blobs = dict(EXECUTION_BLOBS)
            bad_blobs[EXPECTED_PATHS[0]] = "9" * 40
            with self.assertRaisesRegex(ValueError, "diff differs"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(bad_blobs),
                    policy=policy,
                )
            with self.assertRaisesRegex(ValueError, "not a descendant"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(ancestor=False),
                    policy=policy,
                )

    def test_13_commit_diff_rejects_symlinks_and_submodules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal, policy, policy_sha256 = setup_synthetic_root(root)
            normalized = contract.normalize_infrastructure_run_scope(
                sample_run_scope(root, proposal),
                proposal_scope=proposal,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(mode="120000"),
                    policy=policy,
                )
            with self.assertRaisesRegex(ValueError, "submodules"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(mode="160000", object_type="commit"),
                    policy=policy,
                )

    def test_13_registry_rewrite_and_forbidden_python_import_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal, policy, policy_sha256 = setup_synthetic_root(root)
            normalized = contract.normalize_infrastructure_run_scope(
                sample_run_scope(root, proposal),
                proposal_scope=proposal,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            registry_path = root / "research/REGISTRY.jsonl"
            execution_registry = registry_path.read_bytes()
            registry_path.write_text("rewritten\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rewrites or deletes"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(
                        registry_execution_content=execution_registry,
                    ),
                    policy=policy,
                )

            registry_path.write_bytes(execution_registry)
            changed_source = root / EXPECTED_PATHS[0]
            changed_source.write_text("import subprocess\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden capability module"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(),
                    policy=policy,
                )
            changed_source.write_text("from pathlib import Path\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the pure allowlist"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(),
                    policy=policy,
                )
            changed_source.write_text(
                "def predictorBuilder():\n    return None\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "forbidden capability symbol"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(),
                    policy=policy,
                )
            changed_source.write_text(
                "import unittest\nobserved = unittest.main.__globals__\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "forbidden dunder symbol"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(),
                    policy=policy,
                )
            changed_source.write_text(
                'name = "__globals__"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "forbidden dunder string"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(),
                    policy=policy,
                )
            changed_source.write_text(
                'import json\nobserved = json.__dict__["loads"]("{}")\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "non-static dynamic call target|forbidden dunder symbol",
            ):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(),
                    policy=policy,
                )
            changed_source.write_text(
                'import json\nobserved = {"load": json.loads}["load"]("{}")\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-static dynamic call target"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(),
                    policy=policy,
                )
            changed_source.write_text(
                'import operator\nobserved = operator.attrgetter("__globals__")(object())\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "outside the pure allowlist|non-static dynamic call target",
            ):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(),
                    policy=policy,
                )
            changed_source.write_text(
                'import unittest\nif __name__ == "__main__":\n    unittest.main()\n',
                encoding="utf-8",
            )
            contract.verify_infrastructure_commit_diff(
                root,
                normalized,
                runner=fake_git_runner(),
                policy=policy,
            )

    def test_13_committed_test_matrix_evidence_is_exact_and_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal, policy, policy_sha256 = setup_synthetic_root(root)
            normalized = contract.normalize_infrastructure_run_scope(
                sample_run_scope(root, proposal),
                proposal_scope=proposal,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            test_path = root / "research/infra_tests/test_example_infrastructure.py"
            test_path.write_text(
                "import unittest\n\n"
                "class EmptyInfrastructureTests(unittest.TestCase):\n"
                "    pass\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "zero discoverable TestCase methods"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(),
                    policy=policy,
                )

            test_path.write_text(
                "import unittest\n\n"
                "class IncompleteInfrastructureTests(unittest.TestCase):\n"
                "    def test_positive_valid_contract(self):\n"
                "        self.assertTrue(True)\n\n"
                "    def test_negative_unsafe_path(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "map to exactly one committed"):
                contract.verify_infrastructure_commit_diff(
                    root,
                    normalized,
                    runner=fake_git_runner(),
                    policy=policy,
                )

    def test_14_material_hash_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal, policy, policy_sha256 = setup_synthetic_root(root)
            normalized = contract.normalize_infrastructure_run_scope(
                sample_run_scope(root, proposal),
                proposal_scope=proposal,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            (root / "research/synthetic/input.json").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                contract.verify_infrastructure_run_materials(root, normalized)

    def test_14_material_paths_are_safe_and_links_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal, policy, policy_sha256 = setup_synthetic_root(root)

            unsafe_synthetic = sample_run_scope(root, proposal)
            unsafe_synthetic["synthetic_input_hashes"] = [
                {"path": "data/real.json", "sha256": "f" * 64}
            ]
            with self.assertRaisesRegex(ValueError, "outside the synthetic material"):
                contract.normalize_infrastructure_run_scope(
                    unsafe_synthetic,
                    proposal_scope=proposal,
                    policy=policy,
                    policy_sha256=policy_sha256,
                )

            unsafe_config = sample_run_scope(root, proposal)
            unsafe_config["config_hashes"].append(
                {"path": "config/active.json", "sha256": "e" * 64}
            )
            with self.assertRaisesRegex(ValueError, "exact .example.json suffix"):
                contract.normalize_infrastructure_run_scope(
                    unsafe_config,
                    proposal_scope=proposal,
                    policy=policy,
                    policy_sha256=policy_sha256,
                )

            unsafe_environment = sample_run_scope(root, proposal)
            unsafe_environment["dependency_environment_manifest"] = {
                "path": "research/drafts/environment.json",
                "sha256": "d" * 64,
            }
            with self.assertRaisesRegex(ValueError, "outside the synthetic material"):
                contract.normalize_infrastructure_run_scope(
                    unsafe_environment,
                    proposal_scope=proposal,
                    policy=policy,
                    policy_sha256=policy_sha256,
                )

            input_path = root / "research/synthetic/input.json"
            safe_input = input_path.read_bytes()
            input_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "fixture_kind": "synthetic_input",
                        "synthetic": True,
                        "provenance": "code_owned_synthetic_fixture_v1",
                        "contains_real_data": False,
                        "contains_credentials": False,
                        "payload": {"race_id": "copied-real-row"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            real_shaped = contract.normalize_infrastructure_run_scope(
                sample_run_scope(root, proposal),
                proposal_scope=proposal,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            with self.assertRaisesRegex(ValueError, "row-level real-data shape"):
                contract.verify_infrastructure_run_materials(root, real_shaped)
            input_path.write_bytes(safe_input)

            config_path = root / "config/safe.example.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text('{"api_key":"not-a-real-key"}\n', encoding="utf-8")
            unsafe_config_content = sample_run_scope(root, proposal)
            unsafe_config_content["config_hashes"].append(file_ref(root, "config/safe.example.json"))
            unsafe_config_content = contract.normalize_infrastructure_run_scope(
                unsafe_config_content,
                proposal_scope=proposal,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            with self.assertRaisesRegex(ValueError, "populated sensitive key"):
                contract.verify_infrastructure_run_materials(root, unsafe_config_content)

            normalized = contract.normalize_infrastructure_run_scope(
                sample_run_scope(root, proposal),
                proposal_scope=proposal,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            target_path = root / "research/synthetic/target.json"
            target_path.write_bytes(input_path.read_bytes())
            input_path.unlink()
            try:
                input_path.symlink_to(target_path)
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "symlink or junction"):
                contract.verify_infrastructure_run_materials(root, normalized)

    def test_15_create_cli_dry_run_writes_nothing_and_reports_no_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "research").mkdir(parents=True)
            (root / "research/INFRASTRUCTURE_GATE.json").write_bytes(
                self.policy_path.read_bytes()
            )
            (root / "research/INFRASTRUCTURE_EXPERIMENT_TEMPLATE.md").write_bytes(
                (REPO_ROOT / "research/INFRASTRUCTURE_EXPERIMENT_TEMPLATE.md").read_bytes()
            )
            policy_hash = contract.sha256_file(root / "research/INFRASTRUCTURE_GATE.json")
            input_path = root / "proposal_input.json"
            input_path.write_text(
                json.dumps(sample_proposal(policy_hash), ensure_ascii=False),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = create_cli.main(
                    [
                        EXPERIMENT_ID,
                        "--proposal-scope",
                        str(input_path),
                        "--root",
                        str(root),
                        "--dry-run",
                    ]
                )
            result = json.loads(stdout.getvalue())
            self.assertEqual(rc, 0)
            self.assertIsNone(result["numeric_score"])
            self.assertFalse((root / f"research/queue/{EXPERIMENT_ID}.json").exists())

            alternate_policy = root / "alternate-policy.json"
            alternate_policy.write_bytes(self.policy_path.read_bytes())
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                create_cli.main(
                    [
                        EXPERIMENT_ID,
                        "--proposal-scope",
                        str(input_path),
                        "--root",
                        str(root),
                        "--policy",
                        str(alternate_policy),
                        "--dry-run",
                    ]
                )
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("--policy must name the exact code-owned path", stderr.getvalue())

    def test_16_prepare_cli_dry_run_rechecks_git_and_executes_no_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal, policy, policy_sha256 = setup_synthetic_root(root)
            proposal_path = root / f"research/scopes/{EXPERIMENT_ID}.proposal.json"
            queue_path = root / f"research/queue/{EXPERIMENT_ID}.json"
            proposal_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            proposal_path.write_text(contract.canonical_json_text(proposal) + "\n", encoding="utf-8")
            queue = contract.build_infrastructure_queue(
                proposal_scope=proposal,
                proposal_scope_file=f"research/scopes/{EXPERIMENT_ID}.proposal.json",
                experiment_markdown=f"research/experiments/{EXPERIMENT_ID}.md",
                owner="human-owner",
                created_at="2026-08-09T00:00:00Z",
                notes="",
                policy=policy,
            )
            queue_path.write_text(json.dumps(queue, ensure_ascii=False), encoding="utf-8")
            run_input = root / "run_input.json"
            run_input.write_text(
                contract.canonical_json_text(sample_run_scope(root, proposal)) + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = prepare_cli.main(
                    [
                        EXPERIMENT_ID,
                        "--input",
                        str(run_input),
                        "--root",
                        str(root),
                        "--dry-run",
                    ],
                    git_runner=fake_git_runner(),
                )
            result = json.loads(stdout.getvalue())
            self.assertEqual(rc, 0)
            self.assertEqual(result["commands_executed"], 0)
            self.assertFalse(result["execution_authorized"])
            self.assertFalse((root / f"research/scopes/{EXPERIMENT_ID}.run.json").exists())

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                prepare_cli.main(
                    [
                        EXPERIMENT_ID,
                        "--input",
                        str(run_input),
                        "--root",
                        str(root),
                        "--output",
                        str(root / "alternate.run.json"),
                        "--dry-run",
                    ],
                    git_runner=fake_git_runner(),
                )
            self.assertEqual(caught.exception.code, 2)
            self.assertIn(
                "--output must name the exact code-owned lifecycle path",
                stderr.getvalue(),
            )

    def test_17_policy_and_schema_files_are_strict_json(self) -> None:
        paths = (
            REPO_ROOT / "research/INFRASTRUCTURE_GATE.json",
            REPO_ROOT / "research/schemas/infrastructure_safety_proposal_v1.schema.json",
            REPO_ROOT / "research/schemas/infrastructure_safety_run_v1.schema.json",
        )
        for path in paths:
            with self.subTest(path=path.name):
                parsed = contract.strict_json_load(path)
                self.assertIsInstance(parsed, dict)

    def test_18_create_cli_rejects_linked_output_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "research").mkdir(parents=True)
            (root / "research/INFRASTRUCTURE_GATE.json").write_bytes(
                self.policy_path.read_bytes()
            )
            (root / "research/INFRASTRUCTURE_EXPERIMENT_TEMPLATE.md").write_bytes(
                (REPO_ROOT / "research/INFRASTRUCTURE_EXPERIMENT_TEMPLATE.md").read_bytes()
            )
            policy_hash = contract.sha256_file(root / "research/INFRASTRUCTURE_GATE.json")
            proposal_input = root / "proposal_input.json"
            proposal_input.write_text(
                json.dumps(sample_proposal(policy_hash), ensure_ascii=False),
                encoding="utf-8",
            )
            redirected = root / "redirected-scopes"
            redirected.mkdir()
            try:
                (root / "research/scopes").symlink_to(
                    redirected,
                    target_is_directory=True,
                )
            except OSError as exc:
                self.skipTest(f"directory symlink creation is unavailable: {exc}")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                create_cli.main(
                    [
                        EXPERIMENT_ID,
                        "--proposal-scope",
                        str(proposal_input),
                        "--root",
                        str(root),
                        "--dry-run",
                    ]
                )
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("symlink or junction", stderr.getvalue())
            self.assertEqual(list(redirected.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
