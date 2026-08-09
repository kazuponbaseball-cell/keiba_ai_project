from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import scope_contract
import infrastructure_safety_contract


MODULE_PATH = SCRIPT_DIR / "update_registry.py"
REGISTRY_PATH = ROOT / "research" / "REGISTRY.jsonl"
SPEC = importlib.util.spec_from_file_location("update_registry_jsonl_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
update_registry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(update_registry)


REQUIRED_EVENT_TYPES: dict[str, type[object]] = {
    "schema_version": int,
    "event_id": str,
    "sequence": int,
    "experiment_id": str,
    "status": str,
    "occurred_at": str,
    "actor": str,
    "score_components": dict,
    "score_total": int,
    "score_threshold": int,
    "score_threshold_met": bool,
    "proposal_scope_digest": str,
    "revalidated_approval_evidence": list,
    "human_approved": bool,
    "human_prepare_approval_recorded": bool,
    "human_run_approval_recorded": bool,
    "human_shadow_approval_recorded": bool,
    "preparation_authorized": bool,
    "synthetic_fixture_tests_allowed": bool,
    "real_data_execution_allowed": bool,
    "automatic_execution_allowed": bool,
    "execution_authorized": bool,
    "production_approved": bool,
    "merge_approved": bool,
    "buy_approved": bool,
    "production_change_allowed": bool,
    "merge_allowed": bool,
    "buy_logic_change_allowed": bool,
    "formal_buy": bool,
    "send_order": bool,
    "stake": int,
    "execution_kind": str,
    "artifacts": list,
    "notes": str,
    "queue_file": str,
}
V2_EVENT_FIELDS = set(REQUIRED_EVENT_TYPES) | {
    "previous_status",
    "previous_event_id",
    "run_scope_digest",
    "review_digest",
    "github_trust_evidence",
    "approval_evidence",
    "run_scope_file",
}


class RegistryMainProvider:
    """Read-only in-memory GitHub fixture with an immutable-current-main view."""

    repository = "kazuponbaseball-cell/keiba_ai_project"
    base_commit = "a" * 40

    def __init__(self) -> None:
        self.current_main = self.base_commit
        self.registry_content = b""
        self.approvers_content = json.dumps(
            {
                "schema_version": 1,
                "approvers": [{"login": "registry-human"}],
                "denied_login_patterns": ["bot", "codex", "automation"],
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def get_repository(self, repository: str) -> dict[str, object]:
        return {"full_name": self.repository, "default_branch": "main"}

    def get_branch_ref(self, repository: str, branch: str) -> dict[str, object]:
        return {
            "ref": "refs/heads/main",
            "object": {"type": "commit", "sha": self.current_main},
        }

    def compare_commits(
        self,
        repository: str,
        base_commit: str,
        head_commit: str,
    ) -> dict[str, object]:
        return {
            "status": "identical" if head_commit == base_commit else "ahead",
            "url": (
                f"https://api.github.com/repos/{self.repository}/compare/"
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
    ) -> dict[str, object]:
        if path == "research/APPROVERS.json":
            content = self.approvers_content
            blob_sha = "c" * 40
        elif path == "research/REGISTRY.jsonl":
            content = self.registry_content
            blob_sha = "d" * 40
        else:
            raise AssertionError(f"unexpected GitHub fixture path: {path}")
        return {
            "type": "file",
            "path": path,
            "encoding": "base64",
            "sha": blob_sha,
            "content": base64.b64encode(content).decode("ascii"),
        }

    def get_issue_comment(self, repository: str, comment_id: int) -> dict[str, object]:
        raise AssertionError("comment lookup is not expected")

    def merge_registry(self, content: bytes) -> None:
        self.registry_content = content
        self.current_main = hashlib.sha256(content).hexdigest()[:40]


def strict_json_loads(raw: str) -> object:
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(raw, parse_constant=reject_nonstandard_constant)


def approval_evidence(
    comment_id: int,
    approval_type: str = "APPROVED_TO_PREPARE",
    approval_digest: str = "e" * 64,
) -> dict[str, object]:
    body = f"{approval_type} {approval_digest}"
    return {
        "approval_type": approval_type,
        "approval_digest": approval_digest,
        "repository": "kazuponbaseball-cell/keiba_ai_project",
        "issue_number": 17,
        "comment_id": comment_id,
        "url": (
            "https://github.com/kazuponbaseball-cell/keiba_ai_project/"
            f"issues/17#issuecomment-{comment_id}"
        ),
        "author": "kazuponbaseball-cell",
        "author_type": "User",
        "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "created_at": "2026-07-31T00:00:00Z",
        "updated_at": "2026-07-31T00:00:00Z",
    }


def proposal(experiment_id: str, fold_ref: dict[str, str]) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "title": "Registry test",
        "hypothesis": "A hypothesis.",
        "null_hypothesis": "No improvement.",
        "racing_mechanism": "A mechanism.",
        "target_population": "A population.",
        "in_scope": ["research"],
        "out_of_scope": ["production"],
        "expected_changed_paths": ["scripts/research/test.py"],
        "raw_data_sources": ["source"],
        "data_as_of": "T-3",
        "allowed_columns": ["race_id"],
        "forbidden_columns": ["odds"],
        "lineage_hash_requirements": ["sha256"],
        "chronological_fold_design": {"order": "strict"},
        "fold_manifest": fold_ref,
        "purge_embargo": {"days": 1},
        "primary_metric": {"name": "nll"},
        "required_effect": {"delta_lte": -0.001},
        "rejection_gate": ["delta >= 0"],
        "stop_conditions": ["contract failure"],
        "compute_budget": {"minutes": 1},
        "allowed_variant_count": 1,
        "allowed_threshold_search_count": 0,
        "base_commit": "a" * 40,
        "score_components": {
            "independent_information": 25,
            "racing_mechanism": 20,
            "outer_oos_failure_evidence": 20,
            "leakage_safety": 10,
            "minimal_falsifiability": 0,
            "acquisition_implementation_cost": 0,
        },
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def make_queue(root: Path, experiment_id: str) -> Path:
    fold = root / "research" / "manifests" / "fold.json"
    fold.parent.mkdir(parents=True, exist_ok=True)
    fold.write_text('{"fold":1}\n', encoding="utf-8")
    fold_ref = {
        "path": "research/manifests/fold.json",
        "sha256": scope_contract.sha256_file(fold),
    }
    scope = scope_contract.normalize_proposal_scope(
        proposal(experiment_id, fold_ref),
        expected_experiment_id=experiment_id,
    )
    scope_path = root / "research" / "scopes" / f"{experiment_id}.proposal.json"
    scope_path.parent.mkdir(parents=True, exist_ok=True)
    scope_path.write_text(scope_contract.canonical_json_text(scope) + "\n", encoding="utf-8")
    digest = scope_contract.canonical_digest(scope)
    queue = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "status": "proposed",
        "score": {
            "components": scope["score_components"],
            "total": sum(scope["score_components"].values()),
        },
        "proposal_scope": scope,
        "proposal_scope_file": scope_path.relative_to(root).as_posix(),
        "proposal_scope_digest": digest,
        "human_approved_to_prepare": False,
        "human_approved_to_run": False,
        "human_approved_for_shadow": False,
        "automatic_execution_allowed": False,
        "execution_authorized": False,
        "production_approved": False,
        "merge_approved": False,
        "buy_approved": False,
        "production_change_allowed": False,
        "merge_allowed": False,
        "buy_logic_change_allowed": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    queue_path = root / "research" / "queue" / f"{experiment_id}.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(queue) + "\n", encoding="utf-8")
    return queue_path


class RegistryJsonlTests(unittest.TestCase):
    def assert_registry_event_schema(self, event: object, *, location: str) -> None:
        self.assertIsInstance(event, dict, f"{location} must contain an object")
        assert isinstance(event, dict)
        schema_version = event.get("schema_version")
        if schema_version == 3:
            policy, _ = infrastructure_safety_contract.load_gate_policy(
                ROOT / "research" / "INFRASTRUCTURE_GATE.json"
            )
            normalized = infrastructure_safety_contract.normalize_infrastructure_event(
                event,
                policy=policy,
            )
            self.assertEqual(normalized, event, location)
            self.assertEqual(event.get("gate_kind"), "infrastructure_safety_v1")
            self.assertNotIn("score_total", event)
            return
        self.assertEqual(schema_version, 2, location)
        self.assertEqual(set(event), V2_EVENT_FIELDS, location)
        for field, expected_type in REQUIRED_EVENT_TYPES.items():
            self.assertIn(field, event, f"{location} missing {field}")
            self.assertIs(type(event[field]), expected_type, f"{location}.{field}")
        for field in (
            "previous_status",
            "previous_event_id",
            "run_scope_digest",
            "review_digest",
            "github_trust_evidence",
            "approval_evidence",
            "run_scope_file",
        ):
            self.assertIn(field, event)
        self.assertIn(event["status"], update_registry.ALLOWED_STATUSES)
        self.assertRegex(event["proposal_scope_digest"], r"^[0-9a-f]{64}$")
        self.assertEqual(event["score_total"], sum(event["score_components"].values()))
        self.assertTrue(all(isinstance(item, str) for item in event["artifacts"]))
        self.assertTrue(
            all(isinstance(item, dict) for item in event["revalidated_approval_evidence"])
        )

    def test_committed_registry_is_valid_jsonl(self) -> None:
        self.assertTrue(REGISTRY_PATH.is_file())
        for line_number, raw in enumerate(
            REGISTRY_PATH.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            try:
                event = strict_json_loads(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                self.fail(f"{REGISTRY_PATH}:{line_number} invalid JSON: {exc}")
            self.assert_registry_event_schema(event, location=f"line {line_number}")

    def test_load_events_accepts_empty_and_valid_minimal_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.jsonl"
            path.write_text("\n", encoding="utf-8")
            self.assertEqual(update_registry.load_events(path), [])
            events = [
                {"experiment_id": "EXP-001", "status": "proposed"},
                {"experiment_id": "EXP-001", "status": "invalid"},
            ]
            path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(update_registry.load_events(path), events)

    def test_load_events_rejects_malformed_non_object_and_missing_identity(self) -> None:
        cases = (
            ('{"experiment_id":"EXP-001"\n', "invalid JSONL"),
            ("[]\n", "is not an object"),
            ('{"status":"proposed"}\n', "lacks experiment_id/status"),
        )
        for raw, message in cases:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "registry.jsonl"
                path.write_text(raw, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    update_registry.load_events(path)

    def test_load_events_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        cases = (
            (
                '{"experiment_id":"EXP-001","experiment_id":"EXP-002","status":"proposed"}\n',
                "duplicate JSON object key",
            ),
            (
                '{"experiment_id":"EXP-001","status":"proposed","score_total":NaN}\n',
                "non-standard JSON constant",
            ),
        )
        for raw, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "registry.jsonl"
                path.write_text(raw, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    update_registry.load_events(path)

    def test_grant_history_rejects_malformed_or_duplicate_comment_ids(self) -> None:
        valid_evidence = approval_evidence(101)
        malformed_cases = (
            (
                [{"experiment_id": "EXP-001", "status": "approved_to_prepare"}],
                "missing or has malformed approval_evidence",
            ),
            (
                [
                    {
                        "experiment_id": "EXP-001",
                        "status": "approved_to_prepare",
                        "approval_evidence": approval_evidence(0),
                    }
                ],
                "invalid approval comment ID",
            ),
            (
                [
                    {
                        "experiment_id": "EXP-001",
                        "status": "approved_to_prepare",
                        "approval_evidence": approval_evidence(
                            101, approval_type="APPROVED_TO_RUN"
                        ),
                    }
                ],
                "mismatched approval evidence",
            ),
            (
                [
                    {
                        "experiment_id": "EXP-001",
                        "status": "approved_to_prepare",
                        "proposal_scope_digest": "f" * 64,
                        "approval_evidence": approval_evidence(101),
                    }
                ],
                "digest is not bound to its event",
            ),
        )
        for events, message in malformed_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    update_registry.validate_approval_grant_history(events)

        duplicate_events = [
            {
                "experiment_id": "EXP-001",
                "status": "approved_to_prepare",
                "proposal_scope_digest": "e" * 64,
                "approval_evidence": dict(valid_evidence),
            },
            {
                "experiment_id": "EXP-002",
                "status": "approved_to_prepare",
                "proposal_scope_digest": "e" * 64,
                "approval_evidence": dict(valid_evidence),
            },
        ]
        with self.assertRaisesRegex(ValueError, "reused approval comment ID"):
            update_registry.validate_approval_grant_history(duplicate_events)

    def test_revalidation_evidence_does_not_consume_comment_id(self) -> None:
        evidence = approval_evidence(101)
        events = [
            {
                "experiment_id": "EXP-001",
                "status": "approved_to_prepare",
                "proposal_scope_digest": "e" * 64,
                "approval_evidence": evidence,
            },
            {
                "experiment_id": "EXP-001",
                "status": "preparing",
                "approval_evidence": None,
                "revalidated_approval_evidence": [evidence],
            },
        ]
        consumed = update_registry.validate_approval_grant_history(events)
        self.assertEqual(set(consumed), {101})

    def test_registry_event_chain_rejects_duplicate_ids_and_broken_predecessors(self) -> None:
        first = {
            "event_id": "event-1",
            "sequence": 1,
            "experiment_id": "EXP-001",
            "status": "proposed",
            "previous_event_id": None,
            "previous_status": None,
        }
        second = {
            "event_id": "event-2",
            "sequence": 2,
            "experiment_id": "EXP-001",
            "status": "invalid",
            "previous_event_id": "event-1",
            "previous_status": "proposed",
        }
        update_registry.validate_registry_event_chains([first, second])

        duplicate = dict(second, event_id="event-1")
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            update_registry.validate_registry_event_chains([first, duplicate])

        broken = dict(second, previous_event_id="wrong")
        with self.assertRaisesRegex(ValueError, "previous_event_id chain"):
            update_registry.validate_registry_event_chains([first, broken])

        invalid_transition = dict(second, status="running")
        with self.assertRaisesRegex(ValueError, "invalid historical transition"):
            update_registry.validate_registry_event_chains([first, invalid_transition])

    def test_multiple_appends_remain_complete_line_delimited_events(self) -> None:
        experiment_id = "EXP-APPEND-001"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = make_queue(root, experiment_id)
            registry_path = root / "research" / "REGISTRY.jsonl"
            provider = RegistryMainProvider()
            statuses = ("PROPOSED", "INVALID")
            for status in statuses:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = update_registry.main(
                        [
                            experiment_id,
                            status,
                            "--root",
                            str(root),
                            "--queue-file",
                            str(queue_path),
                            "--registry",
                            str(registry_path),
                            "--actor",
                            "registry-test",
                        ],
                        approval_provider=provider,
                    )
                self.assertEqual(exit_code, 0)
                self.assertTrue(json.loads(stdout.getvalue())["appended"])
                provider.merge_registry(registry_path.read_bytes())

            raw_lines = registry_path.read_text(encoding="utf-8").splitlines(keepends=True)
            self.assertEqual(len(raw_lines), 2)
            self.assertTrue(all(line.endswith("\n") for line in raw_lines))
            events = [strict_json_loads(line) for line in raw_lines]

        self.assertEqual([event["status"] for event in events], ["proposed", "invalid"])
        self.assertEqual([event["sequence"] for event in events], [1, 2])
        self.assertEqual(events[1]["previous_event_id"], events[0]["event_id"])
        for index, event in enumerate(events, start=1):
            self.assert_registry_event_schema(event, location=f"event {index}")

    def test_alternate_registry_ledger_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                update_registry.main(
                    [
                        "EXP-ALT-LEDGER",
                        "PROPOSED",
                        "--root",
                        str(root),
                        "--registry",
                        str(root / "research/alternate.jsonl"),
                    ]
                )
            self.assertIn("code-owned research/REGISTRY.jsonl", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
