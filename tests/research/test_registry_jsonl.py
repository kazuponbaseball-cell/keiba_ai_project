from __future__ import annotations

import contextlib
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


def strict_json_loads(raw: str) -> object:
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(raw, parse_constant=reject_nonstandard_constant)


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
        for field, expected_type in REQUIRED_EVENT_TYPES.items():
            self.assertIn(field, event, f"{location} missing {field}")
            self.assertIs(type(event[field]), expected_type, f"{location}.{field}")
        for field in (
            "previous_status",
            "previous_event_id",
            "run_scope_digest",
            "review_digest",
            "approval_evidence",
            "run_scope_file",
        ):
            self.assertIn(field, event)
        self.assertEqual(event["schema_version"], 2)
        self.assertIn(event["status"], update_registry.ALLOWED_STATUSES)
        self.assertRegex(event["proposal_scope_digest"], r"^[0-9a-f]{64}$")
        self.assertEqual(event["score_total"], sum(event["score_components"].values()))
        self.assertTrue(all(isinstance(item, str) for item in event["artifacts"]))

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

    def test_multiple_appends_remain_complete_line_delimited_events(self) -> None:
        experiment_id = "EXP-APPEND-001"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = make_queue(root, experiment_id)
            registry_path = root / "research" / "REGISTRY.jsonl"
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
                        ]
                    )
                self.assertEqual(exit_code, 0)
                self.assertTrue(json.loads(stdout.getvalue())["appended"])

            raw_lines = registry_path.read_text(encoding="utf-8").splitlines(keepends=True)
            self.assertEqual(len(raw_lines), 2)
            self.assertTrue(all(line.endswith("\n") for line in raw_lines))
            events = [strict_json_loads(line) for line in raw_lines]

        self.assertEqual([event["status"] for event in events], ["proposed", "invalid"])
        self.assertEqual([event["sequence"] for event in events], [1, 2])
        self.assertEqual(events[1]["previous_event_id"], events[0]["event_id"])
        for index, event in enumerate(events, start=1):
            self.assert_registry_event_schema(event, location=f"event {index}")


if __name__ == "__main__":
    unittest.main()
