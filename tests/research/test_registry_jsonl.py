from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "research" / "update_registry.py"
REGISTRY_PATH = ROOT / "research" / "REGISTRY.jsonl"

SPEC = importlib.util.spec_from_file_location("update_registry", MODULE_PATH)
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
    "human_approved": bool,
    "human_run_approval_recorded": bool,
    "run_approval_in_effect": bool,
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
    "artifacts": list,
    "notes": str,
    "queue_file": str,
}


def strict_json_loads(raw: str) -> object:
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(raw, parse_constant=reject_nonstandard_constant)


def valid_queue(experiment_id: str) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "human_approved_to_run": False,
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
        "score": {"components": dict(update_registry.SCORE_WEIGHTS)},
    }


class RegistryJsonlTests(unittest.TestCase):
    def assert_registry_event_schema(self, event: object, *, location: str) -> None:
        self.assertIsInstance(event, dict, f"{location} must contain a JSON object")
        assert isinstance(event, dict)

        for field, expected_type in REQUIRED_EVENT_TYPES.items():
            self.assertIn(field, event, f"{location} is missing {field!r}")
            self.assertIs(
                type(event[field]),
                expected_type,
                f"{location} field {field!r} must be {expected_type.__name__}",
            )

        for field in ("previous_status", "previous_event_id"):
            self.assertIn(field, event, f"{location} is missing {field!r}")
            self.assertTrue(
                event[field] is None or type(event[field]) is str,
                f"{location} field {field!r} must be a string or null",
            )

        self.assertEqual(event["schema_version"], 1)
        self.assertGreaterEqual(event["sequence"], 1)
        self.assertRegex(event["experiment_id"], update_registry.SAFE_EXPERIMENT_ID)
        self.assertIn(event["status"], update_registry.ALLOWED_STATUSES)
        self.assertTrue(event["event_id"])
        self.assertTrue(event["occurred_at"])
        self.assertTrue(event["actor"])

        components = event["score_components"]
        self.assertEqual(set(components), set(update_registry.SCORE_WEIGHTS))
        for name, maximum in update_registry.SCORE_WEIGHTS.items():
            self.assertIs(type(components[name]), int)
            self.assertGreaterEqual(components[name], 0)
            self.assertLessEqual(components[name], maximum)
        self.assertEqual(event["score_total"], sum(components.values()))
        self.assertEqual(event["score_threshold"], update_registry.RUN_SCORE_THRESHOLD)
        self.assertTrue(all(type(item) is str for item in event["artifacts"]))

    def test_committed_registry_contains_only_valid_typed_json_objects(self) -> None:
        self.assertTrue(REGISTRY_PATH.is_file(), "committed research/REGISTRY.jsonl is missing")

        for line_number, raw in enumerate(
            REGISTRY_PATH.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            location = f"{REGISTRY_PATH}:{line_number}"
            try:
                event = strict_json_loads(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                self.fail(f"{location} is not valid JSON: {exc}")
            self.assert_registry_event_schema(event, location=location)

    def test_load_events_accepts_empty_and_valid_jsonl(self) -> None:
        valid_events = [
            {"experiment_id": "EXP-VALID-001", "status": "proposed", "value": 1},
            {"experiment_id": "EXP-VALID-001", "status": "running", "value": 2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.jsonl"
            path.write_text("\n  \n", encoding="utf-8")
            self.assertEqual(update_registry.load_events(path), [])

            path.write_text(
                "\n".join(json.dumps(event) for event in valid_events) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(update_registry.load_events(path), valid_events)

    def test_load_events_rejects_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.jsonl"
            path.write_text(
                '{"experiment_id":"EXP-BAD-001","status":"proposed"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                update_registry.load_events(path)

    def test_load_events_rejects_non_object_json(self) -> None:
        for payload in ([], "event", 1, None):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "registry.jsonl"
                path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "is not an object"):
                    update_registry.load_events(path)

    def test_load_events_rejects_missing_or_mistyped_required_fields(self) -> None:
        invalid_events = (
            {"status": "proposed"},
            {"experiment_id": "EXP-BAD-001"},
            {"experiment_id": 1, "status": "proposed"},
            {"experiment_id": "EXP-BAD-001", "status": None},
        )
        for event in invalid_events:
            with self.subTest(event=event), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "registry.jsonl"
                path.write_text(json.dumps(event) + "\n", encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "lacks experiment_id/status"):
                    update_registry.load_events(path)

    def test_multiple_appends_remain_complete_line_delimited_events(self) -> None:
        experiment_id = "EXP-APPEND-001"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = root / "research" / "queue" / f"{experiment_id}.json"
            registry_path = root / "research" / "REGISTRY.jsonl"
            queue_path.parent.mkdir(parents=True)
            queue_path.write_text(
                json.dumps(valid_queue(experiment_id)),
                encoding="utf-8",
            )

            for note in ("first event", "second event"):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exit_code = update_registry.main(
                        [
                            experiment_id,
                            "PROPOSED",
                            "--root",
                            str(root),
                            "--registry",
                            str(registry_path),
                            "--queue-file",
                            str(queue_path),
                            "--actor",
                            "registry-test",
                            "--notes",
                            note,
                        ]
                    )
                self.assertEqual(exit_code, 0)
                self.assertTrue(json.loads(output.getvalue())["appended"])

            raw_lines = registry_path.read_text(encoding="utf-8").splitlines(keepends=True)
            self.assertEqual(len(raw_lines), 2)
            self.assertTrue(all(line.endswith("\n") for line in raw_lines))
            line_events = [strict_json_loads(line) for line in raw_lines]
            loaded_events = update_registry.load_events(registry_path)

        self.assertEqual(loaded_events, line_events)
        self.assertEqual([event["sequence"] for event in loaded_events], [1, 2])
        self.assertEqual([event["notes"] for event in loaded_events], ["first event", "second event"])
        self.assertEqual(loaded_events[1]["previous_event_id"], loaded_events[0]["event_id"])
        for index, event in enumerate(loaded_events, start=1):
            self.assert_registry_event_schema(event, location=f"appended event {index}")


if __name__ == "__main__":
    unittest.main()
