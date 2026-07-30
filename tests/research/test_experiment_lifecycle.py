from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]


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

    def _invoke(
        self,
        main: Callable[[list[str]], int],
        argv: list[str],
    ) -> dict[str, object]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(argv)
        self.assertEqual(exit_code, 0, stderr.getvalue())
        return json.loads(stdout.getvalue())

    def _assert_cli_error(
        self,
        main: Callable[[list[str]], int],
        argv: list[str],
        message: str,
    ) -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main(argv)
        self.assertEqual(caught.exception.code, 2)
        error_text = stderr.getvalue()
        self.assertIn(message, error_text)
        return error_text

    def _create(
        self,
        experiment_id: str,
        scores: dict[str, int] | None = None,
    ) -> dict[str, object]:
        selected_scores = SCORE_75 if scores is None else scores
        argv = [
            experiment_id,
            "--title",
            "Lifecycle contract test",
            "--owner",
            "test-researcher",
            "--hypothesis",
            "The lifecycle remains fail-closed.",
            "--root",
            str(self.root),
        ]
        for name in create_experiment.SCORE_WEIGHTS:
            argv.extend(["--" + name.replace("_", "-"), str(selected_scores[name])])
        return self._invoke(create_experiment.main, argv)

    def _create_argv(
        self,
        experiment_id: str,
        scores: dict[str, int] | None = None,
    ) -> list[str]:
        selected_scores = SCORE_75 if scores is None else scores
        argv = [
            experiment_id,
            "--title",
            "Lifecycle contract test",
            "--owner",
            "test-researcher",
            "--root",
            str(self.root),
        ]
        for name in create_experiment.SCORE_WEIGHTS:
            argv.extend(["--" + name.replace("_", "-"), str(selected_scores[name])])
        return argv

    def _append(
        self,
        experiment_id: str,
        status: str,
        *,
        human_approved: bool = False,
    ) -> dict[str, object]:
        argv = [
            experiment_id,
            status,
            "--actor",
            "test-researcher",
            "--root",
            str(self.root),
        ]
        if human_approved:
            argv.append("--human-approved")
        return self._invoke(update_registry.main, argv)

    def _append_argv(
        self,
        experiment_id: str,
        status: str,
        *,
        human_approved: bool = False,
    ) -> list[str]:
        argv = [
            experiment_id,
            status,
            "--actor",
            "test-researcher",
            "--root",
            str(self.root),
        ]
        if human_approved:
            argv.append("--human-approved")
        return argv

    def _queue(self, experiment_id: str) -> dict[str, object]:
        path = self.root / "research" / "queue" / f"{experiment_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _events(self) -> list[dict[str, object]]:
        registry = self.root / "research" / "REGISTRY.jsonl"
        if not registry.exists():
            return []
        return [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]

    def test_score_74_is_blocked_score(self) -> None:
        result = self._create("exp-score-74", SCORE_74)
        queue = self._queue("exp-score-74")

        self.assertEqual(result["score_total"], 74)
        self.assertEqual(result["status"], "blocked_score")
        self.assertEqual(queue["status"], "blocked_score")
        event = self._append("exp-score-74", "BLOCKED_SCORE")["event"]
        self.assertEqual(event["status"], "blocked_score")

    def test_score_75_is_proposed(self) -> None:
        result = self._create("exp-score-75")
        queue = self._queue("exp-score-75")

        self.assertEqual(result["score_total"], 75)
        self.assertEqual(result["status"], "proposed")
        self.assertFalse(result["automatic_execution_allowed"])
        self.assertFalse(queue["human_approved_to_run"])
        self.assertFalse(queue["automatic_execution_allowed"])
        self.assertFalse(queue["execution_authorized"])

    def test_proposed_experiment_cannot_execute_before_approval(self) -> None:
        experiment_id = "exp-preapproval-gate"
        self._create(experiment_id)
        proposed_event = self._append(experiment_id, "PROPOSED")["event"]
        self.assertFalse(proposed_event["automatic_execution_allowed"])
        self.assertFalse(proposed_event["execution_authorized"])

        self._assert_cli_error(
            update_registry.main,
            self._append_argv(experiment_id, "RUNNING"),
            "invalid transition",
        )

    def test_approved_to_run_requires_explicit_human_approval(self) -> None:
        experiment_id = "exp-human-approval"
        self._create(experiment_id)
        self._append(experiment_id, "PROPOSED")

        self._assert_cli_error(
            update_registry.main,
            self._append_argv(experiment_id, "APPROVED_TO_RUN"),
            "requires an explicit --human-approved flag",
        )
        self.assertEqual([event["status"] for event in self._events()], ["proposed"])

    def test_running_requires_approved_to_run_as_immediately_preceding_event(self) -> None:
        experiment_id = "exp-running-gate"
        self._create(experiment_id)

        self._assert_cli_error(
            update_registry.main,
            self._append_argv(experiment_id, "RUNNING"),
            "invalid transition",
        )
        self.assertEqual(self._events(), [])

    def test_score_change_after_approval_requires_new_experiment_id(self) -> None:
        experiment_id = "exp-score-freeze"
        self._create(experiment_id)
        self._append(experiment_id, "PROPOSED")
        self._append(experiment_id, "APPROVED_TO_RUN", human_approved=True)

        queue_path = self.root / "research" / "queue" / f"{experiment_id}.json"
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        queue["score"]["components"]["acquisition_implementation_cost"] = 1
        queue_path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")

        self._assert_cli_error(
            update_registry.main,
            self._append_argv(experiment_id, "RUNNING"),
            "queue score changed after approval",
        )
        self.assertEqual(
            [event["status"] for event in self._events()],
            ["proposed", "approved_to_run"],
        )

    def test_invalid_transition_is_rejected_without_registry_append(self) -> None:
        experiment_id = "exp-invalid-transition"
        self._create(experiment_id)
        self._append(experiment_id, "PROPOSED")

        self._assert_cli_error(
            update_registry.main,
            self._append_argv(experiment_id, "REVIEW_REQUIRED"),
            "invalid transition",
        )
        self.assertEqual([event["status"] for event in self._events()], ["proposed"])

    def test_create_refuses_to_overwrite_same_experiment_id(self) -> None:
        experiment_id = "exp-no-overwrite"
        self._create(experiment_id)
        experiment_path = self.root / "research" / "experiments" / f"{experiment_id}.md"
        queue_path = self.root / "research" / "queue" / f"{experiment_id}.json"
        original_experiment = experiment_path.read_bytes()
        original_queue = queue_path.read_bytes()

        self._assert_cli_error(
            create_experiment.main,
            self._create_argv(experiment_id),
            "refusing to overwrite existing file",
        )
        self.assertEqual(experiment_path.read_bytes(), original_experiment)
        self.assertEqual(queue_path.read_bytes(), original_queue)

    def test_buy_order_and_stake_remain_disabled_through_running(self) -> None:
        experiment_id = "exp-buy-safety"
        self._create(experiment_id)
        self._append(experiment_id, "PROPOSED")
        self._append(experiment_id, "APPROVED_TO_RUN", human_approved=True)
        self._append(experiment_id, "RUNNING")

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
                    self.assertIs(record[field], False, field)

    def test_registry_cannot_grant_production_merge_or_buy_approval(self) -> None:
        experiment_id = "exp-authority-boundary"
        self._create(experiment_id)

        for prohibited_status in (
            "APPROVED_FOR_PRODUCTION",
            "APPROVED_FOR_MERGE",
            "APPROVED_FOR_BUY",
        ):
            with self.subTest(status=prohibited_status):
                self._assert_cli_error(
                    update_registry.main,
                    self._append_argv(experiment_id, prohibited_status),
                    "production/merge/BUY approval is outside this registry's authority",
                )
        self.assertEqual(self._events(), [])

    def test_invalid_is_terminal_and_cannot_recover(self) -> None:
        experiment_id = "exp-invalid-terminal"
        self._create(experiment_id)
        self._append(experiment_id, "PROPOSED")
        self._append(experiment_id, "INVALID")

        recovery_attempts = (
            ("PROPOSED", False),
            ("APPROVED_TO_RUN", True),
            ("RUNNING", False),
            ("REVIEW_REQUIRED", False),
            ("ARCHIVED", False),
        )
        for status, human_approved in recovery_attempts:
            with self.subTest(status=status):
                self._assert_cli_error(
                    update_registry.main,
                    self._append_argv(
                        experiment_id,
                        status,
                        human_approved=human_approved,
                    ),
                    "invalid transition",
                )
        self.assertEqual(
            [event["status"] for event in self._events()],
            ["proposed", "invalid"],
        )


if __name__ == "__main__":
    unittest.main()
