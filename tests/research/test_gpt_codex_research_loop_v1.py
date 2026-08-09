from __future__ import annotations

import copy
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

import gpt_codex_research_loop_v1 as loop
import gpt_strategy_contract_v1 as contract


EXPERIMENT_ID = "EXP-SYNTHETIC-001"
PROPOSAL_FORBIDDEN_COLUMNS = sorted(contract.REQUIRED_FORBIDDEN_COLUMNS)


def sample_proposal() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "title": "Test one independent racing signal",
        "hypothesis": "The pre-registered signal improves chronological OOS log loss.",
        "null_hypothesis": "The signal does not improve chronological OOS log loss.",
        "racing_mechanism": "The signal measures a pre-race state absent from the baseline.",
        "target_population": "Synthetic contract rows only during preparation.",
        "in_scope": ["Implement one bounded research challenger"],
        "out_of_scope": ["Production and purchase behavior"],
        "expected_changed_paths": ["scripts/research/example_challenger.py"],
        "raw_data_sources": ["Synthetic fixtures"],
        "data_as_of": "2026-08-08T00:00:00+09:00",
        "allowed_columns": ["feature_value", "horse_id", "race_id", "source_event_time"],
        "forbidden_columns": PROPOSAL_FORBIDDEN_COLUMNS,
        "lineage_hash_requirements": ["Input manifest SHA-256"],
        "chronological_fold_design": {
            "design": "synthetic_contract_cases_only",
            "folds": 1,
        },
        "fold_manifest": {
            "path": "research/drafts/synthetic.fold_manifest.json",
            "sha256": "1" * 64,
        },
        "purge_embargo": {"purge": "not_applicable", "embargo": "not_applicable"},
        "primary_metric": {"name": "log_loss", "direction": "lower_is_better"},
        "required_effect": {"maximum_delta": -0.001},
        "rejection_gate": ["Reject if chronological OOS does not improve"],
        "stop_conditions": ["Stop on any contract violation"],
        "compute_budget": {"synthetic_rows": 14, "external_api_calls": 0},
        "allowed_variant_count": 1,
        "allowed_threshold_search_count": 0,
        "base_commit": "2" * 40,
        "score_components": {
            "independent_information": 21,
            "racing_mechanism": 14,
            "outer_oos_failure_evidence": 19,
            "leakage_safety": 15,
            "minimal_falsifiability": 10,
            "acquisition_implementation_cost": 9,
        },
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def sample_strategy() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "strategy_id": "STRATEGY-SYNTHETIC-001",
        "brain_model_id": "gpt-test-model",
        "brain_prompt_hash": "3" * 64,
        "context_manifest_hash": "4" * 64,
        "proposal_scope": sample_proposal(),
        "safety": {
            "actual_codex_dispatch": False,
            "automatic_github_approval": False,
            "candidate_policy_change": False,
            "credential_access": False,
            "external_api_calls": False,
            "formal_buy": False,
            "merge": False,
            "notification_side_effects": False,
            "production_change": False,
            "purchase_path_access": False,
            "real_data_execution": False,
            "send_order": False,
            "stake": 0,
        },
    }


def approval_evidence(proposal_digest: str) -> dict[str, Any]:
    return {
        "approval_type": "APPROVED_TO_PREPARE",
        "approval_digest": proposal_digest,
        "repository": "kazuponbaseball-cell/keiba_ai_project",
        "issue_number": 33,
        "comment_id": 1001,
        "url": "https://github.example.invalid/comment/1001",
        "author": "human-approver",
        "author_type": "User",
        "body": f"APPROVED_TO_PREPARE {proposal_digest}",
        "body_sha256": "5" * 64,
        "created_at": "2026-08-08T00:00:00Z",
        "updated_at": "2026-08-08T00:00:00Z",
    }


def registry_events(strategy: dict[str, Any], *, approved: bool) -> list[dict[str, Any]]:
    normalized = contract.normalize_strategy(strategy)
    proposal_digest = contract.canonical_digest(normalized["proposal_scope"])
    proposed = {
        "sequence": 1,
        "event_id": "event-1",
        "previous_event_id": None,
        "experiment_id": EXPERIMENT_ID,
        "status": "proposed",
        "proposal_scope_digest": proposal_digest,
        "approval_evidence": None,
        "revalidated_approval_evidence": [],
        "preparation_authorized": False,
        "synthetic_fixture_tests_allowed": False,
        "real_data_execution_allowed": False,
        "automatic_execution_allowed": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    if not approved:
        return [proposed]
    evidence = approval_evidence(proposal_digest)
    approved_event = {
        **proposed,
        "sequence": 2,
        "event_id": "event-2",
        "previous_event_id": "event-1",
        "status": "approved_to_prepare",
        "approval_evidence": evidence,
        "preparation_authorized": True,
        "synthetic_fixture_tests_allowed": True,
    }
    preparing = {
        **approved_event,
        "sequence": 3,
        "event_id": "event-3",
        "previous_event_id": "event-2",
        "status": "preparing",
        "approval_evidence": None,
        "revalidated_approval_evidence": [evidence],
    }
    return [proposed, approved_event, preparing]


def write_registry(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )


class GptCodexResearchLoopV1Tests(unittest.TestCase):
    def test_01_valid_single_hypothesis_compiles_deterministically(self) -> None:
        strategy = sample_strategy()
        first = contract.compile_proposal_scope(strategy)
        second = contract.compile_proposal_scope(copy.deepcopy(strategy))
        self.assertEqual(first, second)
        self.assertEqual(first["allowed_variant_count"], 1)

    def test_02_missing_null_hypothesis_is_rejected(self) -> None:
        strategy = sample_strategy()
        del strategy["proposal_scope"]["null_hypothesis"]
        with self.assertRaisesRegex(ValueError, "null_hypothesis"):
            contract.normalize_strategy(strategy)

    def test_03_missing_racing_mechanism_is_rejected(self) -> None:
        strategy = sample_strategy()
        del strategy["proposal_scope"]["racing_mechanism"]
        with self.assertRaisesRegex(ValueError, "racing_mechanism"):
            contract.normalize_strategy(strategy)

    def test_04_multiple_unbounded_variants_are_rejected(self) -> None:
        strategy = sample_strategy()
        strategy["proposal_scope"]["allowed_variant_count"] = 2
        with self.assertRaisesRegex(ValueError, "allowed_variant_count"):
            contract.normalize_strategy(strategy)

    def test_05_threshold_search_request_is_rejected(self) -> None:
        strategy = sample_strategy()
        strategy["proposal_scope"]["allowed_threshold_search_count"] = 1
        with self.assertRaisesRegex(ValueError, "allowed_threshold_search_count"):
            contract.normalize_strategy(strategy)

    def test_06_market_or_result_driven_candidate_request_is_rejected(self) -> None:
        for field in ("current_odds", "official_result", "payout", "popularity"):
            with self.subTest(field=field):
                strategy = sample_strategy()
                strategy["proposal_scope"]["allowed_columns"].append(field)
                with self.assertRaisesRegex(ValueError, "allowed_columns"):
                    contract.normalize_strategy(strategy)

    def test_07_production_or_purchase_path_request_is_rejected(self) -> None:
        for field in ("production_change", "purchase_path_access", "send_order"):
            with self.subTest(field=field):
                strategy = sample_strategy()
                strategy["safety"][field] = True
                with self.assertRaisesRegex(ValueError, field):
                    contract.normalize_strategy(strategy)

    def test_08_credential_or_secret_request_is_rejected(self) -> None:
        strategy = sample_strategy()
        strategy["api_key"] = "not-allowed"
        with self.assertRaisesRegex(ValueError, "api_key"):
            contract.normalize_strategy(strategy)

    def test_09_free_form_shell_command_is_rejected(self) -> None:
        strategy = sample_strategy()
        strategy["proposal_scope"]["in_scope"] = [
            "powershell -Command python unsafe.py"
        ]
        with self.assertRaisesRegex(ValueError, "free-form shell command"):
            contract.normalize_strategy(strategy)

    def test_10_unapproved_prepare_dispatch_is_rejected(self) -> None:
        strategy = sample_strategy()
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = Path(temp_dir) / "registry.jsonl"
            write_registry(registry, registry_events(strategy, approved=False))
            with self.assertRaisesRegex(ValueError, "approval is not active"):
                loop.build_preparation_dispatch_packet(strategy, registry_path=registry)

    def test_11_prepare_approval_emits_synthetic_only_dispatch_packet(self) -> None:
        strategy = sample_strategy()
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = Path(temp_dir) / "registry.jsonl"
            write_registry(registry, registry_events(strategy, approved=True))
            packet = loop.build_preparation_dispatch_packet(strategy, registry_path=registry)
        self.assertEqual(packet["dispatch_mode"], "synthetic_preparation_packet_only")
        self.assertEqual(packet["exact_execution_commands"], [])
        self.assertFalse(packet["actual_codex_dispatch"])
        self.assertFalse(packet["formal_buy"])
        self.assertEqual(packet["stake"], 0)

    def test_12_run_approval_is_required_for_external_api_or_real_data(self) -> None:
        strategy = sample_strategy()
        with self.subTest(case="external_api"):
            unsafe = copy.deepcopy(strategy)
            unsafe["safety"]["external_api_calls"] = True
            with self.assertRaisesRegex(ValueError, "external_api_calls"):
                contract.normalize_strategy(unsafe)
        with self.subTest(case="real_data"):
            events = registry_events(strategy, approved=True)
            events[-1]["real_data_execution_allowed"] = True
            with tempfile.TemporaryDirectory() as temp_dir:
                registry = Path(temp_dir) / "registry.jsonl"
                write_registry(registry, events)
                with self.assertRaisesRegex(ValueError, "real_data_execution_allowed"):
                    loop.build_preparation_dispatch_packet(strategy, registry_path=registry)

    def test_13_result_feedback_preserves_prompt_model_and_artifact_hashes(self) -> None:
        strategy = sample_strategy()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / "registry.jsonl"
            prompt = root / "review_prompt.txt"
            write_registry(registry, registry_events(strategy, approved=True))
            prompt.write_text("Review the synthetic contract result.\n", encoding="utf-8")
            dispatch = loop.build_preparation_dispatch_packet(strategy, registry_path=registry)
            manifest = {"artifacts": [{"path": "synthetic.json", "sha256": "6" * 64}]}
            summary = {"passed": 14, "failed": 0}
            feedback = loop.build_result_feedback_packet(
                dispatch,
                result_manifest=manifest,
                result_summary=summary,
                review_prompt_path=prompt,
            )
        self.assertEqual(feedback["brain_model_id"], strategy["brain_model_id"])
        self.assertEqual(feedback["brain_prompt_hash"], strategy["brain_prompt_hash"])
        self.assertEqual(feedback["context_manifest_hash"], strategy["context_manifest_hash"])
        self.assertEqual(feedback["result_manifest_hash"], contract.canonical_digest(manifest))
        self.assertEqual(feedback["result_summary_hash"], contract.canonical_digest(summary))

    def test_14_identical_input_produces_identical_canonical_digest(self) -> None:
        strategy = sample_strategy()
        reversed_top_level = dict(reversed(list(strategy.items())))
        first = contract.canonical_digest(contract.normalize_strategy(strategy))
        second = contract.canonical_digest(contract.normalize_strategy(reversed_top_level))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
