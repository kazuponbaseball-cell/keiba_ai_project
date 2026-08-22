from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class RegisteredNonpromotionOfflineRootContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(
            (
                ROOT
                / "research"
                / "REGISTERED_NONPROMOTION_OFFLINE_DIAGNOSTIC_V1.json"
            ).read_text(encoding="utf-8")
        )
        cls.agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        cls.charter = (ROOT / "research" / "CHARTER.md").read_text(
            encoding="utf-8"
        )
        cls.decisions = (ROOT / "research" / "DECISIONS.md").read_text(
            encoding="utf-8"
        )
        cls.state = (ROOT / "research" / "STATE.yaml").read_text(encoding="utf-8")
        cls.scorecard = (
            ROOT / "research" / "HYPOTHESIS_SCORECARD.yaml"
        ).read_text(encoding="utf-8")

    def test_new_route_is_separate_and_non_authoritative(self) -> None:
        self.assertEqual(
            self.policy["gate_kind"],
            "registered_nonpromotion_offline_diagnostic_v1",
        )
        self.assertIs(self.policy["authority"], False)
        self.assertEqual(
            self.policy["source_recipe"]["source_gate_kind"],
            "registered_nonpromotion_diagnostic_v1",
        )
        self.assertIn("ordinary strategy 75-point gate", self.decisions)
        self.assertIn("既存strict-v1 policy/schema/recipe/code", self.decisions)

    def test_score_cannot_be_laundered(self) -> None:
        score = self.policy["source_recipe"]["ordinary_strategy_score"]
        self.assertEqual(score["recorded_total"], 46)
        self.assertEqual(score["threshold"], 75)
        self.assertIs(score["threshold_met"], False)
        self.assertEqual(score["status"], "BLOCKED_SCORE")
        self.assertEqual(score["score_credit"], 0)
        self.assertIs(score["threshold_override_allowed"], False)

    def test_lower_assurance_limitations_are_mandatory(self) -> None:
        expected = {
            "single_use_policy": "ONE_ACCEPTED_EXECUTION",
            "single_use_enforcement": "BEST_EFFORT_LOCAL_EXCLUSIVE_RECEIPT",
            "global_replay_proof": False,
            "rollback_resistant": False,
            "durable_remote_ledger": False,
            "network_isolation": "APPLICATION_LEVEL_NOT_OS_SANDBOX",
        }
        result = self.policy["result_contract"]
        activation = self.policy["activation_contract"]
        for key, value in expected.items():
            self.assertEqual(result[key], value)
            self.assertEqual(activation[key], value)
            self.assertIn(f"{key}:", self.state)
            self.assertIn(key, self.scorecard)

    def test_merge_is_availability_not_run_approval(self) -> None:
        activation = self.policy["activation_contract"]
        self.assertIs(
            activation["containing_commit_must_be_human_merged_to_main"], True
        )
        self.assertIs(
            activation["human_main_merge_activates_gate_availability"], True
        )
        self.assertIs(activation["human_main_merge_is_run_approval"], False)
        self.assertIs(
            activation["branch_ci_ready_review_or_chat_is_gate_availability"],
            False,
        )
        keyword = "APPROVED_OFFLINE_NONPROMOTION_DIAGNOSTIC_RUN"
        self.assertEqual(self.policy["run_approval"]["keyword"], keyword)
        self.assertIn(keyword, self.agents)
        self.assertIn(keyword, self.charter)
        self.assertIn(keyword, self.decisions)

    def test_result_ceiling_and_safety_are_fail_closed(self) -> None:
        result = self.policy["result_contract"]
        self.assertEqual(result["source_authority_class"], "B_LOCAL_HASHED")
        self.assertEqual(
            result["evidence_purpose_class"], "DIAGNOSTIC_NONPROMOTION"
        )
        self.assertIs(result["confirmatory"], False)
        self.assertIs(result["promotion_eligible"], False)
        self.assertEqual(result["score_credit"], 0)
        self.assertIs(result["formal_buy"], False)
        self.assertIs(result["send_order"], False)
        self.assertEqual(result["stake"], 0)
        safety = self.policy["safety"]
        for key in (
            "automatic_execution",
            "automatic_github_approval",
            "odds_price_popularity_or_market_access",
            "training",
            "model_inference",
            "recalibration",
            "threshold_search",
            "workload_network_access",
            "workload_external_api_calls",
            "workload_credential_access",
            "free_form_or_workload_subprocess_allowed",
            "purchase_path_access",
            "production_change",
            "shadow_transition",
            "notification_side_effects",
            "order_side_effects",
            "formal_buy",
            "send_order",
        ):
            self.assertIs(safety[key], False, key)
        self.assertIs(
            safety["fixed_read_only_git_control_plane_subprocess_allowed"], True
        )
        self.assertEqual(safety["stake"], 0)

    def test_cross_route_limitations_are_truthful(self) -> None:
        contract = self.policy["cross_route_contract"]
        self.assertIs(contract["strict_route_independent_and_unmodified"], True)
        self.assertIs(
            contract["cross_route_duplicate_execution_technically_prevented"],
            False,
        )
        self.assertIs(contract["cross_route_single_use_guaranteed"], False)
        self.assertIs(
            contract["offline_result_is_global_question_family_consumption"],
            False,
        )

    def test_raw_provenance_is_preaccess_only(self) -> None:
        projection = self.policy["projection_contract"]
        self.assertIs(
            projection["preaccess_raw_provenance_revalidation_required"], True
        )
        self.assertIs(
            projection[
                "preaccess_raw_provenance_revalidation_may_compute_decisions_or_metrics"
            ],
            False,
        )
        self.assertIs(
            projection[
                "runner_code_opens_raw_sources_after_exclusive_start_receipt"
            ],
            False,
        )
        self.assertIs(
            projection["raw_source_filesystem_capability_isolation_claimed"],
            False,
        )
        self.assertIs(projection["decision_and_run_phases_mount_raw_sources"], False)


if __name__ == "__main__":
    unittest.main()
