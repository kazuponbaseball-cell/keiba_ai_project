from __future__ import annotations

import ast
import copy
import importlib.util
import json
import re
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "build_decision_summary_layer_v0.py"
FIXTURE = ROOT / "tests" / "research" / "fixtures" / "decision_summary_layer_v0.synthetic.json"
SCHEMA = ROOT / "docs" / "schemas" / "decision_summary_layer_v0.schema.json"
SAMPLE = ROOT / "docs" / "observations" / "decision_summary_layer_v0" / "synthetic_sample"
SAMPLE_JSON = SAMPLE / "decision_summary_v0.json"
SAMPLE_HTML = SAMPLE / "decision_summary_v0.html"


def load_module():
    spec = importlib.util.spec_from_file_location("decision_summary_layer_v0", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load_module()


def clone(value):
    return copy.deepcopy(value)


def recursive_keys(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key),)
            yield child_path
            yield from recursive_keys(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from recursive_keys(child, path + (str(index),))


def recursive_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from recursive_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_values(child)
    else:
        yield value


class SummaryDomProbe(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth_in_details = 0
        self.summary_horse_ids = []
        self.detail_ids = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "details":
            self.depth_in_details += 1
            self.detail_ids.append(attributes.get("id"))
        if tag == "tr" and attributes.get("data-horse-id"):
            if self.depth_in_details:
                raise AssertionError("all-runner summary row is hidden inside details")
            self.summary_horse_ids.append(attributes["data-horse-id"])

    def handle_endtag(self, tag):
        if tag == "details":
            self.depth_in_details -= 1


class DecisionSummaryLayerV0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.output = json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))
        cls.html = SAMPLE_HTML.read_text(encoding="utf-8")

    def test_module_is_standalone_stdlib_consumer(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        allowed = {"__future__", "argparse", "hashlib", "html", "json", "re", "pathlib", "typing"}
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        self.assertEqual(set(), roots - allowed)
        source_text = SCRIPT.read_text(encoding="utf-8")
        for forbidden_import in (
            "build_race_intelligence_lite_plus",
            "sklearn",
            "requests",
            "subprocess",
            "urllib",
        ):
            self.assertNotIn(f"import {forbidden_import}", source_text)

    def test_sample_has_all_four_exclusive_review_priorities(self):
        MOD.validate_decision_summary(self.output)
        horses = self.output["races"][0]["horse_summaries"]
        self.assertEqual(
            ["PRIMARY_REVIEW", "CONDITIONAL_REVIEW", "FRAGILE", "INSUFFICIENT"],
            [horse["classification"] for horse in horses],
        )
        decision = self.output["races"][0]["decision_summary"]
        self.assertEqual([1], [item["horse_no"] for item in decision["primary_review"]])
        self.assertEqual([2], [item["horse_no"] for item in decision["conditional_review"]])
        self.assertEqual([3], [item["horse_no"] for item in decision["fragile_or_downgrade"]])
        self.assertEqual([4], [item["horse_no"] for item in decision["insufficient_review"]])

    def test_all_runner_identities_are_preserved_in_official_number_order(self):
        projected = MOD.project_horsecard_source(self.source)
        output = MOD.build_decision_summary(self.source)
        source_ids = [
            (race["race_id"], runner["horse_id"], runner["basics"]["horse_no"])
            for race in projected["races"]
            for runner in race["runners"]
        ]
        output_ids = [
            (race["race_id"], horse["horse_id"], horse["basics"]["horse_no"])
            for race in output["races"]
            for horse in race["horse_summaries"]
        ]
        detail_ids = [
            (race["race_id"], detail["horse_id"], detail["horse_no"])
            for race in output["races"]
            for detail in race["evidence_details"]
        ]
        self.assertEqual(source_ids, output_ids)
        self.assertEqual(source_ids, detail_ids)
        self.assertEqual(len(source_ids), output["runner_count"])

    def test_duplicate_and_spoofed_runner_identities_fail_closed(self):
        duplicate_id = clone(self.source)
        duplicate_id["races"][0]["runners"][1]["horse_id"] = "SYN-H01"
        with self.assertRaises(MOD.DecisionSummaryError):
            MOD.build_decision_summary(duplicate_id)
        duplicate_no = clone(self.source)
        duplicate_no["races"][0]["runners"][1]["basics"]["horse_no"] = 1
        with self.assertRaises(MOD.DecisionSummaryError):
            MOD.build_decision_summary(duplicate_no)
        boolean_no = clone(self.source)
        boolean_no["races"][0]["runners"][0]["basics"]["horse_no"] = True
        with self.assertRaises(MOD.DecisionSummaryError):
            MOD.build_decision_summary(boolean_no)

    def test_unknown_and_contaminated_fields_fail_closed_at_every_depth(self):
        forbidden_keys = (
            "ai_score",
            "aiScore",
            "ai-score",
            "rank",
            "gap",
            "confidence",
            "scenario_sensitivity",
            "SLOW",
            "MIDDLE",
            "FAST",
            "odds",
            "market",
            "popularity",
            "payoff",
            "EV",
            "BUY",
            "order",
            "notification",
            "Champion",
        )
        for key in forbidden_keys:
            with self.subTest(key=key):
                poisoned = clone(self.source)
                poisoned["races"][0]["runners"][0][key] = "poison"
                with self.assertRaises(MOD.DecisionSummaryError):
                    MOD.build_decision_summary(poisoned)
        nested = clone(self.source)
        nested["races"][0]["runners"][0]["role_queue_evidence"]["confidence"] = "HIGH"
        with self.assertRaises(MOD.DecisionSummaryError):
            MOD.build_decision_summary(nested)

    def test_forbidden_legacy_scenario_and_market_text_cannot_hide_in_evidence(self):
        poisoned_values = (
            "legacy ai_score was high",
            "rank gap",
            "model confidence",
            "win_probability",
            "scenario_sensitivity",
            "S/M/F",
            "rank_v2 BUY_FLAG SLOW_MODE marketValue",
            "slow middle fast",
            "odds market EV BUY stake",
            "オッズ 人気 払戻 購入",
            "旧漏洩AI順位 市場価格 期待値",
        )
        for field in ("label", "value", "source_note"):
            for poisoned_value in poisoned_values:
                with self.subTest(field=field, poisoned_value=poisoned_value):
                    source = clone(self.source)
                    source["races"][0]["runners"][0]["evidence_details"][0][field] = poisoned_value
                    with self.assertRaises(MOD.DecisionSummaryError):
                        MOD.build_decision_summary(source)

    def test_unobserved_detail_cannot_back_structured_classification_evidence(self):
        source = clone(self.source)
        next(
            item
            for item in source["races"][0]["runners"][0]["evidence_details"]
            if item["evidence_id"] == "A-ROLE-01"
        )["status"] = "UNOBSERVED"
        with self.assertRaises(MOD.DecisionSummaryError):
            MOD.build_decision_summary(source)

    def test_role_and_queue_require_separate_typed_evidence_dimensions(self):
        source = clone(self.source)
        role_queue = source["races"][0]["runners"][0]["role_queue_evidence"]
        role_queue["pressure_rivals"] = []
        role_queue["role_evidence_ids"] = ["A-ROLE-01"]
        role_queue["queue_evidence_ids"] = ["A-ROLE-01"]
        with self.assertRaises(MOD.DecisionSummaryError):
            MOD.build_decision_summary(source)

    def test_actual_lite_plus_or_real_mode_cannot_enter_synthetic_cli_contract(self):
        lite_plus_shape = {
            "schema_version": "race_intelligence_lite_plus_sunday_observation_queue_v0_2",
            "races": [],
        }
        with self.assertRaises(MOD.DecisionSummaryError):
            MOD.build_decision_summary(lite_plus_shape)
        real_mode = clone(self.source)
        real_mode["synthetic_fixture"] = False
        with self.assertRaises(MOD.DecisionSummaryError):
            MOD.build_decision_summary(real_mode)
        materialized = clone(self.source)
        materialized["safety"]["real_data_materialized"] = True
        with self.assertRaises(MOD.DecisionSummaryError):
            MOD.build_decision_summary(materialized)

    def test_json_loader_rejects_duplicate_keys_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            duplicate = Path(temporary) / "duplicate.json"
            duplicate.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
            with self.assertRaises(MOD.DecisionSummaryError):
                MOD.load_json(duplicate)
            nonfinite = Path(temporary) / "nonfinite.json"
            nonfinite.write_text('{"value":NaN}', encoding="utf-8")
            with self.assertRaises(MOD.DecisionSummaryError):
                MOD.load_json(nonfinite)

    def test_clean_ability_is_never_imputed_from_legacy_proxy(self):
        for horse in self.output["races"][0]["horse_summaries"]:
            self.assertEqual("UNKNOWN", horse["ability_status"]["value"])
            self.assertEqual("NOT_AVAILABLE", horse["ability_status"]["availability"])
            self.assertEqual(0, horse["ability_status"]["evidence_count"])
        poisoned = clone(self.source)
        poisoned["races"][0]["runners"][3]["ability"] = {
            "ability_band": "HIGH",
            "recent_standard": "HIGH",
        }
        with self.assertRaises(MOD.DecisionSummaryError):
            MOD.build_decision_summary(poisoned)

    def test_proxy_condition_does_not_fill_current_condition(self):
        source = clone(self.source)
        condition = source["races"][0]["runners"][0]["current_condition_evidence"]
        condition["status"] = "PROXY"
        next(
            item
            for item in source["races"][0]["runners"][0]["evidence_details"]
            if item["evidence_id"] == "A-COND-01"
        )["status"] = "PROXY"
        output = MOD.build_decision_summary(source)
        horse = output["races"][0]["horse_summaries"][0]
        self.assertEqual("UNKNOWN", horse["current_condition_fit"]["value"])
        self.assertEqual("NOT_AVAILABLE", horse["current_condition_fit"]["availability"])
        self.assertIn("CONDITION_CURRENT_NOT_AVAILABLE", horse["reason_codes"])
        self.assertNotIn("CURRENT_CONDITION_SUPPORTIVE", horse["reason_codes"])

    def test_classification_contracts_and_reason_codes(self):
        for race in self.output["races"]:
            for horse in race["horse_summaries"]:
                classification = horse["classification"]
                codes = horse["reason_codes"]
                self.assertEqual(len(codes), len(set(codes)))
                self.assertTrue(set(codes) <= set(MOD.REASON_CODE_CATALOG))
                if classification != "INSUFFICIENT":
                    self.assertTrue(codes)
                if classification == "PRIMARY_REVIEW":
                    self.assertTrue(
                        {"PRIMARY_ROLE_REACHABLE", "PRIMARY_QUEUE_SUPPORTIVE"} <= set(codes)
                    )
                    self.assertGreaterEqual(
                        len({"PRIMARY_ROLE_REACHABLE", "PRIMARY_QUEUE_SUPPORTIVE"} & set(codes)),
                        2,
                    )
                    self.assertEqual([], horse["fragility_triggers"])
                if classification == "CONDITIONAL_REVIEW":
                    self.assertTrue(horse["upside_triggers"])
                if classification == "FRAGILE":
                    self.assertTrue(horse["fragility_triggers"])
                for trigger in horse["upside_triggers"] + horse["fragility_triggers"]:
                    self.assertGreaterEqual(trigger["evidence_count"], 1)

    def test_major_fragility_has_fixed_precedence_over_primary_materials(self):
        source = clone(self.source)
        alpha = source["races"][0]["runners"][0]
        alpha["role_queue_evidence"]["pressure_rivals"][0]["conflict_level"] = "HIGH"
        output = MOD.build_decision_summary(source)
        horse = output["races"][0]["horse_summaries"][0]
        self.assertEqual("FRAGILE", horse["classification"])
        self.assertIn("PRIMARY_ROLE_REACHABLE", horse["reason_codes"])
        self.assertIn("FRAGILE_QUEUE_HIGH_CONFLICT", horse["reason_codes"])
        self.assertTrue(horse["fragility_triggers"])

    def test_queue_only_and_condition_only_fragility_keep_upside_unknown(self):
        queue_only = clone(self.source)
        delta = queue_only["races"][0]["runners"][3]
        delta["role_queue_evidence"]["first_turn_position_cost"] = "HIGH"
        delta["role_queue_evidence"]["lead_dependency"] = "NO"
        delta["role_queue_evidence"]["queue_evidence_ids"] = ["D-MISSING-01"]
        delta["evidence_details"][0].update(
            {"section": "QUEUE", "status": "DERIVED", "label": "first-turn cost", "value": "high"}
        )
        queue_horse = MOD.build_decision_summary(queue_only)["races"][0]["horse_summaries"][3]
        self.assertEqual("FRAGILE", queue_horse["classification"])
        self.assertEqual(1, queue_horse["fragility_triggers"][0]["evidence_count"])
        self.assertTrue(queue_horse["winning_or_in_the_money_world_state"].startswith("NOT_AVAILABLE"))

        condition_only = clone(self.source)
        delta = condition_only["races"][0]["runners"][3]
        delta["current_condition_evidence"] = {
            "fit": "ADVERSE",
            "status": "OBSERVED",
            "evidence_ids": ["D-MISSING-01"],
        }
        delta["evidence_details"][0].update(
            {"section": "CONDITION", "status": "OBSERVED", "label": "current condition", "value": "adverse"}
        )
        condition_horse = MOD.build_decision_summary(condition_only)["races"][0]["horse_summaries"][3]
        self.assertEqual("FRAGILE", condition_horse["classification"])
        self.assertTrue(condition_horse["winning_or_in_the_money_world_state"].startswith("NOT_AVAILABLE"))
        self.assertEqual("FAILURE_CURRENT_CONDITION_FAILS", condition_horse["fragility_triggers"][0]["code"])

    def test_evidence_counts_are_derived_not_self_reported_or_used_as_score(self):
        alpha = self.output["races"][0]["horse_summaries"][0]
        self.assertEqual(4, alpha["evidence_count"]["total_distinct"])
        self.assertEqual(4, alpha["evidence_count"]["detail_items"])
        self.assertIs(False, alpha["evidence_count"]["classification_uses_count"])
        dimension_ids = alpha["evidence_count"]["evidence_ids_by_dimension"]
        self.assertEqual(alpha["evidence_count"]["role"], len(dimension_ids["role"]))
        self.assertEqual(alpha["evidence_count"]["queue"], len(dimension_ids["queue"]))
        self.assertEqual(
            alpha["evidence_count"]["current_condition"],
            len(dimension_ids["current_condition"]),
        )
        for key in ("current_condition", "detail_items", "queue", "role", "total_distinct"):
            self.assertIsInstance(alpha["evidence_count"][key], int)
            self.assertNotIsInstance(alpha["evidence_count"][key], bool)

    def test_no_forbidden_decision_fields_or_smf_values_are_emitted(self):
        forbidden_normalized = {
            "aiscore",
            "rank",
            "ranking",
            "gap",
            "probability",
            "winprobability",
            "placeprobability",
            "odds",
            "market",
            "popularity",
            "payoff",
            "ev",
            "buy",
            "champion",
            "notification",
        }
        allowed_safety_paths = {
            ("safety", "formal_buy"),
            ("safety", "send_order"),
            ("safety", "stake"),
        }
        for path in recursive_keys(self.output):
            if path in allowed_safety_paths:
                continue
            normalized = re.sub(r"[^a-z0-9]", "", path[-1].lower())
            self.assertNotIn(normalized, forbidden_normalized, path)
            self.assertFalse(normalized.endswith("score"), path)
        for value in recursive_values(self.output):
            if isinstance(value, str):
                self.assertNotIn(value, {"SLOW", "MIDDLE", "FAST"})
        self.assertIs(False, self.output["weighted_total_generated"])

    def test_mapping_and_runner_order_do_not_change_canonical_outputs(self):
        reordered = clone(self.source)
        reordered["races"][0]["runners"].reverse()
        reordered = {key: reordered[key] for key in reversed(list(reordered))}
        original_json, original_html = MOD.build_files(self.source)
        reordered_json, reordered_html = MOD.build_files(reordered)
        self.assertEqual(original_json, reordered_json)
        self.assertEqual(original_html, reordered_html)

    def test_builder_does_not_mutate_source(self):
        source = clone(self.source)
        before = clone(source)
        output = MOD.build_decision_summary(source)
        self.assertEqual(before, source)
        output["races"][0]["horse_summaries"][0]["basics"]["horse_name"] = "changed"
        self.assertEqual(before, source)

    def test_output_validator_rejects_rule_anchor_enum_and_shape_tampering(self):
        mutations = []
        real_mode = clone(self.output)
        real_mode["source_mode"] = "REAL_DATA"
        mutations.append(real_mode)
        weighted = clone(self.output)
        weighted["aggregation_rule"] = "WEIGHTED_SCORE"
        mutations.append(weighted)
        bad_hash = clone(self.output)
        bad_hash["source_projection_sha256"] = "invalid"
        mutations.append(bad_hash)
        hostile_role = clone(self.output)
        hostile_role["races"][0]["horse_summaries"][0]["role_expected_position"]["role"] = '<img src=x onerror=alert(1)>'
        mutations.append(hostile_role)
        active_anchor = clone(self.output)
        active_anchor["races"][0]["horse_summaries"][0]["detail_ref"] = "javascript:alert(1)"
        mutations.append(active_anchor)
        extra_nested = clone(self.output)
        extra_nested["races"][0]["horse_summaries"][0]["confidence"]["legacy"] = "HIGH"
        mutations.append(extra_nested)
        semantic_queue = clone(self.output)
        semantic_queue["races"][0]["horse_summaries"][0]["queue_fit"]["value"] = "ADVERSE"
        mutations.append(semantic_queue)
        semantic_confidence = clone(self.output)
        semantic_confidence["races"][0]["horse_summaries"][0]["confidence"]["value"] = "LOW"
        mutations.append(semantic_confidence)
        semantic_count = clone(self.output)
        semantic_count["races"][0]["horse_summaries"][0]["evidence_count"]["total_distinct"] = 99
        mutations.append(semantic_count)
        unsupported_trigger_id = clone(self.output)
        unsupported_trigger_id["races"][0]["horse_summaries"][0]["upside_triggers"][0][
            "evidence_ids"
        ] = ["D-MISSING-01"]
        mutations.append(unsupported_trigger_id)
        count_id_mismatch = clone(self.output)
        count_id_mismatch["races"][0]["horse_summaries"][0]["evidence_count"][
            "evidence_ids_by_dimension"
        ]["role"] = []
        mutations.append(count_id_mismatch)
        forbidden_world = clone(self.output)
        forbidden_world["races"][0]["horse_summaries"][0]["winning_or_in_the_money_world_state"] = "odds BUY"
        mutations.append(forbidden_world)
        for index, mutated in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises(MOD.DecisionSummaryError):
                    MOD.validate_decision_summary(mutated)
                with self.assertRaises(MOD.DecisionSummaryError):
                    MOD.render_decision_summary_html(mutated)

    def test_detail_anchor_encoding_is_injective_for_hyphenated_ids(self):
        self.assertNotEqual(MOD.detail_ref("A-B", "C"), MOD.detail_ref("A", "B-C"))

    def test_committed_sample_is_byte_deterministic_and_verifyable(self):
        expected_json, expected_html = MOD.build_files(self.source)
        self.assertEqual(expected_json, SAMPLE_JSON.read_bytes())
        self.assertEqual(expected_html, SAMPLE_HTML.read_bytes())
        MOD.verify_outputs(self.source, SAMPLE)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            MOD.write_outputs(self.source, output_dir)
            MOD.verify_outputs(self.source, output_dir)

    def test_ui_order_summary_visibility_and_expandable_details(self):
        race_start = self.html.index('<article class="race"')
        race_html = self.html[race_start:]
        header_pos = race_html.index('class="race-header"')
        decision_pos = race_html.index('class="decision-summary"')
        runner_pos = race_html.index('class="all-runners"')
        evidence_pos = race_html.index('class="evidence-section"')
        detail_pos = race_html.index('<details class="horse-evidence"')
        self.assertLess(header_pos, decision_pos)
        self.assertLess(decision_pos, runner_pos)
        self.assertLess(runner_pos, evidence_pos)
        self.assertLess(evidence_pos, detail_pos)
        probe = SummaryDomProbe()
        probe.feed(self.html)
        expected_ids = [runner["horse_id"] for runner in self.source["races"][0]["runners"]]
        self.assertEqual(expected_ids, probe.summary_horse_ids)
        self.assertEqual(
            [MOD.detail_ref("SYN-R01", horse_id).lstrip("#") for horse_id in expected_ids],
            probe.detail_ids,
        )
        self.assertIn("Review priority only", self.html)
        self.assertIn("能力順位ではありません", self.html)

    def test_html_escapes_hostile_horse_and_evidence_text(self):
        source = clone(self.source)
        hostile = '</summary><script>alert("x")</script><img src=x onerror=alert(1)>'
        source["races"][0]["runners"][0]["basics"]["horse_name"] = hostile
        source["races"][0]["runners"][0]["evidence_details"][0]["value"] = hostile
        rendered = MOD.render_decision_summary_html(MOD.build_decision_summary(source))
        self.assertNotIn(hostile, rendered)
        self.assertNotIn("<script>alert", rendered)
        self.assertNotIn("<img src=x", rendered)
        self.assertIn("&lt;/summary&gt;", rendered)

    def test_static_html_has_no_external_or_stateful_hooks(self):
        lower = self.html.lower()
        for forbidden in (
            "<script",
            "fetch(",
            "xmlhttprequest",
            "websocket",
            "eventsource",
            "sendbeacon",
            "serviceworker",
            "localstorage",
            "sessionstorage",
            "<iframe",
            "<form",
            "<base",
            "http://",
            "https://",
            "url(",
            "window.location",
        ):
            self.assertNotIn(forbidden, lower)

    def test_schema_marks_review_priority_and_conditional_contracts(self):
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual("https://json-schema.org/draft/2020-12/schema", schema["$schema"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            MOD.REVIEW_SEMANTICS,
            schema["properties"]["review_priority_semantics"]["const"],
        )
        horse_schema = schema["$defs"]["horseSummary"]
        self.assertFalse(horse_schema["additionalProperties"])
        self.assertIn("evidence_ids", schema["$defs"]["trigger"]["required"])
        self.assertIn(
            "evidence_ids_by_dimension",
            schema["$defs"]["evidenceCount"]["required"],
        )
        conditions = horse_schema["allOf"]
        self.assertTrue(any("CONDITIONAL_REVIEW" in json.dumps(item) for item in conditions))
        self.assertTrue(any("FRAGILE" in json.dumps(item) for item in conditions))


if __name__ == "__main__":
    unittest.main()
