from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "build_race_intelligence_lite_plus.py"
SNAPSHOT = (
    ROOT
    / "docs"
    / "observations"
    / "race_intelligence_lite_plus_20260823"
    / "snapshot_20260822T132226JST"
)
DATA_PATH = SNAPSHOT / "race_intelligence_lite_plus_data.json"
HTML_PATH = SNAPSHOT / "race_intelligence_lite_plus.html"
OFFICIAL_PATH = SNAPSHOT / "official_current_entries.json"
TEMPLATE_PATH = SNAPSHOT / "human_scenario_freeze.template.json"
SOURCE_MANIFEST_PATH = SNAPSHOT / "source_manifest.json"


def load_module():
    spec = importlib.util.spec_from_file_location("race_intelligence_lite_plus", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load_module()


class RaceIntelligenceLitePlusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        cls.html = HTML_PATH.read_text(encoding="utf-8")
        cls.official = json.loads(OFFICIAL_PATH.read_text(encoding="utf-8"))
        cls.template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        cls.source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_stdlib_only_and_no_existing_dashboard_imports(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        allowed = {
            "__future__", "argparse", "csv", "hashlib", "html", "json", "math", "re",
            "statistics", "urllib", "collections", "datetime", "pathlib", "typing", "zoneinfo",
        }
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        self.assertEqual(set(), roots - allowed)
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden_import in (
            "build_keiba_dashboard_html", "build_preday_dashboard_html",
            "build_live_odds_dashboard_html", "weekend_scenario_lab",
        ):
            self.assertNotIn(f"import {forbidden_import}", source)

    def test_fail_closed_for_prediction_and_market_columns(self) -> None:
        for key in (
            "ai_score", "ai_rank", "slow_ai_score", "win_probability", "place_prob",
            "odds", "popularity", "market_price", "roi", "ev", "mispricing",
            "buy_ticket", "champion", "notification_target", "order_id",
        ):
            with self.subTest(key=key):
                with self.assertRaises(MOD.LitePlusError):
                    MOD.reject_forbidden_keys([key], "mutated_fixture")
        # These are allowed only as false-valued safety attestations.
        MOD.reject_forbidden_keys(["historical_market_columns_included"], "safe_fixture")
        with self.assertRaises(MOD.LitePlusError):
            MOD.validate_safety_rows([{"ability_proxy_added_to_ai_score": "true"}], "unsafe")

    def test_committed_observation_is_five_races_and_all_70_horses(self) -> None:
        MOD.validate_observation(self.data)
        self.assertEqual(5, self.data["race_count"])
        self.assertEqual(70, self.data["runner_count"])
        self.assertEqual([14, 14, 10, 16, 16], [race["runner_count"] for race in self.data["races"]])
        keys = [
            (race["race_id"], horse["horse_id"])
            for race in self.data["races"]
            for horse in race["horses"]
        ]
        self.assertEqual(70, len(keys))
        self.assertEqual(70, len(set(keys)))

    def test_every_horse_has_required_blocks_and_transfer_limit(self) -> None:
        horse_blocks = {
            "basics", "role_position", "ability", "pace_role_traits", "course_shape_traits",
            "condition", "transfer_evidence", "scenario_sensitivity", "paths", "uncertainty",
        }
        basics = {
            "horse_name", "frame_no", "horse_no", "jockey", "trainer", "sex", "age",
            "assigned_weight",
        }
        role_fields = {
            "base_role", "position_flexibility", "need_lead", "can_rate", "pressure_rivals",
            "inside_neighbor_context", "outside_neighbor_context", "likely_early_band",
            "queue_confidence", "queue_notes", "lead_frequency_proxy",
        }
        for race in self.data["races"]:
            self.assertEqual(race["runner_count"], len(race["comparison"]))
            for horse in race["horses"]:
                self.assertTrue(horse_blocks <= set(horse))
                self.assertTrue(basics <= set(horse["basics"]))
                self.assertTrue(role_fields <= set(horse["role_position"]))
                self.assertLessEqual(len(horse["transfer_evidence"]), 3)
                self.assertEqual({"SLOW", "MIDDLE", "FAST"}, set(horse["scenario_sensitivity"]))
                for item in horse["scenario_sensitivity"].values():
                    self.assertIn(item["assessment"], MOD.SENSITIVITY_VALUES)

    def test_evidence_missingness_is_explicit(self) -> None:
        evidence_count = 0
        unobserved_count = 0
        for item in MOD.walk_evidence(self.data):
            evidence_count += 1
            self.assertIn(item["status"], MOD.STATUS_VALUES)
            self.assertIn(item["route_match_level"], MOD.ROUTE_MATCH_VALUES)
            if item["status"] == "unobserved":
                unobserved_count += 1
                self.assertTrue(item["missing_reason"])
                self.assertNotEqual(0, item["value"])
        self.assertGreater(evidence_count, 1000)
        self.assertGreater(unobserved_count, 100)

    def test_same_condition_and_unfrozen_scenario_contract(self) -> None:
        self.assertEqual([0, 0, 0, 0, 0], [r["same_condition_evidence"]["exact_count"] for r in self.data["races"]])
        self.assertEqual([32, 100, 47, 163, 7], [r["same_condition_evidence"]["partial_count"] for r in self.data["races"]])
        self.assertEqual([0, 0, 0, 0, 0], [r["same_condition_evidence"]["similar_count"] for r in self.data["races"]])
        for race in self.data["races"]:
            self.assertEqual("unobserved", race["historical_pace_tendency"]["slow_middle_fast"]["status"])
            self.assertEqual("unobserved", race["historical_pace_tendency"]["legacy_shape_distribution"]["status"])
            self.assertEqual("UNFROZEN_HUMAN_INPUT_REQUIRED", race["human_scenario"]["freeze_status"])
            self.assertEqual("unobserved", race["track_condition"]["status"])

    def test_comparison_columns_and_order_are_fixed(self) -> None:
        for race in self.data["races"]:
            self.assertEqual(list(MOD.COMPARISON_COLUMNS), race["comparison_columns"])
            numbers = [int(row["horse_no"]) for row in race["comparison"]]
            self.assertEqual(sorted(numbers), numbers)
            self.assertEqual(
                [horse["basics"]["horse_name"] for horse in race["horses"]],
                [row["horse_name"] for row in race["comparison"]],
            )

    def test_official_current_field_snapshot_is_complete_and_market_free(self) -> None:
        self.assertEqual(5, self.official["race_count"])
        self.assertEqual(70, self.official["runner_count"])
        self.assertTrue(self.official["past_performance_cells_excluded"])
        self.assertFalse(self.official["market_fields_extracted"])
        rows = [runner for race in self.official["races"] for runner in race["runners"]]
        self.assertEqual(70, len(rows))
        for row in rows:
            for key in ("frame_no", "horse_no", "horse_name", "sex", "age", "assigned_weight", "jockey", "trainer"):
                self.assertNotIn(row[key], (None, ""))

    def test_static_html_has_all_cards_and_no_remote_runtime_hooks(self) -> None:
        self.assertEqual(5, self.html.count('<article class="race"'))
        self.assertEqual(70, self.html.count('<details class="horse-card"'))
        for phrase in (
            "Observation-only", "No AI rank", "No probability", "No odds / no stake",
            "No BUY / no order", "Pre-race freeze required", "全頭比較テーブル",
            "Human Scenario / Pre-race Freeze",
        ):
            self.assertIn(phrase, self.html)
        self.assertEqual(69, self.html.count("<th>observed race shape</th>"))
        for sentinel in ("100着", "300着", "400着", "99.9秒"):
            self.assertNotIn(sentinel, self.html)
        lowered = self.html.lower()
        for forbidden in ("fetch(", "xmlhttprequest", "websocket", "localstorage", "form action=", "<script src="):
            self.assertNotIn(forbidden, lowered)

    def test_render_is_deterministic_and_escapes_hostile_text(self) -> None:
        digest = MOD.sha256_bytes(MOD.canonical_json_bytes(self.data))
        first = MOD.render_html(self.data, digest)
        second = MOD.render_html(self.data, digest)
        self.assertEqual(first, second)
        hostile = MOD.render_evidence_item(
            "<script>",
            MOD.evidence("<img src=x>", status="observed", evidence_count=1,
                         confidence="low", route_match_level="partial"),
        )
        self.assertNotIn("<script>", hostile)
        self.assertNotIn("<img src=x>", hostile)

    def test_human_freeze_is_pre_race_hash_bound_and_non_overwriting(self) -> None:
        human = json.loads(json.dumps(self.template))
        for override in human["overrides"]:
            override["main_scenario"] = "MIDDLE"
            override["alternative_scenarios"] = ["FAST"]
            override["confidence"] = "low"
            override["reason"] = "Human observation fixture; no probability"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            human_path = root / "human.json"
            human_path.write_text(json.dumps(human, ensure_ascii=False), encoding="utf-8")
            output = root / "freeze_20260823T140000JST"
            manifest = MOD.freeze_human_scenario(
                DATA_PATH, human_path, output, MOD.parse_iso_datetime("2026-08-23T14:00:00+09:00")
            )
            self.assertTrue((output / "human_scenario_freeze.json").is_file())
            self.assertTrue((output / "freeze_manifest.json").is_file())
            self.assertTrue(manifest["immutable_pre_race"])
            with self.assertRaises(MOD.LitePlusError):
                MOD.freeze_human_scenario(
                    DATA_PATH, human_path, output, MOD.parse_iso_datetime("2026-08-23T14:01:00+09:00")
                )
            with self.assertRaises(MOD.LitePlusError):
                MOD.validate_human_freeze(
                    human, self.data, MOD.parse_iso_datetime("2026-08-23T15:00:00+09:00")
                )

    def test_prohibited_fields_cannot_enter_observation_or_freeze(self) -> None:
        mutated = json.loads(json.dumps(self.data))
        mutated["races"][0]["horses"][0]["ai_score"] = 0.99
        with self.assertRaises(MOD.LitePlusError):
            MOD.validate_observation(mutated)
        human = json.loads(json.dumps(self.template))
        for override in human["overrides"]:
            override.update({"main_scenario": "MIDDLE", "confidence": "low", "reason": "fixture"})
        human["overrides"][0]["odds"] = 1.2
        with self.assertRaises(MOD.LitePlusError):
            MOD.validate_human_freeze(
                human, self.data, MOD.parse_iso_datetime("2026-08-23T14:00:00+09:00")
            )

    def test_overall_confidence_is_capped_by_critical_missingness(self) -> None:
        for race in self.data["races"]:
            for horse in race["horses"]:
                uncertainty = horse["uncertainty"]
                expected = "low" if uncertainty["history_evidence_count"] else "unobserved"
                self.assertEqual(expected, uncertainty["confidence"])
                self.assertIn(uncertainty["history_coverage_confidence"], MOD.CONFIDENCE_VALUES)

    def test_verify_binds_html_json_and_source_manifest(self) -> None:
        MOD.command_verify(types.SimpleNamespace(
            observation=str(DATA_PATH), html=str(HTML_PATH), source_manifest=str(SOURCE_MANIFEST_PATH),
            official_entries=str(OFFICIAL_PATH), freeze_template=str(TEMPLATE_PATH),
        ))
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp) / "fake.html"
            fake.write_text("Observation-only No AI rank No odds / no stake Pre-race freeze required", encoding="utf-8")
            with self.assertRaises(MOD.LitePlusError):
                MOD.command_verify(types.SimpleNamespace(
                    observation=str(DATA_PATH), html=str(fake), source_manifest=str(SOURCE_MANIFEST_PATH),
                    official_entries=str(OFFICIAL_PATH), freeze_template=str(TEMPLATE_PATH),
                ))

    def test_source_manifest_is_allowlisted_and_exclusion_is_explicit(self) -> None:
        kinds = {item["kind"] for item in self.source_manifest["allowlisted_sources"]}
        self.assertEqual(
            {
                "target_race_name_manifest", "route_requirement_cards", "resolved_official_targets",
                "declared_runner_audit", "market_excluded_history", "same_condition_coverage",
                "horse_readiness", "horse_route_coverage", "official_current_entry",
            },
            kinds,
        )
        for item in self.source_manifest["allowlisted_sources"]:
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(MOD.sha256_path(DATA_PATH), self.source_manifest["outputs"]["observation_data"]["sha256"])
        self.assertEqual(MOD.sha256_path(HTML_PATH), self.source_manifest["outputs"]["html"]["sha256"])
        exclusions = " ".join(self.source_manifest["excluded_source_classes"])
        for phrase in ("legacy AI", "prediction/model", "odds/popularity", "BUY", "EXP-033", "EXP-034"):
            self.assertIn(phrase, exclusions)
        self.assertEqual(MOD.SAFETY, self.source_manifest["safety"])


if __name__ == "__main__":
    unittest.main()
