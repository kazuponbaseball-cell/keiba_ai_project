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

QUEUE_SNAPSHOT = (
    ROOT
    / "docs"
    / "observations"
    / "race_intelligence_lite_plus_20260823"
    / "snapshot_20260822T180200JST_queue_v02"
)
QUEUE_DATA_PATH = QUEUE_SNAPSHOT / "race_intelligence_lite_plus_data.json"
QUEUE_HTML_PATH = QUEUE_SNAPSHOT / "race_intelligence_lite_plus.html"
QUEUE_OFFICIAL_PATH = QUEUE_SNAPSHOT / "official_current_entries.json"
QUEUE_TEMPLATE_PATH = QUEUE_SNAPSHOT / "human_scenario_freeze.template.json"
QUEUE_SOURCE_MANIFEST_PATH = QUEUE_SNAPSHOT / "source_manifest.json"

V1_FROZEN_SHA256 = {
    DATA_PATH: "d482f1e4aa7fb7beea357467f82bf54022e64026a8ad00c14e776520b25d05ed",
    HTML_PATH: "e95104f2fd51ea5cd6298a3191d81945d15dba68042160b7ddd7aab326fc7093",
    OFFICIAL_PATH: "ea54020f6434af09a621402026c52d9b64ea568f1f5bde1783116d13f37d9db1",
    TEMPLATE_PATH: "44f3461d15e40bfea38b5b61fb5d414f29c67c493655d5d65927430dc915207b",
    SOURCE_MANIFEST_PATH: "c170e1fb178dcf425b015e7536a7356ac814af75aa36fee458033010896b28b9",
}


def load_module():
    spec = importlib.util.spec_from_file_location("race_intelligence_lite_plus", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load_module()


def queue_run(
    tag: str,
    *,
    surface: str = "turf",
    distance_m: int = 1600,
    first_corner: int | None = 1,
    finish_position: int | None = 5,
    field_size: int | None = None,
) -> dict[str, object]:
    """Minimal transparent history fixture for Queue v0.2 unit contracts."""

    row: dict[str, object] = {
        "tag": tag,
        "surface": surface,
        "distance_m": distance_m,
        "corner1": first_corner or 0,
        "corner2": 0,
        "corner3": 0,
        "corner4": 0,
        "finish_position": "" if finish_position is None else finish_position,
    }
    if field_size is not None:
        row["field_size"] = field_size
    return row


def clone_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


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


class RaceIntelligenceLitePlusQueueV02Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v1_data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        cls.data = json.loads(QUEUE_DATA_PATH.read_text(encoding="utf-8"))
        cls.html = QUEUE_HTML_PATH.read_text(encoding="utf-8")
        cls.official = json.loads(QUEUE_OFFICIAL_PATH.read_text(encoding="utf-8"))
        cls.template = json.loads(QUEUE_TEMPLATE_PATH.read_text(encoding="utf-8"))
        cls.source_manifest = json.loads(QUEUE_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_01_v1_artifact_is_frozen_and_all_recent_role_is_compatible(self) -> None:
        for path, expected in V1_FROZEN_SHA256.items():
            with self.subTest(path=path.name):
                self.assertEqual(expected, MOD.sha256_path(path))
        MOD.command_verify(types.SimpleNamespace(
            observation=str(DATA_PATH), html=str(HTML_PATH),
            source_manifest=str(SOURCE_MANIFEST_PATH), official_entries=str(OFFICIAL_PATH),
            freeze_template=str(TEMPLATE_PATH),
        ))

        v1_roles = {
            (race["race_id"], horse["horse_id"]): horse["role_position"]["base_role"]
            for race in self.v1_data["races"] for horse in race["horses"]
        }
        self.assertEqual(70, len(v1_roles))
        for race in self.data["races"]:
            for horse in race["horses"]:
                old_role = v1_roles[(race["race_id"], horse["horse_id"])]
                current = horse["role_position"]["all_recent_role"]
                if old_role == "unobserved":
                    self.assertEqual("unobserved", current["status"])
                else:
                    self.assertEqual(old_role, current["value"])

    def test_02_target_context_role_uses_fixed_fallback_priority(self) -> None:
        target = {"surface": "turf", "distance_m": 1600}
        runs = [
            queue_run("tight", distance_m=1800, first_corner=1),
            queue_run("wide", distance_m=1900, first_corner=3),
            queue_run("surface", distance_m=2200, first_corner=6),
            queue_run("fallback", surface="dirt", distance_m=1600, first_corner=10),
        ]
        block, _, selected = MOD.target_context_profile(runs, target)
        self.assertEqual("same_surface_distance_200", block["selected_layer"])
        self.assertEqual("lead", block["value"])
        self.assertEqual(["tight"], [run["tag"] for run in selected])
        self.assertEqual(
            {
                "same_surface_distance_200": 1,
                "same_surface_distance_400": 2,
                "same_surface": 3,
                "all_recent_fallback": 4,
            },
            block["layer_evidence_count"],
        )

        cases = (
            (runs[1:], "same_surface_distance_400", "front", "wide"),
            (runs[2:], "same_surface", "stalking", "surface"),
            (runs[3:], "all_recent_fallback", "midpack", "fallback"),
        )
        for candidate_runs, expected_layer, expected_role, expected_tag in cases:
            with self.subTest(layer=expected_layer):
                value, _, chosen = MOD.target_context_profile(candidate_runs, target)
                self.assertEqual(expected_layer, value["selected_layer"])
                self.assertEqual(expected_role, value["value"])
                self.assertEqual([expected_tag], [run["tag"] for run in chosen])

    def test_03_target_context_never_mixes_a_different_surface(self) -> None:
        target = {"surface": "芝", "distance_m": 1600}
        runs = [
            queue_run("newer-dirt-lead", surface="dirt", first_corner=1),
            queue_run("target-turf-stalker", surface="turf", first_corner=6),
        ]
        block, _, selected = MOD.target_context_profile(runs, target)
        self.assertEqual("same_surface_distance_200", block["selected_layer"])
        self.assertEqual("stalking", block["value"])
        self.assertEqual(1, block["layer_evidence_count"]["same_surface_distance_200"])
        self.assertEqual(["target-turf-stalker"], [run["tag"] for run in selected])

    def test_04_distance_bands_are_inclusive_and_narrowest_band_wins(self) -> None:
        target = {"surface": "turf", "distance_m": 1600}
        runs = [
            queue_run("minus200", distance_m=1400),
            queue_run("plus200", distance_m=1800),
            queue_run("minus201", distance_m=1399),
            queue_run("plus201", distance_m=1801),
            queue_run("minus400", distance_m=1200),
            queue_run("plus400", distance_m=2000),
            queue_run("minus401", distance_m=1199),
            queue_run("plus401", distance_m=2001),
        ]
        block, _, selected = MOD.target_context_profile(runs, target)
        self.assertEqual(2, block["layer_evidence_count"]["same_surface_distance_200"])
        self.assertEqual(6, block["layer_evidence_count"]["same_surface_distance_400"])
        self.assertEqual("same_surface_distance_200", block["selected_layer"])
        self.assertEqual(["minus200", "plus200"], [run["tag"] for run in selected])

        wider, _, selected_wider = MOD.target_context_profile(runs[2:], target)
        self.assertEqual("same_surface_distance_400", wider["selected_layer"])
        self.assertEqual(
            ["minus201", "plus201", "minus400", "plus400"],
            [run["tag"] for run in selected_wider],
        )

    def test_05_missing_field_size_fails_safe_without_losing_absolute_position(self) -> None:
        runs = [queue_run("missing", first_corner=4), queue_run("invalid", first_corner=6, field_size=5)]
        absolute = MOD.first_position_summary(runs)
        normalized = MOD.normalized_position_summary(runs)
        self.assertEqual([4, 6], absolute["observed_positions"])
        self.assertEqual("derived", absolute["status"])
        self.assertEqual("unobserved", normalized["status"])
        self.assertIsNone(normalized["median_position_percentile"])
        self.assertEqual([], normalized["observed_percentiles"])
        self.assertEqual(2, normalized["field_size_missing_count"])
        self.assertTrue(normalized["missing_reason"])

        known = MOD.normalized_position_summary([
            queue_run("known", first_corner=4, field_size=13),
        ])
        self.assertEqual("derived", known["status"])
        self.assertEqual(0.25, known["median_position_percentile"])

        for race in self.data["races"]:
            for horse in race["horses"]:
                item = horse["role_position"]["normalized_position_proxy"]
                self.assertEqual("unobserved", item["status"])
                self.assertEqual(0, item["evidence_count"])
                self.assertTrue(item["missing_reason"])
                quartile = horse["role_position"]["forward_propensity"]["observed_front_quartile_rate"]
                self.assertEqual("unobserved", quartile["status"])
                self.assertEqual(0, quartile["denominator"])
                self.assertIsNone(quartile["value"])

    def test_06_observed_rates_reconcile_and_are_not_probability_claims(self) -> None:
        runs = [
            queue_run("lead", first_corner=1),
            queue_run("front", first_corner=2),
            queue_run("stalker", first_corner=5),
        ]
        block = MOD.forward_propensity_block(runs, runs[:2])
        expected = {
            "observed_lead_rate": (1, 3),
            "observed_front4_rate": (2, 3),
            "observed_front_quartile_rate": (0, 0),
            "context_lead_rate": (1, 2),
            "context_front_rate": (2, 2),
        }
        for key, (numerator, denominator) in expected.items():
            with self.subTest(rate=key):
                item = block[key]
                self.assertEqual(numerator, item["numerator"])
                self.assertEqual(denominator, item["denominator"])
                self.assertIn("確率ではない", item["note"])
                MOD.validate_observed_rate_record(item, key)
                if denominator:
                    self.assertEqual(round(numerator / denominator, 6), item["value"])
                else:
                    self.assertIsNone(item["value"])
                    self.assertEqual("unobserved", item["status"])

        empty = MOD.forward_propensity_block([], [])
        for key in expected:
            self.assertEqual(0, empty[key]["denominator"])
            self.assertIsNone(empty[key]["value"])
            self.assertTrue(empty[key]["missing_reason"])
        self.assertFalse(any("probability" in key.lower() for key in MOD.FORWARD_PROPENSITY_KEYS))

    def test_07_dependency_possible_never_becomes_need_lead(self) -> None:
        dependency = MOD.lead_dependency_block([
            queue_run("lead-win", first_corner=1, finish_position=1),
            queue_run("lead-place", first_corner=1, finish_position=2),
            queue_run("nonlead-miss", first_corner=2, finish_position=8),
        ])
        self.assertEqual("dependency_possible", dependency["dependency_status"])
        self.assertEqual(2, dependency["lead_run_count"])
        self.assertEqual(0, dependency["nonlead_good_finish_count"])
        self.assertEqual(0, dependency["can_rate_evidence_count"])
        self.assertIn("possible", dependency["dependency_status"])
        missing_outcome = MOD.lead_dependency_block([
            queue_run("lead-win", first_corner=1, finish_position=1),
            queue_run("lead-place", first_corner=1, finish_position=2),
            queue_run("nonlead-missing", first_corner=2, finish_position=None),
        ])
        self.assertNotEqual("dependency_possible", missing_outcome["dependency_status"])
        self.assertEqual(1, missing_outcome["missing_finish_count"])
        self.assertEqual(0, missing_outcome["nonlead_valid_finish_count"])

        horse_a = clone_json(self.data["races"][0]["horses"][0])
        horse_b = clone_json(self.data["races"][0]["horses"][1])
        for horse in (horse_a, horse_b):
            horse["role_position"]["forward_propensity"]["classification"] = "strong_forward"
            horse["role_position"]["role_range"].update({
                "value": "lead-front", "status": "proxy", "evidence_count": 3,
            })
        horse_a["role_position"]["lead_dependency_evidence"] = dependency
        horse_b["role_position"]["lead_dependency_evidence"] = missing_outcome
        short_first_turn = {"static_route": ["1コーナーまで約240mと短い"]}
        one_sided = MOD.pairwise_pressure(horse_a, horse_b, short_first_turn)
        self.assertNotEqual("high", one_sided["conflict_level"])

        horse_b["role_position"]["lead_dependency_evidence"] = clone_json(dependency)
        two_sided = MOD.pairwise_pressure(horse_a, horse_b, short_first_turn)
        self.assertEqual("high", two_sided["conflict_level"])
        self.assertIn("双方", two_sided["reason"])
        for race in self.data["races"]:
            for horse in race["horses"]:
                self.assertNotIn("need_lead", horse["role_position"])

    def test_08_can_rate_evidence_is_preserved_and_scoped_to_five_runs(self) -> None:
        observed = MOD.lead_dependency_block([
            queue_run("rated-good", first_corner=3, finish_position=2),
            queue_run("lead-good", first_corner=1, finish_position=1),
        ])
        self.assertEqual(1, observed["can_rate_evidence_count"])
        self.assertEqual(1, observed["nonlead_good_finish_count"])
        self.assertEqual("mixed", observed["dependency_status"])

        first_five = [
            queue_run(f"recent-{index}", first_corner=2, finish_position=8)
            for index in range(5)
        ]
        outside_scope = first_five + [queue_run("sixth", first_corner=3, finish_position=1)]
        self.assertEqual(0, MOD.lead_dependency_block(outside_scope)["can_rate_evidence_count"])
        sentinel = MOD.lead_dependency_block([
            queue_run("nonfinish", first_corner=3, finish_position=100),
        ])
        self.assertEqual(0, sentinel["can_rate_evidence_count"])

    def test_09_role_range_is_a_deterministic_ordered_interval(self) -> None:
        item = MOD.role_range_evidence([
            queue_run("lead", first_corner=1),
            queue_run("front", first_corner=3),
            queue_run("stalking", first_corner=6),
        ])
        self.assertEqual("lead-stalking", item["value"])
        self.assertIn("中央値=front", item["note"])
        self.assertEqual({"lead", "front", "stalking"}, MOD.role_range_tokens(item["value"]))

        role_order = list(MOD.ROLE_VALUES[:-1])
        for race in self.data["races"]:
            for horse in race["horses"]:
                role_range = horse["role_position"]["role_range"]
                if role_range["status"] == "unobserved":
                    self.assertEqual("未観測", role_range["value"])
                else:
                    endpoints = str(role_range["value"]).split("-")
                    self.assertIn(len(endpoints), (1, 2))
                    self.assertTrue(all(value in role_order for value in endpoints))
                    self.assertEqual(sorted(endpoints, key=role_order.index), endpoints)
                expected_band = horse["role_position"]["expected_position_band_proxy"]
                self.assertIn(expected_band["confidence"], MOD.CONFIDENCE_VALUES)
                if expected_band["status"] != "unobserved":
                    band_parts = str(expected_band["value"]).split("/")
                    self.assertIn(len(band_parts), (1, 2))
                    self.assertTrue(all(value in role_order for value in band_parts))
                    self.assertEqual(sorted(band_parts, key=role_order.index), band_parts)

    def test_10_pairwise_matrix_is_canonical_self_free_and_reconciled(self) -> None:
        for race in self.data["races"]:
            horses = race["horses"]
            queue = race["queue_pressure_map"]
            pairs = queue["pairwise_pressure_matrix"]["pairs"]
            pair_keys = [
                (int(pair["horse_a"]["horse_no"]), int(pair["horse_b"]["horse_no"]))
                for pair in pairs
            ]
            self.assertEqual(sorted(pair_keys), pair_keys)
            self.assertEqual(len(pair_keys), len(set(pair_keys)))
            self.assertTrue(all(a_no < b_no for a_no, b_no in pair_keys))
            self.assertEqual(len(pairs), queue["eligible_pair_count"])

            classes = {
                int(horse["basics"]["horse_no"]): horse["role_position"]["forward_propensity"]["classification"]
                for horse in horses
            }
            forward_count = sum(value in {"forward", "strong_forward"} for value in classes.values())
            nonforward_count = len(horses) - forward_count
            expected_pairs = (
                len(horses) * (len(horses) - 1) // 2
                - nonforward_count * (nonforward_count - 1) // 2
            )
            self.assertEqual(expected_pairs, len(pairs))
            expected_rivals: dict[int, set[tuple[int, str]]] = {
                number: set() for number in classes
            }
            for pair in pairs:
                a_no = int(pair["horse_a"]["horse_no"])
                b_no = int(pair["horse_b"]["horse_no"])
                self.assertTrue(classes[a_no] in {"forward", "strong_forward"}
                                or classes[b_no] in {"forward", "strong_forward"})
                self.assertIn(pair["conflict_level"], MOD.CONFLICT_LEVEL_VALUES)
                self.assertIn(pair["confidence"], MOD.CONFIDENCE_VALUES)
                self.assertTrue(str(pair["reason"]).strip())
                self.assertIn("判定しない", pair["reason"])
                if pair["conflict_level"] in {"high", "medium"}:
                    expected_rivals[a_no].add((b_no, pair["conflict_level"]))
                    expected_rivals[b_no].add((a_no, pair["conflict_level"]))
            for horse in horses:
                horse_no = int(horse["basics"]["horse_no"])
                actual = {
                    (int(rival["horse_no"]), rival["conflict_level"])
                    for rival in horse["role_position"]["pressure_rivals"]
                }
                self.assertEqual(expected_rivals[horse_no], actual)

    def test_11_queue_summary_counts_reconcile_to_horses_and_pairs(self) -> None:
        for race in self.data["races"]:
            horses = race["horses"]
            queue = race["queue_pressure_map"]
            pairs = queue["pairwise_pressure_matrix"]["pairs"]
            classifications = [
                horse["role_position"]["forward_propensity"]["classification"]
                for horse in horses
            ]
            dependencies = [
                horse["role_position"]["lead_dependency_evidence"] for horse in horses
            ]
            self.assertEqual(
                sum(value in {"forward", "strong_forward"} for value in classifications),
                queue["forward_candidate_count"],
            )
            self.assertEqual(
                sum(value == "strong_forward" for value in classifications),
                queue["strong_forward_candidate_count"],
            )
            self.assertEqual(
                sum(item["dependency_status"] == "dependency_possible" for item in dependencies),
                queue["lead_dependency_possible_count"],
            )
            self.assertEqual(
                sum(item["can_rate_evidence_count"] > 0 for item in dependencies),
                queue["can_rate_observed_count"],
            )
            self.assertEqual(
                sum(pair["conflict_level"] == "high" for pair in pairs),
                queue["high_conflict_pair_count"],
            )
            self.assertEqual(
                sum(pair["conflict_level"] == "medium" for pair in pairs),
                queue["medium_conflict_pair_count"],
            )
            expected_high_cost = [
                {"horse_no": horse["basics"]["horse_no"], "horse_name": horse["basics"]["horse_name"]}
                for horse in horses
                if horse["role_position"]["first_turn_position_cost"]["value"] == "high"
            ]
            self.assertEqual(expected_high_cost, queue["first_turn_high_cost_horses"])
            if not any(
                horse["role_position"]["normalized_position_proxy"]["evidence_count"]
                for horse in horses
            ):
                self.assertEqual("low", queue["queue_pressure_confidence"])
            self.assertEqual(
                sum(value != "unobserved" for value in classifications),
                queue["role_known_horse_count"],
            )
            normalized_count = sum(
                horse["role_position"]["forward_propensity"]["normalized_evidence_count"] > 0
                for horse in horses
            )
            self.assertEqual(normalized_count, queue["normalized_position_horse_count"])
        self.assertEqual(
            "low",
            MOD.queue_pressure_confidence_value(
                known_count=14, normalized_horse_count=1, total_count=14,
            ),
        )
        self.assertEqual(
            "low",
            MOD.queue_pressure_confidence_value(
                known_count=14, normalized_horse_count=11, total_count=14,
            ),
        )
        self.assertEqual(
            "medium",
            MOD.queue_pressure_confidence_value(
                known_count=14, normalized_horse_count=12, total_count=14,
            ),
        )
        inflated = clone_json(self.data)
        inflated["races"][0]["queue_pressure_map"]["queue_pressure_confidence"] = "medium"
        with self.assertRaises(MOD.LitePlusError):
            MOD.validate_observation_v02(inflated)
        inflated_level = clone_json(self.data)
        inflated_level["races"][0]["queue_pressure_map"]["queue_pressure_level"] = "HIGH"
        with self.assertRaises(MOD.LitePlusError):
            MOD.validate_observation_v02(inflated_level)

    def test_12_queue_high_does_not_imply_fast_and_human_freeze_is_independent(self) -> None:
        observation = clone_json(self.data)
        observation["races"][0]["queue_pressure_map"]["queue_pressure_level"] = "HIGH"
        self.assertEqual("", observation["races"][0]["human_scenario"]["main_scenario"])
        digest = MOD.sha256_bytes(MOD.canonical_json_bytes(observation))
        human = MOD.human_template(observation, digest)
        for override in human["overrides"]:
            override["main_scenario"] = "MIDDLE"
            override["alternative_scenarios"] = ["FAST"]
            override["confidence"] = "low"
            override["reason"] = "Queue pressure and pace remain separate human judgments"
        MOD.validate_human_freeze(
            human, observation, MOD.parse_iso_datetime("2026-08-22T18:03:00+09:00")
        )
        self.assertEqual("MIDDLE", human["overrides"][0]["main_scenario"])
        for race in self.data["races"]:
            self.assertIn("HIGH != FAST", race["queue_pressure_map"]["pace_conversion"])
            self.assertIn("never changed automatically", race["queue_pressure_map"]["human_scenario_relation"])
            self.assertEqual("UNFROZEN_HUMAN_INPUT_REQUIRED", race["human_scenario"]["freeze_status"])
        self.assertIn("Queue Pressure HIGH ≠ FAST", self.html)
        self.assertIn("Queue HIGH / Human MIDDLE", self.html)

    def test_13_horse_number_order_and_all_70_cards_are_preserved(self) -> None:
        official_by_race = {race["race_id"]: race for race in self.official["races"]}
        self.assertEqual([1, 2, 3, 4, 5], [race["leg"] for race in self.data["races"]])
        total = 0
        for race in self.data["races"]:
            horse_numbers = [int(horse["basics"]["horse_no"]) for horse in race["horses"]]
            official_numbers = [
                int(runner["horse_no"]) for runner in official_by_race[race["race_id"]]["runners"]
            ]
            comparison_numbers = [int(row["horse_no"]) for row in race["comparison"]]
            display_numbers = [int(horse["display_order"]) for horse in race["horses"]]
            self.assertEqual(sorted(horse_numbers), horse_numbers)
            self.assertEqual(official_numbers, horse_numbers)
            self.assertEqual(horse_numbers, comparison_numbers)
            self.assertEqual(horse_numbers, display_numbers)
            total += len(horse_numbers)
        self.assertEqual(70, total)

    def test_14_no_ranking_and_compound_prediction_keys_fail_closed(self) -> None:
        forbidden = (
            "lead_probability", "lead_dependency_score", "queue_pressure_rank",
            "queue_ranking",
            "queue_pressure_probability", "queue_pressure_score", "position_probability",
            "expected_position_score", "expected_position_rank",
            "forward_propensity_probability", "forward_propensity_score",
            "pairwise_pressure_probability", "queue_pressure_fast_probability",
            "lead_take_probability", "scenario_fast_probability",
            "conflict_probability", "conflict_score", "scenario_probability",
        )
        for key in forbidden:
            with self.subTest(key=key):
                with self.assertRaises(MOD.LitePlusError):
                    MOD.reject_forbidden_keys([key], "compound_fixture")
        MOD.reject_forbidden_keys(
            ["observed_lead_rate", "observed_front4_rate", "context_lead_rate"],
            "allowed_observed_rates",
        )
        self.assertFalse(self.data["safety"]["ranking_claims"])
        self.assertIn("not a rank", self.data["display_order_rule"])
        self.assertIs(False, self.data["data_contract"]["formal_model_connection"])
        for race in self.data["races"]:
            for column in race["comparison_columns"]:
                self.assertNotRegex(column.lower(), r"(^|_)(score|rank|probability)($|_)")

        mutated = clone_json(self.data)
        mutated["races"][0]["horses"][0]["role_position"]["forward_propensity"][
            "lead_probability"
        ] = 0.5
        with self.assertRaises(MOD.LitePlusError):
            MOD.validate_observation_v02(mutated)

    def test_15_v02_artifact_verifies_and_forbidden_scan_passes(self) -> None:
        MOD.validate_observation_v02(self.data)
        MOD.reject_forbidden_recursive(self.data, "queue_v02_fixture")
        MOD.command_verify(types.SimpleNamespace(
            observation=str(QUEUE_DATA_PATH), html=str(QUEUE_HTML_PATH),
            source_manifest=str(QUEUE_SOURCE_MANIFEST_PATH),
            official_entries=str(QUEUE_OFFICIAL_PATH), freeze_template=str(QUEUE_TEMPLATE_PATH),
        ))
        digest = MOD.sha256_bytes(MOD.canonical_json_bytes(self.data))
        self.assertEqual(self.html, MOD.render_observation_html(self.data, digest))
        self.assertEqual(
            MOD.sha256_path(QUEUE_DATA_PATH),
            self.source_manifest["outputs"]["observation_data"]["sha256"],
        )
        self.assertEqual(
            MOD.sha256_path(QUEUE_HTML_PATH), self.source_manifest["outputs"]["html"]["sha256"]
        )

    def test_16_formal_buy_is_strict_boolean_false(self) -> None:
        for safety in (self.data["safety"], self.source_manifest["safety"]):
            self.assertIs(type(safety["formal_buy"]), bool)
            self.assertIs(safety["formal_buy"], False)
        mutated = clone_json(self.data)
        mutated["safety"]["formal_buy"] = 0
        with self.assertRaises(MOD.LitePlusError):
            MOD.validate_observation_v02(mutated)

    def test_17_send_order_is_strict_boolean_false(self) -> None:
        for safety in (self.data["safety"], self.source_manifest["safety"]):
            self.assertIs(type(safety["send_order"]), bool)
            self.assertIs(safety["send_order"], False)
        mutated = clone_json(self.data)
        mutated["safety"]["send_order"] = 0
        with self.assertRaises(MOD.LitePlusError):
            MOD.validate_observation_v02(mutated)

    def test_18_stake_is_strict_integer_zero_not_boolean_false(self) -> None:
        for safety in (self.data["safety"], self.source_manifest["safety"]):
            self.assertIs(type(safety["stake"]), int)
            self.assertEqual(0, safety["stake"])
        mutated = clone_json(self.data)
        mutated["safety"]["stake"] = False
        with self.assertRaises(MOD.LitePlusError):
            MOD.validate_observation_v02(mutated)

    def test_19_launch_and_first_observed_corner_are_explicitly_distinct(self) -> None:
        self.assertNotIn("start_acceleration", set(recursive_keys(self.data)))
        for race in self.data["races"]:
            for horse in race["horses"]:
                role = horse["role_position"]
                self.assertEqual("unobserved", role["launch_speed"]["status"])
                self.assertEqual("未観測", role["launch_speed"]["value"])
                self.assertTrue(role["launch_speed"]["missing_reason"])
                self.assertIn("ではなく", role["first_observed_position"]["note"])
        self.assertEqual(
            "short",
            MOD.first_turn_run_class({"static_route": ["約200mで本線へ合流", "直線412.5m"]}),
        )
        self.assertEqual(
            "long", MOD.first_turn_run_class({"static_route": ["3コーナーまで500m以上"]})
        )
        self.assertEqual(
            "unobserved",
            MOD.first_turn_run_class({"static_route": ["ホームストレッチからスタート", "坂後も200m余り"]}),
        )

    def test_20_static_html_exposes_queue_contract_without_runtime_hooks(self) -> None:
        self.assertEqual(5, self.html.count('<article class="race"'))
        self.assertEqual(70, self.html.count('<details class="horse-card role-v02"'))
        self.assertEqual(70, self.html.count("<h4>Role / Position v0.2</h4>"))
        self.assertEqual(
            5,
            self.html.count('<div class="panel span-2 queue-panel"><h3>Queue Pressure Map</h3>'),
        )
        self.assertEqual(5, self.html.count('<table class="pairwise-table">'))
        for phrase in (
            "Observation-only", "No AI rank", "No probability", "No odds / no stake",
            "No BUY / no order", "Pre-race freeze required", "Forward Propensity",
            "Lead Dependency Evidence", "first observed position", "normalized position proxy",
            "expected position band proxy", "Historical observed rates ≠ current-race probability",
            "Queue Pressure HIGH ≠ FAST", "Human Scenario / Pre-race Freeze",
        ):
            self.assertIn(phrase, self.html)
        lowered = self.html.lower()
        for forbidden in (
            "fetch(", "xmlhttprequest", "websocket", "localstorage", "form action=", "<script src=",
        ):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
