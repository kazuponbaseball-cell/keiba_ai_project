from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_PATH = ROOT / "scripts" / "research" / "build_strict_t3_capture_packet_v1.py"
RUNNER_PATH = ROOT / "scripts" / "research" / "run_strict_t3_shadow_decision_v1.py"
REPORT_PATH = ROOT / "scripts" / "research" / "build_strict_t3_shadow_report_v1.py"
CONFIG_PATH = ROOT / "config" / "strict_t3_shadow_decision_exp011.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CAPTURE = load_module("build_strict_t3_capture_packet_v1", CAPTURE_PATH)
RUNNER = load_module("run_strict_t3_shadow_decision_v1", RUNNER_PATH)
REPORT = load_module("build_strict_t3_shadow_report_v1", REPORT_PATH)


class StrictT3ShadowReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = CAPTURE.load_config(CONFIG_PATH)

    @staticmethod
    def race_id(card_index: int, race_no: int) -> str:
        prefixes = ("2026010104", "2026070204", "2026040204")
        return f"{prefixes[card_index]}{race_no:02d}"

    def build_inputs(self, root: Path) -> tuple[Path, Path, list[Path]]:
        failure_positions = {
            (0, 1), (0, 2), (0, 3),
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 1), (2, 2), (2, 3), (2, 4),
        }
        sources = []
        source_paths: list[Path] = []
        decisions = []
        for card_index in range(3):
            card_dir = root / f"card-{card_index}"
            packets_dir = card_dir / "packets"
            packets_dir.mkdir(parents=True)
            acknowledgements = []
            for race_no in range(1, 13):
                race_id = self.race_id(card_index, race_no)
                ready = (card_index, race_no) not in failure_positions
                candidate = {
                    "schema_version": 1,
                    "experiment_id": "EXP-20260802-010",
                    "cohort_id": "synthetic-source",
                    "card_id": f"card-{card_index}",
                    "race_id": race_id,
                    "race_no": race_no,
                    "scheduled_post_time_asof": "2026-08-02T15:00:00+09:00",
                    "record_status": "CANDIDATE_READY" if ready else "FAILED",
                    "candidate_freeze_contract_ok": ready,
                    "failure_reason_codes": [] if ready else ["CANDIDATE_SOURCE_NOT_READY"],
                    "candidate_horse_id_1": "1" if ready else "",
                    "candidate_horse_id_2": "2" if ready else "",
                    "candidate_pair_key": "1-2" if ready else "",
                    "p_wide_coherent_raw": 0.39 if ready else None,
                    "p_action_calibrated": 0.40 if ready else None,
                    "confidence_gate_pass": ready,
                    "starter_universe_hash_at_freeze": f"universe-{race_id}",
                    "candidate_uses_odds": False,
                    "formal_buy": False,
                    "send_order": False,
                    "stake": 0,
                }
                candidate["candidate_freeze_record_hash"] = CAPTURE.candidate_record_digest(
                    candidate
                )
                relative_path = Path("packets") / f"{race_id}.candidate_freeze.json"
                packet_path = card_dir / relative_path
                packet_path.write_text(
                    CAPTURE.canonical_json(candidate) + "\n", encoding="utf-8"
                )
                acknowledgement = {
                    "experiment_id": "EXP-20260802-010",
                    "race_id": race_id,
                    "race_no": race_no,
                    "record_status": candidate["record_status"],
                    "candidate_freeze_record_hash": candidate[
                        "candidate_freeze_record_hash"
                    ],
                    "candidate_freeze_persist_ack_at": "2026-08-02T14:55:00+09:00",
                    "packet_path": relative_path.as_posix(),
                    "packet_file_sha256": CAPTURE.file_sha256(packet_path),
                    "candidate_uses_odds": False,
                    "formal_buy": False,
                    "send_order": False,
                    "stake": 0,
                }
                acknowledgements.append(acknowledgement)
                decision = {
                    "schema_version": 1,
                    "event_type": "strict_t3_shadow_decision",
                    "experiment_id": self.config["experiment_id"],
                    "source_experiment_id": self.config["source_experiment_id"],
                    "cohort_id": self.config["cohort_id"],
                    "card_id": f"card-{card_index}",
                    "race_id": race_id,
                    "race_no": race_no,
                    "candidate_freeze_record_hash": candidate[
                        "candidate_freeze_record_hash"
                    ],
                    "candidate_pair_key": candidate["candidate_pair_key"],
                    "confidence_gate_pass": ready,
                    "p_action_calibrated": candidate["p_action_calibrated"],
                    "quote_evaluation_entered": ready,
                    "candidate_pair_t3_quote_valid": ready,
                    "t3_wide_odds_low": 4.0 if ready else None,
                    "t3_wide_odds_high": 4.6 if ready else None,
                    "research_expected_return_low": 1.6 if ready else None,
                    "shadow_action": "PAPER_READY" if ready else "NO_BET",
                    "decision_reason": (
                        "STRICT_T3_CONTRACTS_PASSED"
                        if ready
                        else "NO_BET_SOURCE_NOT_READY"
                    ),
                    "measurement_only": False,
                    "t3_quote_selected_asof_time": (
                        "2026-08-02T14:56:55+09:00" if ready else None
                    ),
                    "t3_decision_committed_at": "2026-08-02T14:57:05+09:00",
                    "candidate_uses_odds": False,
                    "candidate_changed_after_odds": False,
                    "formal_buy": False,
                    "send_order": False,
                    "stake": 0,
                    "idempotency_key": CAPTURE.canonical_digest(
                        {"race_id": race_id, "event": "decision"}
                    ),
                }
                decision["t3_decision_record_hash"] = RUNNER._decision_digest(decision)
                RUNNER.verify_decision_record(decision)
                decisions.append(decision)
            ledger_path = card_dir / "candidate_freeze_ledger.jsonl"
            ledger_path.write_text(
                "".join(CAPTURE.canonical_json(row) + "\n" for row in acknowledgements),
                encoding="utf-8",
            )
            source_paths.extend([ledger_path, *packets_dir.glob("*.json")])
            sources.append(
                {
                    "card_id": f"card-{card_index}",
                    "candidate_ledger_jsonl": str(ledger_path),
                    "candidate_ledger_sha256": CAPTURE.file_sha256(ledger_path),
                }
            )
        manifest = {
            "schema_version": 1,
            "experiment_id": self.config["experiment_id"],
            "source_experiment_id": self.config["source_experiment_id"],
            "data_class": "synthetic",
            "sources": sources,
        }
        manifest_path = root / "source_manifest.json"
        manifest_path.write_text(
            CAPTURE.canonical_json(manifest) + "\n", encoding="utf-8"
        )
        source_paths.append(manifest_path)
        decision_path = root / "decisions.jsonl"
        decision_path.write_text(
            "".join(CAPTURE.canonical_json(row) + "\n" for row in decisions),
            encoding="utf-8",
        )
        source_paths.append(decision_path)
        return manifest_path, decision_path, source_paths

    def test_report_has_all_36_rows_and_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, decisions, source_paths = self.build_inputs(root)
            before = {str(path): CAPTURE.file_sha256(path) for path in source_paths}
            payload = REPORT.write_report(
                source_manifest_path=manifest,
                decision_ledger_path=decisions,
                output_json_path=root / "report" / "strict_t3.json",
                output_html_path=root / "report" / "strict_t3.html",
                config=self.config,
                generated_at="2026-08-02T16:00:00+09:00",
            )
            after = {str(path): CAPTURE.file_sha256(path) for path in source_paths}
            report_html = (root / "report" / "strict_t3.html").read_text(
                encoding="utf-8"
            ).lower()
            self.assertEqual(before, after)
            self.assertEqual(payload["record_count"], 36)
            self.assertEqual(payload["candidate_ready_rows"], 25)
            self.assertEqual(payload["source_failure_rows"], 11)
            self.assertEqual(payload["quote_evaluation_rows"], 25)
            self.assertFalse(payload["formal_buy"])
            self.assertFalse(payload["send_order"])
            self.assertEqual(payload["stake"], 0)
            self.assertNotIn("<form", report_html)
            self.assertNotIn("<button", report_html)
            self.assertNotIn("<input", report_html)

    def test_source_failures_have_no_candidate_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, decisions, _ = self.build_inputs(root)
            payload = REPORT.build_report_payload(
                source_manifest_path=manifest,
                decision_ledger_path=decisions,
                config=self.config,
                generated_at="2026-08-02T16:00:00+09:00",
            )
            failed = [
                row for row in payload["rows"] if row["source_candidate_status"] == "FAILED"
            ]
            self.assertEqual(len(failed), 11)
            self.assertTrue(all(row["candidate_pair_key"] == "" for row in failed))
            self.assertTrue(
                all(row["decision_reason"] == "NO_BET_SOURCE_NOT_READY" for row in failed)
            )

    def test_production_dashboard_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "outputs" / "ui" / "live_odds_dashboard.html"
            with self.assertRaisesRegex(ValueError, "production dashboard"):
                REPORT.assert_report_output_path_allowed(path, self.config)

    def test_missing_or_duplicate_decision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, decisions, _ = self.build_inputs(root)
            lines = decisions.read_text(encoding="utf-8").splitlines()
            decisions.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one decision"):
                REPORT.build_report_payload(
                    source_manifest_path=manifest,
                    decision_ledger_path=decisions,
                    config=self.config,
                    generated_at="2026-08-02T16:00:00+09:00",
                )
            decisions.write_text("\n".join(lines + [lines[0]]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate race_id"):
                REPORT.build_report_payload(
                    source_manifest_path=manifest,
                    decision_ledger_path=decisions,
                    config=self.config,
                    generated_at="2026-08-02T16:00:00+09:00",
                )


if __name__ == "__main__":
    unittest.main()
