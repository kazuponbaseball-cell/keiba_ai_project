from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_PATH = ROOT / "scripts" / "research" / "build_strict_t3_capture_packet_v1.py"
RUNNER_PATH = ROOT / "scripts" / "research" / "run_strict_t3_shadow_decision_v1.py"
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


class StrictT3ShadowDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = CAPTURE.load_config(CONFIG_PATH)

    @staticmethod
    def race_id(card_index: int, race_no: int) -> str:
        prefixes = ("2026010104", "2026070204", "2026040204")
        return f"{prefixes[card_index]}{race_no:02d}"

    def candidate(self, race_id: str, race_no: int, *, ready: bool = True, confidence: bool = True) -> dict:
        candidate = {
            "schema_version": 1,
            "experiment_id": "EXP-20260802-010",
            "cohort_id": "synthetic-source",
            "card_id": f"card-{race_id[4:10]}",
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
            "confidence_gate_pass": confidence if ready else False,
            "starter_universe_hash_at_freeze": CAPTURE.canonical_digest(
                {"race_id": race_id, "runners": ["1", "2", "3"]}
            ),
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        candidate["candidate_freeze_record_hash"] = CAPTURE.candidate_record_digest(
            candidate
        )
        return candidate

    def acknowledgement(self, candidate: dict, packet_sha: str, packet_path: str) -> dict:
        return {
            "schema_version": 1,
            "experiment_id": "EXP-20260802-010",
            "cohort_id": "synthetic-source",
            "race_id": candidate["race_id"],
            "race_no": candidate["race_no"],
            "record_status": candidate["record_status"],
            "candidate_freeze_record_hash": candidate["candidate_freeze_record_hash"],
            "candidate_freeze_persist_ack_at": "2026-08-02T14:55:00+09:00",
            "packet_path": packet_path,
            "packet_file_sha256": packet_sha,
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }

    @staticmethod
    def schedule(race_id: str, *, contract_ok: bool = True) -> dict:
        record = {
            "race_id": race_id,
            "scheduled_post_time_used": "2026-08-02T15:00:00+09:00",
            "schedule_version": "schedule-v1",
            "schedule_source_event_time": "2026-08-02T14:54:00+09:00",
            "schedule_received_at": "2026-08-02T14:54:01+09:00",
            "schedule_contract_ok": contract_ok,
        }
        record["schedule_record_hash"] = CAPTURE.schedule_record_digest(record)
        return record

    @staticmethod
    def universe(race_id: str, *, changed: bool = False, scratch: bool = False) -> dict:
        return {
            "race_id": race_id,
            "starter_universe_hash_at_t3": (
                "changed-universe"
                if changed
                else CAPTURE.canonical_digest(
                    {"race_id": race_id, "runners": ["1", "2", "3"]}
                )
            ),
            "starter_universe_unchanged": not changed,
            "scratch_known_by_t3": scratch,
            "universe_observed_at": "2026-08-02T14:56:59+09:00",
        }

    @staticmethod
    def quote(race_id: str, **overrides) -> dict:
        record = {
            "race_id": race_id,
            "ticket_type": "wide",
            "quote_pair_key": "1-2",
            "quote_unique": True,
            "odds_join_started_at": "2026-08-02T14:56:51+09:00",
            "quote_request_started_at": "2026-08-02T14:56:31+09:00",
            "t3_quote_source_event_time": "2026-08-02T14:56:45+09:00",
            "t3_quote_received_at": "2026-08-02T14:56:50+09:00",
            "t3_quote_selected_asof_time": "2026-08-02T14:56:55+09:00",
            "feed_heartbeat_source_event_time": "2026-08-02T14:56:48+09:00",
            "feed_heartbeat_received_at": "2026-08-02T14:56:52+09:00",
            "feed_sequence_id": f"feed-{race_id}",
            "market_status": "OPEN",
            "t3_wide_odds_low": 4.0,
            "t3_wide_odds_high": 4.6,
        }
        record.update(overrides)
        return record

    def write_source_population(
        self,
        root: Path,
        *,
        confidence_fail_races: set[str] | None = None,
        only_one: bool = False,
    ) -> tuple[Path, list[dict]]:
        confidence_fail_races = confidence_fail_races or set()
        sources = []
        all_rows = []
        failure_positions = {
            (0, 1), (0, 2), (0, 3),
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 1), (2, 2), (2, 3), (2, 4),
        }
        card_range = range(1) if only_one else range(3)
        for card_index in card_range:
            card_dir = root / f"card-{card_index}"
            packets_dir = card_dir / "packets"
            packets_dir.mkdir(parents=True)
            ledger_path = card_dir / "candidate_freeze_ledger.jsonl"
            acknowledgements = []
            race_range = (4,) if only_one else range(1, 13)
            for race_no in race_range:
                race_id = self.race_id(card_index, race_no)
                ready = (card_index, race_no) not in failure_positions
                candidate = self.candidate(
                    race_id,
                    race_no,
                    ready=ready,
                    confidence=race_id not in confidence_fail_races,
                )
                relative_packet = Path("packets") / f"{race_id}.candidate_freeze.json"
                candidate_path = card_dir / relative_packet
                candidate_path.write_text(
                    CAPTURE.canonical_json(candidate) + "\n", encoding="utf-8"
                )
                packet_sha = CAPTURE.file_sha256(candidate_path)
                acknowledgement = self.acknowledgement(
                    candidate, packet_sha, relative_packet.as_posix()
                )
                acknowledgements.append(acknowledgement)
                all_rows.append(
                    {
                        "card_id": f"card-{card_index}",
                        "candidate": candidate,
                        "acknowledgement": acknowledgement,
                        "candidate_packet_sha256": packet_sha,
                    }
                )
            ledger_path.write_text(
                "".join(CAPTURE.canonical_json(row) + "\n" for row in acknowledgements),
                encoding="utf-8",
            )
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
        return manifest_path, all_rows

    def write_capture(
        self,
        capture_dir: Path,
        source: dict,
        *,
        quote_overrides: dict | None = None,
        universe_changed: bool = False,
        scratch: bool = False,
        schedule_ok: bool = True,
        captured_at: str = "2026-08-02T14:57:00+09:00",
    ) -> dict:
        candidate = source["candidate"]
        packet = CAPTURE.build_capture_packet(
            candidate=candidate,
            acknowledgement=source["acknowledgement"],
            candidate_packet_sha256=source["candidate_packet_sha256"],
            schedule_record=self.schedule(candidate["race_id"], contract_ok=schedule_ok),
            universe_observation=self.universe(
                candidate["race_id"], changed=universe_changed, scratch=scratch
            ),
            quote_observation=self.quote(
                candidate["race_id"], **(quote_overrides or {})
            ),
            captured_at=captured_at,
            data_class="synthetic",
            config=self.config,
        )
        capture_dir.mkdir(parents=True, exist_ok=True)
        path = capture_dir / f"{candidate['race_id']}.strict_t3_capture.json"
        path.write_text(CAPTURE.canonical_json(packet) + "\n", encoding="utf-8")
        return packet

    def evaluate_one(
        self,
        *,
        confidence: bool = True,
        quote_overrides: dict | None = None,
        universe_changed: bool = False,
        scratch: bool = False,
        schedule_ok: bool = True,
        committed_at: str = "2026-08-02T14:57:05+09:00",
        captured_at: str = "2026-08-02T14:57:00+09:00",
    ) -> dict:
        race_id = self.race_id(0, 4)
        candidate = self.candidate(race_id, 4, confidence=confidence)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidate.json"
            candidate_path.write_text(
                CAPTURE.canonical_json(candidate) + "\n", encoding="utf-8"
            )
            packet_sha = CAPTURE.file_sha256(candidate_path)
            acknowledgement = self.acknowledgement(candidate, packet_sha, "candidate.json")
            source = {
                "candidate": candidate,
                "acknowledgement": acknowledgement,
                "candidate_packet_sha256": packet_sha,
            }
            packet = self.write_capture(
                root / "captures",
                source,
                quote_overrides=quote_overrides,
                universe_changed=universe_changed,
                scratch=scratch,
                schedule_ok=schedule_ok,
                captured_at=captured_at,
            )
            return RUNNER.evaluate_ready_capture(
                candidate=candidate,
                acknowledgement=acknowledgement,
                candidate_packet_sha256=packet_sha,
                capture_packet=packet,
                card_id="card-0",
                config=self.config,
                committed_at=CAPTURE.parse_time(committed_at, self.config["timezone"]),
            )

    def test_complete_population_retains_all_36_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, rows = self.write_source_population(root)
            capture_dir = root / "captures"
            for row in rows:
                if row["candidate"]["record_status"] == "CANDIDATE_READY":
                    self.write_capture(capture_dir, row)
            summary = RUNNER.run_shadow_decisions(
                source_manifest_path=manifest_path,
                capture_packet_dir=capture_dir,
                decision_ledger_path=root / "decisions.jsonl",
                summary_path=root / "summary.json",
                config=self.config,
                now=datetime(2026, 8, 2, 14, 57, 5, tzinfo=ZoneInfo("Asia/Tokyo")),
            )
            decisions = RUNNER.read_decision_jsonl(root / "decisions.jsonl")
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["recorded_target_rows"], 36)
            self.assertEqual(summary["source_failure_rows_preserved"], 11)
            self.assertEqual(summary["source_failure_rows_entering_quote_evaluation"], 0)
            self.assertEqual(summary["quote_evaluation_rows"], 25)
            self.assertEqual(len(decisions), 36)
            failures = [
                row for row in decisions if row["decision_reason"] == "NO_BET_SOURCE_NOT_READY"
            ]
            self.assertTrue(all(row["candidate_pair_key"] == "" for row in failures))

    def test_captured_quote_is_not_committed_before_strict_t3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, rows = self.write_source_population(root, only_one=True)
            capture_dir = root / "captures"
            self.write_capture(capture_dir, rows[0])
            kwargs = {
                "source_manifest_path": manifest_path,
                "capture_packet_dir": capture_dir,
                "decision_ledger_path": root / "decisions.jsonl",
                "summary_path": root / "summary.json",
                "config": self.config,
                "enforce_expected_counts": False,
            }
            before = RUNNER.run_shadow_decisions(
                **kwargs,
                now=datetime(2026, 8, 2, 14, 56, 59, tzinfo=ZoneInfo("Asia/Tokyo")),
            )
            self.assertEqual(before["status"], "IN_PROGRESS")
            self.assertEqual(before["recorded_target_rows"], 0)
            at_cutoff = RUNNER.run_shadow_decisions(
                **kwargs,
                now=datetime(2026, 8, 2, 14, 57, 0, tzinfo=ZoneInfo("Asia/Tokyo")),
            )
            self.assertEqual(at_cutoff["status"], "PASS")
            self.assertEqual(at_cutoff["recorded_target_rows"], 1)

    def test_live_card_worker_captures_then_commits_at_t3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _rows = self.write_source_population(root, only_one=True)
            venue_cname = "pw15orl012026010420260802/ABCDEF"
            detail_cname = "pw155abcS301202601040420260802Z/ABCDEF"
            pages = {
                "pw15oli00/6D": (
                    "<a onclick=\"doAction('/JRADB/accessO.html', "
                    f"'{venue_cname}')\">venue</a>"
                ).encode("cp932"),
                venue_cname: (
                    "<a onclick=\"doAction('/JRADB/accessO.html', "
                    f"'{detail_cname}')\">wide</a>"
                ).encode("cp932"),
                detail_cname: """
                    <p>発走時刻 15:00</p><p>オッズ 14:56 現在</p>
                    <table class="wide"><caption>1</caption><tbody>
                      <tr><th>2</th><td><span class="min">4.0</span><span class="max">4.6</span></td></tr>
                      <tr><th>3</th><td><span class="min">5.0</span><span class="max">5.8</span></td></tr>
                    </tbody></table>
                    <table class="wide"><caption>2</caption><tbody>
                      <tr><th>3</th><td><span class="min">6.0</span><span class="max">6.8</span></td></tr>
                    </tbody></table>
                """.encode("cp932"),
            }

            current = [
                datetime(2026, 8, 2, 14, 56, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
            ]

            def clock() -> datetime:
                observed = current[0]
                current[0] = observed + timedelta(milliseconds=100)
                return observed

            def sleeper(seconds: float) -> None:
                current[0] += timedelta(seconds=seconds)

            summary = RUNNER.run_live_jra_card(
                source_manifest_path=manifest_path,
                capture_packet_dir=root / "captures",
                raw_html_dir=root / "raw",
                decision_ledger_path=root / "decisions.jsonl",
                summary_path=root / "summary.json",
                config=self.config,
                fetch_cname=lambda cname: pages[cname],
                clock=clock,
                sleeper=sleeper,
                enforce_expected_counts=False,
            )
            decisions = RUNNER.read_decision_jsonl(root / "decisions.jsonl")
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0]["shadow_action"], "PAPER_READY")
            self.assertEqual(decisions[0]["candidate_pair_key"], "1-2")
            self.assertFalse(decisions[0]["formal_buy"])
            self.assertFalse(decisions[0]["send_order"])

    def test_confidence_fail_is_measurement_only_no_bet(self) -> None:
        row = self.evaluate_one(confidence=False)
        self.assertEqual(row["shadow_action"], "NO_BET")
        self.assertEqual(row["decision_reason"], "NO_BET_CONFIDENCE_GATE_FAILED")
        self.assertTrue(row["quote_evaluation_entered"])
        self.assertTrue(row["candidate_pair_t3_quote_valid"])
        self.assertTrue(row["measurement_only"])

    def test_source_received_and_selected_after_cutoff_fail(self) -> None:
        fields = (
            "t3_quote_source_event_time",
            "t3_quote_received_at",
            "t3_quote_selected_asof_time",
        )
        for field in fields:
            with self.subTest(field=field):
                overrides = {field: "2026-08-02T14:57:01+09:00"}
                if field == "t3_quote_source_event_time":
                    overrides["t3_quote_received_at"] = "2026-08-02T14:57:01+09:00"
                    overrides["t3_quote_selected_asof_time"] = "2026-08-02T14:57:01+09:00"
                elif field == "t3_quote_received_at":
                    overrides["t3_quote_selected_asof_time"] = "2026-08-02T14:57:01+09:00"
                row = self.evaluate_one(
                    quote_overrides=overrides,
                    captured_at="2026-08-02T14:57:02+09:00",
                )
                self.assertEqual(row["shadow_action"], "NO_BET")
                self.assertEqual(row["decision_reason"], "NO_BET_T3_QUOTE_ASOF_VIOLATION")

    def test_pair_mismatch_and_ambiguous_quote_fail(self) -> None:
        mismatch = self.evaluate_one(quote_overrides={"quote_pair_key": "1-3"})
        ambiguous = self.evaluate_one(quote_overrides={"quote_unique": False})
        self.assertEqual(
            mismatch["decision_reason"], "NO_BET_EXACT_CANDIDATE_QUOTE_INVALID"
        )
        self.assertEqual(
            ambiguous["decision_reason"], "NO_BET_EXACT_CANDIDATE_QUOTE_AMBIGUOUS"
        )

    def test_schedule_universe_and_scratch_fail_closed(self) -> None:
        self.assertEqual(
            self.evaluate_one(schedule_ok=False)["decision_reason"],
            "NO_BET_SCHEDULE_CONTRACT_FAILURE",
        )
        self.assertEqual(
            self.evaluate_one(universe_changed=True)["decision_reason"],
            "NO_BET_STARTER_UNIVERSE_CHANGED",
        )
        self.assertEqual(
            self.evaluate_one(scratch=True)["decision_reason"],
            "NO_BET_SCRATCH_KNOWN_BY_T3",
        )

    def test_decision_deadline_missed_fails_closed(self) -> None:
        row = self.evaluate_one(committed_at="2026-08-02T14:57:11+09:00")
        self.assertEqual(row["decision_reason"], "NO_BET_DECISION_DEADLINE_MISSED")

    def test_value_threshold_is_fixed_at_1_5(self) -> None:
        row = self.evaluate_one(quote_overrides={"t3_wide_odds_low": 3.7})
        self.assertAlmostEqual(row["research_expected_return_low"], 1.48)
        self.assertEqual(row["decision_reason"], "NO_BET_VALUE_BELOW_THRESHOLD")

    def test_second_run_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, rows = self.write_source_population(root)
            capture_dir = root / "captures"
            for row in rows:
                if row["candidate"]["record_status"] == "CANDIDATE_READY":
                    self.write_capture(capture_dir, row)
            kwargs = {
                "source_manifest_path": manifest_path,
                "capture_packet_dir": capture_dir,
                "decision_ledger_path": root / "decisions.jsonl",
                "summary_path": root / "summary.json",
                "config": self.config,
                "now": datetime(2026, 8, 2, 14, 57, 5, tzinfo=ZoneInfo("Asia/Tokyo")),
            }
            first = RUNNER.run_shadow_decisions(**kwargs)
            second = RUNNER.run_shadow_decisions(**kwargs)
            self.assertEqual(first, second)
            self.assertEqual(len(RUNNER.read_decision_jsonl(root / "decisions.jsonl")), 36)

    def test_missing_capture_is_pending_before_cutoff_and_no_bet_after(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _rows = self.write_source_population(root, only_one=True)
            before = RUNNER.run_shadow_decisions(
                source_manifest_path=manifest_path,
                capture_packet_dir=root / "captures",
                decision_ledger_path=root / "decisions.jsonl",
                summary_path=root / "summary.json",
                config=self.config,
                now=datetime(2026, 8, 2, 14, 56, 0, tzinfo=ZoneInfo("Asia/Tokyo")),
                enforce_expected_counts=False,
            )
            self.assertEqual(before["status"], "IN_PROGRESS")
            after = RUNNER.run_shadow_decisions(
                source_manifest_path=manifest_path,
                capture_packet_dir=root / "captures",
                decision_ledger_path=root / "decisions.jsonl",
                summary_path=root / "summary.json",
                config=self.config,
                now=datetime(2026, 8, 2, 14, 57, 1, tzinfo=ZoneInfo("Asia/Tokyo")),
                enforce_expected_counts=False,
            )
            decision = RUNNER.read_decision_jsonl(root / "decisions.jsonl")[0]
            self.assertEqual(after["status"], "PASS")
            self.assertEqual(
                decision["decision_reason"], "NO_BET_T3_QUOTE_NOT_AVAILABLE"
            )

    def test_quote_attempt_ledger_is_append_only_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempts.jsonl"
            candidate = self.candidate(self.race_id(0, 4), 4)
            schedule = self.schedule(candidate["race_id"])
            first = RUNNER._record_quote_attempt(
                path=path,
                candidate=candidate,
                schedule_record=schedule,
                attempt_index=1,
                attempted_at=datetime.fromisoformat("2026-08-02T14:56:30+09:00"),
                decision_reason="NO_BET_T3_QUOTE_STALE",
                accepted=False,
                capture_packet=None,
                config=self.config,
            )
            RUNNER._record_quote_attempt(
                path=path,
                candidate=candidate,
                schedule_record=schedule,
                attempt_index=1,
                attempted_at=datetime.fromisoformat("2026-08-02T14:56:30+09:00"),
                decision_reason="NO_BET_T3_QUOTE_STALE",
                accepted=False,
                capture_packet=None,
                config=self.config,
            )
            rows = RUNNER._read_jsonl(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0], first)

    def test_exp012_live_worker_retries_stale_quote_and_accepts_fresh(self) -> None:
        config = CAPTURE.load_config(
            ROOT / "config" / "race_day_contract_hardening_exp012.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            card_dir = root / "card"
            packets_dir = card_dir / "packets"
            packets_dir.mkdir(parents=True)
            candidate = self.candidate(self.race_id(0, 4), 4)
            candidate["experiment_id"] = config["source_experiment_id"]
            candidate["candidate_freeze_record_hash"] = CAPTURE.candidate_record_digest(
                candidate
            )
            relative_packet = Path("packets") / f"{candidate['race_id']}.candidate_freeze.json"
            candidate_path = card_dir / relative_packet
            candidate_path.write_text(
                CAPTURE.canonical_json(candidate) + "\n", encoding="utf-8"
            )
            packet_sha = CAPTURE.file_sha256(candidate_path)
            acknowledgement = self.acknowledgement(
                candidate, packet_sha, relative_packet.as_posix()
            )
            acknowledgement["experiment_id"] = config["source_experiment_id"]
            ledger_path = card_dir / "candidate_freeze_ledger.jsonl"
            ledger_path.write_text(
                CAPTURE.canonical_json(acknowledgement) + "\n", encoding="utf-8"
            )
            manifest_path = root / "source_manifest.json"
            manifest_path.write_text(
                CAPTURE.canonical_json(
                    {
                        "schema_version": 1,
                        "experiment_id": config["experiment_id"],
                        "source_experiment_id": config["source_experiment_id"],
                        "data_class": "synthetic",
                        "sources": [
                            {
                                "card_id": "synthetic-card",
                                "candidate_ledger_jsonl": str(ledger_path),
                                "candidate_ledger_sha256": CAPTURE.file_sha256(ledger_path),
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            venue_cname = "pw15orl012026010420260802/ABCDEF"
            detail_cname = "pw155abcS301202601040420260802Z/ABCDEF"
            top_page = (
                "<a onclick=\"doAction('/JRADB/accessO.html', "
                f"'{venue_cname}')\">venue</a>"
            ).encode("cp932")
            venue_page = (
                "<a onclick=\"doAction('/JRADB/accessO.html', "
                f"'{detail_cname}')\">wide</a>"
            ).encode("cp932")
            detail_reads = 0
            odds_fetch_times: list[datetime] = []
            current = [
                datetime(2026, 8, 2, 14, 56, 29, 500000, tzinfo=ZoneInfo("Asia/Tokyo"))
            ]

            def clock() -> datetime:
                observed = current[0]
                current[0] = observed + timedelta(milliseconds=50)
                return observed

            def sleeper(seconds: float) -> None:
                current[0] += timedelta(seconds=seconds)

            def fetch_cname(cname: str) -> bytes:
                nonlocal detail_reads
                odds_fetch_times.append(current[0])
                if cname == "pw15oli00/6D":
                    return top_page
                if cname == venue_cname:
                    return venue_page
                if cname != detail_cname:
                    raise KeyError(cname)
                detail_reads += 1
                source_clock = "14:56" if detail_reads == 1 else "14:57"
                return f"""
                    <p>発走時刻 15:01</p><p>オッズ {source_clock} 現在</p>
                    <table class="wide"><caption>1</caption><tbody>
                      <tr><th>2</th><td><span class="min">4.0</span><span class="max">4.6</span></td></tr>
                      <tr><th>3</th><td><span class="min">5.0</span><span class="max">5.8</span></td></tr>
                    </tbody></table>
                    <table class="wide"><caption>2</caption><tbody>
                      <tr><th>3</th><td><span class="min">6.0</span><span class="max">6.8</span></td></tr>
                    </tbody></table>
                """.encode("cp932")

            schedule_calls: list[datetime] = []

            def fetch_schedule(_candidate: dict) -> dict:
                schedule_calls.append(current[0])
                return {
                    "document": "<p>発走時刻 15:01</p>",
                    "source_event_time": "2026-08-02T14:56:20+09:00",
                    "received_at": "2026-08-02T14:56:29+09:00",
                    "locked_at": "2026-08-02T14:56:30+09:00",
                    "source_reference": "synthetic-schedule-only",
                }

            summary = RUNNER.run_live_jra_card(
                source_manifest_path=manifest_path,
                capture_packet_dir=root / "captures",
                raw_html_dir=root / "raw",
                decision_ledger_path=root / "decisions.jsonl",
                summary_path=root / "summary.json",
                config=config,
                fetch_cname=fetch_cname,
                fetch_schedule_document=fetch_schedule,
                clock=clock,
                sleeper=sleeper,
                enforce_expected_counts=False,
            )
            attempts = RUNNER._read_jsonl(
                root / "captures" / f"{candidate['race_id']}.quote_attempts.jsonl"
            )
            decisions = RUNNER.read_decision_jsonl(root / "decisions.jsonl")
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(schedule_calls), 1)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["decision_reason"], "NO_BET_T3_QUOTE_STALE")
            self.assertFalse(attempts[0]["accepted"])
            self.assertTrue(attempts[1]["accepted"])
            self.assertTrue(
                all(
                    observed
                    >= datetime.fromisoformat("2026-08-02T14:57:30+09:00")
                    for observed in odds_fetch_times
                )
            )
            self.assertEqual(decisions[0]["shadow_action"], "PAPER_READY")
            self.assertEqual(decisions[0]["candidate_pair_key"], "1-2")
            self.assertFalse(decisions[0]["formal_buy"])
            self.assertFalse(decisions[0]["send_order"])


if __name__ == "__main__":
    unittest.main()
