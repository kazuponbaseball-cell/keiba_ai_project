from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "schedule_only_provider_exp013.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCHEDULE = load_module(
    "schedule_only_provider_v1",
    ROOT / "scripts" / "research" / "schedule_only_provider_v1.py",
)
CAPTURE = load_module(
    "exp013_capture",
    ROOT / "scripts" / "research" / "build_strict_t3_capture_packet_v1.py",
)
RUNNER = load_module(
    "exp013_runner",
    ROOT / "scripts" / "research" / "run_strict_t3_shadow_decision_v1.py",
)


class ScheduleOnlyProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "schedule_observations.jsonl"
        self.config = CAPTURE.load_config(CONFIG_PATH)
        self.candidate = {
            "race_id": "202604020401",
            "scheduled_post_time_asof": "2026-08-02T15:00:00+09:00",
            "candidate_freeze_record_hash": "c" * 64,
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def record(
        self,
        *,
        observation_id: str = "obs-1",
        version: str = "v1",
        post: str = "2026-08-02T15:01:00+09:00",
        source: str = "2026-08-02T14:56:20+09:00",
        received: str = "2026-08-02T14:56:29+09:00",
    ) -> dict:
        return {
            "schema_version": 1,
            "schedule_provider_id": "file_backed_schedule_v1",
            "schedule_observation_id": observation_id,
            "race_id": self.candidate["race_id"],
            "scheduled_post_time": post,
            "source_event_time": source,
            "received_at": received,
            "source_reference": f"synthetic:{observation_id}:{version}",
            "source_payload_sha256": hashlib.sha256(
                f"{observation_id}:{version}:{post}".encode("ascii")
            ).hexdigest(),
            "schedule_version": version,
        }

    def write(self, *records: dict) -> None:
        self.path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )

    def provider(self, now: str):
        return RUNNER.build_file_schedule_provider(
            self.path,
            config=self.config,
            clock=lambda: datetime.fromisoformat(now),
        )

    def lock(self, now: str) -> dict:
        provider = self.provider(now)
        return RUNNER._schedule_observation(
            candidate=self.candidate,
            fetch_schedule_document=provider,
            clock=lambda: datetime.fromisoformat(now),
            config=self.config,
        )

    def test_valid_unique_schedule(self) -> None:
        self.write(self.record())
        lock = self.lock("2026-08-02T14:56:30+09:00")
        CAPTURE.verify_schedule_lock_record(
            lock, candidate=self.candidate, config=self.config
        )
        self.assertTrue(lock["schedule_contract_ok"])
        self.assertEqual(lock["schedule_provider_id"], "file_backed_schedule_v1")
        self.assertEqual(lock["schedule_observation_id"], "obs-1")
        self.assertFalse(lock.get("candidate_uses_odds", False))
        lock_path = self.root / "schedule_lock.json"
        CAPTURE.write_json_atomic_immutable(lock_path, lock)
        persisted = CAPTURE.load_json_object(lock_path)
        CAPTURE.verify_schedule_lock_record(
            persisted, candidate=self.candidate, config=self.config
        )
        args = RUNNER.build_parser().parse_args(
            [
                "--source-manifest-json",
                "source.json",
                "--capture-packet-dir",
                "captures",
                "--decision-ledger-jsonl",
                "decisions.jsonl",
                "--summary-json",
                "summary.json",
                "--schedule-observations-jsonl",
                str(self.path),
            ]
        )
        self.assertEqual(args.schedule_observations_jsonl, self.path)
        source = (ROOT / "scripts" / "research" / "schedule_only_provider_v1.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("fetch_jra_official_odds", source)
        self.assertNotIn("import requests", source)

    def test_odd_minute_post_preserved(self) -> None:
        self.write(self.record(post="2026-08-02T15:01:00+09:00"))
        lock = self.lock("2026-08-02T14:56:30+09:00")
        self.assertEqual(
            lock["scheduled_post_time_used"], "2026-08-02T15:01:00.000+09:00"
        )
        self.assertEqual(
            CAPTURE.quote_attempt_times(lock, self.config)[0].isoformat(),
            "2026-08-02T14:57:30+09:00",
        )

    def test_latest_received_version_selected(self) -> None:
        first = self.record(
            observation_id="obs-1",
            version="v1",
            post="2026-08-02T15:00:00+09:00",
            source="2026-08-02T14:55:20+09:00",
            received="2026-08-02T14:55:30+09:00",
        )
        latest = self.record(observation_id="obs-2", version="v2")
        self.write(first, latest)
        selected = self.provider("2026-08-02T14:56:30+09:00")(self.candidate)
        self.assertEqual(selected["schedule_observation_id"], "obs-2")
        self.assertEqual(selected["provider_schedule_version"], "v2")

    def test_duplicate_identical_version_is_idempotent(self) -> None:
        record = self.record()
        other_race = self.record(observation_id="obs-other", version="v1")
        other_race["race_id"] = "202604020402"
        self.write(record, record, other_race)
        selected = self.provider("2026-08-02T14:56:30+09:00")(self.candidate)
        self.assertEqual(selected["schedule_observation_id"], "obs-1")

    def test_conflicting_duplicate_version_fails(self) -> None:
        first = self.record(observation_id="obs-1", version="v1")
        conflict = self.record(
            observation_id="obs-2",
            version="v1",
            post="2026-08-02T15:02:00+09:00",
        )
        self.write(first, conflict)
        with self.assertRaisesRegex(
            SCHEDULE.ScheduleProviderContractError,
            "conflicting duplicate schedule_version",
        ):
            self.provider("2026-08-02T14:56:30+09:00")(self.candidate)
        self.write(
            self.record(observation_id="obs-1", version="v1"),
            self.record(observation_id="obs-2", version="v2"),
        )
        with self.assertRaisesRegex(
            SCHEDULE.ScheduleProviderContractError,
            "ambiguous schedule versions",
        ):
            self.provider("2026-08-02T14:56:30+09:00")(self.candidate)

    def test_missing_source_event_time_fails(self) -> None:
        record = self.record()
        del record["source_event_time"]
        self.write(record)
        with self.assertRaisesRegex(
            SCHEDULE.ScheduleProviderContractError, "missing field"
        ):
            self.provider("2026-08-02T14:56:30+09:00")(self.candidate)

    def test_received_before_source_event_fails(self) -> None:
        self.write(
            self.record(
                source="2026-08-02T14:56:29+09:00",
                received="2026-08-02T14:56:20+09:00",
            )
        )
        with self.assertRaisesRegex(
            SCHEDULE.ScheduleProviderContractError, "after received time"
        ):
            self.provider("2026-08-02T14:56:30+09:00")(self.candidate)

    def test_update_received_after_new_cutoff_fails_closed(self) -> None:
        self.write(
            self.record(
                post="2026-08-02T14:59:00+09:00",
                received="2026-08-02T14:56:30+09:00",
            )
        )
        lock = self.lock("2026-08-02T14:56:31+09:00")
        self.assertFalse(lock["schedule_contract_ok"])
        self.assertEqual(
            lock["schedule_failure_reason"], "NO_BET_SCHEDULE_CONTRACT_FAILURE"
        )

    def test_forbidden_odds_field_fails(self) -> None:
        record = self.record()
        record["odds"] = 3.5
        self.write(record)
        with self.assertRaisesRegex(
            SCHEDULE.ScheduleProviderContractError, "forbidden field"
        ):
            self.provider("2026-08-02T14:56:30+09:00")(self.candidate)

    def test_provider_unavailable_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            SCHEDULE.ScheduleProviderUnavailable, "unavailable"
        ):
            self.provider("2026-08-02T14:56:30+09:00")(self.candidate)


if __name__ == "__main__":
    unittest.main()
