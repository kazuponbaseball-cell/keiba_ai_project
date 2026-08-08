from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from scripts.research.import_target_multicard_entry_v1 import (
    RaceMetadata,
    build_entry_rows,
    canonical_candidate_race_id,
    import_multicard,
    load_direct_history,
)

from scripts.research.target_direct_history_v1 import (
    BASELINE_TRACK_CODE,
    PREVIOUS_RACE_COLUMNS,
    apprentice_symbol,
    average_section_time,
    build_all_history_payloads,
    build_race_payloads,
    canonical_horse_id,
    canonical_race_id,
    finish_time_seconds,
    pace_change_index,
    parse_ra_record,
    parse_se_record,
    pci3,
    select_authoritative_history,
)


def _put(raw: bytearray, start: int, length: int, value: str, *, encoding: str = "ascii") -> None:
    encoded = value.encode(encoding)
    if len(encoded) > length:
        raise ValueError((start, length, value))
    raw[start : start + length] = encoded.ljust(length, b" ")


def synthetic_ra(
    *,
    key: str = "2026071105010101",
    distance: int = 1600,
    track_code: str = "12",
    registered: int = 3,
    starters: int = 3,
    race_final3f_tenths: int = 338,
) -> bytes:
    raw = bytearray(b" " * 1270)
    _put(raw, 0, 2, "RA")
    _put(raw, 11, 16, key)
    _put(raw, 697, 4, f"{distance:04d}")
    _put(raw, 705, 2, track_code)
    _put(raw, 881, 2, f"{registered:02d}")
    _put(raw, 883, 2, f"{starters:02d}")
    _put(raw, 888, 1, "1")
    _put(raw, 889, 1, "2")
    for index, lap in enumerate((121, 115, 118, 116, 112, 111, 113, 114)):
        _put(raw, 890 + index * 3, 3, str(lap))
    _put(raw, 975, 3, str(race_final3f_tenths))
    return bytes(raw)


def synthetic_se(
    *,
    key: str = "2026071105010101",
    horse_no: int = 1,
    horse_id: str = "2022100001",
    horse_name: str = "TEST HORSE",
    finish_rank: int = 1,
    finish_time: str = "1320",
    final3f_tenths: int = 338,
    time_diff_tenths: int = 0,
    popularity: int = 2,
    apprentice_code: str = "0",
    abnormal_code: str = "0",
) -> bytes:
    raw = bytearray(b" " * 553)
    _put(raw, 0, 2, "SE")
    _put(raw, 11, 16, key)
    _put(raw, 28, 2, f"{horse_no:02d}")
    _put(raw, 30, 10, horse_id)
    _put(raw, 40, 36, horse_name, encoding="cp932")
    _put(raw, 288, 3, "550")
    _put(raw, 296, 5, "01234")
    _put(raw, 322, 1, apprentice_code)
    _put(raw, 324, 3, "470")
    _put(raw, 327, 1, "+")
    _put(raw, 328, 3, "004")
    _put(raw, 331, 1, abnormal_code)
    _put(raw, 334, 2, f"{finish_rank:02d}")
    _put(raw, 338, 4, finish_time)
    _put(raw, 363, 2, f"{popularity:02d}")
    _put(raw, 390, 3, str(final3f_tenths))
    _put(raw, 531, 4, f"{time_diff_tenths:04d}")
    return bytes(raw)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_current_ra(*, key: str, post_hhmm: str) -> bytes:
    raw = bytearray(synthetic_ra(key=key, registered=1, starters=1))
    _put(raw, 0, 3, "RA2")
    _put(raw, 614, 1, "")
    _put(raw, 616, 2, "13")
    _put(raw, 634, 3, "703")
    _put(raw, 709, 2, "A")
    _put(raw, 873, 4, post_hhmm)
    return bytes(raw)


def synthetic_current_du(
    *,
    race_date: str,
    race_part: str,
    horse_id_8: str,
    horse_no: int = 1,
) -> bytes:
    raw = bytearray(b" " * 160)
    _put(raw, 0, 3, "SE1")
    _put(raw, 3, 8, race_date)
    _put(raw, 11, 8, race_date)
    _put(raw, 19, 8, race_part)
    _put(raw, 27, 1, "1")
    _put(raw, 28, 2, f"{horse_no:02d}")
    _put(raw, 32, 8, horse_id_8)
    _put(raw, 40, 36, f"CURRENT {horse_id_8}", encoding="cp932")
    _put(raw, 82, 2, "03")
    _put(raw, 86, 4, "1234")
    _put(raw, 90, 10, "TRAINER", encoding="cp932")
    _put(raw, 104, 5, "55000")
    _put(raw, 112, 5, "01234")
    _put(raw, 122, 10, "JOCKEY", encoding="cp932")
    return bytes(raw)


def _direct_entry_fixture() -> tuple[
    pd.DataFrame,
    dict[str, object],
    dict[str, RaceMetadata],
    pd.DataFrame,
]:
    runners: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    metadata: dict[str, RaceMetadata] = {}
    histories: list[dict[str, object]] = []
    for race_no in range(1, 13):
        key = f"20260809010105{race_no:02d}"
        race_id = canonical_candidate_race_id(key)
        records.append(
            {
                "target_race_key": key,
                "race_id": race_id,
                "race_no": race_no,
                "runner_count": 2,
                "scheduled_post_time": f"2026-08-09T{8 + race_no:02d}:00:00+09:00",
                "race_domain": "flat",
            }
        )
        metadata[key] = RaceMetadata(
            venue="札幌",
            race_no=race_no,
            surface="芝",
            distance=1200,
            race_class="未勝利",
            race_name="synthetic",
            runner_name_match_count=2,
            target_race_key=key,
            data_category="2",
            race_type_code="13",
            condition_code="703",
            track_code="11",
            course_code="A",
            scheduled_post_hhmm=f"{8 + race_no:02d}00",
            registered_runner_count=2,
            record_hash="a" * 64,
        )
        for horse_no in (1, 2):
            horse_id = f"2026{race_no:02d}{horse_no:04d}"
            runners.append(
                {
                    "target_race_key": key,
                    "race_id": race_id,
                    "race_date": "20260809",
                    "venue_code": "01",
                    "venue": "札幌",
                    "meeting_no": 1,
                    "day_no": 5,
                    "race_no": race_no,
                    "frame_no": horse_no,
                    "horse_no": horse_no,
                    "horse_id": horse_id,
                    "horse_name": f"馬{race_no}-{horse_no}",
                    "age": 3,
                    "trainer_code": "1234",
                    "trainer_name": "調教師",
                    "assigned_weight_kg": 55.0,
                    "jockey_code": "01234",
                    "jockey_name": "騎手",
                }
            )
            histories.append(
                {
                    "target_race_key": key,
                    "horse_no": horse_no,
                    "horse_id": horse_id,
                    **{column: f"value-{race_no}-{horse_no}" for column in PREVIOUS_RACE_COLUMNS},
                    "previous_race_source_date": "20260802",
                    "previous_race_source_record_hash": hashlib.sha256(
                        f"{key}-{horse_no}".encode("ascii")
                    ).hexdigest(),
                    "previous_race_source_system": "JRA_VAN_JV_DATA_SU_SR",
                    "previous_race_source_file_hash": "b" * 64,
                    "previous_race_no_history_reason": "",
                    "previous_race_contract_ok": True,
                    "previous_pci_target_good_run_marker": None,
                }
            )
    return (
        pd.DataFrame(runners),
        {"records": records},
        metadata,
        pd.DataFrame(histories),
    )


class TargetDirectHistoryUnitTests(unittest.TestCase):
    def test_time_and_identity_conversions(self) -> None:
        self.assertEqual(116.3, finish_time_seconds("1563"))
        self.assertIsNone(finish_time_seconds("0000"))
        self.assertEqual("202605010101", canonical_race_id("2026071105010101"))
        self.assertEqual("2022100001", canonical_horse_id("22100001"))

    def test_finish_time_rejects_invalid_seconds(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid JV finish time"):
            finish_time_seconds("1600")

    def test_average_section_and_pci(self) -> None:
        self.assertEqual(11.5, average_section_time(92.0, 1600, 200))
        self.assertEqual(34.5, average_section_time(92.0, 1600, 600))
        self.assertEqual(53.3, pace_change_index(92.0, 33.8, 1600))

    def test_odd_distance_pci_uses_exact_distance(self) -> None:
        expected = round((((70.0 - 35.0) * 600 / 550) / 35.0 - 1) * 100 + 50, 1)
        self.assertEqual(expected, pace_change_index(70.0, 35.0, 1150))

    def test_pci3_rounds_each_horse_before_mean(self) -> None:
        self.assertEqual(47.03, pci3([46.95, 47.04, 47.05]))

    def test_apprentice_mapping_is_explicit(self) -> None:
        self.assertEqual("", apprentice_symbol("0"))
        self.assertEqual("☆", apprentice_symbol("1"))
        self.assertEqual("△", apprentice_symbol("2"))
        self.assertEqual("▲", apprentice_symbol("3"))
        self.assertEqual("★", apprentice_symbol("4"))
        self.assertEqual("◇", apprentice_symbol("9"))
        with self.assertRaisesRegex(ValueError, "unsupported apprentice code"):
            apprentice_symbol("8")

    def test_ra_and_se_offsets(self) -> None:
        race = parse_ra_record(synthetic_ra())
        runner = parse_se_record(synthetic_se(apprentice_code="3"))
        self.assertEqual("2026071105010101", race.target_race_key)
        self.assertEqual(1600, race.distance_m)
        self.assertEqual("芝", race.surface)
        self.assertEqual("1", race.going_code)
        self.assertEqual(33.8, race.race_final3f_seconds)
        self.assertEqual("2022100001", runner.horse_id)
        self.assertEqual(55.0, runner.assigned_weight_kg)
        self.assertEqual("▲", runner.apprentice_symbol)
        self.assertEqual(470, runner.body_weight_kg)
        self.assertEqual(4, runner.body_weight_delta_kg)
        self.assertEqual(92.0, runner.finish_time_seconds)

    def test_race_payload_contract_and_rank(self) -> None:
        race = parse_ra_record(synthetic_ra())
        runners = [
            parse_se_record(synthetic_se(horse_no=1, horse_id="2022100001", finish_rank=1)),
            parse_se_record(
                synthetic_se(
                    horse_no=2,
                    horse_id="2022100002",
                    finish_rank=2,
                    finish_time="1324",
                    final3f_tenths=336,
                    time_diff_tenths=4,
                )
            ),
            parse_se_record(
                synthetic_se(
                    horse_no=3,
                    horse_id="2022100003",
                    finish_rank=3,
                    finish_time="1330",
                    final3f_tenths=350,
                    time_diff_tenths=10,
                )
            ),
        ]
        payloads = build_race_payloads(race, runners)
        self.assertEqual(3, len(payloads))
        first = payloads["2022100001"]
        second = payloads["2022100002"]
        self.assertTrue(first["previous_race_contract_ok"])
        self.assertEqual("20260711", first["previous_race_source_date"])
        self.assertEqual("202605010101", first["前走レースID(新/馬番無)"])
        self.assertEqual(2, first["前走上り3F順"])
        self.assertEqual(1, second["前走上り3F順"])
        self.assertEqual("8", first["前走トラックコード"])
        self.assertEqual(24, sum(column in first for column in PREVIOUS_RACE_COLUMNS))

    def test_obstacle_history_keeps_flat_lap_metrics_blank(self) -> None:
        race = parse_ra_record(synthetic_ra(track_code="56", distance=3000))
        runner = parse_se_record(synthetic_se())
        payload = build_race_payloads(race, [runner])[runner.horse_id]
        for column in ("前走Ave-3F", "前走上り3F", "前走上り3F順", "前PCI", "前走PCI3", "前走RPCI"):
            self.assertIsNone(payload[column])
        self.assertEqual("2", BASELINE_TRACK_CODE["56"])

    def test_build_all_history_rejects_missing_ra(self) -> None:
        runner = parse_se_record(synthetic_se())
        with self.assertRaisesRegex(ValueError, "no matching RA"):
            build_all_history_payloads({}, [runner])

    def test_authoritative_selection_fails_closed_for_local_latest(self) -> None:
        race = parse_ra_record(synthetic_ra())
        runner = parse_se_record(synthetic_se())
        history = build_all_history_payloads({race.target_race_key: race}, [runner])
        current = [{"target_race_key": "2026080905010201", "horse_no": 4, "horse_id": runner.horse_id}]
        selected = select_authoritative_history(
            current,
            history,
            target_date="20260809",
            authoritative_latest_race_id={runner.horse_id: "202630080207"},
        )
        self.assertFalse(selected[0]["previous_race_contract_ok"])
        self.assertEqual("LATEST_PRIOR_RACE_NOT_IN_JRA_HISTORY", selected[0]["previous_race_no_history_reason"])
        self.assertTrue(all(selected[0][column] == "" for column in PREVIOUS_RACE_COLUMNS))

    def test_authoritative_selection_requires_pointer(self) -> None:
        race = parse_ra_record(synthetic_ra())
        runner = parse_se_record(synthetic_se())
        history = build_all_history_payloads({race.target_race_key: race}, [runner])
        current = [{"target_race_key": "2026080905010201", "horse_no": 4, "horse_id": runner.horse_id}]
        selected = select_authoritative_history(
            current,
            history,
            target_date="20260809",
            authoritative_latest_race_id={},
        )
        self.assertFalse(selected[0]["previous_race_contract_ok"])
        self.assertEqual("AUTHORITATIVE_LATEST_RACE_UNAVAILABLE", selected[0]["previous_race_no_history_reason"])

    def test_authoritative_selection_rejects_same_day_history(self) -> None:
        key = "2026080905010101"
        race = parse_ra_record(synthetic_ra(key=key))
        runner = parse_se_record(synthetic_se(key=key))
        history = build_all_history_payloads({key: race}, [runner])
        current = [{"target_race_key": "2026080905010201", "horse_no": 4, "horse_id": runner.horse_id}]
        selected = select_authoritative_history(
            current,
            history,
            target_date="20260809",
            authoritative_latest_race_id={runner.horse_id: race.race_id},
        )
        self.assertFalse(selected[0]["previous_race_contract_ok"])
        self.assertEqual("SAME_DAY_OR_FUTURE_HISTORY", selected[0]["previous_race_no_history_reason"])

    def test_authoritative_selection_accepts_exact_prior_identity(self) -> None:
        race = parse_ra_record(synthetic_ra())
        runner = parse_se_record(synthetic_se())
        history = build_all_history_payloads({race.target_race_key: race}, [runner])
        current = [{"target_race_key": "2026080905010201", "horse_no": 4, "horse_id": runner.horse_id}]
        selected = select_authoritative_history(
            current,
            history,
            target_date="20260809",
            authoritative_latest_race_id={runner.horse_id: race.race_id},
        )
        self.assertTrue(selected[0]["previous_race_contract_ok"])
        self.assertEqual(race.race_id, selected[0]["前走レースID(新/馬番無)"])
        self.assertEqual(4, selected[0]["horse_no"])

    def test_direct_entry_marks_entire_failed_history_race_unsupported(self) -> None:
        runners, manifest, metadata, histories = _direct_entry_fixture()
        failed = histories["target_race_key"].eq("2026080901010503") & histories[
            "horse_no"
        ].eq(2)
        histories.loc[failed, "previous_race_contract_ok"] = False
        histories.loc[
            failed, "previous_race_no_history_reason"
        ] = "LATEST_PRIOR_RACE_NOT_IN_JRA_HISTORY"
        frame = build_entry_rows(
            runners,
            manifest,
            metadata,
            direct_history=histories,
        )
        failed_race = frame["race_id"].eq(canonical_candidate_race_id("2026080901010503"))
        self.assertEqual(24, len(frame))
        self.assertTrue(frame.loc[failed_race, "race_domain"].eq("history_contract_failed").all())
        self.assertTrue(frame.loc[~failed_race, "race_domain"].eq("flat_turf").all())
        self.assertEqual(1, int((~frame["previous_race_contract_ok"].astype(bool)).sum()))
        self.assertEqual(24, sum(column in frame for column in PREVIOUS_RACE_COLUMNS))

    def test_direct_entry_requires_every_current_runner(self) -> None:
        runners, manifest, metadata, histories = _direct_entry_fixture()
        with self.assertRaisesRegex(ValueError, "direct history runner missing"):
            build_entry_rows(
                runners,
                manifest,
                metadata,
                direct_history=histories.iloc[:-1].copy(),
            )

    def test_direct_and_html_history_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_entry_rows(
                pd.DataFrame(),
                {},
                {},
                html_history=pd.DataFrame(),
                direct_history=pd.DataFrame(),
            )

    def test_direct_source_manifest_selects_hash_bound_prior_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ra_path = root / "SU.DAT"
            se_path = root / "SR.DAT"
            manifest_path = root / "history-manifest.json"
            ra_path.write_bytes(synthetic_ra() + b"\r\n")
            se_path.write_bytes(synthetic_se() + b"\r\n")
            current = pd.DataFrame(
                [
                    {
                        "target_race_key": "2026080901010501",
                        "horse_no": 7,
                        "horse_id": "2022100001",
                    }
                ]
            )
            manifest = {
                "schema_version": 1,
                "target_date": "20260809",
                "source_time_contract": "STRICTLY_BEFORE_TARGET_DATE",
                "ra_sources": [{"path": str(ra_path), "sha256": _sha256(ra_path)}],
                "se_sources": [{"path": str(se_path), "sha256": _sha256(se_path)}],
                "authoritative_latest_races": [
                    {"horse_id": "2022100001", "latest_race_id": "202605010101"}
                ],
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            frame, loaded = load_direct_history(
                source_manifest_path=manifest_path,
                source_manifest_sha256=_sha256(manifest_path),
                du=current,
                target_date="20260809",
            )
        self.assertEqual(1, len(frame))
        self.assertTrue(frame.iloc[0]["previous_race_contract_ok"])
        self.assertEqual("202605010101", frame.iloc[0]["前走レースID(新/馬番無)"])
        self.assertEqual(manifest, loaded)

    def test_direct_source_manifest_rejects_manifest_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "history-manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                load_direct_history(
                    source_manifest_path=manifest_path,
                    source_manifest_sha256="0" * 64,
                    du=pd.DataFrame(),
                    target_date="20260809",
                )

    def test_direct_source_manifest_accepts_audited_authority_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ra_path = root / "SU.DAT"
            se_path = root / "SR.DAT"
            manifest_path = root / "history-manifest.json"
            ra_path.write_bytes(synthetic_ra() + b"\r\n")
            se_path.write_bytes(synthetic_se() + b"\r\n")
            current = pd.DataFrame(
                [
                    {
                        "target_race_key": "2026080901010501",
                        "horse_no": 7,
                        "horse_id": "2022100001",
                    }
                ]
            )
            manifest = {
                "schema_version": 1,
                "target_date": "20260809",
                "source_time_contract": "STRICTLY_BEFORE_TARGET_DATE",
                "ra_sources": [{"path": str(ra_path), "sha256": _sha256(ra_path)}],
                "se_sources": [{"path": str(se_path), "sha256": _sha256(se_path)}],
                "authoritative_latest_races": [
                    {
                        "horse_id": "2022100001",
                        "authoritative_latest_race_id": "202605010101",
                        "authority_contract_ok": True,
                        "authority_failure_reason": "",
                        "target_date": "20260809",
                    }
                ],
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            frame, _ = load_direct_history(
                source_manifest_path=manifest_path,
                source_manifest_sha256=_sha256(manifest_path),
                du=current,
                target_date="20260809",
            )
        self.assertEqual(1, len(frame))
        self.assertTrue(frame.iloc[0]["previous_race_contract_ok"])

    def test_direct_source_manifest_rejects_failed_audited_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ra_path = root / "SU.DAT"
            se_path = root / "SR.DAT"
            manifest_path = root / "history-manifest.json"
            ra_path.write_bytes(synthetic_ra() + b"\r\n")
            se_path.write_bytes(synthetic_se() + b"\r\n")
            manifest = {
                "schema_version": 1,
                "target_date": "20260809",
                "source_time_contract": "STRICTLY_BEFORE_TARGET_DATE",
                "ra_sources": [{"path": str(ra_path), "sha256": _sha256(ra_path)}],
                "se_sources": [{"path": str(se_path), "sha256": _sha256(se_path)}],
                "authoritative_latest_races": [
                    {
                        "horse_id": "2022100001",
                        "authoritative_latest_race_id": "",
                        "authority_contract_ok": False,
                        "authority_failure_reason": "UNPROVABLE_LATEST_RACE",
                    }
                ],
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "failed authority row"):
                load_direct_history(
                    source_manifest_path=manifest_path,
                    source_manifest_sha256=_sha256(manifest_path),
                    du=pd.DataFrame(
                        [
                            {
                                "target_race_key": "2026080901010501",
                                "horse_no": 7,
                                "horse_id": "2022100001",
                            }
                        ]
                    ),
                    target_date="20260809",
                )

    def test_direct_multicard_import_is_synthetic_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current_ra_path = root / "DR.DAT"
            current_du_path = root / "DU.DAT"
            prior_ra_path = root / "SU.DAT"
            prior_se_path = root / "SR.DAT"
            source_manifest_path = root / "history-manifest.json"
            historical_path = root / "historical.csv"
            config_path = root / "config.json"
            output_root = root / "output"

            current_ra: list[bytes] = []
            current_du: list[bytes] = []
            target_cards: dict[str, list[dict[str, object]]] = {
                "sapporo": [],
                "niigata": [],
                "chukyo": [],
            }
            authority: list[dict[str, str]] = []
            prior_se: list[bytes] = []
            historical_rows: list[dict[str, str]] = []
            venues = (("sapporo", "01"), ("niigata", "04"), ("chukyo", "07"))
            runner_index = 0
            for slug, venue_code in venues:
                for race_no in range(1, 13):
                    runner_index += 1
                    race_part = f"{venue_code}0105{race_no:02d}"
                    key = "20260809" + race_part
                    race_id = canonical_candidate_race_id(key)
                    horse_id_8 = f"22{runner_index:06d}"
                    horse_id = "20" + horse_id_8
                    post_hhmm = f"{8 + race_no:02d}00"
                    current_ra.append(synthetic_current_ra(key=key, post_hhmm=post_hhmm))
                    current_du.append(
                        synthetic_current_du(
                            race_date="20260809",
                            race_part=race_part,
                            horse_id_8=horse_id_8,
                        )
                    )
                    target_cards[slug].append(
                        {
                            "target_race_key": key,
                            "race_id": race_id,
                            "race_no": race_no,
                            "runner_count": 1,
                            "scheduled_post_time": (
                                f"2026-08-09T{8 + race_no:02d}:00:00+09:00"
                            ),
                            "race_domain": "flat",
                        }
                    )
                    authority.append(
                        {"horse_id": horse_id, "latest_race_id": "202605010101"}
                    )
                    prior_se.append(
                        synthetic_se(
                            horse_no=runner_index,
                            horse_id=horse_id,
                            finish_rank=runner_index,
                            popularity=runner_index,
                        )
                    )
                    historical_rows.append({"horse_id": horse_id, "sex": "牡"})

            current_ra_path.write_bytes(b"\r\n".join(current_ra))
            current_du_path.write_bytes(b"\r\n".join(current_du))
            prior_ra_path.write_bytes(
                synthetic_ra(registered=36, starters=36) + b"\r\n"
            )
            prior_se_path.write_bytes(b"\r\n".join(prior_se))
            pd.DataFrame(historical_rows).to_csv(
                historical_path, index=False, encoding="cp932"
            )

            target_paths: dict[str, Path] = {}
            for slug, records in target_cards.items():
                path = root / f"{slug}-targets.json"
                path.write_text(
                    json.dumps({"records": records}, ensure_ascii=False),
                    encoding="utf-8",
                )
                target_paths[slug] = path

            source_manifest = {
                "schema_version": 1,
                "target_date": "20260809",
                "source_time_contract": "STRICTLY_BEFORE_TARGET_DATE",
                "ra_sources": [
                    {"path": str(prior_ra_path), "sha256": _sha256(prior_ra_path)}
                ],
                "se_sources": [
                    {"path": str(prior_se_path), "sha256": _sha256(prior_se_path)}
                ],
                "authoritative_latest_races": authority,
            }
            source_manifest_path.write_text(
                json.dumps(source_manifest, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            config = {
                "schema_version": 1,
                "experiment_id": "SYNTHETIC",
                "input_sources": {
                    "dr": {"path": str(current_ra_path), "sha256": _sha256(current_ra_path)},
                    "du": {"path": str(current_du_path), "sha256": _sha256(current_du_path)},
                    "direct_history_manifest": {
                        "path": str(source_manifest_path),
                        "sha256": _sha256(source_manifest_path),
                    },
                },
                "input_contract": {
                    "history_mode": "target_direct",
                    "expected_runner_rows": 36,
                    "expected_races": 36,
                    "required_previous_race_columns": 24,
                },
                "history": {"historical_csv": str(historical_path)},
                "cards": [
                    {
                        "slug": slug,
                        "target_manifest": str(target_paths[slug]),
                    }
                    for slug in ("sapporo", "niigata", "chukyo")
                ],
                "safety": {
                    "formal_buy": False,
                    "send_order": False,
                    "stake": 0,
                },
            }
            config_path.write_text(
                json.dumps(config, ensure_ascii=False), encoding="utf-8"
            )
            summary = import_multicard(config_path, output_root)

            config["input_sources"]["direct_history_manifest"] = {
                "path": str(root / "unbound-history-manifest.json"),
                "sha256": "0" * 64,
            }
            config_path.write_text(
                json.dumps(config, ensure_ascii=False), encoding="utf-8"
            )
            override_summary = import_multicard(
                config_path,
                root / "output-override",
                direct_history_manifest_path=source_manifest_path,
                direct_history_manifest_sha256=_sha256(source_manifest_path),
            )

            self.assertEqual("target_direct", summary["history_mode"])
            self.assertEqual(36, summary["runner_rows"])
            self.assertEqual(36, summary["race_count"])
            self.assertEqual(36, summary["experienced_runner_rows_mapped"])
            self.assertEqual(0, summary["history_contract_failures"])
            self.assertEqual(
                _sha256(source_manifest_path),
                override_summary["direct_history_manifest_sha256"],
            )
            self.assertEqual(
                str(source_manifest_path.resolve()),
                override_summary["direct_history_manifest_path"],
            )
            self.assertEqual(3, len(summary["cards"]))
            for card in summary["cards"]:
                self.assertEqual(12, card["rows"])
                self.assertEqual(12, card["races"])
                self.assertEqual(0, card["history_contract_failed_races"])

    def test_exp026_preparation_manifests_match_fixed_identity_universe(self) -> None:
        config_path = ROOT / "config" / "grade_r_card_20260809_exp026.json"
        fold_path = (
            ROOT
            / "research"
            / "drafts"
            / "EXP-20260808-025-card-input.fold_manifest.json"
        )
        source_manifest_path = (
            ROOT
            / "research"
            / "drafts"
            / "EXP-20260808-026-history-source-manifest.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        fold = json.loads(fold_path.read_text(encoding="utf-8"))
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

        self.assertEqual("EXP-20260808-026", config["experiment_id"])
        self.assertEqual("target_direct", config["input_contract"]["history_mode"])
        self.assertEqual(36, config["input_contract"]["expected_races"])
        self.assertEqual(495, config["input_contract"]["expected_runner_rows"])
        self.assertEqual(
            447, config["input_contract"]["expected_experienced_runner_rows"]
        )
        self.assertEqual(48, config["input_contract"]["expected_no_history_runner_rows"])
        self.assertNotIn("html", config["input_sources"])
        self.assertFalse(config["safety"]["formal_buy"])
        self.assertFalse(config["safety"]["send_order"])
        self.assertEqual(0, config["safety"]["stake"])

        direct_source = config["input_sources"]["direct_history_manifest"]
        self.assertEqual(
            "research/drafts/EXP-20260808-026-history-source-manifest.json",
            direct_source["path"],
        )
        self.assertEqual(_sha256(source_manifest_path), direct_source["sha256"])
        self.assertEqual(
            "REAL_SOURCE_BINDING_REQUIRED", source_manifest["preparation_status"]
        )
        self.assertEqual([], source_manifest["ra_sources"])
        self.assertEqual([], source_manifest["se_sources"])
        self.assertEqual([], source_manifest["authoritative_latest_races"])
        self.assertFalse(source_manifest["safety"]["real_data_execution_allowed"])
        with self.assertRaisesRegex(ValueError, "non-empty ra_sources"):
            load_direct_history(
                source_manifest_path=source_manifest_path,
                source_manifest_sha256=_sha256(source_manifest_path),
                du=pd.DataFrame(
                    [
                        {
                            "target_race_key": "2026080901010601",
                            "horse_no": 1,
                            "horse_id": "2022100001",
                        }
                    ]
                ),
                target_date="20260809",
            )

        configured_records: list[dict[str, object]] = []
        for card in config["cards"]:
            manifest_path = ROOT / card["target_manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("EXP-20260808-026", manifest["experiment_id"])
            self.assertFalse(manifest["formal_buy"])
            self.assertFalse(manifest["send_order"])
            self.assertEqual(0, manifest["stake"])
            self.assertFalse(manifest["target_selection_uses_odds"])
            self.assertEqual(12, len(manifest["records"]))
            self.assertEqual(card["expected_runner_rows"], sum(
                int(record["runner_count"]) for record in manifest["records"]
            ))
            self.assertEqual(_sha256(fold_path), manifest["source_fold_manifest"]["sha256"])
            configured_records.extend(manifest["records"])

        self.assertEqual(36, len(configured_records))
        self.assertEqual(36, len({str(row["target_race_key"]) for row in configured_records}))
        self.assertEqual(36, len({str(row["race_id"]) for row in configured_records}))
        self.assertEqual(495, sum(int(row["runner_count"]) for row in configured_records))
        self.assertEqual(
            1,
            sum(row["race_domain"] == "obstacle" for row in configured_records),
        )
        fixed_identity = {
            (
                str(row["target_race_key"]),
                str(row["race_id"]),
                int(row["race_no"]),
                int(row["runner_count"]),
                str(row["scheduled_post_time"]),
            )
            for row in fold["races"]
        }
        prepared_identity = {
            (
                str(row["target_race_key"]),
                str(row["race_id"]),
                int(row["race_no"]),
                int(row["runner_count"]),
                str(row["scheduled_post_time"]),
            )
            for row in configured_records
        }
        self.assertEqual(fixed_identity, prepared_identity)
        for record in configured_records:
            cutoff = datetime.fromisoformat(str(record["candidate_feature_cutoff_time"]))
            post = datetime.fromisoformat(str(record["scheduled_post_time"]))
            self.assertEqual(900, int((post - cutoff).total_seconds()))


if __name__ == "__main__":
    unittest.main()
