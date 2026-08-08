from __future__ import annotations

import unittest

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
        self.assertEqual(set(PREVIOUS_RACE_COLUMNS), set(PREVIOUS_RACE_COLUMNS))
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


if __name__ == "__main__":
    unittest.main()
