from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.research.import_target_multicard_entry_v1 import (
    RaceMetadata,
    _race_class,
    _surface_and_distance,
    build_entry_rows,
    bind_fixed_input_metadata,
    canonical_candidate_race_id,
    match_html_runner_groups_to_du,
    parse_du,
    parse_html_runner_groups,
    parse_ra_race_metadata,
)


def _put(raw: bytearray, start: int, end: int, value: str) -> None:
    encoded = value.encode("cp932")[: end - start]
    raw[start:end] = encoded.ljust(end - start, b" ")


def _du_line(
    *,
    race_date: str = "20260808",
    race_part: str = "01010501",
    horse_no: int = 1,
    horse_id: str = "22000001",
    horse_name: str = "テストホース",
    record_type: str = "SE1",
) -> bytes:
    raw = bytearray(b" " * 160)
    _put(raw, 0, 3, record_type)
    _put(raw, 3, 11, "20260808")
    _put(raw, 11, 19, race_date)
    _put(raw, 19, 27, race_part)
    _put(raw, 27, 28, "1")
    _put(raw, 28, 30, f"{horse_no:02d}")
    _put(raw, 32, 40, horse_id)
    _put(raw, 40, 76, horse_name)
    _put(raw, 82, 84, "03")
    _put(raw, 86, 90, "1234")
    _put(raw, 90, 100, "調教師")
    _put(raw, 104, 109, "55000")
    _put(raw, 112, 117, "01234")
    _put(raw, 122, 132, "騎手")
    return bytes(raw)


def _html_block(frame_no: int, horse_no: int, horse_name: str) -> str:
    return (
        f"<HR><TD NOWRAP >{frame_no}枠 {horse_no}番"
        f"<TD><B>{horse_name}</B><TD>synthetic history"
    )


def _ra_line(
    *,
    target_race_key: str,
    race_name: str = "",
    grade_code: str = "",
    race_type_code: str = "13",
    condition_code: str = "005",
    distance: int = 1600,
    track_code: str = "11",
    post_hhmm: str = "1200",
    runner_count: int = 3,
) -> bytes:
    raw = bytearray(b" " * 890)
    _put(raw, 0, 3, "RA2")
    _put(raw, 3, 11, "20260808")
    _put(raw, 11, 27, target_race_key)
    _put(raw, 32, 92, race_name)
    _put(raw, 572, 592, race_name)
    _put(raw, 614, 615, grade_code)
    _put(raw, 616, 618, race_type_code)
    _put(raw, 634, 637, condition_code)
    _put(raw, 697, 701, f"{distance:04d}")
    _put(raw, 705, 707, track_code)
    _put(raw, 709, 711, "A")
    _put(raw, 873, 877, post_hhmm)
    _put(raw, 881, 883, f"{runner_count:02d}")
    return bytes(raw)


class TargetMulticardEntryTests(unittest.TestCase):
    def test_contract_matrix_contains_exactly_24_synthetic_cases(self) -> None:
        cases = [
            ("芝1200m 1勝クラス", ("芝", 1200), "1勝クラス"),
            ("ダート1800m 未勝利", ("ダ", 1800), "未勝利"),
            ("ダ 1400m 新馬", ("ダ", 1400), "新馬"),
            ("障害 3000m オープン", ("障害", 3000), "オープン"),
            ("1600m 芝 GIII", ("芝", 1600), "GIII"),
            ("2000m ダート GII", ("ダ", 2000), "GII"),
            ("芝・2600m 2勝クラス", ("芝", 2600), "2勝クラス"),
            ("ダ･1200m 3勝クラス", ("ダ", 1200), "3勝クラス"),
            ("芝1800m リステッド", ("芝", 1800), "リステッド"),
            ("芝2000m Listed", ("芝", 2000), "Listed"),
            ("芝2400m GI", ("芝", 2400), "GI"),
            ("芝2200m G1", ("芝", 2200), "G1"),
            ("ダ1700m G2", ("ダ", 1700), "G2"),
            ("芝1400m G3", ("芝", 1400), "G3"),
            ("芝1000m オープン", ("芝", 1000), "オープン"),
            ("ダ1900m 1勝クラス", ("ダ", 1900), "1勝クラス"),
            ("芝3000m 未勝利", ("芝", 3000), "未勝利"),
            ("障害2890m 障害未勝利", ("障害", 2890), "未勝利"),
            ("芝1800m 1勝クラス", ("芝", 1800), "1勝クラス"),
            ("ダ2000m 2勝クラス", ("ダ", 2000), "2勝クラス"),
            ("芝1200m 3勝クラス", ("芝", 1200), "3勝クラス"),
            ("ダ1700m オープン", ("ダ", 1700), "オープン"),
            ("芝1600m GII", ("芝", 1600), "GII"),
            ("障害3000m 未勝利", ("障害", 3000), "未勝利"),
        ]
        self.assertEqual(24, len(cases))
        for text, expected_course, expected_class in cases:
            with self.subTest(text=text):
                self.assertEqual(expected_course, _surface_and_distance(text))
                self.assertEqual(expected_class.lower(), _race_class(text).lower())

    def test_parse_du_keeps_active_rows_and_maps_candidate_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "DU.DAT"
            path.write_bytes(
                b"\r\n".join(
                    [
                        _du_line(horse_no=1, horse_id="22000001", horse_name="アルファ"),
                        _du_line(horse_no=2, horse_id="22000002", horse_name="ベータ", record_type="SE2"),
                        _du_line(horse_no=3, horse_id="22000003", horse_name="除外", record_type="XX1"),
                    ]
                )
            )
            frame = parse_du(path)
        self.assertEqual(2, len(frame))
        self.assertEqual({"202601010501"}, set(frame["race_id"]))
        self.assertEqual("202601010501", canonical_candidate_race_id("2026080801010501"))

    def test_parse_du_rejects_duplicate_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "DU.DAT"
            line = _du_line(horse_no=1, horse_id="22000001", horse_name="アルファ")
            path.write_bytes(line + b"\r\n" + line)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                parse_du(path)

    def test_html_runner_groups_match_du_by_exact_runner_signature(self) -> None:
        source = "".join(
            [
                _html_block(1, 1, "アルファ"),
                _html_block(2, 2, "ベータ"),
                _html_block(3, 3, "ガンマ"),
                _html_block(1, 1, "デルタ"),
                _html_block(2, 2, "イプシロン"),
            ]
        )
        runners = pd.DataFrame(
            [
                {"target_race_key": key, "horse_no": no, "horse_name": name}
                for key, entries in (
                    ("2026080801010501", [(1, "アルファ"), (2, "ベータ"), (3, "ガンマ")]),
                    ("2026080801010502", [(1, "デルタ"), (2, "イプシロン")]),
                )
                for no, name in entries
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(source.encode("cp932"))
            groups = parse_html_runner_groups(
                path, expected_runner_rows=5, expected_races=2
            )
        matches = match_html_runner_groups_to_du(groups, runners)
        self.assertEqual(2, groups["html_group_index"].nunique())
        self.assertEqual(3, matches["2026080801010501"]["runner_name_match_count"])
        self.assertEqual(2, matches["2026080801010502"]["html_group_index"])

    def test_html_runner_group_fails_closed_on_identity_mismatch(self) -> None:
        runners = pd.DataFrame(
            [{"target_race_key": "2026080804020502", "horse_no": 1, "horse_name": "別馬"}]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(_html_block(1, 1, "アルファ").encode("cp932"))
            groups = parse_html_runner_groups(
                path, expected_runner_rows=1, expected_races=1
            )
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            match_html_runner_groups_to_du(groups, runners)

    def test_ra_metadata_uses_official_fixed_byte_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "DR.DAT"
            path.write_bytes(
                _ra_line(
                    target_race_key="2026080801010501",
                    condition_code="703",
                    distance=1700,
                    track_code="24",
                    post_hhmm="1000",
                    runner_count=14,
                )
                + b"\r\n"
                + _ra_line(
                    target_race_key="2026080801010511",
                    race_name="エルムステークス",
                    grade_code="C",
                    condition_code="999",
                    distance=1700,
                    track_code="24",
                    post_hhmm="1525",
                    runner_count=11,
                )
            )
            metadata = parse_ra_race_metadata(path, expected_races=2)
        maiden = metadata["2026080801010501"]
        grade = metadata["2026080801010511"]
        self.assertEqual(("ダ", 1700, "未勝利", "1000"), (
            maiden.surface, maiden.distance, maiden.race_class, maiden.scheduled_post_hhmm
        ))
        self.assertEqual(("Ｇ３", "エルムステークス", 11), (
            grade.race_class, grade.race_name, grade.registered_runner_count
        ))

    def test_bind_metadata_rejects_post_time_or_data_category_drift(self) -> None:
        key = "2026080801010501"
        du = pd.DataFrame(
            [{"target_race_key": key, "horse_no": no, "horse_name": f"馬{no}"} for no in range(1, 4)]
        )
        target = {
            "target_race_key": key,
            "runner_count": 3,
            "scheduled_post_time": "2026-08-08T12:00:00+09:00",
        }
        meta = RaceMetadata(
            venue="札幌", race_no=1, surface="芝", distance=1600,
            race_class="1勝", race_name="", runner_name_match_count=0,
            target_race_key=key, data_category="2", race_type_code="13",
            condition_code="005", track_code="11", scheduled_post_hhmm="1201",
            registered_runner_count=3,
        )
        with self.assertRaisesRegex(ValueError, "post time mismatch"):
            bind_fixed_input_metadata(
                du=du,
                html_matches={key: {"html_group_index": 1, "runner_name_match_count": 3}},
                ra_metadata={key: meta},
                target_records=[target],
            )

    def test_build_entry_rows_preserves_12_race_denominator_and_blanks_market(self) -> None:
        rows = []
        records = []
        metadata = {}
        for race_no in range(1, 13):
            target_key = f"20260808010105{race_no:02d}"
            race_id = canonical_candidate_race_id(target_key)
            records.append(
                {
                    "target_race_key": target_key,
                    "race_id": race_id,
                    "race_no": race_no,
                    "runner_count": 3,
                    "scheduled_post_time": f"2026-08-08T{9 + race_no:02d}:00:00+09:00",
                    "race_domain": "flat",
                }
            )
            metadata[target_key] = RaceMetadata(
                venue="札幌",
                race_no=race_no,
                surface="芝",
                distance=1200,
                race_class="未勝利",
                race_name="札幌 synthetic",
                runner_name_match_count=3,
            )
            for horse_no in range(1, 4):
                rows.append(
                    {
                        "target_race_key": target_key,
                        "race_id": race_id,
                        "race_date": "20260808",
                        "venue_code": "01",
                        "venue": "札幌",
                        "meeting_no": 1,
                        "day_no": 5,
                        "race_no": race_no,
                        "frame_no": horse_no,
                        "horse_no": horse_no,
                        "horse_id": f"2026{race_no:02d}{horse_no:02d}",
                        "horse_name": f"馬{race_no}-{horse_no}",
                        "age": 3,
                        "trainer_code": "1234",
                        "trainer_name": "調教師",
                        "assigned_weight_kg": 55.0,
                        "jockey_code": "01234",
                        "jockey_name": "騎手",
                    }
                )
        manifest = {"records": records}
        frame = build_entry_rows(pd.DataFrame(rows), manifest, metadata)
        self.assertEqual(36, len(frame))
        self.assertEqual(12, frame["race_id"].nunique())
        self.assertTrue(frame["人気"].eq("").all())
        self.assertTrue(frame["単勝オッズ"].eq("").all())
        self.assertTrue(frame["確定着順"].eq("").all())
        self.assertEqual({"flat_turf"}, set(frame["race_domain"]))

    def test_build_entry_rows_rejects_missing_runner(self) -> None:
        manifest = {
            "records": [
                {
                    "target_race_key": f"20260808010105{race_no:02d}",
                    "race_id": f"2026010105{race_no:02d}",
                    "race_no": race_no,
                    "runner_count": 1,
                    "scheduled_post_time": "2026-08-08T12:00:00+09:00",
                    "race_domain": "flat",
                }
                for race_no in range(1, 13)
            ]
        }
        with self.assertRaisesRegex(ValueError, "runner count mismatch"):
            build_entry_rows(pd.DataFrame(), manifest, {})


if __name__ == "__main__":
    unittest.main()
