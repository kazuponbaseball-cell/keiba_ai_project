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
    canonical_candidate_race_id,
    parse_du,
    parse_html_race_metadata,
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


class TargetMulticardEntryTests(unittest.TestCase):
    def test_contract_matrix_contains_exactly_18_synthetic_cases(self) -> None:
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
        ]
        self.assertEqual(18, len(cases))
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

    def test_html_metadata_uses_runner_names_to_select_current_heading(self) -> None:
        runners = pd.DataFrame(
            [
                {
                    "venue": "札幌",
                    "race_no": 1,
                    "target_race_key": "2026080801010501",
                    "horse_name": name,
                }
                for name in ("アルファ", "ベータ", "ガンマ")
            ]
        )
        source = """
        <html><body>
          <h2>札幌1R 過去情報 芝1800m</h2><p>別馬</p>
          <h2>札幌1R 3歳未勝利 芝1200m</h2>
          <p>アルファ ベータ ガンマ</p>
        </body></html>
        """
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(source.encode("cp932"))
            metadata = parse_html_race_metadata(path, runners)
        race = metadata[("札幌", 1)]
        self.assertEqual("芝", race.surface)
        self.assertEqual(1200, race.distance)
        self.assertEqual("未勝利", race.race_class)
        self.assertEqual(3, race.runner_name_match_count)

    def test_html_metadata_fails_closed_on_identity_mismatch(self) -> None:
        runners = pd.DataFrame(
            [
                {
                    "venue": "新潟",
                    "race_no": 2,
                    "target_race_key": "2026080804020502",
                    "horse_name": name,
                }
                for name in ("アルファ", "ベータ", "ガンマ")
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes("<h2>新潟2R 芝1600m</h2><p>アルファ</p>".encode("cp932"))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                parse_html_race_metadata(path, runners)

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
            metadata[("札幌", race_no)] = RaceMetadata(
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
