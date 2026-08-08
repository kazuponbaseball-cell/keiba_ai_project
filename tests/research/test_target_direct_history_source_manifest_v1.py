from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.build_target_direct_history_source_manifest_v1 import (
    COMPLETE_COVERAGE,
    EXPERIMENT_ID,
    PARTIAL_COVERAGE,
    build_manifest,
    expand_history_sources,
    main,
    parse_current_du,
)


TARGET_DATE = "20260809"
TARGET_RACE_PART = "01010601"


def _put(
    raw: bytearray,
    start: int,
    length: int,
    value: str,
    *,
    encoding: str = "ascii",
) -> None:
    encoded = value.encode(encoding)
    if len(encoded) > length:
        raise ValueError((start, length, value))
    raw[start : start + length] = encoded.ljust(length, b" ")


def _du_line(*, horse_no: int, horse_id: str, target_date: str = TARGET_DATE) -> bytes:
    raw = bytearray(b" " * 160)
    _put(raw, 0, 3, "SE1")
    _put(raw, 3, 8, target_date)
    _put(raw, 11, 8, target_date)
    _put(raw, 19, 8, TARGET_RACE_PART)
    _put(raw, 27, 1, str(min(horse_no, 8)))
    _put(raw, 28, 2, f"{horse_no:02d}")
    _put(raw, 32, 8, horse_id[-8:])
    _put(raw, 40, 36, f"CURRENT {horse_no}", encoding="cp932")
    return bytes(raw)


def _ra_line(*, key: str, track_code: str = "11") -> bytes:
    raw = bytearray(b" " * 1270)
    _put(raw, 0, 2, "RA")
    _put(raw, 11, 16, key)
    _put(raw, 697, 4, "1600")
    _put(raw, 705, 2, track_code)
    _put(raw, 881, 2, "01")
    _put(raw, 883, 2, "01")
    _put(raw, 888, 1, "1")
    _put(raw, 889, 1, "1")
    _put(raw, 975, 3, "338")
    return bytes(raw)


def _se_line(*, key: str, horse_id: str, horse_no: int = 1) -> bytes:
    raw = bytearray(b" " * 553)
    _put(raw, 0, 2, "SE")
    _put(raw, 11, 16, key)
    _put(raw, 28, 2, f"{horse_no:02d}")
    _put(raw, 30, 10, horse_id)
    _put(raw, 40, 36, f"HISTORY {horse_no}", encoding="cp932")
    _put(raw, 288, 3, "550")
    _put(raw, 296, 5, "01234")
    _put(raw, 322, 1, "0")
    _put(raw, 324, 3, "470")
    _put(raw, 327, 1, "+")
    _put(raw, 328, 3, "000")
    _put(raw, 331, 1, "0")
    _put(raw, 334, 2, "01")
    _put(raw, 338, 4, "1320")
    _put(raw, 363, 2, "02")
    _put(raw, 390, 3, "338")
    _put(raw, 531, 4, "0000")
    return bytes(raw)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root: Path, horse_ids: tuple[str, ...] = ("2022100001",)) -> None:
        self.root = root
        self.source_root = root / "SE_DATA"
        self.source_dir = self.source_root / "2025"
        self.source_dir.mkdir(parents=True)
        self.du = root / "DU.DAT"
        self.du.write_bytes(
            b"\r\n".join(
                _du_line(horse_no=index, horse_id=horse_id)
                for index, horse_id in enumerate(horse_ids, start=1)
            )
        )
        self.fold = root / "fold.json"
        self.fold.write_text('{"fixed":true}\n', encoding="utf-8")
        self.horse_ids = horse_ids

    def source(self, name: str, *records: bytes, year: str = "2025") -> Path:
        directory = self.source_root / year
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_bytes(b"\r\n".join(records))
        return path

    def build(
        self,
        paths: list[Path],
        *,
        coverage: str = COMPLETE_COVERAGE,
        expected_runner_count: int | None = None,
        du_sha256: str | None = None,
        fold_sha256: str | None = None,
    ) -> dict[str, object]:
        return build_manifest(
            experiment_id=EXPERIMENT_ID,
            du_path=self.du,
            du_sha256=du_sha256 or _sha(self.du),
            fold_manifest_path=self.fold,
            fold_manifest_sha256=fold_sha256 or _sha(self.fold),
            history_paths=paths,
            target_date=TARGET_DATE,
            expected_runner_count=(
                expected_runner_count
                if expected_runner_count is not None
                else len(self.horse_ids)
            ),
            authority_coverage=coverage,
        )


class TargetDirectHistorySourceManifestTests(unittest.TestCase):
    def test_01_selects_latest_supported_jra_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            older = "2026070101010101"
            latest = "2026080104010203"
            path = fixture.source(
                "SU01.DAT",
                _ra_line(key=older),
                _se_line(key=older, horse_id=fixture.horse_ids[0]),
                _ra_line(key=latest),
                _se_line(key=latest, horse_id=fixture.horse_ids[0]),
            )
            manifest = fixture.build([path])
        row = manifest["authoritative_latest_races"][0]
        self.assertEqual("202604010203", row["authoritative_latest_race_id"])
        self.assertEqual("20260801", row["authoritative_latest_race_date"])
        self.assertTrue(row["authority_contract_ok"])
        self.assertEqual("", row["authority_failure_reason"])
        self.assertEqual(1, manifest["summary"]["authority_contract_ok_rows"])

    def test_02_confirms_no_prior_race_only_under_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            path = fixture.source(
                "SU02.DAT",
                _ra_line(key="2026080101010101"),
                _se_line(key="2026080101010101", horse_id="2022999999"),
            )
            manifest = fixture.build([path])
        row = manifest["authoritative_latest_races"][0]
        self.assertTrue(row["authority_contract_ok"])
        self.assertEqual("NO_PRIOR_RACE_CONFIRMED", row["authority_failure_reason"])
        self.assertIsNone(row["authoritative_latest_race_id"])

    def test_03_partial_coverage_cannot_confirm_no_prior_race(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            path = fixture.source(
                "SU03.DAT",
                _ra_line(key="2026080101010101"),
                _se_line(key="2026080101010101", horse_id="2022999999"),
            )
            manifest = fixture.build([path], coverage=PARTIAL_COVERAGE)
        row = manifest["authoritative_latest_races"][0]
        self.assertFalse(row["authority_contract_ok"])
        self.assertEqual("AUTHORITATIVE_HISTORY_UNAVAILABLE", row["authority_failure_reason"])

    def test_04_local_latest_never_falls_back_to_older_jra(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            older = "2026070101010101"
            local = "2026080130010101"
            path = fixture.source(
                "SU04.DAT",
                _ra_line(key=older),
                _se_line(key=older, horse_id=fixture.horse_ids[0]),
                _ra_line(key=local),
                _se_line(key=local, horse_id=fixture.horse_ids[0]),
            )
            manifest = fixture.build([path])
        row = manifest["authoritative_latest_races"][0]
        self.assertEqual("202630010101", row["authoritative_latest_race_id"])
        self.assertEqual("LATEST_PRIOR_RACE_LOCAL", row["authority_failure_reason"])
        self.assertFalse(row["authority_contract_ok"])

    def test_05_foreign_latest_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            older = "2026070101010101"
            foreign = "20260801A1010101"
            path = fixture.source(
                "SU05.DAT",
                _ra_line(key=older),
                _se_line(key=older, horse_id=fixture.horse_ids[0]),
                _ra_line(key=foreign),
                _se_line(key=foreign, horse_id=fixture.horse_ids[0]),
            )
            manifest = fixture.build([path])
        row = manifest["authoritative_latest_races"][0]
        self.assertEqual("2026A1010101", row["authoritative_latest_race_id"])
        self.assertEqual("LATEST_PRIOR_RACE_FOREIGN", row["authority_failure_reason"])
        self.assertFalse(row["authority_contract_ok"])

    def test_06_unknown_venue_latest_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            key = "2026080120010101"
            path = fixture.source(
                "SU06.DAT",
                _ra_line(key=key),
                _se_line(key=key, horse_id=fixture.horse_ids[0]),
            )
            manifest = fixture.build([path])
        row = manifest["authoritative_latest_races"][0]
        self.assertEqual("LATEST_PRIOR_RACE_UNKNOWN", row["authority_failure_reason"])
        self.assertFalse(row["authority_contract_ok"])

    def test_07_same_day_record_is_not_accepted_as_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            prior = "2026080801010101"
            same_day = "2026080901010601"
            path = fixture.source(
                "SU07.DAT",
                _ra_line(key=prior),
                _se_line(key=prior, horse_id=fixture.horse_ids[0]),
                _ra_line(key=same_day),
                _se_line(key=same_day, horse_id=fixture.horse_ids[0]),
            )
            manifest = fixture.build([path])
        row = manifest["authoritative_latest_races"][0]
        self.assertEqual("20260808", row["authoritative_latest_race_date"])
        self.assertTrue(row["authority_contract_ok"])

    def test_08_multiple_latest_races_on_same_date_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            first = "2026080101010101"
            second = "2026080104010101"
            path = fixture.source(
                "SU08.DAT",
                _ra_line(key=first),
                _se_line(key=first, horse_id=fixture.horse_ids[0]),
                _ra_line(key=second),
                _se_line(key=second, horse_id=fixture.horse_ids[0]),
            )
            manifest = fixture.build([path])
        row = manifest["authoritative_latest_races"][0]
        self.assertEqual("MULTIPLE_LATEST_PRIOR_RACES_SAME_DATE", row["authority_failure_reason"])
        self.assertFalse(row["authority_contract_ok"])

    def test_09_missing_ra_for_latest_jra_race_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            missing = "2026080101010101"
            unrelated = "2026070104010101"
            path = fixture.source(
                "SU09.DAT",
                _ra_line(key=unrelated),
                _se_line(key=missing, horse_id=fixture.horse_ids[0]),
            )
            manifest = fixture.build([path])
        row = manifest["authoritative_latest_races"][0]
        self.assertEqual("LATEST_JRA_RACE_MISSING_RA_RECORD", row["authority_failure_reason"])
        self.assertFalse(row["authority_contract_ok"])

    def test_10_jra_record_must_be_feature_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            key = "2026080101010101"
            path = fixture.source(
                "SU10.DAT",
                _ra_line(key=key, track_code="99"),
                _se_line(key=key, horse_id=fixture.horse_ids[0]),
            )
            manifest = fixture.build([path])
        row = manifest["authoritative_latest_races"][0]
        self.assertEqual("LATEST_JRA_RECORD_NOT_FEATURE_READY", row["authority_failure_reason"])
        self.assertFalse(row["authority_contract_ok"])

    def test_11_duplicate_ra_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            key = "2026080101010101"
            one = fixture.source("SU11A.DAT", _ra_line(key=key))
            two = fixture.source("SU11B.DAT", _ra_line(key=key), _se_line(key=key, horse_id=fixture.horse_ids[0]))
            with self.assertRaisesRegex(ValueError, "duplicate RA"):
                fixture.build([one, two])

    def test_12_duplicate_se_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            key = "2026080101010101"
            one = fixture.source("SU12A.DAT", _ra_line(key=key), _se_line(key=key, horse_id=fixture.horse_ids[0]))
            two = fixture.source("SR12B.DAT", _se_line(key=key, horse_id=fixture.horse_ids[0]))
            with self.assertRaisesRegex(ValueError, "duplicate SE"):
                fixture.build([one, two])

    def test_13_duplicate_current_horse_identity_is_rejected(self) -> None:
        payload = b"\r\n".join(
            (
                _du_line(horse_no=1, horse_id="2022100001"),
                _du_line(horse_no=2, horse_id="2022100001"),
            )
        )
        with self.assertRaisesRegex(ValueError, "duplicate current horse"):
            parse_current_du(payload, target_date=TARGET_DATE)

    def test_14_fixed_du_hash_mismatch_is_rejected_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            key = "2026080101010101"
            path = fixture.source(
                "SU14.DAT",
                _ra_line(key=key),
                _se_line(key=key, horse_id=fixture.horse_ids[0]),
            )
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                fixture.build([path], du_sha256="0" * 64)

    def test_15_source_enumeration_is_year_root_and_duplicate_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            one = fixture.source("SU15.DAT", _ra_line(key="2026080101010101"))
            two = fixture.source("SR15.DAT", _se_line(key="2026080101010101", horse_id=fixture.horse_ids[0]), year="2026")
            paths = expand_history_sources(
                [str(one), str(two)],
                allowed_roots=[fixture.source_root],
                allowed_years={"2025", "2026"},
            )
            self.assertEqual([one.resolve(), two.resolve()], paths)
            with self.assertRaisesRegex(ValueError, "declared more than once"):
                expand_history_sources(
                    [str(one), str(one)],
                    allowed_roots=[fixture.source_root],
                    allowed_years={"2025", "2026"},
                )

    def test_16_cli_writes_canonical_safe_manifest_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            key = "2026080101010101"
            source = fixture.source(
                "SU16.DAT",
                _ra_line(key=key),
                _se_line(key=key, horse_id=fixture.horse_ids[0]),
            )
            output = Path(temporary) / "out" / "manifest.json"
            arguments = [
                "--du", str(fixture.du),
                "--du-sha256", _sha(fixture.du),
                "--fold-manifest", str(fixture.fold),
                "--fold-manifest-sha256", _sha(fixture.fold),
                "--history-glob", str(source),
                "--allowed-source-root", str(fixture.source_root),
                "--allowed-year", "2025",
                "--allowed-year", "2026",
                "--target-date", TARGET_DATE,
                "--expected-runner-count", "1",
                "--authority-coverage", COMPLETE_COVERAGE,
                "--output", str(output),
            ]
            self.assertEqual(0, main(arguments))
            raw = output.read_text(encoding="utf-8")
            manifest = json.loads(raw)
            self.assertEqual(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", raw)
            self.assertEqual({"formal_buy": False, "send_order": False, "stake": 0, "prediction_rows": 0, "market_rows": 0}, manifest["safety"])
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                main(arguments)


if __name__ == "__main__":
    unittest.main()
