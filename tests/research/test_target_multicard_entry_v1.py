from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.enrich_prediction_basic_ability_features import load_recent_result_history
from scripts.research.build_grade_r_candidate_freeze_packets_v1 import (
    validate_target_manifest,
)
from scripts.research.import_target_multicard_entry_v1 import (
    PREVIOUS_RACE_COLUMNS,
    PREVIOUS_RACE_SOURCE_ALIASES,
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
from scripts.research.run_grade_r_card_v1 import _candidate_config


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _html_block(
    frame_no: int,
    horse_no: int,
    horse_name: str,
    history_html: str = "",
) -> str:
    return (
        f"<HR><TD NOWRAP >{frame_no}枠 {horse_no}番"
        f"<TD><B>{horse_name}</B><TD>synthetic history{history_html}"
    )


def _history_row(
    source_date: str,
    horse_name: str,
    horse_id: str,
    marker: str,
) -> dict[str, str]:
    row = {
        "\u65e5\u4ed8S": source_date,
        "\u99ac\u540d": horse_name,
        "\u8840\u7d71\u767b\u9332\u756a\u53f7": horse_id,
        "\u5358\u52dd\u30aa\u30c3\u30ba": "99.9",
    }
    for index, aliases in enumerate(PREVIOUS_RACE_SOURCE_ALIASES.values()):
        row[aliases[0]] = f"{marker}-{index:02d}"
    return row


def _history_table(
    *rows: dict[str, str],
    omitted_header: str | None = None,
    include_display_date: bool = False,
) -> str:
    headers = [
        *(["\u65e5\u4ed8"] if include_display_date else []),
        "\u65e5\u4ed8S",
        "\u99ac\u540d",
        "\u8840\u7d71\u767b\u9332\u756a\u53f7",
        *(aliases[0] for aliases in PREVIOUS_RACE_SOURCE_ALIASES.values()),
        "\u5358\u52dd\u30aa\u30c3\u30ba",
    ]
    if omitted_header is not None:
        headers.remove(omitted_header)
    output = ["<TABLE><TR>", *(f"<TH>{header}</TH>" for header in headers), "</TR>"]
    for row in rows:
        output.extend(
            ["<TR>", *(f"<TD>{row.get(header, '')}</TD>" for header in headers), "</TR>"]
        )
    output.append("</TABLE>")
    return "".join(output)


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
    def test_exp019_config_identity_paths_manifests_and_recent_history(self) -> None:
        config_path = ROOT / "config" / "grade_r_card_20260808_exp019.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual("EXP-20260808-019", config["experiment_id"])

        static_files = [
            Path(config["bundle"]["inference_bundle"]),
            Path(config["history"]["baseline_model"]),
            Path(config["history"]["historical_csv"]),
            ROOT / config["history"]["baseline_config"],
        ]
        self.assertTrue(all(path.is_file() for path in static_files))
        self.assertTrue(Path(config["history"]["ability_history_dir"]).is_dir())
        self.assertTrue(all("繝" not in str(path) for path in static_files))
        self.assertEqual(
            config["bundle"]["sha256"],
            _sha256(Path(config["bundle"]["inference_bundle"])),
        )
        for source in config["input_sources"].values():
            source_path = Path(source["path"])
            self.assertTrue(source_path.is_file())
            self.assertEqual(source["sha256"], _sha256(source_path))

        registered = 0
        for card in config["cards"]:
            manifest_path = ROOT / card["target_manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(config["experiment_id"], manifest["experiment_id"])
            records = validate_target_manifest(manifest, _candidate_config(config, manifest))
            self.assertEqual(12, len(records))
            registered += len(records)
        self.assertEqual(36, registered)

        result_path = ROOT / config["history"]["recent_result_globs"][0]
        entry_path = ROOT / config["history"]["entry_globs"][0]
        history = load_recent_result_history([str(result_path)], [str(entry_path)])
        self.assertEqual(459, len(history))
        self.assertEqual("20260802", str(history["date_key"].max()))
        self.assertTrue(history["recent_result_source"].ne("").all())

        self.assertFalse(config["safety"]["formal_buy"])
        self.assertFalse(config["safety"]["send_order"])
        self.assertEqual(0, config["safety"]["stake"])

    def test_exp020_config_freezes_policy_and_history_contract(self) -> None:
        prior = json.loads(
            (ROOT / "config" / "grade_r_card_20260808_exp019.json").read_text(
                encoding="utf-8"
            )
        )
        config = json.loads(
            (ROOT / "config" / "grade_r_card_20260808_exp020.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual("EXP-20260808-020", config["experiment_id"])
        self.assertEqual(prior["candidate_policy"], config["candidate_policy"])
        self.assertEqual(prior["bundle"], config["bundle"])
        self.assertEqual(24, config["input_contract"]["required_previous_race_columns"])
        self.assertEqual(420, config["input_contract"]["expected_experienced_runner_rows"])
        self.assertEqual(42, config["input_contract"]["expected_no_history_runner_rows"])
        self.assertFalse(config["safety"]["formal_buy"])
        self.assertFalse(config["safety"]["send_order"])
        self.assertEqual(0, config["safety"]["stake"])
        for card in config["cards"]:
            manifest = json.loads((ROOT / card["target_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual("EXP-20260808-020", manifest["experiment_id"])
            self.assertEqual(12, len(manifest["records"]))
            self.assertFalse(manifest["target_selection_uses_odds"])
            self.assertFalse(manifest["formal_buy"])
            self.assertFalse(manifest["send_order"])
            self.assertEqual(0, manifest["stake"])

    def test_baseline_prediction_module_imports_with_tracked_loader(self) -> None:
        from src.predict import predict_baseline

        self.assertTrue(callable(predict_baseline.main))

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

    def test_html_history_selects_latest_prior_row_and_maps_allowlist(self) -> None:
        horse_name = "\u30c6\u30b9\u30c8\u30db\u30fc\u30b9"
        horse_id = "2022000001"
        older = _history_row("2026/07/01", horse_name, horse_id, "older")
        latest = _history_row("2026/08/02", horse_name, horse_id, "latest")
        source = _html_block(1, 1, horse_name, _history_table(older, latest))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(source.encode("cp932"))
            groups = parse_html_runner_groups(
                path,
                expected_runner_rows=1,
                expected_races=1,
                target_date="20260808",
            )
        row = groups.iloc[0]
        self.assertEqual("20260802", row["previous_race_source_date"])
        self.assertTrue(row["previous_race_contract_ok"])
        self.assertEqual("", row["previous_race_no_history_reason"])
        for destination, aliases in PREVIOUS_RACE_SOURCE_ALIASES.items():
            self.assertEqual(latest[aliases[0]], row[destination])
        self.assertNotIn("\u5358\u52dd\u30aa\u30c3\u30ba", groups.columns)
        self.assertEqual(24, len(PREVIOUS_RACE_COLUMNS))

    def test_html_history_accepts_equivalent_display_and_sortable_dates(self) -> None:
        horse_name = "\u30c6\u30b9\u30c8\u30db\u30fc\u30b9"
        row = _history_row("2026.8.2", horse_name, "2022000001", "dual-date")
        row["\u65e5\u4ed8"] = "2026. 8. 2"
        source = _html_block(
            1,
            1,
            horse_name,
            _history_table(row, include_display_date=True),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(source.encode("cp932"))
            groups = parse_html_runner_groups(
                path,
                expected_runner_rows=1,
                expected_races=1,
                target_date="20260808",
            )
        self.assertEqual("20260802", groups.iloc[0]["previous_race_source_date"])

    def test_html_history_rejects_disagreeing_date_aliases(self) -> None:
        horse_name = "\u30c6\u30b9\u30c8\u30db\u30fc\u30b9"
        row = _history_row("2026.8.2", horse_name, "2022000001", "dual-date")
        row["\u65e5\u4ed8"] = "2026. 8. 1"
        source = _html_block(
            1,
            1,
            horse_name,
            _history_table(row, include_display_date=True),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(source.encode("cp932"))
            with self.assertRaisesRegex(ValueError, "date aliases disagree"):
                parse_html_runner_groups(
                    path,
                    expected_runner_rows=1,
                    expected_races=1,
                    target_date="20260808",
                )

    def test_html_history_rejects_same_day_or_future_row(self) -> None:
        horse_name = "\u30c6\u30b9\u30c8\u30db\u30fc\u30b9"
        source = _html_block(
            1,
            1,
            horse_name,
            _history_table(_history_row("2026/08/08", horse_name, "2022000001", "same")),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(source.encode("cp932"))
            with self.assertRaisesRegex(ValueError, "same-day or future"):
                parse_html_runner_groups(
                    path,
                    expected_runner_rows=1,
                    expected_races=1,
                    target_date="20260808",
                )

    def test_html_history_rejects_selected_row_identity_mismatch(self) -> None:
        current_name = "\u30c6\u30b9\u30c8\u30db\u30fc\u30b9"
        source = _html_block(
            1,
            1,
            current_name,
            _history_table(_history_row("2026/08/02", "\u5225\u99ac", "2022000001", "bad")),
        )
        runners = pd.DataFrame(
            [
                {
                    "target_race_key": "2026080801010501",
                    "horse_no": 1,
                    "horse_name": current_name,
                    "horse_id": "2022000001",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(source.encode("cp932"))
            groups = parse_html_runner_groups(
                path,
                expected_runner_rows=1,
                expected_races=1,
                target_date="20260808",
            )
        with self.assertRaisesRegex(ValueError, "history horse name identity mismatch"):
            match_html_runner_groups_to_du(groups, runners)

    def test_html_history_rejects_selected_row_horse_id_mismatch(self) -> None:
        current_name = "\u30c6\u30b9\u30c8\u30db\u30fc\u30b9"
        source = _html_block(
            1,
            1,
            current_name,
            _history_table(_history_row("2026/08/02", current_name, "2099999999", "bad-id")),
        )
        runners = pd.DataFrame(
            [
                {
                    "target_race_key": "2026080801010501",
                    "horse_no": 1,
                    "horse_name": current_name,
                    "horse_id": "2022000001",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(source.encode("cp932"))
            groups = parse_html_runner_groups(
                path,
                expected_runner_rows=1,
                expected_races=1,
                target_date="20260808",
            )
        with self.assertRaisesRegex(ValueError, "history horse id identity mismatch"):
            match_html_runner_groups_to_du(groups, runners)

    def test_html_history_marks_true_no_history_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(_html_block(1, 1, "\u65b0\u99ac").encode("cp932"))
            groups = parse_html_runner_groups(
                path,
                expected_runner_rows=1,
                expected_races=1,
                target_date="20260808",
            )
        row = groups.iloc[0]
        self.assertEqual("NO_PRIOR_RACE_TABLE", row["previous_race_no_history_reason"])
        self.assertEqual("", row["previous_race_source_date"])
        self.assertTrue(all(row[column] == "" for column in PREVIOUS_RACE_COLUMNS))

    def test_html_history_ignores_unrelated_layout_table_for_debut_runner(self) -> None:
        source = _html_block(
            1,
            1,
            "\u65b0\u99ac",
            "<TABLE><TR><TD>profile</TD><TD>memo</TD></TR></TABLE>",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(source.encode("cp932"))
            groups = parse_html_runner_groups(
                path,
                expected_runner_rows=1,
                expected_races=1,
                target_date="20260808",
            )
        self.assertEqual(
            "NO_PRIOR_RACE_TABLE",
            groups.iloc[0]["previous_race_no_history_reason"],
        )

    def test_html_history_accepts_legacy_implicit_cell_closures(self) -> None:
        horse_name = "\u30c6\u30b9\u30c8\u30db\u30fc\u30b9"
        row = _history_row("2026/08/02", horse_name, "2022000001", "legacy")
        headers = [
            "\u65e5\u4ed8S",
            "\u99ac\u540d",
            "\u8840\u7d71\u767b\u9332\u756a\u53f7",
            *(aliases[0] for aliases in PREVIOUS_RACE_SOURCE_ALIASES.values()),
        ]
        legacy_table = (
            "<TABLE><TR>"
            + "".join(f"<TD>{header}" for header in headers)
            + "<TR>"
            + "".join(f"<TD>{row[header]}" for header in headers)
            + "</TABLE>"
        )
        source = _html_block(1, 1, horse_name, legacy_table)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(source.encode("cp932"))
            groups = parse_html_runner_groups(
                path,
                expected_runner_rows=1,
                expected_races=1,
                target_date="20260808",
            )
        self.assertEqual("20260802", groups.iloc[0]["previous_race_source_date"])

    def test_html_history_rejects_malformed_required_schema(self) -> None:
        horse_name = "\u30c6\u30b9\u30c8\u30db\u30fc\u30b9"
        missing_source = next(iter(PREVIOUS_RACE_SOURCE_ALIASES.values()))[0]
        source = _html_block(
            1,
            1,
            horse_name,
            _history_table(
                _history_row("2026/08/02", horse_name, "2022000001", "bad"),
                omitted_header=missing_source,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "entry.htm"
            path.write_bytes(source.encode("cp932"))
            with self.assertRaisesRegex(ValueError, "missing required headers"):
                parse_html_runner_groups(
                    path,
                    expected_runner_rows=1,
                    expected_races=1,
                    target_date="20260808",
                )

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
        history_rows = []
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
                html_group_index=race_no,
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
                history_rows.append(
                    {
                        "html_group_index": race_no,
                        "horse_no": horse_no,
                        **{column: f"history-{race_no}-{horse_no}" for column in PREVIOUS_RACE_COLUMNS},
                        "previous_race_source_date": "20260802",
                        "previous_race_source_record_hash": f"hash-{race_no}-{horse_no}",
                        "previous_race_no_history_reason": "",
                        "previous_race_contract_ok": True,
                    }
                )
        manifest = {"records": records}
        frame = build_entry_rows(
            pd.DataFrame(rows),
            manifest,
            metadata,
            html_history=pd.DataFrame(history_rows),
        )
        self.assertEqual(36, len(frame))
        self.assertEqual(12, frame["race_id"].nunique())
        self.assertTrue(frame["人気"].eq("").all())
        self.assertTrue(frame["単勝オッズ"].eq("").all())
        self.assertTrue(frame["確定着順"].eq("").all())
        self.assertEqual({"flat_turf"}, set(frame["race_domain"]))
        self.assertTrue(set(PREVIOUS_RACE_COLUMNS).issubset(frame.columns))
        self.assertTrue(frame["previous_race_source_date"].eq("20260802").all())
        self.assertTrue(frame[PREVIOUS_RACE_COLUMNS[0]].astype(str).str.startswith("history-").all())

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
