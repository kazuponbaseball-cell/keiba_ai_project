from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


RECORD_LENGTH = 1609
PEDIGREE_PATTERN = re.compile(r"(\d{10,16})([^0-9]{1,80})")

ANCESTOR_POSITIONS: list[tuple[str, str, int, str]] = [
    ("sire", "父", 1, "sire"),
    ("dam", "母", 1, "dam"),
    ("sire_sire", "父父", 2, "sire"),
    ("sire_dam", "父母", 2, "sire"),
    ("dam_sire", "母父", 2, "dam"),
    ("dam_dam", "母母", 2, "dam"),
    ("sire_sire_sire", "父父父", 3, "sire"),
    ("sire_sire_dam", "父父母", 3, "sire"),
    ("sire_dam_sire", "父母父", 3, "sire"),
    ("sire_dam_dam", "父母母", 3, "sire"),
    ("dam_sire_sire", "母父父", 3, "dam"),
    ("dam_sire_dam", "母父母", 3, "dam"),
    ("dam_dam_sire", "母母父", 3, "dam"),
    ("dam_dam_dam", "母母母", 3, "dam"),
]


def clean_name(value: str) -> str:
    return value.replace("\u3000", " ").strip()


def iter_um_records(target_data_dir: Path):
    for path in sorted(target_data_dir.glob("UM_DATA/*/UM*.DAT")):
        data = path.read_bytes()
        if len(data) < RECORD_LENGTH:
            continue
        for offset in range(0, len(data) - RECORD_LENGTH + 1, RECORD_LENGTH):
            yield path, data[offset : offset + RECORD_LENGTH]


def parse_record(record: bytes, source_file: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not record.startswith(b"UM"):
        return None, []

    text = record.decode("cp932", errors="ignore")
    horse_id = text[11:21].strip()
    if not horse_id.isdigit():
        return None, []

    pairs: list[tuple[str, str]] = []
    for raw_id, raw_name in PEDIGREE_PATTERN.findall(text[120:1609]):
        name = clean_name(raw_name)
        if name:
            pairs.append((raw_id[-10:], name))

    if len(pairs) < 2:
        return None, []

    row: dict[str, Any] = {
        "血統登録番号": horse_id,
        "ancestor_count": len(pairs),
        "pedigree_complete_14_flag": int(len(pairs) >= len(ANCESTOR_POSITIONS)),
        "source_file": str(source_file),
    }
    long_rows: list[dict[str, Any]] = []
    for idx, (key, label, generation, side) in enumerate(ANCESTOR_POSITIONS):
        ancestor_id = pairs[idx][0] if idx < len(pairs) else ""
        ancestor_name = pairs[idx][1] if idx < len(pairs) else ""
        row[f"{label}馬"] = ancestor_name
        row[f"{key}_id"] = ancestor_id
        row[f"{key}_name"] = ancestor_name
        long_rows.append(
            {
                "血統登録番号": horse_id,
                "position_index": idx + 1,
                "position_key": key,
                "position_label": label,
                "generation": generation,
                "side": side,
                "ancestor_id": ancestor_id,
                "ancestor_name": ancestor_name,
                "source_file": str(source_file),
            }
        )

    return row, long_rows


def build_deep_pedigree_master(target_data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records_by_horse: dict[str, dict[str, Any]] = {}
    long_by_horse: dict[str, list[dict[str, Any]]] = {}
    scanned_records = 0
    parsed_records = 0
    complete_14 = 0
    max_ancestor_count = 0

    for path, record in iter_um_records(target_data_dir):
        scanned_records += 1
        parsed, long_rows = parse_record(record, path)
        if parsed is None:
            continue
        parsed_records += 1
        complete_14 += int(parsed["pedigree_complete_14_flag"])
        max_ancestor_count = max(max_ancestor_count, int(parsed["ancestor_count"]))
        records_by_horse[str(parsed["血統登録番号"])] = parsed
        long_by_horse[str(parsed["血統登録番号"])] = long_rows

    rows = [records_by_horse[k] for k in sorted(records_by_horse)]
    long_rows = [item for k in sorted(long_by_horse) for item in long_by_horse[k]]
    summary = {
        "target_data_dir": str(target_data_dir),
        "scanned_records": scanned_records,
        "parsed_records": parsed_records,
        "unique_horses": len(rows),
        "complete_14_records": complete_14,
        "complete_14_rate": complete_14 / parsed_records if parsed_records else 0.0,
        "max_ancestor_count_detected": max_ancestor_count,
        "position_order": [
            {"position_key": key, "position_label": label, "generation": generation, "side": side}
            for key, label, generation, side in ANCESTOR_POSITIONS
        ],
        "note": "UM_DATA contains 14 ancestor slots in this environment, not a full 5-generation/62-ancestor table.",
    }
    return rows, long_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured 14-slot pedigree from TARGET UM_DATA.")
    parser.add_argument("--target-data-dir", default=r"C:\Users\kazup\Data Lab")
    parser.add_argument("--output-csv", default="data/processed/target/deep_pedigree_master.csv")
    parser.add_argument("--output-long-csv", default="data/processed/target/deep_pedigree_long.csv")
    parser.add_argument("--summary-json", default="outputs/analysis/target_deep_pedigree_master_summary.json")
    args = parser.parse_args()

    target_data_dir = Path(args.target_data_dir)
    output_csv = Path(args.output_csv)
    output_long_csv = Path(args.output_long_csv)
    summary_json = Path(args.summary_json)

    rows, long_rows, summary = build_deep_pedigree_master(target_data_dir)

    base_fields = ["血統登録番号", "ancestor_count", "pedigree_complete_14_flag"]
    ancestor_fields: list[str] = []
    for key, label, _generation, _side in ANCESTOR_POSITIONS:
        ancestor_fields.extend([f"{label}馬", f"{key}_id", f"{key}_name"])
    write_csv(output_csv, rows, [*base_fields, *ancestor_fields, "source_file"])
    write_csv(
        output_long_csv,
        long_rows,
        [
            "血統登録番号",
            "position_index",
            "position_key",
            "position_label",
            "generation",
            "side",
            "ancestor_id",
            "ancestor_name",
            "source_file",
        ],
    )

    summary["output_csv"] = str(output_csv)
    summary["output_long_csv"] = str(output_long_csv)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
