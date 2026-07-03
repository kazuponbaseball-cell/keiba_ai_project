from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


RECORD_LENGTH = 1609
PEDIGREE_PATTERN = re.compile(r"(\d{10,16})([^0-9]{1,60})")


def _clean_name(value: str) -> str:
    return value.replace("\u3000", " ").strip()


def _parse_record(record: bytes, source_file: Path) -> dict[str, str] | None:
    if not record.startswith(b"UM"):
        return None

    text = record.decode("cp932", errors="ignore")
    horse_id = text[11:21].strip()
    if not horse_id.isdigit():
        return None

    pedigree_pairs: list[tuple[str, str]] = []
    for raw_id, raw_name in PEDIGREE_PATTERN.findall(text[120:900]):
        name = _clean_name(raw_name)
        if not name:
            continue
        pedigree_pairs.append((raw_id[-10:], name))

    if len(pedigree_pairs) < 3:
        return None

    names = [name for _ancestor_id, name in pedigree_pairs]
    return {
        "血統登録番号": horse_id,
        "種牡馬": names[0] if len(names) > 0 else "",
        "母馬": names[1] if len(names) > 1 else "",
        "父父馬": names[2] if len(names) > 2 else "",
        "父母馬": names[3] if len(names) > 3 else "",
        "母父馬": names[4] if len(names) > 4 else "",
        "母母馬": names[5] if len(names) > 5 else "",
        "source_file": str(source_file),
    }


def _iter_um_records(target_data_dir: Path):
    for path in sorted(target_data_dir.glob("UM_DATA/*/UM*.DAT")):
        data = path.read_bytes()
        if len(data) < RECORD_LENGTH:
            continue
        for offset in range(0, len(data) - RECORD_LENGTH + 1, RECORD_LENGTH):
            yield path, data[offset : offset + RECORD_LENGTH]


def build_pedigree_master(target_data_dir: Path) -> tuple[list[dict[str, str]], dict[str, int | str]]:
    records_by_horse: dict[str, dict[str, str]] = {}
    scanned_records = 0
    parsed_records = 0

    for path, record in _iter_um_records(target_data_dir):
        scanned_records += 1
        parsed = _parse_record(record, path)
        if not parsed:
            continue
        parsed_records += 1
        records_by_horse[parsed["血統登録番号"]] = parsed

    rows = sorted(records_by_horse.values(), key=lambda row: row["血統登録番号"])
    summary = {
        "target_data_dir": str(target_data_dir),
        "scanned_records": scanned_records,
        "parsed_records": parsed_records,
        "unique_horses": len(rows),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract horse pedigree fields from TARGET UM_DATA master files.")
    parser.add_argument("--target-data-dir", default=r"C:\Users\kazup\Data Lab")
    parser.add_argument("--output-csv", default="data/processed/target/pedigree_master.csv")
    parser.add_argument("--summary-json", default="outputs/analysis/target_pedigree_master_summary.json")
    args = parser.parse_args()

    target_data_dir = Path(args.target_data_dir)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)

    rows, summary = build_pedigree_master(target_data_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["血統登録番号", "種牡馬", "母馬", "母父馬", "母母馬", "父父馬", "父母馬", "source_file"]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary["output_csv"] = str(output_csv)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
