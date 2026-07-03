from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RECORD_LENGTH = 1609
HORSE_ID_SLICE = slice(11, 21)
BREEDER_CODE_SLICE = slice(882, 890)
BREEDER_NAME_SLICE = slice(890, 960)
BIRTHPLACE_SLICE = slice(960, 982)


def _clean_text(raw: bytes) -> str:
    return raw.decode("cp932", errors="ignore").replace("\u3000", " ").strip()


def _parse_record(record: bytes, source_file: Path) -> dict[str, str] | None:
    if not record.startswith(b"UM"):
        return None
    horse_id = record[HORSE_ID_SLICE].decode("ascii", errors="ignore").strip()
    if not horse_id.isdigit():
        return None
    breeder_code = record[BREEDER_CODE_SLICE].decode("ascii", errors="ignore").strip()
    breeder_name = _clean_text(record[BREEDER_NAME_SLICE])
    birthplace = _clean_text(record[BIRTHPLACE_SLICE])
    if not breeder_code and not breeder_name:
        return None
    return {
        "血統登録番号": horse_id,
        "breeder_code": breeder_code,
        "breeder_name": breeder_name,
        "birthplace": birthplace,
        "source_file": str(source_file),
    }


def _iter_um_records(target_data_dir: Path):
    for path in sorted(target_data_dir.glob("UM_DATA/*/UM*.DAT")):
        data = path.read_bytes()
        if len(data) < RECORD_LENGTH:
            continue
        for offset in range(0, len(data) - RECORD_LENGTH + 1, RECORD_LENGTH):
            yield path, data[offset : offset + RECORD_LENGTH]


def build_breeder_master(target_data_dir: Path) -> tuple[list[dict[str, str]], dict[str, int | str]]:
    records_by_horse: dict[str, dict[str, str]] = {}
    scanned_records = 0
    parsed_records = 0
    with_breeder_name = 0

    for path, record in _iter_um_records(target_data_dir):
        scanned_records += 1
        parsed = _parse_record(record, path)
        if not parsed:
            continue
        parsed_records += 1
        if parsed["breeder_name"]:
            with_breeder_name += 1
        records_by_horse[parsed["血統登録番号"]] = parsed

    rows = sorted(records_by_horse.values(), key=lambda row: row["血統登録番号"])
    summary = {
        "target_data_dir": str(target_data_dir),
        "scanned_records": scanned_records,
        "parsed_records": parsed_records,
        "unique_horses": len(rows),
        "with_breeder_name": with_breeder_name,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract breeder code/name from TARGET UM_DATA horse master files.")
    parser.add_argument("--target-data-dir", default=r"C:\Users\kazup\Data Lab")
    parser.add_argument("--output-csv", default="data/processed/target/breeder_master.csv")
    parser.add_argument("--summary-json", default="outputs/analysis/target_breeder_master_summary.json")
    args = parser.parse_args()

    rows, summary = build_breeder_master(Path(args.target_data_dir))
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["血統登録番号", "breeder_code", "breeder_name", "birthplace", "source_file"]
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
