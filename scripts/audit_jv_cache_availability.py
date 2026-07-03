from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = Path(r"C:\ProgramData\JRA-VAN\Data Lab\cache")
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "analysis" / "jv_cache_availability_v1"


RECORD_LENGTHS = {
    "RA": 1272,
    "SE": 555,
}


CACHE_NAME_RE = re.compile(
    r"^(?P<record_type>[A-Z0-9]{2})(?P<cache_kind>[A-Z]{2})"
    r"(?P<target_date>\d{8})(?P<made_at>\d{14})\.jvd$",
    re.IGNORECASE,
)


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def parse_header_size(raw: bytes) -> int | None:
    header = raw[:10].decode("ascii", errors="ignore").strip()
    if not header.isdigit():
        return None
    return int(header)


def inspect_file(path: Path) -> dict[str, Any]:
    match = CACHE_NAME_RE.match(path.name)
    record_type = match.group("record_type").upper() if match else path.name[:2].upper()
    cache_kind = match.group("cache_kind").upper() if match else ""
    target_date = match.group("target_date") if match else ""
    made_at = match.group("made_at") if match else ""

    row: dict[str, Any] = {
        "file_name": path.name,
        "record_type": record_type,
        "cache_kind": cache_kind,
        "target_date": target_date,
        "made_at": made_at,
        "compressed_bytes": path.stat().st_size,
        "last_write_time": path.stat().st_mtime,
        "header_uncompressed_bytes": None,
        "zlib_ok": False,
        "uncompressed_bytes": None,
        "header_matches": False,
        "record_len": RECORD_LENGTHS.get(record_type),
        "record_count": None,
        "record_count_integral": False,
        "trailing_bytes": None,
        "direct_record_id": "",
        "payload_head_hex": "",
        "payload_printable_pct_200": None,
        "error": "",
    }
    try:
        raw = path.read_bytes()
        row["header_uncompressed_bytes"] = parse_header_size(raw)
        payload = zlib.decompress(raw[10:])
        row["zlib_ok"] = True
        row["uncompressed_bytes"] = len(payload)
        row["header_matches"] = row["header_uncompressed_bytes"] == len(payload) - 1
        row["direct_record_id"] = payload[:2].decode("ascii", errors="ignore")
        row["payload_head_hex"] = payload[:32].hex()
        sample = payload[:200]
        if sample:
            row["payload_printable_pct_200"] = round(sum(32 <= b < 127 for b in sample) / len(sample), 4)
        record_len = row["record_len"]
        if record_len:
            body_len = max(0, len(payload) - 1)
            row["record_count"] = body_len // int(record_len)
            row["trailing_bytes"] = body_len % int(record_len)
            row["record_count_integral"] = row["trailing_bytes"] == 0
    except Exception as exc:  # noqa: BLE001 - audit should keep scanning
        row["error"] = str(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local JRA-VAN Data Lab cache availability.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--types", nargs="*", default=["RA", "SE"])
    args = parser.parse_args()

    cache_dir = project_path(args.cache_dir)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for record_type in [t.upper() for t in args.types]:
        rows.extend(inspect_file(path) for path in sorted(cache_dir.glob(f"{record_type}*.jvd")))

    df = pd.DataFrame(rows)
    inventory_csv = output_dir / "jv_cache_inventory.csv"
    by_date_csv = output_dir / "jv_cache_by_date.csv"
    by_kind_csv = output_dir / "jv_cache_by_kind.csv"
    summary_json = output_dir / "summary.json"

    if df.empty:
        df.to_csv(inventory_csv, index=False, encoding="utf-8-sig")
        summary = {
            "cache_dir": str(cache_dir),
            "files": 0,
            "inventory_csv": str(inventory_csv),
            "by_date_csv": str(by_date_csv),
            "by_kind_csv": str(by_kind_csv),
        }
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    df = df.sort_values(["record_type", "target_date", "cache_kind", "made_at", "file_name"], kind="mergesort")
    df.to_csv(inventory_csv, index=False, encoding="utf-8-sig")

    date_cols = ["record_type", "target_date"]
    by_date = (
        df.groupby(date_cols, dropna=False)
        .agg(
            files=("file_name", "count"),
            cache_kinds=("cache_kind", lambda s: ",".join(sorted(set(x for x in s if x)))),
            total_records=("record_count", "sum"),
            zlib_ok_files=("zlib_ok", "sum"),
            integral_files=("record_count_integral", "sum"),
            compressed_mb=("compressed_bytes", lambda s: round(s.sum() / 1024 / 1024, 3)),
            uncompressed_mb=("uncompressed_bytes", lambda s: round(pd.to_numeric(s, errors="coerce").sum() / 1024 / 1024, 3)),
        )
        .reset_index()
    )
    by_date.to_csv(by_date_csv, index=False, encoding="utf-8-sig")

    by_kind = (
        df.groupby(["record_type", "cache_kind"], dropna=False)
        .agg(
            files=("file_name", "count"),
            dates=("target_date", "nunique"),
            total_records=("record_count", "sum"),
            zlib_ok_files=("zlib_ok", "sum"),
            integral_files=("record_count_integral", "sum"),
            first_date=("target_date", "min"),
            last_date=("target_date", "max"),
        )
        .reset_index()
    )
    by_kind.to_csv(by_kind_csv, index=False, encoding="utf-8-sig")

    summary = {
        "cache_dir": str(cache_dir),
        "files": int(len(df)),
        "types": sorted(df["record_type"].dropna().unique().tolist()),
        "dates": int(df["target_date"].replace("", pd.NA).dropna().nunique()),
        "inventory_csv": str(inventory_csv),
        "by_date_csv": str(by_date_csv),
        "by_kind_csv": str(by_kind_csv),
        "record_type_summary": df.groupby("record_type")["record_count"].sum().fillna(0).astype(int).to_dict(),
        "zlib_failures": int((~df["zlib_ok"]).sum()),
        "non_integral_files": int((~df["record_count_integral"]).sum()),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(by_kind.to_string(index=False))


if __name__ == "__main__":
    main()
