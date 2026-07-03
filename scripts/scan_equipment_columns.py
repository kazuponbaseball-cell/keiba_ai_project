from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KEYWORDS = [
    "馬具",
    "ブリン",
    "ブリ",
    "blink",
    "blinker",
    "チーク",
    "シャド",
    "メンコ",
    "馬装",
    "装具",
    "初B",
    "初Ｂ",
    "B着",
    "Ｂ着",
    "B外",
    "Ｂ外",
    "equip",
    "equipment",
]


def _read_header(path: Path) -> list[str]:
    encodings = ["utf-8-sig", "cp932", "utf-8"]
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                sample = f.read(4096)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
                row = next(csv.reader(f, dialect), [])
                return [str(c) for c in row]
        except Exception:
            continue
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan CSV/TSV headers for equipment/blinker-like columns.")
    parser.add_argument("--root", action="append", default=["data", "outputs"])
    parser.add_argument("--output-json", default="outputs/analysis/equipment_column_scan/equipment_column_scan.json")
    parser.add_argument("--output-csv", default="outputs/analysis/equipment_column_scan/equipment_column_scan.csv")
    args = parser.parse_args()

    rows: list[dict] = []
    for root_text in args.root:
        root = Path(root_text)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
                continue
            try:
                if path.stat().st_size == 0:
                    continue
            except OSError:
                continue
            header = _read_header(path)
            if not header:
                continue
            matched = [c for c in header if any(k.lower() in c.lower() for k in KEYWORDS)]
            if matched:
                rows.append({"path": str(path), "columns": matched, "column_count": len(header)})

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "columns", "column_count"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"path": row["path"], "columns": "|".join(row["columns"]), "column_count": row["column_count"]})
    print(json.dumps({"matches": len(rows), "output_json": str(out_json), "output_csv": str(out_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
