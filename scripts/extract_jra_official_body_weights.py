from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


TAG_RE = re.compile(r"<[^>]+>")
ROW_RE = re.compile(r"<tr\b[^>]*>(?P<body>.*?)</tr>", re.IGNORECASE | re.DOTALL)
NUM_TD_RE = re.compile(r"<td\b[^>]*class=\"[^\"]*\bnum\b[^\"]*\"[^>]*>(?P<body>.*?)</td>", re.IGNORECASE | re.DOTALL)
BODY_TD_RE = re.compile(r"<td\b[^>]*class=\"[^\"]*\bh_weight\b[^\"]*\"[^>]*>(?P<body>.*?)</td>", re.IGNORECASE | re.DOTALL)


def strip_tags(value: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", value)).strip()


def num_text(value: str) -> float | None:
    match = re.search(r"[+-]?\d+(?:\.\d+)?", re.sub(r"\s+", "", str(value or "")))
    return float(match.group(0)) if match else None


def parse_snapshot_from_path(path: Path) -> str:
    match = re.match(r"(?P<snapshot>\d{8}_\d{6})_win_place_frame\.html$", path.name)
    return match.group("snapshot") if match else ""


def parse_body_page(path: Path, race_id: str) -> list[dict]:
    raw = path.read_text(encoding="cp932", errors="replace")
    snapshot_at = parse_snapshot_from_path(path)
    rows: list[dict] = []
    for row_match in ROW_RE.finditer(raw):
        tr = row_match.group("body")
        num_match = NUM_TD_RE.search(tr)
        weight_match = BODY_TD_RE.search(tr)
        if not num_match or not weight_match:
            continue
        horse_no = num_text(strip_tags(num_match.group("body")))
        weight_text = strip_tags(weight_match.group("body"))
        body_weight = num_text(weight_text)
        if horse_no is None or body_weight is None:
            continue
        diff_match = re.search(r"\(([+-]?\d+)\)", weight_text)
        body_weight_diff = float(diff_match.group(1)) if diff_match else pd.NA
        rows.append(
            {
                "race_id": race_id,
                "horse_no": int(horse_no),
                "body_weight": int(body_weight),
                "body_weight_diff": body_weight_diff,
                "snapshot_at": snapshot_at,
                "source": "jra_official_win_place_frame",
                "raw_path": str(path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract live body weights from cached JRA official win/place odds HTML.")
    parser.add_argument("--raw-dir", default="data/raw/jra_official_odds")
    parser.add_argument("--date", default="", help="Optional YYYYMMDD filter.")
    parser.add_argument("--output-csv", default="data/processed/live_body_weight/body_weight_latest.csv")
    parser.add_argument("--summary-json", default="outputs/analysis/live_body_weight/jra_official_body_weight_summary.json")
    args = parser.parse_args()

    raw_dir = project_path(args.raw_dir)
    rows: list[dict] = []
    if raw_dir.exists():
        for race_dir in raw_dir.iterdir():
            if not race_dir.is_dir():
                continue
            race_id = race_dir.name
            if args.date and not race_id.startswith(args.date):
                continue
            files = sorted(race_dir.glob("*_win_place_frame.html"))
            if not files:
                continue
            # The raw cache may contain many snapshots per race; only the newest page matters for live operation.
            rows.extend(parse_body_page(files[-1], race_id))

    cols = ["race_id", "horse_no", "body_weight", "body_weight_diff", "snapshot_at", "source", "raw_path"]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out["race_id"] = out["race_id"].astype(str)
        out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce").astype("Int64")
        out["body_weight"] = pd.to_numeric(out["body_weight"], errors="coerce")
        out["body_weight_diff"] = pd.to_numeric(out["body_weight_diff"], errors="coerce")
        out = out[out["race_id"].ne("") & out["horse_no"].notna() & out["body_weight"].notna()].copy()
        out = out.sort_values(["race_id", "horse_no", "snapshot_at"])
        out = out.drop_duplicates(["race_id", "horse_no"], keep="last")

    output = project_path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")

    summary = {
        "output_csv": str(output),
        "rows": int(len(out)),
        "races": int(out["race_id"].nunique()) if not out.empty else 0,
        "date": args.date,
        "raw_dir": str(raw_dir),
    }
    summary_path = project_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
