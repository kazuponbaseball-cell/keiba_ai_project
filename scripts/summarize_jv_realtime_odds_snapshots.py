from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize JV realtime odds raw snapshot metadata.")
    parser.add_argument("--raw-dir", default="data/raw/jv_realtime_odds")
    parser.add_argument("--output-dir", default="outputs/analysis/jv_realtime_odds_snapshots")
    args = parser.parse_args()

    raw_dir = project_path(args.raw_dir)
    rows = []
    expected_cols = [
        "race_key",
        "bet_type",
        "dataspec",
        "snapshot_at",
        "init_return",
        "open_return",
        "record_count",
        "error",
        "raw_path",
        "meta_path",
    ]
    for meta_path in raw_dir.glob("*/*.json"):
        try:
            text = None
            for encoding in ["utf-8-sig", "utf-16", "cp932"]:
                try:
                    text = meta_path.read_text(encoding=encoding)
                    break
                except UnicodeError:
                    continue
            if text is None:
                text = meta_path.read_text()
            meta = json.loads(text)
        except Exception as exc:
            row = {col: None for col in expected_cols}
            row.update({"meta_path": str(meta_path), "error": f"json_read_failed: {exc}"})
            rows.append(row)
            continue
        rows.append(
            {
                "race_key": meta.get("race_key"),
                "bet_type": meta.get("bet_type"),
                "dataspec": meta.get("dataspec"),
                "snapshot_at": meta.get("snapshot_at"),
                "init_return": meta.get("init_return"),
                "open_return": meta.get("open_return"),
                "record_count": meta.get("record_count"),
                "error": meta.get("error"),
                "raw_path": meta.get("raw_path"),
                "meta_path": str(meta_path),
            }
        )
    out = pd.DataFrame(rows)
    out_dir = ensure_dir(project_path(args.output_dir))
    if out.empty:
        out = pd.DataFrame(columns=expected_cols)
    for col in expected_cols:
        if col not in out.columns:
            out[col] = None
    out = out.sort_values(["snapshot_at", "race_key", "bet_type"], na_position="last")
    out.to_csv(out_dir / "snapshot_metadata_summary.csv", index=False, encoding="utf-8-sig")
    status = (
        out.groupby(["bet_type", "open_return", "error"], dropna=False)
        .size()
        .rename("snapshots")
        .reset_index()
        .sort_values("snapshots", ascending=False)
    )
    status.to_csv(out_dir / "snapshot_status_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(out_dir),
        "snapshots": int(len(out)),
        "successful_snapshots": int((pd.to_numeric(out["record_count"], errors="coerce").fillna(0) > 0).sum()),
        "status_summary": status.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
