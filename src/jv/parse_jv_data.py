from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.paths import ensure_dir, project_path
from src.utils.runtime_config import load_runtime_config


def _latest_raw_run(archive_root: Path) -> Path:
    runs = [path for path in archive_root.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No archived JV raw runs found under {archive_root}")
    return sorted(runs)[-1]


def _infer_kind(path: Path) -> str:
    stem = path.stem.lower()
    if "race" in stem or "kyoso" in stem:
        return "race_like"
    if "horse" in stem or "uma" in stem:
        return "horse_like"
    if "odds" in stem:
        return "odds_like"
    if "weight" in stem or "batai" in stem:
        return "body_weight_like"
    return "unknown"


def parse_latest_archive(config_path: str = "config/data_pipeline.json", run_dir: str | None = None) -> dict[str, object]:
    runtime = load_runtime_config(config_path)
    archive_root = project_path(runtime["jv_data"]["raw_archive_root"])
    parsed_root = ensure_dir(project_path(runtime["jv_data"]["parsed_root"]))
    raw_run = project_path(run_dir) if run_dir else _latest_raw_run(archive_root)
    out_dir = ensure_dir(parsed_root / raw_run.name)

    rows: list[dict[str, object]] = []
    for path in sorted(raw_run.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append(
            {
                "raw_run_id": raw_run.name,
                "relative_path": path.relative_to(raw_run).as_posix(),
                "suffix": path.suffix.lower(),
                "size_bytes": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "inferred_kind": _infer_kind(path),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "file_inventory.csv", index=False, encoding="utf-8-sig")

    summary = {
        "raw_run_dir": str(raw_run),
        "parsed_dir": str(out_dir),
        "files": len(frame),
        "kinds": frame["inferred_kind"].value_counts().to_dict() if not frame.empty else {},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight inventory of the latest archived JV raw run.")
    parser.add_argument("--config", default="config/data_pipeline.json")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    summary = parse_latest_archive(args.config, args.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
