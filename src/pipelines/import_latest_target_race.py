from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.import_target_race_csv import main as import_target_race_main
from src.utils.paths import ensure_dir, project_path
from src.utils.runtime_config import load_runtime_config


def _latest_csv(directory: Path, patterns: list[str]) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(directory.glob(pattern))
    files = [path for path in matches if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No TARGET race CSV files found under {directory}")
    return max(files, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the newest TARGET race-detail CSV from the configured inbox.")
    parser.add_argument("--runtime-config", default="config/data_pipeline.json")
    parser.add_argument("--alias-config", default="config/target_race_aliases.json")
    parser.add_argument("--filter-date", default=None)
    parser.add_argument("--filter-race-id", default=None)
    args = parser.parse_args()

    runtime = load_runtime_config(args.runtime_config)
    target_cfg = runtime["target_import"]
    race_dir = ensure_dir(project_path(target_cfg["race_drop_dir"]))
    latest = _latest_csv(race_dir, list(target_cfg.get("race_file_patterns", ["*.csv"])))

    summary = {
        "selected_input_csv": str(latest),
        "race_drop_dir": str(race_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    import sys

    sys.argv = [
        "import_target_race_csv",
        "--input-csv",
        str(latest),
        "--alias-config",
        args.alias_config,
        *([] if args.filter_date is None else ["--filter-date", args.filter_date]),
        *([] if args.filter_race_id is None else ["--filter-race-id", args.filter_race_id]),
    ]
    import_target_race_main()


if __name__ == "__main__":
    main()
