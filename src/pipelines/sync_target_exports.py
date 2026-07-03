from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from src.data.loaders import (
    inference_optional_columns,
    inference_required_columns,
    load_json_config,
)
from src.pipelines.import_latest_target_entry import main as import_latest_entry_main
from src.pipelines.import_latest_target_race import main as import_latest_race_main
from src.pipelines.import_target_fixed_entry_csv import looks_like_fixed_entry_csv
from src.pipelines.import_target_entry_csv import load_aliases as load_entry_aliases
from src.pipelines.import_target_race_csv import TARGET_RACE_COLUMNS, load_aliases as load_race_aliases
from src.utils.paths import ensure_dir, project_path
from src.utils.runtime_config import load_runtime_config


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_path(path)


def _read_header(path: Path, encodings: list[str]) -> list[str] | None:
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return next(csv.reader(f))
        except (StopIteration, UnicodeDecodeError, OSError, csv.Error):
            continue
    return None


def _expected_columns(mode: str, runtime: dict, alias_config: str | None) -> set[str]:
    if mode == "entry":
        feature_config = load_json_config(runtime["pipeline"]["baseline_feature_config"])
        aliases = load_entry_aliases(alias_config or "config/target_entry_aliases.json")
        columns = [
            *inference_required_columns(feature_config),
            *inference_optional_columns(feature_config),
        ]
    else:
        aliases = load_race_aliases(alias_config or "config/target_race_aliases.json")
        columns = TARGET_RACE_COLUMNS

    expected: set[str] = set(columns)
    for column in columns:
        expected.update(aliases.get(column, []))
    return expected


def _score_header(header: list[str] | None, expected: set[str]) -> int:
    if not header:
        return 0
    return len(set(header) & expected)


def _find_latest_export(
    source_dirs: list[Path],
    patterns: list[str],
    expected: set[str],
    min_column_matches: int,
    encodings: list[str],
    modified_after: datetime | None,
) -> tuple[Path, list[str], int]:
    candidates: list[Path] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for pattern in patterns:
            for path in source_dir.glob(pattern):
                if not path.is_file():
                    continue
                if modified_after and datetime.fromtimestamp(path.stat().st_mtime) < modified_after:
                    continue
                candidates.append(path)

    checked: list[tuple[Path, list[str] | None, int]] = []
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        if looks_like_fixed_entry_csv(path):
            return path, [], min_column_matches
        header = _read_header(path, encodings)
        score = _score_header(header, expected)
        checked.append((path, header, score))
        if score >= min_column_matches:
            return path, header or [], score

    detail = [
        {"path": str(path), "matched_columns": score}
        for path, _header, score in checked[:10]
    ]
    raise FileNotFoundError(
        "No TARGET CSV export matched the expected columns. "
        + json.dumps(detail, ensure_ascii=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the newest TARGET CSV export into the project inbox.")
    parser.add_argument("--mode", choices=["entry", "race"], default="entry")
    parser.add_argument("--runtime-config", default="config/data_pipeline.json")
    parser.add_argument("--alias-config", default=None)
    parser.add_argument("--source-dir", action="append", default=None)
    parser.add_argument("--pattern", action="append", default=None)
    parser.add_argument("--min-column-matches", type=int, default=5)
    parser.add_argument("--max-age-days", type=int, default=None)
    parser.add_argument("--allow-old", action="store_true")
    parser.add_argument("--run-import", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runtime = load_runtime_config(args.runtime_config)
    target_cfg = runtime["target_import"]
    source_dirs = [
        _resolve_path(path)
        for path in (args.source_dir or target_cfg.get("source_dirs", []))
    ]
    if not source_dirs:
        raise ValueError("No TARGET source directories configured.")

    patterns = args.pattern or target_cfg.get(f"{args.mode}_file_patterns", ["*.csv"])
    max_age_days = args.max_age_days
    if max_age_days is None:
        max_age_days = int(target_cfg.get("source_max_age_days", 14))
    modified_after = None if args.allow_old else datetime.now() - timedelta(days=max_age_days)
    expected = _expected_columns(args.mode, runtime, args.alias_config)
    selected, header, score = _find_latest_export(
        source_dirs,
        patterns,
        expected,
        args.min_column_matches,
        ["cp932", "utf-8-sig", "utf-8"],
        modified_after,
    )

    drop_key = "entry_drop_dir" if args.mode == "entry" else "race_drop_dir"
    drop_dir = ensure_dir(project_path(target_cfg[drop_key]))
    destination = drop_dir / selected.name

    summary = {
        "mode": args.mode,
        "selected_source_csv": str(selected),
        "destination_csv": str(destination),
        "matched_columns": score,
        "header_columns": len(header),
        "modified_after": None if modified_after is None else modified_after.isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        shutil.copy2(selected, destination)
        if args.run_import:
            if args.mode == "entry" and looks_like_fixed_entry_csv(selected):
                import sys

                sys.argv = [
                    "import_target_fixed_entry_csv",
                    "--input-csv",
                    str(destination),
                    "--runtime-config",
                    args.runtime_config,
                ]
                from src.pipelines.import_target_fixed_entry_csv import main as import_fixed_entry_main

                import_fixed_entry_main()
            else:
                import_latest_entry_main() if args.mode == "entry" else import_latest_race_main()

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
