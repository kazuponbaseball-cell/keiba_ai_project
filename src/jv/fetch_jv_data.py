from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from datetime import datetime
from pathlib import Path

from src.jv.adapters import prepare_source_drop
from src.utils.paths import ensure_dir, project_path
from src.utils.runtime_config import load_runtime_config


def _load_state(path: Path) -> dict[str, dict[str, float | int]]:
    if not path.exists():
        return {"files": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: Path, state: dict[str, dict[str, float | int]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _matching_files(source_dir: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(fnmatch.fnmatch(path.name.lower(), pattern.lower()) for pattern in patterns):
            files.append(path)
    return sorted(files)


def snapshot_jv_drop(config_path: str = "config/data_pipeline.json") -> dict[str, object]:
    runtime = load_runtime_config(config_path)
    jv_cfg = runtime["jv_data"]
    provider_result = prepare_source_drop(config_path)
    source_dir = project_path(jv_cfg["source_drop_dir"])
    archive_root = ensure_dir(project_path(jv_cfg["raw_archive_root"]))
    state_path = project_path(runtime["pipeline"]["state_dir"], "jv_fetch_state.json")

    if not source_dir.exists():
        raise FileNotFoundError(
            f"JV source drop directory not found: {source_dir}. "
            "Create it and place raw JV exports there before running this pipeline."
        )

    state = _load_state(state_path)
    candidates = _matching_files(source_dir, list(jv_cfg.get("file_patterns", ["*"])))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(archive_root / run_id)

    copied: list[dict[str, object]] = []
    unchanged = 0
    for src in candidates:
        rel = src.relative_to(source_dir).as_posix()
        stat = src.stat()
        signature = {
            "size": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        }
        if state["files"].get(rel) == signature:
            unchanged += 1
            continue

        dst = ensure_dir(run_dir / Path(rel).parent) / src.name
        shutil.copy2(src, dst)
        state["files"][rel] = signature
        copied.append(
            {
                "relative_path": rel,
                "archived_path": str(dst),
                "size": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )

    manifest = {
        "run_id": run_id,
        "provider": provider_result,
        "source_dir": str(source_dir),
        "archive_dir": str(run_dir),
        "matched_files": len(candidates),
        "copied_files": len(copied),
        "unchanged_files": unchanged,
        "copied": copied,
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _save_state(state_path, state)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive new or updated JV raw files from a local drop directory.")
    parser.add_argument("--config", default="config/data_pipeline.json")
    args = parser.parse_args()
    manifest = snapshot_jv_drop(args.config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
