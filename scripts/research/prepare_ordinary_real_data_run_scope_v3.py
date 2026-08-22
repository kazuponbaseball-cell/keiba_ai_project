#!/usr/bin/env python3
"""Compile an ordinary_real_data_run_v3 scope without opening real rows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ordinary_real_data_run_contract_v3 import (  # noqa: E402
    RUN_SCOPE_SCHEMA_VERSION,
    normalize_ordinary_real_data_run_scope,
    verify_ordinary_real_data_run_materials,
)
from scope_contract import (  # noqa: E402
    canonical_digest,
    canonical_json_bytes,
    load_frozen_proposal,
    strict_json_load,
)


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _current_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot resolve execution commit: {completed.stderr.strip()}")
    return completed.stdout.strip().lower()


def _resolve(root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else root / value
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {value}") from exc
    return resolved


def _reject_link(path: Path, root: Path, label: str) -> None:
    relative = path.resolve().relative_to(root.resolve())
    current = root.resolve()
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"{label} cannot use a linked repository path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a canonical ordinary_real_data_run_v3 scope. This compiler "
            "verifies code and metadata only; it never opens a real-data row/blob."
        )
    )
    parser.add_argument("experiment_id")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--queue-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    experiment_id = args.experiment_id.strip()
    if not experiment_id:
        parser.error("experiment_id must not be blank")
    root = args.root.resolve()
    input_path = _resolve(root, args.input)
    queue_path = _resolve(
        root,
        args.queue_file
        if args.queue_file is not None
        else Path("research/queue") / f"{experiment_id}.json",
    )
    output_path = _resolve(
        root,
        args.output
        if args.output is not None
        else Path("research/scopes") / f"{experiment_id}.run.json",
    )
    expected_output = root / "research" / "scopes" / f"{experiment_id}.run.json"
    if output_path != expected_output.resolve():
        parser.error(
            "v3 canonical run scope must use research/scopes/<experiment_id>.run.json"
        )
    try:
        _reject_link(input_path, root, "v3 input")
        _reject_link(queue_path, root, "queue")
        _reject_link(output_path.parent, root, "v3 output")
        queue = strict_json_load(queue_path)
        if not isinstance(queue, dict):
            raise ValueError("queue must be a JSON object")
        proposal_scope, _, _ = load_frozen_proposal(root, queue, experiment_id)
        raw_scope = strict_json_load(input_path)
        scope = normalize_ordinary_real_data_run_scope(
            raw_scope,
            proposal_scope=proposal_scope,
        )
        if scope["run_scope_schema_version"] != RUN_SCOPE_SCHEMA_VERSION:
            raise ValueError("v3 compiler received a non-v3 run scope")
        observed_commit = _current_commit(root)
        if scope["execution_commit_sha"] != observed_commit:
            raise ValueError(
                "execution_commit_sha must equal current HEAD when freezing a v3 scope"
            )
        preflight = verify_ordinary_real_data_run_materials(root, scope)
        if preflight.get("real_data_rows_opened") != 0:
            raise ValueError("metadata-only compiler opened a real-data row; fail-close")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    digest = canonical_digest(scope)
    payload = canonical_json_bytes(scope) + b"\n"
    result = {
        "run_scope_schema_version": RUN_SCOPE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "output": output_path.relative_to(root).as_posix(),
        "run_scope_digest": digest,
        "execution_kind": scope["execution_kind"],
        "capability_profile_id": scope["capability_profile"]["profile_id"],
        "real_data_rows_opened": 0,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if output_path.exists():
        parser.error(f"refusing to overwrite existing v3 run scope: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(output_path, flags, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        parser.error(f"cannot atomically create v3 run scope: {exc}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
