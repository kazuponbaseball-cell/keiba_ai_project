from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scope_contract import (  # noqa: E402
    SAFE_EXPERIMENT_ID,
    canonical_digest,
    canonical_json_text,
    load_frozen_proposal,
    normalize_run_scope,
    strict_json_load,
    verify_run_materials,
)


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def current_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot resolve execution commit: {completed.stderr.strip()}")
    return completed.stdout.strip().lower()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and freeze an exact run scope after implementation. "
            "This command reads manifests and hashes only; it never trains, "
            "backtests, evaluates ROI, or authorizes execution."
        )
    )
    parser.add_argument("experiment_id", help="Existing experiment identifier.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Draft strict JSON run scope.",
    )
    parser.add_argument("--root", type=Path, default=default_root(), help="Repository root.")
    parser.add_argument(
        "--queue-file",
        type=Path,
        default=None,
        help="Queue JSON; defaults to research/queue/<experiment_id>.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Canonical output; defaults to research/scopes/<experiment_id>.run.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print only.")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    execution_commit_provider=current_commit,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    experiment_id = args.experiment_id.strip()
    if not SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
        parser.error("experiment_id is not a safe 3-64 character identifier")
    root = args.root.resolve()
    queue_path = (
        args.queue_file.resolve()
        if args.queue_file is not None
        else root / "research" / "queue" / f"{experiment_id}.json"
    )
    if not queue_path.is_file():
        parser.error(f"queue record not found: {queue_path}")
    try:
        queue = strict_json_load(queue_path)
        if not isinstance(queue, dict):
            raise ValueError("queue record must be a JSON object")
        proposal_scope, proposal_digest, _ = load_frozen_proposal(
            root, queue, experiment_id
        )
        run_scope = normalize_run_scope(
            strict_json_load(args.input.resolve()),
            proposal_scope=proposal_scope,
        )
        observed_commit = execution_commit_provider(root)
        if run_scope["execution_commit_sha"] != observed_commit:
            raise ValueError(
                "execution_commit_sha does not equal current HEAD: "
                f"{run_scope['execution_commit_sha']} != {observed_commit}"
            )
        verify_run_materials(root, run_scope)
    except ValueError as exc:
        parser.error(str(exc))

    digest = canonical_digest(run_scope)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else root / "research" / "scopes" / f"{experiment_id}.run.json"
    )
    result = {
        "dry_run": bool(args.dry_run),
        "experiment_id": experiment_id,
        "proposal_scope_digest": proposal_digest,
        "run_scope_digest": digest,
        "run_scope_path": str(output_path),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "real_data_execution_authorized": False,
    }
    if args.dry_run:
        print(json.dumps({**result, "run_scope": run_scope}, ensure_ascii=False, indent=2))
        return 0
    if output_path.exists():
        parser.error(f"refusing to overwrite existing run scope: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json_text(run_scope) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.stderr.close()
        raise SystemExit(1)
