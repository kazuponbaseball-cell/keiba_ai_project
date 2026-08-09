from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from infrastructure_safety_contract import (  # noqa: E402
    GATE_KIND,
    GitRunner,
    SAFE_EXPERIMENT_ID,
    canonical_digest,
    canonical_json_text,
    load_gate_policy,
    normalize_infrastructure_proposal,
    normalize_infrastructure_queue,
    normalize_infrastructure_run_scope,
    reject_linked_repository_path,
    resolve_repository_path,
    strict_json_load,
    verify_infrastructure_commit_diff,
    verify_infrastructure_run_materials,
)


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and freeze an infrastructure_safety_v1 synthetic run scope. "
            "The command hashes files and re-reads a local committed Git diff only. "
            "It never executes the structured argv, calls an external API, reads real "
            "data, calculates ROI, or reaches production/BUY/order/notification paths."
        )
    )
    parser.add_argument("experiment_id")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--queue-file", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    git_runner: GitRunner | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    experiment_id = args.experiment_id.strip()
    if not SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
        parser.error("experiment_id is not a safe 3-64 character identifier")
    root = args.root.resolve()
    queue_path = resolve_repository_path(
        root, f"research/queue/{experiment_id}.json"
    )
    policy_path = resolve_repository_path(root, "research/INFRASTRUCTURE_GATE.json")
    output_path = resolve_repository_path(
        root, f"research/scopes/{experiment_id}.run.json"
    )
    for requested, expected, option in (
        (args.queue_file, queue_path, "--queue-file"),
        (args.policy, policy_path, "--policy"),
        (args.output, output_path, "--output"),
    ):
        if requested is None:
            continue
        candidate = requested if requested.is_absolute() else root / requested
        if candidate.resolve() != expected.resolve():
            parser.error(f"{option} must name the exact code-owned lifecycle path {expected}")
    try:
        reject_linked_repository_path(root, queue_path, "infrastructure queue")
        reject_linked_repository_path(root, policy_path, "gate policy")
        reject_linked_repository_path(root, output_path, "infrastructure run scope")
        policy, policy_sha256 = load_gate_policy(policy_path)
        queue = normalize_infrastructure_queue(
            strict_json_load(queue_path),
            policy=policy,
            policy_sha256=policy_sha256,
        )
        if queue["experiment_id"] != experiment_id:
            raise ValueError("queue experiment_id mismatch")
        proposal_path = resolve_repository_path(root, queue["proposal_scope_file"])
        proposal = normalize_infrastructure_proposal(
            strict_json_load(proposal_path),
            policy=policy,
            policy_sha256=policy_sha256,
            expected_experiment_id=experiment_id,
        )
        if proposal != queue["proposal_scope"]:
            raise ValueError("queue proposal differs from canonical proposal file")
        if canonical_digest(proposal) != queue["proposal_scope_digest"]:
            raise ValueError("queue proposal digest differs from canonical proposal file")
        run_scope = normalize_infrastructure_run_scope(
            strict_json_load(args.input.resolve()),
            proposal_scope=proposal,
            policy=policy,
            policy_sha256=policy_sha256,
        )
        verify_infrastructure_run_materials(root, run_scope)
        verify_infrastructure_commit_diff(
            root,
            run_scope,
            runner=git_runner,
            policy=policy,
        )
    except ValueError as exc:
        parser.error(str(exc))

    run_digest = canonical_digest(run_scope)
    result = {
        "dry_run": bool(args.dry_run),
        "experiment_id": experiment_id,
        "gate_kind": GATE_KIND,
        "proposal_scope_digest": queue["proposal_scope_digest"],
        "run_scope_digest": run_digest,
        "run_scope_path": str(output_path),
        "execution_kind": "synthetic",
        "commands_executed": 0,
        "execution_authorized": False,
        "external_api_calls": False,
        "real_data_execution": False,
        "roi_calculation": False,
        "production_change": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    if args.dry_run:
        print(
            json.dumps(
                {**result, "run_scope": run_scope},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    if output_path.exists():
        parser.error(f"refusing to overwrite existing run scope: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json_text(run_scope) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.stderr.close()
        raise SystemExit(1)
