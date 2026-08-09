from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from infrastructure_safety_contract import (  # noqa: E402
    EVENT_SCHEMA_VERSION,
    GATE_KIND,
    QUEUE_SCHEMA_VERSION,
    SAFE_EXPERIMENT_ID,
    build_infrastructure_queue,
    canonical_digest,
    canonical_json_text,
    evaluate_infrastructure_gate,
    load_gate_policy,
    normalize_infrastructure_proposal,
    normalize_infrastructure_queue,
    reject_linked_repository_path,
    resolve_repository_path,
    strict_json_load,
)


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside repository root: {path}") from exc


def _render_markdown(
    template: str,
    *,
    proposal: dict[str, Any],
    proposal_path: str,
    proposal_digest: str,
    owner: str,
    created_at: str,
) -> str:
    path_lines = "\n".join(f"- `{path}`" for path in proposal["expected_changed_paths"])
    test_lines = ["| Test ID | Kind | Assertion |", "|---|---|---|"]
    for case in proposal["test_matrix"]:
        assertion = case["assertion"].replace("|", "\\|").replace("\n", " ")
        test_lines.append(f"| `{case['test_id']}` | `{case['kind']}` | {assertion} |")
    replacements = {
        "{{EXPERIMENT_ID}}": proposal["experiment_id"],
        "{{TITLE}}": proposal["title"],
        "{{STATUS}}": "proposed",
        "{{OWNER}}": owner,
        "{{CREATED_AT}}": created_at,
        "{{BASE_COMMIT}}": proposal["base_commit"],
        "{{SOURCE_AS_OF}}": proposal["source_as_of"],
        "{{GATE_POLICY_SHA256}}": proposal["gate_policy"]["sha256"],
        "{{PROPOSAL_SCOPE_PATH}}": proposal_path,
        "{{PROPOSAL_SCOPE_DIGEST}}": proposal_digest,
        "{{CHANGE_HYPOTHESIS}}": proposal["change_hypothesis"],
        "{{NULL_HYPOTHESIS}}": proposal["null_hypothesis"],
        "{{SAFETY_OBJECTIVE}}": proposal["safety_objective"],
        "{{EXPECTED_CHANGED_PATHS}}": path_lines,
        "{{TEST_MATRIX}}": "\n".join(test_lines),
    }
    rendered = template
    for marker, replacement in replacements.items():
        rendered = rendered.replace(marker, str(replacement))
    unresolved = sorted(set(part.split("}}", 1)[0] for part in rendered.split("{{")[1:]))
    if unresolved:
        raise ValueError("template contains unresolved marker(s): " + ", ".join(unresolved))
    return rendered.rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a schema-v3 Research OS infrastructure safety proposal. "
            "The command validates and writes local JSON/Markdown only; it never "
            "calls an external API, executes tests, grants approval, reads real data, "
            "or reaches production/BUY/order/notification paths."
        )
    )
    parser.add_argument("experiment_id")
    parser.add_argument("--proposal-scope", type=Path, required=True)
    parser.add_argument("--owner", default="unassigned")
    parser.add_argument("--notes", default="")
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    experiment_id = args.experiment_id.strip()
    if not SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
        parser.error("experiment_id is not a safe 3-64 character identifier")
    root = args.root.resolve()
    policy_path = resolve_repository_path(root, "research/INFRASTRUCTURE_GATE.json")
    template_path = resolve_repository_path(
        root, "research/INFRASTRUCTURE_EXPERIMENT_TEMPLATE.md"
    )
    for requested, expected, option in (
        (args.policy, policy_path, "--policy"),
        (args.template, template_path, "--template"),
    ):
        if requested is None:
            continue
        candidate = requested if requested.is_absolute() else root / requested
        if candidate.resolve() != expected.resolve():
            parser.error(f"{option} must name the exact code-owned path {expected}")
    try:
        reject_linked_repository_path(root, policy_path, "gate policy")
        reject_linked_repository_path(root, template_path, "infrastructure template")
        policy, policy_sha256 = load_gate_policy(policy_path)
        proposal = normalize_infrastructure_proposal(
            strict_json_load(args.proposal_scope.resolve()),
            policy=policy,
            policy_sha256=policy_sha256,
            expected_experiment_id=experiment_id,
        )
        gate_evaluation = evaluate_infrastructure_gate(proposal, policy=policy)
        if gate_evaluation["passed"] is not True:
            raise ValueError("infrastructure proposal did not pass every hard check")
        if not template_path.is_file():
            raise ValueError(f"infrastructure template not found: {template_path}")
    except ValueError as exc:
        parser.error(str(exc))

    research_dir = root / "research"
    scope_path = research_dir / "scopes" / f"{experiment_id}.proposal.json"
    experiment_path = research_dir / "experiments" / f"{experiment_id}.md"
    queue_path = research_dir / "queue" / f"{experiment_id}.json"
    try:
        for path, field in (
            (scope_path, "canonical proposal output"),
            (experiment_path, "infrastructure experiment output"),
            (queue_path, "infrastructure queue output"),
        ):
            reject_linked_repository_path(root, path, field)
    except ValueError as exc:
        parser.error(str(exc))
    collisions = [path for path in (scope_path, experiment_path, queue_path) if path.exists()]
    if collisions:
        parser.error(
            "refusing to overwrite existing file(s): " + ", ".join(str(path) for path in collisions)
        )
    created_at = utc_now()
    owner = args.owner.strip() or "unassigned"
    proposal_digest = canonical_digest(proposal)
    try:
        scope_relative = _relative_path(scope_path, root)
        experiment_relative = _relative_path(experiment_path, root)
        queue = build_infrastructure_queue(
            proposal_scope=proposal,
            proposal_scope_file=scope_relative,
            experiment_markdown=experiment_relative,
            owner=owner,
            created_at=created_at,
            notes=args.notes,
            policy=policy,
        )
        queue = normalize_infrastructure_queue(
            queue,
            policy=policy,
            policy_sha256=policy_sha256,
        )
        markdown = _render_markdown(
            template_path.read_text(encoding="utf-8"),
            proposal=proposal,
            proposal_path=scope_relative,
            proposal_digest=proposal_digest,
            owner=owner,
            created_at=created_at,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        parser.error(str(exc))

    result = {
        "dry_run": bool(args.dry_run),
        "experiment_id": experiment_id,
        "status": "proposed",
        "gate_kind": GATE_KIND,
        "gate_passed": True,
        "queue_schema_version": QUEUE_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "proposal_scope_digest": proposal_digest,
        "numeric_score": None,
        "automatic_execution_allowed": False,
        "execution_authorized": False,
        "external_api_calls": False,
        "real_data_execution": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "proposal_scope_path": str(scope_path),
        "experiment_path": str(experiment_path),
        "queue_path": str(queue_path),
    }
    if args.dry_run:
        print(
            json.dumps(
                {**result, "gate_evaluation": gate_evaluation, "queue": queue},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    for directory in (scope_path.parent, experiment_path.parent, queue_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with scope_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json_text(proposal) + "\n")
        created.append(scope_path)
        with experiment_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(markdown)
        created.append(experiment_path)
        with queue_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(queue, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        created.append(queue_path)
    except (FileExistsError, OSError):
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.stderr.close()
        raise SystemExit(1)
