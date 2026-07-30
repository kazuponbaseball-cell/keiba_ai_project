from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCORE_WEIGHTS = {
    "independent_information": 25,
    "racing_mechanism": 20,
    "outer_oos_failure_evidence": 20,
    "leakage_safety": 15,
    "minimal_falsifiability": 10,
    "acquisition_implementation_cost": 10,
}
RUN_SCORE_THRESHOLD = 75
SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
ALLOWED_STATUSES = {
    "proposed",
    "blocked_score",
    "approved_to_run",
    "running",
    "review_required",
    "rejected",
    "approved_for_shadow",
    "archived",
    "invalid",
}
TRANSITIONS: dict[str | None, set[str]] = {
    None: {"proposed", "blocked_score", "invalid"},
    "blocked_score": {"blocked_score", "archived", "invalid"},
    "proposed": {"proposed", "approved_to_run", "invalid"},
    "approved_to_run": {"approved_to_run", "running", "invalid"},
    "running": {"review_required", "invalid"},
    "review_required": {"review_required", "rejected", "approved_for_shadow", "invalid"},
    "rejected": {"rejected", "invalid"},
    "approved_for_shadow": {"approved_for_shadow", "archived", "invalid"},
    "archived": {"archived", "invalid"},
    "invalid": {"invalid"},
}


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_status(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def validate_experiment_id(value: str) -> str:
    if not SAFE_EXPERIMENT_ID.fullmatch(value):
        raise ValueError(
            "experiment_id must be 3-64 characters and contain only ASCII letters, "
            "digits, '_' or '-'; it must start with a letter or digit"
        )
    return value


def resolve_from_root(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def load_queue(path: Path, experiment_id: str) -> tuple[dict[str, Any], dict[str, int], int]:
    if not path.is_file():
        raise ValueError(f"queue record not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read queue record {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"queue record must be a JSON object: {path}")
    if payload.get("experiment_id") != experiment_id:
        raise ValueError(
            f"queue experiment_id mismatch: expected {experiment_id!r}, "
            f"found {payload.get('experiment_id')!r}"
        )

    for flag in (
        "human_approved_to_run",
        "automatic_execution_allowed",
        "execution_authorized",
        "production_approved",
        "merge_approved",
        "buy_approved",
        "production_change_allowed",
        "merge_allowed",
        "buy_logic_change_allowed",
        "formal_buy",
        "send_order",
    ):
        if payload.get(flag) is not False:
            raise ValueError(f"queue safety flag {flag!r} must remain false before registry approval")
    if payload.get("stake") != 0:
        raise ValueError("queue safety field 'stake' must remain 0")

    score_block = payload.get("score")
    components_raw = score_block.get("components") if isinstance(score_block, dict) else None
    if not isinstance(components_raw, dict):
        raise ValueError("queue record is missing score.components")
    components: dict[str, int] = {}
    for name, weight in SCORE_WEIGHTS.items():
        value = components_raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"score component {name!r} must be an integer")
        if not 0 <= value <= weight:
            raise ValueError(f"score component {name!r} must be between 0 and {weight}")
        components[name] = value
    unexpected = sorted(set(components_raw) - set(SCORE_WEIGHTS))
    if unexpected:
        raise ValueError("queue record contains unknown score component(s): " + ", ".join(unexpected))
    total = sum(components.values())
    return payload, components, total


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"registry event at {path}:{line_number} is not an object")
            if not isinstance(event.get("experiment_id"), str) or not isinstance(event.get("status"), str):
                raise ValueError(f"registry event at {path}:{line_number} lacks experiment_id/status")
            events.append(event)
    return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append one validated state event to the Level-3 JSONL registry. "
            "Production, merge, and BUY approvals are deliberately out of scope."
        )
    )
    parser.add_argument("experiment_id", help="Existing experiment queue identifier.")
    parser.add_argument("status", help="New state (for example PROPOSED or APPROVED_TO_RUN).")
    parser.add_argument(
        "--human-approved",
        action="store_true",
        help="Required for APPROVED_TO_RUN and APPROVED_FOR_SHADOW.",
    )
    parser.add_argument("--actor", default=getpass.getuser(), help="Human or service appending the event.")
    parser.add_argument("--artifact", action="append", default=[], help="Artifact reference; may be repeated.")
    parser.add_argument("--notes", default="", help="Review or transition notes.")
    parser.add_argument("--root", type=Path, default=default_root(), help="Repository root.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("research/REGISTRY.jsonl"),
        help="Append-only JSONL registry path, relative to --root unless absolute.",
    )
    parser.add_argument(
        "--queue-file",
        type=Path,
        default=None,
        help="Queue JSON path; defaults to research/queue/<experiment_id>.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the event without appending it.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        experiment_id = validate_experiment_id(args.experiment_id)
    except ValueError as exc:
        parser.error(str(exc))

    raw_status = args.status.strip()
    status = normalize_status(raw_status)
    if status not in ALLOWED_STATUSES:
        prohibited_hint = "production/merge/BUY approval is outside this registry's authority"
        parser.error(
            f"unsupported status {raw_status!r}; allowed: {', '.join(sorted(ALLOWED_STATUSES))}. "
            f"{prohibited_hint}"
        )
    human_approval_statuses = {"approved_to_run", "approved_for_shadow"}
    if args.human_approved and status not in human_approval_statuses:
        parser.error("--human-approved is valid only for APPROVED_TO_RUN or APPROVED_FOR_SHADOW")

    actor = args.actor.strip()
    if not actor:
        parser.error("--actor must not be blank")
    artifacts = [value.strip() for value in args.artifact]
    if any(not value for value in artifacts):
        parser.error("--artifact values must not be blank")

    root = args.root.resolve()
    registry_path = resolve_from_root(root, args.registry)
    queue_path = (
        resolve_from_root(root, args.queue_file)
        if args.queue_file is not None
        else root / "research" / "queue" / f"{experiment_id}.json"
    )
    try:
        _queue, components, total = load_queue(queue_path, experiment_id)
        events = load_events(registry_path)
    except ValueError as exc:
        parser.error(str(exc))

    history = [event for event in events if event["experiment_id"] == experiment_id]
    previous = history[-1] if history else None
    previous_status = normalize_status(str(previous["status"])) if previous else None
    if previous_status not in TRANSITIONS:
        parser.error(f"registry has unknown previous state for {experiment_id}: {previous_status!r}")
    if status not in TRANSITIONS[previous_status]:
        parser.error(f"invalid transition for {experiment_id}: {previous_status or '<none>'} -> {status}")

    threshold_met = total >= RUN_SCORE_THRESHOLD
    if status == "blocked_score" and threshold_met:
        parser.error(f"BLOCKED_SCORE is invalid because score {total} meets threshold {RUN_SCORE_THRESHOLD}")
    if status == "proposed" and not threshold_met:
        parser.error(f"PROPOSED is invalid because score {total} is below threshold {RUN_SCORE_THRESHOLD}")
    if status == "approved_to_run":
        if not threshold_met:
            parser.error(f"APPROVED_TO_RUN requires score >= {RUN_SCORE_THRESHOLD}; found {total}")
        if not args.human_approved:
            parser.error("APPROVED_TO_RUN requires an explicit --human-approved flag")
    if status == "approved_for_shadow" and not args.human_approved:
        parser.error("APPROVED_FOR_SHADOW requires a separate explicit --human-approved flag")
    if status == "running":
        if previous_status != "approved_to_run":
            parser.error("RUNNING requires the immediately preceding event to be APPROVED_TO_RUN")
        if not bool(previous.get("human_approved")):
            parser.error("preceding APPROVED_TO_RUN event does not contain human_approved=true")
        previous_total = previous.get("score_total")
        if previous_total != total:
            parser.error(
                "queue score changed after approval; append a new proposal and approval before RUNNING"
            )
        if previous.get("score_components") != components:
            parser.error(
                "queue score components changed after approval; append a new proposal and approval before RUNNING"
            )

    prior_run_approval = next(
        (
            event
            for event in reversed(history)
            if normalize_status(str(event.get("status", ""))) == "approved_to_run"
            and bool(event.get("human_approved"))
        ),
        None,
    )
    if prior_run_approval is not None and status != "invalid":
        if prior_run_approval.get("score_total") != total or prior_run_approval.get("score_components") != components:
            parser.error("queue score changed after run approval; use a new experiment_id")
    if status == "approved_for_shadow" and not threshold_met:
        parser.error(f"APPROVED_FOR_SHADOW requires score >= {RUN_SCORE_THRESHOLD}; found {total}")

    prior_human_approval = prior_run_approval is not None
    approval_recorded = bool(args.human_approved) or prior_human_approval
    approval_in_effect = approval_recorded and status in {"approved_to_run", "running"}

    event: dict[str, Any] = {
        "schema_version": 1,
        "event_id": str(uuid.uuid4()),
        "sequence": len(history) + 1,
        "experiment_id": experiment_id,
        "status": status,
        "previous_status": previous_status,
        "previous_event_id": previous.get("event_id") if previous else None,
        "occurred_at": utc_now(),
        "actor": actor,
        "score_components": components,
        "score_total": total,
        "score_threshold": RUN_SCORE_THRESHOLD,
        "score_threshold_met": threshold_met,
        "human_approved": bool(args.human_approved),
        "human_run_approval_recorded": approval_recorded,
        "run_approval_in_effect": approval_in_effect,
        "automatic_execution_allowed": approval_in_effect,
        "execution_authorized": status in {"approved_to_run", "running"} and approval_in_effect,
        "production_approved": False,
        "merge_approved": False,
        "buy_approved": False,
        "production_change_allowed": False,
        "merge_allowed": False,
        "buy_logic_change_allowed": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "artifacts": artifacts,
        "notes": args.notes.strip(),
        "queue_file": str(queue_path),
    }

    if args.dry_run:
        print(json.dumps({"dry_run": True, "registry": str(registry_path), "event": event}, ensure_ascii=False, indent=2))
        return 0

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    with registry_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"appended": True, "registry": str(registry_path), "event": event}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.stderr.close()
        raise SystemExit(1)
