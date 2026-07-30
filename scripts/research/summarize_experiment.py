from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_from_root(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


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
            event = dict(event)
            event["_registry_line"] = line_number
            events.append(event)
    return events


def md(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def artifact_text(event: dict[str, Any]) -> str:
    artifacts = event.get("artifacts", [])
    if not isinstance(artifacts, list):
        return md(artifacts)
    return "<br>".join(md(item) for item in artifacts)


def render_summary(events: list[dict[str, Any]], registry_path: Path, experiment_id: str | None) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if experiment_id is None or event["experiment_id"] == experiment_id:
            grouped[event["experiment_id"]].append(event)

    lines = [
        "# Research Experiment Registry Summary",
        "",
        f"- Generated at (UTC): `{utc_now()}`",
        f"- Registry: `{registry_path}`",
        f"- Filter: `{experiment_id}`" if experiment_id else "- Filter: all experiments",
        "- Autonomous execution: prohibited before approval; allowed only in `approved_to_run` / `running` within approved scope",
        "- Production / merge / BUY approval: always false and outside this registry's authority",
        "",
    ]
    if not grouped:
        lines.extend(["No matching registry events.", ""])
        return "\n".join(lines)

    lines.extend(
        [
            "## Latest state",
            "",
            "| Experiment | Status | Score | Human run approval | Execution authorized | Updated (UTC) | Events |",
            "|---|---|---:|---|---|---|---:|",
        ]
    )
    for key in sorted(grouped):
        history = grouped[key]
        latest = history[-1]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{md(key)}`",
                    f"`{md(latest.get('status'))}`",
                    md(latest.get("score_total")),
                    md(
                        latest.get(
                            "human_run_approval_recorded",
                            latest.get("run_approval_in_effect", latest.get("human_approved", False)),
                        )
                    ),
                    md(latest.get("execution_authorized", False)),
                    md(latest.get("occurred_at")),
                    str(len(history)),
                ]
            )
            + " |"
        )

    for key in sorted(grouped):
        history = grouped[key]
        latest = history[-1]
        lines.extend(
            [
                "",
                f"## {md(key)}",
                "",
                f"- Latest status: `{md(latest.get('status'))}`",
                f"- Score: **{md(latest.get('score_total'))}/{md(latest.get('score_threshold', 75))}**",
                f"- Score threshold met: `{md(latest.get('score_threshold_met', False))}`",
                f"- Human run approval recorded: `{md(latest.get('human_run_approval_recorded', False))}`",
                f"- Human run approval in effect: `{md(latest.get('run_approval_in_effect', False))}`",
                f"- Autonomous execution allowed: `{md(latest.get('automatic_execution_allowed', False))}`",
                f"- Execution authorized: `{md(latest.get('execution_authorized', False))}`",
                f"- Production approved: `{md(latest.get('production_approved', False))}`",
                f"- Merge approved: `{md(latest.get('merge_approved', False))}`",
                f"- BUY approved: `{md(latest.get('buy_approved', False))}`",
                f"- Formal BUY: `{md(latest.get('formal_buy', False))}`",
                f"- Send order: `{md(latest.get('send_order', False))}`",
                f"- Stake: `{md(latest.get('stake', 0))}`",
                "",
                "### Event history",
                "",
                "| Seq | UTC | Status | Previous | Actor | Score | Human approved event | Artifacts | Notes |",
                "|---:|---|---|---|---|---:|---|---|---|",
            ]
        )
        for event in history:
            lines.append(
                "| "
                + " | ".join(
                    [
                        md(event.get("sequence", event.get("_registry_line"))),
                        md(event.get("occurred_at")),
                        f"`{md(event.get('status'))}`",
                        f"`{md(event.get('previous_status'))}`" if event.get("previous_status") else "",
                        md(event.get("actor")),
                        md(event.get("score_total")),
                        md(event.get("human_approved", False)),
                        artifact_text(event),
                        md(event.get("notes")),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render append-only Level-3 registry history and latest state as Markdown."
    )
    parser.add_argument("--experiment-id", default=None, help="Optional exact experiment identifier filter.")
    parser.add_argument("--root", type=Path, default=default_root(), help="Repository root.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("research/REGISTRY.jsonl"),
        help="JSONL registry path, relative to --root unless absolute.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write Markdown here instead of stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    registry_path = resolve_from_root(root, args.registry)
    try:
        events = load_events(registry_path)
    except ValueError as exc:
        parser.error(str(exc))
    rendered = render_summary(events, registry_path, args.experiment_id)

    if args.output is None:
        sys.stdout.write(rendered)
        return 0
    output_path = resolve_from_root(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.stderr.close()
        raise SystemExit(1)
