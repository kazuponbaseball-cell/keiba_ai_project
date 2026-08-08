from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_strict_t3_capture_packet_v1 import (  # noqa: E402
    canonical_digest,
    canonical_json,
    load_config,
    parse_time,
)
from run_strict_t3_shadow_decision_v1 import (  # noqa: E402
    load_candidate_population,
    read_decision_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]


def _normalized_path(path: Path) -> str:
    return path.resolve().as_posix().lower()


def assert_report_output_path_allowed(path: Path, config: dict[str, Any]) -> None:
    normalized = _normalized_path(path)
    for suffix in config["report"]["forbidden_output_suffixes"]:
        if normalized.endswith(str(suffix).replace("\\", "/").lower()):
            raise ValueError("production dashboard output path is forbidden")


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _display(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_report_payload(
    *,
    source_manifest_path: Path,
    decision_ledger_path: Path,
    config: dict[str, Any],
    generated_at: Any,
    enforce_expected_counts: bool = True,
) -> dict[str, Any]:
    _manifest, population = load_candidate_population(
        source_manifest_path,
        config,
        enforce_expected_counts=enforce_expected_counts,
    )
    decisions = [
        row
        for row in read_decision_jsonl(decision_ledger_path)
        if row.get("experiment_id") == config["experiment_id"]
        and row.get("cohort_id") == config["cohort_id"]
    ]
    decision_by_race: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        race_id = str(decision.get("race_id", ""))
        if not race_id or race_id in decision_by_race:
            raise ValueError("decision ledger has missing or duplicate race_id")
        decision_by_race[race_id] = decision

    target_ids = {str(row["candidate"]["race_id"]) for row in population}
    if set(decision_by_race) != target_ids:
        raise ValueError("read-only report requires one decision for every target")
    rows: list[dict[str, Any]] = []
    for source in population:
        candidate = source["candidate"]
        race_id = str(candidate["race_id"])
        decision = decision_by_race[race_id]
        if decision.get("candidate_freeze_record_hash") != candidate.get(
            "candidate_freeze_record_hash"
        ):
            raise ValueError("report candidate hash reference mismatch")
        if str(decision.get("candidate_pair_key", "")) != str(
            candidate.get("candidate_pair_key", "")
        ):
            raise ValueError("report candidate pair reference mismatch")
        rows.append(
            {
                "card_id": source["card_id"],
                "race_id": race_id,
                "race_no": int(candidate.get("race_no", 0)),
                "source_candidate_status": candidate.get("record_status"),
                "candidate_pair_key": decision.get("candidate_pair_key"),
                "confidence_gate_pass": decision.get("confidence_gate_pass"),
                "p_action_calibrated": decision.get("p_action_calibrated"),
                "quote_evaluation_entered": decision.get(
                    "quote_evaluation_entered"
                ),
                "candidate_pair_t3_quote_valid": decision.get(
                    "candidate_pair_t3_quote_valid"
                ),
                "t3_wide_odds_low": decision.get("t3_wide_odds_low"),
                "t3_wide_odds_high": decision.get("t3_wide_odds_high"),
                "research_expected_return_low": decision.get(
                    "research_expected_return_low"
                ),
                "shadow_action": decision.get("shadow_action"),
                "decision_reason": decision.get("decision_reason"),
                "measurement_only": decision.get("measurement_only"),
                "t3_quote_selected_asof_time": decision.get(
                    "t3_quote_selected_asof_time"
                ),
                "t3_decision_committed_at": decision.get(
                    "t3_decision_committed_at"
                ),
                "formal_buy": False,
                "send_order": False,
                "stake": 0,
            }
        )
    rows.sort(key=lambda row: (str(row["card_id"]), int(row["race_no"])))
    action_counts = Counter(str(row["shadow_action"]) for row in rows)
    reason_counts = Counter(str(row["decision_reason"]) for row in rows)
    generated = parse_time(generated_at, config["timezone"])
    payload = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "cohort_id": config["cohort_id"],
        "report_type": "READ_ONLY_STRICT_T3_RESEARCH_SHADOW",
        "generated_at": generated.isoformat(timespec="milliseconds"),
        "record_count": len(rows),
        "candidate_ready_rows": sum(
            row["source_candidate_status"] == "CANDIDATE_READY" for row in rows
        ),
        "source_failure_rows": sum(
            row["source_candidate_status"] == "FAILED" for row in rows
        ),
        "quote_evaluation_rows": sum(
            row["quote_evaluation_entered"] is True for row in rows
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rows": rows,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "roi_calculated": False,
        "production_dashboard_mutated": False,
    }
    payload["report_payload_hash"] = canonical_digest(payload)
    return payload


def render_report_html(payload: dict[str, Any]) -> str:
    table_rows = []
    for row in payload["rows"]:
        cells = (
            row["card_id"],
            row["race_no"],
            row["race_id"],
            row["candidate_pair_key"],
            row["confidence_gate_pass"],
            row["p_action_calibrated"],
            row["t3_wide_odds_low"],
            row["t3_wide_odds_high"],
            row["research_expected_return_low"],
            row["shadow_action"],
            row["decision_reason"],
        )
        table_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(_display(cell))}</td>" for cell in cells)
            + "</tr>"
        )
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Strict T-3 Research Shadow</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; color: #172033; }
    h1 { font-size: 24px; margin: 0 0 8px; }
    .notice { padding: 10px 12px; border: 1px solid #bcc7d6; background: #f5f7fa; }
    table { width: 100%; border-collapse: collapse; margin-top: 18px; font-size: 13px; }
    th, td { border-bottom: 1px solid #dce2ea; padding: 7px 6px; text-align: left; }
    th { position: sticky; top: 0; background: #eef2f7; }
    .meta { color: #58657a; font-size: 13px; }
  </style>
</head>
<body>
  <h1>Strict T-3 Research Shadow</h1>
  <p class="notice">Read-only research evidence. No order path, no stake, no formal BUY.</p>
  <p class="meta">Generated: """ + html.escape(str(payload["generated_at"])) + """ | Rows: """ + str(payload["record_count"]) + """</p>
  <table>
    <thead><tr><th>Card</th><th>Race</th><th>Race ID</th><th>Frozen pair</th><th>Confidence</th><th>p(action)</th><th>Odds low</th><th>Odds high</th><th>Research ER low</th><th>Shadow action</th><th>Reason</th></tr></thead>
    <tbody>""" + "".join(table_rows) + """</tbody>
  </table>
</body>
</html>
"""


def write_report(
    *,
    source_manifest_path: Path,
    decision_ledger_path: Path,
    output_json_path: Path,
    output_html_path: Path,
    config: dict[str, Any],
    generated_at: Any,
    enforce_expected_counts: bool = True,
) -> dict[str, Any]:
    assert_report_output_path_allowed(output_json_path, config)
    assert_report_output_path_allowed(output_html_path, config)
    payload = build_report_payload(
        source_manifest_path=source_manifest_path,
        decision_ledger_path=decision_ledger_path,
        config=config,
        generated_at=generated_at,
        enforce_expected_counts=enforce_expected_counts,
    )
    _write_text_atomic(output_json_path, canonical_json(payload) + "\n")
    _write_text_atomic(output_html_path, render_report_html(payload))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only static Strict-T3 research report."
    )
    parser.add_argument("--source-manifest-json", type=Path, required=True)
    parser.add_argument("--decision-ledger-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--generated-at", default="")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "strict_t3_shadow_decision_exp011.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    generated_at = args.generated_at or datetime.now(
        ZoneInfo(config["timezone"])
    ).isoformat()
    payload = write_report(
        source_manifest_path=args.source_manifest_json,
        decision_ledger_path=args.decision_ledger_jsonl,
        output_json_path=args.output_json,
        output_html_path=args.output_html,
        config=config,
        generated_at=generated_at,
    )
    print(canonical_json({"record_count": payload["record_count"], "report": str(args.output_html)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
