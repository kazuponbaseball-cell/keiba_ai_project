from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable


def verify_read_only_records(records: Iterable[dict[str, Any]]) -> None:
    for record in records:
        if record.get("formal_buy") is not False:
            raise ValueError("formal_buy output violation")
        if record.get("send_order") is not False:
            raise ValueError("send_order output violation")
        if record.get("paper_stake_yen") != 0:
            raise ValueError("paper stake output violation")


def build_payload(
    records: list[dict[str, Any]], *, generated_at: str | None = None
) -> dict[str, Any]:
    verify_read_only_records(records)
    generated = generated_at or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    status_counts = Counter(str(record.get("status", "UNKNOWN")) for record in records)
    reason_counts = Counter(str(record.get("reason", "UNKNOWN")) for record in records)
    return {
        "schema_version": 1,
        "view_type": "research_shadow_read_only",
        "generated_at": generated,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "records": records,
    }


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return escape(str(value))


def render_html(payload: dict[str, Any]) -> str:
    records = list(payload.get("records", []))
    verify_read_only_records(records)
    rows = []
    for record in records:
        status = str(record.get("status", "UNKNOWN"))
        status_class = "ready" if status == "PAPER_READY" else "closed"
        rows.append(
            "<tr>"
            f"<td>{_cell(record.get('race_id'))}</td>"
            f"<td>{_cell(record.get('candidate_pair_key'))}</td>"
            f"<td><span class=\"status {status_class}\">{_cell(status)}</span></td>"
            f"<td>{_cell(record.get('reason'))}</td>"
            f"<td>{_cell(record.get('robust_expected_return'))}</td>"
            f"<td>{_cell(record.get('quote_source_event_time'))}</td>"
            f"<td>{_cell(record.get('quote_received_at'))}</td>"
            "</tr>"
        )
    table_rows = "".join(rows) or (
        '<tr><td colspan="7" class="empty">No research shadow records</td></tr>'
    )
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Race-day Research Shadow</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #f4f6f8; color: #18212f; }}
    header {{ padding: 18px 20px; background: #fff; border-bottom: 1px solid #d9dee6; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; letter-spacing: 0; }}
    .guard {{ color: #8a2d19; font-weight: 700; }}
    main {{ padding: 16px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; font-size: 14px; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e1e5eb; text-align: left; white-space: nowrap; }}
    th {{ background: #eef1f5; }}
    .status {{ font-weight: 700; }}
    .ready {{ color: #17643a; }}
    .closed {{ color: #8a2d19; }}
    .empty {{ text-align: center; color: #667085; }}
  </style>
</head>
<body>
  <header>
    <h1>Race-day Research Shadow</h1>
    <div class="guard">閲覧専用 / 発注無効 / stake 0</div>
    <div>Generated: {_cell(payload.get('generated_at'))} / Records: {_cell(payload.get('record_count'))}</div>
  </header>
  <main>
    <table>
      <thead><tr><th>Race</th><th>Pair</th><th>Status</th><th>Reason</th><th>Robust ER</th><th>Source time</th><th>Received</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </main>
</body>
</html>
"""


def write_artifacts(
    records: list[dict[str, Any]],
    *,
    json_path: Path,
    html_path: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    payload = build_payload(records, generated_at=generated_at)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(render_html(payload), encoding="utf-8")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    verify_read_only_records(records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a separate read-only research shadow sidecar."
    )
    parser.add_argument("--ledger-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = write_artifacts(
        read_jsonl(args.ledger_jsonl),
        json_path=args.output_json,
        html_path=args.output_html,
    )
    print(
        json.dumps(
            {
                "record_count": payload["record_count"],
                "output_json": str(args.output_json),
                "output_html": str(args.output_html),
                "formal_buy": False,
                "send_order": False,
                "stake": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
