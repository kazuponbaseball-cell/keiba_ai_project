from __future__ import annotations

import argparse
import json
import os
import textwrap
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LINE_PUSH_ENDPOINT = "https://api.line.me/v2/bot/message/push"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, low_memory=False)


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_rows(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame.empty:
        return []
    rows: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        venue = text(row.get("venue"))
        effective_date = text(row.get("effective_date"))
        if not venue or not effective_date:
            continue
        rows.append(
            {
                "effective_date": effective_date,
                "venue": venue,
                "weather": text(row.get("weather")),
                "turf_going": text(row.get("turf_going")),
                "dirt_going": text(row.get("dirt_going")),
                "timing": text(row.get("timing")),
                "fetched_at": text(row.get("fetched_at")),
            }
        )
    rows.sort(key=lambda x: (x["effective_date"], x["venue"]))
    return rows


def row_key(row: dict[str, str]) -> str:
    return f"{row.get('effective_date','')}|{row.get('venue','')}"


def comparable(row: dict[str, str]) -> dict[str, str]:
    return {
        "weather": row.get("weather", ""),
        "turf_going": row.get("turf_going", ""),
        "dirt_going": row.get("dirt_going", ""),
    }


def detect_changes(previous: list[dict[str, str]], current: list[dict[str, str]]) -> list[dict[str, str]]:
    prev_by_key = {row_key(row): row for row in previous}
    changes: list[dict[str, str]] = []
    for now in current:
        before = prev_by_key.get(row_key(now))
        if not before:
            continue
        for field, label in [
            ("weather", "天候"),
            ("turf_going", "芝"),
            ("dirt_going", "ダート"),
        ]:
            old = text(before.get(field))
            new = text(now.get(field))
            if old and new and old != new:
                changes.append(
                    {
                        "effective_date": now["effective_date"],
                        "venue": now["venue"],
                        "field": field,
                        "label": label,
                        "old": old,
                        "new": new,
                    }
                )
    return changes


def resolve_dashboard_url(explicit: str) -> str:
    if text(explicit):
        return text(explicit)
    info_path = project_path("outputs/runtime/public_dashboard_tunnel.json")
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8-sig"))
            public_url = text(info.get("public_url"))
            if public_url:
                return f"{public_url.rstrip('/')}/outputs/ui/live_odds_dashboard.html"
            dashboard_url = text(info.get("dashboard_url"))
            if dashboard_url:
                return dashboard_url
        except Exception:
            pass
    return ""


def build_message(changes: list[dict[str, str]], dashboard_url: str) -> str:
    if not changes:
        return ""
    date_text = changes[0]["effective_date"]
    if len(date_text) == 8:
        date_text = f"{date_text[:4]}/{date_text[4:6]}/{date_text[6:]}"
    lines = [
        f"Keiba AI 馬場変化検知 {date_text}",
        "馬場/天候が更新されました。買い目と見送り判定は最新馬場で再計算済みです。",
        "",
    ]
    grouped: dict[str, list[dict[str, str]]] = {}
    for change in changes:
        grouped.setdefault(change["venue"], []).append(change)
    for venue, venue_changes in grouped.items():
        parts = [f"{c['label']} {c['old']}→{c['new']}" for c in venue_changes]
        lines.append(f"- {venue}: " + " / ".join(parts))
    if dashboard_url:
        lines.extend(["", f"画面: {dashboard_url}"])
    return "\n".join(lines).strip()


def send_line_push(message: str, token: str, to: str) -> dict[str, Any]:
    payload = {"to": to, "messages": [{"type": "text", "text": message}]}
    request = urllib.request.Request(
        LINE_PUSH_ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {"ok": True, "status": response.status, "body": body}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "body": body}
    except urllib.error.URLError as exc:
        return {"ok": False, "status": None, "body": str(exc.reason)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect JRA track condition changes and optionally notify LINE.")
    parser.add_argument("--track-csv", default="data/processed/live_track_conditions/current_track_conditions.csv")
    parser.add_argument("--state-json", default="data/processed/notifications/track_condition_change_state.json")
    parser.add_argument("--output-json", default="outputs/analysis/live_track_conditions/track_condition_change_summary.json")
    parser.add_argument("--message-text", default="outputs/notifications/track_condition_change_latest.txt")
    parser.add_argument("--dashboard-url", default="")
    parser.add_argument("--notify-initial", action="store_true")
    parser.add_argument("--no-state-write", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--send-if-configured", action="store_true")
    parser.add_argument("--line-token-env", default="LINE_CHANNEL_ACCESS_TOKEN")
    parser.add_argument("--line-to-env", default="LINE_USER_ID")
    args = parser.parse_args()

    track_path = project_path(args.track_csv)
    state_path = project_path(args.state_json)
    output_path = project_path(args.output_json)
    message_path = project_path(args.message_text)

    current = normalize_rows(read_csv_safe(track_path))
    state = read_state(state_path)
    previous = state.get("rows") if isinstance(state.get("rows"), list) else []
    first_observation = not previous
    changes = current if first_observation and args.notify_initial else detect_changes(previous, current)
    dashboard_url = resolve_dashboard_url(args.dashboard_url)
    message = build_message(changes, dashboard_url)

    if not args.no_state_write:
        state.update(
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "track_csv": str(track_path),
                "rows": current,
                "last_change_at": datetime.now().isoformat(timespec="seconds")
                if changes
                else state.get("last_change_at", ""),
                "last_changes": changes if changes else state.get("last_changes", []),
            }
        )
        write_state(state_path, state)

    message_path.parent.mkdir(parents=True, exist_ok=True)
    message_path.write_text((message or "馬場変化はありません。") + "\n", encoding="utf-8")

    result: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "track_csv": str(track_path),
        "state_json": str(state_path),
        "first_observation": first_observation,
        "state_written": not args.no_state_write,
        "changed": bool(changes),
        "changes": changes,
        "message_text": str(message_path),
        "sent": False,
        "send_reason": "not_requested",
    }

    should_send = bool(changes) and (args.send or args.send_if_configured)
    if should_send:
        token = os.environ.get(args.line_token_env, "")
        to = os.environ.get(args.line_to_env, "")
        if token and to and message:
            line_result = send_line_push(message, token, to)
            result["line"] = line_result
            result["sent"] = bool(line_result.get("ok"))
            result["send_reason"] = "sent" if result["sent"] else "send_failed"
        elif args.send:
            raise SystemExit(
                textwrap.dedent(
                    f"""\
                    LINE credentials are missing.
                    Set environment variables:
                      {args.line_token_env}=<Messaging API channel access token>
                      {args.line_to_env}=<LINE user ID or group ID>
                    """
                )
            )
        else:
            result["send_reason"] = "credentials_missing"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if message:
        print("\n--- message preview ---")
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
