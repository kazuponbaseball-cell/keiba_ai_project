from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import re
import secrets
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import threading
from urllib.parse import parse_qs, urlparse


class AuthHandler(SimpleHTTPRequestHandler):
    username = ""
    password = ""
    no_auth = False
    quiet = False
    realm = "Keiba AI Dashboard"
    project_root = Path(".")
    mirror_root: Path | None = None
    refresh_lock = threading.Lock()

    def log_message(self, format: str, *args) -> None:
        if self.quiet:
            return
        super().log_message(format, *args)

    def send_response(self, code: int, message: str | None = None) -> None:
        super().send_response(code, message)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def _authorized(self) -> bool:
        if self.no_auth:
            return True
        header = self.headers.get("Authorization", "")
        prefix = "Basic "
        if not header.startswith(prefix):
            return False
        try:
            raw = base64.b64decode(header[len(prefix) :], validate=True).decode("utf-8")
        except Exception:
            return False
        if ":" not in raw:
            return False
        user, pwd = raw.split(":", 1)
        return secrets.compare_digest(user, self.username) and secrets.compare_digest(pwd, self.password)

    def _require_auth(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", f'Basic realm="{self.realm}"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("Authentication required.\n".encode("utf-8"))

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _dashboard_date(self) -> str:
        inputs_path = self.project_root / "outputs" / "runtime" / "current_dashboard_inputs.json"
        if inputs_path.exists():
            try:
                payload = json.loads(inputs_path.read_text(encoding="utf-8"))
                date_value = str(payload.get("date", ""))
                if re.fullmatch(r"\d{8}", date_value):
                    return date_value
            except Exception:
                pass
        return datetime.now().strftime("%Y%m%d")

    def _sync_dashboard_mirror(self) -> dict:
        if self.mirror_root is None:
            return {"enabled": False}
        source_root = self.project_root
        target_root = self.mirror_root
        files = [
            Path("outputs/ui/live_odds_dashboard.html"),
            Path("outputs/ui/live_odds_dashboard.summary.json"),
            Path("outputs/runtime/current_dashboard_inputs.json"),
            Path("outputs/analysis/win5_runtime/win5_plan.json"),
        ]
        copied: list[str] = []
        missing: list[str] = []
        for rel_path in files:
            source = source_root / rel_path
            target = target_root / rel_path
            if not source.exists():
                missing.append(str(rel_path))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(rel_path))
        return {
            "enabled": True,
            "source_root": str(source_root),
            "target_root": str(target_root),
            "copied": copied,
            "missing": missing,
        }

    def _refresh_dashboard(
        self,
        mode: str = "quick",
        date_key: str | None = None,
        race_ids: list[str] | None = None,
    ) -> dict:
        script = self.project_root / "scripts" / "run_current_strongest_line_update.ps1"
        if not script.exists():
            raise FileNotFoundError(f"Refresh script was not found: {script}")
        if mode not in {"quick", "full"}:
            raise ValueError(f"Unsupported refresh mode: {mode}")
        target_date = date_key if date_key and re.fullmatch(r"\d{8}", date_key) else self._dashboard_date()

        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Date",
            target_date,
        ]
        # Dashboard refresh is for live odds/body weights/going. Settled-result
        # fetching is handled by the end-of-day review flow, and it makes race-day
        # manual refreshes noticeably slower.
        cmd.append("-SkipResultFetch")
        if race_ids:
            cmd.append("-RaceKeys")
            cmd.append(",".join(race_ids))
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )

        summary_path = self.project_root / "outputs" / "ui" / "live_odds_dashboard.summary.json"
        summary: dict = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}

        payload = {
            "ok": result.returncode == 0,
            "mode": mode,
            "date": target_date,
            "race_ids": race_ids or [],
            "returncode": result.returncode,
            "summary": summary,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        if result.returncode != 0:
            raise RuntimeError(json.dumps(payload, ensure_ascii=False))
        payload["mirror"] = self._sync_dashboard_mirror()
        return payload

    def do_GET(self) -> None:
        if not self._authorized():
            self._require_auth()
            return
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") == "/kazu":
            suffix = f"?{parsed.query}" if parsed.query else ""
            self.path = f"/outputs/ui/live_odds_dashboard.html{suffix}"
        super().do_GET()

    def do_POST(self) -> None:
        if not self._authorized():
            self._require_auth()
            return
        parsed = urlparse(self.path)
        path = parsed.path
        if path != "/api/refresh":
            self._send_json(404, {"ok": False, "error": "Unknown endpoint."})
            return
        query = parse_qs(parsed.query)
        mode = query.get("mode", ["quick"])[0].lower()
        date_key = query.get("date", [""])[0]
        race_ids = []
        for value in query.get("race_id", []) + query.get("race_ids", []):
            race_ids.extend(part for part in re.split(r"[,\s]+", value) if part)
        race_ids = list(dict.fromkeys(race_ids))
        if date_key and not re.fullmatch(r"\d{8}", date_key):
            self._send_json(400, {"ok": False, "error": "date must be YYYYMMDD."})
            return
        if any(not re.fullmatch(r"\d{16}", race_id) for race_id in race_ids):
            self._send_json(400, {"ok": False, "error": "race_id must be a 16-digit JRA race ID."})
            return
        if mode not in {"quick", "full"}:
            self._send_json(400, {"ok": False, "error": "mode must be quick or full."})
            return
        if not self.refresh_lock.acquire(blocking=False):
            self._send_json(409, {"ok": False, "error": "Refresh is already running."})
            return
        try:
            payload = self._refresh_dashboard(mode=mode, date_key=date_key or None, race_ids=race_ids)
            self._send_json(200, payload)
        except subprocess.TimeoutExpired as exc:
            self._send_json(
                504,
                {
                    "ok": False,
                    "error": "Refresh timed out.",
                    "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                    "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                },
            )
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})
        finally:
            self.refresh_lock.release()

    def do_HEAD(self) -> None:
        if not self._authorized():
            self._require_auth()
            return
        super().do_HEAD()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Keiba dashboard with HTTP Basic authentication.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--username", default="keiba")
    parser.add_argument("--password", default="")
    parser.add_argument("--no-auth", action="store_true", help="Disable Basic auth. Use only on loopback/local trusted networks.")
    parser.add_argument("--quiet", action="store_true", help="Suppress request logging for detached background use.")
    parser.add_argument("--directory", default=".")
    parser.add_argument(
        "--project-root",
        default="",
        help="Project root used for refresh scripts. Defaults to --directory. Use this when serving a lightweight mirror.",
    )
    args = parser.parse_args()
    if not args.no_auth and not args.password:
        parser.error("--password is required unless --no-auth is set.")

    AuthHandler.username = args.username
    AuthHandler.password = args.password
    AuthHandler.no_auth = args.no_auth
    AuthHandler.quiet = args.quiet
    directory = Path(args.directory).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else directory
    AuthHandler.project_root = project_root
    AuthHandler.mirror_root = directory if directory != project_root else None
    handler = lambda *h_args, **h_kwargs: AuthHandler(*h_args, directory=str(directory), **h_kwargs)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    if not args.quiet:
        print(f"Serving {directory} on http://{args.host}:{args.port}/")
        if directory != project_root:
            print(f"Refresh project root: {project_root}")
        if args.no_auth:
            print("Basic auth: disabled")
        else:
            print(f"Basic auth username: {args.username}")
    server.serve_forever()


if __name__ == "__main__":
    main()
