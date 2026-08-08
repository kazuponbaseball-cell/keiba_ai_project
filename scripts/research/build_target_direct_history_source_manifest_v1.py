from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.target_direct_history_v1 import (  # noqa: E402
    canonical_horse_id,
    canonical_json,
    parse_ra_record,
    parse_se_record,
)


EXPERIMENT_ID = "EXP-20260808-027"
SOURCE_TIME_CONTRACT = "STRICTLY_BEFORE_TARGET_DATE"
COMPLETE_COVERAGE = "rcov-all-past-current-runners"
PARTIAL_COVERAGE = "partial-history"
JRA_VENUE_CODES = {f"{value:02d}" for value in range(1, 11)}
LOCAL_VENUE_CODES = {f"{value:02d}" for value in range(30, 100)}
FULL_SHA256 = re.compile(r"[0-9a-f]{64}")
SOURCE_NAME = re.compile(r"(?:SU|SR).+\.DAT", re.IGNORECASE)
RACE_KEY = re.compile(r"[0-9]{8}[0-9A-Z]{8}")


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    path_text: str
    sha256: str
    payload: bytes


@dataclass(frozen=True)
class CurrentRunner:
    target_race_key: str
    horse_no: int
    horse_id: str


@dataclass(frozen=True)
class RaceIdentity:
    target_race_key: str
    race_date: str
    venue_code: str
    source_path: str
    source_sha256: str
    record_sha256: str
    raw: bytes


@dataclass(frozen=True)
class RunnerIdentity:
    target_race_key: str
    race_date: str
    venue_code: str
    horse_id: str
    source_path: str
    source_sha256: str
    record_sha256: str
    raw: bytes


def _canonical_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def snapshot_file(path: Path, *, expected_sha256: str | None = None) -> FileSnapshot:
    resolved = path.resolve()
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != after.st_size
    ):
        raise ValueError(f"source changed while being read: {resolved}")
    observed = _sha256(payload)
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        if not FULL_SHA256.fullmatch(expected):
            raise ValueError("expected SHA-256 must contain 64 lowercase hexadecimal characters")
        if observed != expected:
            raise ValueError(f"source hash mismatch: {observed} != {expected}")
    return FileSnapshot(
        path=resolved,
        path_text=_canonical_path(resolved),
        sha256=observed,
        payload=payload,
    )


def _ascii(raw: bytes, start: int, length: int) -> str:
    return raw[start : start + length].decode("ascii", errors="ignore").strip().upper()


def _parse_date(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[0-9]{8}", value):
        raise ValueError(f"{label} must be YYYYMMDD: {value!r}")
    datetime.strptime(value, "%Y%m%d")
    return value


def _parse_race_key(raw: bytes, *, label: str) -> str:
    key = _ascii(raw, 11, 16)
    if not RACE_KEY.fullmatch(key):
        raise ValueError(f"{label} record has invalid race key: {key!r}")
    _parse_date(key[:8], label=f"{label} race date")
    return key


def canonical_authority_race_id(target_race_key: str) -> str:
    key = target_race_key.strip().upper()
    if not RACE_KEY.fullmatch(key):
        raise ValueError(f"invalid authority race key: {target_race_key!r}")
    return key[:4] + key[8:]


def _decode_du(raw: bytes, start: int, end: int) -> str:
    return raw[start:end].decode("cp932", errors="replace").replace("\u3000", "").strip()


def parse_current_du(payload: bytes, *, target_date: str) -> list[CurrentRunner]:
    target_date = _parse_date(target_date, label="target date")
    runners: list[CurrentRunner] = []
    identities: set[tuple[str, int]] = set()
    horse_ids: set[str] = set()
    for raw in payload.splitlines():
        if len(raw) < 40:
            continue
        record_type = _decode_du(raw, 0, 3)
        if record_type not in {"SE1", "SE2"}:
            continue
        race_date = _decode_du(raw, 11, 19)
        race_part = _decode_du(raw, 19, 27)
        horse_no_text = _decode_du(raw, 28, 30)
        if len(race_date) != 8 or len(race_part) != 8 or not horse_no_text.isdigit():
            continue
        if race_date != target_date:
            raise ValueError(f"DU target date drift: {race_date} != {target_date}")
        target_race_key = race_date + race_part
        if not target_race_key.isdigit():
            raise ValueError(f"DU has invalid target race key: {target_race_key!r}")
        horse_no = int(horse_no_text)
        if horse_no <= 0:
            continue
        horse_id = canonical_horse_id(_decode_du(raw, 32, 40))
        if len(horse_id) != 10:
            raise ValueError(f"DU has invalid horse id: {horse_id!r}")
        identity = (target_race_key, horse_no)
        if identity in identities:
            raise ValueError(f"DU has duplicate current runner identity: {identity}")
        if horse_id in horse_ids:
            raise ValueError(f"DU has duplicate current horse id: {horse_id}")
        identities.add(identity)
        horse_ids.add(horse_id)
        runners.append(CurrentRunner(target_race_key, horse_no, horse_id))
    if not runners:
        raise ValueError("DU source contains no active runner records")
    return sorted(runners, key=lambda row: (row.target_race_key, row.horse_no))


def expand_history_sources(
    patterns: Sequence[str],
    *,
    allowed_roots: Sequence[Path],
    allowed_years: set[str],
) -> list[Path]:
    if not patterns:
        raise ValueError("at least one history source pattern is required")
    roots = [root.resolve() for root in allowed_roots]
    if not roots:
        raise ValueError("at least one allowed source root is required")
    matched: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        values = sorted(glob.glob(pattern, recursive=False))
        if not values:
            raise ValueError(f"history source pattern matched no files: {pattern}")
        for value in values:
            path = Path(value).resolve()
            if not path.is_file():
                raise ValueError(f"history source is not a file: {path}")
            canonical = _canonical_path(path)
            if canonical in seen:
                raise ValueError(f"history source was declared more than once: {canonical}")
            matching_roots: list[tuple[Path, Path]] = []
            for root in roots:
                try:
                    matching_roots.append((root, path.relative_to(root)))
                except ValueError:
                    continue
            if len(matching_roots) != 1:
                raise ValueError(f"history source is outside one unique allowed root: {path}")
            _, relative = matching_roots[0]
            if not relative.parts or relative.parts[0] not in allowed_years:
                raise ValueError(f"history source is outside allowed years: {path}")
            if not SOURCE_NAME.fullmatch(path.name):
                raise ValueError(f"history source is not a declared SU/SR DAT file: {path}")
            seen.add(canonical)
            matched.append(path)
    return sorted(matched, key=_canonical_path)


def parse_history_snapshots(
    snapshots: Sequence[FileSnapshot],
) -> tuple[dict[str, RaceIdentity], dict[tuple[str, str], RunnerIdentity], set[str], set[str]]:
    races: dict[str, RaceIdentity] = {}
    runners: dict[tuple[str, str], RunnerIdentity] = {}
    ra_source_paths: set[str] = set()
    se_source_paths: set[str] = set()
    for snapshot in snapshots:
        relevant_records = 0
        for raw in snapshot.payload.splitlines():
            if raw.startswith(b"RA"):
                relevant_records += 1
                ra_source_paths.add(snapshot.path_text)
                key = _parse_race_key(raw, label="RA")
                if key in races:
                    raise ValueError(f"duplicate RA race key: {key}")
                races[key] = RaceIdentity(
                    target_race_key=key,
                    race_date=key[:8],
                    venue_code=key[8:10],
                    source_path=snapshot.path_text,
                    source_sha256=snapshot.sha256,
                    record_sha256=_sha256(raw),
                    raw=raw,
                )
            elif raw.startswith(b"SE"):
                relevant_records += 1
                se_source_paths.add(snapshot.path_text)
                key = _parse_race_key(raw, label="SE")
                horse_id = canonical_horse_id(_ascii(raw, 30, 10))
                if len(horse_id) != 10:
                    raise ValueError(f"SE record has invalid horse id: {horse_id!r}")
                identity = (key, horse_id)
                if identity in runners:
                    raise ValueError(f"duplicate SE runner identity: {identity}")
                runners[identity] = RunnerIdentity(
                    target_race_key=key,
                    race_date=key[:8],
                    venue_code=key[8:10],
                    horse_id=horse_id,
                    source_path=snapshot.path_text,
                    source_sha256=snapshot.sha256,
                    record_sha256=_sha256(raw),
                    raw=raw,
                )
        if relevant_records == 0:
            raise ValueError(f"declared history source contains no RA or SE records: {snapshot.path}")
    if not races:
        raise ValueError("declared history sources contain no RA records")
    if not runners:
        raise ValueError("declared history sources contain no SE records")
    return races, runners, ra_source_paths, se_source_paths


def _venue_class(venue_code: str) -> str:
    if venue_code in JRA_VENUE_CODES:
        return "JRA"
    if venue_code in LOCAL_VENUE_CODES:
        return "LOCAL"
    if re.search(r"[A-Z]", venue_code):
        return "FOREIGN"
    return "UNKNOWN"


def _combined_hash(values: Iterable[str]) -> str:
    return _sha256(canonical_json(sorted(set(values))).encode("utf-8"))


def _authority_row(
    *,
    horse_id: str,
    target_date: str,
    latest: RunnerIdentity | None,
    race: RaceIdentity | None,
    contract_ok: bool,
    reason: str,
    source_system: str,
    source_file_count: int,
) -> dict[str, Any]:
    file_hashes = [value for value in (
        latest.source_sha256 if latest else "",
        race.source_sha256 if race else "",
    ) if value]
    record_hashes = [value for value in (
        latest.record_sha256 if latest else "",
        race.record_sha256 if race else "",
    ) if value]
    return {
        "horse_id": horse_id,
        "authoritative_latest_race_id": (
            canonical_authority_race_id(latest.target_race_key) if latest else None
        ),
        "authoritative_latest_race_date": latest.race_date if latest else None,
        "authority_contract_ok": bool(contract_ok),
        "authority_failure_reason": reason,
        "authority_source_file_hash": _combined_hash(file_hashes) if file_hashes else "",
        "authority_source_record_hash": _combined_hash(record_hashes) if record_hashes else "",
        "authority_source_system": source_system,
        "ra_source_path": race.source_path if race else "",
        "ra_source_sha256": race.source_sha256 if race else "",
        "se_source_path": latest.source_path if latest else "",
        "se_source_sha256": latest.source_sha256 if latest else "",
        "source_event_time_max": latest.race_date if latest else None,
        "source_file_count": source_file_count,
        "target_date": target_date,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def build_authority_rows(
    current_runners: Sequence[CurrentRunner],
    races: dict[str, RaceIdentity],
    history_runners: dict[tuple[str, str], RunnerIdentity],
    *,
    target_date: str,
    authority_coverage: str,
    source_file_count: int,
) -> list[dict[str, Any]]:
    by_horse: dict[str, list[RunnerIdentity]] = {}
    for record in history_runners.values():
        if record.race_date < target_date:
            by_horse.setdefault(record.horse_id, []).append(record)
    rows: list[dict[str, Any]] = []
    for current in current_runners:
        candidates = by_horse.get(current.horse_id, [])
        if not candidates:
            confirmed = authority_coverage == COMPLETE_COVERAGE
            rows.append(
                _authority_row(
                    horse_id=current.horse_id,
                    target_date=target_date,
                    latest=None,
                    race=None,
                    contract_ok=confirmed,
                    reason=(
                        "NO_PRIOR_RACE_CONFIRMED"
                        if confirmed
                        else "AUTHORITATIVE_HISTORY_UNAVAILABLE"
                    ),
                    source_system="JRA_VAN_RCOV_SU_SR",
                    source_file_count=source_file_count,
                )
            )
            continue
        latest_date = max(record.race_date for record in candidates)
        latest_rows = [record for record in candidates if record.race_date == latest_date]
        if len(latest_rows) != 1:
            rows.append(
                _authority_row(
                    horse_id=current.horse_id,
                    target_date=target_date,
                    latest=None,
                    race=None,
                    contract_ok=False,
                    reason="MULTIPLE_LATEST_PRIOR_RACES_SAME_DATE",
                    source_system="JRA_VAN_RCOV_SU_SR",
                    source_file_count=source_file_count,
                )
            )
            continue
        latest = latest_rows[0]
        venue_class = _venue_class(latest.venue_code)
        race = races.get(latest.target_race_key)
        if venue_class != "JRA":
            rows.append(
                _authority_row(
                    horse_id=current.horse_id,
                    target_date=target_date,
                    latest=latest,
                    race=race,
                    contract_ok=False,
                    reason=f"LATEST_PRIOR_RACE_{venue_class}",
                    source_system=f"JRA_VAN_RCOV_{venue_class}",
                    source_file_count=source_file_count,
                )
            )
            continue
        if race is None:
            rows.append(
                _authority_row(
                    horse_id=current.horse_id,
                    target_date=target_date,
                    latest=latest,
                    race=None,
                    contract_ok=False,
                    reason="LATEST_JRA_RACE_MISSING_RA_RECORD",
                    source_system="JRA_VAN_RCOV_JRA",
                    source_file_count=source_file_count,
                )
            )
            continue
        try:
            parse_ra_record(
                race.raw,
                source_file=race.source_path,
                source_file_sha256=race.source_sha256,
            )
            parsed_runner = parse_se_record(
                latest.raw,
                source_file=latest.source_path,
                source_file_sha256=latest.source_sha256,
            )
            if parsed_runner.horse_id != current.horse_id:
                raise ValueError("parsed runner identity drift")
        except ValueError:
            rows.append(
                _authority_row(
                    horse_id=current.horse_id,
                    target_date=target_date,
                    latest=latest,
                    race=race,
                    contract_ok=False,
                    reason="LATEST_JRA_RECORD_NOT_FEATURE_READY",
                    source_system="JRA_VAN_RCOV_JRA",
                    source_file_count=source_file_count,
                )
            )
            continue
        rows.append(
            _authority_row(
                horse_id=current.horse_id,
                target_date=target_date,
                latest=latest,
                race=race,
                contract_ok=True,
                reason="",
                source_system="JRA_VAN_RCOV_JRA",
                source_file_count=source_file_count,
            )
        )
    return rows


def _source_entries(
    snapshots: Sequence[FileSnapshot], source_paths: set[str]
) -> list[dict[str, str]]:
    return [
        {"path": snapshot.path_text, "sha256": snapshot.sha256}
        for snapshot in snapshots
        if snapshot.path_text in source_paths
    ]


def build_manifest(
    *,
    experiment_id: str,
    du_path: Path,
    du_sha256: str,
    fold_manifest_path: Path,
    fold_manifest_sha256: str,
    history_paths: Sequence[Path],
    target_date: str,
    expected_runner_count: int,
    authority_coverage: str,
) -> dict[str, Any]:
    target_date = _parse_date(target_date, label="target date")
    if authority_coverage not in {COMPLETE_COVERAGE, PARTIAL_COVERAGE}:
        raise ValueError(f"unsupported authority coverage: {authority_coverage}")
    du = snapshot_file(du_path, expected_sha256=du_sha256)
    fold = snapshot_file(fold_manifest_path, expected_sha256=fold_manifest_sha256)
    current_runners = parse_current_du(du.payload, target_date=target_date)
    if len(current_runners) != expected_runner_count:
        raise ValueError(
            f"current runner count mismatch: {len(current_runners)} != {expected_runner_count}"
        )
    snapshots = [snapshot_file(path) for path in history_paths]
    races, history_runners, ra_paths, se_paths = parse_history_snapshots(snapshots)
    authority_rows = build_authority_rows(
        current_runners,
        races,
        history_runners,
        target_date=target_date,
        authority_coverage=authority_coverage,
        source_file_count=len(snapshots),
    )
    if len(authority_rows) != expected_runner_count:
        raise ValueError("authority row denominator drift")
    authority_ids = [row["horse_id"] for row in authority_rows]
    if len(authority_ids) != len(set(authority_ids)):
        raise ValueError("authority rows contain duplicate horse ids")
    if any(
        row["authoritative_latest_race_date"] is not None
        and row["authoritative_latest_race_date"] >= target_date
        for row in authority_rows
    ):
        raise ValueError("authority rows contain same-day or future history")
    reason_counts: dict[str, int] = {}
    for row in authority_rows:
        reason = str(row["authority_failure_reason"] or "ACCEPTED")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    source_event_dates = [
        str(row["source_event_time_max"])
        for row in authority_rows
        if row["source_event_time_max"] is not None
    ]
    implementation = snapshot_file(Path(__file__))
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "manifest_type": "target_direct_authoritative_history_source_inventory",
        "target_date": target_date,
        "source_time_contract": SOURCE_TIME_CONTRACT,
        "authority_coverage": authority_coverage,
        "source_event_time_max": max(source_event_dates) if source_event_dates else None,
        "source_file_count": len(snapshots),
        "du_source": {"path": du.path_text, "sha256": du.sha256},
        "fold_manifest": {"path": fold.path_text, "sha256": fold.sha256},
        "implementation": {
            "path": implementation.path_text,
            "sha256": implementation.sha256,
        },
        "ra_sources": _source_entries(snapshots, ra_paths),
        "se_sources": _source_entries(snapshots, se_paths),
        "authoritative_latest_races": authority_rows,
        "summary": {
            "registered_runner_rows": len(current_runners),
            "authority_rows": len(authority_rows),
            "authority_contract_ok_rows": sum(
                bool(row["authority_contract_ok"]) for row in authority_rows
            ),
            "authority_contract_failed_rows": sum(
                not bool(row["authority_contract_ok"]) for row in authority_rows
            ),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "safety": {
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
            "prediction_rows": 0,
            "market_rows": 0,
        },
    }


def write_canonical_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"refusing to overwrite existing manifest: {path}")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only, hash-bound TARGET direct-history source inventory. "
            "This command performs no prediction, market, ROI, BUY, or order operation."
        )
    )
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--du", type=Path, required=True)
    parser.add_argument("--du-sha256", required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest-sha256", required=True)
    parser.add_argument("--history-glob", action="append", required=True)
    parser.add_argument("--allowed-source-root", type=Path, action="append", required=True)
    parser.add_argument("--allowed-year", action="append", choices=("2025", "2026"), required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--expected-runner-count", type=int, required=True)
    parser.add_argument(
        "--authority-coverage",
        choices=(COMPLETE_COVERAGE, PARTIAL_COVERAGE),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    history_paths = expand_history_sources(
        args.history_glob,
        allowed_roots=args.allowed_source_root,
        allowed_years=set(args.allowed_year),
    )
    manifest = build_manifest(
        experiment_id=args.experiment_id,
        du_path=args.du,
        du_sha256=args.du_sha256,
        fold_manifest_path=args.fold_manifest,
        fold_manifest_sha256=args.fold_manifest_sha256,
        history_paths=history_paths,
        target_date=args.target_date,
        expected_runner_count=args.expected_runner_count,
        authority_coverage=args.authority_coverage,
    )
    write_canonical_json_exclusive(args.output, manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
