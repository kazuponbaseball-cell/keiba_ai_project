from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from registered_nonpromotion_contract_v1 import (
    ContractError,
    canonical_digest,
    evaluate_registered_decisions,
    load_strict_json,
)
from registered_nonpromotion_offline_contract_v1 import (
    DEFAULT_BASE_BRANCH,
    DEFAULT_REPOSITORY,
    FIXED_OUTPUT_ROOT,
    GATE_KIND,
    MATERIALIZATION_MANIFEST_PATH,
    POLICY_RELATIVE_PATH,
    PROJECTION_INPUTS,
    RECIPE_ID,
    RECIPE_VERSION,
    RUN_SCOPE_ARTIFACT_DIRECTORY,
    SOURCE_INPUTS,
    compile_offline_run_scope,
    offline_run_scope_artifact_path,
    resolve_offline_registered_recipe,
    verify_canonical_offline_run_scope,
)


MATERIALIZATION_MANIFEST_KIND = "EXACT_USECOLS_PROJECTION_MATERIALIZATION_V1"
RUNNER_TEMPLATE_ID = "REGISTERED_NONPROMOTION_OFFLINE_EXECUTOR_V1"
REPLICA_IDS = ("clean_a", "clean_b")
REPLICA_MODE = "LOGICAL_SAME_PROCESS_SHARED_SEALED_INPUT_BYTES"
FOLD_VALUES = ("fold2", "fold3", "fold4")
FOLD_COUNTS = {"fold2": 1661, "fold3": 1653, "fold4": 432}
RACE_COUNT = 3746
DATE_MIN = "2025-01-05"
DATE_MAX = "2026-02-15"
CALIBRATOR_OFFSET = 0.130654047367905
CALIBRATOR_ABS_TOLERANCE = 1e-12
BOOTSTRAP_REPLICATES = 100000
BOOTSTRAP_SEED = 20260814

CANDIDATE_COLUMNS = (
    "candidate_generated",
    "candidate_key",
    "eligible_race",
    "fold",
    "horse_a",
    "horse_b",
    "p_action_C0_offset",
    "race_date",
    "race_id",
    "top1_wide_prob",
    "venue_code",
)
SETTLEMENT_COLUMNS = (
    "race_id",
    "candidate_key",
    "candidate_hit",
    "official_outcome_completeness",
    "official_wide_pay",
)
FORBIDDEN_PROJECTION_TOKENS = ("odds", "price", "popularity", "market", "roi")
LIMITATIONS = {
    "single_use_policy": "ONE_ACCEPTED_EXECUTION",
    "single_use_enforcement": "BEST_EFFORT_LOCAL_EXCLUSIVE_RECEIPT",
    "global_replay_proof": False,
    "rollback_resistant": False,
    "durable_remote_ledger": False,
    "network_isolation": "APPLICATION_LEVEL_NOT_OS_SANDBOX",
}
NONPROMOTION = {
    "evidence_purpose_class": "DIAGNOSTIC_NONPROMOTION",
    "source_authority_class": "B_LOCAL_HASHED",
    "reused_development_oos": True,
    "strict_t3_rows": 0,
    "confirmatory": False,
    "promotion_eligible": False,
    "score_credit": 0,
    "shadow_transition_supported": False,
    "production_transition_supported": False,
    "formal_buy": False,
    "send_order": False,
    "stake": 0,
}

FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class OfflineFirewallViolation(ContractError):
    pass


class OfflineInvalidAfterStart(ContractError):
    pass


class OfflineRunAlreadyCompleted(ContractError):
    pass


class OfflineRunAlreadyRunning(ContractError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ContractError("artifact is not finite canonical JSON") from exc
    return (text + "\n").encode("utf-8")


def _raw_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_parent(path)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, _canonical_bytes(dict(value)))


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_bytes_exclusive(
        path, b"".join(_canonical_bytes(dict(row)) for row in rows)
    )


def _atomic_publish_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ContractError(f"refusing to overwrite {path.name}")
    # Keep a pre-commit payload outside the consumed output root.  If the
    # process is killed before replace, recovery (under the run lock) removes
    # only this exact fixed-prefix sibling before classifying the tombstone.
    temporary = path.parent.parent / (
        f".{path.parent.name}.result.{uuid.uuid4().hex}.tmp"
    )
    try:
        _write_json_exclusive(temporary, value)
        if path.exists():
            raise ContractError(f"refusing to overwrite {path.name}")
        # The replace is the irreversible local completion point. The temporary
        # payload itself was fsynced; this lower-assurance route does not claim a
        # durable directory-fsync/rollback-resistant commit after replacement.
        os.replace(temporary, path)
    except BaseException:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        raise


def _orphan_result_temp_paths(output_root: Path) -> list[Path]:
    parent = output_root.parent
    if not parent.is_dir() or _is_reparse(parent):
        return []
    prefix = f".{output_root.name}.result."
    suffix = ".tmp"
    matches: list[Path] = []
    for child in parent.iterdir():
        name = child.name
        token = name[len(prefix) : -len(suffix)] if (
            name.startswith(prefix) and name.endswith(suffix)
        ) else ""
        if len(token) == 32 and all(character in "0123456789abcdef" for character in token):
            matches.append(child)
    return matches


def _cleanup_orphan_result_temps(output_root: Path) -> None:
    for path in _orphan_result_temp_paths(output_root):
        if path.is_symlink() or _is_reparse(path) or not path.is_file():
            raise ContractError("unsafe orphan result staging artifact")
        path.unlink()
    if _orphan_result_temp_paths(output_root):
        raise ContractError("orphan result staging artifact cleanup failed")


@contextmanager
def _exclusive_run_lock(output_root: Path) -> Iterator[None]:
    """Hold a process-backed nonblocking lock for the full local invocation."""

    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / f".{output_root.name}.run.lock"
    if lock_path.exists() or lock_path.is_symlink():
        if _is_reparse(lock_path) or not lock_path.is_file():
            raise ContractError("offline run lock path is unsafe")
    flags = os.O_RDWR | os.O_CREAT
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(lock_path), flags | nofollow, 0o600)
    acquired = False
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (BlockingIOError, OSError) as exc:
            raise OfflineRunAlreadyRunning(
                "offline run is already active under the local process lock"
            ) from exc
        yield
    finally:
        if acquired:
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)


def _is_reparse(path: Path) -> bool:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _safe_root(root: Path, *, label: str) -> Path:
    raw = os.fspath(root)
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise ContractError(f"{label} may not be a UNC path")
    lexical = Path(os.path.abspath(raw))
    current = Path(lexical.anchor) if lexical.anchor else Path()
    lexical_parts = lexical.parts[1:] if lexical.anchor else lexical.parts
    for part in lexical_parts:
        current = current / part
        if current.exists() or current.is_symlink():
            try:
                if _is_reparse(current):
                    raise ContractError(
                        f"{label} may not traverse a symlink/junction/reparse point"
                    )
            except OSError as exc:
                raise ContractError(f"{label} cannot be safely inspected") from exc
    resolved = Path(root).resolve(strict=True)
    if not resolved.is_dir() or _is_reparse(resolved):
        raise ContractError(f"{label} must be an existing non-reparse directory")
    return resolved


def _safe_relative(raw: str, *, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw or ":" in raw:
        raise ContractError(f"{label} must be a normalized repository-relative path")
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or pure.as_posix() != raw
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ContractError(f"{label} must be a normalized repository-relative path")
    return pure


def _safe_bound_path(
    root: Path,
    relative: str,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    root = _safe_root(root, label=f"{label} root")
    pure = _safe_relative(relative, label=label)
    candidate = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if _is_reparse(current):
                raise ContractError(f"{label} traverses a symlink/junction/reparse point")
    if must_exist:
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file() or _is_reparse(resolved):
            raise ContractError(f"{label} must be an existing regular file")
    else:
        resolved = candidate.resolve(strict=False)
    try:
        common = os.path.commonpath((os.path.normcase(str(root)), os.path.normcase(str(resolved))))
    except ValueError as exc:
        raise ContractError(f"{label} escapes its root") from exc
    if os.path.normcase(common) != os.path.normcase(str(root)):
        raise ContractError(f"{label} escapes its root")
    return candidate


def _read_verified_source_bytes(
    path: Path, *, expected_sha256: str, role: str
) -> tuple[bytes, dict[str, Any]]:
    with path.open("rb") as handle:
        payload = handle.read()
    observed_sha = hashlib.sha256(payload).hexdigest()
    if observed_sha != expected_sha256:
        raise ContractError(f"raw source hash mismatch: {role}")
    return payload, {
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha,
        "observed_byte_size": len(payload),
    }


def _projected_csv_rows(payload: bytes, usecols: Sequence[str]) -> Iterator[dict[str, str]]:
    """Tokenize CSV records but expose typed semantics only for explicit usecols."""

    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ContractError("CSV source is not valid UTF-8") from exc
    with io.StringIO(decoded, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ContractError("CSV source is empty") from exc
        if len(header) != len(set(header)):
            raise ContractError("CSV source has duplicate headers")
        missing = [column for column in usecols if column not in header]
        if missing:
            raise ContractError(f"CSV source lacks required usecols: {missing}")
        indexes = tuple(header.index(column) for column in usecols)
        for line_number, record in enumerate(reader, start=2):
            if len(record) != len(header):
                raise ContractError(f"CSV record width mismatch at line {line_number}")
            yield {column: record[index] for column, index in zip(usecols, indexes)}


def _required_raw_cell(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"{label} must be a non-empty whitespace-free CSV cell")
    return value


def _parse_bool(value: Any, *, label: str) -> bool:
    raw = _required_raw_cell(value, label=label)
    if raw in {"1", "true", "True"}:
        return True
    if raw in {"0", "false", "False"}:
        return False
    raise ContractError(f"{label} must use a registered boolean serialization")


def _parse_positive_horse(value: Any, *, label: str) -> int:
    raw = _required_raw_cell(value, label=label)
    if not raw.isdigit() or int(raw) <= 0 or str(int(raw)) != raw:
        raise ContractError(f"{label} must be a positive decimal integer without padding")
    return int(raw)


def _candidate_key(horse_a: int, horse_b: int) -> str:
    if horse_a == horse_b:
        raise ContractError("candidate horse numbers must be distinct")
    return f"{min(horse_a, horse_b)}-{max(horse_a, horse_b)}"


def _finite_probability(raw: Any, *, label: str, open_interval: bool = False) -> float:
    text = _required_raw_cell(raw, label=label)
    try:
        value = float(text)
    except ValueError as exc:
        raise ContractError(f"{label} must be binary64-compatible decimal") from exc
    if not math.isfinite(value):
        raise ContractError(f"{label} must be finite")
    if open_interval:
        if not 0.0 < value < 1.0:
            raise ContractError(f"{label} must be in (0,1)")
    elif not 0.0 <= value <= 1.0:
        raise ContractError(f"{label} must be in [0,1]")
    return value


def _expected_p_action(p: float) -> float:
    z = math.log(p) - math.log1p(-p) + CALIBRATOR_OFFSET
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exponential = math.exp(z)
    return exponential / (1.0 + exponential)


def _verify_p_action_formula(p_raw: str, a_raw: str) -> None:
    p = _finite_probability(p_raw, label="top1_wide_prob", open_interval=True)
    a = _finite_probability(a_raw, label="p_action_C0_offset")
    expected = _expected_p_action(p)
    if abs(a - expected) > CALIBRATOR_ABS_TOLERANCE:
        raise ContractError("stored p_action_C0_offset differs from the fixed calibrator formula")


def _fold_counter(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(row["fold"]) for row in rows)


def _fixed_race_date(raw: Any, *, label: str) -> str:
    value = _required_raw_cell(raw, label=label)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ContractError(f"{label} must use canonical YYYY-MM-DD") from exc
    if parsed.strftime("%Y-%m-%d") != value or not DATE_MIN <= value <= DATE_MAX:
        raise ContractError(f"{label} is outside the fixed cohort date range")
    return value


def _materialize_core(
    *,
    repo_root: Path,
    source_root: Path,
    policy: Mapping[str, Any],
    implementation_binding: Mapping[str, str],
    expected_source_inputs: Mapping[str, Mapping[str, str]],
    expected_race_count: int,
    expected_fold_counts: Mapping[str, int],
) -> dict[str, Any]:
    projection = policy["projection_contract"]
    usecols = projection["raw_source_allowed_projection_columns"]
    candidate_relative = projection["candidate_projection_path"]
    settlement_relative = projection["settlement_projection_path"]
    manifest_relative = projection["materialization_manifest_path"]
    candidate_final = _safe_bound_path(
        repo_root, candidate_relative, label="candidate projection", must_exist=False
    )
    settlement_final = _safe_bound_path(
        repo_root, settlement_relative, label="settlement projection", must_exist=False
    )
    manifest_final = _safe_bound_path(
        repo_root, manifest_relative, label="materialization manifest", must_exist=False
    )
    if not (candidate_final.parent == settlement_final.parent == manifest_final.parent):
        raise ContractError("materialized artifacts must share the fixed directory")
    final_directory = candidate_final.parent
    if final_directory.exists():
        raise ContractError("materialization directory already exists; overwrite is forbidden")
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    _safe_bound_path(
        repo_root,
        PurePosixPath(candidate_relative).parent.as_posix(),
        label="materialization directory",
        must_exist=False,
    )
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=".offline-materialize-", dir=final_directory.parent)
    )
    candidate_temp = temporary_directory / candidate_final.name
    settlement_temp = temporary_directory / settlement_final.name
    manifest_temp = temporary_directory / manifest_final.name
    source_bindings: dict[str, dict[str, Any]] = {}
    try:
        source_payloads: dict[str, bytes] = {}
        for role in ("diagnostic_master", "p_action_artifact"):
            specification = expected_source_inputs[role]
            path = _safe_bound_path(
                source_root, specification["path"], label=f"raw source {role}", must_exist=True
            )
            payload, observed = _read_verified_source_bytes(
                path,
                expected_sha256=specification["expected_sha256"],
                role=role,
            )
            source_payloads[role] = payload
            source_bindings[role] = {
                "path": specification["path"],
                **observed,
            }

        master_by_key: dict[tuple[str, str], dict[str, str]] = {}
        candidate_rows: list[dict[str, Any]] = []
        seen_race_ids: set[str] = set()
        for raw in _projected_csv_rows(source_payloads["diagnostic_master"], usecols["diagnostic_master"]):
            fold = _required_raw_cell(raw["fold"], label="fold")
            if fold not in FOLD_VALUES:
                continue
            race_id = _required_raw_cell(raw["race_id"], label="race_id")
            if not race_id.isdigit() or len(race_id) != 16:
                raise ContractError("race_id must be a 16-digit string")
            if race_id in seen_race_ids:
                raise ContractError("diagnostic master must have one candidate row per race")
            seen_race_ids.add(race_id)
            horse_a = _parse_positive_horse(raw["horse_a"], label="horse_a")
            horse_b = _parse_positive_horse(raw["horse_b"], label="horse_b")
            canonical_key = _candidate_key(horse_a, horse_b)
            source_key = _required_raw_cell(raw["top1_pair_key"], label="top1_pair_key")
            if source_key != canonical_key:
                raise ContractError("top1_pair_key differs from the canonical unordered horse pair")
            _verify_p_action_formula(raw["top1_wide_prob"], raw["p_action_C0_offset"])
            key = (race_id, canonical_key)
            if key in master_by_key:
                raise ContractError("duplicate diagnostic-master candidate key")
            master_by_key[key] = raw
            candidate_rows.append(
                {
                    "candidate_generated": _parse_bool(raw["candidate_generated"], label="candidate_generated"),
                    "candidate_key": canonical_key,
                    "eligible_race": _parse_bool(raw["eligible_race"], label="eligible_race"),
                    "fold": fold,
                    "horse_a": horse_a,
                    "horse_b": horse_b,
                    "p_action_C0_offset": _required_raw_cell(raw["p_action_C0_offset"], label="p_action_C0_offset"),
                    "race_date": _fixed_race_date(raw["race_date"], label="race_date"),
                    "race_id": race_id,
                    "top1_wide_prob": _required_raw_cell(raw["top1_wide_prob"], label="top1_wide_prob"),
                    "venue_code": _required_raw_cell(raw["venue_code"], label="venue_code"),
                }
            )

        p_action_by_key: dict[tuple[str, str], dict[str, str]] = {}
        for raw in _projected_csv_rows(source_payloads["p_action_artifact"], usecols["p_action_artifact"]):
            fold = _required_raw_cell(raw["fold"], label="p-action fold")
            if fold not in FOLD_VALUES:
                continue
            key = (
                _required_raw_cell(raw["race_id"], label="p-action race_id"),
                _required_raw_cell(raw["top1_pair_key"], label="p-action top1_pair_key"),
            )
            if key in p_action_by_key:
                raise ContractError("duplicate p-action candidate key")
            p_action_by_key[key] = raw
        if set(master_by_key) != set(p_action_by_key):
            raise ContractError("diagnostic-master and p-action cohort keys differ")
        equality_columns = tuple(projection["candidate_cross_source_equality_columns"])
        equality_vector: list[dict[str, str]] = []
        for key in sorted(master_by_key):
            left = master_by_key[key]
            right = p_action_by_key[key]
            for column in equality_columns:
                if left[column] != right[column]:
                    raise ContractError(f"p-action cross-source equality failed for {column}")
            _verify_p_action_formula(right["top1_wide_prob"], right["p_action_C0_offset"])
            equality_vector.append({column: left[column] for column in equality_columns})
        candidate_rows.sort(key=lambda row: (row["race_id"], row["candidate_key"]))
        if len(candidate_rows) != expected_race_count:
            raise ContractError("candidate projection race count differs from the fixed cohort")
        if _fold_counter(candidate_rows) != Counter(expected_fold_counts):
            raise ContractError("candidate projection fold counts differ from the fixed cohort")
        dates = [str(row["race_date"]) for row in candidate_rows]
        if min(dates) != DATE_MIN or max(dates) != DATE_MAX:
            raise ContractError("candidate projection date range differs from the fixed cohort")
        if any(set(row) != set(CANDIDATE_COLUMNS) for row in candidate_rows):
            raise ContractError("candidate projection schema is not exact")
        candidate_sealed_payload = b"".join(
            _canonical_bytes(row) for row in candidate_rows
        )
        _write_bytes_exclusive(candidate_temp, candidate_sealed_payload)
        candidate_sha = hashlib.sha256(candidate_sealed_payload).hexdigest()
        candidate_size = len(candidate_sealed_payload)

        payoff_specification = expected_source_inputs["official_payoff_source"]
        payoff_path = _safe_bound_path(
            source_root,
            payoff_specification["path"],
            label="raw source official_payoff_source",
            must_exist=True,
        )
        payoff_payload, payoff_observed = _read_verified_source_bytes(
            payoff_path,
            expected_sha256=payoff_specification["expected_sha256"],
            role="official_payoff_source",
        )
        source_bindings["official_payoff_source"] = {
            "path": payoff_specification["path"],
            **payoff_observed,
        }
        candidate_race_ids = {str(row["race_id"]) for row in candidate_rows}
        payoff_by_key: dict[tuple[str, str], str] = {}
        settled_races: set[str] = set()
        payoff_rows_per_race: Counter[str] = Counter()
        for raw in _projected_csv_rows(payoff_payload, usecols["official_payoff_source"]):
            race_id = _required_raw_cell(raw["race_id"], label="payoff race_id")
            if race_id not in candidate_race_ids:
                continue
            horse_a = _parse_positive_horse(raw["horse_a"], label="payoff horse_a")
            horse_b = _parse_positive_horse(raw["horse_b"], label="payoff horse_b")
            key = (race_id, _candidate_key(horse_a, horse_b))
            payoff = _required_raw_cell(raw["wide_pay"], label="wide_pay")
            try:
                payoff_value = float(payoff)
            except ValueError as exc:
                raise ContractError("wide_pay must be binary64-compatible decimal") from exc
            if not math.isfinite(payoff_value) or payoff_value <= 0:
                raise ContractError("wide_pay must be finite and positive")
            if key in payoff_by_key:
                raise ContractError("official payoff source has duplicate race/pair rows")
            payoff_by_key[key] = payoff
            settled_races.add(race_id)
            payoff_rows_per_race[race_id] += 1
        for race_id in candidate_race_ids:
            if not 3 <= payoff_rows_per_race[race_id] <= 7:
                raise ContractError(
                    "official payoff source does not contain a complete 3-to-7-row wide payout set"
                )
        settlement_rows: list[dict[str, Any]] = []
        for candidate in candidate_rows:
            race_id = str(candidate["race_id"])
            key = (race_id, str(candidate["candidate_key"]))
            if race_id not in settled_races:
                raise ContractError("official payoff source lacks an enrolled settled race")
            hit = key in payoff_by_key
            settlement_rows.append(
                {
                    "race_id": race_id,
                    "candidate_key": candidate["candidate_key"],
                    "candidate_hit": hit,
                    "official_outcome_completeness": True,
                    "official_wide_pay": payoff_by_key[key] if hit else None,
                }
            )
        if len(settlement_rows) != expected_race_count:
            raise ContractError("settlement projection race count differs from the fixed cohort")
        if any(set(row) != set(SETTLEMENT_COLUMNS) for row in settlement_rows):
            raise ContractError("settlement projection schema is not exact")
        _write_jsonl_exclusive(settlement_temp, settlement_rows)
        settlement_sha, settlement_size = _raw_sha256(settlement_temp)
        if candidate_sha == settlement_sha:
            raise ContractError("candidate and settlement projections must be distinct")
        if candidate_temp.read_bytes() != candidate_sealed_payload:
            raise ContractError("candidate projection changed after its pre-settlement seal")

        projection_bindings = {
            "candidate_projection": {
                "path": candidate_relative,
                "sha256": candidate_sha,
                "byte_size": candidate_size,
            },
            "settlement_projection": {
                "path": settlement_relative,
                "sha256": settlement_sha,
                "byte_size": settlement_size,
            },
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "gate_kind": GATE_KIND,
            "manifest_kind": MATERIALIZATION_MANIFEST_KIND,
            "authority": False,
            "implementation_binding": dict(implementation_binding),
            "source_bindings": source_bindings,
            "source_usecols": {role: list(usecols[role]) for role in sorted(usecols)},
            "candidate_cross_source_equality": {
                "columns": list(equality_columns),
                "row_count": len(equality_vector),
                "vector_sha256": canonical_digest(equality_vector),
                "exact": True,
            },
            "p_action_formula_attestation": {
                "formula": projection["p_action_lineage_formula"],
                "numeric_semantics": projection["p_action_numeric_semantics"],
                "absolute_tolerance": projection["p_action_absolute_tolerance"],
                "row_count": len(candidate_rows),
                "all_rows_verified": True,
            },
            "candidate_identity_attestation": {
                "formula": projection["candidate_key_formula"],
                "row_count": len(candidate_rows),
                "all_rows_verified": True,
            },
            "ordered_race_id_sha256": canonical_digest(
                [row["race_id"] for row in candidate_rows]
            ),
            "projection_bindings": projection_bindings,
            "projection_rows": {
                "candidate_projection": len(candidate_rows),
                "settlement_projection": len(settlement_rows),
            },
            "projection_columns": {
                "candidate_projection": list(CANDIDATE_COLUMNS),
                "settlement_projection": list(SETTLEMENT_COLUMNS),
            },
            "candidate_projection_sealed_before_settlement_source_open": True,
            "raw_forbidden_semantic_values_selected": False,
            "decisions_computed": False,
            "metrics_computed": False,
            "roi_computed": False,
            "odds_price_popularity_or_market_values_persisted": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        manifest["manifest_digest"] = canonical_digest(manifest)
        _write_json_exclusive(manifest_temp, manifest)
        manifest_sha, manifest_size = _raw_sha256(manifest_temp)
        if final_directory.exists():
            raise ContractError("materialization directory was created concurrently")
        os.replace(temporary_directory, final_directory)
        _fsync_parent(final_directory)
        return {
            "source_bindings": source_bindings,
            "projection_bindings": projection_bindings,
            "materialization_manifest": {
                "path": manifest_relative,
                "sha256": manifest_sha,
                "byte_size": manifest_size,
            },
            "manifest_digest": manifest["manifest_digest"],
            "decisions_computed": False,
            "metrics_computed": False,
            "roi_computed": False,
        }
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)


def _approval_module() -> Any:
    import registered_nonpromotion_offline_approval_v1 as approval

    return approval


def _verify_clean_git_head(root: Path, expected_head: str) -> None:
    if not FULL_GIT_SHA.fullmatch(expected_head):
        raise ContractError("expected current-main SHA is invalid")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.stdout.strip() != expected_head:
        raise ContractError("local HEAD is not the verified current GitHub main")
    status_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if status_result.stdout:
        raise ContractError("worktree is not clean current-main")


def _materialize_fixed_projections_worker(
    *,
    root: Path,
    source_root: Path,
    provider: Any,
    now: str | Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Create the two fixed projections after metadata-only gate availability."""

    repo_root = _safe_root(Path(root), label="repository root")
    raw_root = _safe_root(Path(source_root), label="source root")
    availability = _approval_module().verify_offline_gate_availability(
        root=repo_root,
        provider=provider,
        now=now,
    )
    expected_head = availability.get("verified_current_main_sha")
    if not isinstance(expected_head, str):
        raise ContractError("gate availability lacks verified current main")
    _verify_clean_git_head(repo_root, expected_head)
    registered = resolve_offline_registered_recipe(repo_root)
    return _materialize_core(
        repo_root=repo_root,
        source_root=raw_root,
        policy=registered.policy,
        implementation_binding={
            "implementation_commit": availability["implementation_commit"],
            "runtime_material_bundle_sha256": canonical_digest(
                dict(registered.runtime_material_digests)
            ),
        },
        expected_source_inputs=SOURCE_INPUTS,
        expected_race_count=RACE_COUNT,
        expected_fold_counts=FOLD_COUNTS,
    )


def materialize_fixed_projections(
    *, root: Path, source_root: Path
) -> dict[str, Any]:
    """Canonical materializer with real time and an internal GitHub provider."""

    return _materialize_fixed_projections_worker(
        root=root,
        source_root=source_root,
        provider=_github_provider(),
        now=None,
    )


def _verify_self_digest(value: Mapping[str, Any], *, field: str, label: str) -> str:
    stored = value.get(field)
    if not isinstance(stored, str) or not FULL_SHA256.fullmatch(stored):
        raise ContractError(f"{label} digest is missing or invalid")
    unsigned = dict(value)
    unsigned.pop(field, None)
    if canonical_digest(unsigned) != stored:
        raise ContractError(f"{label} digest mismatch")
    return stored


def _verify_file_binding(root: Path, binding: Mapping[str, Any], *, label: str) -> Path:
    path, _payload = _read_bound_bytes(root, binding, label=label)
    return path


def _read_bound_bytes(
    root: Path, binding: Mapping[str, Any], *, label: str
) -> tuple[Path, bytes]:
    if set(binding) != {"path", "sha256", "byte_size"}:
        raise ContractError(f"{label} binding shape differs from the fixed contract")
    path = _safe_bound_path(
        root, binding["path"], label=label, must_exist=True
    )
    with path.open("rb") as handle:
        payload = handle.read()
    observed_sha = hashlib.sha256(payload).hexdigest()
    if observed_sha != binding.get("sha256") or len(payload) != binding.get("byte_size"):
        raise ContractError(f"{label} hash or byte size differs from the frozen scope")
    return path, payload


def _load_strict_json_bytes(payload: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=strict_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"{label} is not strict UTF-8 JSON") from exc


def _snapshot_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    snapshot = _load_strict_json_bytes(
        _canonical_bytes(dict(value)), label=f"{label} snapshot"
    )
    if not isinstance(snapshot, dict):
        raise ContractError(f"{label} snapshot must be an object")
    return snapshot


def _snapshot_approval_evidence(
    value: Mapping[str, Any],
    *,
    checkpoint: str,
    issue_number: int,
    comment_id: int,
    expected_run_scope_digest: str | None = None,
    expected_original_evidence_digest: str | None = None,
    strict_live_evidence: bool = False,
) -> dict[str, Any]:
    evidence = _snapshot_mapping(value, label=f"{checkpoint} approval evidence")
    _verify_self_digest(
        evidence, field="evidence_digest", label=f"{checkpoint} approval evidence"
    )
    if evidence.get("verification_checkpoint") != checkpoint:
        raise ContractError("approval evidence checkpoint differs from the protected boundary")
    comment = evidence.get("comment")
    if (
        not isinstance(comment, dict)
        or comment.get("issue_number") != issue_number
        or comment.get("comment_id") != comment_id
    ):
        raise ContractError("approval evidence Issue/comment identity differs")
    if strict_live_evidence:
        required = {
            "schema_version": 1,
            "gate_kind": GATE_KIND,
            "run_scope_digest": expected_run_scope_digest,
            "original_evidence_digest": expected_original_evidence_digest,
            "limitations": LIMITATIONS,
            "implementation_current_main_ancestry_verified": True,
            "authority": False,
            "local_offline_permission": True,
            "global_uniqueness_guaranteed": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        for key, expected in required.items():
            if evidence.get(key) != expected:
                raise ContractError(f"approval evidence {key} differs")
    return evidence


def _verify_persisted_mapping(path: Path, expected: Mapping[str, Any], *, label: str) -> None:
    if not path.is_file() or _is_reparse(path):
        raise ContractError(f"{label} is missing or unsafe")
    payload = path.read_bytes()
    if payload != _canonical_bytes(dict(expected)):
        raise ContractError(f"{label} changed after exclusive persistence")
    observed = _load_strict_json_bytes(payload, label=label)
    if observed != dict(expected):
        raise ContractError(f"{label} semantic snapshot differs")


def _verify_materialization_manifest(
    *,
    root: Path,
    registered: Any,
    manifest_binding: Mapping[str, Any] | None,
    verify_candidate_file: bool,
    verify_settlement_file: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy_projection = registered.policy["projection_contract"]
    if manifest_binding is None:
        manifest_path = _safe_bound_path(
            root,
            policy_projection["materialization_manifest_path"],
            label="materialization manifest",
            must_exist=True,
        )
        with manifest_path.open("rb") as handle:
            manifest_payload = handle.read()
    else:
        manifest_path, manifest_payload = _read_bound_bytes(
            root, manifest_binding, label="materialization manifest"
        )
        if manifest_binding.get("path") != policy_projection["materialization_manifest_path"]:
            raise ContractError("materialization manifest path differs from policy")
    manifest = _load_strict_json_bytes(
        manifest_payload, label="materialization manifest"
    )
    if not isinstance(manifest, dict):
        raise ContractError("materialization manifest must be an object")
    expected_keys = {
        "schema_version",
        "gate_kind",
        "manifest_kind",
        "authority",
        "implementation_binding",
        "source_bindings",
        "source_usecols",
        "candidate_cross_source_equality",
        "p_action_formula_attestation",
        "candidate_identity_attestation",
        "ordered_race_id_sha256",
        "projection_bindings",
        "projection_rows",
        "projection_columns",
        "candidate_projection_sealed_before_settlement_source_open",
        "raw_forbidden_semantic_values_selected",
        "decisions_computed",
        "metrics_computed",
        "roi_computed",
        "odds_price_popularity_or_market_values_persisted",
        "formal_buy",
        "send_order",
        "stake",
        "manifest_digest",
    }
    if set(manifest) != expected_keys:
        raise ContractError("materialization manifest has unknown or missing fields")
    _verify_self_digest(manifest, field="manifest_digest", label="materialization manifest")
    scalars = {
        "schema_version": 1,
        "gate_kind": GATE_KIND,
        "manifest_kind": MATERIALIZATION_MANIFEST_KIND,
        "authority": False,
        "candidate_projection_sealed_before_settlement_source_open": True,
        "raw_forbidden_semantic_values_selected": False,
        "decisions_computed": False,
        "metrics_computed": False,
        "roi_computed": False,
        "odds_price_popularity_or_market_values_persisted": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    for key, expected in scalars.items():
        if manifest.get(key) != expected:
            raise ContractError(f"materialization manifest safety field differs: {key}")
    implementation_binding = manifest.get("implementation_binding")
    if not isinstance(implementation_binding, dict) or set(implementation_binding) != {
        "implementation_commit", "runtime_material_bundle_sha256"
    }:
        raise ContractError("materialization implementation binding is invalid")
    if not FULL_GIT_SHA.fullmatch(str(implementation_binding["implementation_commit"])) or not FULL_SHA256.fullmatch(
        str(implementation_binding["runtime_material_bundle_sha256"])
    ):
        raise ContractError("materialization implementation binding digests are invalid")
    expected_usecols = {
        role: list(columns)
        for role, columns in sorted(
            policy_projection["raw_source_allowed_projection_columns"].items()
        )
    }
    if manifest.get("source_usecols") != expected_usecols:
        raise ContractError("materializer usecols attestation differs from policy")
    source_bindings = manifest.get("source_bindings")
    if not isinstance(source_bindings, dict) or set(source_bindings) != set(SOURCE_INPUTS):
        raise ContractError("materialization source bindings are incomplete")
    for role, specification in SOURCE_INPUTS.items():
        binding = source_bindings.get(role)
        if not isinstance(binding, dict) or set(binding) != {
            "path", "expected_sha256", "observed_sha256", "observed_byte_size"
        }:
            raise ContractError(f"materialization source binding is invalid: {role}")
        if (
            binding["path"] != specification["path"]
            or binding["expected_sha256"] != specification["expected_sha256"]
            or binding["observed_sha256"] != specification["expected_sha256"]
            or type(binding["observed_byte_size"]) is not int
            or binding["observed_byte_size"] <= 0
        ):
            raise ContractError(f"materialization source binding differs from policy: {role}")
    equality = manifest.get("candidate_cross_source_equality")
    if equality != {
        "columns": list(policy_projection["candidate_cross_source_equality_columns"]),
        "row_count": RACE_COUNT,
        "vector_sha256": equality.get("vector_sha256") if isinstance(equality, dict) else None,
        "exact": True,
    }:
        raise ContractError("candidate cross-source equality attestation is invalid")
    if not FULL_SHA256.fullmatch(str(equality["vector_sha256"])):
        raise ContractError("candidate cross-source vector digest is invalid")
    formula = manifest.get("p_action_formula_attestation")
    if formula != {
        "formula": policy_projection["p_action_lineage_formula"],
        "numeric_semantics": policy_projection["p_action_numeric_semantics"],
        "absolute_tolerance": policy_projection["p_action_absolute_tolerance"],
        "row_count": RACE_COUNT,
        "all_rows_verified": True,
    }:
        raise ContractError("p-action formula attestation is invalid")
    identity = manifest.get("candidate_identity_attestation")
    if identity != {
        "formula": policy_projection["candidate_key_formula"],
        "row_count": RACE_COUNT,
        "all_rows_verified": True,
    }:
        raise ContractError("candidate identity attestation is invalid")
    if not isinstance(manifest.get("ordered_race_id_sha256"), str) or not FULL_SHA256.fullmatch(
        manifest["ordered_race_id_sha256"]
    ):
        raise ContractError("ordered race-id digest attestation is invalid")
    if manifest.get("projection_rows") != {
        "candidate_projection": RACE_COUNT,
        "settlement_projection": RACE_COUNT,
    }:
        raise ContractError("materialized projection row attestation is invalid")
    if manifest.get("projection_columns") != {
        "candidate_projection": list(CANDIDATE_COLUMNS),
        "settlement_projection": list(SETTLEMENT_COLUMNS),
    }:
        raise ContractError("materialized projection column attestation is invalid")
    for columns in manifest["projection_columns"].values():
        if any(
            token in column.lower()
            for column in columns
            for token in FORBIDDEN_PROJECTION_TOKENS
        ):
            raise ContractError("materialized projection contains a forbidden field")
    projection_bindings = manifest.get("projection_bindings")
    if not isinstance(projection_bindings, dict) or set(projection_bindings) != set(PROJECTION_INPUTS):
        raise ContractError("materialized projection bindings are incomplete")
    for role, expected in PROJECTION_INPUTS.items():
        binding = projection_bindings.get(role)
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256", "byte_size"}:
            raise ContractError(f"projection binding is invalid: {role}")
        if binding["path"] != expected["path"]:
            raise ContractError(f"projection path differs from policy: {role}")
        if not FULL_SHA256.fullmatch(str(binding["sha256"])) or type(binding["byte_size"]) is not int or binding["byte_size"] <= 0:
            raise ContractError(f"projection hash binding is invalid: {role}")
    if projection_bindings["candidate_projection"]["sha256"] == projection_bindings["settlement_projection"]["sha256"]:
        raise ContractError("candidate and settlement projection objects are not distinct")
    if verify_candidate_file:
        _verify_file_binding(
            root,
            projection_bindings["candidate_projection"],
            label="candidate projection",
        )
    if verify_settlement_file:
        _verify_file_binding(
            root,
            projection_bindings["settlement_projection"],
            label="settlement projection",
        )
    return manifest, dict(source_bindings), dict(projection_bindings)


def _runtime_environment_manifest() -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ContractError("NumPy 2.4.3 is required") from exc
    executable = Path(sys.executable).resolve(strict=True)
    executable_sha, executable_size = _raw_sha256(executable)
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_minor_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_executable_sha256": executable_sha,
        "python_executable_byte_size": executable_size,
        "numpy_version": str(np.__version__),
        "runner_template_id": RUNNER_TEMPLATE_ID,
        "application_environment_view_during_workload": "EMPTY",
        "network_isolation": LIMITATIONS["network_isolation"],
    }


def _verify_runtime_environment(expected_digest: str | None = None) -> tuple[dict[str, Any], str]:
    manifest = _runtime_environment_manifest()
    if manifest["python_implementation"] != "CPython" or manifest["python_minor_version"] not in {"3.11", "3.12"}:
        raise ContractError("offline run requires CPython 3.11 or 3.12")
    if manifest["numpy_version"] != "2.4.3":
        raise ContractError("offline run requires NumPy 2.4.3")
    digest = canonical_digest(manifest)
    if expected_digest is not None and digest != expected_digest:
        raise ContractError("runtime environment differs from the frozen scope")
    return manifest, digest


def _utc_now(now: str | Callable[[], str] | None) -> str:
    if isinstance(now, str):
        value = now
    elif callable(now):
        value = now()
    elif now is None:
        value = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    else:
        raise ContractError("now must be an ISO-8601 UTC string, callable, or None")
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContractError("scope sealing time must be ISO-8601 UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError("scope sealing time is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError("scope sealing time must be UTC")
    return value


def _verify_deterministic_materialization_against_raw(
    *,
    repo_root: Path,
    source_root: Path,
    registered: Any,
    availability: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reproduce exact projection bytes at scope seal; no decision or metric is computed."""

    implementation_binding = {
        "implementation_commit": availability["implementation_commit"],
        "runtime_material_bundle_sha256": canonical_digest(
            dict(registered.runtime_material_digests)
        ),
    }
    with tempfile.TemporaryDirectory(prefix="offline-materialization-recheck-") as raw_temp:
        reproduction_root = Path(raw_temp)
        reproduced_evidence = _materialize_core(
            repo_root=reproduction_root,
            source_root=source_root,
            policy=registered.policy,
            implementation_binding=implementation_binding,
            expected_source_inputs=SOURCE_INPUTS,
            expected_race_count=RACE_COUNT,
            expected_fold_counts=FOLD_COUNTS,
        )
        reproduced_manifest, source_bindings, projection_bindings = (
            _verify_materialization_manifest(
                root=reproduction_root,
                registered=registered,
                manifest_binding=reproduced_evidence["materialization_manifest"],
                verify_candidate_file=True,
                verify_settlement_file=True,
            )
        )
        stored_payloads: dict[str, bytes] = {}
        for role, relative, label in (
            (
                "candidate_projection",
                PROJECTION_INPUTS["candidate_projection"]["path"],
                "candidate projection",
            ),
            (
                "settlement_projection",
                PROJECTION_INPUTS["settlement_projection"]["path"],
                "settlement projection",
            ),
            (
                "materialization_manifest",
                MATERIALIZATION_MANIFEST_PATH,
                "materialization manifest",
            ),
        ):
            stored = _safe_bound_path(
                repo_root, relative, label=f"stored {label}", must_exist=True
            ).read_bytes()
            reproduced = _safe_bound_path(
                reproduction_root,
                relative,
                label=f"reproduced {label}",
                must_exist=True,
            ).read_bytes()
            if stored != reproduced:
                raise ContractError(
                    f"stored {label} is not the exact deterministic projection of fixed raw bytes"
                )
            stored_payloads[role] = stored
        for role in ("candidate_projection", "settlement_projection"):
            payload = stored_payloads[role]
            if projection_bindings[role] != {
                "path": PROJECTION_INPUTS[role]["path"],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload),
            }:
                raise ContractError(
                    f"stored {role} bytes differ from their reproduced binding"
                )
        manifest_payload = stored_payloads["materialization_manifest"]
        manifest_binding = {
            "path": MATERIALIZATION_MANIFEST_PATH,
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "byte_size": len(manifest_payload),
        }
        if manifest_binding != reproduced_evidence["materialization_manifest"]:
            raise ContractError("stored manifest bytes differ from their reproduced binding")
        if reproduced_manifest.get("projection_bindings") != projection_bindings:
            raise ContractError("reproduced projection bindings are internally inconsistent")
        return manifest_binding, source_bindings, projection_bindings


def _compile_fixed_run_scope_artifact_worker(
    *,
    root: Path,
    source_root: Path,
    provider: Any,
    now: str | Callable[[], str] | None = None,
) -> tuple[dict[str, Any], Path]:
    repo_root = _safe_root(Path(root), label="repository root")
    raw_root = _safe_root(Path(source_root), label="source root")
    availability = _approval_module().verify_offline_gate_availability(
        root=repo_root,
        provider=provider,
        now=now,
    )
    verified_main = availability.get("verified_current_main_sha")
    if not isinstance(verified_main, str):
        raise ContractError("gate availability lacks verified current main")
    _verify_clean_git_head(repo_root, verified_main)
    registered = resolve_offline_registered_recipe(repo_root)
    _environment, environment_digest = _verify_runtime_environment()
    manifest_binding, source_bindings, projection_bindings = (
        _verify_deterministic_materialization_against_raw(
        repo_root=repo_root,
        source_root=raw_root,
        registered=registered,
        availability=availability,
        )
    )
    trust = availability.get("github_trust")
    if not isinstance(trust, dict):
        raise ContractError("gate availability lacks GitHub trust evidence")
    bindings = {
        "repository": DEFAULT_REPOSITORY,
        "base_branch": DEFAULT_BASE_BRANCH,
        "run_scope_base_commit": availability["implementation_commit"],
        "verified_current_main_sha": verified_main,
        "approvers_blob_sha": trust["approvers_blob_sha"],
        "approvers_content_sha256": trust["approvers_content_sha256"],
        "runtime_material_sha256": dict(registered.runtime_material_digests),
        "source_bindings": source_bindings,
        "projection_bindings": projection_bindings,
        "materialization_manifest": manifest_binding,
        "python_minor_version": _environment["python_minor_version"],
        "numpy_version": _environment["numpy_version"],
        "environment_manifest_sha256": environment_digest,
        "output_root": FIXED_OUTPUT_ROOT,
        "sealed_at": _utc_now(now),
    }
    scope = compile_offline_run_scope(registered, bindings)
    relative = offline_run_scope_artifact_path(scope["run_scope_digest"])
    artifact_path = _safe_bound_path(
        repo_root, relative, label="run scope artifact", must_exist=False
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(artifact_path, scope)
    return scope, artifact_path


def compile_fixed_run_scope_artifact(
    *, root: Path, source_root: Path
) -> tuple[dict[str, Any], Path]:
    """Canonical scope compiler with non-injectable provider and seal time."""

    return _compile_fixed_run_scope_artifact_worker(
        root=root,
        source_root=source_root,
        provider=_github_provider(),
        now=None,
    )


def _load_jsonl_exact_bytes(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("projection JSONL is not UTF-8") from exc
    with io.StringIO(text, newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or line.endswith("\r\n"):
                raise ContractError("projection JSONL must use UTF-8 LF records")
            if not line.strip():
                raise ContractError("projection JSONL cannot contain blank records")
            try:
                value = json.loads(
                    line,
                    parse_constant=reject_constant,
                    object_pairs_hook=strict_object,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ContractError(f"projection JSONL is invalid at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ContractError("projection JSONL records must be objects")
            if _canonical_bytes(value).decode("utf-8") != line:
                raise ContractError("projection JSONL record is not canonical")
            rows.append(value)
    return rows


def _ensure_projection_has_no_market_fields(columns: Sequence[str]) -> None:
    for column in columns:
        lowered = column.lower()
        if any(token in lowered for token in FORBIDDEN_PROJECTION_TOKENS):
            raise ContractError("projection contains a forbidden market/price field")


def _validate_candidate_projection(
    recipe: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    del recipe
    if len(rows) != RACE_COUNT:
        raise ContractError("candidate projection must contain exactly 3,746 rows")
    output: list[dict[str, Any]] = []
    seen_races: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != set(CANDIDATE_COLUMNS):
            raise ContractError("candidate projection schema is not exact")
        _ensure_projection_has_no_market_fields(tuple(raw))
        row = dict(raw)
        race_id = row.get("race_id")
        if not isinstance(race_id, str) or len(race_id) != 16 or not race_id.isdigit():
            raise ContractError("candidate race_id must be a 16-digit string")
        if race_id in seen_races:
            raise ContractError("candidate projection must have one row per race")
        seen_races.add(race_id)
        if type(row.get("horse_a")) is not int or type(row.get("horse_b")) is not int:
            raise ContractError("candidate horse numbers must be integers")
        expected_key = _candidate_key(row["horse_a"], row["horse_b"])
        if row.get("candidate_key") != expected_key:
            raise ContractError("candidate_key differs from its unordered horse pair")
        key = (race_id, expected_key)
        if key in seen_keys:
            raise ContractError("candidate projection has a duplicate key")
        seen_keys.add(key)
        if type(row.get("candidate_generated")) is not bool or type(row.get("eligible_race")) is not bool:
            raise ContractError("candidate flags must be booleans")
        if row.get("fold") not in FOLD_VALUES:
            raise ContractError("candidate fold is outside folds 2-4")
        _fixed_race_date(row.get("race_date"), label="candidate race_date")
        if not isinstance(row.get("venue_code"), str) or not row["venue_code"]:
            raise ContractError("candidate venue_code must be a non-empty string")
        if not isinstance(row.get("top1_wide_prob"), str) or not isinstance(row.get("p_action_C0_offset"), str):
            raise ContractError("candidate p and a must preserve source decimal strings")
        _verify_p_action_formula(row["top1_wide_prob"], row["p_action_C0_offset"])
        output.append(row)
    output.sort(key=lambda row: row["race_id"])
    if _fold_counter(output) != Counter(FOLD_COUNTS):
        raise ContractError("candidate fold counts differ from the fixed cohort")
    dates = [str(row["race_date"]) for row in output]
    if min(dates) != DATE_MIN or max(dates) != DATE_MAX:
        raise ContractError("candidate date range differs from the fixed cohort")
    return output


def _freeze_decisions(
    *,
    recipe: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validated = _validate_candidate_projection(recipe, candidate_rows)
    masks = evaluate_registered_decisions(recipe, validated)
    if len(masks) != RACE_COUNT:
        raise ContractError("registered decision evaluator did not return the fixed cohort")
    mask_by_key = {(row["race_id"], row["candidate_key"]): row for row in masks}
    decision_rows: list[dict[str, Any]] = []
    for row in validated:
        mask = mask_by_key[(row["race_id"], row["candidate_key"])]
        decision_rows.append(
            {
                "race_id": row["race_id"],
                "race_date": row["race_date"],
                "venue_code": row["venue_code"],
                "fold": row["fold"],
                "candidate_key": row["candidate_key"],
                "horse_a": row["horse_a"],
                "horse_b": row["horse_b"],
                "top1_wide_prob": float(row["top1_wide_prob"]),
                "p_action_C0_offset": float(row["p_action_C0_offset"]),
                "d0_eligible": mask["d0_eligible"],
                "d1_eligible": mask["d1_eligible"],
            }
        )
    vector = [
        {
            "race_id": row["race_id"],
            "candidate_key": row["candidate_key"],
            "d0_eligible": row["d0_eligible"],
            "d1_eligible": row["d1_eligible"],
        }
        for row in decision_rows
    ]
    projection = {
        "candidate_projection_digest": canonical_digest(validated),
        "decision_rows_digest": canonical_digest(decision_rows),
        "decision_vector_digest": canonical_digest(vector),
    }
    return decision_rows, projection


def _settled_return(row: Mapping[str, Any]) -> float:
    if type(row.get("candidate_hit")) is not bool:
        raise ContractError("candidate_hit must be boolean")
    payoff = row.get("official_wide_pay")
    if row["candidate_hit"]:
        if not isinstance(payoff, str):
            raise ContractError("hit=true requires a source-decimal official payoff")
        try:
            value = float(payoff)
        except ValueError as exc:
            raise ContractError("official payoff must be numeric") from exc
        if not math.isfinite(value) or value <= 0:
            raise ContractError("hit=true requires a finite positive payoff")
        return value
    if payoff is not None:
        raise ContractError("hit=false requires null official payoff")
    return 0.0


def _validate_settlement_projection(
    rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    if len(rows) != RACE_COUNT:
        raise ContractError("settlement projection must contain exactly 3,746 rows")
    settlements: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != set(SETTLEMENT_COLUMNS):
            raise ContractError("settlement projection schema is not exact")
        _ensure_projection_has_no_market_fields(tuple(raw))
        row = dict(raw)
        race_id = row.get("race_id")
        candidate_key = row.get("candidate_key")
        if not isinstance(race_id, str) or len(race_id) != 16 or not race_id.isdigit() or not isinstance(candidate_key, str):
            raise ContractError("settlement join keys are invalid")
        if row.get("official_outcome_completeness") is not True:
            raise ContractError("every settlement row must be officially complete")
        key = (race_id, candidate_key)
        if key in settlements:
            raise ContractError("settlement projection has a duplicate key")
        row["settled_candidate_return_yen"] = _settled_return(row)
        settlements[key] = row
    decision_keys = {(row["race_id"], row["candidate_key"]) for row in decision_rows}
    if decision_keys != set(settlements):
        raise ContractError("candidate and settlement projection keys differ")
    return settlements


def _apply_returns(
    decision_rows: Sequence[Mapping[str, Any]],
    settlements: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    notional_yen: float,
    return_cap_yen: float | None = None,
    zero_return_race_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    zero_ids = zero_return_race_ids or set()
    paired: list[dict[str, Any]] = []
    for decision in decision_rows:
        settlement = settlements[(decision["race_id"], decision["candidate_key"])]
        base_return = float(settlement["settled_candidate_return_yen"])
        if return_cap_yen is not None:
            base_return = min(base_return, return_cap_yen)
        if decision["race_id"] in zero_ids:
            base_return = 0.0
        row = dict(decision)
        row["candidate_hit"] = settlement["candidate_hit"]
        row["settled_candidate_return_yen"] = float(settlement["settled_candidate_return_yen"])
        for arm in ("d0", "d1"):
            eligible = bool(row[f"{arm}_eligible"])
            row[f"{arm}_stake_yen"] = notional_yen if eligible else 0.0
            row[f"{arm}_return_yen"] = base_return if eligible else 0.0
            row[f"{arm}_profit_yen"] = row[f"{arm}_return_yen"] - row[f"{arm}_stake_yen"]
        row["delta_profit_yen"] = row["d1_profit_yen"] - row["d0_profit_yen"]
        paired.append(row)
    return paired


def _arm_summary(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    bets = sum(bool(row[f"{prefix}_eligible"]) for row in rows)
    hits = sum(bool(row[f"{prefix}_eligible"] and row["candidate_hit"]) for row in rows)
    stake = sum(float(row[f"{prefix}_stake_yen"]) for row in rows)
    returns = sum(float(row[f"{prefix}_return_yen"]) for row in rows)
    return {
        "arm": prefix.upper(),
        "bet_count": bets,
        "hit_count": hits,
        "stake_denominator_yen": stake,
        "return_yen": returns,
        "profit_yen": returns - stake,
        "roi_percent": None if stake == 0 else returns / stake * 100.0,
    }


def _metric_projection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    delta_sum = sum(float(row["delta_profit_yen"]) for row in rows)
    return {
        "enrolled_race_count": len(rows),
        "d0": _arm_summary(rows, "d0"),
        "d1": _arm_summary(rows, "d1"),
        "sum_delta_profit_yen": delta_sum,
        "mean_delta_profit_yen_per_enrolled_race": delta_sum / len(rows),
        "decision_disagreement_count": sum(
            bool(row["d0_eligible"] != row["d1_eligible"]) for row in rows
        ),
    }


def _common_high_payout_set(
    settlements: Mapping[tuple[str, str], Mapping[str, Any]], count: int
) -> list[str]:
    ranked = sorted(
        settlements.values(),
        key=lambda row: (-float(row["settled_candidate_return_yen"]), row["race_id"]),
    )
    return [str(row["race_id"]) for row in ranked[:count]]


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    import numpy as np

    clusters: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        clusters[(str(row["race_date"]), str(row["venue_code"]))].append(
            float(row["delta_profit_yen"])
        )
    keys = sorted(clusters)
    if not keys:
        raise ContractError("bootstrap requires at least one cluster")
    arrays = [np.asarray(clusters[key], dtype=np.float64) for key in keys]
    rng = np.random.Generator(np.random.PCG64(seed))
    statistics = np.empty(replicates, dtype=np.float64)
    n_clusters = len(keys)
    for index in range(replicates):
        draw = rng.integers(0, n_clusters, size=n_clusters, dtype=np.int64, endpoint=False)
        total = 0.0
        count = 0
        for selected in draw:
            values = arrays[int(selected)]
            total += float(values.sum())
            count += int(values.size)
        statistics[index] = total / count
    return {
        "cluster_count": n_clusters,
        "replicates": replicates,
        "seed": seed,
        "rng": "numpy.random.Generator(PCG64)",
        "mean": float(statistics.mean()),
        "one_sided_95_lower_bound": float(np.quantile(statistics, 0.05, method="linear")),
        "distribution_digest": canonical_digest(statistics.tolist()),
    }


def _scientific_projection(
    *,
    recipe: Mapping[str, Any],
    decision_rows: Sequence[Mapping[str, Any]],
    decision_projection: Mapping[str, Any],
    settlement_rows: Sequence[Mapping[str, Any]],
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], str]:
    settlements = _validate_settlement_projection(settlement_rows, decision_rows)
    notional = float(recipe["metric"]["offline_evaluation_notional_yen"])
    paired = _apply_returns(decision_rows, settlements, notional_yen=notional)
    primary = _metric_projection(paired)
    top1 = set(_common_high_payout_set(settlements, 1))
    top3 = set(_common_high_payout_set(settlements, 3))
    sensitivity = {
        "common_top1_return_zeroed": _metric_projection(
            _apply_returns(decision_rows, settlements, notional_yen=notional, zero_return_race_ids=top1)
        ),
        "common_top3_return_zeroed": _metric_projection(
            _apply_returns(decision_rows, settlements, notional_yen=notional, zero_return_race_ids=top3)
        ),
        "common_2000_yen_winsor": _metric_projection(
            _apply_returns(
                decision_rows,
                settlements,
                notional_yen=notional,
                return_cap_yen=float(recipe["sensitivity"]["winsor_cap_yen"]),
            )
        ),
        "top1_race_ids": sorted(top1),
        "top3_race_ids": sorted(top3),
    }
    projection = {
        "primary": primary,
        "sensitivity": sensitivity,
        "bootstrap": _cluster_bootstrap(
            paired,
            replicates=bootstrap_replicates,
            seed=int(recipe["bootstrap"]["seed"]),
        ),
        "candidate_projection_digest": decision_projection["candidate_projection_digest"],
        "decision_vector_digest": decision_projection["decision_vector_digest"],
        "settlement_projection_digest": canonical_digest(
            sorted(settlements.values(), key=lambda row: row["race_id"])
        ),
        "paired_rows_digest": canonical_digest(paired),
        "contract_status": "VALID",
    }
    outcome = "NO_DECISION_EFFECT" if primary["decision_disagreement_count"] == 0 else "DIRECTIONAL_EFFECT"
    return projection, outcome


@contextmanager
def _application_workload_firewall() -> Iterator[None]:
    """Best-effort deny guard; deliberately not represented as an OS sandbox."""

    import urllib.request

    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise OfflineFirewallViolation("forbidden workload capability requested")

    saved = {
        "socket": socket.socket,
        "create_connection": socket.create_connection,
        "getaddrinfo": socket.getaddrinfo,
        "popen": subprocess.Popen,
        "run": subprocess.run,
        "call": subprocess.call,
        "check_call": subprocess.check_call,
        "check_output": subprocess.check_output,
        "urlopen": urllib.request.urlopen,
        "system": os.system,
        "os_popen": os.popen,
        "environ": os.environ,
    }
    socket.socket = denied  # type: ignore[assignment]
    socket.create_connection = denied  # type: ignore[assignment]
    socket.getaddrinfo = denied  # type: ignore[assignment]
    subprocess.Popen = denied  # type: ignore[assignment]
    subprocess.run = denied  # type: ignore[assignment]
    subprocess.call = denied  # type: ignore[assignment]
    subprocess.check_call = denied  # type: ignore[assignment]
    subprocess.check_output = denied  # type: ignore[assignment]
    urllib.request.urlopen = denied  # type: ignore[assignment]
    os.system = denied  # type: ignore[assignment]
    os.popen = denied  # type: ignore[assignment]
    os.environ = {}  # type: ignore[assignment]
    try:
        yield
    finally:
        os.environ = saved["environ"]  # type: ignore[assignment]
        os.popen = saved["os_popen"]  # type: ignore[assignment]
        os.system = saved["system"]  # type: ignore[assignment]
        urllib.request.urlopen = saved["urlopen"]  # type: ignore[assignment]
        subprocess.check_output = saved["check_output"]  # type: ignore[assignment]
        subprocess.check_call = saved["check_call"]  # type: ignore[assignment]
        subprocess.call = saved["call"]  # type: ignore[assignment]
        subprocess.run = saved["run"]  # type: ignore[assignment]
        subprocess.Popen = saved["popen"]  # type: ignore[assignment]
        socket.getaddrinfo = saved["getaddrinfo"]  # type: ignore[assignment]
        socket.create_connection = saved["create_connection"]  # type: ignore[assignment]
        socket.socket = saved["socket"]  # type: ignore[assignment]


def _start_receipt(
    *, run_scope: Mapping[str, Any], approval_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": "OFFLINE_EXCLUSIVE_START",
        "gate_kind": GATE_KIND,
        "lifecycle_state": "RNOD_RUNNING",
        "run_scope_digest": run_scope["run_scope_digest"],
        "semantic_subject_digest": run_scope["semantic_subject_digest"],
        "exact_subject_digest": run_scope["exact_subject_digest"],
        "approval_evidence_digest": approval_evidence.get("evidence_digest")
        or canonical_digest(dict(approval_evidence)),
        "authority": False,
        **LIMITATIONS,
        "retry_count": 0,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    receipt["receipt_digest"] = canonical_digest(receipt)
    return receipt


def _freeze_receipt(
    *,
    run_scope: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    per_replica: Mapping[str, Mapping[str, Any]],
    candidate_file_sha256: str,
) -> dict[str, Any]:
    projections = {
        replica: {
            "candidate_projection_digest": per_replica[replica]["candidate_projection_digest"],
            "decision_rows_digest": per_replica[replica]["decision_rows_digest"],
            "decision_vector_digest": per_replica[replica]["decision_vector_digest"],
        }
        for replica in REPLICA_IDS
    }
    if projections["clean_a"] != projections["clean_b"]:
        raise ContractError("logical replica decision projections differ")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": "OFFLINE_DECISION_FREEZE",
        "gate_kind": GATE_KIND,
        "lifecycle_state": "RNOD_RUNNING",
        "run_scope_digest": run_scope["run_scope_digest"],
        "start_receipt_digest": start_receipt["receipt_digest"],
        "candidate_file_sha256": candidate_file_sha256,
        "replica_mode": REPLICA_MODE,
        "replica_attempt_count": 1,
        "replica_decision_projections": projections,
        "replica_semantic_equality": True,
        "settlement_accessed_at_receipt": False,
        "authority": False,
        **LIMITATIONS,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    receipt["receipt_digest"] = canonical_digest(receipt)
    return receipt


def _write_invalid_after_start(output_root: Path) -> None:
    try:
        invalid_path = output_root / "INVALID.json"
        payload = _canonical_bytes(
            {
                "status": "INVALID",
                "reason_code": "INVALID_AFTER_START_NO_RETRY",
            }
        )
        if invalid_path.exists() or invalid_path.is_symlink():
            if _is_reparse(invalid_path):
                return
            if invalid_path.is_file() and invalid_path.read_bytes() == payload:
                return
        temporary = output_root / ".INVALID.json.tmp"
        if temporary.exists() or temporary.is_symlink():
            if _is_reparse(temporary) or not temporary.is_file():
                return
            temporary.unlink()
        _write_bytes_exclusive(temporary, payload)
        os.replace(temporary, invalid_path)
        _fsync_parent(invalid_path)
    except BaseException:
        return


def _read_canonical_mapping_file(path: Path, *, label: str) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        if _is_reparse(path):
            raise ContractError(f"{label} may not be a symlink/junction/reparse point")
    if not path.is_file():
        raise ContractError(f"{label} is missing")
    payload = path.read_bytes()
    value = _load_strict_json_bytes(payload, label=label)
    if not isinstance(value, dict) or payload != _canonical_bytes(value):
        raise ContractError(f"{label} is not a canonical object")
    return value


def _recover_or_reject_existing_output(
    output_root: Path, run_scope: Mapping[str, Any]
) -> None:
    """Classify a prior local tombstone without opening projections or raw sources."""

    if not output_root.exists():
        return
    if not output_root.is_dir() or _is_reparse(output_root):
        raise OfflineInvalidAfterStart("unsafe existing offline output tombstone")
    invalid_path = output_root / "INVALID.json"
    result_path = output_root / "result.json"
    seal_path = output_root / "result_seal_receipt.json"
    try:
        if (invalid_path.exists() or invalid_path.is_symlink()) and not result_path.exists():
            if _is_reparse(invalid_path) or invalid_path.read_bytes() != _canonical_bytes(
                {
                    "status": "INVALID",
                    "reason_code": "INVALID_AFTER_START_NO_RETRY",
                }
            ):
                raise ContractError("existing INVALID tombstone is not canonical")
            raise OfflineInvalidAfterStart("offline run was already consumed as INVALID")
        if (
            (result_path.exists() or result_path.is_symlink())
            and (seal_path.exists() or seal_path.is_symlink())
            and not (invalid_path.exists() or invalid_path.is_symlink())
        ):
            initial_evidence = _read_canonical_mapping_file(
                output_root / "approval_evidence_initial.json",
                label="existing initial approval evidence",
            )
            candidate_evidence = _read_canonical_mapping_file(
                output_root / "approval_evidence_before_candidate.json",
                label="existing candidate approval evidence",
            )
            start = _read_canonical_mapping_file(
                output_root / "start_receipt.json", label="existing start receipt"
            )
            freeze = _read_canonical_mapping_file(
                output_root / "decision_freeze_receipt.json",
                label="existing decision-freeze receipt",
            )
            final_evidence = _read_canonical_mapping_file(
                output_root / "approval_evidence_before_result.json",
                label="existing result approval evidence",
            )
            seal = _read_canonical_mapping_file(
                seal_path, label="existing result seal"
            )
            result = _read_canonical_mapping_file(
                result_path, label="existing completed result"
            )
            _verify_self_digest(
                initial_evidence,
                field="evidence_digest",
                label="existing initial approval evidence",
            )
            _verify_self_digest(
                candidate_evidence,
                field="evidence_digest",
                label="existing candidate approval evidence",
            )
            _verify_self_digest(
                final_evidence,
                field="evidence_digest",
                label="existing result approval evidence",
            )
            _verify_self_digest(start, field="receipt_digest", label="existing start")
            _verify_self_digest(freeze, field="receipt_digest", label="existing freeze")
            _verify_self_digest(result, field="result_digest", label="existing result")
            _verify_self_digest(seal, field="receipt_digest", label="existing result seal")
            scientific = result.get("scientific_projection")
            replica_digests = result.get("replica_scientific_projection_digests")
            if (
                candidate_evidence.get("verification_checkpoint")
                != "BEFORE_CANDIDATE_OPEN"
                or final_evidence.get("verification_checkpoint")
                != "BEFORE_RESULT_PUBLISH"
                or initial_evidence.get("verification_checkpoint")
                != "INITIAL_APPROVAL"
                or start.get("receipt_kind") != "OFFLINE_EXCLUSIVE_START"
                or start.get("lifecycle_state") != "RNOD_RUNNING"
                or freeze.get("receipt_kind") != "OFFLINE_DECISION_FREEZE"
                or freeze.get("lifecycle_state") != "RNOD_RUNNING"
                or seal.get("receipt_kind") != "OFFLINE_RESULT_SEAL"
                or result.get("lifecycle_state") != "RNOD_COMPLETED"
                or seal.get("lifecycle_state") != "RNOD_RESULT_SEALED"
                or start.get("run_scope_digest") != run_scope["run_scope_digest"]
                or freeze.get("run_scope_digest") != run_scope["run_scope_digest"]
                or result.get("run_scope_digest") != run_scope["run_scope_digest"]
                or seal.get("run_scope_digest") != run_scope["run_scope_digest"]
                or result.get("semantic_subject_digest")
                != run_scope["semantic_subject_digest"]
                or result.get("exact_subject_digest") != run_scope["exact_subject_digest"]
                or start.get("approval_evidence_digest")
                != candidate_evidence["evidence_digest"]
                or freeze.get("start_receipt_digest") != start["receipt_digest"]
                or result.get("decision_freeze_receipt_digest")
                != freeze["receipt_digest"]
                or seal.get("decision_freeze_receipt_digest")
                != freeze["receipt_digest"]
                or result.get("approval_evidence_digest")
                != final_evidence["evidence_digest"]
                or seal.get("approval_evidence_digest")
                != final_evidence["evidence_digest"]
                or seal.get("completed_result_digest") != result["result_digest"]
                or not isinstance(scientific, dict)
                or result.get("scientific_projection_digest")
                != canonical_digest(scientific)
                or seal.get("scientific_projection_digest")
                != result.get("scientific_projection_digest")
                or not isinstance(replica_digests, dict)
                or set(replica_digests) != set(REPLICA_IDS)
                or len(set(replica_digests.values())) != 1
                or next(iter(replica_digests.values()))
                != result.get("scientific_projection_digest")
                or seal.get("replica_scientific_projection_digests")
                != replica_digests
            ):
                raise ContractError("existing completion artifacts do not bind this scope")
            for evidence in (initial_evidence, candidate_evidence, final_evidence):
                if evidence.get("run_scope_digest") != run_scope["run_scope_digest"]:
                    raise ContractError("existing approval evidence binds another scope")
            if initial_evidence.get("original_evidence_digest") is not None:
                raise ContractError("existing initial approval evidence chain differs")
            for evidence in (candidate_evidence, final_evidence):
                if evidence.get("original_evidence_digest") != initial_evidence["evidence_digest"]:
                    raise ContractError("existing approval evidence initial chain differs")
            comments = [
                evidence.get("comment")
                for evidence in (initial_evidence, candidate_evidence, final_evidence)
            ]
            if any(not isinstance(comment, dict) for comment in comments) or not (
                comments[0] == comments[1] == comments[2]
            ):
                raise ContractError("existing approval evidence comment chain differs")
            for key, expected in {
                "gate_kind": GATE_KIND,
                "recipe_id": RECIPE_ID,
                "recipe_version": RECIPE_VERSION,
                "authority": False,
                **LIMITATIONS,
                **NONPROMOTION,
            }.items():
                if result.get(key) != expected:
                    raise ContractError("existing result safety envelope differs")
            bootstrap = scientific.get("bootstrap")
            if (
                scientific.get("contract_status") != "VALID"
                or not isinstance(bootstrap, dict)
                or bootstrap.get("replicates") != BOOTSTRAP_REPLICATES
                or bootstrap.get("seed") != BOOTSTRAP_SEED
            ):
                raise ContractError("existing scientific contract differs")
            raise OfflineRunAlreadyCompleted("offline run already completed; rerun forbidden")
        _write_invalid_after_start(output_root)
        raise OfflineInvalidAfterStart("incomplete prior start was recovered as INVALID")
    except (OfflineInvalidAfterStart, OfflineRunAlreadyCompleted):
        raise
    except BaseException:
        _write_invalid_after_start(output_root)
        raise OfflineInvalidAfterStart("existing output failed closed as INVALID") from None


def _raise_poststart_terminal_after_exception(
    output_root: Path, run_scope: Mapping[str, Any]
) -> NoReturn:
    """Resolve the terminal state after any exception past exclusive start.

    The recovery routine recognizes a canonical result/seal chain as the
    irreversible COMPLETED state.  All other post-start states become the one
    minimal INVALID tombstone.  This keeps asynchronous exceptions at the
    atomic result-replace boundary from creating both terminal states.
    """

    try:
        _recover_or_reject_existing_output(output_root, run_scope)
    except OfflineRunAlreadyCompleted:
        raise
    except OfflineInvalidAfterStart:
        raise OfflineInvalidAfterStart(
            "offline run is INVALID_AFTER_START_NO_RETRY"
        ) from None
    raise AssertionError("output recovery must terminate the invocation")


def _result_seal_receipt(
    *,
    run_scope: Mapping[str, Any],
    decision_freeze_receipt: Mapping[str, Any],
    approval_evidence_digest: str,
    completed_result: Mapping[str, Any],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": "OFFLINE_RESULT_SEAL",
        "gate_kind": GATE_KIND,
        "lifecycle_state": "RNOD_RESULT_SEALED",
        "run_scope_digest": run_scope["run_scope_digest"],
        "decision_freeze_receipt_digest": decision_freeze_receipt["receipt_digest"],
        "approval_evidence_digest": approval_evidence_digest,
        "completed_result_digest": completed_result["result_digest"],
        "scientific_projection_digest": completed_result[
            "scientific_projection_digest"
        ],
        "replica_scientific_projection_digests": dict(
            completed_result["replica_scientific_projection_digests"]
        ),
        "replica_semantic_equality": True,
        "authority": False,
        **LIMITATIONS,
        "retry_count": 0,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    receipt["receipt_digest"] = canonical_digest(receipt)
    return receipt


def _execute_offline_registered_diagnostic_worker(
    *,
    root: Path,
    source_root: Path,
    run_scope: Mapping[str, Any],
    issue_number: int,
    comment_id: int,
    provider: Any | None,
    now: str | Callable[[], str] | None = None,
    _require_live_token: bool = False,
) -> dict[str, Any]:
    """Serialize recovery/execution, then run the private injectable worker."""

    frozen_scope = _snapshot_mapping(run_scope, label="canonical run scope")
    repo_root = _safe_root(Path(root), label="repository root")
    registered = resolve_offline_registered_recipe(repo_root)
    verify_canonical_offline_run_scope(registered, frozen_scope)
    output_root = _safe_bound_path(
        repo_root,
        frozen_scope["output_root"],
        label="fixed output root",
        must_exist=False,
    )
    with _exclusive_run_lock(output_root):
        _cleanup_orphan_result_temps(output_root)
        return _execute_offline_registered_diagnostic_worker_under_lock(
            root=repo_root,
            source_root=source_root,
            run_scope=frozen_scope,
            issue_number=issue_number,
            comment_id=comment_id,
            provider=provider,
            now=now,
            _require_live_token=_require_live_token,
        )


def _execute_offline_registered_diagnostic_worker_under_lock(
    *,
    root: Path,
    source_root: Path,
    run_scope: Mapping[str, Any],
    issue_number: int,
    comment_id: int,
    provider: Any | None,
    now: str | Callable[[], str] | None = None,
    _require_live_token: bool = False,
) -> dict[str, Any]:
    """Run preaccess raw provenance verification, then a projection-only workload."""

    bootstrap_replicates = BOOTSTRAP_REPLICATES
    run_scope = _snapshot_mapping(run_scope, label="canonical run scope")
    repo_root = _safe_root(Path(root), label="repository root")
    registered = resolve_offline_registered_recipe(repo_root)
    verify_canonical_offline_run_scope(registered, run_scope)
    output_root = _safe_bound_path(
        repo_root, run_scope["output_root"], label="fixed output root", must_exist=False
    )
    _recover_or_reject_existing_output(output_root, run_scope)
    if _require_live_token and not any(
        isinstance(os.environ.get(name), str) and bool(os.environ.get(name, "").strip())
        for name in ("GH_TOKEN", "GITHUB_TOKEN")
    ):
        raise ContractError(
            "canonical offline run requires GH_TOKEN or GITHUB_TOKEN for fail-closed remote revalidation"
        )
    if provider is None:
        provider = _github_provider()
    raw_root = _safe_root(Path(source_root), label="source root")
    expected_head = run_scope.get("verified_current_main_sha")
    if not isinstance(expected_head, str):
        raise ContractError("run scope lacks verified current main")
    _verify_clean_git_head(repo_root, expected_head)
    runtime_bindings = run_scope.get("runtime_bindings")
    if not isinstance(runtime_bindings, Mapping):
        raise ContractError("run scope runtime bindings are missing")
    _verify_runtime_environment(str(runtime_bindings["environment_manifest_sha256"]))
    manifest_binding = runtime_bindings.get("materialization_manifest")
    if not isinstance(manifest_binding, Mapping):
        raise ContractError("run scope materialization manifest binding is missing")
    manifest, source_bindings, projection_bindings = _verify_materialization_manifest(
        root=repo_root,
        registered=registered,
        manifest_binding=manifest_binding,
        verify_candidate_file=False,
        verify_settlement_file=False,
    )
    if source_bindings != runtime_bindings.get("source_bindings") or projection_bindings != runtime_bindings.get("projection_bindings"):
        raise ContractError("materialization manifest differs from the frozen scope")
    if manifest.get("implementation_binding") != {
        "implementation_commit": run_scope["run_scope_base_commit"],
        "runtime_material_bundle_sha256": canonical_digest(
            dict(registered.runtime_material_digests)
        ),
    }:
        raise ContractError("materialization implementation binding differs from the run scope")
    approval = _approval_module()
    initial_approval = _snapshot_approval_evidence(
        approval.verify_offline_run_approval(
            root=repo_root,
            run_scope=run_scope,
            issue_number=issue_number,
            comment_id=comment_id,
            provider=provider,
            now=now,
        ),
        checkpoint="INITIAL_APPROVAL",
        issue_number=issue_number,
        comment_id=comment_id,
        expected_run_scope_digest=run_scope["run_scope_digest"],
        expected_original_evidence_digest=None,
        strict_live_evidence=_require_live_token,
    )
    _verify_deterministic_materialization_against_raw(
        repo_root=repo_root,
        source_root=raw_root,
        registered=registered,
        availability={"implementation_commit": run_scope["run_scope_base_commit"]},
    )
    del raw_root, source_root
    candidate_approval = _snapshot_approval_evidence(
        approval.reverify_offline_run_approval(
            root=repo_root,
            run_scope=run_scope,
            approval_evidence=initial_approval,
            provider=provider,
            now=now,
            checkpoint="BEFORE_CANDIDATE_OPEN",
        ),
        checkpoint="BEFORE_CANDIDATE_OPEN",
        issue_number=issue_number,
        comment_id=comment_id,
        expected_run_scope_digest=run_scope["run_scope_digest"],
        expected_original_evidence_digest=initial_approval["evidence_digest"],
        strict_live_evidence=_require_live_token,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    start_staging = Path(
        tempfile.mkdtemp(prefix=".offline-start-", dir=output_root.parent)
    )
    try:
        staged_initial_evidence_path = start_staging / "approval_evidence_initial.json"
        _write_json_exclusive(staged_initial_evidence_path, initial_approval)
        _verify_persisted_mapping(
            staged_initial_evidence_path,
            initial_approval,
            label="initial approval evidence",
        )
        staged_candidate_evidence_path = (
            start_staging / "approval_evidence_before_candidate.json"
        )
        _write_json_exclusive(staged_candidate_evidence_path, candidate_approval)
        _verify_persisted_mapping(
            staged_candidate_evidence_path,
            candidate_approval,
            label="candidate-boundary approval evidence",
        )
        start_receipt = _start_receipt(
            run_scope=run_scope, approval_evidence=candidate_approval
        )
        _write_json_exclusive(start_staging / "start_receipt.json", start_receipt)
        _verify_persisted_mapping(
            start_staging / "start_receipt.json",
            start_receipt,
            label="exclusive start receipt",
        )
        if output_root.exists():
            _recover_or_reject_existing_output(output_root, run_scope)
        os.replace(start_staging, output_root)
    except BaseException:
        if start_staging.exists():
            shutil.rmtree(start_staging)
        if output_root.exists():
            _recover_or_reject_existing_output(output_root, run_scope)
        raise

    try:
        candidate_evidence_path = (
            output_root / "approval_evidence_before_candidate.json"
        )
        _verify_persisted_mapping(
            output_root / "approval_evidence_initial.json",
            initial_approval,
            label="initial approval evidence",
        )
        _verify_persisted_mapping(
            candidate_evidence_path,
            candidate_approval,
            label="candidate-boundary approval evidence",
        )
        _verify_persisted_mapping(
            output_root / "start_receipt.json",
            start_receipt,
            label="exclusive start receipt",
        )
        _candidate_path, candidate_payload = _read_bound_bytes(
            repo_root,
            projection_bindings["candidate_projection"],
            label="candidate projection",
        )
        # NumPy is imported and version-checked before the application firewall is active.
        import numpy  # noqa: F401

        with _application_workload_firewall():
            candidate_rows = _load_jsonl_exact_bytes(candidate_payload)
            ordered_race_digest = canonical_digest(
                [row.get("race_id") for row in sorted(candidate_rows, key=lambda item: str(item.get("race_id")))]
            )
            if ordered_race_digest != manifest["ordered_race_id_sha256"]:
                raise ContractError("ordered race-id digest differs from materialization manifest")
            decisions: dict[str, list[dict[str, Any]]] = {}
            decision_projections: dict[str, dict[str, Any]] = {}
            for replica in REPLICA_IDS:
                replica_rows, replica_projection = _freeze_decisions(
                    recipe=registered.recipe,
                    candidate_rows=candidate_rows,
                )
                decisions[replica] = replica_rows
                decision_projections[replica] = replica_projection
            if (
                _canonical_bytes(decisions["clean_a"])
                != _canonical_bytes(decisions["clean_b"])
                or decision_projections["clean_a"]
                != decision_projections["clean_b"]
            ):
                raise ContractError("logical replica decision semantics differ")
            freeze_receipt = _freeze_receipt(
                run_scope=run_scope,
                start_receipt=start_receipt,
                per_replica=decision_projections,
                candidate_file_sha256=projection_bindings["candidate_projection"]["sha256"],
            )
            _write_json_exclusive(
                output_root / "decision_freeze_receipt.json", freeze_receipt
            )
            _verify_persisted_mapping(
                output_root / "decision_freeze_receipt.json",
                freeze_receipt,
                label="decision-freeze receipt before settlement access",
            )

            _settlement_path, settlement_payload = _read_bound_bytes(
                repo_root,
                projection_bindings["settlement_projection"],
                label="settlement projection",
            )
            settlement_rows = _load_jsonl_exact_bytes(settlement_payload)
            scientific_by_replica: dict[str, dict[str, Any]] = {}
            outcomes: dict[str, str] = {}
            for replica in REPLICA_IDS:
                projection_value, outcome = _scientific_projection(
                    recipe=registered.recipe,
                    decision_rows=decisions[replica],
                    decision_projection=decision_projections[replica],
                    settlement_rows=settlement_rows,
                    bootstrap_replicates=bootstrap_replicates,
                )
                scientific_by_replica[replica] = projection_value
                outcomes[replica] = outcome
            replica_digests = {
                replica: canonical_digest(scientific_by_replica[replica])
                for replica in REPLICA_IDS
            }
            canonical_projection_payloads = {
                _canonical_bytes(scientific_by_replica[replica])
                for replica in REPLICA_IDS
            }
            unique_outcomes = set(outcomes.values())
            if (
                len(canonical_projection_payloads) != 1
                or len(set(replica_digests.values())) != 1
                or len(unique_outcomes) != 1
            ):
                raise ContractError("logical replica scientific projections differ")
            canonical_scientific_projection = _load_strict_json_bytes(
                canonical_projection_payloads.pop(),
                label="replica-equal scientific projection",
            )
            computed_outcome = unique_outcomes.pop()
            del (
                candidate_payload,
                candidate_rows,
                settlement_payload,
                settlement_rows,
                decisions,
                scientific_by_replica,
            )

        # Only hash checks and required approval revalidation occur outside the deny guard;
        # no candidate, settlement, or scientific row payload is passed to those calls.
        _verify_file_binding(
            repo_root,
            manifest_binding,
            label="materialization manifest",
        )
        _verify_file_binding(
            repo_root,
            projection_bindings["candidate_projection"],
            label="candidate projection",
        )
        _verify_file_binding(
            repo_root,
            projection_bindings["settlement_projection"],
            label="settlement projection",
        )
        final_approval = _snapshot_approval_evidence(
            approval.reverify_offline_run_approval(
                root=repo_root,
                run_scope=run_scope,
                approval_evidence=initial_approval,
                provider=provider,
                now=now,
                checkpoint="BEFORE_RESULT_PUBLISH",
            ),
            checkpoint="BEFORE_RESULT_PUBLISH",
            issue_number=issue_number,
            comment_id=comment_id,
            expected_run_scope_digest=run_scope["run_scope_digest"],
            expected_original_evidence_digest=initial_approval["evidence_digest"],
            strict_live_evidence=_require_live_token,
        )
        final_approval_digest = final_approval["evidence_digest"]
        final_evidence_path = output_root / "approval_evidence_before_result.json"
        _write_json_exclusive(final_evidence_path, final_approval)
        _verify_persisted_mapping(
            output_root / "approval_evidence_initial.json",
            initial_approval,
            label="initial approval evidence",
        )
        _verify_persisted_mapping(
            candidate_evidence_path,
            candidate_approval,
            label="candidate-boundary approval evidence",
        )
        _verify_persisted_mapping(
            final_evidence_path,
            final_approval,
            label="result-boundary approval evidence",
        )
        _verify_persisted_mapping(
            output_root / "start_receipt.json",
            start_receipt,
            label="exclusive start receipt",
        )
        _verify_persisted_mapping(
            output_root / "decision_freeze_receipt.json",
            freeze_receipt,
            label="decision-freeze receipt",
        )
        result: dict[str, Any] = {
            "schema_version": 1,
            "gate_kind": GATE_KIND,
            "lifecycle_state": "RNOD_COMPLETED",
            "run_scope_digest": run_scope["run_scope_digest"],
            "semantic_subject_digest": run_scope["semantic_subject_digest"],
            "exact_subject_digest": run_scope["exact_subject_digest"],
            "recipe_id": RECIPE_ID,
            "recipe_version": RECIPE_VERSION,
            "recipe_digest": registered.recipe_digest,
            "approval_evidence_digest": final_approval_digest,
            "source_bindings_digest": canonical_digest(source_bindings),
            "projection_bindings_digest": canonical_digest(projection_bindings),
            "decision_freeze_receipt_digest": freeze_receipt["receipt_digest"],
            "scientific_projection": canonical_scientific_projection,
            "scientific_projection_digest": canonical_digest(canonical_scientific_projection),
            "computed_outcome": computed_outcome,
            "replica_mode": REPLICA_MODE,
            "replica_attempt_count": 1,
            "replica_scientific_projection_digests": replica_digests,
            "replica_semantic_equality": True,
            "authority": False,
            **LIMITATIONS,
            **NONPROMOTION,
        }
        result["result_digest"] = canonical_digest(result)
        result_seal_receipt = _result_seal_receipt(
            run_scope=run_scope,
            decision_freeze_receipt=freeze_receipt,
            approval_evidence_digest=final_approval_digest,
            completed_result=result,
        )
        _write_json_exclusive(
            output_root / "result_seal_receipt.json", result_seal_receipt
        )
        _verify_persisted_mapping(
            output_root / "result_seal_receipt.json",
            result_seal_receipt,
            label="result-seal receipt",
        )
        invalid_path = output_root / "INVALID.json"
        if invalid_path.exists() or invalid_path.is_symlink():
            raise ContractError("INVALID tombstone appeared before result publish")
        _atomic_publish_json(output_root / "result.json", result)
        return result
    except BaseException:
        # `os.replace(..., result.json)` is the local completion point.  An
        # asynchronous exception may arrive after that replace has committed
        # even though `_atomic_publish_json` did not return to this frame.  In
        # that case the already sealed canonical chain is COMPLETED and must
        # never be made ambiguous by adding an INVALID tombstone.
        _raise_poststart_terminal_after_exception(output_root, run_scope)


def execute_offline_registered_diagnostic(
    *,
    root: Path,
    source_root: Path,
    run_scope: Mapping[str, Any],
    issue_number: int,
    comment_id: int,
) -> dict[str, Any]:
    """Canonical live entry point with an internally created provider and real time."""

    safe_root = _safe_root(Path(root), label="repository root")
    frozen_scope = _snapshot_mapping(run_scope, label="canonical live run scope")
    digest = frozen_scope.get("run_scope_digest")
    stored_scope = _load_fixed_scope_artifact(safe_root, digest)
    if _canonical_bytes(stored_scope) != _canonical_bytes(frozen_scope):
        raise ContractError("live run scope differs from its canonical digest-path artifact")
    return _execute_offline_registered_diagnostic_worker(
        root=safe_root,
        source_root=source_root,
        run_scope=frozen_scope,
        issue_number=issue_number,
        comment_id=comment_id,
        provider=None,
        now=None,
        _require_live_token=True,
    )


def _github_provider() -> Any:
    from github_approval import GitHubRestApprovalProvider

    return GitHubRestApprovalProvider()


def _load_fixed_scope_artifact(root: Path, digest: str) -> dict[str, Any]:
    if not isinstance(digest, str) or not FULL_SHA256.fullmatch(digest):
        raise ContractError("run-scope digest must be a full lowercase SHA-256")
    relative = offline_run_scope_artifact_path(digest)
    path = _safe_bound_path(root, relative, label="run scope artifact", must_exist=True)
    raw = path.read_bytes()
    scope = _load_strict_json_bytes(raw, label="run scope artifact")
    if not isinstance(scope, dict) or scope.get("run_scope_digest") != digest:
        raise ContractError("run scope artifact does not match its digest-derived path")
    if raw != _canonical_bytes(scope):
        raise ContractError("run scope artifact is not canonical UTF-8/LF JSON")
    return scope


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact lightweight offline non-promotion diagnostic runner"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser(
        "materialize", help="create fixed decision-free candidate/settlement projections"
    )
    materialize.add_argument("--source-root", type=Path, required=True)

    compile_scope = subparsers.add_parser(
        "compile-scope", help="seal the only fixed run scope at a digest-derived path"
    )
    compile_scope.add_argument("--source-root", type=Path, required=True)

    run = subparsers.add_parser("run", help="execute the one approved offline diagnostic")
    run.add_argument("--run-scope-digest", required=True)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--issue-number", type=int, required=True)
    run.add_argument("--comment-id", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    root = Path(arguments.root)
    try:
        if arguments.command == "materialize":
            evidence = materialize_fixed_projections(
                root=root,
                source_root=arguments.source_root,
            )
            print(json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if arguments.command == "compile-scope":
            scope, path = compile_fixed_run_scope_artifact(
                root=root,
                source_root=arguments.source_root,
            )
            print(
                json.dumps(
                    {
                        "status": "RNOD_RUN_APPROVAL_REQUIRED",
                        "run_scope_digest": scope["run_scope_digest"],
                        "run_scope_path": path.relative_to(root.resolve()).as_posix(),
                        "required_comment": (
                            "APPROVED_OFFLINE_NONPROMOTION_DIAGNOSTIC_RUN "
                            + scope["run_scope_digest"]
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if arguments.command == "run":
            safe_root = _safe_root(root, label="repository root")
            scope = _load_fixed_scope_artifact(
                safe_root, arguments.run_scope_digest
            )
            result = execute_offline_registered_diagnostic(
                root=safe_root,
                source_root=arguments.source_root,
                run_scope=scope,
                issue_number=arguments.issue_number,
                comment_id=arguments.comment_id,
            )
            print(
                json.dumps(
                    {
                        "status": result["lifecycle_state"],
                        "result_digest": result["result_digest"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        raise AssertionError("unreachable command")
    except OfflineRunAlreadyCompleted:
        print(
            '{"reason_code":"ALREADY_COMPLETED_NO_RERUN","status":"RNOD_COMPLETED"}',
            file=sys.stderr,
        )
        return 3
    except OfflineRunAlreadyRunning:
        print(
            '{"reason_code":"LOCAL_RUN_LOCK_HELD","status":"RNOD_RUNNING"}',
            file=sys.stderr,
        )
        return 3
    except OfflineInvalidAfterStart:
        print(
            '{"reason_code":"INVALID_AFTER_START_NO_RETRY","status":"INVALID"}',
            file=sys.stderr,
        )
        return 2
    except Exception:
        print(
            '{"reason_code":"PREACCESS_CONTRACT_FAILURE","status":"BLOCKED_PREACCESS"}',
            file=sys.stderr,
        )
        return 2


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CALIBRATOR_ABS_TOLERANCE",
    "CALIBRATOR_OFFSET",
    "MATERIALIZATION_MANIFEST_KIND",
    "compile_fixed_run_scope_artifact",
    "execute_offline_registered_diagnostic",
    "main",
    "materialize_fixed_projections",
]


if __name__ == "__main__":
    raise SystemExit(main())
