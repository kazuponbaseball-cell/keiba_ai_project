#!/usr/bin/env python3
"""Fail-closed execution broker facade for ``ordinary_real_data_run_v3``.

This module is a consumer of the frozen v3 contract.  It does not add a new
authority source and deliberately does not reinterpret receipts, scopes, or
result manifests.  The current real-host backend is unsupported: real row,
output, receipt, and sealing operations stop before the frozen contract can
open or create anything.

Synthetic fixture support is a separate type.  Its assurance marker can never
be supplied to the real-data factory and it never creates a v3 receipt or an
ephemeral real-data authorization decision.
"""

from __future__ import annotations

import hashlib
import os
import platform
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import ordinary_real_data_run_contract_v3 as v3_contract


BROKER_SCHEMA_VERSION = "ordinary_real_data_execution_broker_v1"
SYNTHETIC_ASSURANCE = "SYNTHETIC_TEST_ONLY"


class ExecutionBoundaryError(RuntimeError):
    """A fail-closed boundary rejection with a stable machine-readable code."""

    def __init__(self, reason_code: str, detail: str) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("reason_code must be a nonempty string")
        if not isinstance(detail, str) or not detail:
            raise ValueError("detail must be a nonempty string")
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


class EnforcementStatus(str, Enum):
    """Only statuses that may be reported for a real-host backend."""

    ENFORCED = "ENFORCED"
    UNSUPPORTED_FAIL_CLOSED = "UNSUPPORTED_FAIL_CLOSED"


@dataclass(frozen=True)
class RealHostEnforcementReport:
    """Code-owned observation of the three mandatory OS enforcement planes."""

    backend_id: str
    host_system: str
    network_isolation: EnforcementStatus
    filesystem_broker_isolation: EnforcementStatus
    resource_supervision: EnforcementStatus
    provenance: str = "REAL_HOST"

    def __post_init__(self) -> None:
        if self.provenance != "REAL_HOST":
            raise ValueError("real-host enforcement provenance must be REAL_HOST")
        if not self.backend_id or not self.host_system:
            raise ValueError("real-host enforcement report requires backend and host")

    @property
    def real_data_ready(self) -> bool:
        return all(
            status is EnforcementStatus.ENFORCED
            for status in (
                self.network_isolation,
                self.filesystem_broker_isolation,
                self.resource_supervision,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "broker_schema_version": BROKER_SCHEMA_VERSION,
            "backend_id": self.backend_id,
            "host_system": self.host_system,
            "provenance": self.provenance,
            "network_isolation": self.network_isolation.value,
            "filesystem_broker_isolation": self.filesystem_broker_isolation.value,
            "resource_supervision": self.resource_supervision.value,
            "real_data_ready": self.real_data_ready,
            "real_data_execution_allowed": False,
            "execution_authorized": False,
            "automatic_execution_allowed": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }


@dataclass(frozen=True)
class SyntheticFixtureEnforcement:
    """Test-only marker; intentionally not shaped like a real-host report."""

    fixture_id: str
    assurance: str = SYNTHETIC_ASSURANCE
    execution_kind: str = "synthetic"

    def __post_init__(self) -> None:
        if not isinstance(self.fixture_id, str) or not self.fixture_id:
            raise ExecutionBoundaryError(
                "SYNTHETIC_FIXTURE_ID_INVALID", "synthetic fixture_id is required"
            )
        if self.assurance != SYNTHETIC_ASSURANCE or self.execution_kind != "synthetic":
            raise ExecutionBoundaryError(
                "SYNTHETIC_BACKEND_NOT_REAL_AUTHORITY",
                "synthetic enforcement marker cannot claim real assurance",
            )


@dataclass(frozen=True)
class BrokerReadRequest:
    """Exact caller claim checked against one frozen v3 allowlist entry."""

    scope_version: str
    execution_kind: str
    capability_profile_id: str
    phase_id: str
    path: str
    access_class: str
    required_capability: str | None
    expected_sha256: str


@dataclass(frozen=True)
class BrokerWriteRequest:
    """Exact caller claim checked against one frozen v3 write allowlist entry."""

    scope_version: str
    execution_kind: str
    capability_profile_id: str
    phase_id: str
    path: str
    required_capability: str


@dataclass
class AccessMeter:
    """Separates real row delivery from synthetic fixture delivery."""

    real_data_rows_opened: int = 0
    synthetic_blobs_opened: int = 0
    bytes_delivered: int = 0


def observe_real_host_enforcement() -> RealHostEnforcementReport:
    """Return the conservative current-host status without capability guessing.

    No supported OS backend is implemented in v1.  In particular, Windows Job
    Objects alone would not prove network and broker-only filesystem isolation,
    and Python monkeypatching is never accepted as an OS enforcement backend.
    """

    host = platform.system() or "Unknown"
    return RealHostEnforcementReport(
        backend_id=f"unsupported-{host.casefold()}-v1",
        host_system=host,
        network_isolation=EnforcementStatus.UNSUPPORTED_FAIL_CLOSED,
        filesystem_broker_isolation=EnforcementStatus.UNSUPPORTED_FAIL_CLOSED,
        resource_supervision=EnforcementStatus.UNSUPPORTED_FAIL_CLOSED,
    )


def _require_real_enforcement(report: RealHostEnforcementReport) -> None:
    """Reject the first unavailable enforcement plane in stable order."""

    if not isinstance(report, RealHostEnforcementReport):
        raise ExecutionBoundaryError(
            "SYNTHETIC_BACKEND_NOT_REAL_AUTHORITY",
            "real-data execution accepts only a code-owned real-host report",
        )
    if report.network_isolation is not EnforcementStatus.ENFORCED:
        raise ExecutionBoundaryError(
            "NETWORK_ISOLATION_UNAVAILABLE",
            "OS-level network isolation is not verifiably enforced",
        )
    if report.filesystem_broker_isolation is not EnforcementStatus.ENFORCED:
        raise ExecutionBoundaryError(
            "BROKER_ONLY_FILESYSTEM_ISOLATION_UNAVAILABLE",
            "the runner can not yet be confined to broker-mediated filesystem access",
        )
    if report.resource_supervision is not EnforcementStatus.ENFORCED:
        raise ExecutionBoundaryError(
            "RESOURCE_ENFORCEMENT_UNAVAILABLE",
            "the frozen timeout/compute/memory/disk budget is not OS-enforced",
        )


def _canonical_real_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Require the caller value to already be exact canonical real-data v3."""

    if not isinstance(scope, dict):
        raise ExecutionBoundaryError(
            "RUN_SCOPE_NOT_EXACT_V3", "run scope must be an exact object"
        )
    if "run_scope_schema_version" not in scope:
        raise ExecutionBoundaryError(
            "LEGACY_V2_REAL_DATA_FORBIDDEN",
            "a versionless legacy-v2 scope cannot request real-data execution",
        )
    version = scope["run_scope_schema_version"]
    if version != v3_contract.RUN_SCOPE_SCHEMA_VERSION:
        raise ExecutionBoundaryError(
            "RUN_SCOPE_VERSION_UNKNOWN", "only exact ordinary_real_data_run_v3 is accepted"
        )
    if scope.get("execution_kind") != "real_data":
        raise ExecutionBoundaryError(
            "EXECUTION_KIND_MISMATCH", "the real broker requires execution_kind=real_data"
        )
    try:
        normalized = v3_contract.normalize_ordinary_real_data_run_scope(
            scope,
            proposal_scope=scope["proposal_scope"],
        )
    except (KeyError, TypeError, ValueError, v3_contract.ContractError) as exc:
        raise ExecutionBoundaryError(
            "RUN_SCOPE_NOT_EXACT_V3", "the frozen v3 normalizer rejected the scope"
        ) from exc
    if normalized != scope:
        raise ExecutionBoundaryError(
            "RUN_SCOPE_NOT_EXACT_V3", "the supplied scope is not its canonical v3 value"
        )
    return normalized


def _repository_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        raise ExecutionBoundaryError("PATH_NOT_ALLOWLISTED", f"{label} is not canonical")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ExecutionBoundaryError("PATH_NOT_ALLOWLISTED", f"{label} escapes the repository")
    return path.as_posix()


def assert_no_link_or_reparse(path: Path, *, label: str) -> Path:
    """Reject symlink/junction/reparse components before resolving a path."""

    lexical = Path(os.path.abspath(os.fspath(path)))
    candidates = [lexical, *lexical.parents]
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for candidate in candidates:
        if not os.path.lexists(candidate):
            continue
        try:
            observed = os.lstat(candidate)
        except OSError as exc:
            raise ExecutionBoundaryError(
                "FILESYSTEM_INSPECTION_FAILED", f"cannot inspect {label}"
            ) from exc
        attributes = getattr(observed, "st_file_attributes", 0)
        if stat.S_ISLNK(observed.st_mode) or attributes & reparse_flag:
            raise ExecutionBoundaryError(
                "SYMLINK_OR_REPARSE_ESCAPE", f"{label} contains a linked/reparse component"
            )
    return lexical


def _full_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _phase_and_read_entry(
    scope: dict[str, Any], request: BrokerReadRequest
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(request, BrokerReadRequest):
        raise ExecutionBoundaryError(
            "BROKER_REQUEST_TYPE_MISMATCH", "typed BrokerReadRequest is required"
        )
    if request.scope_version != scope["run_scope_schema_version"]:
        raise ExecutionBoundaryError("RUN_SCOPE_NOT_EXACT_V3", "request scope version drifted")
    if request.execution_kind != scope["execution_kind"]:
        raise ExecutionBoundaryError("EXECUTION_KIND_MISMATCH", "request execution kind drifted")
    if request.capability_profile_id != scope["capability_profile"]["profile_id"]:
        raise ExecutionBoundaryError(
            "CAPABILITY_PROFILE_MISMATCH", "request capability profile drifted"
        )
    phase = next(
        (item for item in scope["phase_plan"] if item["phase_id"] == request.phase_id),
        None,
    )
    if phase is None:
        raise ExecutionBoundaryError("PHASE_MISMATCH", "request phase is not frozen")
    canonical_path = _repository_path(request.path, "read path")
    entry = next(
        (item for item in scope["read_allowlist"] if item["path"] == canonical_path),
        None,
    )
    if entry is None or canonical_path not in phase["read_paths"]:
        raise ExecutionBoundaryError("PATH_NOT_ALLOWLISTED", "read path is not phase-allowlisted")
    if request.access_class != entry["access_class"]:
        requested_row = request.access_class.endswith("_row_blob")
        frozen_row = entry["access_class"].endswith("_row_blob")
        if requested_row and not frozen_row:
            code = "METADATA_AS_ROW_RELABEL_FORBIDDEN"
        elif frozen_row and not requested_row:
            code = "ROW_AS_METADATA_RELABEL_FORBIDDEN"
        else:
            code = "ACCESS_CLASS_MISMATCH"
        raise ExecutionBoundaryError(code, "requested access class differs from the frozen entry")
    if request.required_capability != entry["required_capability"]:
        raise ExecutionBoundaryError(
            "CAPABILITY_MISMATCH", "requested capability differs from the frozen entry"
        )
    capability = request.required_capability
    if capability is not None:
        if (
            capability not in phase["required_capabilities"]
            or scope["capability_profile"]["capabilities"].get(capability) is not True
        ):
            raise ExecutionBoundaryError(
                "CAPABILITY_MISMATCH", "phase/profile lacks the requested capability"
            )
    if not _full_sha256(request.expected_sha256) or request.expected_sha256 != entry["sha256"]:
        raise ExecutionBoundaryError(
            "INPUT_HASH_BINDING_MISMATCH", "requested input hash differs from the frozen entry"
        )
    return phase, entry


def _phase_and_write_entry(
    scope: dict[str, Any], request: BrokerWriteRequest
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(request, BrokerWriteRequest):
        raise ExecutionBoundaryError(
            "BROKER_REQUEST_TYPE_MISMATCH", "typed BrokerWriteRequest is required"
        )
    if request.scope_version != scope["run_scope_schema_version"]:
        raise ExecutionBoundaryError("RUN_SCOPE_NOT_EXACT_V3", "request scope version drifted")
    if request.execution_kind != scope["execution_kind"]:
        raise ExecutionBoundaryError("EXECUTION_KIND_MISMATCH", "request execution kind drifted")
    if request.capability_profile_id != scope["capability_profile"]["profile_id"]:
        raise ExecutionBoundaryError(
            "CAPABILITY_PROFILE_MISMATCH", "request capability profile drifted"
        )
    phase = next(
        (item for item in scope["phase_plan"] if item["phase_id"] == request.phase_id),
        None,
    )
    if phase is None:
        raise ExecutionBoundaryError("PHASE_MISMATCH", "request phase is not frozen")
    canonical_path = _repository_path(request.path, "write path")
    entry = next(
        (item for item in scope["write_allowlist"] if item["path"] == canonical_path),
        None,
    )
    if entry is None or canonical_path not in phase["write_paths"]:
        raise ExecutionBoundaryError("PATH_NOT_ALLOWLISTED", "write path is not phase-allowlisted")
    if (
        request.required_capability != entry["required_capability"]
        or request.required_capability not in phase["required_capabilities"]
        or scope["capability_profile"]["capabilities"].get(request.required_capability)
        is not True
    ):
        raise ExecutionBoundaryError(
            "CAPABILITY_MISMATCH", "write capability differs from the frozen phase/profile"
        )
    return phase, entry


_REAL_BROKER_FACTORY_TOKEN = object()


class OrdinaryRealDataExecutionBrokerV1:
    """Consumer facade around the frozen v3 receipt/access/seal functions."""

    def __init__(
        self,
        *,
        enforcement: RealHostEnforcementReport,
        meter: AccessMeter,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _REAL_BROKER_FACTORY_TOKEN:
            raise ExecutionBoundaryError(
                "REAL_BROKER_FACTORY_REQUIRED", "real broker construction is code-owned"
            )
        if not isinstance(enforcement, RealHostEnforcementReport):
            raise ExecutionBoundaryError(
                "SYNTHETIC_BACKEND_NOT_REAL_AUTHORITY",
                "synthetic enforcement cannot construct a real broker",
            )
        if not isinstance(meter, AccessMeter):
            raise ExecutionBoundaryError(
                "ACCESS_METER_TYPE_MISMATCH", "real broker requires a typed access meter"
            )
        self._enforcement = observe_real_host_enforcement()
        self._meter = meter

    @property
    def enforcement(self) -> RealHostEnforcementReport:
        return self._enforcement

    @property
    def meter(self) -> AccessMeter:
        return self._meter

    def assert_real_execution_ready(self) -> None:
        observed = observe_real_host_enforcement()
        _require_real_enforcement(observed)
        raise ExecutionBoundaryError(
            "REAL_EXECUTION_BACKEND_NOT_IMPLEMENTED",
            "v1 has no real OS backend even if a report value is forged or monkeypatched",
        )

    def issue_execution_receipt(self, **frozen_v3_arguments: Any) -> dict[str, Any]:
        self.assert_real_execution_ready()
        try:
            return v3_contract.issue_execution_receipt(**frozen_v3_arguments)
        except (AttributeError, KeyError, OSError, TypeError, ValueError, v3_contract.ContractError) as exc:
            raise ExecutionBoundaryError(
                "V3_RECEIPT_ISSUANCE_REJECTED", "the frozen v3 issuer rejected authority"
            ) from exc

    def open_metadata(
        self,
        *,
        root: Path,
        scope: dict[str, Any],
        request: BrokerReadRequest,
    ) -> bytes:
        canonical = _canonical_real_scope(scope)
        _, entry = _phase_and_read_entry(canonical, request)
        if entry["access_class"] != "metadata_manifest":
            raise ExecutionBoundaryError(
                "ROW_AS_METADATA_RELABEL_FORBIDDEN",
                "open_metadata accepts only a frozen metadata_manifest entry",
            )
        try:
            payload = v3_contract.read_authorized_bytes(
                root,
                canonical,
                phase_id=request.phase_id,
                path=request.path,
                metadata_only=True,
            )
        except (AttributeError, KeyError, OSError, TypeError, ValueError, v3_contract.ContractError) as exc:
            raise ExecutionBoundaryError(
                "V3_METADATA_ACCESS_REJECTED", "the frozen v3 metadata gate rejected access"
            ) from exc
        if hashlib.sha256(payload).hexdigest() != request.expected_sha256:
            raise ExecutionBoundaryError(
                "INPUT_HASH_MISMATCH", "metadata bytes changed before caller delivery"
            )
        self._meter.bytes_delivered += len(payload)
        return payload

    def open_row_blob(
        self,
        *,
        root: Path,
        scope: dict[str, Any],
        request: BrokerReadRequest,
        authority_context: dict[str, Any],
    ) -> bytes:
        canonical = _canonical_real_scope(scope)
        _, entry = _phase_and_read_entry(canonical, request)
        if not entry["access_class"].endswith("_row_blob"):
            raise ExecutionBoundaryError(
                "METADATA_AS_ROW_RELABEL_FORBIDDEN",
                "open_row_blob accepts only a frozen row access class",
            )
        self.assert_real_execution_ready()
        try:
            payload = v3_contract.read_authorized_bytes(
                root,
                canonical,
                phase_id=request.phase_id,
                path=request.path,
                authority_context=authority_context,
                metadata_only=False,
            )
        except (AttributeError, KeyError, OSError, TypeError, ValueError, v3_contract.ContractError) as exc:
            raise ExecutionBoundaryError(
                "V3_LIVE_AUTHORITY_REVALIDATION_FAILED",
                "the frozen v3 broker rejected row authority or content",
            ) from exc
        if hashlib.sha256(payload).hexdigest() != request.expected_sha256:
            raise ExecutionBoundaryError(
                "INPUT_HASH_MISMATCH", "row bytes changed before caller delivery"
            )
        self._meter.real_data_rows_opened += 1
        self._meter.bytes_delivered += len(payload)
        return payload

    def open_sealed_input(
        self,
        *,
        root: Path,
        scope: dict[str, Any],
        request: BrokerReadRequest,
        authority_context: dict[str, Any],
    ) -> bytes:
        if not isinstance(request, BrokerReadRequest):
            raise ExecutionBoundaryError(
                "BROKER_REQUEST_TYPE_MISMATCH", "typed BrokerReadRequest is required"
            )
        if request.access_class != "sealed_input_row_blob":
            raise ExecutionBoundaryError(
                "ACCESS_CLASS_MISMATCH",
                "open_sealed_input requires sealed_input_row_blob",
            )
        return self.open_row_blob(
            root=root,
            scope=scope,
            request=request,
            authority_context=authority_context,
        )

    def open_output(
        self,
        *,
        root: Path,
        scope: dict[str, Any],
        request: BrokerWriteRequest,
        payload: bytes,
        authority_context: dict[str, Any],
    ) -> None:
        canonical = _canonical_real_scope(scope)
        _phase_and_write_entry(canonical, request)
        self.assert_real_execution_ready()
        try:
            v3_contract.write_authorized_bytes(
                root,
                canonical,
                phase_id=request.phase_id,
                path=request.path,
                payload=payload,
                authority_context=authority_context,
            )
        except (AttributeError, KeyError, OSError, TypeError, ValueError, v3_contract.ContractError) as exc:
            raise ExecutionBoundaryError(
                "V3_OUTPUT_ACCESS_REJECTED", "the frozen v3 output gate rejected access"
            ) from exc

    def _seal(self, expected_status: str, **frozen_v3_arguments: Any) -> dict[str, Any]:
        self.assert_real_execution_ready()
        manifest = frozen_v3_arguments.get("result_manifest")
        if not isinstance(manifest, dict) or manifest.get("status") != expected_status:
            raise ExecutionBoundaryError(
                "OUTPUT_SEAL_STATUS_MISMATCH", "seal API and v3 manifest status differ"
            )
        try:
            return v3_contract.seal_output_manifest(**frozen_v3_arguments)
        except (AttributeError, KeyError, OSError, TypeError, ValueError, v3_contract.ContractError) as exc:
            raise ExecutionBoundaryError(
                "V3_OUTPUT_SEAL_REJECTED", "the frozen v3 immutable sealer rejected output"
            ) from exc

    def seal_success(self, **frozen_v3_arguments: Any) -> dict[str, Any]:
        return self._seal("success", **frozen_v3_arguments)

    def seal_failure(self, **frozen_v3_arguments: Any) -> dict[str, Any]:
        return self._seal("failure", **frozen_v3_arguments)


def build_real_data_execution_broker(
    *,
    meter: AccessMeter | None = None,
    backend: object | None = None,
) -> OrdinaryRealDataExecutionBrokerV1:
    """Build the only public real broker; backend injection is forbidden."""

    if backend is not None:
        raise ExecutionBoundaryError(
            "SYNTHETIC_BACKEND_NOT_REAL_AUTHORITY",
            "callers cannot inject an enforcement backend into the real factory",
        )
    return OrdinaryRealDataExecutionBrokerV1(
        enforcement=observe_real_host_enforcement(),
        meter=meter if meter is not None else AccessMeter(),
        _factory_token=_REAL_BROKER_FACTORY_TOKEN,
    )


@dataclass(frozen=True)
class SyntheticFixtureEntry:
    path: str
    access_class: str
    required_capability: str | None
    sha256: str


class SyntheticFixtureBrokerV1:
    """Hash-checking fixture broker with no real-data or v3 authority surface."""

    def __init__(
        self,
        *,
        root: Path,
        enforcement: SyntheticFixtureEnforcement,
        entries: Mapping[str, SyntheticFixtureEntry],
        meter: AccessMeter | None = None,
    ) -> None:
        if not isinstance(enforcement, SyntheticFixtureEnforcement):
            raise ExecutionBoundaryError(
                "SYNTHETIC_FIXTURE_MARKER_REQUIRED", "fixture broker requires test-only assurance"
            )
        lexical_root = assert_no_link_or_reparse(root, label="synthetic fixture root")
        resolved = lexical_root.resolve()
        if not resolved.is_dir():
            raise ExecutionBoundaryError(
                "SYNTHETIC_FIXTURE_ROOT_INVALID", "fixture root must be an exact directory"
            )
        self._root = resolved
        self._enforcement = enforcement
        if not isinstance(entries, Mapping):
            raise ExecutionBoundaryError(
                "SYNTHETIC_FIXTURE_POLICY_INVALID", "fixture entries must be a mapping"
            )
        normalized_entries: dict[str, SyntheticFixtureEntry] = {}
        for key, entry in entries.items():
            if not isinstance(entry, SyntheticFixtureEntry):
                raise ExecutionBoundaryError(
                    "SYNTHETIC_FIXTURE_POLICY_INVALID", "fixture entry must be typed"
                )
            canonical_key = _repository_path(key, "synthetic fixture policy path")
            if (
                entry.path != canonical_key
                or not isinstance(entry.access_class, str)
                or not entry.access_class
                or (
                    entry.required_capability is not None
                    and (
                        not isinstance(entry.required_capability, str)
                        or not entry.required_capability
                    )
                )
                or not _full_sha256(entry.sha256)
            ):
                raise ExecutionBoundaryError(
                    "SYNTHETIC_FIXTURE_POLICY_INVALID", "fixture entry is not exact"
                )
            normalized_entries[canonical_key] = entry
        self._entries = normalized_entries
        if meter is not None and not isinstance(meter, AccessMeter):
            raise ExecutionBoundaryError(
                "ACCESS_METER_TYPE_MISMATCH", "fixture broker meter must be typed"
            )
        self._meter = meter if meter is not None else AccessMeter()

    @property
    def meter(self) -> AccessMeter:
        return self._meter

    @property
    def assurance(self) -> str:
        return self._enforcement.assurance

    def open_blob(
        self,
        *,
        path: str,
        access_class: str,
        required_capability: str | None,
        expected_sha256: str,
    ) -> bytes:
        if (
            not isinstance(access_class, str)
            or not access_class
            or (
                required_capability is not None
                and (
                    not isinstance(required_capability, str)
                    or not required_capability
                )
            )
            or not isinstance(expected_sha256, str)
        ):
            raise ExecutionBoundaryError(
                "SYNTHETIC_FIXTURE_REQUEST_INVALID", "fixture request is malformed"
            )
        canonical_path = _repository_path(path, "synthetic fixture path")
        entry = self._entries.get(canonical_path)
        if entry is None:
            raise ExecutionBoundaryError(
                "PATH_NOT_ALLOWLISTED", "synthetic fixture path is not allowlisted"
            )
        if access_class != entry.access_class:
            requested_row = access_class.endswith("_row_blob")
            frozen_row = entry.access_class.endswith("_row_blob")
            code = (
                "METADATA_AS_ROW_RELABEL_FORBIDDEN"
                if requested_row and not frozen_row
                else "ROW_AS_METADATA_RELABEL_FORBIDDEN"
                if frozen_row and not requested_row
                else "ACCESS_CLASS_MISMATCH"
            )
            raise ExecutionBoundaryError(code, "synthetic access-class relabel is forbidden")
        if required_capability != entry.required_capability:
            raise ExecutionBoundaryError(
                "CAPABILITY_MISMATCH", "synthetic fixture capability differs"
            )
        if expected_sha256 != entry.sha256:
            raise ExecutionBoundaryError(
                "INPUT_HASH_BINDING_MISMATCH", "synthetic expected hash differs"
            )
        lexical_target = assert_no_link_or_reparse(
            self._root / Path(canonical_path), label="synthetic fixture target"
        )
        target = lexical_target.resolve()
        if self._root not in target.parents or not target.is_file():
            raise ExecutionBoundaryError(
                "PATH_NOT_ALLOWLISTED", "synthetic fixture target escapes or is not regular"
            )
        try:
            payload = target.read_bytes()
        except OSError as exc:
            raise ExecutionBoundaryError(
                "SYNTHETIC_FIXTURE_READ_FAILED", "cannot read the synthetic fixture blob"
            ) from exc
        self._meter.synthetic_blobs_opened += 1
        if hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise ExecutionBoundaryError(
                "INPUT_HASH_MISMATCH", "synthetic bytes changed before caller delivery"
            )
        self._meter.bytes_delivered += len(payload)
        return payload


def create_exclusive_file(path: Path, payload: bytes) -> None:
    """Small shared O_EXCL primitive used only by synthetic supervisor seals."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        assert_no_link_or_reparse(path.parent, label="exclusive output parent")
        path.parent.mkdir(parents=True, exist_ok=True)
        assert_no_link_or_reparse(path, label="exclusive output path")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ExecutionBoundaryError(
            "OUTPUT_ROOT_NOT_FRESH", "exclusive synthetic seal path already exists"
        ) from exc


__all__ = [
    "AccessMeter",
    "BROKER_SCHEMA_VERSION",
    "BrokerReadRequest",
    "BrokerWriteRequest",
    "EnforcementStatus",
    "ExecutionBoundaryError",
    "OrdinaryRealDataExecutionBrokerV1",
    "RealHostEnforcementReport",
    "SYNTHETIC_ASSURANCE",
    "SyntheticFixtureBrokerV1",
    "SyntheticFixtureEnforcement",
    "SyntheticFixtureEntry",
    "build_real_data_execution_broker",
    "create_exclusive_file",
    "assert_no_link_or_reparse",
    "observe_real_host_enforcement",
]
