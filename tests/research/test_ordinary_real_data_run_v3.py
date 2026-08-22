from __future__ import annotations

import copy
import base64
import contextlib
import hashlib
import inspect
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SCRIPTS = REPO_ROOT / "scripts" / "research"
if str(RESEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS))

import ordinary_real_data_run_contract_v3 as contract
import scope_contract
import update_registry


V3 = "ordinary_real_data_run_v3"
EXP034_PROFILE = "exp034_input_canonicalization_v1"
EXP033_PROFILE = "exp033_leakfree_research_v1"
SYNTHETIC_PROFILE = "synthetic_governance_v1"


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def canonical_bytes(value: Any, *, trailing_lf: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if trailing_lf else b"")


class RowReadTrap:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        self.count += 1
        raise AssertionError("metadata-only preflight attempted a row/blob read")


class FakeGitHubProvider:
    repository = "kazuponbaseball-cell/keiba_ai_project"

    def __init__(self, registry_content: bytes) -> None:
        self.current_main = "c" * 40
        self.registry_content = registry_content
        self.comments: dict[int, dict[str, Any]] = {}
        self.approvers_content = json.dumps(
            {
                "schema_version": 1,
                "approvers": [{"login": "kazuponbaseball-cell"}],
                "denied_login_patterns": ["bot", "codex", "automation"],
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def get_repository(self, repository: str) -> dict[str, Any]:
        return {"full_name": self.repository, "default_branch": "main"}

    def get_branch_ref(self, repository: str, branch: str) -> dict[str, Any]:
        return {
            "ref": "refs/heads/main",
            "object": {"type": "commit", "sha": self.current_main},
        }

    def compare_commits(
        self, repository: str, base_commit: str, head_commit: str
    ) -> dict[str, Any]:
        return {
            "status": "identical" if base_commit == head_commit else "ahead",
            "url": (
                f"https://api.github.com/repos/{self.repository}/compare/"
                f"{base_commit}...{head_commit}"
            ),
            "base_commit": {"sha": base_commit},
            "merge_base_commit": {"sha": base_commit},
        }

    def get_file_contents(
        self, repository: str, path: str, ref: str
    ) -> dict[str, Any]:
        files = getattr(self, "files", {})
        if (path, ref) in files:
            content = files[(path, ref)]
            blob_sha = hashlib.sha1(content).hexdigest()
        elif path == "research/REGISTRY.jsonl":
            if ref != self.current_main:
                raise AssertionError(f"unexpected Registry ref: {ref}")
            content = self.registry_content
            blob_sha = "d" * 40
        elif path == "research/APPROVERS.json":
            content = self.approvers_content
            blob_sha = "e" * 40
        else:
            raise AssertionError(f"unexpected GitHub fixture path: {path}")
        return {
            "type": "file",
            "path": path,
            "encoding": "base64",
            "sha": blob_sha,
            "content": base64.b64encode(content).decode("ascii"),
        }

    def get_issue_comment(self, repository: str, comment_id: int) -> dict[str, Any]:
        return copy.deepcopy(self.comments[comment_id])

    def merge_registry(self, registry_content: bytes) -> None:
        self.registry_content = registry_content
        self.current_main = hashlib.sha256(registry_content).hexdigest()[:40]


class _LegacyV3Fixture(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.scope_counter = 0
        self.refs: dict[str, dict[str, str]] = {}
        for name in (
            "runner_universe",
            "training_source",
            "target_source",
            "feature_output_contract",
            "lineage_output_contract",
            "sealed_feature_release",
            "sealed_lineage_release",
            "producer_run_scope",
            "producer_execution_receipt",
            "producer_result_manifest",
            "label_eligibility",
            "fold",
            "dependency_lock",
        ):
            self._material(
                name,
                f"research/synthetic/v3/{name}.json",
                (json.dumps({"synthetic_fixture": name}, sort_keys=True) + "\n").encode(),
            )
        for name in (
            "runner_rows",
            "training_rows",
            "target_rows",
            "sealed_feature_rows",
            "sealed_lineage_rows",
        ):
            self._material(
                name,
                f"research/synthetic/v3/row_blobs/{name}.jsonl",
                (
                    json.dumps({"synthetic_row_blob": name}, sort_keys=True)
                    + "\n"
                ).encode(),
            )
        for key, relative in (
            ("code_exp034", "scripts/research/run_exp033_input_canonicalization_v0.py"),
            ("code_exp033", "scripts/research/run_leakfree_predraw_baseline_v0.py"),
            ("code_synthetic", "scripts/research/synthetic_v3_runner.py"),
        ):
            self._material(
                key,
                relative,
                f"# synthetic identity for {key}; never executed\n".encode(),
            )
        for key, relative in (
            (
                "config_exp034",
                "research/configs/EXP-20260821-034.input_canonicalization_v0.json",
            ),
            (
                "config_exp033",
                "research/configs/EXP-20260821-033.leakfree_predraw_baseline_v0.json",
            ),
            ("config_synthetic", "config/ordinary_real_data_v3.synthetic.example.json"),
        ):
            self._material(key, relative, b'{"synthetic_fixture":true}\n')

    def _material(self, key: str, relative: str, content: bytes) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.refs[key] = {
            "path": relative,
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    @staticmethod
    def experiment_id(profile_id: str) -> str:
        return "EXP-20260821-033" if profile_id == EXP033_PROFILE else "EXP-20260821-034"

    def proposal(self, profile_id: str = EXP034_PROFILE) -> dict[str, Any]:
        experiment_id = self.experiment_id(profile_id)
        return scope_contract.normalize_proposal_scope(
            scope_contract.strict_json_load(
                REPO_ROOT / "research/scopes" / f"{experiment_id}.proposal.json"
            ),
            expected_experiment_id=experiment_id,
        )

    @staticmethod
    def _profile(profile_id: str) -> dict[str, Any]:
        return {
            "profile_id": profile_id,
            "capabilities": copy.deepcopy(
                contract.CAPABILITY_PROFILES[profile_id]["capabilities"]
            ),
        }

    def _catalog_payload(self, profile_id: str) -> dict[str, Any]:
        capabilities = contract.CAPABILITY_PROFILES[profile_id]["capabilities"]
        phase_reads = sorted(
            name
            for name, enabled in capabilities.items()
            if enabled
            and name
            in {
                "read_real_input_manifests",
                "read_real_runner_rows",
                "read_historical_training_rows",
                "read_sealed_canonical_input_release",
            }
        )
        return {
            "catalog_id": "SYNTHETIC-CATALOG-001",
            "source_release_id": "SYNTHETIC-RELEASE-001",
            "row_count": 70,
            "race_count": 5,
            "runner_count": 70,
            "source_event_time_coverage": {"covered_count": 70, "total_count": 70},
            "received_at_coverage": {"covered_count": 70, "total_count": 70},
            "available_as_of_coverage": {"covered_count": 70, "total_count": 70},
            "max_source_event_time": "2026-08-22T14:55:00Z",
            "max_received_at": "2026-08-22T14:56:00Z",
            "max_available_as_of": "2026-08-22T14:57:00Z",
            "revoked": False,
            "revocation_status": "active",
            "runner_universe_digest": self.refs["runner_universe"]["sha256"],
            "target_date": "2026-08-23",
            "race_ids": [
                "202608230401",
                "202608230402",
                "202608230403",
                "202608230404",
                "202608230405",
            ],
            "phase_read_capabilities": phase_reads,
        }

    def catalog(self, profile_id: str) -> tuple[dict[str, Any], bytes]:
        payload = self._catalog_payload(profile_id)
        payload_bytes = canonical_bytes(payload, trailing_lf=True)
        self._material(
            "catalog_manifest",
            "research/synthetic/v3/input_catalog.payload.json",
            payload_bytes,
        )
        manifest = copy.deepcopy(self.refs["catalog_manifest"])
        wrapper = {
            **payload,
            "manifest": manifest,
            "attestation": {
                "kind": "sha256_bound",
                "content_sha256": manifest["sha256"],
                "signature_sha256": None,
                "signer_identity": None,
            },
        }
        return wrapper, payload_bytes

    def _binding(self, *, sealed: bool, lineage: bool = False) -> dict[str, Any]:
        prefix = "lineage" if lineage else "feature"
        artifact_role = (
            "canonical_feature_lineage"
            if lineage
            else "canonical_target_input_release"
        )
        return {
            "binding_role": "sealed_input_artifact" if sealed else "planned_output_contract",
            "artifact_role": artifact_role,
            "contract": copy.deepcopy(self.refs[f"{prefix}_output_contract"]),
            "producer_run_scope": (
                copy.deepcopy(self.refs["producer_run_scope"]) if sealed else None
            ),
            "producer_execution_receipt": (
                copy.deepcopy(self.refs["producer_execution_receipt"])
                if sealed
                else None
            ),
            "artifact_manifest": (
                copy.deepcopy(self.refs["producer_result_manifest"])
                if sealed
                else None
            ),
            "artifact_sha256": (
                self.refs[f"sealed_{prefix}_rows"]["sha256"] if sealed else None
            ),
        }

    def _read_allowlist(self, profile_id: str) -> list[dict[str, Any]]:
        metadata_names = [
            "catalog_manifest",
            "runner_universe",
            "training_source",
            "target_source",
            "feature_output_contract",
            "lineage_output_contract",
            "label_eligibility",
            "fold",
            "dependency_lock",
        ]
        if profile_id == EXP033_PROFILE:
            metadata_names.extend(
                [
                    "producer_run_scope",
                    "producer_execution_receipt",
                    "producer_result_manifest",
                ]
            )
        entries: dict[str, tuple[str, str, list[str]]] = {
            self.refs[name]["path"]: (
                self.refs[name]["sha256"],
                "metadata_manifest",
                ["metadata_preflight"],
            )
            for name in metadata_names
        }
        if profile_id == EXP034_PROFILE:
            entries.update(
                {
                    self.refs["runner_rows"]["path"]: (
                        self.refs["runner_rows"]["sha256"],
                        "row_blob",
                        ["read_runner_rows"],
                    ),
                    self.refs["training_rows"]["path"]: (
                        self.refs["training_rows"]["sha256"],
                        "row_blob",
                        ["canonicalize_input_release"],
                    ),
                    self.refs["target_rows"]["path"]: (
                        self.refs["target_rows"]["sha256"],
                        "row_blob",
                        ["canonicalize_input_release"],
                    ),
                }
            )
        elif profile_id == EXP033_PROFILE:
            entries.update(
                {
                    self.refs["sealed_feature_rows"]["path"]: (
                        self.refs["sealed_feature_rows"]["sha256"],
                        "row_blob",
                        ["read_sealed_input_release"],
                    ),
                    self.refs["sealed_lineage_rows"]["path"]: (
                        self.refs["sealed_lineage_rows"]["sha256"],
                        "row_blob",
                        ["read_sealed_input_release"],
                    ),
                    self.refs["training_rows"]["path"]: (
                        self.refs["training_rows"]["sha256"],
                        "row_blob",
                        ["read_sealed_input_release"],
                    ),
                }
            )
        return [
            {
                "path": path,
                "sha256": values[0],
                "access_class": values[1],
                "phases": values[2],
            }
            for path, values in sorted(entries.items())
        ]

    @staticmethod
    def _phase_io(
        profile_id: str,
        phase_id: str,
        refs: dict[str, dict[str, str]],
        output_root: str,
    ) -> tuple[list[str], list[str]]:
        reads: list[str] = []
        writes: list[str] = []
        if phase_id == "metadata_preflight":
            metadata_names = [
                "catalog_manifest",
                "runner_universe",
                "training_source",
                "target_source",
                "feature_output_contract",
                "lineage_output_contract",
                "label_eligibility",
                "fold",
                "dependency_lock",
            ]
            if profile_id == EXP033_PROFILE:
                metadata_names.extend(
                    [
                        "producer_run_scope",
                        "producer_execution_receipt",
                        "producer_result_manifest",
                    ]
                )
            reads = sorted(refs[name]["path"] for name in metadata_names)
        elif phase_id == "read_runner_rows":
            reads = [refs["runner_rows"]["path"]]
        elif phase_id == "canonicalize_input_release":
            reads = sorted(
                [refs["training_rows"]["path"], refs["target_rows"]["path"]]
            )
            writes = [f"{output_root}/work"]
        elif phase_id == "read_sealed_input_release":
            reads = sorted(
                [
                    refs["sealed_feature_rows"]["path"],
                    refs["sealed_lineage_rows"]["path"],
                    refs["training_rows"]["path"],
                ]
            )
        elif phase_id == "seal_research_outputs":
            writes = [f"{output_root}/sealed"]
        elif phase_id == "synthetic_fixture_validation" or (
            profile_id == EXP033_PROFILE
            and phase_id
            in {
                "train_research_model",
                "validate_research_model",
                "calibrate_research_model",
                "evaluate_outer_oos_once",
                "infer_target_runners",
            }
        ):
            writes = [f"{output_root}/work"]
        return reads, writes

    def scope(
        self,
        *,
        profile_id: str = EXP034_PROFILE,
        execution_kind: str | None = None,
    ) -> dict[str, Any]:
        proposal = self.proposal(profile_id)
        self.scope_counter += 1
        experiment_id = proposal["experiment_id"]
        policy = contract.CAPABILITY_PROFILES[profile_id]
        kind = execution_kind or policy["execution_kind"]
        catalog, _catalog_bytes = self.catalog(profile_id)
        output_root = (
            f"outputs/research/{experiment_id}/{profile_id}/"
            f"run-{self.scope_counter:03d}"
        )
        cwd = self.root.resolve().as_posix()
        # The contract intentionally targets the CI-supported 3.11/3.12 matrix;
        # this control-plane fixture never executes the frozen argv.
        interpreter = "C:/Python312/python.exe"
        profile_suffix = (
            "exp034"
            if profile_id == EXP034_PROFILE
            else ("exp033" if profile_id == EXP033_PROFILE else "synthetic")
        )
        code_ref = self.refs[f"code_{profile_suffix}"]
        config_ref = self.refs[f"config_{profile_suffix}"]
        data_refs = [
            copy.deepcopy(self.refs[name])
            for name in (
                "catalog_manifest",
                "runner_universe",
                "target_source",
                "training_source",
            )
        ]
        if profile_id == EXP033_PROFILE:
            data_refs.extend(
                [
                    copy.deepcopy(self.refs["producer_run_scope"]),
                    copy.deepcopy(self.refs["producer_execution_receipt"]),
                    copy.deepcopy(self.refs["producer_result_manifest"]),
                ]
            )
        data_refs.sort(key=lambda item: item["path"])
        read_allowlist = self._read_allowlist(profile_id)
        commands: list[dict[str, Any]] = []
        phases: list[dict[str, Any]] = []
        writes: dict[str, set[str]] = {}
        for index, (phase_id, capabilities) in enumerate(policy["phase_plan"], start=1):
            command_id = f"cmd-{index:02d}-{phase_id}"
            commands.append(
                {
                    "command_id": command_id,
                    "phase_id": phase_id,
                    "executable": interpreter,
                    "argv": [
                        interpreter,
                        "-B",
                        code_ref["path"],
                        "--phase",
                        phase_id,
                        "--config",
                        config_ref["path"],
                    ],
                    "working_directory": cwd,
                    "timeout_seconds": 60,
                }
            )
            read_paths, write_paths = self._phase_io(
                profile_id, phase_id, self.refs, output_root
            )
            for path in write_paths:
                writes.setdefault(path, set()).add(phase_id)
            phases.append(
                {
                    "phase_id": phase_id,
                    "required_capabilities": sorted(capabilities),
                    "command_id": command_id,
                    "read_paths": read_paths,
                    "write_paths": write_paths,
                }
            )
        write_allowlist = [
            {"path": path, "phases": sorted(phase_ids)}
            for path, phase_ids in sorted(writes.items())
        ]
        raw = {
            "run_scope_schema_version": V3,
            "proposal_scope": copy.deepcopy(proposal),
            "proposal_scope_digest": scope_contract.canonical_digest(proposal),
            "execution_kind": kind,
            "capability_profile": self._profile(profile_id),
            "execution_commit_sha": "a" * 40,
            "code_hashes": [copy.deepcopy(code_ref)],
            "config_hashes": [copy.deepcopy(config_ref)],
            "data_input_manifest_hashes": data_refs,
            "input_catalog": catalog,
            "runner_universe_manifest_hash": copy.deepcopy(self.refs["runner_universe"]),
            "feature_input_release_hash": self._binding(
                sealed=profile_id == EXP033_PROFILE
            ),
            "training_source_manifest_hash": copy.deepcopy(self.refs["training_source"]),
            "target_source_manifest_hash": copy.deepcopy(self.refs["target_source"]),
            "feature_lineage_manifest_hash": self._binding(
                sealed=profile_id == EXP033_PROFILE,
                lineage=True,
            ),
            "label_eligibility_contract_hash": copy.deepcopy(
                self.refs["label_eligibility"]
            ),
            "fold_manifest_hash": copy.deepcopy(self.refs["fold"]),
            "dependency_environment_lock_hash": copy.deepcopy(
                self.refs["dependency_lock"]
            ),
            "environment": {
                "interpreter_path": interpreter,
                "interpreter_version": "3.12.9",
                "dependency_versions": [
                    {
                        "name": "python",
                        "version": "3.12.9",
                    }
                ],
                "locale": "C.UTF-8",
                "timezone": "Asia/Tokyo",
            },
            "random_seed": 20260823,
            "repository_working_directory": cwd,
            "exact_commands": commands,
            "network_policy": {"mode": "disabled", "allowed_hosts": []},
            "read_allowlist": read_allowlist,
            "write_allowlist": write_allowlist,
            "output_root": output_root,
            "compute_budget": {
                "timeout_seconds": 900,
                "cpu_cores": 2,
                "memory_mib": 4096,
                "disk_mib": 4096,
                "max_model_fits": 1 if profile_id == EXP033_PROFILE else 0,
                "max_outer_oos_evaluations": 1 if profile_id == EXP033_PROFILE else 0,
                "max_target_inference_calls": 1 if profile_id == EXP033_PROFILE else 0,
            },
            "source_cutoff": "2026-08-22T15:00:00Z",
            "as_of": "2026-08-22T15:00:00Z",
            "phase_plan": phases,
            "output_sealing_contract": {
                "schema_version": contract.RESULT_MANIFEST_SCHEMA_VERSION,
                "mode": "append_only_immutable",
                "result_manifest_path": f"{output_root}/result.manifest.json",
                "failure_manifest_path": f"{output_root}/failure.manifest.json",
                "overwrite_allowed": False,
                "partial_output_consumer_eligible": False,
                "failure_manifest_required": True,
                "artifact_roles": sorted(
                    contract.ARTIFACT_ROLES_BY_PROFILE[profile_id]
                ),
                "required_result_fields": list(contract.REQUIRED_RESULT_FIELDS),
            },
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        return contract.normalize_ordinary_real_data_run_scope(
            raw, proposal_scope=proposal
        )

    @staticmethod
    def catalog_payload(scope: dict[str, Any]) -> dict[str, Any]:
        return {
            field: copy.deepcopy(scope["input_catalog"][field])
            for field in contract.CATALOG_PAYLOAD_FIELDS
        }

    @staticmethod
    def approval(keyword: str, approval_digest: str, comment_id: int) -> dict[str, Any]:
        body = f"{keyword} {approval_digest}"
        return {
            "approval_type": keyword,
            "approval_digest": approval_digest,
            "repository": "kazuponbaseball-cell/keiba_ai_project",
            "issue_number": 49,
            "comment_id": comment_id,
            "url": (
                "https://github.com/kazuponbaseball-cell/keiba_ai_project/"
                f"issues/49#issuecomment-{comment_id}"
            ),
            "author": "kazuponbaseball-cell",
            "author_type": "User",
            "body": body,
            "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "created_at": "2026-08-22T02:00:00Z",
            "updated_at": "2026-08-22T02:00:00Z",
        }

    def authority_case(self) -> dict[str, Any]:
        scope = self.scope()
        run_digest = contract.canonical_digest(scope)
        main_sha = "b" * 40
        event_id = "event-running-v3-001"
        preflight = contract.verify_metadata_preflight(
            scope, self.catalog_payload(scope)
        )
        prepare = self.approval(
            "APPROVED_TO_PREPARE", scope["proposal_scope_digest"], 1001
        )
        run = self.approval("APPROVED_TO_RUN", run_digest, 1002)
        event = {
            "event_id": event_id,
            "status": "running",
            "run_scope_digest": run_digest,
            "execution_kind": "real-data",
            "real_data_execution_allowed": True,
            "github_trust_evidence": {"verified_current_main_sha": main_sha},
            "human_prepare_approval_recorded": True,
            "human_run_approval_recorded": True,
            "execution_authorized": True,
            "revalidated_approval_evidence": [prepare, run],
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        registry_events = [
            {
                "event_id": "event-prepare-grant-001",
                "approval_evidence": prepare,
            },
            {
                "event_id": "event-run-grant-001",
                "approval_evidence": run,
            },
            event,
        ]
        registry_bytes = b"".join(
            canonical_bytes(item, trailing_lf=True) for item in registry_events
        )
        provider = FakeGitHubProvider(registry_bytes)
        provider.current_main = main_sha
        for evidence in (prepare, run):
            provider.comments[evidence["comment_id"]] = {
                "id": evidence["comment_id"],
                "html_url": evidence["url"],
                "issue_url": (
                    "https://api.github.com/repos/kazuponbaseball-cell/"
                    f"keiba_ai_project/issues/{evidence['issue_number']}"
                ),
                "user": {
                    "login": evidence["author"],
                    "type": evidence["author_type"],
                },
                "created_at": evidence["created_at"],
                "updated_at": evidence["updated_at"],
                "body": evidence["body"],
            }
        reservation_digest = contract.canonical_digest(
            {
                "run_scope_digest": run_digest,
                "verified_current_main_sha": main_sha,
                "output_root": scope["output_root"],
                "fresh": True,
            }
        )
        receipt = {
            "receipt_schema_version": contract.EXECUTION_RECEIPT_SCHEMA_VERSION,
            "experiment_id": "EXP-20260821-034",
            "running_event_id": event_id,
            "run_scope_digest": run_digest,
            "execution_kind": "real_data",
            "capability_profile_id": EXP034_PROFILE,
            "execution_commit_sha": scope["execution_commit_sha"],
            "verified_current_main_sha": main_sha,
            "execution_commit_compare_status": "ahead",
            "execution_commit_compare_url": (
                "https://api.github.com/repos/kazuponbaseball-cell/keiba_ai_project/"
                f"compare/{scope['execution_commit_sha']}...{main_sha}"
            ),
            "execution_commit_merge_base_sha": scope["execution_commit_sha"],
            "current_main_registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "prepare_approval_comment_id": 1001,
            "run_approval_comment_id": 1002,
            "metadata_preflight_digest": contract.canonical_digest(preflight),
            "capability_profile_digest": contract.canonical_digest(
                scope["capability_profile"]
            ),
            "input_manifest_hashes_digest": contract.canonical_digest(
                scope["data_input_manifest_hashes"]
            ),
            "environment_digest": contract.canonical_digest(scope["environment"]),
            "exact_commands_digest": contract.canonical_digest(scope["exact_commands"]),
            "read_allowlist_digest": contract.canonical_digest(scope["read_allowlist"]),
            "write_allowlist_digest": contract.canonical_digest(scope["write_allowlist"]),
            "output_root": scope["output_root"],
            "output_root_reservation_digest": reservation_digest,
            "output_root_was_fresh": True,
            "issued_at": "2026-08-22T15:01:00Z",
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        return {
            "scope": scope,
            "prepare": prepare,
            "run": run,
            "event": event,
            "receipt": receipt,
            "main_sha": main_sha,
            "preflight": preflight,
            "registry_bytes": registry_bytes,
            "provider": provider,
        }

    @staticmethod
    def authority_context(case: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "running",
            "cli_execution_kind": "real-data",
            "prepare_evidence": case["prepare"],
            "run_evidence": case["run"],
            "execution_commit": case["scope"]["execution_commit_sha"],
            "current_main_sha": case["main_sha"],
            "merged_running_event": case["event"],
            "current_main_registry_bytes": case["registry_bytes"],
            "execution_receipt": case["receipt"],
            "metadata_preflight_receipt": case["preflight"],
            "observed_environment": case["scope"]["environment"],
            "approval_provider": case["provider"],
        }

    def authorize(self, case: dict[str, Any], **overrides: Any) -> bool:
        context = self.authority_context(case)
        context["run_scope"] = case["scope"]
        context.update(overrides)
        return contract.verify_real_data_authorization(**context)


class V3Fixture(_LegacyV3Fixture):
    """Final frozen-v3 fixtures; every data row is synthetic and never preflight-read."""

    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.scope_counter = 0
        self.refs: dict[str, dict[str, str]] = {}
        self.race_ids = [f"20260823040{index}" for index in range(1, 6)]
        identities = [
            {"race_id": race_id, "horse_id": f"H{race_index:02d}{runner:02d}"}
            for race_index, race_id in enumerate(self.race_ids, start=1)
            for runner in range(1, 15)
        ]
        self.identity_digest = contract.canonical_digest(identities)
        identity_jsonl = b"".join(
            canonical_bytes(item, trailing_lf=True) for item in identities
        )
        for name in (
            "runner_rows",
            "target_rows",
            "sealed_feature_rows",
            "sealed_lineage_rows",
        ):
            self._material(
                name,
                f"research/synthetic/v3/row_blobs/{name}.jsonl",
                identity_jsonl,
            )
        self._material(
            "training_rows",
            "research/synthetic/v3/row_blobs/training_rows.jsonl",
            canonical_bytes(identities[0], trailing_lf=True),
        )
        for name in (
            "feature_output_contract",
            "lineage_output_contract",
            "label_eligibility",
            "fold",
        ):
            self._material(
                name,
                f"research/synthetic/v3/contracts/{name}.json",
                canonical_bytes({"synthetic_contract": name}, trailing_lf=True),
            )
        for key, relative in (
            ("code_exp034", "scripts/research/run_exp033_input_canonicalization_v0.py"),
            ("code_exp033", "scripts/research/run_leakfree_predraw_baseline_v0.py"),
            ("code_synthetic", "scripts/research/synthetic_v3_runner.py"),
        ):
            self._material(
                key,
                relative,
                f"# synthetic identity for {key}; never executed\n".encode(),
            )
        self._material(
            "authority_module",
            contract.AUTHORITY_MODULE_PATH,
            (REPO_ROOT / contract.AUTHORITY_MODULE_PATH).read_bytes(),
        )
        for key, relative in (
            (
                "config_exp034",
                "research/configs/EXP-20260821-034.input_canonicalization_v0.json",
            ),
            (
                "config_exp033",
                "research/configs/EXP-20260821-033.leakfree_predraw_baseline_v0.json",
            ),
            ("config_synthetic", "config/ordinary_real_data_v3.synthetic.example.json"),
        ):
            self._material(key, relative, canonical_bytes({"synthetic_fixture": True}, trailing_lf=True))
        for name in (
            "producer_run_scope",
            "producer_execution_receipt",
            "producer_output_attestation",
            "producer_result_manifest",
        ):
            self._material(
                name,
                f"research/synthetic/v3/producer/{name}.json",
                canonical_bytes({"synthetic_producer_envelope": name}, trailing_lf=True),
            )

    def _environment(self, profile_id: str) -> dict[str, Any]:
        dependencies = [{"name": "python", "version": "3.12.9"}]
        if profile_id == EXP033_PROFILE:
            dependencies = [
                {"name": "numpy", "version": "2.1.0"},
                {"name": "pandas", "version": "2.2.0"},
                {"name": "python", "version": "3.12.9"},
            ]
        return {
            "interpreter_path": "C:/Python312/python.exe",
            "interpreter_version": "3.12.9",
            "dependency_versions": dependencies,
            "locale": "C.UTF-8",
            "timezone": "Asia/Tokyo",
        }

    def _row_entries(self, profile_id: str) -> list[dict[str, Any]]:
        if profile_id == SYNTHETIC_PROFILE:
            return []
        if profile_id == EXP034_PROFILE:
            definitions = (
                ("runner_rows", "runner_row_blob", "read_real_runner_rows"),
                (
                    "target_rows",
                    "canonicalization_source_row_blob",
                    "canonicalize_input_release",
                ),
                (
                    "training_rows",
                    "historical_training_row_blob",
                    "read_historical_training_rows",
                ),
            )
            phase_id = "canonicalize_input_release"
        else:
            definitions = (
                ("runner_rows", "runner_row_blob", "read_real_runner_rows"),
                (
                    "training_rows",
                    "historical_training_row_blob",
                    "read_historical_training_rows",
                ),
                (
                    "sealed_feature_rows",
                    "sealed_input_row_blob",
                    "read_sealed_canonical_input_release",
                ),
                (
                    "sealed_lineage_rows",
                    "sealed_input_row_blob",
                    "read_sealed_canonical_input_release",
                ),
            )
            phase_id = "execute_research_plan"
        return sorted(
            (
                {
                    **copy.deepcopy(self.refs[name]),
                    "access_class": access_class,
                    "required_capability": capability,
                    "phases": [phase_id],
                }
                for name, access_class, capability in definitions
            ),
            key=lambda item: item["path"],
        )

    def _input_manifest(
        self,
        *,
        name: str,
        kind: str,
        row_count: int,
        race_count: int,
        runner_count: int,
        row_refs: list[dict[str, Any]],
        identity_digest: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": contract.INPUT_MANIFEST_SCHEMA_VERSION,
            "manifest_id": f"SYNTHETIC-{name.upper().replace('_', '-')}-001",
            "manifest_kind": kind,
            "source_release_id": "SYNTHETIC-RELEASE-001",
            "row_count": row_count,
            "race_count": race_count,
            "runner_count": runner_count,
            "source_event_time_coverage": {
                "covered_count": row_count,
                "total_count": row_count,
            },
            "received_at_coverage": {
                "covered_count": row_count,
                "total_count": row_count,
            },
            "available_as_of_coverage": {
                "covered_count": row_count,
                "total_count": row_count,
            },
            "max_source_event_time": "2026-08-22T14:55:00Z",
            "max_received_at": "2026-08-22T14:56:00Z",
            "max_available_as_of": "2026-08-22T14:57:00Z",
            "revoked": False,
            "revocation_status": "active",
            "row_blob_refs": row_refs,
            "identity_set_sha256": identity_digest,
        }

    def _prepare_profile_materials(
        self, profile_id: str, environment: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rows = self._row_entries(profile_id)
        by_access = {item["access_class"]: item for item in rows}
        empty_digest = contract.canonical_digest([])
        if profile_id == SYNTHETIC_PROFILE:
            specs = (
                ("runner_universe", "synthetic_fixture", 0, 0, 0, []),
                ("training_source", "synthetic_fixture", 0, 0, 0, []),
                ("target_source", "synthetic_fixture", 0, 0, 0, []),
            )
            identity = empty_digest
        elif profile_id == EXP034_PROFILE:
            specs = (
                ("runner_universe", "runner_universe", 20, 5, 70, [by_access["runner_row_blob"]]),
                ("training_source", "training_source", 20, 5, 70, [by_access["historical_training_row_blob"]]),
                ("target_source", "target_source", 30, 5, 70, [by_access["canonicalization_source_row_blob"]]),
            )
            identity = self.identity_digest
        else:
            sealed = [item for item in rows if item["access_class"] == "sealed_input_row_blob"]
            specs = (
                ("runner_universe", "runner_universe", 20, 5, 70, [by_access["runner_row_blob"]]),
                ("training_source", "training_source", 20, 5, 70, [by_access["historical_training_row_blob"]]),
                ("target_source", "target_source", 30, 5, 70, sealed),
            )
            identity = self.identity_digest
        for name, kind, row_count, race_count, runner_count, row_refs in specs:
            manifest = self._input_manifest(
                name=name,
                kind=kind,
                row_count=row_count,
                race_count=race_count,
                runner_count=runner_count,
                row_refs=copy.deepcopy(row_refs),
                identity_digest=identity,
            )
            self._material(
                name,
                f"research/synthetic/v3/manifests/{name}.json",
                canonical_bytes(manifest, trailing_lf=True),
            )
        self._material(
            "dependency_lock",
            "research/synthetic/v3/contracts/dependency_lock.json",
            canonical_bytes(
                {
                    "schema_version": contract.ENVIRONMENT_LOCK_SCHEMA_VERSION,
                    "environment": environment,
                },
                trailing_lf=True,
            ),
        )
        return rows

    def _binding(self, *, sealed: bool, lineage: bool = False) -> dict[str, Any]:
        prefix = "lineage" if lineage else "feature"
        return {
            "binding_role": "sealed_input_artifact" if sealed else "planned_output_contract",
            "artifact_role": (
                "canonical_feature_lineage"
                if lineage
                else "canonical_target_input_release"
            ),
            "contract": copy.deepcopy(self.refs[f"{prefix}_output_contract"]),
            "producer_run_scope": copy.deepcopy(self.refs["producer_run_scope"]) if sealed else None,
            "producer_execution_receipt": copy.deepcopy(self.refs["producer_execution_receipt"]) if sealed else None,
            "producer_output_attestation": copy.deepcopy(self.refs["producer_output_attestation"]) if sealed else None,
            "artifact_manifest": copy.deepcopy(self.refs["producer_result_manifest"]) if sealed else None,
            "artifact_sha256": self.refs[f"sealed_{prefix}_rows"]["sha256"] if sealed else None,
        }

    @staticmethod
    def _artifact_paths(profile_id: str, output_root: str) -> list[dict[str, str]]:
        extension = {
            "canonical_json": ".json",
            "identity_jsonl": ".jsonl",
            "opaque_binary": ".bin",
        }
        return [
            {
                "role": role,
                "path": (
                    f"{output_root}/artifacts/{role}"
                    f"{extension[contract.ARTIFACT_FORMAT_BY_ROLE[role]]}"
                ),
            }
            for role in sorted(contract.ARTIFACT_ROLES_BY_PROFILE[profile_id])
        ]

    def scope(
        self,
        *,
        profile_id: str = EXP034_PROFILE,
        execution_kind: str | None = None,
    ) -> dict[str, Any]:
        self.scope_counter += 1
        proposal = self.proposal(profile_id)
        experiment_id = proposal["experiment_id"]
        policy = contract.CAPABILITY_PROFILES[profile_id]
        kind = execution_kind or policy["execution_kind"]
        environment = self._environment(profile_id)
        row_entries = self._prepare_profile_materials(profile_id, environment)
        output_root = (
            f"outputs/research/{experiment_id}/{profile_id}/"
            f"run-{self.scope_counter:03d}"
        )
        suffix = "exp034" if profile_id == EXP034_PROFILE else (
            "exp033" if profile_id == EXP033_PROFILE else "synthetic"
        )
        runner_ref = self.refs[f"code_{suffix}"]
        config_ref = self.refs[f"config_{suffix}"]
        feature_binding = self._binding(sealed=profile_id == EXP033_PROFILE)
        lineage_binding = self._binding(
            sealed=profile_id == EXP033_PROFILE, lineage=True
        )
        data_refs = sorted(
            [copy.deepcopy(self.refs[name]) for name in ("runner_universe", "training_source", "target_source")],
            key=lambda item: item["path"],
        )
        metadata_refs = [
            *data_refs,
            copy.deepcopy(self.refs["label_eligibility"]),
            copy.deepcopy(self.refs["fold"]),
            copy.deepcopy(self.refs["dependency_lock"]),
            copy.deepcopy(feature_binding["contract"]),
            copy.deepcopy(lineage_binding["contract"]),
        ]
        if profile_id == EXP033_PROFILE:
            for binding in (feature_binding, lineage_binding):
                metadata_refs.extend(
                    copy.deepcopy(binding[field])
                    for field in (
                        "producer_run_scope",
                        "producer_execution_receipt",
                        "producer_output_attestation",
                        "artifact_manifest",
                    )
                )
        metadata_by_path = {item["path"]: item for item in metadata_refs}
        metadata_capability = None if profile_id == SYNTHETIC_PROFILE else "read_real_input_manifests"
        metadata_entries_without_catalog = [
            {
                **copy.deepcopy(metadata_by_path[path]),
                "access_class": "metadata_manifest",
                "required_capability": metadata_capability,
                "phases": ["metadata_preflight"],
            }
            for path in sorted(metadata_by_path)
        ]
        synthetic = profile_id == SYNTHETIC_PROFILE
        row_count = 0 if synthetic else 70
        race_count = 0 if synthetic else 5
        runner_count = 0 if synthetic else 70
        payload = {
            "catalog_id": "SYNTHETIC-CATALOG-001",
            "source_release_id": "SYNTHETIC-RELEASE-001",
            "row_count": row_count,
            "race_count": race_count,
            "runner_count": runner_count,
            "source_event_time_coverage": {"covered_count": row_count, "total_count": row_count},
            "received_at_coverage": {"covered_count": row_count, "total_count": row_count},
            "available_as_of_coverage": {"covered_count": row_count, "total_count": row_count},
            "max_source_event_time": "2026-08-22T14:55:00Z",
            "max_received_at": "2026-08-22T14:56:00Z",
            "max_available_as_of": "2026-08-22T14:57:00Z",
            "revoked": False,
            "revocation_status": "active",
            "runner_universe_digest": self.refs["runner_universe"]["sha256"],
            "runner_identity_digest": contract.canonical_digest([]) if synthetic else self.identity_digest,
            "target_date": "2026-08-23",
            "race_ids": [] if synthetic else self.race_ids,
            "phase_read_capabilities": sorted(
                name
                for name, enabled in policy["capabilities"].items()
                if enabled
                and name in {
                    "read_real_input_manifests",
                    "read_real_runner_rows",
                    "canonicalize_input_release",
                    "read_historical_training_rows",
                    "read_sealed_canonical_input_release",
                }
            ),
            "metadata_manifest_refs": metadata_entries_without_catalog,
            "row_blob_refs": copy.deepcopy(row_entries),
        }
        payload_bytes = canonical_bytes(payload, trailing_lf=True)
        self._material(
            "catalog_manifest",
            "research/synthetic/v3/input_catalog.payload.json",
            payload_bytes,
        )
        catalog_entry = {
            **copy.deepcopy(self.refs["catalog_manifest"]),
            "access_class": "metadata_manifest",
            "required_capability": metadata_capability,
            "phases": ["metadata_preflight"],
        }
        read_allowlist = sorted(
            [catalog_entry, *metadata_entries_without_catalog, *copy.deepcopy(row_entries)],
            key=lambda item: item["path"],
        )
        catalog = {
            **payload,
            "manifest": copy.deepcopy(self.refs["catalog_manifest"]),
            "attestation": {
                "kind": "sha256_bound",
                "content_sha256": self.refs["catalog_manifest"]["sha256"],
                "signature_sha256": None,
                "signer_identity": None,
            },
        }
        artifact_paths = self._artifact_paths(profile_id, output_root)
        execution_phase = next(
            (
                phase_id
                for phase_id, _ in policy["phase_plan"]
                if phase_id not in {"metadata_preflight", "seal_research_outputs"}
            ),
            None,
        )
        write_allowlist: list[dict[str, Any]] = []
        if execution_phase is not None:
            write_allowlist = [
                {
                    "path": item["path"],
                    "required_capability": "write_research_outputs",
                    "phases": [execution_phase],
                }
                for item in artifact_paths
            ]
            write_allowlist.extend(
                {
                    "path": f"{output_root}/{name}",
                    "required_capability": "write_research_outputs",
                    "phases": ["seal_research_outputs"],
                }
                for name in ("failure.manifest.json", "result.manifest.json")
            )
            write_allowlist.sort(key=lambda item: item["path"])
        cwd = self.root.resolve().as_posix()
        commands: list[dict[str, Any]] = []
        phases: list[dict[str, Any]] = []
        metadata_paths = sorted(item["path"] for item in read_allowlist if item["access_class"] == "metadata_manifest")
        row_paths = sorted(item["path"] for item in row_entries)
        artifact_output_paths = sorted(item["path"] for item in artifact_paths)
        manifest_output_paths = sorted(
            [f"{output_root}/failure.manifest.json", f"{output_root}/result.manifest.json"]
        )
        for index, (phase_id, capabilities) in enumerate(policy["phase_plan"], start=1):
            command_id = f"cmd-{index:02d}-{phase_id}"
            commands.append(
                {
                    "command_id": command_id,
                    "phase_id": phase_id,
                    "executable": environment["interpreter_path"],
                    "argv": [
                        environment["interpreter_path"],
                        "-I",
                        "-B",
                        runner_ref["path"],
                        "--phase",
                        phase_id,
                        "--config",
                        config_ref["path"],
                    ],
                    "working_directory": cwd,
                    "timeout_seconds": 60,
                }
            )
            reads = metadata_paths if phase_id == "metadata_preflight" else (
                row_paths if phase_id == execution_phase else []
            )
            writes = artifact_output_paths if phase_id == execution_phase else (
                manifest_output_paths if phase_id == "seal_research_outputs" else []
            )
            phases.append(
                {
                    "phase_id": phase_id,
                    "required_capabilities": sorted(capabilities),
                    "command_id": command_id,
                    "read_paths": reads,
                    "write_paths": writes,
                }
            )
        raw = {
            "run_scope_schema_version": V3,
            "proposal_scope": copy.deepcopy(proposal),
            "proposal_scope_digest": scope_contract.canonical_digest(proposal),
            "execution_kind": kind,
            "capability_profile": self._profile(profile_id),
            "execution_commit_sha": "a" * 40,
            "code_hashes": sorted(
                [copy.deepcopy(self.refs["authority_module"]), copy.deepcopy(runner_ref)],
                key=lambda item: item["path"],
            ),
            "config_hashes": [copy.deepcopy(config_ref)],
            "data_input_manifest_hashes": data_refs,
            "input_catalog": catalog,
            "runner_universe_manifest_hash": copy.deepcopy(self.refs["runner_universe"]),
            "feature_input_release_hash": feature_binding,
            "training_source_manifest_hash": copy.deepcopy(self.refs["training_source"]),
            "target_source_manifest_hash": copy.deepcopy(self.refs["target_source"]),
            "feature_lineage_manifest_hash": lineage_binding,
            "label_eligibility_contract_hash": copy.deepcopy(self.refs["label_eligibility"]),
            "fold_manifest_hash": copy.deepcopy(self.refs["fold"]),
            "dependency_environment_lock_hash": copy.deepcopy(self.refs["dependency_lock"]),
            "environment": environment,
            "random_seed": 20260823,
            "repository_working_directory": cwd,
            "exact_commands": commands,
            "network_policy": {"mode": "disabled", "allowed_hosts": []},
            "read_allowlist": read_allowlist,
            "write_allowlist": write_allowlist,
            "output_root": output_root,
            "compute_budget": {
                "timeout_seconds": 900,
                "cpu_cores": 2,
                "memory_mib": 4096,
                "disk_mib": 4096,
                "max_model_fits": 1 if profile_id == EXP033_PROFILE else 0,
                "max_outer_oos_evaluations": 1 if profile_id == EXP033_PROFILE else 0,
                "max_target_inference_calls": 1 if profile_id == EXP033_PROFILE else 0,
            },
            "source_cutoff": "2026-08-22T15:00:00Z",
            "as_of": "2026-08-22T15:00:00Z",
            "phase_plan": phases,
            "output_sealing_contract": {
                "schema_version": contract.RESULT_MANIFEST_SCHEMA_VERSION,
                "mode": "append_only_immutable",
                "execution_receipt_path": f"{output_root}/execution_receipt.json",
                "result_manifest_path": f"{output_root}/result.manifest.json",
                "failure_manifest_path": f"{output_root}/failure.manifest.json",
                "overwrite_allowed": False,
                "partial_output_consumer_eligible": False,
                "failure_manifest_required": True,
                "artifact_roles": sorted(contract.ARTIFACT_ROLES_BY_PROFILE[profile_id]),
                "artifact_paths": artifact_paths,
                "required_result_fields": list(contract.REQUIRED_RESULT_FIELDS),
            },
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        return contract.normalize_ordinary_real_data_run_scope(raw, proposal_scope=proposal)

    @staticmethod
    def catalog_payload(scope: dict[str, Any]) -> dict[str, Any]:
        return {
            field: copy.deepcopy(scope["input_catalog"][field])
            for field in contract.CATALOG_PAYLOAD_FIELDS
        }

    def authority_case(self, *, persist_receipt: bool = True) -> dict[str, Any]:
        scope = self.scope()
        run_digest = contract.canonical_digest(scope)
        event_main = "b" * 40
        live_main = "c" * 40
        preflight = contract.verify_metadata_preflight(scope, self.catalog_payload(scope))
        prepare = self.approval("APPROVED_TO_PREPARE", scope["proposal_scope_digest"], 1001)
        run = self.approval("APPROVED_TO_RUN", run_digest, 1002)
        statuses = (
            "proposed",
            "approved_to_prepare",
            "preparing",
            "run_approval_required",
            "approved_to_run",
            "running",
        )
        events: list[dict[str, Any]] = []
        for sequence, status in enumerate(statuses, start=1):
            event: dict[str, Any] = {
                "event_id": f"event-{sequence:02d}-{status}",
                "experiment_id": "EXP-20260821-034",
                "status": status,
                "sequence": sequence,
                "previous_event_id": events[-1]["event_id"] if events else None,
                "previous_status": events[-1]["status"] if events else None,
            }
            if status == "approved_to_prepare":
                event["proposal_scope_digest"] = scope["proposal_scope_digest"]
                event["approval_evidence"] = prepare
            if status == "approved_to_run":
                event["run_scope_digest"] = run_digest
                event["approval_evidence"] = run
            events.append(event)
        running = events[-1]
        running.update(
            {
                "run_scope_digest": run_digest,
                "run_scope_file": "research/scopes/EXP-20260821-034.run.json",
                "execution_kind": "real-data",
                "real_data_execution_allowed": False,
                "automatic_execution_allowed": False,
                "github_trust_evidence": {"verified_current_main_sha": event_main},
                "human_prepare_approval_recorded": True,
                "human_run_approval_recorded": True,
                "execution_authorized": False,
                "revalidated_approval_evidence": [prepare, run],
                "formal_buy": False,
                "send_order": False,
                "stake": 0,
            }
        )
        registry_bytes = b"".join(canonical_bytes(item, trailing_lf=True) for item in events)
        provider = FakeGitHubProvider(registry_bytes)
        provider.current_main = live_main
        provider.files = {
            ("research/REGISTRY.jsonl", live_main): registry_bytes,
            ("research/scopes/EXP-20260821-034.run.json", live_main): canonical_bytes(scope, trailing_lf=True),
            ("research/scopes/EXP-20260821-034.proposal.json", live_main): canonical_bytes(scope["proposal_scope"], trailing_lf=True),
        }
        for evidence in (prepare, run):
            provider.comments[evidence["comment_id"]] = {
                "id": evidence["comment_id"],
                "html_url": evidence["url"],
                "issue_url": f"https://api.github.com/repos/kazuponbaseball-cell/keiba_ai_project/issues/{evidence['issue_number']}",
                "user": {"login": evidence["author"], "type": evidence["author_type"]},
                "created_at": evidence["created_at"],
                "updated_at": evidence["updated_at"],
                "body": evidence["body"],
            }
        reservation_digest = contract.canonical_digest(
            {
                "run_scope_digest": run_digest,
                "verified_current_main_sha": live_main,
                "output_root": scope["output_root"],
                "fresh": True,
            }
        )
        receipt = {
            "receipt_schema_version": contract.EXECUTION_RECEIPT_SCHEMA_VERSION,
            "experiment_id": "EXP-20260821-034",
            "running_event_id": running["event_id"],
            "run_scope_digest": run_digest,
            "execution_kind": "real_data",
            "capability_profile_id": EXP034_PROFILE,
            "execution_commit_sha": scope["execution_commit_sha"],
            "verified_current_main_sha": live_main,
            "execution_commit_compare_status": "ahead",
            "execution_commit_compare_url": f"https://api.github.com/repos/kazuponbaseball-cell/keiba_ai_project/compare/{scope['execution_commit_sha']}...{live_main}",
            "execution_commit_merge_base_sha": scope["execution_commit_sha"],
            "current_main_registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "prepare_approval_comment_id": 1001,
            "run_approval_comment_id": 1002,
            "metadata_preflight_digest": contract.canonical_digest(preflight),
            "capability_profile_digest": contract.canonical_digest(scope["capability_profile"]),
            "input_manifest_hashes_digest": contract.canonical_digest(scope["data_input_manifest_hashes"]),
            "environment_digest": contract.canonical_digest(scope["environment"]),
            "exact_commands_digest": contract.canonical_digest(scope["exact_commands"]),
            "read_allowlist_digest": contract.canonical_digest(scope["read_allowlist"]),
            "write_allowlist_digest": contract.canonical_digest(scope["write_allowlist"]),
            "output_root": scope["output_root"],
            "output_root_reservation_digest": reservation_digest,
            "output_root_was_fresh": True,
            "real_data_execution_allowed": False,
            "execution_authorized": False,
            "issued_at": "2026-08-22T15:01:00Z",
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        if persist_receipt:
            receipt_path = self.root / scope["output_sealing_contract"]["execution_receipt_path"]
            receipt_path.parent.mkdir(parents=True, exist_ok=False)
            receipt_path.write_bytes(canonical_bytes(receipt, trailing_lf=True))
        return {
            "scope": scope,
            "prepare": prepare,
            "run": run,
            "event": running,
            "receipt": receipt,
            "main_sha": live_main,
            "event_main_sha": event_main,
            "preflight": preflight,
            "registry_bytes": registry_bytes,
            "provider": provider,
        }

    @staticmethod
    def authority_context(case: dict[str, Any]) -> dict[str, Any]:
        return {
            "root": case.get("root") or Path(case["scope"]["repository_working_directory"]),
            "status": "running",
            "cli_execution_kind": "real-data",
            "prepare_evidence": case["prepare"],
            "run_evidence": case["run"],
            "execution_commit": case["scope"]["execution_commit_sha"],
            "current_main_sha": case["main_sha"],
            "merged_running_event": case["event"],
            "current_main_registry_bytes": case["registry_bytes"],
            "execution_receipt": case["receipt"],
            "metadata_preflight_receipt": case["preflight"],
            "observed_environment": case["scope"]["environment"],
        }

    def authorize(self, case: dict[str, Any], **overrides: Any) -> bool:
        phase_id = (
            "canonicalize_input_release"
            if any(
                item["phase_id"] == "canonicalize_input_release"
                for item in case["scope"]["exact_commands"]
            )
            else case["scope"]["exact_commands"][0]["phase_id"]
        )
        command = next(item for item in case["scope"]["exact_commands"] if item["phase_id"] == phase_id)
        context = self.authority_context(case)
        context.update(
            {
                "run_scope": case["scope"],
                "phase_id": phase_id,
                "observed_argv": command["argv"],
            }
        )
        context.update(overrides)
        with mock.patch.object(contract, "GitHubRestApprovalProvider", return_value=case["provider"]):
            return contract.verify_real_data_authorization(**context)


class CompatibilityTests(V3Fixture):
    def test_01_v2_digests_validation_and_fields_are_unchanged(self) -> None:
        self.assertEqual(
            scope_contract.RUN_FIELDS,
            {
                "proposal_scope",
                "proposal_scope_digest",
                "execution_commit_sha",
                "config_hashes",
                "data_input_manifest_hashes",
                "fold_manifest_hash",
                "runner_universe_manifest_hash",
                "dependency_environment_manifest",
                "seed",
                "exact_execution_commands",
                "formal_buy",
                "send_order",
                "stake",
            },
        )
        expected = {
            "EXP-20260808-030": (
                "890a242b6a14485e233473c96342cfbe66ff3f09178f546a50f2d37f93ab3610",
                "8ef37a63b165c7d1b41b65a3a331c311bfe5957b659921317ca1128d2322bd31",
            ),
            "EXP-20260808-031": (
                "3f97b3d9c57a79ebbcc91746d6e5f27a37253395095019658d49aa5389672410",
                "5aac4066bd3839509990369998568bb8d30df77b2391e11cbf141f980c837804",
            ),
        }
        for experiment_id, (proposal_digest, run_digest) in expected.items():
            with self.subTest(experiment_id=experiment_id):
                proposal = scope_contract.normalize_proposal_scope(
                    scope_contract.strict_json_load(
                        REPO_ROOT / "research/scopes" / f"{experiment_id}.proposal.json"
                    ),
                    expected_experiment_id=experiment_id,
                )
                run = scope_contract.normalize_run_scope(
                    scope_contract.strict_json_load(
                        REPO_ROOT / "research/scopes" / f"{experiment_id}.run.json"
                    ),
                    proposal_scope=proposal,
                )
                version, dispatched = contract.dispatch_ordinary_run_scope(
                    run, proposal_scope=proposal
                )
                self.assertEqual(version, "legacy_v2")
                self.assertEqual(dispatched, run)
                self.assertEqual(scope_contract.canonical_digest(proposal), proposal_digest)
                self.assertEqual(scope_contract.canonical_digest(run), run_digest)

    def test_02_v2_scope_cannot_request_real_data(self) -> None:
        proposal = scope_contract.normalize_proposal_scope(
            scope_contract.strict_json_load(
                REPO_ROOT / "research/scopes/EXP-20260808-030.proposal.json"
            ),
            expected_experiment_id="EXP-20260808-030",
        )
        run = scope_contract.normalize_run_scope(
            scope_contract.strict_json_load(
                REPO_ROOT / "research/scopes/EXP-20260808-030.run.json"
            ),
            proposal_scope=proposal,
        )
        self.assertFalse(
            contract.verify_real_data_authorization(
                status="running",
                run_scope=run,
                cli_execution_kind="real-data",
                prepare_evidence=None,
                run_evidence=None,
                execution_commit=run["execution_commit_sha"],
                current_main_sha="b" * 40,
            )
        )

    def test_22_unknown_null_or_case_changed_version_is_rejected(self) -> None:
        for version in (None, "ordinary_real_data_run_v4", "ORDINARY_REAL_DATA_RUN_V3"):
            with self.subTest(version=version):
                value = self.scope()
                value["run_scope_schema_version"] = version
                with self.assertRaisesRegex(ValueError, "unknown|schema"):
                    contract.dispatch_ordinary_run_scope(
                        value, proposal_scope=self.proposal()
                    )

    def test_23_legacy_registry_events_and_bytes_are_unchanged(self) -> None:
        registry = REPO_ROOT / "research/REGISTRY.jsonl"
        before = registry.read_bytes()
        events = update_registry.load_events(registry)
        self.assertTrue(events)
        self.assertFalse(any("run_scope_schema_version" in event for event in events))
        self.assertTrue(all(event["schema_version"] in {2, 3} for event in events))
        self.assertEqual(registry.read_bytes(), before)


class StrictScopeAndCapabilityTests(V3Fixture):
    def test_json_schema_required_fields_match_frozen_python_contract(self) -> None:
        schema = json.loads(
            (REPO_ROOT / "research/schemas/ordinary_real_data_run_v3.schema.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(schema["required"], list(contract.RUN_FIELDS))
        definitions = schema["$defs"]
        self.assertEqual(
            definitions["capabilities"]["required"],
            list(contract.CAPABILITY_FIELDS),
        )
        self.assertEqual(
            definitions["inputCatalog"]["required"],
            list(contract.CATALOG_FIELDS),
        )
        self.assertEqual(
            definitions["artifactBinding"]["required"],
            list(contract.ARTIFACT_BINDING_FIELDS),
        )
        self.assertEqual(
            definitions["readAllow"]["required"],
            list(contract.READ_ALLOW_FIELDS),
        )
        self.assertEqual(
            definitions["writeAllow"]["required"],
            list(contract.WRITE_ALLOW_FIELDS),
        )
        self.assertEqual(
            definitions["outputContract"]["required"],
            list(contract.OUTPUT_CONTRACT_FIELDS),
        )

    def test_strict_json_duplicate_keys_nan_and_infinity_are_rejected(self) -> None:
        for raw, message in (
            ('{"a":1,"a":2}', "duplicate"),
            ('{"value":NaN}', "constant|finite|NaN"),
            ('{"value":Infinity}', "constant|finite|Infinity"),
        ):
            with self.subTest(raw=raw):
                path = self.root / "invalid.json"
                path.write_text(raw, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    contract.strict_json_load(path)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                contract.canonical_digest({"value": value})

    def test_v3_exact_fields_dispatch_and_canonical_round_trip(self) -> None:
        scope = self.scope()
        self.assertEqual(tuple(scope), contract.RUN_FIELDS)
        version, observed = contract.dispatch_ordinary_run_scope(
            scope, proposal_scope=self.proposal()
        )
        self.assertEqual(version, V3)
        self.assertEqual(observed, scope)

    def test_extra_field_free_shell_and_path_traversal_fail_closed(self) -> None:
        scope = self.scope()
        changed = copy.deepcopy(scope)
        changed["late_override"] = True
        with self.assertRaisesRegex(ValueError, "fields|extra"):
            contract.normalize_ordinary_real_data_run_scope(
                changed, proposal_scope=self.proposal()
            )
        changed = copy.deepcopy(scope)
        changed["exact_commands"][0]["argv"] = [
            changed["environment"]["interpreter_path"],
            "-c",
            "print('unapproved')",
        ]
        with self.assertRaisesRegex(ValueError, "free.form|command"):
            contract.normalize_ordinary_real_data_run_scope(
                changed, proposal_scope=self.proposal()
            )
        for unsafe in (
            "../outside",
            "outputs/research/EXP-20260821-034/../outside",
            "outputs\\research\\mutable",
            "/absolute/output",
            "outputs/research/EXP-20260821-034/latest/run",
        ):
            with self.subTest(unsafe=unsafe):
                changed = copy.deepcopy(scope)
                changed["output_root"] = unsafe
                with self.assertRaisesRegex(ValueError, "path|root|mutable|canonical"):
                    contract.normalize_ordinary_real_data_run_scope(
                        changed, proposal_scope=self.proposal()
                    )

    def test_profiles_are_finite_and_always_forbidden_capabilities_are_false(self) -> None:
        self.assertEqual(
            set(contract.CAPABILITY_PROFILES),
            {SYNTHETIC_PROFILE, EXP034_PROFILE, EXP033_PROFILE},
        )
        for profile_id, policy in contract.CAPABILITY_PROFILES.items():
            with self.subTest(profile_id=profile_id):
                capabilities = policy["capabilities"]
                self.assertEqual(set(capabilities), set(contract.CAPABILITY_FIELDS))
                self.assertTrue(
                    all(
                        capabilities[field] is False
                        for field in contract.ALWAYS_FALSE_CAPABILITIES
                    )
                )

    def test_19_buy_order_notification_and_nonzero_stake_are_rejected(self) -> None:
        mutations = (
            lambda value: value.__setitem__("formal_buy", True),
            lambda value: value.__setitem__("send_order", True),
            lambda value: value.__setitem__("stake", 1),
            lambda value: value["capability_profile"]["capabilities"].__setitem__(
                "notification", True
            ),
            lambda value: value["capability_profile"]["capabilities"].__setitem__(
                "order", True
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(self.scope())
                mutate(changed)
                with self.assertRaisesRegex(
                    ValueError, "formal_buy|send_order|stake|profile|capability|production"
                ):
                    contract.normalize_ordinary_real_data_run_scope(
                        changed, proposal_scope=self.proposal()
                    )

    def test_20_exp034_profile_rejects_training_validation_oos_and_inference(self) -> None:
        profile = self._profile(EXP034_PROFILE)
        for capability in (
            "train_research_model",
            "validate_research_model",
            "calibrate_research_model",
            "evaluate_outer_oos_once",
            "infer_target_runners",
        ):
            with self.subTest(capability=capability):
                self.assertFalse(profile["capabilities"][capability])
                changed = copy.deepcopy(profile)
                changed["capabilities"][capability] = True
                with self.assertRaisesRegex(ValueError, "profile|finite|capability"):
                    contract.validate_capability_profile(
                        changed,
                        execution_kind="real_data",
                        experiment_id="EXP-20260821-034",
                    )

    def test_21_exp033_requires_sealed_exp034_input_artifacts(self) -> None:
        scope = self.scope(profile_id=EXP033_PROFILE)
        for field in ("feature_input_release_hash", "feature_lineage_manifest_hash"):
            self.assertEqual(scope[field]["binding_role"], "sealed_input_artifact")
            self.assertIsNotNone(scope[field]["artifact_manifest"])
            changed = copy.deepcopy(scope)
            changed[field]["binding_role"] = "planned_output_contract"
            changed[field]["artifact_manifest"] = None
            with self.assertRaisesRegex(ValueError, "EXP033|sealed|artifact|planned"):
                contract.normalize_ordinary_real_data_run_scope(
                    changed, proposal_scope=self.proposal(EXP033_PROFILE)
                )


class DigestBindingTests(V3Fixture):
    def assert_bound(self, mutate: Any) -> None:
        case = self.authority_case()
        original_digest = contract.canonical_digest(case["scope"])
        changed = copy.deepcopy(case["scope"])
        mutate(changed)
        self.assertNotEqual(contract.canonical_digest(changed), original_digest)
        case["scope"] = changed
        self.assertFalse(self.authorize(case))

    def test_05_execution_kind_mutation_breaks_digest_and_authority(self) -> None:
        self.assert_bound(lambda value: value.__setitem__("execution_kind", "synthetic"))

    def test_06_capability_mutation_breaks_digest_and_authority(self) -> None:
        self.assert_bound(
            lambda value: value["capability_profile"]["capabilities"].__setitem__(
                "train_research_model", True
            )
        )

    def test_07_input_manifest_mutation_breaks_digest_and_authority(self) -> None:
        self.assert_bound(
            lambda value: value["data_input_manifest_hashes"][0].__setitem__(
                "sha256", sha("changed-input")
            )
        )

    def test_08_runner_universe_mutation_breaks_digest_and_authority(self) -> None:
        self.assert_bound(
            lambda value: value["runner_universe_manifest_hash"].__setitem__(
                "sha256", sha("changed-runners")
            )
        )

    def test_09_environment_mutation_breaks_digest_and_authority(self) -> None:
        self.assert_bound(lambda value: value["environment"].__setitem__("timezone", "UTC"))

    def test_10_exact_command_mutation_breaks_digest_and_authority(self) -> None:
        self.assert_bound(
            lambda value: value["exact_commands"][0]["argv"].append(
                "--unapproved-option"
            )
        )


class MetadataAndAccessTests(V3Fixture):
    def test_synthetic_material_preflight_is_row_empty_and_opens_zero_rows(self) -> None:
        scope = self.scope(profile_id=SYNTHETIC_PROFILE)
        payload_bytes = canonical_bytes(self.catalog_payload(scope), trailing_lf=True)

        def git_blob(root: Path, _commit: str, path: str) -> bytes:
            return (root / path).read_bytes()

        with mock.patch.object(contract, "_git_blob_bytes", side_effect=git_blob):
            receipt = contract.verify_ordinary_real_data_run_materials(
                self.root,
                scope,
                catalog_bytes=payload_bytes,
                output_root_exists=lambda _path: False,
            )
        self.assertEqual(receipt["row_count_metadata"], 0)
        self.assertEqual(receipt["race_count_metadata"], 0)
        self.assertEqual(receipt["runner_count_metadata"], 0)
        self.assertEqual(receipt["real_data_rows_opened"], 0)

    def test_forged_exp034_output_attestation_binding_is_rejected_pre_row(self) -> None:
        scope = self.scope(profile_id=EXP033_PROFILE)
        changed = copy.deepcopy(scope)
        attestation_path = changed["feature_input_release_hash"][
            "producer_output_attestation"
        ]["path"]
        forged_digest = sha("forged-producer-output-attestation")
        for field in ("feature_input_release_hash", "feature_lineage_manifest_hash"):
            changed[field]["producer_output_attestation"]["sha256"] = forged_digest
        for entry in changed["read_allowlist"]:
            if entry["path"] == attestation_path:
                entry["sha256"] = forged_digest
        for entry in changed["input_catalog"]["metadata_manifest_refs"]:
            if entry["path"] == attestation_path:
                entry["sha256"] = forged_digest
        changed = contract.normalize_ordinary_real_data_run_scope(
            changed, proposal_scope=self.proposal(EXP033_PROFILE)
        )

        def git_blob(root: Path, _commit: str, path: str) -> bytes:
            return (root / path).read_bytes()

        with mock.patch.object(contract, "_git_blob_bytes", side_effect=git_blob):
            with self.assertRaisesRegex(
                contract.ContractError, "hash.bound metadata|attestation"
            ):
                contract.verify_ordinary_real_data_run_materials(
                    self.root,
                    changed,
                    catalog_bytes=canonical_bytes(
                        self.catalog_payload(changed), trailing_lf=True
                    ),
                    output_root_exists=lambda _path: False,
                )

    def test_11_incomplete_source_time_metadata_is_rejected(self) -> None:
        scope = self.scope()
        for field in (
            "source_event_time_coverage",
            "received_at_coverage",
            "available_as_of_coverage",
        ):
            with self.subTest(field=field):
                payload = self.catalog_payload(scope)
                payload[field]["covered_count"] = 69
                with self.assertRaisesRegex(ValueError, "cover|coverage"):
                    contract.verify_metadata_preflight(scope, payload)

    def test_12_revoked_input_is_rejected(self) -> None:
        scope = self.scope()
        payload = self.catalog_payload(scope)
        payload["revoked"] = True
        payload["revocation_status"] = "revoked"
        with self.assertRaisesRegex(ValueError, "revok|active"):
            contract.verify_metadata_preflight(scope, payload)

    def test_13_read_allowlist_rejects_outside_path(self) -> None:
        case = self.authority_case()
        with self.assertRaisesRegex(ValueError, "allowlist|outside"):
            contract.verify_access_request(
                case["scope"],
                phase_id="canonicalize_input_release",
                mode="row_read",
                path="data/private/unapproved.jsonl",
                authority_context=self.authority_context(case),
            )

    def test_14_write_allowlist_rejects_outside_path(self) -> None:
        case = self.authority_case()
        with self.assertRaisesRegex(ValueError, "allowlist|outside"):
            contract.verify_access_request(
                case["scope"],
                phase_id="seal_research_outputs",
                mode="write",
                path="outputs/production/model.pkl",
                authority_context=self.authority_context(case),
            )

    def test_incomplete_receipt_cannot_unlock_row_or_write_access(self) -> None:
        scope = self.scope()
        with self.assertRaisesRegex(ValueError, "authority|receipt|verified"):
            contract.verify_access_request(
                scope,
                phase_id="canonicalize_input_release",
                mode="row_read",
                path=self.refs["runner_rows"]["path"],
                authority_context={},
            )
        with self.assertRaisesRegex(ValueError, "authority|receipt|verified"):
            contract.verify_access_request(
                scope,
                phase_id="seal_research_outputs",
                mode="write",
                path=scope["output_sealing_contract"]["result_manifest_path"],
                authority_context={},
            )

    def test_bound_row_and_write_access_accept_exact_live_authority_only(self) -> None:
        case = self.authority_case()
        context = self.authority_context(case)
        command = next(
            item
            for item in case["scope"]["exact_commands"]
            if item["phase_id"] == "canonicalize_input_release"
        )
        with mock.patch.object(
            contract, "_current_git_commit", return_value=case["scope"]["execution_commit_sha"]
        ), mock.patch.object(
            contract.Path, "cwd", return_value=self.root
        ), mock.patch.object(
            contract, "verify_runtime_interpreter_isolation", return_value=None
        ), mock.patch.object(
            contract, "verify_execution_worktree_state", return_value=None
        ), mock.patch.object(
            contract, "observe_runtime_environment", return_value=case["scope"]["environment"]
        ), mock.patch.object(
            contract, "observe_process_argv", return_value=command["argv"]
        ), mock.patch.object(
            contract, "GitHubRestApprovalProvider", return_value=case["provider"]
        ):
            contract.verify_access_request(
                case["scope"],
                phase_id="canonicalize_input_release",
                mode="row_read",
                path=self.refs["runner_rows"]["path"],
                authority_context=context,
            )
        seal_command = next(
            item
            for item in case["scope"]["exact_commands"]
            if item["phase_id"] == "seal_research_outputs"
        )
        with mock.patch.object(
            contract, "_current_git_commit", return_value=case["scope"]["execution_commit_sha"]
        ), mock.patch.object(
            contract.Path, "cwd", return_value=self.root
        ), mock.patch.object(
            contract, "verify_runtime_interpreter_isolation", return_value=None
        ), mock.patch.object(
            contract, "verify_execution_worktree_state", return_value=None
        ), mock.patch.object(
            contract, "observe_runtime_environment", return_value=case["scope"]["environment"]
        ), mock.patch.object(
            contract, "observe_process_argv", return_value=seal_command["argv"]
        ), mock.patch.object(
            contract, "GitHubRestApprovalProvider", return_value=case["provider"]
        ):
            contract.verify_access_request(
                case["scope"],
                phase_id="seal_research_outputs",
                mode="write",
                path=case["scope"]["output_sealing_contract"]["result_manifest_path"],
                authority_context=context,
            )
        self.assertEqual(case["preflight"]["real_data_rows_opened"], 0)

    def test_15_output_root_must_be_fresh(self) -> None:
        scope = self.scope()
        with self.assertRaisesRegex(ValueError, "exists|reuse|overwrite"):
            contract.validate_output_root_fresh(
                self.root, scope, exists=lambda _path: True
            )
        contract.validate_output_root_fresh(
            self.root, scope, exists=lambda _path: False
        )

    def test_24_metadata_preflight_opens_zero_real_data_rows(self) -> None:
        scope = self.scope()
        trap = RowReadTrap()
        receipt = contract.verify_metadata_preflight(
            scope, self.catalog_payload(scope), row_loader=trap
        )
        self.assertEqual(trap.count, 0)
        self.assertEqual(receipt["real_data_rows_opened"], 0)
        self.assertTrue(receipt["source_time_coverage_complete"])

    def test_metadata_preflight_paths_are_disjoint_from_every_row_blob(self) -> None:
        for profile_id in (SYNTHETIC_PROFILE, EXP034_PROFILE, EXP033_PROFILE):
            with self.subTest(profile_id=profile_id):
                scope = self.scope(profile_id=profile_id)
                metadata_phase = next(
                    phase
                    for phase in scope["phase_plan"]
                    if phase["phase_id"] == "metadata_preflight"
                )
                read_entries = {
                    entry["path"]: entry for entry in scope["read_allowlist"]
                }
                row_paths = {
                    path
                    for path, entry in read_entries.items()
                    if entry["access_class"] != "metadata_manifest"
                }
                self.assertTrue(
                    all(
                        read_entries[path]["access_class"] == "metadata_manifest"
                        for path in metadata_phase["read_paths"]
                    )
                )
                self.assertTrue(row_paths.isdisjoint(metadata_phase["read_paths"]))

    def test_catalog_payload_is_constructibly_hash_bound_without_self_reference(self) -> None:
        scope = self.scope()
        payload_bytes = canonical_bytes(self.catalog_payload(scope), trailing_lf=True)

        def git_blob(root: Path, _commit: str, path: str) -> bytes:
            return (root / path).read_bytes()

        with mock.patch.object(contract, "_git_blob_bytes", side_effect=git_blob):
            receipt = contract.verify_ordinary_real_data_run_materials(
                self.root,
                scope,
                catalog_bytes=payload_bytes,
                output_root_exists=lambda _path: False,
            )
        self.assertEqual(receipt["real_data_rows_opened"], 0)


class ApprovalAndAuthorizationTests(V3Fixture):
    def issue_receipt(
        self,
        case: dict[str, Any],
        *,
        atomic_reserver: Any = None,
    ) -> dict[str, Any]:
        kwargs = {
            "root": self.root,
            "status": "running",
            "run_scope": case["scope"],
            "cli_execution_kind": "real-data",
            "prepare_evidence": case["prepare"],
            "run_evidence": case["run"],
            "execution_commit": case["scope"]["execution_commit_sha"],
            "current_main_sha": case["main_sha"],
            "ancestry_evidence": {
                "status": case["receipt"]["execution_commit_compare_status"],
                "url": case["receipt"]["execution_commit_compare_url"],
                "merge_base_sha": case["receipt"][
                    "execution_commit_merge_base_sha"
                ],
            },
            "merged_running_event": case["event"],
            "current_main_registry_bytes": case["registry_bytes"],
            "metadata_preflight_receipt": case["preflight"],
        }
        patches = [
            mock.patch.object(contract, "GitHubRestApprovalProvider", return_value=case["provider"]),
            mock.patch.object(contract.Path, "cwd", return_value=self.root),
            mock.patch.object(contract, "verify_runtime_interpreter_isolation", return_value=None),
            mock.patch.object(contract, "_current_git_commit", return_value=case["scope"]["execution_commit_sha"]),
            mock.patch.object(contract, "verify_execution_worktree_state", return_value=None),
            mock.patch.object(contract, "observe_runtime_environment", return_value=case["scope"]["environment"]),
            mock.patch.object(contract, "verify_ordinary_real_data_run_materials", return_value=case["preflight"]),
            mock.patch.object(contract, "_utc_now_text", return_value="2026-08-22T15:01:00Z"),
        ]
        if atomic_reserver is not None:
            patches.append(mock.patch.object(contract, "_atomic_reserve_output_root", side_effect=atomic_reserver))
        with contextlib.ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            return contract.issue_execution_receipt(**kwargs)

    def test_03_synthetic_scope_never_has_real_data_authority(self) -> None:
        scope = self.scope(profile_id=SYNTHETIC_PROFILE)
        self.assertFalse(
            contract.verify_real_data_authorization(
                status="running",
                run_scope=scope,
                cli_execution_kind="real-data",
                prepare_evidence=None,
                run_evidence=None,
                execution_commit=scope["execution_commit_sha"],
                current_main_sha="b" * 40,
            )
        )

    def test_04_real_data_is_false_before_exact_run_approval(self) -> None:
        case = self.authority_case()
        self.assertFalse(self.authorize(case, run_evidence=None))

    def test_16_edited_or_body_changed_comment_is_rejected(self) -> None:
        case = self.authority_case()
        case["run"]["updated_at"] = "2026-08-22T02:00:01Z"
        self.assertFalse(self.authorize(case))
        case = self.authority_case()
        case["run"]["body"] += " "
        self.assertFalse(self.authorize(case))

    def test_17_prepare_and_run_comment_ids_must_be_distinct(self) -> None:
        case = self.authority_case()
        case["run"]["comment_id"] = case["prepare"]["comment_id"]
        case["receipt"]["run_approval_comment_id"] = case["prepare"]["comment_id"]
        self.assertFalse(self.authorize(case))

    def test_18_synthetic_claim_cannot_override_kind(self) -> None:
        case = self.authority_case()
        case["scope"] = self.scope(profile_id=SYNTHETIC_PROFILE)
        case["event"]["execution_kind"] = "synthetic"
        case["receipt"]["execution_kind"] = "synthetic"
        self.assertFalse(self.authorize(case))

    def test_full_exact_merged_running_receipt_and_environment_are_required(self) -> None:
        case = self.authority_case()
        self.assertTrue(self.authorize(case))
        self.assertFalse(self.authorize(case, merged_running_event=None))
        self.assertFalse(self.authorize(case, execution_receipt=None))
        self.assertFalse(self.authorize(case, metadata_preflight_receipt=None))
        self.assertFalse(self.authorize(case, observed_environment={}))

    def test_caller_fabricated_registry_bundle_is_rejected_by_live_provider(self) -> None:
        case = self.authority_case()
        case["registry_bytes"] += canonical_bytes(
            {"event_id": "caller-fabricated-but-internally-consistent"},
            trailing_lf=True,
        )
        case["receipt"]["current_main_registry_sha256"] = hashlib.sha256(
            case["registry_bytes"]
        ).hexdigest()
        self.assertFalse(self.authorize(case))

    def test_illegal_remote_registry_transition_cannot_authorize(self) -> None:
        case = self.authority_case()
        events = [
            json.loads(line)
            for line in case["registry_bytes"].decode("utf-8").splitlines()
        ]
        events[2]["previous_status"] = "proposed"
        forged_registry = b"".join(
            canonical_bytes(event, trailing_lf=True) for event in events
        )
        case["registry_bytes"] = forged_registry
        case["provider"].registry_content = forged_registry
        case["provider"].files[("research/REGISTRY.jsonl", case["main_sha"])] = (
            forged_registry
        )
        case["receipt"]["current_main_registry_sha256"] = hashlib.sha256(
            forged_registry
        ).hexdigest()
        receipt_path = self.root / case["scope"]["output_sealing_contract"][
            "execution_receipt_path"
        ]
        receipt_path.write_bytes(canonical_bytes(case["receipt"], trailing_lf=True))
        self.assertFalse(self.authorize(case))

    def test_receipt_tampering_for_each_bound_family_is_rejected(self) -> None:
        fields = (
            "capability_profile_digest",
            "input_manifest_hashes_digest",
            "environment_digest",
            "exact_commands_digest",
            "read_allowlist_digest",
            "write_allowlist_digest",
            "output_root_reservation_digest",
        )
        for field in fields:
            with self.subTest(field=field):
                case = self.authority_case()
                case["receipt"][field] = sha(f"tampered-{field}")
                self.assertFalse(self.authorize(case))

    def test_caller_boolean_is_not_an_authorization_parameter(self) -> None:
        self.assertNotIn(
            "bindings_verified",
            inspect.signature(contract.verify_real_data_authorization).parameters,
        )

    def test_execution_receipt_is_issued_only_after_exact_merged_authority(self) -> None:
        case = self.authority_case(persist_receipt=False)
        receipt = self.issue_receipt(case)
        self.assertEqual(receipt, case["receipt"])
        self.assertTrue((self.root / case["scope"]["output_root"]).is_dir())
        self.assertEqual(case["preflight"]["real_data_rows_opened"], 0)

    def test_execution_receipt_rejects_duplicate_output_root_reservation(self) -> None:
        case = self.authority_case(persist_receipt=False)
        self.issue_receipt(case)
        with self.assertRaisesRegex(contract.ContractError, "already present"):
            self.issue_receipt(case)

    def test_crash_after_atomic_reservation_leaves_root_permanently_unusable(self) -> None:
        case = self.authority_case(persist_receipt=False)

        def reserve_then_crash(path: Path) -> bool:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.mkdir()
            raise RuntimeError("synthetic crash after atomic reservation")

        with self.assertRaisesRegex(RuntimeError, "synthetic crash"):
            self.issue_receipt(case, atomic_reserver=reserve_then_crash)
        output_root = self.root / case["scope"]["output_root"]
        self.assertTrue(output_root.is_dir())
        with self.assertRaisesRegex(contract.ContractError, "already present"):
            self.issue_receipt(case)


class UpdateRegistryV3LifecycleTests(V3Fixture):
    lifecycle_experiment_id = "EXP-20260821-034"

    def setUp(self) -> None:
        super().setUp()
        for relative in (
            f"research/scopes/{self.lifecycle_experiment_id}.proposal.json",
            f"research/queue/{self.lifecycle_experiment_id}.json",
            "research/REGISTRY.jsonl",
        ):
            source = REPO_ROOT / relative
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        self.registry_path = self.root / "research/REGISTRY.jsonl"
        self.provider = FakeGitHubProvider(self.registry_path.read_bytes())
        self.scope_value = self.scope()
        self.run_scope_path = (
            self.root
            / "research/scopes"
            / f"{self.lifecycle_experiment_id}.run.json"
        )
        self.run_scope_path.write_bytes(
            contract.canonical_json_bytes(self.scope_value) + b"\n"
        )
        self.run_digest = contract.canonical_digest(self.scope_value)
        events = update_registry.load_events(self.registry_path)
        prepare_event = next(
            event
            for event in events
            if event["experiment_id"] == self.lifecycle_experiment_id
            and event["status"] == "approved_to_prepare"
        )
        prepare = prepare_event["approval_evidence"]
        assert isinstance(prepare, dict)
        self.provider.comments[prepare["comment_id"]] = self._raw_comment(prepare)

    @staticmethod
    def _raw_comment(evidence: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": evidence["comment_id"],
            "html_url": evidence["url"],
            "issue_url": (
                "https://api.github.com/repos/kazuponbaseball-cell/"
                f"keiba_ai_project/issues/{evidence['issue_number']}"
            ),
            "user": {
                "login": evidence["author"],
                "type": evidence["author_type"],
            },
            "created_at": evidence["created_at"],
            "updated_at": evidence["updated_at"],
            "body": evidence["body"],
        }

    def _argv(
        self,
        status: str,
        *,
        comment_id: int | None = None,
        execution_kind: str = "none",
    ) -> list[str]:
        argv = [
            self.lifecycle_experiment_id,
            status,
            "--root",
            str(self.root),
            "--queue-file",
            str(
                self.root
                / "research/queue"
                / f"{self.lifecycle_experiment_id}.json"
            ),
            "--github-repository",
            "kazuponbaseball-cell/keiba_ai_project",
            "--github-base-branch",
            "main",
            "--execution-kind",
            execution_kind,
        ]
        if comment_id is not None:
            argv.extend(
                [
                    "--issue-number",
                    "49",
                    "--approval-comment-id",
                    str(comment_id),
                ]
            )
        return argv

    def _invoke(
        self,
        status: str,
        *,
        comment_id: int | None = None,
        execution_kind: str = "none",
        expect_error: str | None = None,
    ) -> dict[str, Any] | str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        preflight = contract.verify_metadata_preflight(
            self.scope_value,
            self.catalog_payload(self.scope_value),
        )
        with mock.patch.object(
            update_registry,
            "verify_ordinary_real_data_run_materials",
            return_value=preflight,
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if expect_error is None:
                code = update_registry.main(
                    self._argv(
                        status,
                        comment_id=comment_id,
                        execution_kind=execution_kind,
                    ),
                    approval_provider=self.provider,
                    execution_commit_provider=lambda _root: "a" * 40,
                    execution_worktree_verifier=lambda _root, _allowed: None,
                )
                self.assertEqual(code, 0, stderr.getvalue())
                return json.loads(stdout.getvalue())
            with self.assertRaises(SystemExit) as caught:
                update_registry.main(
                    self._argv(
                        status,
                        comment_id=comment_id,
                        execution_kind=execution_kind,
                    ),
                    approval_provider=self.provider,
                    execution_commit_provider=lambda _root: "a" * 40,
                    execution_worktree_verifier=lambda _root, _allowed: None,
                )
            self.assertEqual(caught.exception.code, 2)
            self.assertIn(expect_error, stderr.getvalue())
            return stderr.getvalue()

    def _merge_pending_event(self) -> None:
        self.provider.merge_registry(self.registry_path.read_bytes())

    def test_v3_real_data_lifecycle_dispatches_and_sets_flag_only_at_running(self) -> None:
        required = self._invoke("RUN_APPROVAL_REQUIRED")
        assert isinstance(required, dict)
        self.assertEqual(required["event"]["status"], "run_approval_required")
        self.assertFalse(required["event"]["real_data_execution_allowed"])
        self._merge_pending_event()

        run_comment_id = 990001
        run_evidence = {
            "comment_id": run_comment_id,
            "url": (
                "https://github.com/kazuponbaseball-cell/keiba_ai_project/"
                f"issues/49#issuecomment-{run_comment_id}"
            ),
            "issue_number": 49,
            "author": "kazuponbaseball-cell",
            "author_type": "User",
            "body": f"APPROVED_TO_RUN {self.run_digest}",
            "created_at": "2026-08-22T04:00:00Z",
            "updated_at": "2026-08-22T04:00:00Z",
        }
        self.provider.comments[run_comment_id] = self._raw_comment(run_evidence)
        approved = self._invoke("APPROVED_TO_RUN", comment_id=run_comment_id)
        assert isinstance(approved, dict)
        self.assertEqual(approved["event"]["status"], "approved_to_run")
        self.assertFalse(approved["event"]["real_data_execution_allowed"])
        self._merge_pending_event()

        self.provider.comments[run_comment_id]["updated_at"] = (
            "2026-08-22T04:00:01Z"
        )
        before = self.registry_path.read_bytes()
        self._invoke(
            "RUNNING",
            execution_kind="real-data",
            expect_error="changed after approval",
        )
        self.assertEqual(self.registry_path.read_bytes(), before)
        self.provider.comments[run_comment_id]["updated_at"] = (
            "2026-08-22T04:00:00Z"
        )

        running = self._invoke("RUNNING", execution_kind="real-data")
        assert isinstance(running, dict)
        event = running["event"]
        self.assertEqual(event["status"], "running")
        self.assertEqual(event["execution_kind"], "real-data")
        self.assertFalse(event["real_data_execution_allowed"])
        self.assertFalse(event["execution_authorized"])
        self.assertFalse(event["formal_buy"])
        self.assertFalse(event["send_order"])
        self.assertEqual(event["stake"], 0)

    def test_update_registry_rejects_unknown_version_before_event_append(self) -> None:
        raw = json.loads(self.run_scope_path.read_text(encoding="utf-8"))
        raw["run_scope_schema_version"] = "ordinary_real_data_run_v999"
        self.run_scope_path.write_bytes(canonical_bytes(raw, trailing_lf=True))
        before = self.registry_path.read_bytes()
        self._invoke(
            "RUN_APPROVAL_REQUIRED",
            expect_error="unknown ordinary run_scope_schema_version",
        )
        self.assertEqual(self.registry_path.read_bytes(), before)

    def test_update_registry_rejects_registry_wide_reused_run_comment_id(self) -> None:
        required = self._invoke("RUN_APPROVAL_REQUIRED")
        assert isinstance(required, dict)
        self._merge_pending_event()
        events = update_registry.load_events(self.registry_path)
        consumed_comment_id = next(
            event["approval_evidence"]["comment_id"]
            for event in events
            if event["status"] == "approved_to_run"
            and isinstance(event.get("approval_evidence"), dict)
        )
        before = self.registry_path.read_bytes()
        self._invoke(
            "APPROVED_TO_RUN",
            comment_id=consumed_comment_id,
            expect_error="was already used by an approval grant",
        )
        self.assertEqual(self.registry_path.read_bytes(), before)

    def test_bound_v3_history_uses_each_events_own_proposal_scope(self) -> None:
        current_proposal = self.proposal(EXP033_PROFILE)
        observed_proposals: list[dict[str, Any]] = []

        def fake_loader(
            _root: Path,
            _path: Path,
            proposal_scope: dict[str, Any],
        ) -> tuple[dict[str, Any], str, str]:
            observed_proposals.append(proposal_scope)
            return (
                {"execution_kind": "real_data"},
                sha("prior-exp034-v3-run"),
                contract.RUN_SCOPE_SCHEMA_VERSION,
            )

        event = {
            "schema_version": 2,
            "experiment_id": "EXP-20260821-034",
            "status": "running",
            "execution_kind": "real-data",
            "run_scope_file": "research/scopes/EXP-20260821-034.run.json",
            "run_scope_digest": sha("prior-exp034-v3-run"),
        }
        with mock.patch.object(
            update_registry,
            "load_versioned_ordinary_run_scope",
            side_effect=fake_loader,
        ):
            self.assertFalse(
                update_registry.contains_unbound_legacy_real_data_running(
                    [event],
                    root=self.root,
                    proposal_scope=current_proposal,
                )
            )
        self.assertEqual(
            [item["experiment_id"] for item in observed_proposals],
            ["EXP-20260821-034"],
        )


class OutputSealingTests(V3Fixture):
    def _artifact_payload(self, role: str) -> bytes:
        artifact_format = contract.ARTIFACT_FORMAT_BY_ROLE[role]
        if artifact_format == "opaque_binary":
            return b"synthetic opaque model bytes"
        if artifact_format == "canonical_json":
            return canonical_bytes({"synthetic_report": role}, trailing_lf=True)
        if role in {
            "canonical_runner_universe",
            "canonical_target_input_release",
            "canonical_feature_lineage",
            "target_research_prediction",
        }:
            return (self.root / self.refs["runner_rows"]["path"]).read_bytes()
        return (self.root / self.refs["training_rows"]["path"]).read_bytes()

    def manifest(self, scope: dict[str, Any], status: str) -> dict[str, Any]:
        role_paths = scope["output_sealing_contract"]["artifact_paths"]
        artifacts = [
            {
                "role": item["role"],
                "path": item["path"],
                "sha256": sha(f"artifact-{item['role']}"),
                "row_count": 70,
                "race_count": 5,
                "runner_count": 70,
                "complete": True,
            }
            for item in role_paths
        ]
        partial = []
        if role_paths:
            partial = [
                {
                    "role": role_paths[0]["role"],
                    "path": role_paths[0]["path"],
                    "sha256": sha("partial-failure"),
                    "row_count": 0,
                    "race_count": 0,
                    "runner_count": 0,
                    "complete": False,
                }
            ]
        return {
            "result_manifest_schema_version": contract.RESULT_MANIFEST_SCHEMA_VERSION,
            "experiment_id": scope["proposal_scope"]["experiment_id"],
            "capability_profile_id": scope["capability_profile"]["profile_id"],
            "run_scope_digest": contract.canonical_digest(scope),
            "execution_receipt_digest": sha("execution-receipt"),
            "status": status,
            "generated_at": "2026-08-22T15:30:00Z",
            "as_of": scope["as_of"],
            "output_root": scope["output_root"],
            "artifacts": artifacts if status == "success" else [],
            "partial_outputs": [] if status == "success" else partial,
            "code_hashes_digest": contract.canonical_digest(scope["code_hashes"]),
            "config_hashes_digest": contract.canonical_digest(scope["config_hashes"]),
            "input_manifest_hashes_digest": contract.canonical_digest(
                scope["data_input_manifest_hashes"]
            ),
            "environment_lock_sha256": scope["dependency_environment_lock_hash"]["sha256"],
            "consumer_eligible": status == "success",
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }

    def test_success_manifest_is_complete_hash_bound_and_consumer_eligible(self) -> None:
        scope = self.scope()
        normalized = contract.normalize_result_manifest(
            self.manifest(scope, "success"),
            run_scope=scope,
            run_scope_digest=contract.canonical_digest(scope),
        )
        self.assertTrue(normalized["consumer_eligible"])
        self.assertTrue(all(item["complete"] for item in normalized["artifacts"]))

    def test_failure_or_crash_is_consumer_ineligible_and_fail_closed(self) -> None:
        scope = self.scope()
        run_digest = contract.canonical_digest(scope)
        normalized = contract.normalize_result_manifest(
            self.manifest(scope, "failure"),
            run_scope=scope,
            run_scope_digest=run_digest,
        )
        self.assertFalse(normalized["consumer_eligible"])
        self.assertTrue(normalized["partial_outputs"])
        changed = self.manifest(scope, "failure")
        changed["consumer_eligible"] = True
        with self.assertRaisesRegex(ValueError, "failure|consumer|partial"):
            contract.normalize_result_manifest(
                changed, run_scope=scope, run_scope_digest=run_digest
            )
        changed = self.manifest(scope, "failure")
        changed["status"] = "crash"
        with self.assertRaisesRegex(ValueError, "status|success|failure"):
            contract.normalize_result_manifest(
                changed, run_scope=scope, run_scope_digest=run_digest
            )

    def test_artifact_outside_profile_output_root_is_rejected(self) -> None:
        scope = self.scope()
        raw = self.manifest(scope, "success")
        raw["artifacts"][0]["path"] = (
            "outputs/research/EXP-20260821-033/model/foreign.json"
        )
        with self.assertRaisesRegex(ValueError, "outside|output root"):
            contract.normalize_result_manifest(
                raw,
                run_scope=scope,
                run_scope_digest=contract.canonical_digest(scope),
            )

    def test_output_seal_reads_durable_artifacts_and_rejects_extra_root_file(self) -> None:
        case = self.authority_case()
        scope = case["scope"]
        run_digest = contract.canonical_digest(scope)
        result = self.manifest(scope, "success")
        result["execution_receipt_digest"] = contract.canonical_digest(case["receipt"])
        payloads: dict[str, bytes] = {}
        for artifact in result["artifacts"]:
            payload = self._artifact_payload(artifact["role"])
            artifact["sha256"] = hashlib.sha256(payload).hexdigest()
            counts = contract._derive_artifact_counts(
                artifact["role"], payload, complete=True, run_scope=scope
            )
            (
                artifact["row_count"],
                artifact["race_count"],
                artifact["runner_count"],
            ) = counts
            path = self.root / artifact["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            payloads[artifact["path"]] = payload
        normalized = contract.normalize_result_manifest(
            result, run_scope=scope, run_scope_digest=run_digest
        )
        result_bytes = canonical_bytes(normalized, trailing_lf=True)
        seal_command = next(
            item
            for item in scope["exact_commands"]
            if item["phase_id"] == "seal_research_outputs"
        )

        def verify() -> dict[str, Any]:
            with mock.patch.object(
                contract, "_current_git_commit", return_value=scope["execution_commit_sha"]
            ), mock.patch.object(
                contract.Path, "cwd", return_value=self.root
            ), mock.patch.object(
                contract, "verify_runtime_interpreter_isolation", return_value=None
            ), mock.patch.object(
                contract, "verify_execution_worktree_state", return_value=None
            ), mock.patch.object(
                contract, "observe_runtime_environment", return_value=scope["environment"]
            ), mock.patch.object(
                contract, "observe_process_argv", return_value=seal_command["argv"]
            ), mock.patch.object(
                contract, "GitHubRestApprovalProvider", return_value=case["provider"]
            ), mock.patch.object(
                contract, "_utc_now_text", return_value="2026-08-22T16:00:00Z"
            ):
                return contract.verify_output_seal(
                    root=self.root,
                    run_scope=scope,
                    run_scope_digest=run_digest,
                    execution_receipt=case["receipt"],
                    result_manifest=normalized,
                    result_manifest_path=scope["output_sealing_contract"]["result_manifest_path"],
                    result_manifest_bytes=result_bytes,
                    artifact_bytes_by_path=payloads,
                    authority_context=self.authority_context(case),
                )

        self.assertEqual(verify(), normalized)
        extra = self.root / scope["output_root"] / "unmanifested.tmp"
        extra.write_bytes(b"forbidden")
        with self.assertRaisesRegex(contract.ContractError, "unmanifested|missing"):
            verify()


if __name__ == "__main__":
    unittest.main()
