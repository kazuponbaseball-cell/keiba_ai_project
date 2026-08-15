from __future__ import annotations

import base64
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import github_approval
import registered_nonpromotion_offline_approval_v1 as approval
import registered_nonpromotion_offline_contract_v1 as contract


BASE = "a" * 40
BLOB = "b" * 40
APPROVER = "kazuponbaseball-cell"
ISSUE = 42
COMMENT_ID = 9001


def _normalized_repository_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise AssertionError(f"test repository material contains a UTF-8 BOM: {path}")
    return (
        raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    )


def approvers_bytes() -> bytes:
    payload = {
        "schema_version": 1,
        "approvers": [{"login": APPROVER, "role": "repository_owner"}],
        "denied_login_patterns": [
            "bot",
            "codex",
            "automation",
            "github-actions",
            "dependabot",
        ],
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


class FakeProvider:
    def __init__(self) -> None:
        self.repository = {
            "full_name": contract.DEFAULT_REPOSITORY,
            "default_branch": contract.DEFAULT_BASE_BRANCH,
        }
        self.ref = {
            "ref": "refs/heads/main",
            "object": {"type": "commit", "sha": BASE},
        }
        self.compare = {
            "url": (
                f"{github_approval.GITHUB_API_URL}/repos/"
                f"{contract.DEFAULT_REPOSITORY}/compare/{BASE}...{BASE}"
            ),
            "status": "identical",
            "base_commit": {"sha": BASE},
            "merge_base_commit": {"sha": BASE},
        }
        self.contents_by_path: dict[str, dict[str, object]] = {}
        self.calls: dict[str, int] = {
            "repository": 0,
            "branch_ref": 0,
            "compare": 0,
            "file_contents": 0,
            "issue_comment": 0,
        }
        self.file_content_paths: list[str] = []
        self.add_repository_file(
            github_approval.APPROVERS_PATH,
            approvers_bytes(),
            blob_sha=BLOB,
        )
        for path in sorted(set(contract.RUNTIME_MATERIAL_PATHS.values())):
            self.add_repository_file(
                path,
                _normalized_repository_bytes(ROOT / Path(*Path(path).parts)),
            )
        self.add_repository_file(
            approval.REGISTRY_PATH,
            _normalized_repository_bytes(ROOT / approval.REGISTRY_PATH),
        )
        self.comment: dict[str, object] = {}

    def add_repository_file(
        self,
        path: str,
        content: bytes,
        *,
        blob_sha: str | None = None,
    ) -> None:
        self.contents_by_path[path] = {
            "type": "file",
            "path": path,
            "sha": blob_sha
            or hashlib.sha1(b"blob\0" + path.encode("utf-8") + content).hexdigest(),
            "encoding": "base64",
            "content": base64.b64encode(content).decode("ascii"),
        }

    def get_repository(self, repository: str) -> dict[str, object]:
        self.calls["repository"] += 1
        return copy.deepcopy(self.repository)

    def get_branch_ref(self, repository: str, branch: str) -> dict[str, object]:
        self.calls["branch_ref"] += 1
        return copy.deepcopy(self.ref)

    def compare_commits(
        self,
        repository: str,
        base: str,
        head: str,
    ) -> dict[str, object]:
        self.calls["compare"] += 1
        return copy.deepcopy(self.compare)

    def get_file_contents(
        self,
        repository: str,
        path: str,
        ref: str,
    ) -> dict[str, object]:
        self.calls["file_contents"] += 1
        self.file_content_paths.append(path)
        if ref != BASE or path not in self.contents_by_path:
            raise KeyError(f"unavailable immutable repository material: {path}@{ref}")
        return copy.deepcopy(self.contents_by_path[path])

    def get_issue_comment(
        self,
        repository: str,
        comment_id: int,
    ) -> dict[str, object]:
        self.calls["issue_comment"] += 1
        return copy.deepcopy(self.comment)

    def reset_calls(self) -> None:
        for key in self.calls:
            self.calls[key] = 0
        self.file_content_paths.clear()


class OfflineApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registered = contract.resolve_offline_registered_recipe(ROOT)
        self.provider = FakeProvider()
        source_bindings = {
            role: {
                "path": source["path"],
                "expected_sha256": source["expected_sha256"],
                "observed_sha256": source["expected_sha256"],
                "observed_byte_size": index + 1,
            }
            for index, (role, source) in enumerate(contract.SOURCE_INPUTS.items())
        }
        projection_bindings = {
            "candidate_projection": {
                "path": contract.PROJECTION_INPUTS["candidate_projection"]["path"],
                "sha256": "1" * 64,
                "byte_size": 100,
            },
            "settlement_projection": {
                "path": contract.PROJECTION_INPUTS["settlement_projection"]["path"],
                "sha256": "2" * 64,
                "byte_size": 200,
            },
        }
        bindings = {
            "repository": contract.DEFAULT_REPOSITORY,
            "base_branch": contract.DEFAULT_BASE_BRANCH,
            "run_scope_base_commit": BASE,
            "verified_current_main_sha": BASE,
            "approvers_blob_sha": BLOB,
            "approvers_content_sha256": hashlib.sha256(approvers_bytes()).hexdigest(),
            "runtime_material_sha256": dict(self.registered.runtime_material_digests),
            "source_bindings": source_bindings,
            "projection_bindings": projection_bindings,
            "materialization_manifest": {
                "path": contract.MATERIALIZATION_MANIFEST_PATH,
                "sha256": "4" * 64,
                "byte_size": 300,
            },
            "python_minor_version": "3.12",
            "numpy_version": "2.4.3",
            "environment_manifest_sha256": "3" * 64,
            "output_root": contract.FIXED_OUTPUT_ROOT,
            "sealed_at": "2026-08-15T00:00:00Z",
        }
        self.run_scope = contract.compile_offline_run_scope(self.registered, bindings)
        self.add_comment()

    def add_comment(
        self,
        *,
        body: str | None = None,
        login: str = APPROVER,
        actor_type: str = "User",
        created_at: str = "2026-08-15T00:01:00Z",
        updated_at: str | None = None,
    ) -> None:
        self.provider.comment = {
            "id": COMMENT_ID,
            "issue_url": (
                f"https://api.github.com/repos/{contract.DEFAULT_REPOSITORY}/"
                f"issues/{ISSUE}"
            ),
            "html_url": (
                f"https://github.com/{contract.DEFAULT_REPOSITORY}/issues/{ISSUE}"
                f"#issuecomment-{COMMENT_ID}"
            ),
            "user": {"login": login, "type": actor_type},
            "body": body
            or f"{contract.APPROVAL_KEYWORD} {self.run_scope['run_scope_digest']}",
            "created_at": created_at,
            "updated_at": updated_at if updated_at is not None else created_at,
        }

    def verify(self) -> dict[str, object]:
        return approval.verify_offline_run_approval(
            root=ROOT,
            run_scope=self.run_scope,
            issue_number=ISSUE,
            comment_id=COMMENT_ID,
            provider=self.provider,
            now="2026-08-15T00:02:00Z",
        )

    def reverify(
        self,
        evidence: dict[str, object],
        checkpoint: str,
    ) -> dict[str, object]:
        return approval.reverify_offline_run_approval(
            root=ROOT,
            run_scope=self.run_scope,
            approval_evidence=evidence,
            provider=self.provider,
            now="2026-08-15T00:03:00Z",
            checkpoint=checkpoint,
        )

    def test_exact_comment_grants_only_truthful_local_permission(self) -> None:
        evidence = self.verify()
        self.assertEqual(evidence["verification_checkpoint"], "INITIAL_APPROVAL")
        self.assertIsNone(evidence["original_evidence_digest"])
        self.assertFalse(evidence["authority"])
        self.assertTrue(evidence["local_offline_permission"])
        self.assertFalse(evidence["global_uniqueness_guaranteed"])
        self.assertTrue(evidence["implementation_current_main_ancestry_verified"])
        self.assertEqual(evidence["limitations"], approval.LIMITATIONS)
        self.assertTrue(evidence["ordinary_registry"]["target_comment_id_unused"])
        self.assertFalse(evidence["formal_buy"])
        self.assertFalse(evidence["send_order"])
        self.assertEqual(evidence["stake"], 0)

    def test_metadata_only_gate_availability_proves_main_bytes_not_run_approval(self) -> None:
        evidence = approval.verify_offline_gate_availability(
            root=ROOT,
            provider=self.provider,
            expected_base_commit=BASE,
            now="2026-08-15T00:00:00Z",
        )
        self.assertTrue(evidence["implementation_current_main_ancestry_verified"])
        self.assertTrue(evidence["materialization_permission"])
        self.assertFalse(evidence["run_approval_verified"])
        self.assertFalse(evidence["local_offline_run_permission"])
        self.assertFalse(evidence["authority"])
        self.assertNotIn("human_merge_metadata_verified", evidence)

        inferred = approval.verify_offline_gate_availability(
            root=ROOT,
            provider=self.provider,
            now="2026-08-15T00:00:00Z",
        )
        self.assertEqual(inferred["implementation_commit"], BASE)

    def test_both_protected_boundaries_reverify_all_remote_evidence(self) -> None:
        initial = self.verify()
        for checkpoint in sorted(approval.REVERIFY_CHECKPOINTS):
            with self.subTest(checkpoint=checkpoint):
                current = self.reverify(initial, checkpoint)
                self.assertEqual(current["verification_checkpoint"], checkpoint)
                self.assertEqual(
                    current["original_evidence_digest"],
                    initial["evidence_digest"],
                )
                self.assertNotEqual(current["evidence_digest"], initial["evidence_digest"])

    def test_reverify_refetches_every_bound_file_and_registry(self) -> None:
        initial = self.verify()
        self.provider.reset_calls()
        self.reverify(initial, "BEFORE_CANDIDATE_OPEN")
        self.assertEqual(
            self.provider.calls,
            {
                "repository": 1,
                "branch_ref": 2,
                "compare": 1,
                "file_contents": len(contract.RUNTIME_MATERIAL_PATHS) + 2,
                "issue_comment": 1,
            },
        )
        self.assertEqual(self.provider.file_content_paths[0], github_approval.APPROVERS_PATH)
        self.assertEqual(self.provider.file_content_paths[-1], approval.REGISTRY_PATH)
        self.assertEqual(
            set(self.provider.file_content_paths[1:-1]),
            set(contract.RUNTIME_MATERIAL_PATHS.values()),
        )

    def test_wrong_keyword_preseal_edit_bot_and_nonallowlisted_user_fail_close(self) -> None:
        cases = [
            {
                "body": f"APPROVED_TO_RUN {self.run_scope['run_scope_digest']}",
                "message": "exactly equal",
            },
            {
                "created_at": "2026-08-14T23:59:59Z",
                "message": "after",
            },
            {
                "updated_at": "2026-08-15T00:02:00Z",
                "message": "edited",
            },
            {
                "created_at": "2026-08-15 00:01:00Z",
                "updated_at": "2026-08-15 00:01:00Z",
                "message": "ISO-8601 UTC",
            },
            {
                "login": "codex-bot",
                "actor_type": "Bot",
                "message": "User",
            },
            {
                "login": "somebody-else",
                "message": "allowlisted",
            },
        ]
        for case in cases:
            message = case.pop("message")
            with self.subTest(message=message):
                self.add_comment(**case)
                with self.assertRaisesRegex(contract.ContractError, message):
                    self.verify()

    def test_comment_id_used_by_any_ordinary_grant_is_rejected(self) -> None:
        raw = _normalized_repository_bytes(ROOT / approval.REGISTRY_PATH)
        reused = {
            "schema_version": 2,
            "approval_evidence": {
                "approval_type": "APPROVED_TO_PREPARE",
                "comment_id": COMMENT_ID,
            },
        }
        self.provider.add_repository_file(
            approval.REGISTRY_PATH,
            raw + json.dumps(reused, separators=(",", ":")).encode("utf-8") + b"\n",
        )
        with self.assertRaisesRegex(contract.ContractError, "already used"):
            self.verify()

    def test_registry_is_strict_json_and_remote_only(self) -> None:
        duplicate_key = b'{"approval_evidence":null,"approval_evidence":null}\n'
        self.provider.add_repository_file(approval.REGISTRY_PATH, duplicate_key)
        with self.assertRaisesRegex(contract.ContractError, "duplicate"):
            self.verify()

    def test_remote_runtime_material_tamper_and_missing_file_fail_close(self) -> None:
        path = contract.RUNTIME_MATERIAL_PATHS["approval_verifier_blob_sha256"]
        self.provider.add_repository_file(path, b"tampered runtime material\n")
        with self.assertRaisesRegex(contract.ContractError, "content hash mismatch"):
            self.verify()
        del self.provider.contents_by_path[path]
        with self.assertRaisesRegex(contract.ContractError, "unavailable"):
            self.verify()

    def test_main_head_drift_is_rejected(self) -> None:
        moved = "f" * 40
        self.provider.ref["object"]["sha"] = moved
        self.provider.compare = {
            "url": (
                f"{github_approval.GITHUB_API_URL}/repos/"
                f"{contract.DEFAULT_REPOSITORY}/compare/{BASE}...{moved}"
            ),
            "status": "ahead",
            "base_commit": {"sha": BASE},
            "merge_base_commit": {"sha": BASE},
        }
        with self.assertRaisesRegex(contract.ContractError, "moved or differs"):
            self.verify()

    def test_reverify_detects_comment_and_stored_evidence_changes(self) -> None:
        initial = self.verify()
        self.provider.comment["updated_at"] = "2026-08-15T00:04:00Z"
        with self.assertRaisesRegex(contract.ContractError, "edited"):
            self.reverify(initial, "BEFORE_CANDIDATE_OPEN")

        self.add_comment()
        changed = copy.deepcopy(initial)
        changed["comment"]["author"] = "replacement"
        with self.assertRaisesRegex(contract.ContractError, "evidence changed"):
            self.reverify(changed, "BEFORE_CANDIDATE_OPEN")

    def test_reverify_refetch_rejects_forged_evidence_and_remote_tamper(self) -> None:
        initial = self.verify()
        forged = copy.deepcopy(initial)
        runtime_hashes = self.run_scope["runtime_bindings"]["runtime_material_sha256"]
        hash_by_path = {
            path: runtime_hashes[field]
            for field, path in contract.RUNTIME_MATERIAL_PATHS.items()
        }
        for item in forged["runtime_materials"]["materials"]:
            item["blob_sha"] = "d" * 40
            item["content_sha256"] = hash_by_path[item["path"]]
        nested = dict(forged["runtime_materials"])
        nested.pop("evidence_digest")
        forged["runtime_materials"]["evidence_digest"] = contract.canonical_digest(
            nested
        )
        forged["ordinary_registry"]["blob_sha"] = "e" * 40
        forged["ordinary_registry"]["content_sha256"] = "f" * 64
        forged["ordinary_registry"]["target_comment_id_unused"] = True
        unsigned = dict(forged)
        unsigned.pop("evidence_digest")
        forged["evidence_digest"] = contract.canonical_digest(unsigned)

        runtime_path = contract.RUNTIME_MATERIAL_PATHS["approval_verifier_blob_sha256"]
        self.provider.add_repository_file(runtime_path, b"tampered remote runtime\n")
        for checkpoint in sorted(approval.REVERIFY_CHECKPOINTS):
            with self.subTest(checkpoint=checkpoint, remote="runtime"):
                with self.assertRaisesRegex(contract.ContractError, "content hash mismatch"):
                    self.reverify(forged, checkpoint)

        self.provider.add_repository_file(
            runtime_path,
            _normalized_repository_bytes(ROOT / Path(*Path(runtime_path).parts)),
        )
        raw_registry = _normalized_repository_bytes(ROOT / approval.REGISTRY_PATH)
        reused = {
            "schema_version": 2,
            "approval_evidence": {
                "approval_type": "APPROVED_TO_RUN",
                "comment_id": COMMENT_ID,
            },
        }
        self.provider.add_repository_file(
            approval.REGISTRY_PATH,
            raw_registry
            + json.dumps(reused, separators=(",", ":")).encode("utf-8")
            + b"\n",
        )
        for checkpoint in sorted(approval.REVERIFY_CHECKPOINTS):
            with self.subTest(checkpoint=checkpoint, remote="registry"):
                with self.assertRaisesRegex(contract.ContractError, "already used"):
                    self.reverify(forged, checkpoint)

    def test_reverify_rejects_unknown_checkpoint_and_noninitial_input(self) -> None:
        initial = self.verify()
        with self.assertRaisesRegex(contract.ContractError, "checkpoint"):
            self.reverify(initial, "AFTER_SETTLEMENT_OPEN")
        reverified = self.reverify(initial, "BEFORE_CANDIDATE_OPEN")
        with self.assertRaisesRegex(contract.ContractError, "verification_checkpoint"):
            self.reverify(reverified, "BEFORE_RESULT_PUBLISH")


if __name__ == "__main__":
    unittest.main()
