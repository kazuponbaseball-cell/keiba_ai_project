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
import registered_nonpromotion_authority_verifier_v1 as verifier
import registered_nonpromotion_contract_v1 as contract


BASE = "a" * 40
MAIN = "b" * 40
BLOB = "c" * 40
APPROVER = "kazuponbaseball-cell"


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
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()


class FakeProvider:
    def __init__(self) -> None:
        self.repository = {
            "full_name": contract.DEFAULT_REPOSITORY,
            "default_branch": contract.DEFAULT_BASE_BRANCH,
        }
        self.ref = {"ref": "refs/heads/main", "object": {"type": "commit", "sha": MAIN}}
        self.compare = {
            "url": (
                f"{github_approval.GITHUB_API_URL}/repos/{contract.DEFAULT_REPOSITORY}/compare/"
                f"{BASE}...{MAIN}"
            ),
            "status": "ahead",
            "base_commit": {"sha": BASE},
            "merge_base_commit": {"sha": BASE},
        }
        self.contents_by_path: dict[str, dict[str, object]] = {}
        self.add_repository_file(
            github_approval.APPROVERS_PATH,
            approvers_bytes(),
            blob_sha=BLOB,
        )
        repository_paths = {
            contract.POLICY_RELATIVE_PATH.as_posix(),
            "research/diagnostic_recipes/historical_ai_duplicate_gate_impact_v1.json",
            *contract.RUNTIME_MATERIAL_PATHS.values(),
            *contract.G2_MATERIAL_PATHS,
            *contract.EXPECTED_SCHEMA_PATHS.values(),
        }
        for path in sorted(repository_paths):
            raw = (ROOT / Path(*Path(path).parts)).read_bytes()
            if raw.startswith(b"\xef\xbb\xbf"):
                raise AssertionError(f"test repository material contains UTF-8 BOM: {path}")
            normalized = (
                raw.decode("utf-8")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .encode("utf-8")
            )
            self.add_repository_file(path, normalized)
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
            "content": base64.b64encode(content).decode(),
        }

    def get_repository(self, repository: str) -> dict[str, object]:
        return copy.deepcopy(self.repository)

    def get_branch_ref(self, repository: str, branch: str) -> dict[str, object]:
        return copy.deepcopy(self.ref)

    def compare_commits(self, repository: str, base: str, head: str) -> dict[str, object]:
        return copy.deepcopy(self.compare)

    def get_file_contents(self, repository: str, path: str, ref: str) -> dict[str, object]:
        if ref != BASE or path not in self.contents_by_path:
            raise KeyError(f"unavailable immutable repository material: {path}@{ref}")
        return copy.deepcopy(self.contents_by_path[path])

    def get_issue_comment(self, repository: str, comment_id: int) -> dict[str, object]:
        return copy.deepcopy(self.comment)


class RegisteredApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registered = contract.resolve_registered_recipe(
            ROOT,
            recipe_id="historical_ai_duplicate_gate_impact_v1",
            recipe_version=1,
        )
        self.provider = FakeProvider()
        approvers_sha = __import__("hashlib").sha256(approvers_bytes()).hexdigest()
        bindings = {
            "repository": contract.DEFAULT_REPOSITORY,
            "base_branch": contract.DEFAULT_BASE_BRANCH,
            "run_scope_base_commit": BASE,
            "verified_current_main_sha": MAIN,
            "approvers_blob_sha": BLOB,
            "approvers_content_sha256": approvers_sha,
            "activation_receipt_sha256": "1" * 64,
            "cutover_receipt_sha256": "1" * 64,
            "schema_bundle_sha256": self.registered.schema_bundle_digest,
            "approval_evidence_schema_sha256": "3" * 64,
            "capability_profile_sha256": "4" * 64,
            "policy_blob_sha256": self.registered.policy_file_sha256,
            "recipe_blob_sha256": self.registered.recipe_file_sha256,
            "compiler_blob_sha256": "5" * 64,
            "authority_verifier_blob_sha256": "6" * 64,
            "catalog_validator_blob_sha256": "7" * 64,
            "executor_blob_sha256": "8" * 64,
            "runner_blob_sha256": "9" * 64,
            "result_sealer_blob_sha256": "a" * 64,
            "g2_authority_service_blob_sha256": "b" * 64,
            "phase_lease_schema_sha256": "c" * 64,
            "phase_operation_receipt_schema_sha256": "d" * 64,
            "environment_manifest_sha256": "e" * 64,
            "catalog_release_id": "catalog_release_v1",
            "catalog_release_sha256": "8" * 64,
            "catalog_release_status": "ACTIVE",
            "catalog_release_revoked": False,
            "catalog_status_receipt_sha256": "f" * 64,
            "candidate_entry_sha256": "9" * 64,
            "candidate_schema_sha256": "a" * 64,
            "candidate_provenance_sha256": "b" * 64,
            "p_action_cross_source_equality_attestation_sha256": "c" * 64,
            "candidate_materializer_usecols_sha256": "d" * 64,
            "decision_base_lineage_sha256": "e" * 64,
            "settlement_entry_sha256": "a" * 64,
            "settlement_schema_sha256": "f" * 64,
            "settlement_provenance_sha256": "1" * 64,
            "official_settlement_provenance_sha256": "1" * 64,
            "cohort_manifest_sha256": "b" * 64,
            "ordered_race_set_sha256": "c" * 64,
            "output_root": "outputs/research/RND-001",
            "sealed_at": "2026-08-15T00:00:00Z",
            "expected_pregrant_global_head": "d" * 64,
            "expected_pregrant_subject_head": "e" * 64,
            "cutover_epoch": 1,
            "external_witness_checkpoint_sha256": "1" * 64,
        }
        bindings.update(self.registered.runtime_material_digests)
        self.run_scope = contract.compile_run_scope(self.registered, bindings)
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
        issue = 42
        comment_id = 9001
        self.provider.comment = {
            "id": comment_id,
            "issue_url": (
                f"https://api.github.com/repos/{contract.DEFAULT_REPOSITORY}/issues/{issue}"
            ),
            "html_url": (
                f"https://github.com/{contract.DEFAULT_REPOSITORY}/issues/{issue}"
                f"#issuecomment-{comment_id}"
            ),
            "user": {"login": login, "type": actor_type},
            "body": body
            or f"{contract.APPROVAL_KEYWORD} {self.run_scope['run_scope_digest']}",
            "created_at": created_at,
            "updated_at": updated_at or created_at,
        }

    def verify(self) -> dict[str, object]:
        return verifier.verify_registered_run_approval(
            provider=self.provider,
            registered=self.registered,
            run_scope=self.run_scope,
            issue_number=42,
            comment_id=9001,
            clock=lambda: "2026-08-15T00:02:00Z",
        )

    def test_exact_lane_comment_is_verified_but_not_authority(self) -> None:
        evidence = self.verify()
        self.assertFalse(evidence["authority"])
        self.assertTrue(evidence["global_g2_reservation_required"])
        self.assertEqual(evidence["comment"]["approval_type"], contract.APPROVAL_KEYWORD)

    def test_self_rehashed_subject_alias_is_rejected_before_approval(self) -> None:
        changed = copy.deepcopy(self.run_scope)
        changed["semantic_subject"]["cohort_id"] = "alias_cohort"
        changed["semantic_subject_digest"] = contract.canonical_digest(
            changed["semantic_subject"]
        )
        unsigned = dict(changed)
        unsigned.pop("run_scope_digest")
        changed["run_scope_digest"] = contract.canonical_digest(unsigned)
        with self.assertRaisesRegex(contract.ContractError, "canonical"):
            verifier.verify_registered_run_approval(
                provider=self.provider,
                registered=self.registered,
                run_scope=changed,
                issue_number=42,
                comment_id=9001,
                clock=lambda: "2026-08-15T00:02:00Z",
            )

    def test_ordinary_keyword_is_not_accepted_for_lane(self) -> None:
        self.add_comment(body=f"APPROVED_TO_RUN {self.run_scope['run_scope_digest']}")
        with self.assertRaisesRegex(contract.ContractError, "exactly equal"):
            self.verify()

    def test_comment_must_follow_scope_seal(self) -> None:
        self.add_comment(created_at="2026-08-14T23:59:59Z")
        with self.assertRaisesRegex(contract.ContractError, "after"):
            self.verify()

    def test_edited_or_bot_comment_is_rejected(self) -> None:
        self.add_comment(updated_at="2026-08-15T00:02:00Z")
        with self.assertRaisesRegex(contract.ContractError, "edited"):
            self.verify()
        self.add_comment(login="codex-bot", actor_type="Bot")
        with self.assertRaisesRegex(contract.ContractError, "User"):
            self.verify()

    def test_remote_comment_change_is_detected_on_reverify(self) -> None:
        evidence = self.verify()
        self.provider.comment["body"] = "deleted/replaced"
        with self.assertRaises(contract.ContractError):
            verifier.reverify_registered_run_approval(
                provider=self.provider,
                registered=self.registered,
                run_scope=self.run_scope,
                stored_evidence=evidence,
                clock=lambda: "2026-08-15T00:03:00Z",
            )

    def test_main_head_drift_is_rejected(self) -> None:
        self.provider.ref["object"]["sha"] = "f" * 40
        self.provider.compare["url"] = (
            f"{github_approval.GITHUB_API_URL}/repos/{contract.DEFAULT_REPOSITORY}/compare/"
            f"{BASE}...{'f' * 40}"
        )
        with self.assertRaisesRegex(contract.ContractError, "differs"):
            self.verify()

    def test_remote_runtime_material_tamper_is_rejected(self) -> None:
        path = contract.RUNTIME_MATERIAL_PATHS["compiler_blob_sha256"]
        self.provider.add_repository_file(path, b"tampered runtime material\n")
        with self.assertRaisesRegex(contract.ContractError, "content hash mismatch"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
