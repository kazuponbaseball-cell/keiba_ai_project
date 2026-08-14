from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import github_approval
import registered_nonpromotion_contract_v1 as contract
import shared_g2_durable_ledger_v1 as g2


LANE_FILES = [
    ROOT / "scripts/research/registered_nonpromotion_contract_v1.py",
    ROOT / "scripts/research/registered_nonpromotion_authority_verifier_v1.py",
    ROOT / "scripts/research/registered_nonpromotion_catalog_v1.py",
    ROOT / "scripts/research/registered_nonpromotion_supervised_executor_v1.py",
    ROOT / "scripts/research/registered_nonpromotion_result_sealer_v1.py",
]


class RegisteredNonpromotionFirewallTests(unittest.TestCase):
    def test_ordinary_approval_keywords_are_unchanged(self) -> None:
        self.assertEqual(
            github_approval.APPROVAL_KEYWORDS,
            {"APPROVED_TO_PREPARE", "APPROVED_TO_RUN", "APPROVED_FOR_SHADOW"},
        )
        self.assertNotIn(contract.APPROVAL_KEYWORD, github_approval.APPROVAL_KEYWORDS)

    def test_lane_code_has_no_shell_network_credential_or_dynamic_eval_import(self) -> None:
        forbidden_import_roots = {
            "boto3",
            "ctypes",
            "httpx",
            "keyring",
            "requests",
            "secrets",
            "socket",
            "sqlite3",
            "subprocess",
        }
        forbidden_calls = {"eval", "exec", "compile", "__import__"}
        for path in LANE_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = {alias.name.split(".", 1)[0] for alias in node.names}
                    self.assertFalse(
                        roots & forbidden_import_roots,
                        f"{path} imports forbidden module(s) {roots & forbidden_import_roots}",
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    root = node.module.split(".", 1)[0]
                    self.assertNotIn(root, forbidden_import_roots, str(path))
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, forbidden_calls, str(path))

    def test_repository_policy_remains_non_authoritative(self) -> None:
        policy = json.loads(
            (ROOT / "research/REGISTERED_NONPROMOTION_DIAGNOSTIC_V1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(policy["authority"])
        self.assertEqual(policy["execution_status"], "EXECUTION_FORBIDDEN")
        self.assertFalse(policy["activation_contract"]["lane_activated"])
        self.assertFalse(policy["activation_contract"]["external_g2_configured"])
        self.assertFalse(policy["activation_contract"]["cutover_receipt_present"])
        self.assertFalse(policy["safety"]["formal_buy"])
        self.assertFalse(policy["safety"]["send_order"])
        self.assertEqual(policy["safety"]["stake"], 0)

    def test_default_g2_adapter_always_fails_closed(self) -> None:
        adapter = g2.UnconfiguredSharedG2Adapter()
        with self.assertRaises(g2.SharedG2Unavailable):
            adapter.fetch_current_head()
        with self.assertRaises(g2.SharedG2Unavailable):
            adapter.commit_transaction({})
        with self.assertRaises(g2.SharedG2Unavailable):
            adapter.verify_authenticated_payload(
                domain_separator="test", payload=b"{}", authentication={}
            )


if __name__ == "__main__":
    unittest.main()
