from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import unittest
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import registered_nonpromotion_catalog_v1 as catalog
import registered_nonpromotion_contract_v1 as contract
import registered_nonpromotion_result_sealer_v1 as result_sealer
import shared_g2_durable_ledger_v1 as ledger
import shared_g2_lease_authority_v1 as lease_authority


class SchemaValidationError(AssertionError):
    pass


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise SchemaValidationError(f"duplicate key in {path}: {key}")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise SchemaValidationError(f"non-finite constant in {path}: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{path} must contain one JSON object")
    return value


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(
        left, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) == json.dumps(right, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return type(value) is bool
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return (type(value) is int) or isinstance(value, float)
    if expected == "null":
        return value is None
    raise SchemaValidationError(f"unsupported test-validator type: {expected}")


def _resolve_ref(document: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise SchemaValidationError(f"non-local schema reference: {reference}")
    current: Any = document
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        current = current[token]
    if not isinstance(current, Mapping):
        raise SchemaValidationError(f"schema reference is not an object: {reference}")
    return current


def _validate(
    value: Any,
    schema: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    path: str = "$",
) -> None:
    if "$ref" in schema:
        _validate(value, _resolve_ref(document, schema["$ref"]), document, path=path)
        return

    if "type" in schema and not _type_matches(value, schema["type"]):
        raise SchemaValidationError(f"{path}: expected {schema['type']}")
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise SchemaValidationError(f"{path}: const mismatch")
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        raise SchemaValidationError(f"{path}: enum mismatch")

    for child in schema.get("allOf", []):
        _validate(value, child, document, path=path)
    if "oneOf" in schema:
        matches = 0
        for child in schema["oneOf"]:
            try:
                _validate(value, child, document, path=path)
            except SchemaValidationError:
                continue
            matches += 1
        if matches != 1:
            raise SchemaValidationError(f"{path}: oneOf matched {matches} branches")
    if "not" in schema:
        try:
            _validate(value, schema["not"], document, path=path)
        except SchemaValidationError:
            pass
        else:
            raise SchemaValidationError(f"{path}: forbidden by not")
    if "if" in schema:
        try:
            _validate(value, schema["if"], document, path=path)
        except SchemaValidationError:
            branch = schema.get("else")
        else:
            branch = schema.get("then")
        if branch is not None:
            _validate(value, branch, document, path=path)

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = sorted(set(required) - set(value))
        if missing:
            raise SchemaValidationError(f"{path}: missing fields {missing}")
        properties = schema.get("properties", {})
        pattern_properties = schema.get("patternProperties", {})
        for key, item in value.items():
            matched = False
            if key in properties:
                _validate(item, properties[key], document, path=f"{path}.{key}")
                matched = True
            for pattern, child in pattern_properties.items():
                if re.search(pattern, key):
                    _validate(item, child, document, path=f"{path}.{key}")
                    matched = True
            if not matched and schema.get("additionalProperties") is False:
                raise SchemaValidationError(f"{path}: additional property {key}")
        if len(value) < schema.get("minProperties", 0):
            raise SchemaValidationError(f"{path}: too few properties")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise SchemaValidationError(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaValidationError(f"{path}: too many items")
        if schema.get("uniqueItems"):
            canonical = [
                json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for item in value
            ]
            if len(canonical) != len(set(canonical)):
                raise SchemaValidationError(f"{path}: duplicate array item")
        if "items" in schema:
            item_schema = schema["items"]
            if isinstance(item_schema, list):
                for index, item in enumerate(value[: len(item_schema)]):
                    _validate(item, item_schema[index], document, path=f"{path}[{index}]")
                if len(value) > len(item_schema) and schema.get("additionalItems") is False:
                    raise SchemaValidationError(f"{path}: additional tuple item")
            else:
                for index, item in enumerate(value):
                    _validate(item, item_schema, document, path=f"{path}[{index}]")
        if "contains" in schema:
            if not any(
                _valid(item, schema["contains"], document, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ):
                raise SchemaValidationError(f"{path}: contains did not match")

    if isinstance(value, str):
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise SchemaValidationError(f"{path}: pattern mismatch")
        if schema.get("format") == "date-time":
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise SchemaValidationError(f"{path}: invalid date-time") from exc

    if (type(value) is int) or isinstance(value, float):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path}: above maximum")


def _valid(
    value: Any,
    schema: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    path: str,
) -> bool:
    try:
        _validate(value, schema, document, path=path)
    except SchemaValidationError:
        return False
    return True


class RegisteredNonpromotionSchemaV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = _strict_json(
            ROOT / "research" / "REGISTERED_NONPROMOTION_DIAGNOSTIC_V1.json"
        )
        cls.recipe = _strict_json(
            ROOT
            / "research"
            / "diagnostic_recipes"
            / "historical_ai_duplicate_gate_impact_v1.json"
        )
        cls.schemas = {
            name: _strict_json(ROOT / path)
            for name, path in cls.policy["schema_paths"].items()
        }
        cls.registered = contract.resolve_registered_recipe(
            ROOT,
            recipe_id="historical_ai_duplicate_gate_impact_v1",
            recipe_version=1,
        )

    def bindings(self) -> dict[str, Any]:
        bindings: dict[str, Any] = {key: "1" * 64 for key in contract.RUN_BINDING_KEYS}
        bindings.update(
            {
                "repository": contract.DEFAULT_REPOSITORY,
                "base_branch": contract.DEFAULT_BASE_BRANCH,
                "run_scope_base_commit": "a" * 40,
                "verified_current_main_sha": "b" * 40,
                "approvers_blob_sha": "c" * 40,
                "catalog_release_id": "catalog_release_v1",
                "catalog_release_status": "ACTIVE",
                "catalog_release_revoked": False,
                "output_root": "outputs/research/RND-SCHEMA-FIXTURE",
                "sealed_at": "2026-08-15T00:00:00Z",
                "cutover_epoch": 1,
            }
        )
        bindings.update(self.registered.runtime_material_digests)
        return bindings

    def test_schema_registry_is_exact_and_all_documents_are_closed_draft_07(self) -> None:
        self.assertEqual(self.policy["schema_paths"], contract.EXPECTED_SCHEMA_PATHS)
        self.assertEqual(len(self.schemas), 14)
        for name, schema in self.schemas.items():
            with self.subTest(schema=name):
                self.assertEqual(
                    schema["$schema"], "http://json-schema.org/draft-07/schema#"
                )
                self.assertIs(schema.get("additionalProperties"), False)
                self.assertEqual(set(schema["required"]), set(schema["properties"]))
                for definition_name, definition in schema.get("definitions", {}).items():
                    if definition.get("type") == "object" and "properties" in definition:
                        self.assertIs(
                            definition.get("additionalProperties"),
                            False,
                            f"{name}.{definition_name} is not closed",
                        )
                        self.assertEqual(
                            set(definition.get("required", [])),
                            set(definition["properties"]),
                            f"{name}.{definition_name} fields are not exact",
                        )

    def test_current_policy_and_registered_recipe_validate(self) -> None:
        _validate(self.policy, self.schemas["policy"], self.schemas["policy"])
        _validate(self.recipe, self.schemas["recipe"], self.schemas["recipe"])

        mutated = copy.deepcopy(self.policy)
        mutated["authority"] = True
        with self.assertRaises(SchemaValidationError):
            _validate(mutated, self.schemas["policy"], self.schemas["policy"])

        mutated = copy.deepcopy(self.recipe)
        mutated["unregistered_formula"] = "arbitrary"
        with self.assertRaises(SchemaValidationError):
            _validate(mutated, self.schemas["recipe"], self.schemas["recipe"])

    def test_compiler_run_scope_and_schema_runtime_bindings_are_exact(self) -> None:
        run_scope = contract.compile_run_scope(self.registered, self.bindings())
        _validate(run_scope, self.schemas["run_scope"], self.schemas["run_scope"])

        binding_schema = self.schemas["run_scope"]["definitions"]["runtime_bindings"]
        self.assertEqual(set(binding_schema["required"]), contract.RUN_BINDING_KEYS)
        self.assertEqual(set(binding_schema["properties"]), contract.RUN_BINDING_KEYS)
        self.assertEqual(
            set(self.schemas["run_scope"]["definitions"]["semantic_subject"]["required"]),
            set(run_scope["semantic_subject"]),
        )
        self.assertEqual(
            set(self.schemas["run_scope"]["definitions"]["exact_subject"]["required"]),
            set(run_scope["exact_subject"]),
        )

        mutated = copy.deepcopy(run_scope)
        mutated["runtime_bindings"]["free_form_command"] = "python arbitrary.py"
        with self.assertRaises(SchemaValidationError):
            _validate(mutated, self.schemas["run_scope"], self.schemas["run_scope"])

    def test_catalog_and_shared_g2_wire_field_sets_match_canonical_modules(self) -> None:
        catalog_schema = self.schemas["catalog_release"]
        self.assertEqual(set(catalog_schema["required"]), catalog._RELEASE_FIELDS)
        self.assertEqual(
            set(catalog_schema["definitions"]["catalog_entry"]["required"]),
            catalog._ENTRY_FIELDS,
        )

        lease_schema = self.schemas["phase_lease"]
        self.assertEqual(
            set(lease_schema["definitions"]["phase_lease"]["required"]),
            set(lease_authority.PHASE_LEASE_FIELDS),
        )
        self.assertEqual(
            set(lease_schema["definitions"]["lease_binding"]["required"]),
            set(lease_authority.LEASE_BINDING_FIELDS),
        )

        output_schema = self.schemas["phase_output_seal_receipt"]
        self.assertEqual(
            set(output_schema["definitions"]["phase_output_seal_receipt"]["required"]),
            set(lease_authority.PHASE_OUTPUT_SEAL_FIELDS),
        )
        decision_lease_schema = self.schemas["decision_lease_batch"]
        self.assertEqual(
            set(decision_lease_schema["definitions"]["decision_lease_batch"]["required"]),
            set(lease_authority.DECISION_LEASE_BATCH_FIELDS),
        )
        self.assertEqual(
            set(decision_lease_schema["definitions"]["decision_lease_entry"]["required"]),
            set(lease_authority.DECISION_LEASE_ENTRY_FIELDS),
        )
        decision_consumption_schema = self.schemas["decision_consumption_batch"]
        self.assertEqual(
            set(decision_consumption_schema["definitions"]["decision_consumption_batch"]["required"]),
            set(lease_authority.DECISION_CONSUMPTION_BATCH_FIELDS),
        )
        self.assertEqual(
            set(decision_consumption_schema["definitions"]["decision_consumption_entry"]["required"]),
            set(lease_authority.DECISION_CONSUMPTION_ENTRY_FIELDS),
        )
        self.assertEqual(
            set(self.schemas["lease_consumption_receipt"]["definitions"]["lease_consumption_receipt"]["required"]),
            set(lease_authority.LEASE_CONSUMPTION_FIELDS),
        )
        self.assertEqual(
            set(self.schemas["phase_output_attestation"]["definitions"]["phase_output_attestation"]["required"]),
            set(lease_authority.PHASE_OUTPUT_ATTESTATION_FIELDS),
        )
        self.assertEqual(
            set(self.schemas["result"]["required"]),
            set(result_sealer.SEALED_RESULT_FIELDS),
        )

        authority_schema = self.schemas["authority_receipt"]
        transaction = authority_schema["definitions"]["transaction_receipt"]
        self.assertEqual(set(authority_schema["required"]), set(ledger.ENVELOPE_FIELDS))
        self.assertEqual(
            set(transaction["required"]), set(ledger.TRANSACTION_RECEIPT_FIELDS)
        )
        self.assertEqual(
            set(authority_schema["definitions"]["global_head"]["required"]),
            set(ledger.GLOBAL_HEAD_FIELDS),
        )
        self.assertEqual(
            set(authority_schema["definitions"]["subject_head"]["required"]),
            set(ledger.SUBJECT_HEAD_FIELDS),
        )
        self.assertEqual(
            set(transaction["properties"]["operation_kind"]["enum"]),
            set(ledger.ALLOWED_OPERATION_KINDS),
        )
        self.assertEqual(
            set(authority_schema["definitions"]["subject_head"]["properties"]["subject_kind"]["enum"]),
            set(ledger.SUBJECT_KINDS),
        )
        snapshot_schema = self.schemas["subject_head_snapshot"]
        self.assertEqual(set(snapshot_schema["required"]), set(ledger.ENVELOPE_FIELDS))
        self.assertEqual(
            snapshot_schema["properties"]["payload_type"]["const"],
            ledger.SUBJECT_HEAD_SNAPSHOT_KIND,
        )
        self.assertEqual(
            set(snapshot_schema["definitions"]["subject_head_snapshot"]["required"]),
            set(ledger.SUBJECT_HEAD_SNAPSHOT_FIELDS),
        )
        self.assertEqual(
            set(snapshot_schema["definitions"]["global_head"]["required"]),
            set(ledger.GLOBAL_HEAD_FIELDS),
        )
        self.assertEqual(
            set(snapshot_schema["definitions"]["subject_head"]["required"]),
            set(ledger.SUBJECT_HEAD_FIELDS),
        )
        self.assertEqual(
            set(snapshot_schema["definitions"]["subject_head"]["properties"]["subject_kind"]["enum"]),
            set(ledger.SUBJECT_KINDS),
        )
        self.assertEqual(
            set(self.schemas["cutover_receipt"]["definitions"]["cutover_payload"]["required"]),
            set(ledger.CUTOVER_RECEIPT_FIELDS),
        )
        for schema_name in (
            "cutover_receipt",
            "subject_head_snapshot",
            "phase_lease",
            "decision_lease_batch",
            "decision_consumption_batch",
            "lease_consumption_receipt",
            "phase_output_attestation",
            "phase_output_seal_receipt",
        ):
            self.assertEqual(
                set(self.schemas[schema_name]["required"]),
                set(ledger.ENVELOPE_FIELDS),
            )

    def test_security_negative_constraints_fail_closed(self) -> None:
        for schema_name in (
            "authority_receipt",
            "cutover_receipt",
            "subject_head_snapshot",
            "phase_lease",
            "decision_lease_batch",
            "decision_consumption_batch",
            "lease_consumption_receipt",
            "phase_output_attestation",
            "phase_output_seal_receipt",
        ):
            with self.subTest(schema=schema_name, attack="mixed-case-local-auth"):
                schema = self.schemas[schema_name]
                scheme_schema = schema["definitions"]["authentication"]["properties"][
                    "scheme"
                ]
                with self.assertRaises(SchemaValidationError):
                    _validate("LoCaL", scheme_schema, schema)

        catalog_entry = self.schemas["catalog_release"]["definitions"][
            "catalog_entry"
        ]
        self.assertEqual(catalog_entry["properties"]["row_count"]["const"], 3746)
        self.assertEqual(catalog_entry["properties"]["race_count"]["const"], 3746)
        phase_binding = self.schemas["phase_lease"]["definitions"]["lease_binding"]
        self.assertEqual(phase_binding["properties"]["attempt"]["const"], 1)
        self.assertEqual(phase_binding["properties"]["ttl_seconds"]["maximum"], 3600)

    def test_authenticated_subject_head_snapshot_schema_is_closed(self) -> None:
        digest = "1" * 64
        timestamp = "2026-08-15T00:00:00Z"
        safety = {"formal_buy": False, "send_order": False, "stake": 0}
        global_head = {
            "schema_version": 1,
            "object_kind": ledger.GLOBAL_HEAD_KIND,
            "authority_id": "shared-g2-authority",
            "activation_epoch": "epoch-1",
            "backend_identity_digest": digest,
            "cutover_receipt_digest": digest,
            "sequence": 9,
            "head_digest": digest,
            "observed_at": timestamp,
            "safety": safety,
        }
        snapshot = {
            "schema_version": 1,
            "envelope_kind": ledger.AUTHENTICATED_ENVELOPE_KIND,
            "payload_type": ledger.SUBJECT_HEAD_SNAPSHOT_KIND,
            "payload_digest": digest,
            "payload": {
                "schema_version": 1,
                "snapshot_kind": ledger.SUBJECT_HEAD_SNAPSHOT_KIND,
                "authority_id": "shared-g2-authority",
                "activation_epoch": "epoch-1",
                "backend_identity_digest": digest,
                "cutover_receipt_digest": digest,
                "global_head": global_head,
                "subject_head": {
                    "subject_kind": "PHASE_OUTPUT",
                    "subject_digest": digest,
                    "generation": 0,
                    "sequence": 1,
                    "head_digest": digest,
                    "state_digest": digest,
                },
                "read_at": timestamp,
                "safety": safety,
            },
            "authentication": {
                "scheme": "REMOTE_HSM",
                "key_id": "remote-key-1",
                "signature": "A" * 43,
                "attestation_digest": digest,
                "issued_at": timestamp,
            },
        }
        schema = self.schemas["subject_head_snapshot"]
        _validate(snapshot, schema, schema)

        mutated = copy.deepcopy(snapshot)
        mutated["payload"]["unbound_local_hint"] = "forbidden"
        with self.assertRaises(SchemaValidationError):
            _validate(mutated, schema, schema)

    def test_decision_batch_schemas_require_exact_ordered_two_replica_entries(self) -> None:
        def binding(replica_id: str) -> dict[str, Any]:
            digest = "1" * 64
            return {
                "run_scope_digest": digest,
                "run_generation": 1,
                "recipe_digest": digest,
                "semantic_subject_digest": digest,
                "exact_run_subject_digest": digest,
                "question_family_digest": digest,
                "replica_id": replica_id,
                "phase": "DECISION_FREEZE",
                "attempt": 1,
                "predecessor_receipt_digest": digest,
                "issue_revalidation_receipt_digest": digest,
                "policy_digest": digest,
                "schema_digest": digest,
                "verifier_digest": digest,
                "executor_digest": digest,
                "runner_digest": digest,
                "bound_capability_profile_digest": digest,
                "phase_capability_digest": "77bcf04fa806343dca20a7694c8d2ee19bae80d039cb81a117eb5d0fde4dd531",
                "ttl_seconds": 60,
            }

        def lease_entry(replica_id: str) -> dict[str, Any]:
            return {
                "lease_id": f"lease-{replica_id}",
                "binding": binding(replica_id),
                "binding_digest": "2" * 64,
                "status": "ISSUED",
                "retry_budget": 0,
                "issued_at": "2026-08-15T00:00:00Z",
                "expires_at": "2026-08-15T00:01:00Z",
            }

        lease_schema = self.schemas["decision_lease_batch"]
        leases_contract = lease_schema["definitions"]["decision_lease_batch"][
            "properties"
        ]["leases"]
        canonical = [lease_entry("clean_a"), lease_entry("clean_b")]
        _validate(canonical, leases_contract, lease_schema)
        with self.assertRaises(SchemaValidationError):
            _validate(list(reversed(canonical)), leases_contract, lease_schema)
        poisoned = copy.deepcopy(canonical)
        poisoned[0]["free_form_command"] = "python arbitrary.py"
        with self.assertRaises(SchemaValidationError):
            _validate(poisoned, leases_contract, lease_schema)

        consumption_schema = self.schemas["decision_consumption_batch"]
        consumptions_contract = consumption_schema["definitions"][
            "decision_consumption_batch"
        ]["properties"]["leases"]
        consumptions = [
            {
                "replica_id": replica_id,
                "lease_id": f"lease-{replica_id}",
                "lease_payload_digest": "3" * 64,
                "binding_digest": "4" * 64,
                "dispatch_digest": "5" * 64,
            }
            for replica_id in ("clean_a", "clean_b")
        ]
        _validate(consumptions, consumptions_contract, consumption_schema)
        with self.assertRaises(SchemaValidationError):
            _validate(list(reversed(consumptions)), consumptions_contract, consumption_schema)


if __name__ == "__main__":
    unittest.main()
