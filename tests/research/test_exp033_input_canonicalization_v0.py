from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/research/run_exp033_input_canonicalization_v0.py"
CONFIG_PATH = ROOT / "research/configs/EXP-20260821-034.input_canonicalization_v0.json"
SPEC = importlib.util.spec_from_file_location("exp034_input_canonicalizer", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "e" * 64
PREDICTION_TIME = "2026-08-23T06:00:00Z"
SOURCE_TIME = "2026-08-20T00:00:00Z"
RECEIVED_TIME = "2026-08-20T00:01:00Z"
AVAILABLE_TIME = "2026-08-20T00:02:00Z"


def _bundle():
    return runner.load_and_verify_contract(CONFIG_PATH)


def _runner_source_payload(row):
    return runner.canonical_runner_source_payload(
        row,
        "synthetic/declared_runner_source.jsonl",
    )


def _runner_feature_evidence(row, source_payload):
    payload = runner.runner_feature_dependency_payload(source_payload)
    digest = runner.source_payload_hash(payload)
    return {
        "canonical_source_payload": payload,
        "content_sha256": digest,
        "source_path": source_payload["source_path"],
        "source_event_time": row["source_event_time"],
        "received_at": row["received_at"],
        "available_as_of": row["available_as_of"],
        "source_version": row["source_version"],
    }


def _event_source_evidence(row, source_payload):
    return {
        "canonical_source_payload": source_payload,
        "content_sha256": runner.source_payload_hash(source_payload),
        "source_path": source_payload["source_path"],
        "source_event_time": row["source_event_time"],
        "received_at": row["received_at"],
        "available_as_of": row["available_as_of"],
        "source_version": row["source_version"],
    }


def _runner_fixture(bundle, counts=(14, 14, 10, 16, 16)):
    rows = []
    payloads = {}
    global_ordinal = 0
    for race_ordinal, count in enumerate(counts, start=1):
        race_id = f"SYN-RACE-{race_ordinal:02d}"
        for horse_ordinal in range(1, count + 1):
            global_ordinal += 1
            horse_id = f"SYN-HORSE-{global_ordinal:03d}"
            confirmed = race_ordinal in {3, 5}
            row = {
                "release_family_id": "SYN-RUNNER-UNIVERSE",
                "release_version": 1,
                "parent_manifest_digest": None,
                "as_of": PREDICTION_TIME,
                "race_id": race_id,
                "horse_id": horse_id,
                "horse_name": f"Synthetic Horse {global_ordinal:03d}",
                "race_name": f"Synthetic Race {race_ordinal:02d}",
                "draw_status": "confirmed" if confirmed else "scheduled_pending_draw",
                "entry_stage": "declared_with_draw" if confirmed else "declared_without_draw",
                "runner_status": "declared_active",
                "frame_number": ((horse_ordinal - 1) // 2) + 1 if confirmed else None,
                "horse_number": horse_ordinal if confirmed else None,
                "jockey_id": f"SYN-JOCKEY-{horse_ordinal:03d}",
                "jockey_name": f"Synthetic Jockey {horse_ordinal:03d}",
                "carried_weight": 55.0,
                "trainer_id": f"SYN-TRAINER-{horse_ordinal:03d}",
                "trainer_name": f"Synthetic Trainer {horse_ordinal:03d}",
                "active_for_feature_materialization": True,
                "change_reason": "initial_declaration",
                "source_event_time": SOURCE_TIME,
                "received_at": RECEIVED_TIME,
                "available_as_of": AVAILABLE_TIME,
                "source_version": "SYN-CARD-v1",
                "source_content_sha256": "",
                "missing_reason": "not_applicable" if confirmed else "not_declared_by_source",
                "row_payload_sha256": "",
            }
            self_order = tuple(row)
            assert self_order == bundle.runner_columns
            payload = _runner_source_payload(row)
            rows.append(runner.seal_runner_row(row, payload))
            payloads[(race_id, horse_id)] = payload
    return tuple(rows), payloads


def _manifest_payload(
    kind,
    *,
    row_count=1,
    race_count=1,
    runner_count=1,
    artifact_name=None,
    artifact_hash=HASH_A,
    artifact_path=None,
    version=1,
    parent=None,
    release_family_id="SYN-RELEASE",
    as_of=PREDICTION_TIME,
    source_cutoff=AVAILABLE_TIME,
):
    artifact_name = artifact_name or runner.ARTIFACT_NAME_BY_KIND[kind]
    jsonl_kinds = {
        "runner_universe_manifest",
        "training_source_manifest",
        "target_source_manifest",
        "feature_release_manifest",
        "lineage_manifest",
        "label_eligibility_manifest",
        "release_diff_manifest",
    }
    suffix = "jsonl" if kind in jsonl_kinds else "json"
    artifact_path = artifact_path or f"outputs/research/SYN/{kind}.{suffix}"
    artifacts = [
        {
            "name": artifact_name,
            "path": artifact_path,
            "sha256": artifact_hash,
            "row_count": row_count,
        }
    ]
    return {
        "experiment_id": runner.EXPERIMENT_ID,
        "manifest_kind": kind,
        "release_family_id": release_family_id,
        "release_version": version,
        "parent_manifest_digest": parent,
        "as_of": as_of,
        "source_cutoff": source_cutoff,
        "generator_execution_commit": "d" * 40,
        "generator_script_sha256": HASH_A,
        "config_sha256": HASH_B,
        "dependency_environment_sha256": HASH_C,
        "schema_sha256": HASH_A,
        "input_source_paths_and_sha256": [
            {"path": "synthetic/source.json", "sha256": HASH_B}
        ],
        "output_artifact_paths_and_sha256": [
            {"path": artifact_path, "sha256": artifact_hash}
        ],
        "row_count": row_count,
        "race_count": race_count,
        "runner_count": runner_count,
        "duplicate_count": 0,
        "row_counts": {"total": row_count},
        "race_counts": {"total": race_count},
        "identity_counts": {"runner_count": runner_count},
        "duplicate_and_missing_counts": {"duplicate_key_count": 0, "missing_key_count": 0},
        "missing_reason_distribution": {"not_applicable": row_count},
        "as_of_verdict_counts": {"certified_asof_safe": row_count},
        "label_eligibility_counts": {"not_applicable": row_count},
        "artifacts": artifacts,
        "source_time_completeness": 1.0,
        "source_hash_completeness": 1.0,
        "certification_status": "certified",
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


_ARTIFACT_BYTES_BY_SHA = {}


def _synthetic_manifest_identities(race_count, runner_count):
    if runner_count == 0:
        return []
    assert race_count > 0
    if (race_count, runner_count) == (5, 70):
        counts = (14, 14, 10, 16, 16)
        return [
            (f"SYN-MANIFEST-RACE-{race_ordinal:02d}", f"SYN-MANIFEST-HORSE-{horse_global:03d}")
            for race_ordinal, count in enumerate(counts, start=1)
            for horse_global in range(sum(counts[: race_ordinal - 1]) + 1, sum(counts[:race_ordinal]) + 1)
        ]
    return [
        (f"SYN-MANIFEST-RACE-{(ordinal % race_count) + 1:02d}", f"SYN-MANIFEST-HORSE-{ordinal + 1:03d}")
        for ordinal in range(runner_count)
    ]


def _semantic_artifact_bytes(payload, bundle, *, lineage_hash=HASH_A):
    kind = payload["manifest_kind"]
    path = payload["artifacts"][0]["path"]
    source_path = payload["input_source_paths_and_sha256"][0]["path"]
    if kind in {"training_source_manifest", "label_eligibility_manifest"}:
        identities = [
            ("SYN-TRAINING-RACE-01", f"SYN-TRAINING-HORSE-{ordinal:03d}")
            for ordinal in range(1, payload["runner_count"] + 1)
        ]
    else:
        identities = _synthetic_manifest_identities(payload["race_count"], payload["runner_count"])
    per_race_ordinal = {}
    card_core_by_identity = {}
    for race_id, horse_id in identities:
        number = per_race_ordinal.get(race_id, 0) + 1
        per_race_ordinal[race_id] = number
        card_core_by_identity[(race_id, horse_id)] = {
            "年齢": 4.0,
            "斤量": 55.0,
            "距離": 1600.0,
            "場所": "SYN-VENUE",
            "性別": "牡",
            "騎手コード": f"SYN-J-{number:03d}",
            "調教師コード": f"SYN-T-{number:03d}",
            "芝・ダ": "芝",
            "クラス名": "SYN-CLASS",
            "トラックコード": "SYN-TRACK",
        }
    field_size_by_race = {
        race_id: sum(identity_race == race_id for identity_race, _horse_id in identities)
        for race_id, _horse_id in identities
    }

    def declared_card(identity, *, training=False):
        race_id, horse_id = identity
        prediction_time = "2026-08-01T06:00:00Z" if training else PREDICTION_TIME
        source_time = "2026-07-31T00:00:00Z" if training else SOURCE_TIME
        received_time = "2026-07-31T00:01:00Z" if training else RECEIVED_TIME
        available_time = "2026-07-31T00:02:00Z" if training else AVAILABLE_TIME
        core = card_core_by_identity[identity]
        row = {
            "record_kind": "declared_card",
            "race_id": race_id,
            "horse_id": horse_id,
            "prediction_event_time": prediction_time,
            "source_event_time": source_time,
            "received_at": received_time,
            "available_as_of": available_time,
            "source_version": "SYN-TRAINING-CARD-v1" if training else "SYN-TARGET-CARD-v1",
            "source_content_sha256": "",
            "missing_reason": "not_applicable",
            "年齢": int(core["年齢"]),
            "斤量": core["斤量"],
            "距離": int(core["距離"]),
            "場所": core["場所"],
            "性別": core["性別"],
            "騎手コード": core["騎手コード"],
            "調教師コード": core["調教師コード"],
            "芝・ダ": core["芝・ダ"],
            "クラス名": core["クラス名"],
            "トラックコード": core["トラックコード"],
        }
        event_payload = runner.canonical_event_source_payload(row, source_path, bundle)
        row["source_content_sha256"] = runner.source_payload_hash(event_payload)
        return row

    def completed_result(identity, rank):
        race_id, horse_id = identity
        row = {
            "record_kind": "completed_result",
            "race_id": race_id,
            "horse_id": horse_id,
            "prediction_event_time": "2026-08-01T06:00:00Z",
            "source_event_time": "2026-08-01T07:00:00Z",
            "received_at": "2026-08-01T07:01:00Z",
            "available_as_of": "2026-08-01T07:02:00Z",
            "source_version": "SYN-RESULT-v1",
            "source_content_sha256": "",
            "missing_reason": "not_applicable",
            "確定着順": rank,
            "1角": rank,
            "2角": rank,
            "4角": rank,
            "result_status": "finished",
            "official_finish_rank_raw": rank,
        }
        event_payload = runner.canonical_event_source_payload(row, source_path, bundle)
        row["source_content_sha256"] = runner.source_payload_hash(event_payload)
        return row

    def feature_value(identity, feature, ordinal):
        if feature in card_core_by_identity[identity]:
            return card_core_by_identity[identity][feature]
        if feature == "出走頭数":
            return float(field_size_by_race[identity[0]])
        return float(ordinal + 1) if feature in bundle.numeric_features else f"SYN-CAT-{ordinal:03d}"

    if kind == "runner_universe_manifest":
        rows = []
        per_race_number = {}
        for global_ordinal, (race_id, horse_id) in enumerate(identities, start=1):
            number = per_race_number.get(race_id, 0) + 1
            per_race_number[race_id] = number
            confirmed = len(identities) != 70 or race_id.endswith("-03") or race_id.endswith("-05")
            row = {
                "release_family_id": payload["release_family_id"],
                "release_version": payload["release_version"],
                "parent_manifest_digest": payload["parent_manifest_digest"],
                "as_of": payload["as_of"],
                "race_id": race_id,
                "horse_id": horse_id,
                "horse_name": f"Synthetic {horse_id}",
                "race_name": f"Synthetic {race_id}",
                "draw_status": "confirmed" if confirmed else "scheduled_pending_draw",
                "entry_stage": "declared_with_draw" if confirmed else "declared_without_draw",
                "runner_status": "declared_active",
                "frame_number": ((number - 1) // 2) + 1 if confirmed else None,
                "horse_number": number if confirmed else None,
                "jockey_id": f"SYN-J-{number:03d}",
                "jockey_name": f"Synthetic Jockey {number:03d}",
                "carried_weight": 55.0,
                "trainer_id": f"SYN-T-{number:03d}",
                "trainer_name": f"Synthetic Trainer {number:03d}",
                "active_for_feature_materialization": True,
                "change_reason": "initial_declaration",
                "source_event_time": SOURCE_TIME,
                "received_at": RECEIVED_TIME,
                "available_as_of": AVAILABLE_TIME,
                "source_version": "SYN-RUNNER-v1",
                "source_content_sha256": "",
                "missing_reason": "not_applicable" if confirmed else "not_declared_by_source",
                "row_payload_sha256": "",
            }
            source_payload = runner.canonical_runner_source_payload(row, source_path)
            row["source_content_sha256"] = runner.source_payload_hash(source_payload)
            row["row_payload_sha256"] = runner.row_payload_hash(row)
            rows.append(row)
        return runner.canonical_jsonl_bytes(rows, sort_key=("race_id", "horse_id"))
    if kind in {"target_source_manifest", "training_source_manifest"}:
        rows = []
        for ordinal, (race_id, horse_id) in enumerate(identities, start=1):
            card = declared_card((race_id, horse_id), training=kind == "training_source_manifest")
            rows.append(card)
            if kind == "training_source_manifest":
                rows.append(completed_result((race_id, horse_id), ordinal))
        return runner.canonical_jsonl_bytes(rows, sort_key=("prediction_event_time", "race_id", "horse_id", "record_kind"))
    if kind == "feature_release_manifest":
        rows = []
        for race_id, horse_id in identities:
            row = {
                "race_id": race_id,
                "horse_id": horse_id,
                "prediction_event_time": PREDICTION_TIME,
                "release_family_id": payload["release_family_id"],
                "release_version": payload["release_version"],
                "feature_schema_sha256": runner._feature_schema_digest(bundle),
                "lineage_manifest_sha256": lineage_hash,
            }
            for ordinal, feature in enumerate(bundle.ordered_features):
                row[feature] = feature_value((race_id, horse_id), feature, ordinal)
            rows.append(row)
        return runner.canonical_jsonl_bytes(rows, sort_key=("prediction_event_time", "race_id", "horse_id"))
    if kind == "lineage_manifest":
        rows = []
        target_card_hash_by_identity = {
            identity: declared_card(identity)["source_content_sha256"]
            for identity in identities
        }
        for race_id, horse_id in identities:
            for ordinal, feature in enumerate(bundle.ordered_features):
                dtype = "float64" if feature in bundle.numeric_features else "string"
                value = feature_value((race_id, horse_id), feature, ordinal)
                if feature in runner.RACE_AGGREGATE_FEATURES:
                    source_hashes = sorted(
                        target_card_hash_by_identity[identity]
                        for identity in identities
                        if identity[0] == race_id
                    )
                else:
                    source_hashes = [target_card_hash_by_identity[(race_id, horse_id)]]
                value_binding = runner.feature_value_binding_hash(
                    race_id=race_id,
                    horse_id=horse_id,
                    feature_name=feature,
                    feature_value=value,
                    feature_dtype=dtype,
                    prediction_event_time=PREDICTION_TIME,
                    transformation_name="synthetic_safe_transform",
                    transformation_version="SYN-TRANSFORM-v1",
                    transformation_code_sha256=HASH_B,
                    source_content_sha256_set=source_hashes,
                )
                row = {
                    "race_id": race_id,
                    "horse_id": horse_id,
                    "feature_name": feature,
                    "feature_dtype": dtype,
                    "source_paths": [source_path],
                    "source_versions": ["SYN-TARGET-CARD-v1"],
                    "source_content_sha256_set": source_hashes,
                    "dependency_feature_names": ["declared_card_or_shifted_history"],
                    "dependency_content_sha256_set": sorted({*source_hashes, value_binding}),
                    "transformation_name": "synthetic_safe_transform",
                    "transformation_version": "SYN-TRANSFORM-v1",
                    "transformation_code_sha256": HASH_B,
                    "max_source_event_time": SOURCE_TIME,
                    "max_received_at": RECEIVED_TIME,
                    "max_available_as_of": AVAILABLE_TIME,
                    "prediction_event_time": PREDICTION_TIME,
                    "missing_reason": "not_applicable",
                    "lineage_verdict": runner.LINEAGE_SAFE_VERDICT,
                    "row_payload_sha256": "",
                }
                row["row_payload_sha256"] = runner.row_payload_hash(row)
                rows.append(row)
        return runner.canonical_jsonl_bytes(rows, sort_key=("race_id", "horse_id", "feature_name"))
    if kind == "label_eligibility_manifest":
        rows = []
        for ordinal, (race_id, horse_id) in enumerate(identities, start=1):
            row = {
                "race_id": race_id,
                "horse_id": horse_id,
                "declared_status": "declared_active",
                "starter_status": "starter",
                "official_result_status": "finished",
                "official_finish_rank_raw": ordinal,
                "row_label_eligible": True,
                "race_label_eligible": True,
                "ineligibility_reason": "eligible",
                "source_content_sha256": completed_result((race_id, horse_id), ordinal)["source_content_sha256"],
                "row_payload_sha256": "",
            }
            row["row_payload_sha256"] = runner.row_payload_hash(row)
            rows.append(row)
        return runner.canonical_jsonl_bytes(rows, sort_key=("race_id", "horse_id"))
    if kind == "release_diff_manifest":
        runner_payload = copy.deepcopy(payload)
        runner_payload["manifest_kind"] = "runner_universe_manifest"
        runner_rows = {
            (row["race_id"], row["horse_id"]): row
            for row in (
                json.loads(line)
                for line in _semantic_artifact_bytes(runner_payload, bundle).decode("utf-8").splitlines()
            )
        }
        absent_hash = runner.canonical_digest({"state": "absent"})
        rows = []
        for race_id, horse_id in identities:
            current = runner_rows[(race_id, horse_id)]
            initial = payload["release_version"] == 1
            rows.append(
                {
                    "race_id": race_id,
                    "horse_id": horse_id,
                    "old_row_payload_sha256": absent_hash if initial else HASH_A,
                    "new_row_payload_sha256": current["row_payload_sha256"],
                    "changes": [
                        {
                            "field": "release_version",
                            "old": None if initial else payload["release_version"] - 1,
                            "new": payload["release_version"],
                        }
                    ],
                    "reason": "initial_declaration" if initial else "release_version_successor",
                    "source_content_sha256": current["source_content_sha256"],
                }
            )
        return runner.canonical_jsonl_bytes(rows, sort_key=("race_id", "horse_id"))
    if kind == "canonical_root_manifest":
        value = {
            "experiment_id": payload["experiment_id"],
            "release_family_id": payload["release_family_id"],
            "release_version": payload["release_version"],
            "parent_manifest_digest": payload["parent_manifest_digest"],
            "as_of": payload["as_of"],
            "source_cutoff": payload["source_cutoff"],
            "dependency_manifest_digests": payload["dependency_manifest_digests"],
            "row_count": payload["row_count"],
            "race_count": payload["race_count"],
            "runner_count": payload["runner_count"],
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
    elif kind == "environment_manifest":
        value = {
            "schema_version": 1,
            "experiment_id": runner.EXPERIMENT_ID,
            "variant": runner.VARIANT,
            "environment_contract_version": "exp034-stdlib-canonicalization-v1",
            "python_implementation": "CPython",
            "python_version": "3.12.0-synthetic",
            "platform": "synthetic",
            "executable_sha256": HASH_C,
            "encoding": "UTF-8",
            "line_endings": "LF",
            "timezone": "UTC",
            "locale": "C.UTF-8",
            "pythonhashseed": "0",
            "network_access": False,
            "filesystem_mtime_as_received_at": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
    elif kind == "dependency_lock_manifest":
        value = {
            "schema_version": 1,
            "experiment_id": runner.EXPERIMENT_ID,
            "variant": runner.VARIANT,
            "lock_version": "SYN-STDLIB-LOCK-v1",
            "dependency_policy": "Python standard library only",
            "packages": [],
            "interpreter_sha256": HASH_C,
            "config_sha256": payload["config_sha256"],
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
    else:
        raise AssertionError(f"unsupported semantic artifact kind: {kind}")
    return runner.canonical_json_bytes(value) + b"\n"


def _seal_payload_with_artifact_bytes(payload, bundle=None, *, lineage_hash=HASH_A):
    item = copy.deepcopy(payload)
    bundle = bundle or _bundle()
    artifact = item["artifacts"][0]
    item["schema_sha256"] = runner.artifact_schema_digest(item["manifest_kind"], bundle)
    raw = _semantic_artifact_bytes(item, bundle, lineage_hash=lineage_hash)
    if artifact["path"].endswith(".jsonl"):
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        artifact["row_count"] = len(rows)
        item["row_count"] = len(rows)
        item["row_counts"] = {"total": len(rows)}
        identities = {
            (row["race_id"], row["horse_id"])
            for row in rows
            if "race_id" in row and "horse_id" in row
        }
        if identities:
            race_count = len({race_id for race_id, _horse_id in identities})
            item["race_count"] = race_count
            item["runner_count"] = len(identities)
            item["race_counts"] = {"total": race_count}
            item["identity_counts"] = {"runner_count": len(identities)}
        if rows and all("missing_reason" in row for row in rows):
            distribution = {}
            for row in rows:
                distribution[row["missing_reason"]] = distribution.get(row["missing_reason"], 0) + 1
            item["missing_reason_distribution"] = distribution
        if item["manifest_kind"] == "label_eligibility_manifest":
            eligible = sum(row["row_label_eligible"] is True for row in rows)
            item["label_eligibility_counts"] = {
                "eligible": eligible,
                "ineligible": len(rows) - eligible,
            }
    digest = hashlib.sha256(raw).hexdigest()
    artifact["sha256"] = digest
    item["output_artifact_paths_and_sha256"] = [{"path": artifact["path"], "sha256": digest}]
    manifest = runner.seal_manifest(
        item,
        bundle,
        artifact_bytes_by_path={artifact["path"]: raw},
    )
    _ARTIFACT_BYTES_BY_SHA[digest] = raw
    return manifest, {artifact["path"]: raw}


def _parent_runner_manifest(rows, bundle):
    release_bytes = runner.runner_release_bytes(rows, bundle)
    payload = _manifest_payload(
            "runner_universe_manifest",
            row_count=len(rows),
            race_count=len({row["race_id"] for row in rows}),
            runner_count=len(rows),
            artifact_name="runner_universe_release",
            artifact_hash=runner.sha256_bytes(release_bytes),
            artifact_path="outputs/research/SYN/runner_universe.jsonl",
            release_family_id="SYN-RUNNER-UNIVERSE",
        )
    payload["schema_sha256"] = runner.artifact_schema_digest("runner_universe_manifest", bundle)
    payload["missing_reason_distribution"] = {
        reason: sum(row["missing_reason"] == reason for row in rows)
        for reason in sorted({row["missing_reason"] for row in rows})
    }
    return runner.seal_manifest(
        payload,
        bundle,
        artifact_bytes_by_path={"outputs/research/SYN/runner_universe.jsonl": release_bytes},
    )


def _feature_fixture(bundle, runner_rows):
    cells = []
    payloads = {}
    active_hashes_by_race = {}
    evidence_by_identity = {}
    for active_row in runner_rows:
        if active_row["active_for_feature_materialization"]:
            source_payload = _runner_source_payload(active_row)
            evidence = _runner_feature_evidence(active_row, source_payload)
            evidence_by_identity[(active_row["race_id"], active_row["horse_id"])] = evidence
            active_hashes_by_race.setdefault(active_row["race_id"], set()).add(evidence["content_sha256"])
    active_count_by_race = {
        race_id: len(hashes) for race_id, hashes in active_hashes_by_race.items()
    }
    for row in runner_rows:
        race_id, horse_id = row["race_id"], row["horse_id"]
        direct_evidence = evidence_by_identity[(race_id, horse_id)]
        for ordinal, feature in enumerate(bundle.ordered_features):
            dtype = "float64" if feature in bundle.numeric_features else "string"
            value = float(ordinal + 1) if feature in bundle.numeric_features else f"SYN-CAT-{ordinal:03d}"
            if feature == "出走頭数":
                value = float(active_count_by_race[race_id])
            source_hash = direct_evidence["content_sha256"]
            dependencies = {source_hash}
            if feature in runner.RACE_AGGREGATE_FEATURES:
                dependencies.update(active_hashes_by_race[race_id])
            cell = {
                "release_id": "SYN-FEATURE-RELEASE",
                "release_version": 1,
                "race_id": race_id,
                "horse_id": horse_id,
                "prediction_event_time": PREDICTION_TIME,
                "feature_name": feature,
                "feature_value": value,
                "dtype": dtype,
                "source_path": direct_evidence["source_path"],
                "source_event_time": direct_evidence["source_event_time"],
                "received_at": direct_evidence["received_at"],
                "available_as_of": direct_evidence["available_as_of"],
                "source_version": direct_evidence["source_version"],
                "source_content_hash": source_hash,
                "transform_name": "synthetic_safe_transform",
                "transform_version": "SYN-TRANSFORM-v1",
                "transform_code_hash": HASH_B,
                "dependency_feature_names": ["declared_card_or_shifted_history"],
                "dependency_content_hashes": sorted(dependencies),
                "missing_reason": "not_applicable",
                "asof_safe": True,
                "lineage_status": runner.LINEAGE_SAFE_VERDICT,
            }
            cells.append(cell)
            payloads[(race_id, horse_id, feature)] = copy.deepcopy(direct_evidence)
    return cells, payloads


def _result_fixture(statuses, ranks):
    rows = []
    payloads = {}
    race_id = "SYN-LABEL-RACE"
    result_columns = _bundle().input_contract["event_release_schema"]["historical_result_columns"]
    for ordinal, (status, rank) in enumerate(zip(statuses, ranks), start=1):
        horse_id = f"SYN-LABEL-HORSE-{ordinal:02d}"
        row = {
            "record_kind": "completed_result",
            "race_id": race_id,
            "horse_id": horse_id,
            "prediction_event_time": "2026-01-01T06:00:00Z",
            "source_event_time": "2026-01-01T07:00:00Z",
            "received_at": "2026-01-01T07:01:00Z",
            "available_as_of": "2026-01-01T07:02:00Z",
            "source_version": "SYN-RESULT-v1",
            "source_content_sha256": "",
            "missing_reason": "not_applicable",
            "確定着順": rank,
            "1角": ordinal,
            "2角": ordinal,
            "4角": ordinal,
            "result_status": status,
            "official_finish_rank_raw": rank,
        }
        payload = runner.canonical_event_source_payload(
            row,
            "synthetic/completed_result.jsonl",
            _bundle(),
        )
        row["source_content_sha256"] = runner.source_payload_hash(payload)
        rows.append(row)
        payloads[(race_id, horse_id)] = payload
    declared = []
    declared_payloads = {}
    for ordinal in range(1, len(rows) + 1):
        horse_id = f"SYN-LABEL-HORSE-{ordinal:02d}"
        payload = {
            "canonical_declared_status": {"runner_status": "declared_active"},
            "source_path": "synthetic/declared_status.jsonl",
            "source_raw_columns": ["runner_status"],
        }
        declared.append(
            {
                "race_id": race_id,
                "horse_id": horse_id,
                "runner_status": "declared_active",
                "prediction_event_time": "2026-01-01T06:00:00Z",
                "source_event_time": "2025-12-31T20:00:00Z",
                "received_at": "2025-12-31T20:01:00Z",
                "available_as_of": "2025-12-31T20:02:00Z",
                "source_version": "SYN-CARD-v1",
                "source_content_sha256": runner.source_payload_hash(payload),
                "missing_reason": "not_applicable",
            }
        )
        declared_payloads[(race_id, horse_id)] = payload
    return declared, declared_payloads, rows, payloads


def _manifest(kind, version=1, parent=None):
    row_count = 2 if kind == "training_source_manifest" else 88 if kind == "lineage_manifest" else 1
    manifest, _ = _seal_payload_with_artifact_bytes(
        _manifest_payload(kind, row_count=row_count, version=version, parent=parent)
    )
    return manifest


def _artifact_evidence_for_manifests(manifests):
    evidence = {}
    for manifest in manifests:
        artifact = manifest["artifacts"][0]
        raw = _ARTIFACT_BYTES_BY_SHA.get(artifact["sha256"])
        if raw is None:
            raise AssertionError(f"fixture artifact bytes missing: {artifact['path']}")
        evidence[artifact["path"]] = raw
    return evidence


def _build_semantic_manifest_set(bundle, *, handoff=False):
    target_runner_count = 70 if handoff else 1
    target_race_count = 5 if handoff else 1
    payload_by_kind = {}
    for kind in bundle.config["required_manifest_kinds"]:
        if kind == "canonical_root_manifest":
            continue
        if kind == "lineage_manifest":
            payload = _manifest_payload(
                kind,
                row_count=target_runner_count * 88,
                race_count=target_race_count,
                runner_count=target_runner_count,
            )
        elif kind in {"runner_universe_manifest", "target_source_manifest", "feature_release_manifest", "release_diff_manifest"}:
            payload = _manifest_payload(
                kind,
                row_count=target_runner_count,
                race_count=target_race_count,
                runner_count=target_runner_count,
            )
        elif kind == "training_source_manifest":
            payload = _manifest_payload(
                kind,
                row_count=4,
                race_count=1,
                runner_count=2,
                source_cutoff="2026-08-01T07:02:00Z",
            )
        elif kind == "label_eligibility_manifest":
            payload = _manifest_payload(
                kind,
                row_count=2,
                race_count=1,
                runner_count=2,
                source_cutoff="2026-08-01T07:02:00Z",
            )
        else:
            payload = _manifest_payload(kind, row_count=1, race_count=1, runner_count=1)
        payload_by_kind[kind] = payload

    manifests_by_kind = {}
    lineage, _ = _seal_payload_with_artifact_bytes(payload_by_kind["lineage_manifest"], bundle)
    manifests_by_kind["lineage_manifest"] = lineage
    lineage_artifact_hash = lineage["artifacts"][0]["sha256"]
    for kind, payload in payload_by_kind.items():
        if kind == "lineage_manifest":
            continue
        manifest, _ = _seal_payload_with_artifact_bytes(
            payload,
            bundle,
            lineage_hash=lineage_artifact_hash,
        )
        manifests_by_kind[kind] = manifest
    leaves = [manifests_by_kind[kind] for kind in bundle.config["required_manifest_kinds"] if kind != "canonical_root_manifest"]
    root_payload = _manifest_payload(
        "canonical_root_manifest",
        row_count=sum(leaf["row_count"] for leaf in leaves),
        race_count=target_race_count,
        runner_count=target_runner_count,
        artifact_name="canonical_root",
        artifact_path="outputs/research/SYN/root.json",
    )
    root_payload["dependency_manifest_digests"] = {
        leaf["manifest_kind"]: leaf["content_hash"] for leaf in leaves
    }
    root, _ = _seal_payload_with_artifact_bytes(root_payload, bundle)
    manifests = [*leaves, root]
    return manifests, _artifact_evidence_for_manifests(manifests)


def _reseal_semantic_manifest(payload, raw, bundle):
    item = copy.deepcopy(payload)
    artifact = item["artifacts"][0]
    digest = hashlib.sha256(raw).hexdigest()
    artifact["sha256"] = digest
    item["output_artifact_paths_and_sha256"] = [{"path": artifact["path"], "sha256": digest}]
    if artifact["path"].endswith(".jsonl"):
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        artifact["row_count"] = len(rows)
        item["row_count"] = len(rows)
        item["row_counts"] = {"total": len(rows)}
        identities = {
            (row["race_id"], row["horse_id"])
            for row in rows
            if "race_id" in row and "horse_id" in row
        }
        if identities:
            race_count = len({race_id for race_id, _horse_id in identities})
            item["race_count"] = race_count
            item["runner_count"] = len(identities)
            item["race_counts"] = {"total": race_count}
            item["identity_counts"] = {"runner_count": len(identities)}
        if rows and all("missing_reason" in row for row in rows):
            distribution = {}
            for row in rows:
                reason = row["missing_reason"]
                distribution[reason] = distribution.get(reason, 0) + 1
            item["missing_reason_distribution"] = distribution
        else:
            item["missing_reason_distribution"] = {"not_applicable": len(rows)}
        item["as_of_verdict_counts"] = {"certified_asof_safe": len(rows)}
        if item["manifest_kind"] == "label_eligibility_manifest":
            eligible = sum(row["row_label_eligible"] is True for row in rows)
            item["label_eligibility_counts"] = {
                "eligible": eligible,
                "ineligible": len(rows) - eligible,
            }
        else:
            item["label_eligibility_counts"] = {"not_applicable": len(rows)}
    return runner.seal_manifest(
        item,
        bundle,
        artifact_bytes_by_path={artifact["path"]: raw},
    )


def _build_draw_confirmed_child_manifest_set(parent_manifests, parent_evidence, bundle):
    parent_by_kind = {item["manifest_kind"]: item for item in parent_manifests}
    parent_runner_manifest = parent_by_kind["runner_universe_manifest"]
    parent_runner_digest = parent_runner_manifest["content_hash"]
    parent_runner_path = parent_runner_manifest["artifacts"][0]["path"]
    runner_source_path = parent_runner_manifest["input_source_paths_and_sha256"][0]["path"]
    parent_runner_rows = [
        json.loads(line) for line in parent_evidence[parent_runner_path].decode("utf-8").splitlines()
    ]
    child_runner_rows = []
    for parent in parent_runner_rows:
        child = copy.deepcopy(parent)
        child["release_version"] = 2
        child["parent_manifest_digest"] = parent_runner_digest
        if parent["draw_status"] == "scheduled_pending_draw":
            horse_number = sum(
                other["race_id"] == parent["race_id"] and other["horse_id"] <= parent["horse_id"]
                for other in parent_runner_rows
            )
            child.update(
                {
                    "draw_status": "confirmed",
                    "entry_stage": "declared_with_draw",
                    "frame_number": ((horse_number - 1) // 2) + 1,
                    "horse_number": horse_number,
                    "change_reason": "draw_confirmed",
                    "source_event_time": "2026-08-22T00:00:00Z",
                    "received_at": "2026-08-22T00:01:00Z",
                    "available_as_of": "2026-08-22T00:02:00Z",
                    "source_version": "SYN-RUNNER-v2",
                    "source_content_sha256": "",
                    "missing_reason": "not_applicable",
                }
            )
        if child["source_content_sha256"] == "":
            child_source_payload = runner.canonical_runner_source_payload(child, runner_source_path)
            child["source_content_sha256"] = runner.source_payload_hash(child_source_payload)
        child["row_payload_sha256"] = runner.row_payload_hash(child)
        child_runner_rows.append(child)
    child_runner_raw = runner.canonical_jsonl_bytes(child_runner_rows, sort_key=("race_id", "horse_id"))

    audited_fields = runner._runner_diff_audited_fields(bundle)
    child_by_identity = {(row["race_id"], row["horse_id"]): row for row in child_runner_rows}
    diff_rows = []
    for parent in parent_runner_rows:
        identity = (parent["race_id"], parent["horse_id"])
        child = child_by_identity[identity]
        source_changed = any(
            parent[field] != child[field]
            for field in ("source_event_time", "received_at", "available_as_of", "source_version", "source_content_sha256")
        )
        diff_rows.append(
            {
                "race_id": identity[0],
                "horse_id": identity[1],
                "old_row_payload_sha256": parent["row_payload_sha256"],
                "new_row_payload_sha256": child["row_payload_sha256"],
                "changes": [
                    {"field": field, "old": parent.get(field), "new": child.get(field)}
                    for field in audited_fields
                    if parent.get(field) != child.get(field)
                ],
                "reason": child["change_reason"] if source_changed else "release_version_successor",
                "source_content_sha256": child["source_content_sha256"],
            }
        )
    child_diff_raw = runner.canonical_jsonl_bytes(diff_rows, sort_key=("race_id", "horse_id"))

    synchronized = set(bundle.config["manifest_composition_contract"]["root_synchronized_kinds"])
    reusable = set(bundle.config["manifest_composition_contract"]["immutable_reusable_kinds"])
    child_by_kind = {kind: copy.deepcopy(parent_by_kind[kind]) for kind in reusable}
    child_evidence = {
        parent_by_kind[kind]["artifacts"][0]["path"]: parent_evidence[parent_by_kind[kind]["artifacts"][0]["path"]]
        for kind in reusable
    }
    for kind in synchronized:
        parent_manifest = parent_by_kind[kind]
        payload = {key: copy.deepcopy(value) for key, value in parent_manifest.items() if key != "content_hash"}
        payload["release_version"] = 2
        payload["parent_manifest_digest"] = parent_runner_digest
        if kind == "runner_universe_manifest":
            raw = child_runner_raw
            payload["source_cutoff"] = "2026-08-22T00:02:00Z"
        elif kind == "release_diff_manifest":
            raw = child_diff_raw
            payload["source_cutoff"] = "2026-08-22T00:02:00Z"
        else:
            parent_path = parent_manifest["artifacts"][0]["path"]
            raw = parent_evidence[parent_path]
            if kind == "feature_release_manifest":
                feature_rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
                for row in feature_rows:
                    row["release_version"] = 2
                raw = runner.canonical_jsonl_bytes(
                    feature_rows,
                    sort_key=("prediction_event_time", "race_id", "horse_id"),
                )
        child_manifest = _reseal_semantic_manifest(payload, raw, bundle)
        child_by_kind[kind] = child_manifest
        child_evidence[child_manifest["artifacts"][0]["path"]] = raw

    leaves = [
        child_by_kind[kind]
        for kind in bundle.config["required_manifest_kinds"]
        if kind != "canonical_root_manifest"
    ]
    parent_root = parent_by_kind["canonical_root_manifest"]
    root_payload = {key: copy.deepcopy(value) for key, value in parent_root.items() if key != "content_hash"}
    root_payload["release_version"] = 2
    root_payload["parent_manifest_digest"] = parent_runner_digest
    root_payload["source_cutoff"] = max(item["source_cutoff"] for item in leaves)
    root_payload["dependency_manifest_digests"] = {
        item["manifest_kind"]: item["content_hash"] for item in leaves
    }
    root_payload["row_count"] = sum(item["row_count"] for item in leaves)
    root_payload["artifacts"][0]["row_count"] = root_payload["row_count"]
    root_payload["row_counts"] = {"total": root_payload["row_count"]}
    root_payload["missing_reason_distribution"] = {"not_applicable": root_payload["row_count"]}
    root_payload["as_of_verdict_counts"] = {"certified_asof_safe": root_payload["row_count"]}
    root_payload["label_eligibility_counts"] = {"not_applicable": root_payload["row_count"]}
    root_raw = _semantic_artifact_bytes(root_payload, bundle)
    child_root = _reseal_semantic_manifest(root_payload, root_raw, bundle)
    child_by_kind["canonical_root_manifest"] = child_root
    child_evidence[child_root["artifacts"][0]["path"]] = root_raw
    child_manifests = [child_by_kind[kind] for kind in bundle.config["required_manifest_kinds"]]
    return child_manifests, child_evidence


def _replace_semantic_artifact(manifests, evidence, bundle, kind, raw):
    updated = copy.deepcopy(manifests)
    leaf_index = next(i for i, item in enumerate(updated) if item["manifest_kind"] == kind)
    leaf_payload = {key: value for key, value in updated[leaf_index].items() if key != "content_hash"}
    artifact = leaf_payload["artifacts"][0]
    artifact["sha256"] = hashlib.sha256(raw).hexdigest()
    leaf_payload["output_artifact_paths_and_sha256"] = [
        {"path": artifact["path"], "sha256": artifact["sha256"]}
    ]
    if artifact["path"].endswith(".jsonl"):
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        artifact["row_count"] = len(rows)
        leaf_payload["row_count"] = len(rows)
        leaf_payload["row_counts"] = {"total": len(rows)}
        leaf_payload["as_of_verdict_counts"] = {"certified_asof_safe": len(rows)}
        leaf_payload["duplicate_and_missing_counts"] = {
            "duplicate_key_count": 0,
            "missing_key_count": 0,
        }
        identities = {
            (row["race_id"], row["horse_id"])
            for row in rows
            if "race_id" in row and "horse_id" in row
        }
        if identities:
            race_count = len({race_id for race_id, _horse_id in identities})
            leaf_payload["race_count"] = race_count
            leaf_payload["runner_count"] = len(identities)
            leaf_payload["race_counts"] = {"total": race_count}
            leaf_payload["identity_counts"] = {"runner_count": len(identities)}
        if rows and all("missing_reason" in row for row in rows):
            distribution = {}
            for row in rows:
                distribution[row["missing_reason"]] = distribution.get(row["missing_reason"], 0) + 1
            leaf_payload["missing_reason_distribution"] = distribution
        else:
            leaf_payload["missing_reason_distribution"] = {"not_applicable": len(rows)}
        if kind == "label_eligibility_manifest":
            eligible = sum(row["row_label_eligible"] is True for row in rows)
            leaf_payload["label_eligibility_counts"] = {
                "eligible": eligible,
                "ineligible": len(rows) - eligible,
            }
        else:
            leaf_payload["label_eligibility_counts"] = {"not_applicable": len(rows)}
    updated[leaf_index] = runner.seal_manifest(
        leaf_payload,
        bundle,
        artifact_bytes_by_path={artifact["path"]: raw},
    )
    root_index = next(i for i, item in enumerate(updated) if item["manifest_kind"] == "canonical_root_manifest")
    leaves = [item for item in updated if item["manifest_kind"] != "canonical_root_manifest"]
    root_payload = {key: value for key, value in updated[root_index].items() if key != "content_hash"}
    root_payload["dependency_manifest_digests"] = {
        item["manifest_kind"]: item["content_hash"] for item in leaves
    }
    root_payload["row_count"] = sum(item["row_count"] for item in leaves)
    root_payload["artifacts"][0]["row_count"] = root_payload["row_count"]
    root_payload["row_counts"] = {"total": root_payload["row_count"]}
    root_payload["missing_reason_distribution"] = {"not_applicable": root_payload["row_count"]}
    root_payload["as_of_verdict_counts"] = {"certified_asof_safe": root_payload["row_count"]}
    root_payload["label_eligibility_counts"] = {"not_applicable": root_payload["row_count"]}
    updated[root_index], root_evidence = _seal_payload_with_artifact_bytes(root_payload, bundle)
    new_evidence = dict(evidence)
    new_evidence[artifact["path"]] = raw
    new_evidence.update(root_evidence)
    return updated, new_evidence


def _replace_lineage_and_rebind_feature(manifests, evidence, bundle, lineage_rows):
    lineage_raw = runner.canonical_jsonl_bytes(
        lineage_rows,
        sort_key=("race_id", "horse_id", "feature_name"),
    )
    updated, updated_evidence = _replace_semantic_artifact(
        manifests,
        evidence,
        bundle,
        "lineage_manifest",
        lineage_raw,
    )
    lineage_manifest = next(
        item for item in updated if item["manifest_kind"] == "lineage_manifest"
    )
    feature_manifest = next(
        item for item in updated if item["manifest_kind"] == "feature_release_manifest"
    )
    feature_path = feature_manifest["artifacts"][0]["path"]
    feature_rows = [
        json.loads(line)
        for line in updated_evidence[feature_path].decode("utf-8").splitlines()
    ]
    for row in feature_rows:
        row["lineage_manifest_sha256"] = lineage_manifest["artifacts"][0]["sha256"]
    feature_raw = runner.canonical_jsonl_bytes(
        feature_rows,
        sort_key=("prediction_event_time", "race_id", "horse_id"),
    )
    return _replace_semantic_artifact(
        updated,
        updated_evidence,
        bundle,
        "feature_release_manifest",
        feature_raw,
    )


def _connect_actual_feature_builder(manifests, evidence, bundle):
    by_kind = {item["manifest_kind"]: item for item in manifests}
    runner_manifest = by_kind["runner_universe_manifest"]
    target_manifest = by_kind["target_source_manifest"]
    feature_manifest = by_kind["feature_release_manifest"]
    runner_path = runner_manifest["artifacts"][0]["path"]
    target_path = target_manifest["artifacts"][0]["path"]
    feature_path = feature_manifest["artifacts"][0]["path"]
    runner_source_path = runner_manifest["input_source_paths_and_sha256"][0]["path"]
    target_source_path = target_manifest["input_source_paths_and_sha256"][0]["path"]
    runner_rows = [
        {field: raw[field] for field in bundle.runner_columns}
        for raw in (
            json.loads(line) for line in evidence[runner_path].decode("utf-8").splitlines()
        )
    ]
    target_rows = [json.loads(line) for line in evidence[target_path].decode("utf-8").splitlines()]
    existing_feature_rows = [
        json.loads(line) for line in evidence[feature_path].decode("utf-8").splitlines()
    ]
    runner_payloads = {
        (row["race_id"], row["horse_id"]): runner.canonical_runner_source_payload(
            row, runner_source_path
        )
        for row in runner_rows
    }
    target_by_identity = {
        (row["race_id"], row["horse_id"]): row for row in target_rows
    }
    target_payload_by_identity = {
        identity: runner.canonical_event_source_payload(row, target_source_path, bundle)
        for identity, row in target_by_identity.items()
    }
    target_evidence_by_hash = {
        row["source_content_sha256"]: _event_source_evidence(
            row, target_payload_by_identity[identity]
        )
        for identity, row in target_by_identity.items()
    }
    target_hashes_by_race = {}
    for identity, row in target_by_identity.items():
        target_hashes_by_race.setdefault(identity[0], set()).add(row["source_content_sha256"])
    existing_by_identity = {
        (row["race_id"], row["horse_id"]): row for row in existing_feature_rows
    }
    cells = []
    direct_evidence = {}
    for identity in sorted(existing_by_identity):
        target_row = target_by_identity[identity]
        wide = existing_by_identity[identity]
        focal_hash = target_row["source_content_sha256"]
        for feature in bundle.ordered_features:
            dtype = "float64" if feature in bundle.numeric_features else "string"
            dependencies = (
                target_hashes_by_race[identity[0]]
                if feature in runner.RACE_AGGREGATE_FEATURES
                else {focal_hash}
            )
            cell = {
                "release_id": feature_manifest["release_family_id"],
                "release_version": feature_manifest["release_version"],
                "race_id": identity[0],
                "horse_id": identity[1],
                "prediction_event_time": target_row["prediction_event_time"],
                "feature_name": feature,
                "feature_value": wide[feature],
                "dtype": dtype,
                "source_path": target_source_path,
                "source_event_time": target_row["source_event_time"],
                "received_at": target_row["received_at"],
                "available_as_of": target_row["available_as_of"],
                "source_version": target_row["source_version"],
                "source_content_hash": focal_hash,
                "transform_name": "synthetic_safe_transform",
                "transform_version": "SYN-TRANSFORM-v1",
                "transform_code_hash": HASH_B,
                "dependency_feature_names": ["declared_card_or_shifted_history"],
                "dependency_content_hashes": sorted(dependencies),
                "missing_reason": "not_applicable",
                "asof_safe": True,
                "lineage_status": runner.LINEAGE_SAFE_VERDICT,
            }
            cells.append(cell)
            direct_evidence[(identity[0], identity[1], feature)] = copy.deepcopy(
                target_evidence_by_hash[focal_hash]
            )
    release = runner.build_feature_release(
        cells,
        bundle,
        source_payloads=direct_evidence,
        runner_rows=runner_rows,
        runner_source_payloads=runner_payloads,
        dependency_evidence=target_evidence_by_hash,
    )
    updated, updated_evidence = _replace_semantic_artifact(
        manifests,
        evidence,
        bundle,
        "lineage_manifest",
        release["lineage_bytes"],
    )
    updated, updated_evidence = _replace_semantic_artifact(
        updated,
        updated_evidence,
        bundle,
        "feature_release_manifest",
        release["wide_bytes"],
    )
    return updated, updated_evidence, release


class StaticPrepareContractTests(unittest.TestCase):
    def test_frozen_proposal_and_dependency_hashes(self):
        expected = {
            "research/scopes/EXP-20260821-034.proposal.json": "c6c02b486f1e253f606f4438e4e0a7ea4eb567cd283f00ca37fb25a760d0937e",
            "research/drafts/EXP-20260821-034.fold_manifest.json": "b939dd6504e59d2214324141726bb0b0e91c7c762c99bb508743fee7257737e1",
            "research/drafts/EXP-20260821-034.input_canonicalization_contract.json": "1d72ca2443f3bb49c912074152c5afafb0b6e9e50647ab7b571cb7a072caf21a",
            "research/drafts/EXP-20260821-034.synthetic_fixture_plan.md": "f79e322785063cb41fa610a7f4ee7f904e291faf5cf7a78033edf64fbc32668c",
        }
        for relative, digest in expected.items():
            raw = (ROOT / relative).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
            self.assertNotIn(b"\r", raw)
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        proposal = json.loads((ROOT / "research/scopes/EXP-20260821-034.proposal.json").read_text(encoding="utf-8"))
        self.assertEqual(runner.canonical_digest(proposal), runner.PROPOSAL_DIGEST)

    def test_runner_is_stdlib_only_and_has_no_prohibited_imports_or_paths(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertFalse(any(name.startswith("src") for name in imports))
        self.assertFalse(any(name in {"requests", "urllib", "socket", "subprocess", "pandas", "numpy"} for name in imports))
        for forbidden in ("date/raw", "outputs/analysis", "Champion", "send_order(", "stake="):
            self.assertNotIn(forbidden, source)

    def test_config_and_environment_safety(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(), runner.CONFIG_BYTE_SHA256)
        environment = json.loads(
            (ROOT / "research/drafts/EXP-20260821-034.dependency_environment_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["variant"], runner.VARIANT)
        self.assertEqual(config["frozen_contracts"]["exp033_feature_denylist"]["total_count"], 286)
        self.assertEqual(len(_bundle().denied_features), 286)
        self.assertEqual(config["runtime_authorization"]["real_data_cli_status"], "fail_closed")
        self.assertFalse(config["runtime_authorization"]["run_scope_generation_during_prepare"])
        self.assertTrue(
            any(
                "synthetic value-binding hashes are structural fixture receipts only" in blocker
                for blocker in config["real_data_execution_blockers"]
            )
        )
        self.assertEqual(environment["supported_python"], ["3.11", "3.12"])
        self.assertEqual(environment["runtime"]["dependencies"], [])
        for value in (config, environment):
            self.assertIs(value["formal_buy"], False)
            self.assertIs(value["send_order"], False)
            self.assertIs(type(value["stake"]), int)
            self.assertEqual(value["stake"], 0)

    def test_strict_json_and_canonical_serialization(self):
        with self.assertRaises(runner.ContractError):
            runner.strict_json_loads('{"a":1,"a":2}')
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.assertRaises(runner.ContractError):
                runner.strict_json_loads(f'{{"a":{value}}}')
        left = {"z": 1, "a": [3, 2, 1]}
        right = {"a": [3, 2, 1], "z": 1}
        self.assertEqual(runner.canonical_json_bytes(left), runner.canonical_json_bytes(right))


class RunnerUniverseAndVersioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = _bundle()
        cls.rows, cls.payloads = _runner_fixture(cls.bundle)

    def test_exact_synthetic_predraw_baseline_and_duplicate_rejection(self):
        validated = runner.validate_predraw_baseline_runner_universe(
            self.rows, self.bundle, source_payloads=self.payloads
        )
        self.assertEqual(len(validated), 70)
        self.assertEqual(sum(row["draw_status"] == "scheduled_pending_draw" for row in validated), 44)
        self.assertEqual(sum(row["draw_status"] == "confirmed" for row in validated), 26)
        with self.assertRaises(runner.ContractError):
            runner.validate_runner_universe((*self.rows, self.rows[0]), self.bundle)

    def test_horse_name_join_and_zero_or_provisional_draw_are_rejected(self):
        with self.assertRaises(runner.ContractError):
            runner.join_runner_updates(self.rows, [], join_keys=("race_id", "horse_name"))
        bad = [copy.deepcopy(row) for row in self.rows]
        index = next(i for i, row in enumerate(bad) if row["draw_status"] == "scheduled_pending_draw")
        bad[index]["frame_number"] = 0
        bad[index]["horse_number"] = 0
        bad[index]["row_payload_sha256"] = runner.row_payload_hash(bad[index])
        with self.assertRaises(runner.ContractError):
            runner.validate_runner_universe(bad, self.bundle)

    def test_draw_state_is_race_atomic_and_published_numbers_are_bounded(self):
        mixed = [copy.deepcopy(row) for row in self.rows]
        pending_index = next(i for i, row in enumerate(mixed) if row["draw_status"] == "scheduled_pending_draw")
        mixed[pending_index]["draw_status"] = "confirmed"
        mixed[pending_index]["entry_stage"] = "declared_with_draw"
        mixed[pending_index]["frame_number"] = 1
        mixed[pending_index]["horse_number"] = 1
        mixed[pending_index]["missing_reason"] = "not_applicable"
        mixed_payloads = dict(self.payloads)
        mixed_identity = (mixed[pending_index]["race_id"], mixed[pending_index]["horse_id"])
        mixed_payloads[mixed_identity] = _runner_source_payload(mixed[pending_index])
        mixed[pending_index]["source_content_sha256"] = runner.source_payload_hash(
            mixed_payloads[mixed_identity]
        )
        mixed[pending_index]["row_payload_sha256"] = runner.row_payload_hash(mixed[pending_index])
        with self.assertRaises(runner.ContractError):
            runner.validate_predraw_baseline_runner_universe(
                mixed, self.bundle, source_payloads=mixed_payloads
            )

        out_of_range = [copy.deepcopy(row) for row in self.rows]
        confirmed_index = next(i for i, row in enumerate(out_of_range) if row["draw_status"] == "confirmed")
        out_of_range[confirmed_index]["frame_number"] = 999
        out_of_range[confirmed_index]["horse_number"] = 999
        out_of_range[confirmed_index]["row_payload_sha256"] = runner.row_payload_hash(
            out_of_range[confirmed_index]
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_runner_universe(out_of_range, self.bundle)

    def test_display_name_mutation_does_not_change_identity_join(self):
        renamed = [copy.deepcopy(row) for row in self.rows]
        renamed[0]["horse_name"] = "Changed display-only name"
        renamed[0]["row_payload_sha256"] = runner.row_payload_hash(renamed[0])
        joined = runner.join_runner_updates(renamed, [{"race_id": renamed[0]["race_id"], "horse_id": renamed[0]["horse_id"], "jockey_name": "New Display"}])
        self.assertEqual(len(joined), 70)
        self.assertIn((renamed[0]["race_id"], renamed[0]["horse_id"]), joined)

    def test_requested_schema_maps_to_frozen_artifact_split(self):
        dto = {
            "release_id": "SYN-RUNNER-UNIVERSE",
            "release_version": 1,
            "parent_release_id": None,
            "race_id": "SYN-DTO-RACE",
            "horse_id": "SYN-DTO-HORSE",
            "horse_name": "Synthetic DTO Horse",
            "race_name": "Synthetic DTO Race",
            "event_date": "2026-08-23",
            "post_time": PREDICTION_TIME,
            "jockey_id": "SYN-JOCKEY",
            "trainer_id": "SYN-TRAINER",
            "age": 4,
            "sex": "牡",
            "assigned_weight": 55.0,
            "frame_no": None,
            "horse_no": None,
            "draw_status": "scheduled_pending_draw",
            "entry_stage": "declared_without_draw",
            "runner_status": "declared_active",
            "source_event_time": SOURCE_TIME,
            "received_at": RECEIVED_TIME,
            "available_as_of": AVAILABLE_TIME,
            "source_version": "SYN-v1",
            "source_content_hash": "",
            "missing_reason": "not_applicable",
        }
        payload = {
            "canonical_runner_fields": {
                "race_id": dto["race_id"],
                "horse_id": dto["horse_id"],
                "horse_name": dto["horse_name"],
                "race_name": dto["race_name"],
                "draw_status": dto["draw_status"],
                "entry_stage": dto["entry_stage"],
                "runner_status": dto["runner_status"],
                "frame_number": dto["frame_no"],
                "horse_number": dto["horse_no"],
                "jockey_id": dto["jockey_id"],
                "jockey_name": None,
                "carried_weight": dto["assigned_weight"],
                "trainer_id": dto["trainer_id"],
                "trainer_name": None,
                "active_for_feature_materialization": True,
            },
            "feature_safe_runner_fields": {
                "race_id": dto["race_id"],
                "horse_id": dto["horse_id"],
                "runner_status": dto["runner_status"],
                "jockey_id": dto["jockey_id"],
                "carried_weight": dto["assigned_weight"],
                "trainer_id": dto["trainer_id"],
                "active_for_feature_materialization": True,
            },
            "feature_safe_source_raw_columns": sorted(runner.RUNNER_FEATURE_SAFE_FIELDS),
            "canonical_source_envelope": {
                "race_id": dto["race_id"],
                "horse_id": dto["horse_id"],
                "as_of": PREDICTION_TIME,
                "source_event_time": dto["source_event_time"],
                "received_at": dto["received_at"],
                "available_as_of": dto["available_as_of"],
                "source_version": dto["source_version"],
                "missing_reason": dto["missing_reason"],
            },
            "source_path": "synthetic/declared_runner_source.jsonl",
            "source_raw_columns": sorted(runner.RUNNER_SOURCE_BOUND_FIELDS),
        }
        dto["source_content_hash"] = runner.source_payload_hash(payload)
        canonical, fragment = runner.requested_runner_dto_to_approved_fragments(dto, payload)
        self.assertEqual(tuple(canonical), self.bundle.runner_columns)
        self.assertNotIn("age", canonical)
        self.assertEqual(fragment["年齢"], 4)
        self.assertEqual(fragment["性別"], "牡")
        self.assertEqual(fragment["prediction_event_time"], PREDICTION_TIME)

    def test_source_time_received_at_and_content_hash_fail_closed(self):
        identity = (self.rows[0]["race_id"], self.rows[0]["horse_id"])
        bad_payloads = dict(self.payloads)
        bad_payloads[identity] = {**bad_payloads[identity], "__previously_ignored": "mutated"}
        with self.assertRaises(runner.ContractError):
            runner.validate_runner_universe(self.rows, self.bundle, source_payloads=bad_payloads)
        bad = [copy.deepcopy(row) for row in self.rows]
        bad[0]["received_at"] = None
        bad[0]["row_payload_sha256"] = runner.row_payload_hash(bad[0])
        with self.assertRaises(runner.ContractError):
            runner.validate_runner_universe(bad, self.bundle)
        future = [copy.deepcopy(row) for row in self.rows]
        future[0]["source_event_time"] = "2026-08-24T00:00:00Z"
        future[0]["row_payload_sha256"] = runner.row_payload_hash(future[0])
        with self.assertRaises(runner.ContractError):
            runner.validate_runner_universe(future, self.bundle)
        rebound_rows = [copy.deepcopy(row) for row in self.rows]
        incomplete_payload = copy.deepcopy(self.payloads[identity])
        incomplete_payload["source_raw_columns"] = ["race_id", "horse_id"]
        rebound_rows[0]["source_content_sha256"] = runner.source_payload_hash(incomplete_payload)
        rebound_rows[0]["row_payload_sha256"] = runner.row_payload_hash(rebound_rows[0])
        rebound_payloads = dict(self.payloads, **{})
        rebound_payloads[identity] = incomplete_payload
        with self.assertRaises(runner.ContractError):
            runner.validate_runner_universe(rebound_rows, self.bundle, source_payloads=rebound_payloads)

    def test_naive_time_blank_source_version_and_malformed_hash_are_rejected(self):
        mutations = [
            ("source_event_time", "2026-08-20T00:00:00"),
            ("source_event_time", " 2026-08-20T00:00:00Z"),
            ("source_event_time", "2026-08-20T09:00:00+09:00"),
            ("source_event_time", "2026-08-20 00:00:00Z"),
            ("source_version", ""),
            ("source_content_sha256", "A" * 64),
        ]
        for field, value in mutations:
            with self.subTest(field=field):
                bad = [copy.deepcopy(row) for row in self.rows]
                bad[0][field] = value
                bad[0]["row_payload_sha256"] = runner.row_payload_hash(bad[0])
                with self.assertRaises(runner.ContractError):
                    runner.validate_runner_universe(bad, self.bundle)

    def test_identity_hash_and_core_text_types_are_exact(self):
        for field, value in (
            ("race_id", None),
            ("horse_id", 0),
            ("horse_id", False),
            ("horse_id", f" {self.rows[0]['horse_id']} "),
            ("jockey_name", None),
            ("trainer_name", False),
            ("carried_weight", -1.0),
            ("source_version", " SYN-v1 "),
        ):
            bad = [copy.deepcopy(row) for row in self.rows]
            bad[0][field] = value
            bad[0]["row_payload_sha256"] = runner.row_payload_hash(bad[0])
            with self.subTest(field=field, value=value), self.assertRaises(runner.ContractError):
                runner.validate_runner_universe(bad, self.bundle)
        with self.assertRaises(runner.ContractError):
            runner._require_hash(int("1" * 64), "synthetic hash")
        bad_version = [copy.deepcopy(row) for row in self.rows]
        bad_version[0]["source_version"] = 123
        bad_version[0]["row_payload_sha256"] = runner.row_payload_hash(bad_version[0])
        with self.assertRaises(runner.ContractError):
            runner.validate_runner_universe(bad_version, self.bundle)

    def test_parent_is_immutable_and_draw_confirmed_child_is_diff_complete(self):
        parent = copy.deepcopy(self.rows)
        parent_bytes = runner.runner_release_bytes(parent, self.bundle)
        parent_manifest = _parent_runner_manifest(parent, self.bundle)
        target = next(row for row in parent if row["draw_status"] == "scheduled_pending_draw")
        identity = (target["race_id"], target["horse_id"])
        race_rows = [row for row in parent if row["race_id"] == target["race_id"]]
        updates = []
        payloads = {}
        for horse_number, race_row in enumerate(race_rows, start=1):
            update = {
                "race_id": race_row["race_id"],
                "horse_id": race_row["horse_id"],
                "draw_status": "confirmed",
                "entry_stage": "declared_with_draw",
                "runner_status": "declared_active",
                "frame_number": ((horse_number - 1) // 2) + 1,
                "horse_number": horse_number,
                "active_for_feature_materialization": True,
                "change_reason": "draw_confirmed",
                "source_event_time": "2026-08-22T00:00:00Z",
                "received_at": "2026-08-22T00:01:00Z",
                "available_as_of": "2026-08-22T00:02:00Z",
                "source_version": "SYN-CARD-v2",
                "missing_reason": "not_applicable",
            }
            updates.append(update)
            payloads[(race_row["race_id"], race_row["horse_id"])] = _runner_source_payload({**race_row, **update})
        child, diff = runner.build_child_runner_release(
            parent,
            updates,
            self.bundle,
            child_version=2,
            child_as_of=PREDICTION_TIME,
            parent_manifest=parent_manifest,
            parent_source_payloads=self.payloads,
            source_payloads=payloads,
        )
        self.assertEqual(runner.runner_release_bytes(parent, self.bundle), parent_bytes)
        self.assertEqual(len(child), 70)
        self.assertEqual(len(diff), 70)
        changed = next(row for row in child if (row["race_id"], row["horse_id"]) == identity)
        self.assertEqual(changed["draw_status"], "confirmed")
        self.assertEqual(changed["parent_manifest_digest"], parent_manifest["content_hash"])

    def test_scratch_is_retained_and_unapproved_identity_change_is_rejected(self):
        parent_manifest = _parent_runner_manifest(self.rows, self.bundle)
        target = next(row for row in self.rows if row["draw_status"] == "scheduled_pending_draw")
        identity = (target["race_id"], target["horse_id"])
        update = {
            "race_id": identity[0],
            "horse_id": identity[1],
            "draw_status": "scratched",
            "entry_stage": "declared_without_draw",
            "runner_status": "scratched",
            "frame_number": None,
            "horse_number": None,
            "active_for_feature_materialization": False,
            "change_reason": "official_scratch",
            "source_event_time": "2026-08-22T01:00:00Z",
            "received_at": "2026-08-22T01:01:00Z",
            "available_as_of": "2026-08-22T01:02:00Z",
            "source_version": "SYN-CARD-v2",
            "missing_reason": "not_declared_by_source",
        }
        payload = _runner_source_payload({**target, **update})
        child, diff = runner.build_child_runner_release(
            self.rows,
            [update],
            self.bundle,
            child_version=2,
            child_as_of=PREDICTION_TIME,
            parent_manifest=parent_manifest,
            parent_source_payloads=self.payloads,
            source_payloads={identity: payload},
        )
        retained = next(row for row in child if (row["race_id"], row["horse_id"]) == identity)
        self.assertEqual(retained["runner_status"], "scratched")
        self.assertFalse(retained["active_for_feature_materialization"])
        scratch_diff = next(row for row in diff if (row["race_id"], row["horse_id"]) == identity)
        self.assertIn(
            {"field": "runner_status", "old": "declared_active", "new": "scratched"},
            scratch_diff["changes"],
        )
        bad = dict(update, horse_id="SYN-UNDECLARED-HORSE")
        with self.assertRaises(runner.ContractError):
            runner.build_child_runner_release(
                self.rows,
                [bad],
                self.bundle,
                child_version=2,
                child_as_of=PREDICTION_TIME,
                parent_manifest=parent_manifest,
                parent_source_payloads=self.payloads,
                source_payloads={(bad["race_id"], bad["horse_id"]): payload},
            )

    def test_confirmed_to_pending_transition_is_rejected(self):
        parent_manifest = _parent_runner_manifest(self.rows, self.bundle)
        target = next(row for row in self.rows if row["draw_status"] == "confirmed")
        identity = (target["race_id"], target["horse_id"])
        update = {
            "race_id": identity[0],
            "horse_id": identity[1],
            "draw_status": "scheduled_pending_draw",
            "entry_stage": "declared_without_draw",
            "runner_status": "declared_active",
            "frame_number": None,
            "horse_number": None,
            "active_for_feature_materialization": True,
            "change_reason": "forbidden_rollback",
            "source_event_time": "2026-08-22T02:00:00Z",
            "received_at": "2026-08-22T02:01:00Z",
            "available_as_of": "2026-08-22T02:02:00Z",
            "source_version": "SYN-CARD-v2",
            "missing_reason": "not_declared_by_source",
        }
        payload = _runner_source_payload({**target, **update})
        with self.assertRaises(runner.ContractError):
            runner.build_child_runner_release(
                self.rows,
                [update],
                self.bundle,
                child_version=2,
                child_as_of=PREDICTION_TIME,
                parent_manifest=parent_manifest,
                parent_source_payloads=self.payloads,
                source_payloads={identity: payload},
            )

    def test_postdraw_scratch_retains_published_numbers_and_uniqueness(self):
        parent_manifest = _parent_runner_manifest(self.rows, self.bundle)
        target = next(row for row in self.rows if row["draw_status"] == "confirmed")
        identity = (target["race_id"], target["horse_id"])
        update = {
            "race_id": identity[0],
            "horse_id": identity[1],
            "draw_status": "scratched",
            "entry_stage": "declared_with_draw",
            "runner_status": "scratched",
            "frame_number": target["frame_number"],
            "horse_number": target["horse_number"] + 1,
            "active_for_feature_materialization": False,
            "change_reason": "forbidden_number_rewrite",
            "source_event_time": "2026-08-22T02:00:00Z",
            "received_at": "2026-08-22T02:01:00Z",
            "available_as_of": "2026-08-22T02:02:00Z",
            "source_version": "SYN-CARD-v2",
            "missing_reason": "not_applicable",
        }
        payload = _runner_source_payload({**target, **update})
        with self.assertRaises(runner.ContractError):
            runner.build_child_runner_release(
                self.rows,
                [update],
                self.bundle,
                child_version=2,
                child_as_of=PREDICTION_TIME,
                parent_manifest=parent_manifest,
                parent_source_payloads=self.payloads,
                source_payloads={identity: payload},
            )


class EventFeatureAndLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = _bundle()
        rows, cls.runner_payloads = _runner_fixture(cls.bundle)
        cls.runner_rows = tuple(row for row in rows if row["race_id"] == "SYN-RACE-01")[:2]
        cls.runner_source_payloads = {
            (row["race_id"], row["horse_id"]): cls.runner_payloads[(row["race_id"], row["horse_id"])]
            for row in cls.runner_rows
        }
        cls.cells, cls.feature_payloads = _feature_fixture(cls.bundle, cls.runner_rows)

    def test_target_event_release_contains_cards_and_zero_results(self):
        identity = ("SYN-EVENT-RACE", "SYN-EVENT-HORSE")
        payload = {
            "canonical_event_fields": {},
            "source_path": "synthetic/declared_card.jsonl",
            "source_raw_columns": [],
        }
        card = {
            "record_kind": "declared_card",
            "race_id": identity[0],
            "horse_id": identity[1],
            "prediction_event_time": PREDICTION_TIME,
            "source_event_time": SOURCE_TIME,
            "received_at": RECEIVED_TIME,
            "available_as_of": AVAILABLE_TIME,
            "source_version": "SYN-CARD-v1",
            "source_content_sha256": runner.source_payload_hash(payload),
            "missing_reason": "not_applicable",
            "年齢": 4,
            "斤量": 55.0,
            "距離": 1600,
            "場所": "SYN-VENUE",
            "性別": "牡",
            "騎手コード": "SYN-J",
            "調教師コード": "SYN-T",
            "芝・ダ": "芝",
            "クラス名": "SYN-CLASS",
            "トラックコード": "SYN-TRACK",
        }
        payload = runner.canonical_event_source_payload(
            card,
            "synthetic/declared_card.jsonl",
            self.bundle,
        )
        card["source_content_sha256"] = runner.source_payload_hash(payload)
        validated = runner.validate_event_release(
            [card], self.bundle, source_payloads={("declared_card", *identity): payload}, target_partition=True
        )
        self.assertEqual(len(validated), 1)
        bad = dict(card, 確定着順=1)
        with self.assertRaises(runner.ContractError):
            runner.validate_event_release(
                [bad], self.bundle, source_payloads={("declared_card", *identity): payload}, target_partition=True
            )
        market = dict(card, 人気=1)
        with self.assertRaises(runner.ContractError):
            runner.validate_event_release(
                [market], self.bundle, source_payloads={("declared_card", *identity): payload}, target_partition=True
            )

    def test_event_dtypes_and_completed_result_label_projection_are_exact(self):
        identity = ("SYN-EVENT-RACE", "SYN-EVENT-HORSE")
        card = {
            "record_kind": "declared_card",
            "race_id": identity[0],
            "horse_id": identity[1],
            "prediction_event_time": "2026-01-01T06:00:00Z",
            "source_event_time": "2025-12-31T20:00:00Z",
            "received_at": "2025-12-31T20:01:00Z",
            "available_as_of": "2025-12-31T20:02:00Z",
            "source_version": "SYN-CARD-v1",
            "source_content_sha256": "",
            "missing_reason": "not_applicable",
            "年齢": 4,
            "斤量": 55.0,
            "距離": 1600,
            "場所": "SYN-VENUE",
            "性別": "牡",
            "騎手コード": "SYN-J",
            "調教師コード": "SYN-T",
            "芝・ダ": "芝",
            "クラス名": "SYN-CLASS",
            "トラックコード": "SYN-TRACK",
        }
        card_payload = runner.canonical_event_source_payload(
            card,
            "synthetic/declared_card.jsonl",
            self.bundle,
        )
        card["source_content_sha256"] = runner.source_payload_hash(card_payload)
        result = {
            "record_kind": "completed_result",
            "race_id": identity[0],
            "horse_id": identity[1],
            "prediction_event_time": "2026-01-01T06:00:00Z",
            "source_event_time": "2026-01-01T07:00:00Z",
            "received_at": "2026-01-01T07:01:00Z",
            "available_as_of": "2026-01-01T07:02:00Z",
            "source_version": "SYN-RESULT-v1",
            "source_content_sha256": "",
            "missing_reason": "not_applicable",
            "確定着順": 1,
            "1角": 1,
            "2角": 1,
            "4角": 1,
            "result_status": "finished",
            "official_finish_rank_raw": 1,
        }
        result_payload = runner.canonical_event_source_payload(
            result,
            "synthetic/completed_result.jsonl",
            self.bundle,
        )
        result["source_content_sha256"] = runner.source_payload_hash(result_payload)
        validated = runner.validate_event_release(
            [card, result],
            self.bundle,
            source_payloads={
                ("declared_card", *identity): card_payload,
                ("completed_result", *identity): result_payload,
            },
            target_partition=False,
        )
        self.assertEqual(len(validated), 2)

        bad_card = copy.deepcopy(card)
        bad_card["年齢"] = {"odds": 1.5}
        bad_card_payload = copy.deepcopy(card_payload)
        bad_card_payload["canonical_event_fields"]["年齢"] = bad_card["年齢"]
        bad_card["source_content_sha256"] = runner.source_payload_hash(bad_card_payload)
        with self.assertRaises(runner.ContractError):
            runner.validate_event_release(
                [bad_card],
                self.bundle,
                source_payloads={("declared_card", *identity): bad_card_payload},
                target_partition=True,
            )

        bad_result = copy.deepcopy(result)
        bad_result["確定着順"] = 2
        bad_result_payload = copy.deepcopy(result_payload)
        bad_result_payload["canonical_event_fields"]["確定着順"] = 2
        bad_result["source_content_sha256"] = runner.source_payload_hash(bad_result_payload)
        with self.assertRaises(runner.ContractError):
            runner.validate_event_release(
                [card, bad_result],
                self.bundle,
                source_payloads={
                    ("declared_card", *identity): card_payload,
                    ("completed_result", *identity): bad_result_payload,
                },
                target_partition=False,
            )

    def test_exact_88_feature_release_and_complete_lineage(self):
        release = runner.build_feature_release(
            self.cells,
            self.bundle,
            source_payloads=self.feature_payloads,
            runner_rows=self.runner_rows,
            runner_source_payloads=self.runner_source_payloads,
        )
        self.assertEqual(len(release["wide_rows"]), 2)
        self.assertEqual(len(release["lineage_rows"]), 2 * 88)
        self.assertEqual(len({(row["race_id"], row["horse_id"], row["feature_name"]) for row in release["lineage_rows"]}), 176)
        aggregate = next(row for row in release["lineage_rows"] if row["feature_name"] == "出走頭数")
        expected_runner_hashes = {
            runner.runner_feature_dependency_hash(self.runner_source_payloads[(row["race_id"], row["horse_id"])])
            for row in self.runner_rows
        }
        self.assertTrue(expected_runner_hashes.issubset(set(aggregate["dependency_content_sha256_set"])))

    def test_actual_builder_materializes_the_full_synthetic_5_race_70_runner_release(self):
        rows, payloads = _runner_fixture(self.bundle)
        cells, source_payloads = _feature_fixture(self.bundle, rows)
        release = runner.build_feature_release(
            cells,
            self.bundle,
            source_payloads=source_payloads,
            runner_rows=rows,
            runner_source_payloads=payloads,
        )
        self.assertEqual(len(release["wide_rows"]), 70)
        self.assertEqual(len(release["lineage_rows"]), 70 * 88)
        self.assertEqual(
            {row["出走頭数"] for row in release["wide_rows"]},
            {10.0, 14.0, 16.0},
        )
        self.assertTrue(
            all(
                len(row["source_content_sha256_set"]) >= 1
                for row in release["lineage_rows"]
            )
        )

    def test_lineage_missing_future_hash_and_forbidden_dependencies_fail_closed(self):
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                self.cells[:-1], self.bundle, source_payloads=self.feature_payloads, runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )
        bad_lineage = copy.deepcopy(self.cells)
        bad_lineage[0]["lineage_status"] = "uncertified"
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                bad_lineage, self.bundle, source_payloads=self.feature_payloads, runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )
        future = copy.deepcopy(self.cells)
        future[0]["source_event_time"] = "2026-08-24T00:00:00Z"
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                future, self.bundle, source_payloads=self.feature_payloads, runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )
        forbidden = copy.deepcopy(self.cells)
        forbidden[0]["dependency_feature_names"] = ["race_winner_prior_strength"]
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                forbidden, self.bundle, source_payloads=self.feature_payloads, runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )
        denied = copy.deepcopy(self.cells)
        denied[0]["dependency_feature_names"] = ["prev_race_time_value"]
        self.assertIn("prev_race_time_value", self.bundle.denied_features)
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                denied, self.bundle, source_payloads=self.feature_payloads, runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )
        bad_payloads = dict(self.feature_payloads)
        first = (self.cells[0]["race_id"], self.cells[0]["horse_id"], self.cells[0]["feature_name"])
        bad_payloads[first] = {**bad_payloads[first], "mutated": True}
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                self.cells, self.bundle, source_payloads=bad_payloads, runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )

    def test_dependency_evidence_raw_ingress_and_runner_cutoff_are_bound(self):
        cells = copy.deepcopy(self.cells)
        payloads = copy.deepcopy(self.feature_payloads)
        first = cells[0]
        key = (first["race_id"], first["horse_id"], first["feature_name"])
        old_hash = first["source_content_hash"]
        payloads[key]["canonical_source_payload"]["source_raw_columns"].append("odds")
        payloads[key]["canonical_source_payload"]["raw_values"]["odds"] = 1.5
        new_hash = runner.source_payload_hash(payloads[key]["canonical_source_payload"])
        payloads[key]["content_sha256"] = new_hash
        first["source_content_hash"] = new_hash
        first["dependency_content_hashes"] = [
            new_hash if value == old_hash else value for value in first["dependency_content_hashes"]
        ]
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                cells, self.bundle, source_payloads=payloads, runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )

        cells = copy.deepcopy(self.cells)
        dependency_row = {
            "record_kind": "declared_card",
            "race_id": "SYN-PRIOR-CARD-RACE",
            "horse_id": cells[0]["horse_id"],
            "prediction_event_time": "2026-08-22T06:00:00Z",
            "source_event_time": "2026-08-21T00:00:00Z",
            "received_at": "2026-08-21T00:01:00Z",
            "available_as_of": "2026-08-21T00:02:00Z",
            "source_version": "SYN-HISTORY-CARD-v1",
            "source_content_sha256": "",
            "missing_reason": "not_applicable",
            "年齢": 4,
            "斤量": 55.0,
            "距離": 1600,
            "場所": "SYN-VENUE",
            "性別": "牡",
            "騎手コード": "SYN-JOCKEY-001",
            "調教師コード": "SYN-TRAINER-001",
            "芝・ダ": "芝",
            "クラス名": "SYN-CLASS",
            "トラックコード": "SYN-TRACK",
        }
        dependency_payload = runner.canonical_event_source_payload(
            dependency_row,
            "synthetic/prior_declared_card.jsonl",
            self.bundle,
        )
        dependency_hash = runner.source_payload_hash(dependency_payload)
        dependency_row["source_content_sha256"] = dependency_hash
        cells[0]["dependency_content_hashes"].append(dependency_hash)
        cells[0]["dependency_content_hashes"].sort()
        evidence = {
            dependency_hash: _event_source_evidence(dependency_row, dependency_payload)
        }
        release = runner.build_feature_release(
            cells, self.bundle, source_payloads=self.feature_payloads, runner_rows=self.runner_rows,
            runner_source_payloads=self.runner_source_payloads, dependency_evidence=evidence,
        )
        lineage = next(
            row for row in release["lineage_rows"]
            if (row["race_id"], row["horse_id"], row["feature_name"])
            == (cells[0]["race_id"], cells[0]["horse_id"], cells[0]["feature_name"])
        )
        self.assertEqual(lineage["max_available_as_of"], "2026-08-21T00:02:00Z")
        future_evidence = copy.deepcopy(evidence)
        future_evidence[dependency_hash]["available_as_of"] = "2026-08-24T00:00:00Z"
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                cells, self.bundle, source_payloads=self.feature_payloads, runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads, dependency_evidence=future_evidence,
            )

        hidden_market_payload = {
            "raw_values": {"年齢": 4},
            "source_raw_columns": ["年齢"],
            "odds": 1.5,
        }
        hidden_hash = runner.source_payload_hash(hidden_market_payload)
        hidden_cells = copy.deepcopy(self.cells)
        hidden_cells[0]["dependency_content_hashes"].append(hidden_hash)
        hidden_evidence = {
            hidden_hash: {
                **next(iter(evidence.values())),
                "canonical_source_payload": hidden_market_payload,
                "content_sha256": hidden_hash,
            }
        }
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                hidden_cells, self.bundle, source_payloads=self.feature_payloads, runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads, dependency_evidence=hidden_evidence,
            )

        generic_result_payload = {
            "raw_values": {"official_finish_rank_raw": 1},
            "source_raw_columns": ["official_finish_rank_raw"],
        }
        generic_result_hash = runner.source_payload_hash(generic_result_payload)
        generic_result_cells = copy.deepcopy(self.cells)
        generic_result_cells[0]["dependency_content_hashes"].append(generic_result_hash)
        generic_result_evidence = {
            generic_result_hash: {
                **next(iter(evidence.values())),
                "canonical_source_payload": generic_result_payload,
                "content_sha256": generic_result_hash,
            }
        }
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                generic_result_cells,
                self.bundle,
                source_payloads=self.feature_payloads,
                runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
                dependency_evidence=generic_result_evidence,
            )

        for result_alias in (
            "completed_result",
            "current_result",
            "target_result",
            "future_result",
            "official_result",
        ):
            with self.subTest(result_alias=result_alias):
                disguised_result_payload = {
                    "raw_values": {"record_kind": result_alias, "past3_avg_score": 999.0},
                    "source_raw_columns": ["record_kind", "past3_avg_score"],
                }
                disguised_result_hash = runner.source_payload_hash(disguised_result_payload)
                disguised_result_cells = copy.deepcopy(self.cells)
                disguised_result_cells[0]["dependency_content_hashes"].append(disguised_result_hash)
                disguised_result_evidence = {
                    disguised_result_hash: {
                        **next(iter(evidence.values())),
                        "canonical_source_payload": disguised_result_payload,
                        "content_sha256": disguised_result_hash,
                        "source_path": "synthetic/completed_result.jsonl",
                    }
                }
                with self.assertRaises(runner.ContractError):
                    runner.build_feature_release(
                        disguised_result_cells,
                        self.bundle,
                        source_payloads=self.feature_payloads,
                        runner_rows=self.runner_rows,
                        runner_source_payloads=self.runner_source_payloads,
                        dependency_evidence=disguised_result_evidence,
                    )

        direct_result_cells = copy.deepcopy(self.cells)
        direct_result_payloads = copy.deepcopy(self.feature_payloads)
        direct_key = (
            direct_result_cells[0]["race_id"],
            direct_result_cells[0]["horse_id"],
            direct_result_cells[0]["feature_name"],
        )
        direct_payload = direct_result_payloads[direct_key]["canonical_source_payload"]
        direct_payload["raw_values"] = {"official_finish_rank_raw": 1}
        direct_payload["source_raw_columns"] = ["official_finish_rank_raw"]
        direct_hash = runner.source_payload_hash(direct_payload)
        direct_result_payloads[direct_key]["content_sha256"] = direct_hash
        old_direct_hash = direct_result_cells[0]["source_content_hash"]
        direct_result_cells[0]["source_content_hash"] = direct_hash
        direct_result_cells[0]["dependency_content_hashes"] = [
            direct_hash if value == old_direct_hash else value
            for value in direct_result_cells[0]["dependency_content_hashes"]
        ]
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                direct_result_cells,
                self.bundle,
                source_payloads=direct_result_payloads,
                runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )

        nested_market_payload = copy.deepcopy(self.feature_payloads[
            (self.cells[0]["race_id"], self.cells[0]["horse_id"], self.cells[0]["feature_name"])
        ]["canonical_source_payload"])
        nested_market_payload["raw_values"] = {self.cells[0]["feature_name"]: {"odds": 1.5}}
        nested_hash = runner.source_payload_hash(nested_market_payload)
        nested_cells = copy.deepcopy(self.cells)
        nested_cells[0]["dependency_content_hashes"].append(nested_hash)
        nested_evidence = {
            nested_hash: {
                **next(iter(evidence.values())),
                "canonical_source_payload": nested_market_payload,
                "content_sha256": nested_hash,
                "source_path": nested_market_payload["source_path"],
            }
        }
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                nested_cells,
                self.bundle,
                source_payloads=self.feature_payloads,
                runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
                dependency_evidence=nested_evidence,
            )

        wrong_cutoff = copy.deepcopy(self.cells)
        for cell in wrong_cutoff:
            cell["prediction_event_time"] = "2026-08-24T06:00:00Z"
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                wrong_cutoff, self.bundle, source_payloads=self.feature_payloads, runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )

    def test_current_runner_dependencies_cannot_cross_race_boundaries(self):
        all_rows, all_payloads = _runner_fixture(self.bundle)
        race_one = next(row for row in all_rows if row["race_id"] == "SYN-RACE-01")
        race_two = next(row for row in all_rows if row["race_id"] == "SYN-RACE-02")
        rows = (race_one, race_two)
        payloads = {
            (row["race_id"], row["horse_id"]): all_payloads[(row["race_id"], row["horse_id"])]
            for row in rows
        }
        cells, source_payloads = _feature_fixture(self.bundle, rows)
        foreign_hash = runner.runner_feature_dependency_hash(payloads[(race_two["race_id"], race_two["horse_id"])])
        target = next(
            cell
            for cell in cells
            if cell["race_id"] == race_one["race_id"]
            and cell["feature_name"] in runner.RACE_AGGREGATE_FEATURES
        )
        target["dependency_content_hashes"] = sorted(
            {*target["dependency_content_hashes"], foreign_hash}
        )
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                cells,
                self.bundle,
                source_payloads=source_payloads,
                runner_rows=rows,
                runner_source_payloads=payloads,
            )

    def test_certified_historical_result_raw_lineage_is_context_bound(self):
        cells = copy.deepcopy(self.cells)
        target_index = next(
            index
            for index, cell in enumerate(cells)
            if cell["feature_name"] == "past3_avg_score" and cell["horse_id"] == self.runner_rows[0]["horse_id"]
        )
        target = cells[target_index]
        historical_row = {
            "record_kind": "completed_result",
            "race_id": "SYN-PRIOR-RACE",
            "horse_id": target["horse_id"],
            "prediction_event_time": "2026-08-09T06:00:00Z",
            "source_event_time": "2026-08-09T07:00:00Z",
            "received_at": "2026-08-09T07:01:00Z",
            "available_as_of": "2026-08-09T07:02:00Z",
            "source_version": "SYN-RESULT-v1",
            "source_content_sha256": "",
            "missing_reason": "not_applicable",
            "確定着順": 1,
            "1角": 1,
            "2角": 1,
            "4角": 2,
            "result_status": "finished",
            "official_finish_rank_raw": 1,
        }
        historical_payload = runner.canonical_event_source_payload(
            historical_row,
            "synthetic/prior_completed_result.jsonl",
            self.bundle,
        )
        historical_hash = runner.source_payload_hash(historical_payload)
        historical_row["source_content_sha256"] = historical_hash
        target["dependency_content_hashes"].append(historical_hash)
        target["dependency_content_hashes"].sort()
        evidence = {
            historical_hash: _event_source_evidence(historical_row, historical_payload)
        }
        release = runner.build_feature_release(
            cells,
            self.bundle,
            source_payloads=self.feature_payloads,
            runner_rows=self.runner_rows,
            runner_source_payloads=self.runner_source_payloads,
            dependency_evidence=evidence,
        )
        lineage = next(
            row
            for row in release["lineage_rows"]
            if (row["race_id"], row["horse_id"], row["feature_name"])
            == (target["race_id"], target["horse_id"], target["feature_name"])
        )
        self.assertIn(historical_hash, lineage["dependency_content_sha256_set"])

        current_row = copy.deepcopy(historical_row)
        current_row["race_id"] = target["race_id"]
        current_row["source_content_sha256"] = ""
        current_payload = runner.canonical_event_source_payload(
            current_row,
            "synthetic/prior_completed_result.jsonl",
            self.bundle,
        )
        current_hash = runner.source_payload_hash(current_payload)
        current_row["source_content_sha256"] = current_hash
        bad_cells = copy.deepcopy(self.cells)
        bad_target = bad_cells[target_index]
        bad_target["dependency_content_hashes"].append(current_hash)
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                bad_cells,
                self.bundle,
                source_payloads=self.feature_payloads,
                runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
                dependency_evidence={
                    current_hash: _event_source_evidence(current_row, current_payload)
                },
            )

        impossible_payload = copy.deepcopy(historical_payload)
        impossible_payload["canonical_event_fields"]["official_finish_rank_raw"] = -1
        impossible_hash = runner.source_payload_hash(impossible_payload)
        impossible_cells = copy.deepcopy(self.cells)
        impossible_cells[target_index]["dependency_content_hashes"].append(impossible_hash)
        impossible_cells[target_index]["dependency_content_hashes"].sort()
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                impossible_cells,
                self.bundle,
                source_payloads=self.feature_payloads,
                runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
                dependency_evidence={
                    impossible_hash: {
                        **evidence[historical_hash],
                        "canonical_source_payload": impossible_payload,
                        "content_sha256": impossible_hash,
                    }
                },
            )

    def test_forbidden_transform_nonfinite_unknown_and_reordered_schema_fail_closed(self):
        bad_transform = copy.deepcopy(self.cells)
        bad_transform[0]["transform_name"] = "current_result_確定着順"
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                bad_transform,
                self.bundle,
                source_payloads=self.feature_payloads,
                runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )
        nonfinite = copy.deepcopy(self.cells)
        numeric_index = next(i for i, cell in enumerate(nonfinite) if cell["dtype"] == "float64")
        nonfinite[numeric_index]["feature_value"] = float("inf")
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                nonfinite,
                self.bundle,
                source_payloads=self.feature_payloads,
                runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )
        unknown = copy.deepcopy(self.cells)
        unknown[0]["unknown_generated_column"] = 1
        with self.assertRaises(runner.ContractError):
            runner.build_feature_release(
                unknown,
                self.bundle,
                source_payloads=self.feature_payloads,
                runner_rows=self.runner_rows,
                runner_source_payloads=self.runner_source_payloads,
            )
        release = runner.build_feature_release(
            self.cells,
            self.bundle,
            source_payloads=self.feature_payloads,
            runner_rows=self.runner_rows,
            runner_source_payloads=self.runner_source_payloads,
        )
        reordered = dict(reversed(list(release["wide_rows"][0].items())))
        with self.assertRaises(runner.ContractError):
            runner.validate_wide_feature_release(
                [reordered],
                self.bundle,
                lineage_manifest_sha256=release["lineage_manifest_sha256"],
            )

    def test_target_result_mutation_cannot_change_predraw_feature_release(self):
        first = runner.build_feature_release(
            self.cells, self.bundle, source_payloads=self.feature_payloads, runner_rows=self.runner_rows,
            runner_source_payloads=self.runner_source_payloads,
        )
        second = runner.build_feature_release(
            list(reversed(copy.deepcopy(self.cells))), self.bundle, source_payloads=copy.deepcopy(self.feature_payloads), runner_rows=self.runner_rows,
            runner_source_payloads=self.runner_source_payloads,
        )
        self.assertEqual(first["feature_release_sha256"], second["feature_release_sha256"])
        self.assertEqual(first["lineage_manifest_sha256"], second["lineage_manifest_sha256"])
        self.assertFalse(any("result" in name for name in inspect.signature(runner.build_feature_release).parameters))

    def test_draw_only_runner_mutation_cannot_change_predraw_feature_release(self):
        baseline = runner.build_feature_release(
            self.cells, self.bundle, source_payloads=self.feature_payloads, runner_rows=self.runner_rows,
            runner_source_payloads=self.runner_source_payloads,
        )
        changed_rows = [copy.deepcopy(row) for row in self.runner_rows]
        changed_payloads = copy.deepcopy(self.runner_source_payloads)
        for horse_number, row in enumerate(changed_rows, start=1):
            row["draw_status"] = "confirmed"
            row["entry_stage"] = "declared_with_draw"
            row["frame_number"] = 1
            row["horse_number"] = horse_number
            row["missing_reason"] = "not_applicable"
            identity = (row["race_id"], row["horse_id"])
            changed_payloads[identity] = _runner_source_payload(row)
            changed_rows[horse_number - 1] = runner.seal_runner_row(row, changed_payloads[identity])
        updated = runner.build_feature_release(
            self.cells, self.bundle, source_payloads=self.feature_payloads, runner_rows=changed_rows,
            runner_source_payloads=changed_payloads,
        )
        self.assertEqual(baseline["feature_release_sha256"], updated["feature_release_sha256"])
        self.assertEqual(baseline["lineage_manifest_sha256"], updated["lineage_manifest_sha256"])


class LabelManifestAndHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = _bundle()

    def test_label_exact_permutation_is_eligible(self):
        declared, declared_payloads, results, payloads = _result_fixture(["finished", "finished", "finished"], [1, 2, 3])
        ledger = runner.classify_label_eligibility(
            declared, results, self.bundle, declared_source_payloads=declared_payloads, result_source_payloads=payloads
        )
        self.assertEqual(len(ledger), 3)
        self.assertTrue(all(row["race_label_eligible"] for row in ledger))
        self.assertTrue(all(row["row_label_eligible"] for row in ledger))

    def test_label_inputs_are_default_deny_and_race_time_atomic(self):
        declared, declared_payloads, results, payloads = _result_fixture(["finished", "finished"], [1, 2])
        extra_declared = copy.deepcopy(declared)
        extra_declared[0]["target_result"] = 1
        with self.assertRaises(runner.ContractError):
            runner.classify_label_eligibility(
                extra_declared,
                results,
                self.bundle,
                declared_source_payloads=declared_payloads,
                result_source_payloads=payloads,
            )
        extra_result = copy.deepcopy(results)
        extra_result[0]["odds"] = 1.5
        with self.assertRaises(runner.ContractError):
            runner.classify_label_eligibility(
                declared,
                extra_result,
                self.bundle,
                declared_source_payloads=declared_payloads,
                result_source_payloads=payloads,
            )
        mixed_declared = copy.deepcopy(declared)
        mixed_declared[1]["prediction_event_time"] = "2026-01-01T06:30:00Z"
        with self.assertRaises(runner.ContractError):
            runner.classify_label_eligibility(
                mixed_declared,
                results,
                self.bundle,
                declared_source_payloads=declared_payloads,
                result_source_payloads=payloads,
            )
        mixed_results = copy.deepcopy(results)
        mixed_results[1]["prediction_event_time"] = "2026-01-01T05:30:00Z"
        with self.assertRaises(runner.ContractError):
            runner.classify_label_eligibility(
                declared,
                mixed_results,
                self.bundle,
                declared_source_payloads=declared_payloads,
                result_source_payloads=payloads,
            )

        result_columns = self.bundle.input_contract["event_release_schema"]["historical_result_columns"]
        for row, payload in zip(results, payloads.values()):
            self.assertEqual(
                payload["canonical_event_fields"],
                {field: row[field] for field in result_columns},
            )
            self.assertEqual(
                payload["canonical_result_fields"]["official_finish_rank_raw"],
                row["official_finish_rank_raw"],
            )

    def test_dead_heat_abnormal_gap_and_result_missing_are_whole_race_ineligible(self):
        cases = [
            (["finished", "finished", "finished"], [1, 1, 3]),
            (["finished", "abnormal", "finished"], [1, None, 2]),
            (["finished", "finished", "finished"], [1, 3, 4]),
        ]
        for statuses, ranks in cases:
            with self.subTest(statuses=statuses, ranks=ranks):
                declared, declared_payloads, results, payloads = _result_fixture(statuses, ranks)
                ledger = runner.classify_label_eligibility(
                    declared, results, self.bundle, declared_source_payloads=declared_payloads, result_source_payloads=payloads
                )
                self.assertEqual(len(ledger), 3)
                self.assertTrue(all(not row["race_label_eligible"] for row in ledger))
                self.assertTrue(all(not row["row_label_eligible"] for row in ledger))
        declared, declared_payloads, results, payloads = _result_fixture(["finished", "finished", "finished"], [1, 2, 3])
        dropped = results[:-1]
        dropped_payloads = {key: value for key, value in payloads.items() if key[1] != "SYN-LABEL-HORSE-03"}
        ledger = runner.classify_label_eligibility(
            declared,
            dropped,
            self.bundle,
            declared_source_payloads=declared_payloads,
            result_source_payloads=dropped_payloads,
        )
        self.assertTrue(all(not row["race_label_eligible"] for row in ledger))
        self.assertTrue(all(row["ineligibility_reason"] == "result_missing" for row in ledger))
        self.assertEqual(runner._validate_label_artifact_rows(ledger, self.bundle), ledger)

    def test_precutoff_scratch_is_retained_but_not_labeled(self):
        declared, declared_payloads, results, payloads = _result_fixture(["finished", "finished"], [1, 2])
        scratch_payload = {
            "canonical_declared_status": {
                "runner_status": "scratched",
                "precutoff_starter_status": "scratched",
                "status_available_pre_cutoff": True,
            },
            "source_path": "synthetic/declared_status.jsonl",
            "source_raw_columns": ["precutoff_starter_status", "runner_status", "status_available_pre_cutoff"],
        }
        declared.append(
            {
                "race_id": "SYN-LABEL-RACE",
                "horse_id": "SYN-LABEL-HORSE-03",
                "runner_status": "scratched",
                "precutoff_starter_status": "scratched",
                "status_available_pre_cutoff": True,
                "prediction_event_time": "2026-01-01T06:00:00Z",
                "source_event_time": "2025-12-31T21:00:00Z",
                "received_at": "2025-12-31T21:01:00Z",
                "available_as_of": "2025-12-31T21:02:00Z",
                "source_version": "SYN-CARD-v2",
                "source_content_sha256": runner.source_payload_hash(scratch_payload),
                "missing_reason": "not_declared_by_source",
            }
        )
        declared_payloads[("SYN-LABEL-RACE", "SYN-LABEL-HORSE-03")] = scratch_payload
        ledger = runner.classify_label_eligibility(
            declared,
            results,
            self.bundle,
            declared_source_payloads=declared_payloads,
            result_source_payloads=payloads,
        )
        self.assertEqual(len(ledger), 3)
        scratched = next(row for row in ledger if row["horse_id"] == "SYN-LABEL-HORSE-03")
        self.assertFalse(scratched["row_label_eligible"])
        self.assertTrue(scratched["race_label_eligible"])
        self.assertEqual(scratched["ineligibility_reason"], "precutoff_nonstarter_retained")
        bad_declared = copy.deepcopy(declared)
        bad_declared[-1]["status_available_pre_cutoff"] = "false"
        with self.assertRaises(runner.ContractError):
            runner.classify_label_eligibility(
                bad_declared,
                results,
                self.bundle,
                declared_source_payloads=declared_payloads,
                result_source_payloads=payloads,
            )

    def test_certified_precutoff_nonstarter_and_excluded_are_removed_only_from_effective_starters(self):
        for status in ("nonstarter", "excluded"):
            with self.subTest(status=status):
                declared, declared_payloads, results, payloads = _result_fixture(["finished", "finished"], [1, 2])
                horse_id = "SYN-LABEL-HORSE-03"
                status_payload = {
                    "canonical_declared_status": {
                        "runner_status": "declared_active",
                        "precutoff_starter_status": status,
                        "status_available_pre_cutoff": True,
                    },
                    "source_path": "synthetic/declared_status.jsonl",
                    "source_raw_columns": ["precutoff_starter_status", "runner_status", "status_available_pre_cutoff"],
                }
                declared.append(
                    {
                        "race_id": "SYN-LABEL-RACE",
                        "horse_id": horse_id,
                        "runner_status": "declared_active",
                        "precutoff_starter_status": status,
                        "status_available_pre_cutoff": True,
                        "prediction_event_time": "2026-01-01T06:00:00Z",
                        "source_event_time": "2025-12-31T21:00:00Z",
                        "received_at": "2025-12-31T21:01:00Z",
                        "available_as_of": "2025-12-31T21:02:00Z",
                        "source_version": "SYN-STATUS-v1",
                        "source_content_sha256": runner.source_payload_hash(status_payload),
                        "missing_reason": "not_applicable",
                    }
                )
                declared_payloads[("SYN-LABEL-RACE", horse_id)] = status_payload
                ledger = runner.classify_label_eligibility(
                    declared,
                    results,
                    self.bundle,
                    declared_source_payloads=declared_payloads,
                    result_source_payloads=payloads,
                )
                excluded = next(row for row in ledger if row["horse_id"] == horse_id)
                self.assertEqual(excluded["starter_status"], status)
                self.assertFalse(excluded["row_label_eligible"])
                self.assertTrue(excluded["race_label_eligible"])
                self.assertEqual(excluded["ineligibility_reason"], "precutoff_nonstarter_retained")

    def test_target_labels_absent_rejects_any_result_row(self):
        declared, declared_payloads, results, payloads = _result_fixture(["finished", "finished"], [1, 2])
        with self.assertRaises(runner.ContractError):
            runner.classify_label_eligibility(
                declared,
                results,
                self.bundle,
                target_labels_absent=True,
                declared_source_payloads=declared_payloads,
                result_source_payloads=payloads,
            )
        ledger = runner.classify_label_eligibility(
            declared, [], self.bundle, target_labels_absent=True, declared_source_payloads=declared_payloads
        )
        self.assertEqual(len(ledger), len(declared))
        self.assertTrue(all(not row["race_label_eligible"] for row in ledger))

    def test_full_manifest_rejects_unrooted_precutoff_nonstarter_status_evidence(self):
        manifests, evidence = _build_semantic_manifest_set(self.bundle)
        training = next(item for item in manifests if item["manifest_kind"] == "training_source_manifest")
        training_path = training["artifacts"][0]["path"]
        training_rows = [json.loads(line) for line in evidence[training_path].decode("utf-8").splitlines()]
        template_card = next(row for row in training_rows if row["record_kind"] == "declared_card")
        nonstarter_card = copy.deepcopy(template_card)
        nonstarter_card["horse_id"] = "SYN-TRAINING-HORSE-003"
        nonstarter_card["source_content_sha256"] = ""
        nonstarter_payload = runner.canonical_event_source_payload(
            nonstarter_card,
            training["input_source_paths_and_sha256"][0]["path"],
            self.bundle,
        )
        nonstarter_card["source_content_sha256"] = runner.source_payload_hash(nonstarter_payload)
        training_rows.append(nonstarter_card)
        training_raw = runner.canonical_jsonl_bytes(
            training_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id", "record_kind"),
        )
        altered, altered_evidence = _replace_semantic_artifact(
            manifests, evidence, self.bundle, "training_source_manifest", training_raw
        )

        label = next(item for item in altered if item["manifest_kind"] == "label_eligibility_manifest")
        label_path = label["artifacts"][0]["path"]
        label_rows = [json.loads(line) for line in altered_evidence[label_path].decode("utf-8").splitlines()]
        nonstarter_label = {
            "race_id": nonstarter_card["race_id"],
            "horse_id": nonstarter_card["horse_id"],
            "declared_status": "declared_active",
            "starter_status": "nonstarter",
            "official_result_status": None,
            "official_finish_rank_raw": None,
            "row_label_eligible": False,
            "race_label_eligible": True,
            "ineligibility_reason": "precutoff_nonstarter_retained",
            "source_content_sha256": nonstarter_card["source_content_sha256"],
            "row_payload_sha256": "",
        }
        nonstarter_label["row_payload_sha256"] = runner.row_payload_hash(nonstarter_label)
        label_rows.append(nonstarter_label)
        label_raw = runner.canonical_jsonl_bytes(label_rows, sort_key=("race_id", "horse_id"))
        altered, altered_evidence = _replace_semantic_artifact(
            altered, altered_evidence, self.bundle, "label_eligibility_manifest", label_raw
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

    def test_full_manifest_preserves_a_result_missing_race_as_ineligible(self):
        manifests, evidence = _build_semantic_manifest_set(self.bundle)
        training = next(item for item in manifests if item["manifest_kind"] == "training_source_manifest")
        training_path = training["artifacts"][0]["path"]
        training_rows = [json.loads(line) for line in evidence[training_path].decode("utf-8").splitlines()]
        missing_identity = ("SYN-TRAINING-RACE-01", "SYN-TRAINING-HORSE-002")
        training_rows = [
            row
            for row in training_rows
            if not (
                row["record_kind"] == "completed_result"
                and (row["race_id"], row["horse_id"]) == missing_identity
            )
        ]
        training_raw = runner.canonical_jsonl_bytes(
            training_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id", "record_kind"),
        )
        altered, altered_evidence = _replace_semantic_artifact(
            manifests, evidence, self.bundle, "training_source_manifest", training_raw
        )

        label = next(item for item in altered if item["manifest_kind"] == "label_eligibility_manifest")
        label_path = label["artifacts"][0]["path"]
        label_rows = [json.loads(line) for line in altered_evidence[label_path].decode("utf-8").splitlines()]
        card_hash_by_identity = {
            (row["race_id"], row["horse_id"]): row["source_content_sha256"]
            for row in training_rows
            if row["record_kind"] == "declared_card"
        }
        for row in label_rows:
            identity = (row["race_id"], row["horse_id"])
            row["row_label_eligible"] = False
            row["race_label_eligible"] = False
            row["ineligibility_reason"] = "result_missing"
            if identity == missing_identity:
                row["official_result_status"] = None
                row["official_finish_rank_raw"] = None
                row["source_content_sha256"] = card_hash_by_identity[identity]
            row["row_payload_sha256"] = runner.row_payload_hash(row)
        label_raw = runner.canonical_jsonl_bytes(label_rows, sort_key=("race_id", "horse_id"))
        altered, altered_evidence = _replace_semantic_artifact(
            altered, altered_evidence, self.bundle, "label_eligibility_manifest", label_raw
        )
        runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

    def test_manifest_set_is_content_addressed_and_tamper_evident(self):
        manifests, evidence = _build_semantic_manifest_set(self.bundle)
        root = next(item for item in manifests if item["manifest_kind"] == "canonical_root_manifest")
        root_hash = runner.validate_manifest_set(manifests, self.bundle, artifact_bytes_by_path=evidence)
        self.assertEqual(root_hash, root["content_hash"])
        tampered = copy.deepcopy(manifests)
        next(item for item in tampered if item["manifest_kind"] != "canonical_root_manifest")["row_count"] = 999
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(tampered, self.bundle, artifact_bytes_by_path=evidence)
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(manifests, self.bundle, artifact_bytes_by_path={})

    def test_manifest_connectors_reject_cutoff_core_label_feature_and_diff_tampering(self):
        manifests, evidence = _build_semantic_manifest_set(self.bundle, handoff=True)

        training = next(item for item in manifests if item["manifest_kind"] == "training_source_manifest")
        training_payload = {key: value for key, value in training.items() if key != "content_hash"}
        training_payload["source_cutoff"] = "2026-08-20T00:02:00Z"
        with self.assertRaises(runner.ContractError):
            runner.seal_manifest(
                training_payload,
                self.bundle,
                artifact_bytes_by_path={
                    training["artifacts"][0]["path"]: evidence[training["artifacts"][0]["path"]]
                },
            )

        target = next(item for item in manifests if item["manifest_kind"] == "target_source_manifest")
        target_path = target["artifacts"][0]["path"]
        target_rows = [json.loads(line) for line in evidence[target_path].decode("utf-8").splitlines()]
        target_rows[0]["騎手コード"] = "SYN-J-MISMATCH"
        target_raw = runner.canonical_jsonl_bytes(
            target_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id", "record_kind"),
        )
        altered, altered_evidence = _replace_semantic_artifact(
            manifests, evidence, self.bundle, "target_source_manifest", target_raw
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

        target_rows = [json.loads(line) for line in evidence[target_path].decode("utf-8").splitlines()]
        target_rows[0]["年齢"] = 99
        target_raw = runner.canonical_jsonl_bytes(
            target_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id", "record_kind"),
        )
        altered, altered_evidence = _replace_semantic_artifact(
            manifests, evidence, self.bundle, "target_source_manifest", target_raw
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

        label = next(item for item in manifests if item["manifest_kind"] == "label_eligibility_manifest")
        label_path = label["artifacts"][0]["path"]
        label_rows = [json.loads(line) for line in evidence[label_path].decode("utf-8").splitlines()]
        for ordinal, row in enumerate(label_rows, start=1):
            row["race_id"] = "SYN-ALIEN-LABEL-RACE"
            row["horse_id"] = f"SYN-ALIEN-LABEL-HORSE-{ordinal:02d}"
            row["row_payload_sha256"] = runner.row_payload_hash(row)
        label_raw = runner.canonical_jsonl_bytes(label_rows, sort_key=("race_id", "horse_id"))
        altered, altered_evidence = _replace_semantic_artifact(
            manifests, evidence, self.bundle, "label_eligibility_manifest", label_raw
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

        label_rows = [json.loads(line) for line in evidence[label_path].decode("utf-8").splitlines()]
        for row in label_rows:
            row["ineligibility_reason"] = "TOTALLY_UNCLASSIFIED"
            row["row_payload_sha256"] = runner.row_payload_hash(row)
        label_raw = runner.canonical_jsonl_bytes(label_rows, sort_key=("race_id", "horse_id"))
        with self.assertRaises(runner.ContractError):
            _replace_semantic_artifact(
                manifests, evidence, self.bundle, "label_eligibility_manifest", label_raw
            )

        feature = next(item for item in manifests if item["manifest_kind"] == "feature_release_manifest")
        feature_path = feature["artifacts"][0]["path"]
        feature_rows = [json.loads(line) for line in evidence[feature_path].decode("utf-8").splitlines()]
        feature_rows[0][self.bundle.numeric_features[0]] = 1_000_000.0
        feature_raw = runner.canonical_jsonl_bytes(
            feature_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id"),
        )
        altered, altered_evidence = _replace_semantic_artifact(
            manifests, evidence, self.bundle, "feature_release_manifest", feature_raw
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

        feature_rows = [json.loads(line) for line in evidence[feature_path].decode("utf-8").splitlines()]
        feature_rows[0]["出走頭数"] = 3.0
        feature_raw = runner.canonical_jsonl_bytes(
            feature_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id"),
        )
        altered, altered_evidence = _replace_semantic_artifact(
            manifests, evidence, self.bundle, "feature_release_manifest", feature_raw
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

        lineage = next(item for item in manifests if item["manifest_kind"] == "lineage_manifest")
        lineage_path = lineage["artifacts"][0]["path"]
        lineage_rows = [json.loads(line) for line in evidence[lineage_path].decode("utf-8").splitlines()]
        lineage_rows[0]["max_source_event_time"] = "2026-08-24T00:00:00Z"
        lineage_rows[0]["row_payload_sha256"] = runner.row_payload_hash(lineage_rows[0])
        lineage_raw = runner.canonical_jsonl_bytes(
            lineage_rows, sort_key=("race_id", "horse_id", "feature_name")
        )
        with self.assertRaises(runner.ContractError):
            _replace_semantic_artifact(
                manifests, evidence, self.bundle, "lineage_manifest", lineage_raw
            )

        lineage_rows = [json.loads(line) for line in evidence[lineage_path].decode("utf-8").splitlines()]
        lineage_rows[0]["source_versions"] = [""]
        lineage_rows[0]["row_payload_sha256"] = runner.row_payload_hash(lineage_rows[0])
        lineage_raw = runner.canonical_jsonl_bytes(
            lineage_rows, sort_key=("race_id", "horse_id", "feature_name")
        )
        with self.assertRaises(runner.ContractError):
            _replace_semantic_artifact(
                manifests, evidence, self.bundle, "lineage_manifest", lineage_raw
            )

        feature_rows = [json.loads(line) for line in evidence[feature_path].decode("utf-8").splitlines()]
        feature_rows[0][self.bundle.categorical_features[0]] = ""
        feature_raw = runner.canonical_jsonl_bytes(
            feature_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id"),
        )
        with self.assertRaises(runner.ContractError):
            _replace_semantic_artifact(
                manifests, evidence, self.bundle, "feature_release_manifest", feature_raw
            )

        lineage_rows = [json.loads(line) for line in evidence[lineage_path].decode("utf-8").splitlines()]
        target_lineage = next(row for row in lineage_rows if row["feature_name"] == "past3_avg_score")
        target_lineage["missing_reason"] = "source_value_missing"
        target_lineage["row_payload_sha256"] = runner.row_payload_hash(target_lineage)
        lineage_raw = runner.canonical_jsonl_bytes(
            lineage_rows, sort_key=("race_id", "horse_id", "feature_name")
        )
        altered, altered_evidence = _replace_semantic_artifact(
            manifests, evidence, self.bundle, "lineage_manifest", lineage_raw
        )
        new_lineage = next(item for item in altered if item["manifest_kind"] == "lineage_manifest")
        altered_feature = next(item for item in altered if item["manifest_kind"] == "feature_release_manifest")
        altered_feature_path = altered_feature["artifacts"][0]["path"]
        altered_feature_rows = [
            json.loads(line) for line in altered_evidence[altered_feature_path].decode("utf-8").splitlines()
        ]
        for row in altered_feature_rows:
            row["lineage_manifest_sha256"] = new_lineage["artifacts"][0]["sha256"]
        altered_feature_raw = runner.canonical_jsonl_bytes(
            altered_feature_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id"),
        )
        altered, altered_evidence = _replace_semantic_artifact(
            altered,
            altered_evidence,
            self.bundle,
            "feature_release_manifest",
            altered_feature_raw,
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

        lineage_rows = [json.loads(line) for line in evidence[lineage_path].decode("utf-8").splitlines()]
        orphan = lineage_rows[0]
        orphan_source_hash = "f" * 64
        original_feature_rows = [
            json.loads(line) for line in evidence[feature_path].decode("utf-8").splitlines()
        ]
        orphan_wide = next(
            row
            for row in original_feature_rows
            if (row["race_id"], row["horse_id"]) == (orphan["race_id"], orphan["horse_id"])
        )
        orphan["source_content_sha256_set"] = [orphan_source_hash]
        orphan["dependency_content_sha256_set"] = sorted(
            {
                orphan_source_hash,
                runner.feature_value_binding_hash(
                    race_id=orphan["race_id"],
                    horse_id=orphan["horse_id"],
                    feature_name=orphan["feature_name"],
                    feature_value=orphan_wide[orphan["feature_name"]],
                    feature_dtype=orphan["feature_dtype"],
                    prediction_event_time=orphan["prediction_event_time"],
                    transformation_name=orphan["transformation_name"],
                    transformation_version=orphan["transformation_version"],
                    transformation_code_sha256=orphan["transformation_code_sha256"],
                    source_content_sha256_set=[orphan_source_hash],
                ),
            }
        )
        orphan["row_payload_sha256"] = runner.row_payload_hash(orphan)
        orphan_lineage_raw = runner.canonical_jsonl_bytes(
            lineage_rows, sort_key=("race_id", "horse_id", "feature_name")
        )
        altered, altered_evidence = _replace_semantic_artifact(
            manifests, evidence, self.bundle, "lineage_manifest", orphan_lineage_raw
        )
        new_lineage = next(item for item in altered if item["manifest_kind"] == "lineage_manifest")
        altered_feature = next(item for item in altered if item["manifest_kind"] == "feature_release_manifest")
        altered_feature_path = altered_feature["artifacts"][0]["path"]
        altered_feature_rows = [
            json.loads(line) for line in altered_evidence[altered_feature_path].decode("utf-8").splitlines()
        ]
        for row in altered_feature_rows:
            row["lineage_manifest_sha256"] = new_lineage["artifacts"][0]["sha256"]
        altered_feature_raw = runner.canonical_jsonl_bytes(
            altered_feature_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id"),
        )
        altered, altered_evidence = _replace_semantic_artifact(
            altered,
            altered_evidence,
            self.bundle,
            "feature_release_manifest",
            altered_feature_raw,
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

        diff = next(item for item in manifests if item["manifest_kind"] == "release_diff_manifest")
        diff_path = diff["artifacts"][0]["path"]
        diff_rows = [json.loads(line) for line in evidence[diff_path].decode("utf-8").splitlines()]
        diff_rows[0]["new_row_payload_sha256"] = HASH_A
        diff_raw = runner.canonical_jsonl_bytes(diff_rows, sort_key=("race_id", "horse_id"))
        altered, altered_evidence = _replace_semantic_artifact(
            manifests, evidence, self.bundle, "release_diff_manifest", diff_raw
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

        altered = copy.deepcopy(manifests)
        diff_index = next(i for i, item in enumerate(altered) if item["manifest_kind"] == "release_diff_manifest")
        diff_payload = {key: value for key, value in altered[diff_index].items() if key != "content_hash"}
        diff_payload["source_cutoff"] = "2026-08-22T00:00:00Z"
        altered[diff_index] = runner.seal_manifest(
            diff_payload,
            self.bundle,
            artifact_bytes_by_path={diff_path: evidence[diff_path]},
        )
        root_index = next(i for i, item in enumerate(altered) if item["manifest_kind"] == "canonical_root_manifest")
        root_payload = {key: value for key, value in altered[root_index].items() if key != "content_hash"}
        root_payload["dependency_manifest_digests"] = {
            item["manifest_kind"]: item["content_hash"]
            for item in altered
            if item["manifest_kind"] != "canonical_root_manifest"
        }
        altered[root_index], root_evidence = _seal_payload_with_artifact_bytes(root_payload, self.bundle)
        altered_evidence = dict(evidence)
        altered_evidence.update(root_evidence)
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

    def test_manifest_lineage_sources_are_identity_provenance_and_transform_bound(self):
        manifests, evidence = _build_semantic_manifest_set(self.bundle, handoff=True)
        by_kind = {item["manifest_kind"]: item for item in manifests}
        lineage_path = by_kind["lineage_manifest"]["artifacts"][0]["path"]
        feature_path = by_kind["feature_release_manifest"]["artifacts"][0]["path"]
        target_path = by_kind["target_source_manifest"]["artifacts"][0]["path"]
        base_lineage = [
            json.loads(line) for line in evidence[lineage_path].decode("utf-8").splitlines()
        ]
        wide_rows = [
            json.loads(line) for line in evidence[feature_path].decode("utf-8").splitlines()
        ]
        target_rows = [
            json.loads(line) for line in evidence[target_path].decode("utf-8").splitlines()
        ]
        wide_by_identity = {
            (row["race_id"], row["horse_id"]): row for row in wide_rows
        }
        target_hash_by_identity = {
            (row["race_id"], row["horse_id"]): row["source_content_sha256"]
            for row in target_rows
        }
        focal = next(
            row
            for row in base_lineage
            if row["feature_name"] not in runner.RACE_AGGREGATE_FEATURES
        )
        focal_identity = (focal["race_id"], focal["horse_id"])
        other_identity = next(
            identity
            for identity in target_hash_by_identity
            if identity[0] == focal_identity[0] and identity != focal_identity
        )

        def bind_value(row):
            value = wide_by_identity[(row["race_id"], row["horse_id"])][row["feature_name"]]
            binding = runner.feature_value_binding_hash(
                race_id=row["race_id"],
                horse_id=row["horse_id"],
                feature_name=row["feature_name"],
                feature_value=value,
                feature_dtype=row["feature_dtype"],
                prediction_event_time=row["prediction_event_time"],
                transformation_name=row["transformation_name"],
                transformation_version=row["transformation_version"],
                transformation_code_sha256=row["transformation_code_sha256"],
                source_content_sha256_set=row["source_content_sha256_set"],
            )
            row["dependency_content_sha256_set"] = sorted(
                {*row["source_content_sha256_set"], binding}
            )
            row["row_payload_sha256"] = runner.row_payload_hash(row)

        cross_horse = copy.deepcopy(base_lineage)
        attacked = next(
            row
            for row in cross_horse
            if (row["race_id"], row["horse_id"], row["feature_name"])
            == (focal["race_id"], focal["horse_id"], focal["feature_name"])
        )
        attacked["source_content_sha256_set"] = [target_hash_by_identity[other_identity]]
        bind_value(attacked)
        altered, altered_evidence = _replace_lineage_and_rebind_feature(
            manifests, evidence, self.bundle, cross_horse
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

        runner_manifest = by_kind["runner_universe_manifest"]
        runner_path = runner_manifest["artifacts"][0]["path"]
        runner_source_path = runner_manifest["input_source_paths_and_sha256"][0]["path"]
        runner_rows = [
            json.loads(line) for line in evidence[runner_path].decode("utf-8").splitlines()
        ]
        confirmed_runner = next(row for row in runner_rows if row["draw_status"] == "confirmed")
        confirmed_identity = (confirmed_runner["race_id"], confirmed_runner["horse_id"])
        full_runner_source = copy.deepcopy(base_lineage)
        attacked = next(
            row
            for row in full_runner_source
            if (row["race_id"], row["horse_id"]) == confirmed_identity
            and row["feature_name"] not in runner.RACE_AGGREGATE_FEATURES
        )
        # The full runner payload commits draw_status/frame_number/horse_number.
        # Even a completely resealed manifest must reject it as model lineage;
        # only the explicit runner_feature_safe projection is admissible.
        attacked["source_paths"] = [runner_source_path]
        attacked["source_versions"] = [confirmed_runner["source_version"]]
        attacked["source_content_sha256_set"] = [confirmed_runner["source_content_sha256"]]
        attacked["max_source_event_time"] = confirmed_runner["source_event_time"]
        attacked["max_received_at"] = confirmed_runner["received_at"]
        attacked["max_available_as_of"] = confirmed_runner["available_as_of"]
        bind_value(attacked)
        altered, altered_evidence = _replace_lineage_and_rebind_feature(
            manifests, evidence, self.bundle, full_runner_source
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(
                altered,
                self.bundle,
                artifact_bytes_by_path=altered_evidence,
            )

        forged_provenance = copy.deepcopy(base_lineage)
        attacked = forged_provenance[0]
        attacked["source_paths"] = ["synthetic/forged_pre_cutoff_source.jsonl"]
        attacked["source_versions"] = ["FORGED-SOURCE-v999"]
        attacked["row_payload_sha256"] = runner.row_payload_hash(attacked)
        altered, altered_evidence = _replace_lineage_and_rebind_feature(
            manifests, evidence, self.bundle, forged_provenance
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

        extra_dependency = copy.deepcopy(base_lineage)
        attacked = extra_dependency[0]
        attacked["dependency_content_sha256_set"] = sorted(
            {*attacked["dependency_content_sha256_set"], HASH_D}
        )
        attacked["row_payload_sha256"] = runner.row_payload_hash(attacked)
        altered, altered_evidence = _replace_lineage_and_rebind_feature(
            manifests, evidence, self.bundle, extra_dependency
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

        forged_transform = copy.deepcopy(base_lineage)
        attacked = forged_transform[0]
        attacked["transformation_name"] = "forged_safe_transform"
        attacked["transformation_version"] = "FORGED-v999"
        attacked["transformation_code_sha256"] = "f" * 64
        bind_value(attacked)
        with self.assertRaises(runner.ContractError):
            _replace_lineage_and_rebind_feature(
                manifests, evidence, self.bundle, forged_transform
            )

    def test_training_and_target_race_universes_must_be_disjoint(self):
        manifests, evidence = _build_semantic_manifest_set(self.bundle, handoff=True)
        by_kind = {item["manifest_kind"]: item for item in manifests}
        target_path = by_kind["target_source_manifest"]["artifacts"][0]["path"]
        target_rows = [
            json.loads(line) for line in evidence[target_path].decode("utf-8").splitlines()
        ]
        target_race_id = target_rows[0]["race_id"]
        training = by_kind["training_source_manifest"]
        training_path = training["artifacts"][0]["path"]
        training_source_path = training["input_source_paths_and_sha256"][0]["path"]
        training_rows = [
            json.loads(line) for line in evidence[training_path].decode("utf-8").splitlines()
        ]
        for row in training_rows:
            row["race_id"] = target_race_id
            row["source_content_sha256"] = ""
            payload = runner.canonical_event_source_payload(
                row, training_source_path, self.bundle
            )
            row["source_content_sha256"] = runner.source_payload_hash(payload)
        training_raw = runner.canonical_jsonl_bytes(
            training_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id", "record_kind"),
        )
        altered, altered_evidence = _replace_semantic_artifact(
            manifests, evidence, self.bundle, "training_source_manifest", training_raw
        )
        result_hash_by_identity = {
            (row["race_id"], row["horse_id"]): row["source_content_sha256"]
            for row in training_rows
            if row["record_kind"] == "completed_result"
        }
        label = next(
            item for item in altered if item["manifest_kind"] == "label_eligibility_manifest"
        )
        label_path = label["artifacts"][0]["path"]
        label_rows = [
            json.loads(line)
            for line in altered_evidence[label_path].decode("utf-8").splitlines()
        ]
        for row in label_rows:
            row["race_id"] = target_race_id
            row["source_content_sha256"] = result_hash_by_identity[
                (row["race_id"], row["horse_id"])
            ]
            row["row_payload_sha256"] = runner.row_payload_hash(row)
        label_raw = runner.canonical_jsonl_bytes(
            label_rows, sort_key=("race_id", "horse_id")
        )
        altered, altered_evidence = _replace_semantic_artifact(
            altered,
            altered_evidence,
            self.bundle,
            "label_eligibility_manifest",
            label_raw,
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(altered, self.bundle, artifact_bytes_by_path=altered_evidence)

    def test_manifest_artifact_schema_and_release_composition_are_default_deny(self):
        extra = _manifest_payload("runner_universe_manifest")
        extra["artifacts"][0]["unexpected"] = True
        with self.assertRaises(runner.ContractError):
            runner.seal_manifest(
                extra,
                self.bundle,
                artifact_bytes_by_path={extra["artifacts"][0]["path"]: b""},
            )

        unsorted_sources = _manifest_payload("runner_universe_manifest")
        unsorted_sources["input_source_paths_and_sha256"] = [
            {"path": "synthetic/z-source.json", "sha256": HASH_B},
            {"path": "synthetic/a-source.json", "sha256": HASH_A},
        ]
        with self.assertRaises(runner.ContractError):
            _seal_payload_with_artifact_bytes(unsorted_sources, self.bundle)

        unexpected_bucket = _manifest_payload("runner_universe_manifest")
        unexpected_bucket["as_of_verdict_counts"] = {
            "certified_asof_safe": 1,
            "unapproved_zero_bucket": 0,
        }
        with self.assertRaises(runner.ContractError):
            _seal_payload_with_artifact_bytes(unexpected_bucket, self.bundle)

        manifests, _ = _build_semantic_manifest_set(self.bundle)
        leaves = [item for item in manifests if item["manifest_kind"] != "canonical_root_manifest"]
        feature_index = next(i for i, item in enumerate(leaves) if item["manifest_kind"] == "feature_release_manifest")
        lineage = next(item for item in leaves if item["manifest_kind"] == "lineage_manifest")
        feature_payload = _manifest_payload(
            "feature_release_manifest",
            row_count=1,
            race_count=1,
            runner_count=1,
            version=2,
            parent=HASH_A,
        )
        leaves[feature_index], _ = _seal_payload_with_artifact_bytes(
            feature_payload,
            self.bundle,
            lineage_hash=lineage["artifacts"][0]["sha256"],
        )
        root_payload = _manifest_payload(
            "canonical_root_manifest",
            row_count=sum(leaf["row_count"] for leaf in leaves),
            race_count=1,
            runner_count=1,
            artifact_name="canonical_root",
            artifact_path="outputs/research/SYN/root-mixed.json",
        )
        root_payload["dependency_manifest_digests"] = {
            leaf["manifest_kind"]: leaf["content_hash"] for leaf in leaves
        }
        root, _ = _seal_payload_with_artifact_bytes(root_payload, self.bundle)
        manifests = [*leaves, root]
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest_set(
                manifests,
                self.bundle,
                artifact_bytes_by_path=_artifact_evidence_for_manifests(manifests),
            )

    def test_manifest_paths_and_label_reconciliation_role_are_frozen(self):
        self.assertIn("label_eligibility_manifest", self.bundle.config["required_manifest_kinds"])
        forbidden_source = _manifest_payload("runner_universe_manifest")
        forbidden_source["input_source_paths_and_sha256"] = [
            {"path": "date/raw/netkeiba_odds_market.csv", "sha256": HASH_A}
        ]
        with self.assertRaises(runner.ContractError):
            _seal_payload_with_artifact_bytes(forbidden_source, self.bundle)

        forbidden_output = _manifest_payload(
            "runner_universe_manifest",
            artifact_path="models/Champion.pkl",
        )
        with self.assertRaises(runner.ContractError):
            _seal_payload_with_artifact_bytes(forbidden_output, self.bundle)

        for alias in ("latest.jsonl", "LATEST.jsonl"):
            mutable_alias = _manifest_payload(
                "runner_universe_manifest",
                artifact_path=f"outputs/research/SYN/{alias}",
            )
            with self.subTest(alias=alias), self.assertRaises(runner.ContractError):
                _seal_payload_with_artifact_bytes(mutable_alias, self.bundle)

    def test_manifest_artifact_bytes_are_verified_when_available(self):
        manifests, evidence = _build_semantic_manifest_set(self.bundle)
        manifest = next(item for item in manifests if item["manifest_kind"] == "runner_universe_manifest")
        artifact_path = manifest["artifacts"][0]["path"]
        artifact_bytes = evidence[artifact_path]
        runner.validate_manifest(
            manifest,
            bundle=self.bundle,
            artifact_bytes_by_path={artifact_path: artifact_bytes},
        )
        with self.assertRaises(runner.ContractError):
            runner.validate_manifest(
                manifest,
                bundle=self.bundle,
                artifact_bytes_by_path={artifact_path: b"tampered"},
            )

    def test_manifest_rejects_noncanonical_and_wrong_semantic_artifact_bytes(self):
        noncanonical = b'{ "z": 1, "a": 2 }\n'
        payload = _manifest_payload(
            "label_eligibility_manifest",
            artifact_hash=hashlib.sha256(noncanonical).hexdigest(),
        )
        payload["output_artifact_paths_and_sha256"] = [
            {"path": payload["artifacts"][0]["path"], "sha256": hashlib.sha256(noncanonical).hexdigest()}
        ]
        with self.assertRaises(runner.ContractError):
            runner.seal_manifest(
                payload,
                self.bundle,
                artifact_bytes_by_path={payload["artifacts"][0]["path"]: noncanonical},
            )

        arbitrary = runner.canonical_json_bytes({"odds": 1.5, "target_result": 1}) + b"\n"
        payload = _manifest_payload(
            "label_eligibility_manifest",
            artifact_hash=hashlib.sha256(arbitrary).hexdigest(),
        )
        payload["output_artifact_paths_and_sha256"] = [
            {"path": payload["artifacts"][0]["path"], "sha256": hashlib.sha256(arbitrary).hexdigest()}
        ]
        with self.assertRaises(runner.ContractError):
            runner.seal_manifest(
                payload,
                self.bundle,
                artifact_bytes_by_path={payload["artifacts"][0]["path"]: arbitrary},
            )

    def test_manifest_safety_flags_reject_bool_float_or_nonzero_stake(self):
        base = _manifest_payload(
            "runner_universe_manifest",
            artifact_name="runner_universe_release",
            artifact_path="outputs/research/SYN/a",
            release_family_id="SYN",
        )
        for field, value in (("formal_buy", True), ("send_order", True), ("stake", 0.0), ("stake", 1)):
            with self.subTest(field=field, value=value):
                bad = dict(base)
                bad[field] = value
                with self.assertRaises(runner.ContractError):
                    runner.seal_manifest(
                        bad,
                        self.bundle,
                        artifact_bytes_by_path={bad["artifacts"][0]["path"]: b""},
                    )

    def test_actual_feature_builder_flows_through_manifest_set_and_exp033_handoff(self):
        manifests, evidence = _build_semantic_manifest_set(self.bundle, handoff=True)
        manifests, evidence, release = _connect_actual_feature_builder(
            manifests, evidence, self.bundle
        )
        self.assertEqual(len(release["wide_rows"]), 70)
        self.assertEqual(len(release["lineage_rows"]), 70 * 88)
        root_digest = runner.validate_manifest_set(
            manifests,
            self.bundle,
            artifact_bytes_by_path=evidence,
        )
        expected = runner.derive_exp033_handoff_bindings(
            manifests,
            self.bundle,
            stage="predraw",
            artifact_bytes_by_path=evidence,
            synthetic_fixture=True,
        )
        contract = self.bundle.config["exp033_handoff"]
        handoff = {
            "contract_version": contract["contract_version"],
            "consumer_experiment_id": "EXP-20260821-033",
            **expected,
            "exp033_allowlist_sha256": contract["exp033_allowlist_sha256"],
            "exp033_denylist_sha256": contract["exp033_denylist_sha256"],
            "target_runner_completeness": 1.0,
            "source_time_completeness": 1.0,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        runner.validate_exp033_handoff(
            handoff,
            self.bundle,
            manifests=manifests,
            stage="predraw",
            artifact_bytes_by_path=evidence,
            synthetic_fixture=True,
        )
        self.assertEqual(expected["canonical_root_manifest_sha256"], root_digest)

    def test_exp033_handoff_requires_exact_sealed_bindings(self):
        contract = self.bundle.config["exp033_handoff"]
        manifests, evidence = _build_semantic_manifest_set(self.bundle, handoff=True)
        with self.assertRaises(runner.ContractError):
            runner.derive_exp033_handoff_bindings(
                manifests,
                self.bundle,
                stage="predraw",
                artifact_bytes_by_path=evidence,
            )
        expected = runner.derive_exp033_handoff_bindings(
            manifests,
            self.bundle,
            stage="predraw",
            artifact_bytes_by_path=evidence,
            synthetic_fixture=True,
        )
        handoff = {
            "contract_version": contract["contract_version"],
            "consumer_experiment_id": "EXP-20260821-033",
            **expected,
            "exp033_allowlist_sha256": contract["exp033_allowlist_sha256"],
            "exp033_denylist_sha256": contract["exp033_denylist_sha256"],
            "target_runner_completeness": 1.0,
            "source_time_completeness": 1.0,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
        }
        runner.validate_exp033_handoff(
            handoff,
            self.bundle,
            manifests=manifests,
            stage="predraw",
            artifact_bytes_by_path=evidence,
            synthetic_fixture=True,
        )
        bad = dict(handoff, feature_input_release_sha256=HASH_B)
        with self.assertRaises(runner.ContractError):
            runner.validate_exp033_handoff(
                bad,
                self.bundle,
                manifests=manifests,
                stage="predraw",
                artifact_bytes_by_path=evidence,
                synthetic_fixture=True,
            )
        fake = dict(evidence)
        fake[next(iter(fake))] = b'{"row_count":1}\n'
        with self.assertRaises(runner.ContractError):
            runner.derive_exp033_handoff_bindings(
                manifests,
                self.bundle,
                stage="predraw",
                artifact_bytes_by_path=fake,
                synthetic_fixture=True,
            )
        with self.assertRaises(runner.ContractError):
            runner.derive_exp033_handoff_bindings(
                manifests,
                self.bundle,
                stage="draw_confirmed",
                artifact_bytes_by_path=evidence,
                synthetic_fixture=True,
            )

    def test_draw_confirmed_handoff_binds_parent_root_and_preserves_predraw_inputs(self):
        parent_manifests, parent_evidence = _build_semantic_manifest_set(self.bundle, handoff=True)
        child_manifests, child_evidence = _build_draw_confirmed_child_manifest_set(
            parent_manifests,
            parent_evidence,
            self.bundle,
        )
        parent_by_kind = {item["manifest_kind"]: item for item in parent_manifests}
        parent_runner_digest = parent_by_kind["runner_universe_manifest"]["content_hash"]
        parent_root_digest = parent_by_kind["canonical_root_manifest"]["content_hash"]
        bindings = runner.derive_exp033_handoff_bindings(
            child_manifests,
            self.bundle,
            stage="draw_confirmed",
            artifact_bytes_by_path=child_evidence,
            synthetic_fixture=True,
            parent_manifests=parent_manifests,
            parent_artifact_bytes_by_path=parent_evidence,
            expected_parent_manifest_digest=parent_runner_digest,
            expected_parent_root_manifest_digest=parent_root_digest,
        )
        self.assertEqual(bindings["stage"], "draw_confirmed")
        self.assertEqual(bindings["target_runner_completeness"], 1.0)
        with self.assertRaises(runner.ContractError):
            runner.derive_exp033_handoff_bindings(
                child_manifests,
                self.bundle,
                stage="draw_confirmed",
                artifact_bytes_by_path=child_evidence,
                synthetic_fixture=True,
                parent_manifests=parent_manifests,
                parent_artifact_bytes_by_path=parent_evidence,
                expected_parent_manifest_digest=parent_runner_digest,
                expected_parent_root_manifest_digest=HASH_A,
            )

        child_by_kind = {item["manifest_kind"]: item for item in child_manifests}
        training_path = child_by_kind["training_source_manifest"]["artifacts"][0]["path"]
        training_rows = [json.loads(line) for line in child_evidence[training_path].decode("utf-8").splitlines()]
        first_card = next(row for row in training_rows if row["record_kind"] == "declared_card")
        first_card["年齢"] += 1
        changed_training_raw = runner.canonical_jsonl_bytes(
            training_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id", "record_kind"),
        )
        changed_training_manifests, changed_training_evidence = _replace_semantic_artifact(
            child_manifests,
            child_evidence,
            self.bundle,
            "training_source_manifest",
            changed_training_raw,
        )
        with self.assertRaises(runner.ContractError):
            runner.derive_exp033_handoff_bindings(
                changed_training_manifests,
                self.bundle,
                stage="draw_confirmed",
                artifact_bytes_by_path=changed_training_evidence,
                synthetic_fixture=True,
                parent_manifests=parent_manifests,
                parent_artifact_bytes_by_path=parent_evidence,
                expected_parent_manifest_digest=parent_runner_digest,
                expected_parent_root_manifest_digest=parent_root_digest,
            )

        lineage_path = child_by_kind["lineage_manifest"]["artifacts"][0]["path"]
        lineage_rows = [json.loads(line) for line in child_evidence[lineage_path].decode("utf-8").splitlines()]
        target_lineage = next(row for row in lineage_rows if row["feature_name"] == "past3_avg_score")
        feature_path = child_by_kind["feature_release_manifest"]["artifacts"][0]["path"]
        feature_rows = [json.loads(line) for line in child_evidence[feature_path].decode("utf-8").splitlines()]
        target_feature = next(
            row
            for row in feature_rows
            if (row["race_id"], row["horse_id"])
            == (target_lineage["race_id"], target_lineage["horse_id"])
        )
        target_feature["past3_avg_score"] += 1000.0
        replacement_binding = runner.feature_value_binding_hash(
            race_id=target_lineage["race_id"],
            horse_id=target_lineage["horse_id"],
            feature_name=target_lineage["feature_name"],
            feature_value=target_feature["past3_avg_score"],
            feature_dtype=target_lineage["feature_dtype"],
            prediction_event_time=target_lineage["prediction_event_time"],
            transformation_name=target_lineage["transformation_name"],
            transformation_version=target_lineage["transformation_version"],
            transformation_code_sha256=target_lineage["transformation_code_sha256"],
            source_content_sha256_set=target_lineage["source_content_sha256_set"],
        )
        target_lineage["dependency_content_sha256_set"] = sorted(
            {*target_lineage["dependency_content_sha256_set"], replacement_binding}
        )
        target_lineage["row_payload_sha256"] = runner.row_payload_hash(target_lineage)
        changed_lineage_raw = runner.canonical_jsonl_bytes(
            lineage_rows,
            sort_key=("race_id", "horse_id", "feature_name"),
        )
        changed_feature_manifests, changed_feature_evidence = _replace_semantic_artifact(
            child_manifests,
            child_evidence,
            self.bundle,
            "lineage_manifest",
            changed_lineage_raw,
        )
        changed_by_kind = {item["manifest_kind"]: item for item in changed_feature_manifests}
        new_lineage_hash = changed_by_kind["lineage_manifest"]["artifacts"][0]["sha256"]
        for row in feature_rows:
            row["lineage_manifest_sha256"] = new_lineage_hash
        changed_feature_raw = runner.canonical_jsonl_bytes(
            feature_rows,
            sort_key=("prediction_event_time", "race_id", "horse_id"),
        )
        changed_feature_manifests, changed_feature_evidence = _replace_semantic_artifact(
            changed_feature_manifests,
            changed_feature_evidence,
            self.bundle,
            "feature_release_manifest",
            changed_feature_raw,
        )
        with self.assertRaises(runner.ContractError):
            runner.derive_exp033_handoff_bindings(
                changed_feature_manifests,
                self.bundle,
                stage="draw_confirmed",
                artifact_bytes_by_path=changed_feature_evidence,
                synthetic_fixture=True,
                parent_manifests=parent_manifests,
                parent_artifact_bytes_by_path=parent_evidence,
                expected_parent_manifest_digest=parent_runner_digest,
                expected_parent_root_manifest_digest=parent_root_digest,
            )
class RealDataFailCloseTests(unittest.TestCase):
    def test_real_materialization_refuses_before_loader_or_path_access(self):
        called = []

        def loader(*args, **kwargs):
            called.append((args, kwargs))
            raise AssertionError("loader must never be called")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "does-not-exist-source.json"
            output = root / "must-not-be-created"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = runner.main(
                    [
                        "materialize-real-data",
                        "--config",
                        str(root / "disguised-config.json"),
                        "--source-manifest",
                        str(source),
                        "--output-root",
                        str(output),
                    ],
                    real_source_loader=loader,
                )
            self.assertEqual(code, 3)
            self.assertEqual(called, [])
            self.assertFalse(source.exists())
            self.assertFalse(output.exists())
            self.assertIn(runner.REAL_DATA_BLOCKER_CODE, stderr.getvalue())

    def test_unapproved_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            with self.assertRaises(runner.ContractError):
                runner.load_and_verify_contract(path)
            self.assertFalse(path.exists())

    def test_blocker_report_names_every_unrepresentable_real_contract(self):
        report = runner.real_data_blocker_report(_bundle())
        requirements = "\n".join(report["requirements"])
        for phrase in (
            "status-evidence ledger",
            "per-race official post-time",
            "training 88-feature",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, requirements)


if __name__ == "__main__":
    unittest.main()
