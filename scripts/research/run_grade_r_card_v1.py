from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import build_grade_r_candidate_freeze_packets_v1 as candidate
from scripts.research.import_target_multicard_entry_v1 import import_multicard


FORBIDDEN_COLUMNS = [
    "current_odds",
    "current_popularity",
    "final_odds",
    "market_rank",
    "official_result",
    "payout",
    "popularity",
    "result",
    "roi",
    "t3_odds",
]

FULL_SHA256 = re.compile(r"[0-9a-f]{64}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _history_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("input_contract", {}).get("history_mode", "html"))
    if mode not in {"html", "target_direct"}:
        raise ValueError(f"unsupported history mode: {mode}")
    return mode


def _target_date_token(config: dict[str, Any]) -> str:
    raw = str(config.get("race_date", "")).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    raise ValueError(f"invalid race_date: {raw!r}")


def _source_observed_at(config: dict[str, Any]) -> datetime:
    mode = _history_mode(config)
    source_names = ["dr", "du", "html" if mode == "html" else "direct_history_manifest"]
    observed: list[datetime] = []
    for name in source_names:
        source = config.get("input_sources", {}).get(name)
        if not isinstance(source, dict) or not str(source.get("observed_at", "")).strip():
            raise ValueError(f"{mode} history mode requires {name}.observed_at")
        observed.append(candidate.parse_time(str(source["observed_at"]), config["timezone"]))
    return max(observed)


def _resolve_config_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validated_direct_history_override(
    config: dict[str, Any],
    path: Path | None,
    sha256: str | None,
) -> tuple[Path | None, str | None]:
    mode = _history_mode(config)
    supplied_together = path is not None and sha256 is not None
    supplied_neither = path is None and sha256 is None
    if not supplied_together and not supplied_neither:
        raise ValueError("direct history manifest override requires both path and sha256")
    if mode == "html":
        if not supplied_neither:
            raise ValueError(
                "HTML history mode does not accept a direct history manifest override"
            )
        return None, None
    if not supplied_together:
        raise ValueError(
            "target-direct history mode requires an explicit manifest path and sha256"
        )
    assert path is not None and sha256 is not None
    normalized_sha = str(sha256).strip().lower()
    if not FULL_SHA256.fullmatch(normalized_sha):
        raise ValueError("direct history manifest sha256 must be a full SHA-256")
    source = config.get("input_sources", {}).get("direct_history_manifest")
    if not isinstance(source, dict):
        raise ValueError("target-direct history mode requires a configured source manifest")
    configured_path = _resolve_config_path(str(source.get("path", "")))
    supplied_path = path.resolve()
    if supplied_path != configured_path:
        raise ValueError(
            "direct history manifest path differs from frozen config: "
            f"{supplied_path} != {configured_path}"
        )
    configured_sha = str(source.get("sha256", "")).strip().lower()
    if normalized_sha != configured_sha:
        raise ValueError(
            f"direct history manifest sha256 differs from frozen config: "
            f"{normalized_sha} != {configured_sha}"
        )
    return supplied_path, normalized_sha


def _validate_history_bridge_manifest(config: dict[str, Any]) -> dict[str, Any]:
    required = bool(
        config.get("input_contract", {}).get("require_history_bridge_manifest", False)
    )
    source = config.get("input_sources", {}).get("history_bridge_manifest")
    if not required:
        if source is not None:
            raise ValueError(
                "history bridge manifest is configured but not required by input contract"
            )
        return {"required": False, "contract_ok": True, "artifacts": []}
    if not isinstance(source, dict):
        raise ValueError("required history bridge manifest is not configured")

    manifest_path = _resolve_config_path(str(source.get("path", "")))
    expected_manifest_sha = str(source.get("sha256", "")).strip().lower()
    if not FULL_SHA256.fullmatch(expected_manifest_sha):
        raise ValueError("history bridge manifest sha256 must be a full SHA-256")
    if not manifest_path.is_file():
        raise ValueError(f"history bridge manifest is missing: {manifest_path}")
    actual_manifest_sha = candidate.file_sha256(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            "history bridge manifest hash mismatch: "
            f"{actual_manifest_sha} != {expected_manifest_sha}"
        )

    manifest = load_json(manifest_path)
    if str(manifest.get("experiment_id", "")) != str(config["experiment_id"]):
        raise ValueError("history bridge manifest experiment mismatch")
    if (
        bool(manifest.get("formal_buy"))
        or bool(manifest.get("send_order"))
        or int(manifest.get("stake", -1)) != 0
    ):
        raise ValueError("history bridge manifest violates BUY/order safety")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise ValueError("history bridge manifest must contain exactly two artifacts")
    roles = [str(artifact.get("role", "")) for artifact in artifacts]
    if sorted(roles) != ["entry_snapshot", "recent_results"]:
        raise ValueError("history bridge manifest roles are incomplete or duplicated")

    checked: list[dict[str, Any]] = []
    history = config.get("history", {})
    for artifact in artifacts:
        role = str(artifact["role"])
        artifact_path = _resolve_config_path(str(artifact.get("path", "")))
        history_key = str(artifact.get("config_history_key", ""))
        configured_values = history.get(history_key)
        if not isinstance(configured_values, list):
            raise ValueError(f"history bridge {role} has invalid config binding")
        configured_paths = {
            _resolve_config_path(str(value)) for value in configured_values
        }
        if artifact_path not in configured_paths:
            raise ValueError(f"history bridge {role} is not bound to {history_key}")
        if not artifact_path.is_file():
            raise ValueError(f"history bridge artifact is missing: {artifact_path}")

        expected_sha = str(artifact.get("sha256", "")).strip().lower()
        if not FULL_SHA256.fullmatch(expected_sha):
            raise ValueError(f"history bridge {role} sha256 is invalid")
        actual_sha = candidate.file_sha256(artifact_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"history bridge {role} hash mismatch: {actual_sha} != {expected_sha}"
            )
        expected_bytes = int(artifact.get("byte_count", -1))
        actual_bytes = artifact_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"history bridge {role} byte count mismatch: "
                f"{actual_bytes} != {expected_bytes}"
            )

        frame = pd.read_csv(
            artifact_path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        expected_rows = int(artifact.get("row_count", -1))
        if len(frame) != expected_rows:
            raise ValueError(
                f"history bridge {role} row count mismatch: {len(frame)} != {expected_rows}"
            )
        race_id_column = str(artifact.get("race_id_column", ""))
        horse_key_column = str(artifact.get("horse_key_column", ""))
        required_columns = [race_id_column, horse_key_column]
        if not set(required_columns).issubset(frame.columns):
            raise ValueError(f"history bridge {role} required columns are missing")
        blank_keys = frame[required_columns].apply(
            lambda series: series.str.strip().eq("")
        )
        if blank_keys.any().any():
            raise ValueError(f"history bridge {role} contains blank race/horse keys")

        actual_races = frame[race_id_column].nunique(dropna=False)
        expected_races = int(artifact.get("race_count", -1))
        if actual_races != expected_races:
            raise ValueError(
                f"history bridge {role} race count mismatch: "
                f"{actual_races} != {expected_races}"
            )
        duplicate_rows = int(frame.duplicated([race_id_column, horse_key_column]).sum())
        expected_duplicates = int(artifact.get("duplicate_race_horse_rows", -1))
        if duplicate_rows != expected_duplicates:
            raise ValueError(
                f"history bridge {role} duplicate count mismatch: "
                f"{duplicate_rows} != {expected_duplicates}"
            )

        date_column = artifact.get("date_column")
        if date_column:
            if str(date_column) not in frame.columns:
                raise ValueError(f"history bridge {role} date column is missing")
            dates = frame[str(date_column)].astype(str).str.strip()
        else:
            prefix_length = int(artifact.get("date_from_race_id_prefix", 0))
            if prefix_length <= 0:
                raise ValueError(f"history bridge {role} has no date derivation contract")
            dates = frame[race_id_column].astype(str).str.slice(0, prefix_length)
        if dates.empty or not dates.str.fullmatch(r"\d{8}").all():
            raise ValueError(f"history bridge {role} contains invalid dates")
        actual_min_date = str(dates.min())
        actual_max_date = str(dates.max())
        expected_min_date = str(artifact.get("minimum_date", ""))
        expected_max_date = str(artifact.get("maximum_date", ""))
        if (actual_min_date, actual_max_date) != (
            expected_min_date,
            expected_max_date,
        ):
            raise ValueError(
                f"history bridge {role} date range mismatch: "
                f"{actual_min_date}..{actual_max_date} != "
                f"{expected_min_date}..{expected_max_date}"
            )
        checked.append(
            {
                "role": role,
                "path": str(artifact_path),
                "sha256": actual_sha,
                "byte_count": actual_bytes,
                "row_count": len(frame),
                "race_count": actual_races,
                "duplicate_race_horse_rows": duplicate_rows,
                "minimum_date": actual_min_date,
                "maximum_date": actual_max_date,
                "contract_ok": True,
            }
        )

    return {
        "required": True,
        "contract_ok": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "artifacts": checked,
    }


def _candidate_config(card_config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    top = manifest["target_card"]
    records = manifest.get("records", [])
    race_numbers = sorted(int(record["race_no"]) for record in records)
    source_priority = (
        ["fixed_target_direct_dr_du_and_authority_manifest"]
        if _history_mode(card_config) == "target_direct"
        else ["fixed_target_multicard_html_and_du"]
    )
    return {
        "schema_version": 1,
        "experiment_id": card_config["experiment_id"],
        "cohort_id": manifest["cohort_id"],
        "timezone": card_config["timezone"],
        "target_card": {
            "race_date": top["race_date"],
            "venue_code": top["venue_code"],
            "meeting_no": top["meeting_no"],
            "day_no": top["day_no"],
            "expected_race_numbers": race_numbers,
            "expected_race_count": len(race_numbers),
        },
        "bundle_contract": {
            "model_kind": "linear_top3_set_softmax",
            "candidate_policy": "non_odds_top1_wide_pair_from_coherent_top3_softmax",
            "production_bundle_sha256": card_config["bundle"]["sha256"],
            "production_policy_sha256": card_config["bundle"]["candidate_policy_sha256"],
        },
        "probability_contract": {"set_mass": 1.0, "wide_mass": 3.0, "tolerance": 1e-10},
        "candidate_policy": card_config["candidate_policy"],
        "forbidden_candidate_columns": FORBIDDEN_COLUMNS,
        "runner_snapshot_contract": {
            "source_priority": source_priority,
            "baseline_feature_config": card_config["history"]["baseline_config"],
            "baseline_model": card_config["history"]["baseline_model"],
            "current_market_columns_must_be_blank_before_inference": True,
            "final_runner_snapshot_is_allowlist_only": True,
            "historical_rows_must_precede_target_date": True,
            "source_received_at_must_not_exceed_candidate_cutoff": True,
            "all_target_races_must_remain_in_manifest": True,
            "recent_result_freshness": {
                "required": True,
                "minimum_matched_file_count": 1,
                "minimum_joined_rows": 1,
                "minimum_history_date": card_config["history"]["minimum_history_date"],
                "maximum_history_date": card_config["history"]["maximum_history_date"],
            },
        },
        "race_domain_contract": {
            "enabled": True,
            "source_columns": ["race_domain", "芝・ダ", "surface"],
            "allowed_domains": ["flat_turf", "flat_dirt"],
            "unsupported_reason": "UNSUPPORTED_RACE_TYPE",
        },
        "safety": {
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
            "production_dashboard_write": False,
            "notification": False,
            "credential_access": False,
            "order_module_import": False,
            "real_data_during_preparation": False,
            "roi_calculation": False,
        },
    }


def _failed_record(
    *,
    config: dict[str, Any],
    target: dict[str, Any],
    source: dict[str, Any] | None,
    bundle_sha256: str,
    feature_schema_hash: str,
    input_snapshot_hash: str,
    started_at: datetime,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    record = candidate._base_record(
        config=config,
        target=target,
        source=source,
        bundle_sha256=bundle_sha256,
        feature_schema_hash=feature_schema_hash,
        start_time=started_at,
    )
    record.update(
        {
            "record_status": "FAILED",
            "candidate_freeze_contract_ok": False,
            "failure_reason_codes": [reason],
            "failure_detail": detail,
            "candidate_horse_id_1": "",
            "candidate_horse_id_2": "",
            "candidate_pair_key": "",
            "p_wide_coherent_raw": None,
            "p_action_calibrated": None,
            "top1_top2_margin": None,
            "confidence_gate_pass": False,
            "set_probability_mass_error": None,
            "wide_probability_mass_error": None,
            "probability_contract_ok": False,
            "input_snapshot_hash": input_snapshot_hash,
        }
    )
    record["candidate_freeze_record_hash"] = candidate._candidate_record_hash(record)
    return record


def freeze_card_per_race(
    *,
    config: dict[str, Any],
    target_manifest_path: Path,
    feature_source_manifest_path: Path,
    runner_snapshot_path: Path,
    inference_bundle_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    target_manifest = candidate.load_json_object(target_manifest_path)
    targets = candidate.validate_target_manifest(target_manifest, config)
    source_records = candidate.load_feature_source_records(feature_source_manifest_path)
    bundle = candidate.load_json_object(inference_bundle_path)
    candidate.validate_bundle(bundle)
    bundle_sha256 = candidate.file_sha256(inference_bundle_path)
    if bundle_sha256 != config["bundle_contract"]["production_bundle_sha256"]:
        raise ValueError("frozen inference bundle hash mismatch")
    input_snapshot_hash = candidate.file_sha256(runner_snapshot_path)
    feature_rows = candidate.build_top3_features_from_runner_rows(
        candidate.load_runner_feature_rows(runner_snapshot_path), bundle
    )
    feature_schema_hash = candidate.canonical_digest(bundle["feature_cols"])
    ledger_path = output_dir / "candidate_freeze_ledger.jsonl"
    if ledger_path.exists():
        raise ValueError(f"refusing to overwrite candidate ledger: {ledger_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    zone = ZoneInfo(config["timezone"])
    for target in targets:
        race_id = str(target["race_id"])
        source = source_records.get(race_id)
        started_at = datetime.now(zone)
        cutoff = candidate.parse_time(target["candidate_feature_cutoff_time"], config["timezone"])
        if started_at >= cutoff:
            record = _failed_record(
                config=config,
                target=target,
                source=source,
                bundle_sha256=bundle_sha256,
                feature_schema_hash=feature_schema_hash,
                input_snapshot_hash=input_snapshot_hash,
                started_at=started_at,
                reason="SOURCE_READINESS_DEADLINE_MISSED",
                detail="candidate persistence started after the original registered cutoff",
            )
        else:
            record = candidate.build_candidate_record(
                config=config,
                target=target,
                source=source,
                rows=feature_rows.get(race_id, []),
                bundle=bundle,
                bundle_sha256=bundle_sha256,
                feature_schema_hash=feature_schema_hash,
                input_snapshot_hash=input_snapshot_hash,
                start_time=started_at,
            )
        ack_at = datetime.now(zone)
        if record["record_status"] == "CANDIDATE_READY" and ack_at >= cutoff:
            record = _failed_record(
                config=config,
                target=target,
                source=source,
                bundle_sha256=bundle_sha256,
                feature_schema_hash=feature_schema_hash,
                input_snapshot_hash=input_snapshot_hash,
                started_at=started_at,
                reason="SOURCE_READINESS_DEADLINE_MISSED",
                detail="candidate persistence acknowledgement missed the original registered cutoff",
            )
        packet_relative = Path("packets") / f"{race_id}.candidate_freeze.json"
        packet_path = output_dir / packet_relative
        packet_sha = candidate._write_or_verify_packet(packet_path, record)
        ledger_event = {
            "schema_version": 1,
            "event_type": "candidate_freeze_persist_ack",
            "experiment_id": config["experiment_id"],
            "cohort_id": config["cohort_id"],
            "race_id": race_id,
            "race_no": int(target["race_no"]),
            "record_status": record["record_status"],
            "candidate_freeze_contract_ok": record["candidate_freeze_contract_ok"],
            "failure_reason_codes": record["failure_reason_codes"],
            "candidate_pair_key": record["candidate_pair_key"],
            "race_domain": record["race_domain"],
            "race_domain_source_hash": record["race_domain_source_hash"],
            "batch_readiness_contract_ok": True,
            "candidate_batch_committed_at": record["candidate_batch_committed_at"],
            "candidate_batch_deadline_at": None,
            "candidate_freeze_record_hash": record["candidate_freeze_record_hash"],
            "candidate_freeze_persist_ack_at": ack_at.isoformat(timespec="milliseconds"),
            "packet_path": packet_relative.as_posix(),
            "packet_file_sha256": packet_sha,
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
            "idempotency_key": candidate._idempotency_key(config, race_id),
        }
        candidate.append_jsonl(ledger_path, ledger_event)
        candidate._verify_ledger_record(ledger_event, output_dir)
    ledger = candidate.read_jsonl(ledger_path)
    summary = candidate._build_summary(config=config, targets=targets, ledger=ledger)
    candidate.write_json_atomic(output_dir / "candidate_freeze_summary.json", summary)
    return summary


def _candidate_table(
    output_root: Path,
    cards: list[dict[str, Any]],
    target_date: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for card in cards:
        slug = str(card["slug"])
        raw = pd.read_csv(
            output_root / slug / "raw_entry" / f"entry_snapshot_{target_date}.csv",
            encoding="utf-8-sig",
            dtype=str,
        )
        by_id = raw.drop_duplicates("horse_id").set_index("horse_id")["horse_name"].to_dict()
        ledger = candidate.read_jsonl(
            output_root / slug / "candidate_freeze" / "candidate_freeze_ledger.jsonl"
        )
        for event in ledger:
            packet = load_json(output_root / slug / "candidate_freeze" / event["packet_path"])
            horse_1 = str(packet.get("candidate_horse_id_1", ""))
            horse_2 = str(packet.get("candidate_horse_id_2", ""))
            confidence = bool(packet.get("confidence_gate_pass", False))
            status = str(packet.get("record_status", "FAILED"))
            rows.append(
                {
                    "venue": slug,
                    "race_id": packet["race_id"],
                    "race_no": packet["race_no"],
                    "record_status": status,
                    "candidate_pair_key": packet.get("candidate_pair_key", ""),
                    "horse_name_1": by_id.get(horse_1, ""),
                    "horse_name_2": by_id.get(horse_2, ""),
                    "p_wide_coherent_raw": packet.get("p_wide_coherent_raw"),
                    "p_action_calibrated": packet.get("p_action_calibrated"),
                    "confidence_gate_pass": confidence,
                    "shadow_action": (
                        "PENDING_STRICT_T3"
                        if status == "CANDIDATE_READY" and confidence
                        else "NO_BET_CONFIDENCE"
                        if status == "CANDIDATE_READY"
                        else "NO_BET_CONTRACT"
                    ),
                    "failure_reason_codes": "|".join(packet.get("failure_reason_codes", [])),
                    "candidate_uses_odds": False,
                    "formal_buy": False,
                    "send_order": False,
                    "stake": 0,
                }
            )
    return pd.DataFrame(rows).sort_values(["venue", "race_no"], kind="mergesort")


def _validate_candidate_table(
    table: pd.DataFrame,
    expected_race_ids: set[str],
) -> dict[str, int]:
    required = {
        "race_id",
        "record_status",
        "candidate_uses_odds",
        "formal_buy",
        "send_order",
        "stake",
    }
    missing_columns = sorted(required - set(table.columns))
    if missing_columns:
        raise ValueError(f"candidate table missing required columns: {missing_columns}")
    race_ids = table["race_id"].astype(str)
    duplicated = int(race_ids.duplicated(keep=False).sum())
    observed = set(race_ids)
    missing = expected_race_ids - observed
    unexpected = observed - expected_race_ids
    if duplicated or missing or unexpected or len(table) != len(expected_race_ids):
        raise ValueError(
            "candidate terminal denominator mismatch: "
            f"rows={len(table)}, expected={len(expected_race_ids)}, duplicated={duplicated}, "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if table["record_status"].astype(str).str.strip().eq("").any():
        raise ValueError("candidate table contains a non-terminal blank record status")
    if table["candidate_uses_odds"].astype(bool).any():
        raise ValueError("candidate table contains candidate_uses_odds=true")
    if table["formal_buy"].astype(bool).any() or table["send_order"].astype(bool).any():
        raise ValueError("candidate table violates formal BUY/order safety")
    if pd.to_numeric(table["stake"], errors="coerce").fillna(-1).ne(0).any():
        raise ValueError("candidate table contains nonzero or invalid stake")
    return {
        "registered_target_rows": len(expected_race_ids),
        "terminal_record_rows": len(table),
        "missing_target_rows": 0,
        "duplicate_target_rows": 0,
    }


def run(
    config_path: Path,
    output_root: Path,
    *,
    direct_history_manifest_path: Path | None = None,
    direct_history_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    config = load_json(config_path)
    target_date = _target_date_token(config)
    direct_history_manifest_path, direct_history_manifest_sha256 = (
        _validated_direct_history_override(
            config,
            direct_history_manifest_path,
            direct_history_manifest_sha256,
        )
    )
    candidate.assert_real_data_authorized(ROOT, config["experiment_id"])
    history_bridge_preflight = _validate_history_bridge_manifest(config)
    import_summary = import_multicard(
        config_path,
        output_root,
        direct_history_manifest_path=direct_history_manifest_path,
        direct_history_manifest_sha256=direct_history_manifest_sha256,
    )
    source_observed_at = _source_observed_at(config)
    card_summaries: list[dict[str, Any]] = []
    expected_race_ids: set[str] = set()
    for card in config["cards"]:
        slug = str(card["slug"])
        target_manifest_path = ROOT / card["target_manifest"]
        manifest = load_json(target_manifest_path)
        manifest_date = _target_date_token({"race_date": manifest["target_card"]["race_date"]})
        if manifest_date != target_date:
            raise ValueError(
                f"target manifest date mismatch for {slug}: {manifest_date} != {target_date}"
            )
        if str(manifest.get("experiment_id", "")) != str(config["experiment_id"]):
            raise ValueError(f"target manifest experiment mismatch for {slug}")
        if str(manifest.get("cohort_id", "")) != str(card["cohort_id"]):
            raise ValueError(f"target manifest cohort mismatch for {slug}")
        card_race_ids = {str(record["race_id"]) for record in manifest.get("records", [])}
        if expected_race_ids & card_race_ids:
            raise ValueError(f"duplicate race_id across target manifests for {slug}")
        expected_race_ids.update(card_race_ids)
        adapter_config = _candidate_config(config, manifest)
        adapter_config_path = output_root / slug / "candidate_adapter_config.json"
        write_json_atomic(adapter_config_path, adapter_config)
        adapter_config = candidate.load_adapter_config(adapter_config_path)
        card_root = output_root / slug
        raw_entry = card_root / "raw_entry" / f"entry_snapshot_{target_date}.csv"
        runner_dir = card_root / "runner_snapshot"
        runner_snapshot = runner_dir / f"runner_snapshot_{target_date}.csv"
        source_manifest = runner_dir / f"feature_source_manifest_{target_date}.json"
        prep_summary = candidate.prepare_runner_snapshot(
            target_manifest_path=target_manifest_path,
            raw_entry_path=raw_entry,
            inference_bundle_path=Path(config["bundle"]["inference_bundle"]),
            runner_output_path=runner_snapshot,
            source_manifest_path=source_manifest,
            work_dir=runner_dir / "work",
            config=adapter_config,
            baseline_config_path=ROOT / config["history"]["baseline_config"],
            baseline_model_path=Path(config["history"]["baseline_model"]),
            historical_csv_path=Path(config["history"]["historical_csv"]),
            ability_history_dir=Path(config["history"]["ability_history_dir"]),
            recent_result_globs=list(config["history"]["recent_result_globs"]),
            entry_globs=list(config["history"]["entry_globs"]),
            source_observed_at=source_observed_at,
        )
        freeze_summary = freeze_card_per_race(
            config=adapter_config,
            target_manifest_path=target_manifest_path,
            feature_source_manifest_path=source_manifest,
            runner_snapshot_path=runner_snapshot,
            inference_bundle_path=Path(config["bundle"]["inference_bundle"]),
            output_dir=card_root / "candidate_freeze",
        )
        card_summaries.append(
            {"slug": slug, "runner_preparation": prep_summary, "candidate_freeze": freeze_summary}
        )
    expected_races = int(config.get("input_contract", {}).get("expected_races", 0))
    if len(expected_race_ids) != expected_races:
        raise ValueError(
            f"registered target count mismatch: {len(expected_race_ids)} != {expected_races}"
        )
    table = _candidate_table(output_root, config["cards"], target_date)
    denominator = _validate_candidate_table(table, expected_race_ids)
    table_path = output_root / "candidate_shadow_actions.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "target_date": target_date,
        "history_mode": _history_mode(config),
        "history_bridge_preflight": history_bridge_preflight,
        "import": import_summary,
        "cards": card_summaries,
        "candidate_table": str(table_path),
        "candidate_table_sha256": candidate.file_sha256(table_path),
        "candidate_ready": int(table["record_status"].eq("CANDIDATE_READY").sum()),
        "pending_strict_t3": int(table["shadow_action"].eq("PENDING_STRICT_T3").sum()),
        **denominator,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    write_json_atomic(output_root / "grade_r_card_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one approved Grade-R three-card candidate freeze."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--direct-history-manifest", type=Path)
    parser.add_argument("--direct-history-manifest-sha256")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(
        args.config.resolve(),
        args.output_root.resolve(),
        direct_history_manifest_path=(
            args.direct_history_manifest.resolve()
            if args.direct_history_manifest is not None
            else None
        ),
        direct_history_manifest_sha256=args.direct_history_manifest_sha256,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
