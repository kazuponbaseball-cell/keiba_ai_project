from __future__ import annotations

import argparse
import json
import os
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


def _candidate_config(card_config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    top = manifest["target_card"]
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
            "expected_race_numbers": list(range(1, 13)),
            "expected_race_count": 12,
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
            "source_priority": ["fixed_target_multicard_html_and_du"],
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


def _candidate_table(output_root: Path, cards: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for card in cards:
        slug = str(card["slug"])
        raw = pd.read_csv(
            output_root / slug / "raw_entry" / "entry_snapshot_20260808.csv",
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
                    "formal_buy": False,
                    "send_order": False,
                    "stake": 0,
                }
            )
    return pd.DataFrame(rows).sort_values(["venue", "race_no"], kind="mergesort")


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_json(config_path)
    candidate.assert_real_data_authorized(ROOT, config["experiment_id"])
    import_summary = import_multicard(config_path, output_root)
    source_observed_at = candidate.parse_time(
        config["input_sources"]["html"]["observed_at"], config["timezone"]
    )
    card_summaries: list[dict[str, Any]] = []
    for card in config["cards"]:
        slug = str(card["slug"])
        target_manifest_path = ROOT / card["target_manifest"]
        manifest = load_json(target_manifest_path)
        adapter_config = _candidate_config(config, manifest)
        adapter_config_path = output_root / slug / "candidate_adapter_config.json"
        write_json_atomic(adapter_config_path, adapter_config)
        adapter_config = candidate.load_adapter_config(adapter_config_path)
        card_root = output_root / slug
        raw_entry = card_root / "raw_entry" / "entry_snapshot_20260808.csv"
        runner_dir = card_root / "runner_snapshot"
        runner_snapshot = runner_dir / "runner_snapshot_20260808.csv"
        source_manifest = runner_dir / "feature_source_manifest_20260808.json"
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
    table = _candidate_table(output_root, config["cards"])
    table_path = output_root / "candidate_shadow_actions.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "import": import_summary,
        "cards": card_summaries,
        "candidate_table": str(table_path),
        "candidate_table_sha256": candidate.file_sha256(table_path),
        "candidate_ready": int(table["record_status"].eq("CANDIDATE_READY").sum()),
        "pending_strict_t3": int(table["shadow_action"].eq("PENDING_STRICT_T3").sum()),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    write_json_atomic(output_root / "grade_r_card_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one approved Grade-R three-card candidate freeze.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args.config.resolve(), args.output_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
