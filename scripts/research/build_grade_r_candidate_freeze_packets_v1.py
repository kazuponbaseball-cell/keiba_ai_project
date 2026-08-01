from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
import types
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HASH_FIELDS = {
    "candidate_freeze_record_hash",
    "packet_file_sha256",
}
TARGET_FORBIDDEN_FIELDS = {
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
}
CURRENT_MARKET_ENTRY_COLUMNS = {
    "current_odds",
    "current_popularity",
    "market_rank",
    "odds",
    "popularity",
    "単勝オッズ",
    "人気",
}
RUNNER_SNAPSHOT_REQUIRED_COLUMNS = {
    "race_id",
    "horse_no",
    "horse_id",
    "ai_score",
    "ai_rank",
}
RUNNER_SNAPSHOT_ALLOWED_COLUMNS = [
    "race_id",
    "horse_no",
    "horse_id",
    "horse_name",
    "ai_score",
    "ai_rank",
    "expected_pace",
    "front_running_tendency",
    "closing_tendency",
    "horse_front_run_rate_past5",
    "horse_closer_rate_past5",
    "race_front_runner_count",
    "race_need_lead_count",
    "race_early_pressure_score",
    "race_pace_collapse_risk",
    "race_slow_pace_risk",
    "ability_floor_score_5",
    "ability_stability_score_3",
    "recent_weighted_score_3",
    "condition_adjusted_recent_ability_score",
    "career_shallow_flag",
    "career_growth_zone_flag",
    "basic_ability_history_ready",
]


class CandidateContractError(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_time(value: Any, timezone_name: str) -> datetime:
    if value is None or not str(value).strip():
        raise CandidateContractError("FEATURE_SOURCE_TIME_VIOLATION", "timestamp missing")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    zone = ZoneInfo(timezone_name)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"ledger line {line_number} is not an object")
        records.append(value)
    return records


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _horse_sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value).strip()
    try:
        return (0, int(text), text)
    except ValueError:
        return (1, 0, text)


def canonical_pair(horse_a: Any, horse_b: Any) -> tuple[str, str]:
    values = sorted((str(horse_a).strip(), str(horse_b).strip()), key=_horse_sort_key)
    if not values[0] or values[0] == values[1]:
        raise CandidateContractError("PROBABILITY_CONTRACT_VIOLATION", "invalid pair")
    return values[0], values[1]


def canonical_triplet(values: Iterable[Any]) -> tuple[str, str, str]:
    horses = sorted((str(value).strip() for value in values), key=_horse_sort_key)
    if len(horses) != 3 or any(not horse for horse in horses) or len(set(horses)) != 3:
        raise CandidateContractError(
            "PROBABILITY_CONTRACT_VIOLATION", "triplet must contain three horses"
        )
    return horses[0], horses[1], horses[2]


def load_adapter_config(path: Path) -> dict[str, Any]:
    config = load_json_object(path)
    safety = config.get("safety", {})
    for field in (
        "formal_buy",
        "send_order",
        "production_dashboard_write",
        "notification",
        "credential_access",
        "order_module_import",
        "real_data_during_preparation",
        "roi_calculation",
    ):
        if safety.get(field) is not False:
            raise ValueError(f"{field} must remain false")
    if safety.get("stake") != 0:
        raise ValueError("stake must remain zero")
    policy = config.get("candidate_policy", {})
    if policy.get("name") != "WIDE_1_NON_ODDS_COHERENT_TOP3":
        raise ValueError("candidate policy mismatch")
    if policy.get("candidate_substitution_allowed") is not False:
        raise ValueError("candidate substitution must remain false")
    if policy.get("alternative_pair_search_allowed") is not False:
        raise ValueError("alternative pair search must remain false")
    card = config.get("target_card", {})
    expected = [int(value) for value in card.get("expected_race_numbers", [])]
    if expected != list(range(1, 13)):
        raise ValueError("target card must contain race numbers 1 through 12")
    if int(card.get("expected_race_count", 0)) != len(expected):
        raise ValueError("target card count mismatch")
    return config


def validate_target_manifest(
    manifest: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    if manifest.get("experiment_id") != config.get("experiment_id"):
        raise ValueError("target manifest experiment_id mismatch")
    if manifest.get("cohort_id") != config.get("cohort_id"):
        raise ValueError("target manifest cohort_id mismatch")
    if manifest.get("data_class") not in {"synthetic", "real-data"}:
        raise ValueError("target manifest data_class must be synthetic or real-data")
    expected_card = config["target_card"]
    observed_card = manifest.get("target_card", {})
    for field in ("race_date", "venue_code", "meeting_no", "day_no"):
        if str(observed_card.get(field)) != str(expected_card.get(field)):
            raise ValueError(f"target card {field} mismatch")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("target manifest records must be a list")
    expected_numbers = [int(value) for value in expected_card["expected_race_numbers"]]
    race_numbers = [int(record.get("race_no", 0)) for record in records]
    if sorted(race_numbers) != expected_numbers or len(set(race_numbers)) != len(records):
        raise ValueError("target manifest must contain exactly the registered 12 races")
    race_ids = [str(record.get("race_id", "")).strip() for record in records]
    if any(not race_id for race_id in race_ids) or len(set(race_ids)) != len(race_ids):
        raise ValueError("target race IDs must be present and unique")
    for record in records:
        forbidden = sorted(TARGET_FORBIDDEN_FIELDS.intersection(record))
        if forbidden:
            raise ValueError("target record contains forbidden fields: " + ", ".join(forbidden))
        if record.get("target_registered") is not True:
            raise ValueError("every target race must be pre-registered")
        for field in ("scheduled_post_time", "candidate_feature_cutoff_time"):
            if not str(record.get(field, "")).strip():
                raise ValueError(f"target record missing {field}")
    return sorted(records, key=lambda record: int(record["race_no"]))


def assert_real_data_authorized(root: Path, experiment_id: str) -> None:
    registry_path = root / "research" / "REGISTRY.jsonl"
    events = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("experiment_id") == experiment_id
    ]
    if not events:
        raise ValueError("real-data execution has no registry event")
    latest = events[-1]
    if latest.get("status") != "running":
        raise ValueError("real-data execution requires RUNNING status")
    if latest.get("real_data_execution_allowed") is not True:
        raise ValueError("real-data execution is not authorized")
    if not str(latest.get("run_scope_digest", "")).strip():
        raise ValueError("real-data execution requires a bound run scope")
    if latest.get("formal_buy") is not False or latest.get("send_order") is not False:
        raise ValueError("registry safety flags are not fail-closed")
    if latest.get("stake") != 0:
        raise ValueError("registry stake must remain zero")


def _read_csv_frame(path: Path):
    import pandas as pd

    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, dtype=str)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, dtype=str)


def _atomic_write_csv(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(str(value) for value in columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def _canonical_integer_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _install_data_loader_shim() -> None:
    if "src.data.loaders" in sys.modules:
        return
    import pandas as pd

    from src.utils.paths import project_path

    package = types.ModuleType("src.data")
    package.__path__ = []  # type: ignore[attr-defined]
    loaders = types.ModuleType("src.data.loaders")

    def load_json_config(path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = project_path(str(config_path))
        return json.loads(config_path.read_text(encoding="utf-8"))

    def required_columns(config: dict[str, Any], *, for_prediction: bool = False) -> list[str]:
        data_cfg = config["data"]
        train_cfg = config.get("training", {})
        columns = {
            data_cfg["race_id_column"],
            data_cfg["horse_id_column"],
            data_cfg["horse_name_column"],
            data_cfg["date_column"],
            data_cfg["rank_column"],
            data_cfg["abnormal_column"],
            *config["numeric_features"],
            *config["categorical_features"],
            *config.get("feature_source_columns", []),
            *config.get("optional_feature_source_columns", []),
        }
        for optional in (
            train_cfg.get("race_type_column"),
            train_cfg.get("surface_column"),
            train_cfg.get("venue_column"),
        ):
            if optional:
                columns.add(optional)
        if for_prediction:
            columns.update(config.get("passthrough_prediction_columns", []))
        return sorted(columns)

    def inference_required_columns(config: dict[str, Any]) -> list[str]:
        data_cfg = config["data"]
        train_cfg = config.get("training", {})
        allowed_prefixes = list(config.get("leakage_allowed_prefixes", []))
        source_columns = {
            column
            for column in config.get("feature_source_columns", [])
            if any(str(column).startswith(prefix) for prefix in allowed_prefixes)
        }
        columns = {
            data_cfg["race_id_column"],
            data_cfg["horse_id_column"],
            data_cfg["horse_name_column"],
            data_cfg["date_column"],
            *config["numeric_features"],
            *config["categorical_features"],
            *source_columns,
        }
        for optional in (
            train_cfg.get("race_type_column"),
            train_cfg.get("surface_column"),
            train_cfg.get("venue_column"),
        ):
            if optional:
                columns.add(optional)
        return sorted(columns)

    def inference_optional_columns(config: dict[str, Any]) -> list[str]:
        data_cfg = config["data"]
        columns = set(config.get("passthrough_prediction_columns", []))
        columns.update(config.get("optional_feature_source_columns", []))
        columns.add(data_cfg["abnormal_column"])
        columns.add(data_cfg["rank_column"])
        return sorted(columns)

    def model_numeric_features(config: dict[str, Any]) -> list[str]:
        return [*config["numeric_features"], *config.get("generated_numeric_features", [])]

    def model_categorical_features(config: dict[str, Any]) -> list[str]:
        return [
            *config["categorical_features"],
            *config.get("generated_categorical_features", []),
        ]

    def load_historical_csv(
        config: dict[str, Any], *, columns: list[str] | None = None
    ):
        csv_path = project_path(config["data"]["historical_csv"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Historical CSV not found: {csv_path}")
        return pd.read_csv(
            csv_path,
            encoding=config["data"].get("encoding", "cp932"),
            usecols=columns,
            low_memory=False,
        )

    for name, value in {
        "load_json_config": load_json_config,
        "required_columns": required_columns,
        "inference_required_columns": inference_required_columns,
        "inference_optional_columns": inference_optional_columns,
        "model_numeric_features": model_numeric_features,
        "model_categorical_features": model_categorical_features,
        "load_historical_csv": load_historical_csv,
    }.items():
        setattr(loaders, name, value)
    package.loaders = loaders  # type: ignore[attr-defined]
    sys.modules["src.data"] = package
    sys.modules["src.data.loaders"] = loaders


def _parse_race_class(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    if re.search(r"(?:G3|GIII)(?!I)", text, flags=re.IGNORECASE):
        return "Ｇ３"
    if re.search(r"(?:G2|GII)(?!I)", text, flags=re.IGNORECASE):
        return "Ｇ２"
    if re.search(r"(?:G1|GI)(?!I)", text, flags=re.IGNORECASE):
        return "Ｇ１"
    if re.search(r"(?:リステッド|\(L\))", text, flags=re.IGNORECASE):
        return "OP(L)"
    for label, output in (
        ("3勝クラス", "3勝"),
        ("2勝クラス", "2勝"),
        ("1勝クラス", "1勝"),
        ("新馬", "新馬"),
        ("未勝利", "未勝利"),
        ("オープン", "ｵｰﾌﾟﾝ"),
    ):
        if label in text:
            return output
    return ""


def _parse_going(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    match = re.search(r"馬場\s*[:：]\s*(良|稍重|重|不良|稍|不)", text)
    return {"稍重": "稍", "不良": "不"}.get(match.group(1), match.group(1)) if match else ""


def _parse_track_code(value: str, surface: str) -> str:
    if surface == "ダ":
        return "1"
    if surface != "芝":
        return ""
    text = unicodedata.normalize("NFKC", value)
    return "8" if re.search(r"[\s(]外(?:[\s)]|$)", text) else "0"


def capture_public_entry_snapshot(
    *,
    target_manifest_path: Path,
    baseline_config_path: Path,
    output_csv_path: Path,
    capture_manifest_path: Path,
    cache_dir: Path,
    config: dict[str, Any],
    refresh: bool,
    sleep_seconds: float,
) -> dict[str, Any]:
    import pandas as pd
    from lxml import html

    _install_data_loader_shim()
    from scripts import fetch_netkeiba_entries_snapshot as fetcher
    from src.data.loaders import inference_optional_columns, inference_required_columns

    target_manifest = load_json_object(target_manifest_path)
    targets = validate_target_manifest(target_manifest, config)
    baseline_config = load_json_object(baseline_config_path)
    card = config["target_card"]
    race_date = str(card["race_date"]).replace("-", "")
    venue_code = str(card["venue_code"]).zfill(2)
    meeting = str(card["meeting_no"]).zfill(2)
    day = str(card["day_no"]).zfill(2)
    venue_name = {
        "01": "札幌",
        "02": "函館",
        "03": "福島",
        "04": "新潟",
        "05": "東京",
        "06": "中山",
        "07": "中京",
        "08": "京都",
        "09": "阪神",
        "10": "小倉",
    }.get(venue_code, venue_code)
    columns = list(
        dict.fromkeys(
            [
                *inference_required_columns(baseline_config),
                *inference_optional_columns(baseline_config),
            ]
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    capture_records: list[dict[str, Any]] = []
    original_first_link_id = fetcher._first_link_id

    def flexible_link_id(node: Any, pattern: str) -> str:
        value = original_first_link_id(node, pattern)
        if value:
            return value
        role = "jockey" if "jockey" in pattern else "trainer" if "trainer" in pattern else ""
        if not role:
            return ""
        return original_first_link_id(node, rf"/{role}/(?:result/recent/)?(\d+)")

    fetcher._first_link_id = flexible_link_id
    try:
        for target in targets:
            race_id = str(target["race_id"])
            race_no = int(target["race_no"])
            url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
            cache_path = cache_dir / f"{race_id}.html"
            received_at = datetime.now(ZoneInfo(config["timezone"]))
            try:
                source = fetcher._read_url(
                    url,
                    cache_path,
                    refresh=refresh,
                    sleep_seconds=sleep_seconds,
                )
                received_at = datetime.now(ZoneInfo(config["timezone"]))
                tree = html.fromstring(source)
                meta = fetcher._race_metadata(
                    tree,
                    date=race_date,
                    venue_code=venue_code,
                    meeting=meeting,
                    day=day,
                    race_no=race_no,
                )
                meta["venue"] = venue_name
                race_data_01 = fetcher._first_text(tree, ["//*[contains(@class, 'RaceData01')]"])
                race_data_02 = fetcher._first_text(tree, ["//*[contains(@class, 'RaceData02')]"])
                parsed = fetcher._parse_runner_rows(tree, meta, source_url=url)
                for row in parsed:
                    row["クラス名"] = _parse_race_class(race_data_02)
                    row["トラックコード"] = _parse_track_code(race_data_01, str(meta.get("surface", "")))
                    row["馬場状態"] = _parse_going(race_data_01) or row.get("馬場状態")
                    row["レースID(新/馬番無)"] = race_id
                    row["単勝オッズ"] = ""
                    row["人気"] = ""
                all_rows.extend(parsed)
                status = "CAPTURED" if parsed else "UNAVAILABLE"
                detail = "" if parsed else "no runner rows"
            except Exception as exc:
                received_at = datetime.now(ZoneInfo(config["timezone"]))
                status = "UNAVAILABLE"
                detail = f"{type(exc).__name__}: {exc}"
            capture_records.append(
                {
                    "race_id": race_id,
                    "race_no": race_no,
                    "url": url,
                    "received_at": received_at.isoformat(timespec="milliseconds"),
                    "status": status,
                    "detail": detail,
                    "cache_path": str(cache_path),
                    "cache_sha256": file_sha256(cache_path) if cache_path.exists() else "",
                }
            )
    finally:
        fetcher._first_link_id = original_first_link_id
    output = fetcher._blank_snapshot(columns, all_rows)
    for column in output.columns:
        text = str(column).strip()
        if text in CURRENT_MARKET_ENTRY_COLUMNS or text.lower().startswith("odds_"):
            output[column] = ""
    _atomic_write_csv(output, output_csv_path)
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "manifest_type": "public_entry_identity_capture",
        "target_manifest_path": str(target_manifest_path),
        "target_manifest_sha256": file_sha256(target_manifest_path),
        "baseline_config_path": str(baseline_config_path),
        "baseline_config_sha256": file_sha256(baseline_config_path),
        "output_csv": str(output_csv_path),
        "output_csv_sha256": file_sha256(output_csv_path),
        "records": capture_records,
        "current_market_fields_blank": True,
        "target_selection_uses_odds": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    write_json_atomic(capture_manifest_path, manifest)
    return {
        "captured_races": sum(1 for record in capture_records if record["status"] == "CAPTURED"),
        "unavailable_races": sum(1 for record in capture_records if record["status"] != "CAPTURED"),
        "runner_rows": int(len(output)),
        "output_csv_sha256": manifest["output_csv_sha256"],
        "capture_manifest_sha256": file_sha256(capture_manifest_path),
    }


def _sanitize_entry_snapshot(
    *,
    raw_entry_path: Path,
    output_path: Path,
    targets: list[dict[str, Any]],
    baseline_config: dict[str, Any],
) -> dict[str, Any]:
    frame = _read_csv_frame(raw_entry_path)
    race_no_col = _first_existing_column(frame.columns, ("race_no", "\uff32"))
    horse_no_col = _first_existing_column(frame.columns, ("horse_no", "\u99ac\u756a"))
    horse_id_col = _first_existing_column(
        frame.columns, ("horse_id", "\u8840\u7d71\u767b\u9332\u756a\u53f7")
    )
    if race_no_col is None or horse_no_col is None or horse_id_col is None:
        raise CandidateContractError(
            "CANDIDATE_SOURCE_NOT_READY", "entry identity columns are missing"
        )
    race_map = {int(record["race_no"]): str(record["race_id"]) for record in targets}
    frame["_race_no"] = frame[race_no_col].map(_canonical_integer_text)
    frame = frame[frame["_race_no"].ne("")].copy()
    frame["_race_no"] = frame["_race_no"].astype(int)
    frame = frame[frame["_race_no"].isin(race_map)].copy()
    frame["race_id"] = frame["_race_no"].map(race_map)
    baseline_race_col = str(baseline_config["data"]["race_id_column"])
    frame[baseline_race_col] = frame["race_id"]
    frame["horse_no"] = frame[horse_no_col].map(_canonical_integer_text)
    frame["horse_id"] = frame[horse_id_col].map(_canonical_integer_text)
    frame[horse_no_col] = frame["horse_no"]
    frame[horse_id_col] = frame["horse_id"]

    market_columns: list[str] = []
    for column in frame.columns:
        text = str(column).strip()
        lowered = text.lower()
        if text in CURRENT_MARKET_ENTRY_COLUMNS or lowered.startswith("odds_"):
            frame[column] = ""
            market_columns.append(text)
    invalid_identity = frame["horse_no"].eq("") | frame["horse_id"].eq("")
    if invalid_identity.any():
        raise CandidateContractError(
            "CANDIDATE_SOURCE_NOT_READY", "runner identity is blank after sanitization"
        )
    duplicate_count = int(frame.duplicated(["race_id", "horse_no"]).sum())
    if duplicate_count:
        raise CandidateContractError(
            "STARTER_UNIVERSE_MISMATCH", "duplicate race and horse number in entry"
        )
    observed_races = set(frame["race_id"].astype(str))
    expected_races = set(race_map.values())
    missing_races = sorted(expected_races.difference(observed_races))
    frame = frame.drop(columns=["_race_no"], errors="ignore")
    frame = frame.sort_values(["race_id", "horse_no"], kind="mergesort")
    _atomic_write_csv(frame, output_path)
    return {
        "rows": int(len(frame)),
        "race_count": int(frame["race_id"].nunique()),
        "missing_race_ids": missing_races,
        "market_columns_blanked": sorted(set(market_columns)),
        "sanitized_entry_sha256": file_sha256(output_path),
    }


def _assert_baseline_market_firewall(baseline_config: dict[str, Any]) -> None:
    feature_columns: list[str] = []
    for key in (
        "numeric_features",
        "categorical_features",
        "generated_numeric_features",
        "generated_categorical_features",
    ):
        values = baseline_config.get(key, [])
        if isinstance(values, list):
            feature_columns.extend(str(value) for value in values)
    forbidden = [
        column
        for column in feature_columns
        if column in CURRENT_MARKET_ENTRY_COLUMNS or column.lower().startswith("odds_")
    ]
    if forbidden:
        raise CandidateContractError(
            "FORBIDDEN_CANDIDATE_INPUT_COLUMN",
            "baseline feature schema contains current market columns: "
            + ", ".join(sorted(set(forbidden))),
        )


def _run_baseline_prediction(
    *,
    sanitized_entry_path: Path,
    baseline_config_path: Path,
    baseline_model_path: Path,
    historical_csv_path: Path,
    work_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    from src.predict import predict_baseline

    baseline_config = load_json_object(baseline_config_path)
    _assert_baseline_market_firewall(baseline_config)
    runtime_config = json.loads(json.dumps(baseline_config, ensure_ascii=False))
    runtime_config["data"]["historical_csv"] = str(historical_csv_path.resolve())
    runtime_config_path = work_dir / "baseline_features.runtime.json"
    write_json_atomic(runtime_config_path, runtime_config)
    prediction_path = work_dir / "baseline_predictions.csv"
    if prediction_path.exists():
        return prediction_path, {
            "reused": True,
            "prediction_sha256": file_sha256(prediction_path),
        }
    raw_output_dir = work_dir / "baseline_raw"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    before = set(raw_output_dir.glob("baseline_predictions_*.csv"))
    original_argv = list(sys.argv)
    try:
        sys.argv = [
            "predict_baseline",
            "--config",
            str(runtime_config_path),
            "--model",
            str(baseline_model_path.resolve()),
            "--input-csv",
            str(sanitized_entry_path.resolve()),
            "--output-dir",
            str(raw_output_dir.resolve()),
        ]
        predict_baseline.main()
    finally:
        sys.argv = original_argv
    created = sorted(set(raw_output_dir.glob("baseline_predictions_*.csv")) - before)
    if len(created) != 1:
        raise CandidateContractError(
            "CANDIDATE_SOURCE_NOT_READY", "baseline predictor did not emit exactly one file"
        )
    os.replace(created[0], prediction_path)
    return prediction_path, {
        "reused": False,
        "prediction_sha256": file_sha256(prediction_path),
        "runtime_config_sha256": file_sha256(runtime_config_path),
    }


def _run_basic_ability_enrichment(
    *,
    prediction_path: Path,
    sanitized_entry_path: Path,
    ability_history_dir: Path,
    recent_result_globs: list[str],
    entry_globs: list[str],
    work_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    from scripts.enrich_prediction_basic_ability_features import enrich_prediction

    output_path = work_dir / "baseline_predictions_basic_ability.csv"
    if output_path.exists():
        summary_path = output_path.with_suffix(".basic_ability_summary.json")
        summary = load_json_object(summary_path) if summary_path.exists() else {}
        summary["reused"] = True
        return output_path, summary
    summary = enrich_prediction(
        prediction_path,
        sanitized_entry_path,
        output_path,
        ability_history_dir,
        recent_result_globs=recent_result_globs,
        entry_globs=entry_globs,
    )
    summary["reused"] = False
    return output_path, summary


def _hash_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted({value.resolve() for value in paths if value.is_file()}, key=str):
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return records


def _expand_glob_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(value) for value in glob.glob(pattern))
    return [path for path in paths if path.is_file()]


def finalize_runner_snapshot(
    *,
    enriched_runner_path: Path,
    runner_output_path: Path,
    source_manifest_path: Path,
    raw_entry_path: Path,
    target_manifest_path: Path,
    inference_bundle_path: Path,
    config: dict[str, Any],
    source_observed_at: datetime,
    lineage_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    import pandas as pd

    target_manifest = load_json_object(target_manifest_path)
    targets = validate_target_manifest(target_manifest, config)
    bundle = load_json_object(inference_bundle_path)
    validate_bundle(bundle)
    frame = _read_csv_frame(enriched_runner_path)
    aliases = {
        "horse_name": ("horse_name", "\u99ac\u540d"),
        "horse_no": ("horse_no", "\u99ac\u756a"),
        "horse_id": ("horse_id", "\u8840\u7d71\u767b\u9332\u756a\u53f7"),
    }
    for destination, candidates in aliases.items():
        source = _first_existing_column(frame.columns, candidates)
        if destination not in frame.columns and source is not None:
            frame[destination] = frame[source]
    if "race_id" not in frame.columns:
        race_source = _first_existing_column(
            frame.columns, ("\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)",)
        )
        if race_source is not None:
            frame["race_id"] = frame[race_source]
    missing = sorted(RUNNER_SNAPSHOT_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise CandidateContractError(
            "CANDIDATE_SOURCE_NOT_READY", "runner snapshot columns missing: " + ", ".join(missing)
        )
    for column in ("race_id", "horse_no", "horse_id"):
        frame[column] = frame[column].map(_canonical_integer_text)
    frame = frame[frame["race_id"].isin({str(record["race_id"]) for record in targets})].copy()
    if frame.duplicated(["race_id", "horse_no"]).any():
        raise CandidateContractError(
            "STARTER_UNIVERSE_MISMATCH", "duplicate runner in enriched snapshot"
        )
    populated_market_columns = []
    for column in frame.columns:
        text = str(column).strip()
        if text in CURRENT_MARKET_ENTRY_COLUMNS or text.lower().startswith("odds_"):
            values = frame[column].fillna("").astype(str).str.strip()
            if values.ne("").any():
                populated_market_columns.append(text)
    if populated_market_columns:
        raise CandidateContractError(
            "FORBIDDEN_CANDIDATE_INPUT_COLUMN",
            "enriched runner contains current market data: "
            + ", ".join(sorted(populated_market_columns)),
        )
    for column in RUNNER_SNAPSHOT_ALLOWED_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    runner = frame[RUNNER_SNAPSHOT_ALLOWED_COLUMNS].copy()
    runner = runner.sort_values(["race_id", "horse_no"], kind="mergesort")
    _atomic_write_csv(runner, runner_output_path)
    input_snapshot_hash = file_sha256(runner_output_path)
    bundle_sha256 = file_sha256(inference_bundle_path)
    feature_schema_hash = canonical_digest(bundle["feature_cols"])
    records: list[dict[str, Any]] = []
    for target in targets:
        race_id = str(target["race_id"])
        race_rows = runner[runner["race_id"].eq(race_id)]
        runners = sorted(
            {
                _canonical_integer_text(value)
                for value in race_rows["horse_no"].tolist()
                if _canonical_integer_text(value)
            },
            key=_horse_sort_key,
        )
        cutoff = parse_time(target["candidate_feature_cutoff_time"], config["timezone"])
        source_contract_ok = len(runners) >= 3 and source_observed_at <= cutoff
        record = {
            "race_id": race_id,
            "race_no": int(target["race_no"]),
            "source_url_or_local_path": str(raw_entry_path),
            "source_received_at": source_observed_at.isoformat(timespec="milliseconds"),
            "feature_input_max_source_event_time": source_observed_at.isoformat(timespec="milliseconds"),
            "candidate_feature_cutoff_time": target["candidate_feature_cutoff_time"],
            "runner_ids": runners,
            "starter_universe_hash_at_freeze": canonical_digest(
                {"race_id": race_id, "runners": runners}
            ),
            "input_snapshot_hash": input_snapshot_hash,
            "inference_bundle_hash": bundle_sha256,
            "feature_schema_hash": feature_schema_hash,
            "source_contract_ok": source_contract_ok,
        }
        record["source_record_hash"] = canonical_digest(record)
        records.append(record)
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "cohort_id": config["cohort_id"],
        "manifest_type": "candidate_feature_source_manifest",
        "target_manifest_path": str(target_manifest_path),
        "target_manifest_sha256": file_sha256(target_manifest_path),
        "raw_entry_path": str(raw_entry_path),
        "raw_entry_sha256": file_sha256(raw_entry_path),
        "runner_snapshot_path": str(runner_output_path),
        "runner_snapshot_sha256": input_snapshot_hash,
        "inference_bundle_path": str(inference_bundle_path),
        "inference_bundle_sha256": bundle_sha256,
        "feature_schema_hash": feature_schema_hash,
        "lineage_artifacts": lineage_artifacts,
        "records": records,
        "candidate_uses_odds": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    write_json_atomic(source_manifest_path, manifest)
    return {
        "runner_rows": int(len(runner)),
        "target_races": len(targets),
        "source_contract_ok_races": sum(
            1 for record in records if record["source_contract_ok"]
        ),
        "runner_snapshot_sha256": input_snapshot_hash,
        "source_manifest_sha256": file_sha256(source_manifest_path),
    }


def prepare_runner_snapshot(
    *,
    target_manifest_path: Path,
    raw_entry_path: Path,
    inference_bundle_path: Path,
    runner_output_path: Path,
    source_manifest_path: Path,
    work_dir: Path,
    config: dict[str, Any],
    baseline_config_path: Path | None = None,
    baseline_model_path: Path | None = None,
    historical_csv_path: Path | None = None,
    ability_history_dir: Path | None = None,
    recent_result_globs: list[str] | None = None,
    entry_globs: list[str] | None = None,
    precomputed_enriched_runner_path: Path | None = None,
    source_observed_at: datetime | None = None,
) -> dict[str, Any]:
    target_manifest = load_json_object(target_manifest_path)
    targets = validate_target_manifest(target_manifest, config)
    work_dir.mkdir(parents=True, exist_ok=True)
    observed_at = source_observed_at or datetime.fromtimestamp(
        raw_entry_path.stat().st_mtime, tz=ZoneInfo(config["timezone"])
    )
    lineage_paths = [target_manifest_path, raw_entry_path, inference_bundle_path]
    details: dict[str, Any] = {}
    if precomputed_enriched_runner_path is not None:
        enriched_path = precomputed_enriched_runner_path
        lineage_paths.append(enriched_path)
        details["precomputed_enriched_runner"] = True
    else:
        required_paths = {
            "baseline_config": baseline_config_path,
            "baseline_model": baseline_model_path,
            "historical_csv": historical_csv_path,
            "ability_history_dir": ability_history_dir,
        }
        missing = [name for name, path in required_paths.items() if path is None]
        if missing:
            raise ValueError("runner preparation paths missing: " + ", ".join(missing))
        assert baseline_config_path is not None
        assert baseline_model_path is not None
        assert historical_csv_path is not None
        assert ability_history_dir is not None
        baseline_config = load_json_object(baseline_config_path)
        sanitized_entry_path = work_dir / "entry_snapshot.sanitized.csv"
        details["entry"] = _sanitize_entry_snapshot(
            raw_entry_path=raw_entry_path,
            output_path=sanitized_entry_path,
            targets=targets,
            baseline_config=baseline_config,
        )
        prediction_path, prediction_summary = _run_baseline_prediction(
            sanitized_entry_path=sanitized_entry_path,
            baseline_config_path=baseline_config_path,
            baseline_model_path=baseline_model_path,
            historical_csv_path=historical_csv_path,
            work_dir=work_dir,
        )
        details["baseline"] = prediction_summary
        enriched_path, ability_summary = _run_basic_ability_enrichment(
            prediction_path=prediction_path,
            sanitized_entry_path=sanitized_entry_path,
            ability_history_dir=ability_history_dir,
            recent_result_globs=recent_result_globs or [],
            entry_globs=entry_globs or [],
            work_dir=work_dir,
        )
        details["ability"] = ability_summary
        lineage_paths.extend(
            [baseline_config_path, baseline_model_path, historical_csv_path, prediction_path]
        )
        lineage_paths.extend(ability_history_dir.glob("*"))
        lineage_paths.extend(_expand_glob_paths(recent_result_globs or []))
        lineage_paths.extend(_expand_glob_paths(entry_globs or []))
    lineage_artifacts = _hash_files(lineage_paths)
    summary = finalize_runner_snapshot(
        enriched_runner_path=enriched_path,
        runner_output_path=runner_output_path,
        source_manifest_path=source_manifest_path,
        raw_entry_path=raw_entry_path,
        target_manifest_path=target_manifest_path,
        inference_bundle_path=inference_bundle_path,
        config=config,
        source_observed_at=observed_at,
        lineage_artifacts=lineage_artifacts,
    )
    summary["details"] = details
    return summary


def load_feature_source_records(path: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json_object(path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("feature source manifest records must be a list")
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("feature source record must be an object")
        race_id = str(record.get("race_id", "")).strip()
        if not race_id or race_id in output:
            raise ValueError("feature source race_id must be present and unique")
        claimed_hash = str(record.get("source_record_hash", "")).strip()
        hash_payload = {key: value for key, value in record.items() if key != "source_record_hash"}
        if not _is_sha256(claimed_hash) or canonical_digest(hash_payload) != claimed_hash:
            record = dict(record)
            record["_source_record_hash_valid"] = False
        else:
            record = dict(record)
            record["_source_record_hash_valid"] = True
        output[race_id] = record
    return output


def load_top3_feature_rows(path: Path) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            grouped[str(row.get("race_id", "")).strip()].append(dict(row))
    return headers, dict(grouped)


def _first_value(row: dict[str, str], candidates: Iterable[str]) -> str:
    for candidate in candidates:
        value = str(row.get(candidate, "")).strip()
        if value:
            return value
    return ""


def _optional_float(row: dict[str, str], candidates: Iterable[str]) -> float | None:
    raw = _first_value(row, candidates)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _required_float(
    row: dict[str, str], candidates: Iterable[str], field_name: str
) -> float:
    value = _optional_float(row, candidates)
    if value is None:
        raise CandidateContractError(
            "CANDIDATE_SOURCE_NOT_READY", f"missing/non-numeric runner feature {field_name}"
        )
    return value


def _percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], _horse_sort_key(item[0])))
    result: dict[str, float] = {}
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        percentile = average_rank / len(ordered)
        for index in range(position, end):
            result[ordered[index][0]] = percentile
        position = end
    return result


def _derive_expected_pace(rows: list[dict[str, str]]) -> str:
    first = rows[0]
    pressure = _optional_float(first, ["race_early_pressure_score"])
    collapse = _optional_float(first, ["race_pace_collapse_risk"])
    slow = _optional_float(first, ["race_slow_pace_risk"])
    if pressure is not None and collapse is not None and slow is not None:
        if pressure >= 0.60 or collapse >= 0.55:
            return "fast"
        if slow >= 0.55 and pressure < 0.45:
            return "slow"
        return "middle"
    fallback = _first_value(first, ["expected_pace"]).lower()
    if fallback:
        return fallback
    raise CandidateContractError(
        "CANDIDATE_SOURCE_NOT_READY", "pace inputs and frozen expected_pace are missing"
    )


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _pair_features(
    left: dict[str, float],
    right: dict[str, float],
    *,
    field_size: int,
    front_runner_count: float,
    race_pressure: float,
    expected_pace: str,
) -> dict[str, float]:
    front_density = _clip(front_runner_count / field_size, 0.0, 1.0)
    pressure = _clip(race_pressure, 0.0, 1.0)
    front_hold = max(0.01, 0.40 + (1.0 - pressure) * 0.35 + (0.20 if "slow" in expected_pace else 0.0))
    pace_collapse = max(0.01, 0.20 + pressure * 0.55 + (0.20 if ("fast" in expected_pace or "high" in expected_pace) else 0.0))
    slow_sprint = max(0.01, 0.25 + (1.0 - pressure) * 0.20 + (0.25 if "slow" in expected_pace else 0.0))
    neutral = 0.25
    denominator = front_hold + pace_collapse + slow_sprint + neutral
    scenario = [
        front_hold / denominator,
        pace_collapse / denominator,
        slow_sprint / denominator,
        neutral / denominator,
    ]
    front_left = _clip(left["front"], 0.0, 1.0)
    front_right = _clip(right["front"], 0.0, 1.0)
    closer_left = _clip(left["closer"], 0.0, 1.0)
    closer_right = _clip(right["closer"], 0.0, 1.0)
    mid_left = _clip(1.0 - front_left - closer_left, 0.0, 1.0)
    mid_right = _clip(1.0 - front_right - closer_right, 0.0, 1.0)
    escape_left = _clip(front_left * left["front_rank_pct"], 0.0, 1.0)
    escape_right = _clip(front_right * right["front_rank_pct"], 0.0, 1.0)
    pair_escape_clash = _clip(
        escape_left * escape_right * pressure * (1.0 + front_density), 0.0, 2.0
    )
    pair_front_clash = _clip(
        front_left * front_right * pressure * (1.0 + front_density), 0.0, 2.0
    )
    pair_clash = max(pair_escape_clash, pair_front_clash)
    fits_left = [
        front_left,
        _clip(0.75 * closer_left + 0.25 * mid_left, 0.0, 1.0),
        _clip(0.70 * closer_left + 0.30 * (1.0 - front_left), 0.0, 1.0),
        _clip(0.45 * front_left + 0.25 * mid_left + 0.30 * closer_left, 0.0, 1.0),
    ]
    fits_right = [
        front_right,
        _clip(0.75 * closer_right + 0.25 * mid_right, 0.0, 1.0),
        _clip(0.70 * closer_right + 0.30 * (1.0 - front_right), 0.0, 1.0),
        _clip(0.45 * front_right + 0.25 * mid_right + 0.30 * closer_right, 0.0, 1.0),
    ]
    joint = [left_fit * right_fit for left_fit, right_fit in zip(fits_left, fits_right)]
    joint_fit = sum(weight * value for weight, value in zip(scenario, joint))
    shared_failure = sum(
        weight * (1.0 - left_fit) * (1.0 - right_fit)
        for weight, left_fit, right_fit in zip(scenario, fits_left, fits_right)
    )
    mean_joint = sum(joint) / len(joint)
    scenario_variance = sum((value - mean_joint) ** 2 for value in joint) / len(joint)
    return {
        "pair_joint_fit": joint_fit,
        "pair_clash_score": pair_clash,
        "pair_shared_failure": shared_failure,
        "pair_scenario_variance": scenario_variance,
    }


def build_top3_features_from_runner_rows(
    rows: list[dict[str, str]], bundle: dict[str, Any]
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        race_id = str(row.get("race_id", "")).strip()
        if race_id:
            grouped[race_id].append(row)
    output: dict[str, list[dict[str, str]]] = {}
    for race_id, race_rows in grouped.items():
        by_horse: dict[str, dict[str, str]] = {}
        for row in race_rows:
            horse_id = _first_value(row, ["horse_no", "horse_id"])
            if not horse_id or horse_id in by_horse:
                raise CandidateContractError(
                    "STARTER_UNIVERSE_MISMATCH", f"duplicate/missing runner identity in {race_id}"
                )
            by_horse[horse_id] = row
        runners = sorted(by_horse, key=_horse_sort_key)
        if len(runners) < 3:
            continue
        ai_scores = {
            horse_id: _required_float(by_horse[horse_id], ["ai_score"], "ai_score")
            for horse_id in runners
        }
        ai_ranks = {
            horse_id: _required_float(by_horse[horse_id], ["ai_rank"], "ai_rank")
            for horse_id in runners
        }
        primary_strength = _percentile_ranks(ai_scores)
        denominator = max(1.0, len(runners) - 1.0)
        rank_strength = {
            horse_id: _clip(1.0 - (ai_ranks[horse_id] - 1.0) / denominator, 0.0, 1.0)
            for horse_id in runners
        }
        front_values = {
            horse_id: _optional_float(
                by_horse[horse_id],
                ["front_running_tendency_x", "front_running_tendency", "horse_front_run_rate_past5"],
            )
            for horse_id in runners
        }
        closer_values = {
            horse_id: _optional_float(
                by_horse[horse_id],
                ["closing_tendency_x", "closing_tendency", "horse_closer_rate_past5"],
            )
            for horse_id in runners
        }
        front_rank_pct = _percentile_ranks(
            {horse_id: value if value is not None else 0.0 for horse_id, value in front_values.items()}
        )
        expected_pace = _derive_expected_pace(race_rows)
        first = race_rows[0]
        race_pressure = _optional_float(first, ["race_early_pressure_score"])
        if race_pressure is None:
            race_pressure = 0.5
        front_runner_count = _optional_float(
            first, ["race_front_runner_count_x", "race_front_runner_count", "race_need_lead_count"]
        )
        if front_runner_count is None:
            front_runner_count = len(runners) * 0.25
        runner_context = {
            horse_id: {
                "front": front_values[horse_id] if front_values[horse_id] is not None else 0.0,
                "closer": closer_values[horse_id] if closer_values[horse_id] is not None else 0.0,
                "front_rank_pct": front_rank_pct[horse_id],
            }
            for horse_id in runners
        }
        pair_lookup = {
            canonical_pair(left, right): _pair_features(
                runner_context[left],
                runner_context[right],
                field_size=len(runners),
                front_runner_count=front_runner_count,
                race_pressure=race_pressure,
                expected_pace=expected_pace,
            )
            for left, right in itertools.combinations(runners, 2)
        }
        race_output: list[dict[str, str]] = []
        for triplet in itertools.combinations(runners, 3):
            horse_rows = [by_horse[horse_id] for horse_id in triplet]
            floors = [
                _optional_float(row, ["ability_floor_score_5"]) for row in horse_rows
            ]
            m1c_any_core_missing = float(any(value is None for value in floors))
            floor_values = [value if value is not None else 0.5 for value in floors]
            stability = [
                _optional_float(row, ["ability_stability_score_3"]) for row in horse_rows
            ]
            stability_values = [value if value is not None else 0.5 for value in stability]
            recent_values = [
                _optional_float(row, ["recent_weighted_score_3"]) for row in horse_rows
            ]
            recent_values = [value if value is not None else 0.5 for value in recent_values]
            condition_values = [
                _optional_float(row, ["condition_adjusted_recent_ability_score"])
                for row in horse_rows
            ]
            condition_values = [value if value is not None else 0.5 for value in condition_values]
            experience_values = [
                _optional_float(row, ["career_shallow_flag"]) for row in horse_rows
            ]
            experience_values = [value if value is not None else 0.5 for value in experience_values]
            growth_values = [
                _optional_float(row, ["career_growth_zone_flag"]) for row in horse_rows
            ]
            growth_values = [value if value is not None else 0.5 for value in growth_values]
            pair_values = [pair_lookup[canonical_pair(left, right)] for left, right in itertools.combinations(triplet, 2)]
            sorted_floor = sorted(floor_values)
            sorted_stability = sorted(stability_values)
            feature_row: dict[str, str] = {
                "race_id": race_id,
                "horse_id_1": triplet[0],
                "horse_id_2": triplet[1],
                "horse_id_3": triplet[2],
                "sum_primary_strength": str(sum(primary_strength[horse_id] for horse_id in triplet)),
                "triplet_min_ability_floor": str(sorted_floor[0]),
                "triplet_second_min_ability_floor": str(sorted_floor[1]),
                "triplet_mean_ability_floor": str(sum(floor_values) / 3.0),
                "triplet_min_recent_stability": str(sorted_stability[0]),
                "triplet_second_min_recent_stability": str(sorted_stability[1]),
                "triplet_mean_recent_weighted": str(sum(recent_values) / 3.0),
                "triplet_mean_condition_recent": str(sum(condition_values) / 3.0),
                "triplet_experience_risk_count": str(sum(experience_values)),
                "triplet_growth_zone_count": str(sum(growth_values)),
                "m1c_any_core_missing": str(m1c_any_core_missing),
                "triplet_min_pair_joint_fit": str(min(value["pair_joint_fit"] for value in pair_values)),
                "triplet_max_pair_clash": str(max(value["pair_clash_score"] for value in pair_values)),
                "triplet_max_shared_failure": str(max(value["pair_shared_failure"] for value in pair_values)),
                "triplet_max_pair_scenario_variance": str(max(value["pair_scenario_variance"] for value in pair_values)),
            }
            missing_bundle_features = [
                feature for feature in bundle["feature_cols"] if feature not in feature_row
            ]
            if missing_bundle_features:
                raise CandidateContractError(
                    "CANDIDATE_SOURCE_NOT_READY",
                    "runner builder cannot produce bundle feature(s): "
                    + ", ".join(missing_bundle_features),
                )
            race_output.append(feature_row)
        output[race_id] = race_output
    return output


def load_runner_feature_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def validate_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("model_kind") != "linear_top3_set_softmax":
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "unexpected model kind")
    feature_cols = bundle.get("feature_cols")
    means = bundle.get("mean")
    stds = bundle.get("std")
    weights = bundle.get("weights")
    if not all(isinstance(value, list) for value in (feature_cols, means, stds, weights)):
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "bundle arrays missing")
    if not feature_cols or len({str(value) for value in feature_cols}) != len(feature_cols):
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "invalid feature schema")
    if not (len(feature_cols) == len(means) == len(stds) == len(weights)):
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "bundle length mismatch")
    numbers = [float(value) for value in means + stds + weights]
    if not all(math.isfinite(value) for value in numbers):
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "non-finite bundle value")
    if any(float(value) <= 0.0 for value in stds):
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "standard deviation must be positive")
    temperature = float(bundle.get("temperature", 0.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CandidateContractError("INFERENCE_BUNDLE_INVALID", "temperature must be positive")


def _row_utility(row: dict[str, str], bundle: dict[str, Any]) -> float:
    utility = 0.0
    for column, mean, std, weight in zip(
        bundle["feature_cols"], bundle["mean"], bundle["std"], bundle["weights"]
    ):
        raw = row.get(str(column))
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise CandidateContractError(
                "CANDIDATE_SOURCE_NOT_READY", f"missing/non-numeric feature {column}"
            ) from exc
        if not math.isfinite(value):
            raise CandidateContractError(
                "CANDIDATE_SOURCE_NOT_READY", f"non-finite feature {column}"
            )
        utility += ((value - float(mean)) / float(std)) * float(weight)
    return utility


def _softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exponents = [math.exp(value - maximum) for value in scaled]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]


def _logit_offset_probability(probability: float, intercept: float) -> float:
    clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
    logit = math.log(clipped / (1.0 - clipped)) + intercept
    return 1.0 / (1.0 + math.exp(-logit))


def _validate_source_record(
    source: dict[str, Any],
    *,
    target: dict[str, Any],
    bundle_sha256: str,
    feature_schema_hash: str,
    input_snapshot_hash: str,
    timezone_name: str,
) -> list[str]:
    if source.get("_source_record_hash_valid") is not True:
        raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", "source record hash invalid")
    if source.get("source_contract_ok") is not True:
        raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", "source contract failed")
    if str(source.get("inference_bundle_hash", "")) != bundle_sha256:
        raise CandidateContractError(
            "INFERENCE_BUNDLE_HASH_MISMATCH", "source record bundle hash mismatch"
        )
    if str(source.get("feature_schema_hash", "")) != feature_schema_hash:
        raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", "feature schema hash mismatch")
    for field in ("input_snapshot_hash", "starter_universe_hash_at_freeze"):
        if not _is_sha256(source.get(field)):
            raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", f"invalid {field}")
    if str(source.get("input_snapshot_hash", "")) != input_snapshot_hash:
        raise CandidateContractError(
            "CANDIDATE_SOURCE_NOT_READY", "input snapshot hash mismatch"
        )
    target_cutoff = parse_time(target["candidate_feature_cutoff_time"], timezone_name)
    source_cutoff = parse_time(source.get("candidate_feature_cutoff_time"), timezone_name)
    source_max = parse_time(source.get("feature_input_max_source_event_time"), timezone_name)
    source_received = parse_time(source.get("source_received_at"), timezone_name)
    if source_cutoff != target_cutoff or source_max > target_cutoff or source_received > target_cutoff:
        raise CandidateContractError(
            "FEATURE_SOURCE_TIME_VIOLATION", "feature source is later than the candidate cutoff"
        )
    runners_raw = source.get("runner_ids")
    if not isinstance(runners_raw, list):
        raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", "runner_ids missing")
    runners = sorted({str(value).strip() for value in runners_raw if str(value).strip()}, key=_horse_sort_key)
    if len(runners) < 3 or len(runners) != len(runners_raw):
        raise CandidateContractError("STARTER_UNIVERSE_MISMATCH", "invalid runner universe")
    return runners


def derive_candidate(
    rows: list[dict[str, str]],
    *,
    runners: list[str],
    bundle: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    forbidden = {str(value) for value in config["forbidden_candidate_columns"]}
    for row in rows:
        populated = [field for field in forbidden if str(row.get(field, "")).strip()]
        if populated:
            raise CandidateContractError(
                "FORBIDDEN_CANDIDATE_INPUT_COLUMN",
                "forbidden non-empty fields: " + ", ".join(sorted(populated)),
            )
    horse_columns = [str(value) for value in config["candidate_policy"]["horse_id_columns"]]
    observed: dict[tuple[str, str, str], dict[str, str]] = {}
    universe = set(runners)
    for row in rows:
        triplet = canonical_triplet(row.get(column, "") for column in horse_columns)
        if not set(triplet).issubset(universe) or triplet in observed:
            raise CandidateContractError(
                "STARTER_UNIVERSE_MISMATCH", "triplet duplicate or outside runner universe"
            )
        observed[triplet] = row
    expected = set(itertools.combinations(runners, 3))
    if set(observed) != expected:
        raise CandidateContractError(
            "PROBABILITY_CONTRACT_VIOLATION",
            f"expected {len(expected)} triplets, observed {len(observed)}",
        )
    ordered_triplets = sorted(observed, key=lambda values: tuple(_horse_sort_key(v) for v in values))
    utilities = [_row_utility(observed[triplet], bundle) for triplet in ordered_triplets]
    set_probabilities = _softmax(utilities, float(bundle["temperature"]))
    set_mass_error = abs(sum(set_probabilities) - 1.0)
    tolerance = float(config["probability_contract"]["tolerance"])
    if set_mass_error > tolerance:
        raise CandidateContractError("PROBABILITY_CONTRACT_VIOLATION", "Top3 set mass failed")
    wide: dict[tuple[str, str], float] = defaultdict(float)
    for triplet, probability in zip(ordered_triplets, set_probabilities):
        for horse_a, horse_b in itertools.combinations(triplet, 2):
            wide[canonical_pair(horse_a, horse_b)] += probability
    wide_mass_error = abs(sum(wide.values()) - 3.0)
    if wide_mass_error > tolerance or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0 + tolerance
        for value in wide.values()
    ):
        raise CandidateContractError("PROBABILITY_CONTRACT_VIOLATION", "WIDE mass failed")
    ranked = sorted(
        wide.items(),
        key=lambda item: (
            -item[1],
            _horse_sort_key(item[0][0]),
            _horse_sort_key(item[0][1]),
        ),
    )
    pair, top1_probability = ranked[0]
    top2_probability = ranked[1][1] if len(ranked) > 1 else 0.0
    offset = float(config["candidate_policy"]["action_calibrator_offset_intercept"])
    return {
        "candidate_horse_id_1": pair[0],
        "candidate_horse_id_2": pair[1],
        "candidate_pair_key": f"{pair[0]}-{pair[1]}",
        "p_wide_coherent_raw": top1_probability,
        "p_action_calibrated": _logit_offset_probability(top1_probability, offset),
        "top1_top2_margin": top1_probability - top2_probability,
        "confidence_gate_pass": top1_probability
        >= float(config["candidate_policy"]["primary_confidence_threshold"]),
        "set_probability_mass_error": set_mass_error,
        "wide_probability_mass_error": wide_mass_error,
        "probability_contract_ok": True,
    }


def _candidate_record_hash(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key not in HASH_FIELDS}
    return canonical_digest(payload)


def _base_record(
    *,
    config: dict[str, Any],
    target: dict[str, Any],
    source: dict[str, Any] | None,
    bundle_sha256: str,
    feature_schema_hash: str,
    start_time: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "cohort_id": config["cohort_id"],
        "card_id": f"{config['target_card']['race_date']}_{config['target_card']['venue_code']}",
        "race_id": str(target["race_id"]),
        "race_no": int(target["race_no"]),
        "target_registered": True,
        "scheduled_post_time_asof": target["scheduled_post_time"],
        "candidate_feature_cutoff_time": target["candidate_feature_cutoff_time"],
        "candidate_generation_started_at": start_time.isoformat(timespec="milliseconds"),
        "candidate_generation_completed_at": (start_time + timedelta(milliseconds=1)).isoformat(timespec="milliseconds"),
        "candidate_freeze_committed_at": (start_time + timedelta(milliseconds=2)).isoformat(timespec="milliseconds"),
        "inference_bundle_hash": bundle_sha256,
        "feature_schema_hash": feature_schema_hash,
        "candidate_policy_hash": canonical_digest(config["candidate_policy"]),
        "confidence_policy_hash": canonical_digest(
            {
                "signal": "p_wide_coherent_raw",
                "threshold": config["candidate_policy"]["primary_confidence_threshold"],
            }
        ),
        "input_snapshot_hash": str((source or {}).get("input_snapshot_hash", "")),
        "source_record_hash": str((source or {}).get("source_record_hash", "")),
        "feature_input_max_source_event_time": (source or {}).get("feature_input_max_source_event_time"),
        "starter_universe_hash_at_freeze": str((source or {}).get("starter_universe_hash_at_freeze", "")),
        "runner_count": len((source or {}).get("runner_ids", [])),
        "candidate_uses_odds": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }


def build_candidate_record(
    *,
    config: dict[str, Any],
    target: dict[str, Any],
    source: dict[str, Any] | None,
    rows: list[dict[str, str]],
    bundle: dict[str, Any],
    bundle_sha256: str,
    feature_schema_hash: str,
    input_snapshot_hash: str,
    start_time: datetime,
) -> dict[str, Any]:
    base = _base_record(
        config=config,
        target=target,
        source=source,
        bundle_sha256=bundle_sha256,
        feature_schema_hash=feature_schema_hash,
        start_time=start_time,
    )
    try:
        if source is None or not rows:
            raise CandidateContractError("CANDIDATE_SOURCE_NOT_READY", "source or feature rows missing")
        runners = _validate_source_record(
            source,
            target=target,
            bundle_sha256=bundle_sha256,
            feature_schema_hash=feature_schema_hash,
            input_snapshot_hash=input_snapshot_hash,
            timezone_name=config["timezone"],
        )
        candidate = derive_candidate(rows, runners=runners, bundle=bundle, config=config)
        base.update(candidate)
        base.update(
            {
                "record_status": "CANDIDATE_READY",
                "candidate_freeze_contract_ok": True,
                "failure_reason_codes": [],
            }
        )
    except CandidateContractError as exc:
        base.update(
            {
                "record_status": "FAILED",
                "candidate_freeze_contract_ok": False,
                "failure_reason_codes": [exc.reason],
                "failure_detail": exc.detail,
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
            }
        )
    base["candidate_freeze_record_hash"] = _candidate_record_hash(base)
    return base


def _write_or_verify_packet(packet_path: Path, record: dict[str, Any]) -> str:
    if packet_path.exists():
        existing = load_json_object(packet_path)
        if existing != record:
            raise ValueError(f"immutable candidate packet differs: {packet_path}")
    else:
        write_json_atomic(packet_path, record)
    persisted = load_json_object(packet_path)
    claimed_hash = str(persisted.get("candidate_freeze_record_hash", ""))
    if claimed_hash != _candidate_record_hash(persisted):
        raise ValueError(f"candidate packet hash verification failed: {packet_path}")
    return file_sha256(packet_path)


def _idempotency_key(config: dict[str, Any], race_id: str) -> str:
    return canonical_digest(
        {
            "cohort_id": config["cohort_id"],
            "event_type": "candidate_freeze_persist_ack",
            "experiment_id": config["experiment_id"],
            "race_id": race_id,
        }
    )


def _verify_ledger_record(record: dict[str, Any], output_dir: Path) -> None:
    if record.get("formal_buy") is not False or record.get("send_order") is not False:
        raise ValueError("unsafe ledger row")
    if record.get("stake") != 0 or record.get("candidate_uses_odds") is not False:
        raise ValueError("unsafe stake or odds flag")
    packet_path = output_dir / str(record.get("packet_path", ""))
    if not packet_path.is_file() or file_sha256(packet_path) != record.get("packet_file_sha256"):
        raise ValueError("ledger packet hash verification failed")
    packet = load_json_object(packet_path)
    if packet.get("candidate_freeze_record_hash") != record.get("candidate_freeze_record_hash"):
        raise ValueError("ledger candidate record hash mismatch")


def _build_summary(
    *,
    config: dict[str, Any],
    targets: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    relevant = [
        record
        for record in ledger
        if record.get("experiment_id") == config["experiment_id"]
        and record.get("cohort_id") == config["cohort_id"]
    ]
    target_ids = {str(target["race_id"]) for target in targets}
    counts = Counter(str(record.get("race_id", "")) for record in relevant)
    duplicates = sum(max(0, value - 1) for value in counts.values())
    observed = target_ids.intersection(counts)
    ready = sum(record.get("record_status") == "CANDIDATE_READY" for record in relevant)
    failures = sum(record.get("record_status") == "FAILED" for record in relevant)
    unsafe = sum(
        record.get("formal_buy") is not False
        or record.get("send_order") is not False
        or record.get("stake") != 0
        or record.get("candidate_uses_odds") is not False
        for record in relevant
    )
    missing = sorted(target_ids.difference(observed))
    if duplicates or unsafe or missing:
        status = "INVALID"
    elif ready == len(targets):
        status = "PASS"
    else:
        status = "IN_PROGRESS"
    reasons = Counter(
        reason
        for record in relevant
        for reason in record.get("failure_reason_codes", [])
    )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "cohort_id": config["cohort_id"],
        "status": status,
        "expected_target_rows": len(targets),
        "recorded_target_rows": len(observed),
        "candidate_ready_rows": ready,
        "failed_rows": failures,
        "candidate_freeze_packet_ledger_completeness": (
            len(observed) / len(targets) if targets else 0.0
        ),
        "missing_target_race_ids": missing,
        "duplicate_packet_rows": duplicates,
        "unsafe_rows": unsafe,
        "failure_reason_counts": dict(sorted(reasons.items())),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
        "roi_calculated": False,
    }


def run_adapter(
    *,
    target_manifest_path: Path,
    feature_source_manifest_path: Path,
    top3_feature_csv_path: Path | None,
    runner_feature_csv_path: Path | None,
    inference_bundle_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    now: datetime,
    execution_mode: str,
) -> dict[str, Any]:
    target_manifest = load_json_object(target_manifest_path)
    if target_manifest.get("data_class") != execution_mode:
        raise ValueError("execution mode and target manifest data_class mismatch")
    targets = validate_target_manifest(target_manifest, config)
    source_records = load_feature_source_records(feature_source_manifest_path)
    bundle_sha256 = file_sha256(inference_bundle_path)
    bundle = load_json_object(inference_bundle_path)
    bundle_error: CandidateContractError | None = None
    try:
        validate_bundle(bundle)
    except CandidateContractError as exc:
        bundle_error = exc
    if (top3_feature_csv_path is None) == (runner_feature_csv_path is None):
        raise ValueError("provide exactly one of top3_feature_csv_path or runner_feature_csv_path")
    input_path = top3_feature_csv_path or runner_feature_csv_path
    assert input_path is not None
    input_snapshot_hash = file_sha256(input_path)
    if top3_feature_csv_path is not None:
        _headers, feature_rows = load_top3_feature_rows(top3_feature_csv_path)
    else:
        try:
            feature_rows = build_top3_features_from_runner_rows(
                load_runner_feature_rows(input_path), bundle
            )
        except CandidateContractError as exc:
            feature_rows = {}
            bundle_error = bundle_error or exc
    if execution_mode == "real-data":
        expected = str(config["bundle_contract"]["production_bundle_sha256"])
        if bundle_sha256 != expected:
            bundle_error = CandidateContractError(
                "INFERENCE_BUNDLE_HASH_MISMATCH", "production bundle hash mismatch"
            )
        if bundle.get("candidate_policy") != config["bundle_contract"]["candidate_policy"]:
            bundle_error = CandidateContractError(
                "INFERENCE_BUNDLE_INVALID", "production candidate policy mismatch"
            )
    feature_schema_hash = canonical_digest(bundle.get("feature_cols", []))
    packets_dir = output_dir / "packets"
    ledger_path = output_dir / "candidate_freeze_ledger.jsonl"
    summary_path = output_dir / "candidate_freeze_summary.json"
    existing = read_jsonl(ledger_path)
    existing_by_race: dict[str, dict[str, Any]] = {}
    for record in existing:
        if record.get("experiment_id") != config["experiment_id"] or record.get("cohort_id") != config["cohort_id"]:
            continue
        race_id = str(record.get("race_id", ""))
        if race_id in existing_by_race:
            raise ValueError("duplicate candidate-freeze ledger row")
        _verify_ledger_record(record, output_dir)
        existing_by_race[race_id] = record

    for index, target in enumerate(targets):
        race_id = str(target["race_id"])
        if race_id in existing_by_race:
            continue
        start_time = now + timedelta(milliseconds=index * 10)
        source = source_records.get(race_id)
        rows = feature_rows.get(race_id, [])
        if bundle_error is not None:
            source_for_failure = dict(source or {})
            source_for_failure["source_contract_ok"] = False
            record = _base_record(
                config=config,
                target=target,
                source=source_for_failure,
                bundle_sha256=bundle_sha256,
                feature_schema_hash=feature_schema_hash,
                input_snapshot_hash=input_snapshot_hash,
                start_time=start_time,
            )
            record.update(
                {
                    "record_status": "FAILED",
                    "candidate_freeze_contract_ok": False,
                    "failure_reason_codes": [bundle_error.reason],
                    "failure_detail": bundle_error.detail,
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
                }
            )
            record["candidate_freeze_record_hash"] = _candidate_record_hash(record)
        else:
            record = build_candidate_record(
                config=config,
                target=target,
                source=source,
                rows=rows,
                bundle=bundle,
                bundle_sha256=bundle_sha256,
                feature_schema_hash=feature_schema_hash,
                input_snapshot_hash=input_snapshot_hash,
                start_time=start_time,
            )
        packet_relative = Path("packets") / f"{race_id}.candidate_freeze.json"
        packet_path = output_dir / packet_relative
        packet_sha = _write_or_verify_packet(packet_path, record)
        ack_time = start_time + timedelta(milliseconds=3)
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
            "candidate_freeze_record_hash": record["candidate_freeze_record_hash"],
            "candidate_freeze_persist_ack_at": ack_time.isoformat(timespec="milliseconds"),
            "packet_path": packet_relative.as_posix(),
            "packet_file_sha256": packet_sha,
            "candidate_uses_odds": False,
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
            "idempotency_key": _idempotency_key(config, race_id),
        }
        append_jsonl(ledger_path, ledger_event)
        _verify_ledger_record(ledger_event, output_dir)

    ledger = read_jsonl(ledger_path)
    summary = _build_summary(config=config, targets=targets, ledger=ledger)
    write_json_atomic(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build research-only non-odds Grade-R candidate-freeze packets."
    )
    parser.add_argument(
        "--operation",
        choices=(
            "candidate-freeze",
            "capture-entry-snapshot",
            "prepare-runner-snapshot",
        ),
        default="candidate-freeze",
    )
    parser.add_argument("--target-manifest-json", type=Path)
    parser.add_argument("--feature-source-manifest-json", type=Path)
    feature_input = parser.add_mutually_exclusive_group(required=False)
    feature_input.add_argument("--top3-feature-csv", type=Path)
    feature_input.add_argument("--runner-feature-csv", type=Path)
    parser.add_argument("--inference-bundle-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--raw-entry-csv", type=Path)
    parser.add_argument("--runner-output-csv", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--baseline-config-json", type=Path)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--historical-csv", type=Path)
    parser.add_argument("--ability-history-dir", type=Path)
    parser.add_argument("--recent-result-glob", action="append", default=[])
    parser.add_argument("--entry-glob", action="append", default=[])
    parser.add_argument("--precomputed-enriched-runner-csv", type=Path)
    parser.add_argument("--source-observed-at", default="")
    parser.add_argument("--capture-manifest-json", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--sleep-seconds", type=float, default=0.75)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "grade_r_candidate_freeze_adapter_v1.json",
    )
    parser.add_argument(
        "--execution-mode", choices=("synthetic", "real-data"), default="synthetic"
    )
    parser.add_argument("--now", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_adapter_config(args.config)
    if args.execution_mode == "real-data":
        assert_real_data_authorized(ROOT, str(config["experiment_id"]))
    common_required = {"target_manifest_json": args.target_manifest_json}
    common_missing = [name for name, value in common_required.items() if value is None]
    if common_missing:
        raise ValueError("required arguments missing: " + ", ".join(common_missing))
    assert args.target_manifest_json is not None
    if args.operation == "capture-entry-snapshot":
        capture_required = {
            "baseline_config_json": args.baseline_config_json,
            "raw_entry_csv": args.raw_entry_csv,
            "capture_manifest_json": args.capture_manifest_json,
            "cache_dir": args.cache_dir,
        }
        capture_missing = [
            name for name, value in capture_required.items() if value is None
        ]
        if capture_missing:
            raise ValueError("capture arguments missing: " + ", ".join(capture_missing))
        if args.sleep_seconds < 0:
            raise ValueError("sleep-seconds must be non-negative")
        assert args.baseline_config_json is not None
        assert args.raw_entry_csv is not None
        assert args.capture_manifest_json is not None
        assert args.cache_dir is not None
        summary = capture_public_entry_snapshot(
            target_manifest_path=args.target_manifest_json,
            baseline_config_path=args.baseline_config_json,
            output_csv_path=args.raw_entry_csv,
            capture_manifest_path=args.capture_manifest_json,
            cache_dir=args.cache_dir,
            config=config,
            refresh=args.refresh,
            sleep_seconds=args.sleep_seconds,
        )
        print(canonical_json(summary))
        return 0
    if args.inference_bundle_json is None:
        raise ValueError("required arguments missing: inference_bundle_json")
    assert args.inference_bundle_json is not None
    if args.operation == "prepare-runner-snapshot":
        preparation_required = {
            "feature_source_manifest_json": args.feature_source_manifest_json,
            "raw_entry_csv": args.raw_entry_csv,
            "runner_output_csv": args.runner_output_csv,
            "work_dir": args.work_dir,
        }
        preparation_missing = [
            name for name, value in preparation_required.items() if value is None
        ]
        if preparation_missing:
            raise ValueError(
                "runner preparation arguments missing: " + ", ".join(preparation_missing)
            )
        assert args.feature_source_manifest_json is not None
        assert args.raw_entry_csv is not None
        assert args.runner_output_csv is not None
        assert args.work_dir is not None
        observed_at = (
            parse_time(args.source_observed_at, config["timezone"])
            if args.source_observed_at
            else None
        )
        summary = prepare_runner_snapshot(
            target_manifest_path=args.target_manifest_json,
            raw_entry_path=args.raw_entry_csv,
            inference_bundle_path=args.inference_bundle_json,
            runner_output_path=args.runner_output_csv,
            source_manifest_path=args.feature_source_manifest_json,
            work_dir=args.work_dir,
            config=config,
            baseline_config_path=args.baseline_config_json,
            baseline_model_path=args.baseline_model,
            historical_csv_path=args.historical_csv,
            ability_history_dir=args.ability_history_dir,
            recent_result_globs=args.recent_result_glob,
            entry_globs=args.entry_glob,
            precomputed_enriched_runner_path=args.precomputed_enriched_runner_csv,
            source_observed_at=observed_at,
        )
        print(canonical_json(summary))
        return 0
    candidate_required = {
        "feature_source_manifest_json": args.feature_source_manifest_json,
        "output_dir": args.output_dir,
    }
    candidate_missing = [name for name, value in candidate_required.items() if value is None]
    if candidate_missing:
        raise ValueError("candidate arguments missing: " + ", ".join(candidate_missing))
    if (args.top3_feature_csv is None) == (args.runner_feature_csv is None):
        raise ValueError("candidate-freeze requires exactly one feature input")
    assert args.feature_source_manifest_json is not None
    assert args.output_dir is not None
    now = (
        parse_time(args.now, config["timezone"])
        if args.now
        else datetime.now(ZoneInfo(config["timezone"]))
    )
    summary = run_adapter(
        target_manifest_path=args.target_manifest_json,
        feature_source_manifest_path=args.feature_source_manifest_json,
        top3_feature_csv_path=args.top3_feature_csv,
        runner_feature_csv_path=args.runner_feature_csv,
        inference_bundle_path=args.inference_bundle_json,
        output_dir=args.output_dir,
        config=config,
        now=now,
        execution_mode=args.execution_mode,
    )
    print(canonical_json(summary))
    return 0 if summary["status"] != "INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
