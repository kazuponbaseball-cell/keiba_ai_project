from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse only the approved model family.  In particular, this module does not
# import the production feature, prediction, BUY, notification, or order stack.
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


EXPERIMENT_ID = "EXP-20260821-033"
VARIANT = "leakfree_predraw_baseline_v0"
PROPOSAL_DIGEST = "6993e9f5a6e0d6b2ef726bbc65fd047479a7b1bf79e948689a588f26f034ff6d"
SYNTHETIC_EXECUTION_KIND = "synthetic_fixture"
SYNTHETIC_ID_PREFIX = "SYN-"
DIRECT_LEAKAGE_ROOTS = {
    "race_winner_prior_strength",
    "race_top3_prior_strength_mean",
}
IDENTITY_COLUMNS = ["race_id", "horse_id"]
TIME_COLUMNS = ["prediction_event_time", "source_event_time", "received_at", "available_as_of"]
ENVELOPE_COLUMNS = [
    "record_kind",
    *IDENTITY_COLUMNS,
    *TIME_COLUMNS,
    "source_version",
    "source_content_sha256",
    "missing_reason",
]
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
JST = ZoneInfo("Asia/Tokyo")


class ContractError(ValueError):
    """Fail-closed contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _no_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ContractError(f"non-finite JSON constant is forbidden: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_object_pairs,
        parse_constant=reject_constant,
    )
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ContractError(f"path escapes repository: {relative_path}") from exc
    return path


@dataclass(frozen=True)
class ContractBundle:
    config_path: Path
    config: dict[str, Any]
    allowlist: dict[str, Any]
    denylist: dict[str, Any]
    leakage_manifest: dict[str, Any]
    fold_manifest: dict[str, Any]

    @property
    def numeric_features(self) -> list[str]:
        return list(self.allowlist["numeric_features"])

    @property
    def categorical_features(self) -> list[str]:
        return list(self.allowlist["categorical_features"])

    @property
    def ordered_features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


def _verify_hash_ref(reference: Mapping[str, Any]) -> Path:
    path = _repo_path(str(reference["path"]))
    _require(path.is_file(), f"frozen contract is missing: {reference['path']}")
    payload = path.read_bytes()
    normalized_lf = payload.replace(b"\r\n", b"\n")
    normalized_crlf = normalized_lf.replace(b"\n", b"\r\n")
    accepted = {
        hashlib.sha256(payload).hexdigest(),
        hashlib.sha256(normalized_lf).hexdigest(),
        hashlib.sha256(normalized_crlf).hexdigest(),
    }
    _require(
        reference["sha256"] in accepted,
        f"frozen contract hash mismatch beyond line-ending normalization: {reference['path']}",
    )
    return path


def _active_feature_universe(reference: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    active = load_json(_repo_path(str(reference["path"])))
    numeric = [*active.get("numeric_features", []), *active.get("generated_numeric_features", [])]
    categorical = [
        *active.get("categorical_features", []),
        *active.get("generated_categorical_features", []),
    ]
    _require(len(numeric) == len(set(numeric)), "active numeric feature universe has duplicates")
    _require(len(categorical) == len(set(categorical)), "active categorical feature universe has duplicates")
    return numeric, categorical


def validate_fold_contract(fold_manifest: Mapping[str, Any]) -> None:
    expected_names = [
        "history_warmup_only",
        "train",
        "purge_1",
        "validation",
        "purge_2",
        "calibration",
        "embargo",
        "untouched_outer_oos",
    ]
    blocks = list(fold_manifest.get("ordered_blocks", []))
    _require([item.get("name") for item in blocks] == expected_names, "fold names/order changed")
    previous_end: date | None = None
    for item in blocks:
        start = date.fromisoformat(str(item["start"]))
        end = date.fromisoformat(str(item["end"]))
        _require(start <= end, f"invalid fold interval: {item['name']}")
        if previous_end is not None:
            _require(previous_end < start, f"fold overlap or non-monotone boundary: {item['name']}")
        previous_end = end
    _require(int(fold_manifest.get("purge_embargo_days", -1)) == 28, "purge/embargo changed")
    _require(fold_manifest.get("race_overlap_allowed") is False, "race overlap must be forbidden")
    _require(fold_manifest.get("target_labels_allowed") is False, "target labels must be forbidden")


def validate_partition_race_disjointness(partitions: Mapping[str, Iterable[str]]) -> None:
    owner: dict[str, str] = {}
    for block_name, race_ids in partitions.items():
        for race_id in race_ids:
            key = str(race_id)
            if key in owner:
                raise ContractError(f"race overlap: {key} in {owner[key]} and {block_name}")
            owner[key] = block_name


def load_and_verify_contract(config_path: str | Path) -> ContractBundle:
    path = Path(config_path)
    if not path.is_absolute():
        path = _repo_path(str(path))
    config = load_json(path)

    _require(config.get("experiment_id") == EXPERIMENT_ID, "wrong experiment id")
    _require(config.get("variant") == VARIANT, "only the approved single variant is allowed")
    _require(config.get("research_only") is True, "research_only must be true")
    _require(config.get("formal_buy") is False, "formal_buy must remain false")
    _require(config.get("send_order") is False, "send_order must remain false")
    _require(config.get("stake") == 0, "stake must remain zero")
    for key in [
        "production_change_allowed",
        "champion_change_allowed",
        "notification_allowed",
        "order_allowed",
        "netkeiba_allowed",
    ]:
        _require(config.get(key) is False, f"{key} must remain false")

    _require(
        config.get("runtime_authorization")
        == {
            "synthetic_fixture_execution_kind": SYNTHETIC_EXECUTION_KIND,
            "real_data_required_registry_status": "running",
            "real_data_required_execution_kind": "real-data",
            "real_data_required_run_scope": True,
            "real_data_without_run_scope": "fail_closed",
        },
        "runtime authorization contract changed",
    )

    model = config["model"]
    _require(model["module"] == "src.train.simple_ranker", "model module changed")
    _require(model["class"] == "SimpleRaceRanker", "model family changed")
    _require(float(model["ridge_alpha"]) == 10.0, "ridge_alpha changed")
    _require(int(model["categorical_top_k"]) == 80, "categorical_top_k changed")
    _require(int(model["seed"]) == 20260823, "seed changed")
    _require(int(model["maximum_variants"]) == 1, "variant search is forbidden")
    _require(int(model["maximum_model_fits"]) == 1, "multiple model fits are forbidden")
    _require(int(model["maximum_calibrator_fits"]) == 1, "multiple calibrator fits are forbidden")

    proposal_ref = config["proposal_scope"]
    proposal_path = _repo_path(str(proposal_ref["path"]))
    proposal = load_json(proposal_path)
    _require(canonical_digest(proposal) == PROPOSAL_DIGEST, "proposal scope digest mismatch")
    _require(proposal_ref["digest"] == PROPOSAL_DIGEST, "config proposal digest mismatch")

    frozen = config["frozen_contracts"]
    allow_path = _verify_hash_ref(frozen["feature_allowlist"])
    deny_path = _verify_hash_ref(frozen["feature_denylist"])
    leakage_path = _verify_hash_ref(frozen["leakage_dependency_manifest"])
    fold_path = _verify_hash_ref(frozen["fold_manifest"])
    _verify_hash_ref(frozen["synthetic_fixture_plan"])
    active_path = _verify_hash_ref(frozen["active_reference_config"])
    _verify_hash_ref(frozen["simple_ranker_source"])
    _require(active_path == _repo_path("config/baseline_features_workout.json"), "active config path changed")

    allowlist = load_json(allow_path)
    denylist = load_json(deny_path)
    leakage = load_json(leakage_path)
    fold = load_json(fold_path)

    numeric = list(allowlist["numeric_features"])
    categorical = list(allowlist["categorical_features"])
    denied_numeric = list(denylist["numeric_features"])
    denied_categorical = list(denylist["categorical_features"])
    _require(len(numeric) == 77 and len(categorical) == 11, "allowlist must be 77+11")
    _require(len(denied_numeric) == 279 and len(denied_categorical) == 7, "denylist must be 279+7")
    _require(len(set(numeric + categorical)) == 88, "allowlist duplicates or overlap")
    _require(len(set(denied_numeric + denied_categorical)) == 286, "denylist duplicates or overlap")
    _require(set(numeric + categorical).isdisjoint(denied_numeric + denied_categorical), "allow/deny overlap")
    active_numeric, active_categorical = _active_feature_universe(frozen["active_reference_config"])
    _require(
        set(active_numeric) == set(numeric).union(denied_numeric),
        "numeric allow/deny is not the exact active universe",
    )
    _require(
        set(active_categorical) == set(categorical).union(denied_categorical),
        "categorical allow/deny is not the exact active universe",
    )
    _require(DIRECT_LEAKAGE_ROOTS.issubset(set(denied_numeric)), "direct leakage roots are not denied")
    _require(DIRECT_LEAKAGE_ROOTS.isdisjoint(numeric + categorical), "direct leakage root entered allowlist")
    _require(leakage.get("prefit_feature_to_feature_descendant_count") == 0, "leakage graph changed")
    _require(
        {item["feature"] for item in leakage["direct_leakage_roots"]} == DIRECT_LEAKAGE_ROOTS,
        "leakage root manifest changed",
    )
    validate_fold_contract(fold)
    return ContractBundle(path, config, allowlist, denylist, leakage, fold)


def _parse_aware_timestamp(value: Any, field_name: str) -> datetime:
    text = str(value).strip()
    _require(bool(text), f"{field_name} is empty")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"invalid {field_name}: {text}") from exc
    _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _is_blank(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _category(value: Any, missing_token: str = "__MISSING__") -> str:
    if _is_blank(value):
        return missing_token
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return missing_token if text.lower() in {"nan", "none", "<na>"} else text


def _json_scalar(value: Any) -> Any:
    if _is_blank(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        parsed = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
        _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "source payload datetime is naive")
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float):
            _require(math.isfinite(value), "source payload contains non-finite float")
        return value
    return str(value)


def canonical_source_content_sha256(row: Mapping[str, Any]) -> str:
    payload = {
        str(key): _json_scalar(value)
        for key, value in row.items()
        if key != "source_content_sha256" and not str(key).startswith("__")
    }
    return canonical_digest(payload)


def bind_source_content_hashes(records: pd.DataFrame) -> pd.DataFrame:
    """Bind each synthetic/canonical source row to its exact canonical payload."""
    out = records.copy(deep=True)
    out["source_content_sha256"] = [canonical_source_content_sha256(row) for row in out.to_dict("records")]
    return out


def _finite_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field_name} is not numeric") from exc
    _require(math.isfinite(number), f"{field_name} is non-finite")
    return number


def _is_positive_integer(value: Any) -> bool:
    if _is_blank(value):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number.is_integer() and number > 0


def validate_asof_lineage(records: pd.DataFrame, record_kind: str, bundle: ContractBundle) -> pd.DataFrame:
    required = set(ENVELOPE_COLUMNS)
    missing = sorted(required - set(records.columns))
    _require(not missing, f"{record_kind} source envelope missing columns: {missing}")
    out = records.copy(deep=True)
    _require(len(out) > 0, f"{record_kind} records are empty")
    _require(out[IDENTITY_COLUMNS].notna().all().all(), f"{record_kind} identity is missing")
    for index, row in out.iterrows():
        supplied_hash = str(row["source_content_sha256"]).strip()
        expected_hash = canonical_source_content_sha256(row.to_dict())
        _require(supplied_hash == expected_hash, f"{record_kind}[{index}] source payload hash mismatch")
    out["race_id"] = out["race_id"].map(_category).astype("string")
    out["horse_id"] = out["horse_id"].map(_category).astype("string")
    _require((out["race_id"] != "__MISSING__").all(), f"{record_kind} race_id is blank")
    _require((out["horse_id"] != "__MISSING__").all(), f"{record_kind} horse_id is blank")
    _require(not out.duplicated(IDENTITY_COLUMNS).any(), f"duplicate {record_kind} identity")
    _require(set(out["record_kind"].astype(str)) == {record_kind}, f"wrong record_kind in {record_kind} input")

    parsed: dict[str, list[datetime]] = {name: [] for name in TIME_COLUMNS}
    allowed_missing = set(bundle.config["source_contract"]["allowed_missing_reasons"])
    for index, row in out.iterrows():
        for name in TIME_COLUMNS:
            parsed[name].append(_parse_aware_timestamp(row[name], f"{record_kind}[{index}].{name}"))
        version = _category(row["source_version"])
        content_hash = str(row["source_content_sha256"]).strip()
        reason = str(row["missing_reason"]).strip()
        _require(version != "__MISSING__", f"{record_kind}[{index}] source_version is empty")
        _require(bool(HASH_RE.fullmatch(content_hash)), f"{record_kind}[{index}] content hash is invalid")
        _require(reason in allowed_missing, f"{record_kind}[{index}] missing_reason is not registered")

    for name, values in parsed.items():
        out[f"__{name}_utc"] = values
    card_kind = bundle.config["source_contract"]["card_record_kind"]
    result_kind = bundle.config["source_contract"]["result_record_kind"]
    for index, row in out.iterrows():
        prediction = row["__prediction_event_time_utc"]
        source = row["__source_event_time_utc"]
        received = row["__received_at_utc"]
        available = row["__available_as_of_utc"]
        if record_kind == card_kind:
            _require(source <= received <= available < prediction, f"card as-of order failed at row {index}")
        elif record_kind == result_kind:
            _require(prediction < source <= received <= available, f"result as-of order failed at row {index}")
        else:
            raise ContractError(f"unsupported source record kind: {record_kind}")
    return out


def _validate_cards(cards: pd.DataFrame, bundle: ContractBundle) -> pd.DataFrame:
    card_kind = bundle.config["source_contract"]["card_record_kind"]
    out = validate_asof_lineage(cards, card_kind, bundle)
    required = list(bundle.config["current_card_required_columns"])
    missing = sorted(set(required) - set(out.columns))
    _require(not missing, f"current card missing core fields: {missing}")
    for column in ["年齢", "斤量", "距離"]:
        converted = out[column].map(lambda value: _finite_float(value, column)).astype("float64")
        _require((converted > 0).all(), f"{column} must be positive")
        out[column] = converted
    for column in ["場所", "性別", "騎手コード", "調教師コード", "芝・ダ", "クラス名", "トラックコード"]:
        normalized = out[column].map(_category)
        _require((normalized != "__MISSING__").all(), f"core card field is missing: {column}")
        out[column] = normalized.astype("string")
    _require(set(out["芝・ダ"]).issubset({"芝", "ダ"}), "only JRA turf/dirt cards are permitted")
    race_time_counts = out.groupby("race_id")["__prediction_event_time_utc"].nunique()
    _require((race_time_counts == 1).all(), "one race_id has multiple prediction times")
    for column in ["距離", "場所", "芝・ダ", "クラス名", "トラックコード"]:
        _require((out.groupby("race_id")[column].nunique() == 1).all(), f"race-level card field is inconsistent: {column}")
    _require((out.groupby("race_id").size() >= 2).all(), "every race requires at least two declared runners")
    if "draw_status" in out.columns:
        _require(not (out["draw_status"].astype(str) == "scratched").any(), "scratched runners must be removed before materialization")
    return out


def _validate_results(results: pd.DataFrame, cards: pd.DataFrame, bundle: ContractBundle) -> pd.DataFrame:
    if len(results) == 0:
        columns = list(dict.fromkeys([*ENVELOPE_COLUMNS, *bundle.config["completed_result_required_columns"], *bundle.config["history_optional_columns"]]))
        return pd.DataFrame(columns=columns)
    result_kind = bundle.config["source_contract"]["result_record_kind"]
    out = validate_asof_lineage(results, result_kind, bundle)
    missing = sorted(set(bundle.config["completed_result_required_columns"]) - set(out.columns))
    _require(not missing, f"completed results missing fields: {missing}")
    optional_columns = list(bundle.config["history_optional_columns"])
    for column in optional_columns:
        if column not in out.columns:
            out[column] = pd.NA
    for index, row in out.iterrows():
        optional_missing = any(_is_blank(row[column]) for column in optional_columns)
        reason = str(row["missing_reason"])
        if optional_missing:
            _require(
                reason in {"source_not_recorded", "structurally_not_applicable"},
                f"result row {index} has an optional-field gap without an explicit missing reason",
            )
        else:
            _require(reason == "not_missing", f"result row {index} declares a missing reason but has no gap")
    card_keys = set(map(tuple, cards[IDENTITY_COLUMNS].astype(str).to_numpy()))
    result_keys = set(map(tuple, out[IDENTITY_COLUMNS].astype(str).to_numpy()))
    _require(result_keys.issubset(card_keys), "completed result has no matching card identity")

    card_lookup = cards.set_index(IDENTITY_COLUMNS)
    for index, row in out.iterrows():
        card = card_lookup.loc[(str(row["race_id"]), str(row["horse_id"]))]
        _require(
            row["__prediction_event_time_utc"] == card["__prediction_event_time_utc"],
            f"result/card prediction time mismatch at row {index}",
        )

    field_sizes = cards.groupby("race_id").size().to_dict()
    for race_id, group in out.groupby("race_id", sort=False):
        expected = int(field_sizes[str(race_id)])
        _require(len(group) == expected, f"partial completed result race is forbidden: {race_id}")
        ranks = [_finite_float(value, "確定着順") for value in group["確定着順"]]
        _require(all(number.is_integer() for number in ranks), f"non-integer finish rank: {race_id}")
        _require(sorted(int(number) for number in ranks) == list(range(1, expected + 1)), f"invalid finish ranks: {race_id}")
    return out


def _distance_category(distance: float | None) -> str:
    if distance is None:
        return "__MISSING__"
    if distance <= 1300:
        return "sprint"
    if distance <= 1600:
        return "mile"
    if distance <= 2000:
        return "middle"
    if distance <= 2400:
        return "classic"
    return "long"


def _class_level(value: str) -> float:
    level = 0.0
    for pattern, candidate in [
        ("新馬", 1.0),
        ("未勝利", 1.0),
        ("1勝", 2.0),
        ("2勝", 3.0),
        ("3勝", 4.0),
        ("ｵｰﾌﾟﾝ", 5.0),
        ("オープン", 5.0),
        ("OP", 5.0),
        ("L", 5.5),
        ("Ｇ３", 6.0),
        ("G3", 6.0),
        ("Ｇ２", 7.0),
        ("G2", 7.0),
        ("Ｇ１", 8.0),
        ("G1", 8.0),
    ]:
        if pattern in value:
            level = candidate
    return level


def _optional_position(value: Any) -> float | None:
    if _is_blank(value):
        return None
    number = _finite_float(value, "corner position")
    return number if number > 0 else None


def _mean(values: Iterable[float | None], default: float = 0.0) -> float:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else default


def _rate(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return _mean([float(bool(record[key])) for record in records], 0.0)


def _history_record(card: Mapping[str, Any], result: Mapping[str, Any], field_size: int) -> dict[str, Any]:
    finish = int(_finite_float(result["確定着順"], "確定着順"))
    corner4 = _optional_position(result.get("4角"))
    corner1 = _optional_position(result.get("1角"))
    corner2 = _optional_position(result.get("2角"))
    for label, position in [("1角", corner1), ("2角", corner2), ("4角", corner4)]:
        _require(position is None or position <= field_size, f"{label} exceeds declared field size")
    if corner1 is None:
        corner1 = corner2
    if corner1 is None:
        corner1 = corner4
    corner4_rate = None if corner4 is None else float(np.clip(corner4 / field_size, 0.0, 1.0))
    late_gain = None if corner4 is None else float(corner4 - finish)
    early_move = None if corner4 is None or corner1 is None else float(corner1 - corner4)
    return {
        "race_id": str(card["race_id"]),
        "horse_id": str(card["horse_id"]),
        "event_time": card["__prediction_event_time_utc"],
        "distance": float(card["距離"]),
        "distance_category": _distance_category(float(card["距離"])),
        "venue": str(card["場所"]),
        "surface": str(card["芝・ダ"]),
        "track_code": str(card["トラックコード"]),
        "class_name": str(card["クラス名"]),
        "jockey_code": str(card["騎手コード"]),
        "field_size": int(field_size),
        "finish": finish,
        "target_score": float((field_size + 1.0 - finish) / field_size),
        "target_win": finish == 1,
        "target_top3": finish <= 3,
        "corner4_rate": corner4_rate,
        "front_general": corner4_rate is not None and corner4_rate <= 0.25,
        "stalker_general": corner4_rate is not None and 0.25 < corner4_rate <= 0.45,
        "closer_general": corner4_rate is not None and corner4_rate >= 0.70,
        "front_deep": corner4_rate is not None and corner4_rate <= 0.18,
        "stalker_deep": corner4_rate is not None and 0.18 < corner4_rate <= 0.40,
        "midpack_deep": corner4_rate is not None and 0.40 < corner4_rate < 0.70,
        "closer_deep": corner4_rate is not None and corner4_rate >= 0.70,
        "late_gain": late_gain,
        "early_move": early_move,
        "card_source_event_time": card["__source_event_time_utc"],
        "card_received_at": card["__received_at_utc"],
        "card_available_as_of": card["__available_as_of_utc"],
        "card_source_version": str(card["source_version"]),
        "card_content_sha256": str(card["source_content_sha256"]),
        "card_missing_reason": str(card["missing_reason"]),
        "result_source_event_time": result["__source_event_time_utc"],
        "result_received_at": result["__received_at_utc"],
        "result_available_as_of": result["__available_as_of_utc"],
        "result_source_version": str(result["source_version"]),
        "result_content_sha256": str(result["source_content_sha256"]),
        "result_missing_reason": str(result["missing_reason"]),
    }


def _row_history_features(
    card: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    qa: dict[str, int],
) -> dict[str, Any]:
    ordered_history = sorted(history, key=lambda item: (item["event_time"], item["race_id"]))
    previous = ordered_history[-1] if ordered_history else None
    current_time: datetime = card["__prediction_event_time_utc"]
    current_distance = float(card["距離"])
    current_category = _distance_category(current_distance)
    if previous is None:
        qa["debut_rows"] += 1
        previous_distance = None
        interval_weeks = 0.0
        previous_surface = "__MISSING__"
        previous_track = "__MISSING__"
        previous_class = "__MISSING__"
        previous_jockey = "__MISSING__"
        previous_corner4_rate = 0.5
        qa["missing_previous_corner4_rows"] += 1
    else:
        previous_distance = float(previous["distance"])
        elapsed_days = max(
            0,
            (
                current_time.astimezone(JST).date()
                - previous["event_time"].astimezone(JST).date()
            ).days,
        )
        interval_weeks = float(elapsed_days // 7)
        previous_surface = str(previous["surface"])
        previous_track = str(previous["track_code"])
        previous_class = str(previous["class_name"])
        previous_jockey = str(previous["jockey_code"])
        if previous["corner4_rate"] is None:
            previous_corner4_rate = 0.5
            qa["missing_previous_corner4_rows"] += 1
        else:
            previous_corner4_rate = float(previous["corner4_rate"])

    past3 = ordered_history[-3:]
    past5 = ordered_history[-5:]
    distance_matches = [item for item in ordered_history if item["distance_category"] == current_category]
    venue_matches = [item for item in ordered_history if item["venue"] == str(card["場所"])]
    turf = [item for item in ordered_history if item["surface"] == "芝"]
    dirt = [item for item in ordered_history if item["surface"] == "ダ"]

    past3_front_count = float(sum(bool(item["front_general"]) for item in past3))
    past3_stalker_count = float(sum(bool(item["stalker_general"]) for item in past3))
    past3_closer_count = float(sum(bool(item["closer_general"]) for item in past3))
    front_tendency = (
        0.45 * (1.0 - float(np.clip(previous_corner4_rate, 0.0, 1.0)))
        + 0.40 * float(np.clip(past3_front_count / 3.0, 0.0, 1.0))
        + 0.15 * float(np.clip(past3_stalker_count / 3.0, 0.0, 1.0))
    )
    closing_tendency = (
        0.55 * float(np.clip(previous_corner4_rate, 0.0, 1.0))
        + 0.45 * float(np.clip(past3_closer_count / 3.0, 0.0, 1.0))
    )

    def history_summary(records: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
        starts = float(len(records))
        if not records:
            return 0.0, 0.0, 0.0
        return starts, _rate(records, "target_top3"), _mean(item["target_score"] for item in records)

    same_distance = history_summary(distance_matches)
    same_venue = history_summary(venue_matches)
    turf_starts, turf_top3, turf_score = history_summary(turf)
    dirt_starts, dirt_top3, dirt_score = history_summary(dirt)
    turf_win = _rate(turf, "target_win") if turf else 0.0
    dirt_win = _rate(dirt, "target_win") if dirt else 0.0
    if str(card["芝・ダ"]) == "芝":
        dirt_starts = dirt_top3 = dirt_score = dirt_win = 0.0
    else:
        turf_starts = turf_top3 = turf_score = turf_win = 0.0

    front5 = _rate(past5, "front_deep") if past5 else 0.0
    stalker5 = _rate(past5, "stalker_deep") if past5 else 0.0
    midpack5 = _rate(past5, "midpack_deep") if past5 else 0.0
    closer5 = _rate(past5, "closer_deep") if past5 else 0.0
    valid_position_rates = [float(item["corner4_rate"]) for item in past5 if item["corner4_rate"] is not None]
    position_volatility = float(np.std(valid_position_rates, ddof=0)) if len(valid_position_rates) >= 2 else 0.0
    previous_late_gain = 0.0 if previous is None or previous["late_gain"] is None else float(previous["late_gain"])
    previous_early_move = 0.0 if previous is None or previous["early_move"] is None else float(previous["early_move"])
    need_lead = float(np.clip(0.70 * front5 + 0.30 * float(previous_corner4_rate <= 0.18), 0.0, 1.0))
    can_rate = float(np.clip(front5 + stalker5, 0.0, 1.0))

    current_class = str(card["クラス名"])
    current_jockey = str(card["騎手コード"])
    distance_diff = 0.0 if previous_distance is None else current_distance - previous_distance
    class_move = 0.0 if previous is None else _class_level(current_class) - _class_level(previous_class)
    surface_switch = 0.0 if previous is None else float(previous_surface != str(card["芝・ダ"]))
    short = float(0 < interval_weeks <= 2)
    standard = float(3 <= interval_weeks <= 8)
    layoff = float(9 <= interval_weeks <= 16)
    long_layoff = float(interval_weeks >= 17)
    class_up = float(class_move > 0)
    class_down = float(class_move < 0)
    class_same = float(class_move == 0)
    big_distance = float(abs(distance_diff) >= 400)
    stress = 0.30 * long_layoff + 0.20 * short + 0.20 * big_distance + 0.15 * surface_switch + 0.15 * class_up
    bucket = 1.0 if short else 2.0 if standard else 3.0 if layoff else 4.0 if long_layoff else 0.0

    return {
        "年齢": float(card["年齢"]),
        "斤量": float(card["斤量"]),
        "出走頭数": 0.0,
        "距離": current_distance,
        "間隔": interval_weeks,
        "前距離": 0.0 if previous_distance is None else previous_distance,
        "past3_avg_score": _mean(item["target_score"] for item in past3),
        "past3_avg_corner4_position_rate": _mean(item["corner4_rate"] for item in past3),
        "prev_corner4_position_rate": previous_corner4_rate,
        "past3_front_run_count": past3_front_count,
        "past3_stalker_count": past3_stalker_count,
        "past3_closer_count": past3_closer_count,
        "front_running_tendency": front_tendency,
        "closing_tendency": closing_tendency,
        "race_front_runner_count": 0.0,
        "race_front_runner_ratio": 0.0,
        "race_closer_count": 0.0,
        "race_closer_ratio": 0.0,
        "race_early_pressure_score": 0.0,
        "front_pressure_rank_score": 0.0,
        "distance_diff": distance_diff,
        "class_changed": 0.0 if previous is None else float(previous_class != current_class),
        "jockey_changed": 0.0 if previous is None else float(previous_jockey != current_jockey),
        "same_distance_category_starts": same_distance[0],
        "same_distance_category_top3_rate": same_distance[1],
        "same_distance_category_avg_score": same_distance[2],
        "same_venue_starts": same_venue[0],
        "same_venue_top3_rate": same_venue[1],
        "same_venue_avg_score": same_venue[2],
        "horse_turf_starts": turf_starts,
        "horse_turf_win_rate": turf_win,
        "horse_turf_top3_rate": turf_top3,
        "horse_turf_avg_score": turf_score,
        "horse_dirt_starts": dirt_starts,
        "horse_dirt_win_rate": dirt_win,
        "horse_dirt_top3_rate": dirt_top3,
        "horse_dirt_avg_score": dirt_score,
        "race_surface_top3_rank_score": 0.0,
        "race_distance_top3_rank_score": 0.0,
        "race_weight_light_rank_score": 0.0,
        "horse_front_run_rate_past5": front5,
        "horse_stalker_rate_past5": stalker5,
        "horse_midpack_rate_past5": midpack5,
        "horse_closer_rate_past5": closer5,
        "horse_late_gain_avg_past5": _mean(item["late_gain"] for item in past5),
        "horse_early_move_avg_past5": _mean(item["early_move"] for item in past5),
        "horse_position_volatility_past5": position_volatility,
        "horse_need_lead_rate": need_lead,
        "horse_can_rate_rate": can_rate,
        "prev_late_gain": previous_late_gain,
        "prev_early_move": previous_early_move,
        "race_need_lead_count": 0.0,
        "race_need_lead_ratio": 0.0,
        "race_stalker_count_deep": 0.0,
        "race_midpack_count_deep": 0.0,
        "race_deep_closer_count": 0.0,
        "race_pace_collapse_risk": 0.0,
        "race_slow_pace_risk": 0.0,
        "solo_lead_potential": 0.0,
        "pace_fit_score": 0.0,
        "front_advantage_score": 0.0,
        "closer_advantage_score": 0.0,
        "positioning_advantage_score": 0.0,
        "rotation_short_rest_flag": short,
        "rotation_standard_rest_flag": standard,
        "rotation_layoff_9_16w_flag": layoff,
        "rotation_long_layoff_17w_plus_flag": long_layoff,
        "rotation_distance_up_flag": float(distance_diff >= 200),
        "rotation_distance_down_flag": float(distance_diff <= -200),
        "rotation_big_distance_change_flag": big_distance,
        "rotation_surface_switch_flag": surface_switch,
        "class_move_score": class_move,
        "rotation_class_up_flag": class_up,
        "rotation_class_down_flag": class_down,
        "rotation_same_class_flag": class_same,
        "rotation_stress_score": stress,
        "rotation_bucket_code": bucket,
        "場所": str(card["場所"]),
        "性別": str(card["性別"]),
        "騎手コード": str(card["騎手コード"]),
        "調教師コード": str(card["調教師コード"]),
        "芝・ダ": str(card["芝・ダ"]),
        "クラス名": current_class,
        "トラックコード": str(card["トラックコード"]),
        "前芝・ダ": previous_surface,
        "前走トラックコード": previous_track,
        "distance_category": current_category,
        "previous_distance_category": "__MISSING__" if previous_distance is None else _distance_category(previous_distance),
    }


def _normalized_rank(values: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    count = int(numeric.notna().sum())
    if count <= 1:
        return pd.Series(0.0, index=values.index, dtype="float64")
    rank = numeric.rank(ascending=not higher_is_better, method="average")
    return ((count - rank) / (count - 1)).fillna(0.0).astype("float64")


def _add_race_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for _, indexes in out.groupby("race_id", sort=False).groups.items():
        idx = list(indexes)
        size = float(len(idx))
        out.loc[idx, "出走頭数"] = size
        front = (out.loc[idx, "front_running_tendency"] >= 0.45).astype(float)
        closer = (out.loc[idx, "closing_tendency"] >= 0.45).astype(float)
        front_count = float(front.sum())
        closer_count = float(closer.sum())
        out.loc[idx, "race_front_runner_count"] = front_count
        out.loc[idx, "race_front_runner_ratio"] = front_count / size
        out.loc[idx, "race_closer_count"] = closer_count
        out.loc[idx, "race_closer_ratio"] = closer_count / size
        pressure = 0.7 * front_count / size + 0.3 * min(front_count, 5.0) / 5.0
        out.loc[idx, "race_early_pressure_score"] = pressure
        out.loc[idx, "front_pressure_rank_score"] = _normalized_rank(
            out.loc[idx, "front_running_tendency"], True
        ).to_numpy()

        surface_values = pd.Series(0.0, index=idx, dtype="float64")
        for row_index in idx:
            surface_values.loc[row_index] = (
                float(out.loc[row_index, "horse_turf_top3_rate"])
                if out.loc[row_index, "芝・ダ"] == "芝"
                else float(out.loc[row_index, "horse_dirt_top3_rate"])
            )
        out.loc[idx, "race_surface_top3_rank_score"] = _normalized_rank(surface_values, True).to_numpy()
        out.loc[idx, "race_distance_top3_rank_score"] = _normalized_rank(
            out.loc[idx, "same_distance_category_top3_rate"], True
        ).to_numpy()
        out.loc[idx, "race_weight_light_rank_score"] = _normalized_rank(out.loc[idx, "斤量"], False).to_numpy()

        need = (out.loc[idx, "horse_need_lead_rate"] >= 0.45).astype(float)
        stalker = (out.loc[idx, "horse_stalker_rate_past5"] >= 0.35).astype(float)
        midpack = (out.loc[idx, "horse_midpack_rate_past5"] >= 0.35).astype(float)
        deep_closer = (out.loc[idx, "horse_closer_rate_past5"] >= 0.40).astype(float)
        need_count = float(need.sum())
        stalker_count = float(stalker.sum())
        midpack_count = float(midpack.sum())
        deep_closer_count = float(deep_closer.sum())
        collapse = float(np.clip(0.45 * min(need_count, 5.0) / 5.0 + 0.35 * pressure + 0.20 * front_count / size, 0.0, 1.0))
        slow = float(np.clip(1.0 - 0.55 * min(need_count, 4.0) / 4.0 - 0.30 * pressure - 0.15 * min(stalker_count, 5.0) / 5.0, 0.0, 1.0))
        out.loc[idx, "race_need_lead_count"] = need_count
        out.loc[idx, "race_need_lead_ratio"] = need_count / size
        out.loc[idx, "race_stalker_count_deep"] = stalker_count
        out.loc[idx, "race_midpack_count_deep"] = midpack_count
        out.loc[idx, "race_deep_closer_count"] = deep_closer_count
        out.loc[idx, "race_pace_collapse_risk"] = collapse
        out.loc[idx, "race_slow_pace_risk"] = slow
        solo = need * float(need_count == 1.0) * (0.5 + 0.5 * slow)
        out.loc[idx, "solo_lead_potential"] = solo.to_numpy()

        late_score = np.clip(
            0.55 * out.loc[idx, "horse_closer_rate_past5"]
            + 0.25 * (out.loc[idx, "horse_late_gain_avg_past5"].clip(-3, 6) + 3.0) / 9.0
            + 0.20 * out.loc[idx, "closing_tendency"].clip(0.0, 1.0),
            0.0,
            1.0,
        )
        front_score = np.clip(
            0.45 * out.loc[idx, "horse_front_run_rate_past5"]
            + 0.25 * out.loc[idx, "horse_stalker_rate_past5"]
            + 0.20 * (1.0 - out.loc[idx, "prev_corner4_position_rate"].clip(0.0, 1.0))
            + 0.10 * solo,
            0.0,
            1.0,
        )
        front_adv = front_score * (0.65 * slow + 0.35 * solo)
        closer_adv = late_score * collapse
        positioning = front_adv + closer_adv - 0.20 * out.loc[idx, "horse_position_volatility_past5"]
        pace_fit = 0.45 * front_adv + 0.45 * closer_adv + 0.10 * out.loc[idx, "horse_can_rate_rate"] * (1.0 - collapse)
        out.loc[idx, "front_advantage_score"] = front_adv.to_numpy()
        out.loc[idx, "closer_advantage_score"] = closer_adv.to_numpy()
        out.loc[idx, "positioning_advantage_score"] = positioning.to_numpy()
        out.loc[idx, "pace_fit_score"] = pace_fit.to_numpy()
    return out


def finalize_ordered_schema(generated: pd.DataFrame, bundle: ContractBundle) -> pd.DataFrame:
    expected = [*IDENTITY_COLUMNS, *bundle.ordered_features]
    actual = list(generated.columns)
    _require(len(actual) == len(set(actual)), "generated matrix has duplicate columns")
    _require(actual == expected, f"generated matrix schema mismatch; expected exact ordered 88 features")
    out = generated.copy()
    for column in IDENTITY_COLUMNS:
        out[column] = out[column].map(_category).astype("string")
    _require(not out.duplicated(IDENTITY_COLUMNS).any(), "generated matrix has duplicate identities")
    for column in bundle.numeric_features:
        values = pd.to_numeric(out[column], errors="coerce").astype("float64")
        _require(np.isfinite(values.to_numpy()).all(), f"non-finite numeric feature: {column}")
        out[column] = values
    for column in bundle.categorical_features:
        out[column] = out[column].map(_category).astype("string")
        _require(out[column].notna().all(), f"categorical feature missing after normalization: {column}")
    validate_feature_frame(out, bundle)
    return out


def validate_feature_frame(frame: pd.DataFrame, bundle: ContractBundle) -> None:
    expected = [*IDENTITY_COLUMNS, *bundle.ordered_features]
    _require(list(frame.columns) == expected, "feature frame is not exact identity + ordered 88 allowlist")
    _require(not frame.columns.duplicated().any(), "feature frame has duplicate columns")
    _require(not frame.duplicated(IDENTITY_COLUMNS).any(), "feature frame has duplicate identities")
    for column in IDENTITY_COLUMNS:
        _require(str(frame[column].dtype) == "string", f"identity dtype mismatch: {column}")
        _require(
            bool(frame[column].map(_category).ne("__MISSING__").all()),
            f"feature frame identity is blank: {column}",
        )
    for column in bundle.numeric_features:
        _require(str(frame[column].dtype) == "float64", f"numeric dtype mismatch: {column}")
        _require(np.isfinite(frame[column].to_numpy()).all(), f"non-finite numeric feature: {column}")
    for column in bundle.categorical_features:
        _require(str(frame[column].dtype) == "string", f"categorical dtype mismatch: {column}")
        _require(frame[column].notna().all(), f"categorical missing value: {column}")


@dataclass
class MaterializationResult:
    features: pd.DataFrame
    lineage: pd.DataFrame
    qa: dict[str, Any]


CURRENT_FOCAL_FEATURES = {
    "年齢",
    "斤量",
    "距離",
    "場所",
    "性別",
    "騎手コード",
    "調教師コード",
    "芝・ダ",
    "クラス名",
    "トラックコード",
    "distance_category",
}
RACE_CURRENT_FEATURES = {"出走頭数", "race_weight_light_rank_score"}
RACE_HISTORY_FEATURES = {
    "race_front_runner_count",
    "race_front_runner_ratio",
    "race_closer_count",
    "race_closer_ratio",
    "race_early_pressure_score",
    "front_pressure_rank_score",
    "race_surface_top3_rank_score",
    "race_distance_top3_rank_score",
    "race_need_lead_count",
    "race_need_lead_ratio",
    "race_stalker_count_deep",
    "race_midpack_count_deep",
    "race_deep_closer_count",
    "race_pace_collapse_risk",
    "race_slow_pace_risk",
    "solo_lead_potential",
    "pace_fit_score",
    "front_advantage_score",
    "closer_advantage_score",
    "positioning_advantage_score",
}
CORNER_DEPENDENT_FEATURES = {
    "past3_avg_corner4_position_rate",
    "prev_corner4_position_rate",
    "past3_front_run_count",
    "past3_stalker_count",
    "past3_closer_count",
    "front_running_tendency",
    "closing_tendency",
    "horse_front_run_rate_past5",
    "horse_stalker_rate_past5",
    "horse_midpack_rate_past5",
    "horse_closer_rate_past5",
    "horse_late_gain_avg_past5",
    "horse_early_move_avg_past5",
    "horse_position_volatility_past5",
    "horse_need_lead_rate",
    "horse_can_rate_rate",
    "prev_late_gain",
    "prev_early_move",
    *RACE_HISTORY_FEATURES,
}


def _source_context(card: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hashes = {str(card["source_content_sha256"])}
    source_events = [card["__source_event_time_utc"]]
    received_times = [card["__received_at_utc"]]
    available_times = [card["__available_as_of_utc"]]
    versions = {str(card["source_version"])}
    reasons = {str(card["missing_reason"])}
    for item in history:
        hashes.update([item["card_content_sha256"], item["result_content_sha256"]])
        source_events.extend([item["card_source_event_time"], item["result_source_event_time"]])
        received_times.extend([item["card_received_at"], item["result_received_at"]])
        available_times.extend([item["card_available_as_of"], item["result_available_as_of"]])
        versions.update([item["card_source_version"], item["result_source_version"]])
        reasons.update([item["card_missing_reason"], item["result_missing_reason"]])
    previous = max(history, key=lambda item: (item["event_time"], item["race_id"])) if history else None
    return {
        "hashes": hashes,
        "source_events": source_events,
        "received_times": received_times,
        "available_times": available_times,
        "versions": versions,
        "dependency_missing_reasons": reasons,
        "debut": not history,
        "previous_corner_missing": previous is None or previous["corner4_rate"] is None,
    }


def _merge_source_contexts(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(contexts), "cannot merge an empty lineage context")
    return {
        "hashes": set().union(*(context["hashes"] for context in contexts)),
        "source_events": [value for context in contexts for value in context["source_events"]],
        "received_times": [value for context in contexts for value in context["received_times"]],
        "available_times": [value for context in contexts for value in context["available_times"]],
        "versions": set().union(*(context["versions"] for context in contexts)),
        "dependency_missing_reasons": set().union(
            *(context["dependency_missing_reasons"] for context in contexts)
        ),
        "debut": any(bool(context["debut"]) for context in contexts),
        "previous_corner_missing": any(bool(context["previous_corner_missing"]) for context in contexts),
    }


def _feature_missing_reason(feature: str, context: Mapping[str, Any]) -> str:
    if feature in CURRENT_FOCAL_FEATURES or feature in RACE_CURRENT_FEATURES:
        return "not_missing"
    if bool(context["debut"]):
        return "debut_no_history"
    if feature in CORNER_DEPENDENT_FEATURES and bool(context["previous_corner_missing"]):
        return "source_not_recorded"
    return "not_missing"


def _lineage_rows(
    card: Mapping[str, Any],
    focal_card_context: Mapping[str, Any],
    focal_history_context: Mapping[str, Any],
    race_card_context: Mapping[str, Any],
    race_history_context: Mapping[str, Any],
    bundle: ContractBundle,
) -> list[dict[str, Any]]:
    prediction = card["__prediction_event_time_utc"]
    rows: list[dict[str, Any]] = []
    for feature in bundle.ordered_features:
        if feature in CURRENT_FOCAL_FEATURES:
            context = focal_card_context
        elif feature in RACE_CURRENT_FEATURES:
            context = race_card_context
        elif feature in RACE_HISTORY_FEATURES:
            context = race_history_context
        else:
            context = focal_history_context
        max_available = max(context["available_times"])
        _require(max_available < prediction, "lineage includes unavailable or future source")
        dependency_hashes = sorted(context["hashes"])
        dependency_digest = hashlib.sha256("\n".join(dependency_hashes).encode("ascii")).hexdigest()
        rows.append(
            {
                "race_id": str(card["race_id"]),
                "horse_id": str(card["horse_id"]),
                "feature_name": feature,
                "prediction_event_time": prediction.isoformat().replace("+00:00", "Z"),
                "source_event_time": max(context["source_events"]).isoformat().replace("+00:00", "Z"),
                "received_at": max(context["received_times"]).isoformat().replace("+00:00", "Z"),
                "available_as_of": max_available.isoformat().replace("+00:00", "Z"),
                "source_version": "|".join(sorted(context["versions"])),
                "content_hash": dependency_digest,
                "dependency_count": len(dependency_hashes),
                "missing_reason": _feature_missing_reason(feature, context),
                "dependency_missing_reasons": "|".join(sorted(context["dependency_missing_reasons"])),
                "transformation_version": bundle.config["feature_generation"]["implementation_version"],
                "as_of_safe": True,
            }
        )
    return rows


def materialize_predraw_features(
    cards: pd.DataFrame,
    results: pd.DataFrame,
    bundle: ContractBundle,
) -> MaterializationResult:
    safe_cards = _validate_cards(cards, bundle)
    safe_results = _validate_results(results, safe_cards, bundle)
    card_lookup = safe_cards.set_index(IDENTITY_COLUMNS, drop=False)
    field_sizes = safe_cards.groupby("race_id").size().astype(int).to_dict()
    result_batches: list[dict[str, Any]] = []
    if len(safe_results):
        # A race result becomes history atomically at the last runner's
        # available_as_of.  Never expose a partially received result race.
        for result_race_id, result_group in safe_results.groupby("race_id", sort=False):
            result_batches.append(
                {
                    "race_id": str(result_race_id),
                    "available_as_of": max(result_group["__available_as_of_utc"]),
                    "records": result_group.sort_values("horse_id", kind="mergesort").to_dict("records"),
                }
            )
        result_batches.sort(key=lambda item: (item["available_as_of"], item["race_id"]))
    result_batch_cursor = 0
    history_by_horse: dict[str, list[dict[str, Any]]] = {}
    materialized_identities: set[tuple[str, str]] = set()
    feature_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    lineage_inputs: dict[tuple[str, str], dict[str, Any]] = {}
    qa = {
        "input_card_rows": int(len(safe_cards)),
        "input_result_rows": int(len(safe_results)),
        "debut_rows": 0,
        "missing_previous_corner4_rows": 0,
        "results_applied": 0,
        "future_or_same_event_results_applied": 0,
        "raw_forbidden_columns_ignored": sorted(
            set(cards.columns)
            & (
                set(bundle.denylist["numeric_features"])
                | set(bundle.denylist["categorical_features"])
                | {"確定着順", "target_score", "target_win", "target_top3", "odds", "人気", "ROI", "BUY", "stake"}
            )
        ),
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }

    ordered_cards = safe_cards.sort_values(
        ["__prediction_event_time_utc", "race_id", "horse_id"], kind="mergesort"
    )
    for prediction_time, batch in ordered_cards.groupby("__prediction_event_time_utc", sort=True):
        while result_batch_cursor < len(result_batches):
            result_batch = result_batches[result_batch_cursor]
            if result_batch["available_as_of"] >= prediction_time:
                break
            staged_records: list[tuple[str, dict[str, Any]]] = []
            for result in result_batch["records"]:
                key = (str(result["race_id"]), str(result["horse_id"]))
                _require(key in materialized_identities, "result became available before its card batch was materialized")
                card = card_lookup.loc[key]
                _require(card["__prediction_event_time_utc"] < prediction_time, "same/future race result dependency")
                staged_records.append(
                    (
                        str(result["horse_id"]),
                        _history_record(card, result, int(field_sizes[str(result["race_id"])])),
                    )
                )
            for horse_id, record in staged_records:
                history = history_by_horse.setdefault(horse_id, [])
                history.append(record)
                history.sort(key=lambda item: (item["event_time"], item["race_id"]))
            qa["results_applied"] += len(staged_records)
            result_batch_cursor += 1

        batch_feature_start = len(feature_rows)
        for _, card in batch.iterrows():
            horse_id = str(card["horse_id"])
            history = list(history_by_horse.get(horse_id, []))
            _require(all(item["result_available_as_of"] < prediction_time for item in history), "history as-of violation")
            values = _row_history_features(card, history, qa)
            key = (str(card["race_id"]), horse_id)
            feature_rows.append({"race_id": key[0], "horse_id": key[1], **values})
            lineage_inputs[key] = {"card": card.to_dict(), "history": history}
        for _, card in batch.iterrows():
            materialized_identities.add((str(card["race_id"]), str(card["horse_id"])))
        _require(len(feature_rows) - batch_feature_start == len(batch), "event batch materialization mismatch")

    raw = pd.DataFrame(feature_rows)
    raw = _add_race_aggregates(raw)
    for race_id, race_group in raw.groupby("race_id", sort=False):
        keys = [(str(race_id), str(horse_id)) for horse_id in race_group["horse_id"]]
        card_context_by_key = {
            key: _source_context(lineage_inputs[key]["card"], []) for key in keys
        }
        history_context_by_key = {
            key: _source_context(lineage_inputs[key]["card"], lineage_inputs[key]["history"])
            for key in keys
        }
        race_card_context = _merge_source_contexts(list(card_context_by_key.values()))
        race_history_context = _merge_source_contexts(list(history_context_by_key.values()))
        for key in keys:
            lineage_rows.extend(
                _lineage_rows(
                    lineage_inputs[key]["card"],
                    card_context_by_key[key],
                    history_context_by_key[key],
                    race_card_context,
                    race_history_context,
                    bundle,
                )
            )
    ordered = raw[[*IDENTITY_COLUMNS, *bundle.ordered_features]].copy()
    finalized = finalize_ordered_schema(ordered, bundle)
    lineage = pd.DataFrame(lineage_rows).sort_values(
        ["race_id", "horse_id", "feature_name"], kind="mergesort"
    ).reset_index(drop=True)
    _require(len(lineage) == len(finalized) * 88, "feature-level lineage is incomplete")
    _require(bool(lineage["as_of_safe"].all()), "feature lineage is not fully as-of safe")
    qa["output_rows"] = int(len(finalized))
    qa["feature_count"] = 88
    qa["lineage_rows"] = int(len(lineage))
    qa["lineage_pass_fraction"] = 1.0
    return MaterializationResult(finalized, lineage, qa)


def targets_from_results(cards: pd.DataFrame, results: pd.DataFrame, bundle: ContractBundle) -> pd.DataFrame:
    safe_cards = _validate_cards(cards, bundle)
    safe_results = _validate_results(results, safe_cards, bundle)
    if safe_results.empty:
        return pd.DataFrame(columns=[*IDENTITY_COLUMNS, "target_score", "target_win", "target_top3"])
    field_sizes = safe_cards.groupby("race_id").size().astype(float).to_dict()
    rows = []
    for _, result in safe_results.iterrows():
        size = field_sizes[str(result["race_id"])]
        finish = _finite_float(result["確定着順"], "確定着順")
        rows.append(
            {
                "race_id": str(result["race_id"]),
                "horse_id": str(result["horse_id"]),
                "target_score": float((size + 1.0 - finish) / size),
                "target_win": int(finish == 1),
                "target_top3": int(finish <= 3),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class FitBudget:
    model_fits: int = 0
    calibrator_fits: int = 0

    def consume_model_fit(self) -> None:
        self.model_fits += 1
        _require(self.model_fits <= 1, "multiple model fits are forbidden")

    def consume_calibrator_fit(self) -> None:
        self.calibrator_fits += 1
        _require(self.calibrator_fits <= 1, "multiple calibrator fits are forbidden")


def _model_frame(features: pd.DataFrame, bundle: ContractBundle) -> pd.DataFrame:
    validate_feature_frame(features, bundle)
    return features[bundle.ordered_features].copy()


def _frozen_schema(model: SimpleRaceRanker, bundle: ContractBundle) -> dict[str, Any]:
    _require(model.coefficients_ is not None, "cannot freeze an unfitted model")
    coefficient_hash = hashlib.sha256(
        np.asarray(model.coefficients_, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    schema = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "variant": VARIANT,
        "ordered_features": bundle.ordered_features,
        "numeric_features": bundle.numeric_features,
        "categorical_features": bundle.categorical_features,
        "dtypes": {
            **{name: "float64" for name in bundle.numeric_features},
            **{name: "string" for name in bundle.categorical_features},
        },
        "numeric_missing_policy": "no non-finite value after event-batch materialization",
        "categorical_missing_token": "__MISSING__",
        "unknown_category_policy": "all_zero_reference_encoding",
        "categorical_levels": model.categorical_levels_,
        "numeric_medians": model.numeric_medians_,
        "numeric_means": model.numeric_means_,
        "numeric_stds": model.numeric_stds_,
        "design_feature_names": model.feature_names_,
        "coefficients_sha256": coefficient_hash,
        "model_family": "SimpleRaceRanker",
        "ridge_alpha": model.ridge_alpha,
        "categorical_top_k": model.categorical_top_k,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    schema["schema_digest"] = canonical_digest(schema)
    return schema


def fit_clean_ranker(
    features: pd.DataFrame,
    targets: pd.Series | Sequence[float],
    event_times: pd.Series | Sequence[Any],
    bundle: ContractBundle,
    *,
    execution_kind: str,
    budget: FitBudget,
    real_data_authorized: bool = False,
) -> tuple[SimpleRaceRanker, dict[str, Any]]:
    if execution_kind == SYNTHETIC_EXECUTION_KIND:
        _require(not real_data_authorized, "synthetic fixture cannot claim real-data authorization")
    else:
        raise ContractError(
            "this Prepare execution commit permits synthetic_fixture fits only; "
            "a caller-supplied flag can never authorize real-data execution"
        )
    validate_feature_frame(features, bundle)
    working_features = features.reset_index(drop=True)
    for column in IDENTITY_COLUMNS:
        _require(
            bool(working_features[column].astype(str).str.startswith(SYNTHETIC_ID_PREFIX).all()),
            f"Prepare-stage fit requires reserved synthetic identities: {column}",
        )
    target = pd.to_numeric(pd.Series(list(targets)), errors="coerce").astype("float64")
    _require(len(target) == len(working_features), "target length mismatch")
    _require(np.isfinite(target.to_numpy()).all(), "target is non-finite")
    times = [_parse_aware_timestamp(value, "fit event time") for value in event_times]
    _require(len(times) == len(features), "event time length mismatch")
    ordering = pd.DataFrame(
        {
            "position": list(range(len(working_features))),
            "event_time": times,
            "race_id": working_features["race_id"].astype(str).to_numpy(),
            "horse_id": working_features["horse_id"].astype(str).to_numpy(),
        }
    ).sort_values(["event_time", "race_id", "horse_id"], kind="mergesort")
    ordered_positions = ordering["position"].tolist()
    fit_frame = _model_frame(working_features.iloc[ordered_positions], bundle).reset_index(drop=True)
    fit_frame["__clean_target_score"] = target.iloc[ordered_positions].to_numpy()
    budget.consume_model_fit()
    np.random.seed(int(bundle.config["model"]["seed"]))
    model = SimpleRaceRanker(
        numeric_features=bundle.numeric_features,
        categorical_features=bundle.categorical_features,
        categorical_top_k=int(bundle.config["model"]["categorical_top_k"]),
        ridge_alpha=float(bundle.config["model"]["ridge_alpha"]),
    )
    model.fit(fit_frame, "__clean_target_score")
    schema = _frozen_schema(model, bundle)
    return model, schema


def validate_schema_parity(
    features: pd.DataFrame,
    model: SimpleRaceRanker,
    schema: Mapping[str, Any],
    bundle: ContractBundle,
) -> dict[str, Any]:
    validate_feature_frame(features, bundle)
    _require(schema.get("ordered_features") == bundle.ordered_features, "ordered feature schema changed")
    expected_dtypes = {
        **{name: "float64" for name in bundle.numeric_features},
        **{name: "string" for name in bundle.categorical_features},
    }
    _require(schema.get("dtypes") == expected_dtypes, "dtype schema changed")
    _require(schema.get("categorical_levels") == model.categorical_levels_, "category dictionary changed")
    _require(schema.get("design_feature_names") == model.feature_names_, "design matrix schema changed")
    _require(schema.get("numeric_medians") == model.numeric_medians_, "numeric median contract changed")
    _require(schema.get("numeric_means") == model.numeric_means_, "numeric mean contract changed")
    _require(schema.get("numeric_stds") == model.numeric_stds_, "numeric standard-deviation contract changed")
    _require(model.numeric_features == bundle.numeric_features, "model numeric features changed")
    _require(model.categorical_features == bundle.categorical_features, "model categorical features changed")
    _require(float(model.ridge_alpha) == 10.0, "model ridge_alpha changed")
    _require(int(model.categorical_top_k) == 80, "model categorical_top_k changed")
    _require(model.coefficients_ is not None, "model coefficients are absent")
    coefficient_hash = hashlib.sha256(
        np.asarray(model.coefficients_, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    _require(schema.get("coefficients_sha256") == coefficient_hash, "model coefficient hash changed")
    digest_payload = dict(schema)
    supplied_digest = digest_payload.pop("schema_digest", None)
    _require(supplied_digest == canonical_digest(digest_payload), "schema digest mismatch")
    return {
        "ordered_feature_parity_fraction": 1.0,
        "dtype_parity_fraction": 1.0,
        "category_dictionary_parity_fraction": 1.0,
        "unknown_category_policy": "all_zero_reference_encoding",
        "schema_digest": supplied_digest,
    }


def predict_clean_ranker(
    model: SimpleRaceRanker,
    features: pd.DataFrame,
    schema: Mapping[str, Any],
    bundle: ContractBundle,
) -> pd.DataFrame:
    validate_schema_parity(features, model, schema, bundle)
    scores = np.asarray(model.predict(_model_frame(features, bundle)), dtype=float)
    _require(np.isfinite(scores).all(), "non-finite clean score")
    ranked = features[IDENTITY_COLUMNS].copy()
    ranked["clean_ai_score"] = scores
    ranked = ranked.sort_values(
        ["race_id", "clean_ai_score", "horse_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranked["clean_baseline_rank"] = ranked.groupby("race_id", sort=False).cumcount() + 1
    for _, group in ranked.groupby("race_id", sort=False):
        _require(group["clean_baseline_rank"].tolist() == list(range(1, len(group) + 1)), "rank sequence failure")
    return ranked.sort_values(IDENTITY_COLUMNS, kind="mergesort").reset_index(drop=True)


def validate_target_universe(universe: pd.DataFrame) -> dict[str, Any]:
    required = {"race_id", "horse_id", "draw_status", "entry_stage", "枠番", "馬番"}
    missing = sorted(required - set(universe.columns))
    _require(not missing, f"target universe missing columns: {missing}")
    out = universe.copy()
    out["race_id"] = out["race_id"].map(_category)
    out["horse_id"] = out["horse_id"].map(_category)
    _require((out["race_id"] != "__MISSING__").all(), "target universe has blank race_id")
    _require((out["horse_id"] != "__MISSING__").all(), "target universe has blank horse_id")
    _require(len(out) == 70, "target universe must contain exactly 70 synthetic or authorized real runners")
    _require(out["race_id"].nunique() == 5, "target universe must contain exactly five races")
    _require(not out.duplicated(IDENTITY_COLUMNS).any(), "target universe has duplicate identity")
    allowed = {"confirmed", "scheduled_pending_draw", "scratched"}
    _require(set(out["draw_status"]).issubset(allowed), "unknown draw_status")
    pending = out["draw_status"] == "scheduled_pending_draw"
    _require((out.loc[pending, "entry_stage"] == "declared_without_draw").all(), "pending draw entry_stage mismatch")
    _require(out.loc[pending, "枠番"].map(_is_blank).all(), "pending draw has a provisional frame number")
    _require(out.loc[pending, "馬番"].map(_is_blank).all(), "pending draw has a provisional horse number")
    confirmed = out["draw_status"] == "confirmed"
    for column in ["枠番", "馬番"]:
        valid_confirmed_number = out.loc[confirmed, column].map(_is_positive_integer)
        _require(bool(valid_confirmed_number.all()), f"confirmed draw has invalid {column}")
    return {
        "race_count": 5,
        "runner_count": 70,
        "duplicate_count": 0,
        "scheduled_pending_draw_count": int(pending.sum()),
        "confirmed_count": int(confirmed.sum()),
        "scratched_count": int((out["draw_status"] == "scratched").sum()),
        "horse_id_join_only": True,
    }


@dataclass
class OuterSeal:
    maximum_opens: int = 1
    frozen_hashes: dict[str, str] = field(default_factory=dict)
    open_count: int = 0

    def freeze(self, *, model: str, config: str, calibrator: str, schema: str) -> None:
        hashes = {"model": model, "config": config, "calibrator": calibrator, "schema": schema}
        _require(all(HASH_RE.fullmatch(value or "") for value in hashes.values()), "freeze requires four SHA-256 hashes")
        _require(not self.frozen_hashes, "outer seal already frozen")
        self.frozen_hashes = hashes

    def open(self, loader: Callable[[], Any]) -> Any:
        _require(bool(self.frozen_hashes), "outer data cannot open before model/config/calibrator/schema freeze")
        _require(self.open_count < self.maximum_opens, "outer data open limit exceeded")
        self.open_count += 1
        return loader()


def _latest_registry_event(registry_path: Path, experiment_id: str) -> dict[str, Any]:
    events = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            event = json.loads(line, object_pairs_hook=_no_duplicate_object_pairs)
            if event.get("experiment_id") == experiment_id:
                events.append(event)
    _require(bool(events), "experiment is absent from registry")
    return max(events, key=lambda item: int(item["sequence"]))


def verify_real_data_authorization(run_scope_path: Path, registry_path: Path) -> tuple[dict[str, Any], str]:
    _require(run_scope_path.is_file(), "canonical run scope is missing")
    run_scope = load_json(run_scope_path)
    run_digest = canonical_digest(run_scope)
    latest = _latest_registry_event(registry_path, EXPERIMENT_ID)
    _require(latest.get("status") == "running", "real-data command requires RUNNING registry status")
    _require(latest.get("execution_kind") == "real-data", "registry execution_kind is not real-data")
    _require(latest.get("execution_authorized") is True, "registry does not authorize execution")
    _require(latest.get("real_data_execution_allowed") is True, "registry forbids real-data execution")
    _require(latest.get("run_scope_digest") == run_digest, "registry/run-scope digest mismatch")
    _require(latest.get("human_prepare_approval_recorded") is True, "prepare approval is absent")
    _require(latest.get("human_run_approval_recorded") is True, "run approval is absent")
    for source in [run_scope, latest]:
        _require(source.get("formal_buy") is False, "formal_buy changed")
        _require(source.get("send_order") is False, "send_order changed")
        _require(source.get("stake") == 0, "stake changed")
    return run_scope, run_digest


def _real_command_fail_closed(args: argparse.Namespace, bundle: ContractBundle) -> None:
    run_scope_path = Path(args.run_scope)
    registry_path = Path(args.registry)
    if not run_scope_path.is_absolute():
        run_scope_path = _repo_path(str(run_scope_path))
    if not registry_path.is_absolute():
        registry_path = _repo_path(str(registry_path))
    verify_real_data_authorization(run_scope_path, registry_path)
    raise ContractError(
        "authorization was valid, but the source-time-bound real input release contract is not yet canonicalized; "
        "this execution commit remains fail-closed until the separately approved input manifests exist"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research-only leak-free predraw baseline contract")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-contract")
    validate.add_argument("--config", required=True)
    for command in ["fit-freeze", "evaluate-predict"]:
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument("--run-scope", required=True)
        child.add_argument("--registry", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    bundle = load_and_verify_contract(args.config)
    if args.command == "validate-contract":
        result = {
            "experiment_id": EXPERIMENT_ID,
            "variant": VARIANT,
            "proposal_scope_digest": PROPOSAL_DIGEST,
            "feature_count": len(bundle.ordered_features),
            "numeric_count": len(bundle.numeric_features),
            "categorical_count": len(bundle.categorical_features),
            "leakage_roots_allowed": sorted(DIRECT_LEAKAGE_ROOTS & set(bundle.ordered_features)),
            "formal_buy": False,
            "send_order": False,
            "stake": 0,
            "status": "PREPARE_CONTRACT_VALID",
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    _real_command_fail_closed(args, bundle)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
