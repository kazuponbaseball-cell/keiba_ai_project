from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PREVIOUS_RACE_COLUMNS = (
    "前走レースID(新/馬番無)",
    "前走走破タイム",
    "前走平均1Fタイム",
    "前走基準タイム",
    "前走着差タイム",
    "前走人気",
    "前走頭数",
    "前走出走頭数",
    "前走馬番",
    "前走斤量",
    "前芝・ダ",
    "前距離",
    "前走馬場状態",
    "前走上3F地点差",
    "前走Ave-3F",
    "前走上り3F",
    "前走上り3F順",
    "前PCI",
    "前走PCI3",
    "前走RPCI",
    "前走馬体重",
    "前走馬体重増減",
    "前走騎手コード",
    "前走トラックコード",
)

TRACK_SURFACE = {
    **{f"{code:02d}": "芝" for code in range(10, 20)},
    **{f"{code:02d}": "ダ" for code in range(20, 30)},
    **{f"{code:02d}": "障害" for code in range(51, 60)},
}
BASELINE_TRACK_CODE = {
    **{f"{code:02d}": "0" for code in (10, 11, 13, 14, 15, 16, 17, 19)},
    **{f"{code:02d}": "8" for code in (12, 18)},
    **{f"{code:02d}": "1" for code in range(20, 30)},
    "52": "3",
    "54": "2",
    "56": "2",
}
APPRENTICE_SYMBOL = {
    "0": "",
    "1": "☆",
    "2": "△",
    "3": "▲",
    "4": "★",
    "9": "◇",
}

RA_RECORD_BYTES = 1270
SE_RECORD_BYTES = 553


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ascii(raw: bytes, start: int, length: int) -> str:
    return raw[start : start + length].decode("ascii", errors="ignore").strip()


def _text(raw: bytes, start: int, length: int) -> str:
    value = raw[start : start + length].decode("cp932", errors="replace")
    return unicodedata.normalize("NFKC", value).replace("\u3000", " ").strip()


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def canonical_horse_id(value: str) -> str:
    digits = _digits(value)
    if len(digits) == 8:
        return "20" + digits
    return digits


def canonical_race_id(target_race_key: str) -> str:
    digits = _digits(target_race_key)
    if len(digits) != 16:
        raise ValueError(f"invalid TARGET race key: {target_race_key!r}")
    return digits[:4] + digits[8:]


def race_date(target_race_key: str) -> str:
    digits = _digits(target_race_key)
    if len(digits) != 16:
        raise ValueError(f"invalid TARGET race key: {target_race_key!r}")
    datetime.strptime(digits[:8], "%Y%m%d")
    return digits[:8]


def _optional_int(value: str, *, zero_is_missing: bool = False) -> int | None:
    text = value.strip()
    if not text or not text.isdigit():
        return None
    parsed = int(text)
    if zero_is_missing and parsed == 0:
        return None
    return parsed


def _tenths(value: str) -> float | None:
    text = value.strip()
    if not text or not re.fullmatch(r"[+-]?\d+", text):
        return None
    return int(text) / 10.0


def finish_time_seconds(value: str) -> float | None:
    text = value.strip()
    if not re.fullmatch(r"\d{4}", text) or text == "0000":
        return None
    minutes = int(text[0])
    seconds = int(text[1:3])
    tenths = int(text[3])
    if seconds >= 60:
        raise ValueError(f"invalid JV finish time: {value!r}")
    return minutes * 60.0 + seconds + tenths / 10.0


def round_half_up(value: float, digits: int) -> float:
    if not math.isfinite(value):
        raise ValueError("cannot round a non-finite value")
    quantum = Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def average_section_time(total_seconds: float | None, distance_m: int, section_m: int) -> float | None:
    if total_seconds is None or distance_m <= 0 or section_m <= 0:
        return None
    return round_half_up(total_seconds * section_m / distance_m, 2)


def pace_change_index(
    total_seconds: float | None,
    final3f_seconds: float | None,
    distance_m: int,
) -> float | None:
    if (
        total_seconds is None
        or final3f_seconds is None
        or final3f_seconds <= 0
        or distance_m <= 600
        or total_seconds <= final3f_seconds
    ):
        return None
    pre_final3f_600 = (total_seconds - final3f_seconds) * 600.0 / (distance_m - 600)
    return round_half_up((pre_final3f_600 / final3f_seconds - 1.0) * 100.0 + 50.0, 1)


def pci3(pci_values: Sequence[float | None]) -> float | None:
    valid = [round_half_up(float(value), 1) for value in pci_values if value is not None]
    if len(valid) != 3:
        return None
    return round_half_up(sum(valid) / 3.0, 2)


def apprentice_symbol(code: str) -> str:
    normalized = code.strip() or "0"
    if normalized not in APPRENTICE_SYMBOL:
        raise ValueError(f"unsupported apprentice code: {code!r}")
    return APPRENTICE_SYMBOL[normalized]


@dataclass(frozen=True)
class RaceResult:
    target_race_key: str
    race_id: str
    distance_m: int
    track_code: str
    surface: str
    registered_runners: int
    starters: int
    going_code: str
    race_final3f_seconds: float | None
    lap_seconds: tuple[float, ...]
    source_file: str
    source_file_sha256: str
    record_sha256: str


@dataclass(frozen=True)
class RunnerResult:
    target_race_key: str
    race_id: str
    horse_no: int
    horse_id: str
    horse_name: str
    assigned_weight_kg: float | None
    apprentice_code: str
    apprentice_symbol: str
    jockey_code: str
    body_weight_kg: int | None
    body_weight_delta_kg: int | None
    finish_rank: int | None
    finish_time_seconds: float | None
    time_diff_seconds: float | None
    popularity: int | None
    final3f_seconds: float | None
    abnormal_code: str
    source_file: str
    source_file_sha256: str
    record_sha256: str


def parse_ra_record(
    raw: bytes,
    *,
    source_file: str = "synthetic-ra",
    source_file_sha256: str = "synthetic",
) -> RaceResult:
    if len(raw) < RA_RECORD_BYTES or not raw.startswith(b"RA"):
        raise ValueError("invalid RA record")
    key = _ascii(raw, 11, 16)
    if len(key) != 16 or not key.isdigit():
        raise ValueError("RA record has invalid race key")
    distance = _optional_int(_ascii(raw, 697, 4), zero_is_missing=True)
    registered = _optional_int(_ascii(raw, 881, 2), zero_is_missing=True)
    starters = _optional_int(_ascii(raw, 883, 2), zero_is_missing=True)
    if distance is None or registered is None or starters is None:
        raise ValueError(f"RA record lacks race dimensions: {key}")
    track_code = _ascii(raw, 705, 2)
    surface = TRACK_SURFACE.get(track_code, "")
    if not surface:
        raise ValueError(f"unsupported JV track code: {track_code!r}")
    turf_going = _ascii(raw, 888, 1)
    dirt_going = _ascii(raw, 889, 1)
    going = turf_going if surface == "芝" else dirt_going if surface == "ダ" else ""
    laps = tuple(
        value
        for value in (_tenths(_ascii(raw, 890 + index * 3, 3)) for index in range(25))
        if value is not None
    )
    return RaceResult(
        target_race_key=key,
        race_id=canonical_race_id(key),
        distance_m=distance,
        track_code=track_code,
        surface=surface,
        registered_runners=registered,
        starters=starters,
        going_code=going,
        race_final3f_seconds=_tenths(_ascii(raw, 975, 3)),
        lap_seconds=laps,
        source_file=source_file,
        source_file_sha256=source_file_sha256,
        record_sha256=hashlib.sha256(raw).hexdigest(),
    )


def parse_se_record(
    raw: bytes,
    *,
    source_file: str = "synthetic-se",
    source_file_sha256: str = "synthetic",
) -> RunnerResult:
    if len(raw) < SE_RECORD_BYTES or not raw.startswith(b"SE"):
        raise ValueError("invalid SE record")
    key = _ascii(raw, 11, 16)
    if len(key) != 16 or not key.isdigit():
        raise ValueError("SE record has invalid race key")
    horse_no = _optional_int(_ascii(raw, 28, 2), zero_is_missing=True)
    horse_id = canonical_horse_id(_ascii(raw, 30, 10))
    if horse_no is None or len(horse_id) != 10:
        raise ValueError(f"SE record lacks runner identity: {key}")
    weight_tenths = _optional_int(_ascii(raw, 288, 3), zero_is_missing=True)
    body_weight = _optional_int(_ascii(raw, 324, 3), zero_is_missing=True)
    delta_abs = _optional_int(_ascii(raw, 328, 3))
    delta_sign = _ascii(raw, 327, 1)
    if delta_abs is None:
        body_delta = None
    elif delta_sign == "-":
        body_delta = -delta_abs
    elif delta_sign in {"+", "", "0"}:
        body_delta = delta_abs
    else:
        raise ValueError(f"unsupported body-weight delta sign: {delta_sign!r}")
    apprentice_code = _ascii(raw, 322, 1) or "0"
    return RunnerResult(
        target_race_key=key,
        race_id=canonical_race_id(key),
        horse_no=horse_no,
        horse_id=horse_id,
        horse_name=_text(raw, 40, 36),
        assigned_weight_kg=(weight_tenths / 10.0 if weight_tenths is not None else None),
        apprentice_code=apprentice_code,
        apprentice_symbol=apprentice_symbol(apprentice_code),
        jockey_code=_ascii(raw, 296, 5),
        body_weight_kg=body_weight,
        body_weight_delta_kg=body_delta,
        finish_rank=_optional_int(_ascii(raw, 334, 2), zero_is_missing=True),
        finish_time_seconds=finish_time_seconds(_ascii(raw, 338, 4)),
        time_diff_seconds=_tenths(_ascii(raw, 531, 4)),
        popularity=_optional_int(_ascii(raw, 363, 2), zero_is_missing=True),
        final3f_seconds=_tenths(_ascii(raw, 390, 3)),
        abnormal_code=_ascii(raw, 331, 1),
        source_file=source_file,
        source_file_sha256=source_file_sha256,
        record_sha256=hashlib.sha256(raw).hexdigest(),
    )


def read_ra_files(paths: Iterable[Path]) -> dict[str, RaceResult]:
    records: dict[str, RaceResult] = {}
    for path in sorted((Path(value) for value in paths), key=lambda value: str(value)):
        source_hash = file_sha256(path)
        for raw in path.read_bytes().splitlines():
            if not raw.startswith(b"RA"):
                continue
            record = parse_ra_record(
                raw,
                source_file=str(path),
                source_file_sha256=source_hash,
            )
            if record.target_race_key in records:
                raise ValueError(f"duplicate RA race key: {record.target_race_key}")
            records[record.target_race_key] = record
    return records


def read_se_files(paths: Iterable[Path]) -> list[RunnerResult]:
    records: list[RunnerResult] = []
    identities: set[tuple[str, int]] = set()
    for path in sorted((Path(value) for value in paths), key=lambda value: str(value)):
        source_hash = file_sha256(path)
        for raw in path.read_bytes().splitlines():
            if not raw.startswith(b"SE"):
                continue
            record = parse_se_record(
                raw,
                source_file=str(path),
                source_file_sha256=source_hash,
            )
            identity = (record.target_race_key, record.horse_no)
            if identity in identities:
                raise ValueError(f"duplicate SE runner identity: {identity}")
            identities.add(identity)
            records.append(record)
    return records


def _competition_rank(values: Mapping[str, float | None]) -> dict[str, int | None]:
    valid = sorted(value for value in values.values() if value is not None)
    return {
        key: (valid.index(value) + 1 if value is not None else None)
        for key, value in values.items()
    }


def build_race_payloads(
    race: RaceResult,
    runners: Sequence[RunnerResult],
) -> dict[str, dict[str, Any]]:
    if not runners:
        raise ValueError(f"race has no runner records: {race.target_race_key}")
    if any(runner.target_race_key != race.target_race_key for runner in runners):
        raise ValueError("runner/race key mismatch")
    horse_ids = [runner.horse_id for runner in runners]
    horse_nos = [runner.horse_no for runner in runners]
    if len(horse_ids) != len(set(horse_ids)) or len(horse_nos) != len(set(horse_nos)):
        raise ValueError(f"duplicate runner identity: {race.target_race_key}")

    flat = race.surface in {"芝", "ダ"}
    runner_pci = {
        runner.horse_id: (
            pace_change_index(
                runner.finish_time_seconds,
                runner.final3f_seconds,
                race.distance_m,
            )
            if flat
            else None
        )
        for runner in runners
    }
    top3 = sorted(
        (runner for runner in runners if runner.finish_rank in {1, 2, 3}),
        key=lambda runner: (int(runner.finish_rank or 99), runner.horse_no),
    )
    race_pci3 = pci3([runner_pci[runner.horse_id] for runner in top3]) if flat else None
    winner_times = [
        runner.finish_time_seconds
        for runner in runners
        if runner.finish_rank == 1 and runner.finish_time_seconds is not None
    ]
    winner_time = min(winner_times) if winner_times else None
    race_rpci = (
        pace_change_index(winner_time, race.race_final3f_seconds, race.distance_m)
        if flat
        else None
    )
    final3f_values = {runner.horse_id: runner.final3f_seconds for runner in runners}
    final3f_ranks = _competition_rank(final3f_values) if flat else {
        runner.horse_id: None for runner in runners
    }
    pre_final_times = {
        runner.horse_id: (
            runner.finish_time_seconds - runner.final3f_seconds
            if flat
            and runner.finish_time_seconds is not None
            and runner.final3f_seconds is not None
            else None
        )
        for runner in runners
    }
    valid_pre_final = [value for value in pre_final_times.values() if value is not None]
    leader_pre_final = min(valid_pre_final) if valid_pre_final else None

    output: dict[str, dict[str, Any]] = {}
    for runner in runners:
        pre_final = pre_final_times[runner.horse_id]
        time_gap = runner.time_diff_seconds
        if time_gap is None and winner_time is not None and runner.finish_time_seconds is not None:
            time_gap = round_half_up(runner.finish_time_seconds - winner_time, 1)
        values: dict[str, Any] = {
            "前走レースID(新/馬番無)": race.race_id,
            "前走走破タイム": runner.finish_time_seconds,
            "前走平均1Fタイム": average_section_time(
                runner.finish_time_seconds, race.distance_m, 200
            ),
            "前走基準タイム": "",
            "前走着差タイム": time_gap,
            "前走人気": runner.popularity,
            "前走頭数": race.registered_runners,
            "前走出走頭数": race.starters,
            "前走馬番": runner.horse_no,
            "前走斤量": runner.assigned_weight_kg,
            "前芝・ダ": race.surface,
            "前距離": race.distance_m,
            "前走馬場状態": race.going_code,
            "前走上3F地点差": (
                round_half_up(pre_final - leader_pre_final, 1)
                if pre_final is not None and leader_pre_final is not None
                else None
            ),
            "前走Ave-3F": average_section_time(
                runner.finish_time_seconds, race.distance_m, 600
            ) if flat else None,
            "前走上り3F": runner.final3f_seconds if flat else None,
            "前走上り3F順": final3f_ranks[runner.horse_id],
            "前PCI": runner_pci[runner.horse_id],
            "前走PCI3": race_pci3,
            "前走RPCI": race_rpci,
            "前走馬体重": runner.body_weight_kg,
            "前走馬体重増減": runner.body_weight_delta_kg,
            "前走騎手コード": runner.jockey_code,
            "前走トラックコード": BASELINE_TRACK_CODE.get(race.track_code, ""),
        }
        record_payload = {
            "race_record_sha256": race.record_sha256,
            "runner_record_sha256": runner.record_sha256,
            "values": values,
        }
        values.update(
            {
                "previous_race_source_date": race_date(race.target_race_key),
                "previous_race_source_record_hash": hashlib.sha256(
                    canonical_json(record_payload).encode("utf-8")
                ).hexdigest(),
                "previous_race_source_system": "JRA_VAN_JV_DATA_SU_SR",
                "previous_race_source_file_hash": runner.source_file_sha256,
                "previous_race_no_history_reason": "",
                "previous_race_contract_ok": True,
                "previous_pci_target_good_run_marker": None,
                "_history_horse_name": runner.horse_name,
                "_history_horse_id": runner.horse_id,
            }
        )
        output[runner.horse_id] = values
    return output


def build_all_history_payloads(
    races: Mapping[str, RaceResult],
    runners: Sequence[RunnerResult],
) -> dict[tuple[str, str], dict[str, Any]]:
    by_race: dict[str, list[RunnerResult]] = {}
    for runner in runners:
        by_race.setdefault(runner.target_race_key, []).append(runner)
    unknown = sorted(set(by_race).difference(races))
    if unknown:
        raise ValueError(f"SE records have no matching RA record: {unknown[:3]}")
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(by_race):
        for horse_id, payload in build_race_payloads(races[key], by_race[key]).items():
            output[(key, horse_id)] = payload
    return output


def _blank_selection(
    *,
    current: Mapping[str, Any],
    reason: str,
    contract_ok: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {column: "" for column in PREVIOUS_RACE_COLUMNS}
    payload.update(
        {
            "target_race_key": str(current["target_race_key"]),
            "horse_no": int(current["horse_no"]),
            "horse_id": canonical_horse_id(str(current["horse_id"])),
            "previous_race_source_date": "",
            "previous_race_source_record_hash": hashlib.sha256(
                canonical_json(
                    {
                        "target_race_key": str(current["target_race_key"]),
                        "horse_no": int(current["horse_no"]),
                        "horse_id": canonical_horse_id(str(current["horse_id"])),
                        "reason": reason,
                    }
                ).encode("utf-8")
            ).hexdigest(),
            "previous_race_source_system": "JRA_VAN_JV_DATA_SU_SR",
            "previous_race_source_file_hash": "",
            "previous_race_no_history_reason": reason,
            "previous_race_contract_ok": contract_ok,
            "previous_pci_target_good_run_marker": None,
            "_history_horse_name": "",
            "_history_horse_id": "",
        }
    )
    return payload


def select_authoritative_history(
    current_runners: Sequence[Mapping[str, Any]],
    history_payloads: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    target_date: str,
    authoritative_latest_race_id: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    datetime.strptime(target_date, "%Y%m%d")
    selected: list[dict[str, Any]] = []
    seen_current: set[tuple[str, int]] = set()
    for current in current_runners:
        current_key = str(current["target_race_key"])
        horse_no = int(current["horse_no"])
        identity = (current_key, horse_no)
        if identity in seen_current:
            raise ValueError(f"duplicate current runner identity: {identity}")
        seen_current.add(identity)
        horse_id = canonical_horse_id(str(current["horse_id"] or ""))
        if len(horse_id) != 10:
            raise ValueError(f"invalid current horse id: {current.get('horse_id')!r}")
        if horse_id not in authoritative_latest_race_id:
            selected.append(
                _blank_selection(
                    current=current,
                    reason="AUTHORITATIVE_LATEST_RACE_UNAVAILABLE",
                    contract_ok=False,
                )
            )
            continue
        expected_race_id = authoritative_latest_race_id[horse_id]
        if expected_race_id in {None, ""}:
            selected.append(
                _blank_selection(
                    current=current,
                    reason="NO_PRIOR_RACE_CONFIRMED",
                    contract_ok=True,
                )
            )
            continue
        expected = _digits(str(expected_race_id))
        candidates = [
            (key, payload)
            for (key, candidate_horse_id), payload in history_payloads.items()
            if candidate_horse_id == horse_id and canonical_race_id(key) == expected
        ]
        if len(candidates) != 1:
            selected.append(
                _blank_selection(
                    current=current,
                    reason="LATEST_PRIOR_RACE_NOT_IN_JRA_HISTORY",
                    contract_ok=False,
                )
            )
            continue
        source_key, source_payload = candidates[0]
        source_date = race_date(source_key)
        if source_date >= target_date:
            selected.append(
                _blank_selection(
                    current=current,
                    reason="SAME_DAY_OR_FUTURE_HISTORY",
                    contract_ok=False,
                )
            )
            continue
        output = dict(source_payload)
        output.update(
            {
                "target_race_key": current_key,
                "horse_no": horse_no,
                "horse_id": horse_id,
            }
        )
        selected.append(output)
    return selected
