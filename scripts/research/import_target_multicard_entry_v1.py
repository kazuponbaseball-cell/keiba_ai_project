from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VENUES = {
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
}
FORBIDDEN_CURRENT_FIELDS = {
    "人気",
    "単勝オッズ",
    "確定着順",
    "current_odds",
    "current_popularity",
    "market_rank",
    "final_odds",
    "payout",
    "roi",
}
ODDS_PREFIX = "odds_"
TRACK_SURFACE = {
    **{f"{value:02d}": "芝" for value in range(10, 20)},
    **{f"{value:02d}": "ダ" for value in range(20, 30)},
    **{f"{value:02d}": "障害" for value in range(51, 60)},
}
RA_MINIMUM_LENGTH = 883
RA_RACE_KEY_OFFSET = 11
RA_RACE_KEY_LENGTH = 16


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _decode(raw: bytes, start: int, end: int) -> str:
    return raw[start:end].decode("cp932", errors="replace").replace("\u3000", "").strip()


def _number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical_candidate_race_id(target_race_key: str) -> str:
    digits = re.sub(r"\D", "", target_race_key)
    if len(digits) != 16:
        raise ValueError(f"invalid TARGET race key: {target_race_key}")
    return digits[:4] + digits[8:]


def parse_du(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in path.read_bytes().splitlines():
        if len(raw) < 132:
            continue
        record_type = _decode(raw, 0, 3)
        race_date = _decode(raw, 11, 19)
        race_part = _decode(raw, 19, 27)
        horse_no = _decode(raw, 28, 30)
        if len(race_part) != 8 or len(race_date) != 8:
            continue
        if record_type not in {"SE1", "SE2"} or horse_no in {"", "00"}:
            continue
        venue_code = race_part[:2]
        horse_id_raw = _decode(raw, 32, 40)
        weight_raw = _number(_decode(raw, 104, 109))
        rows.append(
            {
                "target_race_key": f"{race_date}{race_part}",
                "race_id": canonical_candidate_race_id(f"{race_date}{race_part}"),
                "race_date": race_date,
                "venue_code": venue_code,
                "venue": VENUES.get(venue_code, venue_code),
                "meeting_no": int(race_part[2:4]),
                "day_no": int(race_part[4:6]),
                "race_no": int(race_part[6:8]),
                "frame_no": int(_decode(raw, 27, 28) or 0),
                "horse_no": int(horse_no),
                "horse_id": "20" + horse_id_raw if len(horse_id_raw) == 8 else horse_id_raw,
                "horse_name": _decode(raw, 40, 76),
                "age": int(_decode(raw, 82, 84) or 0),
                "trainer_code": _decode(raw, 86, 90),
                "trainer_name": _decode(raw, 90, 100),
                "assigned_weight_kg": weight_raw / 1000.0 if weight_raw is not None else None,
                "jockey_code": _decode(raw, 112, 117),
                "jockey_name": _decode(raw, 122, 132),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("DU source contains no active runner records")
    if frame.duplicated(["target_race_key", "horse_no"]).any():
        raise ValueError("DU source contains duplicate active runner identities")
    return frame.sort_values(["target_race_key", "horse_no"], kind="mergesort")


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", html_module.unescape(value)))


@dataclass(frozen=True)
class RaceMetadata:
    venue: str
    race_no: int
    surface: str
    distance: int | None
    race_class: str
    race_name: str
    runner_name_match_count: int
    target_race_key: str = ""
    html_group_index: int = 0
    data_category: str = ""
    race_type_code: str = ""
    grade_code: str = ""
    condition_code: str = ""
    track_code: str = ""
    course_code: str = ""
    scheduled_post_hhmm: str = ""
    registered_runner_count: int = 0
    record_hash: str = ""


def _surface_and_distance(text: str) -> tuple[str, int | None]:
    patterns = (
        r"(?P<surface>芝|ダート|ダ|障害)\s*[・･]?\s*(?P<distance>\d{3,4})\s*[mM]",
        r"(?P<distance>\d{3,4})\s*[mM]\s*[・･]?\s*(?P<surface>芝|ダート|ダ|障害)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            raw_surface = match.group("surface")
            surface = "ダ" if raw_surface in {"ダ", "ダート"} else raw_surface
            return surface, int(match.group("distance"))
    return "", None


def _race_class(text: str) -> str:
    patterns = (
        r"(新馬|未勝利|1勝クラス|2勝クラス|3勝クラス|オープン|リステッド|Listed|L|GIII|GII|GI|G3|G2|G1)",
        r"(障害[^ ]{0,20})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _normalized_horse_name(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", "", _normalized_text(text))


def parse_html_runner_groups(
    path: Path,
    *,
    expected_runner_rows: int,
    expected_races: int,
) -> pd.DataFrame:
    source = path.read_bytes().decode("cp932", errors="replace")
    block_pattern = re.compile(
        r"<HR>\s*<TD\s+NOWRAP\s*>\s*(?P<frame>\d+)\s*枠\s*"
        r"(?P<horse_no>\d+)\s*番(?P<body>.*?)(?=<HR>\s*<TD\s+NOWRAP|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    rows: list[dict[str, Any]] = []
    group_index = 1
    previous_horse_no: int | None = None
    for match in block_pattern.finditer(source):
        horse_no = int(match.group("horse_no"))
        if previous_horse_no is not None and horse_no <= previous_horse_no:
            group_index += 1
        name_match = re.search(r"<B\b[^>]*>(?P<name>.*?)</B>", match.group("body"), re.I | re.S)
        if name_match is None:
            raise ValueError(f"HTML horse name missing at runner block {len(rows) + 1}")
        horse_name = _normalized_horse_name(name_match.group("name"))
        if not horse_name:
            raise ValueError(f"HTML horse name empty at runner block {len(rows) + 1}")
        rows.append(
            {
                "html_group_index": group_index,
                "frame_no": int(match.group("frame")),
                "horse_no": horse_no,
                "horse_name": horse_name,
            }
        )
        previous_horse_no = horse_no
    frame = pd.DataFrame(rows)
    if len(frame) != expected_runner_rows:
        raise ValueError(f"HTML runner block count mismatch: {len(frame)} != {expected_runner_rows}")
    if frame["html_group_index"].nunique() != expected_races:
        raise ValueError(
            "HTML runner group count mismatch: "
            f"{frame['html_group_index'].nunique()} != {expected_races}"
        )
    if frame.duplicated(["html_group_index", "horse_no"]).any():
        raise ValueError("HTML runner group contains duplicate horse number")
    for group_id, group in frame.groupby("html_group_index", sort=True):
        numbers = group["horse_no"].tolist()
        if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
            raise ValueError(f"HTML runner order is not strictly increasing in group {group_id}")
    return frame


def _runner_signature(frame: pd.DataFrame) -> tuple[tuple[int, str], ...]:
    return tuple(
        sorted(
            (int(row.horse_no), _normalized_horse_name(str(row.horse_name)))
            for row in frame[["horse_no", "horse_name"]].itertuples(index=False)
        )
    )


def match_html_runner_groups_to_du(
    html_groups: pd.DataFrame,
    du: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    html_by_signature: dict[tuple[tuple[int, str], ...], list[int]] = {}
    for group_id, group in html_groups.groupby("html_group_index", sort=True):
        html_by_signature.setdefault(_runner_signature(group), []).append(int(group_id))
    matches: dict[str, dict[str, int]] = {}
    used_groups: set[int] = set()
    for target_race_key, race in du.groupby("target_race_key", sort=True):
        signature = _runner_signature(race)
        candidates = html_by_signature.get(signature, [])
        if len(candidates) != 1:
            raise ValueError(
                f"HTML/DU runner identity mismatch for {target_race_key}: "
                f"candidate_groups={candidates}"
            )
        group_id = candidates[0]
        if group_id in used_groups:
            raise ValueError(f"HTML runner group reused: {group_id}")
        used_groups.add(group_id)
        matches[str(target_race_key)] = {
            "html_group_index": group_id,
            "runner_name_match_count": len(signature),
        }
    expected_groups = set(html_groups["html_group_index"].astype(int).unique())
    if used_groups != expected_groups:
        raise ValueError("HTML runner groups include an unmatched group")
    return matches


def _ascii_field(raw: bytes, start: int, length: int) -> str:
    return raw[start : start + length].decode("ascii", errors="ignore").strip()


def _text_field(raw: bytes, start: int, length: int) -> str:
    return _normalized_text(raw[start : start + length].decode("cp932", errors="replace")).strip()


def _race_class_from_ra(grade_code: str, condition_code: str) -> str:
    grade_map = {"A": "Ｇ１", "B": "Ｇ２", "C": "Ｇ３", "L": "OP(L)"}
    if grade_code in grade_map:
        return grade_map[grade_code]
    return {
        "701": "新馬",
        "703": "未勝利",
        "005": "1勝",
        "010": "2勝",
        "016": "3勝",
        "999": "ｵｰﾌﾟﾝ",
    }.get(condition_code, "")


def parse_ra_race_metadata(
    path: Path,
    *,
    expected_races: int,
) -> dict[str, RaceMetadata]:
    metadata: dict[str, RaceMetadata] = {}
    for line_no, raw in enumerate(path.read_bytes().splitlines(), start=1):
        if len(raw) < RA_MINIMUM_LENGTH or not raw.startswith(b"RA"):
            continue
        data_category = _ascii_field(raw, 2, 1)
        target_race_key = _ascii_field(raw, RA_RACE_KEY_OFFSET, RA_RACE_KEY_LENGTH)
        if len(target_race_key) != 16 or not target_race_key.isdigit():
            raise ValueError(f"invalid RA race key at line {line_no}")
        if target_race_key in metadata:
            raise ValueError(f"duplicate RA race key: {target_race_key}")
        venue_code = target_race_key[8:10]
        track_code = _ascii_field(raw, 705, 2)
        surface = TRACK_SURFACE.get(track_code, "")
        distance_text = _ascii_field(raw, 697, 4)
        runner_count_text = _ascii_field(raw, 881, 2)
        grade_code = _ascii_field(raw, 614, 1)
        condition_code = _ascii_field(raw, 634, 3)
        race_name = _text_field(raw, 32, 60) or _text_field(raw, 572, 20)
        metadata[target_race_key] = RaceMetadata(
            venue=VENUES.get(venue_code, venue_code),
            race_no=int(target_race_key[-2:]),
            surface=surface,
            distance=int(distance_text) if distance_text.isdigit() else None,
            race_class=_race_class_from_ra(grade_code, condition_code),
            race_name=race_name,
            runner_name_match_count=0,
            target_race_key=target_race_key,
            data_category=data_category,
            race_type_code=_ascii_field(raw, 616, 2),
            grade_code=grade_code,
            condition_code=condition_code,
            track_code=track_code,
            course_code=_ascii_field(raw, 709, 2),
            scheduled_post_hhmm=_ascii_field(raw, 873, 4),
            registered_runner_count=(
                int(runner_count_text) if runner_count_text.isdigit() else 0
            ),
            record_hash=hashlib.sha256(raw).hexdigest(),
        )
    if len(metadata) != expected_races:
        raise ValueError(f"RA race count mismatch: {len(metadata)} != {expected_races}")
    return metadata


def bind_fixed_input_metadata(
    *,
    du: pd.DataFrame,
    html_matches: dict[str, dict[str, int]],
    ra_metadata: dict[str, RaceMetadata],
    target_records: list[dict[str, Any]],
) -> dict[str, RaceMetadata]:
    target_by_key = {str(record["target_race_key"]): record for record in target_records}
    du_keys = set(du["target_race_key"].astype(str).unique())
    expected_keys = set(target_by_key)
    if du_keys != expected_keys or set(html_matches) != expected_keys or set(ra_metadata) != expected_keys:
        raise ValueError("HTM/DU/RA/manifest race identity mismatch")
    bound: dict[str, RaceMetadata] = {}
    for key in sorted(expected_keys):
        target = target_by_key[key]
        meta = ra_metadata[key]
        du_count = int(du["target_race_key"].eq(key).sum())
        expected_count = int(target["runner_count"])
        if meta.data_category != "2":
            raise ValueError(f"RA record is not pre-race data category 2: {key}")
        if du_count != expected_count or meta.registered_runner_count != expected_count:
            raise ValueError(
                f"RA/DU/manifest runner count mismatch for {key}: "
                f"{meta.registered_runner_count}/{du_count}/{expected_count}"
            )
        scheduled_hhmm = datetime.fromisoformat(str(target["scheduled_post_time"])).strftime("%H%M")
        if meta.scheduled_post_hhmm != scheduled_hhmm:
            raise ValueError(
                f"RA/manifest post time mismatch for {key}: "
                f"{meta.scheduled_post_hhmm} != {scheduled_hhmm}"
            )
        match = html_matches[key]
        if int(match["runner_name_match_count"]) != expected_count:
            raise ValueError(f"HTML/manifest runner count mismatch for {key}")
        bound[key] = replace(
            meta,
            runner_name_match_count=int(match["runner_name_match_count"]),
            html_group_index=int(match["html_group_index"]),
        )
    return bound


def _baseline_track_code(meta: RaceMetadata) -> str:
    if meta.surface == "ダ":
        return "1"
    if meta.surface != "芝":
        return ""
    # Today's only explicit outer-course code is Niigata turf outer (12).
    return "8" if meta.track_code == "12" else "0"


def _sex_from_history(horse_id: str, history: pd.DataFrame | None) -> str:
    if history is None or history.empty or not horse_id:
        return ""
    id_column = next((column for column in ("血統登録番号", "horse_id") if column in history), None)
    sex_column = next((column for column in ("性別", "sex") if column in history), None)
    if id_column is None or sex_column is None:
        return ""
    ids = history[id_column].astype(str).str.replace(r"\.0$", "", regex=True)
    values = history.loc[ids.eq(str(horse_id)), sex_column].dropna().astype(str)
    return values.iloc[-1].strip() if len(values) else ""


def build_entry_rows(
    du: pd.DataFrame,
    target_manifest: dict[str, Any],
    metadata: dict[str, RaceMetadata],
    *,
    historical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    records = target_manifest.get("records")
    if not isinstance(records, list) or len(records) != 12:
        raise ValueError("target manifest must contain exactly 12 records")
    manifest_by_key = {str(record["target_race_key"]): record for record in records}
    if "target_race_key" not in du.columns:
        raise ValueError("runner count mismatch: DU runner identity is absent")
    selected = du[du["target_race_key"].isin(manifest_by_key)].copy()
    expected_count = sum(int(record["runner_count"]) for record in records)
    if len(selected) != expected_count:
        raise ValueError(f"runner count mismatch: {len(selected)} != {expected_count}")
    output: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        target = manifest_by_key[str(row["target_race_key"])]
        race_meta = metadata[str(row["target_race_key"])]
        race_domain = (
            "obstacle"
            if race_meta.surface == "障害" or str(target.get("race_domain")) == "obstacle"
            else "flat_turf"
            if race_meta.surface == "芝"
            else "flat_dirt"
            if race_meta.surface == "ダ"
            else "unknown"
        )
        output.append(
            {
                "キャリア": "",
                "クラス名": race_meta.race_class,
                "トラックコード": _baseline_track_code(race_meta),
                "レースID(新/馬番無)": target["race_id"],
                "休み明け～戦目": "",
                "出走頭数": int(target["runner_count"]),
                "場所": row["venue"],
                "年齢": row["age"] or "",
                "性別": _sex_from_history(str(row["horse_id"]), historical),
                "斤量": row["assigned_weight_kg"] if row["assigned_weight_kg"] is not None else "",
                "日付": str(row["race_date"])[2:],
                "枠番": row["frame_no"],
                "競走種別": race_meta.race_type_code,
                "芝・ダ": race_meta.surface,
                "血統登録番号": row["horse_id"],
                "調教師コード": row["trainer_code"],
                "距離": race_meta.distance or "",
                "間隔": "",
                "頭数": int(target["runner_count"]),
                "馬名": row["horse_name"],
                "馬場状態": "",
                "馬番": row["horse_no"],
                "騎手コード": row["jockey_code"],
                "レース名": race_meta.race_name or race_meta.race_class,
                "人気": "",
                "単勝オッズ": "",
                "日付S": str(row["race_date"]),
                "異常コード": "0",
                "発走時刻": str(target["scheduled_post_time"]),
                "確定着順": "",
                "騎手": row["jockey_name"],
                "Ｒ": row["race_no"],
                "race_id": target["race_id"],
                "race_no": row["race_no"],
                "horse_no": row["horse_no"],
                "horse_id": row["horse_id"],
                "horse_name": row["horse_name"],
                "race_domain": race_domain,
                "target_race_key": row["target_race_key"],
                "jv_track_code": race_meta.track_code,
                "course_code": race_meta.course_code,
                "grade_code": race_meta.grade_code,
                "condition_code": race_meta.condition_code,
                "html_runner_group_index": race_meta.html_group_index,
                "html_identity_match_count": race_meta.runner_name_match_count,
                "dr_data_kbn": race_meta.data_category,
                "dr_record_hash": race_meta.record_hash,
                "source_url": "",
            }
        )
    frame = pd.DataFrame(output).sort_values(["race_no", "horse_no"], kind="mergesort")
    for column in frame.columns:
        if column in FORBIDDEN_CURRENT_FIELDS or str(column).lower().startswith(ODDS_PREFIX):
            frame[column] = ""
    return frame


def import_multicard(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_json(config_path)
    safety = config.get("safety", {})
    if safety.get("formal_buy") is not False or safety.get("send_order") is not False:
        raise ValueError("unsafe card configuration")
    if int(safety.get("stake", -1)) != 0:
        raise ValueError("stake must be zero")
    sources = config["input_sources"]
    html_path = Path(sources["html"]["path"])
    du_path = Path(sources["du"]["path"])
    dr_path = Path(sources["dr"]["path"])
    for name, path in (("html", html_path), ("du", du_path), ("dr", dr_path)):
        expected = str(sources[name]["sha256"]).lower()
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(f"{name} source hash mismatch: {observed} != {expected}")
    du = parse_du(du_path)
    expected_total = int(config["input_contract"]["expected_runner_rows"])
    expected_races = int(config["input_contract"]["expected_races"])
    if len(du) != expected_total:
        raise ValueError(f"DU active runner count mismatch: {len(du)} != {expected_total}")
    html_groups = parse_html_runner_groups(
        html_path,
        expected_runner_rows=expected_total,
        expected_races=expected_races,
    )
    html_matches = match_html_runner_groups_to_du(html_groups, du)
    ra_metadata = parse_ra_race_metadata(dr_path, expected_races=expected_races)
    manifests: dict[str, dict[str, Any]] = {}
    target_records: list[dict[str, Any]] = []
    for card in config["cards"]:
        manifest = load_json(ROOT / card["target_manifest"])
        manifests[str(card["slug"])] = manifest
        target_records.extend(manifest.get("records", []))
    if len(target_records) != expected_races:
        raise ValueError(f"target manifest race count mismatch: {len(target_records)} != {expected_races}")
    metadata = bind_fixed_input_metadata(
        du=du,
        html_matches=html_matches,
        ra_metadata=ra_metadata,
        target_records=target_records,
    )
    historical_path = Path(config["history"]["historical_csv"])
    historical = pd.read_csv(historical_path, encoding="cp932", low_memory=False)
    cards: list[dict[str, Any]] = []
    for card in config["cards"]:
        manifest = manifests[str(card["slug"])]
        frame = build_entry_rows(du, manifest, metadata, historical=historical)
        card_dir = output_root / str(card["slug"]) / "raw_entry"
        card_dir.mkdir(parents=True, exist_ok=True)
        output_csv = card_dir / "entry_snapshot_20260808.csv"
        frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
        cards.append(
            {
                "slug": card["slug"],
                "rows": int(len(frame)),
                "races": int(frame["race_id"].nunique()),
                "output_csv": str(output_csv),
                "output_sha256": file_sha256(output_csv),
                "unknown_domain_races": int(
                    frame.loc[frame["race_domain"].eq("unknown"), "race_id"].nunique()
                ),
            }
        )
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "html_sha256": file_sha256(html_path),
        "du_sha256": file_sha256(du_path),
        "dr_sha256": file_sha256(dr_path),
        "runner_rows": int(len(du)),
        "race_count": int(du["target_race_key"].nunique()),
        "html_runner_groups": int(html_groups["html_group_index"].nunique()),
        "html_du_identity_matches": int(len(html_matches)),
        "ra_race_rows": int(len(ra_metadata)),
        "cards": cards,
        "candidate_uses_odds": False,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    summary_path = output_root / "entry_import_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a fixed TARGET multi-card HTM/DU pair.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = import_multicard(args.config.resolve(), args.output_root.resolve())
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
