from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from lxml import html


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


def parse_html_race_metadata(
    path: Path,
    runners: pd.DataFrame,
    *,
    minimum_name_match_fraction: float = 0.60,
) -> dict[tuple[str, int], RaceMetadata]:
    source = path.read_bytes().decode("cp932", errors="replace")
    document = html.fromstring(source)
    text = _normalized_text(document.text_content())
    compact_text = re.sub(r"\s+", "", text)
    all_headings = list(
        re.finditer(r"(?:札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)\s*\d{1,2}\s*R", text)
    )
    metadata: dict[tuple[str, int], RaceMetadata] = {}
    for (venue, race_no), group in runners.groupby(["venue", "race_no"], sort=True):
        names = [re.sub(r"\s+", "", str(value)) for value in group["horse_name"]]
        heading = re.compile(rf"{re.escape(str(venue))}\s*{int(race_no)}\s*R", re.IGNORECASE)
        candidates: list[tuple[int, str, int]] = []
        for match in heading.finditer(text):
            next_positions = [item.start() for item in all_headings if item.start() > match.start()]
            end = min(next_positions) if next_positions else min(len(text), match.start() + 30000)
            window = text[match.start() : end]
            compact_window = re.sub(r"\s+", "", window)
            score = sum(bool(name and name in compact_window) for name in names)
            candidates.append((score, window, match.start()))
        if not candidates:
            # TARGET exports sometimes omit the R suffix but retain the 16-digit key.
            key = str(group.iloc[0]["target_race_key"])
            for match in re.finditer(re.escape(key), compact_text):
                start = max(0, match.start() - 1000)
                window = compact_text[start : match.start() + 30000]
                score = sum(bool(name and name in window) for name in names)
                candidates.append((score, window, start))
        if not candidates:
            raise ValueError(f"HTML race heading not found: {venue} {race_no}R")
        score, window, _position = max(candidates, key=lambda item: (item[0], -item[2]))
        minimum_matches = max(1, math.ceil(len(names) * minimum_name_match_fraction))
        if score < minimum_matches:
            raise ValueError(
                f"HTML/DU runner identity mismatch for {venue} {race_no}R: {score}/{len(names)}"
            )
        header_window = window[:5000]
        surface, distance = _surface_and_distance(header_window)
        race_class = _race_class(header_window)
        title = re.sub(r"\s+", " ", header_window[:300]).strip()
        metadata[(str(venue), int(race_no))] = RaceMetadata(
            venue=str(venue),
            race_no=int(race_no),
            surface=surface,
            distance=distance,
            race_class=race_class,
            race_name=title,
            runner_name_match_count=score,
        )
    return metadata


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
    metadata: dict[tuple[str, int], RaceMetadata],
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
        race_meta = metadata[(str(row["venue"]), int(row["race_no"]))]
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
                "トラックコード": race_meta.surface,
                "レースID(新/馬番無)": target["race_id"],
                "休み明け～戦目": "",
                "出走頭数": int(target["runner_count"]),
                "場所": row["venue"],
                "年齢": row["age"] or "",
                "性別": _sex_from_history(str(row["horse_id"]), historical),
                "斤量": row["assigned_weight_kg"] if row["assigned_weight_kg"] is not None else "",
                "日付": str(row["race_date"])[2:],
                "枠番": row["frame_no"],
                "競走種別": "",
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
                "レース名": race_meta.race_name,
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
    for name, path in (("html", html_path), ("du", du_path)):
        expected = str(sources[name]["sha256"]).lower()
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(f"{name} source hash mismatch: {observed} != {expected}")
    du = parse_du(du_path)
    expected_total = int(config["input_contract"]["expected_runner_rows"])
    if len(du) != expected_total:
        raise ValueError(f"DU active runner count mismatch: {len(du)} != {expected_total}")
    metadata = parse_html_race_metadata(html_path, du)
    historical_path = Path(config["history"]["historical_csv"])
    historical = pd.read_csv(historical_path, encoding="cp932", low_memory=False)
    cards: list[dict[str, Any]] = []
    for card in config["cards"]:
        manifest_path = ROOT / card["target_manifest"]
        manifest = load_json(manifest_path)
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
        "runner_rows": int(len(du)),
        "race_count": int(du["target_race_key"].nunique()),
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
