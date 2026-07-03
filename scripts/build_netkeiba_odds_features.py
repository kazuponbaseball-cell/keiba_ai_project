from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.odds_timeline import clean_odds_series, build_odds_timeline_features, merge_odds_timeline_features
RACE_ID_CANDIDATES = [
    "\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)",
    "\u30ec\u30fc\u30b9ID",
]
HORSE_NO_CANDIDATES = ["\u99ac\u756a", "horse_number", "horse_no"]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def read_csv_safe(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def norm_race_id(value: object) -> str:
    raw = str(value or "").strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw


def compact_netkeiba_race_id(race_id: object) -> str:
    raw = norm_race_id(race_id)
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) >= 12 and digits.startswith("20"):
        venue = str(int(digits[4:6]))
        return venue + digits[2:4] + digits[6:12]
    return raw


def _find_date_col(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        normalized = str(column).replace(" ", "").replace("\u3000", "")
        if normalized in {"\u65e5\u4ed8", "\u958b\u50ac\u65e5"}:
            return column
    for column in frame.columns:
        values = frame[column].dropna().astype(str).head(20)
        if len(values) and values.str.match(r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}$").mean() >= 0.5:
            return column
    return None


def _find_required_col(frame: pd.DataFrame, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    normalized = {str(column).replace(" ", "").replace("\u3000", ""): column for column in frame.columns}
    for candidate in candidates:
        key = candidate.replace(" ", "").replace("\u3000", "")
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Entry file is missing {label} column. candidates={candidates}")


def _date_variants(value: str) -> set[str]:
    text = str(value).strip()
    variants = {text, text.replace("-", "."), text.replace("/", "."), text.replace(".", "-"), text.replace(".", "/")}
    compact = re.sub(r"\D", "", text)
    if compact:
        variants.add(compact)
    return variants


def race_ids_from_entry(entry: pd.DataFrame, date_filter: str | None) -> list[str]:
    if entry.empty or "source_url" not in entry.columns:
        return []
    work = entry.copy()
    if date_filter:
        date_col = _find_date_col(work)
        if date_col:
            allowed = _date_variants(date_filter)
            work = work[work[date_col].astype(str).str.strip().isin(allowed)]
    ids: set[str] = set()
    for url in work["source_url"].dropna().astype(str).unique():
        match = re.search(r"race_id=(\d+)", url)
        if match:
            ids.add(match.group(1))
    return sorted(ids)


def load_timelines(race_ids: list[str], timeline_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for race_id in race_ids:
        path = timeline_dir / f"{race_id}.csv"
        if not path.exists():
            continue
        frame = read_csv_safe(path)
        if "race_id" not in frame.columns:
            frame["race_id"] = race_id
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    timeline = pd.concat(frames, ignore_index=True, sort=False)
    timeline["netkeiba_race_id"] = timeline["race_id"].astype(str)
    timeline["race_id"] = timeline["race_id"].map(compact_netkeiba_race_id)
    return timeline


def latest_single_odds(features: pd.DataFrame, output_csv: Path) -> int:
    if features.empty:
        out = pd.DataFrame(
            columns=[
                "race_id",
                "horse_no",
                "live_win_odds",
                "live_popularity",
                "live_place_odds_min",
                "live_place_odds_max",
                "snapshot_at",
                "parser_mode",
            ]
        )
    else:
        out = pd.DataFrame(
            {
                "race_id": features["race_id"].astype(str),
                "horse_no": features["horse_number"],
                "live_win_odds": features["odds_latest_win"],
                "live_popularity": features["odds_latest_popularity"],
                "live_place_odds_min": features["odds_place_min"],
                "live_place_odds_max": features["odds_place_max"],
                "snapshot_at": features["odds_latest_snapshot_at"],
                "parser_mode": "netkeiba_timeline",
            }
        )
        out = out[out["live_win_odds"].notna() | out["live_place_odds_min"].notna()].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return int(len(out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged netkeiba odds features for provisional entry snapshots.")
    parser.add_argument("--entry-csv", required=True)
    parser.add_argument("--date-filter", default="")
    parser.add_argument("--race-id", action="append", default=[])
    parser.add_argument("--timeline-dir", default="data/processed/odds_timeline/netkeiba")
    parser.add_argument("--combined-timeline-csv", default="data/processed/odds_timeline/netkeiba_combined/latest.csv")
    parser.add_argument("--features-csv", default="data/processed/odds_timeline/netkeiba_combined/latest_features.csv")
    parser.add_argument("--single-live-csv", default="data/processed/live_odds/netkeiba_single_odds_latest.csv")
    parser.add_argument("--output-entry-csv", required=True)
    args = parser.parse_args()

    entry_path = project_path(args.entry_csv)
    entry = read_csv_safe(entry_path)
    race_ids = sorted(set(args.race_id) | set(race_ids_from_entry(entry, args.date_filter or None)))
    timeline = load_timelines(race_ids, project_path(args.timeline_dir))
    combined_path = project_path(args.combined_timeline_csv)
    features_path = project_path(args.features_csv)
    output_entry = project_path(args.output_entry_csv)
    single_live = project_path(args.single_live_csv)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    output_entry.parent.mkdir(parents=True, exist_ok=True)

    if timeline.empty:
        timeline.to_csv(combined_path, index=False, encoding="utf-8-sig")
        features = pd.DataFrame()
        merged = entry.copy()
    else:
        race_col = _find_required_col(entry, RACE_ID_CANDIDATES, "race id")
        horse_no_col = _find_required_col(entry, HORSE_NO_CANDIDATES, "horse number")
        timeline.to_csv(combined_path, index=False, encoding="utf-8-sig")
        features = build_odds_timeline_features(timeline)
        features.to_csv(features_path, index=False, encoding="utf-8-sig")
        feature_payload_cols = [col for col in features.columns if col not in {"race_id", "horse_number"}]
        stale_cols = [col for col in feature_payload_cols if col in entry.columns]
        entry_for_merge = entry.drop(columns=stale_cols) if stale_cols else entry
        merged = merge_odds_timeline_features(
            entry_for_merge,
            features,
            race_col=race_col,
            horse_number_col=horse_no_col,
        )
        if "odds_latest_win" in merged.columns:
            if "\u5358\u52dd\u30aa\u30c3\u30ba" not in merged.columns:
                merged["\u5358\u52dd\u30aa\u30c3\u30ba"] = pd.NA
            valid_latest = clean_odds_series(merged["odds_latest_win"])
            existing = clean_odds_series(merged["\u5358\u52dd\u30aa\u30c3\u30ba"])
            merged["\u5358\u52dd\u30aa\u30c3\u30ba"] = existing
            mask = valid_latest.notna()
            merged.loc[mask, "\u5358\u52dd\u30aa\u30c3\u30ba"] = valid_latest.loc[mask]
        if "odds_latest_popularity" in merged.columns:
            if "\u4eba\u6c17" not in merged.columns:
                merged["\u4eba\u6c17"] = pd.NA
            mask = merged["odds_latest_popularity"].notna()
            merged.loc[mask, "\u4eba\u6c17"] = merged.loc[mask, "odds_latest_popularity"]
    single_rows = latest_single_odds(features, single_live)
    merged.to_csv(output_entry, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "entry_csv": str(entry_path),
                "output_entry_csv": str(output_entry),
                "race_ids": len(race_ids),
                "timeline_rows": int(len(timeline)),
                "features_rows": int(len(features)),
                "matched_entry_rows": int(merged["odds_latest_win"].notna().sum()) if "odds_latest_win" in merged.columns else 0,
                "combined_timeline_csv": str(combined_path),
                "features_csv": str(features_path),
                "single_live_csv": str(single_live),
                "single_live_rows": single_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
