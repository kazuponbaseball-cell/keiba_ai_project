from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


NETKEIBA_ODDS_URL = "https://race.netkeiba.com/odds/index.html?race_id={race_id}&type={odds_type}"
NETKEIBA_ODDS_API_URL = (
    "https://race.netkeiba.com/api/api_get_jra_odds.html"
    "?pid=api_get_jra_odds&input=UTF-8&output=json&race_id={race_id}"
    "&type={api_odds_type}&action=init&sort=odds&compress=0"
)
API_ODDS_TYPE = {"b1": "1", "1": "1"}
SNAPSHOT_COLUMNS = [
    "snapshot_at",
    "provider",
    "race_id",
    "odds_type",
    "official_datetime",
    "odds_status",
    "update_count",
    "frame_number",
    "horse_number",
    "horse_name",
    "win_odds",
    "place_odds_min",
    "place_odds_max",
    "popularity_estimated",
    "source_file",
]


def _resolve_path(path: str) -> Path:
    return Path(path).resolve()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _fetch_bytes(url: str, destination: Path, *, referer: str | None = None) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
    }
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return data


def _fetch_html(url: str, destination: Path) -> None:
    _fetch_bytes(url, destination)


def _fetch_text(url: str, destination: Path) -> str:
    data = _fetch_bytes(url, destination, referer="https://race.netkeiba.com/")
    text = data.decode("utf-8", errors="ignore")
    destination.write_text(text, encoding="utf-8")
    return text


def _num(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace(",", "").strip()
    if text in {"", "---", "---.-", "nan", "None"}:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _odds_num(value: Any) -> float | None:
    parsed = _num(value)
    if parsed is None or parsed < 1.0 or parsed >= 999.0:
        return None
    return parsed


def _empty_snapshot() -> pd.DataFrame:
    return pd.DataFrame(columns=SNAPSHOT_COLUMNS)


def _read_html_tables(html_path: Path) -> list[pd.DataFrame]:
    try:
        return pd.read_html(str(html_path), encoding="euc-jp")
    except Exception:
        return []


def _guess_html_snapshot(html_path: Path, *, race_id: str, odds_type: str, snapshot_at: str) -> pd.DataFrame:
    """Best-effort enrichment only. API odds are authoritative for current use."""
    tables = _read_html_tables(html_path)
    if not tables:
        return _empty_snapshot()

    table = max(tables, key=len).copy()
    table.columns = [str(column).replace("\n", "").replace(" ", "") for column in table.columns]

    number_col = None
    for column in table.columns:
        values = pd.to_numeric(table[column], errors="coerce")
        valid = values.dropna()
        if len(valid) and valid.between(1, 18).mean() >= 0.8:
            number_col = column
            break
    if number_col is None:
        return _empty_snapshot()

    odds_col = None
    for column in table.columns:
        if column == number_col:
            continue
        values = table[column].map(_odds_num)
        valid = values.dropna()
        if len(valid) and valid.between(1.0, 999.0, inclusive="left").mean() >= 0.6:
            odds_col = column
            break

    name_col = None
    for column in table.columns:
        if column in {number_col, odds_col}:
            continue
        values = table[column].astype(str)
        if values.str.contains(r"[ぁ-んァ-ン一-龥A-Za-z]", regex=True).mean() >= 0.5:
            name_col = column
            break

    frame_col = None
    for column in table.columns:
        if column == number_col:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        valid = values.dropna()
        if len(valid) and valid.between(1, 8).mean() >= 0.8:
            frame_col = column
            break

    out = pd.DataFrame(
        {
            "snapshot_at": snapshot_at,
            "provider": "netkeiba",
            "race_id": race_id,
            "odds_type": odds_type,
            "official_datetime": pd.NA,
            "odds_status": pd.NA,
            "update_count": pd.NA,
            "frame_number": table[frame_col] if frame_col else pd.NA,
            "horse_number": table[number_col],
            "horse_name": table[name_col] if name_col else "",
            "win_odds": table[odds_col].map(_odds_num) if odds_col else pd.NA,
            "place_odds_min": pd.NA,
            "place_odds_max": pd.NA,
            "popularity_estimated": pd.NA,
            "source_file": str(html_path),
        }
    )
    out["horse_number"] = pd.to_numeric(out["horse_number"], errors="coerce").astype("Int64")
    out["frame_number"] = pd.to_numeric(out["frame_number"], errors="coerce").astype("Int64")
    odds = pd.to_numeric(out["win_odds"], errors="coerce")
    out["popularity_estimated"] = odds.rank(method="min", ascending=True).astype("Int64")
    out.loc[odds.isna(), "popularity_estimated"] = pd.NA
    return out.dropna(subset=["horse_number"]).sort_values("horse_number")


def parse_tansho_html(html_path: Path, *, race_id: str, odds_type: str, snapshot_at: str) -> pd.DataFrame:
    return _guess_html_snapshot(html_path, race_id=race_id, odds_type=odds_type, snapshot_at=snapshot_at)


def parse_tansho_api(
    api_path: Path,
    *,
    race_id: str,
    odds_type: str,
    snapshot_at: str,
    html_snapshot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    payload = json.loads(api_path.read_text(encoding="utf-8"))
    data = payload.get("data") or {}
    odds = data.get("odds") or {}
    win_odds = odds.get("1") or {}
    place_odds = odds.get("2") or {}
    if not win_odds and not place_odds:
        raise ValueError(f"No API odds found in {api_path}: {payload}")

    names: dict[int, dict[str, object]] = {}
    if html_snapshot is not None and not html_snapshot.empty:
        for row in html_snapshot.itertuples(index=False):
            horse_number = int(row.horse_number) if pd.notna(row.horse_number) else None
            if horse_number is None:
                continue
            names[horse_number] = {
                "frame_number": row.frame_number,
                "horse_name": row.horse_name,
            }

    rows = []
    horse_numbers = sorted({*win_odds.keys(), *place_odds.keys()}, key=lambda value: int(value))
    for key in horse_numbers:
        horse_number = int(key)
        win = win_odds.get(key) or [None, None, None]
        place = place_odds.get(key) or [None, None, None]
        rows.append(
            {
                "snapshot_at": snapshot_at,
                "provider": "netkeiba",
                "race_id": race_id,
                "odds_type": odds_type,
                "official_datetime": data.get("official_datetime"),
                "odds_status": payload.get("status"),
                "update_count": payload.get("update_count"),
                "frame_number": names.get(horse_number, {}).get("frame_number", pd.NA),
                "horse_number": horse_number,
                "horse_name": names.get(horse_number, {}).get("horse_name", ""),
                "win_odds": _odds_num(win[0] if len(win) > 0 else None),
                "place_odds_min": _odds_num(place[0] if len(place) > 0 else None),
                "place_odds_max": _odds_num(place[1] if len(place) > 1 else None),
                "popularity_estimated": int(win[2]) if len(win) > 2 and str(win[2]).isdigit() else pd.NA,
                "source_file": str(api_path),
            }
        )
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS).sort_values("horse_number")


def append_timeline(snapshot: pd.DataFrame, timeline_path: Path) -> pd.DataFrame:
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    if timeline_path.exists():
        existing = pd.read_csv(timeline_path, encoding="utf-8-sig")
        combined = pd.concat([existing, snapshot], ignore_index=True)
        combined = combined.drop_duplicates(["snapshot_at", "race_id", "odds_type", "horse_number"], keep="last")
    else:
        combined = snapshot
    combined = combined.sort_values(["snapshot_at", "horse_number"])
    combined.to_csv(timeline_path, index=False, encoding="utf-8-sig")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and store a timestamped netkeiba odds snapshot.")
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--odds-type", default="b1", help="netkeiba odds type. b1 is win/place page.")
    parser.add_argument("--html-file", default=None, help="Parse an existing HTML file instead of fetching.")
    parser.add_argument("--api-file", default=None, help="Parse an existing API JSON file instead of fetching.")
    parser.add_argument("--raw-dir", default="data/raw/netkeiba_odds")
    parser.add_argument("--timeline-dir", default="data/processed/odds_timeline/netkeiba")
    parser.add_argument("--snapshot-at", default=None, help="Override snapshot timestamp, e.g. 20260614_093000.")
    args = parser.parse_args()

    snapshot_at = args.snapshot_at or _timestamp()
    raw_dir = _resolve_path(args.raw_dir) / args.race_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.html_file:
        html_path = _resolve_path(args.html_file)
        fetched_url = None
    else:
        html_path = raw_dir / f"{snapshot_at}_{args.odds_type}.html"
        fetched_url = NETKEIBA_ODDS_URL.format(race_id=args.race_id, odds_type=args.odds_type)
        _fetch_html(fetched_url, html_path)

    api_path = _resolve_path(args.api_file) if args.api_file else None
    api_url = None
    if api_path is None and not args.html_file:
        api_odds_type = API_ODDS_TYPE.get(args.odds_type, args.odds_type)
        api_url = NETKEIBA_ODDS_API_URL.format(race_id=args.race_id, api_odds_type=api_odds_type)
        api_path = html_path.with_suffix(".api.json")
        _fetch_text(api_url, api_path)

    html_snapshot = parse_tansho_html(html_path, race_id=args.race_id, odds_type=args.odds_type, snapshot_at=snapshot_at)
    if api_path and api_path.exists():
        try:
            snapshot = parse_tansho_api(
                api_path,
                race_id=args.race_id,
                odds_type=args.odds_type,
                snapshot_at=snapshot_at,
                html_snapshot=html_snapshot,
            )
        except ValueError:
            snapshot = html_snapshot
    else:
        snapshot = html_snapshot

    if snapshot.empty:
        raise ValueError(f"No odds snapshot could be parsed for race_id={args.race_id}.")

    timeline_path = _resolve_path(args.timeline_dir) / f"{args.race_id}.csv"
    timeline = append_timeline(snapshot, timeline_path)

    latest_columns = [
        column
        for column in ["horse_number", "horse_name", "win_odds", "place_odds_min", "place_odds_max", "popularity_estimated"]
        if column in snapshot.columns
    ]
    latest = snapshot[latest_columns].to_dict(orient="records")
    summary = {
        "race_id": args.race_id,
        "odds_type": args.odds_type,
        "snapshot_at": snapshot_at,
        "fetched_url": fetched_url,
        "api_url": api_url,
        "html_path": str(html_path),
        "api_path": str(api_path) if api_path else None,
        "snapshot_rows": int(len(snapshot)),
        "timeline_path": str(timeline_path),
        "timeline_rows": int(len(timeline)),
        "odds_available_rows": int(snapshot["win_odds"].notna().sum()),
        "latest": latest,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
