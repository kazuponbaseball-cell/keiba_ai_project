from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from lxml import html


ROOT = Path(__file__).resolve().parents[1]
JRA_ODDS_URL = "https://www.jra.go.jp/JRADB/accessO.html"
ODDS_TOP_CNAME = "pw15oli00/6D"


BET_BY_DIGIT = {
    "1": "win_place_frame",
    "4": "umaren",
    "5": "wide",
    "6": "umatan",
    "7": "trio",
    "8": "trifecta",
}

PAIR_BETS = {"umaren", "wide"}
SINGLE_BETS = {"win_place_frame"}

VENUE_RE = re.compile(
    r"pw15orl0*(?P<jyo>\d{2})(?P<year>\d{4})(?P<kaiji>\d{2})(?P<nichiji>\d{2})(?P<date>\d{8})/[0-9A-F]+"
)
RACE_RE = re.compile(
    r"pw15(?P<digit>\d).*?S3(?P<jyo>\d{2})(?P<year>\d{4})(?P<kaiji>\d{2})(?P<nichiji>\d{2})(?P<race>\d{2})(?P<date>\d{8})Z(?:99)?/[0-9A-F]+"
)
DO_ACTION_RE = re.compile(r"doAction\('/JRADB/accessO\.html(?:#[^']*)?',\s*'([^']+)'\)")


@dataclass(frozen=True)
class VenueLink:
    cname: str
    race_prefix: str
    date_key: str
    jyo: str
    kaiji: str
    nichiji: str


@dataclass(frozen=True)
class RaceOddsLink:
    cname: str
    race_id: str
    ticket_type: str


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype={"race_id": str}, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False)


def merge_existing_odds(existing_path: Path, fresh: pd.DataFrame, *, race_ids: set[str], key_cols: list[str]) -> pd.DataFrame:
    if not race_ids or not existing_path.exists():
        return fresh
    try:
        existing = read_csv(existing_path)
    except Exception:
        return fresh
    if existing.empty:
        return fresh
    if "race_id" not in existing.columns:
        return fresh
    existing["race_id"] = existing["race_id"].astype(str)
    kept = existing[~existing["race_id"].isin(race_ids)].copy()
    out = pd.concat([kept, fresh], ignore_index=True, sort=False)
    for col in key_cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out.drop_duplicates(key_cols, keep="last")
    sort_cols = [col for col in key_cols if col in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def date_key(value: str) -> str:
    text = str(value or "").strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        return digits
    if len(digits) == 6:
        yy = int(digits[:2])
        return f"{2000 + yy if yy < 80 else 1900 + yy}{digits[2:]}"
    if not text:
        return datetime.now().strftime("%Y%m%d")
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y%m%d")
    return digits[:8]


def normalize_race_key(raw: str, *, fallback_date: str) -> str:
    digits = re.sub(r"\D", "", str(raw or ""))
    if len(digits) >= 16:
        return digits[:16]
    if len(digits) == 12 and fallback_date:
        # netkeiba-style YYYY + jyo + kaiji + nichiji + race.
        return f"{fallback_date}{digits[4:6]}{digits[6:8]}{digits[8:10]}{digits[10:12]}"
    return digits


def split_values(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part for part in re.split(r"[,\s]+", str(value or "")) if part)
    return out


def post_cname(cname: str, *, timeout: float = 20.0, retries: int = 2) -> bytes:
    data = urllib.parse.urlencode({"cname": cname}).encode("ascii")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://www.jra.go.jp/keiba/",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(JRA_ODDS_URL, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"JRA official odds request failed cname={cname}: {last_error}")


def decode_jra_html(raw: bytes) -> str:
    for enc in ("shift_jis", "cp932", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp932", errors="replace")


def extract_cnames(text: str) -> list[str]:
    return list(dict.fromkeys(DO_ACTION_RE.findall(text)))


def parse_venue_links(text: str, *, target_date: str) -> list[VenueLink]:
    links: list[VenueLink] = []
    for cname in extract_cnames(text):
        match = VENUE_RE.search(cname)
        if not match:
            continue
        if target_date and match.group("date") != target_date:
            continue
        race_prefix = f"{match.group('date')}{match.group('jyo')}{match.group('kaiji')}{match.group('nichiji')}"
        links.append(
            VenueLink(
                cname=cname,
                race_prefix=race_prefix,
                date_key=match.group("date"),
                jyo=match.group("jyo"),
                kaiji=match.group("kaiji"),
                nichiji=match.group("nichiji"),
            )
        )
    return list(dict.fromkeys(links))


def parse_race_odds_links(text: str, *, bet_types: set[str]) -> list[RaceOddsLink]:
    links: list[RaceOddsLink] = []
    seen: set[tuple[str, str]] = set()
    for cname in extract_cnames(text):
        match = RACE_RE.search(cname)
        if not match:
            continue
        ticket_type = BET_BY_DIGIT.get(match.group("digit"))
        if ticket_type not in bet_types:
            continue
        race_id = (
            f"{match.group('date')}{match.group('jyo')}{match.group('kaiji')}"
            f"{match.group('nichiji')}{match.group('race')}"
        )
        key = (race_id, ticket_type)
        if key in seen:
            continue
        seen.add(key)
        links.append(RaceOddsLink(cname=cname, race_id=race_id, ticket_type=ticket_type))
    return links


def num_text(value: str) -> float | None:
    text = re.sub(r"\s+", "", str(value or ""))
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def odds_num_text(value: str) -> float | None:
    value_num = num_text(value)
    if value_num is None or value_num < 1.0 or value_num >= 999.0:
        return None
    return value_num


def class_contains(name: str) -> str:
    return f"contains(concat(' ', normalize-space(@class), ' '), ' {name} ')"


def node_text(node) -> str:
    return " ".join(part.strip() for part in node.xpath(".//text()") if part and part.strip())


def parse_pair_page(text: str, *, race_id: str, ticket_type: str, snapshot_at: str, cname: str) -> list[dict]:
    tree = html.fromstring(text)
    table_class = "wide" if ticket_type == "wide" else "umaren"
    rows: list[dict] = []
    for table in tree.xpath(f"//table[{class_contains(table_class)}]"):
        caption = node_text(table.xpath(".//caption")[0]) if table.xpath(".//caption") else ""
        a_no = num_text(caption)
        if a_no is None:
            continue
        for tr in table.xpath(".//tbody/tr"):
            th = tr.xpath("./th")
            td = tr.xpath("./td")
            if not th or not td:
                continue
            b_no = num_text(node_text(th[0]))
            if b_no is None:
                continue
            odds_min = num_text(" ".join(td[0].xpath(".//*[contains(@class, 'min')]/text()")))
            odds_max = num_text(" ".join(td[0].xpath(".//*[contains(@class, 'max')]/text()")))
            odds = odds_min if odds_min is not None else num_text(node_text(td[0]))
            if odds is None or odds <= 0:
                continue
            lo = int(min(a_no, b_no))
            hi = int(max(a_no, b_no))
            rows.append(
                {
                    "race_id": race_id,
                    "ticket_type": ticket_type,
                    "a_no": lo,
                    "b_no": hi,
                    "live_pay_per100": round(float(odds) * 100.0, 1),
                    "live_odds": float(odds),
                    "popularity": pd.NA,
                    "snapshot_at": snapshot_at,
                    "parser_mode": "jra_official_html_pair",
                    "source": "jra_official",
                    "live_odds_min": odds_min if odds_min is not None else odds,
                    "live_odds_max": odds_max if odds_max is not None else odds,
                    "cname": cname,
                }
            )
    return rows


def parse_single_page(text: str, *, race_id: str, snapshot_at: str, cname: str) -> list[dict]:
    tree = html.fromstring(text)
    rows: list[dict] = []
    for tr in tree.xpath(f"//table[{class_contains('tanpuku')}]//tbody/tr"):
        num_nodes = tr.xpath(f"./td[{class_contains('num')}]")
        if not num_nodes:
            continue
        horse_no = num_text(node_text(num_nodes[0]))
        if horse_no is None:
            continue
        win_nodes = tr.xpath(f"./td[{class_contains('odds_tan')}]")
        place_nodes = tr.xpath(f"./td[{class_contains('odds_fuku')}]")
        win_odds = odds_num_text(node_text(win_nodes[0])) if win_nodes else None
        place_min = None
        place_max = None
        if place_nodes:
            place_min = odds_num_text(" ".join(place_nodes[0].xpath(".//*[contains(@class, 'min')]/text()")))
            place_max = odds_num_text(" ".join(place_nodes[0].xpath(".//*[contains(@class, 'max')]/text()")))
            if place_min is None:
                place_min = odds_num_text(node_text(place_nodes[0]))
        rows.append(
            {
                "race_id": race_id,
                "horse_no": int(horse_no),
                "live_win_odds": win_odds if win_odds is not None else pd.NA,
                "live_popularity": pd.NA,
                "live_place_odds_min": place_min if place_min is not None else pd.NA,
                "live_place_odds_max": place_max if place_max is not None else pd.NA,
                "snapshot_at": snapshot_at,
                "parser_mode": "jra_official_html_single",
                "source": "jra_official",
                "cname": cname,
            }
        )
    return rows


def write_html(raw_dir: Path, *, race_id: str, ticket_type: str, snapshot_at: str, text: str) -> Path:
    out_dir = raw_dir / race_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{snapshot_at}_{ticket_type}.html"
    path.write_text(text, encoding="cp932", errors="replace")
    return path


def fetch_detail_link(link: RaceOddsLink, *, raw_dir: Path, snapshot_at: str) -> tuple[list[dict], list[dict], dict | None, dict | None]:
    try:
        text = decode_jra_html(post_cname(link.cname))
        raw_path = write_html(raw_dir, race_id=link.race_id, ticket_type=link.ticket_type, snapshot_at=snapshot_at, text=text)
        fetched = {"race_id": link.race_id, "ticket_type": link.ticket_type, "cname": link.cname, "raw_path": str(raw_path)}
        pair_rows: list[dict] = []
        single_rows: list[dict] = []
        if link.ticket_type in PAIR_BETS:
            pair_rows.extend(parse_pair_page(text, race_id=link.race_id, ticket_type=link.ticket_type, snapshot_at=snapshot_at, cname=link.cname))
        elif link.ticket_type in SINGLE_BETS:
            single_rows.extend(parse_single_page(text, race_id=link.race_id, snapshot_at=snapshot_at, cname=link.cname))
        return pair_rows, single_rows, fetched, None
    except Exception as exc:
        return [], [], None, {
            "stage": "detail",
            "race_id": link.race_id,
            "ticket_type": link.ticket_type,
            "cname": link.cname,
            "error": str(exc),
        }


def finalize_pair(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "race_id",
        "ticket_type",
        "a_no",
        "b_no",
        "live_pay_per100",
        "live_odds",
        "popularity",
        "snapshot_at",
        "parser_mode",
        "source",
        "live_odds_min",
        "live_odds_max",
        "cname",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["race_id"] = df["race_id"].astype(str)
    df["a_no"] = pd.to_numeric(df["a_no"], errors="coerce").astype("Int64")
    df["b_no"] = pd.to_numeric(df["b_no"], errors="coerce").astype("Int64")
    df["live_pay_per100"] = pd.to_numeric(df["live_pay_per100"], errors="coerce")
    df["live_odds"] = pd.to_numeric(df["live_odds"], errors="coerce")
    df = df[df["race_id"].ne("") & df["a_no"].notna() & df["b_no"].notna() & df["live_odds"].gt(0)].copy()
    df = df.sort_values(["race_id", "ticket_type", "snapshot_at", "a_no", "b_no"])
    df = df.drop_duplicates(["race_id", "ticket_type", "a_no", "b_no"], keep="last")
    return df[cols]


def finalize_single(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "race_id",
        "horse_no",
        "live_win_odds",
        "live_popularity",
        "live_place_odds_min",
        "live_place_odds_max",
        "snapshot_at",
        "parser_mode",
        "source",
        "cname",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["race_id"] = df["race_id"].astype(str)
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    for col in ["live_win_odds", "live_popularity", "live_place_odds_min", "live_place_odds_max"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["live_win_odds", "live_place_odds_min", "live_place_odds_max"]:
        df[col] = df[col].where(df[col].ge(1.0) & df[col].lt(999.0))
    odds_rank = df.groupby("race_id")["live_win_odds"].rank(method="min", ascending=True)
    df["live_popularity"] = df["live_popularity"].where(df["live_popularity"].notna(), odds_rank)
    df = df[df["race_id"].ne("") & df["horse_no"].notna()].copy()
    df = df.sort_values(["race_id", "snapshot_at", "horse_no"])
    df = df.drop_duplicates(["race_id", "horse_no"], keep="last")
    return df[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live odds from JRA official odds pages and normalize to project CSVs.")
    parser.add_argument("--date", default="", help="YYYYMMDD / YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--race-keys", nargs="*", default=[], help="16-digit JRA-VAN race IDs, or comma-separated list.")
    parser.add_argument(
        "--bet-types",
        nargs="*",
        default=["win_place_frame", "umaren", "wide"],
        choices=["win_place_frame", "umaren", "wide", "umatan", "trio", "trifecta"],
    )
    parser.add_argument("--raw-dir", default="data/raw/jra_official_odds")
    parser.add_argument("--pair-output-csv", default="data/processed/live_odds/jra_official_pair_odds_latest.csv")
    parser.add_argument("--single-output-csv", default="data/processed/live_odds/jra_official_single_odds_latest.csv")
    parser.add_argument("--summary-json", default="outputs/analysis/jra_official_odds/summary.json")
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum parallel detail-page fetches. 1 keeps sequential behavior.")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge fetched race rows into existing output CSVs instead of replacing all races.",
    )
    args = parser.parse_args()

    target_date = date_key(args.date)
    wanted_keys = {
        normalize_race_key(key, fallback_date=target_date)
        for key in split_values(args.race_keys)
        if normalize_race_key(key, fallback_date=target_date)
    }
    bet_types = set(args.bet_types)
    raw_dir = project_path(args.raw_dir)
    snapshot_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    pair_rows: list[dict] = []
    single_rows: list[dict] = []
    errors: list[dict] = []
    fetched_links: list[dict] = []

    top_text = decode_jra_html(post_cname(ODDS_TOP_CNAME))
    venue_links = parse_venue_links(top_text, target_date=target_date)
    if wanted_keys:
        prefixes = {key[:14] for key in wanted_keys if len(key) >= 16}
        venue_links = [link for link in venue_links if link.race_prefix in prefixes]

    race_links: list[RaceOddsLink] = []
    for venue in venue_links:
        try:
            venue_text = decode_jra_html(post_cname(venue.cname))
            race_links.extend(parse_race_odds_links(venue_text, bet_types=bet_types))
            time.sleep(args.sleep_seconds)
        except Exception as exc:
            errors.append({"stage": "venue", "cname": venue.cname, "error": str(exc)})

    if wanted_keys:
        race_links = [link for link in race_links if link.race_id in wanted_keys]

    max_workers = max(1, int(args.max_workers))
    if max_workers == 1:
        for link in race_links:
            pairs, singles, fetched, error = fetch_detail_link(link, raw_dir=raw_dir, snapshot_at=snapshot_at)
            pair_rows.extend(pairs)
            single_rows.extend(singles)
            if fetched is not None:
                fetched_links.append(fetched)
            if error is not None:
                errors.append(error)
            time.sleep(args.sleep_seconds)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for link in race_links:
                futures.append(executor.submit(fetch_detail_link, link, raw_dir=raw_dir, snapshot_at=snapshot_at))
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
            for future in as_completed(futures):
                pairs, singles, fetched, error = future.result()
                pair_rows.extend(pairs)
                single_rows.extend(singles)
                if fetched is not None:
                    fetched_links.append(fetched)
                if error is not None:
                    errors.append(error)

    pair = finalize_pair(pair_rows)
    single = finalize_single(single_rows)

    pair_out = project_path(args.pair_output_csv)
    single_out = project_path(args.single_output_csv)
    if args.merge_existing and wanted_keys:
        pair = merge_existing_odds(
            pair_out,
            pair,
            race_ids=wanted_keys,
            key_cols=["race_id", "ticket_type", "a_no", "b_no"],
        )
        single = merge_existing_odds(
            single_out,
            single,
            race_ids=wanted_keys,
            key_cols=["race_id", "horse_no"],
        )
    pair_out.parent.mkdir(parents=True, exist_ok=True)
    single_out.parent.mkdir(parents=True, exist_ok=True)
    pair.to_csv(pair_out, index=False, encoding="utf-8-sig")
    single.to_csv(single_out, index=False, encoding="utf-8-sig")

    summary = {
        "source": "jra_official",
        "date": target_date,
        "wanted_race_keys": sorted(wanted_keys),
        "venue_links": len(venue_links),
        "detail_links": len(race_links),
        "max_workers": max_workers,
        "sleep_seconds": args.sleep_seconds,
        "merge_existing": bool(args.merge_existing),
        "pair_output_csv": str(pair_out),
        "single_output_csv": str(single_out),
        "pair_rows": int(len(pair)),
        "single_rows": int(len(single)),
        "pair_by_ticket_type": pair["ticket_type"].value_counts().to_dict() if not pair.empty else {},
        "single_races": int(single["race_id"].nunique()) if not single.empty else 0,
        "fetched_links": fetched_links,
        "errors": errors,
    }
    summary_path = project_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors and not (pair_rows or single_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
