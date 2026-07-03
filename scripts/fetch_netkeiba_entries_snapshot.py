from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from lxml import html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import inference_optional_columns, inference_required_columns, load_json_config
from src.utils.paths import ensure_dir, project_path


VENUES = {
    "02": "函館",
    "03": "福島",
    "05": "東京",
    "09": "阪神",
    "10": "小倉",
}

DEFAULT_RACE_KEYS = {
    "20260620": [("05", "03", "05"), ("09", "03", "05"), ("02", "01", "03")],
    "20260621": [("05", "03", "06"), ("09", "03", "06"), ("02", "01", "04")],
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).replace("\xa0", " ").split())
    return text.strip()


def _node_text(node: Any) -> str:
    return _clean(" ".join(node.xpath(".//text()")))


def _first_text(tree: html.HtmlElement, xpaths: list[str]) -> str:
    for xp in xpaths:
        nodes = tree.xpath(xp)
        if not nodes:
            continue
        value = nodes[0]
        if hasattr(value, "xpath"):
            text = _node_text(value)
        else:
            text = _clean(value)
        if text:
            return text
    return ""


def _first_link_id(node: html.HtmlElement, pattern: str) -> str:
    for href in node.xpath(".//a/@href"):
        match = re.search(pattern, href)
        if match:
            return match.group(1)
    return ""


def _parse_sex_age(value: str) -> tuple[str, Any]:
    match = re.search(r"([牡牝セ])\s*(\d+)", value)
    if not match:
        return "", pd.NA
    return match.group(1), int(match.group(2))


def _race_id(date: str, venue_code: str, meeting: str, day: str, race_no: int) -> str:
    return f"{date[:4]}{venue_code}{meeting}{day}{race_no:02d}"


def _short_race_id(date: str, venue_code: str, meeting: str, day: str, race_no: int) -> int:
    return int(f"{int(venue_code)}{date[2:4]}{meeting}{day}{race_no:02d}")


def _read_url(url: str, cache_path: Path, *, refresh: bool, sleep_seconds: float) -> str:
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 KeibaAI/1.0",
            "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
    candidates = []
    for enc in ("utf-8", "euc-jp", "cp932"):
        decoded = raw.decode(enc, errors="replace")
        candidates.append((decoded.count("\ufffd"), decoded))
    text = min(candidates, key=lambda item: item[0])[1]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return text


def _race_metadata(tree: html.HtmlElement, *, date: str, venue_code: str, meeting: str, day: str, race_no: int) -> dict[str, Any]:
    race_name = _first_text(tree, [
        "//*[contains(@class, 'RaceName')]",
        "//h1",
        "//title",
    ])
    race_name = re.sub(r"\s*出馬表.*$", "", race_name).strip()

    race_data_01 = _first_text(tree, ["//*[contains(@class, 'RaceData01')]"])
    race_data_02 = _first_text(tree, ["//*[contains(@class, 'RaceData02')]"])
    if not race_data_01:
        body_text = _node_text(tree)
        match = re.search(r"(\d{1,2}:\d{2})発走\s*/\s*([芝ダ障])(\d+)m", body_text)
        race_data_01 = match.group(0) if match else ""
    surface = ""
    distance = pd.NA
    post_time = ""
    match = re.search(r"(\d{1,2}:\d{2})発走\s*/\s*([芝ダ障])(\d+)m", race_data_01)
    if match:
        post_time = match.group(1)
        surface = "ダ" if match.group(2) == "ダ" else match.group(2)
        distance = int(match.group(3))

    field_size = pd.NA
    match = re.search(r"(\d+)頭", race_data_02 or _node_text(tree))
    if match:
        field_size = int(match.group(1))

    dt = datetime.strptime(date, "%Y%m%d")
    return {
        "race_id_full": _race_id(date, venue_code, meeting, day, race_no),
        "race_id_short": _short_race_id(date, venue_code, meeting, day, race_no),
        "date_int": int(date[2:]),
        "date_s": f"{dt.year}.{dt.month}.{dt.day}",
        "venue": VENUES.get(venue_code, venue_code),
        "race_no": race_no,
        "race_name": race_name,
        "post_time": post_time,
        "surface": surface,
        "distance": distance,
        "field_size": field_size,
    }


def _select_cell(row: html.HtmlElement, class_name: str) -> html.HtmlElement | None:
    nodes = row.xpath(f"./td[contains(concat(' ', normalize-space(@class), ' '), ' {class_name} ')]")
    return nodes[0] if nodes else None


def _select_cell_prefix(row: html.HtmlElement, prefix: str) -> html.HtmlElement | None:
    for cell in row.xpath("./td"):
        classes = (cell.get("class") or "").split()
        if any(cls.startswith(prefix) for cls in classes):
            return cell
    return None


def _na_if_blank_marker(value: str) -> Any:
    text = _clean(value)
    if not text or text in {"---.-", "**", "-"}:
        return pd.NA
    return text


def _parse_runner_rows(tree: html.HtmlElement, meta: dict[str, Any], *, source_url: str) -> list[dict[str, Any]]:
    tables = tree.xpath("//table[contains(concat(' ', normalize-space(@class), ' '), ' ShutubaTable ')]")
    context = tables[0] if tables else tree
    rows = context.xpath(".//tr[contains(concat(' ', normalize-space(@class), ' '), ' HorseList ')]")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        waku_node = _select_cell_prefix(row, "Waku")
        umaban_node = _select_cell_prefix(row, "Umaban")
        horse_node = _select_cell(row, "HorseInfo")
        sex_age_node = _select_cell(row, "Barei")
        jockey_node = _select_cell(row, "Jockey")
        trainer_node = _select_cell(row, "Trainer")

        cells = row.xpath("./td")
        weight_node = cells[5] if len(cells) > 5 else None
        odds = _first_text(row, [".//span[starts-with(@id, 'odds-')]"])
        popularity = _first_text(row, [".//span[starts-with(@id, 'ninki-')]"])

        horse_name = _first_text(row, [".//*[contains(@class, 'HorseName')]//a", ".//*[contains(@class, 'HorseName')]"])
        if not horse_name and horse_node is not None:
            horse_name = _node_text(horse_node).split()[0]
        sex, age = _parse_sex_age(_node_text(sex_age_node) if sex_age_node is not None else "")

        parsed.append(
            {
                "日付": meta["date_int"],
                "日付S": meta["date_s"],
                "場所": meta["venue"],
                "Ｒ": meta["race_no"],
                "レース名": meta["race_name"],
                "発走時刻": meta["post_time"],
                "芝・ダ": meta["surface"],
                "距離": meta["distance"],
                "頭数": meta["field_size"],
                "出走頭数": meta["field_size"],
                "レースID(新/馬番無)": meta["race_id_short"],
                "枠番": _node_text(waku_node) if waku_node is not None else pd.NA,
                "馬番": _node_text(umaban_node) if umaban_node is not None else pd.NA,
                "馬名": horse_name,
                "血統登録番号": _first_link_id(horse_node if horse_node is not None else row, r"/horse/(\d+)"),
                "性別": sex,
                "年齢": age,
                "斤量": _node_text(weight_node) if weight_node is not None else pd.NA,
                "騎手": _node_text(jockey_node) if jockey_node is not None else "",
                "騎手コード": _first_link_id(jockey_node if jockey_node is not None else row, r"/jockey/(\d+)"),
                "調教師コード": _first_link_id(trainer_node if trainer_node is not None else row, r"/trainer/(\d+)"),
                "馬場状態": pd.NA,
                "単勝オッズ": _na_if_blank_marker(odds),
                "人気": _na_if_blank_marker(popularity),
                "異常コード": 0,
                "確定着順": pd.NA,
                "source_url": source_url,
            }
        )
    return parsed


def _blank_snapshot(columns: list[str], rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = pd.DataFrame(index=range(len(rows)), columns=columns)
    source = pd.DataFrame(rows)
    for col in source.columns:
        if col in out.columns:
            out[col] = source[col]
    if "source_url" not in out.columns:
        out["source_url"] = source["source_url"]
    return out


def _race_keys_from_target_card(path: str, dates: list[str]) -> list[tuple[str, str, str, str, int]]:
    card = pd.read_csv(project_path(path), encoding="utf-8-sig", low_memory=False)
    if "race_id" not in card.columns:
        raise ValueError(f"TARGET card CSV is missing race_id: {path}")
    work = card.copy()
    if dates:
        work = work[work["race_id"].astype(str).str[:8].isin(set(dates))]
    if work.empty:
        raise ValueError(f"No races in TARGET card for dates={dates}: {path}")

    keys: list[tuple[str, str, str, str, int]] = []
    for race_id in sorted(work["race_id"].astype(str).dropna().unique()):
        if not re.fullmatch(r"\d{16}", race_id):
            continue
        keys.append((race_id[:8], race_id[8:10], race_id[10:12], race_id[12:14], int(race_id[14:16])))
    if not keys:
        raise ValueError(f"No valid 16-digit race_id values in TARGET card: {path}")
    return keys


def _race_keys(dates: list[str], *, target_card_csv: str | None = None) -> list[tuple[str, str, str, str, int]]:
    if target_card_csv:
        return _race_keys_from_target_card(target_card_csv, dates)
    keys: list[tuple[str, str, str, str, int]] = []
    for date in dates:
        if date not in DEFAULT_RACE_KEYS:
            raise ValueError(f"No built-in race key map for {date}. Use --target-card-csv for custom races.")
        for venue_code, meeting, day in DEFAULT_RACE_KEYS[date]:
            for race_no in range(1, 13):
                keys.append((date, venue_code, meeting, day, race_no))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public netkeiba race cards into a provisional entry snapshot.")
    parser.add_argument("--dates", nargs="+", default=["20260620", "20260621"])
    parser.add_argument("--feature-config", default="config/baseline_features.json")
    parser.add_argument("--output-csv", default="data/datasets/inference/weekly/entry_snapshot_netkeiba_20260620_20260621.csv")
    parser.add_argument("--raw-output-csv", default="data/raw/external/netkeiba_entries/netkeiba_entries_raw_20260620_20260621.csv")
    parser.add_argument("--cache-dir", default="data/raw/external/netkeiba_entries/html")
    parser.add_argument("--target-card-csv", default=None, help="Optional TARGET DE entry-card CSV. Race keys are derived from race_id.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    args = parser.parse_args()

    config = load_json_config(args.feature_config)
    columns = list(dict.fromkeys([*inference_required_columns(config), *inference_optional_columns(config)]))
    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    cache_dir = project_path(args.cache_dir)

    for date, venue_code, meeting, day, race_no in _race_keys(args.dates, target_card_csv=args.target_card_csv):
        race_id = _race_id(date, venue_code, meeting, day, race_no)
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        cache_path = cache_dir / f"{race_id}.html"
        try:
            page = _read_url(url, cache_path, refresh=args.refresh, sleep_seconds=args.sleep_seconds)
            tree = html.fromstring(page)
            meta = _race_metadata(tree, date=date, venue_code=venue_code, meeting=meeting, day=day, race_no=race_no)
            rows = _parse_runner_rows(tree, meta, source_url=url)
            if not rows:
                errors.append({"race_id": race_id, "url": url, "error": "no_runner_rows"})
                continue
            all_rows.extend(rows)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            errors.append({"race_id": race_id, "url": url, "error": str(exc)})

    if not all_rows:
        raise RuntimeError(f"No entry rows fetched. errors={errors[:5]}")

    output = _blank_snapshot(columns, all_rows)
    raw = pd.DataFrame(all_rows)
    output_path = project_path(args.output_csv)
    raw_path = project_path(args.raw_output_csv)
    ensure_dir(output_path.parent)
    ensure_dir(raw_path.parent)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    summary = {
        "output_csv": str(output_path),
        "raw_output_csv": str(raw_path),
        "rows": int(len(output)),
        "races": int(output["レースID(新/馬番無)"].nunique()) if "レースID(新/馬番無)" in output.columns else 0,
        "dates": sorted(output["日付S"].dropna().unique().tolist()) if "日付S" in output.columns else [],
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
