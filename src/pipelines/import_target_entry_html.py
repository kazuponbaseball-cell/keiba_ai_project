from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.data.loaders import (
    inference_optional_columns,
    inference_required_columns,
    load_json_config,
)
from src.utils.paths import ensure_dir, project_path
from src.utils.runtime_config import load_runtime_config


VENUE_CODES = {
    "札幌": "01",
    "函館": "02",
    "福島": "03",
    "新潟": "04",
    "東京": "05",
    "中山": "06",
    "中京": "07",
    "京都": "08",
    "阪神": "09",
    "小倉": "10",
}


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("$", "").replace("*", "").strip()


def _split_sex_age(value: object) -> tuple[str | pd.NA, int | pd.NA]:
    text = _clean_text(value)
    match = re.match(r"([牡牝セ])(\d+)", text)
    if not match:
        return pd.NA, pd.NA
    return match.group(1), int(match.group(2))


def _parse_metadata(html_text: str) -> dict[str, object]:
    date_match = re.search(r"(\d{4})年\s*(\d+)月\s*(\d+)日", html_text)
    meeting_match = re.search(r"(\d+)回([^0-9<]+?)(\d+)日目", html_text)
    race_match = re.search(r"【\s*([０-９0-9]+)Ｒ】.*?第\d+回([^<]+)", html_text, re.S)
    surface_match = re.search(r"<B>(芝|ダ|障)(\d+)m", html_text)
    post_match = re.search(r"\[(\d{1,2}:\d{2})発走\]", html_text)
    field_match = re.search(r"(\d+)頭", html_text)

    if not (date_match and meeting_match and race_match and surface_match):
        raise ValueError("Could not parse TARGET race metadata from HTML.")

    year, month, day = map(int, date_match.groups())
    kaiji = int(meeting_match.group(1))
    venue = meeting_match.group(2).strip()
    nichiji = int(meeting_match.group(3))
    race_no = int(race_match.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789")))
    race_name = _clean_text(race_match.group(2))
    venue_code = VENUE_CODES.get(venue)
    if venue_code is None:
        raise ValueError(f"Unknown venue in TARGET HTML: {venue}")

    race_id = f"{venue_code}{year % 100:02d}{kaiji}{nichiji}{race_no:02d}"
    return {
        "日付": int(f"{year % 100:02d}{month:02d}{day:02d}"),
        "日付S": f"{year}.{month}.{day}",
        "場所": venue,
        "Ｒ": race_no,
        "レース名": race_name,
        "芝・ダ": surface_match.group(1),
        "距離": int(surface_match.group(2)),
        "頭数": int(field_match.group(1)) if field_match else pd.NA,
        "出走頭数": int(field_match.group(1)) if field_match else pd.NA,
        "発走時刻": post_match.group(1) if post_match else pd.NA,
        "レースID(新/馬番無)": race_id,
    }


def build_snapshot(html_path: Path, columns: list[str], horses_path: Path) -> pd.DataFrame:
    html_text = html_path.read_text(encoding="cp932", errors="replace")
    metadata = _parse_metadata(html_text)
    table = pd.read_html(html_path, encoding="cp932")[0]

    out = pd.DataFrame(index=table.index, columns=columns)
    for key, value in metadata.items():
        if key in out.columns:
            out[key] = value

    mappings = {
        "枠": "枠番",
        "番": "馬番",
        "馬名": "馬名",
        "替 騎手": "騎手",
        "斤量": "斤量",
    }
    for source_col, dest_col in mappings.items():
        if source_col in table.columns and dest_col in out.columns:
            out[dest_col] = table[source_col].map(_clean_text)

    if "性齢" in table.columns:
        sex_age = table["性齢"].map(_split_sex_age)
        if "性別" in out.columns:
            out["性別"] = sex_age.map(lambda item: item[0])
        if "年齢" in out.columns:
            out["年齢"] = sex_age.map(lambda item: item[1])

    if "馬名" in out.columns and horses_path.exists():
        horses = pd.read_csv(horses_path, encoding="utf-8-sig", low_memory=False)
        horses = horses.sort_values("日付").drop_duplicates("馬名", keep="last")
        enrich_cols = [
            "馬名",
            "血統登録番号",
            "キャリア",
            "騎手コード",
            "調教師コード",
            "トラックコード",
        ]
        available = [col for col in enrich_cols if col in horses.columns]
        out = out.merge(horses[available], on="馬名", how="left", suffixes=("", "_hist"))
        for col in available:
            hist_col = f"{col}_hist"
            if hist_col in out.columns and col in columns:
                out[col] = out[col].combine_first(out[hist_col])
                out = out.drop(columns=[hist_col])

    if "異常コード" in out.columns:
        out["異常コード"] = 0
    if "確定着順" in out.columns:
        out["確定着順"] = pd.NA
    return out[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TARGET entry HTML into entry_snapshot.csv.")
    parser.add_argument("--input-html", required=True)
    parser.add_argument("--runtime-config", default="config/data_pipeline.json")
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--horses-csv", default="data/processed/normalized/horses_latest.csv")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    runtime = load_runtime_config(args.runtime_config)
    feature_config_path = args.feature_config or runtime["pipeline"]["baseline_feature_config"]
    feature_config = load_json_config(feature_config_path)
    columns = list(dict.fromkeys([
        *inference_required_columns(feature_config),
        *inference_optional_columns(feature_config),
    ]))

    html_path = project_path(args.input_html)
    snapshot = build_snapshot(html_path, columns, project_path(args.horses_csv))
    output_path = project_path(args.output_csv or runtime["datasets"]["weekly_entry_file"])
    ensure_dir(output_path.parent)
    snapshot.to_csv(output_path, index=False, encoding="utf-8-sig")

    missing_required = [
        col for col in inference_required_columns(feature_config)
        if col in snapshot.columns and snapshot[col].isna().all()
    ]
    summary = {
        "input_html": str(html_path),
        "output_csv": str(output_path),
        "rows": int(len(snapshot)),
        "race_id": snapshot["レースID(新/馬番無)"].iloc[0] if "レースID(新/馬番無)" in snapshot else None,
        "matched_horse_ids": int(snapshot["血統登録番号"].notna().sum()) if "血統登録番号" in snapshot else 0,
        "missing_required_values": missing_required,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
