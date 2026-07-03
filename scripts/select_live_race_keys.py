from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VENUE_CODE_BY_NAME = {
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


def _project_path(path: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    p = Path(path)
    return p if p.is_absolute() else root / p


def _date_key(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits[:8] if len(digits) >= 8 else text
    return parsed.strftime("%Y%m%d")


def _race_key_from_row(row: pd.Series) -> str:
    race_id = str(row.get("race_id", "")).strip()
    race_digits = "".join(ch for ch in race_id if ch.isdigit())
    if len(race_digits) >= 16:
        return race_digits[:16]
    date = _date_key(row.get("日付S", race_id[:8]))
    venue = str(row.get("venue", "")).strip()
    venue_code = VENUE_CODE_BY_NAME.get(venue)
    if not venue_code and len(race_id) >= 10:
        venue_code = race_id[8:10]
    race_no = int(float(row.get("Ｒ", race_id[-2:] if len(race_id) >= 2 else 0)))
    return f"{date}{venue_code}{race_no:02d}" if date and venue_code and race_no else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Select JRA-VAN realtime odds race keys from a scored runner CSV.")
    parser.add_argument("--scored-csv", default="outputs/analysis/risk_models_v1/investment_features_with_risk_models.csv")
    parser.add_argument("--date", default="", help="YYYY-MM-DD, YYYYMMDD, or a date present in 日付S. Empty means all dates in the CSV.")
    parser.add_argument("--venues", nargs="*", default=[], help="Venue names such as 東京 新潟. Empty means all venues.")
    parser.add_argument("--races", nargs="*", type=int, default=[], help="Race numbers. Empty means all races.")
    parser.add_argument("--output-json", default="", help="Optional path to write selected race metadata.")
    args = parser.parse_args()

    df = pd.read_csv(_project_path(args.scored_csv), dtype={"race_id": str}, low_memory=False)
    needed = ["race_id", "日付S", "venue", "Ｒ", "レース名"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    races = df[needed].drop_duplicates("race_id").copy()
    races["date_key"] = races["日付S"].map(_date_key)
    races["race_no"] = pd.to_numeric(races["Ｒ"], errors="coerce").fillna(0).astype(int)
    if args.date:
        target_date = _date_key(args.date)
        races = races[races["date_key"].eq(target_date)].copy()
    if args.venues:
        races = races[races["venue"].isin(args.venues)].copy()
    if args.races:
        races = races[races["race_no"].isin(args.races)].copy()

    if races.empty:
        payload = []
        if args.output_json:
            out = _project_path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("")
        return

    races["race_key"] = [_race_key_from_row(row) for _, row in races.iterrows()]
    races = races[races["race_key"].ne("")].sort_values(["date_key", "venue", "race_no"])
    payload = [
        {
            "race_key": row.race_key,
            "race_id": row.race_id,
            "date": row.date_key,
            "venue": row.venue,
            "race_no": int(row.race_no),
            "race_name": str(row.レース名),
        }
        for row in races.itertuples(index=False)
    ]

    if args.output_json:
        out = _project_path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(",".join(item["race_key"] for item in payload))


if __name__ == "__main__":
    main()
