from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
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


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def dec(raw: bytes, start: int, end: int) -> str:
    return raw[start:end].decode("cp932", errors="replace").replace("\u3000", "").strip()


def num(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_du(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return pd.DataFrame()
    for raw in path.read_bytes().splitlines():
        if len(raw) < 132:
            continue
        record_type = dec(raw, 0, 3)
        update_date = dec(raw, 3, 11)
        race_date = dec(raw, 11, 19)
        race_part = dec(raw, 19, 27)
        if len(race_part) != 8:
            continue
        jyo = race_part[:2]
        kaiji = race_part[2:4]
        nichiji = race_part[4:6]
        race_no = race_part[6:8]
        race_id = f"{race_date}{race_part}"
        frame_no = dec(raw, 27, 28)
        horse_no = dec(raw, 28, 30)
        weight_raw = dec(raw, 104, 109)
        rows.append(
            {
                "record_type": record_type,
                "is_active_runner": record_type == "SE2" and horse_no not in {"", "00"},
                "update_date": update_date,
                "race_date": race_date,
                "race_id": race_id,
                "venue_code": jyo,
                "venue": VENUES.get(jyo, jyo),
                "kaiji": kaiji,
                "nichiji": nichiji,
                "race_no": int(race_no) if race_no.isdigit() else race_no,
                "frame_no": int(frame_no) if frame_no.isdigit() else pd.NA,
                "horse_no": int(horse_no) if horse_no.isdigit() else pd.NA,
                "horse_id": ("20" + dec(raw, 32, 40)) if len(dec(raw, 32, 40)) == 8 else dec(raw, 32, 40),
                "horse_name": dec(raw, 40, 76),
                "age": int(dec(raw, 82, 84)) if dec(raw, 82, 84).isdigit() else pd.NA,
                "trainer_code": dec(raw, 86, 90),
                "trainer_name_raw": dec(raw, 90, 100),
                "assigned_weight_kg": (num(weight_raw) / 1000.0) if num(weight_raw) is not None else pd.NA,
                "jockey_code": dec(raw, 112, 117),
                "jockey_name": dec(raw, 122, 132),
                "tail_raw": dec(raw, 132, 157),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["race_id", "is_active_runner", "horse_no"], ascending=[True, False, True])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a safe sidecar entry card from TARGET DE_DATA DU*.DAT.")
    parser.add_argument("--target-root", default="C:/Users/kazup/Data Lab")
    parser.add_argument("--date", required=True, help="YYYYMMDD race date.")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    du_path = Path(args.target_root) / "DE_DATA" / args.date[:4] / f"DU{args.date}.DAT"
    output_csv = project_path(args.output_csv or f"data/processed/target/entry_card_de_{args.date}.csv")
    summary_json = project_path(args.summary_json or f"outputs/analysis/target_de_entry_card_{args.date}.json")

    card = parse_du(du_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    card.to_csv(output_csv, index=False, encoding="utf-8-sig")

    active = card[card["is_active_runner"].eq(True)] if not card.empty else card
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": args.date,
        "source_du": str(du_path),
        "output_csv": str(output_csv),
        "rows": int(len(card)),
        "active_runners": int(len(active)),
        "races": int(active["race_id"].nunique()) if not active.empty else 0,
        "record_type_counts": card["record_type"].value_counts().to_dict() if not card.empty else {},
        "active_runners_by_venue": active["venue"].value_counts().to_dict() if not active.empty else {},
        "note": "Sidecar extraction only. This does not overwrite weekly entry_snapshot.csv.",
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
