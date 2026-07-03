from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


BET_TYPE_BY_DATASPEC = {
    "0B32": "umaren",
    "0B33": "wide",
    "0B34": "umatan",
}


def _read_text(path: Path) -> str:
    for encoding in ["cp932", "utf-8-sig", "utf-16"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    return path.read_text(errors="ignore")


def _read_meta(path: Path) -> dict:
    text = _read_text(path)
    return json.loads(text) if text.strip() else {}


def _iter_snapshot_files(raw_dir: Path) -> list[tuple[Path, dict]]:
    rows: list[tuple[Path, dict]] = []
    for meta_path in raw_dir.glob("*/*.json"):
        try:
            meta = _read_meta(meta_path)
        except Exception:
            continue
        raw_path = Path(str(meta.get("raw_path") or ""))
        if not raw_path.exists():
            raw_path = meta_path.with_suffix(".txt")
        if raw_path.exists():
            rows.append((raw_path, meta))
    return rows


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def _pairs_for_field_size(field_size: int = 18) -> list[tuple[int, int]]:
    return [(a, b) for a in range(1, field_size + 1) for b in range(a + 1, field_size + 1)]


def _parse_numeric_tokens_record(record: str, *, race_id: str, ticket_type: str, snapshot_at: str) -> list[dict]:
    """Best-effort parser for pair odds records.

    JV/Data Lab O2/O3/O4 records are fixed-width, but live samples were not
    available when this parser was added. This fallback extracts pair-number
    tokens such as 0105 followed by plausible odds/pay tokens. It intentionally
    marks rows as parser_mode=heuristic so they can be audited before live use.
    """

    rows = []
    compact = re.sub(r"\s+", " ", record)
    pattern = re.compile(r"(?<!\d)([01]\d[01]\d)(?!\d)[^\d]{0,8}(\d{2,6})(?:[^\d]{0,8}(\d{1,4}))?")
    for match in pattern.finditer(compact):
        pair = match.group(1)
        a = int(pair[:2])
        b = int(pair[2:])
        if not (1 <= a <= 18 and 1 <= b <= 18 and a != b):
            continue
        pay = int(match.group(2))
        if pay < 100:
            pay *= 10
        rows.append(
            {
                "race_id": race_id,
                "ticket_type": ticket_type,
                "a_no": min(a, b),
                "b_no": max(a, b),
                "live_pay_per100": float(pay),
                "live_odds": float(pay) / 100.0,
                "popularity": int(match.group(3)) if match.group(3) else pd.NA,
                "snapshot_at": snapshot_at,
                "parser_mode": "heuristic_pair_token",
            }
        )
    return rows


def _parse_dense_pair_odds(record: str, *, race_id: str, ticket_type: str, snapshot_at: str) -> list[dict]:
    """Fallback for dense fixed-width numeric blocks.

    Many JRA-VAN odds tables store all combinations in pair order. If a record
    contains enough numeric-only payload, treat consecutive 4-digit chunks as
    payout per 100 yen for 18-horse pair order. This is conservative and only
    emits rows when at least 60 plausible chunks are present.
    """

    digits = _digits(record)
    if len(digits) < 240:
        return []

    # Skip a likely header prefix by searching for the first long run that yields
    # enough plausible pair payouts. This avoids hardcoding a record layout until
    # a successful live sample can be audited.
    pairs = _pairs_for_field_size(18)
    best_rows: list[dict] = []
    for offset in range(0, min(120, len(digits) - 240)):
        chunks = [digits[i : i + 4] for i in range(offset, min(len(digits), offset + len(pairs) * 4), 4)]
        if len(chunks) < 60:
            continue
        values = [int(c) for c in chunks if len(c) == 4]
        plausible = [v for v in values if 100 <= v <= 9999]
        if len(plausible) < 60:
            continue
        rows = []
        for (a, b), value in zip(pairs, values):
            if value <= 0:
                continue
            rows.append(
                {
                    "race_id": race_id,
                    "ticket_type": ticket_type,
                    "a_no": a,
                    "b_no": b,
                    "live_pay_per100": float(value),
                    "live_odds": float(value) / 100.0,
                    "popularity": pd.NA,
                    "snapshot_at": snapshot_at,
                    "parser_mode": f"dense_4digit_offset_{offset}",
                }
            )
        if len(rows) > len(best_rows):
            best_rows = rows
    return best_rows


def _parse_raw_file(raw_path: Path, meta: dict) -> pd.DataFrame:
    dataspec = str(meta.get("dataspec") or "")
    ticket_type = str(meta.get("bet_type") or BET_TYPE_BY_DATASPEC.get(dataspec) or "")
    if ticket_type not in {"wide", "umaren", "umatan"}:
        return pd.DataFrame()
    race_id = str(meta.get("race_key") or raw_path.parent.name)
    snapshot_at = str(meta.get("snapshot_at") or raw_path.stem.split("_")[0])
    text = _read_text(raw_path)
    records = [line for line in text.splitlines() if line.strip()]
    rows: list[dict] = []
    for record in records:
        parsed = _parse_numeric_tokens_record(record, race_id=race_id, ticket_type=ticket_type, snapshot_at=snapshot_at)
        if not parsed:
            parsed = _parse_dense_pair_odds(record, race_id=race_id, ticket_type=ticket_type, snapshot_at=snapshot_at)
        rows.extend(parsed)
    return pd.DataFrame(rows)


def _normalize_manual_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
    rename = {
        "bet": "ticket_type",
        "type": "ticket_type",
        "horse_a": "a_no",
        "horse_b": "b_no",
        "umaban_a": "a_no",
        "umaban_b": "b_no",
        "odds": "live_odds",
        "pay": "live_pay_per100",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = {"race_id", "ticket_type", "a_no", "b_no"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manual csv missing columns: {sorted(missing)}")
    if "live_pay_per100" not in df.columns:
        if "live_odds" not in df.columns:
            raise ValueError("manual csv needs live_pay_per100 or live_odds")
        df["live_pay_per100"] = pd.to_numeric(df["live_odds"], errors="coerce") * 100.0
    if "live_odds" not in df.columns:
        df["live_odds"] = pd.to_numeric(df["live_pay_per100"], errors="coerce") / 100.0
    if "snapshot_at" not in df.columns:
        df["snapshot_at"] = pd.NA
    df["parser_mode"] = "manual_or_external_csv"
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["race_id", "ticket_type", "a_no", "b_no", "live_pay_per100", "live_odds", "popularity", "snapshot_at", "parser_mode"]
        )
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["ticket_type"] = out["ticket_type"].astype(str)
    out["a_no"] = pd.to_numeric(out["a_no"], errors="coerce")
    out["b_no"] = pd.to_numeric(out["b_no"], errors="coerce")
    lo = np.minimum(out["a_no"], out["b_no"])
    hi = np.maximum(out["a_no"], out["b_no"])
    out["a_no"] = lo.astype("Int64")
    out["b_no"] = hi.astype("Int64")
    out["live_pay_per100"] = pd.to_numeric(out["live_pay_per100"], errors="coerce")
    out["live_odds"] = pd.to_numeric(out["live_odds"], errors="coerce")
    out = out[out["race_id"].notna() & out["a_no"].notna() & out["b_no"].notna() & out["live_pay_per100"].gt(0)].copy()
    out = out.sort_values(["race_id", "ticket_type", "snapshot_at", "a_no", "b_no"])
    out = out.drop_duplicates(["race_id", "ticket_type", "a_no", "b_no"], keep="last")
    return out[["race_id", "ticket_type", "a_no", "b_no", "live_pay_per100", "live_odds", "popularity", "snapshot_at", "parser_mode"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize JV/Data Lab realtime pair odds to live-odds CSV for betting gates.")
    parser.add_argument("--raw-dir", default="data/raw/jv_realtime_odds")
    parser.add_argument("--manual-csv", default=None, help="Optional already-exported odds CSV to normalize.")
    parser.add_argument("--output-csv", default="data/processed/live_odds/realtime_pair_odds_latest.csv")
    parser.add_argument("--summary-json", default="outputs/analysis/live_odds_normalization/summary.json")
    args = parser.parse_args()

    frames = []
    if args.manual_csv:
        frames.append(_normalize_manual_csv(project_path(args.manual_csv)))
    for raw_path, meta in _iter_snapshot_files(project_path(args.raw_dir)):
        frames.append(_parse_raw_file(raw_path, meta))
    normalized = _finalize(pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame())

    output_csv = project_path(args.output_csv)
    ensure_dir(output_csv.parent)
    normalized.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary_path = project_path(args.summary_json)
    ensure_dir(summary_path.parent)
    payload = {
        "output_csv": str(output_csv),
        "rows": int(len(normalized)),
        "by_ticket_type": normalized["ticket_type"].value_counts().to_dict() if not normalized.empty else {},
        "parser_modes": normalized["parser_mode"].value_counts().to_dict() if not normalized.empty else {},
        "warning": "Heuristic JV raw parsing must be audited against a successful live sample before production betting.",
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
