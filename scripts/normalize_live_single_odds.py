from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _project_path(path: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    p = Path(path)
    return p if p.is_absolute() else root / p


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _read_text(path: Path) -> str:
    for encoding in ("cp932", "utf-8-sig", "utf-16", "utf-8"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    return path.read_text(errors="ignore")


def _read_meta(path: Path) -> dict:
    text = _read_text(path)
    return json.loads(text) if text.strip() else {}


def _normalize_manual_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
    rename = {
        "馬番": "horse_no",
        "umaban": "horse_no",
        "horse_number": "horse_no",
        "単勝": "live_win_odds",
        "単勝オッズ": "live_win_odds",
        "win_odds": "live_win_odds",
        "odds": "live_win_odds",
        "人気": "live_popularity",
        "単勝人気": "live_popularity",
        "popularity": "live_popularity",
        "複勝下限": "live_place_odds_min",
        "place_odds_min": "live_place_odds_min",
        "複勝上限": "live_place_odds_max",
        "place_odds_max": "live_place_odds_max",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = {"race_id", "horse_no"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manual single odds csv missing columns: {sorted(missing)}")
    if "live_win_odds" not in df.columns:
        df["live_win_odds"] = pd.NA
    if "live_popularity" not in df.columns:
        df["live_popularity"] = pd.NA
    if "live_place_odds_min" not in df.columns:
        df["live_place_odds_min"] = pd.NA
    if "live_place_odds_max" not in df.columns:
        df["live_place_odds_max"] = pd.NA
    if "snapshot_at" not in df.columns:
        df["snapshot_at"] = pd.NA
    df["parser_mode"] = "manual_or_external_csv"
    return df


def _parse_win_place_record(record: str, *, race_id: str, snapshot_at: str) -> list[dict]:
    """Best-effort parser for JV 0B31 win/place records.

    This is intentionally conservative. It looks for repeated horse-number +
    odds-like numeric groups. Successful live samples should still be audited
    before production betting uses the parsed values.
    """

    rows: list[dict] = []
    compact = re.sub(r"\s+", " ", record)
    pattern = re.compile(r"(?<!\d)([01]\d)(?!\d)[^\d]{0,8}(\d{2,5})(?:[^\d]{0,8}(\d{1,2}))?")
    seen: set[int] = set()
    for match in pattern.finditer(compact):
        horse_no = int(match.group(1))
        if not (1 <= horse_no <= 18) or horse_no in seen:
            continue
        raw_odds = int(match.group(2))
        odds = raw_odds / 10.0 if raw_odds >= 10 else float(raw_odds)
        if not (1.0 <= odds < 999.0):
            continue
        seen.add(horse_no)
        rows.append(
            {
                "race_id": race_id,
                "horse_no": horse_no,
                "live_win_odds": odds,
                "live_popularity": int(match.group(3)) if match.group(3) else pd.NA,
                "live_place_odds_min": pd.NA,
                "live_place_odds_max": pd.NA,
                "snapshot_at": snapshot_at,
                "parser_mode": "heuristic_win_place_token",
            }
        )
    return rows


def _iter_raw_frames(raw_dir: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for meta_path in raw_dir.glob("*/*.json"):
        try:
            meta = _read_meta(meta_path)
        except Exception:
            continue
        dataspec = str(meta.get("dataspec") or "")
        bet_type = str(meta.get("bet_type") or "")
        if dataspec != "0B31" and bet_type != "win_place_frame":
            continue
        raw_path = Path(str(meta.get("raw_path") or ""))
        if not raw_path.exists():
            raw_path = meta_path.with_suffix(".txt")
        if not raw_path.exists():
            continue
        race_id = str(meta.get("race_key") or raw_path.parent.name)
        snapshot_at = str(meta.get("snapshot_at") or raw_path.stem.split("_")[0])
        rows: list[dict] = []
        for record in _read_text(raw_path).splitlines():
            if record.strip():
                rows.extend(_parse_win_place_record(record, race_id=race_id, snapshot_at=snapshot_at))
        if rows:
            frames.append(pd.DataFrame(rows))
    return frames


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "race_id",
        "horse_no",
        "live_win_odds",
        "live_popularity",
        "live_place_odds_min",
        "live_place_odds_max",
        "snapshot_at",
        "parser_mode",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["horse_no"] = _num(out["horse_no"]).astype("Int64")
    for col in ["live_win_odds", "live_popularity", "live_place_odds_min", "live_place_odds_max"]:
        out[col] = _num(out[col]) if col in out.columns else pd.NA
    odds_rank = out.groupby("race_id")["live_win_odds"].rank(method="min", ascending=True)
    out["live_popularity"] = out["live_popularity"].where(out["live_popularity"].notna(), odds_rank)
    out = out[out["race_id"].notna() & out["horse_no"].notna()].copy()
    out = out.sort_values(["race_id", "snapshot_at", "horse_no"])
    out = out.drop_duplicates(["race_id", "horse_no"], keep="last")
    return out[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize realtime win/place odds to runner-level live odds CSV.")
    parser.add_argument("--raw-dir", default="data/raw/jv_realtime_odds")
    parser.add_argument("--manual-csv", default="")
    parser.add_argument("--output-csv", default="data/processed/live_odds/realtime_single_odds_latest.csv")
    parser.add_argument("--summary-json", default="outputs/analysis/live_odds_normalization/single_summary.json")
    args = parser.parse_args()

    frames = []
    if args.manual_csv:
        frames.append(_normalize_manual_csv(_project_path(args.manual_csv)))
    frames.extend(_iter_raw_frames(_project_path(args.raw_dir)))
    out = _finalize(pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame())

    output = _project_path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")

    summary = {
        "output_csv": str(output),
        "rows": int(len(out)),
        "parser_modes": out["parser_mode"].value_counts().to_dict() if not out.empty else {},
        "warning": "JV raw win/place parser is heuristic until a successful live sample is audited. Manual/TARGET CSV is preferred for production.",
    }
    summary_path = _project_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
