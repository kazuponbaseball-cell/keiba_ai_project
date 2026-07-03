from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"[\s_\-()\[\]/　]+", "", text)


def _num(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip().replace(",", "")
    if not text or text in {"0", "00", "000", "0000", "000000000"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def _read_csv(path: Path, encoding: str | None) -> pd.DataFrame:
    encodings = [encoding] if encoding else []
    encodings.extend(["utf-8-sig", "cp932", "shift_jis", "utf-8"])
    seen: set[str] = set()
    for enc in encodings:
        if not enc or enc in seen:
            continue
        seen.add(enc)
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, low_memory=False)


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {_normalize_name(col): col for col in df.columns}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = _column_lookup(df)
    for candidate in candidates:
        key = _normalize_name(candidate)
        if key in lookup:
            return lookup[key]
    return None


def _find_wide_col(df: pd.DataFrame, index: int, suffix: str) -> str | None:
    candidates = [
        f"haraimodoshi_wide_{index}{suffix}",
        f"ワイド払戻{index}{suffix}",
        f"ワイド払戻{index}{suffix.upper()}",
        f"ワイド{index}{suffix}",
        f"wide_{index}{suffix}",
        f"wide{index}{suffix}",
    ]
    found = _find_col(df, candidates)
    if found:
        return found

    suffix_key = suffix.lower()
    for col in df.columns:
        key = _normalize_name(col)
        if f"wide{index}{suffix_key}" in key or f"ワイド払戻{index}{suffix_key}" in key:
            return col
    return None


def _normalize_pair(value: object) -> tuple[int | None, int | None]:
    text = _text(value).replace("-", "").replace(" ", "").replace("　", "")
    if "." in text:
        text = text.split(".", 1)[0]
    text = text.zfill(4)
    if not text.strip("0"):
        return None, None
    try:
        horse_a = int(text[:2])
        horse_b = int(text[2:4])
    except ValueError:
        return None, None
    if horse_a <= 0 or horse_b <= 0:
        return None, None
    return tuple(sorted((horse_a, horse_b)))


def _build_race_id(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    race_col = _find_col(frame, ["race_id", "race_id_without_horse", "レースID(新/馬番無)", "レースid"])
    if race_col:
        return frame, race_col

    year = _find_col(frame, ["kaisai_nen", "開催年"])
    mmdd = _find_col(frame, ["kaisai_tsukihi", "開催月日"])
    venue = _find_col(frame, ["keibajo_code", "競馬場コード"])
    race_no = _find_col(frame, ["race_bango", "レース番号"])
    if not all([year, mmdd, venue, race_no]):
        raise ValueError(
            "Race key columns are missing. Need race_id or 開催年/開催月日/競馬場コード/レース番号."
        )

    out = frame.copy()
    out["race_id"] = (
        out[year].map(_text).str.zfill(4)
        + out[mmdd].map(_text).str.zfill(4)
        + out[venue].map(_text).str.zfill(2)
        + out[race_no].map(_text).str.zfill(2)
    )
    return out, "race_id"


def normalize_wide_payoffs(frame: pd.DataFrame) -> pd.DataFrame:
    frame, race_col = _build_race_id(frame)
    wide_columns = {
        i: (
            _find_wide_col(frame, i, "a"),
            _find_wide_col(frame, i, "b"),
            _find_wide_col(frame, i, "c"),
        )
        for i in range(1, 8)
    }
    if not any(pair_col and pay_col for pair_col, pay_col, _ in wide_columns.values()):
        available = ", ".join(map(str, frame.columns[:30]))
        raise ValueError(f"No wide payoff columns found. First columns: {available}")

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        race_id = _text(row[race_col])
        if not race_id:
            continue
        for pair_col, pay_col, pop_col in wide_columns.values():
            if pair_col is None or pay_col is None:
                continue
            horse_a, horse_b = _normalize_pair(row[pair_col])
            pay = _num(row[pay_col])
            if horse_a is None or horse_b is None or pay is None:
                continue
            popularity = _num(row[pop_col]) if pop_col else None
            rows.append(
                {
                    "race_id": race_id,
                    "horse_a": horse_a,
                    "horse_b": horse_b,
                    "wide_pay": int(pay),
                    "wide_popularity": int(popularity) if popularity is not None else None,
                }
            )

    out = pd.DataFrame(rows, columns=["race_id", "horse_a", "horse_b", "wide_pay", "wide_popularity"])
    if not out.empty:
        out = out.drop_duplicates(["race_id", "horse_a", "horse_b"], keep="first")
        out = out.sort_values(["race_id", "horse_a", "horse_b"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize TARGET/PC-KEIBA jvd_hr wide payoff CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--encoding", default=None)
    parser.add_argument("--output-csv", default="data/processed/target/wide_payoffs.csv")
    args = parser.parse_args()

    src = Path(args.input_csv)
    frame = _read_csv(src, args.encoding)
    out = normalize_wide_payoffs(frame)
    dst = Path(args.output_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"rows={len(out)} output={dst}")


if __name__ == "__main__":
    main()
