from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MALE = "牡"
GELDING = "セ"

COL_RACE_ID = "レースID(新/馬番無)"
COL_HORSE_ID = "血統登録番号"
COL_HORSE_NO = "馬番"
COL_HORSE_NAME = "馬名"
COL_DATE = "日付"
COL_SEX = "性別"
COL_POPULARITY = "人気"
COL_SURFACE = "芝・ダ"
COL_DISTANCE = "距離"
COL_FINISH = "確定着順"

HISTORY_COLUMNS = [
    "horse_id",
    "race_id",
    "race_date",
    "horse_no",
    "horse_name",
    "sex",
    "finish",
    "surface",
    "distance",
    "gelding_start_no_since_transition",
]

GELDING_FEATURE_COLUMNS = [
    "gelding_phase",
    "gelding_start_no_since_transition",
    "gelding_risk_score",
    "gelding_value_score",
    "gelding_context_note",
    "known_gelding_debut_flag",
    "known_gelding_second_start_flag",
    "known_gelding_third_start_flag",
    "known_gelding_4plus_start_flag",
    "gelding_debut_unpopular_flag",
    "gelding_debut_surface_switch_flag",
    "gelding_second_shorten_flag",
]


def read_csv_any(path: str | Path, **kwargs: object) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def num_series(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def text_series(series: pd.Series | None, index: pd.Index, default: str = "") -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=str)
    return series.astype("string").fillna(default).astype(str)


def clean_race_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(16)


def parse_race_date_from_frame(df: pd.DataFrame, race_id_col: str | None, date_col: str | None) -> pd.Series:
    index = df.index
    parsed = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    if race_id_col:
        race_id = df[race_id_col].astype(str).str.extract(r"(\d{8})", expand=False)
        parsed = pd.to_datetime(race_id, format="%Y%m%d", errors="coerce")
    if date_col:
        raw = df[date_col].astype(str).str.replace(r"\.0$", "", regex=True)
        direct_source = raw.where(raw.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$", na=False))
        direct = pd.to_datetime(direct_source, errors="coerce")
        parsed = parsed.fillna(direct)
        compact = raw.str.replace(r"\D", "", regex=True).str.zfill(6)
        yy = pd.to_numeric(compact.str.slice(0, 2), errors="coerce")
        mm = compact.str.slice(2, 4)
        dd = compact.str.slice(4, 6)
        year = np.where(yy >= 70, 1900 + yy, 2000 + yy)
        ymd = pd.Series(year, index=index).astype("Int64").astype(str) + mm + dd
        parsed = parsed.fillna(pd.to_datetime(ymd, format="%Y%m%d", errors="coerce"))
    return parsed


def normalize_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    race_id_col = find_col(df, [COL_RACE_ID, "race_id"])
    horse_id_col = find_col(df, [COL_HORSE_ID, "horse_id"])
    if not race_id_col or not horse_id_col:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    date_col = find_col(df, [COL_DATE, "date"])
    horse_no_col = find_col(df, [COL_HORSE_NO, "horse_no"])
    horse_name_col = find_col(df, [COL_HORSE_NAME, "horse_name"])
    sex_col = find_col(df, [COL_SEX, "sex"])
    finish_col = find_col(df, [COL_FINISH, "finish"])
    surface_col = find_col(df, [COL_SURFACE, "surface"])
    distance_col = find_col(df, [COL_DISTANCE, "distance"])

    out = pd.DataFrame(index=df.index)
    out["horse_id"] = df[horse_id_col].astype("string").str.replace(r"\.0$", "", regex=True).fillna("")
    out["race_id"] = clean_race_id(df[race_id_col])
    out["race_date"] = parse_race_date_from_frame(df, race_id_col, date_col)
    out["horse_no"] = num_series(df[horse_no_col] if horse_no_col else None, df.index)
    out["horse_name"] = text_series(df[horse_name_col] if horse_name_col else None, df.index)
    out["sex"] = text_series(df[sex_col] if sex_col else None, df.index)
    out["finish"] = num_series(df[finish_col] if finish_col else None, df.index)
    out["surface"] = text_series(df[surface_col] if surface_col else None, df.index)
    out["distance"] = num_series(df[distance_col] if distance_col else None, df.index)
    out = out[out["horse_id"].ne("") & out["race_date"].notna()].copy()
    out = out.drop_duplicates(["horse_id", "race_id"], keep="last")
    return out


def add_history_transition_counts(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    work = history.sort_values(["horse_id", "race_date", "race_id"]).copy()
    prev_sex = work.groupby("horse_id", sort=False)["sex"].shift(1)
    is_gelding = work["sex"].eq(GELDING)
    known_debut = (is_gelding & prev_sex.eq(MALE)).fillna(False)
    transition_no = known_debut.astype(int).groupby(work["horse_id"], sort=False).cumsum()
    after_known_transition = is_gelding & transition_no.gt(0)
    work["gelding_start_no_since_transition"] = after_known_transition.astype(int).groupby(
        [work["horse_id"], transition_no],
        sort=False,
    ).cumsum()
    work.loc[~after_known_transition, "gelding_start_no_since_transition"] = 0
    return work[HISTORY_COLUMNS].copy()


def build_gelding_history(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        header = read_csv_any(p, nrows=0)
        usecols = [
            col
            for col in [
                COL_DATE,
                COL_RACE_ID,
                COL_HORSE_ID,
                COL_HORSE_NO,
                COL_HORSE_NAME,
                COL_SEX,
                COL_FINISH,
                COL_SURFACE,
                COL_DISTANCE,
            ]
            if col in header.columns
        ]
        if COL_RACE_ID not in usecols or COL_HORSE_ID not in usecols:
            continue
        frames.append(normalize_history_frame(read_csv_any(p, usecols=usecols)))
    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    history = pd.concat(frames, ignore_index=True)
    history = history.drop_duplicates(["horse_id", "race_id"], keep="last")
    return add_history_transition_counts(history)


def _current_base(entry: pd.DataFrame) -> pd.DataFrame:
    race_id_col = find_col(entry, [COL_RACE_ID, "race_id"])
    horse_id_col = find_col(entry, [COL_HORSE_ID, "horse_id"])
    if not race_id_col or not horse_id_col:
        return pd.DataFrame(index=entry.index)
    date_col = find_col(entry, [COL_DATE, "date"])
    horse_no_col = find_col(entry, [COL_HORSE_NO, "horse_no"])
    sex_col = find_col(entry, [COL_SEX, "sex"])
    popularity_col = find_col(entry, [COL_POPULARITY, "popularity"])
    surface_col = find_col(entry, [COL_SURFACE, "surface"])
    distance_col = find_col(entry, [COL_DISTANCE, "distance"])

    cur = pd.DataFrame(index=entry.index)
    cur["_entry_order"] = np.arange(len(entry))
    cur["horse_id"] = entry[horse_id_col].astype("string").str.replace(r"\.0$", "", regex=True).fillna("")
    cur["race_id"] = clean_race_id(entry[race_id_col])
    cur["race_date"] = parse_race_date_from_frame(entry, race_id_col, date_col)
    cur["horse_no"] = num_series(entry[horse_no_col] if horse_no_col else None, entry.index)
    cur["sex"] = text_series(entry[sex_col] if sex_col else None, entry.index)
    cur["popularity"] = num_series(entry[popularity_col] if popularity_col else None, entry.index)
    cur["surface"] = text_series(entry[surface_col] if surface_col else None, entry.index)
    cur["distance"] = num_series(entry[distance_col] if distance_col else None, entry.index)
    return cur


def enrich_current_entries_with_gelding_context(entry: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = entry.copy()
    for col in GELDING_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col.endswith("_flag") or col.endswith("_score") or col == "gelding_start_no_since_transition" else ""
    if entry.empty or history.empty:
        out["gelding_phase"] = out["gelding_phase"].replace("", "unknown")
        return out

    cur = _current_base(entry)
    if cur.empty:
        out["gelding_phase"] = out["gelding_phase"].replace("", "unknown")
        return out
    hist = history.copy()
    hist["horse_id"] = hist["horse_id"].astype("string").str.replace(r"\.0$", "", regex=True).fillna("")
    hist["race_date"] = pd.to_datetime(hist["race_date"], errors="coerce")
    hist = hist[hist["horse_id"].ne("") & hist["race_date"].notna()].copy()
    hist = hist.sort_values(["race_date", "horse_id", "race_id"])
    cur_valid = cur[cur["horse_id"].ne("") & cur["race_date"].notna()].sort_values(["race_date", "horse_id"]).copy()
    if cur_valid.empty:
        out["gelding_phase"] = out["gelding_phase"].replace("", "unknown")
        return out

    merged = pd.merge_asof(
        cur_valid,
        hist.rename(
            columns={
                "race_date": "prev_race_date",
                "sex": "prev_sex",
                "finish": "prev_finish",
                "surface": "prev_surface",
                "distance": "prev_distance",
                "gelding_start_no_since_transition": "prev_gelding_start_no_since_transition",
            }
        ).sort_values(["prev_race_date", "horse_id"]),
        by="horse_id",
        left_on="race_date",
        right_on="prev_race_date",
        allow_exact_matches=False,
        direction="backward",
    )

    cur_sex = merged["sex"].astype("string").fillna("")
    prev_sex = merged["prev_sex"].astype("string").fillna("")
    is_gelding = cur_sex.eq(GELDING)
    prev_start_no = pd.to_numeric(merged["prev_gelding_start_no_since_transition"], errors="coerce").fillna(0).astype(int)
    known_debut = is_gelding & prev_sex.eq(MALE)
    second = is_gelding & prev_sex.eq(GELDING) & prev_start_no.eq(1)
    third = is_gelding & prev_sex.eq(GELDING) & prev_start_no.eq(2)
    four_plus = is_gelding & prev_sex.eq(GELDING) & prev_start_no.ge(3)
    established = is_gelding & prev_sex.eq(GELDING) & prev_start_no.eq(0)
    first_seen = is_gelding & prev_sex.eq("")

    merged["gelding_start_no_since_transition"] = 0
    merged.loc[known_debut, "gelding_start_no_since_transition"] = 1
    merged.loc[second, "gelding_start_no_since_transition"] = 2
    merged.loc[third, "gelding_start_no_since_transition"] = 3
    merged.loc[four_plus, "gelding_start_no_since_transition"] = (prev_start_no + 1).clip(upper=4)
    merged["gelding_phase"] = "non_gelding"
    merged.loc[first_seen, "gelding_phase"] = "first_seen_as_gelding_unknown_timing"
    merged.loc[established, "gelding_phase"] = "established_gelding_unknown_transition"
    merged.loc[known_debut, "gelding_phase"] = "known_gelding_debut"
    merged.loc[second, "gelding_phase"] = "known_gelding_second_start"
    merged.loc[third, "gelding_phase"] = "known_gelding_third_start"
    merged.loc[four_plus, "gelding_phase"] = "known_gelding_4plus_start"

    popularity = pd.to_numeric(merged["popularity"], errors="coerce")
    surface_switch = (
        merged["surface"].astype("string").fillna("").ne(merged["prev_surface"].astype("string").fillna(""))
        & merged["prev_surface"].notna()
        & merged["prev_surface"].astype("string").fillna("").ne("")
    )
    distance_diff = pd.to_numeric(merged["distance"], errors="coerce") - pd.to_numeric(merged["prev_distance"], errors="coerce")
    shorten = distance_diff.le(-200)
    prev_finish = pd.to_numeric(merged["prev_finish"], errors="coerce")

    merged["known_gelding_debut_flag"] = known_debut.astype(float)
    merged["known_gelding_second_start_flag"] = second.astype(float)
    merged["known_gelding_third_start_flag"] = third.astype(float)
    merged["known_gelding_4plus_start_flag"] = four_plus.astype(float)
    merged["gelding_debut_unpopular_flag"] = (known_debut & popularity.ge(4)).astype(float)
    merged["gelding_debut_surface_switch_flag"] = (known_debut & surface_switch).astype(float)
    merged["gelding_second_shorten_flag"] = (second & shorten).astype(float)

    risk = pd.Series(0.0, index=merged.index)
    value = pd.Series(0.0, index=merged.index)
    risk += known_debut.astype(float) * 0.18
    risk += (known_debut & popularity.ge(4)).astype(float) * 0.20
    risk += (known_debut & surface_switch).astype(float) * 0.25
    risk += (known_debut & prev_finish.ge(6)).astype(float) * 0.08
    risk += first_seen.astype(float) * 0.22
    value += (known_debut & popularity.between(1, 3)).astype(float) * 0.04
    value += second.astype(float) * 0.07
    value += (second & shorten).astype(float) * 0.04
    value += third.astype(float) * 0.02
    merged["gelding_risk_score"] = risk.clip(0.0, 1.0)
    merged["gelding_value_score"] = value.clip(0.0, 1.0)

    notes = []
    for _, row in merged.iterrows():
        phase = str(row.get("gelding_phase", ""))
        if phase == "known_gelding_debut":
            if row.get("gelding_debut_surface_switch_flag", 0) >= 1:
                notes.append("去勢明け初戦＋芝ダ替わりは強く割引")
            elif row.get("gelding_debut_unpopular_flag", 0) >= 1:
                notes.append("去勢明け初戦の人気薄は割引")
            else:
                notes.append("去勢明け初戦だが市場支持あり")
        elif phase == "known_gelding_second_start":
            if row.get("gelding_second_shorten_flag", 0) >= 1:
                notes.append("去勢2戦目＋距離短縮は妙味候補")
            else:
                notes.append("去勢2戦目で上積み候補")
        elif phase == "known_gelding_third_start":
            notes.append("去勢3戦目は相手・複系で注意")
        elif phase == "first_seen_as_gelding_unknown_timing":
            notes.append("セ馬初見で手術時期不明")
        else:
            notes.append("")
    merged["gelding_context_note"] = notes

    feature_map = merged.set_index("_entry_order")[GELDING_FEATURE_COLUMNS]
    for col in GELDING_FEATURE_COLUMNS:
        out.loc[feature_map.index, col] = feature_map[col]
    out["gelding_phase"] = out["gelding_phase"].replace("", "unknown")
    numeric_cols = [c for c in GELDING_FEATURE_COLUMNS if c.endswith("_flag") or c.endswith("_score") or c == "gelding_start_no_since_transition"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out
