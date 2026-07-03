from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEATURES = [
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
    "body_weight_backfilled/owner_breeder_enriched/train_features_with_same_day_bias_v3_retro_body_breeder.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
    "body_weight_backfilled/owner_breeder_enriched/test_features_with_same_day_bias_v3_retro_body_breeder.csv",
]
DEFAULT_FRONT3F = "outputs/analysis/estimated_front3f_race_quality_v1/estimated_runner_front3f_used.csv"
DEFAULT_OUT = "outputs/analysis/historical_condition_lap_context_v1/condition_lap_baselines.csv"


RACE_COL = "レースID(新/馬番無)"
DATE_COL = "日付"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv_any(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def to_num(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    raw = str(value).strip().replace(",", "")
    if not raw:
        return np.nan
    try:
        return float(raw)
    except Exception:
        pass
    match = re.match(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$", raw)
    if match:
        minutes = float(match.group(1) or 0.0)
        seconds = float(match.group(2))
        return minutes * 60.0 + seconds
    return np.nan


def parse_date(value: object) -> pd.Timestamp:
    raw = "" if value is None or pd.isna(value) else str(value).strip()
    if not raw:
        return pd.NaT
    raw = raw.replace(".0", "")
    if re.fullmatch(r"\d{6}", raw):
        try:
            return pd.to_datetime(raw, format="%y%m%d")
        except Exception:
            pass
    for fmt in ("%Y%m%d", "%y%m%d", "%Y.%m.%d", "%Y-%m-%d"):
        try:
            dt = pd.to_datetime(raw, format=fmt)
            return dt
        except Exception:
            continue
    return pd.to_datetime(raw, errors="coerce")


def class_group(value: object) -> str:
    raw = "" if value is None or pd.isna(value) else str(value)
    if "新馬" in raw:
        return "新馬"
    if "未勝利" in raw:
        return "未勝利"
    if "1勝" in raw or "500万" in raw:
        return "1勝"
    if "2勝" in raw or "1000万" in raw:
        return "2勝"
    if "3勝" in raw or "1600万" in raw:
        return "3勝"
    if any(token in raw for token in ["Ｇ１", "G1", "GⅠ", "ＧⅠ"]):
        return "G1"
    if any(token in raw for token in ["Ｇ２", "G2", "GⅡ", "ＧⅡ"]):
        return "G2"
    if any(token in raw for token in ["Ｇ３", "G3", "GⅢ", "ＧⅢ"]):
        return "G3"
    if "ｵｰﾌﾟﾝ" in raw or "オープン" in raw or raw.upper() == "OP":
        return "OP"
    if "L" in raw or "リステッド" in raw:
        return "L"
    return raw.strip() or "不明"


def normalize_surface(value: object) -> str:
    raw = "" if value is None or pd.isna(value) else str(value).strip()
    if raw.startswith("芝") or raw.lower() in {"turf", "grass"}:
        return "芝"
    if raw.startswith("ダ") or raw.lower() in {"dirt", "sand"}:
        return "ダ"
    if raw.startswith("障"):
        return "障"
    return raw


def load_feature_races(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    keep = [
        RACE_COL,
        DATE_COL,
        "場所",
        "芝・ダ",
        "距離",
        "馬場状態",
        "クラス名",
        "走破タイム",
        "平均1Fタイム",
        "Ave-3F",
        "RPCI",
        "PCI3",
        "確定着順",
    ]
    for path in paths:
        if not path.exists():
            continue
        header = read_csv_any(path, nrows=0)
        usecols = [col for col in keep if col in header.columns]
        if not usecols or RACE_COL not in usecols:
            continue
        frames.append(read_csv_any(path, usecols=usecols))
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw[RACE_COL] = raw[RACE_COL].astype(str).str.replace(r"\.0$", "", regex=True)
    raw["race_date"] = raw[DATE_COL].apply(parse_date) if DATE_COL in raw.columns else pd.NaT
    raw["venue"] = raw.get("場所", "").astype(str).str.strip()
    raw["surface"] = raw.get("芝・ダ", "").map(normalize_surface)
    raw["distance"] = pd.to_numeric(raw.get("距離"), errors="coerce")
    raw["going"] = raw.get("馬場状態", "").astype(str).str.strip()
    raw["class_group"] = raw.get("クラス名", "").map(class_group)
    for col in ["走破タイム", "平均1Fタイム", "Ave-3F", "RPCI", "PCI3", "確定着順"]:
        if col in raw.columns:
            raw[col] = raw[col].map(to_num)

    race_rows: list[dict[str, Any]] = []
    for race_id, part in raw.groupby(RACE_COL, sort=False):
        part = part.copy()
        top = part.iloc[0]
        finish = pd.to_numeric(part.get("走破タイム"), errors="coerce")
        winner_time = np.nan
        if "確定着順" in part.columns:
            winners = part.loc[pd.to_numeric(part["確定着順"], errors="coerce").eq(1), "走破タイム"]
            if winners.notna().any():
                winner_time = float(winners.dropna().iloc[0])
        if not np.isfinite(winner_time) and finish.notna().any():
            winner_time = float(finish.min())
        race_rows.append(
            {
                "race_id": str(race_id),
                "race_date": top.get("race_date"),
                "venue": top.get("venue", ""),
                "surface": top.get("surface", ""),
                "distance": top.get("distance", np.nan),
                "going": top.get("going", ""),
                "class_group": top.get("class_group", "不明"),
                "winning_time_sec": winner_time,
                "avg_1f_sec": np.nanmean(pd.to_numeric(part.get("平均1Fタイム"), errors="coerce")),
                "ave_3f_sec": np.nanmean(pd.to_numeric(part.get("Ave-3F"), errors="coerce")),
                "rpci": np.nanmean(pd.to_numeric(part.get("RPCI"), errors="coerce")),
                "pci3": np.nanmean(pd.to_numeric(part.get("PCI3"), errors="coerce")),
            }
        )
    races = pd.DataFrame(race_rows)
    races = races.dropna(subset=["race_date", "venue", "surface", "distance"])
    races["distance"] = races["distance"].astype(int)
    return races


def load_front3f_races(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    cols = ["race_id", "distance_m", "race_first3f_sec", "race_last3f_sec", "race_total_time_sec"]
    header = read_csv_any(path, nrows=0)
    usecols = [c for c in cols if c in header.columns]
    if "race_id" not in usecols:
        return pd.DataFrame()
    frame = read_csv_any(path, usecols=usecols)
    frame["race_id"] = frame["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    race = frame.drop_duplicates("race_id", keep="first").copy()
    for col in ["race_first3f_sec", "race_last3f_sec", "race_total_time_sec"]:
        if col in race.columns:
            race[col] = pd.to_numeric(race[col], errors="coerce")
    return race


def estimate_pass_1000m_sec(frame: pd.DataFrame) -> pd.Series:
    distance = pd.to_numeric(frame.get("distance"), errors="coerce")
    first3f = pd.to_numeric(frame.get("race_first3f_sec"), errors="coerce")
    last3f = pd.to_numeric(frame.get("race_last3f_sec"), errors="coerce")
    total = pd.to_numeric(frame.get("winning_time_sec"), errors="coerce")
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    ok = distance.ge(1000) & first3f.notna() & last3f.notna() & total.notna()
    if not ok.any():
        return out
    middle_distance = (distance - 1200.0).clip(lower=0.0)
    middle_time = (total - first3f - last3f).clip(lower=0.0)
    position = 1000.0
    early = pd.Series(False, index=frame.index)
    if early.any():
        out.loc[early] = first3f.loc[early] * (position / 600.0)
    late = ok & (distance - 600.0).le(position)
    if late.any():
        remaining = (distance.loc[late] - position).clip(lower=0.0)
        out.loc[late] = total.loc[late] - last3f.loc[late] * (remaining / 600.0)
    middle = ok & ~early & ~late & middle_distance.gt(0)
    if middle.any():
        out.loc[middle] = first3f.loc[middle] + middle_time.loc[middle] * ((position - 600.0) / middle_distance.loc[middle])
    return out


def aggregate_context(races: pd.DataFrame, years: int, today: pd.Timestamp | None = None) -> pd.DataFrame:
    if races.empty:
        return pd.DataFrame()
    ref = today if today is not None and not pd.isna(today) else pd.Timestamp.today().normalize()
    start = ref - pd.DateOffset(years=years)
    scoped = races[(races["race_date"] >= start) & (races["race_date"] < ref)].copy()
    if scoped.empty:
        scoped = races.copy()
    rows: list[pd.DataFrame] = []
    scope_defs = [
        ("同場同距離×クラス×馬場", ["venue", "surface", "distance", "class_group", "going"]),
        ("同場同距離×クラス", ["venue", "surface", "distance", "class_group"]),
        ("同場同距離×馬場", ["venue", "surface", "distance", "going"]),
        ("同場同距離", ["venue", "surface", "distance"]),
        ("同芝ダ同距離×クラス", ["surface", "distance", "class_group"]),
        ("同芝ダ同距離", ["surface", "distance"]),
    ]
    metrics = {
        "winning_time_sec": "avg_winning_time_sec",
        "race_first3f_sec": "avg_front3f_sec",
        "race_last3f_sec": "avg_last3f_sec",
        "pass_1000m_sec": "avg_1000m_sec",
        "rpci": "avg_rpci",
        "pci3": "avg_pci3",
        "avg_1f_sec": "avg_1f_sec",
        "ave_3f_sec": "avg_ave3f_sec",
    }
    for scope, keys in scope_defs:
        present = [key for key in keys if key in scoped.columns]
        agg_map = {src: "mean" for src in metrics if src in scoped.columns}
        if not present or not agg_map:
            continue
        agg = scoped.groupby(present, dropna=False, as_index=False).agg(
            sample_count=("race_id", "nunique"),
            **{dst: (src, "mean") for src, dst in metrics.items() if src in scoped.columns},
        )
        agg["scope"] = scope
        agg["years"] = years
        for col in ["venue", "surface", "distance", "class_group", "going"]:
            if col not in agg.columns:
                agg[col] = "__ALL__"
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out[
        [
            "years",
            "scope",
            "venue",
            "surface",
            "distance",
            "class_group",
            "going",
            "sample_count",
            *[dst for dst in metrics.values() if dst in out.columns],
        ]
    ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build historical same-condition lap/time baselines for dashboard display.")
    parser.add_argument("--feature-csv", action="append", default=[], help="Historical feature CSV. Can be specified multiple times.")
    parser.add_argument("--front3f-csv", default=DEFAULT_FRONT3F)
    parser.add_argument("--output-csv", default=DEFAULT_OUT)
    parser.add_argument("--as-of-date", default="", help="YYYYMMDD reference date. Defaults to today.")
    args = parser.parse_args()

    feature_paths = [project_path(p) for p in (args.feature_csv or DEFAULT_FEATURES)]
    races = load_feature_races(feature_paths)
    front3f = load_front3f_races(project_path(args.front3f_csv))
    if not front3f.empty and not races.empty:
        races = races.merge(front3f, on="race_id", how="left")
        races["winning_time_sec"] = races["race_total_time_sec"].fillna(races["winning_time_sec"])
    races["pass_1000m_sec"] = estimate_pass_1000m_sec(races)

    ref = pd.to_datetime(args.as_of_date, format="%Y%m%d", errors="coerce") if args.as_of_date else pd.Timestamp.today().normalize()
    out = pd.concat([aggregate_context(races, 5, ref), aggregate_context(races, 10, ref)], ignore_index=True, sort=False)
    output = project_path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    print(
        {
            "output_csv": str(output),
            "source_races": int(races["race_id"].nunique()) if not races.empty else 0,
            "rows": int(len(out)),
            "years": sorted(out["years"].dropna().unique().tolist()) if not out.empty else [],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
