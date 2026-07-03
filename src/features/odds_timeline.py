from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ODDS_TIMELINE_FEATURE_COLUMNS = [
    "race_id",
    "horse_number",
    "odds_snapshot_count",
    "odds_valid_snapshot_count",
    "odds_first_snapshot_at",
    "odds_latest_snapshot_at",
    "odds_elapsed_minutes",
    "odds_first_win",
    "odds_prev_win",
    "odds_latest_win",
    "odds_first_popularity",
    "odds_prev_popularity",
    "odds_latest_popularity",
    "odds_win_change_from_first_pct",
    "odds_win_change_from_prev_pct",
    "odds_drop_from_first_pct",
    "odds_drop_from_prev_pct",
    "odds_popularity_change_from_first",
    "odds_popularity_change_from_prev",
    "odds_place_min",
    "odds_place_max",
    "odds_place_mid",
    "odds_win_place_ratio",
    "odds_drop_velocity_per_min",
    "odds_steam_flag",
    "odds_drift_flag",
]

ODDS_SENTINEL_MAX = 999.0


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def clean_odds_series(series: pd.Series, *, min_value: float = 1.0) -> pd.Series:
    """Return numeric odds, treating capped/missing placeholders as unavailable."""
    values = _num(series)
    return values.where(values.ge(min_value) & values.lt(ODDS_SENTINEL_MAX))


def clean_odds_value(value: object, *, min_value: float = 1.0) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        parsed = float(str(value).replace(",", "").strip())
    except Exception:
        return None
    if min_value <= parsed < ODDS_SENTINEL_MAX:
        return parsed
    return None


def _snapshot_time(values: pd.Series) -> pd.Series:
    text = values.astype("string")
    parsed = pd.to_datetime(text, format="%Y%m%d_%H%M%S", errors="coerce")
    if parsed.notna().all():
        return parsed
    fallback = pd.to_datetime(text[parsed.isna()], errors="coerce")
    parsed.loc[parsed.isna()] = fallback
    return parsed


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def build_odds_timeline_features(timeline: pd.DataFrame) -> pd.DataFrame:
    required = {"race_id", "horse_number", "snapshot_at", "win_odds"}
    missing = required - set(timeline.columns)
    if missing:
        raise ValueError(f"Odds timeline is missing columns: {sorted(missing)}")
    if timeline.empty:
        return pd.DataFrame(columns=ODDS_TIMELINE_FEATURE_COLUMNS)

    df = timeline.copy()
    df["snapshot_time"] = _snapshot_time(df["snapshot_at"])
    df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce").astype("Int64")
    for col in ["win_odds", "place_odds_min", "place_odds_max", "popularity_estimated"]:
        if col not in df.columns:
            df[col] = np.nan
    df["win_odds"] = clean_odds_series(df["win_odds"])
    df["place_odds_min"] = clean_odds_series(df["place_odds_min"])
    df["place_odds_max"] = clean_odds_series(df["place_odds_max"])
    df["popularity_estimated"] = _num(df["popularity_estimated"])

    df = df[df["race_id"].notna() & df["horse_number"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=ODDS_TIMELINE_FEATURE_COLUMNS)
    df = df.sort_values(["race_id", "horse_number", "snapshot_time", "snapshot_at"], kind="mergesort")

    rows = []
    for (race_id, horse_number), part in df.groupby(["race_id", "horse_number"], sort=False):
        valid = part[part["win_odds"].notna()].copy()
        source = valid if len(valid) else part
        first = source.iloc[0]
        latest = source.iloc[-1]
        previous = source.iloc[-2] if len(source) >= 2 else first

        first_win = float(first["win_odds"]) if pd.notna(first["win_odds"]) else np.nan
        latest_win = float(latest["win_odds"]) if pd.notna(latest["win_odds"]) else np.nan
        prev_win = float(previous["win_odds"]) if pd.notna(previous["win_odds"]) else np.nan
        first_pop = float(first["popularity_estimated"]) if pd.notna(first["popularity_estimated"]) else np.nan
        latest_pop = float(latest["popularity_estimated"]) if pd.notna(latest["popularity_estimated"]) else np.nan
        prev_pop = float(previous["popularity_estimated"]) if pd.notna(previous["popularity_estimated"]) else np.nan

        elapsed_minutes = np.nan
        if pd.notna(first["snapshot_time"]) and pd.notna(latest["snapshot_time"]):
            elapsed_minutes = max(0.0, (latest["snapshot_time"] - first["snapshot_time"]).total_seconds() / 60.0)

        place_min = float(latest["place_odds_min"]) if pd.notna(latest["place_odds_min"]) else np.nan
        place_max = float(latest["place_odds_max"]) if pd.notna(latest["place_odds_max"]) else np.nan
        place_values = [value for value in [place_min, place_max] if np.isfinite(value)]
        place_mid = float(np.mean(place_values)) if place_values else np.nan
        win_change_first = latest_win / first_win - 1.0 if first_win and latest_win else np.nan
        win_change_prev = latest_win / prev_win - 1.0 if prev_win and latest_win else np.nan
        odds_drop_first = first_win / latest_win - 1.0 if first_win and latest_win else np.nan
        odds_drop_prev = prev_win / latest_win - 1.0 if prev_win and latest_win else np.nan
        pop_change_first = first_pop - latest_pop if pd.notna(first_pop) and pd.notna(latest_pop) else np.nan
        pop_change_prev = prev_pop - latest_pop if pd.notna(prev_pop) and pd.notna(latest_pop) else np.nan
        place_mid = place_mid if np.isfinite(place_mid) else np.nan
        win_place_ratio = latest_win / place_mid if latest_win and place_mid else np.nan
        velocity = odds_drop_first / elapsed_minutes if elapsed_minutes and np.isfinite(odds_drop_first) else np.nan

        rows.append(
            {
                "race_id": str(race_id),
                "horse_number": int(horse_number),
                "odds_snapshot_count": int(len(part)),
                "odds_valid_snapshot_count": int(len(valid)),
                "odds_first_snapshot_at": first["snapshot_at"],
                "odds_latest_snapshot_at": latest["snapshot_at"],
                "odds_elapsed_minutes": elapsed_minutes,
                "odds_first_win": first_win,
                "odds_prev_win": prev_win,
                "odds_latest_win": latest_win,
                "odds_first_popularity": first_pop,
                "odds_prev_popularity": prev_pop,
                "odds_latest_popularity": latest_pop,
                "odds_win_change_from_first_pct": win_change_first,
                "odds_win_change_from_prev_pct": win_change_prev,
                "odds_drop_from_first_pct": odds_drop_first,
                "odds_drop_from_prev_pct": odds_drop_prev,
                "odds_popularity_change_from_first": pop_change_first,
                "odds_popularity_change_from_prev": pop_change_prev,
                "odds_place_min": place_min,
                "odds_place_max": place_max,
                "odds_place_mid": place_mid,
                "odds_win_place_ratio": win_place_ratio,
                "odds_drop_velocity_per_min": velocity,
                "odds_steam_flag": float((odds_drop_first >= 0.15) or (pop_change_first >= 2)) if np.isfinite(odds_drop_first) or np.isfinite(pop_change_first) else 0.0,
                "odds_drift_flag": float((win_change_first >= 0.25) or (pop_change_first <= -2)) if np.isfinite(win_change_first) or np.isfinite(pop_change_first) else 0.0,
            }
        )

    return pd.DataFrame(rows, columns=ODDS_TIMELINE_FEATURE_COLUMNS)


def build_odds_timeline_features_from_file(path: str | Path) -> pd.DataFrame:
    return build_odds_timeline_features(pd.read_csv(path, encoding="utf-8-sig"))


def merge_odds_timeline_features(
    frame: pd.DataFrame,
    odds_features: pd.DataFrame,
    *,
    race_col: str,
    horse_number_col: str = "馬番",
) -> pd.DataFrame:
    if race_col not in frame.columns:
        raise ValueError(f"Frame is missing race column: {race_col}")
    if horse_number_col not in frame.columns:
        raise ValueError(f"Frame is missing horse number column: {horse_number_col}")

    out = frame.copy()
    features = odds_features.copy()
    out["_race_key"] = out[race_col].astype(str)
    out["_horse_number_key"] = pd.to_numeric(out[horse_number_col], errors="coerce").astype("Int64")
    features["_race_key"] = features["race_id"].astype(str)
    features["_horse_number_key"] = pd.to_numeric(features["horse_number"], errors="coerce").astype("Int64")
    feature_cols = [col for col in features.columns if col not in {"race_id", "horse_number", "_race_key", "_horse_number_key"}]
    merged = out.merge(features[["_race_key", "_horse_number_key", *feature_cols]], on=["_race_key", "_horse_number_key"], how="left")
    return merged.drop(columns=["_race_key", "_horse_number_key"])
