from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "jra_official_results"
DEFAULT_LAPS = ROOT / "data" / "processed" / "jra_official_race_laps" / "race_laps.csv"
DEFAULT_RUNNERS = ROOT / "data" / "processed" / "jra_official_results" / "result_runners.csv"
DEFAULT_HISTORY = ROOT / "data" / "processed" / "jra_official_race_laps" / "official_result_lap_history_features.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "analysis" / "official_lap_coverage_v1"

DEFAULT_HISTORICAL_RUNNER_CSVS = [
    ROOT
    / "data"
    / "datasets"
    / "cache"
    / "workout_lap_pedigree_interactions_confirmed_opponent_2023plus"
    / "train_features.csv",
    ROOT
    / "data"
    / "datasets"
    / "cache"
    / "workout_lap_pedigree_interactions_confirmed_opponent_2023plus"
    / "test_features.csv",
]


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


def clean_race_id(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(16)


def latest_weekly_entry_csv() -> Path | None:
    base = ROOT / "data" / "datasets" / "inference" / "weekly"
    if not base.exists():
        return None
    files = sorted(
        base.glob("entry_snapshot_*_target_de_overlay_enriched_workout_knowledge.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def raw_html_races(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not raw_dir.exists():
        return pd.DataFrame(columns=["race_id", "raw_html_files"])
    for race_dir in sorted(raw_dir.iterdir()):
        if not race_dir.is_dir() or not race_dir.name.isdigit() or len(race_dir.name) != 16:
            continue
        files = list(race_dir.glob("*_result.html"))
        if not files:
            continue
        rows.append(
            {
                "race_id": race_dir.name,
                "raw_html_files": len(files),
                "latest_raw_html_mtime": max(p.stat().st_mtime for p in files),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["race_id", "raw_html_files", "latest_raw_html_mtime"])
    return out


def race_set_from_csv(path: Path, race_id_candidates: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["race_id"])
    header = read_csv_any(path, nrows=0)
    race_col = next((c for c in race_id_candidates if c in header.columns), None)
    if race_col is None:
        return pd.DataFrame(columns=["race_id"])
    df = read_csv_any(path, usecols=[race_col])
    out = pd.DataFrame({"race_id": clean_race_id(df[race_col])})
    out = out[out["race_id"].str.fullmatch(r"\d{16}", na=False)]
    return out.drop_duplicates("race_id")


def race_rows_from_csv(path: Path, race_id_candidates: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["race_id"])
    header = read_csv_any(path, nrows=0)
    race_col = next((c for c in race_id_candidates if c in header.columns), None)
    if race_col is None:
        return pd.DataFrame(columns=["race_id"])
    usecols = [race_col]
    for col in ("horse_no", "馬番", "鬥ｬ逡ｪ"):
        if col in header.columns:
            usecols.append(col)
            break
    df = read_csv_any(path, usecols=usecols)
    df["race_id"] = clean_race_id(df[race_col])
    df = df[df["race_id"].str.fullmatch(r"\d{16}", na=False)].copy()
    return df


def source_coverage(source_name: str, expected: pd.DataFrame, status: pd.DataFrame) -> dict[str, Any]:
    if expected.empty:
        return {
            "source": source_name,
            "expected_races": 0,
            "raw_html_races": 0,
            "lap_races": 0,
            "runner_result_races": 0,
            "history_ready_races": 0,
            "lap_coverage_pct": None,
            "runner_coverage_pct": None,
            "history_ready_pct": None,
        }
    merged = expected[["race_id"]].drop_duplicates().merge(status, on="race_id", how="left")
    for col in ["has_raw_html", "has_official_lap", "has_result_runners", "has_history_ready"]:
        if col not in merged.columns:
            merged[col] = False
        merged[col] = merged[col].fillna(False).astype(bool)
    denom = len(merged)
    return {
        "source": source_name,
        "expected_races": int(denom),
        "raw_html_races": int(merged["has_raw_html"].sum()),
        "lap_races": int(merged["has_official_lap"].sum()),
        "runner_result_races": int(merged["has_result_runners"].sum()),
        "history_ready_races": int(merged["has_history_ready"].sum()),
        "lap_coverage_pct": round(float(merged["has_official_lap"].mean() * 100.0), 1),
        "runner_coverage_pct": round(float(merged["has_result_runners"].mean() * 100.0), 1),
        "history_ready_pct": round(float(merged["has_history_ready"].mean() * 100.0), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official JRA race-lap coverage against current and historical datasets.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--laps-csv", type=Path, default=DEFAULT_LAPS)
    parser.add_argument("--runners-csv", type=Path, default=DEFAULT_RUNNERS)
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--entry-csv", type=Path, default=None)
    parser.add_argument("--historical-runner-csv", action="append", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    raw_dir = project_path(args.raw_dir)
    laps_csv = project_path(args.laps_csv)
    runners_csv = project_path(args.runners_csv)
    history_csv = project_path(args.history_csv)
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = raw_html_races(raw_dir)
    laps = race_set_from_csv(laps_csv, ["race_id"])
    runners = race_rows_from_csv(runners_csv, ["race_id"])
    history = race_rows_from_csv(history_csv, ["race_id"])

    if not history.empty and "official_lap_history_ready" in read_csv_any(history_csv, nrows=0).columns:
        h = read_csv_any(history_csv, usecols=["race_id", "official_lap_history_ready"])
        h["race_id"] = clean_race_id(h["race_id"])
        h["official_lap_history_ready"] = pd.to_numeric(h["official_lap_history_ready"], errors="coerce").fillna(0)
        history_ready = h[h["official_lap_history_ready"].gt(0)][["race_id"]].drop_duplicates()
    else:
        history_ready = pd.DataFrame(columns=["race_id"])

    all_races = pd.concat(
        [
            raw[["race_id"]],
            laps[["race_id"]],
            runners[["race_id"]],
            history[["race_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates("race_id")
    if all_races.empty:
        status = pd.DataFrame(columns=["race_id"])
    else:
        status = all_races.copy()
        status = status.merge(raw[["race_id", "raw_html_files"]], on="race_id", how="left")
        status["has_raw_html"] = status["raw_html_files"].fillna(0).gt(0)
        status["has_official_lap"] = status["race_id"].isin(set(laps["race_id"]))
        status["has_result_runners"] = status["race_id"].isin(set(runners["race_id"]))
        status["has_history_row"] = status["race_id"].isin(set(history["race_id"]))
        status["has_history_ready"] = status["race_id"].isin(set(history_ready["race_id"]))
        status["date"] = status["race_id"].str[:8]
        status["venue_code"] = status["race_id"].str[8:10]
        status["race_no"] = pd.to_numeric(status["race_id"].str[-2:], errors="coerce").astype("Int64")

    latest_entry = project_path(args.entry_csv) if args.entry_csv else latest_weekly_entry_csv()
    race_id_columns = [
        "race_id",
        "raceId",
        "レースID",
        "レースID(新/馬番無)",
        "繝ｬ繝ｼ繧ｹID(譁ｰ/鬥ｬ逡ｪ辟｡)",
    ]
    entry = race_set_from_csv(latest_entry, race_id_columns) if latest_entry else pd.DataFrame(columns=["race_id"])

    historical_paths = [project_path(p) for p in (args.historical_runner_csv or DEFAULT_HISTORICAL_RUNNER_CSVS)]
    hist_expected = pd.concat(
        [race_set_from_csv(path, race_id_columns) for path in historical_paths],
        ignore_index=True,
    ).drop_duplicates("race_id")

    if not status.empty:
        coverage_by_date = (
            status.groupby("date")
            .agg(
                raw_html_races=("has_raw_html", "sum"),
                lap_races=("has_official_lap", "sum"),
                runner_result_races=("has_result_runners", "sum"),
                history_ready_races=("has_history_ready", "sum"),
                total_known_races=("race_id", "nunique"),
            )
            .reset_index()
            .sort_values("date")
        )
    else:
        coverage_by_date = pd.DataFrame()
    coverage_by_date.to_csv(out_dir / "coverage_by_date.csv", index=False, encoding="utf-8-sig")

    status.to_csv(out_dir / "race_status.csv", index=False, encoding="utf-8-sig")
    if latest_entry is not None:
        entry_missing = entry.merge(status[["race_id", "has_official_lap", "has_result_runners"]], on="race_id", how="left")
        entry_missing[["has_official_lap", "has_result_runners"]] = entry_missing[
            ["has_official_lap", "has_result_runners"]
        ].fillna(False)
        entry_missing = entry_missing[~entry_missing["has_official_lap"] | ~entry_missing["has_result_runners"]].copy()
        entry_missing.to_csv(out_dir / "missing_latest_entry_races.csv", index=False, encoding="utf-8-sig")

    coverage_sources = pd.DataFrame(
        [
            source_coverage("latest_weekly_entry", entry, status),
            source_coverage("historical_train_test", hist_expected, status),
            source_coverage("official_raw_store", raw[["race_id"]], status),
        ]
    )
    coverage_sources.to_csv(out_dir / "coverage_sources.csv", index=False, encoding="utf-8-sig")

    summary = {
        "out_dir": str(out_dir),
        "raw_dir": str(raw_dir),
        "laps_csv": str(laps_csv),
        "runners_csv": str(runners_csv),
        "history_csv": str(history_csv),
        "latest_entry_csv": str(latest_entry) if latest_entry else "",
        "historical_runner_csvs": [str(p) for p in historical_paths],
        "raw_html_races": int(raw["race_id"].nunique()) if not raw.empty else 0,
        "official_lap_races": int(laps["race_id"].nunique()) if not laps.empty else 0,
        "official_runner_result_races": int(runners["race_id"].nunique()) if not runners.empty else 0,
        "official_runner_rows": int(len(runners)),
        "official_lap_history_rows": int(len(history)),
        "official_lap_history_ready_races": int(history_ready["race_id"].nunique()) if not history_ready.empty else 0,
        "coverage_sources": coverage_sources.to_dict(orient="records"),
        "note": (
            "History-ready stays low until the same horses appear again in the official-result lap store. "
            "Backfill older official result laps or TARGET/JRA-VAN lap exports before using these features in ROI tests."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
