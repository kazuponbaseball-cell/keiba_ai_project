from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "outputs" / "analysis" / "target_ra_lap_store_update_v1" / "summary.json"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def run_step(name: str, cmd: list[str], *, allow_fail: bool = False) -> dict[str, Any]:
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    result = {
        "name": name,
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
    }
    if completed.returncode != 0 and not allow_fail:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(completed.returncode)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh TARGET/JRA-VAN RA race laps, lap-history features, coverage audit, and shadow ROI overlay."
    )
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--skip-overlay", action="store_true")
    args = parser.parse_args()

    summary_json = project_path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    steps.append(
        run_step(
            "extract_target_ra_race_laps",
            [
                sys.executable,
                "scripts/extract_target_ra_race_laps_all.py",
                "--start-year",
                str(args.start_year),
                "--end-year",
                str(args.end_year),
            ],
        )
    )
    steps.append(
        run_step(
            "build_target_ra_lap_history_features",
            [
                sys.executable,
                "scripts/build_official_lap_history_features.py",
                "--laps-csv",
                "data/processed/target_ra_race_laps/race_laps.csv",
                "--output-csv",
                "data/processed/target_ra_race_laps/target_ra_lap_history_features.csv",
                "--summary-json",
                "outputs/analysis/target_ra_lap_history_features_v1/summary.json",
            ],
        )
    )
    steps.append(
        run_step(
            "audit_target_ra_lap_coverage",
            [
                sys.executable,
                "scripts/audit_official_lap_coverage.py",
                "--laps-csv",
                "data/processed/target_ra_race_laps/race_laps.csv",
                "--out-dir",
                "outputs/analysis/target_ra_lap_coverage_v1",
            ],
        )
    )
    if not args.skip_overlay:
        steps.append(
            run_step(
                "evaluate_target_ra_official_lap_history_overlay",
                [sys.executable, "scripts/evaluate_target_ra_official_lap_history_overlay.py"],
                allow_fail=False,
            )
        )

    summary = {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "steps": steps,
        "race_laps_csv": str(ROOT / "data" / "processed" / "target_ra_race_laps" / "race_laps.csv"),
        "lap_history_csv": str(
            ROOT / "data" / "processed" / "target_ra_race_laps" / "target_ra_lap_history_features.csv"
        ),
        "coverage_summary_json": str(ROOT / "outputs" / "analysis" / "target_ra_lap_coverage_v1" / "summary.json"),
        "overlay_summary_json": str(
            ROOT / "outputs" / "analysis" / "target_ra_official_lap_history_overlay_v1" / "summary.json"
        ),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
