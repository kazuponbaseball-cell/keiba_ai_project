from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "jra_official_results"
DEFAULT_SUMMARY = ROOT / "outputs" / "analysis" / "official_lap_store_update_v1" / "summary.json"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def run_step(name: str, args: list[str]) -> dict[str, Any]:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return {
        "name": name,
        "command": args,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-5000:],
        "stderr_tail": result.stderr[-5000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh extracted official JRA race laps, result runners, lap-history features, and coverage audit from cached official result HTML."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    raw_dir = project_path(args.raw_dir)
    summary_json = project_path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    steps.append(
        run_step(
            "extract_official_race_laps",
            [
                sys.executable,
                "scripts/extract_jra_official_race_laps.py",
                "--raw-dir",
                str(raw_dir),
            ],
        )
    )
    steps.append(
        run_step(
            "extract_official_result_runners",
            [
                sys.executable,
                "scripts/extract_jra_official_result_runners.py",
                "--raw-dir",
                str(raw_dir),
            ],
        )
    )
    steps.append(
        run_step(
            "build_official_result_lap_history_features",
            [
                sys.executable,
                "scripts/build_official_lap_history_features.py",
                "--runner-csv",
                "data/processed/jra_official_results/result_runners.csv",
                "--output-csv",
                "data/processed/jra_official_race_laps/official_result_lap_history_features.csv",
                "--summary-json",
                "outputs/analysis/official_lap_history_features_v1/official_result_summary.json",
            ],
        )
    )
    steps.append(
        run_step(
            "audit_official_lap_coverage",
            [
                sys.executable,
                "scripts/audit_official_lap_coverage.py",
                "--raw-dir",
                str(raw_dir),
            ],
        )
    )

    payload = {
        "raw_dir": str(raw_dir),
        "summary_json": str(summary_json),
        "steps": steps,
        "failed_steps": [step for step in steps if step["returncode"] != 0],
        "outputs": {
            "race_laps_csv": str(ROOT / "data" / "processed" / "jra_official_race_laps" / "race_laps.csv"),
            "result_runners_csv": str(ROOT / "data" / "processed" / "jra_official_results" / "result_runners.csv"),
            "official_result_lap_history_csv": str(
                ROOT
                / "data"
                / "processed"
                / "jra_official_race_laps"
                / "official_result_lap_history_features.csv"
            ),
            "coverage_summary_json": str(ROOT / "outputs" / "analysis" / "official_lap_coverage_v1" / "summary.json"),
        },
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["failed_steps"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
