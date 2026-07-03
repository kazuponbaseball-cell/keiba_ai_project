from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_umaren_odds_cap(script_path: Path) -> float | None:
    if not script_path.exists():
        return None
    text = script_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"MAX_FINAL_BUY_UMAREN_ODDS\s*=\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def csv_metrics(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"exists": path.exists(), "rows": 0, "cols": 0}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str}, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp932", dtype={"race_id": str}, low_memory=False)
    out: dict[str, Any] = {
        "exists": True,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "races": int(df["race_id"].nunique()) if "race_id" in df.columns else None,
    }
    for col in ("runtime_stake_yen", "stake_yen"):
        if col in df.columns:
            out[f"{col}_sum"] = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze current strongest strategy as Champion manifest.")
    parser.add_argument("--output-dir", default="outputs/analysis/champion_strategy_freeze_v1")
    parser.add_argument("--runtime-summary-json", default="outputs/analysis/current_strongest_runtime_v1/summary.json")
    parser.add_argument("--longterm-summary-json", default="outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates_default/summary.json")
    parser.add_argument("--stress-summary-json", default="outputs/analysis/low_prob_high_odds_guard_v1/summary.json")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracked_files = [
        "scripts/build_current_strongest_tickets.py",
        "scripts/run_current_strongest_line_update.ps1",
        "scripts/freeze_current_strongest_decision_snapshot.py",
        "scripts/analyze_low_prob_high_odds_guard.py",
        "scripts/analyze_calibration_and_fractional_kelly.py",
        "outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates_default/recommended_runtime_tickets.csv",
        "outputs/analysis/current_strongest_runtime_v1/current_strongest_all_candidates.csv",
        "outputs/analysis/current_strongest_runtime_v1/selected_after_live_safety.csv",
    ]
    file_rows = []
    for rel in tracked_files:
        path = project_path(rel)
        file_rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else None,
                "sha256": file_sha256(path),
            }
        )

    build_script = project_path("scripts/build_current_strongest_tickets.py")
    runtime_summary = read_json(project_path(args.runtime_summary_json))
    longterm_summary = read_json(project_path(args.longterm_summary_json))
    stress_summary = read_json(project_path(args.stress_summary_json))

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "champion_id": "current_strongest_mcs_pbo_strict_v1",
        "status": "frozen_champion",
        "freeze_rule": "Do not tune Champion on 2024-2026 backtest. All changes must be evaluated as Challenger shadow policies.",
        "core_policy": {
            "ticket_type": "umaren",
            "max_final_buy_umaren_odds": extract_umaren_odds_cap(build_script),
            "max_per_day": 4,
            "max_per_race": 1,
            "stress_roi_target_range": "140-160pct top10-hit-removed / odds-cap stress view",
            "primary_success_metric": "out-of-sample ROI with top-hit removal and fixed 100-yen parallel accounting",
        },
        "runtime_summary": runtime_summary,
        "longterm_backtest": longterm_summary.get("best_by_top10_removed_roi") or longterm_summary.get("selected_policy"),
        "odds_cap_stress": stress_summary.get("best_guard"),
        "files": file_rows,
        "csv_metrics": {
            "current_candidates": csv_metrics(project_path("outputs/analysis/current_strongest_runtime_v1/current_strongest_all_candidates.csv")),
            "current_selected": csv_metrics(project_path("outputs/analysis/current_strongest_runtime_v1/selected_after_live_safety.csv")),
            "longterm_recommended": csv_metrics(project_path("outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates_default/recommended_runtime_tickets.csv")),
        },
    }

    (out_dir / "champion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(file_rows).to_csv(out_dir / "champion_file_hashes.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
