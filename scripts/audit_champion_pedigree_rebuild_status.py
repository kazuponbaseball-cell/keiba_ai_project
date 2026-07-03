from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_TRACKED_PATHS = [
    "src/features/baseline.py",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/train_features_with_same_day_bias_v3_retro.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_breeder_enriched/train_features_with_same_day_bias_v3_retro_body_breeder.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_breeder_enriched/test_features_with_same_day_bias_v3_retro_body_breeder.csv",
    "models/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/baseline_ranker.pkl",
    "models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl",
    "models/body_context_same_day_bias_v3_retro/baseline_ranker.pkl",
    "models/body_owner_numeric_breeder_context_same_day_bias_v3_retro/baseline_ranker.pkl",
    "outputs/analysis/pair_probability_runtime_v1/pair_calibrated_runtime_tickets.csv",
    "outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates_default/recommended_runtime_tickets.csv",
    "outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates_default/recommended_all_tickets.csv",
    "outputs/analysis/champion_strategy_freeze_v1/champion_manifest.json",
]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    return value


def file_row(rel: str, reference_mtime: float | None) -> dict[str, Any]:
    path = project_path(rel)
    exists = path.exists()
    mtime = path.stat().st_mtime if exists else None
    stale = bool(reference_mtime is not None and mtime is not None and mtime < reference_mtime)
    return {
        "path": rel,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "modified_at": datetime.fromtimestamp(mtime).isoformat(timespec="seconds") if mtime is not None else None,
        "older_than_pedigree_fix": stale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether Champion artifacts predate the pedigree same-race leakage fix.")
    parser.add_argument("--output-dir", default="outputs/analysis/champion_pedigree_rebuild_status_v1")
    parser.add_argument("--pedigree-fix-file", default="src/features/baseline.py")
    parser.add_argument("--path", action="append", default=None, help="Additional artifact path to audit.")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fix_path = project_path(args.pedigree_fix_file)
    fix_mtime = fix_path.stat().st_mtime if fix_path.exists() else None
    paths = list(dict.fromkeys([*DEFAULT_TRACKED_PATHS, *(args.path or [])]))
    rows = [file_row(path, fix_mtime) for path in paths]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "artifact_staleness.csv", index=False, encoding="utf-8-sig")

    stale_existing = df[df["exists"] & df["older_than_pedigree_fix"]].copy()
    missing = df[~df["exists"]].copy()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pedigree_fix_file": args.pedigree_fix_file,
        "pedigree_fix_modified_at": datetime.fromtimestamp(fix_mtime).isoformat(timespec="seconds") if fix_mtime else None,
        "tracked_artifacts": int(len(df)),
        "existing_artifacts": int(df["exists"].sum()),
        "stale_existing_artifacts": int(len(stale_existing)),
        "missing_artifacts": int(len(missing)),
        "champion_rebuild_required": bool(len(stale_existing) > 0),
        "stale_paths": stale_existing["path"].tolist(),
        "missing_paths": missing["path"].tolist(),
        "recommended_next_step": (
            "Rebuild feature caches, retrain affected models, then rerun pair probability, MCS/PBO, odds guards, and Champion freeze."
            if len(stale_existing) > 0
            else "No stale tracked artifact found relative to the pedigree fix timestamp."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
