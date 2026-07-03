from __future__ import annotations

import argparse
import json

import pandas as pd

from src.data.loaders import (
    inference_optional_columns,
    inference_required_columns,
    load_json_config,
)
from src.utils.paths import ensure_dir, project_path
from src.utils.runtime_config import load_runtime_config


def _write_template(path: str, columns: list[str]) -> str:
    template_path = project_path(path)
    ensure_dir(template_path.parent)
    if not template_path.exists():
        pd.DataFrame(columns=columns).to_csv(template_path, index=False, encoding="utf-8-sig")
    return str(template_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or scaffold the weekly inference snapshot dataset.")
    parser.add_argument("--runtime-config", default="config/data_pipeline.json")
    parser.add_argument("--feature-config", default=None)
    args = parser.parse_args()

    runtime = load_runtime_config(args.runtime_config)
    feature_config_path = args.feature_config or runtime["pipeline"]["baseline_feature_config"]
    feature_config = load_json_config(feature_config_path)

    required_cols = inference_required_columns(feature_config)
    optional_cols = inference_optional_columns(feature_config)
    template_columns = sorted(dict.fromkeys([*required_cols, *optional_cols]))
    template_path = _write_template(runtime["datasets"]["weekly_template_file"], template_columns)
    weekly_path = project_path(runtime["datasets"]["weekly_entry_file"])

    if not weekly_path.exists():
        message = {
            "status": "template_created",
            "message": "Weekly entry snapshot not found yet. Fill the template and rerun.",
            "template_path": template_path,
            "expected_weekly_entry_file": str(weekly_path),
            "required_columns": required_cols,
            "optional_columns": optional_cols,
        }
        print(json.dumps(message, ensure_ascii=False, indent=2))
        return

    try:
        weekly_df = pd.read_csv(weekly_path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        weekly_df = pd.read_csv(
            weekly_path,
            encoding=feature_config["data"].get("encoding", "cp932"),
            low_memory=False,
        )
    missing = [col for col in required_cols if col not in weekly_df.columns]
    if missing:
        raise ValueError(f"Weekly inference snapshot is missing columns: {missing}")
    for col in optional_cols:
        if col not in weekly_df.columns:
            weekly_df[col] = pd.NA

    out_root = ensure_dir(project_path(runtime["datasets"]["inference_root"], "weekly"))
    out_path = out_root / "weekly_inference_dataset.csv"
    weekly_df[template_columns].to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = {
        "status": "ready",
        "input_path": str(weekly_path),
        "output_path": str(out_path),
        "rows": int(len(weekly_df)),
        "required_columns": len(required_cols),
        "optional_columns": len(optional_cols),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
