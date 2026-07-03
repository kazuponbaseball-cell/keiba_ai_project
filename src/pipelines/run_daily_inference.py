from __future__ import annotations

import argparse
import json
import subprocess
import sys

from src.utils.paths import project_path
from src.utils.runtime_config import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the weekly inference dataset and run baseline inference.")
    parser.add_argument("--runtime-config", default="config/data_pipeline.json")
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()

    runtime = load_runtime_config(args.runtime_config)
    feature_config_path = args.feature_config or runtime["pipeline"]["baseline_feature_config"]

    build_cmd = [
        args.python_exe,
        "-m",
        "src.pipelines.build_weekly_inference_dataset",
        "--runtime-config",
        args.runtime_config,
        "--feature-config",
        feature_config_path,
    ]
    subprocess.run(build_cmd, check=True)

    input_csv = runtime["datasets"]["weekly_entry_file"]
    if not project_path(input_csv).exists():
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "message": "Weekly entry snapshot not found yet. Template has been prepared instead.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    predict_cmd = [
        args.python_exe,
        "-m",
        "src.predict.predict_baseline",
        "--config",
        feature_config_path,
        "--model",
        runtime["pipeline"]["baseline_model_path"],
        "--input-csv",
        input_csv,
    ]
    subprocess.run(predict_cmd, check=True)


if __name__ == "__main__":
    main()
