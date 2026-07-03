from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild train datasets and retrain the baseline ranker.")
    parser.add_argument("--runtime-config", default="config/data_pipeline.json")
    parser.add_argument("--feature-config", default="config/baseline_features.json")
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()

    subprocess.run(
        [
            args.python_exe,
            "-m",
            "src.pipelines.build_train_dataset",
            "--runtime-config",
            args.runtime_config,
            "--feature-config",
            args.feature_config,
        ],
        check=True,
    )
    subprocess.run(
        [
            args.python_exe,
            "-m",
            "src.train.train_baseline",
            "--config",
            args.feature_config,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
