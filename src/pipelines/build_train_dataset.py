from __future__ import annotations

import argparse
import json

from src.data.loaders import load_historical_csv, load_json_config, required_columns
from src.features.baseline import prepare_training_frame, split_by_recent_dates
from src.jv.build_normalized_tables import build_normalized_tables
from src.utils.paths import ensure_dir, project_path
from src.utils.runtime_config import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized tables and a baseline training dataset.")
    parser.add_argument("--runtime-config", default="config/data_pipeline.json")
    parser.add_argument("--feature-config", default=None)
    args = parser.parse_args()

    runtime = load_runtime_config(args.runtime_config)
    feature_config_path = args.feature_config or runtime["pipeline"]["baseline_feature_config"]
    feature_config = load_json_config(feature_config_path)

    raw = load_historical_csv(feature_config, columns=required_columns(feature_config, for_prediction=True))
    frame = prepare_training_frame(raw, feature_config)
    normalized = build_normalized_tables(args.runtime_config, feature_config_path, frame=frame)
    train_df, test_df, cutoff = split_by_recent_dates(frame, feature_config)

    out_root = ensure_dir(project_path(runtime["datasets"]["train_root"]))
    train_path = out_root / "baseline_train_dataset.csv"
    test_path = out_root / "baseline_temporal_test_dataset.csv"
    metadata_path = out_root / "baseline_train_dataset_metadata.json"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    metadata = {
        "normalized": normalized,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "temporal_test_cutoff_date": int(cutoff),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
