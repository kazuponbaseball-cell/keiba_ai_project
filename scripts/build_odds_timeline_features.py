from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.odds_timeline import build_odds_timeline_features_from_file
from src.utils.paths import ensure_dir, project_path


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build market-movement features from an odds timeline CSV.")
    parser.add_argument("--timeline-csv", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    timeline_path = _resolve(args.timeline_csv)
    features = build_odds_timeline_features_from_file(timeline_path)
    output_path = _resolve(args.output_csv) if args.output_csv else timeline_path.with_name(timeline_path.stem + "_features.csv")
    ensure_dir(output_path.parent)
    features.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "timeline_csv": str(timeline_path),
                "output_csv": str(output_path),
                "rows": int(len(features)),
                "odds_valid_rows": int((features["odds_valid_snapshot_count"] > 0).sum()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
