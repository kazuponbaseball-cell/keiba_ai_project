from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate extracted TARGET workout CSV files.")
    parser.add_argument("--input-csv", nargs="+", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    frames = []
    for path_text in args.input_csv:
        path = Path(path_text)
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    subset = [col for col in ["source_type", "tracen_kubun", "workout_date", "workout_time", "horse_id", "course"] if col in combined.columns]
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "input_files": args.input_csv,
                "rows": int(len(combined)),
                "workout_date_min": str(combined["workout_date"].min()) if "workout_date" in combined.columns else None,
                "workout_date_max": str(combined["workout_date"].max()) if "workout_date" in combined.columns else None,
                "output_csv": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
