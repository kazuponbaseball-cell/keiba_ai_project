from __future__ import annotations

import argparse
import json

from src.jv.fetch_jv_data import snapshot_jv_drop
from src.jv.parse_jv_data import parse_latest_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive and inventory the latest JV raw drop.")
    parser.add_argument("--config", default="config/data_pipeline.json")
    args = parser.parse_args()

    archived = snapshot_jv_drop(args.config)
    parsed = parse_latest_archive(args.config)
    print(json.dumps({"archived": archived, "parsed": parsed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
