from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.loaders import (
    inference_optional_columns,
    inference_required_columns,
    load_json_config,
)
from src.utils.paths import ensure_dir, project_path
from src.utils.runtime_config import load_runtime_config


def load_aliases(path: str | Path) -> dict[str, list[str]]:
    alias_path = Path(path)
    if not alias_path.is_absolute():
        alias_path = project_path(str(alias_path))
    with alias_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return None


def _build_snapshot(
    source: pd.DataFrame,
    *,
    required_cols: list[str],
    optional_cols: list[str],
    aliases: dict[str, list[str]],
) -> pd.DataFrame:
    snapshot = pd.DataFrame(index=source.index)
    all_cols = list(dict.fromkeys([*required_cols, *optional_cols]))

    for col in all_cols:
        if col in source.columns:
            snapshot[col] = source[col]
            continue
        picked = _pick_column(source, aliases.get(col, []))
        if picked is not None:
            snapshot[col] = picked
            continue
        snapshot[col] = pd.NA

    if "レースID(新/馬番無)" not in source.columns and "レースID(新)" in source.columns:
        snapshot["レースID(新/馬番無)"] = source["レースID(新)"].astype("string")

    if "日付S" in snapshot.columns and snapshot["日付S"].isna().all() and "日付(yyyy.mm.dd)" in source.columns:
        snapshot["日付S"] = source["日付(yyyy.mm.dd)"]

    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TARGET entry CSV into the project's weekly entry snapshot.")
    parser.add_argument("--input-csv", required=True, help="Path to a TARGET entry CSV export.")
    parser.add_argument("--runtime-config", default="config/data_pipeline.json")
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--alias-config", default="config/target_entry_aliases.json")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--filter-date", default=None, help="Optional 日付 filter, e.g. 260510 or 20260510 depending on source.")
    parser.add_argument("--filter-race-id", default=None, help="Optional レースID(新/馬番無) filter.")
    args = parser.parse_args()

    runtime = load_runtime_config(args.runtime_config)
    feature_config_path = args.feature_config or runtime["pipeline"]["baseline_feature_config"]
    feature_config = load_json_config(feature_config_path)
    aliases = load_aliases(args.alias_config)

    required_cols = inference_required_columns(feature_config)
    optional_cols = inference_optional_columns(feature_config)
    source_path = project_path(args.input_csv)
    source = pd.read_csv(source_path, encoding=feature_config["data"].get("encoding", "cp932"), low_memory=False)

    snapshot = _build_snapshot(
        source,
        required_cols=required_cols,
        optional_cols=optional_cols,
        aliases=aliases,
    )

    if args.filter_date and "日付" in snapshot.columns:
        snapshot = snapshot[snapshot["日付"].astype("string") == str(args.filter_date)].copy()
    if args.filter_race_id and "レースID(新/馬番無)" in snapshot.columns:
        snapshot = snapshot[snapshot["レースID(新/馬番無)"].astype("string") == str(args.filter_race_id)].copy()

    if snapshot.empty:
        raise ValueError("No rows matched the provided filters.")

    missing_required = [col for col in required_cols if snapshot[col].isna().all()]
    output_path = project_path(
        args.output_csv or runtime["datasets"]["weekly_entry_file"]
    )
    ensure_dir(output_path.parent)
    snapshot.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_csv": str(source_path),
        "output_csv": str(output_path),
        "rows": int(len(snapshot)),
        "required_columns": len(required_cols),
        "optional_columns": len(optional_cols),
        "all_null_required_columns": missing_required,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
