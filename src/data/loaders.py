from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import project_path


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_path(str(config_path))
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def required_columns(config: dict[str, Any], *, for_prediction: bool = False) -> list[str]:
    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    columns = {
        data_cfg["race_id_column"],
        data_cfg["horse_id_column"],
        data_cfg["horse_name_column"],
        data_cfg["date_column"],
        data_cfg["rank_column"],
        data_cfg["abnormal_column"],
        *config["numeric_features"],
        *config["categorical_features"],
        *config.get("feature_source_columns", []),
        *config.get("optional_feature_source_columns", []),
    }
    for optional_col in [
        train_cfg.get("race_type_column"),
        train_cfg.get("surface_column"),
        train_cfg.get("venue_column"),
    ]:
        if optional_col:
            columns.add(optional_col)
    if for_prediction:
        columns.update(config.get("passthrough_prediction_columns", []))
    return sorted(columns)


def inference_required_columns(config: dict[str, Any]) -> list[str]:
    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    allowed_prefixes = list(config.get("leakage_allowed_prefixes", []))
    inference_source_columns = {
        col
        for col in config.get("feature_source_columns", [])
        if any(str(col).startswith(prefix) for prefix in allowed_prefixes)
    }

    columns = {
        data_cfg["race_id_column"],
        data_cfg["horse_id_column"],
        data_cfg["horse_name_column"],
        data_cfg["date_column"],
        *config["numeric_features"],
        *config["categorical_features"],
        *inference_source_columns,
    }
    for optional_col in [
        train_cfg.get("race_type_column"),
        train_cfg.get("surface_column"),
        train_cfg.get("venue_column"),
    ]:
        if optional_col:
            columns.add(optional_col)
    return sorted(columns)


def inference_optional_columns(config: dict[str, Any]) -> list[str]:
    data_cfg = config["data"]
    columns = set(config.get("passthrough_prediction_columns", []))
    columns.update(config.get("optional_feature_source_columns", []))
    columns.add(data_cfg["abnormal_column"])
    columns.add(data_cfg["rank_column"])
    return sorted(columns)


def model_numeric_features(config: dict[str, Any]) -> list[str]:
    return [*config["numeric_features"], *config.get("generated_numeric_features", [])]


def model_categorical_features(config: dict[str, Any]) -> list[str]:
    return [*config["categorical_features"], *config.get("generated_categorical_features", [])]


def load_historical_csv(config: dict[str, Any], *, columns: list[str] | None = None) -> pd.DataFrame:
    csv_path = project_path(config["data"]["historical_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Historical CSV not found: {csv_path}")
    return pd.read_csv(
        csv_path,
        encoding=config["data"].get("encoding", "cp932"),
        usecols=columns,
        low_memory=False,
    )
