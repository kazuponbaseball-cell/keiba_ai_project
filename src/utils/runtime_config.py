from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.paths import project_path


def load_runtime_config(path: str | Path = "config/data_pipeline.json") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_path(str(config_path))
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)

