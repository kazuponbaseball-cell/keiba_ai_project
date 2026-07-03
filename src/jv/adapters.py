from __future__ import annotations

import subprocess
from pathlib import Path

from src.utils.paths import ensure_dir, project_path
from src.utils.runtime_config import load_runtime_config


def prepare_source_drop(config_path: str = "config/data_pipeline.json") -> dict[str, str]:
    runtime = load_runtime_config(config_path)
    jv_cfg = runtime["jv_data"]
    provider = jv_cfg.get("provider", {})
    source_dir = ensure_dir(project_path(jv_cfg["source_drop_dir"]))
    provider_name = provider.get("name", "local_drop")

    if provider_name == "local_drop":
        return {
            "provider": provider_name,
            "source_dir": str(source_dir),
            "message": "Waiting for raw JV export files to be placed in the local drop directory.",
        }

    if provider_name == "external_command":
        command = str(provider.get("external_command", "")).strip()
        if not command:
            raise ValueError("JV provider is external_command but no external_command is configured.")
        workdir_value = str(provider.get("external_workdir", "")).strip()
        workdir = project_path(workdir_value) if workdir_value else project_path()
        subprocess.run(command, cwd=workdir, shell=True, check=True)
        return {
            "provider": provider_name,
            "source_dir": str(source_dir),
            "message": "External JV export command completed.",
        }

    raise ValueError(f"Unsupported JV provider: {provider_name}")
