from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path("resources/pivot_config.yaml")

_cached_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")

    with _CONFIG_PATH.open("r") as f:
        user_config = yaml.safe_load(f) or {}

    if not isinstance(user_config, dict):
        raise ValueError(f"Top-level config in {_CONFIG_PATH} must be a mapping")

    _cached_config = user_config
    return _cached_config
