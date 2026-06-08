from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


CONFIG_ENV_VAR = "AFML_CONFIG_PATH"
DATA_CONFIG_ENV_VAR = "AFML_DATA_CONFIG_PATH"

DEFAULT_DATA_CONFIG_PATH = Path(__file__).resolve().with_name("afml_data_config.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root() / path


def _resolve_config_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return resolve_repo_path(path)


def load_afml_data_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = _resolve_config_path(
        path or os.environ.get(CONFIG_ENV_VAR) or os.environ.get(DATA_CONFIG_ENV_VAR, DEFAULT_DATA_CONFIG_PATH),
    )
    if not config_path.exists():
        raise FileNotFoundError(f"AFML data config does not exist: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"AFML data config root must be a JSON object: {config_path}")
    payload["_config_path"] = str(config_path)
    payload["_data_config_path"] = str(config_path)
    return payload


def load_afml_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_afml_data_config(path)


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section {name!r} must be a JSON object.")
    return value


def string_tuple(value: Any, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def int_tuple(value: Any, *, default: tuple[int, ...] = ()) -> tuple[int, ...]:
    if value is None:
        return default
    return tuple(int(item) for item in value)
