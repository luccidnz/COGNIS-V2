from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from cognis.config import CeilingMode, MasteringConfig, MasteringMode


def _to_builtin(value: Any) -> Any:
    if is_dataclass(value):
        return _to_builtin(asdict(value))
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def serialize_json(data: Any) -> str:
    return json.dumps(_to_builtin(data), indent=2, sort_keys=True) + "\n"


def serialize_config(config: MasteringConfig) -> str:
    data = asdict(config)
    data["mode"] = data["mode"].value
    data["ceiling_mode"] = data["ceiling_mode"].value
    return serialize_json(data)


def deserialize_config(json_str: str) -> MasteringConfig:
    data = json.loads(json_str)
    data["mode"] = MasteringMode(data["mode"])
    data["ceiling_mode"] = CeilingMode(data["ceiling_mode"])
    return MasteringConfig(**data)


def serialize_recipe(recipe: dict[str, Any]) -> str:
    return serialize_json(recipe)
