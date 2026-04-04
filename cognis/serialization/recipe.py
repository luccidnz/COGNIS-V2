import json
from dataclasses import asdict
from typing import Any, Dict
from cognis.config import MasteringConfig, MasteringMode, CeilingMode

def serialize_config(config: MasteringConfig) -> str:
    d = asdict(config)
    d['mode'] = d['mode'].value
    d['ceiling_mode'] = d['ceiling_mode'].value
    return json.dumps(d, indent=2)

def deserialize_config(json_str: str) -> MasteringConfig:
    d = json.loads(json_str)
    d['mode'] = MasteringMode(d['mode'])
    d['ceiling_mode'] = CeilingMode(d['ceiling_mode'])
    return MasteringConfig(**d)

def serialize_recipe(recipe: Dict[str, Any]) -> str:
    return json.dumps(recipe, indent=2)
