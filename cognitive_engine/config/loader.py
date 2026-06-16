from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

from cognitive_engine.config.schema import EngineConfig, MemoryConfig, NumericDemoConfig, ThresholdConfig


T = TypeVar("T")


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _construct(dataclass_type: Type[T], data: Dict[str, Any]) -> T:
    if dataclass_type is EngineConfig:
        thresholds = _construct(ThresholdConfig, data.get("thresholds", {}))
        memory = _construct(MemoryConfig, data.get("memory", {}))
        numeric_demo = _construct(NumericDemoConfig, data.get("numeric_demo", {}))
        payload = {**data, "thresholds": thresholds, "memory": memory, "numeric_demo": numeric_demo}
        return dataclass_type(**payload)
    return dataclass_type(**data)


def load_engine_config(path: str | Path | None = None) -> EngineConfig:
    defaults = asdict(EngineConfig())
    if path is None:
        return EngineConfig()

    config_path = Path(path)
    override = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    merged = _merge_dicts(defaults, override)
    return _construct(EngineConfig, merged)

