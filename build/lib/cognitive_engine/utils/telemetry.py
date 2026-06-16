from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from cognitive_engine.core.types import TraceEvent
from cognitive_engine.interfaces.base import TelemetrySink


class JsonlTelemetrySink(TelemetrySink):
    name = "jsonl_telemetry_sink"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[Dict[str, Any]] = []

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "path": str(self.path)}

    def emit(self, event: TraceEvent) -> None:
        self._buffer.append(
            {
                "stage": event.stage,
                "message": event.message,
                "payload": event.payload,
                "timestamp": event.timestamp.isoformat(),
            }
        )

    def flush(self) -> None:
        if not self._buffer:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for item in self._buffer:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._buffer.clear()

