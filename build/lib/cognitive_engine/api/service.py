from __future__ import annotations

from typing import Any, Dict

from cognitive_engine.core.engine import CognitiveEngine


class CognitiveAPIService:
    def __init__(self, engine: CognitiveEngine) -> None:
        self.engine = engine

    def ingest_text(self, text: str) -> Dict[str, Any]:
        response = self.engine.process(text, allow_learning=True)
        return self._serialize(response)

    def ask_text(self, text: str) -> Dict[str, Any]:
        response = self.engine.process(text, allow_learning=False)
        return self._serialize(response)

    def train_numeric(self, a: float, b: float, operation: str, target: float) -> Dict[str, Any]:
        response = self.engine.process(
            {"modality": "numeric", "a": a, "b": b, "operation": operation, "target": target},
            allow_learning=True,
        )
        return self._serialize(response)

    def infer_numeric(self, a: float, b: float, operation: str) -> Dict[str, Any]:
        response = self.engine.process({"modality": "numeric", "a": a, "b": b, "operation": operation}, allow_learning=False)
        return self._serialize(response)

    def _serialize(self, response: Any) -> Dict[str, Any]:
        return {
            "text": response.text,
            "importance_action": response.importance.action,
            "importance_score": response.importance.importance_score,
            "confidence_score": response.inference.confidence,
            "learning_applied": response.learning_applied,
            "trace_count": len(response.traces),
            "memory_snapshot": self.engine.snapshot(),
        }

