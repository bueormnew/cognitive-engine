from __future__ import annotations

from typing import Any, Dict

import torch

from cognitive_engine.core.types import ProcessedInput
from cognitive_engine.interfaces.base import InputProcessor
from cognitive_engine.utils.numeric import OPERATIONS
from cognitive_engine.utils.text import detect_intent, hashed_token_ids, normalize_text, tokenize_text


class TextInputProcessor(InputProcessor):
    name = "text_input_processor"

    def __init__(self, vocab_size: int = 4096, max_length: int = 96, device: str = "cpu") -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.device = device

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "vocab_size": self.vocab_size, "max_length": self.max_length}

    def supports(self, payload: Any) -> bool:
        return isinstance(payload, str)

    def process(self, payload: Any) -> ProcessedInput:
        text = normalize_text(str(payload))
        tokens = tokenize_text(text)[: self.max_length]
        token_ids = hashed_token_ids(tokens, self.vocab_size) or [0]
        tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        attention = torch.ones_like(tensor)
        return ProcessedInput(
            raw_input=payload,
            modality="text",
            normalized_text=text,
            tokens=tokens,
            token_tensor=tensor,
            attention_mask=attention,
            metadata={"intent_hint": detect_intent(text)},
        )


class NumericInputProcessor(InputProcessor):
    name = "numeric_input_processor"

    def __init__(self, device: str = "cpu", scale: float = 12.0) -> None:
        self.device = device
        self.scale = scale
        self.operation_to_id = dict(OPERATIONS)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "scale": self.scale, "operations": self.operation_to_id}

    def supports(self, payload: Any) -> bool:
        return isinstance(payload, dict) and payload.get("modality") == "numeric"

    def process(self, payload: Any) -> ProcessedInput:
        left = float(payload["a"])
        right = float(payload["b"])
        operation = str(payload["operation"])
        features = torch.tensor([[left / self.scale, right / self.scale]], dtype=torch.float32, device=self.device)
        metadata = {
            "a": left,
            "b": right,
            "scale": self.scale,
            "operation_id": self.operation_to_id[operation],
            "intent_hint": "knowledge_share" if "target" in payload else "question",
        }
        if "target" in payload:
            metadata["target"] = float(payload["target"])
        return ProcessedInput(
            raw_input=payload,
            modality="numeric",
            numeric_tensor=features,
            operation=operation,
            metadata=metadata,
        )
