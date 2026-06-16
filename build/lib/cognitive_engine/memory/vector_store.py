from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class NumpyVectorIndex:
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    def upsert(self, record_id: str, embedding: np.ndarray) -> None:
        norm = np.linalg.norm(embedding) or 1.0
        self.embeddings[record_id] = embedding.astype(np.float32) / norm

    def delete(self, record_id: str) -> None:
        self.embeddings.pop(record_id, None)

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.embeddings:
            return []
        query = embedding.astype(np.float32)
        query /= np.linalg.norm(query) or 1.0
        scores = [(record_id, float(np.dot(query, stored))) for record_id, stored in self.embeddings.items()]
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

