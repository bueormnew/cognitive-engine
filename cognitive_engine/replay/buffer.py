from __future__ import annotations

import random
from collections import deque
from typing import Any, Deque, Dict, List

from cognitive_engine.core.types import ReplaySample
from cognitive_engine.interfaces.base import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    name = "prioritized_replay_buffer"

    def __init__(self, capacity: int = 512) -> None:
        self.capacity = capacity
        self.buffer: Deque[ReplaySample] = deque(maxlen=capacity)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "capacity": self.capacity, "size": len(self.buffer)}

    def add(self, sample: ReplaySample) -> None:
        self.buffer.append(sample)

    def sample(self, batch_size: int) -> List[ReplaySample]:
        if not self.buffer:
            return []
        batch_size = min(batch_size, len(self.buffer))
        weights = [max(sample.priority, 1e-4) for sample in self.buffer]
        return random.choices(list(self.buffer), weights=weights, k=batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def rescale(self, factor: float = 0.97) -> int:
        touched = 0
        for sample in self.buffer:
            sample.priority *= factor
            touched += 1
        return touched

