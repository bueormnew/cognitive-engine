from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ThresholdConfig:
    learn: float = 0.62
    reinforce: float = 0.75
    consolidate: float = 0.82
    uncertainty: float = 0.45
    contradiction: float = 0.55


@dataclass
class MemoryConfig:
    short_term_capacity: int = 24
    episodic_capacity: int = 128
    semantic_capacity: int = 512
    replay_capacity: int = 512
    retrieval_top_k: int = 5
    consolidation_interval: int = 10


@dataclass
class NumericDemoConfig:
    operations: List[str] = field(default_factory=lambda: ["add", "sub", "mul"])
    max_operand: int = 12
    train_size: int = 2048
    val_size: int = 512
    batch_size: int = 64
    epochs: int = 25
    replay_batch_size: int = 32


@dataclass
class EngineConfig:
    seed: int = 13
    device: str = "cpu"
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    numeric_demo: NumericDemoConfig = field(default_factory=NumericDemoConfig)
    language_hints: Dict[str, str] = field(default_factory=lambda: {"default": "multilingual-lite"})

