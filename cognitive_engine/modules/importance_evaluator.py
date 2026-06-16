from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cognitive_engine.config.schema import ThresholdConfig
from cognitive_engine.core.types import ImportanceAssessment, MemoryBundle, SemanticState, tensor_to_numpy
from cognitive_engine.interfaces.base import ImportanceEvaluator


class AdaptiveImportanceEvaluator(ImportanceEvaluator):
    name = "adaptive_importance_evaluator"

    def __init__(self, thresholds: ThresholdConfig) -> None:
        self.thresholds = thresholds

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "thresholds": self.thresholds.__dict__}

    def evaluate(self, semantic_state: SemanticState, memory_bundle: MemoryBundle) -> ImportanceAssessment:
        if memory_bundle.semantic_long_term:
            similarities = [record.metadata.get("similarity", 0.0) for record in memory_bundle.semantic_long_term]
            redundancy = float(np.mean(similarities))
            novelty = 1.0 - redundancy
        else:
            redundancy = 0.0
            novelty = 1.0

        intent_weight = {
            "correction": 0.95,
            "knowledge_share": 0.85,
            "preference": 0.9,
            "question": 0.42,
            "small_talk": 0.12,
        }
        utility = intent_weight.get(semantic_state.intent, 0.55)
        concept_count = len(semantic_state.concepts)
        coherence = min(1.0, 0.35 + 0.1 * concept_count)
        frequency = min(1.0, len(memory_bundle.short_term) / 10.0)
        correction_signal = 1.0 if semantic_state.intent == "correction" else 0.15
        future_relevance = min(
            1.0,
            0.25 + 0.2 * concept_count + (0.25 if semantic_state.intent in {"knowledge_share", "preference"} else 0.0),
        )
        if semantic_state.intent == "preference":
            future_relevance = min(1.0, future_relevance + 0.1)
        contradiction_risk = 0.15 + (0.35 if semantic_state.intent == "correction" and redundancy > 0.7 else 0.0)
        importance_score = (
            novelty * 0.28
            + utility * 0.24
            + coherence * 0.12
            + future_relevance * 0.18
            + correction_signal * 0.1
            + frequency * 0.08
            - redundancy * 0.12
            - contradiction_risk * 0.08
        )
        importance_score = float(max(0.0, min(1.0, importance_score)))
        confidence_score = float(max(0.05, min(1.0, coherence * (1.0 - contradiction_risk * 0.5))))
        learning_priority = float(max(0.0, min(1.0, importance_score * 0.7 + confidence_score * 0.3)))

        if contradiction_risk > self.thresholds.contradiction and confidence_score < self.thresholds.uncertainty:
            action = "uncertain"
            rationale = "Potential contradiction detected, holding update until corroboration arrives."
        elif learning_priority >= self.thresholds.consolidate:
            action = "consolidate"
            rationale = "High-value information with durable relevance; consolidate into long-term semantic memory."
        elif learning_priority >= self.thresholds.reinforce:
            action = "reinforce"
            rationale = "Related knowledge already exists; reinforce instead of broad plastic updates."
        elif learning_priority >= self.thresholds.learn:
            action = "learn"
            rationale = "Novel information passed learning threshold and can be compressed for memory."
        elif semantic_state.intent == "preference" and confidence_score >= 0.65:
            action = "learn"
            rationale = "User preference retained despite moderate novelty because it affects future dialogue behavior."
        else:
            action = "ignore"
            rationale = "Signal is too weak or redundant to justify memory update."

        return ImportanceAssessment(
            novelty=novelty,
            utility=utility,
            frequency=frequency,
            coherence=coherence,
            correction_signal=correction_signal,
            redundancy=redundancy,
            future_relevance=future_relevance,
            contradiction_risk=contradiction_risk,
            importance_score=importance_score,
            confidence_score=confidence_score,
            learning_priority=learning_priority,
            action=action,
            rationale=rationale,
        )
