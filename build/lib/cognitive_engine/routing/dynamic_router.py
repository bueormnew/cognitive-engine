from __future__ import annotations

from typing import Any, Dict, Optional

from cognitive_engine.core.types import ProcessedInput, RoutingDecision, SemanticState
from cognitive_engine.interfaces.base import Router


class AdaptiveDynamicRouter(Router):
    name = "adaptive_dynamic_router"

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name}

    def route(self, processed: ProcessedInput, semantic_state: Optional[SemanticState] = None) -> RoutingDecision:
        if processed.modality == "numeric":
            return RoutingDecision(
                active_modules=["numeric_semantic_encoder", "plastic_arithmetic_module", "stable_reasoning_core"],
                consult_memories=["semantic_long_term", "episodic"],
                update_memory="target" in processed.metadata,
                engage_plasticity=True,
                compute_budget=0.45,
                rationale="Numeric task routed to lightweight plastic arithmetic pathway with optional replay updates.",
            )

        intent = semantic_state.intent if semantic_state is not None else processed.metadata.get("intent_hint", "statement")
        engage_plasticity = intent in {"knowledge_share", "preference", "correction"}
        consult_memories = ["short_term", "working_memory", "semantic_long_term"]
        if intent == "question":
            consult_memories.append("episodic")
        return RoutingDecision(
            active_modules=[
                "text_input_processor",
                "text_semantic_encoder",
                "adaptive_importance_evaluator",
                "semantic_knowledge_compressor",
                "stable_reasoning_core",
            ],
            consult_memories=consult_memories,
            update_memory=intent != "question",
            engage_plasticity=engage_plasticity,
            compute_budget=0.72 if engage_plasticity else 0.38,
            rationale=f"Text task routed by intent={intent}; memory lookup prioritized over global weight updates.",
        )

