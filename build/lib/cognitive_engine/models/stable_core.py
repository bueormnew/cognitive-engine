from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from cognitive_engine.core.types import CoreInference, MemoryBundle, RoutingDecision, SemanticState
from cognitive_engine.interfaces.base import StableCore


class StableReasoningCore(nn.Module, StableCore):
    name = "stable_reasoning_core"

    def __init__(self, latent_dim: int = 64, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.projector = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh(), nn.Linear(latent_dim, latent_dim))
        for parameter in self.projector.parameters():
            parameter.requires_grad = False
        self.to(self.device)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "latent_dim": 64, "frozen": True}

    def infer(
        self,
        semantic_state: SemanticState,
        memory_bundle: MemoryBundle,
        routing: RoutingDecision,
        plastic_output: Optional[CoreInference] = None,
    ) -> CoreInference:
        latent = self._project(semantic_state.pooled_embedding.float().to(self.device))
        if semantic_state.modality == "numeric" and plastic_output is not None:
            summaries = [record.summary for record in memory_bundle.semantic_long_term[:2]]
            memory_hint = " | ".join(summaries) if summaries else "No previous arithmetic consolidation available."
            explanation = (
                f"Resultado estimado: {plastic_output.prediction:.3f}. "
                f"Confianza={plastic_output.confidence:.3f}. "
                f"Memoria relevante: {memory_hint}"
            )
            return CoreInference(
                prediction=plastic_output.prediction,
                confidence=plastic_output.confidence,
                explanation=explanation,
                hidden_state=latent.squeeze(0),
                artifacts={"memory_hint": memory_hint, "route": routing.rationale},
            )

        summaries = [record.summary for record in memory_bundle.semantic_long_term[:3]]
        current_action = semantic_state.metadata.get("importance_action", "observe")
        if semantic_state.intent == "question":
            if summaries:
                answer = " | ".join(summaries)
            else:
                answer = "No hay conocimiento consolidado suficiente; la consulta queda en modo exploratorio."
        elif current_action in {"learn", "reinforce", "consolidate"}:
            answer = f"Información analizada y comprimida como: {semantic_state.compressed_context}"
        else:
            answer = f"Información observada sin actualización fuerte de memoria: {semantic_state.compressed_context}"
        return CoreInference(
            prediction=answer,
            confidence=0.72,
            explanation=answer,
            hidden_state=latent.squeeze(0),
            artifacts={
                "active_concepts": memory_bundle.working_memory.get("active_concepts", []),
                "retrieved_summaries": summaries,
            },
        )

    def _project(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        vector = pooled_embedding.unsqueeze(0)
        if vector.shape[-1] < 64:
            vector = torch.nn.functional.pad(vector, (0, 64 - vector.shape[-1]))
        elif vector.shape[-1] > 64:
            vector = vector[:, :64]
        return self.projector(vector)

