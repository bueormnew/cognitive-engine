from __future__ import annotations

from typing import Any, Dict, List, Tuple
from uuid import uuid4

from cognitive_engine.core.types import CompressedKnowledge, ImportanceAssessment, SemanticState, tensor_to_numpy
from cognitive_engine.interfaces.base import KnowledgeCompressor


class SemanticKnowledgeCompressor(KnowledgeCompressor):
    name = "semantic_knowledge_compressor"

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name}

    def compress(self, semantic_state: SemanticState, importance: ImportanceAssessment) -> CompressedKnowledge:
        concepts = [concept.label for concept in semantic_state.concepts]
        relations = self._build_relations(semantic_state, concepts)
        summary = self._rewrite_summary(semantic_state, concepts, importance.action)
        return CompressedKnowledge(
            record_id=str(uuid4()),
            source_type=semantic_state.modality,
            summary=summary,
            concepts=concepts,
            relations=relations,
            embedding=tensor_to_numpy(semantic_state.pooled_embedding.float()),
            importance=importance.importance_score,
            confidence=importance.confidence_score,
            provenance={"intent": semantic_state.intent, "compressed_context": semantic_state.compressed_context},
            metadata={"action": importance.action, "raw_modality": semantic_state.modality},
        )

    def _build_relations(self, semantic_state: SemanticState, concepts: List[str]) -> List[Tuple[str, str, str]]:
        if len(concepts) < 2:
            return [(concepts[0], semantic_state.intent, semantic_state.compressed_context)] if concepts else []
        root = concepts[0]
        return [(root, semantic_state.intent, concept) for concept in concepts[1:4]]

    def _rewrite_summary(self, semantic_state: SemanticState, concepts: List[str], action: str) -> str:
        if semantic_state.modality == "numeric":
            a = semantic_state.metadata["a"]
            b = semantic_state.metadata["b"]
            op = semantic_state.metadata["operation"]
            return f"Arithmetic pattern observed: {a:g} {op} {b:g} with action={action}"
        concept_text = ", ".join(concepts[:4]) if concepts else semantic_state.compressed_context
        if semantic_state.intent == "preference":
            return f"User preference extracted: {concept_text}"
        if semantic_state.intent == "correction":
            return f"Correction compressed: {concept_text}"
        if semantic_state.intent == "knowledge_share":
            return f"Knowledge compressed: {concept_text}"
        return f"Contextual memory: {semantic_state.compressed_context}"

