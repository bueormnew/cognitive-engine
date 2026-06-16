from __future__ import annotations

from collections import Counter, deque
from typing import Any, Deque, Dict, List

import numpy as np

from cognitive_engine.config.schema import MemoryConfig
from cognitive_engine.core.types import CompressedKnowledge, ConsolidationReport, MemoryBundle, MemoryQuery, SemanticState
from cognitive_engine.memory.vector_store import NumpyVectorIndex


class HierarchicalMemorySystem:
    name = "hierarchical_memory_system"

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.embedding_dim = 64
        self.short_term: Deque[CompressedKnowledge] = deque(maxlen=config.short_term_capacity)
        self.episodic: Deque[CompressedKnowledge] = deque(maxlen=config.episodic_capacity)
        self.semantic_records: Dict[str, CompressedKnowledge] = {}
        self.vector_index = NumpyVectorIndex()
        self.working_state: Dict[str, Any] = {
            "active_intent": None,
            "active_concepts": [],
            "recent_entities": [],
            "session_goals": [],
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "short_term_capacity": self.config.short_term_capacity,
            "episodic_capacity": self.config.episodic_capacity,
            "semantic_capacity": self.config.semantic_capacity,
        }

    def update_working_memory(self, semantic_state: SemanticState) -> Dict[str, Any]:
        concepts = [concept.label for concept in semantic_state.concepts[:6]]
        self.working_state["active_intent"] = semantic_state.intent
        self.working_state["active_concepts"] = concepts
        self.working_state["recent_entities"] = semantic_state.entities[:6]
        return dict(self.working_state)

    def retrieve(self, query: MemoryQuery) -> MemoryBundle:
        semantic_hits = []
        aligned_query = self._align_embedding(query.embedding)
        query_roots = {concept.lower()[:4] for concept in query.concepts if len(concept) >= 4}
        scored_records = []
        for record_id, record in self.semantic_records.items():
            record_vector = record.embedding / (np.linalg.norm(record.embedding) or 1.0)
            query_vector = aligned_query / (np.linalg.norm(aligned_query) or 1.0)
            vector_score = float(np.dot(query_vector, record_vector))
            record_roots = {concept.lower()[:4] for concept in record.concepts if len(concept) >= 4}
            lexical_score = 0.0
            if query_roots:
                lexical_score = len(query_roots & record_roots) / len(query_roots)
            score = 0.65 * vector_score + 0.35 * lexical_score
            if query.intent == "question" and lexical_score > 0:
                score += 0.25 * lexical_score
            if query.intent == "question" and any(root in {"pref", "gust"} for root in query_roots):
                if record.provenance.get("intent") == "preference" or "preference" in record.summary.lower():
                    score += 0.35
            scored_records.append((record_id, score))

        scored_records.sort(key=lambda item: item[1], reverse=True)
        for record_id, score in scored_records[: query.top_k]:
            record = self.semantic_records[record_id]
            enriched = CompressedKnowledge(
                **{
                    **record.__dict__,
                    "metadata": {**record.metadata, "similarity": score},
                }
            )
            semantic_hits.append(enriched)
        return MemoryBundle(
            short_term=list(self.short_term)[-query.top_k:],
            working_memory=dict(self.working_state),
            semantic_long_term=semantic_hits,
            episodic=list(self.episodic)[-query.top_k:],
        )

    def write(self, knowledge: CompressedKnowledge) -> None:
        knowledge.embedding = self._align_embedding(knowledge.embedding)
        self.short_term.append(knowledge)
        self.episodic.append(knowledge)
        existing_id = self._find_duplicate(knowledge)
        if existing_id:
            existing = self.semantic_records[existing_id]
            existing.reinforced_count += 1
            existing.importance = max(existing.importance, knowledge.importance)
            existing.confidence = max(existing.confidence, knowledge.confidence)
            existing.summary = knowledge.summary if knowledge.importance >= existing.importance else existing.summary
            existing.concepts = list(dict.fromkeys(existing.concepts + knowledge.concepts))
            self.vector_index.upsert(existing_id, existing.embedding)
            return

        if len(self.semantic_records) >= self.config.semantic_capacity:
            weakest = min(
                self.semantic_records.items(),
                key=lambda item: item[1].importance * item[1].confidence * max(item[1].reinforced_count, 1),
            )[0]
            self.semantic_records.pop(weakest, None)
            self.vector_index.delete(weakest)
        self.semantic_records[knowledge.record_id] = knowledge
        self.vector_index.upsert(knowledge.record_id, knowledge.embedding)

    def snapshot(self) -> Dict[str, Any]:
        concept_counter = Counter()
        for record in self.semantic_records.values():
            concept_counter.update(record.concepts)
        return {
            "short_term_size": len(self.short_term),
            "episodic_size": len(self.episodic),
            "semantic_size": len(self.semantic_records),
            "top_concepts": concept_counter.most_common(10),
            "working_state": dict(self.working_state),
        }

    def consolidate(self) -> ConsolidationReport:
        merged = 0
        pruned = 0
        records = list(self.semantic_records.values())
        visited = set()
        for left in records:
            if left.record_id in visited:
                continue
            for right in records:
                if left.record_id == right.record_id or right.record_id in visited:
                    continue
                similarity = float(np.dot(left.embedding, right.embedding))
                overlap = len(set(left.concepts) & set(right.concepts))
                if similarity > 0.97 or overlap >= 3:
                    left.reinforced_count += right.reinforced_count
                    left.importance = max(left.importance, right.importance)
                    left.confidence = max(left.confidence, right.confidence)
                    left.concepts = list(dict.fromkeys(left.concepts + right.concepts))
                    visited.add(right.record_id)
                    merged += 1
        for record_id in visited:
            self.semantic_records.pop(record_id, None)
            self.vector_index.delete(record_id)

        if len(self.semantic_records) > self.config.semantic_capacity:
            items = sorted(
                self.semantic_records.values(),
                key=lambda record: record.importance * record.confidence * max(record.reinforced_count, 1),
                reverse=True,
            )
            keep = {record.record_id for record in items[: self.config.semantic_capacity]}
            for record_id in list(self.semantic_records):
                if record_id not in keep:
                    self.semantic_records.pop(record_id, None)
                    self.vector_index.delete(record_id)
                    pruned += 1
        return ConsolidationReport(
            merged_records=merged,
            pruned_records=pruned,
            replay_reweighted=0,
            notes="Merged highly similar concepts and trimmed low-value semantic residues.",
        )

    def _find_duplicate(self, knowledge: CompressedKnowledge) -> str | None:
        hits = self.vector_index.search(knowledge.embedding, top_k=1)
        if not hits:
            return None
        record_id, score = hits[0]
        if score > 0.985 or len(set(self.semantic_records[record_id].concepts) & set(knowledge.concepts)) >= 3:
            return record_id
        return None

    def _align_embedding(self, embedding: np.ndarray) -> np.ndarray:
        if embedding.shape[0] < self.embedding_dim:
            return np.pad(embedding, (0, self.embedding_dim - embedding.shape[0]))
        if embedding.shape[0] > self.embedding_dim:
            return embedding[: self.embedding_dim]
        return embedding
