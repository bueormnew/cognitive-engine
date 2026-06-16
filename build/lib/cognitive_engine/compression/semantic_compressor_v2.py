from __future__ import annotations

from typing import Any, Dict, List
from uuid import uuid4

from cognitive_engine.compression.knowledge_compressor import SemanticKnowledgeCompressor
from cognitive_engine.core.types import (
    CompressionBundle,
    GraphEdge,
    GraphNode,
    GraphPatch,
    ImportanceAssessment,
    KnowledgeTriple,
    PreferenceRecord,
    ProcedureMemory,
    SemanticState,
)


class SemanticCompressorV2(SemanticKnowledgeCompressor):
    name = "semantic_compressor_v2"

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "base": "semantic_knowledge_compressor"}

    def compress_v2(self, semantic_state: SemanticState, importance: ImportanceAssessment, project_id: str = "global") -> CompressionBundle:
        knowledge = self.compress(semantic_state, importance)
        concepts = [concept.label for concept in semantic_state.concepts]
        triples = self._triples(semantic_state, concepts)
        graph_patch = self._graph_patch(semantic_state, concepts, triples, project_id)
        preferences = self._preferences(semantic_state, concepts)
        procedures = self._procedures(semantic_state, concepts)
        return CompressionBundle(
            compressed_knowledge=[knowledge],
            triples=triples,
            graph_patch=graph_patch,
            preferences=preferences,
            procedures=procedures,
            validation_requests=[] if importance.confidence_score > 0.65 else [f"Validate low confidence memory: {knowledge.summary}"],
        )

    def _triples(self, semantic_state: SemanticState, concepts: List[str]) -> List[KnowledgeTriple]:
        triples: List[KnowledgeTriple] = []
        if not concepts:
            return triples
        subject = concepts[0]
        for concept in concepts[1:5]:
            triples.append(KnowledgeTriple(subject=subject, relation=semantic_state.intent, object=concept, confidence=0.75))
        if semantic_state.intent == "preference":
            triples.append(KnowledgeTriple(subject="user", relation="prefers", object=", ".join(concepts[:4]), confidence=0.82))
        return triples

    def _graph_patch(self, semantic_state: SemanticState, concepts: List[str], triples: List[KnowledgeTriple], project_id: str) -> GraphPatch:
        nodes: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []

        def node(label: str, node_type: str) -> GraphNode:
            key = f"{node_type}:{label}".lower()
            if key not in nodes:
                nodes[key] = GraphNode(node_id=str(uuid4()), node_type=node_type, label=label, project_id=project_id, confidence=0.75)
            return nodes[key]

        intent_node = node(semantic_state.intent, "Concept")
        for concept in concepts:
            concept_node = node(concept, "Concept")
            edges.append(
                GraphEdge(
                    edge_id=str(uuid4()),
                    source_id=intent_node.node_id,
                    target_id=concept_node.node_id,
                    relation="mentions",
                    confidence=0.7,
                )
            )
        for triple in triples:
            source = node(triple.subject, "Concept")
            target = node(triple.object, "Concept")
            edges.append(
                GraphEdge(
                    edge_id=str(uuid4()),
                    source_id=source.node_id,
                    target_id=target.node_id,
                    relation=triple.relation,
                    confidence=triple.confidence,
                )
            )
        return GraphPatch(nodes=list(nodes.values()), edges=edges, source="semantic_compressor_v2")

    def _preferences(self, semantic_state: SemanticState, concepts: List[str]) -> List[PreferenceRecord]:
        if semantic_state.intent != "preference":
            return []
        value = ", ".join(concepts[:5]) if concepts else semantic_state.compressed_context
        return [PreferenceRecord(key="user.preference", value=value, confidence=0.82, source="semantic_compressor_v2")]

    def _procedures(self, semantic_state: SemanticState, concepts: List[str]) -> List[ProcedureMemory]:
        lowered = semantic_state.compressed_context.lower()
        if not any(marker in lowered for marker in ["fix", "debug", "test", "error", "bug", "correccion", "corrige"]):
            return []
        title = f"Procedure from {semantic_state.intent}: {', '.join(concepts[:3])}"
        steps = [
            "Inspect the relevant project graph nodes.",
            "Reproduce the observed issue or requirement.",
            "Apply the smallest compatible change.",
            "Run focused validation before broad validation.",
        ]
        return [
            ProcedureMemory(
                procedure_id=str(uuid4()),
                title=title,
                steps=steps,
                domains=concepts[:5],
                confidence=0.68,
            )
        ]

