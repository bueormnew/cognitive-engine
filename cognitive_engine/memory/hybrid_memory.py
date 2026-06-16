from __future__ import annotations

from typing import Any, Dict, List

from cognitive_engine.core.types import (
    CompressedKnowledge,
    GraphQuery,
    MemoryBundleV2,
    MemoryQueryV2,
    PreferenceRecord,
    ProcedureMemory,
    ProjectMemoryRecord,
)
from cognitive_engine.memory.graph_memory import CognitiveGraphMemory


class HybridCognitiveMemory:
    name = "hybrid_cognitive_memory"

    def __init__(self, base_memory: Any, graph_memory: CognitiveGraphMemory) -> None:
        self.base_memory = base_memory
        self.graph_memory = graph_memory
        self.procedural_records: List[ProcedureMemory] = []
        self.project_records: Dict[str, ProjectMemoryRecord] = {}
        self.preferences: Dict[str, PreferenceRecord] = {}

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base": self.base_memory.describe(),
            "graph": self.graph_memory.describe(),
            "procedures": len(self.procedural_records),
            "projects": len(self.project_records),
            "preferences": len(self.preferences),
        }

    def update_working_memory(self, semantic_state: Any) -> Dict[str, Any]:
        return self.base_memory.update_working_memory(semantic_state)

    def retrieve(self, query: Any) -> Any:
        return self.base_memory.retrieve(query)

    def retrieve_v2(self, query: MemoryQueryV2) -> MemoryBundleV2:
        base = self.base_memory.retrieve(query)
        graph_query = query.graph_query or GraphQuery(seeds=query.concepts, project_id=query.project_id, top_k=query.top_k)
        subgraph = self.graph_memory.query_subgraph(graph_query) if query.include_graph else None
        project_records = []
        if query.include_project:
            if query.project_id and query.project_id in self.project_records:
                project_records.append(self.project_records[query.project_id])
            else:
                project_records.extend(list(self.project_records.values())[: query.top_k])
        procedures = self._search_procedures(query.concepts, query.top_k) if query.include_procedures else []
        preferences = list(self.preferences.values())[: query.top_k] if query.include_preferences else []
        return MemoryBundleV2(
            short_term=base.short_term,
            working_memory=base.working_memory,
            semantic_long_term=base.semantic_long_term,
            episodic=base.episodic,
            procedural=procedures,
            project=project_records,
            graph_subgraph=subgraph,
            preferences=preferences,
            evidence=[],
        )

    def write(self, knowledge: CompressedKnowledge) -> None:
        self.base_memory.write(knowledge)

    def write_v2(self, bundle: Any) -> None:
        for knowledge in getattr(bundle, "compressed_knowledge", []):
            self.base_memory.write(knowledge)
        if getattr(bundle, "graph_patch", None):
            self.graph_memory.apply_patch(bundle.graph_patch)
        for procedure in getattr(bundle, "procedures", []):
            self.add_procedure(procedure)
        for preference in getattr(bundle, "preferences", []):
            self.preferences[preference.key] = preference

    def add_project_record(self, record: ProjectMemoryRecord) -> None:
        self.project_records[record.project_id] = record

    def add_procedure(self, procedure: ProcedureMemory) -> None:
        existing = {item.procedure_id: item for item in self.procedural_records}
        existing[procedure.procedure_id] = procedure
        self.procedural_records = list(existing.values())

    def snapshot(self) -> Dict[str, Any]:
        base = self.base_memory.snapshot()
        return {
            **base,
            "graph": self.graph_memory.snapshot(),
            "procedural_size": len(self.procedural_records),
            "project_size": len(self.project_records),
            "preference_size": len(self.preferences),
        }

    def consolidate(self) -> Any:
        return self.base_memory.consolidate()

    def _search_procedures(self, concepts: List[str], top_k: int) -> List[ProcedureMemory]:
        roots = {concept.lower()[:4] for concept in concepts if len(concept) >= 4}
        scored = []
        for procedure in self.procedural_records:
            text = " ".join([procedure.title, *procedure.domains, *procedure.steps]).lower()
            score = sum(1 for root in roots if root in text)
            scored.append((score, procedure))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item for score, item in scored[:top_k] if score > 0]

