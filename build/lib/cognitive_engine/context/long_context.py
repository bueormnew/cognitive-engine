from __future__ import annotations

from typing import Any, Dict, Iterable, List

from cognitive_engine.core.types import ContextPackage, MemoryBundleV2, SemanticStateV2


class LongContextManager:
    name = "long_context_manager"

    def __init__(self, default_budget: int = 32768) -> None:
        self.default_budget = default_budget

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "default_budget": self.default_budget}

    def compose(self, request: Dict[str, Any]) -> ContextPackage:
        state: SemanticStateV2 = request["semantic_state"]
        memory: MemoryBundleV2 = request["memory_bundle"]
        budget = int(request.get("context_budget", self.default_budget))
        specialist_context = request.get("specialist_context", [])
        sections: Dict[str, str] = {}
        sections["task"] = state.compressed_context
        sections["working_memory"] = str(memory.working_memory)
        sections["semantic_memory"] = "\n".join(record.summary for record in memory.semantic_long_term[:8])
        sections["episodic_memory"] = "\n".join(record.summary for record in memory.episodic[:5])
        sections["project_memory"] = "\n".join(
            f"{record.project_id}: files={record.files_indexed}, deps={', '.join(record.dependencies[:8])}"
            for record in memory.project
        )
        sections["procedural_memory"] = "\n".join(
            f"{procedure.title}: {'; '.join(procedure.steps[:3])}" for procedure in memory.procedural
        )
        if memory.graph_subgraph:
            nodes = ", ".join(f"{node.node_type}:{node.label}" for node in memory.graph_subgraph.nodes[:20])
            edges = ", ".join(f"{edge.relation}" for edge in memory.graph_subgraph.edges[:20])
            sections["graph_memory"] = f"nodes={nodes}\nedges={edges}\n{memory.graph_subgraph.rationale}"
        sections["preferences"] = "\n".join(f"{pref.key}: {pref.value}" for pref in memory.preferences)
        sections["specialists"] = "\n".join(str(item) for item in specialist_context)
        prompt = self._fit_budget(sections, budget)
        return ContextPackage(prompt=prompt, evidence_ids=[], token_estimate=self._estimate_tokens(prompt), sections=sections)

    def _fit_budget(self, sections: Dict[str, str], budget: int) -> str:
        parts = []
        remaining = budget
        for name, content in sections.items():
            if not content:
                continue
            text = f"[{name}]\n{content.strip()}"
            tokens = self._estimate_tokens(text)
            if tokens > remaining:
                approx_chars = max(0, remaining * 4)
                text = text[:approx_chars]
                tokens = self._estimate_tokens(text)
            if tokens <= remaining and text.strip():
                parts.append(text)
                remaining -= tokens
            if remaining <= 0:
                break
        return "\n\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

