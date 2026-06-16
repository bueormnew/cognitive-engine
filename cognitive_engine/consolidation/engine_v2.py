from __future__ import annotations

from typing import Any, Dict

from cognitive_engine.core.types import ConsolidationReport
from cognitive_engine.interfaces.base import Consolidator


class CognitiveConsolidationEngineV2(Consolidator):
    name = "cognitive_consolidation_engine_v2"

    def __init__(self, memory_system: Any, replay_buffer: Any) -> None:
        self.memory_system = memory_system
        self.replay_buffer = replay_buffer
        self.cycles = 0

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "cycles": self.cycles}

    def run(self) -> ConsolidationReport:
        self.cycles += 1
        base_report = self.memory_system.consolidate()
        graph_snapshot = {}
        if hasattr(self.memory_system, "graph_memory"):
            graph_snapshot = self.memory_system.graph_memory.snapshot()
        replay_reweighted = 0
        if hasattr(self.replay_buffer, "rescale"):
            replay_reweighted = self.replay_buffer.rescale(0.985)
        notes = (
            "V2 light sleep consolidation completed: semantic merge, graph snapshot, replay decay, "
            f"graph_nodes={graph_snapshot.get('nodes', 0)}, graph_edges={graph_snapshot.get('edges', 0)}."
        )
        return ConsolidationReport(
            merged_records=base_report.merged_records,
            pruned_records=base_report.pruned_records,
            replay_reweighted=replay_reweighted,
            notes=notes,
        )

