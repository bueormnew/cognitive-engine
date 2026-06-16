from __future__ import annotations

from typing import Any, Dict

from cognitive_engine.core.types import ConsolidationReport
from cognitive_engine.interfaces.base import Consolidator


class BackgroundConsolidationEngine(Consolidator):
    name = "background_consolidation_engine"

    def __init__(self, memory_system: Any, replay_buffer: Any) -> None:
        self.memory_system = memory_system
        self.replay_buffer = replay_buffer

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name}

    def run(self) -> ConsolidationReport:
        report = self.memory_system.consolidate()
        if hasattr(self.replay_buffer, "rescale"):
            report.replay_reweighted = self.replay_buffer.rescale()
        report.notes += " Replay priorities were decayed to maintain balanced rehearsal."
        return report

