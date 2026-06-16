from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from cognitive_engine.core.types import (
    CompressedKnowledge,
    ConsolidationReport,
    CoreInference,
    ImportanceAssessment,
    MemoryBundle,
    MemoryQuery,
    NumericBatch,
    ProcessedInput,
    ReplaySample,
    RoutingDecision,
    RoutingDecisionV2,
    SemanticState,
    SemanticStateV2,
    TraceEvent,
)


class ConfigurableComponent(ABC):
    name: str

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError


class InputProcessor(ConfigurableComponent, ABC):
    @abstractmethod
    def supports(self, payload: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def process(self, payload: Any) -> ProcessedInput:
        raise NotImplementedError


class SemanticEncoder(ConfigurableComponent, ABC):
    @abstractmethod
    def encode(self, processed: ProcessedInput) -> SemanticState:
        raise NotImplementedError


class ImportanceEvaluator(ConfigurableComponent, ABC):
    @abstractmethod
    def evaluate(self, semantic_state: SemanticState, memory_bundle: MemoryBundle) -> ImportanceAssessment:
        raise NotImplementedError


class KnowledgeCompressor(ConfigurableComponent, ABC):
    @abstractmethod
    def compress(self, semantic_state: SemanticState, importance: ImportanceAssessment) -> CompressedKnowledge:
        raise NotImplementedError


class MemoryStore(ConfigurableComponent, ABC):
    @abstractmethod
    def write(self, knowledge: CompressedKnowledge) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: MemoryQuery) -> List[CompressedKnowledge]:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> List[CompressedKnowledge]:
        raise NotImplementedError


class WorkingMemory(ConfigurableComponent, ABC):
    @abstractmethod
    def update(self, semantic_state: SemanticState) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError


class Router(ConfigurableComponent, ABC):
    @abstractmethod
    def route(self, processed: ProcessedInput, semantic_state: Optional[SemanticState] = None) -> RoutingDecision:
        raise NotImplementedError


class StableCore(ConfigurableComponent, ABC):
    @abstractmethod
    def infer(
        self,
        semantic_state: SemanticState,
        memory_bundle: MemoryBundle,
        routing: RoutingDecision,
        plastic_output: Optional[CoreInference] = None,
    ) -> CoreInference:
        raise NotImplementedError


class PlasticLearner(ConfigurableComponent, ABC):
    @abstractmethod
    def predict(self, semantic_state: SemanticState) -> CoreInference:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: NumericBatch, replay_batch: Optional[NumericBatch] = None) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def parameter_drift(self) -> float:
        raise NotImplementedError


class ReplayBuffer(ConfigurableComponent, ABC):
    @abstractmethod
    def add(self, sample: ReplaySample) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int) -> List[ReplaySample]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class Consolidator(ConfigurableComponent, ABC):
    @abstractmethod
    def run(self) -> ConsolidationReport:
        raise NotImplementedError


class TelemetrySink(ConfigurableComponent, ABC):
    @abstractmethod
    def emit(self, event: TraceEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        raise NotImplementedError


class LearnedRouter(Router, ABC):
    @abstractmethod
    def route_v2(self, semantic_state: SemanticStateV2, runtime_context: Dict[str, Any]) -> RoutingDecisionV2:
        raise NotImplementedError


class GraphMemoryStore(ConfigurableComponent, ABC):
    @abstractmethod
    def apply_patch(self, patch: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def query_subgraph(self, query: Any) -> Any:
        raise NotImplementedError


class Specialist(ConfigurableComponent, ABC):
    @abstractmethod
    def can_handle(self, semantic_state: SemanticStateV2) -> float:
        raise NotImplementedError

    @abstractmethod
    def prepare_context(self, semantic_state: SemanticStateV2, memory_bundle: Any) -> Dict[str, Any]:
        raise NotImplementedError


class LongContextManager(ConfigurableComponent, ABC):
    @abstractmethod
    def compose(self, request: Dict[str, Any]) -> Any:
        raise NotImplementedError


class StabilityGovernor(ConfigurableComponent, ABC):
    @abstractmethod
    def validate_update(self, update: Dict[str, Any]) -> Any:
        raise NotImplementedError
