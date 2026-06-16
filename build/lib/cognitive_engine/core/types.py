from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class ProcessedInput:
    raw_input: Any
    modality: str
    normalized_text: str = ""
    tokens: List[str] = field(default_factory=list)
    token_tensor: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    numeric_tensor: Optional[torch.Tensor] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticConcept:
    label: str
    weight: float
    source: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticState:
    raw_input: Any
    modality: str
    sequence_embedding: Optional[torch.Tensor]
    pooled_embedding: torch.Tensor
    intent: str
    entities: List[str]
    concepts: List[SemanticConcept]
    concept_graph_edges: List[Tuple[str, str, float]]
    compressed_context: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportanceAssessment:
    novelty: float
    utility: float
    frequency: float
    coherence: float
    correction_signal: float
    redundancy: float
    future_relevance: float
    contradiction_risk: float
    importance_score: float
    confidence_score: float
    learning_priority: float
    action: str
    rationale: str


@dataclass
class CompressedKnowledge:
    record_id: str
    source_type: str
    summary: str
    concepts: List[str]
    relations: List[Tuple[str, str, str]]
    embedding: np.ndarray
    importance: float
    confidence: float
    provenance: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reinforced_count: int = 1


@dataclass
class MemoryQuery:
    embedding: np.ndarray
    top_k: int = 5
    modality: Optional[str] = None
    intent: Optional[str] = None
    concepts: List[str] = field(default_factory=list)


@dataclass
class MemoryBundle:
    short_term: List[CompressedKnowledge] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    semantic_long_term: List[CompressedKnowledge] = field(default_factory=list)
    episodic: List[CompressedKnowledge] = field(default_factory=list)


@dataclass
class RoutingDecision:
    active_modules: List[str]
    consult_memories: List[str]
    update_memory: bool
    engage_plasticity: bool
    compute_budget: float
    rationale: str


@dataclass
class CoreInference:
    prediction: Any
    confidence: float
    explanation: str
    hidden_state: Optional[torch.Tensor] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplaySample:
    sample_id: str
    payload: Dict[str, Any]
    target: Any
    priority: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TraceEvent:
    stage: str
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EngineResponse:
    text: str
    semantic_state: SemanticState
    importance: ImportanceAssessment
    routing: RoutingDecision
    memory_bundle: MemoryBundle
    inference: CoreInference
    traces: List[TraceEvent]
    learning_applied: bool


@dataclass
class NumericBatch:
    features: torch.Tensor
    operation_ids: torch.Tensor
    targets: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    epoch: int
    loss: float
    mae: float
    accuracy: float
    replay_ratio: float
    plastic_norm: float
    stable_drift: float


@dataclass
class ConsolidationReport:
    merged_records: int
    pruned_records: int
    replay_reweighted: int
    notes: str


@dataclass
class KnowledgeTriple:
    subject: str
    relation: str
    object: str
    confidence: float = 0.7
    evidence: List[str] = field(default_factory=list)


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    label: str
    namespace: str = ""
    project_id: str = "global"
    confidence: float = 0.7
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None


@dataclass
class GraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    confidence: float = 0.7
    evidence_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphPatch:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    source: str = "unknown"


@dataclass
class GraphQuery:
    seeds: List[str] = field(default_factory=list)
    relation_filter: List[str] = field(default_factory=list)
    project_id: Optional[str] = None
    max_depth: int = 2
    top_k: int = 12


@dataclass
class GraphSubgraph:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    rationale: str = ""


@dataclass
class ProcedureMemory:
    procedure_id: str
    title: str
    steps: List[str]
    domains: List[str]
    confidence: float
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class ProjectMemoryRecord:
    project_id: str
    root_path: str
    files_indexed: int
    modules: List[str]
    dependencies: List[str]
    tests: List[str]
    commands: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferenceRecord:
    key: str
    value: str
    confidence: float
    source: str
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EvidenceRecord:
    evidence_id: str
    source: str
    quote: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticStateV2(SemanticState):
    code_symbols: List[str] = field(default_factory=list)
    graph_candidates: List[GraphPatch] = field(default_factory=list)
    uncertainty: float = 0.0
    modality_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryQueryV2(MemoryQuery):
    project_id: Optional[str] = None
    include_graph: bool = True
    include_procedures: bool = True
    include_project: bool = True
    include_preferences: bool = True
    graph_query: Optional[GraphQuery] = None


@dataclass
class MemoryBundleV2(MemoryBundle):
    procedural: List[ProcedureMemory] = field(default_factory=list)
    project: List[ProjectMemoryRecord] = field(default_factory=list)
    graph_subgraph: Optional[GraphSubgraph] = None
    preferences: List[PreferenceRecord] = field(default_factory=list)
    evidence: List[EvidenceRecord] = field(default_factory=list)


@dataclass
class RoutingDecisionV2(RoutingDecision):
    selected_specialists: List[str] = field(default_factory=list)
    memory_plan: Dict[str, float] = field(default_factory=dict)
    tool_plan: List[str] = field(default_factory=list)
    context_budget: int = 8192
    learning_action: str = "observe"
    consolidation_action: str = "none"
    confidence: float = 0.5
    gate_scores: Dict[str, float] = field(default_factory=dict)
    fallback_plan: str = "heuristic_router"


@dataclass
class CompressionBundle:
    compressed_knowledge: List[CompressedKnowledge] = field(default_factory=list)
    triples: List[KnowledgeTriple] = field(default_factory=list)
    graph_patch: GraphPatch = field(default_factory=GraphPatch)
    replay_samples: List[ReplaySample] = field(default_factory=list)
    validation_requests: List[str] = field(default_factory=list)
    preferences: List[PreferenceRecord] = field(default_factory=list)
    procedures: List[ProcedureMemory] = field(default_factory=list)


@dataclass
class ContextPackage:
    prompt: str
    evidence_ids: List[str]
    token_estimate: int
    sections: Dict[str, str] = field(default_factory=dict)


@dataclass
class StabilityDecision:
    approved: bool
    confidence: float
    risk_score: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class EngineResponseV2(EngineResponse):
    routing_v2: Optional[RoutingDecisionV2] = None
    memory_bundle_v2: Optional[MemoryBundleV2] = None
    context_package: Optional[ContextPackage] = None
    stability_decision: Optional[StabilityDecision] = None


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
