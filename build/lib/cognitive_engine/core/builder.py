from __future__ import annotations

from pathlib import Path

from cognitive_engine.adapters.plastic_numeric_adapter import PlasticArithmeticModule
from cognitive_engine.compression.knowledge_compressor import SemanticKnowledgeCompressor
from cognitive_engine.compression.semantic_compressor_v2 import SemanticCompressorV2
from cognitive_engine.config.loader import load_engine_config
from cognitive_engine.config.schema import EngineConfig
from cognitive_engine.consolidation.engine import BackgroundConsolidationEngine
from cognitive_engine.consolidation.engine_v2 import CognitiveConsolidationEngineV2
from cognitive_engine.core.engine import CognitiveEngine
from cognitive_engine.core.engine_v2 import CognitiveEngineV2
from cognitive_engine.core.registry import GLOBAL_REGISTRY
from cognitive_engine.context.long_context import LongContextManager
from cognitive_engine.memory.graph_memory import CognitiveGraphMemory
from cognitive_engine.memory.hybrid_memory import HybridCognitiveMemory
from cognitive_engine.memory.project_memory import ProjectIndexer
from cognitive_engine.memory.stores import HierarchicalMemorySystem
from cognitive_engine.models.stable_core import StableReasoningCore
from cognitive_engine.modules.importance_evaluator import AdaptiveImportanceEvaluator
from cognitive_engine.modules.input_processing import NumericInputProcessor, TextInputProcessor
from cognitive_engine.modules.semantic_understanding import HybridSemanticEncoder, NumericSemanticEncoder, TextSemanticEncoder
from cognitive_engine.replay.buffer import PrioritizedReplayBuffer
from cognitive_engine.routing.dynamic_router import AdaptiveDynamicRouter
from cognitive_engine.routing.learned_router import LearnedCognitiveRouter
from cognitive_engine.specialists.runtime import SpecialistRuntime
from cognitive_engine.stability.governor import StabilityGovernorV2
from cognitive_engine.training.online_trainer import OnlineTrainer
from cognitive_engine.utils.seeding import set_seed
from cognitive_engine.utils.telemetry import JsonlTelemetrySink


class EngineBuilder:
    def __init__(self, config: EngineConfig | None = None, config_path: str | None = None) -> None:
        self.config = config or load_engine_config(config_path)
        self._register_defaults()

    def build(self) -> CognitiveEngine:
        set_seed(self.config.seed)
        telemetry = GLOBAL_REGISTRY.create("telemetry", "jsonl", path=Path("artifacts") / "logs" / "engine_trace.jsonl")
        replay_buffer = GLOBAL_REGISTRY.create("replay", "prioritized", capacity=self.config.memory.replay_capacity)
        memory_system = GLOBAL_REGISTRY.create("memory", "hierarchical", config=self.config.memory)
        processors = [
            GLOBAL_REGISTRY.create("processor", "numeric", device=self.config.device, scale=float(self.config.numeric_demo.max_operand)),
            GLOBAL_REGISTRY.create("processor", "text", device=self.config.device),
        ]
        semantic_encoder = GLOBAL_REGISTRY.create(
            "semantic_encoder",
            "hybrid",
            text_encoder=GLOBAL_REGISTRY.create("semantic_encoder", "text", device=self.config.device),
            numeric_encoder=GLOBAL_REGISTRY.create("semantic_encoder", "numeric", device=self.config.device),
        )
        plastic_learner = GLOBAL_REGISTRY.create(
            "plasticity",
            "arithmetic",
            device=self.config.device,
            output_scale=float(self.config.numeric_demo.max_operand**2),
        )
        trainer = OnlineTrainer(
            plastic_learner=plastic_learner,
            replay_buffer=replay_buffer,
            replay_batch_size=self.config.numeric_demo.replay_batch_size,
            device=self.config.device,
        )
        consolidator = BackgroundConsolidationEngine(memory_system=memory_system, replay_buffer=replay_buffer)
        return CognitiveEngine(
            config=self.config,
            processors=processors,
            semantic_encoder=semantic_encoder,
            importance_evaluator=GLOBAL_REGISTRY.create("importance", "adaptive", thresholds=self.config.thresholds),
            compressor=GLOBAL_REGISTRY.create("compressor", "semantic"),
            memory_system=memory_system,
            router=GLOBAL_REGISTRY.create("router", "adaptive"),
            stable_core=GLOBAL_REGISTRY.create("stable_core", "reasoning", device=self.config.device),
            plastic_learner=plastic_learner,
            trainer=trainer,
            replay_buffer=replay_buffer,
            consolidator=consolidator,
            telemetry=telemetry,
        )

    def build_v2(self) -> CognitiveEngineV2:
        set_seed(self.config.seed)
        telemetry = GLOBAL_REGISTRY.create("telemetry", "jsonl", path=Path("artifacts") / "logs" / "engine_trace_v2.jsonl")
        replay_buffer = GLOBAL_REGISTRY.create("replay", "prioritized", capacity=self.config.memory.replay_capacity)
        base_memory = GLOBAL_REGISTRY.create("memory", "hierarchical", config=self.config.memory)
        graph_memory = GLOBAL_REGISTRY.create("memory", "graph")
        memory_system = GLOBAL_REGISTRY.create("memory", "hybrid_v2", base_memory=base_memory, graph_memory=graph_memory)
        processors = [
            GLOBAL_REGISTRY.create("processor", "numeric", device=self.config.device, scale=float(self.config.numeric_demo.max_operand)),
            GLOBAL_REGISTRY.create("processor", "text", device=self.config.device),
        ]
        semantic_encoder = GLOBAL_REGISTRY.create(
            "semantic_encoder",
            "hybrid",
            text_encoder=GLOBAL_REGISTRY.create("semantic_encoder", "text", device=self.config.device),
            numeric_encoder=GLOBAL_REGISTRY.create("semantic_encoder", "numeric", device=self.config.device),
        )
        plastic_learner = GLOBAL_REGISTRY.create(
            "plasticity",
            "arithmetic",
            device=self.config.device,
            output_scale=float(self.config.numeric_demo.max_operand**2),
        )
        trainer = OnlineTrainer(
            plastic_learner=plastic_learner,
            replay_buffer=replay_buffer,
            replay_batch_size=self.config.numeric_demo.replay_batch_size,
            device=self.config.device,
        )
        return CognitiveEngineV2(
            config=self.config,
            processors=processors,
            semantic_encoder=semantic_encoder,
            importance_evaluator=GLOBAL_REGISTRY.create("importance", "adaptive", thresholds=self.config.thresholds),
            compressor_v2=GLOBAL_REGISTRY.create("compressor", "semantic_v2"),
            memory_system=memory_system,
            router=GLOBAL_REGISTRY.create("router", "learned", device=self.config.device),
            stable_core=GLOBAL_REGISTRY.create("stable_core", "reasoning", device=self.config.device),
            plastic_learner=plastic_learner,
            trainer=trainer,
            replay_buffer=replay_buffer,
            consolidator=GLOBAL_REGISTRY.create("consolidator", "cognitive_v2", memory_system=memory_system, replay_buffer=replay_buffer),
            telemetry=telemetry,
            specialist_runtime=GLOBAL_REGISTRY.create("specialists", "runtime"),
            context_manager=GLOBAL_REGISTRY.create("context", "long"),
            stability_governor=GLOBAL_REGISTRY.create("stability", "governor_v2"),
            project_indexer=GLOBAL_REGISTRY.create("project", "indexer", graph_memory=graph_memory),
        )

    def _register_defaults(self) -> None:
        GLOBAL_REGISTRY.register("telemetry", "jsonl", JsonlTelemetrySink)
        GLOBAL_REGISTRY.register("replay", "prioritized", PrioritizedReplayBuffer)
        GLOBAL_REGISTRY.register("memory", "hierarchical", HierarchicalMemorySystem)
        GLOBAL_REGISTRY.register("memory", "graph", CognitiveGraphMemory)
        GLOBAL_REGISTRY.register("memory", "hybrid_v2", HybridCognitiveMemory)
        GLOBAL_REGISTRY.register("processor", "numeric", NumericInputProcessor)
        GLOBAL_REGISTRY.register("processor", "text", TextInputProcessor)
        GLOBAL_REGISTRY.register("semantic_encoder", "text", TextSemanticEncoder)
        GLOBAL_REGISTRY.register("semantic_encoder", "numeric", NumericSemanticEncoder)
        GLOBAL_REGISTRY.register("semantic_encoder", "hybrid", HybridSemanticEncoder)
        GLOBAL_REGISTRY.register("importance", "adaptive", AdaptiveImportanceEvaluator)
        GLOBAL_REGISTRY.register("compressor", "semantic", SemanticKnowledgeCompressor)
        GLOBAL_REGISTRY.register("compressor", "semantic_v2", SemanticCompressorV2)
        GLOBAL_REGISTRY.register("router", "adaptive", AdaptiveDynamicRouter)
        GLOBAL_REGISTRY.register("router", "learned", LearnedCognitiveRouter)
        GLOBAL_REGISTRY.register("stable_core", "reasoning", StableReasoningCore)
        GLOBAL_REGISTRY.register("plasticity", "arithmetic", PlasticArithmeticModule)
        GLOBAL_REGISTRY.register("consolidator", "cognitive_v2", CognitiveConsolidationEngineV2)
        GLOBAL_REGISTRY.register("specialists", "runtime", SpecialistRuntime)
        GLOBAL_REGISTRY.register("context", "long", LongContextManager)
        GLOBAL_REGISTRY.register("stability", "governor_v2", StabilityGovernorV2)
        GLOBAL_REGISTRY.register("project", "indexer", ProjectIndexer)
