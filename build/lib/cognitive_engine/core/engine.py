from __future__ import annotations

from typing import Any, Dict, List, Sequence

from cognitive_engine.config.schema import EngineConfig
from cognitive_engine.core.types import EngineResponse, MemoryQuery, TraceEvent, tensor_to_numpy


class CognitiveEngine:
    def __init__(
        self,
        config: EngineConfig,
        processors: Sequence[Any],
        semantic_encoder: Any,
        importance_evaluator: Any,
        compressor: Any,
        memory_system: Any,
        router: Any,
        stable_core: Any,
        plastic_learner: Any,
        trainer: Any,
        replay_buffer: Any,
        consolidator: Any,
        telemetry: Any,
    ) -> None:
        self.config = config
        self.processors = list(processors)
        self.semantic_encoder = semantic_encoder
        self.importance_evaluator = importance_evaluator
        self.compressor = compressor
        self.memory_system = memory_system
        self.router = router
        self.stable_core = stable_core
        self.plastic_learner = plastic_learner
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.consolidator = consolidator
        self.telemetry = telemetry
        self.step_counter = 0

    def process(self, payload: Any, allow_learning: bool = True) -> EngineResponse:
        traces: List[TraceEvent] = []
        processed = self._select_processor(payload).process(payload)
        traces.append(self._trace("input", "Payload processed.", {"modality": processed.modality, "metadata": processed.metadata}))

        semantic_state = self.semantic_encoder.encode(processed)
        self.memory_system.update_working_memory(semantic_state)
        traces.append(self._trace("semantic", "Semantic state created.", {"intent": semantic_state.intent, "concepts": [c.label for c in semantic_state.concepts]}))

        route = self.router.route(processed, semantic_state)
        query = MemoryQuery(
            embedding=tensor_to_numpy(semantic_state.pooled_embedding.float()),
            top_k=self.config.memory.retrieval_top_k,
            modality=processed.modality,
            intent=semantic_state.intent,
            concepts=[concept.label for concept in semantic_state.concepts],
        )
        memory_bundle = self.memory_system.retrieve(query)
        traces.append(
            self._trace(
                "memory_retrieval",
                "Memory bundle retrieved.",
                {
                    "short_term": len(memory_bundle.short_term),
                    "semantic_long_term": len(memory_bundle.semantic_long_term),
                    "episodic": len(memory_bundle.episodic),
                },
            )
        )

        importance = self.importance_evaluator.evaluate(semantic_state, memory_bundle)
        semantic_state.metadata["importance_action"] = importance.action
        traces.append(
            self._trace(
                "importance",
                importance.rationale,
                {
                    "importance_score": importance.importance_score,
                    "confidence_score": importance.confidence_score,
                    "learning_priority": importance.learning_priority,
                    "action": importance.action,
                },
            )
        )

        plastic_output = None
        if route.engage_plasticity and processed.modality == "numeric":
            plastic_output = self.plastic_learner.predict(semantic_state)
            traces.append(self._trace("plastic_inference", plastic_output.explanation, plastic_output.artifacts))

        inference = self.stable_core.infer(semantic_state, memory_bundle, route, plastic_output)
        traces.append(self._trace("stable_core", inference.explanation, {"confidence": inference.confidence}))

        learning_applied = False
        if allow_learning and route.update_memory and importance.action not in {"ignore", "uncertain"}:
            knowledge = self.compressor.compress(semantic_state, importance)
            self.memory_system.write(knowledge)
            learning_applied = True
            traces.append(self._trace("memory_write", "Compressed knowledge stored.", {"record_id": knowledge.record_id, "summary": knowledge.summary}))

        if allow_learning and processed.modality == "numeric" and semantic_state.metadata.get("target") is not None:
            training_metrics = self.trainer.observe(semantic_state, priority=importance.learning_priority)
            if training_metrics is not None:
                learning_applied = True
                traces.append(self._trace("online_training", "Plastic module updated.", training_metrics))

        self.step_counter += 1
        if self.step_counter % self.config.memory.consolidation_interval == 0:
            report = self.consolidator.run()
            traces.append(
                self._trace(
                    "consolidation",
                    report.notes,
                    {
                        "merged_records": report.merged_records,
                        "pruned_records": report.pruned_records,
                        "replay_reweighted": report.replay_reweighted,
                    },
                )
            )

        for event in traces:
            self.telemetry.emit(event)
        self.telemetry.flush()

        return EngineResponse(
            text=str(inference.explanation),
            semantic_state=semantic_state,
            importance=importance,
            routing=route,
            memory_bundle=memory_bundle,
            inference=inference,
            traces=traces,
            learning_applied=learning_applied,
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "steps": self.step_counter,
            "memory": self.memory_system.snapshot(),
            "replay_size": len(self.replay_buffer),
            "plastic_drift": self.plastic_learner.parameter_drift(),
        }

    def describe_architecture(self) -> Dict[str, Any]:
        return {
            "processors": [processor.describe() for processor in self.processors],
            "semantic_encoder": self.semantic_encoder.describe(),
            "importance_evaluator": self.importance_evaluator.describe(),
            "compressor": self.compressor.describe(),
            "memory_system": self.memory_system.describe(),
            "router": self.router.describe(),
            "stable_core": self.stable_core.describe(),
            "plastic_learner": self.plastic_learner.describe(),
            "replay_buffer": self.replay_buffer.describe(),
            "consolidator": self.consolidator.describe(),
            "telemetry": self.telemetry.describe(),
        }

    def replace_component(self, slot: str, component: Any) -> Any:
        if not hasattr(self, slot):
            raise AttributeError(f"Engine has no component slot named '{slot}'.")
        previous = getattr(self, slot)
        setattr(self, slot, component)
        return previous

    def _select_processor(self, payload: Any) -> Any:
        for processor in self.processors:
            if processor.supports(payload):
                return processor
        raise TypeError(f"No input processor supports payload type: {type(payload)!r}")

    def _trace(self, stage: str, message: str, payload: Dict[str, Any]) -> TraceEvent:
        return TraceEvent(stage=stage, message=message, payload=payload)
