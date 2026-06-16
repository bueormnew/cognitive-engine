from __future__ import annotations

from typing import Any, Dict, List, Sequence

from cognitive_engine.config.schema import EngineConfig
from cognitive_engine.core.types import (
    EngineResponseV2,
    GraphQuery,
    MemoryQueryV2,
    SemanticState,
    SemanticStateV2,
    TraceEvent,
    tensor_to_numpy,
)


class CognitiveEngineV2:
    def __init__(
        self,
        config: EngineConfig,
        processors: Sequence[Any],
        semantic_encoder: Any,
        importance_evaluator: Any,
        compressor_v2: Any,
        memory_system: Any,
        router: Any,
        stable_core: Any,
        plastic_learner: Any,
        trainer: Any,
        replay_buffer: Any,
        consolidator: Any,
        telemetry: Any,
        specialist_runtime: Any,
        context_manager: Any,
        stability_governor: Any,
        project_indexer: Any,
    ) -> None:
        self.config = config
        self.processors = list(processors)
        self.semantic_encoder = semantic_encoder
        self.importance_evaluator = importance_evaluator
        self.compressor_v2 = compressor_v2
        self.memory_system = memory_system
        self.router = router
        self.stable_core = stable_core
        self.plastic_learner = plastic_learner
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.consolidator = consolidator
        self.telemetry = telemetry
        self.specialist_runtime = specialist_runtime
        self.context_manager = context_manager
        self.stability_governor = stability_governor
        self.project_indexer = project_indexer
        self.step_counter = 0

    def process(self, payload: Any, allow_learning: bool = True, project_path: str | None = None, project_id: str | None = None) -> EngineResponseV2:
        traces: List[TraceEvent] = []
        if project_path:
            project_record = self.project_indexer.index_project(project_path, project_id)
            self.memory_system.add_project_record(project_record)
            project_id = project_record.project_id
            traces.append(self._trace("project_index", "Project indexed into graph memory.", project_record.__dict__))

        processed = self._select_processor(payload).process(payload)
        traces.append(self._trace("input_v2", "Payload processed.", {"modality": processed.modality, "metadata": processed.metadata}))

        base_state = self.semantic_encoder.encode(processed)
        semantic_state = self._upgrade_state(base_state)
        self.memory_system.update_working_memory(semantic_state)
        traces.append(
            self._trace(
                "semantic_v2",
                "SemanticStateV2 created.",
                {
                    "intent": semantic_state.intent,
                    "concepts": [concept.label for concept in semantic_state.concepts],
                    "code_symbols": semantic_state.code_symbols,
                },
            )
        )

        runtime_context = {
            "project_id": project_id,
            "has_project_memory": bool(self.memory_system.project_records),
            "long_context": len(str(payload)) > 4000 or bool(project_path),
        }
        routing = self.router.route_v2(semantic_state, runtime_context)
        traces.append(self._trace("routing_v2", routing.rationale, routing.gate_scores))

        query = MemoryQueryV2(
            embedding=tensor_to_numpy(semantic_state.pooled_embedding.float()),
            top_k=self.config.memory.retrieval_top_k,
            modality=processed.modality,
            intent=semantic_state.intent,
            concepts=[concept.label for concept in semantic_state.concepts] + semantic_state.code_symbols,
            project_id=project_id,
            graph_query=GraphQuery(
                seeds=[concept.label for concept in semantic_state.concepts] + semantic_state.code_symbols,
                project_id=project_id,
                top_k=16,
            ),
        )
        memory_bundle = self.memory_system.retrieve_v2(query)
        traces.append(
            self._trace(
                "memory_v2",
                "Hybrid memory bundle retrieved.",
                {
                    "semantic": len(memory_bundle.semantic_long_term),
                    "episodic": len(memory_bundle.episodic),
                    "procedural": len(memory_bundle.procedural),
                    "project": len(memory_bundle.project),
                    "graph_nodes": len(memory_bundle.graph_subgraph.nodes) if memory_bundle.graph_subgraph else 0,
                },
            )
        )

        specialist_context = self.specialist_runtime.prepare_context(semantic_state, memory_bundle, routing.selected_specialists)
        context_package = self.context_manager.compose(
            {
                "semantic_state": semantic_state,
                "memory_bundle": memory_bundle,
                "specialist_context": specialist_context,
                "context_budget": routing.context_budget,
            }
        )
        traces.append(
            self._trace(
                "context_v2",
                "Context package composed.",
                {"token_estimate": context_package.token_estimate, "sections": list(context_package.sections)},
            )
        )

        importance = self.importance_evaluator.evaluate(semantic_state, memory_bundle)
        semantic_state.metadata["importance_action"] = importance.action
        stability = self.stability_governor.validate_update(
            {
                "confidence": importance.confidence_score,
                "contradiction_risk": importance.contradiction_risk,
                "plastic_drift": self.plastic_learner.parameter_drift(),
            }
        )
        traces.append(self._trace("stability_v2", "Stability decision computed.", stability.__dict__))

        plastic_output = None
        if routing.engage_plasticity and processed.modality == "numeric":
            plastic_output = self.plastic_learner.predict(semantic_state)
            traces.append(self._trace("plastic_v2", plastic_output.explanation, plastic_output.artifacts))

        inference = self.stable_core.infer(semantic_state, memory_bundle, routing, plastic_output)
        if context_package.sections.get("specialists"):
            inference.explanation = f"{inference.explanation}\n\nSpecialist context: {context_package.sections['specialists']}"
        traces.append(self._trace("stable_core_v2", inference.explanation, {"confidence": inference.confidence}))

        learning_applied = False
        if (
            allow_learning
            and routing.update_memory
            and routing.learning_action == "learn"
            and importance.action not in {"ignore", "uncertain"}
            and stability.approved
        ):
            compression = self.compressor_v2.compress_v2(semantic_state, importance, project_id=project_id or "global")
            self.memory_system.write_v2(compression)
            learning_applied = True
            traces.append(
                self._trace(
                    "memory_write_v2",
                    "CompressionBundle written to hybrid memory.",
                    {
                        "knowledge": len(compression.compressed_knowledge),
                        "triples": len(compression.triples),
                        "graph_nodes": len(compression.graph_patch.nodes),
                        "procedures": len(compression.procedures),
                        "preferences": len(compression.preferences),
                    },
                )
            )

        if allow_learning and processed.modality == "numeric" and semantic_state.metadata.get("target") is not None:
            training_metrics = self.trainer.observe(semantic_state, priority=importance.learning_priority)
            if training_metrics is not None:
                learning_applied = True
                traces.append(self._trace("online_training_v2", "Plastic module updated.", training_metrics))

        self.step_counter += 1
        if self.step_counter % self.config.memory.consolidation_interval == 0 or routing.consolidation_action != "none":
            report = self.consolidator.run()
            traces.append(self._trace("consolidation_v2", report.notes, report.__dict__))

        for event in traces:
            self.telemetry.emit(event)
        self.telemetry.flush()

        return EngineResponseV2(
            text=str(inference.explanation),
            semantic_state=semantic_state,
            importance=importance,
            routing=routing,
            memory_bundle=memory_bundle,
            inference=inference,
            traces=traces,
            learning_applied=learning_applied,
            routing_v2=routing,
            memory_bundle_v2=memory_bundle,
            context_package=context_package,
            stability_decision=stability,
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": "v2",
            "steps": self.step_counter,
            "memory": self.memory_system.snapshot(),
            "router": self.router.describe(),
            "specialists": self.specialist_runtime.describe(),
            "context_manager": self.context_manager.describe(),
            "stability": self.stability_governor.describe(),
            "replay_size": len(self.replay_buffer),
            "plastic_drift": self.plastic_learner.parameter_drift(),
        }

    def describe_architecture(self) -> Dict[str, Any]:
        return {
            "version": "v2",
            "processors": [processor.describe() for processor in self.processors],
            "semantic_encoder": self.semantic_encoder.describe(),
            "router": self.router.describe(),
            "memory_system": self.memory_system.describe(),
            "compressor": self.compressor_v2.describe(),
            "specialists": self.specialist_runtime.describe(),
            "context_manager": self.context_manager.describe(),
            "stability_governor": self.stability_governor.describe(),
            "stable_core": self.stable_core.describe(),
            "plastic_learner": self.plastic_learner.describe(),
            "consolidator": self.consolidator.describe(),
        }

    def _upgrade_state(self, state: SemanticState) -> SemanticStateV2:
        text = str(state.raw_input)
        code_symbols = self._extract_code_symbols(text)
        return SemanticStateV2(
            raw_input=state.raw_input,
            modality=state.modality,
            sequence_embedding=state.sequence_embedding,
            pooled_embedding=state.pooled_embedding,
            intent=state.intent,
            entities=state.entities,
            concepts=state.concepts,
            concept_graph_edges=state.concept_graph_edges,
            compressed_context=state.compressed_context,
            metadata=dict(state.metadata),
            code_symbols=code_symbols,
            graph_candidates=[],
            uncertainty=max(0.0, 1.0 - min(1.0, len(state.concepts) / 6.0)),
            modality_features={"contains_code": bool(code_symbols), "raw_length": len(text)},
        )

    def _extract_code_symbols(self, text: str) -> List[str]:
        symbols = []
        for raw in text.replace("(", " ").replace(")", " ").replace(":", " ").replace(".", " ").split():
            token = raw.strip("`'\"")
            if not token:
                continue
            if token.endswith(".py") or token in {"class", "def", "pytest", "import"}:
                symbols.append(token)
            elif "_" in token or (token[:1].isupper() and len(token) > 2):
                symbols.append(token)
        return list(dict.fromkeys(symbols))[:12]

    def _select_processor(self, payload: Any) -> Any:
        for processor in self.processors:
            if processor.supports(payload):
                return processor
        raise TypeError(f"No input processor supports payload type: {type(payload)!r}")

    def _trace(self, stage: str, message: str, payload: Dict[str, Any]) -> TraceEvent:
        return TraceEvent(stage=stage, message=message, payload=payload)
