from cognitive_engine import EngineBuilder
from cognitive_engine.core.types import SemanticConcept, SemanticStateV2


def _state(text: str, intent: str = "statement") -> SemanticStateV2:
    import torch

    concepts = [SemanticConcept(label=item, weight=1.0, source="test") for item in text.lower().split()[:6]]
    return SemanticStateV2(
        raw_input=text,
        modality="text",
        sequence_embedding=None,
        pooled_embedding=torch.zeros(64),
        intent=intent,
        entities=[],
        concepts=concepts,
        concept_graph_edges=[],
        compressed_context=text,
        metadata={"token_count": len(text.split())},
        code_symbols=["pytest"] if "pytest" in text.lower() else [],
    )


def test_learned_router_can_be_trained_from_examples():
    engine = EngineBuilder(config_path="configs/default.yaml").build_v2()
    router = engine.router
    examples = [
        (_state("python pytest bug in project", "statement"), {"project_id": "repo", "has_project_memory": True}, [1, 1, 1, 1, 1, 1, 1, 1]),
        (_state("hola", "small_talk"), {}, [0, 0, 0, 0, 0, 0, 0, 0]),
        (_state("que recuerdas del proyecto", "question"), {"project_id": "repo", "has_project_memory": True}, [1, 1, 1, 1, 0, 0, 0, 0]),
    ]
    metrics = router.train_from_examples(examples, epochs=8)
    decision = router.route_v2(_state("python pytest bug in project"), {"project_id": "repo", "has_project_memory": True})

    assert metrics["examples"] == 3
    assert "python" in decision.selected_specialists or "general_coding" in decision.selected_specialists
    assert decision.gate_scores["graph_memory"] > 0.2

