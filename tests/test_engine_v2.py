from cognitive_engine import EngineBuilder


def test_engine_v2_indexes_project_and_uses_graph_memory():
    engine = EngineBuilder(config_path="configs/default.yaml").build_v2()
    response = engine.process(
        "En Python este bug de pytest falla por import circular en cognitive_engine/core/engine.py",
        project_path=".",
    )

    snapshot = engine.snapshot()
    assert response.memory_bundle_v2 is not None
    assert response.routing_v2 is not None
    assert snapshot["memory"]["graph"]["nodes"] > 0
    assert snapshot["memory"]["graph"]["edges"] > 0
    assert snapshot["memory"]["project_size"] == 1
    assert "python" in response.text.lower()
    assert any(trace.stage == "context_v2" for trace in response.traces)


def test_engine_v2_learns_preference_without_breaking_v1_contracts():
    engine = EngineBuilder(config_path="configs/default.yaml").build_v2()
    response = engine.process("Prefiero codigo modular con interfaces claras y pruebas enfocadas")

    snapshot = engine.snapshot()
    assert response.learning_applied
    assert snapshot["memory"]["preference_size"] >= 1
    assert response.stability_decision is not None
    assert response.stability_decision.approved

