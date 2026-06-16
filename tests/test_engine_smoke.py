from cognitive_engine import EngineBuilder


def test_engine_processes_text_and_numeric():
    engine = EngineBuilder(config_path="configs/default.yaml").build()

    text_response = engine.process("Aprendi que LoRA permite adaptar un submodulo sin tocar todo el modelo")
    assert "Información" in text_response.text
    assert text_response.importance.action in {"learn", "reinforce", "consolidate"}

    numeric_response = engine.process({"modality": "numeric", "a": 1, "b": 2, "operation": "add", "target": 3})
    assert "Resultado estimado" in numeric_response.text
    assert engine.snapshot()["replay_size"] >= 1

