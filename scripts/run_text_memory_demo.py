from __future__ import annotations

from collections import Counter
from pathlib import Path

from cognitive_engine import EngineBuilder
from cognitive_engine.utils.visualization import dump_json, plot_stream_statistics


STREAM = [
    "Aprendi que Area2D detecta cuerpos dinamicos en Godot",
    "Prefiero respuestas tecnicas y directas",
    "Mi proyecto usa PyTorch para prototipos modulares",
    "hola",
    "En realidad la correccion importante es que el nodo correcto era RigidBody2D, no Area2D",
    "Qdrant sirve para memoria vectorial con busqueda semantica",
    "Esto es ruido ruido ruido ruido",
    "Me gusta conservar patrones utiles y descartar conversaciones vacias",
    "LoRA ayuda a adaptar submodulos sin reentrenar todo el sistema",
    "Las correcciones del usuario deben reforzarse mas que el small talk",
]

QUERIES = [
    "¿Que recuerdas sobre mis preferencias?",
    "¿Que recuerdas sobre el proyecto?",
]


def main() -> None:
    engine = EngineBuilder(config_path="configs/default.yaml").build()
    steps = []
    stats = {
        "importance": [],
        "confidence": [],
        "semantic_memory_size": [],
        "episodic_memory_size": [],
    }
    actions = Counter()

    for item in STREAM:
        response = engine.process(item, allow_learning=True)
        snapshot = engine.snapshot()
        steps.append(
            {
                "input": item,
                "response": response.text,
                "importance_action": response.importance.action,
                "importance_score": response.importance.importance_score,
                "confidence_score": response.importance.confidence_score,
                "learning_applied": response.learning_applied,
                "memory_snapshot": snapshot,
            }
        )
        stats["importance"].append(response.importance.importance_score)
        stats["confidence"].append(response.importance.confidence_score)
        stats["semantic_memory_size"].append(snapshot["memory"]["semantic_size"])
        stats["episodic_memory_size"].append(snapshot["memory"]["episodic_size"])
        actions[response.importance.action] += 1

    query_outputs = []
    for query in QUERIES:
        response = engine.process(query, allow_learning=False)
        query_outputs.append({"query": query, "response": response.text})

    plots = plot_stream_statistics(stats, Path("artifacts") / "plots")
    payload = {
        "stream_steps": steps,
        "query_outputs": query_outputs,
        "action_histogram": dict(actions),
        "final_snapshot": engine.snapshot(),
        "plots": plots,
    }
    output_path = dump_json(payload, Path("artifacts") / "reports" / "text_memory_demo.json")
    print(output_path)


if __name__ == "__main__":
    main()

