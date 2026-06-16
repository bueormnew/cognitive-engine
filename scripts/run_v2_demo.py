from __future__ import annotations

from pathlib import Path

from cognitive_engine import EngineBuilder
from cognitive_engine.utils.visualization import dump_json


STREAM = [
    "Prefiero codigo modular con interfaces claras y pruebas enfocadas",
    "En Python este bug de pytest falla por import circular en cognitive_engine/core/engine.py",
    "Godot Area2D detecta cuerpos, pero para fisica dinamica debo revisar RigidBody2D",
    "Necesito recordar que este proyecto usa registry y dependency injection",
    "hola hola ruido sin valor",
    "Cuando un test falla, primero reproducirlo y luego aplicar el cambio minimo",
]


def main() -> None:
    engine = EngineBuilder(config_path="configs/default.yaml").build_v2()
    outputs = []
    project_indexed = False
    for item in STREAM:
        response = engine.process(item, project_path="." if not project_indexed else None)
        project_indexed = True
        outputs.append(
            {
                "input": item,
                "text": response.text,
                "learning_applied": response.learning_applied,
                "routing": response.routing_v2.gate_scores if response.routing_v2 else {},
                "selected_specialists": response.routing_v2.selected_specialists if response.routing_v2 else [],
                "context_tokens": response.context_package.token_estimate if response.context_package else 0,
                "stability": response.stability_decision.__dict__ if response.stability_decision else {},
                "traces": [trace.stage for trace in response.traces],
            }
        )

    query = engine.process("Que recuerdas sobre los bugs de Python y las decisiones del proyecto?", allow_learning=False)
    snapshot = engine.snapshot()
    report = {
        "outputs": outputs,
        "query": {
            "text": query.text,
            "routing": query.routing_v2.gate_scores if query.routing_v2 else {},
            "graph_nodes_returned": len(query.memory_bundle_v2.graph_subgraph.nodes) if query.memory_bundle_v2 and query.memory_bundle_v2.graph_subgraph else 0,
        },
        "snapshot": snapshot,
        "architecture": engine.describe_architecture(),
    }
    output_path = dump_json(report, Path("artifacts") / "reports" / "v2_demo_report.json")
    print(output_path)


if __name__ == "__main__":
    main()

