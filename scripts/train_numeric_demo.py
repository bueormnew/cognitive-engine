from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from cognitive_engine import EngineBuilder
from cognitive_engine.utils.numeric import enumerate_numeric_dataset
from cognitive_engine.utils.visualization import dump_json, plot_training_metrics


def _extract_training_payload(response) -> dict:
    for trace in response.traces:
        if trace.stage == "online_training":
            return dict(trace.payload)
    return {}


def main() -> None:
    builder = EngineBuilder(config_path="configs/default.yaml")
    engine = builder.build()
    config = builder.config
    dataset = enumerate_numeric_dataset(
        max_operand=config.numeric_demo.max_operand,
        operations=config.numeric_demo.operations,
        seed=config.seed,
    )
    split_index = int(len(dataset) * 0.8)
    train_data = dataset[:split_index]
    val_data = dataset[split_index:]

    epoch_metrics = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state = deepcopy(engine.plastic_learner.state_dict())
    for epoch in range(1, config.numeric_demo.epochs + 1):
        running = {"loss": 0.0, "mae": 0.0, "accuracy": 0.0, "replay_ratio": 0.0, "plastic_norm": 0.0}
        updates = 0
        for sample in train_data:
            response = engine.process(
                {
                    "modality": "numeric",
                    "a": sample["a"],
                    "b": sample["b"],
                    "operation": sample["operation"],
                    "target": sample["target"],
                },
                allow_learning=True,
            )
            payload = _extract_training_payload(response)
            if payload:
                updates += 1
                for key in running:
                    running[key] += payload.get(key, 0.0)

        averaged = {key: value / max(updates, 1) for key, value in running.items()}
        validation = engine.trainer.evaluate(val_data)
        epoch_metrics.append(
            {
                "epoch": epoch,
                "loss": averaged["loss"],
                "mae": validation["mae"],
                "accuracy": validation["accuracy"],
                "replay_ratio": averaged["replay_ratio"],
                "plastic_norm": averaged["plastic_norm"],
                "stable_drift": 0.0,
                "semantic_memory_size": engine.snapshot()["memory"]["semantic_size"],
            }
        )
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch
            best_state = deepcopy(engine.plastic_learner.state_dict())
        print(
            f"epoch={epoch} loss={averaged['loss']:.4f} val_mae={validation['mae']:.4f} "
            f"val_acc={validation['accuracy']:.4f} drift={averaged['plastic_norm']:.4f}"
        )

    checkpoint_path = Path("artifacts") / "checkpoints" / "best_numeric_plastic_module.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    engine.plastic_learner.load_state_dict(best_state)

    evaluation_cases = [
        {"a": 2, "b": 3, "operation": "add", "target": 5},
        {"a": 9, "b": 4, "operation": "sub", "target": 5},
        {"a": 6, "b": 7, "operation": "mul", "target": 42},
        {"a": 12, "b": 12, "operation": "add", "target": 24},
    ]
    predictions = []
    for case in evaluation_cases:
        response = engine.process(
            {"modality": "numeric", "a": case["a"], "b": case["b"], "operation": case["operation"]},
            allow_learning=False,
        )
        predictions.append(
            {
                **case,
                "prediction": response.inference.prediction,
                "confidence": response.inference.confidence,
                "response": response.text,
            }
        )

    plots = plot_training_metrics(epoch_metrics, Path("artifacts") / "plots")
    payload = {
        "epoch_metrics": epoch_metrics,
        "validation_final": engine.trainer.evaluate(val_data),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "checkpoint_path": str(checkpoint_path),
        "prediction_cases": predictions,
        "final_snapshot": engine.snapshot(),
        "plots": plots,
        "train_size": len(train_data),
        "val_size": len(val_data),
    }
    output_path = dump_json(payload, Path("artifacts") / "reports" / "numeric_training_report.json")
    print(output_path)


if __name__ == "__main__":
    main()
