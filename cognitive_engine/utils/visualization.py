from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def plot_training_metrics(metrics: List[Dict[str, float]], output_dir: str | Path) -> List[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in metrics]
    losses = [item["loss"] for item in metrics]
    maes = [item["mae"] for item in metrics]
    accuracies = [item["accuracy"] for item in metrics]
    drifts = [item["plastic_norm"] for item in metrics]

    plots: List[str] = []
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label="Loss")
    plt.plot(epochs, maes, label="MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training loss and MAE")
    plt.legend()
    loss_path = output_path / "training_loss_mae.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=180)
    plt.close()
    plots.append(str(loss_path))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, color="#1f77b4", label="Accuracy")
    plt.plot(epochs, drifts, color="#d62728", label="Plastic drift")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Accuracy and plastic drift")
    plt.legend()
    acc_path = output_path / "accuracy_drift.png"
    plt.tight_layout()
    plt.savefig(acc_path, dpi=180)
    plt.close()
    plots.append(str(acc_path))
    return plots


def plot_stream_statistics(stats: Dict[str, List[float]], output_dir: str | Path) -> List[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    steps = list(range(1, len(stats["importance"]) + 1))
    plots: List[str] = []

    plt.figure(figsize=(10, 6))
    plt.plot(steps, stats["importance"], label="Importance")
    plt.plot(steps, stats["confidence"], label="Confidence")
    plt.xlabel("Stream step")
    plt.ylabel("Score")
    plt.title("Importance and confidence through the stream")
    plt.legend()
    importance_path = output_path / "importance_confidence_stream.png"
    plt.tight_layout()
    plt.savefig(importance_path, dpi=180)
    plt.close()
    plots.append(str(importance_path))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, stats["semantic_memory_size"], label="Semantic memory size")
    plt.plot(steps, stats["episodic_memory_size"], label="Episodic memory size")
    plt.xlabel("Stream step")
    plt.ylabel("Records")
    plt.title("Memory growth over the learning stream")
    plt.legend()
    memory_path = output_path / "memory_growth_stream.png"
    plt.tight_layout()
    plt.savefig(memory_path, dpi=180)
    plt.close()
    plots.append(str(memory_path))
    return plots


def dump_json(payload: Dict[str, object], output_file: str | Path) -> str:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)

