from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import torch

from cognitive_engine.core.types import NumericBatch, ReplaySample, SemanticState, TrainingMetrics


class OnlineTrainer:
    def __init__(self, plastic_learner: Any, replay_buffer: Any, replay_batch_size: int = 32, device: str = "cpu") -> None:
        self.plastic_learner = plastic_learner
        self.replay_buffer = replay_buffer
        self.replay_batch_size = replay_batch_size
        self.device = device

    def observe(self, semantic_state: SemanticState, priority: float) -> Optional[Dict[str, float]]:
        target = semantic_state.metadata.get("target")
        if target is None:
            return None

        sample = ReplaySample(
            sample_id=str(uuid4()),
            payload={
                "features": semantic_state.metadata["numeric_features"].detach().cpu().tolist(),
                "operation_id": int(semantic_state.metadata["operation_id"]),
            },
            target=float(target),
            priority=max(0.05, priority),
        )
        self.replay_buffer.add(sample)
        current_batch = self._batch_from_semantic_state(semantic_state)
        replay_samples = self.replay_buffer.sample(self.replay_batch_size)
        replay_batch = self._batch_from_samples(replay_samples) if replay_samples else None
        metrics = self.plastic_learner.train_step(current_batch, replay_batch)
        metrics["replay_size"] = float(len(self.replay_buffer))
        return metrics

    def evaluate(self, dataset: Iterable[Dict[str, float]]) -> Dict[str, float]:
        predictions = []
        targets = []
        self.plastic_learner.eval()
        with torch.no_grad():
            for item in dataset:
                features = torch.tensor([item["features"]], dtype=torch.float32, device=self.device)
                operation_id = torch.tensor([int(item["operation_id"])], dtype=torch.long, device=self.device)
                batch = NumericBatch(features=features, operation_ids=operation_id)
                semantic_state = SemanticState(
                    raw_input=item,
                    modality="numeric",
                    sequence_embedding=None,
                    pooled_embedding=torch.zeros(32, device=self.device),
                    intent=f"arithmetic_{item['operation']}",
                    entities=[],
                    concepts=[],
                    concept_graph_edges=[],
                    compressed_context="evaluation",
                    metadata={
                        "numeric_features": features.squeeze(0),
                        "operation_id": int(item["operation_id"]),
                        "operation": item["operation"],
                        "a": item["a"],
                        "b": item["b"],
                    },
                )
                inference = self.plastic_learner.predict(semantic_state)
                predictions.append(float(inference.prediction))
                targets.append(float(item["target"]))
        prediction_tensor = torch.tensor(predictions)
        target_tensor = torch.tensor(targets)
        mae = torch.mean(torch.abs(prediction_tensor - target_tensor)).item()
        accuracy = torch.mean((torch.abs(prediction_tensor - target_tensor) < 0.5).float()).item()
        return {"mae": float(mae), "accuracy": float(accuracy)}

    def _batch_from_semantic_state(self, semantic_state: SemanticState) -> NumericBatch:
        features = semantic_state.metadata["numeric_features"].float().to(self.device).unsqueeze(0)
        operation_id = torch.tensor([semantic_state.metadata["operation_id"]], dtype=torch.long, device=self.device)
        target = torch.tensor([float(semantic_state.metadata["target"])], dtype=torch.float32, device=self.device)
        return NumericBatch(features=features, operation_ids=operation_id, targets=target)

    def _batch_from_samples(self, samples: List[ReplaySample]) -> NumericBatch:
        features = torch.tensor([sample.payload["features"] for sample in samples], dtype=torch.float32, device=self.device)
        operation_ids = torch.tensor([sample.payload["operation_id"] for sample in samples], dtype=torch.long, device=self.device)
        targets = torch.tensor([sample.target for sample in samples], dtype=torch.float32, device=self.device)
        return NumericBatch(features=features, operation_ids=operation_ids, targets=targets)

