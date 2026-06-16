from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from cognitive_engine.core.types import CoreInference, NumericBatch, SemanticState
from cognitive_engine.interfaces.base import PlasticLearner


class LowRankResidual(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.frozen = nn.Linear(input_dim, hidden_dim)
        self.adapter_a = nn.Linear(input_dim, rank, bias=False)
        self.adapter_b = nn.Linear(rank, hidden_dim, bias=False)
        self.scale = alpha / rank
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.frozen(inputs) + self.adapter_b(self.adapter_a(inputs)) * self.scale


class PlasticArithmeticModule(nn.Module, PlasticLearner):
    name = "plastic_arithmetic_module"

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 48,
        operations: int = 3,
        output_scale: float = 144.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.operations = operations
        self.output_scale = output_scale
        self.adapter = LowRankResidual(input_dim + operations, hidden_dim)
        self.hidden = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.direct_head = nn.Linear(input_dim + operations, operations)
        self.residual_head = nn.Linear(hidden_dim, operations)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.loss_fn = nn.SmoothL1Loss()
        self.to(self.device)
        self.optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=2e-3, weight_decay=1e-4)
        self._initial_trainable_state = deepcopy(self._trainable_state_dict())

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "operations": self.operations, "trainable_parameters": sum(p.numel() for p in self.trainable_parameters())}

    def predict(self, semantic_state: SemanticState) -> CoreInference:
        batch = self._batch_from_semantic_state(semantic_state)
        predictions, confidence = self._forward(batch.features, batch.operation_ids)
        value = float(predictions.squeeze(0).detach().cpu().item())
        confidence_score = float(confidence.squeeze(0).detach().cpu().item())
        return CoreInference(
            prediction=value,
            confidence=max(0.01, min(0.999, confidence_score)),
            explanation=f"Plastic arithmetic pathway predicted {value:.3f} for operation {semantic_state.metadata['operation']}.",
            artifacts={"raw_prediction": value, "confidence": confidence_score},
        )

    def train_step(self, batch: NumericBatch, replay_batch: Optional[NumericBatch] = None) -> Dict[str, float]:
        self.train()
        features = batch.features
        operation_ids = batch.operation_ids
        targets = batch.targets
        replay_ratio = 0.0
        if replay_batch is not None and replay_batch.targets is not None:
            replay_ratio = len(replay_batch.targets) / max(len(targets), 1)
            features = torch.cat([features, replay_batch.features], dim=0)
            operation_ids = torch.cat([operation_ids, replay_batch.operation_ids], dim=0)
            targets = torch.cat([targets, replay_batch.targets], dim=0)

        predictions, confidence = self._forward(features, operation_ids)
        normalized_targets = targets / self.output_scale
        main_loss = self.loss_fn(predictions / self.output_scale, normalized_targets)
        confidence_loss = F.mse_loss(confidence, torch.exp(-torch.abs(predictions - targets) / self.output_scale))
        stability_penalty = 5e-4 * self.parameter_drift()
        loss = main_loss + 0.15 * confidence_loss + stability_penalty

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.trainable_parameters()), max_norm=1.0)
        self.optimizer.step()

        mae = torch.mean(torch.abs(predictions - targets)).item()
        accuracy = torch.mean((torch.abs(predictions - targets) < 0.5).float()).item()
        return {
            "loss": float(loss.item()),
            "mae": float(mae),
            "accuracy": float(accuracy),
            "replay_ratio": float(replay_ratio),
            "plastic_norm": float(self.parameter_drift()),
            "stable_drift": 0.0,
        }

    def parameter_drift(self) -> float:
        initial = self._initial_trainable_state
        total = 0.0
        for name, current in self._trainable_state_dict().items():
            total += torch.norm(current - initial[name]).item()
        return total

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _forward(self, features: torch.Tensor, operation_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        op_one_hot = F.one_hot(operation_ids, num_classes=self.operations).float()
        merged = torch.cat([features, op_one_hot], dim=-1)
        hidden = self.hidden(self.adapter(merged))
        outputs = self.direct_head(merged) + 0.35 * self.residual_head(hidden)
        gathered = outputs.gather(1, operation_ids.unsqueeze(1)).squeeze(1) * self.output_scale
        confidence = torch.sigmoid(self.confidence_head(hidden)).squeeze(1)
        return gathered, confidence

    def _batch_from_semantic_state(self, semantic_state: SemanticState) -> NumericBatch:
        features = semantic_state.metadata["numeric_features"].float().to(self.device).unsqueeze(0)
        operation_id = torch.tensor([semantic_state.metadata["operation_id"]], dtype=torch.long, device=self.device)
        target = semantic_state.metadata.get("target")
        targets = None
        if target is not None:
            targets = torch.tensor([float(target)], dtype=torch.float32, device=self.device)
        return NumericBatch(features=features, operation_ids=operation_id, targets=targets)

    def _trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: tensor.detach().clone()
            for name, tensor in self.state_dict().items()
            if self.state_dict()[name].dtype.is_floating_point and self._is_trainable_name(name)
        }

    def _is_trainable_name(self, name: str) -> bool:
        if name.startswith("adapter.frozen"):
            return False
        return True
