from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn

from cognitive_engine.core.types import ProcessedInput, RoutingDecision, RoutingDecisionV2, SemanticState, SemanticStateV2
from cognitive_engine.interfaces.base import LearnedRouter
from cognitive_engine.routing.dynamic_router import AdaptiveDynamicRouter


class LearnedCognitiveRouter(nn.Module, LearnedRouter):
    name = "learned_cognitive_router"

    def __init__(self, fallback_router: AdaptiveDynamicRouter | None = None, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.fallback_router = fallback_router or AdaptiveDynamicRouter()
        self.policy = nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 8), nn.Sigmoid())
        self.to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        self.trained_steps = 0

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "trained_steps": self.trained_steps, "fallback": self.fallback_router.name}

    def route(self, processed: ProcessedInput, semantic_state: SemanticState | None = None) -> RoutingDecision:
        return self.fallback_router.route(processed, semantic_state)

    def route_v2(self, semantic_state: SemanticStateV2, runtime_context: Dict[str, Any]) -> RoutingDecisionV2:
        features = self._features(semantic_state, runtime_context).to(self.device)
        with torch.no_grad():
            gates = self.policy(features).squeeze(0).detach().cpu()
        gate_scores = {
            "semantic_memory": float(gates[0]),
            "episodic_memory": float(gates[1]),
            "graph_memory": float(gates[2]),
            "project_memory": float(gates[3]),
            "procedural_memory": float(gates[4]),
            "specialists": float(gates[5]),
            "tools": float(gates[6]),
            "learn": float(gates[7]),
        }
        gate_scores = self._calibrate_gates(semantic_state, runtime_context, gate_scores)
        selected_specialists = self._select_specialists(semantic_state, gate_scores)
        consult_memories = [name for name, score in gate_scores.items() if name.endswith("memory") and score >= 0.35]
        if not consult_memories:
            consult_memories = ["semantic_memory"]
        update_memory = gate_scores["learn"] >= 0.42 and semantic_state.intent != "question"
        engage_plasticity = bool(selected_specialists) or update_memory
        compute_budget = min(1.0, 0.25 + 0.08 * len(consult_memories) + 0.15 * len(selected_specialists))
        context_budget = 8192
        if gate_scores["project_memory"] > 0.55 or gate_scores["graph_memory"] > 0.55:
            context_budget = 32768
        if runtime_context.get("long_context"):
            context_budget = 131072
        consolidation_action = "light_sleep" if gate_scores["learn"] > 0.75 else "none"
        learning_action = "learn" if update_memory else "observe"
        rationale = (
            f"learned gates memory={consult_memories}, specialists={selected_specialists}, "
            f"learn={gate_scores['learn']:.2f}, graph={gate_scores['graph_memory']:.2f}"
        )
        return RoutingDecisionV2(
            active_modules=["semantic_backbone_v2", "learned_cognitive_router", "hybrid_cognitive_memory", "stable_core"],
            consult_memories=consult_memories,
            update_memory=update_memory,
            engage_plasticity=engage_plasticity,
            compute_budget=compute_budget,
            rationale=rationale,
            selected_specialists=selected_specialists,
            memory_plan={name: score for name, score in gate_scores.items() if name.endswith("memory")},
            tool_plan=["lsp", "pytest"] if gate_scores["tools"] > 0.55 else [],
            context_budget=context_budget,
            learning_action=learning_action,
            consolidation_action=consolidation_action,
            confidence=float(torch.mean(gates).item()),
            gate_scores=gate_scores,
            fallback_plan="adaptive_dynamic_router",
        )

    def train_from_examples(self, examples: Iterable[Tuple[SemanticStateV2, Dict[str, Any], List[float]]], epochs: int = 80) -> Dict[str, float]:
        rows = list(examples)
        if not rows:
            return {"loss": 0.0, "examples": 0}
        features = torch.cat([self._features(state, context) for state, context, _ in rows], dim=0).to(self.device)
        targets = torch.tensor([target for _, _, target in rows], dtype=torch.float32, device=self.device)
        loss_value = 0.0
        for _ in range(epochs):
            pred = self.policy(features)
            loss = nn.functional.binary_cross_entropy(pred, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.item())
            self.trained_steps += 1
        return {"loss": loss_value, "examples": len(rows)}

    def _features(self, semantic_state: SemanticState, runtime_context: Dict[str, Any]) -> torch.Tensor:
        concepts = len(semantic_state.concepts)
        token_count = float(semantic_state.metadata.get("token_count", 0))
        intent = semantic_state.intent
        text = str(semantic_state.raw_input).lower()
        features = [
            1.0 if semantic_state.modality == "text" else 0.0,
            1.0 if semantic_state.modality == "numeric" else 0.0,
            1.0 if intent == "question" else 0.0,
            1.0 if intent in {"knowledge_share", "preference", "correction"} else 0.0,
            min(1.0, concepts / 8.0),
            min(1.0, token_count / 256.0),
            1.0 if any(marker in text for marker in ["code", "python", "pytest", "class", "function", "error", "bug"]) else 0.0,
            1.0 if runtime_context.get("project_id") else 0.0,
            1.0 if runtime_context.get("has_project_memory") else 0.0,
            1.0 if runtime_context.get("long_context") else 0.0,
        ]
        return torch.tensor([features], dtype=torch.float32, device=self.device)

    def _select_specialists(self, semantic_state: SemanticState, gates: Dict[str, float]) -> List[str]:
        if gates["specialists"] < 0.35:
            return []
        text = str(semantic_state.raw_input).lower()
        selected = []
        if any(marker in text for marker in ["python", "pytest", ".py", "pip", "torch"]):
            selected.append("python")
        if "godot" in text or "gdscript" in text:
            selected.append("godot")
        if "rust" in text or "cargo" in text:
            selected.append("rust")
        if "cuda" in text:
            selected.append("cuda")
        return selected or ["general_coding"]

    def _calibrate_gates(self, semantic_state: SemanticState, runtime_context: Dict[str, Any], gates: Dict[str, float]) -> Dict[str, float]:
        calibrated = dict(gates)
        text = str(semantic_state.raw_input).lower()
        is_noise = semantic_state.intent == "small_talk" or any(marker in text for marker in ["ruido", "spam", "sin valor"])
        is_learning_intent = semantic_state.intent in {"knowledge_share", "preference", "correction"}
        is_code = any(marker in text for marker in ["python", "pytest", "class", "def ", "bug", "error", ".py", "godot", "rust", "cuda"])

        if is_noise and not is_code:
            calibrated["learn"] = min(calibrated["learn"], 0.12)
            calibrated["specialists"] = min(calibrated["specialists"], 0.12)
            calibrated["tools"] = min(calibrated["tools"], 0.12)
            calibrated["procedural_memory"] = min(calibrated["procedural_memory"], 0.2)

        if semantic_state.intent == "question":
            calibrated["learn"] = min(calibrated["learn"], 0.18)
            calibrated["semantic_memory"] = max(calibrated["semantic_memory"], 0.72)
            calibrated["episodic_memory"] = max(calibrated["episodic_memory"], 0.55)

        if is_learning_intent:
            calibrated["learn"] = max(calibrated["learn"], 0.72)
            calibrated["semantic_memory"] = max(calibrated["semantic_memory"], 0.6)

        if is_code:
            calibrated["specialists"] = max(calibrated["specialists"], 0.76)
            calibrated["tools"] = max(calibrated["tools"], 0.58)
            calibrated["graph_memory"] = max(calibrated["graph_memory"], 0.68)
            calibrated["procedural_memory"] = max(calibrated["procedural_memory"], 0.58)

        if runtime_context.get("project_id") or runtime_context.get("has_project_memory"):
            calibrated["project_memory"] = max(calibrated["project_memory"], 0.7)
            calibrated["graph_memory"] = max(calibrated["graph_memory"], 0.65)

        return {key: max(0.0, min(1.0, value)) for key, value in calibrated.items()}
