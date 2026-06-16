from __future__ import annotations

from typing import Any, Dict, List

from cognitive_engine.core.types import StabilityDecision
from cognitive_engine.interfaces.base import StabilityGovernor


class StabilityGovernorV2(StabilityGovernor):
    name = "stability_governor_v2"

    def __init__(self, max_drift: float = 25.0, min_confidence: float = 0.35) -> None:
        self.max_drift = max_drift
        self.min_confidence = min_confidence

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "max_drift": self.max_drift, "min_confidence": self.min_confidence}

    def validate_update(self, update: Dict[str, Any]) -> StabilityDecision:
        reasons: List[str] = []
        risk = 0.0
        confidence = float(update.get("confidence", 0.5))
        drift = float(update.get("plastic_drift", 0.0))
        contradiction = float(update.get("contradiction_risk", 0.0))
        if confidence < self.min_confidence:
            risk += 0.35
            reasons.append("low confidence")
        if drift > self.max_drift:
            risk += 0.35
            reasons.append("plastic drift above limit")
        if contradiction > 0.55:
            risk += 0.3
            reasons.append("high contradiction risk")
        approved = risk < 0.5
        if approved and not reasons:
            reasons.append("update within stability limits")
        return StabilityDecision(approved=approved, confidence=confidence, risk_score=min(1.0, risk), reasons=reasons)

