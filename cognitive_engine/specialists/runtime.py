from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from cognitive_engine.core.types import MemoryBundleV2, SemanticStateV2
from cognitive_engine.interfaces.base import Specialist


@dataclass
class SpecialistManifest:
    name: str
    version: str
    domains: List[str]
    tools: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodingSpecialist(Specialist):
    def __init__(self, manifest: SpecialistManifest) -> None:
        self.manifest = manifest
        self.name = manifest.name

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.manifest.version,
            "domains": self.manifest.domains,
            "tools": self.manifest.tools,
            "procedures": self.manifest.procedures,
        }

    def can_handle(self, semantic_state: SemanticStateV2) -> float:
        text = str(semantic_state.raw_input).lower()
        concepts = " ".join([concept.label.lower() for concept in semantic_state.concepts])
        haystack = f"{text} {concepts}"
        hits = sum(1 for domain in self.manifest.domains if domain.lower() in haystack)
        if hits:
            return min(1.0, 0.45 + 0.2 * hits)
        if self.name == "general_coding" and any(marker in haystack for marker in ["code", "bug", "test", "class", "function"]):
            return 0.45
        return 0.0

    def prepare_context(self, semantic_state: SemanticStateV2, memory_bundle: MemoryBundleV2) -> Dict[str, Any]:
        procedures = [
            procedure.title
            for procedure in memory_bundle.procedural
            if any(domain in self.manifest.domains for domain in procedure.domains)
        ]
        return {
            "specialist": self.name,
            "tools": self.manifest.tools,
            "procedures": procedures or self.manifest.procedures,
            "domains": self.manifest.domains,
        }


class SpecialistRuntime:
    name = "specialist_runtime"

    def __init__(self) -> None:
        self.specialists: Dict[str, CodingSpecialist] = {}
        self._register_defaults()

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "specialists": {name: spec.describe() for name, spec in self.specialists.items()}}

    def register(self, specialist: CodingSpecialist) -> None:
        self.specialists[specialist.name] = specialist

    def select(self, semantic_state: SemanticStateV2, requested: List[str] | None = None, threshold: float = 0.35) -> List[CodingSpecialist]:
        if requested:
            resolved = [self.specialists[name] for name in requested if name in self.specialists]
            if resolved:
                return resolved
        scored = [(specialist.can_handle(semantic_state), specialist) for specialist in self.specialists.values()]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [specialist for score, specialist in scored if score >= threshold][:3]

    def prepare_context(self, semantic_state: SemanticStateV2, memory_bundle: MemoryBundleV2, requested: List[str]) -> List[Dict[str, Any]]:
        return [specialist.prepare_context(semantic_state, memory_bundle) for specialist in self.select(semantic_state, requested)]

    def _register_defaults(self) -> None:
        manifests = [
            SpecialistManifest("python", "2.0.0", ["python", "pytest", "pip", "torch"], ["pytest", "ruff", "mypy"], ["run focused pytest", "inspect imports"]),
            SpecialistManifest("godot", "2.0.0", ["godot", "gdscript", "area2d", "rigidbody2d"], ["godot"], ["inspect nodes", "validate signals"]),
            SpecialistManifest("rust", "2.0.0", ["rust", "cargo", "borrow"], ["cargo test", "clippy"], ["check ownership", "run cargo test"]),
            SpecialistManifest("cpp", "2.0.0", ["c++", "cpp", "cmake"], ["cmake", "ctest"], ["check headers", "run targeted build"]),
            SpecialistManifest("cuda", "2.0.0", ["cuda", "kernel", "gpu"], ["nvcc"], ["check shapes", "check device capability"]),
            SpecialistManifest("kubernetes", "2.0.0", ["kubernetes", "k8s", "helm"], ["kubectl", "helm"], ["validate manifests", "inspect rollout"]),
            SpecialistManifest("unreal", "2.0.0", ["unreal", "blueprint", "ue"], ["unrealbuildtool"], ["inspect modules", "validate blueprint boundary"]),
            SpecialistManifest("general_coding", "2.0.0", ["code", "bug", "test"], ["git", "pytest"], ["reproduce", "patch", "validate"]),
        ]
        for manifest in manifests:
            self.register(CodingSpecialist(manifest))

