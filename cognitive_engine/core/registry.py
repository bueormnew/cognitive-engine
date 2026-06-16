from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict


Factory = Callable[..., Any]


class ComponentRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, Dict[str, Factory]] = defaultdict(dict)

    def register(self, category: str, name: str, factory: Factory) -> None:
        self._factories[category][name] = factory

    def create(self, category: str, name: str, **kwargs: Any) -> Any:
        try:
            factory = self._factories[category][name]
        except KeyError as exc:
            raise KeyError(f"Component '{name}' is not registered in category '{category}'.") from exc
        return factory(**kwargs)

    def available(self, category: str) -> Dict[str, Factory]:
        return dict(self._factories.get(category, {}))


GLOBAL_REGISTRY = ComponentRegistry()

