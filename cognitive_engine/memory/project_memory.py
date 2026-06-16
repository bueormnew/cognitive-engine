from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List

from cognitive_engine.core.types import GraphPatch, ProjectMemoryRecord
from cognitive_engine.memory.graph_memory import CognitiveGraphMemory


class ProjectIndexer:
    name = "project_indexer"

    def __init__(self, graph_memory: CognitiveGraphMemory) -> None:
        self.graph_memory = graph_memory

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name}

    def index_project(self, root_path: str | Path, project_id: str | None = None) -> ProjectMemoryRecord:
        root = Path(root_path).resolve()
        project_id = project_id or root.name
        py_files = [path for path in root.rglob("*.py") if ".venv" not in path.parts and "__pycache__" not in path.parts]
        modules: List[str] = []
        dependencies: set[str] = set()
        tests: List[str] = []

        project_node = self.graph_memory.upsert_node(project_id, "Project", project_id=project_id, metadata={"root": str(root)})
        for file_path in py_files:
            rel = str(file_path.relative_to(root))
            file_node = self.graph_memory.upsert_node(rel, "File", namespace=project_id, project_id=project_id)
            self.graph_memory.upsert_edge(project_node, file_node, "owns", metadata={"path": rel})
            if "test" in file_path.name:
                tests.append(rel)
            module_name = rel.replace("\\", ".").replace("/", ".").removesuffix(".py")
            modules.append(module_name)
            module_node = self.graph_memory.upsert_node(module_name, "Module", namespace=project_id, project_id=project_id)
            self.graph_memory.upsert_edge(file_node, module_node, "defines")
            self._index_python_file(file_path, file_node, module_node, dependencies, project_id)

        return ProjectMemoryRecord(
            project_id=project_id,
            root_path=str(root),
            files_indexed=len(py_files),
            modules=sorted(modules)[:80],
            dependencies=sorted(dependencies),
            tests=sorted(tests),
            commands=["pytest -q"] if tests else [],
            metadata={"graph": self.graph_memory.snapshot()},
        )

    def _index_python_file(self, file_path: Path, file_node: Any, module_node: Any, dependencies: set[str], project_id: str) -> None:
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            tree = ast.parse(file_path.read_text(encoding="latin-1"))
        except SyntaxError:
            return

        for item in ast.walk(tree):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = self.graph_memory.upsert_node(item.name, "Function", namespace=str(file_path), project_id=project_id, metadata={"line": item.lineno})
                self.graph_memory.upsert_edge(file_node, fn, "defines")
                self.graph_memory.upsert_edge(module_node, fn, "defines")
            elif isinstance(item, ast.ClassDef):
                cls = self.graph_memory.upsert_node(item.name, "Class", namespace=str(file_path), project_id=project_id, metadata={"line": item.lineno})
                self.graph_memory.upsert_edge(file_node, cls, "defines")
                self.graph_memory.upsert_edge(module_node, cls, "defines")
            elif isinstance(item, ast.Import):
                for alias in item.names:
                    dep = alias.name.split(".")[0]
                    dependencies.add(dep)
                    dep_node = self.graph_memory.upsert_node(dep, "Dependency", project_id=project_id)
                    self.graph_memory.upsert_edge(file_node, dep_node, "imports")
            elif isinstance(item, ast.ImportFrom) and item.module:
                dep = item.module.split(".")[0]
                dependencies.add(dep)
                dep_node = self.graph_memory.upsert_node(dep, "Dependency", project_id=project_id)
                self.graph_memory.upsert_edge(file_node, dep_node, "imports")

