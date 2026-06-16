from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List
from uuid import uuid4

import networkx as nx

from cognitive_engine.core.types import GraphEdge, GraphNode, GraphPatch, GraphQuery, GraphSubgraph
from cognitive_engine.interfaces.base import GraphMemoryStore


class CognitiveGraphMemory(GraphMemoryStore):
    name = "cognitive_graph_memory"

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.label_index: Dict[str, str] = {}

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "nodes": len(self.nodes), "edges": len(self.edges)}

    def upsert_node(
        self,
        label: str,
        node_type: str,
        namespace: str = "",
        project_id: str = "global",
        confidence: float = 0.75,
        metadata: Dict[str, Any] | None = None,
    ) -> GraphNode:
        key = self._key(label, node_type, namespace, project_id)
        existing_id = self.label_index.get(key)
        if existing_id:
            node = self.nodes[existing_id]
            node.confidence = max(node.confidence, confidence)
            node.version += 1
            node.metadata.update(metadata or {})
            self.graph.nodes[existing_id].update(asdict(node))
            return node

        node = GraphNode(
            node_id=str(uuid4()),
            node_type=node_type,
            label=label,
            namespace=namespace,
            project_id=project_id,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.nodes[node.node_id] = node
        self.label_index[key] = node.node_id
        self.graph.add_node(node.node_id, **asdict(node))
        return node

    def upsert_edge(
        self,
        source: GraphNode,
        target: GraphNode,
        relation: str,
        weight: float = 1.0,
        confidence: float = 0.75,
        evidence_ids: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> GraphEdge:
        for edge_id, edge in self.edges.items():
            if edge.source_id == source.node_id and edge.target_id == target.node_id and edge.relation == relation:
                edge.weight = max(edge.weight, weight)
                edge.confidence = max(edge.confidence, confidence)
                edge.evidence_ids = list(dict.fromkeys(edge.evidence_ids + (evidence_ids or [])))
                edge.metadata.update(metadata or {})
                self.graph[source.node_id][target.node_id][edge_id].update(asdict(edge))
                return edge

        edge = GraphEdge(
            edge_id=str(uuid4()),
            source_id=source.node_id,
            target_id=target.node_id,
            relation=relation,
            weight=weight,
            confidence=confidence,
            evidence_ids=evidence_ids or [],
            metadata=metadata or {},
        )
        self.edges[edge.edge_id] = edge
        self.graph.add_edge(source.node_id, target.node_id, key=edge.edge_id, **asdict(edge))
        return edge

    def apply_patch(self, patch: GraphPatch) -> None:
        id_map: Dict[str, GraphNode] = {}
        for node in patch.nodes:
            upserted = self.upsert_node(
                label=node.label,
                node_type=node.node_type,
                namespace=node.namespace,
                project_id=node.project_id,
                confidence=node.confidence,
                metadata=node.metadata,
            )
            id_map[node.node_id] = upserted
        for edge in patch.edges:
            source = id_map.get(edge.source_id) or self.nodes.get(edge.source_id)
            target = id_map.get(edge.target_id) or self.nodes.get(edge.target_id)
            if source and target:
                self.upsert_edge(source, target, edge.relation, edge.weight, edge.confidence, edge.evidence_ids, edge.metadata)

    def query_subgraph(self, query: GraphQuery) -> GraphSubgraph:
        if not self.nodes:
            return GraphSubgraph(rationale="Graph is empty.")

        seeds = self._resolve_seeds(query)
        visited = set(seeds)
        frontier = set(seeds)
        for _ in range(max(query.max_depth, 0)):
            next_frontier = set()
            for node_id in frontier:
                neighbors = set(self.graph.successors(node_id)) | set(self.graph.predecessors(node_id))
                next_frontier.update(neighbors)
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier

        nodes = [self.nodes[node_id] for node_id in visited if self._matches_project(self.nodes[node_id], query)]
        node_ids = {node.node_id for node in nodes}
        edges = [
            edge
            for edge in self.edges.values()
            if edge.source_id in node_ids
            and edge.target_id in node_ids
            and (not query.relation_filter or edge.relation in query.relation_filter)
        ]
        nodes = nodes[: query.top_k]
        node_ids = {node.node_id for node in nodes}
        edges = [edge for edge in edges if edge.source_id in node_ids and edge.target_id in node_ids]
        return GraphSubgraph(nodes=nodes, edges=edges, rationale=f"Expanded {len(seeds)} seed nodes to {len(nodes)} nodes.")

    def snapshot(self) -> Dict[str, Any]:
        type_counts: Dict[str, int] = {}
        relation_counts: Dict[str, int] = {}
        for node in self.nodes.values():
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1
        for edge in self.edges.values():
            relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges), "node_types": type_counts, "relations": relation_counts}

    def _resolve_seeds(self, query: GraphQuery) -> List[str]:
        if not query.seeds:
            return list(self.nodes)[: query.top_k]
        resolved = []
        normalized_seeds = [seed.lower() for seed in query.seeds]
        for node_id, node in self.nodes.items():
            label = node.label.lower()
            if any(seed in label or label in seed for seed in normalized_seeds):
                if self._matches_project(node, query):
                    resolved.append(node_id)
        return resolved[: query.top_k] or list(self.nodes)[: min(query.top_k, len(self.nodes))]

    def _matches_project(self, node: GraphNode, query: GraphQuery) -> bool:
        return query.project_id is None or node.project_id in {query.project_id, "global"}

    def _key(self, label: str, node_type: str, namespace: str, project_id: str) -> str:
        return f"{project_id}:{namespace}:{node_type}:{label}".lower()

