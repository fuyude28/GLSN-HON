from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable, Sequence

import csv
import networkx as nx


@dataclass(frozen=True)
class FinderGraph:
    """Small immutable undirected graph wrapper for FINDER-ND."""

    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    node_mapping: dict[Hashable, int] | None = None
    reverse_mapping: dict[int, Hashable] | None = None

    @classmethod
    def from_edges(cls, num_nodes: int, edges: Iterable[tuple[int, int]]) -> "FinderGraph":
        normalized = set()
        for raw_u, raw_v in edges:
            u = int(raw_u)
            v = int(raw_v)
            if u == v:
                continue
            if not (0 <= u < num_nodes and 0 <= v < num_nodes):
                raise ValueError(f"edge ({u}, {v}) outside node range 0..{num_nodes - 1}")
            normalized.add((u, v) if u < v else (v, u))
        return cls(num_nodes=num_nodes, edges=tuple(sorted(normalized)))

    @classmethod
    def from_networkx(cls, graph: nx.Graph) -> "FinderGraph":
        nodes = list(graph.nodes())
        mapping = {node: idx for idx, node in enumerate(nodes)}
        reverse = {idx: node for node, idx in mapping.items()}
        edges = [(mapping[u], mapping[v]) for u, v in graph.edges()]
        return cls.from_edges(len(nodes), edges).with_mapping(mapping, reverse)

    @classmethod
    def read_edgelist(cls, path: str | Path, delimiter: str | None = None) -> "FinderGraph":
        graph = nx.read_edgelist(str(path), delimiter=delimiter, nodetype=str)
        return cls.from_networkx(graph)

    @classmethod
    def read_csv_edge_table(
        cls,
        path: str | Path,
        source_col: str = "node1",
        target_col: str = "node2",
    ) -> "FinderGraph":
        mapping: dict[str, int] = {}
        edges: list[tuple[int, int]] = []

        def get_id(raw_node: str) -> int:
            node = raw_node.strip()
            if node not in mapping:
                mapping[node] = len(mapping)
            return mapping[node]

        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if source_col not in (reader.fieldnames or []) or target_col not in (reader.fieldnames or []):
                raise ValueError(
                    f"CSV must contain columns {source_col!r} and {target_col!r}; "
                    f"found {reader.fieldnames}"
                )
            for row in reader:
                edges.append((get_id(row[source_col]), get_id(row[target_col])))

        reverse = {idx: node for node, idx in mapping.items()}
        return cls.from_edges(len(mapping), edges).with_mapping(mapping, reverse)

    def with_mapping(
        self, mapping: dict[Hashable, int], reverse: dict[int, Hashable]
    ) -> "FinderGraph":
        return FinderGraph(self.num_nodes, self.edges, dict(mapping), dict(reverse))

    def original_node(self, node: int) -> Hashable:
        if self.reverse_mapping is None:
            return node
        return self.reverse_mapping[node]

    def internal_node(self, node: Hashable) -> int:
        if self.node_mapping is None:
            return int(node)
        return self.node_mapping[node]

    def adjacency(self) -> list[set[int]]:
        adj = [set() for _ in range(self.num_nodes)]
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def active_nodes(self, removed: Sequence[int] | set[int]) -> list[int]:
        removed_set = set(removed)
        return [node for node in range(self.num_nodes) if node not in removed_set]

    def residual_edges(self, removed: Sequence[int] | set[int]) -> list[tuple[int, int]]:
        removed_set = set(removed)
        return [(u, v) for u, v in self.edges if u not in removed_set and v not in removed_set]

    def candidate_nodes(self, removed: Sequence[int] | set[int]) -> list[int]:
        removed_set = set(removed)
        candidates = set()
        for u, v in self.edges:
            if u in removed_set or v in removed_set:
                continue
            candidates.add(u)
            candidates.add(v)
        return sorted(candidates)

    def largest_component_size(self, removed: Sequence[int] | set[int]) -> int:
        removed_set = set(removed)
        active = set(range(self.num_nodes)) - removed_set
        if not active:
            return 0

        adj = self.adjacency()
        seen: set[int] = set()
        best = 0
        for start in sorted(active):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                for nbr in adj[node]:
                    if nbr in active and nbr not in seen:
                        seen.add(nbr)
                        stack.append(nbr)
            best = max(best, size)
        return best

    def has_residual_edges(self, removed: Sequence[int] | set[int]) -> bool:
        removed_set = set(removed)
        return any(u not in removed_set and v not in removed_set for u, v in self.edges)

    def to_networkx(self, removed: Sequence[int] | set[int] = ()) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.active_nodes(removed))
        graph.add_edges_from(self.residual_edges(removed))
        return graph


def barabasi_albert_graph(num_nodes: int, m: int = 4, seed: int | None = None) -> FinderGraph:
    if num_nodes <= 1:
        return FinderGraph.from_edges(num_nodes, [])
    safe_m = max(1, min(m, num_nodes - 1))
    return FinderGraph.from_networkx(nx.barabasi_albert_graph(num_nodes, safe_m, seed=seed))
