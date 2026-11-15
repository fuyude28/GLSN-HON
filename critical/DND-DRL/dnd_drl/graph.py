"""Directed graph input and residual GSCC measurements."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable, Sequence

import igraph as ig
import networkx as nx


@dataclass(frozen=True)
class DirectedGraph:
    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    node_mapping: dict[Hashable, int] | None = None
    reverse_mapping: dict[int, Hashable] | None = None

    @classmethod
    def from_edges(cls, num_nodes: int, edges: Iterable[tuple[int, int]]) -> DirectedGraph:
        ordered = set()
        for source, target in edges:
            u, v = int(source), int(target)
            if not (0 <= u < num_nodes and 0 <= v < num_nodes):
                raise ValueError(f"edge ({u}, {v}) outside node range 0..{num_nodes - 1}")
            ordered.add((u, v))
        return cls(num_nodes, tuple(sorted(ordered)))

    @classmethod
    def read_csv_edge_table(
        cls, path: str | Path, source_col: str = "FromNode", target_col: str = "ToNode"
    ) -> DirectedGraph:
        mapping: dict[str, int] = {}
        edges: list[tuple[int, int]] = []

        def node_id(label: str) -> int:
            if label not in mapping:
                mapping[label] = len(mapping)
            return mapping[label]

        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if source_col not in reader.fieldnames or target_col not in reader.fieldnames:
                raise ValueError(f"expected columns {source_col}, {target_col}; got {reader.fieldnames}")
            for row in reader:
                edges.append((node_id(row[source_col]), node_id(row[target_col])))
        graph = cls.from_edges(len(mapping), edges)
        return cls(graph.num_nodes, graph.edges, mapping, {v: k for k, v in mapping.items()})

    def original_node(self, node: int) -> Hashable:
        return node if self.reverse_mapping is None else self.reverse_mapping[node]

    def active_nodes(self, removed: Sequence[int] | set[int]) -> list[int]:
        excluded = set(removed)
        return [node for node in range(self.num_nodes) if node not in excluded]

    def residual_edges(self, removed: Sequence[int] | set[int]) -> list[tuple[int, int]]:
        excluded = set(removed)
        return [(u, v) for u, v in self.edges if u not in excluded and v not in excluded]

    def gscc_state(self, removed: Sequence[int] | set[int] = ()) -> tuple[int, list[int]]:
        """Largest SCC size and nodes in all tied largest nontrivial SCCs."""
        residual = ig.Graph(n=self.num_nodes, edges=self.residual_edges(removed),
                            directed=True)
        components = residual.connected_components(mode="strong")
        sizes = components.sizes()
        largest = max(sizes, default=0)
        if largest <= 1:
            return largest, []
        return largest, [node for node, component in enumerate(components.membership)
                         if sizes[component] == largest]

    def gscc_size(self, removed: Sequence[int] | set[int] = ()) -> int:
        return self.gscc_state(removed)[0]

    def gscc_candidates(self, removed: Sequence[int] | set[int] = ()) -> list[int]:
        return self.gscc_state(removed)[1]


def directed_ba_graph(
    num_nodes: int, m: int = 2, reciprocity: float = 0.1,
    hub_source_bias: float = 0.6, seed: int | None = None
) -> DirectedGraph:
    """Orient sparse BA edges, preferentially making hubs the edge source."""
    if not 0 <= reciprocity <= 1 or not 0 <= hub_source_bias <= 1:
        raise ValueError("reciprocity and hub_source_bias must be between 0 and 1")
    if num_nodes <= 1:
        return DirectedGraph.from_edges(num_nodes, ())
    rng = random.Random(seed)
    undirected = nx.barabasi_albert_graph(num_nodes, min(max(1, m), num_nodes - 1), seed=seed)
    degrees = dict(undirected.degree())
    edges = []
    for u, v in undirected.edges():
        if rng.random() < reciprocity:
            edges.extend(((u, v), (v, u)))
        else:
            if degrees[u] == degrees[v]:
                forward = rng.random() < 0.5
            else:
                forward = ((degrees[u] > degrees[v])
                           if rng.random() < hub_source_bias else
                           (degrees[u] < degrees[v]))
            edges.append((u, v) if forward else (v, u))
    return DirectedGraph.from_edges(num_nodes, edges)
