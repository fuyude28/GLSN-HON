"""Exact single-node CoreHD ordering on an undirected projection."""

from __future__ import annotations

import heapq

from corehd_dismantle import tree_break


def corehd_order(num_nodes: int, directed_edges: tuple[tuple[int, int], ...]
                 ) -> list[int]:
    """Iteratively remove max degree in the live 2-core, then break trees.

    Leaves pruned from the 2-core remain in the graph. The heap maintains
    core degrees incrementally, matching one-at-a-time 2-core recomputation.
    """
    neighbors = {node: set() for node in range(num_nodes)}
    for source, target in directed_edges:
        if source != target:
            neighbors[source].add(target)
            neighbors[target].add(source)

    core = set(range(num_nodes))
    degree = {node: len(adjacent) for node, adjacent in neighbors.items()}
    queue = [node for node, value in degree.items() if value <= 1]
    heap = [(-degree[node], node) for node in core]
    heapq.heapify(heap)

    def peel() -> None:
        while queue:
            node = queue.pop()
            if node not in core or degree[node] > 1:
                continue
            core.remove(node)
            for neighbor in neighbors[node] & core:
                degree[neighbor] -= 1
                if degree[neighbor] <= 1:
                    queue.append(neighbor)
                else:
                    heapq.heappush(heap, (-degree[neighbor], neighbor))

    peel()
    removed = []
    while core:
        minus_degree, node = heapq.heappop(heap)
        if node not in core or -minus_degree != degree[node]:
            continue
        core.remove(node)
        removed.append(node)
        for neighbor in neighbors[node] & core:
            degree[neighbor] -= 1
            if degree[neighbor] <= 1:
                queue.append(neighbor)
            else:
                heapq.heappush(heap, (-degree[neighbor], neighbor))
        peel()

    active = set(range(num_nodes)) - set(removed)
    tree_break(active, {node: {v: 1.0 for v in adjacent}
                        for node, adjacent in neighbors.items()}, 1, removed)
    removed.extend(sorted(active))
    return removed
