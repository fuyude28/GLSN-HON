"""Exact single-input control centrality via Hosoe's stem-cycle packing.

For each input node, the accessible directed subgraph is augmented with one
input vertex, zero-weight self-loops, and zero-weight returns to the input.
A maximum-weight cycle cover then consists of one input-rooted stem and
vertex-disjoint cycles. Positive weights count original/input edges.
"""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import igraph as ig
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "DND-DRL"))
from dnd_drl.graph import DirectedGraph

_WORKER_GRAPH: DirectedGraph | None = None
_WORKER_TOPOLOGY: ig.Graph | None = None


def control_centrality(graph: DirectedGraph, source: int,
                       topology: ig.Graph | None = None) -> tuple[int, int]:
    """Return exact generic controllable dimension and accessible-node count."""
    topology = topology or ig.Graph(n=graph.num_nodes, edges=graph.edges,
                                    directed=True)
    reachable = sorted(topology.subcomponent(source, mode="out"))
    index = {node: pos for pos, node in enumerate(reachable)}
    input_vertex = len(reachable)
    costs = {(pos, pos): 2 for pos in range(input_vertex + 1)}
    costs.update({(pos, input_vertex): 2 for pos in range(input_vertex)})
    costs[(input_vertex, index[source])] = 1
    for u, v in graph.edges:
        if u in index and v in index:
            costs[(index[u], index[v])] = 1
    rows, columns = zip(*costs)
    values = np.fromiter(costs.values(), dtype=np.int8)
    assignment = csr_matrix((values, (rows, columns)),
                            shape=(input_vertex + 1, input_vertex + 1))
    assigned_rows, assigned_columns = min_weight_full_bipartite_matching(
        assignment)
    score = int(np.sum(assignment[assigned_rows, assigned_columns].A1 == 1))
    return score, input_vertex


def _init_worker(graph: DirectedGraph) -> None:
    global _WORKER_GRAPH, _WORKER_TOPOLOGY
    _WORKER_GRAPH = graph
    _WORKER_TOPOLOGY = ig.Graph(n=graph.num_nodes, edges=graph.edges,
                                directed=True)


def _score_worker(node: int) -> tuple[int, int, int]:
    score, reachable = control_centrality(_WORKER_GRAPH, node,
                                           _WORKER_TOPOLOGY)
    return node, score, reachable


def score_graph(graph: DirectedGraph, out: Path, workers: int = 1) -> None:
    topology = ig.Graph(n=graph.num_nodes, edges=graph.edges, directed=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("state_node", "control_centrality", "normalized_cc",
                         "reachable_nodes"))
        if workers > 1:
            pool = ProcessPoolExecutor(max_workers=workers,
                                       initializer=_init_worker,
                                       initargs=(graph,))
            records = pool.map(_score_worker, range(graph.num_nodes),
                               chunksize=8)
        else:
            pool = None
            records = ((node, *control_centrality(graph, node, topology))
                       for node in range(graph.num_nodes))
        try:
            for node, score, reachable in records:
                writer.writerow((graph.original_node(node), score,
                                 score / graph.num_nodes, reachable))
                if (node + 1) % 100 == 0:
                    handle.flush()
                    print(f"scored {node + 1}/{graph.num_nodes} nodes; "
                          f"elapsed {time.perf_counter() - start:.1f}s", flush=True)
        finally:
            if pool is not None:
                pool.shutdown()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path,
                        default=HERE.parents[3] / "data/network-ho.csv")
    parser.add_argument("--out", type=Path,
                        default=HERE / "results/control_centrality_scores.csv")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    score_graph(DirectedGraph.read_csv_edge_table(args.graph), args.out,
                args.workers)
    print("saved", args.out)


if __name__ == "__main__":
    main()
