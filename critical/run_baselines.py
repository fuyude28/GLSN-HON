"""Training-free directed dismantling baselines on the higher-order GLSN."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import igraph as ig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "DND-DRL"))

from dnd_drl.env import DirectedDismantlingEnv
from dnd_drl.graph import DirectedGraph

METHODS = ("dynamic_total_degree", "dynamic_in_degree",
           "dynamic_out_degree", "dynamic_pagerank", "random")
CURVE_FIELDS = ("method", "seed", "batch_round", "removed", "removed_state",
                "gscc_size", "gscc_fraction", "cumulative_auc")
SUMMARY_FIELDS = ("method", "seed", "num_states", "num_edges", "initial_gscc",
                  "batch_fraction", "adaptive_rounds", "removals", "auc", "elapsed_seconds")


def rank_candidates(graph: DirectedGraph, removed: set[int],
                    candidates: list[int], method: str, rng: random.Random,
                    batch_size: int) -> list[int]:
    if method == "random":
        return rng.sample(candidates, batch_size)

    edges = graph.residual_edges(removed)
    if method == "dynamic_pagerank":
        active = graph.active_nodes(removed)
        mapping = {node: index for index, node in enumerate(active)}
        residual = ig.Graph(n=len(active),
                            edges=[(mapping[u], mapping[v]) for u, v in edges],
                            directed=True)
        centrality = residual.pagerank(damping=0.85, directed=True)
        score = lambda node: centrality[mapping[node]]
    else:
        indegree = [0] * graph.num_nodes
        outdegree = [0] * graph.num_nodes
        for source, target in edges:
            outdegree[source] += 1
            indegree[target] += 1
        if method == "dynamic_in_degree":
            score = lambda node: indegree[node]
        elif method == "dynamic_out_degree":
            score = lambda node: outdegree[node]
        else:
            score = lambda node: indegree[node] + outdegree[node]
    return sorted(candidates, key=lambda node: (-score(node), node))[:batch_size]


def dismantle(graph: DirectedGraph, method: str, seed: int,
              batch_fraction: float, curve_writer: csv.DictWriter,
              log_interval: int = 500) -> dict[str, int | float | str]:
    rng = random.Random(seed)
    env = DirectedDismantlingEnv(graph)
    initial_gscc = env.gscc_size
    area = 0.0
    rounds = 0
    start = time.perf_counter()
    curve_writer.writerow(dict(method=method, seed=seed if method == "random" else "",
                               batch_round=0, removed=0, removed_state="",
                               gscc_size=initial_gscc,
                               gscc_fraction=initial_gscc / graph.num_nodes,
                               cumulative_auc=0.0))
    while not env.is_terminal():
        candidates = env.candidate_actions()
        remaining = graph.num_nodes - len(env.removed)
        batch_size = min(len(candidates), max(1, math.ceil(batch_fraction * remaining)))
        actions = rank_candidates(graph, env.removed_set, candidates, method,
                                  rng, batch_size)
        rounds += 1
        for step, size in env.remove_batch(actions):
            area += size / (graph.num_nodes ** 2)
            curve_writer.writerow(dict(
                method=method, seed=seed if method == "random" else "",
                batch_round=rounds, removed=step,
                removed_state=graph.original_node(env.removed[step - 1]),
                gscc_size=size, gscc_fraction=size / graph.num_nodes,
                cumulative_auc=area))
        if log_interval > 0 and len(env.removed) % log_interval < batch_size:
            print(f"{method} seed={seed} removed={len(env.removed)} GSCC={env.gscc_size}",
                  flush=True)
    result = dict(method=method, seed=seed if method == "random" else "",
                  num_states=graph.num_nodes, num_edges=len(graph.edges),
                  initial_gscc=initial_gscc, batch_fraction=batch_fraction,
                  adaptive_rounds=rounds, removals=len(env.removed),
                  auc=area, elapsed_seconds=time.perf_counter() - start)
    print(result, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/network-ho.csv")
    parser.add_argument("--method", choices=METHODS,
                        help="Run one baseline; omit to run all methods.")
    parser.add_argument("--batch-fraction", type=float, default=0.01)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--out-dir", type=Path,
                        default=HERE.parent / "DND-DRL" / "results" / "source_data")
    args = parser.parse_args()
    if not 0 < args.batch_fraction <= 1:
        parser.error("batch-fraction must be in (0, 1]")
    graph = DirectedGraph.read_csv_edge_table(args.graph)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    methods = (args.method,) if args.method else METHODS
    with (args.out_dir / "baseline_curves.csv").open("w", encoding="utf-8", newline="") as curve_file, \
         (args.out_dir / "baseline_summary.csv").open("w", encoding="utf-8", newline="") as summary_file:
        curves = csv.DictWriter(curve_file, fieldnames=CURVE_FIELDS)
        summary = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        curves.writeheader()
        summary.writeheader()
        for method in methods:
            seeds = range(args.random_seed, args.random_seed + args.random_repeats) \
                if method == "random" else (args.random_seed,)
            for seed in seeds:
                summary.writerow(dismantle(graph, method, seed, args.batch_fraction,
                                           curves, args.log_interval))
                curve_file.flush()
                summary_file.flush()
    print("saved", args.out_dir / "baseline_curves.csv",
          args.out_dir / "baseline_summary.csv")


if __name__ == "__main__":
    main()
