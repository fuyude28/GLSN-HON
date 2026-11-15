"""Apply CoreHD and TAD rankings to the higher-order network's GSCC."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE / "CoreHD"), str(HERE / "TAD"),
                str(HERE.parent / "DND-DRL")]

from corehd_ho import corehd_order
from dnd_drl.env import DirectedDismantlingEnv
from dnd_drl.graph import DirectedGraph


def evaluate(graph: DirectedGraph, order: list[int], curve_path: Path,
             batch_fraction: float = 0.01) -> dict:
    env = DirectedDismantlingEnv(graph)
    rank = {node: position for position, node in enumerate(order)}
    if len(rank) != graph.num_nodes:
        raise ValueError("ranking must contain every state node exactly once")
    start = time.perf_counter()
    area = 0.0
    rounds = 0
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("batch_round", "removed", "removed_state", "gscc_size",
                         "gscc_fraction", "cumulative_auc"))
        writer.writerow((0, 0, "", env.gscc_size,
                         env.gscc_size / graph.num_nodes, 0))
        while not env.is_terminal():
            candidates = env.candidate_actions()
            count = min(len(candidates), max(1, math.ceil(
                batch_fraction * (graph.num_nodes - len(env.removed)))))
            selected = sorted(candidates, key=rank.__getitem__)[:count]
            rounds += 1
            for step, size in env.remove_batch(selected):
                area += size / graph.num_nodes ** 2
                writer.writerow((rounds, step, graph.original_node(env.removed[step - 1]),
                                 size, size / graph.num_nodes, area))
            if rounds % 10 == 0:
                print(f"round={rounds}, removed={len(env.removed)}, GSCC={env.gscc_size}",
                      flush=True)
    return {"num_states": graph.num_nodes, "num_edges": len(graph.edges),
            "initial_gscc": graph.gscc_size(), "batch_fraction": batch_fraction,
            "adaptive_rounds": rounds, "removals": len(env.removed),
            "auc": area, "evaluation_seconds": time.perf_counter() - start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("corehd", "tad"), required=True)
    parser.add_argument("--graph", type=Path,
                        default=ROOT / "data/network-ho.csv")
    parser.add_argument("--batch-fraction", type=float, default=0.01)
    args = parser.parse_args()
    graph = DirectedGraph.read_csv_edge_table(args.graph)
    method_dir = HERE / ("CoreHD" if args.method == "corehd" else "TAD")
    results = method_dir / "results"
    results.mkdir(exist_ok=True)
    start = time.perf_counter()
    order_path = results / f"{args.method}_order.csv"
    if args.method == "corehd":
        order = corehd_order(graph.num_nodes, graph.edges)
        details = {"projection": "unweighted undirected, single-node 2-core ranking; "
                 "tree breaking to isolated vertices"}
        with order_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("rank", "state_node"))
            writer.writerows((index, graph.original_node(node))
                             for index, node in enumerate(order, 1))
    else:
        process = subprocess.run(
            [sys.executable, str(HERE / "TAD" / "run_tad_rank.py"),
             "--graph", str(args.graph), "--out", str(order_path)],
            check=True, capture_output=True, text=True)
        details = json.loads(process.stdout)
        with order_path.open(encoding="utf-8", newline="") as handle:
            order = [graph.node_mapping[row["state_node"]]
                     for row in csv.DictReader(handle)]
    ordering_seconds = time.perf_counter() - start
    print(f"{args.method} ordered {len(order)} nodes in {ordering_seconds:.1f}s",
          flush=True)
    summary = evaluate(graph, order, results / f"{args.method}_curve.csv",
                       args.batch_fraction)
    summary.update(method=args.method, ordering_seconds=ordering_seconds,
                   ranking="static ranking restricted to current largest SCC",
                   **details)
    (results / f"{args.method}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
