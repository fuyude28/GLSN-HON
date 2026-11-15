"""Evaluate static control-centrality ranking on GSCC dismantling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "DND-DRL"))
from dnd_drl.env import DirectedDismantlingEnv
from dnd_drl.graph import DirectedGraph


def evaluate(graph: DirectedGraph, scores: dict[str, int],
             batch_fraction: float, curve_path: Path,
             summary_path: Path) -> dict[str, int | float]:
    env = DirectedDismantlingEnv(graph)
    area = 0.0
    rounds = 0
    start = time.perf_counter()
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("batch_round", "removed", "removed_state", "gscc_size",
                         "gscc_fraction", "cumulative_auc"))
        writer.writerow((0, 0, "", env.gscc_size,
                         env.gscc_size / graph.num_nodes, 0))
        while not env.is_terminal():
            candidates = env.candidate_actions()
            batch = min(len(candidates), max(1, math.ceil(
                batch_fraction * (graph.num_nodes - len(env.removed)))))
            selected = sorted(candidates,
                              key=lambda node: (-scores[str(graph.original_node(node))], node))[:batch]
            rounds += 1
            for step, size in env.remove_batch(selected):
                area += size / (graph.num_nodes ** 2)
                writer.writerow((rounds, step, graph.original_node(env.removed[step - 1]),
                                 size, size / graph.num_nodes, area))
    result = dict(num_states=graph.num_nodes, num_edges=len(graph.edges),
                  initial_gscc=graph.gscc_size(), batch_fraction=batch_fraction,
                  adaptive_rounds=rounds, removals=len(env.removed),
                  auc=area, elapsed_seconds=time.perf_counter() - start)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path,
                        default=HERE.parents[3] / "data/network-ho.csv")
    parser.add_argument("--scores", type=Path,
                        default=HERE / "results/control_centrality_scores.csv")
    parser.add_argument("--batch-fraction", type=float, default=0.01)
    parser.add_argument("--curve-out", type=Path,
                        default=HERE / "results/control_centrality_curve.csv")
    parser.add_argument("--summary-out", type=Path,
                        default=HERE / "results/control_centrality_summary.json")
    args = parser.parse_args()
    graph = DirectedGraph.read_csv_edge_table(args.graph)
    with args.scores.open(encoding="utf-8", newline="") as handle:
        scores = {row["state_node"]: int(row["control_centrality"])
                  for row in csv.DictReader(handle)}
    if len(scores) != graph.num_nodes:
        raise ValueError("control-centrality scores must cover every state node")
    result = evaluate(graph, scores, args.batch_fraction,
                      args.curve_out, args.summary_out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
