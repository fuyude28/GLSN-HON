"""Score a directed network with trained DND and evaluate static removal."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import igraph as ig
import torch

from dnd_model import DNDModel
from features import graph_features


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "DND-DRL"))
from dnd_drl.graph import DirectedGraph


def predict_scores(graph: DirectedGraph, checkpoint: Path) -> tuple[list[float], dict]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = DNDModel()
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    topology = ig.Graph(n=graph.num_nodes, edges=graph.edges, directed=True)
    data = graph_features(topology)
    with torch.no_grad():
        scores = model(data.x, data.edge_index).tolist()
    return scores, saved


def dismantle(graph: DirectedGraph, scores: list[float], target_fraction: float,
              batch_fraction: float, candidate_pool: str) -> tuple[list[tuple], dict]:
    """Rank once; after each removal measure LSCC on the residual graph."""
    n = graph.num_nodes
    target = max(1, math.floor(target_fraction * n))
    size, candidates = graph.gscc_state()
    removed: list[int] = []
    removed_set: set[int] = set()
    area = 0.0
    rounds = 0
    curve = [(0, 0, "", size, size / n, 0.0)]
    start = time.perf_counter()
    ranking = sorted(range(n), key=lambda node: (-scores[node], node))
    while size > target:
        allowed = (set(candidates) if candidate_pool == "gscc"
                   else set(range(n)) - removed_set)
        count = (1 if batch_fraction == 0 else
                 max(1, math.ceil(batch_fraction * (n - len(removed)))))
        selected = [node for node in ranking if node in allowed][:count]
        if not selected:
            raise RuntimeError("no nodes left while LSCC remains above target")
        rounds += 1
        for node in selected:
            removed.append(node)
            removed_set.add(node)
            size, candidates = graph.gscc_state(removed_set)
            area += size / (n * n)
            curve.append((rounds, len(removed), graph.original_node(node),
                          size, size / n, area))
            if size <= target:
                break
    summary = {"num_states": n, "num_edges": len(graph.edges),
               "initial_lscc": curve[0][3], "target_lscc": target,
               "target_fraction": target_fraction,
               "candidate_pool": candidate_pool,
               "batch_fraction": batch_fraction, "batch_rounds": rounds,
               "removals": len(removed), "partial_auc": area,
               "elapsed_seconds": time.perf_counter() - start}
    return curve, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path,
                        default=HERE.parents[3] / "data/network-ho.csv")
    parser.add_argument("--checkpoint", type=Path,
                        default=HERE / "results/dnd_model.pt")
    parser.add_argument("--scores-out", type=Path,
                        default=HERE / "results/dnd_scores.csv")
    parser.add_argument("--curve-out", type=Path,
                        default=HERE / "results/dnd_curve.csv")
    parser.add_argument("--summary-out", type=Path,
                        default=HERE / "results/dnd_summary.json")
    parser.add_argument("--target-fraction", type=float, default=0.1)
    parser.add_argument("--batch-fraction", type=float, default=0.0)
    parser.add_argument("--candidate-pool", choices=("all", "gscc"),
                        default="all")
    args = parser.parse_args()
    graph = DirectedGraph.read_csv_edge_table(args.graph)
    scores, checkpoint = predict_scores(graph, args.checkpoint)
    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("state_node", "predicted_probability"))
        writer.writerows((graph.original_node(node), score)
                         for node, score in enumerate(scores))
    curve, summary = dismantle(graph, scores, args.target_fraction,
                               args.batch_fraction, args.candidate_pool)
    summary.update(training_graphs=checkpoint["num_training_graphs"],
                   training_nodes=checkpoint["train_nodes"],
                   training_epochs=checkpoint["epochs"])
    args.curve_out.parent.mkdir(parents=True, exist_ok=True)
    with args.curve_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("batch_round", "removed", "removed_state",
                         "lscc_size", "lscc_fraction", "cumulative_auc"))
        writer.writerows(curve)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if checkpoint["train_nodes"] != [25]:
        print("Note: checkpoint is not trained on the paper's 25-node graphs.")


if __name__ == "__main__":
    main()
