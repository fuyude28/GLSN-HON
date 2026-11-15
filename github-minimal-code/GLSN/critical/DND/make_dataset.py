"""Generate directed power-law graphs with exact minimum-set DND labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import igraph as ig
import torch

from features import graph_features
from labels import minimum_set_labels


HERE = Path(__file__).resolve().parent


def build_dataset(num_graphs: int, nodes: int, edges: list[int],
                  exponent_min: float, exponent_max: float,
                  target_fraction: float, seed: int,
                  max_combinations: int | None = None) -> list:
    rng = random.Random(seed)
    ig.set_random_number_generator(random.Random(seed))
    dataset = []
    for index in range(num_graphs):
        graph = ig.Graph.Static_Power_Law(
            n=nodes, m=edges[index % len(edges)],
            exponent_in=rng.uniform(exponent_min, exponent_max),
            exponent_out=rng.uniform(exponent_min, exponent_max),
            allowed_edge_types="simple")
        label = minimum_set_labels(graph, target_fraction, max_combinations)
        data = graph_features(graph)
        data.y = torch.tensor(label.probabilities, dtype=torch.float32)
        data.min_removals = label.min_removals
        data.optimal_set_count = label.optimal_set_count
        data.combinations_checked = label.combinations_checked
        dataset.append(data)
        print(f"{index + 1}/{num_graphs}: edges={graph.ecount()} "
              f"min_removals={label.min_removals} "
              f"optimal_sets={label.optimal_set_count} "
              f"checked={label.combinations_checked}", flush=True)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-graphs", type=int, default=150)
    parser.add_argument("--nodes", type=int, default=25)
    parser.add_argument("--edges", type=int, nargs="+", default=[100, 150, 200])
    parser.add_argument("--exponent-min", type=float, default=2.0)
    parser.add_argument("--exponent-max", type=float, default=2.9)
    parser.add_argument("--target-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-combinations", type=int,
                        help="Abort a graph rather than use incomplete labels")
    parser.add_argument("--out", type=Path, default=HERE / "results/train_data.pt")
    args = parser.parse_args()
    dataset = build_dataset(args.num_graphs, args.nodes, args.edges,
                            args.exponent_min, args.exponent_max,
                            args.target_fraction, args.seed,
                            args.max_combinations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.out)
    metadata = {"num_graphs": args.num_graphs, "nodes": args.nodes,
                "edges": args.edges, "exponent_min": args.exponent_min,
                "exponent_max": args.exponent_max,
                "target_fraction": args.target_fraction, "seed": args.seed,
                "labels": "frequency across all minimum-cardinality sets"}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2),
                                             encoding="utf-8")
    print(f"Saved {len(dataset)} labeled graphs to {args.out}")


if __name__ == "__main__":
    main()
