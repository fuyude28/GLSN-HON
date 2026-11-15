"""Calculate original TAD ranking in a separate sparse-numerics process."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tad_sparse import tad_order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    mapping: dict[str, int] = {}
    with args.graph.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        pairs = []
        for row in reader:
            u, v = row["FromNode"], row["ToNode"]
            if u not in mapping:
                mapping[u] = len(mapping)
            if v not in mapping:
                mapping[v] = len(mapping)
            pairs.append((mapping[u], mapping[v]))
    edges = tuple(sorted(set(pairs)))
    order, details = tad_order(len(mapping), edges)
    labels = list(mapping)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("rank", "state_node"))
        writer.writerows((index, labels[node]) for index, node in enumerate(order, 1))
    print(json.dumps({"num_nodes": len(mapping), "num_edges": len(edges), **details}))


if __name__ == "__main__":
    main()
