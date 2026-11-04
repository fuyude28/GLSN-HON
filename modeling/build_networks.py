"""Build first-order and higher-order directed networks from port-call sequences."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from pyhon import BuildNetwork
from pyhon import BuildRulesFastParameterFreeFreq as BuildRules

ROOT = Path(__file__).resolve().parents[1]


def read_trajectories(path: Path) -> list:
    """Read a headered CSV: vessel ID, then hyphen-separated port IDs."""
    trajectories = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle)
        next(rows)
        for row in rows:
            if len(row) < 2 or not row[1].strip():
                continue
            movements = [port for port, _ in
                         itertools.groupby(row[1].strip().split("-"))]
            if len(movements) >= 2:
                trajectories.append([row[0].strip(), movements])
    return trajectories


def sequence_to_node(sequence: tuple[str, ...]) -> str:
    return sequence[-1] + "|" + ".".join(reversed(sequence[:-1]))


def write_edges(path: Path, edges) -> None:
    edges = list(dict.fromkeys(edges))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FromNode", "ToNode"])
        writer.writerows(edges)
    nodes = {node for edge in edges for node in edge}
    print(f"{path.name}: {len(nodes)} nodes, {len(edges)} directed edges")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Headered trajectory CSV: vessel ID, port1-port2-...")
    parser.add_argument("--max-order", type=int, default=10)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/modeling")
    args = parser.parse_args()
    if args.max_order < 1 or args.min_support < 1:
        parser.error("max-order and min-support must be positive")
    trajectories = read_trajectories(args.input)
    if not trajectories:
        parser.error("input contains no trajectories with at least two distinct consecutive ports")
    args.out.mkdir(parents=True, exist_ok=True)
    write_edges(args.out / "network-fo.csv",
                ((source + "|", target + "|")
                 for _, movement in trajectories
                 for source, target in zip(movement, movement[1:])))
    BuildRules.Verbose = False
    BuildNetwork.Verbose = False
    rules = BuildRules.ExtractRules(trajectories, args.max_order, args.min_support)
    network = BuildNetwork.BuildNetwork(rules)
    write_edges(args.out / "network-ho.csv",
                ((sequence_to_node(source), sequence_to_node(target))
                 for source in network for target in network[source]))


if __name__ == "__main__":
    main()
