"""Protect higher-order states against targeted attacks and random failures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import igraph as ig

from dnd_drl.graph import DirectedGraph

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("DND-DRL",)


def protection_order(graph: DirectedGraph, path: Path) -> dict[str, list[int]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        sequence = [graph.node_mapping[row["state_node"]] for row in csv.DictReader(handle)]
    return {"DND-DRL": sequence}


def largest_scc(graph: ig.Graph, failed: set[int]) -> tuple[int, list[int]]:
    """Active induced GSCC; the input graph keeps stable original vertex IDs."""
    active = [vertex for vertex in range(graph.vcount()) if vertex not in failed]
    if not active:
        return 0, []
    residual = graph.induced_subgraph(active)
    components = residual.connected_components(mode="strong")
    sizes = components.sizes()
    largest = max(sizes)
    if largest <= 1:
        return 1, []
    candidates = [active[i] for i, component in enumerate(components.membership)
                  if sizes[component] == largest]
    return largest, candidates


def pick_protected(order: list[int], attacked: set[int],
                   protected: set[int], count: int) -> list[int]:
    return [node for node in order if node not in attacked and node not in protected][:count]


def random_attack(ranking: list[int], active: set[int], protected: set[int],
                  effective: set[int], count: int) -> list[int]:
    return [node for node in ranking if node in active and node in effective
            and node not in protected][:count]


def targeted_attack(ranking: list[int], active: set[int], protected: set[int],
                    effective: set[int], count: int) -> list[int]:
    return random_attack(ranking, active, protected, effective, count)


def run_trial(graph: DirectedGraph, topology: ig.Graph, order: list[int],
              mode: str, steps: int, batch_size: int,
              failure_order: list[int] | None = None,
              no_protection: bool = False) -> list[dict]:
    """Apply one protection batch before each attack batch (fixed initial 1%)."""
    attacked: set[int] = set()
    protected: set[int] = set()
    initial, _ = largest_scc(topology, attacked)
    records = [dict(round=0, attacked=0, protected=0, gscc=initial,
                    gscc_fraction=1.0, attack_fraction=0.0)]
    for step in range(1, steps + 1):
        size, _ = largest_scc(topology, attacked)
        if size <= 1:
            break
        selection = ([] if no_protection else
                     pick_protected(order, attacked, protected, batch_size))
        if not no_protection and len(selection) < batch_size:
            raise ValueError("protection ordering is exhausted before requested rounds")
        protected.update(selection)

        active = set(range(graph.num_nodes)) - attacked
        # Effective nodes have at least one currently surviving edge.
        effective = set()
        for source, target in graph.edges:
            if source in active and target in active:
                effective.add(source)
                effective.add(target)
        if mode == "targeted":
            failures = targeted_attack(failure_order, active, protected,
                                       effective, batch_size)
        else:
            failures = random_attack(failure_order, active, protected,
                                     effective, batch_size)
        if len(failures) < batch_size:
            raise ValueError(f"{mode} attack order cannot supply {batch_size} "
                             f"effective targets at round {step}")
        attacked.update(failures)
        size, _ = largest_scc(topology, attacked)
        records.append(dict(round=step, attacked=len(attacked),
                            protected=len(protected), gscc=size,
                            gscc_fraction=size / initial,
                            attack_fraction=len(attacked) / graph.num_nodes))
    return records


def write_curves(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/network-ho.csv")
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/dismantling/protection")
    parser.add_argument("--order", type=Path, default=ROOT / "outputs/dismantling/removal_sequence.csv")
    args = parser.parse_args()
    if args.rounds < 1 or args.random_repeats < 1:
        parser.error("rounds and random-repeats must be positive")
    graph = DirectedGraph.read_csv_edge_table(args.graph)
    topology = ig.Graph(n=graph.num_nodes, edges=graph.edges, directed=True)
    batch = math.ceil(0.01 * graph.num_nodes)
    orders = protection_order(graph, args.order)
    # Random protection draws from the initial GSCC before other states.
    _, initial_core = largest_scc(topology, set())
    core = set(initial_core)
    methods = (*METHODS, "Random protection", "No protection")
    all_rows = []
    for mode in ("targeted", "random"):
        for repeat in range(args.random_repeats):
            seed = args.seed + repeat
            rng = random.Random(seed)
            failure_order = (orders["DND-DRL"] if mode == "targeted" else
                             rng.sample(range(graph.num_nodes), graph.num_nodes))
            for method in methods:
                if mode == "targeted" and repeat > 0 and method != "Random protection":
                    continue
                order = (orders[method] if method in orders else
                         random.Random(args.seed + 10000 + repeat).sample(
                             initial_core, len(initial_core)) +
                         [node for node in range(graph.num_nodes) if node not in core])
                rows = run_trial(graph, topology, order, mode,
                                 args.rounds, batch, failure_order,
                                 no_protection=method == "No protection")
                for row in rows:
                    all_rows.append(dict(mode=mode, method=method, repeat=repeat,
                                         seed=seed, **row))
                print(f"{mode} seed={seed} {method}: "
                      f"rounds={len(rows) - 1} GSCC={rows[-1]['gscc']} "
                      f"attacked={rows[-1]['attacked']}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_curves(args.out_dir / "protection_curves.csv", all_rows)
    metadata = dict(graph=str(args.graph), order=str(args.order),
                    nodes=graph.num_nodes, edges=len(graph.edges),
                    initial_gscc=graph.gscc_size(), protected_and_attacked_per_round=batch,
                    rounds=args.rounds, random_repeats=args.random_repeats,
                    random_seeds=list(range(args.seed, args.seed + args.random_repeats)),
                    protection_methods=list(methods),
                    targeted_attacker="saved DND-DRL dismantling sequence; skip protected, removed and isolated states",
                    random_failure_pool="unprotected active nodes with incident active edges",
                    coordinate="GSCC/initial_GSCC vs attacked/N")
    (args.out_dir / "protection_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
